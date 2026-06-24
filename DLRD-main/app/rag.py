"""RAG (Retrieval-Augmented Generation) for the chat endpoint.

Retrieves the most relevant nodes and edges from the knowledge graph for a
given question using purely textual matching — no embeddings, no ML libraries,
no external calls. The LLM call goes through the same guarded client
(app/llm.py + app/http_guard.py) as all other enrichment calls.

Retrieval pipeline:
  1. Keyword extraction from the question (stopword-filtered tokens).
  2. Lexical scoring of every node  (name / path / tags / summary).
  3. Lexical scoring of every edge  (type / description / source+target names).
  4. Structural name-match boost: nodes whose name appears verbatim in the
     question are promoted to the front regardless of lexical score.
  5. Lineage expansion: edges connecting two retained nodes are always
     included first; directly scored edges are added up to the cap.
  6. Final selection: top _TOP_K nodes + top _TOP_K edges by score.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from .config import DATA_DIR, Settings
from .enrich import sanitize_for_enrichment  # the SAME redactor the story path uses
from .llm import LLMClient

log = logging.getLogger("data_lineage_retro_documentation.rag")

_GRAPH_PATH = DATA_DIR / "knowledge-graph.json"
_CONTEXT_PATH = DATA_DIR / "project_context.json"  # sidecar document context (optional)
_MAX_DOC_CONTEXT_CHARS = 2_500  # cap the document block added to the prompt

# English + French stop words excluded from token matching.
_STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "to", "of", "in", "for", "on", "with",
    "as", "by", "from", "at", "into", "through", "how", "what", "which",
    "who", "where", "when", "why", "that", "this", "it", "its", "and",
    "or", "not", "no", "but", "so", "all", "any", "some", "such", "used",
    # French
    "le", "la", "les", "un", "une", "des", "du", "de", "en", "et", "ou",
    "est", "sont", "que", "qui", "dans", "sur", "avec", "par", "pour",
    "comment", "quel", "quelle", "il", "elle", "ils", "elles", "ce",
    "ces", "cet", "cette", "je", "tu", "nous", "vous",
}

_TOP_K = 150               # nodes and edges injected into the prompt
_MAX_CONTEXT_CHARS = 8_000  # hard cap on context block size (enforced in build_rag_prompt)
_MAX_HISTORY_TURNS = 6     # conversation turns kept in the prompt
_CHAT_MAX_TOKENS = 1_500   # generous budget for RAG answers


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> set[str]:
    """Extract meaningful lowercase tokens (letters + underscores, len > 1)."""
    tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", text.lower())
    return {t for t in tokens if len(t) > 1 and t not in _STOP_WORDS}


def _score_node(node: dict, q_tokens: set[str]) -> float:
    """Score a node against question tokens using field-weighted matching.

    Weights: name exact-token (3.0) > name substring (1.5) > path segment (2.0)
             > tag token (1.5) > summary token (1.0).
    """
    if not q_tokens:
        return 0.0

    score = 0.0
    name = node.get("name", "")
    file_path = node.get("filePath", "") or ""
    summary = node.get("summary", "")
    tags = node.get("tags", [])

    name_tokens = _tokenize(name)
    path_tokens = _tokenize(re.sub(r"[/._]", " ", file_path))
    summary_tokens = _tokenize(summary)
    tag_tokens = _tokenize(" ".join(str(t) for t in tags))

    for token in q_tokens:
        if token in name_tokens:
            score += 3.0    # exact token match in name  — highest signal
        elif token in name.lower():
            score += 1.5    # substring match in name
        if token in path_tokens:
            score += 2.0    # path segment (directory / filename component)
        if token in summary_tokens:
            score += 1.0    # summary prose
        if token in tag_tokens:
            score += 1.5    # explicit tag

    return score


def _score_edge(edge: dict, q_tokens: set[str], node_by_id: dict[str, dict]) -> float:
    """Score an edge against question tokens.

    Matched fields and weights:
      - source node name (2.0) and target node name (2.0)  — lineage signal
      - edge type / relation  (1.5)                        — structural keyword
      - edge description / label  (1.0)                    — prose annotation
    """
    if not q_tokens:
        return 0.0

    score = 0.0
    src_node = node_by_id.get(edge.get("source", ""), {})
    tgt_node = node_by_id.get(edge.get("target", ""), {})
    src_tokens = _tokenize(src_node.get("name", ""))
    tgt_tokens = _tokenize(tgt_node.get("name", ""))
    type_tokens = _tokenize(edge.get("type", ""))
    # "description" carries prose annotations like "purge -> write -> read"
    # or "feeds via table1, table2"; "label" is an alternative field name.
    desc = edge.get("description", "") or edge.get("label", "") or ""
    desc_tokens = _tokenize(desc)

    for token in q_tokens:
        if token in src_tokens:
            score += 2.0    # query keyword names the source node
        if token in tgt_tokens:
            score += 2.0    # query keyword names the target node
        if token in type_tokens:
            score += 1.5    # relation type keyword (imports, calls, reads_from…)
        if token in desc_tokens:
            score += 1.0    # prose annotation on the edge

    return score


def _diverse_sample(nodes: list[dict], k: int) -> list[dict]:
    """Return up to k nodes with type diversity (fallback for zero-score)."""
    by_type: dict[str, list[dict]] = {}
    for n in nodes:
        by_type.setdefault(n.get("type", "file"), []).append(n)
    result: list[dict] = []
    type_keys = list(by_type)
    i = 0
    while len(result) < k and i < 200:
        t = type_keys[i % len(type_keys)]
        bucket = by_type[t]
        idx = i // len(type_keys)
        if idx < len(bucket):
            result.append(bucket[idx])
        i += 1
    return result[:k]


def retrieve_context(
    question: str, graph: dict, top_k: int = _TOP_K
) -> tuple[list[dict], list[dict]]:
    """Return (context_nodes, context_edges) relevant to the question.

    Pure text-based retrieval — no embeddings, no ML libraries:
      1. Keyword extraction  — lowercase tokens, stopwords removed.
      2. Node scoring        — weighted match across name / path / tags / summary.
      3. Node boost          — verbatim name matches forced to the front.
      4. Top-k nodes         — top_k best-scored nodes (diverse fallback when all
                               scores are zero).
      5. Edge scoring        — weighted match across type / description /
                               source-name / target-name.
      6. Edge selection      — edges connecting two retained nodes (lineage
                               expansion) come first; directly scored edges
                               (score > 0) fill the rest up to top_k.

    Both ``nodes`` (English) and ``noeuds`` (French) field names are accepted,
    and likewise ``edges`` / ``arretes``, so the function works with both the
    standard schema and any French-keyed variant of the graph.
    """
    # Accept both English and French field names
    nodes: list[dict] = graph.get("nodes") or graph.get("noeuds") or []
    edges: list[dict] = graph.get("edges") or graph.get("arretes") or []

    if not nodes:
        return [], []

    q_tokens = _tokenize(question)
    node_by_id = {n["id"]: n for n in nodes}

    # ── 1. Score and rank nodes ──────────────────────────────────────────────
    ranked = sorted(nodes, key=lambda n: -_score_node(n, q_tokens))

    # ── 2. Structural name-match boost ───────────────────────────────────────
    # A node whose name appears verbatim in the question (≥ 3 chars) is
    # almost certainly what the user is asking about — promote it unconditionally.
    q_lower     = question.lower()
    boosted     = [n for n in ranked
                   if len(n.get("name", "")) >= 3 and n["name"].lower() in q_lower]
    boosted_ids = {n["id"] for n in boosted}
    unboosted   = [n for n in ranked if n["id"] not in boosted_ids]
    ranked      = boosted + unboosted

    # ── 3. Select top nodes ──────────────────────────────────────────────────
    if all(_score_node(n, q_tokens) == 0.0 for n in ranked[:top_k]):
        # No keyword matched anything — return a structurally diverse sample
        # so the chat always has something meaningful to work with.
        top_nodes = _diverse_sample(nodes, top_k)
    else:
        top_nodes = ranked[:top_k]

    top_node_ids = {n["id"] for n in top_nodes}

    # ── 4. Score edges ───────────────────────────────────────────────────────
    edge_scores = [_score_edge(e, q_tokens, node_by_id) for e in edges]
    sorted_edge_idx = sorted(range(len(edges)), key=lambda i: -edge_scores[i])

    # ── 5. Lineage expansion + edge selection ────────────────────────────────
    # Priority 1 — connecting edges: both endpoints are in the retained node
    # set. These capture the data-flow lineage between the selected nodes and
    # are included regardless of their own text score.
    connecting_idx = [
        i for i in sorted_edge_idx
        if edges[i].get("source") in top_node_ids
        and edges[i].get("target") in top_node_ids
    ]
    connecting_set = set(connecting_idx)

    # Priority 2 — directly scored edges: at least one keyword matched the
    # edge's own fields (type / description / endpoint names). Included even
    # when one or both endpoint nodes did not make the top-k node list, so
    # cross-boundary lineage is still surfaced.
    scored_extra_idx = [
        i for i in sorted_edge_idx
        if i not in connecting_set and edge_scores[i] > 0
    ]

    all_edge_idx  = (connecting_idx + scored_extra_idx)[:top_k]
    context_edges = [edges[i] for i in all_edge_idx]

    return top_nodes, context_edges


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def _fmt_node(node: dict) -> str:
    nid = node.get("id", "?")
    ntype = node.get("type", "?")
    name = node.get("name", "?")
    file_path = node.get("filePath", "")
    summary = node.get("summary", "")
    tags = node.get("tags", [])
    attributes = node.get("attributes", {})

    lines = [f"[[{nid}]] {ntype.upper()} «{name}»"]
    if file_path:
        lines.append(f"  path: {file_path}")
    if summary:
        lines.append(f"  summary: {summary}")
    if tags:
        lines.append(f"  tags: {', '.join(str(t) for t in tags[:5])}")
    # Business labels from the variable dictionary, when present — surfaced so
    # answers can speak in business terms. Same guarded egress + redactor.
    if isinstance(attributes, dict) and attributes:
        attr_str = "; ".join(f"{k}: {v}" for k, v in list(attributes.items())[:8])
        lines.append(f"  attributes: {attr_str}")
    return "\n".join(lines)


def _fmt_edge(edge: dict, node_by_id: dict[str, dict]) -> str:
    src_name = node_by_id.get(edge.get("source", ""), {}).get("name", edge.get("source", "?"))
    tgt_name = node_by_id.get(edge.get("target", ""), {}).get("name", edge.get("target", "?"))
    etype = edge.get("type", "?")
    return f"  {src_name} --[{etype}]--> {tgt_name}"


def _load_doc_context() -> dict | None:
    """Read the sidecar document context store if present (best-effort)."""
    if not _CONTEXT_PATH.exists():
        return None
    try:
        return json.loads(_CONTEXT_PATH.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - optional artifact; never break the chat
        return None


def _doc_context_block(doc_context: dict | None, context_nodes: list[dict]) -> str:
    """A compact, redacted document block: the global narrative plus the doc
    snippets attached (by exact-name match) to the nodes currently in context."""
    if not doc_context:
        return ""
    parts: list[str] = []
    narrative = str(doc_context.get("globalNarrative") or "").strip()
    if narrative:
        parts.append("Overall (from project documents): " + narrative)
    node_ctx = doc_context.get("nodes") or {}
    seen_snippets: set[str] = set()
    for node in context_nodes:
        for snip in node_ctx.get(node.get("id", ""), [])[:2]:
            text = str(snip.get("snippet") or "").strip()
            key = text[:60]
            if text and key not in seen_snippets:
                seen_snippets.add(key)
                parts.append(f"- {node.get('name', '?')}: {text}")
    if not parts:
        return ""
    block = sanitize_for_enrichment("\n".join(parts))
    return block[:_MAX_DOC_CONTEXT_CHARS]


def build_rag_prompt(
    question: str,
    context_nodes: list[dict],
    context_edges: list[dict],
    history: list[dict],
    graph: dict,
    settings: Settings,
    doc_context: dict | None = None,
) -> list[dict]:
    """Build the message list for the RAG chat completion."""
    project = graph.get("project", {})
    project_name = project.get("name", "the project")
    project_desc = project.get("description", "")
    languages = project.get("languages", [])

    node_by_id = {n["id"]: n for n in context_nodes}

    # Build context block (respecting the character budget)
    blocks: list[str] = []
    total = 0
    for node in context_nodes:
        block = _fmt_node(node)
        if total + len(block) > _MAX_CONTEXT_CHARS:
            break
        blocks.append(block)
        total += len(block)

    edge_lines = [_fmt_edge(e, node_by_id) for e in context_edges[:60]]
    context_text = "\n\n".join(blocks)
    if edge_lines:
        context_text += "\n\nRelationships:\n" + "\n".join(edge_lines)

    lang_note = f"Languages: {', '.join(languages)}." if languages else ""
    desc_note = f"Description: {project_desc}" if project_desc else ""

    system_content = (
        f'You are an expert software analyst helping a developer understand "{project_name}".\n'
        + (f"{desc_note}\n" if desc_note else "")
        + (f"{lang_note}\n" if lang_note else "")
        + "\n"
        "You have access to the knowledge graph extract below. Each entry is formatted as:\n"
        "[[node-id]] TYPE «name»\n"
        "  path: file/path\n"
        "  summary: what this node does\n\n"
        "Answer the user's question based ONLY on the provided context. Be specific:\n"
        "- Cite node names and file paths explicitly.\n"
        "- When referencing a node, include its ID in double brackets exactly as it appears,\n"
        "  e.g. [[file:app/llm.py]] — this lets the user navigate to it in the graph.\n"
        "- If you cannot answer from the context, say so clearly.\n\n"
        "KNOWLEDGE GRAPH CONTEXT:\n"
        + context_text
    )

    # Append document-sourced context (sidecar store) when available. It is
    # already redacted+summarized; we re-redact in-path as defense-in-depth and
    # mark it approximate. No new egress — same guarded client as the rest.
    doc_block = _doc_context_block(doc_context, context_nodes)
    if doc_block:
        system_content += (
            "\n\nPROJECT DOCUMENTS (summarized from prose docs; approximate, "
            "secondary to the graph facts above):\n" + doc_block
        )

    # Redact the fully-assembled context before it reaches the internal service —
    # "redact before every LLM call", matching the story path (story.py). This
    # covers every project-derived input composed above: node fields, dictionary-
    # derived attributes, edge/relationship names, and the project name +
    # description. The document block is already redacted; re-running the redactor
    # is idempotent (it only replaces credential tokens, leaving prior
    # <redacted> markers and all other text unchanged).
    messages: list[dict] = [
        {"role": "system", "content": sanitize_for_enrichment(system_content)}
    ]

    # Append recent conversation history (up to _MAX_HISTORY_TURNS pairs)
    for turn in history[-(2 * _MAX_HISTORY_TURNS) :]:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": question})
    return messages


# ---------------------------------------------------------------------------
# Citation extraction
# ---------------------------------------------------------------------------

def _extract_cited_ids(answer: str, valid_ids: set[str]) -> list[str]:
    """Parse [[node-id]] citations from the LLM answer, validated against graph."""
    raw = re.findall(r"\[\[([^\[\]\n]+)\]\]", answer)
    seen: set[str] = set()
    result: list[str] = []
    for nid in raw:
        nid = nid.strip()
        if nid in valid_ids and nid not in seen:
            result.append(nid)
            seen.add(nid)
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def answer_question(
    question: str,
    history: list[dict],
    llm: LLMClient,
    settings: Settings,
) -> dict:
    """Full RAG pipeline: retrieve → prompt → LLM call → extract sources.

    The only outbound network call goes through the existing guarded LLM client.
    Returns a dict with keys: answer (str), source_ids (list[str]), sources (list[dict]).
    """
    if not _GRAPH_PATH.exists():
        return {
            "answer": "No knowledge graph found. Please run an analysis first.",
            "source_ids": [],
            "sources": [],
        }

    try:
        graph = json.loads(_GRAPH_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        log.error("Failed to read knowledge graph for RAG: %s", exc)
        return {
            "answer": "Failed to read the knowledge graph.",
            "source_ids": [],
            "sources": [],
        }

    context_nodes, context_edges = retrieve_context(question, graph)
    messages = build_rag_prompt(
        question=question,
        context_nodes=context_nodes,
        context_edges=context_edges,
        history=history,
        graph=graph,
        settings=settings,
        doc_context=_load_doc_context(),
    )

    answer = llm.chat(messages, max_tokens=_CHAT_MAX_TOKENS)

    all_nodes = graph.get("nodes") or graph.get("noeuds") or []
    node_by_id = {n["id"]: n for n in all_nodes}
    valid_ids = set(node_by_id)
    source_ids = _extract_cited_ids(answer, valid_ids)

    # Fall back to top context nodes if the LLM didn't emit any citations.
    if not source_ids:
        source_ids = [n["id"] for n in context_nodes[:5]]

    sources = []
    for nid in source_ids:
        node = node_by_id.get(nid)
        if node:
            sources.append({
                "id": nid,
                "name": node.get("name", ""),
                "type": node.get("type", ""),
                "filePath": node.get("filePath"),
                "summary": node.get("summary", ""),
            })

    return {"answer": answer, "source_ids": source_ids, "sources": sources}
