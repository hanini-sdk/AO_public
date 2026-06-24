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
from collections import deque
from difflib import SequenceMatcher
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
_CHAT_MAX_TOKENS    = 1_500  # budget for the final RAG answer
_EXTRACT_MAX_TOKENS = 60     # budget for the entity-extraction pre-call

# BFS + fuzzy retrieval parameters
_FUZZY_CUTOFF  = 0.75   # minimum similarity ratio for fuzzy name match (0–1)
_BFS_MAX_DEPTH = 4      # maximum hops from the seed nodes in the lineage graph

# Keywords that signal the user wants upstream (sources) vs downstream (consumers)
_UPSTREAM_KEYWORDS = {
    "vient", "source", "provient", "alimenté", "calculé", "dépend", "origine",
    "comes", "fed", "derived", "computed", "origine", "calcul", "input",
}
_DOWNSTREAM_KEYWORDS = {
    "utilisé", "impacte", "consommé", "envoyé", "cible", "impact", "alimente",
    "used", "impacts", "flows", "consumed", "feeds", "output", "downstream",
}


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


def _detect_direction(question: str) -> str:
    """Infer lineage direction from the question vocabulary.

    Returns "upstream", "downstream", or "both" (default).
    """
    q = _tokenize(question)
    up   = bool(q & _UPSTREAM_KEYWORDS)
    down = bool(q & _DOWNSTREAM_KEYWORDS)
    if up and not down:  return "upstream"
    if down and not up:  return "downstream"
    return "both"


def _seg_best(token: str, value: str) -> float:
    """Best SequenceMatcher similarity between a token and any segment of value.

    value is split on separators (. _ - : /) so that "db.schema.col_revenu"
    is matchable by the token "revenu" alone (score ~1.0 on that segment).
    """
    segs = [value.lower()] + [
        p for p in re.split(r"[._\-:/]", value.lower()) if len(p) > 2
    ]
    return max(SequenceMatcher(None, token, seg).ratio() for seg in segs)


def _edge_match_score(edge: dict, q_tokens: set[str]) -> float:
    """Score an edge by fuzzy matching tokens against its source and target.

    For each token:
      score_src = best similarity(token, source segments)
      score_tgt = best similarity(token, target segments)
      contribution = max(score_src, score_tgt)   ← the edge is relevant if
                                                    EITHER endpoint matches
    Return the sum of contributions across all tokens.
    """
    src = edge.get("source", "")
    tgt = edge.get("target", "")
    if not src and not tgt:
        return 0.0

    total = 0.0
    for token in q_tokens:
        score_src = _seg_best(token, src) if src else 0.0
        score_tgt = _seg_best(token, tgt) if tgt else 0.0
        total += max(score_src, score_tgt)
    return total


def _bfs_edges(
    seed_values: set[str],
    edges: list[dict],
    max_depth: int = _BFS_MAX_DEPTH,
    direction: str = "both",
) -> list[dict]:
    """BFS traversal from seed endpoint values along the edge graph.

    Returns edges in BFS discovery order (closest to the seed first), so the
    most directly relevant lineage appears at the top of the result list.

    ``direction`` controls which edges are followed:
      - "downstream" — follow source → target  (what does X feed into?)
      - "upstream"   — follow target → source  (where does X come from?)
      - "both"       — follow both directions
    """
    # Adjacency indexes keyed on raw source/target values
    out_adj: dict[str, list[dict]] = {}   # source_value  → outgoing edges
    in_adj:  dict[str, list[dict]] = {}   # target_value  → incoming edges
    for e in edges:
        src, tgt = e.get("source", ""), e.get("target", "")
        if src: out_adj.setdefault(src, []).append(e)
        if tgt: in_adj.setdefault(tgt, []).append(e)

    visited:    set[str]     = set(seed_values)
    seen_edges: set[tuple]   = set()          # (source, target, type) dedup key
    result:     list[dict]   = []
    frontier:   deque        = deque((v, 0) for v in seed_values)

    def _emit(e: dict) -> None:
        key = (e.get("source"), e.get("target"), e.get("type"))
        if key not in seen_edges:
            seen_edges.add(key)
            result.append(e)

    def _neighbors(value: str) -> list[tuple[str, dict]]:
        found: list[tuple[str, dict]] = []
        if direction in ("downstream", "both"):
            for e in out_adj.get(value, []):
                found.append((e.get("target", ""), e))
        if direction in ("upstream", "both"):
            for e in in_adj.get(value, []):
                found.append((e.get("source", ""), e))
        return found

    # Depth-0: all edges that directly touch any seed (both directions)
    for seed in seed_values:
        for e in out_adj.get(seed, []) + in_adj.get(seed, []):
            _emit(e)

    # BFS expansion
    while frontier:
        value, depth = frontier.popleft()
        if depth >= max_depth:
            continue
        for neighbor, edge in _neighbors(value):
            if neighbor and neighbor not in visited:
                visited.add(neighbor)
                _emit(edge)
                frontier.append((neighbor, depth + 1))

    return result


def retrieve_context(
    question: str, graph: dict, top_k: int = _TOP_K
) -> tuple[list[dict], list[dict]]:
    """Return ([], context_edges) relevant to the question.

    Pure edge-based lineage retrieval — no nodes, no embeddings:
      1. Score every edge with _edge_match_score: for each query token compute
         fuzzy similarity against source segments AND target segments, take the
         max of the two, sum over all tokens.
      2. Edges whose score >= _FUZZY_CUTOFF become BFS seeds.
      3. BFS expands from their endpoints along the full edge graph up to
         _BFS_MAX_DEPTH hops, following the lineage direction inferred from
         the question vocabulary.
      4. Fallback: if no seed reached cutoff, return all edges sorted by score
         (best-effort for generic questions).
    """
    edges: list[dict] = graph.get("edges") or graph.get("arretes") or []

    if not edges:
        return [], []

    q_tokens = _tokenize(question)

    # ── 1. Score every edge ───────────────────────────────────────────────────
    scored = sorted(
        ((e, _edge_match_score(e, q_tokens)) for e in edges),
        key=lambda x: -x[1],
    )

    # ── 2. Seed edges: score >= cutoff ────────────────────────────────────────
    seed_edges = [e for e, s in scored if s >= _FUZZY_CUTOFF]
    seed_values: set[str] = set()
    for e in seed_edges:
        if e.get("source"): seed_values.add(e["source"])
        if e.get("target"): seed_values.add(e["target"])
    log.debug("seed edges: %d, seed values: %s", len(seed_edges), seed_values)

    # ── 3. BFS from seed endpoints ────────────────────────────────────────────
    direction = _detect_direction(question)
    if seed_values:
        context_edges = _bfs_edges(seed_values, edges, direction=direction)[:top_k]
    else:
        # Fallback: no entity matched — return top-scored edges as-is
        context_edges = [e for e, s in scored if s > 0][:top_k]

    return [], context_edges


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
# Entity extraction (LLM pre-call)
# ---------------------------------------------------------------------------

def _extract_entity(question: str, llm: LLMClient) -> str | None:
    """Extract the variable/column/table name from the question via a cheap LLM call.

    Returns the raw name string so the fuzzy matcher works on the exact entity
    and is not polluted by the surrounding question text (e.g. "donne moi les
    colonnes liées à" would otherwise produce false positive token matches).
    Returns None on failure or when no specific name is found; the caller then
    falls back to passing the full question to retrieve_context.
    """
    prompt = (
        "Extract the exact variable, column, or table name mentioned in the question. "
        "Return ONLY the name, no explanation, no punctuation. "
        "If no specific name is mentioned, return null.\n\n"
        f"Question: {question}\n\nName:"
    )
    try:
        raw = llm.chat(
            [{"role": "user", "content": prompt}],
            max_tokens=_EXTRACT_MAX_TOKENS,
        ).strip().strip('"\'')
        if not raw or raw.lower() in ("null", "none", "aucun", "n/a"):
            return None
        log.debug("Entity extracted: %r", raw)
        return raw
    except Exception as exc:
        log.debug("Entity extraction failed (%s), using full question", exc)
        return None


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

    entity = _extract_entity(question, llm)
    context_nodes, context_edges = retrieve_context(entity or question, graph)
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