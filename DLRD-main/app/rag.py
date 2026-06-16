"""RAG (Retrieval-Augmented Generation) for the chat endpoint.

Retrieves the most relevant nodes and edges from the knowledge graph for a
given question using purely lexical + structural matching — no external
embeddings, no new network egress. The LLM call goes through the same
guarded client (app/llm.py + app/http_guard.py) as all other enrichment calls.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from .config import DATA_DIR, Settings
from .llm import LLMClient

log = logging.getLogger("data_lineage_retro_documentation.rag")

_GRAPH_PATH = DATA_DIR / "knowledge-graph.json"

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

_TOP_K = 12                # nodes injected into the prompt
_EXPAND_TOP = 5            # top scorers whose 1-hop neighbors are added
_MAX_CONTEXT_CHARS = 8_000  # hard cap on context block size
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
    """Score a node against the question tokens using field-weighted matching."""
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
            score += 3.0                    # exact token in name
        elif token in name.lower():
            score += 1.5                    # substring match in name
        if token in path_tokens:
            score += 2.0                    # path segment match
        if token in summary_tokens:
            score += 1.0                    # summary token match
        if token in tag_tokens:
            score += 1.5                    # tag match

    return score


_RRF_K = 60  # standard Reciprocal Rank Fusion constant


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


def _rrf_merge(
    lex_rank: dict[str, int],
    sem_rank: dict[str, int],
    n_total: int,
) -> dict[str, float]:
    """Reciprocal Rank Fusion over lexical and semantic rankings.

    rrf(d) = 1/(k + rank_lex(d)) + 1/(k + rank_sem(d))

    Nodes absent from sem_rank (index unavailable or not returned) are
    treated as ranked last (rank = n_total), giving them the floor RRF
    contribution 1/(k + n_total) from the semantic side.
    """
    all_ids = set(lex_rank) | set(sem_rank)
    scores: dict[str, float] = {}
    for nid in all_ids:
        r_lex = lex_rank.get(nid, n_total)
        r_sem = sem_rank.get(nid, n_total)
        scores[nid] = 1.0 / (_RRF_K + r_lex) + 1.0 / (_RRF_K + r_sem)
    return scores


def retrieve_context(
    question: str, graph: dict, top_k: int = _TOP_K
) -> tuple[list[dict], list[dict]]:
    """Return (context_nodes, context_edges) relevant to the question.

    Hybrid retrieval strategy:
      1. Lexical ranking — TF-IDF-like token scoring on name/path/summary/tags.
      2. Semantic ranking — cosine similarity via local embedding index
         (EmbeddingIndex); silently skipped when index unavailable → fallback
         to lexical-only.
      3. Fusion — Reciprocal Rank Fusion (RRF, k=60) merges the two ranked
         lists without requiring score normalisation.
      4. Structural boost — nodes whose name appears verbatim in the question
         are forced to the top regardless of their RRF score.
      5. 1-hop expansion — direct neighbours of the top-_EXPAND_TOP nodes are
         added to enrich context up to top_k total.
    """
    nodes: list[dict] = graph.get("nodes", [])
    edges: list[dict] = graph.get("edges", [])

    if not nodes:
        return [], []

    q_tokens = _tokenize(question)

    # ── 1. Lexical ranking ───────────────────────────────────────────────────
    lex_sorted = sorted(nodes, key=lambda n: -_score_node(n, q_tokens))
    lex_rank   = {n["id"]: i for i, n in enumerate(lex_sorted)}

    # ── 2. Semantic ranking (best-effort) ────────────────────────────────────
    sem_rank: dict[str, int] = {}
    try:
        from .embeddings import EmbeddingIndex
        index = EmbeddingIndex.load_if_valid(nodes)
        if index is not None:
            sem_results = index.search(question, top_k=len(nodes))
            sem_rank    = {nid: i for i, (nid, _) in enumerate(sem_results)}
    except Exception as exc:      # defensive: never let embedding errors abort RAG
        log.debug("Semantic search skipped: %s", exc)

    # ── 3. RRF fusion ────────────────────────────────────────────────────────
    if sem_rank:
        rrf_scores = _rrf_merge(lex_rank, sem_rank, len(nodes))
        base_sorted = sorted(nodes, key=lambda n: -rrf_scores.get(n["id"], 0.0))
    else:
        # Lexical-only: preserve existing behaviour exactly
        base_sorted = lex_sorted

    # ── 4. Structural name-match boost ───────────────────────────────────────
    # A node whose name appears verbatim in the question (≥ 3 chars, case-
    # insensitive) is almost certainly what the user is asking about — force it
    # to the front regardless of RRF or lexical score.
    q_lower    = question.lower()
    boosted    = [n for n in base_sorted if len(n.get("name", "")) >= 3
                  and n["name"].lower() in q_lower]
    unboosted  = [n for n in base_sorted if n not in boosted]
    ranked     = boosted + unboosted

    # ── 5. Top-k selection ───────────────────────────────────────────────────
    # If purely lexical and all scores are zero, fall back to a diverse sample.
    if not sem_rank and all(_score_node(n, q_tokens) == 0.0 for n in ranked[:top_k]):
        top_nodes = _diverse_sample(nodes, top_k)
    else:
        top_nodes = ranked[:top_k]

    # ── 6. 1-hop expansion ───────────────────────────────────────────────────
    seed_ids   = {n["id"] for n in top_nodes[:_EXPAND_TOP]}
    node_by_id = {n["id"]: n for n in nodes}
    neighbor_ids: set[str] = set()
    for edge in edges:
        src, tgt = edge.get("source", ""), edge.get("target", "")
        if src in seed_ids and tgt not in seed_ids:
            neighbor_ids.add(tgt)
        elif tgt in seed_ids and src not in seed_ids:
            neighbor_ids.add(src)

    top_ids = {n["id"] for n in top_nodes}
    budget  = top_k - len(top_nodes)
    extra   = [
        node_by_id[nid]
        for nid in list(neighbor_ids)[:budget]
        if nid in node_by_id and nid not in top_ids
    ]

    context_nodes = top_nodes + extra
    context_ids   = {n["id"] for n in context_nodes}
    context_edges = [
        e for e in edges
        if e.get("source") in context_ids and e.get("target") in context_ids
    ]

    return context_nodes, context_edges


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

    lines = [f"[[{nid}]] {ntype.upper()} «{name}»"]
    if file_path:
        lines.append(f"  path: {file_path}")
    if summary:
        lines.append(f"  summary: {summary}")
    if tags:
        lines.append(f"  tags: {', '.join(str(t) for t in tags[:5])}")
    return "\n".join(lines)


def _fmt_edge(edge: dict, node_by_id: dict[str, dict]) -> str:
    src_name = node_by_id.get(edge.get("source", ""), {}).get("name", edge.get("source", "?"))
    tgt_name = node_by_id.get(edge.get("target", ""), {}).get("name", edge.get("target", "?"))
    etype = edge.get("type", "?")
    return f"  {src_name} --[{etype}]--> {tgt_name}"


def build_rag_prompt(
    question: str,
    context_nodes: list[dict],
    context_edges: list[dict],
    history: list[dict],
    graph: dict,
    settings: Settings,
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

    messages: list[dict] = [{"role": "system", "content": system_content}]

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
    )

    answer = llm.chat(messages, max_tokens=_CHAT_MAX_TOKENS)

    node_by_id = {n["id"]: n for n in graph.get("nodes", [])}
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
