"""Project story synthesis — one ordered narrative built from the graph.

The story is synthesized FROM the existing per-node summaries + the lineage/graph
edges, never from raw source code. It is a map-reduce: per-node summaries are the
map (already produced by enrichment) -> per-section prose is the first reduce
(one LLM call per narrative section over that section's node summaries) -> the
overview is the final reduce over a few representative summaries. Every LLM call
goes through the EXISTING internal LLM client; every synthesis input is passed
through the EXISTING credential redactor (sanitize_for_enrichment) as defense in
depth, so no raw SQL or secret can leave even though the summaries are already
redacted. The persisted artifact carries only entity names + descriptions already
present in the redacted graph, plus the ids of the nodes each section describes
(for a later interactive tour). Generation never aborts: each call degrades to a
deterministic narrative built straight from the summaries.

Narrative order: overview -> orchestrators (entry points) -> data flow following
lineage (sources -> transformations -> outputs) -> components (code projects) ->
glossary of key tables and columns.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone

from .config import LANGUAGE_NAMES, Settings
from .enrich import sanitize_for_enrichment
from .llm import LLMClient

log = logging.getLogger("data_lineage_retro_documentation.story")

STORY_VERSION = "1.0.0"

# Input/size guards (there is no token limiter in the stack — the generator must
# bound its own prompts). Mirrors enrich.py's explicit slicing.
_MAX_SUMMARIES_PER_SECTION = 40
_MAX_GLOSSARY_TABLES = 40
_MAX_GLOSSARY_COLS = 12
_SECTION_MAX_TOKENS = 500
_OVERVIEW_MAX_TOKENS = 450


def graph_fingerprint(graph: dict) -> str:
    """Content hash over node (id, type, summary) + edge (source, target, type).

    Stable across re-runs that don't change the modelled content, so the story
    can be regenerated exactly when the graph's content changes and served from
    cache otherwise.
    """
    h = hashlib.sha256()
    for n in sorted(graph.get("nodes", []) or [], key=lambda x: str(x.get("id", ""))):
        h.update(
            (str(n.get("id", "")) + "\x00" + str(n.get("type", "")) + "\x00"
             + str(n.get("summary", "")) + "\n").encode("utf-8", "replace")
        )
    for e in sorted(
        graph.get("edges", []) or [],
        key=lambda x: (str(x.get("source", "")), str(x.get("target", "")), str(x.get("type", ""))),
    ):
        h.update(
            (str(e.get("source", "")) + "\x00" + str(e.get("target", "")) + "\x00"
             + str(e.get("type", "")) + "\n").encode("utf-8", "replace")
        )
    return h.hexdigest()


# ------------------------------------------------------------------- helpers
def _node_line(n: dict) -> str:
    name = n.get("name") or n.get("id") or "(unnamed)"
    summ = (n.get("summary") or "").strip()
    return f"- {name}: {summ}" if summ else f"- {name}"


def _summaries_block(nodes: list[dict], cap: int = _MAX_SUMMARIES_PER_SECTION) -> tuple[str, int]:
    """Redacted, capped bullet list of the nodes' summaries + how many were
    dropped by the cap. Redaction is defense-in-depth — the summaries are already
    produced by the redacted enrichment path."""
    shown = nodes[:cap]
    text = "\n".join(_node_line(n) for n in shown)
    return sanitize_for_enrichment(text), len(nodes) - len(shown)


def _lang_clause(settings: Settings) -> str:
    return f"Answer in {LANGUAGE_NAMES.get(settings.language, 'English')}."


def _write_section(
    llm: LLMClient,
    settings: Settings,
    *,
    key: str,
    instruction: str,
    nodes: list[dict],
    fallback_intro: str,
) -> str | None:
    """One reduce step: synthesize a section's prose from its node summaries.

    Returns None when there are no nodes (the section is skipped). On any LLM
    failure it returns a deterministic narrative (intro + the summaries), so the
    section is never empty and generation never aborts.
    """
    if not nodes:
        return None
    block, extra = _summaries_block(nodes)
    fallback = fallback_intro + "\n\n" + block + (f"\n\n(+{extra} more not shown.)" if extra > 0 else "")
    system = (
        "You are a technical writer producing plain, factual project documentation. "
        "Write clear, well-structured prose in short paragraphs. Use ONLY the information "
        "given; do not invent details and do not describe how this documentation was "
        "produced. " + _lang_clause(settings)
    )
    user = (
        f"{instruction}\n\n"
        "Write 1-3 short paragraphs of prose. Do NOT add a heading (it is added separately) "
        "and do NOT just repeat the list.\n\n"
        f"Entities and their descriptions:\n{block}"
        + (f"\n(and {extra} more not shown)" if extra > 0 else "")
    )
    try:
        text = llm.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=_SECTION_MAX_TOKENS,
        ).strip()
        return text or fallback
    except Exception as exc:  # noqa: BLE001 - degrade to deterministic narrative
        log.debug("story section %r failed: %s", key, exc)
        return fallback


def _write_overview(
    llm: LLMClient, settings: Settings, name: str, description: str,
    files: list[dict], tables: list[dict], columns: list[dict], orchestrators: list[dict],
) -> str:
    # R2: redact the project name + description before they enter the prompt (or
    # the fallback) — "redaction before every LLM call" with no exceptions.
    name = sanitize_for_enrichment(name)
    description = sanitize_for_enrichment(description)
    counts = f"{len(files)} files, {len(tables)} tables, {len(columns)} columns"
    reps = (orchestrators[:5] or files[:5]) + tables[:8]
    block = sanitize_for_enrichment("\n".join(_node_line(n) for n in reps))
    fallback = (description.strip() or f"{name} is a data/code project.") + f" It comprises {counts}."
    system = (
        "You are a technical writer producing plain, factual project documentation. "
        "Write a short, engaging overview in 1-2 paragraphs. Use ONLY the information given; "
        "do not invent details and do not describe how this documentation was produced. "
        + _lang_clause(settings)
    )
    user = (
        f"Project: {name}\n"
        f"Known description: {description.strip() or '(none)'}\n"
        f"Scale: {counts}\n\n"
        f"Representative entities:\n{block}\n\n"
        "Write a 1-2 paragraph overview of what this project does and how its parts fit together."
    )
    try:
        text = llm.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=_OVERVIEW_MAX_TOKENS,
        ).strip()
        return text or fallback
    except Exception as exc:  # noqa: BLE001
        log.debug("story overview failed: %s", exc)
        return fallback


def _classify(graph: dict) -> dict:
    """Bucket nodes by role using the graph's data-flow edges (file<->table
    writes_to/reads_from, table<-table provenance, file->file calls/runs)."""
    nodes = graph.get("nodes", []) or []
    edges = graph.get("edges", []) or []
    by_id = {n["id"]: n for n in nodes if n.get("id")}

    files = [n for n in nodes if n.get("type") == "file"]
    tables = [n for n in nodes if n.get("type") == "table"]
    columns = [n for n in nodes if n.get("type") == "column"]

    produced_by: dict[str, set] = {}   # table id -> files that write it
    read_by: dict[str, set] = {}       # table id -> files that read it
    prov_up: dict[str, set] = {}       # table id -> upstream table ids
    prov_down: dict[str, set] = {}     # table id -> downstream table ids
    file_runs_file: dict[str, set] = {}  # file id -> file ids it calls/runs

    for e in edges:
        s, t, ty = e.get("source"), e.get("target"), e.get("type")
        st = by_id.get(s, {}).get("type")
        tt = by_id.get(t, {}).get("type")
        if ty == "writes_to" and st == "file" and tt == "table":
            produced_by.setdefault(t, set()).add(s)
        elif ty == "reads_from":
            if st == "file" and tt == "table":
                read_by.setdefault(t, set()).add(s)
            elif st == "table" and tt == "table":
                prov_up.setdefault(s, set()).add(t)
                prov_down.setdefault(t, set()).add(s)
        elif st == "file" and tt == "file" and ty in ("calls", "imports"):
            file_runs_file.setdefault(s, set()).add(t)

    orch_ids = set(file_runs_file)
    for n in files:
        if set(n.get("tags") or []) & {"shell", "joblist", "ksh", "bash", "zsh"}:
            orch_ids.add(n["id"])
    orchestrators = [n for n in files if n["id"] in orch_ids]

    def has_down(tid: str) -> bool:
        return bool(read_by.get(tid)) or bool(prov_down.get(tid))

    # File-based classification (robust even without table<-table lineage):
    # a table nobody writes is an external source; one nobody reads/feeds is a
    # terminal output; the rest are intermediate transforms.
    sources = [t for t in tables if not produced_by.get(t["id"]) and has_down(t["id"])]
    sinks = [t for t in tables if produced_by.get(t["id"]) and not has_down(t["id"])]
    src_ids = {t["id"] for t in sources}
    sink_ids = {t["id"] for t in sinks}
    transforms = [t for t in tables if t["id"] not in src_ids and t["id"] not in sink_ids]

    return {
        "by_id": by_id, "edges": edges,
        "files": files, "tables": tables, "columns": columns,
        "orchestrators": orchestrators, "orch_ids": orch_ids,
        "sources": sources, "transforms": transforms, "sinks": sinks,
    }


def _build_glossary(c: dict) -> tuple[str | None, list[str]]:
    """Deterministic glossary (no LLM): key tables + their columns, or key files
    for a code project. Markdown bullet list."""
    by_id = c["by_id"]
    lines: list[str] = []
    ids: list[str] = []
    if c["tables"]:
        table_ids = {t["id"] for t in c["tables"]}
        col_ids = {col["id"] for col in c["columns"]}
        cols_by_table: dict[str, list[str]] = {}
        for e in c["edges"]:
            if e.get("type") == "contains" and e.get("source") in table_ids and e.get("target") in col_ids:
                cols_by_table.setdefault(e["source"], []).append(e["target"])
        for t in c["tables"][:_MAX_GLOSSARY_TABLES]:
            ids.append(t["id"])
            lines.append(f"- **{t.get('name')}** — {(t.get('summary') or '').strip()}")
            cols = cols_by_table.get(t["id"], [])
            if cols:
                shown = cols[:_MAX_GLOSSARY_COLS]
                ids.extend(shown)
                names = [by_id.get(cid, {}).get("name", cid) for cid in shown]
                extra = len(cols) - len(shown)
                lines.append("  - columns: " + ", ".join(names) + (f", +{extra}" if extra > 0 else ""))
    else:
        for f in c["files"][:_MAX_GLOSSARY_TABLES]:
            ids.append(f["id"])
            lines.append(f"- **{f.get('name')}** — {(f.get('summary') or '').strip()}")
    body = "\n".join(lines).strip()
    return (body or None), ids


def generate_story(graph: dict, llm: LLMClient, settings: Settings) -> dict:
    """Synthesize the ordered project story dict from the graph + the LLM.

    Reuses the caller's open LLMClient (do not close it here). Never raises for
    content reasons — each section degrades to a deterministic narrative.
    """
    project = graph.get("project", {}) or {}
    name = project.get("name") or "this project"
    description = str(project.get("description") or "")
    c = _classify(graph)

    sections: list[dict] = []

    def add(sec_id: str, title: str, body: str | None, node_ids: list[str]) -> None:
        if body:
            sections.append({"id": sec_id, "title": title, "body": body, "nodeIds": list(node_ids)})

    add("overview", "Overview",
        _write_overview(llm, settings, name, description,
                        c["files"], c["tables"], c["columns"], c["orchestrators"]),
        [])

    add("orchestration", "Orchestration and entry points",
        _write_section(llm, settings, key="orchestration", nodes=c["orchestrators"],
                       instruction=f"Describe the entry-point scripts that orchestrate {name} and how they drive the work, in execution order where it is clear.",
                       fallback_intro="Entry-point scripts that orchestrate the work:"),
        [n["id"] for n in c["orchestrators"]])

    add("sources", "Data sources",
        _write_section(llm, settings, key="sources", nodes=c["sources"],
                       instruction="Describe the source tables that feed the pipeline (the raw inputs it reads).",
                       fallback_intro="Source tables (raw inputs):"),
        [n["id"] for n in c["sources"]])

    add("transformations", "Transformations",
        _write_section(llm, settings, key="transformations", nodes=c["transforms"],
                       instruction="Describe the intermediate tables and how data is transformed as it flows from the sources toward the outputs.",
                       fallback_intro="Intermediate tables where data is transformed:"),
        [n["id"] for n in c["transforms"]])

    add("outputs", "Outputs",
        _write_section(llm, settings, key="outputs", nodes=c["sinks"],
                       instruction="Describe the final output tables (sinks) the pipeline produces.",
                       fallback_intro="Output tables (sinks):"),
        [n["id"] for n in c["sinks"]])

    if not c["tables"]:
        non_orch = [n for n in c["files"] if n["id"] not in c["orch_ids"]] or c["files"]
        add("components", "Key components",
            _write_section(llm, settings, key="components", nodes=non_orch,
                           instruction=f"Describe the key components and modules of {name} and the role each plays.",
                           fallback_intro="Key components:"),
            [n["id"] for n in non_orch])

    glossary_body, glossary_ids = _build_glossary(c)
    add("glossary", "Glossary", glossary_body, glossary_ids)

    return {
        "version": STORY_VERSION,
        "generatedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "graphFingerprint": graph_fingerprint(graph),
        "language": settings.language,
        "title": f"{name} — Project Story",
        "sections": sections,
    }
