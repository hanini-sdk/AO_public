"""Auxiliary prose-document context (sidecar — never touches the graph topology).

Free-prose project documents (``.txt`` and Word ``.docx``/``.docm``) carry intent
that is not in the code: what the pipeline is FOR, what a table means in business
terms, why a step exists. This module discovers and reads that prose so a later
phase can summarize it (through the existing guarded LLM client + redactor) into
a sidecar CONTEXT STORE — meaning only, never new nodes or edges.

This file (Task A) is the deterministic READ layer:
  * discover ``.txt`` + Word documents within the project root (bounded walk,
    symlinks not followed, caps on count / per-file size / total bytes),
  * read ``.txt`` as best-effort UTF-8,
  * read Word VALUES/TEXT ONLY — a ``.docx``/``.docm`` is a zip; we read only
    ``word/document.xml`` text runs with the stdlib (zipfile + xml.etree), so no
    dependency is required and no macro / embedded object is ever evaluated. A
    pinned ``python-docx`` is used only as a lazy fallback if the stdlib path
    fails; its absence degrades to skipping that one file (reported as a count).
  * load the existing project-tree inventory text as structure context.

Nothing here calls the network or an LLM; that is the next phase.
"""

from __future__ import annotations

import logging
import os
import re
import time
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .config import LANGUAGE_NAMES, Settings
from .enrich import sanitize_for_enrichment
from .scanner import IGNORE_DIRS

if TYPE_CHECKING:  # avoid importing the LLM client at module load
    from .llm import LLMClient

log = logging.getLogger("data_lineage_retro_documentation.documents")

# Recognised prose-document extensions. .docm (macro-enabled Word) is read the
# same text-only way; its macros live in a separate zip member we never open.
TXT_EXTS = (".txt", ".text")
WORD_EXTS = (".docx", ".docm")
DOC_EXTS = TXT_EXTS + WORD_EXTS

# Bounds — keep the read layer cheap and safe on a big repository.
_MAX_DOCS = 50
_MAX_DOC_BYTES = 5_000_000
_MAX_TOTAL_BYTES = 30_000_000
_MAX_TEXT_CHARS = 200_000  # per document, after extraction

_VCS_DIRS = {".git", ".hg", ".svn", ".bzr"}
_SKIP_DIRS = IGNORE_DIRS | _VCS_DIRS
_W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


@dataclass
class DocText:
    """One discovered document: a short display label, its extension, and the
    extracted plain text. ``label`` is a basename for display in the LOCAL
    context store only — never emitted to the sanitized run report."""
    label: str
    ext: str
    text: str


def _norm_ext(name: str) -> str:
    return Path(name).suffix.lower()


def _read_txt(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _read_docx_stdlib(path: Path) -> str:
    """Extract paragraph text from a Word file using only the stdlib.

    Opens the zip and reads ONLY ``word/document.xml`` — the text body. The
    macro project (``word/vbaProject.bin`` in a .docm) and any embedded objects
    are never opened, so nothing executable is touched."""
    with zipfile.ZipFile(str(path)) as z:
        with z.open("word/document.xml") as fh:
            data = fh.read()
    root = ET.fromstring(data)
    paragraphs: list[str] = []
    for para in root.iter(_W_NS + "p"):
        runs = [t.text for t in para.iter(_W_NS + "t") if t.text]
        if runs:
            paragraphs.append("".join(runs))
    return "\n".join(paragraphs)


def _read_word(path: Path) -> str | None:
    """Word text via the stdlib; lazy ``python-docx`` only as a fallback. Returns
    None (skip) if neither can read it — never raises, never runs macros."""
    try:
        return _read_docx_stdlib(path)
    except Exception as exc:  # noqa: BLE001 - malformed/locked file -> try fallback
        log.debug("stdlib docx read failed: %s", exc)
    try:
        import docx  # python-docx — optional, lazy; pinned in requirements
        document = docx.Document(str(path))
        return "\n".join(p.text for p in document.paragraphs)
    except Exception as exc:  # noqa: BLE001 - reader unavailable/failed -> skip
        log.debug("python-docx fallback unavailable/failed: %s", exc)
        return None


def discover_documents(project_root: str) -> tuple[list[DocText], dict]:
    """Discover + read prose documents within ``project_root``.

    Returns (docs, stats). ``stats`` is count-only (safe for the run report):
    found/read totals split by kind, plus the count of Word files skipped for
    lack of a reader. Bounded walk; symlinked directories are not followed."""
    stats = {
        "txt_found": 0, "word_found": 0, "read": 0,
        "failed": 0, "word_skipped_no_reader": 0,
    }
    docs: list[DocText] = []
    try:
        root = Path(project_root).resolve()
        if not root.is_dir():
            return docs, stats
    except OSError:
        return docs, stats

    total_bytes = 0
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = [
            d for d in dirnames
            if d not in _SKIP_DIRS
            and not d.startswith(".")
            and not os.path.islink(os.path.join(dirpath, d))
        ]
        for fn in sorted(filenames):
            ext = _norm_ext(fn)
            if ext not in DOC_EXTS:
                continue
            full = Path(dirpath) / fn
            try:
                if full.is_symlink() or not full.is_file():
                    continue
                size = full.stat().st_size
            except OSError:
                continue
            if size > _MAX_DOC_BYTES:
                continue
            if total_bytes + size > _MAX_TOTAL_BYTES:
                return docs, stats
            is_word = ext in WORD_EXTS
            if is_word:
                stats["word_found"] += 1
            else:
                stats["txt_found"] += 1
            text = _read_word(full) if is_word else _read_txt(full)
            if text is None:
                stats["failed"] += 1
                if is_word:
                    stats["word_skipped_no_reader"] += 1
                continue
            total_bytes += size
            docs.append(DocText(label=fn, ext=ext, text=text[:_MAX_TEXT_CHARS]))
            stats["read"] += 1
            if len(docs) >= _MAX_DOCS:
                return docs, stats
    return docs, stats


def load_project_tree_text(data_dir: str | Path) -> str:
    """Best-effort read of the project-tree inventory (structure context). Empty
    string when absent/unreadable."""
    try:
        path = Path(data_dir) / "project_tree.txt"
        if path.is_file():
            return path.read_text(encoding="utf-8", errors="replace")[:_MAX_TEXT_CHARS]
    except OSError:
        pass
    return ""


# ===========================================================================
# Task B — summarize documents into a sidecar CONTEXT STORE.
#
# Every document's text is passed through the EXISTING credential redactor
# (sanitize_for_enrichment) BEFORE any LLM call, and every call reuses the
# caller's EXISTING guarded LLM client — no new egress path is introduced. Large
# prose is chunked, each chunk summarized, and the chunk summaries aggregated
# into a per-document summary. The store is a sidecar — it never adds a node or
# an edge.
#
# The real company constraint on the internal service is a request RATE limit
# (~10 calls/minute), NOT a total volume limit — so the phase is throttled by
# call RATE, not capped at a fixed number of calls. ALL discovered documents
# (within the file-read safety caps above) are summarized; the limiter spaces
# the calls out to stay under the rate. The phase still runs after the graph is
# served (non-blocking) and stays best-effort: any call failure degrades to an
# empty summary (recorded as not-summarized), never raising or blocking.
# ===========================================================================

# The ONE knob that bounds the document phase: maximum internal-service calls
# per minute. Set below the stated ~10/min so there is headroom. Tune here.
DOC_LLM_CALLS_PER_MINUTE = 8


class _RateLimiter:
    """Deterministic minimum-spacing throttle. Guarantees no more than
    ``per_minute`` calls in any 60s window by enforcing a fixed minimum gap
    between consecutive calls (sleeping the remainder when a call comes early).
    Monotonic clock; no wall-clock dependency."""

    def __init__(self, per_minute: int) -> None:
        self.min_spacing = 60.0 / max(1, int(per_minute))
        self._last = 0.0  # monotonic timestamp of the previous call (0 = none yet)

    def wait(self) -> None:
        if self._last:
            remaining = self.min_spacing - (time.monotonic() - self._last)
            if remaining > 1e-3:  # skip sub-ms sleeps (OS timer granularity)
                time.sleep(remaining)
        self._last = time.monotonic()


_CHUNK_CHARS = 6000
_MAX_CHUNKS_PER_DOC = 4
_MAX_DOC_SUMMARY_CHARS = 1200
_DOC_SUMMARY_TOKENS = 180
_GLOBAL_NARRATIVE_TOKENS = 360
CONTEXT_STORE_VERSION = 1


def _chunk_text(text: str) -> list[str]:
    """Split prose into <= _CHUNK_CHARS chunks on paragraph boundaries, capped to
    _MAX_CHUNKS_PER_DOC (oversize tails are dropped — recorded by the caller)."""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= _CHUNK_CHARS:
        return [text]
    chunks: list[str] = []
    buf: list[str] = []
    size = 0
    for para in text.split("\n\n"):
        p = para.strip()
        if not p:
            continue
        if size + len(p) > _CHUNK_CHARS and buf:
            chunks.append("\n\n".join(buf))
            buf, size = [], 0
            if len(chunks) >= _MAX_CHUNKS_PER_DOC:
                return chunks
        buf.append(p)
        size += len(p)
    if buf and len(chunks) < _MAX_CHUNKS_PER_DOC:
        chunks.append("\n\n".join(buf))
    return chunks[:_MAX_CHUNKS_PER_DOC]


def _lang_clause(settings: Settings) -> str:
    return f"Answer in {LANGUAGE_NAMES.get(settings.language, 'English')}."


def _safe_chat(llm: "LLMClient", messages: list[dict], max_tokens: int,
               limiter: "_RateLimiter") -> str:
    """One rate-limited, guarded completion. The limiter spaces calls to stay
    under the per-minute rate; ANY failure (including an egress block) degrades
    to an empty string so the document phase never raises or blocks."""
    limiter.wait()  # throttle BEFORE the call so the rate is never exceeded
    try:
        return (llm.chat(messages, max_tokens=max_tokens) or "").strip()
    except Exception as exc:  # noqa: BLE001 - best-effort; never fatal
        log.debug("document summary call failed: %s", exc)
        return ""


def _chunk_messages(project_name: str, chunk: str, settings: Settings) -> list[dict]:
    return [
        {"role": "system", "content":
            "You summarize a project document for engineers. Capture purpose, the "
            "data/entities it describes, and any business meaning, in 2-4 sentences. "
            "Plain text only. " + _lang_clause(settings)},
        {"role": "user", "content":
            f"Project: {project_name}\n\nDocument excerpt:\n{chunk}\n\nWrite the summary."},
    ]


def _narrative_messages(project_name: str, doc_summaries: list[tuple[str, str]],
                        tree_text: str, settings: Settings) -> list[dict]:
    joined = "\n".join(f"- {label}: {summ}" for label, summ in doc_summaries[:20])
    tree_excerpt = (tree_text or "")[:4000]
    return [
        {"role": "system", "content":
            "You write a short, high-level narrative of a software/data project from "
            "its document summaries and file structure. 3-6 sentences, plain text. "
            + _lang_clause(settings)},
        {"role": "user", "content":
            f"Project: {project_name}\n\nDocument summaries:\n{joined or '(none)'}\n\n"
            f"Project structure (excerpt):\n{tree_excerpt or '(none)'}\n\n"
            "Write the project narrative."},
    ]


# ----- Task C: deterministic exact-name association (no LLM, no new topology) --
# A document is attached to a node ONLY where the node's exact name (or its
# dictionary business label) occurs verbatim in the document's (redacted) text.
# This is a pure string match — the LLM only ever summarizes; it never decides
# which node a passage belongs to. Short names are noise-prone, so a name below
# the minimum length only matches via its business label; the dotted/longer name
# wins (longest-first, non-overlapping) so a suffix never steals a qualified hit.
_MIN_BARE_NAME_LEN = 4
_MAX_SNIPPETS_PER_NODE = 3
_MAX_SNIPPET_CHARS = 600


def _build_name_index(graph: dict) -> dict[str, set[str]]:
    """Lowercased match token -> set of node ids. Indexes node names at or above
    the minimum length, plus every dictionary business label (any length)."""
    index: dict[str, set[str]] = {}
    for n in graph.get("nodes", []) or []:
        nid = n.get("id")
        if not nid:
            continue
        name = (n.get("name") or "").strip()
        if name and len(name) >= _MIN_BARE_NAME_LEN:
            index.setdefault(name.lower(), set()).add(nid)
        attrs = n.get("attributes") or {}
        label = str(attrs.get("business_label") or "").strip()
        if label:
            index.setdefault(label.lower(), set()).add(nid)
    return index


def _associate_nodes(store: dict, index: dict[str, set[str]],
                     doc_redacted: dict[str, str]) -> int:
    """For each summarized document, attach its summary to every node whose name/
    label occurs verbatim in the document text. Returns the number of distinct
    nodes that gained context. Mutates store['nodes'] only — never the graph."""
    if not index:
        return 0
    keys = sorted(index.keys(), key=len, reverse=True)  # longest-first
    pattern = re.compile(
        r"(?<![\w])(" + "|".join(re.escape(k) for k in keys) + r")(?![\w])",
        re.IGNORECASE,
    )
    store_nodes: dict = store.setdefault("nodes", {})
    for entry in store.get("documents", []):
        summary = entry.get("summary") or ""
        if not summary:
            continue  # prose we could not summarize contributes only to the narrative
        text = doc_redacted.get(entry["id"], "")
        if not text:
            continue
        matched: set[str] = set()
        for m in pattern.finditer(text):
            matched.update(index.get(m.group(1).lower(), ()))
        for nid in matched:
            lst = store_nodes.setdefault(nid, [])
            if len(lst) >= _MAX_SNIPPETS_PER_NODE:
                continue
            if any(s.get("docId") == entry["id"] for s in lst):
                continue
            lst.append({
                "docId": entry["id"],
                "label": entry["label"],
                "snippet": summary[:_MAX_SNIPPET_CHARS],
            })
    return len(store_nodes)


def build_context_store(
    graph: dict,
    docs: list[DocText],
    tree_text: str,
    llm: "LLMClient",
    settings: Settings,
    *,
    calls_per_minute: int = DOC_LLM_CALLS_PER_MINUTE,
) -> tuple[dict, dict]:
    """Summarize ``docs`` into a sidecar context store (global narrative + per-doc
    summaries). ALL documents are summarized; calls are throttled to stay under
    ``calls_per_minute`` (no total-volume cap). Returns (store, stats). Topology
    is never touched; Task C fills the per-node associations. Never raises."""
    project = graph.get("project") or {}
    project_name = project.get("name") or "this project"

    limiter = _RateLimiter(calls_per_minute)
    calls_made = 0
    summarized = 0
    not_summarized = 0
    chunks_total = 0
    documents: list[dict] = []
    doc_summaries: list[tuple[str, str]] = []
    doc_redacted: dict[str, str] = {}  # doc id -> redacted text (for Task C matching)

    for i, doc in enumerate(docs):
        redacted = sanitize_for_enrichment(doc.text)  # redaction BEFORE any LLM call
        doc_redacted[f"d{i}"] = redacted
        chunks = _chunk_text(redacted)
        chunks_total += len(chunks)
        chunk_summaries: list[str] = []
        for ch in chunks:  # every chunk of every document is summarized (throttled)
            s = _safe_chat(llm, _chunk_messages(project_name, ch, settings),
                           _DOC_SUMMARY_TOKENS, limiter)
            calls_made += 1
            if s:
                chunk_summaries.append(s)
        if chunk_summaries:
            summary = (chunk_summaries[0] if len(chunk_summaries) == 1
                       else " ".join(chunk_summaries))[:_MAX_DOC_SUMMARY_CHARS]
            summarized += 1
        else:
            summary = ""  # empty doc or every chunk call failed -> not summarized
            not_summarized += 1
        documents.append({
            "id": f"d{i}", "label": doc.label, "ext": doc.ext,
            "summary": summary, "chunks": len(chunks), "summarized": bool(summary),
        })
        if summary:
            doc_summaries.append((doc.label, summary))

    # Global narrative: one (throttled) call; if it fails, fall back to a
    # deterministic concatenation of the per-document summaries (no LLM).
    global_narrative = ""
    if doc_summaries or tree_text:
        global_narrative = _safe_chat(
            llm, _narrative_messages(project_name, doc_summaries, tree_text, settings),
            _GLOBAL_NARRATIVE_TOKENS, limiter,
        )
        calls_made += 1
        if not global_narrative:
            global_narrative = " ".join(s for _, s in doc_summaries)[:_MAX_DOC_SUMMARY_CHARS]

    store = {
        "version": CONTEXT_STORE_VERSION,
        "globalNarrative": global_narrative,
        "documents": documents,
        "nodes": {},  # node_id -> [ {docId, label, snippet} ]
    }

    # Task C: attach each summarized document to the nodes it names verbatim.
    nodes_with_context = _associate_nodes(store, _build_name_index(graph), doc_redacted)

    stats = {
        "summarized": summarized,
        "not_summarized": not_summarized,
        "chunks": chunks_total,
        "llm_calls_made": calls_made,
        "calls_per_minute": int(calls_per_minute),
        "nodes_with_context": nodes_with_context,
    }
    return store, stats
