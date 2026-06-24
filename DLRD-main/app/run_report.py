"""Sanitized run report - a per-run plain-text artifact for architecture review.

On each run, write run_report.txt to the run-output directory (gitignored). Its
sole purpose is to let a reviewer who CANNOT see the data understand the run:
what was attempted, what succeeded against the code's intended steps, what
failed, the overall flow, and whether the objective (a knowledge graph) was met.

The file is meant to be copied off a secure machine, so it is sanitized BY
CONSTRUCTION. The report's data model (``RunReport``) holds ONLY integer
counters, validated file-extension tokens, fixed stage labels, and enumerated
outcome codes. There is no field able to hold a path, a file/table/column/
variable name, or any SQL/data value - so none can reach the report. The writer
formats only those aggregates plus fixed English templates. A final
project-root assertion in ``write_run_report`` is a cheap secondary catch.

It reuses the identifier-free run-diagnostics counters; it does not change or
duplicate the diagnostics' own collection. No LLM and no network.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .project_tree import _NO_EXT_LABEL, handled_extensions

log = logging.getLogger("data_lineage_retro_documentation.run_report")

_OTHER_EXT_LABEL = "(other)"

# Fixed vocabularies. The collector accepts ONLY these labels/codes, validated
# extension tokens, and integer counts - never a project string.
_STAGE_LABELS = (
    "Scan files", "Parse structure", "Enrich summaries", "Assemble graph",
    "Validate graph", "Write knowledge graph", "Write project story",
    "Write run diagnostics", "Build embedding index",
)
_STAGE_OUTCOMES = ("completed", "skipped", "errored")
# Fixed SQL parse-failure category codes (mirrors sqlproc.SQL_FAIL_CATEGORIES).
# Duplicated as a local constant so run_report stays free of the sqlglot import;
# these are constant labels, never derived from SQL, so the allowlist keys built
# from them remain identifier-free.
_SQL_FAIL_CATEGORIES = (
    "sigil_interpolation", "bteq_dot_command", "missing_terminator",
    "unsupported_teradata", "tokenizer_error", "empty_after_preprocess", "other",
)
_SQL_FILE_GAP_REASONS = (
    "files_unparsed", "empty_after_preprocess", "all_statements_failed",
    "parsed_no_entities",
)
_COUNTER_NAMES = frozenset({
    "source_files_parsed", "grammar_parse_failures",
    "sql_parse_failures", "sql_recovered_from_shell",
    "sql_comment_blocks_detected", "sql_comment_blocks_recovered",
    "sql_comment_rejected_parse_failed", "sql_comment_rejected_no_dml",
    "sql_comment_rejected_other", "sql_statements_recovered_from_comments",
    "sql_comment_lines", "sql_lines_total",
    "tables", "columns", "column_lineage_edges",
    "columns_without_lineage", "tables_without_provenance",
    "refs_resolved", "refs_unresolved", "refs_skipped", "refs_internal_calls",
    "enriched", "enrich_failed",
    # Document phase (counts only — no doc names/content ever reach the report).
    "doc_txt_found", "doc_word_found", "doc_read", "doc_summarized",
    "doc_not_summarized", "doc_failed", "doc_word_skipped_no_reader",
    "doc_chunks", "doc_llm_calls_made", "doc_calls_per_minute",
    "doc_nodes_with_context",
}
    | {f"sqlfail_active_{c}" for c in _SQL_FAIL_CATEGORIES}
    | {f"sqlfail_commented_{c}" for c in _SQL_FAIL_CATEGORIES}
    | {f"sqlgap_{r}" for r in _SQL_FILE_GAP_REASONS})
# An extension token: a dot then short word characters, no separators or spaces.
_STRICT_EXT_RE = re.compile(r"\.[a-z0-9_+-]{1,16}\Z")


def _norm_ext(ext: str) -> str:
    """Map a raw extension to a safe, generic token: the no-extension label, a
    validated extension, or the catch-all '(other)' for anything unusual. This
    guarantees no path/identifier token can ride into the report as an extension."""
    if not ext or ext == _NO_EXT_LABEL:
        return _NO_EXT_LABEL
    low = ext.lower()
    return low if _STRICT_EXT_RE.match(low) else _OTHER_EXT_LABEL


@dataclass
class _ExtRow:
    found: int = 0          # files of this extension in the project tree
    parsed: int = 0         # files of this extension the pipeline parsed
    handled: bool = False


@dataclass
class RunReport:
    """Sanitized run-report data model. Holds ONLY integer counters, validated
    file-extension tokens, fixed stage labels, and enumerated outcome codes.
    There is no field able to hold a path, name, identifier, or content value."""
    run_completed: str = "no"      # yes | partial | no
    objective_met: str = "no"      # yes | no
    files_in_tree: int = 0
    files_scanned: int = 0
    files_unreadable: int = 0
    files_oversize: int = 0
    by_ext: dict = field(default_factory=dict)    # ext token -> _ExtRow
    counters: dict = field(default_factory=dict)  # fixed name -> int
    stages: list = field(default_factory=list)    # (stage_label, outcome_code)

    # -- guarded mutators: the only way data enters the model --
    def set_state(self, run_completed: str, objective_met: str) -> None:
        assert run_completed in ("yes", "partial", "no")
        assert objective_met in ("yes", "no")
        self.run_completed = run_completed
        self.objective_met = objective_met

    def set_ext(self, ext: str, found: int, parsed: int, handled: bool) -> None:
        token = self._safe_ext(ext)
        row = self.by_ext.get(token) or _ExtRow()
        row.found += int(found)
        row.parsed += int(parsed)
        row.handled = row.handled or bool(handled)
        self.by_ext[token] = row

    def set_counter(self, name: str, value: int) -> None:
        assert name in _COUNTER_NAMES, name
        self.counters[name] = int(value)

    def add_stage(self, label: str, outcome: str) -> None:
        assert label in _STAGE_LABELS, label
        assert outcome in _STAGE_OUTCOMES, outcome
        self.stages.append((label, outcome))

    @staticmethod
    def _safe_ext(ext: str) -> str:
        if ext in (_NO_EXT_LABEL, _OTHER_EXT_LABEL):
            return ext
        if _STRICT_EXT_RE.match(ext or ""):
            return ext.lower()
        raise ValueError("run report rejects a non-extension key")


def build_run_report(*, diag: dict, ext_counts: dict, scanned_by_ext: dict,
                     parsed_by_ext: dict, column_lineage_edges: int,
                     story_ok: bool, embed_skipped: bool,
                     doc_stats: dict | None = None) -> RunReport:
    """Populate a RunReport from the run-diagnostics counters (already
    identifier-free) plus the per-extension and lineage counts. Only integers,
    extension tokens, and fixed codes are written into the collector."""
    m = diag.get("metrics", {})
    files = m.get("files", {})
    nodes = m.get("nodes", {})
    sql = m.get("sql", {})
    refs = m.get("references", {})
    lin = m.get("lineage", {})
    enr = m.get("enrichment", {})
    rec = sql.get("recovery", {}) or {}
    rej = rec.get("rejected", {}) or {}

    rep = RunReport()

    # --- header state ---
    objective = "yes" if int(nodes.get("total", 0)) > 0 else "no"
    grammar_failures = sum(int(v) for v in files.get("parse_failed_by_category", {}).values())
    any_failure = (
        grammar_failures > 0
        or int(sql.get("parse_failures", 0)) > 0
        or int(enr.get("failed", 0)) > 0
        or not story_ok
    )
    completed = "no" if objective == "no" else ("partial" if any_failure else "yes")
    rep.set_state(completed, objective)

    # --- files by extension ---
    handled = handled_extensions()
    found_map: Counter = Counter()
    for k, v in (ext_counts or {}).items():
        found_map[_norm_ext(k)] += int(v)
    parsed_map: Counter = Counter()
    for k, v in (parsed_by_ext or {}).items():
        parsed_map[_norm_ext(k)] += int(v)

    rep.files_in_tree = int(sum(found_map.values()))
    rep.files_scanned = int(sum(int(v) for v in (scanned_by_ext or {}).values()))
    rep.files_unreadable = int(files.get("skipped", 0))
    rep.files_oversize = int(files.get("oversize", 0))
    for token in set(found_map) | set(parsed_map):
        is_handled = token not in (_NO_EXT_LABEL, _OTHER_EXT_LABEL) and token in handled
        rep.set_ext(token, found_map.get(token, 0), parsed_map.get(token, 0), is_handled)

    # --- parsing ---
    rep.set_counter("source_files_parsed", files.get("parsed", 0))
    rep.set_counter("grammar_parse_failures", grammar_failures)
    rep.set_counter("sql_parse_failures", sql.get("parse_failures", 0))
    rep.set_counter("sql_recovered_from_shell", sql.get("macro_statements_recovered", 0))
    rep.set_counter("sql_comment_blocks_detected", rec.get("detected", 0))
    rep.set_counter("sql_comment_blocks_recovered", rec.get("recovered", 0))
    rep.set_counter("sql_comment_rejected_parse_failed", rej.get("parse_failed", 0))
    rep.set_counter("sql_comment_rejected_no_dml", rej.get("no_dml", 0))
    rep.set_counter("sql_comment_rejected_other", rej.get("other", 0))
    rep.set_counter("sql_statements_recovered_from_comments", rec.get("statements", 0))
    rep.set_counter("sql_comment_lines", sql.get("comment_lines", 0))
    rep.set_counter("sql_lines_total", sql.get("total_lines", 0))

    # --- SQL parse failures by fixed category, per surface ---
    fail_cat = sql.get("fail_categories", {}) or {}
    active_cat = fail_cat.get("active", {}) or {}
    commented_cat = fail_cat.get("commented", {}) or {}
    for cat in _SQL_FAIL_CATEGORIES:
        rep.set_counter(f"sqlfail_active_{cat}", int(active_cat.get(cat, 0)))
        rep.set_counter(f"sqlfail_commented_{cat}", int(commented_cat.get(cat, 0)))
    file_gap = sql.get("file_gap", {}) or {}
    for reason in _SQL_FILE_GAP_REASONS:
        rep.set_counter(f"sqlgap_{reason}", int(file_gap.get(reason, 0)))

    # --- columns / lineage ---
    rep.set_counter("tables", sql.get("tables", 0))
    rep.set_counter("columns", sql.get("columns", 0))
    rep.set_counter("column_lineage_edges", int(column_lineage_edges))
    rep.set_counter("columns_without_lineage", lin.get("columns_without_lineage", 0))
    rep.set_counter("tables_without_provenance", lin.get("tables_without_provenance", 0))

    # --- references ---
    rep.set_counter("refs_resolved", refs.get("resolved", 0))
    rep.set_counter("refs_unresolved", refs.get("missing", 0))
    rep.set_counter("refs_skipped", refs.get("skipped", 0))
    rep.set_counter("refs_internal_calls", refs.get("function_calls", 0))

    # --- enrichment ---
    rep.set_counter("enriched", enr.get("enriched", 0))
    rep.set_counter("enrich_failed", enr.get("failed", 0))

    # --- document phase (counts only) ---
    ds = doc_stats or {}
    rep.set_counter("doc_txt_found", ds.get("txt_found", 0))
    rep.set_counter("doc_word_found", ds.get("word_found", 0))
    rep.set_counter("doc_read", ds.get("read", 0))
    rep.set_counter("doc_summarized", ds.get("summarized", 0))
    rep.set_counter("doc_not_summarized", ds.get("not_summarized", 0))
    rep.set_counter("doc_failed", ds.get("failed", 0))
    rep.set_counter("doc_word_skipped_no_reader", ds.get("word_skipped_no_reader", 0))
    rep.set_counter("doc_chunks", ds.get("chunks", 0))
    rep.set_counter("doc_llm_calls_made", ds.get("llm_calls_made", 0))
    rep.set_counter("doc_calls_per_minute", ds.get("calls_per_minute", 0))
    rep.set_counter("doc_nodes_with_context", ds.get("nodes_with_context", 0))

    # --- stages (core stages completed because the report runs only after them) ---
    for label in ("Scan files", "Parse structure", "Enrich summaries",
                  "Assemble graph", "Validate graph", "Write knowledge graph"):
        rep.add_stage(label, "completed")
    rep.add_stage("Write project story", "completed" if story_ok else "errored")
    rep.add_stage("Write run diagnostics", "completed")
    rep.add_stage("Build embedding index", "skipped" if embed_skipped else "completed")
    return rep


def render_run_report(rep: RunReport) -> str:
    """Format the RunReport as plain text from fixed English templates plus the
    collector's integers, extension tokens, stage labels, and outcome codes."""
    c = rep.counters
    generated = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    lines = [
        "RUN REPORT (sanitized for review)",
        f"Generated (UTC): {generated}",
        "Contains only fixed templates, counts, file extensions, stage labels,",
        "and outcome codes - no names, paths, identifiers, or data values.",
        "",
        f"Run completed: {rep.run_completed}.",
        f"Objective met (knowledge graph produced): {rep.objective_met}.",
        "",
        "FILES BY EXTENSION",
        f"  Files in project tree: {rep.files_in_tree}.",
        f"  Files scanned as source candidates: {rep.files_scanned}.",
        f"  Files dropped as unreadable/empty/binary: {rep.files_unreadable}.",
        f"  Files over the size limit: {rep.files_oversize}.",
    ]
    # handled extensions first, then unhandled; each alphabetical
    for token, row in sorted(rep.by_ext.items(), key=lambda kv: (not kv[1].handled, kv[0])):
        if row.handled:
            lines.append(f"  {token:<16} HANDLED:   found {row.found}, parsed {row.parsed}.")
        else:
            lines.append(f"  {token:<16} UNHANDLED: found {row.found}, "
                         f"skipped as unhandled (no parser).")
    lines += [
        "",
        "PARSING",
        f"  Source files parsed: {c.get('source_files_parsed', 0)}.",
        f"  Recognized-grammar files that failed to parse: {c.get('grammar_parse_failures', 0)}.",
        f"  SQL statements/blocks that failed to parse: {c.get('sql_parse_failures', 0)}.",
        f"  SQL statements recovered from shell macro bodies: {c.get('sql_recovered_from_shell', 0)}.",
        f"  Commented-SQL blocks detected: {c.get('sql_comment_blocks_detected', 0)}, "
        f"recovered: {c.get('sql_comment_blocks_recovered', 0)}.",
        f"    rejected by reason: parse_failed {c.get('sql_comment_rejected_parse_failed', 0)}, "
        f"no_dml {c.get('sql_comment_rejected_no_dml', 0)}, "
        f"other {c.get('sql_comment_rejected_other', 0)}.",
        f"  Statements recovered from commented SQL: "
        f"{c.get('sql_statements_recovered_from_comments', 0)}.",
        f"  SQL comment lines (--): {c.get('sql_comment_lines', 0)}; "
        f"SQL lines total: {c.get('sql_lines_total', 0)}.",
        "",
        "SQL PARSE FAILURES BY CATEGORY",
        "  (fixed category codes only; no SQL, identifiers, or error messages)",
        "  Active statements:",
    ]
    for cat in _SQL_FAIL_CATEGORIES:
        lines.append(f"    {cat:<24} {c.get('sqlfail_active_' + cat, 0)}.")
    lines.append("  Commented-SQL recovery (the parse_failed slice, by family):")
    for cat in _SQL_FAIL_CATEGORIES:
        lines.append(f"    {cat:<24} {c.get('sqlfail_commented_' + cat, 0)}.")
    lines += [
        "  File-level gap (recognized .sql files that yielded nothing), by reason:",
        f"    files with no extraction:  {c.get('sqlgap_files_unparsed', 0)}.",
        f"    empty_after_preprocess:    {c.get('sqlgap_empty_after_preprocess', 0)}.",
        f"    all_statements_failed:     {c.get('sqlgap_all_statements_failed', 0)}.",
        f"    parsed_no_entities:        {c.get('sqlgap_parsed_no_entities', 0)}.",
        "",
        "COLUMNS",
        f"  Tables identified: {c.get('tables', 0)}.",
        f"  Columns extracted: {c.get('columns', 0)}.",
        f"  Column-lineage edges built: {c.get('column_lineage_edges', 0)}.",
        f"  Columns without column-lineage (unsupported DML or none): "
        f"{c.get('columns_without_lineage', 0)}.",
        f"  Tables without provenance: {c.get('tables_without_provenance', 0)}.",
        "",
        "REFERENCES",
        f"  Resolved: {c.get('refs_resolved', 0)}.",
        f"  Unresolved (searched but not identified): {c.get('refs_unresolved', 0)}.",
        f"  Skipped: {c.get('refs_skipped', 0)}.",
        f"  Internal function calls (not file references): {c.get('refs_internal_calls', 0)}.",
        "",
        "ENRICHMENT",
        f"  Files summarized by the internal service: {c.get('enriched', 0)}.",
        f"  Files where summarization failed (deterministic fallback used): "
        f"{c.get('enrich_failed', 0)}.",
        "",
        "PROJECT DOCUMENTS",
        "  (sidecar context only — never adds graph nodes/edges; counts only)",
        f"  Prose documents found: .txt {c.get('doc_txt_found', 0)}, "
        f"Word {c.get('doc_word_found', 0)}; read {c.get('doc_read', 0)}, "
        f"failed {c.get('doc_failed', 0)} "
        f"(Word skipped for no reader: {c.get('doc_word_skipped_no_reader', 0)}).",
        f"  Summarized: {c.get('doc_summarized', 0)}; "
        f"not summarized (budget/empty): {c.get('doc_not_summarized', 0)}; "
        f"chunks: {c.get('doc_chunks', 0)}.",
        f"  Internal-service calls made: {c.get('doc_llm_calls_made', 0)} "
        f"(throttled to {c.get('doc_calls_per_minute', 0)} per minute).",
        f"  Nodes that gained document context: {c.get('doc_nodes_with_context', 0)}.",
        "",
        "STAGES",
    ]
    for label, outcome in rep.stages:
        lines.append(f"  {label}: {outcome}.")
    if any(label == "Build embedding index" and outcome == "skipped"
           for label, outcome in rep.stages):
        lines.append("  Note: the embedding stage is skipped (semantic search "
                     "disabled); retrieval is lexical.")
    return "\n".join(lines) + "\n"


def write_run_report(out_path: Path, project_root: str, **kwargs) -> bool:
    """Build, sanitize-check, and write run_report.txt.

    Returns True on write. Defense-in-depth: the by-construction data model
    already excludes project strings; as a cheap secondary catch this refuses to
    write if the project-root path string somehow appears in the rendered text.
    """
    rep = build_run_report(**kwargs)
    text = render_run_report(rep)
    candidates = set()
    try:
        candidates.add(str(Path(project_root)))
        candidates.add(str(Path(project_root).resolve()))
        candidates.add(Path(project_root).as_posix())
    except Exception:  # noqa: BLE001 - path normalization must never break the run
        candidates.add(str(project_root))
    for cand in candidates:
        if cand and cand in text:
            log.error("run report sanitization assertion failed; report not written.")
            return False
    out_path.write_text(text, encoding="utf-8")
    return True
