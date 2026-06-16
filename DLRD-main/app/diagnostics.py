"""Deterministic run diagnostics — a structural health report safe to share.

PRIVACY CONTRACT (R3). The report contains ONLY: aggregate integer counts,
ratios/percentages, DLRD's fixed taxonomy (node types, edge types, language
labels, missing-node kinds, error CATEGORIES, layer names — all DLRD's own
vocabulary), a schema version, and a timestamp. It NEVER contains a file / table
/ column / variable name, a file path, a data value, SQL text, a node summary /
description, or a raw error message / stack trace (errors are categorized).

This is strictly LESS than what already (post-redaction) reaches the internal
LLM — there, names + descriptions ride along; here, names are removed entirely.
Enforcement is in code: ``build_diagnostics`` reads names from the graph ONLY to
COUNT them and never copies one into the returned dict, and ``render_diagnostics_md``
interpolates only numbers and fixed category strings.

No LLM, no egress: derived from in-memory data, written to a local file.
"""

from __future__ import annotations

import collections
from datetime import datetime, timezone

from .parser import _SHELL_NON_SCRIPT

DIAG_VERSION = "1.0.0"
_OVERSIZE_THRESHOLD = 2000

# Health-check thresholds (sane defaults).
_MISSING_RATIO_WARN = 0.25          # > 25% of references unresolved
_NO_PROVENANCE_FRACTION_WARN = 0.5  # > 50% of tables with no provenance


# ----------------------------------------------------------- pipeline tallies
# These read in-memory analysis objects (scanned files / parses / enrichments)
# and return COUNT-ONLY dicts; no name/path/value is ever read into the result.
def language_counts(scanned) -> dict:
    """Scanned-file count per language label (DLRD vocabulary)."""
    return dict(collections.Counter(str(getattr(sf, "language", "?")) for sf in scanned))


def parse_outcomes(parses) -> dict:
    """parsed count + failures bucketed by CATEGORY. A recognised-grammar file
    (grammar_key set) that did not parse is a tree-sitter-error; unsupported types
    (grammar_key None) and empty SQL are NOT failures."""
    parsed = 0
    ts_failed = 0
    for p in parses:
        if getattr(p, "parse_ok", False):
            parsed += 1
        elif getattr(p, "grammar_key", None) is not None and getattr(p, "language", None) not in ("sql", "joblist"):
            ts_failed += 1
    return {"parsed": parsed, "failed": ({"tree-sitter-error": ts_failed} if ts_failed else {})}


def enrich_outcomes(enrichments) -> dict:
    """enriched / distilled / truncated / failed counts from the enrichment map."""
    vals = list(enrichments.values())
    return {
        "enriched": sum(1 for x in vals if getattr(x, "llm_ok", False)),
        "failed": sum(1 for x in vals if not getattr(x, "llm_ok", False)),
        "distilled": sum(1 for x in vals if getattr(x, "distilled", False)),
        "truncated": sum(1 for x in vals if getattr(x, "truncated", False)),
    }


def sql_outcomes(parses) -> dict:
    """SQL-block tallies from each file's SqlResult.stats (counts only): macro-body
    statements recovered, statement-level parse failures, full-line ``--`` comment
    counts, and the commented-SQL recovery outcome buckets."""
    macro = 0
    parse_failures = 0
    comment_lines = 0
    total_lines = 0
    rec: collections.Counter = collections.Counter()
    for p in parses:
        s = getattr(p, "sql", None)
        if s is None:
            continue
        stats = getattr(s, "stats", None) or {}
        for key, val in stats.get("by_type", {}).items():
            if key.startswith("recovered_") and key != "recovered_skipped":
                macro += int(val)
            elif key == "parse_error":
                parse_failures += int(val)
        comment_lines += int(stats.get("comment_lines", 0))
        total_lines += int(stats.get("total_lines", 0))
        rec.update({str(k): int(v) for k, v in (stats.get("recovery") or {}).items()})
    return {
        "macro_recovered": macro,
        "parse_failures": parse_failures,
        "comment_lines": comment_lines,
        "total_lines": total_lines,
        "recovery": {
            "detected": rec.get("detected", 0),
            "recovered": rec.get("recovered", 0),
            "statements": rec.get("statements", 0),
            "rejected": {  # relaxed gate -> parse_failed / no_dml / other only
                "parse_failed": rec.get("reject_parse_failed", 0),
                "no_dml": rec.get("reject_no_dml", 0),
                "other": rec.get("reject_other", 0),
            },
        },
    }


def _components(id_set: set, adj: dict) -> tuple[int, int]:
    """(component count, largest component size) over the undirected node graph."""
    seen: set = set()
    count = 0
    largest = 0
    for start in id_set:
        if start in seen:
            continue
        count += 1
        size = 0
        stack = [start]
        seen.add(start)
        while stack:
            cur = stack.pop()
            size += 1
            for nb in adj.get(cur, ()):  # neighbours
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        if size > largest:
            largest = size
    return count, largest


def build_diagnostics(
    graph: dict,
    *,
    scan_stats: dict | None = None,
    parse_stats: dict | None = None,
    enrich_stats: dict | None = None,
    ref_stats: dict | None = None,
    sql_stats: dict | None = None,
) -> dict:
    """Compute structural-health metrics + health checks from the FINAL graph plus
    the threaded tally-only counters. Names in the graph are read solely to COUNT;
    none is ever placed into the returned dict (see the module privacy contract)."""
    scan_stats = scan_stats or {}
    parse_stats = parse_stats or {}
    enrich_stats = enrich_stats or {}
    ref_stats = ref_stats or {}
    sql_stats = sql_stats or {}

    nodes = graph.get("nodes", []) or []
    edges = graph.get("edges", []) or []
    layers = graph.get("layers", []) or []

    # --- ids + duplicate detection (counts only) ---
    id_set: set = set()
    dup_ids = 0
    for n in nodes:
        nid = n.get("id")
        if not nid:
            continue
        if nid in id_set:
            dup_ids += 1
        else:
            id_set.add(nid)

    # --- type histograms (DLRD vocabulary, never names) ---
    type_hist = collections.Counter(str(n.get("type", "?")) for n in nodes)
    edge_hist = collections.Counter(str(e.get("type", "?")) for e in edges)

    # --- missing nodes by kind + shell-builtin collision count ---
    missing_by_kind: collections.Counter = collections.Counter()
    missing_builtin_collision = 0
    for n in nodes:
        if n.get("type") != "missing":
            continue
        tags = n.get("tags") or []
        kind = "sql-file" if "sql-file" in tags else ("script" if "script" in tags else "unknown")
        missing_by_kind[kind] += 1
        # COUNT only: does this missing name collide with a shell builtin/keyword?
        name = str(n.get("name") or "")
        if name.rsplit("/", 1)[-1].lower() in _SHELL_NON_SCRIPT:
            missing_builtin_collision += 1

    # --- adjacency / degree for components + orphans (ids stay internal) ---
    adj: dict = collections.defaultdict(list)
    degree: collections.Counter = collections.Counter()
    by_id_type = {n["id"]: n.get("type") for n in nodes if n.get("id")}
    reads_from_total = 0
    writes_to_total = 0
    has_provenance: set = set()
    col_lineage_nodes: set = set()
    table_ids = {nid for nid, ty in by_id_type.items() if ty == "table"}
    column_ids = {nid for nid, ty in by_id_type.items() if ty == "column"}
    for e in edges:
        s, t, ty = e.get("source"), e.get("target"), e.get("type")
        if s in id_set and t in id_set:
            adj[s].append(t)
            adj[t].append(s)
            degree[s] += 1
            degree[t] += 1
        if ty == "writes_to":
            writes_to_total += 1
            if t in table_ids:
                has_provenance.add(t)          # a file writes/produces this table
        elif ty == "reads_from":
            reads_from_total += 1
            st, tt = by_id_type.get(s), by_id_type.get(t)
            if st == "table" and tt == "table":
                has_provenance.add(s)          # s is populated from t
            elif st == "column" and tt == "column":
                col_lineage_nodes.add(s)
                col_lineage_nodes.add(t)
    orphans = sum(1 for nid in id_set if degree[nid] == 0)
    comp_count, comp_largest = _components(id_set, adj)
    tables_without_provenance = len(table_ids - has_provenance)
    columns_without_lineage = len(column_ids - col_lineage_nodes)

    # --- per-layer counts (layer names are DLRD's architectural labels) ---
    nodes_per_layer = {str(L.get("name", L.get("id", "?"))): len(L.get("nodeIds", []) or []) for L in layers}
    empty_layers = sum(1 for c in nodes_per_layer.values() if c == 0)

    # --- references / missing ratio ---
    resolved = int(ref_stats.get("resolved", 0))
    missing_refs = int(ref_stats.get("missing", 0))
    skipped_refs = int(ref_stats.get("skipped", 0))
    function_refs = int(ref_stats.get("function_calls", 0))  # bare refs to project functions
    # function calls are internal (not file references), so they stay out of the
    # resolved/missing/skipped denominator that the missing-ratio check uses.
    total_refs = resolved + missing_refs + skipped_refs
    missing_ratio = (missing_refs / total_refs) if total_refs else 0.0

    parse_failed = {str(k): int(v) for k, v in (parse_stats.get("failed") or {}).items()}
    parse_failed_total = sum(parse_failed.values())
    enrich_failed = int(enrich_stats.get("failed", 0))

    # commented-SQL recovery outcome buckets (tally-only; fixed reason labels)
    sql_rec = sql_stats.get("recovery") or {}
    sql_rec_rej = sql_rec.get("rejected") or {}
    sql_recovery = {
        "detected": int(sql_rec.get("detected", 0)),
        "recovered": int(sql_rec.get("recovered", 0)),
        "statements": int(sql_rec.get("statements", 0)),
        "rejected": {  # relaxed gate -> parse_failed / no_dml / other only
            "parse_failed": int(sql_rec_rej.get("parse_failed", 0)),
            "no_dml": int(sql_rec_rej.get("no_dml", 0)),
            "other": int(sql_rec_rej.get("other", 0)),
        },
    }

    metrics = {
        "files": {
            "scanned": int(scan_stats.get("scanned", type_hist.get("file", 0))),
            "by_language": {str(k): int(v) for k, v in (scan_stats.get("by_language") or {}).items()},
            "skipped": int(scan_stats.get("skipped", 0)),
            "oversize": int(scan_stats.get("oversize", 0)),
            "parsed": int(parse_stats.get("parsed", 0)),
            "parse_failed_by_category": parse_failed,
        },
        "nodes": {
            "total": len(nodes),
            "by_type": dict(type_hist),
            "missing_by_kind": dict(missing_by_kind),
            "oversized": len(nodes) > _OVERSIZE_THRESHOLD,
        },
        "edges": {"total": len(edges), "by_type": dict(edge_hist)},
        "references": {"resolved": resolved, "missing": missing_refs, "skipped": skipped_refs,
                       "function_calls": function_refs},
        "sql": {
            "tables": type_hist.get("table", 0),
            "columns": type_hist.get("column", 0),
            "macro_statements_recovered": int(sql_stats.get("macro_recovered", 0)),
            "parse_failures": int(sql_stats.get("parse_failures", 0)),
            "total_lines": int(sql_stats.get("total_lines", 0)),
            "comment_lines": int(sql_stats.get("comment_lines", 0)),
            "recovery": sql_recovery,
        },
        "enrichment": {
            "enriched": int(enrich_stats.get("enriched", 0)),
            "distilled": int(enrich_stats.get("distilled", 0)),
            "truncated": int(enrich_stats.get("truncated", 0)),
            "failed": enrich_failed,
        },
        "shape": {
            "components": comp_count,
            "largest_component": comp_largest,
            "orphans": orphans,
            "nodes_per_layer": nodes_per_layer,
            "duplicate_node_ids": dup_ids,
        },
        "lineage": {
            "tables_without_provenance": tables_without_provenance,
            "columns_without_lineage": columns_without_lineage,
            "reads_from": reads_from_total,
            "writes_to": writes_to_total,
        },
    }

    # --- health checks (each label is fixed text + numbers, never a name) ---
    checks: list[dict] = []

    def add(ok: bool, label: str) -> None:
        checks.append({"ok": bool(ok), "label": label})

    table_count = type_hist.get("table", 0)
    if table_count > 0:
        add(reads_from_total > 0, f"reads_from present ({reads_from_total})")
    add(missing_builtin_collision == 0, f"missing matching shell builtins: {missing_builtin_collision}")
    if total_refs > 0:
        add(missing_ratio <= _MISSING_RATIO_WARN, f"missing ratio {missing_ratio * 100:.1f}%")
    add(parse_failed_total == 0, f"parse failures: {parse_failed_total}")
    add(int(metrics["sql"]["parse_failures"]) == 0, f"SQL parse failures: {metrics['sql']['parse_failures']}")
    add(enrich_failed == 0, f"enrichment failures: {enrich_failed}")
    if table_count > 0:
        frac = tables_without_provenance / table_count
        add(frac <= _NO_PROVENANCE_FRACTION_WARN, f"{tables_without_provenance} tables without provenance")
    add(empty_layers == 0, f"layers with 0 nodes: {empty_layers}")
    add(dup_ids == 0, f"duplicate node ids: {dup_ids}")

    return {
        "version": DIAG_VERSION,
        "generatedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "metrics": metrics,
        "checks": checks,
    }


def _hist_line(hist: dict) -> str:
    """`type count, type count` — sorted by count desc then name. Vocabulary only."""
    items = sorted(hist.items(), key=lambda kv: (-kv[1], str(kv[0])))
    return ", ".join(f"{k} {v:,}" for k, v in items) or "(none)"


def render_diagnostics_md(diag: dict) -> str:
    """Deterministic templated prose. Interpolates ONLY numbers + fixed category
    strings + layer names (DLRD's own vocabulary) — never a node/file name."""
    m = diag.get("metrics", {})
    f = m.get("files", {})
    n = m.get("nodes", {})
    e = m.get("edges", {})
    r = m.get("references", {})
    sql = m.get("sql", {})
    enr = m.get("enrichment", {})
    shape = m.get("shape", {})
    lin = m.get("lineage", {})

    by_lang = f.get("by_language", {})
    lang_bits = ", ".join(f"{v:,} {k}" for k, v in sorted(by_lang.items(), key=lambda kv: (-kv[1], kv[0])))
    failed = f.get("parse_failed_by_category", {})
    failed_bits = ", ".join(f"{v} {k}" for k, v in sorted(failed.items())) or "0"
    layer_bits = " / ".join(f"{name} {cnt:,}" for name, cnt in shape.get("nodes_per_layer", {}).items()) or "(none)"
    sql_rec = sql.get("recovery", {}) or {}
    sql_rej = sql_rec.get("rejected", {}) or {}
    sql_rej_total = sum(int(v) for v in sql_rej.values())

    lines = [
        f"# DLRD run diagnostics  (v{diag.get('version', '?')} · {diag.get('generatedAt', '')})",
        "",
        f"Files: {f.get('scanned', 0):,} scanned"
        + (f" ({lang_bits})" if lang_bits else "")
        + f"; {f.get('skipped', 0):,} skipped; {f.get('oversize', 0):,} over size limit; "
        + f"{f.get('parsed', 0):,} parsed; {sum(failed.values())} failed ({failed_bits}).",
        f"Graph: {n.get('total', 0):,} nodes, {e.get('total', 0):,} edges. "
        + f"Oversized (>{_OVERSIZE_THRESHOLD}): {'yes' if n.get('oversized') else 'no'}.",
        f"Nodes by type: {_hist_line(n.get('by_type', {}))}.",
        f"Missing by kind: {_hist_line(n.get('missing_by_kind', {}))}.",
        f"Edges by type: {_hist_line(e.get('by_type', {}))}.",
        f"References: {r.get('resolved', 0):,} resolved, {r.get('missing', 0):,} missing, "
        + f"{r.get('skipped', 0):,} skipped, {r.get('function_calls', 0):,} internal function calls.",
        f"SQL: {sql.get('tables', 0):,} tables, {sql.get('columns', 0):,} columns, "
        + f"{sql.get('macro_statements_recovered', 0):,} macro statements recovered, "
        + f"{sql.get('parse_failures', 0):,} parse failures.",
        f"SQL lines extracted: {sql.get('total_lines', 0):,}   |   "
        + f"SQL comment lines (--): {sql.get('comment_lines', 0):,}.",
        f"Commented-SQL recovery: {sql_rec.get('detected', 0):,} detected, "
        + f"{sql_rec.get('recovered', 0):,} recovered, {sql_rej_total:,} rejected.",
        f"  rejected by reason: parse_failed {sql_rej.get('parse_failed', 0):,}, "
        + f"no_dml {sql_rej.get('no_dml', 0):,}, other {sql_rej.get('other', 0):,}.",
        f"Statements recovered from comments: {sql_rec.get('statements', 0):,}.",
        f"Enrichment: {enr.get('enriched', 0):,} enriched, {enr.get('distilled', 0):,} distilled, "
        + f"{enr.get('truncated', 0):,} truncated, {enr.get('failed', 0):,} failed.",
        f"Shape: {shape.get('components', 0):,} components (largest {shape.get('largest_component', 0):,}), "
        + f"{shape.get('orphans', 0):,} orphans; layers [{layer_bits}].",
        f"Lineage: {lin.get('tables_without_provenance', 0):,} tables w/o provenance, "
        + f"{lin.get('columns_without_lineage', 0):,} columns w/o lineage "
        + f"(reads_from {lin.get('reads_from', 0):,}, writes_to {lin.get('writes_to', 0):,}).",
        "",
        "Health checks:",
    ]
    for c in diag.get("checks", []):
        lines.append(f"  {'OK ' if c.get('ok') else 'WARN'} {c.get('label', '')}")
    return "\n".join(lines) + "\n"
