"""Tests for app/diagnostics.py — structural-health report + the privacy contract.

Self-contained: `python tests/test_diagnostics.py` or via pytest. The key test is
the LEAK test: a graph stuffed with recognizable identifiers must produce a
diagnostics report (JSON + Markdown) that contains NONE of them — only aggregate
counts and DLRD's fixed vocabulary.
"""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app import diagnostics, sqlproc       # noqa: E402
from app.graph import build_graph           # noqa: E402
from app.scanner import ScannedFile         # noqa: E402
from app.parser import FileParse            # noqa: E402
from app.enrich import FileEnrichment       # noqa: E402
from app.config import Settings             # noqa: E402

# Recognizable identifiers planted in the graph; none may appear in the report.
SECRETS = [
    "ZZZ_SECRET_TABLE", "zzz_secret_col", "SRC_X",          # SQL table/column names
    "confidential", "load_secret", "run_secret",            # file paths / names
    "ghost_secret",                                         # a missing-script name
    "ZZZ_SECRET_PROJECT",                                   # project name
]


def _graph_with_secrets():
    sqltext = "INSERT INTO ZZZ_SECRET_TABLE (zzz_secret_col) SELECT s.zzz_secret_col FROM SRC_X s;"
    sql_fp = FileParse("confidential/load_secret.sql", "sql", None, True, [], [], [], {},
                       sql=sqlproc.extract(sqltext))
    sh_fp = FileParse("confidential/run_secret.sh", "shell", "shell", True, [], [],
                      ["ghost_secret.sh"], {})
    scanned = [
        ScannedFile(abs_path="a", rel_path="confidential/load_secret.sql", ext=".sql",
                    language="sql", size_bytes=80),
        ScannedFile(abs_path="b", rel_path="confidential/run_secret.sh", ext=".sh",
                    language="shell", size_bytes=40),
    ]
    enrich = {
        "confidential/load_secret.sql": FileEnrichment(summary="loads ZZZ_SECRET_TABLE",
                                                       layer="Data", complexity="simple",
                                                       tags=["sql"], members={}, llm_ok=True),
        "confidential/run_secret.sh": FileEnrichment(summary="runs confidential load_secret",
                                                     layer="Utility", complexity="simple",
                                                     tags=["shell"], members={}, llm_ok=False),
    }
    refc: dict = {}
    g = build_graph(project_name="ZZZ_SECRET_PROJECT", project_root=".", scanned=scanned,
                    parses=[sql_fp, sh_fp], enrichments=enrich, settings=Settings(),
                    description="confidential project", frameworks=[], diagnostics=refc)
    return g, refc, [sql_fp, sh_fp], scanned, enrich


def _diag(g, refc, parses, scanned, enrich):
    return diagnostics.build_diagnostics(
        g,
        scan_stats={"scanned": len(scanned), "skipped": 0, "oversize": 0,
                    "by_language": diagnostics.language_counts(scanned)},
        parse_stats=diagnostics.parse_outcomes(parses),
        enrich_stats=diagnostics.enrich_outcomes(enrich),
        ref_stats=refc.get("references", {}),
        sql_stats=diagnostics.sql_outcomes(parses),
    )


def test_diagnostics_no_identifier_leak():
    g, refc, parses, scanned, enrich = _graph_with_secrets()
    diag = _diag(g, refc, parses, scanned, enrich)
    md = diagnostics.render_diagnostics_md(diag)
    blob = json.dumps(diag, ensure_ascii=False) + "\n" + md
    for s in SECRETS:
        assert s not in blob, f"identifier leaked into diagnostics: {s}"
    # ...while still carrying real structural signal (so it's not empty-by-cheating):
    assert diag["metrics"]["nodes"]["total"] > 0
    assert diag["metrics"]["nodes"]["by_type"].get("table", 0) >= 2
    assert diag["metrics"]["nodes"]["missing_by_kind"].get("script", 0) == 1


def test_diagnostics_field_presence():
    g, refc, parses, scanned, enrich = _graph_with_secrets()
    diag = _diag(g, refc, parses, scanned, enrich)
    assert diag["version"] and diag["generatedAt"]
    m = diag["metrics"]
    for section in ("files", "nodes", "edges", "references", "sql", "enrichment", "shape", "lineage"):
        assert section in m, f"missing section: {section}"
    assert "by_type" in m["nodes"] and "missing_by_kind" in m["nodes"]
    assert set(m["references"]) == {"resolved", "missing", "skipped", "function_calls"}
    assert m["references"]["missing"] == 1                       # ghost_secret.sh
    assert {"tables", "columns", "macro_statements_recovered", "parse_failures"} <= set(m["sql"])
    assert {"enriched", "distilled", "truncated", "failed"} <= set(m["enrichment"])
    assert m["enrichment"]["failed"] == 1                        # the shell file's llm_ok=False
    assert {"components", "orphans", "nodes_per_layer", "duplicate_node_ids"} <= set(m["shape"])
    assert isinstance(diag["checks"], list) and diag["checks"]
    md = diagnostics.render_diagnostics_md(diag)
    assert "DLRD run diagnostics" in md and "Health checks:" in md


def test_diagnostics_health_checks_flag_warnings():
    # an empty graph: the duplicate-id / parse checks are OK, but ratios are clean
    diag = diagnostics.build_diagnostics({"nodes": [], "edges": [], "layers": []})
    labels = {c["label"]: c["ok"] for c in diag["checks"]}
    assert all(isinstance(v, bool) for v in labels.values())
    assert any("duplicate node ids" in lbl for lbl in labels)


# A macro-load block: a genuine commented INSERT...SELECT the recovery accepts,
# a qualified INSERT with no SELECT it rejects (incomplete_statement), and a prose
# line it rejects (no_dml). Every identifier is recognizable, to prove none leaks.
_RECOVERY_SQL = (
    "exec $DB_x.ZZZSECRETPROC();\n"
    "-- Statement 1: load the entity table\n"
    "-- insert into $DB_z.ZZZ_TARGET_TBL (zzz_col_a, zzz_col_b)\n"
    "-- select s.zzz_col_a, s.zzz_col_b from $DB_y.ZZZ_SOURCE_TBL s\n"
    ";\n"
    "exec $DB_w.ZZZOTHERPROC();\n"
    "-- insert into $DB_q.ZZZ_AUDIT_LOG\n"                 # qualified target, no SELECT
    ";\n"
    "exec $DB_v.ZZZTHIRDPROC();\n"
    "-- this step refreshes the cache and notifies downstream teams\n"   # prose, non-DML
    ";\n"
)
_RECOVERY_SECRETS = [
    "ZZZ_TARGET_TBL", "zzz_col_a", "zzz_col_b", "ZZZ_SOURCE_TBL", "ZZZ_AUDIT_LOG",
    "ZZZSECRETPROC", "ZZZOTHERPROC", "ZZZTHIRDPROC", "$DB_z", "macro_load",
]


def _recovery_diag():
    rel = "confidential/macro_load.sql"
    fp = FileParse(rel, "sql", None, True, [], [], [], {}, sql=sqlproc.extract(_RECOVERY_SQL))
    scanned = [ScannedFile(abs_path="a", rel_path=rel, ext=".sql", language="sql",
                           size_bytes=len(_RECOVERY_SQL))]
    enrich = {rel: FileEnrichment(summary="loads entities", layer="Data", complexity="simple",
                                  tags=["sql"], members={}, llm_ok=True)}
    refc: dict = {}
    g = build_graph(project_name="ZZZSECRET_PROJECT", project_root=".", scanned=scanned,
                    parses=[fp], enrichments=enrich, settings=Settings(),
                    description="confidential", frameworks=[], diagnostics=refc)
    diag = diagnostics.build_diagnostics(
        g,
        scan_stats={"scanned": 1, "skipped": 0, "oversize": 0,
                    "by_language": diagnostics.language_counts(scanned)},
        parse_stats=diagnostics.parse_outcomes([fp]),
        enrich_stats=diagnostics.enrich_outcomes(enrich),
        ref_stats=refc.get("references", {}),
        sql_stats=diagnostics.sql_outcomes([fp]),
    )
    return diag


def test_diagnostics_sql_comment_and_recovery_metrics():
    diag = _recovery_diag()
    sql = diag["metrics"]["sql"]
    # METRIC 1 — full-line `--` comments counted on the raw block
    assert sql["comment_lines"] == 5
    assert sql["total_lines"] == len(_RECOVERY_SQL.splitlines())          # 11
    # METRIC 2 — commented-SQL recovery outcomes (relaxed gate: parse-sanity only)
    rec = sql["recovery"]
    assert rec["detected"] == 3
    # both INSERTs parse as real DML -> recovered (the strict incomplete_statement
    # / not_schema_qualified gate is gone; the INSERT-without-SELECT now recovers)
    assert rec["recovered"] == 2 and rec["statements"] == 2
    assert rec["rejected"]["no_dml"] == 1                                 # the prose line
    assert rec["rejected"]["parse_failed"] == 0 and rec["rejected"]["other"] == 0
    # the reject buckets are now exactly {parse_failed, no_dml, other}
    assert set(rec["rejected"]) == {"parse_failed", "no_dml", "other"}
    # the partition holds: every detected candidate is recovered or rejected once
    assert rec["detected"] == rec["recovered"] + sum(rec["rejected"].values())


def test_diagnostics_recovery_no_identifier_leak():
    diag = _recovery_diag()
    md = diagnostics.render_diagnostics_md(diag)
    blob = json.dumps(diag, ensure_ascii=False) + "\n" + md
    for s in _RECOVERY_SECRETS:
        assert s not in blob, f"identifier leaked into diagnostics: {s}"
    # the new SQL lines are present (counts + fixed labels only)
    assert "SQL comment lines (--):" in md
    assert "Commented-SQL recovery:" in md
    assert "rejected by reason: parse_failed" in md
    assert "Statements recovered from comments:" in md


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {t.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"ERROR {t.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run() else 0)
