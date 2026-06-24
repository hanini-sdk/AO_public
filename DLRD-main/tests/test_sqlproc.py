"""Regression tests for app/sqlproc.py — Teradata table-level lineage.

Self-contained: run directly with `python tests/test_sqlproc.py` (no pytest
needed) or via `pytest tests/`. Statements are CLEAN synthetic SQL modelled on
the real anonymized samples (which are intentionally garbled and must not be
used to measure recall). Covers the Phase-2 statement set plus the temporal
cases S1-S5 and the real derived-table / inline-qualifier VALIDTIME forms, and
asserts the phantom-table guard (no temporal keyword ever becomes a "table").
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app import sqlproc  # noqa: E402

_PHANTOM = {"nonsequenced", "sequenced", "current", "validtime", "period"}


def _has_lineage(r, target, expected_sources):
    return any(
        tgt == target and set(expected_sources).issubset(set(srcs))
        for tgt, srcs in r.lineage
    )


def _no_phantom(r):
    names = [n for tgt, srcs in r.lineage for n in (tgt, *srcs)]
    names += [o.name for o in r.objects] + list(r.table_refs)
    return all(n.strip().lower() not in _PHANTOM for n in names)


# --- Phase-2 regression: the non-temporal statement set still works ----------
def test_basic_statement_set():
    sql = """
    CREATE MULTISET TABLE $ws.D_DIM AS (SELECT id, v FROM $ws.S_RAW) WITH DATA;
    INSERT INTO $ws.D_FACT SELECT a.id, b.v FROM $ws.D_DIM a JOIN $ws.L_LK b ON a.id=b.id;
    MERGE INTO $ws.D_FACT t USING $ws.D_DELTA s ON t.id=s.id WHEN MATCHED THEN UPDATE SET t.v=s.v;
    UPDATE $ws.D_FACT FROM $ws.D_OVR o SET v=o.v WHERE D_FACT.id=o.id;
    CREATE VIEW $ws.V_REP AS SELECT * FROM $ws.D_FACT;
    REPLACE PROCEDURE $ws.load_proc() BEGIN END;
    EXEC $ws.load_proc;
    COLLECT STATISTICS ON $ws.D_FACT COLUMN (id);
    DELETE $ws.D_STAGE;
    """
    r = sqlproc.extract(sql)
    assert _has_lineage(r, "$ws.D_DIM", ["$ws.S_RAW"])           # CTAS
    assert _has_lineage(r, "$ws.D_FACT", ["$ws.D_DIM", "$ws.L_LK"])  # INSERT..JOIN
    assert _has_lineage(r, "$ws.D_FACT", ["$ws.D_DELTA"])        # MERGE
    assert _has_lineage(r, "$ws.D_FACT", ["$ws.D_OVR"])          # UPDATE..FROM
    assert _has_lineage(r, "$ws.V_REP", ["$ws.D_FACT"])          # CREATE VIEW
    assert any(o.kind == "view" and o.name == "$ws.V_REP" for o in r.objects)
    assert any(o.kind == "procedure" and o.name == "$ws.load_proc" for o in r.objects)
    assert "$ws.load_proc" in r.proc_calls
    # COLLECT STATS + DELETE carry no lineage (D_STAGE has no edge)
    assert not any(tgt == "$ws.D_STAGE" for tgt, _ in r.lineage)
    assert _no_phantom(r)


# --- Temporal cases S1-S5 (the proposed-fix targets) -------------------------
def test_S1_standalone_temporal_select_parses_no_lineage():
    sql = ("NONSEQUENCED VALIDTIME SELECT t1.k, t2.v FROM db.t1 t1 JOIN db.t2 t2 "
           "ON t1.k=t2.k AND PERIOD(t2.vs, t2.ve) CONTAINS (t1.dt);")
    r = sqlproc.extract(sql)
    assert r.lineage == []          # bare SELECT has no write target — by design
    assert _no_phantom(r)


def test_S2_temporal_prefixed_insert():
    sql = ("NONSEQUENCED VALIDTIME INSERT INTO db.target "
           "SELECT t1.k, t2.v FROM db.t1 t1 JOIN db.t2 t2 "
           "ON t1.k=t2.k AND PERIOD(t2.vs, t2.ve) CONTAINS (t1.dt);")
    r = sqlproc.extract(sql)
    assert _has_lineage(r, "db.target", ["db.t1", "db.t2"])
    assert _no_phantom(r)


def test_S3_derived_table_wrap_no_phantom():
    sql = ("INSERT INTO db.target SELECT x.k, x.v FROM ( "
           "NONSEQUENCED VALIDTIME SELECT t1.k AS k, t2.v AS v FROM db.t1 t1 JOIN db.t2 t2 "
           "ON t1.k=t2.k AND PERIOD(t2.vs, t2.ve) CONTAINS (t1.dt) ) x;")
    r = sqlproc.extract(sql)
    assert _has_lineage(r, "db.target", ["db.t1", "db.t2"])
    assert _no_phantom(r)          # the old 'NONSEQUENCED'-as-table edge is gone


def test_S4_sequenced_validtime():
    sql = ("SEQUENCED VALIDTIME INSERT INTO db.target "
           "SELECT t1.k FROM db.t1 t1 JOIN db.t2 t2 ON t1.k=t2.k;")
    r = sqlproc.extract(sql)
    assert _has_lineage(r, "db.target", ["db.t1", "db.t2"])
    assert _no_phantom(r)


def test_S5_current_validtime():
    sql = ("CURRENT VALIDTIME INSERT INTO db.target "
           "SELECT t1.k FROM db.t1 t1 JOIN db.t2 t2 ON t1.k=t2.k;")
    r = sqlproc.extract(sql)
    assert _has_lineage(r, "db.target", ["db.t1", "db.t2"])
    assert _no_phantom(r)


# --- Real-world VALIDTIME forms seen in the samples --------------------------
def test_inline_table_qualifier_validtime():
    # `FROM <table> nonsequenced validtime AS alias` (as in s06.sql)
    sql = ("INSERT INTO db.target SELECT a.k FROM db.t1 a "
           "JOIN db.t2 nonsequenced validtime b ON a.k=b.k;")
    r = sqlproc.extract(sql)
    assert _has_lineage(r, "db.target", ["db.t1", "db.t2"])
    assert _no_phantom(r)


def test_derived_union_with_period_predicates():
    # Modelled on test_3 copy.sql: INSERT .. SELECT FROM (NONSEQUENCED VALIDTIME
    # SELECT .. JOIN .. ON .. AND PERIOD(..) CONTAINS (..)) UNION ALL (...).
    sql = ("INSERT INTO $ws.J_OUT (a, b) "
           "SELECT v.a, v.b FROM ( "
           "NONSEQUENCED VALIDTIME SELECT b.a AS a, z.b AS b "
           "FROM $ws.D_BASE b JOIN $core.X_REF z "
           "ON b.k=z.k AND PERIOD(z.vs, z.ve) CONTAINS (b.dt) ) AS v;")
    r = sqlproc.extract(sql)
    assert _has_lineage(r, "$ws.J_OUT", ["$ws.D_BASE", "$core.X_REF"])
    assert _no_phantom(r)


def test_phantom_guard_on_nonneutralized_nested_period():
    # Nested-paren PERIOD operand is NOT neutralized (documented caveat); the
    # phantom guard must still prevent any corrupt edge (skip, never corrupt).
    sql = ("NONSEQUENCED VALIDTIME INSERT INTO db.target "
           "SELECT t1.k FROM db.t1 t1 JOIN db.t2 t2 "
           "ON t1.k=t2.k AND PERIOD(CAST(t2.vs AS DATE), t2.ve) CONTAINS (t1.dt);")
    r = sqlproc.extract(sql)
    assert _no_phantom(r)  # may or may not yield lineage, but never a phantom edge


# --- Phase A: per-statement read / write / purge operations -----------------
def test_opsA_purge_delete_truncate_variants():
    for sql in ("DELETE db.t;", "DELETE db.t ALL;", "DELETE FROM db.t;",
                "TRUNCATE TABLE db.t;", "TRUNCATE db.t;"):
        r = sqlproc.extract(sql)
        assert "purge" in r.table_ops.get("db.t", set()), sql


def test_opsA_readonly_select_is_read():
    r = sqlproc.extract("SELECT a FROM db.r1 JOIN db.r2 ON r1.k=r2.k;")
    assert r.table_ops.get("db.r1") == ["read"]
    assert r.table_ops.get("db.r2") == ["read"]
    assert r.lineage == []          # a bare SELECT has no write target


def test_opsA_delete_with_subquery_reads():
    r = sqlproc.extract("DELETE db.t WHERE k IN (SELECT k FROM db.src);")
    assert "purge" in r.table_ops.get("db.t", set())
    assert "read" in r.table_ops.get("db.src", set())


def test_opsA_aggregation_purge_then_load():
    # one file that purges then reloads the same table -> ordered purge, write
    r = sqlproc.extract("DELETE db.fact ALL; INSERT INTO db.fact SELECT x.k FROM db.src x;")
    assert r.table_ops.get("db.fact") == ["purge", "write"]
    assert r.table_ops.get("db.src") == ["read"]


def test_opsA_write_and_read_roles():
    r = sqlproc.extract("INSERT INTO db.tgt SELECT a.k FROM db.s1 a JOIN db.s2 b ON a.k=b.k;")
    assert r.table_ops.get("db.tgt") == ["write"]
    assert r.table_ops.get("db.s1") == ["read"]
    assert r.table_ops.get("db.s2") == ["read"]


# --- Phase B: in-house '#' directive handling -------------------------------
def test_opsB_hash_insert_clean_body():
    r = sqlproc.extract("#insert into $ws.TGT select a.k from $ws.SRC a;")
    assert r.table_ops.get("$ws.TGT") == ["write"]
    assert r.table_ops.get("$ws.SRC") == ["read"]
    assert _has_lineage(r, "$ws.TGT", ["$ws.SRC"])


def test_opsB_hash_insert_garbled_body_recovers_target():
    # body is intentionally unparsable; the #insert target must still be a write
    r = sqlproc.extract("#insert into $ws.TGT mclmoshw )( from where garbage;")
    assert "write" in r.table_ops.get("$ws.TGT", set())


def test_opsB_primary_index_stripped():
    # `#primary index(...)` is DDL metadata: dropped, never a table/phantom
    r = sqlproc.extract(
        "create multiset table $ws.D as (select id from $ws.S) with data #primary index(id);"
    )
    assert _has_lineage(r, "$ws.D", ["$ws.S"])
    assert not any(t in ("id", "primary", "index") for t in r.table_refs)


def test_opsB_inline_hash_ident_not_comment():
    # a trailing inline #IDENT column alias must not swallow the rest as a comment
    r = sqlproc.extract("insert into $ws.TGT select a.k #AWK_DNW_EP from $ws.SRC a;")
    assert r.table_ops.get("$ws.TGT") == ["write"]
    assert "$ws.SRC" in r.table_refs


# --- Part 2: ordered operation sequences (execution order) -------------------
def test_ops_order_purge_write_read():
    # the prompt's reference case: DELETE -> INSERT -> SELECT a table
    r = sqlproc.extract(
        "DELETE db.t; INSERT INTO db.t SELECT k FROM db.src; SELECT k FROM db.t;"
    )
    assert r.table_ops.get("db.t") == ["purge", "write", "read"]
    assert r.table_ops.get("db.src") == ["read"]


def test_ops_order_self_reference_read_before_write():
    # INSERT INTO T SELECT ... FROM T -> within one statement, read precedes write
    r = sqlproc.extract("INSERT INTO db.t SELECT a.k FROM db.t a;")
    assert r.table_ops.get("db.t") == ["read", "write"]


def test_ops_order_consecutive_collapse_nonconsecutive_preserved():
    # consecutive duplicate ops collapse...
    r1 = sqlproc.extract(
        "INSERT INTO db.t SELECT k FROM db.s1; INSERT INTO db.t SELECT k FROM db.s2;"
    )
    assert r1.table_ops.get("db.t") == ["write"]
    # ...but a non-consecutive transition (write, read, write) is preserved
    r2 = sqlproc.extract(
        "INSERT INTO db.t SELECT k FROM db.s1; "
        "SELECT k FROM db.t; "
        "INSERT INTO db.t SELECT k FROM db.s2;"
    )
    assert r2.table_ops.get("db.t") == ["write", "read", "write"]


# --- C1/C2: used columns per table + column-to-column lineage ----------------
def _cols(r, table):
    return sorted(r.used_columns.get(table, set()))


def _has_col_lineage(r, tgt_table, tgt_col, src_table, src_col):
    return (tgt_table, tgt_col, src_table, src_col) in r.column_lineage


def test_used_columns_insert_select_qualified():
    r = sqlproc.extract("INSERT INTO db.tgt (id, amt) SELECT s.cust, s.total FROM db.src s;")
    assert _cols(r, "db.src") == ["cust", "total"]
    assert _cols(r, "db.tgt") == ["amt", "id"]


def test_used_columns_standalone_select_single_table():
    r = sqlproc.extract("SELECT col1, col2 FROM db.only;")
    assert _cols(r, "db.only") == ["col1", "col2"]
    assert r.column_lineage == []          # no write target -> no column lineage


def test_used_columns_delete_where():
    r = sqlproc.extract("DELETE FROM db.t WHERE flag = 1;")
    assert _cols(r, "db.t") == ["flag"]


def test_column_lineage_explicit_target_list():
    r = sqlproc.extract("INSERT INTO db.tgt (id, amt) SELECT s.cust, s.total FROM db.src s;")
    assert _has_col_lineage(r, "db.tgt", "id", "db.src", "cust")
    assert _has_col_lineage(r, "db.tgt", "amt", "db.src", "total")


def test_column_lineage_implicit_names():
    # No explicit column list -> the projection output names are the target columns
    r = sqlproc.extract("INSERT INTO db.tgt SELECT cust, total FROM db.src;")
    assert _has_col_lineage(r, "db.tgt", "cust", "db.src", "cust")
    assert _has_col_lineage(r, "db.tgt", "total", "db.src", "total")


def test_column_lineage_ctas_multi_source_expression():
    # A derived column (y + z AS yz) traces back to BOTH source columns
    r = sqlproc.extract("CREATE TABLE db.tgt AS (SELECT a.x AS x, a.y + a.z AS yz FROM db.src a) WITH DATA;")
    assert _has_col_lineage(r, "db.tgt", "x", "db.src", "x")
    assert _has_col_lineage(r, "db.tgt", "yz", "db.src", "y")
    assert _has_col_lineage(r, "db.tgt", "yz", "db.src", "z")


def test_column_lineage_join_resolves_aliases():
    r = sqlproc.extract("INSERT INTO db.t SELECT a.id, b.name FROM db.a a JOIN db.b b ON a.id=b.id;")
    assert _has_col_lineage(r, "db.t", "id", "db.a", "id")
    assert _has_col_lineage(r, "db.t", "name", "db.b", "name")


def test_column_extraction_star_is_graceful():
    # SELECT * yields no column nodes (no exp.Column) but never crashes
    r = sqlproc.extract("INSERT INTO db.t SELECT * FROM db.s;")
    assert r.column_lineage == []
    assert r.table_ops.get("db.t") == ["write"]
    assert r.table_ops.get("db.s") == ["read"]


def test_column_extraction_unparseable_is_skipped():
    # A broken statement must not crash column extraction; the good one survives
    r = sqlproc.extract("INSERT INTO db.t (a) SELECT a FROM db.s; this is not sql at all !!!;")
    assert _has_col_lineage(r, "db.t", "a", "db.s", "a")


def test_columns_no_phantom_temporal_keyword():
    r = sqlproc.extract(
        "NONSEQUENCED VALIDTIME INSERT INTO db.target "
        "SELECT t1.k, t2.v FROM db.t1 t1 JOIN db.t2 t2 "
        "ON t1.k=t2.k AND PERIOD(t2.vs, t2.ve) CONTAINS (t1.dt);"
    )
    owners = set(r.used_columns) | {s for _, _, s, _ in r.column_lineage}
    assert all(o.strip().lower() not in _PHANTOM for o in owners)


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
