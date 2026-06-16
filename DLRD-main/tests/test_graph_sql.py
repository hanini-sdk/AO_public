"""Graph-assembly tests for the SQL data-lineage edges (Phases C + E).

Self-contained: run directly with `python tests/test_graph_sql.py` (no pytest
needed) or via `pytest tests/`. Drives app.graph.build_graph with synthetic SQL
files (parsed by the real sqlproc pipeline) and asserts the aggregated
file<->table operation edges, the single "Data" layer, the table<->table
provenance edges, and the Phase-E file->file "feeds" (depends_on) edges.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app import sqlproc                       # noqa: E402
from app.graph import build_graph             # noqa: E402
from app.scanner import ScannedFile           # noqa: E402
from app.parser import FileParse              # noqa: E402
from app.enrich import FileEnrichment         # noqa: E402
from app.config import Settings               # noqa: E402

_DATA_LAYER_ID = "layer:data"


def _graph(files):
    """files: list of (rel_path, sql_text) -> built graph dict."""
    scanned, parses, enrich = [], [], {}
    for rel, sql in files:
        scanned.append(ScannedFile(abs_path=rel, rel_path=rel, ext=".sql",
                                   language="sql", size_bytes=len(sql)))
        fp = FileParse(rel_path=rel, language="sql", grammar_key=None,
                       parse_ok=True, symbols=[], imports=[])
        fp.sql = sqlproc.extract(sql)
        parses.append(fp)
        # deliberately mis-label the layer as Other to prove graph.py forces Data
        enrich[rel] = FileEnrichment(summary="s", layer="Other", complexity="simple",
                                     tags=["sql"], members={}, llm_ok=False)
    return build_graph(project_name="t", project_root=".", scanned=scanned,
                       parses=parses, enrichments=enrich, settings=Settings(),
                       description="", frameworks=[])


def _edges(g, **kw):
    return [e for e in g["edges"] if all(e.get(k) == v for k, v in kw.items())]


def _node_id(g, **kw):
    for n in g["nodes"]:
        if all(n.get(k) == v for k, v in kw.items()):
            return n["id"]
    return None


def _data_layer(g):
    return next((L for L in g["layers"] if L["id"] == _DATA_LAYER_ID), None)


# --- Phase C: aggregated file<->table op-edges -------------------------------
def test_C_write_and_read_op_edges():
    g = _graph([("load_x.sql", "INSERT INTO db.x SELECT a.k FROM db.src a;")])
    fx = _node_id(g, type="file", filePath="load_x.sql")
    tx = _node_id(g, type="table", name="db.x")
    ts = _node_id(g, type="table", name="db.src")
    we = _edges(g, source=fx, target=tx)
    assert len(we) == 1 and we[0]["type"] == "writes_to" and we[0]["description"] == "write"
    re_ = _edges(g, source=fx, target=ts)
    assert len(re_) == 1 and re_[0]["type"] == "reads_from" and re_[0]["description"] == "read"


def test_C_purge_is_writes_to():
    g = _graph([("p.sql", "DELETE db.x ALL;")])
    fx = _node_id(g, type="file", filePath="p.sql")
    tx = _node_id(g, type="table", name="db.x")
    e = _edges(g, source=fx, target=tx)
    assert len(e) == 1 and e[0]["type"] == "writes_to" and e[0]["description"] == "purge"


def test_C_op_aggregation_single_edge():
    # purge then reload of the same table -> ONE edge with the ORDERED op sequence
    g = _graph([("rw.sql", "DELETE db.x ALL; INSERT INTO db.x SELECT a.k FROM db.src a;")])
    fx = _node_id(g, type="file", filePath="rw.sql")
    tx = _node_id(g, type="table", name="db.x")
    e = _edges(g, source=fx, target=tx)
    assert len(e) == 1
    assert e[0]["type"] == "writes_to"
    assert e[0]["description"] == "purge → write"  # execution order, arrow-joined


def test_C_no_file_table_contains():
    g = _graph([("load_x.sql", "CREATE TABLE db.x AS (SELECT k FROM db.src) WITH DATA;")])
    fx = _node_id(g, type="file", filePath="load_x.sql")
    table_ids = {n["id"] for n in g["nodes"] if n["type"] == "table"}
    assert not [e for e in g["edges"]
                if e["type"] == "contains" and e["source"] == fx and e["target"] in table_ids]


def test_C_single_data_layer():
    g = _graph([("load_x.sql", "INSERT INTO db.x SELECT a.k FROM db.src a;")])
    fx = _node_id(g, type="file", filePath="load_x.sql")
    layer = _data_layer(g)
    assert layer is not None
    assert fx in layer["nodeIds"]                       # SQL file forced into Data
    for n in g["nodes"]:
        if n["type"] == "table":
            assert n["id"] in layer["nodeIds"]          # every table in Data


def test_C_table_to_table_provenance_kept():
    g = _graph([("load_x.sql", "INSERT INTO db.x SELECT a.k FROM db.src a;")])
    tx = _node_id(g, type="table", name="db.x")
    ts = _node_id(g, type="table", name="db.src")
    assert len(_edges(g, source=tx, target=ts, type="reads_from")) == 1


# --- Phase E: file->file "feeds" (depends_on) edges --------------------------
def test_E_feeds_depends_on():
    g = _graph([
        ("producer.sql", "INSERT INTO db.x SELECT a.k FROM db.src a;"),  # writes db.x
        ("consumer.sql", "INSERT INTO db.y SELECT b.k FROM db.x b;"),    # reads db.x
    ])
    fprod = _node_id(g, type="file", filePath="producer.sql")
    fcons = _node_id(g, type="file", filePath="consumer.sql")
    e = _edges(g, source=fcons, target=fprod, type="depends_on")
    assert len(e) == 1 and e[0]["description"] == "feeds via db.x"  # names the shared table


def test_E_no_self_feed():
    # a file that writes then reads the same table must not depend on itself
    g = _graph([("solo.sql", "INSERT INTO db.x SELECT a.k FROM db.x a;")])
    assert _edges(g, type="depends_on") == []


def test_E_purge_only_is_not_a_feed():
    # a pure DELETE/TRUNCATE removes data; it does not "produce" a feed for a reader
    g = _graph([
        ("truncate.sql", "DELETE db.x ALL;"),
        ("reader.sql", "INSERT INTO db.y SELECT b.k FROM db.x b;"),
    ])
    assert _edges(g, type="depends_on") == []


def test_E_feed_is_single_edge_naming_shared_tables():
    # producer writes x and y; consumer reads both -> ONE edge naming both tables
    g = _graph([
        ("prod.sql", "INSERT INTO db.x SELECT k FROM db.s; INSERT INTO db.y SELECT k FROM db.s;"),
        ("cons.sql", "INSERT INTO db.z SELECT a.k FROM db.x a JOIN db.y b ON a.k=b.k;"),
    ])
    fprod = _node_id(g, type="file", filePath="prod.sql")
    fcons = _node_id(g, type="file", filePath="cons.sql")
    e = _edges(g, source=fcons, target=fprod, type="depends_on")
    assert len(e) == 1
    assert e[0]["description"] == "feeds via db.x, db.y"


def test_E_feeds_names_capped_with_plus_n():
    # >3 shared tables -> show the first 3 names then "+N"
    prod = "".join(f"INSERT INTO db.{t} SELECT k FROM db.s; " for t in "abcde")
    cons = "".join(f"SELECT k FROM db.{t}; " for t in "abcde")
    g = _graph([("prod.sql", prod), ("cons.sql", cons)])
    fprod = _node_id(g, type="file", filePath="prod.sql")
    fcons = _node_id(g, type="file", filePath="cons.sql")
    e = _edges(g, source=fcons, target=fprod, type="depends_on")
    assert len(e) == 1
    assert e[0]["description"] == "feeds via db.a, db.b, db.c, +2"


# --- C1/C2: column nodes + column<-column lineage edges ----------------------
def _col_id(g, table, col):
    return _node_id(g, type="column", name=col, summary=f"Column {col} of {table}.")


def test_C_columns_become_nodes_under_their_table():
    g = _graph([("load.sql", "INSERT INTO db.tgt (id, amt) SELECT s.cust, s.total FROM db.src s;")])
    tgt = _node_id(g, type="table", name="db.tgt")
    src = _node_id(g, type="table", name="db.src")
    id_col = _col_id(g, "db.tgt", "id")
    amt_col = _col_id(g, "db.tgt", "amt")
    cust_col = _col_id(g, "db.src", "cust")
    assert id_col and amt_col and cust_col
    # each column is attached to its table via a contains edge
    assert len(_edges(g, source=tgt, target=id_col, type="contains")) == 1
    assert len(_edges(g, source=src, target=cust_col, type="contains")) == 1


def test_C_columns_join_the_data_layer():
    g = _graph([("load.sql", "INSERT INTO db.t (a) SELECT s.a FROM db.s s;")])
    col = _col_id(g, "db.t", "a")
    data = _data_layer(g)
    assert col and data is not None and col in data["nodeIds"]


def test_C_column_lineage_reads_from_edge():
    g = _graph([("load.sql", "INSERT INTO db.tgt (id) SELECT s.cust FROM db.src s;")])
    tgt_col = _col_id(g, "db.tgt", "id")
    src_col = _col_id(g, "db.src", "cust")
    e = _edges(g, source=tgt_col, target=src_col, type="reads_from")
    assert len(e) == 1  # target column reads_from its source column


def test_C_no_columns_for_star_select():
    g = _graph([("load.sql", "INSERT INTO db.t SELECT * FROM db.s;")])
    assert _node_id(g, type="column") is None  # SELECT * yields no column nodes


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
