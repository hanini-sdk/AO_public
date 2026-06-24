"""Phase 3 — SQL embedded in bteq heredocs of shell scripts.

Self-contained: run directly with `python tests/test_shell_sql.py` or via
`pytest tests/`. Drives the real tree-sitter-bash parse + app.parser's bteq
heredoc extraction, then a few graph-assembly assertions (shell files gain DATA
edges + shell->shell feeds, but stay code nodes — not the Data layer).
"""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app import parser as P                    # noqa: E402
from app.graph import build_graph              # noqa: E402
from app.scanner import ScannedFile            # noqa: E402
from app.parser import FileParse               # noqa: E402
from app.enrich import FileEnrichment          # noqa: E402
from app.config import Settings                # noqa: E402


def _shell_result(src: str):
    """Parse shell `src` with tree-sitter-bash and run app.parser's bteq heredoc
    extraction (same path parse_file uses for .sh files)."""
    from tree_sitter import Language, Parser
    import tree_sitter_bash as tsb

    b = src.encode("utf-8")
    tree = Parser(Language(tsb.language())).parse(b)
    _, _, variables = P._collect_shell_refs(tree.root_node, b)
    return P._extract_shell_sql(tree.root_node, b, variables)


def _ops(res, table):
    return (res.table_ops.get(table) if res else None)


# --- Part A: per-case extraction --------------------------------------------
def test_arbitrary_delimiter():
    r = _shell_result("bteq <<EOBTQ\nINSERT INTO db.tgt SELECT a FROM db.src;\nEOBTQ\n")
    assert _ops(r, "db.tgt") == ["write"]
    assert _ops(r, "db.src") == ["read"]


def test_quoted_delimiter_no_expansion():
    # <<'EOF' disables expansion: $DB must NOT become 'prod'
    r = _shell_result("DB=prod\nbteq <<'EOF'\nSELECT a FROM ${DB}.src;\nEOF\n")
    assert r is not None
    assert not any("prod" in t.lower() for t in r.table_refs)   # not expanded
    assert r.table_refs                                          # but a table was still found


def test_unquoted_delimiter_expands_var():
    r = _shell_result("DB=prod\nbteq <<EOF\nSELECT a FROM ${DB}.src;\nEOF\n")
    assert "prod.src" in r.table_refs                            # $DB -> prod


def test_dash_heredoc_strips_tabs():
    src = "bteq <<-EOF\n\tSELECT a FROM db.t1;\n\tEOF\n"          # leading tabs (<<-)
    r = _shell_result(src)
    assert _ops(r, "db.t1") == ["read"]


def test_piped_into_bteq():
    # `cat file | bteq <<EOF` — heredoc is bteq's stdin even mid-pipeline
    r = _shell_result("cat input.txt | bteq <<EOF\nINSERT INTO db.p SELECT a FROM db.q;\nEOF\n")
    assert _ops(r, "db.p") == ["write"]
    assert _ops(r, "db.q") == ["read"]


def test_heredoc_inside_function():
    src = "load() {\n  bteq <<EOF\nINSERT INTO db.f SELECT a FROM db.g;\nEOF\n}\n"
    r = _shell_result(src)
    assert _ops(r, "db.f") == ["write"]
    assert _ops(r, "db.g") == ["read"]


def test_escaped_dollar_is_literal():
    # \${DB} is a literal ${DB}, NOT expanded to 'prod'
    r = _shell_result("DB=prod\nbteq <<EOF\nSELECT a FROM \\${DB}.keep;\nEOF\n")
    assert r is not None
    assert "prod.keep" not in r.table_refs
    assert not any("prod" in t.lower() for t in r.table_refs)


def test_multiple_heredocs_order_preserved():
    src = (
        "bteq <<EOF\nDELETE FROM db.x ALL;\nEOF\n"
        "bteq <<EOF\nINSERT INTO db.x SELECT a FROM db.s;\nEOF\n"
        "bteq <<EOF\nSELECT a FROM db.x;\nEOF\n"
    )
    r = _shell_result(src)
    assert _ops(r, "db.x") == ["purge", "write", "read"]        # source order across heredocs


def test_embedded_logon_credentials_stripped():
    src = (
        "bteq <<EOF\n"
        ".LOGON tdpid/svcuser,SUPERSECRET\n"
        "INSERT INTO db.t SELECT a FROM db.s;\n"
        ".LOGOFF\n"
        "EOF\n"
    )
    r = _shell_result(src)
    assert _ops(r, "db.t") == ["write"]
    assert _ops(r, "db.s") == ["read"]
    blob = " ".join(list(r.table_refs) + [o.name for o in r.objects])
    assert "SUPERSECRET" not in blob and "svcuser" not in blob and "tdpid" not in blob


def test_no_bteq_heredoc_returns_none():
    # a non-bteq heredoc must not be picked up
    assert _shell_result("cat <<EOF\njust some text\nEOF\n") is None
    assert _shell_result("echo hello\n") is None


# --- Part A: graph integration (DATA edges, code layer, shell->shell feeds) ---
def _shell_fp(rel: str, src: str) -> FileParse:
    return FileParse(rel, "shell", "shell", True, [], [], [], {}, sql=_shell_result(src))


def _graph(files):
    """files: list of (rel, shell_src). Shell files enriched as 'Utility'."""
    scanned, parses, enrich = [], [], {}
    for rel, src in files:
        scanned.append(ScannedFile(abs_path=rel, rel_path=rel, ext=".sh",
                                   language="shell", size_bytes=len(src)))
        parses.append(_shell_fp(rel, src))
        enrich[rel] = FileEnrichment(summary="s", layer="Utility", complexity="simple",
                                     tags=["shell"], members={}, llm_ok=False)
    return build_graph(project_name="t", project_root=".", scanned=scanned, parses=parses,
                       enrichments=enrich, settings=Settings(), description="", frameworks=[])


def _node_id(g, **kw):
    for n in g["nodes"]:
        if all(n.get(k) == v for k, v in kw.items()):
            return n["id"]
    return None


def _layer(g, lid):
    return next((L for L in g["layers"] if L["id"] == lid), None)


def test_graph_shell_gets_data_edge_but_stays_code_node():
    g = _graph([("load.sh", "bteq <<EOF\nINSERT INTO db.x SELECT a FROM db.src;\nEOF\n")])
    fid = _node_id(g, type="file", filePath="load.sh")
    tx = _node_id(g, type="table", name="db.x")
    # DATA op edge from the shell file to the table
    we = [e for e in g["edges"] if e["source"] == fid and e["target"] == tx]
    assert len(we) == 1 and we[0]["type"] == "writes_to"
    # ...but the shell file stays a code node: Utility layer, NOT Data
    data = _layer(g, "layer:data")
    util = _layer(g, "layer:utility")
    assert data is not None and fid not in data["nodeIds"]
    assert util is not None and fid in util["nodeIds"]
    assert tx in data["nodeIds"]                                 # the table is Data


def test_graph_shell_to_shell_feeds_emerges():
    g = _graph([
        ("producer.sh", "bteq <<EOF\nINSERT INTO db.y SELECT a FROM db.s;\nEOF\n"),
        ("consumer.sh", "bteq <<EOF\nINSERT INTO db.z SELECT a FROM db.y;\nEOF\n"),
    ])
    fp = _node_id(g, type="file", filePath="producer.sh")
    fc = _node_id(g, type="file", filePath="consumer.sh")
    feeds = [e for e in g["edges"] if e["source"] == fc and e["target"] == fp and e["type"] == "depends_on"]
    assert len(feeds) == 1 and feeds[0]["description"] == "feeds via db.y"


# --- Part B: bteq sinks pointing at a separate .sql file (runs edge) ---------
def test_bteq_stdin_redirect_reference():
    r = _shell_result("bteq < load.sql\n")
    assert r is not None and "load.sql" in r.references


def test_cat_piped_into_bteq_reference():
    r = _shell_result("cat /sql/daily.sql | bteq\n")
    assert r is not None and any(ref.endswith("daily.sql") for ref in r.references)


def test_run_file_in_heredoc_reference():
    # .RUN FILE= inside a bteq heredoc is captured by sqlproc (Part A path)
    r = _shell_result("bteq <<EOF\n.RUN FILE=sub.sql\nEOF\n")
    assert r is not None and "sub.sql" in r.references


def _graph_mixed(shell_files, sql_files):
    from app import sqlproc
    scanned, parses, enrich = [], [], {}
    for rel, src in shell_files:
        scanned.append(ScannedFile(abs_path=rel, rel_path=rel, ext=".sh", language="shell", size_bytes=len(src)))
        parses.append(_shell_fp(rel, src))
        enrich[rel] = FileEnrichment(summary="s", layer="Utility", complexity="simple", tags=["shell"], members={}, llm_ok=False)
    for rel, txt in sql_files:
        scanned.append(ScannedFile(abs_path=rel, rel_path=rel, ext=".sql", language="sql", size_bytes=len(txt)))
        parses.append(FileParse(rel, "sql", None, True, [], [], [], {}, sql=sqlproc.extract(txt)))
        enrich[rel] = FileEnrichment(summary="s", layer="Data", complexity="simple", tags=["sql"], members={}, llm_ok=False)
    return build_graph(project_name="t", project_root=".", scanned=scanned, parses=parses,
                       enrichments=enrich, settings=Settings(), description="", frameworks=[])


def test_graph_bteq_stdin_runs_edge():
    g = _graph_mixed([("run.sh", "bteq < load.sql\n")],
                     [("load.sql", "INSERT INTO db.t SELECT a FROM db.s;")])
    fsh = _node_id(g, type="file", filePath="run.sh")
    fsql = _node_id(g, type="file", filePath="load.sql")
    e = [x for x in g["edges"] if x["source"] == fsh and x["target"] == fsql and x["type"] == "imports"]
    assert len(e) == 1 and e[0].get("description") == "runs"


def test_graph_cat_pipe_runs_edge():
    g = _graph_mixed([("run.sh", "cat load.sql | bteq\n")],
                     [("load.sql", "SELECT a FROM db.s;")])
    fsh = _node_id(g, type="file", filePath="run.sh")
    fsql = _node_id(g, type="file", filePath="load.sql")
    e = [x for x in g["edges"] if x["source"] == fsh and x["target"] == fsql and x["type"] == "imports"]
    assert len(e) == 1 and e[0].get("description") == "runs"


def test_graph_unresolved_reference_becomes_missing_node():
    # A concrete absent reference is no longer silently dropped: it materializes a
    # layerless "missing" node + an imports/"runs" edge (hidden behind MISSING DATA).
    g = _graph_mixed([("run.sh", "bteq < not_in_project.sql\n")], [])
    fsh = _node_id(g, type="file", filePath="run.sh")
    miss = [n for n in g["nodes"] if n["type"] == "missing" and n["name"] == "not_in_project.sql"]
    assert len(miss) == 1
    e = [x for x in g["edges"] if x["source"] == fsh and x["target"] == miss[0]["id"] and x["type"] == "imports"]
    assert len(e) == 1 and e[0].get("description") == "runs"
    assert all(miss[0]["id"] not in L["nodeIds"] for L in g["layers"])  # layerless


# --- Missing-reference nodes: concrete script/SQL refs absent from the project -
def _calls_fp(rel: str, calls, lang: str = "shell") -> FileParse:
    gk = "shell" if lang == "shell" else None
    return FileParse(rel, lang, gk, True, [], [], list(calls), {})


def _graph_parses(parses, exts):
    """Build a graph from explicit FileParse objects. exts: rel -> file extension."""
    scanned, enrich = [], {}
    for fp in parses:
        scanned.append(ScannedFile(abs_path=fp.rel_path, rel_path=fp.rel_path,
                                   ext=exts.get(fp.rel_path, ".sh"),
                                   language=fp.language, size_bytes=10))
        enrich[fp.rel_path] = FileEnrichment(summary="s", layer="Utility", complexity="simple",
                                             tags=[fp.language], members={}, llm_ok=False)
    return build_graph(project_name="t", project_root=".", scanned=scanned, parses=parses,
                       enrichments=enrich, settings=Settings(), description="", frameworks=[])


def _missing(g):
    return [n for n in g["nodes"] if n.get("type") == "missing"]


def _in_no_layer(g, nid):
    return all(nid not in L["nodeIds"] for L in g["layers"])


def test_missing_present_script_resolves_no_missing_node():
    g = _graph_parses([_calls_fp("main.sh", ["child.sh"]), _calls_fp("child.sh", [])],
                      {"main.sh": ".sh", "child.sh": ".sh"})
    assert _missing(g) == []                                  # present file -> resolved, no missing node
    main = _node_id(g, type="file", filePath="main.sh")
    child = _node_id(g, type="file", filePath="child.sh")
    assert [e for e in g["edges"] if e["source"] == main and e["target"] == child and e["type"] == "calls"]


def test_missing_absent_script_creates_one_node_and_edge():
    g = _graph_parses([_calls_fp("main.sh", ["ghost.sh"])], {"main.sh": ".sh"})
    miss = _missing(g)
    assert len(miss) == 1
    m = miss[0]
    assert m["name"] == "ghost.sh" and "missing" in m["tags"] and "script" in m["tags"]
    main = _node_id(g, type="file", filePath="main.sh")
    e = [x for x in g["edges"] if x["source"] == main and x["target"] == m["id"]]
    assert len(e) == 1 and e[0]["type"] == "calls"
    assert _in_no_layer(g, m["id"])                           # OFF-invariant: missing node in no layer grouping


def test_missing_variable_glob_flag_refs_skipped():
    g = _graph_parses([_calls_fp("main.sh", ["$STEP", "load_*.sh", "-x"])], {"main.sh": ".sh"})
    assert _missing(g) == []
    main = _node_id(g, type="file", filePath="main.sh")
    assert [e for e in g["edges"] if e["source"] == main and e["type"] == "calls"] == []


def test_missing_ambiguous_basename_skipped_not_fabricated():
    # two scanned files share the basename 'util.sh' -> an unqualified ref is ambiguous
    g = _graph_parses([_calls_fp("main.sh", ["util.sh"]),
                       _calls_fp("a/util.sh", []), _calls_fp("b/util.sh", [])],
                      {"main.sh": ".sh", "a/util.sh": ".sh", "b/util.sh": ".sh"})
    assert _missing(g) == []                                  # the file exists (ambiguously) -> no fabrication


def test_missing_dedup_one_node_multiple_referrers():
    g = _graph_parses([_calls_fp("a.sh", ["ghost.sh"]), _calls_fp("b.sh", ["ghost.sh"])],
                      {"a.sh": ".sh", "b.sh": ".sh"})
    miss = _missing(g)
    assert len(miss) == 1                                     # one node per unique missing name
    mid = miss[0]["id"]
    a = _node_id(g, type="file", filePath="a.sh")
    b = _node_id(g, type="file", filePath="b.sh")
    assert {e["source"] for e in g["edges"] if e["target"] == mid and e["type"] == "calls"} == {a, b}


def test_missing_joblist_absent_keeps_index_on_edge():
    g = _graph_parses([_calls_fp("jobs.list", ["ghost_job.sh"], lang="joblist")],
                      {"jobs.list": ".list"})
    miss = _missing(g)
    assert len(miss) == 1
    job = _node_id(g, type="file", filePath="jobs.list")
    e = [x for x in g["edges"] if x["source"] == job and x["target"] == miss[0]["id"]]
    assert len(e) == 1 and e[0]["type"] == "calls" and e[0].get("index") == 0


def test_missing_self_reference_not_fabricated():
    # a script that re-execs / lists its OWN basename must NOT become a "missing" node
    g = _graph_parses([_calls_fp("dir/run.sh", ["run.sh"])], {"dir/run.sh": ".sh"})
    assert _missing(g) == []
    # and the same via a SQL .RUN FILE= naming the script's own basename
    fp = _shell_fp("dir/load.sh", "bteq <<EOF\n.RUN FILE=load.sh\nEOF\n")
    g2 = _graph_parses([fp], {"dir/load.sh": ".sh"})
    assert _missing(g2) == []


def test_missing_sql_run_file_absent_creates_missing_node():
    # a bteq heredoc with .RUN FILE= pointing at an absent .sql -> one missing node + runs edge
    fp = _shell_fp("run.sh", "bteq <<EOF\n.RUN FILE=absent_sub.sql\nEOF\n")
    g = _graph_parses([fp], {"run.sh": ".sh"})
    miss = _missing(g)
    assert len(miss) == 1 and miss[0]["name"] == "absent_sub.sql" and "sql-file" in miss[0]["tags"]
    run = _node_id(g, type="file", filePath="run.sh")
    e = [x for x in g["edges"] if x["source"] == run and x["target"] == miss[0]["id"]]
    assert len(e) == 1 and e[0]["type"] == "imports" and e[0].get("description") == "runs"
    assert _in_no_layer(g, miss[0]["id"])


def test_missing_shell_keywords_and_interpreters_skipped():
    # reserved words / builtins / interpreters (even path-qualified) are never
    # script refs -> no missing node, no calls edge (graph.py backstop)
    g = _graph_parses([_calls_fp("run.sh", ["break", "true", "exit", "/bin/true", "bash", "continue", ":"])],
                      {"run.sh": ".sh"})
    assert _missing(g) == []
    main = _node_id(g, type="file", filePath="run.sh")
    assert [e for e in g["edges"] if e["source"] == main and e["type"] == "calls"] == []


def test_shell_keyword_calls_filtered_at_source():
    # the over-capture is fixed at the source: _collect_shell_refs drops keywords
    from tree_sitter import Language, Parser
    import tree_sitter_bash as tsb
    src = "while true; do\n  break\ndone\n/bin/true\nbash other.sh\n./run_load.sh\nprocess.ksh\n"
    b = src.encode("utf-8")
    tree = Parser(Language(tsb.language())).parse(b)
    _, calls, _ = P._collect_shell_refs(tree.root_node, b)
    for kw in ("true", "break", "/bin/true", "bash"):
        assert kw not in calls, f"{kw} should be filtered at source"
    assert "./run_load.sh" in calls   # a real script invocation still survives
    assert "process.ksh" in calls


def test_missing_real_file_named_like_builtin_resolves():
    # a REAL scanned script whose bare name equals a shell builtin keeps its edge
    # (denylist runs AFTER resolution, so it only suppresses phantoms)
    g = _graph_parses([_calls_fp("main.sh", ["test", "source"]),
                       _calls_fp("test.sh", []), _calls_fp("source.sh", [])],
                      {"main.sh": ".sh", "test.sh": ".sh", "source.sh": ".sh"})
    assert _missing(g) == []                                  # resolved, not fabricated as missing
    main = _node_id(g, type="file", filePath="main.sh")
    t = _node_id(g, type="file", filePath="test.sh")
    s = _node_id(g, type="file", filePath="source.sh")
    calls = {(e["source"], e["target"]) for e in g["edges"] if e["type"] == "calls"}
    assert (main, t) in calls and (main, s) in calls          # real edges kept


def test_missing_unresolved_builtin_still_skipped():
    # the same builtin names with NO matching file still produce no missing node
    g = _graph_parses([_calls_fp("main.sh", ["test", "source", "break", "true"])], {"main.sh": ".sh"})
    assert _missing(g) == []
    main = _node_id(g, type="file", filePath="main.sh")
    assert [e for e in g["edges"] if e["source"] == main and e["type"] == "calls"] == []


# --- Launcher pattern: scripts invoked THROUGH wrapper functions by path arg ---
def _real_shell_fp(rel: str, src: str) -> FileParse:
    """A shell FileParse from a REAL tree-sitter parse — carrying both the function
    symbols and the script-call refs, exactly as parse_file would for a .sh file."""
    from tree_sitter import Language, Parser
    import tree_sitter_bash as tsb

    b = src.encode("utf-8")
    tree = Parser(Language(tsb.language())).parse(b)
    symbols = P._collect_symbols(tree.root_node, b, "shell")
    _, calls, variables = P._collect_shell_refs(tree.root_node, b)
    return FileParse(rel, "shell", "shell", True, symbols, [], calls, variables)


def test_launcher_script_path_args_captured_at_source():
    # *.sh path arguments handed to a wrapper command (a project function, an
    # interpreter) are captured by basename; an echo'd .sh is not an execution.
    from tree_sitter import Language, Parser
    import tree_sitter_bash as tsb

    src = (
        'function func_RUN_STEP { /bin/ksh "$2"; }\n'
        'func_RUN_STEP "S1" "$ELTA_ROOT/dir/child_one.sh"\n'
        'bash deploy.sh\n'
        'echo "skip $ROOT/not_a_call.sh"\n'
    )
    b = src.encode("utf-8")
    tree = Parser(Language(tsb.language())).parse(b)
    _, calls, _ = P._collect_shell_refs(tree.root_node, b)
    assert "child_one.sh" in calls        # wrapper-passed path arg -> basename
    assert "deploy.sh" in calls           # interpreter-passed script arg
    assert "func_RUN_STEP" in calls       # command token still captured (suppressed in graph)
    assert "not_a_call.sh" not in calls   # echo's arg is a printed string, not a launch
    assert "$ELTA_ROOT/dir/child_one.sh" not in calls   # never the full variable path


def test_launcher_functions_not_missing_real_scripts_linked():
    # The real orchestrator shape: wrapper functions launch the script whose PATH
    # is passed as an argument. Functions must NOT become missing nodes; the path
    # args resolve (present) or become MISSING SCRIPT nodes (absent), by basename.
    launcher_src = (
        'function func_RUN_STEP {\n'
        '  /bin/ksh "$2"\n'
        '}\n'
        'function func_RUN_PARALLEL_STEP {\n'
        '  /bin/ksh "$2" &\n'
        '}\n'
        'func_RUN_STEP          "STEP101"        "$ELTA_ROOT/scripts/exec/LDPS/PU00_LDPS_XZ01_S_PU_LTS_ENTITY.sh"\n'
        'func_RUN_PARALLEL_STEP "STEP_PUNG_201"  "$ELTA_ROOT/scripts/exec/LDPS/present_child.sh"\n'
        './really_absent.sh\n'
    )
    launcher = _real_shell_fp("orch/PU00_CHARGEMENT_PU_LTS.sh", launcher_src)
    present = _calls_fp("scripts/exec/LDPS/present_child.sh", [])
    g = _graph_parses([launcher, present],
                      {"orch/PU00_CHARGEMENT_PU_LTS.sh": ".sh",
                       "scripts/exec/LDPS/present_child.sh": ".sh"})
    missing_names = {n["name"] for n in _missing(g)}
    launcher_id = _node_id(g, type="file", filePath="orch/PU00_CHARGEMENT_PU_LTS.sh")

    # (a) wrappers are FUNCTION nodes contained by the file, never missing
    assert "func_RUN_STEP" not in missing_names
    assert "func_RUN_PARALLEL_STEP" not in missing_names
    for fname in ("func_RUN_STEP", "func_RUN_PARALLEL_STEP"):
        fid = _node_id(g, type="function", name=fname)
        assert fid is not None
        assert any(e["source"] == launcher_id and e["target"] == fid and e["type"] == "contains"
                   for e in g["edges"])

    # (b) the ABSENT path arg -> a MISSING SCRIPT node labeled by its basename
    absent = [n for n in _missing(g) if n["name"] == "PU00_LDPS_XZ01_S_PU_LTS_ENTITY.sh"]
    assert len(absent) == 1 and "missing" in absent[0]["tags"] and "script" in absent[0]["tags"]

    # (c) the PRESENT path arg -> a resolved calls edge (no missing node)
    present_id = _node_id(g, type="file", filePath="scripts/exec/LDPS/present_child.sh")
    assert [e for e in g["edges"]
            if e["source"] == launcher_id and e["target"] == present_id and e["type"] == "calls"]
    assert "present_child.sh" not in missing_names

    # (d) a genuinely-absent direct path-qualified call still -> missing
    assert "really_absent.sh" in missing_names

    # node labels carry basenames only — never a $VAR/... path (R3)
    assert not any("$" in n["name"] or "/" in n["name"] for n in _missing(g))


def test_launcher_function_call_tallied_separately():
    # the reference tally distinguishes a suppressed internal function call from a
    # resolved / missing / skipped file reference (run-diagnostics accuracy)
    launcher = _real_shell_fp("main.sh",
                              'function step { :; }\nstep "x" "child.sh"\nstep\n')
    child = _calls_fp("child.sh", [])
    refc: dict = {}
    scanned, enrich = [], {}
    for fp in (launcher, child):
        scanned.append(ScannedFile(abs_path=fp.rel_path, rel_path=fp.rel_path, ext=".sh",
                                   language="shell", size_bytes=10))
        enrich[fp.rel_path] = FileEnrichment(summary="s", layer="Utility", complexity="simple",
                                             tags=["shell"], members={}, llm_ok=False)
    build_graph(project_name="t", project_root=".", scanned=scanned, parses=[launcher, child],
                enrichments=enrich, settings=Settings(), description="", frameworks=[],
                diagnostics=refc)
    refs = refc["references"]
    assert refs["function_calls"] >= 1   # the bare `step` call(s) -> internal function
    assert refs["resolved"] >= 1         # the "child.sh" path arg -> resolved file
    assert refs["missing"] == 0


# --- Commented-SQL recovery: `--`-commented SQL read as real, tagged SQL --------
def _sql_fp(rel: str, block: str, lang: str = "sql") -> FileParse:
    from app import sqlproc
    gk = "shell" if lang == "shell" else None
    return FileParse(rel, lang, gk, True, [], [], [], {}, sql=sqlproc.extract(block))


def test_commented_insert_columns_become_tagged_nodes_in_lineage():
    # exec + a commented multi-line INSERT...SELECT into an escaped ${...}-qualified
    # target, with a column unique to the comment -> real table + column nodes,
    # tagged "commented", reachable in lineage.
    block = (
        "exec $DB_EXE.PROC_LOAD();\n"
        "-- Statement 1: load authorizations\n"
        "-- insert into \\${DB_PREP}.PW_AUTORISATION_PU\n"
        "--   ( ID_PU , CD_TYP_APPRO_BALE_2 , DT_MAJ )\n"
        "-- select s.ID_PU , s.CD_TYP_APPRO_BALE_2 , s.DT_MAJ\n"
        "-- from \\${DB_STG}.STG_AUTORISATION s\n"
        "-- ;\n"
    )
    g = _graph_parses([_sql_fp("orch/run_load.sh", block, "shell")], {"orch/run_load.sh": ".sh"})
    tgt = next((n for n in g["nodes"] if n["type"] == "table"
                and n["name"] == "${DB_PREP}.PW_AUTORISATION_PU"), None)
    assert tgt is not None and "commented" in tgt["tags"]            # escaped sigil resolved + tagged
    col = next((n for n in g["nodes"] if n["type"] == "column"
                and n["name"] == "CD_TYP_APPRO_BALE_2"), None)
    assert col is not None and "commented" in col["tags"]           # comment-unique column
    touching = [e for e in g["edges"] if col["id"] in (e["source"], e["target"])]
    assert {"contains", "reads_from"} <= {e["type"] for e in touching}  # in lineage
    assert all(e.get("commented") for e in touching)                # recovered edges tagged


def test_commented_pure_prose_no_phantom_nodes():
    # a commented prose block must not parse into DML -> no table/column phantoms
    block = (
        "exec $DB.P();\n"
        "-- this routine reconciles balances and notifies the operations team\n"
        "-- it does not modify any table directly\n"
    )
    g = _graph_parses([_sql_fp("p.sql", block, "sql")], {"p.sql": ".sql"})
    assert [n for n in g["nodes"] if n["type"] in ("table", "column")] == []


def test_commented_reuses_active_nodes_and_tags_only_unique():
    # active SQL writes db.shared(col_x); a comment adds db.shared(col_x, col_y).
    # db.shared + db.shared.col_x are REUSED (active, untagged); only the
    # comment-unique col_y is created tagged, and its contains edge is tagged.
    block = (
        "INSERT INTO db.shared (col_x) SELECT a.col_x FROM db.src a;\n"
        "exec $DB.P();\n"
        "-- insert into db.shared (col_x, col_y) select b.col_x, b.col_y from db.other b ;\n"
    )
    g = _graph_parses([_sql_fp("load.sql", block, "sql")], {"load.sql": ".sql"})
    shared = [n for n in g["nodes"] if n["type"] == "table" and n["name"] == "db.shared"]
    assert len(shared) == 1 and "commented" not in shared[0]["tags"]      # reused active table
    sh_colx = [n for n in g["nodes"] if n["id"] == "column:db.shared.col_x"]
    assert len(sh_colx) == 1 and "commented" not in sh_colx[0]["tags"]    # reused active column
    sh_coly = next((n for n in g["nodes"] if n["id"] == "column:db.shared.col_y"), None)
    assert sh_coly is not None and "commented" in sh_coly["tags"]         # comment-unique column
    coly_contains = [e for e in g["edges"]
                     if e["target"] == sh_coly["id"] and e["type"] == "contains"]
    assert len(coly_contains) == 1 and coly_contains[0].get("commented") is True


def test_commented_credential_not_leaked_into_graph():
    # a credential in a comment never reaches a node name/summary/tag; recovery of
    # the surrounding SQL still works (names are bare identifiers), and the redactor
    # catches the comment-prefixed .LOGON form in raw content.
    from app.enrich import sanitize_for_enrichment
    block = (
        "exec $DB.PROC();\n"
        "-- .LOGON tdpid/svcacct,SUPERSECRETPW\n"
        ";\n"
        "-- insert into db.audit (uid) select s.uid from db.src s "
        "where s.api_key = 'SECRETTOKENVALUE' ;\n"
    )
    g = _graph_parses([_sql_fp("c.sql", block, "sql")], {"c.sql": ".sql"})
    blob = json.dumps(g)
    for secret in ("SUPERSECRETPW", "SECRETTOKENVALUE", "svcacct", "tdpid"):
        assert secret not in blob, f"credential leaked into graph: {secret}"
    audit = next((n for n in g["nodes"] if n["type"] == "table" and n["name"] == "db.audit"), None)
    assert audit is not None and "commented" in audit["tags"]            # recovery still works
    assert "SUPERSECRETPW" not in sanitize_for_enrichment(block)         # redactor catches it


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
