"""Tests for app/story.py — project story synthesis from the graph.

Self-contained: run directly with `python tests/test_story.py` or via `pytest`.
Drives the real build_graph pipeline on synthetic SQL, then story.generate_story
with stub/dead LLM clients (no network). Asserts the section ordering + per-section
node ids, the deterministic fallback when the LLM is unavailable, the content
fingerprint, and that no credential ever reaches the story.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app import sqlproc, story                  # noqa: E402
from app.graph import build_graph               # noqa: E402
from app.scanner import ScannedFile             # noqa: E402
from app.parser import FileParse                # noqa: E402
from app.enrich import FileEnrichment           # noqa: E402
from app.config import Settings                 # noqa: E402


class _StubLLM:
    """Returns canned prose so the section/overview paths run without a network."""
    def __init__(self):
        self.calls = 0

    def chat(self, messages, max_tokens=None, temperature=None):
        self.calls += 1
        return "Prose: " + (messages[-1]["content"][:30].replace("\n", " "))


class _DeadLLM:
    """Always fails — exercises the deterministic fallback in every section."""
    def chat(self, *a, **k):
        raise RuntimeError("llm down")


def _graph(files, shell=None):
    scanned, parses, enrich = [], [], {}
    for rel, txt in files:
        scanned.append(ScannedFile(abs_path=rel, rel_path=rel, ext=".sql", language="sql", size_bytes=len(txt)))
        parses.append(FileParse(rel, "sql", None, True, [], [], [], {}, sql=sqlproc.extract(txt)))
        enrich[rel] = FileEnrichment(summary=f"summary of {rel}", layer="Data", complexity="simple",
                                     tags=["sql"], members={}, llm_ok=False)
    for rel, src in (shell or []):
        scanned.append(ScannedFile(abs_path=rel, rel_path=rel, ext=".sh", language="shell", size_bytes=len(src)))
        parses.append(FileParse(rel, "shell", "shell", True, [], [], [], {}, sql=None))
        enrich[rel] = FileEnrichment(summary=f"summary of {rel}", layer="Utility", complexity="simple",
                                     tags=["shell"], members={}, llm_ok=False)
    return build_graph(project_name="ETL", project_root=".", scanned=scanned, parses=parses,
                       enrichments=enrich, settings=Settings(), description="A sales ETL.", frameworks=[])


_FILES = [
    ("dim.sql", "CREATE TABLE wh.dim AS (SELECT s.a AS id FROM stg.raw s) WITH DATA;"),
    ("fact.sql", "INSERT INTO wh.fact (id) SELECT d.id FROM wh.dim d JOIN stg.ext e ON e.id=d.id;"),
]


def test_sections_ordered_and_have_node_ids():
    g = _graph(_FILES)
    st = story.generate_story(g, _StubLLM(), Settings())
    ids = [s["id"] for s in st["sections"]]
    assert ids[0] == "overview"
    assert ids[-1] == "glossary"
    for sid in ("sources", "transformations", "outputs"):
        assert sid in ids
    # every non-overview section records the ids of the nodes it describes
    by_id = {s["id"]: s for s in st["sections"]}
    assert by_id["sources"]["nodeIds"]          # stg.raw / stg.ext
    assert by_id["transformations"]["nodeIds"]  # wh.dim
    assert by_id["outputs"]["nodeIds"]          # wh.fact


def test_classification_source_transform_sink():
    g = _graph(_FILES)
    st = story.generate_story(g, _StubLLM(), Settings())
    by_id = {s["id"]: s for s in st["sections"]}
    names = {n["id"]: n["name"] for n in g["nodes"]}

    def section_names(sid):
        return {names[i] for i in by_id[sid]["nodeIds"] if i in names}

    assert {"stg.raw", "stg.ext"} <= section_names("sources")   # external inputs
    assert "wh.dim" in section_names("transformations")          # produced and read
    assert "wh.fact" in section_names("outputs")                 # produced, not read


def test_glossary_lists_tables_and_columns_without_llm():
    g = _graph(_FILES)
    st = story.generate_story(g, _StubLLM(), Settings())
    glossary = next(s for s in st["sections"] if s["id"] == "glossary")
    assert "wh.dim" in glossary["body"] and "columns:" in glossary["body"]


def test_fallback_when_llm_unavailable():
    g = _graph(_FILES)
    st = story.generate_story(g, _DeadLLM(), Settings())
    assert st["sections"]                                   # still produced
    overview = next(s for s in st["sections"] if s["id"] == "overview")
    assert overview["body"].strip()                        # non-empty deterministic body


def test_fingerprint_stable_and_stamped():
    g = _graph(_FILES)
    fp1 = story.graph_fingerprint(g)
    fp2 = story.graph_fingerprint(g)
    assert fp1 == fp2 and len(fp1) == 64
    st = story.generate_story(g, _StubLLM(), Settings())
    assert st["graphFingerprint"] == fp1


def test_no_credentials_in_story():
    # A bteq heredoc carrying a .LOGON credential must never surface in the story
    shell_src = "bteq <<EOF\n.LOGON tdpid/svc,SUPERSECRET\nINSERT INTO wh.t SELECT a FROM stg.s;\nEOF\n"
    from app import parser as P
    from tree_sitter import Language, Parser
    import tree_sitter_bash as tsb
    b = shell_src.encode("utf-8")
    tree = Parser(Language(tsb.language())).parse(b)
    _, _, variables = P._collect_shell_refs(tree.root_node, b)
    sqlres = P._extract_shell_sql(tree.root_node, b, variables)
    fp = FileParse("run.sh", "shell", "shell", True, [], [], [], {}, sql=sqlres)
    scanned = [ScannedFile(abs_path="run.sh", rel_path="run.sh", ext=".sh", language="shell", size_bytes=len(shell_src))]
    enrich = {"run.sh": FileEnrichment(summary="loader", layer="Utility", complexity="simple", tags=["shell"], members={}, llm_ok=False)}
    g = build_graph(project_name="X", project_root=".", scanned=scanned, parses=[fp],
                    enrichments=enrich, settings=Settings(), description="", frameworks=[])
    import json as _json
    st = story.generate_story(g, _StubLLM(), Settings())
    blob = _json.dumps(st)
    assert "SUPERSECRET" not in blob and "svc" not in blob and "LOGON" not in blob


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
