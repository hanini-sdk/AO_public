"""Analysis orchestration: scan → parse → enrich → assemble → validate.

Runs in a background thread and publishes thread-safe progress that the browser
polls. The single network egress (LLM enrichment) happens through the guarded
client built in ``llm.py``; every other phase is fully local and deterministic.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

from . import diagnostics as diag_mod
from . import enrich as enrich_mod
from . import graph as graph_mod
from . import parser as parser_mod
from . import scanner as scanner_mod
from .config import DATA_DIR, Settings
from .llm import LLMClient
from .validate import validate_graph

log = logging.getLogger("data_lineage_retro_documentation.pipeline")

GRAPH_PATH = DATA_DIR / "knowledge-graph.json"
DASHBOARD_CONFIG_PATH = DATA_DIR / "config.json"
STORY_PATH = DATA_DIR / "story.json"
# Sidecar document-context store (prose summaries + per-node associations).
# Local-only artifact under the gitignored DATA_DIR — never tracked/committed.
PROJECT_CONTEXT_PATH = DATA_DIR / "project_context.json"
DIAG_JSON_PATH = DATA_DIR / "diagnostics.json"
DIAG_MD_PATH = DATA_DIR / "diagnostics.md"


@dataclass
class Progress:
    status: str = "idle"          # idle | running | done | error
    phase: str = ""
    current_file: str = ""
    processed: int = 0
    total: int = 0
    percent: float = 0.0
    message: str = ""
    error: str = ""
    project_name: str = ""
    stats: dict | None = None


class AnalysisManager:
    """Owns the single background analysis run + its progress state."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._progress = Progress()
        self._thread: threading.Thread | None = None
        self.project_root: str | None = None  # remembered for /file-content.json

    # ----------------------------------------------------------- progress
    def _set(self, **kw) -> None:
        with self._lock:
            for k, v in kw.items():
                setattr(self._progress, k, v)

    def get(self) -> dict:
        with self._lock:
            return asdict(self._progress)

    def is_running(self) -> bool:
        with self._lock:
            return self._progress.status == "running"

    # -------------------------------------------------------------- start
    def start(self, project_root: str, project_name: str, settings: Settings) -> tuple[bool, str]:
        if self.is_running():
            return False, "An analysis is already running."
        root = Path(project_root)
        if not root.is_dir():
            return False, f"Not a directory: {project_root}"
        with self._lock:
            self._progress = Progress(status="running", phase="Starting", project_name=project_name)
        self._thread = threading.Thread(
            target=self._run, args=(str(root.resolve()), project_name, settings), daemon=True
        )
        self._thread.start()
        return True, "Analysis started."

    # ---------------------------------------------------------------- run
    def _run(self, project_root: str, project_name: str, settings: Settings) -> None:
        llm: LLMClient | None = None
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)

            # 0) project tree inventory (structure only; opens zero files).
            # Written first so it exists even if a later stage fails. Best-effort.
            tree_ext_counts = None
            try:
                from . import project_tree
                tree_ext_counts = project_tree.write_project_tree(
                    project_root, DATA_DIR / "project_tree.txt"
                )
            except Exception as exc:  # noqa: BLE001 - inventory is best-effort
                log.debug("project tree inventory failed: %s", exc)

            # 1) scan ---------------------------------------------------
            self._set(phase="Scanning files", percent=2.0, current_file="")
            scan_stats: dict = {}
            scanned = scanner_mod.scan(project_root, settings, stats=scan_stats)
            if not scanned:
                self._set(status="error", error="No analysable source files were found in the selected project.")
                return
            self._set(total=len(scanned), message=f"Found {len(scanned)} files")

            # 2) parse (deterministic, no LLM) --------------------------
            self._set(phase="Parsing structure", percent=6.0)
            parses: list[parser_mod.FileParse] = []
            abs_by_rel: dict[str, str] = {}
            for i, sf in enumerate(scanned):
                parses.append(parser_mod.parse_file(sf))
                abs_by_rel[sf.rel_path] = sf.abs_path
                if i % 10 == 0 or i == len(scanned) - 1:
                    self._set(percent=6.0 + 18.0 * (i + 1) / len(scanned),
                              current_file=sf.rel_path, processed=i + 1)

            # 3) enrich via LLMAAS (the single network egress) ----------
            self._set(phase="Enriching with LLMAAS", percent=25.0, processed=0, current_file="")
            llm = LLMClient(settings)

            def on_progress(done: int, total: int, rel: str) -> None:
                self._set(processed=done, total=total, current_file=rel,
                          percent=25.0 + 65.0 * done / max(total, 1))

            enrichments = enrich_mod.enrich_files(llm, parses, abs_by_rel, settings, on_progress)

            # 4) assemble ----------------------------------------------
            self._set(phase="Assembling graph", percent=92.0, current_file="")
            frameworks = graph_mod.detect_frameworks(project_root)
            languages = sorted({sf.language for sf in scanned})
            file_summaries = [
                (p.rel_path, enrichments[p.rel_path].summary)
                for p in parses if p.rel_path in enrichments
            ]
            description = enrich_mod.summarize_project(
                llm, project_name, file_summaries, languages, frameworks, settings
            )
            ref_collect: dict = {}
            graph = graph_mod.build_graph(
                project_name=project_name,
                project_root=project_root,
                scanned=scanned,
                parses=parses,
                enrichments=enrichments,
                settings=settings,
                description=description,
                frameworks=frameworks,
                diagnostics=ref_collect,
            )

            # 5) validate ----------------------------------------------
            self._set(phase="Validating graph", percent=96.0)
            graph, vreport = validate_graph(graph)
            for issue in vreport.issues:
                log.info("validation: %s", issue)

            # 5b) variable-dictionary enrichment — attach business labels from
            # any tabular dictionary files (CSV/Excel) in the project onto the
            # matching table/column nodes. Deterministic, fully local: no LLM,
            # no egress, no new nodes/edges — only string attributes added to
            # existing nodes. Best-effort: a failure never discards the graph.
            self._set(phase="Applying variable dictionary", percent=97.0)
            try:
                from . import dictionary as dict_mod
                dict_stats = dict_mod.enrich_nodes_with_dictionaries(graph, project_root)
                log.info("variable dictionary: %s", dict_stats)
            except Exception as exc:  # noqa: BLE001 - enrichment is best-effort
                log.debug("variable dictionary enrichment failed: %s", exc)

            # 6) write outputs (no secret is ever written here) ---------
            # The graph is written + served HERE; the dashboard fetches it
            # directly (no dependency on run status). Everything below — including
            # the document phase — runs AFTER this point, so the graph and the
            # dashboard are available without waiting for any of it.
            GRAPH_PATH.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
            DASHBOARD_CONFIG_PATH.write_text(
                json.dumps({"autoUpdate": False, "outputLanguage": settings.language}, ensure_ascii=False),
                encoding="utf-8",
            )

            # 6a) document context (sidecar) — discover prose docs, summarize them
            # through the SAME guarded client + redactor into project_context.json,
            # and associate them to nodes by exact name. Runs AFTER the graph is
            # written (so the graph/dashboard never wait on it), NEVER touches the
            # graph topology, is bounded by a strict LLM call budget, and is fully
            # best-effort: any failure or slowness leaves the graph + dashboard
            # intact. Placed before the story so the Learn narrative can use it.
            self._set(phase="Reading project documents", percent=97.5)
            doc_context = None
            doc_stats = None
            try:
                from . import documents as doc_mod
                docs, disc_stats = doc_mod.discover_documents(project_root)
                tree_text = doc_mod.load_project_tree_text(DATA_DIR)
                doc_context, build_stats = doc_mod.build_context_store(
                    graph, docs, tree_text, llm, settings
                )
                PROJECT_CONTEXT_PATH.write_text(
                    json.dumps(doc_context, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                doc_stats = {**disc_stats, **build_stats}
                log.info("document context: %s", doc_stats)
            except Exception as exc:  # noqa: BLE001 - document phase is best-effort
                log.debug("document phase failed: %s", exc)

            # 6b) project story — one ordered narrative synthesized from the
            # graph's node summaries + lineage (never raw source) via the SAME
            # llm client, passed through the SAME credential redactor. Regenerated
            # whenever analysis runs (i.e. when the graph changes). Non-fatal: a
            # story failure must never discard the just-written graph.
            self._set(phase="Writing project story", percent=98.0)
            story_ok = False
            try:
                from . import story as story_mod
                story = story_mod.generate_story(graph, llm, settings, doc_context=doc_context)
                STORY_PATH.write_text(
                    json.dumps(story, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                story_ok = True
            except Exception as exc:  # noqa: BLE001 - story is best-effort, graph already saved
                log.debug("story generation failed: %s", exc)

            # 6c) run diagnostics — a deterministic structural-health report with
            # ZERO identifiers (aggregate counts + DLRD's own vocabulary only).
            # Local file, no LLM, no egress. Best-effort: never discard the graph.
            self._set(phase="Writing run diagnostics", percent=99.0)
            diag = None
            try:
                scan_stats["scanned"] = len(scanned)
                scan_stats["by_language"] = diag_mod.language_counts(scanned)
                diag = diag_mod.build_diagnostics(
                    graph,
                    scan_stats=scan_stats,
                    parse_stats=diag_mod.parse_outcomes(parses),
                    enrich_stats=diag_mod.enrich_outcomes(enrichments),
                    ref_stats=ref_collect.get("references", {}),
                    sql_stats=diag_mod.sql_outcomes(parses),
                )
                DIAG_JSON_PATH.write_text(
                    json.dumps(diag, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                DIAG_MD_PATH.write_text(diag_mod.render_diagnostics_md(diag), encoding="utf-8")
            except Exception as exc:  # noqa: BLE001 - diagnostics are best-effort
                log.debug("diagnostics generation failed: %s", exc)

            # 6d) build the local embedding index for the chat's semantic search.
            # Pure local computation — no model call, no egress. Requires the
            # optional embedding backend to be installed and its model cached;
            # silently skipped otherwise so the analysis never fails on a missing
            # optional dependency. The chat falls back to lexical search when the
            # index is absent.
            self._set(phase="Building embedding index", percent=99.5)
            embed_skipped = True
            try:
                from . import embeddings as embed_mod
                embed_skipped = embed_mod.EmbeddingIndex.build(graph.get("nodes", [])) is None
            except ImportError:
                log.info("embedding backend not installed — index skipped (lexical chat fallback active).")
            except Exception as exc:  # noqa: BLE001 - embeddings are best-effort
                log.debug("embedding index failed: %s", exc)

            # 6e) sanitized run report (review-safe; counts + fixed templates only).
            # Reuses the run-diagnostics counters; per-extension and column-lineage
            # counts are tallied here (counts only, never names). Best-effort.
            self._set(phase="Writing run report", percent=99.8)
            try:
                from . import run_report as run_report_mod
                if diag is not None:
                    scanned_by_ext: dict = {}
                    parsed_by_ext: dict = {}
                    for sf, p in zip(scanned, parses):
                        scanned_by_ext[sf.ext] = scanned_by_ext.get(sf.ext, 0) + 1
                        if getattr(p, "parse_ok", False):
                            parsed_by_ext[sf.ext] = parsed_by_ext.get(sf.ext, 0) + 1
                    col_ids = {n["id"] for n in graph.get("nodes", []) if n.get("type") == "column"}
                    col_lineage_edges = sum(
                        1 for e in graph.get("edges", [])
                        if e.get("type") == "reads_from"
                        and e.get("source") in col_ids and e.get("target") in col_ids
                    )
                    run_report_mod.write_run_report(
                        DATA_DIR / "run_report.txt",
                        project_root,
                        diag=diag,
                        ext_counts=tree_ext_counts or {},
                        scanned_by_ext=scanned_by_ext,
                        parsed_by_ext=parsed_by_ext,
                        column_lineage_edges=col_lineage_edges,
                        story_ok=story_ok,
                        embed_skipped=embed_skipped,
                        doc_stats=doc_stats,
                    )
            except Exception as exc:  # noqa: BLE001 - run report is best-effort
                log.debug("run report failed: %s", exc)

            self.project_root = project_root
            self._set(status="done", phase="Done", percent=100.0, current_file="",
                      message="Analysis complete.", stats=vreport.stats)
            log.info("analysis complete: %s", vreport.stats)
        except Exception as exc:  # noqa: BLE001
            log.exception("analysis failed")
            self._set(status="error", error=f"{type(exc).__name__}: {exc}")
        finally:
            if llm is not None:
                llm.close()


# Module-level singleton used by the server.
manager = AnalysisManager()
