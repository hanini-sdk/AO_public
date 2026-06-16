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

            # 6) write outputs (no secret is ever written here) ---------
            GRAPH_PATH.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
            DASHBOARD_CONFIG_PATH.write_text(
                json.dumps({"autoUpdate": False, "outputLanguage": settings.language}, ensure_ascii=False),
                encoding="utf-8",
            )

            # 6b) project story — one ordered narrative synthesized from the
            # graph's node summaries + lineage (never raw source) via the SAME
            # llm client, passed through the SAME credential redactor. Regenerated
            # whenever analysis runs (i.e. when the graph changes). Non-fatal: a
            # story failure must never discard the just-written graph.
            self._set(phase="Writing project story", percent=98.0)
            try:
                from . import story as story_mod
                story = story_mod.generate_story(graph, llm, settings)
                STORY_PATH.write_text(
                    json.dumps(story, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            except Exception as exc:  # noqa: BLE001 - story is best-effort, graph already saved
                log.debug("story generation failed: %s", exc)

            # 6c) run diagnostics — a deterministic structural-health report with
            # ZERO identifiers (aggregate counts + DLRD's own vocabulary only).
            # Local file, no LLM, no egress. Best-effort: never discard the graph.
            self._set(phase="Writing run diagnostics", percent=99.0)
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

            # 6d) build local embedding index for the RAG chat (semantic search).
            # Pure local computation — no LLM, no egress. Requires fastembed to be
            # installed and the model to be cached; silently skipped otherwise so
            # the analysis never fails because of a missing optional dependency.
            # The chat feature falls back to lexical search when the index is absent.
            self._set(phase="Building embedding index", percent=99.5)
            try:
                from . import embeddings as embed_mod
                embed_mod.EmbeddingIndex.build(graph.get("nodes", []))
            except ImportError:
                log.info("fastembed not installed — embedding index skipped (lexical RAG fallback active).")
            except Exception as exc:  # noqa: BLE001 - embeddings are best-effort
                log.debug("embedding index failed: %s", exc)

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
