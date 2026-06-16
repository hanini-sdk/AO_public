"""FastAPI backend.

Responsibilities:
  * serve the config/analysis shell (web/) and the compiled real dashboard
    (web/dashboard/) as static files;
  * expose the data endpoints the dashboard fetches (/knowledge-graph.json,
    /file-content.json, /config.json, ...);
  * expose the small JSON API the shell uses (config, test-connection, analyze,
    progress).

Security posture: binds to 127.0.0.1 only (see __main__.py), sends a strict CSP
that forbids any external network use by the page, never logs file contents or
the api key, never executes analysed code, and protects /file-content.json with
path-traversal checks + a graph allow-list. The ONLY outbound network the whole
process makes is the guarded LLMAAS client in llm.py.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import os
import shutil
import uuid
import zipfile
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response

from .config import DASHBOARD_DIR, DATA_DIR, UPLOADS_DIR, WEB_DIR, Settings
from .llm import LLMClient
from .pipeline import GRAPH_PATH, STORY_PATH, manager

log = logging.getLogger("data_lineage_retro_documentation.server")

# Ensure correct MIME types for the bundled static assets (Windows registries
# often miss these), so the strict nosniff header never rejects a font/script.
for _ext, _type in {
    ".js": "application/javascript", ".mjs": "application/javascript",
    ".css": "text/css", ".svg": "image/svg+xml",
    ".woff": "font/woff", ".woff2": "font/woff2",
    ".json": "application/json",
}.items():
    mimetypes.add_type(_type, _ext)

app = FastAPI(title="Data-Lineage and Retro-Documentation (local)", docs_url=None, redoc_url=None, openapi_url=None)

_CSP = (
    "default-src 'self'; "
    "script-src 'self'; "
    "style-src 'self' 'unsafe-inline'; "   # React inline styles + Tailwind
    "img-src 'self' data:; "
    "font-src 'self' data:; "
    "connect-src 'self'; "                 # blocks the page from any external host
    "worker-src 'self' blob:; "
    "object-src 'none'; base-uri 'self'; form-action 'self'; frame-ancestors 'none'"
)

# Source-preview limit (mirrors the upstream dev server).
_MAX_SOURCE_FILE_BYTES = 1024 * 1024
_SOURCE_LANG_BY_EXT = {
    "py": "python", "js": "javascript", "mjs": "javascript", "jsx": "jsx",
    "ts": "typescript", "tsx": "tsx", "java": "java", "go": "go", "rs": "rust",
    "c": "c", "h": "c", "cc": "cpp", "cpp": "cpp", "hpp": "cpp", "cs": "csharp",
    "rb": "ruby", "php": "php", "json": "json", "yaml": "yaml", "yml": "yaml",
    "md": "markdown", "html": "markup", "css": "css", "sh": "bash", "sql": "sql",
    "toml": "toml", "xml": "markup", "txt": "text",
}


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Content-Security-Policy"] = _CSP
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Frame-Options"] = "DENY"
    return response


# ============================================================ shell (web/) ====
@app.get("/")
async def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/app.js")
async def shell_js() -> FileResponse:
    return FileResponse(WEB_DIR / "app.js", media_type="application/javascript")


@app.get("/styles.css")
async def shell_css() -> FileResponse:
    return FileResponse(WEB_DIR / "styles.css", media_type="text/css")


@app.get("/favicon.svg")
async def favicon() -> Response:
    f = DASHBOARD_DIR / "favicon.svg"
    if f.exists():
        return FileResponse(f, media_type="image/svg+xml")
    return Response(status_code=204)


# ============================================ data endpoints for dashboard ====
def _serve_data_file(path: Path) -> JSONResponse:
    if not path.exists():
        raise HTTPException(status_code=404, detail="Not found")
    try:
        return JSONResponse(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to read data file")


@app.get("/knowledge-graph.json")
async def knowledge_graph() -> JSONResponse:
    if not GRAPH_PATH.exists():
        return JSONResponse({"error": "No knowledge graph yet. Run an analysis first."}, status_code=404)
    return _serve_data_file(GRAPH_PATH)


@app.get("/config.json")
async def dashboard_config() -> JSONResponse:
    cfg = DATA_DIR / "config.json"
    if cfg.exists():
        return _serve_data_file(cfg)
    return JSONResponse({"autoUpdate": False, "outputLanguage": "en"})


@app.get("/meta.json")
async def dashboard_meta() -> Response:
    meta = DATA_DIR / "meta.json"
    if meta.exists():
        return _serve_data_file(meta)
    return Response(status_code=404)  # optional file; the dashboard tolerates this


@app.get("/story.json")
async def project_story() -> Response:
    if STORY_PATH.exists():
        return _serve_data_file(STORY_PATH)
    return Response(status_code=404)  # optional cached artifact; the dashboard tolerates this


@app.get("/domain-graph.json")
async def domain_graph() -> Response:
    return Response(status_code=404)  # v2 feature, not generated


@app.get("/diff-overlay.json")
async def diff_overlay() -> Response:
    return Response(status_code=404)  # v2 feature, not generated


_graph_files_cache: tuple[float, set[str]] | None = None


def _graph_file_set() -> set[str]:
    """Set of node filePaths in the current graph (the source-preview allow-list)."""
    global _graph_files_cache
    if not GRAPH_PATH.exists():
        return set()
    mtime = GRAPH_PATH.stat().st_mtime
    if _graph_files_cache and _graph_files_cache[0] == mtime:
        return _graph_files_cache[1]
    paths: set[str] = set()
    try:
        data = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
        for n in data.get("nodes", []):
            fp = n.get("filePath")
            if isinstance(fp, str):
                paths.add(fp)
    except Exception:
        paths = set()
    _graph_files_cache = (mtime, paths)
    return paths


@app.get("/file-content.json")
async def file_content(path: str = "") -> JSONResponse:
    """Serve a source file for the CodeViewer — scoped to the analysed project.

    Defends against path traversal and only serves files that appear in the
    knowledge graph. Never serves anything outside the analysed project root.
    """
    project_root = manager.project_root
    if not project_root:
        return JSONResponse({"error": "No analysis has been run yet."}, status_code=404)
    if not path or "\0" in path or os.path.isabs(path):
        return JSONResponse({"error": "Invalid path"}, status_code=400)

    norm = os.path.normpath(path)
    if norm in (".", "..") or norm.startswith(".." + os.sep) or os.path.isabs(norm):
        return JSONResponse({"error": "Path must stay inside the project"}, status_code=400)

    root_real = os.path.realpath(project_root)
    abs_file = os.path.realpath(os.path.join(root_real, norm))
    if abs_file != root_real and not abs_file.startswith(root_real + os.sep):
        return JSONResponse({"error": "Path must stay inside the project"}, status_code=400)

    rel_posix = os.path.relpath(abs_file, root_real).replace(os.sep, "/")
    if rel_posix not in _graph_file_set():
        return JSONResponse({"error": "File is not in the knowledge graph"}, status_code=404)

    try:
        stat = os.stat(abs_file)
    except OSError:
        return JSONResponse({"error": "File not found"}, status_code=404)
    if not os.path.isfile(abs_file):
        return JSONResponse({"error": "Not a file"}, status_code=400)
    if stat.st_size > _MAX_SOURCE_FILE_BYTES:
        return JSONResponse({"error": "File is too large to preview"}, status_code=413)

    raw = Path(abs_file).read_bytes()
    if b"\x00" in raw:
        return JSONResponse({"error": "Binary files cannot be previewed"}, status_code=415)
    content = raw.decode("utf-8", "replace")
    ext = rel_posix.rsplit(".", 1)[-1].lower() if "." in rel_posix else ""
    return JSONResponse({
        "path": rel_posix,
        "language": _SOURCE_LANG_BY_EXT.get(ext, "text"),
        "content": content,
        "sizeBytes": len(raw),
        "lineCount": 0 if not content else content.count("\n") + 1,
    })


# ============================================================== shell API =====
@app.get("/api/status")
async def api_status() -> dict:
    s = Settings.load()
    return {
        "configured": s.is_configured(),
        "running": manager.is_running(),
        "has_graph": GRAPH_PATH.exists(),
        "project_name": manager.get().get("project_name", ""),
    }


@app.get("/api/config")
async def api_get_config() -> dict:
    return Settings.load().to_public_dict()


@app.post("/api/config")
async def api_save_config(payload: dict) -> dict:
    s = Settings.load()
    s.update_public(payload)
    s.set_api_key(payload.get("api_key"))
    s.save()
    return s.to_public_dict()


@app.post("/api/test-connection")
async def api_test_connection(payload: dict) -> dict:
    s = Settings.load()
    s.update_public(payload)
    s.set_api_key(payload.get("api_key"))
    if not (s.api_base and s.api_key and s.model):
        return {"ok": False, "message": "Please fill in apiBase, apiKey and model first."}
    llm = LLMClient(s)
    try:
        ok, message = llm.test_connection()
        return {"ok": ok, "message": message}
    finally:
        llm.close()


@app.post("/api/analyze/folder")
async def api_analyze_folder(payload: dict) -> dict:
    s = Settings.load()
    if not s.is_configured():
        raise HTTPException(status_code=400, detail="Not configured. Save apiBase/apiKey/model first.")
    folder = (payload.get("path") or "").strip()
    if not folder:
        raise HTTPException(status_code=400, detail="Please provide a folder path.")
    root = Path(folder).expanduser()
    if not root.is_dir():
        raise HTTPException(status_code=400, detail=f"Not a directory: {folder}")
    name = (payload.get("name") or root.name or "project").strip()
    ok, message = manager.start(str(root.resolve()), name, s)
    if not ok:
        raise HTTPException(status_code=409, detail=message)
    return {"ok": True, "message": message, "project_name": name}


def _safe_extract_zip(zip_path: Path, dest: Path) -> None:
    """Extract a zip with zip-slip protection (never write outside dest)."""
    dest_real = os.path.realpath(dest)
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            name = member.filename
            if not name or name.endswith("/"):
                continue
            if os.path.isabs(name) or "\0" in name:
                continue
            target = os.path.realpath(os.path.join(dest_real, name))
            if target != dest_real and not target.startswith(dest_real + os.sep):
                continue  # skip path-traversal entries
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with zf.open(member) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)


def _project_root_in(extract_dir: Path) -> Path:
    """If the zip has a single top-level folder, treat that as the project root."""
    entries = [e for e in extract_dir.iterdir() if not e.name.startswith("__MACOSX")]
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return extract_dir


@app.post("/api/analyze/zip")
async def api_analyze_zip(file: UploadFile, name: str = "") -> dict:
    s = Settings.load()
    if not s.is_configured():
        raise HTTPException(status_code=400, detail="Not configured. Save apiBase/apiKey/model first.")
    if not (file.filename or "").lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Please upload a .zip file.")

    # Fresh uploads workspace each run (previous uploads are cleaned up).
    if UPLOADS_DIR.exists():
        shutil.rmtree(UPLOADS_DIR, ignore_errors=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

    job = uuid.uuid4().hex
    zip_path = UPLOADS_DIR / f"{job}.zip"
    extract_dir = UPLOADS_DIR / job
    extract_dir.mkdir(parents=True, exist_ok=True)
    try:
        with open(zip_path, "wb") as out:
            shutil.copyfileobj(file.file, out)
        _safe_extract_zip(zip_path, extract_dir)
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid zip archive.")
    finally:
        zip_path.unlink(missing_ok=True)  # the archive itself is not needed after extraction

    root = _project_root_in(extract_dir)
    default_name = Path(file.filename).stem if file.filename else root.name
    name = (name or default_name or "project").strip()
    ok, message = manager.start(str(root.resolve()), name, s)
    if not ok:
        raise HTTPException(status_code=409, detail=message)
    return {"ok": True, "message": message, "project_name": name}


@app.post("/api/regenerate-story")
async def api_regenerate_story(force: bool = False) -> dict:
    """(Re)generate the cached project story from the EXISTING graph, reusing the
    single guarded LLM client + the existing redactor. Serves the cache when it
    is still in sync with the graph unless force=true. No new egress path — the
    only outbound is the existing LLMAAS client; no rescan, no raw source sent.
    """
    if manager.is_running():
        raise HTTPException(status_code=409, detail="An analysis is running. Try again when it finishes.")
    if not GRAPH_PATH.exists():
        raise HTTPException(status_code=404, detail="No knowledge graph yet. Run an analysis first.")
    try:
        graph = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to read the knowledge graph.")

    from . import story as story_mod
    fingerprint = story_mod.graph_fingerprint(graph)
    # Serve a fresh cached story without touching the LLM (no config needed).
    if not force and STORY_PATH.exists():
        try:
            cached = json.loads(STORY_PATH.read_text(encoding="utf-8"))
            if cached.get("graphFingerprint") == fingerprint:
                return {"ok": True, "cached": True, "story": cached}
        except Exception:
            pass  # unreadable/stale cache -> regenerate below

    # Regeneration is the only path that calls the model — require config here.
    s = Settings.load()
    if not s.is_configured():
        raise HTTPException(status_code=400, detail="Not configured. Save apiBase/apiKey/model first.")
    llm = LLMClient(s)
    try:
        story = story_mod.generate_story(graph, llm, s)
    finally:
        llm.close()
    STORY_PATH.write_text(json.dumps(story, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True, "cached": False, "story": story}


@app.get("/api/progress")
async def api_progress() -> dict:
    return manager.get()


# ============================================== dashboard static (mounted) ====
def mount_dashboard() -> None:
    """Mount the compiled dashboard if it has been built into web/dashboard/."""
    from fastapi.staticfiles import StaticFiles

    if (DASHBOARD_DIR / "index.html").exists():
        app.mount("/dashboard", StaticFiles(directory=str(DASHBOARD_DIR), html=True), name="dashboard")
        log.info("dashboard mounted from %s", DASHBOARD_DIR)
    else:
        @app.get("/dashboard")
        @app.get("/dashboard/{rest:path}")
        async def dashboard_missing(rest: str = "") -> JSONResponse:  # pragma: no cover
            return JSONResponse(
                {"error": "Dashboard not built. Build it into web/dashboard/ (see README)."},
                status_code=503,
            )


mount_dashboard()
