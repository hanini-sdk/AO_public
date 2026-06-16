"""Deterministic project scanner — no LLM, no code execution.

Walks a directory, detects languages by extension, and skips version-control
metadata, dependency/build directories, virtual environments, binaries, lock
files and oversized files. Returns plain metadata; file contents are read lazily
by the parser/enricher as UTF-8 text (never imported or executed).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from .config import Settings

# Directories we never descend into (matched by exact name).
IGNORE_DIRS = {
    "node_modules", "bower_components", "jspm_packages", "vendor",
    "dist", "build", "out", "target", "bin", "obj", ".next", ".nuxt",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox",
    "venv", ".venv", "env", ".env.d", "site-packages",
    "coverage", ".coverage", "htmlcov", ".cache", ".parcel-cache",
    ".gradle", ".terraform", "Pods", "DerivedData",
    "__snapshots__", ".turbo", ".svelte-kit",
}

# Whole files we skip (lock files / vendored bundles — large, low signal).
IGNORE_FILES = {
    "package-lock.json", "pnpm-lock.yaml", "yarn.lock", "npm-shrinkwrap.json",
    "poetry.lock", "Pipfile.lock", "Cargo.lock", "composer.lock", "Gemfile.lock",
    "go.sum",
}

# Binary / non-source extensions we never read.
BINARY_EXTS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp", ".tiff", ".svg",
    ".pdf", ".zip", ".tar", ".gz", ".tgz", ".bz2", ".7z", ".rar", ".xz",
    ".exe", ".dll", ".so", ".dylib", ".bin", ".dat", ".class", ".jar", ".war",
    ".pyc", ".pyo", ".o", ".a", ".lib", ".node", ".wasm", ".obj",
    ".woff", ".woff2", ".ttf", ".otf", ".eot",
    ".mp3", ".mp4", ".mov", ".avi", ".wav", ".flac", ".ogg", ".webm",
    ".min.js", ".min.css", ".map",
}

# Extension -> language label. The label feeds project.languages and the
# parser's grammar dispatch. Anything not listed but textual still becomes a
# file node (language "other").
EXT_LANGUAGE: dict[str, str] = {
    ".py": "python", ".pyi": "python",
    ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".mts": "typescript", ".cts": "typescript", ".tsx": "typescript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".c": "c", ".h": "c",
    ".cc": "cpp", ".cpp": "cpp", ".cxx": "cpp", ".hpp": "cpp", ".hh": "cpp", ".hxx": "cpp",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    # Recognized but not tree-sitter parsed (file nodes only):
    ".json": "json", ".yaml": "yaml", ".yml": "yaml", ".toml": "toml",
    ".md": "markdown", ".markdown": "markdown", ".rst": "rst",
    ".html": "html", ".htm": "html", ".vue": "vue", ".svelte": "svelte",
    ".css": "css", ".scss": "scss", ".sass": "sass", ".less": "less",
    ".sh": "shell", ".bash": "shell", ".zsh": "shell", ".ksh": "shell",
    ".list": "joblist",   # job manifest: one shell-script filename per line
    ".sql": "sql", ".bteq": "sql", ".btq": "sql",
    ".xml": "xml", ".kt": "kotlin", ".kts": "kotlin",
    ".swift": "swift", ".scala": "scala", ".lua": "lua", ".r": "r",
    ".pl": "perl", ".pm": "perl", ".dart": "dart", ".ex": "elixir", ".exs": "elixir",
    ".tf": "terraform", ".dockerfile": "dockerfile",
}

_NUL = b"\x00"


@dataclass
class ScannedFile:
    abs_path: str
    rel_path: str        # posix-style, relative to the project root
    ext: str
    language: str
    size_bytes: int


def detect_language(name: str) -> str | None:
    lower = name.lower()
    if lower == "dockerfile" or lower.startswith("dockerfile."):
        return "dockerfile"
    if lower == "makefile":
        return "makefile"
    ext = Path(lower).suffix
    # handle compound suffixes like .min.js / .d.ts already excluded as binary
    return EXT_LANGUAGE.get(ext)


def _looks_binary(path: str) -> bool:
    try:
        with open(path, "rb") as fh:
            return _NUL in fh.read(4096)
    except OSError:
        return True


# Shebang interpreters that mean "this is a shell script" (cheap insurance for
# extensionless files like `run_load` with `#!/usr/bin/env ksh`).
_SHEBANG_SHELL_RE = re.compile(rb"^#!.*\b(bash|ksh|zsh|dash|sh)\b")


def _shebang_language(path: str) -> str | None:
    try:
        with open(path, "rb") as fh:
            first = fh.readline(256)
    except OSError:
        return None
    if first.startswith(b"#!") and _SHEBANG_SHELL_RE.match(first):
        return "shell"
    return None


def scan(root: str | Path, settings: Settings, stats: dict | None = None) -> list[ScannedFile]:
    """Return the analysable files under ``root`` (deterministic ordering).

    ``stats`` (optional) is a tally-only out-param for run diagnostics; when
    provided it receives ``skipped`` (recognised-language files dropped as empty /
    unreadable / binary-content) and ``oversize`` (over the size limit) counts. It
    does not change which files are returned.
    """
    root_path = Path(root).resolve()
    results: list[ScannedFile] = []
    skipped = 0
    oversize = 0

    for dirpath, dirnames, filenames in os.walk(root_path):
        # Prune ignored / hidden directories in place so os.walk skips them.
        dirnames[:] = sorted(
            d for d in dirnames if d not in IGNORE_DIRS and not d.startswith(".")
        )
        for filename in sorted(filenames):
            if filename in IGNORE_FILES or filename.startswith("."):
                continue
            ext = Path(filename.lower()).suffix
            if ext in BINARY_EXTS or filename.lower().endswith((".min.js", ".min.css")):
                continue
            abs_path = os.path.join(dirpath, filename)
            language = detect_language(filename)
            if language is None and ext == "":
                # Extensionless file — route shell scripts via their shebang.
                language = _shebang_language(abs_path)
            if language is None:
                continue  # unrecognized type — not a source candidate, not counted
            # Recognised-language file from here: count any drop as skipped/oversize.
            try:
                size = os.path.getsize(abs_path)
            except OSError:
                skipped += 1
                continue
            if size == 0:
                skipped += 1
                continue
            if size > settings.max_file_bytes:
                oversize += 1
                continue
            if _looks_binary(abs_path):
                skipped += 1
                continue
            rel = os.path.relpath(abs_path, root_path).replace(os.sep, "/")
            results.append(
                ScannedFile(
                    abs_path=abs_path,
                    rel_path=rel,
                    ext=ext,
                    language=language,
                    size_bytes=size,
                )
            )
            if len(results) >= settings.max_files:
                if stats is not None:
                    stats["skipped"] = skipped
                    stats["oversize"] = oversize
                return results
    if stats is not None:
        stats["skipped"] = skipped
        stats["oversize"] = oversize
    return results
