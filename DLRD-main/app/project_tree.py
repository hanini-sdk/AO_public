"""Project tree inventory - a per-run plain-text artifact (structure only).

On each run, walk the analysed project directory and write project_tree.txt to
the run-output directory (gitignored). The walk uses directory metadata ONLY
(names + tree shape): it opens zero files and reads zero file bytes. Every entry
is included with no filtering - hidden files, dot-directories, vendored
directories - because the inventory's purpose is a complete structural census.

The file ends with an EXTENSIONS section: every distinct file extension with its
count, flagged HANDLED (a structural parser exists in the pipeline) or UNHANDLED
(no parser yet). ``handled_extensions`` is the single source of truth for that
classification, shared with the sanitized run report so the two artifacts agree.

No LLM and no network: derived from directory metadata, written to a local file.
"""

from __future__ import annotations

import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

_NO_EXT_LABEL = "(no extension)"
_MAX_TREE_DEPTH = 64   # defensive bound against pathological nesting


def handled_extensions() -> set[str]:
    """Extensions the pipeline has a structural parser for - the single source
    of truth shared by both run artifacts so their HANDLED/UNHANDLED flags agree.

    Derived from the parser's tree-sitter grammar map (which already includes
    shell) plus the extensions whose language is parsed by the dedicated SQL /
    job-list parsers. Everything else becomes a file node only (UNHANDLED).
    """
    from .parser import GRAMMAR_BY_EXT          # tree-sitter grammars (incl. shell)
    from .scanner import EXT_LANGUAGE
    handled = {e.lower() for e in GRAMMAR_BY_EXT}
    handled |= {ext.lower() for ext, lang in EXT_LANGUAGE.items()
                if lang in ("sql", "joblist")}
    return handled


def _ext_of(name: str) -> str:
    """Lowercase extension of a file name, or '' when it has none."""
    return os.path.splitext(name)[1].lower()


def _walk_tree(dir_path: str, prefix: str, depth: int,
               lines: list[str], ext_counts: Counter, totals: dict) -> None:
    """Append an indented listing of ``dir_path``. Directory metadata ONLY: it
    reads os.scandir entry types and never opens a file. Symlinks are listed but
    not followed (no cycles, no target reads)."""
    if depth > _MAX_TREE_DEPTH:
        lines.append(prefix + "[max depth reached]")
        return
    try:
        entries = list(os.scandir(dir_path))
    except OSError:
        lines.append(prefix + "[unreadable directory]")
        return
    dirs, files = [], []
    for e in entries:
        try:
            is_dir = e.is_dir(follow_symlinks=False)
        except OSError:
            is_dir = False
        (dirs if is_dir else files).append(e)
    dirs.sort(key=lambda e: e.name.lower())
    files.sort(key=lambda e: e.name.lower())
    for d in dirs:
        totals["dirs"] += 1
        lines.append(f"{prefix}{d.name}/")
        _walk_tree(d.path, prefix + "    ", depth + 1, lines, ext_counts, totals)
    for fentry in files:
        totals["files"] += 1
        lines.append(prefix + fentry.name)
        ext = _ext_of(fentry.name)
        ext_counts[ext if ext else _NO_EXT_LABEL] += 1


def build_project_tree(project_root: str) -> tuple[str, Counter]:
    """Return ``(tree_text, ext_counts)``.

    Pure directory-metadata walk - opens zero files, reads zero bytes.
    ``ext_counts`` maps each extension (or the no-extension label) to a file
    count and is reused by the run report so the two artifacts agree.
    """
    root = Path(project_root)
    root_label = root.name or str(root)
    ext_counts: Counter = Counter()
    totals = {"dirs": 0, "files": 0}
    body: list[str] = []
    _walk_tree(str(root), "    ", 1, body, ext_counts, totals)

    handled = handled_extensions()
    generated = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    out: list[str] = [
        "PROJECT TREE INVENTORY",
        f"Generated (UTC): {generated}",
        "Structure only: enumerated from directory metadata; no file was opened.",
        "",
        f"{root_label}/",
    ]
    out.extend(body)
    out += [
        "",
        f"Totals: {totals['dirs']} folders, {totals['files']} files.",
        "",
        "EXTENSIONS  (HANDLED = a structural parser exists in the pipeline; "
        "classified by extension)",
    ]
    for ext in sorted(e for e in ext_counts if e != _NO_EXT_LABEL):
        flag = "HANDLED" if ext in handled else "UNHANDLED"
        out.append(f"  {ext:<16} {ext_counts[ext]:>7}  {flag}")
    if _NO_EXT_LABEL in ext_counts:
        out.append(f"  {_NO_EXT_LABEL:<16} {ext_counts[_NO_EXT_LABEL]:>7}  UNHANDLED")
    return "\n".join(out) + "\n", ext_counts


def write_project_tree(project_root: str, out_path: Path) -> Counter:
    """Build and write project_tree.txt; return the extension census for reuse
    by the run report (so the two artifacts agree on extensions)."""
    text, ext_counts = build_project_tree(project_root)
    out_path.write_text(text, encoding="utf-8")
    return ext_counts
