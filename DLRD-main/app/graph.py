"""Assemble the knowledge graph in the exact format the real dashboard consumes.

Format derived from the upstream dashboard's Zod schema (core/src/schema.ts):
  { version, project{name,languages,frameworks,description,analyzedAt,gitCommitHash},
    nodes[], edges[], layers[], tour[] }
Layers are a TOP-LEVEL array grouping node ids (not a per-node field). No secret
(apiKey/apiBase) ever appears here.
"""

from __future__ import annotations

import json
import os
import posixpath
from datetime import datetime, timezone
from pathlib import Path

from .config import Settings
from .enrich import FileEnrichment
from .parser import FileParse, _SHELL_NON_SCRIPT
from .scanner import ScannedFile

VERSION = "1.0.0"

# Friendly metadata for the prompt's layer taxonomy, in display order.
LAYER_META: list[tuple[str, str, str]] = [
    ("API", "API Layer", "Entry points, routes, controllers, and request/response handling."),
    ("Service", "Service Layer", "Business logic, orchestration, and application services."),
    ("Data", "Data Layer", "Persistence, models, schemas, queries, and data access."),
    ("UI", "UI Layer", "User interface, components, views, and presentation."),
    ("Utility", "Utility Layer", "Shared helpers, utilities, configuration, and tooling."),
    ("Other", "Other", "Files that do not fit a single architectural layer."),
]
_LAYER_SLUG = {name: f"layer:{name.lower()}" for name, _, _ in LAYER_META}

_W_CONTAINS, _W_IMPORTS, _W_CALLS = 0.5, 0.7, 0.4
_W_LINEAGE = 0.8           # table<-table / file<->table data-flow edges
_W_FEEDS = 0.6             # file->file "feeds" (derived producer/consumer) edge
_MAX_SQL_LINEAGE_PER_FILE = 2000
# Phase E file->file "feeds" edges: skip tables touched by more than this many
# writer or reader files (hot/shared staging tables would otherwise explode into
# W*R edges), and cap how many feeds edges any single consumer file accumulates.
_MAX_FEED_FANOUT = 25
_MAX_FEEDS_PER_FILE = 50
_FEEDS_NAMES_SHOWN = 3     # name up to this many shared tables on a feeds edge, then +N
# C1/C2 column granularity: every used column (a column actually referenced in the
# scripts) becomes a node and is traceable. No per-table cap — a lineage tool must
# never silently drop columns. Columns are hidden by default (top-bar COL toggle).


def _feeds_desc(tables: list[str]) -> str:
    """Describe a feeds edge by the shared table(s): `feeds via X` or
    `feeds via T1, T2, T3, +N`. Only table names + the fixed word "feeds via"."""
    names = sorted(set(tables))
    shown = names[:_FEEDS_NAMES_SHOWN]
    extra = len(names) - len(shown)
    label = ", ".join(shown)
    if extra > 0:
        label += f", +{extra}"
    return "feeds via " + label


def _short(name: str) -> str:
    """Last dotted component of a table id (drop db/schema prefix for summaries)."""
    return name.rsplit(".", 1)[-1] if name else name
_MAX_CALLS_PER_FUNC = 20

# Manifest filename -> dependency keyword -> framework display name.
_FRAMEWORK_SIGNALS = {
    "react": "React", "next": "Next.js", "vue": "Vue", "svelte": "Svelte",
    "@angular/core": "Angular", "express": "Express", "fastify": "Fastify",
    "@nestjs/core": "NestJS", "vite": "Vite", "tailwindcss": "TailwindCSS",
    "zustand": "Zustand", "redux": "Redux",
    "fastapi": "FastAPI", "flask": "Flask", "django": "Django",
    "starlette": "Starlette", "uvicorn": "uvicorn", "pydantic": "Pydantic",
    "sqlalchemy": "SQLAlchemy", "pandas": "pandas", "numpy": "NumPy",
    "torch": "PyTorch", "tensorflow": "TensorFlow",
    "spring": "Spring", "gin-gonic/gin": "Gin", "actix": "Actix-web",
    "rails": "Rails", "laravel": "Laravel", "tree-sitter": "tree-sitter",
    "openai": "OpenAI SDK",
}
_MANIFESTS = [
    "package.json", "requirements.txt", "pyproject.toml", "setup.py", "setup.cfg",
    "Pipfile", "go.mod", "Cargo.toml", "pom.xml", "build.gradle", "build.gradle.kts",
    "Gemfile", "composer.json",
]


# --------------------------------------------------------------------------- git
def git_commit_hash(project_root: str | Path) -> str:
    """Read the current commit hash from .git without executing git. '' if none."""
    git_dir = Path(project_root) / ".git"
    head = git_dir / "HEAD"
    try:
        content = head.read_text(encoding="utf-8").strip()
    except OSError:
        return ""
    if not content.startswith("ref:"):
        return content  # detached HEAD
    ref = content[4:].strip()
    ref_path = git_dir / ref
    try:
        return ref_path.read_text(encoding="utf-8").strip()
    except OSError:
        pass
    packed = git_dir / "packed-refs"
    try:
        for line in packed.read_text(encoding="utf-8").splitlines():
            if line and not line.startswith(("#", "^")) and line.endswith(ref):
                return line.split()[0]
    except OSError:
        pass
    return ""


# -------------------------------------------------------------------- frameworks
def detect_frameworks(project_root: str | Path) -> list[str]:
    root = Path(project_root)
    found: list[str] = []

    def add(name: str) -> None:
        if name not in found:
            found.append(name)

    for manifest in _MANIFESTS:
        path = root / manifest
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if manifest in ("package.json", "composer.json"):
            try:
                data = json.loads(text)
                deps = {}
                for key in ("dependencies", "devDependencies", "require", "require-dev"):
                    if isinstance(data.get(key), dict):
                        deps.update(data[key])
                for dep in deps:
                    if dep in _FRAMEWORK_SIGNALS:
                        add(_FRAMEWORK_SIGNALS[dep])
            except Exception:
                pass
        else:
            low = text.lower()
            for needle, name in _FRAMEWORK_SIGNALS.items():
                if needle in low:
                    add(name)
    return found


# -------------------------------------------------------------- import resolution
_JS_EXTS = [".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"]
_SHELL_EXTS = (".sh", ".ksh", ".bash", ".zsh")


def _resolve_import(raw: str, importer_rel: str, grammar_key: str | None, fileset: set[str]) -> str | None:
    if not raw:
        return None
    importer_dir = posixpath.dirname(importer_rel)

    def first_in_set(candidates: list[str]) -> str | None:
        for c in candidates:
            norm = posixpath.normpath(c).lstrip("./")
            if norm in fileset and norm != importer_rel:
                return norm
        return None

    if grammar_key in ("javascript", "typescript", "tsx"):
        if not raw.startswith("."):
            return None  # bare/package import → external, no edge
        base = posixpath.normpath(posixpath.join(importer_dir, raw))
        cands = [base] + [base + e for e in _JS_EXTS] + [posixpath.join(base, "index" + e) for e in _JS_EXTS]
        return first_in_set(cands)

    if grammar_key == "python":
        if raw.startswith("."):
            dots = len(raw) - len(raw.lstrip("."))
            rest = raw[dots:].replace(".", "/")
            base = importer_dir
            for _ in range(dots - 1):
                base = posixpath.dirname(base)
            target = posixpath.join(base, rest) if rest else base
            return first_in_set([target + ".py", posixpath.join(target, "__init__.py")])
        mod = raw.replace(".", "/")
        cands = [mod + ".py", posixpath.join(mod, "__init__.py")]
        hit = first_in_set(cands)
        if hit:
            return hit
        for c in cands:  # suffix match for src-layout roots
            norm = posixpath.normpath(c)
            for f in fileset:
                if f == norm or f.endswith("/" + norm):
                    return f if f != importer_rel else None
        return None

    if grammar_key in ("c", "cpp"):
        cands = [posixpath.join(importer_dir, raw), raw]
        hit = first_in_set(cands)
        if hit:
            return hit
        bn = posixpath.basename(raw)  # last-resort basename match
        matches = [f for f in fileset if posixpath.basename(f) == bn and f != importer_rel]
        return matches[0] if len(matches) == 1 else None

    if grammar_key == "java":
        target = raw.replace(".", "/") + ".java"
        for f in fileset:
            if f == target or f.endswith("/" + target):
                return f
        return None

    if grammar_key == "shell":  # source/. includes — resolve by path, then basename
        if "$" in raw or "*" in raw:
            return None
        hit = first_in_set([posixpath.join(importer_dir, raw), raw])
        if hit:
            return hit
        bn = posixpath.basename(raw)
        matches = [f for f in fileset if posixpath.basename(f) == bn and f != importer_rel]
        return matches[0] if len(matches) == 1 else None

    return None  # go/rust/csharp/php — skip to keep the graph clean


# status from _classify_script_ref -> ref_counts bucket (run-diagnostics tally).
_REF_TALLY_KEY = {"skip": "skipped", "resolved": "resolved",
                  "missing": "missing", "function": "function_calls"}


def _classify_script_ref(raw: str, importer_rel: str, fileset: set[str],
                         by_basename: dict[str, list[str]],
                         shell_functions: frozenset[str] = frozenset()) -> tuple[str, str | None]:
    """Classify a shell / job-list / SQL-file reference against the scanned project:

      ("skip", None)     — not a concrete name (empty, or contains $ / * / leading '-'),
                           OR ambiguous (>=2 basename matches: the file exists, so do
                           not fabricate a missing node);
      ("function", None) — a BARE command token (no path separator, no script
                           extension) naming a shell function DEFINED in the
                           project; the function shadows any like-named external
                           command, so this is an internal call, never a file
                           reference (only considered when ``shell_functions`` is
                           supplied — the shell `calls` channel);
      ("resolved", rel)  — a path match, or exactly one basename match;
      ("missing", name)  — a concrete name that passed the filters but matches ZERO
                           scanned files (referenced but absent from the project).

    Resolved semantics are unchanged from the original _resolve_script — the only
    additions are splitting the zero-match outcome out of the old catch-all None,
    and (when ``shell_functions`` is given) recognising internal function calls.
    """
    if not raw:
        return ("skip", None)
    raw = raw.strip().strip('"').strip("'")
    if not raw or "$" in raw or "*" in raw or raw.startswith("-"):
        return ("skip", None)
    # Internal shell-function call: a bare command token (no path, no script
    # extension) naming a function defined in the project. Shell resolves a
    # defined function before any like-named external command/file, so this is
    # never a script reference and must not become a missing node. Path-qualified
    # or *.sh tokens are real file refs and bypass this (they are not bare).
    if shell_functions and "/" not in raw and not raw.lower().endswith(_SHELL_EXTS) \
            and raw in shell_functions:
        return ("function", None)
    importer_dir = posixpath.dirname(importer_rel)
    for cand in (posixpath.join(importer_dir, raw), raw):
        norm = posixpath.normpath(cand).lstrip("./")
        if norm in fileset and norm != importer_rel:
            return ("resolved", norm)
    bn = posixpath.basename(raw)
    hits: list[str] = []
    for name in (bn, *(bn + e for e in _SHELL_EXTS)):
        hits.extend(by_basename.get(name, []))
    hits = [h for h in dict.fromkeys(hits) if h != importer_rel]
    if len(hits) == 1:
        return ("resolved", hits[0])
    if len(hits) >= 2:
        return ("skip", None)     # ambiguous — the file exists; do not fabricate a missing node
    # Zero matches — NOT a real scanned file. Only now suppress phantoms: a shell
    # keyword / builtin / interpreter, or a self-reference, must not become a
    # missing node. (A REAL scanned file whose name equals a builtin already
    # resolved above, by path or basename, so its edge is kept.)
    if bn.lower() in _SHELL_NON_SCRIPT:
        return ("skip", None)     # shell keyword / builtin / interpreter, not a script ref
    if bn == posixpath.basename(importer_rel):
        return ("skip", None)     # self-reference (a script naming its own basename)
    return ("missing", bn)        # concrete name, zero matches -> referenced but not found


def _resolve_script(raw: str, importer_rel: str, fileset: set[str],
                    by_basename: dict[str, list[str]]) -> str | None:
    """Resolved rel-path for a script reference, or None (skip / missing / ambiguous).
    Thin wrapper over _classify_script_ref that preserves the original contract."""
    status, value = _classify_script_ref(raw, importer_rel, fileset, by_basename)
    return value if status == "resolved" else None


# ----------------------------------------------------------------------- assembly
def build_graph(
    *,
    project_name: str,
    project_root: str,
    scanned: list[ScannedFile],
    parses: list[FileParse],
    enrichments: dict[str, FileEnrichment],
    settings: Settings,
    description: str,
    frameworks: list[str],
    diagnostics: dict | None = None,
) -> dict:
    nodes: list[dict] = []
    edges: list[dict] = []
    used_ids: set[str] = set()
    ref_counts = {"resolved": 0, "missing": 0, "skipped": 0, "function_calls": 0}  # tally-only, for run diagnostics

    parse_by_rel = {p.rel_path: p for p in parses}
    file_id_by_rel: dict[str, str] = {}
    sym_id: dict[tuple[str, int], str] = {}       # (rel, symbol.key) -> node id
    layer_by_node: dict[str, str] = {}
    func_by_name: dict[str, list[str]] = {}
    func_by_name_file: dict[tuple[str, str], list[str]] = {}

    def mk_id(base: str) -> str:
        nid = base
        if nid in used_ids:
            i = 2
            while f"{base}#{i}" in used_ids:
                i += 1
            nid = f"{base}#{i}"
        used_ids.add(nid)
        return nid

    # Placeholder nodes for concrete script/SQL-file references that are absent
    # from the analyzed project (one per unique name, deduped). Created on demand
    # by the reference loops below.
    missing_id: dict[str, str] = {}

    def missing_node(name: str, kind: str) -> str:
        """Create-or-get a red placeholder node for a referenced-but-absent file.

        ``kind`` is the expected category of the missing reference — "script"
        (from a script-call / job-list) or "sql-file" (from a SQL .RUN FILE /
        bteq reference). A missing node is always an absent referenced FILE; there
        is no missing table or column. The kind is baked into the summary + tags
        so the dashboard can label the node. It carries ONLY the referenced name
        + this fixed kind — the file does not exist, so there is nothing to read
        and zero data exposure. Deliberately NOT added to ``layer_by_node``, so it
        never appears in any layer's grouping or counts.
        """
        if name in missing_id:
            return missing_id[name]
        nid = mk_id(f"missing:{name}")
        missing_id[name] = nid
        label = "script" if kind == "script" else "SQL file"
        nodes.append({
            "id": nid,
            "type": "missing",
            "name": name,
            "summary": f"Referenced {label} not found in the analyzed project: {name}.",
            "tags": ["missing", kind],
            "complexity": "simple",
        })
        return nid

    # --- nodes: one file node per scanned file + its parsed symbols ---
    for sf in scanned:
        rel = sf.rel_path
        enr = enrichments.get(rel)
        if enr is None:
            from .enrich import _fallback_enrichment  # defensive; normally present
            enr = _fallback_enrichment(parse_by_rel.get(rel, FileParse(rel, sf.language, None, False, [], [])))
        file_id = mk_id(f"file:{rel}")
        file_id_by_rel[rel] = file_id
        layer_by_node[file_id] = enr.layer
        file_tags = enr.tags or [sf.language]
        if sf.language == "joblist" and "joblist" not in file_tags:
            file_tags = ["joblist", *file_tags]
        nodes.append({
            "id": file_id,
            "type": "file",
            "name": os.path.basename(rel),
            "filePath": rel,
            "summary": enr.summary,
            "tags": file_tags,
            "complexity": enr.complexity,
        })

        fp = parse_by_rel.get(rel)
        if not fp:
            continue
        for s in fp.symbols:
            base = f"{s.kind}:{rel}:{s.name}"
            if base in used_ids:
                base = f"{s.kind}:{rel}:{s.name}:{s.line_start}"
            nid = mk_id(base)
            sym_id[(rel, s.key)] = nid
            layer_by_node[nid] = enr.layer
            member = enr.members.get(s.name, {})
            nodes.append({
                "id": nid,
                "type": s.kind,  # "function" | "class"
                "name": s.name,
                "filePath": rel,
                "lineRange": [s.line_start, s.line_end],
                "summary": member.get("summary") or f"{s.kind.capitalize()} '{s.name}'.",
                "tags": [sf.language, s.kind],
                "complexity": member.get("complexity") or "moderate",
            })
            if s.kind == "function":
                func_by_name.setdefault(s.name, []).append(nid)
                func_by_name_file.setdefault((rel, s.name), []).append(nid)

    # --- contains edges (file->symbol, class->method) ---
    for fp in parses:
        rel = fp.rel_path
        file_id = file_id_by_rel.get(rel)
        if not file_id:
            continue
        for s in fp.symbols:
            child = sym_id.get((rel, s.key))
            if not child:
                continue
            if s.parent_key is not None and (rel, s.parent_key) in sym_id:
                parent = sym_id[(rel, s.parent_key)]
            else:
                parent = file_id
            edges.append({"source": parent, "target": child, "type": "contains",
                          "direction": "forward", "weight": _W_CONTAINS})

    # --- imports edges (resolved file->file) ---
    fileset = set(file_id_by_rel)
    seen_import: set[tuple[str, str]] = set()
    for fp in parses:
        src_id = file_id_by_rel.get(fp.rel_path)
        if not src_id:
            continue
        for raw in fp.imports:
            tgt_rel = _resolve_import(raw, fp.rel_path, fp.grammar_key, fileset)
            if not tgt_rel or tgt_rel == fp.rel_path:
                continue
            tgt_id = file_id_by_rel[tgt_rel]
            pair = (src_id, tgt_id)
            if pair in seen_import:
                continue
            seen_import.add(pair)
            edges.append({"source": src_id, "target": tgt_id, "type": "imports",
                          "direction": "forward", "weight": _W_IMPORTS})

    # --- calls edges (intra-project, unique-name resolution) ---
    seen_call: set[tuple[str, str]] = set()
    for fp in parses:
        rel = fp.rel_path
        for s in fp.symbols:
            if s.kind != "function" or not s.calls:
                continue
            caller = sym_id.get((rel, s.key))
            if not caller:
                continue
            added = 0
            for callee in s.calls:
                if added >= _MAX_CALLS_PER_FUNC:
                    break
                same = [c for c in func_by_name_file.get((rel, callee), []) if c != caller]
                if len(same) == 1:
                    target = same[0]
                else:
                    glob = [c for c in func_by_name.get(callee, []) if c != caller]
                    if len(glob) != 1:
                        continue
                    target = glob[0]
                pair = (caller, target)
                if pair in seen_call:
                    continue
                seen_call.add(pair)
                edges.append({"source": caller, "target": target, "type": "calls",
                              "direction": "forward", "weight": _W_CALLS})
                added += 1

    # --- script-call & job-list edges (shell/.list file -> file "calls") ---
    shell_by_basename: dict[str, list[str]] = {}
    for sf in scanned:
        if sf.language == "shell":
            shell_by_basename.setdefault(posixpath.basename(sf.rel_path), []).append(sf.rel_path)
    # Shell functions defined anywhere in the project. A bare command token that
    # names one is an internal function call (the function shadows any like-named
    # external command), not a script reference, so it never becomes a missing
    # node. Used for the shell `calls` channel only; the SQL channel keeps its
    # original resolution (functions are not passed there).
    shell_functions = frozenset(
        s.name for fp in parses if fp.language == "shell"
        for s in fp.symbols if s.kind == "function" and s.name
    )
    seen_script_call: set[tuple[str, str]] = set()
    for fp in parses:
        src_id = file_id_by_rel.get(fp.rel_path)
        if not src_id or not fp.script_calls:
            continue
        is_joblist = fp.language == "joblist"
        for i, raw in enumerate(fp.script_calls):
            status, value = _classify_script_ref(raw, fp.rel_path, fileset, shell_by_basename, shell_functions)
            ref_counts[_REF_TALLY_KEY[status]] += 1
            if status == "resolved":
                if value == fp.rel_path:
                    continue
                tgt_id = file_id_by_rel[value]
            elif status == "missing":
                tgt_id = missing_node(value, "script")
            else:
                continue  # skip (non-concrete / ambiguous) or internal function call — no node/edge
            pair = (src_id, tgt_id)
            if pair in seen_script_call:
                continue
            seen_script_call.add(pair)
            edge = {"source": src_id, "target": tgt_id, "type": "calls",
                    "direction": "forward", "weight": _W_CALLS}
            if is_joblist:
                edge["index"] = i  # preserve manifest order
            edges.append(edge)

    # --- SQL: global table nodes + table-level lineage (Phase 2) ---
    sql_parses = [fp for fp in parses if getattr(fp, "sql", None) is not None]
    if sql_parses:
        # basename index for .RUN FILE resolution (shell + sql files)
        run_basename: dict[str, list[str]] = {}
        for sf in scanned:
            if sf.language in ("shell", "sql"):
                run_basename.setdefault(posixpath.basename(sf.rel_path), []).append(sf.rel_path)

        # 1) gather every referenced table + which file (if any) creates it
        all_tables: set[str] = set()
        table_creator: dict[str, str] = {}
        table_is_view: dict[str, bool] = {}
        upstream: dict[str, set] = {}    # table -> sources that feed it
        downstream: dict[str, set] = {}  # table -> targets it feeds
        table_writers: dict[str, set] = {}  # table -> files that write (produce) it
        table_readers: dict[str, set] = {}  # table -> files that read it
        for fp in sql_parses:
            r = fp.sql
            all_tables.update(t for t in r.table_refs if t)
            for o in r.objects:
                if o.kind in ("table", "view"):
                    all_tables.add(o.name)
                    table_creator.setdefault(o.name, fp.rel_path)
                    if o.kind == "view":
                        table_is_view[o.name] = True
            for tgt, srcs in r.lineage:
                for s in srcs:
                    upstream.setdefault(tgt, set()).add(s)
                    downstream.setdefault(s, set()).add(tgt)
            for tbl, ops in r.table_ops.items():       # producer/consumer map (Phase E)
                if "write" in ops:                     # purge alone removes data, not a feed
                    table_writers.setdefault(tbl, set()).add(fp.rel_path)
                if "read" in ops:
                    table_readers.setdefault(tbl, set()).add(fp.rel_path)

        # 2) one global table node per unique name (deterministic neighbor summary)
        table_id: dict[str, str] = {}
        for name in sorted(all_tables):
            if not name:
                continue
            tid = mk_id(f"table:{name}")
            table_id[name] = tid
            layer_by_node[tid] = "Data"
            is_view = table_is_view.get(name, False)
            ups = sorted(upstream.get(name, []))[:6]
            downs = sorted(downstream.get(name, []))[:6]
            bits = []
            if ups:
                bits.append("populated from " + ", ".join(_short(u) for u in ups))
            if downs:
                bits.append("feeds " + ", ".join(_short(d) for d in downs))
            summary = (f"{'View' if is_view else 'Table'} {name}"
                       + (" — " + "; ".join(bits) + "." if bits else "."))
            node = {"id": tid, "type": "table", "name": name, "summary": summary,
                    "tags": ["view"] if is_view else ["table"], "complexity": "simple"}
            if name in table_creator:
                node["filePath"] = table_creator[name]
            nodes.append(node)

        # 2b) column nodes (C1): one node per (table, column) actually referenced
        #     in the scripts, attached under its table via a `contains` edge. The
        #     column set also absorbs the endpoints of any column lineage so every
        #     lineage edge has both nodes. Columns live in the Data layer with
        #     their table; they are hidden in the graph until the user opts in.
        cols_by_table: dict[str, set] = {}
        for fp in sql_parses:
            for tbl, cols in fp.sql.used_columns.items():
                if tbl in table_id:
                    cols_by_table.setdefault(tbl, set()).update(c for c in cols if c)
        col_lineage: list = []  # (tgt_table, tgt_col, src_table, src_col), de-duped
        seen_col_lineage: set = set()
        for fp in sql_parses:
            for tup in fp.sql.column_lineage:
                tt, tc, st, sc = tup
                if tt in table_id and st in table_id and tup not in seen_col_lineage:
                    seen_col_lineage.add(tup)
                    cols_by_table.setdefault(tt, set()).add(tc)
                    cols_by_table.setdefault(st, set()).add(sc)
                    col_lineage.append(tup)

        column_id: dict[tuple, str] = {}
        for tbl in sorted(cols_by_table):
            t_id = table_id.get(tbl)
            if not t_id:
                continue
            for col in sorted(cols_by_table[tbl]):
                cid = mk_id(f"column:{tbl}.{col}")
                column_id[(tbl, col)] = cid
                layer_by_node[cid] = "Data"
                col_node = {"id": cid, "type": "column", "name": col,
                            "summary": f"Column {col} of {tbl}.",
                            "tags": ["column"], "complexity": "simple"}
                if tbl in table_creator:
                    col_node["filePath"] = table_creator[tbl]
                nodes.append(col_node)
                edges.append({"source": t_id, "target": cid, "type": "contains",
                              "direction": "forward", "weight": _W_CONTAINS})

        # 3) procedures/macros as file-owned function nodes (+ contains)
        for fp in sql_parses:
            file_id = file_id_by_rel.get(fp.rel_path)
            if not file_id:
                continue
            enr = enrichments.get(fp.rel_path)
            for o in fp.sql.objects:
                if o.kind not in ("procedure", "macro"):
                    continue
                pid = mk_id(f"function:{fp.rel_path}:{o.name}")
                layer_by_node[pid] = "Data"  # SQL procs/macros live in the single Data layer
                member = enr.members.get(o.name, {}) if enr else {}
                nodes.append({
                    "id": pid, "type": "function", "name": o.name, "filePath": fp.rel_path,
                    "summary": member.get("summary") or f"{o.kind.capitalize()} '{o.name}'.",
                    "tags": ["sql", o.kind],
                    "complexity": member.get("complexity") or "moderate",
                })
                edges.append({"source": file_id, "target": pid, "type": "contains",
                              "direction": "forward", "weight": _W_CONTAINS})
                func_by_name.setdefault(o.name, []).append(pid)

        # 4) aggregated file->table operation edges (one per file/table, op-set
        #    in `description`), table<-table lineage provenance, exec, .RUN FILE
        seen_sql: set[tuple[str, str, str]] = set()

        def _sql_edge(s: str, t: str, ty: str, w: float, desc: str = "",
                      commented: bool = False) -> bool:
            if s and t and s != t and (s, t, ty) not in seen_sql:
                seen_sql.add((s, t, ty))
                edge = {"source": s, "target": t, "type": ty,
                        "direction": "forward", "weight": w}
                if desc:
                    edge["description"] = desc
                if commented:
                    edge["commented"] = True  # provenance: derived from `--`-commented SQL
                edges.append(edge)
                return True
            return False

        for fp in sql_parses:
            file_id = file_id_by_rel.get(fp.rel_path)
            r = fp.sql
            if file_id and fp.language == "sql":
                # Only real SQL files move to the Data layer; shell files that
                # merely embed bteq SQL stay code nodes (they just gain DATA edges).
                layer_by_node[file_id] = "Data"

            # one aggregated edge per (file, table) carrying the operation-set;
            # writes_to if the file writes/purges the table, else reads_from.
            n_ops = 0
            for tbl in sorted(r.table_ops):
                if n_ops >= _MAX_SQL_LINEAGE_PER_FILE:
                    break
                t_id = table_id.get(tbl)
                if not (file_id and t_id):
                    continue
                seq = r.table_ops[tbl]  # ordered op sequence (execution order)
                desc = " → ".join(seq)  # arrow-joined, e.g. "purge -> write -> read"
                ty = "writes_to" if ({"write", "purge"} & set(seq)) else "reads_from"
                _sql_edge(file_id, t_id, ty, _W_LINEAGE, desc)
                n_ops += 1

            # table <- table provenance, kept as a (filterable) layer of its own
            n_prov = 0
            for tgt, srcs in r.lineage:
                if n_prov >= _MAX_SQL_LINEAGE_PER_FILE:
                    break
                t_id = table_id.get(tgt)
                if not t_id:
                    continue
                for s in srcs:
                    s_id = table_id.get(s)
                    if s_id:
                        _sql_edge(t_id, s_id, "reads_from", _W_LINEAGE)  # table <- table
                    n_prov += 1

            for pname in r.proc_calls:
                cands = func_by_name.get(pname, [])
                if file_id and len(cands) == 1:
                    _sql_edge(file_id, cands[0], "calls", _W_CALLS)
            for ref in r.references:  # .RUN FILE= and bteq < x.sql / cat x.sql | bteq
                if not file_id:
                    continue
                status, value = _classify_script_ref(ref, fp.rel_path, fileset, run_basename)
                ref_counts[_REF_TALLY_KEY[status]] += 1
                if status == "resolved" and value != fp.rel_path:
                    _sql_edge(file_id, file_id_by_rel[value], "imports", _W_IMPORTS, "runs")
                elif status == "missing":
                    _sql_edge(file_id, missing_node(value, "sql-file"), "imports", _W_IMPORTS, "runs")

        # 4b) column <- column lineage (C2): a written column reads_from the source
        #     column(s) it derives from. Same reads_from convention as table<-table
        #     provenance, so the transitive lineage tracer extends to columns.
        for (tt, tc, st, sc) in col_lineage:
            tcid = column_id.get((tt, tc))
            scid = column_id.get((st, sc))
            if tcid and scid:
                _sql_edge(tcid, scid, "reads_from", _W_LINEAGE)

        # 4c) commented (recovered) SQL: read documented `--`-commented SQL as real
        #     SQL. A table/column already created from ACTIVE SQL is REUSED (not
        #     duplicated); only entities unique to comments are created, tagged
        #     "commented". Their edges are added (also tagged) and deduped against
        #     active edges via `seen_sql` (active wins). Recovered nodes/edges are
        #     ordinary graph members — searchable and traversable in lineage.
        def _commented_table(name: str, is_view: bool = False) -> str | None:
            if not name:
                return None
            if name in table_id:
                return table_id[name]                  # reuse active / earlier commented node
            tid = mk_id(f"table:{name}")
            table_id[name] = tid
            layer_by_node[tid] = "Data"
            nodes.append({
                "id": tid, "type": "table", "name": name,
                "summary": f"{'View' if is_view else 'Table'} {name} (referenced in commented SQL).",
                "tags": (["view"] if is_view else ["table"]) + ["commented"],
                "complexity": "simple",
            })
            return tid

        def _commented_column(tbl: str, col: str) -> str | None:
            if not (tbl and col):
                return None
            key = (tbl, col)
            if key in column_id:
                return column_id[key]                  # reuse existing (active or commented) node
            t_id = _commented_table(tbl)
            if not t_id:
                return None
            cid = mk_id(f"column:{tbl}.{col}")
            column_id[key] = cid
            layer_by_node[cid] = "Data"
            nodes.append({
                "id": cid, "type": "column", "name": col,
                "summary": f"Column {col} of {tbl} (referenced in commented SQL).",
                "tags": ["column", "commented"], "complexity": "simple",
            })
            edges.append({"source": t_id, "target": cid, "type": "contains",
                          "direction": "forward", "weight": _W_CONTAINS, "commented": True})
            return cid

        for fp in sql_parses:
            cr = getattr(fp.sql, "commented", None)
            if cr is None:
                continue
            file_id = file_id_by_rel.get(fp.rel_path)
            view_names = {o.name for o in cr.objects if o.kind == "view"}
            for name in sorted(set(cr.table_refs)
                               | {o.name for o in cr.objects if o.kind in ("table", "view")}):
                _commented_table(name, name in view_names)
            for tbl in sorted(cr.used_columns):
                for col in sorted(c for c in cr.used_columns[tbl] if c):
                    _commented_column(tbl, col)
            if file_id:                                 # file -> table operation edges
                n_ops = 0
                for tbl in sorted(cr.table_ops):
                    if n_ops >= _MAX_SQL_LINEAGE_PER_FILE:
                        break
                    t_id = _commented_table(tbl)
                    if not t_id:
                        continue
                    seq = cr.table_ops[tbl]
                    ty = "writes_to" if ({"write", "purge"} & set(seq)) else "reads_from"
                    _sql_edge(file_id, t_id, ty, _W_LINEAGE, " → ".join(seq), commented=True)
                    n_ops += 1
            for tgt, srcs in cr.lineage:                # table <- table lineage
                t_id = _commented_table(tgt)
                if not t_id:
                    continue
                for s in srcs:
                    s_id = _commented_table(s)
                    if s_id:
                        _sql_edge(t_id, s_id, "reads_from", _W_LINEAGE, commented=True)
            for (tt, tc, st, sc) in cr.column_lineage:  # column <- column lineage
                tcid = _commented_column(tt, tc)
                scid = _commented_column(st, sc)
                if tcid and scid:
                    _sql_edge(tcid, scid, "reads_from", _W_LINEAGE, commented=True)

        # 5) file->file "feeds" edges (Phase E): file A writes table X and file B
        #    reads X  =>  B depends_on A, described by the shared table(s). Skip
        #    hot/shared tables (writers/readers over the fan-out cap) and limit
        #    how many feeds any one consumer file accrues.
        shared_tables: dict[tuple[str, str], list[str]] = {}  # (consumer, producer) -> [tables]
        for tbl in sorted(set(table_writers) & set(table_readers)):
            writers, readers = table_writers[tbl], table_readers[tbl]
            if len(writers) > _MAX_FEED_FANOUT or len(readers) > _MAX_FEED_FANOUT:
                continue  # hot/shared staging table -> skip to avoid W*R blow-up
            for b in sorted(readers):
                for a in sorted(writers):
                    if a == b:
                        continue  # a file reading what it just wrote is not a feed
                    shared_tables.setdefault((b, a), []).append(tbl)

        feeds_per_file: dict[str, int] = {}
        for (b, a) in sorted(shared_tables):  # one enriched edge per consumer/producer pair
            if feeds_per_file.get(b, 0) >= _MAX_FEEDS_PER_FILE:
                continue
            b_id, a_id = file_id_by_rel.get(b), file_id_by_rel.get(a)
            if not (b_id and a_id):
                continue
            desc = _feeds_desc(shared_tables[(b, a)])
            if _sql_edge(b_id, a_id, "depends_on", _W_FEEDS, desc):
                feeds_per_file[b] = feeds_per_file.get(b, 0) + 1

    # --- layers (group node ids by assigned layer) ---
    layers: list[dict] = []
    for key, name, desc in LAYER_META:
        node_ids = [nid for nid, layer in layer_by_node.items() if layer == key]
        if node_ids:
            layers.append({"id": _LAYER_SLUG[key], "name": name, "description": desc, "nodeIds": node_ids})

    languages = sorted({sf.language for sf in scanned})
    project = {
        "name": project_name,
        "languages": languages,
        "frameworks": frameworks,
        "description": description,
        "analyzedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "gitCommitHash": git_commit_hash(project_root),
    }
    if diagnostics is not None:
        diagnostics["references"] = ref_counts
    return {
        "version": VERSION,
        "project": project,
        "nodes": nodes,
        "edges": edges,
        "layers": layers,
        "tour": [],
    }
