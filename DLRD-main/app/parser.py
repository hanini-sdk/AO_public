"""Deterministic structural parsing with tree-sitter — text only, never executed.

For each file we extract functions, classes/methods (as functions nested in a
class), imports and (where practical) intra-project call names. Grammars are
loaded lazily and cached; any load/parse failure degrades to "no symbols" for
that file (the file node is still produced by ``graph.py``) — never fatal.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from .scanner import ScannedFile

log = logging.getLogger("data_lineage_retro_documentation.parser")

# ext -> grammar key
GRAMMAR_BY_EXT = {
    ".py": "python", ".pyi": "python",
    ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".mts": "typescript", ".cts": "typescript",
    ".tsx": "tsx",
    ".java": "java", ".go": "go", ".rs": "rust",
    ".c": "c", ".h": "c",
    ".cc": "cpp", ".cpp": "cpp", ".cxx": "cpp", ".hpp": "cpp", ".hh": "cpp", ".hxx": "cpp",
    ".cs": "csharp", ".rb": "ruby", ".php": "php",
    ".sh": "shell", ".bash": "shell", ".zsh": "shell", ".ksh": "shell",
}

# grammar key -> (pip module suffix, language attribute)
GRAMMAR_LOADERS = {
    "python": ("python", "language"),
    "javascript": ("javascript", "language"),
    "typescript": ("typescript", "language_typescript"),
    "tsx": ("typescript", "language_tsx"),
    "java": ("java", "language"),
    "go": ("go", "language"),
    "rust": ("rust", "language"),
    "c": ("c", "language"),
    "cpp": ("cpp", "language"),
    "csharp": ("c_sharp", "language"),
    "ruby": ("ruby", "language"),
    "php": ("php", "language_php"),
    "shell": ("bash", "language"),
}

FUNCTION_TYPES = {
    "python": {"function_definition"},
    "javascript": {"function_declaration", "generator_function_declaration", "method_definition"},
    "typescript": {"function_declaration", "generator_function_declaration", "method_definition"},
    "tsx": {"function_declaration", "generator_function_declaration", "method_definition"},
    "java": {"method_declaration", "constructor_declaration"},
    "go": {"function_declaration", "method_declaration"},
    "rust": {"function_item"},
    "c": {"function_definition"},
    "cpp": {"function_definition"},
    "csharp": {"method_declaration", "constructor_declaration", "local_function_statement"},
    "ruby": {"method", "singleton_method"},
    "php": {"function_definition", "method_declaration"},
    "shell": {"function_definition"},
}

CLASS_TYPES = {
    "python": {"class_definition"},
    "javascript": {"class_declaration"},
    "typescript": {"class_declaration", "abstract_class_declaration", "interface_declaration", "enum_declaration"},
    "tsx": {"class_declaration", "abstract_class_declaration", "interface_declaration", "enum_declaration"},
    "java": {"class_declaration", "interface_declaration", "enum_declaration", "record_declaration", "annotation_type_declaration"},
    "go": {"type_spec"},
    "rust": {"struct_item", "enum_item", "trait_item", "union_item"},
    "c": {"struct_specifier", "union_specifier", "enum_specifier"},
    "cpp": {"class_specifier", "struct_specifier", "union_specifier", "enum_specifier"},
    "csharp": {"class_declaration", "interface_declaration", "struct_declaration", "enum_declaration", "record_declaration"},
    "ruby": {"class", "module"},
    "php": {"class_declaration", "interface_declaration", "trait_declaration", "enum_declaration"},
    "shell": set(),  # shell has no classes
}

IMPORT_TYPES = {
    "python": {"import_statement", "import_from_statement"},
    "javascript": {"import_statement", "export_statement"},
    "typescript": {"import_statement", "export_statement"},
    "tsx": {"import_statement", "export_statement"},
    "java": {"import_declaration"},
    "go": {"import_declaration"},
    "rust": {"use_declaration"},
    "c": {"preproc_include"},
    "cpp": {"preproc_include"},
    "csharp": {"using_directive"},
    "ruby": set(),  # via require/require_relative call heuristic
    "php": {"namespace_use_declaration"},
}

CALL_GRAMMARS = {"python", "javascript", "typescript", "tsx", "java"}
_CALL_NODE_TYPES = {"call", "call_expression", "method_invocation"}
# Per-call-node-type field that holds the callee. Default is "function" (Python
# `call`, JS `call_expression`); Java's `method_invocation` instead exposes the
# invoked method through its "name" field (the receiver, if any, is the separate
# "object" field), so `obj.foo()` / `this.foo()` / `foo()` all yield "foo" — the
# same callee-name shape the other languages produce.
_CALL_CALLEE_FIELD = {"method_invocation": "name"}
_NAME_TYPES = {
    "identifier", "type_identifier", "field_identifier", "constant", "name",
    "property_identifier", "scoped_identifier", "scoped_type_identifier",
}
_MAX_SYMBOLS_PER_FILE = 120

_lang_cache: dict[str, "object | None"] = {}


def _get_language(grammar_key: str):
    if grammar_key in _lang_cache:
        return _lang_cache[grammar_key]
    lang = None
    spec = GRAMMAR_LOADERS.get(grammar_key)
    if spec:
        module_suffix, attr = spec
        try:
            from tree_sitter import Language

            mod = __import__("tree_sitter_" + module_suffix)
            lang = Language(getattr(mod, attr)())
        except Exception as exc:  # grammar/tree-sitter unavailable → degrade
            log.debug("grammar %s unavailable: %s", grammar_key, exc)
            lang = None
    _lang_cache[grammar_key] = lang
    return lang


@dataclass
class Symbol:
    key: int
    kind: str            # "function" | "class"
    name: str
    line_start: int      # 1-based
    line_end: int
    parent_key: int | None
    calls: list[str] = field(default_factory=list)


@dataclass
class FileParse:
    rel_path: str
    language: str
    grammar_key: str | None
    parse_ok: bool
    symbols: list[Symbol]
    imports: list[str]
    # File-level references (shell + .list). script_calls -> "calls" edges,
    # imports (source/.) -> "imports" edges. variables: literal NAME->value map
    # captured for later phases (SQL var interpolation); unused in Phase 1.
    script_calls: list[str] = field(default_factory=list)
    variables: dict[str, str] = field(default_factory=dict)
    # SQL extraction result (sqlproc.SqlResult) for .sql/.bteq/.btq files; None
    # for everything else. Consumed by graph.py to build table nodes + lineage.
    sql: "object | None" = None


def _text(node, src: bytes) -> str:
    return src[node.start_byte : node.end_byte].decode("utf-8", "replace")


def _find_descendant(node, types: set[str], max_depth: int = 4):
    stack = [(node, 0)]
    while stack:
        n, d = stack.pop(0)
        if n is not node and n.type in types:
            return n
        if d < max_depth:
            for c in n.children:
                stack.append((c, d + 1))
    return None


def _node_name(node, src: bytes, grammar_key: str) -> str | None:
    nm = node.child_by_field_name("name")
    if nm is not None:
        return _text(nm, src).strip()
    if grammar_key in ("c", "cpp") and node.type == "function_definition":
        decl = node.child_by_field_name("declarator")
        if decl is not None:
            ident = _find_descendant(
                decl, {"identifier", "field_identifier", "qualified_identifier", "operator_name"}
            )
            if ident is not None:
                return _text(ident, src).strip()
    for ch in node.named_children:
        if ch.type in _NAME_TYPES:
            return _text(ch, src).strip()
    return None


def _is_class_definition(node, grammar_key: str) -> bool:
    if grammar_key == "go":
        return any(c.type in ("struct_type", "interface_type") for c in node.named_children)
    if grammar_key in ("c", "cpp"):
        return any(
            c.type in ("field_declaration_list", "enumerator_list") for c in node.children
        )
    return True


def _callee_name(fn_node, src: bytes) -> str | None:
    if fn_node is None:
        return None
    t = fn_node.type
    if t == "identifier":
        return _text(fn_node, src)
    if t == "attribute":  # python a.b.c -> c
        attr = fn_node.child_by_field_name("attribute")
        return _text(attr, src) if attr is not None else None
    if t == "member_expression":  # js a.b -> b
        prop = fn_node.child_by_field_name("property")
        return _text(prop, src) if prop is not None else None
    return None


def _collect_calls(func_node, src: bytes) -> list[str]:
    names: set[str] = set()
    stack = [func_node]
    while stack:
        n = stack.pop()
        if n.type in _CALL_NODE_TYPES:
            field = _CALL_CALLEE_FIELD.get(n.type, "function")
            nm = _callee_name(n.child_by_field_name(field), src)
            if nm and nm.isidentifier():
                names.add(nm)
        stack.extend(n.children)
    return sorted(names)


def _collect_symbols(root, src: bytes, grammar_key: str) -> list[Symbol]:
    symbols: list[Symbol] = []
    counter = [0]
    fn_types = FUNCTION_TYPES.get(grammar_key, set())
    cls_types = CLASS_TYPES.get(grammar_key, set())
    collect_calls = grammar_key in CALL_GRAMMARS

    def visit(node, enclosing_class_key: int | None) -> None:
        if len(symbols) >= _MAX_SYMBOLS_PER_FILE:
            return
        if node.type in cls_types and _is_class_definition(node, grammar_key):
            name = _node_name(node, src, grammar_key)
            new_enclosing = enclosing_class_key
            if name:
                key = counter[0]; counter[0] += 1
                symbols.append(
                    Symbol(key, "class", name, node.start_point[0] + 1,
                           node.end_point[0] + 1, enclosing_class_key)
                )
                new_enclosing = key
            for ch in node.children:
                visit(ch, new_enclosing)
            return
        if node.type in fn_types:
            name = _node_name(node, src, grammar_key)
            if name:
                key = counter[0]; counter[0] += 1
                calls = _collect_calls(node, src) if collect_calls else []
                symbols.append(
                    Symbol(key, "function", name, node.start_point[0] + 1,
                           node.end_point[0] + 1, enclosing_class_key, calls)
                )
            return  # do not descend into a function body to mine more symbols
        for ch in node.children:
            visit(ch, enclosing_class_key)

    visit(root, None)
    return symbols


def _strip_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] in "\"'`" and s[-1] == s[0]:
        return s[1:-1]
    return s


def _collect_imports(root, src: bytes, grammar_key: str) -> list[str]:
    targets: list[str] = []
    import_types = IMPORT_TYPES.get(grammar_key, set())
    stack = [root]
    while stack:
        n = stack.pop()
        t = n.type
        if grammar_key == "ruby" and t == "call":
            fn = n.child_by_field_name("function") or n.child_by_field_name("method")
            fn_name = _text(fn, src) if fn is not None else ""
            if fn_name in ("require", "require_relative"):
                lit = _find_descendant(n, {"string", "string_content"}, max_depth=3)
                if lit is not None:
                    targets.append(_strip_quotes(_text(lit, src)))
        elif t in import_types:
            if grammar_key == "python":
                if t == "import_from_statement":
                    mod = n.child_by_field_name("module_name")
                    if mod is not None:
                        targets.append(_text(mod, src).strip())
                else:
                    for ch in n.named_children:
                        if ch.type in ("dotted_name", "aliased_import"):
                            dn = ch if ch.type == "dotted_name" else _find_descendant(ch, {"dotted_name"}, 2)
                            if dn is not None:
                                targets.append(_text(dn, src).strip())
            elif grammar_key in ("javascript", "typescript", "tsx"):
                source = n.child_by_field_name("source") or _find_descendant(n, {"string"}, 2)
                if source is not None:
                    targets.append(_strip_quotes(_text(source, src)))
            elif grammar_key in ("c", "cpp"):
                lit = _find_descendant(n, {"string_literal"}, 2)  # local includes only
                if lit is not None:
                    targets.append(_strip_quotes(_text(lit, src)))
            elif grammar_key == "go":
                for lit in _iter_descendants(n, {"interpreted_string_literal", "raw_string_literal"}):
                    targets.append(_strip_quotes(_text(lit, src)))
            else:  # java / rust / csharp / php — take the qualified path text
                ident = _find_descendant(
                    n, {"scoped_identifier", "qualified_name", "namespace_name",
                        "namespace_use_clause", "identifier", "name"}, 3
                )
                if ident is not None:
                    targets.append(_text(ident, src).strip().rstrip(";"))
        stack.extend(n.children)
    # de-dup preserving order
    seen: set[str] = set()
    out = []
    for tgt in targets:
        if tgt and tgt not in seen:
            seen.add(tgt)
            out.append(tgt)
    return out


def _iter_descendants(node, types: set[str]):
    stack = [node]
    while stack:
        n = stack.pop()
        if n is not node and n.type in types:
            yield n
        stack.extend(n.children)


# ----------------------------------------------------------------- shell refs
_SHELL_EXTS = (".sh", ".ksh", ".bash", ".zsh")
_MAX_SCRIPT_CALLS = 200
# Builtins / coreutils / external tools — never project scripts. Skipped as
# bare-word call candidates; path-like or *.sh names are always considered.
_SHELL_BUILTINS = {
    "echo", "printf", "cd", "pwd", "export", "set", "unset", "read", "eval",
    "exec", "exit", "return", "local", "declare", "typeset", "readonly", "shift",
    "trap", "wait", "sleep", "true", "false", "test", "[", "[[", "alias", "type",
    "cat", "grep", "egrep", "fgrep", "sed", "awk", "cut", "sort", "uniq", "head",
    "tail", "tr", "wc", "tee", "xargs", "find", "ls", "rm", "cp", "mv", "mkdir",
    "rmdir", "touch", "chmod", "chown", "ln", "basename", "dirname", "expr",
    "let", "seq", "date", "env", "mktemp", "getopts", "logger", "kill", "ps",
    "gzip", "gunzip", "zip", "unzip", "tar", "ssh", "scp", "sftp", "rsync",
    "curl", "wget", "ping", "nc", "mail", "mailx", "sendmail", "head", "paste",
    "python", "python2", "python3", "perl", "ruby", "node", "java", "make",
    "git", "svn", "docker", "kubectl", "aws", "gcloud",
    "bteq", "tdload", "fastload", "multiload", "mload", "tbuild", "tpt", "fexp",
    "sqlplus", "psql", "mysql", "db2", "isql", "hive", "beeline", "spark-submit",
}

# Shell reserved words, builtins, and interpreter names. The invoked SCRIPT is
# the relevant token in a command; these are never project-script references and
# must never become script_calls / "calls" edges / missing nodes — even when
# path-qualified (e.g. /bin/true, which `looks_script` would otherwise admit).
# Checked before the looks-like-a-script test here, and re-checked as a backstop
# in graph.py's _classify_script_ref (which also covers job-list refs).
_SHELL_NON_SCRIPT = frozenset({
    # reserved words
    "if", "then", "else", "elif", "fi", "case", "esac", "for", "while", "until",
    "do", "done", "function", "select", "in", "time", "!",
    # builtins
    "break", "continue", "return", "exit", "shift", "true", "false", ":", ".",
    "source", "eval", "exec", "export", "readonly", "local", "declare", "typeset",
    "unset", "set", "read", "echo", "printf", "test", "[", "[[", "cd", "pwd",
    "wait", "trap", "getopts", "hash", "type", "command", "builtin", "let", "umask",
    # interpreters
    "bash", "ksh", "sh", "zsh", "dash",
})


def _command_args(cmd_node) -> list:
    """Argument nodes of a `command`, excluding its name and any redirects."""
    out = []
    for ch in cmd_node.named_children:
        if ch.type == "command_name" or ch.type.endswith("_redirect"):
            continue
        out.append(ch)
    return out


def _script_arg_ref(raw: str) -> str | None:
    """A command ARGUMENT that denotes a shell-script path -> its bare basename.

    Orchestrators launch scripts through wrapper commands that take the real
    script PATH as an argument (``func_RUN_STEP "STEP" "$ROOT/dir/run.sh"``,
    ``bash deploy.sh``). We reduce such an argument to its basename (``run.sh``),
    which drops any ``$VAR/`` / directory prefix — so only a concrete filename,
    never a variable path or value, is ever surfaced. Returns None for anything
    that is not a concrete ``*.sh``-style name (a variable, an assignment, a flag)."""
    bn = _strip_quotes(raw).strip().rsplit("/", 1)[-1]   # drop $VAR/ + dir prefix
    if not bn.lower().endswith(_SHELL_EXTS):
        return None
    if "$" in bn or "=" in bn or bn.startswith("-"):
        return None
    return bn


def _collect_shell_refs(root, src: bytes) -> tuple[list[str], list[str], dict[str, str]]:
    """Return (source/. includes, script-call candidates, literal var map)."""
    includes: list[str] = []
    script_calls: list[str] = []
    variables: dict[str, str] = {}
    stack = [root]
    while stack:
        n = stack.pop()
        t = n.type
        if t == "variable_assignment":
            nm = n.child_by_field_name("name")
            val = n.child_by_field_name("value")
            if nm is not None:
                name = _text(nm, src)
                if val is not None and val.type in ("word", "string", "raw_string", "number"):
                    variables[name] = _strip_quotes(_text(val, src))
                else:
                    variables.setdefault(name, "")  # known, but not a static literal
        elif t == "command":
            name_node = n.child_by_field_name("name")
            cmd = _text(name_node, src).strip() if name_node is not None else ""
            if not cmd or "$" in cmd:
                pass
            elif cmd in ("source", "."):
                args = _command_args(n)
                if args:
                    inc = _strip_quotes(_text(args[0], src)).strip()
                    if inc and "$" not in inc:
                        includes.append(inc)
            else:
                base = cmd.rsplit("/", 1)[-1].lower()
                if base in _SHELL_NON_SCRIPT:
                    pass  # reserved word / builtin / interpreter — never a script ref
                else:
                    looks_script = ("/" in cmd) or cmd.endswith(_SHELL_EXTS)
                    if looks_script or base not in _SHELL_BUILTINS:
                        script_calls.append(cmd)
                # Launcher pattern: the invoked command (often a project function,
                # or an interpreter like `bash`/`ksh`) is handed the REAL script
                # path as an argument. Capture *.sh-style path arguments by
                # basename. Only for non-builtin commands: a known tool/coreutil
                # (echo, printf, test/[, cp, mv, rm, grep, find, cat, ...) takes a
                # *.sh arg as DATA, not as an execution — interpreters are not in
                # that set, so `bash deploy.sh` still resolves deploy.sh.
                if base not in _SHELL_BUILTINS:
                    for arg in _command_args(n):
                        ref = _script_arg_ref(_text(arg, src))
                        if ref:
                            script_calls.append(ref)
        stack.extend(n.children)

    def _dedupe(xs: list[str]) -> list[str]:
        return list(dict.fromkeys(xs))

    return _dedupe(includes), _dedupe(script_calls)[:_MAX_SCRIPT_CALLS], variables


# ------------------------------------------------- bteq heredoc SQL (Phase 3)
# Shell scripts embed Teradata SQL in `bteq` heredocs. We locate those heredocs
# via the tree-sitter-bash AST (robust to arbitrary/quoted delimiters, <<-, and
# pipelines), slice the body text, best-effort interpolate this file's shell
# vars, and feed each body through the existing sqlproc engine — so the embedded
# SQL inherits credential stripping, sigil/temporal handling and ordered ops.
# Parse-only: AST + string slicing only; the shell/bteq is never executed.
_VAR_BRACE_RE = re.compile(r"\$\{(\w+)\}")
_VAR_PLAIN_RE = re.compile(r"\$(\w+)")
_ESC_DOLLAR = "\x00ESCAPED_DOLLAR\x00"


def _interpolate_heredoc(body: str, variables: dict[str, str]) -> str:
    """Best-effort `$VAR` / `${VAR}` expansion using this file's literal var map.

    `\\$` is a literal '$' (no expansion); unknown / non-literal vars are left as
    `$VAR` so sqlproc's sigil normalisation still merges the table. This file's
    map only — no cross-file, command-substitution or array resolution.
    """
    body = body.replace("\\$", _ESC_DOLLAR)

    def _sub(m):
        val = variables.get(m.group(1))
        return val if val else m.group(0)  # known-and-non-empty -> value; else keep placeholder

    body = _VAR_BRACE_RE.sub(_sub, body)
    body = _VAR_PLAIN_RE.sub(_sub, body)
    return body.replace(_ESC_DOLLAR, "$")


def _child_of_type(node, type_name: str):
    for c in node.children:
        if c.type == type_name:
            return c
    return None


def _stdin_command(parent):
    """The command consuming stdin of a redirected_statement: its body command,
    or the last stage of a body pipeline (`cat x | bteq`)."""
    if parent is None or parent.type != "redirected_statement":
        return None
    body = parent.child_by_field_name("body")
    if body is None:
        return None
    if body.type == "command":
        return body
    if body.type == "pipeline":
        cmds = [c for c in body.children if c.type == "command"]
        return cmds[-1] if cmds else None
    return None


def _cmd_basename(cmd, src: bytes) -> str:
    if cmd is None:
        return ""
    name_node = cmd.child_by_field_name("name")
    name = _text(name_node, src).strip() if name_node is not None else ""
    return name.rsplit("/", 1)[-1].lower()


def _heredoc_consumer_is_bteq(hr, src: bytes) -> bool:
    """True if this heredoc_redirect feeds a `bteq` command — directly
    (`bteq <<EOF`, even with a trailing `| sed`) or as the last pipeline stage
    (`cat x | bteq <<EOF`)."""
    return _cmd_basename(_stdin_command(hr.parent), src) == "bteq"


def _merge_sql_entities(into, other) -> None:
    """Merge SqlResult ENTITIES (objects/lineage/ops/columns) only, preserving the
    per-table op order (consecutive duplicates collapse across the boundary).
    Stats are merged separately by _merge_sql_result."""
    into.objects.extend(other.objects)
    into.lineage.extend(other.lineage)
    into.proc_calls.extend(other.proc_calls)
    into.references.extend(other.references)
    into.table_refs |= other.table_refs
    for tbl, seq in other.table_ops.items():
        cur = into.table_ops.setdefault(tbl, [])
        for op in seq:
            if not cur or cur[-1] != op:
                cur.append(op)
    for tbl, cols in other.used_columns.items():       # C1 — union used columns
        into.used_columns.setdefault(tbl, set()).update(cols)
    into.column_lineage.extend(other.column_lineage)   # C2 — append column lineage


def _merge_sql_result(into, other) -> None:
    """Merge one heredoc's SqlResult into the file's accumulator: active entities,
    tally-only diagnostics stats, and the recovered `commented` sibling (entities
    only — its recovery stats are already folded into ``stats``)."""
    _merge_sql_entities(into, other)
    from . import sqlproc                               # tally-only diagnostics stats
    into.stats = sqlproc.merge_stats(into.stats, other.stats)  # sum across heredocs (incl recovery)
    if other.commented is not None:                    # recovered commented SQL (entities only)
        if into.commented is None:
            into.commented = other.commented
        else:
            _merge_sql_entities(into.commented, other.commented)


def _is_input_redirect(fr) -> bool:
    """True for a `< file` file_redirect (stdin), not `>` / `>>` / fd dups."""
    return any((not c.is_named) and c.type == "<" for c in fr.children)


def _collect_bteq_file_refs(root, src: bytes) -> list[str]:
    """Part B — bteq sinks that point at a *separate* .sql file rather than inline
    SQL: `bteq < x.sql` and `cat x.sql | bteq`. Returns referenced filenames (to
    be resolved within the project + linked with a `runs` edge). `.RUN FILE=` is
    already captured by sqlproc inside heredoc bodies (Part A)."""
    refs: list[str] = []
    stack = [root]
    while stack:
        n = stack.pop()
        if n.type == "file_redirect" and _is_input_redirect(n):
            if _cmd_basename(_stdin_command(n.parent), src) == "bteq":
                dest = n.child_by_field_name("destination")
                if dest is not None:
                    refs.append(_strip_quotes(_text(dest, src)).strip())
        elif n.type == "pipeline":
            cmds = [c for c in n.children if c.type == "command"]
            if cmds and _cmd_basename(cmds[-1], src) == "bteq":
                for c in cmds[:-1]:
                    if _cmd_basename(c, src) == "cat":
                        for a in _command_args(c):  # cat's file arg(s)
                            w = _strip_quotes(_text(a, src)).strip()
                            if w and not w.startswith("-") and "$" not in w:
                                refs.append(w)
        stack.extend(n.children)
    return [r for r in list(dict.fromkeys(refs)) if r]


def _extract_shell_sql(root, src: bytes, variables: dict[str, str]):
    """SqlResult merged from every `bteq` heredoc in a shell file (source order),
    or None if the file has no bteq SQL. Each body is interpolated then run
    through sqlproc end-to-end (preprocess() strips .LOGON / dot-commands first,
    so credentials never reach the graph). Parse-only — no execution."""
    from . import sqlproc

    redirects = []
    stack = [root]
    while stack:
        n = stack.pop()
        if n.type == "heredoc_redirect":
            redirects.append(n)
        stack.extend(n.children)
    redirects.sort(key=lambda n: n.start_byte)  # source order

    merged = None
    for hr in redirects:
        if not _heredoc_consumer_is_bteq(hr, src):
            continue
        body = _child_of_type(hr, "heredoc_body")
        if body is None:
            continue
        start = _child_of_type(hr, "heredoc_start")
        delim = _text(start, src) if start is not None else ""
        raw = _text(body, src)
        # Quoted delimiter (<<'EOF' / <<"EOF") disables expansion; otherwise expand.
        sql_text = raw if delim[:1] in ("'", '"') else _interpolate_heredoc(raw, variables)
        res = sqlproc.extract(sql_text)
        if merged is None:
            merged = res
        else:
            _merge_sql_result(merged, res)

    file_refs = _collect_bteq_file_refs(root, src)  # Part B: bteq < x.sql / cat x.sql | bteq
    if merged is None and not file_refs:
        return None
    if merged is None:
        merged = sqlproc.SqlResult()  # carry just the run-references (no inline SQL)
    merged.references.extend(file_refs)
    merged.proc_calls = list(dict.fromkeys(merged.proc_calls))
    merged.references = list(dict.fromkeys(merged.references))
    if not (merged.objects or merged.table_refs or merged.proc_calls or merged.references
            or merged.commented is not None):  # keep a file whose only SQL is commented
        return None
    return merged


def _parse_sql(sf: ScannedFile) -> FileParse:
    """Parse a standalone SQL/BTEQ file via sqlproc (table-level lineage)."""
    from . import sqlproc
    try:
        text = Path(sf.abs_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return FileParse(sf.rel_path, sf.language, None, False, [], [])
    result = sqlproc.extract(text)
    parse_ok = bool(result.objects or result.table_refs or result.proc_calls)
    return FileParse(sf.rel_path, sf.language, None, parse_ok, [], [], [], {}, sql=result)


def _parse_joblist(sf: ScannedFile) -> FileParse:
    """Parse a .list job manifest: one shell-script filename per line."""
    calls: list[str] = []
    try:
        text = Path(sf.abs_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return FileParse(sf.rel_path, sf.language, None, False, [], [])
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        code = stripped.split("#", 1)[0].strip()  # drop inline comments
        if code:
            calls.append(code.split()[0])  # filename is the first token
    return FileParse(sf.rel_path, sf.language, None, True, [], [], calls, {})


def parse_file(sf: ScannedFile) -> FileParse:
    """Parse a single scanned file; degrade gracefully, never raise."""
    if sf.language == "joblist":
        return _parse_joblist(sf)
    if sf.language == "sql":
        return _parse_sql(sf)
    grammar_key = GRAMMAR_BY_EXT.get(sf.ext)
    if grammar_key is None and sf.language == "shell":
        grammar_key = "shell"  # extensionless shell scripts (shebang-detected)
    empty = FileParse(sf.rel_path, sf.language, grammar_key, False, [], [])
    if grammar_key is None:
        return empty  # recognized but non-parsed type → file node only
    lang = _get_language(grammar_key)
    if lang is None:
        return empty
    try:
        src = Path(sf.abs_path).read_bytes()
    except OSError:
        return empty
    try:
        from tree_sitter import Parser

        tree = Parser(lang).parse(src)
        symbols = _collect_symbols(tree.root_node, src, grammar_key)
        sql = None
        if grammar_key == "shell":
            imports, script_calls, variables = _collect_shell_refs(tree.root_node, src)
            # Phase 3: SQL embedded in bteq heredocs -> table-ops on the shell file.
            sql = _extract_shell_sql(tree.root_node, src, variables)
        else:
            imports = _collect_imports(tree.root_node, src, grammar_key)
            script_calls, variables = [], {}
        return FileParse(sf.rel_path, sf.language, grammar_key, True, symbols,
                         imports, script_calls, variables, sql=sql)
    except Exception as exc:  # noqa: BLE001 — degrade to file-only on any failure
        log.debug("parse failed for %s: %s", sf.rel_path, exc)
        return empty
