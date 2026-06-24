"""Shared Teradata SQL processing: BTEQ preprocessing + table-level lineage.

Pure-local and read-only: parses SQL text with sqlglot's ``teradata`` dialect to
extract created objects (tables/views/procedures/macros) and table-level lineage
(target reads_from sources). It never executes SQL and never connects to a
database. Used by Phase 2 (standalone .sql/.bteq/.btq files) and reused by
Phase 3 (SQL blocks extracted from shell heredocs).

Robustness model: each statement is parsed independently inside a try/except, so
one unparseable statement (intentionally-broken SQL, an unsupported #-directive,
etc.) is skipped without affecting the rest — extraction is best-effort and never
fatal.
"""

from __future__ import annotations

import collections
import logging
import re
from dataclasses import dataclass, field

# sqlglot logs a warning whenever it falls back to a generic Command on
# unsupported syntax (e.g. COLLECT STATISTICS). Silence it — we handle those.
logging.getLogger("sqlglot").setLevel(logging.CRITICAL)

try:
    import sqlglot
    from sqlglot import errors, exp
    _SQLGLOT_OK = True
except Exception:  # pragma: no cover - degrade if sqlglot unavailable
    _SQLGLOT_OK = False

_DIALECT = "teradata"
_MAX_STATEMENTS = 5000
_MAX_STMT_CHARS = 200_000


@dataclass
class SqlObject:
    name: str   # normalized identifier (db.name or name), original case preserved
    kind: str   # "table" | "view" | "procedure" | "macro"


@dataclass
class SqlResult:
    objects: list[SqlObject] = field(default_factory=list)            # created objects
    lineage: list[tuple[str, list[str]]] = field(default_factory=list)  # (target, sources)
    proc_calls: list[str] = field(default_factory=list)              # exec/call proc names
    references: list[str] = field(default_factory=list)              # .RUN FILE targets
    table_refs: set[str] = field(default_factory=set)                # every table referenced
    table_ops: dict = field(default_factory=dict)                    # table -> ordered [read|write|purge]
    used_columns: dict = field(default_factory=dict)                 # table -> set[str] of referenced columns
    column_lineage: list = field(default_factory=list)               # (tgt_table, tgt_col, src_table, src_col)
    stats: dict = field(default_factory=dict)                        # parsed/skipped counts
    commented: "SqlResult | None" = None                             # entities recovered from `--`-commented SQL


# --------------------------------------------------------------- BTEQ preprocess
_DOTCMD_RE = re.compile(r"^\s*\.\s*[A-Za-z]")
_RUNFILE_RE = re.compile(r"^\s*\.\s*RUN\s+FILE\s*=?\s*(\S+)", re.IGNORECASE)


def _clean_ref(raw: str) -> str:
    return raw.strip().rstrip(";").strip().strip('"').strip("'")


def preprocess(text: str) -> tuple[list[str], list[str]]:
    """Strip BTEQ dot-commands + comments and split into statements.

    Returns (statements, references). ``references`` are ``.RUN FILE=`` targets.
    Statements still contain the literal ``${VAR}`` / ``:param`` sigils; the
    extractor protects them per-statement just before sqlglot parsing.
    """
    references: list[str] = []
    kept: list[str] = []
    for line in text.splitlines():
        if _DOTCMD_RE.match(line):
            m = _RUNFILE_RE.match(line)
            if m:
                references.append(_clean_ref(m.group(1)))
            continue  # drop every BTEQ dot-command line
        kept.append(line)
    return _split_statements("\n".join(kept)), references


def _split_statements(sql: str) -> list[str]:
    """Split on ';' and strip -- / /* */ comments, respecting string literals."""
    out: list[str] = []
    buf: list[str] = []
    i, n, quote = 0, len(sql), None
    while i < n:
        c = sql[i]
        if quote is not None:
            buf.append(c)
            if c == quote:
                if i + 1 < n and sql[i + 1] == quote:  # doubled-quote escape
                    buf.append(sql[i + 1]); i += 2; continue
                quote = None
            i += 1; continue
        if c in ("'", '"'):
            quote = c; buf.append(c); i += 1; continue
        if c == "-" and i + 1 < n and sql[i + 1] == "-":   # line comment (incl --#..#)
            while i < n and sql[i] != "\n":
                i += 1
            continue
        if c == "/" and i + 1 < n and sql[i + 1] == "*":   # block comment
            i += 2
            while i + 1 < n and not (sql[i] == "*" and sql[i + 1] == "/"):
                i += 1
            i += 2; continue
        if c == ";":
            s = "".join(buf).strip()
            if s:
                out.append(s)
            buf = []; i += 1; continue
        buf.append(c); i += 1
    s = "".join(buf).strip()
    if s:
        out.append(s)
    return out


# --------------------------------------------------------------------- sigils
# $VAR / ${VAR} (optionally backslash-escaped) and :param bind variables.
_SIGIL_RE = re.compile(r"\\?\$\{[^}]*\}|\\?\$[A-Za-z_]\w*")
_BIND_RE = re.compile(r":(?![=:])[A-Za-z_]\w*")


def _protect_sigils(stmt: str) -> tuple[str, dict[str, str]]:
    """Replace $VAR/${VAR} with parse-safe identifiers; map them back later.

    A backslash-escaped ``\\${VAR}`` is treated as a literal ``${VAR}``. ``:param``
    binds are scalar (irrelevant to lineage) and collapse to a constant token.
    """
    mapping: dict[str, str] = {}
    rev: dict[str, str] = {}
    counter = [0]

    def repl(m: re.Match) -> str:
        orig = m.group(0)
        if orig.startswith("\\"):
            orig = orig[1:]  # \${VAR} -> literal ${VAR}
        if orig in rev:
            return rev[orig]
        ph = f"zqsig{counter[0]}"
        counter[0] += 1
        mapping[ph] = orig
        rev[orig] = ph
        return ph

    protected = _SIGIL_RE.sub(repl, stmt)
    protected = _BIND_RE.sub("zqbind", protected)
    return protected, mapping


def _restore(name: str, mapping: dict[str, str]) -> str:
    for ph, orig in mapping.items():
        if ph in name:
            name = name.replace(ph, orig)
    return name


def _norm(name: str | None) -> str:
    if not name:
        return ""
    return name.replace('"', "").replace("`", "").strip()


# --------------------------------------------------------------- temporal syntax
# Teradata temporal qualifiers/predicates that sqlglot's dialect does not parse.
# They are filters, not table references, so neutralizing them before parsing
# cannot change WHICH tables are read/written — it only lets sqlglot recover the
# tables (otherwise the statement fails-and-skips, or a keyword leaks as a fake
# table; see the phantom guard below).
_VALIDTIME_RE = re.compile(r"\b(?:NONSEQUENCED|SEQUENCED|CURRENT)?\s*VALIDTIME\b", re.IGNORECASE)
# NOTE: this PERIOD-predicate regex assumes NON-NESTED parens in the PERIOD
# operands and the comparison operand — true in the real Teradata samples (the
# operands are simple column refs, e.g. PERIOD(z.vt_start, z.vt_end) CONTAINS
# (b.event_dt)). If a nested-paren PERIOD predicate ever appears it will NOT be
# neutralized; the phantom-table guard (_PHANTOM) then makes the statement skip
# cleanly rather than emit a corrupt edge.
_PERIOD_PRED_RE = re.compile(
    r"PERIOD\s*\([^)]*\)\s*(?:CONTAINS|OVERLAPS|MEETS|PRECEDES|SUCCEEDS)\s*\([^)]*\)",
    re.IGNORECASE,
)


def _neutralize_temporal(stmt: str) -> str:
    """Strip (NONSEQUENCED|SEQUENCED|CURRENT)? VALIDTIME and replace a
    PERIOD(...) <op> (...) predicate with 1=1 (keeping the surrounding ON/WHERE
    clause valid). Applied per-statement just before sigil protection + parse."""
    s = _VALIDTIME_RE.sub(" ", stmt)
    return _PERIOD_PRED_RE.sub("1=1", s)


# In-house '#' directives seen in the real BTEQ-style files. `#insert into X` is
# the dominant write form, so we rewrite it to real SQL (and capture the target
# even if the SELECT body is garbled). `#primary index(...)` is DDL metadata
# (dropped). Other inline `#IDENT` (column aliases like #AWK_DNW_EP) are turned
# into a parse-safe token so the leading '#' is not read as a comment.
_HASH_PRIMARY_IDX_RE = re.compile(r"#\s*primary\s+index\s*\([^)]*\)", re.IGNORECASE)
_HASH_INSERT_RE = re.compile(r"#\s*insert\s+into\s+([A-Za-z0-9_.\"${}]+)", re.IGNORECASE)
_HASH_IDENT_RE = re.compile(r"#([A-Za-z_]\w*)")


def _normalize_directives(stmt: str) -> tuple[str, str | None]:
    """Rewrite `#`-directives to parseable SQL. Returns (stmt, implicit_insert_target)."""
    s = _HASH_PRIMARY_IDX_RE.sub(" ", stmt)
    m = _HASH_INSERT_RE.search(s)
    implicit_target = _norm(m.group(1)) if m else None
    if m:
        s = _HASH_INSERT_RE.sub(r"INSERT INTO \1", s)
    s = _HASH_IDENT_RE.sub(r"hash_\1", s)  # remaining inline #IDENT -> parse-safe token
    return s, implicit_target


# Reserved words that can leak from un-neutralized temporal syntax as a fake
# "table" (e.g. a missed temporal qualifier where sqlglot reads `NONSEQUENCED`
# as a FROM item). A silent wrong lineage edge is worse than a skip, so any such
# leak is dropped. Kept tight (exact bare-word match) so a real table such as
# `db.period` is never dropped.
_PHANTOM = {"nonsequenced", "sequenced", "current", "validtime", "period"}


def _is_phantom(name: str) -> bool:
    return name.strip().lower() in _PHANTOM


# ------------------------------------------------------------------ extraction
_CREATE_PROC_RE = re.compile(
    r"^\s*(?:CREATE|REPLACE)\s+(?:PROCEDURE|MACRO|FUNCTION)\s+([A-Za-z0-9_.\"${}]+)",
    re.IGNORECASE,
)
_EXEC_RE = re.compile(r"^\s*(?:EXEC(?:UTE)?|CALL)\s+([A-Za-z0-9_.\"${}]+)", re.IGNORECASE)


def _table_name(node) -> str | None:
    try:
        if node is None:
            return None
        if isinstance(node, exp.Table):
            return exp.table_name(node)
        t = node.find(exp.Table)
        return exp.table_name(t) if t is not None else None
    except Exception:
        return None


def _all_sources(e, target_norm: str, sigmap: dict[str, str]) -> list[str]:
    cte = {c.alias for c in e.find_all(exp.CTE)}
    sources: list[str] = []
    for t in e.find_all(exp.Table):
        nm = exp.table_name(t)
        if not nm or nm in cte:
            continue
        norm = _norm(_restore(nm, sigmap))
        if norm and norm != target_norm and norm not in sources and not _is_phantom(norm):
            sources.append(norm)
    return sources


def _table_node(node):
    """The exp.Table node within `node` (the write/purge target position), or None."""
    if node is None:
        return None
    if isinstance(node, exp.Table):
        return node
    try:
        return node.find(exp.Table)
    except Exception:
        return None


def _read_tables(e, target_node, sigmap: dict[str, str]) -> list[str]:
    """Tables read by the statement: every table position except the write/purge
    target node. A self-referenced target (INSERT INTO T SELECT ... FROM T) is
    kept, so the op order can place its read before its write."""
    cte = {c.alias for c in e.find_all(exp.CTE)}
    out: list[str] = []
    for t in e.find_all(exp.Table):
        if t is target_node:
            continue
        nm = exp.table_name(t)
        if not nm or nm in cte:
            continue
        norm = _norm(_restore(nm, sigmap))
        if norm and not _is_phantom(norm) and norm not in out:
            out.append(norm)
    return out


def _append_op(result: SqlResult, table: str, op: str) -> None:
    """Append `op` to the table's ordered (execution-order) op sequence for this
    file, collapsing only *consecutive* duplicates: write,write,read -> write,read
    while write,read,write is preserved."""
    t = (table or "").strip()
    if not t or _is_phantom(t):
        return
    result.table_refs.add(t)
    seq = result.table_ops.setdefault(t, [])
    if not seq or seq[-1] != op:
        seq.append(op)


# --------------------------------------------------------------- column lineage
# C1/C2: extract the *used* columns per table (columns referenced in the scripts,
# not a full DDL schema) and column-to-column lineage for INSERT ... SELECT /
# CREATE ... AS SELECT. Purely textual over the sqlglot AST — no DB, no schema,
# no execution. Names are restored from the sigil map exactly like table names,
# and the phantom guard drops temporal keywords that leak as fake identifiers.
def _ident_text(node) -> str:
    """Best-effort identifier text from an Identifier / Column / wrapper node."""
    try:
        if node is None:
            return ""
        return node.name if hasattr(node, "name") else ""
    except Exception:
        return ""


def _alias_table_map(scope, sigmap: dict[str, str]) -> tuple[dict[str, str], str | None]:
    """Map every table alias / name written in ``scope`` to its real (restored,
    normalized) table name, and return the single source table when the scope has
    exactly one (so an unqualified column can be attributed without guessing).
    CTE names are excluded so a WITH alias is never mistaken for a table."""
    cte = {c.alias for c in scope.find_all(exp.CTE)}
    amap: dict[str, str] = {}
    reals: list[str] = []
    for t in scope.find_all(exp.Table):
        nm = exp.table_name(t)
        if not nm or nm in cte:
            continue
        real = _norm(_restore(nm, sigmap))
        if not real or _is_phantom(real):
            continue
        if real not in reals:
            reals.append(real)
        alias = _norm(_restore(t.alias, sigmap)) if t.alias else ""
        if alias:
            amap[alias] = real
        amap.setdefault(_norm(_restore(t.name, sigmap)), real)  # bare last identifier
        amap.setdefault(real, real)                              # fully-qualified name
    single = reals[0] if len(reals) == 1 else None
    return amap, single


def _column_owner(c, amap: dict[str, str], single: str | None,
                  sigmap: dict[str, str]) -> tuple[str, str] | None:
    """(table, column) for an exp.Column, resolving its qualifier through ``amap``.
    An unqualified column attaches to ``single`` only when the scope is
    unambiguous; otherwise it is skipped (best-effort, never guess in a join)."""
    col = _norm(_restore(c.name, sigmap))
    if not col or _is_phantom(col):
        return None
    qual = _norm(_restore(c.table, sigmap)) if c.table else ""
    tbl = amap.get(qual) if qual else single
    if not tbl or _is_phantom(tbl):
        return None
    return tbl, col


def _collect_used_columns(e, sigmap: dict[str, str], result: SqlResult) -> None:
    """C1 — record every column referenced in the statement, attributed to its
    table. Qualified columns resolve via the alias map; unqualified columns only
    when the statement touches a single table."""
    amap, single = _alias_table_map(e, sigmap)
    if not amap:
        return
    for c in e.find_all(exp.Column):
        owner = _column_owner(c, amap, single, sigmap)
        if owner is not None:
            result.used_columns.setdefault(owner[0], set()).add(owner[1])


def _proj_output_name(proj, sigmap: dict[str, str]) -> str:
    """Output column name of a SELECT projection: its alias, else the bare column
    name. '' for an unaliased expression / star (no nameable output column)."""
    if isinstance(proj, exp.Alias):
        return _norm(_restore(proj.alias, sigmap))
    if isinstance(proj, exp.Column):
        return _norm(_restore(proj.name, sigmap))
    return ""


def _select_for_lineage(e):
    """The SELECT whose projections define the written columns (INSERT ... SELECT
    / CREATE ... AS SELECT). The first SELECT of a UNION supplies the names."""
    q = e.expression
    if q is None:
        return None
    if isinstance(q, exp.Subquery):
        q = q.this
    return q if isinstance(q, exp.Select) else q.find(exp.Select)


def _insert_target_columns(e, sigmap: dict[str, str]) -> list[str]:
    """Explicit INSERT column list — INSERT INTO t (a, b) — or [] when implicit."""
    this = e.this if isinstance(e, exp.Insert) else None
    if isinstance(this, exp.Schema):
        return [_norm(_restore(_ident_text(x), sigmap)) for x in this.expressions]
    return []


def _collect_column_lineage(e, target_table: str, sigmap: dict[str, str],
                            result: SqlResult) -> None:
    """C2 — link each written column of ``target_table`` to the source column(s)
    it derives from. Best-effort: a projection whose output column cannot be named
    or whose sources cannot be resolved simply contributes no edge."""
    if not target_table or _is_phantom(target_table):
        return
    sel = _select_for_lineage(e)
    if sel is None:
        return
    projections = [p for p in sel.expressions if not isinstance(p, exp.Star)]
    if not projections:
        return
    amap, single = _alias_table_map(sel, sigmap)  # source scope = the SELECT only
    explicit = _insert_target_columns(e, sigmap)
    for i, proj in enumerate(projections):
        out_col = explicit[i] if i < len(explicit) else _proj_output_name(proj, sigmap)
        if not out_col or _is_phantom(out_col):
            continue
        result.used_columns.setdefault(target_table, set()).add(out_col)
        seen: set[tuple[str, str]] = set()
        for c in proj.find_all(exp.Column):
            owner = _column_owner(c, amap, single, sigmap)
            if owner is None or owner in seen:
                continue
            seen.add(owner)
            src_tbl, src_col = owner
            result.used_columns.setdefault(src_tbl, set()).add(src_col)
            result.column_lineage.append((target_table, out_col, src_tbl, src_col))


# Standalone read-only query nodes (their tables are all reads).
_QUERY_TYPES = (exp.Select, exp.Union, exp.Intersect, exp.Except)


def _handle(e, stmt: str, sigmap: dict[str, str], result: SqlResult) -> bool:
    # Op order within a statement = execution order: the source query is read
    # first, then the target is written/purged (so INSERT INTO T SELECT FROM T
    # records read T before write T).
    try:
        _collect_used_columns(e, sigmap, result)  # C1 — used columns per table
    except Exception:
        pass
    if isinstance(e, exp.Create):
        kind = (e.args.get("kind") or "").upper()
        if kind in ("TABLE", "VIEW"):
            tnode = _table_node(e.this)
            target = _norm(_restore(_table_name(e.this) or "", sigmap))
            if not target or _is_phantom(target):
                return False
            result.objects.append(SqlObject(target, "view" if kind == "VIEW" else "table"))
            for r in _read_tables(e, tnode, sigmap):  # CTAS reads...
                _append_op(result, r, "read")
            _append_op(result, target, "write")       # ...then create/write
            sources = _all_sources(e, target, sigmap)  # CTAS lineage (excludes self)
            if sources:
                result.lineage.append((target, sources))
            try:
                _collect_column_lineage(e, target, sigmap, result)  # C2 — CTAS columns
            except Exception:
                pass
            return True
        if kind in ("PROCEDURE", "MACRO", "FUNCTION"):
            m = _CREATE_PROC_RE.match(stmt)
            name = _norm(m.group(1)) if m else ""
            if name:
                result.objects.append(SqlObject(name, "macro" if kind == "MACRO" else "procedure"))
                return True
            return False
        return False
    if isinstance(e, (exp.Insert, exp.Merge, exp.Update)):  # read sources, then write target
        tnode = _table_node(e.this)
        target = _norm(_restore(_table_name(e.this) or "", sigmap))
        if not target or _is_phantom(target):
            return False
        for r in _read_tables(e, tnode, sigmap):
            _append_op(result, r, "read")
        _append_op(result, target, "write")
        sources = _all_sources(e, target, sigmap)
        if sources:
            result.lineage.append((target, sources))
        if isinstance(e, exp.Insert):
            try:
                _collect_column_lineage(e, target, sigmap, result)  # C2 — INSERT...SELECT columns
            except Exception:
                pass
        return True
    if isinstance(e, (exp.Delete, exp.TruncateTable)):  # read subquery tables, then purge target
        tnode = _table_node(e.this) or e.find(exp.Table)  # DELETE <t> has this=None
        target = _norm(_restore(exp.table_name(tnode) if tnode is not None else "", sigmap))
        for r in _read_tables(e, tnode, sigmap):  # tables read in a WHERE/USING subquery
            _append_op(result, r, "read")
        if target and not _is_phantom(target):
            _append_op(result, target, "purge")
        return True
    if isinstance(e, _QUERY_TYPES):  # standalone SELECT / set-operation -> reads only
        for r in _read_tables(e, None, sigmap):
            _append_op(result, r, "read")
        return True
    return False


# ----------------------------------------------------- commented-SQL recovery
# Shell orchestrators document the SQL that really executes as `--`-commented
# blocks — a called macro's body after an `exec`/`call`, or a generated Teradata
# request with no preceding exec:
#     exec $DB_x.PROC(...)
#     -- Statement 1: <prose>:
#     -- insert into $DB_z.t (cols) select ... from $DB_y.stg ;
# This commented SQL is NOT dead code: it faithfully describes real execution, so
# we read it AS IF UNCOMMENTED and surface its tables/columns/lineage (the graph
# tags them "commented"). preprocess() drops these lines as comments, so we
# recover them separately into a sibling SqlResult: open a region on EVERY
# contiguous `--` block (an exec/call line is a non-exclusive hint that also opens
# one), lift one comment level off each `--` line, drop `Statement N:` prose
# headers, split on `Statement N:` / nested exec / bare `;`, and re-feed each
# block to the normal extractor. The gate is now a minimal parse-sanity check
# (extract_statements with recovered=True): a block is kept iff it parses into
# >=1 real DML/DDL statement (delete/insert/update/create/merge); pure prose that
# does not parse is skipped. Recovered entities are TAGGED in the graph, so
# strictness moves from "reject aggressively" to "accept and label" — residual
# risk (a prose line parsing as DML) is contained by the tag, never exposure.
_COMMENT_LINE_RE = re.compile(r"^\s*--\s?")
_STMT_HEADER_RE = re.compile(r"^\s*Statement\s+\d+\s*:", re.IGNORECASE)
_BARE_TERMINATOR_RE = re.compile(r"^[\s;]*$")  # blank, or only ';' -> statement boundary
_RECOVERED_KINDS = (exp.Insert, exp.Merge, exp.Update, exp.Delete, exp.TruncateTable, exp.Create)


def _recover_commented_statements(text: str) -> list[str]:
    """Lift EVERY contiguous `--`-commented SQL block into candidate statement(s).

    A region opens on any `--` comment line (an exec/call line is a non-exclusive
    hint that also opens one) and stays open across comment lines, blank lines, and
    bare `;` terminators until a real (non-comment) line resumes. Inside it, each
    `--` line is un-commented one level and a `Statement N:` header or nested
    exec/call line is dropped and flushes the previous statement; a bare `;`
    flushes the accumulated lines as one candidate. Candidates are validated
    downstream by parsing, so prose that does not parse into real SQL is discarded
    there rather than here."""
    out: list[str] = []
    buf: list[str] = []
    in_region = False

    def flush() -> None:
        block = "\n".join(buf).strip()
        buf.clear()
        if block:
            out.append(block)

    for line in text.splitlines():
        if _EXEC_RE.match(line):              # exec/call -> hint: (re)open a region
            flush()
            in_region = True
            continue
        if _COMMENT_LINE_RE.match(line):      # ANY '--' line opens/continues a region
            in_region = True
            body = _COMMENT_LINE_RE.sub("", line, count=1)
            # A 'Statement N:' header or a nested exec/call ends the previous
            # statement and is itself dropped (a header is prose; a call is not a
            # body). This splits multi-statement blocks that share a single ';'
            # terminator, and stops a documented nested call from being glued onto
            # — and thereby suppressing — the real DML that follows it.
            if _STMT_HEADER_RE.match(body) or _EXEC_RE.match(body):
                flush()
            else:
                buf.append(body)
            continue
        if not in_region:
            continue
        if _BARE_TERMINATOR_RE.match(line):   # blank or bare ';' inside a region
            if line.strip():                  # a ';' ends a statement; a blank does not
                flush()
            continue
        flush()                               # a real (non-comment) line ends the region
        in_region = False
    flush()
    return out


# ------------------------------------------------- parse-failure categorization
# A FIXED set of category codes for WHY a statement failed to parse. These are
# constant strings chosen in code — NEVER derived from SQL text or from a
# sqlglot error message (those embed the offending snippet and must never be
# stored or emitted). The classifier below inspects the in-memory text and the
# exception type with cheap deterministic probes and returns exactly ONE code;
# only the code (and an incremented counter) ever leaves this module.
SQL_FAIL_CATEGORIES = (
    "sigil_interpolation",    # unprotected ${...}/&var/$var substitution marker
    "bteq_dot_command",       # a BTEQ dot-command reached the parser
    "missing_terminator",     # unbalanced quotes/parens — unterminated statement
    "unsupported_teradata",   # sqlglot recognized a command but could not parse it
    "tokenizer_error",        # failure at the tokenize stage (vs parse stage)
    "empty_after_preprocess", # nothing left to parse after preprocessing
    "other",                  # uncategorized
)

# Substitution markers _protect_sigils does NOT neutralize (it handles
# $VAR/${VAR}/:bind). A surviving ${...}, bare $name, or &macro in the text the
# parser actually saw is the signal that interpolation broke tokenization.
_LEFTOVER_SIGIL_RE = re.compile(r"\$\{|\$[A-Za-z_]|&[A-Za-z_]")


def _is_tokenize_error(exc) -> bool:
    if exc is None or not _SQLGLOT_OK:
        return False
    # sqlglot names this TokenError (older trees: TokenizeError) — tolerate both.
    tok = getattr(errors, "TokenError", None) or getattr(errors, "TokenizeError", None)
    try:
        return tok is not None and isinstance(exc, tok)
    except Exception:
        return False


def _looks_unterminated(s: str) -> bool:
    """Cheap balance probe: an unterminated statement usually has mismatched
    parentheses or an odd number of single quotes."""
    return s.count("(") != s.count(")") or (s.count("'") % 2 == 1)


def _classify_sql_failure(seen: str, e, exc) -> str:
    """Map a parse failure to ONE fixed category code from the in-memory text
    (``seen`` is the protected text the parser actually received) and the
    exception type. Returns a code from SQL_FAIL_CATEGORIES — never any SQL,
    identifier, or error message."""
    s = (seen or "").strip()
    if not s:
        return "empty_after_preprocess"
    if _is_tokenize_error(exc):
        return "tokenizer_error"
    if _DOTCMD_RE.match(s):
        return "bteq_dot_command"
    if _LEFTOVER_SIGIL_RE.search(s):
        return "sigil_interpolation"
    if _looks_unterminated(s):
        return "missing_terminator"
    if _SQLGLOT_OK and e is not None and isinstance(e, exp.Command):
        return "unsupported_teradata"
    return "other"


def extract_statements(statements: list[str], result: SqlResult, *, recovered: bool = False) -> None:
    """Extract objects/lineage from preprocessed statements into ``result``.

    With ``recovered=True`` the statements come from lifted `--`-commented SQL:
    each is kept iff it parses into a real DML/DDL statement (a minimal parse-sanity
    gate); pure prose that does not parse is skipped. Stats accumulate across calls,
    so a recovery pass extends the first pass."""
    parsed = (result.stats or {}).get("parsed", 0)
    skipped = (result.stats or {}).get("skipped", 0)
    by_type: collections.Counter = collections.Counter((result.stats or {}).get("by_type", {}))
    recovery: collections.Counter = collections.Counter((result.stats or {}).get("recovery", {}))
    # Per-surface parse-failure category tallies (fixed codes only; see
    # _classify_sql_failure). active = real statements; commented = the
    # parse_failed slice of the commented-SQL recovery, refined into families.
    fail_active: collections.Counter = collections.Counter((result.stats or {}).get("fail_active", {}))
    fail_commented: collections.Counter = collections.Counter((result.stats or {}).get("fail_commented", {}))
    for stmt in statements[:_MAX_STATEMENTS]:
        if len(stmt) > _MAX_STMT_CHARS:
            skipped += 1; by_type["too_long"] += 1
            if recovered:  # a candidate too long to parse still counts as attempted
                recovery["detected"] += 1; recovery["reject_other"] += 1
            continue
        stmt, implicit_target = _normalize_directives(stmt)  # #insert into / #primary index / #IDENT
        stmt = _neutralize_temporal(stmt)  # Teradata VALIDTIME / PERIOD predicates
        em = _EXEC_RE.match(stmt)
        if em and not recovered:  # a recovered block is a macro body, not a call
            result.proc_calls.append(_norm(em.group(1)))
            parsed += 1; by_type["exec/call"] += 1; continue
        protected, sigmap = _protect_sigils(stmt)
        e = None
        parse_exc: Exception | None = None
        try:
            e = sqlglot.parse_one(protected, read=_DIALECT, error_level=errors.ErrorLevel.IGNORE)
        except Exception as exc:
            e = None
            parse_exc = exc
        if recovered:
            # Minimal parse-sanity gate: a candidate is kept iff it parses into a
            # real DML/DDL statement; pure prose is skipped. Recovered entities are
            # tagged "commented" in the graph, so we accept-and-label rather than
            # reject aggressively. Reject reasons: parse_failed / no_dml / other.
            recovery["detected"] += 1
            ok = False
            reason: str | None = None
            if e is None or isinstance(e, exp.Command):
                reason = "parse_failed"            # sqlglot produced no usable statement
                # Refine the parse_failed bucket into a fixed failure family.
                fail_commented[_classify_sql_failure(protected, e, parse_exc)] += 1
            elif not isinstance(e, _RECOVERED_KINDS):
                reason = "no_dml"                  # parsed, but not delete/insert/update/create
            else:
                try:
                    ok = _handle(e, stmt, sigmap, result)
                except Exception:
                    ok = False
                if not ok:
                    reason = "other"               # parsed as DML but nothing extractable
            if ok:
                parsed += 1; by_type["recovered_" + type(e).__name__] += 1
                recovery["recovered"] += 1; recovery["statements"] += 1
            else:
                skipped += 1; by_type["recovered_skipped"] += 1
                recovery["reject_" + reason] += 1
            continue
        ok = False
        if e is not None and not isinstance(e, exp.Command):
            try:
                ok = _handle(e, stmt, sigmap, result)
            except Exception:
                ok = False
        if ok:
            parsed += 1; by_type[type(e).__name__] += 1
        elif implicit_target and not _is_phantom(implicit_target):
            _append_op(result, implicit_target, "write")  # #insert recovered even if body unparsable
            parsed += 1; by_type["hash_insert"] += 1
        else:
            skipped += 1
            label = ("parse_error" if e is None else
                     "command/empty" if isinstance(e, exp.Command) else
                     "unhandled_" + type(e).__name__)
            by_type[label] += 1
            # Categorize the active-statement failure into a fixed family. This
            # covers every failure surface (parse_error / command / unhandled),
            # not only the e-is-None subset the legacy parse_failures counts.
            fail_active[_classify_sql_failure(protected, e, parse_exc)] += 1
    result.proc_calls = list(dict.fromkeys(result.proc_calls))
    result.stats = {"parsed": parsed, "skipped": skipped, "by_type": dict(by_type),
                    "recovery": dict(recovery),
                    "fail_active": dict(fail_active), "fail_commented": dict(fail_commented)}


_SQL_COMMENT_LINE_RE = re.compile(r"^\s*--")


def _count_comment_lines(text: str) -> tuple[int, int]:
    """(total lines, full-line ``--`` comment lines) of a raw SQL block, counted on
    the RAW text BEFORE comment stripping. A full-line comment has ``--`` as its
    first non-blank characters; ``/* */`` blocks, shell ``#`` and inline trailing
    ``--`` are NOT counted."""
    lines = text.splitlines()
    return len(lines), sum(1 for ln in lines if _SQL_COMMENT_LINE_RE.match(ln))


def merge_stats(into: dict, other: dict) -> dict:
    """Sum two tally-only stats dicts: scalar counts plus the by_type / recovery
    Counters. Diagnostics-only — stats never affect extraction output — used when a
    shell file's multiple bteq heredocs collapse into one SqlResult."""
    out = dict(into or {})
    for key in ("parsed", "skipped", "total_lines", "comment_lines"):
        out[key] = int(out.get(key, 0)) + int((other or {}).get(key, 0))
    for nested in ("by_type", "recovery", "fail_active", "fail_commented"):
        merged = collections.Counter(out.get(nested, {}))
        merged.update((other or {}).get(nested, {}))
        out[nested] = dict(merged)
    return out


def extract(text: str) -> SqlResult:
    """Preprocess + extract from raw SQL/BTEQ text (the Phase-2/3 entry point)."""
    result = SqlResult()
    if not _SQLGLOT_OK:
        result.stats = {"parsed": 0, "skipped": 0, "sqlglot_available": False}
        return result
    statements, references = preprocess(text)
    result.references = references
    extract_statements(statements, result)
    # Read `--`-commented SQL as real SQL into a SEPARATE sibling result, so the
    # graph can tag its entities "commented" and de-dup them against active ones.
    recovered = _recover_commented_statements(text)
    if recovered:
        candidates: list[str] = []
        for block in recovered:
            candidates.extend(_split_statements(block))
        if candidates:
            commented = SqlResult()
            extract_statements(candidates, commented, recovered=True)
            if commented.table_refs or commented.objects or commented.column_lineage:
                result.commented = commented              # entities for build_graph
            result.stats = merge_stats(result.stats, commented.stats)  # recovery diagnostics
    # Run-diagnostics (tally-only): full-line `--` comments in the raw block.
    total_lines, comment_lines = _count_comment_lines(text)
    result.stats["total_lines"] = total_lines
    result.stats["comment_lines"] = comment_lines
    return result
