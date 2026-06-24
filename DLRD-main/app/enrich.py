"""LLM enrichment — natural-language summaries + architectural layer per file.

One LLM request per file returns the file summary, an architectural layer
(API / Service / Data / UI / Utility / Other), a complexity rating, tags, and a
one-line summary for each parsed member (function/class). This keeps the call
count to ~one-per-file while still enriching every node. Requests run with
bounded concurrency (<= settings.concurrency, hard-capped at 5). Any failure
falls back to a deterministic, offline summary so analysis never aborts.
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .config import LANGUAGE_NAMES, Settings
from .llm import LLMClient
from .parser import FileParse

log = logging.getLogger("data_lineage_retro_documentation.enrich")

LAYER_CHOICES = ["API", "Service", "Data", "UI", "Utility", "Other"]
COMPLEXITY_CHOICES = {"simple", "moderate", "complex"}

_LAYER_HINTS = [
    ("API", ("route", "router", "controller", "endpoint", "handler", "/api", "rest", "graphql", "server", "urls", "views")),
    ("Data", ("model", "models", "schema", "repository", "repo", "dao", "migration", "entity", "/db", "database", "store", "persistence", "dataset")),
    ("UI", ("component", "components", "/ui", "view", "views", "page", "pages", "widget", "frontend", "/css", "style", "styles", ".vue", ".svelte", ".tsx", ".jsx")),
    ("Service", ("service", "services", "usecase", "use_case", "domain", "logic", "core", "manager", "orchestrat", "worker", "pipeline")),
    ("Utility", ("util", "utils", "helper", "helpers", "config", "settings", "common", "shared", "/lib", "tool", "tools", "constant")),
]


@dataclass
class FileEnrichment:
    summary: str
    layer: str
    complexity: str
    tags: list[str]
    members: dict[str, dict] = field(default_factory=dict)  # name -> {summary, complexity}
    llm_ok: bool = True
    # Diagnostics-only outcome flags (set by _enrich_one). Behavior-neutral.
    distilled: bool = False
    truncated: bool = False


def heuristic_layer(rel_path: str) -> str:
    p = rel_path.lower()
    if "test" in p or "spec" in p or "__tests__" in p:
        return "Other"
    for layer, needles in _LAYER_HINTS:
        if any(n in p for n in needles):
            return layer
    return "Other"


def _normalize_layer(raw: object, rel_path: str) -> str:
    if isinstance(raw, str):
        cleaned = raw.strip().lower()
        for choice in LAYER_CHOICES:
            if cleaned == choice.lower():
                return choice
        aliases = {
            "data": "Data", "database": "Data", "persistence": "Data", "model": "Data",
            "ui": "UI", "frontend": "UI", "presentation": "UI", "view": "UI",
            "api": "API", "controller": "API", "endpoint": "API", "web": "API",
            "service": "Service", "business": "Service", "logic": "Service", "domain": "Service",
            "util": "Utility", "utility": "Utility", "utils": "Utility", "helper": "Utility",
            "config": "Utility", "infrastructure": "Utility",
        }
        if cleaned in aliases:
            return aliases[cleaned]
    return heuristic_layer(rel_path)


def _normalize_complexity(raw: object, default: str = "moderate") -> str:
    if isinstance(raw, str):
        c = raw.strip().lower()
        if c in COMPLEXITY_CHOICES:
            return c
        m = {"low": "simple", "easy": "simple", "medium": "moderate",
             "high": "complex", "hard": "complex", "trivial": "simple"}
        if c in m:
            return m[c]
    return default


def _heuristic_complexity(fp: FileParse) -> str:
    n = len(fp.symbols)
    span = max((s.line_end for s in fp.symbols), default=0)
    if n <= 2 and span < 60:
        return "simple"
    if n >= 12 or span > 400:
        return "complex"
    return "moderate"


def _fallback_enrichment(fp: FileParse) -> FileEnrichment:
    name = Path(fp.rel_path).name
    n_fn = sum(1 for s in fp.symbols if s.kind == "function")
    n_cls = sum(1 for s in fp.symbols if s.kind == "class")
    bits = []
    if n_cls:
        bits.append(f"{n_cls} class{'es' if n_cls != 1 else ''}")
    if n_fn:
        bits.append(f"{n_fn} function{'s' if n_fn != 1 else ''}")
    detail = f" defining {' and '.join(bits)}" if bits else ""
    summary = f"{fp.language} source file '{name}'{detail}."
    members = {
        s.name: {"summary": f"{s.kind.capitalize()} '{s.name}' in {name}.",
                 "complexity": "moderate"}
        for s in fp.symbols
    }
    return FileEnrichment(
        summary=summary,
        layer=heuristic_layer(fp.rel_path),
        complexity=_heuristic_complexity(fp),
        tags=[fp.language],
        members=members,
        llm_ok=False,
    )


def _build_messages(fp: FileParse, content: str, settings: Settings) -> list[dict]:
    sql = getattr(fp, "sql", None)
    if sql is not None and sql.objects:
        member_lines = "\n".join(f"- {o.kind} {o.name}" for o in sql.objects[:80])
    else:
        member_lines = "\n".join(
            f"- {s.kind} {s.name} (lines {s.line_start}-{s.line_end})" for s in fp.symbols[:80]
        ) or "(no functions or classes were parsed)"
    sql_context = (
        "This is a Teradata SQL / BTEQ ETL script — summarize the data "
        "transformation it performs and the tables it reads and writes.\n\n"
        if (fp.language == "sql" or _embedded_sql(fp) is not None) else ""
    )
    language_name = LANGUAGE_NAMES.get(settings.language, "English")
    lang_clause = (
        f"Write every natural-language value (the summary, tags and member "
        f"summaries) in {language_name}. The JSON keys themselves stay in English."
    )
    system = (
        "You are a senior software architect documenting a codebase. "
        "You analyse one source file at a time and reply with ONLY a single JSON "
        "object, no prose, no markdown fences. " + lang_clause
    )
    user = f"""File: {fp.rel_path}
Language: {fp.language}

{sql_context}Parsed members:
{member_lines}

Return a JSON object with EXACTLY these keys:
{{
  "summary": "1-3 sentence explanation of what this file does and its role",
  "layer": "one of: API, Service, Data, UI, Utility, Other",
  "complexity": "one of: simple, moderate, complex",
  "tags": ["3-6 short lowercase keyword tags"],
  "members": [
    {{"name": "<exact member name from the list above>", "summary": "one sentence", "complexity": "simple|moderate|complex"}}
  ]
}}

Source (may be truncated):
```
{content}
```"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _parse_response(data: object, fp: FileParse) -> FileEnrichment:
    if not isinstance(data, dict):
        return _fallback_enrichment(fp)
    summary = str(data.get("summary") or "").strip() or _fallback_enrichment(fp).summary
    layer = _normalize_layer(data.get("layer"), fp.rel_path)
    complexity = _normalize_complexity(data.get("complexity"), _heuristic_complexity(fp))
    tags_raw = data.get("tags")
    tags = [str(t).strip().lower() for t in tags_raw if str(t).strip()][:6] if isinstance(tags_raw, list) else [fp.language]
    if not tags:
        tags = [fp.language]

    members: dict[str, dict] = {}
    raw_members = data.get("members")
    if isinstance(raw_members, list):
        for m in raw_members:
            if isinstance(m, dict) and m.get("name"):
                members[str(m["name"])] = {
                    "summary": str(m.get("summary") or "").strip(),
                    "complexity": _normalize_complexity(m.get("complexity"), "moderate"),
                }
    # ensure every parsed symbol has an entry (deterministic fallback per member)
    lower_index = {k.lower(): v for k, v in members.items()}
    for s in fp.symbols:
        if s.name not in members:
            members[s.name] = lower_index.get(
                s.name.lower(),
                {"summary": f"{s.kind.capitalize()} '{s.name}'.", "complexity": "moderate"},
            )
    return FileEnrichment(summary, layer, complexity, tags, members, llm_ok=True)


# --- credential redaction for the LLM enrichment payload --------------------
# Defense-in-depth (R1/R5): the enrichment path sends raw file content to the
# LLMAAS for summaries, and that content — especially .sh wrappers around bteq —
# can carry secrets (.LOGON credentials, exported passwords). We redact the
# secret token *in place*, keeping the keyword and line structure so the model
# still sees what each line is and the summary quality is unaffected. Patterns
# are line/word-anchored and conservative to avoid mangling ordinary code; add a
# (compiled_pattern, replacement) pair to `_REDACTORS` to cover a new form.
#
# Pattern set (all case-insensitive, value-only redaction, linear-time / no
# nested quantifiers so no ReDoS):
#   1. BTEQ `.LOGON` / `.CONNECT` — line-anchored (leading whitespace and an
#      optional `--` / `#` comment prefix allowed, so a commented-out `-- .LOGON`
#      is redacted too): keep the dot-command, redact the rest of the line.
#   2. Credential-variable assignment `<prefix><KEYWORD>=value` — one consolidated
#      rule (optionally `export `-prefixed). An optional variable-name prefix is
#      allowed, so DB_PASSWORD=, TD_PASSWORD=, ETL_PW=, … all match: the keyword
#      is matched as the var-name suffix immediately before `=`, and the full var
#      name is kept. Keyword set: PASSWORD, PASSWD, PW, MDP, MOT_DE_PASSE /
#      MOTDEPASSE, PASSPHRASE, SECRET, TOKEN, API_KEY/APIKEY, ACCESS_KEY/
#      ACCESSKEY, SECRET_KEY/SECRETKEY, AUTH_TOKEN/AUTHTOKEN.
#      Guards: the keyword must sit immediately before `=`, so `PWD=` (PW then D,
#      not `=`), `secret_sauce=`, `BYPASS=` etc. never match; `PASS` is excluded
#      entirely (BYPASS / COMPASS / SURPASS); `PWD` (cwd var) stays excluded.
#   3. Teradata `IDENTIFIED BY 'pw'` / `IDENTIFIED BY pw` (quoted or unquoted):
#      redact the value, keep the keyword.
#   4. Basic-auth in URLs / connection strings `scheme://user:pw@host`: redact
#      only the password between `:` and `@`, keep `user` + `@host`. Anchored on
#      `://…:…@` so `git@host`, `host:5432`, and emails are NOT matched.
#   5. `Bearer <token>` (incl. inside an `Authorization:` header): redact token.
#
# Deliberately NOT redacted — these only ever egress to the internal LLMAAS, so a
# gap reduces rather than increases risk, and each would over-match real code:
#   * bare space-separated `password <value>` (.netrc-style) — collides with
#     prose/comments that merely mention a password;
#   * bare `-w` / `-p <value>` CLI flags — collide with many legitimate flags;
#   * generic high-entropy token detection — needs entropy heuristics, out of
#     scope for this conservative anchored-pattern redactor.
#
# Credential variable-name keywords, matched as the suffix of <prefix><KEYWORD>=.
# Longer / compound names first; `_?` allows the underscore-free spelling. `PASS`
# is intentionally absent (too ambiguous: BYPASS / COMPASS / SURPASS).
_CRED_VAR_KEYWORDS = (
    r"MOT_DE_PASSE|MOTDEPASSE|PASSPHRASE|PASSWORD|PASSWD"
    r"|SECRET_?KEY|ACCESS_?KEY|AUTH_?TOKEN|API_?KEY|SECRET|TOKEN|MDP|PW"
)
_REDACTORS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^([ \t]*(?:--+[ \t]*|#[ \t]*)?\.(?:LOGON|CONNECT))\b.*$", re.IGNORECASE | re.MULTILINE),
     r"\1 <redacted>"),
    # Consolidated credential-variable assignment: <optional prefix><keyword>=value.
    # `\w*` allows the prefix; the keyword must land immediately before `=`.
    (re.compile(r"\b(\w*(?:" + _CRED_VAR_KEYWORDS + r"))=\S+", re.IGNORECASE),
     r"\1=<redacted>"),
    (re.compile(r"\b(IDENTIFIED\s+BY\s+)(?:'[^']*'|\"[^\"]*\"|\S+)", re.IGNORECASE),
     r"\1<redacted>"),
    (re.compile(r"(://[^/:@\s]+:)[^@/\s]+(@)"),
     r"\1<redacted>\2"),
    (re.compile(r"\b(BEARER)\s+[^\s\"']+", re.IGNORECASE),  # stop at a closing quote
     r"\1 <redacted>"),
]


def sanitize_for_enrichment(text: str) -> str:
    """Redact credential tokens from file content before it is sent to the LLM.

    Preserves line structure and every non-credential character; only the secret
    token itself is replaced with ``<redacted>``. Pure stdlib, no logging.
    """
    for pattern, repl in _REDACTORS:
        text = pattern.sub(repl, text)
    return text


# --- compact distillation for large SQL-heavy shell orchestrators -------------
# A bteq shell wrapper runs to thousands of lines of repeated boilerplate
# (.Logon, .Set, $NOTIFY_COMMAND, .If ErrorLevel goto, .Label). Sent raw, the
# LLM only ever sees the truncated head and misses most steps. Instead we send a
# COMPACT, COMPLETE distillation: the per-step `-- Statement N: <intent>` lines
# (business intent) + the data operations already extracted into fp.sql (tables
# read/written/purged + key columns). It covers every step, fits the budget, and
# carries only names + fixed vocabulary — never raw SQL. It still passes through
# sanitize_for_enrichment before the LLM (a Statement-N line could carry a secret).
_STMT_INTENT_RE = re.compile(r"^\s*--\s*Statement\s+\d+\s*:\s*(.*?)\s*:?\s*$", re.IGNORECASE)
_MAX_DISTILL_STEPS = 250    # per-step intent lines (one short line each)
_MAX_DISTILL_TABLES = 150   # tables in the operations section
_MAX_DISTILL_COLS = 12      # key columns named per table
_MAX_DISTILL_LINEAGE = 60   # target<-sources lineage lines
_MAX_INTENT_CHARS = 200     # cap each intent line so one runaway line can't blow the budget
_MAX_DISTILL_OPS = 40       # cap the op sequence rendered per table (consecutive dups already collapsed)


def _embedded_sql(fp: FileParse):
    """The file's SqlResult when it carries extracted SQL (table ops / objects /
    refs), else None. True for .sql files and for shell wrappers around bteq."""
    sql = getattr(fp, "sql", None)
    if sql is None:
        return None
    if (getattr(sql, "table_ops", None) or getattr(sql, "objects", None)
            or getattr(sql, "table_refs", None)):
        return sql
    return None


def _statement_intents(text: str, cap: int) -> list[str]:
    """Per-step intents documented as `-- Statement N: <intent>` lines, in script
    order, capped. The trailing ':' on the header line (if any) is dropped."""
    out: list[str] = []
    for line in text.splitlines():
        m = _STMT_INTENT_RE.match(line)
        if m:
            desc = m.group(1).strip()
            if desc:
                # Redact BEFORE slicing: the length cap must never split a
                # credential. The basic-auth URL rule (://user:pass@host) is
                # anchored on the trailing '@', so a tail-cut that drops '@host'
                # would otherwise strand user:pass past the later global redactor.
                out.append(sanitize_for_enrichment(desc).strip()[:_MAX_INTENT_CHARS])
                if len(out) >= cap:
                    break
    return out


def _distill_sql_shell(fp: FileParse, text: str, sql) -> str:
    """Compact, COMPLETE distillation of a SQL-heavy shell orchestrator: per-step
    intents + the data operations from the embedded SQL. Replaces the raw script
    (mostly boilerplate) so the summary reflects the whole pipeline, not its head."""
    lines: list[str] = [
        "This is a Teradata SQL / BTEQ ETL orchestration script. The following is "
        "a structured distillation of the whole script (not its raw text):",
        "",
    ]
    intents = _statement_intents(text, _MAX_DISTILL_STEPS)
    if intents:
        lines.append("Per-step intent (in script order):")
        lines.extend(f"{i}. {d}" for i, d in enumerate(intents, 1))
        lines.append("")
    ops = getattr(sql, "table_ops", None) or {}
    if ops:
        used = getattr(sql, "used_columns", None) or {}
        lines.append("Data operations (table: ordered operations | key columns):")
        for tbl in sorted(ops)[:_MAX_DISTILL_TABLES]:
            seq_ops = ops[tbl]
            seq = " -> ".join(seq_ops[:_MAX_DISTILL_OPS]) + (" -> ..." if len(seq_ops) > _MAX_DISTILL_OPS else "")
            cols = sorted(used.get(tbl, set()))[:_MAX_DISTILL_COLS]
            colpart = (" | columns: " + ", ".join(cols)) if cols else ""
            lines.append(f"- {tbl}: {seq}{colpart}")
        if len(ops) > _MAX_DISTILL_TABLES:
            lines.append(f"- ... (+{len(ops) - _MAX_DISTILL_TABLES} more tables)")
        lines.append("")
    lineage = getattr(sql, "lineage", None) or []
    if lineage:
        lines.append("Table lineage (target <- sources):")
        for tgt, srcs in lineage[:_MAX_DISTILL_LINEAGE]:
            lines.append(f"- {tgt} <- {', '.join(srcs[:6])}")
        lines.append("")
    objects = getattr(sql, "objects", None) or []
    if objects:
        lines.append("Defined objects: "
                     + ", ".join(f"{o.kind} {o.name}" for o in objects[:40]))
    return "\n".join(lines).strip()


def _enrich_one(llm: LLMClient, fp: FileParse, abs_path: str, settings: Settings) -> FileEnrichment:
    try:
        raw = Path(abs_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return _fallback_enrichment(fp)
    # SQL-heavy shell orchestrator -> summarize a compact, complete distillation
    # (intents + extracted operations) instead of the raw, truncated script. A
    # pure .sql file keeps its raw content (the file IS the data logic).
    sql = _embedded_sql(fp)
    distilled = sql is not None and fp.language != "sql"
    content = _distill_sql_shell(fp, raw, sql) if distilled else raw
    # Defense-in-depth: redact credentials BEFORE the content reaches the LLM —
    # covers both the raw text and the distilled payload.
    content = sanitize_for_enrichment(content)
    truncated = len(content) > settings.llm_char_limit
    if truncated:
        content = content[: settings.llm_char_limit] + "\n... [truncated]"
    try:
        data = llm.chat_json(_build_messages(fp, content, settings))
        result = _parse_response(data, fp)
    except Exception as exc:  # noqa: BLE001
        log.debug("enrichment failed for %s: %s", fp.rel_path, exc)
        result = _fallback_enrichment(fp)
    result.distilled = distilled
    result.truncated = truncated
    return result


def enrich_files(
    llm: LLMClient,
    parses: list[FileParse],
    abs_by_rel: dict[str, str],
    settings: Settings,
    progress_cb: Callable[[int, int, str], None] | None = None,
) -> dict[str, FileEnrichment]:
    """Enrich all files with bounded concurrency. Returns rel_path -> enrichment."""
    results: dict[str, FileEnrichment] = {}
    total = len(parses)
    done = 0
    workers = max(1, min(settings.concurrency, 5))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_enrich_one, llm, fp, abs_by_rel.get(fp.rel_path, fp.rel_path), settings): fp
            for fp in parses
        }
        for fut in as_completed(futures):
            fp = futures[fut]
            try:
                results[fp.rel_path] = fut.result()
            except Exception:  # noqa: BLE001
                results[fp.rel_path] = _fallback_enrichment(fp)
            done += 1
            if progress_cb:
                progress_cb(done, total, fp.rel_path)
    return results


def summarize_project(
    llm: LLMClient,
    project_name: str,
    file_summaries: list[tuple[str, str]],
    languages: list[str],
    frameworks: list[str],
    settings: Settings,
) -> str:
    fallback = (
        f"{project_name}: a {', '.join(languages) or 'multi-language'} project"
        + (f" using {', '.join(frameworks)}" if frameworks else "")
        + f" with {len(file_summaries)} analysed files."
    )
    sample = "\n".join(f"- {rel}: {summ}" for rel, summ in file_summaries[:40])
    language_name = LANGUAGE_NAMES.get(settings.language, "English")
    lang_clause = f"Answer in {language_name}."
    messages = [
        {"role": "system", "content": "You summarise software projects in one or two sentences. Reply with plain text only. " + lang_clause},
        {"role": "user", "content": f"Project name: {project_name}\nLanguages: {', '.join(languages)}\nFrameworks: {', '.join(frameworks) or 'unknown'}\n\nFile summaries:\n{sample}\n\nWrite a 1-2 sentence description of what this project does."},
    ]
    try:
        text = llm.chat(messages, max_tokens=160).strip()
        return text or fallback
    except Exception as exc:  # noqa: BLE001
        log.debug("project summary failed: %s", exc)
        return fallback
