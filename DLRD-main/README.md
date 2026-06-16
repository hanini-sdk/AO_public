# Data-Lineage and Retro-Documentation (local)

Turn any project into an interactive **knowledge graph** you can explore in your
browser — fully **local**, with a single, auditable network egress to your
internal LLM service (LLMAAS, OpenAI-compatible).

- **Deterministic structural analysis** (tree-sitter): files, functions, classes,
  imports, calls — the analysed code is **parsed as text, never executed**.
- **LLM enrichment** (your LLMAAS): one batched request per file produces a
  natural-language summary, an architectural layer (API / Service / Data / UI /
  Utility / Other), a complexity rating and tags.
- **Real dashboard**: the interactive dashboard from the open-source
  [Understand-Anything](https://github.com/Lum1104/Understand-Anything) project,
  adapted to be **100% offline** and compiled to static files in `web/dashboard/`.

> **Absolute security rule:** the application never opens an outbound connection
> to anything other than the configured `apiBase`. See
> [Single network egress](#single-network-egress--how-to-audit-it).

---

## How it’s meant to be used

You build/prepare this on a **personal machine** (internet access, public npm),
push it to your **personal git**, and pull it onto your **work machine**, where it
runs against internal code and the internal LLMAAS. On the work machine you only:

1. install the Python dependencies (from your internal mirror),
2. enter `apiBase` + `apiKey` + `model` in the browser,
3. analyse a local folder or an uploaded `.zip`, and explore the dashboard.

**No Node, no npm on the work machine** — it serves the already-compiled
`web/dashboard/` static files and runs Python only.

---

## Install & run on the work machine (Python only)

Prerequisites: **Python 3.11+** recommended (works on 3.10). No internet beyond
your LLMAAS host.

```powershell
# 1) create a virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell
# source .venv/bin/activate          # macOS / Linux

# 2) install pinned deps FROM YOUR INTERNAL MIRROR (not public PyPI)
pip install --index-url https://<your-internal-pypi-mirror>/simple -r requirements.txt

# 3) launch — opens http://127.0.0.1:8765/ in your browser
python -m app
```

Then in the browser:

1. **Connection** — enter `API base URL` (keep any `/v1`), `API key`, `model`,
   pick the output language (English or French), and (for gemma-like models) untick *“supports system
   messages”*. If your LLMAAS uses an internal/private CA, set **CA certificate path**
   to your PEM bundle (e.g. `cacert.pem`) so its TLS certificate verifies; leave it
   blank for public endpoints. TLS verification is always on (there is no way to
   disable it). Click **Test connection**, then **Save**.
2. **Analyse** — choose a **local folder path** or **upload a `.zip`**, optionally
   name the project, click **Start analysis**. Watch progress; on completion you
   are redirected to the dashboard.

Configuration is stored only in the gitignored `config.local.json` (or via
environment variables / `.env` — see `.env.example`). The API key is never logged
and never written into the generated graph.

---

## Try it offline (no real LLMAAS) — demo

A tiny local stand-in LLMAAS is included for demos. It runs on `127.0.0.1`, so it
is consistent with the egress guard.

```powershell
# terminal 1 — the mock endpoint
python tools/mock_llmaas.py --port 8900

# terminal 2 — the app
python -m app
```

In the browser set **API base URL** = `http://127.0.0.1:8900/v1`, **API key** =
anything, **model** = `mock-model`, then analyse the bundled `sample-project/`
folder. You’ll get a real graph rendered in the real dashboard.

---

## Single network egress — how to audit it

**Everything funnels through one module.** The OpenAI SDK is built with a custom
`httpx` client whose transport inspects **every** request (including redirects)
and refuses any host / port / scheme that is not exactly your `apiBase`.

- The only file that performs outbound networking: **`app/http_guard.py`**
  (`EgressGuard` + `GuardedTransport` + `build_guarded_client`).
- The only consumer: **`app/llm.py`** (`OpenAI(..., http_client=<guarded>)`).

Audit commands (all should return *nothing* surprising):

```powershell
# Only http_guard.py should appear as importing a network client:
grep -rnE "import (httpx|requests|aiohttp|socket)|from (httpx|requests|aiohttp)|urlopen" app

# The OpenAI client must be constructed only in llm.py, with http_client=guarded:
grep -rn "OpenAI(" app ; grep -rn "build_guarded" app

# The compiled dashboard makes no external calls — every http(s):// literal is an
# inert namespace / error-link / license string (w3.org, eclipse.org elk, json-schema,
# reactflow.dev, react.dev, opensource.org, tailwindcss.com):
grep -rhoE "https?://[a-zA-Z0-9._/-]+" web/dashboard --include=*.js --include=*.css --include=*.html | sort -u
```

Runtime backstops, in addition to the guard:

- The server **binds to `127.0.0.1` only** (hard-coded in `app/__main__.py`).
- A strict **Content-Security-Policy** (`default-src 'self'; connect-src 'self'; …`)
  prevents the dashboard page from reaching any external host.
- **Recommended firewall rule:** allow outbound only to your LLMAAS host; deny the
  rest. The guard already enforces this in-process; the firewall is defence-in-depth.

The analysed project’s source is parsed as text with tree-sitter and is **never
imported, executed, or `eval`’d**. Uploaded archives are extracted locally with
zip-slip protection and cleaned up on the next upload.

---

## npm is never used on the work machine

- npm/Node is used **only on the personal build machine**, occasionally, from the
  **public** npm registry (pinned in `tools/dashboard/.npmrc`).
- No project configuration points at any internal npm registry.
- The work machine installs **Python only** and serves `web/dashboard/` as-is.

### Rebuilding the dashboard (personal machine only — optional)

The compiled dashboard is already committed in `web/dashboard/`. To rebuild it:

```bash
cd tools/dashboard
npm install        # PUBLIC registry, forced via .npmrc
npm run build      # → ../../web/dashboard/
```

Adaptations applied to the upstream dashboard (see `LICENSE` attribution):
removed the Google Fonts `<link>` (fonts self-hosted via `@fontsource`), removed
the shared-link token gate (the loopback bind is the boundary here), and vendored
the three pure-TS core modules (`schema`, `search`, `types`) under
`tools/dashboard/vendor/core/` so the heavy/native `core` package is never needed.

---

## Project structure

```
app/                 backend (FastAPI)
  __main__.py        launcher — uvicorn on 127.0.0.1, opens the browser
  server.py          routes, static serving, data endpoints, analysis API, CSP
  config.py          settings load/save (gitignored; key never logged/committed)
  http_guard.py      THE single network egress point (EgressGuard + guarded httpx)
  llm.py             guarded OpenAI client (system-msg fold, retries, JSON repair)
  scanner.py         deterministic file walk + language detection + ignore rules
  parser.py          tree-sitter structural parsing (functions/classes/imports/calls)
  enrich.py          batched LLM enrichment (summary + layer + complexity + tags)
  graph.py           assembles knowledge-graph.json in the dashboard's schema
  validate.py        referential integrity + de-duplication
  pipeline.py        orchestration + thread-safe progress
web/                 config + analysis shell (index.html, app.js, styles.css)
web/dashboard/       the real dashboard, compiled to static — COMMITTED, shipped
tools/dashboard/     dashboard build source (personal machine; node_modules ignored)
tools/mock_llmaas.py local OpenAI-compatible stub for offline demos
sample-project/      small multi-layer project for the demo
requirements.txt  .env.example  .gitignore  LICENSE  README.md
```

The generated `knowledge-graph.json` (and a small `config.json`) are written to
`data/` at use time and are **not** committed.

## knowledge-graph.json format

The backend emits exactly the schema the real dashboard validates (derived from
its Zod schema). Layers are a **top-level array** grouping node ids — not a
per-node field:

```jsonc
{
  "version": "1.0.0",
  "project": { "name", "languages":[], "frameworks":[], "description",
               "analyzedAt": "ISO-8601", "gitCommitHash" },
  "nodes": [{ "id", "type": "file|function|class|…", "name", "summary",
              "tags":[], "complexity": "simple|moderate|complex",
              "filePath"?, "lineRange"?: [start,end] }],
  "edges": [{ "source", "target", "type": "imports|contains|calls|…",
              "direction": "forward|backward|bidirectional", "weight": 0..1 }],
  "layers": [{ "id": "layer:api", "name", "description", "nodeIds": [] }],
  "tour": []
}
```

No `apiBase` / `apiKey` ever appears in this file.

## Roadmap (v2 — not in this version)

Semantic search via embeddings (through the LLMAAS), guided tours, business-domain
view, diff/impact analysis, and incremental updates. The dashboard already
contains the UI for several of these; the backend would generate the extra data
(`domain-graph.json`, `diff-overlay.json`, `tour`) through the same single egress.

## License & attribution

MIT — see [`LICENSE`](LICENSE). The dashboard is adapted from the MIT-licensed
[Understand-Anything](https://github.com/Lum1104/Understand-Anything) by
Yuxiang Lin; the original copyright and license are retained.
