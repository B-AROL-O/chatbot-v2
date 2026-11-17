# AGENTS.md — SME Chatbot

## Project Overview

SME Chatbot is a Python-based chatbot that answers questions over a company's own data. It currently ships three data domains, built for a packaging machinery manufacturer:

1. **CDO records** — structured orders, machines, customers, and container/cap samples
2. **Machine Manuals** — PDF documentation for the manufacturer's machines
3. **Documents** — the manufacturer's DocumentService library (manuals, CDO reports, certifications), synced via a separate ingestion pipeline

A FastAPI backend (`src/sme_chatbot/`) owns all RAG/chat logic behind one streaming chat endpoint; a Gradio app (`frontends/gradio_app.py`) is a thin HTTP/SSE client of that backend — the same contract a future .NET portal proxy will consume. LLM-driven tool calling (SQL + semantic search) picks the domain's toolset server-side.

See `docs/specs/2026-07-02-backend-split-and-docsvc-ingestion-plan.md` for the implementation plan behind this split (implemented).

## Critical Context

### CDO = "Conferme d'Ordine"
**"CDO" stands for "Conferme d'Ordine"** — an Italian term meaning **"Order Confirmation"**. It refers to the manufacturer's internal order documents. This is the domain's vocabulary, not a generic tech acronym. When reading or writing code/docs:
- `cdo_esportata` = the exported order confirmation number (PK)
- `cdo_prodotto_principale` = the main product on an order
- The CDO data model links orders → machines → containers → caps

### Backend / Frontend Split

```text
src/sme_chatbot/
├── config.py               ← env-driven settings (single source of truth, see .env.example)
├── api/
│   ├── main.py             ← FastAPI app: GET /health, GET /config, POST /chat/message (SSE)
│   ├── auth.py             ← get_identity(): AUTH_MODE=dev|jwt → RequestIdentity
│   ├── schemas.py          ← request/response models
│   └── streaming.py        ← ChatEvent → SSE frames
├── core/
│   ├── chat.py             ← stream_chat() — framework-agnostic streaming chat loop (ChatEvent generator)
│   ├── domains.py          ← DomainConfig registry: "sample_cdo" | "sample_manuals" | "documents"
│   └── events.py           ← ChatEvent dataclasses (Token/ToolCall/ToolResult/Done/Error)
├── tools/                  ← query_cdo_db, search_docs, search_manuals, search_documents, read_pages
│                              (+ access_filter.py, lexical_index.py)
├── clients/docsvc.py       ← DocumentService REST client (MSAL, paging, download)
├── ingestion/ingest_docsvc.py  ← DocumentService → ChromaDB ingestion pipeline (CLI)
└── eval/                   ← eval runner CLIs (sme-chatbot-eval, sme-chatbot-eval-retrieval)

frontends/gradio_app.py     ← thin HTTP/SSE client of the FastAPI backend (no RAG logic)
```

### Architecture at a Glance

The README's [Architecture](README.md#architecture) section has the same two
views as Mermaid diagrams; keep both in sync when the architecture changes.

```text
Offline — building the data stores
  data/input/CDO/*.xlsx (3 spreadsheets)
      → db/init_db.py + db/etl.py               → data/cdo.db (SQLite, 15+ tables)
  data/input/CDO/ documents
      → ingest_sample_cdo.py      (legacy)      → data/chroma_cdo_db      (collection cdo_docs)
  data/input/manuals/
      → ingest_sample_manuals.py  (legacy)      → data/chroma_manuals_db  (collection manuals)
  DocumentService REST API  (Entra ID token via MSAL, presigned downloads)
      → sme-ingest-docsvc (ingestion/ + clients/docsvc.py)
                                                → data/chroma_docsvc_db   (collection docsvc)
      → sme-ingest-docsvc --build-lexical-index → data/chroma_docsvc_db/lexical_fts.sqlite3 (FTS5)

Runtime — answering a chat message
  frontends/gradio_app.py / sme-chatbot-eval   (HTTP/SSE clients only; same contract
                                                the future .NET portal proxy will consume)
      → api/main.py  GET /health (open) · GET /config · POST /chat/message (SSE)
          → api/auth.py  AUTH_MODE=dev | jwt (Entra ID JWKS) → RequestIdentity
          → core/chat.py: stream_chat()  ⇄  LLM endpoint (OpenAI-compatible, LLM_BASE_URL)
                                            [traced to Langfuse when LANGFUSE_ENABLED]
              dispatches the tools of the request's domain:
              sample_cdo     ├── tools/cdo_sql.py     query_cdo_db (read-only SQL) → data/cdo.db
                             ├── tools/cdo_rag.py     search_docs                 → cdo_docs
              sample_manuals ├── tools/manuals_rag.py search_manuals              → manuals
              documents      └── tools/docsvc_rag.py  search_documents, read_pages → docsvc
                                   ├── access_filter.py  identity + matricola → Chroma `where`
                                   └── lexical_index.py  BM25 candidates, re-fetched from
                                                         Chroma under the same `where`
```

## Key Files

| Path | Purpose |
| ------ | --------- |
| `src/sme_chatbot/api/main.py` | FastAPI app factory: `GET /health`, `GET /config`, `POST /chat/message` (SSE). Entry point `sme-chatbot-api` runs it on `:8000`. |
| `src/sme_chatbot/api/auth.py` | `get_identity` dependency: `AUTH_MODE=dev` (optional `X-User-Id` header) or `jwt` (Entra ID bearer token, JWKS fetched lazily). Tools only ever see the resulting `RequestIdentity` |
| `src/sme_chatbot/core/chat.py` | Framework-agnostic streaming chat loop (`stream_chat`), ported from the old `make_chat_fn`. **Do not change the loop logic without understanding tool call accumulation.** |
| `src/sme_chatbot/core/llm_api.py` | `LLM_API` switch: `chat_completions` (default) or `responses`. Adapts the `/v1/responses` stream into chat-completions-shaped chunks so `stream_chat`'s loop is the same on both |
| `src/sme_chatbot/core/domains.py` | `DomainConfig` registry — system prompt + tool definitions per domain (`sample_cdo`, `sample_manuals`, `documents`) |
| `src/sme_chatbot/config.py` | Single source of truth for env vars — mirrors `.env.example` |
| `frontends/gradio_app.py` | Thin Gradio HTTP/SSE client of the backend — no imports from `sme_chatbot`, HTTP only |
| `src/sme_chatbot/tools/cdo_sql.py` | Read-only SQL executor: sanitizes input (only `SELECT`/`WITH`, blocks writes), opens SQLite in read-only mode, caps at 200 rows |
| `src/sme_chatbot/tools/cdo_rag.py` | Chroma semantic search: lazy-loaded embedding model + persistent collection |
| `src/sme_chatbot/tools/manuals_rag.py` | Manual-specific RAG tool (separate Chroma collection) |
| `src/sme_chatbot/tools/docsvc_rag.py` | `search_documents` (hybrid dense + BM25, RRF-fused) and `read_pages` over the DocumentService collection, with `access_filter.py`-built identity/scope filters; `resolve_serial` scopes a conversation to the matricola it names |
| `src/sme_chatbot/tools/lexical_index.py` | SQLite FTS5 keyword index over the same chunk ids (`sme-ingest-docsvc --build-lexical-index`). A recall aid only: hits are re-fetched from Chroma under the caller's `where`, which stays the access gate |
| `src/sme_chatbot/clients/docsvc.py` | DocumentService REST client (MSAL client-credentials, paging, presigned download) |
| `src/sme_chatbot/ingestion/ingest_docsvc.py` | DocumentService → ChromaDB ingestion pipeline; CLI entry point `sme-ingest-docsvc` |
| `src/sme_chatbot/eval/` | Automated eval runner over `$EVAL_DATASETS_DIR/*-domande.csv`; CLI entry point `sme-chatbot-eval`. Pure scorers in `scoring.py` (no HTTP/Langfuse coupling, so a Langfuse harness can import them); run records land in `$EVAL_RESULTS_DIR`. Both default to a checkout of the private eval-data repo at `data/eval/` — the golden sets and run records are deployment data and are not in this repo. See `docs/procedures/running-evals.md`. `sme-chatbot-eval-retrieval` (`eval/retrieval.py`) ranks the gold page per question with no LLM, for retrieval changes. `sme-chatbot-eval-trend` (`eval/trend.py`) charts verdict shares and latency across all committed runs as one HTML page |
| `alloy/client.alloy` | Grafana Alloy config for the opt-in `alloy` compose service (profile `observability`): ships this stack's container logs to the shared Loki at `LOKI_URL` under `job="chatbot"`. See `docs/procedures/docker-deployment.md` "Shipping logs to Loki" |
| `db/schema.sql` | Normalized schema: `dim_*` lookup tables → `machine`, `"order"`, `container_sample`, `cap_sample` |
| `db/etl.py` | ETL pipeline (5 phases): Excel headers → dim tables → core tables, with `_get_or_insert` for dedup |
| `db/init_db.py` | One-shot DB initialization from schema.sql |
| `ingest_sample_cdo.py` | PDF/doc ingestion: chunking → `paraphrase-multilingual-MiniLM-L12-v2` embeddings → Chroma (untouched local-dir pipeline) |
| `ingest_sample_manuals.py` | Same, for the machine-manuals collection (untouched local-dir pipeline) |

## Data Model Highlights

### Three join keys across all tables
- **`matricola`** — stable machine identity (PK on `machine`, FK on orders, containers, caps)
- **`cdo_prodotto_principale`** — main product per order line
- **`codice_campione`** — container/cap sample code (PK on sample tables)

### SQL reserved word
The table `order` is a reserved keyword in SQLite — it is always quoted as `"order"` in ETL and tool code. **Do not remove the quotes.**

### Key dimensions
- `dim_machine_type`: Tipo codes (AL, EL, M, MB, SC, TS, TSI) → Italian labels
- `dim_model`: machine models
- `dim_product_category`: Product types (with cancelled flag)
- `dim_sector`: Mercantile sectors (M:M via `order_sector`)
- `dim_customer`, `dim_end_user`, `dim_buyer_ref`: Customer entities
- `dim_country` + continent

### Sample tables
- `container_sample`: Bottle/closure geometry specs (diameters, heights, widths)
- `cap_sample`: Cap/closure specs (heights, diameters, teeth count, neck finish)

## Environment Setup

### Required env vars

Copy `.env.example` to `.env` and fill in real values — it documents every
variable, grouped by component (LLM, backend auth, Chroma paths, Gradio
frontend, DocumentService ingestion, eval, logging, log shipping). At minimum, set
`LLM_BASE_URL`, `LLM_API_TOKEN`, `LLM_MODEL` for the backend to start.

### Startup (Docker Compose)

```bash
# Backend API (FastAPI) — http://localhost:8000
docker compose up chatbot-api

# Gradio frontend (thin HTTP/SSE client of the backend) — http://localhost:7860
docker compose up chatbot-gradio

# Both
docker compose up
```

### Startup (local, with uv)

```bash
uv sync
uv run python -m db.init_db          # create DB schema
uv run python -m db.etl              # load data (requires Excel files in data/input/CDO/)
uv run python ingest_sample_cdo.py       # populate the CDO vector store
uv run python ingest_sample_manuals.py   # populate the manuals vector store
uv run sme-chatbot-api              # FastAPI backend on :8000
uv run python frontends/gradio_app.py    # Gradio UI on :7860 (separate terminal)
uv run sme-ingest-docsvc --limit 5  # optional: populate the "documents" domain from DocumentService
```

### Pre-seed DB with sample data

```bash
# There are shell scripts to import pre-downloaded samples:
bash scripts/import_sample_cdo.sh
bash scripts/import_sample_manuals.sh
```

## Testing

```bash
uv run pytest tests/unit tests/backend_api -v   # new stack: no network/LLM/creds required
uv run pytest tests/ -v                         # everything, including the read-only API suites
```

`tests/unit/` and `tests/backend_api/` cover the `src/sme_chatbot` core loop, access filter, DocSvc client, ingestion, and the FastAPI endpoint (fake LLM, no network). They also cover the
eval runner (`test_eval_dataset.py`, `test_eval_scoring.py`, `test_eval_report.py`,
`test_eval_runner.py`) — the scoring suite is the load-bearing one: it pins the
chapter-reference/matricola filters that keep `anchor_recall` from scoring
answers against their own cross-references. The eval tests run on the synthetic set in
`tests/fixtures/eval/`; `test_eval_golden_set.py` pins the real golden set and runs only
when the eval repo is checked out at `data/eval/` (skipped otherwise).

## Linting

The Super-Linter workflow lints the **whole codebase** on every PR and push to
the default branch (the workflow triggers on `main` and `dev-v2`, so either
name works), not just the files a PR touched — so a red Lint job is about the repo's real state, never an accident
of which files you happened to open. Keep it green.

Python lint is pinned by the repo-root `.ruff.toml`, which CI reads too
(`PYTHON_RUFF_CONFIG_FILE` in `.github/workflows/super-linter.yml`). Reproduce
the gate exactly, with no Docker:

```bash
uvx ruff check .          # same config CI uses; add --fix for the autofixable ones
```

That file also records *why* each rule group is on and why `BLE001` is
deliberately off. Change the rule set there, not per-file with `# noqa` —
`RUF100` is enabled, so a `noqa` that suppresses nothing is itself an error.

To reproduce the non-Python linters (markdown, shellcheck, JSON, hadolint,
actionlint, gitleaks), run the pinned image over a clean checkout:

```bash
git archive HEAD | tar -x -C /tmp/lintall && cd /tmp/lintall && git init -q -b main . \
  && git add -A && git commit -qm base
docker run --rm -e RUN_LOCAL=true -e VALIDATE_ALL_CODEBASE=true -e DEFAULT_BRANCH=main \
  -e PYTHON_RUFF_CONFIG_FILE=../../.ruff.toml \
  -e VALIDATE_GITLEAKS=true -e VALIDATE_PYTHON_RUFF=true -e VALIDATE_YAML=true \
  -e VALIDATE_JSON=true -e VALIDATE_MARKDOWN=true -e VALIDATE_BASH=true \
  -e VALIDATE_BASH_EXEC=true -e VALIDATE_DOCKERFILE_HADOLINT=true \
  -e VALIDATE_GITHUB_ACTIONS=true -e VALIDATE_ENV=true \
  -v /tmp/lintall:/tmp/lint ghcr.io/super-linter/super-linter:v8.7.0
```

## Current State & Todo

See `TODO.md`:
- [x] Backend/frontend split + DocumentService ingestion — see `docs/specs/2026-07-02-backend-split-and-docsvc-ingestion-plan.md` (implemented)

## Coding Patterns to Follow

1. **Lazy loading** — Embedding models and Chroma clients are loaded on first tool call, not at import time (`_ensure_loaded()` pattern in `cdo_rag.py`). The FastAPI app follows the same rule: the OpenAI client in `api/main.py` is a lazily-constructed module-level singleton, never built at import time. Keep this for fast imports.
   The opt-in startup warm-up (`RETRIEVAL_WARMUP`, `tools/warmup.py`) runs after import,
   in a background thread, and goes through the same locked `_ensure_loaded()`.

2. **Read-only SQL guard** — The SQL tool uses two layers of defense: regex sanitization (`_is_safe()`) and SQLite URI mode `file:db?mode=ro`. Never bypass or weaken either layer.

3. **Streaming tool dispatch** — `core/chat.py`'s `stream_chat()` accumulates streaming tool-call fragments across chunks (not just final tokens), yielding `ChatEvent`s (`TokenEvent`, `ToolCallEvent`, `ToolResultEvent`, `DoneEvent`, `ErrorEvent`) rather than cumulative strings. Tool arguments arrive in pieces. Do not revert to the old two-phase design.

4. **Thinking block stripping** — Qwen3 chain-of-thought is stripped before yielding tokens; because the loop now yields deltas (not the cumulative string), stripping buffers inside an open `<think>` block until it closes rather than regexing the whole string at once.

5. **Italian terminology** — Column names, type codes, and domain vocabulary are Italian. `matricola`, `cdo_esportata`, `codice_campione` etc. are proper nouns in the domain. Do not translate them or rename columns without updating ALL ETL mappings.

6. **Data path convention** — All data lives under `data/`:
   - `data/cdo.db` — SQLite database
   - `data/chroma_cdo_db/` — Chroma vector store (CDO domain)
   - `data/chroma_manuals_db/` — Chroma vector store (manuals domain)
   - `data/chroma_docsvc_db/` — Chroma vector store (documents domain, from DocumentService)
   - `data/input/CDO/` — Source Excel files for ETL
   - `data/eval/` — checkout of the private eval repo (golden sets in `datasets/`, run records in `results/`)

## Working Method

`docs/operating-manual.md` is the working method for this project: read the real request, decompose along verification lines, spend effort where the risk lives, re-derive instead of recognize, label guesses, attack the conclusion, verdict-first communication. Its most-used procedures are available as project slash commands in `.claude/commands/`:

| Command | When to run it | Manual section |
| --------- | ---------------- | ---------------- |
| `/risk-map` | At the start of a non-trivial task, before spreading effort | §3 |
| `/decompose` | When a problem is too big to verify in one piece | §2 |
| `/verify-claim` | Whenever a conclusion rests on one load-bearing claim | §4 |
| `/attack` | After reaching a conclusion, before delivering it | §6 |
| `/self-test` | Before sending any substantial answer | final section |

## Relevant Documents

`docs/README.md` is the index of all documentation, organized as:
`docs/specs/` (specs and contracts referenced from code — paths are stable,
do not move), `docs/adr/` (Architecture Decision Records),
`docs/procedures/` (up-to-date how-tos). Proposals and analyses live in
GitHub issues.

- `docs/specs/2026-07-02-backend-split-and-docsvc-ingestion-plan.md` — Backend/frontend split + DocSvc ingestion implementation plan (implemented; see WP1–WP9)
- `docs/procedures/docker-deployment.md` — Deployment details
- `docs/procedures/release-chatbot.md` — Release runbook: cutting a release, pulling the published image, GHCR auth. A release is triggered by *pushing* the `chatbotv2/v*` tag — a tag left local produces no Release and no image; verify with `git ls-remote --tags origin`
- `docs/procedures/running-evals.md` — run the golden eval question sets with `sme-chatbot-eval` and read the report
