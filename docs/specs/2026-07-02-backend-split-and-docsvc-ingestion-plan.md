# Codebase Preparation Plan — Backend/Frontend Split + DocumentService Ingestion

- **Document type:** Implementation plan (subagent-executable work packages)
- **Status:** Implemented (WP1–WP9 complete; kept in `docs/specs/` — not moved to `docs/historical/` — because §4 contracts are still cited by path from `src/`, `tests/`, and other docs)
- **Date:** 2026-07-02
- **Parent document:** `docs/historical/2026-06-24-chatbot_architecture_report.md` (Phases 1–2) — *removed from the repo with the rest of `docs/historical/`*
- **Since written:** the original company, host and collection names were
  replaced with neutral ones, and the two read-only API test suites it
  mentions were removed. The
  DocumentService credential setup it points to is now in
  `docs/procedures/docsvc-ingestion.md`.
- **Author intent:** Each Work Package (WP) below is written to be executed by an independent agent. Section 4 ("Shared contracts") is **frozen**: no WP may change it unilaterally. If a WP discovers a contract problem, it must stop and report rather than improvise.

---

## 1. Goals

Prepare the `sme-chatbot` PoC codebase for the architecture described in `docs/historical/2026-06-24-chatbot_architecture_report.md` by delivering two things:

1. **Backend/frontend split** — a FastAPI backend (`POST /chat/message` with SSE streaming, `GET /health`) that owns all RAG/chat logic, and a Gradio app rewritten as a **thin HTTP client** of that backend. Gradio dogfoods the exact contract the .NET portal proxy will consume later.
2. **DocumentService ingestion** — a new, self-contained ingestion script that pulls documents from the DocumentService REST API (`https://documents-api.example.com`), downloads content via presigned URLs, and indexes chunks into a new ChromaDB collection with `document_id` + business-key metadata (serial, customer, category, source) for query-time access filtering.

### Decisions already made (do not re-litigate)

| Decision | Choice |
| --- | --- |
| Split shape | FastAPI backend now; Gradio becomes an HTTP/SSE client of it |
| Repo layout | `src/` package layout (`src/sme_chatbot/…`) for all **new** code |
| DocSvc ingestion | **Separate third script** — `ingest_sample_cdo.py` and `ingest_sample_manuals.py` stay untouched; no shared-ingestion-library refactor in this phase |
| Chunk scoping metadata | `document_id`, `version_id`, `source`, `category` + flattened business keys (`serial`, `customer`, …). Query-time filter by permitted `document_id` list / business keys. **Not** `user_id`/`tenant_id` at ingestion time |
| Chat API shape | One endpoint, `domain` field (`"sample_cdo"` \| `"sample_manuals"` \| `"documents"`) selects prompt + toolset server-side |
| Backend auth | Pluggable: `AUTH_MODE=dev` (identity from optional `X-User-Id` header; default for local Gradio) and `AUTH_MODE=jwt` (Entra ID bearer validation; config only in this phase, full wiring later) |
| DocSvc ingestion scope | All accessible documents, no category filter by default; every chunk stamped with `category`; `--category`/`--limit`/`--dry-run` CLI flags for partial runs |

---

## 2. Current state (what an agent will find)

| Path | Role today | Fate |
| --- | --- | --- |
| `chatbot_core.py` | OpenAI client + streaming tool-dispatch loop (`make_chat_fn`) + Gradio launcher (`run`). Gradio import at module level | Logic ported into `src/sme_chatbot/core/`; file deleted in WP9 |
| `chatbot_cdo.py`, `chatbot_manuals.py` | Domain launchers (system prompt + tools → `chatbot_core.run`) | Prompts move to `src/sme_chatbot/core/domains.py`; files deleted in WP9 |
| `tools/__init__.py` | CDO tool registry (`TOOL_DEFINITIONS`, `dispatch`) for `query_cdo_db` + `search_docs` | Moved (git mv) into `src/sme_chatbot/tools/` in WP2 |
| `tools/cdo_rag.py`, `tools/cdo_sql.py`, `tools/manuals_rag.py` | RAG/SQL tools over `data/chroma_cdo_db` (`cdo_docs`), `data/chroma_manuals_db` (`manuals`), and the CDO SQLite DB | Moved with the package; internals unchanged except import paths |
| `ingest_sample_cdo.py`, `ingest_sample_manuals.py` | Local-directory ingestion into ChromaDB | **Untouched** (per decision) |
| `tests/documentservice_api/` | Read-only pytest suite against DocumentService dev; MSAL client-credentials fixtures | Untouched; **reference implementation** for auth + API contract |
| `pyproject.toml` | Flat project, no package config; deps include `gradio`, `openai`, `chromadb`, `sentence-transformers` | Extended in WP1 |
| `data/chroma_cdo_db`, `data/chroma_manuals_db` | Existing vector stores | Untouched; new store added at `data/chroma_docsvc_db` |

Known good news: the `make_chat_fn` streaming bug listed in the architecture report §12.2 is already fixed (see `TODO.md`, checked item; the current single-streaming-loop implementation in `chatbot_core.py` is the fixed version). Port it as-is.

---

## 3. Target layout

```text
src/sme_chatbot/
├── __init__.py
├── config.py                  # env-driven settings (single source of truth)
├── core/
│   ├── __init__.py
│   ├── chat.py                # framework-agnostic streaming chat loop (ChatEvent generator)
│   ├── events.py              # ChatEvent dataclasses
│   └── domains.py             # DomainConfig registry: "sample_cdo", "sample_manuals", "documents"
├── tools/                     # git mv of root tools/ + one new module
│   ├── __init__.py            # CDO registry (moved)
│   ├── cdo_rag.py             # moved
│   ├── cdo_sql.py             # moved
│   ├── manuals_rag.py         # moved
│   ├── docsvc_rag.py          # NEW — search over the DocSvc collection with metadata filters
│   └── access_filter.py       # NEW — identity/scope → Chroma where-filter builder
├── clients/
│   ├── __init__.py
│   └── docsvc.py              # NEW — DocumentService REST client (MSAL, paging, download)
├── api/
│   ├── __init__.py
│   ├── main.py                # FastAPI app factory; /health, /chat/message
│   ├── schemas.py             # Pydantic request/response models
│   ├── auth.py                # AUTH_MODE=dev|jwt → RequestIdentity dependency
│   └── streaming.py           # ChatEvent → SSE translation
└── ingestion/
    ├── __init__.py
    └── ingest_docsvc.py       # NEW — self-contained DocSvc → ChromaDB pipeline (CLI)

frontends/
└── gradio_app.py              # NEW — thin HTTP/SSE client of the FastAPI backend

tests/
├── unit/                      # NEW — core loop, access filter, docsvc client (mocked)
├── backend_api/               # NEW — FastAPI TestClient tests (fake LLM)
└── documentservice_api/       # existing, untouched

Dockerfile                     # NEW — backend container
.env.example                   # NEW — all env vars documented
```

Root scripts `ingest_sample_cdo.py` / `ingest_sample_manuals.py` remain at the root, untouched. Root `chatbot_*.py` are deleted only in WP9 (cleanup), after the new stack is verified.

---

## 4. Shared contracts (FROZEN — read before any WP)

### 4.1 Chat API

`GET /health` → `200 {"status": "ok", "domains": ["sample_cdo", "sample_manuals", "documents"], "version": "0.5.2"}`. No auth.
`version` was added additively later (existing clients that only read `status`/`domains` are unaffected); it is the release tag baked into the image at build time, `"dev"` for a local/manual build.

`POST /chat/message` — request (Pydantic, `api/schemas.py`):

```json
{
  "domain": "sample_manuals",
  "message": "Come si sostituisce la testata?",
  "history": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "stream": true
}
```

- `domain`: required, one of the registered domain keys; unknown → `422`.
- `history`: optional (default `[]`); roles restricted to `user`/`assistant`.
- `stream`: default `true`. With `stream=false` the endpoint returns a single JSON `ChatResponse` (same fields as the `done` event) — useful for tests and the future .NET proxy's first iteration.

SSE stream (`text/event-stream`, via `sse-starlette`), event types:

| `event:` | `data:` (JSON) | Notes |
| --- | --- | --- |
| `token` | `{"text": "<delta>"}` | Cumulative rendering is the client's job; server sends deltas, already `<think>`-stripped |
| `tool_call` | `{"name": "search_manuals", "arguments": {...}}` | Emitted when the model requests a tool |
| `tool_result` | `{"name": "...", "ok": true, "citations": [Citation, ...]}` | Result payload NOT streamed in full; citations extracted if the tool returned RAG results |
| `done` | `{"content": "<full answer>", "citations": [Citation, ...], "usage": {"rounds": 2}}` | Always the final event on success |
| `error` | `{"message": "..."}` | Terminal; sent instead of `done` |

`Citation` object: `{"source": "<filename>", "page": 3, "document_id": "<uuid|null>", "distance": 0.42, "filename": "<filename|null>", "repository": "<origin store|null>"}` — `document_id` is null for chunks from the two legacy collections.

`filename` and `repository` were added additively later; clients that only read `source` are unaffected. `source` is always the document's *name*: `search_documents` used to report the origin network share (`ACME-NETWORK-SHARE`) under that key, which is now `repository` and never a citation label.

### 4.2 RequestIdentity and auth modes

```python
@dataclass
class RequestIdentity:
    user_id: str | None          # dev: X-User-Id header or None; jwt: oid/sub claim
    permitted_document_ids: list[str] | None  # None = unrestricted (this phase's default)
    scopes: dict[str, str]       # optional business-key scope, e.g. {"serial": "91123"}
```

- `AUTH_MODE=dev` (default): no `Authorization` required. `user_id` from optional `X-User-Id` header. `permitted_document_ids=None`, `scopes={}`.
- `AUTH_MODE=jwt`: validate `Authorization: Bearer` against Entra ID (`python-jose`, JWKS from `https://login.microsoftonline.com/{tenant}/discovery/v2.0/keys`, audience from env). 401 on failure. Claim→scope mapping is a stub in this phase (returns unrestricted identity after validation) — the ACL mapping exercise (report §12.3) fills it in later.
- Handlers and tools never read headers/tokens; they receive `RequestIdentity` only.

### 4.3 DocSvc chunk metadata schema (ChromaDB collection `docsvc`)

- Store: `data/chroma_docsvc_db` (new `PersistentClient` path), collection `docsvc`.
- Embedding model: `paraphrase-multilingual-MiniLM-L12-v2` (same as existing pipelines — query vectors must match).
- Chunking: same sliding window as existing scripts (`CHUNK_SIZE=1000`, `CHUNK_OVERLAP=200` chars).
- Chunk ID: `"{document_id}::v{version_number}::p{page}::c{chunk_idx}"` — deterministic, idempotent re-runs.
- Metadata per chunk (ChromaDB accepts **scalars only** — str/int/float/bool; never nested dicts):

| Key | Type | Source |
| --- | --- | --- |
| `document_id` | str (UUID) | `documentId` |
| `version_id` | str (UUID) | `versionId` |
| `version_number` | int | `versionNumber` (fall back to 1 if absent) |
| `source` | str | `source` (e.g. `Sharepoint`) |
| `category` | str | `category` (e.g. `Manual`, `CDO`; `""` if absent) |
| `filename` | str | `displayName` |
| `page` | int | parser page number (1 for docx/doc) |
| `chunk` | int | chunk index within page |
| `language`, `internal_version`, … | str | flattened from the document's `metadata` JSON object: every **scalar** entry is copied as `str(value)`; non-scalar entries are skipped with a log line |
| `serial`, `base_filename`, `doc_type`, `classification_source`, `needs_review`, … | str | flattened from the document's `tags` array (`["key:value", …]`), split on the first `:`, key unprefixed, value lowercased — see the malformed-tag handling and collision-policy notes below |

  **`serial` is a `tags` key, not a `metadata` key** — the DTO's `metadata` object only carries incidental scalars such as `language`/`internal_version`. Live DTO example:

  ```json
  "tags": ["base_filename:z91465abstr", "doc_type:abs", "serial:91465"],
  "metadata": { "language": "TR", "internal_version": "1" }
  ```

  **Malformed tags:** each tag is split on the *first* `:` only (values may contain `.`/`-`); non-string entries and strings without a `:` are skipped with a log line; a key repeated within one document's `tags` keeps the first value (warning logged).

  **Collision policy:** the base/reserved keys above (`document_id`, `version_id`, `version_number`, `source`, `category`, `filename`, `page`, `chunk`) and `metadata`-flattened keys are written first; a `tags`-derived key is only added if its name doesn't already exist in chunk metadata. Reserved keys and DTO `metadata` always win over `tags` on a name collision.

  **Case:** tag values are stored lowercased (the OpenAPI tag pattern already forbids uppercase, but flattening lowercases unconditionally as a second guarantee). Because ChromaDB `where` equality is case-sensitive, query-side filter values built from business keys (`tools/access_filter.py`) must also be lowercased.

  **Observed tag key space** (for future readers, not exhaustive): `serial`, `base_filename`, `doc_type`, `classification_source`, `needs_review`. `serial` values are heterogeneous — some numeric (`91465`), some prefixed (`m01691`) — and only the numeric family aligns with `matricola` in `data/cdo.db`.

- **Version lifecycle:** the pipeline ingests the **latest version only**. Before adding chunks for a document, delete stale chunks: `collection.delete(where={"$and": [{"document_id": <id>}, {"version_id": {"$ne": <latest_version_id>}}]})`.

### 4.4 DocumentService REST contract (observed on dev; see `tests/documentservice_api/`)

- Base URL: `https://documents-api.example.com`, prefix `/api/v1/documents`.
- Auth: Entra ID **client-credentials** (MSAL `ConfidentialClientApplication`, scope `{DOCSVC_AUDIENCE}/.default`). Read role: `Documents.Read`. Copy the token-acquisition pattern from `tests/documentservice_api/conftest.py` (`_acquire_token`).
- `GET /documents?page=N&pageSize=M[&source=…][&category=…][&tag=key:value]` → `200 {"data": [DocumentResponseDto…], "pagination": {"page", "pageSize", "totalItems", "totalPages"}}`. `pageSize` max 100. Malformed tag → 400 ProblemDetails.
- `DocumentResponseDto` guaranteed fields: `documentId`, `versionId`, `displayName`, `status`, `tags` (array, required by the OpenAPI spec — see §4.3 for its `key:value` flattening). Also expected (tolerate absence — use `.get()`): `versionNumber`, `source`, `category`, `extension`, `mimeType`, `metadata` (object), `createdAt`, `deleted`.
- `GET /documents/{id}` → latest version DTO; 404 for unknown/malformed id.
- `GET /documents/{id}/download` → `200 {"url", "expiresAt", "versionId"}` (presigned SAS, short TTL — download immediately) | `404` no blob | `502` Azure unreachable. **Both 404 and 502 must be tolerated per-document** (log + skip, don't abort the run).
- `GET /documents/{id}/versions` → paged envelope of versions (has `versionId`).
- Optional `X-User-Id` header for audit.

### 4.5 Environment variables (single table — `.env.example` mirrors this)

| Var | Default | Used by |
| --- | --- | --- |
| `LLM_BASE_URL` | — (required by backend) | backend |
| `LLM_API_TOKEN` | `""` | backend |
| `LLM_MODEL` | `gpt-6-luna` | backend |
| `AUTH_MODE` | `dev` | backend |
| `JWT_TENANT_ID`, `JWT_AUDIENCE` | unset (required when `AUTH_MODE=jwt`) | backend |
| `CHROMA_CDO_DIR` | `data/chroma_cdo_db` | backend tools |
| `CHROMA_MANUALS_DIR` | `data/chroma_manuals_db` | backend tools |
| `CHROMA_DOCSVC_DIR` | `data/chroma_docsvc_db` | backend tools + ingestion |
| `CHATBOT_API_URL` | `http://localhost:8000` | Gradio frontend |
| `DOCSVC_BASE_URL` | `https://documents-api.example.com` | ingestion (same names as test suite) |
| `DOCSVC_TENANT_ID`, `DOCSVC_AUTHORITY`, `DOCSVC_AUDIENCE`, `DOCSVC_SCOPE` | see `docs/procedures/docsvc-ingestion.md` | ingestion |
| `DOCSVC_READ_CLIENT_ID`, `DOCSVC_READ_CLIENT_SECRET` | unset | ingestion |
| `DOCSVC_VERIFY_SSL` | `true` | ingestion |
| `DOCSVC_USER_ID` | unset (audit header) | ingestion |
| `LOG_LEVEL` | `INFO` | all |
| `APP_VERSION` | `dev` | backend (set by the release build, not by hand) |

`src/sme_chatbot/config.py` loads these once (plain `os.environ` + `python-dotenv`; `pydantic-settings` acceptable but not required). Existing root scripts keep their own env handling — do not touch them.

### 4.6 New dependencies (added in WP1, used everywhere)

`fastapi>=0.115`, `uvicorn>=0.30`, `sse-starlette>=2.1`, `httpx>=0.27`, `msal>=1.28`, `python-jose[cryptography]>=3.3`. Dev group: `respx>=0.21` (httpx mocking), `pytest-asyncio>=0.24`. HTTP client choice is **httpx everywhere new** (DocSvc client, Gradio frontend, tests); `requests` stays only in the untouched test suites.

---

## 5. Work packages

Sizes: S ≈ <150 changed lines, M ≈ 150–400, L ≈ 400+.

### WP1 — Package scaffold & dependencies (S) — **no dependencies; everything else depends on this**

1. `pyproject.toml`: add deps from §4.6; add `[build-system]` (hatchling) and `[tool.hatch.build.targets.wheel] packages = ["src/sme_chatbot"]` (or setuptools equivalent); add `[project.scripts]`: `sme-chatbot-api = "sme_chatbot.api.main:cli"`, `sme-ingest-docsvc = "sme_chatbot.ingestion.ingest_docsvc:main"`.
2. Create `src/sme_chatbot/` skeleton: all `__init__.py` files, empty `config.py` with the §4.5 vars, `frontends/` dir.
3. `uv sync` must pass; `uv run python -c "import sme_chatbot"` must pass.

**Acceptance:** clean `uv sync`; package importable; existing root scripts still run (`uv run python -c "import chatbot_core"` may fail on missing `LLM_BASE_URL` — that pre-existing behaviour is fine, import machinery must not be what breaks).

### WP2 — Core extraction: framework-agnostic chat loop + domain registry (M) — depends on WP1

1. `git mv tools src/sme_chatbot/tools` — adjust relative imports inside the moved files (`from .cdo_rag import …` stays valid); parameterise the two hardcoded `CHROMA_DIR` constants via `config.py` values (defaults unchanged).
2. `core/events.py`: dataclasses `TokenEvent(text)`, `ToolCallEvent(name, arguments)`, `ToolResultEvent(name, ok, citations)`, `DoneEvent(content, citations, usage)`, `ErrorEvent(message)`; `Citation` per §4.1.
3. `core/chat.py`: port the loop from root `chatbot_core.py` `make_chat_fn` into `def stream_chat(client, model, domain: DomainConfig, message, history, identity: RequestIdentity, max_tool_rounds=5) -> Iterator[ChatEvent]`. Changes from the original: yield `TokenEvent` deltas instead of cumulative Gradio strings; keep `<think>`-stripping (strip across delta boundaries — buffer inside an open
   `<think>` until it closes).
   Also: yield `ToolCallEvent`/`ToolResultEvent` around dispatch; collect citations from any tool result whose JSON has `results[*].source/page` (and `document_id` when present); finish with `DoneEvent`.
   No `gradio`, no `fastapi` imports here. The OpenAI client is constructed in `config.py`/caller, not at module import time (fixes the current import-time `EnvironmentError`).
4. `core/domains.py`: `DomainConfig(key, title, system_prompt, tool_definitions, dispatch)`; registry `DOMAINS: dict[str, DomainConfig]` with `"sample_cdo"` (prompt from root `chatbot_cdo.py`, tools from `sme_chatbot.tools`) and `"sample_manuals"` (prompt from root `chatbot_manuals.py`, tools from `sme_chatbot.tools.manuals_rag`). `"documents"` is registered by WP7 — leave a comment marker.
5. Update root `chatbot_cdo.py`/`chatbot_manuals.py` imports (`from src… via installed package`) so the legacy Gradio path still works until WP9. `chatbot_core.py` stays as-is for them.

**Acceptance:** unit-testable loop — with a scripted fake OpenAI client (streams: text-only; one tool round; JSON-broken tool args), `stream_chat` yields the expected event sequence. Legacy `uv run python chatbot_manuals.py` still launches (manual smoke, LLM reachable).

### WP3 — FastAPI backend (M) — depends on WP2

1. `api/schemas.py`: `ChatRequest`, `ChatMessage`, `ChatResponse`, `Citation` exactly per §4.1.
2. `api/auth.py`: `get_identity` dependency implementing §4.2 (`dev` + `jwt` modes; `jwt` validates signature/audience/expiry via `python-jose` with JWKS fetched at startup and cached; claim→scope mapping stubbed to unrestricted).
3. `api/streaming.py`: `ChatEvent → ServerSentEvent(event=…, data=json)` mapper.
4. `api/main.py`: `create_app()` factory; `GET /health`; `POST /chat/message` — resolves domain (422 unknown), builds OpenAI client from config (module-level singleton), returns `EventSourceResponse` when `stream=true` else runs the generator to completion and returns `ChatResponse`. Uncaught generator exception → `error` SSE event, and non-stream mode → 500 JSON. `cli()` entry point runs uvicorn
   on `:8000`.
5. Request logging: one INFO line per request with domain, user_id, message length, rounds used.

**Acceptance:** `uv run sme-chatbot-api` serves; `curl :8000/health` OK; `curl -N -X POST :8000/chat/message -d '{"domain":"sample_manuals","message":"test"}' -H 'content-type: application/json'` streams SSE ending in `done` (with real LLM configured); `stream:false` returns JSON. Unknown domain → 422.

### WP4 — Gradio frontend as HTTP client (S/M) — depends on WP3 contract only (§4.1); can be built in parallel against the spec

1. `frontends/gradio_app.py`: `gr.ChatInterface` whose chat fn calls `POST {CHATBOT_API_URL}/chat/message` with `httpx` (`stream=True`), parses SSE lines (`event:`/`data:` pairs — hand-rolled parser or `httpx-sse`), accumulates `token` deltas and yields the growing string; on `done`, appends a compact citation footer (`— fonte: file.pdf p.12`); on `error`, yields the message. Domain selector:
   `gr.Dropdown` populated from `GET /health` `domains`, passed per-message (use ChatInterface `additional_inputs`). Sends `X-User-Id` header if `GRADIO_USER_ID` env is set.
2. No imports from `sme_chatbot` internals — HTTP only (this is the point of the exercise).
3. History note: send prior turns as §4.1 `history` (map from Gradio messages format).

**Acceptance:** with WP3 running, `uv run python frontends/gradio_app.py` gives a working streamed chat with citations for both domains; killing the backend mid-chat shows a clean error message, not a traceback.

### WP5 — DocumentService client (M) — depends on WP1 only; parallel with WP2/3/4

`clients/docsvc.py`, sync `httpx`-based:

1. `DocSvcSettings.from_env()` reading the `DOCSVC_*` vars (§4.5, semantics identical to `tests/documentservice_api/conftest.py`).
2. `TokenProvider`: MSAL client-credentials with expiry-aware caching (re-acquire when <5 min remaining; MSAL's internal cache makes this nearly free). Raise `DocSvcAuthError` with an actionable message when creds/config missing.
3. `DocSvcClient`:
   - `iter_documents(source=None, category=None, tag=None, page_size=50) -> Iterator[dict]` — transparent pagination (`pagination.totalPages`), yields raw DTO dicts.
   - `get_document(document_id) -> dict`.
   - `get_download_link(document_id) -> dict` — returns DownloadLinkDto; raises `DocSvcNotReady(status)` on 404/502 (callers skip-and-log).
   - `download_content(document_id, dest: Path) -> Path` — gets link, immediately streams the presigned URL to `dest` (separate plain `httpx` GET — **no** Authorization header to the blob URL).
   - Retries: 3 attempts, exponential backoff on 429/5xx/connect errors for API calls (not for the 404/502 skip cases); timeout 30 s; sends `X-User-Id` when configured.
4. No ChromaDB / embedding / parsing imports — pure API client.

**Acceptance:** unit tests with `respx` covering: token injected; pagination walks 3 pages; 404/502 download → `DocSvcNotReady`; retry on one 500 then success; missing creds → `DocSvcAuthError`.

### WP6 — DocSvc ingestion script (M/L) — depends on WP5; parallel with WP3/4

`ingestion/ingest_docsvc.py` — **self-contained pipeline** (per decision, chunking/parsing code is *copied* from `ingest_sample_manuals.py`, not shared): `chunk_text`, `extract_pages` (pdf via `pymupdf4llm`, docx via `python-docx`, doc via `antiword`), embedding via `SentenceTransformer`.

1. CLI (argparse): `--category`, `--source`, `--tag key:value`, `--limit N` (docs, after filtering), `--dry-run` (list + parse decisions, no download/embed/write), `--page-size` (default 50).
2. Flow per document DTO: skip if `deleted`; skip unsupported extension (log); skip if all chunk IDs for `(document_id, version_id)` already present (cheap idempotency: probe one deterministic ID via `collection.get`); else `download_content` to a `tempfile.TemporaryDirectory` (suffix from `extension`), parse → chunk → embed (batch 32) → stale-version delete per §4.3 → `collection.add` in batches
   of 500.
3. Metadata exactly per §4.3, including the flatten-scalars rule for the DTO `metadata` object.
4. Resilience: per-document try/except — parse/download failures log WARNING and count as skipped; run always ends with a summary line (`N added, N skipped, N failed, collection size`). Missing DocSvc creds → exit code 2 with the same guidance message the test suite README gives.
5. Entry point `main()` wired to the `sme-ingest-docsvc` script from WP1.

**Acceptance:** unit test with a mocked `DocSvcClient` (fixture PDF on disk instead of a real download) writes chunks with the exact §4.3 metadata/ID scheme into a tmp ChromaDB, and a second run adds 0. Against dev (creds available): `sme-ingest-docsvc --limit 3 --dry-run` lists 3 documents and their parse decisions; then `--limit 3` populates `data/chroma_docsvc_db`.

### WP7 — DocSvc RAG tool + access filter + "documents" domain (M) — depends on WP2 (registry) and §4.3 (schema; does NOT need WP6 executed)

1. `tools/access_filter.py`: `build_where(identity: RequestIdentity, category=None, serial=None, customer=None, document_ids=None) -> dict | None` — combines caller filters with identity restrictions (`identity.permitted_document_ids` → `{"document_id": {"$in": [...]}}`; `identity.scopes` → equality clauses); multiple clauses under `{"$and": [...]}`; returns `None` when unrestricted. Pure function
   — unit-test heavy (Chroma `where` syntax: `$in`, `$and`, plain equality).
2. `tools/docsvc_rag.py`: mirror of `manuals_rag.py` against `CHROMA_DOCSVC_DIR`/`docsvc`; tool `search_documents(query, n_results=5, category=None, serial=None, customer=None)`; applies `build_where`; results include `document_id`, `version_id`, `source`, `category`, `filename`, `page`, `distance`. `TOOL_DEFINITIONS` + `dispatch` exported. The tool's dispatch receives `identity` via a
   `functools.partial`/closure created in `domains.py` — the LLM never sees or controls identity parameters.
3. `core/domains.py`: register domain `"documents"` — system prompt: assistant over the document library (manuals, CDO reports, certifications) synced from the DocumentService; instructs citing filename + page and staying within retrieved content; answer in the user's language (mirror the tone of the existing two prompts).
4. Empty/absent collection must degrade to the same friendly error pattern `manuals_rag.py` uses ("Run ingest… first"), not an exception.

**Acceptance:** unit tests for `build_where` combinations; integration-style test seeding a tmp Chroma collection with §4.3 metadata and asserting `search_documents` respects `category` and `permitted_document_ids` filters (a chunk outside the permitted set is never returned regardless of similarity). `GET /health` now lists `documents`.

### WP8 — Test suites for the new stack (M) — depends on WP2/3/5/6/7 (can be split per-area across agents)

- `tests/unit/test_core_chat.py` — event-sequence tests with fake OpenAI streams (WP2 acceptance cases + max-rounds cutoff).
- `tests/unit/test_access_filter.py`, `tests/unit/test_docsvc_rag.py` — WP7 cases.
- `tests/unit/test_docsvc_client.py` — WP5 respx cases.
- `tests/unit/test_ingest_docsvc.py` — WP6 idempotency + metadata cases.
- `tests/backend_api/test_chat_endpoint.py` — FastAPI `TestClient` with the fake LLM monkeypatched in: health, 422 unknown domain, non-stream JSON response, SSE stream parse (token→done), dev-mode `X-User-Id` propagation, jwt-mode 401 without token.
- Keep the repo convention: new suites must not require network/LLM/creds — everything mocked; they run in CI with plain `uv run pytest tests/unit tests/backend_api`.

**Acceptance:** all new tests pass; existing `tests/documentservice_api` behaviour (skip semantics without creds) unchanged.

### WP9 — Config, docs, container, cleanup (S/M) — depends on WP3/4/6 verified

1. `.env.example` mirroring §4.5 with comments; verify `.gitignore` covers `.env`.
2. `Dockerfile` (multi-stage, uv-based, CPU-only) running `sme-chatbot-api`; document (not necessarily implement) a `docker-compose` service entry alongside the portal per report §3.5.
3. `README.md`: new architecture diagram (backend + Gradio client + 3 collections), quickstart (`uv sync` → run API → run Gradio → run ingestion), env var table pointer.
4. Delete root `chatbot_core.py`, `chatbot_cdo.py`, `chatbot_manuals.py`; sweep docs for references (`docs/historical/2026-06-24-chatbot_architecture_report.md` §3.3 table, `docs/historical/2026-06-24-chatbot-demo.md`, `docs/specs/rag-ingestion.md`) and add a one-line "superseded by src/sme_chatbot — see this plan" note where they're mentioned (don't rewrite history in dated docs).
5. Update `TODO.md`.

**Acceptance:** fresh-clone quickstart works end-to-end following only the README; `git grep chatbot_core` finds no live code references.

---

## 6. Execution order

```text
Wave 1:  WP1
Wave 2:  WP2 ─┐            WP5 ─┐          (parallel)
Wave 3:  WP3 ─┤  WP7 ─┐    WP6 ─┤          (WP3∥WP6; WP7 after WP2)
Wave 4:  WP4 ─┴───────┴─────────┴─ WP8    (WP4 needs WP3 running; WP8 per-area, parallelizable)
Wave 5:  WP9
```

Critical path: WP1 → WP2 → WP3 → WP4. DocSvc track (WP5 → WP6) is fully parallel to it. WP7 sits between the tracks but only needs WP2 + the frozen §4.3 schema.

**Conflict avoidance for parallel agents:** WP2 owns `src/sme_chatbot/core/` + `src/sme_chatbot/tools/` (moved files); WP5 owns `clients/`; WP3 owns `api/`; WP6 owns `ingestion/`; WP7 owns `tools/docsvc_rag.py` + `tools/access_filter.py` + a 10-line append to `core/domains.py`. `pyproject.toml` is touched **only** by WP1. `config.py` keys are all declared in WP1 (from §4.5) so later WPs only
read it.

---

## 7. End-to-end verification (after Wave 4)

1. `uv run pytest tests/unit tests/backend_api` — green, no network.
2. `uv run pytest tests/documentservice_api -c tests/documentservice_api/pytest.ini` — unchanged skip/pass behaviour.
3. With LLM configured: start API, ask a manuals question through Gradio; confirm streamed tokens + citation footer; confirm one `tool_call`/`tool_result` pair in server logs.
4. With DocSvc creds: `sme-ingest-docsvc --limit 5`; re-run → `0 added`; ask a question in the `documents` domain and get a cited answer with a `document_id` in the citation.
5. `curl` the SSE endpoint directly (the .NET-proxy view of the world) and archive the transcript in the PR description.

## 8. Explicitly out of scope (later phases, per the architecture report)

.NET proxy controller and React chat UI (report Phase 3); real JWT claim→ACL mapping (blocked on report §12.3); SQL Server vector migration (Phase 4); Langfuse observability (Phase 4 — but the `ChatEvent` stream and single `stream_chat` entry point were shaped to make `@observe` wrapping trivial); event-driven ingestion triggers (webhook/queue — this phase is CLI/cron); write-path DocumentService
operations.

## 9. Risks & notes for executing agents

- **ChromaDB metadata is scalars-only** — nested `metadata` from DocSvc must be flattened (§4.3) or `collection.add` raises.
- **Presigned URLs expire in 5–15 min** — download immediately after `get_download_link`; never queue links.
- **`<think>` stripping across stream deltas** (WP2): the legacy code strips on the cumulative string; the delta-based port must buffer inside an open `<think>` block or fragments will leak to clients.
- **Import-time side effects**: legacy `chatbot_core.py` raises at import without `LLM_BASE_URL` and connects eagerly. The new package must construct clients lazily (WP2.3) or tests can't import modules.
- **`pageSize` cap is 100** on `GET /documents`; page from 1.
- **Dev data is shared** — ingestion is read-only against DocSvc, but be conservative with `--limit` on first runs; the corpus size on dev is unknown.
- **Legacy scripts keep running until WP9** — don't break `chatbot_cdo.py`/`chatbot_manuals.py` imports in Waves 2–4; CI for them is manual smoke only.

<!-- EOF -->
