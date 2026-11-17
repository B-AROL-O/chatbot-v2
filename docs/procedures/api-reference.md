# API Reference — sme-chatbot backend

How to call the FastAPI backend (`src/sme_chatbot/api/main.py`) as a client —
the Gradio frontend and the future .NET portal proxy both consume exactly
this. The frozen wire contract lives in
[`specs/2026-07-02-backend-split-and-docsvc-ingestion-plan.md`](../specs/2026-07-02-backend-split-and-docsvc-ingestion-plan.md)
§4.1/§4.2 — this page is the practical how-to; if the two ever disagree, the
spec wins and this page has drifted.

Base URL locally: `http://localhost:8000`. Interactive Swagger UI at
`/docs`, raw schema at `/openapi.json` (both come from FastAPI for free — not
hand-maintained).

## Authentication

Controlled by `AUTH_MODE` (see `.env.example`):

| Mode | Behavior |
| ---- | -------- |
| `dev` (default) | No `Authorization` required. Optional `X-User-Id` header sets the caller's `user_id`; identity is otherwise unrestricted. |
| `jwt` | Requires `Authorization: Bearer <token>`, validated against Entra ID (JWKS fetched lazily and cached for 1h). A validated token currently still yields an unrestricted identity — claim→scope mapping is a stub (see `api/auth.py`). |

Handlers and tools never see the raw header/token — only the resolved
`RequestIdentity` (`user_id`, `permitted_document_ids`, `scopes`).

## `GET /health`

No auth, regardless of `AUTH_MODE` — an orchestrator restarting a stuck pod
must never need credentials. Liveness only: the live domain registry, build
version, and process uptime.

```json
{"status": "ok", "domains": ["documents", "sample_cdo", "sample_manuals"], "version": "0.5.2", "uptime_seconds": 123.4}
```

`version` is the release tag baked in at image build time, `"dev"` for a
local/manual build. `uptime_seconds` is time since the worker process
started, not since the last request.

Anything about how the backend is *configured* (model, LLM URL, auth mode)
lives in `GET /config` instead — see below.

## `GET /config`

Static backend configuration, gated by the same auth as `/chat/message` (see
[Authentication](#authentication) above) — open in `AUTH_MODE=dev`, requires
a bearer token in `AUTH_MODE=jwt`. Secrets (API tokens, JWT signing config,
etc.) are never included.

```json
{
  "version": "0.5.2",
  "model": "gpt-6-luna",
  "llm_base_url": "https://...",
  "auth_mode": "dev",
  "domains": ["documents", "sample_cdo", "sample_manuals"]
}
```

## `POST /chat/message`

Request body:

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

| Field | Required | Notes |
| ----- | -------- | ----- |
| `domain` | yes | One of the `GET /health` domain keys (`sample_cdo`, `sample_manuals`, `documents`). Unknown key → `422`. |
| `message` | yes | The user's latest turn. |
| `history` | no (default `[]`) | Prior turns, `role` restricted to `user`/`assistant`. |
| `stream` | no (default `true`) | See below. |

### `stream: true` (default) — SSE

Response is `text/event-stream` (via `sse-starlette`). Client renders
`token` deltas cumulatively; the stream always ends in exactly one of
`done`/`error`.

| `event:` | `data:` shape | Notes |
| -------- | ------------- | ----- |
| `token` | `{"text": "<delta>"}` | Already `<think>`-stripped (Qwen3 chain-of-thought) |
| `tool_call` | `{"name": "search_manuals", "arguments": {...}}` | Emitted when the model requests a tool, before dispatch |
| `tool_result` | `{"name": "...", "ok": true, "citations": [Citation, ...]}` | Full tool payload is deliberately not streamed — only success + citations |
| `done` | `{"content": "<full answer>", "citations": [Citation, ...], "usage": Usage}` | Terminal on success |
| `error` | `{"message": "..."}` | Terminal, sent instead of `done` |

### `stream: false` — single JSON response

Same fields as the `done` event, returned as one `ChatResponse` body once
the loop finishes:

```json
{"content": "...", "citations": [...], "usage": {"rounds": 2, "timings": {...}}}
```

An error mid-loop becomes an HTTP `500` with `{"detail": "<message>"}`
instead of an `error` event.

### `usage` object

```json
{
  "rounds": 2,
  "round_limit_hit": false,
  "timings": {
    "setup_ms": 40,
    "total_ms": 900,
    "ttft_ms": 800,
    "llm_ms": 650,
    "tool_ms": 250,
    "llm_calls": [
      {"round": 0, "first_chunk_ms": 100, "total_ms": 150, "finish": "tool_calls"},
      {"round": 1, "first_chunk_ms": 400, "total_ms": 500, "finish": "stop"}
    ],
    "tools": [{"name": "search_documents", "ms": 250, "ok": true}]
  }
}
```

- `rounds` is the only field a client must rely on; `timings` is additive and
  informational, so consumers should ignore keys they don't know.
- `round_limit_hit` is `true` when the turn used up its tool rounds and the
  answer came from the forced tools-less call. `rounds` alone
  can't tell this apart from an answer given in the last allowed round. The
  backend logs it on each request's `chat/message` line, so the hit rate can
  be counted in Loki, and eval reports count these answers.
- `setup_ms` is the time spent before the first LLM call (identity binding,
  `resolve_serial`). `total_ms` is the loop itself, so end to end is roughly
  `setup_ms + total_ms`.
- `ttft_ms` is measured from the start of the loop to the first *visible*
  (`<think>`-stripped) token, and is `null` if none was emitted.
  `first_chunk_ms` of each LLM call is the first chunk of any kind, so a long
  silent thinking phase shows up there.
- `finish` is `tool_calls`, `stop`, or `forced_final` (the tools-less call made
  after hitting the round cap).
- Token counts and cost are not part of `usage` yet.

### `Citation` object

```json
{
  "source": "manuale_TSI200.pdf",
  "page": 3,
  "document_id": "1234-uuid-or-null",
  "distance": 0.42,
  "filename": "manuale_TSI200.pdf",
  "repository": "ACME-NETWORK-SHARE"
}
```

- `source` is always the document's display name (never the origin share).
- `document_id` is `null` for chunks from the two legacy local-dir Chroma
  collections (`sample_cdo`, `sample_manuals`); populated for `documents`
  (DocumentService-backed).
- `filename`/`repository` are additive — clients reading only
  `source` are unaffected.

### Validation errors

Malformed request body or unknown `domain` → `422` with FastAPI's standard
`HTTPValidationError` shape (`detail: [{loc, msg, type, ...}]`).

## Regenerating this doc's source of truth

The schema above was captured from a live app instance (no server run
needed — the OpenAI client is a lazy singleton):

```bash
uv run python -c "
import json
from sme_chatbot.api.main import create_app
print(json.dumps(create_app().openapi(), indent=2))
"
```

Or, with the backend actually running (`uv run sme-chatbot-api` or
`docker compose up chatbot-api`), fetch it directly:

```bash
curl -s http://localhost:8000/openapi.json | jq .
```

Note the generated schema only fully describes `GET /health` and the
request side of `POST /chat/message` — the response is `{}` in
`openapi.json` because the endpoint returns either an SSE stream or a plain
`ChatResponse` depending on `stream`, which FastAPI can't express as one
static response schema (`response_model=None` in `api/main.py`). The SSE
event table above is the actual contract; it isn't visible in `/docs`.
