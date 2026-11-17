# Langfuse Setup (LLM Call Tracing)

How to obtain Langfuse credentials and wire them into the backend so every
LLM call (`chat.completions.create`) is traced — prompt, completion, tokens,
latency, and cost.

Langfuse integration is **opt-in**: with `LANGFUSE_ENABLED=false` (the default)
the backend uses the plain `openai.OpenAI` client and nothing changes. When
enabled, `src/sme_chatbot/api/main.py::_get_client()` swaps in Langfuse's
drop-in OpenAI wrapper (`langfuse.openai.OpenAI`); `core/chat.py` is untouched.


## Prerequisites

| Requirement | Notes |
| ------------- | ------- |
| A Langfuse instance | Langfuse Cloud (`https://cloud.langfuse.com`) or a self-hosted deployment |
| Backend env file | Copy `.env.example` to `.env` if you haven't already |

---

## Step 1 — Create a project and API keys

On Langfuse Cloud, API keys are scoped to a **project**, nested under an
**organization**. After logging in:

1. **Create an Organization** (if you don't have one) — from the home screen,
   click **New Organization** and name it (e.g. `MY_ORG`).
2. **Create a Project** inside that org — **New Project**, name it
   (e.g. `sme-chatbot`).
3. **Generate API keys** — open the project, go to **Settings → API Keys →
   Create new API keys**. You get:
   - **Public Key** — `pk-lf-...`
   - **Secret Key** — `sk-lf-...` (**shown only once — copy it immediately**)
   - **Host** — the base URL of your instance (see Step 2 for the region caveat)

The same three fields exist on a self-hosted instance under the project's
**Settings → API Keys**; the host is your own deployment URL.

---

## Step 2 — Fill in `.env`

Add (or edit) the Langfuse block in your `.env`:

```text
LANGFUSE_ENABLED=true
LANGFUSE_BASE_URL=https://cloud.langfuse.com
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
```

`LANGFUSE_BASE_URL` matches the name shown in Langfuse's Project API Key
dialog. (The SDK also honors the older `LANGFUSE_HOST`, but it is deprecated —
use `LANGFUSE_BASE_URL`.)

Two things to watch:

- **Region matters.** Langfuse Cloud has separate EU and US regions with
  separate hosts. If the URL bar shows `us.cloud.langfuse.com` after login, set
  `LANGFUSE_BASE_URL=https://us.cloud.langfuse.com` — pointing at the wrong
  region means traces silently never appear.
- **No inline comments.** Per the warning at the top of `.env.example`, some
  dotenv parsers used here (docker compose `env_file`, `--env-file`, shell
  sourcing) do not strip a trailing `# comment`, which would corrupt the key
  value. Keep each value on its own clean line.

The backend reads these via `src/sme_chatbot/config.py`. The Langfuse SDK
itself also reads `LANGFUSE_BASE_URL` / `LANGFUSE_PUBLIC_KEY` /
`LANGFUSE_SECRET_KEY` from the environment automatically.

---

## Step 3 — Restart and verify

Restart the backend so it picks up the new env and rebuilds the client
singleton:

```bash
uv run sme-chatbot-api          # local
# or
docker compose up chatbot-api    # Docker
```

Send a chat request:

```bash
curl -N -X POST http://localhost:8000/chat/message \
  -H 'Content-Type: application/json' \
  -d '{"domain": "sample_cdo", "message": "hello", "stream": false}'
```

Then open your Langfuse project → **Tracing**. A generation should appear with
the prompt, completion, token counts, latency, and (where the backend reports
usage) cost.

---

## Troubleshooting

| Symptom | Likely cause |
| --------- | -------------- |
| No traces appear | `LANGFUSE_BASE_URL` points at the wrong region; or `LANGFUSE_ENABLED` is not `true`; or the backend was not restarted |
| Backend won't start / import error | `langfuse` not installed — run `uv sync` |
| Traces appear but **token/cost fields are empty** | The LLM backend isn't returning usage. For streaming, the OpenAI-compatible server must support `stream_options.include_usage`; some local backends (e.g. Ollama) may not populate it. This is a backend-side gap, not a bug in the integration |
| Auth error to Langfuse | Public/secret key mismatch, or keys belong to a different project/region than `LANGFUSE_BASE_URL` |

---

## Scope and follow-ups

This pass covers **LLM call tracing only**. Deliberately out of scope (natural
follow-ups):

- Nested spans around tool dispatch (SQL / RAG calls) inside `core/chat.py`.
- Request-level trace tagging (`user_id` / `domain` / session) — needs a spike
  to confirm contextvar propagation across the SSE streaming path.
- Thumbs-up/down feedback scoring.
