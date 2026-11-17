# Docker Deployment

How to run the sme-chatbot stack as Docker containers using the
`docker-compose.yml` at the project root.

Both services share a single image built from the root `Dockerfile`
(multi-stage, `uv`-based, CPU-only — see the Dockerfile header comment for
why torch never pulls CUDA). `chatbot-api` runs the FastAPI backend
(`sme-chatbot-api`); `chatbot-gradio` reuses the same image but overrides the
command to run the thin Gradio HTTP/SSE client instead
(`frontends/gradio_app.py`).

---

## Services

| Service | Entry point | Host port |
| --- | --- | --- |
| `chatbot-api` | `sme-chatbot-api` (FastAPI backend: `/health`, `/config`, `/chat/message`) | 8000 |
| `chatbot-gradio` | `frontends/gradio_app.py` (HTTP/SSE client of `chatbot-api`) | 7860 |

`chatbot-gradio` talks to `chatbot-api` over the compose network at
`http://chatbot-api:8000` (set via `CHATBOT_API_URL` in `docker-compose.yml`).

A future portal-side proxy (a .NET web portal calling `chatbot-api`
directly) is not implemented in this compose file — see
`docs/adr/0001-backend-owns-chat-logic-frontends-are-thin-clients.md`.

---

## Prerequisites

| Tool | Check |
| --- | --- |
| Docker Engine ≥ 24 | `docker --version` |
| Docker Compose plugin | `docker compose version` |

---

## Step 1 — Populate `data/`

Both containers mount `./data` at runtime; the directory must exist and be
populated **before** you start the containers.

Run the ETL and ingest scripts once from the project root (inside the dev
container or any Python ≥ 3.13 environment with `uv` — `requires-python` is
`>=3.13` and `.python-version`/CI use 3.13, while the image itself runs on
`python:3.14-slim`):

```bash
# Build the SQLite database
uv run python db/init_db.py
uv run python db/etl.py

# Build the Documents vector store from the DocumentService API (the happy path)
uv run sme-ingest-docsvc --limit 5
```

`ingest_sample_cdo.py` and `ingest_sample_manuals.py` are optional, legacy
local-directory pipelines for developers who don't have valid DocumentService
credentials:

```bash
# Optional: build the CDO vector store from local files (needs data/input/CDO/)
uv run python ingest_sample_cdo.py

# Optional: build the Manuals vector store from local files (needs data/input/manuals/)
uv run python ingest_sample_manuals.py
```

After this, `data/` should contain:

```text
data/
├── cdo.db
├── chroma_docsvc_db/          # from sme-ingest-docsvc
├── chroma_cdo_db/             # only if ingest_sample_cdo.py was run
└── chroma_manuals_db/         # only if ingest_sample_manuals.py was run
```

The containers run as the unprivileged user `app`, **uid/gid 1000**, not
root. The backend writes to `./data` (Chroma persists its collections there),
so the host directory must be writable by uid 1000:

```bash
sudo chown -R 1000:1000 data/
```

On most single-user Linux hosts and in the dev container the first user is
already uid 1000 (`id -u` to check), so there is nothing to do.

---

## Step 2 — Create `.env`

Copy the example file and fill in your credentials:

```bash
cp .env.example .env
# edit .env — set LLM_BASE_URL, LLM_API_TOKEN, LLM_MODEL at minimum
```

See `.env.example` for the full variable reference (LLM, backend auth, Chroma
paths, Gradio frontend, DocumentService ingestion, logging).

---

## Step 3 — Build the image

```bash
docker compose build
```

The build pre-downloads the sentence-transformers embedding model
(`paraphrase-multilingual-MiniLM-L12-v2`, ~400 MB CPU weights) into
`/app/.cache/huggingface` (`HF_HOME`), so the first request doesn't have to
fetch it.

If the model download requires a Hugging Face token, pass it as a build
argument:

```bash
docker compose build --build-arg HF_TOKEN=hf_xxx
```

---

## Step 4 — Run

**One service only:**

```bash
docker compose up chatbot-api       # http://localhost:8000
docker compose up chatbot-gradio    # http://localhost:7860 (needs chatbot-api running)
```

**Both:**

```bash
docker compose up
```

Add `-d` to run in the background.

The image declares a `HEALTHCHECK` that polls `GET /health` on port 8000, so
`docker compose ps` shows `chatbot-api` as `(healthy)` once it is serving.
`chatbot-gradio` declares `depends_on: condition: service_healthy`, so
`docker compose up` starts it only after the backend is healthy. Nothing
listens on :8000 in `chatbot-gradio`, so compose overrides the image's
healthcheck for it with one that polls Gradio's own `GET /config` on :7860.
That endpoint answers even while the backend is down, so `(healthy)` there
means the frontend is up, not the backend.

---

## Stopping

```bash
docker compose down
```

---

## Port conflict

If something already occupies host port 8000 or 7860, find the owner:

```bash
ss -tlnp sport = :8000
```

Either stop that process or change the host-side port in `docker-compose.yml`:

```yaml
ports:
  - "8001:8000"   # use 8001 on the host instead
```

---

## Environment variables

All variables from `.env` are forwarded to the containers. Key ones (full
reference: `.env.example`):

| Variable | Purpose |
| --- | --- |
| `LLM_BASE_URL` | OpenAI-compatible endpoint (e.g. Ollama, Azure OpenAI) |
| `LLM_API_TOKEN` | Bearer token for the LLM endpoint |
| `LLM_MODEL` | Model name (default `gpt-6-luna`; e.g. `qwen3.6:35b`) |
| `LLM_API` | `chat_completions` (default; `/v1/chat/completions`, needed for Ollama) or `responses` (`/v1/responses`, Azure OpenAI only). Set `responses` if Azure rejects tool calls with "Function tools with reasoning_effort are not supported ... in /v1/chat/completions". `LLM_SEED` is not sent on `responses` |
| `AUTH_MODE` | `dev` (default, no token) or `jwt` (Entra ID bearer) — backend auth |
| `CHATBOT_API_URL` | Set automatically for `chatbot-gradio` in `docker-compose.yml` |
| `RETRIEVAL_WARMUP` | `off` (default), `on` or `full` — load the retrieval stores at startup instead of on the first request; see below |
| `HF_TOKEN` | Hugging Face token (only needed if the embedding model is gated) |
| `LOG_LEVEL` | Python log level — `INFO` (default) or `DEBUG` |

### Warming up retrieval at startup (`RETRIEVAL_WARMUP`)

By default the embedding model and the Chroma stores load on the first
request, so the first question after every restart or release waits
minutes for them. `RETRIEVAL_WARMUP` does that work at startup instead, in a
background thread: `/health` answers at once, and a request that arrives
during the warm-up waits for the same load instead of starting a second one.

| Value | What it does |
| --- | --- |
| `off` (default) | Nothing; stores load on first use |
| `on` | Loads the model, opens the collections, runs one query of each kind |
| `full` | Reads the documents store's files (about 9.6 GB) into the OS page cache, then does `on` |

Measured on the full documents store (857k chunks) in a devcontainer after
dropping the page cache:

| | warm-up | 1st request setup | searches on the first questions |
| --- | --- | --- | --- |
| `off` | — | 233 s | 0.5–17.5 s |
| `on` | 237 s | 0.5 s | 0.5–15.7 s |
| `full` | 166 s | 0.1 s | 0.3–0.5 s |

Use `full` when the host has the RAM to keep the store cached next to the
LLM; otherwise use `on`. Follow progress with
`docker compose logs chatbot-api | grep warm-up`; the last line is
`warm-up (full) finished in … s`.

The image also sets `HF_HUB_OFFLINE=1`: the model is baked in at build time,
so loading it makes no calls to huggingface.co (about 3 s saved per load).

---

## Useful commands

```bash
# Rebuild after source changes (data volume is preserved)
docker compose build --no-cache && docker compose up

# Tail logs from one service
docker compose logs -f chatbot-api

# Open a shell in a running container
docker compose exec chatbot-api bash

# Remove containers, networks, and local image
docker compose down --rmi local
```

---

## Running ingestion from the image (no source checkout needed)

`sme-ingest-docsvc` (`src/sme_chatbot/ingestion/ingest_docsvc.py`) is baked
into the image's venv and already on `PATH` — the same image used for
`chatbot-api`/`chatbot-gradio` can run it directly, with no `uv`/Python
checkout. This is the way to (re)populate `data/chroma_docsvc_db/` on a host
that only has `docker-compose.yml` + `.env`, such as the remote hosts in
[Deploying a released image to a remote host](#deploying-a-released-image-to-a-remote-host-eg-deploy-host)
below.

With `docker-compose.yml` + `.env` present (reuses the `./data:/app/data`
volume and `env_file` already declared for `chatbot-api`):

```bash
# one-off container: runs ingestion, then exits
docker compose run --rm chatbot-api sme-ingest-docsvc --limit 5

# or, if chatbot-api is already running:
docker compose exec chatbot-api sme-ingest-docsvc --limit 5

# rebuild the lexical (FTS5) index afterwards, same way
docker compose run --rm chatbot-api sme-ingest-docsvc --build-lexical-index
```

With only the image itself (no compose file), wire the same volume/env-file
by hand:

```bash
docker run --rm \
  --env-file .env \
  -v "$(pwd)/data:/app/data" \
  ghcr.io/b-arol-o/chatbotv2:latest \
  sme-ingest-docsvc --limit 5
```

Only the `DOCSVC_*` vars (`DOCSVC_TENANT_ID`, `DOCSVC_READ_CLIENT_ID`,
`DOCSVC_READ_CLIENT_SECRET`, `DOCSVC_BASE_URL`, `DOCSVC_AUDIENCE`) are needed —
`config.py` defaults `LLM_*` to empty strings and ingestion never touches the
LLM client. Drop `--limit 5` for a full sync; add `--category`/`--source`/
`--tag` to filter, or `--dry-run` to preview without downloading/embedding.
The resulting `docsvc` Chroma collection lands under
`data/chroma_docsvc_db/` on the host and is picked up by `chatbot-api`
lazily, on its next request — no restart needed.

---

## Deploying a released image to a remote host (e.g. `deploy-host`)

Everything above builds the image locally from source. On a deployment/test
host you instead **pull a published release image** — see
`docs/procedures/release-chatbot.md` for how images are built and tagged
(`ghcr.io/b-arol-o/chatbotv2:<version>`) and for
what to do if the registry refuses the pull.

This section walks through a real run on `deploy-host`; adjust host, paths,
and the data archive filename to your case.

### Step 1 — Create a working directory on the host

```bash
ssh user@deploy-host
mkdir -p $HOME/chatbotv2-test/data/archive
cd $HOME/chatbotv2-test
```

Docker Compose names the project, and so the containers and network, after
this directory. To rename an existing deployment directory, stop the stack
first, or the running containers are left outside the new project:

```bash
cd ~/<old-dir> && docker compose down
mv ~/<old-dir> ~/chatbotv2-test
cd ~/chatbotv2-test && docker compose up -d
```

### Step 2 — Copy over `docker-compose.yml`, `.env.example`, and a data backup

Copy these from your workstation to the host — via `scp`, or by opening the
host as a VS Code Remote-SSH target and drag-dropping the files into the
Explorer:

- `docker-compose.yml` and `.env.example` from the repo root → `~/chatbotv2-test/`
- The `alloy/` directory from the repo root → `~/chatbotv2-test/alloy/`
  (only needed for [Shipping logs to Loki](#shipping-logs-to-loki))
- A pre-populated `data/` backup archive (zip/tar of `cdo.db` +
  `chroma_*_db/`, produced elsewhere per [Step 1 — Populate `data/`](#step-1--populate-data)
  above) → `~/chatbotv2-test/data/archive/`

Extract the archive into `data/`:

```bash
cd $HOME/chatbotv2-test
unzip data/archive/<your-backup>.zip
# should populate data/cdo.db, data/chroma_cdo_db/, data/chroma_docsvc_db/, etc.
```

### Step 3 — Create `.env`

```bash
cp .env.example .env
```

Edit `.env` for the target environment. For example, to point at an
internally hosted Ollama endpoint instead of a cloud LLM:

```diff
--- .env.example
+++ .env
@@ -5,14 +5,14 @@
 # === LLM (FastAPI backend, src/sme_chatbot/api) — required ===
-LLM_BASE_URL=https://your-llm-endpoint/api/v1
-LLM_API_TOKEN=changeme
-LLM_MODEL=gpt-6-luna
+# LLM_BASE_URL=https://your-llm-endpoint/api/v1
+# LLM_API_TOKEN=changeme
+# LLM_MODEL=gpt-6-luna

 # === Example: Ollama instead of a hosted LLM ===
-# LLM_BASE_URL=http://deploy-host:11434/v1
-# LLM_API_TOKEN=your-token
-# LLM_MODEL=qwen3.6:35b
+LLM_BASE_URL=http://deploy-host:11434/v1
+LLM_API_TOKEN=your-token
+LLM_MODEL=qwen3.6:35b
```

### Step 4 — No edit needed: `docker-compose.yml` already pins a registry image

The root `docker-compose.yml` sets `image: *chatbot-image` (via the
`x-chatbot-image` anchor, default
`ghcr.io/b-arol-o/chatbotv2:latest`) *alongside*
`build: .` on both services. That means the file you copied over in Step 2
already works for `docker compose pull` as-is — there's no swap to make.

`build: .` stays in the file even though `deploy-host` has no Dockerfile or
source checked out — that's harmless. `docker compose pull` only looks at
`image:`, and the `docker compose up -d` right after it finds the image
already present locally, so it skips building. Just don't run
`docker compose build` (or `up --build`) on this host — with no Dockerfile in
`.`, that would fail. If you'd rather remove the possibility entirely, comment
out `build: .` on both services after copying the file over.

To deploy a specific version instead of the `:latest` default, set
`CHATBOT_IMAGE` before pulling (see
[Upgrading or rolling back](#upgrading-or-rolling-back) below), or edit the
default in the `x-chatbot-image:` line.

### Step 5 — Pull and start

The image is public, so no registry login is needed. If the pull fails with
`unauthorized`, see `docs/procedures/release-chatbot.md` "Pulling the image".

```bash
docker compose pull
docker compose up -d
```

Expected output:

```text
[+] up 3/3
 ✔ Network chatbotv2-test_default            Created
 ✔ Container chatbotv2-test-chatbot-api-1    Started
 ✔ Container chatbotv2-test-chatbot-gradio-1 Started
```

### Step 6 — Verify

```bash
docker compose logs -f
```

Look for `Application startup complete.` on `chatbot-api`, and a `GET
http://chatbot-api:8000/health "HTTP/1.1 200 OK"` line from `chatbot-gradio`
confirming it can reach the backend. Then browse to
`http://deploy-host:7860` (Gradio) or `curl http://deploy-host:8000/health`.

### Upgrading or rolling back

Bump (or revert) the tag and re-pull — no rebuild needed:

```bash
CHATBOT_IMAGE=ghcr.io/b-arol-o/chatbotv2:<version> docker compose pull
CHATBOT_IMAGE=ghcr.io/b-arol-o/chatbotv2:<version> docker compose up -d
```

Or edit the default in the `x-chatbot-image:` line in `docker-compose.yml` if
you'd rather not pass the env var each time.

`scripts/deploy_chatbot_remotely.sh <version>` wraps the pull/up/verify sequence
above — it polls `/health` until it reports the requested version instead of
eyeballing `docker compose logs -f`, and prints `docker compose ps` plus
recent logs if the rollout doesn't come up in time. Run it directly on the
host (from the deployment directory, alongside `docker-compose.yml`), or from
your workstation with `REMOTE_USER`/`REMOTE_HOST` set — it then pipes itself
over `ssh` and runs there, no manual copy needed:

```bash
REMOTE_USER=user REMOTE_HOST=deploy-host scripts/deploy_chatbot_remotely.sh <version>
```

---

## Shipping logs to Loki

The `alloy` service in `docker-compose.yml` ships the logs of this stack's
containers to a shared Loki at `LOKI_URL`, where they can be read in the
Grafana that fronts it. It follows Option A of
`external/observability/docs/guides/connect-an-application.md` (the
observability stack's own repo), which explains the endpoint, the tenancy and
the troubleshooting in full.

It is **off by default**. The service sits in the `observability` compose
profile, so `docker compose up` on a dev box does not start it.

### What gets shipped, under which labels

| Label | Value | Source |
| --- | --- | --- |
| `job` | `chatbot` | fixed in `alloy/client.alloy`. Unique across the estate; follows the shared Loki's naming convention (product name, brand as its own label) |
| `service_name` | `chatbot` | fixed. Must equal `job`, or trace↔log correlation never resolves |
| `brand` | `LOKI_BRAND` | default `acme`. Lets streams be split by brand |
| `env` | `ALLOY_ENV` | exactly `dev`, `staging` or `prod`. Any other value drops out of retention and dashboards |
| `hostname` | `HOSTNAME_LABEL` | e.g. `deploy-host` |
| `compose_service` | `chatbot-api`, `chatbot-gradio`, `alloy` | the compose service name |
| `container` | the container name | Docker |

Only containers labelled `com.sme-chatbot.telemetry.job=chatbot` are shipped, so
other workloads on the same host never land under this `job`. All three
services in `docker-compose.yml` carry the label.

Metrics and traces are not shipped yet. The application does not export
OTLP, so the guide's OTLP pipelines would have nothing to forward.

### One-time setup

1. **Credential.** Ask the observability stack owner for an *ingest*
   credential of your own (`chatbot-agent`). It is write-only. They add
   it on the Loki host as described in §2 of the guide.
2. **Network.** The deployment host needs outbound `443` to the Loki
   endpoint's host, resolved through the corporate DNS. Prove the
   path from the host before starting the agent:

   ```bash
   curl -sS -o /dev/null -w 'healthz %{http_code}\n' https://loki.example.com/healthz   # 200
   curl -sS -o /dev/null -w 'push %{http_code}\n' -u chatbot-agent \
     -H 'Content-Type: application/json' https://loki.example.com/loki/api/v1/push \
     --data-binary '{"streams":[{"stream":{"job":"smoke","env":"dev"},
       "values":[["'"$(date +%s)"'000000000","hello"]]}]}'                                 # 204
   ```

   `(6) Could not resolve host` means DNS. A hang means a missing firewall
   rule. **401** means a wrong password. **403** means the credential is not
   in the proxy's `map` block yet.
3. **Config.** Copy `alloy/` next to `docker-compose.yml` on the host
   (Step 2 above), then uncomment and fill the *Log shipping to Loki* block
   in `.env`:

   ```bash
   COMPOSE_PROFILES=observability
   ALLOY_ENV=prod
   HOSTNAME_LABEL=deploy-host
   LOKI_URL=https://loki.example.com/loki/api/v1/push   # no default; required
   INGEST_USERNAME=chatbot-agent
   INGEST_PASSWORD=<the password you were given>
   ```

   `.env` is also the `env_file` of the chatbot containers, so the ingest
   password is visible in their environment too. It can only write logs,
   never read them.

### Start and verify

```bash
docker compose up -d                 # COMPOSE_PROFILES in .env adds alloy
docker compose logs -f alloy         # no "level=error" lines; a 401/403 shows here
```

Then, in Grafana's Loki datasource, query `{job="chatbot"}`. Lines from
`chatbot-api` and `chatbot-gradio` should show up within a few seconds.

After editing `alloy/client.alloy`, validate it before restarting the agent:

```bash
docker run --rm -v "$PWD/alloy:/etc/alloy:ro" grafana/alloy:v1.20.1 validate /etc/alloy
docker compose restart alloy
```
