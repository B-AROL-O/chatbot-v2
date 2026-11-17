# Multi-stage, uv-based, CPU-only image for the FastAPI backend
# (src/sme_chatbot/api) — entry point `sme-chatbot-api` on port 8000.
#
# CPU-only: pyproject.toml pins `torch` to the official PyTorch CPU wheel
# index (see the [tool.uv.sources] comment there), so `uv sync --frozen`
# never pulls the CUDA runtime (~2 GB of nvidia-*/cuda-* packages) — this
# service only runs embedding *inference*, never GPU training.
#
# The legacy Gradio launchers (chatbot_core.py / chatbot_cdo.py /
# chatbot_manuals.py) that this image used to serve are gone; the Gradio
# frontend (frontends/gradio_app.py) is now a thin HTTP/SSE client and
# reuses this same image, only overriding CMD (see docker-compose.yml's
# `chatbot-gradio` service) — that is why `frontends/` is copied too even
# though the default CMD below never runs it.

# ---- builder ---------------------------------------------------------------
FROM python:3.15-slim AS builder

# Install uv from its official distroless image, pinned (tag + digest) to the
# same version CI and the release workflow use (UV_VERSION in
# .github/workflows/ci-chatbot.yml) - bump all three together.
COPY --from=ghcr.io/astral-sh/uv:0.11.26@sha256:3d868e555f8f1dbc324afa005066cd11e1053fc4743b9808ca8025283e65efa5 /uv /usr/local/bin/uv

WORKDIR /app

# HF_HOME lives under /app (not /root/.cache) so the model downloaded below
# can be copied into the runtime stage and is readable by its non-root user.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    HF_HOME=/app/.cache/huggingface

# Install dependencies first (cached across source-only changes), then the
# project itself in a second layer.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src/ src/
RUN uv sync --frozen --no-dev

# Pre-download the multilingual embedding model (~400 MB CPU weights) so it
# is baked into the image and first request latency is low. HF_TOKEN is only
# needed if the model is gated. Builder-stage only - the runtime stage below
# copies the venv/src, never this ARG, so it never reaches the final image.
# hadolint ignore=DL3064
ARG HF_TOKEN=""
RUN HF_TOKEN="${HF_TOKEN}" uv run python -c \
    "from sentence_transformers import SentenceTransformer; \
     SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

# ---- runtime -----------------------------------------------------------
FROM python:3.15-slim AS runtime

# antiword: converts legacy .doc files to text (used by ingestion scripts).
# Version pinned to python:3.14-slim's current trixie candidate - bump
# deliberately (`apt-cache policy antiword` inside the base image) rather
# than letting it float.
RUN apt-get update \
    && apt-get install -y --no-install-recommends antiword=0.37-17 \
    && rm -rf /var/lib/apt/lists/*

# Unprivileged runtime user. uid/gid 1000 is fixed (not left to useradd) so
# the host-side ./data bind mount can be chowned to match - see
# docs/procedures/docker-deployment.md "Step 1".
RUN groupadd --gid 1000 app \
    && useradd --uid 1000 --gid app --create-home --shell /usr/sbin/nologin app \
    && install -d -o app -g app /app

WORKDIR /app

# Copy the fully-populated virtual environment, application source and the
# pre-downloaded embedding model from the builder stage; no uv/build tooling
# ships in the final image.
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --from=builder --chown=app:app /app/src /app/src
COPY --from=builder --chown=app:app /app/.cache /app/.cache
COPY --from=builder --chown=app:app /app/pyproject.toml /app/uv.lock ./
COPY --chown=app:app frontends/ frontends/

# HF_HUB_OFFLINE: the model is baked in above, so loading it needs no network.
# Without it, every load still asks huggingface.co for updates (about 35
# requests at startup). Changing the embedding model therefore
# needs a rebuild, which it does anyway to match the stored vectors.
ENV PATH="/app/.venv/bin:${PATH}" \
    HF_HOME=/app/.cache/huggingface \
    HF_HUB_OFFLINE=1

# Set by the release workflow's `docker build --build-arg` (the release tag);
# defaults to "dev" for local/manual builds. Surfaced at GET /health and in
# the OpenAPI info.version — read via config.APP_VERSION.
ARG APP_VERSION=dev
ENV APP_VERSION="${APP_VERSION}"

# Numeric (= the app user above) so orchestrators can verify runAsNonRoot.
USER 1000:1000

EXPOSE 8000

# python, not curl: the slim base image ships no curl. /health is
# unauthenticated in every AUTH_MODE (see api/main.py). The Gradio service
# reuses this image with another CMD and nothing on :8000, so
# docker-compose.yml disables the check there.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"]

CMD ["sme-chatbot-api"]
