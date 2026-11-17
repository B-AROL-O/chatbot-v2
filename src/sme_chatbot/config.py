"""Env-driven settings — single source of truth for the new backend stack.

Loads every variable listed in the plan's shared-contracts table
(``docs/specs/2026-07-02-backend-split-and-docsvc-ingestion-plan.md`` §4.5)
via plain ``os.environ`` (mirroring existing scripts) plus ``python-dotenv``
so a local ``.env`` is picked up the same way it is for the root scripts.

Kept deliberately dependency-light: module-level constants, no pydantic.
Values are read once at import time; call sites should import the module
(``from sme_chatbot import config``) rather than the individual constants
so monkeypatching ``os.environ`` in tests before import still works via
``importlib.reload`` if ever needed.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

# Load .env (if present) before reading any variable below. Safe to call
# repeatedly / when no .env file exists (no-op).
load_dotenv()


def _get_bool(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


def _get_optional(name: str, cast):
    """An optional numeric env var: unset or blank -> ``None``, junk -> a startup error.

    ``0`` is a real value (temperature 0), so callers must test ``is not None``,
    never truthiness.
    """
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        return cast(raw)
    except ValueError as exc:
        raise ValueError(f"{name}={raw!r} is not a valid {cast.__name__}") from exc


def _get_choice(name: str, choices: tuple[str, ...]) -> str:
    """One of ``choices`` (case-insensitive); unset or blank -> the first, junk -> a startup error."""
    raw = os.environ.get(name, "").strip().lower() or choices[0]
    if raw not in choices:
        raise ValueError(f"{name}={raw!r} must be one of {', '.join(choices)}")
    return raw


# --- LLM / backend -----------------------------------------------------

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "")
LLM_API_TOKEN = os.environ.get("LLM_API_TOKEN", "")
# Default: gpt-6-luna, the v0.10.0 default (was gpt-6-sol).
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-6-luna")
# Sampling. Unset (the default) sends nothing, so the server's own defaults apply
# (for qwen3.6 on Ollama: temperature 1, no seed, i.e. different answers on every
# run). Set both for reproducible eval runs.
LLM_TEMPERATURE = _get_optional("LLM_TEMPERATURE", float)
LLM_SEED = _get_optional("LLM_SEED", int)
# Upper bound on the tokens of ONE LLM call (reasoning included for thinking models).
# Unset sends nothing. A thinking model with no bound can generate for many minutes
# without ever answering (observed at temperature 0); a call cut off by this limit
# ends with finish "length" in usage.timings instead.
LLM_MAX_TOKENS = _get_optional("LLM_MAX_TOKENS", int)
# Reasoning control, passed through as the OpenAI `reasoning_effort` parameter. Unset sends
# nothing (the model reasons as usual). On Ollama (qwen3.8:27b, 0.40.4) only "none" has an
# effect: it turns reasoning off ("low"/"high" behave like the default, and `think: false`
# in the /v1 body is ignored). Azure gpt-5 takes "minimal", "low", "medium" or "high".
LLM_REASONING_EFFORT = _get_optional("LLM_REASONING_EFFORT", str)
# Reasoning control for the ONE retry made when a call ends on max_tokens before producing
# any visible text (reasoning can use the whole budget). Unset retries with the settings
# above, which for a thinking model will probably reason at length again; "none" on Ollama
# makes the retry answer directly.
LLM_RETRY_REASONING_EFFORT = _get_optional("LLM_RETRY_REASONING_EFFORT", str)
# Which OpenAI API the chat loop calls: "chat_completions" (the default,
# /v1/chat/completions, what Ollama and other OpenAI-compatible servers implement) or
# "responses" (/v1/responses). Azure has refused function tools together with
# reasoning_effort on /v1/chat/completions for gpt-6 models; "responses" keeps both.
# On "responses" LLM_SEED is not sent: that API has no seed.
LLM_API = _get_choice("LLM_API", ("chat_completions", "responses"))
# Search on the user's question before the first LLM call (documents domain, only when a
# matricola is in scope), so the model can answer in its first call instead of spending one
# to ask for the search: each LLM call costs about 3 s of TTFT. Off by default.
CHAT_PRE_SEARCH = _get_bool("CHAT_PRE_SEARCH", "false")
# Results the pre-search returns. Unset = the search tool's own default (8). Fewer results make
# the first LLM call's prompt shorter (it spends seconds reading them) at some cost in recall.
CHAT_PRE_SEARCH_RESULTS = _get_optional("CHAT_PRE_SEARCH_RESULTS", int)

# --- Auth ----------------------------------------------------------------

AUTH_MODE = os.environ.get("AUTH_MODE", "dev")
JWT_TENANT_ID = os.environ.get("JWT_TENANT_ID", "")
JWT_AUDIENCE = os.environ.get("JWT_AUDIENCE", "")

# --- ChromaDB collections --------------------------------------------------

CHROMA_CDO_DIR = os.environ.get("CHROMA_CDO_DIR", "data/chroma_cdo_db")
CHROMA_MANUALS_DIR = os.environ.get("CHROMA_MANUALS_DIR", "data/chroma_manuals_db")
CHROMA_DOCSVC_DIR = os.environ.get("CHROMA_DOCSVC_DIR", "data/chroma_docsvc_db")

# Collection names inside those stores. Ingestion writes and the tools read the
# same name; scripts/rename_chroma_collection.py moves an existing store over.
CHROMA_CDO_COLLECTION = os.environ.get("CHROMA_CDO_COLLECTION", "cdo_docs")
CHROMA_MANUALS_COLLECTION = os.environ.get("CHROMA_MANUALS_COLLECTION", "manuals")
CHROMA_DOCSVC_COLLECTION = os.environ.get("CHROMA_DOCSVC_COLLECTION", "docsvc")

# --- Retrieval (documents domain) -------------------------------------------

# "hybrid" (dense + BM25 fused, the default) or "dense" (the pre-0.8 behaviour).
# Hybrid silently degrades to dense when the lexical index file is absent.
RETRIEVAL_MODE = os.environ.get("RETRIEVAL_MODE", "hybrid").strip().lower()
# SQLite FTS5 side index; build with `sme-ingest-docsvc --build-lexical-index`.
LEXICAL_INDEX_PATH = os.environ.get("LEXICAL_INDEX_PATH", "").strip() or os.path.join(
    CHROMA_DOCSVC_DIR, "lexical_fts.sqlite3"
)

# Load the retrieval stores at API startup instead of on the first request,
# in a background thread: "off" (the default), "on" (load the
# model, open the collections, run one query of each kind) or "full" ("on",
# then read the documents store's files once into the OS page cache, about
# 9 GB; only useful with the RAM to keep them there).
RETRIEVAL_WARMUP = _get_choice("RETRIEVAL_WARMUP", ("off", "on", "full"))

# --- Gradio frontend -------------------------------------------------------

CHATBOT_API_URL = os.environ.get("CHATBOT_API_URL", "http://localhost:8000")

# --- DocumentService (ingestion) -------------------------------------------

DOCSVC_BASE_URL = os.environ.get("DOCSVC_BASE_URL", "")
DOCSVC_TENANT_ID = os.environ.get("DOCSVC_TENANT_ID", "")
DOCSVC_AUTHORITY = os.environ.get("DOCSVC_AUTHORITY", "").strip() or (
    f"https://login.microsoftonline.com/{DOCSVC_TENANT_ID}" if DOCSVC_TENANT_ID else ""
)
DOCSVC_AUDIENCE = os.environ.get("DOCSVC_AUDIENCE", "")
DOCSVC_SCOPE = os.environ.get("DOCSVC_SCOPE", "").strip() or (
    f"{DOCSVC_AUDIENCE}/.default" if DOCSVC_AUDIENCE else ""
)
DOCSVC_READ_CLIENT_ID = os.environ.get("DOCSVC_READ_CLIENT_ID", "")
DOCSVC_READ_CLIENT_SECRET = os.environ.get("DOCSVC_READ_CLIENT_SECRET", "")
DOCSVC_VERIFY_SSL = _get_bool("DOCSVC_VERIFY_SSL", "true")
DOCSVC_USER_ID = os.environ.get("DOCSVC_USER_ID", "")

# --- Langfuse (LLM observability) ------------------------------------------

LANGFUSE_ENABLED = _get_bool("LANGFUSE_ENABLED", "false")
LANGFUSE_BASE_URL = os.environ.get("LANGFUSE_BASE_URL", "")
LANGFUSE_PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY", "")

# --- Eval (sme-chatbot-eval and friends) ------------------------------------

# The golden sets and run records are deployment data, kept in a separate
# private eval-data repo. The defaults expect it cloned at
# data/eval/ (gitignored, like the rest of data/); relative paths are
# taken from the current directory, as the CLIs' own path flags are.
EVAL_DATASETS_DIR = os.environ.get("EVAL_DATASETS_DIR", "").strip() or "data/eval/datasets"
EVAL_RESULTS_DIR = os.environ.get("EVAL_RESULTS_DIR", "").strip() or "data/eval/results"

# --- Logging ----------------------------------------------------------------

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

# --- Deployment --------------------------------------------------------------

# Baked in at image build time by the release workflow (--build-arg, from the
# release tag) — see Dockerfile and .github/workflows/release-chatbot.yml.
# Not meant to be set by hand; local/dev runs fall back to "dev".
APP_VERSION = os.environ.get("APP_VERSION", "dev")
