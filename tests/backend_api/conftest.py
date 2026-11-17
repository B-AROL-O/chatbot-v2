"""Shared fixtures for ``tests/backend_api`` — FastAPI ``TestClient`` against
the backend from ``src/sme_chatbot/api`` with ``stream_chat`` monkeypatched
to a fake generator (no network, no LLM, no ChromaDB).

Contract under test: ``docs/specs/2026-07-02-backend-split-and-docsvc-ingestion-plan.md``
§4.1 (chat API / SSE), §4.2 (auth modes).

Every test is isolated from the developer's real ``.env`` (``sme_chatbot.config``
loads it at import time) by the autouse ``_clean_config`` fixture below, per the
repo convention that new suites run clean in CI regardless of local secrets.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator

import pytest
from fastapi.testclient import TestClient

import sme_chatbot.api.auth as auth_mod
import sme_chatbot.api.main as main_mod
from sme_chatbot import config
from sme_chatbot.core.events import ChatEvent


@pytest.fixture(autouse=True)
def _clean_config(monkeypatch):
    """Pin every config value the API layer reads, independent of the repo's
    real ``.env`` (which sets a live ``LLM_BASE_URL`` for local dev use)."""
    monkeypatch.setattr(config, "AUTH_MODE", "dev", raising=False)
    monkeypatch.setattr(config, "LLM_BASE_URL", "http://fake-llm.invalid/v1", raising=False)
    monkeypatch.setattr(config, "LLM_API_TOKEN", "test-token", raising=False)
    monkeypatch.setattr(config, "LLM_MODEL", "test-model", raising=False)
    # Unset, whatever the developer's .env says: strict stream_chat fakes below do not take them.
    monkeypatch.setattr(config, "LLM_TEMPERATURE", None, raising=False)
    monkeypatch.setattr(config, "LLM_SEED", None, raising=False)
    monkeypatch.setattr(config, "LLM_MAX_TOKENS", None, raising=False)
    monkeypatch.setattr(config, "LLM_REASONING_EFFORT", None, raising=False)
    monkeypatch.setattr(config, "LLM_RETRY_REASONING_EFFORT", None, raising=False)
    monkeypatch.setattr(config, "LLM_API", "chat_completions", raising=False)
    monkeypatch.setattr(config, "JWT_TENANT_ID", "test-tenant-id", raising=False)
    monkeypatch.setattr(config, "JWT_AUDIENCE", "test-audience", raising=False)
    monkeypatch.setattr(config, "LANGFUSE_ENABLED", False, raising=False)
    monkeypatch.setattr(config, "LANGFUSE_BASE_URL", "", raising=False)
    monkeypatch.setattr(config, "LANGFUSE_PUBLIC_KEY", "", raising=False)
    monkeypatch.setattr(config, "LANGFUSE_SECRET_KEY", "", raising=False)
    monkeypatch.setattr(config, "APP_VERSION", "test-version", raising=False)
    # The OpenAI client is a lazily-built module-level singleton in main.py;
    # reset it so every test rebuilds it against the patched config above
    # rather than reusing whatever the first test in the session produced.
    monkeypatch.setattr(main_mod, "_client", None, raising=False)
    yield


@pytest.fixture()
def block_jwks(monkeypatch):
    """Fail loudly if a test path accidentally triggers a real JWKS fetch.

    Used by the jwt-mode 401 tests: the missing/malformed-token path must
    short-circuit before ``httpx.get`` is ever called.
    """

    def _boom(*_args, **_kwargs):
        raise AssertionError("unexpected network call to fetch JWKS in a unit test")

    monkeypatch.setattr(auth_mod.httpx, "get", _boom)


@pytest.fixture()
def set_stream_chat(monkeypatch):
    """Return a setter that monkeypatches the ``stream_chat`` symbol imported
    into ``sme_chatbot.api.main``'s module namespace (what the endpoint
    actually calls)."""

    def _set(fake: Callable[..., Iterator[ChatEvent]]) -> None:
        monkeypatch.setattr(main_mod, "stream_chat", fake)

    return _set


@pytest.fixture()
def app_client(_clean_config) -> TestClient:
    # Explicit dependency (rather than relying on autouse-fixture ordering):
    # `create_app()` reads `config.APP_VERSION` synchronously into the
    # FastAPI app's `version=` at construction time, so `_clean_config` must
    # have already run before `create_app()` is called here.
    app = main_mod.create_app()
    return TestClient(app)


def parse_sse(raw: str) -> list[tuple[str, dict]]:
    """Parse a raw ``text/event-stream`` body into ``(event, json_data)`` pairs."""
    events: list[tuple[str, dict]] = []
    cur_event: str | None = None
    cur_data: str | None = None
    for line in raw.splitlines():
        if line.startswith("event:"):
            cur_event = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            cur_data = line.split(":", 1)[1].strip()
        elif line == "" and cur_event is not None:
            events.append((cur_event, json.loads(cur_data) if cur_data else {}))
            cur_event, cur_data = None, None
    if cur_event is not None:
        events.append((cur_event, json.loads(cur_data) if cur_data else {}))
    return events
