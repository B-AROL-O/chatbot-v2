"""Tests for ``_get_client()``'s ``LANGFUSE_ENABLED`` branch (``api/main.py``).

Uses a fake ``langfuse.openai`` module (injected into ``sys.modules``) rather
than the real Langfuse SDK: in the installed version (v4, OTel-based),
``langfuse.openai.OpenAI`` *is* ``openai.OpenAI`` — tracing is installed as a
process-wide monkeypatch of openai's internals at import time, not via a
distinct subclass — so asserting on ``type(client)`` against the real package
can't distinguish the two branches. The fake isolates the test to what
``_get_client()`` itself controls (which class it imports and constructs)
without patching real openai internals for the rest of the test session.
"""

from __future__ import annotations

import sys
import types

import sme_chatbot.api.main as main_mod
from sme_chatbot import config


class _FakeLangfuseOpenAI:
    def __init__(self, *, base_url, api_key):
        self.base_url = base_url
        self.api_key = api_key


def test_get_client_langfuse_disabled_returns_plain_openai(monkeypatch):
    monkeypatch.setattr(main_mod, "_client", None, raising=False)
    monkeypatch.setattr(config, "LANGFUSE_ENABLED", False, raising=False)

    client = main_mod._get_client()

    assert type(client) is main_mod.OpenAI


def test_get_client_langfuse_enabled_uses_langfuse_wrapper(monkeypatch):
    fake_module = types.ModuleType("langfuse.openai")
    fake_module.OpenAI = _FakeLangfuseOpenAI
    monkeypatch.setitem(sys.modules, "langfuse.openai", fake_module)

    monkeypatch.setattr(main_mod, "_client", None, raising=False)
    monkeypatch.setattr(config, "LANGFUSE_ENABLED", True, raising=False)
    monkeypatch.setattr(config, "LLM_BASE_URL", "http://fake-llm.invalid/v1", raising=False)
    monkeypatch.setattr(config, "LLM_API_TOKEN", "test-token", raising=False)

    client = main_mod._get_client()

    assert isinstance(client, _FakeLangfuseOpenAI)
    assert client.base_url == "http://fake-llm.invalid/v1"
    assert client.api_key == "test-token"
