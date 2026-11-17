"""Unit tests for the per-request ``documents`` domain (plan WP2a)."""

from __future__ import annotations

import json

from sme_chatbot.core import domains
from sme_chatbot.core.events import RequestIdentity


def test_documents_domain_states_the_resolved_serial_and_defaults_it(monkeypatch):
    monkeypatch.setattr(domains, "resolve_serial", lambda message, history=None: "90001")
    calls = []
    from sme_chatbot.tools import docsvc_rag

    monkeypatch.setitem(
        docsvc_rag._HANDLERS, "search_documents", lambda identity=None, **kw: calls.append((identity, kw)) or {}
    )
    identity = RequestIdentity(user_id="alice")
    domain = domains.documents_domain_for(identity, "Quanto pesa la 90001?")

    assert "matricola 90001" in domain.system_prompt
    assert domain.system_prompt.startswith(domains.DOCUMENTS_SYSTEM_PROMPT)
    json.loads(domain.dispatch("search_documents", {"query": "peso"}))
    assert calls == [(identity, {"query": "peso", "serial": "90001"})]


def test_documents_domain_without_a_serial_says_so(monkeypatch):
    monkeypatch.setattr(domains, "resolve_serial", lambda message, history=None: None)
    domain = domains.documents_domain_for(RequestIdentity(), "Qual è la coppia di una vite M12?")
    assert domain.system_prompt.startswith(domains.DOCUMENTS_SYSTEM_PROMPT)
    assert "No machine serial number" in domain.system_prompt
    assert domains.DOMAINS["documents"].system_prompt == domains.DOCUMENTS_SYSTEM_PROMPT


def test_documents_domain_defines_a_pre_search_only_with_a_serial(monkeypatch):
    # Without a machine in scope the prompt says "do not search yet"
    monkeypatch.setattr(domains, "resolve_serial", lambda message, history=None: "90001")
    assert domains.documents_domain_for(RequestIdentity(), "Quanto pesa la 90001?").pre_search == (
        "search_documents",
        {"query": "Quanto pesa la 90001?"},
    )
    monkeypatch.setattr(domains, "resolve_serial", lambda message, history=None: None)
    assert domains.documents_domain_for(RequestIdentity(), "Quanto pesa?").pre_search is None
