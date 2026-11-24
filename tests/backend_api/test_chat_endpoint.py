"""FastAPI ``TestClient`` suite for ``POST /chat/message``, ``GET /health``,
and ``GET /config``.

Contract: ``docs/specs/2026-07-02-backend-split-and-docsvc-ingestion-plan.md``
§4.1 (chat API / SSE event table) and §4.2 (``RequestIdentity`` / auth modes).

No network, no LLM: ``sme_chatbot.core.chat.stream_chat`` is monkeypatched to
scripted fake generators (see ``conftest.py::set_stream_chat``), and
``sme_chatbot.config`` is pinned by the autouse ``_clean_config`` fixture so
these tests do not depend on the developer's real ``.env``.
"""

from __future__ import annotations

import threading

import pytest
from conftest import parse_sse
from fastapi.testclient import TestClient

import sme_chatbot.api.main as main_mod
from sme_chatbot import config
from sme_chatbot.core.events import (
    Citation,
    DoneEvent,
    ErrorEvent,
    TokenEvent,
    ToolCallEvent,
    ToolResultEvent,
)

# ---------------------------------------------------------------------------
# Fake stream_chat generators (all match stream_chat's call signature).
# ---------------------------------------------------------------------------


def _fake_full_turn(client, model, domain, message, history, identity, max_tool_rounds=5):
    """Text -> one tool round -> done, per the WP3/WP8 reference scenario."""
    yield TokenEvent(text="Hello ")
    yield TokenEvent(text="world")
    yield ToolCallEvent(name="search_manuals", arguments={"query": "x"})
    yield ToolResultEvent(
        name="search_manuals",
        ok=True,
        citations=[Citation(source="manual.pdf", page=3, document_id=None, distance=0.12)],
    )
    yield DoneEvent(
        content="Hello world",
        citations=[Citation(source="manual.pdf", page=3, document_id=None, distance=0.12)],
        usage={"rounds": 1},
    )


def _fake_error_event(client, model, domain, message, history, identity, max_tool_rounds=5):
    """stream_chat reports failure the "clean" way: a terminal ErrorEvent."""
    yield TokenEvent(text="partial")
    yield ErrorEvent(message="boom")


def _fake_raises(client, model, domain, message, history, identity, max_tool_rounds=5):
    """stream_chat (or code around it) blows up instead of yielding ErrorEvent."""
    yield TokenEvent(text="partial")
    raise RuntimeError("kaboom")


def _make_identity_capture(store: dict):
    def _fake(client, model, domain, message, history, identity, max_tool_rounds=5):
        store["identity"] = identity
        yield DoneEvent(content="ok", citations=[], usage={"rounds": 0})

    return _fake


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


def test_health_returns_200_and_ok_status(app_client):
    resp = app_client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_health_domains_include_sample_cdo_sample_manuals_documents(app_client):
    resp = app_client.get("/health")
    body = resp.json()
    assert isinstance(body["domains"], list)
    assert {"sample_cdo", "sample_manuals", "documents"} <= set(body["domains"])


def test_health_requires_no_auth_even_in_jwt_mode(app_client, monkeypatch, block_jwks):
    # §4.1: "No auth" for /health, regardless of AUTH_MODE.
    monkeypatch.setattr(config, "AUTH_MODE", "jwt")
    resp = app_client.get("/health")
    assert resp.status_code == 200


def test_health_reports_app_version(app_client):
    """Regression: /health must expose the deployed version."""
    resp = app_client.get("/health")
    assert resp.json()["version"] == "test-version"


def test_health_version_defaults_to_dev_when_app_version_unset(monkeypatch):
    """No APP_VERSION build arg (local/manual build) -> "dev", not a crash."""
    monkeypatch.setattr(config, "APP_VERSION", "dev", raising=False)
    app = main_mod.create_app()
    resp = TestClient(app).get("/health")
    assert resp.json()["version"] == "dev"


def test_openapi_info_version_matches_app_version(app_client):
    """The stale-forever "0.1.0" this issue reported is gone from info.version."""
    resp = app_client.get("/openapi.json")
    assert resp.json()["info"]["version"] == "test-version"


def test_health_reports_uptime_seconds(app_client):
    resp = app_client.get("/health")
    assert resp.json()["uptime_seconds"] >= 0


# ---------------------------------------------------------------------------
# GET /config
# ---------------------------------------------------------------------------


def test_config_returns_static_backend_settings(app_client):
    resp = app_client.get("/config")
    assert resp.status_code == 200
    body = resp.json()
    assert body["model"] == "test-model"
    assert body["llm_base_url"] == "http://fake-llm.invalid/v1"
    assert body["auth_mode"] == "dev"
    assert body["version"] == "test-version"
    assert {"sample_cdo", "sample_manuals", "documents"} <= set(body["domains"])


def test_config_never_includes_secrets(app_client):
    resp = app_client.get("/config")
    body_text = resp.text
    assert "test-token" not in body_text  # LLM_API_TOKEN must never leak


def test_config_open_in_dev_mode(app_client):
    resp = app_client.get("/config")
    assert resp.status_code == 200


def test_config_requires_auth_in_jwt_mode(app_client, monkeypatch, block_jwks):
    monkeypatch.setattr(config, "AUTH_MODE", "jwt")
    resp = app_client.get("/config")
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Request validation (422s)
# ---------------------------------------------------------------------------


def test_unknown_domain_returns_422(app_client):
    resp = app_client.post("/chat/message", json={"domain": "bogus", "message": "hi"})
    assert resp.status_code == 422


def test_malformed_history_role_returns_422(app_client):
    resp = app_client.post(
        "/chat/message",
        json={
            "domain": "sample_manuals",
            "message": "hi",
            "history": [{"role": "system", "content": "x"}],
        },
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# stream: false -> JSON ChatResponse
# ---------------------------------------------------------------------------


def test_stream_false_returns_chat_response_shape(app_client, set_stream_chat):
    set_stream_chat(_fake_full_turn)
    resp = app_client.post(
        "/chat/message", json={"domain": "sample_manuals", "message": "hi", "stream": False}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["content"] == "Hello world"
    assert body["usage"]["rounds"] == 1
    assert isinstance(body["usage"]["timings"]["setup_ms"], int)
    assert isinstance(body["citations"], list) and len(body["citations"]) == 1
    citation = body["citations"][0]
    assert citation["source"] == "manual.pdf"
    assert citation["page"] == 3
    assert citation["document_id"] is None


# ---------------------------------------------------------------------------
# stream: true -> SSE
# ---------------------------------------------------------------------------


def test_stream_true_event_sequence(app_client, set_stream_chat):
    set_stream_chat(_fake_full_turn)
    with app_client.stream(
        "POST", "/chat/message", json={"domain": "sample_manuals", "message": "hi", "stream": True}
    ) as resp:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        raw = "".join(resp.iter_text())

    events = parse_sse(raw)
    names = [name for name, _ in events]
    assert names == ["token", "token", "tool_call", "tool_result", "done"]


def test_stream_true_token_events_carry_text(app_client, set_stream_chat):
    set_stream_chat(_fake_full_turn)
    with app_client.stream(
        "POST", "/chat/message", json={"domain": "sample_manuals", "message": "hi", "stream": True}
    ) as resp:
        raw = "".join(resp.iter_text())

    events = parse_sse(raw)
    token_payloads = [data for name, data in events if name == "token"]
    assert [t["text"] for t in token_payloads] == ["Hello ", "world"]


def test_stream_true_tool_call_and_result_payloads(app_client, set_stream_chat):
    set_stream_chat(_fake_full_turn)
    with app_client.stream(
        "POST", "/chat/message", json={"domain": "sample_manuals", "message": "hi", "stream": True}
    ) as resp:
        raw = "".join(resp.iter_text())

    events = dict(parse_sse(raw))
    assert events["tool_call"] == {"name": "search_manuals", "arguments": {"query": "x"}}
    assert events["tool_result"]["name"] == "search_manuals"
    assert events["tool_result"]["ok"] is True
    assert events["tool_result"]["citations"][0]["source"] == "manual.pdf"


def test_stream_true_done_payload_has_content_citations_usage(app_client, set_stream_chat):
    set_stream_chat(_fake_full_turn)
    with app_client.stream(
        "POST", "/chat/message", json={"domain": "sample_manuals", "message": "hi", "stream": True}
    ) as resp:
        raw = "".join(resp.iter_text())

    events = dict(parse_sse(raw))
    done = events["done"]
    assert done["content"] == "Hello world"
    assert done["usage"]["rounds"] == 1
    assert isinstance(done["usage"]["timings"]["setup_ms"], int)
    assert done["citations"][0]["source"] == "manual.pdf"
    assert done["citations"][0]["page"] == 3


# ---------------------------------------------------------------------------
# Dev-mode identity propagation
# ---------------------------------------------------------------------------


def test_dev_mode_x_user_id_header_propagates_to_identity(app_client, set_stream_chat):
    captured: dict = {}
    set_stream_chat(_make_identity_capture(captured))

    resp = app_client.post(
        "/chat/message",
        json={"domain": "sample_manuals", "message": "hi", "stream": False},
        headers={"X-User-Id": "alice"},
    )
    assert resp.status_code == 200
    assert captured["identity"].user_id == "alice"
    assert captured["identity"].permitted_document_ids is None
    assert captured["identity"].scopes == {}


def test_dev_mode_missing_x_user_id_header_yields_none(app_client, set_stream_chat):
    captured: dict = {}
    set_stream_chat(_make_identity_capture(captured))

    resp = app_client.post(
        "/chat/message", json={"domain": "sample_manuals", "message": "hi", "stream": False}
    )
    assert resp.status_code == 200
    assert captured["identity"].user_id is None


# ---------------------------------------------------------------------------
# jwt-mode 401 (no token) — must not perform a JWKS network fetch
# ---------------------------------------------------------------------------


def test_jwt_mode_401_without_token_and_no_jwks_fetch(app_client, monkeypatch, block_jwks):
    monkeypatch.setattr(config, "AUTH_MODE", "jwt")

    resp = app_client.post(
        "/chat/message", json={"domain": "sample_manuals", "message": "hi", "stream": False}
    )
    assert resp.status_code == 401


def test_jwt_mode_401_with_non_bearer_authorization_header(app_client, monkeypatch, block_jwks):
    monkeypatch.setattr(config, "AUTH_MODE", "jwt")

    resp = app_client.post(
        "/chat/message",
        json={"domain": "sample_manuals", "message": "hi", "stream": False},
        headers={"Authorization": "Token abc123"},
    )
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_non_stream_error_event_returns_500(app_client, set_stream_chat):
    set_stream_chat(_fake_error_event)
    resp = app_client.post(
        "/chat/message", json={"domain": "sample_manuals", "message": "hi", "stream": False}
    )
    assert resp.status_code == 500
    assert "boom" in resp.json()["detail"]


def test_non_stream_generator_exception_returns_500(app_client, set_stream_chat):
    set_stream_chat(_fake_raises)
    resp = app_client.post(
        "/chat/message", json={"domain": "sample_manuals", "message": "hi", "stream": False}
    )
    assert resp.status_code == 500
    assert "kaboom" in resp.json()["detail"]


def test_stream_error_event_emits_terminal_error_sse(app_client, set_stream_chat):
    set_stream_chat(_fake_error_event)
    with app_client.stream(
        "POST", "/chat/message", json={"domain": "sample_manuals", "message": "hi", "stream": True}
    ) as resp:
        assert resp.status_code == 200
        raw = "".join(resp.iter_text())

    events = parse_sse(raw)
    assert events[-1][0] == "error"
    assert events[-1][1]["message"] == "boom"


def test_stream_generator_exception_emits_terminal_error_sse(app_client, set_stream_chat):
    set_stream_chat(_fake_raises)
    with app_client.stream(
        "POST", "/chat/message", json={"domain": "sample_manuals", "message": "hi", "stream": True}
    ) as resp:
        assert resp.status_code == 200
        raw = "".join(resp.iter_text())

    events = parse_sse(raw)
    assert events[-1][0] == "error"
    assert "kaboom" in events[-1][1]["message"]


# ---------------------------------------------------------------------------
# documents domain is built per request (identity + matricola), plan WP2a
# ---------------------------------------------------------------------------


def test_documents_domain_is_built_per_request_with_identity_and_message(app_client, set_stream_chat, monkeypatch):
    from dataclasses import replace

    from sme_chatbot.core.domains import DOMAINS

    seen: dict = {}

    def _fake_domain_for(identity, message, history):
        seen.update(identity=identity, message=message, history=history)
        return replace(DOMAINS["documents"], system_prompt="scoped prompt")

    def _capture(**kwargs):
        seen["prompt"] = kwargs["domain"].system_prompt
        yield from _fake_full_turn(**kwargs)

    monkeypatch.setattr(main_mod, "documents_domain_for", _fake_domain_for)
    set_stream_chat(_capture)
    resp = app_client.post(
        "/chat/message",
        json={"domain": "documents", "message": "Quanto pesa la 90001?", "stream": False},
        headers={"X-User-Id": "alice"},
    )
    assert resp.status_code == 200
    assert seen["identity"].user_id == "alice"
    assert seen["message"] == "Quanto pesa la 90001?"
    assert seen["prompt"] == "scoped prompt"


# ---------------------------------------------------------------------------
# Per-stage timings: setup_ms merged next to stream_chat's timings
# ---------------------------------------------------------------------------


def _fake_turn_with_timings(client, model, domain, message, history, identity, max_tool_rounds=5):
    yield DoneEvent(
        content="ok",
        citations=[],
        usage={
            "rounds": 2,
            "timings": {"total_ms": 900, "ttft_ms": 800, "llm_ms": 650, "tool_ms": 250, "tool_wall_ms": 180},
        },
    )


def _slow_documents_domain(monkeypatch):
    import time
    from dataclasses import replace

    from sme_chatbot.core.domains import DOMAINS

    def _slow_domain_for(identity, message, history):
        time.sleep(0.03)  # stands in for resolve_serial's Chroma lookups
        return replace(DOMAINS["documents"], system_prompt="scoped prompt")

    monkeypatch.setattr(main_mod, "documents_domain_for", _slow_domain_for)


def test_stream_false_usage_merges_setup_ms_with_loop_timings(app_client, set_stream_chat, monkeypatch):
    _slow_documents_domain(monkeypatch)
    set_stream_chat(_fake_turn_with_timings)

    resp = app_client.post(
        "/chat/message", json={"domain": "documents", "message": "hi", "stream": False}
    )

    timings = resp.json()["usage"]["timings"]
    assert timings["setup_ms"] >= 30
    assert (timings["total_ms"], timings["ttft_ms"], timings["llm_ms"], timings["tool_ms"]) == (900, 800, 650, 250)


def test_stream_true_done_usage_merges_setup_ms_with_loop_timings(app_client, set_stream_chat, monkeypatch):
    _slow_documents_domain(monkeypatch)
    set_stream_chat(_fake_turn_with_timings)

    with app_client.stream(
        "POST", "/chat/message", json={"domain": "documents", "message": "hi", "stream": True}
    ) as resp:
        raw = "".join(resp.iter_text())

    timings = dict(parse_sse(raw))["done"]["usage"]["timings"]
    assert timings["setup_ms"] >= 30
    assert timings["total_ms"] == 900


def test_request_log_carries_durations_but_not_the_message(app_client, set_stream_chat, caplog):
    set_stream_chat(_fake_turn_with_timings)

    with caplog.at_level("INFO", logger=main_mod.logger.name):
        app_client.post(
            "/chat/message",
            json={"domain": "sample_manuals", "message": "SEGRETO-DEL-CLIENTE", "stream": False},
        )

    line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("chat/message"))
    assert "rounds=2" in line
    assert "round_limit_hit=False" in line
    assert "total_ms=900" in line
    assert "ttft_ms=800" in line
    assert "llm_ms=650" in line
    assert "tool_ms=250" in line
    assert "tool_wall_ms=180" in line
    assert "setup_ms=" in line
    assert "SEGRETO-DEL-CLIENTE" not in line


def test_request_log_flags_a_turn_that_hit_the_round_limit(app_client, set_stream_chat, caplog):
    """The per-request line (shipped to Loki) is where the cap's hit rate gets counted."""

    def _fake(client, model, domain, message, history, identity, max_tool_rounds=5):
        yield DoneEvent(content="ok", citations=[], usage={"rounds": 6, "round_limit_hit": True, "timings": {}})

    set_stream_chat(_fake)

    with caplog.at_level("INFO", logger=main_mod.logger.name):
        app_client.post("/chat/message", json={"domain": "sample_manuals", "message": "hi", "stream": False})

    line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("chat/message"))
    assert "round_limit_hit=True" in line


# ---------------------------------------------------------------------------
# Sampling settings: LLM_TEMPERATURE / LLM_SEED reach stream_chat and /config
# ---------------------------------------------------------------------------


def _capture_sampling(store: dict):
    def _fake(client, model, domain, message, history, identity, max_tool_rounds=5, **sampling):
        store["sampling"] = sampling
        yield DoneEvent(content="ok", citations=[], usage={"rounds": 1})

    return _fake


def test_unset_sampling_passes_nothing_to_stream_chat(app_client, set_stream_chat):
    seen: dict = {}
    set_stream_chat(_capture_sampling(seen))

    app_client.post("/chat/message", json={"domain": "sample_manuals", "message": "hi", "stream": False})

    assert seen["sampling"] == {}


def test_configured_sampling_reaches_stream_chat(app_client, set_stream_chat, monkeypatch):
    monkeypatch.setattr(config, "LLM_TEMPERATURE", 0.0)
    monkeypatch.setattr(config, "LLM_SEED", 42)
    seen: dict = {}
    set_stream_chat(_capture_sampling(seen))

    app_client.post("/chat/message", json={"domain": "sample_manuals", "message": "hi", "stream": False})

    assert seen["sampling"] == {"temperature": 0.0, "seed": 42}


def test_config_reports_the_sampling_settings_so_eval_runs_record_them(app_client, monkeypatch):
    assert app_client.get("/config").json()["temperature"] is None
    assert app_client.get("/config").json()["seed"] is None

    monkeypatch.setattr(config, "LLM_TEMPERATURE", 0.0)
    monkeypatch.setattr(config, "LLM_SEED", 42)

    body = app_client.get("/config").json()
    assert body["temperature"] == 0.0
    assert body["seed"] == 42


def test_max_tokens_reaches_stream_chat_and_config(app_client, set_stream_chat, monkeypatch):
    assert app_client.get("/config").json()["max_tokens"] is None
    monkeypatch.setattr(config, "LLM_MAX_TOKENS", 4096)
    seen: dict = {}
    set_stream_chat(_capture_sampling(seen))

    app_client.post("/chat/message", json={"domain": "sample_manuals", "message": "hi", "stream": False})

    assert seen["sampling"] == {"max_tokens": 4096}
    assert app_client.get("/config").json()["max_tokens"] == 4096


def test_reasoning_effort_reaches_stream_chat_and_config(app_client, set_stream_chat, monkeypatch):
    assert app_client.get("/config").json()["reasoning_effort"] is None
    monkeypatch.setattr(config, "LLM_REASONING_EFFORT", "none")
    seen: dict = {}
    set_stream_chat(_capture_sampling(seen))

    app_client.post("/chat/message", json={"domain": "sample_manuals", "message": "hi", "stream": False})

    assert seen["sampling"] == {"reasoning_effort": "none"}
    assert app_client.get("/config").json()["reasoning_effort"] == "none"


def test_retry_reasoning_effort_reaches_stream_chat_and_config(app_client, set_stream_chat, monkeypatch):
    assert app_client.get("/config").json()["retry_reasoning_effort"] is None
    monkeypatch.setattr(config, "LLM_RETRY_REASONING_EFFORT", "none")
    seen: dict = {}
    set_stream_chat(_capture_sampling(seen))

    app_client.post("/chat/message", json={"domain": "sample_manuals", "message": "hi", "stream": False})

    assert seen["sampling"] == {"retry_reasoning_effort": "none"}
    assert app_client.get("/config").json()["retry_reasoning_effort"] == "none"


def test_pre_search_reaches_stream_chat_and_config_only_when_enabled(app_client, set_stream_chat, monkeypatch):
    seen: dict = {}
    set_stream_chat(_capture_sampling(seen))
    app_client.post("/chat/message", json={"domain": "sample_manuals", "message": "hi", "stream": False})
    assert "pre_search" not in seen["sampling"]
    assert app_client.get("/config").json()["pre_search"] is False

    monkeypatch.setattr(config, "CHAT_PRE_SEARCH", True)
    app_client.post("/chat/message", json={"domain": "sample_manuals", "message": "hi", "stream": False})
    assert seen["sampling"] == {"pre_search": True}
    assert app_client.get("/config").json()["pre_search"] is True


def test_tool_concurrency_reaches_stream_chat_only_when_not_the_default(app_client, set_stream_chat, monkeypatch):
    seen: dict = {}
    set_stream_chat(_capture_sampling(seen))
    app_client.post("/chat/message", json={"domain": "sample_manuals", "message": "hi", "stream": False})
    assert "tool_concurrency" not in seen["sampling"]
    assert app_client.get("/config").json()["tool_concurrency"] == 4

    monkeypatch.setattr(config, "CHAT_TOOL_CONCURRENCY", 1)
    app_client.post("/chat/message", json={"domain": "sample_manuals", "message": "hi", "stream": False})
    assert seen["sampling"] == {"tool_concurrency": 1}
    assert app_client.get("/config").json()["tool_concurrency"] == 1


def test_pre_search_results_sets_n_results_on_the_pre_search_only(app_client, set_stream_chat, monkeypatch):
    # Fewer pre-search results shorten the first LLM call's prompt
    from dataclasses import replace

    from sme_chatbot.core.domains import DOMAINS

    def _fake_domain_for(identity, message, history):
        return replace(DOMAINS["documents"], pre_search=("search_documents", {"query": message}))

    seen: dict = {}

    def _capture(**kwargs):
        seen["pre_search"] = kwargs["domain"].pre_search
        yield from _fake_full_turn(**kwargs)

    monkeypatch.setattr(main_mod, "documents_domain_for", _fake_domain_for)
    set_stream_chat(_capture)
    body = {"domain": "documents", "message": "Quanto pesa la 90001?", "stream": False}

    app_client.post("/chat/message", json=body)
    assert seen["pre_search"] == ("search_documents", {"query": "Quanto pesa la 90001?"})
    assert app_client.get("/config").json()["pre_search_results"] is None

    monkeypatch.setattr(config, "CHAT_PRE_SEARCH_RESULTS", 5)
    app_client.post("/chat/message", json=body)
    assert seen["pre_search"] == ("search_documents", {"query": "Quanto pesa la 90001?", "n_results": 5})
    assert app_client.get("/config").json()["pre_search_results"] == 5


# ---------------------------------------------------------------------------
# Startup warm-up
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["on", "full"])
def test_startup_runs_the_warm_up_in_the_background(monkeypatch, mode):
    started = threading.Event()
    seen: list[str] = []
    monkeypatch.setattr(config, "RETRIEVAL_WARMUP", mode)
    monkeypatch.setattr(main_mod, "warm_up", lambda m: seen.append(m) or started.set())

    with TestClient(main_mod.create_app()) as client:
        assert client.get("/health").status_code == 200
        assert started.wait(timeout=5)

    assert seen == [mode]


def test_startup_skips_the_warm_up_when_off(monkeypatch):
    monkeypatch.setattr(config, "RETRIEVAL_WARMUP", "off")
    monkeypatch.setattr(main_mod, "warm_up", lambda m: pytest.fail("warm-up ran"))

    with TestClient(main_mod.create_app()) as client:
        assert client.get("/health").status_code == 200


def test_llm_api_reaches_stream_chat_and_config_only_when_not_the_default(app_client, set_stream_chat, monkeypatch):
    seen: dict = {}
    set_stream_chat(_capture_sampling(seen))
    assert app_client.get("/config").json()["llm_api"] == "chat_completions"

    app_client.post("/chat/message", json={"domain": "sample_manuals", "message": "hi", "stream": False})
    assert seen["sampling"] == {}

    monkeypatch.setattr(config, "LLM_API", "responses")
    app_client.post("/chat/message", json={"domain": "sample_manuals", "message": "hi", "stream": False})
    assert seen["sampling"] == {"llm_api": "responses"}
    assert app_client.get("/config").json()["llm_api"] == "responses"
