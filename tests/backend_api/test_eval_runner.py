"""WP10 — the eval runner end-to-end against the **real** FastAPI app.

No network, no LLM, no ChromaDB: ``sme_chatbot.core.chat.stream_chat`` is
monkeypatched to a scripted fake through the existing ``set_stream_chat``
fixture, and every request still travels the real routing, validation,
``response_model`` and error handling of ``api/main.py`` — which is the point.
A test that stubbed the endpoint too would prove only that the runner can talk
to itself.

The runner's own ``httpx.Client`` reaches the app through a ``MockTransport``
that replays each request into a ``TestClient``. That keeps ``EvalClient``
exactly as production builds it (same retry loop, same headers, same JSON
handling) while giving the test the outbound headers to assert on.
"""

from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

import sme_chatbot.api.main as main_mod
from sme_chatbot.core.chat import _content_filter_message, _output_filter_message
from sme_chatbot.core.events import Citation, DoneEvent, ErrorEvent
from sme_chatbot.eval.client import BackendUnavailable, EvalClient
from sme_chatbot.eval.dataset import load_dataset
from sme_chatbot.eval.report import build_summary
from sme_chatbot.eval.runner import run_dataset
from sme_chatbot.eval.tracing import NullTracer, make_tracer

DATASET = "tests/fixtures/eval/fixture-alpha-domande.csv"


@pytest.fixture(scope="module")
def dataset():
    return load_dataset(DATASET)


@pytest.fixture()
def sent_requests() -> list[httpx.Request]:
    return []


@pytest.fixture()
def eval_client(app_client: TestClient, sent_requests):
    """An ``EvalClient`` whose requests land on the real app under test."""

    def handler(request: httpx.Request) -> httpx.Response:
        sent_requests.append(request)
        forwarded = {
            key: value
            for key, value in request.headers.items()
            if key.lower() not in ("host", "content-length", "transfer-encoding")
        }
        response = app_client.request(
            request.method,
            request.url.path,
            content=request.content,
            headers=forwarded,
        )
        return httpx.Response(
            response.status_code,
            content=response.content,
            headers={"content-type": response.headers.get("content-type", "application/json")},
        )

    client = EvalClient(
        "http://backend.test",
        transport=httpx.MockTransport(handler),
        max_retries=2,
        sleep=lambda _seconds: None,
    )
    yield client
    client.close()


def _answering_fake(answer: str, citations: list[Citation], rounds: int = 1):
    def fake(client, model, domain, message, history, identity, max_tool_rounds=5):
        yield DoneEvent(content=answer, citations=list(citations), usage={"rounds": rounds})

    return fake


GOLD = Citation(
    source="90001_manual_IT.pdf",
    page=18,
    document_id="doc-1",
    distance=0.11,
    filename="90001_manual_IT.pdf",
)


# ---------------------------------------------------------------------------
# /health preflight
# ---------------------------------------------------------------------------


def test_health_preflight_reads_version_and_domains(eval_client):
    health = eval_client.require_domain("documents")
    assert health["status"] == "ok"
    assert health["version"] == "test-version"
    assert "documents" in health["domains"]


def test_config_reads_the_backends_own_runtime_model(eval_client):
    """The run header must record what the *backend* is actually configured
    with, not whatever `LLM_MODEL` happens to be set to on the runner's own
    host — those two can differ (e.g. the runner's `.env` vs. a remote
    deployment's)."""
    backend_config = eval_client.config()
    assert backend_config["model"] == "test-model"
    assert backend_config["llm_base_url"] == "http://fake-llm.invalid/v1"
    assert backend_config["auth_mode"] == "dev"


def test_an_unreachable_backend_raises_backend_unavailable_for_config():
    def refuse(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with (
        EvalClient("http://down.test", transport=httpx.MockTransport(refuse)) as client,
        pytest.raises(BackendUnavailable, match="config failed"),
    ):
        client.config()


def test_an_unknown_domain_aborts_before_any_question_is_asked(eval_client, sent_requests):
    with pytest.raises(BackendUnavailable, match="not served"):
        eval_client.require_domain("nope")
    assert all(r.url.path == "/health" for r in sent_requests)


def test_an_unreachable_backend_raises_backend_unavailable():
    def refuse(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with (
        EvalClient("http://down.test", transport=httpx.MockTransport(refuse)) as client,
        pytest.raises(BackendUnavailable, match="health failed"),
    ):
        client.health()


# ---------------------------------------------------------------------------
# a full run
# ---------------------------------------------------------------------------


def test_full_run_over_the_real_endpoint(dataset, eval_client, set_stream_chat, sent_requests):
    set_stream_chat(_answering_fake("Il dato non è presente nel manuale.", [GOLD], rounds=2))

    results = run_dataset(list(dataset.items), eval_client, domain="documents")

    assert len(results) == dataset.n_items
    assert [r.run_key for r in results] == [i.id for i in dataset.items]
    assert all(r.error is None for r in results)
    assert all(r.rounds == 2 for r in results)
    assert all(r.scores is not None for r in results)

    posts = [r for r in sent_requests if r.url.path == "/chat/message"]
    assert len(posts) == dataset.n_items
    # Every question is its own session: the stateless API plus an empty
    # `history` is what makes the legend's "separate sessions" free.
    assert all(b'"history": []' in r.content or b'"history":[]' in r.content for r in posts)
    assert all(r.headers["x-user-id"] == "eval-runner" for r in posts)

    # The scripted answer is a refusal, so the three "Non rispondibile" items
    # are the ones it gets right.
    correct = {r.run_key for r in results if r.verdict == "S"}
    assert {"Q29", "Q30", "Q31"} & correct == {"Q29"}  # Q30/Q31 lack their rubric justification


def test_repeat_keys_each_ask_separately(dataset, eval_client, set_stream_chat):
    set_stream_chat(_answering_fake("850 kg.", [GOLD]))
    items = [i for i in dataset.items if i.id == "Q02"]

    results = run_dataset(items, eval_client, domain="documents", repeat=3)

    assert [r.run_key for r in results] == ["Q02#1", "Q02#2", "Q02#3"]
    assert all(r.verdict == "S" for r in results)


def test_concurrency_preserves_dataset_order(dataset, eval_client, set_stream_chat):
    set_stream_chat(_answering_fake("qualcosa", []))
    items = list(dataset.items)[:8]

    results = run_dataset(items, eval_client, domain="documents", concurrency=4)

    assert [r.run_key for r in results] == [i.id for i in items]


# ---------------------------------------------------------------------------
# failures
# ---------------------------------------------------------------------------


def test_a_backend_500_is_recorded_as_error_and_excluded_from_percentages(
    dataset, eval_client, set_stream_chat
):
    def failing(client, model, domain, message, history, identity, max_tool_rounds=5):
        yield ErrorEvent(message="retrieval exploded")

    set_stream_chat(failing)
    items = [i for i in dataset.items if i.id in ("Q01", "Q02")]

    results = run_dataset(items, eval_client, domain="documents")

    assert all(r.verdict == "ERROR" for r in results)
    assert all(r.status_code == 500 for r in results)
    assert all("retrieval exploded" in (r.error or "") for r in results)
    # A transport failure is never a wrong answer: no scores, no verdict, and
    # out of every denominator (plan §7 R5).
    assert all(r.scores is None for r in results)

    summary = build_summary(results, {"dataset": {}})
    assert summary["verdicts"]["ERROR"] == 2
    assert summary["verdicts"]["N"] == 0
    assert summary["scored_items"] == 0
    assert summary["pct_correct_of_scored"] is None


def test_one_failing_item_never_aborts_the_run(dataset, eval_client, set_stream_chat):
    def sometimes(client, model, domain, message, history, identity, max_tool_rounds=5):
        if "pesa" in message.lower():
            yield ErrorEvent(message="boom")
        else:
            yield DoneEvent(content="9000 bph.", citations=[GOLD], usage={"rounds": 1})

    set_stream_chat(sometimes)
    items = [i for i in dataset.items if i.id in ("Q01", "Q02", "Q04")]

    results = run_dataset(items, eval_client, domain="documents")

    assert [r.verdict for r in results] == ["S", "ERROR", "N"]


def test_429_is_retried_and_then_succeeds():
    """The LLM endpoint's rate limit must cost a delay, not a wall of false ``N`` verdicts."""
    attempts: list[int] = []

    def flaky(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) == 1:
            return httpx.Response(429, headers={"retry-after": "1"}, json={"detail": "slow down"})
        return httpx.Response(200, json={"content": "850 kg.", "citations": [], "usage": {"rounds": 1}})

    slept: list[float] = []
    with EvalClient(
        "http://backend.test",
        transport=httpx.MockTransport(flaky),
        max_retries=3,
        sleep=slept.append,
    ) as client:
        result = client.ask("documents", "Quanto pesa?")

    assert result.ok
    assert result.attempts == 2
    assert result.content == "850 kg."
    assert slept == [1.0]  # Retry-After honoured instead of the backoff curve


def test_a_non_retryable_status_is_not_retried():
    attempts: list[int] = []

    def unknown_domain(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        return httpx.Response(422, json={"detail": "Unknown domain: 'nope'"})

    with EvalClient("http://backend.test", transport=httpx.MockTransport(unknown_domain), max_retries=3) as client:
        result = client.ask("nope", "x")

    assert len(attempts) == 1
    assert result.attempts == 1
    assert "Unknown domain" in (result.error or "")


def test_retries_are_bounded_and_the_item_ends_as_an_error():
    attempts: list[int] = []

    def always_503(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        return httpx.Response(503, json={"detail": "unavailable"})

    with EvalClient(
        "http://backend.test",
        transport=httpx.MockTransport(always_503),
        max_retries=2,
        sleep=lambda _s: None,
    ) as client:
        result = client.ask("documents", "x")

    assert len(attempts) == 3  # 1 attempt + 2 retries
    assert result.attempts == 3
    assert not result.ok


@pytest.mark.parametrize("detail", [_content_filter_message(["sexual:medium"]), _output_filter_message([])])
def test_a_content_filter_block_is_not_retried(eval_client, set_stream_chat, sent_requests, detail):
    """The block is deterministic, so a retry would only repeat the whole chat round."""

    def filtered(client, model, domain, message, history, identity, max_tool_rounds=5):
        yield ErrorEvent(message=detail)

    set_stream_chat(filtered)

    result = eval_client.ask("documents", "Ogni quante ore vanno lubrificati i cuscinetti?")

    assert len([r for r in sent_requests if r.url.path == "/chat/message"]) == 1
    assert result.attempts == 1
    assert result.status_code == 500
    assert "content filter" in (result.error or "")


# ---------------------------------------------------------------------------
# WP7 — tracing stays off unless it is switched on
# ---------------------------------------------------------------------------


def test_no_traceparent_header_when_langfuse_is_disabled(dataset, eval_client, set_stream_chat, sent_requests):
    set_stream_chat(_answering_fake("ok", []))
    tracer = make_tracer(enabled=False)
    assert isinstance(tracer, NullTracer)

    run_dataset([dataset.items[0]], eval_client, domain="documents", tracer=tracer)

    posts = [r for r in sent_requests if r.url.path == "/chat/message"]
    assert posts
    assert all("traceparent" not in {k.lower() for k in r.headers} for r in posts)


def test_the_null_tracer_yields_no_headers_and_no_trace_id():
    with NullTracer().item_span("Q01", item_id="Q01") as (headers, trace_id):
        assert headers == {}
        assert trace_id is None


def test_tracing_falls_back_to_a_null_tracer_rather_than_failing_a_run(monkeypatch):
    """Observability is never worth losing a whole run over."""
    import sme_chatbot.eval.tracing as tracing_mod

    def boom() -> None:
        raise RuntimeError("no otel here")

    monkeypatch.setattr(tracing_mod, "OtelTracer", boom)
    assert isinstance(tracing_mod.make_tracer(enabled=True), NullTracer)


# ---------------------------------------------------------------------------
# the app under test is the real one
# ---------------------------------------------------------------------------


def test_the_runner_talks_to_the_real_app_factory(app_client):
    assert isinstance(app_client.app, type(main_mod.create_app()))
