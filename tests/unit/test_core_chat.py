"""Event-sequence tests for `sme_chatbot.core.chat.stream_chat` (WP8 / WP2 acceptance).

Drives `stream_chat` against a scripted fake OpenAI streaming client (no
network, no LLM, no creds) and asserts the exact `ChatEvent` sequence for each
scenario listed in the plan's WP2 acceptance criteria plus the extra WP8
cases (max-rounds cutoff, unrecoverable exception):

    docs/specs/2026-07-02-backend-split-and-docsvc-ingestion-plan.md §5 WP2, WP8

All fixtures/fakes live in this file per the WP8a file-ownership split (no
`tests/unit/conftest.py`).
"""

from __future__ import annotations

import contextvars
import json
import threading
from types import SimpleNamespace

import pytest

from sme_chatbot.core.chat import stream_chat
from sme_chatbot.core.domains import DomainConfig
from sme_chatbot.core.events import (
    DoneEvent,
    ErrorEvent,
    RequestIdentity,
    TokenEvent,
    ToolCallEvent,
    ToolResultEvent,
)

# ---------------------------------------------------------------------------
# Fakes: a scripted OpenAI-style streaming client, built from the same shapes
# `openai`'s SDK yields (`chunk.choices[0].delta.content` /
# `.delta.tool_calls[i].function.{name,arguments}` / `.finish_reason`).
# ---------------------------------------------------------------------------


def make_chunk(content=None, tool_calls=None, finish_reason=None):
    """Build one fake streaming chunk (`chunk.choices[0]...`)."""
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice])


def make_empty_choices_chunk():
    """The Ollama-style empty-`choices` sentinel chunk that `stream_chat` skips."""
    return SimpleNamespace(choices=[])


def make_tool_call_delta(index, id_=None, name=None, arguments=None):
    """Build one fake `delta.tool_calls[i]` fragment."""
    function = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(index=index, id=id_, function=function)


class FakeStreamClient:
    """Stand-in for an OpenAI client: `client.chat.completions.create(...)`.

    Constructed with a list of "rounds"; each round is a list of chunks that
    `stream_chat` will iterate over as if streaming. One round is consumed
    per underlying API call (i.e. per tool-calling loop iteration).
    """

    def __init__(self, chunk_lists):
        self._chunk_lists = list(chunk_lists)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, model, messages, tools=None, stream=True, **extra):
        self.calls.append(
            {"model": model, "messages": messages, "tools": tools, "stream": stream, "extra": extra}
        )
        return iter(self._chunk_lists.pop(0))


class BoomClient:
    """A client whose first API call raises, to exercise the terminal ErrorEvent path."""

    def __init__(self, exc: Exception):
        self._exc = exc
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        raise self._exc


def make_domain(dispatch=None, tool_definitions=None) -> DomainConfig:
    return DomainConfig(
        key="test",
        title="Test domain",
        system_prompt="You are a test assistant.",
        tool_definitions=tool_definitions or [],
        dispatch=dispatch or (lambda name, args: "{}"),
    )


def run(client, dispatch=None, max_tool_rounds=5, message="hi", history=None, **sampling):
    """Run `stream_chat` to completion and return the list of yielded events."""
    domain = make_domain(dispatch=dispatch)
    identity = RequestIdentity(user_id="u1")
    return list(
        stream_chat(
            client,
            model="test-model",
            domain=domain,
            message=message,
            history=history or [],
            identity=identity,
            max_tool_rounds=max_tool_rounds,
            **sampling,
        )
    )


# ---------------------------------------------------------------------------
# (a) text-only stream -> TokenEvents + DoneEvent with full content
# ---------------------------------------------------------------------------


def test_text_only_stream_yields_tokens_then_done():
    client = FakeStreamClient(
        [
            [
                make_chunk(content="Hello"),
                make_chunk(content=" world"),
                make_chunk(content=None, finish_reason="stop"),
            ]
        ]
    )

    events = run(client)

    assert [type(e) for e in events] == [TokenEvent, TokenEvent, DoneEvent]
    assert events[0].text == "Hello"
    assert events[1].text == " world"
    done = events[2]
    assert done.content == "Hello world"
    assert done.citations == []
    assert done.usage["rounds"] == 1
    # Single API call for a text-only (no tool round) response.
    assert len(client.calls) == 1


def test_text_only_stream_skips_empty_choices_sentinel_chunks():
    """Ollama emits empty-`choices` sentinel chunks mid-stream; they must be
    ignored rather than raising (see the `if not chunk.choices: continue` guard).
    """
    client = FakeStreamClient(
        [
            [
                make_chunk(content="Hi"),
                make_empty_choices_chunk(),
                make_chunk(content=None, finish_reason="stop"),
            ]
        ]
    )

    events = run(client)

    assert [type(e) for e in events] == [TokenEvent, DoneEvent]
    assert events[-1].content == "Hi"


# ---------------------------------------------------------------------------
# (b) one tool round -> ToolCallEvent/ToolResultEvent with citation
# extraction, then DoneEvent with usage rounds
# ---------------------------------------------------------------------------


def test_one_tool_round_emits_tool_events_with_citations_then_done():
    dispatch_calls = []

    def fake_dispatch(name, args):
        dispatch_calls.append((name, args))
        return json.dumps(
            {
                "results": [
                    {"text": "chunk text", "source": "manual.pdf", "page": 12, "distance": 0.1}
                ]
            }
        )

    client = FakeStreamClient(
        [
            [
                make_chunk(
                    tool_calls=[
                        make_tool_call_delta(0, id_="call_1", name="search_manuals", arguments="")
                    ]
                ),
                make_chunk(tool_calls=[make_tool_call_delta(0, arguments='{"query": ')]),
                make_chunk(tool_calls=[make_tool_call_delta(0, arguments='"foo"}')]),
                make_chunk(finish_reason="tool_calls"),
            ],
            [
                make_chunk(content="Per il manuale, pag. 12."),
                make_chunk(finish_reason="stop"),
            ],
        ]
    )

    events = run(client, dispatch=fake_dispatch)

    assert [type(e) for e in events] == [
        ToolCallEvent,
        ToolResultEvent,
        TokenEvent,
        DoneEvent,
    ]

    tool_call, tool_result, token, done = events

    assert tool_call.name == "search_manuals"
    assert tool_call.arguments == {"query": "foo"}
    assert dispatch_calls == [("search_manuals", {"query": "foo"})]

    assert tool_result.ok is True
    assert len(tool_result.citations) == 1
    citation = tool_result.citations[0]
    assert citation.source == "manual.pdf"
    assert citation.page == 12
    assert citation.distance == 0.1
    assert citation.document_id is None

    assert token.text == "Per il manuale, pag. 12."

    assert done.content == "Per il manuale, pag. 12."
    assert done.citations == [citation]
    assert done.usage["rounds"] == 2

    # Two API calls: the tool-requesting round, then the answer round. The
    # second call's messages must include the tool result so the model can
    # ground its answer in it.
    assert len(client.calls) == 2
    second_call_messages = client.calls[1]["messages"]
    tool_messages = [m for m in second_call_messages if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["tool_call_id"] == "call_1"


def test_tool_dispatch_failure_yields_ok_false_with_no_citations():
    def failing_dispatch(name, args):
        raise RuntimeError("db unreachable")

    client = FakeStreamClient(
        [
            [
                make_chunk(
                    tool_calls=[
                        make_tool_call_delta(
                            0, id_="call_1", name="query_cdo_db", arguments='{"sql": "select 1"}'
                        )
                    ]
                ),
                make_chunk(finish_reason="tool_calls"),
            ],
            [
                make_chunk(content="Sorry, something went wrong."),
                make_chunk(finish_reason="stop"),
            ],
        ]
    )

    events = run(client, dispatch=failing_dispatch)

    tool_call, tool_result, token, done = events
    assert isinstance(tool_call, ToolCallEvent)
    assert isinstance(tool_result, ToolResultEvent)
    assert tool_result.ok is False
    assert tool_result.citations == []
    # The loop survives a dispatch exception and completes normally.
    assert isinstance(done, DoneEvent)
    assert done.citations == []


# ---------------------------------------------------------------------------
# (c) JSON-broken tool arguments -> loop survives (WP2 acceptance)
# ---------------------------------------------------------------------------


def test_broken_json_tool_arguments_survives_and_continues():
    """Malformed accumulated `arguments` JSON must not crash `stream_chat`.

    Per `core/chat.py`, a `json.JSONDecodeError` on the accumulated tool-call
    arguments yields `ToolCallEvent(name, arguments={})` (empty, since parsing
    failed) followed immediately by `ToolResultEvent(ok=False)`, appends an
    `{"error": ...}` tool message, and the loop proceeds to the next round
    WITHOUT ever calling `domain.dispatch` for that tool call.
    """
    dispatch_calls = []

    def dispatch(name, args):
        dispatch_calls.append((name, args))
        return "{}"

    client = FakeStreamClient(
        [
            [
                make_chunk(
                    tool_calls=[
                        make_tool_call_delta(
                            0, id_="call_1", name="search_manuals", arguments="{not valid json"
                        )
                    ]
                ),
                make_chunk(finish_reason="tool_calls"),
            ],
            [
                make_chunk(content="Recovered answer."),
                make_chunk(finish_reason="stop"),
            ],
        ]
    )

    events = run(client, dispatch=dispatch)

    assert [type(e) for e in events] == [
        ToolCallEvent,
        ToolResultEvent,
        TokenEvent,
        DoneEvent,
    ]
    tool_call, tool_result, token, done = events

    assert tool_call.name == "search_manuals"
    assert tool_call.arguments == {}
    assert tool_result.ok is False
    assert tool_result.citations == []
    # dispatch is never reached for a tool call whose arguments failed to parse.
    assert dispatch_calls == []

    assert done.content == "Recovered answer."
    assert done.usage["rounds"] == 2

    # The tool message appended to the transcript carries an error payload.
    second_call_messages = client.calls[1]["messages"]
    tool_messages = [m for m in second_call_messages if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    error_payload = json.loads(tool_messages[0]["content"])
    assert "error" in error_payload


def _two_call_round(second_arguments='{"q": "b"}'):
    """One LLM round asking for two tool calls, then an answer round."""
    return [
        [
            make_chunk(tool_calls=[make_tool_call_delta(0, id_="call_a", name="search_a", arguments='{"q": "a"}')]),
            make_chunk(tool_calls=[make_tool_call_delta(1, id_="call_b", name="search_b", arguments=second_arguments)]),
            make_chunk(finish_reason="tool_calls"),
        ],
        [make_chunk(content="Done."), make_chunk(finish_reason="stop")],
    ]


def _citing_dispatch(name, args):
    return json.dumps({"results": [{"source": f"{name}.pdf", "page": 1}]})


def test_two_calls_in_one_round_run_in_order_with_interleaved_events():
    """Pins the sequential behaviour that ``tool_concurrency=1`` must keep exactly."""
    ran = []

    def dispatch(name, args):
        ran.append((name, args))
        return _citing_dispatch(name, args)

    client = FakeStreamClient(_two_call_round())

    events = run(client, dispatch=dispatch, tool_concurrency=1)

    assert [(type(e).__name__, getattr(e, "name", None)) for e in events] == [
        ("ToolCallEvent", "search_a"),
        ("ToolResultEvent", "search_a"),
        ("ToolCallEvent", "search_b"),
        ("ToolResultEvent", "search_b"),
        ("TokenEvent", None),
        ("DoneEvent", None),
    ]
    assert ran == [("search_a", {"q": "a"}), ("search_b", {"q": "b"})]
    done = events[-1]
    assert [c.source for c in done.citations] == ["search_a.pdf", "search_b.pdf"]
    assert [t["name"] for t in done.usage["timings"]["tools"]] == ["search_a", "search_b"]
    messages = client.calls[1]["messages"]
    assert [tc["id"] for tc in messages[-3]["tool_calls"]] == ["call_a", "call_b"]
    assert [(m["role"], m["tool_call_id"]) for m in messages[-2:]] == [("tool", "call_a"), ("tool", "call_b")]


# ---------------------------------------------------------------------------
# tool calls of one round run concurrently (issue #4)
# ---------------------------------------------------------------------------


def _event_names(events):
    return [(type(e).__name__, getattr(e, "name", None)) for e in events]


def _tool_messages(client):
    return [(m["tool_call_id"], json.loads(m["content"])) for m in client.calls[1]["messages"] if m["role"] == "tool"]


@pytest.mark.parametrize(("tool_concurrency", "overlap"), [(2, True), (1, False)])
def test_two_slow_calls_overlap_only_when_concurrency_allows_it(tool_concurrency, overlap):
    # Each call waits for the other at the barrier: it is passed only if both run at once.
    # Sequential calls can never meet there, so they time out (quickly) and fail instead.
    barrier = threading.Barrier(2, timeout=5 if overlap else 0.2)

    def dispatch(name, args):
        barrier.wait()
        return _citing_dispatch(name, args)

    client = FakeStreamClient(_two_call_round())

    events = run(client, dispatch=dispatch, tool_concurrency=tool_concurrency)

    assert [e.ok for e in events if isinstance(e, ToolResultEvent)] == [overlap, overlap]
    assert isinstance(events[-1], DoneEvent)


def test_concurrent_calls_send_every_call_first_then_results_in_call_order():
    """The second call finishes first; events, messages, citations and timings keep call order."""
    b_done = threading.Event()

    def dispatch(name, args):
        if name == "search_a":
            assert b_done.wait(timeout=5), "search_b never finished: the calls did not overlap"
        else:
            b_done.set()
        return _citing_dispatch(name, args)

    client = FakeStreamClient(_two_call_round())

    events = run(client, dispatch=dispatch, tool_concurrency=4)

    assert _event_names(events) == [
        ("ToolCallEvent", "search_a"),
        ("ToolCallEvent", "search_b"),
        ("ToolResultEvent", "search_a"),
        ("ToolResultEvent", "search_b"),
        ("TokenEvent", None),
        ("DoneEvent", None),
    ]
    assert [e.arguments for e in events[:2]] == [{"q": "a"}, {"q": "b"}]
    assert all(e.ok for e in events if isinstance(e, ToolResultEvent))
    done = events[-1]
    assert [c.source for c in done.citations] == ["search_a.pdf", "search_b.pdf"]
    assert [t["name"] for t in done.usage["timings"]["tools"]] == ["search_a", "search_b"]
    assert [call_id for call_id, _ in _tool_messages(client)] == ["call_a", "call_b"]


def test_a_failing_concurrent_call_does_not_affect_the_other():
    def dispatch(name, args):
        if name == "search_a":
            raise RuntimeError("chroma down")
        return _citing_dispatch(name, args)

    client = FakeStreamClient(_two_call_round())

    events = run(client, dispatch=dispatch, tool_concurrency=4)

    results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert [(r.name, r.ok) for r in results] == [("search_a", False), ("search_b", True)]
    assert results[0].citations == []
    assert [c.source for c in events[-1].citations] == ["search_b.pdf"]
    (_, a_payload), (_, b_payload) = _tool_messages(client)
    assert "chroma down" in a_payload["error"]
    assert "results" in b_payload
    assert [t["ok"] for t in events[-1].usage["timings"]["tools"]] == [False, True]


def test_invalid_arguments_next_to_a_valid_call_are_rejected_without_dispatch():
    ran = []

    def dispatch(name, args):
        ran.append((name, threading.current_thread()))
        return _citing_dispatch(name, args)

    client = FakeStreamClient(_two_call_round(second_arguments="{oops"))

    events = run(client, dispatch=dispatch, tool_concurrency=4)

    assert _event_names(events)[:4] == [
        ("ToolCallEvent", "search_a"),
        ("ToolCallEvent", "search_b"),
        ("ToolResultEvent", "search_a"),
        ("ToolResultEvent", "search_b"),
    ]
    assert events[1].arguments == {}
    assert [e.ok for e in events if isinstance(e, ToolResultEvent)] == [True, False]
    # Only one call left to run: nothing to overlap, so it runs inline.
    assert ran == [("search_a", threading.current_thread())]
    assert "Invalid arguments" in _tool_messages(client)[1][1]["error"]
    assert events[-1].usage["timings"]["tools"][1] == {"name": "search_b", "ms": 0, "ok": False}


def test_a_single_call_runs_inline_on_the_loop_thread():
    threads = []

    def dispatch(name, args):
        threads.append(threading.current_thread())
        return "{}"

    run(FakeStreamClient(_tool_round_then_answer()), dispatch=dispatch, tool_concurrency=4)

    assert threads == [threading.current_thread()]


@pytest.mark.parametrize(("tool_concurrency", "expected_peak"), [(2, 2), (3, 3)])
def test_no_more_calls_run_at_once_than_tool_concurrency(tool_concurrency, expected_peak):
    # Every call stays busy until all three are running at once, or a short timeout passes.
    # So the peak reached is exactly the number of calls the pool lets run together.
    all_running = threading.Condition()
    active = 0
    peak = 0

    def dispatch(name, args):
        nonlocal active, peak
        with all_running:
            active += 1
            peak = max(peak, active)
            all_running.notify_all()
            all_running.wait_for(lambda: active >= 3, timeout=0.3)
            active -= 1
        return "{}"

    client = FakeStreamClient(
        [
            [
                make_chunk(tool_calls=[make_tool_call_delta(i, id_=f"call_{n}", name=f"search_{n}", arguments="{}")])
                for i, n in enumerate("abc")
            ]
            + [make_chunk(finish_reason="tool_calls")],
            [make_chunk(content="ok"), make_chunk(finish_reason="stop")],
        ]
    )

    events = run(client, dispatch=dispatch, tool_concurrency=tool_concurrency)

    assert peak == expected_peak
    assert [e.ok for e in events if isinstance(e, ToolResultEvent)] == [True, True, True]


def test_closing_the_stream_mid_round_does_not_wait_for_a_running_call():
    """A client that disconnects must not hold its request thread until a slow tool returns."""
    release = threading.Event()
    b_finished = threading.Event()

    def dispatch(name, args):
        if name == "search_b":
            release.wait(timeout=5)
            b_finished.set()
        return "{}"

    domain = make_domain(dispatch=dispatch)
    stream = stream_chat(
        FakeStreamClient(_two_call_round()), model="m", domain=domain, message="hi", history=[],
        identity=RequestIdentity(user_id="u1"), tool_concurrency=4,
    )
    first_result = next(e for e in stream if isinstance(e, ToolResultEvent))
    assert first_result.name == "search_a"

    stream.close()  # search_b is still blocked in its worker thread

    closed_before_b_finished = not b_finished.is_set()
    release.set()
    assert closed_before_b_finished


_request_tag: contextvars.ContextVar[str] = contextvars.ContextVar("request_tag", default="unset")


def test_concurrent_calls_see_the_callers_context_variables():
    seen = []
    main_thread = threading.current_thread()

    def dispatch(name, args):
        seen.append((name, _request_tag.get(), threading.current_thread() is main_thread))
        _request_tag.set(f"changed by {name}")  # must not leak into the caller or the other call
        return "{}"

    token = _request_tag.set("req-42")
    try:
        run(FakeStreamClient(_two_call_round()), dispatch=dispatch, tool_concurrency=4)
        assert _request_tag.get() == "req-42"
    finally:
        _request_tag.reset(token)

    assert sorted(seen) == [("search_a", "req-42", False), ("search_b", "req-42", False)]


# ---------------------------------------------------------------------------
# (d) <think> block split across multiple deltas -> no fragment leaks
# ---------------------------------------------------------------------------


def test_think_block_split_across_deltas_leaves_no_fragments():
    client = FakeStreamClient(
        [
            [
                make_chunk(content="<thi"),
                make_chunk(content="nk>ignore this reasoning"),
                make_chunk(content="</th"),
                make_chunk(content="ink>Real answer factual content"),
                make_chunk(finish_reason="stop"),
            ]
        ]
    )

    events = run(client)

    token_events = [e for e in events if isinstance(e, TokenEvent)]
    assert len(token_events) == 1
    assert token_events[0].text == "Real answer factual content"

    for e in token_events:
        assert "<think" not in e.text
        assert "</think" not in e.text
        assert "ignore this reasoning" not in e.text

    done = events[-1]
    assert isinstance(done, DoneEvent)
    assert done.content == "Real answer factual content"
    assert "ignore this reasoning" not in done.content


def test_think_block_split_with_visible_text_before_and_after():
    """A <think> block sandwiched between two visible spans, each itself
    split across chunk boundaries, must still yield only the visible parts
    with none of the tag or reasoning text leaking.
    """
    client = FakeStreamClient(
        [
            [
                make_chunk(content="Before"),
                make_chunk(content="Vis <th"),
                make_chunk(content="ink>secret reasoning "),
                make_chunk(content="continues</think"),
                make_chunk(content=">AfterVis"),
                make_chunk(finish_reason="stop"),
            ]
        ]
    )

    events = run(client)
    token_events = [e for e in events if isinstance(e, TokenEvent)]
    visible = "".join(e.text for e in token_events)

    assert visible == "BeforeVis AfterVis"
    assert "secret reasoning" not in visible
    assert "<think" not in visible and "</think" not in visible

    done = events[-1]
    assert done.content == "BeforeVis AfterVis"


# ---------------------------------------------------------------------------
# (e) max_tool_rounds cutoff
# ---------------------------------------------------------------------------


def test_max_tool_rounds_cutoff_terminates_with_done():
    def endless_tool_round():
        return [
            make_chunk(
                tool_calls=[
                    make_tool_call_delta(0, id_="call_x", name="search_manuals", arguments='{"query": "x"}')
                ]
            ),
            make_chunk(finish_reason="tool_calls"),
        ]

    max_tool_rounds = 2
    # max_tool_rounds+1 rounds get executed before the cutoff forces a DoneEvent,
    # plus one final tools-less round forced to produce an answer.
    client = FakeStreamClient(
        [endless_tool_round() for _ in range(max_tool_rounds + 1)]
        + [[make_chunk(content="Ecco la risposta."), make_chunk(finish_reason="stop")]]
    )

    def dispatch(name, args):
        return json.dumps({"results": [{"source": "s.pdf", "page": 1}]})

    events = run(client, dispatch=dispatch, max_tool_rounds=max_tool_rounds)

    assert isinstance(events[-1], DoneEvent)
    assert events[-1].usage["rounds"] == max_tool_rounds + 1
    assert events[-1].content == "Ecco la risposta."
    # max_tool_rounds+1 API calls before the cutoff (the last one's tool
    # request is what trips it and is never dispatched), plus one more
    # tools-less call forced to produce a final answer.
    assert len(client.calls) == max_tool_rounds + 2
    assert client.calls[-1]["tools"] is None
    # Only the rounds *before* the cutoff round get dispatched and surface a
    # ToolCallEvent/ToolResultEvent pair + citations; the cutoff round's own
    # tool request short-circuits straight to the forced final answer without
    # dispatching.
    tool_call_events = [e for e in events if isinstance(e, ToolCallEvent)]
    assert len(tool_call_events) == max_tool_rounds
    assert len(events[-1].citations) == max_tool_rounds


def test_max_tool_rounds_cutoff_forces_an_answer_when_every_round_was_silent():
    """Every round is a pure tool call with no narration
    text, so the pre-fix cutoff returned empty content. The forced final
    tools-less round must now synthesize an answer from gathered context."""

    def silent_tool_round():
        return [
            make_chunk(
                tool_calls=[
                    make_tool_call_delta(0, id_="call_x", name="search_documents", arguments="{}")
                ]
            ),
            make_chunk(finish_reason="tool_calls"),
        ]

    max_tool_rounds = 2
    client = FakeStreamClient(
        [silent_tool_round() for _ in range(max_tool_rounds + 1)]
        + [[make_chunk(content="M12 = 47 Nm."), make_chunk(finish_reason="stop")]]
    )

    def dispatch(name, args):
        return json.dumps({"results": [{"source": "s.pdf", "page": 9}]})

    events = run(client, dispatch=dispatch, max_tool_rounds=max_tool_rounds)

    done = events[-1]
    assert isinstance(done, DoneEvent)
    assert done.content == "M12 = 47 Nm."
    assert done.usage["rounds"] == max_tool_rounds + 1
    # The nudge message telling the model to stop calling tools is present.
    forced_call_messages = client.calls[-1]["messages"]
    assert forced_call_messages[-1] == {
        "role": "user",
        "content": (
            "You have reached the maximum number of tool calls for this "
            "turn. Answer now using only the information already gathered "
            "above; do not request further tool calls. If part of the "
            "answer was not found, say so plainly — do not offer to "
            "search again in a later message."
        ),
    }


# ---------------------------------------------------------------------------
# (f) unrecoverable exception -> ErrorEvent terminal
# ---------------------------------------------------------------------------


def test_unrecoverable_exception_yields_single_terminal_error_event():
    client = BoomClient(RuntimeError("boom: connection refused"))

    events = run(client)

    assert len(events) == 1
    assert isinstance(events[0], ErrorEvent)
    assert "boom: connection refused" in events[0].message


def test_a_content_filter_block_yields_a_readable_error_not_the_raw_azure_json(azure_content_filter_error, caplog):
    """Q169 surfaced Azure's JSON dump to the user as an HTTP 500."""
    events = run(BoomClient(azure_content_filter_error))

    assert len(events) == 1
    assert isinstance(events[0], ErrorEvent)
    message = events[0].message
    assert "content filter (sexual:medium)" in message
    assert "innererror" not in message and "{" not in message
    # A provider refusal is expected behaviour, not a crash: warned, no traceback.
    assert any("content filter" in r.getMessage() and r.levelname == "WARNING" for r in caplog.records)
    assert not any(r.exc_info for r in caplog.records)


def make_filtered_chunk(**results):
    """The chunk Azure ends a stream with when it filters the answer itself."""
    chunk = make_chunk(finish_reason="content_filter")
    chunk.choices[0].content_filter_results = {
        category: {"filtered": True, "severity": severity} for category, severity in results.items()
    }
    return chunk


def test_an_answer_cut_off_by_the_content_filter_ends_in_an_error_not_done(caplog):
    """Azure can filter the answer partway; the loop used to pass the cut-off text off as complete."""
    client = FakeStreamClient(
        [[make_chunk(content="Prima del primo avviamento, "), make_filtered_chunk(sexual="medium")]]
    )

    events = run(client)

    assert [type(e) for e in events] == [TokenEvent, ErrorEvent]
    assert "content filter (sexual:medium)" in events[-1].message
    assert "incomplete" in events[-1].message
    assert any("content filter" in r.getMessage() and r.levelname == "WARNING" for r in caplog.records)
    assert not any(r.exc_info for r in caplog.records)


def test_an_unreported_category_still_reads_as_a_content_filter_stop():
    client = FakeStreamClient([[make_chunk(content="x"), make_chunk(finish_reason="content_filter")]])

    events = run(client)

    assert isinstance(events[-1], ErrorEvent)
    assert "category not reported" in events[-1].message


def test_the_forced_final_answer_is_checked_for_the_content_filter_too():
    tool_round = [
        make_chunk(tool_calls=[make_tool_call_delta(0, id_="c1", name="search_manuals", arguments="{}")]),
        make_chunk(finish_reason="tool_calls"),
    ]
    client = FakeStreamClient([tool_round, [make_chunk(content="Ecco"), make_filtered_chunk(sexual="medium")]])

    events = run(client, dispatch=lambda name, args: json.dumps({"results": []}), max_tool_rounds=0)

    assert not any(isinstance(e, DoneEvent) for e in events)
    assert isinstance(events[-1], ErrorEvent)
    assert "sexual:medium" in events[-1].message


def test_unrecoverable_exception_mid_stream_yields_error_event():
    """The exception need not occur on the API call itself — an iterator that
    raises partway through a stream must also terminate in a single ErrorEvent,
    with no dangling TokenEvents left unflushed after the failure point.
    """

    def boom_iter():
        yield make_chunk(content="partial")
        raise RuntimeError("stream dropped")

    class MidStreamBoomClient:
        def __init__(self):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, **kwargs):
            return boom_iter()

    events = run(MidStreamBoomClient())

    # A TokenEvent for the successfully-streamed fragment may or may not have
    # been emitted before the failure, but the stream must end in exactly one
    # ErrorEvent and nothing after it.
    assert isinstance(events[-1], ErrorEvent)
    assert "stream dropped" in events[-1].message
    assert sum(isinstance(e, ErrorEvent) for e in events) == 1


# ---------------------------------------------------------------------------
# History plumbing sanity check (not in the required list, but cheap and
# guards a real regression: prior turns must reach the model as-is).
# ---------------------------------------------------------------------------


def test_history_and_system_prompt_are_prepended_to_messages():
    client = FakeStreamClient([[make_chunk(content="ok"), make_chunk(finish_reason="stop")]])
    history = [
        {"role": "user", "content": "previous question"},
        {"role": "assistant", "content": "previous answer"},
    ]

    run(client, history=history, message="new question")

    sent_messages = client.calls[0]["messages"]
    assert sent_messages[0] == {"role": "system", "content": "You are a test assistant."}
    assert sent_messages[1:3] == history
    assert sent_messages[-1] == {"role": "user", "content": "new question"}


# ---------------------------------------------------------------------------
# (g) DoneEvent.content spans the whole turn, not just the last round
# ---------------------------------------------------------------------------


def test_visible_text_survives_a_following_tool_round():
    """Text streamed before a tool call must reach `done`.

    A `TokenEvent` was emitted for the round-1 text, so a token-accumulating
    client already shows it; when `DoneEvent.content` was reset per round it
    came back empty, and the Gradio client (which renders `done.content`)
    replaced the visible answer with a bare citation footer.
    """
    client = FakeStreamClient(
        [
            [
                make_chunk(content="Ecco quanto ho trovato: "),
                make_chunk(
                    tool_calls=[
                        make_tool_call_delta(
                            0, id_="call_1", name="search_manuals", arguments='{"query": "x"}'
                        )
                    ]
                ),
                make_chunk(finish_reason="tool_calls"),
            ],
            # The model answers by citing the tool result and emits no prose of
            # its own in this round - the shape that used to lose the answer.
            [make_chunk(finish_reason="stop")],
        ]
    )

    def dispatch(name, args):
        return json.dumps({"results": [{"source": "manual.pdf", "page": 12}]})

    events = run(client, dispatch=dispatch)

    done = events[-1]
    assert isinstance(done, DoneEvent)
    assert done.content == "Ecco quanto ho trovato: "
    # And it equals what a token-accumulating client holds.
    assert done.content == "".join(e.text for e in events if isinstance(e, TokenEvent))


def test_visible_text_accumulates_across_several_rounds():
    client = FakeStreamClient(
        [
            [
                make_chunk(content="Controllo il manuale. "),
                make_chunk(
                    tool_calls=[
                        make_tool_call_delta(
                            0, id_="c1", name="search_manuals", arguments='{"query": "a"}'
                        )
                    ]
                ),
                make_chunk(finish_reason="tool_calls"),
            ],
            [
                make_chunk(content="Ora la tabella coppie. "),
                make_chunk(
                    tool_calls=[
                        make_tool_call_delta(
                            0, id_="c2", name="search_manuals", arguments='{"query": "b"}'
                        )
                    ]
                ),
                make_chunk(finish_reason="tool_calls"),
            ],
            [
                make_chunk(content="M12 = 47 Nm (manual.pdf, p. 9)."),
                make_chunk(finish_reason="stop"),
            ],
        ]
    )

    def dispatch(name, args):
        return json.dumps({"results": [{"source": "manual.pdf", "page": 9}]})

    events = run(client, dispatch=dispatch)

    assert events[-1].content == (
        "Controllo il manuale. Ora la tabella coppie. M12 = 47 Nm (manual.pdf, p. 9)."
    )


def test_max_tool_rounds_cutoff_keeps_the_text_streamed_so_far():
    """The round-cap exit path is the other way the empty answers happened."""

    def talking_tool_round(text):
        return [
            make_chunk(content=text),
            make_chunk(
                tool_calls=[
                    make_tool_call_delta(0, id_="cx", name="search_manuals", arguments='{"q": 1}')
                ]
            ),
            make_chunk(finish_reason="tool_calls"),
        ]

    max_tool_rounds = 2
    client = FakeStreamClient(
        [
            talking_tool_round("uno. "),
            talking_tool_round("due. "),
            talking_tool_round("tre. "),
            [make_chunk(content="quattro."), make_chunk(finish_reason="stop")],
        ]
    )

    def dispatch(name, args):
        return json.dumps({"results": [{"source": "s.pdf", "page": 1}]})

    events = run(client, dispatch=dispatch, max_tool_rounds=max_tool_rounds)

    done = events[-1]
    assert isinstance(done, DoneEvent)
    # Text streamed alongside the cutoff round's own tool call ("tre. ") plus
    # the forced final tools-less round's answer ("quattro.").
    assert done.content == "uno. due. tre. quattro."


# ---------------------------------------------------------------------------
# per-stage timings in DoneEvent.usage["timings"]
# ---------------------------------------------------------------------------


class FakeClock:
    """Deterministic stand-in for ``chat._now``; advanced by the fakes, never by sleeping."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance_ms(self, ms):
        self.t += ms / 1000


def timed(clock, steps):
    """Yield chunks, advancing ``clock`` by ``ms`` just before each one."""
    for ms, chunk in steps:
        clock.advance_ms(ms)
        yield chunk


def test_timings_split_llm_and_tool_time_and_ttft(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr("sme_chatbot.core.chat._now", clock)

    def slow_dispatch(name, args):
        clock.advance_ms(250)
        return "{}"

    client = FakeStreamClient(
        [
            timed(
                clock,
                [
                    (100, make_chunk(tool_calls=[make_tool_call_delta(0, id_="c1", name="search_documents", arguments="{}")])),
                    (50, make_chunk(finish_reason="tool_calls")),
                ],
            ),
            timed(
                clock,
                [
                    (400, make_chunk(content="Risposta")),
                    (100, make_chunk(finish_reason="stop")),
                ],
            ),
        ]
    )

    done = run(client, dispatch=slow_dispatch)[-1]

    assert isinstance(done, DoneEvent)
    assert done.usage["rounds"] == 2
    timings = done.usage["timings"]
    assert timings["total_ms"] == 900
    assert timings["llm_ms"] == 650
    assert timings["tool_ms"] == 250
    # 150 (call 1) + 250 (tool) + 400 (silence before the first visible token)
    assert timings["ttft_ms"] == 800
    assert timings["llm_calls"] == [
        {"round": 0, "first_chunk_ms": 100, "total_ms": 150, "finish": "tool_calls"},
        {"round": 1, "first_chunk_ms": 400, "total_ms": 500, "finish": "stop"},
    ]
    assert timings["tools"] == [{"name": "search_documents", "ms": 250, "ok": True}]


def test_timings_record_the_forced_final_call_after_the_cap(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr("sme_chatbot.core.chat._now", clock)

    def tool_round():
        return timed(
            clock,
            [
                (10, make_chunk(tool_calls=[make_tool_call_delta(0, id_="c", name="t", arguments="{}")])),
                (10, make_chunk(finish_reason="tool_calls")),
            ],
        )

    client = FakeStreamClient(
        [tool_round(), tool_round(), timed(clock, [(30, make_chunk(content="Fine."))])]
    )

    done = run(client, max_tool_rounds=1)[-1]

    timings = done.usage["timings"]
    assert [c["finish"] for c in timings["llm_calls"]] == ["tool_calls", "tool_calls", "forced_final"]
    assert timings["llm_calls"][-1]["first_chunk_ms"] == 30
    assert len(timings["tools"]) == 1  # the cap round never dispatches its tool calls
    assert timings["ttft_ms"] is not None


def test_a_turn_that_hits_the_round_cap_is_flagged_round_limit_hit():
    """The forced-final path must say so in ``usage``, so cap hits can be counted."""
    client = FakeStreamClient([_tool_round(), _tool_round(), [make_chunk(content="Fine.")]])

    done = run(client, max_tool_rounds=1)[-1]

    assert done.usage["round_limit_hit"] is True


def test_an_answer_in_the_last_allowed_round_is_not_flagged_round_limit_hit():
    """Same ``rounds`` as a cap hit, but the model answered on its own: not a cap hit."""
    client = FakeStreamClient([_tool_round(), [make_chunk(content="Fine."), make_chunk(finish_reason="stop")]])

    done = run(client, max_tool_rounds=1)[-1]

    assert done.usage["rounds"] == 2
    assert done.usage["round_limit_hit"] is False


def _tool_round():
    return [
        make_chunk(tool_calls=[make_tool_call_delta(0, id_="c", name="t", arguments="{}")]),
        make_chunk(finish_reason="tool_calls"),
    ]


def test_timings_ttft_is_none_when_no_visible_token_is_emitted():
    client = FakeStreamClient([[make_chunk(content=None, finish_reason="stop")]])

    done = run(client)[-1]

    assert done.usage["timings"]["ttft_ms"] is None
    assert done.usage["timings"]["tools"] == []


def test_timings_record_a_tool_with_broken_arguments_as_not_ok():
    client = FakeStreamClient(
        [
            [
                make_chunk(tool_calls=[make_tool_call_delta(0, id_="c", name="t", arguments="{oops")]),
                make_chunk(finish_reason="tool_calls"),
            ],
            [make_chunk(content="ok"), make_chunk(finish_reason="stop")],
        ]
    )

    done = run(client)[-1]

    assert done.usage["timings"]["tools"] == [{"name": "t", "ms": 0, "ok": False}]


# ---------------------------------------------------------------------------
# sampling settings: sent only when configured
# ---------------------------------------------------------------------------


def _tool_round_then_answer():
    return [
        [
            make_chunk(tool_calls=[make_tool_call_delta(0, id_="c1", name="t", arguments="{}")]),
            make_chunk(finish_reason="tool_calls"),
        ],
        [make_chunk(content="ok"), make_chunk(finish_reason="stop")],
    ]


def test_sampling_is_not_sent_when_unset():
    """An unconfigured backend must pass exactly what it did before: the server's own defaults apply."""
    client = FakeStreamClient(_tool_round_then_answer())

    run(client)

    assert [c["extra"] for c in client.calls] == [{}, {}]


def test_temperature_and_seed_are_sent_on_every_call():
    client = FakeStreamClient(_tool_round_then_answer())

    run(client, temperature=0.2, seed=42)

    assert [c["extra"] for c in client.calls] == [{"temperature": 0.2, "seed": 42}] * 2


def test_temperature_zero_is_sent_not_mistaken_for_unset():
    client = FakeStreamClient(_tool_round_then_answer())

    run(client, temperature=0.0)

    assert [c["extra"] for c in client.calls] == [{"temperature": 0.0}] * 2


def test_seed_alone_is_sent_without_a_temperature():
    client = FakeStreamClient(_tool_round_then_answer())

    run(client, seed=0)

    assert [c["extra"] for c in client.calls] == [{"seed": 0}] * 2


def test_sampling_is_also_sent_on_the_forced_final_call_after_the_round_cap():
    def tool_round():
        return [
            make_chunk(tool_calls=[make_tool_call_delta(0, id_="c", name="t", arguments="{}")]),
            make_chunk(finish_reason="tool_calls"),
        ]

    client = FakeStreamClient([tool_round(), tool_round(), [make_chunk(content="Fine.")]])

    run(client, max_tool_rounds=1, temperature=0.0, seed=7)

    assert len(client.calls) == 3
    assert all(c["extra"] == {"temperature": 0.0, "seed": 7} for c in client.calls)


# ---------------------------------------------------------------------------
# max_tokens: bounds one LLM call; a call cut off by it is recorded as "length"
# ---------------------------------------------------------------------------


def test_max_tokens_is_sent_on_every_call_with_the_other_settings():
    client = FakeStreamClient(_tool_round_then_answer())

    run(client, temperature=0.2, seed=42, max_tokens=4096)

    assert [c["extra"] for c in client.calls] == [{"temperature": 0.2, "seed": 42, "max_tokens": 4096}] * 2


def test_max_tokens_is_also_sent_on_the_forced_final_call():
    def tool_round():
        return [
            make_chunk(tool_calls=[make_tool_call_delta(0, id_="c", name="t", arguments="{}")]),
            make_chunk(finish_reason="tool_calls"),
        ]

    client = FakeStreamClient([tool_round(), tool_round(), [make_chunk(content="Fine.")]])

    run(client, max_tool_rounds=1, max_tokens=512)

    assert [c["extra"] for c in client.calls] == [{"max_tokens": 512}] * 3


def test_a_call_cut_off_by_max_tokens_is_recorded_as_length_and_warned_about(caplog):
    client = FakeStreamClient([[make_chunk(content="La produzione è"), make_chunk(finish_reason="length")]])

    with caplog.at_level("WARNING", logger="sme_chatbot.core.chat"):
        done = run(client, max_tokens=64)[-1]

    assert isinstance(done, DoneEvent)
    assert done.content == "La produzione è"
    assert [c["finish"] for c in done.usage["timings"]["llm_calls"]] == ["length"]
    assert any("max_tokens=64" in r.getMessage() for r in caplog.records)


def test_a_call_that_ends_normally_is_still_recorded_as_stop(caplog):
    client = FakeStreamClient([[make_chunk(content="ok"), make_chunk(finish_reason="stop")]])

    with caplog.at_level("WARNING", logger="sme_chatbot.core.chat"):
        done = run(client, max_tokens=64)[-1]

    assert [c["finish"] for c in done.usage["timings"]["llm_calls"]] == ["stop"]
    assert not [r for r in caplog.records if "max_tokens" in r.getMessage()]


# ---------------------------------------------------------------------------
# reasoning_effort: passed through when set
# ---------------------------------------------------------------------------


def test_reasoning_effort_is_sent_on_every_call_with_the_other_settings():
    client = FakeStreamClient(_tool_round_then_answer())

    run(client, temperature=0.2, seed=42, max_tokens=4096, reasoning_effort="none")

    expected = {"temperature": 0.2, "seed": 42, "max_tokens": 4096, "reasoning_effort": "none"}
    assert [c["extra"] for c in client.calls] == [expected] * 2


def test_reasoning_effort_is_also_sent_on_the_forced_final_call():
    def tool_round():
        return [
            make_chunk(tool_calls=[make_tool_call_delta(0, id_="c", name="t", arguments="{}")]),
            make_chunk(finish_reason="tool_calls"),
        ]

    client = FakeStreamClient([tool_round(), tool_round(), [make_chunk(content="Fine.")]])

    run(client, max_tool_rounds=1, reasoning_effort="low")

    assert [c["extra"] for c in client.calls] == [{"reasoning_effort": "low"}] * 3


def test_reasoning_effort_is_not_sent_when_unset():
    client = FakeStreamClient(_tool_round_then_answer())

    run(client)

    assert all("reasoning_effort" not in c["extra"] for c in client.calls)


# ---------------------------------------------------------------------------
# a call that runs out of output budget before showing anything: one retry, then an error
# ---------------------------------------------------------------------------


def _ends_on_length(content=None):
    chunks = [make_chunk(content=content)] if content else []
    return chunks + [make_chunk(finish_reason="length")]


def test_a_call_cut_off_before_any_text_is_retried_once_without_tools():
    client = FakeStreamClient([_ends_on_length(), [make_chunk(content="Risposta."), make_chunk(finish_reason="stop")]])

    events = run(client, max_tokens=100)

    assert isinstance(events[-1], DoneEvent)
    assert events[-1].content == "Risposta."
    assert [type(e) for e in events] == [TokenEvent, DoneEvent]
    assert len(client.calls) == 2
    assert client.calls[1]["tools"] is None  # the retry is tool-less
    assert "used all of its output budget" in client.calls[1]["messages"][-1]["content"]
    assert [c["finish"] for c in events[-1].usage["timings"]["llm_calls"]] == ["length", "length_retry"]


def test_the_retry_uses_retry_reasoning_effort_and_keeps_the_other_settings():
    client = FakeStreamClient([_ends_on_length(), [make_chunk(content="Ok"), make_chunk(finish_reason="stop")]])

    run(client, temperature=0.2, seed=42, max_tokens=100, reasoning_effort="low", retry_reasoning_effort="none")

    assert client.calls[0]["extra"] == {"temperature": 0.2, "seed": 42, "max_tokens": 100, "reasoning_effort": "low"}
    assert client.calls[1]["extra"] == {"temperature": 0.2, "seed": 42, "max_tokens": 100, "reasoning_effort": "none"}


def test_without_retry_reasoning_effort_the_retry_keeps_the_original_settings():
    client = FakeStreamClient([_ends_on_length(), [make_chunk(content="Ok"), make_chunk(finish_reason="stop")]])

    run(client, max_tokens=100, reasoning_effort="low")

    assert client.calls[1]["extra"] == {"max_tokens": 100, "reasoning_effort": "low"}


def test_a_retry_that_also_runs_out_of_budget_is_an_error_not_an_empty_answer():
    client = FakeStreamClient([_ends_on_length(), _ends_on_length()])

    events = run(client, max_tokens=100)

    assert isinstance(events[-1], ErrorEvent)
    assert "max_tokens=100" in events[-1].message
    assert not any(isinstance(e, DoneEvent) for e in events)
    assert len(client.calls) == 2  # exactly one retry


def test_a_visible_but_cut_off_answer_is_delivered_without_a_retry():
    client = FakeStreamClient([_ends_on_length("La produzione è")])

    events = run(client, max_tokens=100)

    assert events[-1].content == "La produzione è"
    assert len(client.calls) == 1


def test_a_tool_call_cut_off_by_the_limit_is_retried_without_tools():
    truncated = [
        make_chunk(tool_calls=[make_tool_call_delta(0, id_="c", name="search_documents", arguments='{"query": "pro')]),
        make_chunk(finish_reason="length"),
    ]
    client = FakeStreamClient([truncated, [make_chunk(content="Ok"), make_chunk(finish_reason="stop")]])

    events = run(client, max_tokens=100)

    assert events[-1].content == "Ok"
    assert client.calls[1]["tools"] is None


def test_the_retry_is_not_swallowed_by_an_unclosed_think_block_in_the_failed_call():
    client = FakeStreamClient([[make_chunk(content="<think>sto ragionando"), make_chunk(finish_reason="length")],
                               [make_chunk(content="Ok"), make_chunk(finish_reason="stop")]])

    events = run(client, max_tokens=100)

    assert events[-1].content == "Ok"


def test_a_forced_final_answer_cut_off_before_any_text_is_retried_too():
    def tool_round():
        return [
            make_chunk(tool_calls=[make_tool_call_delta(0, id_="c", name="t", arguments="{}")]),
            make_chunk(finish_reason="tool_calls"),
        ]

    client = FakeStreamClient([tool_round(), tool_round(), _ends_on_length(),
                               [make_chunk(content="Ecco."), make_chunk(finish_reason="stop")]])

    events = run(client, max_tool_rounds=1, max_tokens=100, retry_reasoning_effort="none")

    assert events[-1].content == "Ecco."
    assert len(client.calls) == 4
    assert client.calls[3]["extra"] == {"max_tokens": 100, "reasoning_effort": "none"}
    assert [c["finish"] for c in events[-1].usage["timings"]["llm_calls"]][-2:] == ["length", "length_retry"]


# ---------------------------------------------------------------------------
# pre-search: the domain's search runs before the first LLM call
# ---------------------------------------------------------------------------


def _pre_search_domain(dispatch):
    domain = make_domain(dispatch=dispatch)
    domain.pre_search = ("search_documents", {"query": "Quanto pesa la 90001?"})
    return domain


def _run_domain(client, domain, **kwargs):
    return list(
        stream_chat(client, model="m", domain=domain, message="Quanto pesa la 90001?", history=[],
                    identity=RequestIdentity(user_id="u1"), **kwargs)
    )


def test_pre_search_runs_before_the_first_llm_call_as_a_tool_call_and_result():
    ran = []

    def dispatch(name, args):
        ran.append((name, args))
        return json.dumps({"results": [{"filename": "m.pdf", "page": 18, "text": "850 kg"}]})

    client = FakeStreamClient([[make_chunk(content="Pesa 850 kg."), make_chunk(finish_reason="stop")]])

    events = _run_domain(client, _pre_search_domain(dispatch), pre_search=True)

    assert ran == [("search_documents", {"query": "Quanto pesa la 90001?"})]
    first_call = client.calls[0]["messages"]
    assert [m["role"] for m in first_call] == ["system", "user", "assistant", "tool"]
    assert first_call[2]["tool_calls"][0]["id"] == first_call[3]["tool_call_id"] == "pre_search_0"
    assert isinstance(events[0], ToolCallEvent) and isinstance(events[1], ToolResultEvent)
    done = events[-1]
    assert done.content == "Pesa 850 kg."
    assert [c.page for c in done.citations] == [18]
    assert done.usage["rounds"] == 1
    assert [t["name"] for t in done.usage["timings"]["tools"]] == ["search_documents"]


def test_no_pre_search_unless_enabled():
    ran = []
    client = FakeStreamClient([[make_chunk(content="ok")]])

    _run_domain(client, _pre_search_domain(lambda n, a: ran.append(n) or "{}"))

    assert ran == []
    assert [m["role"] for m in client.calls[0]["messages"]] == ["system", "user"]


def test_pre_search_is_skipped_when_the_domain_defines_none():
    client = FakeStreamClient([[make_chunk(content="ok")]])

    _run_domain(client, make_domain(), pre_search=True)

    assert [m["role"] for m in client.calls[0]["messages"]] == ["system", "user"]


def test_a_failing_pre_search_becomes_an_error_result_not_a_crash():
    def dispatch(name, args):
        raise RuntimeError("chroma down")

    client = FakeStreamClient([[make_chunk(content="Non riesco a cercare.")]])

    events = _run_domain(client, _pre_search_domain(dispatch), pre_search=True)

    assert isinstance(events[-1], DoneEvent)
    assert events[1].ok is False
    assert "chroma down" in client.calls[0]["messages"][3]["content"]
