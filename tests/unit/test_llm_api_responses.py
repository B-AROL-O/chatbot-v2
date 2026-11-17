"""`stream_chat` on `LLM_API=responses`.

Drives the real loop against a scripted fake of `client.responses.create(stream=True)`
built from the Responses event shapes (`response.output_text.delta`,
`response.output_item.added`, `response.function_call_arguments.delta`,
`response.completed` / `response.incomplete`), and checks that the adapter in
`core/llm_api.py` turns them into the same `ChatEvent`s as a chat-completions stream.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

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

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Search.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
        },
    }
]


def text(delta):
    return SimpleNamespace(type="response.output_text.delta", delta=delta, output_index=0)


def call_added(output_index, call_id, name):
    item = SimpleNamespace(type="function_call", call_id=call_id, name=name, arguments="")
    return SimpleNamespace(type="response.output_item.added", item=item, output_index=output_index)


def call_args(output_index, delta):
    return SimpleNamespace(type="response.function_call_arguments.delta", delta=delta, output_index=output_index)


def call_done(output_index, call_id, name, arguments):
    item = SimpleNamespace(type="function_call", call_id=call_id, name=name, arguments=arguments)
    return SimpleNamespace(type="response.output_item.done", item=item, output_index=output_index)


def reasoning_added():
    item = SimpleNamespace(type="reasoning")
    return SimpleNamespace(type="response.output_item.added", item=item, output_index=0)


def completed():
    return SimpleNamespace(type="response.completed", response=SimpleNamespace(incomplete_details=None))


def incomplete(reason):
    details = SimpleNamespace(reason=reason)
    return SimpleNamespace(type="response.incomplete", response=SimpleNamespace(incomplete_details=details))


class FakeResponsesClient:
    """`client.responses.create(...)`; one scripted event list per call. No `chat` attribute,
    so any call that reached `/v1/chat/completions` would fail loudly."""

    def __init__(self, event_lists):
        self._event_lists = list(event_lists)
        self.calls = []
        self.responses = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return iter(self._event_lists.pop(0))


def run(client, dispatch=None, max_tool_rounds=5, history=None, **sampling):
    domain = DomainConfig(
        key="test",
        title="Test domain",
        system_prompt="You are a test assistant.",
        tool_definitions=TOOLS,
        dispatch=dispatch or (lambda name, args: "{}"),
    )
    return list(
        stream_chat(
            client,
            model="gpt-6-luna",
            domain=domain,
            message="hi",
            history=history or [],
            identity=RequestIdentity(user_id="u1"),
            max_tool_rounds=max_tool_rounds,
            llm_api="responses",
            **sampling,
        )
    )


def test_streamed_text_becomes_tokens_and_done():
    client = FakeResponsesClient([[reasoning_added(), text("Hel"), text("lo"), completed()]])

    events = run(client)

    assert [type(e) for e in events] == [TokenEvent, TokenEvent, DoneEvent]
    assert events[-1].content == "Hello"
    assert events[-1].usage["timings"]["llm_calls"][0]["finish"] == "stop"


def test_tool_call_arguments_split_across_events_are_accumulated_and_dispatched():
    seen = []

    def dispatch(name, args):
        seen.append((name, args))
        return json.dumps({"results": [{"filename": "M.pdf", "page": 3}]})

    client = FakeResponsesClient(
        [
            [
                reasoning_added(),
                call_added(1, "call_a", "search_documents"),
                call_args(1, '{"que'),
                call_args(1, 'ry": "tor'),
                call_args(1, 'que"}'),
                call_done(1, "call_a", "search_documents", '{"query": "torque"}'),
                completed(),
            ],
            [text("See page 3."), completed()],
        ]
    )

    events = run(client, dispatch=dispatch)

    assert seen == [("search_documents", {"query": "torque"})]
    assert [type(e) for e in events] == [ToolCallEvent, ToolResultEvent, TokenEvent, DoneEvent]
    assert events[-1].content == "See page 3."
    assert [c.filename for c in events[-1].citations] == ["M.pdf"]
    # The second call carries the call and its output as Responses input items.
    second_input = client.calls[1]["input"]
    assert second_input[-2] == {
        "type": "function_call",
        "call_id": "call_a",
        "name": "search_documents",
        "arguments": '{"query": "torque"}',
    }
    assert second_input[-1]["type"] == "function_call_output"
    assert second_input[-1]["call_id"] == "call_a"


def test_arguments_sent_only_on_item_done_are_still_used():
    seen = []
    client = FakeResponsesClient(
        [
            [
                call_added(0, "call_a", "search_documents"),
                call_done(0, "call_a", "search_documents", '{"query": "x"}'),
                completed(),
            ],
            [text("ok"), completed()],
        ]
    )

    run(client, dispatch=lambda name, args: seen.append(args) or "{}")

    assert seen == [{"query": "x"}]


def test_messages_tools_and_settings_are_mapped_to_the_responses_shape():
    client = FakeResponsesClient([[text("ok"), completed()]])
    history = [{"role": "user", "content": "before"}, {"role": "assistant", "content": "earlier answer"}]

    run(client, history=history, temperature=0.2, seed=42, max_tokens=4096, reasoning_effort="low")

    call = client.calls[0]
    assert call["model"] == "gpt-6-luna"
    assert call["stream"] is True
    assert call["store"] is False
    assert call["input"] == [
        {"role": "system", "content": "You are a test assistant."},
        {"role": "user", "content": "before"},
        {"role": "assistant", "content": "earlier answer"},
        {"role": "user", "content": "hi"},
    ]
    assert call["tools"] == [
        {
            "type": "function",
            "name": "search_documents",
            "description": "Search.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            "strict": False,
        }
    ]
    assert call["temperature"] == 0.2
    assert call["max_output_tokens"] == 4096
    assert call["reasoning"] == {"effort": "low"}
    # Not Responses parameters: seed has no equivalent, the other two were renamed.
    assert not {"seed", "max_tokens", "reasoning_effort"} & call.keys()


def test_max_output_tokens_triggers_the_one_tool_less_retry():
    client = FakeResponsesClient(
        [[reasoning_added(), incomplete("max_output_tokens")], [text("Short answer."), completed()]]
    )

    events = run(client, max_tokens=100, reasoning_effort="low", retry_reasoning_effort="none")

    assert [type(e) for e in events] == [TokenEvent, DoneEvent]
    assert events[-1].content == "Short answer."
    assert [c["finish"] for c in events[-1].usage["timings"]["llm_calls"]] == ["length", "length_retry"]
    retry = client.calls[1]
    assert "tools" not in retry
    assert retry["reasoning"] == {"effort": "none"}
    assert retry["max_output_tokens"] == 100


def test_content_filter_stop_ends_on_an_error_not_done():
    client = FakeResponsesClient([[text("Partial"), incomplete("content_filter")]])

    events = run(client)

    assert [type(e) for e in events] == [TokenEvent, ErrorEvent]
    assert "content filter" in events[-1].message


def test_a_content_filter_error_event_is_reported_as_a_filter_block():
    client = FakeResponsesClient([[SimpleNamespace(type="error", message="flagged", code="content_filter")]])

    events = run(client)

    assert [type(e) for e in events] == [ErrorEvent]
    assert "content filter" in events[0].message
    assert "flagged" not in events[0].message


def test_a_failed_response_ends_on_an_error():
    failed = SimpleNamespace(
        type="response.failed",
        response=SimpleNamespace(error=SimpleNamespace(code="server_error", message="upstream broke")),
    )
    client = FakeResponsesClient([[failed]])

    events = run(client)

    assert [type(e) for e in events] == [ErrorEvent]
    assert events[0].message == "upstream broke"


# ---------------------------------------------------------------------------
# chat_completions: the provider's tools + reasoning_effort refusal names the fix
# ---------------------------------------------------------------------------


class _BadRequest(Exception):
    status_code = 400


def test_tools_with_reasoning_rejection_names_the_config_fix():
    exc = _BadRequest(
        "Error code: 400 - Function tools with reasoning_effort are not supported for gpt-6-luna in "
        "/v1/chat/completions. To use function tools, use /v1/responses or set reasoning_effort to 'none'."
    )

    def boom(**kwargs):
        raise exc

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=boom)))
    domain = DomainConfig(key="test", title="t", system_prompt="s", tool_definitions=TOOLS, dispatch=lambda n, a: "{}")

    events = list(
        stream_chat(client, "gpt-6-luna", domain, "hi", [], RequestIdentity(user_id="u1"), reasoning_effort="low")
    )

    assert [type(e) for e in events] == [ErrorEvent]
    assert "LLM_API=responses" in events[0].message
    assert "LLM_API=chat_completions" in events[0].message
    assert "Error code" not in events[0].message
