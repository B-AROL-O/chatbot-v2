"""Which OpenAI API the chat loop streams from.

``chat_completions`` (the default) calls ``/v1/chat/completions`` exactly as the loop
always has, so Ollama and other OpenAI-compatible servers need no change.

``responses`` calls ``/v1/responses`` instead. Azure has rejected function tools combined
with ``reasoning_effort`` on ``/v1/chat/completions`` for gpt-6 models and pointed at
``/v1/responses``. The adapter here takes the loop's chat-completions messages and
settings, and turns the Responses event stream back into chat-completions-shaped chunks
(``chunk.choices[0].delta.content`` / ``.delta.tool_calls`` / ``.finish_reason``), so
``stream_chat``'s tool-call accumulation and ``ChatEvent`` contract stay as they are.

The calls are stateless (``store=False``): the loop sends the whole conversation every
time, as it does with chat completions. Reasoning items are not sent back between tool
rounds, so the model starts each round's reasoning afresh, as it does today.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from types import SimpleNamespace

logger = logging.getLogger(__name__)

CHAT_COMPLETIONS = "chat_completions"
RESPONSES = "responses"
LLM_APIS = (CHAT_COMPLETIONS, RESPONSES)


class ResponsesStreamError(Exception):
    """A Responses stream ended with ``error`` or ``response.failed`` instead of an exception.

    ``body`` mirrors an ``openai.APIStatusError``'s, so ``content_filter.filtered_categories``
    recognises a content-filter failure here too.
    """

    def __init__(self, message: str, code: str | None = None):
        super().__init__(message)
        self.body = {"code": code, "message": message}


def open_stream(
    client, llm_api: str, model: str, messages: list[dict], tools: list[dict] | None, gen_params: dict
) -> Iterator:
    """Start one streaming LLM call; yields chat-completions-shaped chunks either way."""
    if llm_api == RESPONSES:
        return _responses_stream(client, model, messages, tools, gen_params)
    if tools is None:
        return client.chat.completions.create(model=model, messages=messages, stream=True, **gen_params)
    return client.chat.completions.create(model=model, messages=messages, tools=tools, stream=True, **gen_params)


def _to_input(messages: list[dict]) -> list[dict]:
    """Chat-completions messages -> Responses ``input`` items."""
    items: list[dict] = []
    for msg in messages:
        role = msg.get("role")
        if role == "tool":
            items.append({"type": "function_call_output", "call_id": msg["tool_call_id"], "output": msg["content"]})
            continue
        if msg.get("content"):
            items.append({"role": role, "content": msg["content"]})
        for tc in msg.get("tool_calls") or []:
            items.append(
                {
                    "type": "function_call",
                    "call_id": tc["id"],
                    "name": tc["function"]["name"],
                    "arguments": tc["function"]["arguments"],
                }
            )
    return items


def _to_tools(tools: list[dict]) -> list[dict]:
    """Chat-completions tool definitions -> Responses function tools.

    ``strict`` is sent as ``False``: the Responses API defaults it to true, which requires
    every property to be listed as required, and our tools have optional arguments.
    """
    return [
        {
            "type": "function",
            "name": t["function"]["name"],
            "description": t["function"].get("description"),
            "parameters": t["function"].get("parameters"),
            "strict": t["function"].get("strict", False),
        }
        for t in tools
    ]


def _to_params(gen_params: dict) -> dict:
    """The loop's chat-completions settings -> their Responses names."""
    params: dict = {}
    for key, value in gen_params.items():
        if key == "max_tokens":
            params["max_output_tokens"] = value
        elif key == "reasoning_effort":
            params["reasoning"] = {"effort": value}
        elif key == "seed":
            # /v1/responses has no seed; LLM_SEED only applies to chat completions.
            logger.debug("LLM_API=responses: seed=%s is not sent (the Responses API has no seed)", value)
        else:
            params[key] = value
    return params


def _chunk(content=None, tool_calls=None, finish_reason=None):
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=finish_reason)])


def _tool_call(index: int, id_=None, name=None, arguments=None):
    return SimpleNamespace(index=index, id=id_, function=SimpleNamespace(name=name, arguments=arguments))


def _responses_stream(client, model, messages, tools, gen_params) -> Iterator:
    kwargs: dict = {"model": model, "input": _to_input(messages), "stream": True, "store": False}
    if tools is not None:
        kwargs["tools"] = _to_tools(tools)
    kwargs.update(_to_params(gen_params))

    # Function calls by output_index. A call's arguments normally arrive as deltas; one
    # that got none (a server that sends them only on output_item.done) is sent whole then.
    calls: dict[int, bool] = {}
    for event in client.responses.create(**kwargs):
        kind = getattr(event, "type", None)
        if kind == "response.output_text.delta":
            if event.delta:
                yield _chunk(content=event.delta)
        elif kind == "response.output_item.added" and event.item.type == "function_call":
            calls[event.output_index] = False
            yield _chunk(tool_calls=[_tool_call(event.output_index, id_=event.item.call_id, name=event.item.name)])
        elif kind == "response.function_call_arguments.delta":
            if event.delta:
                calls[event.output_index] = True
                yield _chunk(tool_calls=[_tool_call(event.output_index, arguments=event.delta)])
        elif kind == "response.output_item.done" and event.item.type == "function_call":
            if not calls.get(event.output_index) and event.item.arguments:
                calls[event.output_index] = True
                yield _chunk(tool_calls=[_tool_call(event.output_index, arguments=event.item.arguments)])
        elif kind == "response.completed":
            yield _chunk(finish_reason="tool_calls" if calls else "stop")
        elif kind == "response.incomplete":
            details = event.response.incomplete_details
            reason = getattr(details, "reason", None)
            if reason == "max_output_tokens":
                yield _chunk(finish_reason="length")
            elif reason == "content_filter":
                yield _chunk(finish_reason="content_filter")
            else:
                yield _chunk(finish_reason="tool_calls" if calls else "stop")
        elif kind == "response.failed":
            error = event.response.error
            raise ResponsesStreamError(
                getattr(error, "message", None) or "The LLM response failed", getattr(error, "code", None)
            )
        elif kind == "error":
            raise ResponsesStreamError(event.message, event.code)
        # Anything else (reasoning items, content parts, ...) carries nothing the loop uses.
