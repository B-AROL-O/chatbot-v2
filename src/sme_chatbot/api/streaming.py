"""``ChatEvent`` -> Server-Sent Event mapping, per plan §4.1's SSE table.

Kept as pure functions (no FastAPI/request state) so it's trivially unit
testable independent of the app wiring in ``main.py``.
"""

from __future__ import annotations

import json

from sse_starlette.sse import ServerSentEvent

from sme_chatbot.core.events import (
    ChatEvent,
    Citation,
    DoneEvent,
    ErrorEvent,
    TokenEvent,
    ToolCallEvent,
    ToolResultEvent,
)


def _citation_dict(citation: Citation) -> dict:
    return {
        "source": citation.source,
        "page": citation.page,
        "document_id": citation.document_id,
        "distance": citation.distance,
        "filename": citation.filename,
        "repository": citation.repository,
    }


def event_to_sse(event: ChatEvent) -> ServerSentEvent:
    """Map one ``ChatEvent`` to the ``ServerSentEvent`` wire shape from §4.1."""
    if isinstance(event, TokenEvent):
        return ServerSentEvent(event="token", data=json.dumps({"text": event.text}))

    if isinstance(event, ToolCallEvent):
        return ServerSentEvent(
            event="tool_call",
            data=json.dumps({"name": event.name, "arguments": event.arguments}),
        )

    if isinstance(event, ToolResultEvent):
        return ServerSentEvent(
            event="tool_result",
            data=json.dumps(
                {
                    "name": event.name,
                    "ok": event.ok,
                    "citations": [_citation_dict(c) for c in event.citations],
                }
            ),
        )

    if isinstance(event, DoneEvent):
        return ServerSentEvent(
            event="done",
            data=json.dumps(
                {
                    "content": event.content,
                    "citations": [_citation_dict(c) for c in event.citations],
                    "usage": event.usage,
                }
            ),
        )

    if isinstance(event, ErrorEvent):
        return ServerSentEvent(event="error", data=json.dumps({"message": event.message}))

    raise TypeError(f"Unknown ChatEvent type: {type(event)!r}")


def error_sse(message: str) -> ServerSentEvent:
    """Build a terminal ``error`` SSE event for cases ``stream_chat`` itself
    can't report (e.g. an exception raised by the glue code around it)."""
    return ServerSentEvent(event="error", data=json.dumps({"message": message}))
