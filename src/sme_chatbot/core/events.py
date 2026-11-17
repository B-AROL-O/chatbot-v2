"""Chat event dataclasses shared by the streaming loop and the API layer.

Frozen shape per ``docs/specs/2026-07-02-backend-split-and-docsvc-ingestion-plan.md``
§4.1 (Chat API / SSE event types) and §4.2 (``RequestIdentity``). ``core/chat.py``
yields these; ``api/streaming.py`` (WP3) maps them onto SSE ``event:``/``data:``
pairs. Deliberately free of any web-framework import.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Citation:
    """A single source reference extracted from a RAG tool result.

    ``source`` is always the *human-readable label* of the cited document —
    its filename — whichever tool produced it. ``filename`` carries the same
    value when the tool reported one explicitly, and ``repository`` names the
    origin store (e.g. a network share), which is metadata *about* the
    document, never a substitute for its name.

    ``document_id`` is ``None`` for chunks coming from the two legacy Chroma
    collections (``cdo_docs``, ``manuals``); it is populated for the
    DocumentService-backed ``documents`` domain.
    """

    source: str | None
    page: int | None
    document_id: str | None = None
    distance: float | None = None
    filename: str | None = None
    repository: str | None = None


@dataclass
class TokenEvent:
    """A streamed assistant-text delta, already ``<think>``-stripped."""

    text: str


@dataclass
class ToolCallEvent:
    """Emitted when the model requests a tool call, before it is dispatched."""

    name: str
    arguments: dict


@dataclass
class ToolResultEvent:
    """Emitted after a tool call is dispatched.

    The full tool result payload is intentionally NOT included (per §4.1) —
    only whether it succeeded and any citations extracted from it.
    """

    name: str
    ok: bool
    citations: list[Citation] = field(default_factory=list)


@dataclass
class DoneEvent:
    """Terminal event on success: the full answer, aggregated citations, usage."""

    content: str
    citations: list[Citation] = field(default_factory=list)
    usage: dict = field(default_factory=dict)


@dataclass
class ErrorEvent:
    """Terminal event on failure — sent instead of ``DoneEvent``."""

    message: str


# Union of everything ``stream_chat`` can yield.
ChatEvent = TokenEvent | ToolCallEvent | ToolResultEvent | DoneEvent | ErrorEvent


@dataclass
class RequestIdentity:
    """Caller identity resolved by the API auth layer (§4.2).

    Handlers and tools receive this only — they never read headers or tokens
    directly. ``permitted_document_ids=None`` means unrestricted (this phase's
    default for both ``AUTH_MODE=dev`` and the stubbed ``AUTH_MODE=jwt`` claim
    mapping).
    """

    user_id: str | None = None
    permitted_document_ids: list[str] | None = None
    scopes: dict[str, str] = field(default_factory=dict)
