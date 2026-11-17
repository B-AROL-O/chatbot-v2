"""Pydantic request/response models for the chat API.

Shapes are frozen by ``docs/specs/2026-07-02-backend-split-and-docsvc-ingestion-plan.md``
§4.1 — do not change field names/types here without re-checking that section.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """One turn of prior conversation history."""

    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    """``POST /chat/message`` request body."""

    domain: str
    message: str
    history: list[ChatMessage] = Field(default_factory=list)
    stream: bool = True


class Citation(BaseModel):
    """A single source reference, mirroring ``core.events.Citation``.

    ``filename``/``repository`` were added later; they are additive to
    the §4.1 shape, so existing clients that only read ``source`` keep working
    (``source`` now always carries the document name, never its network share).
    """

    source: str | None = None
    page: int | None = None
    document_id: str | None = None
    distance: float | None = None
    filename: str | None = None
    repository: str | None = None


class ChatResponse(BaseModel):
    """Non-streaming response body — same fields as the ``done`` SSE event."""

    content: str
    citations: list[Citation] = Field(default_factory=list)
    usage: dict = Field(default_factory=dict)
