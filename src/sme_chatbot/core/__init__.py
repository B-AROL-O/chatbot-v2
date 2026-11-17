"""Framework-agnostic chat loop and domain registry (WP2)."""

from .chat import stream_chat
from .domains import DOMAINS, DomainConfig
from .events import (
    ChatEvent,
    Citation,
    DoneEvent,
    ErrorEvent,
    RequestIdentity,
    TokenEvent,
    ToolCallEvent,
    ToolResultEvent,
)

__all__ = [
    "ChatEvent",
    "Citation",
    "DOMAINS",
    "DomainConfig",
    "DoneEvent",
    "ErrorEvent",
    "RequestIdentity",
    "TokenEvent",
    "ToolCallEvent",
    "ToolResultEvent",
    "stream_chat",
]
