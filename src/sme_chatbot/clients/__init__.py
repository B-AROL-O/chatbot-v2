"""REST clients for external services (DocumentService, etc.) (WP5)."""

from sme_chatbot.clients.docsvc import (
    DocSvcAuthError,
    DocSvcClient,
    DocSvcError,
    DocSvcNotReady,
    DocSvcSettings,
    TokenProvider,
)

__all__ = [
    "DocSvcAuthError",
    "DocSvcClient",
    "DocSvcError",
    "DocSvcNotReady",
    "DocSvcSettings",
    "TokenProvider",
]
