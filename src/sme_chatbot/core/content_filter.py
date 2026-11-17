"""Recognise an LLM provider's content-filter rejection.

Azure OpenAI screens every prompt and answers a flagged one with HTTP 400,
``code: "content_filter"``, and a per-category breakdown under
``innererror.content_filter_result``. On the machine manuals this produces
false positives: lubrication procedures have been flagged as "sexual" at
severity "medium" (eval items Q26, Q75 and Q169). The block is deterministic,
so retrying the same prompt does not help, and the raw error is a JSON dump no
user should see.

Azure can also filter the *answer* partway through a streamed response. That
raises nothing: the stream just ends with ``finish_reason: "content_filter"``,
and the chunk carries the breakdown under ``content_filter_results``.
``OutputFiltered`` turns that into an exception, so the chat loop handles both
cases in one place.

Duck-typed on the exception's ``body`` so this needs no ``openai`` import.
"""

from __future__ import annotations


def filtered_categories(exc: BaseException) -> list[str] | None:
    """``["sexual:medium", ...]`` for a content-filter rejection, else ``None``.

    An empty list means the provider reported a content-filter block without
    naming the category. Callers must test ``is not None``.
    """
    body = getattr(exc, "body", None)
    if not isinstance(body, dict):
        return None
    if isinstance(body.get("error"), dict):  # some clients keep the envelope
        body = body["error"]
    if body.get("code") != "content_filter":
        return None
    return categories_in((body.get("innererror") or {}).get("content_filter_result"))


def categories_in(result) -> list[str]:
    """The flagged entries of a ``content_filter_result(s)`` breakdown, e.g. ``["sexual:medium"]``."""
    if not isinstance(result, dict):
        return []
    return [
        f"{category}:{verdict.get('severity', 'unknown')}"
        for category, verdict in result.items()
        if isinstance(verdict, dict) and verdict.get("filtered")
    ]


class OutputFiltered(Exception):
    """The provider's content filter stopped a streamed answer partway through."""

    def __init__(self, categories: list[str]):
        super().__init__(describe_block(categories))
        self.categories = categories


def describe_block(categories: list[str]) -> str:
    """Human-readable summary, e.g. ``"blocked by the LLM provider's content filter (sexual:medium)"``."""
    flagged = ", ".join(categories) or "category not reported"
    return f"blocked by the LLM provider's content filter ({flagged})"
