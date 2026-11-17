"""Shared fixtures for the unit suite."""

from __future__ import annotations

import httpx
import openai
import pytest

# The body Azure OpenAI returned for eval item Q26's judge prompt on 2026-10-26.
AZURE_CONTENT_FILTER_BODY = {
    "message": (
        "The response was filtered due to the prompt triggering Azure OpenAI's content management policy. "
        "Please modify your prompt and retry."
    ),
    "type": None,
    "param": "prompt",
    "code": "content_filter",
    "status": 400,
    "innererror": {
        "code": "ResponsibleAIPolicyViolation",
        "content_filter_result": {
            "hate": {"filtered": False, "severity": "safe"},
            "jailbreak": {"detected": False, "filtered": False},
            "self_harm": {"filtered": False, "severity": "safe"},
            "sexual": {"filtered": True, "severity": "medium"},
            "violence": {"filtered": False, "severity": "safe"},
        },
    },
}


@pytest.fixture
def azure_content_filter_error() -> openai.BadRequestError:
    """A real ``openai.BadRequestError`` built from the response Azure sends, not a look-alike."""
    response = httpx.Response(
        400,
        json={"error": AZURE_CONTENT_FILTER_BODY},
        request=httpx.Request("POST", "https://llm.invalid/openai/v1/chat/completions"),
    )
    return openai.BadRequestError(
        f"Error code: 400 - {{'error': {AZURE_CONTENT_FILTER_BODY!r}}}",
        response=response,
        body=AZURE_CONTENT_FILTER_BODY,
    )
