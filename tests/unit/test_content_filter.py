"""Recognising an LLM provider's content-filter rejection."""

from __future__ import annotations

import httpx
import openai

from sme_chatbot.core.content_filter import describe_block, filtered_categories


def test_an_azure_content_filter_rejection_names_the_flagged_category(azure_content_filter_error):
    assert filtered_categories(azure_content_filter_error) == ["sexual:medium"]


def test_the_error_envelope_is_unwrapped_too(azure_content_filter_error):
    class Wrapped(Exception):
        body = {"error": azure_content_filter_error.body}

    assert filtered_categories(Wrapped()) == ["sexual:medium"]


def test_a_block_without_a_category_breakdown_is_still_a_block():
    class Bare(Exception):
        body = {"code": "content_filter", "message": "filtered"}

    categories = filtered_categories(Bare())
    assert categories == []  # not None: callers test `is not None`
    assert describe_block(categories) == "blocked by the LLM provider's content filter (category not reported)"


def test_other_errors_are_not_mistaken_for_a_block():
    response = httpx.Response(400, request=httpx.Request("POST", "https://llm.invalid/v1/chat/completions"))
    temperature = openai.BadRequestError(
        "Unsupported value: 'temperature'",
        response=response,
        body={"code": "unsupported_value", "param": "temperature"},
    )
    assert filtered_categories(temperature) is None
    assert filtered_categories(TimeoutError("timed out")) is None
