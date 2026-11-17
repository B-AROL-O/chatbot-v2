"""Tests for the Gradio client's SSE handling.

The reported symptom was answers that showed a citation footer and no answer
text. The frontend half of that bug is here: the ``done`` event's ``content``
was trusted unconditionally, and an empty ``content`` (always present in the
payload, so the ``.get(..., accumulated)`` default never fired) replaced a
fully streamed answer with nothing but the footer.

Drives ``stream_chat_response`` against a scripted fake ``httpx.Client`` - no
network, no backend, no LLM.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "frontends"))

import gradio_app
from gradio_app import _NO_ANSWER_MESSAGE, stream_chat_response


def _sse(event: str, payload: dict) -> list[str]:
    """One SSE frame as ``iter_lines()`` would deliver it (no trailing \\n)."""
    return [f"event: {event}", f"data: {json.dumps(payload)}", ""]


class _FakeResponse:
    def __init__(self, lines: list[str], status_code: int = 200):
        self.status_code = status_code
        self._lines = lines

    def iter_lines(self):
        return iter(self._lines)

    def read(self) -> bytes:  # only used on the non-200 path
        return b"boom"

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeClient:
    """Stand-in for ``httpx.Client``: context manager exposing ``.stream()``."""

    def __init__(self, lines: list[str], status_code: int = 200):
        self._lines = lines
        self._status_code = status_code
        self.requests: list[dict] = []

    def __call__(self, *args, **kwargs):  # httpx.Client(timeout=...)
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def stream(self, method, url, json=None, headers=None):
        self.requests.append({"method": method, "url": url, "json": json})
        return _FakeResponse(self._lines, self._status_code)


@pytest.fixture
def run_stream(monkeypatch):
    """Run ``stream_chat_response`` over a scripted SSE line list."""

    def _run(lines: list[str]) -> list[str]:
        monkeypatch.setattr(gradio_app.httpx, "Client", _FakeClient(lines))
        return list(stream_chat_response("domanda", [], "documents"))

    return _run


CITATIONS = [
    {"filename": "90001_manual_IT.pdf", "page": 19, "distance": 0.2},
    {"filename": "90001_spare_parts_catalogue_IT.pdf", "page": 57, "distance": 0.4},
]


def test_empty_done_content_does_not_wipe_the_streamed_answer(run_stream):
    """The footer must not replace the whole answer.

    The backend now accumulates across tool rounds, so this shape should no
    longer reach the client - but a stale backend, or any future round-scoped
    regression, must not cost the user their answer.
    """
    lines = [
        *_sse("token", {"text": "M12 = 47 Nm "}),
        *_sse("token", {"text": "(90001_manual_IT.pdf, p. 9)."}),
        *_sse("done", {"content": "", "citations": CITATIONS, "usage": {"rounds": 3}}),
    ]

    final = run_stream(lines)[-1]

    assert final.startswith("M12 = 47 Nm (90001_manual_IT.pdf, p. 9).")
    assert "— fonte: 90001_manual_IT.pdf, p. 19" in final
    assert _NO_ANSWER_MESSAGE not in final


def test_done_content_is_used_when_present(run_stream):
    lines = [
        *_sse("token", {"text": "M12 = 47 Nm."}),
        *_sse("done", {"content": "M12 = 47 Nm.", "citations": [], "usage": {}}),
    ]

    assert run_stream(lines)[-1] == "M12 = 47 Nm."


def test_no_answer_at_all_says_so_instead_of_showing_a_bare_footer(run_stream):
    """Sources with no answer text is the symptom users reported.

    When the turn really produced nothing (e.g. the tool-calling loop hit its
    round cap without answering), say that, rather than rendering a citation
    footer under an empty message.
    """
    lines = [*_sse("done", {"content": "", "citations": CITATIONS, "usage": {"rounds": 6}})]

    final = run_stream(lines)[-1]

    assert final.startswith(_NO_ANSWER_MESSAGE)
    # The consulted sources are still listed, so the turn stays diagnosable.
    assert "— fonte: 90001_manual_IT.pdf, p. 19" in final


def test_error_event_is_surfaced_verbatim(run_stream):
    lines = [*_sse("error", {"message": "LLM non raggiungibile"})]

    assert run_stream(lines) == ["LLM non raggiungibile"]


def test_an_error_after_tokens_keeps_the_partial_answer(run_stream):
    """A content-filter stop mid-answer must not wipe the text already shown."""
    lines = [
        *_sse("token", {"text": "Prima del primo avviamento"}),
        *_sse("error", {"message": "The answer was stopped partway"}),
    ]

    assert run_stream(lines)[-1] == "Prima del primo avviamento\n\n⚠️ The answer was stopped partway"
