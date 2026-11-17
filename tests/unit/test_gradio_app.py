"""Unit tests for the Gradio frontend's history mapping.

Gradio 6 delivers each history turn's ``content`` as a list of content
blocks (``[{"text": ..., "type": "text"}]``), not a plain string. The
fixtures below use the exact shape observed on the wire from Gradio 6.19
(captured via a browser network trace while reproducing the bug).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "frontends"))

from gradio_app import _to_backend_history


def _turn(role: str, text: str) -> dict:
    """A history turn exactly as Gradio 6.19 delivers it to the chat fn."""
    return {
        "role": role,
        "metadata": None,
        "content": [{"text": text, "type": "text"}],
        "options": None,
    }


def test_gradio6_content_blocks_are_forwarded():
    """Content-block turns must not be dropped."""
    history = [
        _turn("user", "Quante macchine ha ordinato ACME?"),
        _turn("assistant", "ACME ha ordinato 3 macchine."),
    ]
    assert _to_backend_history(history) == [
        {"role": "user", "content": "Quante macchine ha ordinato ACME?"},
        {"role": "assistant", "content": "ACME ha ordinato 3 macchine."},
    ]


def test_plain_string_content_still_supported():
    history = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]
    assert _to_backend_history(history) == history


def test_citation_footer_stripped_from_content_block_turns():
    history = [_turn("assistant", "Vedi il manuale.\n\n— fonte: manual.pdf p.12")]
    assert _to_backend_history(history) == [
        {"role": "assistant", "content": "Vedi il manuale."}
    ]


def test_multiple_text_blocks_are_joined():
    history = [
        {
            "role": "user",
            "content": [
                {"text": "part one ", "type": "text"},
                {"text": "part two", "type": "text"},
            ],
        }
    ]
    assert _to_backend_history(history) == [
        {"role": "user", "content": "part one part two"}
    ]


def test_textless_and_foreign_turns_are_dropped():
    history = [
        {"role": "user", "content": [{"file": {"path": "x.pdf"}, "type": "file"}]},
        {"role": "system", "content": "not forwarded"},
        {"role": "user", "content": None},
        _turn("user", "kept"),
    ]
    assert _to_backend_history(history) == [{"role": "user", "content": "kept"}]


def test_mixed_file_and_text_blocks_keep_the_text():
    history = [
        {
            "role": "user",
            "content": [
                {"file": {"path": "x.pdf"}, "type": "file"},
                {"text": "please summarize this", "type": "text"},
            ],
        }
    ]
    assert _to_backend_history(history) == [
        {"role": "user", "content": "please summarize this"}
    ]
