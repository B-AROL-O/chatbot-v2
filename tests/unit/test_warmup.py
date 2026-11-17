"""Tests for ``sme_chatbot.tools.warmup``: the startup warm-up
is opt-in and can never take the API down."""

from __future__ import annotations

import pytest

from sme_chatbot.tools import cdo_rag, docsvc_rag, manuals_rag, warmup


@pytest.fixture()
def calls(monkeypatch, tmp_path):
    """Record what the warm-up touches; no store or model is loaded."""
    seen: list = []
    monkeypatch.setattr(docsvc_rag, "warm_up", lambda read_files: seen.append(("documents", read_files)))
    for module in (cdo_rag, manuals_rag):
        monkeypatch.setattr(module, "CHROMA_DIR", tmp_path / module.__name__)
        monkeypatch.setattr(module, "_ensure_loaded", lambda m=module: seen.append(m.__name__))
    return seen


def test_off_touches_nothing(calls):
    warmup.warm_up("off")

    assert calls == []


@pytest.mark.parametrize(("mode", "read_files"), [("on", False), ("full", True)])
def test_documents_store_is_warmed_and_absent_stores_are_skipped(calls, mode, read_files):
    warmup.warm_up(mode)

    assert calls == [("documents", read_files)]


def test_existing_sample_stores_are_loaded(calls):
    manuals_rag.CHROMA_DIR.mkdir()

    warmup.warm_up("on")

    assert calls == [("documents", False), manuals_rag.__name__]


def test_a_failure_is_logged_not_raised(monkeypatch, caplog):
    def boom(read_files):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(docsvc_rag, "warm_up", boom)

    warmup.warm_up("on")

    assert "warm-up failed" in caplog.text
