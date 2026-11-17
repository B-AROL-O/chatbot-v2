"""Tests for the Gradio client's ``GET /health``-derived startup state:
domain list, and the deployed backend version shown in the UI description.

``fetch_health`` backs both ``fetch_domains`` (kept for standalone use) and
``build_app``'s version display with a single request - no second round trip
just to read one more field off the same response.

Drives against a scripted fake ``httpx.get`` - no network, no backend.
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "frontends"))

import gradio_app
from gradio_app import (
    FALLBACK_DOMAINS,
    build_app,
    fetch_domains,
    fetch_health,
)


class _FakeResponse:
    def __init__(self, json_body: dict, status_code: int = 200):
        self._json = json_body
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("boom", request=None, response=self)

    def json(self) -> dict:
        return self._json


def _fake_get(json_body: dict, status_code: int = 200):
    def _get(url, timeout=None):
        return _FakeResponse(json_body, status_code)

    return _get


def _raising_get(exc: Exception):
    def _get(url, timeout=None):
        raise exc

    return _get


# ---------------------------------------------------------------------------
# fetch_health / fetch_domains
# ---------------------------------------------------------------------------


def test_fetch_health_returns_domains_and_version_in_one_request(monkeypatch):
    calls = []

    def _get(url, timeout=None):
        calls.append(url)
        return _FakeResponse({"status": "ok", "domains": ["sample_manuals"], "version": "0.5.3"})

    monkeypatch.setattr(gradio_app.httpx, "get", _get)

    domains, version = fetch_health("http://backend")

    assert domains == ["sample_manuals"]
    assert version == "0.5.3"
    assert len(calls) == 1


def test_fetch_health_falls_back_when_backend_unreachable(monkeypatch):
    monkeypatch.setattr(
        gradio_app.httpx, "get", _raising_get(httpx.ConnectError("refused"))
    )

    domains, version = fetch_health("http://backend")

    assert domains == list(FALLBACK_DOMAINS)
    assert version is None


def test_fetch_health_falls_back_domains_but_keeps_version_when_domains_missing(monkeypatch):
    """A malformed-but-reachable response: no domains, but version is real signal."""
    monkeypatch.setattr(
        gradio_app.httpx, "get", _fake_get({"status": "ok", "domains": [], "version": "0.5.3"})
    )

    domains, version = fetch_health("http://backend")

    assert domains == list(FALLBACK_DOMAINS)
    assert version == "0.5.3"


def test_fetch_domains_delegates_to_fetch_health(monkeypatch):
    monkeypatch.setattr(
        gradio_app.httpx,
        "get",
        _fake_get({"status": "ok", "domains": ["documents"], "version": "0.5.3"}),
    )

    assert fetch_domains("http://backend") == ["documents"]


# ---------------------------------------------------------------------------
# build_app description
# ---------------------------------------------------------------------------


def test_build_app_description_shows_the_backend_version(monkeypatch):
    monkeypatch.setattr(
        gradio_app.httpx,
        "get",
        _fake_get({"status": "ok", "domains": ["sample_manuals"], "version": "0.5.3"}),
    )

    demo = build_app("http://backend")

    assert "http://backend" in demo.description
    assert "v0.5.3" in demo.description


def test_build_app_description_shows_versione_sconosciuta_when_backend_down(monkeypatch):
    monkeypatch.setattr(
        gradio_app.httpx, "get", _raising_get(httpx.ConnectError("refused"))
    )

    demo = build_app("http://backend")

    assert "versione sconosciuta" in demo.description


@pytest.mark.parametrize("version", ["dev", "0.5.3", "0.5.3-rc.1"])
def test_build_app_description_formats_any_version_string(monkeypatch, version):
    monkeypatch.setattr(
        gradio_app.httpx,
        "get",
        _fake_get({"status": "ok", "domains": ["sample_manuals"], "version": version}),
    )

    demo = build_app("http://backend")

    assert f"v{version}" in demo.description
