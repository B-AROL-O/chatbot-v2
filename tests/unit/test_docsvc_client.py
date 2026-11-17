"""respx-mocked tests for `sme_chatbot.clients.docsvc` (WP8 / WP5 acceptance).

No network, no MSAL, no real creds: `TokenProvider` is faked out entirely for
the client-behaviour tests, and the one test that exercises the real
`TokenProvider` (missing creds -> `DocSvcAuthError`) never reaches MSAL
because the guard in `_ensure_app` fires first.

See docs/specs/2026-07-02-backend-split-and-docsvc-ingestion-plan.md
§4.4 (REST contract), §4.5 (env vars), §5 WP5 (acceptance).

All fixtures/fakes live in this file per the WP8a file-ownership split (no
`tests/unit/conftest.py`).
"""

from __future__ import annotations

import httpx
import pytest
import respx

from sme_chatbot.clients import docsvc
from sme_chatbot.clients.docsvc import (
    DocSvcAuthError,
    DocSvcClient,
    DocSvcError,
    DocSvcNotReady,
    DocSvcSettings,
    TokenProvider,
)

BASE_URL = "https://documents-api-test.example.com"


def make_settings(**overrides) -> DocSvcSettings:
    defaults = dict(
        base_url=BASE_URL,
        verify_ssl=True,
        tenant_id="tenant-1",
        authority="https://login.microsoftonline.com/tenant-1",
        audience="api://svc",
        scope="api://svc/.default",
        client_id="client-1",
        client_secret="secret-1",
        user_id="",
    )
    defaults.update(overrides)
    return DocSvcSettings(**defaults)


class FakeTokenProvider:
    """Stand-in for `TokenProvider` so client-behaviour tests never touch MSAL."""

    def __init__(self, token="fake-token-123"):
        self.token = token
        self.calls = 0

    def get_token(self):
        self.calls += 1
        return self.token


@pytest.fixture(autouse=True)
def no_real_sleep(monkeypatch):
    """Retry backoff would otherwise really sleep ~0.5s+1s per retrying test."""
    monkeypatch.setattr(docsvc.time, "sleep", lambda *_: None)


# ---------------------------------------------------------------------------
# Bearer token injected into every API call
# ---------------------------------------------------------------------------


def test_bearer_token_injected_into_requests():
    settings = make_settings()
    token_provider = FakeTokenProvider()
    with respx.mock(base_url=BASE_URL) as mock:
        route = mock.get("/api/v1/documents/doc-1").mock(
            return_value=httpx.Response(200, json={"documentId": "doc-1"})
        )
        client = DocSvcClient(settings=settings, token_provider=token_provider)
        doc = client.get_document("doc-1")

    assert doc == {"documentId": "doc-1"}
    assert route.calls.last.request.headers["Authorization"] == "Bearer fake-token-123"
    assert token_provider.calls == 1


def test_user_id_header_sent_when_configured():
    settings = make_settings(user_id="user-42")
    token_provider = FakeTokenProvider()
    with respx.mock(base_url=BASE_URL) as mock:
        route = mock.get("/api/v1/documents/doc-1").mock(
            return_value=httpx.Response(200, json={"documentId": "doc-1"})
        )
        client = DocSvcClient(settings=settings, token_provider=token_provider)
        client.get_document("doc-1")

    assert route.calls.last.request.headers["X-User-Id"] == "user-42"


def test_user_id_header_absent_when_not_configured():
    settings = make_settings(user_id="")
    token_provider = FakeTokenProvider()
    with respx.mock(base_url=BASE_URL) as mock:
        route = mock.get("/api/v1/documents/doc-1").mock(
            return_value=httpx.Response(200, json={"documentId": "doc-1"})
        )
        client = DocSvcClient(settings=settings, token_provider=token_provider)
        client.get_document("doc-1")

    assert "X-User-Id" not in route.calls.last.request.headers


# ---------------------------------------------------------------------------
# Pagination walks all pages transparently
# ---------------------------------------------------------------------------


def test_pagination_walks_three_pages():
    settings = make_settings()
    token_provider = FakeTokenProvider()

    def responder(request):
        page = int(request.url.params.get("page", "1"))
        assert int(request.url.params["pageSize"]) == 2
        data = [{"documentId": f"doc-{page}-a"}, {"documentId": f"doc-{page}-b"}]
        return httpx.Response(
            200,
            json={
                "data": data,
                "pagination": {"page": page, "pageSize": 2, "totalItems": 6, "totalPages": 3},
            },
        )

    with respx.mock(base_url=BASE_URL) as mock:
        route = mock.get("/api/v1/documents").mock(side_effect=responder)
        client = DocSvcClient(settings=settings, token_provider=token_provider)
        docs = list(client.iter_documents(page_size=2))

    assert [d["documentId"] for d in docs] == [
        "doc-1-a",
        "doc-1-b",
        "doc-2-a",
        "doc-2-b",
        "doc-3-a",
        "doc-3-b",
    ]
    assert len(route.calls) == 3


def test_pagination_page_size_capped_at_max():
    settings = make_settings()
    token_provider = FakeTokenProvider()

    with respx.mock(base_url=BASE_URL) as mock:
        route = mock.get("/api/v1/documents").mock(
            return_value=httpx.Response(
                200,
                json={"data": [], "pagination": {"page": 1, "pageSize": 100, "totalItems": 0, "totalPages": 1}},
            )
        )
        client = DocSvcClient(settings=settings, token_provider=token_provider)
        list(client.iter_documents(page_size=500))

    assert int(route.calls.last.request.url.params["pageSize"]) == 100


def test_iter_documents_forwards_source_category_and_tag_filters():
    settings = make_settings()
    token_provider = FakeTokenProvider()

    with respx.mock(base_url=BASE_URL) as mock:
        route = mock.get("/api/v1/documents").mock(
            return_value=httpx.Response(
                200,
                json={"data": [], "pagination": {"page": 1, "pageSize": 50, "totalItems": 0, "totalPages": 1}},
            )
        )
        client = DocSvcClient(settings=settings, token_provider=token_provider)
        list(client.iter_documents(source="Sharepoint", category="Manual", tag="serial:91123"))

    params = route.calls.last.request.url.params
    assert params["source"] == "Sharepoint"
    assert params["category"] == "Manual"
    assert params["tag"] == "serial:91123"


# ---------------------------------------------------------------------------
# Download-link 404/502 -> DocSvcNotReady, and NOT retried
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", [404, 502])
def test_download_link_not_ready_is_not_retried(status):
    settings = make_settings()
    token_provider = FakeTokenProvider()
    with respx.mock(base_url=BASE_URL) as mock:
        route = mock.get("/api/v1/documents/doc-x/download").mock(return_value=httpx.Response(status))
        client = DocSvcClient(settings=settings, token_provider=token_provider)
        with pytest.raises(DocSvcNotReady) as excinfo:
            client.get_download_link("doc-x")

    assert excinfo.value.status == status
    # Not retried: exactly one call even though 502 is otherwise a retryable
    # status for ordinary API calls.
    assert len(route.calls) == 1


# ---------------------------------------------------------------------------
# One 500 then success -> retry works (ordinary API calls ARE retried)
# ---------------------------------------------------------------------------


def test_retry_on_one_500_then_success():
    settings = make_settings()
    token_provider = FakeTokenProvider()
    with respx.mock(base_url=BASE_URL) as mock:
        route = mock.get("/api/v1/documents/doc-y").mock(
            side_effect=[httpx.Response(500), httpx.Response(200, json={"documentId": "doc-y"})]
        )
        client = DocSvcClient(settings=settings, token_provider=token_provider)
        doc = client.get_document("doc-y")

    assert doc == {"documentId": "doc-y"}
    assert len(route.calls) == 2


def test_retries_exhausted_raises_docsvc_error():
    settings = make_settings()
    token_provider = FakeTokenProvider()
    with respx.mock(base_url=BASE_URL) as mock:
        route = mock.get("/api/v1/documents/doc-z").mock(return_value=httpx.Response(500))
        client = DocSvcClient(settings=settings, token_provider=token_provider)
        with pytest.raises(DocSvcError):
            client.get_document("doc-z")

    # RETRY_ATTEMPTS = 3: the call is attempted 3 times total before giving up.
    assert len(route.calls) == 3


# ---------------------------------------------------------------------------
# Missing creds -> DocSvcAuthError (no MSAL/network touched)
# ---------------------------------------------------------------------------


def test_missing_client_credentials_raise_auth_error():
    settings = make_settings(client_id="", client_secret="")
    provider = TokenProvider(settings=settings)
    with pytest.raises(DocSvcAuthError):
        provider.get_token()


def test_missing_authority_raises_auth_error():
    settings = make_settings(authority="")
    provider = TokenProvider(settings=settings)
    with pytest.raises(DocSvcAuthError):
        provider.get_token()


# ---------------------------------------------------------------------------
# download_content: presigned URL fetched with NO Authorization header
# ---------------------------------------------------------------------------


def test_download_content_streams_presigned_url_without_auth_header(tmp_path):
    settings = make_settings()
    token_provider = FakeTokenProvider()
    blob_url = "https://blobstore.example.com/container/file.pdf?sig=abc"

    with respx.mock() as mock:
        mock.get(f"{BASE_URL}/api/v1/documents/doc-z/download").mock(
            return_value=httpx.Response(
                200,
                json={"url": blob_url, "expiresAt": "2026-01-01T00:00:00Z", "versionId": "v1"},
            )
        )
        blob_route = mock.get(blob_url).mock(return_value=httpx.Response(200, content=b"hello world"))

        client = DocSvcClient(settings=settings, token_provider=token_provider)
        dest = tmp_path / "out.pdf"
        result = client.download_content("doc-z", dest)

    assert result == dest
    assert dest.read_bytes() == b"hello world"
    assert "Authorization" not in blob_route.calls.last.request.headers
    # The download-link API call itself IS authenticated.
    assert token_provider.calls == 1


def test_download_content_creates_parent_directories(tmp_path):
    settings = make_settings()
    token_provider = FakeTokenProvider()
    blob_url = "https://blobstore.example.com/container/file.pdf?sig=abc"

    with respx.mock() as mock:
        mock.get(f"{BASE_URL}/api/v1/documents/doc-z/download").mock(
            return_value=httpx.Response(
                200, json={"url": blob_url, "expiresAt": "2026-01-01T00:00:00Z", "versionId": "v1"}
            )
        )
        mock.get(blob_url).mock(return_value=httpx.Response(200, content=b"data"))

        client = DocSvcClient(settings=settings, token_provider=token_provider)
        dest = tmp_path / "nested" / "dir" / "out.pdf"
        client.download_content("doc-z", dest)

    assert dest.exists()
    assert dest.read_bytes() == b"data"


def test_download_content_propagates_not_ready_from_link_lookup(tmp_path):
    """If the link itself isn't ready (404/502), `download_content` must
    surface `DocSvcNotReady` rather than attempting to stream a blob."""
    settings = make_settings()
    token_provider = FakeTokenProvider()

    with respx.mock(base_url=BASE_URL) as mock:
        mock.get("/api/v1/documents/doc-z/download").mock(return_value=httpx.Response(404))
        client = DocSvcClient(settings=settings, token_provider=token_provider)
        with pytest.raises(DocSvcNotReady):
            client.download_content("doc-z", tmp_path / "out.pdf")
