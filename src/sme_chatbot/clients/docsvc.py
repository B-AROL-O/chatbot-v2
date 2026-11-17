"""DocumentService REST client (WP5).

Sync ``httpx``-based client for the DocumentService API
(at ``DOCSVC_BASE_URL``, prefix ``/api/v1/documents``), used by
the DocSvc ingestion pipeline (WP6/WP7).

Auth model: Entra ID **client-credentials** via MSAL
(``msal.ConfidentialClientApplication``), scope ``{DOCSVC_AUDIENCE}/.default``.
The ``DOCSVC_*`` env vars follow the usual derivation (authority derived from
tenant, scope derived from audience when not set explicitly). Those semantics
already live in :mod:`sme_chatbot.config` (single source of truth); this
module only reads the resulting constants, it does not re-derive them.

See ``docs/specs/2026-07-02-backend-split-and-docsvc-ingestion-plan.md``
§4.4 (REST contract), §4.5 (env vars) and §9 (presigned URLs expire fast;
``pageSize`` capped at 100).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import msal

from sme_chatbot import config

logger = logging.getLogger(__name__)

API_PREFIX = "/api/v1/documents"
DEFAULT_TIMEOUT = 30.0
MAX_PAGE_SIZE = 100
RETRY_ATTEMPTS = 3
RETRY_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
TOKEN_EXPIRY_MARGIN_SECONDS = 5 * 60


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class DocSvcError(Exception):
    """Base class for all DocumentService client errors."""


class DocSvcAuthError(DocSvcError):
    """Credentials/config are missing, or MSAL token acquisition failed."""


class DocSvcNotReady(DocSvcError):
    """Document content isn't downloadable yet (404/502 from ``/download``).

    Callers should log and skip the document rather than aborting a run.
    """

    def __init__(self, status: int):
        self.status = status
        super().__init__(f"DocumentService download link not ready (HTTP {status})")


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DocSvcSettings:
    """DocSvc connection settings, sourced from ``sme_chatbot.config``.

    ``config.py`` already implements the authority/scope derivation
    (``DOCSVC_AUTHORITY`` falls back to
    ``https://login.microsoftonline.com/{DOCSVC_TENANT_ID}``;
    ``DOCSVC_SCOPE`` falls back to ``{DOCSVC_AUDIENCE}/.default``) — this
    class does not fork that logic.
    """

    base_url: str
    verify_ssl: bool
    tenant_id: str
    authority: str
    audience: str
    scope: str
    client_id: str
    client_secret: str
    user_id: str

    @classmethod
    def from_env(cls) -> DocSvcSettings:
        return cls(
            base_url=config.DOCSVC_BASE_URL.rstrip("/"),
            verify_ssl=config.DOCSVC_VERIFY_SSL,
            tenant_id=config.DOCSVC_TENANT_ID,
            authority=config.DOCSVC_AUTHORITY,
            audience=config.DOCSVC_AUDIENCE,
            scope=config.DOCSVC_SCOPE,
            client_id=config.DOCSVC_READ_CLIENT_ID,
            client_secret=config.DOCSVC_READ_CLIENT_SECRET,
            user_id=config.DOCSVC_USER_ID,
        )


# ---------------------------------------------------------------------------
# Token acquisition
# ---------------------------------------------------------------------------
class TokenProvider:
    """Acquires and caches Entra ID access tokens via MSAL client-credentials.

    Re-acquires when fewer than ``TOKEN_EXPIRY_MARGIN_SECONDS`` remain on the
    cached token. MSAL keeps its own internal token cache too, so even the
    "re-acquire" path is nearly free once warm.
    """

    def __init__(self, settings: DocSvcSettings | None = None):
        self._settings = settings or DocSvcSettings.from_env()
        self._app: msal.ConfidentialClientApplication | None = None
        self._token: str | None = None
        self._expires_at: float = 0.0

    def _ensure_app(self) -> msal.ConfidentialClientApplication:
        if self._app is None:
            s = self._settings
            if not (s.base_url and s.client_id and s.client_secret and s.authority and s.scope):
                raise DocSvcAuthError(
                    "DocumentService credentials are not configured. Set "
                    "DOCSVC_BASE_URL, DOCSVC_TENANT_ID (or DOCSVC_AUTHORITY), DOCSVC_AUDIENCE "
                    "(or DOCSVC_SCOPE), and DOCSVC_READ_CLIENT_ID / "
                    "DOCSVC_READ_CLIENT_SECRET — see "
                    "docs/procedures/docsvc-ingestion.md."
                )
            self._app = msal.ConfidentialClientApplication(
                s.client_id, authority=s.authority, client_credential=s.client_secret
            )
        return self._app

    def get_token(self) -> str:
        now = time.monotonic()
        if self._token and now < self._expires_at - TOKEN_EXPIRY_MARGIN_SECONDS:
            return self._token

        app = self._ensure_app()
        try:
            result = app.acquire_token_for_client(scopes=[self._settings.scope])
        except Exception as exc:  # pragma: no cover - defensive, MSAL network errors
            raise DocSvcAuthError(f"Failed to acquire DocumentService token: {exc}") from exc

        if not isinstance(result, dict) or "access_token" not in result:
            detail = result.get("error_description") or result.get("error") if isinstance(result, dict) else result
            raise DocSvcAuthError(
                f"Failed to acquire DocumentService token: {detail}. Check "
                "DOCSVC_READ_CLIENT_ID/DOCSVC_READ_CLIENT_SECRET and that the "
                "app registration holds the Documents.Read app role."
            )

        self._token = result["access_token"]
        self._expires_at = now + float(result.get("expires_in", 0))
        return self._token


# ---------------------------------------------------------------------------
# REST client
# ---------------------------------------------------------------------------
def _sleep_backoff(attempt: int) -> None:
    """Exponential backoff before retry ``attempt`` (1-indexed retry count)."""
    time.sleep(0.5 * (2 ** (attempt - 1)))


class DocSvcClient:
    """Sync client for the DocumentService REST API.

    Not thread-safe to share across threads without external locking (mirrors
    ``httpx.Client``'s own guidance); create one per worker/process.
    """

    def __init__(
        self,
        settings: DocSvcSettings | None = None,
        token_provider: TokenProvider | None = None,
    ):
        self._settings = settings or DocSvcSettings.from_env()
        self._token_provider = token_provider or TokenProvider(self._settings)
        self._client = httpx.Client(
            base_url=self._settings.base_url,
            verify=self._settings.verify_ssl,
            timeout=DEFAULT_TIMEOUT,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> DocSvcClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self._token_provider.get_token()}",
        }
        if self._settings.user_id:
            headers["X-User-Id"] = self._settings.user_id
        return headers

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        not_ready_statuses: frozenset[int] = frozenset(),
    ) -> httpx.Response:
        """Issue a request with retry-with-backoff on 429/5xx/connect errors.

        ``not_ready_statuses`` (e.g. {404, 502} for the download-link
        endpoint) short-circuit immediately as ``DocSvcNotReady`` — they are
        the documented skip cases, never retried.
        """
        last_exc: Exception | None = None
        response: httpx.Response | None = None
        for attempt in range(1, RETRY_ATTEMPTS + 1):
            try:
                response = self._client.request(method, path, params=params, headers=self._headers())
            except httpx.TransportError as exc:
                last_exc = exc
                if attempt < RETRY_ATTEMPTS:
                    logger.warning("DocumentService %s %s failed (%s), retrying", method, path, exc)
                    _sleep_backoff(attempt)
                    continue
                raise DocSvcError(f"Network error calling DocumentService {method} {path}: {exc}") from exc

            if response.status_code in not_ready_statuses:
                raise DocSvcNotReady(response.status_code)

            if response.status_code in RETRY_STATUS_CODES and attempt < RETRY_ATTEMPTS:
                logger.warning(
                    "DocumentService %s %s returned %s, retrying (attempt %d/%d)",
                    method,
                    path,
                    response.status_code,
                    attempt,
                    RETRY_ATTEMPTS,
                )
                _sleep_backoff(attempt)
                continue

            return response

        assert response is not None or last_exc is not None
        raise DocSvcError(f"Exhausted retries calling DocumentService {method} {path}") from last_exc

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise DocSvcError(
                f"DocumentService {response.request.method} {response.request.url} "
                f"returned {response.status_code}: {response.text[:300]}"
            ) from exc

    def iter_documents(
        self,
        source: str | None = None,
        category: str | None = None,
        tag: str | None = None,
        page_size: int = 50,
    ) -> Iterator[dict[str, Any]]:
        """Yield raw ``DocumentResponseDto`` dicts, paging transparently."""
        page_size = min(page_size, MAX_PAGE_SIZE)
        page = 1
        total_pages = 1
        while page <= total_pages:
            params: dict[str, Any] = {"page": page, "pageSize": page_size}
            if source:
                params["source"] = source
            if category:
                params["category"] = category
            if tag:
                params["tag"] = tag

            response = self._request("GET", API_PREFIX, params=params)
            self._raise_for_status(response)
            body = response.json()

            yield from body.get("data", [])

            pagination = body.get("pagination") or {}
            total_pages = pagination.get("totalPages", page)
            page += 1

    def get_document(self, document_id: str) -> dict[str, Any]:
        response = self._request("GET", f"{API_PREFIX}/{document_id}")
        self._raise_for_status(response)
        return response.json()

    def get_download_link(self, document_id: str) -> dict[str, Any]:
        """Return the ``DownloadLinkDto``; raises ``DocSvcNotReady`` on 404/502."""
        response = self._request(
            "GET",
            f"{API_PREFIX}/{document_id}/download",
            not_ready_statuses=frozenset({404, 502}),
        )
        self._raise_for_status(response)
        return response.json()

    def download_content(self, document_id: str, dest: Path) -> Path:
        """Fetch the presigned link and immediately stream it to ``dest``.

        Presigned URLs expire in minutes — the blob GET happens right after
        obtaining the link, via a bare ``httpx`` request carrying no
        ``Authorization`` header (the SAS token in the URL is the only auth
        the blob endpoint expects/accepts).
        """
        link = self.get_download_link(document_id)
        url = link["url"]

        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)

        with httpx.stream("GET", url, timeout=DEFAULT_TIMEOUT) as response:
            response.raise_for_status()
            with dest.open("wb") as fh:
                for chunk in response.iter_bytes():
                    fh.write(chunk)

        return dest
