"""WP2 — a thin HTTP client of the chatbot backend.

Deliberately imports **nothing from the rest of ``sme_chatbot`` except
``config``**, the same dogfooding rule ``frontends/gradio_app.py`` follows: the
runner must exercise the public contract a .NET proxy would consume, not reach
into the app it is measuring.

Uses the non-streaming path (``stream: false``), which returns one
``ChatResponse`` body — ``{content, citations, usage}`` — and turns an in-loop
failure into HTTP 500 ``{"detail": …}`` rather than a truncated answer. That is
exactly the three things the runner needs, in one round trip.

A request that ultimately fails yields an :class:`AskResult` with ``error`` set
and **no answer**. The runner records that as ``ERROR`` and excludes it from
every percentage — a transport failure is never a wrong answer (plan §7 R5).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import httpx

from sme_chatbot import config

logger = logging.getLogger(__name__)

DEFAULT_USER_ID = "eval-runner"
DEFAULT_TIMEOUT = 180.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF = 2.0
# Cap on `Retry-After`, so a backend asking for an hour does not hang the run.
MAX_RETRY_AFTER = 60.0

RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})

# The backend turns a content-filter block (prompt or answer) into HTTP 500
# with a message containing this phrase (``core/chat.py``). The block is
# deterministic, so retrying only repeats the full chat round. Matched
# on text because the client must not import the backend's code.
CONTENT_FILTER_MARKER = "LLM provider's content filter"


class BackendUnavailable(RuntimeError):
    """``GET /health`` failed, or the requested domain is not served."""


@dataclass
class AskResult:
    """One ``POST /chat/message`` round trip, successful or not."""

    content: str = ""
    citations: list[dict] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    latency_ms: int = 0
    status_code: int | None = None
    attempts: int = 0
    trace_id: str | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def rounds(self) -> int | None:
        value = self.usage.get("rounds")
        return value if isinstance(value, int) else None


class EvalClient:
    """``GET /health`` + ``POST /chat/message`` with backoff, nothing else."""

    def __init__(
        self,
        api_url: str | None = None,
        *,
        user_id: str = DEFAULT_USER_ID,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff: float = DEFAULT_BACKOFF,
        verify: bool = True,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.api_url = (api_url or config.CHATBOT_API_URL).rstrip("/")
        self.user_id = user_id
        self.max_retries = max(0, max_retries)
        self.backoff = backoff
        self._sleep = sleep
        self._client = httpx.Client(
            base_url=self.api_url,
            timeout=timeout,
            verify=verify,
            transport=transport,
            headers={"content-type": "application/json", "X-User-Id": user_id},
        )

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> EvalClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # -- endpoints ---------------------------------------------------------

    def health(self) -> dict:
        """``GET /health``; raises :class:`BackendUnavailable` if unreachable."""
        try:
            response = self._client.get("/health", timeout=30.0)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise BackendUnavailable(f"GET {self.api_url}/health failed: {exc}") from exc

    def config(self) -> dict:
        """``GET /config``; raises :class:`BackendUnavailable` if unreachable.

        The runner's own host may have a different (or no) ``LLM_MODEL`` set
        than the backend under test — recording the CLI host's env var into
        the report would silently mislabel which model actually answered.
        This is the backend's own runtime config.
        """
        try:
            response = self._client.get("/config", timeout=30.0)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise BackendUnavailable(f"GET {self.api_url}/config failed: {exc}") from exc

    def require_domain(self, domain: str) -> dict:
        """``GET /health`` and abort early if *domain* is not served.

        Running 36 questions against a backend that does not serve the domain
        produces 36 identical 422s; one clear message up front is cheaper.
        """
        health = self.health()
        domains = health.get("domains") or []
        if domain not in domains:
            raise BackendUnavailable(
                f"domain {domain!r} not served by {self.api_url} (available: {', '.join(domains) or 'none'})"
            )
        return health

    def ask(
        self,
        domain: str,
        message: str,
        *,
        history: list[dict] | None = None,
        headers: dict[str, str] | None = None,
    ) -> AskResult:
        """One non-streaming question. Never raises for a backend/transport failure."""
        payload = {
            "domain": domain,
            "message": message,
            "history": history or [],
            "stream": False,
        }
        started = time.perf_counter()
        last_error = "no attempt made"
        status: int | None = None
        attempts_made = 0

        for attempt in range(self.max_retries + 1):
            attempts_made = attempt + 1
            try:
                response = self._client.post("/chat/message", json=payload, headers=headers)
                status = response.status_code
                if response.status_code == 200:
                    body = response.json()
                    return AskResult(
                        content=body.get("content") or "",
                        citations=list(body.get("citations") or []),
                        usage=dict(body.get("usage") or {}),
                        latency_ms=_elapsed_ms(started),
                        status_code=status,
                        attempts=attempt + 1,
                    )
                last_error = f"HTTP {response.status_code}: {_detail(response)}"
                if response.status_code not in RETRYABLE_STATUS or CONTENT_FILTER_MARKER in last_error:
                    break
                delay = _retry_after(response) or self.backoff * (2**attempt)
            except httpx.HTTPError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                delay = self.backoff * (2**attempt)

            if attempt >= self.max_retries:
                break
            logger.warning("eval request failed (%s); retrying in %.1fs", last_error, delay)
            self._sleep(delay)

        return AskResult(
            latency_ms=_elapsed_ms(started),
            status_code=status,
            attempts=attempts_made,
            error=last_error,
        )


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _detail(response: httpx.Response) -> str:
    """The FastAPI ``{"detail": …}`` message, or the raw body, truncated."""
    try:
        body = response.json()
    except ValueError:
        return response.text[:500]
    if isinstance(body, dict) and "detail" in body:
        return str(body["detail"])[:500]
    return str(body)[:500]


def _retry_after(response: httpx.Response) -> float | None:
    raw = response.headers.get("retry-after")
    if not raw:
        return None
    try:
        return min(float(raw), MAX_RETRY_AFTER)
    except ValueError:
        return None
