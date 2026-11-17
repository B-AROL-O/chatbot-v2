"""FastAPI app factory: ``GET /health``, ``GET /config``, ``POST /chat/message``
(plan §5 WP3; ``/config`` added later).

The OpenAI client is a lazily-initialized module-level singleton — it is
never constructed at import time (the legacy ``chatbot_core.py`` raised at
import without ``LLM_BASE_URL``; this module must import cleanly regardless).
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace

from fastapi import Depends, FastAPI, HTTPException
from openai import OpenAI
from sse_starlette.sse import EventSourceResponse

from sme_chatbot import config
from sme_chatbot.api.auth import get_identity
from sme_chatbot.api.schemas import ChatRequest, ChatResponse
from sme_chatbot.api.schemas import Citation as CitationSchema
from sme_chatbot.api.streaming import error_sse, event_to_sse
from sme_chatbot.core.chat import stream_chat
from sme_chatbot.core.domains import DOMAINS, documents_domain_for
from sme_chatbot.core.events import DoneEvent, ErrorEvent, RequestIdentity
from sme_chatbot.tools.warmup import warm_up

logger = logging.getLogger(__name__)

_client: OpenAI | None = None

# Process start time, for /health's uptime_seconds. Module-level so it
# reflects when the worker process came up, not when create_app() last ran
# (tests call create_app() repeatedly and must not each reset the clock).
_START_TIME = time.monotonic()


def _get_client() -> OpenAI:
    """Lazily build (and cache) the module-level OpenAI client singleton.

    When ``LANGFUSE_ENABLED``, swaps in Langfuse's drop-in OpenAI wrapper so
    every ``chat.completions.create`` and ``responses.create`` call is auto-traced (prompt, completion,
    tokens, latency, cost) with no changes to ``core/chat.py``. The import is
    deferred inside the branch so the module still imports cleanly with
    ``LANGFUSE_ENABLED`` unset and zero Langfuse config set.
    """
    global _client
    if _client is None:
        if config.LANGFUSE_ENABLED:
            from langfuse.openai import OpenAI as LangfuseOpenAI

            _client = LangfuseOpenAI(
                base_url=config.LLM_BASE_URL, api_key=config.LLM_API_TOKEN or "not-set"
            )
        else:
            _client = OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_TOKEN or "not-set")
    return _client


@asynccontextmanager
async def _lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Start the optional retrieval warm-up without blocking
    startup: ``/health`` must answer while the stores load."""
    if config.RETRIEVAL_WARMUP != "off":
        threading.Thread(
            target=warm_up, args=(config.RETRIEVAL_WARMUP,), name="retrieval-warmup", daemon=True
        ).start()
    yield


def create_app() -> FastAPI:
    logging.basicConfig(level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO))

    app = FastAPI(title="SME Chatbot API", version=config.APP_VERSION, lifespan=_lifespan)

    @app.get("/health")
    def health() -> dict:
        # Unauthenticated regardless of AUTH_MODE (plan §4.1) — an
        # orchestrator restarting a stuck pod must never need credentials.
        # Keep this payload cheap: liveness + the minimum an orchestrator or
        # a fresh Gradio client needs to route requests. Anything about how
        # the backend is *configured* (model, LLM URL, auth mode, ...)
        # belongs in GET /config instead.
        return {
            "status": "ok",
            "domains": sorted(DOMAINS.keys()),
            "version": config.APP_VERSION,
            "uptime_seconds": round(time.monotonic() - _START_TIME, 3),
        }

    @app.get("/config")
    def get_config(_: RequestIdentity = Depends(get_identity)) -> dict:
        # Static backend configuration — separated from /health
        # so this can require auth (same identity dependency as /chat/message)
        # while /health stays open for orchestrators. Secrets (API tokens,
        # client secrets, signing config) are deliberately never included.
        return {
            "version": config.APP_VERSION,
            "model": config.LLM_MODEL,
            "llm_api": config.LLM_API,
            "llm_base_url": config.LLM_BASE_URL,
            "temperature": config.LLM_TEMPERATURE,
            "seed": config.LLM_SEED,
            "max_tokens": config.LLM_MAX_TOKENS,
            "reasoning_effort": config.LLM_REASONING_EFFORT,
            "retry_reasoning_effort": config.LLM_RETRY_REASONING_EFFORT,
            "pre_search": config.CHAT_PRE_SEARCH,
            "pre_search_results": config.CHAT_PRE_SEARCH_RESULTS,
            "auth_mode": config.AUTH_MODE,
            "domains": sorted(DOMAINS.keys()),
        }

    @app.post("/chat/message", response_model=None)
    def chat_message(
        req: ChatRequest, identity: RequestIdentity = Depends(get_identity)
    ):
        domain = DOMAINS.get(req.domain)
        if domain is None:
            raise HTTPException(
                status_code=422, detail=f"Unknown domain: {req.domain!r}"
            )

        history = [m.model_dump() for m in req.history]
        setup_started = time.perf_counter()
        if domain.key == "documents":
            # Per-request: binds the caller's identity into the tools (the
            # access gate) and the matricola named in the conversation.
            domain = documents_domain_for(identity, req.message, history)
        if domain.pre_search is not None and config.CHAT_PRE_SEARCH_RESULTS is not None:
            name, args = domain.pre_search
            domain = replace(domain, pre_search=(name, {**args, "n_results": config.CHAT_PRE_SEARCH_RESULTS}))
        client = _get_client()
        # Everything before the first LLM call (identity binding, resolve_serial);
        # reported next to stream_chat's own timings, which start after it.
        setup_ms = int((time.perf_counter() - setup_started) * 1000)

        # Only what is configured, so an unset backend passes exactly what it did before.
        sampling = {
            key: value
            for key, value in (
                ("temperature", config.LLM_TEMPERATURE),
                ("seed", config.LLM_SEED),
                ("max_tokens", config.LLM_MAX_TOKENS),
                ("reasoning_effort", config.LLM_REASONING_EFFORT),
                ("retry_reasoning_effort", config.LLM_RETRY_REASONING_EFFORT),
                ("pre_search", config.CHAT_PRE_SEARCH or None),
                ("llm_api", None if config.LLM_API == "chat_completions" else config.LLM_API),
            )
            if value is not None
        }

        def run_chat():
            return stream_chat(
                client=client,
                model=config.LLM_MODEL,
                domain=domain,
                message=req.message,
                history=history,
                identity=identity,
                **sampling,
            )

        if req.stream:
            return EventSourceResponse(
                _stream_events(run_chat(), domain.key, identity, req.message, setup_ms)
            )

        return _collect_response(run_chat(), domain.key, identity, req.message, setup_ms)

    return app


def _add_setup_ms(usage: dict, setup_ms: int) -> None:
    """Merge the pre-LLM setup time into ``usage["timings"]``."""
    usage.setdefault("timings", {})["setup_ms"] = setup_ms


def _log_request(domain_key: str, identity: RequestIdentity, message: str, usage: dict) -> None:
    # Durations only, never message content.
    timings = usage.get("timings") or {}
    logger.info(
        "chat/message domain=%s user_id=%s message_len=%d rounds=%s round_limit_hit=%s "
        "setup_ms=%s total_ms=%s ttft_ms=%s llm_ms=%s tool_ms=%s",
        domain_key,
        identity.user_id,
        len(message),
        usage.get("rounds", 0),
        usage.get("round_limit_hit", False),
        timings.get("setup_ms"),
        timings.get("total_ms"),
        timings.get("ttft_ms"),
        timings.get("llm_ms"),
        timings.get("tool_ms"),
    )


def _stream_events(
    events, domain_key: str, identity: RequestIdentity, message: str, setup_ms: int = 0
):
    """Generator mapping ``ChatEvent``s to SSE, logging once at the end.

    ``stream_chat`` already converts internal failures into a terminal
    ``ErrorEvent`` rather than raising, but this still guards against an
    exception escaping the glue code itself (e.g. a bug in the SSE mapper).
    """
    usage: dict = {}
    try:
        for event in events:
            if isinstance(event, DoneEvent):
                _add_setup_ms(event.usage, setup_ms)
                usage = event.usage
            yield event_to_sse(event)
    except Exception as exc:  # broad on purpose: must not crash the SSE stream
        logger.exception("chat/message: uncaught error while streaming")
        yield error_sse(str(exc))
    finally:
        _log_request(domain_key, identity, message, usage)


def _collect_response(
    events, domain_key: str, identity: RequestIdentity, message: str, setup_ms: int = 0
) -> ChatResponse:
    """Drain the event generator and build a ``ChatResponse`` (stream=false)."""
    final: DoneEvent | None = None
    error: ErrorEvent | None = None
    usage: dict = {}
    try:
        for event in events:
            if isinstance(event, DoneEvent):
                final = event
                _add_setup_ms(final.usage, setup_ms)
                usage = final.usage
            elif isinstance(event, ErrorEvent):
                error = event
    except Exception as exc:  # broad on purpose: uncaught generator exception -> 500
        logger.exception("chat/message: uncaught error (non-streaming)")
        _log_request(domain_key, identity, message, usage)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    _log_request(domain_key, identity, message, usage)

    if error is not None:
        raise HTTPException(status_code=500, detail=error.message)
    if final is None:
        raise HTTPException(status_code=500, detail="No response generated")

    return ChatResponse(
        content=final.content,
        citations=[
            CitationSchema(
                source=c.source,
                page=c.page,
                document_id=c.document_id,
                distance=c.distance,
                filename=c.filename,
                repository=c.repository,
            )
            for c in final.citations
        ],
        usage=final.usage,
    )


def cli() -> None:
    """Entry point for the ``sme-chatbot-api`` script — runs uvicorn on :8000."""
    import uvicorn

    uvicorn.run(create_app(), host="0.0.0.0", port=8000, log_level=config.LOG_LEVEL.lower())


if __name__ == "__main__":
    cli()
