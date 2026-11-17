"""WP7 — Langfuse-ready, not Langfuse-dependent.

When ``LANGFUSE_ENABLED`` is set, the runner opens one span per item and injects
the W3C ``traceparent`` header into the request. ``opentelemetry-api`` is already
in the lock file (transitively, via langfuse), so this costs no new dependency.

**Stated limitation.** Until the backend adds a root span on its
non-streaming path, the backend does *not* adopt the inbound context — each LLM
round still becomes its own root trace. The ``Trace Langfuse`` column therefore
carries the *runner's* trace id: real, and useful for correlating by timestamp,
but not yet a link to one joined backend trace. When WP1 lands, this file needs
no change.

With ``LANGFUSE_ENABLED`` unset, :func:`make_tracer` returns a no-op and **no
Langfuse or OpenTelemetry module is imported at all** — the same rule
``api/main.py::_get_client()`` follows.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class NullTracer:
    """No Langfuse: no span, no header, no trace id."""

    enabled = False

    @contextmanager
    def item_span(self, span_name: str, **_attributes: object) -> Iterator[tuple[dict[str, str], str | None]]:
        del span_name
        yield {}, None


class OtelTracer:
    """One OpenTelemetry span per item, with W3C context injected into headers."""

    enabled = True

    def __init__(self) -> None:
        # Imported inside the enabled branch on purpose (see module docstring).
        from opentelemetry import trace
        from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

        self._trace = trace
        self._propagator = TraceContextTextMapPropagator()
        self._tracer = trace.get_tracer("sme_chatbot.eval")
        self._langfuse = _try_langfuse_client()

    @contextmanager
    def item_span(self, span_name: str, **attributes: object) -> Iterator[tuple[dict[str, str], str | None]]:
        with self._tracer.start_as_current_span(f"eval.{span_name}") as span:
            for key, value in attributes.items():
                if value is not None:
                    # OTel attributes are scalars; stringify anything else.
                    scalar = value if isinstance(value, str | bool | int | float) else str(value)
                    span.set_attribute(f"eval.{key}", scalar)
            headers: dict[str, str] = {}
            self._propagator.inject(headers)
            context = span.get_span_context()
            trace_id = format(context.trace_id, "032x") if context and context.trace_id else None
            yield headers, trace_id
        if self._langfuse is not None:
            try:
                self._langfuse.flush()
            except Exception as exc:  # degrade: a flush failure must not fail the run
                logger.warning("Langfuse flush failed: %s", exc)


def _try_langfuse_client() -> object | None:
    """Best-effort Langfuse client, purely so spans actually get exported.

    Failure here is not fatal: without it the runner still produces real W3C
    trace ids and records them, it just does not ship the spans anywhere.
    """
    try:
        from langfuse import get_client

        return get_client()
    except Exception as exc:  # broad on purpose: any Langfuse setup failure degrades
        logger.warning("Langfuse client unavailable (%s); trace ids recorded but not exported", exc)
        return None


def make_tracer(enabled: bool) -> NullTracer | OtelTracer:
    """A tracer for the run — :class:`NullTracer` unless tracing is enabled."""
    if not enabled:
        return NullTracer()
    try:
        return OtelTracer()
    except Exception as exc:  # broad on purpose: tracing is never worth failing a run
        logger.warning("tracing requested but unavailable (%s); continuing without it", exc)
        return NullTracer()
