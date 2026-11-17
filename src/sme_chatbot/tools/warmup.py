"""Optional retrieval warm-up at backend startup.

Every RAG tool loads its embedding model and opens its Chroma collection on
the first call, so the first request after a restart pays for all of it:
5 min 49 s on a cold backend in the 2026-10-14 baseline, against about 8 s
warm. ``warm_up`` does that loading ahead of the first request instead.

The API's lifespan hook runs it in a background thread, so ``/health``
answers at once; a request that arrives during the warm-up waits on the
same per-module load lock instead of loading a second copy. Off by default
(``RETRIEVAL_WARMUP``) so tests and the Gradio-only container stay fast.
Lazy loading at *import* time (``AGENTS.md``) is unaffected.
"""

from __future__ import annotations

import logging
import time

from sme_chatbot.tools import cdo_rag, docsvc_rag, manuals_rag

logger = logging.getLogger(__name__)


def warm_up(mode: str) -> None:
    """Warm the configured stores: ``"on"`` loads and queries them once,
    ``"full"`` also reads the documents store's files into the page cache."""
    if mode == "off":
        return
    started = time.perf_counter()
    logger.info("warm-up (%s) started", mode)
    try:
        # The documents domain first: it is the one deployed, and the slowest.
        docsvc_rag.warm_up(read_files=mode == "full")
        for module in (manuals_rag, cdo_rag):
            if module.CHROMA_DIR.exists():
                error = module._ensure_loaded()
                if error:
                    logger.warning("warm-up: %s not loaded: %s", module.__name__, error)
    except Exception:  # broad on purpose: a failed warm-up must not take the API down
        logger.exception("warm-up failed; stores will load on first use")
        return
    logger.info("warm-up (%s) finished in %.1f s", mode, time.perf_counter() - started)
