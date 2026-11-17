"""Semantic search tool over machine manuals embedded in Chroma
(data/chroma_manuals_db).

Self-contained: exports ``search_manuals`` plus the ``TOOL_DEFINITIONS`` and
``dispatch`` used by ``chatbot_manuals.py``, so it stays independent from the CDO
tools in ``tools/__init__.py``. Uses the same collection and embedding model as
``ingest_sample_manuals.py`` so query vectors match the stored document vectors.
"""

import json
import logging
import threading
from pathlib import Path

from sme_chatbot import config

logger = logging.getLogger(__name__)

CHROMA_DIR = Path(config.CHROMA_MANUALS_DIR)
COLLECTION_NAME = config.CHROMA_MANUALS_COLLECTION
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# Loaded lazily on first query so importing the module stays cheap.
_model = None
_collection = None
_load_lock = threading.Lock()


def _ensure_loaded() -> str | None:
    """Load the embedding model and Chroma collection. Returns an error string
    on failure, or None on success.

    Thread-safe: concurrent first calls (FastAPI's worker threads, the startup
    warm-up) wait for one load instead of each loading the model. The
    collection is opened before the model, so a missing collection costs no
    model load, and ``_collection`` is published last, so the lock-free fast
    path below never sees it without ``_model``.
    """
    global _model, _collection
    if _collection is not None:
        return None

    with _load_lock:
        if _collection is not None:
            return None

        if not CHROMA_DIR.exists():
            return f"Vector store not found at {CHROMA_DIR}. Run ingest_sample_manuals.py first."

        try:
            import chromadb
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - dependency guard
            return f"Semantic search unavailable: {exc}"

        chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))
        try:
            collection = chroma.get_collection(COLLECTION_NAME)
        except Exception:  # broad on purpose: chroma raises a bare error if missing
            return f"Collection {COLLECTION_NAME!r} not found. Run ingest_sample_manuals.py first."
        if _model is None:
            logger.info("Loading embedding model %s", EMBEDDING_MODEL)
            _model = SentenceTransformer(EMBEDDING_MODEL)
        _collection = collection
    return None


def search_manuals(query: str, n_results: int = 5) -> dict:
    """Semantic search over machine manuals.

    Returns {"results": [{"text", "source", "page", "distance"}, ...]}.
    """
    error = _ensure_loaded()
    if error:
        return {"error": error}

    n_results = max(1, min(int(n_results), 20))
    logger.info("search_manuals: %r (n=%d)", query, n_results)

    vec = _model.encode([query])[0].tolist()
    res = _collection.query(query_embeddings=[vec], n_results=n_results)

    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    results = []
    for i, text in enumerate(docs):
        meta = metas[i] if i < len(metas) else {}
        results.append(
            {
                "text": text,
                "source": meta.get("filename"),
                "page": meta.get("page"),
                "distance": dists[i] if i < len(dists) else None,
            }
        )
    return {"results": results}


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "search_manuals",
            "description": (
                "Semantic search over machine manuals and spare-parts "
                "documents. Use this for how-to, procedure, component or "
                "specification questions about the machines. Returns relevant "
                "text chunks with source filename and page number."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language search query.",
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Number of chunks to return (default 5).",
                    },
                },
                "required": ["query"],
            },
        },
    },
]

_HANDLERS = {
    "search_manuals": search_manuals,
}


def dispatch(name: str, args: dict) -> str:
    """Execute a tool call by name and return its result as a JSON string."""
    handler = _HANDLERS.get(name)
    if handler is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        result = handler(**args)
    except TypeError as exc:
        result = {"error": f"Bad arguments for {name}: {exc}"}
    except Exception as exc:  # broad on purpose: never crash the chat loop
        logger.exception("Tool %s failed", name)
        result = {"error": f"Tool {name} failed: {exc}"}
    return json.dumps(result, ensure_ascii=False, default=str)
