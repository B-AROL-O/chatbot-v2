"""Semantic search tool over documents embedded in Chroma (data/chroma_cdo_db).

Uses the same collection and embedding model as ingest_sample_cdo.py so query vectors
match the stored document vectors.
"""

import logging
import threading
from pathlib import Path

from sme_chatbot import config

logger = logging.getLogger(__name__)

CHROMA_DIR = Path(config.CHROMA_CDO_DIR)
COLLECTION_NAME = config.CHROMA_CDO_COLLECTION
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# Loaded lazily on first query so importing the tools package stays cheap.
_model = None
_collection = None
_load_lock = threading.Lock()
_encode_lock = threading.Lock()


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
            return f"Vector store not found at {CHROMA_DIR}. Run ingest_sample_cdo.py first."

        try:
            import chromadb
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - dependency guard
            return f"Semantic search unavailable: {exc}"

        chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))
        try:
            collection = chroma.get_collection(COLLECTION_NAME)
        except Exception:  # broad on purpose: chroma raises a bare error if missing
            return f"Collection {COLLECTION_NAME!r} not found. Run ingest_sample_cdo.py first."
        if _model is None:
            logger.info("Loading embedding model %s", EMBEDDING_MODEL)
            _model = SentenceTransformer(EMBEDDING_MODEL)
        _collection = collection
    return None


def _embed(text: str) -> list[float]:
    """Embed one query. Serialized: the module's one model is shared by every thread that
    searches (concurrent requests, and the concurrent tool calls of one round), and its
    HF fast tokenizer is not safe to call from two threads at once."""
    with _encode_lock:
        vector = _model.encode([text])[0]
    return vector.tolist()


def search_docs(query: str, n_results: int = 5) -> dict:
    """Semantic search over technical documents.

    Returns {"results": [{"text", "source", "page", "distance"}, ...]}.
    """
    error = _ensure_loaded()
    if error:
        return {"error": error}

    n_results = max(1, min(int(n_results), 20))
    logger.info("search_docs: %r (n=%d)", query, n_results)

    vec = _embed(query)
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
