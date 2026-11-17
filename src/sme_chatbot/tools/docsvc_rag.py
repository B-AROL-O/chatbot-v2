"""Semantic search tool over the DocumentService-backed corpus embedded in
Chroma (``data/chroma_docsvc_db`` by default, see ``config.CHROMA_DOCSVC_DIR``).

Mirrors ``manuals_rag.py``'s shape (lazy-loaded embedding model + collection,
friendly degradation when the store/collection is missing) but adds
query-time access control: every query is scoped through
``access_filter.build_where`` so a caller can never see chunks outside their
``RequestIdentity``.

Two LLM-visible tools (``TOOL_DEFINITIONS``): ``search_documents`` (hybrid
dense + BM25, see ``lexical_index.py``) and ``read_pages`` (whole pages, to
follow the manual's cross-references). Neither exposes ``identity`` — it is
never a parameter the model can set. ``make_dispatch(identity)`` binds the
caller's identity into a closure so the API layer can hand each per-request
``DomainConfig`` a dispatcher that cannot be overridden by tool-call
arguments.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from sme_chatbot import config
from sme_chatbot.tools.access_filter import build_where

if TYPE_CHECKING:
    from sme_chatbot.core.events import RequestIdentity

logger = logging.getLogger(__name__)

# NOTE: `RequestIdentity` is imported lazily inside the functions below (not
# at module level) to avoid a circular import: `sme_chatbot.core` eagerly
# imports `core.domains`, which imports this module to build the "documents"
# domain's tool registry. A module-level `from sme_chatbot.core.events
# import RequestIdentity` here would re-enter that partly-initialized import
# chain. See the identical note in `access_filter.py`.

CHROMA_DIR = Path(config.CHROMA_DOCSVC_DIR)
COLLECTION_NAME = config.CHROMA_DOCSVC_COLLECTION
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
            return f"Vector store not found at {CHROMA_DIR}. Run ingestion (sme-ingest-docsvc) first."

        try:
            import chromadb
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - dependency guard
            return f"Semantic search unavailable: {exc}"

        chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))
        try:
            collection = chroma.get_collection(COLLECTION_NAME)
        except Exception:  # broad on purpose: chroma raises a bare error if missing
            return f"Collection {COLLECTION_NAME!r} not found. Run ingestion (sme-ingest-docsvc) first."
        if _model is None:
            logger.info("Loading embedding model %s", EMBEDDING_MODEL)
            _model = SentenceTransformer(EMBEDDING_MODEL)
        _collection = collection
    return None


# --- Startup warm-up --------------------------------------------


def warm_up(read_files: bool = False) -> None:
    """Pay the first request's loading costs before it arrives.

    Loads the model and opens the collection, then runs one query of each
    kind a request makes: a dense query (Chroma loads the HNSW index into
    memory on the first one), a metadata lookup like ``resolve_serial``'s
    and a lexical query.

    ``read_files`` first reads every file of the store and the lexical index
    once, so their pages are in the OS page cache. Without it, on a cold
    machine every query that touches a new region of the store pays for
    random disk reads: the first queries above took 172 s against 69 s to
    read all 9.6 GB in order, and searches stayed at about 15 s for some
    questions afterwards (under 0.5 s with the files read). It only helps
    while the machine has the RAM to keep the files cached.
    """
    started = time.perf_counter()
    if read_files:
        paths = {p.resolve() for p in CHROMA_DIR.rglob("*") if p.is_file()}
        if config.RETRIEVAL_MODE == "hybrid" and Path(config.LEXICAL_INDEX_PATH).is_file():
            paths.add(Path(config.LEXICAL_INDEX_PATH).resolve())
        total = sum(_read_file(p) for p in sorted(paths))
        logger.info(
            "warm-up: read %d files (%.1f GB) in %.1f s", len(paths), total / 1e9, time.perf_counter() - started
        )

    step = time.perf_counter()
    error = _ensure_loaded()
    if error:
        logger.warning("warm-up: documents store not loaded: %s", error)
        return
    logger.info("warm-up: model and collection loaded in %.1f s", time.perf_counter() - step)

    step = time.perf_counter()
    _collection.query(query_embeddings=[_model.encode(["warm-up"])[0].tolist()], n_results=1)
    _collection.get(where={"serial": "warm-up"}, limit=1)
    lexical = _get_lexical()
    if lexical is not None:
        lexical.search("warm-up", limit=1)
    logger.info("warm-up: first queries ran in %.1f s", time.perf_counter() - step)
    logger.info("warm-up: documents store ready in %.1f s", time.perf_counter() - started)


def _read_file(path: Path, chunk_size: int = 8 << 20) -> int:
    """Read ``path`` once, discarding the data, so the OS caches its pages."""
    buffer = bytearray(chunk_size)
    total = 0
    with path.open("rb", buffering=0) as handle:
        while n := handle.readinto(buffer):
            total += n
    return total


# --- Lexical half of the hybrid retriever -----------------------------------

# `False` = looked for and absent (don't stat the file on every query);
# `None` = not looked for yet.
_lexical = None

# Reciprocal Rank Fusion constant; 60 is the value from the original RRF
# paper and the usual default — it flattens the gap between rank 1 and 2 so
# neither retriever can dominate on a single strong hit.
_RRF_K = 60
# Candidates pulled from *each* retriever before fusion.
_CANDIDATES = 20
# RRF weights. The dense list counts half: on the 90001 set it is the noisier
# ranker (recall@5 0.29 vs 0.88 lexical), and at equal weight it pushed
# lexical rank-1 table hits down to rank 2-4. At half weight hybrid matches
# lexical at the depth returned (recall@5 0.88, @10 0.91) while keeping
# dense's paraphrase matches for LLM-written queries.
_DENSE_WEIGHT = 0.5
_LEXICAL_WEIGHT = 1.0


def _get_lexical():
    """The FTS5 index, or ``None`` when hybrid retrieval is off or unbuilt."""
    global _lexical
    if config.RETRIEVAL_MODE != "hybrid":
        return None
    if _lexical is None:
        from sme_chatbot.tools.lexical_index import LexicalIndex

        index = LexicalIndex(config.LEXICAL_INDEX_PATH)
        if index.exists():
            _lexical = index
        else:
            logger.warning(
                "Lexical index %s not found; search_documents is dense-only. "
                "Build it with `sme-ingest-docsvc --build-lexical-index`.",
                config.LEXICAL_INDEX_PATH,
            )
            _lexical = False
    return _lexical or None


def _dense_hits(query: str, where: dict | None, k: int) -> list[tuple[str, str, dict, float | None]]:
    vec = _model.encode([query])[0].tolist()
    query_kwargs = {"query_embeddings": [vec], "n_results": k}
    if where is not None:
        query_kwargs["where"] = where
    res = _collection.query(**query_kwargs)
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    return [
        (ids[i], docs[i], metas[i] if i < len(metas) else {}, dists[i] if i < len(dists) else None)
        for i in range(len(docs))
    ]


def _lexical_hits(query: str, where: dict | None, k: int) -> list[tuple[str, str, dict, float | None]]:
    """BM25 candidates, re-fetched from Chroma **with the same ``where``**.

    The re-fetch is the access-control gate: the FTS pre-filter is only a
    recall aid and may be looser than ``where`` (see ``lexical_index``).
    """
    lexical = _get_lexical()
    if lexical is None:
        return []
    ids = lexical.search(query, limit=k, where=where)
    if not ids:
        return []
    get_kwargs = {"ids": ids, "include": ["documents", "metadatas"]}
    if where is not None:
        get_kwargs["where"] = where
    got = _collection.get(**get_kwargs)
    by_id = {
        got["ids"][i]: (got["documents"][i], got["metadatas"][i] or {}) for i in range(len(got["ids"]))
    }
    return [(i, by_id[i][0], by_id[i][1], None) for i in ids if i in by_id]


def _fuse(
    dense: list[tuple[str, str, dict, float | None]],
    lexical: list[tuple[str, str, dict, float | None]],
) -> list[tuple[str, str, dict, float | None]]:
    """Weighted Reciprocal Rank Fusion; keeps the dense distance when a chunk has one."""
    scores: dict[str, float] = {}
    hits: dict[str, tuple[str, str, dict, float | None]] = {}
    for weight, ranked in ((_DENSE_WEIGHT, dense), (_LEXICAL_WEIGHT, lexical)):
        for rank, hit in enumerate(ranked):
            chunk_id = hit[0]
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (_RRF_K + rank + 1)
            if chunk_id not in hits or hits[chunk_id][3] is None:
                hits[chunk_id] = hit
    return [hits[c] for c in sorted(scores, key=lambda c: -scores[c])]


def _result_entry(text: str, meta: dict, distance: float | None) -> dict:
    return {
        "text": text,
        "document_id": meta.get("document_id"),
        "version_id": meta.get("version_id"),
        "filename": meta.get("filename"),
        "repository": meta.get("source"),
        "category": meta.get("category"),
        "page": meta.get("page"),
        "language": meta.get("language"),
        "distance": distance,
    }


def search_documents(
    query: str,
    n_results: int = 8,
    category: str | None = None,
    serial: str | None = None,
    customer: str | None = None,
    language: str | None = None,
    identity: RequestIdentity | None = None,
) -> dict:
    """Hybrid (dense + BM25) search over the DocumentService-backed library.

    Results are restricted to what ``identity`` permits (see
    ``access_filter.build_where``) regardless of what the caller/LLM passes
    for ``category``/``serial``/``customer``/``language``. Both retrievers
    run under the same ``where``; lexical candidates are re-fetched from
    Chroma with it, so neither path can widen access.

    Returns {"results": [{"text", "document_id", "version_id", "filename",
    "repository", "category", "page", "language", "distance"}, ...]}.
    ``distance`` is the dense distance, or ``None`` for a chunk only the
    lexical retriever found.

    ``filename`` is the document's name; ``repository`` is the origin network
    share the document was synced from (the two were previously
    conflated under a single ``source`` key, and the citation footer showed
    the share instead of the document).

    ``language`` narrows to one document language (e.g. ``"IT"``/``"DE"``) —
    added because manuals translated into multiple languages
    are near-duplicate chunks competing for the same top-k slots, so an
    unscoped query can answer an Italian question by citing only the German
    manual. Callers (``core/domains.py``'s system prompt) are expected to
    retry without this filter if it yields zero results, since not every
    document has a translation in the asked language.

    Without a lexical index (not built, or ``RETRIEVAL_MODE=dense``) this is
    the original dense-only search.
    """
    error = _ensure_loaded()
    if error:
        return {"error": error}

    if identity is None:
        from sme_chatbot.core.events import RequestIdentity as _RequestIdentity

        identity = _RequestIdentity()

    n_results = max(1, min(int(n_results), 20))
    hits = _search(query, n_results, identity, category, serial, customer, language)
    if not hits and language is not None:
        # The language filter is an exact match on a value stored verbatim, and bilingual
        # manuals carry a combined code ("it-ro"): a narrowing that leaves nothing is retried
        # without it, so the model is not told the documents are silent.
        logger.info("search_documents: no results with language=%r; retrying without it", language)
        hits = _search(query, n_results, identity, category, serial, customer, None)
    return {"results": [_result_entry(text, meta, dist) for _, text, meta, dist in hits]}


def _search(query, n_results, identity, category, serial, customer, language) -> list:
    where = build_where(identity, category=category, serial=serial, customer=customer, language=language)
    logger.info("search_documents: %r (n=%d, where=%r)", query, n_results, where)
    k = max(n_results, _CANDIDATES)
    dense = _dense_hits(query, where, k if _get_lexical() else n_results)
    lexical = _lexical_hits(query, where, k)
    fused = _fuse(dense, lexical) if lexical else dense
    return fused[:n_results]


# --- read_pages: follow the manual's own cross-references (plan WP4) ---------

_MAX_PAGES_PER_CALL = 4
# A page footer: the printed page number alone on its line.
_FOOTER_NUMBER_RE = re.compile(r"^\s*(\d{1,4})\s*$", re.MULTILINE)
# Cache: (filename, version_id) -> {printed_page: pdf_page}
_printed_maps: dict[tuple[str, str], dict[int, int]] = {}


def _and(*clauses: dict | None) -> dict | None:
    present = [c for c in clauses if c]
    if not present:
        return None
    if len(present) == 1:
        return present[0]
    return {"$and": present}


def _join_chunks(chunks: list[str]) -> str:
    """Rejoin one page's chunks, dropping the ingestion overlap if present."""
    from sme_chatbot.ingestion.ingest_docsvc import CHUNK_OVERLAP

    text = ""
    for chunk in chunks:
        if text and chunk[:CHUNK_OVERLAP] == text[-CHUNK_OVERLAP:]:
            text += chunk[CHUNK_OVERLAP:]
        else:
            text += ("\n" if text else "") + chunk
    return text


def _printed_number(pdf_page: int, text: str) -> int | None:
    """The printed page number in ``text``'s footer, if one is recognisable.

    Printed numbers never exceed the PDF page (front matter only adds pages),
    and a footer is short, so the last standalone number in range wins.
    """
    candidates = [int(m) for m in _FOOTER_NUMBER_RE.findall(text) if 0 < int(m) <= pdf_page]
    return candidates[-1] if candidates else None


def _group_pages(got: dict) -> dict[int, tuple[dict, str]]:
    by_page: dict[int, list[tuple[int, str, dict]]] = {}
    for i, text in enumerate(got.get("documents") or []):
        meta = got["metadatas"][i] or {}
        by_page.setdefault(int(meta.get("page", 0)), []).append((int(meta.get("chunk", 0)), text, meta))
    return {
        page: (parts[0][2], _join_chunks([t for _, t, _ in sorted(parts, key=lambda p: p[0])]))
        for page, parts in by_page.items()
    }


def _printed_to_pdf(filename: str, where: dict | None) -> tuple[dict[int, int], int | None]:
    """Map printed page -> PDF page for one document, plus the typical offset."""
    probe = _collection.get(where=where, limit=1, include=["metadatas"])
    if not probe["ids"]:
        return {}, None
    key = (filename, str((probe["metadatas"][0] or {}).get("version_id")))
    if key not in _printed_maps:
        pages = _group_pages(_collection.get(where=where, include=["documents", "metadatas"]))
        mapping: dict[int, int] = {}
        for pdf_page, (_, text) in sorted(pages.items()):
            printed = _printed_number(pdf_page, text)
            if printed is not None:
                mapping.setdefault(printed, pdf_page)
        _printed_maps[key] = mapping
    mapping = _printed_maps[key]
    offsets = sorted(pdf - printed for printed, pdf in mapping.items())
    return mapping, (offsets[len(offsets) // 2] if offsets else None)


def read_pages(
    filename: str,
    pages: list[int],
    printed: bool = False,
    identity: RequestIdentity | None = None,
) -> dict:
    """Return whole pages of one document, e.g. to follow "vedere … a pagina 84".

    ``printed=True`` treats ``pages`` as the numbers printed on the paper
    (what the manual's own cross-references use), mapped to PDF pages from
    the page footers. Access is enforced by the same ``build_where`` as
    ``search_documents``: a document the caller may not see reads as absent.
    """
    error = _ensure_loaded()
    if error:
        return {"error": error}
    if identity is None:
        from sme_chatbot.core.events import RequestIdentity as _RequestIdentity

        identity = _RequestIdentity()

    try:
        wanted = [int(p) for p in (pages if isinstance(pages, list) else [pages])][:_MAX_PAGES_PER_CALL]
    except (TypeError, ValueError):
        return {"error": "pages must be a list of integers"}
    if not wanted:
        return {"error": "pages must not be empty"}

    doc_where = _and(build_where(identity), {"filename": filename})
    logger.info("read_pages: %r pages=%r printed=%s", filename, wanted, printed)

    requested = {p: p for p in wanted}
    if printed:
        mapping, offset = _printed_to_pdf(filename, doc_where)
        if not mapping and offset is None:
            return {"results": [], "note": f"No readable pages found for {filename!r}."}
        requested = {p: mapping.get(p, p + offset if offset is not None else p) for p in wanted}

    got = _collection.get(
        where=_and(doc_where, {"page": {"$in": sorted(set(requested.values()))}}),
        include=["documents", "metadatas"],
    )
    grouped = _group_pages(got)
    results = []
    for asked, pdf_page in requested.items():
        if pdf_page not in grouped:
            continue
        meta, text = grouped[pdf_page]
        entry = _result_entry(text, meta, None)
        if printed:
            entry["printed_page"] = asked
        results.append(entry)
    if not results:
        return {"results": [], "note": f"Pages {wanted} not found in {filename!r}."}
    return {"results": results}


# --- Machine scoping (plan WP2a) ---------------------------------------------

# A machine is named by its matricola. The library holds these shapes (1,150
# distinct values, 2026-10-15): 5 digits ("90001", 44%); M or A plus 4-7 digits
# ("M01234", "A1234", "M0012345", 52%); 3 digits plus a letter a-e ("123B", 2%);
# and about 25 dotted customer names ("acme.040"), not matched here. Matching only
# the first used to leave more than half of the library's machines unscoped.
#
# A candidate is a hint, not a serial: ``resolve_serial`` keeps one only if the
# library has documents tagged with it, so "400V" or "D1234" are dropped there.
# Quantities are excluded up front ("20000 ore", "12000 bph", "90001,5").
_SERIAL_CANDIDATE_RE = re.compile(
    r"(?<!\w)(?<!\d[.,])(?:"
    r"\d{5}(?!\w|[.,]\d)(?!\s*(?:(?:ore|ora|h|hours?|bph|pezzi|kg|mm|nm|bar|rpm|l/min|lux|kw|v|hz|m)\b|°|%))"
    r"|[MA]\d{4,7}(?!\w|[.,]\d)"
    r"|\d{3}[A-E](?!\w)"
    r")",
    re.IGNORECASE,
)

# At most this many candidates are looked up per message: each is one Chroma query.
_MAX_SERIAL_LOOKUPS = 6


def detect_serials(text: str) -> list[str]:
    """Candidate matricole mentioned in ``text``, unique (ignoring case), in order."""
    seen: dict[str, str] = {}
    for match in _SERIAL_CANDIDATE_RE.finditer(text or ""):
        seen.setdefault(match.group(0).lower(), match.group(0))
    return list(seen.values())


def _is_known_serial(serial: str) -> bool:
    if _ensure_loaded() is not None:
        return False
    try:
        return bool(_collection.get(where={"serial": serial.lower()}, limit=1)["ids"])
    except Exception:  # broad on purpose: scoping is best-effort, never fatal
        logger.exception("serial lookup failed for %r", serial)
        return False


def resolve_serial(message: str, history: list[dict] | None = None) -> str | None:
    """The one matricola the conversation is about, or ``None``.

    Looks at the current message first, then earlier user turns (latest
    first). A candidate counts only if the library has documents tagged with
    it, so "20000 ore" style quantities that slip past the regex are dropped.
    Two different known serials in the same message is ambiguous: ``None``.
    """
    turns = [message] + [
        m.get("content") or "" for m in reversed(history or []) if m.get("role") == "user"
    ]
    for turn in turns:
        known = [s for s in detect_serials(turn)[:_MAX_SERIAL_LOOKUPS] if _is_known_serial(s)]
        if len(known) == 1:
            return known[0]
        if len(known) > 1:
            return None
    return None


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": (
                "Hybrid keyword + semantic search over the document library "
                "synced from the DocumentService (manuals, CDO reports, "
                "certifications, and other documents). Use this for how-to, "
                "procedure, specification or document-lookup questions. Queries "
                "work best with the specific terms the document would use "
                "(component names, table titles, units). Returns relevant text "
                "chunks with document id, filename, category, origin repository "
                "and page number. Always cite the `filename` and `page` — never "
                "the `repository`, which is only the network share the document "
                "was synced from. Results are automatically restricted to what "
                "the current user is permitted to see."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Search query, written in the language of the documents "
                            "(e.g. Italian for an Italian manual) using the terms the "
                            "manual would use."
                        ),
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Number of chunks to return (default 8, max 20).",
                    },
                    "category": {
                        "type": "string",
                        "description": "Optional document category filter (e.g. 'Manual', 'CDO').",
                    },
                    "serial": {
                        "type": "string",
                        "description": "Optional machine serial number filter.",
                    },
                    "customer": {
                        "type": "string",
                        "description": "Optional customer name filter.",
                    },
                    "language": {
                        "type": "string",
                        "description": (
                            "Optional document language filter (e.g. 'IT', 'DE', 'EN'). "
                            "Set this to the language of the user's question when manuals "
                            "may exist in multiple language translations, so results don't "
                            "cite a translation the user cannot read. If it returns no "
                            "results, retry the same query without this filter."
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_pages",
            "description": (
                "Read whole pages of one document. Use it to follow a "
                "cross-reference ('vedere il sottocapitolo 9.5 a pagina 37', "
                "'tabella a pagina 8'), to see a full table when a search result "
                "shows only part of it, or to read the page before/after a hit. "
                "Up to 4 pages per call."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "The `filename` of a document returned by search_documents.",
                    },
                    "pages": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Page numbers to read (max 4).",
                    },
                    "printed": {
                        "type": "boolean",
                        "description": (
                            "true if `pages` are the numbers printed on the paper, as the "
                            "manual's own cross-references use; false (default) for the "
                            "PDF page numbers that search results report."
                        ),
                    },
                },
                "required": ["filename", "pages"],
            },
        },
    },
]

_HANDLERS = {
    "search_documents": search_documents,
    "read_pages": read_pages,
}


def make_dispatch(
    identity: RequestIdentity | None = None,
    default_serial: str | None = None,
) -> Callable[[str, dict], str]:
    """Build a ``dispatch(name, arguments) -> str`` closure bound to
    ``identity``.

    The LLM only ever supplies ``name``/``arguments`` (matching the tool
    schema above, which has no ``identity`` field); this factory is how the
    caller's ``RequestIdentity`` reaches the tools without ever being exposed
    as an LLM-controllable parameter. The API layer calls this per-request
    with the resolved identity when building that request's ``DomainConfig``.

    ``default_serial`` (plan WP2a) is the matricola the conversation names
    (``resolve_serial``). It fills ``serial`` on a ``search_documents`` call
    that omits it, so machine scoping no longer depends on the model
    remembering to pass it. An explicit ``serial`` from the model still wins:
    comparing two machines is legitimate. This is scoping, not access
    control — ``identity`` remains the only access gate.
    """
    # `identity` is passed through as-is, including `None` — the tools
    # already substitute a fully-unrestricted `RequestIdentity()` when they
    # receive `None` (see above). Resolving the default here too would force
    # a `core.events` import at module-load time (this function also backs
    # the module-level default `dispatch` below) and reintroduce the same
    # import cycle this module's other lazy imports are avoiding.
    bound_identity = identity

    def _dispatch(name: str, args: dict) -> str:
        handler = _HANDLERS.get(name)
        if handler is None:
            return json.dumps({"error": f"Unknown tool: {name}"})
        if name == "search_documents" and default_serial and not args.get("serial"):
            args = {**args, "serial": default_serial}
        try:
            result = handler(identity=bound_identity, **args)
        except TypeError as exc:
            result = {"error": f"Bad arguments for {name}: {exc}"}
        except Exception as exc:  # broad on purpose: never crash the chat loop
            logger.exception("Tool %s failed", name)
            result = {"error": f"Tool {name} failed: {exc}"}
        return json.dumps(result, ensure_ascii=False, default=str)

    return _dispatch


# Default unrestricted dispatch (identity=None -> fully unrestricted
# RequestIdentity()), for direct/manual use and as the DomainConfig.dispatch
# registered in core/domains.py. Per-request identity binding for the API
# layer should call `make_dispatch(identity)` instead.
dispatch = make_dispatch()
