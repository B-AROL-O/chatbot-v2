"""DocumentService -> ChromaDB ingestion pipeline (WP6).

Self-contained CLI that pulls documents from the DocumentService REST API
(via :mod:`sme_chatbot.clients.docsvc`), downloads content through presigned
URLs, chunks/embeds it, and indexes the result into a dedicated ChromaDB
collection (``CHROMA_DOCSVC_COLLECTION`` at ``CHROMA_DOCSVC_DIR``) for query-time access
filtering by ``document_id`` / business-key metadata.

Per the implementation plan
(``docs/specs/2026-07-02-backend-split-and-docsvc-ingestion-plan.md``,
decision table + WP6), this pipeline is deliberately **self-contained**: the
chunking/page-extraction logic below is *copied* from ``ingest_sample_manuals.py``
(pdf via ``pymupdf4llm``, docx via ``python-docx``, doc via ``antiword``), not
imported from it, because there is no shared-ingestion-library refactor in
this phase.

Frozen contracts this module implements exactly:

- §4.3 — chunk ID scheme, metadata schema (scalars-only!), stale-version delete.
- §4.4 — DocumentService REST semantics (consumed via ``DocSvcClient``).
- §4.5 — environment variables (consumed via ``sme_chatbot.config``).
- §9   — risks: Chroma metadata must be flat scalars; presigned URLs expire
  in minutes (download immediately, right before parsing); be conservative
  against shared dev data (hence ``--limit``/``--dry-run``).

Note: heavy dependencies (``chromadb``, ``sentence_transformers``,
``pymupdf4llm``, ``docx``) are imported at module level (matching
``ingest_sample_manuals.py``'s style) but the *embedding model itself* is
lazy-loaded on first use (``_get_embedding_model()``) and no ChromaDB client
is constructed at import time — so ``import sme_chatbot.ingestion.ingest_docsvc``
stays fast and network-free.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import chromadb
import pymupdf4llm

from sme_chatbot import config
from sme_chatbot.clients.docsvc import DocSvcAuthError, DocSvcClient, DocSvcNotReady

logger = logging.getLogger(__name__)

COLLECTION_NAME = config.CHROMA_DOCSVC_COLLECTION
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBED_BATCH_SIZE = 32
ADD_BATCH_SIZE = 500

SUPPORTED_EXTENSIONS = {"pdf", "docx", "doc"}

# Message shown when DocSvc credentials/config are missing at startup.
# Mirrors the guidance embedded in DocSvcAuthError (see clients/docsvc.py):
# both point at docs/procedures/docsvc-ingestion.md.
_MISSING_CREDS_GUIDANCE = (
    "DocumentService credentials are not configured. Set DOCSVC_TENANT_ID "
    "(or DOCSVC_AUTHORITY), DOCSVC_AUDIENCE (or DOCSVC_SCOPE), and "
    "DOCSVC_READ_CLIENT_ID / DOCSVC_READ_CLIENT_SECRET — see "
    "docs/procedures/docsvc-ingestion.md for how to obtain them."
)


# ---------------------------------------------------------------------------
# Chunking / parsing (copied from ingest_sample_manuals.py — kept in sync
# manually, not shared, per the plan's decision to keep ingestion scripts
# independent).
# ---------------------------------------------------------------------------
def chunk_text(text: str) -> list[str]:
    if not text.strip():
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - CHUNK_OVERLAP
    return chunks


def _extract_docx_pages(path: Path) -> list[tuple[int, str]]:
    """Extract a .docx as a single logical page of joined paragraph text."""
    from docx import Document

    doc = Document(str(path))
    text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return [(1, text)] if text.strip() else []


def _extract_doc_pages(path: Path) -> list[tuple[int, str]]:
    try:
        result = subprocess.run(
            ["antiword", str(path)],
            capture_output=True,
            text=True,
            timeout=60,
            # A non-zero antiword exit becomes CalledProcessError, which the
            # handler below turns into a logged skip. Without check= a failed
            # conversion silently yielded empty text and no warning.
            check=True,
        )
        text = result.stdout
    except FileNotFoundError:
        logger.warning("Skipping %s: 'antiword' not found on PATH", path.name)
        return []
    except Exception as exc:
        logger.warning("Skipping %s: antiword failed: %s", path.name, exc)
        return []
    return [(1, text)] if text.strip() else []


def _extract_pdf_pages_fallback(path: Path) -> list[tuple[int, str]]:
    """Per-page fallback for PDFs where whole-document markdown conversion fails.

    pymupdf4llm's table/stripe layout heuristics (``cluster_stripes`` in
    ``pymupdf4llm/helpers/utils.py``) can raise ``ValueError: min() iterable
    argument is empty`` on certain page layouts. Retrying page-by-page lets
    the good pages still convert to markdown; a page that fails even in
    isolation falls back further to plain-text extraction so the document
    isn't dropped entirely over one problematic page.
    """
    import pymupdf

    doc = pymupdf.open(str(path))
    try:
        results = []
        for page_index in range(doc.page_count):
            page_number = page_index + 1
            try:
                md_pages = pymupdf4llm.to_markdown(doc, pages=[page_index], page_chunks=True)
                text = md_pages[0]["text"] if md_pages else ""
            except Exception as exc:
                logger.warning(
                    "pymupdf4llm markdown conversion failed for %s page %d (%s); "
                    "falling back to plain text for this page",
                    path.name,
                    page_number,
                    exc,
                )
                text = doc.load_page(page_index).get_text()
            if text.strip():
                results.append((page_number, text))
        return results
    finally:
        doc.close()


def extract_pages(path: Path) -> list[tuple[int, str]]:
    """Return a list of (page_number, text) for a supported document."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            pages = pymupdf4llm.to_markdown(str(path), page_chunks=True)
            return [(p["metadata"]["page_number"], p["text"]) for p in pages]
        except Exception as exc:
            logger.warning(
                "pymupdf4llm markdown conversion failed for %s (%s); retrying page-by-page",
                path.name,
                exc,
            )
            return _extract_pdf_pages_fallback(path)
    if suffix == ".docx":
        return _extract_docx_pages(path)
    if suffix == ".doc":
        return _extract_doc_pages(path)
    logger.warning("Skipping %s: unsupported file type %r", path.name, suffix)
    return []


# ---------------------------------------------------------------------------
# Lazy-loaded embedding model
# ---------------------------------------------------------------------------
_embedding_model = None


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model %s", EMBEDDING_MODEL)
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


# ---------------------------------------------------------------------------
# Metadata helpers (§4.3 — scalars only, flatten DTO metadata)
# ---------------------------------------------------------------------------
_SCALAR_TYPES = (str, int, float, bool)


def _flatten_scalar_metadata(raw_metadata: Any) -> dict[str, str]:
    """Flatten the DTO's ``metadata`` object into scalar-only chunk metadata.

    Every scalar entry (str/int/float/bool) is copied as ``str(value)``.
    Non-scalar entries (nested dicts/lists/None) are skipped with a log
    line — ChromaDB metadata accepts scalars only (§9 risk).
    """
    flattened: dict[str, str] = {}
    if not isinstance(raw_metadata, dict):
        return flattened
    for key, value in raw_metadata.items():
        if isinstance(value, (bool, str, int, float)):
            flattened[key] = str(value)
        else:
            logger.info(
                "Skipping non-scalar metadata key %r (type %s) for flattening",
                key,
                type(value).__name__,
            )
    return flattened


_RESERVED_METADATA_KEYS = {
    "document_id",
    "version_id",
    "version_number",
    "source",
    "category",
    "filename",
    "page",
    "chunk",
}


def _flatten_tags(raw_tags: Any) -> dict[str, str]:
    """Flatten the DTO's ``tags`` array (``["key:value", ...]``) into chunk metadata.

    Each tag is split on the *first* ``:`` only — tag values may legitimately
    contain ``.`` and ``-``, so splitting on every colon would corrupt them.
    Values are lowercased unconditionally: the OpenAPI tag pattern
    (``^[a-z0-9_-]+:[a-z0-9_\\-\\.]+$``) already forbids uppercase, but this
    guarantees one canonical form rather than relying on that pattern
    holding forever.
    Non-string entries and strings without a ``:`` are skipped with a log
    line, same style as ``_flatten_scalar_metadata``. A key repeated within
    one document's tags is first-wins, with a warning.
    """
    flattened: dict[str, str] = {}
    if not isinstance(raw_tags, list):
        return flattened
    for tag in raw_tags:
        if not isinstance(tag, str):
            logger.info("Skipping non-string tag %r for flattening", tag)
            continue
        key, sep, value = tag.partition(":")
        if not sep:
            logger.info("Skipping malformed tag %r (missing ':') for flattening", tag)
            continue
        if key in flattened:
            logger.warning("Duplicate tag key %r in tags; keeping first value %r", key, flattened[key])
            continue
        flattened[key] = value.lower()
    return flattened


def _first_chunk_id(document_id: str, version_number: int) -> str:
    """Deterministic id of the very first chunk of a document version.

    Used as a cheap idempotency probe: page 1, chunk 0 always exists if the
    document produced any chunk at all.
    """
    return f"{document_id}::v{version_number}::p1::c0"


def _chunk_id(document_id: str, version_number: int, page: int, chunk_idx: int) -> str:
    return f"{document_id}::v{version_number}::p{page}::c{chunk_idx}"


def _build_chunk_metadata(
    doc: dict[str, Any],
    *,
    document_id: str,
    version_id: str,
    version_number: int,
    source: str,
    category: str,
    filename: str,
    page: int,
    chunk_idx: int,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "document_id": document_id,
        "version_id": version_id,
        "version_number": version_number,
        "source": source,
        "category": category,
        "filename": filename,
        "page": page,
        "chunk": chunk_idx,
    }
    metadata.update(_flatten_scalar_metadata(doc.get("metadata")))
    # Tags never overwrite a key already present: reserved base keys and
    # DTO-derived metadata both win over tags in this single pass.
    for key, value in _flatten_tags(doc.get("tags")).items():
        if key in _RESERVED_METADATA_KEYS or key in metadata:
            continue
        metadata[key] = value
    return metadata


# ---------------------------------------------------------------------------
# Per-document pipeline
# ---------------------------------------------------------------------------
class _DocumentSkipped(Exception):
    """Raised internally to short-circuit a document's processing with a reason."""


def _plan_document(doc: dict[str, Any]) -> tuple[str, str, str]:
    """Validate a DTO and return (document_id, version_id, extension) or raise
    ``_DocumentSkipped`` with a human-readable reason."""
    document_id = doc.get("documentId")
    version_id = doc.get("versionId")
    display_name = doc.get("displayName") or document_id or "<unknown>"

    if doc.get("deleted"):
        raise _DocumentSkipped(f"{display_name}: marked deleted")

    if not document_id or not version_id:
        raise _DocumentSkipped(f"{display_name}: missing documentId/versionId")

    extension = (doc.get("extension") or "").lstrip(".").lower()
    if not extension:
        # Fall back to guessing from displayName's suffix.
        extension = Path(display_name).suffix.lstrip(".").lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise _DocumentSkipped(f"{display_name}: unsupported extension {extension!r}")

    return document_id, version_id, extension


def _process_document(
    doc: dict[str, Any],
    *,
    client: DocSvcClient,
    collection,
    dry_run: bool,
    lexical=None,
) -> str:
    """Process a single document DTO.

    ``lexical`` (a ``LexicalIndex``, optional) is kept in step with the
    collection so hybrid search sees new documents without a rebuild.

    Returns one of ``"added"``, ``"skipped"``, ``"failed"``.
    """
    try:
        document_id, version_id, extension = _plan_document(doc)
    except _DocumentSkipped as exc:
        logger.info("Skipping document: %s", exc)
        return "skipped"

    display_name = doc.get("displayName") or document_id
    version_number = doc.get("versionNumber")
    try:
        version_number = int(version_number)
    except (TypeError, ValueError):
        version_number = 1

    probe_id = _first_chunk_id(document_id, version_number)
    existing = collection.get(ids=[probe_id])
    if existing.get("ids"):
        logger.info(
            "Skipping %s (document_id=%s v%d): already ingested",
            display_name,
            document_id,
            version_number,
        )
        return "skipped"

    if dry_run:
        logger.info(
            "[dry-run] would ingest %s (document_id=%s, version_id=%s, v%d, ext=%s)",
            display_name,
            document_id,
            version_id,
            version_number,
            extension,
        )
        return "skipped"

    source = doc.get("source") or ""
    category = doc.get("category") or ""

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            dest = Path(tmpdir) / f"document.{extension}"
            client.download_content(document_id, dest)
            pages = extract_pages(dest)
    except DocSvcNotReady as exc:
        logger.warning("Skipping %s: download not ready (%s)", display_name, exc)
        return "skipped"
    except Exception as exc:
        logger.warning("Failed to download/parse %s: %s", display_name, exc)
        return "failed"

    if not pages:
        logger.warning("No extractable text for %s; skipping", display_name)
        return "skipped"

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []

    for page_num, text in pages:
        for chunk_idx, chunk in enumerate(chunk_text(text)):
            ids.append(_chunk_id(document_id, version_number, page_num, chunk_idx))
            documents.append(chunk)
            metadatas.append(
                _build_chunk_metadata(
                    doc,
                    document_id=document_id,
                    version_id=version_id,
                    version_number=version_number,
                    source=source,
                    category=category,
                    filename=display_name,
                    page=page_num,
                    chunk_idx=chunk_idx,
                )
            )

    if not ids:
        logger.warning("No chunks produced for %s (empty text); skipping", display_name)
        return "skipped"

    try:
        model = _get_embedding_model()
        vecs = model.encode(documents, batch_size=EMBED_BATCH_SIZE)
        embeddings = [v.tolist() for v in vecs]

        for i in range(0, len(ids), ADD_BATCH_SIZE):
            collection.add(
                ids=ids[i : i + ADD_BATCH_SIZE],
                embeddings=embeddings[i : i + ADD_BATCH_SIZE],
                documents=documents[i : i + ADD_BATCH_SIZE],
                metadatas=metadatas[i : i + ADD_BATCH_SIZE],
            )

        # Stale-version delete (§4.3): remove chunks from any version of this
        # document other than the one we just ingested.
        collection.delete(
            where={
                "$and": [
                    {"document_id": document_id},
                    {"version_id": {"$ne": version_id}},
                ]
            }
        )
    except Exception as exc:
        logger.warning("Failed to embed/write chunks for %s: %s", display_name, exc)
        return "failed"

    if lexical is not None:
        # Not fatal: Chroma is the source of truth, and a missed document only
        # costs lexical recall until the next `--build-lexical-index`.
        try:
            lexical.add(ids, documents, metadatas)
            lexical.delete_document(document_id, keep_ids=ids)
        except Exception as exc:
            logger.warning("Failed to update lexical index for %s: %s", display_name, exc)

    logger.info("%s: added %d chunks (v%d)", display_name, len(ids), version_number)
    return "added"


# ---------------------------------------------------------------------------
# Lexical index build (plan WP1): FTS5 side table from the existing
# collection — no download, no re-embedding, no DocumentService credentials.
# ---------------------------------------------------------------------------
LEXICAL_BUILD_BATCH = 5000


def build_lexical_index(collection, index_path: Path, *, batch_size: int = LEXICAL_BUILD_BATCH) -> int:
    """(Re)build the FTS5 index at ``index_path`` from every chunk in ``collection``.

    Builds into a sibling temp file and renames it into place, so a running
    API keeps reading the previous index until it reopens the file (restart).
    Returns the number of chunks indexed.
    """
    import os

    from sme_chatbot.tools.lexical_index import LexicalIndex

    index_path = Path(index_path)
    tmp_path = index_path.with_name(index_path.name + ".building")
    tmp_path.unlink(missing_ok=True)
    index = LexicalIndex(tmp_path)
    # List every id once, then fetch by id: Chroma's `offset` paging rescans
    # from the start on every page, which is quadratic over ~857k chunks.
    all_ids = collection.get(include=[])["ids"]
    total = len(all_ids)
    written = 0
    try:
        for start in range(0, total, batch_size):
            got = collection.get(ids=all_ids[start : start + batch_size], include=["documents", "metadatas"])
            written += index.add(got["ids"], got["documents"], got["metadatas"])
            logger.info("lexical index: %d / %d chunks", written, total)
    finally:
        index.close()
    os.replace(tmp_path, index_path)
    logger.info("Lexical index written to %s (%d chunks)", index_path, written)
    return written


# ---------------------------------------------------------------------------
# Tag backfill (WP3): patch tags onto already-ingested chunks in place.
# ---------------------------------------------------------------------------
def _backfill_document_tags(
    doc: dict[str, Any],
    *,
    collection,
    dry_run: bool,
) -> str:
    """Patch tag metadata onto a document's already-ingested chunks.

    Metadata-only: never adds/removes/re-embeds chunks, only calls
    ``collection.update()`` (which merges metadata in place). Returns one of
    ``"patched"``, ``"skipped"``, ``"version_mismatch"``, ``"failed"``.
    """
    document_id = doc.get("documentId")
    version_id = doc.get("versionId")
    display_name = doc.get("displayName") or document_id or "<unknown>"

    if doc.get("deleted"):
        logger.info("Skipping backfill for %s: marked deleted", display_name)
        return "skipped"

    if not document_id or not version_id:
        logger.info("Skipping backfill for %s: missing documentId/versionId", display_name)
        return "skipped"

    try:
        existing = collection.get(where={"document_id": document_id}, include=["metadatas"])
    except Exception as exc:
        logger.warning("Failed to read existing chunks for %s: %s", display_name, exc)
        return "failed"

    ids = existing.get("ids") or []
    metadatas = existing.get("metadatas") or []
    if not ids:
        logger.info("Skipping backfill for %s: not yet ingested (no existing chunks)", display_name)
        return "skipped"

    # All chunks of one (document_id, version_id) share an identical metadata
    # baseline by construction (§4.3 - only page/chunk vary), but an earlier
    # interrupted run could in principle have left chunks split across two
    # version_ids (e.g. a crash mid-way through _process_document's
    # stale-version delete). Check every chunk, not just a representative
    # one, so that gap can't silently stamp new tags onto stale chunks.
    if any(meta.get("version_id") != version_id for meta in metadatas):
        logger.info(
            "Skipping backfill for %s: stored version_id(s) != DTO version_id %r (needs re-ingestion)",
            display_name,
            version_id,
        )
        return "version_mismatch"

    stored_metadata = metadatas[0] if metadatas else {}
    tag_updates = {
        key: value
        for key, value in _flatten_tags(doc.get("tags")).items()
        if key not in _RESERVED_METADATA_KEYS and key not in stored_metadata
    }

    if not tag_updates:
        logger.info("Skipping backfill for %s: no new tag keys to patch", display_name)
        return "skipped"

    if dry_run:
        logger.info(
            "[dry-run] would patch %d chunks for %s with tags %s",
            len(ids),
            display_name,
            tag_updates,
        )
        # Unlike _process_document's dry-run (which returns "skipped"), this
        # returns "patched" so the run summary directly answers "what would
        # --dry-run patch" (issue's acceptance criterion) rather than folding
        # it into the same bucket as "nothing to do here".
        return "patched"

    try:
        for i in range(0, len(ids), ADD_BATCH_SIZE):
            batch_ids = ids[i : i + ADD_BATCH_SIZE]
            collection.update(ids=batch_ids, metadatas=[tag_updates] * len(batch_ids))
    except Exception as exc:
        logger.warning("Failed to patch chunks for %s: %s", display_name, exc)
        return "failed"

    logger.info("%s: patched %d chunks with tags %s", display_name, len(ids), tag_updates)
    return "patched"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_tag(tag: str | None) -> str | None:
    return tag


def _iter_filtered_documents(
    client: DocSvcClient,
    *,
    category: str | None,
    source: str | None,
    tag: str | None,
    page_size: int,
    limit: int | None,
) -> Iterable[dict[str, Any]]:
    for count, doc in enumerate(
        client.iter_documents(source=source, category=category, tag=tag, page_size=page_size)
    ):
        if limit is not None and count >= limit:
            return
        yield doc


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sme-ingest-docsvc",
        description=(
            "Ingest documents from the DocumentService into the "
            "documents-domain ChromaDB collection (CHROMA_DOCSVC_COLLECTION)."
        ),
    )
    parser.add_argument("--category", default=None, help="Filter by DocumentService category")
    parser.add_argument("--source", default=None, help="Filter by DocumentService source (e.g. Sharepoint)")
    parser.add_argument("--tag", default=None, metavar="key:value", help="Filter by tag, e.g. serial:91123")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Process at most N documents (after filtering)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List documents and parse decisions only; no download/embed/write",
    )
    parser.add_argument(
        "--backfill-tags",
        action="store_true",
        help=(
            "Patch tag metadata (serial, doc_type, ...) onto chunks already ingested "
            "before WP1, instead of ingesting new documents. Metadata-only: no "
            "download, no re-embedding. Combine with --dry-run/--limit to preview."
        ),
    )
    parser.add_argument(
        "--build-lexical-index",
        action="store_true",
        help=(
            "(Re)build the keyword (FTS5) index used by hybrid search from the chunks "
            "already in the collection, then exit. No DocumentService access, no "
            "re-embedding. Re-run after --backfill-tags so its serial filters are current."
        ),
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=50,
        help="DocumentService listing page size (default: 50, capped at 100 server-side)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=config.LOG_LEVEL.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    args = build_arg_parser().parse_args(argv)

    if args.build_lexical_index:
        chroma = chromadb.PersistentClient(path=str(Path(config.CHROMA_DOCSVC_DIR)))
        try:
            collection = chroma.get_collection(COLLECTION_NAME)
        except Exception:  # broad on purpose: chroma raises a bare error if missing
            print(f"Collection {COLLECTION_NAME!r} not found; ingest first.", file=sys.stderr)
            return 1
        build_lexical_index(collection, Path(config.LEXICAL_INDEX_PATH))
        return 0

    try:
        client = DocSvcClient()
        # Force credential validation up front (before any listing/download)
        # so missing-creds failures exit cleanly with guidance, per WP6 spec.
        client._token_provider.get_token()  # private on purpose: intentional early check
    except DocSvcAuthError:
        print(_MISSING_CREDS_GUIDANCE, file=sys.stderr)
        return 2

    chroma_dir = Path(config.CHROMA_DOCSVC_DIR)
    chroma_dir.mkdir(parents=True, exist_ok=True)
    chroma = chromadb.PersistentClient(path=str(chroma_dir))
    collection = chroma.get_or_create_collection(COLLECTION_NAME)

    lexical = None
    if not args.backfill_tags and not args.dry_run and Path(config.LEXICAL_INDEX_PATH).exists():
        from sme_chatbot.tools.lexical_index import LexicalIndex

        lexical = LexicalIndex(config.LEXICAL_INDEX_PATH)

    counts = {"added": 0, "patched": 0, "skipped": 0, "version_mismatch": 0, "failed": 0}

    try:
        for doc in _iter_filtered_documents(
            client,
            category=args.category,
            source=args.source,
            tag=_parse_tag(args.tag),
            page_size=args.page_size,
            limit=args.limit,
        ):
            try:
                if args.backfill_tags:
                    outcome = _backfill_document_tags(doc, collection=collection, dry_run=args.dry_run)
                else:
                    outcome = _process_document(
                        doc,
                        client=client,
                        collection=collection,
                        dry_run=args.dry_run,
                        lexical=lexical,
                    )
            except Exception as exc:  # belt-and-braces: never let one bad doc abort the run
                logger.warning(
                    "Unexpected error processing document %s: %s",
                    doc.get("documentId", "<unknown>"),
                    exc,
                )
                outcome = "failed"

            counts[outcome] = counts.get(outcome, 0) + 1
    finally:
        client.close()
        if lexical is not None:
            lexical.close()

    if args.backfill_tags:
        logger.info(
            "Backfill done. %d patched, %d skipped, %d version-mismatch, %d failed, collection size: %d",
            counts["patched"],
            counts["skipped"],
            counts["version_mismatch"],
            counts["failed"],
            collection.count(),
        )
    else:
        logger.info(
            "Done. %d added, %d skipped, %d failed, collection size: %d",
            counts["added"],
            counts["skipped"],
            counts["failed"],
            collection.count(),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
