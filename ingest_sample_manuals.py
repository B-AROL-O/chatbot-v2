"""Ingest machine manuals into a dedicated Chroma vector store.

Mirrors ``ingest_sample_cdo.py`` but targets the manuals corpus extracted under
``data/input/manuals`` (see ``scripts/import_manuals.sh``) and stores vectors in
a separate collection so it never touches the CDO ``cdo_docs`` data.

Supports PDF (via pymupdf4llm), ``.docx`` (via python-docx), and legacy ``.doc``
(via pypandoc/Pandoc).
"""

print("Initialising (loading dependencies, this takes ~30s on first run)...", flush=True)

import logging
import os
import subprocess
from pathlib import Path

import chromadb
import pymupdf4llm
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DOCS_DIR = Path("data/input/manuals")
CHROMA_DIR = Path("data/chroma_manuals_db")
COLLECTION_NAME = os.environ.get("CHROMA_MANUALS_COLLECTION", "manuals")
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


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


def main() -> None:
    doc_files = sorted(
        p
        for ext in ("*.pdf", "*.docx", "*.doc")
        for p in DOCS_DIR.rglob(ext)
    )
    if not doc_files:
        logger.error("No PDF/DOC/DOCX files found in %s", DOCS_DIR)
        return

    logger.info("Found %d documents in %s", len(doc_files), DOCS_DIR)
    logger.info("Loading embedding model %s", EMBEDDING_MODEL)
    model = SentenceTransformer(EMBEDDING_MODEL)

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = chroma.get_or_create_collection(COLLECTION_NAME)

    existing_ids: set[str] = set(collection.get(include=[])["ids"])
    logger.info(
        "Collection %r has %d existing chunks", COLLECTION_NAME, len(existing_ids)
    )

    total_added = 0
    total_skipped = 0

    for doc_path in doc_files:
        logger.info("Processing %s", doc_path.name)
        pages = extract_pages(doc_path)

        file_added = 0
        file_skipped = 0

        ids: list[str] = []
        embeddings: list[list[float]] = []
        documents: list[str] = []
        metadatas: list[dict] = []

        for page_num, text in pages:
            chunks = chunk_text(text)

            for chunk_idx, chunk in enumerate(chunks):
                doc_id = f"{doc_path.name}::p{page_num}::c{chunk_idx}"
                if doc_id in existing_ids:
                    file_skipped += 1
                    continue

                ids.append(doc_id)
                documents.append(chunk)
                metadatas.append(
                    {"filename": doc_path.name, "page": page_num, "chunk": chunk_idx}
                )

        if ids:
            logger.info("Embedding %d new chunks from %s", len(ids), doc_path.name)
            vecs = model.encode(documents, show_progress_bar=True, batch_size=32)
            embeddings = [v.tolist() for v in vecs]

            # upsert in batches to avoid memory issues
            batch = 500
            for i in range(0, len(ids), batch):
                collection.add(
                    ids=ids[i : i + batch],
                    embeddings=embeddings[i : i + batch],
                    documents=documents[i : i + batch],
                    metadatas=metadatas[i : i + batch],
                )
            file_added = len(ids)

        logger.info(
            "%s: %d added, %d skipped", doc_path.name, file_added, file_skipped
        )
        total_added += file_added
        total_skipped += file_skipped

    logger.info(
        "Done. Total: %d added, %d skipped. Collection size: %d",
        total_added,
        total_skipped,
        collection.count(),
    )


if __name__ == "__main__":
    main()
