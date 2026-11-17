"""Unit tests for ``sme_chatbot.ingestion.ingest_docsvc`` (WP6).

Exercises the per-document pipeline (``_process_document``) and the CLI
entry point (``main``) against:

- a fixture PDF generated with PyMuPDF into ``tmp_path`` at test time (no
  fixture files committed to the repo, no network),
- a fake ``DocSvcClient`` that copies the fixture PDF instead of hitting the
  real DocumentService,
- a fake ``SentenceTransformer`` that returns fixed-size vectors instead of
  downloading the real embedding model,
- a tmp-dir ChromaDB ``PersistentClient``.

Covers the WP6/WP8 acceptance cases: exact §4.3 chunk-ID scheme and metadata
keys (including flatten-scalars / skip-nested), idempotent second run (adds
0), stale-version chunk deletion on a new ``version_id``, deleted-DTO and
unsupported-extension skips, and ``--dry-run`` writing nothing.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import chromadb
import pymupdf
import pytest

from sme_chatbot.ingestion import ingest_docsvc as mod

# ---------------------------------------------------------------------------
# Fixture PDF, generated fresh into tmp_path (no on-disk fixture in the repo)
# ---------------------------------------------------------------------------


@pytest.fixture()
def fixture_pdf(tmp_path) -> Path:
    """A small single-page PDF with enough text to span multiple chunks
    (CHUNK_SIZE=1000, CHUNK_OVERLAP=200 -> ~2.2k chars should yield >= 2
    chunks), generated at test time so no binary fixture lives in the repo."""
    path = tmp_path / "sample.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 40
    page.insert_textbox(pymupdf.Rect(20, 20, 570, 800), text, fontsize=8)
    doc.save(str(path))
    doc.close()
    return path


class _FakeVector:
    """Stand-in for a numpy row returned by ``SentenceTransformer.encode()``."""

    def __init__(self, dim: int = 8):
        self._dim = dim

    def tolist(self) -> list[float]:
        return [0.1] * self._dim


class _FakeSentenceTransformer:
    """Drop-in replacement for ``sentence_transformers.SentenceTransformer``.

    Never touches the network / downloads a model; returns fixed-size fake
    embedding vectors so the ingestion pipeline can run end-to-end offline.
    """

    def __init__(self, *args, **kwargs):
        pass

    def encode(self, texts, batch_size=32, **kwargs):
        return [_FakeVector() for _ in texts]


class FakeDocSvcClient:
    """Minimal stand-in for ``sme_chatbot.clients.docsvc.DocSvcClient``.

    Only implements what ``_process_document``/``main()`` touch:
    ``download_content()``, ``iter_documents()`` (used by the CLI-level
    tests), ``close()``, and (for ``main()``'s startup credential check) a
    ``_token_provider`` with ``get_token()``.
    """

    def __init__(self, fixture_path: Path, documents: list[dict] | None = None):
        self._fixture_path = fixture_path
        self._documents = documents or []
        self._token_provider = SimpleNamespace(get_token=lambda: "fake-token")
        self.download_calls: list[str] = []
        self.closed = False

    def download_content(self, document_id: str, dest: Path) -> Path:
        self.download_calls.append(document_id)
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self._fixture_path, dest)
        return dest

    def iter_documents(self, source=None, category=None, tag=None, page_size=50):
        yield from self._documents

    def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def _patch_embedding_model(monkeypatch):
    """Ensure no real SentenceTransformer / model download is ever touched,
    and reset the module-level lazy singleton before each test."""
    monkeypatch.setattr("sentence_transformers.SentenceTransformer", _FakeSentenceTransformer)
    monkeypatch.setattr(mod, "_embedding_model", None)
    yield


@pytest.fixture
def collection(tmp_path):
    chroma = chromadb.PersistentClient(path=str(tmp_path / "chroma_docsvc_db"))
    return chroma.get_or_create_collection(mod.COLLECTION_NAME)


@pytest.fixture
def fake_client(fixture_pdf):
    return FakeDocSvcClient(fixture_pdf)


def _dto(
    *,
    document_id="11111111-1111-1111-1111-111111111111",
    version_id="v1",
    version_number=2,
    display_name="sample.pdf",
    extension="pdf",
    source="Sharepoint",
    category="Manual",
    metadata=None,
    tags=None,
    deleted=False,
):
    return {
        "documentId": document_id,
        "versionId": version_id,
        "versionNumber": version_number,
        "displayName": display_name,
        "status": "active",
        "extension": extension,
        "source": source,
        "category": category,
        "metadata": metadata if metadata is not None else {},
        "tags": tags if tags is not None else [],
        "deleted": deleted,
    }


# ---------------------------------------------------------------------------
# Exact §4.3 chunk-ID scheme + metadata, including flatten-scalars/skip-nested
# ---------------------------------------------------------------------------


def test_chunk_ids_and_metadata_exact_schema(fake_client, collection):
    doc = _dto(
        metadata={
            "language": "IT",
            "internal_version": "1",
            "customer": "Acme Corp",
            "flag": True,
            "count": 3,
            "ratio": 1.5,
            "nested": {"a": 1},
            "tags_list": ["a", "b"],
            "empty": None,
        },
        tags=["base_filename:z91123sample", "doc_type:manual", "serial:91123"],
    )

    outcome = mod._process_document(doc, client=fake_client, collection=collection, dry_run=False)
    assert outcome == "added"

    # Deterministic first-chunk id per §4.3: "{document_id}::v{version}::p{page}::c{idx}".
    first_id = "11111111-1111-1111-1111-111111111111::v2::p1::c0"
    got = collection.get(ids=[first_id], include=["metadatas", "documents"])
    assert got["ids"] == [first_id], "first chunk id must follow {document_id}::v{version}::p{page}::c{idx}"

    meta = got["metadatas"][0]
    assert meta["document_id"] == "11111111-1111-1111-1111-111111111111"
    assert meta["version_id"] == "v1"
    assert meta["version_number"] == 2
    assert isinstance(meta["version_number"], int)
    assert meta["source"] == "Sharepoint"
    assert meta["category"] == "Manual"
    assert meta["filename"] == "sample.pdf"
    assert meta["page"] == 1
    assert isinstance(meta["page"], int)
    assert meta["chunk"] == 0
    assert isinstance(meta["chunk"], int)

    # Flattened scalar business-key metadata (§4.3): every scalar copied as
    # str(value); non-scalars (nested dict/list) and None must NOT appear.
    # Both tag-derived and metadata-derived paths are real: serial comes from
    # tags (as DocumentService provides it), while language/internal_version
    # come from the metadata object (as other real applications provide them).
    assert meta["serial"] == "91123", "serial must be flattened from tags"
    assert meta["doc_type"] == "manual", "doc_type from tags must be lowercased"
    assert meta["base_filename"] == "z91123sample", "base_filename from tags"
    assert meta["language"] == "IT", "language from metadata"
    assert meta["internal_version"] == "1", "internal_version from metadata"
    assert meta["customer"] == "Acme Corp", "customer from metadata"
    assert meta["flag"] == "True"
    assert meta["count"] == "3"
    assert meta["ratio"] == "1.5"
    assert "nested" not in meta
    assert "tags_list" not in meta
    assert "empty" not in meta

    # All chunk ids for this document/version follow the exact naming scheme.
    all_chunks = collection.get(where={"document_id": "11111111-1111-1111-1111-111111111111"}, include=[])
    assert len(all_chunks["ids"]) >= 2, "fixture text (~2.2k chars) should yield multiple chunks"
    for cid in all_chunks["ids"]:
        assert cid.startswith("11111111-1111-1111-1111-111111111111::v2::p")
        parts = cid.split("::")
        assert len(parts) == 4
        assert parts[1] == "v2"
        assert parts[2].startswith("p")
        assert parts[3].startswith("c")


def test_metadata_scalars_only_types_accepted_by_chroma(fake_client, collection):
    """Belt-and-braces: collection.add() must not raise on the flattened
    metadata (i.e. every value really is a ChromaDB-acceptable scalar)."""
    doc = _dto(document_id="66666666-6666-6666-6666-666666666666", metadata={"weight_kg": 12.5, "active": False})
    outcome = mod._process_document(doc, client=fake_client, collection=collection, dry_run=False)
    assert outcome == "added"
    meta = collection.get(
        ids=["66666666-6666-6666-6666-666666666666::v2::p1::c0"], include=["metadatas"]
    )["metadatas"][0]
    assert meta["weight_kg"] == "12.5"
    assert meta["active"] == "False"


def test_tags_are_flattened_into_metadata(fake_client, collection):
    doc = _dto(
        document_id="77777777-7777-7777-7777-777777777777",
        metadata={"language": "TR", "internal_version": "1"},
        tags=["base_filename:Z91465Abstr", "doc_type:ABS", "serial:91465"],
    )
    outcome = mod._process_document(doc, client=fake_client, collection=collection, dry_run=False)
    assert outcome == "added"
    meta = collection.get(
        ids=["77777777-7777-7777-7777-777777777777::v2::p1::c0"], include=["metadatas"]
    )["metadatas"][0]

    assert meta["serial"] == "91465"
    assert meta["doc_type"] == "abs"
    assert meta["base_filename"] == "z91465abstr"
    assert meta["language"] == "TR"
    assert meta["internal_version"] == "1"


def test_tag_cannot_overwrite_reserved_or_dto_metadata_key(fake_client, collection):
    doc = _dto(
        document_id="88888888-8888-8888-8888-888888888888",
        category="Manual",
        metadata={"serial": "999"},
        tags=["category:evil", "serial:should-not-win", "source:evil"],
    )
    outcome = mod._process_document(doc, client=fake_client, collection=collection, dry_run=False)
    assert outcome == "added"
    meta = collection.get(
        ids=["88888888-8888-8888-8888-888888888888::v2::p1::c0"], include=["metadatas"]
    )["metadatas"][0]

    assert meta["category"] == "Manual"
    assert meta["source"] == "Sharepoint"
    assert meta["serial"] == "999"


def test_malformed_tags_are_skipped(fake_client, collection):
    doc = _dto(
        document_id="99999999-9999-9999-9999-999999999999",
        tags=["noseparator", 123, "valid_key:valid_value"],
    )
    outcome = mod._process_document(doc, client=fake_client, collection=collection, dry_run=False)
    assert outcome == "added"
    meta = collection.get(
        ids=["99999999-9999-9999-9999-999999999999::v2::p1::c0"], include=["metadatas"]
    )["metadatas"][0]

    assert meta["valid_key"] == "valid_value"
    assert "noseparator" not in meta


def test_flatten_tags_first_wins_on_duplicate_key():
    flattened = mod._flatten_tags(["serial:111", "serial:222"])
    assert flattened == {"serial": "111"}


def test_flatten_tags_non_list_input_returns_empty():
    assert mod._flatten_tags(None) == {}
    assert mod._flatten_tags("not-a-list") == {}


# ---------------------------------------------------------------------------
# Idempotency: second run against the same (document_id, version) adds 0
# ---------------------------------------------------------------------------


def test_second_run_is_idempotent(fake_client, collection):
    doc = _dto()

    first_outcome = mod._process_document(doc, client=fake_client, collection=collection, dry_run=False)
    assert first_outcome == "added"
    size_after_first = collection.count()
    assert size_after_first > 0

    downloads_before = len(fake_client.download_calls)

    second_outcome = mod._process_document(doc, client=fake_client, collection=collection, dry_run=False)
    assert second_outcome == "skipped"
    assert collection.count() == size_after_first, "re-running the same version must add 0 new chunks"
    assert len(fake_client.download_calls) == downloads_before, "idempotent skip must not re-download"


# ---------------------------------------------------------------------------
# Stale-version delete: a new version_id for the same document_id replaces
# the old version's chunks
# ---------------------------------------------------------------------------


def test_stale_version_is_deleted_on_new_version(fake_client, collection):
    document_id = "22222222-2222-2222-2222-222222222222"
    doc_v1 = _dto(document_id=document_id, version_id="version-A", version_number=1)
    doc_v2 = _dto(document_id=document_id, version_id="version-B", version_number=2)

    outcome_v1 = mod._process_document(doc_v1, client=fake_client, collection=collection, dry_run=False)
    assert outcome_v1 == "added"
    v1_chunks = collection.get(where={"document_id": document_id}, include=[])
    assert len(v1_chunks["ids"]) > 0
    assert all("::v1::" in cid for cid in v1_chunks["ids"])

    outcome_v2 = mod._process_document(doc_v2, client=fake_client, collection=collection, dry_run=False)
    assert outcome_v2 == "added"

    remaining = collection.get(where={"document_id": document_id}, include=["metadatas"])
    assert len(remaining["ids"]) > 0
    # Every remaining chunk must belong to the new version; the old
    # version's chunks must have been deleted per the §4.3 rule.
    for cid, meta in zip(remaining["ids"], remaining["metadatas"], strict=True):
        assert "::v1::" not in cid
        assert meta["version_id"] == "version-B"
        assert meta["version_number"] == 2


# ---------------------------------------------------------------------------
# Skip cases: dry-run, deleted DTO, unsupported extension
# ---------------------------------------------------------------------------


def test_dry_run_does_not_download_or_write(fake_client, collection):
    doc = _dto(document_id="33333333-3333-3333-3333-333333333333")
    outcome = mod._process_document(doc, client=fake_client, collection=collection, dry_run=True)
    assert outcome == "skipped"
    assert fake_client.download_calls == []
    assert collection.count() == 0


def test_deleted_document_is_skipped(fake_client, collection):
    doc = _dto(document_id="44444444-4444-4444-4444-444444444444", deleted=True)
    outcome = mod._process_document(doc, client=fake_client, collection=collection, dry_run=False)
    assert outcome == "skipped"
    assert fake_client.download_calls == []
    assert collection.count() == 0


def test_unsupported_extension_is_skipped(fake_client, collection):
    doc = _dto(document_id="55555555-5555-5555-5555-555555555555", extension="xlsx")
    outcome = mod._process_document(doc, client=fake_client, collection=collection, dry_run=False)
    assert outcome == "skipped"
    assert fake_client.download_calls == []
    assert collection.count() == 0


def test_missing_document_id_or_version_id_is_skipped(fake_client, collection):
    doc = _dto(document_id="77777777-7777-7777-7777-777777777777")
    doc["versionId"] = None
    outcome = mod._process_document(doc, client=fake_client, collection=collection, dry_run=False)
    assert outcome == "skipped"
    assert fake_client.download_calls == []
    assert collection.count() == 0


# ---------------------------------------------------------------------------
# --backfill-tags (WP3): patch tags onto already-ingested chunks in place
# ---------------------------------------------------------------------------


def _seed_chunk(
    collection,
    *,
    document_id,
    version_id,
    version_number=1,
    page=1,
    chunk_idx=0,
    extra_metadata=None,
    text="seed chunk text",
):
    """Insert a chunk directly (bypassing _process_document) to simulate a
    document ingested before WP1 shipped tag-flattening, i.e. chunk metadata
    with the reserved §4.3 keys but no tag-derived keys."""
    metadata = {
        "document_id": document_id,
        "version_id": version_id,
        "version_number": version_number,
        "source": "Sharepoint",
        "category": "Manual",
        "filename": "seed.pdf",
        "page": page,
        "chunk": chunk_idx,
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    collection.add(
        ids=[mod._chunk_id(document_id, version_number, page, chunk_idx)],
        embeddings=[[0.1] * 8],
        documents=[text],
        metadatas=[metadata],
    )


def _get_metadatas(collection, document_id: str) -> list[dict]:
    """Type-safe wrapper around ``collection.get(...)["metadatas"]``: chromadb
    types that field as ``Optional`` (it's only populated when "metadatas" is
    requested via ``include``), so callers must not subscript/iterate it
    directly."""
    result = collection.get(where={"document_id": document_id}, include=["metadatas"])
    return result["metadatas"] or []


def test_backfill_patches_missing_tags_without_touching_embeddings_or_documents(collection):
    document_id = "aaaaaaaa-0000-0000-0000-000000000001"
    _seed_chunk(collection, document_id=document_id, version_id="v1", page=1, chunk_idx=0)
    _seed_chunk(collection, document_id=document_id, version_id="v1", page=1, chunk_idx=1)
    before = collection.get(where={"document_id": document_id}, include=["metadatas", "documents", "embeddings"])

    doc = _dto(document_id=document_id, version_id="v1", version_number=1, tags=["serial:91465", "doc_type:abs"])
    outcome = mod._backfill_document_tags(doc, collection=collection, dry_run=False)
    assert outcome == "patched"

    after = collection.get(where={"document_id": document_id}, include=["metadatas", "documents", "embeddings"])
    assert after["ids"] == before["ids"], "backfill must not add/remove chunks"
    assert after["documents"] == before["documents"], "backfill must not touch document text"
    before_embeddings = before["embeddings"] if before["embeddings"] is not None else []
    after_embeddings = after["embeddings"] if after["embeddings"] is not None else []
    for before_emb, after_emb in zip(before_embeddings, after_embeddings, strict=True):
        assert list(before_emb) == list(after_emb), "backfill must not touch embeddings"
    for meta in after["metadatas"] or []:
        assert meta["serial"] == "91465"
        assert meta["doc_type"] == "abs"
        # Pre-existing reserved metadata is untouched by the patch.
        assert meta["category"] == "Manual"


def test_backfill_dry_run_writes_nothing(collection):
    document_id = "aaaaaaaa-0000-0000-0000-000000000002"
    _seed_chunk(collection, document_id=document_id, version_id="v1")

    doc = _dto(document_id=document_id, version_id="v1", version_number=1, tags=["serial:999"])
    outcome = mod._backfill_document_tags(doc, collection=collection, dry_run=True)
    assert outcome == "patched"

    meta = _get_metadatas(collection, document_id)[0]
    assert "serial" not in meta, "--dry-run must write nothing"


def test_backfill_tags_cannot_overwrite_already_present_keys(collection):
    document_id = "aaaaaaaa-0000-0000-0000-000000000003"
    _seed_chunk(collection, document_id=document_id, version_id="v1", extra_metadata={"serial": "111"})

    doc = _dto(document_id=document_id, version_id="v1", version_number=1, tags=["serial:222"])
    outcome = mod._backfill_document_tags(doc, collection=collection, dry_run=False)
    assert outcome == "skipped", "no new tag keys to patch (serial already present) -> skipped"

    meta = _get_metadatas(collection, document_id)[0]
    assert meta["serial"] == "111", "existing value must never be overwritten by a later backfill run"


def test_backfill_tags_cannot_overwrite_reserved_keys(collection):
    document_id = "aaaaaaaa-0000-0000-0000-000000000004"
    _seed_chunk(collection, document_id=document_id, version_id="v1")

    doc = _dto(document_id=document_id, version_id="v1", version_number=1, tags=["category:evil", "serial:91465"])
    outcome = mod._backfill_document_tags(doc, collection=collection, dry_run=False)
    assert outcome == "patched"

    meta = _get_metadatas(collection, document_id)[0]
    assert meta["category"] == "Manual", "a tag must never overwrite a reserved chunk-metadata key"
    assert meta["serial"] == "91465"


def test_backfill_rerun_is_a_noop(collection):
    document_id = "aaaaaaaa-0000-0000-0000-000000000005"
    _seed_chunk(collection, document_id=document_id, version_id="v1")
    doc = _dto(document_id=document_id, version_id="v1", version_number=1, tags=["serial:91465"])

    first = mod._backfill_document_tags(doc, collection=collection, dry_run=False)
    assert first == "patched"
    size_after_first = collection.count()
    meta_after_first = _get_metadatas(collection, document_id)[0]

    second = mod._backfill_document_tags(doc, collection=collection, dry_run=False)
    assert second == "skipped", "re-running with nothing new to patch is a no-op"
    assert collection.count() == size_after_first
    meta_after_second = _get_metadatas(collection, document_id)[0]
    assert meta_after_second == meta_after_first


def test_backfill_version_mismatch_is_counted_separately(collection):
    document_id = "aaaaaaaa-0000-0000-0000-000000000006"
    _seed_chunk(collection, document_id=document_id, version_id="stale-version")

    doc = _dto(document_id=document_id, version_id="current-version", version_number=2, tags=["serial:91465"])
    outcome = mod._backfill_document_tags(doc, collection=collection, dry_run=False)
    assert outcome == "version_mismatch"

    meta = _get_metadatas(collection, document_id)[0]
    assert "serial" not in meta, "a version mismatch must never be patched"


def test_backfill_partial_version_mismatch_across_chunks_is_still_flagged(collection):
    """Regression for checking every chunk's version_id, not just the first:
    an earlier interrupted run could leave one document's chunks split
    across two version_ids (crash mid-way through the stale-version
    delete)."""
    document_id = "aaaaaaaa-0000-0000-0000-000000000007"
    _seed_chunk(collection, document_id=document_id, version_id="current-version", page=1, chunk_idx=0)
    _seed_chunk(collection, document_id=document_id, version_id="stale-version", page=1, chunk_idx=1)

    doc = _dto(document_id=document_id, version_id="current-version", version_number=1, tags=["serial:91465"])
    outcome = mod._backfill_document_tags(doc, collection=collection, dry_run=False)
    assert outcome == "version_mismatch"

    for meta in _get_metadatas(collection, document_id):
        assert "serial" not in meta


def test_backfill_not_yet_ingested_document_is_skipped(collection):
    doc = _dto(document_id="aaaaaaaa-0000-0000-0000-000000000008", version_id="v1", tags=["serial:91465"])
    outcome = mod._backfill_document_tags(doc, collection=collection, dry_run=False)
    assert outcome == "skipped"


def test_backfill_deleted_document_is_skipped(collection):
    document_id = "aaaaaaaa-0000-0000-0000-000000000009"
    _seed_chunk(collection, document_id=document_id, version_id="v1")

    doc = _dto(document_id=document_id, version_id="v1", tags=["serial:91465"], deleted=True)
    outcome = mod._backfill_document_tags(doc, collection=collection, dry_run=False)
    assert outcome == "skipped"

    meta = _get_metadatas(collection, document_id)[0]
    assert "serial" not in meta


def test_backfill_missing_ids_is_skipped(collection):
    doc = _dto(document_id="aaaaaaaa-0000-0000-0000-000000000010", tags=["serial:91465"])
    doc["versionId"] = None
    outcome = mod._backfill_document_tags(doc, collection=collection, dry_run=False)
    assert outcome == "skipped"


def test_backfill_batches_updates_at_ADD_BATCH_SIZE(collection, monkeypatch):
    """Writes must be bounded (issue: unbounded collection.get()/update() over
    the whole corpus throws chromadb's 'too many SQL variables' error)."""
    monkeypatch.setattr(mod, "ADD_BATCH_SIZE", 2)
    document_id = "aaaaaaaa-0000-0000-0000-000000000011"
    for i in range(5):
        _seed_chunk(collection, document_id=document_id, version_id="v1", page=1, chunk_idx=i)

    update_calls: list[int] = []
    real_update = collection.update

    def _counting_update(*, ids, metadatas):
        update_calls.append(len(ids))
        return real_update(ids=ids, metadatas=metadatas)

    monkeypatch.setattr(collection, "update", _counting_update)

    doc = _dto(document_id=document_id, version_id="v1", version_number=1, tags=["serial:91465"])
    outcome = mod._backfill_document_tags(doc, collection=collection, dry_run=False)
    assert outcome == "patched"
    assert update_calls == [2, 2, 1], "writes must be chunked at ADD_BATCH_SIZE, not sent in one unbounded call"

    for meta in _get_metadatas(collection, document_id):
        assert meta["serial"] == "91465"


# ---------------------------------------------------------------------------
# CLI-level: main() with a mocked DocSvcClient class, tmp Chroma dir
# ---------------------------------------------------------------------------


@pytest.fixture()
def cli_env(tmp_path, monkeypatch, fixture_pdf):
    """Point the CLI's Chroma dir at tmp_path and stub DocSvcClient
    construction so main() never touches real credentials/network."""
    monkeypatch.setattr(mod.config, "CHROMA_DOCSVC_DIR", str(tmp_path / "chroma_docsvc_db"))

    documents = [
        _dto(document_id="88888888-8888-8888-8888-888888888888", version_id="v1", version_number=1),
    ]
    fake = FakeDocSvcClient(fixture_pdf, documents=documents)
    monkeypatch.setattr(mod, "DocSvcClient", lambda *a, **kw: fake)
    return SimpleNamespace(tmp_path=tmp_path, fake_client=fake, documents=documents)


def test_cli_dry_run_writes_nothing(cli_env):
    exit_code = mod.main(["--dry-run"])
    assert exit_code == 0
    assert cli_env.fake_client.download_calls == []

    chroma_dir = Path(cli_env.tmp_path) / "chroma_docsvc_db"
    # get_or_create_collection() itself creates the on-disk store, but
    # --dry-run must add zero chunks/documents to it.
    chroma = chromadb.PersistentClient(path=str(chroma_dir))
    collection = chroma.get_or_create_collection(mod.COLLECTION_NAME)
    assert collection.count() == 0


def test_cli_real_run_adds_chunks_and_second_run_adds_zero(cli_env):
    exit_code = mod.main([])
    assert exit_code == 0
    assert len(cli_env.fake_client.download_calls) == 1

    chroma_dir = Path(cli_env.tmp_path) / "chroma_docsvc_db"
    chroma = chromadb.PersistentClient(path=str(chroma_dir))
    collection = chroma.get_or_create_collection(mod.COLLECTION_NAME)
    size_after_first_run = collection.count()
    assert size_after_first_run > 0

    # Second CLI invocation against the same documents: idempotent, 0 added.
    exit_code = mod.main([])
    assert exit_code == 0
    assert len(cli_env.fake_client.download_calls) == 1, "idempotent skip must not re-download"
    collection = chroma.get_or_create_collection(mod.COLLECTION_NAME)
    assert collection.count() == size_after_first_run


def test_cli_backfill_tags_patches_existing_chunks(tmp_path, monkeypatch, fixture_pdf):
    """CLI-level: --backfill-tags patches a pre-WP1 corpus (chunks that exist
    but were ingested before tag-flattening shipped) without downloading or
    re-embedding anything."""
    monkeypatch.setattr(mod.config, "CHROMA_DOCSVC_DIR", str(tmp_path / "chroma_docsvc_db"))
    chroma_dir = Path(tmp_path) / "chroma_docsvc_db"
    chroma_dir.mkdir(parents=True, exist_ok=True)
    chroma = chromadb.PersistentClient(path=str(chroma_dir))
    collection = chroma.get_or_create_collection(mod.COLLECTION_NAME)

    document_id = "bbbbbbbb-0000-0000-0000-000000000001"
    _seed_chunk(collection, document_id=document_id, version_id="v1", page=1, chunk_idx=0)
    _seed_chunk(collection, document_id=document_id, version_id="v1", page=1, chunk_idx=1)

    documents = [_dto(document_id=document_id, version_id="v1", version_number=1, tags=["serial:91465"])]
    fake = FakeDocSvcClient(fixture_pdf, documents=documents)
    monkeypatch.setattr(mod, "DocSvcClient", lambda *a, **kw: fake)

    exit_code = mod.main(["--backfill-tags"])
    assert exit_code == 0
    assert fake.download_calls == [], "backfill must never download document content"

    chroma = chromadb.PersistentClient(path=str(chroma_dir))
    collection = chroma.get_or_create_collection(mod.COLLECTION_NAME)
    assert collection.count() == 2, "backfill must not add/remove chunks"
    for meta in _get_metadatas(collection, document_id):
        assert meta["serial"] == "91465"

    # Re-running is a no-op.
    exit_code = mod.main(["--backfill-tags"])
    assert exit_code == 0
    assert collection.count() == 2


def test_cli_backfill_tags_dry_run_writes_nothing(tmp_path, monkeypatch, fixture_pdf):
    monkeypatch.setattr(mod.config, "CHROMA_DOCSVC_DIR", str(tmp_path / "chroma_docsvc_db"))
    chroma_dir = Path(tmp_path) / "chroma_docsvc_db"
    chroma_dir.mkdir(parents=True, exist_ok=True)
    chroma = chromadb.PersistentClient(path=str(chroma_dir))
    collection = chroma.get_or_create_collection(mod.COLLECTION_NAME)

    document_id = "bbbbbbbb-0000-0000-0000-000000000002"
    _seed_chunk(collection, document_id=document_id, version_id="v1")

    documents = [_dto(document_id=document_id, version_id="v1", version_number=1, tags=["serial:91465"])]
    fake = FakeDocSvcClient(fixture_pdf, documents=documents)
    monkeypatch.setattr(mod, "DocSvcClient", lambda *a, **kw: fake)

    exit_code = mod.main(["--backfill-tags", "--dry-run"])
    assert exit_code == 0
    assert fake.download_calls == []

    chroma = chromadb.PersistentClient(path=str(chroma_dir))
    collection = chroma.get_or_create_collection(mod.COLLECTION_NAME)
    meta = _get_metadatas(collection, document_id)[0]
    assert "serial" not in meta, "--backfill-tags --dry-run must write nothing"


def test_cli_missing_credentials_exits_2(tmp_path, monkeypatch):
    monkeypatch.setattr(mod.config, "CHROMA_DOCSVC_DIR", str(tmp_path / "chroma_docsvc_db"))

    class _AuthFailingClient:
        def __init__(self, *a, **kw):
            self._token_provider = SimpleNamespace(get_token=_raise_auth_error)

    monkeypatch.setattr(mod, "DocSvcClient", _AuthFailingClient)

    exit_code = mod.main([])
    assert exit_code == 2


def _raise_auth_error():
    from sme_chatbot.clients.docsvc import DocSvcAuthError

    raise DocSvcAuthError("missing creds")
