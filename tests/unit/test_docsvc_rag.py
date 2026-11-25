"""Unit/integration tests for ``sme_chatbot.tools.docsvc_rag`` (WP7).

Exercises ``search_documents`` against a tmp-dir Chroma collection seeded
with §4.3-shaped metadata, using a deterministic fake embedder so no model
download / network access is required. Covers the WP7/WP8 acceptance cases:

- ``category`` filter is respected
- a chunk outside ``permitted_document_ids`` is NEVER returned, regardless
  of similarity
- an empty permitted set returns zero results
- an absent collection/store degrades to a friendly "run ingestion first"
  message, never an exception
- identity binding survives the ``make_dispatch``/``dispatch`` -> JSON
  round trip used by the chat loop's tool dispatch

All Chroma stores live under ``tmp_path`` — nothing is written under
``data/``, and the module is reloaded per-test with ``CHROMA_DOCSVC_DIR``
monkeypatched so the suite is independent of any real ``.env``.
"""

from __future__ import annotations

import importlib
import json
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from sme_chatbot.core.events import RequestIdentity


class _FakeModel:
    """Deterministic fake embedder: encodes text -> a fixed 2-D vector
    derived from a hash of the string, so no model download is needed and
    results are reproducible across the test and the seeding helper."""

    def encode(self, texts):
        vecs = []
        for t in texts:
            h = abs(hash(t)) % 1000
            vecs.append([h / 1000.0, (1000 - h) / 1000.0])
        return np.array(vecs)


@pytest.fixture()
def docsvc_rag_module(tmp_path, monkeypatch):
    """Import ``docsvc_rag`` fresh, pointed at a tmp Chroma dir.

    Reloads ``config`` first so ``CHROMA_DOCSVC_DIR`` picks up the
    monkeypatched env var, then reloads ``docsvc_rag`` so its module-level
    ``CHROMA_DIR`` constant (captured at import time from ``config``) is
    recomputed against the tmp path.
    """
    monkeypatch.setenv("CHROMA_DOCSVC_DIR", str(tmp_path / "chroma_docsvc_db"))

    from sme_chatbot import config as config_module

    importlib.reload(config_module)

    from sme_chatbot.tools import docsvc_rag as docsvc_rag_module

    importlib.reload(docsvc_rag_module)

    yield docsvc_rag_module

    # Leave no lazy-loaded state (fake model/collection) bleeding into other
    # tests that import the module fresh again.
    docsvc_rag_module._model = None
    docsvc_rag_module._collection = None


def _seed_collection(chroma_dir, chunks):
    """chunks: list of (id, text, metadata) triples. Embeds with the same
    deterministic fake used by the tests so query vectors are comparable."""
    import chromadb

    from sme_chatbot import config

    fake = _FakeModel()
    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_or_create_collection(config.CHROMA_DOCSVC_COLLECTION)
    ids = [c[0] for c in chunks]
    texts = [c[1] for c in chunks]
    metas = [c[2] for c in chunks]
    embeddings = fake.encode(texts).tolist()
    collection.add(ids=ids, documents=texts, metadatas=metas, embeddings=embeddings)
    return collection


_CHUNKS = [
    (
        "doc-A::v1::p1::c0",
        "How to replace the head unit on the AL machine.",
        {
            "document_id": "doc-A",
            "version_id": "v1",
            "version_number": 1,
            "source": "Sharepoint",
            "category": "Manual",
            "filename": "manual_al.pdf",
            "page": 1,
            "chunk": 0,
            "serial": "91123",
        },
    ),
    (
        "doc-B::v1::p3::c0",
        "How to replace the head unit -- CDO report variant.",
        {
            "document_id": "doc-B",
            "version_id": "v1",
            "version_number": 1,
            "source": "Sharepoint",
            "category": "CDO",
            "filename": "cdo_report.pdf",
            "page": 3,
            "chunk": 0,
            "serial": "91999",
        },
    ),
    (
        "doc-SECRET::v1::p1::c0",
        "How to replace the head unit -- confidential customer variant.",
        {
            "document_id": "doc-SECRET",
            "version_id": "v1",
            "version_number": 1,
            "source": "Sharepoint",
            "category": "Manual",
            "filename": "secret.pdf",
            "page": 1,
            "chunk": 0,
            "serial": "91123",
        },
    ),
]


def _seed_and_patch(module, monkeypatch, chunks=_CHUNKS):
    """Seed the tmp Chroma collection and wire the module's lazy-loaded
    globals directly to the fake model + collection (bypassing
    ``_ensure_loaded``'s real SentenceTransformer/model-download path)."""
    import chromadb

    chroma_dir = module.CHROMA_DIR
    chroma_dir.mkdir(parents=True, exist_ok=True)
    _seed_collection(chroma_dir, chunks)

    monkeypatch.setattr(module, "_model", _FakeModel())
    client = chromadb.PersistentClient(path=str(chroma_dir))
    monkeypatch.setattr(module, "_collection", client.get_collection(module.COLLECTION_NAME))


# ---------------------------------------------------------------------------
# Absent store / collection -> friendly message, never an exception
# ---------------------------------------------------------------------------


def test_absent_store_gives_friendly_message_not_exception(docsvc_rag_module):
    # CHROMA_DIR does not exist at all (nothing has been ingested yet).
    result = docsvc_rag_module.search_documents("come si sostituisce la testata?")
    assert "error" in result
    assert "ingestion" in result["error"].lower() or "run" in result["error"].lower()


def test_absent_collection_in_existing_dir_gives_friendly_message(docsvc_rag_module):
    # The Chroma dir exists (e.g. leftover empty PersistentClient store) but
    # the collection was never created.
    import chromadb

    docsvc_rag_module.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    chromadb.PersistentClient(path=str(docsvc_rag_module.CHROMA_DIR))  # no collection created

    result = docsvc_rag_module.search_documents("query")
    assert "error" in result
    assert "run" in result["error"].lower() or "ingestion" in result["error"].lower()


# ---------------------------------------------------------------------------
# category filter respected
# ---------------------------------------------------------------------------


def test_category_filter_excludes_other_categories(docsvc_rag_module, monkeypatch):
    _seed_and_patch(docsvc_rag_module, monkeypatch)

    res = docsvc_rag_module.search_documents("replace the head unit", n_results=10, category="Manual")
    assert "error" not in res
    returned_ids = {r["document_id"] for r in res["results"]}
    assert returned_ids <= {"doc-A", "doc-SECRET"}
    assert "doc-B" not in returned_ids


# ---------------------------------------------------------------------------
# permitted_document_ids: never returns chunks outside the permitted set
# ---------------------------------------------------------------------------


def test_permitted_document_ids_excludes_out_of_scope_chunk_regardless_of_similarity(
    docsvc_rag_module, monkeypatch
):
    _seed_and_patch(docsvc_rag_module, monkeypatch)

    # doc-SECRET's text is the closest lexical/semantic match to the query
    # among the three (same phrasing as doc-A/doc-B) -- it must still never
    # come back once identity excludes it, no category filter applied.
    identity = RequestIdentity(permitted_document_ids=["doc-A", "doc-B"])
    res = docsvc_rag_module.search_documents("replace the head unit", n_results=10, identity=identity)
    assert "error" not in res
    returned_ids = {r["document_id"] for r in res["results"]}
    assert "doc-SECRET" not in returned_ids
    assert returned_ids <= {"doc-A", "doc-B"}


def test_empty_permitted_set_returns_zero_results(docsvc_rag_module, monkeypatch):
    _seed_and_patch(docsvc_rag_module, monkeypatch)

    identity_none = RequestIdentity(permitted_document_ids=[])
    res = docsvc_rag_module.search_documents("replace the head unit", n_results=10, identity=identity_none)
    assert "error" not in res
    assert res["results"] == []


def test_result_shape_has_all_documented_fields(docsvc_rag_module, monkeypatch):
    _seed_and_patch(docsvc_rag_module, monkeypatch)

    identity_a_only = RequestIdentity(permitted_document_ids=["doc-A"])
    res = docsvc_rag_module.search_documents("replace the head unit", n_results=10, identity=identity_a_only)
    assert len(res["results"]) == 1
    entry = res["results"][0]
    for key in (
        "text",
        "document_id",
        "version_id",
        "filename",
        "repository",
        "category",
        "page",
        "language",
        "distance",
    ):
        assert key in entry
    assert entry["document_id"] == "doc-A"
    assert entry["filename"] == "manual_al.pdf"
    # The origin network share is reported as `repository`, and no
    # longer under a `source` key that the citation layer would mistake for
    # the document's own name.
    assert entry["repository"] == "Sharepoint"
    assert "source" not in entry


# ---------------------------------------------------------------------------
# language filter: IT/DE translations of the same manual
# must not cross-contaminate each other's results.
# ---------------------------------------------------------------------------


def test_language_filter_excludes_other_language_translation(docsvc_rag_module, monkeypatch):
    chunks = [
        (
            "doc-IT::v1::p1::c0",
            "coppia di serraggio vite M12: 47 Nm",
            {
                "document_id": "doc-IT",
                "version_id": "v1",
                "version_number": 1,
                "source": "Sharepoint",
                "category": "Manual",
                "filename": "90001_manual_IT.pdf",
                "page": 44,
                "chunk": 0,
                "serial": "90001",
                "language": "IT",
            },
        ),
        (
            "doc-DE::v1::p1::c0",
            "Anzugsdrehmoment Schraube M12: 47 Nm",
            {
                "document_id": "doc-DE",
                "version_id": "v1",
                "version_number": 1,
                "source": "Sharepoint",
                "category": "Manual",
                "filename": "90001_manual_DE.pdf",
                "page": 44,
                "chunk": 0,
                "serial": "90001",
                "language": "DE",
            },
        ),
    ]
    _seed_and_patch(docsvc_rag_module, monkeypatch, chunks=chunks)

    res = docsvc_rag_module.search_documents(
        "coppia di serraggio vite M12", n_results=10, language="it"
    )
    assert "error" not in res
    returned_ids = {r["document_id"] for r in res["results"]}
    assert returned_ids == {"doc-IT"}
    assert "doc-DE" not in returned_ids


def _lang_chunk(doc, language, text="coppia di serraggio vite M12: 47 Nm"):
    meta = {"document_id": doc, "version_id": "v1", "version_number": 1, "source": "Sharepoint", "category": "manual",
            "filename": f"{doc}.pdf", "page": 44, "chunk": 0, "serial": "90001", "language": language}
    return (f"{doc}::v1::p44::c0", text, meta)


def test_language_filter_matches_a_lowercase_stored_language(docsvc_rag_module, monkeypatch):
    # The 2026-10-15 archive stores "it"/"de"; an uppercase-only clause returned nothing
    _seed_and_patch(docsvc_rag_module, monkeypatch, chunks=[_lang_chunk("doc-it", "it"), _lang_chunk("doc-de", "de")])

    res = docsvc_rag_module.search_documents("coppia di serraggio vite M12", n_results=10, language="IT")

    assert {r["document_id"] for r in res["results"]} == {"doc-it"}


def test_a_language_filter_that_leaves_nothing_is_retried_without_it(docsvc_rag_module, monkeypatch):
    # Bilingual manuals carry a combined code ("it-ro") that no single-language filter matches
    _seed_and_patch(docsvc_rag_module, monkeypatch, chunks=[_lang_chunk("doc-itro", "it-ro")])

    res = docsvc_rag_module.search_documents("coppia di serraggio vite M12", n_results=10, language="it")

    assert {r["document_id"] for r in res["results"]} == {"doc-itro"}


def test_the_fallback_never_widens_access(docsvc_rag_module, monkeypatch):
    _seed_and_patch(docsvc_rag_module, monkeypatch, chunks=[_lang_chunk("doc-itro", "it-ro")])
    identity = RequestIdentity(permitted_document_ids=[])

    res = docsvc_rag_module.search_documents("coppia di serraggio vite M12", n_results=10, language="it", identity=identity)

    assert res["results"] == []


# ---------------------------------------------------------------------------
# Identity binding through make_dispatch/dispatch JSON round trip
# ---------------------------------------------------------------------------


def test_dispatch_and_make_dispatch_bind_identity_through_json_round_trip(docsvc_rag_module, monkeypatch):
    chunks = [
        (
            "doc-A::v1::p1::c0",
            "spare parts list",
            {
                "document_id": "doc-A",
                "version_id": "v1",
                "version_number": 1,
                "source": "Sharepoint",
                "category": "Manual",
                "filename": "manual_al.pdf",
                "page": 1,
                "chunk": 0,
            },
        ),
        (
            "doc-B::v1::p1::c0",
            "spare parts list",
            {
                "document_id": "doc-B",
                "version_id": "v1",
                "version_number": 1,
                "source": "Sharepoint",
                "category": "Manual",
                "filename": "manual_b.pdf",
                "page": 1,
                "chunk": 0,
            },
        ),
    ]
    _seed_and_patch(docsvc_rag_module, monkeypatch, chunks=chunks)

    # Default module-level dispatch: unrestricted (identity=None ->
    # search_documents substitutes RequestIdentity()).
    raw = docsvc_rag_module.dispatch("search_documents", {"query": "spare parts", "n_results": 10})
    payload = json.loads(raw)
    assert {r["document_id"] for r in payload["results"]} == {"doc-A", "doc-B"}

    # make_dispatch(identity) binds identity into the closure; the
    # LLM-supplied arguments never carry it, yet the JSON-encoded result is
    # scoped to what the bound identity permits.
    scoped_dispatch = docsvc_rag_module.make_dispatch(RequestIdentity(permitted_document_ids=["doc-A"]))
    raw = scoped_dispatch("search_documents", {"query": "spare parts", "n_results": 10})
    payload = json.loads(raw)
    assert {r["document_id"] for r in payload["results"]} == {"doc-A"}

    # Unknown tool name -> graceful JSON error, never an exception.
    raw = docsvc_rag_module.dispatch("nope", {})
    assert json.loads(raw)["error"] == "Unknown tool: nope"


# ---------------------------------------------------------------------------
# Hybrid retrieval (plan WP1): lexical candidates fused with dense ones, and
# the lexical path can never widen access.
# ---------------------------------------------------------------------------


def _manual_meta(document_id, page, chunk=0, serial="90001", filename="90001_manual_IT.pdf"):
    return {
        "document_id": document_id,
        "version_id": "v1",
        "version_number": 1,
        "source": "Sharepoint",
        "category": "manual",
        "filename": filename,
        "page": page,
        "chunk": chunk,
        "serial": serial,
        "language": "IT",
    }


_MANUAL_CHUNKS = [
    ("m::v1::p18::c0", "4.3 DATI TECNICI<br>PRODUZIONE NOMINALE<br>Pezzi/ora 9000 bph\n\n8 \n", _manual_meta("m", 18)),
    ("m::v1::p19::c0", "4.4 TABELLA COPPIE DI SERRAGGIO VITI\n\n9 \n", _manual_meta("m", 19)),
] + [(f"m::v1::p{p}::c0", f"testo generico pagina {p}", _manual_meta("m", p)) for p in range(20, 60)]


def _build_lexical(module, chunks):
    from sme_chatbot.tools.lexical_index import LexicalIndex

    index = LexicalIndex(module.config.LEXICAL_INDEX_PATH)
    index.add([c[0] for c in chunks], [c[1] for c in chunks], [c[2] for c in chunks])
    index.close()
    module._lexical = None  # force a fresh lazy load against the new file


def test_hybrid_finds_the_table_chunk_dense_alone_ranks_low(docsvc_rag_module, monkeypatch):
    # The fake embedder ranks by hash, i.e. arbitrarily: only the lexical
    # half can reliably put page 18 in the top 3.
    _seed_and_patch(docsvc_rag_module, monkeypatch, chunks=_MANUAL_CHUNKS)
    _build_lexical(docsvc_rag_module, _MANUAL_CHUNKS)
    res = docsvc_rag_module.search_documents("produzione nominale", n_results=3, serial="90001")
    pages = [r["page"] for r in res["results"]]
    assert 18 in pages


def test_lexical_path_never_returns_a_chunk_outside_identity(docsvc_rag_module, monkeypatch):
    """The FTS pre-filter is a recall aid only: even when the lexical index's
    own metadata would let a chunk through, the Chroma re-fetch with the real
    ``where`` must drop it."""
    secret = ("s::v1::p1::c0", "PRODUZIONE NOMINALE riservata", _manual_meta("s", 1, filename="secret.pdf"))
    _seed_and_patch(docsvc_rag_module, monkeypatch, chunks=_MANUAL_CHUNKS + [secret])
    # Index the secret chunk under a *lying* document_id, so the SQL
    # pre-filter (document_id IN permitted) would accept it.
    lying = (secret[0], secret[1], {**secret[2], "document_id": "m"})
    _build_lexical(docsvc_rag_module, _MANUAL_CHUNKS + [lying])

    identity = RequestIdentity(permitted_document_ids=["m"])
    res = docsvc_rag_module.search_documents("produzione nominale riservata", n_results=20, identity=identity)
    assert res["results"]
    assert all(r["document_id"] == "m" for r in res["results"])


def test_dense_mode_ignores_the_lexical_index(docsvc_rag_module, monkeypatch):
    _seed_and_patch(docsvc_rag_module, monkeypatch, chunks=_MANUAL_CHUNKS)
    _build_lexical(docsvc_rag_module, _MANUAL_CHUNKS)
    monkeypatch.setattr(docsvc_rag_module.config, "RETRIEVAL_MODE", "dense")
    assert docsvc_rag_module._get_lexical() is None


# ---------------------------------------------------------------------------
# read_pages (plan WP4)
# ---------------------------------------------------------------------------


def test_read_pages_rejoins_chunks_without_the_overlap(docsvc_rag_module, monkeypatch):
    from sme_chatbot.ingestion.ingest_docsvc import chunk_text

    page_text = "".join(f"riga {i:04d} della tabella. " for i in range(80))
    chunks = [
        (f"m::v1::p19::c{i}", chunk, _manual_meta("m", 19, chunk=i)) for i, chunk in enumerate(chunk_text(page_text))
    ]
    assert len(chunks) > 1
    _seed_and_patch(docsvc_rag_module, monkeypatch, chunks=chunks)

    res = docsvc_rag_module.read_pages("90001_manual_IT.pdf", [19])
    assert res["results"][0]["text"] == page_text
    assert res["results"][0]["page"] == 19


def test_read_pages_maps_printed_page_numbers_from_footers(docsvc_rag_module, monkeypatch):
    _seed_and_patch(docsvc_rag_module, monkeypatch, chunks=_MANUAL_CHUNKS)
    res = docsvc_rag_module.read_pages("90001_manual_IT.pdf", [8], printed=True)
    assert [(r["page"], r["printed_page"]) for r in res["results"]] == [(18, 8)]


def test_read_pages_respects_identity(docsvc_rag_module, monkeypatch):
    _seed_and_patch(docsvc_rag_module, monkeypatch, chunks=_MANUAL_CHUNKS)
    res = docsvc_rag_module.read_pages(
        "90001_manual_IT.pdf", [18], identity=RequestIdentity(permitted_document_ids=["other"])
    )
    assert res["results"] == []


def test_read_pages_caps_the_number_of_pages(docsvc_rag_module, monkeypatch):
    _seed_and_patch(docsvc_rag_module, monkeypatch, chunks=_MANUAL_CHUNKS)
    res = docsvc_rag_module.read_pages("90001_manual_IT.pdf", list(range(20, 30)))
    assert len(res["results"]) == docsvc_rag_module._MAX_PAGES_PER_CALL


# ---------------------------------------------------------------------------
# Machine scoping (plan WP2a)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Quanto pesa la chiuditrice 90001?", ["90001"]),
        ("Sulla 90001 la capsula è troppo avvitata.", ["90001"]),
        ("What is the minimum sterile air pressure for machine 90001?", ["90001"]),
        ("interventi all'intervallo di 20000 ore sulla 90001", ["90001"]),
        ("produzione di 9000 bph", []),
        ("valore 90001,5 misurato", []),
        ("Qual è la coppia di serraggio di una vite M12?", []),
        # The library's other shapes: M/A + 4-7 digits, and 3 digits + a letter a-e.
        ("Qual è la potenza elettrica totale installata sulla 123B?", ["123B"]),
        ("Quanto pesa la macchina ACME BETA con matricola 123B?", ["123B"]),
        ("Qual è il livello di pressione sonora in dB(A) della M0012345?", ["M0012345"]),
        ("Confronta la M01234 con la A1234", ["M01234", "A1234"]),
        ("la 123b o la 123B?", ["123b"]),
        ("macchina 90001 e 123B", ["90001", "123B"]),
        # ...and what must not become a candidate.
        ("alimentazione 400V trifase, 120h di lavoro", []),
        ("codice PN1234 e B1234", []),
        ("vite M12 e M8, 5000 ore", []),
        ("riferimento M05532x", []),
        ("pag.115 del capitolo 11", []),
    ],
)
def test_detect_serials(docsvc_rag_module, text, expected):
    assert docsvc_rag_module.detect_serials(text) == expected


def test_resolve_serial_only_accepts_serials_the_library_knows(docsvc_rag_module, monkeypatch):
    _seed_and_patch(docsvc_rag_module, monkeypatch, chunks=_MANUAL_CHUNKS)
    assert docsvc_rag_module.resolve_serial("la 90001 e la 99999") == "90001"
    assert docsvc_rag_module.resolve_serial("nessuna macchina") is None
    history = [{"role": "user", "content": "parliamo della 90001"}, {"role": "assistant", "content": "ok 12345"}]
    assert docsvc_rag_module.resolve_serial("e il peso?", history) == "90001"


def test_default_serial_fills_only_a_missing_serial(docsvc_rag_module, monkeypatch):
    calls = []
    monkeypatch.setitem(
        docsvc_rag_module._HANDLERS, "search_documents", lambda identity=None, **kw: calls.append(kw) or {"results": []}
    )
    dispatch = docsvc_rag_module.make_dispatch(default_serial="90001")
    dispatch("search_documents", {"query": "peso"})
    dispatch("search_documents", {"query": "peso", "serial": "91400"})
    assert [c["serial"] for c in calls] == ["90001", "91400"]


def _serial_chunks(*serials):
    """One page per serial, tagged the way the ingestion stores it: lowercase."""
    return [
        (f"{s}::v1::p1::c0", f"manuale {s}", _manual_meta(s, 1, serial=s, filename=f"{s.upper()}_manual_it.pdf"))
        for s in serials
    ]


def test_resolve_serial_finds_alphanumeric_serials_stored_in_lowercase(docsvc_rag_module, monkeypatch):
    """Serials like "123B" and "M0012345" were invisible to the 5-digit-only detector, so no
    machine was bound and the prompt told the model to ask which machine it was."""
    _seed_and_patch(docsvc_rag_module, monkeypatch, chunks=_serial_chunks("123b", "m0012345", "90001"))

    assert docsvc_rag_module.resolve_serial("Quanto pesa la macchina con matricola 123B?") == "123B"
    assert docsvc_rag_module.resolve_serial("Qual è il livello sonoro della M0012345?") == "M0012345"
    assert docsvc_rag_module.resolve_serial("e il peso?", [{"role": "user", "content": "parliamo della 123B"}]) == "123B"


def test_resolve_serial_drops_alphanumeric_candidates_the_library_does_not_know(docsvc_rag_module, monkeypatch):
    _seed_and_patch(docsvc_rag_module, monkeypatch, chunks=_serial_chunks("123b"))

    assert docsvc_rag_module.resolve_serial("la M99999 o la 604A?") is None
    assert docsvc_rag_module.resolve_serial("la 123B e la M99999") == "123B"


def test_two_known_serials_in_one_message_are_still_ambiguous(docsvc_rag_module, monkeypatch):
    _seed_and_patch(docsvc_rag_module, monkeypatch, chunks=_serial_chunks("123b", "m0012345"))

    assert docsvc_rag_module.resolve_serial("confronta la 123B con la M0012345") is None


def test_resolve_serial_looks_up_a_bounded_number_of_candidates(docsvc_rag_module, monkeypatch):
    """Each candidate is a Chroma query, so a message full of part codes cannot trigger unbounded lookups."""
    looked_up: list[str] = []
    monkeypatch.setattr(docsvc_rag_module, "_is_known_serial", lambda s: looked_up.append(s) or False)
    message = " ".join(f"M{10000 + n}" for n in range(20))

    assert docsvc_rag_module.resolve_serial(message) is None
    assert len(looked_up) == docsvc_rag_module._MAX_SERIAL_LOOKUPS



# ---------------------------------------------------------------------------
# Loading: lock, no model load without a collection, startup warm-up
# ---------------------------------------------------------------------------


@pytest.fixture()
def counted_model_loads(monkeypatch):
    """Replace ``SentenceTransformer`` with a slow fake that counts its loads."""
    import sentence_transformers

    loads: list[str] = []

    class _CountingModel(_FakeModel):
        def __init__(self, name):
            loads.append(name)
            time.sleep(0.2)  # wide window for a second thread to race in

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", _CountingModel)
    return loads


def test_missing_collection_never_loads_the_model(docsvc_rag_module, counted_model_loads):
    """It used to load the model on every call, since ``_collection`` stayed ``None``."""
    import chromadb

    docsvc_rag_module.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    chromadb.PersistentClient(path=str(docsvc_rag_module.CHROMA_DIR))  # no collection created

    for _ in range(3):
        assert "error" in docsvc_rag_module.search_documents("query")
    assert counted_model_loads == []


def test_concurrent_first_calls_load_the_model_once(docsvc_rag_module, counted_model_loads):
    _seed_collection(docsvc_rag_module.CHROMA_DIR, _CHUNKS)

    with ThreadPoolExecutor(max_workers=4) as pool:
        errors = list(pool.map(lambda _: docsvc_rag_module._ensure_loaded(), range(4)))

    assert errors == [None] * 4
    assert counted_model_loads == [docsvc_rag_module.EMBEDDING_MODEL]


def test_warm_up_loads_the_store_and_reads_its_files(docsvc_rag_module, counted_model_loads, monkeypatch):
    _seed_collection(docsvc_rag_module.CHROMA_DIR, _CHUNKS)
    read: list = []
    monkeypatch.setattr(docsvc_rag_module, "_read_file", lambda path: read.append(path) or path.stat().st_size)

    docsvc_rag_module.warm_up(read_files=True)

    assert docsvc_rag_module._collection is not None
    assert counted_model_loads == [docsvc_rag_module.EMBEDDING_MODEL]
    assert any(p.name == "chroma.sqlite3" for p in read)


def test_warm_up_without_a_store_only_logs(docsvc_rag_module, counted_model_loads, caplog):
    docsvc_rag_module.warm_up(read_files=True)

    assert docsvc_rag_module._collection is None
    assert "not loaded" in caplog.text


def test_read_file_reads_the_whole_file(docsvc_rag_module, tmp_path):
    path = tmp_path / "blob.bin"
    path.write_bytes(b"x" * 1000)

    assert docsvc_rag_module._read_file(path, chunk_size=64) == 1000


def test_concurrent_first_lexical_lookups_build_one_index(docsvc_rag_module, monkeypatch, caplog):
    """Two searches of one round can both reach ``_get_lexical`` first (review on #17)."""
    from sme_chatbot.tools import lexical_index

    built: list[str] = []

    class _SlowMissingIndex:
        def __init__(self, path):
            built.append(path)
            time.sleep(0.1)  # wide window for a second thread to race in

        def exists(self):
            return False

    monkeypatch.setattr(lexical_index, "LexicalIndex", _SlowMissingIndex)
    monkeypatch.setattr(docsvc_rag_module.config, "RETRIEVAL_MODE", "hybrid")
    monkeypatch.setattr(docsvc_rag_module, "_lexical", None)

    with ThreadPoolExecutor(max_workers=4) as pool:
        found = list(pool.map(lambda _: docsvc_rag_module._get_lexical(), range(4)))

    assert found == [None] * 4
    assert len(built) == 1
    assert caplog.text.count("not found; search_documents is dense-only") == 1
