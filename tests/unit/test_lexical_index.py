"""Unit tests for ``sme_chatbot.tools.lexical_index`` (plan WP1).

Pure SQLite FTS5 under ``tmp_path`` — no Chroma, no embedding model.
"""

from __future__ import annotations

import pytest

from sme_chatbot.tools.lexical_index import LexicalIndex, _translate, normalise, query_terms


def _meta(document_id, serial="90001", language="IT", filename="90001_manual_IT.pdf"):
    return {"document_id": document_id, "serial": serial, "language": language, "filename": filename}


@pytest.fixture()
def index(tmp_path):
    idx = LexicalIndex(tmp_path / "lexical.sqlite3")
    idx.add(
        ["d1::p18::c0", "d1::p19::c1", "d1::p41::c0", "d2::p18::c0"],
        [
            "**==> picture [527 x 434] intentionally omitted <==**<br>PRODUZIONE NOMINALE<br>Pezzi/ora 9000 bph",
            "|M 12|1,75|18|10|92|<br>|M 16|2|24|14|315|",
            "Verifiche e controlli prima della messa in funzione della chiuditrice.",
            "PRODUZIONE NOMINALE<br>Pezzi/ora 6000 bph",
        ],
        [_meta("d1"), _meta("d1"), _meta("d1"), _meta("d2", serial="91400")],
    )
    yield idx
    idx.close()


def test_normalise_strips_pymupdf_markup_and_unifies_thread_sizes():
    text = normalise("**==> picture [1 x 2] intentionally omitted <==**<br>|M 12|92|")
    assert "picture" not in text
    assert "<br>" not in text and "|" not in text
    assert "M12" in text


def test_query_terms_drop_stopwords_and_keep_numbers():
    assert query_terms("Qual è la coppia di serraggio di una vite M 12?") == ["coppia", "serraggio", "vite", "m12"]
    assert "8" in query_terms("classe 8")


def test_search_ranks_the_chunk_holding_the_terms_first(index):
    assert index.search("produzione nominale", where={"serial": "90001"})[0] == "d1::p18::c0"


def test_thread_size_matches_with_or_without_space(index):
    assert index.search("coppia vite M12")[0] == "d1::p19::c1"


def test_where_prefilter_narrows_to_the_serial(index):
    hits = index.search("produzione nominale", where={"serial": "91400"})
    assert hits == ["d2::p18::c0"]


def test_fts_operators_in_user_text_are_inert(index):
    # Would be a syntax error (or a different query) if passed to MATCH raw.
    assert index.search('produzione AND "nominale" OR NOT (') != []


def test_translate_drops_what_it_cannot_express_so_it_only_widens():
    clauses, params = _translate(
        {"$and": [{"serial": "90001"}, {"version_id": {"$ne": "v1"}}, {"document_id": {"$in": ["a", "b"]}}]}
    )
    assert clauses == ["m.serial = ?", "m.document_id IN (?, ?)"]
    assert params == ["90001", "a", "b"]


def test_delete_document_keeps_only_the_current_version(index):
    index.add(["d1::v2::p1::c0"], ["nuova versione produzione"], [_meta("d1")])
    index.delete_document("d1", keep_ids=["d1::v2::p1::c0"])
    assert index.search("nominale", where={"document_id": "d1"}) == []
    assert index.search("nuova versione") == ["d1::v2::p1::c0"]


def test_add_replaces_an_existing_chunk_id(index):
    index.add(["d1::p41::c0"], ["testo sostituito"], [_meta("d1")])
    assert index.count() == 4
    assert index.search("messa in funzione") == []


def test_light_stemming_matches_inflections(index):
    index.add(["d1::p18::c1"], ["PESO DELLA CHIUDITRICE Peso 850 kg."], [_meta("d1")])
    assert index.search("Quanto pesa la chiuditrice?")[0] == "d1::p18::c1"


def test_a_serial_pinned_by_where_is_not_a_search_term(tmp_path):
    idx = LexicalIndex(tmp_path / "l.sqlite3")
    # Every chunk of a machine's documents tends to mention its serial; as a
    # term it would rank the header-only chunk above the one with the answer.
    idx.add(
        ["a", "b"],
        ["90001 90001 90001 intestazione", "produzione nominale 9000 bph"],
        [_meta("d1"), _meta("d1")],
    )
    assert idx.search("produzione nominale della 90001", where={"serial": "90001"})[0] == "b"
    idx.close()
