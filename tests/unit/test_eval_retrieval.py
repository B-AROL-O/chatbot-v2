"""Unit tests for the offline retrieval benchmark (``eval/retrieval.py``, plan WP0).

The retrievers are stubbed: these tests pin the hit rule (gold document +
printed-page mapping) and the recall arithmetic, not retrieval quality.
"""

from __future__ import annotations

from sme_chatbot.eval import retrieval
from sme_chatbot.eval.dataset import EvalItem
from sme_chatbot.tools import docsvc_rag


def _item(**kw) -> EvalItem:
    base = dict(
        id="Q01",
        categoria="Dato tecnico semplice",
        difficolta=1,
        domanda="Qual è la produzione nominale della chiuditrice 90001?",
        risposta_attesa="9000 pezzi/ora",
        comportamento_atteso="",
        capitolo_fonte="4.3",
        pag_stampata="8",
        documento="90001_manual_IT.pdf",
        serial="90001",
        gold_pages=frozenset({8}),
    )
    base.update(kw)
    return EvalItem(**base)


def _hit(page, filename="90001_manual_IT.pdf"):
    return (f"id-{filename}-{page}", "text", {"filename": filename, "page": page}, None)


def test_rank_item_applies_offset_tolerance_and_gold_document(monkeypatch):
    dense = [_hit(18, "90001_manual_DE.pdf"), _hit(40), _hit(19)]  # DE page 18 is not the gold doc
    lexical = [_hit(18)]
    monkeypatch.setattr(docsvc_rag, "_dense_hits", lambda q, where, k: dense)
    monkeypatch.setattr(docsvc_rag, "_lexical_hits", lambda q, where, k: lexical)

    rank = retrieval.rank_item(_item(), 20, offset=10, tolerance=2)

    assert rank.expected_pdf_pages == [18]
    assert rank.ranks["dense"] == 3  # page 19 is within ±2 of 18
    assert rank.ranks["lexical"] == 1
    assert rank.ranks["hybrid"] is not None


def test_rank_item_passes_the_items_serial_as_filter(monkeypatch):
    seen = []
    monkeypatch.setattr(docsvc_rag, "_dense_hits", lambda q, where, k: seen.append(where) or [])
    monkeypatch.setattr(docsvc_rag, "_lexical_hits", lambda q, where, k: [])
    rank = retrieval.rank_item(_item(), 5, offset=10, tolerance=2)
    assert seen == [{"serial": "90001"}]
    assert rank.ranks == {"dense": None, "lexical": None, "hybrid": None}


def test_recall_at_counts_ranks_within_k():
    ranks = [
        retrieval.ItemRank("Q1", "", [18], {"dense": 1, "lexical": None, "hybrid": 3}),
        retrieval.ItemRank("Q2", "", [19], {"dense": 7, "lexical": 2, "hybrid": 1}),
    ]
    assert retrieval.recall_at(ranks, "dense", 5) == 0.5
    assert retrieval.recall_at(ranks, "hybrid", 5) == 1.0
    assert retrieval.recall_at(ranks, "lexical", 1) == 0.0
    assert "| @5 |" in retrieval.render(ranks, (1, 5))
