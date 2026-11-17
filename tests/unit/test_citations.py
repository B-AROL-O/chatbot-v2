"""Citation plumbing tests, end to end of the label path.

The footer used to read ``— fonte: ACME-NETWORK-SHARE p.501`` — the origin
network share instead of the document. Two halves are covered here:

* ``core.chat._extract_citations`` — the backend mapping from a RAG tool's
  JSON result to ``Citation``; ``source`` must always end up as the document
  name, whichever of the three tools produced the result.
* ``gradio_app._format_citation_footer`` — the frontend rendering, which
  groups by document instead of dumping one line per retrieved chunk, and
  prunes to what the answer actually cites.

No network, no LLM, no vector store.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "frontends"))

from gradio_app import _format_citation_footer, _strip_citation_footer

from sme_chatbot.core.chat import _extract_citations

# ---------------------------------------------------------------------------
# Backend: tool result -> Citation
# ---------------------------------------------------------------------------


def test_docsvc_result_labels_the_citation_with_the_filename():
    """The document name, not the network share."""
    result = json.dumps(
        {
            "results": [
                {
                    "text": "…",
                    "document_id": "62dc1150-aaaa",
                    "version_id": "v1",
                    "filename": "20506_spare_parts_catalogue_IT.pdf",
                    "repository": "ACME-NETWORK-SHARE",
                    "category": "manual",
                    "page": 501,
                    "distance": 0.31,
                }
            ]
        }
    )

    (citation,) = _extract_citations(result)

    assert citation.source == "20506_spare_parts_catalogue_IT.pdf"
    assert citation.filename == "20506_spare_parts_catalogue_IT.pdf"
    assert citation.repository == "ACME-NETWORK-SHARE"
    assert citation.document_id == "62dc1150-aaaa"
    assert citation.page == 501
    assert citation.distance == 0.31


def test_legacy_source_only_result_is_unchanged():
    """``search_docs``/``search_manuals`` already put the filename in ``source``."""
    result = json.dumps(
        {"results": [{"text": "…", "source": "91123_manual_IT.pdf", "page": 9, "distance": 0.1}]}
    )

    (citation,) = _extract_citations(result)

    assert citation.source == "91123_manual_IT.pdf"
    assert citation.filename is None
    assert citation.repository is None
    assert citation.document_id is None


def test_filename_wins_over_a_stale_source_key():
    """Belt and braces: if both keys arrive, the document name is authoritative."""
    result = json.dumps(
        {"results": [{"filename": "manuale_XY_IT.pdf", "source": "XY-NETWORK-SHARE", "page": 24}]}
    )

    (citation,) = _extract_citations(result)

    assert citation.source == "manuale_XY_IT.pdf"


def test_non_result_payloads_yield_no_citations():
    assert _extract_citations(json.dumps({"error": "boom"})) == []
    assert _extract_citations(json.dumps({"results": "nope"})) == []
    assert _extract_citations("not json at all") == []
    assert _extract_citations(json.dumps({"results": [{"text": "no locator"}]})) == []


# ---------------------------------------------------------------------------
# Frontend: Citation list -> footer text
# ---------------------------------------------------------------------------


def _citation(filename, page, distance=None, document_id=None, repository=None):
    return {
        "source": filename,
        "filename": filename,
        "page": page,
        "distance": distance,
        "document_id": document_id,
        "repository": repository,
    }


def test_footer_groups_pages_under_one_line_per_document():
    citations = [
        _citation("20506_manual_IT.pdf", 9, distance=0.10, document_id="doc-a"),
        _citation("20506_spare_parts_catalogue_IT.pdf", 1, distance=0.20, document_id="doc-b"),
        _citation("20506_manual_IT.pdf", 79, distance=0.30, document_id="doc-a"),
        _citation("20506_manual_IT.pdf", 22, distance=0.15, document_id="doc-a"),
    ]

    assert _format_citation_footer(citations) == (
        "\n\n— fonte: 20506_manual_IT.pdf, pp. 9, 22, 79"
        "\n— fonte: 20506_spare_parts_catalogue_IT.pdf, p. 1"
    )


def test_footer_orders_documents_by_best_distance():
    citations = [
        _citation("far.pdf", 1, distance=0.90, document_id="doc-far"),
        _citation("near.pdf", 2, distance=0.05, document_id="doc-near"),
    ]

    assert _format_citation_footer(citations).splitlines()[2].startswith("— fonte: near.pdf")


def test_same_page_of_two_documents_is_not_collapsed():
    """The old ``(source, page)`` dedup key silently dropped one of these.

    ``source`` was the network share and therefore identical for both, so
    ``(source, 9)`` matched twice and one document vanished.
    """
    citations = [
        _citation("ACME-first.pdf", 9, distance=0.1, document_id="doc-a"),
        _citation("ACME-second.pdf", 9, distance=0.2, document_id="doc-b"),
    ]

    footer = _format_citation_footer(citations)

    assert "ACME-first.pdf, p. 9" in footer
    assert "ACME-second.pdf, p. 9" in footer


def test_duplicate_chunks_of_the_same_page_render_once():
    citations = [_citation("m.pdf", 4, distance=0.1, document_id="d")] * 3
    assert _format_citation_footer(citations) == "\n\n— fonte: m.pdf, p. 4"


def test_documents_without_ids_are_grouped_by_filename():
    """Legacy collections supply no ``document_id``; the label is the key."""
    citations = [_citation("m.pdf", 1), _citation("m.pdf", 2)]
    assert _format_citation_footer(citations) == "\n\n— fonte: m.pdf, pp. 1, 2"


def test_duplicate_copies_of_one_file_render_as_one_line():
    """Seen live: `M05586_manual_fr.pdf` is in the store under four document
    ids, two of them full copies, so a retrieval can return the same file and
    page twice under different ids. Grouping by id printed the identical line
    twice; the reader must see one."""
    citations = [
        _citation("M05586_manual_fr.pdf", 69, distance=4.66, document_id="8993cb6d"),
        _citation("M05586_manual_fr.pdf", 69, distance=4.71, document_id="b7869a6f"),
        _citation("M05586_manual_fr.pdf", 12, distance=5.02, document_id="b7869a6f"),
    ]

    assert _format_citation_footer(citations) == "\n\n— fonte: M05586_manual_fr.pdf, pp. 12, 69"


def test_citation_without_a_page_renders_the_document_alone():
    assert _format_citation_footer([_citation("m.pdf", None)]) == "\n\n— fonte: m.pdf"


def test_unlabelled_citations_are_skipped():
    assert _format_citation_footer([{"source": None, "page": 3}]) == ""
    assert _format_citation_footer([]) == ""


def test_overflowing_documents_and_pages_are_reported_not_dropped():
    citations = [
        _citation(f"doc{i}.pdf", 1, distance=i / 100, document_id=f"d{i}") for i in range(10)
    ]
    citations += [_citation("doc0.pdf", p, distance=0.0, document_id="d0") for p in range(2, 20)]

    footer = _format_citation_footer(citations)
    lines = footer.strip().splitlines()

    assert len(lines) == 9  # 8 documents + the overflow note
    assert lines[0] == "— fonte: doc0.pdf, pp. 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, … (+9)"
    assert lines[-1] == "— … e altri 2 documenti"


def test_footer_is_still_strippable_from_history():
    """The history round-trip must not send the footer back to the model."""
    answer = "La procedura è descritta nel manuale."
    footer = _format_citation_footer(
        [
            _citation("m.pdf", 9, distance=0.1, document_id="d"),
            _citation("n.pdf", 501, distance=0.2, document_id="e"),
        ]
    )

    assert _strip_citation_footer(answer + footer) == answer


# ---------------------------------------------------------------------------
# Frontend: pruning the retrieval dump against the answer text
# ---------------------------------------------------------------------------


def test_footer_lists_only_the_documents_the_answer_cites():
    """The footer was longest exactly when the answer was emptiest."""
    citations = [
        _citation("20506_manual_IT.pdf", 9, distance=0.10, document_id="doc-a"),
        _citation("noise_a.pdf", 3, distance=0.40, document_id="doc-b"),
        _citation("noise_b.pdf", 7, distance=0.50, document_id="doc-c"),
    ]
    answer = "La procedura è a pagina 9 di 20506_manual_IT.pdf."

    assert _format_citation_footer(citations, answer) == (
        "\n\n— fonte: 20506_manual_IT.pdf, p. 9"
        "\n— … e altre 2 fonti consultate, non citate nella risposta"
    )


def test_all_documents_are_listed_when_the_answer_cites_none():
    """No signal in the answer means no pruning — better verbose than lossy."""
    citations = [
        _citation("a.pdf", 1, distance=0.1, document_id="doc-a"),
        _citation("b.pdf", 2, distance=0.2, document_id="doc-b"),
    ]

    assert _format_citation_footer(citations, "Non ho trovato nulla.") == (
        "\n\n— fonte: a.pdf, p. 1\n— fonte: b.pdf, p. 2"
    )


def test_pruning_is_off_by_default_for_callers_that_pass_no_answer():
    citations = [_citation("a.pdf", 1, distance=0.1, document_id="doc-a")]
    assert _format_citation_footer(citations) == "\n\n— fonte: a.pdf, p. 1"
