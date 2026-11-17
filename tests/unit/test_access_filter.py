"""Unit tests for ``sme_chatbot.tools.access_filter.build_where`` (WP7).

Covers the full combination matrix described in
``docs/specs/2026-07-02-backend-split-and-docsvc-ingestion-plan.md`` §5 (WP7
acceptance) and §4.2 (``RequestIdentity``):

- unrestricted identity, no filters -> ``None``
- a single clause is returned bare (no ``$and``)
- multiple clauses combine under ``{"$and": [...]}``
- ``permitted_document_ids`` -> ``{"document_id": {"$in": [...]}}``
- an empty permitted list (or an empty caller/identity intersection) ->
  the "matches nothing" sentinel
- ``identity.scopes`` -> equality clauses
- caller-supplied ``document_ids`` are intersected with ``permitted_document_ids``
  (narrowed, never widened)

Pure-function tests: no ChromaDB, no network, no model download.
"""

from __future__ import annotations

from sme_chatbot.core.events import RequestIdentity
from sme_chatbot.tools.access_filter import build_where

_SENTINEL = "__none__"


# ---------------------------------------------------------------------------
# Unrestricted / no-op cases
# ---------------------------------------------------------------------------


def test_unrestricted_identity_no_filters_returns_none():
    identity = RequestIdentity()
    assert build_where(identity) is None


def test_unrestricted_identity_empty_scopes_still_none():
    identity = RequestIdentity(permitted_document_ids=None, scopes={})
    assert build_where(identity, category=None, serial=None, customer=None) is None


# ---------------------------------------------------------------------------
# Single clause -> bare dict (no $and)
# ---------------------------------------------------------------------------


def test_single_explicit_filter_is_bare_clause():
    identity = RequestIdentity()
    where = build_where(identity, category="manual")
    assert where == {"category": "manual"}


def test_single_serial_filter_is_bare_clause():
    identity = RequestIdentity()
    where = build_where(identity, serial="91123")
    assert where == {"serial": "91123"}


def test_single_customer_filter_is_bare_clause():
    identity = RequestIdentity()
    where = build_where(identity, customer="acme")
    assert where == {"customer": "acme"}


def test_single_scope_is_bare_clause():
    identity = RequestIdentity(scopes={"serial": "91123"})
    where = build_where(identity)
    assert where == {"serial": "91123"}


def test_permitted_document_ids_alone_is_bare_clause():
    identity = RequestIdentity(permitted_document_ids=["a", "b"])
    where = build_where(identity)
    assert where == {"document_id": {"$in": ["a", "b"]}}


# ---------------------------------------------------------------------------
# Multiple clauses -> $and
# ---------------------------------------------------------------------------


def test_multiple_explicit_filters_combine_under_and():
    identity = RequestIdentity()
    where = build_where(identity, category="manual", serial="91123")
    assert where == {"$and": [{"category": "manual"}, {"serial": "91123"}]}


def test_three_explicit_filters_combine_under_and_in_order():
    identity = RequestIdentity()
    where = build_where(identity, category="manual", serial="91123", customer="acme")
    assert where == {
        "$and": [
            {"category": "manual"},
            {"serial": "91123"},
            {"customer": "acme"},
        ]
    }


def test_scopes_plus_explicit_filter_combine_under_and():
    identity = RequestIdentity(scopes={"serial": "91123"})
    where = build_where(identity, category="manual")
    assert where == {"$and": [{"serial": "91123"}, {"category": "manual"}]}


def test_permitted_document_ids_plus_category_combine_under_and():
    identity = RequestIdentity(permitted_document_ids=["a", "b"])
    where = build_where(identity, category="manual")
    assert where == {
        "$and": [
            {"document_id": {"$in": ["a", "b"]}},
            {"category": "manual"},
        ]
    }


def test_everything_combined_in_documented_order():
    identity = RequestIdentity(
        permitted_document_ids=["a", "b", "c"],
        scopes={"customer": "acme"},
    )
    where = build_where(identity, category="manual", serial="91123", document_ids=["a", "b"])
    assert where == {
        "$and": [
            {"document_id": {"$in": ["a", "b"]}},
            {"customer": "acme"},
            {"category": "manual"},
            {"serial": "91123"},
        ]
    }


# ---------------------------------------------------------------------------
# permitted_document_ids -> $in, including the empty-set sentinel
# ---------------------------------------------------------------------------


def test_empty_permitted_document_ids_matches_nothing_sentinel():
    identity = RequestIdentity(permitted_document_ids=[])
    where = build_where(identity)
    assert where == {"document_id": {"$in": [_SENTINEL]}}
    # sanity: sentinel can never equal a real DocumentService UUID
    assert _SENTINEL not in ["a", "b"]


def test_empty_permitted_document_ids_with_other_filters_still_and():
    identity = RequestIdentity(permitted_document_ids=[])
    where = build_where(identity, category="manual")
    assert where == {
        "$and": [
            {"document_id": {"$in": [_SENTINEL]}},
            {"category": "manual"},
        ]
    }


# ---------------------------------------------------------------------------
# scopes -> equality clauses (one per key)
# ---------------------------------------------------------------------------


def test_multiple_scopes_each_become_their_own_clause():
    identity = RequestIdentity(scopes={"serial": "91123", "customer": "acme"})
    where = build_where(identity)
    assert where == {"$and": [{"serial": "91123"}, {"customer": "acme"}]}


# ---------------------------------------------------------------------------
# Caller document_ids: unrestricted identity, no identity restriction to apply
# ---------------------------------------------------------------------------


def test_caller_document_ids_without_identity_restriction_pass_through():
    identity = RequestIdentity()  # permitted_document_ids=None -> unrestricted
    where = build_where(identity, document_ids=["a", "b"])
    assert where == {"document_id": {"$in": ["a", "b"]}}


def test_caller_empty_document_ids_without_identity_restriction_is_sentinel():
    identity = RequestIdentity()
    where = build_where(identity, document_ids=[])
    assert where == {"document_id": {"$in": [_SENTINEL]}}


# ---------------------------------------------------------------------------
# Caller document_ids intersected with permitted (narrow, never widen)
# ---------------------------------------------------------------------------


def test_caller_document_ids_intersected_with_permitted():
    identity = RequestIdentity(permitted_document_ids=["a", "b", "c"])
    where = build_where(identity, document_ids=["b", "c", "z"])
    assert where == {"document_id": {"$in": ["b", "c"]}}


def test_caller_document_ids_cannot_widen_beyond_permitted():
    """A caller asking for a document_id outside their permitted set must
    never see it added to the effective filter."""
    identity = RequestIdentity(permitted_document_ids=["a"])
    where = build_where(identity, document_ids=["a", "not-permitted"])
    assert where == {"document_id": {"$in": ["a"]}}
    assert "not-permitted" not in where["document_id"]["$in"]


def test_intersection_empty_matches_nothing_sentinel():
    identity = RequestIdentity(permitted_document_ids=["a", "b"])
    where = build_where(identity, document_ids=["x", "y"])
    assert where == {"document_id": {"$in": [_SENTINEL]}}


def test_intersection_preserves_caller_order_within_permitted():
    identity = RequestIdentity(permitted_document_ids=["z", "a", "b"])
    where = build_where(identity, document_ids=["a", "b", "c"])
    assert where == {"document_id": {"$in": ["a", "b"]}}


# ---------------------------------------------------------------------------
# Case normalization: Chroma `where` equality is case-sensitive,
# but DocumentService stores tag values (and category) lowercase while users
# type whatever case appears in filenames, e.g. "M01691".
# ---------------------------------------------------------------------------


def test_explicit_serial_is_lowercased_to_match_stored_value():
    identity = RequestIdentity()
    where = build_where(identity, serial="M01691")
    assert where == {"serial": "m01691"}


def test_explicit_category_is_lowercased_to_match_stored_value():
    identity = RequestIdentity()
    where = build_where(identity, category="Manual")
    assert where == {"category": "manual"}


def test_explicit_customer_is_lowercased_to_match_stored_value():
    identity = RequestIdentity()
    where = build_where(identity, customer="ACME")
    assert where == {"customer": "acme"}


def test_scope_serial_is_lowercased():
    identity = RequestIdentity(scopes={"serial": "M01691"})
    where = build_where(identity)
    assert where == {"serial": "m01691"}


def test_scope_customer_and_category_are_lowercased():
    identity = RequestIdentity(scopes={"customer": "ACME", "category": "Manual"})
    where = build_where(identity)
    assert where == {"$and": [{"customer": "acme"}, {"category": "manual"}]}


def test_scope_source_is_not_lowercased():
    """`source` values are genuinely uppercase in stored data
    (e.g. "ACME-NETWORK-SHARE") — folding them would break matches."""
    identity = RequestIdentity(scopes={"source": "ACME-NETWORK-SHARE"})
    where = build_where(identity)
    assert where == {"source": "ACME-NETWORK-SHARE"}


def test_document_id_in_list_is_not_lowercased():
    """document_id values are DocumentService UUIDs and must pass through
    untouched, including any uppercase hex digits."""
    identity = RequestIdentity(permitted_document_ids=["ABC-123-DEF"])
    where = build_where(identity)
    assert where == {"document_id": {"$in": ["ABC-123-DEF"]}}


def test_no_match_sentinel_is_not_lowercased():
    identity = RequestIdentity(permitted_document_ids=[])
    where = build_where(identity)
    assert where == {"document_id": {"$in": [_SENTINEL]}}
    assert _SENTINEL == "__none__"


# ---------------------------------------------------------------------------
# language filter: stored verbatim from the DTO's
# `metadata`, and both cases exist (live example `"TR"`; the 2026-10-15 archive
# stores `"it"`), so the clause matches either.
# ---------------------------------------------------------------------------


def test_single_language_filter_is_bare_clause_matching_either_case():
    identity = RequestIdentity()
    where = build_where(identity, language="it")
    assert where == {"language": {"$in": ["IT", "it"]}}


def test_language_filter_is_the_same_whatever_case_the_caller_uses():
    identity = RequestIdentity()
    assert build_where(identity, language="DE") == build_where(identity, language="de") == {"language": {"$in": ["DE", "de"]}}


def test_language_combines_with_other_filters_under_and():
    identity = RequestIdentity()
    where = build_where(identity, serial="90001", language="it")
    assert where == {"$and": [{"serial": "90001"}, {"language": {"$in": ["IT", "it"]}}]}
