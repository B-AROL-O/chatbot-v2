"""Identity/scope -> Chroma ``where``-filter builder for the ``documents`` domain.

Pure function, no ChromaDB import needed — it only assembles the dict shape
Chroma's ``where`` argument expects (equality clauses, ``$in``, ``$and``).

See ``docs/specs/2026-07-02-backend-split-and-docsvc-ingestion-plan.md`` §4.2
(``RequestIdentity``) and §4.3 (chunk metadata schema — the business keys this
module builds equality clauses against).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Deferred: `sme_chatbot.core` eagerly imports `core.domains`, which in
    # turn imports this package's tools (including this module) to build the
    # domain registry. A module-level import here would re-enter that partly
    # -initialized chain and raise ImportError. `RequestIdentity` is only
    # used for type hints, so this is safe under `from __future__ import
    # annotations` — no runtime import needed.
    from sme_chatbot.core.events import RequestIdentity

# Sentinel document_id that can never match a real chunk. Used to render an
# "access to nothing" restriction (an empty ``permitted_document_ids`` list,
# or an empty intersection between it and caller-supplied ``document_ids``)
# as a where-filter that matches zero rows, since Chroma's ``$in`` requires a
# non-empty list.
_NO_MATCH_SENTINEL = "__none__"

# Chroma `where` equality is case-sensitive, but DocumentService stores tag
# values (and thus these business keys) lowercase while users/the LLM type
# whatever case appears in filenames (e.g. "M01691"). These are the only keys
# whose *values* get folded to lowercase before building a clause — `source`
# is genuinely uppercase in stored data, and `document_id` is a UUID that must
# pass through untouched, so neither belongs in this set.
_LOWERCASE_VALUE_KEYS = {"serial", "customer", "category"}


def build_where(
    identity: RequestIdentity,
    category: str | None = None,
    serial: str | None = None,
    customer: str | None = None,
    language: str | None = None,
    document_ids: list[str] | None = None,
) -> dict | None:
    """Combine caller-supplied filters with identity restrictions into a
    Chroma ``where`` filter.

    - ``identity.permitted_document_ids`` (``None`` = unrestricted) becomes a
      ``{"document_id": {"$in": [...]}}`` clause. If the caller also passed
      ``document_ids``, the two lists are intersected (the caller can only
      narrow, never widen, what identity permits).
    - An empty permitted set (``[]``) — or an intersection that comes out
      empty — means "access to nothing": rendered as
      ``{"document_id": {"$in": ["__none__"]}}``, a filter that matches no
      real chunk (real ``document_id`` values are DocumentService UUIDs).
    - ``identity.scopes`` becomes one equality clause per key
      (e.g. ``{"serial": "91123"}``).
    - Explicit ``category``/``serial``/``customer``/``language`` kwargs each
      become their own equality clause.
    - ``serial``/``customer``/``category`` values are lowercased before
      building their clause (both from kwargs and from ``identity.scopes``),
      since Chroma's ``where`` equality is case-sensitive but stored values
      are always lowercase. Other keys (e.g. ``source``, which is genuinely
      uppercase) and ``document_id`` are left untouched.
    - ``language`` matches either case: ``{"language": {"$in": ["IT", "it"]}}``.
      It is flattened from the DocumentService document's ``metadata`` object
      (§4.3 of the ingestion plan) and stored verbatim, and both conventions
      exist: the live example this was found with is ``"TR"``,
      and the archived index of 2026-10-15 stores ``"it"``/``"de"``. An
      uppercase-only clause matched nothing on that index, so every search
      with a language filter came back empty.
    - Multiple clauses combine under ``{"$and": [...]}``; Chroma rejects
      ``$and`` with fewer than two members, so a single clause is returned
      bare.
    - Returns ``None`` when nothing restricts the query (fully unrestricted
      identity, no scopes, no explicit filters).
    """
    clauses: list[dict] = []

    permitted = identity.permitted_document_ids
    if permitted is not None:
        if document_ids is not None:
            allowed_ids = [doc_id for doc_id in document_ids if doc_id in permitted]
        else:
            allowed_ids = list(permitted)
        clauses.append({"document_id": {"$in": allowed_ids or [_NO_MATCH_SENTINEL]}})
    elif document_ids is not None:
        clauses.append({"document_id": {"$in": list(document_ids) or [_NO_MATCH_SENTINEL]}})

    for key, value in identity.scopes.items():
        if key in _LOWERCASE_VALUE_KEYS:
            value = value.lower()
        clauses.append({key: value})

    if category is not None:
        clauses.append({"category": category.lower()})
    if serial is not None:
        clauses.append({"serial": serial.lower()})
    if customer is not None:
        clauses.append({"customer": customer.lower()})
    if language is not None:
        clauses.append({"language": {"$in": sorted({language.upper(), language.lower()})}})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}
