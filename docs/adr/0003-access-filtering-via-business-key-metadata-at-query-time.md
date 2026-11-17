# 0003. Document access is filtered at query time via business-key chunk metadata

- **Status:** Accepted (implemented; JWT claim→scope mapping still stubbed)
- **Date:** 2026-07-02 — backfilled 2026-07-07 from
  [`docs/specs/2026-07-02-backend-split-and-docsvc-ingestion-plan.md`](../specs/2026-07-02-backend-split-and-docsvc-ingestion-plan.md)
  §1 ("Decisions already made") and §4.2–4.3

## Context

The `documents` domain (DocumentService corpus in ChromaDB collection `docsvc`)
must not return chunks a caller is not entitled to see. Two options: bake caller
entitlements into chunks at ingestion time (`user_id`/`tenant_id` tags), or tag chunks
with **business keys** and resolve entitlements per request.

## Decision

Chunks are stamped at ingestion with business-key metadata only: `document_id`,
`version_id`, `source`, `category`, plus flattened scalar keys from the document's
`metadata` object and `tags` array (`serial`, `customer`, …) — see
[the spec's §4.3](../specs/2026-07-02-backend-split-and-docsvc-ingestion-plan.md#43-docsvc-chunk-metadata-schema-chromadb-collection-docsvc)
for exactly which fields come from which source; `serial` is a `tags` key, not a
`metadata` key. **No user or tenant identifiers are written at
ingestion time.** At query time, the request's `RequestIdentity`
(`permitted_document_ids` + `scopes`) is translated by
`tools/access_filter.py::build_where()` into a ChromaDB `where` filter, so
out-of-scope chunks are **eliminated before retrieval**, not ranked lower. Callers can
narrow but never widen what identity permits; an empty permitted set matches nothing.

## Consequences

- Entitlement changes (user moves group, permission revoked) need no re-ingestion —
  only the identity resolution changes.
- The filter is exactly as trustworthy as the identity resolution feeding it; in
  `AUTH_MODE=jwt` the claim→scope mapping is currently a stub returning an
  unrestricted identity, so real enforcement waits on that mapping (the planned
  machine-docs work must define how business groups map into `scopes`).
- ChromaDB metadata values are scalars only, so scoping dimensions must be expressible
  as flat key/value pairs.
