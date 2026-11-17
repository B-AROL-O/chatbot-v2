# DocumentService Ingestion (`sme-ingest-docsvc`)

How to run the `documents`-domain ingestion pipeline
(`src/sme_chatbot/ingestion/ingest_docsvc.py`, CLI entry point
`sme-ingest-docsvc`) — both its normal ingest mode and the `--backfill-tags`
mode added in WP3 to retrofit tag metadata onto chunks ingested before it
existed.

See [`specs/2026-07-02-backend-split-and-docsvc-ingestion-plan.md`
§4.3](../specs/2026-07-02-backend-split-and-docsvc-ingestion-plan.md#43-docsvc-chunk-metadata-schema-chromadb-collection-docsvc)
for the frozen chunk-metadata schema this pipeline implements.

## Prerequisites

DocumentService credentials configured in `.env`. Missing credentials fail
fast with guidance printed to stderr before any listing or download happens.

### Obtaining the credentials

DocumentService uses JWT Bearer auth with Entra ID **app roles** (`roles`
claim), obtained via the OAuth2 client-credentials flow. Ingestion only reads,
so it needs an app holding `Documents.Read`:

1. An Entra **app registration** for the ingestion.
2. The DocumentService app role `Documents.Read` assigned to that app's
   service principal, with **admin consent** granted.
3. A client secret for it.

| Var | Default | Meaning |
| --- | --- | --- |
| `DOCSVC_BASE_URL` | none | service base URL |
| `DOCSVC_VERIFY_SSL` | `true` | verify the service's TLS certificate |
| `DOCSVC_TENANT_ID` | _(unset)_ | Entra tenant id |
| `DOCSVC_AUTHORITY` | `https://login.microsoftonline.com/{tenant}` | MSAL authority (derived from tenant) |
| `DOCSVC_AUDIENCE` | _(unset)_ | resource / app-ID-URI, e.g. `api://<svc-app-id>` |
| `DOCSVC_SCOPE` | `<audience>/.default` | client-credentials scope override |
| `DOCSVC_READ_CLIENT_ID` / `DOCSVC_READ_CLIENT_SECRET` | _(unset)_ | the `Documents.Read` app |
| `DOCSVC_USER_ID` | _(unset)_ | optional `X-User-Id` audit header (GUID) |

Secrets go in `.env` only — never commit them.

---

## Normal ingestion

```bash
uv run sme-ingest-docsvc --limit 5 --dry-run   # preview: list + parse decisions, no writes
uv run sme-ingest-docsvc --limit 5             # populate data/chroma_docsvc_db
uv run sme-ingest-docsvc                       # full run, no limit
```

For each document: skip if `deleted`, skip unsupported extensions, skip if
already ingested (probed via the first chunk's deterministic ID), else
download → parse → chunk → embed → write, then delete any chunks left over
from a prior version of the same document (`document_id` match,
`version_id` mismatch). Re-running is idempotent — already-ingested documents
add 0 chunks.

Useful filters: `--category`, `--source`, `--tag key:value`, `--page-size`
(listing page size, default 50, capped at 100 server-side).

---

## `--backfill-tags` mode

Documents ingested before tag flattening existed (pre-WP1) have chunks
without `serial`/`doc_type`/etc. metadata. `--backfill-tags` patches those
keys onto **already-ingested** chunks without re-ingesting:

```bash
uv run sme-ingest-docsvc --backfill-tags --dry-run   # preview what would be patched
uv run sme-ingest-docsvc --backfill-tags --limit 20  # patch a batch
uv run sme-ingest-docsvc --backfill-tags             # patch everything
```

What it does and doesn't do, per document DTO:

- **Metadata-only.** Calls `collection.update()` (merges metadata onto
  existing chunk IDs) — never `collection.add()`. No download, no re-parsing,
  no re-embedding.
- **Skips documents not yet ingested.** If a document's `document_id` has no
  chunks in `docsvc` at all, backfill has nothing to patch (run normal
  ingestion for it instead).
- **Skips documents already fully tagged.** If every `tags`-derived key for
  a document already exists in its stored chunk metadata, there's nothing new
  to add.
- **Refuses to patch across a version mismatch.** If any stored chunk's
  `version_id` differs from the DTO's current `version_id` (e.g. an earlier
  run was interrupted mid stale-version-delete), the document is reported as
  `version_mismatch` and left alone — it needs normal re-ingestion, not a
  tag patch, so a backfill can't silently stamp new tags onto stale chunks.
- **Idempotent and resumable.** Safe to re-run, safe to `Ctrl-C` and resume
  later with the same filters — each document is patched independently and a
  document with nothing new to add is simply skipped on the next pass.
- **Same collision rule as normal ingestion** (§4.3): a `tags` key already
  present as a reserved key or as `metadata`-flattened data on that chunk is
  never overwritten.

Combine with `--category`/`--source`/`--tag`/`--limit` exactly as in normal
ingestion mode to scope a backfill run — for example, to backfill only
documents in one category first and check the run summary before continuing.

The run summary distinguishes `patched` / `skipped` / `version_mismatch` /
`failed` counts (normal-mode runs report `added` / `skipped` / `failed`
instead — there's no `patched` or `version_mismatch` outcome in that mode).

---

## `--build-lexical-index` mode

Hybrid search (`RETRIEVAL_MODE=hybrid`, the default) fuses the dense
embeddings with a BM25 keyword index: a SQLite FTS5 file at
`LEXICAL_INDEX_PATH` (default `$CHROMA_DOCSVC_DIR/lexical_fts.sqlite3`).
Without that file the backend logs a warning and stays dense-only, so a
deployment keeps working before the index exists.

Build it from the chunks already in the collection:

```bash
uv run sme-ingest-docsvc --build-lexical-index
```

- **No DocumentService access, no re-embedding.** It reads every chunk from
  `docsvc` and writes the FTS5 file. It needs no credentials.
- **Builds aside, then swaps.** It writes `lexical_fts.sqlite3.building` and
  renames it into place, so a running API keeps its old index until restarted.
- **Kept current by normal ingestion.** Once the file exists, every document
  ingested afterwards is added to it (and its stale versions removed), so a
  rebuild is only needed after `--backfill-tags` (the index copies `serial`,
  `language`, … for pre-filtering) or if the file is lost.
- **Access control is unaffected.** The index is a recall aid only: every
  keyword hit is re-fetched from Chroma under the caller's `where` filter
  before it is returned.
