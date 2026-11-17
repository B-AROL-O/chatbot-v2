# HOWTO — View / Inspect SQLite3 Files

Options for browsing the SQLite databases in this project, and the workarounds
needed for the Chroma vector-store files.

Databases you may want to inspect:

| File | Contents |
| ------ | ---------- |
| `data/cdo.db` | Normalized CDO order/machine/sample data (ETL output) |
| `data/chroma_cdo_db/chroma.sqlite3` | Chroma bookkeeping for the CDO vector store |
| `data/chroma_manuals_db/chroma.sqlite3` | Chroma bookkeeping for the manuals vector store |
| `data/chroma_docsvc_db/chroma.sqlite3` | Chroma bookkeeping for the DocumentService vector store |

---

## Two gotchas with the Chroma files

1. **Size** — after a full `sme-ingest-docsvc` run the file exceeds the free
   tier of the VS Code *SQLite Viewer* extension ("Database is too large.
   Upgrade to SQLite Viewer PRO"). Use one of the free tools below instead.

2. **Dangling foreign key** — Chroma's own schema declares
   `segments.collection TEXT REFERENCES collection(id)`, but the referenced
   table is actually named `collections` (plural). SQLite never validates FK
   targets at DDL time, so the file is perfectly healthy — but any viewer that
   introspects foreign keys via peewee (e.g. `sqlite_web`) crashes at startup
   with `KeyError: 'collection'`. See the per-tool workarounds below.

---

## Option 1 — Datasette (recommended)

Free, no size limit, tolerates the dangling Chroma FK, works on the original
file untouched, browser-based (VS Code auto-forwards the port from the dev
container):

```bash
uvx datasette --host 0.0.0.0 --port 8001 data/chroma_docsvc_db/chroma.sqlite3
```

Then open <http://localhost:8001>. Works the same for `data/cdo.db`. Pass
`--immutable` instead of the positional path if you want a hard guarantee that
the viewer cannot lock or modify a store the ingestion pipeline writes to:

```bash
uvx datasette --host 0.0.0.0 --port 8001 --immutable data/chroma_docsvc_db/chroma.sqlite3
```

## Option 2 — sqlite3 CLI

Always available, no server needed. Read-only URI mode is the safe default:

```bash
sqlite3 "file:data/chroma_docsvc_db/chroma.sqlite3?mode=ro" ".tables"
sqlite3 "file:data/cdo.db?mode=ro" 'SELECT COUNT(*) FROM "order";'
```

Useful dot-commands: `.tables`, `.schema <table>`, `.mode box` (pretty
output), `.headers on`.

## Option 3 — sqlite_web

Nice single-table browser UI, but it crashes on Chroma files because of the
dangling FK (gotcha 2 above). It works fine directly on `data/cdo.db`:

```bash
uv run pipx run sqlite_web --host 0.0.0.0 data/cdo.db
```

For a Chroma file, patch the FK typo on a **copy** first (never edit the live
DB Chroma is using):

```bash
cp data/chroma_docsvc_db/chroma.sqlite3 /tmp/chroma-fixed.sqlite3
sqlite3 /tmp/chroma-fixed.sqlite3 <<'EOF'
.dbconfig defensive off
PRAGMA writable_schema=1;
UPDATE sqlite_master SET sql = replace(sql, 'REFERENCES collection(id)', 'REFERENCES collections(id)')
 WHERE name='segments' AND type='table';
PRAGMA writable_schema=0;
EOF
uv run pipx run sqlite_web --host 0.0.0.0 /tmp/chroma-fixed.sqlite3
```

The `.dbconfig defensive off` line is required — modern sqlite3 CLIs block
`sqlite_master` writes by default.

## Option 4 — VS Code SQLite Viewer extension

The [SQLite Viewer](https://marketplace.visualstudio.com/items?itemName=qwtel.sqlite-viewer)
extension is convenient for small files (e.g. `data/cdo.db`, or a
`--limit`-restricted ingestion run), but the free tier refuses files above its
size cap — the symptom that prompted this page. No workaround short of a
PRO license; use Datasette for large files.

## Option 5 — DB Browser for SQLite (GUI)

[DB Browser for SQLite](https://sqlitebrowser.org/) (`sqlitebrowser`) is a
solid desktop GUI with no size limit. It needs a desktop environment, so it is
a host-machine option rather than a dev-container one.

---

## Note on what the Chroma sqlite file contains

`chroma.sqlite3` holds collection/segment bookkeeping and the
`embedding_metadata` table (document text and metadata per chunk). The actual
embedding vectors live in the binary segment directories next to it, so a
SQLite viewer shows you the chunks and metadata but not the vectors — which is
usually exactly what you want when debugging ingestion.
