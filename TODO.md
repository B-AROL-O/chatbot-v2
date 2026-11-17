# sme-chatbot - TODO

## Useful commands

### Ingest DocumentService and update ChromaDB

```bash
# Check connectivity to DocumentService (DOCSVC_BASE_URL from .env)
curl "$DOCSVC_BASE_URL/api/v1/documents"

# same 20, for real
uv run sme-ingest-docsvc --limit 20

# or, with no --limit, process everything pending
uv run sme-ingest-docsvc

# Create a report of the updated Chroma DB
uv run python scripts/analyze_chroma_db.py \
    data/chroma_docsvc_db/chroma.sqlite3 \
    -o "data/chroma_docsvc_db/chroma_docsvc_report-$(date +%Y-%m-%d-%H%M%S)".md
```
### Create an archive of the `data/` folder

```bash
# Create a .zip file under data/archive
scripts/backup_data_folder.sh

# Inspect today's archives
# ls -la data/archive/*
for f in data/archive/$(date +%Y-%m-%d)-bk-*-data-*; do
    ls -la "$f"
    sha256sum "$f"
done
```

<!-- EOF -->
