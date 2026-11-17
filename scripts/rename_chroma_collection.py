#!/usr/bin/env python3
"""Rename a collection inside an existing Chroma store, in place.

The backend looks collections up by name (`CHROMA_CDO_COLLECTION`,
`CHROMA_MANUALS_COLLECTION`, `CHROMA_DOCSVC_COLLECTION`). When those names
change, an existing store can be moved over with this script instead of being
re-ingested: the embeddings, documents and metadata stay as they are, and the
FTS5 lexical index (keyed by chunk id) keeps working.

Stop the backend and any ingestion first. The script refuses to run when the
old collection is missing or the new name is already taken. Ingestion uses
`get_or_create_collection`, so running it before the rename leaves an empty
collection under the new name; delete that one, then rename.

Examples
--------
  # Show what would happen:
  scripts/rename_chroma_collection.py data/chroma_docsvc_db old_name docsvc --dry-run

  # Rename:
  scripts/rename_chroma_collection.py data/chroma_docsvc_db old_name docsvc
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import chromadb


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rename a collection inside a Chroma store.")
    parser.add_argument("store", type=Path, help="Chroma PersistentClient directory, e.g. data/chroma_docsvc_db")
    parser.add_argument("old", help="current collection name")
    parser.add_argument("new", help="new collection name")
    parser.add_argument("--dry-run", action="store_true", help="check and report, change nothing")
    args = parser.parse_args(argv)

    if not (args.store / "chroma.sqlite3").is_file():
        print(f"error: no Chroma store at {args.store}", file=sys.stderr)
        return 1

    client = chromadb.PersistentClient(path=str(args.store))
    names = {c.name for c in client.list_collections()}
    if args.old not in names:
        print(f"error: no collection {args.old!r} in {args.store} (found: {sorted(names)})", file=sys.stderr)
        return 1
    if args.new in names:
        print(f"error: collection {args.new!r} already exists in {args.store}", file=sys.stderr)
        return 1

    collection = client.get_collection(args.old)
    count = collection.count()
    if args.dry_run:
        print(f"would rename {args.old!r} -> {args.new!r} ({count} chunks) in {args.store}")
        return 0

    collection.modify(name=args.new)
    renamed = client.get_collection(args.new)
    print(f"renamed {args.old!r} -> {args.new!r} ({renamed.count()} chunks) in {args.store}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
