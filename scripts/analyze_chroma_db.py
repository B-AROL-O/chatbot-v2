#!/usr/bin/env python3
"""Analyze a Chroma `chroma.sqlite3` file and emit a Markdown report.

Introspects the SQLite file Chroma uses for its metadata segment (tables,
tenants/databases/collections/segments, per-collection embedding counts,
metadata key distributions, ingestion/sync lag) and renders the whole thing
as a self-contained Markdown report, including a Mermaid `erDiagram` of the
schema generated from `sqlite_master` + `PRAGMA` introspection (not
hardcoded), so the script keeps working across Chroma schema versions and
against any collection layout, not just this project's `docsvc` one.

The database is opened read-only (`mode=ro`) so this is safe to point at a
`chroma.sqlite3` that a live ingestion process is currently writing to.

Examples
--------
  # Print the report to stdout:
  scripts/analyze_chroma_db.py data/chroma_docsvc_db/chroma.sqlite3

  # Write it to a file instead:
  scripts/analyze_chroma_db.py data/chroma_docsvc_db/chroma.sqlite3 \\
      -o docs/chroma_docsvc_report.md
"""

from __future__ import annotations

import argparse
import datetime
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path

TOP_N_VALUES = 20
MAX_METADATA_KEYS_SAMPLED = 100


# --------------------------------------------------------------------------- #
# Data access
# --------------------------------------------------------------------------- #

def connect_readonly(db_path: Path) -> sqlite3.Connection:
    uri = f"file:{db_path.resolve()}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def table_names(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    return [r[0] for r in rows]


def row_count(conn: sqlite3.Connection, table: str) -> int:
    return conn.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0]


def table_columns(conn: sqlite3.Connection, table: str) -> list[tuple]:
    """(cid, name, type, notnull, default, pk) per PRAGMA table_info."""
    return conn.execute(f"PRAGMA table_info('{table}')").fetchall()


def table_foreign_keys(conn: sqlite3.Connection, table: str) -> list[tuple]:
    """(id, seq, ref_table, from_col, to_col, on_update, on_delete, match)."""
    return conn.execute(f"PRAGMA foreign_key_list('{table}')").fetchall()


def has_table(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone() is not None


# --------------------------------------------------------------------------- #
# Domain model
# --------------------------------------------------------------------------- #

@dataclass
class CollectionReport:
    id: str
    name: str
    dimension: int | None
    database_id: str | None
    metadata_segment_id: str | None
    vector_segment_id: str | None
    embedding_count: int = 0
    distinct_ids_in_metadata: int = 0
    metadata_max_seq_id: int | None = None
    vector_max_seq_id: int | None = None
    string_key_distributions: dict[str, list[tuple[str, int]]] = field(default_factory=dict)
    string_key_distinct_counts: dict[str, int] = field(default_factory=dict)
    int_key_stats: dict[str, tuple[int, int, float]] = field(default_factory=dict)


def fetch_collections(conn: sqlite3.Connection) -> list[dict]:
    if not has_table(conn, "collections"):
        return []
    cols = {c[1] for c in table_columns(conn, "collections")}
    select_cols = ["id", "name"]
    for optional in ("dimension", "database_id"):
        if optional in cols:
            select_cols.append(optional)
    query = f"SELECT {', '.join(select_cols)} FROM collections"
    rows = conn.execute(query).fetchall()
    return [dict(zip(select_cols, row, strict=True)) for row in rows]


def fetch_segments_for_collection(conn: sqlite3.Connection, collection_id: str) -> list[dict]:
    if not has_table(conn, "segments"):
        return []
    rows = conn.execute(
        "SELECT id, type, scope, collection FROM segments WHERE collection = ?",
        (collection_id,),
    ).fetchall()
    return [dict(id=r[0], type=r[1], scope=r[2], collection=r[3]) for r in rows]


def build_collection_report(conn: sqlite3.Connection, coll: dict) -> CollectionReport:
    segments = fetch_segments_for_collection(conn, coll["id"])
    meta_seg = next((s for s in segments if s["scope"] == "METADATA"), None)
    vec_seg = next((s for s in segments if s["scope"] == "VECTOR"), None)

    report = CollectionReport(
        id=coll["id"],
        name=coll.get("name", "?"),
        dimension=coll.get("dimension"),
        database_id=coll.get("database_id"),
        metadata_segment_id=meta_seg["id"] if meta_seg else None,
        vector_segment_id=vec_seg["id"] if vec_seg else None,
    )

    if report.metadata_segment_id and has_table(conn, "embeddings"):
        report.embedding_count = conn.execute(
            "SELECT count(*) FROM embeddings WHERE segment_id = ?",
            (report.metadata_segment_id,),
        ).fetchone()[0]

    if has_table(conn, "max_seq_id"):
        for seg_id, attr in (
            (report.metadata_segment_id, "metadata_max_seq_id"),
            (report.vector_segment_id, "vector_max_seq_id"),
        ):
            if not seg_id:
                continue
            row = conn.execute(
                "SELECT seq_id FROM max_seq_id WHERE segment_id = ?", (seg_id,)
            ).fetchone()
            if row is not None:
                setattr(report, attr, row[0])

    if not (report.metadata_segment_id and has_table(conn, "embedding_metadata")
            and has_table(conn, "embeddings")):
        return report

    report.distinct_ids_in_metadata = conn.execute(
        """
        SELECT count(DISTINCT m.id) FROM embedding_metadata m
        JOIN embeddings e ON e.id = m.id
        WHERE e.segment_id = ?
        """,
        (report.metadata_segment_id,),
    ).fetchone()[0]

    keys = [
        r[0] for r in conn.execute(
            """
            SELECT DISTINCT m.key FROM embedding_metadata m
            JOIN embeddings e ON e.id = m.id
            WHERE e.segment_id = ?
            ORDER BY m.key
            LIMIT ?
            """,
            (report.metadata_segment_id, MAX_METADATA_KEYS_SAMPLED),
        ).fetchall()
    ]

    for key in keys:
        if key == "chroma:document":
            continue  # raw chunk text, not a categorical/numeric facet
        distinct_count = conn.execute(
            """
            SELECT count(DISTINCT m.string_value) FROM embedding_metadata m
            JOIN embeddings e ON e.id = m.id
            WHERE e.segment_id = ? AND m.key = ? AND m.string_value IS NOT NULL
            """,
            (report.metadata_segment_id, key),
        ).fetchone()[0]
        if distinct_count:
            report.string_key_distinct_counts[key] = distinct_count
            report.string_key_distributions[key] = conn.execute(
                """
                SELECT m.string_value, count(*) AS n FROM embedding_metadata m
                JOIN embeddings e ON e.id = m.id
                WHERE e.segment_id = ? AND m.key = ? AND m.string_value IS NOT NULL
                GROUP BY m.string_value ORDER BY n DESC LIMIT ?
                """,
                (report.metadata_segment_id, key, TOP_N_VALUES),
            ).fetchall()
            continue

        int_stats = conn.execute(
            """
            SELECT min(m.int_value), max(m.int_value), avg(m.int_value)
            FROM embedding_metadata m
            JOIN embeddings e ON e.id = m.id
            WHERE e.segment_id = ? AND m.key = ? AND m.int_value IS NOT NULL
            """,
            (report.metadata_segment_id, key),
        ).fetchone()
        if int_stats and int_stats[0] is not None:
            report.int_key_stats[key] = int_stats

    return report


def fetch_ingestion_queue_status(conn: sqlite3.Connection) -> dict | None:
    if not has_table(conn, "embeddings_queue"):
        return None
    total = conn.execute("SELECT count(*) FROM embeddings_queue").fetchone()[0]
    bounds = conn.execute(
        "SELECT min(seq_id), max(seq_id), min(created_at), max(created_at) "
        "FROM embeddings_queue"
    ).fetchone()
    by_op = conn.execute(
        "SELECT operation, count(*) FROM embeddings_queue GROUP BY operation"
    ).fetchall()
    return {
        "total": total,
        "min_seq_id": bounds[0],
        "max_seq_id": bounds[1],
        "oldest": bounds[2],
        "newest": bounds[3],
        "by_operation": by_op,
    }


# --------------------------------------------------------------------------- #
# Mermaid ER diagram (generated from live schema introspection)
# --------------------------------------------------------------------------- #

_MERMAID_SAFE_TYPE = {
    "INTEGER": "int",
    "TEXT": "string",
    "BLOB": "blob",
    "TIMESTAMP": "timestamp",
    "REAL": "float",
}


def _mermaid_type(sql_type: str) -> str:
    return _MERMAID_SAFE_TYPE.get((sql_type or "").upper(), (sql_type or "any").lower() or "any")


def build_mermaid_er_diagram(conn: sqlite3.Connection, tables: list[str]) -> str:
    lines = ["erDiagram"]

    for table in tables:
        cols = table_columns(conn, table)
        fk_from_cols = {fk[3] for fk in table_foreign_keys(conn, table)}
        lines.append(f"    {table} {{")
        for _cid, name, sql_type, _notnull, _default, pk in cols:
            key_tag = ""
            if pk:
                key_tag = "PK"
            elif name in fk_from_cols:
                key_tag = "FK"
            col_line = f"        {_mermaid_type(sql_type)} {name}"
            if key_tag:
                col_line += f" {key_tag}"
            lines.append(col_line)
        lines.append("    }")

    seen_edges: set[tuple[str, str, str]] = set()
    unresolved: list[str] = []
    for table in tables:
        for fk in table_foreign_keys(conn, table):
            _id, _seq, ref_table, from_col, _to_col, *_rest = fk
            edge_key = (table, ref_table, from_col)
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            if ref_table not in tables:
                # Declared FK target doesn't match any real table (seen in
                # Chroma's own schema: segments.collection REFERENCES
                # collection(id), but the table is named "collections").
                # SQLite never validates this, so surface it instead of
                # silently dropping the edge.
                unresolved.append(
                    f"{table}.{from_col} -> {ref_table} (declared FK target "
                    f"table not found)"
                )
                continue
            # child (many) references parent (one) via from_col.
            lines.append(f'    {ref_table} ||--o{{ {table} : "{from_col}"')

    if unresolved:
        lines.append("    %% Unresolved declared foreign keys (target table")
        lines.append("    %% name doesn't exist verbatim in this schema):")
        for note in unresolved:
            lines.append(f"    %%   {note}")

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Markdown rendering
# --------------------------------------------------------------------------- #

def _md_table(headers: list[str], rows: list[list]) -> str:
    if not rows:
        return "_(no rows)_\n"
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_report(conn: sqlite3.Connection, db_path: Path) -> str:
    stat = db_path.stat()
    generated_at = datetime.datetime.now(datetime.UTC).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )
    db_mtime = datetime.datetime.fromtimestamp(
        stat.st_mtime, tz=datetime.UTC
    ).strftime("%Y-%m-%d %H:%M:%S UTC")

    tables = table_names(conn)
    row_counts = {t: row_count(conn, t) for t in tables}

    tenants = conn.execute("SELECT * FROM tenants").fetchall() if has_table(conn, "tenants") else []
    databases = conn.execute("SELECT * FROM databases").fetchall() if has_table(conn, "databases") else []

    collections = fetch_collections(conn)
    collection_reports = [build_collection_report(conn, c) for c in collections]

    queue_status = fetch_ingestion_queue_status(conn)
    mermaid_diagram = build_mermaid_er_diagram(conn, tables)

    lines: list[str] = []
    lines.append(f"# Chroma DB Analysis: `{db_path.name}`")
    lines.append("")
    lines.append(f"- **Path:** `{db_path}`")
    lines.append(f"- **Size:** {stat.st_size / (1024 * 1024):.1f} MiB "
                 f"({stat.st_size:,} bytes)")
    lines.append(f"- **Last modified:** {db_mtime}")
    lines.append(f"- **Report generated:** {generated_at}")
    lines.append("")

    lines.append("## Overview")
    lines.append("")
    lines.append(_md_table(
        ["Table", "Rows"],
        [[t, f"{row_counts.get(t, 0):,}"] for t in tables],
    ))

    lines.append("## Tenants / Databases")
    lines.append("")
    if tenants:
        lines.append(_md_table(["tenant_id"], [[t[0]] for t in tenants]))
    if databases:
        lines.append(_md_table(["id", "name", "tenant_id"], [list(d) for d in databases]))
    if not tenants and not databases:
        lines.append("_No `tenants`/`databases` tables found._\n")

    lines.append("## Collections")
    lines.append("")
    if collection_reports:
        lines.append(_md_table(
            ["Name", "id", "dimension", "database_id",
             "embeddings", "distinct ids", "metadata seq_id", "vector seq_id"],
            [
                [
                    c.name, c.id, c.dimension if c.dimension is not None else "-",
                    c.database_id or "-",
                    f"{c.embedding_count:,}",
                    f"{c.distinct_ids_in_metadata:,}",
                    c.metadata_max_seq_id if c.metadata_max_seq_id is not None else "-",
                    c.vector_max_seq_id if c.vector_max_seq_id is not None else "-",
                ]
                for c in collection_reports
            ],
        ))
    else:
        lines.append("_No collections found._\n")

    for c in collection_reports:
        lines.append(f"### Collection detail: `{c.name}`")
        lines.append("")

        if (c.metadata_max_seq_id is not None and c.vector_max_seq_id is not None
                and c.metadata_max_seq_id != c.vector_max_seq_id):
            lag = c.metadata_max_seq_id - c.vector_max_seq_id
            lines.append(
                f"> ⚠️ Vector index (HNSW) segment lags the metadata segment by "
                f"**{lag:,}** sequence id(s) (metadata seq_id="
                f"{c.metadata_max_seq_id:,}, vector seq_id={c.vector_max_seq_id:,}). "
                f"This is expected while an ingestion pipeline is actively writing; "
                f"if no ingestion is running, it indicates the HNSW index hasn't "
                f"been flushed/persisted yet."
            )
            lines.append("")

        if c.string_key_distinct_counts:
            lines.append("**Metadata key cardinality:**")
            lines.append("")
            lines.append(_md_table(
                ["Key", "Distinct values"],
                [[k, f"{v:,}"] for k, v in sorted(c.string_key_distinct_counts.items())],
            ))
            for key, dist in c.string_key_distributions.items():
                lines.append(f"**Top values for `{key}`:**")
                lines.append("")
                lines.append(_md_table(
                    ["Value", "Count"],
                    [[v if v is not None else "_(null)_", f"{n:,}"] for v, n in dist],
                ))

        if c.int_key_stats:
            lines.append("**Numeric metadata stats:**")
            lines.append("")
            lines.append(_md_table(
                ["Key", "Min", "Max", "Avg"],
                [[k, mn, mx, f"{avg:.2f}"] for k, (mn, mx, avg) in c.int_key_stats.items()],
            ))

        if not c.string_key_distinct_counts and not c.int_key_stats:
            lines.append("_No embedding metadata keys found for this collection._\n")

    lines.append("## Ingestion / Sync Status")
    lines.append("")
    if queue_status is None:
        lines.append("_No `embeddings_queue` table found._\n")
    else:
        lines.append(_md_table(
            ["Pending rows", "seq_id range", "Oldest entry", "Newest entry"],
            [[
                f"{queue_status['total']:,}",
                f"{queue_status['min_seq_id']} - {queue_status['max_seq_id']}"
                if queue_status['total'] else "-",
                queue_status["oldest"] or "-",
                queue_status["newest"] or "-",
            ]],
        ))
        if queue_status["by_operation"]:
            lines.append("**Queue rows by operation code:**")
            lines.append("")
            lines.append(_md_table(
                ["operation", "count"],
                [[op, f"{n:,}"] for op, n in queue_status["by_operation"]],
            ))
        if queue_status["total"] > 0:
            lines.append(
                "> ℹ️ Non-zero pending rows in `embeddings_queue` usually mean an "
                "ingestion process is (or was, if it crashed) actively writing to "
                "this database.\n"
            )

    lines.append("## Schema")
    lines.append("")
    lines.append(
        "Full column listing per table, introspected via `PRAGMA table_info` "
        "(**PK**) and `PRAGMA foreign_key_list` (**FK**)."
    )
    lines.append("")
    for table in tables:
        cols = table_columns(conn, table)
        fks = table_foreign_keys(conn, table)
        fk_targets = {fk[3]: fk[2] for fk in fks}  # from_col -> ref_table
        lines.append(f"**`{table}`**")
        lines.append("")
        rows = []
        for _cid, name, sql_type, notnull, default, pk in cols:
            flags = []
            if pk:
                flags.append("PK")
            if name in fk_targets:
                ref_table = fk_targets[name]
                note = "" if ref_table in tables else " ⚠️ unresolved"
                flags.append(f"FK → {ref_table}{note}")
            if notnull:
                flags.append("NOT NULL")
            rows.append([name, sql_type or "-", ", ".join(flags) or "-", default or "-"])
        lines.append(_md_table(["Column", "Type", "Flags", "Default"], rows))

    lines.append("## Entity-Relationship Diagram")
    lines.append("")
    lines.append("```mermaid")
    lines.append(mermaid_diagram)
    lines.append("```")
    lines.append("")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="analyze_chroma_db.py",
        description="Analyze a Chroma chroma.sqlite3 file and produce a "
                    "Markdown report (schema, collections, embedding metadata "
                    "distributions, ingestion status, Mermaid ER diagram).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("db_path", type=Path,
                        help="Path to the chroma.sqlite3 file to analyze.")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Write the report to this file instead of stdout.")
    args = parser.parse_args(argv)

    if not args.db_path.is_file():
        parser.error(f"no such file: {args.db_path}")

    conn = connect_readonly(args.db_path)
    try:
        report = build_report(conn, args.db_path)
    finally:
        conn.close()

    if args.output:
        args.output.write_text(report, encoding="utf-8")
        print(f"Report written to {args.output}", file=sys.stderr)
    else:
        print(report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
