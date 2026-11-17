"""SQLite FTS5 keyword index over the DocumentService chunk corpus.

The dense retriever (MiniLM, 128-token window) cannot rank markdown table
chunks: on the 90001 golden set it placed the gold page in the top 20 for
3/10 probes, while BM25 over the *same* chunks ranked it first for 9/10.
This module is the
lexical half of the hybrid retriever in ``docsvc_rag.py``.

Design points:

- **Side tables, same chunk ids.** An FTS5 table holds only the text; a
  plain table ``chunk_meta`` (same rowid) holds the Chroma chunk id and the
  filter metadata, *indexed*. FTS5 ``UNINDEXED`` columns cannot be indexed,
  so keeping ids there made every delete-by-id a full scan (quadratic
  ingestion). A lexical hit is fused with dense hits and re-fetched from
  Chroma by id. Built by ingestion, or backfilled from an existing collection with
  ``sme-ingest-docsvc --build-lexical-index`` (no re-embedding).
- **Access control stays in Chroma.** The SQL pre-filter below is a *recall*
  optimisation only — it may be looser than the ``where`` filter (it drops
  clauses it cannot translate), never stricter. Callers must re-fetch the
  candidate ids from Chroma *with the same ``where``* before returning
  anything; ``docsvc_rag._lexical_hits`` does, and a unit test pins it.
- **stdlib only.** ``sqlite3`` ships FTS5; no new dependency.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import threading
from collections.abc import Iterable
from pathlib import Path

logger = logging.getLogger(__name__)

TABLE = "chunks_fts"
META = "chunk_meta"

# Metadata keys mirrored as UNINDEXED columns, so the common filters
# (`serial`, `language`, ...) can narrow the candidate set *before* ranking.
# Without this, a query over ~857k chunks would rank every machine's manual
# and the 90001 page could fall outside the candidate window.
FILTER_COLUMNS = ("document_id", "filename", "serial", "language", "category", "customer", "source")

_SCHEMA = [
    f"CREATE VIRTUAL TABLE IF NOT EXISTS {TABLE} USING fts5("
    "text, tokenize = 'unicode61 remove_diacritics 2')",
    f"CREATE TABLE IF NOT EXISTS {META} (rowid INTEGER PRIMARY KEY, chunk_id TEXT NOT NULL UNIQUE, "
    + ", ".join(f"{c} TEXT" for c in FILTER_COLUMNS)
    + ")",
    f"CREATE INDEX IF NOT EXISTS {META}_document_id ON {META}(document_id)",
    f"CREATE INDEX IF NOT EXISTS {META}_serial ON {META}(serial)",
]

# --- Text normalisation (shared by indexing and querying) ------------------

_PICTURE_RE = re.compile(r"\*\*==> picture \[[^\]]*\] intentionally omitted <==\*\*")
_PICTURE_TEXT_RE = re.compile(r"\*\*----- (?:Start|End) of picture text -----\*\*")
_MARKUP_RE = re.compile(r"<br>|[|*#_`]")
# "M 12" and "M12" are the same thread size; the torque tables print both.
_THREAD_RE = re.compile(r"\bM\s+(\d{1,2})\b")

# Words that carry no retrieval signal in either question language. Kept
# short on purpose: BM25's IDF already down-weights common words, this only
# removes the ones that would otherwise match nearly every chunk.
_STOPWORDS = frozenset(
    [
        # Italian
        "il", "lo", "la", "i", "gli", "le", "di", "del", "della", "dello", "dei", "degli",
        "delle", "da", "dal", "dalla", "dai", "in", "nel", "nella", "nei", "nelle", "su", "sul",
        "sulla", "sui", "sulle", "con", "per", "tra", "fra", "un", "uno", "una", "e", "ed", "o",
        "a", "al", "alla", "ai", "che", "chi", "cosa", "come", "quale", "quali", "qual",
        "quanto", "quanta", "quanti", "quante", "ogni", "non", "si", "se", "va", "vanno",
        "devo", "deve", "posso", "sono", "è", "ha", "hanno", "mi", "ti", "ci", "vi", "ne",
        "dato",
        # English
        "the", "a", "an", "of", "to", "in", "on", "at", "for", "with", "from", "by", "is",
        "are", "be", "what", "which", "how", "many", "much", "does", "do", "can", "should",
        "must", "and", "or", "this", "that", "machine",
    ]
)


def normalise(text: str) -> str:
    """Strip pymupdf4llm markup and unify thread sizes, for matching only."""
    text = _PICTURE_RE.sub(" ", text)
    text = _PICTURE_TEXT_RE.sub(" ", text)
    text = _MARKUP_RE.sub(" ", text)
    return _THREAD_RE.sub(r"M\1", text)


def query_terms(query: str) -> list[str]:
    """Tokens of ``query`` worth matching, lowercased, order-preserving, unique."""
    seen: dict[str, None] = {}
    for token in re.findall(r"\w+", normalise(query).lower()):
        if token in _STOPWORDS or (len(token) < 2 and not token.isdigit()):
            continue
        seen.setdefault(token, None)
    return list(seen)


_TRAILING_VOWELS_RE = re.compile(r"[aeiouàèéìòù]+$")


def _stem(term: str) -> str | None:
    """A light Italian stem for prefix matching, or ``None`` to match exactly.

    Strips trailing vowels so inflections meet (``pesa``/``peso`` -> ``pes*``,
    ``lavata``/``lavaggio`` -> ``lav*``). There is no Italian stemmer in FTS5;
    on the 90001 set this lifts lexical recall@1 from 0.53 to 0.65, and BM25's
    IDF keeps short, common stems from dominating.
    """
    if not term.isalpha() or len(term) < 4:
        return None
    stem = _TRAILING_VOWELS_RE.sub("", term)
    return stem if len(stem) >= 3 else None


def _match_expression(terms: list[str]) -> str:
    # Every term quoted, so FTS5 operators/keywords in user text are inert.
    parts = []
    for term in terms:
        stem = _stem(term)
        quoted = '"' + (stem or term).replace('"', '""') + '"'
        parts.append(quoted + "*" if stem else quoted)
    return " OR ".join(parts)


def _where_values(where: dict | None, key: str) -> set[str]:
    """Every equality value ``where`` pins ``key`` to (any nesting under ``$and``)."""
    if not where:
        return set()
    if "$and" in where:
        return set().union(*(_where_values(sub, key) for sub in where["$and"]))
    cond = where.get(key)
    if cond is None:
        return set()
    if isinstance(cond, dict):
        return {str(v).lower() for v in cond.get("$in", [])} | ({str(cond["$eq"]).lower()} if "$eq" in cond else set())
    return {str(cond).lower()}


# --- Where-filter translation (recall only, see module docstring) ----------


def _translate(where: dict | None) -> tuple[list[str], list]:
    """Translate the subset of a Chroma ``where`` we can express in SQL.

    Returns ``(sql_clauses, params)`` to AND together. Unsupported keys or
    operators are *dropped*, which can only widen the candidate set; the
    Chroma re-fetch enforces the real filter.
    """
    if not where:
        return [], []
    if "$and" in where:
        clauses: list[str] = []
        params: list = []
        for sub in where["$and"]:
            c, p = _translate(sub)
            clauses += c
            params += p
        return clauses, params
    if len(where) != 1:
        return [], []
    key, cond = next(iter(where.items()))
    if key not in FILTER_COLUMNS:
        return [], []
    if isinstance(cond, dict):
        if set(cond) == {"$in"} and cond["$in"]:
            values = [str(v) for v in cond["$in"]]
            return [f"m.{key} IN ({', '.join('?' for _ in values)})"], values
        if set(cond) == {"$eq"}:
            return [f"m.{key} = ?"], [str(cond["$eq"])]
        return [], []
    return [f"m.{key} = ?"], [str(cond)]


# --- Index handle ------------------------------------------------------------


class LexicalIndex:
    """Thin wrapper around one FTS5 database file."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._conn: sqlite3.Connection | None = None
        # One connection shared by FastAPI's worker threads; serialise use of
        # it rather than rely on how the SQLite library was compiled.
        self._lock = threading.Lock()

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # check_same_thread=False: FastAPI runs sync endpoints in a thread
            # pool (queries hold `self._lock`); writes only happen from the
            # single-threaded ingestion CLI.
            self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
            for statement in _SCHEMA:
                self._conn.execute(statement)
        return self._conn

    def exists(self) -> bool:
        return self.path.exists()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def add(self, ids: Iterable[str], documents: Iterable[str], metadatas: Iterable[dict]) -> int:
        """Insert (or replace) chunks. Returns the number of rows written."""
        conn = self._connect()
        rows = list(zip(ids, documents, metadatas, strict=True))
        if not rows:
            return 0
        self._delete_ids(conn, [chunk_id for chunk_id, _, _ in rows])
        placeholders = ", ".join("?" for _ in range(1 + len(FILTER_COLUMNS)))
        for chunk_id, text, meta in rows:
            meta = meta or {}
            cur = conn.execute(
                f"INSERT INTO {META} (chunk_id, {', '.join(FILTER_COLUMNS)}) VALUES ({placeholders})",
                (chunk_id, *(_as_text(meta.get(c)) for c in FILTER_COLUMNS)),
            )
            conn.execute(f"INSERT INTO {TABLE} (rowid, text) VALUES (?, ?)", (cur.lastrowid, normalise(text or "")))
        conn.commit()
        return len(rows)

    @staticmethod
    def _delete_rowids(conn: sqlite3.Connection, rowids: list[int]) -> None:
        conn.executemany(f"DELETE FROM {TABLE} WHERE rowid = ?", [(r,) for r in rowids])
        conn.executemany(f"DELETE FROM {META} WHERE rowid = ?", [(r,) for r in rowids])

    def _delete_ids(self, conn: sqlite3.Connection, chunk_ids: list[str]) -> None:
        rowids: list[int] = []
        for start in range(0, len(chunk_ids), 500):
            batch = chunk_ids[start : start + 500]
            rowids += [
                r[0]
                for r in conn.execute(
                    f"SELECT rowid FROM {META} WHERE chunk_id IN ({', '.join('?' for _ in batch)})", batch
                )
            ]
        self._delete_rowids(conn, rowids)

    def delete_document(self, document_id: str, keep_ids: Iterable[str] = ()) -> None:
        """Remove a document's rows, except ``keep_ids`` (stale-version delete)."""
        conn = self._connect()
        keep = set(keep_ids)
        rowids = [
            rowid
            for rowid, chunk_id in conn.execute(f"SELECT rowid, chunk_id FROM {META} WHERE document_id = ?", [document_id])
            if chunk_id not in keep
        ]
        self._delete_rowids(conn, rowids)
        conn.commit()

    def count(self) -> int:
        return self._connect().execute(f"SELECT count(*) FROM {META}").fetchone()[0]

    def search(self, query: str, limit: int = 20, where: dict | None = None) -> list[str]:
        """Return up to ``limit`` chunk ids, best BM25 first.

        ``where`` narrows candidates for recall; it is **not** an access
        check (see module docstring).
        """
        # A serial the search is already filtered to matches every chunk in
        # scope, so as a term it only adds noise ("la 90001" in the question).
        # Dropping it took lexical recall@5 on the 90001 set from 0.50 to 0.85.
        pinned = _where_values(where, "serial")
        terms = [t for t in query_terms(query) if t not in pinned]
        if not terms:
            return []
        clauses, params = _translate(where)
        sql = f"SELECT m.chunk_id FROM {TABLE} JOIN {META} m ON m.rowid = {TABLE}.rowid WHERE {TABLE} MATCH ?"
        if clauses:
            sql += " AND " + " AND ".join(clauses)
        sql += f" ORDER BY bm25({TABLE}) LIMIT ?"
        try:
            with self._lock:
                rows = self._connect().execute(sql, [_match_expression(terms), *params, limit]).fetchall()
        except sqlite3.Error as exc:
            logger.warning("lexical search failed for %r: %s", query, exc)
            return []
        return [r[0] for r in rows]


def _as_text(value) -> str | None:
    return None if value is None else str(value)
