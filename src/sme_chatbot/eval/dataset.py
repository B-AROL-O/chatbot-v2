"""WP1 — load ``$EVAL_DATASETS_DIR/*-domande.csv`` into :class:`EvalItem` objects.

The golden set's headers are Italian and carry non-ASCII (``Difficoltà (1-3)``).
They are mapped **once, here**, and never spread through the codebase — the same
discipline the ETL applies to its Italian column names (``AGENTS.md`` §Coding
Patterns 5). Header lookup is accent- and case-insensitive so a re-export of the
workbook with slightly different accents still loads.

Three files make up a question set, only the first of which is required:

``<stem>-domande.csv``
    the questions, checked in as received (``datasets/README.md`` in the eval repo: read-only).
``<stem>-legenda.csv``
    column meanings and execution notes; carried along for provenance.
``derived/<stem>-rubric.csv``
    the optional, **derived** scoring sidecar (``ID,required,forbidden,note``).
    It lives under ``derived/`` rather than next to the sources so the
    as-received folder stays unmixed (plan §8 Q2).

Every dataset carries its own ``sha256`` and row count, so a run scored against
a later edit of the answer key is detectable (plan §7 R4).
"""

from __future__ import annotations

import csv
import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from sme_chatbot import config
from sme_chatbot.eval.scoring import Anchor, detect_language, extract_value_anchors, required_anchors

# --- source column names, mapped once -------------------------------------

COL_ID = "ID"
COL_CATEGORIA = "Categoria"
COL_DIFFICOLTA = "Difficoltà (1-3)"
COL_DOMANDA = "Domanda"
COL_RISPOSTA_ATTESA = "Risposta attesa"
COL_COMPORTAMENTO = "Comportamento atteso"
COL_CAPITOLO = "Capitolo fonte"
COL_PAGINA = "Pag. stampata"
COL_DOCUMENTO = "Documento"

# The workbook's yellow columns — filled in by the runner, never read from.
COL_RISPOSTA_CORRETTA = "Risposta corretta? (S/N/P)"
COL_FONTI_CORRETTE = "Fonti corrette? (S/N)"
COL_HA_CITATO = "Ha citato? (S/N)"
COL_NOTE_ESITO = "Note esito"
COL_TRACE = "Trace Langfuse"

YELLOW_COLUMNS = (
    COL_RISPOSTA_CORRETTA,
    COL_FONTI_CORRETTE,
    COL_HA_CITATO,
    COL_NOTE_ESITO,
    COL_TRACE,
)

# Categoria values that carry a behavioural expectation (plan WP1 table).
CATEGORIA_REFUSAL = "Non rispondibile"
CATEGORIA_PREMISE = "Premessa falsa"
CATEGORIA_DISAMBIGUATION = ("Tabella ambigua", "Scoping macchina")

_SERIAL_RE = re.compile(r"\b\d{5}\b")

# A `Pag. stampata` range wider than this is a typo, not a page span.
_MAX_PAGE_SPAN = 50


class DatasetError(RuntimeError):
    """The question set could not be read or is missing a required column."""


def normalise_header(name: str) -> str:
    """Accent-free, lowercase, whitespace-collapsed form of a header cell."""
    decomposed = unicodedata.normalize("NFD", name.replace("﻿", ""))
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", stripped).strip().lower()


def parse_pages(raw: str) -> frozenset[int]:
    """Parse ``Pag. stampata`` into a set of printed page numbers.

    Handles every shape the file actually uses: a single page (``8``), a range
    (``108-109``), a list (``29, 178``) and "no page" (``—``). A range wider
    than :data:`_MAX_PAGE_SPAN` is treated as a typo and reduced to its two
    endpoints rather than exploding into hundreds of pages.
    """
    pages: set[int] = set()
    for part in re.split(r"[;,]", raw or ""):
        part = part.strip()
        if not part:
            continue
        span = re.fullmatch(r"(\d+)\s*[-–—]\s*(\d+)", part)
        if span:
            start, end = int(span.group(1)), int(span.group(2))
            if start > end:
                start, end = end, start
            if end - start <= _MAX_PAGE_SPAN:
                pages.update(range(start, end + 1))
            else:
                pages.update({start, end})
            continue
        for number in re.findall(r"\d+", part):
            pages.add(int(number))
    return frozenset(pages)


def _split_patterns(raw: str) -> list[str]:
    """Split a rubric cell on ``|`` into individual patterns."""
    return [p.strip() for p in (raw or "").split("|") if p.strip()]


@dataclass(frozen=True)
class EvalItem:
    """One question plus everything derived from it that scoring needs."""

    # --- verbatim source columns
    id: str
    categoria: str
    difficolta: int | None
    domanda: str
    risposta_attesa: str
    comportamento_atteso: str
    capitolo_fonte: str
    pag_stampata: str
    documento: str
    raw: dict[str, str] = field(repr=False, default_factory=dict)

    # --- derived (plan WP1 table)
    serial: str | None = None
    language: str = "it"
    expects_refusal: bool = False
    expects_premise_correction: bool = False
    expects_disambiguation: bool = False
    gold_pages: frozenset[int] = frozenset()
    value_anchors: tuple[Anchor, ...] = ()

    # --- from the optional rubric sidecar
    required: tuple[str, ...] = ()
    forbidden: tuple[str, ...] = ()
    rubric_note: str = ""

    @property
    def gold_document(self) -> str:
        return self.documento

    @property
    def anchors(self) -> tuple[Anchor, ...]:
        """Everything ``anchor_recall`` checks: auto values + rubric ``required``."""
        return self.value_anchors + tuple(required_anchors(list(self.required)))

    @property
    def requires_qualification(self) -> bool:
        """Does a bare single value count as a failure for this item?

        ``expects_disambiguation`` is a *category* flag, and the category covers
        two different situations. Q06 ("coppia di una M12") is genuinely
        ambiguous — the manual holds two tables, so the answer key carries two
        values in the same unit and one number alone is wrong even when it is a
        real number. Q07/Q08 name the material in the question itself, so the
        answer key carries a single value and a bare number is simply correct.

        The distinction is therefore read off the answer key rather than
        hard-coded per ID: an item needs qualification when its expected answer
        holds two or more values sharing a unit, or no values at all (Q34, where
        the expected answer *is* "ask which machine").
        """
        if not self.expects_disambiguation:
            return False
        if not self.value_anchors:
            return True
        units = [a.unit for a in self.value_anchors if a.unit is not None]
        return len(units) != len(set(units))

    @property
    def numeric_id(self) -> int:
        """``Q07`` -> 7; used by ``--only`` range expansion. ``-1`` if unparsable."""
        match = re.search(r"(\d+)", self.id)
        return int(match.group(1)) if match else -1


@dataclass(frozen=True)
class EvalDataset:
    """A loaded question set, with the provenance a run record must carry."""

    path: Path
    sha256: str
    columns: tuple[str, ...]
    items: tuple[EvalItem, ...]
    legend_path: Path | None = None
    rubric_path: Path | None = None

    @property
    def n_items(self) -> int:
        return len(self.items)

    def describe(self) -> dict:
        """The ``dataset`` block of ``results.json``."""
        return {
            "path": self.path.as_posix(),
            "sha256": self.sha256,
            "rows": self.n_items,
            "columns": list(self.columns),
            "legend_path": self.legend_path.as_posix() if self.legend_path else None,
            "rubric_path": self.rubric_path.as_posix() if self.rubric_path else None,
        }


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def legend_path_for(dataset_path: Path) -> Path | None:
    """``…-domande.csv`` -> ``…-legenda.csv`` when the sibling exists.

    A set split per machine (e.g. ``fixture-alpha-domande.csv``) shares one legend
    (``eval-chatbot-legenda.csv``), so shorter hyphen-prefixes of the stem are
    tried after the exact one.
    """
    stem = dataset_path.name.removesuffix("-domande.csv")
    if stem == dataset_path.name:
        return None
    parts = stem.split("-")
    for end in range(len(parts), 0, -1):
        candidate = dataset_path.with_name("-".join(parts[:end]) + "-legenda.csv")
        if candidate.is_file():
            return candidate
    return None


def rubric_path_for(dataset_path: Path) -> Path | None:
    """``…-domande.csv`` -> ``derived/…-rubric.csv`` when it exists (plan §8 Q2)."""
    name = dataset_path.name.replace("-domande.csv", "-rubric.csv")
    candidate = dataset_path.parent / "derived" / name
    return candidate if candidate.is_file() else None


def load_rubric(path: Path) -> dict[str, dict[str, str]]:
    """Read the derived rubric sidecar, keyed by question ID."""
    rubric: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return rubric
        lookup = {normalise_header(name): name for name in reader.fieldnames}
        if "id" not in lookup:
            raise DatasetError(f"{path}: rubric is missing an 'ID' column")
        for row in reader:
            item_id = (row.get(lookup["id"]) or "").strip()
            if not item_id:
                continue
            rubric[item_id] = {
                "required": (row.get(lookup.get("required", ""), "") or "").strip(),
                "forbidden": (row.get(lookup.get("forbidden", ""), "") or "").strip(),
                "note": (row.get(lookup.get("note", ""), "") or "").strip(),
            }
    return rubric


def _build_item(row: dict[str, str], get: dict[str, str], rubric: dict[str, dict[str, str]]) -> EvalItem:
    def cell(column: str) -> str:
        source = get.get(normalise_header(column))
        return (row.get(source, "") if source else "").strip()

    domanda = cell(COL_DOMANDA)
    categoria = cell(COL_CATEGORIA)
    risposta_attesa = cell(COL_RISPOSTA_ATTESA)
    serial_match = _SERIAL_RE.search(domanda)
    serial = serial_match.group(0) if serial_match else None

    difficolta_raw = cell(COL_DIFFICOLTA)
    try:
        difficolta = int(difficolta_raw) if difficolta_raw else None
    except ValueError:
        difficolta = None

    item_id = cell(COL_ID)
    entry = rubric.get(item_id, {})

    return EvalItem(
        id=item_id,
        categoria=categoria,
        difficolta=difficolta,
        domanda=domanda,
        risposta_attesa=risposta_attesa,
        comportamento_atteso=cell(COL_COMPORTAMENTO),
        capitolo_fonte=cell(COL_CAPITOLO),
        pag_stampata=cell(COL_PAGINA),
        documento=cell(COL_DOCUMENTO),
        raw=dict(row),
        serial=serial,
        language=detect_language(domanda),
        expects_refusal=categoria == CATEGORIA_REFUSAL,
        expects_premise_correction=categoria == CATEGORIA_PREMISE,
        expects_disambiguation=categoria in CATEGORIA_DISAMBIGUATION,
        gold_pages=parse_pages(cell(COL_PAGINA)),
        value_anchors=tuple(extract_value_anchors(risposta_attesa, serial=serial)),
        required=tuple(_split_patterns(entry.get("required", ""))),
        forbidden=tuple(_split_patterns(entry.get("forbidden", ""))),
        rubric_note=entry.get("note", ""),
    )


def load_dataset(path: str | Path, *, rubric: bool = True) -> EvalDataset:
    """Read one ``*-domande.csv`` question set (plus its legend/rubric siblings)."""
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise DatasetError(f"dataset not found: {dataset_path}")

    with dataset_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise DatasetError(f"{dataset_path}: no header row")
        columns = tuple(name.replace("﻿", "") for name in reader.fieldnames)
        lookup = {normalise_header(name): name for name in reader.fieldnames}
        for required_column in (COL_ID, COL_DOMANDA, COL_RISPOSTA_ATTESA):
            if normalise_header(required_column) not in lookup:
                raise DatasetError(f"{dataset_path}: missing required column {required_column!r}")
        rows = [row for row in reader if (row.get(lookup[normalise_header(COL_ID)]) or "").strip()]

    rubric_file = rubric_path_for(dataset_path) if rubric else None
    rubric_data = load_rubric(rubric_file) if rubric_file else {}

    items = tuple(_build_item(row, lookup, rubric_data) for row in rows)
    return EvalDataset(
        path=dataset_path,
        sha256=sha256_of(dataset_path),
        columns=columns,
        items=items,
        legend_path=legend_path_for(dataset_path),
        rubric_path=rubric_file,
    )


def discover_datasets(root: str | Path | None = None) -> list[Path]:
    """Every ``*-domande.csv`` under *root* (default ``$EVAL_DATASETS_DIR``) — the issue anticipates more than one."""
    return sorted(Path(root if root is not None else config.EVAL_DATASETS_DIR).glob("*-domande.csv"))


class NoMatchError(DatasetError):
    """``--only`` selected nothing from one question set.

    A subclass so the CLI can tell "this file has none of the requested IDs"
    (fine when several files are run at once) from a genuinely broken dataset.
    """


_ONLY_SEPARATOR = re.compile(r"[,\s]+")


def _read_only_file(path_text: str) -> list[str]:
    """Tokens of a ``--only @FILE`` ID list: IDs or ranges, comma/space/newline separated.

    ``#`` starts a comment that runs to the end of the line, so a committed list
    can say *why* each question is in it.
    """
    if not path_text:
        raise DatasetError("--only '@' needs a file name, e.g. --only @data/eval/datasets/derived/smoke-ids.txt")
    try:
        text = Path(path_text).read_text(encoding="utf-8")
    except OSError as exc:
        raise DatasetError(f"--only @{path_text}: cannot read the file ({exc.strerror or exc})") from exc
    return [
        chunk
        for line in text.splitlines()
        for chunk in _ONLY_SEPARATOR.split(line.split("#", 1)[0].strip())
        if chunk
    ]


def parse_only(spec: str, items: tuple[EvalItem, ...]) -> list[EvalItem]:
    """Filter items by a ``--only`` spec such as ``Q01,Q05-Q08`` or ``@FILE``.

    Ranges are expanded over the numeric part of the ID, so ``Q05-Q08`` selects
    every item whose number falls in ``[5, 8]`` regardless of how many exist.
    A token starting with ``@`` names a file of IDs and ranges (see
    ``_read_only_file``); it may be mixed with literal IDs.
    """
    tokens: list[str] = []
    for chunk in _ONLY_SEPARATOR.split(spec.strip()):
        if chunk.startswith("@"):
            tokens.extend(_read_only_file(chunk[1:]))
        elif chunk:
            tokens.append(chunk)

    wanted_ids: set[str] = set()
    ranges: list[tuple[int, int]] = []
    for chunk in tokens:
        span = re.fullmatch(r"([A-Za-z]*)(\d+)\s*-\s*([A-Za-z]*)(\d+)", chunk)
        if span:
            ranges.append((int(span.group(2)), int(span.group(4))))
        else:
            wanted_ids.add(chunk.strip().upper())

    selected = [
        item
        for item in items
        if item.id.upper() in wanted_ids or any(low <= item.numeric_id <= high for low, high in ranges)
    ]
    if not selected:
        raise NoMatchError(f"--only {spec!r} matched no questions")
    return selected
