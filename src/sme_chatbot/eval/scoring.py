"""WP4/WP5 — deterministic scorers and the verdict roll-up.

Every function here is **pure over** ``(item, answer, citations)`` with no HTTP,
no Langfuse and no filesystem access, so the Langfuse harness's ``run_experiment``
evaluators can import them directly (plan §6 — the harness's single design constraint).

What each scorer can and cannot see is stated on its docstring. None of them is
"correctness": :func:`roll_up` is a roll-up, and it is allowed to say ``"?"``
(undetermined).

Text handling
-------------
All matching runs over a *folded* form of the text (:func:`fold`): lowercase,
accents stripped, NBSP/typographic dashes normalised. The fold is deliberately
**length-preserving** (one character in, one character out), so offsets found in
the folded text slice the original text verbatim — that is what lets the report
quote an anchor exactly as the answer key wrote it.

Numbers
-------
``Risposta attesa`` is Italian technical prose, so:

- ``0,37`` is a decimal, ``12.000`` is twelve thousand, and a dotted number that
  is *not* a ``\\d{1,3}(\\.\\d{3})+`` group (``11.5.3``, ``9.5``, ``8.8``) is a
  chapter reference, not a value. Chapter references are dropped — without that
  filter Q14/Q18/Q19/Q26 score against cross-references and the metric is
  meaningless (plan §1.1).
- A number glued to letters or hyphens (``TC-200``, ``M24``, ``A2-70``,
  ``4-2-1``, ``2006/42``) is part of a code, not a standalone value, and is
  dropped. Codes that matter as answer content are pinned in the rubric
  sidecar's ``required`` column instead.
- A number introduced by ``p.``/``pag.``/``cap.``/``art.``/... is a reference,
  and the item's own ``matricola`` is the question echoed back; both are dropped.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only, no runtime import cycle
    from sme_chatbot.eval.dataset import EvalItem

# ---------------------------------------------------------------------------
# Text folding
# ---------------------------------------------------------------------------

# 1:1 character replacements applied before folding, so the result stays the
# same length as the input (see the module docstring).
_TRANSLATION = str.maketrans(
    {
        " ": " ",  # NBSP
        " ": " ",  # figure space
        " ": " ",  # thin space
        " ": " ",  # narrow NBSP
        "–": "-",  # en dash
        "—": "-",  # em dash
        "−": "-",  # minus sign
        "’": "'",  # right single quote
        "“": '"',
        "”": '"',
    }
)


def _fold_char(ch: str) -> str:
    """Fold one character to exactly one character (lowercase, accent-free)."""
    decomposed = unicodedata.normalize("NFD", ch)
    base = "".join(c for c in decomposed if not unicodedata.combining(c)) or ch
    lowered = base.lower()
    return lowered[0] if lowered else ch


def fold(text: str) -> str:
    """Lowercase + strip accents + normalise spaces/dashes, preserving length.

    Length preservation is a contract, not an accident: :func:`extract_value_anchors`
    slices the *original* string with offsets found in the folded one.
    """
    translated = text.translate(_TRANSLATION)
    return "".join(_fold_char(c) for c in translated)


def squash(text: str) -> str:
    """Fold, collapse whitespace, and unify ``e'`` with ``è`` — for phrase matching.

    Italian text reaches us both accented (``non è presente``) and typed flat
    (``non e' presente``); folding turns the first into ``non e presente``, and
    the ``e'`` rule turns the second into the same thing, so one lexicon entry
    covers both. Only the standalone token is rewritten, so elisions that carry
    the apostrophe on another word (``un'altra``, ``po'``) are untouched.

    Length is *not* preserved here — this form is for substring matching only,
    never for slicing the original text.
    """
    folded = re.sub(r"\s+", " ", fold(text)).strip()
    return re.sub(r"\be'(?=\s|\Z)", "e", folded)


# ---------------------------------------------------------------------------
# Number tokenisation
# ---------------------------------------------------------------------------

# A numeric run, rejected when glued to a letter, digit, underscore, hyphen or
# slash on either side (so `TC-200`, `A2-70`, `2006/42` and `90001_manual` yield
# nothing). The first alternative catches space-grouped thousands (`12 000`).
_NUMBER_TOKEN_RE = re.compile(
    r"(?<![0-9a-z_/-])"
    r"(?P<tok>[0-9]{1,3}(?: [0-9]{3})+(?![0-9])|[0-9](?:[0-9.,/-]*[0-9])?)"
    r"(?![0-9a-z_])"
)

_DECIMAL_RE = re.compile(r"\d+(?:\.\d{3})*,\d+\Z")
_THOUSANDS_RE = re.compile(r"\d{1,3}(?:[. ]\d{3})+\Z")
_INTEGER_RE = re.compile(r"\d+\Z")

# Words that turn the number after them into a cross-reference, not a value.
_REF_PREFIX_RE = re.compile(
    r"(?:^|[^a-z])"
    r"(?:cap|capitolo|capitoli|par|paragrafo|paragrafi|p|pp|pag|pagg|pagina|pagine|page|pages"
    r"|art|articolo|articoli|punto|punti|vedi|cfr|tab|tabella|tabelle|fig|figura|figure"
    r"|nota|note|rev|revisione|chapter|section|sez|sezione"
    r"|rimando a|rimandi a|rimanda a|riferimento|rif)"
    r"\.?\s*\Z"
)

# Unit alias -> canonical unit key. Deliberately tight: a wide vocabulary turns
# unrelated words into "units" and silently changes what a number means.
_UNIT_ALIASES: dict[str, str] = {
    "mm": "mm",
    "millimetri": "mm",
    "millimetro": "mm",
    "cm": "cm",
    "centimetri": "cm",
    "centimetro": "cm",
    "m": "m",
    "metri": "m",
    "metro": "m",
    "kg": "kg",
    "chilogrammi": "kg",
    "kilogrammi": "kg",
    "bar": "bar",
    "nm": "nm",
    "n.m": "nm",
    "n m": "nm",
    "newton metri": "nm",
    "newton-metri": "nm",
    "newtonmetri": "nm",
    "newton metro": "nm",
    "kw": "kw",
    "kilowatt": "kw",
    "chilowatt": "kw",
    "l/min": "l/min",
    "lt/min": "l/min",
    "litri/min": "l/min",
    "l/minuto": "l/min",
    "litri al minuto": "l/min",
    "°c": "°c",
    "° c": "°c",
    "gradi centigradi": "°c",
    "gradi celsius": "°c",
    "gradi": "°c",
    "c": "°c",
    "%": "%",
    "per cento": "%",
    "percento": "%",
    "ore": "h",
    "ora": "h",
    "h": "h",
    "hours": "h",
    "hour": "h",
    "hrs": "h",
    "bph": "bph",
    "pezzi/ora": "bph",
    "pezzi/h": "bph",
    "pz/h": "bph",
    "bottiglie/ora": "bph",
    "pezzi ora": "bph",
    "lux": "lux",
}

_UNIT_RE = re.compile(
    r"[ \t]*(?P<unit>"
    + "|".join(re.escape(a) for a in sorted(_UNIT_ALIASES, key=len, reverse=True))
    + r")(?![0-9a-z])"
)


_DOT_DECIMAL_RE = re.compile(r"\d+\.\d+\Z")


def _decimal_string(plain: str) -> str | None:
    try:
        return format(Decimal(plain).normalize(), "f")
    except InvalidOperation:  # pragma: no cover - unreachable for the patterns above
        return None


def _canonical_numbers(token: str, *, strict: bool) -> list[str]:
    """Every plausible numeric reading of *token*; empty if it is not a value.

    Empty means "this run of digits is a chapter reference, a range or a code" —
    see the module docstring for why that distinction carries the whole metric.

    ``strict`` is the asymmetry that makes the metric both safe and usable:

    - On the **answer key** (``strict=True``) a dotted number is a chapter
      reference unless it is an unambiguous thousands group. Being wrong here
      invents an anchor out of ``cap. 9.5``, which is the failure mode the whole
      filter exists to prevent.
    - In an **answer** (``strict=False``) the same token is also read as a
      decimal, because a model writing Italian values in English notation
      (``0.37 kW`` for ``0,37 kW``) must still match. Being generous here can at
      worst match an anchor that is genuinely present in another notation; it
      cannot invent an expectation.
    """
    if "/" in token or "-" in token:
        return []  # code (`2006/42`) or range/compound (`4-2-1`)

    readings: list[str] = []

    def add(plain: str) -> None:
        value = _decimal_string(plain)
        if value is not None and value not in readings:
            readings.append(value)

    if _DECIMAL_RE.match(token):  # italian decimal comma, e.g. `0,37` / `12.000,50`
        add(token.replace(".", "").replace(",", "."))
    elif _THOUSANDS_RE.match(token):  # `12.000` / `12 000`
        add(token.replace(".", "").replace(" ", ""))
        # `1.500` is twelve-hundred-style thousands in Italian but a decimal in
        # English. Only the answer side gets the second reading.
        if not strict and _DOT_DECIMAL_RE.match(token):
            add(token)
    elif _INTEGER_RE.match(token) or (not strict and _DOT_DECIMAL_RE.match(token)):
        # A plain integer, or — answers only — a decimal in English notation
        # (`0.37` where the key wrote `0,37`). The answer key never takes the
        # second half of that branch, which is what keeps `cap. 9.5` from
        # becoming an anchor.
        add(token)
    return readings


# ---------------------------------------------------------------------------
# Anchors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Anchor:
    """One machine-checkable fragment of ``Risposta attesa``.

    ``kind="value"`` — a number (with optional unit) extracted automatically.
    ``kind="required"`` — a literal/stem pattern pinned by the rubric sidecar.
    """

    kind: str
    text: str  # verbatim source span (value) or the rubric pattern (required)
    value: str | None = None  # canonical number, e.g. "0.37"
    unit: str | None = None  # canonical unit key, e.g. "nm"

    @property
    def key(self) -> tuple[str, str | None, str | None]:
        return (self.kind, self.value if self.kind == "value" else self.text, self.unit)

    def describe(self) -> str:
        return self.text.strip()


@dataclass(frozen=True)
class NumberToken:
    """A number found in an answer, with the unit written next to it (if any)."""

    value: str
    unit: str | None


def _unit_at(folded: str, pos: int) -> tuple[str | None, int]:
    """Canonical unit starting at *pos* in *folded*, and where it ends."""
    unit_match = _UNIT_RE.match(folded, pos)
    if unit_match is None:
        return None, pos
    return _UNIT_ALIASES[unit_match.group("unit")], unit_match.end()


def find_numbers(text: str) -> list[NumberToken]:
    """Every standalone number in *text*, with its adjacent unit.

    Chapter references and code-embedded digits are excluded by the same rules
    used on the answer key, so ``cap. 9.5`` in an answer is not a value.
    """
    folded = fold(text)
    out: list[NumberToken] = []
    for match in _NUMBER_TOKEN_RE.finditer(folded):
        readings = _canonical_numbers(match.group("tok"), strict=False)
        if not readings:
            continue
        unit, _ = _unit_at(folded, match.end())
        out.extend(NumberToken(value=value, unit=unit) for value in readings)
    return out


def extract_value_anchors(expected: str, *, serial: str | None = None) -> list[Anchor]:
    """Value anchors from ``Risposta attesa`` (WP4 item 1).

    Drops chapter/page/article references, code-embedded digits and the item's
    own ``matricola``, then de-duplicates on ``(value, unit)`` — Q32 lists
    ``0,37 kW`` three times for three motors, and "did the answer contain 0.37"
    is one check, not three.
    """
    folded = fold(expected)
    anchors: list[Anchor] = []
    seen: set[tuple[str, str | None, str | None]] = set()
    for match in _NUMBER_TOKEN_RE.finditer(folded):
        readings = _canonical_numbers(match.group("tok"), strict=True)
        if not readings:
            continue
        value = readings[0]
        if serial and value == serial:
            continue
        if _REF_PREFIX_RE.search(folded[: match.start()]):
            continue
        unit, end = _unit_at(folded, match.end())
        anchor = Anchor(kind="value", text=expected[match.start() : end], value=value, unit=unit)
        if anchor.key in seen:
            continue
        seen.add(anchor.key)
        anchors.append(anchor)
    return anchors


def required_anchors(patterns: list[str]) -> list[Anchor]:
    """Rubric ``required`` entries as anchors (one per ``|``-separated entry)."""
    return [Anchor(kind="required", text=p.strip()) for p in patterns if p.strip()]


# ---------------------------------------------------------------------------
# Rubric pattern matching (`required` / `forbidden`)
# ---------------------------------------------------------------------------


def match_pattern(pattern: str, text: str) -> str | None:
    """Match one rubric pattern against *text*; return the matched span or ``None``.

    Syntax, kept to two operators so the sidecar stays reviewable at a glance:

    - ``~`` separates alternatives — ``proibit*~vietat*`` matches either.
    - a trailing ``*`` makes the alternative a stem — ``sversament*`` matches
      ``sversamenti``/``sversamento``.

    Matching is over the folded, whitespace-collapsed text, with a word boundary
    at each end that is a word character (so ``orario`` does not match inside
    ``antiorario``, and ``232`` does not match inside ``1232``).
    """
    haystack = squash(text)
    for raw_alternative in pattern.split("~"):
        alternative = raw_alternative.strip()
        if not alternative:
            continue
        stem = alternative.endswith("*")
        needle = squash(alternative[:-1] if stem else alternative)
        if not needle:
            continue
        left = r"\b" if needle[0].isalnum() else ""
        # A stem swallows the rest of the word, so the report quotes the word
        # that actually matched ("sversamenti"), not the stem it was cut to.
        right = r"\w*" if stem else (r"\b" if needle[-1].isalnum() else "")
        found = re.search(left + re.escape(needle) + right, haystack)
        if found:
            return found.group(0)
    return None


def anchor_matches(anchor: Anchor, answer: str, numbers: list[NumberToken] | None = None) -> str | None:
    """Is *anchor* present in *answer*? Return the matched span, else ``None``.

    For a value anchor the number must match on numeric boundaries (``92`` is
    not found inside ``192``) and the unit must not *contradict*: an answer that
    writes ``8 ore`` does not satisfy an ``8 bar`` anchor, while one that writes
    a bare ``61`` still satisfies ``61 Nm`` — a missing or unrecognised unit
    is treated as unknown, not as a mismatch, because the alternative is a wall
    of false negatives on perfectly good prose.
    """
    if anchor.kind == "required":
        return match_pattern(anchor.text, answer)
    tokens = find_numbers(answer) if numbers is None else numbers
    for token in tokens:
        if token.value != anchor.value:
            continue
        if anchor.unit is not None and token.unit is not None and token.unit != anchor.unit:
            continue
        return f"{token.value} {token.unit}" if token.unit else token.value
    return None


# ---------------------------------------------------------------------------
# Lexicons (WP4 item 3)
# ---------------------------------------------------------------------------

# Phrases are written already folded (lowercase, accent-free) because that is
# the form `squash()` produces: "non è presente" -> "non e presente".
_REFUSAL_PHRASES = [
    # Italian
    "non e presente",
    "non presente",
    "non e riportato",
    "non riportato",
    "non e riportata",
    "non e indicato",
    "non indicato",
    "non e indicata",
    "non viene indicato",
    "non viene riportato",
    "non e specificato",
    "non specificato",
    "non e specificata",
    "non specifica",
    "non indica",
    "non riporta",
    "non risulta",
    "non e applicabile",
    "non applicabile",
    "non e disponibile",
    "non disponibile",
    "non e documentato",
    "non contiene",
    "non compare",
    "non esiste",
    "non fornisce",
    "non copre",
    "non trovo",
    "non ho trovato",
    "non sono riuscito a trovare",
    "non e possibile determinare",
    "nessun riferimento",
    "nessuna informazione",
    "nessun dato",
    "non e reperibile",
    "non reperibile",
    "non e menzionato",
    "non menzionato",
    # gpt-5's phrasings of a correct refusal, missed until the gpt-5 re-runs:
    # "non ci sono istruzioni per una TC-100", "i documenti non la coprono",
    # "non ne riportano le istruzioni"
    "non ci sono",
    "non coprono",
    "non la coprono",
    "non lo coprono",
    "non ne riporta",
    # qwen3.8 with the refusal prompt nudges, on the wrong-component items (Q82/Q83):
    # "la 90001 non monta teste di chiusura TC-100"
    "non monta",
    # English
    "not present",
    "is not specified",
    "not specified",
    "does not specify",
    "does not contain",
    "does not provide",
    "does not report",
    "not available",
    "not reported",
    "no information",
    "not found",
    "cannot find",
    "could not find",
    "unable to find",
    "not documented",
    "no mention",
    "not mentioned",
    "not covered",
    "does not cover",
    "no reference",
]

_CLARIFICATION_PHRASES = [
    "quale materiale",
    "quale macchina",
    "quale matricola",
    "quale modello",
    "quale classe",
    "quale tipo",
    "di quale macchina",
    "di quale materiale",
    "puoi specificare",
    "puo specificare",
    "potresti specificare",
    "puoi indicare",
    "puo indicare",
    "potresti indicare",
    "puoi precisare",
    "per favore specifica",
    "specificare la matricola",
    "indicare la matricola",
    "which machine",
    "which material",
    "which model",
    "which serial",
    "could you specify",
    "can you specify",
    "please specify",
    "please provide",
]

_QUALIFICATION_PHRASES = [
    "dipende dal materiale",
    "dipende dalla macchina",
    "dipende dal modello",
    "a seconda del materiale",
    "a seconda della macchina",
    "in base al materiale",
    "acciaio zincato",
    "acciaio inox",
    "aisi 304",
    "a2-70",
    "classe 8.8",
    "depends on the material",
    "depends on the machine",
]

_PREMISE_CORRECTION_PHRASES = [
    "la premessa",
    "correggere una premessa",
    "premessa errata",
    "premessa non corretta",
    "non e corretto",
    "non e corretta",
    "non e esatto",
    "non e esatta",
    "e errata",
    "e errato",
    "e sbagliato",
    "e sbagliata",
    "in realta",
    "al contrario",
    "contrariamente",
    "va corretto",
    "va corretta",
    "correggo",
    "una precisazione",
    "that is not correct",
    "this is not correct",
    "is incorrect",
    "is not accurate",
    "on the contrary",
]


def _find_phrase(phrases: list[str], text: str) -> str | None:
    haystack = squash(text)
    for phrase in phrases:
        if phrase in haystack:
            return phrase
    return None


# ---------------------------------------------------------------------------
# Citations
# ---------------------------------------------------------------------------

# `DOCUMENTS_SYSTEM_PROMPT` tells the model to cite inline as `filename, p. N`.
# The filename part deliberately allows no whitespace: a bounded negated class
# keeps this linear-time on long answers.
_INLINE_CITATION_RE = re.compile(
    r"(?P<file>[^\s,;()\[\]]{1,120}\.(?:pdf|docx?|xlsx?))"
    r"\s*[,;:]?\s*"
    r"(?:p|pp|pag|pagg|pagina|pagine|page|pages)\.?\s*(?P<page>\d+)",
    re.IGNORECASE,
)

_LANG_SUFFIX_RE = re.compile(r"_(it|en|de|fr|es|pt|ru|zh)(?:_|\.|\Z)", re.IGNORECASE)

_IT_STOPWORDS = {
    "il", "lo", "la", "i", "gli", "le", "di", "del", "della", "dello", "dei",
    "degli", "delle", "che", "chi", "cosa", "come", "quale", "quali", "qual",
    "quanto", "quanta", "quante", "quanti", "per", "con", "sulla", "sul",
    "sui", "sulle", "una", "uno", "un", "non", "sono", "ed", "ma", "devo",
    "deve", "posso", "dal", "dalla", "nel", "nella", "alla", "al", "ai",
    "ogni", "macchina", "manuale", "pagina", "capitolo", "secondo",
}
_EN_STOPWORDS = {
    "the", "what", "which", "who", "how", "is", "are", "was", "were", "for",
    "with", "from", "this", "that", "these", "those", "does", "do", "did",
    "can", "should", "must", "and", "or", "of", "to", "in", "on", "at",
    "machine", "manual", "page", "chapter", "minimum", "maximum", "according",
}


def detect_language(text: str) -> str:
    """``"it"`` / ``"en"`` by stopword vote; ties (and empty text) go to ``"it"``.

    Good enough for this corpus — the golden set has exactly one English item
    (Q35) — and it is reported, never used as a gate.
    """
    words = re.findall(r"[a-z']+", fold(text))
    italian = sum(1 for w in words if w in _IT_STOPWORDS)
    english = sum(1 for w in words if w in _EN_STOPWORDS)
    return "en" if english > italian else "it"


def _citation_name(citation: dict) -> str:
    return str(citation.get("filename") or citation.get("source") or "").strip()


def _document_key(name: str) -> str:
    """Comparable form of a document name: basename, folded, extension dropped."""
    base = re.split(r"[\\/]", name.strip())[-1]
    base = re.sub(r"\.(pdf|docx?|xlsx?|txt|md)\Z", "", base, flags=re.IGNORECASE)
    return squash(base)


# ---------------------------------------------------------------------------
# Scorer outputs
# ---------------------------------------------------------------------------


@dataclass
class AnchorScore:
    """``anchor_recall`` (WP4 item 1) — recall, explicitly not correctness."""

    matched: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    score: float | None = None  # None == no anchors, i.e. nothing to check

    @property
    def total(self) -> int:
        return len(self.matched) + len(self.missing)


@dataclass
class CitationScore:
    """``gold_doc_cited`` + ``citation_noise`` (WP4 items 4-5)."""

    n_citations: int = 0
    gold_document: str = ""
    gold_doc_cited: bool = False
    documents: list[str] = field(default_factory=list)
    other_documents: list[str] = field(default_factory=list)
    pages: list[int] = field(default_factory=list)
    language_suffixes: dict[str, int] = field(default_factory=dict)
    wrong_language_documents: list[str] = field(default_factory=list)


@dataclass
class PageScore:
    """``page_match_approx`` (WP4 item 6) — always "~", never a gate."""

    gold_pages: list[int] = field(default_factory=list)
    expected_pdf_pages: list[int] = field(default_factory=list)
    cited_pages: list[int] = field(default_factory=list)
    matched: list[int] = field(default_factory=list)
    offset: int = 0
    tolerance: int = 0
    score: float | None = None  # None == no printed page in the answer key


@dataclass
class ItemScores:
    """Every scorer's output for one answer, plus the WP5 roll-up."""

    anchors: AnchorScore = field(default_factory=AnchorScore)
    forbidden_hits: list[str] = field(default_factory=list)
    refusal_detected: bool = False
    refusal_span: str | None = None
    clarification_detected: bool = False
    clarification_span: str | None = None
    qualification_detected: bool = False
    qualification_span: str | None = None
    premise_correction_detected: bool = False
    premise_correction_span: str | None = None
    citations: CitationScore = field(default_factory=CitationScore)
    pages: PageScore = field(default_factory=PageScore)
    has_inline_citation: bool = False
    inline_citation_span: str | None = None
    answer_empty: bool = False
    answer_language: str = "it"
    language_match: bool = True
    verdict: str = "?"
    verdict_reasons: list[str] = field(default_factory=list)

    @property
    def disambiguation_detected(self) -> bool:
        """Asked for the missing detail, or qualified the answer by material."""
        return self.clarification_detected or self.qualification_detected


# ---------------------------------------------------------------------------
# Individual scorers
# ---------------------------------------------------------------------------


def score_anchors(item: EvalItem, answer: str) -> AnchorScore:
    """Fraction of the answer key's anchors present in *answer*.

    Honest limitation, repeated in every report: this is **recall**, not
    correctness — an answer carrying both ``47 Nm`` and a spurious ``147 Nm``
    still scores 1.0. The rubric's ``forbidden`` column is the counterweight.
    """
    anchors = list(item.anchors)
    if not anchors:
        return AnchorScore(score=None)
    numbers = find_numbers(answer)
    matched: list[str] = []
    missing: list[str] = []
    for anchor in anchors:
        if anchor_matches(anchor, answer, numbers) is not None:
            matched.append(anchor.describe())
        else:
            missing.append(anchor.describe())
    return AnchorScore(matched=matched, missing=missing, score=len(matched) / len(anchors))


def score_forbidden(item: EvalItem, answer: str) -> list[str]:
    """Rubric denylist hits (WP4 item 2) — the counterweight to recall."""
    hits: list[str] = []
    for pattern in item.forbidden:
        found = match_pattern(pattern, answer)
        if found is not None:
            hits.append(f"{pattern} -> {found!r}")
    return hits


def score_citations(item: EvalItem, citations: list[dict]) -> CitationScore:
    """``gold_doc_cited`` + ``citation_noise``: the direct measurement of the empty-answer-under-a-footer bug.

    ``citation_noise`` has no expected value — it is tracked, never gated: the
    distinct-document count and the ``_IT``/``_DE``/``_EN`` breakdown are what
    over-citation and cross-language contamination look like in numbers.
    """
    score = CitationScore(n_citations=len(citations), gold_document=item.gold_document)
    gold_key = _document_key(item.gold_document)
    gold_lang = _LANG_SUFFIX_RE.search(item.gold_document)
    gold_lang_code = gold_lang.group(1).lower() if gold_lang else None

    for citation in citations:
        name = _citation_name(citation)
        if name:
            key = _document_key(name)
            if key not in score.documents:
                score.documents.append(key)
            if key == gold_key:
                score.gold_doc_cited = True
            elif key not in score.other_documents:
                score.other_documents.append(key)
            lang = _LANG_SUFFIX_RE.search(name)
            if lang:
                code = lang.group(1).lower()
                score.language_suffixes[code] = score.language_suffixes.get(code, 0) + 1
                if gold_lang_code and code != gold_lang_code and key not in score.wrong_language_documents:
                    score.wrong_language_documents.append(key)
        page = citation.get("page")
        if isinstance(page, int) and page not in score.pages:
            score.pages.append(page)
    score.pages.sort()
    return score


def score_pages(item: EvalItem, citations: list[dict], *, offset: int, tolerance: int) -> PageScore:
    """``page_match_approx`` — printed page + ``offset`` vs cited PDF page.

    ``Pag. stampata`` is the page *printed on the paper*; ``Citation.page`` is
    the PDF page, roughly ten pages later because of front matter. This can only
    ever be approximate, so it is rendered as "~" and never gates a verdict. It
    becomes exact only if ingestion stores ``page_printed``.
    """
    gold = sorted(item.gold_pages)
    cited = sorted({c["page"] for c in citations if isinstance(c.get("page"), int)})
    score = PageScore(
        gold_pages=gold,
        expected_pdf_pages=[p + offset for p in gold],
        cited_pages=cited,
        offset=offset,
        tolerance=tolerance,
    )
    if not gold:
        return score
    for page in gold:
        target = page + offset
        if any(abs(c - target) <= tolerance for c in cited):
            score.matched.append(page)
    score.score = len(score.matched) / len(gold)
    return score


def score_inline_citation(answer: str) -> tuple[bool, str | None]:
    """``has_inline_citation`` — ``filename, p. N`` in the answer *text*.

    Deliberately **not** the same signal as ``n_citations > 0`` (assumption A6):
    the backend can return citations the model never wrote into its prose. Both
    are reported; neither is derived from the other.
    """
    match = _INLINE_CITATION_RE.search(answer)
    return (match is not None, match.group(0) if match else None)


# ---------------------------------------------------------------------------
# WP5 — verdict roll-up
# ---------------------------------------------------------------------------


def score_item(
    item: EvalItem,
    answer: str,
    citations: list[dict],
    *,
    page_offset: int = 10,
    page_tolerance: int = 2,
) -> ItemScores:
    """Run every scorer, then roll the verdict up. Pure; no I/O."""
    scores = ItemScores()
    scores.anchors = score_anchors(item, answer)
    scores.forbidden_hits = score_forbidden(item, answer)

    refusal = _find_phrase(_REFUSAL_PHRASES, answer)
    scores.refusal_detected, scores.refusal_span = refusal is not None, refusal

    clarification = _find_phrase(_CLARIFICATION_PHRASES, answer)
    if clarification is None:
        # A clarifying question the lexicon missed: the answer *ends* by asking
        # something. Quoting the user's own question mid-answer does not count.
        tail = answer.rstrip()
        if tail.endswith("?"):
            clarification = tail.splitlines()[-1][-160:].strip()
    scores.clarification_detected, scores.clarification_span = clarification is not None, clarification

    qualification = _find_phrase(_QUALIFICATION_PHRASES, answer)
    scores.qualification_detected, scores.qualification_span = qualification is not None, qualification

    premise = _find_phrase(_PREMISE_CORRECTION_PHRASES, answer)
    scores.premise_correction_detected, scores.premise_correction_span = premise is not None, premise

    scores.citations = score_citations(item, citations)
    scores.pages = score_pages(item, citations, offset=page_offset, tolerance=page_tolerance)
    scores.has_inline_citation, scores.inline_citation_span = score_inline_citation(answer)
    scores.answer_empty = not answer.strip()
    scores.answer_language = detect_language(answer)
    scores.language_match = scores.answer_language == item.language

    scores.verdict, scores.verdict_reasons = roll_up(item, scores)
    return scores


def _behavioural_failures(item: EvalItem, scores: ItemScores) -> list[str]:
    """Which behavioural expectations for this item's category were missed."""
    failures: list[str] = []
    if item.expects_refusal and not scores.refusal_detected:
        failures.append("refusal expected (categoria 'Non rispondibile') but none detected")
    if item.expects_premise_correction and not scores.premise_correction_detected:
        failures.append("false premise not corrected (categoria 'Premessa falsa')")
    if item.requires_qualification:
        # The legend accepts either behaviour: ask which material/machine, or
        # give every qualified value. Full anchor recall *is* the second one.
        answered_fully = scores.anchors.score is not None and scores.anchors.score >= 1.0
        if not (scores.disambiguation_detected or answered_fully):
            failures.append("answer neither qualified nor followed by a clarifying question")
    return failures


def roll_up(item: EvalItem, scores: ItemScores) -> tuple[str, list[str]]:
    """Map the scorers onto the golden set's own vocabulary (WP5).

    ``S`` correct · ``P`` partial · ``N`` wrong · ``?`` undetermined. ``ERROR``
    is set by the runner for transport failures and never reaches here.

    Order matters, and each branch states the evidence it rests on so a reviewer
    can overrule it from ``report.md`` alone:

    1. A rubric ``forbidden`` hit is decisive — that is what the denylist is for.
    2. A missed refusal is ``N`` (plan WP5); any other missed behavioural
       expectation is a downgrade, not an automatic failure.
    3. With no anchors at all, a behavioural expectation *is* the whole answer
       key (Q29 "Non presente nel manuale.", Q34 "chiedi di quale macchina"), so
       the behavioural check decides. With no anchors and no behavioural
       expectation, the item is ``?`` — that is the honest answer, not a guess.
    """
    reasons: list[str] = []
    failures = _behavioural_failures(item, scores)
    recall = scores.anchors.score

    if scores.answer_empty:
        # A distinct failure mode from a wrong answer, and one the runner would
        # otherwise report as a bare "anchor recall 0/N": the tool-calling loop
        # returned citations but never generated any text.
        # It is still `N` - nothing was answered - but the report must say which.
        reasons.append(
            f"answer was empty - no text generated, but {scores.citations.n_citations} citation(s) returned"
        )
        return "N", reasons

    if scores.forbidden_hits:
        reasons.append("forbidden content present: " + "; ".join(scores.forbidden_hits))
        return "N", reasons

    if item.expects_refusal and not scores.refusal_detected:
        reasons.append("refusal expected (categoria 'Non rispondibile') but none detected")
        return "N", reasons

    if recall is None:
        if failures:
            reasons.extend(failures)
            return "N", reasons
        if item.expects_refusal:
            reasons.append(f"refusal detected ({scores.refusal_span!r}); the answer key expects exactly that")
            return "S", reasons
        if item.expects_premise_correction or item.requires_qualification:
            reasons.append("behavioural expectation met; answer key holds no machine-checkable value")
            return "S", reasons
        reasons.append("no value anchors and no rubric entry - not machine-checkable")
        return "?", reasons

    reasons.append(
        f"anchor recall {len(scores.anchors.matched)}/{scores.anchors.total}"
        + (f" (missing: {', '.join(scores.anchors.missing)})" if scores.anchors.missing else "")
    )

    if item.expects_refusal:
        # Anchors on a refusal item are the *justification* ("le tabelle si
        # fermano a M24"), never the answer itself: they can downgrade S to P,
        # never to N. A correct refusal phrased without the justification is
        # partial, not wrong.
        reasons.insert(0, f"refusal detected ({scores.refusal_span!r})")
        return ("S", reasons) if recall >= 1.0 else ("P", reasons)

    if failures:
        reasons.extend(failures)
        return ("P", reasons) if recall > 0 else ("N", reasons)
    if recall >= 1.0:
        return "S", reasons
    if recall > 0:
        return "P", reasons
    return "N", reasons


def sources_correct(scores: ItemScores) -> str:
    """The golden set's ``Fonti corrette? (S/N)`` column.

    The legend is explicit that this is ``N`` "anche quando la risposta è giusta
    ma le fonti citate sono troppe, sbagliate o in lingua diversa dalla domanda"
    — so citing the right document *plus* three others is ``N``.
    """
    if not scores.citations.gold_doc_cited:
        return "N"
    if scores.citations.other_documents:
        return "N"
    return "S"
