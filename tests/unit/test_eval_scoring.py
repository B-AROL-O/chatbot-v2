"""WP4/WP5 — the scorers and the verdict roll-up.

The risk of this whole tool lives here (plan §4 WP10): a scorer that looks
authoritative while matching chapter numbers, or one that misses ``47 Nm``
because the answer wrote ``47`` without a unit, produces a report that is worse
than no report. So the cases below are deliberately adversarial, and several follow the
phrasing of real golden-set items (the ``datasets/*-domande.csv`` sets in the
private eval repo). The values, models and machine details are made up.
"""

from __future__ import annotations

import pytest

from sme_chatbot.eval.dataset import EvalItem
from sme_chatbot.eval.scoring import (
    Anchor,
    anchor_matches,
    detect_language,
    extract_value_anchors,
    find_numbers,
    fold,
    match_pattern,
    roll_up,
    score_anchors,
    score_citations,
    score_inline_citation,
    score_item,
    score_pages,
    sources_correct,
    squash,
)


def make_item(**overrides) -> EvalItem:
    """An EvalItem with harmless defaults; override only what a case is about."""
    defaults = dict(
        id="QXX",
        categoria="Dato tecnico semplice",
        difficolta=1,
        domanda="Domanda di prova sulla 90001?",
        risposta_attesa="",
        comportamento_atteso="",
        capitolo_fonte="",
        pag_stampata="",
        documento="90001_manual_IT.pdf",
        serial="90001",
        language="it",
    )
    defaults.update(overrides)
    expected = defaults.get("risposta_attesa", "")
    if "value_anchors" not in defaults:
        defaults["value_anchors"] = tuple(extract_value_anchors(expected, serial=defaults.get("serial")))
    return EvalItem(**defaults)


# ---------------------------------------------------------------------------
# Text folding
# ---------------------------------------------------------------------------


def test_fold_is_length_preserving():
    """Offsets found in folded text slice the original verbatim — a contract."""
    for text in ("Ogni 40 ore", "±5% è la tolleranza", "12 000 BPH", "café — 3 teste", "20 °C"):
        assert len(fold(text)) == len(text), text


def test_fold_strips_accents_and_case():
    assert fold("Non È PRESENTE") == "non e presente"


def test_squash_unifies_the_two_ways_italian_writes_e_accented():
    assert squash("Il dato non è presente") == squash("Il dato non e' presente")
    # ...without touching elisions that carry the apostrophe on another word.
    assert squash("un'altra cosa") == "un'altra cosa"


# ---------------------------------------------------------------------------
# Number extraction: the chapter-reference / code / matricola filters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # values
        ("9000 pezzi/ora", ["9000"]),
        ("0,55 kW", ["0.55"]),
        ("2,20 kW", ["2.2"]),  # trailing zero normalised, so 1,50 == 1.5 == 1.50
        ("9.000 bph", ["9000"]),  # italian thousands separator
        ("9 000 BPH", ["9000"]),  # space-grouped thousands
        ("±5%", ["5"]),
        ("tra 5 °C e 40 °C", ["5", "40"]),
        # chapter references
        ("rimando a 11.5.3", []),
        ("cap. 9.5", []),
        ("(cap. 11.6.6.2, p. 164)", []),
        ("9.1.1", []),
        # page / article references
        ("p. 118", []),
        ("pag. 37", []),
        ("art. 2 punto g)", []),
        # codes
        ("teste TC-200 e pinze XP770-M", []),
        ("acciaio AISI 304 A2-70", ["304"]),  # `304` stands alone; `A2-70` does not
        ("grado ISO 4-2-1 della norma ISO 8573", ["8573"]),
        ("Direttiva 2006/42/CE", []),
        ("le tabelle si fermano a M24", []),
        ("90001_manual_IT.pdf", []),
    ],
)
def test_value_anchor_filters(text, expected):
    assert [a.value for a in extract_value_anchors(text)] == expected


def test_the_items_own_matricola_is_not_an_anchor():
    text = "Non applicabile: la 90001 monta teste TC-200."
    assert extract_value_anchors(text, serial="90001") == []
    # ...but the same number is a real value for an item that does not own it.
    assert [a.value for a in extract_value_anchors(text, serial=None)] == ["90001"]


def test_anchors_carry_their_unit_and_verbatim_span():
    anchors = extract_value_anchors("Portata 300 l/min; pressione minima 4 bar.")
    assert [(a.value, a.unit) for a in anchors] == [("300", "l/min"), ("4", "bar")]
    assert [a.describe() for a in anchors] == ["300 l/min", "4 bar"]


def test_duplicate_values_collapse_to_one_anchor():
    text = "sollevamento 0,55 kW, orientatore 0,55 kW, rotazione 2,20 kW, nastro 0,55 kW"
    assert [a.value for a in extract_value_anchors(text)] == ["0.55", "2.2"]


# ---------------------------------------------------------------------------
# Matching: boundaries and units
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("anchor_text", "answer", "matches"),
    [
        # word boundaries
        ("47 Nm", "La coppia è 47 Nm.", True),
        ("47 Nm", "La coppia è 147 Nm.", False),
        ("7 bar", "Pressione massima 17 bar.", False),
        ("7 bar", "Pressione massima 7 bar.", True),
        # the unit must not contradict...
        ("7 bar", "Ogni 7 ore.", False),
        ("1000 h", "Ogni 1000 ore di funzionamento.", True),  # `h` and `ore` are the same unit
        # ...but a missing or unknown unit is unknown, not a mismatch
        ("61 Nm", "La coppia di serraggio è di 61.", True),
        ("61 Nm", "La coppia è 61 newton metri.", True),
        # separator normalisation in the answer
        ("9000 pezzi/ora", "Produzione: 9.000 bph.", True),
        ("9000 pezzi/ora", "Produzione: 9 000 pezzi/ora.", True),
        ("0,55 kW", "Il motore è da 0.55 kW.", True),
        ("0,55 kW", "Il motore è da 0,550 kW.", True),
        ("2,20 kW", "Rotazione teste: 2,2 kW.", True),
        # a chapter reference in the answer is not a value
        ("5%", "Vedi cap. 9.5.1", False),
    ],
)
def test_anchor_matching(anchor_text, answer, matches):
    anchor = extract_value_anchors(anchor_text)[0]
    assert (anchor_matches(anchor, answer) is not None) is matches


def test_find_numbers_reads_units_next_to_the_number():
    assert [(t.value, t.unit) for t in find_numbers("47 Nm e 8 ore")] == [("47", "nm"), ("8", "h")]


def test_an_answer_may_write_an_italian_decimal_in_english_notation():
    """`0.37` in an answer must still satisfy the key's `0,37` (plan WP4).

    The reading is deliberately one-way: the *answer key* never reads `9.5` as a
    decimal, because that is how `cap. 9.5` would become an anchor.
    """
    assert [t.value for t in find_numbers("0.55 kW")] == ["0.55"]
    assert [t.value for t in find_numbers("9.000 bph")] == ["9000", "9"]
    assert extract_value_anchors("rimando a 9.5") == []


# ---------------------------------------------------------------------------
# Rubric patterns
# ---------------------------------------------------------------------------


def test_word_boundaries_stop_the_classic_substring_traps():
    # The whole reason Q33's rubric can say `required: orario`.
    assert match_pattern("orario", "il senso di rotazione è orario") == "orario"
    assert match_pattern("orario", "la giostra ruota in senso antiorario") is None
    # ...and why `forbidden: 315` cannot fire on `1315`.
    assert match_pattern("315", "il valore è 315 Nm") == "315"
    assert match_pattern("315", "il valore è 1315 Nm") is None


def test_stem_and_alternative_operators():
    assert match_pattern("sversament*", "dopo sversamenti di prodotto") == "sversamenti"
    assert match_pattern("sversament*", "dopo uno sversamento") == "sversamento"
    assert match_pattern("proibit*~vietat*", "è vietato l'uso") == "vietato"
    assert match_pattern("proibit*~vietat*", "è consentito l'uso") is None


def test_patterns_ignore_accents_case_and_whitespace_runs():
    assert match_pattern("quasi-macchina", "È una QUASI-MACCHINA ai sensi...") is not None
    assert match_pattern("tutte le teste", "su  tutte\nle teste") is not None


# ---------------------------------------------------------------------------
# anchor_recall
# ---------------------------------------------------------------------------


def test_anchor_recall_is_none_when_there_is_nothing_to_check():
    item = make_item(risposta_attesa="Isolare tutte le fonti di energia.")
    assert score_anchors(item, "qualunque risposta").score is None


def test_anchor_recall_reports_matched_and_missing_verbatim():
    item = make_item(risposta_attesa="Portata 300 l/min; pressione minima 4 bar, massima 7 bar.")
    score = score_anchors(item, "La portata è 300 l/min e la pressione minima 4 bar.")
    assert score.matched == ["300 l/min", "4 bar"]
    assert score.missing == ["7 bar"]
    assert score.score == pytest.approx(2 / 3)


def test_recall_is_recall_not_correctness():
    """Stated limitation, asserted so it cannot regress into a silent claim."""
    item = make_item(risposta_attesa="47 Nm per acciaio zincato 8.8.")
    score = score_anchors(item, "La coppia è 47 Nm, oppure 147 Nm secondo l'altra tabella.")
    assert score.score == 1.0  # the spurious 147 is invisible here; `forbidden` is the counterweight


# ---------------------------------------------------------------------------
# Refusal / premise / clarification lexicons, on real phrasings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "answer",
    [
        "Il livello di pressione sonora non è presente nel manuale.",
        "Il dato non è riportato nella documentazione disponibile.",
        "Il manuale non specifica questo valore.",
        "Non ho trovato questa informazione; contatta il Servizio Assistenza Tecnica.",
        "Nessuna informazione disponibile su questo punto.",
        "This value is not specified in the manual.",
        "The documentation does not contain this information.",
        # gpt-5: correct refusals the lexicon used to miss (Q83, Q82, Q102)
        "Le procedure riportate sono per la TC-200; non ci sono istruzioni per una TC-100.",
        "Se è stata montata una TC-100, i documenti in libreria non la coprono.",
        "I documenti disponibili non ne riportano le istruzioni.",
        "La 90001 non monta teste di chiusura TC-100: il manuale descrive la TC-200.",
    ],
)
def test_refusal_lexicon_fires_on_real_phrasings(answer):
    item = make_item(categoria="Non rispondibile", expects_refusal=True)
    assert score_item(item, answer, []).refusal_detected


@pytest.mark.parametrize(
    "answer",
    [
        "La produzione nominale è di 9000 pezzi/ora.",
        "La coppia di serraggio è 47 Nm per acciaio zincato.",
    ],
)
def test_refusal_lexicon_does_not_fire_on_a_plain_answer(answer):
    item = make_item()
    assert not score_item(item, answer, []).refusal_detected


def test_premise_correction_detected_on_q33_phrasings():
    item = make_item(categoria="Premessa falsa", expects_premise_correction=True)
    for answer in (
        "La premessa non è corretta: il senso di rotazione è orario.",
        "In realtà la giostra ruota in senso orario.",
        "Attenzione, l'affermazione è errata: la rotazione è oraria.",
        # Q33/Q85 in the 2026-10-16 full-set run, scored N before "una premessa" was covered
        "Devo prima correggere una premessa della domanda: secondo il manuale, la giostra di chiusura "
        "della 90001 **non ruota in senso antiorario, ma in senso orario**.",
        "Il senso di rotazione indicato è una premessa non corretta: la giostra ruota in senso orario.",
    ):
        assert score_item(item, answer, []).premise_correction_detected, answer


def test_clarifying_question_is_detected_by_lexicon_or_by_ending_in_one():
    item = make_item(categoria="Scoping macchina", expects_disambiguation=True)
    assert score_item(item, "Di quale macchina si tratta?", []).clarification_detected
    assert score_item(item, "Per rispondere mi serve la matricola. Puoi indicarla?", []).clarification_detected
    # A question mark inside the text, with the answer not ending in one, is not
    # a clarifying question.
    echo = "Hai chiesto: qual è la coppia? La coppia è 47 Nm."
    assert not score_item(item, echo, []).clarification_detected


# ---------------------------------------------------------------------------
# Citations
# ---------------------------------------------------------------------------

GOLD = {"source": "90001_manual_IT.pdf", "page": 18, "filename": "90001_manual_IT.pdf"}
OTHER_DE = {"source": "20111_manual_DE.pdf", "page": 4, "filename": "20111_manual_DE.pdf"}


def test_gold_document_detection_and_noise_counting():
    item = make_item()
    score = score_citations(item, [GOLD, OTHER_DE, dict(GOLD, page=19)])
    assert score.gold_doc_cited is True
    assert score.other_documents == ["20111_manual_de"]
    assert score.n_citations == 3
    assert score.pages == [4, 18, 19]
    assert score.language_suffixes == {"it": 2, "de": 1}
    assert score.wrong_language_documents == ["20111_manual_de"]


def test_sources_correct_is_n_when_the_answer_over_cites():
    """The legend: N "anche quando la risposta è giusta ma le fonti sono troppe"."""
    item = make_item()
    assert sources_correct(score_item(item, "x", [GOLD])) == "S"
    assert sources_correct(score_item(item, "x", [GOLD, OTHER_DE])) == "N"
    assert sources_correct(score_item(item, "x", [])) == "N"


@pytest.mark.parametrize(
    ("answer", "found"),
    [
        ("La produzione è 9000 bph (90001_manual_IT.pdf, p. 18).", True),
        ("Fonte: 90001_manual_IT.pdf, pag. 18", True),
        ("Source: 90001_manual_EN.pdf, page 18", True),
        ("La produzione è 9000 bph [1].", False),
        ("Vedi il manuale a pagina 18.", False),
    ],
)
def test_inline_citation_regex(answer, found):
    assert score_inline_citation(answer)[0] is found


def test_inline_citation_is_not_the_same_signal_as_n_citations():
    """Assumption A6: the two are reported separately and never collapsed."""
    item = make_item()
    scores = score_item(item, "La produzione è 9000 bph.", [GOLD])
    assert scores.citations.n_citations == 1
    assert scores.has_inline_citation is False


# ---------------------------------------------------------------------------
# page_match_approx
# ---------------------------------------------------------------------------


def test_page_match_applies_the_offset_within_tolerance():
    item = make_item(pag_stampata="8", gold_pages=frozenset({8}))
    score = score_pages(item, [{"page": 19}], offset=10, tolerance=2)
    assert score.expected_pdf_pages == [18]
    assert score.matched == [8]
    assert score.score == 1.0


def test_page_match_misses_outside_tolerance():
    item = make_item(gold_pages=frozenset({8}))
    assert score_pages(item, [{"page": 40}], offset=10, tolerance=2).score == 0.0


def test_page_match_over_a_range():
    item = make_item(gold_pages=frozenset({108, 109}))
    score = score_pages(item, [{"page": 118}], offset=10, tolerance=2)
    assert score.matched == [108, 109]  # 118 is within ±2 of both 118 and 119


def test_page_match_is_skipped_when_the_answer_key_has_no_page():
    item = make_item(gold_pages=frozenset())
    assert score_pages(item, [{"page": 18}], offset=10, tolerance=2).score is None


def test_page_match_never_gates_a_verdict():
    """A wrong page must not turn a fully answered item into anything but S."""
    item = make_item(risposta_attesa="850 kg.", pag_stampata="8", gold_pages=frozenset({8}))
    scores = score_item(item, "La chiuditrice pesa 850 kg.", [dict(GOLD, page=999)])
    assert scores.pages.score == 0.0
    assert scores.verdict == "S"


# ---------------------------------------------------------------------------
# language
# ---------------------------------------------------------------------------


def test_language_detection():
    assert detect_language("What is the minimum sterile air pressure for machine 90001?") == "en"
    assert detect_language("Quale portata richiede la macchina 90001?") == "it"
    assert detect_language("") == "it"


def test_language_mismatch_is_tracked_not_gated():
    item = make_item(risposta_attesa="4 bar.", language="en")
    scores = score_item(item, "La pressione minima è 4 bar.", [])
    assert scores.language_match is False
    assert scores.verdict == "S"


# ---------------------------------------------------------------------------
# WP5 verdict roll-up
# ---------------------------------------------------------------------------


def test_full_recall_is_s():
    item = make_item(risposta_attesa="850 kg.")
    assert score_item(item, "Pesa 850 kg.", []).verdict == "S"


def test_partial_recall_is_p():
    item = make_item(risposta_attesa="minima 4 bar, massima 7 bar.")
    assert score_item(item, "La pressione minima è 4 bar.", []).verdict == "P"


def test_zero_recall_is_n():
    item = make_item(risposta_attesa="850 kg.")
    assert score_item(item, "Pesa 2500 kg.", []).verdict == "N"


def test_a_forbidden_hit_is_decisive():
    item = make_item(risposta_attesa="6 teste.", forbidden=("2 teste",))
    scores = score_item(item, "La macchina ha 6 teste, non 2 teste.", [])
    assert scores.anchors.score == 1.0
    assert scores.verdict == "N"
    assert "forbidden" in scores.verdict_reasons[0]


def test_q33_corrected_premise_is_s_and_accepted_premise_is_n():
    # Q33's rubric: `required: orario`, no `forbidden` - a correct answer has to repeat
    # "antiorario" to negate it, and the denylist is negation-blind.
    item = make_item(categoria="Premessa falsa", expects_premise_correction=True, required=("orario",))
    corrected = (
        "Devo prima correggere una premessa della domanda: la giostra della 90001 non ruota in senso "
        "antiorario, ma in senso orario. Il lato di ingresso delle bottiglie non è indicato."
    )
    assert score_item(item, corrected, []).verdict == "S"
    accepted = "Poiché la giostra ruota in senso antiorario, le bottiglie entrano da sinistra."
    assert score_item(item, accepted, []).verdict == "N"


def test_missing_a_required_refusal_is_n():
    item = make_item(categoria="Non rispondibile", expects_refusal=True)
    assert score_item(item, "Il livello di pressione sonora è 71 dB(A).", []).verdict == "N"


def test_a_correct_refusal_with_nothing_else_to_check_is_s():
    item = make_item(categoria="Non rispondibile", expects_refusal=True, risposta_attesa="Non presente nel manuale.")
    assert score_item(item, "Il dato non è presente nel manuale.", []).verdict == "S"


def test_on_a_refusal_item_anchors_can_only_downgrade_s_to_p():
    """Q30: a correct refusal that omits "le tabelle si fermano a M24" is partial.

    Anchors on a "Non rispondibile" row are the justification, not the answer,
    so a missing one must never make a correct refusal read as *wrong*.
    """
    item = make_item(
        categoria="Non rispondibile",
        expects_refusal=True,
        risposta_attesa="Non presente: le tabelle si fermano a M24.",
        required=("M24",),
    )
    assert score_item(item, "Il manuale non riporta la coppia per viti M30.", []).verdict == "P"
    assert score_item(item, "Non è riportata: le tabelle arrivano solo fino a M24.", []).verdict == "S"


def test_q06_one_bare_number_is_a_failure_even_when_the_number_is_right():
    """The legend calls this out explicitly for the ambiguous-table case."""
    item = make_item(
        categoria="Tabella ambigua",
        expects_disambiguation=True,
        risposta_attesa="47 Nm per acciaio zincato classe 8.8; 38 Nm per acciaio AISI 304 A2-70.",
    )
    assert item.requires_qualification is True
    bare = score_item(item, "La coppia di serraggio di una M12 è 47 Nm.", [])
    assert bare.verdict == "P"
    assert "qualified" in bare.verdict_reasons[-1]

    both = score_item(item, "47 Nm per acciaio zincato 8.8; 38 Nm per AISI 304 A2-70.", [])
    assert both.verdict == "S"

    asked = score_item(item, "Dipende dal materiale: di quale materiale è la vite?", [])
    assert asked.verdict == "N"  # asked correctly, but gave no value at all


def test_q07_is_not_penalised_for_answering_a_fully_specified_question():
    item = make_item(categoria="Tabella ambigua", expects_disambiguation=True, risposta_attesa="61 Nm.")
    assert item.requires_qualification is False
    assert score_item(item, "La coppia è 61 Nm.", []).verdict == "S"


def test_q34_scoping_is_decided_by_the_behavioural_check_alone():
    item = make_item(categoria="Scoping macchina", expects_disambiguation=True, serial=None, risposta_attesa="")
    assert score_item(item, "Di quale macchina si tratta?", []).verdict == "S"
    assert score_item(item, "La coppia di una M12 è 47 Nm.", []).verdict == "N"


def test_undetermined_when_there_is_nothing_to_check():
    item = make_item(risposta_attesa="Sintesi del ciclo di produzione, senza numeri.")
    scores = score_item(item, "L'orientatore seleziona le capsule nella tramoggia.", [])
    assert scores.verdict == "?"
    assert "not machine-checkable" in scores.verdict_reasons[0]


def test_roll_up_is_pure_and_reusable_from_a_scores_object():
    """The Langfuse harness's evaluators import these directly (plan §6) — no hidden state."""
    item = make_item(risposta_attesa="850 kg.")
    scores = score_item(item, "Pesa 850 kg.", [])
    assert roll_up(item, scores) == (scores.verdict, scores.verdict_reasons)


def test_an_empty_answer_is_n_and_says_so_in_its_own_words():
    """Observed against the real backend: 33 citations, zero answer text.

    That is the "empty answer under a citation footer" bug, and it is a
    determinate failure even on an item with nothing machine-checkable — so it
    is ``N``, not ``?``. Reporting it as a bare "anchor recall 0/1" would read
    as "the chatbot said something wrong" when it said nothing at all.
    """
    item = make_item(risposta_attesa="850 kg.")
    scores = score_item(item, "", [{"source": "a.pdf", "page": 1}])
    assert scores.verdict == "N"
    assert scores.answer_empty is True
    assert "answer was empty" in scores.verdict_reasons[0]
    assert "1 citation(s)" in scores.verdict_reasons[0]

    empty_prose = make_item(risposta_attesa="Isolare tutte le fonti di energia.")
    assert score_item(empty_prose, "   \n ", []).verdict == "N"


def test_anchor_matches_accepts_a_prebuilt_number_list():
    anchor = Anchor(kind="value", text="47 Nm", value="47", unit="nm")
    assert anchor_matches(anchor, "irrelevant", find_numbers("la coppia è 47 Nm")) == "47 nm"
