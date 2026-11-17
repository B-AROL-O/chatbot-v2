"""WP1 oracles for the eval dataset loader.

These run against ``tests/fixtures/eval/``: a synthetic set on a fictitious
machine (90001) whose rows keep the IDs and the quirks of the real golden set's
original block (Q01-Q36) that the plan's oracles are about (§4 WP1):
``Q34.serial is None``, ``Q35.language == "en"``, ``Q13.gold_pages == {108, 109}``,
``Q29.gold_pages == set()``, the ``classe 8.8`` and ``ISO 4-2-1`` anchor traps.

The real set lives in the private eval repo. The checks that only make
sense on it (row counts, families, every ``Pag. stampata`` value it contains) are
in ``test_eval_golden_set.py`` and run when that repo is checked out.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sme_chatbot.eval.dataset import (
    COL_ID,
    DatasetError,
    EvalDataset,
    NoMatchError,
    load_dataset,
    parse_only,
    parse_pages,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET = REPO_ROOT / "tests" / "fixtures" / "eval" / "fixture-alpha-domande.csv"


@pytest.fixture(scope="module")
def dataset() -> EvalDataset:
    return load_dataset(DATASET)


def _item(dataset: EvalDataset, item_id: str):
    return next(i for i in dataset.items if i.id == item_id)


# ---------------------------------------------------------------------------
# Shape and provenance
# ---------------------------------------------------------------------------


def test_loads_every_row_in_file_order(dataset):
    assert dataset.n_items == 22
    ids = [i.id for i in dataset.items]
    assert len(set(ids)) == len(ids)
    assert ids[:3] == ["Q01", "Q02", "Q03"]
    assert ids[-1] == "Q35"


def test_records_sha256_and_row_count_for_provenance(dataset):
    # A run scored against a later edit of the answer key must be detectable
    # (plan §7 R4), so the hash has to be of the file as read.
    assert len(dataset.sha256) == 64
    described = dataset.describe()
    assert described["rows"] == dataset.n_items == 22
    assert described["sha256"] == dataset.sha256
    assert described["path"].endswith("fixture-alpha-domande.csv")


def test_preserves_source_column_order_and_header_text(dataset):
    assert dataset.columns[0] == COL_ID
    # The accented header is carried verbatim - results.csv writes it back.
    assert "Difficoltà (1-3)" in dataset.columns
    assert dataset.columns[-5:] == (
        "Risposta corretta? (S/N/P)",
        "Fonti corrette? (S/N)",
        "Ha citato? (S/N)",
        "Note esito",
        "Trace Langfuse",
    )


def test_finds_the_legend_and_rubric_siblings(dataset):
    assert dataset.legend_path is not None
    # Shared by every set whose stem extends its prefix: fixture-alpha -> fixture.
    assert dataset.legend_path.name == "fixture-legenda.csv"
    # The rubric is derived, so it lives under derived/ (plan §8 Q2).
    assert dataset.rubric_path is not None
    assert dataset.rubric_path.parent.name == "derived"


def test_rubric_can_be_switched_off():
    without = load_dataset(DATASET, rubric=False)
    assert without.rubric_path is None
    assert all(not i.required and not i.forbidden for i in without.items)


def test_missing_dataset_raises_dataset_error(tmp_path):
    with pytest.raises(DatasetError):
        load_dataset(tmp_path / "nope-domande.csv")


def test_missing_required_column_raises(tmp_path):
    bad = tmp_path / "bad-domande.csv"
    bad.write_text("ID,Categoria,Domanda\nQ01,x,y\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="Risposta attesa"):
        load_dataset(bad)


# ---------------------------------------------------------------------------
# Derived fields (the plan's WP1 oracle)
# ---------------------------------------------------------------------------


def test_serial_is_extracted_and_absent_only_for_q34(dataset):
    by_id = {i.id: i for i in dataset.items}
    assert by_id["Q34"].serial is None
    assert all(i.serial == "90001" for i in dataset.items if i.id != "Q34")


def test_language_is_english_only_for_q35(dataset):
    by_id = {i.id: i for i in dataset.items}
    assert by_id["Q35"].language == "en"
    assert all(i.language == "it" for i in dataset.items if i.id != "Q35")


def test_serial_is_absent_exactly_when_scope_is_nessuna(dataset):
    """`Scope` says how the question names the machine; `serial` is what was extracted."""
    for item in dataset.items:
        scope = item.raw["Scope"]
        assert (item.serial is None) == (scope == "nessuna"), f"{item.id}: scope {scope!r}, serial {item.serial!r}"
        if item.serial is not None:
            assert item.serial == item.raw["Matricola"], item.id


def test_gold_pages_range_and_empty(dataset):
    assert set(_item(dataset, "Q13").gold_pages) == {108, 109}
    assert set(_item(dataset, "Q29").gold_pages) == set()
    assert set(_item(dataset, "Q23").gold_pages) == {29, 178}
    assert set(_item(dataset, "Q24").gold_pages) == {4, 10}
    assert set(_item(dataset, "Q01").gold_pages) == {8}


def test_behavioural_flags_follow_categoria(dataset):
    assert [i.id for i in dataset.items if i.expects_refusal] == ["Q29", "Q30", "Q31"]
    assert [i.id for i in dataset.items if i.expects_premise_correction] == ["Q33"]
    assert [i.id for i in dataset.items if i.expects_disambiguation] == ["Q06", "Q07", "Q08", "Q34"]


def test_requires_qualification_separates_q06_from_q07_q08(dataset):
    """The category flag covers two different situations; the gate must not.

    Q06 asks for an M12 torque without naming the material, so the manual's two
    tables both apply and one bare number is a failure. Q07/Q08 name the
    material in the question, so their answer key holds a single value and a
    bare number is simply correct — gating them on disambiguation would mark
    every correct answer partial.
    """
    assert _item(dataset, "Q06").requires_qualification is True
    assert _item(dataset, "Q07").requires_qualification is False
    assert _item(dataset, "Q08").requires_qualification is False
    assert _item(dataset, "Q34").requires_qualification is True


# ---------------------------------------------------------------------------
# Anchors (the §1.1 finding this whole metric rests on)
# ---------------------------------------------------------------------------

# Every row whose `Risposta attesa` contains only cross-references, codes or the
# matricola. Without the filters these score against chapter numbers and the
# metric is meaningless (plan §1.1 / WP4 item 1).
ZERO_ANCHOR_IDS = ["Q23", "Q24", "Q26", "Q28", "Q29", "Q30", "Q31", "Q33", "Q34"]


def test_cross_reference_rows_yield_no_value_anchors(dataset):
    # Q26 cites `cap. 11.6.6.2, p. 164`, Q30 a thread size, Q31 the matricola and part codes.
    for item_id in ("Q26", "Q30", "Q31"):
        item = _item(dataset, item_id)
        assert item.value_anchors == (), f"{item_id} scored against cross-references: {item.value_anchors}"


def test_exactly_the_expected_rows_have_no_value_anchors(dataset):
    empty = [i.id for i in dataset.items if not i.value_anchors]
    assert empty == ZERO_ANCHOR_IDS


@pytest.mark.parametrize(
    ("item_id", "expected"),
    [
        ("Q01", ["9000"]),
        ("Q02", ["850"]),
        ("Q03", ["300", "1", "20", "4", "7"]),
        ("Q06", ["47", "38", "304"]),  # `classe 8.8` is dotted -> dropped as a chapter ref
        ("Q13", ["1000", "500"]),
        ("Q22", ["5", "40", "80", "2000", "250"]),
        ("Q25", ["8573"]),  # `ISO 4-2-1` is a compound code, not three values
        ("Q32", ["0.55", "2.2"]),  # three motors at 0,55 kW de-duplicate to one anchor
    ],
)
def test_value_anchor_extraction(dataset, item_id, expected):
    assert [a.value for a in _item(dataset, item_id).value_anchors] == expected


def test_rubric_entries_are_folded_into_the_anchor_set(dataset):
    q33 = _item(dataset, "Q33")
    assert q33.required == ("orario",)
    assert q33.forbidden == ()  # a correct answer repeats "antiorario" to negate it
    # `anchors` is what anchor_recall checks: auto values plus rubric `required`.
    assert [a.kind for a in q33.anchors] == ["required"]
    assert _item(dataset, "Q05").forbidden == ("2 teste",)


def test_every_rubric_row_documents_its_provenance(dataset):
    """The rubric may only operationalise expectations the golden set states."""
    rubric_ids = {i.id for i in dataset.items if i.required or i.forbidden}
    assert rubric_ids, "rubric sidecar did not load"
    for item in dataset.items:
        if item.required or item.forbidden:
            assert item.rubric_note.strip(), f"{item.id}: rubric row has no note"


# ---------------------------------------------------------------------------
# Pag. stampata parser — every distinct value in the real golden set's original
# block, plus the shapes the legend allows but that block happens not to use.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("8", {8}),
        ("9", {9}),
        ("7", {7}),
        ("20", {20}),
        ("29", {29}),
        ("31", {31}),
        ("37", {37}),
        ("38", {38}),
        ("83", {83}),
        ("106", {106}),
        ("107", {107}),
        ("110", {110}),
        ("179", {179}),
        ("104-105", {104, 105}),
        ("108-109", {108, 109}),
        ("29, 178", {29, 178}),
        ("4, 10", {4, 10}),
        ("5, 37", {5, 37}),
        ("—", set()),
        # shapes the parser must survive even though the golden set has none
        ("", set()),
        ("   ", set()),
        ("12–14", {12, 13, 14}),  # en dash
        ("14-12", {12, 13, 14}),  # reversed range
        ("1-500", {1, 500}),  # absurd span -> endpoints only, never 500 pages
        ("n/a", set()),
    ],
)
def test_parse_pages(raw, expected):
    assert set(parse_pages(raw)) == expected


def test_every_distinct_value_in_the_file_parses(dataset):
    distinct = {i.pag_stampata for i in dataset.items}
    for raw in distinct:
        pages = parse_pages(raw)
        if any(ch.isdigit() for ch in raw):
            assert pages, f"{raw!r} parsed to nothing"
        else:
            assert pages == frozenset()


# ---------------------------------------------------------------------------
# --only
# ---------------------------------------------------------------------------


def test_parse_only_ids_and_ranges(dataset):
    selected = parse_only("Q01,Q05-Q08", dataset.items)
    assert [i.id for i in selected] == ["Q01", "Q05", "Q06", "Q07", "Q08"]


def test_parse_only_is_case_insensitive_and_space_tolerant(dataset):
    assert [i.id for i in parse_only("q02 q03", dataset.items)] == ["Q02", "Q03"]


def test_parse_only_that_matches_nothing_is_an_error(dataset):
    with pytest.raises(DatasetError, match="matched no questions"):
        parse_only("Q999", dataset.items)


def test_parse_only_that_matches_nothing_is_a_no_match_error(dataset):
    """A DatasetError subclass, so the CLI can skip a file that holds none of the IDs."""
    with pytest.raises(NoMatchError):
        parse_only("Q999", dataset.items)
    assert issubclass(NoMatchError, DatasetError)


def test_parse_only_reads_ids_and_ranges_from_a_file_and_ignores_comments(dataset, tmp_path):
    ids = tmp_path / "ids.txt"
    ids.write_text("# smoke subset\nQ01  # simple fact\nQ05-Q06, Q13\n\n   # nothing here\n", encoding="utf-8")

    selected = parse_only(f"@{ids}", dataset.items)

    assert [i.id for i in selected] == ["Q01", "Q05", "Q06", "Q13"]


def test_parse_only_mixes_a_file_with_literal_ids(dataset, tmp_path):
    ids = tmp_path / "ids.txt"
    ids.write_text("Q02\n", encoding="utf-8")

    assert [i.id for i in parse_only(f"Q07,@{ids}", dataset.items)] == ["Q02", "Q07"]


def test_parse_only_with_an_unreadable_file_is_a_dataset_error(dataset, tmp_path):
    with pytest.raises(DatasetError, match="cannot read"):
        parse_only(f"@{tmp_path / 'missing.txt'}", dataset.items)


def test_parse_only_with_a_bare_at_sign_is_a_dataset_error(dataset):
    with pytest.raises(DatasetError, match="needs a file name"):
        parse_only("@", dataset.items)


def test_parse_only_with_an_empty_file_is_a_no_match_error(dataset, tmp_path):
    ids = tmp_path / "ids.txt"
    ids.write_text("# nothing selected yet\n", encoding="utf-8")

    with pytest.raises(NoMatchError):
        parse_only(f"@{ids}", dataset.items)
