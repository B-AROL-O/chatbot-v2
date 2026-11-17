"""Oracles on the **real** golden set, in the private eval repo.

The loader's behaviour is tested on the synthetic ``tests/fixtures/eval/`` set,
which always runs. What only the real files can answer is pinned here: the row
counts, that Q01-Q36 are still the original block, that every family has its
one ``nuda`` row, and that every ``Pag. stampata`` value they contain parses.

Runs when ``$EVAL_DATASETS_DIR`` (default ``data/eval/datasets``, a checkout of
the eval-data repo at ``data/eval/``) holds the sets;
skipped otherwise, like the live-API suites without credentials.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sme_chatbot import config
from sme_chatbot.eval.cli import main
from sme_chatbot.eval.dataset import EvalDataset, discover_datasets, load_dataset, parse_pages

REPO_ROOT = Path(__file__).resolve().parents[2]


def _under_repo(path: str) -> Path:
    return Path(path) if Path(path).is_absolute() else REPO_ROOT / path


DATASETS_DIR = _under_repo(config.EVAL_DATASETS_DIR)
RESULTS_DIR = _under_repo(config.EVAL_RESULTS_DIR)
DATASET = DATASETS_DIR / "eval-chatbot-arol-domande.csv"

pytestmark = pytest.mark.skipif(
    not DATASET.is_file(),
    reason=f"golden set not checked out at {DATASETS_DIR} (clone the eval-data repo at data/eval/)",
)

ORIGINAL_IDS = [f"Q{n:02d}" for n in range(1, 37)]

# Every original row whose `Risposta attesa` contains only cross-references, codes
# or the matricola (plan §1.1 / WP4 item 1).
ZERO_ANCHOR_IDS = [
    "Q14", "Q18", "Q19", "Q20", "Q21", "Q23", "Q24",
    "Q26", "Q27", "Q28", "Q29", "Q30", "Q31", "Q33", "Q34",
]


@pytest.fixture(scope="module")
def dataset() -> EvalDataset:
    return load_dataset(DATASET)


@pytest.fixture(scope="module")
def original(dataset: EvalDataset):
    """The original 36 questions, all on matricola 20349 and unchanged since 2026-08-31."""
    return [i for i in dataset.items if i.id in set(ORIGINAL_IDS)]


def test_loads_every_arol_row_with_the_original_block_first(dataset, original):
    assert dataset.n_items == 116
    ids = [i.id for i in dataset.items]
    assert len(set(ids)) == len(ids)
    assert ids[:36] == ORIGINAL_IDS
    assert [i.id for i in original] == ORIGINAL_IDS
    assert dataset.legend_path is not None and dataset.legend_path.name == "eval-chatbot-legenda.csv"
    assert dataset.rubric_path is not None


def test_the_plans_wp1_oracle_on_the_original_block(original):
    by_id = {i.id: i for i in original}
    assert by_id["Q34"].serial is None
    assert all(i.serial == "20349" for i in original if i.id != "Q34")
    assert by_id["Q35"].language == "en"
    assert set(by_id["Q13"].gold_pages) == {108, 109}
    assert set(by_id["Q29"].gold_pages) == set()
    assert [i.id for i in original if i.expects_refusal] == ["Q29", "Q30", "Q31"]
    assert [i.id for i in original if i.expects_disambiguation] == ["Q06", "Q07", "Q08", "Q34"]


def test_exactly_the_expected_original_rows_have_no_value_anchors(original):
    assert [i.id for i in original if not i.value_anchors] == ZERO_ANCHOR_IDS


def test_serial_is_absent_exactly_when_scope_is_nessuna(dataset):
    for item in dataset.items:
        scope = item.raw["Scope"]
        assert (item.serial is None) == (scope == "nessuna"), f"{item.id}: scope {scope!r}, serial {item.serial!r}"
        if item.serial is not None:
            assert item.serial == item.raw["Matricola"], item.id


def test_every_row_belongs_to_a_family_that_exists(dataset):
    ids = {i.id for i in dataset.items}
    for item in dataset.items:
        assert item.raw["Famiglia"] in ids, f"{item.id}: family {item.raw['Famiglia']!r} is not a row"
    families: dict[str, list[str]] = {}
    for item in dataset.items:
        families.setdefault(item.raw["Famiglia"], []).append(item.raw["Scope"])
    for family, scopes in families.items():
        assert scopes.count("nuda") == 1, f"family {family}: {scopes}"


def test_every_rubric_row_documents_its_provenance(dataset):
    rubric_ids = {i.id for i in dataset.items if i.required or i.forbidden}
    assert rubric_ids, "rubric sidecar did not load"
    for item in dataset.items:
        if item.required or item.forbidden:
            assert item.rubric_note.strip(), f"{item.id}: rubric row has no note"


def test_every_distinct_page_value_in_every_set_parses(original):
    # The 19 is the original block; the later rows add more shapes, all of which must parse too.
    assert len({i.pag_stampata for i in original}) == 19
    for path in discover_datasets(DATASETS_DIR):
        for raw in {i.pag_stampata for i in load_dataset(path).items}:
            pages = parse_pages(raw)
            if any(ch.isdigit() for ch in raw):
                assert pages, f"{path.name}: {raw!r} parsed to nothing"
            else:
                assert pages == frozenset()


def test_question_ids_are_unique_across_the_question_sets():
    """The premise of one ``--only`` list spanning every set."""
    ids = [item.id for path in discover_datasets(DATASETS_DIR) for item in load_dataset(path).items]
    assert len(discover_datasets(DATASETS_DIR)) >= 3
    assert len(ids) == len(set(ids))


def test_rescore_honours_only(capsys):
    """`--rescore … --only Q26,Q75` used to rescore (and judge) the whole run."""
    # The newest full-set run record of the main set, whatever deployment it ran on.
    runs = sorted(RESULTS_DIR.glob("*-full-*-eval-chatbot-arol/results.json"))
    if not runs:
        pytest.skip(f"no full-set run record checked out under {RESULTS_DIR}")
    source = runs[-1]

    code = main(["--dataset", str(DATASET), "--rescore", str(source), "--only", "Q26,Q75", "--no-write"])

    assert code == 0
    assert f"rescored: 2 item(s) from {source}" in capsys.readouterr().out
