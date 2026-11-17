"""``sme-chatbot-eval`` over several question sets with one ``--only`` list.

Uses ``--dry-run`` against the three synthetic sets in ``tests/fixtures/eval/``
(``$EVAL_DATASETS_DIR`` pointed there), so no backend is involved. The question
IDs are unique across the files, which is what lets one ID list span them; the
same property of the real sets is pinned in ``test_eval_golden_set.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sme_chatbot import config
from sme_chatbot.eval.cli import main
from sme_chatbot.eval.dataset import discover_datasets, load_dataset

EVAL_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "eval"


@pytest.fixture(autouse=True)
def _fixture_eval_dir(monkeypatch):
    monkeypatch.setattr(config, "EVAL_DATASETS_DIR", str(EVAL_DIR))


@pytest.fixture(scope="module")
def datasets():
    return {path.name: load_dataset(path) for path in discover_datasets(EVAL_DIR)}


def _first_id(datasets, name_part: str) -> str:
    name = next(n for n in datasets if name_part in n)
    return datasets[name].items[0].id


def test_discover_defaults_to_eval_datasets_dir(datasets):
    assert discover_datasets() == discover_datasets(EVAL_DIR)
    assert len(datasets) == 3


def test_question_ids_are_unique_across_the_question_sets(datasets):
    """The premise of a single ID list spanning several files."""
    ids = [item.id for dataset in datasets.values() for item in dataset.items]
    assert len(ids) == len(set(ids))


def test_an_id_list_spanning_sets_skips_the_files_that_hold_none_of_it(datasets, capsys):
    only = f"{_first_id(datasets, 'alpha')},{_first_id(datasets, 'beta')}"

    code = main(["--dry-run", "--only", only])

    out = capsys.readouterr().out
    assert code == 0
    assert out.count("asking  : 1 question(s)") == 2
    skipped = [line for line in out.splitlines() if line.startswith("skipped :")]
    assert len(skipped) == 1
    assert "gamma" in skipped[0]


def test_an_only_file_works_across_sets(datasets, capsys, tmp_path):
    ids = tmp_path / "ids.txt"
    ids.write_text(
        f"# one per manufacturer\n{_first_id(datasets, 'alpha')}\n{_first_id(datasets, 'gamma')}\n{_first_id(datasets, 'beta')}\n",
        encoding="utf-8",
    )

    code = main(["--dry-run", "--only", f"@{ids}"])

    out = capsys.readouterr().out
    assert code == 0
    assert out.count("asking  : 1 question(s)") == 3
    assert "skipped :" not in out


def test_an_id_that_exists_nowhere_is_an_error_not_a_silent_empty_run(capsys):
    code = main(["--dry-run", "--only", "Q9999"])

    captured = capsys.readouterr()
    assert code == 1
    assert "matched no questions in any of" in captured.err


def test_a_single_dataset_with_no_match_is_still_an_error(capsys):
    code = main(["--dry-run", "--dataset", str(EVAL_DIR / "fixture-alpha-domande.csv"), "--only", "Q9999"])

    captured = capsys.readouterr()
    assert code == 1
    assert "matched no questions" in captured.err
    assert "skipped" not in captured.out


def test_no_eval_checkout_says_where_the_data_lives(monkeypatch, tmp_path, capsys):
    """The golden sets moved out of this repo; a bare run must say how to get them."""
    monkeypatch.setattr(config, "EVAL_DATASETS_DIR", str(tmp_path / "missing"))

    code = main(["--dry-run"])

    err = capsys.readouterr().err
    assert code == 1
    assert "eval-data repo" in err
    assert "EVAL_DATASETS_DIR" in err
