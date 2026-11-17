"""WP6 — the four run artifacts.

``results.csv`` is the one with a hard external contract: it is meant to paste
straight back into ``the golden set's .xlsx``, so its column order and header
text must come from the source file rather than from a constant in the code.
"""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from sme_chatbot.eval.dataset import load_dataset
from sme_chatbot.eval.report import (
    APPENDED_COLUMNS,
    build_run_header,
    build_summary,
    load_results_json,
    render_report_md,
    run_directory,
    write_all,
)
from sme_chatbot.eval.runner import EvalResult, rescore
from sme_chatbot.eval.scoring import score_item

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET = REPO_ROOT / "tests" / "fixtures" / "eval" / "fixture-alpha-domande.csv"

GOLD_CITATION = {
    "source": "90001_manual_IT.pdf",
    "page": 18,
    "document_id": "doc-1",
    "distance": 0.12,
    "filename": "90001_manual_IT.pdf",
    "repository": "ACME-NETWORK-SHARE",
}


@pytest.fixture(scope="module")
def dataset():
    return load_dataset(DATASET)


def _result(dataset, item_id: str, answer: str, citations: list[dict], *, error: str | None = None) -> EvalResult:
    item = next(i for i in dataset.items if i.id == item_id)
    result = EvalResult(
        item=item,
        run_key=item.id,
        answer="" if error else answer,
        citations=[] if error else citations,
        usage={} if error else {"rounds": 2},
        latency_ms=1234,
        status_code=500 if error else 200,
        attempts=4 if error else 1,
        error=error,
    )
    if error:
        result.verdict_auto = "ERROR"
        return result
    result.scores = score_item(item, answer, citations)
    result.verdict_auto = result.scores.verdict
    return result


@pytest.fixture(scope="module")
def results(dataset):
    return [
        _result(dataset, "Q01", "La produzione è 9000 bph (90001_manual_IT.pdf, p. 18).", [GOLD_CITATION]),
        _result(dataset, "Q02", "Pesa circa 900 kg.", [GOLD_CITATION]),
        _result(dataset, "Q28", "L'orientatore seleziona le capsule nella tramoggia.", []),
        _result(dataset, "Q13", "", [], error="HTTP 500: boom"),
    ]


@pytest.fixture(scope="module")
def run_header(dataset):
    return build_run_header(
        dataset=dataset,
        api_url="https://backend.invalid",
        domain="documents",
        health={"status": "ok", "domains": ["documents"], "version": "v0.7.0"},
        options={"repeat": 1, "concurrency": 1, "page_offset": 10},
        started_at=datetime(2026, 9, 14, 8, 0, tzinfo=UTC),
        finished_at=datetime(2026, 9, 14, 8, 5, tzinfo=UTC),
        llm_model="gpt-5",
        backend_config={
            "version": "v0.7.0",
            "model": "gpt-5",
            "llm_base_url": "https://llm.invalid/v1",
            "auth_mode": "dev",
            "domains": ["documents"],
        },
    )


@pytest.fixture(scope="module")
def written(tmp_path_factory, dataset, results, run_header):
    out = tmp_path_factory.mktemp("run") / "2026-09-14-test"
    write_all(out, dataset=dataset, results=results, run_header=run_header, domain="documents")
    return out


# ---------------------------------------------------------------------------
# run directory
# ---------------------------------------------------------------------------


def test_run_directory_is_dated_and_named():
    path = run_directory("data/eval/results", "baseline", now=datetime(2026, 9, 14, tzinfo=UTC))
    assert path.name == "2026-09-14-baseline"
    assert path.parent.as_posix().endswith("data/eval/results")


def test_write_all_produces_exactly_the_four_artifacts(written):
    assert sorted(p.name for p in written.iterdir()) == ["report.md", "results.csv", "results.json", "summary.json"]


# ---------------------------------------------------------------------------
# results.csv — the paste-back contract
# ---------------------------------------------------------------------------


def test_results_csv_preserves_source_column_order_and_header_text(written, dataset):
    with (written / "results.csv").open(encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle))
    assert header[: len(dataset.columns)] == list(dataset.columns)
    assert header[len(dataset.columns) :] == list(APPENDED_COLUMNS)


def test_results_csv_fills_the_yellow_columns(written):
    rows = _csv_rows(written)
    q01 = rows["Q01"]
    assert q01["Risposta corretta? (S/N/P)"] == "S"
    assert q01["Fonti corrette? (S/N)"] == "S"
    assert q01["Ha citato? (S/N)"] == "S"
    assert "anchor recall" in q01["Note esito"]
    assert q01["Risposta ottenuta"].startswith("La produzione")
    assert q01["Fonti citate"] == "90001_manual_IT.pdf"
    assert q01["Latenza (ms)"] == "1234"
    assert q01["Round"] == "2"


def test_results_csv_keeps_the_untouched_source_cells_verbatim(written, dataset):
    rows = _csv_rows(written)
    q01_item = next(i for i in dataset.items if i.id == "Q01")
    assert rows["Q01"]["Domanda"] == q01_item.domanda
    assert rows["Q01"]["Risposta attesa"] == q01_item.risposta_attesa
    assert rows["Q01"]["Documento"] == "90001_manual_IT.pdf"


def test_results_csv_leaves_judgement_columns_empty_for_an_error(written):
    row = _csv_rows(written)["Q13"]
    assert row["Risposta corretta? (S/N/P)"] == "ERROR"
    # An unanswered question has no source or citation judgement to record —
    # writing "N" there would read as "the backend cited the wrong document".
    assert row["Fonti corrette? (S/N)"] == ""
    assert row["Ha citato? (S/N)"] == ""
    assert row["Note esito"].startswith("ERROR: HTTP 500")


def _csv_rows(written: Path) -> dict[str, dict[str, str]]:
    with (written / "results.csv").open(encoding="utf-8-sig", newline="") as handle:
        return {row["ID"]: row for row in csv.DictReader(handle)}


# ---------------------------------------------------------------------------
# results.json — the reproducibility record
# ---------------------------------------------------------------------------


def test_results_json_run_header_carries_the_provenance(written, dataset):
    payload = json.loads((written / "results.json").read_text(encoding="utf-8"))
    run = payload["run"]
    assert run["api_url"] == "https://backend.invalid"
    assert run["backend_version"] == "v0.7.0"
    assert run["backend_domains"] == ["documents"]
    assert run["domain"] == "documents"
    assert run["llm_model"] == "gpt-5"
    assert run["backend_config"]["llm_base_url"] == "https://llm.invalid/v1"
    assert run["backend_config"]["auth_mode"] == "dev"
    assert run["dataset"]["sha256"] == dataset.sha256
    assert run["dataset"]["rows"] == dataset.n_items
    assert run["options"]["page_offset"] == 10
    assert run["session_protocol"].startswith("isolated")
    assert payload["limitations"]


def test_results_json_records_request_answer_citations_and_scores(written):
    items = {i["run_key"]: i for i in json.loads((written / "results.json").read_text(encoding="utf-8"))["items"]}
    q01 = items["Q01"]
    assert q01["request"] == {
        "domain": "documents",
        "message": q01["expected"]["risposta_attesa"] and q01["request"]["message"],
        "history": [],
        "stream": False,
    }
    assert q01["response"]["citations"] == [GOLD_CITATION]
    assert q01["response"]["usage"] == {"rounds": 2}
    assert q01["scores"]["anchor_recall"] == 1.0
    assert q01["scores"]["anchors_matched"] == ["9000 pezzi/ora"]
    assert q01["scores"]["gold_doc_cited"] is True
    assert q01["verdict"] == "S"


def test_results_json_records_tokens_and_cost_as_null_not_zero(written):
    """So a later token-count fix is visible as a change *from null* (plan WP4.9)."""
    items = json.loads((written / "results.json").read_text(encoding="utf-8"))["items"]
    for item in items:
        assert item["prompt_tokens"] is None
        assert item["completion_tokens"] is None
        assert item["cost"] is None


def test_results_json_gives_an_error_item_no_verdict_and_no_scores(written):
    items = {i["run_key"]: i for i in json.loads((written / "results.json").read_text(encoding="utf-8"))["items"]}
    q13 = items["Q13"]
    assert q13["error"] == "HTTP 500: boom"
    assert q13["scores"] is None
    assert q13["verdict"] == "ERROR"


# ---------------------------------------------------------------------------
# summary.json
# ---------------------------------------------------------------------------


def test_summary_excludes_error_from_percentages(results, run_header):
    summary = build_summary(results, run_header)
    assert summary["verdicts"] == {"S": 1, "P": 1, "N": 1, "?": 0, "ERROR": 1}
    # 4 asked, but only S+P+N are scorable: 3. One of those is correct.
    assert summary["scored_items"] == 3
    assert summary["pct_correct_of_scored"] == 33.3
    assert summary["errors"] == 1


def test_summary_excludes_undetermined_from_percentages(run_header):
    """`?` is counted and displayed, never folded into a "% correct"."""
    bare = load_dataset(DATASET, rubric=False)
    q28 = next(i for i in bare.items if i.id == "Q28")
    undetermined = EvalResult(item=q28, run_key="Q28", answer="Una descrizione qualsiasi.", citations=[])
    undetermined.scores = score_item(q28, undetermined.answer, [])
    undetermined.verdict_auto = undetermined.scores.verdict

    assert undetermined.verdict == "?"
    summary = build_summary([undetermined], run_header)
    assert summary["verdicts"]["?"] == 1
    assert summary["scored_items"] == 0
    assert summary["pct_correct_of_scored"] is None


def test_summary_carries_the_fields_102_needs_to_diff_two_runs(results, run_header):
    summary = build_summary(results, run_header)
    for key in ("backend_version", "dataset_sha256", "mean_latency_ms", "p95_latency_ms",
                "mean_rounds", "mean_citations", "mean_anchor_recall", "gold_doc_cited"):
        assert key in summary, key
    assert summary["mean_rounds"] == 2.0


# ---------------------------------------------------------------------------
# report.md
# ---------------------------------------------------------------------------


def test_report_shows_the_backends_own_config_not_just_the_model(written):
    """LLM model must read as the backend's own `/config`, plus the other
    fields that endpoint returns — not the runner host's local `.env`."""
    text = (written / "report.md").read_text(encoding="utf-8")
    assert "| LLM model (backend config) | gpt-5 |" in text
    assert "| LLM base URL (backend config) | https://llm.invalid/v1 |" in text
    assert "| Auth mode (backend config) | dev |" in text


def test_report_says_how_much_it_leans_on_the_derived_rubric(written):
    """The rubric is where scoring could quietly drift (plan §7 R6).

    A reader must be able to see, without opening any CSV, how many verdicts
    exist only because a derived file said so.
    """
    text = (written / "report.md").read_text(encoding="utf-8")
    assert "Items scored only via the rubric | 1 of 4" in text
    assert "--no-rubric" in text


def test_report_states_its_limitations_on_its_face(written):
    text = (written / "report.md").read_text(encoding="utf-8")
    assert "recall, not correctness" in text
    assert "never folded into a percentage" in text
    assert "deviation" in text  # the isolated-session protocol
    assert "does not decide whether the chatbot is good enough" in text


def test_report_puts_failures_before_the_per_question_detail(written):
    text = (written / "report.md").read_text(encoding="utf-8")
    assert text.index("## Failures first") < text.index("## Per question")
    assert "S 1 · P 1 · N 1 · ? 0 · ERROR 1" in text


def test_report_shows_matched_and_missing_anchors_verbatim(written):
    """A reviewer must be able to overrule the machine without opening the JSON."""
    text = (written / "report.md").read_text(encoding="utf-8")
    assert "`850 kg`" in text  # Q02's missing anchor, quoted as the answer key wrote it
    assert "trovati:" in text and "mancanti:" in text


def test_report_renders_an_error_item_without_inventing_scores(written):
    text = (written / "report.md").read_text(encoding="utf-8")
    assert "**ERROR** — HTTP 500: boom" in text


def test_report_survives_pipes_and_newlines_in_free_text(dataset, run_header):
    """Question text goes into Markdown tables; a stray `|` must not break them."""
    item = next(i for i in dataset.items if i.id == "Q01")
    nasty = EvalResult(item=item, run_key="Q01", answer="a | b\nsecond line", citations=[])
    nasty.scores = score_item(item, nasty.answer, [])
    nasty.verdict_auto = nasty.scores.verdict
    text = render_report_md(run_header, [nasty], build_summary([nasty], run_header), dataset=dataset)
    failures_table = text.split("## Failures first")[1].split("##")[0]
    for line in failures_table.strip().splitlines():
        if line.startswith("|"):
            assert line.count("|") - line.count("\\|") == 6


def test_report_adds_a_disagreement_section_only_when_the_judge_disagrees(dataset, run_header, results):
    baseline = render_report_md(run_header, results, build_summary(results, run_header), dataset=dataset)
    assert "## Disagreements" not in baseline

    judged = list(results)
    judged[0].verdict_judge, judged[0].judge_reason = "N", "the value is wrong"
    try:
        text = render_report_md(run_header, judged, build_summary(judged, run_header), dataset=dataset)
        assert "## Disagreements (human review queue)" in text
        assert "the value is wrong" in text
        # The judge never overwrites an anchored S.
        assert judged[0].verdict == "S"
    finally:
        judged[0].verdict_judge, judged[0].judge_reason = None, None



@pytest.mark.parametrize(("errors", "shown"), [(0, False), (27, True)])
def test_report_says_when_judge_calls_failed(dataset, run_header, results, errors, shown):
    """A run whose judge failed on every call must not read as judged."""
    header = {**run_header, "judge": {"judge_model": "gpt-5", "judge_calls": 27, "judge_errors": errors}}
    text = render_report_md(header, results, build_summary(results, header), dataset=dataset)
    assert ("**Judge errors: 27 of 27 judge calls failed**" in text) is shown


def test_report_counts_content_filter_blocks_apart(dataset, run_header, results):
    judge = {"judge_model": "gpt-5", "judge_calls": 106, "judge_errors": 2, "judge_content_filtered": 2}
    header = {**run_header, "judge": judge}
    text = render_report_md(header, results, build_summary(results, header), dataset=dataset)
    assert (
        "**Judge errors: 2 of 106 judge calls failed** (2 blocked by the LLM provider's content filter)" in text
    )


# ---------------------------------------------------------------------------
# rescoring a committed run (plan §5 step 3)
# ---------------------------------------------------------------------------


def test_rescore_round_trips_a_committed_results_json(written, dataset):
    """The baseline ships before scoring exists; this is how it gets scored.

    The expensive part of a run is the captured answers, and they are all in
    ``results.json`` — so re-deriving verdicts must never need the backend again.
    """
    header, restored, warnings = load_results_json(written / "results.json", dataset)

    assert warnings == []
    assert [r.run_key for r in restored] == ["Q01", "Q02", "Q28", "Q13"]
    assert restored[0].answer.startswith("La produzione")
    assert restored[0].citations == [GOLD_CITATION]
    assert restored[0].usage == {"rounds": 2}
    assert restored[3].error == "HTTP 500: boom"
    assert header["backend_version"] == "v0.7.0"
    # Scores are deliberately not restored - rescoring is the whole point.
    assert all(r.scores is None for r in restored)

    rescored = rescore(restored, page_offset=10, page_tolerance=2)
    assert [r.verdict for r in rescored] == ["S", "N", "P", "ERROR"]
    # An ERROR carries no answer, so it stays an ERROR rather than becoming `N`.
    assert rescored[3].scores is None


def test_rescore_flags_an_answer_key_that_changed_since_the_run(written, dataset, tmp_path):
    """Plan §7 R4: a run scored against a later edit must not pass silently."""
    edited = tmp_path / "edited-domande.csv"
    edited.write_bytes(DATASET.read_bytes() + b"Q99,Extra,1,d,d,,,,x,,,,,\n")

    _, restored, warnings = load_results_json(written / "results.json", load_dataset(edited, rubric=False))

    assert any("answer key changed since the run" in w for w in warnings)
    assert any("not comparable" in w for w in warnings)
    assert restored, "the answers are still usable; the caller is just warned"


def test_rescore_reports_questions_that_no_longer_exist(written, tmp_path):
    trimmed = tmp_path / "trimmed-domande.csv"
    rows = DATASET.read_text(encoding="utf-8-sig").splitlines(keepends=True)
    keep = [rows[0]] + [r for r in rows[1:] if not r.startswith("Q02,")]
    trimmed.write_text("".join(keep), encoding="utf-8")

    _, restored, warnings = load_results_json(written / "results.json", load_dataset(trimmed, rubric=False))

    assert any("Q02" in w and "no such question" in w for w in warnings)
    assert [r.run_key for r in restored] == ["Q01", "Q28", "Q13"]


def test_results_csv_writes_into_the_header_the_file_actually_uses(tmp_path, dataset, run_header):
    """A re-exported workbook must not silently produce empty verdict columns.

    The loader resolves headers accent- and case-insensitively; the writer has
    to resolve them the same way, or a lower-cased `id` header would leave the
    yellow columns blank with no error anywhere.
    """
    rows = DATASET.read_text(encoding="utf-8-sig").splitlines(keepends=True)
    rows[0] = rows[0].replace("ID,", "id,", 1).replace("Risposta corretta? (S/N/P)", "RISPOSTA CORRETTA? (S/N/P)")
    variant = tmp_path / "variant-domande.csv"
    variant.write_text("".join(rows), encoding="utf-8")

    reexported = load_dataset(variant, rubric=False)
    item = next(i for i in reexported.items if i.id == "Q02")
    result = EvalResult(item=item, run_key="Q02", answer="Pesa 850 kg.", citations=[])
    result.scores = score_item(item, result.answer, [])
    result.verdict_auto = result.scores.verdict

    out = tmp_path / "out"
    write_all(out, dataset=reexported, results=[result], run_header=run_header, domain="documents")

    with (out / "results.csv").open(encoding="utf-8-sig", newline="") as handle:
        written_row = next(iter(csv.DictReader(handle)))
    assert written_row["id"] == "Q02"
    assert written_row["RISPOSTA CORRETTA? (S/N/P)"] == "S"


# ---------------------------------------------------------------------------
# backend stage timings
# ---------------------------------------------------------------------------


def _timed_result(dataset, item_id, *, rounds, latency_ms, ttft, llm, tool, setup):
    result = _result(dataset, item_id, "Pesa circa 900 kg.", [GOLD_CITATION])
    result.latency_ms = latency_ms
    result.usage = {
        "rounds": rounds,
        "timings": {"total_ms": llm + tool, "ttft_ms": ttft, "llm_ms": llm, "tool_ms": tool, "setup_ms": setup},
    }
    return result


@pytest.fixture(scope="module")
def timed_results(dataset):
    return [
        _timed_result(dataset, "Q01", rounds=2, latency_ms=9000, ttft=1000, llm=7000, tool=1500, setup=500),
        _timed_result(dataset, "Q02", rounds=2, latency_ms=11000, ttft=2000, llm=9000, tool=1500, setup=500),
        _timed_result(dataset, "Q28", rounds=6, latency_ms=80000, ttft=3000, llm=70000, tool=9000, setup=1000),
    ]


def test_summary_aggregates_the_backend_stage_timings(timed_results, run_header):
    summary = build_summary(timed_results, run_header)

    assert summary["p50_ttft_ms"] == 2000.0
    assert summary["p95_ttft_ms"] == 3000.0
    assert summary["mean_setup_ms"] == pytest.approx(666.7)
    assert summary["mean_llm_ms"] == pytest.approx(28666.7)
    assert summary["mean_tool_ms"] == pytest.approx(4000.0)
    assert summary["p50_latency_ms"] == 11000.0
    assert summary["latency_by_rounds"] == {
        "2": {"n": 2, "median_ms": 10000.0},
        "6": {"n": 1, "median_ms": 80000.0},
    }


def test_summary_of_a_pre_237_run_has_null_timings_not_zero(results, run_header):
    """`results` carries ``usage == {"rounds": 2}`` only, as every committed run does today."""
    summary = build_summary(results, run_header)

    for key in ("p50_ttft_ms", "p95_ttft_ms", "mean_setup_ms", "mean_llm_ms", "mean_tool_ms"):
        assert summary[key] is None, key
    # The by-rounds view only needs `rounds` and the client-side latency, so it works for old runs too.
    assert summary["latency_by_rounds"] == {"2": {"n": 3, "median_ms": 1234.0}}


def test_report_shows_ttft_and_the_latency_by_rounds_table(tmp_path, dataset, timed_results, run_header):
    out = tmp_path / "2026-10-14-timed"
    write_all(out, dataset=dataset, results=timed_results, run_header=run_header, domain="documents")

    report = (out / "report.md").read_text(encoding="utf-8")

    assert "| p50 time to first token (ms) | 2000.0 |" in report
    assert "### Latency by tool rounds" in report
    assert "| 6 | 1 | 80000 |" in report


def test_report_of_a_pre_237_run_renders_dashes_for_ttft(written):
    report = (written / "report.md").read_text(encoding="utf-8")

    assert "| p50 time to first token (ms) | — |" in report
    assert "| Mean setup / LLM / tool time (ms) | — |" in report


def test_results_csv_has_a_ttft_column_blank_when_unknown(tmp_path, dataset, timed_results, run_header, results):
    out = tmp_path / "2026-10-14-timed"
    write_all(out, dataset=dataset, results=timed_results, run_header=run_header, domain="documents")
    rows = _csv_rows(out)
    assert rows["Q01"]["TTFT (ms)"] == "1000"
    assert rows["Q28"]["TTFT (ms)"] == "3000"

    old = tmp_path / "2026-10-14-old"
    write_all(old, dataset=dataset, results=results, run_header=run_header, domain="documents")
    assert _csv_rows(old)["Q01"]["TTFT (ms)"] == ""


def test_timings_survive_the_results_json_round_trip(tmp_path, dataset, timed_results, run_header):
    out = tmp_path / "2026-10-14-timed"
    write_all(out, dataset=dataset, results=timed_results, run_header=run_header, domain="documents")

    _, restored, warnings = load_results_json(out / "results.json", dataset)

    assert warnings == []
    assert [r.timing("ttft_ms") for r in restored] == [1000, 2000, 3000]
    assert restored[0].timing("no_such_key") is None


# ---------------------------------------------------------------------------
# subset runs (--only) say so on their face
# ---------------------------------------------------------------------------


def _with_only(run_header, only):
    header = dict(run_header)
    header["options"] = {**run_header["options"], "only": only}
    return header


def test_summary_of_a_full_run_has_no_subset_marker(results, run_header):
    summary = build_summary(results, run_header)

    assert summary["only"] is None
    assert summary["dataset_rows"] == run_header["dataset"]["rows"]


def test_summary_records_the_only_spec_and_the_size_of_the_question_set(results, run_header):
    summary = build_summary(results, _with_only(run_header, "@data/eval/datasets/derived/smoke-ids.txt"))

    assert summary["only"] == "@data/eval/datasets/derived/smoke-ids.txt"
    assert summary["n_items"] == len(results)
    assert summary["dataset_rows"] == run_header["dataset"]["rows"]


def test_report_flags_a_subset_run_and_a_full_run_is_not_flagged(tmp_path, dataset, results, run_header):
    subset_dir = tmp_path / "2026-10-14-subset"
    write_all(subset_dir, dataset=dataset, results=results, run_header=_with_only(run_header, "Q01,Q02"), domain="documents")
    full_dir = tmp_path / "2026-10-14-full"
    write_all(full_dir, dataset=dataset, results=results, run_header=run_header, domain="documents")

    subset_report = (subset_dir / "report.md").read_text(encoding="utf-8")
    full_report = (full_dir / "report.md").read_text(encoding="utf-8")

    assert "**Subset run.** `--only Q01,Q02`" in subset_report
    assert "not a full run" in subset_report
    assert "Subset run" not in full_report


# ---------------------------------------------------------------------------
# calls cut off by max_tokens are counted, so a cap that truncates answers is visible
# ---------------------------------------------------------------------------


def _with_calls(result, *finishes):
    result.usage = {"rounds": len(finishes), "timings": {"llm_calls": [{"round": n, "finish": f} for n, f in enumerate(finishes)]}}
    return result


def test_summary_counts_the_llm_calls_cut_off_by_max_tokens(dataset, run_header):
    results = [
        _with_calls(_result(dataset, "Q01", "ok", [GOLD_CITATION]), "tool_calls", "length"),
        _with_calls(_result(dataset, "Q02", "ok", [GOLD_CITATION]), "length", "length"),
        _with_calls(_result(dataset, "Q28", "ok", [GOLD_CITATION]), "tool_calls", "stop"),
    ]

    assert build_summary(results, run_header)["truncated_calls"] == 3


def test_summary_reports_no_truncated_calls_for_a_run_without_timings(results, run_header):
    assert build_summary(results, run_header)["truncated_calls"] == 0


# ---------------------------------------------------------------------------
# answers forced by the tool-round limit are counted, so the cap's hit rate is tracked
# ---------------------------------------------------------------------------


def test_summary_counts_the_answers_forced_by_the_round_limit(dataset, run_header):
    results = [
        _with_calls(_result(dataset, "Q01", "ok", [GOLD_CITATION]), "tool_calls", "tool_calls", "forced_final"),
        _with_calls(_result(dataset, "Q02", "ok", [GOLD_CITATION]), "tool_calls", "forced_final", "length_retry"),
        _with_calls(_result(dataset, "Q28", "ok", [GOLD_CITATION]), "tool_calls", "stop"),
    ]

    assert build_summary(results, run_header)["round_limit_hits"] == 2


def test_summary_reports_no_round_limit_hits_for_a_run_without_timings(results, run_header):
    assert build_summary(results, run_header)["round_limit_hits"] == 0


def test_report_counts_questions_not_answers_for_a_repeated_subset_run(tmp_path, dataset, results, run_header):
    """--repeat 2 over 2 questions is 4 items; the report must say 2 questions, each asked 2 times."""
    header = _with_only(run_header, "Q01,Q02")
    header["options"] = {**header["options"], "repeat": 2}
    four = [results[0], results[1], results[0], results[1]]
    out = tmp_path / "2026-10-15-repeat"
    write_all(out, dataset=dataset, results=four, run_header=header, domain="documents")

    report = (out / "report.md").read_text(encoding="utf-8")
    summary = build_summary(four, header)

    assert summary["repeat"] == 2
    assert f"selected 2 of {run_header['dataset']['rows']} questions in this set, each asked 2 times" in report


def test_report_of_a_single_pass_subset_does_not_mention_repeats(tmp_path, dataset, results, run_header):
    out = tmp_path / "2026-10-15-single"
    write_all(out, dataset=dataset, results=results, run_header=_with_only(run_header, "Q01,Q02"), domain="documents")

    report = (out / "report.md").read_text(encoding="utf-8")

    assert f"selected {len(results)} of {run_header['dataset']['rows']} questions in this set." in report
    assert "each asked" not in report
