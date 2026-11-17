"""``sme-chatbot-eval-trend``: run grouping, scope, aggregates and output (no network)."""

from __future__ import annotations

import json
import re

from sme_chatbot.eval import trend


def _item(qid, verdict, latency_ms=1000, ttft_ms=None, error=None):
    usage = {"rounds": 2}
    if ttft_ms is not None:
        usage["timings"] = {"ttft_ms": ttft_ms}
    return {
        "id": qid,
        "verdict": verdict,
        "latency_ms": latency_ms,
        "error": error,
        "response": None if error else {"usage": usage},
    }


def _write_run(root, dirname, items, *, started_at, run_name, only=None, rows=None, model="qwen3.8:27b", **cfg):
    d = root / dirname
    d.mkdir()
    header = {
        "started_at": started_at,
        "llm_model": model,
        "backend_version": "dev",
        "git_sha": "abcdef1234",
        "options": {"run_name": run_name, "only": only, "repeat": 1},
        "dataset": {"rows": rows if rows is not None else len(items)},
        "backend_config": {"model": model, **cfg},
    }
    (d / "results.json").write_text(json.dumps({"run": header, "items": items}), encoding="utf-8")


def test_per_set_directories_of_one_run_are_merged_into_one_point(tmp_path):
    common = dict(started_at="2026-10-16T10:00:00+00:00", run_name="full")
    _write_run(tmp_path, "2026-10-16-full-eval-chatbot-alpha", [_item("Q1", "S"), _item("Q2", "N")], rows=116, **common)
    _write_run(tmp_path, "2026-10-16-full-eval-chatbot-beta", [_item("Q143", "P")], rows=30, **common)
    _write_run(tmp_path, "2026-10-16-full-eval-chatbot-gamma", [_item("Q117", "?")], rows=26, **common)

    [run] = trend.load_runs(tmp_path)

    assert run.key == "2026-10-16-full"
    assert run.sets == ["eval-chatbot-alpha", "eval-chatbot-beta", "eval-chatbot-gamma"]
    assert run.scope == "all 172 questions"
    assert run.n_answers == 4
    assert run.verdicts == {"S": 1, "P": 1, "N": 1, "?": 1, "ERROR": 0}
    assert run.pct_correct_of_scored == 33.3  # S of S+P+N, like summary.json


def test_a_single_directory_run_from_before_the_split_is_its_own_point(tmp_path):
    _write_run(tmp_path, "2026-09-16-baseline", [_item("Q1", "N")], started_at="2026-09-16T08:00:00", run_name="baseline", rows=36)

    [run] = trend.load_runs(tmp_path)

    assert (run.key, run.run_name, run.scope, run.sets) == ("2026-09-16-baseline", "baseline", "all 36 questions", ["-"])


def test_scope_names_the_only_list(tmp_path):
    _write_run(
        tmp_path, "2026-10-15-smoke-eval-chatbot-alpha", [_item("Q1", "S")],
        started_at="2026-10-15T08:00:00", run_name="smoke", only="@data/eval/datasets/derived/smoke-ids.txt", rows=116,
    )
    _write_run(tmp_path, "2026-10-15-two-eval-chatbot-alpha", [_item("Q1", "S")], started_at="2026-10-15T09:00:00", run_name="two", only="Q01,Q05")

    assert [r.scope for r in trend.load_runs(tmp_path)] == ["smoke-ids", "only Q01,Q05"]


def test_percentiles_come_from_the_answers_not_from_averaged_summaries(tmp_path):
    # alpha: 1 s, 2 s, 3 s; beta: 10 s. Averaging the two sets' medians would give 6 s.
    common = dict(started_at="2026-10-16T10:00:00", run_name="full")
    _write_run(tmp_path, "2026-10-16-full-eval-chatbot-alpha",
               [_item("Q1", "S", 1000, 500), _item("Q2", "S", 2000, 900), _item("Q3", "S", 3000, 1500)], **common)
    _write_run(tmp_path, "2026-10-16-full-eval-chatbot-beta", [_item("Q143", "S", 10000, 8000)], **common)

    [run] = trend.load_runs(tmp_path)

    assert run.p50_latency_s == 3.0  # same rounding rule as report._percentile: index round(0.5 * 3) = 2
    assert run.p95_latency_s == 10.0
    assert run.p50_ttft_s == 1.5


def test_errors_count_as_verdicts_but_not_in_latency_and_old_runs_have_no_ttft(tmp_path):
    _write_run(tmp_path, "2026-09-18-remote",
               [_item("Q1", "S", 2000), _item("Q2", "ERROR", 99000, error="timeout")],
               started_at="2026-09-18T08:00:00", run_name="remote")

    [run] = trend.load_runs(tmp_path)

    assert run.verdicts["ERROR"] == 1
    assert run.p95_latency_s == 2.0
    assert run.p50_ttft_s is None


def test_runs_are_ordered_by_start_time_and_settings_keep_only_what_is_set(tmp_path):
    _write_run(tmp_path, "2026-10-17-b", [_item("Q1", "S")], started_at="2026-10-17T12:00:00", run_name="b",
               temperature=0.2, seed=42, pre_search=True, search_budget=None, reasoning_effort="none")
    _write_run(tmp_path, "2026-10-17-a", [_item("Q1", "S")], started_at="2026-10-17T09:00:00", run_name="a", pre_search=False)

    runs = trend.load_runs(tmp_path)

    assert [r.run_name for r in runs] == ["a", "b"]
    assert runs[0].settings == {}
    assert runs[1].settings == {"temperature": 0.2, "seed": 42, "reasoning_effort": "none", "pre_search": True}


def test_select_filters_by_scope_match_and_since(tmp_path):
    _write_run(tmp_path, "2026-09-16-baseline", [_item("Q1", "S")], started_at="2026-09-16T08:00:00", run_name="baseline", rows=36)
    _write_run(tmp_path, "2026-10-16-full-eval-chatbot-alpha", [_item("Q1", "S")], started_at="2026-10-16T08:00:00", run_name="full", rows=172)
    runs = trend.load_runs(tmp_path)

    assert [r.run_name for r in trend.select(runs, scope="all 172 questions")] == ["full"]
    assert [r.run_name for r in trend.select(runs, match="baseline")] == ["baseline"]
    assert [r.run_name for r in trend.select(runs, since="2026-10-01")] == ["full"]


def test_directories_without_results_json_are_skipped(tmp_path):
    (tmp_path / "README.md").write_text("x")
    (tmp_path / "2026-10-17-half-written").mkdir()
    _write_run(tmp_path, "2026-10-17-ok", [_item("Q1", "S")], started_at="2026-10-17T08:00:00", run_name="ok")

    assert [r.run_name for r in trend.load_runs(tmp_path)] == ["ok"]


def test_html_embeds_the_runs_and_targets_and_escapes_a_closing_script_tag(tmp_path):
    _write_run(tmp_path, "2026-10-17-x", [_item("Q1", "S", 4000, 2000)], started_at="2026-10-17T08:00:00", run_name="x</script>")
    page = trend.render_html(trend.load_runs(tmp_path), title="T & co", targets={"p50_ttft": 5.0})

    assert page.startswith("<title>T &amp; co</title>")
    data = re.search(r"var RUNS = (.*?), TARGETS = (.*?);\n", page)
    assert json.loads(data.group(1).replace("<\\/", "</"))[0]["run_name"] == "x</script>"
    assert json.loads(data.group(2)) == {"p50_ttft": 5.0}
    assert page.count("</script>") == 1  # only the template's own


def test_markdown_table_has_one_row_per_run_with_shares(tmp_path):
    _write_run(tmp_path, "2026-10-17-x", [_item("Q1", "S"), _item("Q2", "N")], started_at="2026-10-17T08:00:00", run_name="x")

    lines = trend.render_markdown(trend.load_runs(tmp_path)).splitlines()

    assert len(lines) == 3
    assert "| 2026-10-17 | `x` | all 2 questions | qwen3.8:27b | 2 | 1 (50%) | 0 (0%) | 1 (50%) |" in lines[2]


def test_cli_writes_the_page_and_rejects_a_bad_target(tmp_path, capsys):
    _write_run(tmp_path, "2026-10-17-x", [_item("Q1", "S")], started_at="2026-10-17T08:00:00", run_name="x")
    out = tmp_path / "trend.html"

    assert trend.main(["--results-dir", str(tmp_path), "--out", str(out), "--target", "p50_ttft=5", "--markdown"]) == 0
    assert "var RUNS" in out.read_text(encoding="utf-8")
    assert "| Date |" in capsys.readouterr().out

    assert trend.main(["--results-dir", str(tmp_path), "--out", str(out), "--target", "ttft=5"]) == 2
    assert trend.main(["--results-dir", str(tmp_path), "--out", str(out), "--scope", "nothing"]) == 1


def test_llm_api_shows_in_settings_only_when_not_the_default(tmp_path):
    _write_run(tmp_path, "2026-11-04-a", [_item("Q1", "S")], started_at="2026-11-04T09:00:00", run_name="a",
               llm_api="chat_completions")
    _write_run(tmp_path, "2026-11-04-b", [_item("Q1", "S")], started_at="2026-11-04T12:00:00", run_name="b",
               llm_api="responses")

    runs = trend.load_runs(tmp_path)

    assert [r.settings for r in runs] == [{}, {"llm_api": "responses"}]
