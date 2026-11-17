"""WP6 — the committed run artifacts.

One directory per run, ``$EVAL_RESULTS_DIR/<YYYY-MM-DD>-<run-name>/``, holding:

``results.json``
    the reproducibility record — a ``run`` header (UTC timestamps, git SHA, API
    URL, backend ``version``+``domains``, dataset path/sha256/rows, every CLI
    option, runner version) and one entry per item with the request sent, the
    full answer, the full ``citations[]``, ``usage``, latency, every scorer's
    output and the verdict. This is the file scoring can be re-applied to later.
``results.csv``
    the golden set's own columns, in source order, with the yellow ones filled —
    designed to paste straight back into the golden set's ``.xlsx``.
``report.md``
    the human-readable deliverable: failures first, then one
    section per question showing *why*, verbatim, so a reviewer can overrule the
    machine without opening the JSON.
``summary.json``
    a small flat aggregate, so two runs can be diffed mechanically.

Token counts and cost are recorded as ``null``, never ``0``: the backend's
``usage`` carries ``rounds`` and ``timings`` but no token counts
(``stream_options.include_usage`` is never passed), so a later
fix shows up as a change *from null* rather than as a number that quietly
started being right. Runs recorded before the backend timings existed have no ``timings``; the timing
aggregates are then ``null`` / empty rather than ``0``.
"""

from __future__ import annotations

import csv
import json
import statistics
import subprocess
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from sme_chatbot.eval import RUNNER_VERSION
from sme_chatbot.eval.dataset import (
    COL_FONTI_CORRETTE,
    COL_HA_CITATO,
    COL_ID,
    COL_NOTE_ESITO,
    COL_RISPOSTA_CORRETTA,
    COL_TRACE,
    EvalDataset,
    normalise_header,
)
from sme_chatbot.eval.runner import SCORED_VERDICTS, VERDICT_ORDER, EvalResult

APPENDED_COLUMNS = ("Risposta ottenuta", "Fonti citate", "Latenza (ms)", "Round", "TTFT (ms)")

SESSION_PROTOCOL_NOTE = (
    "Every question was sent as its own request with `history: []`. The API is "
    "stateless, so each item is a fresh session by construction — which satisfies "
    "the legend's \"Q18/Q19 and Q03/Q35/Q36 in separate sessions\" and makes \"run "
    "Q34 last\" moot. It is also a **deviation** from the manual protocol: Q10 and "
    "Q19 are phrased as conversational follow-ups and are harder in isolation than "
    "a human tester found them."
)

LIMITATIONS = (
    "`anchor_recall` is **recall, not correctness**: an answer containing both the "
    "expected value and a spurious one still scores 1.0. The rubric's `forbidden` "
    "column is the counterweight.",
    "Page matching is approximate (`~`) and never gates a verdict: `Pag. stampata` "
    "is the printed page, `Citation.page` is the PDF page, and the offset between "
    "them is an estimate.",
    "Token counts and cost are not available from the backend and are recorded as "
    "`null`, not `0`.",
    "`?` means undetermined — no value anchor and no rubric entry. It is counted "
    "and displayed separately, never folded into a percentage.",
)


def git_sha() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def run_directory(out_root: str | Path, run_name: str, *, now: datetime | None = None) -> Path:
    stamp = (now or datetime.now(UTC)).strftime("%Y-%m-%d")
    return Path(out_root) / f"{stamp}-{run_name}"


# ---------------------------------------------------------------------------
# results.json
# ---------------------------------------------------------------------------


def build_run_header(
    *,
    dataset: EvalDataset,
    api_url: str,
    domain: str,
    health: dict,
    options: dict,
    started_at: datetime,
    finished_at: datetime,
    llm_model: str | None = None,
    backend_config: dict | None = None,
    judge_info: dict | None = None,
) -> dict:
    header = {
        "runner_version": RUNNER_VERSION,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "git_sha": git_sha(),
        "api_url": api_url,
        "domain": domain,
        "backend_version": health.get("version"),
        "backend_domains": health.get("domains"),
        "llm_model": llm_model,
        "backend_config": backend_config,
        "dataset": dataset.describe(),
        "options": options,
        "session_protocol": "isolated: every item sent with history: []",
        "judge": judge_info,
    }
    return header


def _scores_payload(result: EvalResult) -> dict | None:
    if result.scores is None:
        return None
    scores = result.scores
    return {
        "anchor_recall": scores.anchors.score,
        "anchors_matched": scores.anchors.matched,
        "anchors_missing": scores.anchors.missing,
        "forbidden_hits": scores.forbidden_hits,
        "refusal_detected": scores.refusal_detected,
        "refusal_span": scores.refusal_span,
        "clarification_detected": scores.clarification_detected,
        "clarification_span": scores.clarification_span,
        "qualification_detected": scores.qualification_detected,
        "qualification_span": scores.qualification_span,
        "premise_correction_detected": scores.premise_correction_detected,
        "premise_correction_span": scores.premise_correction_span,
        "gold_doc_cited": scores.citations.gold_doc_cited,
        "citation_noise": asdict(scores.citations),
        "page_match_approx": asdict(scores.pages),
        "has_inline_citation": scores.has_inline_citation,
        "inline_citation_span": scores.inline_citation_span,
        "answer_language": scores.answer_language,
        "language_match": scores.language_match,
        "verdict_reasons": scores.verdict_reasons,
    }


def _item_payload(result: EvalResult, *, domain: str) -> dict:
    item = result.item
    return {
        "run_key": result.run_key,
        "id": item.id,
        "repeat_index": result.repeat_index,
        "request": {
            "domain": domain,
            "message": item.domanda,
            "history": [],
            "stream": False,
        },
        "expected": {
            "categoria": item.categoria,
            "difficolta": item.difficolta,
            "risposta_attesa": item.risposta_attesa,
            "comportamento_atteso": item.comportamento_atteso,
            "capitolo_fonte": item.capitolo_fonte,
            "pag_stampata": item.pag_stampata,
            "documento": item.documento,
            "serial": item.serial,
            "language": item.language,
            "expects_refusal": item.expects_refusal,
            "expects_premise_correction": item.expects_premise_correction,
            "expects_disambiguation": item.expects_disambiguation,
            "requires_qualification": item.requires_qualification,
            "gold_pages": sorted(item.gold_pages),
            "value_anchors": [a.describe() for a in item.value_anchors],
            "rubric_required": list(item.required),
            "rubric_forbidden": list(item.forbidden),
            "rubric_note": item.rubric_note,
        },
        "response": {
            "content": result.answer,
            "citations": result.citations,
            "usage": result.usage,
            "status_code": result.status_code,
            "attempts": result.attempts,
        },
        "latency_ms": result.latency_ms,
        "rounds": result.rounds,
        "prompt_tokens": None,
        "completion_tokens": None,
        "cost": None,
        "trace_id": result.trace_id,
        "error": result.error,
        "scores": _scores_payload(result),
        "verdict_auto": result.verdict_auto,
        "verdict_judge": result.verdict_judge,
        "judge_reason": result.judge_reason,
        "verdict": result.verdict,
    }


def load_results_json(path: Path, dataset: EvalDataset) -> tuple[dict, list[EvalResult], list[str]]:
    """Rebuild a run's answers from a committed ``results.json`` (plan §5 step 3).

    This is what lets the baseline ship *before* scoring exists and be scored
    retroactively: the captured answers and citations are the expensive part, and
    they are all in the file. Returns ``(run_header, results, warnings)``.

    The dataset a rescore runs against is not necessarily the one the run used,
    so the recorded ``sha256`` is compared and any drift is reported rather than
    silently absorbed (plan §7 R4). Scores are deliberately *not* restored —
    rescoring is the point.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    run_header = payload.get("run") or {}
    warnings: list[str] = []

    recorded = (run_header.get("dataset") or {}).get("sha256")
    if recorded and recorded != dataset.sha256:
        warnings.append(
            f"answer key changed since the run: results.json records sha256 {recorded[:12]}, "
            f"{dataset.path} is now {dataset.sha256[:12]} - verdicts are not comparable with the original"
        )

    by_id = {item.id: item for item in dataset.items}
    results: list[EvalResult] = []
    for entry in payload.get("items") or []:
        item = by_id.get(entry.get("id"))
        if item is None:
            warnings.append(f"{entry.get('run_key') or entry.get('id')}: no such question in {dataset.path.name}")
            continue
        response = entry.get("response") or {}
        results.append(
            EvalResult(
                item=item,
                run_key=entry.get("run_key") or item.id,
                repeat_index=entry.get("repeat_index") or 1,
                answer=response.get("content") or "",
                citations=list(response.get("citations") or []),
                usage=dict(response.get("usage") or {}),
                latency_ms=entry.get("latency_ms") or 0,
                status_code=response.get("status_code"),
                attempts=response.get("attempts") or 0,
                trace_id=entry.get("trace_id"),
                error=entry.get("error"),
            )
        )
    return run_header, results, warnings


def write_results_json(path: Path, run_header: dict, results: Sequence[EvalResult], *, domain: str) -> Path:
    payload = {
        "run": run_header,
        "limitations": list(LIMITATIONS),
        "items": [_item_payload(r, domain=domain) for r in results],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# results.csv
# ---------------------------------------------------------------------------


def _note_esito(result: EvalResult) -> str:
    if result.error:
        return f"ERROR: {result.error}"
    if result.scores is None:
        return ""
    parts = list(result.scores.verdict_reasons)
    if result.verdict_judge:
        parts.append(f"judge={result.verdict_judge}: {result.judge_reason or ''}".strip())
    return " | ".join(p for p in parts if p)


def write_results_csv(path: Path, dataset: EvalDataset, results: Sequence[EvalResult]) -> Path:
    """Write the source columns verbatim, with the yellow ones filled.

    Column order and header text come from the loaded file, not from a constant
    here, so a re-export of the workbook keeps pasting back cleanly.
    """
    fieldnames = list(dataset.columns) + [c for c in APPENDED_COLUMNS if c not in dataset.columns]
    # Write into the header the *file* uses, resolved the same accent- and
    # case-insensitive way the loader reads it. Hardcoding "ID" here would make
    # a re-exported workbook silently produce empty verdict columns rather than
    # fail, which is the worst way for this to go wrong.
    actual = {normalise_header(name): name for name in fieldnames}

    def column(canonical: str) -> str:
        return actual.get(normalise_header(canonical), canonical)

    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for result in results:
            row = dict(result.item.raw)
            row[column(COL_ID)] = result.run_key
            row[column(COL_RISPOSTA_CORRETTA)] = result.verdict
            row[column(COL_FONTI_CORRETTE)] = "" if result.error else result.sources_verdict
            row[column(COL_HA_CITATO)] = _ha_citato(result)
            row[column(COL_NOTE_ESITO)] = _note_esito(result)
            row[column(COL_TRACE)] = result.trace_id or ""
            row["Risposta ottenuta"] = result.answer
            row["Fonti citate"] = "; ".join(result.cited_documents)
            row["Latenza (ms)"] = result.latency_ms
            row["Round"] = "" if result.rounds is None else result.rounds
            ttft = result.timing("ttft_ms")
            row["TTFT (ms)"] = "" if ttft is None else ttft
            writer.writerow(row)
    return path


def _ha_citato(result: EvalResult) -> str:
    if result.error or result.scores is None:
        return ""
    return "S" if result.scores.has_inline_citation else "N"


# ---------------------------------------------------------------------------
# summary.json
# ---------------------------------------------------------------------------


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return float(ordered[index])


def _mean(values: Sequence[float], digits: int = 1) -> float | None:
    return round(statistics.fmean(values), digits) if values else None


def _timing_values(results: Sequence[EvalResult], key: str) -> list[float]:
    return [float(v) for r in results if (v := r.timing(key)) is not None]


def latency_by_rounds(results: Sequence[EvalResult]) -> dict[str, dict]:
    """Per tool-round count: how many answers and their median end-to-end latency.

    The view that showed the round cap is the slow tail. Keys are strings
    so the mapping survives a JSON round trip unchanged.
    """
    grouped: dict[int, list[float]] = {}
    for r in results:
        if not r.error and r.rounds is not None:
            grouped.setdefault(r.rounds, []).append(float(r.latency_ms))
    return {
        str(rounds): {"n": len(values), "median_ms": statistics.median(values)}
        for rounds, values in sorted(grouped.items())
    }


def build_summary(results: Sequence[EvalResult], run_header: dict) -> dict:
    counts = Counter(r.verdict for r in results)
    answered = [r for r in results if not r.error]
    latencies = [float(r.latency_ms) for r in answered]
    rounds = [float(r.rounds) for r in answered if r.rounds is not None]
    citation_counts = [float(len(r.citations)) for r in answered]
    recalls = [r.scores.anchors.score for r in answered if r.scores and r.scores.anchors.score is not None]
    scored = sum(counts.get(v, 0) for v in SCORED_VERDICTS)

    return {
        "runner_version": RUNNER_VERSION,
        "started_at": run_header.get("started_at"),
        "api_url": run_header.get("api_url"),
        "backend_version": run_header.get("backend_version"),
        "domain": run_header.get("domain"),
        "dataset_sha256": (run_header.get("dataset") or {}).get("sha256"),
        "n_items": len(results),
        # A subset run (``--only``) must never be mistaken for a full one: the
        # spec and the size of the question set are part of the summary.
        "only": (run_header.get("options") or {}).get("only"),
        "dataset_rows": (run_header.get("dataset") or {}).get("rows"),
        "repeat": (run_header.get("options") or {}).get("repeat") or 1,
        "verdicts": {v: counts.get(v, 0) for v in VERDICT_ORDER},
        # Denominator is S+P+N only: `?` was never determined and `ERROR` never
        # got an answer, so neither belongs in a percentage.
        "scored_items": scored,
        "pct_correct_of_scored": round(100.0 * counts.get("S", 0) / scored, 1) if scored else None,
        "mean_latency_ms": round(statistics.fmean(latencies), 1) if latencies else None,
        "p50_latency_ms": _percentile(latencies, 0.5),
        "p95_latency_ms": _percentile(latencies, 0.95),
        # Backend-side stage timings; null when the backend predates them.
        "p50_ttft_ms": _percentile(_timing_values(answered, "ttft_ms"), 0.5),
        "p95_ttft_ms": _percentile(_timing_values(answered, "ttft_ms"), 0.95),
        "mean_setup_ms": _mean(_timing_values(answered, "setup_ms")),
        "mean_llm_ms": _mean(_timing_values(answered, "llm_ms")),
        "mean_tool_ms": _mean(_timing_values(answered, "tool_ms")),
        "latency_by_rounds": latency_by_rounds(results),
        "mean_rounds": round(statistics.fmean(rounds), 2) if rounds else None,
        # LLM calls that ended on max_tokens (finish "length"): answers that may be missing or cut short.
        "truncated_calls": sum(
            1 for r in answered for call in (r.usage.get("timings") or {}).get("llm_calls", []) if call.get("finish") == "length"
        ),
        # Turns that ran out of tool rounds and had their answer forced. Counted from the
        # "forced_final" call rather than usage["round_limit_hit"], so runs recorded before that flag count too.
        "round_limit_hits": sum(
            1 for r in answered if any(call.get("finish") == "forced_final" for call in (r.usage.get("timings") or {}).get("llm_calls", []))
        ),
        "mean_citations": round(statistics.fmean(citation_counts), 2) if citation_counts else None,
        "mean_anchor_recall": round(statistics.fmean(recalls), 3) if recalls else None,
        "gold_doc_cited": sum(1 for r in answered if r.scores and r.scores.citations.gold_doc_cited),
        "inline_citation": sum(1 for r in answered if r.scores and r.scores.has_inline_citation),
        "errors": len(results) - len(answered),
        "mean_prompt_tokens": None,
        "mean_completion_tokens": None,
        "total_cost": None,
    }


def _stage_means(summary: dict) -> str | None:
    parts = [summary.get(k) for k in ("mean_setup_ms", "mean_llm_ms", "mean_tool_ms")]
    if all(p is None for p in parts):
        return None
    return " / ".join("—" if p is None else f"{p:.0f}" for p in parts)


def write_summary_json(path: Path, summary: dict) -> Path:
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# report.md
# ---------------------------------------------------------------------------

_VERDICT_LABEL = {
    "S": "S (corretta)",
    "P": "P (parziale)",
    "N": "N (errata)",
    "?": "? (indeterminata)",
    "ERROR": "ERROR (nessuna risposta)",
}


def _md_escape(text: str) -> str:
    """Make free text safe inside a Markdown table cell."""
    return text.replace("|", "\\|").replace("\n", " ").strip()


def _counts_table(title: str, grouped: dict[str, Counter]) -> list[str]:
    """A verdict breakdown table, one row per group, in key order."""
    lines = [
        f"### {title}",
        "",
        f"| {title} | S | P | N | ? | ERROR | tot |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in sorted(grouped):
        row = grouped[key]
        cells = " | ".join(str(row.get(v, 0)) for v in VERDICT_ORDER)
        lines.append(f"| {_md_escape(key) or '—'} | {cells} | {sum(row.values())} |")
    lines.append("")
    return lines


def _summary_line(counts: Counter) -> str:
    return " · ".join(f"{v} {counts.get(v, 0)}" for v in VERDICT_ORDER)


def _anchor_detail(result: EvalResult) -> str:
    if result.scores is None:
        return "—"
    anchors = result.scores.anchors
    if anchors.score is None:
        return "nessun anchor (voce non verificabile automaticamente)"
    matched = ", ".join(f"`{a}`" for a in anchors.matched) or "—"
    missing = ", ".join(f"`{a}`" for a in anchors.missing) or "—"
    return f"trovati: {matched}<br>mancanti: {missing}"


def _page_detail(result: EvalResult) -> str:
    if result.scores is None:
        return "—"
    pages = result.scores.pages
    if pages.score is None:
        return "—  (nessuna pagina nel set)"
    return (
        f"~ {len(pages.matched)}/{len(pages.gold_pages)} "
        f"(stampata {pages.gold_pages} → PDF attesa {pages.expected_pdf_pages} "
        f"±{pages.tolerance}; citate {pages.cited_pages or '—'})"
    )


def render_report_md(
    run_header: dict,
    results: Sequence[EvalResult],
    summary: dict,
    *,
    dataset: EvalDataset,
) -> str:
    counts = Counter(r.verdict for r in results)
    lines: list[str] = []
    add = lines.append

    add("# Eval run report")
    add("")
    add(f"**{_summary_line(counts)}** over {len(results)} asked questions.")
    add("")

    add("## Run")
    add("")
    add("| Field | Value |")
    add("| --- | --- |")
    rows = [
        ("Started (UTC)", run_header.get("started_at")),
        ("Finished (UTC)", run_header.get("finished_at")),
        ("Runner version", run_header.get("runner_version")),
        ("Repo commit", run_header.get("git_sha")),
        ("API URL", run_header.get("api_url")),
        ("Backend version", run_header.get("backend_version")),
        ("Backend domains", ", ".join(run_header.get("backend_domains") or []) or None),
        ("Domain", run_header.get("domain")),
        ("LLM model (backend config)", run_header.get("llm_model")),
        ("LLM base URL (backend config)", (run_header.get("backend_config") or {}).get("llm_base_url")),
        ("Auth mode (backend config)", (run_header.get("backend_config") or {}).get("auth_mode")),
        ("Dataset", (run_header.get("dataset") or {}).get("path")),
        ("Dataset sha256", (run_header.get("dataset") or {}).get("sha256")),
        ("Dataset rows", (run_header.get("dataset") or {}).get("rows")),
        ("Rubric", (run_header.get("dataset") or {}).get("rubric_path") or "none"),
    ]
    for label, value in rows:
        add(f"| {label} | {_md_escape(str(value)) if value is not None else '—'} |")
    # Say out loud how much of this report rests on the derived rubric rather
    # than on the as-received answer key, so the dependency cannot hide.
    rubric_only = sum(1 for r in results if r.item.required and not r.item.value_anchors)
    if rubric_only:
        add(
            f"| Items scored only via the rubric | {rubric_only} of {len(results)} "
            f"(re-run with `--no-rubric` to see them as `?`) |"
        )
    options = run_header.get("options") or {}
    add(f"| CLI options | `{_md_escape(json.dumps(options, ensure_ascii=False))}` |")
    judge_info = run_header.get("judge")
    if judge_info:
        add(f"| Judge | `{_md_escape(json.dumps(judge_info, ensure_ascii=False))}` |")
    add("")

    add("## How to read this report")
    add("")
    add(
        "Verdicts use the golden set's own vocabulary (`legenda`): **S** correct, "
        "**P** partial, **N** wrong — plus **?** *undetermined* and **ERROR** "
        "(transport failure), which the workbook does not have."
    )
    add("")
    for limitation in LIMITATIONS:
        add(f"- {limitation}")
    add("")
    add(f"**Session protocol.** {SESSION_PROTOCOL_NOTE}")
    add("")
    if judge_info:
        add(
            "**Judge.** An LLM judge resolved the `?` items only; it never overwrote an "
            "anchored S/N and never saw the deterministic verdict. It runs on the same "
            "model family as the chatbot under test — a known weak oracle — and judge "
            "runs are not reproducible. Disagreements are listed below and are the "
            "human review queue."
        )
        add("")
        judge_errors = judge_info.get("judge_errors") or 0
        if judge_errors:
            # A judge that failed on every call must not read as one that ran.
            filtered = judge_info.get("judge_content_filtered") or 0
            # A content-filter block is deterministic and needs a different fix
            # (provider policy) from a transient failure, so it is counted apart.
            filtered_note = f" ({filtered} blocked by the LLM provider's content filter)" if filtered else ""
            add(
                f"**Judge errors: {judge_errors} of {judge_info.get('judge_calls')} judge calls failed**"
                f"{filtered_note}; those items keep their deterministic verdict "
                "(see `judge_reason` in `results.json`)."
            )
            add("")
    add(
        "This runner does not decide whether the chatbot is good enough. It produces a "
        "reviewable record; thresholds are a separate decision."
    )
    add("")

    add("## Summary")
    add("")
    add("| Verdict | Count |")
    add("| --- | ---: |")
    for verdict in VERDICT_ORDER:
        add(f"| {_VERDICT_LABEL[verdict]} | {counts.get(verdict, 0)} |")
    add(f"| **Total asked** | **{len(results)}** |")
    add("")
    add(
        f"Percentages are over the **{summary.get('scored_items', 0)} scored** items "
        f"(S+P+N); `?` and `ERROR` are excluded by construction."
    )
    add("")
    if summary.get("only"):
        # n_items counts every answer, repeats included: with --repeat 2, 9 questions are 18 items.
        repeat = summary.get("repeat") or 1
        questions = (summary.get("n_items") or 0) // repeat
        asked = f", each asked {repeat} times" if repeat > 1 else ""
        add(
            f"**Subset run.** `--only {summary['only']}` selected {questions} of "
            f"{summary.get('dataset_rows', '?')} questions in this set{asked}. It is a fast check, not a "
            f"full run: do not compare its percentages with a full run's."
        )
        add("")
    metric_rows = [
        ("Correct, % of scored", summary.get("pct_correct_of_scored")),
        ("Mean anchor recall", summary.get("mean_anchor_recall")),
        ("Gold document cited", f"{summary.get('gold_doc_cited')}/{len(results)}"),
        ("Inline citation present", f"{summary.get('inline_citation')}/{len(results)}"),
        ("Mean latency (ms)", summary.get("mean_latency_ms")),
        ("p50 latency (ms)", summary.get("p50_latency_ms")),
        ("p95 latency (ms)", summary.get("p95_latency_ms")),
        ("p50 time to first token (ms)", summary.get("p50_ttft_ms")),
        ("p95 time to first token (ms)", summary.get("p95_ttft_ms")),
        ("Mean setup / LLM / tool time (ms)", _stage_means(summary)),
        ("Mean tool rounds", summary.get("mean_rounds")),
        ("LLM calls cut off by max_tokens", summary.get("truncated_calls")),
        ("Answers forced by the tool-round limit", summary.get("round_limit_hits")),
        ("Mean citations per answer", summary.get("mean_citations")),
        ("Tokens / cost", "not available from the backend (`null`)"),
    ]
    add("| Metric | Value |")
    add("| --- | ---: |")
    for label, value in metric_rows:
        add(f"| {label} | {value if value is not None else '—'} |")
    add("")

    by_rounds = summary.get("latency_by_rounds") or {}
    if by_rounds:
        add("### Latency by tool rounds")
        add("")
        add("| Rounds | Answers | Median latency (ms) |")
        add("| ---: | ---: | ---: |")
        for rounds, row in by_rounds.items():
            add(f"| {rounds} | {row['n']} | {row['median_ms']:.0f} |")
        add("")

    grouped_category: dict[str, Counter] = {}
    grouped_difficulty: dict[str, Counter] = {}
    for result in results:
        grouped_category.setdefault(result.item.categoria, Counter())[result.verdict] += 1
        key = str(result.item.difficolta) if result.item.difficolta is not None else "—"
        grouped_difficulty.setdefault(key, Counter())[result.verdict] += 1

    lines.extend(_counts_table("Categoria", grouped_category))
    lines.extend(_counts_table("Difficoltà", grouped_difficulty))

    disagreements = [r for r in results if r.disagreement]
    if disagreements:
        add("## Disagreements (human review queue)")
        add("")
        add("| ID | Deterministic | Judge | Judge reason |")
        add("| --- | --- | --- | --- |")
        for result in disagreements:
            add(
                f"| {result.run_key} | {result.verdict_auto} | {result.verdict_judge} | "
                f"{_md_escape(result.judge_reason or '')} |"
            )
        add("")

    failures = [r for r in results if r.verdict in ("N", "ERROR")]
    partials = [r for r in results if r.verdict == "P"]
    if failures or partials:
        add("## Failures first")
        add("")
        add("| ID | Verdict | Categoria | Domanda | Why |")
        add("| --- | --- | --- | --- | --- |")
        for result in failures + partials:
            why = result.error or "; ".join(result.scores.verdict_reasons if result.scores else [])
            add(
                f"| {result.run_key} | {result.verdict} | {_md_escape(result.item.categoria)} | "
                f"{_md_escape(result.item.domanda)} | {_md_escape(why)} |"
            )
        add("")

    add("## Per question")
    add("")
    for result in results:
        item = result.item
        add(f"### {result.run_key} — {_md_escape(item.categoria)} (difficoltà {item.difficolta or '—'})")
        add("")
        add(f"**Verdetto: {result.verdict}**" + (f" · judge: {result.verdict_judge}" if result.verdict_judge else ""))
        add("")
        add(f"- **Domanda** — {item.domanda}")
        add(f"- **Risposta attesa** — {item.risposta_attesa}")
        add(f"- **Comportamento atteso** — {item.comportamento_atteso}")
        add("")
        if result.error:
            add(f"> **ERROR** — {result.error}")
            add("")
            continue
        add("**Risposta ottenuta**")
        add("")
        add("> " + (result.answer.replace("\n", "\n> ") if result.answer.strip() else "*(risposta vuota)*"))
        add("")
        add("| Check | Esito |")
        add("| --- | --- |")
        add(f"| Anchor | {_anchor_detail(result)} |")
        if result.scores and result.scores.forbidden_hits:
            add(f"| Forbidden | {_md_escape('; '.join(result.scores.forbidden_hits))} |")
        if item.expects_refusal:
            add(f"| Rifiuto atteso | {'sì' if result.scores and result.scores.refusal_detected else 'NO'} |")
        if item.expects_premise_correction:
            detected = result.scores and result.scores.premise_correction_detected
            add(f"| Correzione premessa | {'sì' if detected else 'NO'} |")
        if item.requires_qualification:
            detected = result.scores and result.scores.disambiguation_detected
            add(f"| Disambiguazione/qualificazione | {'sì' if detected else 'NO'} |")
        add(f"| Fonti corrette | {result.sources_verdict} (gold: `{_md_escape(item.documento)}`) |")
        add(f"| Fonti citate | {_md_escape('; '.join(result.cited_documents) or '—')} |")
        add(f"| Citazione inline | {_ha_citato(result) or '—'} |")
        add(f"| Pagine ~ | {_md_escape(_page_detail(result))} |")
        add(f"| Lingua | {result.scores.answer_language if result.scores else '—'} (attesa {item.language}) |")
        add(f"| Latenza / round | {result.latency_ms} ms / {result.rounds if result.rounds is not None else '—'} |")
        if result.trace_id:
            add(f"| Trace | `{result.trace_id}` |")
        add("")
        if result.scores and result.scores.verdict_reasons:
            add("*Perché:* " + _md_escape("; ".join(result.scores.verdict_reasons)))
            add("")
        if result.judge_reason:
            add(f"*Judge ({result.verdict_judge}):* {_md_escape(result.judge_reason)}")
            add("")

    add("---")
    add("")
    add(
        f"Generated by `sme-chatbot-eval` {RUNNER_VERSION} from "
        f"`{dataset.path.as_posix()}` (sha256 `{dataset.sha256[:16]}…`)."
    )
    add("")
    return "\n".join(lines)


def write_report_md(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# one call to write them all
# ---------------------------------------------------------------------------


def write_all(
    out_dir: Path,
    *,
    dataset: EvalDataset,
    results: Sequence[EvalResult],
    run_header: dict,
    domain: str,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_summary(results, run_header)
    paths = {
        "results.json": write_results_json(out_dir / "results.json", run_header, results, domain=domain),
        "results.csv": write_results_csv(out_dir / "results.csv", dataset, results),
        "summary.json": write_summary_json(out_dir / "summary.json", summary),
        "report.md": write_report_md(
            out_dir / "report.md",
            render_report_md(run_header, results, summary, dataset=dataset),
        ),
    }
    return paths
