"""Trend chart over the committed eval runs (``$EVAL_RESULTS_DIR``).

    uv run sme-chatbot-eval-trend --out eval-trend.html
    uv run sme-chatbot-eval-trend --scope "all 172 questions" --markdown

Reads every run record under ``--results-dir`` and writes one self-contained HTML
page: the share of S / P / N / ? / ERROR per run as stacked columns, the p50 and
p95 end-to-end latency and time to first token per run, and a table with the
numbers behind both. ``--markdown`` prints the table for an issue comment.

**One run, several directories.** Since 2026-10-14 a run writes one directory per
question set (``<date>-<run-name>-eval-chatbot-alpha``, ``-beta``, ...); those are
merged into one point. Percentiles are recomputed from the per-answer latencies
with the same helper as ``summary.json``, never averaged across sets.

**Comparability.** Runs ask different questions: the 36-question 90001 set, all 172
questions, the smoke list, the cap-hit lists. Each point carries a *scope* (the
question set it asked, from ``--only`` and the dataset size) and the chart marks
every change of scope with a dashed line. Two runs are comparable only within one
scope, and only on the same model; ``--scope`` keeps one.

Counts are per answer: a ``--repeat 2`` run counts each question twice.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from sme_chatbot import config
from sme_chatbot.eval.report import _percentile

VERDICTS = ("S", "P", "N", "?", "ERROR")

# `<date>-<run-name>-eval-chatbot-<set>`: one directory per question set of one run.
_SET_DIR_RE = re.compile(r"^(?P<run>\d{4}-\d{2}-\d{2}-.+?)-(?P<set>eval-chatbot-[a-z0-9-]+)$")


@dataclass
class RunPoint:
    """One eval run, merged across its per-set directories."""

    key: str  # `<date>-<run-name>`
    started_at: str
    run_name: str
    scope: str
    model: str | None
    backend_version: str | None
    git_sha: str | None
    repeat: int
    sets: list[str] = field(default_factory=list)
    dirs: list[str] = field(default_factory=list)
    n_answers: int = 0
    verdicts: dict[str, int] = field(default_factory=dict)
    pct_correct_of_scored: float | None = None
    p50_latency_s: float | None = None
    p95_latency_s: float | None = None
    p50_ttft_s: float | None = None
    p95_ttft_s: float | None = None
    settings: dict = field(default_factory=dict)

    @property
    def date(self) -> str:
        return self.started_at[:10]


def _scope(only: str | None, total_rows: int) -> str:
    """The question set a run asked: the `--only` list's name, or the whole dataset."""
    if not only:
        return f"all {total_rows} questions"
    if only.startswith("@"):
        return Path(only[1:]).stem
    return f"only {only}"


def _settings(backend_config: dict | None) -> dict:
    """The backend settings that change answers, as reported by `/config` (absent in older runs)."""
    keep = ("llm_api", "temperature", "seed", "max_tokens", "reasoning_effort", "search_budget", "pre_search")
    cfg = backend_config or {}
    # llm_api is shown only when it is not the default, like the unset settings.
    return {k: cfg[k] for k in keep if cfg.get(k) not in (None, False, "chat_completions")}


def _seconds(ms: float | None) -> float | None:
    return None if ms is None else round(ms / 1000, 1)


def load_runs(results_dir: Path) -> list[RunPoint]:
    """Every run under *results_dir*, oldest first. Directories without `results.json` are skipped."""
    groups: dict[str, list[Path]] = {}
    for path in sorted(p for p in results_dir.iterdir() if (p / "results.json").is_file()):
        match = _SET_DIR_RE.match(path.name)
        groups.setdefault(match["run"] if match else path.name, []).append(path)

    runs = []
    for key, paths in groups.items():
        records = [json.loads((p / "results.json").read_text(encoding="utf-8")) for p in paths]
        header = records[0]["run"]
        options = header.get("options") or {}
        items = [item for record in records for item in record["items"]]
        answered = [i for i in items if not i.get("error")]
        counts = Counter(i.get("verdict") for i in items)
        scored = sum(counts.get(v, 0) for v in ("S", "P", "N"))
        latencies = [float(i["latency_ms"]) for i in answered if i.get("latency_ms") is not None]
        ttfts = [
            float(t)
            for i in answered
            if isinstance(t := ((i.get("response") or {}).get("usage") or {}).get("timings", {}).get("ttft_ms"), int)
        ]
        runs.append(
            RunPoint(
                key=key,
                started_at=min(r["run"].get("started_at") or "" for r in records),
                run_name=options.get("run_name") or key[11:],
                scope=_scope(options.get("only"), sum((r["run"].get("dataset") or {}).get("rows") or 0 for r in records)),
                model=header.get("llm_model") or (header.get("backend_config") or {}).get("model"),
                backend_version=header.get("backend_version"),
                git_sha=(header.get("git_sha") or "")[:7] or None,
                repeat=options.get("repeat") or 1,
                sets=[_SET_DIR_RE.match(p.name)["set"] if _SET_DIR_RE.match(p.name) else "-" for p in paths],
                dirs=[p.name for p in paths],
                n_answers=len(items),
                verdicts={v: counts.get(v, 0) for v in VERDICTS},
                pct_correct_of_scored=round(100.0 * counts.get("S", 0) / scored, 1) if scored else None,
                p50_latency_s=_seconds(_percentile(latencies, 0.5)),
                p95_latency_s=_seconds(_percentile(latencies, 0.95)),
                p50_ttft_s=_seconds(_percentile(ttfts, 0.5)),
                p95_ttft_s=_seconds(_percentile(ttfts, 0.95)),
                settings=_settings(header.get("backend_config")),
            )
        )
    return sorted(runs, key=lambda r: (r.started_at, r.key))


def select(runs: list[RunPoint], *, scope: str | None = None, match: str | None = None, since: str | None = None):
    """Filter by exact scope, by a regex on the run key, and by start date (YYYY-MM-DD)."""
    pattern = re.compile(match) if match else None
    return [
        r
        for r in runs
        if (scope is None or r.scope == scope)
        and (pattern is None or pattern.search(r.key))
        and (since is None or r.date >= since)
    ]


def _pct(n: int, total: int) -> str:
    return f"{100 * n / total:.0f}%" if total else "-"


def render_markdown(runs: list[RunPoint]) -> str:
    """The table behind the chart, for pasting into an issue comment."""
    head = "| Date | Run | Scope | Model | Answers | S | P | N | ? | ERROR | S of scored | p50 / p95 latency | p50 TTFT |"
    lines = [head, "| " + " | ".join("---" for _ in range(head.count("|") - 1)) + " |"]
    for r in runs:
        v, n = r.verdicts, r.n_answers
        lat = f"{r.p50_latency_s} / {r.p95_latency_s} s" if r.p50_latency_s is not None else "-"
        ttft = f"{r.p50_ttft_s} s" if r.p50_ttft_s is not None else "-"
        cells = [
            r.date, f"`{r.run_name}`", r.scope, r.model or "-", f"{n}" + (f" (×{r.repeat})" if r.repeat > 1 else ""),
            *(f"{v[k]} ({_pct(v[k], n)})" for k in VERDICTS),
            f"{r.pct_correct_of_scored}%" if r.pct_correct_of_scored is not None else "-", lat, ttft,
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def render_html(runs: list[RunPoint], *, title: str, targets: dict[str, float] | None = None) -> str:
    """A self-contained page (inline SVG drawn by a small script, no external requests)."""
    data = json.dumps([{**asdict(r), "date": r.date} for r in runs], ensure_ascii=False)
    return (
        _TEMPLATE.replace("__TITLE__", html.escape(title))
        .replace("__DATA__", data.replace("</", "<\\/"))
        .replace("__TARGETS__", json.dumps(targets or {}))
    )


def _parse_targets(values: list[str]) -> dict[str, float]:
    targets = {}
    for value in values:
        name, _, seconds = value.partition("=")
        if name not in ("p50_latency", "p95_latency", "p50_ttft", "p95_ttft") or not seconds:
            raise argparse.ArgumentTypeError(f"--target {value!r}: expected e.g. p50_latency=10")
        targets[name] = float(seconds)
    return targets


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sme-chatbot-eval-trend", description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(config.EVAL_RESULTS_DIR),
        help=f"Run records to chart (default: {config.EVAL_RESULTS_DIR})",
    )
    parser.add_argument("--out", type=Path, default=Path("eval-trend.html"), help="HTML file to write")
    parser.add_argument("--scope", help='Keep one question set, e.g. "all 172 questions" or "smoke-ids"')
    parser.add_argument("--match", help="Keep runs whose directory key matches this regex")
    parser.add_argument("--since", help="Keep runs started on or after this date (YYYY-MM-DD)")
    parser.add_argument("--title", default="chatbot eval trend")
    parser.add_argument(
        "--target",
        action="append",
        default=[],
        metavar="NAME=SECONDS",
        help="Reference line on the time charts: p50_latency, p95_latency, p50_ttft or p95_ttft (repeatable)",
    )
    parser.add_argument("--markdown", action="store_true", help="Also print the table as Markdown")
    parser.add_argument("--list-scopes", action="store_true", help="List the scopes and their run counts, then exit")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        targets = _parse_targets(args.target)
    except argparse.ArgumentTypeError as exc:
        print(exc, file=sys.stderr)
        return 2
    if not args.results_dir.is_dir():
        print(f"no such directory: {args.results_dir}", file=sys.stderr)
        return 2
    runs = load_runs(args.results_dir)
    if args.list_scopes:
        for scope, n in Counter(r.scope for r in runs).most_common():
            print(f"{n:3}  {scope}")
        return 0
    runs = select(runs, scope=args.scope, match=args.match, since=args.since)
    if not runs:
        print("no runs match", file=sys.stderr)
        return 1
    args.out.write_text(render_html(runs, title=args.title, targets=targets), encoding="utf-8")
    print(f"wrote {args.out} ({len(runs)} runs)")
    if args.markdown:
        print()
        print(render_markdown(runs), end="")
    return 0


_TEMPLATE = r"""<title>__TITLE__</title>
<style>
/* Verdict colours: S blue, P amber, N grey; ? and ERROR added. Validated with the
   dataviz palette checks (CVD and normal-vision separation pass in both modes); N and ? are grey on
   purpose (no credit), told apart by lightness, the in-segment labels and the table. */
:root {
  color-scheme: light;
  --bg: #f6f7f9; --surface: #ffffff; --fg: #1b2230; --muted: #5d6677; --grid: #dde1e8;
  --s: #1f6fb5; --p: #d9a21b; --n: #a3a9b4; --u: #6b7385; --e: #c23b3b;
  --on-s: #ffffff; --on-p: #1b2230; --on-n: #1b2230; --on-u: #ffffff; --on-e: #ffffff;
  --l1: #1f6fb5; --l2: #eb6834; --target: #5d6677;
  --font: "IBM Plex Sans", system-ui, sans-serif; --mono: "IBM Plex Mono", ui-monospace, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  color-scheme: dark;
  --bg: #141922; --surface: #1b212c; --fg: #e8ebf1; --muted: #9aa3b5; --grid: #2d3544;
  --s: #3987e5; --p: #c98500; --n: #566075; --u: #8a93a6; --e: #e66767;
  --on-s: #0e1420; --on-p: #14100a; --on-n: #e8ebf1; --on-u: #0e1420; --on-e: #0e1420;
  --l1: #5aa2e6; --l2: #d95926; --target: #9aa3b5; } }
:root[data-theme="dark"] {
  color-scheme: dark;
  --bg: #141922; --surface: #1b212c; --fg: #e8ebf1; --muted: #9aa3b5; --grid: #2d3544;
  --s: #3987e5; --p: #c98500; --n: #566075; --u: #8a93a6; --e: #e66767;
  --on-s: #0e1420; --on-p: #14100a; --on-n: #e8ebf1; --on-u: #0e1420; --on-e: #0e1420;
  --l1: #5aa2e6; --l2: #d95926; --target: #9aa3b5; }
body { background: var(--bg); color: var(--fg); font-family: var(--font); margin: 0; padding-inline: 16px; padding-block: 24px; }
main { max-width: 1100px; margin: 0 auto; display: flex; flex-direction: column; gap: 20px; }
h1 { font-size: 1.4rem; margin: 0; text-wrap: balance; }
h2 { font-size: 1rem; margin: 0 0 4px; }
p { margin: 4px 0 0; color: var(--muted); font-size: .9rem; line-height: 1.5; }
.card { background: var(--surface); border: 1px solid var(--grid); border-radius: 8px; padding: 16px; min-width: 0; }
.filters { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; font-size: .85rem; }
.filters select { font: inherit; padding: 4px 8px; border-radius: 6px; border: 1px solid var(--grid); background: var(--surface); color: var(--fg); }
.legend { display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: .8rem; color: var(--muted); margin-bottom: 8px; }
.legend i { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; vertical-align: -1px; }
.legend i.line { height: 2px; border-radius: 0; vertical-align: 3px; }
.legend i.dash { height: 0; border-top: 2px dashed var(--target); background: none; vertical-align: 3px; }
.scroll { overflow-x: auto; }
svg { display: block; }
svg text { font-family: var(--mono); fill: var(--muted); font-size: 11px; }
svg .seg { font-weight: 600; font-size: 11px; pointer-events: none; }
svg .hit:hover { fill: var(--fg); fill-opacity: .06; }
.tip { position: fixed; pointer-events: none; background: var(--surface); color: var(--fg); border: 1px solid var(--grid);
  border-radius: 6px; padding: 8px 10px; font-size: .8rem; line-height: 1.45; box-shadow: 0 4px 16px rgb(0 0 0 / .12); max-width: 320px; z-index: 2; }
.tip b { font-weight: 600; }
table { border-collapse: collapse; width: 100%; font-size: .8rem; font-variant-numeric: tabular-nums; }
th, td { text-align: right; padding: 6px 8px; border-bottom: 1px solid var(--grid); white-space: nowrap; }
th:nth-child(-n+4), td:nth-child(-n+4) { text-align: left; }
th { color: var(--muted); font-weight: 500; }
code { font-family: var(--mono); font-size: .78rem; }
</style>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;600&display=swap">
<main>
  <header>
    <h1>__TITLE__</h1>
    <p>One column per eval run, oldest first. Dashed vertical lines mark a change of question set (scope);
    the time lines also break where the model changes. Runs are comparable only within one scope and on the
    same model. Counts are per answer.</p>
  </header>
  <div class="filters"><label for="scope">Question set</label><select id="scope"></select><span id="count"></span></div>
  <section class="card">
    <h2>Verdicts, share of answers</h2>
    <div class="legend">
      <span><i style="background:var(--s)"></i>S correct</span><span><i style="background:var(--p)"></i>P partial</span>
      <span><i style="background:var(--n)"></i>N wrong</span><span><i style="background:var(--u)"></i>? undetermined</span>
      <span><i style="background:var(--e)"></i>ERROR</span>
    </div>
    <div class="scroll"><svg id="verdicts" role="img" aria-label="Share of verdicts per run"></svg></div>
  </section>
  <section class="card">
    <h2>End-to-end latency</h2>
    <div class="legend"><span><i class="line" style="background:var(--l1)"></i>p50</span><span><i class="line" style="background:var(--l2)"></i>p95</span><span class="tgt-legend" hidden><i class="dash"></i>target</span></div>
    <div class="scroll"><svg id="latency" role="img" aria-label="p50 and p95 latency per run, seconds"></svg></div>
  </section>
  <section class="card">
    <h2>Time to first token</h2>
    <div class="legend"><span><i class="line" style="background:var(--l1)"></i>p50</span><span><i class="line" style="background:var(--l2)"></i>p95</span><span class="tgt-legend" hidden><i class="dash"></i>target</span></div>
    <p id="ttft-note"></p>
    <div class="scroll"><svg id="ttft" role="img" aria-label="p50 and p95 time to first token per run, seconds"></svg></div>
  </section>
  <section class="card scroll">
    <table><thead><tr><th>Date</th><th>Run</th><th>Scope</th><th>Model</th><th>Answers</th><th>S</th><th>P</th><th>N</th><th>?</th><th>ERROR</th>
    <th>S of scored</th><th>p50 latency</th><th>p95 latency</th><th>p50 TTFT</th><th>Settings</th></tr></thead><tbody id="rows"></tbody></table>
  </section>
</main>
<div class="tip" id="tip" hidden></div>
<script>
var RUNS = __DATA__, TARGETS = __TARGETS__;
var NS = "http://www.w3.org/2000/svg", tip = document.getElementById("tip");
var SEGS = [["S", "--s", "--on-s"], ["P", "--p", "--on-p"], ["N", "--n", "--on-n"], ["?", "--u", "--on-u"], ["ERROR", "--e", "--on-e"]];
var L = 48, T = 12, H = 220, SLOT = 56, BW = 36, FOOT = 140;  // SLOT/BW are re-fitted per render
function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]; }); }
function mk(svg, n, a, t) { var e = document.createElementNS(NS, n); for (var k in a) e.setAttribute(k, a[k]); if (t != null) e.textContent = t; svg.appendChild(e); return e; }
function settings(r) { return Object.keys(r.settings).map(function (k) { return k + "=" + r.settings[k]; }).join(", "); }
function describe(r) {
  var v = r.verdicts, n = r.n_answers;
  return "<b>" + esc(r.run_name) + "</b> · " + r.date + "<br>" + esc(r.scope) + " · " + esc(r.model || "model ?") +
    (r.repeat > 1 ? " · ×" + r.repeat : "") + "<br>" +
    SEGS.map(function (s) { return s[0] + " " + v[s[0]] + " (" + Math.round(100 * v[s[0]] / n) + "%)"; }).join(" · ") +
    "<br>S of scored " + (r.pct_correct_of_scored == null ? "-" : r.pct_correct_of_scored + "%") +
    "<br>latency p50 " + (r.p50_latency_s == null ? "-" : r.p50_latency_s + " s") + ", p95 " + (r.p95_latency_s == null ? "-" : r.p95_latency_s + " s") +
    "<br>TTFT p50 " + (r.p50_ttft_s == null ? "-" : r.p50_ttft_s + " s") +
    (settings(r) ? "<br><code>" + esc(settings(r)) + "</code>" : "");
}
function hover(el, r) {
  el.addEventListener("mousemove", function (e) {
    tip.innerHTML = describe(r); tip.hidden = false;
    var x = e.clientX + 14, y = e.clientY + 14, w = tip.offsetWidth, h = tip.offsetHeight;
    if (x + w > innerWidth - 8) x = e.clientX - w - 14; if (y + h > innerHeight - 8) y = e.clientY - h - 14;
    tip.style.left = x + "px"; tip.style.top = y + "px";
  });
  el.addEventListener("mouseleave", function () { tip.hidden = true; });
}
function frame(svg, runs, yMax, fmt, ticks) {
  svg.innerHTML = "";
  var W = Math.max(SLOT * runs.length, 320);
  svg.setAttribute("width", L + W + 8); svg.setAttribute("height", T + H + FOOT);
  svg.setAttribute("viewBox", "0 0 " + (L + W + 8) + " " + (T + H + FOOT));
  var y = function (v) { return T + H - v / yMax * H; };
  ticks.forEach(function (v) {
    mk(svg, "line", {x1: L, x2: L + W, y1: y(v), y2: y(v), stroke: "var(--grid)", "stroke-width": 1});
    mk(svg, "text", {x: L - 8, y: y(v) + 4, "text-anchor": "end"}, fmt(v));
  });
  runs.forEach(function (r, i) {
    var cx = L + SLOT * i + SLOT / 2;
    mk(svg, "text", {x: cx, y: T + H + 16, "text-anchor": "middle"}, r.date.slice(5));
    mk(svg, "text", {x: cx, y: T + H + 30, "text-anchor": "middle"}, "n=" + r.n_answers);
    // run names are long and similar: rotated, so neighbours never overlap, and cut from the
    // front, since variants of one experiment differ at the end (…-re-none, …-re-none-ps)
    var name = r.run_name.length > 20 ? "…" + r.run_name.slice(-19) : r.run_name, ny = T + H + 42;
    mk(svg, "text", {x: cx + 4, y: ny, "text-anchor": "end", transform: "rotate(-45 " + (cx + 4) + " " + ny + ")"}, name);
    if (i > 0 && r.scope !== runs[i - 1].scope) {
      var gx = L + SLOT * i;
      mk(svg, "line", {x1: gx, x2: gx, y1: T, y2: T + H, stroke: "var(--muted)", "stroke-dasharray": "3 4", "stroke-width": 1});
    }
  });
  return {y: y, W: W};
}
// maxima whose quarters are whole seconds
function niceMax(v) { var steps = [4, 8, 12, 20, 40, 60, 80, 100, 120, 160, 200, 300, 400]; for (var i = 0; i < steps.length; i++) if (v <= steps[i]) return steps[i]; return Math.ceil(v / 400) * 400; }
function drawVerdicts(runs) {
  var svg = document.getElementById("verdicts"), f = frame(svg, runs, 100, function (v) { return v + "%"; }, [0, 25, 50, 75, 100]);
  runs.forEach(function (r, i) {
    var x = L + SLOT * i + (SLOT - BW) / 2, acc = 0;
    SEGS.forEach(function (s) {
      var pc = 100 * r.verdicts[s[0]] / r.n_answers, top = f.y(acc + pc), h = f.y(acc) - top;
      if (h > 0) {
        mk(svg, "rect", {x: x, y: top + 1, width: BW, height: Math.max(h - 2, 0.5), rx: 3, fill: "var(" + s[1] + ")"});
        if (h >= 16) mk(svg, "text", {class: "seg", x: x + BW / 2, y: top + h / 2 + 4, "text-anchor": "middle", style: "fill:var(" + s[2] + ")"}, Math.round(pc));
      }
      acc += pc;
    });
    hover(mk(svg, "rect", {class: "hit", x: L + SLOT * i, y: T, width: SLOT, height: H, fill: "transparent"}), r);
  });
}
function drawLines(id, runs, k50, k95, tk50, tk95) {
  var svg = document.getElementById(id), vals = [];
  runs.forEach(function (r) { [r[k50], r[k95]].forEach(function (v) { if (v != null) vals.push(v); }); });
  [TARGETS[tk50], TARGETS[tk95]].forEach(function (v) { if (v != null) vals.push(v); });
  var yMax = niceMax(Math.max.apply(null, vals.concat([1]))), ticks = [0, yMax / 4, yMax / 2, 3 * yMax / 4, yMax];
  var f = frame(svg, runs, yMax, function (v) { return Math.round(v) + " s"; }, ticks);
  [[tk50, TARGETS[tk50]], [tk95, TARGETS[tk95]]].forEach(function (t) {
    if (t[1] == null) return;
    mk(svg, "line", {x1: L, x2: L + f.W, y1: f.y(t[1]), y2: f.y(t[1]), stroke: "var(--target)", "stroke-dasharray": "6 4", "stroke-width": 1.5});
    mk(svg, "text", {x: L + 6, y: f.y(t[1]) - 5, "text-anchor": "start", style: "fill:var(--fg)"}, t[0].split("_")[0] + " target " + t[1] + " s");
  });
  [[k50, "--l1"], [k95, "--l2"]].forEach(function (s) {
    var d = "", pen = false;
    runs.forEach(function (r, i) {
      var v = r[s[0]], cx = L + SLOT * i + SLOT / 2;
      // no line across a change of scope or model, or a missing value: those points are not comparable
      if (v == null || (i > 0 && (r.scope !== runs[i - 1].scope || r.model !== runs[i - 1].model))) pen = false;
      if (v == null) return;
      d += (pen ? "L" : "M") + cx + " " + f.y(v) + " "; pen = true;
    });
    if (d) mk(svg, "path", {d: d, fill: "none", stroke: "var(" + s[1] + ")", "stroke-width": 2, "stroke-linejoin": "round"});
    runs.forEach(function (r, i) {
      var v = r[s[0]]; if (v == null) return;
      mk(svg, "circle", {cx: L + SLOT * i + SLOT / 2, cy: f.y(v), r: 4, fill: "var(" + s[1] + ")", stroke: "var(--surface)", "stroke-width": 2});
    });
  });
  runs.forEach(function (r, i) { hover(mk(svg, "rect", {class: "hit", x: L + SLOT * i, y: T, width: SLOT, height: H, fill: "transparent"}), r); });
}
function drawTable(runs) {
  document.getElementById("rows").innerHTML = runs.map(function (r) {
    var v = r.verdicts, n = r.n_answers, cell = function (k) { return v[k] + " <span style='color:var(--muted)'>(" + Math.round(100 * v[k] / n) + "%)</span>"; };
    var s = function (x) { return x == null ? "-" : x + " s"; };
    return "<tr><td>" + r.date + "</td><td><code>" + esc(r.run_name) + "</code></td><td>" + esc(r.scope) + "</td><td>" + esc(r.model || "-") +
      "</td><td>" + n + (r.repeat > 1 ? " (×" + r.repeat + ")" : "") + "</td><td>" + SEGS.map(function (g) { return cell(g[0]); }).join("</td><td>") +
      "</td><td>" + (r.pct_correct_of_scored == null ? "-" : r.pct_correct_of_scored + "%") + "</td><td>" + s(r.p50_latency_s) + "</td><td>" +
      s(r.p95_latency_s) + "</td><td>" + s(r.p50_ttft_s) + "</td><td><code>" + esc(settings(r)) + "</code></td></tr>";
  }).join("");
}
function render() {
  var scope = document.getElementById("scope").value, runs = RUNS.filter(function (r) { return !scope || r.scope === scope; });
  // A few runs spread over the card's width; many runs keep 56 px each and scroll.
  var avail = document.querySelector(".scroll").clientWidth - L - 8;
  SLOT = Math.max(56, Math.min(160, avail / Math.max(runs.length, 1)));
  BW = Math.min(56, Math.round(SLOT * 0.6));
  document.getElementById("count").textContent = runs.length + " runs";
  drawVerdicts(runs);
  drawLines("latency", runs, "p50_latency_s", "p95_latency_s", "p50_latency", "p95_latency");
  drawLines("ttft", runs, "p50_ttft_s", "p95_ttft_s", "p50_ttft", "p95_ttft");
  var missing = runs.filter(function (r) { return r.p50_ttft_s == null; }).length;
  document.getElementById("ttft-note").textContent = missing ? missing + " of " + runs.length + " runs predate the backend timings and have no time to first token." : "";
  drawTable(runs);
}
var counts = {}; RUNS.forEach(function (r) { counts[r.scope] = (counts[r.scope] || 0) + 1; });
document.getElementById("scope").innerHTML = "<option value=''>All (" + RUNS.length + ")</option>" +
  Object.keys(counts).sort(function (a, b) { return counts[b] - counts[a]; }).map(function (s) { return "<option value='" + esc(s) + "'>" + esc(s) + " (" + counts[s] + ")</option>"; }).join("");
document.getElementById("scope").addEventListener("change", render);
if (Object.keys(TARGETS).length) document.querySelectorAll(".tgt-legend").forEach(function (e) { e.hidden = false; });
render();
</script>
"""


if __name__ == "__main__":
    sys.exit(main())
