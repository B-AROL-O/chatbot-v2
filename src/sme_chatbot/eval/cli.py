"""WP8 — the ``sme-chatbot-eval`` entry point.

    uv run sme-chatbot-eval \\
      --dataset tests/fixtures/eval/fixture-alpha-domande.csv \\
      --api-url http://localhost:8000 \\
      --domain documents \\
      --run-name baseline \\
      --out data/eval/results

Both paths default to ``$EVAL_DATASETS_DIR`` / ``$EVAL_RESULTS_DIR``: a checkout of
the private eval repo at ``data/eval/`` unless set otherwise.

Exit code ``0`` on a completed run **regardless of verdicts**, ``1`` on a run
that could not complete (backend unreachable, domain missing, dataset
unreadable, ``--judge`` given but the judge fails its probe). Verdicts deliberately do not set the exit code: gate thresholds are
out of scope, and baking one in now would make the first red run look like a
broken tool.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

from sme_chatbot import config
from sme_chatbot.eval import RUNNER_VERSION
from sme_chatbot.eval.client import BackendUnavailable, EvalClient
from sme_chatbot.eval.dataset import DatasetError, NoMatchError, discover_datasets, load_dataset, parse_only
from sme_chatbot.eval.judge import JudgeUnavailable, LlmJudge
from sme_chatbot.eval.report import build_run_header, load_results_json, run_directory, write_all
from sme_chatbot.eval.runner import VERDICT_ORDER, EvalResult, rescore, run_dataset
from sme_chatbot.eval.tracing import make_tracer

logger = logging.getLogger(__name__)

DEFAULT_DOMAIN = "documents"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sme-chatbot-eval",
        description=(
            "Run a golden question set from $EVAL_DATASETS_DIR against a chatbot backend "
            "and write a run record under $EVAL_RESULTS_DIR, to commit in the eval repo."
        ),
    )
    parser.add_argument(
        "--dataset",
        default=None,
        metavar="CSV",
        help=f"Question set to run (default: every {config.EVAL_DATASETS_DIR}/*-domande.csv)",
    )
    parser.add_argument(
        "--api-url",
        default=config.CHATBOT_API_URL,
        help="Backend base URL (default: $CHATBOT_API_URL, else http://localhost:8000)",
    )
    parser.add_argument("--domain", default=DEFAULT_DOMAIN, help=f"Chat domain (default: {DEFAULT_DOMAIN})")
    parser.add_argument("--run-name", default="run", help="Name of the run directory's suffix")
    parser.add_argument(
        "--out", default=config.EVAL_RESULTS_DIR, help=f"Output root (default: {config.EVAL_RESULTS_DIR})"
    )
    parser.add_argument("--user-id", default="eval-runner", help="X-User-Id sent with every request")
    parser.add_argument(
        "--only",
        default=None,
        metavar="SPEC",
        help="Subset: IDs/ranges 'Q01,Q05-Q08' or @FILE, one ID list that spans all question sets",
    )
    parser.add_argument("--repeat", type=int, default=1, metavar="N", help="Ask each question N times")
    parser.add_argument("--concurrency", type=int, default=1, help="Parallel requests (default 1)")
    parser.add_argument("--timeout", type=float, default=180.0, help="Read timeout in seconds (default 180)")
    parser.add_argument("--max-retries", type=int, default=3, help="Retries on 429/5xx (default 3)")
    parser.add_argument("--page-offset", type=int, default=10, help="Printed page -> PDF page offset (default 10)")
    parser.add_argument("--page-tolerance", type=int, default=2, help="Page match tolerance (default 2)")
    parser.add_argument("--judge", action="store_true", help="Resolve '?' items with an LLM judge")
    parser.add_argument("--judge-model", default=None, help="Judge model (default: $LLM_MODEL)")
    parser.add_argument(
        "--judge-temperature",
        type=float,
        default=0.0,
        help="Judge temperature (default 0; dropped automatically if the model rejects it)",
    )
    parser.add_argument(
        "--judge-timeout",
        type=float,
        default=60.0,
        help="Per-request judge timeout in seconds (default 60)",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Skip TLS verification (the deployed instance is reached by IP)",
    )
    parser.add_argument(
        "--no-rubric",
        action="store_true",
        help="Ignore the derived rubric sidecar (required/forbidden patterns)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the item list and exit without calling the backend",
    )
    parser.add_argument("--no-write", action="store_true", help="Run and print, but write no artifacts")
    parser.add_argument(
        "--rescore",
        default=None,
        metavar="RESULTS_JSON",
        help=(
            "Score a committed results.json again instead of asking the backend. "
            "Writes a new run directory; the original is never modified."
        ),
    )
    return parser


def _options_payload(args: argparse.Namespace) -> dict:
    """Every CLI option, recorded verbatim into the run header."""
    return {
        key: value
        for key, value in sorted(vars(args).items())
        if key not in ("dataset",)  # recorded under `dataset` with its sha256
    }


def _resolve_datasets(args: argparse.Namespace) -> list[Path]:
    if args.dataset:
        return [Path(args.dataset)]
    found = discover_datasets(config.EVAL_DATASETS_DIR)
    if not found:
        raise DatasetError(
            f"no *-domande.csv found under {config.EVAL_DATASETS_DIR}/: clone "
            "the eval-data repo at data/eval/, or set EVAL_DATASETS_DIR"
        )
    return found


def _rescore_run(args: argparse.Namespace, dataset, name_suffix: str, items) -> int:
    """Re-apply scoring to answers a previous run already captured.

    ``items`` is the ``--only`` subset (every item without it): only the
    captured answers to those questions are rescored, and judged with ``--judge``.
    """
    source = Path(args.rescore)
    if not source.is_file():
        raise DatasetError(f"results.json not found: {source}")

    run_header, captured, warnings = load_results_json(source, dataset)
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)

    wanted = {item.id for item in items}
    captured = [result for result in captured if result.item.id in wanted]

    judge = _make_judge(args)
    results = rescore(captured, page_offset=args.page_offset, page_tolerance=args.page_tolerance, judge=judge)
    counts = {v: sum(1 for r in results if r.verdict == v) for v in VERDICT_ORDER}
    print(f"rescored: {len(results)} item(s) from {source}")
    print(" · ".join(f"{v} {counts[v]}" for v in VERDICT_ORDER))

    if args.no_write:
        print("(--no-write: no artifacts written)")
        return 0

    # A run record is append-only (results/README.md in the eval repo): a rescore
    # writes a *new* dated directory and leaves the original untouched.
    header = dict(run_header)
    header["dataset"] = dataset.describe()
    header["options"] = _options_payload(args)
    header["rescored_from"] = source.as_posix()
    header["rescore_warnings"] = warnings
    if judge is not None:
        header["judge"] = judge.describe()
    header["runner_version"] = RUNNER_VERSION

    out_dir = run_directory(args.out, args.run_name + name_suffix, now=datetime.now(UTC))
    paths = write_all(
        out_dir, dataset=dataset, results=results, run_header=header, domain=header.get("domain") or args.domain
    )
    print(f"\nwrote {len(paths)} files to {out_dir}")
    return 0


def _make_judge(args: argparse.Namespace) -> LlmJudge | None:
    """Build and probe the judge, or return ``None`` without ``--judge``.

    Raises ``JudgeUnavailable`` when the probe fails: a run that asked for a
    judge must not complete with zero judge verdicts and look judged.
    """
    if not args.judge:
        return None
    judge = LlmJudge(model=args.judge_model, temperature=args.judge_temperature, timeout=args.judge_timeout)
    judge.probe()
    temperature = "server default" if judge.temperature is None else judge.temperature
    print(f"judge   : {judge.model} temperature={temperature} (probe ok)")
    return judge


def _run_one_dataset(args: argparse.Namespace, dataset_path: Path, *, name_suffix: str = "") -> int:
    dataset = load_dataset(dataset_path, rubric=not args.no_rubric)
    items = list(dataset.items)
    if args.only:
        items = parse_only(args.only, dataset.items)

    print(f"dataset : {dataset.path} ({dataset.n_items} rows, sha256 {dataset.sha256[:12]})")
    print(f"rubric  : {dataset.rubric_path or 'none'}")

    if args.rescore:
        return _rescore_run(args, dataset, name_suffix, items)

    print(f"asking  : {len(items)} question(s) x {max(1, args.repeat)}")

    if args.dry_run:
        for item in items:
            anchors = ", ".join(a.describe() for a in item.anchors) or "—"
            print(f"  {item.id:>5}  {item.categoria:<28} anchors: {anchors}")
        return 0

    judge = _make_judge(args)

    tracer = make_tracer(config.LANGFUSE_ENABLED)
    started_at = datetime.now(UTC)

    with EvalClient(
        args.api_url,
        user_id=args.user_id,
        timeout=args.timeout,
        max_retries=args.max_retries,
        verify=not args.insecure,
    ) as client:
        health = client.require_domain(args.domain)
        try:
            backend_config = client.config()
        except BackendUnavailable as exc:
            print(f"warning : GET /config failed ({exc}); LLM model will be recorded as unknown", file=sys.stderr)
            backend_config = {}
        print(
            f"backend : {client.api_url} version={health.get('version')} domains={health.get('domains')} "
            f"model={backend_config.get('model', 'unknown')}"
        )

        def progress(result: EvalResult) -> None:
            marker = result.verdict
            detail = result.error or f"{result.latency_ms} ms, {len(result.citations)} citations"
            print(f"  {result.run_key:>6}  {marker:<5} {detail}")

        results = run_dataset(
            items,
            client,
            domain=args.domain,
            repeat=args.repeat,
            concurrency=args.concurrency,
            page_offset=args.page_offset,
            page_tolerance=args.page_tolerance,
            tracer=tracer,
            judge=judge,
            on_result=progress,
        )

    finished_at = datetime.now(UTC)
    run_header = build_run_header(
        dataset=dataset,
        api_url=client.api_url,
        domain=args.domain,
        health=health,
        options=_options_payload(args),
        started_at=started_at,
        finished_at=finished_at,
        llm_model=backend_config.get("model"),
        backend_config=backend_config or None,
        # Described after the run: the effective temperature and the call/error
        # counts are only known once the judge has been used.
        judge_info=judge.describe() if judge is not None else None,
    )

    counts = {v: sum(1 for r in results if r.verdict == v) for v in VERDICT_ORDER}
    print("\n" + " · ".join(f"{v} {counts[v]}" for v in VERDICT_ORDER))

    if args.no_write:
        print("(--no-write: no artifacts written)")
        return 0

    out_dir = run_directory(args.out, args.run_name + name_suffix, now=started_at)
    paths = write_all(out_dir, dataset=dataset, results=results, run_header=run_header, domain=args.domain)
    print(f"\nwrote {len(paths)} files to {out_dir}:")
    for name in sorted(paths):
        print(f"  {paths[name]}")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=config.LOG_LEVEL.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = build_arg_parser().parse_args(argv)
    print(f"sme-chatbot-eval {RUNNER_VERSION}")

    try:
        datasets = _resolve_datasets(args)
    except DatasetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    # More than one question set in one invocation would otherwise write them
    # all into the same run directory; keep each set's record separate.
    multiple = len(datasets) > 1
    skipped = 0
    for index, dataset_path in enumerate(datasets):
        if index:
            print()
        suffix = "-" + dataset_path.name.replace("-domande.csv", "") if multiple else ""
        try:
            _run_one_dataset(args, dataset_path, name_suffix=suffix)
        except NoMatchError as exc:
            # One --only list can span several files (IDs are unique across them);
            # a file holding none of the IDs is simply not part of the subset.
            if not multiple:
                print(f"error: {exc}", file=sys.stderr)
                return 1
            print(f"skipped : {dataset_path.name}: {exc}")
            skipped += 1
        except (DatasetError, BackendUnavailable, JudgeUnavailable) as exc:
            # A run that could not start or complete: exit 1. Verdicts never
            # set the exit code (see the module docstring).
            print(f"error: {exc}", file=sys.stderr)
            return 1
    if skipped == len(datasets):
        print(f"error: --only {args.only!r} matched no questions in any of {len(datasets)} datasets", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
