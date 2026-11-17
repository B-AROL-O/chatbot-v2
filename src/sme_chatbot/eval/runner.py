"""WP3 — drive the question set through the backend, one item at a time.

Sequential by default (``--concurrency 1``, assumption A2: the LLM endpoint's
rate limit is still unresolved). Per item: clock it, call, capture, score, continue. **One failing
item never aborts the run.**

Session protocol
----------------
Every item is one request with ``history: []``. The API is stateless — prior
turns arrive in the request body — so each question is a fresh session by
construction. That satisfies the legend's "Q18/Q19 and Q03/Q35/Q36 in separate
sessions" and makes "run Q34 last" moot: there is no context to pollute.

This is a **deviation from the manual protocol and it must stay visible in the
report**: Q10 ("*E* l'orientatore tappi della 90001…") and Q19 ("Sulla 90001 la
capsula è troppo avvitata.") are phrased as conversational follow-ups, so in
isolation they are harder than a human tester found them. Both still carry the
matricola, so they remain answerable. ``--history-mode csv-chain`` (replay the
preceding turns of a declared group) is specified in the plan but deliberately
not built in phase 1.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace

from sme_chatbot.eval.client import AskResult, EvalClient
from sme_chatbot.eval.dataset import EvalItem
from sme_chatbot.eval.scoring import ItemScores, score_item, sources_correct
from sme_chatbot.eval.tracing import NullTracer, OtelTracer

logger = logging.getLogger(__name__)

VERDICT_ERROR = "ERROR"
VERDICT_ORDER = ("S", "P", "N", "?", VERDICT_ERROR)
# Verdicts that belong in a percentage. `?` was never determined and `ERROR`
# never got an answer; folding either into a "% correct" would be a lie.
SCORED_VERDICTS = ("S", "P", "N")


@dataclass
class EvalResult:
    """One asked question: what went out, what came back, and how it scored."""

    item: EvalItem
    run_key: str  # "Q01" or "Q01#2" when --repeat > 1
    repeat_index: int = 1
    answer: str = ""
    citations: list[dict] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    latency_ms: int = 0
    status_code: int | None = None
    attempts: int = 0
    trace_id: str | None = None
    error: str | None = None
    scores: ItemScores | None = None
    verdict_auto: str = VERDICT_ERROR
    verdict_judge: str | None = None
    judge_reason: str | None = None

    @property
    def verdict(self) -> str:
        """What the report shows: the judge resolves ``?`` and nothing else.

        An ``S``/``N``/``P`` with a concrete anchor behind it is never
        overwritten by the judge (plan WP9) — where the two disagree, the item
        goes to the report's disagreement section for a human, rather than one
        silently winning.
        """
        if self.verdict_auto == "?" and self.verdict_judge:
            return self.verdict_judge
        return self.verdict_auto

    @property
    def disagreement(self) -> bool:
        return (
            self.verdict_judge is not None
            and self.verdict_auto in SCORED_VERDICTS
            and self.verdict_judge != self.verdict_auto
        )

    @property
    def rounds(self) -> int | None:
        value = self.usage.get("rounds")
        return value if isinstance(value, int) else None

    def timing(self, key: str) -> int | None:
        """One of the backend's ``usage["timings"]`` values, ``None`` if absent.

        Results recorded before the backend timings existed carry no ``timings``; they must still load
        and render, so absence is the normal answer here, never an error.
        """
        value = (self.usage.get("timings") or {}).get(key)
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    @property
    def sources_verdict(self) -> str:
        return sources_correct(self.scores) if self.scores else "N"

    @property
    def cited_documents(self) -> list[str]:
        seen: list[str] = []
        for citation in self.citations:
            name = str(citation.get("filename") or citation.get("source") or "").strip()
            if name and name not in seen:
                seen.append(name)
        return seen


@dataclass
class RunOutcome:
    """Everything one run produced, in dataset order."""

    results: list[EvalResult] = field(default_factory=list)
    health: dict = field(default_factory=dict)

    def counts(self) -> dict[str, int]:
        counts = dict.fromkeys(VERDICT_ORDER, 0)
        for result in self.results:
            counts[result.verdict] = counts.get(result.verdict, 0) + 1
        return counts

    @property
    def errors(self) -> list[EvalResult]:
        return [r for r in self.results if r.error]


def expand_items(items: Sequence[EvalItem], repeat: int) -> list[tuple[EvalItem, int, str]]:
    """``(item, repeat_index, run_key)`` triples.

    With the default ``--repeat 1`` the key is the plain question ID, so
    ``results.csv`` pastes straight back into the workbook; only a repeated run
    disambiguates with ``Q01#1``/``Q01#2``.
    """
    triples: list[tuple[EvalItem, int, str]] = []
    for item in items:
        for index in range(1, max(1, repeat) + 1):
            key = item.id if repeat <= 1 else f"{item.id}#{index}"
            triples.append((item, index, key))
    return triples


def run_dataset(
    items: Sequence[EvalItem],
    client: EvalClient,
    *,
    domain: str,
    repeat: int = 1,
    concurrency: int = 1,
    page_offset: int = 10,
    page_tolerance: int = 2,
    tracer: NullTracer | OtelTracer | None = None,
    judge: Callable[[EvalItem, str], tuple[str, str]] | None = None,
    on_result: Callable[[EvalResult], None] | None = None,
) -> list[EvalResult]:
    """Ask every item once (or ``repeat`` times) and score each answer."""
    tracer = tracer or NullTracer()
    triples = expand_items(items, repeat)

    def run_one(triple: tuple[EvalItem, int, str]) -> EvalResult:
        item, index, key = triple
        return _run_item(
            item,
            index,
            key,
            client,
            domain=domain,
            page_offset=page_offset,
            page_tolerance=page_tolerance,
            tracer=tracer,
            judge=judge,
        )

    results: list[EvalResult] = []
    if concurrency <= 1:
        for triple in triples:
            result = run_one(triple)
            results.append(result)
            if on_result:
                on_result(result)
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            for result in pool.map(run_one, triples):
                results.append(result)
                if on_result:
                    on_result(result)
    return results


def _run_item(
    item: EvalItem,
    repeat_index: int,
    run_key: str,
    client: EvalClient,
    *,
    domain: str,
    page_offset: int,
    page_tolerance: int,
    tracer: NullTracer | OtelTracer,
    judge: Callable[[EvalItem, str], tuple[str, str]] | None,
) -> EvalResult:
    logger.info("asking %s (%s)", run_key, item.categoria)
    with tracer.item_span(
        run_key,
        item_id=item.id,
        categoria=item.categoria,
        domain=domain,
        serial=item.serial,
    ) as (headers, trace_id):
        ask: AskResult = client.ask(domain, item.domanda, history=[], headers=headers or None)

    result = EvalResult(
        item=item,
        run_key=run_key,
        repeat_index=repeat_index,
        answer=ask.content,
        citations=ask.citations,
        usage=ask.usage,
        latency_ms=ask.latency_ms,
        status_code=ask.status_code,
        attempts=ask.attempts,
        trace_id=trace_id,
        error=ask.error,
    )

    if ask.error is not None:
        # A transport failure is not a wrong answer: no scoring, no verdict,
        # and the item is excluded from every percentage (plan §7 R5).
        result.verdict_auto = VERDICT_ERROR
        logger.warning("%s failed: %s", run_key, ask.error)
        return result

    result.scores = score_item(
        item,
        ask.content,
        ask.citations,
        page_offset=page_offset,
        page_tolerance=page_tolerance,
    )
    result.verdict_auto = result.scores.verdict

    if judge is not None:
        _apply_judge(result, judge)
    return result


def _apply_judge(result: EvalResult, judge: Callable[[EvalItem, str], tuple[str, str]]) -> None:
    """Record the judge's verdict on ``result``; a judge failure is recorded, not raised."""
    try:
        result.verdict_judge, result.judge_reason = judge(result.item, result.answer)
    except Exception as exc:  # degrade: a judge failure must not lose the answer
        logger.warning("judge failed for %s: %s", result.run_key, exc)
        result.verdict_judge, result.judge_reason = None, f"judge error: {exc}"


def rescore(
    results: Sequence[EvalResult],
    *,
    page_offset: int,
    page_tolerance: int,
    judge: Callable[[EvalItem, str], tuple[str, str]] | None = None,
) -> list[EvalResult]:
    """Re-apply scoring to already-captured answers (plan §5 step 3).

    The baseline run ships before scoring exists; this is how a committed
    ``results.json`` is scored retroactively without re-asking the backend.
    With ``judge`` the stored answers are judged again too, replacing whatever
    judge verdict (or judge error) the source run recorded.
    """
    rescored: list[EvalResult] = []
    for result in results:
        if result.error is not None:
            rescored.append(result)
            continue
        scores = score_item(
            result.item,
            result.answer,
            result.citations,
            page_offset=page_offset,
            page_tolerance=page_tolerance,
        )
        updated = replace(result, scores=scores, verdict_auto=scores.verdict)
        if judge is not None:
            _apply_judge(updated, judge)
        rescored.append(updated)
    return rescored
