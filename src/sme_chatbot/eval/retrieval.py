"""Offline retrieval benchmark — WP0 of the retrieval-improvement plan.

    uv run sme-chatbot-eval-retrieval --dataset tests/fixtures/eval/fixture-alpha-domande.csv

Sends each golden question straight to the ``documents`` retriever — no LLM,
no HTTP — and reports whether the gold page comes back, per strategy
(``dense``, ``lexical``, ``hybrid``). One run takes seconds, against the
35 minutes of a full ``sme-chatbot-eval`` run, so it is the oracle for any
retrieval change (fusion, chunking, embedding model, reranker).

A hit is a result from the gold document (``Documento`` column, compared as
``scoring._document_key``) whose PDF page is within ``--page-tolerance`` of
``Pag. stampata + --page-offset`` — the runner's own page mapping. Items with
no gold page (refusals, scoping) are skipped. Each query runs as production
does after plan WP2a: scoped to the item's matricola, no language filter.

This measures *retrieval of the literal question*. The LLM rewrites queries
and can take several rounds, so a miss here is not a guaranteed wrong
answer — but a hit at rank 1 almost always becomes a right one.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from sme_chatbot.eval.dataset import DatasetError, EvalItem, discover_datasets, load_dataset, parse_only
from sme_chatbot.eval.scoring import _document_key

logger = logging.getLogger(__name__)

STRATEGIES = ("dense", "lexical", "hybrid")
DEFAULT_KS = (1, 5, 10, 20)


@dataclass
class ItemRank:
    id: str
    question: str
    expected_pdf_pages: list[int]
    ranks: dict[str, int | None] = field(default_factory=dict)  # strategy -> 1-based rank, None = miss


def _hit(item: EvalItem, result: tuple[str, str, dict, float | None], *, offset: int, tolerance: int) -> bool:
    meta = result[2] or {}
    if _document_key(str(meta.get("filename") or "")) != _document_key(item.gold_document):
        return False
    page = meta.get("page")
    return isinstance(page, int) and any(abs(page - (p + offset)) <= tolerance for p in item.gold_pages)


def rank_item(item: EvalItem, depth: int, *, offset: int, tolerance: int) -> ItemRank:
    """Rank of the first gold hit under each strategy (``None`` = not in ``depth``)."""
    from sme_chatbot.core.events import RequestIdentity
    from sme_chatbot.tools import docsvc_rag
    from sme_chatbot.tools.access_filter import build_where

    where = build_where(RequestIdentity(), serial=item.serial)
    dense = docsvc_rag._dense_hits(item.domanda, where, depth)
    lexical = docsvc_rag._lexical_hits(item.domanda, where, depth)
    ranked = {"dense": dense, "lexical": lexical, "hybrid": docsvc_rag._fuse(dense, lexical)[:depth]}

    out = ItemRank(
        id=item.id,
        question=item.domanda,
        expected_pdf_pages=sorted(p + offset for p in item.gold_pages),
    )
    for strategy, results in ranked.items():
        out.ranks[strategy] = next(
            (i + 1 for i, r in enumerate(results) if _hit(item, r, offset=offset, tolerance=tolerance)), None
        )
    return out


def recall_at(ranks: list[ItemRank], strategy: str, k: int) -> float:
    if not ranks:
        return 0.0
    return sum(1 for r in ranks if r.ranks[strategy] is not None and r.ranks[strategy] <= k) / len(ranks)


def render(ranks: list[ItemRank], ks: tuple[int, ...]) -> str:
    lines = ["| ID | PDF page | " + " | ".join(STRATEGIES) + " | Domanda |", "| --- | --- |" + " ---: |" * len(STRATEGIES) + " --- |"]
    for r in ranks:
        cells = [str(r.ranks[s]) if r.ranks[s] is not None else "—" for s in STRATEGIES]
        lines.append(f"| {r.id} | {', '.join(map(str, r.expected_pdf_pages))} | " + " | ".join(cells) + f" | {r.question} |")
    lines += ["", "| recall@k | " + " | ".join(STRATEGIES) + " |", "| --- |" + " ---: |" * len(STRATEGIES)]
    for k in ks:
        lines.append(f"| @{k} | " + " | ".join(f"{recall_at(ranks, s, k):.2f}" for s in STRATEGIES) + " |")
    lines.append(f"\n{len(ranks)} items with a gold page.")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sme-chatbot-eval-retrieval",
        description="Rank of the gold page per question under dense / lexical / hybrid retrieval (no LLM).",
    )
    parser.add_argument("--dataset", action="append", help="A *-domande.csv (repeatable; default: all in $EVAL_DATASETS_DIR)")
    parser.add_argument("--only", default=None, help="Subset, e.g. Q01,Q05-Q08")
    parser.add_argument("--depth", type=int, default=20, help="Results fetched per strategy (default 20)")
    parser.add_argument("--page-offset", type=int, default=10)
    parser.add_argument("--page-tolerance", type=int, default=2)
    parser.add_argument("--json", type=Path, default=None, help="Also write the ranks as JSON here")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING)
    args = build_arg_parser().parse_args(argv)

    from sme_chatbot.tools import docsvc_rag

    error = docsvc_rag._ensure_loaded()
    if error:
        print(error, file=sys.stderr)
        return 1
    if docsvc_rag._get_lexical() is None:
        print("warning: no lexical index; the lexical/hybrid columns equal dense-only.", file=sys.stderr)

    ranks: list[ItemRank] = []
    try:
        paths = [Path(p) for p in args.dataset] if args.dataset else discover_datasets()
        for path in paths:
            items = load_dataset(path).items
            if args.only:
                items = tuple(parse_only(args.only, items))
            ranks += [
                rank_item(item, args.depth, offset=args.page_offset, tolerance=args.page_tolerance)
                for item in items
                if item.gold_pages
            ]
    except DatasetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(render(ranks, tuple(k for k in DEFAULT_KS if k <= args.depth)))
    if args.json:
        args.json.write_text(json.dumps([asdict(r) for r in ranks], indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
