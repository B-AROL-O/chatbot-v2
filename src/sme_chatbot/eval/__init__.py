"""Automated eval runner for the golden question sets under ``$EVAL_DATASETS_DIR``.

Reads a ``*-domande.csv`` question set, drives a running chatbot backend over
its public HTTP contract (``POST /chat/message`` with ``stream: false``),
scores each answer against ``Risposta attesa`` deterministically, and writes
run records under ``$EVAL_RESULTS_DIR``. Both are
directories of the private eval-data repo.

Implements the eval-runner plan (WP1-WP10).

Module map:

- :mod:`~sme_chatbot.eval.dataset` — WP1, CSV -> ``EvalItem`` (+ rubric sidecar)
- :mod:`~sme_chatbot.eval.scoring` — WP4/WP5, pure scorers + verdict roll-up
- :mod:`~sme_chatbot.eval.client`  — WP2, thin HTTP client of the backend
- :mod:`~sme_chatbot.eval.runner`  — WP3, per-item orchestration
- :mod:`~sme_chatbot.eval.report`  — WP6, ``results.json`` / ``results.csv`` /
  ``report.md`` / ``summary.json`` writers
- :mod:`~sme_chatbot.eval.tracing` — WP7, opt-in W3C ``traceparent`` injection
- :mod:`~sme_chatbot.eval.judge`   — WP9, opt-in LLM judge
- :mod:`~sme_chatbot.eval.cli`     — WP8, the ``sme-chatbot-eval`` entry point

Nothing here is imported at backend start-up: the eval package is a client of
the API, never a part of it.
"""

from __future__ import annotations

# Bumped whenever a change alters scores or artifact shape, so a committed
# run record always says which runner produced it (results.json `run` header).
RUNNER_VERSION = "1.0.0"

__all__ = ["RUNNER_VERSION"]
