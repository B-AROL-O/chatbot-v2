"""WP9 — the opt-in LLM judge (``--judge``).

Uses the same ``LLM_BASE_URL`` / ``LLM_API_TOKEN`` / ``LLM_MODEL`` the backend
uses (or ``--judge-model``), through the ``openai`` client already in the
dependency set.

Rules that keep it honest, all of them enforced here or in the runner:

- The judge **never sees the deterministic verdict** — no anchoring.
- Both verdicts are recorded side by side (``verdict_auto`` / ``verdict_judge``);
  where they disagree the item is listed in the report's *disagreement* section,
  which is the human review queue and the most valuable page of the report.
- The judge resolves ``?`` items; it never overwrites an ``S``/``N`` that has a
  concrete anchor behind it (see ``EvalResult.verdict``).
- The judge is **not** shown retrieved chunk text. Groundedness is a
  confidentiality decision, not a technical one, and it is not taken here.
- Judge runs are not reproducible. ``results.json`` records the judge model and
  the temperature that was actually *sent* (``judge_temperature``, ``None`` =
  server default), and ``report.md`` says so on its face.
- A judge that never worked must not look like one that ran: the CLI
  probes it once before the run, and ``describe()`` records how many calls
  were made and how many failed.

Models that accept only their default temperature (Azure gpt-5 answers
``temperature=0`` with HTTP 400) are handled by dropping the parameter once and
for all on the first such rejection, rather than failing every call.

Known weak oracle, accepted with the limitation stated (plan §8 Q3): the judge
runs on the same model family as the chatbot. The blast radius is bounded — it
only resolves ``?`` — and ``judge_model`` is recorded so a later switch shows up
as a change rather than a silent improvement.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass

from sme_chatbot import config
from sme_chatbot.core.content_filter import describe_block, filtered_categories
from sme_chatbot.eval.dataset import EvalItem

logger = logging.getLogger(__name__)

VALID_VERDICTS = ("S", "N", "P")

DEFAULT_TIMEOUT = 60.0
DEFAULT_MAX_RETRIES = 2

PROBE_PROMPT = 'Rispondi solo con questo oggetto JSON: {"verdict": "S", "reason": "ok"}'

SYSTEM_PROMPT = (
    "Sei un valutatore di risposte di un chatbot tecnico su manuali di "
    "macchine. Ricevi la domanda, la risposta attesa verificata sul manuale, il "
    "comportamento atteso e la risposta effettiva del chatbot.\n"
    "Assegna un verdetto secondo questa legenda:\n"
    "  S = corretta (contenuto giusto e comportamento atteso rispettato)\n"
    "  P = parziale (contenuto giusto ma incompleto o senza le qualificazioni richieste)\n"
    "  N = errata (contenuto sbagliato, inventato, o comportamento atteso non rispettato)\n"
    "Giudica il contenuto, non lo stile. Una risposta che rifiuta correttamente "
    "quando il dato non esiste e' corretta (S). Rispondi SOLO con un oggetto JSON "
    '{"verdict": "S"|"N"|"P", "reason": "una frase"}.'
)


@dataclass
class JudgeVerdict:
    verdict: str
    reason: str


class JudgeUnavailable(RuntimeError):
    """The judge could not answer the preflight probe."""


class JudgeContentFiltered(RuntimeError):
    """The provider's content filter refused the judge prompt for one item.

    Deterministic for a given prompt, so the item is not retried; the runner
    records it as the item's ``judge_reason``.
    """


def _rejects_temperature(exc: Exception) -> bool:
    """True for an HTTP 400 that names the ``temperature`` parameter.

    Duck-typed on ``status_code`` so the check needs no ``openai`` import and
    works with any client that raises the same shape.
    """
    return getattr(exc, "status_code", None) == 400 and "temperature" in str(exc).lower()


class LlmJudge:
    """Callable ``(item, answer) -> (verdict, reason)`` for the runner."""

    def __init__(
        self,
        *,
        model: str | None = None,
        base_url: str | None = None,
        api_token: str | None = None,
        temperature: float | None = 0.0,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        client: object | None = None,
    ) -> None:
        self.model = model or config.LLM_MODEL
        self.base_url = base_url or config.LLM_BASE_URL
        self.temperature_requested = temperature
        # What is actually sent; ``None`` after the server rejected the requested value.
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.calls = 0
        self.errors = 0
        self.content_filtered = 0
        self._lock = threading.Lock()
        self._client = client
        self._api_token = api_token if api_token is not None else config.LLM_API_TOKEN

    def _ensure_client(self) -> object:
        """Lazily build the OpenAI client — never at import time (AGENTS.md §1)."""
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self._api_token or "not-set",
                timeout=self.timeout,
                max_retries=self.max_retries,
            )
        return self._client

    def describe(self) -> dict:
        return {
            "judge_model": self.model,
            "judge_base_url": self.base_url,
            "judge_temperature": self.temperature,
            "judge_temperature_requested": self.temperature_requested,
            "judge_timeout": self.timeout,
            "judge_calls": self.calls,
            "judge_errors": self.errors,
            "judge_content_filtered": self.content_filtered,
            "judge_sees_deterministic_verdict": False,
            "judge_sees_retrieved_chunks": False,
        }

    def build_prompt(self, item: EvalItem, answer: str) -> str:
        return (
            f"DOMANDA:\n{item.domanda}\n\n"
            f"RISPOSTA ATTESA (verificata sul manuale):\n{item.risposta_attesa}\n\n"
            f"COMPORTAMENTO ATTESO:\n{item.comportamento_atteso}\n\n"
            f"RISPOSTA DEL CHATBOT:\n{answer or '(nessuna risposta)'}\n"
        )

    def _complete(self, user_prompt: str) -> str:
        """One chat completion, dropping ``temperature`` if the model rejects it."""
        client = self._ensure_client()
        request = {
            "model": self.model,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        }
        temperature = self.temperature
        try:
            if temperature is not None:
                response = client.chat.completions.create(**request, temperature=temperature)
            else:
                response = client.chat.completions.create(**request)
        except Exception as exc:
            if temperature is None or not _rejects_temperature(exc):
                raise
            with self._lock:
                if self.temperature is not None:
                    logger.warning(
                        "judge model %s rejected temperature=%s; using the server default from now on: %s",
                        self.model,
                        temperature,
                        exc,
                    )
                    self.temperature = None
            response = client.chat.completions.create(**request)
        return response.choices[0].message.content or ""

    def probe(self) -> None:
        """Send one trivial request; raise ``JudgeUnavailable`` if it fails.

        Run before the questions so an unreachable or misconfigured judge stops
        the run up front instead of failing silently on every item.
        """
        try:
            raw = self._complete(PROBE_PROMPT)
        except Exception as exc:
            raise JudgeUnavailable(f"judge {self.model} at {self.base_url or 'default URL'} failed: {exc}") from exc
        if parse_verdict(raw).verdict == "?":
            raise JudgeUnavailable(f"judge {self.model} returned an unusable probe reply: {raw[:200]!r}")

    def judge(self, item: EvalItem, answer: str) -> JudgeVerdict:
        with self._lock:
            self.calls += 1
        try:
            raw = self._complete(self.build_prompt(item, answer))
        except Exception as exc:
            categories = filtered_categories(exc)
            with self._lock:
                self.errors += 1
                if categories is not None:
                    self.content_filtered += 1
            if categories is not None:
                raise JudgeContentFiltered(describe_block(categories)) from exc
            raise
        return parse_verdict(raw)

    def __call__(self, item: EvalItem, answer: str) -> tuple[str, str]:
        result = self.judge(item, answer)
        return result.verdict, result.reason


def parse_verdict(raw: str) -> JudgeVerdict:
    """Parse the judge's JSON reply, tolerating a fenced or padded response.

    An unparsable reply becomes ``?`` with the raw text as the reason rather
    than an exception: a broken judge must not cost the captured answer.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.split("\n", 1)[-1] if "\n" in text else text
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        text = text[start : end + 1]
    try:
        payload = json.loads(text)
    except (ValueError, TypeError):
        return JudgeVerdict(verdict="?", reason=f"unparsable judge reply: {raw[:200]!r}")
    verdict = str(payload.get("verdict", "")).strip().upper()[:1]
    reason = str(payload.get("reason", "")).strip()
    if verdict not in VALID_VERDICTS:
        return JudgeVerdict(verdict="?", reason=f"judge returned an unknown verdict {verdict!r}: {reason}")
    return JudgeVerdict(verdict=verdict, reason=reason)
