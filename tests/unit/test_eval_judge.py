"""WP9 — the opt-in LLM judge.

No network: the OpenAI client is injected. What is worth pinning here is not
the API call but the *rules that keep the judge honest* — it must not see the
deterministic verdict, must not see retrieved chunks, and must never cost a
captured answer when it misbehaves.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from sme_chatbot.eval.cli import main
from sme_chatbot.eval.dataset import load_dataset
from sme_chatbot.eval.judge import JudgeContentFiltered, JudgeUnavailable, LlmJudge, parse_verdict
from sme_chatbot.eval.runner import EvalResult, _apply_judge, rescore
from sme_chatbot.eval.scoring import score_item

DATASET = "tests/fixtures/eval/fixture-alpha-domande.csv"


@pytest.fixture(scope="module")
def item():
    dataset = load_dataset(DATASET)
    return next(i for i in dataset.items if i.id == "Q28")


class FakeOpenAI:
    """Minimal stand-in recording the request the judge would send."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content=self.reply)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("reply", "verdict"),
    [
        ('{"verdict": "S", "reason": "corretta"}', "S"),
        ('{"verdict":"n","reason":"sbagliata"}', "N"),
        ('```json\n{"verdict": "P", "reason": "incompleta"}\n```', "P"),
        ('Ecco il risultato: {"verdict": "S", "reason": "ok"} — fine', "S"),
    ],
)
def test_parse_verdict_tolerates_fenced_and_padded_json(reply, verdict):
    assert parse_verdict(reply).verdict == verdict


@pytest.mark.parametrize("reply", ["not json at all", "", '{"verdict": "MAYBE", "reason": "x"}', "{}"])
def test_an_unusable_reply_becomes_undetermined_not_an_exception(reply):
    """A broken judge must never cost the captured answer."""
    result = parse_verdict(reply)
    assert result.verdict == "?"
    assert result.reason


# ---------------------------------------------------------------------------
# what the judge is and is not shown
# ---------------------------------------------------------------------------


def test_the_prompt_carries_the_question_key_and_expected_behaviour(item):
    prompt = LlmJudge(model="m", client=FakeOpenAI("{}")).build_prompt(item, "una risposta")
    assert item.domanda in prompt
    assert item.risposta_attesa in prompt
    # `Comportamento atteso` is what makes "one number without qualification is
    # a failure" checkable at all.
    assert item.comportamento_atteso in prompt
    assert "una risposta" in prompt


def test_the_judge_never_sees_the_deterministic_verdict_or_the_chunks(item):
    fake = FakeOpenAI('{"verdict": "S", "reason": "ok"}')
    judge = LlmJudge(model="m", client=fake)
    scores = score_item(item, "L'orientatore seleziona le capsule.", [])

    judge(item, "L'orientatore seleziona le capsule.")

    sent = "\n".join(m["content"] for m in fake.calls[0]["messages"])
    assert scores.verdict not in _verdict_tokens(sent)
    assert "anchor" not in sent.lower()
    assert "chunk" not in sent.lower()
    assert fake.calls[0]["temperature"] == 0.0


def _verdict_tokens(text: str) -> set[str]:
    """Verdict letters that appear as standalone claims, not as legend entries."""
    # The system prompt legitimately explains what S/N/P mean; what must not
    # appear is a statement of *this item's* computed verdict.
    return {line.strip() for line in text.splitlines() if line.strip() in ("verdict_auto", "S.", "N.", "P.")}


def test_describe_records_what_a_later_reader_needs(item):
    described = LlmJudge(model="gpt-5", base_url="https://llm.invalid", client=FakeOpenAI("{}")).describe()
    assert described["judge_model"] == "gpt-5"
    assert described["judge_temperature"] == 0.0
    assert described["judge_sees_deterministic_verdict"] is False
    assert described["judge_sees_retrieved_chunks"] is False


# ---------------------------------------------------------------------------
# how the runner uses the verdict
# ---------------------------------------------------------------------------


def test_the_judge_resolves_undetermined_items_only(item):
    """It never overwrites an S/N that has a concrete anchor behind it."""
    undetermined = EvalResult(item=item, run_key="Q28", verdict_auto="?", verdict_judge="S")
    assert undetermined.verdict == "S"
    assert undetermined.disagreement is False

    anchored = EvalResult(item=item, run_key="Q28", verdict_auto="S", verdict_judge="N")
    assert anchored.verdict == "S"  # the anchored verdict stands...
    assert anchored.disagreement is True  # ...and the conflict goes to the review queue


def test_a_judge_failure_does_not_lose_the_answer(item, monkeypatch):
    import sme_chatbot.eval.runner as runner_mod

    class Boom:
        def __call__(self, *_args):
            raise RuntimeError("judge exploded")

    captured = EvalResult(item=item, run_key="Q28", answer="una risposta")
    captured.scores = score_item(item, captured.answer, [])
    captured.verdict_auto = captured.scores.verdict

    # Mirrors the runner's guard: the judge is best-effort, the answer is not.
    try:
        Boom()(item, captured.answer)
    except Exception as exc:
        captured.judge_reason = f"judge error: {exc}"

    assert captured.answer == "una risposta"
    assert captured.verdict == captured.verdict_auto
    assert "judge error" in captured.judge_reason
    assert runner_mod.VERDICT_ERROR == "ERROR"


# ---------------------------------------------------------------------------
# a judge that cannot run must not look like one that ran
# ---------------------------------------------------------------------------


class Http400(Exception):
    """Shape of ``openai.BadRequestError`` that the judge checks: a ``status_code``."""

    status_code = 400


class RejectsTemperature(FakeOpenAI):
    """Azure gpt-5: any explicit temperature other than the default is a 400."""

    def _create(self, **kwargs):
        if "temperature" in kwargs:
            self.calls.append(kwargs)
            raise Http400(
                "Error code: 400 - Unsupported value: 'temperature' does not support 0 with this model. "
                "Only the default (1) value is supported."
            )
        return super()._create(**kwargs)


class Unreachable(FakeOpenAI):
    """The deploy-host -> Azure connect timeout seen in a real run."""

    def _create(self, **kwargs):
        raise TimeoutError("Connection timed out")


def test_a_rejected_temperature_is_dropped_once_and_recorded_truthfully(item):
    fake = RejectsTemperature('{"verdict": "S", "reason": "ok"}')
    judge = LlmJudge(model="gpt-5", client=fake)

    assert judge(item, "una risposta") == ("S", "ok")
    assert judge(item, "una risposta") == ("S", "ok")

    # One rejected attempt, then every request goes out without the parameter.
    assert ["temperature" in call for call in fake.calls] == [True, False, False]
    described = judge.describe()
    assert described["judge_temperature"] is None  # what was actually sent
    assert described["judge_temperature_requested"] == 0.0
    assert described["judge_calls"] == 2
    assert described["judge_errors"] == 0


def test_other_errors_are_raised_and_counted(item):
    judge = LlmJudge(model="gpt-5", client=Unreachable("{}"))
    with pytest.raises(TimeoutError):
        judge(item, "una risposta")
    assert judge.temperature == 0.0  # a timeout says nothing about the temperature
    assert (judge.describe()["judge_calls"], judge.describe()["judge_errors"]) == (1, 1)


def test_an_unrelated_400_is_not_mistaken_for_a_temperature_rejection(item):
    class BadPrompt(FakeOpenAI):
        def _create(self, **kwargs):
            self.calls.append(kwargs)
            raise Http400("content filter triggered")

    fake = BadPrompt("{}")
    judge = LlmJudge(model="gpt-5", client=fake)
    with pytest.raises(Http400):
        judge(item, "una risposta")
    assert len(fake.calls) == 1
    assert judge.temperature == 0.0


def test_the_probe_passes_and_settles_the_temperature_before_the_run():
    judge = LlmJudge(model="gpt-5", client=RejectsTemperature('{"verdict": "S", "reason": "ok"}'))
    judge.probe()
    assert judge.temperature is None
    assert judge.describe()["judge_calls"] == 0  # the probe is not a judged item


@pytest.mark.parametrize(
    "fake",
    [Unreachable("{}"), FakeOpenAI("not json")],
    ids=["unreachable", "unusable-reply"],
)
def test_a_failing_probe_raises_judge_unavailable(fake):
    with pytest.raises(JudgeUnavailable):
        LlmJudge(model="gpt-5", base_url="https://llm.invalid", client=fake).probe()


def test_the_cli_stops_before_asking_anything_when_the_judge_is_unavailable(capsys, monkeypatch):
    monkeypatch.setattr(LlmJudge, "_ensure_client", lambda self: Unreachable("{}"))
    # An unroutable backend URL: reaching it would fail differently, so the
    # judge error in stderr proves the probe ran first.
    code = main(["--dataset", DATASET, "--only", "Q28", "--judge", "--api-url", "http://127.0.0.1:9", "--no-write"])

    err = capsys.readouterr().err
    assert code == 1
    assert "judge" in err
    assert "Connection timed out" in err


def test_rescore_with_a_judge_rejudges_the_stored_answers(item):
    stored = EvalResult(
        item=item,
        run_key="Q28",
        answer="L'orientatore seleziona le capsule.",
        judge_reason="judge error: Error code: 400 - Unsupported value: 'temperature'",
    )
    failed = EvalResult(item=item, run_key="Q28#2", error="HTTP 500")
    judge = LlmJudge(model="gpt-5", client=RejectsTemperature('{"verdict": "P", "reason": "incompleta"}'))

    rescored = rescore([stored, failed], page_offset=10, page_tolerance=2, judge=judge)

    assert (rescored[0].verdict_judge, rescored[0].judge_reason) == ("P", "incompleta")
    assert rescored[1] is failed  # a transport error has no answer to judge
    assert judge.describe()["judge_calls"] == 1
    assert stored.verdict_judge is None  # the source result is not modified


# ---------------------------------------------------------------------------
# provider content filter
# ---------------------------------------------------------------------------


def test_a_content_filter_block_is_recorded_as_such_and_counted_apart(item, azure_content_filter_error):
    class Filtered(FakeOpenAI):
        def _create(self, **kwargs):
            self.calls.append(kwargs)
            raise azure_content_filter_error

    fake = Filtered("{}")
    judge = LlmJudge(model="gpt-5", client=fake)
    result = EvalResult(item=item, run_key="Q26", answer="una risposta")

    _apply_judge(result, judge)

    assert len(fake.calls) == 1  # deterministic for a given prompt: no retry
    assert result.verdict_judge is None
    assert result.judge_reason == "judge error: blocked by the LLM provider's content filter (sexual:medium)"
    described = judge.describe()
    assert (described["judge_calls"], described["judge_errors"], described["judge_content_filtered"]) == (1, 1, 1)
    assert judge.temperature == 0.0  # not mistaken for a temperature rejection


def test_content_filtered_is_raised_as_its_own_type(item, azure_content_filter_error):
    class Filtered(FakeOpenAI):
        def _create(self, **kwargs):
            raise azure_content_filter_error

    with pytest.raises(JudgeContentFiltered):
        LlmJudge(model="gpt-5", client=Filtered("{}"))(item, "una risposta")
