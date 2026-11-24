"""Framework-agnostic streaming chat loop.

Port of the single-streaming-loop tool-dispatch design from the legacy
``chatbot_core.py::make_chat_fn`` (see
``docs/specs/2026-07-02-backend-split-and-docsvc-ingestion-plan.md`` §5 WP2).
Differences from the legacy version:

- Yields ``TokenEvent`` *deltas* instead of cumulative strings (Gradio wanted
  the running total; SSE clients want deltas and render the total themselves).
- ``<think>...</think>`` stripping (Qwen3 chain-of-thought) works across
  streaming delta boundaries — the legacy code re-stripped the whole
  cumulative string on every chunk, which is not an option once we only see
  one delta at a time. See ``_ThinkStripper`` below and plan §9 risk note.
- Emits ``ToolCallEvent``/``ToolResultEvent`` around each dispatched tool call,
  extracting ``Citation`` objects from RAG-shaped tool results.
- No ``gradio``/``fastapi`` import; no OpenAI client construction here — the
  caller (API layer) builds and passes the client in.
- On hitting ``max_tool_rounds``, makes one extra tools-less completion call
  to force a text answer from whatever context was already gathered, instead
  of returning empty content (the legacy cutoff returned
  ``visible_accum`` as-is, which was empty whenever every round up to the cap
  was a pure tool call with no narration text).
"""

from __future__ import annotations

import contextvars
import json
import logging
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

from .content_filter import OutputFiltered, categories_in, describe_block, filtered_categories
from .domains import DomainConfig
from .events import (
    ChatEvent,
    Citation,
    DoneEvent,
    ErrorEvent,
    RequestIdentity,
    TokenEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from .llm_api import CHAT_COMPLETIONS, open_stream

logger = logging.getLogger(__name__)

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"

# Monotonic clock for the per-turn timings; a module attribute so tests can
# drive it deterministically instead of sleeping.
_now = time.perf_counter

# ``stream_chat``'s own default for ``tool_concurrency``. The API layer passes the
# configured value only when it differs from this, like the other optional settings.
DEFAULT_TOOL_CONCURRENCY = 4


def _ms(seconds: float) -> int:
    return int(seconds * 1000)


class _TurnTimer:
    """Collects the per-stage timings of one ``stream_chat`` turn.

    Reported as ``DoneEvent.usage["timings"]``. ``ttft_ms`` is the first
    *visible* (``<think>``-stripped) token of the turn; ``first_chunk_ms`` of
    each LLM call is the first chunk of any kind, so a long silent thinking
    phase shows up there instead of being hidden.
    """

    def __init__(self) -> None:
        self._start = _now()
        self.ttft_ms: int | None = None
        self.llm_calls: list[dict] = []
        self.tools: list[dict] = []

    def mark_visible(self) -> None:
        if self.ttft_ms is None:
            self.ttft_ms = _ms(_now() - self._start)

    def record_llm_call(
        self, round_num: int, started: float, first_chunk: float | None, finish: str
    ) -> None:
        self.llm_calls.append(
            {
                "round": round_num,
                "first_chunk_ms": None if first_chunk is None else _ms(first_chunk - started),
                "total_ms": _ms(_now() - started),
                "finish": finish,
            }
        )

    def record_tool(self, name: str, ms: int, ok: bool) -> None:
        self.tools.append({"name": name, "ms": ms, "ok": ok})

    def usage(self, rounds: int, round_limit_hit: bool = False) -> dict:
        return {
            "rounds": rounds,
            # True when the turn ran out of tool rounds and its answer was forced;
            # ``rounds`` alone can't tell that from an answer given in the last allowed round.
            "round_limit_hit": round_limit_hit,
            "timings": {
                "total_ms": _ms(_now() - self._start),
                "ttft_ms": self.ttft_ms,
                "llm_ms": sum(c["total_ms"] for c in self.llm_calls),
                "tool_ms": sum(t["ms"] for t in self.tools),
                "llm_calls": self.llm_calls,
                "tools": self.tools,
            },
        }


def _longest_partial_suffix(text: str, tag: str) -> int:
    """Length of the longest suffix of ``text`` that is a proper prefix of ``tag``.

    Used to hold back a trailing fragment that might be the start of ``tag``
    continuing in the next streamed delta (e.g. buffer holds "...<thi" and the
    next delta starts with "nk>").
    """
    max_k = min(len(text), len(tag) - 1)
    for k in range(max_k, 0, -1):
        if text.endswith(tag[:k]):
            return k
    return 0


class _ThinkStripper:
    """Incrementally strips ``<think>…</think>`` blocks from a delta stream.

    The legacy implementation ran a regex over the *cumulative* string on
    every chunk. That doesn't work when we only emit deltas: a naive per-delta
    regex would leak partial tag fragments (e.g. yield ``"<thi"`` then
    ``"nk>reasoning..."``) whenever a tag is split across two stream chunks.
    This class buffers internally and only releases text once it is certain
    it is not part of an in-progress tag.
    """

    def __init__(self) -> None:
        self._buf = ""
        self._in_think = False
        self._started = False  # becomes True once real (non-newline) visible text is emitted

    def feed(self, delta_text: str) -> str:
        """Feed a new raw delta; return the visible (stripped) delta, if any."""
        if not delta_text:
            return ""
        self._buf += delta_text
        pieces: list[str] = []

        while True:
            if not self._in_think:
                idx = self._buf.find(_THINK_OPEN)
                if idx == -1:
                    keep = _longest_partial_suffix(self._buf, _THINK_OPEN)
                    cut = len(self._buf) - keep
                    if cut > 0:
                        pieces.append(self._buf[:cut])
                    self._buf = self._buf[cut:]
                    break
                if idx > 0:
                    pieces.append(self._buf[:idx])
                self._buf = self._buf[idx + len(_THINK_OPEN) :]
                self._in_think = True
                continue
            else:
                idx = self._buf.find(_THINK_CLOSE)
                if idx == -1:
                    keep = _longest_partial_suffix(self._buf, _THINK_CLOSE)
                    # Nothing inside an open <think> block is ever emitted;
                    # only retain a tail that could be a split closing tag.
                    self._buf = self._buf[len(self._buf) - keep :] if keep else ""
                    break
                self._buf = self._buf[idx + len(_THINK_CLOSE) :]
                self._in_think = False
                continue

        visible = "".join(pieces)
        if not self._started:
            visible = visible.lstrip("\n")
            if visible:
                self._started = True
        return visible


def _extract_citations(tool_result_json: str) -> list[Citation]:
    """Extract ``Citation`` objects from a tool result JSON string.

    Recognises the ``{"results": [{"source", "page", ...}, ...]}`` shape
    produced by ``search_docs``/``search_manuals`` and the ``{"filename",
    "repository", "document_id", "page", ...}`` shape produced by
    ``search_documents``. Anything else (errors, unrelated shapes) yields no
    citations.

    ``Citation.source`` is normalised to the document's name: tools that
    report a ``filename`` win over a bare ``source``, so the footer can never
    again label a chunk with its origin network share.
    """
    try:
        data = json.loads(tool_result_json)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(data, dict):
        return []
    results = data.get("results")
    if not isinstance(results, list):
        return []

    citations = []
    for item in results:
        if not isinstance(item, dict):
            continue
        if not ({"source", "filename", "page"} & item.keys()):
            continue
        filename = item.get("filename")
        citations.append(
            Citation(
                source=filename or item.get("source"),
                page=item.get("page"),
                document_id=item.get("document_id"),
                distance=item.get("distance"),
                filename=filename,
                repository=item.get("repository"),
            )
        )
    return citations


def _parse_tool_arguments(raw: str) -> tuple[dict | None, str | None]:
    """``(arguments, None)``, or ``(None, error_result)`` when ``raw`` is not valid JSON.

    A call with invalid arguments is never dispatched: the error result goes back to the model instead.
    """
    try:
        return json.loads(raw or "{}"), None
    except json.JSONDecodeError as exc:
        return None, json.dumps({"error": f"Invalid arguments: {exc}"})


def _run_tool(dispatch, name: str, args: dict, label: str) -> tuple[str, bool, int]:
    """Dispatch one tool call. Returns ``(result, ok, ms)``.

    Touches no state of the turn, so it can run off the loop's thread: the caller records
    the outcome with ``_finish_tool``. ``label`` names the call in the failure log.
    """
    started = _now()
    try:
        result = dispatch(name, args)
        ok = True
    except Exception as exc:  # broad on purpose: never crash the chat loop
        logger.exception("%s %s failed", label, name)
        result = json.dumps({"error": f"Tool {name} failed: {exc}"})
        ok = False
    return result, ok, _ms(_now() - started)


def _finish_tool(
    call_id: str,
    name: str,
    result: str,
    ok: bool,
    ms: int,
    messages: list[dict],
    timer: _TurnTimer,
    all_citations: list[Citation],
) -> ToolResultEvent:
    """Record one finished tool call in the turn: timing, citations, tool message. Returns its event."""
    timer.record_tool(name, ms, ok)
    citations = _extract_citations(result) if ok else []
    all_citations.extend(citations)
    messages.append({"role": "tool", "tool_call_id": call_id, "content": result})
    return ToolResultEvent(name=name, ok=ok, citations=citations)


def _dispatch_round_concurrently(
    calls: list[dict],
    dispatch,
    tool_concurrency: int,
    messages: list[dict],
    timer: _TurnTimer,
    all_citations: list[Citation],
) -> Iterator[ChatEvent]:
    """Run one round's tool calls at the same time, up to ``tool_concurrency`` at once.

    Every ``ToolCallEvent`` of the round comes first, then the ``ToolResultEvent``s in the
    model's call order, each as soon as it and the calls before it are done. Results, tool
    messages, citations and timings are all recorded here, on the loop's thread, in call
    order, whatever order the calls finish in, so the conversation stays deterministic.

    A call with invalid arguments is never dispatched. Each dispatched call runs in its own
    copy of the caller's context: one ``Context`` cannot be entered by two threads at once.
    With a single valid call there is nothing to overlap, and it runs inline.
    """
    parsed = [(tc, *_parse_tool_arguments(tc["arguments"])) for tc in calls]
    for tc, args, invalid in parsed:
        yield ToolCallEvent(name=tc["name"], arguments={} if invalid is not None else args)
        if invalid is None:
            logger.info("Tool call: %s(%s)", tc["name"], args)

    valid = [i for i, (_, _, invalid) in enumerate(parsed) if invalid is None]
    pool = None
    if len(valid) > 1:
        pool = ThreadPoolExecutor(max_workers=min(tool_concurrency, len(valid)), thread_name_prefix="tool")
    try:
        futures = {}
        if pool is not None:
            for i in valid:
                tc, args, _ = parsed[i]
                context = contextvars.copy_context()
                futures[i] = pool.submit(context.run, _run_tool, dispatch, tc["name"], args, "Tool")
        for i, (tc, args, invalid) in enumerate(parsed):
            if invalid is not None:
                result, ok, ms = invalid, False, 0
            elif pool is None:
                result, ok, ms = _run_tool(dispatch, tc["name"], args, "Tool")
            else:
                result, ok, ms = futures[i].result()
            yield _finish_tool(tc["id"], tc["name"], result, ok, ms, messages, timer, all_citations)
    finally:
        # Normally every call is done by now. If the consumer closed the stream early, don't
        # block on calls still running: they finish in the background and are discarded.
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)


_LENGTH_RETRY_PROMPT = (
    "Your previous attempt used all of its output budget before giving an answer. "
    "Answer now, briefly, using only the information already gathered above; do "
    "not request tool calls and do not reason at length. If part of the answer "
    "was not found, say so plainly."
)


def _stream_final_call(
    client,
    llm_api: str,
    model: str,
    messages: list[dict],
    gen_params: dict,
    timer: _TurnTimer,
    round_num: int,
    label: str,
) -> Iterator[ChatEvent]:
    """One tool-less streaming call. Yields ``TokenEvent``s; returns ``(visible_text, finish_reason)``.

    Uses its own ``<think>`` stripper: a stripper carried over from an earlier call could
    still be inside an unclosed block and swallow this call's whole answer. A call that ends
    on ``max_tokens`` is recorded as ``"length"`` whatever ``label`` says, so the eval report
    counts it.
    """
    started = _now()
    first_chunk: float | None = None
    stripper = _ThinkStripper()
    visible = ""
    finish_reason = None
    flagged: list[str] = []
    for chunk in open_stream(client, llm_api, model, messages, None, gen_params):
        if first_chunk is None:
            first_chunk = _now()
        if not chunk.choices:  # Ollama empty-choices sentinel
            continue
        choice = chunk.choices[0]
        finish_reason = choice.finish_reason or finish_reason
        flagged = categories_in(getattr(choice, "content_filter_results", None)) or flagged
        if choice.delta.content:
            piece = stripper.feed(choice.delta.content)
            if piece:
                timer.mark_visible()
                visible += piece
                yield TokenEvent(text=piece)
    if finish_reason == "content_filter":
        raise OutputFiltered(flagged)
    timer.record_llm_call(round_num, started, first_chunk, "length" if finish_reason == "length" else label)
    return visible, finish_reason


def _retry_after_length(
    client, llm_api, model, messages, gen_params, retry_reasoning_effort, timer, round_num, max_tokens
) -> Iterator[ChatEvent]:
    """The one retry for a call that ran out of output budget before showing anything.

    Tool-less, with a short "answer now" instruction, and with ``retry_reasoning_effort`` in
    place of ``reasoning_effort`` when it is set. Returns the visible text (``""`` if the
    retry produced none either).
    """
    logger.warning(
        "LLM call hit max_tokens=%s without producing an answer; retrying once without tools", max_tokens
    )
    messages.append({"role": "user", "content": _LENGTH_RETRY_PROMPT})
    params = gen_params
    if retry_reasoning_effort is not None:
        params = {**gen_params, "reasoning_effort": retry_reasoning_effort}
    text, _ = yield from _stream_final_call(client, llm_api, model, messages, params, timer, round_num, "length_retry")
    return text


def _no_answer_error(max_tokens: int | None) -> ErrorEvent:
    return ErrorEvent(
        message=f"The model reached its output limit (max_tokens={max_tokens}) without producing an answer."
    )


def _content_filter_message(categories: list[str]) -> str:
    return (
        f"The request was {describe_block(categories)}. This is usually a false positive on "
        "technical manual text; rephrasing the question may help."
    )


def _output_filter_message(categories: list[str]) -> str:
    return (
        f"The answer was stopped partway by the LLM provider's content filter "
        f"({', '.join(categories) or 'category not reported'}), so it is incomplete. "
        "This is usually a false positive on technical manual text; rephrasing the question may help."
    )


_TOOLS_WITH_REASONING_MESSAGE = (
    "The LLM provider refused function tools together with reasoning_effort for this model on "
    "LLM_API={llm_api}. Set LLM_API=responses on the backend, or LLM_REASONING_EFFORT=none."
)


def _is_tools_with_reasoning_rejection(exc: BaseException) -> bool:
    """Azure's 400 for tools plus ``reasoning_effort`` on ``/v1/chat/completions``.

    Seen as: "Function tools with reasoning_effort are not supported for gpt-6-luna in
    /v1/chat/completions. To use function tools, use /v1/responses or set reasoning_effort
    to 'none'."
    """
    text = str(exc)
    return getattr(exc, "status_code", None) == 400 and "reasoning_effort" in text and "tools" in text.lower()


def stream_chat(
    client,
    model: str,
    domain: DomainConfig,
    message: str,
    history: list[dict],
    identity: RequestIdentity,
    max_tool_rounds: int = 5,
    temperature: float | None = None,
    seed: int | None = None,
    max_tokens: int | None = None,
    reasoning_effort: str | None = None,
    retry_reasoning_effort: str | None = None,
    pre_search: bool = False,
    llm_api: str = CHAT_COMPLETIONS,
    tool_concurrency: int = DEFAULT_TOOL_CONCURRENCY,
) -> Iterator[ChatEvent]:
    """Run one user turn against ``domain`` and yield ``ChatEvent``s.

    ``identity`` is accepted (and logged) for forward compatibility with
    identity-aware domains (WP7): tools that need it are bound via closure in
    the ``DomainConfig.dispatch`` callable itself, so this loop never needs to
    thread it through the OpenAI tool-calling protocol.

    ``pre_search``: when true and the domain defines one (``DomainConfig.pre_search``), that
    search runs before the first LLM call and goes into the conversation as an assistant tool
    call plus its result, so the model can answer in its first call instead of spending one
    to ask for the search (each LLM call costs about 3 s of TTFT). The model can
    still search again.

    ``llm_api``: ``"chat_completions"`` (the default) or ``"responses"``; see
    ``llm_api.py``. Either way the loop below sees chat-completions-shaped chunks.

    ``tool_concurrency``: how many tool calls of one round may run at the same time
    (``CHAT_TOOL_CONCURRENCY``); see ``_dispatch_round_concurrently``. ``1`` keeps the
    sequential loop exactly: each call's ``ToolCallEvent`` then its ``ToolResultEvent``, one
    call after another. A round with a single call always takes that path.
    """
    logger.info(
        "stream_chat: domain=%s user=%s message=%r (history length: %d)",
        domain.key,
        identity.user_id,
        message,
        len(history),
    )

    # Generation settings are sent only when set, so an unconfigured backend keeps
    # the server's own defaults. ``is not None``: temperature 0 is a real value.
    gen_params: dict = {}
    if temperature is not None:
        gen_params["temperature"] = temperature
    if seed is not None:
        gen_params["seed"] = seed
    if max_tokens is not None:
        gen_params["max_tokens"] = max_tokens
    if reasoning_effort is not None:
        gen_params["reasoning_effort"] = reasoning_effort

    messages: list[dict] = (
        [{"role": "system", "content": domain.system_prompt}]
        + list(history)
        + [{"role": "user", "content": message}]
    )

    all_citations: list[Citation] = []
    timer = _TurnTimer()
    # Accumulated visible text for the *whole turn*, not just the last round.
    # A `TokenEvent` is emitted for every visible delta of every round, so a
    # client that concatenates tokens holds exactly this string; `DoneEvent.
    # content` has to match it or the client is forced to choose between its
    # own accumulation and ours. It used to be reset per round, so an answer
    # streamed in round N followed by one more tool round came back as `""` —
    # and the Gradio client, which renders `done.content`, replaced the
    # on-screen answer with nothing but the citation footer. The per-round `raw` below stays per-round: it feeds that round's
    # assistant message in the OpenAI tool-calling protocol.
    visible_accum = ""

    try:
        if pre_search and domain.pre_search is not None:
            name, args = domain.pre_search
            call_id = "pre_search_0"
            messages.append(
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
                    ],
                }
            )
            yield ToolCallEvent(name=name, arguments=args)
            logger.info("Pre-search: %s(%s)", name, args)
            result, ok, ms = _run_tool(domain.dispatch, name, args, "Pre-search")
            yield _finish_tool(call_id, name, result, ok, ms, messages, timer, all_citations)

        for round_num in range(max_tool_rounds + 1):
            call_started = _now()
            first_chunk_at: float | None = None
            stream = open_stream(client, llm_api, model, messages, domain.tool_definitions, gen_params)

            stripper = _ThinkStripper()
            raw = ""
            tool_calls_acc: dict[int, dict] = {}
            finish_reason = None
            flagged: list[str] = []

            for chunk in stream:
                if first_chunk_at is None:
                    first_chunk_at = _now()
                if not chunk.choices:  # Ollama emits empty-choices sentinel chunks
                    continue
                slot_chunk = chunk.choices[0]
                finish_reason = slot_chunk.finish_reason or finish_reason
                # Azure attaches a per-category breakdown to each choice; keep the last flagged one.
                flagged = categories_in(getattr(slot_chunk, "content_filter_results", None)) or flagged
                delta = slot_chunk.delta

                if delta.content:
                    raw += delta.content
                    visible_delta = stripper.feed(delta.content)
                    if visible_delta:
                        timer.mark_visible()
                        visible_accum += visible_delta
                        yield TokenEvent(text=visible_delta)

                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        slot = tool_calls_acc.setdefault(
                            tc.index, {"id": "", "name": "", "arguments": ""}
                        )
                        if tc.id:
                            slot["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                slot["name"] += tc.function.name
                            if tc.function.arguments:
                                slot["arguments"] += tc.function.arguments

            if finish_reason == "content_filter":
                raise OutputFiltered(flagged)

            wants_tools = finish_reason == "tool_calls" and bool(tool_calls_acc)
            if wants_tools:
                finish = "tool_calls"
            elif finish_reason == "length":
                finish = "length"
                logger.warning(
                    "LLM call hit max_tokens=%s in round %d; the answer may be missing or cut short",
                    max_tokens,
                    round_num,
                )
            else:
                finish = "stop"
            timer.record_llm_call(round_num, call_started, first_chunk_at, finish)

            if not wants_tools:
                if finish == "length" and not visible_accum.strip():
                    # Out of output budget before showing anything (a thinking model can spend all
                    # of it reasoning): one tool-less retry, then an error rather than an empty answer.
                    retry_text = yield from _retry_after_length(
                        client, llm_api, model, messages, gen_params, retry_reasoning_effort, timer, round_num, max_tokens
                    )
                    if not retry_text.strip():
                        yield _no_answer_error(max_tokens)
                        return
                    visible_accum += retry_text
                logger.info("Assistant response: %r", visible_accum)
                yield DoneEvent(
                    content=visible_accum,
                    citations=all_citations,
                    usage=timer.usage(round_num + 1),
                )
                return

            if round_num >= max_tool_rounds:
                logger.warning(
                    "Hit max_tool_rounds=%d; forcing a final answer without further tool calls",
                    max_tool_rounds,
                )
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "You have reached the maximum number of tool calls for this "
                            "turn. Answer now using only the information already gathered "
                            "above; do not request further tool calls. If part of the "
                            "answer was not found, say so plainly — do not offer to "
                            "search again in a later message."
                        ),
                    }
                )
                final_text, final_finish = yield from _stream_final_call(
                    client, llm_api, model, messages, gen_params, timer, round_num, "forced_final"
                )
                visible_accum += final_text
                if final_finish == "length" and not visible_accum.strip():
                    retry_text = yield from _retry_after_length(
                        client, llm_api, model, messages, gen_params, retry_reasoning_effort, timer, round_num, max_tokens
                    )
                    if not retry_text.strip():
                        yield _no_answer_error(max_tokens)
                        return
                    visible_accum += retry_text
                yield DoneEvent(
                    content=visible_accum,
                    citations=all_citations,
                    usage=timer.usage(round_num + 1, round_limit_hit=True),
                )
                return

            # Append the assistant turn that contained the tool calls.
            assistant_msg: dict = {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": tc["arguments"]},
                    }
                    for tc in tool_calls_acc.values()
                ],
            }
            if raw:
                assistant_msg["content"] = raw
            messages.append(assistant_msg)

            calls = list(tool_calls_acc.values())
            if tool_concurrency > 1 and len(calls) > 1:
                yield from _dispatch_round_concurrently(
                    calls, domain.dispatch, tool_concurrency, messages, timer, all_citations
                )
                continue

            # Dispatch each tool call and append its result, one after another.
            for tc in calls:
                args, invalid = _parse_tool_arguments(tc["arguments"])
                if invalid is not None:
                    yield ToolCallEvent(name=tc["name"], arguments={})
                    yield _finish_tool(tc["id"], tc["name"], invalid, False, 0, messages, timer, all_citations)
                    continue

                yield ToolCallEvent(name=tc["name"], arguments=args)
                logger.info("Tool call: %s(%s)", tc["name"], args)
                result, ok, ms = _run_tool(domain.dispatch, tc["name"], args, "Tool")
                yield _finish_tool(tc["id"], tc["name"], result, ok, ms, messages, timer, all_citations)
    except OutputFiltered as exc:
        # The answer stopped partway: the tokens already sent are not a
        # complete answer, so end on an error rather than a DoneEvent.
        logger.warning("stream_chat: answer %s", describe_block(exc.categories))
        yield ErrorEvent(message=_output_filter_message(exc.categories))
        return
    except Exception as exc:  # broad on purpose: surface as a terminal ErrorEvent
        categories = filtered_categories(exc)
        if categories is not None:
            # Not a bug and not retryable: the provider refused the prompt.
            # The prompt includes retrieved manual text, so the user's own words
            # may not be what was flagged.
            logger.warning("stream_chat: %s", describe_block(categories))
            yield ErrorEvent(message=_content_filter_message(categories))
            return
        if _is_tools_with_reasoning_rejection(exc):
            logger.warning("stream_chat: provider rejected function tools with reasoning_effort: %s", exc)
            yield ErrorEvent(message=_TOOLS_WITH_REASONING_MESSAGE.format(llm_api=llm_api))
            return
        logger.exception("stream_chat: unrecoverable error")
        yield ErrorEvent(message=str(exc))
        return
