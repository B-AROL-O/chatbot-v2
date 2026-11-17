"""Gradio frontend — thin HTTP/SSE client of the SME chatbot backend (WP4).

Talks to the FastAPI backend (``sme_chatbot.api.main``) exclusively over
HTTP, per the frozen contract in
``docs/specs/2026-07-02-backend-split-and-docsvc-ingestion-plan.md`` §4.1.
Deliberately has **no imports from ``sme_chatbot``** — this file dogfoods
the exact same contract the future .NET portal proxy will consume. Only
stdlib, ``httpx`` and ``gradio`` are used.

Run standalone with the backend already up (``uv run sme-chatbot-api``):

    uv run python frontends/gradio_app.py
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable, Iterator

import gradio as gr
import httpx

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger(__name__)

CHATBOT_API_URL = os.environ.get("CHATBOT_API_URL", "http://localhost:8000").rstrip("/")
GRADIO_USER_ID = os.environ.get("GRADIO_USER_ID")
CHATBOT_SHARE = os.environ.get("CHATBOT_SHARE", "false").strip().lower() in ("1", "true", "yes")

# Used only if the backend is unreachable at startup, when we can't ask
# GET /health for the real domain list.
FALLBACK_DOMAINS = ["sample_cdo", "sample_manuals", "documents"]

_CITATION_FOOTER_MARK = "\n\n— fonte:"

# Caps on the citation footer. Retrieval can return dozens of chunks for one
# turn; grouping by document already collapses most of that, but a broad query
# over a large library can still cite many documents or many pages of one
# catalogue. Overflow is stated explicitly rather than silently dropped.
_MAX_FOOTER_DOCUMENTS = 8
_MAX_FOOTER_PAGES_PER_DOCUMENT = 10

# Shown when the turn ends with retrieved sources but no answer text at all
# (e.g. the tool-calling loop hit its round cap without ever generating one).
# A bare citation footer under an empty answer reads as a broken UI and hides
# the fact that nothing was answered.
_NO_ANSWER_MESSAGE = (
    "⚠️ Non sono riuscito a formulare una risposta a questa domanda. "
    "Le fonti consultate sono elencate qui sotto; prova a riformulare la "
    "domanda o a indicare il numero di matricola della macchina."
)


def _friendly_backend_down_message(api_url: str) -> str:
    return (
        "⚠️ Non riesco a contattare il backend del chatbot "
        f"({api_url}). Riprova più tardi."
    )


def _headers() -> dict[str, str]:
    headers = {"content-type": "application/json"}
    if GRADIO_USER_ID:
        headers["X-User-Id"] = GRADIO_USER_ID
    return headers


def fetch_health(api_url: str = CHATBOT_API_URL, timeout: float = 5.0) -> tuple[list[str], str | None]:
    """``GET /health`` once; return ``(domains, version)``.

    One request backs both ``fetch_domains`` (kept for standalone/backward-
    compatible use) and the version shown in the UI description
    (``build_app``) - no second round trip just to read one more field.

    Falls back to ``(FALLBACK_DOMAINS, None)`` if the backend is unreachable
    or returns something unexpected — this is what lets the module be
    imported and the UI built while the backend is down. ``None`` means
    "unknown", not "the field doesn't exist" — a real backend always sets
    ``version`` (``config.APP_VERSION``).
    """
    try:
        resp = httpx.get(f"{api_url}/health", timeout=timeout)
        resp.raise_for_status()
        body = resp.json()
        version = body.get("version")
        domains = body.get("domains")
        if domains:
            return list(domains), version
        logger.warning("Backend /health returned no domains; using fallback list %s", FALLBACK_DOMAINS)
        return list(FALLBACK_DOMAINS), version
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning(
            "Backend unreachable at startup (%s: %s); using fallback domain list %s",
            api_url,
            exc,
            FALLBACK_DOMAINS,
        )
    return list(FALLBACK_DOMAINS), None


def fetch_domains(api_url: str = CHATBOT_API_URL, timeout: float = 5.0) -> list[str]:
    """``GET /health`` and return its ``domains`` list — see ``fetch_health``."""
    domains, _version = fetch_health(api_url, timeout)
    return domains


def _strip_citation_footer(content: str) -> str:
    """Undo ``_format_citation_footer`` so history sent back upstream is clean."""
    idx = content.find(_CITATION_FOOTER_MARK)
    if idx == -1:
        return content
    return content[:idx]


def _content_to_text(content: object) -> str | None:
    """Normalize a history turn's ``content`` to plain text.

    Gradio 6 delivers ``content`` as a list of content blocks
    (``[{"text": ..., "type": "text"}]``); plain strings are kept for
    backward compatibility. Returns ``None`` when the turn carries no text
    at all (files/components), so callers can drop it.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = [
            block["text"]
            for block in content
            if isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        ]
        if texts:
            return "".join(texts)
    return None


def _to_backend_history(history: list[dict]) -> list[dict]:
    """Map Gradio's openai-style history to the §4.1 ``history`` array.

    Only ``user``/``assistant`` text turns are forwarded; citation
    footers this module appended to assistant turns are stripped back out
    so the backend only ever sees the model's own words.
    """
    out = []
    for turn in history:
        role = turn.get("role")
        if role not in ("user", "assistant"):
            continue
        content = _content_to_text(turn.get("content"))
        if content is None:
            # Turns with no text (files/components) aren't part of the
            # backend contract - drop rather than send garbage.
            continue
        if role == "assistant":
            content = _strip_citation_footer(content)
        out.append({"role": role, "content": content})
    return out


def _group_citations(citations: list[dict]) -> list[dict]:
    """Collapse per-chunk citations into one group per cited document.

    De-duplication keys on the label — the document's filename — which is
    also what the reader sees, so the footer can never show the same line
    twice. The old key was ``(source, page)`` with ``source`` set to the
    origin network share, which collapsed two *distinct* documents cited at
    the same page into one line; now that ``source`` carries the filename,
    that defect is gone.

    Keying on ``document_id`` instead would be wrong here: the same file is
    present in the store under several document ids (``M05586_manual_fr.pdf``
    has four, two of them full 319-chunk copies), so an id-keyed footer
    repeats one filename verbatim. That duplication is an ingestion/
    DocumentService data issue, not a rendering one — the footer just must
    not put it in front of the user.

    Groups come back ordered by their best (lowest) semantic distance, so the
    most relevant source leads; distance-less citations (legacy tool payloads)
    sort last but keep their arrival order, so the footer is never reshuffled
    arbitrarily.
    """
    groups: dict[object, dict] = {}
    for c in citations:
        label = c.get("filename") or c.get("source")
        if not label:
            continue
        group = groups.setdefault(label, {"label": label, "pages": [], "distance": None})
        page = c.get("page")
        if page is not None and page not in group["pages"]:
            group["pages"].append(page)
        distance = c.get("distance")
        if isinstance(distance, (int, float)) and (
            group["distance"] is None or distance < group["distance"]
        ):
            group["distance"] = distance

    return sorted(
        groups.values(),
        key=lambda g: (g["distance"] is None, g["distance"] if g["distance"] is not None else 0.0),
    )


def _format_group(group: dict) -> str:
    """Render one grouped document as ``— fonte: file.pdf, pp. 9, 22, 79``."""
    pages = sorted(p for p in group["pages"] if isinstance(p, int))
    pages += [p for p in group["pages"] if not isinstance(p, int)]
    if not pages:
        return f"— fonte: {group['label']}"
    shown = pages[:_MAX_FOOTER_PAGES_PER_DOCUMENT]
    rendered = ", ".join(str(p) for p in shown)
    if len(pages) > len(shown):
        rendered += f", … (+{len(pages) - len(shown)})"
    prefix = "p." if len(pages) == 1 else "pp."
    return f"— fonte: {group['label']}, {prefix} {rendered}"


def _format_citation_footer(citations: list[dict], answer: str = "") -> str:
    """One ``— fonte: file.pdf, pp. 9, 22, 79`` line per cited document.

    Grouping by document (rather than one line per retrieved chunk) is what
    turns the footer from a retrieval dump into a citation list: a 34-chunk
    turn over three documents becomes three lines.

    ``answer`` prunes the list further. The backend hands over *every* chunk
    every tool call retrieved, used or not — which is why the footer was
    longest exactly when the answer was emptiest. When the answer
    names at least one of the retrieved documents by filename, only those are
    listed in full; the rest are counted on a trailing line rather than
    dropped, so nothing the answer rests on can go missing. When the answer
    names none of them we have no signal, and every document is listed.
    """
    groups = _group_citations(citations)
    if not groups:
        return ""

    mentioned = [g for g in groups if g["label"] in answer]
    if mentioned:
        listed, unlisted = mentioned, len(groups) - len(mentioned)
        overflow = "— … e altre {n} fonti consultate, non citate nella risposta"
    else:
        listed, unlisted = groups, 0
        overflow = "— … e altri {n} documenti"

    lines = [_format_group(g) for g in listed[:_MAX_FOOTER_DOCUMENTS]]
    hidden = unlisted + max(0, len(listed) - _MAX_FOOTER_DOCUMENTS)
    if hidden > 0:
        lines.append(overflow.format(n=hidden))
    return "\n\n" + "\n".join(lines)


def _parse_sse_stream(lines: Iterable[str]) -> Iterator[tuple[str, str]]:
    """Hand-rolled SSE parser: yields ``(event, data)`` pairs.

    ``lines`` is an iterable of raw text lines with no trailing newline
    (as produced by ``httpx.Response.iter_lines()``). Handles the minimal
    subset of the SSE framing ``sse-starlette`` emits: an ``event:`` line,
    one or more ``data:`` lines (joined with ``\\n`` if repeated), and a
    blank line terminating each event. Comment lines (``:...``) are
    ignored/used as keep-alives.
    """
    event = "message"
    data_lines: list[str] = []
    for line in lines:
        if line == "":
            if data_lines:
                yield event, "\n".join(data_lines)
            event = "message"
            data_lines = []
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:") :].strip())
    if data_lines:
        yield event, "\n".join(data_lines)


def stream_chat_response(
    message: str,
    history: list[dict],
    domain: str,
    api_url: str = CHATBOT_API_URL,
) -> Iterator[str]:
    """POST ``/chat/message`` and yield the growing answer string.

    Factored out of the ``gr.ChatInterface`` fn so it can be driven directly
    in tests (see the throwaway integration check) without going through
    Gradio's wrapper. Never raises - connection issues yield a clean,
    user-facing message instead of propagating a traceback.
    """
    payload = {
        "domain": domain,
        "message": message,
        "history": _to_backend_history(history),
        "stream": True,
    }
    accumulated = ""
    try:
        with (
            httpx.Client(timeout=httpx.Timeout(10.0, read=120.0)) as client,
            client.stream(
                "POST", f"{api_url}/chat/message", json=payload, headers=_headers()
            ) as response,
        ):
            if response.status_code != 200:
                body = response.read()
                logger.warning("Backend returned HTTP %s: %r", response.status_code, body[:500])
                yield _friendly_backend_down_message(api_url)
                return

            for event, data in _parse_sse_stream(response.iter_lines()):
                if not data:
                    continue
                try:
                    parsed = json.loads(data)
                except json.JSONDecodeError:
                    logger.warning("Malformed SSE data for event=%s: %r", event, data)
                    continue

                if event == "token":
                    accumulated += parsed.get("text", "")
                    yield accumulated
                elif event == "done":
                    # `content` is always present in the `done` payload, so
                    # `.get(..., accumulated)` never defaulted - an empty
                    # string overwrote a fully streamed answer with nothing
                    # but the footer. Prefer whatever
                    # actually has text; both should now agree, since the
                    # backend accumulates across tool rounds too.
                    content = parsed.get("content") or accumulated
                    if not content:
                        content = _NO_ANSWER_MESSAGE
                    footer = _format_citation_footer(parsed.get("citations") or [], content)
                    yield content + footer
                    return
                elif event == "error":
                    message = parsed.get("message") or "Si è verificato un errore imprevisto."
                    # Keep what already streamed: an error after tokens (e.g. the provider's
                    # content filter stopping the answer partway) explains the text, it
                    # doesn't replace it.
                    yield f"{accumulated}\n\n⚠️ {message}" if accumulated else message
                    return
                # tool_call / tool_result: no UI surface yet - ignored.
    except httpx.HTTPError as exc:
        logger.warning("Backend connection error talking to %s: %s", api_url, exc)
        if accumulated:
            yield accumulated + "\n\n⚠️ Connessione al backend interrotta."
        else:
            yield _friendly_backend_down_message(api_url)


def make_chat_fn(api_url: str = CHATBOT_API_URL):
    """Build the ``fn`` callable for ``gr.ChatInterface``.

    Signature matches ``additional_inputs=[domain_dropdown]``: Gradio calls
    ``fn(message, history, domain)``.
    """

    def chat_fn(message: str, history: list[dict], domain: str) -> Iterator[str]:
        yield from stream_chat_response(message, history, domain, api_url=api_url)

    return chat_fn


def build_app(api_url: str = CHATBOT_API_URL) -> gr.Blocks:
    """Construct the ``gr.ChatInterface`` Blocks app.

    Safe to call with the backend down: ``fetch_health`` guards its own
    network call and falls back to a static domain list (and an unknown
    version) with a warning.
    """
    domains, version = fetch_health(api_url)
    default_domain = domains[0] if domains else FALLBACK_DOMAINS[0]
    version_label = f"v{version}" if version else "versione sconosciuta"

    domain_dropdown = gr.Dropdown(
        choices=domains,
        value=default_domain,
        label="Dominio",
        info="Seleziona l'ambito su cui interrogare l'assistente SME",
    )

    demo = gr.ChatInterface(
        fn=make_chat_fn(api_url),
        additional_inputs=[domain_dropdown],
        title="SME Chatbot",
        description=f"Client HTTP/SSE del backend SME Chatbot ({api_url}, {version_label}).",
    )
    return demo


if __name__ == "__main__":
    app = build_app()
    app.queue()
    if CHATBOT_SHARE:
        logger.info("CHATBOT_SHARE is set - launching with a public gradio.live share link")
    app.launch(share=CHATBOT_SHARE)
