"""Domain registry: system prompt + tool definitions + dispatcher per chat domain.

``stream_chat`` (see ``core/chat.py``) is domain-agnostic; it just needs a
``DomainConfig`` to know which system prompt to prepend, which tool schemas to
advertise to the LLM, and how to execute a tool call by name.

System prompts below are copied verbatim from the legacy launchers
(``chatbot_cdo.py``, ``chatbot_manuals.py``) per the migration plan §5 WP2.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from sme_chatbot.tools import TOOL_DEFINITIONS as _CDO_TOOL_DEFINITIONS
from sme_chatbot.tools import dispatch as _cdo_dispatch
from sme_chatbot.tools.docsvc_rag import TOOL_DEFINITIONS as _DOCUMENTS_TOOL_DEFINITIONS
from sme_chatbot.tools.docsvc_rag import dispatch as _documents_dispatch
from sme_chatbot.tools.docsvc_rag import make_dispatch as make_documents_dispatch
from sme_chatbot.tools.docsvc_rag import resolve_serial
from sme_chatbot.tools.manuals_rag import TOOL_DEFINITIONS as _MANUALS_TOOL_DEFINITIONS
from sme_chatbot.tools.manuals_rag import dispatch as _manuals_dispatch

if TYPE_CHECKING:
    from sme_chatbot.core.events import RequestIdentity

# `make_documents_dispatch` is imported above purely to be re-exported (see the
# note under DOMAINS); listing it here is what tells ruff that, and states this
# module's public surface rather than leaving it implicit.
__all__ = [
    "CDO_SYSTEM_PROMPT",
    "DOCUMENTS_SYSTEM_PROMPT",
    "DOMAINS",
    "MANUALS_SYSTEM_PROMPT",
    "DomainConfig",
    "documents_domain_for",
    "make_documents_dispatch",
]


@dataclass
class DomainConfig:
    """Everything ``stream_chat`` needs to run one domain's conversation."""

    key: str
    title: str
    system_prompt: str
    tool_definitions: list[dict]
    # (name, arguments) -> JSON string result. Signature is deliberately
    # identity-agnostic: the LLM only ever sees `name`/`arguments`. A domain
    # whose tools need `RequestIdentity` (e.g. WP7's `search_documents`, which
    # must apply an access-control `where` filter) binds identity via
    # `functools.partial`/closure when the DomainConfig entry is constructed
    # per-request — see plan §5 WP7 item 2.
    dispatch: Callable[[str, dict], str]
    # (tool name, arguments) of the search `stream_chat` may run before the first LLM call,
    # when the backend enables it (`CHAT_PRE_SEARCH`); None = no pre-search.
    pre_search: tuple[str, dict] | None = None


CDO_SYSTEM_PROMPT = (
    "You are an assistant for engineering teams analysing CDO (customer "
    "data object) records. You have two tools:\n"
    "- `query_cdo_db`: read-only SQL over structured data (machines, orders, "
    "customers, countries, sectors, container and cap samples). Use it for "
    "counts, filters, aggregations and any factual lookup.\n"
    "- `search_docs`: semantic search over technical documents (manuals, "
    "specs, knowledge base). Use it for descriptive or how-to questions.\n"
    "Prefer the database for precise figures. When you cite documents, mention "
    "the source filename and page. Answer in the user's language."
)

MANUALS_SYSTEM_PROMPT = (
    "You are an assistant for the manufacturer's machine manuals. You have one tool:\n"
    "- `search_manuals`: semantic search over the machine manuals and "
    "spare-parts documents. Use it to answer how-to, procedure, component and "
    "specification questions.\n"
    "Always cite the source filename and page for every fact you state, but "
    "only the specific pages you actually relied on to build the answer — "
    "never the full list of retrieved chunks. If the manuals don't cover the "
    "question, say so. Answer in the user's language."
)

DOCUMENTS_SYSTEM_PROMPT = (
    "You are an assistant over the manufacturer's document library (manuals, CDO "
    "reports, certifications, and other documents) kept in sync from the "
    "DocumentService. You have two tools:\n"
    "- `search_documents`: keyword + semantic search over that document "
    "library. Use it to answer how-to, procedure, specification or "
    "document-lookup questions. Results are already restricted to what the "
    "current user is permitted to see.\n"
    "- `read_pages`: read whole pages of one document. Use it to follow the "
    "document's own cross-references (e.g. 'vedere il sottocapitolo 9.5 a "
    "pagina 37' -> `printed: true`, pages [37]), to see a complete table when "
    "a result shows only part of it, and to read the pages a result points "
    "to. Specifications, torque values, maintenance intervals and "
    "troubleshooting remedies are usually in tables: read the table rather "
    "than answering from a paragraph that merely mentions it.\n"
    "Machine scoping: manuals, torque values, and maintenance intervals "
    "differ between machines, so once you know the user's machine serial "
    "number (matricola) — including a bare number the user states directly "
    "in their own message, e.g. 'la chiuditrice 90001' or 'la 90001' means "
    "matricola 90001; treat that as known and use it, do not ask the user "
    "to confirm it — or from earlier in the conversation, pass it as "
    "`serial` on every subsequent call. If a question is machine-specific "
    "(spec, torque, maintenance interval, troubleshooting, part number) and "
    "no number identifying the machine appears anywhere in the "
    "conversation, ask for it before searching — do not silently answer "
    "from another machine's manual, since that is a safety-relevant error, "
    "not just a relevance one. End such a request with a direct question.\n"
    "Language scoping: the same manual often exists translated into "
    "multiple languages as near-duplicate documents. Pass `language` "
    "matching the language of the user's question (e.g. 'IT' for Italian) "
    "so you don't end up citing a translation the user cannot read; if that "
    "returns nothing, retry without `language`, and write the query in the "
    "language of the documents that exist (e.g. Italian terms for an "
    "Italian-only manual) — then answer in the user's language.\n"
    "Search persistently: if the first results don't contain the answer, "
    "search again with the terms the manual itself would use (section or "
    "table titles, component names, units), or read the referenced pages. "
    "Never end by offering to search or to read a page ('Vuoi che lo "
    "faccia?') — if a tool call could answer it, make the call now. Before "
    "saying a machine's documents do not give a value, read its data sheets "
    "with `read_pages`: technical data and identification (manufacturer, "
    "model, production, number of heads, direction of rotation) are usually "
    "in the first chapters ('Dati tecnici', 'Caratteristiche della "
    "macchina'). Only after that, say plainly that the documents do not "
    "contain it.\n"
    "Answer carefully:\n"
    "- Check premises: if the question states something the documents "
    "contradict (e.g. a direction of rotation, a component model), say so "
    "first and give the documented fact.\n"
    "- Check the machine's components: if the question names a component or "
    "model the machine does not have, say that it does not apply and name "
    "the one it has, rather than answering for the other.\n"
    "- When several table rows or tables apply (materials, strength classes, "
    "work shifts, service intervals), give each value with its qualifier, or "
    "ask which one applies. Include the table's footnotes and notes.\n"
    "- Prefer completeness to brevity: give every value the documents give "
    "for the question (each measurement point, each check, each component) "
    "with what it applies to, and the full details of what you cite (e.g. "
    "the manufacturer's name and address, the directives and standards a "
    "declaration lists).\n"
    "- If a value is not in the table (e.g. a size beyond its last row), say "
    "it is not given; do not extrapolate or substitute a neighbouring row.\n"
    "- If a total is not stated, list the documented parts and say that any "
    "sum is derived by you.\n"
    "- For safety instructions, procedures and troubleshooting, keep the "
    "document's own wording and give every listed cause with its remedy.\n"
    "Cite your sources inline as `filename, p. N`, using the `filename` "
    "field of the result verbatim — never a bracketed number like [1], and "
    "never the `repository`. Cite only the specific pages you actually used "
    "to build the answer, not the full list of retrieved chunks — citing "
    "twenty pages is as unusable to the reader as citing none. Answer only "
    "from the retrieved content — if the documents don't cover the "
    "question, say so rather than guessing. Answer in the user's language."
)


def documents_domain_for(
    identity: RequestIdentity,
    message: str,
    history: list[dict] | None = None,
) -> DomainConfig:
    """Per-request ``documents`` domain: identity-bound, machine-scoped.

    Binds ``identity`` into the tool dispatcher (the access gate), and — plan
    WP2a — resolves the matricola the conversation names in code rather than
    trusting the prompt: it becomes the default ``serial`` of every search and
    is stated in the system prompt, so the model neither re-asks for it nor
    searches other machines' manuals.
    """
    serial = resolve_serial(message, history)
    prompt = DOCUMENTS_SYSTEM_PROMPT
    if serial:
        prompt += (
            f"\n\nMachine in scope for this conversation: matricola {serial}, "
            "as stated by the user. Searches are already filtered to it; do "
            "not ask the user for the serial number."
        )
    else:
        # Local eval (2026-09-24): with no matricola in the question, the
        # general rule above was not enough — the model answered Q34 ("coppia
        # di una vite M12?") with one machine's value, unqualified. Stating the
        # absence per request, like the serial above, is what the model heeds.
        prompt += (
            "\n\nNo machine serial number (matricola) has been identified in "
            "this conversation. If the question depends on the machine — a "
            "specification, torque value, maintenance interval, "
            "troubleshooting step or part — do not search yet and do not "
            "give a value from any manual: ask the user which machine "
            "(matricola) it is, ending your reply with that question."
        )
    return replace(
        DOMAINS["documents"],
        system_prompt=prompt,
        dispatch=make_documents_dispatch(identity, default_serial=serial),
        # Only with a machine in scope: without one the prompt says not to search yet. The
        # dispatcher adds the serial, as it does for the model's own searches.
        pre_search=("search_documents", {"query": message}) if serial else None,
    )


DOMAINS: dict[str, DomainConfig] = {
    "sample_cdo": DomainConfig(
        key="sample_cdo",
        title="Sample CDO Assistant",
        system_prompt=CDO_SYSTEM_PROMPT,
        tool_definitions=_CDO_TOOL_DEFINITIONS,
        dispatch=_cdo_dispatch,
    ),
    "sample_manuals": DomainConfig(
        key="sample_manuals",
        title="Sample Manuals Assistant",
        system_prompt=MANUALS_SYSTEM_PROMPT,
        tool_definitions=_MANUALS_TOOL_DEFINITIONS,
        dispatch=_manuals_dispatch,
    ),
    # DocumentService-backed RAG domain (tools in tools/docsvc_rag.py +
    # tools/access_filter.py) — see plan §5 WP7. `dispatch` below is the
    # default unrestricted dispatcher (identity=None); the API layer builds a
    # per-request DomainConfig with `dispatch=make_documents_dispatch(identity)`
    # (re-exported here) so tool calls are scoped to the caller's
    # RequestIdentity without the LLM ever seeing or controlling it.
    "documents": DomainConfig(
        key="documents",
        title="Document Library Assistant",
        system_prompt=DOCUMENTS_SYSTEM_PROMPT,
        tool_definitions=_DOCUMENTS_TOOL_DEFINITIONS,
        dispatch=_documents_dispatch,
    ),
}

# `make_documents_dispatch` (imported above from tools.docsvc_rag) is
# re-exported from this module for the API layer (WP3): build an
# identity-bound dispatcher for the "documents" domain's DomainConfig on a
# per-request basis, e.g.
#   domain = replace(DOMAINS["documents"], dispatch=make_documents_dispatch(identity))
