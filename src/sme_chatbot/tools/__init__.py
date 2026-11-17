"""CDO tools exposed to a chatbot LLM via OpenAI-style function calling.

Exports:
  TOOL_DEFINITIONS  - JSON schemas passed as `tools=` to the chat completion API
  dispatch(name, args) -> str  - run a tool call and return its JSON result
"""

import json
import logging

from .cdo_rag import search_docs
from .cdo_sql import SCHEMA_DESCRIPTION, query_cdo_db

logger = logging.getLogger(__name__)

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "query_cdo_db",
            "description": (
                "Run a read-only SQL SELECT against the CDO database for factual "
                "questions about machines, orders, customers, countries, sectors "
                "and packaging samples (containers/caps). Use this for counts, "
                "filters, aggregations and joins over structured data.\n\n"
                + SCHEMA_DESCRIPTION
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "A single read-only SQLite SELECT statement.",
                    }
                },
                "required": ["sql"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": (
                "Semantic search over technical documents (manuals, specs). "
                "Use this for open-ended or descriptive questions about how "
                "things work, procedures, or specifications that are not in "
                "the structured database. Returns relevant text chunks with "
                "source filename and page number."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language search query.",
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Number of chunks to return (default 5).",
                    },
                },
                "required": ["query"],
            },
        },
    },
]

_HANDLERS = {
    "query_cdo_db": query_cdo_db,
    "search_docs": search_docs,
}


def dispatch(name: str, args: dict) -> str:
    """Execute a tool call by name and return its result as a JSON string."""
    handler = _HANDLERS.get(name)
    if handler is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        result = handler(**args)
    except TypeError as exc:
        result = {"error": f"Bad arguments for {name}: {exc}"}
    except Exception as exc:  # broad on purpose: never crash the chat loop
        logger.exception("Tool %s failed", name)
        result = {"error": f"Tool {name} failed: {exc}"}
    return json.dumps(result, ensure_ascii=False, default=str)
