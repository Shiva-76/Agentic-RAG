"""
web_search.py
-------------
Tool: search_web

Uses DuckDuckGo (ddgs package) — no API key required.
Wrapped in exponential-backoff retry since DDG has no official rate-limit contract.
"""

from __future__ import annotations

import logging
import time

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_RESULTS = 3
MAX_RETRY_ATTEMPTS = 4
BASE_DELAY = 2.0
BACKOFF_FACTOR = 2.0

# ---------------------------------------------------------------------------
# Pydantic input schema
# ---------------------------------------------------------------------------

class WebSearchInput(BaseModel):
    query: str = Field(..., description="Search query for the live web")

# ---------------------------------------------------------------------------
# Gemini function declaration
# ---------------------------------------------------------------------------

WEB_SEARCH_TOOL_DECL = {
    "name": "search_web",
    "description": (
        "Searches the live web using DuckDuckGo. "
        "Returns up to 3 results, each with a title, URL, and snippet. "
        "Use for current events, prices, or topics not covered by the knowledge base."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query string."},
        },
        "required": ["query"],
    },
}

# ---------------------------------------------------------------------------
# Tool function
# ---------------------------------------------------------------------------

def search_web(query: str) -> str:
    """
    Search the live web with DuckDuckGo.

    Returns a formatted string of up to 3 results (title, url, snippet),
    or an error string if all retries fail.
    """
    logger.info("Web search: %r", query)

    delay = BASE_DELAY
    last_exc: Exception | None = None

    for attempt in range(1, MAX_RETRY_ATTEMPTS + 1):
        try:
            from ddgs import DDGS
            results = DDGS().text(query, max_results=MAX_RESULTS)

            if not results:
                return "NO_WEB_RESULTS"

            lines: list[str] = []
            for rank, r in enumerate(results[:MAX_RESULTS], 1):
                lines.append(
                    f"[Web Result {rank}]\n"
                    f"Title: {r.get('title', 'N/A')}\n"
                    f"URL: {r.get('href', 'N/A')}\n"
                    f"Snippet: {r.get('body', 'N/A')}\n"
                )
            return "\n---\n".join(lines)

        except Exception as exc:
            last_exc = exc
            logger.warning(
                "DuckDuckGo search failed (attempt %d/%d): %s — retrying in %.1fs",
                attempt, MAX_RETRY_ATTEMPTS, exc, delay,
            )
            time.sleep(delay)
            delay *= BACKOFF_FACTOR

    logger.error("Web search exhausted retries. Last error: %s", last_exc)
    return f"WEB_SEARCH_ERROR: {last_exc}"
