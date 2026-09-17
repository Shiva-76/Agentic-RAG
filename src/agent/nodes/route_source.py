"""
nodes/route_source.py
---------------------
Node 3: SourceRouter

LLM decides which retrieval source(s) to use:
  "kb"          → private knowledge base (Qdrant)
  "web"         → live web (DuckDuckGo)
  "calculator"  → arithmetic evaluator
  combinations  → ["kb", "web"] etc.
"""

from __future__ import annotations

import logging
from pydantic import BaseModel

from src.agent.groq_client import call_groq_json
from src.agent.system_prompt import ROUTE_SOURCE_PROMPT

logger = logging.getLogger(__name__)


class _RouteResponse(BaseModel):
    sources: list[str]
    reasoning: str


_VALID_SOURCES = {"kb", "web"}


def node_route_source(state: dict) -> dict:
    """
    Input state keys:  rewritten_query
    Output state keys: chosen_sources, node_logs
    """
    query = state["rewritten_query"]
    prompt = f"Query: {query}"

    logger.info("[SourceRouter] query=%r", query)

    result = call_groq_json(prompt, _RouteResponse, system_instruction=ROUTE_SOURCE_PROMPT)

    # Sanitize — only allow known sources
    chosen = [s for s in result.sources if s in _VALID_SOURCES]
    if not chosen:
        chosen = ["kb"]  # safe fallback

    logger.info("[SourceRouter] -> sources=%s (%s)", chosen, result.reasoning)

    return {
        "chosen_sources": chosen,
        "node_logs": [{
            "node": "source_router",
            "chosen_sources": chosen,
            "reasoning": result.reasoning,
        }],
    }
