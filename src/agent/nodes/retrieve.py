"""
nodes/retrieve.py
-----------------
Node 4: Retrieve

Executes the tool(s) chosen by SourceRouter and aggregates results
into a single retrieval_context string for the Synthesize node.

No LLM call here — pure tool execution.
"""

from __future__ import annotations

import logging

from src.tools.knowledge_base import search_knowledge_base
from src.tools.web_search import search_web

logger = logging.getLogger(__name__)


def node_retrieve(state: dict) -> dict:
    """
    Input state keys:  rewritten_query, chosen_sources
    Output state keys: retrieval_context, node_logs
    """
    query = state["rewritten_query"]
    sources = state.get("chosen_sources", ["kb"])

    logger.info("[Retrieve] query=%r sources=%s", query, sources)

    context_parts: list[str] = []
    tool_log: list[dict] = []

    for source in sources:
        if source == "kb":
            result = search_knowledge_base(query)
            label = "Knowledge Base"
        elif source == "web":
            result = search_web(query)
            label = "Web Search"
        else:
            logger.warning("[Retrieve] Unknown source %r — skipping", source)
            continue

        logger.info("[Retrieve] %s result length: %d chars", label, len(result))
        context_parts.append(f"=== {label} Results ===\n{result}")
        tool_log.append({"source": source, "result_length": len(result), "result_preview": result[:200]})

    retrieval_context = "\n\n".join(context_parts) if context_parts else "NO_CONTEXT_RETRIEVED"

    return {
        "retrieval_context": retrieval_context,
        "node_logs": [{
            "node": "retrieve",
            "sources_used": sources,
            "tools": tool_log,
        }],
    }
