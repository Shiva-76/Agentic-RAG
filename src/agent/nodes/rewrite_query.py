"""
nodes/rewrite_query.py
----------------------
Node 1: QueryRewriter

LLM rewrites the raw user query into a more precise version before retrieval.
On retry turns, incorporates the failure_context hint from the RelevanceGuard.
"""

from __future__ import annotations

import logging
from pydantic import BaseModel

from src.agent.groq_client import call_groq_json
from src.agent.system_prompt import REWRITE_PROMPT

logger = logging.getLogger(__name__)


class _RewriteResponse(BaseModel):
    rewritten_query: str
    reasoning: str


def node_rewrite_query(state: dict) -> dict:
    """
    Input state keys:  user_query, retry_count (optional), failure_context (optional)
    Output state keys: rewritten_query, node_logs
    """
    raw_query = state["user_query"]
    retry_count = state.get("retry_count", 0)
    failure_context = state.get("failure_context", "")

    # Build the prompt — on retries, include what went wrong
    if retry_count > 0 and failure_context:
        prompt = (
            f"Original query: {raw_query}\n"
            f"Previous rewritten query: {state.get('rewritten_query', raw_query)}\n"
            f"Relevance guard feedback: {failure_context}\n\n"
            f"Rewrite the query taking the feedback into account."
        )
    else:
        prompt = f"Query to rewrite: {raw_query}"

    logger.info("[QueryRewriter] raw=%r retry=%d", raw_query, retry_count)

    result = call_groq_json(prompt, _RewriteResponse, system_instruction=REWRITE_PROMPT)

    logger.info("[QueryRewriter] -> %r", result.rewritten_query)

    return {
        "rewritten_query": result.rewritten_query,
        "node_logs": [{
            "node": "query_rewriter",
            "input": raw_query,
            "output": result.rewritten_query,
            "reasoning": result.reasoning,
            "retry": retry_count,
        }],
    }
