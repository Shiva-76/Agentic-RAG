"""
nodes/needs_retrieval.py
------------------------
Node 2: NeedsRetrieval (intent classifier)

LLM decides whether external retrieval is needed.
Greetings and trivial chit-chat → direct answer path (no tools).
Everything factual → retrieval path.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pydantic import BaseModel

from src.agent.groq_client import call_groq_json
from src.agent.system_prompt import NEEDS_RETRIEVAL_PROMPT

logger = logging.getLogger(__name__)


class _NeedsRetrievalResponse(BaseModel):
    needs_retrieval: bool
    reason: str


def node_needs_retrieval(state: dict) -> dict:
    """
    Input state keys:  rewritten_query
    Output state keys: needs_retrieval, node_logs
    """
    query = state["rewritten_query"]
    prompt = f"User query: {query}"

    logger.info("[NeedsRetrieval] query=%r", query)

    current_date = datetime.now().strftime("%A, %B %d, %Y")
    sys_prompt = f"{NEEDS_RETRIEVAL_PROMPT}\n\nSystem Note: The current date is {current_date}. If the user asks for the current date, you do NOT need retrieval."

    result = call_groq_json(prompt, _NeedsRetrievalResponse, system_instruction=sys_prompt)

    logger.info("[NeedsRetrieval] -> needs_retrieval=%s (%s)", result.needs_retrieval, result.reason)

    return {
        "needs_retrieval": result.needs_retrieval,
        "node_logs": [{
            "node": "needs_retrieval",
            "needs_retrieval": result.needs_retrieval,
            "reason": result.reason,
        }],
    }
