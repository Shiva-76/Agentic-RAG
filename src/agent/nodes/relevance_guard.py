"""
nodes/relevance_guard.py
------------------------
Node 6: RelevanceGuard  ← THE GUARDRAIL

LLM evaluates whether the draft answer:
  1. Actually addresses the user's question
  2. Is grounded in the retrieved context (no hallucinations)
  3. Contains citations for every factual claim
  4. Handles conflicts and gaps honestly

If pass=False and retry_count < MAX_RETRIES, the graph loops back
to QueryRewriter with an improvement hint in failure_context.
"""

from __future__ import annotations

import logging
from pydantic import BaseModel

from src.agent.groq_client import call_groq_json
from src.agent.gemini_client import MAX_RETRIES
from src.agent.system_prompt import RELEVANCE_GUARD_PROMPT

logger = logging.getLogger(__name__)


class _GuardResponse(BaseModel):
    pass_: bool = False  # "pass" is a Python keyword — groq_client remaps it
    reason: str = ""
    improvement_hint: str = ""


def node_relevance_guard(state: dict) -> dict:
    """
    Input state keys:  rewritten_query, retrieval_context, draft_answer, retry_count
    Output state keys: relevance_pass, failure_context, retry_count, final_answer, node_logs
    """
    query = state["rewritten_query"]
    context = state.get("retrieval_context", "")
    draft = state.get("draft_answer", "")
    retry_count = state.get("retry_count", 0)
    needs_retrieval = state.get("needs_retrieval", True)

    logger.info("[RelevanceGuard] evaluating draft (retry=%d)", retry_count)

    # If no retrieval was needed, there is no context to ground against.
    # The synthesize node answered from internal knowledge. Pass automatically.
    if not needs_retrieval:
        logger.info("[RelevanceGuard] needs_retrieval=False — auto-passing")
        return {
            "relevance_pass": True,
            "final_answer": draft,
            "failure_context": "",
            "node_logs": [{
                "node": "relevance_guard",
                "pass": True,
                "reason": "Auto-passed general knowledge query (needs_retrieval=False)",
                "hint": "",
                "retry_count": retry_count,
            }],
        }

    prompt = (
        f"Query: {query}\n\n"
        f"Retrieved context:\n{context[:2000]}\n\n"
        f"Draft answer:\n{draft}\n\n"
        f"Evaluate relevance and groundedness."
    )

    try:
        result = call_groq_json(prompt, _GuardResponse, system_instruction=RELEVANCE_GUARD_PROMPT)
        passed = result.pass_
        reason = result.reason
        hint = result.improvement_hint
    except Exception as exc:
        logger.error("[RelevanceGuard] evaluation failed: %s — defaulting to pass", exc)
        passed = True
        reason = "Guard evaluation failed — defaulting to pass"
        hint = ""

    logger.info("[RelevanceGuard] pass=%s reason=%r", passed, reason)

    updates: dict = {
        "relevance_pass": passed,
        "node_logs": [{
            "node": "relevance_guard",
            "pass": passed,
            "reason": reason,
            "hint": hint,
            "retry_count": retry_count,
        }],
    }

    if passed:
        # Promote draft to final answer
        updates["final_answer"] = draft
        updates["failure_context"] = ""
    else:
        new_retry = retry_count + 1
        if new_retry >= MAX_RETRIES:
            # Graceful failure — don't loop indefinitely
            logger.warning("[RelevanceGuard] Max retries (%d) reached — surfacing graceful failure", MAX_RETRIES)
            updates["final_answer"] = (
                "I was unable to produce a fully relevant and grounded answer after "
                f"{MAX_RETRIES} attempts. Here is my best effort:\n\n{draft}\n\n"
                f"_Guardrail note: {reason}_"
            )
            updates["relevance_pass"] = True  # force exit from loop
            updates["retry_count"] = new_retry
        else:
            updates["retry_count"] = new_retry
            updates["failure_context"] = hint or reason

    return updates
