"""
nodes/synthesize.py
-------------------
Node 5: Synthesize

LLM drafts a cited answer from:
  - The rewritten query
  - The conversation history
  - The retrieval context

Uses Groq openai/gpt-oss-120b — the most capable model on this account,
good at following citation and conflict-disclosure instructions.

NOTE: Gemini thinking is no longer used here to avoid free-tier daily quota
exhaustion. Thought traces are still stored (empty list) for UI compatibility.
"""

from __future__ import annotations

import logging
import json
import time
from datetime import datetime

from langsmith import traceable
from src.agent.groq_client import call_groq_json, groq_client
from src.agent.system_prompt import SYNTHESIZE_PROMPT
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Use the biggest available model for synthesis quality
SYNTHESIS_MODEL = "openai/gpt-oss-120b"


class _SynthesisResponse(BaseModel):
    answer: str
    sources_used: list[str] = []


def node_synthesize(state: dict) -> dict:
    """
    Input state keys:  rewritten_query, retrieval_context, needs_retrieval, conversation_history
    Output state keys: draft_answer, thought_parts, conversation_history, node_logs
    """
    query = state["rewritten_query"]
    context = state.get("retrieval_context", "")
    needs_retrieval = state.get("needs_retrieval", True)
    history = list(state.get("conversation_history", []))

    logger.info("[Synthesize] query=%r needs_retrieval=%s context_len=%d",
                query, needs_retrieval, len(context))

    # Three distinct cases
    if not needs_retrieval:
        # Greeting / general knowledge — respond directly
        prompt = f"User message: {query}\n\nAnswer the user directly and accurately based on your internal knowledge. Do not search."
    elif context and context != "NO_CONTEXT_RETRIEVED":
        # Retrieval succeeded — cite everything from context
        prompt = (
            f"User question: {query}\n\n"
            f"Retrieved context:\n{context}\n\n"
            f"Answer using ONLY the retrieved context above. "
            f"Cite every factual claim with its source."
        )
    else:
        # Retrieval attempted but empty
        prompt = (
            f"User question: {query}\n\n"
            f"No relevant context was retrieved from the knowledge base or web. "
            f"Honestly tell the user you could not find relevant information. "
            f"Do NOT fabricate an answer."
        )

    # Add conversation context if any
    if history:
        history_text = "\n".join(
            f"{turn.get('role','').upper()}: {_extract_text(turn)}"
            for turn in history[-6:]  # last 3 turns
        )
        prompt = f"Conversation so far:\n{history_text}\n\n{prompt}"

    # Override model for synthesis — use the largest available
    from src.agent.groq_client import groq_client
    import time

    @traceable(run_type="llm", name="groq_synthesize")
    def _synthesize(p: str) -> str:
        current_date = datetime.now().strftime("%A, %B %d, %Y")
        if not needs_retrieval:
            base_prompt = "You are a helpful AI assistant. Answer the user directly and accurately based on your internal knowledge. Answer in plain text markdown. Do not output JSON."
        else:
            base_prompt = SYNTHESIZE_PROMPT
            
        system_instruction = f"{base_prompt}\n\nSystem Note: The current date is {current_date}."
        delay = 5.0
        for attempt in range(1, 6):
            try:
                response = groq_client.chat.completions.create(
                    model=SYNTHESIS_MODEL,
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": p},
                    ],
                    temperature=0.3,
                    max_tokens=1024,
                )
                return response.choices[0].message.content.strip()
            except Exception as exc:
                exc_str = str(exc)
                is_retriable = any(k in exc_str for k in ["429", "503", "502", "rate_limit", "overloaded"])
                if is_retriable and attempt < 5:
                    logger.warning("[Synthesize] Retriable error attempt %d/5 — retrying in %.0fs", attempt, delay)
                    time.sleep(delay)
                    delay *= 2
                else:
                    return f"I encountered an error generating a response: {exc_str[:100]}"
        return "Failed to generate response after retries."

    draft_answer = _synthesize(prompt)

    logger.info("[Synthesize] draft length=%d", len(draft_answer))

    # Update conversation history (simple format for next turn)
    new_user = {"role": "user", "parts": [{"text": query}]}
    new_model = {"role": "model", "parts": [{"text": draft_answer}]}

    return {
        "draft_answer": draft_answer,
        "thought_parts": [],   # no thinking traces on Groq — UI handles empty list gracefully
        "conversation_history": [new_user, new_model],
        "node_logs": [{
            "node": "synthesize",
            "model": SYNTHESIS_MODEL,
            "draft_length": len(draft_answer),
            "thought_count": 0,
        }],
    }


def _extract_text(turn: dict) -> str:
    """Pull text from a Gemini-format Content dict."""
    parts = turn.get("parts", [])
    texts = []
    for p in parts:
        if isinstance(p, dict) and "text" in p:
            texts.append(p["text"])
        elif hasattr(p, "text") and p.text:
            texts.append(p.text)
    return " ".join(texts)[:300]
