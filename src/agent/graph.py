"""
graph.py
--------
LangGraph state machine — upgraded 6-node architecture with guardrail loop.

Topology:
    START
      → rewrite_query        (always first — clarifies the raw query)
      → needs_retrieval?
            No → synthesize  (direct answer — greetings, chit-chat)
            Yes → route_source → retrieve → synthesize
      → relevance_guard      (guardrail)
            pass  → END
            fail  → rewrite_query (retry, max 3x) → ... → END
"""

from __future__ import annotations

import operator
from typing import TypedDict, Annotated

from langgraph.graph import StateGraph, END

from src.agent.nodes import (
    node_rewrite_query,
    node_needs_retrieval,
    node_route_source,
    node_retrieve,
    node_synthesize,
    node_relevance_guard,
)

# ---------------------------------------------------------------------------
# Shared state schema
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    # ── Input ───────────────────────────────────────────────────────────────
    user_query: str                                        # raw user input

    # ── Conversation history (Gemini Content format) ─────────────────────
    # Annotated with operator.add so each node can append without overwriting
    conversation_history: Annotated[list[dict], operator.add]

    # ── Node outputs ────────────────────────────────────────────────────────
    rewritten_query: str                                   # QueryRewriter output
    needs_retrieval: bool                                  # NeedsRetrieval output
    chosen_sources: list[str]                              # SourceRouter output
    retrieval_context: str                                 # Retrieve output
    draft_answer: str                                      # Synthesize output

    # ── Guardrail ───────────────────────────────────────────────────────────
    relevance_pass: bool                                   # RelevanceGuard verdict
    retry_count: int                                       # loop counter (cap=3)
    failure_context: str                                   # hint fed back to rewriter

    # ── Final ────────────────────────────────────────────────────────────────
    final_answer: str                                      # displayed to user

    # ── UI / observability ──────────────────────────────────────────────────
    thought_parts: Annotated[list[str], operator.add]      # Gemini reasoning traces
    node_logs: Annotated[list[dict], operator.add]         # per-node metadata

# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def route_after_needs_retrieval(state: AgentState) -> str:
    """After NeedsRetrieval: skip tools for greetings, else go to SourceRouter."""
    return "route_source" if state.get("needs_retrieval", True) else "synthesize"


def route_after_guard(state: AgentState) -> str:
    """After RelevanceGuard: end if pass (or max retries hit), else retry."""
    if state.get("relevance_pass", True):
        return END
    return "rewrite_query"  # retry loop

# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    builder = StateGraph(AgentState)

    # Register nodes
    builder.add_node("rewrite_query",    node_rewrite_query)
    builder.add_node("needs_retrieval",  node_needs_retrieval)
    builder.add_node("route_source",     node_route_source)
    builder.add_node("retrieve",         node_retrieve)
    builder.add_node("synthesize",       node_synthesize)
    builder.add_node("relevance_guard",  node_relevance_guard)

    # Entry point
    builder.set_entry_point("rewrite_query")

    # Linear edges
    builder.add_edge("rewrite_query",   "needs_retrieval")
    builder.add_edge("route_source",    "retrieve")
    builder.add_edge("retrieve",        "synthesize")
    builder.add_edge("synthesize",      "relevance_guard")

    # Conditional: needs_retrieval? → route_source OR synthesize (direct)
    builder.add_conditional_edges(
        "needs_retrieval",
        route_after_needs_retrieval,
        {"route_source": "route_source", "synthesize": "synthesize"},
    )

    # Conditional: relevance_guard → END (pass) OR rewrite_query (retry)
    builder.add_conditional_edges(
        "relevance_guard",
        route_after_guard,
        {END: END, "rewrite_query": "rewrite_query"},
    )

    return builder.compile()


# Module-level compiled graph — import this in app.py
graph = build_graph()


# ---------------------------------------------------------------------------
# Helper: run one turn
# ---------------------------------------------------------------------------

def run_turn(
    user_query: str,
    conversation_history: list[dict] | None = None,
) -> AgentState:
    """
    Run a single user turn through the full graph.
    Returns the final AgentState.
    """
    initial_state: AgentState = {
        "user_query": user_query,
        "conversation_history": conversation_history or [],
        "rewritten_query": "",
        "needs_retrieval": True,
        "chosen_sources": [],
        "retrieval_context": "",
        "draft_answer": "",
        "relevance_pass": False,
        "retry_count": 0,
        "failure_context": "",
        "final_answer": "",
        "thought_parts": [],
        "node_logs": [],
    }
    return graph.invoke(initial_state)
