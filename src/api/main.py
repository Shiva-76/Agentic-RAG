"""
main.py
-------
FastAPI application for Verity — Agentic RAG System.

Endpoints:
  GET  /health          → liveness check
  POST /chat            → run one agent turn, stream node events via SSE
  GET  /history/{id}   → return conversation history for a session
  DELETE /history/{id} → clear conversation history for a session
  GET  /docs            → auto-generated OpenAPI docs (FastAPI default)
"""

from __future__ import annotations

import json
import sys
import os
import logging
from pathlib import Path

# Make src importable when launched as `uvicorn src.api.main:app`
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True), override=True)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from src.api.schemas import ChatRequest, HealthResponse
from src.api.session import get_history, set_history, clear_history
from src.agent.graph import run_turn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Verity — Agentic RAG API",
    description=(
        "6-node self-correcting RAG agent with query rewriting, "
        "source routing, and a relevance guardrail loop. "
        "Powered by LangGraph · Groq · Gemini · Qdrant."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# CORS — allow React dev server
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite dev server
        "http://localhost:3000",   # CRA fallback
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    return HealthResponse(status="ok", project="verity-rag", version="1.0.0")


@app.post("/chat", tags=["Agent"])
def chat(request: ChatRequest):
    """
    Run one agent turn. Streams Server-Sent Events (SSE):

      event: node_update  → emitted for each node after the run completes
      event: final_answer → the agent's final answer
      event: error        → if the agent raised an exception

    SSE format:  data: <json>\\n\\n
    """
    def event_stream():
        try:
            history = get_history(request.session_id)

            # Run the full LangGraph pipeline (synchronous)
            result = run_turn(
                user_query=request.query,
                conversation_history=history,
            )

            # Persist updated conversation history
            set_history(request.session_id, result.get("conversation_history", []))

            # Replay node logs as SSE events
            for log in result.get("node_logs", []):
                payload = json.dumps({"event": "node_update", "data": log})
                yield f"data: {payload}\n\n"

            # Emit thought traces if any
            thoughts = result.get("thought_parts", [])
            if thoughts:
                payload = json.dumps({
                    "event": "thought_traces",
                    "data": {"thoughts": thoughts}
                })
                yield f"data: {payload}\n\n"

            # Emit final answer
            payload = json.dumps({
                "event": "final_answer",
                "data": {
                    "answer": result.get("final_answer", ""),
                    "rewritten_query": result.get("rewritten_query", ""),
                    "retry_count": result.get("retry_count", 0),
                    "sources_used": result.get("chosen_sources", []),
                }
            })
            yield f"data: {payload}\n\n"

            # Signal stream end
            yield "data: {\"event\": \"done\"}\n\n"

        except Exception as exc:
            logger.error("Agent error: %s", exc, exc_info=True)
            payload = json.dumps({
                "event": "error",
                "data": {"message": str(exc)}
            })
            yield f"data: {payload}\n\n"
            yield "data: {\"event\": \"done\"}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering
        },
    )


@app.get("/history/{session_id}", tags=["Session"])
def get_session_history(session_id: str):
    """Return the conversation history for a session."""
    return {"session_id": session_id, "history": get_history(session_id)}


@app.delete("/history/{session_id}", tags=["Session"])
def delete_session_history(session_id: str):
    """Clear conversation history for a session (new chat)."""
    clear_history(session_id)
    return {"session_id": session_id, "status": "cleared"}
