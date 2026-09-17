"""
schemas.py
----------
Pydantic models for FastAPI request/response contracts.
"""
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User's message")
    session_id: str = Field(default="default", description="Session identifier for conversation history")


class HealthResponse(BaseModel):
    status: str
    project: str
    version: str


class NodeEvent(BaseModel):
    event: str          # "node_update" | "final_answer" | "error"
    node: str | None = None
    data: dict
