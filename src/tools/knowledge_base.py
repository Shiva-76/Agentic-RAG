"""
knowledge_base.py
-----------------
Tool: search_knowledge_base

Full Phase 3 implementation:
  1. Embed query with task_type=RETRIEVAL_QUERY (gemini-embedding-001, 768-dim)
  2. Query Qdrant collection "private_knowledge"
  3. Drop chunks below MIN_SIMILARITY floor
  4. Return formatted string or "NO_RELEVANT_RESULTS"
"""

from __future__ import annotations

import logging
import os
import time

from dotenv import load_dotenv, find_dotenv
from google import genai
from google.genai import types as genai_types
from langsmith import traceable
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

load_dotenv(find_dotenv(usecwd=True), override=True)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COLLECTION_NAME = "private_knowledge"
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 768
MIN_SIMILARITY: float = 0.6   # tunable — verify against your real corpus
MAX_RETRY_ATTEMPTS = 5
BASE_DELAY = 1.0
BACKOFF_FACTOR = 2.0

# ---------------------------------------------------------------------------
# Pydantic input schema
# ---------------------------------------------------------------------------

class KBSearchInput(BaseModel):
    query: str = Field(..., description="Natural-language search query")
    top_k: int = Field(5, ge=1, le=20, description="Number of chunks to retrieve")

# ---------------------------------------------------------------------------
# Gemini function declaration (used in tool config for synthesize node)
# ---------------------------------------------------------------------------

KNOWLEDGE_BASE_TOOL_DECL = {
    "name": "search_knowledge_base",
    "description": (
        "Searches the user's private documents stored in the local Qdrant vector DB. "
        "Returns matching text chunks with source filename, page number, and similarity score. "
        "Returns 'NO_RELEVANT_RESULTS' if nothing clears the relevance floor."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Natural-language search query."},
            "top_k": {"type": "integer", "description": "Number of chunks to return (default 5)."},
        },
        "required": ["query"],
    },
}

# ---------------------------------------------------------------------------
# Singleton clients
# ---------------------------------------------------------------------------
_gemini_client: genai.Client | None = None
_qdrant_client: QdrantClient | None = None


def _get_gemini() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set.")
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


def _get_qdrant() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        url = os.getenv("QDRANT_URL", "http://localhost:6333")
        api_key = os.getenv("QDRANT_API_KEY") or None
        _qdrant_client = QdrantClient(url=url, api_key=api_key)
    return _qdrant_client

# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

@traceable(run_type="embedding", name="gemini_embed")
def _embed_with_retry(text: str) -> list[float]:
    delay = BASE_DELAY
    for attempt in range(1, MAX_RETRY_ATTEMPTS + 1):
        try:
            result = _get_gemini().models.embed_content(
                model=EMBEDDING_MODEL,
                contents=text,
                config=genai_types.EmbedContentConfig(
                    task_type="RETRIEVAL_QUERY",
                    output_dimensionality=EMBEDDING_DIM,
                ),
            )
            return result.embeddings[0].values
        except Exception as exc:
            is_rate_limit = (
                "429" in str(exc)
                or "RESOURCE_EXHAUSTED" in str(exc)
                or getattr(exc, "status_code", None) == 429
            )
            if is_rate_limit and attempt < MAX_RETRY_ATTEMPTS:
                logger.warning("Rate limit hit embedding query. Retrying in %.1fs", delay)
                time.sleep(delay)
                delay *= BACKOFF_FACTOR
            else:
                raise
    raise RuntimeError("Exhausted embedding retry attempts")

# ---------------------------------------------------------------------------
# Tool function
# ---------------------------------------------------------------------------

def search_knowledge_base(query: str, top_k: int = 5) -> str:
    """
    Search the private knowledge base in Qdrant.

    Returns a formatted string of relevant chunks, or the literal string
    "NO_RELEVANT_RESULTS" if nothing clears the MIN_SIMILARITY floor.
    """
    logger.info("KB search: %r (top_k=%d)", query, top_k)

    try:
        query_vector = _embed_with_retry(query)
    except Exception as exc:
        logger.error("Embedding failed: %s", exc)
        return "NO_RELEVANT_RESULTS"

    try:
        qdrant = _get_qdrant()
        results = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )
        hits = results.points
    except Exception as exc:
        logger.error("Qdrant query failed: %s", exc)
        return "NO_RELEVANT_RESULTS"

    # Apply similarity floor
    passing = [h for h in hits if h.score >= MIN_SIMILARITY]
    logger.info(
        "KB results: %d returned, %d above MIN_SIMILARITY=%.2f",
        len(hits), len(passing), MIN_SIMILARITY,
    )

    if not passing:
        return "NO_RELEVANT_RESULTS"

    lines: list[str] = []
    for rank, hit in enumerate(passing, 1):
        payload = hit.payload or {}
        lines.append(
            f"[KB Result {rank}]\n"
            f"Score: {hit.score:.4f}\n"
            f"Source: {payload.get('source_filename', 'unknown')}, "
            f"p.{payload.get('page_number', '?')}\n"
            f"Text: {payload.get('text', '').strip()}\n"
        )

    return "\n---\n".join(lines)
