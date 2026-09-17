"""
gemini_client.py
----------------
Thin wrapper around google-genai Client with:
  - Singleton instantiation (one client per process)
  - Exponential-backoff retry on HTTP 429 (rate-limit) responses
  - Centralised model constants
  - call_llm_json()  — structured JSON output for classifier/router/guard nodes
  - call_llm_with_thinking() — for the synthesize node with thought traces
"""

import time
import json
import logging
import os
from pathlib import Path
from functools import wraps
from typing import Callable, TypeVar, Any, Type

from google import genai
from google.genai import types as genai_types
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel

load_dotenv(find_dotenv(usecwd=True), override=True)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model constants — change these in one place if/when you upgrade
# ---------------------------------------------------------------------------
GEMINI_FLASH_MODEL = "gemini-3.6-flash"          # reasoning + tool calling
GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"  # embeddings
EMBEDDING_DIM = 768                              # MRL-reduced output dims
MAX_RETRIES = 3                                  # guardrail loop cap

# ---------------------------------------------------------------------------
# Singleton client
# ---------------------------------------------------------------------------
_api_key = os.getenv("GEMINI_API_KEY")
if not _api_key:
    raise ValueError(
        "GEMINI_API_KEY not set. Copy .env.template to .env and fill in your key."
    )
client = genai.Client(api_key=_api_key)

# ---------------------------------------------------------------------------
# Retry decorator — exponential backoff on 429
# ---------------------------------------------------------------------------
_RT = TypeVar("_RT")

def with_retry(
    max_attempts: int = 6,
    base_delay: float = 15.0,
    backoff_factor: float = 2.0,
) -> Callable:
    """
    Decorator that retries on HTTP 429 (rate limit) and 503 (overload).

    Free-tier gemini-3.6-flash = 5 RPM. Each pipeline turn uses 3-5 calls,
    so retry windows can be 10–60 s. Base delay of 15 s is intentional.
    Delays: 15s → 30s → 60s → 120s → 240s (default, 6 attempts).
    """
    def decorator(fn: Callable[..., _RT]) -> Callable[..., _RT]:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> _RT:
            delay = base_delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except Exception as exc:
                    exc_str = str(exc)
                    is_retriable = (
                        "429" in exc_str
                        or "RESOURCE_EXHAUSTED" in exc_str
                        or "503" in exc_str
                        or "UNAVAILABLE" in exc_str
                        or getattr(exc, "status_code", None) in (429, 503)
                    )
                    if is_retriable and attempt < max_attempts:
                        logger.warning(
                            "Retriable error (attempt %d/%d) — retrying in %.0fs: %s",
                            attempt, max_attempts, delay, exc_str[:120],
                        )
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        raise
            raise RuntimeError("Exhausted retry attempts")
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# call_llm_json — structured output for classifier/router/guard nodes
# ---------------------------------------------------------------------------

def call_llm_json(
    prompt: str,
    response_schema: Type[BaseModel],
    system_instruction: str = "",
) -> BaseModel:
    """
    Call Gemini with response_mime_type='application/json' and a Pydantic
    response_schema. Returns a validated Pydantic model instance.

    Wrapped with retry logic for 429s.
    """
    @with_retry()
    def _call() -> BaseModel:
        config_kwargs: dict[str, Any] = {
            "response_mime_type": "application/json",
            "response_schema": response_schema,
        }
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        response = client.models.generate_content(
            model=GEMINI_FLASH_MODEL,
            contents=prompt,
            config=genai_types.GenerateContentConfig(**config_kwargs),
        )
        return response_schema.model_validate_json(response.text)

    return _call()


# ---------------------------------------------------------------------------
# call_llm_with_thinking — for synthesize node with thought traces
# ---------------------------------------------------------------------------

def call_llm_with_thinking(
    contents: list[dict],
    system_instruction: str,
    tools: list | None = None,
) -> genai_types.GenerateContentResponse:
    """
    Call Gemini with thinking enabled. Returns the raw response so the caller
    can iterate over parts (thought vs text vs function_call).

    Wrapped with retry logic for 429s.
    """
    @with_retry()
    def _call() -> genai_types.GenerateContentResponse:
        config_kwargs: dict[str, Any] = {
            "system_instruction": system_instruction,
            "thinking_config": genai_types.ThinkingConfig(include_thoughts=True),
        }
        if tools:
            config_kwargs["tools"] = tools

        return client.models.generate_content(
            model=GEMINI_FLASH_MODEL,
            contents=contents,
            config=genai_types.GenerateContentConfig(**config_kwargs),
        )

    return _call()
