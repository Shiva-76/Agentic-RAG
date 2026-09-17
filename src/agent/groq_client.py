"""
groq_client.py
--------------
Groq API client for the 4 fast structured-output nodes:
  - QueryRewriter
  - NeedsRetrieval
  - SourceRouter
  - RelevanceGuard

Uses Groq's OpenAI-compatible API with JSON mode.
Model: llama-3.3-70b-versatile (reliable JSON, 30 RPM free tier).

Rate limits (free tier as of 2025):
  - 30 RPM / 14,400 RPD for llama-3.3-70b-versatile
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Type

from dotenv import load_dotenv, find_dotenv
from groq import Groq
from langsmith import traceable
from pydantic import BaseModel

load_dotenv(find_dotenv(usecwd=True), override=True)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model constant — verified available on this Groq account
# ---------------------------------------------------------------------------
GROQ_MODEL = "qwen/qwen3.8-27b"      # free tier, strong JSON, good daily limits
# Alternatives: "groq/compound" (30 RPM, 250 RPD), "openai/gpt-oss-120b" (paid)

# ---------------------------------------------------------------------------
# Singleton client
# ---------------------------------------------------------------------------
_groq_api_key = os.getenv("GROQ_API_KEY")
if not _groq_api_key:
    raise ValueError(
        "GROQ_API_KEY not set. Get a free key at https://console.groq.com "
        "and add it to your .env file."
    )
groq_client = Groq(api_key=_groq_api_key)

# ---------------------------------------------------------------------------
# Retry helper (handles 429 + 503 from Groq)
# ---------------------------------------------------------------------------
_MAX_ATTEMPTS = 5
_BASE_DELAY = 5.0
_BACKOFF = 2.0


def _is_retriable(exc: Exception) -> bool:
    exc_str = str(exc)
    return (
        "429" in exc_str
        or "rate_limit" in exc_str.lower()
        or "503" in exc_str
        or "502" in exc_str
        or "overloaded" in exc_str.lower()
        or getattr(exc, "status_code", None) in (429, 502, 503)
    )


# ---------------------------------------------------------------------------
# call_groq_json — structured JSON output via Groq
# ---------------------------------------------------------------------------

@traceable(run_type="llm", name="groq_json")
def call_groq_json(
    prompt: str,
    response_schema: Type[BaseModel],
    system_instruction: str = "",
) -> BaseModel:
    """
    Call Groq with JSON mode and parse the response into a Pydantic model.

    Uses response_format={"type": "json_object"} and instructs the model
    to match the schema via the system prompt.
    """
    # Build schema hint from the Pydantic model
    schema_json = json.dumps(response_schema.model_json_schema(), indent=2)

    system_parts = []
    if system_instruction:
        system_parts.append(system_instruction)
    system_parts.append(
        f"You MUST respond with valid JSON that matches this schema exactly:\n{schema_json}"
    )
    system_content = "\n\n".join(system_parts)

    delay = _BASE_DELAY
    last_exc: Exception | None = None

    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            response = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,   # low temp for consistent structured output
                max_tokens=512,
            )
            raw_json = response.choices[0].message.content
            # Handle "pass" keyword collision in Python
            data = json.loads(raw_json)
            if "pass" in data and "pass_" not in data:
                data["pass_"] = data.pop("pass")
            return response_schema.model_validate(data)

        except Exception as exc:
            last_exc = exc
            if _is_retriable(exc) and attempt < _MAX_ATTEMPTS:
                logger.warning(
                    "[Groq] Retriable error attempt %d/%d — retrying in %.0fs: %s",
                    attempt, _MAX_ATTEMPTS, delay, str(exc)[:120],
                )
                time.sleep(delay)
                delay *= _BACKOFF
            else:
                raise

    raise RuntimeError(f"Groq exhausted retries. Last error: {last_exc}")
