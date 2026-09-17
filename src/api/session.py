"""
session.py
----------
In-memory session store: maps session_id → conversation_history.

In production you'd swap this for Redis or a DB. For the portfolio
build, in-memory is fine and makes the architecture easy to explain.
"""
from __future__ import annotations

import threading
from typing import Dict, List

# Thread-safe dictionary: session_id → list of Gemini-format Content dicts
_store: Dict[str, List[dict]] = {}
_lock = threading.Lock()


def get_history(session_id: str) -> List[dict]:
    """Return conversation history for a session (empty list if new)."""
    with _lock:
        return list(_store.get(session_id, []))


def set_history(session_id: str, history: List[dict]) -> None:
    """Replace the full conversation history for a session."""
    with _lock:
        _store[session_id] = list(history)


def clear_history(session_id: str) -> None:
    """Clear conversation history (e.g. on 'new chat')."""
    with _lock:
        _store.pop(session_id, None)


def list_sessions() -> List[str]:
    """Return all active session IDs."""
    with _lock:
        return list(_store.keys())
