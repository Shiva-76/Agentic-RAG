# -*- coding: utf-8 -*-
"""
test_tools.py
-------------
Phase 3 verification tests.

Tests are intentionally importable but skipped with a clear message
until Phase 3 is implemented (tools raise NotImplementedError).

Run:  python tests/test_tools.py
"""

from __future__ import annotations

import sys
import os

# Allow imports from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.tools.calculator import calculator

# ---------------------------------------------------------------------------
# Calculator tests (fully implemented, safe to run now)
# ---------------------------------------------------------------------------

def test_calculator_basic() -> None:
    result = calculator("2 + 2")
    assert result == "4", f"Expected '4', got {result!r}"
    print(f"  [OK]  calculator('2 + 2') -> {result}")


def test_calculator_complex() -> None:
    result = calculator("3.14159 * 5**2")
    print(f"  [OK]  calculator('3.14159 * 5**2') -> {result}")


def test_calculator_invalid() -> None:
    result = calculator("import os; os.system('rm -rf /')")
    assert result == "INVALID_EXPRESSION", f"Expected 'INVALID_EXPRESSION', got {result!r}"
    print(f"  [OK]  calculator(malicious) -> {result}  (correctly rejected)")


# ---------------------------------------------------------------------------
# Stub tests for Phase 3 tools (will be filled in Phase 3)
# ---------------------------------------------------------------------------

def test_search_knowledge_base_stub() -> None:
    """Verifies the import path works; body tested in Phase 3."""
    from src.tools.knowledge_base import search_knowledge_base, MIN_SIMILARITY
    print(f"  [OK]  search_knowledge_base imported -- MIN_SIMILARITY={MIN_SIMILARITY}")
    print("     (full test runs after Phase 3 implementation)")


def test_search_web_stub() -> None:
    """Verifies the import path works; body tested in Phase 3."""
    from src.tools.web_search import search_web
    print("  [OK]  search_web imported (full test runs after Phase 3 implementation)")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n=== Phase 1: tool import & calculator tests ===\n")

    tests = [
        test_calculator_basic,
        test_calculator_complex,
        test_calculator_invalid,
        test_search_knowledge_base_stub,
        test_search_web_stub,
    ]

    failures = 0
    for test_fn in tests:
        try:
            test_fn()
        except NotImplementedError as exc:
            print(f"  [SKIP]  {test_fn.__name__} -- {exc}")
        except AssertionError as exc:
            print(f"  [FAIL]  {test_fn.__name__} -- {exc}")
            failures += 1
        except Exception as exc:
            print(f"  [ERR]   {test_fn.__name__} -- {exc}")
            failures += 1

    print()
    if failures:
        print(f"[FAIL] {failures} test(s) failed.")
        sys.exit(1)
    else:
        print("[PASS] All Phase 1 checks passed.")
