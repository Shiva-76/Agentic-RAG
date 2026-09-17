"""
test_graph.py
-------------
Phase 4/5 verification tests.

Tests each node in isolation with mock/real inputs, then tests the full
graph on the 6 canonical test cases from the implementation plan.

Run:  python tests/test_graph.py
"""

from __future__ import annotations

import sys
import os

# Allow imports from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def ok(msg: str) -> None:
    print(f"  [OK]   {msg}")


def skip(msg: str) -> None:
    print(f"  [SKIP] {msg}")


def fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


# ---------------------------------------------------------------------------
# Tool tests (Phase 3)
# ---------------------------------------------------------------------------

def test_calculator() -> bool:
    from src.tools.calculator import calculator
    assert calculator("2 ** 10") == "1024", "2^10 should be 1024"
    assert calculator("import os") == "INVALID_EXPRESSION", "Should reject code"
    ok("calculator — basic and injection tests pass")
    return True


def test_kb_import() -> bool:
    from src.tools.knowledge_base import search_knowledge_base, MIN_SIMILARITY
    ok(f"search_knowledge_base imported — MIN_SIMILARITY={MIN_SIMILARITY}")
    return True


def test_web_import() -> bool:
    from src.tools.web_search import search_web
    ok("search_web imported")
    return True


# ---------------------------------------------------------------------------
# Node import tests
# ---------------------------------------------------------------------------

def test_node_imports() -> bool:
    from src.agent.nodes import (
        node_rewrite_query,
        node_needs_retrieval,
        node_route_source,
        node_retrieve,
        node_synthesize,
        node_relevance_guard,
    )
    ok("All 6 nodes imported successfully")
    return True


def test_graph_import() -> bool:
    from src.agent.graph import graph, run_turn, AgentState
    ok("graph, run_turn, AgentState imported")
    return True


# ---------------------------------------------------------------------------
# Full graph integration tests (require GEMINI_API_KEY + Qdrant running)
# ---------------------------------------------------------------------------

def test_A_greeting() -> bool:
    """Test A: Greeting → no tool calls, warm reply."""
    from src.agent.graph import run_turn
    result = run_turn("Hi there!")
    answer = result.get("final_answer", "")
    node_names = [l["node"] for l in result.get("node_logs", [])]
    assert "retrieve" not in node_names, f"Should not retrieve for greeting, got nodes: {node_names}"
    assert answer, "Should produce an answer"
    ok(f"Test A (greeting) — 0 retrieval nodes, answer: {answer[:80]!r}")
    return True


def test_B_kb_question() -> bool:
    """Test B: Document question → KB retrieval, cited answer."""
    from src.agent.graph import run_turn
    result = run_turn("What does my document say about CNN accuracy in medical imaging?")
    answer = result.get("final_answer", "")
    node_names = [l["node"] for l in result.get("node_logs", [])]
    assert answer, "Should produce an answer"
    ok(f"Test B (KB) — nodes={node_names}, answer[:80]={answer[:80]!r}")
    return True


def test_C_web_question() -> bool:
    """Test C: Current events → web retrieval."""
    from src.agent.graph import run_turn
    result = run_turn("What is the current price of Bitcoin?")
    answer = result.get("final_answer", "")
    assert answer, "Should produce an answer"
    ok(f"Test C (web) — answer[:80]={answer[:80]!r}")
    return True


def test_D_calculator() -> bool:
    """Test D: Math → calculator."""
    from src.agent.graph import run_turn
    result = run_turn("What is 3.14159 * 100 squared?")
    answer = result.get("final_answer", "")
    assert answer, "Should produce an answer"
    ok(f"Test D (calculator) — answer[:80]={answer[:80]!r}")
    return True


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fast", action="store_true",
        help="Skip inter-test pauses (use on paid tier with higher RPM)"
    )
    parser.add_argument(
        "--imports-only", action="store_true",
        help="Run only import/unit tests, skip integration tests"
    )
    args = parser.parse_args()

    # Free-tier: 5 RPM, each turn = 3-5 calls → need ~65s between tests
    INTER_TEST_PAUSE = 0 if args.fast else 65

    import_tests = [
        ("Calculator tool", test_calculator),
        ("KB tool import", test_kb_import),
        ("Web tool import", test_web_import),
        ("Node imports", test_node_imports),
        ("Graph import", test_graph_import),
    ]

    integration_tests = [
        ("Test A — Greeting (no retrieval)", test_A_greeting),
        ("Test B — KB question", test_B_kb_question),
        ("Test C — Web question", test_C_web_question),
        ("Test D — Calculator", test_D_calculator),
    ]

    failures = 0

    section("Import & unit tests")
    for name, fn in import_tests:
        try:
            fn()
        except Exception as exc:
            fail(f"{name}: {exc}")
            failures += 1

    if not args.imports_only:
        section("Integration tests (require API key + Qdrant)")
        if INTER_TEST_PAUSE > 0:
            print(f"  [INFO] Free-tier mode: {INTER_TEST_PAUSE}s pause between tests to respect 5 RPM cap")

        for i, (name, fn) in enumerate(integration_tests):
            if i > 0 and INTER_TEST_PAUSE > 0:
                print(f"\n  ... waiting {INTER_TEST_PAUSE}s for RPM window to reset ...")
                time.sleep(INTER_TEST_PAUSE)
            try:
                fn()
            except NotImplementedError as exc:
                skip(f"{name} — {exc}")
            except AssertionError as exc:
                fail(f"{name}: {exc}")
                failures += 1
            except Exception as exc:
                fail(f"{name}: {type(exc).__name__}: {exc}")
                failures += 1

    print()
    if failures:
        print(f"[FAIL] {failures} test(s) failed.")
        sys.exit(1)
    else:
        print("[PASS] All tests passed.")

