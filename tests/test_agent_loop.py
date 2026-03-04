"""
Regression tests for AgentLoop._extract_summary.

These tests require agentscope to be installed (skipped otherwise).
They do NOT require any LLM/network access — _extract_summary is pure data
transformation that never calls self.goal_evaluator.

Bug being tested:
    _extract_summary() only read `phase_results` key (legacy format).
    OrchestratorAgent returns `{completed_tasks, remaining_gaps}` directly —
    no `phase_results` key — so GoalEvaluator saw empty completed_tasks and
    triggered a second (unnecessary) iteration.
"""
import sys
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

# Skip entire module when agentscope is not installed.
# In the container agentscope IS installed so all tests run.
# In a bare WSL dev environment without agentscope the module is skipped.
# NOTE: no sys.modules patching here — patching at module level would pollute
# the process-wide sys.modules and cause test_executor_http_requests.py to
# receive a MagicMock instead of the real agentscope (ScopeMismatch root cause).
pytest.importorskip("agentscope", reason="agentscope not installed")

from loop.agent_loop import AgentLoop  # noqa: E402
from loop.iteration_context import LoopIterationSummary  # noqa: E402


# ── Minimal AgentLoop factory (no real coordinator/evaluator needed) ──────────

class _DummyCoordinator:
    progress_callback = None


def _make_loop() -> AgentLoop:
    return AgentLoop(
        coordinator=_DummyCoordinator(),
        goal_evaluator=MagicMock(),
        max_iterations=3,
    )


# ── Legacy format ──────────────────────────────────────────────────────────────

def test_extract_summary_legacy_format():
    """
    Legacy coordinator result with phase_results → completed_tasks populated.
    """
    result: Dict[str, Any] = {
        "status": "completed",
        "phase_results": [
            {
                "phase": 1,
                "phase_name": "Analysis",
                "status": "completed",
                "worker_results": {
                    "analyzer": {
                        "status": "success",
                        "output": "Found 3 APIs",
                        "task_description": "Analyze API specification",
                    }
                },
            },
            {
                "phase": 2,
                "phase_name": "Testing",
                "status": "completed",
                "worker_results": {
                    "executor": {
                        "status": "completed",
                        "output": "All tests passed",
                        "task_description": "Execute API tests",
                    }
                },
            },
        ],
    }

    loop = _make_loop()
    summary: LoopIterationSummary = loop._extract_summary(result, iteration=0)

    assert len(summary.completed_tasks) == 2, (
        f"Expected 2 completed tasks from phase_results, got: {summary.completed_tasks}"
    )
    assert summary.status == "success"
    assert summary.phase_count == 2


def test_extract_summary_legacy_partial():
    """
    phase_results with mixed success/failed workers → only success entries counted.
    """
    result: Dict[str, Any] = {
        "status": "partial",
        "phase_results": [
            {
                "phase": 1,
                "phase_name": "Phase A",
                "status": "partial",
                "worker_results": {
                    "worker_ok": {"status": "success", "output": "done", "task_description": "Task A"},
                    "worker_fail": {"status": "failed", "output": "error", "task_description": "Task B"},
                },
            }
        ],
    }

    loop = _make_loop()
    summary = loop._extract_summary(result, iteration=0)

    assert len(summary.completed_tasks) == 1
    assert summary.completed_tasks[0] == "Task A"
    assert summary.status == "partial"


# ── OrchestratorAgent format (REGRESSION) ─────────────────────────────────────

def test_extract_summary_orchestrator_format():
    """
    REGRESSION: OrchestratorAgent result has no phase_results key.
    completed_tasks should be extracted from the top-level list.
    """
    result: Dict[str, Any] = {
        "status": "completed",
        "completed_tasks": [
            "Configured http_client with base_url=https://httpbin.org",
            "Executed GET /cookies/set?user=test → 200 OK",
            "Verified cookie persistence: user=test",
        ],
        "remaining_gaps": [],
    }

    loop = _make_loop()
    summary = loop._extract_summary(result, iteration=0)

    assert len(summary.completed_tasks) == 3, (
        f"REGRESSION: orchestrator completed_tasks not extracted. "
        f"Got: {summary.completed_tasks}"
    )
    assert summary.status == "success", (
        f"Expected loop_status='success' for orchestrator 'completed', got '{summary.status}'"
    )


def test_extract_summary_orchestrator_format_failed():
    """
    OrchestratorAgent result with status=failed → loop_status should be failed.
    """
    result: Dict[str, Any] = {
        "status": "failed",
        "completed_tasks": [],
        "remaining_gaps": ["Could not connect to target service"],
    }

    loop = _make_loop()
    summary = loop._extract_summary(result, iteration=0)

    assert summary.status == "failed"


def test_extract_summary_empty_orchestrator():
    """
    OrchestratorAgent returns completed tasks + remaining_gaps=[] → loop_status=success.

    This is the specific scenario that caused the double-iteration bug:
    The old code left completed_tasks=[] so GoalEvaluator got no evidence of
    completion, evaluated achieved=False, and ran a second iteration.
    """
    result: Dict[str, Any] = {
        "status": "completed",
        "completed_tasks": [
            "Tested cookie persistence end-to-end",
        ],
        "remaining_gaps": [],
    }

    loop = _make_loop()
    summary = loop._extract_summary(result, iteration=0)

    assert summary.completed_tasks, (
        "completed_tasks must be non-empty so GoalEvaluator can evaluate achieved=True "
        "and stop the loop after the first iteration."
    )
    assert summary.status == "success"
