"""Pytest fixtures for OpsRAG eval harness.

Two cost guards:
- Session: hard $2.00 ceiling per test session (via pytest_sessionfinish).
- Per-test: $0.10 soft warning, log only, doesn't fail.
"""
from __future__ import annotations

import logging
from collections.abc import Generator

import pytest

# The eval harness requires the optional `eval` extra (deepeval). When it is
# absent, ignore this directory's tests at collection (collect_ignore_glob)
# rather than erroring on the imports below -- so the default suite / CI
# collection stay green without the extra. The fixtures that use the guarded
# names only run when the (now-ignored) tests run, so the guard is safe.
try:
    from opsrag.eval.adapters.vertex_judge import VertexGeminiJudge
    from opsrag.eval.usage_hook import get_usage_total
except ModuleNotFoundError:  # pragma: no cover - exercised only without the extra
    collect_ignore_glob = ["test_*.py"]
    VertexGeminiJudge = None  # type: ignore[assignment]
    get_usage_total = None  # type: ignore[assignment]

_log = logging.getLogger("opsrag.eval.conftest")

# Session-scoped cost tracking. Set in pytest_sessionstart, checked in
# pytest_sessionfinish so the failure surfaces in the test report.
_SESSION_START_COST: dict[str, float] = {}
_SESSION_COST_LIMIT_USD = 2.00
_PER_TEST_WARN_USD = 0.10


def pytest_sessionstart(session: pytest.Session) -> None:
    if get_usage_total is None:  # eval extra absent -> hooks no-op
        return
    _SESSION_START_COST["start"] = get_usage_total()
    _log.info("eval session start: cumulative cost = $%.4f", _SESSION_START_COST["start"])


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if get_usage_total is None:  # eval extra absent -> hooks no-op
        return
    end = get_usage_total()
    delta = end - _SESSION_START_COST.get("start", 0.0)
    _log.info("eval session end: cost delta = $%.4f", delta)
    if delta > _SESSION_COST_LIMIT_USD:
        # Emit a clear ERROR but don't crash pytest reporting; surface in stderr.
        _log.error(
            "EVAL SESSION COST EXCEEDED: $%.4f > $%.2f limit",
            delta, _SESSION_COST_LIMIT_USD,
        )
        # Set non-zero exit so CI flags it.
        session.exitstatus = max(exitstatus, 5)


@pytest.fixture(autouse=True)
def per_test_cost_warning() -> Generator[None, None, None]:
    """Soft warning per test if a single test costs more than $0.10."""
    start = get_usage_total()
    yield
    delta = get_usage_total() - start
    if delta > _PER_TEST_WARN_USD:
        _log.warning("test exceeded $%.2f: $%.4f", _PER_TEST_WARN_USD, delta)


@pytest.fixture(scope="session")
def judge() -> VertexGeminiJudge:
    """Shared judge instance -- Vertex Gemini Pro, default for faithfulness/GEval."""
    return VertexGeminiJudge(model_name="gemini-2.5-pro")


@pytest.fixture(scope="session")
def opsrag_url() -> str:
    """Where to hit the running OpsRAG instance for retrieval + generation."""
    import os
    return os.environ.get("OPSRAG_URL", "http://localhost:8000")
