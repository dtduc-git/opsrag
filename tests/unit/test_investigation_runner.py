"""Tests for the Engine B (InvestigationRunner) guardrails ported from the
retired hypothesis-tree engine: the per-run budget + per-hypothesis
evidence citations on the evaluator."""
from __future__ import annotations

import time

import pytest

from opsrag.investigations.evaluator import (
    HypothesisVerdict,
    HypothesisVerdictBatch,
    evaluate_hypotheses,
)
from opsrag.investigations.runner import (
    MAX_INVESTIGATION_TOOL_CALLS,
    MAX_INVESTIGATION_WALL_CLOCK_SEC,
    _RunBudget,
)

# --- budget breakers ------------------------------------------------


def test_run_budget_wall_clock_breaker():
    fresh = _RunBudget(started_at=time.monotonic())
    assert not fresh.wall_clock_exceeded()
    stale = _RunBudget(started_at=time.monotonic() - (MAX_INVESTIGATION_WALL_CLOCK_SEC + 10))
    assert stale.wall_clock_exceeded()


def test_run_budget_tool_cap_breaker():
    b = _RunBudget(started_at=time.monotonic())
    assert not b.tool_budget_exhausted()
    b.tool_calls = MAX_INVESTIGATION_TOOL_CALLS - 1
    assert not b.tool_budget_exhausted()
    b.tool_calls = MAX_INVESTIGATION_TOOL_CALLS
    assert b.tool_budget_exhausted()


# --- per-hypothesis citations --------------------------------------


class _FakeFlash:
    """Returns a fixed verdict batch carrying supporting_tools."""

    async def generate_structured(self, messages, schema, system_prompt=None, purpose=None):
        return HypothesisVerdictBatch(verdicts=[
            HypothesisVerdict(
                hypothesis_id="h1", status="confirmed",
                evidence="pod OOMKilled", confidence=0.8,
                supporting_tools=["#1 k8s_get_pod", "#3 prometheus_query"],
            ),
        ])


class _BoomFlash:
    async def generate_structured(self, messages, schema, system_prompt=None, purpose=None):
        raise RuntimeError("evaluator down")


@pytest.mark.asyncio
async def test_evaluator_returns_supporting_tools():
    batch = await evaluate_hypotheses(
        llm=_FakeFlash(),
        hypotheses=[{"id": "h1", "text": "OOM", "discriminating_tools": ["k8s_get_pod"]}],
        evidence_pool="#1 [k8s_get_pod] result: OOMKilled",
        incident_target="svc",
    )
    assert batch.verdicts[0].supporting_tools == ["#1 k8s_get_pod", "#3 prometheus_query"]


@pytest.mark.asyncio
async def test_evaluator_fallback_has_empty_citations():
    batch = await evaluate_hypotheses(
        llm=_BoomFlash(),
        hypotheses=[{"id": "h1", "text": "x"}],
        evidence_pool="",
    )
    assert batch.verdicts[0].status == "untested"
    assert batch.verdicts[0].supporting_tools == []
