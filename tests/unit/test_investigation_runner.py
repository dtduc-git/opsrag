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
from opsrag.investigations.event_types import EventType
from opsrag.investigations.runner import (
    _GENERIC_INVESTIGATION_ERROR,
    MAX_INVESTIGATION_TOOL_CALLS,
    MAX_INVESTIGATION_WALL_CLOCK_SEC,
    InvestigationDeps,
    InvestigationRunner,
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


# --- SSE stack-trace / raw-exception sanitization (track sse, M7) ------
#
# Every error path that feeds an investigation SSE payload must emit the
# generic string -- never str(exc)/traceback -- mirroring the #110
# query-path fix. The real detail must still reach the server logs (and,
# for tool errors, the in-process reasoner history).

# A unique, unmistakable substring of the raised exception text. If this
# leaks into any SSE payload the assertions below fail.
_SECRET = "SECRET-trace-/srv/opsrag/runner.py:1337-do-not-leak"


class _CapturingStore:
    """Captures every emit_event payload so tests can assert what would
    have crossed the SSE boundary. Mirrors the InvestigationEventStore
    surface the runner touches (append_event + mark_status)."""

    def __init__(self) -> None:
        self.events: list[dict] = []

    async def append_event(self, *, investigation_id, event_type, payload=None, tags=None):
        self.events.append({"event_type": event_type, "payload": payload or {}})
        return len(self.events)

    async def mark_status(self, *args, **kwargs):
        return None

    def payloads_for(self, event_type) -> list[dict]:
        return [e["payload"] for e in self.events if e["event_type"] == event_type]


def _runner_with(store, **deps_overrides) -> InvestigationRunner:
    deps_overrides.setdefault("flash_llm", None)
    deps_overrides.setdefault("pro_llm", None)
    deps = InvestigationDeps(event_store=store, **deps_overrides)
    return InvestigationRunner(deps)


class _BoomEmbedder:
    async def embed_query(self, text):
        raise RuntimeError(_SECRET)


class _StoreWithSearch:
    """A runbook/investigation store stub that exposes search-like methods
    so the lane reaches embed_query (which raises _SECRET)."""

    async def search(self, *args, **kwargs):
        return []

    async def search_similar(self, *args, **kwargs):
        return []


class _BoomTool:
    """A reasoner tool whose call() raises an exception carrying _SECRET."""

    name = "k8s_get_pod"

    async def call(self, client, args):
        raise RuntimeError(_SECRET)


def _assert_no_secret(payload: dict) -> None:
    blob = repr(payload)
    assert _SECRET not in blob, f"raw exception text leaked into SSE payload: {payload!r}"


@pytest.mark.asyncio
async def test_lane_a_error_payload_is_generic():
    # embedder.embed_query raises -> lane_a hits the except branch. The
    # returned dict is emitted verbatim as the LANE_A_COMPLETED payload.
    runner = _runner_with(
        _CapturingStore(), embedder=_BoomEmbedder(), runbook_store=_StoreWithSearch(),
    )
    result = await runner._lane_a("boom alert", target=None)
    assert result["error"] == _GENERIC_INVESTIGATION_ERROR
    _assert_no_secret(result)


@pytest.mark.asyncio
async def test_lane_b_error_payload_is_generic():
    runner = _runner_with(
        _CapturingStore(), embedder=_BoomEmbedder(), investigation_cache=_StoreWithSearch(),
    )
    result = await runner._lane_b("boom alert")
    assert result["error"] == _GENERIC_INVESTIGATION_ERROR
    _assert_no_secret(result)


@pytest.mark.asyncio
async def test_lane_c_error_payload_is_generic():
    # A Rootly URL routes to the tool registry; the tool raises _SECRET.
    class _BoomRootly:
        async def call(self, client, args):
            raise RuntimeError(_SECRET)

    runner = _runner_with(
        _CapturingStore(), tool_registry={"rootly_get_alert": _BoomRootly()},
    )
    url = "https://x.rootly.com/account/alerts/abc123"
    result = await runner._lane_c(url)
    assert result["error"] == _GENERIC_INVESTIGATION_ERROR
    # The "summary" field must not interpolate the exception text either.
    _assert_no_secret(result)


class _ToolCall:
    def __init__(self, name, args=None):
        self.name = name
        self.args = args or {}


class _Resp:
    def __init__(self, tool_calls):
        self.tool_calls = tool_calls
        self.content = ""


class _OneCallProLLM:
    """A pro_llm that, when asked to pick tools, returns one call to the
    boom tool so the reasoner round dispatches it."""

    async def generate_with_tools(self, *, messages, tools, system_prompt=None,
                                  temperature=0.0, max_tokens=2048, purpose=None):
        return _Resp([_ToolCall("k8s_get_pod", {})])


@pytest.mark.asyncio
async def test_reasoner_tool_error_sse_generic_but_history_keeps_detail():
    """The TOOL_RESULT SSE payload must carry a generic 'tool execution
    failed' string, while the in-process history STILL keeps the real
    str(exc) for the evaluator/reasoner (it never leaves the process)."""
    store = _CapturingStore()
    runner = _runner_with(
        store,
        pro_llm=_OneCallProLLM(),
        tool_registry={"k8s_get_pod": _BoomTool()},
    )

    history = await runner._reasoner_tool_round(
        inv_id="inv-1",
        alert_text="alert",
        # discriminating_tools surfaces the boom tool into tool_specs.
        hypotheses=[{"id": "h1", "text": "OOM", "discriminating_tools": ["k8s_get_pod"]}],
        target=None,
        insight={},
    )

    # In-process history keeps the real exception text.
    assert len(history) == 1
    assert _SECRET in history[0]["error"]

    # The SSE TOOL_RESULT payload is generic -- no exception text.
    tool_results = store.payloads_for(EventType.TOOL_RESULT)
    assert tool_results, "expected a TOOL_RESULT event"
    assert tool_results[-1]["error"] == "tool execution failed"
    for p in tool_results:
        _assert_no_secret(p)
    # And no other emitted payload (REASONER_STEP, TOOL_CALLED) leaks it.
    for ev in store.events:
        _assert_no_secret(ev["payload"])


@pytest.mark.asyncio
async def test_run_one_failure_payload_is_generic():
    store = _CapturingStore()
    runner = _runner_with(store)

    async def _boom_pipeline(inv_id, alert_text):
        raise RuntimeError(_SECRET)

    runner._run_pipeline = _boom_pipeline  # type: ignore[assignment]
    await runner.run_one("inv-1", "alert")

    failed = store.payloads_for(EventType.INVESTIGATION_FAILED)
    assert failed, "expected an INVESTIGATION_FAILED event"
    assert failed[-1]["error"] == _GENERIC_INVESTIGATION_ERROR
    for p in failed:
        _assert_no_secret(p)
