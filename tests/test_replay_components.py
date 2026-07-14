"""Replayed-turn rich-component rebuild.

Charts + the investigation plan are emitted only as transient live SSE events;
on replay we re-derive them from the persisted ``tool_message_history`` / ``plan``
channels. These tests lock in that a persisted Prometheus matrix rebuilds a
``TimeseriesChart`` and a persisted plan rebuilds an ``InvestigationPlan`` --
guarding the "charts vanish on refresh" regression.
"""
from __future__ import annotations

import json

from opsrag.sessions.replay_components import rebuild_rich_components


def _matrix_tool_history() -> list[dict]:
    payload = {
        "data": {
            "resultType": "matrix",
            "result": [
                {
                    "metric": {"__name__": "container_cpu_usage_seconds_total", "pod": "p1"},
                    "values": [[1700000000, "0.5"], [1700000060, "0.7"]],
                }
            ],
        }
    }
    return [
        {"role": "tool_call", "name": "prometheus_query_range",
         "args": {"query": "sum(rate(container_cpu_usage_seconds_total[5m]))"}},
        {"role": "tool_result", "name": "prometheus_query_range",
         "response": {"text": json.dumps(payload)}},
    ]


def test_rebuild_chart_from_persisted_matrix():
    comps = rebuild_rich_components(_matrix_tool_history(), None)
    charts = [c for c in comps if c["component"] == "TimeseriesChart"]
    assert len(charts) == 1
    series = charts[0]["props"]["series"]
    assert series and series[0]["points"] == [
        {"ts": 1700000000.0, "value": 0.5},
        {"ts": 1700000060.0, "value": 0.7},
    ]


def test_rebuild_plan_component():
    plan = [{"id": "s1", "hypothesis": "check cpu", "status": "done"}]
    comps = rebuild_rich_components(None, plan)
    plans = [c for c in comps if c["component"] == "InvestigationPlan"]
    assert len(plans) == 1
    assert plans[0]["props"]["items"] == plan


def test_rebuild_chart_and_plan_together():
    plan = [{"id": "s1", "hypothesis": "h", "status": "pending"}]
    comps = rebuild_rich_components(_matrix_tool_history(), plan)
    kinds = {c["component"] for c in comps}
    assert kinds == {"TimeseriesChart", "InvestigationPlan"}


def test_rebuild_empty_is_noop():
    assert rebuild_rich_components(None, None) == []
    assert rebuild_rich_components([], []) == []
