"""`render_chart` engine tool -- spec validation + history extraction.

Locks in that the reasoner's render_chart call turns into a `Chart` component
both live and on replay, and that a malformed spec is rejected (so the LLM gets
an error instead of a blank/garbage chart).
"""
from __future__ import annotations

import json

from opsrag.agent.services.chart_tool import (
    RENDER_CHART_TOOL_SPEC,
    TOOL_NAME,
    build_chart_spec,
    extract_chart_from_history,
    to_component,
)


def test_tool_spec_shape():
    assert RENDER_CHART_TOOL_SPEC["name"] == "render_chart"
    props = RENDER_CHART_TOOL_SPEC["input_schema"]["properties"]
    assert props["type"]["enum"] == ["line", "bar", "pie"]
    assert set(RENDER_CHART_TOOL_SPEC["input_schema"]["required"]) == {"type", "title", "series"}


def test_build_valid_line_spec():
    spec = build_chart_spec({
        "type": "line",
        "title": "GCP PRD cost / month",
        "unit": "USD",
        "series": [{"label": "cost", "points": [
            {"x": "May 2026", "y": 49508.64},
            {"x": "June 2026", "y": "51472.20"},  # string y coerced
        ]}],
    })
    assert spec is not None
    assert spec["type"] == "line" and spec["unit"] == "USD"
    assert spec["series"][0]["points"] == [
        {"x": "May 2026", "y": 49508.64},
        {"x": "June 2026", "y": 51472.20},
    ]


def test_build_rejects_bad_type_and_empty_series():
    assert build_chart_spec({"type": "scatter", "title": "x", "series": []}) is None
    assert build_chart_spec({"type": "bar", "title": "x", "series": []}) is None
    # A series whose points all have non-numeric y drops out -> no plottable data.
    assert build_chart_spec({
        "type": "bar", "title": "x",
        "series": [{"label": "s", "points": [{"x": "a", "y": "NaNish"}]}],
    }) is None


def test_extract_from_history_roundtrip():
    spec = build_chart_spec({
        "type": "bar", "title": "Cost by project",
        "series": [{"label": "cost", "points": [{"x": "prod", "y": 100.0}]}],
    })
    history = [
        {"role": "tool_call", "name": TOOL_NAME, "args": {"type": "bar"}},
        {"role": "tool_result", "name": TOOL_NAME, "response": {"text": json.dumps(spec)}},
    ]
    comps = extract_chart_from_history(history)
    assert comps == [to_component(spec)]
    assert comps[0]["component"] == "Chart"


def test_extract_ignores_other_tools_and_bad_json():
    history = [
        {"role": "tool_result", "name": "prometheus_query", "response": {"text": "{}"}},
        {"role": "tool_result", "name": TOOL_NAME, "response": {"text": "{not json"}},
    ]
    assert extract_chart_from_history(history) == []
    assert extract_chart_from_history(None) == []
