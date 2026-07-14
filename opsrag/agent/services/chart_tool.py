"""`render_chart` reasoner tool -- generic, tool-agnostic data visualization.

Any tool result (billing, k8s counts, cost breakdowns, comparisons -- anything
numeric/tabular) can be turned into a chart by the reasoner calling
``render_chart`` with a small spec. This deliberately does NOT live next to any
one data source: the reasoner extracts the series from whatever tool output it
has, then calls ``render_chart`` once. New tools get charts for free -- they
just call this.

Like ``plan_tool``, this is a **pure-function service** (no LangGraph/LLM/IO):

- ``RENDER_CHART_TOOL_SPEC``      -- MCP-shaped tool spec the reasoner sees.
- ``build_chart_spec(args)``      -- validate/normalize a raw tool-call payload
                                     into a safe chart spec (or ``None``).
- ``to_component(spec)``          -- ``{component, props}`` envelope, same shape
                                     the frontend rich-component registry renders.
- ``extract_chart_from_history``  -- rebuild chart components from a turn's
                                     persisted ``tool_message_history`` (used by
                                     BOTH the live SSE stream and replay, so a
                                     chart survives refresh/history reload).

Wiring (elsewhere, not here):
- ``tool_caller_node`` in ``agent/nodes/multi_agent.py`` offers
  ``RENDER_CHART_TOOL_SPEC`` and intercepts the call: validates via
  ``build_chart_spec`` and appends a ``render_chart`` tool_result carrying the
  spec into ``tool_message_history``.
- ``api/routes.py`` (live) and ``sessions/replay_components.py`` (replay) call
  ``extract_chart_from_history`` to emit/rebuild the ``Chart`` component.
"""
from __future__ import annotations

import json
import logging
from typing import Any

_log = logging.getLogger("opsrag.agent.services.chart_tool")

# Supported chart kinds. Kept intentionally small (v1): line for time-series,
# bar for categorical/temporal comparison, pie for distribution.
_VALID_TYPES: frozenset[str] = frozenset({"line", "bar", "pie"})

# Defensive caps so a degenerate spec can't blow the SSE payload / DOM.
_MAX_SERIES = 8
_MAX_POINTS = 500

TOOL_NAME = "render_chart"


# --- tool spec the reasoner LLM sees --------------------------------

RENDER_CHART_TOOL_SPEC: dict[str, Any] = {
    "name": TOOL_NAME,
    "description": (
        "Render an interactive chart from data you already retrieved. Use this "
        "INSTEAD of a markdown table whenever the user asks for a chart/graph or "
        "when a visual makes numeric data clearer -- cost over time, cost/usage "
        "breakdowns, counts, comparisons. Extract the series from your prior tool "
        "results and pass them here. type=line for a time-series (e.g. cost per "
        "month/day), type=bar for a categorical comparison (e.g. cost per project/"
        "service), type=pie for a share-of-total distribution. Each point's `x` is "
        "the category or time label (a string, e.g. 'May 2026' or a project/service "
        "name) and `y` is the numeric value. Call this at most once per turn."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "type": {"type": "string", "enum": ["line", "bar", "pie"]},
            "title": {"type": "string"},
            # NB: single JSON-Schema types only -- the `["string","null"]` union
            # form crashes Vertex Gemini's tool-spec adapter (.upper() on type).
            "unit": {"type": "string"},
            "x_label": {"type": "string"},
            "y_label": {"type": "string"},
            "series": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string"},
                        "points": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "x": {"type": "string"},
                                    "y": {"type": "number"},
                                },
                                "required": ["x", "y"],
                            },
                        },
                    },
                    "required": ["label", "points"],
                },
            },
        },
        "required": ["type", "title", "series"],
    },
}


# --- pure-function helpers ------------------------------------------


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_chart_spec(args: dict[str, Any]) -> dict[str, Any] | None:
    """Validate + normalize a raw ``render_chart`` payload into a safe spec.

    Returns the normalized spec, or ``None`` if it carries no plottable data
    (bad type / no valid series) so the caller can surface an error to the LLM.
    Never raises on bad input.
    """
    if not isinstance(args, dict):
        return None
    ctype = args.get("type")
    if ctype not in _VALID_TYPES:
        return None
    title = str(args.get("title") or "").strip() or "Chart"

    series_out: list[dict] = []
    for s in (args.get("series") or [])[:_MAX_SERIES]:
        if not isinstance(s, dict):
            continue
        points_out: list[dict] = []
        for p in (s.get("points") or [])[:_MAX_POINTS]:
            if not isinstance(p, dict):
                continue
            y = _coerce_float(p.get("y"))
            if y is None:
                continue
            # x kept as a string label (time or category); coerce non-strings.
            x = p.get("x")
            x = str(x) if x is not None else ""
            points_out.append({"x": x, "y": y})
        if points_out:
            series_out.append({
                "label": str(s.get("label") or "").strip() or "series",
                "points": points_out,
            })

    if not series_out:
        return None

    spec: dict[str, Any] = {
        "type": ctype,
        "title": title,
        "series": series_out,
    }
    # Optional labels -- only include when non-empty so props stay tidy.
    for key in ("unit", "x_label", "y_label"):
        val = args.get(key)
        if isinstance(val, str) and val.strip():
            spec[key] = val.strip()
    return spec


def to_component(spec: dict[str, Any]) -> dict[str, Any]:
    """Envelope a validated spec as the frontend rich-component payload.

    Shape mirrors the live ``render_component`` event and ``plan_tool``::

        {"component": "Chart", "props": {<spec>}}
    """
    return {"component": "Chart", "props": dict(spec)}


def extract_chart_from_history(tool_message_history: list[dict] | None) -> list[dict]:
    """Rebuild ``Chart`` components from persisted ``render_chart`` tool results.

    Scans the turn's tool history for ``render_chart`` tool_results (whose
    ``response.text`` carries the JSON spec written by the interceptor) and
    returns one ``{component, props}`` per valid chart. Best-effort: malformed
    entries are skipped, never raised.
    """
    components: list[dict] = []
    for msg in tool_message_history or []:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "tool_result" or (msg.get("name") or "") != TOOL_NAME:
            continue
        resp = msg.get("response") or {}
        text = resp.get("text") if isinstance(resp, dict) else None
        if not isinstance(text, str) or not text.strip():
            continue
        try:
            spec = json.loads(text)
        except Exception:  # noqa: BLE001 -- skip a corrupt spec, keep the rest
            continue
        if isinstance(spec, dict) and spec.get("series"):
            components.append(to_component(spec))
    return components


__all__ = [
    "TOOL_NAME",
    "RENDER_CHART_TOOL_SPEC",
    "build_chart_spec",
    "to_component",
    "extract_chart_from_history",
]
