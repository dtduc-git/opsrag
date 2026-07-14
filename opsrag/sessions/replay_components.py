"""Rebuild renderable rich-components for a REPLAYED assistant turn.

Interactive charts (Prometheus ``TimeseriesChart``) and the
``InvestigationPlan`` are emitted only as transient SSE ``render_component``
events during the live stream -- they were never persisted, so reopening a
conversation (or a page refresh) lost them while the text answer survived.

Their SOURCE data *is* persisted: ``tool_message_history`` and ``plan`` are
accumulated ``AgentState`` channels that land in each checkpoint's
``channel_values``. So on replay we re-derive the exact same component dicts
using the SAME extractor the live stream uses -- guaranteeing identical
rendering with no schema/migration change.

Returned shape matches the live ``render_component`` payload the frontend
already knows how to render::

    [{"component": "TimeseriesChart", "props": {...}}, ...]

Imports are function-local so this module never participates in an import
cycle with the (heavy) ``opsrag.api.routes`` module at load time.
"""
from __future__ import annotations

import logging

_log = logging.getLogger("opsrag.sessions.replay_components")


def rebuild_rich_components(
    tool_message_history: list[dict] | None,
    plan: list[dict] | None,
) -> list[dict]:
    """Re-derive the chart + plan components for a persisted turn.

    Best-effort: any extraction failure yields fewer components, never an
    error -- a missing chart on replay is the pre-existing behaviour, so it
    must never break history loading.
    """
    components: list[dict] = []
    # Charts -- reuse the live-stream extractor verbatim so replay renders
    # identically to the original turn.
    try:
        from opsrag.api.routes import _extract_chart_components

        components.extend(_extract_chart_components(tool_message_history or []))
    except Exception as exc:  # noqa: BLE001 -- charts are a nice-to-have
        _log.warning("replay chart rebuild failed (non-fatal): %s", exc)
    # Generic `render_chart` charts (tool-agnostic) -- same extractor the live
    # stream uses, so replay renders them identically.
    try:
        from opsrag.agent.services.chart_tool import extract_chart_from_history

        components.extend(extract_chart_from_history(tool_message_history or []))
    except Exception as exc:  # noqa: BLE001
        _log.warning("replay render_chart rebuild failed (non-fatal): %s", exc)
    # Investigation plan -- same component the live stream emits via
    # `render_component: InvestigationPlan`.
    try:
        if plan:
            from opsrag.agent.services.plan_tool import to_sse_event

            components.append(to_sse_event(plan))
    except Exception as exc:  # noqa: BLE001
        _log.warning("replay plan rebuild failed (non-fatal): %s", exc)
    return components
