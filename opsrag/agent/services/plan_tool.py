"""`update_plan` reasoner tool -- externalizes the investigation plan.

The reasoner LLM emits `update_plan` tool calls to keep an explicit,
operator-visible list of hypotheses it is tracking. Each round of tool
results, the reasoner can:

- add a new hypothesis (`status="open"` by default),
- transition a hypothesis to `testing`, `validated`, `invalidated`, or
  `deferred`,
- update the `evidence_so_far` one-liner,
- name the `next_tool` it intends to call next (informational only -- we
  do not drive dispatch from the plan).

This module is a **pure-function service** -- no LangGraph imports, no
LLM calls, no I/O. It exposes:

- `empty_plan()`            -- fresh empty plan.
- `update_plan(plan, ups)`  -- merge a list of partial-item updates into
                              the plan; returns `(new_plan, stats)`.
- `render_plan_for_prompt`  -- compact text rendering for the reasoner's
                              next-turn prompt.
- `to_sse_event(plan)`      -- payload envelope for the SSE
                              `render_component: plan_update` event.
- `PLAN_TOOL_SPEC`          -- MCP-shaped tool spec the reasoner LLM sees.

Wiring (done in follow-up edits, NOT here):
- `OpsRAGState` gains a `plan: list[dict]` field.
- `reasoner_node` in `agent/nodes/multi_agent.py` registers
  `update_plan` alongside the live MCP tools, intercepts the call,
  applies `update_plan(...)`, and pushes the new plan back onto state.
- `api/routes.py` forwards a `render_component: plan_update` event
  (payload from `to_sse_event`) on every plan mutation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

PlanStatus = Literal["open", "testing", "validated", "invalidated", "deferred"]

_VALID_STATUSES: tuple[PlanStatus, ...] = (
    "open",
    "testing",
    "validated",
    "invalidated",
    "deferred",
)
_TERMINAL_STATUSES: frozenset[str] = frozenset({"validated", "invalidated"})


@dataclass
class PlanItem:
    """In-code view of a single hypothesis in the investigation plan.

    The on-the-wire shape is plain `dict` (TypedDict-friendly, JSON-safe);
    this dataclass is provided for typed construction at call sites that
    prefer it. `update_plan` accepts dicts only.
    """

    id: str
    hypothesis: str
    status: PlanStatus = "open"
    next_tool: str | None = None
    evidence_so_far: str = ""
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "hypothesis": self.hypothesis,
            "status": self.status,
            "next_tool": self.next_tool,
            "evidence_so_far": self.evidence_so_far,
            "confidence": self.confidence,
        }


# --- tool spec the reasoner LLM sees --------------------------------

PLAN_TOOL_SPEC: dict[str, Any] = {
    "name": "update_plan",
    "description": (
        "Externalize your investigation plan. After each round of tool calls, "
        "call update_plan with a list of items reflecting your current state of "
        "thinking. Mark hypotheses validated/invalidated as evidence comes in, "
        "add new hypotheses you want to test next, defer ones you'll come back to. "
        "This is visible to the operator in real time -- keep hypothesis text concise."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "updates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id":              {"type": "string"},
                        "hypothesis":      {"type": "string"},
                        "status":          {
                            "type": "string",
                            "enum": list(_VALID_STATUSES),
                        },
                        # NB: JSON-Schema's union-with-null form (`["string","null"]`)
                        # crashes Vertex Gemini's tool-spec adapter (it calls
                        # .upper() on the type value). Stick to a single type;
                        # the LLM can omit the field when there's no next tool.
                        "next_tool":       {"type": "string"},
                        "evidence_so_far": {"type": "string"},
                        "confidence":      {"type": "number", "minimum": 0, "maximum": 1},
                    },
                    "required": ["id"],
                },
            },
        },
        "required": ["updates"],
    },
}


# --- pure-function helpers ------------------------------------------


def empty_plan() -> list[dict]:
    """Return a fresh, empty investigation plan."""
    return []


def _new_item(update: dict) -> dict:
    """Build a new plan item from an update dict, filling defaults.

    Caller has already ensured `id` is present.
    """
    item: dict[str, Any] = {
        "id": update["id"],
        "hypothesis": update.get("hypothesis", ""),
        "status": update.get("status", "open"),
        "next_tool": update.get("next_tool"),
        "evidence_so_far": update.get("evidence_so_far", ""),
        "confidence": _clamp_confidence(update.get("confidence", 0.0)),
    }
    if item["status"] not in _VALID_STATUSES:
        item["status"] = "open"
    if item["status"] in _TERMINAL_STATUSES:
        item["next_tool"] = None
    return item


def _clamp_confidence(value: Any) -> float:
    """Coerce + clamp a confidence value into [0.0, 1.0]. Non-numeric -> 0.0."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def update_plan(
    current_plan: list[dict],
    updates: list[dict],
) -> tuple[list[dict], dict]:
    """Merge `updates` into `current_plan`.

    Each update is a partial PlanItem keyed by `id`:
    - If `id` exists in `current_plan`: merge fields. Only keys present
      in the update overwrite; other fields are preserved.
    - If `id` is new: append as a new item. Must have at least `id` and
      `hypothesis` -- otherwise the update is skipped silently (an empty
      hypothesis is not a useful plan entry).
    - If an item's status transitions to `validated` or `invalidated`,
      `next_tool` is forced to `None` regardless of the update payload.

    Returns `(new_plan, stats)`. `stats` is a dict::

        {"added": N, "updated": N, "validated_now": N, "invalidated_now": N}

    Pure function -- no I/O, no mutation of the input list or its items.
    """
    # Index by id without mutating input items.
    index: dict[str, dict] = {}
    order: list[str] = []
    for item in current_plan:
        if not isinstance(item, dict):
            continue
        item_id = item.get("id")
        if not isinstance(item_id, str) or not item_id:
            continue
        index[item_id] = dict(item)
        order.append(item_id)

    stats = {"added": 0, "updated": 0, "validated_now": 0, "invalidated_now": 0}

    for update in updates or []:
        if not isinstance(update, dict):
            continue
        item_id = update.get("id")
        if not isinstance(item_id, str) or not item_id:
            continue

        if item_id in index:
            existing = index[item_id]
            prev_status = existing.get("status")
            merged = dict(existing)
            # Only keys actually present in the update overwrite.
            for key in ("hypothesis", "status", "next_tool", "evidence_so_far", "confidence"):
                if key in update:
                    if key == "confidence":
                        merged[key] = _clamp_confidence(update[key])
                    elif key == "status":
                        new_status = update[key]
                        if new_status in _VALID_STATUSES:
                            merged[key] = new_status
                    else:
                        merged[key] = update[key]
            # Terminal status forces next_tool = None.
            if merged.get("status") in _TERMINAL_STATUSES:
                merged["next_tool"] = None
            index[item_id] = merged
            stats["updated"] += 1
            new_status = merged.get("status")
            if new_status != prev_status:
                if new_status == "validated":
                    stats["validated_now"] += 1
                elif new_status == "invalidated":
                    stats["invalidated_now"] += 1
        else:
            # New items need a hypothesis to be meaningful.
            hypothesis = update.get("hypothesis")
            if not isinstance(hypothesis, str) or not hypothesis.strip():
                continue
            new_item = _new_item(update)
            index[item_id] = new_item
            order.append(item_id)
            stats["added"] += 1
            status = new_item.get("status")
            if status == "validated":
                stats["validated_now"] += 1
            elif status == "invalidated":
                stats["invalidated_now"] += 1

    new_plan = [index[i] for i in order]
    return new_plan, stats


def render_plan_for_prompt(plan: list[dict]) -> str:
    """Compact text rendering of the current plan, for the reasoner's
    next-turn prompt.

    Shape::

        Current plan (3 items):
          [h1] testing -- Pod is OOM-killed (next: prometheus_query_range)
          [h2] open    -- Helm template renders bad YAML
          [h3] validated (0.85) -- Memory limits set too low
    """
    if not plan:
        return "Current plan (0 items):"

    lines = [f"Current plan ({len(plan)} items):"]
    # Width-align the status column for readability.
    max_status_width = max(len(str(item.get("status", "open"))) for item in plan)

    for item in plan:
        item_id = item.get("id", "?")
        status = str(item.get("status", "open"))
        hypothesis = str(item.get("hypothesis", "")).strip()
        next_tool = item.get("next_tool")
        confidence = item.get("confidence")

        # Status segment -- append "(conf)" only when terminal-validated.
        status_segment = status.ljust(max_status_width)
        if status == "validated" and isinstance(confidence, (int, float)):
            status_segment = f"{status} ({float(confidence):.2f})"

        suffix = ""
        if next_tool and status not in _TERMINAL_STATUSES:
            suffix = f" (next: {next_tool})"

        lines.append(f"  [{item_id}] {status_segment} -- {hypothesis}{suffix}")

    return "\n".join(lines)


def to_sse_event(plan: list[dict]) -> dict:
    """Return the payload for an SSE `render_component: plan_update` event.

    Shape::

        {"component": "InvestigationPlan", "props": {"items": [...]}}
    """
    return {
        "component": "InvestigationPlan",
        "props": {"items": list(plan)},
    }


__all__ = [
    "PlanItem",
    "PLAN_TOOL_SPEC",
    "empty_plan",
    "update_plan",
    "render_plan_for_prompt",
    "to_sse_event",
]
