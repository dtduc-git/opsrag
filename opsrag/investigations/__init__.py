"""Investigation event ledger -- DB-as-SoR for the Investigate page.

Public surface:
  - InvestigationEventStore -- CRUD + event append + tail-cursor reads
  - emit_event -- write a single event to the ledger via an existing pool
  - InvestigationStatus / EventType -- string constants for type-safety

Architecture (Option B refactor 2026-05-27):
  Every node in the Investigate-mode LangGraph emits structured events
  ("HYPOTHESIS_EVALUATED", "TOOL_RESULT", "INSIGHT_READY", ...) via
  emit_event(). The events land in opsrag_investigation_events.
  GET /investigations/{id}/events?since=N is a thin tail-cursor over
  that table -- the browser's EventSource just replays rows by sequence.

  Why: previously we tried to pipe live state through adispatch_custom_event +
  streaming HTTP. Closed tab = lost stream. Refused-generator answer = stuck
  cards. With DB ledger: refresh recovers, generator failures don't gate
  upstream card updates, and per-hypothesis verdicts come from a structured
  Pydantic evaluator call (NOT a regex over markdown).
"""
from opsrag.investigations.event_types import EventType
from opsrag.investigations.store import (
    InvestigationEventStore,
    InvestigationStatus,
    emit_event,
)

__all__ = [
    "InvestigationEventStore",
    "emit_event",
    "InvestigationStatus",
    "EventType",
]
