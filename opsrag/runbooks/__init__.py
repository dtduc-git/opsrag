"""Hand-authored runbook store + retrieval.

Layout:
  - taxonomy.py -- closed enums for failure_class, symptom_class, etc.
  - models.py   -- Pydantic models for Runbook + RunbookVersion + RunbookHit
  - store.py    -- RunbookStore class (CRUD + hybrid retrieval)
  - tagger.py   -- Flash auto-tagger for tagging investigations (Phase 3)
  - generator.py -- Pro runbook-from-investigation generator (Phase 4)
  - lane.py     -- Lane A retrieval node (Phase 5)

Hand-authored runbooks ALWAYS rank above RAG-indexed runbooks for the
same query -- the binary priority is enforced at the retrieval layer
(Lane A), not by score weighting. See lane.py for the merging logic.
"""
from opsrag.runbooks.models import (
    Runbook,
    RunbookCreate,
    RunbookHit,
    RunbookUpdate,
    RunbookVersion,
)
from opsrag.runbooks.store import RunbookStore
from opsrag.runbooks.taxonomy import (
    FAILURE_CLASSES,
    RESOLUTION_CLASSES,
    SEVERITIES,
    SYMPTOM_CLASSES,
)

__all__ = [
    "Runbook",
    "RunbookCreate",
    "RunbookHit",
    "RunbookUpdate",
    "RunbookVersion",
    "RunbookStore",
    "FAILURE_CLASSES",
    "RESOLUTION_CLASSES",
    "SEVERITIES",
    "SYMPTOM_CLASSES",
]
