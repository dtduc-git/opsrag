"""Sub-sprint 3 V1 -- investigation cache.

Separate from the Q&A cache (`opsrag.qa_cache`). The Q&A cache is
keyed on the question alone and serves stable corpus answers; this
cache stores tool-path investigation outcomes (the calls made, the
trace data fetched, the synthesized answer) and surfaces relevant
past investigations to the reasoner as additional context.

V1 scope per the plan:
- New Qdrant collection `opsrag_investigations`
- Schema: question, answer, tool_call_audit, model_route_decision,
  created_at, embedding
- Writer triggered after every successful tool-path completion
- Cosine-similarity search; reasoner reads top-K as context
- No tag taxonomy, no funneled search, no decay (those land in V2)
"""
from opsrag.agent.cache.investigation_cache import (
    DEFAULT_INVESTIGATION_COLLECTION,
    DEFAULT_INVESTIGATION_THRESHOLD,
    InvestigationCache,
    InvestigationHit,
)

__all__ = [
    "DEFAULT_INVESTIGATION_COLLECTION",
    "DEFAULT_INVESTIGATION_THRESHOLD",
    "InvestigationCache",
    "InvestigationHit",
]
