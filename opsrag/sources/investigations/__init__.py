"""Investigations source -- turn past tool-path investigations into
RAG-ingestible reference documents.

Distinct from `qa_cache` (verbatim short-circuit) and the live
`investigation_cache` (top-K context for the reasoner). This source
is for the **slow loop**: a daily batch promotes settled, useful past
investigations into the main corpus so future queries can retrieve
them as REFERENCE -- not source-of-truth.

Why "reference, not source-of-truth"? Past investigations encode
specific snapshots of live state. "Kafka cluster has 3 brokers today"
holds for some weeks then is wrong. Indexed under a clearly-tagged
historical-reference shape, with a prominent timestamp, the reasoner
can use them as hints ("we saw similar last month, the answer was X
-- verify if still true with current tools") without confusing them
for canonical knowledge.
"""

from opsrag.sources.investigations.source import (
    DEFAULT_MAX_AGE_DAYS,
    DEFAULT_MIN_AGE_DAYS,
    InvestigationsSource,
)

__all__ = [
    "InvestigationsSource",
    "DEFAULT_MIN_AGE_DAYS",
    "DEFAULT_MAX_AGE_DAYS",
]
