"""OpsRAG-specific metrics on top of DeepEval primitives.

DeepEval ships ContextualRecall, ContextualPrecision, Faithfulness,
AnswerRelevancy -- useful but not perfectly aligned with our DevOps
golden-set design (which keys on file paths, not free-text expected
output). The metrics here fill the gap:

- SourceRecallMetric: % of expected_sources present in the retrieved
  source list. Doesn't need a judge -- pure set arithmetic.
- MustContainMetric: required substrings are in the answer.
- MustNotContainMetric: forbidden substrings are absent (hallucination guard).

Faithfulness goes through a custom GEval wrapper using VertexGeminiJudge
(see metrics/faithfulness.py).
"""
from opsrag.eval.metrics.faithfulness import FaithfulnessJudgeMetric
from opsrag.eval.metrics.must_contain import MustContainMetric, MustNotContainMetric
from opsrag.eval.metrics.ranking import (
    MRRMetric,
    NDCGAtKMetric,
    RankPrecisionAtKMetric,
    RankRecallAtKMetric,
    RetrievalRestraintMetric,
)
from opsrag.eval.metrics.source_recall import SourceRecallMetric

__all__ = [
    "SourceRecallMetric",
    "MustContainMetric",
    "MustNotContainMetric",
    "FaithfulnessJudgeMetric",
    "RankPrecisionAtKMetric",
    "RankRecallAtKMetric",
    "NDCGAtKMetric",
    "RetrievalRestraintMetric",
    "MRRMetric",
]
