"""SourceRecallMetric -- fraction of required sources found in retrieved.

Pure set arithmetic, no LLM judge. Reads `expected_sources` and
`acceptable_sources` from the test case metadata (set by
`loaders.to_llm_test_case`). The "retrieved sources" come from
`LLMTestCase.retrieval_context` (or the explicit `retrieved_sources`
metadata key when provided).

Matching delegated to `loaders.match_path` -- see that module for the
canonical-form, suffix-on-boundary, and stem-only fallback rules.

Semantics
---------
- `expected_sources` non-empty -> **AND** semantics.
  Score = |expected intersect retrieved| / |expected| in [0, 1].
- `expected_sources` empty AND `acceptable_sources` non-empty -> **OR** semantics.
  Score = 1.0 if any acceptable source is in retrieved, else 0.0.
- Both empty -> vacuously 1.0 (no requirement to satisfy).

Use `acceptable_sources` when a question has multiple valid groundings
(e.g., a literal YAML file AND a prose doc that describes it). See
`golden/README.md` for authoring guidance.
"""
from __future__ import annotations

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

from opsrag.eval.loaders import match_path


class SourceRecallMetric(BaseMetric):
    """Fraction of expected source paths that appear in the retrieval context."""

    def __init__(self, threshold: float = 0.8, strict_mode: bool = False):
        self.threshold = threshold
        self.strict_mode = strict_mode
        self.score: float = 0.0
        self.success: bool = False
        self.reason: str = ""
        self.error: str | None = None
        # True when the golden defines neither expected nor acceptable sources.
        # The 1.0 it returns is vacuous; the report excludes skipped cases from
        # the (gated) aggregate so a category of unlabeled goldens can't pass
        # the regression gate on free 1.0s. Mirrors the ranking metrics.
        self.skipped: bool = False

    def measure(self, test_case: LLMTestCase) -> float:
        meta = test_case.metadata or {}
        expected = list(meta.get("expected_sources") or [])
        acceptable = list(meta.get("acceptable_sources") or [])
        self.skipped = not expected and not acceptable
        # Use explicit retrieved_sources from metadata when set (raw paths);
        # fall back to retrieval_context which may be content-formatted strings.
        retrieved_list = meta.get("retrieved_sources")
        if retrieved_list is None:
            retrieved_list = test_case.retrieval_context or []

        # AND-semantics over expected_sources when present.
        if expected:
            hits: list[str] = []
            for e in expected:
                if any(match_path(e, r) for r in retrieved_list):
                    hits.append(e)
            self.score = len(hits) / len(expected)
            self.success = self.score >= self.threshold
            missing = sorted(set(expected) - set(hits))
            if missing:
                self.reason = (
                    f"{len(hits)}/{len(expected)} expected sources retrieved; "
                    f"missing: {missing[:3]}{'...' if len(missing) > 3 else ''}"
                )
            else:
                self.reason = f"all {len(expected)} expected sources retrieved"
            return self.score

        # OR-semantics over acceptable_sources when expected is empty.
        if acceptable:
            matched = [a for a in acceptable if any(match_path(a, r) for r in retrieved_list)]
            self.score = 1.0 if matched else 0.0
            self.success = bool(matched)
            if matched:
                self.reason = (
                    f"matched {len(matched)}/{len(acceptable)} acceptable source(s) "
                    f"(OR-semantics): {matched[:2]}{'...' if len(matched) > 2 else ''}"
                )
            else:
                self.reason = (
                    f"no acceptable source matched (OR over {len(acceptable)} "
                    f"alternatives)"
                )
            return self.score

        # Both empty -- vacuously satisfied.
        self.score = 1.0
        self.success = True
        self.reason = "no expected_sources or acceptable_sources to match"
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self) -> str:
        return "SourceRecall"
