"""Ranking metrics -- Precision@K, Recall@K, MRR.

Closes the gap left by SourceRecallMetric: SourceRecall scores a hit even
if the doc is at position 50 of 50. A regression where the right doc moves
from rank 1 to rank 10 passes the existing eval but kills perceived quality
(the LLM rarely reads past the top few sources). These three metrics catch
that class of regression.

All three are pure set / position arithmetic over the same
`expected_sources` / `retrieved_sources` metadata SourceRecall reads, so
they cost zero LLM calls and add no sigma noise. They share `match_path` with
SourceRecall -- chunker-stability rules (canonical form, page-id stem
fallback) live in opsrag.eval.loaders.

Conventions
-----------
- `retrieved_sources` is treated as rank-ordered (position 0 = top hit).
  OpsRAG's `/query` returns sources in reranker order, so this holds.
- P@K denominator is min(K, len(retrieved)) -- i.e., we don't punish a
  golden that only returned 3 sources by padding the denominator to K.
- R@K is `|expected intersect retrieved[:K]| / |expected|`. A golden with NO
  expected AND NO acceptable sources is marked `skipped` (the score is
  meaningless without a relevant set); the report excludes skipped cases from
  the aggregate mean instead of counting a free 1.0.
- MRR is over the *first* retrieved source that matches *any* expected
  source. Standard convention for multi-relevant retrieval evaluation;
  ranges 1.0 (top hit relevant) down to 0.0 (no hit anywhere in the list).
"""
from __future__ import annotations

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

from opsrag.eval.loaders import match_path


def _meta_lists(
    test_case: LLMTestCase,
) -> tuple[list[str], list[str], list[str]]:
    """Pull (expected, acceptable, retrieved) out of test-case metadata.

    Ranking metrics treat `expected_sources union acceptable_sources` as the
    "relevant set" for precision and MRR -- surfacing any acceptable
    alternative in top-K counts as a relevant hit. Recall keeps the
    required/required-or-acceptable split (see Recall@K.measure).
    """
    meta = test_case.metadata or {}
    expected = list(meta.get("expected_sources") or [])
    acceptable = list(meta.get("acceptable_sources") or [])
    retrieved = meta.get("retrieved_sources")
    if retrieved is None:
        retrieved = test_case.retrieval_context or []
    return expected, acceptable, list(retrieved)


def _is_relevant(retrieved: str, expected: list[str], acceptable: list[str]) -> bool:
    """A retrieved doc is 'relevant' if it matches any expected OR any
    acceptable source. Used by Precision@K and MRR -- both care about
    'did we surface a valid grounding', not which specific one."""
    for e in expected:
        if match_path(e, retrieved):
            return True
    for a in acceptable:
        if match_path(a, retrieved):
            return True
    return False


def _expected_hits_in_topk(
    expected: list[str], retrieved: list[str], k: int
) -> list[str]:
    """Subset of `expected` that has a match in `retrieved[:k]`."""
    top = retrieved[:k]
    return [e for e in expected if any(match_path(e, r) for r in top)]


def _any_acceptable_in_topk(
    acceptable: list[str], retrieved: list[str], k: int
) -> bool:
    top = retrieved[:k]
    return any(any(match_path(a, r) for r in top) for a in acceptable)


class RankPrecisionAtKMetric(BaseMetric):
    """Fraction of top-K retrieved sources matching ANY relevant source.

    P@K = |{r in retrieved[:K] : r matches some expected OR acceptable source}| / min(K, |retrieved|)

    "Relevant" expands to the union of `expected_sources` (AND-required)
    and `acceptable_sources` (OR-alternatives). Vacuously 1.0 when
    `retrieved` is empty -- SourceRecall flags the missing-retrieval case
    separately.
    """

    def __init__(self, k: int = 5, threshold: float = 0.4):
        self.k = k
        self.threshold = threshold
        self.score: float = 0.0
        self.success: bool = False
        self.reason: str = ""
        self.error: str | None = None
        # True when the golden defines no relevant set (no expected AND no
        # acceptable sources). The score is then meaningless -- aggregation
        # excludes it instead of letting it inflate/deflate the mean.
        self.skipped: bool = False

    def measure(self, test_case: LLMTestCase) -> float:
        expected, acceptable, retrieved = _meta_lists(test_case)
        if not expected and not acceptable:
            self.skipped = True
            self.score = 1.0
            self.success = True
            self.reason = "skipped: golden has no expected/acceptable sources"
            return self.score
        if not retrieved:
            self.score = 1.0
            self.success = True
            self.reason = "no retrieved sources"
            return self.score
        top = retrieved[: self.k]
        hits = [r for r in top if _is_relevant(r, expected, acceptable)]
        self.score = len(hits) / len(top)
        self.success = self.score >= self.threshold
        relevant_set_size = len(expected) + len(acceptable)
        self.reason = (
            f"{len(hits)}/{len(top)} top-{self.k} sources matched a relevant "
            f"source (relevant_set={relevant_set_size})"
        )
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self) -> str:
        return f"Precision@{self.k}"


class RankRecallAtKMetric(BaseMetric):
    """Fraction of required sources present in the top-K retrieved.

    Mirrors `SourceRecallMetric`'s AND/OR semantics, restricted to top-K:
    - `expected_sources` non-empty -> AND over expected:
      R@K = |{e in expected : found in retrieved[:K]}| / |expected|
    - `expected_sources` empty AND `acceptable_sources` non-empty -> OR:
      R@K = 1.0 if any acceptable source in retrieved[:K] else 0.0
    - Both empty -> 1.0 (vacuous)
    """

    def __init__(self, k: int = 10, threshold: float = 0.7):
        self.k = k
        self.threshold = threshold
        self.score: float = 0.0
        self.success: bool = False
        self.reason: str = ""
        self.error: str | None = None
        self.skipped: bool = False

    def measure(self, test_case: LLMTestCase) -> float:
        expected, acceptable, retrieved = _meta_lists(test_case)
        if expected:
            hits = _expected_hits_in_topk(expected, retrieved, self.k)
            self.score = len(hits) / len(expected)
            self.success = self.score >= self.threshold
            missing = sorted(set(expected) - set(hits))
            if missing:
                self.reason = (
                    f"{len(hits)}/{len(expected)} expected sources in top-{self.k}; "
                    f"missing: {missing[:3]}{'...' if len(missing) > 3 else ''}"
                )
            else:
                self.reason = (
                    f"all {len(expected)} expected sources in top-{self.k}"
                )
            return self.score
        if acceptable:
            found = _any_acceptable_in_topk(acceptable, retrieved, self.k)
            self.score = 1.0 if found else 0.0
            self.success = found
            self.reason = (
                f"any-acceptable-in-top-{self.k}: "
                f"{'matched' if found else 'no match'} "
                f"(OR over {len(acceptable)} alternatives)"
            )
            return self.score
        self.skipped = True
        self.score = 1.0
        self.success = True
        self.reason = "skipped: golden has no expected/acceptable sources"
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self) -> str:
        return f"Recall@{self.k}"


class MRRMetric(BaseMetric):
    """Reciprocal rank of the first retrieved source matching any relevant one.

    MRR = 1 / rank_of_first_hit (1-indexed); 0.0 when no hit exists.

    "Relevant" = `expected_sources union acceptable_sources`. For multi-source
    goldens this is the standard "got something good near the top" measure
    -- finds the first relevant doc and rewards surfacing it close to
    position 1. Resilient to multi-grounding goldens because any acceptable
    alternative counts.
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.score: float = 0.0
        self.success: bool = False
        self.reason: str = ""
        self.error: str | None = None
        self.skipped: bool = False

    def measure(self, test_case: LLMTestCase) -> float:
        expected, acceptable, retrieved = _meta_lists(test_case)
        if not expected and not acceptable:
            self.skipped = True
            self.score = 1.0
            self.success = True
            self.reason = "skipped: golden has no expected/acceptable sources"
            return self.score
        for i, r in enumerate(retrieved, start=1):
            if _is_relevant(r, expected, acceptable):
                self.score = 1.0 / i
                self.success = self.score >= self.threshold
                self.reason = f"first relevant match at rank {i}"
                return self.score
        self.score = 0.0
        self.success = False
        self.reason = "no relevant source found in retrieved list"
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self) -> str:
        return "MRR"
