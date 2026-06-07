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

import math

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
            # We already returned for the no-relevant-set case above, so a
            # relevant set DOES exist here -- retrieving nothing is a total
            # precision miss (0.0), not a vacuous pass. Scoring 1.0 would mask a
            # retrieval-zeroing regression on the precision axis.
            self.score = 0.0
            self.success = False
            self.reason = "no retrieved sources but golden expects some"
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


class RetrievalRestraintMetric(BaseMetric):
    """Retrieval-side assertion for negative goldens.

    The negative set previously only checked the ANSWER (must_not_contain) -- it
    never asserted anything about what the retriever SURFACED. But the failure
    mode for a "doesn't exist in the corpus" query is the retriever dragging in
    loosely-related chunks that then tempt the generator to hallucinate; the
    weak-retrieval gate is supposed to trim those to (near) nothing. This metric
    asserts `len(retrieved) <= max_retrieved_sources` when a golden sets that
    cap, catching a regression that disables the gate even when the answer text
    happens to still look fine.

    Skipped (no opinion) when the golden doesn't set `max_retrieved_sources`.
    """

    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold
        self.score: float = 0.0
        self.success: bool = False
        self.reason: str = ""
        self.error: str | None = None
        self.skipped: bool = False

    def measure(self, test_case: LLMTestCase) -> float:
        meta = test_case.metadata or {}
        cap = meta.get("max_retrieved_sources")
        if cap is None:
            self.skipped = True
            self.score = 1.0
            self.success = True
            self.reason = "skipped: golden sets no max_retrieved_sources"
            return self.score
        _, _, retrieved = _meta_lists(test_case)
        n = len(retrieved)
        self.success = n <= int(cap)
        self.score = 1.0 if self.success else 0.0
        self.reason = (
            f"retrieved {n} source(s); cap is {cap} "
            f"({'within' if self.success else 'OVER'} restraint budget)"
        )
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self) -> str:
        return "RetrievalRestraint"


class NDCGAtKMetric(BaseMetric):
    """Normalized Discounted Cumulative Gain over the top-K retrieved sources.

    Where Recall@K asks "did the relevant docs appear in top-K?" and MRR only
    looks at the FIRST hit, NDCG@K is position-aware over ALL relevant docs in
    the window: a relevant doc at rank 1 is worth more than the same doc at
    rank 5, and surfacing TWO relevant docs beats one. This catches re-ranking
    regressions that Recall@K (a set test) and MRR (first-hit only) both miss.

    Binary relevance gain (1 if the retrieved doc matches any expected OR
    acceptable source, else 0). DCG = sum gain_i / log2(i+1), 1-indexed.
    IDCG = the DCG of the ideal ordering (all relevant docs packed at the top,
    capped at K and at the size of the relevant set). NDCG = DCG / IDCG.
    """

    def __init__(self, k: int = 5, threshold: float = 0.5):
        self.k = k
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
        top = retrieved[: self.k]
        # Credit each relevant document at most ONCE, and treat `acceptable` as
        # an OR-set that contributes a single relevant slot. Counting every
        # acceptable match as its own gain (the old behaviour) sized IDCG as if
        # all of [A,B,C] should appear, so a perfect retrieval (A at rank 1)
        # scored dcg=1.0 / idcg=2.13 = 0.47 -- the metric capped below 1.0 on
        # CORRECT behaviour. Mirror the relevant-set accounting in DCG and IDCG.
        expected_hit: set[str] = set()
        acceptable_used = False
        gains: list[int] = []
        for r in top:
            matched_e = next(
                (e for e in expected if e not in expected_hit and match_path(e, r)),
                None,
            )
            if matched_e is not None:
                expected_hit.add(matched_e)
                gains.append(1)
            elif not acceptable_used and any(match_path(a, r) for a in acceptable):
                acceptable_used = True
                gains.append(1)
            else:
                gains.append(0)
        dcg = sum(g / math.log2(i + 1) for i, g in enumerate(gains, start=1))
        # Relevant universe: each expected source + (the whole acceptable OR-set
        # counts once). Cap the ideal run at K so NDCG stays in [0, 1]; DCG can
        # never exceed this IDCG because each relevant doc is credited once.
        relevant_n = len(expected) + (1 if acceptable else 0)
        ideal_n = min(self.k, relevant_n)
        idcg = sum(1 / math.log2(i + 1) for i in range(1, ideal_n + 1))
        self.score = (dcg / idcg) if idcg > 0 else 0.0
        self.success = self.score >= self.threshold
        self.reason = (
            f"NDCG@{self.k}={self.score:.2f} ({sum(gains)} relevant in top-{self.k})"
        )
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self) -> str:
        return f"NDCG@{self.k}"


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
