"""MustContainMetric and MustNotContainMetric.

Substring assertions on the actual answer. Case-insensitive by default.
No LLM judge -- fast, deterministic, free.

MustContainMetric: every required substring must be present (AND).
MustNotContainMetric: no forbidden substring may be present.

Both metrics succeed vacuously when their substring list is empty.
"""
from __future__ import annotations

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase


class MustContainMetric(BaseMetric):
    """All required substrings present in actual_output (case-insensitive)."""

    def __init__(self, threshold: float = 1.0, case_sensitive: bool = False):
        self.threshold = threshold
        self.case_sensitive = case_sensitive
        self.score: float = 0.0
        self.success: bool = False
        self.reason: str = ""
        self.error: str | None = None

    def measure(self, test_case: LLMTestCase) -> float:
        meta = test_case.metadata or {}
        required = list(meta.get("must_contain") or [])
        text = test_case.actual_output or ""
        haystack = text if self.case_sensitive else text.lower()

        if not required:
            self.score = 1.0
            self.success = True
            self.reason = "no must_contain substrings configured"
            return self.score

        missing: list[str] = []
        for sub in required:
            needle = sub if self.case_sensitive else sub.lower()
            if needle not in haystack:
                missing.append(sub)

        present = len(required) - len(missing)
        self.score = present / len(required)
        self.success = self.score >= self.threshold
        if missing:
            self.reason = f"{present}/{len(required)} present; missing: {missing}"
        else:
            self.reason = f"all {len(required)} substrings present"
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self) -> str:
        return "MustContain"


class MustNotContainMetric(BaseMetric):
    """No forbidden substring present in actual_output (hallucination guard)."""

    def __init__(self, threshold: float = 1.0, case_sensitive: bool = False):
        self.threshold = threshold
        self.case_sensitive = case_sensitive
        self.score: float = 0.0
        self.success: bool = False
        self.reason: str = ""
        self.error: str | None = None

    def measure(self, test_case: LLMTestCase) -> float:
        meta = test_case.metadata or {}
        forbidden = list(meta.get("must_not_contain") or [])
        text = test_case.actual_output or ""
        haystack = text if self.case_sensitive else text.lower()

        if not forbidden:
            self.score = 1.0
            self.success = True
            self.reason = "no must_not_contain substrings configured"
            return self.score

        hits: list[str] = []
        for sub in forbidden:
            needle = sub if self.case_sensitive else sub.lower()
            if needle in haystack:
                hits.append(sub)

        # Score: 1.0 if no forbidden hits, 0.0 if any present.
        # Linearly interpolate across multiple -- useful when many forbidden
        # substrings to track partial regression.
        self.score = (len(forbidden) - len(hits)) / len(forbidden)
        self.success = self.score >= self.threshold
        if hits:
            self.reason = f"FORBIDDEN substring(s) present: {hits}"
        else:
            self.reason = f"none of {len(forbidden)} forbidden substrings present"
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self) -> str:
        return "MustNotContain"
