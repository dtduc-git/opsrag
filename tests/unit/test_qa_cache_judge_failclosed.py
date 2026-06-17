"""R3 (per-category lower band) + R14 (fail-closed on judge outage).

R3: `judge_match` takes a `min_score` (the per-category cosine floor that
`lookup` accepted the hit at). The effective judge lower band resolves to
`max(min_score, JUDGE_LOWER)` (else JUDGE_LOWER when min_score is None), so
high-floor categories (e.g. MIXED 0.96) anchor the judge band at their
floor instead of the hardcoded 0.93, and the band never dips below the
cosine threshold the caller already enforced.

R14: when the borderline-band judge LLM errors, default fail-CLOSED
(return False -> force a fresh corpus query) instead of serving a
wrong-but-close cached answer. `qa_judge_fail_open` (config / env) flips
it back to fail-open (return True).
"""
from __future__ import annotations

import asyncio
import importlib

import opsrag.qa_cache_judge as judge


def _reload_clean(monkeypatch):
    monkeypatch.delenv("OPSRAG_QA_JUDGE_UPPER", raising=False)
    monkeypatch.delenv("OPSRAG_QA_JUDGE", raising=False)
    monkeypatch.delenv("OPSRAG_QA_JUDGE_FAIL_OPEN", raising=False)
    return importlib.reload(judge)


class _DiffLLM:
    """Answers DIFFERENT so a judge-run is distinguishable from auto-accept."""

    def __init__(self) -> None:
        self.called = False

    async def generate(self, **kwargs):  # noqa: ANN003
        self.called = True

        class _R:
            content = "DIFFERENT"

        return _R()


class _ExplodingLLM:
    """Raises inside generate() to simulate a judge/Vertex outage."""

    def __init__(self) -> None:
        self.called = False

    async def generate(self, **kwargs):  # noqa: ANN003
        self.called = True
        raise RuntimeError("vertex flash unavailable")


# --------------------------------------------------------------------------
# R3 -- per-category lower band
# --------------------------------------------------------------------------

def test_forensic_hit_above_093_is_judged(monkeypatch):
    """A FORENSIC hit (min_score=0.92) at cosine 0.935 sits in the resolved
    band [max(0.92, 0.93)=0.93, 0.99) and IS judged -> DIFFERENT rejects."""
    m = _reload_clean(monkeypatch)
    llm = _DiffLLM()
    ok = asyncio.run(m.judge_match(
        current_query="why did pipeline 1531171 fail",
        cached_question="why did pipeline 1531172 fail",
        cosine=0.935,
        llm=llm,
        min_score=0.92,
    ))
    assert llm.called is True       # 0.935 >= max(0.92, 0.93) = 0.93
    assert ok is False             # DIFFERENT -> force miss


def test_below_resolved_lower_defends_to_true(monkeypatch):
    """A hit at 0.925 is below the resolved lower band max(0.92, 0.93)=0.93,
    so it is defensively allowed without judging (caller is source of truth
    at the lower bound)."""
    m = _reload_clean(monkeypatch)
    llm = _DiffLLM()
    ok = asyncio.run(m.judge_match(
        current_query="why did pipeline 1531171 fail",
        cached_question="why did pipeline 1531172 fail",
        cosine=0.925,
        llm=llm,
        min_score=0.92,
    ))
    assert llm.called is False
    assert ok is True


def test_min_score_above_093_raises_lower_band(monkeypatch):
    """min_score above JUDGE_LOWER (e.g. MIXED 0.96) raises the lower band:
    a 0.95 hit is now BELOW the effective floor and is not judged."""
    m = _reload_clean(monkeypatch)
    llm = _DiffLLM()
    ok = asyncio.run(m.judge_match(
        current_query="rotate the ssl cert",
        cached_question="rotate the prod ssl cert",
        cosine=0.95,
        llm=llm,
        min_score=0.96,
    ))
    assert llm.called is False      # 0.95 < max(0.96, 0.93) = 0.96
    assert ok is True


# --------------------------------------------------------------------------
# R14 -- fail-closed (default) / fail-open (opt-in) on judge outage
# --------------------------------------------------------------------------

def test_judge_exception_default_fail_closed(monkeypatch):
    """LLM error in-band (cosine 0.95) defaults to fail-CLOSED -> False."""
    m = _reload_clean(monkeypatch)
    llm = _ExplodingLLM()
    ok = asyncio.run(m.judge_match(
        current_query="SRE goals 2025",
        cached_question="SRE goals 2026",
        cosine=0.95,
        llm=llm,
    ))
    assert llm.called is True
    assert ok is False              # fail-closed: force a fresh corpus query
    assert m.stats()["errors"] == 1


def test_judge_exception_fail_open_via_configure(monkeypatch):
    """configure(qa_judge_fail_open=True) flips the outage path to allow."""
    m = _reload_clean(monkeypatch)
    m.configure(qa_judge_fail_open=True)
    llm = _ExplodingLLM()
    ok = asyncio.run(m.judge_match(
        current_query="SRE goals 2025",
        cached_question="SRE goals 2026",
        cosine=0.95,
        llm=llm,
    ))
    assert llm.called is True
    assert ok is True               # fail-open: serve the cached candidate
    assert m.stats()["errors"] == 1


def test_judge_exception_fail_open_via_env(monkeypatch):
    """OPSRAG_QA_JUDGE_FAIL_OPEN env wins over the config default."""
    m = _reload_clean(monkeypatch)
    monkeypatch.setenv("OPSRAG_QA_JUDGE_FAIL_OPEN", "1")
    llm = _ExplodingLLM()
    ok = asyncio.run(m.judge_match(
        current_query="SRE goals 2025",
        cached_question="SRE goals 2026",
        cosine=0.95,
        llm=llm,
    ))
    assert ok is True
