"""M4 (a): the judge auto-accept upper band is config-overridable.

`cfg.qa_judge_upper` (QACacheConfig, default 0.99) populates the module
default via `configure(...)`; `OPSRAG_QA_JUDGE_UPPER` env wins when set.
Raising the upper bound means MORE borderline hits go through the LLM
judge instead of auto-accepting on cosine alone (quality > latency).
"""
from __future__ import annotations

import asyncio
import importlib

import opsrag.qa_cache_judge as judge


def _reload_clean(monkeypatch):
    monkeypatch.delenv("OPSRAG_QA_JUDGE_UPPER", raising=False)
    monkeypatch.delenv("OPSRAG_QA_JUDGE", raising=False)
    return importlib.reload(judge)


class _FakeLLM:
    """Records whether the judge actually called the model, and answers
    DIFFERENT so we can tell a judge-run apart from an auto-accept."""

    def __init__(self) -> None:
        self.called = False

    async def generate(self, **kwargs):  # noqa: ANN003
        self.called = True

        class _R:
            content = "DIFFERENT"

        return _R()


def test_default_upper_is_099(monkeypatch):
    m = _reload_clean(monkeypatch)
    assert m._get_judge_upper() == 0.99


def test_configure_sets_upper(monkeypatch):
    m = _reload_clean(monkeypatch)
    m.configure(qa_judge_upper=0.97)
    assert m._get_judge_upper() == 0.97
    # Back-compat constant tracks the configured value.
    assert m.JUDGE_UPPER == 0.97


def test_env_overrides_config(monkeypatch):
    m = _reload_clean(monkeypatch)
    m.configure(qa_judge_upper=0.97)
    monkeypatch.setenv("OPSRAG_QA_JUDGE_UPPER", "0.995")
    assert m._get_judge_upper() == 0.995


def test_judge_runs_in_widened_band(monkeypatch):
    """At cosine 0.98 the default 0.99 band keeps the judge IN PLAY:
    a borderline match that would auto-accept under the old 0.97 ceiling
    now gets the LLM judge and is correctly rejected as DIFFERENT."""
    m = _reload_clean(monkeypatch)
    llm = _FakeLLM()
    ok = asyncio.run(m.judge_match(
        current_query="SRE goals 2025",
        cached_question="SRE goals 2026",
        cosine=0.98,
        llm=llm,
    ))
    assert llm.called is True       # judge ran (0.98 < 0.99 upper)
    assert ok is False             # DIFFERENT -> force miss


def test_above_upper_skips_judge(monkeypatch):
    """At cosine == upper band the judge is skipped (auto-accept)."""
    m = _reload_clean(monkeypatch)
    m.configure(qa_judge_upper=0.97)
    llm = _FakeLLM()
    ok = asyncio.run(m.judge_match(
        current_query="rotate the ssl cert",
        cached_question="rotate the production ssl cert",
        cosine=0.98,           # >= 0.97 configured upper
        llm=llm,
    ))
    assert llm.called is False      # auto-accepted, no judge call
    assert ok is True


def test_lowering_upper_auto_accepts_what_default_would_judge(monkeypatch):
    """Same 0.98 hit: with the upper lowered to 0.97 it auto-accepts;
    pins the override actually moves the band."""
    m = _reload_clean(monkeypatch)
    m.configure(qa_judge_upper=0.97)
    llm = _FakeLLM()
    ok = asyncio.run(m.judge_match(
        current_query="SRE goals 2025",
        cached_question="SRE goals 2026",
        cosine=0.98,
        llm=llm,
    ))
    assert llm.called is False
    assert ok is True
