"""Pricing: config-overridable rates + Gemini-3 / embedding coverage.

Regression guard for the cost-telemetry gap where preview / MaaS / custom
models (gemini-3-flash-preview, qwen3-coder-*-maas, gemini-embedding-001) fell
through to cost=0. cost is recomputed from raw token events at read time, so a
priced model -> non-zero /usage cost.
"""
from __future__ import annotations

import pytest

from opsrag.llms import pricing


@pytest.fixture(autouse=True)
def _clear_overrides():
    # Each test starts with no operator overrides; restore after.
    pricing.set_overrides({}, {})
    yield
    pricing.set_overrides({}, {})


def _usd(model, pt, ct):
    return pricing.cost_usd_micros(model, pt, ct) / 1e8


def test_gemini3_flash_priced_and_prefix_suffix_matches():
    # $0.50 in / $3.00 out per 1M (documented public rate).
    assert _usd("gemini-3-flash-preview", 1_000_000, 0) == pytest.approx(0.50)
    assert _usd("gemini-3-flash-preview", 0, 1_000_000) == pytest.approx(3.00)
    # litellm "vertex_ai/" prefix resolves via suffix-match.
    assert _usd("vertex_ai/gemini-3-flash-preview", 1_000_000, 1_000_000) == pytest.approx(3.50)


def test_gemini_embedding_001_priced():
    assert pricing.cost_usd_micros("gemini-embedding-001", 1_000_000, 0) > 0


def test_unpriced_model_is_zero_then_override_prices_it():
    model = "qwen/qwen3-coder-480b-a35b-instruct-maas"
    assert pricing.cost_usd_micros(model, 1_000_000, 1_000_000) == 0  # no built-in
    pricing.set_overrides(
        {"qwen3-coder-480b-a35b-instruct-maas": (int(0.45 * 1e8), int(1.80 * 1e8))}, {}
    )
    # Suffix-match resolves the "qwen/" provider prefix to the override key.
    assert _usd(model, 1_000_000, 1_000_000) == pytest.approx(2.25)  # 0.45 + 1.80


def test_override_wins_over_builtin():
    # A built-in model re-priced via override uses the override value.
    base = _usd("gemini-3-flash-preview", 1_000_000, 0)
    pricing.set_overrides({"gemini-3-flash-preview": (int(9.99 * 1e8), 0)}, {})
    assert _usd("gemini-3-flash-preview", 1_000_000, 0) == pytest.approx(9.99)
    assert base != pytest.approx(9.99)


def test_has_price_covers_token_and_per_call():
    # per-call-priced reranker: no token rate, but has_price is True (no log).
    assert pricing.cost_usd_micros("semantic-ranker-default-004", 1000, 1000) == 0
    assert pricing.has_price("semantic-ranker-default-004") is True
    # token-priced model
    assert pricing.has_price("gemini-3-flash-preview") is True
    # genuinely unknown
    assert pricing.has_price("totally-made-up-model-xyz") is False


def test_usage_page_pricing_honors_token_override():
    # The in-memory /usage breakdown (usage._pricing_for) must consult overrides
    # too -- not just the persisted per-user rollup. Regression guard for the
    # gap where token overrides reached cost_usd_micros but NOT the /usage page.
    from opsrag import usage
    model = "my-maas/custom-llm-xyz"
    assert usage._pricing_for(model) == (0.0, 0.0)  # unknown
    pricing.set_overrides({"custom-llm-xyz": (int(0.50 * 1e8), int(3.00 * 1e8))}, {})
    assert usage._pricing_for(model) == pytest.approx((0.50, 3.00))  # suffix-match + override


def test_per_call_override():
    assert pricing.per_call_cost_micros("my-custom-ranker") == 0
    pricing.set_overrides({}, {"my-custom-ranker": 250_000})
    assert pricing.per_call_cost_micros("my-custom-ranker") == 250_000
    # built-in reranker still resolves when no override shadows it.
    assert pricing.per_call_cost_micros("semantic-ranker-default-004") == 100_000
