"""Parity tests for the unified pricing tables.

`opsrag/llms/pricing.py` is the single source of truth:

  - `_RATES` (micro-cents per 1M tokens) feeds `cost_usd_micros` -- the
    persisted/DB cost path used by `usage_persistence.py`.
  - `_PER_CALL_RATES` (micro-cents per call) feeds `per_call_cost_micros`
    -- the reranker path (those record zero tokens).

`opsrag/usage.py` DERIVES its USD-per-1M-token rates from the same tables
(`_pricing_for` / `_per_call_for`). These tests guard that:

  1. Every default model id used in the app resolves to a NON-ZERO cost.
  2. The reranker per-call cost is representable (> 0) for every default
     reranker id.
  3. usage.py and pricing.py never diverge -- `_pricing_for` exactly
     mirrors `cost_usd_micros`, and `_per_call_for` mirrors
     `per_call_cost_micros`, for every priced model id.
"""
from __future__ import annotations

import pytest

from opsrag.llms.pricing import (
    _PER_CALL_RATES,
    _RATES,
    cost_usd_micros,
    per_call_cost_micros,
)
from opsrag.usage import _MICROS_PER_USD, _per_call_for, _pricing_for

# Default model ids the app ships with (see config.py / factory.py /
# model_bundles.py / llms/openai.py / embedders/*). Each MUST price to a
# non-zero cost or telemetry silently undercounts.
DEFAULT_TOKEN_MODELS = [
    "gpt-4o",                                    # llms/openai.py default
    "text-embedding-3-large",                    # config.py / embedders default embedder
    "gemini-2.5-flash",                          # gemini default (fast lane)
    "gemini-2.5-pro",                            # gemini default (reason lane)
    "anthropic.claude-opus-4-8",                 # bedrock default
    "anthropic.claude-sonnet-4-6",               # bedrock reason default
    "cohere.embed-v4:0",                         # bedrock embedder default
]

DEFAULT_RERANKER_MODELS = [
    "cohere.rerank-v3-5:0",                      # factory.py / bedrock reranker default
    "semantic-ranker-default-004",               # factory.py / vertex reranker default
    "amazon.rerank-v1:0",
    "semantic-ranker-default-003",
]

# Region-prefixed inference profiles the telemetry actually sees -- these
# must resolve via suffix-match to the same base-id cost.
REGION_PREFIXED = [
    "us.anthropic.claude-opus-4-8",
    "us.cohere.embed-v4:0",
    "us.cohere.rerank-v3-5:0",
]


@pytest.mark.parametrize("model", DEFAULT_TOKEN_MODELS)
def test_default_token_model_has_nonzero_cost(model: str) -> None:
    """A 1k-prompt / 1k-completion call must cost > 0 micro-cents."""
    micros = cost_usd_micros(model, 1_000, 1_000)
    assert micros > 0, f"{model} priced at $0 via cost_usd_micros"


@pytest.mark.parametrize("model", DEFAULT_RERANKER_MODELS)
def test_default_reranker_per_call_cost_representable(model: str) -> None:
    """Rerankers record zero tokens; the per-call table is their ONLY
    cost and must be > 0 and exactly representable in micro-cents."""
    micros = per_call_cost_micros(model)
    assert micros > 0, f"{model} per-call cost is 0"
    # Representable: an integer count of micro-cents (no fractional loss).
    assert isinstance(micros, int)
    # Token path must NOT also charge for a reranker (no double-count).
    assert cost_usd_micros(model, 0, 0) == 0


@pytest.mark.parametrize("model", REGION_PREFIXED)
def test_region_prefixed_profiles_resolve(model: str) -> None:
    """`us.`/`eu.`/`apac.` inference profiles resolve to the base id's cost."""
    if model.endswith("rerank-v3-5:0"):
        assert per_call_cost_micros(model) > 0
    else:
        assert cost_usd_micros(model, 1_000, 1_000) > 0


@pytest.mark.parametrize("model", sorted(_RATES.keys()))
def test_usage_pricing_matches_pricing_token(model: str) -> None:
    """No divergence: usage.py `_pricing_for` (USD/1M) is exactly the
    pricing.py `_RATES` value converted micro-cents/1M -> USD/1M."""
    in_usd, out_usd = _pricing_for(model)
    exp_in, exp_out = _RATES[model]
    assert in_usd == exp_in / _MICROS_PER_USD
    assert out_usd == exp_out / _MICROS_PER_USD


@pytest.mark.parametrize("model", sorted(_PER_CALL_RATES.keys()))
def test_usage_per_call_matches_pricing(model: str) -> None:
    """No divergence: usage.py `_per_call_for` (USD/call) is exactly the
    pricing.py per-call rate converted micro-cents -> USD."""
    assert _per_call_for(model) == _PER_CALL_RATES[model] / _MICROS_PER_USD


def test_no_token_cost_divergence_for_large_call() -> None:
    """End-to-end parity on a realistic call: the USD the in-memory
    summary computes equals the DB micro-cent cost / 1e8, for every
    token-priced model."""
    pt, ct = 250_000, 40_000
    for model in _RATES:
        in_usd, out_usd = _pricing_for(model)
        usage_usd = pt * in_usd / 1_000_000 + ct * out_usd / 1_000_000
        # cost_usd_micros truncates per-term (integer math); the derived
        # float path matches to well within display precision (6 dp).
        db_usd = cost_usd_micros(model, pt, ct) / _MICROS_PER_USD
        assert usage_usd == pytest.approx(db_usd, abs=1e-6), model


def test_gemini_rates_reconciled() -> None:
    """The prior 2x Flash divergence is gone: pricing.py now carries the
    fuller public rates (Flash 0.15/0.60, Pro 1.25/10.00 per 1M)."""
    flash_in, flash_out = _pricing_for("gemini-2.5-flash")
    assert (flash_in, flash_out) == (0.15, 0.60)
    pro_in, pro_out = _pricing_for("gemini-2.5-pro")
    assert (pro_in, pro_out) == (1.25, 10.0)


def test_previously_missing_models_now_priced() -> None:
    """The keys that were absent from `_RATES` (DB charged $0) are present."""
    for model in (
        "gpt-4o",
        "gpt-4o-mini",
        "text-embedding-3-large",
        "text-embedding-3-small",
        "text-embedding-005",
    ):
        assert cost_usd_micros(model, 1_000, 0) > 0, model
