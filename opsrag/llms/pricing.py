"""M2 -- token pricing in micro-cents.

A *micro-cent* = 1 / 100_000_000 of a US dollar. We store costs as
integers in this unit so the per-event arithmetic never touches
floats -- Postgres can sum them losslessly, and `decimal`-vs-`float`
rounding surprises stop existing.

Conversion to display::

    micros / 100_000_000  ->  USD

The constants below are encoded as "micro-cents per **1M tokens**"
because that's the granularity public price sheets use::

    GEMINI_2_0_FLASH_INPUT = 7_500_000
        = $0.075 / 1M tokens
        = 7_500_000 micro-cents / 1M tokens

To get the cost of N tokens at rate R (micro-cents / 1M tokens)::

    micros = N * R // 1_000_000

Integer-only math, intentional. The truncation error on a single 1k-token
call is at most 1 micro-cent = $0.00000001 -- well below display precision.

Public pricing as of 2026-06-16. This is the SINGLE source of truth for
cost: opsrag/usage.py derives its in-memory `/usage` rates from `_RATES`
and `_PER_CALL_RATES` here (no second table), so the persisted DB cost
path and the live summary can never diverge. Update when a provider
publishes a new sheet; the constants are at module scope so a single edit
propagates everywhere.
"""
from __future__ import annotations

import logging

_log = logging.getLogger("opsrag.llms.pricing")

# --- Gemini 2.5 (Vertex AI) -------------------------------------------------
# Flash: $0.15 input / $0.60 output per 1M tokens.
# Pro:   $1.25 input / $10.00 output per 1M tokens (authoritative public sheet;
# this reconciles the prior 2x divergence with usage.py, which had the fuller
# public Flash 0.15/0.60 and Pro 1.25/10.00 rates).
GEMINI_2_5_FLASH_INPUT = 15_000_000
GEMINI_2_5_FLASH_OUTPUT = 60_000_000
GEMINI_2_5_PRO_INPUT = 125_000_000
GEMINI_2_5_PRO_OUTPUT = 1_000_000_000

# Gemini 2.0 Flash is cheaper than 2.5 Flash: $0.075 input / $0.30 output.
GEMINI_2_0_FLASH_INPUT = 7_500_000
GEMINI_2_0_FLASH_OUTPUT = 30_000_000

# --- Claude (via Anthropic API or AnthropicVertex) --------------------------
# Sonnet 4: ~$3 input / $15 output per 1M tokens (rounded from public sheet).
CLAUDE_SONNET_4_INPUT = 300_000_000
CLAUDE_SONNET_4_OUTPUT = 1_500_000_000

# Opus 4: $15 / $75 per 1M tokens.
CLAUDE_OPUS_4_INPUT = 1_500_000_000
CLAUDE_OPUS_4_OUTPUT = 7_500_000_000

# Haiku 4: $0.80 / $4.00 per 1M tokens.
CLAUDE_HAIKU_4_INPUT = 80_000_000
CLAUDE_HAIKU_4_OUTPUT = 400_000_000

# Haiku 4.5: $1.00 / $5.00 per 1M tokens.
CLAUDE_HAIKU_4_5_INPUT = 100_000_000
CLAUDE_HAIKU_4_5_OUTPUT = 500_000_000

# Titan Text Embeddings v2 (Bedrock): $0.02 / 1M input tokens, no output.
TITAN_EMBED_V2_INPUT = 2_000_000
TITAN_EMBED_V2_OUTPUT = 0

# Cohere Embed v4 (Bedrock): $0.12 / 1M input tokens, no output.
COHERE_EMBED_V4_INPUT = 12_000_000
COHERE_EMBED_V4_OUTPUT = 0

# --- OpenAI GPT-4o family ---------------------------------------------------
# GPT-4o:      $2.50 input / $10.00 output per 1M tokens.
# GPT-4o-mini: $0.15 input / $0.60 output per 1M tokens.
GPT_4O_INPUT = 250_000_000
GPT_4O_OUTPUT = 1_000_000_000
GPT_4O_MINI_INPUT = 15_000_000
GPT_4O_MINI_OUTPUT = 60_000_000

# --- OpenAI text-embedding-3 (per 1M input tokens, no output) ---------------
# Large: $0.13 / 1M; Small: $0.02 / 1M.
OPENAI_EMBED_3_LARGE_INPUT = 13_000_000
OPENAI_EMBED_3_SMALL_INPUT = 2_000_000

# --- Vertex text-embedding-005 (per 1M input tokens, no output) -------------
# $0.025 / 1M input tokens.
VERTEX_EMBED_005_INPUT = 2_500_000


# Model name -> (input_rate, output_rate) in micro-cents per 1M tokens.
# Keys cover the common spellings each provider uses; suffix-match
# below handles cross-region inference profiles.
_RATES: dict[str, tuple[int, int]] = {
    # Gemini
    "gemini-2.5-flash": (GEMINI_2_5_FLASH_INPUT, GEMINI_2_5_FLASH_OUTPUT),
    "gemini-2.5-flash-lite": (GEMINI_2_5_FLASH_INPUT, GEMINI_2_5_FLASH_OUTPUT),
    "gemini-2.5-pro": (GEMINI_2_5_PRO_INPUT, GEMINI_2_5_PRO_OUTPUT),
    "gemini-2.0-flash": (GEMINI_2_0_FLASH_INPUT, GEMINI_2_0_FLASH_OUTPUT),
    # Claude (Anthropic native + Vertex spellings + Bedrock spellings)
    "claude-sonnet-4-20250514": (CLAUDE_SONNET_4_INPUT, CLAUDE_SONNET_4_OUTPUT),
    "claude-sonnet-4@20250514": (CLAUDE_SONNET_4_INPUT, CLAUDE_SONNET_4_OUTPUT),
    "anthropic.claude-sonnet-4-20250514-v1:0": (CLAUDE_SONNET_4_INPUT, CLAUDE_SONNET_4_OUTPUT),
    "claude-opus-4-20250514": (CLAUDE_OPUS_4_INPUT, CLAUDE_OPUS_4_OUTPUT),
    "claude-haiku-4-20250514": (CLAUDE_HAIKU_4_INPUT, CLAUDE_HAIKU_4_OUTPUT),
    # Current generation (Bedrock ids; the suffix-match below maps the
    # region-prefixed inference profiles, e.g. "us.anthropic.claude-opus-4-8").
    "anthropic.claude-opus-4-8": (CLAUDE_OPUS_4_INPUT, CLAUDE_OPUS_4_OUTPUT),
    "anthropic.claude-sonnet-4-6": (CLAUDE_SONNET_4_INPUT, CLAUDE_SONNET_4_OUTPUT),
    "anthropic.claude-haiku-4-5-20251001-v1:0": (CLAUDE_HAIKU_4_5_INPUT, CLAUDE_HAIKU_4_5_OUTPUT),
    # OpenAI GPT-4o family
    "gpt-4o": (GPT_4O_INPUT, GPT_4O_OUTPUT),
    "gpt-4o-mini": (GPT_4O_MINI_INPUT, GPT_4O_MINI_OUTPUT),
    # Embeddings
    "amazon.titan-embed-text-v2:0": (TITAN_EMBED_V2_INPUT, TITAN_EMBED_V2_OUTPUT),
    # Cohere Embed v4 (Bedrock id; suffix-match maps "us.cohere.embed-v4:0").
    "cohere.embed-v4:0": (COHERE_EMBED_V4_INPUT, COHERE_EMBED_V4_OUTPUT),
    "text-embedding-3-large": (OPENAI_EMBED_3_LARGE_INPUT, 0),
    "text-embedding-3-small": (OPENAI_EMBED_3_SMALL_INPUT, 0),
    "text-embedding-005": (VERTEX_EMBED_005_INPUT, 0),
}


# Per-call cost for models charged per request rather than per token
# (Vertex Discovery Engine ranker, Bedrock Rerank API). In micro-cents
# per call. Reranker call sites record `call_count=1` with zero tokens,
# so this is the ONLY cost they incur -- there is no token cost to double-
# count. Public pricing is ~$1 per 1K rank/rerank requests == $0.001/call
# == 100_000 micro-cents/call.
_PER_CALL_RATES: dict[str, int] = {
    # Vertex semantic ranker -- ~$1 / 1K rank requests (up to 200 records each).
    "semantic-ranker-default-004": 100_000,
    "semantic-ranker-default-003": 100_000,
    # Bedrock Rerank API -- ~$1 / 1K rerank queries (any model).
    "cohere.rerank-v3-5:0": 100_000,
    "amazon.rerank-v1:0": 100_000,
}


def per_call_cost_micros(model: str) -> int:
    """Per-call cost in micro-cents for per-request-priced models (rerankers).

    Returns 0 for token-priced models (the common case). Mirrors the
    exact + suffix-match lookup used by :func:`cost_usd_micros` so
    region-prefixed inference profiles (e.g. "us.cohere.rerank-v3-5:0")
    resolve to the base id.
    """
    rate = _PER_CALL_RATES.get(model)
    if rate is None:
        for key, val in _PER_CALL_RATES.items():
            if model.endswith(key):
                rate = val
                break
    return rate or 0


# Models we've already warned about -- keep the logger noise bounded.
_unknown_warned: set[str] = set()


def cost_usd_micros(model: str, prompt_tokens: int, completion_tokens: int) -> int:
    """Cost in micro-cents for one call.

    Returns 0 (and logs a one-time warning) for unknown models. Negative
    token counts are clamped to 0 -- callers shouldn't pass them but if
    they do we don't want a negative cost in the DB.
    """
    rates = _RATES.get(model)
    if rates is None:
        # Suffix match for cross-region Bedrock profiles, e.g.
        # `apac.anthropic.claude-sonnet-4-...` falls back to
        # `anthropic.claude-sonnet-4-...`.
        for key, val in _RATES.items():
            if model.endswith(key):
                rates = val
                break
    if rates is None:
        if model not in _unknown_warned:
            _log.warning("no pricing for model=%s; cost will be 0", model)
            _unknown_warned.add(model)
        return 0

    in_rate, out_rate = rates
    pt = max(0, int(prompt_tokens))
    ct = max(0, int(completion_tokens))
    return (pt * in_rate) // 1_000_000 + (ct * out_rate) // 1_000_000


def cost_to_usd_str(micros: int) -> str:
    """Format a micro-cent integer as a USD string with 6 decimals.

    >>> cost_to_usd_str(12_300_000)
    '$0.123000'
    >>> cost_to_usd_str(1_230)
    '$0.000012'
    >>> cost_to_usd_str(0)
    '$0.000000'
    """
    # 1 USD == 100_000_000 micro-cents. Render with 6 decimals so even
    # sub-cent values show; trailing zeros are kept for table alignment
    # in the admin dashboard.
    dollars = micros / 100_000_000
    return f"${dollars:.6f}"
