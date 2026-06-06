"""M2 -- token pricing in micro-cents.

A *micro-cent* = 1 / 100_000_000 of a US dollar. We store costs as
integers in this unit so the per-event arithmetic never touches
floats -- Postgres can sum them losslessly, and `decimal`-vs-`float`
rounding surprises stop existing.

Conversion to display::

    micros / 100_000_000  ->  USD

The constants below are encoded as "micro-cents per **1M tokens**"
because that's the granularity public price sheets use::

    GEMINI_2_5_FLASH_INPUT = 7_500_000
        = $0.075 / 1M tokens
        = 7_500_000 micro-cents / 1M tokens

To get the cost of N tokens at rate R (micro-cents / 1M tokens)::

    micros = N * R // 1_000_000

Integer-only math, intentional. The truncation error on a single 1k-token
call is at most 1 micro-cent = $0.00000001 -- well below display precision.

Public pricing as of 2026-05-15. Update when GCP / Anthropic publishes a
new sheet; the constants are deliberately at module scope so a single
edit propagates everywhere.
"""
from __future__ import annotations

import logging

_log = logging.getLogger("opsrag.llms.pricing")

# --- Gemini 2.5 (Vertex AI) -------------------------------------------------
# Flash: $0.075 input / $0.30 output per 1M tokens.
# Pro:   $1.25  input / $5.00 output per 1M tokens (under-128k context).
GEMINI_2_5_FLASH_INPUT = 7_500_000
GEMINI_2_5_FLASH_OUTPUT = 30_000_000
GEMINI_2_5_PRO_INPUT = 125_000_000
GEMINI_2_5_PRO_OUTPUT = 500_000_000

# --- Gemini 2.0 (legacy aliases, same Flash rates) --------------------------
GEMINI_2_0_FLASH_INPUT = GEMINI_2_5_FLASH_INPUT
GEMINI_2_0_FLASH_OUTPUT = GEMINI_2_5_FLASH_OUTPUT

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
    # Embeddings
    "amazon.titan-embed-text-v2:0": (TITAN_EMBED_V2_INPUT, TITAN_EMBED_V2_OUTPUT),
    # Cohere Embed v4 (Bedrock id; suffix-match maps "us.cohere.embed-v4:0").
    "cohere.embed-v4:0": (COHERE_EMBED_V4_INPUT, COHERE_EMBED_V4_OUTPUT),
}


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
