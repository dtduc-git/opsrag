"""Hard limits + circuit breakers for the investigation agent.

All constants live here so they're one diff to tune. Anything that
gates tree growth, recursion depth, or wall-clock/budget should be
declared in this module -- DO NOT scatter constants across nodes.

Reference: Datadog Bits AI SRE -- "early SRE agents scaled by performing
more tool calls and prompting an LLM to summarize the responses [which]
slowly degraded model performance or exceeded the context window limit."
Hard caps stop the tree from blowing up cost or latency.
"""
from __future__ import annotations

# -- Tree shape ------------------------------------------------------
MAX_DEPTH: int = 5
"""Hard ceiling on recursion. A node at depth=MAX_DEPTH cannot recurse."""

SOFT_DEPTH: int = 3
"""Past this depth, fanout + confidence requirements tighten so the tree
narrows toward a single causal chain instead of fanning wider."""

# -- Fanout schedule -------------------------------------------------
# Per-depth cap on siblings the LLM may emit. Each entry says "at this
# depth, ask for AT MOST N hypotheses". The schedule narrows
# progressively because the space-of-distinct-mechanisms collapses
# naturally as we drill deeper -- depth 0 spans whole subsystems, depth
# 3 should already be one mechanism wide.
#
# Sibling-cosine dedup runs in addition to this schedule: even if the
# LLM honors the cap, near-duplicate siblings are rejected post-hoc.
FANOUT_SCHEDULE: dict[int, int] = {
    0: 5,  # root hypotheses -- broad subsystem spread
    1: 4,  # narrow one layer
    2: 3,
    3: 2,
}
"""Per-depth fanout. Anything deeper than the max key uses
`FANOUT_DEEP_DEFAULT`. Tuning lever: shrink the deeper entries to
narrow the search faster."""

FANOUT_DEEP_DEFAULT: int = 2
"""Cap applied at any depth not present in `FANOUT_SCHEDULE` -- i.e.
depth >= max(FANOUT_SCHEDULE)+1. Matches the legacy
`MAX_HYPOTHESES_DEEP` value for backward compatibility."""

MIN_CONFIDENCE_TO_RECURSE: float = 0.7
"""Validated node needs confidence >= this at depth <= SOFT_DEPTH to spawn
children. Below this we mark the node validated-but-shallow and stop."""

MIN_CONFIDENCE_DEEP: float = 0.8
"""Stricter threshold past SOFT_DEPTH -- only highly-confident nodes
warrant a deeper search."""

# -- Global circuit breakers -----------------------------------------
# Any of these tripping -> graceful termination: mark remaining pending
# nodes inconclusive with a circuit_breaker_* reason and synthesize from
# whatever evidence is already collected.

MAX_TOTAL_NODES: int = 50
"""Hard kill if the tree fans out beyond this regardless of depth."""

MAX_TOTAL_TOOL_CALLS: int = 300
"""Sum of retrievals + LLM judge + LLM gen calls across the investigation."""

MAX_INVESTIGATION_DURATION_SEC: float = 300.0
"""5 minutes wall clock. P95 latency target is ~2 min; this is the kill."""

MAX_LLM_TOKENS_PER_INVESTIGATION: int = 1_000_000
"""Safety net against runaway prompt growth. Realistic single
investigation is ~50K tokens; cap is 20x headroom."""

# -- Per-call timeouts (hard, interrupting) --------------------------
# The wall-clock breaker (MAX_INVESTIGATION_DURATION_SEC) is only checked
# at node boundaries, so a single hung provider await could block far
# past it. These wrap each external await in asyncio.wait_for so one
# stuck retrieval/LLM/embed call degrades gracefully instead of hanging
# the whole investigation (and every other request on the event loop).
PER_CALL_RETRIEVAL_TIMEOUT_SEC: float = 30.0
"""Max wall-clock for a single retrieve()/embed() await. On timeout the
call degrades to empty results rather than raising."""

PER_CALL_LLM_TIMEOUT_SEC: float = 90.0
"""Max wall-clock for a single LLM generate() await. On timeout the call
degrades to an empty response (-> inconclusive verdict / no hypotheses)."""

# -- Decoding --------------------------------------------------------
HYPOTHESIS_GEN_TEMPERATURE: float = 0.7
"""Temperature for ROOT + SUB hypothesis generation only. The judge and
synthesis calls stay at 0.0 (deterministic). Hypothesis generation is the
one place we WANT spread: a greedy temp-0 decoder collapses the 3-5
"diverse" hypotheses toward one restated cause, fighting the prompt's
diversity mandate. Keep this > 0 to actually get distinct candidates."""

# -- Loop prevention -------------------------------------------------
DUPLICATE_ANCESTOR_COSINE_THRESHOLD: float = 0.9
"""If a child statement embeds within this cosine of any ancestor on the
path-to-root, we treat it as a rephrase and skip -- chokes the loop where
the LLM rewords the parent under a new label."""

DUPLICATE_SIBLING_COSINE_THRESHOLD: float = 0.85
"""Tighter threshold for sibling-level dedup. Siblings are expected to
be semantically closer than ancestor/descendant pairs (they all
decompose the SAME parent), so the bar for "near-duplicate" is set
lower than the ancestor check. When a candidate child cosines above
this against any already-accepted sibling, we reject it as
`duplicate_sibling`."""

# -- Retrieval defaults ----------------------------------------------
EVIDENCE_TOP_K: int = 6
"""How many chunks to fetch per hypothesis test. Kept small -- the
evidence-judge prompt should see focused context, not a dump."""

BOOTSTRAP_TOP_K: int = 4
"""Chunks fetched for runbook + past-incident bootstrap query."""

MAX_TOOL_CALLS_PER_HYPOTHESIS: int = 2
"""How many LIVE telemetry tools (datadog/rootly/code/...) the agent may
invoke to test ONE hypothesis. Bits-AI-style targeted telemetry: a couple
of focused signals per hypothesis, never a dump-all-at-once."""

# -- Confidence calibration ------------------------------------------
INCONCLUSIVE_CONFIDENCE_CEILING: float = 0.5
"""When the evidence-judge returns 'inconclusive', clamp confidence to
this so the node can't accidentally pass the recurse threshold."""

INVALIDATED_CONFIDENCE_FLOOR: float = 0.0
"""Invalidated nodes always carry confidence 0 -- no downstream node
should treat them as anything but a dead branch."""


def threshold_for_depth(depth: int) -> float:
    """Confidence threshold the node must clear to spawn children at
    the given depth. Tightens past SOFT_DEPTH."""
    if depth > SOFT_DEPTH:
        return MIN_CONFIDENCE_DEEP
    return MIN_CONFIDENCE_TO_RECURSE


def fanout_for_depth(depth: int) -> int:
    """Max siblings the LLM may emit AT this depth.

    Argument is the depth of the nodes being GENERATED (root = 0,
    immediate children of root = 1, etc.). Looks up
    `FANOUT_SCHEDULE` directly and falls back to `FANOUT_DEEP_DEFAULT`
    past the last keyed entry.
    """
    if depth in FANOUT_SCHEDULE:
        return FANOUT_SCHEDULE[depth]
    if depth < 0:
        return FANOUT_SCHEDULE[0]
    return FANOUT_DEEP_DEFAULT


# -- Backward-compatibility aliases ----------------------------------
# External callers may still `from .limits import MAX_HYPOTHESES_PER_LEVEL`
# or `MAX_HYPOTHESES_DEEP`. Keep the symbols pointing at the schedule's
# anchor values so nothing breaks at import time.
MAX_HYPOTHESES_PER_LEVEL: int = FANOUT_SCHEDULE[0]
"""DEPRECATED -- alias for `FANOUT_SCHEDULE[0]`. Use
`fanout_for_depth(depth)` instead. Kept exported so older imports don't
crash."""

MAX_HYPOTHESES_DEEP: int = FANOUT_DEEP_DEFAULT
"""DEPRECATED -- alias for `FANOUT_DEEP_DEFAULT`. Use
`fanout_for_depth(depth)` instead. Kept exported so older imports don't
crash."""
