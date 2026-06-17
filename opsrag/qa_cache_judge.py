"""Flash-based cache-hit validator.

Sits in front of the Q&A cache lookup: when raw cosine lands in the
borderline band [0.93, 0.99) (lower bound widens to the per-category
floor that `lookup` accepted the hit at, e.g. FORENSIC 0.92, when the
caller passes `min_score`; upper bound is `cfg.qa_judge_upper`), call
Vertex Flash with a 1-shot judge prompt to check semantic equivalence
between the current query and the cached question. If Flash says NO,
treat as cache miss.

Why this exists
---------------
Cosine alone has a documented FP rate ~99% at 0.93 (InfoQ banking
case study). Examples that cosine misses but a human/LLM catches:

  - "SRE goals 2025" vs "SRE goals 2026"     -- year flips meaning
  - "how to add kafka topic" vs "how to delete kafka topic"
                                              -- verb flips meaning
  - "rotate cert in prod" vs "...in staging" -- env flips meaning

The discriminator-token regex catches some (years, IDs >= 2 digits).
Flash judge catches the rest: paraphrase + intent-shift + env/verb
swaps.

Cost / latency
--------------
~$0.00005/judge call, ~200-300ms latency. Triggered only on borderline
hits (~9% of all queries assuming 30% hit rate x 30% borderline).
Result: <$0.01/month at SRE-team scale.

Skip rules
----------
- cosine >= upper band (cfg.qa_judge_upper, default 0.99; env
  OPSRAG_QA_JUDGE_UPPER wins) -> skip judge, trust the threshold.
- cosine < band lower bound (max(min_score, 0.93)) -> already a miss;
  never reaches judge.
- env var OPSRAG_QA_JUDGE=0 -> bypass judge entirely (graceful disable).
- LLM not provided -> bypass (graceful degrade).
- LLM throws -> fail-CLOSED (return False, force miss): a borderline hit
  re-queries the corpus rather than serving a wrong-but-close cached
  answer (QUALITY > latency). Set cfg.qa_judge_fail_open=True (env
  OPSRAG_QA_JUDGE_FAIL_OPEN wins) to prefer availability and serve the
  cached candidate when the judge is down.

Stats are exposed via the module-level counter dict so /cache/summary
can report judge_calls, judge_yes, judge_no.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

_log = logging.getLogger("opsrag.qa_cache.judge")

# Cosine band where the judge runs. Below: regular miss. Above: trust
# the high score and skip the judge call.
JUDGE_LOWER = 0.93
# Auto-accept upper band. Config-overridable via `cfg.qa_judge_upper`
# (QACacheConfig, default 0.99) so operators can widen the band the LLM
# judge runs across without a rebuild -- raising the upper bound means MORE
# borderline hits get the judge instead of being auto-accepted on cosine
# alone (quality > latency). `OPSRAG_QA_JUDGE_UPPER` env still wins.
JUDGE_UPPER_DEFAULT = 0.99
_judge_upper: float = JUDGE_UPPER_DEFAULT
# Back-compat module constant: kept so existing imports/tests still see a
# value. The live band is resolved through `_get_judge_upper()`.
JUDGE_UPPER = JUDGE_UPPER_DEFAULT

# R14 -- fail mode when the borderline-band judge LLM itself errors. Default
# fail-CLOSED (False): an in-band judge error returns False so the hit falls
# through to a fresh corpus query rather than serving a wrong-but-close cached
# answer (QUALITY > latency). Set via `configure(qa_judge_fail_open=...)` from
# `cfg.qa_cache.qa_judge_fail_open`; the `OPSRAG_QA_JUDGE_FAIL_OPEN` env wins.
JUDGE_FAIL_OPEN_DEFAULT = False
_judge_fail_open: bool = JUDGE_FAIL_OPEN_DEFAULT


def configure(
    *,
    qa_judge_upper: float | None = None,
    qa_judge_fail_open: bool | None = None,
) -> None:
    """Apply the resolved `cfg.qa_judge_*` values as the module defaults.

    Called once at boot by the config wiring. The `OPSRAG_QA_JUDGE_UPPER` /
    `OPSRAG_QA_JUDGE_FAIL_OPEN` env vars still override these when present
    (see `_get_judge_upper` / `_get_judge_fail_open`)."""
    global _judge_upper, JUDGE_UPPER, _judge_fail_open
    if qa_judge_upper is not None:
        _judge_upper = float(qa_judge_upper)
        JUDGE_UPPER = _judge_upper
    if qa_judge_fail_open is not None:
        _judge_fail_open = bool(qa_judge_fail_open)


def _get_judge_upper() -> float:
    """Resolve the auto-accept upper band. Env wins (config-overridable
    latency pattern), else the config-derived default (default 0.99)."""
    env = os.environ.get("OPSRAG_QA_JUDGE_UPPER")
    if env is not None:
        try:
            return float(env)
        except ValueError:
            _log.warning(
                "OPSRAG_QA_JUDGE_UPPER=%r is not a float; using %.3f",
                env, _judge_upper,
            )
    return _judge_upper


def _get_judge_fail_open() -> bool:
    """Resolve the judge-error fail mode. Env wins (config-overridable
    pattern), else the config-derived default (default False = fail-closed)."""
    env = os.environ.get("OPSRAG_QA_JUDGE_FAIL_OPEN")
    if env is not None:
        return env.strip().lower() in ("1", "true", "yes", "on")
    return _judge_fail_open


@dataclass
class JudgeStats:
    calls: int = 0
    yes: int = 0
    no: int = 0
    errors: int = 0
    skipped_high_conf: int = 0


_stats = JudgeStats()


def stats() -> dict:
    return {
        "calls": _stats.calls,
        "yes": _stats.yes,
        "no": _stats.no,
        "errors": _stats.errors,
        "skipped_high_conf": _stats.skipped_high_conf,
    }


def is_enabled() -> bool:
    return os.environ.get("OPSRAG_QA_JUDGE", "1").lower() in ("1", "true", "yes", "on")


_PROMPT = """You are deciding whether two operations queries are asking the SAME thing.

Two queries are the SAME if a correct answer to one would also be a correct answer to the other.
Two queries are DIFFERENT if any of these flip:
- year / version / cycle number (2025 vs 2026, v3.7 vs v4.0)
- service / environment (prod vs staging, service-a vs service-b)
- verb / intent (add vs delete, enable vs disable, list vs count)
- specific entity (pipeline 123 vs 456, TICKET-7890 vs TICKET-7891)
- scope (one cycle vs all cycles, one repo vs all repos)

Examples:
  A: "Tell me SRE goals of cycle 7 2025"   B: "Tell me SRE goals of cycle 8 2025"        -> DIFFERENT (cycle number)
  A: "How to rotate the SSL cert"          B: "How do I rotate the production SSL cert"  -> SAME (paraphrase)
  A: "Why did pipeline 1531171 fail"       B: "Why did pipeline 1531172 fail"            -> DIFFERENT (pipeline id)
  A: "Add a kafka topic"                   B: "Delete a kafka topic"                     -> DIFFERENT (verb)
  A: "service-a slow in prod"              B: "service-a slow in staging"                -> DIFFERENT (env)

Now classify these two:

Query A: {a}
Query B: {b}

Reply with ONLY one word: SAME or DIFFERENT."""


async def judge_match(
    *,
    current_query: str,
    cached_question: str,
    cosine: float,
    llm,
    min_score: float | None = None,
) -> bool:
    """Return True if the cached entry is OK to serve, False to force miss.

    ``min_score`` is the per-category cosine floor that ``lookup`` actually
    accepted this hit at (the qacache track threads it through). Per-category
    floors can sit *below* ``JUDGE_LOWER`` (e.g. FORENSIC 0.92) -- so a 0.925
    FORENSIC hit passes ``lookup`` yet, under a hardcoded ``JUDGE_LOWER`` lower
    bound, would default-allow WITHOUT being judged. We resolve the effective
    judge lower bound to ``max(min_score, JUDGE_LOWER)`` so the judge runs
    across ``[lower, upper)`` and that 0.925 FORENSIC hit IS judged. When
    ``min_score`` is None we fall back to ``JUDGE_LOWER`` (legacy behaviour).

    Behaviour summary (upper band = `_get_judge_upper()`,
    lower band = `max(min_score, JUDGE_LOWER)` when min_score given else
    `JUDGE_LOWER`):
      - cosine >= upper band      -> True (skip judge, high confidence)
      - lower <= cosine < upper band -> call Flash judge
      - cosine < lower            -> caller shouldn't have asked us; default True
                                    (cosine threshold filter is the source of
                                    truth at the lower bound)
      - judge disabled (env / no LLM) -> True (graceful degrade)
      - judge errors              -> fail-CLOSED (False) so a borderline hit
                                    re-queries the corpus; fail-OPEN (True) only
                                    when `qa_judge_fail_open` is set.

    Identical strings short-circuit to True without calling the LLM.
    """
    if not is_enabled() or llm is None:
        return True
    if not current_query or not cached_question:
        return True
    if current_query.strip().lower() == cached_question.strip().lower():
        return True  # exact-match short-circuit
    if cosine >= _get_judge_upper():
        _stats.skipped_high_conf += 1
        return True
    lower = max(min_score, JUDGE_LOWER) if min_score is not None else JUDGE_LOWER
    if cosine < lower:
        return True  # caller already past threshold; defensive

    _stats.calls += 1
    try:
        resp = await llm.generate(
            messages=[{"role": "user", "content": _PROMPT.format(
                a=current_query[:500], b=cached_question[:500],
            )}],
            temperature=0.0,
            max_tokens=4,
            purpose="qa_cache_judge",
        )
        text = (getattr(resp, "content", None) or getattr(resp, "text", "") or "").strip().upper()
        # Be permissive on parsing -- Flash sometimes includes punctuation
        # or a leading word.
        is_same = text.startswith("SAME") or " SAME" in text
        is_diff = text.startswith("DIFFERENT") or " DIFFERENT" in text
        if is_diff and not is_same:
            _stats.no += 1
            _log.info(
                "judge=DIFFERENT cosine=%.3f cur=%r cached=%r",
                cosine, current_query[:80], cached_question[:80],
            )
            return False
        # Treat ambiguous / SAME / unrecognized -> allow (default-allow).
        _stats.yes += 1
        return True
    except Exception as exc:
        _stats.errors += 1
        fail_open = _get_judge_fail_open()
        _log.warning(
            "judge call failed (cosine=%.3f): %s -- %s",
            cosine, exc,
            "serving cached hit (fail-open)" if fail_open
            else "forcing cache miss (fail-closed)",
        )
        # R14 -- fail-CLOSED by default: a borderline hit falls through to a
        # fresh corpus query when the safeguard is down (QUALITY > latency).
        # Only serve the cached candidate when explicitly fail-open.
        return fail_open
