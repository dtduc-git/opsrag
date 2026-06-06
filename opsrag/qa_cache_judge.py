"""Flash-based cache-hit validator.

Sits in front of the Q&A cache lookup: when raw cosine lands in the
borderline band [0.93, 0.97], call Vertex Flash with a 1-shot judge
prompt to check semantic equivalence between the current query and
the cached question. If Flash says NO, treat as cache miss.

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
- cosine >= HIGH_CONFIDENCE_SKIP_JUDGE (default 0.97) -> skip judge,
  trust the threshold.
- cosine < band lower bound -> already a miss; never reaches judge.
- env var OPSRAG_QA_JUDGE=0 -> bypass judge entirely (graceful disable).
- LLM not provided -> bypass (graceful degrade).
- LLM throws -> return True (default-allow); cache hit served per
  legacy behaviour. We choose default-allow over default-deny so a
  Vertex outage doesn't lock all hits.

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
JUDGE_UPPER = 0.97


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
) -> bool:
    """Return True if the cached entry is OK to serve, False to force miss.

    Behaviour summary:
      - cosine >= JUDGE_UPPER     -> True (skip judge, high confidence)
      - JUDGE_LOWER <= cosine < JUDGE_UPPER -> call Flash judge
      - cosine < JUDGE_LOWER      -> caller shouldn't have asked us; default True
                                    (cosine threshold filter is the source of
                                    truth at the lower bound)
      - judge disabled (env / no LLM) -> True (graceful degrade)
      - judge errors                  -> True (default-allow)

    Identical strings short-circuit to True without calling the LLM.
    """
    if not is_enabled() or llm is None:
        return True
    if not current_query or not cached_question:
        return True
    if current_query.strip().lower() == cached_question.strip().lower():
        return True  # exact-match short-circuit
    if cosine >= JUDGE_UPPER:
        _stats.skipped_high_conf += 1
        return True
    if cosine < JUDGE_LOWER:
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
        _log.warning("judge call failed (cosine=%.3f): %s", cosine, exc)
        return True  # default-allow on error
