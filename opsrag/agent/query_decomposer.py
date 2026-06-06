"""T1.1 -- Multi-query decomposition.

When a query asks about MULTIPLE distinct sources (compare/contrast,
list-with-filter, multi-hop traces), splitting it into 2-4 focused
sub-queries and RRF-merging the parallel retrieval pools beats a
single retrieval pass that has to find both halves at once.

Multi-document recall is a structural ceiling that single-document
retrieval fixes cannot move -- decomposition is the only thing that
does.

Design rules:
  1. Default to ONE sub-query (the original). The bot's existing
     hybrid_search is good enough for most queries; we only want to
     pay the LLM + parallel-retrieval cost when the question actually
     spans multiple sources.
  2. Sub-query #1 is ALWAYS the user's original query verbatim.
     This is defensive: the merged pool can only improve over a
     single-pool baseline, never regress.
  3. Cap at 4 sub-queries to keep latency + cost bounded.
  4. Skipped entirely (returns [query]) when `OPSRAG_DECOMPOSE_QUERIES`
     env var is anything other than "1" / "true". Off by default
     until validated by golden eval.

Used by `opsrag.mcp.knowledge._h_knowledge_search` to fan out
parallel `hybrid_search` calls. Output merged via cross-pool RRF
(`opsrag.vectorstores.rrf.rrf_merge_pools`).
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass

from opsrag.interfaces.llm import LLMProvider

_log = logging.getLogger("opsrag.agent.query_decomposer")

_MAX_SUB_QUERIES = 4
_LLM_MAX_TOKENS = 600


@dataclass
class Decomposition:
    """Output of decompose_query."""
    sub_queries: list[str]      # 1..4 sub-queries; [0] is always the original
    reason: str                  # short explanation (logged, not user-facing)
    used_llm: bool               # False when feature-flag off or heuristic short-circuit


def is_enabled() -> bool:
    return os.environ.get("OPSRAG_DECOMPOSE_QUERIES", "").lower() in ("1", "true", "yes")


# Cheap heuristic gate -- skip the LLM call when the query is clearly
# single-target. Saves ~$0.0002 + ~500ms per single-target query.
# Multi-target signals: explicit comparison verbs, "and"/"AND" between
# clauses, "list all ... and ...", multi-hop "explain ... and how it
# integrates with ...". Conservative -- over-skip is fine (just means
# we retrieve as if it were one query, current behavior).
# Two regexes -- `_MULTI_TARGET_HINT` is case-INSENSITIVE for normal
# multi-target verbs/phrases; `_EXPLICIT_AND_CLAUSE` is case-SENSITIVE
# because capitalized `AND` between clauses is an explicit author signal
# (used heavily in the eval goldens: "Per X, AND per Y", "According to
# X..., AND according to Y...", "What does ... do, AND which file ...").
# Splitting the two prevents naive lowercase "X and Y" prose from
# triggering over-decomposition.
_MULTI_TARGET_HINT = re.compile(
    r"\b("
    r"compare(?:\s+\w+){0,3}\s+(?:vs|versus|to|with|against)"
    r"|differen(?:ce|ces|t)\s+between"
    # "how X integrate(s)/interact(s)/differ(s)/relate(s) with/to Y" --
    # accept "does/do/is" AND pronouns ("it") AND article+noun ("the X").
    r"|how\s+(?:does|do|is|it|they|the\s+\w+)?\s*\w[\w-]*\s+(?:integrate|interact|differ|relate)s?\s+(?:with|to|from|across)"
    # Standalone "integrate(s)/integration with" -- strong multi-target signal
    # even without the "how X" framing ("explain X and how it integrates with <svc-a>").
    r"|\b(?:integrate|integrates|integration)\s+(?:with|to|across)\b"
    r"|both\s+\w+\s+and\s+\w+"
    r"|\w+\s+(?:and|AND|&)\s+\w+\s+(?:in|across|between)"
    r"|list\s+all\s+\w+\s+that\s+(?:use|reference|depend|call)"
    r"|across\s+(?:both|all)\s+\w+\s+and\s+\w+"
    # Two distinct "Per X ..." clauses or two "According to X ..." clauses
    # implies the user wants information from two distinct sources.
    r"|per\s+the?\s+\w[\w\s-]{0,40}?,?\s+(?:and\s+per|AND\s+per)\s+"
    r"|according\s+to\s+\w[\w\s-]{0,40}?,?\s+(?:and\s+according|AND\s+according)\s+to"
    # "X end-to-end" usually means "explain the full flow across components"
    r"|\bend[-\s]to[-\s]end\b"
    r"|\s+\?\s*\w+.*\?"   # multiple question marks in one input
    r")",
    re.IGNORECASE,
)
# Case-SENSITIVE: capitalized "AND" between clauses is an explicit
# author signal that the question has two distinct halves. Avoid the
# false-positive of lowercase "X and Y" in normal prose by requiring
# the all-caps form. Common patterns this catches:
#   "... include <file-a>, AND what stage do ..."
#   "..., AND which artifact does ..."
#   "..., AND according to ..."
#   "..., AND per <svc-a>'s metadata file ..."
_EXPLICIT_AND_CLAUSE = re.compile(r"[,;?]\s*AND\s+(?:what|which|where|how|why|per|according|the|how)")


def _is_multi_target(query: str) -> bool:
    return bool(_MULTI_TARGET_HINT.search(query) or _EXPLICIT_AND_CLAUSE.search(query))


_SCHEMA = {
    "type": "object",
    "properties": {
        "needs_decomposition": {
            "type": "boolean",
            "description": (
                "True only when the question explicitly requires combining "
                "information from MULTIPLE distinct sources (compare, contrast, "
                "end-to-end flow across services, multi-hop integration). "
                "False for typical single-topic questions."
            ),
        },
        "sub_queries": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "When needs_decomposition is true, 2-4 self-contained "
                "retrieval queries. The first sub-query MUST be the user's "
                "original question verbatim. When false, leave empty."
            ),
        },
        "reason": {
            "type": "string",
            "description": "One short sentence explaining the decision.",
        },
    },
    "required": ["needs_decomposition", "reason"],
}


_SYSTEM_PROMPT = """You decide whether a DevOps/SRE knowledge-base query needs to be split into multiple sub-queries for retrieval, OR if a single pass is enough.

Default to NO decomposition. Most questions are single-topic and the retriever handles them well in one pass.

Split into 2-4 sub-queries when the question explicitly requires combining information from MULTIPLE distinct sources or services. The clearest signals:

  - "Compare X vs Y" / "differences between X and Y"
  - "What does X do, AND what does Y do" (capitalized "AND" between two question clauses)
  - "According to file-A, ..., AND according to file-B, ..." (two anchored clauses)
  - "Per X..., what is W; AND per Y..., what is Z" (two parallel anchored clauses)
  - "Explain X end-to-end and how it integrates with Y"
  - "Trace the request from edge to handler" (multi-hop spanning components)

Examples that SHOULD split (placeholders like <file-a>, <svc-a>, <runbook-a>
stand in for real names; the SHAPE of the decomposition is what matters):

  - "What condition does <file-a> use to include <file-b>, AND what stage do the unit-test jobs in <file-b> run in?"
    -> sub_queries: [
        "<original>",
        "<file-a> condition to include <file-b>",
        "<file-b> unit-test jobs stage definition"
      ]

  - "What script does <file-a>'s post-release job invoke, AND which artifact does the delivery job in <file-b> declare it triggers from?"
    -> sub_queries: [
        "<original>",
        "<file-a> post-release job script",
        "delivery job trigger artifact <file-b>"
      ]

  - "According to the <svc-a> <runbook-a>, where is the replica-count baseline stored? AND according to the <runbook-b>, what severity is assigned at the alerting thresholds?"
    -> sub_queries: [
        "<original>",
        "<svc-a> <runbook-a> replica-count baseline",
        "<runbook-b> severity threshold"
      ]

  - "Per the architecture overview, what gateway sits between the ingress and the service pods, AND per <svc-a>'s metadata file, what specific deployments declare it?"
    -> sub_queries: [
        "<original>",
        "architecture overview gateway between ingress and service pods",
        "<svc-a> metadata file deployments declarations"
      ]

  - "Compare how <env-a> values differ from <env-b> for <svc-a>"
    -> sub_queries: [
        "<original>",
        "<svc-a> <env-a> deploy values",
        "<svc-a> <env-b> deploy values"
      ]

Do NOT split for typical questions like:
  - "How is POST /api/v1/<endpoint> routed?" (single trace, no AND-clause)
  - "Which service handles <feature>?" (single topic)
  - "List runbooks for <svc-a>" (listing -- retrieval handles it)
  - "What does the auth middleware do?" (single component)

Rules:
1. When you DO split, the FIRST sub-query MUST be the user's original question VERBATIM. The follow-up sub-queries (2-4) are FOCUSED versions targeting individual aspects/anchors.
2. Each follow-up sub-query must be self-contained (a complete retrieval query, not a fragment). Include the anchor (file name, runbook name, service name) so retrieval can hit the right chunk.
3. Maximum 4 sub-queries total.
4. When NOT splitting, leave sub_queries empty and set needs_decomposition=false.
5. The strongest signal for splitting is **capitalized "AND" between two question clauses**, or "Per X..., AND per Y..." / "According to X..., AND according to Y..." patterns. Don't second-guess those.

Return strictly the JSON shape requested."""


def _normalize_sub_queries(raw: list, original: str) -> list[str]:
    """Defensive cleanup of LLM output: dedupe, cap at 4, ensure first
    sub-query is the original query verbatim."""
    cleaned: list[str] = []
    seen: set[str] = set()
    # Force original at position 0
    cleaned.append(original)
    seen.add(original.strip().lower())
    for sq in raw:
        if not isinstance(sq, str):
            continue
        s = sq.strip()
        if not s:
            continue
        key = s.lower()
        if key in seen:
            continue
        cleaned.append(s)
        seen.add(key)
        if len(cleaned) >= _MAX_SUB_QUERIES:
            break
    return cleaned


async def decompose_query(query: str, llm: LLMProvider | None) -> Decomposition:
    """Return 1..4 sub-queries for parallel retrieval.

    When the feature flag is off, when no LLM is bound, OR when the
    query doesn't trigger the multi-target heuristic, returns
    `Decomposition(sub_queries=[query], ...)` (single-query passthrough).

    Logs at INFO on every entry -- this gives the eval harness a
    deterministic way to count decomposer fire rate from backend logs
    (the previous version only logged the "split" success path, so
    short-circuit and no-split decisions were invisible).
    """
    query = (query or "").strip()
    if not query:
        return Decomposition(sub_queries=[query], reason="empty query", used_llm=False)

    if not is_enabled():
        _log.info("decompose: flag-off | query=%r", query[:80])
        return Decomposition(
            sub_queries=[query], reason="feature flag off", used_llm=False,
        )

    if llm is None:
        _log.info("decompose: no-llm | query=%r", query[:80])
        return Decomposition(
            sub_queries=[query], reason="no LLM bound", used_llm=False,
        )

    # Heuristic short-circuit: queries without any multi-target signal
    # almost certainly don't need decomposition. Skipping the LLM call
    # here cuts ~500ms off the latency budget for the typical 80%+ of
    # queries that are single-topic.
    if not _is_multi_target(query):
        _log.info("decompose: heuristic-skip | query=%r", query[:80])
        return Decomposition(
            sub_queries=[query],
            reason="no multi-target signal in query",
            used_llm=False,
        )
    _log.info("decompose: heuristic-pass | query=%r -- calling LLM", query[:80])

    try:
        response = await llm.generate(
            messages=[{"role": "user", "content": query}],
            system_prompt=_SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=_LLM_MAX_TOKENS,
            response_schema=_SCHEMA,
            purpose="query-decompose",
        )
    except Exception as exc:
        _log.warning("decomposer LLM call failed: %s -- using original", exc)
        return Decomposition(
            sub_queries=[query], reason=f"LLM error: {exc}", used_llm=False,
        )

    raw = (response.content or "").strip()
    try:
        data = json.loads(raw)
    except Exception:
        _log.warning("decomposer returned non-JSON: %s", raw[:200])
        return Decomposition(
            sub_queries=[query], reason="LLM returned non-JSON", used_llm=True,
        )

    if not data.get("needs_decomposition"):
        return Decomposition(
            sub_queries=[query],
            reason=str(data.get("reason", ""))[:200],
            used_llm=True,
        )

    raw_subs = data.get("sub_queries") or []
    if not isinstance(raw_subs, list) or len(raw_subs) < 2:
        # LLM said needs_decomposition but didn't return enough -- fall back.
        return Decomposition(
            sub_queries=[query],
            reason="LLM said split but returned <2 sub-queries",
            used_llm=True,
        )

    cleaned = _normalize_sub_queries(raw_subs, query)
    if len(cleaned) < 2:
        # After dedupe, only the original survived -- no actual split.
        return Decomposition(
            sub_queries=[query],
            reason="sub-queries collapsed to original after dedupe",
            used_llm=True,
        )

    _log.info(
        "decomposed query (%d sub-queries): %s",
        len(cleaned), str(data.get("reason", ""))[:120],
    )
    return Decomposition(
        sub_queries=cleaned,
        reason=str(data.get("reason", ""))[:200],
        used_llm=True,
    )
