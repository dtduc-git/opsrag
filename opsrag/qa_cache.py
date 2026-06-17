"""Semantic Q&A cache -- short-circuits the agent when a sufficiently
similar question was answered before.

Architecture:
- Separate Qdrant collection (default `opsrag_qa_cache`) keyed by the
  embedding of the original question.
- After embedding the new query, look up nearest neighbour. If cosine
  similarity >= threshold AND not expired AND not flagged low-quality,
  return the cached answer + sources without running the rest of the
  agent graph.
- On cache miss, the agent runs normally and the final
  (question, answer, sources) is written back into the cache.

TTL is content-type aware:
- procedural runbook answer: 14 days (knowledge drifts slowly)
- listing answer: 1 day (file lists change with reindex)
- live status / time-sensitive: bypass entirely (skip patterns below)

Invalidation: when /index/repo reindexes a repo, drop all cache entries
whose `source_repos` payload contains that repo. Cache is best-effort --
any internal error logs and falls through to a fresh run.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass

from qdrant_client import AsyncQdrantClient
from qdrant_client import models as qm

from opsrag.qa_cache_normalize import normalize_query

_log = logging.getLogger("opsrag.qa_cache")

DEFAULT_COLLECTION = "opsrag_qa_cache"
DEFAULT_THRESHOLD = 0.93
DEFAULT_TTL_SECONDS = 14 * 24 * 3600  # 14 days

# Skip cache for queries that explicitly ask about live state.
# A hit on these would be stale and dangerous (e.g. "is service X up right now?").
_SKIP_PATTERNS = re.compile(
    r"\b(current|latest|now|today|right now|just now|recently)\b",
    re.IGNORECASE,
)
# Possessive / user-scoped pronouns that often imply per-user state.
_USER_SCOPED = re.compile(r"\b(my|our|your)\b", re.IGNORECASE)

# NOTE: an earlier `_SPECIFIC_ID_PATTERNS` regex (GitLab URL paths +
# `pipeline 123`, `MR !789`, commit SHAs) was removed 2026-05-09 PM
# because it was vendor-hardcoded and didn't generalize across MCPs.
# Replaced by a path-agnostic rule in `query_with_session*` that skips
# the Q&A cache entirely for tool-calling-path answers, since those
# describe live state. Future Sub-sprint 3 introduces the dedicated
# `opsrag_investigations` collection for tool-path caching with tags.

# Discriminator tokens -- entities that, when they differ between two
# queries, must NOT cache-hit even if cosine is high. Originally one
# regex `\b\d{2,}\b`; extended with NER-style structured patterns to
# catch the cases that pure-digit regex misses:
#
#   - years / IDs (existing)        -> 2025 vs 2026
#   - semver / version              -> 3.7.0 vs 4.0.0
#   - ticket prefixes               -> TICKET-7890 vs TICKET-7891
#   - environment names             -> prod vs staging vs dev
#   - commit SHA fragments          -> a1b2c3d vs e4f5a6b
#   - kebab-case service / repo IDs -> service-a vs service-b
#   - GitLab paths                  -> /pipelines/123 vs /pipelines/456
#
# All patterns case-insensitive on the input. Each match contributes a
# normalized token to a set; cache miss when sets differ.
_DISCRIMINATOR_NUMBER = re.compile(r"\b\d{2,}\b")

# 4-digit years (1900-2099 range) -- caught by NUMBER too but kept
# explicit in case future tokenization changes.
_DISCRIMINATOR_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")

# Semver / version-like (3.7, 1.2.3, v4.0). Tagged with "semver:" prefix
# so "3.7" and "37" don't collide.
_DISCRIMINATOR_SEMVER = re.compile(r"\bv?(\d+\.\d+(?:\.\d+)?)\b")

# Ticket / incident IDs -- common Jira-style A-Z+digit (e.g. INC-,
# TICKET-, OPS-). Captured WITH prefix because the prefix itself is
# meaningful (one project != another).
_DISCRIMINATOR_TICKET = re.compile(r"\b([A-Z]{2,5}-\d{2,})\b")

# Environment names. Built DYNAMICALLY at call time from the active
# DeploymentContext (Constitution Principle VI -- the engine carries no
# org-specific knowledge). The regex matches the generic defaults below
# PLUS whatever environment names the runtime deployment supplies. We
# rebuild on each call because active_deployment() can change after
# startup, so a frozen module-level constant would bake in stale (and
# potentially org-specific) names.
_GENERIC_ENV_DEFAULTS = ("prod", "production", "staging", "dev", "test")


def _env_discriminator_regex() -> re.Pattern[str]:
    """Compile the environment-name discriminator regex from the generic
    defaults plus the active deployment's environments.

    De-duplicates, lowercases, and regex-escapes each token. Longer
    alternatives are ordered first so that multi-word / hyphenated names
    (e.g. "pre-prod") win over their prefixes during alternation."""
    from opsrag.agent.prompt_render import active_deployment

    tokens: set[str] = set()
    for name in _GENERIC_ENV_DEFAULTS:
        tokens.add(name.lower())
    for name in active_deployment().environments:
        name = (name or "").strip().lower()
        if name:
            tokens.add(name)
    # Longest-first so alternation prefers the most specific match.
    ordered = sorted(tokens, key=lambda t: (-len(t), t))
    alternation = "|".join(re.escape(t) for t in ordered)
    return re.compile(rf"\b({alternation})\b", re.IGNORECASE)


def _service_discriminator_regex() -> re.Pattern[str] | None:
    """Compile a service-name discriminator regex from the active
    deployment's service inventory. Mirrors ``_env_discriminator_regex``.

    The kebab-case extractor (``_DISCRIMINATOR_KEBAB``, >= 2 segments)
    already separates multi-segment service names (``api-gateway`` vs
    ``kafka-broker``). It MISSES single-token service names (``auth``,
    ``billing``, ``redis``) because they have no hyphen and fall below the
    length / structure floor -- so "auth logs" and "billing logs" yield
    identical discriminator sets and wrong-service cache hits leak.

    This regex closes that gap by matching the operator's declared service
    names verbatim (whole-word, case-insensitive). Built per call because
    ``active_deployment()`` can change after startup (same rationale as the
    env regex). Returns ``None`` when the inventory is empty so the caller
    skips the layer entirely (no regression for org-free deployments)."""
    from opsrag.agent.prompt_render import active_deployment

    tokens = {
        n.strip().lower()
        for n in active_deployment().services
        if n and n.strip()
    }
    if not tokens:
        return None
    # Longest-first so alternation prefers the most specific match (a
    # multi-word / hyphenated service wins over a contained prefix).
    ordered = sorted(tokens, key=lambda t: (-len(t), t))
    alternation = "|".join(re.escape(t) for t in ordered)
    return re.compile(rf"\b({alternation})\b", re.IGNORECASE)

# Commit SHA fragment (>= 7 hex chars).
_DISCRIMINATOR_SHA = re.compile(r"\b([0-9a-f]{7,40})\b")

# Kebab-case names with >= 2 segments -- typical for service / repo /
# pod names (api-gateway, kafka-broker, payments-svc, payments-svc-prod-1).
# Filter very short matches (e.g. "a-b") via length floor.
_DISCRIMINATOR_KEBAB = re.compile(r"\b([a-z][a-z0-9]+(?:-[a-z0-9]+){1,4})\b")

# GitLab-ish paths: capture the leaf id after the type segment.
# `/pipelines/1531171`, `/jobs/5022983`, `/merge_requests/9540`,
# `/commit/abcdef1`. Captured with type prefix to disambiguate.
_DISCRIMINATOR_GL_PATH = re.compile(
    r"/(?:pipelines|jobs|merge_requests|commit|builds)/([0-9a-f]+)",
    re.IGNORECASE,
)

# Filter list -- kebab-case matches that aren't actually service names.
# These would otherwise create spurious mismatches because they appear
# in nearly every query (e.g. "sre-bot" in greetings).
_KEBAB_NOISE = frozenset({
    "right-now", "real-time", "build-up", "follow-up", "sign-off",
    "step-by-step", "end-to-end",
})


# Compare-ranking counters -- exposed via /cache/summary so we can see
# per-layer extractor activity. `disagreements` increments when the
# combined set differs from the regex-only set (i.e., spaCy added a
# token the regex layer didn't produce).
@dataclass(slots=False)
class _DiscriminatorStats:
    calls: int = 0
    regex_tokens_total: int = 0
    spacy_tokens_total: int = 0
    spacy_added_tokens: int = 0  # tokens spaCy contributed beyond regex
    disagreements: int = 0       # calls where spaCy produced >=1 new token


_disc_stats = _DiscriminatorStats()


def discriminator_stats() -> dict:
    return {
        "calls": _disc_stats.calls,
        "regex_tokens_total": _disc_stats.regex_tokens_total,
        "spacy_tokens_total": _disc_stats.spacy_tokens_total,
        "spacy_added_tokens": _disc_stats.spacy_added_tokens,
        "disagreements": _disc_stats.disagreements,
    }


def _discriminator_tokens(query: str) -> frozenset[str]:
    """Extract the set of meaning-shifting tokens from a query.

    Two-layer ensemble:
      1. Regex (precision-first) -- structured tokens (years, semver,
         tickets, envs, SHAs, kebab-case, GitLab paths).
      2. spaCy NER (recall-first) -- free-form entities (PERSON, ORG,
         GPE, DATE, CARDINAL, ...). Opt-in via `OPSRAG_QA_NER_SPACY=1`;
         no-op otherwise.

    Returns the UNION of both layers. Caller compares two sets for
    equality; any difference -> force cache miss. Per-layer counters
    in `_disc_stats` let `/cache/summary` rank extractor activity.
    """
    if not query:
        return frozenset()
    q = query.lower()
    tokens: set[str] = set()

    # Plain multi-digit numbers (existing behaviour).
    for m in _DISCRIMINATOR_NUMBER.findall(q):
        tokens.add(f"num:{m}")
    # Years are also numbers but tagged distinctly so a query mentioning
    # "2025 brokers" doesn't collide with "brokers count 2025".
    for m in _DISCRIMINATOR_YEAR.findall(q):
        tokens.add(f"year:{m}")
    # Semver -- captured as the dotted form, e.g. "3.7" or "1.2.3".
    for m in _DISCRIMINATOR_SEMVER.findall(q):
        tokens.add(f"semver:{m}")
    # Ticket IDs -- keep the prefix since "OPS-7890" vs "INC-7890" are
    # different incidents on different trackers.
    for m in _DISCRIMINATOR_TICKET.findall(query):  # case-sensitive on input
        tokens.add(f"ticket:{m.upper()}")
    # Environments -- canonicalize only the generic long form
    # "production"->"prod". The rest of the env vocabulary comes from the
    # active deployment context, so we keep deployment-supplied names
    # verbatim (lowercased) rather than mapping org-specific shorthands.
    env_canon = {"production": "prod"}
    for m in _env_discriminator_regex().findall(q):
        tokens.add(f"env:{env_canon.get(m.lower(), m.lower())}")
    # Commit SHAs -- only count strings that look hex AND aren't already
    # captured as numbers (filter pure-digit). Avoid double-counting.
    for m in _DISCRIMINATOR_SHA.findall(q):
        if not m.isdigit():
            tokens.add(f"sha:{m}")
    # Kebab-case service / repo names. Filter false positives.
    for m in _DISCRIMINATOR_KEBAB.findall(q):
        if m not in _KEBAB_NOISE and len(m) >= 5:  # length floor cuts noise
            tokens.add(f"kebab:{m}")
    # Declared service names -- catches SINGLE-TOKEN services (auth,
    # billing, redis) the kebab extractor structurally misses, so
    # "auth logs" vs "billing logs" no longer collide into a wrong-service
    # cache hit. Built per call (inventory can change post-startup); empty
    # inventory -> regex is None -> skip the layer (no regression).
    _svc_re = _service_discriminator_regex()
    if _svc_re is not None:
        for m in _svc_re.findall(q):
            tokens.add(f"svc:{m.lower()}")
    # GitLab-path leaf IDs.
    for m in _DISCRIMINATOR_GL_PATH.findall(q):
        tokens.add(f"glpath:{m.lower()}")

    # Track regex-layer stats before adding spaCy.
    _disc_stats.calls += 1
    regex_tokens = frozenset(tokens)
    _disc_stats.regex_tokens_total += len(regex_tokens)

    # Layer 2: spaCy NER. Opt-in via OPSRAG_QA_NER_SPACY=1; no-op
    # otherwise. Lazy-load + graceful degrade -- see qa_cache_ner.py.
    from opsrag.qa_cache_ner import extract_entities as _ner_extract
    spacy_tokens = _ner_extract(query)
    _disc_stats.spacy_tokens_total += len(spacy_tokens)
    if spacy_tokens:
        added = spacy_tokens - regex_tokens
        if added:
            _disc_stats.spacy_added_tokens += len(added)
            _disc_stats.disagreements += 1
        tokens |= spacy_tokens

    return frozenset(tokens)


@dataclass
class CacheHit:
    question: str
    answer: str
    sources: list[str]
    sources_content: list[dict]  # [{source, content}] -- same shape as /query response
    similarity: float
    age_seconds: float
    # Pre-rendered clickable URLs aligned with `sources` (None where the
    # source has no URL). Empty list = legacy entry written before the
    # field existed; caller derives URLs from `sources` instead.
    source_urls: list[str | None] = None  # type: ignore[assignment]
    # Sub-sprint 5 phase-2 -- stale-while-revalidate. True when the entry
    # is past its TTL but served anyway because the caller passed
    # `serve_stale=True`. The caller is expected to (a) flag this in the
    # response so the UI can show an "updating..." badge, and (b) trigger
    # a background revalidation so the next user gets the fresh answer.
    is_stale: bool = False

    def __post_init__(self) -> None:
        if self.source_urls is None:
            self.source_urls = []


# Answer-quality predicate used by `store()` to refuse caching obvious
# garbage. Tuned conservatively -- better to skip caching a few odd-but-
# valid answers than to poison the cache with degenerate output.
#
# Triggers -- ANY one trips the predicate:
#   - too short (< 40 visible chars) -- most legit answers are paragraphs
#   - too few alphabetic chars (< 25) -- pure punctuation / pure numbers
#   - dominant-char ratio > 70% -- the 2592-dashes incident
#   - distinct-char count <= 3 -- single repeating glyph
#   - whitespace ratio > 95% -- all blank
def _is_degenerate_answer(answer: str) -> bool:
    if not answer:
        return True
    stripped = answer.strip()
    if len(stripped) < 40:
        return True
    alpha = sum(1 for c in stripped if c.isalpha())
    if alpha < 25:
        return True
    # Whitespace-dominant
    if stripped and (stripped.count(" ") + stripped.count("\n") + stripped.count("\t")) / len(stripped) > 0.95:
        return True
    distinct = set(stripped)
    if len(distinct) <= 3:
        return True
    # Single-character dominance -- catches 2592 dashes, ====== headers,
    # ****** redaction blocks, etc.
    if stripped:
        from collections import Counter
        top_char, top_count = Counter(stripped).most_common(1)[0]
        # Ignore space dominance for prose (whitespace is checked above).
        if top_char not in (" ", "\n", "\t") and top_count / len(stripped) > 0.70:
            return True
    return False


def should_skip_cache(query: str) -> bool:
    """True for time-sensitive or user-scoped queries. Caller bypasses
    cache. Tool-calling path skip happens at write-time in `graph.py`
    based on `tool_path_active`, not at lookup-time here."""
    if not query:
        return True
    if _SKIP_PATTERNS.search(query):
        return True
    if _USER_SCOPED.search(query):
        return True
    return False


def _make_point_id(question: str, user_scope: str | None = None) -> str:
    """Deterministic UUID5 of the NORMALIZED question text, so re-asking
    the same query (modulo politeness/contractions) overwrites instead
    of duplicating. The original question stays on the stored payload
    for display purposes.

    T3.2 -- normalize first so "please how do I deploy" and "how do I
    deploy" hash to the same point ID. Lifts cache hit rate ~5-10pp
    without changing the cosine threshold.

    ``user_scope`` namespaces the id so a memory-influenced answer scoped
    to one user can't collide with (and overwrite) the shared entry -- or
    another user's entry -- for the same question."""
    normalized = normalize_query(question) or question.strip().lower()
    if user_scope:
        normalized = f"{normalized}|scope:{user_scope}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, normalized))


def _is_collection_missing(exc: BaseException) -> bool:
    # Qdrant returns 404 + "Collection `X` doesn't exist" when the collection
    # was dropped out-of-band (manual cleanup, external wipe). Detect so the
    # cache can self-heal instead of failing every call until restart.
    msg = str(exc)
    return "404" in msg or "doesn't exist" in msg or "Not found" in msg


class QAVectorCache:
    """Qdrant-backed semantic cache for (question -> answer, sources)."""

    def __init__(
        self,
        client: AsyncQdrantClient,
        collection: str = DEFAULT_COLLECTION,
        dimension: int = 768,
        threshold: float = DEFAULT_THRESHOLD,
        default_ttl_seconds: int = DEFAULT_TTL_SECONDS,
    ):
        self._client = client
        self._collection = collection
        self._dimension = dimension
        self._threshold = threshold
        self._default_ttl = default_ttl_seconds
        self._ensured = False

    async def ensure_collection(self) -> None:
        if self._ensured:
            return
        existing = await self._client.get_collections()
        names = {c.name for c in existing.collections}
        if self._collection not in names:
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=qm.VectorParams(
                    size=self._dimension, distance=qm.Distance.COSINE,
                ),
            )
            for field in ("repos", "source_repos", "quality_flag", "user_scope"):
                try:
                    await self._client.create_payload_index(
                        collection_name=self._collection,
                        field_name=field,
                        field_schema=qm.PayloadSchemaType.KEYWORD,
                    )
                except Exception:
                    pass
        self._ensured = True

    async def lookup(
        self,
        embedding: list[float],
        current_query: str | None = None,
        *,
        user_id: str | None = None,
        serve_stale: bool = False,
        max_stale_seconds: int = 7 * 24 * 3600,
        min_score: float | None = None,
    ) -> CacheHit | None:
        """Find the closest cached question. None if no good match.

        If `current_query` is provided, the discriminator-token set of the
        cached question must match exactly -- protects against year/ID/version
        leaks where cosine alone passes but the query semantically differs.

        `min_score` overrides the per-instance cosine floor (`self._threshold`)
        for this single call. The caller threads the per-category
        `policy_for(category)['qa_threshold']` here so tighter categories
        (MIXED 0.96, INFRA_GRAPH 0.94, FORENSIC 0.92, ...) get their own floor
        instead of the one global threshold. None falls back to
        `self._threshold` (legacy behaviour).

        `user_id` scopes the search: the caller is eligible only for SHARED
        entries (no `user_scope`) plus entries scoped to this same user. A
        memory-influenced answer cached for another user is never served here,
        even on a near-perfect cosine match. When `user_id` is None, only
        shared entries are eligible.

        Stale-while-revalidate: when `serve_stale=True`, an entry past
        its TTL but within `max_stale_seconds` of expiry is returned
        with `is_stale=True`. The caller is expected to mark the
        response and trigger a background refresh.
        """
        await self.ensure_collection()
        # Shared-or-mine filter. IsEmpty matches legacy + shared entries; the
        # match clause adds this user's own scoped entries. Without a user_id
        # only shared entries are eligible.
        shared = qm.IsEmptyCondition(is_empty=qm.PayloadField(key="user_scope"))
        if user_id:
            query_filter = qm.Filter(should=[
                shared,
                qm.FieldCondition(key="user_scope", match=qm.MatchValue(value=user_id)),
            ])
        else:
            query_filter = qm.Filter(must=[shared])
        try:
            result = await self._client.query_points(
                collection_name=self._collection,
                query=embedding,
                limit=1,
                with_payload=True,
                query_filter=query_filter,
            )
        except Exception as exc:
            if _is_collection_missing(exc):
                self._ensured = False
                await self.ensure_collection()
                return None
            _log.warning("cache lookup failed: %s", exc)
            return None
        if not result.points:
            return None

        hit = result.points[0]
        payload = hit.payload or {}
        score = float(hit.score)
        floor = self._threshold if min_score is None else float(min_score)
        if score < floor:
            return None
        if payload.get("quality_flag") == "low":
            return None
        if current_query is not None:
            cached_q = payload.get("question", "") or ""
            if _discriminator_tokens(current_query) != _discriminator_tokens(cached_q):
                _log.info(
                    "cache miss: discriminator-token mismatch (cur=%r cached=%r score=%.3f)",
                    current_query, cached_q, score,
                )
                return None
        ttl = int(payload.get("ttl_seconds", self._default_ttl))
        created = float(payload.get("created_at", 0))
        age = time.time() - created
        is_stale = False
        if created and age > ttl:
            if not serve_stale:
                return None  # legacy behavior -- drop expired
            # SWR window: serve stale only if within max_stale_seconds
            # of expiry. Beyond that, treat as fully expired.
            if (age - ttl) > max_stale_seconds:
                return None
            is_stale = True

        cached_answer = payload.get("answer", "")
        # Backstop: refuse to serve a degenerate answer even if it's
        # already in the cache (pre-fix entry, race, or future bug that
        # bypasses the write-side guard). Covers the legacy entries
        # written before `_is_degenerate_answer` landed.
        if _is_degenerate_answer(cached_answer):
            _log.warning(
                "qa_cache.lookup: skipping degenerate cached answer "
                "(len=%d, score=%.3f) for Q=%r",
                len(cached_answer), score, (current_query or "")[:80],
            )
            return None
        return CacheHit(
            question=payload.get("question", ""),
            answer=cached_answer,
            sources=list(payload.get("sources") or []),
            sources_content=list(payload.get("sources_content") or []),
            similarity=score,
            age_seconds=age,
            source_urls=list(payload.get("source_urls") or []),
            is_stale=is_stale,
        )

    async def store(
        self,
        question: str,
        embedding: list[float],
        answer: str,
        sources: list[str],
        sources_content: list[dict] | None = None,
        source_repos: list[str] | None = None,
        ttl_seconds: int | None = None,
        source_urls: list[str | None] | None = None,
        user_scope: str | None = None,
    ) -> None:
        """Cache a (question -> answer) entry.

        ``user_scope`` is set ONLY for answers that wove in per-user memories
        (Mem0). A scoped entry carries a ``user_scope`` payload field and is
        served back only to that same user (see ``lookup``); shared
        knowledge-base answers leave it unset so they stay globally cacheable.
        This closes the cross-user leak without gutting the hit rate for the
        common shared-question case.
        """
        if not question or not answer:
            return
        # Quality guard -- refuse to cache degenerate answers. Without
        # this, the generator's worst outputs poison the cache and get
        # served via semantic-match on every paraphrased follow-up.
        # Real incident 2026-05-25: an "ASCII flow diagram" turn produced
        # 2592 chars of literally only `-`, was marked grounded
        # (vacuously -- no factual claim to dispute), got cached, then
        # served verbatim to the next user via 96% match.
        if _is_degenerate_answer(answer):
            _log.warning(
                "qa_cache.store: REFUSING to cache degenerate answer "
                "(len=%d, distinct_chars=%d, alpha=%d) for Q=%r",
                len(answer),
                len(set(answer)),
                sum(1 for c in answer if c.isalpha()),
                question[:80],
            )
            return
        await self.ensure_collection()
        # R10: record the REAL repos at write time. ``source_repos`` is the
        # authoritative set the caller derived from the actual retrieved
        # chunks; ``_derive_repos(sources)`` is only a best-effort path
        # heuristic. We store the real set under ``repos`` so invalidation
        # matches on ground truth, and keep ``source_repos`` (heuristic when
        # nothing real was passed) for backward-compat / fallback matching.
        real_repos = list(source_repos) if source_repos else []
        heuristic_repos = _derive_repos(sources)
        repos = real_repos or heuristic_repos
        # Cap stored content per source to keep cache rows small.
        capped_content = [
            {"source": c.get("source", ""), "content": (c.get("content", "") or "")[:2500]}
            for c in (sources_content or [])
        ]
        payload = {
            "question": question,
            "question_hash": hashlib.sha1(question.encode()).hexdigest()[:12],
            "answer": answer,
            "sources": sources,
            "source_urls": list(source_urls) if source_urls else [],
            "sources_content": capped_content,
            # ``repos`` = ground-truth repos (caller-provided) when available,
            # else the heuristic. ``source_repos`` kept for legacy readers.
            "repos": repos,
            "source_repos": repos,
            "ttl_seconds": int(ttl_seconds or self._default_ttl),
            "created_at": time.time(),
            "quality_flag": "ok",
        }
        # Only stamp user_scope when the answer is user-specific. Leaving it
        # absent (not null) keeps the IsEmpty("shared") lookup filter matching
        # legacy + shared entries.
        if user_scope:
            payload["user_scope"] = user_scope
        try:
            await self._client.upsert(
                collection_name=self._collection,
                points=[qm.PointStruct(
                    id=_make_point_id(question, user_scope),
                    vector=embedding,
                    payload=payload,
                )],
                wait=False,
            )
        except Exception as exc:
            if _is_collection_missing(exc):
                self._ensured = False
                try:
                    await self.ensure_collection()
                    await self._client.upsert(
                        collection_name=self._collection,
                        points=[qm.PointStruct(
                            id=_make_point_id(question),
                            vector=embedding,
                            payload=payload,
                        )],
                        wait=False,
                    )
                    return
                except Exception as retry_exc:
                    _log.warning("cache store retry failed: %s", retry_exc)
                    return
            _log.warning("cache store failed: %s", exc)

    async def invalidate_repo(self, repo: str) -> int:
        """Delete cache entries whose answer was sourced from `repo`.
        Hooked into /index/repo so reindex flushes affected cache.

        R10: match primarily on the stored ground-truth ``repos`` field
        (set from the caller's real source repos in ``store``); keep the
        heuristic-derived ``source_repos`` field as a fallback so legacy
        entries written before ``repos`` existed still get flushed. The
        two-key ``should`` is OR semantics -- an entry tagged on either
        field is dropped."""
        await self.ensure_collection()
        try:
            await self._client.delete(
                collection_name=self._collection,
                points_selector=qm.FilterSelector(filter=qm.Filter(
                    should=[
                        qm.FieldCondition(
                            key="repos",
                            match=qm.MatchValue(value=repo),
                        ),
                        qm.FieldCondition(
                            key="source_repos",
                            match=qm.MatchValue(value=repo),
                        ),
                    ]
                )),
                wait=False,
            )
            return -1  # Qdrant doesn't return delete count
        except Exception as exc:
            _log.warning("cache invalidate_repo(%s) failed: %s", repo, exc)
            return 0

    async def purge(
        self,
        *,
        all: bool = False,
        repo: str | None = None,
        quality: str | None = None,
        older_than_seconds: int | None = None,
        question_substring: str | None = None,
    ) -> int:
        """Multi-strategy purge. Returns count purged or -1 if Qdrant
        doesn't return one. Combines filters with AND semantics; pass
        only one for clear intent.

        Strategies:
          - all=True              -- drop the entire collection contents
          - repo="<owner/repo>"   -- entries that touched this source repo
          - quality="low"         -- entries flagged low-quality (thumbs-down feedback)
          - older_than_seconds=N  -- entries created more than N seconds ago
          - question_substring=S  -- entries whose question contains S
        """
        await self.ensure_collection()
        if all:
            try:
                # Drop & re-create empty so the index sticks around.
                await self._client.delete_collection(self._collection)
                self._ensured = False
                await self.ensure_collection()
                return -1
            except Exception as exc:
                _log.warning("cache purge all failed: %s", exc)
                return 0

        must: list = []
        if repo:
            must.append(qm.FieldCondition(
                key="source_repos", match=qm.MatchValue(value=repo),
            ))
        if quality:
            must.append(qm.FieldCondition(
                key="quality_flag", match=qm.MatchValue(value=quality),
            ))
        if older_than_seconds is not None and older_than_seconds > 0:
            cutoff = time.time() - int(older_than_seconds)
            must.append(qm.FieldCondition(
                key="created_at", range=qm.Range(lt=cutoff),
            ))
        if question_substring:
            must.append(qm.FieldCondition(
                key="question", match=qm.MatchText(text=question_substring),
            ))

        if not must:
            return 0
        try:
            await self._client.delete(
                collection_name=self._collection,
                points_selector=qm.FilterSelector(filter=qm.Filter(must=must)),
                wait=False,
            )
            return -1
        except Exception as exc:
            _log.warning("cache purge failed (filters=%d): %s", len(must), exc)
            return 0

    async def flag_low_quality(self, question: str) -> None:
        """User feedback -- mark the cached entry low-quality so it's skipped.
        Triggered by thumbs-down in the UI (future)."""
        try:
            await self._client.set_payload(
                collection_name=self._collection,
                points=[_make_point_id(question)],
                payload={"quality_flag": "low"},
                wait=False,
            )
        except Exception as exc:
            _log.warning("cache flag_low_quality failed: %s", exc)

    async def stats(self) -> dict:
        try:
            await self.ensure_collection()
            info = await self._client.get_collection(self._collection)
            return {
                "name": self._collection,
                "points_count": info.points_count or 0,
                "threshold": self._threshold,
                "default_ttl_seconds": self._default_ttl,
            }
        except Exception as exc:
            _log.warning("cache stats failed: %s", exc)
            return {"name": self._collection, "error": str(exc)}


def _derive_repos(sources: list[str]) -> list[str]:
    """Pull `<owner>/<repo>` prefixes from source paths. Best-effort."""
    out: set[str] = set()
    for s in sources or []:
        if not s or s.startswith("<"):  # synthetic chunks
            continue
        parts = s.split("/")
        if len(parts) >= 2:
            # Heuristic: the first 2-4 segments form the repo name
            for take in (4, 3, 2):
                candidate = "/".join(parts[:take])
                if "/" in candidate and not candidate.endswith(("yaml", "yml", "md", "tf", "json")):
                    out.add(candidate)
                    break
    return sorted(out)


def is_enabled() -> bool:
    """Toggle via env var; default ON since cache is safe (always stores
    fresh entries on miss, skips for time-sensitive queries)."""
    val = os.environ.get("OPSRAG_QA_CACHE", "1")
    return val.lower() in ("1", "true", "yes", "on")
