"""Sub-sprint 3 V1 -- Qdrant-backed investigation cache.

Stores tool-path investigation outcomes so future similar questions
can reuse the prior reasoning instead of re-drilling from scratch.

Distinct from `opsrag.qa_cache` (which serves cached corpus answers
verbatim above a high cosine threshold). This cache is:

- Looser threshold (`DEFAULT_INVESTIGATION_THRESHOLD = 0.85`) -- past
  investigations are templates to ADAPT from, not exact answers.
- Top-K returned (default 3), not single-best -- the reasoner reads
  several past investigations as context.
- Stores the FULL audit (tool calls, args, latencies, errors) +
  the synthesized answer + model route decision, not just Q + A.
- No automatic short-circuit -- the reasoner decides how much past
  context to use; we never bypass the agent on a hit.

V2 (deferred): tag taxonomy, funneled search, confidence decay.

Schema (Qdrant point payload):
  question:        str
  answer:          str
  tool_call_audit: list[dict]
  model_route_decision: dict
  thread_id:       str
  user_id:         str
  user_scope:      str (present ONLY when the answer wove in per-user
                   memories -- absent for shared investigations)
  created_at:      float (unix seconds)

Cross-user scoping: investigations are SHARED by default (matching the
documented shared / scope-gated authz model -- any user may learn from
another user's tool-path reasoning). The one carve-out mirrors
`opsrag.qa_cache`: when an answer wove in per-user Mem0 memories it is
stamped with ``user_scope`` and returned ONLY to its author, so a
recalled personal fact never leaks into another user's reasoning on a
high cosine match.
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field

from qdrant_client import AsyncQdrantClient
from qdrant_client import models as qm

_log = logging.getLogger("opsrag.agent.cache.investigation")

DEFAULT_INVESTIGATION_COLLECTION = "opsrag_investigations"
DEFAULT_INVESTIGATION_THRESHOLD = 0.85
DEFAULT_INVESTIGATION_TOP_K = 3
DEFAULT_VECTOR_SIZE = 768  # text-embedding-005

# V1 confidence decay -- 6-month half-life floor at 50%, per the user's
# spec: `decay_factor = max(0.5, 1.0 - (age_days / 180))`. Applied to
# raw cosine similarity before the threshold check, so an old hit needs
# higher raw similarity to clear the same threshold.
DECAY_FLOOR = 0.5
DECAY_HALF_LIFE_DAYS = 180.0


def adjusted_similarity(raw_similarity: float, age_seconds: float) -> float:
    """V1 confidence decay. Linear decay from 1.0 at age 0 down to
    DECAY_FLOOR at DECAY_HALF_LIFE_DAYS, floored thereafter."""
    age_days = max(0.0, float(age_seconds) / 86400.0)
    decay = max(DECAY_FLOOR, 1.0 - (age_days / DECAY_HALF_LIFE_DAYS))
    return float(raw_similarity) * decay


def decay_factor_for_age(age_seconds: float) -> float:
    """The multiplier applied to raw similarity for an investigation of
    the given age. Used by `audit` tooling and tests."""
    age_days = max(0.0, float(age_seconds) / 86400.0)
    return max(DECAY_FLOOR, 1.0 - (age_days / DECAY_HALF_LIFE_DAYS))


@dataclass
class InvestigationHit:
    """One past investigation matching the current query.

    `similarity` is the raw cosine score from Qdrant; `decayed_similarity`
    is the age-adjusted score the search uses for ranking + threshold
    comparison. `feedback` summarizes thumbs-up / thumbs-down if any."""
    question: str
    answer: str
    tool_call_audit: list[dict] = field(default_factory=list)
    model_route_decision: dict = field(default_factory=dict)
    similarity: float = 0.0
    decayed_similarity: float = 0.0
    age_seconds: float = 0.0
    thread_id: str = ""
    investigation_id: str = ""
    feedback: dict = field(default_factory=dict)
    # Author of the originating turn. `user_scope` is set ONLY when the
    # answer wove in per-user memories -- such hits are returned only to
    # this same `user_scope` (see `search`). Exposed so callers/tests can
    # observe the cross-user scoping decision.
    user_id: str = ""
    user_scope: str | None = None


class InvestigationCache:
    """Async Qdrant-backed store + search for tool-path investigations."""

    def __init__(
        self,
        qdrant: AsyncQdrantClient,
        collection: str = DEFAULT_INVESTIGATION_COLLECTION,
        vector_size: int = DEFAULT_VECTOR_SIZE,
        threshold: float = DEFAULT_INVESTIGATION_THRESHOLD,
        top_k: int = DEFAULT_INVESTIGATION_TOP_K,
    ):
        self._qdrant = qdrant
        self._collection = collection
        self._vector_size = vector_size
        self._threshold = threshold
        self._top_k = top_k
        self._ensured = False

    async def _ensure_collection(self) -> None:
        if self._ensured:
            return
        try:
            await self._qdrant.get_collection(self._collection)
        except Exception:
            try:
                await self._qdrant.create_collection(
                    collection_name=self._collection,
                    vectors_config=qm.VectorParams(
                        size=self._vector_size, distance=qm.Distance.COSINE,
                    ),
                )
                # Index user_scope so the shared-or-mine search filter is
                # served efficiently (mirrors qa_cache).
                try:
                    await self._qdrant.create_payload_index(
                        collection_name=self._collection,
                        field_name="user_scope",
                        field_schema=qm.PayloadSchemaType.KEYWORD,
                    )
                except Exception:
                    pass
                _log.info("created investigation cache collection %s", self._collection)
            except Exception as exc:
                _log.warning(
                    "investigation cache collection ensure failed: %s -- search/store will no-op",
                    exc,
                )
                return
        self._ensured = True

    async def store(
        self,
        *,
        question: str,
        embedding: list[float],
        answer: str,
        tool_call_audit: list[dict] | None = None,
        model_route_decision: dict | None = None,
        thread_id: str = "",
        user_id: str = "",
        user_scope: str | None = None,
    ) -> str | None:
        """Store one investigation. Returns the point id or None on
        failure. Idempotent on re-call (each call generates a new uuid).

        ``user_scope`` is set ONLY for answers that wove in per-user
        memories (Mem0); pass ``user_id`` as the scope in that case
        (mirrors the graph.py ``_user_scope = user_id if
        result.get("user_memories") else None`` pattern). A scoped entry
        is returned only to that same user (see ``search``); shared
        investigations leave it unset so any user can learn from them."""
        await self._ensure_collection()
        if not self._ensured:
            return None
        if not embedding:
            return None
        point_id = str(uuid.uuid4())
        payload = {
            "question": question,
            "answer": answer,
            "tool_call_audit": tool_call_audit or [],
            "model_route_decision": model_route_decision or {},
            "thread_id": thread_id,
            "user_id": user_id,
            "created_at": time.time(),
        }
        # Only stamp user_scope when the answer is user-specific. Leaving
        # it absent (not null) keeps the IsEmpty("shared") search filter
        # matching legacy + shared entries.
        if user_scope:
            payload["user_scope"] = user_scope
        try:
            await self._qdrant.upsert(
                collection_name=self._collection,
                points=[qm.PointStruct(id=point_id, vector=embedding, payload=payload)],
            )
            _log.info(
                "investigation stored id=%s thread=%s tools=%d answer_chars=%d",
                point_id, thread_id, len(tool_call_audit or []), len(answer),
            )
            return point_id
        except Exception as exc:
            _log.warning("investigation store failed: %s", exc)
            return None

    async def search(
        self,
        embedding: list[float],
        *,
        top_k: int | None = None,
        threshold: float | None = None,
        user_id: str | None = None,
    ) -> list[InvestigationHit]:
        """Top-K past investigations above `threshold` (compared against
        DECAY-ADJUSTED similarity). Empty list on miss.

        Algorithm:
          1. Pull a wider Qdrant top-K' (3x requested) by raw cosine,
             pre-filtered by the shared-or-mine scope clause.
          2. Apply V1 confidence decay: `adjusted = raw x decay(age)`.
          3. Re-sort by adjusted score, drop entries below `threshold`,
             return up to `top_k`.

        Effect: a 6-month-old 0.95-cosine hit decays to 0.475 and falls
        below the 0.85 threshold, while a 1-day-old 0.86 stays at ~0.86
        and gets through. Older investigations need higher raw
        similarity to clear the same bar.

        `user_id` scopes the search the same way `qa_cache.lookup` does:
        the caller is eligible for SHARED investigations (no `user_scope`)
        plus those scoped to this same user. A memory-influenced
        investigation cached for another user is never returned here --
        otherwise its answer + tool_call_audit would leak that user's
        personal facts into a different user's reasoning. When `user_id`
        is None only shared investigations are eligible.
        """
        await self._ensure_collection()
        if not self._ensured or not embedding:
            return []
        k = top_k or self._top_k
        thr = threshold if threshold is not None else self._threshold
        # Pull wider so decay can pull in older but high-raw hits the
        # threshold-on-raw filter would have included.
        wide_k = max(k * 3, k)
        # Shared-or-mine filter. IsEmpty matches legacy + shared entries;
        # the match clause adds this user's own scoped entries. Without a
        # user_id only shared entries are eligible.
        shared = qm.IsEmptyCondition(is_empty=qm.PayloadField(key="user_scope"))
        if user_id:
            query_filter = qm.Filter(should=[
                shared,
                qm.FieldCondition(key="user_scope", match=qm.MatchValue(value=user_id)),
            ])
        else:
            query_filter = qm.Filter(must=[shared])
        try:
            # qdrant-client >= 1.10 deprecated `search`. `query_points` is the
            # current API and returns the same .points list with .score/.payload.
            # Don't pre-filter by threshold here -- we re-rank on decay-adjusted.
            result = await self._qdrant.query_points(
                collection_name=self._collection,
                query=embedding,
                limit=wide_k,
                with_payload=True,
                query_filter=query_filter,
            )
            hits = result.points
        except Exception as exc:
            _log.warning("investigation search failed: %s", exc)
            return []

        now = time.time()
        out: list[InvestigationHit] = []
        for h in hits:
            p = h.payload or {}
            age_seconds = max(0.0, now - float(p.get("created_at") or now))
            raw = float(h.score)
            adjusted = adjusted_similarity(raw, age_seconds)
            if adjusted < thr:
                continue
            out.append(InvestigationHit(
                question=p.get("question", ""),
                answer=p.get("answer", ""),
                tool_call_audit=p.get("tool_call_audit", []) or [],
                model_route_decision=p.get("model_route_decision", {}) or {},
                similarity=raw,
                decayed_similarity=adjusted,
                age_seconds=age_seconds,
                thread_id=p.get("thread_id", ""),
                investigation_id=str(h.id),
                feedback=p.get("feedback", {}) or {},
                user_id=p.get("user_id", "") or "",
                user_scope=p.get("user_scope"),
            ))
        out.sort(key=lambda x: x.decayed_similarity, reverse=True)
        out = out[:k]
        if out:
            _log.info(
                "investigation search hits=%d top_raw=%.3f top_decayed=%.3f age_d=%.1f",
                len(out), out[0].similarity, out[0].decayed_similarity,
                out[0].age_seconds / 86400,
            )
        return out

    async def record_feedback(
        self,
        investigation_id: str,
        *,
        thumbs: str,
        correction: str | None = None,
    ) -> bool:
        """Attach thumbs-up / thumbs-down feedback to a past investigation. `thumbs` must
        be `"up"` or `"down"`. Optional free-text `correction` lets the
        user explain why an answer was wrong (used by V2 audit). Stored
        on the existing point's payload (no separate collection)."""
        if thumbs not in ("up", "down"):
            return False
        await self._ensure_collection()
        if not self._ensured:
            return False
        # Read existing payload + feedback dict, append, re-store.
        try:
            points = await self._qdrant.retrieve(
                collection_name=self._collection,
                ids=[investigation_id],
                with_payload=True,
            )
        except Exception as exc:
            _log.warning("retrieve for feedback failed id=%s: %s", investigation_id, exc)
            return False
        if not points:
            return False
        payload = points[0].payload or {}
        feedback = payload.get("feedback") or {"up": 0, "down": 0, "corrections": []}
        if thumbs == "up":
            feedback["up"] = int(feedback.get("up", 0)) + 1
        else:
            feedback["down"] = int(feedback.get("down", 0)) + 1
        if correction:
            corrections = list(feedback.get("corrections") or [])
            corrections.append({"text": correction[:1000], "ts": time.time()})
            feedback["corrections"] = corrections[:20]  # cap at 20

        try:
            await self._qdrant.set_payload(
                collection_name=self._collection,
                payload={"feedback": feedback},
                points=[investigation_id],
            )
            _log.info(
                "feedback recorded id=%s thumbs=%s up=%d down=%d",
                investigation_id, thumbs, feedback.get("up", 0), feedback.get("down", 0),
            )
            return True
        except Exception as exc:
            _log.warning("set_payload feedback failed id=%s: %s", investigation_id, exc)
            return False

    async def purge(
        self,
        *,
        all: bool = False,
        older_than_seconds: int | None = None,
        thumbs_down_only: bool = False,
        question_substring: str | None = None,
    ) -> int:
        """Multi-strategy purge for investigation cache. Mirrors the
        Q&A cache surface but with investigation-specific filters
        (thumbs-down feedback). Returns count purged or -1 if Qdrant
        doesn't return one."""
        await self._ensure_collection()
        if not self._ensured:
            return 0
        if all:
            try:
                await self._qdrant.delete_collection(self._collection)
                self._ensured = False
                await self._ensure_collection()
                return -1
            except Exception as exc:
                _log.warning("investigation purge all failed: %s", exc)
                return 0
        must: list = []
        if older_than_seconds is not None and older_than_seconds > 0:
            cutoff = time.time() - int(older_than_seconds)
            must.append(qm.FieldCondition(
                key="created_at", range=qm.Range(lt=cutoff),
            ))
        if thumbs_down_only:
            must.append(qm.FieldCondition(
                key="feedback.down", range=qm.Range(gt=0),
            ))
        if question_substring:
            must.append(qm.FieldCondition(
                key="question", match=qm.MatchText(text=question_substring),
            ))
        if not must:
            return 0
        try:
            await self._qdrant.delete(
                collection_name=self._collection,
                points_selector=qm.FilterSelector(filter=qm.Filter(must=must)),
            )
            return -1
        except Exception as exc:
            _log.warning("investigation purge failed (filters=%d): %s", len(must), exc)
            return 0

    async def count(self) -> int:
        """Total investigations cached. Used for V2 30-investigation gate."""
        await self._ensure_collection()
        if not self._ensured:
            return 0
        try:
            info = await self._qdrant.get_collection(self._collection)
            return int(info.points_count or 0)
        except Exception:
            return 0
