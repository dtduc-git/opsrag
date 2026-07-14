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

# Keep in sync with opsrag/mcp/runbooks._STORE_ID_PREFIX (duplicated
# deliberately -- importing it would couple agent/cache -> mcp).
_RB_PREFIX = "rb-"

# Thread-id resolver scroll page size (module-level so tests can shrink it).
_RESOLVE_PAGE_LIMIT = 256


def extract_loaded_runbook_ids(tool_call_audit: list[dict] | None) -> list[str]:
    """rb-ids of every TAB-store runbook a stored answer successfully
    loaded: audit rows with name==runbook_load, no error, args.name=rb-*.
    Order-preserving, deduped. This is what lets answer feedback credit
    the runbooks that produced the answer.

    Known limit: runbook_load reports "runbook not found" as an error
    PAYLOAD (no exception), which the audit records success-shaped -- such
    phantom ids pass this filter but are harmless downstream
    (record_thumbs UPDATE matches 0 rows / warns on a non-UUID id)."""
    out: list[str] = []
    seen: set[str] = set()
    for row in tool_call_audit or []:
        if not isinstance(row, dict):
            continue
        if row.get("name") != "runbook_load" or "error" in row:
            continue
        name = str((row.get("args") or {}).get("name") or "")
        if name.startswith(_RB_PREFIX):
            rb_id = name[len(_RB_PREFIX):]
            if rb_id and rb_id not in seen:
                seen.add(rb_id)
                out.append(rb_id)
    return out


@dataclass
class FeedbackResult:
    """Outcome of record_feedback. Truthiness mirrors the old bool return
    so callers that only check `if result:` keep working unchanged."""

    ok: bool
    point_id: str | None = None
    runbook_ids: list[str] = field(default_factory=list)
    # The resolved investigation's question + answer, surfaced so the
    # feedback audit row (Postgres opsrag_feedback) can show WHAT was rated.
    # Slack/channel feedback used to persist thumbs only -> blank dashboard.
    query: str | None = None
    answer: str | None = None

    def __bool__(self) -> bool:
        return self.ok

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
        # thread_id KEYWORD index -- OUTSIDE the create-only branch so
        # pre-existing deployments get it too (idempotent in Qdrant). It
        # serves the feedback thread-id resolver; even if it fails, the
        # scroll below still works as a (small-collection) full scan.
        try:
            await self._qdrant.create_payload_index(
                collection_name=self._collection,
                field_name="thread_id",
                field_schema=qm.PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass
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

    async def _resolve_point(
        self, investigation_id: str, *, answer_snippet: str | None = None,
    ) -> tuple[str, dict] | None:
        """Resolve a feedback target to (point_id, payload).

        The UI can only send the REAL point UUID for live tool-path
        answers -- cache-hit answers and replayed sessions never received
        one, so the FE falls back to the THREAD-shaped id
        (``<uuid>_<8hex>``). A raw ``retrieve`` on that shape is a Qdrant
        400, which used to silently drop the feedback. Non-UUID ids are
        resolved here by the ``thread_id`` payload field instead; when a
        thread has several stored turns, an ``answer_snippet`` substring
        match picks the thumbed turn. A PROVIDED snippet that matches no
        stored answer means the thumbed turn was never stored under this
        thread (qa-cache hits store nothing; SWR refreshes store under
        ``<thread>__swr``) -- refuse rather than misattribute the thumbs
        + correction to an unrelated prior turn. Newest-wins only when
        no snippet was sent (Slack/channel lanes)."""
        try:
            uuid.UUID(investigation_id)
        except ValueError:
            # Thread-shaped id -> filter by the stored thread_id payload.
            # Paginate fully: point ids are random uuid4s, so scroll order
            # is uncorrelated with created_at and a single page could miss
            # the newest/matching turn on long threads.
            candidates: list[tuple[str, dict]] = []
            offset = None
            try:
                while True:
                    points, offset = await self._qdrant.scroll(
                        collection_name=self._collection,
                        scroll_filter=qm.Filter(must=[
                            qm.FieldCondition(
                                key="thread_id",
                                match=qm.MatchValue(value=investigation_id),
                            ),
                        ]),
                        limit=_RESOLVE_PAGE_LIMIT,
                        offset=offset,
                        with_payload=True,
                    )
                    candidates.extend((str(p.id), p.payload or {}) for p in points)
                    if offset is None:
                        break
            except Exception as exc:
                _log.warning(
                    "thread-id resolve for feedback failed id=%s: %s",
                    investigation_id, exc,
                )
                return None
            if not candidates:
                return None
            if answer_snippet:
                candidates = [
                    c for c in candidates
                    if answer_snippet in str(c[1].get("answer") or "")
                ]
                if not candidates:
                    return None
            return max(candidates, key=lambda c: float(c[1].get("created_at") or 0.0))

        try:
            points = await self._qdrant.retrieve(
                collection_name=self._collection,
                ids=[investigation_id],
                with_payload=True,
            )
        except Exception as exc:
            _log.warning("retrieve for feedback failed id=%s: %s", investigation_id, exc)
            return None
        if not points:
            return None
        return str(points[0].id), points[0].payload or {}

    async def record_feedback(
        self,
        investigation_id: str,
        *,
        thumbs: str,
        correction: str | None = None,
        answer_snippet: str | None = None,
    ) -> FeedbackResult:
        """Attach thumbs-up / thumbs-down feedback to a past investigation. `thumbs` must
        be `"up"` or `"down"`. Optional free-text `correction` lets the
        user explain why an answer was wrong (used by V2 audit). Stored
        on the existing point's payload (no separate collection).

        Accepts the real point UUID or a thread-shaped id (see
        ``_resolve_point``). Returns a truthy/falsy ``FeedbackResult``
        carrying the resolved point id and the rb-ids of every tab
        runbook the answer loaded -- callers use those to credit the
        runbooks' thumbs counters."""
        if thumbs not in ("up", "down"):
            return FeedbackResult(ok=False)
        await self._ensure_collection()
        if not self._ensured:
            return FeedbackResult(ok=False)
        resolved = await self._resolve_point(
            investigation_id, answer_snippet=answer_snippet,
        )
        if resolved is None:
            return FeedbackResult(ok=False)
        point_id, payload = resolved
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
                points=[point_id],
            )
            _log.info(
                "feedback recorded id=%s (requested=%s) thumbs=%s up=%d down=%d",
                point_id, investigation_id, thumbs,
                feedback.get("up", 0), feedback.get("down", 0),
            )
            return FeedbackResult(
                ok=True,
                point_id=point_id,
                runbook_ids=extract_loaded_runbook_ids(payload.get("tool_call_audit")),
                query=(payload.get("question") or None),
                answer=(payload.get("answer") or None),
            )
        except Exception as exc:
            _log.warning("set_payload feedback failed id=%s: %s", point_id, exc)
            return FeedbackResult(ok=False)

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
