"""Qdrant-backed cache of past investigation results.

Used by the hypothesis-driven investigation agent to inject relevant
prior investigations into the bootstrap context - same pattern as the
chat path's `opsrag.agent.cache.investigation_cache.InvestigationCache`,
but with a richer payload (alert_text, service hints, validated_chain
summary, outcome) tailored to the agent's hypothesis-gen prompt.

Lives in a SEPARATE Qdrant collection (`opsrag_agent_investigations`)
to keep chat-path data and agent-path data cleanly partitioned -
they're written by different code paths and tuned for different
retrieval thresholds.

Pattern reference: `opsrag/agent/cache/investigation_cache.py`.
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field

try:
    from qdrant_client import AsyncQdrantClient
    from qdrant_client.http import models as qm
except ImportError:  # pragma: no cover - Qdrant is a runtime dep
    AsyncQdrantClient = None  # type: ignore[assignment]
    qm = None  # type: ignore[assignment]

_log = logging.getLogger("opsrag.agents.investigation.result_cache")

DEFAULT_COLLECTION = "opsrag_agent_investigations"
DEFAULT_VECTOR_SIZE = 768  # text-embedding-005
DEFAULT_TOP_K = 3
# Tighter than chat's 0.85 because investigation alerts share a lot
# of boilerplate ([P2][prod/k8s/X] prefix, "High X requests" phrasing).
# We want true semantic match on the underlying mechanism.
DEFAULT_THRESHOLD = 0.78
# Past investigations decay over 60 days: a 60-day-old result is worth
# 50% of a fresh one for ranking purposes. Older results need a higher
# raw cosine to clear the threshold.
DECAY_HALF_LIFE_DAYS = 60


def _decay_factor(age_seconds: float) -> float:
    """Exponential decay: 1.0 fresh -> 0.5 at half-life -> ~0.0 at 4x HL."""
    if age_seconds <= 0:
        return 1.0
    age_days = age_seconds / 86400.0
    return 0.5 ** (age_days / DECAY_HALF_LIFE_DAYS)


@dataclass
class PastInvestigation:
    """One past-investigation entry returned by `search()`."""

    investigation_id: str
    alert_text: str
    service_hint: str
    namespace_hint: str
    env_hint: str
    final_root_cause: str
    outcome: str
    validated_chain_summary: list[str]  # one short line per validated node
    tool_calls_used: list[str]
    raw_similarity: float
    adjusted_similarity: float
    age_seconds: float
    feedback: dict = field(default_factory=dict)

    @property
    def age_days(self) -> int:
        return int(self.age_seconds / 86400)


class InvestigationResultCache:
    """Async Qdrant-backed store + search for agent investigation results.

    Embedding contract: caller is responsible for embedding the
    `(alert_text + service + namespace + env)` composite string with
    the same embedder used at query time, so cosine search is apples-
    to-apples.
    """

    def __init__(
        self,
        qdrant,
        collection: str = DEFAULT_COLLECTION,
        vector_size: int = DEFAULT_VECTOR_SIZE,
        threshold: float = DEFAULT_THRESHOLD,
        top_k: int = DEFAULT_TOP_K,
    ):
        self._qdrant = qdrant
        self._collection = collection
        self._vector_size = vector_size
        self._threshold = threshold
        self._top_k = top_k
        self._ensured = False

    async def ensure_collection(self) -> None:
        if self._ensured:
            return
        if qm is None:
            _log.warning("qdrant_client not importable - cache disabled")
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
                _log.info("created agent investigation cache %s", self._collection)
            except Exception as exc:
                _log.warning(
                    "investigation result cache ensure failed: %s - store/search no-op",
                    exc,
                )
                return
        self._ensured = True

    @staticmethod
    def compose_query(alert_text: str, service_hint: str | None, namespace_hint: str | None, env_hint: str | None) -> str:
        """The exact string we embed for both store + search. Including
        the hints scopes the cache so different services with similar
        alert templates don't collide."""
        parts = [alert_text.strip()]
        if service_hint:
            parts.append(f"service:{service_hint}")
        if namespace_hint:
            parts.append(f"namespace:{namespace_hint}")
        if env_hint:
            parts.append(f"env:{env_hint}")
        return " | ".join(parts)

    async def store(
        self,
        *,
        investigation_id: str,
        alert_text: str,
        service_hint: str | None,
        namespace_hint: str | None,
        env_hint: str | None,
        embedding: list[float],
        final_root_cause: str,
        outcome: str,
        validated_chain_summary: list[str],
        tool_calls_used: list[str],
        # Full state for history replay. Optional so existing callers
        # that only care about the bootstrap-context search keep working
        # unchanged.
        nodes_full: list[dict] | None = None,
        root_ids: list[str] | None = None,
        final_chain_node_ids: list[str] | None = None,
        bootstrap_findings: list[str] | None = None,
        summary: dict | None = None,
    ) -> str | None:
        """Persist one investigation result. Returns the qdrant point
        id, or None on failure. Caller must have computed the
        embedding using `compose_query()` for retrieval symmetry.

        Storage policy: ALL outcomes are persisted (so the history UI
        can show inconclusive / dead-end runs alongside validated ones).
        Retrieval-for-bootstrap-context (`search()`) still ranks by
        cosine + decay so validated runs naturally outrank others.
        """
        await self.ensure_collection()
        if not self._ensured or not embedding:
            return None
        point_id = str(uuid.uuid4())
        payload = {
            "investigation_id": investigation_id,
            "alert_text": alert_text,
            "service_hint": service_hint or "",
            "namespace_hint": namespace_hint or "",
            "env_hint": env_hint or "",
            "final_root_cause": final_root_cause,
            "outcome": outcome,
            "validated_chain_summary": validated_chain_summary,
            "tool_calls_used": tool_calls_used,
            "created_at": time.time(),
            "feedback": {},
            # Full state for history-replay (Tier B).
            "nodes_full": nodes_full or [],
            "root_ids": root_ids or [],
            "final_chain_node_ids": final_chain_node_ids or [],
            "bootstrap_findings": bootstrap_findings or [],
            "summary": summary or {},
        }
        try:
            await self._qdrant.upsert(
                collection_name=self._collection,
                points=[qm.PointStruct(id=point_id, vector=embedding, payload=payload)],
            )
            _log.info(
                "investigation result stored id=%s service=%s chain_len=%d tools=%d",
                point_id, service_hint, len(validated_chain_summary), len(tool_calls_used),
            )
            return point_id
        except Exception as exc:
            _log.warning("investigation result store failed: %s", exc)
            return None

    async def search(
        self,
        embedding: list[float],
        *,
        top_k: int | None = None,
        threshold: float | None = None,
    ) -> list[PastInvestigation]:
        """Top-K past investigations above `threshold` on decay-adjusted
        cosine. Empty list on miss.

        Algorithm (same shape as chat InvestigationCache):
          1. Pull 3x top_k by raw cosine.
          2. Multiply by exponential age decay.
          3. Sort by adjusted, drop sub-threshold, return top_k.
        """
        await self.ensure_collection()
        if not self._ensured or not embedding:
            return []
        k = top_k or self._top_k
        thr = threshold if threshold is not None else self._threshold
        wide_k = max(k * 3, k)
        try:
            # qdrant-client >= 1.10 deprecated `search` in favour of
            # `query_points`. .points carries .score + .payload identically.
            result = await self._qdrant.query_points(
                collection_name=self._collection,
                query=embedding,
                limit=wide_k,
                with_payload=True,
            )
            hits = result.points
        except Exception as exc:
            _log.warning("investigation result search failed: %s", exc)
            return []

        now = time.time()
        results: list[PastInvestigation] = []
        for h in hits:
            payload = h.payload or {}
            created_at = float(payload.get("created_at", now))
            age_seconds = max(0.0, now - created_at)
            adjusted = float(h.score) * _decay_factor(age_seconds)
            if adjusted < thr:
                continue
            results.append(PastInvestigation(
                investigation_id=str(payload.get("investigation_id", "")),
                alert_text=str(payload.get("alert_text", "")),
                service_hint=str(payload.get("service_hint", "")),
                namespace_hint=str(payload.get("namespace_hint", "")),
                env_hint=str(payload.get("env_hint", "")),
                final_root_cause=str(payload.get("final_root_cause", "")),
                outcome=str(payload.get("outcome", "")),
                validated_chain_summary=list(payload.get("validated_chain_summary") or []),
                tool_calls_used=list(payload.get("tool_calls_used") or []),
                raw_similarity=float(h.score),
                adjusted_similarity=adjusted,
                age_seconds=age_seconds,
                feedback=dict(payload.get("feedback") or {}),
            ))
        results.sort(key=lambda r: -r.adjusted_similarity)
        return results[:k]

    # --- History API (Tier B replay) ---------------------------------

    async def list_recent(self, limit: int = 50) -> list[dict]:
        """Return up to `limit` investigations sorted by `created_at`
        desc. Lightweight payload (no full node tree) for the sidebar
        listing; call `get_full()` to fetch one for replay.
        """
        await self.ensure_collection()
        if not self._ensured:
            return []
        # Pull a generous window then sort in Python - Qdrant scroll
        # doesn't natively sort, and the cache rarely exceeds a few
        # thousand points so this is cheap.
        items: list[dict] = []
        try:
            offset = None
            page_cap = max(limit * 4, 200)
            while len(items) < page_cap:
                page, next_off = await self._qdrant.scroll(
                    collection_name=self._collection,
                    limit=min(256, page_cap - len(items)),
                    with_payload=True,
                    with_vectors=False,
                    offset=offset,
                )
                for pt in page or []:
                    p = pt.payload or {}
                    items.append({
                        "investigation_id": str(p.get("investigation_id", "")),
                        "alert_text": str(p.get("alert_text", "")),
                        "service_hint": str(p.get("service_hint", "")),
                        "namespace_hint": str(p.get("namespace_hint", "")),
                        "env_hint": str(p.get("env_hint", "")),
                        "outcome": str(p.get("outcome", "")),
                        "final_root_cause": str(p.get("final_root_cause", "")),
                        "created_at": float(p.get("created_at", 0.0)),
                    })
                if not next_off:
                    break
                offset = next_off
        except Exception as exc:
            _log.warning("investigation list_recent failed: %s", exc)
            return []
        items.sort(key=lambda r: -r["created_at"])
        return items[:limit]

    async def get_full(self, investigation_id: str) -> dict | None:
        """Fetch one investigation by `investigation_id` payload key.
        Returns the full payload (nodes, edges, summary, etc.) needed
        to rehydrate the tree visualization on the UI side. Returns
        None on miss.
        """
        await self.ensure_collection()
        if not self._ensured or not investigation_id:
            return None
        try:
            flt = qm.Filter(must=[
                qm.FieldCondition(
                    key="investigation_id",
                    match=qm.MatchValue(value=investigation_id),
                )
            ])
            page, _ = await self._qdrant.scroll(
                collection_name=self._collection,
                scroll_filter=flt,
                limit=1,
                with_payload=True,
                with_vectors=False,
            )
            if not page:
                return None
            return dict(page[0].payload or {})
        except Exception as exc:
            _log.warning("investigation get_full(%s) failed: %s", investigation_id, exc)
            return None
