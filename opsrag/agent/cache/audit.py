"""V1 audit tooling for the investigation cache.

Companion to `investigation_cache.py`. Functions surface stale or
low-quality investigations for manual review. V2 will extend with tag
quality + embedding consistency checks.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from qdrant_client import AsyncQdrantClient

from opsrag.agent.cache.investigation_cache import (
    DEFAULT_INVESTIGATION_COLLECTION,
    decay_factor_for_age,
)

_log = logging.getLogger("opsrag.agent.cache.audit")

_STALE_AGE_DAYS = 180


async def list_low_quality_investigations(
    qdrant: AsyncQdrantClient,
    *,
    collection: str = DEFAULT_INVESTIGATION_COLLECTION,
    min_score: int = 0,
    limit: int = 50,
) -> list[dict]:
    """Investigations with `thumbs_down > thumbs_up` OR with corrections
    recorded. Useful for retraining / re-investigating bad answers.

    `min_score` filters out items where the negative-positive delta is
    too small to be meaningful (default 0 = any net-negative).
    """
    out: list[dict] = []
    cursor = None
    for _ in range(50):  # safety bound
        try:
            points, cursor = await qdrant.scroll(
                collection_name=collection,
                limit=200,
                offset=cursor,
                with_payload=True,
            )
        except Exception as exc:
            _log.warning("scroll failed: %s", exc)
            return out
        for p in points:
            payload = p.payload or {}
            fb = payload.get("feedback") or {}
            up = int(fb.get("up", 0))
            down = int(fb.get("down", 0))
            corrections = fb.get("corrections") or []
            net_negative = down - up
            has_corrections = bool(corrections)
            if (net_negative >= min_score and net_negative > 0) or has_corrections:
                out.append({
                    "id": str(p.id),
                    "question": payload.get("question", ""),
                    "answer": (payload.get("answer", "") or "")[:300],
                    "feedback": fb,
                    "created_at": payload.get("created_at"),
                    "thread_id": payload.get("thread_id", ""),
                })
                if len(out) >= limit:
                    return out
        if not cursor:
            break
    return out


async def list_stale_investigations(
    qdrant: AsyncQdrantClient,
    *,
    collection: str = DEFAULT_INVESTIGATION_COLLECTION,
    older_than_days: float = _STALE_AGE_DAYS,
    limit: int = 50,
) -> list[dict]:
    """Investigations older than `older_than_days`. These are floor-decayed
    in search ranking, so their useful lifetime is essentially over.
    Surface for periodic cleanup or re-investigation.
    """
    out: list[dict] = []
    cursor = None
    cutoff = time.time() - (older_than_days * 86400)
    for _ in range(50):
        try:
            points, cursor = await qdrant.scroll(
                collection_name=collection,
                limit=200,
                offset=cursor,
                with_payload=True,
            )
        except Exception as exc:
            _log.warning("scroll failed: %s", exc)
            return out
        for p in points:
            payload = p.payload or {}
            created_at = float(payload.get("created_at") or 0)
            if created_at and created_at < cutoff:
                age_days = (time.time() - created_at) / 86400
                out.append({
                    "id": str(p.id),
                    "question": payload.get("question", ""),
                    "age_days": round(age_days, 1),
                    "decay_factor": round(decay_factor_for_age(time.time() - created_at), 3),
                    "feedback": payload.get("feedback", {}) or {},
                })
                if len(out) >= limit:
                    return out
        if not cursor:
            break
    out.sort(key=lambda x: x["age_days"], reverse=True)
    return out


async def cache_summary(
    qdrant: AsyncQdrantClient,
    *,
    collection: str = DEFAULT_INVESTIGATION_COLLECTION,
) -> dict[str, Any]:
    """Top-level metrics for the investigation cache."""
    try:
        info = await qdrant.get_collection(collection)
    except Exception:
        return {"total": 0, "available": False}
    total = int(info.points_count or 0)
    if total == 0:
        return {"total": 0, "available": True, "stale": 0, "low_quality": 0}
    stale = await list_stale_investigations(qdrant, collection=collection, limit=10000)
    low = await list_low_quality_investigations(qdrant, collection=collection, limit=10000)
    return {
        "total": total,
        "available": True,
        "stale_older_than_days": _STALE_AGE_DAYS,
        "stale": len(stale),
        "low_quality": len(low),
    }
