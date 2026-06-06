"""Hook into OpsRAG's existing `/usage` endpoint to track per-eval-test cost.

Used by conftest.py to enforce session + per-test cost ceilings. Reads
total cumulative cost from the running OpsRAG container -- eval calls
must hit the live API for retrieval, and judge calls hit Vertex Gemini
Pro directly (separate billing path, see vertex_judge.py).
"""
from __future__ import annotations

import logging
import os

import httpx

_log = logging.getLogger("opsrag.eval.usage_hook")
_OPSRAG_URL = os.environ.get("OPSRAG_URL", "http://localhost:8000")


def get_usage_total() -> float:
    """Return cumulative USD cost from /usage endpoint, or 0.0 on error."""
    try:
        resp = httpx.get(f"{_OPSRAG_URL}/usage", timeout=5.0)
        resp.raise_for_status()
        return float(resp.json().get("total_estimated_cost_usd", 0.0))
    except Exception as exc:
        _log.warning("usage hook failed: %s", exc)
        return 0.0
