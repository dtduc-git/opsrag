"""Token usage tracker -- aggregates LLM token consumption and cost estimates.

Thread-safe atomic counters. Tracks per-model and per-session usage.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

# Cost per 1M tokens (input/output) -- approximate public pricing
_PRICING: dict[str, tuple[float, float]] = {
    # Anthropic
    "claude-sonnet-4-20250514": (3.0, 15.0),
    "claude-opus-4-20250514": (15.0, 75.0),
    "claude-haiku-4-20250514": (0.8, 4.0),
    # Vertex AI (same Claude models via Vertex)
    "claude-sonnet-4@20250514": (3.0, 15.0),
    # Gemini
    "gemini-2.0-flash": (0.075, 0.3),
    "gemini-2.5-flash": (0.15, 0.6),
    "gemini-2.5-pro": (1.25, 10.0),
    "gemini-2.5-flash-lite": (0.075, 0.3),
    # OpenAI
    "gpt-4o": (2.5, 10.0),
    "gpt-4o-mini": (0.15, 0.6),
    # Bedrock (Claude pricing is the same)
    "anthropic.claude-sonnet-4-20250514-v1:0": (3.0, 15.0),
    # Current Claude generation (Bedrock ids; region-prefixed inference
    # profiles like "us.anthropic.*" normalize to these -- see _pricing_for).
    "anthropic.claude-opus-4-8": (15.0, 75.0),
    "anthropic.claude-sonnet-4-6": (3.0, 15.0),   # Sonnet 4.6 (reason/pro model)
    "anthropic.claude-haiku-4-5-20251001-v1:0": (1.0, 5.0),
    # Embeddings -- per 1M tokens
    "text-embedding-3-large": (0.13, 0.0),
    "text-embedding-3-small": (0.02, 0.0),
    "text-embedding-005": (0.025, 0.0),
    "amazon.titan-embed-text-v2:0": (0.02, 0.0),
    # Cohere Embed v4 (Bedrock): $0.12 / 1M input tokens, no output.
    # Telemetry sees "us.cohere.embed-v4:0"; _pricing_for strips the
    # "us." region prefix to this base id. Both forms listed for safety.
    "cohere.embed-v4:0": (0.12, 0.0),
    "us.cohere.embed-v4:0": (0.12, 0.0),
}

# Per-call pricing for models charged per request rather than per token
# (Vertex Discovery Engine ranker, Bedrock Rerank API). USD per call.
_PRICING_PER_CALL: dict[str, float] = {
    # Vertex semantic ranker -- public pricing is ~$1 per 1K rank requests
    # (each request can rerank up to 200 records). Approximate; refine
    # when actual GCP bill numbers come in.
    "semantic-ranker-default-004": 0.001,
    "semantic-ranker-default-003": 0.001,
    # Bedrock Rerank API -- ~$1 per 1K rerank queries (per request, any model).
    "cohere.rerank-v3-5:0": 0.001,
    "amazon.rerank-v1:0": 0.001,
}

# Cross-region inference-profile prefixes (e.g. "us.anthropic.claude-...").
# Stripped before a pricing lookup so the profile id maps to the base model.
_REGION_PREFIXES = ("us.", "eu.", "apac.")


def _normalize_model(model: str) -> str:
    for p in _REGION_PREFIXES:
        if model.startswith(p):
            return model[len(p):]
    return model


def _pricing_for(model: str) -> tuple[float, float]:
    """(input, output) USD per 1M tokens. Tries exact, the region-stripped
    id, then a suffix match (defensive). (0, 0) when unknown."""
    for cand in (model, _normalize_model(model)):
        if cand in _PRICING:
            return _PRICING[cand]
    for key, val in _PRICING.items():
        if model.endswith(key):
            return val
    return (0.0, 0.0)


def _per_call_for(model: str) -> float:
    for cand in (model, _normalize_model(model)):
        if cand in _PRICING_PER_CALL:
            return _PRICING_PER_CALL[cand]
    return 0.0


# Purpose tags split usage into the cost categories the UI surfaces.
# Indexing-side: embed-index, contextual-chunk. Query-side: generation,
# embed-query, rerank, query-rewrite, grade, route, hallucination-check.
# `unknown` covers calls not yet tagged.
_INDEXING_PURPOSES = frozenset({"embed-index", "contextual-chunk"})


@dataclass
class PurposeUsage:
    """Per-purpose slice of a model's usage. Lets the UI separate
    indexing cost from query cost without splitting models."""

    call_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_latency_ms: float = 0.0


@dataclass
class ModelUsage:
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    call_count: int = 0
    total_latency_ms: float = 0.0
    first_call: float = 0.0
    last_call: float = 0.0
    by_purpose: dict[str, PurposeUsage] = field(default_factory=dict)


class UsageTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._models: dict[str, ModelUsage] = {}
        self._sessions: dict[str, dict[str, int]] = {}
        self._start_time = time.time()
        # Optional persistence sink. Wired by the lifespan once the
        # Postgres pool is open. Sync callable so record() never has to
        # await; the sink itself buffers and flushes on a background
        # task. Failures are swallowed -- telemetry must never break
        # the path that produced it.
        self._persist_hook: callable | None = None

    def set_persistence_hook(self, fn) -> None:
        self._persist_hook = fn

    def record(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float = 0.0,
        session_id: str | None = None,
        purpose: str | None = None,
        user_oid: str | None = None,
    ) -> None:
        """Record a model call.

        `purpose` tags the call's cost category -- `embed-index` /
        `embed-query` / `rerank` / `generation` / `query-rewrite` /
        `contextual-chunk` / `grade` / `route` / `hallucination-check`.
        Untagged calls land in `unknown`.
        """
        now = time.time()
        bucket = purpose or "unknown"

        # M2 -- per-user attribution. Provider call sites (bedrock /
        # vertex / openai / litellm + the embedders) don't thread the
        # request's identity through every `record()` call, so fall back
        # to the request-scoped contextvar set by the query handler.
        # This is what lets the "By user" / "Mine" dashboards populate.
        # Best-effort: any import/lookup failure leaves user_oid as-is so
        # telemetry never breaks the path that produced it.
        if user_oid is None:
            try:
                from opsrag.auth import current_user_oid_var
                user_oid = current_user_oid_var.get()
            except Exception:
                pass
        with self._lock:
            if model not in self._models:
                self._models[model] = ModelUsage(model=model, first_call=now)
            m = self._models[model]
            m.input_tokens += input_tokens
            m.output_tokens += output_tokens
            m.call_count += 1
            m.total_latency_ms += latency_ms
            m.last_call = now

            pu = m.by_purpose.setdefault(bucket, PurposeUsage())
            pu.call_count += 1
            pu.input_tokens += input_tokens
            pu.output_tokens += output_tokens
            pu.total_latency_ms += latency_ms

            if session_id:
                sess = self._sessions.setdefault(session_id, {})
                sess["input_tokens"] = sess.get("input_tokens", 0) + input_tokens
                sess["output_tokens"] = sess.get("output_tokens", 0) + output_tokens
                sess["call_count"] = sess.get("call_count", 0) + 1

        # Persistence: outside the lock so a slow buffer enqueue can't
        # block other recorders. The hook itself is sync + non-blocking
        # (an in-memory append). Errors are swallowed.
        if self._persist_hook is not None:
            try:
                self._persist_hook(
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_ms=latency_ms,
                    session_id=session_id,
                    purpose=bucket,
                    user_oid=user_oid,
                )
            except Exception:
                pass

    def seed_historical(
        self,
        model: str,
        purpose: str,
        input_tokens: int,
        output_tokens: int,
        call_count: int,
        latency_ms: float,
    ) -> None:
        """Backfill historical totals from persistent storage at startup.

        Mirrors `record()` but writes pre-aggregated counts instead of
        per-call increments. Safe to call multiple times for the same
        `(model, purpose)` -- totals are additive -- but the lifespan only
        runs it once with already-summed numbers.
        """
        with self._lock:
            if model not in self._models:
                self._models[model] = ModelUsage(model=model, first_call=time.time())
            m = self._models[model]
            m.input_tokens += input_tokens
            m.output_tokens += output_tokens
            m.call_count += call_count
            m.total_latency_ms += latency_ms

            pu = m.by_purpose.setdefault(purpose, PurposeUsage())
            pu.call_count += call_count
            pu.input_tokens += input_tokens
            pu.output_tokens += output_tokens
            pu.total_latency_ms += latency_ms

    def get_summary(self) -> dict:
        with self._lock:
            total_in = sum(m.input_tokens for m in self._models.values())
            total_out = sum(m.output_tokens for m in self._models.values())
            total_calls = sum(m.call_count for m in self._models.values())
            total_cost = sum(self._estimate_cost(m) for m in self._models.values())
            uptime_s = time.time() - self._start_time

            # Per-purpose roll-up across all models. Lets the UI show
            # "indexing cost" and "query cost" as separate top-line numbers.
            by_purpose: dict[str, dict] = {}
            for m in self._models.values():
                for pname, pu in m.by_purpose.items():
                    p_pricing = _pricing_for(m.model)
                    p_per_call = _per_call_for(m.model)
                    p_cost = (
                        pu.input_tokens * p_pricing[0] / 1_000_000
                        + pu.output_tokens * p_pricing[1] / 1_000_000
                        + pu.call_count * p_per_call
                    )
                    bucket = by_purpose.setdefault(pname, {
                        "call_count": 0, "input_tokens": 0, "output_tokens": 0,
                        "total_latency_ms": 0.0, "estimated_cost_usd": 0.0,
                        "category": "indexing" if pname in _INDEXING_PURPOSES else "query",
                    })
                    bucket["call_count"] += pu.call_count
                    bucket["input_tokens"] += pu.input_tokens
                    bucket["output_tokens"] += pu.output_tokens
                    bucket["total_latency_ms"] += pu.total_latency_ms
                    bucket["estimated_cost_usd"] += p_cost

            # Round on the way out and add avg_latency.
            for b in by_purpose.values():
                b["avg_latency_ms"] = round(
                    b["total_latency_ms"] / b["call_count"], 1
                ) if b["call_count"] else 0.0
                b["estimated_cost_usd"] = round(b["estimated_cost_usd"], 6)
                del b["total_latency_ms"]

            indexing_cost = sum(
                b["estimated_cost_usd"] for b in by_purpose.values()
                if b["category"] == "indexing"
            )
            query_cost = sum(
                b["estimated_cost_usd"] for b in by_purpose.values()
                if b["category"] == "query"
            )

            models = {}
            for name, m in self._models.items():
                cost = self._estimate_cost(m)
                avg_latency = m.total_latency_ms / m.call_count if m.call_count else 0
                models[name] = {
                    "input_tokens": m.input_tokens,
                    "output_tokens": m.output_tokens,
                    "call_count": m.call_count,
                    "avg_latency_ms": round(avg_latency, 1),
                    "estimated_cost_usd": round(cost, 6),
                    # Per-purpose breakdown for this model.
                    "by_purpose": {
                        pname: {
                            "call_count": pu.call_count,
                            "input_tokens": pu.input_tokens,
                            "output_tokens": pu.output_tokens,
                            "avg_latency_ms": round(
                                pu.total_latency_ms / pu.call_count, 1
                            ) if pu.call_count else 0.0,
                        }
                        for pname, pu in m.by_purpose.items()
                    },
                }

            return {
                "total_input_tokens": total_in,
                "total_output_tokens": total_out,
                "total_calls": total_calls,
                "total_estimated_cost_usd": round(total_cost, 6),
                "indexing_cost_usd": round(indexing_cost, 6),
                "query_cost_usd": round(query_cost, 6),
                "uptime_seconds": round(uptime_s, 1),
                "active_sessions": len(self._sessions),
                "models": models,
                "by_purpose": by_purpose,
            }

    def get_session_usage(self, session_id: str) -> dict | None:
        with self._lock:
            return self._sessions.get(session_id)

    @staticmethod
    def _estimate_cost(m: ModelUsage) -> float:
        # _pricing_for handles exact + region-prefix-stripped + suffix match
        # (e.g. "us.anthropic.claude-opus-4-8" -> "anthropic.claude-opus-4-8").
        pricing = _pricing_for(m.model)
        token_cost = (
            m.input_tokens * pricing[0] / 1_000_000
            + m.output_tokens * pricing[1] / 1_000_000
        )
        # Per-call pricing (rerankers) is additive on top of token pricing.
        return token_cost + (m.call_count * _per_call_for(m.model))


# Global singleton
tracker = UsageTracker()
