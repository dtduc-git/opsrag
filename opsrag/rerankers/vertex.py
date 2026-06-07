"""Vertex AI Discovery Engine reranker -- hosted cross-encoder via :rank endpoint.

Calls the semantic ranker (default model: semantic-ranker-default-004). Auth
uses Application Default Credentials. Requires the Discovery Engine API to
be enabled on the project.

Endpoint:
  POST https://discoveryengine.googleapis.com/v1/projects/{project}
       /locations/{location}/rankingConfigs/default_ranking_config:rank

The :rank endpoint accepts up to 200 records per call and ~512 tokens per
record. Long candidates (parent chunks at 1024 tokens, augmented chunks,
multi-paragraph runbook sections) are split into overlapping windows; each
window is sent as its own record, and scores are max-pooled back to the
parent candidate. This way the reranker actually sees the whole chunk
instead of judging on a silently-truncated prefix.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from opsrag.interfaces.reranker import RerankResult
from opsrag.interfaces.vectorstore import SearchResult
from opsrag.usage import tracker as _usage_tracker

_log = logging.getLogger("opsrag.rerankers.vertex")

_DEFAULT_MODEL = "semantic-ranker-default-004"
_MAX_RECORDS = 200
# Per-record limit. Reranker model caps at ~512 tokens; at the shared
# CHARS_PER_TOKEN=3 (opsrag.tokenization) this is ~500 tokens, safely
# under the cap. Old value was 4000 chars (>=1000 tokens for prose, >=2000
# for dense YAML/HCL), which the API was silently truncating server-side.
# 1500 chars is still ~1000 tokens for dense YAML/HCL (~1.5 chars/token) --
# over the model's ~512-token record cap -- so use 1000 chars to stay under it.
_MAX_CHARS_PER_RECORD = 1000
_WINDOW_OVERLAP_CHARS = 200  # ~65 tokens of carryover between windows


def _split_windows(content: str, max_windows: int) -> list[str]:
    """Split `content` into <=_MAX_CHARS_PER_RECORD overlapping windows.

    Bounded by `max_windows` so a pathologically long candidate (e.g., a
    full-doc fallback when section parsing fails) can't blow past the
    reranker's 200-record cap when there are many candidates. The first
    `max_windows` slices cover the start of the document; tail content
    past the budget is dropped -- same failure mode as the pre-fix code,
    but only when the document is genuinely longer than the budget allows.
    """
    if not content:
        return [""]
    if len(content) <= _MAX_CHARS_PER_RECORD:
        return [content]
    step = _MAX_CHARS_PER_RECORD - _WINDOW_OVERLAP_CHARS
    windows: list[str] = []
    start = 0
    while start < len(content) and len(windows) < max_windows:
        end = min(start + _MAX_CHARS_PER_RECORD, len(content))
        windows.append(content[start:end])
        if end == len(content):
            break
        start += step
    return windows


class VertexReranker:
    # semantic-ranker-default-004 returns calibrated [0,1] scores -- this is the
    # model the 0.05 noise floor was originally tuned against, so keep it; trust
    # at 0.65 matches the same 0..1 scale.
    score_floor = 0.05
    trust_score = 0.65

    def __init__(
        self,
        project: str,
        model: str = _DEFAULT_MODEL,
        location: str = "global",
        timeout: float = 20.0,
    ):
        if not project:
            raise ValueError("Vertex reranker requires a GCP project id")
        self._project = project
        self._model = model or _DEFAULT_MODEL
        self._location = location or "global"
        self._endpoint = (
            f"https://discoveryengine.googleapis.com/v1/projects/{self._project}"
            f"/locations/{self._location}/rankingConfigs/default_ranking_config:rank"
        )
        self._client = httpx.AsyncClient(timeout=timeout)
        self._token_provider: Any = None  # google.auth credentials, lazy

    async def close(self) -> None:
        await self._client.aclose()

    def _get_token(self) -> str:
        # Lazy import so unit tests with MockTransport don't need ADC.
        if self._token_provider is None:
            import google.auth
            import google.auth.transport.requests

            creds, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            self._token_provider = (creds, google.auth.transport.requests.Request())
        creds, request = self._token_provider
        if not creds.valid:
            creds.refresh(request)
        return creds.token

    async def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 5,
    ) -> list[RerankResult]:
        if not results:
            return []

        candidates = results[:_MAX_RECORDS]
        # Fair-share window budget: with _MAX_RECORDS=200 and N candidates,
        # each candidate gets `200 // N` windows (>=1). Typical input is
        # 10-50 candidates, leaving 4-20 windows each -- comfortably more
        # than the 2 windows a 1024-token parent chunk needs.
        max_windows_per_candidate = max(1, _MAX_RECORDS // max(1, len(candidates)))
        records: list[dict[str, Any]] = []
        for i, r in enumerate(candidates):
            for w_idx, window in enumerate(
                _split_windows(r.chunk.content or "", max_windows_per_candidate)
            ):
                records.append(
                    {
                        "id": f"{i}:{w_idx}",
                        "title": (r.chunk.source_path or "")[:200],
                        "content": window,
                    }
                )
                if len(records) >= _MAX_RECORDS:
                    break
            if len(records) >= _MAX_RECORDS:
                break

        # Ask the API to score *all* records (not just top_k of them) so
        # we can max-pool windows back to candidates locally. Otherwise
        # `topN=top_k` would drop windows belonging to valid candidates
        # whose best window happened to rank lower than another candidate's
        # best window -- exactly the regression the sliding split is meant
        # to fix.
        payload = {
            "model": self._model,
            "query": query,
            "records": records,
            "ignoreRecordDetailsInResponse": True,
            "topN": len(records),
        }

        headers = {
            "Authorization": f"Bearer {self._get_token()}",
            "Content-Type": "application/json",
        }

        # Retry on 429/5xx with exponential backoff + jitter.
        import asyncio
        import random
        last_exc: Exception | None = None
        data = None
        t0 = time.perf_counter()
        for attempt in range(5):
            try:
                resp = await self._client.post(self._endpoint, headers=headers, json=payload)
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise httpx.HTTPStatusError(
                        f"vertex rerank {resp.status_code}", request=resp.request, response=resp,
                    )
                resp.raise_for_status()
                data = resp.json()
                break
            except (httpx.HTTPStatusError, httpx.TimeoutException) as exc:
                last_exc = exc
                if attempt == 4:
                    raise
                delay = 1.0 * (2 ** attempt) + random.uniform(0, 0.5)
                _log.warning(
                    "vertex rerank retry %d/5 in %.1fs: %s",
                    attempt + 1, delay, str(exc)[:160],
                )
                await asyncio.sleep(delay)
        if data is None:
            raise last_exc or RuntimeError("vertex rerank: no data")

        # Record usage. Vertex `:rank` is priced per request (not per
        # token), so we record `call_count=1` with zero tokens -- the
        # tracker's per-call pricing table converts this to USD.
        latency_ms = (time.perf_counter() - t0) * 1000
        _usage_tracker.record(
            model=self._model,
            input_tokens=0,
            output_tokens=0,
            latency_ms=latency_ms,
            purpose="rerank",
        )

        # Max-pool: each candidate's score = max score across its windows.
        # Max (vs. mean) is the right aggregate for retrieval -- one strong
        # window is enough to justify surfacing the chunk, and weak tails
        # shouldn't drag down a chunk whose head is on-topic.
        scores_by_candidate: dict[int, float] = {}
        for item in data.get("records", []):
            # ID format: "{candidate_idx}:{window_idx}" when we sent
            # sliding windows, but tolerate a bare integer too in case
            # the API echoes pre-windowing IDs or a unit test stubs them.
            cand_str = str(item.get("id", "")).split(":", 1)[0]
            try:
                idx = int(cand_str)
            except (TypeError, ValueError):
                continue
            if idx < 0 or idx >= len(candidates):
                continue
            score = float(item.get("score", 0.0))
            prior = scores_by_candidate.get(idx)
            if prior is None or score > prior:
                scores_by_candidate[idx] = score

        ranked = sorted(
            scores_by_candidate.items(), key=lambda kv: kv[1], reverse=True
        )
        out = [
            RerankResult(
                chunk=candidates[idx].chunk,
                relevance_score=score,
            )
            for idx, score in ranked
        ]
        return out[:top_k]
