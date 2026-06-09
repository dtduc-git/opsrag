"""Vertex AI embeddings -- GCP-native, uses Application Default Credentials.

Uses the google-cloud-aiplatform SDK. Auth via:
  - GOOGLE_APPLICATION_CREDENTIALS env var pointing to a service account JSON
  - gcloud auth application-default login (for local dev)
  - GKE Workload Identity (for production)

Includes exponential-backoff retry on 429 (per-minute token quota) and 5xx
errors so parallel bulk indexing doesn't crater on rate limits.

Requires: pip install google-cloud-aiplatform
"""
from __future__ import annotations

import asyncio
import logging
import random
import time

from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

from opsrag.tokenization import CHARS_PER_TOKEN, estimate_tokens
from opsrag.usage import tracker as _usage_tracker

_log = logging.getLogger("opsrag.embedders.vertex")

_MODEL_DIMENSIONS: dict[str, int] = {
    "text-embedding-005": 768,
    "text-embedding-004": 768,
    "textembedding-gecko@003": 768,
    # P3 -- gemini-embedding-001 defaults to 3072d but supports Matryoshka
    # down to smaller sizes; we use 768 to match the existing Qdrant schema
    # so we can drop in without recreating collections. Per Google's
    # announcement, this model "supports 100+ languages as well as code"
    # and supports a CODE_RETRIEVAL_QUERY task type for code-specific
    # retrieval (the entire reason for adding it).
    "gemini-embedding-001": 768,
}

# Task types Vertex supports for gemini-embedding-001 (and most newer models):
#   RETRIEVAL_DOCUMENT    -- documents being indexed (default for ingest)
#   RETRIEVAL_QUERY       -- user query for prose-style search (default)
#   CODE_RETRIEVAL_QUERY  -- user query for code-specific search (P3 code lane)
#   SEMANTIC_SIMILARITY / CLASSIFICATION / CLUSTERING / FACT_VERIFICATION / QUESTION_ANSWERING
# We expose only the three retrieval task types to keep call sites simple.

_MAX_RETRIES = 6
_BASE_BACKOFF = 1.5  # seconds; 1.5, 3, 6, 12, 24, 48 -> ~94s worst case


def _retryable_exc_types() -> tuple[type[Exception], ...]:
    """Typed google-api-core exceptions that warrant a retry.

    Resolved lazily because ``google.api_core`` ships with the optional
    ``vertex`` extra -- importing it at module load would break the bare
    install. Returns an empty tuple if the package is absent, in which case
    ``_is_retryable`` falls back to string matching alone.
    """
    try:
        from google.api_core import exceptions as gexc
    except Exception:  # pragma: no cover - only when the extra is absent
        return ()
    # 429 quota, 503 unavailable, 504/deadline, 500 internal. TooManyRequests
    # is the HTTP-transport sibling of the gRPC ResourceExhausted.
    return tuple(
        t
        for t in (
            getattr(gexc, "ResourceExhausted", None),
            getattr(gexc, "TooManyRequests", None),
            getattr(gexc, "ServiceUnavailable", None),
            getattr(gexc, "DeadlineExceeded", None),
            getattr(gexc, "InternalServerError", None),
        )
        if t is not None
    )

# Vertex text-embedding-005 limits per request:
#   - max 250 inputs (count)
#   - max 20,000 tokens summed across inputs
# Whichever caps first. Two-layer defense:
#   1. Conservative pre-batch sizing (this constants block) -- handles the
#      99% case fast.
#   2. Adaptive split-on-400 in _embed_with_retry -- recovers when the real
#      tokenizer counts more tokens than our estimate (dense YAML/HCL can
#      tokenize at ~1-1.5 chars/token, way below the 4 used for prose).
_MAX_INPUTS_PER_REQUEST = 128
_MAX_TOKENS_PER_REQUEST = 14000
# Shared with opsrag.chunkers.parent_child so the two layers agree on
# "tokens per chunk". See opsrag/tokenization.py for the rationale.
_CHARS_PER_TOKEN = CHARS_PER_TOKEN


def _is_retryable(exc: Exception) -> bool:
    # Prefer typed google-api-core exceptions -- robust to message wording /
    # localization changes that fragile string matching misses. The string
    # match below is kept only as a fallback for transports that raise a bare
    # Exception (or when the api-core extra isn't importable).
    retryable_types = _retryable_exc_types()
    if retryable_types and isinstance(exc, retryable_types):
        return True
    msg = str(exc).lower()
    return (
        "429" in msg
        or "resource exhausted" in msg
        or "rate limit" in msg
        or "quota" in msg
        or "503" in msg
        or "service unavailable" in msg
        or "internal error" in msg
        or "deadline" in msg
    )


def _is_token_overflow(exc: Exception) -> bool:
    """Detect Vertex 400 'input token count exceeds limit' specifically."""
    msg = str(exc).lower()
    return "input token count" in msg and "supports up to" in msg


class VertexAIEmbeddings:
    def __init__(
        self,
        model: str = "text-embedding-005",
        project: str | None = None,
        location: str = "us-central1",
        batch_size: int = 128,
        document_task_type: str = "RETRIEVAL_DOCUMENT",
        query_task_type: str = "RETRIEVAL_QUERY",
        output_dimensionality: int | None = None,
    ):
        """Vertex text embedder.

        P3 -- added `document_task_type`, `query_task_type`, and
        `output_dimensionality` so the SAME class can serve both the
        prose-retrieval lane (default args, text-embedding-005) and a
        code-retrieval lane (model=gemini-embedding-001,
        query_task_type=CODE_RETRIEVAL_QUERY, output_dimensionality=768
        for Matryoshka compatibility with existing 768d Qdrant schema).
        """
        import vertexai
        vertexai.init(project=project, location=location)
        self._model_name = model
        self._batch_size = batch_size
        _known_dim = _MODEL_DIMENSIONS.get(model)
        if output_dimensionality is None and _known_dim is None:
            # Fail closed: silently defaulting an unknown model to 768 would
            # create a permanently wrong-dimension collection on first boot
            # (the dimension guard is a no-op on a fresh collection).
            raise ValueError(
                f"Unknown Vertex embedding model {model!r} and no "
                f"output_dimensionality set. Set embedding.dimension explicitly "
                f"or use a known model: {sorted(_MODEL_DIMENSIONS)}"
            )
        self._dimension = output_dimensionality or _known_dim
        self._engine = TextEmbeddingModel.from_pretrained(model)
        self._document_task_type = document_task_type
        self._query_task_type = query_task_type
        self._output_dimensionality = output_dimensionality

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    async def _embed_with_retry(self, inputs: list[TextEmbeddingInput]):
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                # Run the sync SDK call in a thread so we don't block the loop.
                # P3 -- pass output_dimensionality when set (Matryoshka for
                # gemini-embedding-001 to fit the existing 768d schema).
                if self._output_dimensionality is not None:
                    return await asyncio.to_thread(
                        self._engine.get_embeddings,
                        inputs,
                        output_dimensionality=self._output_dimensionality,
                    )
                return await asyncio.to_thread(self._engine.get_embeddings, inputs)
            except Exception as exc:
                last_exc = exc

                # Adaptive split: real tokens > our estimate. Halve the batch
                # and recurse. Catches dense YAML/HCL where chars-per-token
                # is closer to 1 than 2.
                if _is_token_overflow(exc) and len(inputs) > 1:
                    mid = len(inputs) // 2
                    _log.warning(
                        "vertex token overflow on %d-input batch, splitting %d/%d",
                        len(inputs), mid, len(inputs) - mid,
                    )
                    left = await self._embed_with_retry(inputs[:mid])
                    right = await self._embed_with_retry(inputs[mid:])
                    return list(left) + list(right)

                # 429 / 5xx / transient: exponential backoff with jitter.
                if not _is_retryable(exc) or attempt == _MAX_RETRIES - 1:
                    raise
                delay = _BASE_BACKOFF * (2 ** attempt) + random.uniform(0, 1.0)
                _log.warning(
                    "vertex embed retry %d/%d in %.1fs: %s",
                    attempt + 1, _MAX_RETRIES, delay, str(exc)[:160],
                )
                await asyncio.sleep(delay)
        raise last_exc  # type: ignore[misc]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        all_vectors: list[list[float]] = []
        batch: list[TextEmbeddingInput] = []
        batch_tokens = 0

        async def flush() -> None:
            nonlocal batch, batch_tokens
            if not batch:
                return
            t0 = time.perf_counter()
            tokens_in_batch = batch_tokens
            results = await self._embed_with_retry(batch)
            latency_ms = (time.perf_counter() - t0) * 1000
            all_vectors.extend(r.values for r in results)
            # `embed-index` purpose covers all bulk text embedding --
            # dwarfs `embed-query` by orders of magnitude in practice.
            # Token count is our pre-batch estimate; Vertex doesn't
            # return per-call token usage, so this is approximate but
            # consistent across calls.
            _usage_tracker.record(
                model=self._model_name,
                input_tokens=tokens_in_batch,
                output_tokens=0,
                latency_ms=latency_ms,
                purpose="embed-index",
            )
            batch = []
            batch_tokens = 0

        for text in texts:
            est_tokens = estimate_tokens(text)
            # Defensive: a single chunk near or above the per-request token cap
            # would never fit. Truncate at the API limit to avoid 400s.
            if est_tokens > _MAX_TOKENS_PER_REQUEST:
                text = text[: _MAX_TOKENS_PER_REQUEST * _CHARS_PER_TOKEN]
                est_tokens = _MAX_TOKENS_PER_REQUEST
                _log.warning("truncated oversize chunk to fit Vertex 20K-token cap")

            # Flush before adding if this text would push us over either limit.
            if batch and (
                batch_tokens + est_tokens > _MAX_TOKENS_PER_REQUEST
                or len(batch) >= _MAX_INPUTS_PER_REQUEST
            ):
                await flush()

            batch.append(TextEmbeddingInput(text=text, task_type=self._document_task_type))
            batch_tokens += est_tokens

        await flush()
        return all_vectors

    async def embed_query(self, query: str) -> list[float]:
        inputs = [TextEmbeddingInput(text=query, task_type=self._query_task_type)]
        t0 = time.perf_counter()
        results = await self._embed_with_retry(inputs)
        latency_ms = (time.perf_counter() - t0) * 1000
        _usage_tracker.record(
            model=self._model_name,
            input_tokens=estimate_tokens(query),
            output_tokens=0,
            latency_ms=latency_ms,
            purpose="embed-query",
        )
        return results[0].values
