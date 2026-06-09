"""Unit tests for the OpenAI embedder's robustness behaviour.

Covers two properties the Vertex/Bedrock embedders already have but OpenAI
historically lacked:
  1. The configured Matryoshka `dimension` is forwarded as the OpenAI
     `dimensions` parameter for text-embedding-3-* models.
  2. Transient errors (rate limit / 5xx / timeout) are retried with backoff
     instead of bubbling up on the first failure.

No network calls -- the AsyncOpenAI client is fully mocked.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from openai import RateLimitError

from opsrag.embedders.openai import OpenAIEmbeddings
from opsrag.usage import tracker


def _fake_resp(vectors: list[list[float]], prompt_tokens: int | None = None):
    """Build a stand-in for the OpenAI embeddings response object.

    When ``prompt_tokens`` is given, attach a usage object mimicking the
    OpenAI SDK's ``resp.usage.prompt_tokens``; otherwise omit usage entirely
    to exercise the estimate fallback.
    """
    resp = MagicMock()
    resp.data = [MagicMock(embedding=v) for v in vectors]
    if prompt_tokens is None:
        resp.usage = None
    else:
        resp.usage = MagicMock(prompt_tokens=prompt_tokens)
    return resp


def _rate_limit_error() -> RateLimitError:
    req = httpx.Request("POST", "https://api.openai.com/v1/embeddings")
    return RateLimitError("slow down", response=httpx.Response(429, request=req), body=None)


def _make_embedder(create_mock: AsyncMock, **kwargs) -> OpenAIEmbeddings:
    emb = OpenAIEmbeddings(api_key="test", **kwargs)
    client = MagicMock()
    client.embeddings.create = create_mock
    # Bypass real client construction.
    emb._client = client
    return emb


async def test_dimensions_forwarded_when_configured() -> None:
    """A configured dimension must be passed through as `dimensions` for the
    text-embedding-3-* Matryoshka models."""
    create = AsyncMock(return_value=_fake_resp([[0.1, 0.2]]))
    emb = _make_embedder(create, model="text-embedding-3-large", dimension=256)

    await emb.embed_query("hello")

    create.assert_awaited_once()
    assert create.await_args.kwargs["dimensions"] == 256
    assert create.await_args.kwargs["model"] == "text-embedding-3-large"


async def test_dimensions_omitted_for_legacy_model() -> None:
    """ada-002 does not support `dimensions`; it must not be sent."""
    create = AsyncMock(return_value=_fake_resp([[0.0]]))
    emb = _make_embedder(create, model="text-embedding-ada-002")

    await emb.embed_query("hi")

    assert "dimensions" not in create.await_args.kwargs


async def test_transient_error_is_retried(monkeypatch) -> None:
    """A transient RateLimitError on the first call must be retried, not raised."""
    # Don't actually sleep through the backoff in the test.
    monkeypatch.setattr("opsrag.embedders.openai.asyncio.sleep", AsyncMock())

    create = AsyncMock(side_effect=[_rate_limit_error(), _fake_resp([[1.0, 2.0]])])
    emb = _make_embedder(create, model="text-embedding-3-small", dimension=512)

    out = await emb.embed_query("retry me")

    assert out == [1.0, 2.0]
    assert create.await_count == 2  # failed once, succeeded on retry


async def test_retry_gives_up_after_max_attempts(monkeypatch) -> None:
    """Persistent transient errors eventually re-raise once retries exhaust."""
    monkeypatch.setattr("opsrag.embedders.openai.asyncio.sleep", AsyncMock())
    monkeypatch.setattr("opsrag.embedders.openai._MAX_RETRIES", 3)

    create = AsyncMock(side_effect=_rate_limit_error())
    emb = _make_embedder(create, model="text-embedding-3-small", dimension=512)

    with pytest.raises(RateLimitError):
        await emb.embed_query("always fails")
    assert create.await_count == 3


# --------------------------- usage telemetry (#13a) ---------------------------


async def test_embed_texts_records_usage() -> None:
    """Indexing embeddings must be recorded under `embed-index` so the OpenAI
    provider isn't blind on cost (it previously recorded nothing)."""
    model = "text-embedding-3-small-usage-index"
    create = AsyncMock(return_value=_fake_resp([[0.1], [0.2]], prompt_tokens=42))
    emb = _make_embedder(create, model="text-embedding-3-small", dimension=1)
    # Use a unique model key in the tracker so the assertion is isolated.
    emb._model = model

    await emb.embed_texts(["alpha", "beta"])

    row = tracker.get_summary()["models"][model]
    assert "embed-index" in row["by_purpose"]
    # API-reported prompt_tokens is preferred over the estimate.
    assert row["by_purpose"]["embed-index"]["input_tokens"] == 42
    assert row["by_purpose"]["embed-index"]["call_count"] == 1


async def test_embed_query_records_usage() -> None:
    """Query embeddings must be recorded under `embed-query`."""
    model = "text-embedding-3-small-usage-query"
    create = AsyncMock(return_value=_fake_resp([[0.3]], prompt_tokens=7))
    emb = _make_embedder(create, model="text-embedding-3-small", dimension=1)
    emb._model = model

    await emb.embed_query("how do i restart the pod")

    row = tracker.get_summary()["models"][model]
    assert "embed-query" in row["by_purpose"]
    assert row["by_purpose"]["embed-query"]["input_tokens"] == 7


async def test_usage_falls_back_to_estimate_when_absent() -> None:
    """When the API omits usage, telemetry still populates from the estimate
    (a non-zero token count) rather than recording zero."""
    model = "text-embedding-3-small-usage-estimate"
    create = AsyncMock(return_value=_fake_resp([[0.9]]))  # no prompt_tokens
    emb = _make_embedder(create, model="text-embedding-3-small", dimension=1)
    emb._model = model

    await emb.embed_query("a reasonably long query string for estimation")

    row = tracker.get_summary()["models"][model]
    assert row["by_purpose"]["embed-query"]["input_tokens"] > 0
