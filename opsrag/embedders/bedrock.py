"""AWS Bedrock embeddings -- uses Amazon Titan Text Embeddings v2.

Auth via standard AWS credential chain:
  - Environment: AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY + AWS_DEFAULT_REGION
  - Shared credentials file (~/.aws/credentials)
  - IAM role (EC2/ECS/EKS)
  - SSO: aws sso login --profile your-profile

Requires: pip install boto3
"""
from __future__ import annotations

import asyncio
import json
import time

import boto3

from opsrag.tokenization import estimate_tokens
from opsrag.usage import tracker as _usage_tracker

_MODEL_DIMENSIONS: dict[str, int] = {
    "amazon.titan-embed-text-v2:0": 1024,
    "amazon.titan-embed-text-v1": 1536,
    "cohere.embed-english-v3": 1024,
    "cohere.embed-multilingual-v3": 1024,
    # Cohere Embed v4 (Matryoshka: 256/512/1024/1536). Needs an inference
    # profile id, e.g. "us.cohere.embed-v4:0". Default to the full 1536.
    "us.cohere.embed-v4:0": 1536,
    "global.cohere.embed-v4:0": 1536,
    "cohere.embed-v4:0": 1536,
}


class BedrockEmbeddings:
    def __init__(
        self,
        model: str = "amazon.titan-embed-text-v2:0",
        region: str | None = None,
        profile: str | None = None,
        dimension: int | None = None,
        batch_size: int = 25,
    ):
        session = boto3.Session(
            region_name=region,
            profile_name=profile,
        )
        self._client = session.client("bedrock-runtime")
        self._model = model
        self._batch_size = batch_size
        _known_dim = _MODEL_DIMENSIONS.get(model)
        if dimension is None and _known_dim is None:
            # Fail closed -- see vertex.py: a silent 1024 fallback for an
            # unknown model bakes a wrong-dim collection on first boot.
            raise ValueError(
                f"Unknown Bedrock embedding model {model!r} and no dimension set. "
                f"Set embedding.dimension explicitly or use a known model: "
                f"{sorted(_MODEL_DIMENSIONS)}"
            )
        self._dimension = dimension or _known_dim
        self._is_titan = model.startswith("amazon.titan")
        # Cohere Embed v4 changed the wire format vs v3: request takes
        # output_dimension + embedding_types, response nests vectors under
        # embeddings.float (vs v3's embeddings[]). Detect by family.
        self._is_cohere_v4 = "cohere.embed-v4" in model
        # Cohere accepts a list of texts per request; 96 is the documented cap.
        self._cohere_batch = 96

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model

    def _invoke(
        self, text: str, input_type: str = "search_document"
    ) -> tuple[list[float], int]:
        """Return (embedding, input_token_count). Titan reports the real
        token count via ``inputTextTokenCount``; Cohere embed doesn't, so we
        fall back to our estimate so usage/cost telemetry still populates."""
        if self._is_titan:
            body = json.dumps({
                "inputText": text,
                "dimensions": self._dimension,
            })
        elif self._is_cohere_v4:
            body = json.dumps({
                "texts": [text],
                "input_type": input_type,
                "output_dimension": self._dimension,
                "embedding_types": ["float"],
            })
        else:  # Cohere Embed v3
            body = json.dumps({
                "texts": [text],
                "input_type": input_type,
            })

        resp = self._client.invoke_model(
            modelId=self._model,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        result = json.loads(resp["body"].read())

        if self._is_titan:
            tokens = int(result.get("inputTextTokenCount") or estimate_tokens(text))
            return result["embedding"], tokens
        if self._is_cohere_v4:
            # embeddings: {"float": [[...]]} -- Cohere doesn't return a token
            # count, so estimate for usage/cost telemetry.
            return result["embeddings"]["float"][0], estimate_tokens(text)
        return result["embeddings"][0], estimate_tokens(text)  # v3

    def _invoke_cohere_batch(
        self, texts: list[str], input_type: str
    ) -> list[list[float]]:
        """Embed up to ~96 texts in ONE Cohere request (v3/v4). Cohere accepts
        a `texts` list, so batching cuts the per-chunk HTTP round-trips by ~the
        batch size -- the difference between minutes and ~1.5h for a full
        re-index. Titan has no batch input, so this is Cohere-only."""
        if self._is_cohere_v4:
            body = json.dumps({
                "texts": texts,
                "input_type": input_type,
                "output_dimension": self._dimension,
                "embedding_types": ["float"],
            })
        else:  # Cohere Embed v3
            body = json.dumps({"texts": texts, "input_type": input_type})
        resp = self._client.invoke_model(
            modelId=self._model, body=body,
            contentType="application/json", accept="application/json",
        )
        result = json.loads(resp["body"].read())
        if self._is_cohere_v4:
            return result["embeddings"]["float"]
        return result["embeddings"]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # boto3 invoke is sync -> run off the event loop, in parallel.
        t0 = time.perf_counter()
        if self._is_titan:
            # Titan: one inputText per call -> parallel single invokes.
            results = await asyncio.gather(
                *(asyncio.to_thread(self._invoke, t, "search_document") for t in texts)
            )
            vecs = [v for v, _ in results]
            total_tokens = sum(tok for _, tok in results)
        else:
            # Cohere: batch many texts per request, batches run concurrently.
            bs = self._cohere_batch
            batches = [texts[i:i + bs] for i in range(0, len(texts), bs)]
            batch_vecs = await asyncio.gather(
                *(asyncio.to_thread(self._invoke_cohere_batch, b, "search_document")
                  for b in batches)
            )
            vecs = [v for bv in batch_vecs for v in bv]
            total_tokens = sum(estimate_tokens(t) for t in texts)
        latency_ms = (time.perf_counter() - t0) * 1000
        # `embed-index` purpose -> the indexing cost bucket in Usage & Cost.
        _usage_tracker.record(
            model=self._model,
            input_tokens=total_tokens,
            output_tokens=0,
            latency_ms=latency_ms,
            purpose="embed-index",
        )
        return vecs

    async def embed_query(self, query: str) -> list[float]:
        t0 = time.perf_counter()
        vec, tokens = await asyncio.to_thread(self._invoke, query, "search_query")
        latency_ms = (time.perf_counter() - t0) * 1000
        _usage_tracker.record(
            model=self._model,
            input_tokens=tokens,
            output_tokens=0,
            latency_ms=latency_ms,
            purpose="embed-query",
        )
        return vec
