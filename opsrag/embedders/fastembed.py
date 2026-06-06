"""FastEmbed embeddings -- local ONNX models, no API key required.

Uses the fastembed library to run embedding models entirely on-device.
Great for local dev, air-gapped environments, or when you don't have
an OpenAI/Vertex API key.

Default model: BAAI/bge-small-en-v1.5 (384 dimensions, ~45MB download).
First call downloads the model; subsequent calls are instant.

Requires: pip install fastembed
"""
from __future__ import annotations

from fastembed import TextEmbedding

_MODEL_DIMENSIONS: dict[str, int] = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "nomic-ai/nomic-embed-text-v1.5": 768,
    "jinaai/jina-embeddings-v2-base-code": 768,
}


_QUERY_PREFIXES: dict[str, str] = {
    "bge": "Represent this sentence for searching relevant passages: ",
    "nomic": "search_query: ",
}

_DOC_PREFIXES: dict[str, str] = {
    "nomic": "search_document: ",
}


class FastEmbedEmbeddings:
    def __init__(
        self,
        model: str = "BAAI/bge-small-en-v1.5",
        batch_size: int = 8,  # Small batches to limit ONNX native memory
        dimension: int | None = None,
    ):
        self._model_name = model
        self._batch_size = batch_size
        _known_dim = _MODEL_DIMENSIONS.get(model)
        if dimension is None and _known_dim is None:
            # Fail closed for consistency with the cloud embedders: a silent
            # 768 fallback for an unmapped model bakes a wrong-dim collection.
            raise ValueError(
                f"Unknown FastEmbed model {model!r} and no dimension set. Set "
                f"embedding.dimension explicitly or use a known model: "
                f"{sorted(_MODEL_DIMENSIONS)}"
            )
        self._dimension = dimension or _known_dim
        self._engine = TextEmbedding(model_name=model)
        low = model.lower()
        self._query_prefix = next((v for k, v in _QUERY_PREFIXES.items() if k in low), "")
        self._doc_prefix = next((v for k, v in _DOC_PREFIXES.items() if k in low), "")

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._doc_prefix:
            texts = [self._doc_prefix + t for t in texts]
        embeddings = list(self._engine.embed(texts, batch_size=self._batch_size))
        return [emb.tolist() for emb in embeddings]

    async def embed_query(self, query: str) -> list[float]:
        text = (self._query_prefix + query) if self._query_prefix else query
        result = list(self._engine.embed([text]))
        return result[0].tolist()
