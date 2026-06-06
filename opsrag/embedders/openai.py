"""OpenAI embeddings provider -- direct openai SDK, no LangChain."""
from __future__ import annotations

from openai import AsyncOpenAI

_MODEL_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-large": 3072,
    "text-embedding-3-small": 1536,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbeddings:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "text-embedding-3-large",
        dimension: int | None = None,
        batch_size: int = 128,
    ):
        self._api_key = api_key
        self._client: AsyncOpenAI | None = None
        self._model = model
        self._batch_size = batch_size
        _known_dim = _MODEL_DIMENSIONS.get(model)
        if dimension is None and _known_dim is None:
            # Fail closed -- see vertex.py: a silent 1536 fallback for an
            # unknown model bakes a wrong-dim collection on first boot.
            raise ValueError(
                f"Unknown OpenAI embedding model {model!r} and no dimension set. "
                f"Set embedding.dimension explicitly or use a known model: "
                f"{sorted(_MODEL_DIMENSIONS)}"
            )
        self._dimension = dimension or _known_dim

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(api_key=self._api_key)
        return self._client

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model

    def _call_kwargs(self, inputs: list[str]) -> dict:
        kwargs: dict = {"model": self._model, "input": inputs}
        if self._model.startswith("text-embedding-3"):
            kwargs["dimensions"] = self._dimension
        return kwargs

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            resp = await self._get_client().embeddings.create(**self._call_kwargs(batch))
            vectors.extend(d.embedding for d in resp.data)
        return vectors

    async def embed_query(self, query: str) -> list[float]:
        resp = await self._get_client().embeddings.create(**self._call_kwargs([query]))
        return resp.data[0].embedding
