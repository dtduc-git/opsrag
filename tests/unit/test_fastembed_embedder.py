"""Unit tests for the FastEmbed embedder's event-loop friendliness (#5).

The fastembed ONNX ``embed()`` is a SYNC, CPU-bound call. If it ran directly
inside the ``async def`` it would block the event loop for the whole batch,
stalling every other concurrent coroutine (e.g. live queries during a bulk
index). These tests assert the sync work is offloaded to a worker thread via
``asyncio.to_thread`` -- i.e. it runs on a DIFFERENT thread than the loop.

The real ONNX engine is replaced with a tiny fake so no model is downloaded.
"""
from __future__ import annotations

import threading

import numpy as np
import pytest

fastembed = pytest.importorskip("fastembed")

from opsrag.embedders.fastembed import FastEmbedEmbeddings  # noqa: E402


class _FakeEngine:
    """Stand-in for fastembed.TextEmbedding that records the thread it ran on
    and returns numpy vectors shaped like the real engine's output."""

    def __init__(self) -> None:
        self.embed_threads: list[int] = []

    def embed(self, texts, batch_size: int | None = None):
        # Record which thread this synchronous call executed on.
        self.embed_threads.append(threading.get_ident())
        for _ in texts:
            yield np.array([0.1, 0.2, 0.3], dtype=np.float32)


def _make_embedder(engine: _FakeEngine) -> FastEmbedEmbeddings:
    emb = FastEmbedEmbeddings.__new__(FastEmbedEmbeddings)
    # Set only the attributes the embed_* methods actually touch -- bypasses
    # the real TextEmbedding(model_name=...) construction (model download).
    emb._model_name = "BAAI/bge-small-en-v1.5"
    emb._batch_size = 8
    emb._dimension = 384
    emb._engine = engine
    emb._query_prefix = ""
    emb._doc_prefix = ""
    return emb


async def test_embed_texts_offloads_sync_embed_to_thread() -> None:
    engine = _FakeEngine()
    emb = _make_embedder(engine)
    loop_thread = threading.get_ident()

    vecs = await emb.embed_texts(["alpha", "beta"])

    assert vecs == [[pytest.approx(0.1), pytest.approx(0.2), pytest.approx(0.3)]] * 2
    assert engine.embed_threads, "engine.embed was never called"
    # The sync embed must NOT have run on the event-loop thread.
    assert engine.embed_threads[0] != loop_thread, (
        "sync ONNX embed ran on the event loop thread (no to_thread offload)"
    )


async def test_embed_query_offloads_sync_embed_to_thread() -> None:
    engine = _FakeEngine()
    emb = _make_embedder(engine)
    loop_thread = threading.get_ident()

    vec = await emb.embed_query("how do i restart the pod")

    assert vec == [pytest.approx(0.1), pytest.approx(0.2), pytest.approx(0.3)]
    assert engine.embed_threads[0] != loop_thread, (
        "sync ONNX embed ran on the event loop thread (no to_thread offload)"
    )


async def test_embed_texts_empty_skips_engine() -> None:
    engine = _FakeEngine()
    emb = _make_embedder(engine)
    assert await emb.embed_texts([]) == []
    assert engine.embed_threads == []  # engine never touched
