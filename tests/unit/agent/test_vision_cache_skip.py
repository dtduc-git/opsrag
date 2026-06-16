"""Fix 2: an image turn MUST bypass the QA semantic cache (correctness).

The QA cache is keyed on the (text) question embedding only -- it is
image-blind. So for any turn carrying attached images we must:

  * SKIP the cache lookup -- an image turn whose text happens to match a
    prior text-only question must not be served the image-blind cached
    answer.
  * SKIP the cache store -- an answer that depended on an image must not be
    written back under the text-only key, where a later text-only turn
    could be served it.

These tests drive both agent entrypoints (``query_with_session`` and
``query_with_session_events``) with a stub ``qa_cache`` that records every
``lookup`` / ``store`` call, and assert neither is touched when images are
present (while staying touched on the text-only path, as a control).
"""
from __future__ import annotations

import asyncio

from opsrag.agent.graph import query_with_session, query_with_session_events
from opsrag.llms.content import ImagePart

PNG = b"\x89PNG\r\nfake"


class _FakeEmbedder:
    async def embed_query(self, text):  # noqa: ANN001
        return [0.0, 1.0, 0.0]


class _RecordingCache:
    """Records lookup/store invocations; lookup always misses."""

    def __init__(self) -> None:
        self.lookups = 0
        self.stores = 0

    async def lookup(self, *a, **k):  # noqa: ANN001, ANN002
        self.lookups += 1
        return None

    async def store(self, *a, **k):  # noqa: ANN001, ANN002
        self.stores += 1
        return "cache-id"


class _FakeInvokeGraph:
    async def ainvoke(self, initial, config=None) -> dict:  # noqa: ANN001
        return {
            "query": initial.get("query"),
            "generation": "an answer",
            "generation_grounded": True,
            "grounding_checked": True,
            "final_chunks": [],
            "current_step": "done",
        }


class _FakeStreamGraph:
    async def astream_events(self, initial, config=None, version=None):  # noqa: ANN001
        yield {
            "event": "on_chain_end",
            "name": "LangGraph",
            "data": {
                "output": {
                    "query": initial.get("query"),
                    "generation": "an answer",
                    "generation_grounded": True,
                    "grounding_checked": True,
                    "final_chunks": [],
                    "sources_searched": [],
                }
            },
        }


def _run_invoke(cache, images):
    return asyncio.run(
        query_with_session(
            compiled_graph=_FakeInvokeGraph(),
            query="what does this show",
            user_id="u1",
            thread_id="t1",
            embedder=_FakeEmbedder(),
            qa_cache=cache,
            images=images,
        )
    )


def _run_events(cache, images):
    async def _go():
        async for _ in query_with_session_events(
            _FakeStreamGraph(),
            query="what does this show",
            user_id="u1",
            thread_id="t1",
            embedder=_FakeEmbedder(),
            qa_cache=cache,
            images=images,
        ):
            pass

    asyncio.run(_go())


# --- invoke path ----------------------------------------------------------
def test_invoke_image_turn_skips_lookup_and_store():
    cache = _RecordingCache()
    _run_invoke(cache, images=[ImagePart(PNG, "image/png", "a.png")])
    assert cache.lookups == 0, "image turn must NOT read the text-keyed cache"
    assert cache.stores == 0, "image-dependent answer must NOT be cached"


def test_invoke_text_turn_still_uses_cache():
    """Control: the text-only path is unchanged (lookup + store happen)."""
    cache = _RecordingCache()
    _run_invoke(cache, images=None)
    assert cache.lookups == 1
    assert cache.stores == 1


# --- events path ----------------------------------------------------------
def test_events_image_turn_skips_lookup_and_store():
    cache = _RecordingCache()
    _run_events(cache, images=[ImagePart(PNG, "image/png", "a.png")])
    assert cache.lookups == 0, "image turn must NOT read the text-keyed cache"
    assert cache.stores == 0, "image-dependent answer must NOT be cached"


def test_events_text_turn_still_uses_cache():
    cache = _RecordingCache()
    _run_events(cache, images=None)
    assert cache.lookups == 1
    assert cache.stores == 1
