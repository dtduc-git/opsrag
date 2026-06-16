"""F4: stale-while-revalidate (SWR) must be a per-request argument, not a
process-global env-var toggle.

Previously the background revalidation task flipped
``os.environ["OPSRAG_QA_CACHE_SWR"]="0"`` across an ``await`` and restored it
in a ``finally``. Because ``os.environ`` is process-global, concurrent
requests reading it at the cache-lookup sites could observe SWR disabled, and
overlapping revalidations could leave it stuck at ``"0"``.

The fix:
  * ``_swr_env_default()`` reads ``OPSRAG_QA_CACHE_SWR`` as *read-only* config.
  * ``query_with_session(..., serve_stale=...)`` overrides per call.
  * ``_swr_revalidate`` passes ``serve_stale=False`` instead of mutating env.

These tests assert the ``serve_stale`` value forwarded to ``qa_cache.lookup``
follows the argument when given, falls back to the env default otherwise, and
that ``_swr_env_default`` never mutates the environment.
"""
from __future__ import annotations

import asyncio
import os

from opsrag.agent.graph import (
    _swr_env_default,
    query_with_session,
    query_with_session_events,
)


class _FakeEmbedder:
    async def embed_query(self, text):  # noqa: ANN001
        return [0.0, 1.0, 0.0]


class _ServeStaleRecordingCache:
    """Records the ``serve_stale`` kwarg seen on each lookup; always misses."""

    def __init__(self) -> None:
        self.serve_stale_seen: list[object] = []

    async def lookup(self, *a, serve_stale=None, **k):  # noqa: ANN001, ANN002
        self.serve_stale_seen.append(serve_stale)
        return None

    async def store(self, *a, **k):  # noqa: ANN001, ANN002
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


def _run(cache, **kwargs):
    return asyncio.run(
        query_with_session(
            compiled_graph=_FakeInvokeGraph(),
            query="why did the deploy fail",
            user_id="u1",
            thread_id="t1",
            embedder=_FakeEmbedder(),
            qa_cache=cache,
            **kwargs,
        )
    )


def _run_events(cache, **kwargs):
    async def _go():
        async for _ in query_with_session_events(
            _FakeStreamGraph(),
            query="why did the deploy fail",
            user_id="u1",
            thread_id="t1",
            embedder=_FakeEmbedder(),
            qa_cache=cache,
            **kwargs,
        ):
            pass

    asyncio.run(_go())


# --- explicit serve_stale overrides the env, both directions ---------------
def test_serve_stale_false_overrides_enabled_env(monkeypatch):
    monkeypatch.setenv("OPSRAG_QA_CACHE_SWR", "1")
    cache = _ServeStaleRecordingCache()
    _run(cache, serve_stale=False)
    assert cache.serve_stale_seen == [False]


def test_serve_stale_true_overrides_disabled_env(monkeypatch):
    monkeypatch.setenv("OPSRAG_QA_CACHE_SWR", "0")
    cache = _ServeStaleRecordingCache()
    _run(cache, serve_stale=True)
    assert cache.serve_stale_seen == [True]


# --- default (serve_stale=None) falls back to the env-derived default ------
def test_default_follows_env_enabled(monkeypatch):
    monkeypatch.setenv("OPSRAG_QA_CACHE_SWR", "1")
    cache = _ServeStaleRecordingCache()
    _run(cache)  # serve_stale defaults to None
    assert cache.serve_stale_seen == [True]


def test_default_follows_env_disabled(monkeypatch):
    monkeypatch.setenv("OPSRAG_QA_CACHE_SWR", "0")
    cache = _ServeStaleRecordingCache()
    _run(cache)
    assert cache.serve_stale_seen == [False]


def test_default_when_env_unset_is_enabled(monkeypatch):
    monkeypatch.delenv("OPSRAG_QA_CACHE_SWR", raising=False)
    cache = _ServeStaleRecordingCache()
    _run(cache)
    assert cache.serve_stale_seen == [True]


# --- events path forwards serve_stale to the same lookup site --------------
def test_events_serve_stale_false_overrides_enabled_env(monkeypatch):
    monkeypatch.setenv("OPSRAG_QA_CACHE_SWR", "1")
    cache = _ServeStaleRecordingCache()
    _run_events(cache, serve_stale=False)
    assert cache.serve_stale_seen == [False]


def test_events_default_follows_env_enabled(monkeypatch):
    monkeypatch.setenv("OPSRAG_QA_CACHE_SWR", "1")
    cache = _ServeStaleRecordingCache()
    _run_events(cache)
    assert cache.serve_stale_seen == [True]


# --- _swr_env_default is read-only: it must not mutate the environment ------
def test_swr_env_default_is_read_only(monkeypatch):
    monkeypatch.setenv("OPSRAG_QA_CACHE_SWR", "1")
    before = dict(os.environ)
    assert _swr_env_default() is True
    assert dict(os.environ) == before

    monkeypatch.setenv("OPSRAG_QA_CACHE_SWR", "no")
    assert _swr_env_default() is False

    monkeypatch.delenv("OPSRAG_QA_CACHE_SWR", raising=False)
    assert _swr_env_default() is True
