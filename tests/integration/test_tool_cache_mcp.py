"""Integration test (T086): the TOOL_CACHE MCP module against its fake backend.

This module is SPECIAL: it is a read-through cache that wraps idempotent calls
on OTHER MCPs, so it has no tools of its own (registry tool_names is empty and
build_fake() returns an empty FakeMCP). The meaningful coverage for this module
is the cache behaviour itself -- get_or_compute computes once per (name, args)
within TTL, recomputes for different args, recomputes after expiry, and applies
a shorter negative TTL to cached errors. Those are exercised here against a
fresh ToolOutputCache instance so the global default cache is never touched.
"""
from __future__ import annotations

import pytest

from opsrag.mcp.registry import REGISTRY
from opsrag.mcp.tool_cache import (
    ToolOutputCache,
    build_fake,
    get_default_cache,
)

# --- fake/registry uniformity --------------------------------------

def test_fake_exposes_no_tools() -> None:
    fake = build_fake()
    assert fake.tool_names() == []
    assert fake.client is None


def test_registry_declares_no_tools() -> None:
    assert REGISTRY["tool_cache"].tool_names == ()


def test_default_cache_is_singleton() -> None:
    # Smoke-check the module's real public accessor without mutating it.
    assert get_default_cache() is get_default_cache()
    assert isinstance(get_default_cache(), ToolOutputCache)


# --- cache behaviour (the real coverage for this module) -----------

@pytest.mark.asyncio
async def test_same_args_compute_once() -> None:
    # Fresh instance -> deterministic, isolated from the global default cache.
    cache = ToolOutputCache()
    calls = {"n": 0}

    async def compute():
        calls["n"] += 1
        return calls["n"]

    args = {"a": 1, "b": 2}
    first = await cache.get_or_compute("prometheus_query", args, compute)
    second = await cache.get_or_compute("prometheus_query", args, compute)

    assert first == 1
    assert second == 1  # cache hit -> same cached value, no recompute
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_arg_order_does_not_matter() -> None:
    # Canonical (sorted-keys) key: {a,b} and {b,a} hit the same entry.
    cache = ToolOutputCache()
    calls = {"n": 0}

    async def compute():
        calls["n"] += 1
        return calls["n"]

    await cache.get_or_compute("prometheus_query", {"a": 1, "b": 2}, compute)
    await cache.get_or_compute("prometheus_query", {"b": 2, "a": 1}, compute)

    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_different_args_recompute() -> None:
    cache = ToolOutputCache()
    calls = {"n": 0}

    async def compute():
        calls["n"] += 1
        return calls["n"]

    a = await cache.get_or_compute("prometheus_query", {"a": 1}, compute)
    b = await cache.get_or_compute("prometheus_query", {"a": 2}, compute)

    assert a == 1
    assert b == 2  # different args -> miss -> recompute
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_different_tool_name_recompute() -> None:
    cache = ToolOutputCache()
    calls = {"n": 0}

    async def compute():
        calls["n"] += 1
        return calls["n"]

    args = {"a": 1}
    await cache.get_or_compute("prometheus_query", args, compute)
    await cache.get_or_compute("k8s_list_pods", args, compute)

    assert calls["n"] == 2  # same args, different tool -> distinct keys


@pytest.mark.asyncio
async def test_ttl_expiry_recomputes() -> None:
    # ttl=0 means every entry is already expired on the next read -> recompute.
    # Focused TTL test that needs no real sleeping.
    cache = ToolOutputCache(default_ttl=0)
    calls = {"n": 0}

    async def compute():
        calls["n"] += 1
        return calls["n"]

    args = {"a": 1}
    first = await cache.get_or_compute("unknown_tool_uses_default_ttl", args, compute)
    second = await cache.get_or_compute("unknown_tool_uses_default_ttl", args, compute)

    assert first == 1
    assert second == 2  # expired -> recompute
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_negative_caching_reraises_without_recompute() -> None:
    # Errors are cached (negative_ttl) and re-raised on subsequent gets within
    # TTL, so the agent does not retry a dead query repeatedly.
    cache = ToolOutputCache(negative_ttl=60)
    calls = {"n": 0}

    class Boom(Exception):
        pass

    async def compute():
        calls["n"] += 1
        raise Boom("dead query")

    args = {"q": "broken"}
    with pytest.raises(Boom):
        await cache.get_or_compute("datadog_query_metrics", args, compute)
    with pytest.raises(Boom):
        await cache.get_or_compute("datadog_query_metrics", args, compute)

    assert calls["n"] == 1  # second raise came from the cached error entry


@pytest.mark.asyncio
async def test_stats_track_hits_and_misses() -> None:
    cache = ToolOutputCache()

    async def compute():
        return 1

    args = {"a": 1}
    await cache.get_or_compute("prometheus_query", args, compute)  # miss
    await cache.get_or_compute("prometheus_query", args, compute)  # hit

    stats = await cache.stats()
    assert stats["misses"] == 1
    assert stats["hits"] == 1
