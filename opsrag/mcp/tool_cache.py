"""Tool-output micro-cache with per-tool TTL + negative caching.

Env toggle:
  OPSRAG_TOOL_CACHE  -- "0"/"false"/"off" disables caching globally.
                       The wrapper still calls `compute()` directly so
                       behaviour stays correct. Default: enabled.


Wraps every MCP tool dispatch with an in-memory TTL cache keyed by
`(tool_name, sorted_args_hash)`. Per-tool TTL is configured below
based on the bench reality of each MCP:

  - Live metrics (prometheus, datadog logs, k8s pod state):  ~30s
  - Recent state (k8s lists, monitor configs):                ~60s
  - Frozen post-execution (terminal pipeline status,          ~24h
    closed incident, postmortem):
  - Immutable (commit, file at sha):                          forever

Cache keys:

  Hash of canonicalized (sorted-keys) args JSON. Same args in any
  order => same key. Errors are ALSO cached, with a shorter "negative"
  TTL -- saves the agent from retrying a dead Datadog query 5 times in
  one minute.

Storage:

  In-process dict with LRU eviction (bounded size) + TTL check on read.
  Suitable for single-replica dev / small prod. Swap for Redis later
  when running multi-replica without changing call-sites -- interface
  is a single `await get_or_compute(...)` so the storage backend is
  pluggable.

Stats:

  `cache.stats()` returns hit/miss/eviction counts per tool -- exposed
  via `/usage` and the cache purge UI page.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

_log = logging.getLogger("opsrag.mcp.tool_cache")


# Per-tool TTL in seconds. Names match the MCP tool registry; missing
# names fall through to `_DEFAULT_TTL`. Convention: live data -> seconds;
# recent state -> minute(s); frozen -> 24h+; immutable -> ~forever.
_TOOL_TTLS: dict[str, int] = {
    # -- Prometheus (live metrics) -------------------------------------
    "prometheus_query":            30,
    "prometheus_query_range":      30,
    "prometheus_label_values":   3600,
    "prometheus_series":           60,
    "prometheus_metric_metadata": 3600,
    "prometheus_targets":         300,
    # -- Kubernetes (mostly live) --------------------------------------
    "k8s_list_pods":               30,
    "k8s_list_deployments":        60,
    "k8s_describe_pod":            30,
    "k8s_pod_logs":                15,
    "k8s_get_events":              60,
    "k8s_top_pods":                30,
    "k8s_list_services":          300,
    "k8s_list_namespaces":        300,
    "k8s_describe_node":          120,
    # -- Cloud SQL (live but slow-moving) ------------------------------
    "cloudsql_list_instances":    300,
    "cloudsql_get_instance":      120,
    "cloudsql_get_metrics":        60,
    "cloudsql_list_databases":    300,
    "cloudsql_list_users":        600,
    "cloudsql_list_backups":      600,
    "cloudsql_list_operations":   120,
    # -- Datadog (mix of live / recent) --------------------------------
    # `datadog_search_logs` removed 2026-05-21 -- this deployment routes
    # logs to Elasticsearch (`elasticsearch_*` tools), not Datadog.
    "datadog_search_traces":       30,
    "datadog_search_events":       60,
    "datadog_list_monitors":      120,
    "datadog_get_monitor":         60,
    "datadog_query_metrics":       60,
    "datadog_list_slos":          300,
    "datadog_get_slo":             60,
    # -- Rootly (incidents) --------------------------------------------
    # Open incidents are live; resolved are frozen. We can't tell from
    # name alone -- use modest TTL. Resolved-incident detail tools (post-
    # mortem) are safer to cache long.
    "rootly_search":               60,
    "rootly_list_incidents":       60,
    "rootly_get_incident":        300,
    "rootly_get_post_mortem":   86400,  # finalized markdown
    "rootly_get_severity":      86400,
    "rootly_get_environment":   86400,
    "rootly_get_incident_role": 86400,
    # -- GitLab --------------------------------------------------------
    # Pipeline-by-id terminal results are immutable. List-style queries
    # change as new pipelines run.
    "gitlab_get_pipeline":     86400,    # status terminal once finished
    "gitlab_list_pipelines":     120,
    "gitlab_get_pipeline_job":  86400,
    "gitlab_list_pipeline_jobs": 300,
    "gitlab_get_commit":        86400 * 30,  # commit SHA is immutable
    "gitlab_list_commits":       300,
    "gitlab_get_file_contents":  300,
    "gitlab_list_merge_requests":120,
    "gitlab_get_merge_request": 1800,
    "gitlab_list_projects":     3600,
}

_DEFAULT_TTL = 60
_NEGATIVE_TTL = 30          # Errors / no-data cached briefly to absorb retry storms.
_MAX_ENTRIES = 5000         # LRU cap. Each entry ~5-50KB -> ~25-250MB max.
_VALUE_SIZE_LIMIT = 500_000  # Skip caching responses bigger than this (bytes).


def _canonical_key(tool_name: str, args: dict | None) -> str:
    """Sorted-keys JSON hash so {a:1,b:2} and {b:2,a:1} -> same key."""
    arg_blob = json.dumps(args or {}, sort_keys=True, default=str)
    return hashlib.sha256(f"{tool_name}\x00{arg_blob}".encode()).hexdigest()[:32]


@dataclass
class _Entry:
    value: Any
    is_error: bool
    expires_at: float
    size_bytes: int = 0


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    negative_hits: int = 0
    evictions: int = 0
    skipped_too_big: int = 0
    by_tool: dict[str, dict] = field(default_factory=dict)


class ToolOutputCache:
    """LRU + TTL cache for MCP tool results, with negative caching.

    Use `get_or_compute(tool_name, args, compute)` -- the compute coroutine
    only runs on miss. Errors raised by `compute` are caught, cached as
    negative entries with `_NEGATIVE_TTL`, then re-raised so the agent
    sees the same exception path as today.

    Negative-cached errors are re-raised on subsequent gets within TTL,
    so the caller-visible behaviour is identical (no silent error swallow).
    """

    def __init__(
        self,
        max_entries: int = _MAX_ENTRIES,
        default_ttl: int = _DEFAULT_TTL,
        negative_ttl: int = _NEGATIVE_TTL,
    ):
        self._store: OrderedDict[str, _Entry] = OrderedDict()
        self._lock = asyncio.Lock()
        self._max_entries = max_entries
        self._default_ttl = default_ttl
        self._negative_ttl = negative_ttl
        self._stats = CacheStats()
        # Secondary index -- tool_name -> set of keys, for O(1) per-tool purge.
        self._tool_index: dict[str, set[str]] = {}

    # --------------------------- public API ----------------------------

    def ttl_for(self, tool_name: str) -> int:
        return _TOOL_TTLS.get(tool_name, self._default_ttl)

    async def get_or_compute(
        self,
        tool_name: str,
        args: dict | None,
        compute: Callable[[], Awaitable[Any]],
    ) -> Any:
        key = _canonical_key(tool_name, args)
        now = time.time()

        # Read lock: check + touch LRU
        async with self._lock:
            entry = self._store.get(key)
            if entry is not None and entry.expires_at > now:
                self._store.move_to_end(key)  # LRU touch
                self._bump(tool_name, "hits" if not entry.is_error else "negative_hits")
                self._stats.hits += 0 if entry.is_error else 1
                self._stats.negative_hits += 1 if entry.is_error else 0
                if entry.is_error:
                    raise entry.value  # re-raise cached exception
                return entry.value
            # expired / miss
            if entry is not None:
                self._store.pop(key, None)
            self._stats.misses += 1
            self._bump(tool_name, "misses")

        # Compute outside lock so concurrent callers for *different* keys
        # don't serialize. Caveat: two concurrent callers for the SAME
        # key both miss and both compute. Acceptable given low expected
        # concurrency on identical keys; can add per-key in-flight dedup
        # later if needed.
        ttl = self.ttl_for(tool_name)
        try:
            value = await compute()
        except Exception as exc:
            await self._put(key, tool_name, _Entry(
                value=exc, is_error=True,
                expires_at=time.time() + self._negative_ttl,
            ))
            raise

        # Skip caching giant responses
        try:
            size = len(json.dumps(value, default=str))
        except Exception:
            size = 0
        if size > _VALUE_SIZE_LIMIT:
            async with self._lock:
                self._stats.skipped_too_big += 1
            return value

        await self._put(key, tool_name, _Entry(
            value=value, is_error=False,
            expires_at=time.time() + ttl, size_bytes=size,
        ))
        return value

    async def stats(self) -> dict:
        async with self._lock:
            return {
                "hits": self._stats.hits,
                "misses": self._stats.misses,
                "negative_hits": self._stats.negative_hits,
                "evictions": self._stats.evictions,
                "skipped_too_big": self._stats.skipped_too_big,
                "size": len(self._store),
                "max_entries": self._max_entries,
                "by_tool": dict(self._stats.by_tool),
            }

    async def purge(
        self,
        *,
        tool_name: str | None = None,
        all: bool = False,
    ) -> int:
        """Drop entries matching the filter. Returns count purged."""
        async with self._lock:
            if all:
                n = len(self._store)
                self._store.clear()
                self._tool_index.clear()
                return n
            if tool_name is None:
                return 0
            keys = self._tool_index.pop(tool_name, set())
            for k in keys:
                self._store.pop(k, None)
            return len(keys)

    # --------------------- internals ---------------------

    async def _put(self, key: str, tool_name: str, entry: _Entry) -> None:
        async with self._lock:
            self._store[key] = entry
            self._store.move_to_end(key)
            self._tool_index.setdefault(tool_name, set()).add(key)
            # LRU evict
            while len(self._store) > self._max_entries:
                evicted_key, _ = self._store.popitem(last=False)
                self._stats.evictions += 1
                for s in self._tool_index.values():
                    s.discard(evicted_key)

    def _bump(self, tool_name: str, field: str) -> None:
        bucket = self._stats.by_tool.setdefault(tool_name, {
            "hits": 0, "misses": 0, "negative_hits": 0,
        })
        if field in bucket:
            bucket[field] += 1


# Module-level singleton -- the wiring layer (mcp/__init__.py or the
# tool_caller_node) holds a reference. Tests can construct their own
# instance with smaller TTLs.
_default_cache: ToolOutputCache | None = None


def get_default_cache() -> ToolOutputCache:
    global _default_cache
    if _default_cache is None:
        _default_cache = ToolOutputCache()
    return _default_cache


# Convenience wrapper for MCP tool handler functions. Usage:
#
#   async def _h_prometheus_query(args):
#       return await cached_call("prometheus_query", args,
#                                lambda: _real_prometheus_query(args))
#
def is_enabled() -> bool:
    return os.environ.get("OPSRAG_TOOL_CACHE", "1").lower() in ("1", "true", "yes", "on")


async def cached_call(
    tool_name: str,
    args: dict | None,
    compute: Callable[[], Awaitable[Any]],
    *,
    cache: ToolOutputCache | None = None,
) -> Any:
    if not is_enabled():
        # Bypass cache entirely; identical raise/return semantics.
        return await compute()
    return await (cache or get_default_cache()).get_or_compute(
        tool_name, args, compute,
    )


# --- fake backend (FR-012; integration tests) ----------------------

def build_fake():
    """Return a FakeMCP for registry uniformity.

    This module is a read-through cache that wraps idempotent calls on
    OTHER MCPs -- it exposes no tools of its own (registry tool_names is
    empty). ``build_fake`` exists only so the per-MCP fake contract is
    uniform across the registry; the cache's real behaviour is covered
    directly via ``get_default_cache`` / ``ToolOutputCache`` in tests.
    """
    from opsrag.mcp._fake import FakeMCP

    return FakeMCP(tools=[], client=None)
