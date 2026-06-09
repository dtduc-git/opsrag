"""Pluggable rate-limit backends for the OpsRAG API.

Two backends share one small interface so the request-rate limiter
(:class:`opsrag.api.middleware.RateLimitMiddleware`) and the login
brute-force throttle (:class:`opsrag.auth.login.LoginRateLimiter`) can run
either in-process or against a shared Redis instance WITHOUT changing their
call sites.

* :class:`MemoryRateLimitBackend` -- wraps the existing in-process logic.
  This is the default and is byte-identical to the pre-backend behavior:
  per-key sliding-window timestamps for the request limiter, and
  failed-attempt counters with a temporary lockout for login.

* :class:`RedisRateLimitBackend` -- shares state across replicas. The
  request limiter uses a fixed-window counter (atomic ``INCR`` + ``EXPIRE``
  on the first hit of a window); login uses a failure counter (``INCR`` +
  ``EXPIRE``) plus a ``SETEX`` lockout key.

The ``redis`` import is LAZY -- only pulled in when the Redis backend is
actually constructed -- mirroring how ``boto3`` is imported inside
``opsrag.llms.bedrock.BedrockLLM.__init__``. This keeps the module (and
everything that imports it, e.g. the API server) importable on a build
WITHOUT the optional ``redis`` extra.
"""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class RateDecision:
    """Outcome of a single request-rate check.

    ``allowed`` is False when the caller is over the limit. ``remaining`` is
    the number of requests still permitted in the current window (>= 0).
    ``retry_after`` is seconds until the caller may retry (only meaningful
    when ``allowed`` is False).
    """

    allowed: bool
    remaining: int
    retry_after: int = 0


class RateLimitBackend(Protocol):
    """Storage seam shared by the request limiter and the login throttle."""

    # --- per-request rpm limiter -------------------------------------------
    async def hit(self, key: str, limit: int, window_seconds: float) -> RateDecision:
        """Record one request for ``key`` and report whether it's allowed."""
        ...

    # --- login brute-force lockout -----------------------------------------
    async def record_login_failure(
        self,
        key: str,
        *,
        max_attempts: int,
        window_seconds: float,
        lockout_seconds: float,
    ) -> bool:
        """Record a failed login. Return True iff ``key`` is NOW locked."""
        ...

    async def record_login_success(self, key: str) -> None:
        """Clear all failure/lockout state for ``key``."""
        ...

    async def login_locked(self, key: str) -> bool:
        """Return True iff ``key`` is currently locked out."""
        ...

    async def login_retry_after(self, key: str) -> int:
        """Seconds until ``key``'s lockout expires (0 if not locked)."""
        ...


# ---------------------------------------------------------------------------
# In-process backend (default) -- preserves the original behavior exactly.
# ---------------------------------------------------------------------------
@dataclass
class _Attempts:
    count: int = 0
    first_ts: float = 0.0
    locked_until: float = 0.0


@dataclass
class MemoryRateLimitBackend:
    """In-process backend. Single-replica only; state is per-process.

    The request limiter keeps a per-key list of monotonic timestamps and
    prunes the window on each hit (sliding window). Login keeps a per-key
    failure counter with a temporary lockout. This mirrors the original
    in-line logic in ``RateLimitMiddleware`` / ``LoginRateLimiter`` so the
    memory path stays behaviorally unchanged.
    """

    _buckets: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    _login: dict[str, _Attempts] = field(default_factory=dict)

    async def hit(self, key: str, limit: int, window_seconds: float) -> RateDecision:
        now = time.monotonic()
        window_start = now - window_seconds

        timestamps = [t for t in self._buckets[key] if t > window_start]
        self._buckets[key] = timestamps

        if len(timestamps) >= limit:
            retry_after = int(window_seconds - (now - timestamps[0])) + 1
            return RateDecision(allowed=False, remaining=0, retry_after=retry_after)

        timestamps.append(now)
        remaining = max(0, limit - len(timestamps))
        return RateDecision(allowed=True, remaining=remaining)

    async def record_login_failure(
        self,
        key: str,
        *,
        max_attempts: int,
        window_seconds: float,
        lockout_seconds: float,
    ) -> bool:
        now = time.monotonic()
        a = self._login.get(key)
        if a is None or (now - a.first_ts) > window_seconds:
            a = _Attempts(count=0, first_ts=now)
            self._login[key] = a
        a.count += 1
        if a.count >= max_attempts:
            a.locked_until = now + lockout_seconds
            return True
        return False

    async def record_login_success(self, key: str) -> None:
        self._login.pop(key, None)

    async def login_locked(self, key: str) -> bool:
        now = time.monotonic()
        a = self._login.get(key)
        return bool(a and a.locked_until > now)

    async def login_retry_after(self, key: str) -> int:
        now = time.monotonic()
        a = self._login.get(key)
        if not a or a.locked_until <= now:
            return 0
        return int(a.locked_until - now) + 1


# ---------------------------------------------------------------------------
# Redis backend -- shared state across replicas.
# ---------------------------------------------------------------------------
# Key namespace. Distinct prefixes keep request-rate and login state apart so
# an operator can scan/flush them independently.
_RPM_PREFIX = "opsrag:rl:rpm:"
_LOGIN_FAIL_PREFIX = "opsrag:rl:login:fail:"
_LOGIN_LOCK_PREFIX = "opsrag:rl:login:lock:"


def make_redis_client(url: str):
    """Build a ``redis.asyncio`` client from ``url``.

    The import is LAZY (function-local) so the optional ``redis`` extra is
    only required when the Redis backend is actually selected -- mirroring
    the lazy ``boto3`` import in ``opsrag.llms.bedrock``.
    """
    import redis.asyncio as aioredis  # noqa: PLC0415  (lazy: optional extra)

    return aioredis.from_url(url, encoding="utf-8", decode_responses=True)


@dataclass
class RedisRateLimitBackend:
    """Distributed backend backed by a ``redis.asyncio`` client.

    * Request limiter: a fixed-window counter -- ``INCR`` the per-window key
      and set ``EXPIRE`` to the window length on the first hit. The window
      bucket is derived from wall-clock time so all replicas agree on
      boundaries. Approximate (fixed vs. sliding) but cheap and atomic, and
      sufficient to bound aggregate request rate across replicas.
    * Login: ``INCR`` a failure counter (``EXPIRE`` = attempt window) and,
      once it reaches ``max_attempts``, write a ``SETEX`` lockout key whose
      remaining TTL is the authoritative retry-after.

    The client is injected so tests can pass a fake; production wiring builds
    one via :func:`make_redis_client` and PINGs it at startup (fail-fast).
    """

    client: object  # redis.asyncio.Redis (typed loosely to avoid the import)

    async def hit(self, key: str, limit: int, window_seconds: float) -> RateDecision:
        window = max(1, int(window_seconds))
        bucket = int(time.time()) // window
        rkey = f"{_RPM_PREFIX}{key}:{bucket}"

        count = int(await self.client.incr(rkey))
        if count == 1:
            # First request in this window -- arm the TTL so the counter
            # self-expires. Only on the first INCR to avoid sliding the TTL.
            await self.client.expire(rkey, window)

        if count > limit:
            ttl = await self.client.ttl(rkey)
            retry_after = int(ttl) if ttl and ttl > 0 else window
            return RateDecision(allowed=False, remaining=0, retry_after=retry_after)

        remaining = max(0, limit - count)
        return RateDecision(allowed=True, remaining=remaining)

    async def record_login_failure(
        self,
        key: str,
        *,
        max_attempts: int,
        window_seconds: float,
        lockout_seconds: float,
    ) -> bool:
        fkey = f"{_LOGIN_FAIL_PREFIX}{key}"
        count = int(await self.client.incr(fkey))
        if count == 1:
            await self.client.expire(fkey, max(1, int(window_seconds)))
        if count >= max_attempts:
            await self.client.setex(
                f"{_LOGIN_LOCK_PREFIX}{key}", max(1, int(lockout_seconds)), "1"
            )
            # Reset the failure counter so post-lockout attempts start clean.
            await self.client.delete(fkey)
            return True
        return False

    async def record_login_success(self, key: str) -> None:
        await self.client.delete(
            f"{_LOGIN_FAIL_PREFIX}{key}", f"{_LOGIN_LOCK_PREFIX}{key}"
        )

    async def login_locked(self, key: str) -> bool:
        return bool(await self.client.exists(f"{_LOGIN_LOCK_PREFIX}{key}"))

    async def login_retry_after(self, key: str) -> int:
        ttl = await self.client.ttl(f"{_LOGIN_LOCK_PREFIX}{key}")
        return int(ttl) if ttl and ttl > 0 else 0
