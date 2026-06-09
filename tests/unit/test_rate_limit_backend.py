"""Unit tests for the pluggable rate-limit backends.

Covers:
  * MemoryRateLimitBackend reproduces the original in-process behavior
    (sliding-window rpm + failed-login lockout).
  * RedisRateLimitBackend issues INCR/EXPIRE against a fake async redis
    client and enforces the rpm limit.
  * The login lockout works through the redis backend (SETEX lock key).
  * Selecting the redis backend with an unreachable client fails fast at
    wiring time (require-redis semantic).

No real Redis is used -- a tiny in-memory fake implements the handful of
async commands the backend relies on.
"""
from __future__ import annotations

import pytest

from opsrag.api.rate_limit_backend import (
    MemoryRateLimitBackend,
    RedisRateLimitBackend,
)


# ---------------------------------------------------------------------------
# Fake redis.asyncio client -- just enough surface for the backend.
# ---------------------------------------------------------------------------
class FakeRedis:
    """In-memory stand-in for ``redis.asyncio.Redis``.

    Tracks call counts so tests can assert INCR/EXPIRE were issued. TTLs are
    stored but do NOT auto-expire (tests drive lock/unlock explicitly via
    ``record_login_success`` / fresh keys), which is enough to verify the
    command wiring and limit enforcement.
    """

    def __init__(self, *, fail_ping: bool = False):
        self.store: dict[str, int] = {}
        self.ttls: dict[str, int] = {}
        self._fail_ping = fail_ping
        self.incr_calls = 0
        self.expire_calls = 0
        self.setex_calls = 0

    async def ping(self):
        if self._fail_ping:
            raise ConnectionError("redis unreachable")
        return True

    async def incr(self, key: str) -> int:
        self.incr_calls += 1
        self.store[key] = self.store.get(key, 0) + 1
        return self.store[key]

    async def expire(self, key: str, seconds: int) -> bool:
        self.expire_calls += 1
        self.ttls[key] = int(seconds)
        return True

    async def ttl(self, key: str) -> int:
        if key not in self.store and key not in self.ttls:
            return -2
        return self.ttls.get(key, -1)

    async def setex(self, key: str, seconds: int, value) -> bool:
        self.setex_calls += 1
        self.store[key] = 1
        self.ttls[key] = int(seconds)
        return True

    async def delete(self, *keys: str) -> int:
        n = 0
        for k in keys:
            if k in self.store:
                del self.store[k]
                n += 1
            self.ttls.pop(k, None)
        return n

    async def exists(self, key: str) -> int:
        return 1 if key in self.store else 0


# ===========================================================================
# Memory backend -- matches the original behavior.
# ===========================================================================
async def test_memory_rpm_allows_under_limit_then_blocks():
    be = MemoryRateLimitBackend()
    # limit=3 in a 60s window: first three allowed, fourth blocked.
    d1 = await be.hit("k", 3, 60.0)
    d2 = await be.hit("k", 3, 60.0)
    d3 = await be.hit("k", 3, 60.0)
    d4 = await be.hit("k", 3, 60.0)
    assert (d1.allowed, d2.allowed, d3.allowed) == (True, True, True)
    # remaining counts down 2,1,0 as the original middleware reported.
    assert (d1.remaining, d2.remaining, d3.remaining) == (2, 1, 0)
    assert d4.allowed is False
    assert d4.remaining == 0
    assert d4.retry_after >= 1


async def test_memory_rpm_is_per_key():
    be = MemoryRateLimitBackend()
    assert (await be.hit("a", 1, 60.0)).allowed is True
    assert (await be.hit("a", 1, 60.0)).allowed is False
    # A different key has its own bucket.
    assert (await be.hit("b", 1, 60.0)).allowed is True


async def test_memory_login_lockout_and_reset():
    be = MemoryRateLimitBackend()
    kw = dict(max_attempts=3, window_seconds=300.0, lockout_seconds=900.0)
    assert await be.record_login_failure("k", **kw) is False
    assert await be.record_login_failure("k", **kw) is False
    # 3rd failure trips the lockout.
    assert await be.record_login_failure("k", **kw) is True
    assert await be.login_locked("k") is True
    assert await be.login_retry_after("k") > 0
    # Success clears it.
    await be.record_login_success("k")
    assert await be.login_locked("k") is False
    assert await be.login_retry_after("k") == 0


# ===========================================================================
# Redis backend -- INCR/EXPIRE wiring + limit enforcement.
# ===========================================================================
async def test_redis_rpm_issues_incr_expire_and_enforces_limit():
    fake = FakeRedis()
    be = RedisRateLimitBackend(client=fake)

    d1 = await be.hit("user1", 2, 60.0)
    d2 = await be.hit("user1", 2, 60.0)
    d3 = await be.hit("user1", 2, 60.0)

    assert d1.allowed is True and d2.allowed is True
    assert d3.allowed is False  # 3rd request exceeds limit=2
    assert d3.retry_after >= 1

    # INCR on every hit; EXPIRE armed exactly once (first hit of the window).
    assert fake.incr_calls == 3
    assert fake.expire_calls == 1
    # remaining counts down from the limit.
    assert d1.remaining == 1
    assert d2.remaining == 0


async def test_redis_rpm_separate_windows_have_independent_keys():
    fake = FakeRedis()
    be = RedisRateLimitBackend(client=fake)
    # Two different logical keys never collide.
    assert (await be.hit("a", 1, 60.0)).allowed is True
    assert (await be.hit("a", 1, 60.0)).allowed is False
    assert (await be.hit("b", 1, 60.0)).allowed is True


async def test_redis_login_lockout_via_setex():
    fake = FakeRedis()
    be = RedisRateLimitBackend(client=fake)
    kw = dict(max_attempts=3, window_seconds=300.0, lockout_seconds=900.0)

    assert await be.record_login_failure("k", **kw) is False
    assert await be.record_login_failure("k", **kw) is False
    # 3rd failure writes the lockout key via SETEX.
    assert await be.record_login_failure("k", **kw) is True
    assert fake.setex_calls == 1
    assert await be.login_locked("k") is True
    # retry_after reflects the lockout TTL.
    assert await be.login_retry_after("k") == 900

    # A successful login clears both the failure counter and the lock.
    await be.record_login_success("k")
    assert await be.login_locked("k") is False
    assert await be.login_retry_after("k") == 0


# ===========================================================================
# Login limiter (auth layer) honors the redis backend.
# ===========================================================================
async def test_login_rate_limiter_locks_out_via_redis_backend():
    pytest.importorskip("pwdlib")  # auth.login pulls pwdlib (login extra); skip if absent
    from opsrag.auth.login import LoginRateLimiter

    fake = FakeRedis()
    be = RedisRateLimitBackend(client=fake)
    rl = LoginRateLimiter(
        max_attempts=3, window_seconds=300.0, lockout_seconds=900.0, backend=be
    )

    assert await rl.is_locked_async("k") is False
    assert await rl.record_failure_async("k") is False
    assert await rl.record_failure_async("k") is False
    assert await rl.record_failure_async("k") is True  # 3rd locks
    assert await rl.is_locked_async("k") is True
    assert await rl.retry_after_async("k") == 900
    assert fake.setex_calls == 1

    await rl.record_success_async("k")
    assert await rl.is_locked_async("k") is False


async def test_login_rate_limiter_memory_path_unchanged_without_backend():
    """No backend -> async wrappers fall through to the in-process logic,
    keeping the existing (synchronous) behavior byte-identical."""
    pytest.importorskip("pwdlib")  # auth.login pulls pwdlib (login extra); skip if absent
    from opsrag.auth.login import LoginRateLimiter

    rl = LoginRateLimiter(max_attempts=3)
    assert await rl.record_failure_async("k") is False
    assert await rl.record_failure_async("k") is False
    assert await rl.record_failure_async("k") is True
    assert await rl.is_locked_async("k") is True
    # The synchronous surface still works identically.
    rl2 = LoginRateLimiter(max_attempts=2)
    assert rl2.record_failure("x") is False
    assert rl2.record_failure("x") is True
    assert rl2.is_locked("x") is True


# ===========================================================================
# Wiring: selecting redis with an unreachable client fails fast.
# ===========================================================================
def test_build_redis_backend_fails_fast_when_unreachable(monkeypatch):
    from opsrag.api import server
    from opsrag.config import OpsRAGConfig

    cfg = OpsRAGConfig()
    cfg.api.rate_limit_backend = "redis"
    cfg.api.redis_url_env = "OPSRAG_TEST_REDIS_URL"
    monkeypatch.setenv("OPSRAG_TEST_REDIS_URL", "redis://unreachable:6379/0")

    # Avoid touching the real redis client: hand back a fake that fails PING.
    monkeypatch.setattr(
        server, "make_redis_client", lambda url: FakeRedis(fail_ping=True)
    )

    with pytest.raises(RuntimeError) as exc:
        server._build_rate_limit_backend(cfg)
    assert "redis" in str(exc.value).lower()


def test_build_redis_backend_fails_fast_when_url_missing(monkeypatch):
    from opsrag.api import server
    from opsrag.config import OpsRAGConfig

    cfg = OpsRAGConfig()
    cfg.api.rate_limit_backend = "redis"
    cfg.api.redis_url_env = "OPSRAG_TEST_REDIS_URL_MISSING"
    monkeypatch.delenv("OPSRAG_TEST_REDIS_URL_MISSING", raising=False)

    with pytest.raises(RuntimeError) as exc:
        server._build_rate_limit_backend(cfg)
    assert "OPSRAG_TEST_REDIS_URL_MISSING" in str(exc.value)


def test_build_memory_backend_is_default_and_does_no_redis(monkeypatch):
    from opsrag.api import server
    from opsrag.config import OpsRAGConfig

    cfg = OpsRAGConfig()
    assert cfg.api.rate_limit_backend == "memory"

    # If the memory path ever tried to build a redis client, this would raise.
    def _boom(url):  # pragma: no cover - must NOT be called
        raise AssertionError("memory backend must not touch redis")

    monkeypatch.setattr(server, "make_redis_client", _boom)
    be = server._build_rate_limit_backend(cfg)
    assert isinstance(be, MemoryRateLimitBackend)
