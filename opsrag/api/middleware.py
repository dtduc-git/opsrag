"""Security and rate-limiting middleware for the OpsRAG API."""
from __future__ import annotations

import time
from collections import defaultdict

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

# Paths that always skip auth and rate limiting.
_PUBLIC_PATHS = frozenset({"/health", "/docs", "/openapi.json", "/redoc"})

# Step 8 -- endpoints that mutate state (or cost meaningful Vertex tokens
# on a per-call basis) and therefore require an API key when keys are
# configured. Read-only endpoints like POST /query are intentionally not
# in this list -- they're cost-tracked but easy to expose to a Slack bot
# without requiring callers to manage a secret.
#
# Each tuple is (HTTP method, path prefix). Prefix is a leading-string
# match on `request.url.path`, so `/sessions/` covers both
# `/sessions/{thread_id}` and `/sessions/{thread_id}/messages` --
# downstream method-routing keeps GET /sessions/* read-only.
# WARNING: APIKeyAuthMiddleware below is NOT registered in api/server.py --
# the live auth seam is the per-route `require_scope(Scope.ADMIN/CHAT)`
# dependency in api/routes.py (OIDC/login-aware). This list therefore gates
# NOTHING at runtime; do not add routes here expecting enforcement. Kept only
# for the (currently unused) optional API-key layer. Route gating lives on the
# handlers, not here.
ADMIN_ROUTES: tuple[tuple[str, str], ...] = (
    ("POST", "/index/repo"),
    ("DELETE", "/sessions/"),
)


class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    """Validates an ``X-API-Key`` header on admin endpoints.

    Semantics:
      - If ``api_keys`` is empty, ALL routes are open (local dev mode).
      - Otherwise, only routes matched by ``admin_routes`` require a
        valid key. Read-only routes (POST /query, GET /usage, etc.)
        stay open even when keys are configured. This lets us deploy
        OpsRAG behind a Slack bot or read-only frontend without forcing
        every caller to manage a secret, while still gating
        state-mutating + expensive-to-trigger operations.
    """

    def __init__(
        self,
        app,
        api_keys: set[str] | None = None,
        admin_routes: tuple[tuple[str, str], ...] | None = None,
    ):
        super().__init__(app)
        self._keys = api_keys or set()
        self._admin_routes = admin_routes if admin_routes is not None else ADMIN_ROUTES

    def _is_admin(self, request: Request) -> bool:
        path = request.url.path
        method = request.method.upper()
        for m, prefix in self._admin_routes:
            if method == m.upper() and path.startswith(prefix):
                return True
        return False

    async def dispatch(self, request: Request, call_next) -> Response:
        if not self._keys:
            return await call_next(request)

        if request.url.path in _PUBLIC_PATHS:
            return await call_next(request)

        # Non-admin routes pass through even when keys are configured.
        if not self._is_admin(request):
            return await call_next(request)

        key = request.headers.get("X-API-Key", "")
        if key not in self._keys:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key (admin endpoint)"},
            )

        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding-window rate limiter keyed by API key or client IP.

    Uses an in-memory store -- suitable for single-process deployments.
    For multi-replica setups, swap in a Redis-backed implementation.
    """

    def __init__(
        self,
        app,
        requests_per_minute: int = 60,
        enabled: bool = True,
    ):
        super().__init__(app)
        self._rpm = requests_per_minute
        self._window = 60.0  # seconds
        self._enabled = enabled
        self._buckets: dict[str, list[float]] = defaultdict(list)

    def _get_key(self, request: Request) -> str:
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"key:{api_key}"
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return f"ip:{forwarded.split(',')[0].strip()}"
        client = request.client
        return f"ip:{client.host}" if client else "ip:unknown"

    async def dispatch(self, request: Request, call_next) -> Response:
        if not self._enabled:
            return await call_next(request)

        if request.url.path in _PUBLIC_PATHS:
            return await call_next(request)

        key = self._get_key(request)
        now = time.monotonic()
        window_start = now - self._window

        # Prune old entries
        timestamps = self._buckets[key]
        self._buckets[key] = [t for t in timestamps if t > window_start]

        if len(self._buckets[key]) >= self._rpm:
            retry_after = int(self._window - (now - self._buckets[key][0])) + 1
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded"},
                headers={"Retry-After": str(retry_after)},
            )

        self._buckets[key].append(now)

        response = await call_next(request)
        remaining = max(0, self._rpm - len(self._buckets[key]))
        response.headers["X-RateLimit-Limit"] = str(self._rpm)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response
