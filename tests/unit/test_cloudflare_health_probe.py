"""Cloudflare health-probe auth + 401/403 diagnostics.

Debugged live 2026-07-13: the `/readyz` prober GETs the Cloudflare health
URL WITHOUT any Authorization header, so backend logs show a 400 every
10s (readinessProbe period) regardless of token health -- pure noise that
misdirected an actual token-scope incident. Two fixes under test:

1. The probe sends the integration's auth headers when the registry entry
   declares a `health_headers_fn`, and the probe URL is token-type
   agnostic (`/zones?per_page=1` works for user AND account tokens --
   `/user/tokens/verify` 401s for healthy account-owned tokens).
   Readiness semantics stay reachability-only: an auth failure must NOT
   flip the pod NotReady (a third-party token expiry may not take the
   backend out of rotation) -- it surfaces in the detail string instead.

2. `_cf_get` no longer reports 401 (invalid/expired token) with the same
   "missing the required scope" message as 403 -- that message sent the
   operator to fix scopes while the token itself was the problem.
"""
from __future__ import annotations

import pytest

from opsrag.mcp import cloudflare as cf
from opsrag.mcp.registry import REGISTRY


class _FakeResponse:
    def __init__(self, status_code: int, text: str = "") -> None:
        self.status_code = status_code
        self.text = text


class _FakeAsyncClient:
    """Stands in for httpx.AsyncClient; records the request it saw."""

    last_headers: dict | None = None
    next_status: int = 200

    def __init__(self, *a, **kw) -> None: ...

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url, headers=None, params=None):
        type(self).last_headers = headers
        return _FakeResponse(type(self).next_status)


@pytest.fixture
def _fake_httpx(monkeypatch):
    import httpx

    _FakeAsyncClient.last_headers = None
    _FakeAsyncClient.next_status = 200
    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)
    yield _FakeAsyncClient


@pytest.fixture
def _bound_cf():
    cf.bind(token="test-token-value")
    yield
    cf.bind(token=None)  # reset module globals


# -- registry wiring ---------------------------------------------------------


def test_registry_cloudflare_probe_is_token_type_agnostic():
    integ = REGISTRY["cloudflare"]
    # /user/tokens/verify 401s for healthy ACCOUNT-owned tokens; probe the
    # capability the tools actually need instead.
    assert integ.health_url_template == (
        "https://api.cloudflare.com/client/v4/zones?per_page=1"
    )
    assert integ.health_headers_fn is not None


def test_registry_headers_fn_returns_bearer_when_bound(_bound_cf):
    headers = REGISTRY["cloudflare"].health_headers_fn()
    assert headers == {"Authorization": "Bearer test-token-value"}


def test_registry_headers_fn_empty_when_unbound():
    cf.bind(token=None)
    assert REGISTRY["cloudflare"].health_headers_fn() == {}


# -- probe behavior ----------------------------------------------------------


@pytest.mark.asyncio
async def test_probe_url_sends_headers(_fake_httpx):
    from opsrag.api.routes_health import _probe_url

    ok, detail = await _probe_url(
        "https://api.cloudflare.com/client/v4/zones?per_page=1",
        headers={"Authorization": "Bearer x"},
    )
    assert ok is True
    assert _fake_httpx.last_headers == {"Authorization": "Bearer x"}
    assert "auth OK" in detail


@pytest.mark.asyncio
async def test_probe_auth_failure_surfaces_in_detail_but_stays_reachable(_fake_httpx):
    from opsrag.api.routes_health import _probe_url

    _fake_httpx.next_status = 401
    ok, detail = await _probe_url(
        "https://api.cloudflare.com/client/v4/zones?per_page=1",
        headers={"Authorization": "Bearer bad"},
    )
    # Reachability-only readiness: token problems must not flip NotReady.
    assert ok is True
    assert "AUTH FAILED" in detail and "401" in detail


@pytest.mark.asyncio
async def test_probe_without_headers_keeps_plain_reachability_wording(_fake_httpx):
    from opsrag.api.routes_health import _probe_url

    _fake_httpx.next_status = 400
    ok, detail = await _probe_url("https://example.com/x")
    assert ok is True
    assert detail == "reachable (HTTP 400)"


# -- 401 vs 403 diagnostics in _cf_get ---------------------------------------


@pytest.mark.asyncio
async def test_cf_get_401_reports_invalid_token_not_scopes(_fake_httpx, _bound_cf):
    _fake_httpx.next_status = 401
    with pytest.raises(cf.MCPCloudflareError) as ei:
        await cf._cf_get("/zones")
    msg = str(ei.value)
    assert "invalid" in msg.lower() or "expired" in msg.lower()
    assert "missing the required scope" not in msg
    # Reason stays "forbidden" so existing tolerate-filters keep working.
    assert ei.value.reason == "forbidden"


@pytest.mark.asyncio
async def test_cf_get_403_still_reports_missing_scope(_fake_httpx, _bound_cf):
    _fake_httpx.next_status = 403
    with pytest.raises(cf.MCPCloudflareError) as ei:
        await cf._cf_get("/zones")
    assert "missing the required scope" in str(ei.value)
    assert ei.value.reason == "forbidden"
