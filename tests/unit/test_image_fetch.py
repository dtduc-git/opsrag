"""FIX 3 + FIX 4 — hardened channel image fetch (SSRF + DoS + token scrub).

These tests exercise ``opsrag.channels.image_fetch.fetch_image_bytes`` -- the
shared helper all four channel adapters delegate to. The guards under test:

  * scheme allow-list (https only);
  * SSRF host/IP block (private / loopback / link-local / reserved /
    multicast / unspecified resolved addresses are refused) -- ``getaddrinfo``
    is monkeypatched so NO real DNS / network ever happens;
  * absolute size ceiling (streamed body aborts past ``hard_max_bytes``;
    ``Content-Length`` is rejected up front);
  * credential scrubbing -- a raised error never carries the URL path
    (Telegram bot token lives in the path) or auth headers.
"""
from __future__ import annotations

import socket

import pytest

from opsrag.channels.image_fetch import (
    DEFAULT_HARD_MAX_BYTES,
    ImageFetchError,
    fetch_image_bytes,
)


# ---------------------------------------------------------------------------
# Test doubles: a fake httpx stream + a getaddrinfo that resolves to a chosen IP
# ---------------------------------------------------------------------------
class _FakeStreamResponse:
    """Minimal async-context-manager mimic of ``httpx`` streaming response."""

    def __init__(self, chunks: list[bytes], *, status: int = 200,
                 content_length: int | None = None) -> None:
        self._chunks = chunks
        self.status_code = status
        self.headers = {}
        if content_length is not None:
            self.headers["content-length"] = str(content_length)

    async def __aenter__(self) -> _FakeStreamResponse:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            import httpx
            raise httpx.HTTPStatusError(
                "error", request=None, response=None,  # type: ignore[arg-type]
            )

    async def aiter_bytes(self):  # noqa: ANN201
        for c in self._chunks:
            yield c


class _FakeClient:
    """Stands in for ``httpx.AsyncClient`` -- yields a scripted stream."""

    def __init__(self, response: _FakeStreamResponse | Exception) -> None:
        self._response = response

    async def __aenter__(self) -> _FakeClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    def stream(self, method: str, url: str, **kwargs):  # noqa: ANN201
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


def _patch_resolve(monkeypatch: pytest.MonkeyPatch, ip: str) -> None:
    """Force ``socket.getaddrinfo`` (used by the helper) to return ``ip``."""

    def fake_getaddrinfo(host, *args, **kwargs):  # noqa: ANN001, ANN202
        family = socket.AF_INET6 if ":" in ip else socket.AF_INET
        return [(family, socket.SOCK_STREAM, 6, "", (ip, 0))]

    monkeypatch.setattr(
        "opsrag.channels.image_fetch.socket.getaddrinfo", fake_getaddrinfo,
    )


def _patch_client(monkeypatch: pytest.MonkeyPatch,
                  response: _FakeStreamResponse | Exception) -> None:
    """Force the helper to use our fake httpx client."""
    import opsrag.channels.image_fetch as mod

    def fake_async_client(*args, **kwargs):  # noqa: ANN202
        return _FakeClient(response)

    monkeypatch.setattr(mod.httpx, "AsyncClient", fake_async_client)


# ---------------------------------------------------------------------------
# Scheme guard
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_rejects_non_https_scheme(monkeypatch: pytest.MonkeyPatch) -> None:
    # http:// must be refused before any DNS/network -- getaddrinfo would raise
    # if it were ever reached.
    def boom(*a, **k):  # noqa: ANN002, ANN003, ANN202
        raise AssertionError("getaddrinfo must not be called for non-https")

    monkeypatch.setattr(
        "opsrag.channels.image_fetch.socket.getaddrinfo", boom,
    )
    assert await fetch_image_bytes("http://example.com/x.png") is None


@pytest.mark.asyncio
async def test_rejects_http_metadata_url(monkeypatch: pytest.MonkeyPatch) -> None:
    # The classic cloud-metadata SSRF target over http -> refused by scheme.
    assert await fetch_image_bytes("http://169.254.169.254/latest/meta-data/") is None


# ---------------------------------------------------------------------------
# SSRF IP block
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
@pytest.mark.parametrize("ip", [
    "169.254.169.254",  # link-local cloud metadata
    "127.0.0.1",        # loopback
    "10.0.0.5",         # private 10/8
    "192.168.1.10",     # private 192.168/16
    "::1",              # IPv6 loopback
    "0.0.0.0",          # unspecified
])
async def test_rejects_private_or_special_resolved_ip(
    monkeypatch: pytest.MonkeyPatch, ip: str,
) -> None:
    _patch_resolve(monkeypatch, ip)
    # Client should never be reached; if it is, fail loudly.
    _patch_client(monkeypatch, AssertionError("must not connect to blocked IP"))
    assert await fetch_image_bytes("https://evil.example.com/x.png") is None


@pytest.mark.asyncio
async def test_rejects_ip_literal_host_loopback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Host is already an IP literal -> checked directly (getaddrinfo may still
    # be called but should resolve the literal to itself).
    _patch_resolve(monkeypatch, "127.0.0.1")
    _patch_client(monkeypatch, AssertionError("must not connect"))
    assert await fetch_image_bytes("https://127.0.0.1/x.png") is None


# ---------------------------------------------------------------------------
# DNS-rebinding / TOCTOU: the connection must target the validated IP, and the
# helper must NOT let httpx do a second, independent DNS resolution.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_pins_connection_to_validated_ip_no_second_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fetch must connect to the IP we validated -- not re-resolve the host.

    This is the core DNS-rebinding defence. We make ``getaddrinfo`` return a
    PUBLIC IP (passes validation) and count how many times it is called. After
    the helper picks that IP, the actual connection must target THAT IP, and the
    helper must not trigger a second resolution (which an attacker could rebind
    to a private IP). We assert via the real ``_PinnedIPTransport``: it rewrites
    the request URL host to the pinned IP and sets SNI back to the hostname.
    """
    import httpx

    public_ip = "93.184.216.34"

    call_count = {"n": 0}

    def counting_getaddrinfo(host, *args, **kwargs):  # noqa: ANN001, ANN202
        call_count["n"] += 1
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (public_ip, 0))]

    monkeypatch.setattr(
        "opsrag.channels.image_fetch.socket.getaddrinfo", counting_getaddrinfo,
    )

    captured: dict = {}

    # Intercept the BASE transport (the real socket layer) -- the
    # _PinnedIPTransport.handle_async_request still runs and does the pinning,
    # so we observe exactly the host/SNI/Host the request is sent with, and
    # short-circuit before any real socket is opened.
    async def base_handle(self, request):  # noqa: ANN001, ANN202
        captured["connect_host"] = request.url.host
        captured["sni_hostname"] = request.extensions.get("sni_hostname")
        captured["host_header"] = request.headers.get("Host")
        return httpx.Response(200, content=b"PNGDATA", request=request)

    monkeypatch.setattr(
        httpx.AsyncHTTPTransport, "handle_async_request", base_handle,
    )

    out = await fetch_image_bytes("https://cdn.example.com/x.png")

    assert out == b"PNGDATA"
    # The socket connect targets the VALIDATED (public) IP, not a re-resolved host.
    assert captured["connect_host"] == public_ip
    # TLS SNI + cert verification target the ORIGINAL hostname, not the IP.
    assert captured["sni_hostname"] == "cdn.example.com"
    # Host header preserved for origin-server routing.
    assert captured["host_header"] == "cdn.example.com"
    # getaddrinfo was called exactly once (our validation) -- httpx never
    # performed a second, rebindable resolution.
    assert call_count["n"] == 1


@pytest.mark.asyncio
async def test_pinned_transport_rewrites_host_and_sets_sni() -> None:
    """Unit-level proof the transport pins the IP while keeping hostname TLS.

    Verifies _PinnedIPTransport in isolation: the outgoing request's URL host
    becomes the pinned IP (so the TCP connect goes there), but the SNI extension
    and Host header stay the original hostname (so TLS cert verification and
    origin routing remain correct -- TLS is not weakened).
    """
    import httpx

    from opsrag.channels.image_fetch import _PinnedIPTransport

    transport = _PinnedIPTransport(
        pinned_ip="93.184.216.34", original_host="cdn.example.com",
    )

    seen: dict = {}

    async def base_handle(self, request):  # noqa: ANN001, ANN202
        seen["host"] = request.url.host
        seen["sni"] = request.extensions.get("sni_hostname")
        seen["host_header"] = request.headers.get("Host")
        return httpx.Response(200, content=b"ok", request=request)

    # Call through the real handle_async_request, stubbing only the super().
    import unittest.mock as mock

    with mock.patch.object(
        httpx.AsyncHTTPTransport, "handle_async_request", base_handle,
    ):
        req = httpx.Request("GET", "https://cdn.example.com/x.png")
        resp = await transport.handle_async_request(req)

    assert resp.status_code == 200
    assert seen["host"] == "93.184.216.34"      # connect target pinned to IP
    assert seen["sni"] == "cdn.example.com"      # SNI/cert verify = hostname
    assert seen["host_header"] == "cdn.example.com"  # routing host preserved


# ---------------------------------------------------------------------------
# Happy path (public IP)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_allows_public_host_returns_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_resolve(monkeypatch, "93.184.216.34")  # example.com, public
    _patch_client(monkeypatch, _FakeStreamResponse([b"PNG", b"data"]))
    out = await fetch_image_bytes("https://example.com/x.png")
    assert out == b"PNGdata"


@pytest.mark.asyncio
async def test_passes_headers_through(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_resolve(monkeypatch, "93.184.216.34")
    captured: dict = {}

    class _CapClient(_FakeClient):
        def stream(self, method, url, **kwargs):  # noqa: ANN001, ANN201
            captured["headers"] = kwargs.get("headers")
            return _FakeStreamResponse([b"ok"])

    import opsrag.channels.image_fetch as mod
    monkeypatch.setattr(mod.httpx, "AsyncClient", lambda *a, **k: _CapClient(None))  # type: ignore[arg-type]
    out = await fetch_image_bytes(
        "https://example.com/x.png", headers={"Authorization": "Bearer x"},
    )
    assert out == b"ok"
    assert captured["headers"] == {"Authorization": "Bearer x"}


# ---------------------------------------------------------------------------
# Size ceiling (DoS)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_aborts_when_stream_exceeds_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_resolve(monkeypatch, "93.184.216.34")
    # Three 1 MiB chunks but a 2 MiB ceiling -> must abort, return None.
    chunk = b"x" * (1024 * 1024)
    _patch_client(monkeypatch, _FakeStreamResponse([chunk, chunk, chunk]))
    out = await fetch_image_bytes(
        "https://example.com/big.png", hard_max_bytes=2 * 1024 * 1024,
    )
    assert out is None


@pytest.mark.asyncio
async def test_rejects_oversize_content_length_up_front(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_resolve(monkeypatch, "93.184.216.34")
    # Content-Length says 999 MiB; the body iterator should never be drained.
    resp = _FakeStreamResponse([b"should-not-read"], content_length=999 * 1024 * 1024)
    _patch_client(monkeypatch, resp)
    out = await fetch_image_bytes(
        "https://example.com/huge.png", hard_max_bytes=1024,
    )
    assert out is None


@pytest.mark.asyncio
async def test_default_hard_max_is_generous() -> None:
    # Sanity: the module default ceiling is a real, generous number.
    assert DEFAULT_HARD_MAX_BYTES >= 10 * 1024 * 1024


# ---------------------------------------------------------------------------
# FIX 4 — token / credential scrubbing
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_transport_error_does_not_leak_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx

    token = "123456:SUPER_SECRET_BOT_TOKEN"
    url = f"https://api.telegram.org/file/bot{token}/photos/x.jpg"
    _patch_resolve(monkeypatch, "93.184.216.34")
    # Transport error whose message embeds the full URL (httpx does this).
    _patch_client(monkeypatch, httpx.ConnectError(f"failed connecting to {url}"))

    with pytest.raises(ImageFetchError) as ei:
        await fetch_image_bytes(url, raise_on_error=True)
    msg = str(ei.value)
    assert token not in msg
    assert "/file/bot" not in msg  # path scrubbed entirely
    assert "api.telegram.org" in msg  # host is fine to log


@pytest.mark.asyncio
async def test_default_returns_none_on_transport_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx

    _patch_resolve(monkeypatch, "93.184.216.34")
    _patch_client(monkeypatch, httpx.ConnectError("boom https://x/secret/path"))
    # Default mode (no raise_on_error) swallows -> None, never propagates the URL.
    assert await fetch_image_bytes("https://example.com/x.png") is None
