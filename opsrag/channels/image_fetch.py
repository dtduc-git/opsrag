"""Hardened, shared image fetch for the channel adapters (FIX 3 + FIX 4).

All four channel adapters (Slack / Telegram / Discord / Teams) need to pull an
inbound attachment's bytes over HTTP. A naive ``httpx.get(url); resp.content``
has two security problems:

  * **SSRF** -- Teams' ``contentUrl`` (and, in principle, any attacker-influenced
    URL) could point at an internal address such as the cloud metadata endpoint
    ``http://169.254.169.254/`` or ``http://127.0.0.1/``. Fetching it blindly
    turns the bot into a confused deputy.
  * **Unbounded download (DoS)** -- ``resp.content`` buffers the WHOLE body
    before any size check, so a multi-GB attachment blows memory before the
    dispatcher's configurable ``max_bytes`` check ever runs.

A third, related concern (FIX 4) is **credential leakage**: the Telegram file
URL embeds the bot token in its PATH (``/file/bot<token>/...``) and Slack passes
the bot token in an ``Authorization`` header. ``httpx`` exceptions include the
full request URL, and the dispatcher logs ``err=%s`` -- so a raw transport error
would print the token to the logs.

:func:`fetch_image_bytes` centralises the fix so every adapter inherits it:

  1. **Scheme allow-list** -- only ``https`` is accepted (all four platforms use
     https). Anything else is refused before any DNS or socket work.
  2. **SSRF IP block** -- the host is resolved with ``socket.getaddrinfo`` and
     EVERY resolved address is checked; if any is private / loopback /
     link-local / reserved / multicast / unspecified the fetch is refused. An
     IP-literal host is checked directly (it still goes through getaddrinfo,
     which returns the literal).
  3. **Size ceiling** -- the body is STREAMED and aborted the moment the
     accumulated size exceeds ``hard_max_bytes``; a ``Content-Length`` that
     already exceeds the ceiling is rejected up front so we never start the
     download. This is an absolute safety ceiling -- the dispatcher still
     enforces the precise, configurable per-image ``max_bytes`` as a second
     layer.
  4. **Credential scrub** -- transport / HTTP errors are caught and never
     re-raised with the URL path or headers; only ``scheme + host`` (and an
     HTTP status code where available) appear in :class:`ImageFetchError` /
     log lines, so a Telegram token in the path can never leak.

By default the helper returns ``None`` on any failure (the dispatcher already
degrades a missing image to text-only -- spec FR-014). Pass
``raise_on_error=True`` to get a sanitized :class:`ImageFetchError` instead
(useful in tests / when a caller wants to log it itself).

DNS-rebinding / TOCTOU fix: between our ``getaddrinfo`` validation and httpx's
own DNS resolution + connect there used to be a time-of-check-to-time-of-use
window. An attacker controlling DNS could return a PUBLIC IP to our check and a
PRIVATE IP to httpx's connect (DNS rebinding) -> SSRF to an internal service.
We close that window with :class:`_PinnedIPTransport`: we resolve + validate the
host ONCE, pick a validated IP, and make httpx connect to THAT EXACT IP -- httpx
never does a second, independent resolution. Crucially the TLS handshake still
uses the ORIGINAL hostname for SNI and for certificate-name verification (via
httpcore's ``sni_hostname`` request extension, which becomes ``server_hostname``
on the default, hostname-verifying SSL context), and the ``Host`` header stays
the original hostname so the origin server still routes correctly. We pin the
connect target, not the trust anchor -- cert verification is NOT weakened.

A Teams ``contentUrl`` host allowlist (e.g. ``*.teams.microsoft.com`` /
``*.sharepoint.com``) would be a nice-to-have additional layer, but the IP
block is the required protection and is platform-agnostic.
"""
from __future__ import annotations

import ipaddress
import logging
import socket
from urllib.parse import urlsplit

import httpx

_log = logging.getLogger("opsrag.channels.image_fetch")

# Absolute safety ceiling on a single image download (DoS guard). Generous --
# the dispatcher's configurable per-image ``max_bytes`` (VisionConfig) is the
# precise limit; this is just the "never buffer more than this" backstop.
DEFAULT_HARD_MAX_BYTES = 25 * 1024 * 1024  # 25 MiB

# Per-fetch timeout. Generous enough for a large image over a slow link, short
# enough that a hung/slow-loris endpoint can't tie up the worker indefinitely.
_TIMEOUT_S = 30.0

# Streaming read chunk hint (httpx default is fine; named for clarity).
_ALLOWED_SCHEMES = ("https",)


class ImageFetchError(Exception):
    """A sanitized image-fetch failure.

    The message NEVER contains the request path, query, or headers -- only the
    scheme+host (and an HTTP status when known) -- so a credential carried in
    the URL path (Telegram bot token) or an ``Authorization`` header can never
    leak into logs.
    """


def _is_blocked_ip(ip_str: str) -> bool:
    """True iff ``ip_str`` is an address we must refuse (SSRF guard).

    Blocks private, loopback, link-local (incl. 169.254.169.254 cloud
    metadata), reserved, multicast, and unspecified ranges -- IPv4 and IPv6.
    An unparseable address is treated as blocked (fail closed).
    """
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return True
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def _resolve_and_check_host(host: str) -> str | None:
    """Resolve ``host``, validate EVERY address, and return one validated IP.

    Uses ``socket.getaddrinfo`` so an IP-literal host resolves to itself and a
    DNS name resolves to its real addresses. Returns ``None`` (fail closed) if
    resolution fails, returns no addresses, or ANY resolved address is in a
    blocked range. Otherwise returns the FIRST resolved IP (already validated)
    so the caller can pin the connection to that exact address -- closing the
    DNS-rebinding TOCTOU window (we never let httpx re-resolve).

    We validate all addresses but return the first because httpx connects to a
    single host; pinning any one validated IP is sufficient and httpx's own
    retry/connect logic only needs one target.
    """
    if not host:
        return None
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except (socket.gaierror, OSError):
        return None
    if not infos:
        return None
    first_ip: str | None = None
    for info in infos:
        sockaddr = info[4]
        ip_str = sockaddr[0]
        if _is_blocked_ip(ip_str):
            return None  # ANY blocked address => refuse the whole host
        if first_ip is None:
            first_ip = ip_str
    return first_ip


class _PinnedIPTransport(httpx.AsyncHTTPTransport):
    """An ``httpx`` transport that connects to a pre-validated IP, not the DNS.

    The whole point: we already resolved + validated the host's IP(s) with
    ``socket.getaddrinfo``; if we let httpx resolve the hostname again at connect
    time, a DNS-rebinding attacker could swap in a private IP between our check
    and httpx's connect (TOCTOU). So we rewrite the outgoing request's URL host
    to the validated IP -- httpcore then opens the TCP socket to THAT IP and
    performs NO further name resolution.

    TLS is NOT weakened: we set the ``sni_hostname`` request extension to the
    original hostname, which httpcore passes as ``server_hostname`` to the TLS
    handshake on its default SSL context (``check_hostname=True``,
    ``CERT_REQUIRED``). So SNI and certificate-name verification both target the
    real hostname -- exactly as a normal request would -- while the socket goes
    to the IP we vetted. We also keep the ``Host`` header set to the original
    hostname so the origin server routes the request correctly.
    """

    def __init__(self, *, pinned_ip: str, original_host: str, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._pinned_ip = pinned_ip
        self._original_host = original_host

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        # Preserve the original hostname for routing + TLS, then point the
        # connection at the validated IP.
        # 1. Host header -> original hostname (origin-server routing). httpx
        #    derives the default Host header from the URL host, which we are
        #    about to overwrite, so set it explicitly here.
        request.headers["Host"] = self._original_host
        # 2. SNI + certificate verification -> original hostname. Without this,
        #    httpcore would verify the cert against the IP and break TLS for
        #    every real CDN. With it, verification stays against the hostname.
        request.extensions = {**request.extensions, "sni_hostname": self._original_host}
        # 3. Connect target -> the validated IP. Rewriting the URL host makes
        #    httpcore's origin (and thus its TCP connect) use the IP with NO
        #    second DNS lookup -- this is what closes the rebinding window.
        request.url = request.url.copy_with(host=self._pinned_ip)
        return await super().handle_async_request(request)


def _safe_host(url: str) -> str:
    """Extract just the host of ``url`` for safe (credential-free) logging."""
    try:
        return urlsplit(url).hostname or "?"
    except ValueError:
        return "?"


async def fetch_image_bytes(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    hard_max_bytes: int = DEFAULT_HARD_MAX_BYTES,
    raise_on_error: bool = False,
) -> bytes | None:
    """Fetch an image's bytes over https with SSRF + size + credential guards.

    Args:
        url: the image URL. MUST be https; the host must resolve only to public
            addresses (see module docstring).
        headers: optional request headers (e.g. a Slack ``Authorization``
            bearer). NEVER included in any error/log text.
        hard_max_bytes: absolute size ceiling -- the download aborts past this.
        raise_on_error: when True, raise a sanitized :class:`ImageFetchError`
            on failure instead of returning ``None``.

    Returns:
        the image bytes, or ``None`` on any refusal/failure (unless
        ``raise_on_error`` is set).
    """
    host = _safe_host(url)

    def _fail(reason: str, *, status: int | None = None) -> None:
        # Build a message from scheme+host (+status) ONLY -- never the path,
        # query, or headers (Telegram token lives in the path; Slack token in a
        # header). This is the FIX-4 credential scrub.
        detail = f"image fetch refused host={host} reason={reason}"
        if status is not None:
            detail += f" status={status}"
        if raise_on_error:
            raise ImageFetchError(detail)
        _log.warning("%s", detail)

    # --- 1. Scheme allow-list (before any DNS/socket work) ------------------
    parts = urlsplit(url)
    if parts.scheme not in _ALLOWED_SCHEMES:
        _fail(f"scheme={parts.scheme or 'none'}")
        return None

    # --- 2. SSRF IP block (and capture the validated IP to pin) ------------
    validated_ip = _resolve_and_check_host(parts.hostname or "")
    if validated_ip is None:
        _fail("host-resolves-to-blocked-or-unresolvable-ip")
        return None

    # --- 3 + 4. Stream the body with a size ceiling; scrub any error -------
    # Pin the connection to the IP we just validated so httpx cannot re-resolve
    # the hostname (DNS-rebinding TOCTOU defence). TLS SNI + cert verification
    # still target the original hostname -- see _PinnedIPTransport.
    try:
        timeout = httpx.Timeout(_TIMEOUT_S)
        transport = _PinnedIPTransport(
            pinned_ip=validated_ip,
            original_host=parts.hostname or "",
        )
        async with httpx.AsyncClient(
            timeout=timeout, follow_redirects=False, transport=transport,
        ) as client:
            async with client.stream("GET", url, headers=headers) as resp:
                resp.raise_for_status()

                # Early bail on an oversized Content-Length so we never start
                # draining a giant body.
                clen = resp.headers.get("content-length")
                if clen is not None:
                    try:
                        if int(clen) > hard_max_bytes:
                            _fail("content-length-exceeds-ceiling")
                            return None
                    except (TypeError, ValueError):
                        pass  # malformed header -> fall through to streaming guard

                buf = bytearray()
                async for chunk in resp.aiter_bytes():
                    buf.extend(chunk)
                    if len(buf) > hard_max_bytes:
                        # Abort: do NOT keep accumulating. The context manager
                        # closes the connection on exit.
                        _fail("body-exceeds-ceiling")
                        return None
                return bytes(buf)
    except httpx.HTTPStatusError as exc:
        # FIX 4: take ONLY the status code -- never str(exc) (it embeds the URL).
        status = getattr(getattr(exc, "response", None), "status_code", None)
        _fail("http-error", status=status)
        return None
    except (httpx.HTTPError, OSError):
        # FIX 4: never str(exc) here -- httpx transport errors embed the full
        # request URL (Telegram token is in the path). Use host only.
        _fail("transport-error")
        return None
