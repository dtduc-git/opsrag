"""Cookie sessions + rotating refresh tokens + CSRF (auth Tier 2).

Per the design decision (DESIGN 1, FINDING 9): COOKIE SESSIONS, not a
self-hosted IdP. On login the backend sets:

* a signed, ``HttpOnly; Secure; SameSite=Lax`` **session cookie**
  carrying the user id + roles, signed (HMAC) with a server key and
  carrying a max-age the middleware enforces; and
* an opaque, high-entropy **refresh token** (also delivered as an
  ``HttpOnly`` cookie). The refresh token is stored ONLY as a SHA-256
  hash at rest (:func:`opsrag.auth.user_store.hash_token`) and ROTATES
  on every use -- presenting a refresh token mints a new one and revokes
  the old (token-reuse detection seam).

CSRF: because the SPA authenticates via a cookie, mutating requests are
protected with the **double-submit** pattern -- a non-``HttpOnly`` CSRF
cookie whose value the SPA echoes in an ``X-CSRF-Token`` header; the
server checks the two match (constant-time).

Signing key (security guard, per the brief): the key comes from a
``SecretStr`` sourced from a **path or env var only**. Inline key
material in config is REFUSED at construction time -- a key pasted into
YAML would be logged on a Pydantic validation error and committed to
git. :func:`load_signing_key` enforces "path/env, never inline".

This module is pure crypto + cookie plumbing; it has no FastAPI route
dependencies, so it unit-tests without a running app. The signer is
``itsdangerous.TimestampSigner`` over a JSON payload (compact, tamper-
evident, with a server-checked timestamp/max-age).
"""
from __future__ import annotations

import base64
import hmac
import json
import os
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from itsdangerous import BadSignature, SignatureExpired, TimestampSigner

# Opaque-token byte length. 32 bytes = 256 bits of entropy -> a
# url-safe-base64 string of ~43 chars. Far beyond brute-force.
_REFRESH_TOKEN_BYTES = 32
_CSRF_TOKEN_BYTES = 32

# Namespace salt so a session signature can never be replayed as a CSRF
# signature (same key, different purpose).
_SESSION_SALT = b"opsrag.auth.session.v1"


class InlineKeyMaterialError(ValueError):
    """Raised when a signing key is provided inline rather than via a
    path/env reference. Inline secrets leak via config dumps + git."""


def load_signing_key(
    *,
    key_path: str | None = None,
    key_env: str | None = None,
    inline: str | None = None,
) -> bytes:
    """Resolve the session signing key from a path or env var ONLY.

    Exactly one of ``key_path`` / ``key_env`` must yield material.
    ``inline`` is accepted only to be REJECTED loudly: if a caller passes
    inline key material we raise :class:`InlineKeyMaterialError` so the
    forbidden path fails fast and visibly rather than silently working
    (and getting committed to git / logged on a validation error).

    Returns the raw key bytes. Raises ``ValueError`` when no source
    resolves to non-empty material.
    """
    if inline:
        raise InlineKeyMaterialError(
            "inline signing key material is forbidden; supply "
            "session.signing_key_path or session.signing_key_env instead"
        )
    if key_path:
        p = Path(key_path)
        if not p.exists():
            raise ValueError(f"session signing key path does not exist: {key_path}")
        data = p.read_bytes().strip()
        if not data:
            raise ValueError(f"session signing key file is empty: {key_path}")
        return data
    if key_env:
        val = os.environ.get(key_env, "")
        if not val:
            raise ValueError(
                f"session signing key env var {key_env!r} is unset or empty"
            )
        return val.encode("utf-8")
    raise ValueError(
        "no signing key source: set session.signing_key_path or "
        "session.signing_key_env"
    )


@dataclass(frozen=True)
class SessionPayload:
    """The verified contents of a session cookie."""

    user_id: str
    email: str | None
    roles: tuple[str, ...]


def generate_opaque_token(nbytes: int = _REFRESH_TOKEN_BYTES) -> str:
    """A url-safe, high-entropy opaque token (refresh / CSRF)."""
    return secrets.token_urlsafe(nbytes)


class SessionManager:
    """Mints + verifies the signed session cookie, the rotating refresh
    token, and the CSRF token. One instance per process; thread-safe
    (``itsdangerous`` signers are stateless after construction).

    Args:
      signing_key: raw key bytes (use :func:`load_signing_key`).
      session_ttl_seconds: session-cookie max-age (default 15 min).
      refresh_ttl_seconds: refresh-token lifetime (default 14 days).
      cookie_secure: set the ``Secure`` flag (default True; tests/local
        http can pass False).
      cookie_samesite: ``lax`` (default), ``strict``, or ``none``.
      cookie_domain: optional cookie ``Domain`` attribute.
    """

    SESSION_COOKIE = "opsrag_session"
    REFRESH_COOKIE = "opsrag_refresh"
    CSRF_COOKIE = "opsrag_csrf"
    CSRF_HEADER = "x-csrf-token"

    def __init__(
        self,
        signing_key: bytes,
        *,
        session_ttl_seconds: int = 900,
        refresh_ttl_seconds: int = 14 * 24 * 3600,
        cookie_secure: bool = True,
        cookie_samesite: str = "lax",
        cookie_domain: str | None = None,
    ) -> None:
        if not signing_key:
            raise ValueError("SessionManager requires a non-empty signing key")
        self._signer = TimestampSigner(signing_key, salt=_SESSION_SALT)
        self._session_ttl = session_ttl_seconds
        self._refresh_ttl = refresh_ttl_seconds
        self.cookie_secure = cookie_secure
        self.cookie_samesite = cookie_samesite
        self.cookie_domain = cookie_domain

    # ------------------------------------------------------------------
    # Session token (signed, stateless).
    # ------------------------------------------------------------------
    def mint_session(
        self, *, user_id: str, email: str | None, roles: tuple[str, ...]
    ) -> str:
        """Sign a session payload into a compact cookie value."""
        payload = {
            "uid": user_id,
            "email": email,
            "roles": list(roles),
        }
        raw = base64.urlsafe_b64encode(
            json.dumps(payload, separators=(",", ":")).encode("utf-8")
        )
        signed = self._signer.sign(raw)
        return signed.decode("ascii")

    def verify_session(self, token: str | None) -> SessionPayload | None:
        """Verify + decode a session cookie value.

        Returns ``None`` on a missing, tampered, or expired token (the
        signer enforces ``max_age`` against its embedded timestamp). Any
        decode failure is treated as "no valid session" -- never raises
        into the request path."""
        if not token:
            return None
        try:
            raw = self._signer.unsign(
                token.encode("ascii"), max_age=self._session_ttl
            )
        except (BadSignature, SignatureExpired):
            return None
        except Exception:
            return None
        try:
            data = json.loads(base64.urlsafe_b64decode(raw).decode("utf-8"))
            return SessionPayload(
                user_id=str(data["uid"]),
                email=data.get("email"),
                roles=tuple(data.get("roles") or ()),
            )
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Refresh token (opaque, server-tracked, rotating, hashed at rest).
    # ------------------------------------------------------------------
    def new_refresh_token(self) -> tuple[str, str, datetime]:
        """Return ``(raw_token, token_hash, expires_at)``.

        Persist ``token_hash`` + ``expires_at`` via the user store; send
        ``raw_token`` to the client in the refresh cookie. The raw token
        is never stored server-side."""
        from opsrag.auth.user_store import hash_token

        raw = generate_opaque_token()
        expires = datetime.now(UTC) + timedelta(seconds=self._refresh_ttl)
        return raw, hash_token(raw), expires

    # ------------------------------------------------------------------
    # CSRF double-submit.
    # ------------------------------------------------------------------
    def new_csrf_token(self) -> str:
        return generate_opaque_token(_CSRF_TOKEN_BYTES)

    @staticmethod
    def verify_csrf(cookie_value: str | None, header_value: str | None) -> bool:
        """Constant-time double-submit check: the CSRF cookie value must
        equal the value echoed in the request header."""
        if not cookie_value or not header_value:
            return False
        return hmac.compare_digest(cookie_value, header_value)

    # ------------------------------------------------------------------
    # Cookie helpers (set/clear on a Starlette/FastAPI Response).
    # ------------------------------------------------------------------
    def set_login_cookies(
        self,
        response: Any,
        *,
        session_token: str,
        refresh_token: str,
        csrf_token: str,
    ) -> None:
        """Set the session, refresh, and CSRF cookies on ``response``.

        The session + refresh cookies are ``HttpOnly`` (JS cannot read
        them). The CSRF cookie is intentionally NOT ``HttpOnly`` so the
        SPA can read it and echo it in the ``X-CSRF-Token`` header
        (double-submit)."""
        common = {
            "secure": self.cookie_secure,
            "samesite": self.cookie_samesite,
        }
        if self.cookie_domain:
            common["domain"] = self.cookie_domain
        response.set_cookie(
            self.SESSION_COOKIE,
            session_token,
            max_age=self._session_ttl,
            httponly=True,
            path="/",
            **common,
        )
        response.set_cookie(
            self.REFRESH_COOKIE,
            refresh_token,
            max_age=self._refresh_ttl,
            httponly=True,
            # path="/" (NOT the scoped "/auth/refresh"): the SPA reaches this
            # backend under an "/api" reverse-proxy prefix that nginx strips
            # before the app, so the browser-visible URL is "/api/auth/refresh".
            # A cookie scoped to "/auth/refresh" fails RFC 6265 path-match
            # against "/api/auth/refresh" and is NEVER sent -> silent refresh
            # cannot work. "/" matches the session + CSRF cookies and is prefix-
            # agnostic. The token stays HttpOnly + single-use (rotated on every
            # refresh), so riding along on all requests is a negligible exposure
            # delta for a value JS cannot read. Keep in sync with clear_cookies.
            path="/",
            **common,
        )
        response.set_cookie(
            self.CSRF_COOKIE,
            csrf_token,
            max_age=self._refresh_ttl,
            httponly=False,
            path="/",
            **common,
        )

    def clear_cookies(self, response: Any) -> None:
        """Expire all auth cookies (logout)."""
        for name, path in (
            (self.SESSION_COOKIE, "/"),
            (self.REFRESH_COOKIE, "/"),   # must match set_login_cookies (was "/auth/refresh")
            (self.CSRF_COOKIE, "/"),
        ):
            response.delete_cookie(
                name,
                path=path,
                domain=self.cookie_domain,
                secure=self.cookie_secure,
                samesite=self.cookie_samesite,
            )
