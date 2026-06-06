"""Generic OIDC Bearer-token verifier.

Replaces the upstream Pomerium-specific path (``opsrag.auth.pomerium``)
with a provider-neutral OIDC verifier that discovers JWKS from any
standards-compliant issuer (Dex, Keycloak, Okta, Auth0, Azure AD,
Google, ...). Wire it up via the ``auth`` block in ``config.yaml``::

    auth:
      issuer: https://idp.example.com         # required
      audience: opsrag                        # required
      jwks_cache_seconds: 300                 # default

At startup the app factory builds one ``OIDCVerifier`` from
``settings.auth`` and attaches it to ``app.state.oidc_verifier``. The
``opsrag.auth.middleware`` dependency reads the verifier off
``request.app.state`` and verifies the Bearer token on every non-health
request.

What the verifier does:

1. On first use (or on JWKS cache miss / TTL expiry), GETs
   ``<issuer>/.well-known/openid-configuration`` and reads
   ``jwks_uri``.
2. Fetches the JWKS document; caches the JWKs by ``kid`` with a TTL.
3. Verifies each incoming JWT's signature against the matching JWK,
   the ``iss`` claim against the configured issuer, the ``aud`` claim
   against the configured audience, and the ``exp`` claim against the
   wall clock.
4. Returns a ``CurrentUser`` constructed from the standard OIDC claims
   (``sub``, ``email``, ``name``, ``picture``, ``groups`` / ``roles``).
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
import jwt
from fastapi import HTTPException

_log = logging.getLogger("opsrag.auth.oidc")


# Standard OIDC algs we'll accept. RS256 covers ~all public IdPs;
# ES256 covers most Cloud IdPs and Dex; the rest are common enough
# that we accept them by default. Operators with stricter requirements
# can override via ``OIDCVerifier(algorithms=...)``.
DEFAULT_ALGORITHMS: tuple[str, ...] = (
    "RS256", "RS384", "RS512",
    "ES256", "ES384", "ES512",
    "PS256", "PS384", "PS512",
)


@dataclass(frozen=True)
class CurrentUser:
    """Identity attached to the current request.

    Always-present sentinel -- handlers can read attributes without
    None-checks. Use :meth:`anonymous` for the unauthenticated case
    (only reachable when the route opts out of auth, e.g. /healthz).

    RBAC: ``roles`` and ``scopes`` carry the resolved authorization
    state. They are computed once (from the IdP ``groups`` claim plus
    the configured ``role_mappings`` and the ``is_admin`` signal) by
    :func:`opsrag.auth.scopes.attach_authz` / the auth dependency, and
    then read by :func:`opsrag.auth.scopes.has_scope` and the
    ``require_scope`` route guards. ``has_scope`` is the single
    authoritative check shared by the ``/me`` payload and the guards so
    the UI never shows nav the server then 403s.

    In OPEN mode the user carries every scope (no enforcement); see
    :meth:`anonymous`, which is the open-mode identity and returns all
    scopes.
    """

    sub: str | None
    email: str | None
    name: str | None
    picture_url: str | None
    groups: tuple[str, ...]
    is_anonymous: bool
    roles: frozenset[str] = field(default_factory=frozenset)
    scopes: frozenset[str] = field(default_factory=frozenset)

    @classmethod
    def anonymous(cls) -> CurrentUser:
        """Open-mode / unauthenticated identity.

        Carries ALL scopes so that, in open mode (no auth configured),
        every request passes every ``require_scope`` guard -- preserving
        today's zero-config behavior where everyone can do everything.
        ``ALL_SCOPES`` is imported lazily to avoid an import cycle
        (scopes.py imports CurrentUser).
        """
        from opsrag.auth.scopes import ALL_SCOPES

        return cls(
            sub=None,
            email=None,
            name=None,
            picture_url=None,
            groups=(),
            is_anonymous=True,
            roles=frozenset(),
            scopes=frozenset(ALL_SCOPES),
        )

    def with_authz(
        self, *, roles: frozenset[str], scopes: frozenset[str]
    ) -> CurrentUser:
        """Return a copy of this user with resolved ``roles``/``scopes``.

        ``CurrentUser`` is frozen, so authz resolution produces a new
        instance rather than mutating in place."""
        return CurrentUser(
            sub=self.sub,
            email=self.email,
            name=self.name,
            picture_url=self.picture_url,
            groups=self.groups,
            is_anonymous=self.is_anonymous,
            roles=roles,
            scopes=scopes,
        )

    def is_member_of(self, group: str | None) -> bool:
        """True iff this user's groups claim contains ``group``.

        Anonymous users are never group members; callers with no group
        configured are never group members either (fail-closed)."""
        if self.is_anonymous or not group:
            return False
        return group in self.groups

    def has_scope(self, scope: str) -> bool:
        """True iff ``scope`` is in this user's resolved scope set.

        Delegates to :func:`opsrag.auth.scopes.has_scope` so there is a
        single authoritative implementation."""
        from opsrag.auth.scopes import has_scope

        return has_scope(self, scope)

    # --- Back-compat bridge for the legacy Pomerium-shape call sites --
    # ``opsrag.api.routes`` (and usage attribution) still read ``.oid``;
    # the OIDC subject IS the stable user id, so alias it. This lets the
    # dual ``get_current_user_dep`` converge on this one OIDC shape
    # without a same-commit rewrite of every ``.oid`` reader. Remove the
    # alias once routes.py migrates to ``.sub`` (scheduled T057-T060).
    @property
    def oid(self) -> str | None:
        return self.sub


class OIDCVerifier:
    """OIDC discovery + JWT verification with a cached JWKS.

    Construct once at app startup; reuse for the lifetime of the
    process. Thread-safe -- the cache lock protects against the JWKS
    being fetched twice on a cold cache under concurrent load.
    """

    def __init__(
        self,
        *,
        issuer: str,
        audience: str,
        algorithms: tuple[str, ...] = DEFAULT_ALGORITHMS,
        jwks_cache_seconds: int = 300,
        http_timeout_seconds: float = 5.0,
    ) -> None:
        if not issuer:
            raise ValueError("OIDCVerifier requires an issuer URL")
        if not audience:
            raise ValueError("OIDCVerifier requires an audience")
        self._issuer = issuer.rstrip("/")
        self._audience = audience
        self._algorithms = list(algorithms)
        self._jwks_cache_seconds = max(1, jwks_cache_seconds)
        self._http_timeout = http_timeout_seconds
        self._jwks_uri: str | None = None
        # kid -> parsed key (loaded by PyJWT's algorithm classes).
        self._keys: dict[str, Any] = {}
        self._jwks_fetched_at: float = 0.0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Discovery + JWKS caching.
    # ------------------------------------------------------------------
    @property
    def issuer(self) -> str:
        return self._issuer

    @property
    def audience(self) -> str:
        return self._audience

    def _discovery_url(self) -> str:
        return f"{self._issuer}/.well-known/openid-configuration"

    def _fetch_discovery(self) -> dict[str, Any]:
        url = self._discovery_url()
        resp = httpx.get(url, timeout=self._http_timeout)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict) or "jwks_uri" not in data:
            raise ValueError(
                f"OIDC discovery at {url} is missing 'jwks_uri'"
            )
        # Spec: the discovered issuer MUST match the requested issuer.
        # Mismatch is a misconfiguration or worse (DNS hijack); refuse.
        disc_issuer = (data.get("issuer") or "").rstrip("/")
        if disc_issuer and disc_issuer != self._issuer:
            raise ValueError(
                "OIDC discovery issuer mismatch: configured "
                f"{self._issuer!r}, discovery returned {disc_issuer!r}"
            )
        return data

    def _fetch_jwks(self, jwks_uri: str) -> dict[str, Any]:
        resp = httpx.get(jwks_uri, timeout=self._http_timeout)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict) or "keys" not in data:
            raise ValueError(
                f"unexpected JWKS shape at {jwks_uri}: missing 'keys' array"
            )
        return data

    def _load_keys(self, jwks: dict[str, Any]) -> dict[str, Any]:
        """Parse a JWKS document into ``{kid: key}``.

        Algorithm class is picked per-JWK from the ``kty`` / ``alg``
        fields, so a mixed issuer (some RSA, some ECDSA keys) works.
        """
        from jwt.algorithms import (
            ECAlgorithm,
            RSAAlgorithm,
        )
        out: dict[str, Any] = {}
        for jwk in jwks.get("keys", []):
            kid = jwk.get("kid")
            kty = (jwk.get("kty") or "").upper()
            if not kid:
                continue
            try:
                if kty == "RSA":
                    out[kid] = RSAAlgorithm.from_jwk(jwk)
                elif kty == "EC":
                    out[kid] = ECAlgorithm.from_jwk(jwk)
                else:
                    _log.debug(
                        "skipping JWK kid=%s with unsupported kty=%s",
                        kid, kty,
                    )
            except Exception as exc:  # noqa: BLE001
                _log.warning(
                    "failed to parse jwk kid=%s: %s", kid, exc,
                )
        return out

    def _refresh_keys(self) -> None:
        """Refresh discovery + JWKS under the lock."""
        with self._lock:
            now = time.monotonic()
            if (
                self._keys
                and now - self._jwks_fetched_at < self._jwks_cache_seconds
            ):
                return  # another thread refreshed
            if self._jwks_uri is None:
                discovery = self._fetch_discovery()
                self._jwks_uri = discovery["jwks_uri"]
            assert self._jwks_uri is not None
            jwks = self._fetch_jwks(self._jwks_uri)
            self._keys = self._load_keys(jwks)
            self._jwks_fetched_at = now
            if not self._keys:
                _log.warning(
                    "OIDC JWKS at %s contained no usable keys",
                    self._jwks_uri,
                )

    def _key_for(self, kid: str) -> Any:
        # Cold cache or TTL expired? Refresh.
        now = time.monotonic()
        if (
            not self._keys
            or kid not in self._keys
            or now - self._jwks_fetched_at >= self._jwks_cache_seconds
        ):
            self._refresh_keys()
        try:
            return self._keys[kid]
        except KeyError as exc:
            raise jwt.PyJWTError(f"unknown JWT kid: {kid}") from exc

    # ------------------------------------------------------------------
    # Verification.
    # ------------------------------------------------------------------
    def verify(self, token: str) -> dict[str, Any]:
        """Verify a Bearer token and return the validated claims.

        Raises ``HTTPException(401)`` on any verification failure; the
        detail string is short and does not echo the token.
        """
        if not token:
            raise HTTPException(status_code=401, detail="missing token")
        try:
            header = jwt.get_unverified_header(token)
        except jwt.PyJWTError:
            raise HTTPException(status_code=401, detail="malformed token")
        kid = header.get("kid")
        if not kid:
            raise HTTPException(status_code=401, detail="missing kid")
        try:
            key = self._key_for(kid)
        except jwt.PyJWTError:
            raise HTTPException(status_code=401, detail="unknown kid")
        try:
            claims = jwt.decode(
                token,
                key=key,
                algorithms=self._algorithms,
                audience=self._audience,
                issuer=self._issuer,
                options={
                    "require": ["exp", "iss", "aud", "sub"],
                    "verify_signature": True,
                    "verify_exp": True,
                    "verify_iss": True,
                    "verify_aud": True,
                },
            )
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="token expired")
        except jwt.InvalidAudienceError:
            raise HTTPException(status_code=401, detail="invalid audience")
        except jwt.InvalidIssuerError:
            raise HTTPException(status_code=401, detail="invalid issuer")
        except jwt.PyJWTError:
            raise HTTPException(status_code=401, detail="invalid token")
        return claims

    def verify_to_user(self, token: str) -> CurrentUser:
        """Verify a Bearer token and return the resulting ``CurrentUser``.

        Extracts the standard OIDC claims (``sub``, ``email``, ``name``,
        ``picture``, ``groups`` or ``roles``). Non-standard claim names
        are not supported; an issuer that emits non-standard claims
        should be configured to also emit the standard ones, or a thin
        adapter should normalise them before this verifier runs.
        """
        claims = self.verify(token)
        groups_claim = claims.get("groups") or claims.get("roles") or ()
        if isinstance(groups_claim, str):
            groups_tuple = (groups_claim,)
        else:
            groups_tuple = tuple(str(g) for g in groups_claim)
        return CurrentUser(
            sub=str(claims["sub"]),
            email=claims.get("email"),
            name=claims.get("name"),
            picture_url=claims.get("picture"),
            groups=groups_tuple,
            is_anonymous=False,
        )


# ---------------------------------------------------------------------------
# Factory helpers.
# ---------------------------------------------------------------------------
def build_verifier_from_settings(auth_cfg: Any) -> OIDCVerifier:
    """Construct an ``OIDCVerifier`` from a Pydantic ``AuthConfig``-shaped
    object. Accepts any object exposing ``issuer``, ``audience``, and
    ``jwks_cache_seconds`` attributes (duck-typed so tests can pass a
    simple dataclass)."""
    return OIDCVerifier(
        issuer=auth_cfg.issuer,
        audience=auth_cfg.audience,
        jwks_cache_seconds=getattr(auth_cfg, "jwks_cache_seconds", 300),
    )
