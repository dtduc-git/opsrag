"""M1 -- Pomerium identity verifier + request -> CurrentUser extraction.

Pomerium sits in front of the OpsRAG backend in production and forwards
the verified user's identity as a signed JWT in the
``X-Pomerium-Jwt-Assertion`` request header. The JWT is signed with
ES256 (P-256 ECDSA) by Pomerium and the matching JWKS is exposed at a
well-known URL on the same proxy (typically ``/.well-known/pomerium/jwks.json``).

We trust ONLY the JWT -- never the raw `X-User-Email` / `X-User-Id` style
headers Pomerium also forwards, since those could be spoofed by anyone
who can reach the backend directly (e.g. cluster-internal traffic).

Local dev / CI sets ``tracking_user.enabled = false`` and the entire
identity path becomes a no-op that returns
:class:`CurrentUser.anonymous`.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

import httpx
import jwt
from fastapi import HTTPException, Request

_log = logging.getLogger("opsrag.auth.pomerium")

# JWKS cache TTL -- Pomerium rotates signing keys infrequently (months),
# so a 1h cache is safely under that horizon while still flushing within
# a reasonable window after rotation. The verifier falls back to a
# fresh fetch on signature failure.
_JWKS_TTL_SECONDS = 3600.0


@dataclass(frozen=True)
class CurrentUser:
    """Identity attached to the current FastAPI request.

    Always-present sentinel -- handlers can call :meth:`is_admin` and
    read attributes without None-checks. Use :meth:`anonymous` for the
    unauthenticated / tracking-disabled case.
    """

    oid: str | None
    email: str | None
    name: str | None
    picture_url: str | None
    groups: tuple[str, ...]
    is_anonymous: bool

    @classmethod
    def anonymous(cls) -> CurrentUser:
        return cls(
            oid=None,
            email=None,
            name=None,
            picture_url=None,
            groups=(),
            is_anonymous=True,
        )

    def is_admin_for(self, admin_group_oid: str | None) -> bool:
        """True iff this user's `groups` claim contains the admin group
        OID. Anonymous users are never admin; callers with no admin
        group configured are never admin either (fail-closed)."""
        if self.is_anonymous or not admin_group_oid:
            return False
        return admin_group_oid in self.groups


class PomeriumVerifier:
    """JWT verifier with a cached JWKS.

    Construct once at app startup; reuse for the lifetime of the
    process. Thread-safe -- the cache lock protects against the JWKS
    being fetched twice on a cache miss under concurrent load.
    """

    def __init__(
        self,
        *,
        jwks_url: str,
        expected_audience: str | None = None,
        algorithms: tuple[str, ...] = ("ES256",),
        jwks_ttl_seconds: float = _JWKS_TTL_SECONDS,
        http_timeout_seconds: float = 5.0,
        discovery_fetches: int = 5,
    ) -> None:
        self._jwks_url = jwks_url
        self._expected_audience = expected_audience
        self._algorithms = list(algorithms)
        self._jwks_ttl = jwks_ttl_seconds
        self._http_timeout = http_timeout_seconds
        # Pomerium HA: each `pomerium-authorize` replica generates its
        # own signing key if `signing_key` isn't shared via a Secret.
        # The JWKS endpoint returns whichever replica's key the
        # load-balancer happened to route to. We cope by accumulating
        # every kid we've ever seen, refetching N times on miss to
        # maximize the chance of hitting all replicas.
        self._known_keys: dict[str, Any] = {}
        self._discovery_fetches = max(1, discovery_fetches)
        self._jwks_fetched_at: float = 0.0
        self._lock = threading.Lock()

    def _fetch_jwks(self) -> dict[str, Any]:
        # Sync HTTP -- verifier is called from request paths, but the
        # cache means the first request pays the cost and subsequent
        # ones hit the in-memory copy. PyJWT's PyJWKClient ALSO does a
        # blocking fetch, so this is no worse than the standard pattern.
        resp = httpx.get(self._jwks_url, timeout=self._http_timeout)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict) or "keys" not in data:
            raise ValueError(
                f"unexpected JWKS shape at {self._jwks_url}: missing 'keys' array"
            )
        return data

    def _merge_jwks(self, jwks: dict[str, Any]) -> int:
        """Merge a JWKS response into `_known_keys`. Returns the number
        of NEW kids discovered. Caller holds the lock."""
        from jwt.algorithms import ECAlgorithm
        added = 0
        for jwk in jwks.get("keys", []):
            kid = jwk.get("kid")
            if not kid or kid in self._known_keys:
                continue
            try:
                self._known_keys[kid] = ECAlgorithm.from_jwk(jwk)
                added += 1
            except Exception as exc:  # noqa: BLE001
                _log.warning("failed to parse jwk kid=%s: %s", kid, exc)
        return added

    def _discover_keys(self, *, fetches: int) -> int:
        """Call `_fetch_jwks` `fetches` times and merge unique kids.
        Returns count of new kids learned. Caller holds the lock.

        With Pomerium HA running separate signing keys per replica,
        repeated fetches go through the LB to different pods. After a
        few round-trips we usually see every replica's key.
        """
        total_new = 0
        for _ in range(fetches):
            try:
                jwks = self._fetch_jwks()
            except Exception as exc:  # noqa: BLE001
                _log.warning("jwks fetch failed: %s", exc)
                continue
            total_new += self._merge_jwks(jwks)
        return total_new

    def _ensure_initial(self) -> None:
        """First-call population. Caller holds the lock."""
        if self._jwks_fetched_at == 0.0:
            self._discover_keys(fetches=self._discovery_fetches)
            self._jwks_fetched_at = time.time()

    def _get_jwks(self, *, force_refresh: bool = False) -> dict[str, Any]:
        """Compatibility shim -- returns the union of all keys we've
        ever seen, in the standard `{keys: [...]}` shape. Kept for
        backwards-compat with tests and any external readers.
        """
        with self._lock:
            now = time.time()
            if force_refresh or (now - self._jwks_fetched_at) >= self._jwks_ttl:
                self._discover_keys(fetches=self._discovery_fetches)
                self._jwks_fetched_at = now
            # Rebuild a JWKS-shaped dict from accumulated keys. We only
            # store the parsed public-key objects, so we expose just
            # kids -- callers that need the raw JWK should grab it
            # directly. The kid list is what the error message uses.
            return {"keys": [{"kid": k} for k in self._known_keys]}

    def _public_key_for_kid(self, kid: str, *, force_refresh: bool = False) -> Any:
        with self._lock:
            self._ensure_initial()
            if not force_refresh and kid in self._known_keys:
                return self._known_keys[kid]
            # Miss -> multi-fetch to try to discover the replica that
            # signed this kid. Each refetch goes through the LB and
            # has a chance of landing on a different pomerium-authorize
            # pod.
            self._discover_keys(fetches=self._discovery_fetches)
            self._jwks_fetched_at = time.time()
            return self._known_keys.get(kid)

    def verify(self, jwt_assertion: str) -> dict[str, Any]:
        """Verify the signature + claims of a Pomerium-forwarded JWT.

        Returns the decoded claims dict on success. Raises a
        ``jwt.InvalidTokenError`` subclass on any failure (bad sig,
        expired, audience mismatch, unknown kid).

        On a kid-mismatch we transparently retry once with a fresh
        JWKS fetch -- covers the case where Pomerium just rotated keys
        and our cached copy is stale.
        """
        unverified_header = jwt.get_unverified_header(jwt_assertion)
        kid = unverified_header.get("kid")
        if not kid:
            raise jwt.InvalidTokenError("JWT header missing 'kid'")

        key = self._public_key_for_kid(kid)
        if key is None:
            # Stale cache -> force a refresh and try once more. If still
            # missing, the JWT was signed with a key Pomerium has since
            # rotated out. The user's session cookie is stale; the only
            # fix is to sign out + back in.
            _log.info("jwks miss for kid=%s, force-refreshing", kid)
            key = self._public_key_for_kid(kid, force_refresh=True)
            if key is None:
                # Surface the current JWKS kids so the log line tells
                # an operator exactly what rotated.
                current_kids = [
                    k.get("kid") for k in self._get_jwks().get("keys", [])
                ]
                raise jwt.InvalidTokenError(
                    f"Pomerium session is stale (signed with kid={kid!r} "
                    f"but current JWKS only has {current_kids}). Sign out "
                    f"at /.pomerium/sign_out and sign back in to refresh."
                )

        decode_kwargs: dict[str, Any] = {
            "algorithms": self._algorithms,
        }
        if self._expected_audience:
            decode_kwargs["audience"] = self._expected_audience
        else:
            # Pomerium sets `aud` but we'll skip the check when no
            # expected audience is configured (single-route deploys).
            decode_kwargs["options"] = {"verify_aud": False}

        return jwt.decode(jwt_assertion, key=key, **decode_kwargs)


def _claims_to_user(claims: dict[str, Any]) -> CurrentUser:
    """Map Pomerium JWT claims -> CurrentUser.

    Pomerium standard claims (when fed by Azure AD via OIDC) include:
      - `sub` / `oid`: stable user object id (we prefer `oid`)
      - `email`: user email
      - `name`: display name
      - `picture`: avatar URL (optional, OIDC standard)
      - `groups`: list of group OIDs (Azure AD) or names (other IdPs)

    We coerce groups to a tuple so the CurrentUser remains hashable.
    """
    raw_groups = claims.get("groups") or []
    if isinstance(raw_groups, list):
        groups = tuple(str(g) for g in raw_groups)
    else:
        groups = ()
    # Pomerium puts the upstream IdP's OID in `oid` when available;
    # falls back to `sub` (the IdP-issued subject).
    oid = claims.get("oid") or claims.get("sub")
    return CurrentUser(
        oid=str(oid) if oid else None,
        email=claims.get("email"),
        name=claims.get("name"),
        picture_url=claims.get("picture"),
        groups=groups,
        is_anonymous=False,
    )


async def extract_current_user(
    request: Request,
    verifier: PomeriumVerifier | None,
    *,
    require_auth: bool,
) -> CurrentUser:
    """Read + verify Pomerium identity from the incoming request.

    Args:
      request: the FastAPI request.
      verifier: configured PomeriumVerifier, or None if tracking is
        disabled in config (or the integration layer hasn't wired it).
      require_auth: when True, missing/invalid identity raises 401;
        when False, falls back to :meth:`CurrentUser.anonymous`.

    The ``verifier is None`` short-circuit is deliberate: it lets the
    integration layer atomically toggle the whole feature off without
    touching any handler. The same effect can be achieved by setting
    ``tracking_user.enabled = false`` -- we honour both gates.
    """
    if verifier is None:
        return CurrentUser.anonymous()

    assertion = request.headers.get("X-Pomerium-Jwt-Assertion")
    if not assertion:
        if require_auth:
            raise HTTPException(
                status_code=401,
                detail="missing Pomerium identity",
            )
        return CurrentUser.anonymous()

    try:
        claims = verifier.verify(assertion)
    except jwt.InvalidTokenError as exc:
        _log.warning("rejected Pomerium JWT: %s", exc)
        if require_auth:
            raise HTTPException(
                status_code=401,
                detail=f"invalid Pomerium identity: {exc}",
            ) from exc
        return CurrentUser.anonymous()
    except Exception as exc:  # noqa: BLE001 -- JWKS fetch error etc.
        _log.warning("Pomerium verifier error: %s", exc)
        if require_auth:
            raise HTTPException(
                status_code=401,
                detail="identity verification failed",
            ) from exc
        return CurrentUser.anonymous()

    return _claims_to_user(claims)
