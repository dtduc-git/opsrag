"""SSO via Authlib: Google, Microsoft/Entra (OIDC), GitHub (OAuth2).

Terminates the OAuth2 Authorization Code flow **server-side** (the
backend is the redirect terminator; no browser PKCE), then resolves the
external identity to a local :class:`opsrag.auth.user_store.AuthUser` and
mints the same cookie session as password login.

Providers
---------
* **google** / **microsoft** -- OIDC: discovered via
  ``server_metadata_url`` (``.well-known/openid-configuration``).
  ``authorize_access_token`` returns an ``id_token`` Authlib parses +
  validates (incl. the ``nonce``); we read ``sub``/``email``/
  ``email_verified``/``name``/``picture`` from the claims.
* **github** -- plain OAuth2 (no ``id_token``). We exchange the code for
  an access token, then GET ``/user`` + ``/user/emails`` to find the
  PRIMARY VERIFIED email. ``email_verified`` is taken from GitHub's
  ``verified`` flag on that email.

Security (per the brief / DESIGN 1)
-----------------------------------
* **state** -- CSRF protection on the redirect. We mint a signed,
  single-use ``state`` and verify it on callback.
* **nonce** -- replay protection for the OIDC ``id_token``. We mint a
  ``nonce``, bind it to the ``state``, and Authlib verifies the returned
  ``id_token``'s ``nonce`` claim matches.
* **email_verified-required identity linking** (account-takeover guard).
  We REFUSE to link a federated identity to a local account by email
  unless the IdP asserts ``email_verified``. An unverified-email IdP
  response can create a *new* federated-only account but can NEVER be
  auto-linked to (or take over) an existing local/password account.

Role/scope mapping: the IdP ``groups`` claim (when present) is mapped to
opsrag roles via :func:`opsrag.auth.scopes.resolve_roles` +
``role_mappings``; scopes are derived from those roles. This shares the
SINGLE authoritative scope model in ``opsrag.auth.scopes`` (no duplicate
role table).

The transport (Authlib OAuth client) is injected, so tests pass a mock
client and never touch a live IdP.
"""
from __future__ import annotations

import hmac
import logging
import secrets
from dataclasses import dataclass
from typing import Any

_log = logging.getLogger("opsrag.auth.sso")

# Canonical provider names. "microsoft" covers Entra ID / Azure AD.
SUPPORTED_PROVIDERS = ("google", "microsoft", "github")

# Default OIDC discovery URLs. Entra uses the common endpoint by default;
# a single-tenant deployment overrides ``server_metadata_url`` with its
# tenant-specific issuer.
_OIDC_METADATA = {
    "google": "https://accounts.google.com/.well-known/openid-configuration",
    "microsoft": (
        "https://login.microsoftonline.com/common/v2.0/.well-known/"
        "openid-configuration"
    ),
}
_GITHUB_AUTHORIZE = "https://github.com/login/oauth/authorize"
_GITHUB_TOKEN = "https://github.com/login/oauth/access_token"
_GITHUB_API = "https://api.github.com/"


class SSOError(Exception):
    """Any SSO-flow failure (bad state, missing verified email, etc.)."""


@dataclass(frozen=True)
class ProviderConfig:
    """Per-provider OAuth app credentials + scopes."""

    enabled: bool = False
    client_id: str | None = None
    client_secret: str | None = None
    scopes: tuple[str, ...] = ()
    # Single-tenant Entra override: full tenant issuer metadata URL.
    server_metadata_url: str | None = None


@dataclass(frozen=True)
class ExternalIdentity:
    """Normalized identity extracted from any provider's callback."""

    provider: str
    subject: str
    email: str | None
    email_verified: bool
    name: str | None = None
    picture_url: str | None = None
    groups: tuple[str, ...] = ()


def _default_scopes(provider: str) -> tuple[str, ...]:
    if provider == "github":
        return ("read:user", "user:email")
    # OIDC providers: openid is required; email/profile for the claims.
    return ("openid", "email", "profile")


def build_oauth_registry(
    providers: dict[str, ProviderConfig],
    *,
    oauth: Any | None = None,
) -> Any:
    """Build (or populate) an Authlib ``OAuth`` registry from config.

    Only ENABLED providers with both ``client_id`` and ``client_secret``
    are registered. ``oauth`` may be injected (tests); otherwise a fresh
    ``authlib.integrations.starlette_client.OAuth`` is created.
    """
    if oauth is None:
        from authlib.integrations.starlette_client import OAuth

        oauth = OAuth()
    for name, cfg in providers.items():
        if name not in SUPPORTED_PROVIDERS:
            raise ValueError(f"unsupported SSO provider: {name!r}")
        if not cfg.enabled:
            continue
        if not cfg.client_id or not cfg.client_secret:
            _log.warning("SSO provider %s enabled but missing client credentials", name)
            continue
        scopes = cfg.scopes or _default_scopes(name)
        client_kwargs = {"scope": " ".join(scopes)}
        if name == "github":
            oauth.register(
                name=name,
                client_id=cfg.client_id,
                client_secret=cfg.client_secret,
                access_token_url=_GITHUB_TOKEN,
                authorize_url=_GITHUB_AUTHORIZE,
                api_base_url=_GITHUB_API,
                client_kwargs=client_kwargs,
            )
        else:
            oauth.register(
                name=name,
                client_id=cfg.client_id,
                client_secret=cfg.client_secret,
                server_metadata_url=cfg.server_metadata_url or _OIDC_METADATA[name],
                client_kwargs=client_kwargs,
            )
    return oauth


def new_state() -> str:
    """A high-entropy single-use OAuth ``state`` value."""
    return secrets.token_urlsafe(32)


def new_nonce() -> str:
    """A high-entropy OIDC ``nonce`` value."""
    return secrets.token_urlsafe(32)


def verify_state(expected: str | None, received: str | None) -> bool:
    """Constant-time ``state`` comparison (CSRF guard on the callback)."""
    if not expected or not received:
        return False
    return hmac.compare_digest(expected, received)


def _normalize_oidc_claims(provider: str, claims: dict[str, Any]) -> ExternalIdentity:
    """Map OIDC userinfo/id_token claims to :class:`ExternalIdentity`.

    ``email_verified`` may arrive as a bool or the string ``"true"``
    (some IdPs). We treat ONLY a real truthy bool / ``"true"`` as
    verified -- absent/false/unknown is unverified (fail-closed)."""
    ev_raw = claims.get("email_verified")
    if isinstance(ev_raw, str):
        email_verified = ev_raw.strip().lower() == "true"
    else:
        email_verified = bool(ev_raw)
    groups_claim = claims.get("groups") or claims.get("roles") or ()
    if isinstance(groups_claim, str):
        groups = (groups_claim,)
    else:
        groups = tuple(str(g) for g in groups_claim)
    return ExternalIdentity(
        provider=provider,
        subject=str(claims["sub"]),
        email=(claims.get("email") or None),
        email_verified=email_verified,
        name=claims.get("name"),
        picture_url=claims.get("picture"),
        groups=groups,
    )


async def fetch_identity(
    provider: str,
    *,
    oauth_client: Any,
    request: Any,
    expected_nonce: str | None = None,
) -> ExternalIdentity:
    """Exchange the callback code for a token and return the normalized
    external identity.

    ``oauth_client`` is the registered Authlib app for ``provider``
    (``oauth.create_client(provider)`` or ``oauth.<provider>``). For OIDC
    providers we call ``authorize_access_token`` (which validates the
    ``id_token`` incl. ``nonce``) and read the parsed ``userinfo``. For
    GitHub we read the access token then call the user + emails APIs.

    Raises :class:`SSOError` when no usable subject/email is found.
    """
    if provider == "github":
        return await _fetch_github_identity(oauth_client, request)

    token = await oauth_client.authorize_access_token(request)
    # Authlib stores parsed id_token claims under "userinfo" after
    # validating signature, iss, aud, exp, and nonce.
    claims = token.get("userinfo")
    if not claims:
        # Fallback: explicitly parse the id_token (older Authlib paths).
        claims = await oauth_client.parse_id_token(
            request, token, nonce=expected_nonce
        )
    if not claims or not claims.get("sub"):
        raise SSOError(f"{provider}: id_token missing 'sub'")
    return _normalize_oidc_claims(provider, dict(claims))


async def _fetch_github_identity(oauth_client: Any, request: Any) -> ExternalIdentity:
    """GitHub (no id_token): token exchange + /user + /user/emails."""
    token = await oauth_client.authorize_access_token(request)
    resp = await oauth_client.get("user", token=token)
    user = resp.json()
    sub = user.get("id")
    if sub is None:
        raise SSOError("github: /user response missing 'id'")

    # Find the primary verified email. GitHub's /user.email can be null
    # (private email) so /user/emails is authoritative.
    email: str | None = None
    email_verified = False
    try:
        emails_resp = await oauth_client.get("user/emails", token=token)
        emails = emails_resp.json()
        primary = next(
            (e for e in emails if e.get("primary") and e.get("verified")),
            None,
        )
        if primary is None:
            primary = next((e for e in emails if e.get("verified")), None)
        if primary is not None:
            email = primary.get("email")
            email_verified = bool(primary.get("verified"))
    except Exception as exc:  # noqa: BLE001
        _log.warning("github: /user/emails fetch failed: %s", exc)

    if email is None and user.get("email"):
        # Public profile email; GitHub does not assert verification here.
        email = user.get("email")
        email_verified = False

    return ExternalIdentity(
        provider="github",
        subject=str(sub),
        email=(email.lower() if email else None),
        email_verified=email_verified,
        name=user.get("name") or user.get("login"),
        picture_url=user.get("avatar_url"),
        groups=(),
    )


async def resolve_or_link_user(
    identity: ExternalIdentity,
    *,
    store: Any,
) -> Any:
    """Resolve ``identity`` to a local :class:`AuthUser`, creating or
    linking as the account-takeover guard allows.

    Resolution order:
      1. **Existing federated link** ``(provider, subject)`` -> return its
         user. This is the steady-state path and needs no email check.
      2. **No link yet**: we may auto-link by verified email ONLY when
         ``identity.email_verified`` is True. We look up a local user by
         that email; if found, link the identity to it. If the IdP did
         NOT assert ``email_verified``, we MUST NOT link to an existing
         account (that is the takeover vector) -- instead we create a
         fresh federated-only account (password_hash NULL) keyed by the
         email, marked unverified.
      3. **No existing user by email**: create a new account. Its
         ``email_verified`` mirrors the IdP assertion.

    Returns the resolved :class:`AuthUser`.
    """
    existing_link = await store.get_identity(identity.provider, identity.subject)
    if existing_link is not None:
        user = await store.get_user_by_id(existing_link.user_id)
        if user is not None:
            return user
        # Dangling link (user deleted) -- fall through to recreate.

    email = identity.email
    if identity.email_verified and email:
        # Safe to consider linking to an existing account by verified email.
        existing_user = await store.get_user_by_email(email)
        if existing_user is not None:
            await store.link_identity(
                provider=identity.provider,
                subject=identity.subject,
                user_id=existing_user.id,
                email=email,
                email_verified=True,
            )
            return existing_user

    # No safe link target: create a new account.
    #
    # Account-takeover guard: when the IdP did NOT assert email_verified,
    # we refuse to attach to ANY existing account. If a (possibly
    # password) account already owns this email, creating a duplicate is
    # blocked by the unique email constraint, so we surface a clear error
    # rather than silently linking.
    if email and not identity.email_verified:
        clash = await store.get_user_by_email(email)
        if clash is not None:
            raise SSOError(
                f"{identity.provider} did not assert email_verified for "
                f"{email!r}; refusing to link to the existing account "
                "(account-takeover guard)"
            )

    # Persist the default interactive role explicitly so it's visible/editable
    # in the admin Users & Roles view (rather than relying on the login-time
    # resolve_roles fallback). DEFAULT_ROLE = chat + investigate.
    from opsrag.auth.scopes import DEFAULT_ROLE

    new_user = await store.create_user(
        email=email or f"{identity.provider}:{identity.subject}@sso.local",
        password_hash=None,
        email_verified=bool(identity.email_verified),
        roles=(DEFAULT_ROLE,),
        name=identity.name,
        picture_url=identity.picture_url,
    )
    await store.link_identity(
        provider=identity.provider,
        subject=identity.subject,
        user_id=new_user.id,
        email=email,
        email_verified=bool(identity.email_verified),
    )
    return new_user
