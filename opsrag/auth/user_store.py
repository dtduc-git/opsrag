"""Auth Tier-2 user store: local users, federated identities, refresh sessions.

This is the persistence layer for first-party login (password + SSO).
It is deliberately separate from the ``oid``-keyed analytics
``opsrag_user`` table (``opsrag.auth.store.UserStore``): that one records
*who showed up* for usage attribution; THIS one holds *credentials and
login identities* (password hashes, federated provider links, and
hashed refresh-session tokens). Conflating them would put a credential
column on an analytics table.

Three entities (see migration ``0006_auth_login.sql``):

* **auth user** -- ``id`` (uuid), ``email`` (unique, lowercased),
  ``email_verified``, ``password_hash`` (nullable: SSO-only accounts
  have no password), ``roles`` (text[]; operator-assigned roles layered
  on top of IdP-group-derived roles), timestamps.
* **federated identity** -- ``(provider, subject)`` unique; links an
  external IdP account (google/microsoft/github) to a local user, with
  the provider-asserted ``email`` + ``email_verified`` recorded at link
  time for the account-takeover guard.
* **refresh session** -- an opaque rotating refresh token stored ONLY as
  a SHA-256 hash (never plaintext at rest), with ``expires_at`` and a
  ``revoked_at`` tombstone. Rotation issues a new row and revokes the old.

Two implementations behind one ABC (:class:`AuthUserStore`):

* :class:`InMemoryAuthUserStore` -- for tests and zero-dependency local
  dev. No Postgres required.
* :class:`PostgresAuthUserStore` -- reuses the project's
  ``psycopg_pool.AsyncConnectionPool`` pattern (mirrors
  ``opsrag.auth.store.UserStore`` / ``opsrag.mcp_server.token_store``).

All methods are ``async`` so the two implementations are
interchangeable from the FastAPI handlers.
"""
from __future__ import annotations

import abc
import hashlib
import threading
import uuid
from dataclasses import dataclass, replace
from datetime import UTC, datetime


# ---------------------------------------------------------------------------
# Row dataclasses (the wire shape between the store and the auth handlers).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AuthUser:
    """A first-party login account."""

    id: str
    email: str
    email_verified: bool
    password_hash: str | None
    roles: tuple[str, ...] = ()
    name: str | None = None
    picture_url: str | None = None
    # Per-connector RBAC overrides (see opsrag.auth.connector_perms):
    # explicit per-user grants/denials layered on top of role-derived access.
    # `connectors_deny` wins over everything (role grants, default-allow, admin).
    connectors_allow: tuple[str, ...] = ()
    connectors_deny: tuple[str, ...] = ()
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(frozen=True)
class FederatedIdentity:
    """A link from an external IdP account to a local :class:`AuthUser`."""

    provider: str
    subject: str
    user_id: str
    email: str | None
    email_verified: bool


@dataclass(frozen=True)
class RefreshSession:
    """A server-tracked refresh token (stored hashed)."""

    id: str
    user_id: str
    token_hash: str
    expires_at: datetime
    created_at: datetime | None = None
    revoked_at: datetime | None = None

    @property
    def is_active(self) -> bool:
        now = datetime.now(UTC)
        exp = self.expires_at
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=UTC)
        return self.revoked_at is None and exp > now


def hash_token(raw_token: str) -> str:
    """SHA-256 hex of an opaque token. Refresh tokens are high-entropy
    random strings, so a fast hash (not a slow password KDF) is correct:
    there is nothing to brute-force. We store ONLY this digest at rest."""
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()


def _norm_email(email: str) -> str:
    return email.strip().lower()


# ---------------------------------------------------------------------------
# Abstract store.
# ---------------------------------------------------------------------------
class AuthUserStore(abc.ABC):
    """Async persistence contract for login users/identities/sessions."""

    # --- users ---------------------------------------------------------
    @abc.abstractmethod
    async def create_user(
        self,
        *,
        email: str,
        password_hash: str | None,
        email_verified: bool = False,
        roles: tuple[str, ...] = (),
        name: str | None = None,
        picture_url: str | None = None,
    ) -> AuthUser: ...

    @abc.abstractmethod
    async def get_user_by_email(self, email: str) -> AuthUser | None: ...

    @abc.abstractmethod
    async def get_user_by_id(self, user_id: str) -> AuthUser | None: ...

    @abc.abstractmethod
    async def set_password_hash(self, user_id: str, password_hash: str) -> None: ...

    @abc.abstractmethod
    async def mark_email_verified(self, user_id: str) -> None: ...

    @abc.abstractmethod
    async def set_connector_overrides(
        self,
        user_id: str,
        *,
        allow: tuple[str, ...],
        deny: tuple[str, ...],
    ) -> None:
        """Replace a user's per-connector allow/deny override lists."""
        ...

    @abc.abstractmethod
    async def set_roles(self, user_id: str, roles: tuple[str, ...]) -> None:
        """Replace a user's operator-assigned roles (RBAC admin action)."""
        ...

    @abc.abstractmethod
    async def list_users(self, *, limit: int = 200) -> list[AuthUser]:
        """List users (newest first) for the admin Users & Roles view."""
        ...

    # --- federated identities -----------------------------------------
    @abc.abstractmethod
    async def get_identity(
        self, provider: str, subject: str
    ) -> FederatedIdentity | None: ...

    @abc.abstractmethod
    async def link_identity(
        self,
        *,
        provider: str,
        subject: str,
        user_id: str,
        email: str | None,
        email_verified: bool,
    ) -> FederatedIdentity: ...

    # --- refresh sessions ---------------------------------------------
    @abc.abstractmethod
    async def create_refresh_session(
        self, *, user_id: str, token_hash: str, expires_at: datetime
    ) -> RefreshSession: ...

    @abc.abstractmethod
    async def get_refresh_session(self, token_hash: str) -> RefreshSession | None: ...

    @abc.abstractmethod
    async def revoke_refresh_session(self, token_hash: str) -> None: ...

    @abc.abstractmethod
    async def revoke_all_for_user(self, user_id: str) -> None: ...


# ---------------------------------------------------------------------------
# In-memory implementation (tests + zero-dep local dev).
# ---------------------------------------------------------------------------
class InMemoryAuthUserStore(AuthUserStore):
    """Thread-safe in-process store. No external dependencies."""

    def __init__(self) -> None:
        self._users: dict[str, AuthUser] = {}
        self._email_index: dict[str, str] = {}  # email -> user_id
        self._identities: dict[tuple[str, str], FederatedIdentity] = {}
        self._sessions: dict[str, RefreshSession] = {}  # token_hash -> session
        self._lock = threading.Lock()

    async def create_user(
        self,
        *,
        email: str,
        password_hash: str | None,
        email_verified: bool = False,
        roles: tuple[str, ...] = (),
        name: str | None = None,
        picture_url: str | None = None,
    ) -> AuthUser:
        e = _norm_email(email)
        now = datetime.now(UTC)
        with self._lock:
            if e in self._email_index:
                raise ValueError(f"user already exists: {e}")
            user = AuthUser(
                id=str(uuid.uuid4()),
                email=e,
                email_verified=email_verified,
                password_hash=password_hash,
                roles=tuple(roles),
                name=name,
                picture_url=picture_url,
                created_at=now,
                updated_at=now,
            )
            self._users[user.id] = user
            self._email_index[e] = user.id
        return user

    async def get_user_by_email(self, email: str) -> AuthUser | None:
        with self._lock:
            uid = self._email_index.get(_norm_email(email))
            return self._users.get(uid) if uid else None

    async def get_user_by_id(self, user_id: str) -> AuthUser | None:
        with self._lock:
            return self._users.get(user_id)

    async def set_password_hash(self, user_id: str, password_hash: str) -> None:
        with self._lock:
            u = self._users.get(user_id)
            if u is not None:
                self._users[user_id] = replace(
                    u, password_hash=password_hash, updated_at=datetime.now(UTC)
                )

    async def mark_email_verified(self, user_id: str) -> None:
        with self._lock:
            u = self._users.get(user_id)
            if u is not None:
                self._users[user_id] = replace(
                    u, email_verified=True, updated_at=datetime.now(UTC)
                )

    async def set_roles(self, user_id: str, roles: tuple[str, ...]) -> None:
        with self._lock:
            u = self._users.get(user_id)
            if u is not None:
                self._users[user_id] = replace(
                    u, roles=tuple(roles), updated_at=datetime.now(UTC)
                )

    async def set_connector_overrides(
        self,
        user_id: str,
        *,
        allow: tuple[str, ...],
        deny: tuple[str, ...],
    ) -> None:
        with self._lock:
            u = self._users.get(user_id)
            if u is not None:
                self._users[user_id] = replace(
                    u,
                    connectors_allow=tuple(allow),
                    connectors_deny=tuple(deny),
                    updated_at=datetime.now(UTC),
                )

    async def list_users(self, *, limit: int = 200) -> list[AuthUser]:
        with self._lock:
            users = list(self._users.values())
        users.sort(
            key=lambda u: u.created_at or datetime.min.replace(tzinfo=UTC),
            reverse=True,
        )
        return users[:limit]

    async def get_identity(
        self, provider: str, subject: str
    ) -> FederatedIdentity | None:
        with self._lock:
            return self._identities.get((provider, subject))

    async def link_identity(
        self,
        *,
        provider: str,
        subject: str,
        user_id: str,
        email: str | None,
        email_verified: bool,
    ) -> FederatedIdentity:
        ident = FederatedIdentity(
            provider=provider,
            subject=subject,
            user_id=user_id,
            email=_norm_email(email) if email else None,
            email_verified=email_verified,
        )
        with self._lock:
            self._identities[(provider, subject)] = ident
        return ident

    async def create_refresh_session(
        self, *, user_id: str, token_hash: str, expires_at: datetime
    ) -> RefreshSession:
        sess = RefreshSession(
            id=str(uuid.uuid4()),
            user_id=user_id,
            token_hash=token_hash,
            expires_at=expires_at,
            created_at=datetime.now(UTC),
            revoked_at=None,
        )
        with self._lock:
            self._sessions[token_hash] = sess
        return sess

    async def get_refresh_session(self, token_hash: str) -> RefreshSession | None:
        with self._lock:
            return self._sessions.get(token_hash)

    async def revoke_refresh_session(self, token_hash: str) -> None:
        with self._lock:
            s = self._sessions.get(token_hash)
            if s is not None and s.revoked_at is None:
                self._sessions[token_hash] = replace(
                    s, revoked_at=datetime.now(UTC)
                )

    async def revoke_all_for_user(self, user_id: str) -> None:
        with self._lock:
            now = datetime.now(UTC)
            for th, s in list(self._sessions.items()):
                if s.user_id == user_id and s.revoked_at is None:
                    self._sessions[th] = replace(s, revoked_at=now)


# ---------------------------------------------------------------------------
# Postgres implementation.
# ---------------------------------------------------------------------------
class PostgresAuthUserStore(AuthUserStore):
    """Async Postgres-backed store. Schema lives in migration
    ``0006_auth_login.sql`` (the migration is the source of truth)."""

    def __init__(self, dsn: str, *, min_pool: int = 1, max_pool: int = 4) -> None:
        # Imported lazily so the in-memory store (tests) needs no psycopg.
        from psycopg_pool import AsyncConnectionPool

        self._dsn = dsn
        self._pool = AsyncConnectionPool(
            conninfo=dsn,
            min_size=min_pool,
            max_size=max_pool,
            open=False,
            kwargs={"autocommit": True, "prepare_threshold": 0},
        )
        self._opened = False

    async def open(self) -> None:
        await self._pool.open()
        self._opened = True

    async def close(self) -> None:
        if self._opened:
            await self._pool.close()
            self._opened = False

    async def init_schema(self) -> None:
        """No-op: the migration framework owns the DDL (parity with the
        other ``*Store`` classes so the boot sequence calls it uniformly)."""
        return None

    @staticmethod
    def _user_from_row(row: tuple) -> AuthUser:
        return AuthUser(
            id=str(row[0]),
            email=row[1],
            email_verified=bool(row[2]),
            password_hash=row[3],
            roles=tuple(row[4] or ()),
            name=row[5],
            picture_url=row[6],
            connectors_allow=tuple(row[7] or ()),
            connectors_deny=tuple(row[8] or ()),
            created_at=row[9],
            updated_at=row[10],
        )

    _USER_COLS = (
        "id, email, email_verified, password_hash, roles, "
        "name, picture_url, connectors_allow, connectors_deny, "
        "created_at, updated_at"
    )

    async def create_user(
        self,
        *,
        email: str,
        password_hash: str | None,
        email_verified: bool = False,
        roles: tuple[str, ...] = (),
        name: str | None = None,
        picture_url: str | None = None,
    ) -> AuthUser:
        uid = str(uuid.uuid4())
        e = _norm_email(email)
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"""
                    INSERT INTO opsrag_auth_user
                      (id, email, email_verified, password_hash, roles,
                       name, picture_url, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
                    RETURNING {self._USER_COLS}
                    """,
                    (uid, e, email_verified, password_hash, list(roles), name, picture_url),
                )
                row = await cur.fetchone()
        return self._user_from_row(row)

    async def get_user_by_email(self, email: str) -> AuthUser | None:
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"SELECT {self._USER_COLS} FROM opsrag_auth_user WHERE email = %s",
                    (_norm_email(email),),
                )
                row = await cur.fetchone()
        return self._user_from_row(row) if row else None

    async def get_user_by_id(self, user_id: str) -> AuthUser | None:
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"SELECT {self._USER_COLS} FROM opsrag_auth_user WHERE id = %s",
                    (user_id,),
                )
                row = await cur.fetchone()
        return self._user_from_row(row) if row else None

    async def set_password_hash(self, user_id: str, password_hash: str) -> None:
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "UPDATE opsrag_auth_user SET password_hash = %s, updated_at = NOW() "
                    "WHERE id = %s",
                    (password_hash, user_id),
                )

    async def mark_email_verified(self, user_id: str) -> None:
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "UPDATE opsrag_auth_user SET email_verified = TRUE, updated_at = NOW() "
                    "WHERE id = %s",
                    (user_id,),
                )

    async def set_roles(self, user_id: str, roles: tuple[str, ...]) -> None:
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "UPDATE opsrag_auth_user SET roles = %s, updated_at = NOW() "
                    "WHERE id = %s",
                    (list(roles), user_id),
                )

    async def set_connector_overrides(
        self,
        user_id: str,
        *,
        allow: tuple[str, ...],
        deny: tuple[str, ...],
    ) -> None:
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "UPDATE opsrag_auth_user SET connectors_allow = %s, "
                    "connectors_deny = %s, updated_at = NOW() WHERE id = %s",
                    (list(allow), list(deny), user_id),
                )

    async def list_users(self, *, limit: int = 200) -> list[AuthUser]:
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"SELECT {self._USER_COLS} FROM opsrag_auth_user "
                    "ORDER BY created_at DESC LIMIT %s",
                    (limit,),
                )
                rows = await cur.fetchall()
        return [self._user_from_row(r) for r in rows]

    async def get_identity(
        self, provider: str, subject: str
    ) -> FederatedIdentity | None:
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT provider, subject, user_id, email, email_verified "
                    "FROM opsrag_auth_identity WHERE provider = %s AND subject = %s",
                    (provider, subject),
                )
                row = await cur.fetchone()
        if not row:
            return None
        return FederatedIdentity(
            provider=row[0],
            subject=row[1],
            user_id=str(row[2]),
            email=row[3],
            email_verified=bool(row[4]),
        )

    async def link_identity(
        self,
        *,
        provider: str,
        subject: str,
        user_id: str,
        email: str | None,
        email_verified: bool,
    ) -> FederatedIdentity:
        e = _norm_email(email) if email else None
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO opsrag_auth_identity
                      (provider, subject, user_id, email, email_verified, created_at)
                    VALUES (%s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (provider, subject) DO UPDATE SET
                      email = EXCLUDED.email,
                      email_verified = EXCLUDED.email_verified
                    """,
                    (provider, subject, user_id, e, email_verified),
                )
        return FederatedIdentity(
            provider=provider,
            subject=subject,
            user_id=user_id,
            email=e,
            email_verified=email_verified,
        )

    async def create_refresh_session(
        self, *, user_id: str, token_hash: str, expires_at: datetime
    ) -> RefreshSession:
        sid = str(uuid.uuid4())
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO opsrag_auth_refresh_session
                      (id, user_id, token_hash, expires_at, created_at)
                    VALUES (%s, %s, %s, %s, NOW())
                    """,
                    (sid, user_id, token_hash, expires_at),
                )
        return RefreshSession(
            id=sid,
            user_id=user_id,
            token_hash=token_hash,
            expires_at=expires_at,
            created_at=datetime.now(UTC),
            revoked_at=None,
        )

    async def get_refresh_session(self, token_hash: str) -> RefreshSession | None:
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT id, user_id, token_hash, expires_at, created_at, revoked_at "
                    "FROM opsrag_auth_refresh_session WHERE token_hash = %s",
                    (token_hash,),
                )
                row = await cur.fetchone()
        if not row:
            return None
        return RefreshSession(
            id=str(row[0]),
            user_id=str(row[1]),
            token_hash=row[2],
            expires_at=row[3],
            created_at=row[4],
            revoked_at=row[5],
        )

    async def revoke_refresh_session(self, token_hash: str) -> None:
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "UPDATE opsrag_auth_refresh_session SET revoked_at = NOW() "
                    "WHERE token_hash = %s AND revoked_at IS NULL",
                    (token_hash,),
                )

    async def revoke_all_for_user(self, user_id: str) -> None:
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "UPDATE opsrag_auth_refresh_session SET revoked_at = NOW() "
                    "WHERE user_id = %s AND revoked_at IS NULL",
                    (user_id,),
                )
