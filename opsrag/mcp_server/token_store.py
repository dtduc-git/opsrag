"""Postgres-backed MCP token store.

Schema is defined by `opsrag/db/migrations/0003_mcp_tokens.sql`:

    CREATE TABLE opsrag_mcp_token (
      id           UUID PRIMARY KEY,
      user_oid     UUID NOT NULL REFERENCES opsrag_user(oid) ON DELETE CASCADE,
      token_sha256 BYTEA NOT NULL UNIQUE,
      name         TEXT NOT NULL,
      created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      expires_at   TIMESTAMPTZ,
      revoked_at   TIMESTAMPTZ,
      last_used_at TIMESTAMPTZ
    );

Tokens are 32 bytes of `os.urandom`, base64-url-encoded, prefixed
`opsrag_`. Plaintext is returned to the caller ONLY at creation time;
the DB stores `sha256(plaintext)` as BYTEA. Lookup at validate-time is
a single indexed B-tree probe on `token_sha256` filtered by the
`opsrag_mcp_token_active` partial index (`WHERE revoked_at IS NULL`).

Fire-and-forget `last_used_at` bump: validation must not block on a
DB UPDATE on the hot path -- the SSE / JSON-RPC inbox handlers call
`validate()` on every message. The UPDATE is dispatched as a detached
task; failures are logged but invisible to the caller.
"""
from __future__ import annotations

import base64
import hashlib
import logging
import secrets
import uuid
from datetime import UTC, datetime
from typing import Any

from psycopg_pool import AsyncConnectionPool

_log = logging.getLogger("opsrag.mcp_server.token_store")

TOKEN_PREFIX = "opsrag_"
# 32 bytes of entropy. After urlsafe_b64encode this yields 43 chars (no
# padding). With the `opsrag_` prefix the full token is 50 chars --
# comfortable in an `Authorization: Bearer` header.
_TOKEN_ENTROPY_BYTES = 32


class MCPTokenStore:
    """Async Postgres-backed MCP token store.

    Same `AsyncConnectionPool` pattern as `opsrag.feedback_store` /
    `opsrag.usage_persistence` so all DB-touching modules share one
    Postgres connection topology.
    """

    def __init__(
        self,
        dsn: str,
        *,
        min_pool: int = 1,
        max_pool: int = 5,
    ) -> None:
        self._dsn = dsn
        self._pool = AsyncConnectionPool(
            conninfo=dsn,
            min_size=min_pool,
            max_size=max_pool,
            open=False,
            kwargs={"autocommit": True, "prepare_threshold": 0},
        )
        self._opened = False

    # --- lifecycle --------------------------------------------------

    async def open(self) -> None:
        await self._pool.open()
        self._opened = True
        _log.info("mcp_token_store opened")

    async def close(self) -> None:
        if self._opened:
            await self._pool.close()
            self._opened = False

    # --- token generation helpers ----------------------------------

    @staticmethod
    def _generate_plaintext() -> str:
        """Return a fresh URL-safe token of the form `opsrag_<43chars>`."""
        raw = secrets.token_bytes(_TOKEN_ENTROPY_BYTES)
        # `urlsafe_b64encode` returns bytes; strip the trailing `=` pad
        # so the token is plain alnum + `-` + `_`. 32 bytes -> 43 chars.
        body = base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")
        return f"{TOKEN_PREFIX}{body}"

    @staticmethod
    def _hash(plaintext: str) -> bytes:
        """Compute sha256(plaintext) -- returns the 32-byte digest."""
        return hashlib.sha256(plaintext.encode("utf-8")).digest()

    # --- CRUD ------------------------------------------------------

    async def create(
        self,
        user_oid: str,
        name: str,
        expires_at: datetime | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Create a new MCP token. Returns `(plaintext, row_metadata)`.

        The plaintext is shown to the caller exactly once -- store it
        client-side, never log it. Row metadata is safe to log / return
        in subsequent list calls; it has no `token` / `token_sha256`
        fields.
        """
        if not self._opened:
            raise RuntimeError("token store not opened")
        if not name or not name.strip():
            raise ValueError("token name must be non-empty")
        name = name.strip()[:120]  # cap to keep UI clean

        plaintext = self._generate_plaintext()
        digest = self._hash(plaintext)
        token_id = str(uuid.uuid4())

        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO opsrag_mcp_token "
                    "(id, user_oid, token_sha256, name, expires_at) "
                    "VALUES (%s, %s, %s, %s, %s) "
                    "RETURNING id, user_oid, name, created_at, expires_at",
                    (token_id, user_oid, digest, name, expires_at),
                )
                row = await cur.fetchone()
        meta = {
            "id": str(row[0]),
            "user_oid": str(row[1]),
            "name": row[2],
            "created_at": row[3].isoformat() if row[3] else None,
            "expires_at": row[4].isoformat() if row[4] else None,
            "revoked_at": None,
            "last_used_at": None,
        }
        return plaintext, meta

    async def validate(self, plaintext_token: str) -> dict[str, Any] | None:
        """Validate a bearer token. Returns the active row dict, or
        ``None`` if the token is missing, malformed, revoked, or expired.

        Bumps `last_used_at` in a fire-and-forget detached task so the
        SSE / message hot path never waits on a DB UPDATE round-trip.
        """
        if not self._opened:
            return None
        if not plaintext_token or not plaintext_token.startswith(TOKEN_PREFIX):
            return None
        digest = self._hash(plaintext_token)
        try:
            async with self._pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        "SELECT id, user_oid, name, created_at, "
                        "  expires_at, revoked_at, last_used_at "
                        "FROM opsrag_mcp_token "
                        "WHERE token_sha256 = %s",
                        (digest,),
                    )
                    row = await cur.fetchone()
        except Exception as exc:
            _log.warning("mcp token validate query failed: %s", exc)
            return None
        if row is None:
            return None
        token_id, user_oid, name, created_at, expires_at, revoked_at, last_used_at = row
        if revoked_at is not None:
            return None
        # Compare in UTC. `expires_at` is TIMESTAMPTZ; psycopg returns an
        # aware datetime. `datetime.now(timezone.utc)` is also aware.
        now = datetime.now(UTC)
        if expires_at is not None and expires_at <= now:
            return None
        # Fire-and-forget last_used_at bump. Wrapped in try/except inside
        # the inner coro so a transient DB blip doesn't surface as an
        # unhandled exception on the loop.
        import asyncio

        async def _bump() -> None:
            try:
                async with self._pool.connection() as conn2:
                    async with conn2.cursor() as cur2:
                        await cur2.execute(
                            "UPDATE opsrag_mcp_token SET last_used_at = NOW() "
                            "WHERE id = %s",
                            (token_id,),
                        )
            except Exception as exc:
                _log.debug("last_used_at bump failed for %s: %s", token_id, exc)

        # We're already in an async context (validate is awaited); this is
        # safe. If no running loop, fall back to a sync no-op.
        try:
            asyncio.get_running_loop().create_task(_bump())
        except RuntimeError:
            pass

        return {
            "id": str(token_id),
            "user_oid": str(user_oid),
            "name": name,
            "created_at": created_at.isoformat() if created_at else None,
            "expires_at": expires_at.isoformat() if expires_at else None,
            "revoked_at": None,
            "last_used_at": last_used_at.isoformat() if last_used_at else None,
        }

    async def list_for_user(self, user_oid: str) -> list[dict[str, Any]]:
        """List a user's ACTIVE tokens (excludes revoked). Excludes
        plaintext and hash -- only the metadata needed for the UI's
        token-management view. Ordered created_at DESC.

        Revoked tokens are filtered out: once you click revoke, the row
        disappears from the UI. If we ever add an "audit history" tab
        that needs revoked rows, add a separate `list_for_user_audit()`
        method rather than re-introducing the mixed-state list.
        """
        if not self._opened:
            return []
        try:
            async with self._pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        "SELECT id, name, created_at, expires_at, "
                        "  revoked_at, last_used_at "
                        "FROM opsrag_mcp_token "
                        "WHERE user_oid = %s AND revoked_at IS NULL "
                        "ORDER BY created_at DESC",
                        (user_oid,),
                    )
                    rows = await cur.fetchall()
        except Exception as exc:
            _log.warning("mcp token list query failed: %s", exc)
            return []
        out: list[dict[str, Any]] = []
        for r in rows:
            out.append({
                "id": str(r[0]),
                "name": r[1],
                "created_at": r[2].isoformat() if r[2] else None,
                "expires_at": r[3].isoformat() if r[3] else None,
                "revoked_at": r[4].isoformat() if r[4] else None,
                "last_used_at": r[5].isoformat() if r[5] else None,
            })
        return out

    async def revoke(self, token_id: str, user_oid: str) -> bool:
        """Revoke a token. The `user_oid` predicate prevents one user
        from revoking another user's tokens, even if the IDs leak.

        Returns ``True`` iff a row was updated (active token belonging
        to this user). Idempotent: calling twice returns ``False`` the
        second time.
        """
        if not self._opened:
            return False
        try:
            async with self._pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        "UPDATE opsrag_mcp_token "
                        "SET revoked_at = NOW() "
                        "WHERE id = %s AND user_oid = %s "
                        "  AND revoked_at IS NULL",
                        (token_id, user_oid),
                    )
                    return cur.rowcount > 0
        except Exception as exc:
            _log.warning("mcp token revoke failed: %s", exc)
            return False
