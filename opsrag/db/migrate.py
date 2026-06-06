"""Django-style SQL migration runner for OpsRAG.

Design notes
------------
- Migrations are plain ``.sql`` files in :mod:`opsrag.db.migrations`,
  named ``0001_<slug>.sql``. They are applied in lex order so always
  zero-pad the prefix to at least 4 digits.
- Each migration runs inside a single transaction. On error we
  rollback and re-raise so a partial schema never lands.
- We record sha256 of every applied file in ``opsrag_schema_migrations``.
  If a recorded file is edited later we refuse to continue and ask
  the operator to either revert or write a follow-up migration. This
  is the same contract Django, Flyway and Liquibase use.
- The runner has no application dependencies -- it speaks straight to
  psycopg/psycopg_pool so it can run from a one-shot job, an entrypoint
  hook, or a developer laptop without booting the full app.

CLI
---
    python -m opsrag.db.migrate up        # apply all pending
    python -m opsrag.db.migrate status    # show applied vs pending
    python -m opsrag.db.migrate backup    # shells out to scripts/db-backup.sh
    python -m opsrag.db.migrate restore <path>   # shells out to scripts/db-restore.sh

DSN comes from ``OPSRAG_POSTGRES_DSN`` (preferred) or ``POSTGRES_DSN``.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import logging
import os
import subprocess
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from psycopg_pool import AsyncConnectionPool

_log = logging.getLogger("opsrag.db.migrate")

MIGRATIONS_TABLE = "opsrag_schema_migrations"

# Cluster-wide Postgres advisory lock key for the migration run.
# pg_advisory_lock takes a 64-bit integer; we derive a stable value
# from the ASCII bytes of "OpsRAGDB" so two replicas of opsrag-backend
# starting up concurrently serialize their apply_all() calls.
# Advisory locks auto-release if the session crashes -- no zombie locks.
MIGRATION_LOCK_KEY = 0x4F_70_73_52_41_47_44_42  # "OpsRAGDB"

_CREATE_MIGRATIONS_TABLE = f"""
CREATE TABLE IF NOT EXISTS {MIGRATIONS_TABLE} (
    id          TEXT PRIMARY KEY,
    applied_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    checksum    TEXT NOT NULL
)
"""


@dataclass(frozen=True)
class Migration:
    """One on-disk migration file.

    A migration ``0002_user_attribution`` is the pair
    ``0002_user_attribution.sql`` (forward) plus the OPTIONAL
    ``0002_user_attribution.down.sql`` (reverse). Migrations without
    a paired down file are forward-only -- :func:`roll_back` refuses
    them with a clear error.
    """

    id: str               # e.g. "0002_user_attribution" (no .sql)
    path: Path
    sql: str
    checksum: str         # sha256 hex of the forward file bytes
    down_path: Path | None = None
    down_sql: str | None = None
    down_checksum: str | None = None

    @property
    def reversible(self) -> bool:
        return self.down_sql is not None


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def discover_migrations(dir_path: Path) -> list[Migration]:
    """Return all ``*.sql`` migrations sorted lexicographically.

    Lex sort works as long as the numeric prefix is zero-padded
    (``0001``, ``0010``, ``0100`` -- not ``1``, ``10``, ``100``). We
    enforce 4-digit padding by convention; the tests pin this.
    """
    if not dir_path.exists():
        return []
    out: list[Migration] = []
    # Filter out `.down.sql` from the forward sweep -- they're paired
    # to a forward file by stem, not standalone migrations.
    forward_paths = [
        p for p in sorted(dir_path.glob("*.sql"))
        if not p.name.endswith(".down.sql")
    ]
    for sql_path in forward_paths:
        raw = sql_path.read_bytes()
        down_path = sql_path.with_name(f"{sql_path.stem}.down.sql")
        down_sql: str | None = None
        down_checksum: str | None = None
        if down_path.exists():
            down_raw = down_path.read_bytes()
            down_sql = down_raw.decode("utf-8")
            down_checksum = _sha256_hex(down_raw)
        out.append(
            Migration(
                id=sql_path.stem,
                path=sql_path,
                sql=raw.decode("utf-8"),
                checksum=_sha256_hex(raw),
                down_path=down_path if down_path.exists() else None,
                down_sql=down_sql,
                down_checksum=down_checksum,
            )
        )
    return out


def _default_migrations_dir() -> Path:
    return Path(__file__).resolve().parent / "migrations"


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

async def ensure_migrations_table(pool: AsyncConnectionPool) -> None:
    """Idempotent: creates ``opsrag_schema_migrations`` if missing."""
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(_CREATE_MIGRATIONS_TABLE)


async def applied_migrations(pool: AsyncConnectionPool) -> set[str]:
    """Return the set of migration ids already recorded as applied."""
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(f"SELECT id FROM {MIGRATIONS_TABLE}")
            rows = await cur.fetchall()
    return {row[0] for row in rows}


async def _applied_with_checksums(pool: AsyncConnectionPool) -> dict[str, str]:
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(f"SELECT id, checksum FROM {MIGRATIONS_TABLE}")
            rows = await cur.fetchall()
    return {row[0]: row[1] for row in rows}


def _verify_checksums(
    migrations: Iterable[Migration],
    recorded: dict[str, str],
) -> None:
    """Raise if any already-applied migration's file has been edited."""
    for m in migrations:
        prev = recorded.get(m.id)
        if prev is not None and prev != m.checksum:
            raise RuntimeError(
                f"migration {m.id} was edited after being applied "
                f"-- refusing to continue. Either revert the file or "
                f"write a new migration. "
                f"(recorded checksum={prev[:12]}..., on-disk={m.checksum[:12]}...)"
            )


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------

async def apply_all(
    pool: AsyncConnectionPool,
    *,
    migrations_dir: Path | None = None,
    dry_run: bool = False,
) -> list[str]:
    """Apply every pending migration in a single transaction each.

    Returns the list of migration ids that were applied (empty if
    nothing to do). When ``dry_run`` is True we print the plan but
    don't execute or record anything.
    """
    dir_path = migrations_dir or _default_migrations_dir()
    migrations = discover_migrations(dir_path)
    if not migrations:
        print(f"no migrations found in {dir_path}")
        return []

    await ensure_migrations_table(pool)

    # Serialize concurrent replicas with a session-scoped advisory lock.
    # The lock is held for the lifetime of the connection -- if the pod
    # crashes, Postgres releases it automatically. We do every operation
    # under this single connection so the lock outlives any per-migration
    # transaction. Other replicas block on pg_advisory_lock until we
    # release; they then re-read applied_migrations() inside the lock
    # and see our work as already done.
    applied: list[str] = []
    async with pool.connection() as lock_conn:
        async with lock_conn.cursor() as lock_cur:
            await lock_cur.execute(
                "SELECT pg_advisory_lock(%s)", (MIGRATION_LOCK_KEY,),
            )
        try:
            # Re-read applied set INSIDE the lock -- covers the case
            # where we waited for another replica and that replica
            # applied everything before releasing.
            async with lock_conn.cursor() as cur:
                await cur.execute(
                    f"SELECT id, checksum FROM {MIGRATIONS_TABLE}",
                )
                rows = await cur.fetchall()
                recorded = {row[0]: row[1] for row in rows}
            _verify_checksums(migrations, recorded)

            pending = [m for m in migrations if m.id not in recorded]
            if not pending:
                print("no pending migrations")
                return []

            for m in pending:
                if dry_run:
                    print(f"would apply {m.id} ({len(m.sql)} bytes)")
                    applied.append(m.id)
                    continue

                print(f"applying {m.id}...", end="", flush=True)
                t0 = time.time()
                try:
                    async with lock_conn.transaction():
                        async with lock_conn.cursor() as cur:
                            await cur.execute(m.sql)
                            await cur.execute(
                                f"INSERT INTO {MIGRATIONS_TABLE} "
                                f"(id, checksum) VALUES (%s, %s)",
                                (m.id, m.checksum),
                            )
                except Exception as exc:
                    elapsed_ms = int((time.time() - t0) * 1000)
                    print(f" FAILED after {elapsed_ms}ms: {exc}")
                    raise
                elapsed_ms = int((time.time() - t0) * 1000)
                print(f" ok ({elapsed_ms}ms)")
                applied.append(m.id)
        finally:
            async with lock_conn.cursor() as lock_cur:
                await lock_cur.execute(
                    "SELECT pg_advisory_unlock(%s)", (MIGRATION_LOCK_KEY,),
                )

    return applied


# ---------------------------------------------------------------------------
# Rollback (down migrations)
# ---------------------------------------------------------------------------

async def roll_back(
    pool: AsyncConnectionPool,
    *,
    count: int | None = None,
    to_id: str | None = None,
    migrations_dir: Path | None = None,
    dry_run: bool = False,
) -> list[str]:
    """Roll back applied migrations using their paired ``.down.sql`` files.

    Exactly one of ``count`` or ``to_id`` must be provided:

      - ``count=N``  -- roll back the LAST N applied migrations.
      - ``to_id=X``  -- roll back every migration applied AFTER X
        (inclusive ordering -- X stays applied).

    Returns the list of migration ids that were rolled back, newest-first
    (i.e. the order in which down.sql files were executed).

    Refuses to proceed if any candidate migration is missing its
    paired ``.down.sql`` file -- the operator must decide whether to
    write one, or restore from backup instead.
    """
    if (count is None) == (to_id is None):
        raise ValueError("provide exactly one of `count` or `to_id`")

    dir_path = migrations_dir or _default_migrations_dir()
    migrations = discover_migrations(dir_path)
    by_id = {m.id: m for m in migrations}

    await ensure_migrations_table(pool)

    # Read applied rows ordered by applied_at DESC (newest first) so the
    # rollback order is well-defined even if migration ids ever stop
    # being lex-sortable for some reason.
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                f"SELECT id, applied_at FROM {MIGRATIONS_TABLE} "
                f"ORDER BY applied_at DESC, id DESC"
            )
            applied_rows = await cur.fetchall()
    applied_ids_newest_first = [r[0] for r in applied_rows]

    if not applied_ids_newest_first:
        print("no applied migrations -- nothing to roll back")
        return []

    # Pick which ids to revert.
    if count is not None:
        if count < 1:
            raise ValueError("count must be >= 1")
        targets = applied_ids_newest_first[:count]
    else:
        assert to_id is not None
        if to_id not in applied_ids_newest_first:
            raise RuntimeError(
                f"migration {to_id!r} is not currently applied -- "
                f"nothing to roll back to. Applied: "
                f"{applied_ids_newest_first}"
            )
        # Roll back everything newer than to_id (exclusive).
        targets = []
        for mid in applied_ids_newest_first:
            if mid == to_id:
                break
            targets.append(mid)

    if not targets:
        print(f"already at {to_id} -- no rollback needed")
        return []

    # Validate each target has a paired down.sql AND that the on-disk
    # forward checksum still matches what's recorded (so we know we're
    # looking at the same migration the operator actually applied).
    recorded = await _applied_with_checksums(pool)
    missing: list[str] = []
    edited: list[str] = []
    for mid in targets:
        m = by_id.get(mid)
        if m is None:
            missing.append(f"{mid} (file deleted from disk)")
            continue
        if not m.reversible:
            missing.append(f"{mid} (no .down.sql paired)")
            continue
        prev = recorded.get(mid)
        if prev is not None and prev != m.checksum:
            edited.append(mid)

    if missing:
        raise RuntimeError(
            "refusing to roll back -- missing down migrations:\n  "
            + "\n  ".join(missing)
            + "\n\nOptions: (a) write the missing .down.sql files, "
            "(b) restore from a pg_dump backup instead "
            "(scripts/db-restore.sh)."
        )
    if edited:
        raise RuntimeError(
            f"refusing to roll back -- forward .sql files for these "
            f"already-applied migrations have been edited: {edited}. "
            f"Restore from backup or revert the file."
        )

    print(f"rolling back {len(targets)} migration(s): {targets}")

    reverted: list[str] = []
    for mid in targets:
        m = by_id[mid]
        assert m.down_sql is not None
        if dry_run:
            print(f"would roll back {mid} ({len(m.down_sql)} bytes)")
            reverted.append(mid)
            continue

        print(f"rolling back {mid}...", end="", flush=True)
        t0 = time.time()
        try:
            async with pool.connection() as conn:
                async with conn.transaction():
                    async with conn.cursor() as cur:
                        await cur.execute(m.down_sql)
                        await cur.execute(
                            f"DELETE FROM {MIGRATIONS_TABLE} WHERE id = %s",
                            (mid,),
                        )
        except Exception as exc:
            elapsed_ms = int((time.time() - t0) * 1000)
            print(f" FAILED after {elapsed_ms}ms: {exc}")
            raise
        elapsed_ms = int((time.time() - t0) * 1000)
        print(f" ok ({elapsed_ms}ms)")
        reverted.append(mid)

    return reverted


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

async def status(
    pool: AsyncConnectionPool,
    *,
    migrations_dir: Path | None = None,
) -> None:
    """Print which migrations are applied vs pending."""
    dir_path = migrations_dir or _default_migrations_dir()
    migrations = discover_migrations(dir_path)
    await ensure_migrations_table(pool)
    recorded = await _applied_with_checksums(pool)

    if not migrations:
        print(f"no migrations found in {dir_path}")
        return

    print(f"migrations directory: {dir_path}")
    print(f"{'STATUS':<10} {'REVERSIBLE':<11} {'ID':<40} {'CHECKSUM':<14}")
    print("-" * 78)
    for m in migrations:
        prev = recorded.get(m.id)
        if prev is None:
            flag = "pending"
        elif prev != m.checksum:
            flag = "EDITED!"
        else:
            flag = "applied"
        rev = "yes" if m.reversible else "no"
        print(f"{flag:<10} {rev:<11} {m.id:<40} {m.checksum[:12]}")

    # Flag any rows that exist in the DB but not on disk (a deleted
    # migration file -- usually an operator error worth surfacing).
    orphan = sorted(set(recorded) - {m.id for m in migrations})
    if orphan:
        print()
        print("WARNING: recorded but missing from disk:")
        for oid in orphan:
            print(f"  - {oid}")


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------

def _resolve_dsn() -> str:
    dsn = os.getenv("OPSRAG_POSTGRES_DSN") or os.getenv("POSTGRES_DSN")
    if not dsn:
        print(
            "ERROR: neither OPSRAG_POSTGRES_DSN nor POSTGRES_DSN is set",
            file=sys.stderr,
        )
        sys.exit(2)
    return dsn


async def _with_pool(coro_factory) -> None:
    dsn = _resolve_dsn()
    pool = AsyncConnectionPool(
        conninfo=dsn,
        min_size=1,
        max_size=2,
        open=False,
        # IMPORTANT: do NOT set prepare_threshold=0 here. Postgres can't
        # prepare multi-statement SQL, and DDL migration files commonly
        # contain several statements separated by semicolons. Leave
        # prepare_threshold at its default (5) so single-statement
        # housekeeping queries (SELECT, INSERT into the migrations
        # table) still get the fast path but multi-statement files
        # execute directly.
        kwargs={"autocommit": False},
    )
    await pool.open()
    try:
        await coro_factory(pool)
    finally:
        await pool.close()


def _repo_root() -> Path:
    # opsrag/db/migrate.py -> repo root is parents[2]
    return Path(__file__).resolve().parents[2]


def _run_script(script_name: str, *args: str) -> int:
    script = _repo_root() / "scripts" / script_name
    if not script.exists():
        print(f"ERROR: {script} not found", file=sys.stderr)
        return 2
    cmd = [str(script), *args]
    _log.info("running %s", " ".join(cmd))
    return subprocess.call(cmd)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m opsrag.db.migrate",
        description="OpsRAG SQL migration runner + backup/restore wrappers.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    up = sub.add_parser("up", help="apply all pending migrations")
    up.add_argument(
        "--dry-run",
        action="store_true",
        help="print plan without executing",
    )

    sub.add_parser("status", help="show applied vs pending migrations")

    down = sub.add_parser(
        "down",
        help="roll back one or more applied migrations via their .down.sql",
    )
    down_target = down.add_mutually_exclusive_group(required=True)
    down_target.add_argument(
        "--count",
        type=int,
        help="number of most-recent migrations to roll back",
    )
    down_target.add_argument(
        "--to",
        dest="to_id",
        help="roll back every migration applied after this id (id stays applied)",
    )
    down.add_argument(
        "--dry-run",
        action="store_true",
        help="print plan without executing",
    )

    sub.add_parser("backup", help="run scripts/db-backup.sh")

    restore = sub.add_parser("restore", help="run scripts/db-restore.sh <path>")
    restore.add_argument("path", help="path to a .sql.gz backup")
    restore.add_argument(
        "--force",
        action="store_true",
        help="skip the confirmation prompt",
    )

    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _build_parser().parse_args(argv)

    if args.cmd == "up":
        async def _run(pool: AsyncConnectionPool) -> None:
            await apply_all(pool, dry_run=args.dry_run)
        asyncio.run(_with_pool(_run))
        return 0

    if args.cmd == "status":
        async def _run(pool: AsyncConnectionPool) -> None:
            await status(pool)
        asyncio.run(_with_pool(_run))
        return 0

    if args.cmd == "down":
        async def _run(pool: AsyncConnectionPool) -> None:
            await roll_back(
                pool,
                count=args.count,
                to_id=args.to_id,
                dry_run=args.dry_run,
            )
        asyncio.run(_with_pool(_run))
        return 0

    if args.cmd == "backup":
        return _run_script("db-backup.sh")

    if args.cmd == "restore":
        extra = ["--force"] if args.force else []
        return _run_script("db-restore.sh", args.path, *extra)

    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
