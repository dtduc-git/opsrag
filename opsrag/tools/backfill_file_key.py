"""Backfill the `file_key` payload field onto pre-existing Qdrant points.

`file_key = repo + "\\x00" + source_path` (see opsrag/vectorstores/qdrant.py
:_make_file_key) is the keyword-indexed exact-match key the fast delete path
(`vector_store.use_file_key_delete`) filters on. Points written before that
field existed must be backfilled BEFORE the flag is flipped ON -- a translated
delete silently MISSES points without the field, so stale chunks would keep
matching queries. Metadata-only: `set_payload` merges one key into existing
payloads; vectors are never touched and nothing is re-embedded.

Rollout order (enforced here where possible):
  1. Deploy the image whose `upsert()` writes file_key (new writes self-populate).
  2. Run this tool per collection:   backfill -> exhaustive verify -> index.
  3. Only then set `vector_store.use_file_key_delete: true` (+ restart).

Durability guards (each failure mode was observed or adversarially reviewed):
  - `set_payload(..., wait=True)`: no fire-and-forget -- an unacked write may
    never land and sampling would not notice.
  - Per-call retry with backoff; the run REFUSES to report success (exit 1)
    while any point still failed.
  - Verify is EXHAUSTIVE, not sampled: `count(IsEmptyCondition(file_key))`
    must be 0 server-side before `--create-index` proceeds.
  - Throttled scroll/write loop (--sleep-ms) for the single-replica Qdrant.
    QUIESCE INGESTION while this runs (suspend the indexer CronJob / avoid
    manual Jobs): a bulk write sweep during heavy `wait=True` ingestion can
    saturate the write executor and time out unrelated ingest writes.

Usage (from a backend pod, which has network reach + the opsrag package):
  python -m opsrag.tools.backfill_file_key \
      --url http://qdrant:6333 \
      --collection <main-collection> [--collection <code-collection>] \
      [--dry-run | --verify-only | --create-index] [--page-size 512] [--sleep-ms 100]
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

from qdrant_client import AsyncQdrantClient
from qdrant_client import models as qm

from opsrag.vectorstores.qdrant import _make_file_key

_log = logging.getLogger("opsrag.tools.backfill_file_key")

_RETRIES = 5
_BACKOFF_BASE_S = 0.5


def group_missing_file_key(points) -> dict[str, list]:
    """Group a scroll page's points by the file_key each should carry,
    EXCLUDING points whose stored file_key already matches (idempotent
    re-runs skip completed work). Chunks of the same file share a key, so
    one `set_payload` per group covers many points."""
    groups: dict[str, list] = {}
    for p in points:
        payload = p.payload or {}
        key = _make_file_key(payload.get("repo"), payload.get("source_path"))
        if payload.get("file_key") == key:
            continue
        groups.setdefault(key, []).append(p.id)
    return groups


async def _set_payload_with_retry(
    client: AsyncQdrantClient, collection: str, key: str, ids: list
) -> bool:
    """One durable (wait=True) set_payload, retried with exponential backoff.
    Returns False -- never raises -- when all retries are exhausted, so the
    caller can aggregate failures and refuse to succeed."""
    for attempt in range(_RETRIES):
        try:
            await client.set_payload(
                collection_name=collection,
                payload={"file_key": key},
                points=ids,
                wait=True,
            )
            return True
        except Exception as exc:  # noqa: BLE001 -- aggregate, don't die mid-sweep
            wait_s = _BACKOFF_BASE_S * (2**attempt)
            _log.warning(
                "set_payload failed (attempt %d/%d, %d point(s), retry in %.1fs): %s",
                attempt + 1, _RETRIES, len(ids), wait_s, exc,
            )
            await asyncio.sleep(wait_s)
    return False


async def count_missing(client: AsyncQdrantClient, collection: str) -> int:
    """EXHAUSTIVE server-side count of points still missing file_key. This is
    the gate for --create-index and for flipping use_file_key_delete ON --
    sampling cannot prove the 100% coverage the delete path depends on."""
    res = await client.count(
        collection_name=collection,
        count_filter=qm.Filter(
            must=[qm.IsEmptyCondition(is_empty=qm.PayloadField(key="file_key"))]
        ),
        exact=True,
    )
    return res.count


async def backfill_collection(
    client: AsyncQdrantClient,
    collection: str,
    *,
    page_size: int,
    sleep_ms: int,
    dry_run: bool,
) -> tuple[int, int]:
    """Scroll every point; set_payload the missing/stale file_keys.
    Returns (points_updated, points_failed)."""
    updated = failed = scanned = 0
    offset = None
    while True:
        points, offset = await client.scroll(
            collection_name=collection,
            limit=page_size,
            offset=offset,
            with_payload=["repo", "source_path", "file_key"],
            with_vectors=False,
        )
        scanned += len(points)
        for key, ids in group_missing_file_key(points).items():
            if dry_run:
                updated += len(ids)
                continue
            if await _set_payload_with_retry(client, collection, key, ids):
                updated += len(ids)
            else:
                failed += len(ids)
        if scanned and scanned % 50_000 < page_size:
            _log.info(
                "[%s] scanned=%d updated=%d failed=%d", collection, scanned, updated, failed
            )
        if offset is None:
            break
        if sleep_ms:
            await asyncio.sleep(sleep_ms / 1000)
    _log.info(
        "[%s] backfill %s: scanned=%d updated=%d failed=%d",
        collection, "DRY-RUN" if dry_run else "done", scanned, updated, failed,
    )
    return updated, failed


async def create_index(client: AsyncQdrantClient, collection: str) -> None:
    """KEYWORD payload index on file_key. Runs AFTER the exhaustive verify so
    the index build sees fully-populated values. Idempotent."""
    await client.create_payload_index(
        collection_name=collection,
        field_name="file_key",
        field_schema=qm.PayloadSchemaType.KEYWORD,
        wait=True,
    )
    _log.info("[%s] keyword index on file_key created", collection)


async def run(args: argparse.Namespace) -> int:
    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    client = AsyncQdrantClient(url=args.url, api_key=api_key, timeout=args.timeout)
    exit_code = 0
    try:
        for collection in args.collection:
            if not args.verify_only:
                _, failed = await backfill_collection(
                    client, collection,
                    page_size=args.page_size, sleep_ms=args.sleep_ms,
                    dry_run=args.dry_run,
                )
                if failed:
                    _log.error(
                        "[%s] %d point(s) FAILED after %d retries -- re-run to repair; "
                        "do NOT create the index or flip use_file_key_delete",
                        collection, failed, _RETRIES,
                    )
                    exit_code = 1
                    continue
            if args.dry_run:
                continue
            missing = await count_missing(client, collection)
            _log.info("[%s] exhaustive verify: %d point(s) missing file_key", collection, missing)
            if missing:
                _log.error(
                    "[%s] verify FAILED -- do NOT create the index or flip "
                    "use_file_key_delete until this is 0",
                    collection,
                )
                exit_code = 1
                continue
            if args.create_index:
                await create_index(client, collection)
        return exit_code
    finally:
        await client.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--url", required=True, help="Qdrant base URL")
    parser.add_argument(
        "--collection", action="append", required=True,
        help="Target collection; repeat for several (main + code)",
    )
    parser.add_argument("--api-key-env", default=None, help="Env var holding the Qdrant API key")
    parser.add_argument("--page-size", type=int, default=512)
    parser.add_argument("--sleep-ms", type=int, default=100, help="Pause between scroll pages")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--dry-run", action="store_true", help="Count work; write nothing")
    parser.add_argument(
        "--verify-only", action="store_true",
        help="Skip the backfill; just run the exhaustive missing-file_key count",
    )
    parser.add_argument(
        "--create-index", action="store_true",
        help="Create the keyword index -- only runs when the exhaustive verify hits 0",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
