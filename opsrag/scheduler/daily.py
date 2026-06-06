"""Daily indexing job + scheduler factory.

The scheduler runs `daily_index_job` once per day. The job walks the
configured repos and calls into the same `IngestionPipeline.index_repo`
that powers `POST /index/repo` and the startup auto-index. We don't
duplicate indexing logic -- we just trigger it on a cron.

Invariants:
- The job is idempotent. `IngestionPipeline.index_repo` skips files
  whose content_hash hasn't changed (Step 7 -- `indexed_files` table).
  Running twice in the same day is harmless beyond a few embed calls
  for new files.
- Failures on one repo don't cascade. `_index_one` catches and logs;
  other repos still run.
- Concurrency is bounded by `parallel_limit` (default 3) to respect
  Vertex per-minute token quotas.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

_log = logging.getLogger("opsrag.scheduler.daily")


async def daily_index_job(
    repo_pairs: list[tuple[str, str]],
    pipeline: Any,
    parallel_limit: int = 3,
    source_scopes: list[tuple[str, str]] | None = None,
) -> None:
    """Run a daily indexing pass with bounded parallelism.

    `repo_pairs` -- list of `(repo, branch)` for git sources via
    `pipeline.index_repo`. Existing single-source path.

    `source_scopes` -- list of `(source_type, scope)` for non-git
    sources (Confluence space keys, Rootly project ids, etc.) via
    `pipeline.index_source`. Each tuple's source_type must be
    registered in `pipeline.sources`. Phase 2 addition.

    Both lists are exercised in the same run; non-git sources run
    sequentially after git so they don't compete for Vertex token
    quota. Per-item failures are isolated.
    """
    source_scopes = source_scopes or []
    if not repo_pairs and not source_scopes:
        _log.info("daily-index: no repos or sources configured, skipping")
        return

    sem = asyncio.Semaphore(max(1, parallel_limit))

    async def _index_repo(repo: str, branch: str) -> None:
        async with sem:
            try:
                _log.info("daily-index starting repo=%s branch=%s", repo, branch)
                count = await pipeline.index_repo(repo, branch=branch)
                _log.info("daily-index done repo=%s chunks=%d", repo, count)
            except Exception as exc:
                _log.warning("daily-index failed repo=%s: %s", repo, exc)

    async def _index_source(source_type: str, scope: str) -> None:
        async with sem:
            try:
                _log.info(
                    "daily-index starting source=%s scope=%s",
                    source_type, scope,
                )
                count = await pipeline.index_source(source_type, scope)
                _log.info(
                    "daily-index done source=%s scope=%s chunks=%d",
                    source_type, scope, count,
                )
            except Exception as exc:
                _log.warning(
                    "daily-index failed source=%s scope=%s: %s",
                    source_type, scope, exc,
                )

    _log.info(
        "daily-index batch start: %d repos + %d source-scopes, parallel_limit=%d",
        len(repo_pairs), len(source_scopes), parallel_limit,
    )
    # Run git repos first, then non-git sources. Two reasons: keeps
    # log timing predictable, and avoids a noisy mix of source types
    # in stdout when triaging.
    if repo_pairs:
        await asyncio.gather(
            *(_index_repo(repo, branch) for repo, branch in repo_pairs),
            return_exceptions=False,
        )
    if source_scopes:
        await asyncio.gather(
            *(_index_source(s, sc) for s, sc in source_scopes),
            return_exceptions=False,
        )
    _log.info("daily-index batch done")


def build_scheduler(
    scheduler_cfg: Any,
    job_callable: Any,
    job_id: str = "opsrag-daily-index",
) -> Any:
    """Construct an `AsyncIOScheduler` with the daily job pre-registered.

    Returns an unstarted scheduler -- caller is responsible for `start()`
    and `shutdown()` (typically wired into FastAPI lifespan).

    `scheduler_cfg` is a `SchedulerConfig`. `job_callable` is a zero-arg
    coroutine function (the daily run) -- bind your repo list + pipeline
    via a closure or `functools.partial` before passing in.

    Uses an in-memory jobstore intentionally. SQLAlchemyJobStore would
    require the job and its arguments to be serializable, but our job
    closes over a live IngestionPipeline (DB connections, Vertex client)
    which is not. The cron schedule is reproduced from config on every
    container start, so a persistent jobstore buys nothing for a daily
    cron. Multi-replica deployments would need to revisit -- use
    SQLAlchemyJobStore + a string-reference job that resolves the
    pipeline at fire time from app state.
    """
    from apscheduler.jobstores.memory import MemoryJobStore
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.triggers.cron import CronTrigger

    scheduler = AsyncIOScheduler(
        jobstores={"default": MemoryJobStore()},
        timezone=scheduler_cfg.timezone,
    )
    _log.info("scheduler: using in-memory jobstore (rebuilt from config on restart)")

    trigger = CronTrigger(
        hour=scheduler_cfg.cron_hour,
        minute=scheduler_cfg.cron_minute,
        timezone=scheduler_cfg.timezone,
        jitter=max(0, int(scheduler_cfg.jitter_seconds)),
    )

    # `replace_existing=True` so a redeploy with a tweaked cron updates
    # the persisted job instead of erroring on duplicate id.
    scheduler.add_job(
        job_callable,
        trigger=trigger,
        id=job_id,
        replace_existing=True,
        misfire_grace_time=3600,  # accept up to 1h late firing after restarts
        coalesce=True,             # collapse multiple missed fires into one
        max_instances=1,           # never overlap the daily job with itself
    )
    return scheduler
