"""Step 6 -- daily indexing scheduler.

APScheduler `AsyncIOScheduler` + `SQLAlchemyJobStore` on Postgres so the
schedule survives container restarts. Fires one job per day at the
configured local hour (default 02:00 Asia/Ho_Chi_Minh) with +/-15min
jitter; the job iterates the configured repos with a semaphore-bounded
parallelism cap (same as the startup auto-index path).

Public API:
- `build_scheduler(cfg.scheduler, indexer)` -> `AsyncIOScheduler` ready to
  start.
- `daily_index_job(repo_pairs, pipeline, parallel_limit)` -> the coroutine
  the cron trigger fires.

Designed for one scheduler instance per FastAPI process. Multi-replica
deployments can share the Postgres jobstore safely (APScheduler's row
locking ensures only one replica claims each fire).
"""
from __future__ import annotations

from opsrag.scheduler.daily import build_scheduler, daily_index_job

__all__ = ["build_scheduler", "daily_index_job"]
