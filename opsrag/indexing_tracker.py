"""Indexing progress tracker -- tracks per-repo file/chunk counts and status."""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum

# Cap on the in-memory job-history ring buffer. The tracker is in-process
# and resets on restart (same as the per-source state), so this is a
# best-effort recent-history view, not a durable audit log.
_MAX_JOB_HISTORY = 200


class RepoStatus(str, Enum):
    QUEUED = "queued"
    LISTING = "listing"
    INDEXING = "indexing"
    DONE = "done"
    FAILED = "failed"


@dataclass
class RepoProgress:
    repo: str
    branch: str
    status: RepoStatus = RepoStatus.QUEUED
    total_files: int = 0
    indexed_files: int = 0
    skipped_files: int = 0
    total_chunks: int = 0
    entities_found: int = 0
    error: str | None = None
    started_at: float = 0.0
    finished_at: float = 0.0
    # Source kind, used by the UI to group repos vs Confluence vs Slack
    # vs ... Defaults to "git" so existing call sites stay unchanged.
    source_type: str = "git"
    # Human-readable label for the UI. Used when the `repo` field is an
    # opaque identifier (e.g. Slack `slack:CC448TKTQ` -> display
    # "slack:#devops"). Set lazily by ingestion code via
    # `set_display_name`. UI falls back to `repo` when None.
    display_name: str | None = None

    @property
    def processed_files(self) -> int:
        """Files we've finished processing -- both ones that produced chunks and
        ones we deliberately skipped (no parser claim, parse error, etc.)."""
        return self.indexed_files + self.skipped_files

    @property
    def percent(self) -> float:
        if self.total_files == 0:
            return 0.0
        return round(self.processed_files / self.total_files * 100, 1)

    @property
    def elapsed_seconds(self) -> float:
        end = self.finished_at or time.time()
        if self.started_at == 0:
            return 0.0
        return round(end - self.started_at, 1)


@dataclass
class JobRun:
    """A single indexing run for one source.

    Unlike ``RepoProgress`` (which holds the *current* state per source and
    is overwritten on each re-index), a JobRun is appended once per run and
    kept in a ring buffer so the UI can show run *history* -- when each run
    started, how long it took, and whether it succeeded or failed (with the
    error). ``kind`` is "run" for live ingestion and "restored" for entries
    reconstructed from the vector store at startup.
    """
    id: int
    repo: str
    branch: str
    source_type: str
    display_name: str | None
    status: str  # "running" | "success" | "failed"
    started_at: float
    finished_at: float = 0.0
    chunks_indexed: int = 0
    files_indexed: int = 0
    error: str | None = None
    kind: str = "run"  # "run" | "restored"

    @property
    def duration_seconds(self) -> float:
        end = self.finished_at or time.time()
        if self.started_at == 0:
            return 0.0
        return round(end - self.started_at, 1)


class IndexingTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._repos: dict[str, RepoProgress] = {}
        # Job-history ring buffer (newest-first) + the in-flight job per key
        # so done/failed transitions can finalise the run that started it.
        self._jobs: deque[JobRun] = deque(maxlen=_MAX_JOB_HISTORY)
        self._current_job: dict[str, JobRun] = {}
        self._job_counter = 0

    # -- job-history helpers (callers already hold self._lock) -------------
    def _next_job_id(self) -> int:
        self._job_counter += 1
        return self._job_counter

    def _job_dict(self, j: JobRun) -> dict:
        return {
            "id": j.id,
            "repo": j.repo,
            "branch": j.branch,
            "source_type": j.source_type,
            "display_name": j.display_name,
            "status": j.status,
            "started_at": j.started_at,
            "finished_at": j.finished_at or None,
            "duration_seconds": j.duration_seconds,
            "chunks_indexed": j.chunks_indexed,
            "files_indexed": j.files_indexed,
            "error": j.error,
            "kind": j.kind,
        }

    def queue_repo(self, repo: str, branch: str, source_type: str = "git") -> None:
        with self._lock:
            key = f"{repo}@{branch}"
            self._repos[key] = RepoProgress(
                repo=repo, branch=branch, source_type=source_type,
            )

    def ensure_queued(self, repo: str, branch: str, source_type: str = "git") -> None:
        """Idempotent queue -- create tracker entry if not already present.

        Used by paths that don't go through the auto-index startup flow
        (manual POST /index/repo, webhook reindex, etc.). Without this,
        start_listing/start_indexing silently no-op and the dashboard
        stays empty even while files are being indexed.
        """
        with self._lock:
            key = f"{repo}@{branch}"
            if key not in self._repos:
                self._repos[key] = RepoProgress(
                    repo=repo, branch=branch, source_type=source_type,
                )

    def start_listing(self, repo: str, branch: str) -> None:
        with self._lock:
            key = f"{repo}@{branch}"
            if key in self._repos:
                rp = self._repos[key]
                rp.status = RepoStatus.LISTING
                rp.started_at = time.time()
                # Open a new job-history run for this source.
                job = JobRun(
                    id=self._next_job_id(),
                    repo=rp.repo,
                    branch=rp.branch,
                    source_type=rp.source_type,
                    display_name=rp.display_name,
                    status="running",
                    started_at=rp.started_at,
                )
                self._current_job[key] = job
                self._jobs.appendleft(job)

    def start_indexing(self, repo: str, branch: str, total_files: int) -> None:
        with self._lock:
            key = f"{repo}@{branch}"
            if key in self._repos:
                self._repos[key].status = RepoStatus.INDEXING
                self._repos[key].total_files = total_files

    def file_indexed(self, repo: str, branch: str, chunks: int, entities: int = 0) -> None:
        with self._lock:
            key = f"{repo}@{branch}"
            if key in self._repos:
                self._repos[key].indexed_files += 1
                self._repos[key].total_chunks += chunks
                self._repos[key].entities_found += entities

    def file_skipped(self, repo: str, branch: str) -> None:
        with self._lock:
            key = f"{repo}@{branch}"
            if key in self._repos:
                # Track skipped separately so the dashboard reports honest counts.
                # Progress percent already counts skipped via processed_files.
                self._repos[key].skipped_files += 1

    def repo_done(self, repo: str, branch: str) -> None:
        with self._lock:
            key = f"{repo}@{branch}"
            if key in self._repos:
                rp = self._repos[key]
                rp.status = RepoStatus.DONE
                rp.finished_at = time.time()
                self._finalize_job(key, rp, "success", None)

    def _finalize_job(self, key: str, rp: RepoProgress, status: str, error: str | None) -> None:
        """Close out the in-flight run for ``key`` (or synthesise one if the
        run was never opened via start_listing). Caller holds self._lock."""
        job = self._current_job.pop(key, None)
        if job is None:
            job = JobRun(
                id=self._next_job_id(),
                repo=rp.repo,
                branch=rp.branch,
                source_type=rp.source_type,
                display_name=rp.display_name,
                status=status,
                started_at=rp.started_at or time.time(),
            )
            self._jobs.appendleft(job)
        job.status = status
        job.error = error
        job.finished_at = time.time()
        job.chunks_indexed = rp.total_chunks
        job.files_indexed = rp.indexed_files
        job.display_name = rp.display_name

    def backfill_done(
        self,
        repo: str,
        branch: str,
        source_type: str,
        total_chunks: int,
        total_files: int,
    ) -> None:
        """Repopulate a 'done' entry from existing Qdrant state.

        The tracker is in-memory and resets every restart, so without
        this the dashboard forgets everything that was indexed before
        the process started -- Confluence disappears entirely, git repos
        show 0 chunks. Call once at startup after the vector store is
        ready. Skips entries that are currently in-flight so an active
        run isn't overwritten by stale numbers.
        """
        with self._lock:
            key = f"{repo}@{branch}"
            existing = self._repos.get(key)
            if existing and existing.status not in (RepoStatus.DONE, RepoStatus.QUEUED):
                return
            now = time.time()
            self._repos[key] = RepoProgress(
                repo=repo,
                branch=branch,
                source_type=source_type,
                status=RepoStatus.DONE,
                total_files=total_files,
                indexed_files=total_files,
                total_chunks=total_chunks,
                started_at=now,
                finished_at=now,
            )
            # Record a "restored" job so the run-history view isn't empty
            # for sources indexed before this process started.
            self._jobs.appendleft(JobRun(
                id=self._next_job_id(),
                repo=repo,
                branch=branch,
                source_type=source_type,
                display_name=None,
                status="success",
                started_at=now,
                finished_at=now,
                chunks_indexed=total_chunks,
                files_indexed=total_files,
                kind="restored",
            ))

    def set_display_name(self, repo: str, branch: str, name: str) -> None:
        """Attach a human-readable label to an existing tracker entry."""
        with self._lock:
            key = f"{repo}@{branch}"
            if key in self._repos:
                self._repos[key].display_name = name

    def repo_failed(self, repo: str, branch: str, error: str) -> None:
        with self._lock:
            key = f"{repo}@{branch}"
            if key in self._repos:
                rp = self._repos[key]
                rp.status = RepoStatus.FAILED
                rp.error = error
                rp.finished_at = time.time()
                self._finalize_job(key, rp, "failed", error)

    def get_summary(self) -> dict:
        with self._lock:
            repos = []
            total_files = 0
            total_indexed = 0
            total_chunks = 0
            for rp in self._repos.values():
                repos.append({
                    "repo": rp.repo,
                    "branch": rp.branch,
                    "status": rp.status.value,
                    "source_type": rp.source_type,
                    "display_name": rp.display_name,
                    "total_files": rp.total_files,
                    "indexed_files": rp.indexed_files,
                    "skipped_files": rp.skipped_files,
                    "processed_files": rp.processed_files,
                    "total_chunks": rp.total_chunks,
                    "entities_found": rp.entities_found,
                    "percent": rp.percent,
                    "elapsed_seconds": rp.elapsed_seconds,
                    "error": rp.error,
                    # Raw epoch timestamps -- additive, ignored by the UI, but
                    # needed by PostgresIndexStore.flush so the durable row can
                    # recompute percent/elapsed on read across pods.
                    "started_at": rp.started_at,
                    "finished_at": rp.finished_at,
                })
                total_files += rp.total_files
                total_indexed += rp.indexed_files
                total_chunks += rp.total_chunks

            return {
                "total_repos": len(self._repos),
                "total_files": total_files,
                "total_indexed": total_indexed,
                "total_chunks": total_chunks,
                "repos": repos,
            }

    def get_jobs(self) -> dict:
        """Run-history view: every indexing run (newest-first) with its
        status, timing, and -- when failed -- the error. Powers the
        Operations -> Indexing Jobs page (distinct from the Sources catalog,
        which shows current per-source state)."""
        with self._lock:
            jobs = [self._job_dict(j) for j in self._jobs]
            running = sum(1 for j in self._jobs if j.status == "running")
            failed = sum(1 for j in self._jobs if j.status == "failed")
            return {
                "jobs": jobs,
                "total": len(jobs),
                "running": running,
                "failed": failed,
            }


indexing_tracker = IndexingTracker()
