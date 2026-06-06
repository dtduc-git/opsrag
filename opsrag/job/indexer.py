"""Ephemeral indexing Job entrypoint.

Replaces the always-on ``indexer`` deployment: instead of a long-running pod
that holds the single-writer in-memory tracker, indexing now runs as a
run-to-completion process (a k8s Job in prod, ``docker compose run`` locally).
It builds the same providers + ingestion pipeline the API uses, writes progress
to the durable Postgres job-state (so backend pods reflect it live), indexes
the requested target(s), then exits.

Usage::

    python -m opsrag.job.indexer --repo devops/foo [--branch master]
    python -m opsrag.job.indexer --source confluence --scope SRE
    python -m opsrag.job.indexer --all          # every configured repo + source

Exit code is 0 on success, 1 if any target failed (so a k8s Job surfaces the
failure). The durable job-state carries the per-target status/error regardless.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

from opsrag.config import OpsRAGConfig
from opsrag.factory import build_providers
from opsrag.indexing_tracker import indexing_tracker
from opsrag.ingestion.pipeline import IngestionPipeline

_log = logging.getLogger("opsrag.job.indexer")


async def _maybe_migrate(cfg: OpsRAGConfig) -> None:
    """Apply DB migrations (idempotent). Mirrors the API lifespan so a Job that
    runs before the API is rolled still has the job-state tables. Opt out with
    OPSRAG_AUTO_MIGRATE=false."""
    if cfg.session.provider != "postgres":
        return
    if os.environ.get("OPSRAG_AUTO_MIGRATE", "true").lower() in ("false", "0", "no", "off"):
        return
    dsn = cfg.session.dsn or os.environ.get(cfg.session.dsn_env, "")
    if not dsn:
        return
    try:
        from psycopg_pool import AsyncConnectionPool

        from opsrag.db.migrate import apply_all
        pool = AsyncConnectionPool(conninfo=dsn, min_size=1, max_size=1, open=False,
                                   kwargs={"autocommit": False})
        await pool.open()
        try:
            await apply_all(pool)
        finally:
            await pool.close()
    except Exception as exc:  # noqa: BLE001
        _log.warning("migrations failed (%s); proceeding -- tables may be missing", exc)


def _build_pipeline(providers) -> IngestionPipeline:
    """Construct the ingestion pipeline with the SAME wiring as the API
    (opsrag.api.server) so a Job indexes identically to in-process indexing."""
    return IngestionPipeline(
        scm=providers.scm,
        parsers=providers.parsers,
        chunker=providers.chunker,
        embedder=providers.embedder,
        vector_store=providers.vector_store,
        graph_store=providers.graph_store,
        entity_extractor=providers.entity_extractor,
        llm=providers.llm,
        indexed_files=providers.indexed_files,
        sources=providers.sources,
        code_embedder=providers.code_embedder,
        code_vector_store=providers.code_vector_store,
        light_graph=providers.light_graph,
    )


async def _open_stores(providers) -> None:
    """Open the async stores the pipeline + flush loop need. Vector store
    (Qdrant) connects lazily, so only the Postgres-backed stores need open()."""
    from opsrag.indexed_files.postgres import PostgresIndexedFilesTracker
    if isinstance(providers.indexed_files, PostgresIndexedFilesTracker):
        await providers.indexed_files.open()
    if providers.light_graph is not None:
        try:
            await providers.light_graph.open()
        except Exception as exc:  # noqa: BLE001
            _log.warning("light-graph open failed (%s); entity-expansion edges disabled", exc)
            providers.light_graph = None
            # The pipeline already captured the ref; clear it so it no-ops.
    if providers.index_store is not None:
        await providers.index_store.open()


async def _close_stores(providers) -> None:
    from opsrag.indexed_files.postgres import PostgresIndexedFilesTracker
    try:
        if hasattr(providers.scm, "close"):
            await providers.scm.close()
    except Exception:  # noqa: BLE001
        pass
    for store in (providers.light_graph, providers.index_store):
        if store is not None:
            try:
                await store.close()
            except Exception:  # noqa: BLE001
                pass
    if isinstance(providers.indexed_files, PostgresIndexedFilesTracker):
        try:
            await providers.indexed_files.close()
        except Exception:  # noqa: BLE001
            pass


def _resolve_branch(cfg: OpsRAGConfig, branch: str | None) -> str:
    return branch or getattr(getattr(cfg, "scm", None), "default_branch", None) or "main"


async def _run(args: argparse.Namespace) -> int:
    cfg = OpsRAGConfig.load()
    await _maybe_migrate(cfg)
    providers = build_providers(cfg)
    await _open_stores(providers)
    pipeline = _build_pipeline(providers)

    # Set the pipeline's light_graph to the (possibly-cleared) provider ref so a
    # failed open doesn't leave a half-open store wired in.
    pipeline.light_graph = providers.light_graph

    # Build the target list.
    targets: list[tuple[str, dict]] = []
    if args.all:
        for repo, branch in cfg.scm.repos_with_branch():
            targets.append(("repo", {"repo": repo, "branch": branch}))
        # Sources (Confluence, etc.) -- index every configured scope.
        for source_type, scopes in _configured_source_scopes(cfg).items():
            for scope in scopes:
                targets.append(("source", {"source_type": source_type, "scope": scope}))
    elif args.repo:
        targets.append(("repo", {"repo": args.repo, "branch": _resolve_branch(cfg, args.branch)}))
    elif args.source:
        if not args.scope:
            _log.error("--source requires --scope")
            return 2
        targets.append(("source", {"source_type": args.source, "scope": args.scope}))
    else:
        _log.error("nothing to do: pass --repo, --source/--scope, or --all")
        return 2

    # Start the durable-state flush loop (this Job is the writer).
    stop_event = asyncio.Event()
    flush_task = None
    if providers.index_store is not None:
        from opsrag.indexing.pg_store import flush_loop
        flush_task = asyncio.create_task(
            flush_loop(providers.index_store, indexing_tracker, stop_event=stop_event)
        )

    failures = 0
    try:
        for kind, params in targets:
            try:
                if kind == "repo":
                    repo, branch = params["repo"], params["branch"]
                    indexing_tracker.queue_repo(repo, branch)
                    n = await pipeline.index_repo(repo, branch)
                    _log.info("indexed repo=%s branch=%s chunks=%d", repo, branch, n)
                else:
                    st, scope = params["source_type"], params["scope"]
                    indexing_tracker.ensure_queued(f"{st}:{scope}", st, source_type=st)
                    n = await pipeline.index_source(st, scope)
                    _log.info("indexed source=%s scope=%s chunks=%d", st, scope, n)
            except Exception as exc:  # noqa: BLE001
                failures += 1
                _log.error("indexing target failed %s %s: %s", kind, params, exc)
                if kind == "repo":
                    indexing_tracker.repo_failed(params["repo"], params["branch"], str(exc))
                else:
                    indexing_tracker.repo_failed(f"{params['source_type']}:{params['scope']}",
                                                 params["source_type"], str(exc))
    finally:
        stop_event.set()
        if flush_task is not None:
            try:
                await asyncio.wait_for(flush_task, timeout=10)
            except (TimeoutError, Exception):  # noqa: BLE001
                pass
        await _close_stores(providers)

    return 1 if failures else 0


def _configured_source_scopes(cfg: OpsRAGConfig) -> dict[str, list[str]]:
    """Map configured non-git sources to their scopes for ``--all``. Currently
    Confluence (its space allow-list). Extend as new sources gain auto-index."""
    out: dict[str, list[str]] = {}
    conf = getattr(cfg, "confluence", None)
    if conf is not None and getattr(conf, "enabled", False):
        spaces = list(getattr(conf, "spaces_allowlist", []) or [])
        if spaces:
            out["confluence"] = spaces
    return out


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=os.environ.get("OPSRAG_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    parser = argparse.ArgumentParser(prog="opsrag.job.indexer", description=__doc__)
    parser.add_argument("--repo", help="git repo to index (e.g. devops/foo)")
    parser.add_argument("--branch", help="branch (default: scm.default_branch)")
    parser.add_argument("--source", help="non-git source type (e.g. confluence)")
    parser.add_argument("--scope", help="source scope (e.g. a Confluence space key)")
    parser.add_argument("--all", action="store_true",
                        help="index every configured repo + source (scheduled reindex)")
    args = parser.parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
