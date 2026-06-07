"""Local-filesystem indexer (T061).

Indexes documents from a local directory tree -- runbooks, postmortems, K8s
manifests, Terraform, markdown -- into the configured vector store, reusing
the same ``IngestionPipeline`` the SCM and webhook paths use. Each file is
wrapped in a ``RepoFile`` (the pipeline's unit of work), parsed by the first
matching parser, chunked, embedded, and upserted.

This powers ``scripts/seed-sample-corpus.sh`` and the quickstart's
"index the bundled sample corpus" step::

    python -m opsrag.ingestion.indexer samples/

Run inside the API container (so it shares the configured providers /
network), or locally with the same config + env.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import logging
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from opsrag.interfaces.scm import RepoFile

_log = logging.getLogger("opsrag.ingestion.indexer")

# File globs the indexer considers. MUST mirror IngestionPipeline.index_repo's
# default patterns so the local/quickstart corpus indexes the SAME file kinds as
# production SCM indexing -- otherwise local lacks all source code and eval runs
# against a corpus that doesn't match what production retrieves over.
DEFAULT_PATTERNS: tuple[str, ...] = (
    # Docs + IaC + config
    "**/*.md", "**/*.markdown",
    "**/*.tf", "**/*.hcl",
    "**/*.yaml", "**/*.yml",
    "**/Dockerfile",
    "**/*.tpl", "**/*.gotmpl",
    "**/*.json",
    "**/Makefile",
    # Source code (parity with pipeline.py SCM globs)
    "**/*.py", "**/*.pyi",
    "**/*.js", "**/*.jsx", "**/*.mjs", "**/*.cjs",
    "**/*.ts", "**/*.tsx",
    "**/*.vue",
    "**/*.go",
    "**/*.java", "**/*.kt", "**/*.kts",
    "**/*.html", "**/*.htm",
    "**/*.css", "**/*.scss", "**/*.sass", "**/*.less",
    "**/*.sh", "**/*.bash",
)


def _iter_files(root: Path, patterns: Iterable[str]) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for pat in patterns:
        for p in sorted(root.glob(pat)):
            if p.is_file() and p not in seen:
                seen.add(p)
                out.append(p)
    return out


def _to_repo_file(path: Path, root: Path, repo: str, branch: str) -> RepoFile | None:
    try:
        content = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError) as exc:
        _log.warning("skipping unreadable file %s: %s", path, exc)
        return None
    rel = path.relative_to(root).as_posix()
    stat = path.stat()
    return RepoFile(
        path=rel,
        content=content,
        sha=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        last_modified=datetime.fromtimestamp(stat.st_mtime, tz=UTC),
        repo=repo,
        branch=branch,
        metadata={"source": "local_fs", "root": str(root)},
    )


async def index_local_path(
    pipeline,
    root: str | Path,
    *,
    repo: str = "samples",
    branch: str = "local",
    patterns: Iterable[str] | None = None,
) -> dict[str, int]:
    """Index every matching file under ``root`` through ``pipeline``.

    Returns a summary ``{files_seen, files_indexed, chunks}``. A file that
    no parser claims, or that the dedup tracker has already seen, contributes
    zero chunks but is still counted in ``files_seen``.
    """
    root = Path(root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"index root is not a directory: {root}")

    files = _iter_files(root, patterns or DEFAULT_PATTERNS)
    summary = {"files_seen": len(files), "files_indexed": 0, "chunks": 0}

    for path in files:
        rf = _to_repo_file(path, root, repo, branch)
        if rf is None:
            continue
        try:
            n = await pipeline._process_file(rf)
        except Exception as exc:  # noqa: BLE001 -- one bad file must not abort the run
            _log.warning("indexing failed for %s: %s", rf.path, exc)
            continue
        if n > 0:
            summary["files_indexed"] += 1
            summary["chunks"] += n
            _log.info("indexed %s (%d chunks)", rf.path, n)
    return summary


def _build_pipeline_from_config():
    """Construct an IngestionPipeline from the active config + providers.
    Imported lazily so ``index_local_path`` stays usable with a hand-built
    pipeline in tests without pulling in the whole provider stack."""
    from opsrag.config import Settings
    from opsrag.factory import build_providers
    from opsrag.ingestion.pipeline import IngestionPipeline

    settings = Settings.load()
    # Install deployment context for any prompt rendering during ingestion
    # (e.g. contextual chunking). Empty default -> org-free.
    from opsrag.agent.prompt_render import set_active_deployment
    set_active_deployment(settings.deployment)
    providers = build_providers(settings)
    pipeline = IngestionPipeline(
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
    )
    return pipeline, providers


async def _amain(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    pipeline, providers = _build_pipeline_from_config()

    # Open the dedup tracker if it needs an explicit connection (Postgres).
    open_fn = getattr(providers.indexed_files, "open", None)
    if callable(open_fn):
        try:
            await open_fn()
        except Exception as exc:  # noqa: BLE001 -- dedup is best-effort
            _log.warning("indexed_files.open failed (continuing without dedup): %s", exc)

    summary = await index_local_path(
        pipeline,
        args.path,
        repo=args.repo,
        branch=args.branch,
    )
    _log.info(
        "done: %d file(s) seen, %d indexed, %d chunk(s) written to collection",
        summary["files_seen"],
        summary["files_indexed"],
        summary["chunks"],
    )
    # Non-zero exit if nothing was indexed AND files were present -- a likely
    # misconfiguration (wrong path, no parser matches, vector store down).
    if summary["files_seen"] > 0 and summary["files_indexed"] == 0:
        _log.error("no files were indexed; check the path and provider config")
        return 1
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m opsrag.ingestion.indexer",
        description="Index a local directory of docs into the vector store.",
    )
    parser.add_argument("path", help="directory to index (e.g. samples/)")
    parser.add_argument(
        "--repo",
        default="samples",
        help="logical repo label stored on each chunk (default: samples)",
    )
    parser.add_argument(
        "--branch",
        default="local",
        help="logical branch label stored on each chunk (default: local)",
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_amain(args)))


if __name__ == "__main__":
    main()
