"""Unit test (T061): the local-filesystem indexer discovers files, wraps
them as RepoFiles, and drives them through the pipeline's _process_file,
without needing a live vector store.
"""
from __future__ import annotations

import pytest

from opsrag.ingestion.indexer import DEFAULT_PATTERNS, _iter_files, index_local_path


class _StubPipeline:
    """Records the RepoFiles it is asked to process; returns a fixed chunk
    count so the indexer's accounting can be asserted."""

    def __init__(self, chunks_per_file: int = 2) -> None:
        self.processed: list = []
        self._chunks = chunks_per_file

    async def _process_file(self, rf) -> int:
        self.processed.append(rf)
        return self._chunks


def _write(root, rel, text):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


@pytest.mark.asyncio
async def test_indexes_matching_files(tmp_path):
    _write(tmp_path, "runbooks/a.md", "# A runbook\nrunbook body")
    _write(tmp_path, "manifests/x.yaml", "apiVersion: v1\nkind: ConfigMap\n")
    _write(tmp_path, "notes.txt", "ignored: not a matched extension")

    pipeline = _StubPipeline(chunks_per_file=2)
    summary = await index_local_path(pipeline, tmp_path)

    # The .txt is not in DEFAULT_PATTERNS -> not seen.
    assert summary["files_seen"] == 2
    assert summary["files_indexed"] == 2
    assert summary["chunks"] == 4
    paths = sorted(rf.path for rf in pipeline.processed)
    assert paths == ["manifests/x.yaml", "runbooks/a.md"]


@pytest.mark.asyncio
async def test_repo_file_fields_are_populated(tmp_path):
    _write(tmp_path, "runbooks/deploy.md", "# Deploy runbook\nsteps")
    pipeline = _StubPipeline()
    await index_local_path(pipeline, tmp_path, repo="samples", branch="local")

    rf = pipeline.processed[0]
    assert rf.path == "runbooks/deploy.md"  # relative, posix
    assert rf.repo == "samples"
    assert rf.branch == "local"
    assert len(rf.sha) == 64  # sha256 hex
    assert rf.metadata.get("source") == "local_fs"


@pytest.mark.asyncio
async def test_zero_chunk_files_not_counted_as_indexed(tmp_path):
    _write(tmp_path, "runbooks/a.md", "# A\n")
    pipeline = _StubPipeline(chunks_per_file=0)
    summary = await index_local_path(pipeline, tmp_path)
    assert summary["files_seen"] == 1
    assert summary["files_indexed"] == 0
    assert summary["chunks"] == 0


@pytest.mark.asyncio
async def test_missing_directory_raises(tmp_path):
    with pytest.raises(NotADirectoryError):
        await index_local_path(_StubPipeline(), tmp_path / "does-not-exist")


def test_iter_files_dedupes_overlapping_patterns(tmp_path):
    _write(tmp_path, "a.yaml", "k: v")
    # A pattern set where two globs could match the same file.
    files = _iter_files(tmp_path, ("**/*.yaml", "**/*.yaml"))
    assert len(files) == 1


def test_default_patterns_cover_expected_doc_kinds():
    joined = " ".join(DEFAULT_PATTERNS)
    for needle in ("*.md", "*.tf", "*.yaml", "Dockerfile"):
        assert needle in joined
