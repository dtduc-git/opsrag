"""Integration test (T076): the code-search MCP tools against the fake backend.

Exercises representative tools through build_fake() with no network and no real
repo. build_fake() is self-contained: it stands up a throwaway git repo in a
temp dir, points the module's cache root at it, and registers the repo in the
allowlist so the handlers' real `git grep` / `git ls-files` subprocesses run
against canned files. See FR-012 and the GitLab reference test for the pattern.
"""
from __future__ import annotations

import pytest

from opsrag.mcp.code import build_fake
from opsrag.mcp.registry import REGISTRY


@pytest.fixture
def fake():
    f = build_fake()
    try:
        yield f
    finally:
        f.close()


def test_fake_exposes_registry_tool_set(fake) -> None:
    # The fake's tools must match exactly what the registry declares.
    assert set(fake.tool_names()) == set(REGISTRY["code"].tool_names)


@pytest.mark.asyncio
async def test_list_repos(fake) -> None:
    result = await fake.call("code_list_repos", {})
    assert "group/example-repo" in result["repos"]
    assert result["count"] >= 1


@pytest.mark.asyncio
async def test_grep_known_string(fake) -> None:
    result = await fake.call(
        "code_grep", {"repo": "group/example-repo", "pattern": "handle_request"}
    )
    assert "error" not in result
    assert result["count"] >= 1
    paths = {hit["path"] for hit in result["hits"]}
    assert "src/app.py" in paths


@pytest.mark.asyncio
async def test_read_file_canned_content(fake) -> None:
    result = await fake.call(
        "code_read_file", {"repo": "group/example-repo", "path": "README.md"}
    )
    assert "error" not in result
    assert "example-repo" in result["content"]
    assert result["total_lines"] >= 1


@pytest.mark.asyncio
async def test_unknown_tool_raises(fake) -> None:
    with pytest.raises(KeyError):
        await fake.call("code_does_not_exist", {})
