"""Unit tests for the GitHub MCP tools against the offline fake backend.

Exercises every tool through build_fake() with no network and no
GITHUB_TOKEN, asserting shape-faithful parsed responses. The fake swaps
the module-level `_get` for a canned responder (restored on teardown).
"""
from __future__ import annotations

import pytest

from opsrag.mcp.github import GITHUB_TOOLS, build_fake, get_tool

EXPECTED_TOOLS = {
    "github_get_file_contents",
    "github_get_repository_tree",
    "github_search_code",
    "github_list_commits",
    "github_get_commit",
    "github_list_pull_requests",
    "github_get_pull_request",
    "github_list_issues",
    "github_search_issues",
    "github_list_workflow_runs",
    "github_get_workflow_run",
    "github_get_job_logs",
    "github_list_releases",
}


@pytest.fixture
def fake():
    f = build_fake()
    try:
        yield f
    finally:
        f.close()  # restore the real module-level _get


def test_tool_set_matches_exactly(fake) -> None:
    assert set(t.name for t in GITHUB_TOOLS) == EXPECTED_TOOLS
    assert set(fake.tool_names()) == EXPECTED_TOOLS


@pytest.mark.asyncio
async def test_get_file_contents_decodes_base64(fake) -> None:
    res = await fake.call(
        "github_get_file_contents",
        {"owner": "octo", "repo": "api", "path": "src/app.py", "ref": "main"},
    )
    assert res["type"] == "file"
    assert res["path"] == "src/app.py"
    assert res["content"] == "print('hello world')\n"
    assert res["is_binary"] is False


@pytest.mark.asyncio
async def test_get_file_contents_directory_listing(fake) -> None:
    res = await fake.call(
        "github_get_file_contents",
        {"owner": "octo", "repo": "api", "path": "src"},
    )
    assert res["type"] == "dir"
    assert res["count"] == 2
    assert {e["name"] for e in res["entries"]} == {"app.py", "util.py"}


@pytest.mark.asyncio
async def test_get_repository_tree(fake) -> None:
    res = await fake.call(
        "github_get_repository_tree", {"owner": "octo", "repo": "api"}
    )
    assert res["tree_sha"] == "treesha"
    assert res["count"] == 3
    paths = {t["path"] for t in res["tree"]}
    assert "src/app.py" in paths


@pytest.mark.asyncio
async def test_search_code(fake) -> None:
    res = await fake.call("github_search_code", {"q": "TODO repo:octo/api"})
    assert res["total_count"] == 1
    assert res["items"][0]["repository"] == "octo/api"
    assert res["items"][0]["path"] == "src/app.py"


@pytest.mark.asyncio
async def test_list_commits(fake) -> None:
    res = await fake.call("github_list_commits", {"owner": "octo", "repo": "api"})
    assert res["count"] == 1
    c = res["commits"][0]
    assert c["sha"] == "deadbeef"
    assert c["author"] == "Octo Cat"
    assert c["login"] == "octocat"


@pytest.mark.asyncio
async def test_get_commit(fake) -> None:
    res = await fake.call(
        "github_get_commit", {"owner": "octo", "repo": "api", "sha": "deadbeef"}
    )
    assert res["sha"] == "deadbeef"
    assert res["stats"]["additions"] == 2
    assert res["files"][0]["filename"] == "src/app.py"


@pytest.mark.asyncio
async def test_list_pull_requests(fake) -> None:
    res = await fake.call(
        "github_list_pull_requests", {"owner": "octo", "repo": "api", "state": "open"}
    )
    assert res["count"] == 1
    pr = res["pull_requests"][0]
    assert pr["number"] == 42
    assert pr["head"] == "feature"
    assert pr["base"] == "main"


@pytest.mark.asyncio
async def test_get_pull_request_includes_files(fake) -> None:
    res = await fake.call(
        "github_get_pull_request", {"owner": "octo", "repo": "api", "number": 42}
    )
    assert res["number"] == 42
    assert res["user"] == "octocat"
    assert res["files"][0]["filename"] == "src/app.py"
    assert res["files"][0]["status"] == "modified"


@pytest.mark.asyncio
async def test_list_issues(fake) -> None:
    res = await fake.call("github_list_issues", {"owner": "octo", "repo": "api"})
    assert res["count"] == 1
    issue = res["issues"][0]
    assert issue["number"] == 7
    assert issue["labels"] == ["bug"]
    assert issue["is_pull_request"] is False


@pytest.mark.asyncio
async def test_search_issues(fake) -> None:
    res = await fake.call(
        "github_search_issues", {"q": "repo:octo/api is:issue is:open"}
    )
    assert res["total_count"] == 1
    assert res["items"][0]["number"] == 7
    assert res["items"][0]["is_pull_request"] is False


@pytest.mark.asyncio
async def test_list_workflow_runs(fake) -> None:
    res = await fake.call(
        "github_list_workflow_runs", {"owner": "octo", "repo": "api", "status": "completed"}
    )
    assert res["total_count"] == 1
    run = res["workflow_runs"][0]
    assert run["id"] == 12345
    assert run["conclusion"] == "success"


@pytest.mark.asyncio
async def test_get_workflow_run_includes_jobs(fake) -> None:
    res = await fake.call(
        "github_get_workflow_run", {"owner": "octo", "repo": "api", "run_id": 12345}
    )
    assert res["id"] == 12345
    assert len(res["jobs"]) == 1
    job = res["jobs"][0]
    assert job["id"] == 9001
    assert job["conclusion"] == "success"
    assert {s["name"] for s in job["steps"]} == {"Set up job", "pytest"}


@pytest.mark.asyncio
async def test_get_job_logs(fake) -> None:
    res = await fake.call(
        "github_get_job_logs", {"owner": "octo", "repo": "api", "job_id": 9001}
    )
    assert "Job succeeded" in res["logs"]
    assert res["total_chars"] > 0


@pytest.mark.asyncio
async def test_list_releases(fake) -> None:
    res = await fake.call("github_list_releases", {"owner": "octo", "repo": "api"})
    assert res["count"] == 1
    rel = res["releases"][0]
    assert rel["tag_name"] == "v1.2.3"
    assert rel["prerelease"] is False


@pytest.mark.asyncio
async def test_unknown_tool_raises(fake) -> None:
    with pytest.raises(KeyError):
        await fake.call("github_does_not_exist", {})


def test_get_tool_lookup() -> None:
    assert get_tool("github_get_commit").name == "github_get_commit"
    with pytest.raises(KeyError):
        get_tool("nope")
