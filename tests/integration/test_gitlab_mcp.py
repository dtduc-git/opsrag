"""Integration test (T079): the GitLab MCP tools against the fake backend.

Exercises representative tools through build_fake() with no network, asserting
shape-faithful responses and the registry's declared tool set. This is the
reference test for the per-MCP fake pattern (FR-012).
"""
from __future__ import annotations

import pytest

from opsrag.mcp.gitlab import build_fake
from opsrag.mcp.registry import REGISTRY


@pytest.fixture
def fake():
    return build_fake()


def test_fake_exposes_registry_tool_set(fake) -> None:
    # The fake's tools must match exactly what the registry declares.
    assert set(fake.tool_names()) == set(REGISTRY["gitlab"].tool_names)


@pytest.mark.asyncio
async def test_list_pipelines(fake) -> None:
    result = await fake.call("gitlab_list_pipelines", {"project_id": "group/project"})
    assert isinstance(result, list) and result
    assert result[0]["status"] == "success"


@pytest.mark.asyncio
async def test_get_merge_request(fake) -> None:
    result = await fake.call(
        "gitlab_get_merge_request", {"project_id": "group/project", "merge_request_iid": 7}
    )
    assert result["iid"] == 7
    assert result["state"] == "merged"


@pytest.mark.asyncio
async def test_get_project(fake) -> None:
    result = await fake.call("gitlab_get_project", {"project_id": "group/project"})
    assert result["default_branch"] == "main"


@pytest.mark.asyncio
async def test_unknown_tool_raises(fake) -> None:
    with pytest.raises(KeyError):
        await fake.call("gitlab_does_not_exist", {})
