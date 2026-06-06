"""Integration test (T080): the KNOWLEDGE MCP tool against the fake backend.

Exercises knowledge_search through build_fake() with no embeddings, no
vector store, and no network, asserting shape-faithful canned hits and
the registry's declared tool set. Follows the GitLab reference
(tests/integration/test_gitlab_mcp.py) for the per-MCP fake pattern
(FR-012).
"""
from __future__ import annotations

import pytest

from opsrag.mcp.knowledge import build_fake
from opsrag.mcp.registry import REGISTRY


@pytest.fixture
def fake():
    f = build_fake()
    yield f
    f.close()  # restore prior module state (unbind fake providers)


def test_fake_exposes_registry_tool_set(fake) -> None:
    # The fake's tools must match exactly what the registry declares.
    assert set(fake.tool_names()) == set(REGISTRY["knowledge"].tool_names)


@pytest.mark.asyncio
async def test_knowledge_search_returns_canned_hits(fake) -> None:
    result = await fake.call("knowledge_search", {"query": "SRE access requests"})
    assert result["query"] == "SRE access requests"
    assert result["count"] >= 1
    hits = result["results"]
    assert isinstance(hits, list) and hits

    top = hits[0]
    # Shape-faithful fields the handler always emits.
    assert top["source"] == "kb/sre/access-requests.md"
    assert top["repo"] == "sre-kb"
    assert top["priority"] == "user-correction"
    assert isinstance(top["score"], float) and top["score"] > 0.0
    assert "access" in top["content"].lower()
    # url / title surfaced from metadata.
    assert top["url"] == "https://docs.example.com/access-requests"
    assert top["title"] == "Access Requests"


@pytest.mark.asyncio
async def test_knowledge_search_honours_k(fake) -> None:
    result = await fake.call("knowledge_search", {"query": "runbook", "k": 2})
    assert result["count"] == 2
    assert len(result["results"]) == 2
