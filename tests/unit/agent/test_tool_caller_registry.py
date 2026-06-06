"""Regression: the tool executor must be able to run every tool the reasoner
is allowed to offer -- not just GitLab.

History: `_registry()` originally returned only ``GITLAB_TOOLS``, while the
reasoner offered ``filter_enabled(ALL_MCP_TOOLS)``. So whenever the reasoner
picked a non-GitLab tool (code_*, datadog_*, rootly_*, ...), the executor
rejected it as "unknown tool" and the tool path returned a generic non-answer.
The two sides must draw from the same set.
"""
from __future__ import annotations


def test_registry_matches_reasoner_offered_set():
    """Executor registry == the reasoner's offered toolset (gating off => all)."""
    from opsrag.agent.nodes.tool_caller import _registry
    from opsrag.mcp import ALL_MCP_TOOLS
    from opsrag.mcp_server.registry_loader import filter_enabled

    reg = _registry()
    offered = {t.name for t in filter_enabled(ALL_MCP_TOOLS)} or {
        t.name for t in ALL_MCP_TOOLS
    }
    assert set(reg) == offered


def test_registry_includes_code_tools():
    """The code_* family (lazy-clone source tools) must be executable -- these
    are exactly the tools that used to be rejected as 'unknown tool'."""
    from opsrag.agent.nodes.tool_caller import _registry

    reg = _registry()
    for name in (
        "code_grep",
        "code_list_repos",
        "code_read_file",
        "code_dependency_lookup",
    ):
        assert name in reg, f"{name} not executable by the tool_caller"


def test_registry_not_gitlab_only():
    """Guard against a regression back to the GitLab-only registry."""
    from opsrag.agent.nodes.tool_caller import _registry
    from opsrag.mcp import GITLAB_TOOLS

    reg = _registry()
    gitlab_names = {t.name for t in GITLAB_TOOLS}
    assert set(reg) - gitlab_names, "registry collapsed back to GitLab-only"
