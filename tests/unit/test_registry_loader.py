"""Tests for MCP enabled-integration gating (T087/T091)."""
from __future__ import annotations

import pytest

from opsrag.config import Settings
from opsrag.mcp.registry import REGISTRY
from opsrag.mcp_server import registry_loader as rl


@pytest.fixture(autouse=True)
def _reset():
    rl.set_active_enabled(None)
    yield
    rl.set_active_enabled(None)


def test_enabled_integration_names() -> None:
    settings = Settings.model_validate(
        {"mcp": {"gitlab": {"enabled": True}, "datadog": {"enabled": False}}}
    )
    assert rl.enabled_integration_names(settings) == ("gitlab",)


def test_enabled_tool_names_unions_registry() -> None:
    settings = Settings.model_validate({"mcp": {"gitlab": {"enabled": True}}})
    assert rl.enabled_tool_names(settings) == frozenset(REGISTRY["gitlab"].tool_names)


def test_filter_off_by_default_returns_all() -> None:
    # No setter called -> gating disabled -> superset unchanged.
    tools = [type("T", (), {"name": n})() for n in ("gitlab_get_project", "k8s_get_pod")]
    assert rl.filter_enabled(tools) == tools


def test_filter_to_enabled_only() -> None:
    rl.set_active_enabled(["gitlab"])
    tools = [
        type("T", (), {"name": n})()
        for n in ("gitlab_get_project", "k8s_get_pod", "datadog_get_trace")
    ]
    kept = {t.name for t in rl.filter_enabled(tools)}
    assert "gitlab_get_project" in kept
    assert "k8s_get_pod" not in kept
    assert "datadog_get_trace" not in kept


def test_all_disabled_gates_everything() -> None:
    rl.set_active_enabled([])  # nothing enabled
    tools = [type("T", (), {"name": "gitlab_get_project"})()]
    assert rl.filter_enabled(tools) == []


def test_agent_registry_respects_gating() -> None:
    from opsrag.agent.nodes import multi_agent

    rl.set_active_enabled(["gitlab"])
    reg = multi_agent._registry()
    assert reg, "expected gitlab tools when gitlab enabled"
    assert all(name.startswith("gitlab_") for name in reg)

    rl.set_active_enabled([])
    assert multi_agent._registry() == {}
