"""Tests: external MCP connectors (config.external_mcp) participate in the
same enabled-gating, per-connector system-prompt binding, and RBAC
enumeration as native config.mcp connectors (External MCP Adapter Task 5)."""
from __future__ import annotations

from opsrag.api._connector_perms import _enabled_and_restricted
from opsrag.config import Settings
from opsrag.mcp_server.registry_loader import (
    connector_system_prompt,
    enabled_integration_names,
    set_connector_system_prompts,
)


def _settings():
    return Settings.model_validate({"external_mcp": {"sentry_mcp": {
        "enabled": True, "restricted": False, "url": "https://x/mcp",
        "system_prompt": "Sentry read-only.", "tool_allowlist": ["find_projects"],
    }}})


def test_external_appears_in_enabled_names():
    assert "sentry_mcp" in enabled_integration_names(_settings())


def test_external_system_prompt_bound():
    set_connector_system_prompts(_settings())
    assert connector_system_prompt("sentry_mcp") == "Sentry read-only."


def test_external_enabled_not_restricted():
    enabled, restricted = _enabled_and_restricted(_settings())
    assert "sentry_mcp" in enabled and "sentry_mcp" not in restricted
