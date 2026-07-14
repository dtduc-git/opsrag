# tests/test_external_mcp_config.py
import pytest
from pydantic import ValidationError

from opsrag.config import Settings


def test_external_mcp_parses_on_its_own_field():
    s = Settings.model_validate({
        "external_mcp": {
            "sentry_mcp": {
                "enabled": True,
                "restricted": False,
                "transport": "streamable_http",
                "url": "https://mcp.sentry.dev/mcp",
                "auth_env": "SENTRY_MCP_TOKEN",
                "auth_scheme": "Sentry-Bearer",
                "read_only": True,
                "tool_allowlist": ["find_projects", "search_events"],
                "tool_denylist": ["execute_sentry_tool"],
                "system_prompt": "Sentry read-only.",
            }
        }
    })
    blk = s.external_mcp["sentry_mcp"]
    assert blk.enabled is True and blk.restricted is False
    assert blk.url.endswith("/mcp") and blk.auth_scheme == "Sentry-Bearer"
    assert blk.tool_allowlist == ["find_projects", "search_events"]


def test_external_mcp_rejected_under_mcp_key():
    with pytest.raises(ValidationError):
        Settings.model_validate({"mcp": {"external_servers": {"url": "x"}}})


def test_external_mcp_name_colliding_with_native_connector_rejected():
    with pytest.raises(ValidationError):
        Settings.model_validate({"external_mcp": {"datadog": {"url": "x"}}})


def test_external_mcp_non_colliding_name_accepted():
    s = Settings.model_validate({"external_mcp": {"sentry_mcp": {"url": "x"}}})
    assert "sentry_mcp" in s.external_mcp
