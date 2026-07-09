"""Per-connector MCP system-prompt: an operator note on config.mcp.<name>.
system_prompt is surfaced to the reasoner's tool-selection prompt (appended to
that connector's tool descriptions), so routing is configurable per deployment
instead of hardcoded (e.g. "Datadog = tracing only; logs in Elasticsearch").
"""
from __future__ import annotations

from types import SimpleNamespace

from opsrag.config_mcp import MCPConfigBlock
from opsrag.mcp_server import registry_loader as rl


def test_field_defaults_none():
    assert MCPConfigBlock().system_prompt is None


def _settings(**mcp):
    return SimpleNamespace(mcp=mcp)


def test_bind_and_lookup():
    rl.set_connector_system_prompts(_settings(
        datadog=MCPConfigBlock(enabled=True, system_prompt="  Datadog = tracing only.  "),
        elasticsearch=MCPConfigBlock(enabled=True, system_prompt="Main tool for logging."),
        gcp=MCPConfigBlock(enabled=True),  # no note
    ))
    assert rl.connector_system_prompt("datadog") == "Datadog = tracing only."  # trimmed
    assert rl.connector_system_prompt("elasticsearch") == "Main tool for logging."
    assert rl.connector_system_prompt("gcp") is None       # unset -> None
    assert rl.connector_system_prompt("unknown") is None
    assert rl.connector_system_prompt(None) is None


def test_rebind_replaces():
    rl.set_connector_system_prompts(_settings(datadog=MCPConfigBlock(system_prompt="first")))
    assert rl.connector_system_prompt("datadog") == "first"
    rl.set_connector_system_prompts(_settings())        # empty -> cleared
    assert rl.connector_system_prompt("datadog") is None


def test_appended_to_reasoner_tool_desc():
    """_tool_specs() appends the connector note to that connector's tools."""
    from opsrag.agent.nodes import multi_agent
    from opsrag.mcp_server.registry_loader import connector_for_tool

    # pick a real datadog tool name from the registry
    dd_tool = next((t for t, c in rl._TOOL_TO_CONNECTOR.items() if c == "datadog"), None)
    assert dd_tool, "expected at least one datadog tool in the registry"

    rl.set_connector_system_prompts(_settings(
        datadog=MCPConfigBlock(enabled=True, system_prompt="TRACING ONLY HERE."),
    ))
    # force datadog enabled for the tool-spec render
    rl.set_active_enabled(["datadog"])
    try:
        specs = {s["name"]: s["description"] for s in multi_agent._tool_specs_for_llm()}
    finally:
        rl.set_active_enabled(None)
    assert dd_tool in specs
    assert "TRACING ONLY HERE." in specs[dd_tool]
    assert "[Deployment note:" in specs[dd_tool]
    assert connector_for_tool(dd_tool) == "datadog"
