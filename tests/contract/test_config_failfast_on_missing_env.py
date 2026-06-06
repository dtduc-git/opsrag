"""Contract test (T072): enabling an MCP without its required env/config
fails fast with the canonical MCP_MISCONFIGURED:<name>:<missing> error
(FR-004). Parametrised over all 14 registered integrations.
"""
from __future__ import annotations

import pytest

from opsrag.config import Settings
from opsrag.mcp.registry import (
    REGISTRY,
    MCPMisconfigured,
    validate_enabled_mcps,
)

# Integrations that require at least one env var or config key -> enabling
# them with a scrubbed environment must raise.
GATED = sorted(
    name
    for name, integ in REGISTRY.items()
    if integ.required_env or integ.required_config
)
# Integrations with no required env/config -> enabling them is always valid.
UNGATED = sorted(
    name
    for name, integ in REGISTRY.items()
    if not integ.required_env and not integ.required_config
)


@pytest.mark.parametrize("name", GATED)
def test_enable_without_requirements_fails_fast(name: str) -> None:
    settings = Settings.model_validate({"mcp": {name: {"enabled": True}}})
    # Empty env so no required var is satisfied; empty deployment context so
    # no required-config path resolves.
    with pytest.raises(MCPMisconfigured) as exc_info:
        validate_enabled_mcps(settings, env={})
    msg = str(exc_info.value)
    assert msg.startswith(f"MCP_MISCONFIGURED:{name}:")
    # The missing item is one of the integration's declared requirements.
    integ = REGISTRY[name]
    assert exc_info.value.missing in (integ.required_env + integ.required_config)


@pytest.mark.parametrize("name", UNGATED)
def test_enable_without_requirements_ok_when_ungated(name: str) -> None:
    settings = Settings.model_validate({"mcp": {name: {"enabled": True}}})
    # Should not raise: these integrations declare no required env/config.
    validate_enabled_mcps(settings, env={})


def test_satisfied_env_passes() -> None:
    # Datadog needs DD_API_KEY + DD_APP_KEY; providing them passes.
    settings = Settings.model_validate({"mcp": {"datadog": {"enabled": True}}})
    validate_enabled_mcps(settings, env={"DD_API_KEY": "x", "DD_APP_KEY": "y"})


def test_disabled_mcp_never_validated() -> None:
    # A disabled integration is never checked even with an empty env.
    settings = Settings.model_validate({"mcp": {"datadog": {"enabled": False}}})
    validate_enabled_mcps(settings, env={})
