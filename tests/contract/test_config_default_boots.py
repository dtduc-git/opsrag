"""Contract test (T041): the shipped config.yaml validates and yields the
minimal-deployment shape promised by the quickstart - every MCP disabled and
the null graph backend.

Mirrors contracts/config-schema.md. Note: the contract doc spells the null
graph provider as ``null``; the implementation's enum value is ``none`` (the
factory's NullGraphStore branch keys on ``provider == "none"``). This test
asserts the implemented value, which is the source of truth (FR-019).
"""
from __future__ import annotations

from pathlib import Path

from opsrag.config import Settings
from opsrag.mcp.registry import REGISTRY

# The known MCP integrations (general open-source catalog). Kept in lockstep
# with KNOWN_MCP_NAMES / MCP_CONFIG_TYPES / the registry.
EXPECTED_MCP_NAMES = {
    "aws", "azure", "cloudflare", "cloudwatch", "code", "datadog",
    "elasticsearch", "gcp", "github", "gitlab", "grafana", "knowledge",
    "kubernetes", "loki", "pagerduty", "prometheus", "rootly", "runbooks",
    "sentry", "slack", "splunk", "stackdriver", "tool_cache",
}


def test_registry_has_exactly_the_known_mcps() -> None:
    assert set(REGISTRY) == EXPECTED_MCP_NAMES


def test_default_config_validates(config_path: Path) -> None:
    # Must not raise: the shipped default is always loadable.
    settings = Settings.load(config_path)
    assert settings is not None


def test_default_config_disables_all_mcps(config_path: Path) -> None:
    settings = Settings.load(config_path)
    # Every known integration is present and disabled by default (FR-011).
    assert set(settings.mcp) == EXPECTED_MCP_NAMES
    for name, block in settings.mcp.items():
        assert block.enabled is False, f"mcp.{name} should ship disabled"


def test_default_config_uses_null_graph_backend(config_path: Path) -> None:
    settings = Settings.load(config_path)
    # "none" is the implemented spelling of the FR-019 null backend.
    assert settings.knowledge_graph.provider == "none"
