"""Contract test (T103): the chart's values.yaml mcp key set equals the
MCPIntegration registry exactly (drift in either direction fails the build).
"""
from __future__ import annotations

from pathlib import Path

import yaml

from opsrag.mcp.registry import REGISTRY

REPO_ROOT = Path(__file__).resolve().parents[2]
VALUES = REPO_ROOT / "deploy" / "helm" / "opsrag" / "values.yaml"


def test_values_mcp_keys_equal_registry() -> None:
    values = yaml.safe_load(VALUES.read_text())
    chart_mcps = set(values.get("mcp", {}))
    assert chart_mcps == set(REGISTRY), (
        f"values.yaml mcp keys drift from registry: "
        f"only-in-values={chart_mcps - set(REGISTRY)}, "
        f"only-in-registry={set(REGISTRY) - chart_mcps}"
    )


def test_every_mcp_defaults_disabled() -> None:
    values = yaml.safe_load(VALUES.read_text())
    for name, block in values["mcp"].items():
        assert block.get("enabled") is False, f"mcp.{name} must default to disabled"


def test_schema_required_mcps_match_registry() -> None:
    import json

    schema = json.loads((VALUES.parent / "values.schema.json").read_text())
    required = set(schema["properties"]["mcp"]["required"])
    assert required == set(REGISTRY)
