"""Contract test (T104): rendering with an MCP enabled puts the expected
OPSRAG_MCP_<NAME>_ENABLED=true env var on the api container, and the others
stay false. Asserts the values->container wiring mechanically (no template
scanning). Skips cleanly if `helm` is not installed.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from opsrag.mcp.registry import REGISTRY

REPO_ROOT = Path(__file__).resolve().parents[2]
CHART = REPO_ROOT / "deploy" / "helm" / "opsrag"

pytestmark = pytest.mark.skipif(shutil.which("helm") is None, reason="helm not installed")


def _render(*set_args: str) -> list[dict]:
    cmd = ["helm", "template", "opsrag", str(CHART)]
    for s in set_args:
        cmd += ["--set", s]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    return [d for d in yaml.safe_load_all(out) if d]


def _api_env(manifests: list[dict]) -> dict[str, str]:
    for m in manifests:
        if m.get("kind") == "Deployment" and m["metadata"]["name"].endswith("opsrag"):
            for c in m["spec"]["template"]["spec"]["containers"]:
                if c["name"] == "api":
                    return {e["name"]: e.get("value") for e in c.get("env", [])}
    raise AssertionError("api container not found in rendered Deployment")


def test_enabling_one_mcp_sets_its_env_true() -> None:
    env = _api_env(_render("mcp.gitlab.enabled=true"))
    assert env["OPSRAG_MCP_GITLAB_ENABLED"] == "true"
    # A different integration stays false.
    assert env["OPSRAG_MCP_DATADOG_ENABLED"] == "false"


def test_default_render_all_mcp_envs_false() -> None:
    env = _api_env(_render())
    for name in REGISTRY:
        key = f"OPSRAG_MCP_{name.upper()}_ENABLED"
        assert env[key] == "false", f"{key} should default false"


def test_every_registry_mcp_has_an_env_var() -> None:
    env = _api_env(_render())
    for name in REGISTRY:
        assert f"OPSRAG_MCP_{name.upper()}_ENABLED" in env
