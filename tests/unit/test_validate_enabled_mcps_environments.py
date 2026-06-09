"""prometheus MCP validation accepts the `environments:` registry, not only
legacy KUBECONFIG / k8s.clusters."""
from __future__ import annotations

import pytest

from opsrag.config import (
    EnvironmentsConfig,
    EnvironmentTarget,
    K8sClusterCoords,
    K8sConfig,
    K8sTarget,
    OpsRAGConfig,
    PrometheusTarget,
)
from opsrag.mcp.registry import MCPMisconfigured, validate_enabled_mcps


def _cfg_prom_enabled() -> OpsRAGConfig:
    cfg = OpsRAGConfig()
    cfg.mcp["prometheus"].enabled = True
    return cfg


def test_prometheus_valid_with_environments_prometheus_target():
    cfg = _cfg_prom_enabled()
    cfg.environments = EnvironmentsConfig(
        targets={"prod": EnvironmentTarget(
            prometheus=PrometheusTarget(reach="direct", url="http://prom"))}
    )
    validate_enabled_mcps(cfg, env={})  # no KUBECONFIG, no legacy clusters -> valid


def test_prometheus_valid_with_environments_kubernetes_target():
    cfg = _cfg_prom_enabled()
    cfg.environments = EnvironmentsConfig(
        targets={"prod": EnvironmentTarget(
            kubernetes=K8sTarget(mode="kubeconfig", context="c"))}
    )
    validate_enabled_mcps(cfg, env={})  # k8s_proxy prometheus rides the k8s target


def test_prometheus_valid_with_legacy_k8s_clusters():
    cfg = _cfg_prom_enabled()
    cfg.k8s = K8sConfig(clusters={"prod": K8sClusterCoords(project="p", location="l", name="n")})
    validate_enabled_mcps(cfg, env={})


def test_prometheus_valid_with_kubeconfig_env():
    cfg = _cfg_prom_enabled()
    validate_enabled_mcps(cfg, env={"KUBECONFIG": "/x"})


def test_prometheus_invalid_with_nothing():
    cfg = _cfg_prom_enabled()
    with pytest.raises(MCPMisconfigured):
        validate_enabled_mcps(cfg, env={})
