"""Unit tests for the unified multi-environment registry resolver."""
from __future__ import annotations

import pytest

from opsrag.config import (
    ElasticsearchConfig,
    EnvironmentsConfig,
    EnvironmentTarget,
    EsTarget,
    K8sClusterCoords,
    K8sConfig,
    K8sTarget,
    OpsRAGConfig,
    PrometheusTarget,
)
from opsrag.environments import (
    EnvironmentResolutionError,
    available_environments,
    bind_environments,
    default_environment,
    reset_environments,
    resolve_environment,
)


@pytest.fixture(autouse=True)
def _reset():
    reset_environments()
    yield
    reset_environments()


def test_explicit_registry_binds_and_resolves():
    cfg = OpsRAGConfig()
    cfg.environments = EnvironmentsConfig(
        default="prod",
        targets={
            "prod": EnvironmentTarget(
                kubernetes=K8sTarget(mode="gke", project="p", location="l", name="n"),
                prometheus=PrometheusTarget(reach="direct", url="http://prom"),
                elasticsearch=EsTarget(reach="direct", url="http://es"),
            ),
            "staging": EnvironmentTarget(
                kubernetes=K8sTarget(mode="kubeconfig", context="stg-ctx"),
            ),
        },
    )
    bind_environments(cfg)
    assert available_environments() == ["prod", "staging"]
    assert default_environment() == "prod"
    # default resolves to prod
    assert resolve_environment().kubernetes.name == "n"
    assert resolve_environment("staging").kubernetes.context == "stg-ctx"
    assert resolve_environment("prod").prometheus.url == "http://prom"


def test_unknown_env_raises():
    cfg = OpsRAGConfig()
    cfg.environments = EnvironmentsConfig(targets={"prod": EnvironmentTarget()})
    bind_environments(cfg)
    with pytest.raises(EnvironmentResolutionError):
        resolve_environment("nope")


def test_empty_registry_raises():
    cfg = OpsRAGConfig()
    bind_environments(cfg)
    assert available_environments() == []
    with pytest.raises(EnvironmentResolutionError):
        resolve_environment()


def test_legacy_synthesis_from_k8s_clusters_and_es():
    cfg = OpsRAGConfig()
    cfg.k8s = K8sConfig(
        default_cluster="prod",
        clusters={"prod": K8sClusterCoords(project="pp", location="us-east1", name="prod-gke")},
    )
    cfg.elasticsearch = ElasticsearchConfig(
        enabled=True, url="https://es.prod:9200", api_key_env="ES_KEY", default_index="app-*",
    )
    bind_environments(cfg)
    assert available_environments() == ["prod"]
    assert default_environment() == "prod"
    t = resolve_environment("prod")
    assert t.kubernetes.mode == "gke" and t.kubernetes.name == "prod-gke"
    # legacy prometheus preserves the old hardcoded service for behavior parity
    assert t.prometheus.service == "monitoring-main-prometheus"
    assert t.elasticsearch.url == "https://es.prod:9200"
    assert t.elasticsearch.index_pattern == "app-*"


def test_legacy_es_only_creates_default_env():
    cfg = OpsRAGConfig()
    cfg.elasticsearch = ElasticsearchConfig(enabled=True, url="https://es:9200")
    bind_environments(cfg)
    assert available_environments() == ["default"]
    assert resolve_environment().elasticsearch.url == "https://es:9200"
