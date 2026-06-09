"""Unit tests for the ENV-DRIVEN resolution layer of the Kubernetes MCP.

Covers the refactor that made `opsrag/mcp/kubernetes.py` consume the unified
`environments:` registry (Approach A) instead of reading
`active_deployment().kubernetes.clusters` directly:

  - `_resolve_cluster` picks the default / explicit env, honours the
    `env` -> `cluster` alias precedence, and raises a clear error when none
    is configured.
  - `_known_clusters` / `_default_cluster` derive from the registry.
  - The shared `cluster_api_access(env)` helper resolves the right
    `K8sTarget` and returns `{host, token, verify}` for both `mode=gke`
    and `mode=kubeconfig` -- with ALL network / auth mocked (we test the
    resolution layer, not live GKE / kubeconfig auth).

`reset_environments` runs around every test so the process-global registry
never leaks between cases (the resolver is a process-global, like
`prompt_render.active_deployment`).
"""
from __future__ import annotations

import pytest

from opsrag.config import (
    EnvironmentsConfig,
    EnvironmentTarget,
    K8sTarget,
    OpsRAGConfig,
    PrometheusTarget,
)
from opsrag.environments import (
    available_environments,
    bind_environments,
    default_environment,
    reset_environments,
)
from opsrag.mcp import kubernetes as k8s


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    # Ensure no OPSRAG_K8S_DEFAULT_CLUSTER override leaks in from the env.
    monkeypatch.delenv("OPSRAG_K8S_DEFAULT_CLUSTER", raising=False)
    reset_environments()
    k8s._api_access.clear()
    yield
    reset_environments()
    k8s._api_access.clear()


def _bind_two_envs():
    """A gke env ('prod', the default) + a kubeconfig env ('staging')."""
    cfg = OpsRAGConfig()
    cfg.environments = EnvironmentsConfig(
        default="prod",
        targets={
            "prod": EnvironmentTarget(
                kubernetes=K8sTarget(
                    mode="gke", project="proj-p", location="us-east1", name="prod-gke",
                ),
                prometheus=PrometheusTarget(reach="k8s_proxy"),
            ),
            "staging": EnvironmentTarget(
                kubernetes=K8sTarget(mode="kubeconfig", context="stg-ctx"),
            ),
        },
    )
    bind_environments(cfg)


# --- resolution layer ----------------------------------------------------


def test_available_and_default_come_from_registry():
    _bind_two_envs()
    assert k8s._known_clusters() == ["prod", "staging"]
    assert available_environments() == ["prod", "staging"]
    assert k8s._default_cluster() == "prod"
    assert default_environment() == "prod"


def test_resolve_cluster_picks_default_when_no_arg():
    _bind_two_envs()
    assert k8s._resolve_cluster({}) == "prod"


def test_resolve_cluster_explicit_env_wins():
    _bind_two_envs()
    assert k8s._resolve_cluster({"env": "staging"}) == "staging"


def test_resolve_cluster_cluster_alias_still_works():
    _bind_two_envs()
    # Back-compat: callers/alerts that pass `cluster` keep working.
    assert k8s._resolve_cluster({"cluster": "staging"}) == "staging"


def test_resolve_cluster_env_takes_precedence_over_cluster_alias():
    _bind_two_envs()
    assert k8s._resolve_cluster({"env": "prod", "cluster": "staging"}) == "prod"


def test_resolve_cluster_env_var_override(monkeypatch):
    # OPSRAG_K8S_DEFAULT_CLUSTER overrides the registry default when no arg.
    _bind_two_envs()
    monkeypatch.setenv("OPSRAG_K8S_DEFAULT_CLUSTER", "staging")
    assert k8s._resolve_cluster({}) == "staging"


def test_resolve_cluster_unknown_env_is_returned_then_fails_at_use():
    # `_resolve_cluster` only picks a NAME; an unknown name surfaces as an
    # EnvironmentResolutionError when the connection layer resolves it.
    _bind_two_envs()
    name = k8s._resolve_cluster({"env": "nope"})
    assert name == "nope"


def test_resolve_cluster_empty_registry_raises():
    cfg = OpsRAGConfig()
    bind_environments(cfg)  # no environments -> empty registry
    assert available_environments() == []
    with pytest.raises(RuntimeError) as ei:
        k8s._resolve_cluster({})
    # The error mentions the environments: block (the new config surface).
    assert "environments:" in str(ei.value)


# --- shared helper: cluster_api_access (network mocked) ------------------


@pytest.mark.asyncio
async def test_cluster_api_access_gke_branch(monkeypatch):
    """gke mode -> uses _gke_get_cluster + _get_adc_token + a CA file path.
    No real GCP Container API call, no ADC, no kubernetes_asyncio import."""
    _bind_two_envs()
    monkeypatch.setattr(k8s, "_ensure_imports", lambda: None)

    async def _fake_gke(project, location, name):
        assert (project, location, name) == ("proj-p", "us-east1", "prod-gke")
        return {"endpoint": "1.2.3.4", "ca_cert_b64": "ZHVtbXk="}  # "dummy"

    async def _fake_token():
        return "tok-123"

    monkeypatch.setattr(k8s, "_gke_get_cluster", _fake_gke)
    monkeypatch.setattr(k8s, "_get_adc_token", _fake_token)
    monkeypatch.setattr(k8s, "_materialize_ca_cert", lambda key, b64: f"/tmp/{key}.crt")

    access = await k8s.cluster_api_access("prod")
    assert access == {
        "host": "https://1.2.3.4",
        "token": "tok-123",
        "verify": "/tmp/prod.crt",
    }
    # Cached by env.
    assert k8s._api_access["prod"] is access


@pytest.mark.asyncio
async def test_cluster_api_access_kubeconfig_branch(monkeypatch):
    """kubeconfig mode -> builds a client for the target context and reads
    host/token/CA off the parsed Configuration. The k8s_config client
    builder is faked so no kubeconfig file / network is touched."""
    _bind_two_envs()
    monkeypatch.setattr(k8s, "_ensure_imports", lambda: None)

    class _FakeCfg:
        host = "https://stg.example.com/"
        api_key = {"BearerToken": "Bearer stg-tok"}
        ssl_ca_cert = "/etc/ssl/stg-ca.crt"

    class _FakeApiClient:
        configuration = _FakeCfg()

        async def close(self):
            return None

    captured = {}

    async def _fake_new_client(config_file=None, context=None):
        captured["context"] = context
        return _FakeApiClient()

    # `k8s_config` is the deferred module attr; install a stub with the
    # one function cluster_api_access calls.
    class _FakeK8sConfig:
        new_client_from_config = staticmethod(_fake_new_client)

    monkeypatch.setattr(k8s, "k8s_config", _FakeK8sConfig)

    access = await k8s.cluster_api_access("staging")
    assert captured["context"] == "stg-ctx"
    assert access == {
        "host": "https://stg.example.com",  # trailing slash stripped
        "token": "stg-tok",                 # "Bearer " prefix stripped
        "verify": "/etc/ssl/stg-ca.crt",
    }


@pytest.mark.asyncio
async def test_cluster_api_access_unknown_env_raises(monkeypatch):
    _bind_two_envs()
    monkeypatch.setattr(k8s, "_ensure_imports", lambda: None)
    from opsrag.environments import EnvironmentResolutionError

    with pytest.raises(EnvironmentResolutionError):
        await k8s.cluster_api_access("does-not-exist")


@pytest.mark.asyncio
async def test_cluster_api_access_none_uses_default(monkeypatch):
    """env=None resolves to the registry default ('prod', a gke env)."""
    _bind_two_envs()
    monkeypatch.setattr(k8s, "_ensure_imports", lambda: None)

    async def _fake_gke(project, location, name):
        return {"endpoint": "9.9.9.9", "ca_cert_b64": "ZHVtbXk="}

    async def _fake_token():
        return "tok-default"

    monkeypatch.setattr(k8s, "_gke_get_cluster", _fake_gke)
    monkeypatch.setattr(k8s, "_get_adc_token", _fake_token)
    monkeypatch.setattr(k8s, "_materialize_ca_cert", lambda key, b64: f"/tmp/{key}.crt")

    access = await k8s.cluster_api_access(None)
    assert access["host"] == "https://9.9.9.9"
    assert "prod" in k8s._api_access


def test_invalidate_env_api_cache_clears_targeted_and_all():
    _bind_two_envs()
    k8s._api_access["prod"] = {"host": "h"}
    k8s._api_access["staging"] = {"host": "h2"}
    k8s.invalidate_env_api_cache("prod")
    assert "prod" not in k8s._api_access and "staging" in k8s._api_access
    k8s.invalidate_env_api_cache()  # None -> clear all
    assert k8s._api_access == {}


def test_register_clusters_is_back_compat_noop_for_resolution():
    # The shim still records legacy coords (old imports/tests) but does NOT
    # feed env resolution -- the registry stays empty until bind_environments.
    reset_environments()
    k8s.register_clusters({"prod": {"project": "p", "location": "l", "name": "n"}})
    assert available_environments() == []  # resolution unaffected
    assert k8s._cluster_coords["prod"]["name"] == "n"  # legacy dict populated
