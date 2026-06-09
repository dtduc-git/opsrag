"""Unit tests for the ENV-DRIVEN resolution layer of the Prometheus MCP.

Covers the refactor that made `opsrag/mcp/prometheus.py` consume the unified
`environments:` registry (Approach A) instead of the hardcoded
`DEFAULT_PROMETHEUS_SERVICE` / `PROMETHEUS_NAMESPACE` / `PROMETHEUS_PORT`
constants:

  - `_resolve_prometheus_env` picks the default / explicit env, honours the
    `env` -> `cluster` alias precedence, and raises a clear error when none
    is configured.
  - `_resolve_prometheus_target` returns the env's `PrometheusTarget` and a
    clear "prometheus not configured for env X" error when absent.
  - `_select_service` routes `istio: true` to `extra_services["istio"]`.
  - `reach=k8s_proxy`: the proxy URL's namespace / service / port / host all
    come from the resolved target + `cluster_api_access(env)` -- NOT the old
    constants. The 401 ADC-refresh-and-retry path is preserved.
  - `reach=direct`: the request hits `target.url` directly, with the optional
    bearer pulled from `target.bearer_token_env`.

ALL network is mocked: `cluster_api_access` is monkeypatched and `httpx`'s
AsyncClient is replaced with a capture stub -- no live calls. The
process-global registry is reset around every test.
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
    bind_environments,
    default_environment,
    reset_environments,
)
from opsrag.mcp import prometheus as prom


# Old hardcoded values -- the refactor must NOT reproduce these unless the
# target explicitly carries them. The custom targets below use deliberately
# different namespace/service/port so any leak is caught.
_OLD_SERVICE = "monitoring-main-prometheus"
_OLD_NAMESPACE = "monitoring"
_OLD_PORT = 9090


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.delenv("OPSRAG_PROMETHEUS_DEFAULT_CLUSTER", raising=False)
    monkeypatch.delenv("OPSRAG_K8S_DEFAULT_CLUSTER", raising=False)
    reset_environments()
    yield
    reset_environments()


# --- capture stub for httpx -------------------------------------------------


class _FakeResponse:
    def __init__(self, payload, status_code=200, text=""):
        self._payload = payload
        self.status_code = status_code
        self.text = text

    def json(self):
        return self._payload


class _FakeAsyncClient:
    """Stand-in for httpx.AsyncClient. Records every GET (url/params/headers/
    verify) into the shared `calls` list and returns a queued response. A
    list of responses lets a single test drive the 401 -> retry sequence."""

    calls: list[dict] = []
    responses: list = []

    def __init__(self, *, verify=True, timeout=None):
        self._verify = verify

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url, params=None, headers=None):
        _FakeAsyncClient.calls.append({
            "url": url,
            "params": params or {},
            "headers": headers or {},
            "verify": self._verify,
        })
        if _FakeAsyncClient.responses:
            return _FakeAsyncClient.responses.pop(0)
        return _FakeResponse({"status": "success", "data": {}})


@pytest.fixture
def fake_httpx(monkeypatch):
    """Install the capture stub as `httpx.AsyncClient` (prometheus does a
    local `import httpx` inside `_proxy_get`)."""
    import httpx

    _FakeAsyncClient.calls = []
    _FakeAsyncClient.responses = []
    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)
    return _FakeAsyncClient


# --- registries -------------------------------------------------------------


def _bind_k8s_proxy_env():
    """One env ('prod', default) reached via the cluster API service-proxy,
    with NON-default namespace/service/port + an istio extra service."""
    cfg = OpsRAGConfig()
    cfg.environments = EnvironmentsConfig(
        default="prod",
        targets={
            "prod": EnvironmentTarget(
                kubernetes=K8sTarget(mode="gke", project="p", location="l", name="n"),
                prometheus=PrometheusTarget(
                    reach="k8s_proxy",
                    namespace="observability",
                    service="prom-stack-server",
                    port=9091,
                    extra_services={"istio": "prom-stack-istio"},
                ),
            ),
        },
    )
    bind_environments(cfg)


def _bind_direct_env():
    """One env ('edge', default) reached at a direct URL with a bearer env."""
    cfg = OpsRAGConfig()
    cfg.environments = EnvironmentsConfig(
        default="edge",
        targets={
            "edge": EnvironmentTarget(
                prometheus=PrometheusTarget(
                    reach="direct",
                    url="https://prom.edge.example.com",
                    bearer_token_env="OPSRAG_TEST_PROM_BEARER",
                ),
            ),
        },
    )
    bind_environments(cfg)


# --- resolution layer -------------------------------------------------------


def test_default_env_comes_from_registry():
    _bind_k8s_proxy_env()
    assert default_environment() == "prod"
    assert prom._resolve_prometheus_env({}) == "prod"


def test_env_arg_and_cluster_alias_precedence():
    _bind_k8s_proxy_env()
    # explicit env wins; cluster is the back-compat alias.
    assert prom._resolve_prometheus_env({"env": "prod"}) == "prod"
    assert prom._resolve_prometheus_env({"cluster": "prod"}) == "prod"
    assert prom._resolve_prometheus_env({"env": "a", "cluster": "b"}) == "a"


def test_env_var_override(monkeypatch):
    _bind_k8s_proxy_env()
    monkeypatch.setenv("OPSRAG_PROMETHEUS_DEFAULT_CLUSTER", "other")
    assert prom._resolve_prometheus_env({}) == "other"


def test_empty_registry_raises():
    bind_environments(OpsRAGConfig())  # no environments
    with pytest.raises(RuntimeError) as ei:
        prom._resolve_prometheus_env({})
    assert "environments:" in str(ei.value)


def test_env_without_prometheus_target_errors():
    cfg = OpsRAGConfig()
    cfg.environments = EnvironmentsConfig(
        default="prod",
        targets={"prod": EnvironmentTarget(
            kubernetes=K8sTarget(mode="kubeconfig", context="c"),
        )},
    )
    bind_environments(cfg)
    with pytest.raises(RuntimeError) as ei:
        prom._resolve_prometheus_target("prod")
    msg = str(ei.value)
    assert "prometheus not configured for env 'prod'" in msg


def test_select_service_main_vs_istio():
    _bind_k8s_proxy_env()
    target = prom._resolve_prometheus_target("prod")
    assert prom._select_service(target, {}) == "prom-stack-server"
    assert prom._select_service(target, {"istio": True}) == "prom-stack-istio"
    # istio requested but not configured -> falls back to the main service.
    plain = PrometheusTarget(reach="k8s_proxy", service="only-main")
    assert prom._select_service(plain, {"istio": True}) == "only-main"


# --- reach=k8s_proxy: URL is built from the target, not the constants -------


@pytest.mark.asyncio
async def test_k8s_proxy_url_uses_target_namespace_service_port(fake_httpx, monkeypatch):
    _bind_k8s_proxy_env()

    async def _fake_access(env):
        assert env == "prod"
        return {"host": "https://api.prod:6443", "token": "tok-xyz", "verify": "/ca.crt"}

    monkeypatch.setattr(prom, "cluster_api_access", _fake_access)

    fake_httpx.responses = [_FakeResponse({
        "status": "success",
        "data": {"resultType": "vector", "result": []},
    })]

    result = await prom._h_query(None, {"query": "up"})

    assert result["cluster"] == "prod"
    assert result["service"] == "prom-stack-server"

    call = fake_httpx.calls[-1]
    # URL: host + the TARGET's namespace/service/port -- NOT the old constants.
    assert call["url"] == (
        "https://api.prod:6443/api/v1/namespaces/observability"
        "/services/prom-stack-server:9091/proxy/api/v1/query"
    )
    assert "observability" in call["url"] and _OLD_NAMESPACE + "/" not in call["url"]
    assert "prom-stack-server" in call["url"] and _OLD_SERVICE not in call["url"]
    assert ":9091/" in call["url"] and f":{_OLD_PORT}/" not in call["url"]
    # Auth + TLS verify come from cluster_api_access.
    assert call["headers"]["Authorization"] == "Bearer tok-xyz"
    assert call["verify"] == "/ca.crt"
    assert call["params"]["query"] == "up"


@pytest.mark.asyncio
async def test_k8s_proxy_istio_routes_to_extra_service(fake_httpx, monkeypatch):
    _bind_k8s_proxy_env()

    async def _fake_access(env):
        return {"host": "https://api.prod:6443", "token": "t", "verify": "/ca.crt"}

    monkeypatch.setattr(prom, "cluster_api_access", _fake_access)
    fake_httpx.responses = [_FakeResponse({
        "status": "success", "data": {"resultType": "vector", "result": []},
    })]

    result = await prom._h_query(None, {"query": "up", "istio": True})
    assert result["service"] == "prom-stack-istio"
    assert "/services/prom-stack-istio:9091/" in fake_httpx.calls[-1]["url"]


@pytest.mark.asyncio
async def test_k8s_proxy_401_refreshes_and_retries(fake_httpx, monkeypatch):
    """401 -> refresh_adc_token + invalidate_env_api_cache + retry once."""
    _bind_k8s_proxy_env()

    async def _fake_access(env):
        return {"host": "https://api.prod:6443", "token": "t", "verify": "/ca.crt"}

    refreshed = {"count": 0}
    invalidated = []

    async def _fake_refresh():
        refreshed["count"] += 1

    def _fake_invalidate(env=None):
        invalidated.append(env)

    monkeypatch.setattr(prom, "cluster_api_access", _fake_access)
    monkeypatch.setattr(prom, "refresh_adc_token", _fake_refresh)
    monkeypatch.setattr(prom, "invalidate_env_api_cache", _fake_invalidate)

    fake_httpx.responses = [
        _FakeResponse({}, status_code=401, text="unauthorized"),
        _FakeResponse({"status": "success", "data": {"resultType": "vector", "result": []}}),
    ]

    result = await prom._h_query(None, {"query": "up"})

    assert refreshed["count"] == 1
    assert invalidated == ["prod"]
    assert len(fake_httpx.calls) == 2  # original + one retry
    assert result["cluster"] == "prod"
    assert "error" not in result


# --- reach=direct: URL is target.url, optional bearer from env --------------


@pytest.mark.asyncio
async def test_direct_reach_hits_target_url_with_bearer(fake_httpx, monkeypatch):
    _bind_direct_env()
    monkeypatch.setenv("OPSRAG_TEST_PROM_BEARER", "edge-secret")
    # cluster_api_access must NEVER be called for direct reach.
    async def _boom(env):
        raise AssertionError("cluster_api_access must not be called for reach=direct")
    monkeypatch.setattr(prom, "cluster_api_access", _boom)

    fake_httpx.responses = [_FakeResponse({
        "status": "success", "data": {"resultType": "vector", "result": []},
    })]

    result = await prom._h_query(None, {"query": "up"})

    call = fake_httpx.calls[-1]
    assert call["url"] == "https://prom.edge.example.com/api/v1/query"
    assert call["headers"]["Authorization"] == "Bearer edge-secret"
    assert call["params"]["query"] == "up"
    assert result["cluster"] == "edge"


@pytest.mark.asyncio
async def test_direct_reach_no_bearer_when_env_unset(fake_httpx, monkeypatch):
    _bind_direct_env()
    monkeypatch.delenv("OPSRAG_TEST_PROM_BEARER", raising=False)
    monkeypatch.setattr(prom, "cluster_api_access", None)  # unused

    fake_httpx.responses = [_FakeResponse({
        "status": "success", "data": {"resultType": "vector", "result": []},
    })]

    await prom._h_query(None, {"query": "up"})
    assert "Authorization" not in fake_httpx.calls[-1]["headers"]


@pytest.mark.asyncio
async def test_direct_reach_missing_url_is_structured_error(fake_httpx, monkeypatch):
    cfg = OpsRAGConfig()
    cfg.environments = EnvironmentsConfig(
        default="edge",
        targets={"edge": EnvironmentTarget(
            prometheus=PrometheusTarget(reach="direct", url=None),
        )},
    )
    bind_environments(cfg)
    monkeypatch.setattr(prom, "cluster_api_access", None)

    result = await prom._h_query(None, {"query": "up"})
    # _h_query maps a non-success proxy payload into {cluster, error, ...}.
    assert result["cluster"] == "edge"
    assert "no `url`" in result["error"]
    assert fake_httpx.calls == []  # never issued a request


# --- tool schema documents env + keeps cluster alias ------------------------


def test_tool_schema_exposes_env_and_cluster_alias():
    tool = prom.get_tool("prometheus_query")
    props = tool.input_schema["properties"]
    assert "env" in props
    assert "cluster" in props  # back-compat alias retained
