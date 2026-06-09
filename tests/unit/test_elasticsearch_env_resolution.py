"""Unit tests for the ENV-DRIVEN resolution layer of the Elasticsearch MCP.

Covers the refactor that made ``opsrag/mcp/elasticsearch.py`` consume the
unified ``environments:`` registry (Approach A) instead of a single global
``_BOUND`` endpoint:

  - ``_config(env)`` returns the right PER-ENV ``EsTarget`` bundle (url / index
    / backend / verify_ssl) and applies the env's ``fields`` mapping.
  - ``env`` arg selects the env; back-compat ``cluster`` alias works; omitting
    both -> the registry default env.
  - ``reach=direct`` builds the base URL + API-key header; an env with no ES
    target / no URL surfaces a clean structured error.
  - ``reach=port_forward`` / ``proxy`` reach the in-cluster service THROUGH the
    k8s API server (``cluster_api_access`` mocked -- no live cluster).
  - End-to-end handler calls go through a MOCKED httpx (no live ES).

``reset_environments`` runs around every test so the process-global registry
never leaks between cases.
"""
from __future__ import annotations

import pytest

from opsrag.config import (
    EnvironmentsConfig,
    EnvironmentTarget,
    EsTarget,
    OpsRAGConfig,
)
from opsrag.environments import (
    EnvironmentResolutionError,
    available_environments,
    bind_environments,
    default_environment,
    reset_environments,
)
from opsrag.mcp import elasticsearch as es


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    # No stray ES credential env vars leaking in from the host.
    for var in ("PROD_ES_KEY", "STG_ES_USER", "STG_ES_PASS"):
        monkeypatch.delenv(var, raising=False)
    reset_environments()
    yield
    reset_environments()


def _bind_two_envs():
    """Two ES envs with DIFFERENT url / index / fields / backend.

    - prod: direct ES, API-key auth, ECK-style field mapping.
    - staging: direct OpenSearch, basic auth, generic (empty) field mapping.
    """
    cfg = OpsRAGConfig()
    cfg.environments = EnvironmentsConfig(
        default="prod",
        targets={
            "prod": EnvironmentTarget(
                elasticsearch=EsTarget(
                    reach="direct",
                    url="https://es.prod.example.com:9200",
                    api_key_env="PROD_ES_KEY",
                    index_pattern="app-logs-*",
                    backend="elasticsearch",
                    verify_ssl=True,
                    fields={
                        "timestamp": "@timestamp",
                        "service": "kubernetes.labels.app_name",
                    },
                ),
            ),
            "staging": EnvironmentTarget(
                elasticsearch=EsTarget(
                    reach="direct",
                    url="http://os.staging.svc:9200",
                    username_env="STG_ES_USER",
                    password_env="STG_ES_PASS",
                    index_pattern="staging-logs-*",
                    backend="opensearch",
                    verify_ssl=False,
                ),
            ),
        },
    )
    bind_environments(cfg)


# --- resolution layer ----------------------------------------------------


def test_registry_default_and_available():
    _bind_two_envs()
    assert available_environments() == ["prod", "staging"]
    assert default_environment() == "prod"


def test_config_default_env_when_no_arg(monkeypatch):
    _bind_two_envs()
    monkeypatch.setenv("PROD_ES_KEY", "secret-prod-key")
    cfg = es._config(None)  # default -> prod
    assert cfg["url"] == "https://es.prod.example.com:9200"
    assert cfg["default_index"] == "app-logs-*"
    assert cfg["backend"] == "elasticsearch"
    assert cfg["verify_ssl"] is True
    assert cfg["headers"]["Authorization"] == "ApiKey secret-prod-key"
    assert cfg["auth"] is None


def test_config_per_env_returns_right_target(monkeypatch):
    _bind_two_envs()
    monkeypatch.setenv("STG_ES_USER", "stg-user")
    monkeypatch.setenv("STG_ES_PASS", "stg-pass")
    cfg = es._config("staging")
    # Different url / index / backend / TLS than prod.
    assert cfg["url"] == "http://os.staging.svc:9200"
    assert cfg["default_index"] == "staging-logs-*"
    assert cfg["backend"] == "opensearch"
    assert cfg["verify_ssl"] is False
    # Basic auth (not API key) for this env.
    assert cfg["auth"] == ("stg-user", "stg-pass")
    assert "Authorization" not in cfg["headers"]


def test_field_mapping_applied_per_env():
    _bind_two_envs()
    prod = es._config("prod")
    # prod maps logical -> physical ECK fields.
    assert es._map_field(prod, "timestamp") == "@timestamp"
    assert es._map_field(prod, "service") == "kubernetes.labels.app_name"
    # unmapped logical name passes through unchanged.
    assert es._map_field(prod, "level") == "level"
    # staging has no field mapping -> everything passes through.
    stg = es._config("staging")
    assert es._map_field(stg, "timestamp") == "timestamp"
    assert es._map_field(stg, "service") == "service"


def test_unknown_env_raises_resolution_error():
    _bind_two_envs()
    with pytest.raises(EnvironmentResolutionError):
        es._config("does-not-exist")


def test_empty_registry_raises_resolution_error():
    bind_environments(OpsRAGConfig())  # no environments
    assert available_environments() == []
    with pytest.raises(EnvironmentResolutionError):
        es._config(None)


def test_env_without_es_target_clean_error():
    cfg = OpsRAGConfig()
    cfg.environments = EnvironmentsConfig(
        default="bare",
        targets={"bare": EnvironmentTarget()},  # no elasticsearch block
    )
    bind_environments(cfg)
    with pytest.raises(es.MCPElasticsearchError) as ei:
        es._config("bare")
    assert ei.value.reason == "bad_env"
    assert "elasticsearch not configured for env" in str(ei.value)


def test_resolve_env_precedence():
    # env arg wins over cluster alias; cluster alias works; neither -> None.
    assert es._resolve_env({"env": "a", "cluster": "b"}) == "a"
    assert es._resolve_env({"cluster": "b"}) == "b"
    assert es._resolve_env({}) is None


# --- _base: reach modes (network mocked) ---------------------------------


@pytest.mark.asyncio
async def test_base_direct_requires_url():
    cfg = OpsRAGConfig()
    cfg.environments = EnvironmentsConfig(
        default="nodb",
        targets={
            "nodb": EnvironmentTarget(
                elasticsearch=EsTarget(reach="direct", url=None),
            ),
        },
    )
    bind_environments(cfg)
    bundle = es._config("nodb")
    with pytest.raises(es.MCPElasticsearchError) as ei:
        await es._base(bundle)
    assert ei.value.reason == "bad_env"


@pytest.mark.asyncio
async def test_base_port_forward_tunnels_through_cluster(monkeypatch):
    """reach=port_forward builds the API-server service-proxy URL via
    cluster_api_access and PRESERVES the ES API-key Authorization header."""
    cfg = OpsRAGConfig()
    cfg.environments = EnvironmentsConfig(
        default="eck",
        targets={
            "eck": EnvironmentTarget(
                elasticsearch=EsTarget(
                    reach="port_forward",
                    service="eck-infra-logs-es-http",
                    namespace="eck-infra",
                    port=9200,
                    api_key_env="PROD_ES_KEY",
                ),
            ),
        },
    )
    bind_environments(cfg)
    monkeypatch.setenv("PROD_ES_KEY", "es-api-key")

    from opsrag.mcp import kubernetes as k8s

    async def _fake_access(env):
        assert env == "eck"
        return {"host": "https://api.eck.example.com", "token": "cluster-tok", "verify": "/tmp/ca.crt"}

    monkeypatch.setattr(k8s, "cluster_api_access", _fake_access)

    bundle = es._config("eck")
    base, headers, auth, verify = await es._base(bundle)
    assert base == (
        "https://api.eck.example.com/api/v1/namespaces/eck-infra/"
        "services/http:eck-infra-logs-es-http:9200/proxy"
    )
    # port_forward preserves the ES API key (NOT clobbered by the cluster token).
    assert headers["Authorization"] == "ApiKey es-api-key"
    assert auth is None
    assert verify == "/tmp/ca.crt"


@pytest.mark.asyncio
async def test_base_proxy_uses_cluster_bearer_when_no_es_auth(monkeypatch):
    """reach=proxy with no ES creds -> the cluster bearer token authenticates."""
    cfg = OpsRAGConfig()
    cfg.environments = EnvironmentsConfig(
        default="px",
        targets={
            "px": EnvironmentTarget(
                elasticsearch=EsTarget(
                    reach="proxy", service="es-http", namespace="logs", port=9200,
                ),
            ),
        },
    )
    bind_environments(cfg)

    from opsrag.mcp import kubernetes as k8s

    async def _fake_access(env):
        return {"host": "https://api.px", "token": "ctok", "verify": False}

    monkeypatch.setattr(k8s, "cluster_api_access", _fake_access)

    base, headers, auth, verify = await es._base(es._config("px"))
    assert base.endswith("/services/http:es-http:9200/proxy")
    assert headers["Authorization"] == "Bearer ctok"
    assert verify is False


@pytest.mark.asyncio
async def test_base_port_forward_requires_service_and_namespace():
    cfg = OpsRAGConfig()
    cfg.environments = EnvironmentsConfig(
        default="bad",
        targets={
            "bad": EnvironmentTarget(
                elasticsearch=EsTarget(reach="port_forward", service=None, namespace=None),
            ),
        },
    )
    bind_environments(cfg)
    with pytest.raises(es.MCPElasticsearchError) as ei:
        await es._base(es._config("bad"))
    assert ei.value.reason == "bad_env"


# --- handler end-to-end with MOCKED httpx --------------------------------


class _FakeResponse:
    def __init__(self, status_code=200, json_data=None, text="{}"):
        self.status_code = status_code
        self._json = json_data if json_data is not None else {}
        self.text = text

    def json(self):
        return self._json


class _FakeAsyncClient:
    """Captures the URL/headers/verify the handler would hit; returns canned
    JSON. Stands in for httpx.AsyncClient -- no sockets."""

    calls: list[dict] = []

    def __init__(self, *, headers=None, auth=None, timeout=None, verify=None):
        self._headers = headers or {}
        self._auth = auth
        self._verify = verify

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url, params=None):
        _FakeAsyncClient.calls.append(
            {"method": "GET", "url": url, "params": params, "headers": self._headers,
             "auth": self._auth, "verify": self._verify}
        )
        return _FakeResponse(json_data={"cluster_name": "c", "status": "green",
                                        "number_of_nodes": 1}, text="{}")

    async def post(self, url, json=None):
        _FakeAsyncClient.calls.append(
            {"method": "POST", "url": url, "json": json, "headers": self._headers,
             "auth": self._auth, "verify": self._verify}
        )
        return _FakeResponse(json_data={"hits": {"hits": []}}, text="{}")


@pytest.mark.asyncio
async def test_handler_default_env_hits_default_url(monkeypatch):
    """A handler with NO env arg targets the default env's URL (single-env
    callers unaffected)."""
    _bind_two_envs()
    monkeypatch.setenv("PROD_ES_KEY", "k")
    _FakeAsyncClient.calls = []
    monkeypatch.setattr(es.httpx, "AsyncClient", _FakeAsyncClient)

    res = await es.get_tool("elasticsearch_cluster_health").handler(None, {})
    assert res["env"] is None  # default selected (echoed env arg)
    assert res["status"] == "green"
    call = _FakeAsyncClient.calls[-1]
    assert call["url"] == "https://es.prod.example.com:9200/_cluster/health"
    assert call["headers"]["Authorization"] == "ApiKey k"
    assert call["verify"] is True


@pytest.mark.asyncio
async def test_handler_env_arg_targets_other_env(monkeypatch):
    _bind_two_envs()
    monkeypatch.setenv("STG_ES_USER", "u")
    monkeypatch.setenv("STG_ES_PASS", "p")
    _FakeAsyncClient.calls = []
    monkeypatch.setattr(es.httpx, "AsyncClient", _FakeAsyncClient)

    res = await es.get_tool("elasticsearch_cluster_health").handler(None, {"env": "staging"})
    assert res["env"] == "staging"
    call = _FakeAsyncClient.calls[-1]
    assert call["url"] == "http://os.staging.svc:9200/_cluster/health"
    assert call["auth"] == ("u", "p")
    assert call["verify"] is False  # staging verify_ssl=False


@pytest.mark.asyncio
async def test_handler_cluster_alias_targets_other_env(monkeypatch):
    _bind_two_envs()
    monkeypatch.setenv("STG_ES_USER", "u")
    monkeypatch.setenv("STG_ES_PASS", "p")
    _FakeAsyncClient.calls = []
    monkeypatch.setattr(es.httpx, "AsyncClient", _FakeAsyncClient)

    res = await es.get_tool("elasticsearch_cluster_health").handler(None, {"cluster": "staging"})
    assert res["env"] == "staging"
    assert _FakeAsyncClient.calls[-1]["url"].startswith("http://os.staging.svc:9200")


@pytest.mark.asyncio
async def test_search_applies_index_pattern_and_timestamp_sort(monkeypatch):
    """Default index comes from the env's index_pattern; default sort uses the
    env's mapped timestamp field."""
    _bind_two_envs()
    monkeypatch.setenv("PROD_ES_KEY", "k")
    _FakeAsyncClient.calls = []
    monkeypatch.setattr(es.httpx, "AsyncClient", _FakeAsyncClient)

    await es.get_tool("elasticsearch_search").handler(None, {"q": "level:error"})
    call = _FakeAsyncClient.calls[-1]
    # default index_pattern app-logs-* used in the path.
    assert call["url"] == "https://es.prod.example.com:9200/app-logs-*/_search"
    # default sort on the env's PHYSICAL timestamp field.
    assert call["json"]["sort"] == [{"@timestamp": {"order": "desc"}}]


@pytest.mark.asyncio
async def test_search_service_filter_uses_mapped_field(monkeypatch):
    """`service` arg is mapped to the env's physical service field."""
    _bind_two_envs()
    monkeypatch.setenv("PROD_ES_KEY", "k")
    _FakeAsyncClient.calls = []
    monkeypatch.setattr(es.httpx, "AsyncClient", _FakeAsyncClient)

    await es.get_tool("elasticsearch_search").handler(
        None, {"q": "level:error", "service": "payments"},
    )
    body = _FakeAsyncClient.calls[-1]["json"]
    must = body["query"]["bool"]["must"]
    assert {"term": {"kubernetes.labels.app_name": "payments"}} in must


@pytest.mark.asyncio
async def test_handler_unknown_env_raises(monkeypatch):
    _bind_two_envs()
    monkeypatch.setattr(es.httpx, "AsyncClient", _FakeAsyncClient)
    with pytest.raises(EnvironmentResolutionError):
        await es.get_tool("elasticsearch_cluster_health").handler(None, {"env": "ghost"})


# --- back-compat shim ----------------------------------------------------


def test_bind_is_noop_shim():
    # The legacy single-endpoint bind() is a no-op now; resolution flows
    # through the registry. Calling it must not raise or alter resolution.
    _bind_two_envs()
    es.bind(object())  # arbitrary arg accepted, ignored
    assert available_environments() == ["prod", "staging"]
