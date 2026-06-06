"""Integration test (T081): the Kubernetes MCP tools against the fake backend.

Exercises representative tools through build_fake() with no kube-apiserver,
no kubeconfig, and no network, asserting shape-faithful responses and the
registry's declared tool set. Follows the GitLab reference test
(tests/integration/test_gitlab_mcp.py) and the per-MCP fake contract (FR-012).

The fake reuses the upstream-style hand-rolled "FakeApiClient" pattern: fake
per-resource API objects returning real kubernetes_asyncio V1 models (see
opsrag/mcp/kubernetes.py build_fake()).
"""
from __future__ import annotations

import pytest

# Optional-dependency skip guard: the `kubernetes_asyncio` package may be
# absent in a trimmed CI image. build_fake() builds real V1 models, so it
# needs the package; skip the whole module cleanly if it's missing.
pytest.importorskip(
    "kubernetes_asyncio",
    reason="kubernetes_asyncio not installed; skipping K8s MCP integration test",
)

from opsrag.mcp.kubernetes import build_fake  # noqa: E402
from opsrag.mcp.registry import REGISTRY  # noqa: E402


@pytest.fixture
def fake():
    f = build_fake()
    try:
        yield f
    finally:
        f.close()  # restore deployment context + module fake state


def test_fake_exposes_registry_tool_set(fake) -> None:
    # The fake's tools must match exactly what the registry declares.
    assert set(fake.tool_names()) == set(REGISTRY["kubernetes"].tool_names)


@pytest.mark.asyncio
async def test_list_pods(fake) -> None:
    result = await fake.call("k8s_list_pods", {"namespace": "default"})
    assert result["cluster"] == "example-cluster"
    assert result["count"] == len(result["pods"]) >= 1
    pod = result["pods"][0]
    assert pod["namespace"] == "default"
    assert pod["phase"] == "Running"
    # Aggregated container signal is surfaced shape-faithfully.
    assert "worst_termination_summary" in pod
    assert pod["containers"][0]["name"] == "app"


@pytest.mark.asyncio
async def test_get_deployment(fake) -> None:
    result = await fake.call(
        "k8s_get_deployment", {"namespace": "default", "name": "example-deployment"}
    )
    assert result["cluster"] == "example-cluster"
    dep = result["deployment"]
    assert dep["name"] == "example-deployment"
    assert dep["replicas_desired"] == 3
    assert dep["replicas_ready"] == 3
    assert dep["containers"][0]["image"].startswith("registry.example.com/")


@pytest.mark.asyncio
async def test_get_service(fake) -> None:
    result = await fake.call(
        "k8s_get_service", {"namespace": "default", "name": "example-service"}
    )
    svc = result["service"]
    assert svc["name"] == "example-service"
    assert svc["type"] == "ClusterIP"
    assert svc["ports"][0]["port"] == 80


@pytest.mark.asyncio
async def test_unknown_tool_raises(fake) -> None:
    with pytest.raises(KeyError):
        await fake.call("k8s_does_not_exist", {})
