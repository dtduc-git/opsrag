"""Tests for the prompt-render helper and its process-level active
deployment context (T035b + T056 foundation)."""
from __future__ import annotations

import pytest

from opsrag.agent.prompt_render import (
    active_deployment,
    render,
    set_active_deployment,
)
from opsrag.context import DeploymentContext


@pytest.fixture(autouse=True)
def _reset_active():
    # Each test starts and ends with no active deployment so the process
    # global never leaks between tests.
    set_active_deployment(None)
    yield
    set_active_deployment(None)


def test_empty_context_renders_blank_substitutions():
    out = render("services: [{services_csv}]")
    assert out == "services: []"


def test_explicit_ctx_overrides_active():
    set_active_deployment(DeploymentContext(services=["x"]))
    out = render("[{services_csv}]", ctx=DeploymentContext(services=["a", "b"]))
    assert out == "[a, b]"


def test_active_deployment_flows_without_explicit_ctx():
    set_active_deployment(DeploymentContext(services=["api", "web"]))
    assert render("[{services_csv}]") == "[api, web]"


def test_unknown_placeholder_raises():
    with pytest.raises(KeyError):
        render("{not_a_real_key}")


def test_active_deployment_defaults_to_empty_context():
    assert isinstance(active_deployment(), DeploymentContext)
    assert active_deployment().services == []
