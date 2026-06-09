"""Unified multi-environment registry resolver (Approach A).

A process-global registry mapping env name -> EnvironmentTarget (how to
reach that env's kubernetes / prometheus / elasticsearch). Bound once at
startup from ``Settings.environments``; when no explicit registry is set,
it is SYNTHESIZED from the legacy ``k8s`` / ``elasticsearch`` /
``deployment`` blocks so existing deployments keep working.

Mirrors the active-deployment / active-enabled globals: a single setter at
startup, pure lookups thereafter. Lookup misses raise structured errors
(never a silent default) -- consistent with DeploymentContext semantics.

See docs/superpowers/specs/2026-06-09-multi-env-environments-registry-design.md
"""
from __future__ import annotations

import logging
import os
from typing import Any

from opsrag.config import EnvironmentTarget, EsTarget, K8sTarget, PrometheusTarget

_log = logging.getLogger("opsrag.environments")

# Historical hardcoded prometheus service. The legacy-synthesis path keeps
# using it so deployments that relied on the old constant behave identically
# after the refactor (the NEW model default is the vendor-neutral
# kube-prometheus-stack name).
_LEGACY_PROM_SERVICE = "monitoring-main-prometheus"
_LEGACY_PROM_ISTIO = "monitoring-istio-prometheus"

_REGISTRY: dict[str, EnvironmentTarget] | None = None
_DEFAULT_ENV: str | None = None


class EnvironmentResolutionError(Exception):
    """Raised when an env name cannot be resolved to a target."""


def bind_environments(cfg: Any) -> None:
    """Install the active environment registry. Call once at startup."""
    global _REGISTRY, _DEFAULT_ENV
    envs = getattr(cfg, "environments", None)
    targets = dict(getattr(envs, "targets", {}) or {}) if envs is not None else {}
    if targets:
        _REGISTRY = targets
        _DEFAULT_ENV = getattr(envs, "default", None) or next(iter(targets), None)
        _log.info(
            "environments registry bound: %d env(s) %s (default=%s)",
            len(targets), sorted(targets), _DEFAULT_ENV,
        )
        return

    synth, default = _synthesize_legacy(cfg)
    _REGISTRY = synth
    _DEFAULT_ENV = (getattr(envs, "default", None) if envs is not None else None) or default
    if synth:
        _log.warning(
            "no `environments:` registry set -- synthesized %d env(s) %s from "
            "legacy k8s/elasticsearch/deployment config (DEPRECATED; migrate to "
            "the `environments:` block). default=%s",
            len(synth), sorted(synth), _DEFAULT_ENV,
        )
    else:
        _log.info(
            "no environments configured -- k8s/prometheus/elasticsearch tools "
            "report no-environment until configured"
        )


def _synthesize_legacy(cfg: Any) -> tuple[dict[str, EnvironmentTarget], str | None]:
    """Build a registry from the legacy k8s/elasticsearch/deployment blocks."""
    k8s = getattr(cfg, "k8s", None)
    es = getattr(cfg, "elasticsearch", None)
    deployment = getattr(cfg, "deployment", None)

    gke_clusters = dict(getattr(k8s, "clusters", {}) or {})  # env -> K8sClusterCoords
    ctx_clusters: dict[str, str] = {}
    if deployment is not None:
        k8s_ctx = getattr(deployment, "kubernetes", None)
        ctx_clusters = dict(getattr(k8s_ctx, "clusters", {}) or {})  # env -> context name

    env_names = list(dict.fromkeys([*gke_clusters.keys(), *ctx_clusters.keys()]))
    es_target = _legacy_es_target(es)  # legacy ES is a single global endpoint

    if not env_names:
        if es_target is not None:
            return ({"default": EnvironmentTarget(elasticsearch=es_target)}, "default")
        return ({}, None)

    registry: dict[str, EnvironmentTarget] = {}
    for name in env_names:
        if name in gke_clusters:
            coords = gke_clusters[name]
            k8s_t = K8sTarget(
                mode="gke",
                project=getattr(coords, "project", None),
                location=getattr(coords, "location", None),
                name=getattr(coords, "name", None),
            )
        else:
            k8s_t = K8sTarget(mode="kubeconfig", context=ctx_clusters.get(name))
        prom_t = PrometheusTarget(
            reach="k8s_proxy", namespace="monitoring",
            service=_LEGACY_PROM_SERVICE, port=9090,
            extra_services={"istio": _LEGACY_PROM_ISTIO},
        )
        registry[name] = EnvironmentTarget(
            kubernetes=k8s_t, prometheus=prom_t, elasticsearch=es_target,
        )
    default = getattr(k8s, "default_cluster", None) or env_names[0]
    return registry, default


def _legacy_es_target(es: Any) -> EsTarget | None:
    if es is None or not getattr(es, "enabled", False):
        return None
    url = (getattr(es, "url", "") or "").strip()
    if not url:
        url_env = getattr(es, "url_env", "ES_URL") or "ES_URL"
        url = (os.environ.get(url_env) or "").strip()
    backend = getattr(es, "backend", "elasticsearch") or "elasticsearch"
    if backend not in ("elasticsearch", "opensearch"):
        backend = "elasticsearch"
    return EsTarget(
        reach="direct",
        url=url or None,
        api_key_env=(getattr(es, "api_key_env", None) or None),
        username_env=(getattr(es, "username_env", None) or None),
        password_env=(getattr(es, "password_env", None) or None),
        index_pattern=(getattr(es, "default_index", "*") or "*"),
        backend=backend,
        verify_ssl=bool(getattr(es, "verify_ssl", True)),
    )


def available_environments() -> list[str]:
    """Sorted list of configured env names ([] when unbound/empty)."""
    return sorted(_REGISTRY or {})


def default_environment() -> str | None:
    return _DEFAULT_ENV


def resolve_environment(name: str | None = None) -> EnvironmentTarget:
    """Resolve an env name (or the default) to its target. Raises
    EnvironmentResolutionError on an unknown env / empty registry."""
    if not _REGISTRY:
        raise EnvironmentResolutionError(
            "no environments configured. Set the `environments:` block "
            "(or legacy k8s.clusters / elasticsearch) in config."
        )
    key = name or _DEFAULT_ENV or next(iter(_REGISTRY))
    target = _REGISTRY.get(key)
    if target is None:
        raise EnvironmentResolutionError(
            f"unknown environment {key!r}. Configured: {available_environments()}."
        )
    return target


def reset_environments() -> None:
    """Test helper -- clear the bound registry."""
    global _REGISTRY, _DEFAULT_ENV
    _REGISTRY = None
    _DEFAULT_ENV = None
