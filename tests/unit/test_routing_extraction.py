"""Routing/topology extractor tests -- Kong, Ingress, Istio, Gateway API, Traefik."""
from __future__ import annotations

import yaml

from opsrag.extractors.routing import extract_routing, parse_k8s_fqdn


def _by_label(ents: dict, label: str):
    return [e for e in ents.values() if e.label == label]


def _rel_types(rels: dict):
    return {k[2] for k in rels}


def test_parse_k8s_fqdn():
    assert parse_k8s_fqdn(
        "service-a-appservice-component-a.service-a.svc.cluster.local"
    ) == ("service-a-appservice-component-a", "service-a")
    assert parse_k8s_fqdn("reviews.prod.svc") == ("reviews", "prod")
    assert parse_k8s_fqdn("api.example.com") is None


def test_kong_declarative_the_typical_case():
    """Two Kong routes /abc and /def pointing at two components of service-a."""
    doc = yaml.safe_load(
        """
        _format_version: "3.0"
        services:
          - name: comp-a
            url: http://service-a-appservice-component-a.service-a.svc.cluster.local:8080
            routes:
              - name: r-abc
                paths: ["/abc"]
          - name: comp-b
            host: service-a-appservice-component-b.service-a.svc.cluster.local
            routes:
              - name: r-def
                paths: ["/def"]
        """
    )
    ents, rels = extract_routing([doc], "repo:kong.yaml")

    routes = {e.properties["path"]: e for e in _by_label(ents, "Route")}
    assert set(routes) == {"/abc", "/def"}

    # Both components resolve to Services in namespace service-a.
    svc_names = {e.name for e in _by_label(ents, "Service")}
    assert "service-a-appservice-component-a" in svc_names
    assert "service-a-appservice-component-b" in svc_names
    ns = {e.name for e in _by_label(ents, "Namespace")}
    assert ns == {"service-a"}

    # The key edges exist: Gateway-HAS_ROUTE->Route-ROUTES_TO->Service-IN_NAMESPACE->Namespace
    assert {"HAS_ROUTE", "ROUTES_TO", "IN_NAMESPACE"} <= _rel_types(rels)

    # /abc actually points at component-a (not component-b).
    abc_id = routes["/abc"].id
    comp_a_id = next(e.id for e in _by_label(ents, "Service")
                     if e.name == "service-a-appservice-component-a")
    assert (abc_id, comp_a_id, "ROUTES_TO") in rels


def test_k8s_ingress():
    doc = yaml.safe_load(
        """
        apiVersion: networking.k8s.io/v1
        kind: Ingress
        metadata: {name: web, namespace: shop}
        spec:
          ingressClassName: nginx
          rules:
            - host: shop.example.com
              http:
                paths:
                  - path: /cart
                    backend: {service: {name: cart, port: {number: 8080}}}
        """
    )
    ents, rels = extract_routing([doc], "repo:ing.yaml")
    assert any(e.label == "Route" and e.properties["path"] == "/cart" for e in ents.values())
    assert any(e.label == "Service" and e.name == "cart" for e in ents.values())
    assert "ROUTES_TO" in _rel_types(rels)


def test_istio_virtualservice():
    doc = yaml.safe_load(
        """
        apiVersion: networking.istio.io/v1beta1
        kind: VirtualService
        metadata: {name: reviews, namespace: prod}
        spec:
          hosts: ["reviews.example.com"]
          http:
            - match: [{uri: {prefix: /reviews}}]
              route:
                - destination: {host: reviews.prod.svc.cluster.local}
        """
    )
    ents, rels = extract_routing([doc], "repo:vs.yaml")
    assert any(e.label == "Route" and e.properties["path"] == "/reviews" for e in ents.values())
    assert any(e.label == "Service" and e.name == "reviews" for e in ents.values())


def test_gateway_api_httproute():
    doc = yaml.safe_load(
        """
        apiVersion: gateway.networking.k8s.io/v1
        kind: HTTPRoute
        metadata: {name: r, namespace: prod}
        spec:
          parentRefs: [{name: prod-gw}]
          hostnames: ["api.example.com"]
          rules:
            - matches: [{path: {type: PathPrefix, value: /v1}}]
              backendRefs: [{name: api-svc, port: 80}]
        """
    )
    ents, rels = extract_routing([doc], "repo:hr.yaml")
    assert any(e.label == "Gateway" and e.properties.get("type") == "gateway-api" for e in ents.values())
    assert any(e.label == "Service" and e.name == "api-svc" for e in ents.values())
    assert "ROUTES_TO" in _rel_types(rels)


def test_host_cluster_middleware():
    """Kong route on a host, with a plugin, in a gitops cluster path."""
    doc = yaml.safe_load(
        """
        _format_version: "3.0"
        services:
          - name: api
            url: http://api.prod.svc.cluster.local:8080
            routes:
              - name: r1
                paths: ["/v1"]
                hosts: ["api.example.com"]
                plugins:
                  - name: rate-limiting
                  - name: jwt
        """
    )
    ents, rels = extract_routing([doc], "gitops:clusters/prod/kong/api.yaml")
    hosts = {e.name for e in _by_label(ents, "Host")}
    assert "api.example.com" in hosts
    mws = {e.name for e in _by_label(ents, "Middleware")}
    assert {"rate-limiting", "jwt"} <= mws
    clusters = {e.name for e in _by_label(ents, "Cluster")}
    assert "prod" in clusters
    assert {"HAS_HOST", "USES_MIDDLEWARE", "IN_CLUSTER"} <= _rel_types(rels)


def test_traefik_ingressroute():
    doc = yaml.safe_load(
        """
        apiVersion: traefik.io/v1alpha1
        kind: IngressRoute
        metadata: {name: ir, namespace: web}
        spec:
          routes:
            - match: Host(`t.example.com`) && PathPrefix(`/api`)
              services: [{name: backend, port: 80}]
        """
    )
    ents, rels = extract_routing([doc], "repo:ir.yaml")
    assert any(e.label == "Route" and e.properties["path"] == "/api" for e in ents.values())
    assert any(e.label == "Service" and e.name == "backend" for e in ents.values())
