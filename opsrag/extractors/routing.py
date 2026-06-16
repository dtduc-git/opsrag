"""Routing / topology extraction across the common OSS ingress + mesh stacks.

Turns API-gateway / service-mesh routing into graph structure so OpsRAG can
answer the multi-hop questions pure vector RAG can't:

  * "What does route /abc point to?"        Route(/abc) -ROUTES_TO-> Service(...)
  * "Which routes hit service-a?"           Namespace(service-a) <-IN_NAMESPACE-
                                              Service(component-a,b) <-ROUTES_TO- Route
  * "Blast radius of service-a's gateway?"  Gateway -HAS_ROUTE-> Route -ROUTES_TO-> ...

Providers covered (all map onto the SAME node/edge vocabulary):
  - Kong            declarative (decK / DB-less): top-level `services` + `routes`
  - Kubernetes      Ingress (networking.k8s.io)  -> nginx / ALB / GCE / HAProxy / Kong-ingress
  - Istio           VirtualService, Gateway, ServiceEntry (networking.istio.io)
  - Gateway API     HTTPRoute / GRPCRoute / Gateway (gateway.networking.k8s.io)
  - Traefik         IngressRoute (traefik.io / traefik.containo.us)
  - Contour         HTTPProxy (projectcontour.io)
  - Ambassador      Mapping (getambassador.io)
  - APISIX          ApisixRoute (apisix.apache.org)

Nodes: Gateway, Route, Endpoint, Namespace, Service.
Edges: HAS_ROUTE, ROUTES_TO, RESOLVES_TO, IN_NAMESPACE, COMPONENT_OF.

Service IDs reuse `_eid("Service", f"k8s:{ns}/{name}")` so an upstream referenced
by a route MERGES with its Deployment/Service node (RuleBasedExtractor._extract_k8s).
"""
from __future__ import annotations

import hashlib
import re
from urllib.parse import urlparse

from opsrag.extractors.schema import sanitize_value
from opsrag.interfaces.graphstore import Entity, Relationship

# <svc>.<ns>.svc[.cluster.local] -- service labels carry no dots, so the first
# label is the service and the second the namespace.
_K8S_FQDN_RE = re.compile(
    r"^([a-z0-9][a-z0-9-]*)\.([a-z0-9][a-z0-9-]*)\.svc(?:\.[a-z0-9.-]+)?\.?$", re.I
)
# Traefik matcher DSL: Host(`a`), PathPrefix(`/b`), Path(`/c`)
_TRAEFIK_HOST_RE = re.compile(r"Host\(`([^`]+)`\)")
_TRAEFIK_PATH_RE = re.compile(r"Path(?:Prefix)?\(`([^`]+)`\)")

# Cluster / gitops-env detection from the file path. CONSERVATIVE on purpose:
# `overlays/<x>` and `environments/<x>` frequently hold APP names (not envs) in
# real monorepos, so we only trust (1) an explicit env-token path segment, or
# (2) a literal `clusters/<name>` directory. Anything else -> no cluster (better
# to miss than to mint a bogus "external-dns"/"kong" cluster node).
_CLUSTER_PATH_RE = re.compile(r"(?:^|/)clusters?/([a-z0-9][a-z0-9._-]*)", re.I)
_ENV_TOKENS = frozenset({
    "prod", "production", "staging", "stage", "preprod",
    "dev", "development", "qa", "uat", "sandbox",
})
# Ingress annotation prefixes that denote a middleware/policy (auth, rate-limit,
# cors, etc.) -> a Middleware node attached to the route.
_MW_ANNOTATION_HINTS = (
    "auth", "rate-limit", "ratelimit", "cors", "jwt", "oauth", "whitelist",
    "limit-", "canary", "rewrite", "ssl-redirect", "modsecurity", "waf",
)


def cluster_from_source(source_id: str) -> str | None:
    """Derive a cluster/env name from the source path (gitops layout).

    Env-token segment (prod/staging/dev/...) wins first -- it's the reliable
    signal -- then a literal `clusters/<name>` directory. Returns None otherwise."""
    path = source_id.split(":", 1)[1] if ":" in source_id else source_id
    for seg in path.lower().split("/"):
        if seg in _ENV_TOKENS:
            return seg
    m = _CLUSTER_PATH_RE.search(path)
    if m:
        return sanitize_value(m.group(1))
    return None


def _eid(label: str, key: str) -> str:
    """Deterministic id, matching RuleBasedExtractor._eid so nodes merge."""
    h = hashlib.sha1(key.encode()).hexdigest()[:12]
    return f"{label.lower()}:{key}:{h}"


def _host_port(host_or_url: str) -> tuple[str, int | None]:
    s = (host_or_url or "").strip()
    if not s:
        return "", None
    if "://" in s:
        u = urlparse(s)
        return (u.hostname or ""), u.port
    if ":" in s and s.rsplit(":", 1)[1].isdigit():
        h, p = s.rsplit(":", 1)
        return h, int(p)
    return s, None


def parse_k8s_fqdn(host: str) -> tuple[str, str] | None:
    """`<svc>.<ns>.svc[.cluster.local]` -> (service, namespace), else None."""
    m = _K8S_FQDN_RE.match((host or "").strip())
    return (m.group(1), m.group(2)) if m else None


def _ns_node(ns: str, source_id: str, ents: dict) -> str:
    nid = _eid("Namespace", ns)
    ents.setdefault(
        nid, Entity(id=nid, label="Namespace", name=ns, properties={}, source_chunk_id=source_id)
    )
    return nid


def _rel(rels: dict, src: str, tgt: str, rtype: str, source_id: str) -> None:
    rels[(src, tgt, rtype)] = Relationship(
        source_id=src, target_id=tgt, rel_type=rtype,
        properties={"source_chunk_id": source_id},
    )


def _upstream_node(
    host_or_url: str, source_id: str, ents: dict, rels: dict, *, default_ns: str | None = None,
) -> str | None:
    """Resolve a routing upstream to a node id.

    Cluster-local FQDN -> Service{name,namespace} (+ IN_NAMESPACE / COMPONENT_OF
    so it rolls up to the logical service = namespace). A bare service name with a
    known `default_ns` -> same-namespace Service. Anything else -> Endpoint."""
    host, _port = _host_port(host_or_url)
    host = sanitize_value(host)
    if not host:
        return None
    parsed = parse_k8s_fqdn(host)
    if parsed is None and default_ns and "." not in host:
        parsed = (host, default_ns)
    if parsed is not None:
        svc, ns = parsed
        sid = _eid("Service", f"k8s:{ns}/{svc}")
        ents.setdefault(
            sid, Entity(id=sid, label="Service", name=svc,
                        properties={"namespace": ns}, source_chunk_id=source_id)
        )
        nid = _ns_node(ns, source_id, ents)
        _rel(rels, sid, nid, "IN_NAMESPACE", source_id)
        _rel(rels, sid, nid, "COMPONENT_OF", source_id)
        return sid
    eid = _eid("Endpoint", host)
    ents.setdefault(
        eid, Entity(id=eid, label="Endpoint", name=host, properties={}, source_chunk_id=source_id)
    )
    return eid


def _gateway_node(name: str, gtype: str, source_id: str, ents: dict) -> str:
    name = sanitize_value(name) or gtype
    gid = _eid("Gateway", f"{gtype}:{name}")
    ents.setdefault(
        gid, Entity(id=gid, label="Gateway", name=name,
                    properties={"type": gtype}, source_chunk_id=source_id)
    )
    return gid


def _host_node(host: str, source_id: str, ents: dict) -> str | None:
    host = sanitize_value(host)
    if not host:
        return None
    hid = _eid("Host", host)
    ents.setdefault(
        hid, Entity(id=hid, label="Host", name=host, properties={}, source_chunk_id=source_id)
    )
    return hid


def _middleware_node(name: str, mtype: str, source_id: str, ents: dict) -> str | None:
    name = sanitize_value(name)
    if not name:
        return None
    mid = _eid("Middleware", f"{mtype}:{name}")
    ents.setdefault(
        mid, Entity(id=mid, label="Middleware", name=name,
                    properties={"type": mtype}, source_chunk_id=source_id)
    )
    return mid


def _add_route(
    gw_id: str, upstream_ids, *, paths, hosts, name: str, source_id: str, ents: dict, rels: dict,
    middlewares: list | None = None, mw_type: str = "plugin",
) -> None:
    """Create Route node(s) per path; wire Gateway -HAS_ROUTE-> Route -ROUTES_TO->
    upstream(s), Route -HAS_HOST-> Host, and Route -USES_MIDDLEWARE-> Middleware."""
    if isinstance(upstream_ids, str) or upstream_ids is None:
        upstream_ids = [upstream_ids] if upstream_ids else []
    path_list = [sanitize_value(p) for p in (paths or ["/"]) if sanitize_value(p)] or ["/"]
    host = sanitize_value((list(hosts) or [""])[0]) if hosts else ""
    mw_ids = [m for m in (_middleware_node(n, mw_type, source_id, ents) for n in (middlewares or [])) if m]
    for path in path_list:
        rid = _eid("Route", f"{gw_id}:{host}{path}")
        disp = f"{host}{path}" if host else path
        ents.setdefault(
            rid, Entity(id=rid, label="Route", name=disp,
                        properties={"path": path, "host": host, "route_name": sanitize_value(name)},
                        source_chunk_id=source_id)
        )
        _rel(rels, gw_id, rid, "HAS_ROUTE", source_id)
        for up in upstream_ids:
            if up:
                _rel(rels, rid, up, "ROUTES_TO", source_id)
        hid = _host_node(host, source_id, ents)
        if hid:
            _rel(rels, rid, hid, "HAS_HOST", source_id)
        for mid in mw_ids:
            _rel(rels, rid, mid, "USES_MIDDLEWARE", source_id)


# --- per-provider handlers --------------------------------------------------

def _kong_plugins(obj: dict) -> list:
    return [p.get("name") for p in (obj.get("plugins") or []) if isinstance(p, dict) and p.get("name")]


def _kong_declarative(doc: dict, source_id: str, ents: dict, rels: dict) -> None:
    gw_id = _gateway_node("kong", "kong", source_id, ents)
    svc_upstream: dict[str, str | None] = {}
    svc_plugins: dict[str, list] = {}
    for svc in doc.get("services") or []:
        if not isinstance(svc, dict):
            continue
        sname = sanitize_value(svc.get("name") or "")
        up = _upstream_node(svc.get("url") or svc.get("host") or "", source_id, ents, rels)
        plugins = _kong_plugins(svc)
        if sname:
            svc_upstream[sname] = up
            svc_plugins[sname] = plugins
        for r in svc.get("routes") or []:
            if isinstance(r, dict):
                _add_route(gw_id, up, paths=r.get("paths"), hosts=r.get("hosts"),
                           name=r.get("name", ""), source_id=source_id, ents=ents, rels=rels,
                           middlewares=plugins + _kong_plugins(r), mw_type="kong-plugin")
    for r in doc.get("routes") or []:
        if not isinstance(r, dict):
            continue
        ref = r.get("service")
        ref_name = sanitize_value((ref.get("name") if isinstance(ref, dict) else ref) or "")
        up = svc_upstream.get(ref_name)
        _add_route(gw_id, up, paths=r.get("paths"), hosts=r.get("hosts"),
                   name=r.get("name", ""), source_id=source_id, ents=ents, rels=rels,
                   middlewares=svc_plugins.get(ref_name, []) + _kong_plugins(r), mw_type="kong-plugin")


def _ingress(m: dict, source_id: str, ents: dict, rels: dict) -> None:
    meta = m.get("metadata") or {}
    ns = sanitize_value(meta.get("namespace") or "default")
    spec = m.get("spec") or {}
    annotations = meta.get("annotations") or {}
    gclass = spec.get("ingressClassName") or annotations.get("kubernetes.io/ingress.class")
    gw_id = _gateway_node(gclass or "ingress", "ingress", source_id, ents)
    # Middleware from annotations: konghq.com/plugins, or any annotation key that
    # names a policy (auth / rate-limit / cors / jwt / ...).
    mws: list = []
    for k, v in annotations.items():
        kl = str(k).lower()
        if kl.endswith("konghq.com/plugins"):
            mws.extend(str(v).split(","))
        elif any(h in kl for h in _MW_ANNOTATION_HINTS):
            mws.append(kl.rsplit("/", 1)[-1])
    for rule in spec.get("rules") or []:
        if not isinstance(rule, dict):
            continue
        host = sanitize_value(rule.get("host") or "")
        for p in ((rule.get("http") or {}).get("paths") or []):
            if not isinstance(p, dict):
                continue
            backend = p.get("backend") or {}
            svc = (backend.get("service") or {}).get("name") or backend.get("serviceName") or ""
            up = _upstream_node(svc, source_id, ents, rels, default_ns=ns) if svc else None
            _add_route(gw_id, up, paths=[p.get("path") or "/"], hosts=[host],
                       name=meta.get("name", ""), source_id=source_id, ents=ents, rels=rels,
                       middlewares=mws, mw_type="ingress-annotation")


def _istio_virtualservice(m: dict, source_id: str, ents: dict, rels: dict) -> None:
    meta = m.get("metadata") or {}
    ns = sanitize_value(meta.get("namespace") or "default")
    spec = m.get("spec") or {}
    gws = spec.get("gateways") or ["mesh"]
    gw_id = _gateway_node(f"istio:{sanitize_value(str(gws[0]))}", "istio", source_id, ents)
    hosts = [sanitize_value(h) for h in (spec.get("hosts") or [])]
    for http in spec.get("http") or []:
        if not isinstance(http, dict):
            continue
        paths = []
        for match in http.get("match") or []:
            uri = (match or {}).get("uri") or {}
            paths.append(uri.get("prefix") or uri.get("exact") or uri.get("regex") or "/")
        ups = []
        for route in http.get("route") or []:
            dest = (route or {}).get("destination") or {}
            up = _upstream_node(dest.get("host", ""), source_id, ents, rels, default_ns=ns)
            if up:
                ups.append(up)
        _add_route(gw_id, ups, paths=paths or ["/"], hosts=hosts,
                   name=meta.get("name", ""), source_id=source_id, ents=ents, rels=rels)


def _istio_gateway(m: dict, source_id: str, ents: dict, rels: dict) -> None:
    meta = m.get("metadata") or {}
    _gateway_node(f"istio:{sanitize_value(meta.get('name',''))}", "istio", source_id, ents)


def _istio_serviceentry(m: dict, source_id: str, ents: dict, rels: dict) -> None:
    # External services registered into the mesh -> Endpoint nodes (dependencies).
    for h in (m.get("spec") or {}).get("hosts") or []:
        _upstream_node(sanitize_value(h), source_id, ents, rels)


def _gatewayapi_route(m: dict, source_id: str, ents: dict, rels: dict) -> None:
    meta = m.get("metadata") or {}
    ns = sanitize_value(meta.get("namespace") or "default")
    spec = m.get("spec") or {}
    parents = spec.get("parentRefs") or [{}]
    pname = (parents[0] or {}).get("name", "gateway")
    gw_id = _gateway_node(f"gatewayapi:{sanitize_value(str(pname))}", "gateway-api", source_id, ents)
    hosts = [sanitize_value(h) for h in (spec.get("hostnames") or [])]
    for rule in spec.get("rules") or []:
        if not isinstance(rule, dict):
            continue
        paths = []
        for match in rule.get("matches") or []:
            path = (match or {}).get("path") or {}
            paths.append(path.get("value") or "/")
        ups = []
        for b in rule.get("backendRefs") or []:
            bn = (b or {}).get("name", "")
            bns = sanitize_value((b or {}).get("namespace") or ns)
            up = _upstream_node(bn, source_id, ents, rels, default_ns=bns) if bn else None
            if up:
                ups.append(up)
        _add_route(gw_id, ups, paths=paths or ["/"], hosts=hosts,
                   name=meta.get("name", ""), source_id=source_id, ents=ents, rels=rels)


def _traefik_ingressroute(m: dict, source_id: str, ents: dict, rels: dict) -> None:
    meta = m.get("metadata") or {}
    ns = sanitize_value(meta.get("namespace") or "default")
    gw_id = _gateway_node("traefik", "traefik", source_id, ents)
    for route in (m.get("spec") or {}).get("routes") or []:
        if not isinstance(route, dict):
            continue
        match = route.get("match") or ""
        hosts = _TRAEFIK_HOST_RE.findall(match)
        paths = _TRAEFIK_PATH_RE.findall(match) or ["/"]
        ups = []
        for s in route.get("services") or []:
            sn = (s or {}).get("name", "")
            up = _upstream_node(sn, source_id, ents, rels, default_ns=ns) if sn else None
            if up:
                ups.append(up)
        mws = [mw.get("name") for mw in (route.get("middlewares") or []) if isinstance(mw, dict) and mw.get("name")]
        _add_route(gw_id, ups, paths=paths, hosts=hosts,
                   name=meta.get("name", ""), source_id=source_id, ents=ents, rels=rels,
                   middlewares=mws, mw_type="traefik-middleware")


def _contour_httpproxy(m: dict, source_id: str, ents: dict, rels: dict) -> None:
    meta = m.get("metadata") or {}
    ns = sanitize_value(meta.get("namespace") or "default")
    spec = m.get("spec") or {}
    host = sanitize_value((spec.get("virtualhost") or {}).get("fqdn") or "")
    gw_id = _gateway_node("contour", "contour", source_id, ents)
    for route in spec.get("routes") or []:
        if not isinstance(route, dict):
            continue
        paths = [c.get("prefix") for c in (route.get("conditions") or []) if isinstance(c, dict) and c.get("prefix")] or ["/"]
        ups = []
        for s in route.get("services") or []:
            sn = (s or {}).get("name", "")
            up = _upstream_node(sn, source_id, ents, rels, default_ns=ns) if sn else None
            if up:
                ups.append(up)
        _add_route(gw_id, ups, paths=paths, hosts=[host],
                   name=meta.get("name", ""), source_id=source_id, ents=ents, rels=rels)


def _ambassador_mapping(m: dict, source_id: str, ents: dict, rels: dict) -> None:
    meta = m.get("metadata") or {}
    ns = sanitize_value(meta.get("namespace") or "default")
    spec = m.get("spec") or {}
    gw_id = _gateway_node("ambassador", "ambassador", source_id, ents)
    up = _upstream_node(spec.get("service", ""), source_id, ents, rels, default_ns=ns)
    _add_route(gw_id, up, paths=[spec.get("prefix") or "/"], hosts=[spec.get("host") or ""],
               name=meta.get("name", ""), source_id=source_id, ents=ents, rels=rels)


def _apisix_route(m: dict, source_id: str, ents: dict, rels: dict) -> None:
    meta = m.get("metadata") or {}
    ns = sanitize_value(meta.get("namespace") or "default")
    gw_id = _gateway_node("apisix", "apisix", source_id, ents)
    for http in (m.get("spec") or {}).get("http") or []:
        if not isinstance(http, dict):
            continue
        match = http.get("match") or {}
        hosts = [sanitize_value(h) for h in (match.get("hosts") or [])]
        paths = match.get("paths") or ["/"]
        ups = []
        for b in http.get("backends") or []:
            sn = (b or {}).get("serviceName", "")
            up = _upstream_node(sn, source_id, ents, rels, default_ns=ns) if sn else None
            if up:
                ups.append(up)
        plugins = list((http.get("plugins") or {}).keys()) if isinstance(http.get("plugins"), dict) else \
            [p.get("name") for p in (http.get("plugins") or []) if isinstance(p, dict) and p.get("name")]
        _add_route(gw_id, ups, paths=paths, hosts=hosts,
                   name=http.get("name") or meta.get("name", ""),
                   source_id=source_id, ents=ents, rels=rels,
                   middlewares=plugins, mw_type="apisix-plugin")


def extract_routing(manifests: list, source_id: str) -> tuple[dict, dict]:
    """Dispatch each YAML doc to its provider handler. Returns (entities, rels)
    keyed by id / (src,tgt,type) so repeated entities/edges de-dupe."""
    ents: dict = {}
    rels: dict = {}
    for m in manifests:
        if not isinstance(m, dict):
            continue
        api = (m.get("apiVersion") or "").lower()
        kind = m.get("kind") or ""
        try:
            if not kind and (m.get("services") or m.get("routes")):
                _kong_declarative(m, source_id, ents, rels)
            elif kind == "Ingress":
                _ingress(m, source_id, ents, rels)
            # Exact-match the apiVersion GROUP (the part before "/"): apiVersion
            # is "group/version", so comparing the parsed group with == is both
            # correct AND clears CodeQL's incomplete-url-substring-sanitization
            # finding (a substring/`in` check on a domain-like string trips it).
            # Applies to istio, gateway-api, contour and ambassador below.
            elif api.split("/", 1)[0] == "networking.istio.io" and kind == "VirtualService":
                _istio_virtualservice(m, source_id, ents, rels)
            elif api.split("/", 1)[0] == "networking.istio.io" and kind == "Gateway":
                _istio_gateway(m, source_id, ents, rels)
            elif api.split("/", 1)[0] == "networking.istio.io" and kind == "ServiceEntry":
                _istio_serviceentry(m, source_id, ents, rels)
            elif api.split("/", 1)[0] == "gateway.networking.k8s.io" and kind in (
                "HTTPRoute", "GRPCRoute", "TCPRoute", "TLSRoute"
            ):
                _gatewayapi_route(m, source_id, ents, rels)
            elif api.split("/", 1)[0] == "gateway.networking.k8s.io" and kind == "Gateway":
                _gateway_node(f"gatewayapi:{sanitize_value((m.get('metadata') or {}).get('name',''))}",
                              "gateway-api", source_id, ents)
            elif ("traefik" in api) and kind in ("IngressRoute", "IngressRouteTCP"):
                _traefik_ingressroute(m, source_id, ents, rels)
            elif api.split("/", 1)[0] == "projectcontour.io" and kind == "HTTPProxy":
                _contour_httpproxy(m, source_id, ents, rels)
            elif api.split("/", 1)[0] == "getambassador.io" and kind == "Mapping":
                _ambassador_mapping(m, source_id, ents, rels)
            elif "apisix" in api and kind == "ApisixRoute":
                _apisix_route(m, source_id, ents, rels)
        except Exception:
            # Routing extraction is best-effort + non-fatal -- a malformed
            # manifest must never break the (already-completed) vector index.
            continue

    # Cluster lane: if the source path encodes a cluster/env (gitops layout),
    # attach every Service/Route from this file to it.
    cluster = cluster_from_source(source_id)
    if cluster and ents:
        cid = _eid("Cluster", cluster)
        ents.setdefault(
            cid, Entity(id=cid, label="Cluster", name=cluster, properties={}, source_chunk_id=source_id)
        )
        for e in list(ents.values()):
            if e.label in ("Service", "Route"):
                _rel(rels, e.id, cid, "IN_CLUSTER", source_id)
    return ents, rels
