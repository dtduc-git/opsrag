"""Build the routing/topology graph from LIVE Kubernetes state.

The industry-standard source for an infra knowledge graph is the cluster's
*actual* state (rendered objects), not Helm chart templates (which don't
parse) -- see AWS DevOps Agent "learned topology", Cartography, KubeVela, and
the Rendered-Manifests pattern. The good news: the rendered objects the K8s
API returns (Ingress / VirtualService / HTTPRoute / Service / ...) are the
EXACT shapes ``opsrag.extractors.routing.extract_routing`` already handles, so
this module just feeds cluster objects through the same extractor.

Inputs accepted (same code path):
  - a list of K8s object dicts straight from the API (the live lister), or
  - a ``kubectl get ingress,svc,virtualservices,httproutes,gateways -A -o yaml``
    dump (a ``{"kind":"List","items":[...]}`` envelope or a bare list).

Everything is attributed to ``source_id = "cluster:<name>"`` so re-ingesting a
cluster reference-counts cleanly (delete_by_source then re-upsert).
"""
from __future__ import annotations

from typing import Any

from opsrag.extractors.routing import _eid, _rel, extract_routing
from opsrag.interfaces.graphstore import Entity


def normalize_objects(payload: Any) -> list[dict]:
    """Accept a bare list, a kubectl ``List`` envelope, or a single object."""
    if isinstance(payload, dict):
        if payload.get("kind") == "List" and isinstance(payload.get("items"), list):
            return [o for o in payload["items"] if isinstance(o, dict)]
        if "kind" in payload:
            return [payload]
        return []
    if isinstance(payload, list):
        return [o for o in payload if isinstance(o, dict)]
    return []


def build_cluster_graph(objects: list[dict], cluster: str) -> tuple[list, list]:
    """Run the routing extractor over live cluster objects and attach a
    Cluster node. Returns (entities, relationships)."""
    source_id = f"cluster:{cluster}"
    ents, rels = extract_routing(objects, source_id)
    if ents:
        cid = _eid("Cluster", cluster)
        ents.setdefault(
            cid,
            Entity(id=cid, label="Cluster", name=cluster, properties={}, source_chunk_id=source_id),
        )
        for e in list(ents.values()):
            if e.label in ("Service", "Route", "Gateway"):
                _rel(rels, e.id, cid, "IN_CLUSTER", source_id)
    return list(ents.values()), list(rels.values())


async def ingest_cluster(objects: list[dict], cluster: str, graph_store: Any) -> dict:
    """Reference-counted refresh of one cluster's routing graph."""
    objs = normalize_objects(objects)
    ents, rels = build_cluster_graph(objs, cluster)
    try:
        await graph_store.delete_by_source([f"cluster:{cluster}"])
    except Exception:
        pass
    n = await graph_store.upsert_entities(ents) if ents else 0
    m = await graph_store.upsert_relationships(rels) if rels else 0
    return {
        "cluster": cluster,
        "objects_ingested": len(objs),
        "entities": n if isinstance(n, int) else len(ents),
        "relationships": m if isinstance(m, int) else len(rels),
    }
