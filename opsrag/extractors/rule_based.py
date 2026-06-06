"""Rule-based entity extractor for structured files.

Extracts entities from Terraform, Kubernetes YAML, Helm values, and
Dockerfiles using regex/heuristics -- more reliable than LLM for
structured formats.
"""
from __future__ import annotations

import asyncio
import hashlib
import re

import yaml

from opsrag.extractors.routing import cluster_from_source, extract_routing
from opsrag.interfaces.entity_extractor import ExtractionResult
from opsrag.interfaces.graphstore import Entity, Relationship
from opsrag.interfaces.parser import DocType, ParsedDocument

_TF_RESOURCE_RE = re.compile(r'resource\s+"(\w+)"\s+"(\w+)"')
_TF_MODULE_RE = re.compile(r'module\s+"(\w+)"')
_K8S_SERVICE_RE = re.compile(r'name:\s*"?([a-z0-9][-a-z0-9]*)"?', re.I)
# Signal that a doc (possibly classified generic) is an ingress/mesh routing
# config worth running through the routing extractor.
_ROUTING_HINT_RE = re.compile(
    r"kind:\s*(Ingress|VirtualService|HTTPRoute|GRPCRoute|IngressRoute|HTTPProxy|Mapping|ApisixRoute|Gateway)"
    r"|_format_version|^\s*routes:|^\s*services:",
    re.I | re.M,
)


class RuleBasedExtractor:
    async def extract(
        self,
        doc: ParsedDocument,
        existing_entities: list[Entity] | None = None,
    ) -> ExtractionResult:
        # Internals do regex + yaml.safe_load_all which are CPU-bound and
        # blocking. Push to the thread pool so the event loop stays free
        # for FastAPI request handlers while indexing churns.
        return await asyncio.to_thread(self._extract_sync, doc)

    def _extract_sync(self, doc: ParsedDocument) -> ExtractionResult:
        dt = doc.doc_type
        source_id = f"{doc.source.repo}:{doc.source.path}"

        if dt == DocType.TERRAFORM:
            return self._extract_terraform(doc, source_id)
        if dt == DocType.KUBERNETES:
            return self._extract_k8s(doc, source_id)
        if dt == DocType.HELM:
            return self._extract_helm(doc, source_id)
        if dt == DocType.DOCKERFILE:
            return self._extract_dockerfile(doc, source_id)

        # Fallback: ingress/mesh routing configs (Kong decK, Istio, Gateway API,
        # Traefik, Contour, Ambassador, APISIX, or a bare Ingress) may not be
        # classified as KUBERNETES. Run the routing extractor when the content
        # carries a routing signal.
        if _ROUTING_HINT_RE.search(doc.content or ""):
            try:
                manifests = list(yaml.safe_load_all(doc.content))
            except Exception:
                manifests = []
            r_ents, r_rels = extract_routing(manifests, source_id)
            if r_ents or r_rels:
                return ExtractionResult(
                    entities=list(r_ents.values()),
                    relationships=list(r_rels.values()),
                )
        return ExtractionResult()

    def _extract_terraform(self, doc: ParsedDocument, source_id: str) -> ExtractionResult:
        entities: list[Entity] = []
        rels: list[Relationship] = []
        for rtype, rname in _TF_RESOURCE_RE.findall(doc.content):
            eid = self._eid("Config", f"tf:{rtype}.{rname}")
            entities.append(
                Entity(
                    id=eid, label="Config", name=f"{rtype}.{rname}",
                    properties={"provider": rtype.split("_")[0], "resource_type": rtype},
                    source_chunk_id=source_id,
                )
            )
        for mname in _TF_MODULE_RE.findall(doc.content):
            eid = self._eid("Config", f"tf:module.{mname}")
            entities.append(
                Entity(
                    id=eid, label="Config", name=f"module.{mname}",
                    properties={"type": "module"},
                    source_chunk_id=source_id,
                )
            )
        return ExtractionResult(entities=entities, relationships=rels)

    def _extract_k8s(self, doc: ParsedDocument, source_id: str) -> ExtractionResult:
        entities: list[Entity] = []
        rels: list[Relationship] = []
        try:
            manifests = list(yaml.safe_load_all(doc.content))
        except Exception:
            return ExtractionResult()

        for m in manifests:
            if not isinstance(m, dict):
                continue
            kind = m.get("kind", "")
            meta = m.get("metadata", {}) or {}
            name = meta.get("name", "")
            ns = meta.get("namespace", "default")
            if not name:
                continue

            if kind in ("Deployment", "StatefulSet", "DaemonSet", "Service"):
                eid = self._eid("Service", f"k8s:{ns}/{name}")
                entities.append(
                    Entity(
                        id=eid, label="Service", name=name,
                        properties={"kind": kind, "namespace": ns},
                        source_chunk_id=source_id,
                    )
                )
            elif kind in ("ConfigMap", "Secret"):
                eid = self._eid("Config", f"k8s:{ns}/{name}")
                entities.append(
                    Entity(
                        id=eid, label="Config", name=name,
                        properties={"kind": kind, "namespace": ns},
                        source_chunk_id=source_id,
                    )
                )

            spec = m.get("spec", {}) or {}
            containers = (spec.get("template", {}) or {}).get("spec", {}) or {}
            for c in containers.get("containers", []):
                image = c.get("image", "")
                if image:
                    repo_name = image.split(":")[0].rsplit("/", 1)[-1]
                    rid = self._eid("Repository", f"image:{repo_name}")
                    entities.append(
                        Entity(id=rid, label="Repository", name=repo_name,
                               properties={"image": image}, source_chunk_id=source_id)
                    )
                    svc_id = self._eid("Service", f"k8s:{ns}/{name}")
                    rels.append(Relationship(
                        source_id=svc_id, target_id=rid, rel_type="LIVES_IN",
                        properties={"source_chunk_id": source_id},
                    ))

        # Cluster lane: attach k8s workloads to the cluster/env from the path.
        cluster = cluster_from_source(source_id)
        if cluster:
            cid = self._eid("Cluster", cluster)
            entities.append(Entity(id=cid, label="Cluster", name=cluster,
                                   properties={}, source_chunk_id=source_id))
            for e in list(entities):
                if e.label == "Service":
                    rels.append(Relationship(source_id=e.id, target_id=cid,
                                             rel_type="IN_CLUSTER",
                                             properties={"source_chunk_id": source_id}))

        # Routing/topology lane: Ingress + mesh CRDs (Istio/Gateway API/Traefik/
        # Contour/Ambassador/APISIX) + Kong decK docs riding in the same YAML.
        r_ents, r_rels = extract_routing(manifests, source_id)
        entities.extend(r_ents.values())
        rels.extend(r_rels.values())
        return ExtractionResult(entities=entities, relationships=rels)

    def _extract_helm(self, doc: ParsedDocument, source_id: str) -> ExtractionResult:
        try:
            data = yaml.safe_load(doc.content) or {}
        except Exception:
            return ExtractionResult()
        if not isinstance(data, dict):
            return ExtractionResult()
        entities: list[Entity] = []
        name = data.get("nameOverride") or data.get("fullnameOverride") or doc.title
        if name:
            eid = self._eid("Config", f"helm:{name}")
            entities.append(
                Entity(id=eid, label="Config", name=name,
                       properties={"type": "helm_values"}, source_chunk_id=source_id)
            )
        return ExtractionResult(entities=entities)

    def _extract_dockerfile(self, doc: ParsedDocument, source_id: str) -> ExtractionResult:
        entities: list[Entity] = []
        for line in doc.content.splitlines():
            if line.strip().upper().startswith("FROM "):
                image = line.split()[1] if len(line.split()) > 1 else ""
                if image:
                    name = image.split(":")[0].rsplit("/", 1)[-1]
                    eid = self._eid("Repository", f"image:{name}")
                    entities.append(
                        Entity(id=eid, label="Repository", name=name,
                               properties={"base_image": image}, source_chunk_id=source_id)
                    )
        return ExtractionResult(entities=entities)

    @staticmethod
    def _eid(label: str, key: str) -> str:
        h = hashlib.sha1(key.encode()).hexdigest()[:12]
        return f"{label.lower()}:{key}:{h}"
