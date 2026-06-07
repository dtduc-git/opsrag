"""Kubernetes manifest parser -- handles multi-doc YAML manifests.

Each manifest (Deployment, Service, ConfigMap, etc.) becomes a section.
Extracts metadata, labels, container specs for downstream entity extraction.
"""
from __future__ import annotations

import yaml

from opsrag.interfaces.parser import DocSection, DocType, ParsedDocument
from opsrag.interfaces.scm import RepoFile

_K8S_KINDS = {
    "Deployment", "StatefulSet", "DaemonSet", "ReplicaSet",
    "Service", "Ingress", "ConfigMap", "Secret",
    "CronJob", "Job", "Pod", "Namespace",
    "ServiceAccount", "ClusterRole", "ClusterRoleBinding",
    "Role", "RoleBinding", "NetworkPolicy",
    "PersistentVolumeClaim", "PersistentVolume",
    "HorizontalPodAutoscaler", "PodDisruptionBudget",
}

_K8S_PATH_HINTS = ("k8s/", "kubernetes/", "manifests/", "/deploy/", "/base/", "/overlays/")


class K8sManifestParser:
    def supports(self, file_path: str, content: str) -> bool:
        if not file_path.lower().endswith((".yaml", ".yml")):
            return False
        low = file_path.lower()
        if any(h in low for h in _K8S_PATH_HINTS):
            return True
        # Detect by content: look for apiVersion + kind
        return "apiVersion:" in content and "kind:" in content

    def detect_doc_type(self, file: RepoFile) -> DocType:
        return DocType.KUBERNETES

    def parse(self, file: RepoFile) -> ParsedDocument:
        try:
            docs = list(yaml.safe_load_all(file.content))
        except Exception:
            docs = []

        sections: list[DocSection] = []
        for manifest in docs:
            if not isinstance(manifest, dict):
                continue
            kind = manifest.get("kind", "Unknown")
            meta = manifest.get("metadata", {}) or {}
            name = meta.get("name", "unnamed")
            namespace = meta.get("namespace", "default")
            heading = f"{kind}/{namespace}/{name}"

            # sort_keys=False: keep the manifest's authored key order so the
            # chunk text tracks the source for BM25 exact-match (sorting reorders
            # `replicas`/`image`/`env` away from how operators wrote/grep them).
            # Comments are still lost -- PyYAML can't round-trip them.
            body = yaml.dump(manifest, default_flow_style=False, sort_keys=False).strip()

            section_type = self._classify_kind(kind)
            sections.append(DocSection(
                heading=heading,
                content=body,
                level=1,
                section_type=section_type,
            ))

        if not sections:
            sections = [DocSection(
                heading=file.path, content=file.content, level=0,
                section_type="k8s_raw",
            )]

        title = sections[0].heading if sections else file.path
        return ParsedDocument(
            content=file.content,
            doc_type=DocType.KUBERNETES,
            title=title,
            source=file,
            metadata={
                "repo": file.repo,
                "branch": file.branch,
                "path": file.path,
                "sha": file.sha,
                "manifest_count": len(sections),
            },
            sections=sections,
            references=[],
        )

    @staticmethod
    def _classify_kind(kind: str) -> str:
        workloads = {"Deployment", "StatefulSet", "DaemonSet", "ReplicaSet", "CronJob", "Job", "Pod"}
        networking = {"Service", "Ingress", "NetworkPolicy"}
        config = {"ConfigMap", "Secret"}
        rbac = {"ServiceAccount", "ClusterRole", "ClusterRoleBinding", "Role", "RoleBinding"}
        storage = {"PersistentVolumeClaim", "PersistentVolume"}

        if kind in workloads:
            return "k8s_workload"
        if kind in networking:
            return "k8s_networking"
        if kind in config:
            return "k8s_config"
        if kind in rbac:
            return "k8s_rbac"
        if kind in storage:
            return "k8s_storage"
        return "k8s_other"
