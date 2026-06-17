"""Kubernetes manifest parser -- handles multi-doc YAML manifests.

Each manifest (Deployment, Service, ConfigMap, etc.) becomes a section.
Extracts metadata, labels, container specs for downstream entity extraction.
"""
from __future__ import annotations

from io import StringIO

from ruamel.yaml import YAML

from opsrag.interfaces.parser import DocSection, DocType, ParsedDocument
from opsrag.interfaces.scm import RepoFile

# Round-trip loader: preserves comments + anchors attached to nodes so ops
# rationale (e.g. `# bumped to 5 replicas after incident X`) survives into the
# emitted chunk text. PyYAML's safe_load/dump discarded both.
_YAML = YAML(typ="rt")
_YAML.preserve_quotes = True


def _dump_node(node: object) -> str:
    """Round-trip a (sub-)node back to YAML text, comments + anchors intact.

    ruamel only dumps to a stream, so route through StringIO. Comments stay
    attached to the CommentedMap/CommentedSeq node, so re-dumping a doc (or a
    helm/alert sub-node) re-emits them.
    """
    buf = StringIO()
    _YAML.dump(node, buf)
    return buf.getvalue().strip()


def _dump_key_slice(parent: object, key: object, value: object) -> str:
    """Dump a single ``{key: value}`` chunk, preserving that key's comment.

    A comment written *next to a top-level key* (``replicas: 5  # rationale``)
    is stored on the PARENT CommentedMap's ``.ca.items[key]``, not on the value
    node. A naive ``{key: value}`` dump would drop it. So we build a one-key
    CommentedMap and copy the parent's comment association for this key across.
    The value sub-node keeps its own attached comments/anchors automatically.

    Falls back to a plain ``key: value`` line for unrepresentable values.
    """
    try:
        from ruamel.yaml.comments import CommentedMap

        slice_map = CommentedMap()
        slice_map[key] = value
        # Carry the inline / preceding comment attached to this key on the
        # parent across to the slice so it re-emits next to the key.
        parent_ca = getattr(parent, "ca", None)
        if parent_ca is not None and key in getattr(parent_ca, "items", {}):
            slice_map.ca.items[key] = parent_ca.items[key]
        return _dump_node(slice_map)
    except Exception:
        return f"{key}: {value}"

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
            docs = list(_YAML.load_all(file.content))
        except Exception:
            docs = []

        sections: list[DocSection] = []
        for manifest in docs:
            # ruamel returns CommentedMap (a dict subclass); `.get()`/iteration
            # work unchanged. Skip non-mapping docs (lists/scalars/None).
            if not isinstance(manifest, dict):
                continue
            kind = manifest.get("kind", "Unknown")
            meta = manifest.get("metadata", {}) or {}
            name = meta.get("name", "unnamed")
            namespace = meta.get("namespace", "default")
            heading = f"{kind}/{namespace}/{name}"

            # Round-trip dump keeps the manifest's authored key order (so the
            # chunk text tracks the source for BM25 exact-match) AND the inline
            # / standalone comments + anchors attached to this doc -- the ops
            # rationale operators write next to `replicas`/`image`/`env`.
            body = _dump_node(manifest)

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
