"""R7: YAML comments + anchors survive ingestion so ops rationale is indexed.

The k8s/Helm/alert/generic YAML parsers switched from PyYAML safe_load/dump
(which DROPS comments + anchors) to ruamel.yaml round-trip. A comment like
``# bumped to 5 replicas after incident X`` written next to ``replicas: 5``
must now appear in the emitted chunk text so it gets embedded + BM25'd.

These tests assert the comment lands in the chunk text while the EXISTING
chunking structure (split by doc / key / monitor) is preserved.
"""
from __future__ import annotations

from datetime import UTC, datetime

from opsrag.interfaces.scm import RepoFile
from opsrag.parsers.alert import AlertParser
from opsrag.parsers.generic import GenericConfigParser
from opsrag.parsers.helm import HelmParser
from opsrag.parsers.k8s import K8sManifestParser


def _rf(path: str, content: str) -> RepoFile:
    return RepoFile(
        path=path,
        content=content,
        sha="x",
        last_modified=datetime(2026, 1, 1, tzinfo=UTC),
        repo="r/p",
        branch="main",
    )


# --------------------------------------------------------------------------- #
# k8s manifests                                                               #
# --------------------------------------------------------------------------- #


_K8S_WITH_COMMENT = """\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api
  namespace: prod
spec:
  replicas: 5  # bumped to 5 replicas after incident X
  template:
    spec:
      containers:
        - name: api
          image: registry/api:1.2.3
"""


def test_k8s_inline_comment_lands_in_chunk():
    doc = K8sManifestParser().parse(_rf("k8s/api/deployment.yaml", _K8S_WITH_COMMENT))
    assert len(doc.sections) == 1
    body = doc.sections[0].content
    assert "replicas: 5" in body
    assert "bumped to 5 replicas after incident X" in body


def test_k8s_multidoc_still_splits_per_doc():
    multi = _K8S_WITH_COMMENT + "---\n" + (
        "apiVersion: v1\n"
        "kind: Service  # internal LB only\n"
        "metadata:\n"
        "  name: api\n"
        "  namespace: prod\n"
        "spec:\n"
        "  ports:\n"
        "    - port: 80\n"
    )
    doc = K8sManifestParser().parse(_rf("k8s/api/all.yaml", multi))
    # Two documents -> two sections, each per-doc comment preserved.
    assert len(doc.sections) == 2
    headings = [s.heading for s in doc.sections]
    assert headings == ["Deployment/prod/api", "Service/prod/api"]
    assert "bumped to 5 replicas after incident X" in doc.sections[0].content
    assert "internal LB only" in doc.sections[1].content


def test_k8s_anchor_survives():
    anchored = """\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api
  namespace: prod
spec:
  template:
    metadata:
      labels: &lbls
        app: api  # canonical label set
    spec:
      selector: *lbls
"""
    doc = K8sManifestParser().parse(_rf("k8s/api/deployment.yaml", anchored))
    body = doc.sections[0].content
    # ruamel re-emits the anchor (&lbls) + alias (*lbls) instead of inlining.
    assert "&lbls" in body
    assert "*lbls" in body
    assert "canonical label set" in body


def test_k8s_empty_and_non_mapping_docs_skipped():
    # Leading empty doc + a bare-list doc -> both skipped, only the Deployment
    # mapping becomes a section (structure logic unchanged).
    content = "---\n---\n" + _K8S_WITH_COMMENT + "---\n- just\n- a\n- list\n"
    doc = K8sManifestParser().parse(_rf("k8s/api/deployment.yaml", content))
    assert len(doc.sections) == 1
    assert "bumped to 5 replicas after incident X" in doc.sections[0].content


# --------------------------------------------------------------------------- #
# Helm values.yaml                                                            #
# --------------------------------------------------------------------------- #


_VALUES_WITH_COMMENT = """\
replicaCount: 5  # bumped after the Black Friday incident
image:
  repository: registry/api
  tag: "1.2.3"  # pinned -- do not float to latest
resources:
  limits:
    cpu: 500m
"""


def test_helm_values_chunks_per_key_with_comment():
    doc = HelmParser().parse(_rf("charts/api/values.yaml", _VALUES_WITH_COMMENT))
    by_heading = {s.heading: s.content for s in doc.sections}
    # Per-top-level-key split preserved.
    assert set(by_heading) == {"replicaCount", "image", "resources"}
    # Inline comment on a SCALAR top-level key (stored on parent map) survives.
    assert "bumped after the Black Friday incident" in by_heading["replicaCount"]
    # Inline comment nested inside a mapping value survives too.
    assert "pinned -- do not float to latest" in by_heading["image"]
    assert "repository: registry/api" in by_heading["image"]


def test_helm_values_section_types_unchanged():
    doc = HelmParser().parse(_rf("charts/api/values.yaml", _VALUES_WITH_COMMENT))
    types = {s.heading: s.section_type for s in doc.sections}
    assert types["image"] == "values_image"
    assert types["resources"] == "values_resources"


# --------------------------------------------------------------------------- #
# Alert definitions                                                           #
# --------------------------------------------------------------------------- #


_DATADOG_MONITORS = """\
monitors:
  - name: api-5xx
    type: metric alert
    query: "avg(last_5m):sum:api.5xx{*} > 10"  # threshold raised post-incident X
    message: "API 5xx spike"
"""


def test_alert_monitor_chunks_with_comment():
    doc = AlertParser().parse(_rf("monitoring/datadog.yaml", _DATADOG_MONITORS))
    assert len(doc.sections) == 1
    body = doc.sections[0].content
    assert doc.sections[0].heading == "monitor: api-5xx"
    assert "threshold raised post-incident X" in body
    assert "api.5xx" in body


# --------------------------------------------------------------------------- #
# Generic YAML fallback                                                       #
# --------------------------------------------------------------------------- #


def test_generic_yaml_key_split_keeps_comment():
    content = (
        "stages:\n"
        "  - build  # only build on tags now (was build+deploy)\n"
        "  - deploy\n"
        "variables:\n"
        "  RETRIES: 3  # was 1, bumped after flaky deploys\n"
    )
    doc = GenericConfigParser().parse(_rf(".gitlab-ci.yml", content))
    by_heading = {s.heading: s.content for s in doc.sections}
    assert set(by_heading) == {"stages", "variables"}
    assert "only build on tags now" in by_heading["stages"]
    assert "bumped after flaky deploys" in by_heading["variables"]
