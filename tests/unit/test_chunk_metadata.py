"""Unit tests for the generalized chunk-metadata feature.

Covers:
  - enrich_metadata derives doc_type / environment / tier / tags / language
    deterministically for representative paths (runbook md, helm prod
    values.yaml, postmortem, code file).
  - chunker sets content_hash + chunk_index/chunk_count + heading_path.
  - author hashing (default) + plaintext gate.
  - existing parser facets still present (helm_file_type, section types,
    runbook/postmortem flags).
"""
from __future__ import annotations

from datetime import UTC, datetime

from opsrag.chunkers.parent_child import ParentChildChunker
from opsrag.ingestion import metadata as md
from opsrag.ingestion.enrich import enrich_metadata
from opsrag.interfaces.parser import DocType
from opsrag.interfaces.scm import RepoFile
from opsrag.parsers.helm import HelmParser
from opsrag.parsers.markdown import GenericMarkdownParser as MarkdownParser
from opsrag.parsers.postmortem import PostmortemParser
from opsrag.parsers.runbook import RunbookParser


def _file(path: str, content: str, **kw) -> RepoFile:
    return RepoFile(
        path=path,
        content=content,
        sha=kw.get("sha", "deadbeef"),
        last_modified=kw.get("last_modified", datetime(2026, 4, 2, 11, 0, tzinfo=UTC)),
        repo=kw.get("repo", "org/repo"),
        branch=kw.get("branch", "main"),
        metadata=kw.get("metadata", {}),
    )


# --------------------------------------------------------------------------
# enrich_metadata: deterministic derivation
# --------------------------------------------------------------------------

def test_enrich_runbook_markdown():
    meta = {"section_heading": "Rollback procedure"}
    out = enrich_metadata(
        meta,
        path="docs/runbooks/checkout-rollback.md",
        text="Restart the redis pod and check latency.",
        source_type="git",
        struct_doc_type=DocType.RUNBOOK,
    )
    assert out["doc_type"] == "runbook"
    assert out["source_system"] == "git"
    assert out["language"] == "en"
    assert out["valid"] is True
    # tags sniffed deterministically from content.
    assert "redis" in out["tags"]
    assert "latency" in out["tags"]


def test_enrich_helm_prod_values():
    meta = {"helm_file_type": "values", "section_heading": "image"}
    out = enrich_metadata(
        meta,
        path="charts/payments/values-prod.yaml",
        text="image:\n  repository: payments\n  tag: 1.2.3",
        source_type="git",
        struct_doc_type=DocType.HELM,
    )
    assert out["doc_type"] == "helm_values"
    assert out["environment"] == "prod"
    assert out["language"] == "yaml"
    # existing helm facet untouched.
    assert out["helm_file_type"] == "values"


def test_enrich_postmortem_staging():
    meta = {"postmortem": True}
    out = enrich_metadata(
        meta,
        path="incidents/2026/staging-payments-outage-postmortem.md",
        text="Root cause: kafka deadlock.",
        source_type="git",
        struct_doc_type=DocType.POSTMORTEM,
    )
    # path convention (postmortem) wins for doc_type.
    assert out["doc_type"] == "postmortem"
    assert out["environment"] == "staging"
    assert "kafka" in out["tags"]


def test_enrich_code_file():
    meta = {}
    out = enrich_metadata(
        meta,
        path="services/checkout/app/auth.py",
        text="def authenticate(user): ...",
        source_type="git",
        struct_doc_type=DocType.PYTHON,
    )
    assert out["doc_type"] == "code"
    assert out["language"] == "python"
    assert out["valid"] is True


def test_enrich_archived_marks_invalid():
    out = enrich_metadata(
        {}, path="docs/archive/old-runbook.md", text="stale", source_type="git",
        struct_doc_type=DocType.RUNBOOK,
    )
    assert out["valid"] is False


def test_enrich_is_deterministic_and_idempotent():
    args = dict(path="docs/runbooks/r.md", text="redis", source_type="git",
                struct_doc_type=DocType.RUNBOOK)
    a = enrich_metadata({}, **args)
    b = enrich_metadata({}, **args)
    assert a == b
    # Re-running on an already-enriched dict is a no-op (idempotent).
    c = enrich_metadata(dict(a), **args)
    assert c == a


def test_enrich_never_clobbers_explicit_values():
    meta = {"doc_type": "wiki", "environment": "dev", "source_system": "confluence"}
    out = enrich_metadata(
        meta, path="charts/x/values-prod.yaml", text="", source_type="git",
        struct_doc_type=DocType.HELM,
    )
    assert out["doc_type"] == "wiki"          # explicit wins
    assert out["environment"] == "dev"        # explicit wins over path-derived env
    assert out["source_system"] == "confluence"


# --------------------------------------------------------------------------
# author hashing
# --------------------------------------------------------------------------

def test_hash_author_stable_and_prefixed():
    h1 = md.hash_author("Jane@Corp.com")
    h2 = md.hash_author("jane@corp.com ")
    assert h1 == h2                      # case + whitespace insensitive
    assert h1.startswith("anon:")
    assert "@" not in h1


def test_apply_author_hashes_by_default(monkeypatch):
    monkeypatch.delenv("OPSRAG_STORE_AUTHOR_PLAINTEXT", raising=False)
    meta = {}
    md.apply_author(meta, "jane@corp.com")
    assert meta["author"].startswith("anon:")
    assert meta["author_hashed"] is True


def test_apply_author_plaintext_when_gated(monkeypatch):
    monkeypatch.setenv("OPSRAG_STORE_AUTHOR_PLAINTEXT", "1")
    meta = {}
    md.apply_author(meta, "jane@corp.com")
    assert meta["author"] == "jane@corp.com"
    assert meta["author_hashed"] is False


def test_apply_author_noop_for_empty():
    meta = {}
    md.apply_author(meta, None)
    md.apply_author(meta, "")
    assert "author" not in meta


def test_content_hash_self_describing():
    h = md.content_hash("hello")
    assert h.startswith("sha256:")
    assert md.content_hash("hello") == h
    assert md.content_hash("world") != h


# --------------------------------------------------------------------------
# chunker tier: chunk_index/chunk_count/content_hash/heading_path
# --------------------------------------------------------------------------

def _parse_chunks(parser, file):
    doc = parser.parse(file)
    return ParentChildChunker().chunk(doc)


def test_chunker_sets_positional_and_hash_facets():
    content = (
        "# Checkout Runbook\n\n"
        "## Rollback procedure\n\nFirst restart the redis pod.\n\n"
        "## Escalation\n\nPage the on-call.\n"
    )
    chunks = _parse_chunks(RunbookParser(), _file("runbooks/checkout.md", content))
    assert chunks, "expected chunks"
    total = len(chunks)
    for i, c in enumerate(chunks):
        assert c.metadata["chunk_index"] == i
        assert c.metadata["chunk_count"] == total
        assert c.metadata["content_hash"].startswith("sha256:")
        assert isinstance(c.metadata.get("heading_path"), list)
    # heading_path breadcrumb includes the section heading on a parent.
    parents = [c for c in chunks if c.chunk_type == "parent"]
    assert any("Rollback procedure" in p.metadata["heading_path"] for p in parents)


def test_chunk_index_unique_and_contiguous():
    content = "# Doc\n\n" + ("para. " * 400)
    chunks = _parse_chunks(MarkdownParser(), _file("docs/x.md", content))
    idxs = sorted(c.metadata["chunk_index"] for c in chunks)
    assert idxs == list(range(len(chunks)))


# --------------------------------------------------------------------------
# existing parser facets must still be present (backward compat)
# --------------------------------------------------------------------------

def test_helm_existing_facets_preserved_plus_new():
    chart = "name: payments\nversion: 1.8.2\nappVersion: 2.0\ndescription: pay svc\n"
    doc = HelmParser().parse(_file("charts/payments/Chart.yaml", chart))
    assert doc.metadata["helm_file_type"] == "chart"
    assert doc.metadata["repo"] == "org/repo"
    assert doc.metadata["sha"] == "deadbeef"
    # new parser-tier facets
    assert doc.metadata["service"] == "payments"
    assert doc.metadata["version"] == "1.8.2"
    assert doc.metadata["source_system"] == "git"
    assert doc.metadata["updated_at"] == "2026-04-02T11:00:00+00:00"


def test_helm_values_subtypes_preserved():
    values = "image:\n  tag: v1\nresources:\n  limits:\n    cpu: 1\n"
    doc = HelmParser().parse(_file("charts/payments/values.yaml", values))
    types = {s.section_type for s in doc.sections}
    assert "values_image" in types
    assert "values_resources" in types


def test_runbook_and_postmortem_flags_preserved():
    rb = RunbookParser().parse(_file("ops/runbook.md", "# Runbook\n\nsteps"))
    assert rb.metadata["runbook"] is True
    pm = PostmortemParser().parse(_file("incidents/pm.md", "# Postmortem\n\nrca"))
    assert pm.metadata["postmortem"] is True


def test_markdown_provenance_facets():
    doc = MarkdownParser().parse(_file("services/checkout/README.md", "# Checkout\n\nhi"))
    assert doc.metadata["source_system"] == "git"
    assert doc.metadata["updated_at"].startswith("2026-04-02")
    # path-derived scalar service.
    assert doc.metadata["service"] == "checkout"


def test_author_from_source_metadata_is_hashed(monkeypatch):
    monkeypatch.delenv("OPSRAG_STORE_AUTHOR_PLAINTEXT", raising=False)
    f = _file("wiki/page.md", "# P\n\nx", metadata={"author": "jane@corp.com",
                                                    "source_type": "confluence",
                                                    "url": "https://wiki/p"})
    doc = MarkdownParser().parse(f)
    assert doc.metadata["author"].startswith("anon:")
    assert doc.metadata["author_hashed"] is True
    assert doc.metadata["source_system"] == "confluence"
    assert doc.metadata["url"] == "https://wiki/p"
