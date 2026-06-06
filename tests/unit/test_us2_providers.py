"""Verification of the US2 optional provider extensions (T096-T100).

These providers are carried forward from upstream and wired into the factory;
they require optional extras (google-cloud, boto3, pgvector, neo4j, cohere) so
the imports skip cleanly when an extra is absent.
"""
from __future__ import annotations

import importlib
import inspect

import pytest


def _import_or_skip(module: str):
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as exc:
        if exc.name and not exc.name.startswith("opsrag"):
            pytest.skip(f"optional dependency not installed: {exc.name}")
        raise


@pytest.mark.parametrize(
    "module,cls",
    [
        ("opsrag.llms.vertex", "VertexAILLM"),        # T096
        ("opsrag.llms.bedrock", "BedrockLLM"),         # T096
        ("opsrag.embedders.vertex", "VertexAIEmbeddings"),   # T097
        ("opsrag.embedders.bedrock", "BedrockEmbeddings"),   # T097
        ("opsrag.vectorstores.pgvector", "PgVectorStore"),   # T098
        ("opsrag.graphstores.neo4j", "Neo4jGraphStore"),     # T099
        ("opsrag.rerankers.cohere", "CohereReranker"),       # T100
        ("opsrag.rerankers.noop", "NoOpReranker"),           # T100
    ],
)
def test_provider_extension_imports(module: str, cls: str) -> None:
    mod = _import_or_skip(module)
    assert hasattr(mod, cls), f"{module} missing {cls}"


def test_factory_wires_optional_providers() -> None:
    import opsrag.factory as factory

    src = inspect.getsource(factory.build_providers)
    # Each optional provider has a discriminated branch in the factory.
    for needle in (
        'provider == "vertex"',
        'provider == "bedrock"',
        'provider == "pgvector"',
        'provider == "neo4j"',
        'provider == "cohere"',
    ):
        assert needle in src, f"factory missing branch: {needle}"


def test_fastembed_reranker_present() -> None:
    mod = _import_or_skip("opsrag.rerankers.fastembed_reranker")
    assert hasattr(mod, "FastEmbedReranker")
