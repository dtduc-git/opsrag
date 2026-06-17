"""Smoke/conformance tests for the US1 provider set (T046-T055).

These verify the ported provider modules import and that the concrete
classes satisfy their interface protocols, without standing up any external
service. Providers gated behind optional extras (e.g. fastembed) are skipped
when the extra isn't installed in the current environment.
"""
from __future__ import annotations

import importlib

import pytest

from opsrag.interfaces.llm import LLMProvider


def _import_or_skip(module: str):
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as exc:
        # Distinguish "our module missing" (a real failure) from "optional
        # third-party dep missing" (skip).
        if exc.name and not exc.name.startswith("opsrag"):
            pytest.skip(f"optional dependency not installed: {exc.name}")
        raise


# ---- LLM providers (T046, T047) -------------------------------------------
def test_anthropic_llm_conforms() -> None:
    from opsrag.llms.anthropic import AnthropicLLM

    llm = AnthropicLLM(api_key="test")
    assert isinstance(llm, LLMProvider)
    assert llm.model_name


def test_openai_llm_conforms() -> None:
    from opsrag.llms.openai import OpenAILLM

    llm = OpenAILLM(api_key="test")
    assert isinstance(llm, LLMProvider)
    assert llm.model_name


def test_factory_has_openai_llm_branch() -> None:
    import inspect

    import opsrag.factory as factory

    src = inspect.getsource(factory.build_providers)
    assert 'config.llm.provider == "openai"' in src


# ---- Embedders (T048, T049) -----------------------------------------------
def test_openai_embedder_imports() -> None:
    mod = _import_or_skip("opsrag.embedders.openai")
    assert hasattr(mod, "OpenAIEmbeddings")


def test_fastembed_embedder_imports() -> None:
    mod = _import_or_skip("opsrag.embedders.fastembed")
    assert hasattr(mod, "FastEmbedEmbeddings")


# ---- Vector store (T050) ---------------------------------------------------
def test_qdrant_vectorstore_imports() -> None:
    from opsrag.vectorstores.qdrant import QdrantVectorStore

    assert QdrantVectorStore is not None


# ---- Sessions + memory (T051, T052) ---------------------------------------
@pytest.mark.parametrize(
    "module,cls",
    [
        ("opsrag.sessions.postgres", "PostgresSessionStore"),
        ("opsrag.sessions.memory", "InMemorySessionStore"),
        ("opsrag.memory.postgres", "PostgresMemoryStore"),
        ("opsrag.memory.memory", "InMemoryMemoryStore"),
    ],
)
def test_session_and_memory_stores_import(module: str, cls: str) -> None:
    mod = _import_or_skip(module)
    assert hasattr(mod, cls)


# ---- Chunkers (T053) -------------------------------------------------------
@pytest.mark.parametrize(
    "module,cls",
    [
        ("opsrag.chunkers.fixed_size", "FixedSizeChunker"),
        ("opsrag.chunkers.parent_child", "ParentChildChunker"),
    ],
)
def test_chunkers_import(module: str, cls: str) -> None:
    mod = _import_or_skip(module)
    assert hasattr(mod, cls)


# ---- Parsers (T054) --------------------------------------------------------
@pytest.mark.parametrize(
    "module,cls",
    [
        ("opsrag.parsers.markdown", "GenericMarkdownParser"),
        ("opsrag.parsers.runbook", "RunbookParser"),
        ("opsrag.parsers.postmortem", "PostmortemParser"),
    ],
)
def test_parsers_import(module: str, cls: str) -> None:
    mod = _import_or_skip(module)
    assert hasattr(mod, cls)


# ---- Agent graph (T055) ----------------------------------------------------
def test_agent_graph_builders_exist() -> None:
    from opsrag.agent import graph

    # At least the minimal + full builders the server wires must exist.
    assert hasattr(graph, "build_minimal_graph")
    assert hasattr(graph, "build_full_graph")
