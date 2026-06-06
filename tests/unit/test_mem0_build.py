"""Unit tests for `build_mem0_store` config construction.

`mem0.Memory.from_config` is monkeypatched so no live Qdrant / LLM is needed.
We assert the config dict reuses the project's Qdrant collection, maps provider
names, threads `infer`, and leaves the graph store OFF.
"""
from __future__ import annotations

import sys
import types

from opsrag.config import OpsRAGConfig
from opsrag.memory.mem0_store import Mem0ServiceMemory, build_mem0_store


def _install_fake_mem0(monkeypatch):
    """Install a fake `mem0` module exposing `Memory.from_config`."""
    captured = {}

    class _FakeMemory:
        @staticmethod
        def from_config(config_dict):
            captured["config"] = config_dict
            return object()  # stand-in mem0 instance

    fake_mod = types.ModuleType("mem0")
    fake_mod.Memory = _FakeMemory
    monkeypatch.setitem(sys.modules, "mem0", fake_mod)
    return captured


def test_build_reuses_qdrant_collection_and_graph_off(monkeypatch):
    captured = _install_fake_mem0(monkeypatch)
    cfg = OpsRAGConfig()
    store = build_mem0_store(
        cfg,
        qdrant_client_or_url=None,
        llm_cfg=cfg.llm,
        embed_cfg=cfg.embedding,
    )
    assert isinstance(store, Mem0ServiceMemory)
    conf = captured["config"]
    # Qdrant reused with the configured mem0 collection.
    assert conf["vector_store"]["provider"] == "qdrant"
    assert conf["vector_store"]["config"]["collection_name"] == cfg.memory.mem0_collection
    # No graph store key -> graph OFF.
    assert "graph_store" not in conf


def test_build_maps_provider_names(monkeypatch):
    captured = _install_fake_mem0(monkeypatch)
    cfg = OpsRAGConfig()
    cfg.llm.provider = "vertex"
    cfg.embedding.provider = "vertex"
    build_mem0_store(cfg, None, cfg.llm, cfg.embedding)
    conf = captured["config"]
    # anthropic/openai map straight; vertex -> gemini (llm) / vertexai (embed).
    assert conf["llm"]["provider"] == "gemini"
    assert conf["embedder"]["provider"] == "vertexai"


def test_build_threads_infer_from_config(monkeypatch):
    _install_fake_mem0(monkeypatch)
    cfg = OpsRAGConfig()
    cfg.memory.mem0_infer = False
    store = build_mem0_store(cfg, None, cfg.llm, cfg.embedding)
    assert store._infer is False


def test_build_prefers_injected_client_over_url(monkeypatch):
    captured = _install_fake_mem0(monkeypatch)
    cfg = OpsRAGConfig()
    sentinel_client = object()
    build_mem0_store(cfg, sentinel_client, cfg.llm, cfg.embedding)
    qconf = captured["config"]["vector_store"]["config"]
    assert qconf.get("client") is sentinel_client
    assert "url" not in qconf


def test_build_uses_url_string_when_given(monkeypatch):
    captured = _install_fake_mem0(monkeypatch)
    cfg = OpsRAGConfig()
    build_mem0_store(cfg, "http://qdrant:6333", cfg.llm, cfg.embedding)
    qconf = captured["config"]["vector_store"]["config"]
    assert qconf.get("url") == "http://qdrant:6333"
    assert "client" not in qconf
