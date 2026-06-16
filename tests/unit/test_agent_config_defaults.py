"""Focused defaults for the AgentConfig retrieval/grounding spine.

These pin the F12 (MMR-on + content-dedup) and F6 (default-path grounding)
config defaults so a silent regression of the shipped behaviour fails fast.
"""
from opsrag.config import AgentConfig, ElasticsearchConfig, K8sConfig, Settings


def test_mmr_diversity_on_by_default():
    # F12: MMR diversity is enabled by default (was 0.0 / disabled).
    assert AgentConfig().rerank_diversity == 0.3
    assert Settings().agent.rerank_diversity == 0.3


def test_content_dedup_defaults():
    # F12: exact-duplicate dedup on; near-duplicate Jaccard merge off.
    a = AgentConfig()
    assert a.rerank_content_dedup is True
    assert a.rerank_content_dedup_threshold == 0.0


def test_verify_grounding_default_on():
    # F6: the fail-closed groundedness gate runs on the default path.
    assert AgentConfig().verify_grounding_default is True


def test_legacy_live_tool_blocks_marked_deprecated():
    # F10(c): the legacy live-tool connection blocks point operators at
    # environments.targets.
    assert "DEPRECATED: prefer environments.targets" in (K8sConfig.__doc__ or "")
    assert "DEPRECATED: prefer environments.targets" in (
        ElasticsearchConfig.__doc__ or ""
    )
