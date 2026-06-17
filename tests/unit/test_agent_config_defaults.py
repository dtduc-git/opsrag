"""Focused defaults for the AgentConfig retrieval/grounding spine.

These pin the F12 (MMR-on + content-dedup) and F6 (default-path grounding)
config defaults so a silent regression of the shipped behaviour fails fast.
"""
from opsrag.config import (
    AgentConfig,
    ElasticsearchConfig,
    K8sConfig,
    LLMConfig,
    Settings,
)


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


def test_agent_mode_default_is_full():
    # config-spine L15: schema default matches the bundled config.yaml
    # (mode: full), so an env/YAML-less load picks the shipped path. This is a
    # deploy-footgun fix, not a behaviour change for configured deployments
    # (config.yaml already sets mode: full).
    assert AgentConfig().mode == "full"
    assert Settings().agent.mode == "full"


def test_agent_mode_hybrid_still_accepted_for_back_compat():
    # "hybrid" stays valid in the Literal for back-compat; server.py warns and
    # maps hybrid->full (a different track). Construction must not raise.
    assert AgentConfig(mode="hybrid").mode == "hybrid"


def test_llm_client_timeout_and_retry_defaults():
    # config-spine: bounded provider client tail latency. These are robustness
    # knobs (timeout/retry), not answer-shaping params -- defaults pinned so a
    # silent regression of the shipped values fails fast.
    c = LLMConfig()
    assert c.request_timeout == 120.0
    assert c.connect_timeout == 10.0
    assert c.max_retries == 2
    # And the same through the Settings spine.
    sc = Settings().llm
    assert sc.request_timeout == 120.0
    assert sc.connect_timeout == 10.0
    assert sc.max_retries == 2


def test_llm_timeout_fields_are_floats_and_overridable():
    # Overridable via construction (env/YAML) without a rebuild; types coerce.
    c = LLMConfig(request_timeout=30, connect_timeout=5, max_retries=4)
    assert c.request_timeout == 30.0
    assert isinstance(c.request_timeout, float)
    assert c.connect_timeout == 5.0
    assert isinstance(c.connect_timeout, float)
    assert c.max_retries == 4


def test_legacy_live_tool_blocks_marked_deprecated():
    # F10(c): the legacy live-tool connection blocks point operators at
    # environments.targets.
    assert "DEPRECATED: prefer environments.targets" in (K8sConfig.__doc__ or "")
    assert "DEPRECATED: prefer environments.targets" in (
        ElasticsearchConfig.__doc__ or ""
    )
