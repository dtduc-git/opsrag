"""Config-spine defaults for H4 (vector distance metric) + M4 (QA-cache
judge/NER knobs).

These pin the newly-added config fields the sibling tracks depend on so a
silent rename/default-drift fails fast. They are pure schema assertions --
no providers, no I/O.
"""
from opsrag.config import (
    AgentConfig,
    QACacheConfig,
    Settings,
    VectorStoreConfig,
)

# --- H4: VectorStoreConfig.distance ---------------------------------------


def test_distance_default_is_cosine():
    # cosine default keeps all current deployments byte-identical.
    assert VectorStoreConfig().distance == "cosine"
    assert Settings().vector_store.distance == "cosine"


def test_distance_accepts_dot_and_euclid():
    # The other two metrics the factory will wire into Qdrant + pgvector.
    assert VectorStoreConfig(distance="dot").distance == "dot"
    assert VectorStoreConfig(distance="euclid").distance == "euclid"


def test_distance_rejects_unknown_metric():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        VectorStoreConfig(distance="manhattan")


def test_distance_present_on_code_vector_store_lane():
    # The optional code lane reuses VectorStoreConfig, so the metric is
    # available there too (qdrant-only code lane).
    cvs = VectorStoreConfig(provider="qdrant", distance="dot")
    assert cvs.distance == "dot"


# --- M4: QACacheConfig judge/NER knobs ------------------------------------


def test_qa_judge_upper_default():
    # LLM judge runs across [floor, qa_judge_upper]; >= upper auto-accepts.
    assert QACacheConfig().qa_judge_upper == 0.99
    assert Settings().qa_cache.qa_judge_upper == 0.99


def test_qa_ner_guard_default_on():
    # spaCy NER entity-swap discriminator enabled by default; graceful-degrades
    # when the model is absent (a wiring/runtime concern, not config).
    assert QACacheConfig().qa_ner_guard is True
    assert Settings().qa_cache.qa_ner_guard is True


def test_qa_cache_knobs_overridable():
    # Tunable via construction (env/YAML) without a rebuild; types coerce.
    c = QACacheConfig(qa_judge_upper=0.95, qa_ner_guard=False)
    assert c.qa_judge_upper == 0.95
    assert isinstance(c.qa_judge_upper, float)
    assert c.qa_ner_guard is False


def test_qa_cache_block_on_settings_spine():
    # The block is threadable from the Settings root (server.py reads it).
    s = Settings()
    assert isinstance(s.qa_cache, QACacheConfig)


# --- R14: QACacheConfig.qa_judge_fail_open --------------------------------


def test_qa_judge_fail_open_default_closed():
    # QUALITY > latency: a borderline-band judge LLM error defaults to
    # re-query (fail-CLOSED) rather than serving a wrong-but-close cached
    # answer. Default must be False.
    assert QACacheConfig().qa_judge_fail_open is False
    assert Settings().qa_cache.qa_judge_fail_open is False


def test_qa_judge_fail_open_overridable():
    # Operators can opt into availability-over-correctness (fail-open).
    c = QACacheConfig(qa_judge_fail_open=True)
    assert c.qa_judge_fail_open is True


# --- R4: AgentConfig.verify_artifacts_default -----------------------------


def test_verify_artifacts_default_on():
    # Precise artifact/citation verify_answer runs on the multi_agent path by
    # default (+1 LLM verification call/turn). Distinct from the groundedness
    # gate; both default True (fail-closed, grounding-first).
    assert AgentConfig().verify_artifacts_default is True
    assert Settings().agent.verify_artifacts_default is True


def test_verify_artifacts_default_overridable():
    # Set false to drop the extra verification call (latency/cost trade-off).
    a = AgentConfig(verify_artifacts_default=False)
    assert a.verify_artifacts_default is False


def test_verify_grounding_default_still_on():
    # C1 guard: the pre-existing groundedness gate on multi_agent stays on and
    # is independent of the new artifact verify knob.
    assert AgentConfig().verify_grounding_default is True
    assert Settings().agent.verify_grounding_default is True
