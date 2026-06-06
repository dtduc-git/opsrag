"""Unit tests for resolve_cloud_bundle (models feature, DESIGN 4)."""
from __future__ import annotations

from opsrag.config import ModelsConfig, ModelSpec, Settings
from opsrag.model_bundles import CLOUD_BUNDLES, resolve_cloud_bundle


def _fresh(**kwargs) -> Settings:
    return Settings(**kwargs)


def test_aws_bundle_fills_unset_slots():
    cfg = _fresh(cloud_provider="aws")
    resolve_cloud_bundle(cfg)

    # Classic llm slot <- bundle reason (strong model = Bedrock Sonnet 4.6).
    assert cfg.llm.provider == "bedrock"
    assert cfg.llm.model == "us.anthropic.claude-sonnet-4-6"

    # Classic embedding slot <- bundle embed (Cohere Embed v4).
    assert cfg.embedding.provider == "bedrock"
    assert cfg.embedding.model == "us.cohere.embed-v4:0"
    # Dimension filled to 1536 for aws.
    assert cfg.embedding.dimension == 1536

    # Classic reranker slot <- bundle rerank (Cohere -- no Bedrock reranker).
    assert cfg.reranker.provider == "bedrock"
    assert cfg.reranker.model == "cohere.rerank-v3-5:0"

    # Per-purpose models block populated.
    assert cfg.models is not None
    assert cfg.models.reason.provider == "bedrock"
    assert cfg.models.reason.model == "us.anthropic.claude-sonnet-4-6"
    assert cfg.models.tool_call.model == "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    assert cfg.models.summarize.model == "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    assert cfg.models.extract.model == "us.anthropic.claude-haiku-4-5-20251001-v1:0"


def test_gcp_bundle_fills_unset_slots():
    cfg = _fresh(cloud_provider="gcp")
    resolve_cloud_bundle(cfg)

    assert cfg.llm.provider == "vertex"
    assert cfg.llm.model == "gemini-2.5-flash"

    assert cfg.embedding.provider == "vertex"
    assert cfg.embedding.model == "gemini-embedding-001"
    assert cfg.embedding.dimension == 3072

    assert cfg.reranker.provider == "vertex"
    assert cfg.reranker.model == "semantic-ranker-default-004"

    # Pro escalation folded into the bundle.
    assert cfg.agent.pro_model == "gemini-2.5-pro"

    assert cfg.models is not None
    assert cfg.models.reason.model == "gemini-2.5-flash"
    assert cfg.models.tool_call.model == "gemini-2.5-flash"


def test_explicit_llm_model_not_overridden():
    # Operator pinned an explicit llm model -> bundle must NOT override it.
    cfg = _fresh(
        cloud_provider="aws",
        llm={"provider": "anthropic", "model": "claude-haiku-pinned"},
    )
    resolve_cloud_bundle(cfg)
    assert cfg.llm.provider == "anthropic"
    assert cfg.llm.model == "claude-haiku-pinned"


def test_explicit_embedding_dimension_not_overridden():
    cfg = _fresh(
        cloud_provider="aws",
        embedding={"provider": "bedrock", "model": "amazon.titan-embed-text-v2:0", "dimension": 512},
    )
    resolve_cloud_bundle(cfg)
    # Explicit dimension wins over the bundle's 1024.
    assert cfg.embedding.dimension == 512


def test_explicit_models_purpose_field_not_overridden():
    cfg = _fresh(
        cloud_provider="aws",
        models=ModelsConfig(reason=ModelSpec(model="my-custom-opus")),
    )
    resolve_cloud_bundle(cfg)
    # Explicit model field on reason wins; the unset provider field is filled.
    assert cfg.models.reason.model == "my-custom-opus"
    assert cfg.models.reason.provider == "bedrock"


def test_cloud_provider_none_is_noop():
    cfg = _fresh()  # cloud_provider defaults to None
    resolve_cloud_bundle(cfg)
    # Classic defaults untouched.
    assert cfg.llm.provider == "anthropic"
    assert cfg.llm.model == "claude-sonnet-4-20250514"
    assert cfg.embedding.provider == "openai"
    assert cfg.embedding.dimension is None
    assert cfg.reranker.provider == "noop"
    assert cfg.models is None
    assert cfg.agent.pro_model is None


def test_resolver_is_deterministic_and_does_not_alias_table():
    a = _fresh(cloud_provider="aws")
    b = _fresh(cloud_provider="aws")
    resolve_cloud_bundle(a)
    resolve_cloud_bundle(b)
    assert a.models.reason.model == b.models.reason.model
    # Mutating one resolved settings must not affect the shared table.
    a.models.reason.model = "mutated"
    assert CLOUD_BUNDLES["aws"]["reason"].model == "us.anthropic.claude-sonnet-4-6"
    assert b.models.reason.model == "us.anthropic.claude-sonnet-4-6"


# --- env-var model/provider overrides (no rebuild needed) -------------------

def test_env_cloud_provider_switch(monkeypatch):
    """OPSRAG_CLOUD_PROVIDER selects the bundle without YAML."""
    for v in ("OPSRAG_CLOUD_PROVIDER", "OPSRAG_LLM_MODEL", "OPSRAG_PRO_MODEL"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("OPSRAG_CLOUD_PROVIDER", "gcp")
    cfg = Settings.load(path="/nonexistent-config.yaml")
    assert cfg.cloud_provider == "gcp"
    assert cfg.llm.model == "gemini-2.5-flash"      # gcp bundle reason
    assert cfg.agent.pro_model == "gemini-2.5-pro"  # gcp bundle pro


def test_env_model_overrides_win_over_bundle(monkeypatch):
    """Per-knob env overrides beat the cloud-bundle defaults (env > bundle)."""
    monkeypatch.setenv("OPSRAG_CLOUD_PROVIDER", "aws")
    monkeypatch.setenv("OPSRAG_PRO_MODEL", "us.anthropic.claude-opus-4-8")
    monkeypatch.setenv("OPSRAG_LLM_MODEL", "custom-llm")
    monkeypatch.setenv("OPSRAG_RERANKER_MODEL", "custom-rerank")
    monkeypatch.setenv("OPSRAG_EMBEDDING_MODEL", "custom-embed")
    monkeypatch.setenv("OPSRAG_EMBEDDING_DIMENSION", "1024")
    cfg = Settings.load(path="/nonexistent-config.yaml")
    assert cfg.agent.pro_model == "us.anthropic.claude-opus-4-8"  # opt back into Opus
    assert cfg.llm.model == "custom-llm"
    assert cfg.reranker.model == "custom-rerank"
    assert cfg.embedding.model == "custom-embed"
    assert cfg.embedding.dimension == 1024


def test_env_invalid_values_ignored(monkeypatch):
    """A bad cloud provider / non-int dimension is ignored, not fatal."""
    for v in ("OPSRAG_LLM_MODEL", "OPSRAG_PRO_MODEL", "OPSRAG_EMBEDDING_MODEL"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("OPSRAG_CLOUD_PROVIDER", "azure")        # invalid
    monkeypatch.setenv("OPSRAG_EMBEDDING_DIMENSION", "huge")    # not an int
    cfg = Settings.load(path="/nonexistent-config.yaml")
    assert cfg.cloud_provider is None        # rejected -> no bundle applied
    assert cfg.embedding.dimension is None   # rejected -> stays unset
