"""Contract test (T042): an unknown top-level config key is rejected.

The root ``Settings`` model is declared with ``extra="forbid"``, so a typo'd
or unrecognised top-level key fails validation rather than being silently
ignored. Mirrors contracts/config-schema.md.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from opsrag.config import Settings


def test_unknown_top_level_key_raises() -> None:
    with pytest.raises(ValidationError) as exc_info:
        Settings.model_validate({"bogus_top_level_key": True})
    # The error should name the offending key so operators can find the typo.
    assert "bogus_top_level_key" in str(exc_info.value)


def test_valid_minimal_config_still_accepted() -> None:
    # A sibling sanity check: an empty config (all defaults) is valid, proving
    # the rejection above is about the *unknown key*, not strictness in general.
    settings = Settings.model_validate({})
    assert settings is not None


def test_unimplemented_chunking_strategy_rejected_at_load() -> None:
    # "semantic" is not implemented by the chunker factory. It must be rejected
    # at config-load (clear pydantic error) rather than deferring to a runtime
    # NotImplementedError when the chunker is built. The error should name the
    # field so operators can find the bad value.
    with pytest.raises(ValidationError) as exc_info:
        Settings.model_validate({"chunking": {"strategy": "semantic"}})
    assert "chunking.strategy" in str(exc_info.value)


def test_implemented_chunking_strategies_still_accepted() -> None:
    # The two strategies the factory can build remain valid.
    for strategy in ("parent_child", "fixed_size"):
        settings = Settings.model_validate({"chunking": {"strategy": strategy}})
        assert settings.chunking.strategy == strategy


@pytest.mark.parametrize("provider", ["cohere", "ollama"])
def test_unimplemented_embedding_provider_rejected_at_load(provider: str) -> None:
    # "cohere"/"ollama" are NOT built by the embedder factory
    # (opsrag/factory.py) -- they fell through to a runtime
    # NotImplementedError. They must be rejected at config-load with a clear
    # pydantic error naming the field, mirroring the chunking.strategy guard.
    with pytest.raises(ValidationError) as exc_info:
        Settings.model_validate({"embedding": {"provider": provider}})
    assert "embedding.provider" in str(exc_info.value)


def test_implemented_embedding_providers_still_accepted() -> None:
    # Every provider the factory can build remains valid at load.
    for provider in ("openai", "vertex", "bedrock", "fastembed", "litellm"):
        settings = Settings.model_validate({"embedding": {"provider": provider}})
        assert settings.embedding.provider == provider
