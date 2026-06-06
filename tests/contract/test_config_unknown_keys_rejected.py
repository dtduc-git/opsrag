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
