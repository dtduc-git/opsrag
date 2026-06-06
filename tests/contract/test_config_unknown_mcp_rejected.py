"""Contract test (T071): an unknown key under ``mcp:`` is rejected.

The ``mcp`` map only admits the 14 known integration names; a typo'd or
unrecognised integration name fails validation rather than being silently
accepted. Mirrors contracts/config-schema.md.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from opsrag.config import Settings
from opsrag.mcp.registry import REGISTRY


def test_unknown_mcp_key_rejected() -> None:
    with pytest.raises(ValidationError) as exc_info:
        Settings.model_validate({"mcp": {"bogus_integration": {"enabled": False}}})
    assert "bogus_integration" in str(exc_info.value)


def test_all_known_mcp_keys_accepted() -> None:
    # The complement: every registered name validates.
    cfg = {"mcp": {name: {"enabled": False} for name in REGISTRY}}
    settings = Settings.model_validate(cfg)
    assert set(settings.mcp) == set(REGISTRY)
