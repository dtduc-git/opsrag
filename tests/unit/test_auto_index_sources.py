"""Unit tests: config-driven auto-index source targets.

The nightly `--all` indexer Job used to hardcode ONLY Confluence in
`_configured_source_scopes`, so Slack/Rootly -- enabled in config, tokens
present -- were silently never indexed (Slack sat stale at one manual run;
Rootly at 0 chunks). The scheduler path in server.py duplicated the same
per-source scope logic as four hardcoded if-blocks.

Fix under test: every source config block carries an operator-facing
`auto_index` flag (config.yaml / values, NOT code) and self-describes its
targets via `auto_index_targets() -> [(source_type, scope)]`. A generic
resolver on Settings duck-types over ALL config blocks -- adding a new
source never touches the indexer or server again.
"""
from __future__ import annotations

from opsrag.config import Settings


def _cfg(**overrides) -> Settings:
    base = {
        "confluence": {"enabled": True, "spaces_allowlist": ["SRE", "~personal"]},
        "slack": {"enabled": True, "channels_allowlist": ["C0000000001", ""]},
        "rootly": {"enabled": True, "scope": "acme"},
        "investigation_history": {"enabled": True},
    }
    base.update(overrides)
    return Settings.model_validate(base)


# ------------------------------------------------------------- resolver --

def test_collects_all_enabled_sources_with_scopes():
    targets = _cfg().auto_index_source_targets()
    assert ("confluence", "SRE") in targets
    assert ("slack", "C0000000001") in targets
    assert ("rootly", "acme") in targets
    assert ("investigation-history", "opsrag") in targets


def test_personal_confluence_spaces_and_empty_scopes_filtered():
    targets = _cfg().auto_index_source_targets()
    scopes = [s for t, s in targets]
    assert "~personal" not in scopes  # defensive personal-space filter
    assert "" not in scopes           # empty slack channel id dropped


def test_auto_index_flag_excludes_source_without_disabling_it():
    cfg = _cfg(slack={
        "enabled": True, "auto_index": False,
        "channels_allowlist": ["C0000000001"],
    })
    types = {t for t, _ in cfg.auto_index_source_targets()}
    assert "slack" not in types
    # the connector itself stays enabled (manual --source runs still work)
    assert cfg.slack.enabled is True


def test_disabled_source_excluded():
    cfg = _cfg(rootly={"enabled": False, "scope": "acme"})
    types = {t for t, _ in cfg.auto_index_source_targets()}
    assert "rootly" not in types


def test_source_with_no_scopes_excluded():
    cfg = _cfg(confluence={"enabled": True, "spaces_allowlist": []})
    types = {t for t, _ in cfg.auto_index_source_targets()}
    assert "confluence" not in types


def test_auto_index_defaults_on():
    # Operators who enabled a connector expect the nightly run to cover it;
    # auto_index exists to OPT OUT in config, not as a second enable switch.
    cfg = _cfg()
    assert cfg.confluence.auto_index is True
    assert cfg.slack.auto_index is True
    assert cfg.rootly.auto_index is True
    assert cfg.investigation_history.auto_index is True


# ------------------------------------------------- job-indexer wiring --

def test_indexer_scopes_intersect_available_providers():
    """The Job must only target sources whose provider actually got built
    (e.g. token missing at runtime) -- otherwise a configured-but-unbuilt
    source fails the whole run's exit code."""
    from opsrag.job.indexer import _configured_source_scopes

    cfg = _cfg()
    scopes = _configured_source_scopes(cfg, available={"confluence", "slack"})
    assert scopes == {"confluence": ["SRE"], "slack": ["C0000000001"]}


def test_indexer_scopes_unrestricted_when_available_unknown():
    from opsrag.job.indexer import _configured_source_scopes

    scopes = _configured_source_scopes(_cfg(), available=None)
    assert scopes["confluence"] == ["SRE"]
    assert scopes["slack"] == ["C0000000001"]
    assert scopes["rootly"] == ["acme"]
    assert scopes["investigation-history"] == ["opsrag"]
