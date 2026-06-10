"""Unit tests for P1 wiring: channels config + role-gated boot.

Covers:
  * ``Settings`` carries a ``channels`` block (default all-disabled) AND keeps
    the legacy ``slack_bot`` block.
  * ``_mirror_legacy_slack_bot``: an enabled legacy ``slack_bot`` mirrors into
    ``channels.slack`` (deprecation path) but does NOT override an already
    enabled ``channels.slack``.
  * ``channels.boot.build_and_start`` role-gating (design D6):
      - ``api`` / ``backend`` roles start NO worker (returns None);
      - a channel role with the channel DISABLED starts no worker;
      - a disabled channel's adapter module is never imported (importlib not
        called) -- the SDK stays out of the import graph;
      - the slackbot role with channels.slack enabled boots the SlackAdapter
        (its ``connect`` is awaited) without touching the network (we stub the
        adapter class).

No network: the agent graph + providers + caches are simple stubs and the
Slack adapter class is swapped for a fake.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import opsrag.channels.boot as boot_mod
from opsrag.channels.config import ChannelsConfig
from opsrag.config import Settings


# ---------------------------------------------------------------------------
# Config: channels block + legacy mirror
# ---------------------------------------------------------------------------
def test_settings_has_channels_and_slack_bot() -> None:
    cfg = Settings()
    assert isinstance(cfg.channels, ChannelsConfig)
    assert cfg.channels.slack.enabled is False
    assert cfg.channels.telegram.enabled is False
    assert cfg.channels.discord.enabled is False
    assert cfg.channels.teams.enabled is False
    # Legacy block preserved.
    assert cfg.slack_bot.enabled is False


def test_legacy_slack_bot_mirrors_into_channels_slack(tmp_path) -> None:
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        "slack_bot:\n"
        "  enabled: true\n"
        "  channels_allowlist: [C0ABCDEF]\n"
        "  per_user_daily_quota: 17\n"
    )
    cfg = Settings.load(yaml_path)
    assert cfg.channels.slack.enabled is True
    assert cfg.channels.slack.allowlist == ["C0ABCDEF"]
    assert cfg.channels.slack.per_user_daily_quota == 17


def test_explicit_channels_slack_wins_over_legacy(tmp_path) -> None:
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        "slack_bot:\n"
        "  enabled: true\n"
        "  channels_allowlist: [C0LEGACY]\n"
        "channels:\n"
        "  slack:\n"
        "    enabled: true\n"
        "    allowlist: [C0NEWONE]\n"
    )
    cfg = Settings.load(yaml_path)
    # The new block is kept verbatim; the legacy block does NOT clobber it.
    assert cfg.channels.slack.allowlist == ["C0NEWONE"]


# ---------------------------------------------------------------------------
# Boot: role-gating (design D6)
# ---------------------------------------------------------------------------
def _cfg_with(slack_enabled: bool) -> SimpleNamespace:
    channels = ChannelsConfig()
    channels.slack.enabled = slack_enabled
    channels.slack.allowlist = ["C0ABCDEF"] if slack_enabled else []
    return SimpleNamespace(channels=channels)


@pytest.mark.asyncio
async def test_api_role_starts_no_worker(monkeypatch) -> None:
    called = {"import": False}

    def _spy_import(name):  # pragma: no cover - should never run
        called["import"] = True
        raise AssertionError("importlib must not be called on the api role")

    monkeypatch.setattr(boot_mod.importlib, "import_module", _spy_import)
    out = await boot_mod.build_and_start(
        "api", _cfg_with(True), object(), object(), SimpleNamespace(),
    )
    assert out is None
    assert called["import"] is False


@pytest.mark.asyncio
async def test_disabled_channel_never_imports_sdk(monkeypatch) -> None:
    called = {"import": False}

    def _spy_import(name):  # pragma: no cover - should never run
        called["import"] = True
        raise AssertionError("disabled channel must not import its adapter")

    monkeypatch.setattr(boot_mod.importlib, "import_module", _spy_import)
    out = await boot_mod.build_and_start(
        "slackbot", _cfg_with(False), object(), object(), SimpleNamespace(),
    )
    assert out is None
    assert called["import"] is False


@pytest.mark.asyncio
async def test_slackbot_role_boots_adapter(monkeypatch) -> None:
    connected = {"sink": None}

    class _FakeAdapter:
        name = "slack"

        def __init__(self, config):
            self.config = config

        async def connect(self, sink):
            connected["sink"] = sink

        async def close(self):
            pass

    fake_module = SimpleNamespace(SlackAdapter=_FakeAdapter)
    monkeypatch.setattr(
        boot_mod.importlib, "import_module", lambda name: fake_module,
    )
    caches = SimpleNamespace(
        qa_cache=None, investigation_cache=None,
        semantic_router=None, feedback_store=None,
    )
    adapter = await boot_mod.build_and_start(
        "slackbot", _cfg_with(True), object(), object(), caches,
    )
    assert isinstance(adapter, _FakeAdapter)
    # connect() was awaited with the dispatcher as the sink.
    assert connected["sink"] is not None
    # The dispatcher is the CoreSink the adapter received.
    assert hasattr(connected["sink"], "on_message")
    assert hasattr(connected["sink"], "on_feedback")
