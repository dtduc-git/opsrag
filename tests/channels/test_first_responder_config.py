"""Config-model tests for the Slack first-responder block."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from opsrag.channels.config import (
    FirstResponderChannelConfig,
    FirstResponderConfig,
    SlackChannelConfig,
)


def test_slack_config_parses_first_responder_block():
    cfg = SlackChannelConfig.model_validate(
        {
            "enabled": True,
            "first_responder": {
                "enabled": True,
                "channels": {
                    "C0EXAMPLE01": {
                        "request_app_allowlist": ["A0EXAMPLE01"],
                        "oncall_handle": "S0EXAMPLE01",
                        "include_direct": True,
                        "reply_in_thread": True,
                    }
                },
            },
        }
    )
    assert cfg.first_responder.enabled is True
    chan = cfg.first_responder.channels["C0EXAMPLE01"]
    assert chan.request_app_allowlist == ["A0EXAMPLE01"]
    assert chan.oncall_handle == "S0EXAMPLE01"
    assert chan.include_direct is True
    assert chan.daily_quota == 500  # default


def test_first_responder_defaults_disabled_and_empty():
    cfg = SlackChannelConfig()
    assert cfg.first_responder.enabled is False
    assert cfg.first_responder.channels == {}


def test_channel_key_must_be_slack_channel_id():
    with pytest.raises(ValidationError):
        FirstResponderConfig.model_validate(
            {"channels": {"not-a-channel": {"oncall_handle": "S0EXAMPLE01"}}}
        )


def test_request_app_allowlist_rejects_non_app_id():
    with pytest.raises(ValidationError):
        FirstResponderChannelConfig.model_validate(
            {"request_app_allowlist": ["lowercase"], "oncall_handle": "S0EXAMPLE01"}
        )


def test_oncall_handle_accepts_bare_or_subteam_form():
    a = FirstResponderChannelConfig(oncall_handle="S0EXAMPLE01")
    b = FirstResponderChannelConfig(oncall_handle="<!subteam^S0EXAMPLE01>")
    assert a.oncall_handle == "S0EXAMPLE01"
    assert b.oncall_handle == "<!subteam^S0EXAMPLE01>"


def test_oncall_handle_rejects_garbage():
    with pytest.raises(ValidationError):
        FirstResponderChannelConfig(oncall_handle="@someone")


def test_agent_name_and_oncall_display_defaults():
    cfg = FirstResponderChannelConfig()
    assert cfg.agent_name == "OpsRAG"
    assert cfg.oncall_display == ""


def test_agent_name_empty_coerces_to_default():
    cfg = FirstResponderChannelConfig(agent_name="")
    assert cfg.agent_name == "OpsRAG"


def test_agent_name_and_oncall_display_are_trimmed():
    cfg = FirstResponderChannelConfig(agent_name="  Bob  ", oncall_display="  x  ")
    assert cfg.agent_name == "Bob"
    assert cfg.oncall_display == "x"
