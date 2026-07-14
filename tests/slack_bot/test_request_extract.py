"""Pure classifier + extractor tests for the Slack first-responder."""
from __future__ import annotations

from opsrag.channels.config import FirstResponderChannelConfig
from opsrag.slack_bot.request_extract import (
    RequestKind,
    classify,
    extract_query,
    extract_requester,
    is_ignorable_subtype,
)

CHAN = FirstResponderChannelConfig(
    request_app_allowlist=["A0EXAMPLE01"],
    oncall_handle="S0EXAMPLE01",
    include_direct=True,
)


def test_workflow_post_from_allowlisted_app_is_workflow():
    event = {"subtype": "bot_message", "bot_id": "B0EXAMPLE01",
             "app_id": "A0EXAMPLE01", "text": "Env: prd", "ts": "1.1"}
    assert classify(event, CHAN) is RequestKind.WORKFLOW


def test_other_bot_is_ignored():
    event = {"subtype": "bot_message", "bot_id": "BROOTLY", "app_id": "AROOTLY",
             "text": "alert firing", "ts": "1.1"}
    assert classify(event, CHAN) is RequestKind.IGNORE


def test_human_message_is_direct_when_enabled():
    event = {"user": "U123", "text": "how do I restart the pod?", "ts": "1.1"}
    assert classify(event, CHAN) is RequestKind.DIRECT


def test_human_message_ignored_when_direct_disabled():
    chan = FirstResponderChannelConfig(
        request_app_allowlist=["A0EXAMPLE01"], oncall_handle="S0EXAMPLE01",
        include_direct=False,
    )
    event = {"user": "U123", "text": "hi", "ts": "1.1"}
    assert classify(event, chan) is RequestKind.IGNORE


def test_message_changed_is_ignorable_subtype():
    assert is_ignorable_subtype({"subtype": "message_changed"}) is True
    assert is_ignorable_subtype({"subtype": "channel_join"}) is True


def test_bot_message_is_not_ignorable_subtype():
    # bot_message is a real request candidate (the workflow), not junk.
    assert is_ignorable_subtype({"subtype": "bot_message"}) is False


def test_plain_message_is_not_ignorable_subtype():
    assert is_ignorable_subtype({"ts": "1.1", "text": "hi"}) is False


def test_extract_strips_bot_mention():
    event = {"text": "<@U0BOT> please help with prd", "ts": "1.1"}
    assert extract_query(event) == "please help with prd"


def test_extract_preserves_non_leading_mention():
    event = {"text": "can <@U0TEAM> help with prd?", "ts": "1.1"}
    # A mid-string teammate mention must survive (only leading mentions are stripped).
    assert extract_query(event) == "can <@U0TEAM> help with prd?"


def test_extract_strips_multiple_leading_mentions():
    event = {"text": "<@U0OPSRAG> <@U0TEAM> restart the pod", "ts": "1.1"}
    assert extract_query(event) == "restart the pod"


def test_extract_falls_back_to_block_text_when_text_empty():
    event = {
        "text": "",
        "blocks": [
            {"type": "section", "text": {"type": "mrkdwn", "text": "Environment: prd"}},
            {"type": "section", "text": {"type": "plain_text", "text": "Desc: pods crashing"}},
        ],
        "ts": "1.1",
    }
    out = extract_query(event)
    assert "Environment: prd" in out
    assert "Desc: pods crashing" in out


# --- extract_requester -----------------------------------------------------


def test_extract_requester_direct_human_returns_user_id():
    event = {"user": "U123", "text": "how do I restart the pod?", "ts": "1.1"}
    assert extract_requester(event) == "U123"


def test_extract_requester_labeled_field_wins_over_preceding_mention():
    # A workflow/bot post: an unrelated mention appears before the labeled
    # Requester field -- the field must win, not the first mention seen.
    event = {
        "subtype": "bot_message", "bot_id": "B1", "app_id": "A1",
        "text": "cc <@U0TEAM> Requester: <@U999>",
        "ts": "1.1",
    }
    assert extract_requester(event) == "U999"


def test_extract_requester_single_mention_no_label():
    event = {
        "subtype": "bot_message", "bot_id": "B1", "app_id": "A1",
        "text": "Ping <@U555> about this outage",
        "ts": "1.1",
    }
    assert extract_requester(event) == "U555"


def test_extract_requester_ambiguous_multiple_mentions_returns_none():
    event = {
        "subtype": "bot_message", "bot_id": "B1", "app_id": "A1",
        "text": "cc <@U1> and <@U2>",
        "ts": "1.1",
    }
    assert extract_requester(event) is None


def test_extract_requester_no_mentions_returns_none():
    event = {
        "subtype": "bot_message", "bot_id": "B1", "app_id": "A1",
        "text": "Env: prd, pods crashing",
        "ts": "1.1",
    }
    assert extract_requester(event) is None


def test_extract_requester_from_flattened_blocks_when_text_empty():
    event = {
        "subtype": "bot_message", "bot_id": "B1", "app_id": "A1",
        "text": "",
        "blocks": [
            {"type": "section", "text": {"type": "mrkdwn", "text": "Requester: <@U777>"}},
        ],
        "ts": "1.1",
    }
    assert extract_requester(event) == "U777"
