"""Task 13 — Slack adapter extracts image files + authed ``fetch_image``."""
from __future__ import annotations

from opsrag.channels.adapters.slack.adapter import _event_to_inbound


def test_slack_extracts_image_files() -> None:
    event = {
        "channel": "C1",
        "user": "U1",
        "text": "hi",
        "ts": "1.1",
        "team": "T1",
        "files": [
            {"url_private": "https://files.slack/a.png", "mimetype": "image/png", "size": 10},
            {"url_private": "https://files.slack/b.pdf", "mimetype": "application/pdf"},
        ],
    }
    inbound = _event_to_inbound(event, is_dm=True)
    assert len(inbound.images) == 1
    assert inbound.images[0].url == "https://files.slack/a.png"
    assert inbound.images[0].mime_type == "image/png"
    assert inbound.images[0].size == 10


def test_slack_image_without_url_private_is_skipped() -> None:
    event = {
        "channel": "C1",
        "user": "U1",
        "text": "hi",
        "ts": "1.1",
        "team": "T1",
        "files": [{"mimetype": "image/png"}],  # no url_private
    }
    inbound = _event_to_inbound(event, is_dm=True)
    assert inbound.images == ()


def test_slack_no_files_yields_no_images() -> None:
    event = {"channel": "C1", "user": "U1", "text": "hi", "ts": "1.1", "team": "T1"}
    inbound = _event_to_inbound(event, is_dm=False)
    assert inbound.images == ()
