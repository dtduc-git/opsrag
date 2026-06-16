"""Task 14 — Teams adapter extracts image attachments + ``fetch_image``."""
from __future__ import annotations

from opsrag.channels.adapters.teams.router import activity_to_inbound


def test_teams_extracts_image_attachments() -> None:
    activity = {
        "id": "a1",
        "text": "look",
        "from": {"id": "u1"},
        "conversation": {"id": "c1", "conversationType": "personal"},
        "attachments": [
            {"contentType": "image/png", "contentUrl": "https://teams/a.png", "name": "a.png"},
            {"contentType": "text/html", "content": "<p>hi</p>"},
        ],
    }
    inbound = activity_to_inbound(activity)
    assert len(inbound.images) == 1
    assert inbound.images[0].url == "https://teams/a.png"
    assert inbound.images[0].mime_type == "image/png"


def test_teams_image_without_content_url_is_skipped() -> None:
    activity = {
        "id": "a2",
        "text": "look",
        "from": {"id": "u1"},
        "conversation": {"id": "c1", "conversationType": "personal"},
        "attachments": [{"contentType": "image/png"}],  # no contentUrl
    }
    inbound = activity_to_inbound(activity)
    assert inbound.images == ()


def test_teams_no_attachments_yields_no_images() -> None:
    activity = {
        "id": "a3",
        "text": "look",
        "from": {"id": "u1"},
        "conversation": {"id": "c1", "conversationType": "personal"},
    }
    inbound = activity_to_inbound(activity)
    assert inbound.images == ()
