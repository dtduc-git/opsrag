"""Task 11 — Telegram adapter extracts ``ImageRef``s + ``fetch_image``."""
from __future__ import annotations

from opsrag.channels.adapters.telegram.adapter import TelegramAdapter


def _adapter() -> TelegramAdapter:
    a = TelegramAdapter.__new__(TelegramAdapter)
    a._token = "TKN"
    a._bot_username = "opsrag_bot"
    return a


def test_photo_message_yields_largest_image_ref() -> None:
    a = _adapter()
    message = {
        "message_id": 5,
        "chat": {"id": 7, "type": "private"},
        "from": {"id": 9},
        "caption": "see this",
        "photo": [
            {"file_id": "small", "file_size": 100, "width": 90, "height": 90},
            {"file_id": "big", "file_size": 9000, "width": 1280, "height": 1280},
        ],
    }
    inbound = a._message_to_inbound(message)
    assert inbound is not None
    assert inbound.text == "see this"
    assert len(inbound.images) == 1
    assert inbound.images[0].file_id == "big"  # largest PhotoSize
    assert inbound.images[0].mime_type == "image/jpeg"


def test_image_document_yields_image_ref() -> None:
    a = _adapter()
    message = {
        "message_id": 6,
        "chat": {"id": 7, "type": "private"},
        "from": {"id": 9},
        "caption": "diagram",
        "document": {
            "file_id": "doc1",
            "mime_type": "image/png",
            "file_size": 4242,
        },
    }
    inbound = a._message_to_inbound(message)
    assert inbound is not None
    assert len(inbound.images) == 1
    assert inbound.images[0].file_id == "doc1"
    assert inbound.images[0].mime_type == "image/png"
    assert inbound.images[0].size == 4242


def test_non_image_document_is_ignored() -> None:
    a = _adapter()
    message = {
        "message_id": 7,
        "chat": {"id": 7, "type": "private"},
        "from": {"id": 9},
        "text": "logs",
        "document": {"file_id": "d", "mime_type": "text/plain"},
    }
    inbound = a._message_to_inbound(message)
    assert inbound is not None
    assert inbound.images == ()


def test_text_only_message_has_no_images() -> None:
    a = _adapter()
    message = {
        "message_id": 8,
        "chat": {"id": 7, "type": "private"},
        "from": {"id": 9},
        "text": "hello",
    }
    inbound = a._message_to_inbound(message)
    assert inbound is not None
    assert inbound.images == ()
