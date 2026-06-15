"""Task 9: channel-neutral image types (ImageRef + InboundMessage.images)."""
from __future__ import annotations

from opsrag.channels.types import ImageRef, InboundMessage


def test_inbound_defaults_to_no_images():
    msg = InboundMessage(
        channel_id="c", user_id="u", text="hi", message_id="m",
        thread_id=None, is_dm=True, workspace=None,
    )
    assert msg.images == ()


def test_image_ref_fields():
    ref = ImageRef(file_id="F1", url="https://x/y.png", mime_type="image/png", size=10)
    assert ref.file_id == "F1"
    assert ref.url == "https://x/y.png"
    assert ref.mime_type == "image/png"
    assert ref.size == 10
