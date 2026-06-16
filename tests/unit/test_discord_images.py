"""Task 12 — Discord adapter extracts image attachments + ``fetch_image``."""
from __future__ import annotations

from opsrag.channels.adapters.discord.adapter import _message_to_inbound


class _Att:
    def __init__(self, url: str, ct: str, size: int = 12) -> None:
        self.url = url
        self.content_type = ct
        self.size = size
        self.filename = "a.png"


class _Author:
    def __init__(self, uid: int) -> None:
        self.id = uid
        self.bot = False
        self.name = "user"
        self.display_name = "user"


class _DMChannel:
    __qualname__ = "DMChannel"

    def __init__(self, cid: int) -> None:
        self.id = cid

        class _T:
            name = "private"

            def __str__(self) -> str:
                return "private"

        self.type = _T()


class _Msg:
    def __init__(self) -> None:
        self.id = 55
        self.content = "look at this"
        self.author = _Author(9)
        self.channel = _DMChannel(7)
        self.guild = None
        self.mentions = []
        self.attachments = [
            _Att("https://cdn/a.png", "image/png"),
            _Att("https://cdn/x.txt", "text/plain"),
        ]


_BOT = _Author(100)
_BOT.bot = True


def test_discord_extracts_only_image_attachments() -> None:
    inbound = _message_to_inbound(_Msg(), _BOT)
    assert inbound is not None
    assert len(inbound.images) == 1
    assert inbound.images[0].url == "https://cdn/a.png"
    assert inbound.images[0].mime_type == "image/png"
    assert inbound.images[0].size == 12


def test_discord_no_attachments_yields_no_images() -> None:
    msg = _Msg()
    msg.attachments = []
    inbound = _message_to_inbound(msg, _BOT)
    assert inbound is not None
    assert inbound.images == ()
