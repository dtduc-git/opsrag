"""Channel name -> adapter class, and role -> channel maps.

The ``ADAPTERS`` values are ``"module:Class"`` strings rather than imported
classes **on purpose**: a disabled channel must never import its SDK
(``discord.py`` / ``botbuilder`` are optional extras). ``boot`` does the
lazy ``importlib`` resolution only for the channel that is actually enabled
on this role.

See design doc ``specs/002-channel-bots/design.md`` section 3.7.
"""
from __future__ import annotations

# String-based so disabled channels never import their SDK.
ADAPTERS: dict[str, str] = {
    "slack": "opsrag.channels.adapters.slack.adapter:SlackAdapter",
    "telegram": "opsrag.channels.adapters.telegram.adapter:TelegramAdapter",
    "discord": "opsrag.channels.adapters.discord.adapter:DiscordAdapter",
    "teams": "opsrag.channels.adapters.teams.adapter:TeamsAdapter",
}

# Each outbound worker role maps to exactly one channel. Teams has NO
# worker role -- it is a webhook mounted on the API role (Bot Framework
# pushes inbound activities to a public endpoint).
ROLE_TO_CHANNEL: dict[str, str] = {
    "slackbot": "slack",
    "telegrambot": "telegram",
    "discordbot": "discord",
}
