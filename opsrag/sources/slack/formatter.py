"""Render a SlackThread to Markdown (with YAML frontmatter).

The output flows directly into the existing Markdown parser + chunker,
so the document looks like any other indexed doc to the retrieval layer.
"""
from __future__ import annotations

import re
from collections.abc import Awaitable, Callable

from opsrag.sources.slack.client import SlackThread, slack_ts_to_datetime

# Slack-specific text formatting we want to flatten before chunking:
#   <@U123ABC>     mention             -> @display-name
#   <#C123ABC|name> channel reference  -> #name
#   <!here>, <!channel>                -> @here, @channel
#   <https://...|title>                -> [title](https://...)
#   <https://...>                      -> https://...
_USER_MENTION_RE = re.compile(r"<@([UW][A-Z0-9]+)(?:\|([^>]+))?>")
_CHAN_MENTION_RE = re.compile(r"<#([CG][A-Z0-9]+)(?:\|([^>]+))?>")
_BROADCAST_RE = re.compile(r"<!(here|channel|everyone)(?:\|[^>]+)?>")
_LINK_RE = re.compile(r"<(https?://[^|>]+)(?:\|([^>]+))?>")

# Blocks of credential-shaped strings we redact before storing. RAG
# should never echo a Slack/AWS/GitHub token back at a user.
_REDACTIONS = [
    (re.compile(r"xox[abps]-[A-Za-z0-9-]{10,}"), "[redacted-slack-token]"),
    (re.compile(r"ghp_[A-Za-z0-9]{20,}"), "[redacted-github-token]"),
    (re.compile(r"gho_[A-Za-z0-9]{20,}"), "[redacted-github-oauth]"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "[redacted-aws-key]"),
]


async def render_thread(
    thread: SlackThread,
    *,
    channel_name: str,
    resolve_user: Callable[[str], Awaitable[str]],
) -> str:
    """Return a Markdown document for one Slack thread.

    `resolve_user` is the async lookup `(user_id) -> display_name`,
    typically `SlackClient.resolve_user`. Passing it as a callable
    keeps this module client-agnostic and trivially testable.
    """
    msgs = (thread.root, *thread.replies)
    # Pre-resolve every user mentioned anywhere (root, replies, or in
    # message text) so the rendering loop never awaits per-token.
    user_ids: set[str] = set()
    for m in msgs:
        if m.user_id:
            user_ids.add(m.user_id)
        for match in _USER_MENTION_RE.finditer(m.text):
            user_ids.add(match.group(1))
    name_map: dict[str, str] = {}
    for uid in user_ids:
        name_map[uid] = await resolve_user(uid)

    def _flatten(text: str) -> str:
        text = _USER_MENTION_RE.sub(
            lambda mt: "@" + (mt.group(2) or name_map.get(mt.group(1), mt.group(1))),
            text,
        )
        text = _CHAN_MENTION_RE.sub(lambda mt: "#" + (mt.group(2) or mt.group(1)), text)
        text = _BROADCAST_RE.sub(lambda mt: "@" + mt.group(1), text)
        text = _LINK_RE.sub(
            lambda mt: f"[{mt.group(2)}]({mt.group(1)})" if mt.group(2) else mt.group(1),
            text,
        )
        for pat, repl in _REDACTIONS:
            text = pat.sub(repl, text)
        return text

    # Frontmatter
    root_dt = slack_ts_to_datetime(thread.root.ts)
    latest_dt = slack_ts_to_datetime(thread.latest_ts)
    participants = sorted({
        name_map.get(m.user_id, m.user_id or "bot")
        for m in msgs
        if m.user_id
    })
    frontmatter = (
        "---\n"
        "type: slack-thread\n"
        f"channel: \"#{channel_name}\"\n"
        f"channel_id: \"{thread.channel_id}\"\n"
        f"thread_ts: \"{thread.thread_ts}\"\n"
        f"started_at: \"{root_dt.isoformat()}\"\n"
        f"last_reply_at: \"{latest_dt.isoformat()}\"\n"
        f"message_count: {len(msgs)}\n"
        f"participants: [{', '.join(repr(p) for p in participants)}]\n"
        "---\n\n"
    )

    title = _flatten(thread.root.text or "").strip().split("\n", 1)[0][:120]
    if not title:
        title = f"Thread in #{channel_name}"
    body = [f"# {title}\n"]

    for m in msgs:
        author = "bot"
        if m.user_id:
            author = "@" + name_map.get(m.user_id, m.user_id)
        elif m.bot_id:
            # Slack bot subtypes can carry a username in the raw payload
            # but we deliberately don't store it (skip_bot_messages
            # filter usually drops these). Render bot messages
            # generically so they don't impersonate a real user.
            author = "@bot"
        when = slack_ts_to_datetime(m.ts).strftime("%Y-%m-%d %H:%M UTC")
        text = _flatten(m.text or "").strip()
        if not text:
            continue
        body.append(f"**{author}** _({when})_:\n\n{text}\n")

    return frontmatter + "\n".join(body)
