"""Pull the thread's parent + reply chain as prior context.

When a user @-mentions the bot inside an existing thread, we fetch the
thread root + all prior replies and prepend a serialized "PRIOR THREAD
MESSAGES" block to the query. This lets the agent see both:

  * the **parent** -- usually the alert / question being discussed
    (Rootly / Datadog / Prometheus, or a human-posted question)
  * the **reply chain** -- earlier follow-ups, including the user's
    own previous turns ("that pipeline" -> the URL they mentioned 2
    messages ago)

Filtering rules (in order applied):
  1. Skip ONLY the bot's *own* historical replies (matched by
     ``user == client.self_user_id`` OR ``bot_id == client.self_bot_id``).
     OTHER bots (Rootly, Datadog, Prometheus Alertmanager) are kept --
     their messages ARE the context the user is asking about.
  2. Skip the current @mention message itself (the handler already
     passes its text as the primary query, no need to duplicate).
  3. If total assembled > max_chars, drop OLDEST messages first --
     recency wins. Always preserves the parent if at all possible.

Returns ``""`` when:
  - the event has no ``thread_ts`` (standalone @mention)
  - ``thread_ts == event.ts`` (the user @mentioned in the thread root
    itself -- there is no prior context yet)
  - all messages filter out (e.g. only the bot has posted before)
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from opsrag.slack_bot.client import SlackBotClient

_log = logging.getLogger("opsrag.slack_bot.thread_context")


def _display_name_from_info(info: dict, user_id: str) -> str:
    """Pick the best display name from a ``users.info`` payload."""
    if not info:
        return user_id
    profile = info.get("profile") or {}
    name = (
        profile.get("display_name")
        or profile.get("real_name")
        or info.get("real_name")
        or info.get("name")
        or user_id
    )
    return name or user_id


def _is_our_own_message(
    msg: dict, self_user_id: str | None, self_bot_id: str | None,
) -> bool:
    """True if this message was posted by OUR bot.

    Only filter our OWN reply history -- feedback-loop avoidance. Other
    bots (Rootly alerts etc.) MUST stay so the agent sees the alert
    payload it's being asked to triage.
    """
    if self_user_id and msg.get("user") == self_user_id:
        return True
    if self_bot_id and msg.get("bot_id") == self_bot_id:
        return True
    return False


async def assemble_thread_context(
    event: dict,
    client: SlackBotClient,
    *,
    max_messages: int = 20,
    max_chars: int = 2000,
) -> str:
    """Fetch parent + replies and return a serialized context block.

    See module docstring for filtering rules. Returns ``""`` when
    there's no usable context.
    """
    thread_ts = event.get("thread_ts")
    current_ts = event.get("ts")
    channel = event.get("channel")

    if not thread_ts or not channel:
        return ""
    if thread_ts == current_ts:
        return ""

    try:
        raw_messages = await client.fetch_thread_replies(
            channel=channel, thread_ts=thread_ts, limit=max_messages,
        )
    except Exception:  # noqa: BLE001
        _log.warning(
            "fetch_thread_replies failed channel=%s thread_ts=%s",
            channel, thread_ts, exc_info=True,
        )
        return ""

    self_user_id = getattr(client, "self_user_id", None)
    self_bot_id = getattr(client, "self_bot_id", None)

    # Walk oldest->newest (Slack returns chronological).
    filtered: list[dict] = []
    for msg in raw_messages:
        if _is_our_own_message(msg, self_user_id, self_bot_id):
            continue
        if current_ts is not None and msg.get("ts") == current_ts:
            continue
        text = (msg.get("text") or "").strip()
        if not text:
            continue
        filtered.append(msg)

    if not filtered:
        return ""

    # Resolve display names. Bot messages (Rootly, Datadog) carry a
    # ``username`` field and no ``user``; fall back to that so the
    # agent can tell who said what.
    resolved: list[tuple[str, str]] = []
    for msg in filtered:
        user_id = msg.get("user") or ""
        if user_id:
            try:
                info = await client.get_user_info(user_id)
                name = _display_name_from_info(info, user_id)
            except Exception:  # noqa: BLE001
                _log.debug("get_user_info(%s) failed", user_id, exc_info=True)
                name = user_id
        else:
            name = msg.get("username") or "bot"
        resolved.append((name, (msg.get("text") or "").strip()))

    header = "PRIOR THREAD MESSAGES:"
    lines = [f"{n}: {t}" for n, t in resolved]

    # Greedy-from-newest truncation; oldest gets dropped first.
    selected_reverse: list[str] = []
    running = len(header) + 1  # +1 for newline after header
    for line in reversed(lines):
        cost = len(line) + 1  # +1 for trailing newline
        if running + cost > max_chars and selected_reverse:
            break
        if running + cost > max_chars and not selected_reverse:
            # Newest line alone overflows -- truncate it but keep
            # something rather than returning nothing.
            keep = max(0, max_chars - running - 1)
            if keep <= 0:
                break
            selected_reverse.append(line[:keep] + "...")
            running += keep + 2
            break
        selected_reverse.append(line)
        running += cost

    if not selected_reverse:
        return ""

    selected = list(reversed(selected_reverse))
    return "\n".join([header, *selected])
