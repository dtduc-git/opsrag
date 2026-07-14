"""Channel-neutral 👍/👎 feedback persistence.

This is the persistence half of ``slack_bot/handler.py::on_block_action``
with the Slack parsing removed (the adapter already produced a neutral
:class:`~opsrag.channels.types.FeedbackEvent`) and the ephemeral confirm
moved out (the dispatcher calls ``adapter.confirm_feedback`` afterwards).

Two best-effort writes, neither blocks the other:
  1. ``investigation_cache.record_feedback`` -- increments the up/down
     counter on the cached investigation so high-feedback investigations
     rank higher in past-similar retrieval.
  2. ``feedback_store.record`` -- append-only audit row in Postgres, used
     by SRE eval dashboards. ``user_id`` is namespaced
     ``"<channel_name>:<user>"`` so feedback from different platforms is
     attributable and never collides.

No platform calls happen here.

See design doc ``specs/002-channel-bots/design.md`` section 3.4.
"""
from __future__ import annotations

import logging
from typing import Any

from opsrag.channels.types import FeedbackEvent

_log = logging.getLogger("opsrag.channels.feedback")


async def record_feedback(
    fb: FeedbackEvent,
    *,
    investigation_cache: Any,
    feedback_store: Any,
    channel_name: str,
) -> bool:
    """Persist a feedback event. Returns True if the event was well-formed.

    Malformed events (bad thumbs / missing investigation id) are ignored
    and return ``False`` so the dispatcher can skip the ephemeral confirm.
    """
    if not isinstance(fb, FeedbackEvent):
        _log.warning("feedback: rejecting non-FeedbackEvent %r", type(fb))
        return False

    thumbs = (fb.thumbs or "").strip()
    investigation_id = (fb.investigation_id or "").strip()
    if thumbs not in ("up", "down") or not investigation_id:
        _log.warning(
            "feedback: malformed event thumbs=%r id=%r (ignored)",
            thumbs, investigation_id,
        )
        return False

    direction = 1 if thumbs == "up" else -1
    namespaced_user = f"{channel_name}:{fb.user_id or 'unknown'}"

    # The resolved investigation's question+answer, threaded from the cache
    # write below into the Postgres audit row so the Retrieval-Quality
    # dashboard shows WHAT was rated (channel feedback carries no snippets
    # of its own -- the click payload only has the investigation id).
    query_snippet: str | None = None
    answer_snippet: str | None = None

    # -- 1. investigation_cache (best-effort)
    if investigation_cache is not None:
        try:
            fb_result = await investigation_cache.record_feedback(
                investigation_id,
                thumbs=thumbs,
                correction=None,
            )
            if fb_result:
                query_snippet = getattr(fb_result, "query", None)
                answer_snippet = getattr(fb_result, "answer", None)
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "feedback: investigation_cache write failed id=%s err=%s",
                investigation_id, exc,
            )

    # Ungrounded / LOW-confidence answers aren't cached -> nothing resolved
    # above. Fall back to the answer text the adapter captured from the click
    # payload so the dashboard card still shows WHAT was rated.
    if not answer_snippet:
        answer_snippet = getattr(fb, "answer_snippet", "") or None

    # -- 2. feedback_store (best-effort)
    if feedback_store is not None:
        try:
            await feedback_store.record(
                investigation_id=investigation_id,
                direction=direction,
                thread_id=fb.thread_id,
                user_id=namespaced_user,
                note=None,
                query_snippet=query_snippet,
                answer_snippet=answer_snippet,
            )
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "feedback: feedback_store write failed id=%s err=%s",
                investigation_id, exc,
            )

    _log.info(
        "feedback ok: channel=%s thumbs=%s investigation=%s user=%s",
        channel_name, thumbs, investigation_id, namespaced_user,
    )
    return True
