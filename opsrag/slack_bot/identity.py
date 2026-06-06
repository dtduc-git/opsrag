"""Slack user -> :class:`opsrag.auth.pomerium.CurrentUser` mapping.

Phase 1 (this module): every Slack event is mapped to
:meth:`CurrentUser.anonymous` with a *deterministic* synthetic ``oid``
of the shape ``slack-bot:<workspace>:<slack_user_id>``. This keeps the
identity returned to the agent uniformly typed (so reasoner / Phoenix
trace logging do not have to special-case Slack), while still allowing
us to group traces by Slack source + Slack user.

Phase 2 (deferred -- DO NOT IMPLEMENT YET): look up ``users.info`` on
the Slack client, extract ``email``, then query
``opsrag.auth.store.UserStore`` for the Pomerium ``oid`` that owns that
email. If we find a hit we promote the user from anonymous to identified
and inherit their Pomerium groups for ACL checks. If we don't, we fall
back to the Phase 1 synthetic identity below.
"""
from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from opsrag.auth.pomerium import CurrentUser

if TYPE_CHECKING:  # pragma: no cover - import-only typing
    from opsrag.slack_bot.client import SlackBotClient


_log = logging.getLogger("opsrag.slack_bot.identity")

# Default workspace label used inside the synthetic oid. The Slack
# event payload usually includes ``team`` (workspace ID, e.g. ``T0...``);
# we use that when present, otherwise fall back to this constant so the
# oid still has 3 colon-separated segments.
_DEFAULT_WORKSPACE = "unknown"


def _synthetic_oid(workspace: str, slack_user_id: str) -> str:
    """Compose the deterministic Phase 1 oid.

    Kept as a free function (not a private method) so tests can pin
    the exact format without instantiating anything.
    """
    return f"slack-bot:{workspace}:{slack_user_id}"


async def slack_user_to_current_user(
    event: dict[str, Any],
    *,
    client: SlackBotClient | None = None,
) -> CurrentUser:
    """Translate a Slack event payload to a :class:`CurrentUser`.

    Parameters
    ----------
    event:
        The raw Slack event dict (``app_mention`` or ``message.im``).
        ``user`` and (optionally) ``team`` keys are read.
    client:
        Reserved for Phase 2 -- kept in the signature so the handler can
        already pass it without later refactor churn. Phase 1 ignores it.

    Returns
    -------
    CurrentUser
        ``CurrentUser.anonymous()`` with the ``oid`` field overwritten
        to the deterministic synthetic identifier. ``is_anonymous`` is
        still ``True`` so authorisation code that gates on it (e.g.
        admin checks) keeps fail-closed semantics -- the Slack identity
        is *traceable* but not *authenticated*.
    """
    # Tolerate missing/garbled events: Slack should always send a `user`
    # key for the two event types we subscribe to, but if anything is
    # weird we still want to return a usable CurrentUser rather than
    # crash the handler.
    slack_user_id = (event or {}).get("user") or "unknown-user"
    workspace = (event or {}).get("team") or _DEFAULT_WORKSPACE
    oid = _synthetic_oid(workspace, slack_user_id)

    # TODO(phase2): when client is provided, resolve email via
    # `users.info` and look up Pomerium oid in opsrag.auth.store.UserStore.
    # If hit, return a non-anonymous CurrentUser with the real oid,
    # email, name, picture_url, groups.
    base = CurrentUser.anonymous()
    return replace(base, oid=oid)
