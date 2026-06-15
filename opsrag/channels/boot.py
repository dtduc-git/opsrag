"""Generic per-role channel boot -- with the role-gating fix (D6).

``build_and_start`` is called from the FastAPI lifespan. It is the single
place the role-gating bug is fixed: a channel worker starts **iff**
``OPSRAG_ROLE`` maps to a channel AND that channel is enabled. On the
``api`` role (and any other non-channel role) it returns ``None`` -- so no
outbound worker boots, and N API replicas never open N duplicate Socket
Mode / gateway connections.

Teams is handled separately on the ``api`` role as a webhook router (see
``§6``); it is NOT booted here -- a ``# TODO`` hook is left below.

See design doc ``specs/002-channel-bots/design.md`` section 3.7.
"""
from __future__ import annotations

import importlib
import logging
from typing import Any

from opsrag.channels.dispatcher import ChannelDispatcher
from opsrag.channels.permission import ChannelPermission
from opsrag.channels.registry import ADAPTERS, ROLE_TO_CHANNEL

_log = logging.getLogger("opsrag.channels.boot")


def _load_adapter_class(channel_name: str) -> type:
    """Lazily resolve ``"module:Class"`` -> the adapter class.

    Importing here (not at module top) keeps a disabled channel's SDK out
    of the import graph entirely.
    """
    target = ADAPTERS[channel_name]
    module_path, _, class_name = target.partition(":")
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


async def build_and_start(
    role: str,
    cfg: Any,
    agent_graph: Any,
    providers: Any,
    caches: Any,
) -> Any | None:
    """Boot the channel worker for ``role``, or return ``None``.

    Returns the connected adapter (for lifespan shutdown via
    ``adapter.close()``), or ``None`` when this role runs no outbound
    channel worker.

    ``caches`` is a simple namespace/object exposing ``qa_cache``,
    ``investigation_cache``, ``semantic_router``, ``feedback_store``
    (any may be ``None``).
    """
    channel_name = ROLE_TO_CHANNEL.get(role or "")
    if channel_name is None:
        # Not a channel worker role (e.g. "api", "backend").
        # TODO(teams): on the "api" role, the lifespan separately mounts
        # the Teams webhook router when cfg.channels.teams.enabled -- that
        # is NOT this function's job (Teams is inbound-only, no worker).
        _log.debug("boot: role=%r maps to no channel worker", role)
        return None

    channels_cfg = getattr(cfg, "channels", None)
    channel_cfg = getattr(channels_cfg, channel_name, None) if channels_cfg else None
    if channel_cfg is None or not getattr(channel_cfg, "enabled", False):
        _log.info(
            "boot: role=%s channel=%s not enabled -- worker not started",
            role, channel_name,
        )
        return None

    adapter_cls = _load_adapter_class(channel_name)

    permission = ChannelPermission(
        allowed_channels=set(getattr(channel_cfg, "allowlist", []) or []),
        per_user_daily_quota=int(getattr(channel_cfg, "per_user_daily_quota", 200)),
        allowed_dm_users=set(getattr(channel_cfg, "dm_allowlist", []) or []),
    )

    adapter = adapter_cls(channel_cfg)

    dispatcher = ChannelDispatcher(
        adapter=adapter,
        agent_graph=agent_graph,
        providers=providers,
        permission=permission,
        web_ui_base_url=getattr(channel_cfg, "web_ui_base_url", "") or "",
        thread_context_message_cap=int(
            getattr(channel_cfg, "thread_context_message_cap", 20)
        ),
        qa_cache=getattr(caches, "qa_cache", None),
        investigation_cache=getattr(caches, "investigation_cache", None),
        semantic_router=getattr(caches, "semantic_router", None),
        feedback_store=getattr(caches, "feedback_store", None),
    )

    await adapter.connect(dispatcher)
    _log.info("boot: role=%s channel=%s worker started", role, channel_name)
    return adapter
