"""Microsoft Teams channel adapter (Bot Framework webhook on the API role).

Teams is NOT an outbound worker: the Bot Framework PUSHES activities to a
public HTTPS endpoint, so this channel is a FastAPI router mounted on the
``api`` role (see :func:`opsrag.channels.adapters.teams.router.build_teams_router`)
rather than a Socket Mode / gateway worker. The router constructs ONE
:class:`~opsrag.channels.adapters.teams.adapter.TeamsAdapter` + a
``ChannelDispatcher`` and drives them from each inbound ``Activity``.

The ``botbuilder`` SDK (``teams`` extra) is imported **lazily** inside the
adapter's outbound methods and the router's request handler, so importing this
package on the ``api`` role never requires the extra (the api role still boots
without ``botbuilder`` installed).
"""
from __future__ import annotations

from opsrag.channels.adapters.teams.adapter import TeamsAdapter

__all__ = ["TeamsAdapter"]
