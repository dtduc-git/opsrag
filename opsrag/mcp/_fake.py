"""Shared contract for MCP fake backends (FR-012).

Each integration module exposes a ``build_fake()`` that returns a
:class:`FakeMCP`: the module's tool list plus whatever it takes to invoke
those tools WITHOUT any network / live backend. Integration tests use it
uniformly::

    from opsrag.mcp.gitlab import build_fake

    async def test_list_pipelines():
        fake = build_fake()
        result = await fake.call("gitlab_list_pipelines", {"project_id": "g/p"})
        assert result  # canned data, no network

Two faking strategies are accommodated, because the runtime tool-caller
treats families differently (see opsrag.agent.nodes.multi_agent):

- GitLab-style tools receive a live client object as the handler's first
  arg. Their ``build_fake`` sets ``client`` to a fake that mimics that
  object's surface.
- All other families ignore the client arg (``client=None``) and reach a
  module-internal connection bound at startup. Their ``build_fake`` wires
  that internal state to a fake and leaves ``client=None``.

Either way the test calls ``fake.call(name, args)`` and gets canned data.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class FakeMCP:
    """A module's tools wired to a fake backend for offline testing."""

    tools: list  # list[MCPTool] (each module defines its own MCPTool shape)
    client: Any = None  # passed as the handler's first arg; None for self-contained families
    teardown: Callable[[], Any] | None = None  # optional cleanup (unbind/restore)

    def tool_names(self) -> list[str]:
        return [t.name for t in self.tools]

    def _tool(self, name: str):
        for t in self.tools:
            if t.name == name:
                return t
        raise KeyError(f"tool not in this fake MCP: {name!r} (have {self.tool_names()})")

    async def call(self, name: str, args: dict) -> Any:
        """Invoke a tool handler against the fake backend."""
        tool = self._tool(name)
        # Modules use either `.call(client, args)` or a bare `.handler(client, args)`.
        if hasattr(tool, "call"):
            return await tool.call(self.client, args)
        return await tool.handler(self.client, args)

    def close(self) -> None:
        if self.teardown is not None:
            self.teardown()
