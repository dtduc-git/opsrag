"""Confluence Cloud connector.

Public API:

- `ConfluenceClient` -- async REST client (auth, pagination, retry).
- `ConfluenceSource` -- `SourceProtocol` implementation that yields
  rendered Markdown pages.
- `Space`, `PageRef`, `Page` -- small dataclasses used by the client.
"""
from opsrag.sources.confluence.client import (
    ConfluenceClient,
    Page,
    PageRef,
    Space,
)
from opsrag.sources.confluence.source import ConfluenceSource

__all__ = ["ConfluenceClient", "ConfluenceSource", "Page", "PageRef", "Space"]
