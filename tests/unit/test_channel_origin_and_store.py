"""Unit tests for the public-channel-conversation origin helpers + the
owner-agnostic ``list_sessions_by_prefixes`` store method.

A conversation's privacy is encoded in its ``thread_id`` prefix
(``<platform>-thread:`` = SHARED/public, ``<platform>-dm:`` = private 1:1,
plain web ids = private). These tests pin that predicate and the prefix-based
store filter that the public-channel read API relies on. See
``docs/superpowers/specs/2026-06-15-public-channel-conversations-design.md``.
"""
from __future__ import annotations

import asyncio

import pytest
from langgraph.checkpoint.base import empty_checkpoint

from opsrag.channels.origin import (
    PUBLIC_CHANNEL_THREAD_PREFIXES,
    is_public_channel_thread,
    platform_of,
)
from opsrag.sessions.memory import InMemorySessionStore

# Every platform that has a public (shared) channel prefix.
_PLATFORMS = ("slack", "discord", "telegram", "teams")


# ---------------------------------------------------------------------------
# origin helpers
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("platform", _PLATFORMS)
def test_thread_prefix_is_public_with_correct_platform(platform):
    tid = f"{platform}-thread:CHANNEL123:1700000000.0001"
    assert is_public_channel_thread(tid) is True
    assert platform_of(tid) == platform


@pytest.mark.parametrize("platform", _PLATFORMS)
def test_dm_prefix_is_not_public(platform):
    # Private 1:1 DMs must never be treated as public, and have no platform.
    tid = f"{platform}-dm:CHANNEL123"
    assert is_public_channel_thread(tid) is False
    assert platform_of(tid) is None


def test_web_thread_id_is_not_public():
    # A plain web thread id (no platform prefix) is private.
    assert is_public_channel_thread("user_abcd") is False
    assert platform_of("user_abcd") is None


def test_none_thread_id_is_not_public():
    assert is_public_channel_thread(None) is False
    assert platform_of(None) is None


def test_empty_thread_id_is_not_public():
    assert is_public_channel_thread("") is False
    assert platform_of("") is None


def test_public_prefixes_cover_every_platform():
    # The exported prefix tuple is the single source of truth the store filters
    # on; assert it is exactly one ``-thread:`` prefix per platform.
    assert set(PUBLIC_CHANNEL_THREAD_PREFIXES) == {f"{p}-thread:" for p in _PLATFORMS}


# ---------------------------------------------------------------------------
# store: list_sessions_by_prefixes (owner-agnostic, prefix-filtered)
# ---------------------------------------------------------------------------
def _seed(store: InMemorySessionStore, thread_id: str, owner: str) -> None:
    """Persist one checkpoint for ``thread_id`` owned by ``owner`` -- mirrors
    what graph.py writes (the owner rides the configurable ``user_id``)."""
    saver = store.get_checkpointer()
    cfg = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": "",
            "user_id": owner,
        }
    }
    saver.put(cfg, empty_checkpoint(), {}, {})


def _seeded_store() -> InMemorySessionStore:
    store = InMemorySessionStore()
    # A mix: two shared-channel threads (different bot owners), a private DM,
    # and a web-owned thread.
    _seed(store, "slack-thread:C1:1700000000.0001", "slack-bot-oid")
    _seed(store, "discord-thread:G3:42", "discord-bot-oid")
    _seed(store, "telegram-dm:U2", "telegram-bot-oid")
    _seed(store, "user_abcd", "web-user-oid")
    return store


def test_list_sessions_by_prefixes_returns_only_thread_prefixed():
    store = _seeded_store()
    rows = asyncio.run(
        store.list_sessions_by_prefixes(PUBLIC_CHANNEL_THREAD_PREFIXES)
    )
    tids = {r["thread_id"] for r in rows}
    # ONLY the two ``-thread:`` conversations -- the DM and the web thread are
    # excluded.
    assert tids == {"slack-thread:C1:1700000000.0001", "discord-thread:G3:42"}


def test_list_sessions_by_prefixes_is_owner_agnostic():
    store = _seeded_store()
    rows = asyncio.run(
        store.list_sessions_by_prefixes(PUBLIC_CHANNEL_THREAD_PREFIXES)
    )
    # Two distinct bot owners both show up -- no owner filtering is applied.
    owners = {r["user_id"] for r in rows}
    assert owners == {"slack-bot-oid", "discord-bot-oid"}


def test_list_sessions_by_prefixes_empty_prefixes_returns_nothing():
    store = _seeded_store()
    assert asyncio.run(store.list_sessions_by_prefixes(())) == []


def test_list_sessions_by_prefixes_single_platform_filter():
    store = _seeded_store()
    rows = asyncio.run(store.list_sessions_by_prefixes(("slack-thread:",)))
    assert {r["thread_id"] for r in rows} == {"slack-thread:C1:1700000000.0001"}


def test_list_sessions_regression_still_owner_scoped():
    # Regression: the existing per-owner ``list_sessions`` is untouched -- each
    # owner sees only their own thread, and a bot owner does NOT see the web
    # thread (or vice-versa).
    store = _seeded_store()
    web = asyncio.run(store.list_sessions("web-user-oid"))
    assert {r["thread_id"] for r in web} == {"user_abcd"}

    slack = asyncio.run(store.list_sessions("slack-bot-oid"))
    assert {r["thread_id"] for r in slack} == {"slack-thread:C1:1700000000.0001"}

    # include_all (admin path) still returns everything, unchanged.
    every = asyncio.run(store.list_sessions("anyone", include_all=True))
    assert {r["thread_id"] for r in every} == {
        "slack-thread:C1:1700000000.0001",
        "discord-thread:G3:42",
        "telegram-dm:U2",
        "user_abcd",
    }
