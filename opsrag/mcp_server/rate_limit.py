"""In-process sliding-window rate limiter for external MCP token calls.

Two caps applied per-token:

  - **Per-tool**: 60 calls / 60s for each (token, tool_name) pair.
    Prevents a single token from hammering one tool (e.g. tight loop
    over `prometheus_query`).
  - **Global**: 600 calls / 3600s per token across all tools.
    Bounds total cost imposed by any single client.

Sliding window is implemented with a `collections.deque` per key. On
each check we drop expired entries from the left, then test length
against the cap. The lock is a plain `threading.Lock` because the
limiter is shared between the async dispatcher and (potentially) the
audit-flush worker; we acquire it for microseconds at a time.

The limiter is process-local -- it does NOT survive a restart and it
does NOT coordinate across pods. That is by design: this layer is the
fast in-pod guard; cross-pod absolute caps belong in Pomerium/Cloud
Armor upstream.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass

_PER_TOOL_WINDOW_S = 60.0
_PER_TOOL_CAP = 60

_GLOBAL_WINDOW_S = 3600.0
_GLOBAL_CAP = 600


@dataclass(frozen=True)
class RateLimitCaps:
    """Knobs exposed for tests. Production code uses module-level
    constants via the no-arg constructor."""

    per_tool_window_s: float = _PER_TOOL_WINDOW_S
    per_tool_cap: int = _PER_TOOL_CAP
    global_window_s: float = _GLOBAL_WINDOW_S
    global_cap: int = _GLOBAL_CAP


class TokenRateLimiter:
    """Sliding-window limiter keyed by token_id (global) and
    (token_id, tool_name) (per-tool).

    Thread-safe. `allow()` is the only mutator: it eagerly records
    the call as "spent" before returning True. If `allow()` returns
    False the call is not recorded (so a denied attempt doesn't push
    a legitimate call out of the window).
    """

    def __init__(self, caps: RateLimitCaps | None = None) -> None:
        self._caps = caps or RateLimitCaps()
        self._lock = threading.Lock()
        # token_id -> deque[timestamp]
        self._global: dict[str, deque[float]] = {}
        # (token_id, tool_name) -> deque[timestamp]
        self._per_tool: dict[tuple[str, str], deque[float]] = {}

    @staticmethod
    def _evict(dq: deque[float], horizon: float) -> None:
        """Pop entries older than `horizon` from the left of the deque."""
        while dq and dq[0] < horizon:
            dq.popleft()

    def allow(
        self, token_id: str, tool_name: str, *, _now: float | None = None
    ) -> tuple[bool, str | None]:
        """Check and record a call.

        Returns `(allowed, reason)`. `reason` is `None` on success;
        on denial it's a short human-readable string suitable for
        embedding in a JSON-RPC error.
        """
        now = time.monotonic() if _now is None else _now
        caps = self._caps
        with self._lock:
            # --- per-tool gate ---
            pt_key = (token_id, tool_name)
            pt_dq = self._per_tool.setdefault(pt_key, deque())
            self._evict(pt_dq, now - caps.per_tool_window_s)
            if len(pt_dq) >= caps.per_tool_cap:
                return (
                    False,
                    (
                        f"per-tool rate limit exceeded: "
                        f"{caps.per_tool_cap} calls / {int(caps.per_tool_window_s)}s "
                        f"on tool {tool_name!r}"
                    ),
                )
            # --- global gate ---
            g_dq = self._global.setdefault(token_id, deque())
            self._evict(g_dq, now - caps.global_window_s)
            if len(g_dq) >= caps.global_cap:
                return (
                    False,
                    (
                        f"global rate limit exceeded: "
                        f"{caps.global_cap} calls / {int(caps.global_window_s)}s "
                        f"on token"
                    ),
                )
            # --- record both ---
            pt_dq.append(now)
            g_dq.append(now)
        return True, None

    def snapshot(self, token_id: str) -> dict:
        """Return current window counts for a token. Diagnostic only."""
        with self._lock:
            g_dq = self._global.get(token_id) or deque()
            per_tool = {
                tool: len(dq)
                for (tok, tool), dq in self._per_tool.items()
                if tok == token_id
            }
            return {
                "global_count": len(g_dq),
                "global_cap": self._caps.global_cap,
                "per_tool_counts": per_tool,
                "per_tool_cap": self._caps.per_tool_cap,
            }
