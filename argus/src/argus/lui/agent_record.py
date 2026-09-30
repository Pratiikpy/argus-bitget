"""Where the Track 2 agent publishes its record, and the one reader of it.

Shared by the ``/agent`` page (`lui/agent_page.py`) and the console's answer about the agent
(`lui/agent_answer.py`): a page module may be imported only by channels and pages, so the location
and the reader live here, and both read the same file the same way.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any

from argus.lui.trace import trace_module
from argus.truth import http

RECORD = "https://t2-sentiment-agent-run2.vercel.app"
"""Run 2, the scored run. ``/agent`` read run 1's record until 2026-09-29, two days after run 1
closed and run 2 began, so it showed a finished run as the live one (first-user audit)."""
RUN_1_RECORD = "https://t2-sentiment-agent-live.vercel.app"
"""Run 1: disclosed, not scored."""
REPO = "https://github.com/Pratiikpy/t2-sentiment-agent"
CACHE_SECONDS = 300.0
"""The record republishes hourly; five minutes keeps a busy page from refetching on every view
while never showing a record more than one publish behind."""

Fetch = Callable[[str], Mapping[str, Any] | None]
_CACHE: dict[str, tuple[float, Mapping[str, Any] | None]] = {}


def fetch_json(name: str) -> Mapping[str, Any] | None:
    """One file of the public record, or None when it cannot be read. Cached briefly."""
    hit = _CACHE.get(name)
    if hit is not None and time.monotonic() - hit[0] < CACHE_SECONDS:
        return hit[1]
    try:
        loaded = http.fetch_json(f"{RECORD}/{name}", timeout=8)
    except http.RpcError:
        loaded = None
    value = loaded if isinstance(loaded, dict) else None
    if value is not None:
        _CACHE[name] = (time.monotonic(), value)
    return value


trace_module(globals())
