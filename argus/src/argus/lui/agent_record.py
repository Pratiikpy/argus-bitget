"""Where the Track 2 agent publishes its record, and the one reader of it.

Shared by the ``/agent`` page (`lui/agent_page.py`) and the console's answer about the agent
(`lui/agent_answer.py`): a page module may be imported only by channels and pages, so the location
and the reader live here, and both read the same file the same way.
"""

from __future__ import annotations

import re
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


def rejections(orders: Mapping[str, Any] | None) -> tuple[int, str]:
    """The orders the venue refused, counted and explained from the record's own order log.

    Run 2 sent 93 orders and 74 were refused, and the page showed "93 orders sent, 19 fills" with
    nothing between them (a judge, round 20, row 717). Grouped by what was refused and why, with
    the time span, so a reader sees a cluster of retries rather than 74 separate failures."""
    rows = [o for o in (orders or {}).get("orders") or [] if o.get("state") == "rejected"]
    if not rows:
        return 0, ""
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for o in rows:
        message = str((o.get("rejection") or {}).get("message") or "no message")
        message = re.sub(r"^HTTP \d+ from Bitget:\s*", "", message)[:90]
        key = (str(o.get("symbol")), str(o.get("purpose") or "order").replace("_", " "), message)
        groups.setdefault(key, []).append(o)
    parts = []
    for (symbol, purpose, message), group in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        stamps = sorted(str(o.get("submitted_at") or "")[:16] for o in group)
        span = (f"{stamps[0]}Z" if stamps[0] == stamps[-1] else
                f"{stamps[0]}Z to {stamps[-1][11:]}Z") if stamps and stamps[0] else ""
        parts.append(f"{len(group)} {purpose}{'s' if len(group) > 1 else ''} on {symbol}"
                     + (f", {span}" if span else "") + f" — “{message}”")
    every = (orders or {}).get("orders") or []
    later = all(any(o.get("state") == "filled" and o.get("symbol") == symbol
                    and str(o.get("submitted_at") or "") > max(str(g.get("submitted_at") or "")
                                                               for g in group)
                    for o in every)
                for (symbol, _, _), group in groups.items())
    retried = max(len(g) for g in groups.values())
    sent = (orders or {}).get("counts", {}).get("sent")
    return len(rows), (
        f"The venue refused {len(rows)}" + (f" of the {sent} orders sent" if sent else " orders")
        + ": " + "; ".join(parts) + "."
        + (" Each name had a later order filled, so the refusals were the venue's "
           "state at the time rather than an order it could never take." if later else "")
        + (f" The agent has no back-off on a refused exit: it sent the same one {retried} times. "
           f"Recorded as a known defect and not changed mid-run, so the scored record stays the "
           f"one that ran." if retried > 3 else ""))
