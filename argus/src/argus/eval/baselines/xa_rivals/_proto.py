"""Line protocol shared by the Python rival runners (one JSON object per line on stdin/stdout).

A runner is started by `eval/xa_arena.py` in its own process with the rival's clone as working
directory. It receives the tape one bar at a time — never the future — keeps its own history, and
answers each ``step`` with orders. Rival code sometimes prints; ``sys.stdout`` is pointed at stderr
while rival code runs so a stray print can never corrupt the protocol.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from typing import Any, TextIO

HOUR_MS = 3_600_000


class Tape:
    """What the runner has been shown so far."""

    def __init__(self, init: dict[str, Any]) -> None:
        self.init = init
        self.keys: list[str] = list(init["keys"])
        self.hours: list[int] = []
        self.bars: dict[str, list[list[float] | None]] = {k: [] for k in self.keys}
        self.funding: dict[str, list[tuple[int, float]]] = {}
        self.fng: list[tuple[int, float]] = []

    def push(self, msg: dict[str, Any]) -> None:
        i = int(msg["i"])
        if len(self.hours) > i:
            return
        self.hours.append(int(msg["ts"]) - HOUR_MS)
        for k in self.keys:
            self.bars[k].append(msg["bars"].get(k))
        for sym, ts, rate in msg.get("funding", []):
            self.funding.setdefault(sym, []).append((int(ts), float(rate)))
        for ts, value in msg.get("fng", []):
            self.fng.append((int(ts), float(value)))

    @property
    def i(self) -> int:
        return len(self.hours) - 1

    def close(self, key: str, back: int = 0) -> float | None:
        """Close ``back`` bars before the latest; None when that bar did not exist."""
        j = self.i - back
        if j < 0:
            return None
        b = self.bars[key][j]
        return None if b is None else float(b[3])

    def change(self, key: str, hours: int) -> float | None:
        a, b = self.close(key, hours), self.close(key)
        return None if a is None or b is None or a <= 0 else b / a - 1

    def daily_rows(self, key: str, *, limit: int) -> list[list[float]]:
        """``[ts, open, high, low, close]`` UTC-day candles from hourly bars, the current partial
        day last (as the venue's candles endpoint returns it)."""
        days: dict[int, list[float]] = {}
        for j, ts in enumerate(self.hours):
            b = self.bars[key][j]
            if b is None:
                continue
            d = ts // (24 * HOUR_MS) * 24 * HOUR_MS
            if d not in days:
                days[d] = [float(d), b[0], b[1], b[2], b[3]]
            else:
                row = days[d]
                row[2] = max(row[2], b[1])
                row[3] = min(row[3], b[2])
                row[4] = b[3]
        rows = [days[d] for d in sorted(days)]
        return rows[-limit:]

    def hourly_rows(self, key: str, *, limit: int) -> list[list[float]]:
        """``[ts, open, high, low, close, baseVol, quoteVol]`` rows, oldest first."""
        out: list[list[float]] = []
        for j in range(max(0, self.i - limit + 1), self.i + 1):
            b = self.bars[key][j]
            if b is not None:
                out.append([float(self.hours[j]), b[0], b[1], b[2], b[3], 0.0, b[4]])
        return out


def serve(make: Callable[[dict[str, Any], Tape], Any]) -> None:
    """Run the protocol loop. ``make(init, tape)`` returns an object with ``step(msg, tape)``
    returning a reply dict and ``finish()`` returning a diagnostics dict."""
    out: TextIO = sys.stdout
    sys.stdout = sys.stderr
    tape: Tape | None = None
    agent: Any = None
    for line in sys.stdin:
        if not line.strip():
            continue
        msg = json.loads(line)
        kind = msg.get("type")
        reply: dict[str, Any] | None
        try:
            if kind == "init":
                tape = Tape(msg)
                agent = make(msg, tape)
                reply = {"ok": True}
            elif kind == "bar":
                assert tape is not None
                tape.push(msg)
                reply = None
            elif kind == "step":
                assert tape is not None
                tape.push(msg)
                reply = agent.step(msg, tape)
            elif kind == "finish":
                reply = agent.finish() if agent is not None else {}
            else:
                reply = {"error": f"unknown message type {kind!r}"}
        except Exception as exc:
            import traceback

            reply = {"error": f"{type(exc).__name__}: {exc}",
                     "trace": traceback.format_exc()[-3000:]}
        if reply is not None:
            out.write(json.dumps(reply, default=str) + "\n")
            out.flush()
        if kind == "finish":
            break
