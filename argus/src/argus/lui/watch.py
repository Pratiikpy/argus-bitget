"""Alerts: a trader names a level, and the desk says when the market reaches it.

**Why this exists.** Rivals watch a thesis and message the trader when its invalidation level
trades; ARGUS answered questions and then forgot them (audit finding 25). A trader who asks "where
is my stop?" wants to hear when it is reached, not to keep asking.

**What can be watched.** A price level on any contract Bitget lists — ``/watch NVDA below 170``,
``/watch BTC above 100000`` — and a funding extreme, ``/watch MSTR funding 0.05%``, which fires when
the live rate per settlement reaches that size in either direction (crowded positioning, the
signal the funding answers read). Levels are checked against Bitget's own ticker board, one call
for every contract, so a sweep costs one request however many watches exist.

**How it behaves.** A watch fires once and is then removed, so a price that hovers around a level
does not message every minute; the alert says so and how to set it again. Each chat keeps at most
:data:`MAX_PER_CHAT`. Watches are kept in a JSON file outside the repository (chat ids are personal
data): ``ARGUS_WATCH_STORE``, else ``~/.argus/watches.json``, written atomically.

**Where it runs.** Only in the always-on bot (``python -m argus.lui.telegram_bot --poll``), which
sweeps every :data:`SWEEP_SECONDS`. The webhook behind the hosted console runs as short-lived
serverless calls with no disk that outlives them, so there a watch could never fire; the bot says
that instead of pretending to keep one.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MAX_PER_CHAT = 10
SWEEP_SECONDS = 60.0
DEFAULT_FUNDING = 0.0005
"""0.05% per settlement: about five times Bitget's usual 0.01%, where funding answers call a side
crowded."""

_LEVEL = re.compile(
    r"^\s*(?P<name>[A-Za-z0-9.]{1,24})\s+(?P<op>above|below|over|under|>=?|<=?)\s*\$?"
    r"(?P<level>\d[\d,]*(?:\.\d+)?)\s*$", re.I)
_FUNDING = re.compile(
    r"^\s*(?P<name>[A-Za-z0-9.]{1,24})\s+funding(?:\s+(?P<level>\d+(?:\.\d+)?)\s*(?P<pct>%)?)?\s*$",
    re.I)


@dataclass(frozen=True, slots=True)
class Watch:
    id: str
    chat_id: int
    symbol: str
    kind: str
    """``above``, ``below`` or ``funding``."""
    level: float
    set_at: str

    def describe(self) -> str:
        name = self.symbol.removesuffix("USDT")
        if self.kind == "funding":
            return f"{name} funding at {self.level:.3%} a settlement or more, either side"
        return f"{name} {self.kind} {self.level:,.6g}"


def parse(text: str, resolve: Any = None) -> tuple[str, str, float] | str:
    """``(symbol, kind, level)`` from the words after ``/watch``, or what was wrong with them."""
    if resolve is None:
        from argus.market.universe import resolve
    level_match = _LEVEL.match(text)
    funding_match = _FUNDING.match(text)
    match = level_match or funding_match
    if match is None:
        return ("Say what to watch, for example: /watch NVDA below 170 · /watch BTC above 100000 "
                "· /watch MSTR funding 0.05%")
    found = resolve(match.group("name"))
    if found is None:
        return f"Bitget lists no contract called {match.group('name')}."
    symbol = str(found[0])
    if funding_match is not None:
        raw = funding_match.group("level")
        level = DEFAULT_FUNDING if raw is None else float(raw) / (
            100 if funding_match.group("pct") or float(raw) >= 0.01 else 1)
        return symbol, "funding", level
    op = str(level_match.group("op")).lower() if level_match else ""
    kind = "above" if op in ("above", "over", ">", ">=") else "below"
    level = float(str(level_match.group("level")).replace(",", "")) if level_match else 0.0
    if level <= 0:
        return "A price level must be above zero."
    return symbol, kind, level


class WatchStore:
    """Every chat's watches, in one JSON file, written whole and atomically."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or Path(os.environ.get("ARGUS_WATCH_STORE", "")
                                 or Path.home() / ".argus" / "watches.json")
        self._lock = threading.Lock()

    def all(self) -> list[Watch]:
        try:
            rows = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        out = []
        for row in rows if isinstance(rows, list) else []:
            try:
                out.append(Watch(id=str(row["id"]), chat_id=int(row["chat_id"]),
                                 symbol=str(row["symbol"]), kind=str(row["kind"]),
                                 level=float(row["level"]), set_at=str(row["set_at"])))
            except (KeyError, TypeError, ValueError):
                continue
        return out

    def _write(self, watches: list[Watch]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle, name = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        with os.fdopen(handle, "w", encoding="utf-8") as out:
            json.dump([asdict(w) for w in watches], out, indent=1)
        os.replace(name, self.path)

    def for_chat(self, chat_id: int) -> list[Watch]:
        return [w for w in self.all() if w.chat_id == chat_id]

    def add(self, chat_id: int, symbol: str, kind: str, level: float,
            now: datetime | None = None) -> Watch | str:
        with self._lock:
            watches = self.all()
            if sum(1 for w in watches if w.chat_id == chat_id) >= MAX_PER_CHAT:
                return (f"This chat already has {MAX_PER_CHAT} watches; remove one with "
                        f"/unwatch first.")
            watch = Watch(id=uuid.uuid4().hex[:8], chat_id=chat_id, symbol=symbol, kind=kind,
                          level=level,
                          set_at=(now or datetime.now(UTC)).isoformat(timespec="minutes"))
            self._write([*watches, watch])
            return watch

    def remove(self, chat_id: int, index: int) -> Watch | None:
        """The chat's ``index``-th watch (1-based, as /watches lists them), removed."""
        with self._lock:
            watches = self.all()
            mine = [w for w in watches if w.chat_id == chat_id]
            if not 1 <= index <= len(mine):
                return None
            gone = mine[index - 1]
            self._write([w for w in watches if w.id != gone.id])
            return gone

    def discard(self, ids: set[str]) -> None:
        with self._lock:
            self._write([w for w in self.all() if w.id not in ids])

    def clear(self, chat_id: int) -> None:
        with self._lock:
            self._write([w for w in self.all() if w.chat_id != chat_id])


def fired(watch: Watch, ticker: Any) -> str | None:
    """The alert text when ``ticker`` (a Bitget ticker) meets the watch, else None."""
    name = watch.symbol.removesuffix("USDT")
    last = float(ticker.last)
    day = float(ticker.change_24h)
    if watch.kind == "above" and last >= watch.level:
        side = f"at or above your {watch.level:,.6g}"
    elif watch.kind == "below" and last <= watch.level:
        side = f"at or below your {watch.level:,.6g}"
    elif watch.kind == "funding" and abs(float(ticker.funding_rate)) >= watch.level:
        rate = float(ticker.funding_rate)
        crowd = "longs are paying shorts" if rate > 0 else "shorts are paying longs"
        return (f"Alert: {name} funding is {rate:+.3%} a settlement, past your {watch.level:.3%} "
                f"— {crowd}, the sign of a crowded side (watch set {watch.set_at[:10]}). Ask "
                f"“is {name} crowded?” for the full read. This watch is now removed; "
                f"/watch sets it again.")
    else:
        return None
    return (f"Alert: {name} traded {last:,.6g} on Bitget, {side} ({day:+.1%} on the day; watch "
            f"set {watch.set_at[:10]}). Ask “what next for {name}?” for the analysis. "
            f"This watch is now removed; /watch sets it again.")


def sweep(store: WatchStore, tickers: Mapping[str, Any]) -> list[tuple[int, str]]:
    """Every watch the board meets, as (chat_id, text); those watches are removed."""
    alerts: list[tuple[int, str]] = []
    done: set[str] = set()
    for watch in store.all():
        ticker = tickers.get(watch.symbol)
        if ticker is None:
            continue
        text = fired(watch, ticker)
        if text is not None:
            alerts.append((watch.chat_id, text))
            done.add(watch.id)
    if done:
        store.discard(done)
    return alerts


__all__ = ["DEFAULT_FUNDING", "MAX_PER_CHAT", "SWEEP_SECONDS", "Watch", "WatchStore", "fired",
           "parse", "sweep"]
