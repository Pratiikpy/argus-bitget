"""Scheduled briefs on a chat's saved book: before the US open and after the close (build-list 4.9).

**What a brief is.** The guided task's first step (`lui/guide.py`, "Brief me on my book"): each
holding's 24-hour move split into the market's part and its own, the latest headline naming it with
its link, where the book's risk sits, and the holding to look at first. It is asked through the
console's own path (``server.handle_ask``), so a brief is exactly what the console would answer at
that moment — nothing is precomputed or cached for it.

**When.** Two slots, on New York weekdays, in New York time so the daylight-saving change cannot
move them against the market: ``premarket`` at 08:30 (an hour before the open, after the 08:30 data
releases print) and ``close`` at 16:15 (after the closing auction). A slot is sent once per day per
chat; a bot that was down at the slot sends it on its next sweep the same day, never the next day.
US market holidays are not skipped: the brief still reads a live book and says what moved.

**What was taken, and from where.** PanWatch (MIT, the build list's reference) runs pre-market and
close briefs for a saved watchlist with push delivery; the shape — two fixed market-time slots,
one brief per slot, delivered to where the trader already is — is taken; its content is not: the
brief here is the console's own answer. Storage follows `lui/watch.py`: one JSON file outside the
repository (chat ids are personal data), ``ARGUS_BRIEF_STORE`` else ``~/.argus/briefs.json``,
written atomically.

**Where it runs.** In the always-on bot (``python -m argus.lui.telegram_bot --poll``), which checks
the slots on every sweep. Behind the hosted webhook nothing outlives a call, so a subscription made
there could never fire; ``/brief`` answers at once there, and ``/brief daily`` says what it needs.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, time
from pathlib import Path
from typing import Any, Final
from zoneinfo import ZoneInfo

NEW_YORK: Final = ZoneInfo("America/New_York")
SLOTS: Final = {"premarket": time(8, 30), "close": time(16, 15)}
QUESTION: Final = "Brief me on my book: what moved each holding, and what's in the news?"


@dataclass
class Subscription:
    chat_id: int
    slots: list[str] = field(default_factory=lambda: list(SLOTS))
    sent: dict[str, str] = field(default_factory=dict)
    """The New York date each slot was last sent, so a slot goes out once a day."""


class BriefStore:
    """Every chat's brief subscription, in one JSON file, written whole and atomically."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or Path(os.environ.get("ARGUS_BRIEF_STORE", "")
                                 or Path.home() / ".argus" / "briefs.json")
        self._lock = threading.Lock()

    def all(self) -> list[Subscription]:
        try:
            rows = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        out = []
        for row in rows if isinstance(rows, list) else []:
            try:
                slots = [s for s in row.get("slots", []) if s in SLOTS]
                sent = {str(k): str(v) for k, v in (row.get("sent") or {}).items() if k in SLOTS}
                out.append(Subscription(chat_id=int(row["chat_id"]), slots=slots, sent=sent))
            except (KeyError, TypeError, ValueError, AttributeError):
                continue
        return out

    def _write(self, subs: list[Subscription]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle, name = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        with os.fdopen(handle, "w", encoding="utf-8") as out:
            json.dump([asdict(s) for s in subs], out, indent=1)
        os.replace(name, self.path)

    def get(self, chat_id: int) -> Subscription | None:
        return next((s for s in self.all() if s.chat_id == chat_id), None)

    def subscribe(self, chat_id: int, slots: list[str]) -> Subscription:
        with self._lock:
            subs = [s for s in self.all() if s.chat_id != chat_id]
            sub = Subscription(chat_id=chat_id, slots=[s for s in SLOTS if s in slots])
            self._write([*subs, sub])
            return sub

    def unsubscribe(self, chat_id: int) -> bool:
        with self._lock:
            subs = self.all()
            kept = [s for s in subs if s.chat_id != chat_id]
            if len(kept) == len(subs):
                return False
            self._write(kept)
            return True

    def mark_sent(self, chat_id: int, slot: str, day: str) -> None:
        with self._lock:
            subs = self.all()
            for sub in subs:
                if sub.chat_id == chat_id:
                    sub.sent[slot] = day
            self._write(subs)


def due(sub: Subscription, now: datetime) -> list[str]:
    """The slots this chat should be sent now: a weekday in New York, the slot's time passed
    today, and not yet sent today."""
    local = now.astimezone(NEW_YORK)
    if local.weekday() >= 5:
        return []
    today = local.date().isoformat()
    return [slot for slot in sub.slots
            if local.time() >= SLOTS[slot] and sub.sent.get(slot) != today]


def parse_slots(rest: str) -> list[str] | None:
    """``/brief daily`` (both), ``/brief premarket``, ``/brief close``; None when unread."""
    words = rest.lower().split()
    if not words or words[0] in ("daily", "on", "both", "subscribe"):
        return list(SLOTS)
    chosen = [w for w in words if w in SLOTS]
    return chosen or None


def describe(sub: Subscription) -> str:
    when = {"premarket": "08:30 New York, an hour before the open",
            "close": "16:15 New York, after the close"}
    return " and ".join(when[s] for s in sub.slots)


def send_due(store: BriefStore, now: datetime, *, book_of: Any, brief_of: Any,
             deliver: Any) -> list[tuple[int, str]]:
    """Send every brief that is due; return (chat_id, slot) for each one sent. A chat with no
    saved book is told once how to save one rather than sent an empty brief."""
    sent: list[tuple[int, str]] = []
    day = now.astimezone(NEW_YORK).date().isoformat()
    for sub in store.all():
        for slot in due(sub, now):
            book = book_of(sub.chat_id)
            label = "Before the open" if slot == "premarket" else "After the close"
            try:
                text = (brief_of(sub.chat_id, book, label) if book else
                        f"{label}: no book is saved in this chat, so there is nothing to brief. "
                        f"Save one with /book 40% NVDA, 30% MSFT, 30% AAPL.")
                deliver(sub.chat_id, text)
            except Exception:
                continue  # not marked sent: the next sweep today tries again
            store.mark_sent(sub.chat_id, slot, day)
            sent.append((sub.chat_id, slot))
    return sent
