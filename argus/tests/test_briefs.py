"""Scheduled briefs on a chat's saved book (`lui/briefs.py`, build-list 4.9), offline."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from argus.lui import briefs, telegram_bot
from argus.lui.telegram_bot import ChatState, handle_update

# Monday 5 Oct 2026: New York is on daylight time, UTC-4
BEFORE_OPEN = datetime(2026, 10, 5, 12, 0, tzinfo=UTC)    # 08:00 New York
AFTER_PREMARKET = datetime(2026, 10, 5, 12, 31, tzinfo=UTC)  # 08:31
AFTER_CLOSE = datetime(2026, 10, 5, 20, 16, tzinfo=UTC)   # 16:16
SATURDAY = datetime(2026, 10, 10, 14, 0, tzinfo=UTC)


def _message(text: str, chat: int = 7) -> dict[str, Any]:
    return {"message": {"chat": {"id": chat}, "text": text}}


class TestSchedule:
    def test_slots_fall_in_new_york_time(self) -> None:
        sub = briefs.Subscription(chat_id=1)
        assert briefs.due(sub, BEFORE_OPEN) == []
        assert briefs.due(sub, AFTER_PREMARKET) == ["premarket"]
        assert briefs.due(sub, AFTER_CLOSE) == ["premarket", "close"]
        assert briefs.due(sub, SATURDAY) == []
        # in winter New York is UTC-5: 13:31 UTC is 08:31 there
        assert briefs.due(sub, datetime(2026, 12, 7, 13, 31, tzinfo=UTC)) == ["premarket"]
        assert briefs.due(sub, datetime(2026, 12, 7, 12, 31, tzinfo=UTC)) == []

    def test_a_slot_goes_once_a_day(self) -> None:
        sub = briefs.Subscription(chat_id=1, sent={"premarket": "2026-10-05"})
        assert briefs.due(sub, AFTER_CLOSE) == ["close"]
        assert briefs.due(briefs.Subscription(chat_id=1, slots=["close"]), AFTER_PREMARKET) == []

    def test_reading_the_command(self) -> None:
        assert briefs.parse_slots("daily") == ["premarket", "close"]
        assert briefs.parse_slots("") == ["premarket", "close"]
        assert briefs.parse_slots("close") == ["close"]
        assert briefs.parse_slots("whenever") is None


class TestSending:
    def test_due_briefs_are_sent_and_marked(self, tmp_path: Path) -> None:
        store = briefs.BriefStore(tmp_path / "briefs.json")
        store.subscribe(1, ["premarket", "close"])
        store.subscribe(2, ["close"])
        books = {1: "40% NVDA, 60% BTC", 2: ""}
        sent: list[tuple[int, str]] = []
        got = briefs.send_due(store, AFTER_CLOSE, book_of=books.get,
                              brief_of=lambda chat, book, label: f"{label}: {book}",
                              deliver=lambda chat, text: sent.append((chat, text)))
        assert sorted(got) == [(1, "close"), (1, "premarket"), (2, "close")]
        assert (1, "Before the open: 40% NVDA, 60% BTC") in sent
        assert any(chat == 2 and "no book is saved" in text for chat, text in sent)
        # a second sweep the same day sends nothing
        assert briefs.send_due(store, AFTER_CLOSE, book_of=books.get,
                               brief_of=lambda *a: "x", deliver=lambda *a: None) == []

    def test_a_failed_delivery_is_tried_again(self, tmp_path: Path) -> None:
        store = briefs.BriefStore(tmp_path / "briefs.json")
        store.subscribe(1, ["premarket"])

        def down(chat: int, text: str) -> None:
            raise OSError("telegram unreachable")

        assert briefs.send_due(store, AFTER_PREMARKET, book_of=lambda c: "100% BTC",
                               brief_of=lambda *a: "brief", deliver=down) == []
        assert briefs.due(store.all()[0], AFTER_PREMARKET) == ["premarket"]


class TestBot:
    def test_a_bare_brief_asks_the_briefing(self) -> None:
        asked: list[str] = []

        def ask(text: str, prior: list[str], **kw: Any) -> dict[str, Any]:
            asked.append(text)
            return {"lines": ["Bottom line: the thread to pull is BTC."], "sources": []}

        states = {7: ChatState(book="50% BTC, 50% ETH")}
        replies = handle_update(_message("/brief"), states, ask=ask)
        assert asked == [telegram_bot.BRIEF_QUESTION]
        assert "thread to pull" in replies[0][1]
        # without a book it says how to save one, and asks nothing
        assert "none is saved" in handle_update(_message("/brief", chat=8), {}, ask=ask)[0][1]

    def test_a_schedule_needs_the_always_on_bot(self, tmp_path: Path) -> None:
        states = {7: ChatState(book="50% BTC, 50% ETH")}
        said = handle_update(_message("/brief daily"), states)[0][1]
        assert said.startswith("Scheduled briefs need the always-on bot")
        store = briefs.BriefStore(tmp_path / "briefs.json")
        said = handle_update(_message("/brief daily"), states, briefs=store)[0][1]
        assert "08:30 New York" in said and "16:15 New York" in said
        assert store.get(7) is not None
        assert handle_update(_message("/brief off"), states, briefs=store)[0][1] == (
            "Scheduled briefs stopped.")
        assert store.get(7) is None
