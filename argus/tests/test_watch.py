"""Alerts on a level or a funding extreme (`lui/watch.py`) and the bot's commands, offline."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from argus.lui import telegram_bot as bot
from argus.lui import watch


def _resolve(name: str) -> tuple[str, str] | None:
    table = {"NVDA": "NVDAUSDT", "BTC": "BTCUSDT", "MSTR": "MSTRUSDT"}
    symbol = table.get(name.upper())
    return None if symbol is None else (symbol, "")


@dataclass
class _Ticker:
    last: Decimal
    change_24h: Decimal
    funding_rate: Decimal


@pytest.mark.parametrize(("text", "expected"), [
    ("NVDA below 170", ("NVDAUSDT", "below", 170.0)),
    ("btc above 100,000", ("BTCUSDT", "above", 100000.0)),
    ("NVDA < 170.5", ("NVDAUSDT", "below", 170.5)),
    ("MSTR funding 0.05%", ("MSTRUSDT", "funding", 0.0005)),
    ("MSTR funding", ("MSTRUSDT", "funding", watch.DEFAULT_FUNDING)),
])
def test_watches_are_read(text: str, expected: tuple[str, str, float]) -> None:
    got = watch.parse(text, resolve=_resolve)
    assert not isinstance(got, str)
    assert got[:2] == expected[:2]
    assert got[2] == pytest.approx(expected[2])


@pytest.mark.parametrize(("text", "says"), [
    ("", "Say what to watch"), ("NVDA sideways", "Say what to watch"),
    ("ZZZZ below 3", "Bitget lists no contract called ZZZZ"),
])
def test_bad_watches_say_what_is_wrong(text: str, says: str) -> None:
    got = watch.parse(text, resolve=_resolve)
    assert isinstance(got, str) and got.startswith(says)


def test_a_watch_fires_once_and_only_when_met(tmp_path: Path) -> None:
    store = watch.WatchStore(tmp_path / "w.json")
    store.add(7, "NVDAUSDT", "below", 170.0)
    store.add(7, "MSTRUSDT", "funding", 0.0005)
    store.add(8, "BTCUSDT", "above", 100000.0)
    quiet = {"NVDAUSDT": _Ticker(Decimal("171"), Decimal("0.01"), Decimal("0.0001")),
             "MSTRUSDT": _Ticker(Decimal("150"), Decimal("0"), Decimal("-0.0004")),
             "BTCUSDT": _Ticker(Decimal("99999"), Decimal("0"), Decimal("0.0001"))}
    assert watch.sweep(store, quiet) == []
    moved = {**quiet, "NVDAUSDT": _Ticker(Decimal("169.8"), Decimal("-0.02"), Decimal("0")),
             "MSTRUSDT": _Ticker(Decimal("150"), Decimal("0"), Decimal("-0.0006"))}
    alerts = watch.sweep(store, moved)
    assert [chat for chat, _ in alerts] == [7, 7]
    assert alerts[0][1].startswith("Alert: NVDA traded 169.8 on Bitget, at or below your 170")
    assert "shorts are paying longs" in alerts[1][1]
    assert watch.sweep(store, moved) == []
    assert [w.symbol for w in store.all()] == ["BTCUSDT"]


def test_a_chat_is_capped_and_can_remove_and_clear(tmp_path: Path) -> None:
    store = watch.WatchStore(tmp_path / "w.json")
    for n in range(watch.MAX_PER_CHAT):
        assert not isinstance(store.add(1, "NVDAUSDT", "above", 200.0 + n), str)
    assert isinstance(store.add(1, "NVDAUSDT", "above", 999.0), str)
    removed = store.remove(1, 1)
    assert removed is not None and removed.level == 200.0
    assert store.remove(1, 99) is None
    store.clear(1)
    assert store.for_chat(1) == []


def _update(text: str, chat: int = 5) -> dict[str, Any]:
    return {"update_id": 1, "message": {"chat": {"id": chat}, "text": text}}


def _no_ask(*a: Any, **k: Any) -> dict[str, Any]:
    raise AssertionError("a command must not reach the desk")


def test_the_bot_sets_lists_and_removes_watches(tmp_path: Path,
                                                monkeypatch: pytest.MonkeyPatch) -> None:
    from argus.market import universe

    monkeypatch.setattr(universe, "resolve", _resolve)
    store = watch.WatchStore(tmp_path / "w.json")
    states: dict[int, bot.ChatState] = {}
    set_reply = bot.handle_update(_update("/watch NVDA below 170"), states, ask=_no_ask,
                                  watches=store)
    assert set_reply[0][1].startswith("Watching NVDA below 170.")
    listed = bot.handle_update(_update("/watches"), states, ask=_no_ask, watches=store)
    assert listed[0][1].startswith("Your watches: 1. NVDA below 170")
    removed = bot.handle_update(_update("/unwatch 1"), states, ask=_no_ask, watches=store)
    assert removed[0][1] == "Removed: NVDA below 170."


def test_the_webhook_bot_says_why_it_keeps_no_watches() -> None:
    reply = bot.handle_update(_update("/watch NVDA below 170"), {}, ask=_no_ask)
    assert reply[0][1].startswith("Watches need the always-on bot")
