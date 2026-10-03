"""Tripwires (lui/research/tripwire.py): reading the trader's plan, keeping it in memory, and
replaying it against hourly candles made by hand, so each verdict can be read off the series."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest

from argus.lui import memory as mem
from argus.lui.research import tripwire
from argus.market.history import Candle

SET = datetime(2026, 10, 3, 9, 30, tzinfo=UTC)


def _candle(ts: datetime, low: float, high: float, close: float) -> Candle:
    return Candle(ts=ts, open=Decimal(str(close)), high=Decimal(str(high)),
                  low=Decimal(str(low)), close=Decimal(str(close)), volume=Decimal(1))


def _series(monkeypatch: pytest.MonkeyPatch, rows: list[tuple[int, float, float, float]]) -> None:
    """``rows``: (hour offset from 09:00 on the set day, low, high, close)."""
    from argus.market import history

    base = SET.replace(minute=0)

    def window(symbol: str, **_kw: Any) -> list[Candle]:
        return [_candle(base + timedelta(hours=h), lo, hi, c) for h, lo, hi, c in rows]

    monkeypatch.setattr(history, "fetch_window", window)


class TestReading:
    @pytest.mark.parametrize(("said", "side", "level", "action"), [
        ("If BTC drops below 80k I'll sell half", "below", 80000.0, "reduce"),
        ("once SOL climbs above 150 I would take profits", "above", 150.0, "reduce"),
        ("If ETH climbs above 3000 I will add 10%", "above", 3000.0, "add"),
        ("if ETH hits 2400 I am going to hedge my book", "below", 2400.0, "hedge"),
    ])
    def test_plans_are_read(self, said: str, side: str, level: float, action: str) -> None:
        plan = tripwire.read(said, price_now=2600 if "ETH" in said else None, now=SET)
        assert plan is not None
        assert (plan.side, plan.level, plan.action) == (side, level, action)
        assert plan.set_at == int(SET.timestamp())

    def test_an_unsided_level_needs_the_price(self) -> None:
        said = "when NVDA breaks 200, I\u2019ll add 10%"
        assert tripwire.read(said) is None
        plan = tripwire.read(said, price_now=180)
        assert plan is not None
        assert plan.side == "above"

    def test_not_a_plan(self) -> None:
        assert tripwire.read("what is BTC trading at") is None
        assert tripwire.read("BTC dropped below 80k yesterday") is None

    def test_value_round_trip_and_memory(self) -> None:
        plan = tripwire.read("If BTC drops below 80k I'll sell half", now=SET)
        assert plan is not None
        value = tripwire.value_of(plan)
        assert len(value) <= 40
        assert tripwire.from_value("BTCUSDT", value, plan.words) == plan
        facts = mem.extract("If BTC drops below 80k I'll sell half", SET,
                            price_of=lambda _s: 84000.0)
        (fact,) = [f for f in facts if f.kind == "tripwire"]
        # one tripwire per side of a name: a stop below and a take-profit above both stay
        assert (fact.subject, fact.price_at) == ("BTCUSDT|below", 84000.0)
        assert tripwire.symbol_of(fact.subject) == "BTCUSDT"
        assert mem.parse(mem.dumps(facts))[0].kind == "tripwire"  # survives the browser store


class TestReplay:
    PLAN = tripwire.Plan(symbol="BTCUSDT", side="below", level=80000.0, action="reduce",
                         words="If BTC drops below 80k I'll sell half",
                         set_at=int(SET.timestamp()))

    def test_a_level_traded_before_the_plan_was_set_does_not_fire_it(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        # 09:00 (the hour the plan was set in) dipped to 79,500; nothing after it did
        _series(monkeypatch, [(0, 79500, 81000, 80500), (1, 80200, 81000, 80800),
                              (2, 80300, 81200, 81000)])
        status = tripwire.check(self.PLAN, "2026-10-03", 80500.0, now=SET + timedelta(hours=3))
        assert status is not None
        assert status.fired_at is None
        assert "not reached" in tripwire.status_line(status)

    def test_a_fire_is_replayed_from_the_level(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _series(monkeypatch, [(0, 80500, 81000, 80800), (1, 79800, 80600, 79900),
                              (2, 75000, 79900, 76000)])
        status = tripwire.check(self.PLAN, "2026-10-03", 80800.0, now=SET + timedelta(hours=3))
        assert status is not None
        assert status.fired_at == SET.replace(minute=0) + timedelta(hours=1)
        line = tripwire.status_line(status)
        # 76,000 against the 80,000 level: stepping out avoided a 5.0% further fall
        assert "fired" in line
        assert "avoided a 5.0% further fall" in line

    def test_adding_after_a_breakout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        plan = tripwire.Plan(symbol="ETHUSDT", side="above", level=3000.0, action="add",
                             words="If ETH climbs above 3000 I will add 10%",
                             set_at=int(SET.timestamp()))
        _series(monkeypatch, [(0, 2900, 2950, 2940), (1, 2950, 3010, 3005), (2, 3000, 3200, 3150)])
        status = tripwire.check(plan, "2026-10-03", 2940.0, now=SET + timedelta(hours=3))
        assert status is not None
        assert "adding then caught a 5.0% rise" in tripwire.status_line(status)

    @pytest.mark.parametrize(("said", "side", "action"), [
        # bare present-tense verbs a first-time user typed (round 27)
        ("if btc drops under 80k i sell all", "below", "reduce"),
        ("if eth goes above 3000 i take profit", "above", "reduce"),
    ])
    def test_bare_verbs_are_plans(self, said: str, side: str, action: str) -> None:
        plan = tripwire.read(said, now=SET)
        assert plan is not None and (plan.side, plan.action) == (side, action)

    def test_a_stop_and_a_take_profit_on_one_name_are_two_facts(self) -> None:
        below = mem.extract("if ETH drops below 2400 I sell all", SET)
        above = mem.extract("if ETH climbs above 3000 I take profit", SET)
        kept = mem.merge(below, above)
        trips = [f for f in kept if f.kind == "tripwire"]
        assert {f.subject for f in trips} == {"ETHUSDT|below", "ETHUSDT|above"}
        assert tripwire.latest(kept) is not None

    def test_change_delete_and_follow_up_asks(self) -> None:
        changed = tripwire.CHANGE.search("change it to 85k")
        assert changed is not None and changed.group("lvl") == "85" and changed.group("k")
        gone = tripwire.DELETE.search("delete the eth one")
        assert gone is not None and gone.group("sym") == "eth"
        assert tripwire.DELETE.search("remove all my tripwires")
        assert tripwire.FOLLOW_UP.search("how far are we from it")
        assert tripwire.about_tripwires(["if BTC drops below 80k I'll sell half"])
        assert not tripwire.about_tripwires(["what is BTC trading at"])

    def test_the_ask_reads_check_my_tripwires(self) -> None:
        assert tripwire.ASKED.search("check my tripwires")
        assert tripwire.ASKED.search("did my tripwire fire?")
        assert not tripwire.ASKED.search("what is a tripwire in trading")
