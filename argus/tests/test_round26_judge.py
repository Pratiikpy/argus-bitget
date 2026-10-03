"""Round 26's judge findings, run offline: each new answer is computed from a small hand-made
series whose result can be read off it, and each route is checked to fire on the ask it was built
for and to stay quiet on the near-miss that used to steal it."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest

from argus.lui import thesis_answer
from argus.lui.research import desk_followups, hindsight
from argus.market.history import Candle


def _candle(ts: datetime, close: float) -> Candle:
    px = Decimal(str(close))
    return Candle(ts=ts, open=px, high=px, low=px, close=px, volume=Decimal(1))


def _hours(n: int) -> list[datetime]:
    start = datetime(2026, 9, 1, tzinfo=UTC)
    return [start + timedelta(hours=i) for i in range(n)]


class TestRoutesFireOnTheirOwnAsk:
    def test_coinbase_falling_is_not_a_crypto_crash(self) -> None:
        shock = desk_followups._CRYPTO_SHOCK
        assert shock.search("How exposed am I to a 20% crypto crash?")
        assert shock.search("what if crypto drops 30%")
        assert not shock.search("If COIN drops 15%, what's my portfolio change?")
        assert not shock.search("what if only BTC drops 30%?")

    def test_one_name_beating_another_is_not_an_earnings_beat(self) -> None:
        assert desk_followups.earnings_lines("why did one beat the other?", ["AMD vs NVDA"]) is None

    def test_falsify_reads_what_evidence_would_make_it_wrong(self) -> None:
        assert thesis_answer.FALSIFY.search("what evidence would make this thesis wrong?")
        assert thesis_answer.FALSIFY.search("what data would prove it wrong")
        assert not thesis_answer.FALSIFY.search("what evidence is there for it?")

    def test_unread_figures_are_declined_by_name(self) -> None:
        mstr = desk_followups.unread_lines("how many BTC does MSTR hold?")
        assert mstr is not None and "does not read" in mstr[0] and "8-K" in mstr[0]
        rates = desk_followups.unread_lines("ETH staking yield vs the USDT savings rate?")
        assert rates is not None and "Earn page" in rates[0]
        assert desk_followups.unread_lines("what's the ETH price?") is None


class TestFollowsAndBeta:
    def test_a_name_that_moves_double_has_beta_two(self, monkeypatch: pytest.MonkeyPatch) -> None:
        hours = _hours(200)
        btc = {h: (0.01 if i % 2 else -0.008) * (1 + i % 5 / 10) for i, h in enumerate(hours)}
        coin = {h: 2 * r for h, r in btc.items()}
        monkeypatch.setattr(desk_followups, "_hourly_returns",
                            lambda s, days=30: coin if s.startswith("COIN") else btc)
        lines = desk_followups.follows_lines("and if BTC rises 5% too, does COIN usually follow?",
                                             [])
        assert lines is not None
        assert "2.00x BTC" in lines[0] and "+1.00" in lines[0] and "yes, closely" in lines[0]
        # the stated move, not a fixed 10%: +5% on BTC carries COIN +10%
        assert "BTC +5% would carry COIN about +10.0%" in lines[1]

    def test_weighted_beta_and_the_largest_contributor(self, monkeypatch: pytest.MonkeyPatch
                                                      ) -> None:
        hours = _hours(200)
        btc = {h: (0.01 if i % 2 else -0.01) for i, h in enumerate(hours)}
        series = {"BTCUSDT": btc, "ETHUSDT": {h: 1.5 * r for h, r in btc.items()},
                  "AAPLUSDT": {h: 0.0 + (0.002 if i % 3 else -0.004) for i, h in enumerate(hours)}}
        monkeypatch.setattr(desk_followups, "_hourly_returns", lambda s, days=30: series[s])
        book = "ETH 50%\nAAPL 50%"
        lines = desk_followups.book_beta_lines("What's my weighted beta to BTC?", [], book)
        assert lines is not None and "ETH 50% x beta 1.50 = 0.75" in lines[1]
        most = desk_followups.book_beta_lines("which holding contributes most?",
                                              ["What's my weighted beta to BTC?"], book)
        assert most is not None and most[0].startswith("Bottom line: ETH contributes most")


class TestBookVolatility:
    def test_stated_cash_damps_the_book(self, monkeypatch: pytest.MonkeyPatch) -> None:
        days = [date(2026, 9, 1) + timedelta(days=i) for i in range(30)]
        swing = {d: (0.02 if i % 2 else -0.02) for i, d in enumerate(days)}
        monkeypatch.setattr(desk_followups, "_daily_returns", lambda s, n: swing)
        ask = "My book: 60% BTC, 25% ETH, 15% USDT. What's my 30-day volatility?"
        lines = desk_followups.book_vol_lines(ask, [], "")
        assert lines is not None and "15% cash" in lines[1]
        # both coins swing alike, so the book swings 85% as much as one of them
        alone = desk_followups.book_vol_lines("is that higher than holding BTC alone?", [ask], "")
        assert alone is not None and alone[0].startswith("Bottom line: lower")
        assert "15% in cash damps it" in alone[0]


    def test_a_stablecoin_read_as_a_holding_is_cash(self, monkeypatch: pytest.MonkeyPatch
                                                    ) -> None:
        days = [date(2026, 9, 1) + timedelta(days=i) for i in range(30)]
        swing = {d: (0.01 if i % 2 else -0.01) for i, d in enumerate(days)}
        monkeypatch.setattr(desk_followups, "_daily_returns", lambda s, n: swing)
        lines = desk_followups.book_vol_lines(
            "My portfolio: 50% BTC, 30% SOL, 20% USDC. How volatile has it been over 30 days?",
            [], "")
        assert lines is not None and "BTC 50%, SOL 30%, 20% cash" in lines[1]
        assert "USDC" not in lines[1]


class TestStatedDays:
    TODAY = date(2026, 10, 3)

    @pytest.mark.parametrize(("asked", "start"), [
        ("every Friday this year", date(2026, 1, 1)),
        ("a lump sum on 1 January", date(2026, 1, 1)),
        ("bought on Nov 15th", date(2025, 11, 15)),
        ("since Jan 2025", date(2025, 1, 1)),
    ])
    def test_a_day_without_a_year_is_its_latest_past_one(self, asked: str, start: date) -> None:
        assert hindsight.entry_date(asked, self.TODAY) == start


@dataclass
class _Day:
    day: date
    close: float


class TestAveraging:
    def test_every_monday_buys_on_mondays_only(self) -> None:
        start = date(2026, 1, 1)
        days = [_Day(start + timedelta(days=i), 100.0 + i) for i in range(200)]
        ask = "Did buying ETH every Monday in 2026 beat buying it all on Jan 1?"
        assert hindsight.DCA_Q.search(ask)
        lines, _src, data = hindsight.dca(ask, "ETHUSDT", today=date(2026, 7, 19),
                                          daily=lambda _t: days)
        mondays = sum(1 for d in days if d.day.weekday() == 0)
        assert data["dca"]["buys"] == mondays
        assert "every Monday" in lines[0]
        # a rising series: buying it all on day one comes out ahead
        assert "buying at once came out ahead" in lines[0]


class TestRatiosAndRotation:
    def _closes(self, monkeypatch: pytest.MonkeyPatch, a: float, b: float) -> None:
        from argus.market import history

        def window(symbol: str, **_kw: Any) -> list[Candle]:
            base = datetime(2026, 7, 1, tzinfo=UTC)
            step = a if symbol.startswith("ETH") else b
            return [_candle(base + timedelta(days=i), 100 * (1 + step) ** i) for i in range(95)]

        monkeypatch.setattr(history, "fetch_window", window)
        monkeypatch.setattr(history, "fetch_range", lambda s, **kw: window(s))

    def test_a_ratio_that_compounds_up(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._closes(monkeypatch, 0.002, 0.0)
        lines = desk_followups.ratio_lines("how has ETH/BTC trended in the last 90 days?",
                                           datetime(2026, 10, 3, tzinfo=UTC))
        assert lines is not None and lines[0].startswith("Bottom line: ETH/BTC rose")
        assert "ETH outperformed BTC" in lines[0]

    def test_rotation_conditions_all_met_on_a_rising_ratio(self, monkeypatch: pytest.MonkeyPatch
                                                          ) -> None:
        self._closes(monkeypatch, 0.002, 0.0)
        lines = desk_followups.rotate_conditions_lines(
            "what would you need to see first?", ["Should I rotate from BTC to ETH now?"])
        assert lines is not None and "3 of the 3" in lines[0]
        assert desk_followups.rotate_conditions_lines("what would you need to see first?",
                                                      ["what is BTC at?"]) is None


class TestFundingRule:
    def test_counts_wins_after_negative_settlements(self, monkeypatch: pytest.MonkeyPatch
                                                   ) -> None:
        from argus.market import crossasset_feed, history

        start = datetime(2026, 7, 1, tzinfo=UTC)
        # every settlement 8h apart; odd ones negative, and the price only rises
        settlements = [(int((start + timedelta(hours=8 * i)).timestamp() * 1000),
                        -0.0001 if i % 2 else 0.0001) for i in range(240)]
        monkeypatch.setattr(crossasset_feed, "fetch_funding", lambda s: settlements)
        monkeypatch.setattr(history, "fetch_range", lambda s, **kw: [
            _candle(start + timedelta(hours=i), 100 + i) for i in range(24 * 90)])
        lines = desk_followups.funding_rule_lines(
            "Thesis: buy BTC whenever funding goes negative. Backtest it over a year.", [])
        assert lines is not None
        assert "over the next 24 hours BTC was up 100% (" in lines[0]
        assert "better than a coin flip" in lines[0]
        assert "not a year" in lines[0]
        again = desk_followups.funding_rule_lines(
            "what's the win rate?",
            ["Thesis: buy BTC whenever funding goes negative. Backtest it over a year."])
        assert again == lines
