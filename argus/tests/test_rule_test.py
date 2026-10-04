"""Rules in plain words run as backtests (`lui/research/rule_test.py`), with the regime split of
build-list 3.5. Offline: every price series is generated here."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest

from argus.lui import server
from argus.lui.research import rule_test as rt


def _wave(n: int = 900, drift: float = 0.0004) -> list[float]:
    """A price with cycles and a drift: up legs, down legs and sideways stretches."""
    return [100 * math.exp(drift * i + 0.25 * math.sin(i / 37) + 0.08 * math.sin(i / 7))
            for i in range(n)]


class TestReading:
    @pytest.mark.parametrize(("text", "kind", "params"), [
        ("backtest buying BTC when RSI drops below 30 and selling above 70", "rsi",
         {"n": 14, "lo": 30, "hi": 70}),
        ("backtest RSI(9) under 25 on ETH", "rsi", {"n": 9, "lo": 25, "hi": 75}),
        ("how would a 50/200 day moving average crossover on ETH have done?", "cross",
         {"fast": 50, "slow": 200, "ema": 0.0}),
        ("backtest the 20 and 50 EMA cross on SOL", "cross", {"fast": 20, "slow": 50, "ema": 1.0}),
        ("backtest holding BTC only above its 200-day average", "above", {"n": 200}),
        ("backtest buying the 20-day high on SOL, selling the 10-day low", "breakout",
         {"n": 20, "m": 10}),
        ("test a strategy: buy NVDA when it falls 3% in a day, hold a week", "move",
         {"pct": 0.03, "down": 1.0, "days": 7, "sessions": 5}),
        ("backtest buying ETH after it jumps 5%, hold 3 days", "move",
         {"pct": 0.05, "down": 0.0, "days": 3}),
        # past tense, as a trader asks it on the live console (2026-10-05)
        ("would buying ETH every time it dropped 5% in a day and holding 10 days have made "
         "money? backtest it", "move", {"pct": 0.05, "down": 1.0, "days": 10}),
    ])
    def test_each_rule_family(self, text: str, kind: str, params: dict[str, float]) -> None:
        rule = rt.read_rule(text)
        assert rule is not None and rule.kind == kind
        for key, value in params.items():
            assert rule.params[key] == pytest.approx(value), key

    def test_an_unstated_part_is_said_to_be_assumed(self) -> None:
        rule = rt.read_rule("backtest buying SOL when it drops 4%")
        assert rule is not None and rule.params["days"] == rt.DEFAULT_HOLD
        assert any("no holding period stated" in a for a in rule.assumed)

    def test_a_question_with_no_rule_is_not_a_backtest(self) -> None:
        assert rt.lines("is NVDA overbought?") is None
        assert rt.lines("how has BTC done this year?") is None
        assert rt.lines("how did BTC do this year?") is None
        assert rt.ASKED.search("how did a 20/100 EMA crossover do on gold?")
        assert rt.read_rule("how did a 20/100 EMA crossover do on gold?") is not None


class TestNoLookAhead:
    """freqtrade's lookahead analysis, applied to our own rules: the position on day i must be
    the same whether or not the days after i exist."""

    @pytest.mark.parametrize("text", [
        "backtest RSI below 30 above 70 on BTC", "backtest 20/50 sma cross on BTC",
        "backtest the 30-day high on BTC", "backtest BTC above its 100-day average",
        "backtest buying BTC when it falls 2% in a day, hold 4 days",
    ])
    def test_truncation_changes_nothing_before_the_cut(self, text: str) -> None:
        rule = rt.read_rule(text)
        assert rule is not None
        closes = _wave()
        full = rt.positions(rule, closes)
        for cut in (300, 555, 800):
            assert rt.positions(rule, closes[:cut]) == full[:cut]

    def test_wilder_rsi_against_a_hand_count(self) -> None:
        closes = [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08, 45.89,
                  46.03, 45.61, 46.28, 46.28]
        # the StockCharts worked example's closes, rounded to cents: the 14 changes sum to 3.34
        # up and 1.40 down, so the first RSI(14) is 100 - 100 / (1 + 3.34 / 1.40) = 70.46
        # (StockCharts prints 70.53 from its unrounded closes)
        assert rt._rsi(closes, 14)[14] == pytest.approx(100 - 100 / (1 + 3.34 / 1.40))
        # and Wilder's smoothing for the next day: (13 x prior average + today's change) / 14
        nxt = [*closes, 46.00]
        gain, loss = (3.34 * 13 / 14 + 0.0) / 14, (1.40 * 13 / 14 + 0.28) / 14
        assert rt._rsi(nxt, 14)[15] == pytest.approx(100 - 100 / (1 + gain / loss))


class TestRegimes:
    def test_labels_come_from_the_trailing_move(self) -> None:
        closes = [100.0] * 100 + [130.0] + [70.0]
        labels = rt.regimes(closes, days=90, edge=0.2)
        assert labels[0] is None and labels[99] == "chop"
        assert labels[100] == "bull" and labels[101] == "bear"

    def test_the_split_compounds_within_each_regime(self) -> None:
        closes = [100.0, 110.0, 99.0, 99.0]
        net = [0.10, 0.0, 0.0]
        rows = rt.regime_split(closes, net, ["bull", "bear", "bear"])
        bull = next(r for r in rows if r.regime == "bull")
        bear = next(r for r in rows if r.regime == "bear")
        assert bull.rule == pytest.approx(0.10) and bull.hold == pytest.approx(0.10)
        assert bear.rule == pytest.approx(0.0) and bear.hold == pytest.approx(-0.10)

    def test_losing_less_is_not_called_an_edge(self) -> None:
        rows = [rt.RegimeRow("bull", 100, 0.1, 0.5), rt.RegimeRow("bear", 50, -0.2, -0.3)]
        said = rt.regime_verdict(rows)
        assert "only by losing less (-20% against -30%)" in said and "edge" not in said
        assert rt.regime_verdict([rt.RegimeRow("bear", 50, 0.1, -0.3),
                                  rt.RegimeRow("bull", 50, 0.0, 0.4)]).startswith(
            "the edge lives in the bear regime alone (+10% against -30%)")


def test_the_answer_end_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    closes = _wave()
    stamps = [datetime(2022, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(len(closes))]
    monkeypatch.setattr(rt, "_closes", lambda s: (stamps, closes, "Bitget's daily closes"))
    from argus.market import crossasset_feed

    monkeypatch.setattr(crossasset_feed, "fetch_funding", lambda s: [])
    said = rt.lines("backtest the 20/50 sma cross on BTC")
    assert said is not None
    assert said[0].startswith("Bottom line: the rule (long when the 20-day simple moving average "
                              "is above the 50-day, flat when below) returned")
    assert "against" in said[0] and ("beat holding" in said[0] or "trailed holding" in said[0])
    assert any(line.startswith("By regime (each day labelled by the trailing 90-day move") for
               line in said)
    assert any(line.startswith("Out of sample (the last 35% of the days") for line in said)
    # the console routes it
    got = server.handle_ask("backtest the 20/50 sma cross on BTC", [])
    assert got["lines"][0] == said[0]


def test_too_little_history_beyond_the_lookback_is_said(monkeypatch: pytest.MonkeyPatch) -> None:
    closes = _wave(300)
    stamps = [datetime(2025, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(300)]
    monkeypatch.setattr(rt, "_closes", lambda s: (stamps, closes, "Bitget's daily closes"))
    said = rt.lines("backtest holding gold only above its 200-day average")
    assert said is not None and "leaving too few to test on" in said[0]
