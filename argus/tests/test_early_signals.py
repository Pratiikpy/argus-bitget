"""The early-signal watchlist (lui/research/early_signals.py): the score recomputed by hand on
made-up tickers and candles, and the ask it answers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest

from argus.lui.research import early_signals
from argus.market.history import Candle


@dataclass
class _Ticker:
    last: Decimal
    base_volume: Decimal
    funding_rate: Decimal


def _days(close: list[float], volume: float) -> list[Candle]:
    start = datetime(2026, 9, 1, tzinfo=UTC)
    return [Candle(ts=start + timedelta(days=i), open=Decimal(str(c)), high=Decimal(str(c)),
                   low=Decimal(str(c)), close=Decimal(str(c)), volume=Decimal(str(volume)))
            for i, c in enumerate(close)]


def test_a_component_score_recomputes_by_hand() -> None:
    s = early_signals.Scored(symbol="XUSDT", turnover=3e7, volume_ratio=2.0, move_5d=-0.05,
                             funding=0.00025, trending=True)
    # volume (2-1)/2 of 35 = 17.5; momentum 5% of 10% of 25 = 12.5; funding half of 20 = 10;
    # attention 20
    assert s.parts == pytest.approx({"volume": 17.5, "momentum": 12.5, "funding": 10.0,
                                     "attention": 20.0})
    assert s.score == pytest.approx(60.0)
    capped = early_signals.Scored(symbol="Y", turnover=1, volume_ratio=50.0, move_5d=0.9,
                                  funding=-0.01, trending=False)
    assert capped.score == pytest.approx(80.0)  # each part is capped at its weight


def test_scan_ranks_and_skips_thin_books(monkeypatch: pytest.MonkeyPatch) -> None:
    from argus.lui.research import desk_answers
    from argus.market import history

    tickers = {
        # $40m today against $10m a day before: 4x volume, flat price
        "AAAUSDT": _Ticker(Decimal(10), Decimal(4_000_000), Decimal("0.0001")),
        # $30m today against $30m before, +20% over five days
        "BBBUSDT": _Ticker(Decimal(12), Decimal(2_500_000), Decimal("0.0001")),
        "THINUSDT": _Ticker(Decimal(1), Decimal(1_000), Decimal("0.01")),  # under $20m: skipped
        # a stablecoin: skipped however much it trades
        "USDCUSDT": _Ticker(Decimal(1), Decimal(1_000_000_000), Decimal(0)),
    }
    candles = {"AAAUSDT": _days([10.0] * 26, 1_000_000),
               "BBBUSDT": _days([10.0] * 20 + [10.0, 10.4, 10.8, 11.2, 11.6, 12.0], 3_000_000)}
    monkeypatch.setattr(desk_answers, "_tickers", lambda: tickers)

    def fetch_range(symbol: str, **_kw: Any) -> list[Candle]:
        return candles[symbol]

    monkeypatch.setattr(history, "fetch_range", fetch_range)
    scored = early_signals.scan(trending_symbols=frozenset({"BBB"}))
    assert [s.symbol for s in scored] == ["BBBUSDT", "AAAUSDT"]
    aaa = next(s for s in scored if s.symbol == "AAAUSDT")
    assert aaa.volume_ratio == pytest.approx(4.0)
    assert aaa.parts["volume"] == pytest.approx(35.0)
    bbb = scored[0]
    assert bbb.move_5d == pytest.approx(0.2)
    assert bbb.trending


def test_the_ask() -> None:
    for asked in ("show me today's early signals", "any unusual volume on Bitget?",
                  "what should I look at today?", "scan the market"):
        assert early_signals.ASKED.search(asked), asked
    assert not early_signals.ASKED.search("what is BTC trading at")
