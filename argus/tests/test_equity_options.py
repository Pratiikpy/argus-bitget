"""Equity option questions answered from a synthetic Cboe chain, offline."""

from __future__ import annotations

import math
from datetime import UTC, date, datetime
from typing import Any

import pytest

from argus.lui import watchlist
from argus.lui.research import equity_options, rule_test
from argus.market import options

TODAY = date(2026, 10, 5)
SPOT = 270.0
PRIOR = ["I'm long 100 shares of AAPL bought at 190"]
EXPIRIES = {date(2026, 10, 16): 0.24, date(2026, 10, 23): 0.25, date(2026, 11, 6): 0.26,
            date(2026, 11, 20): 0.28}


def _row(expiry: date, right: str, strike: float, iv: float) -> dict[str, Any]:
    days = (expiry - TODAY).days
    intrinsic = max(0.0, strike - SPOT) if right == "P" else max(0.0, SPOT - strike)
    time_value = SPOT * iv * math.sqrt(days / 365) * 0.4 * math.exp(-abs(strike - SPOT) / 25)
    mid = intrinsic + time_value
    return {"option": f"AAPL{expiry:%y%m%d}{right}{int(strike * 1000):08d}",
            "bid": round(mid - 0.05, 2), "ask": round(mid + 0.05, 2), "iv": iv,
            "delta": -0.3 if right == "P" else 0.3,
            "open_interest": 100, "volume": 10}


def _chain() -> dict[str, Any]:
    rows = [_row(e, r, float(k), iv + (0.02 if r == "P" else 0.0))
            for e, iv in EXPIRIES.items() for k in range(240, 305, 5) for r in "CP"]
    return {"timestamp": "2026-10-05 17:30:00",
            "data": {"symbol": "AAPL", "current_price": SPOT, "iv30": 26.5, "options": rows}}


def _closes() -> tuple[list[datetime], list[float], str]:
    closes = [250.0 * (1.01 if i % 2 else 0.995) ** (i % 7 + 1) for i in range(60)]
    stamps = [datetime(2026, 7, 1, tzinfo=UTC) for _ in closes]
    return stamps, closes, "test closes"


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(options, "fetch_chain", lambda symbol, **_: _chain())
    monkeypatch.setattr(rule_test, "daily_closes", lambda symbol: _closes())
    monkeypatch.setattr(watchlist, "earnings_date",
                        lambda ticker, today: watchlist.Report(
                            ticker, date(2026, 10, 29), "after the close", "test calendar"))


def _joined(out: list[str] | None) -> str:
    assert out is not None
    return "\n".join(out)


def test_protective_put_uses_prior_position() -> None:
    out = equity_options.lines(
        "What would a 30-day protective put 5% below the current price cost me, as a percent of "
        "the position?", PRIOR, today=TODAY)
    text = _joined(out)
    assert out is not None and out[0].startswith("Bottom line: ")
    # 5% below 270 is 256.5; the nearest listed strike is 255 (256.5 is 1.5 from it, 260 is 3.5)
    assert "255 put (5.6% below the price) expiring 06 Nov 2026 (32 days)" in text
    put = next(c for c in map(options.parse_contract, _chain()["data"]["options"])
               if c and c.right == "P" and c.strike == 255 and c.expiry == date(2026, 11, 6))
    assert f"${put.mid:.2f} a share" in out[0]
    assert f"{put.mid / SPOT * 100:.2f}% of the position" in out[0]
    assert "100 shares" in out[0]
    assert "against your entry at 190.00" in text.lower() or "Against your entry at 190.00" in text
    assert f"bid ${put.bid:.2f}, ask ${put.ask:.2f}" in text.lower()
    assert out[-1].startswith("Data: Cboe delayed options chain, 2026-10-05 13:30 New York")
    assert out[-1].endswith("Not advice.")


def test_covered_call_annualised_yield() -> None:
    out = equity_options.lines(
        "Now what if I instead sold a covered call 10% above the price — what premium would I "
        "collect?", PRIOR, today=TODAY)
    assert out is not None
    call = next(c for c in map(options.parse_contract, _chain()["data"]["options"])
                if c and c.right == "C" and c.strike == 295 and c.expiry == date(2026, 11, 6))
    annual = call.mid / SPOT * 365 / 32 * 100
    assert "295 call (9.3% above the price) expiring 06 Nov 2026" in out[0]
    assert f"{annual:.1f}% annualised" in out[0]
    assert f"${call.mid:.2f} a share" in out[0]
    assert "The cap: above 295" in _joined(out)
    assert "given up" in _joined(out)


def test_iv_vs_realised_with_report_date() -> None:
    out = equity_options.lines(
        "What is AAPL's implied volatility for the nearest monthly expiry versus its 30-day "
        "realised volatility, and when does it next report?", (), today=TODAY)
    text = _joined(out)
    assert out is not None
    # nearest third-Friday expiry at least 7 days out is 16 Oct (11 days)
    assert "16 Oct 2026 (11 days)" in out[0]
    assert "is 25.0% a year" in out[0]  # mean of the 16 Oct call (24%) and put (26%) IVs
    assert "realised over the last 30 trading days" in out[0]
    assert "29 Oct 2026, after the close" in text
    assert "falls before it" in text
    assert "Cboe's own 30-day implied volatility reads 26.5%" in text


def test_symbol_taken_from_prior_turn() -> None:
    out = equity_options.lines("what would a 10% out-of-the-money put cost for next month?",
                               ["tell me about AAPL"], today=TODAY)
    assert out is not None and "AAPL" in out[0]
    assert equity_options.lines("what would a 10% out-of-the-money put cost?", [],
                                today=TODAY) is None


def test_none_for_other_questions() -> None:
    for text in ("What is AAPL's price?",
                 "BTC 30-day implied volatility and 25-delta skew",
                 "what is the put/call ratio on AAPL",
                 "How big a move are options pricing for AAPL's next earnings?",
                 "sell a covered call on bitcoin",
                 "should I buy more AAPL"):
        assert equity_options.lines(text, PRIOR, today=TODAY) is None, text


def test_missing_chain_is_said(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(symbol: str, **_: Any) -> dict[str, Any]:
        raise RuntimeError("down")

    monkeypatch.setattr(options, "fetch_chain", boom)
    out = equity_options.lines("AAPL protective put 5% below", [], today=TODAY)
    assert out is not None and "did not answer" in out[0]
    assert len(out) == 1


def test_realised_unavailable_still_gives_implied(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(symbol: str) -> Any:
        raise RuntimeError("down")

    monkeypatch.setattr(rule_test, "daily_closes", boom)
    out = equity_options.lines("AAPL implied volatility nearest monthly vs realised", [],
                               today=TODAY)
    assert out is not None and "could not be read just now" in out[0]
