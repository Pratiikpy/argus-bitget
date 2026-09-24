"""Bitget's data service entries the console now reads: each parsed, each worded, none trusted.

Every call is replaced by a fixture shaped like the live response recorded on 2026-09-24, so the
parsing and the sentences are pinned without the network.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest

from argus.market import bitget_positioning as bp


def _serve(monkeypatch: pytest.MonkeyPatch, table: dict[str, list[dict[str, Any]]]) -> None:
    monkeypatch.setattr(bp, "_safe", lambda entry, **_: table.get(entry, []))


def test_stock_mood_reads_the_score_and_its_history(monkeypatch: pytest.MonkeyPatch) -> None:
    _serve(monkeypatch, {"sentiment_market_fear_greed": [{
        "score": 34.63, "rating": "fear", "previous_1_week": 28.29,
        "previous_1_month": 54.97}]})
    assert bp.stock_mood() == ("US stock-market fear & greed: 35 (fear); a week ago 28, a month "
                               "ago 55 (Bitget's data service, bitget-mcp-server).")


def test_a_missing_or_malformed_reading_is_no_line(monkeypatch: pytest.MonkeyPatch) -> None:
    _serve(monkeypatch, {"sentiment_market_fear_greed": [{"score": "n/a"}]})
    assert bp.stock_mood() is None
    _serve(monkeypatch, {})
    assert bp.stock_mood() is None and bp.crypto_mood() is None and bp.us_stock_brief() is None


def test_positioning_compares_the_crowd_with_the_largest_traders(
        monkeypatch: pytest.MonkeyPatch) -> None:
    today = datetime.now(UTC).date()
    _serve(monkeypatch, {
        "crypto_futures_long_short_ratio": [{"long_account": 0.74}],
        "crypto_futures_long_short_top_position_ratio": [{"long_account": 0.61}],
        "crypto_indicators_hyperliquid_whale_sentiment": [{
            "total_long_position_val": 1.154e9, "total_short_position_val": 7.10e8}],
        "crypto_futures_liquidations": [
            {"date": f"{today - timedelta(days=1)}T00:00:00Z", "long_liquidations": 16.4e6,
             "short_liquidations": 7.3e6},
            {"date": f"{today}T00:00:00Z", "long_liquidations": 1.0, "short_liquidations": 99.0e6},
        ],
    })
    lines = bp.positioning("ETH").lines()
    assert "74% of all accounts are long, against 61% of the largest traders' positions" in lines[0]
    assert "the crowd is longer than the largest traders" in lines[0]
    assert "$1,154m long and $710m short in ETH" in lines[1]
    # today's row is still filling; the last full day is yesterday's
    assert f"liquidations on {today - timedelta(days=1)}" in lines[2]
    assert "the longs took the larger flush" in lines[2]


def test_an_ex_dividend_date_is_shown_only_when_it_is_near(
        monkeypatch: pytest.MonkeyPatch) -> None:
    today = date(2026, 9, 24)
    _serve(monkeypatch, {"equity_fundamental_dividends": [
        {"ex_dividend_date": "2026-10-10", "amount": 0.25, "payment_date": "2026-10-30"},
        {"ex_dividend_date": "2026-07-10", "amount": 0.25, "payment_date": "2026-07-30"}]})
    line = bp.next_ex_dividend("NVDA", today=today)
    assert line is not None and line.startswith("NVDA goes ex-dividend on 10 Oct 2026 ($0.25")
    _serve(monkeypatch, {"equity_fundamental_dividends": [
        {"ex_dividend_date": "2026-09-09", "amount": 0.25, "payment_date": "2026-09-30"}]})
    assert bp.next_ex_dividend("NVDA", today=today) is None


def test_a_bitcoin_treasury_is_valued_against_the_company(
        monkeypatch: pytest.MonkeyPatch) -> None:
    _serve(monkeypatch, {"crypto_institutional_company_flow": [
        {"company_name": "Strategy", "ticker": "MSTR", "holding_balance": 846842.0,
         "ts": "1790092800000"},
        {"company_name": "Strategy", "ticker": "MSTR", "holding_balance": 840000.0,
         "ts": "1790006400000"},
        {"company_name": "Coinbase", "ticker": "COIN", "holding_balance": 16949.0,
         "ts": "1790092800000"}]})
    line = bp.bitcoin_treasury("MSTR", 61.0e9, 83_518.0)
    assert line is not None
    assert line.startswith("Strategy holds 846,842 BTC, worth $70.7bn at $83,518")
    assert "0.86x its bitcoin" in line
    assert bp.bitcoin_treasury("TSLA", 1e12, 83_518.0) is None


def test_etf_flows_are_not_read() -> None:
    import inspect

    assert '"crypto_etf_flows"' not in inspect.getsource(bp).split('"""', 2)[2]
