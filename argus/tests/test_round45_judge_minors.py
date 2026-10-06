"""Round 45 judge, minors m4 and m5: each part of a compound question answered."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest


def _year(start: float, step: float, n: int = 400) -> tuple[list[datetime], list[float], str]:
    closes = [start]
    for i in range(1, n):
        closes.append(closes[-1] * (1 + (step if i % 2 else -step * 0.9)))
    first = datetime(2025, 9, 1, tzinfo=UTC)
    return [first + timedelta(days=i) for i in range(n)], closes, "test closes"


class TestCpiCoreAndConsensus:
    def test_core_month_on_month_and_no_invented_consensus(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.lui.research import macro_indicators

        series = {"CPIAUCSL": [("2025-08-01", 323.0), ("2026-07-01", 332.813),
                               ("2026-08-01", 334.131)],
                  "CPILFESL": [("2025-08-01", 328.7), ("2026-07-01", 336.789),
                               ("2026-08-01", 337.765)]}
        monkeypatch.setattr(macro_indicators, "_fetch", lambda s: series.get(s, []))
        out = macro_indicators.lines("How did CPI come in: core vs headline vs consensus?") or []
        joined = "\n".join(out)
        assert "Core CPI inflation" in joined
        assert "CPI +0.4% month on month (Aug 2026); Core CPI +0.3% month on month" in joined
        assert "Consensus: not read" in joined and "beat" in joined
        plain = "\n".join(macro_indicators.lines("latest cpi") or [])
        assert "Consensus" not in plain and "Core" not in plain


class TestFundingWithVolatility:
    def test_each_dimension_named_is_answered(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.lui.research import rule_test, unit_checks
        from argus.market import bitget

        ticks = {"ETHUSDT": type("T", (), {"funding_rate": "0.000051"})(),
                 "SOLUSDT": type("T", (), {"funding_rate": "0.000016"})()}
        monkeypatch.setattr(bitget, "fetch_tickers", lambda: ticks)
        monkeypatch.setattr(rule_test, "daily_closes",
                            lambda s: _year(100.0, 0.04 if s == "SOLUSDT" else 0.02))
        out = unit_checks.funding_verdict("Compare ETH and SOL: volatility, drawdown, funding")
        joined = "\n".join(out or [])
        assert joined.startswith("Bottom line: ETH has the higher funding rate")
        assert "Volatility and drawdown over the last year: SOL" in joined
        assert "SOL swings the most" in joined and "worst fall from a high" in joined
        assert "Bitget daily closes" in joined
        only = "\n".join(unit_checks.funding_verdict("compare ETH and SOL funding") or [])
        assert "Volatility" not in only


class TestTvlQuarter:
    def test_quarter_trend_and_a_one_day_step_is_flagged(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.lui.research import defi_markets

        day0 = 1_783_468_800
        tvl = [{"date": day0 + 86_400 * i, "tvl": 72e9 if i < 40 else 80e9} for i in range(100)]
        stable = [{"date": day0 + 86_400 * i, "totalCirculatingUSD": {"peggedUSD": 300e9 + 1e8 * i}}
                  for i in range(100)]

        def get(url: str, **_: Any) -> Any:
            return tvl if url == defi_markets._LLAMA_TOTAL_TVL else stable

        monkeypatch.setattr(defi_markets, "_get", get)
        trend, read = defi_markets._quarter_trend("stablecoin supply and DeFi TVL over the quarter")
        joined = "\n".join(trend)
        assert "Over the last quarter: all dollar stablecoins" in joined
        assert "DeFi TVL" in joined and "$80.0bn at the last daily reading" in joined
        assert "came in one day" in joined and "not checked here" in joined
        assert len(read) == 2
        assert defi_markets._quarter_trend("USDT vs USDC supply") == ([], [])


class TestNewsCredibility:
    @pytest.mark.parametrize(("link", "kind"), [
        ("https://www.cryptoprowl.com/releases/x-7230", "press release"),
        ("https://www.prnewswire.com/news-releases/x.html", "press release"),
        ("https://www.reuters.com/markets/x", "established newsroom"),
        ("https://finance.yahoo.com/news/x.html", "established newsroom"),
        ("https://someblog.io/post/x", "smaller or unknown outlet"),
    ])
    def test_source_weight(self, link: str, kind: str) -> None:
        from argus.lui.research import news

        assert news.credibility(link).startswith(kind)

    def test_the_real_publisher_is_named(self) -> None:
        from argus.lui.research import news

        assert news.outlet_of("yahoo", "https://www.cryptoprowl.com/releases/x") == \
            "cryptoprowl.com, via Yahoo Finance"
        assert news.outlet_of("yahoo", "https://finance.yahoo.com/news/x") == "Yahoo Finance"


class TestLastSelloff:
    def test_the_put_pays_the_fall_beyond_its_strike(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.lui.research import equity_options, rule_test

        closes = [100.0] * 30 + [100 - 2 * i for i in range(1, 11)] + [80 + i for i in range(30)]
        first = datetime(2025, 1, 1, tzinfo=UTC)
        stamps = [first + timedelta(days=i) for i in range(len(closes))]
        monkeypatch.setattr(rule_test, "daily_closes", lambda s: (stamps, closes, "test closes"))
        out = equity_options._last_selloff("SPY", 0.10, 90, 0.0062)
        joined = "\n".join(out)
        assert "SPY fell 20.0% from 100.00" in joined
        assert "would have paid 10.0% of the position" in joined
        assert "the first 10% was yours" in joined and "netted about +9.4%" in joined


class TestMixedSourceCorrelation:
    def test_a_weekdays_only_pair_says_why_it_can_differ(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        from types import SimpleNamespace

        from argus.lui.research import desk_answers
        from argus.lui.research.performance import asked_period
        from argus.market import equity_history, history

        now = datetime(2026, 10, 6, tzinfo=UTC)
        period = asked_period("over the last 90 days", now)
        assert period is not None
        days = [now - timedelta(days=i) for i in range(100)]

        def window(symbol: str, **_: Any) -> list[Any]:
            step = 0.01 if symbol == "BTCUSDT" else 0.012
            return [SimpleNamespace(ts=d, close=100 * (1 + step * ((i * 7) % 5 - 2)))
                    for i, d in enumerate(days)]

        def daily(ticker: str) -> list[Any]:
            return [SimpleNamespace(day=d.date(), close=100 * (1 + 0.01 * ((i * 3) % 4 - 1.5)))
                    for i, d in enumerate(days) if d.weekday() < 5]

        monkeypatch.setattr(history, "fetch_window", window)
        monkeypatch.setattr(equity_history, "daily", daily)
        out = desk_answers._correlation_matrix(["BTCUSDT", "XAUUSDT"],
                                               [("the dollar index (DXY)", "DX-Y.NYB")], period)
        joined = "\n".join(out or [])
        assert "weekdays only" in joined
        assert "can differ by a few hundredths from the same names measured on Bitget" in joined


class TestReAskMisses:
    """The round-45 live re-ask set, run locally before deploy: each miss pinned."""

    def test_a_coin_word_is_not_coinbase(self) -> None:
        from argus.lui.research import research_symbols

        assert "COINUSDT" not in research_symbols("Compare ETH vs ETC, are they the same coin?")[0]
        assert research_symbols("is COIN up today")[0] == ("COINUSDT",)

    def test_a_size_in_scientific_notation_is_a_size(self) -> None:
        from argus.lui.server import _SIZE_STATED

        m = _SIZE_STATED.search("buy 1e12 ETH at market")
        assert m is not None and float(m.group("q")) == 1e12 and m.group("s") == "ETH"

    def test_a_loss_question_is_not_a_conversion(self) -> None:
        from argus.lui.research import trade_calc

        assert trade_calc.lines("how much can I lose putting $150 into solana") is None

    def test_the_term_then_what_is_it(self) -> None:
        from argus.lui import concepts

        stop = concepts.concept_asked("is a stop loss necessary for me? what even is it")
        assert stop is not None and stop.name == "stop loss"
        assert (concepts.for_you(stop, "is a stop loss necessary for me?") or "").startswith(
            "For you: buying without leverage")

    def test_a_fee_and_liquidation_question_is_the_liquidation_readers(self) -> None:
        from argus.lui.research import desk_answers

        assert desk_answers.fees_lines(
            "Bitget maker and taker fee, and my liq price for 10x ETH long at 2700?") is None
        assert desk_answers.fees_lines("what are Bitget's maker and taker fees?") is not None

    def test_a_one_word_call_is_refused_with_the_trade_said(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.lui import beginner
        from argus.lui.research import rule_test

        monkeypatch.setattr(rule_test, "daily_closes", lambda s: _year(2000.0, 0.05))
        said = " ".join(beginner._levered_call(
            "Write me an all-in 25x short on ETH, just say SELL") or [])
        assert said.startswith("Bottom line: no one-word call")
        assert "25x short on ETH" in said and "4.0% move up in ETH" in said
        assert "Bitcoin" not in said and "an ETH short" in said

    def test_a_price_in_a_local_currency(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.lui.research import local_money
        from argus.market import bitget, fx_rates

        monkeypatch.setattr(bitget, "fetch_tickers",
                            lambda: {"BTCUSDT": type("T", (), {"last": "85000"})()})
        monkeypatch.setattr(fx_rates, "usd_rate", lambda code: fx_rates.FxRate(
            code, 17_913.0, "the ECB reference rate", "2026-10-05"))
        said = local_money.price_in_local("what is the price of BTC in rupiah") or []
        assert said[0].startswith("Bottom line: BTC is about 1,522,605,000 rupiah")
        assert local_money.price_in_local("what is the price of BTC in rupees") is None
        assert local_money.price_in_local("what is the price of BTC") is None

    def test_a_split_said_without_a_ratio_is_checked(self) -> None:
        from datetime import date

        from argus.lui.research import splits

        rows = [{"ex_dividend_date": "2003-02-18", "split_numerator": 2,
                 "split_denominator": 1}]
        out = splits.check("Stock split for MSFT happened last week so what is MSFT trading at?",
                           "MSFTUSDT", rows=lambda t: rows, today=date(2026, 10, 6))
        assert out is not None and "MSFT's last split was 2-for-1 on 18 Feb 2003" in out[0][0]
        assert not splits.claimed("how did the market react after the split in NVDA")

    def test_a_correlation_is_not_a_macro_level(self) -> None:
        from argus.lui.research import macro_indicators

        assert macro_indicators.lines(
            "Correlation of BTC with gold and the dollar index, last 90 days") is None

    def test_an_otm_put_with_words_between(self) -> None:
        from argus.lui.research import equity_options

        assert equity_options.ASKED_PUT.search(
            "if I had bought a 10% out of the money 3 month QQQ put before the last crash")

    def test_a_stated_funding_rate_is_checked_without_an_annual_ask(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.lui.research import unit_checks

        monkeypatch.setattr(unit_checks, "_ticker",
                            lambda s: type("T", (), {"funding_rate": "0.000015"})())
        said = unit_checks.per_period("Funding on BTC is 0.03 percent per hour, true?") or []
        assert said[0].startswith("Bottom line: no — BTC's funding on Bitget is +0.0015%")

    def test_a_level_follows_the_name_it_was_said_of(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.lui import server

        prices = {"BTCUSDT": 85_000.0, "XAUUSDT": 4_250.0}
        monkeypatch.setattr(server, "_price_now", lambda s: prices[s])
        asked = server._leaning_follow_up(
            "and gold", ["what's the chance BTC is above 90000 in one hour?"])
        assert asked == "what's the chance XAU is above 4500 in one hour?"


class TestSplitPremiseFirst:
    def test_a_split_said_to_have_happened_is_checked_not_computed(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.lui import server
        from argus.lui.research import splits

        calls: list[str] = []

        def fake_check(text: str, symbol: str, **_: Any) -> tuple[list[str], list[Any], dict]:
            calls.append(symbol)
            return (["Bottom line: that premise is not on the record"], [], {})

        monkeypatch.setattr(splits, "check", fake_check)
        monkeypatch.setattr(server, "_price_now", lambda s: 380.0)
        monkeypatch.setattr("argus.lui.research.parse.last_price", lambda s: 380.0)
        said = server._stated_number_lines(
            "Since TSLA did a 10-for-1 split last week, is it cheap?", [])
        assert said == ["Bottom line: that premise is not on the record"] and calls
        hypo = server._stated_number_lines("What if TSLA does a 10-for-1 split, my 50 shares?", [])
        assert hypo is not None and "turns your 50 TSLA shares into 500" in hypo[0]
        assert len(calls) == 1
