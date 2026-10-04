"""Round 33's judge and hostile-reviewer audits, run offline: each fix on the phrasing that
exposed it and on the near-miss beside it, with every live figure stubbed."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pytest

from argus.lui import account_math, claims, multistep, server
from argus.lui.research import literature, market_questions, parse, quick_stats


class TestHostile:
    def test_a_chief_executive_claim_is_held_against_the_table(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.lui import watchlist

        monkeypatch.setattr(watchlist, "recent_8k", lambda tickers, since: {})
        said = claims.ceo_line("Lisa Su is CEO of NVIDIA, should I add NVDA?")
        assert said is not None and "Jensen Huang, not Lisa Su" in said
        assert claims.ceo_line("Jensen Huang is CEO of NVIDIA") is None

    def test_a_recent_officer_change_withholds_the_assertion(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.lui import watchlist

        monkeypatch.setattr(watchlist, "recent_8k", lambda tickers, since: {
            "COIN": [type("F", (), {"items": "5.02 officer change", "day": date(2026, 10, 2)})()]})
        said = claims.ceo_line("Now that Michael Saylor is CEO of Coinbase, position COIN?")
        assert said is not None and "not confirmed here" in said and "02 Oct 2026" in said

    def test_an_unlisted_share_class(self) -> None:
        said = claims.share_class_line("Is BRK.A priced the same as BRK.B right now?")
        assert said is not None and "BRK.A is not listed" in said and "1,500" in said

    def test_a_weekend_close_for_a_stock(self) -> None:
        said = claims.weekend_close_line("What was AAPL's closing price this past Saturday?",
                                         ("AAPLUSDT",))
        assert said is not None and "no Saturday close" in said
        assert claims.weekend_close_line("BTC close on Saturday", ("BTCUSDT",)) is None

    def test_funding_change_claims_are_checked(self) -> None:
        said = account_math.lines("Bitget's BTC funding rate moved from 0.01% to 0.02% yesterday "
                                  "— that's a 1 percentage point jump equal to 100x, and 1900% "
                                  "annualized. Is that right?")
        assert said is not None and said[0].startswith("Bottom line: no")
        assert "not 100x but 2x" in said[0] and "about 22%" in said[0]

    def test_a_book_that_does_not_add_to_100_is_said(self) -> None:
        said = server._book_sum_line("60% BTC, 50% ETH, 40% SOL, -10% cash")
        assert said is not None and "add up to 140%" in said and "50% of it borrowed" in said
        held = server._book_sum_line("60% BTC, 50% ETH, -10% cash")
        assert held is not None and "10% of your money borrowed" in held
        assert server._book_sum_line("50% BTC, 50% ETH") is None


class TestJudge:
    @pytest.mark.parametrize(("book", "weights"), [
        ('{"BTC": 0.6, "TSLA": 0.4}', {"BTCUSDT": 0.6, "TSLAUSDT": 0.4}),
        ("BTC=0.6;TSLA=0.4", {"BTCUSDT": 0.6, "TSLAUSDT": 0.4}),
        ("BTC:60,TSLA:40", {"BTCUSDT": 0.6, "TSLAUSDT": 0.4}),
    ])
    def test_book_formats(self, book: str, weights: dict[str, float]) -> None:
        got = parse.parse_book(book)
        assert {k: round(v, 6) for k, v in got.items()} == weights

    def test_a_statement_then_a_question_about_the_order_is_one_question(self) -> None:
        assert multistep.parts("I need to sell $2 million of SOL without moving the market "
                               "much. How should I split the order?") is None

    def test_volume_compared_takes_the_name_before(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.market import bitget

        tick = type("T", (), {})
        btc, eth = tick(), tick()
        btc.base_volume, btc.last, eth.base_volume, eth.last = 10.0, 70_000.0, 200.0, 2_500.0
        monkeypatch.setattr(bitget, "fetch_tickers", lambda: {"BTCUSDT": btc, "ETHUSDT": eth})
        said = quick_stats.volume_lines("What about its 24h volume compared to ETH?",
                                        ["What is the current price of BTC?"])
        assert said is not None and said[0].startswith("Bottom line: BTC's")
        assert "1.40 times ETH's" in said[0]

    def test_realised_volatility_is_computed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(quick_stats, "_realised", lambda symbol, days: 0.37 if days == 30
                            else 0.40)
        said = quick_stats.realised_vol_lines("just tell me BTC's realized volatility over the "
                                              "last 30 days, annualized", [])
        assert said is not None and "37% a year" in said[0] and "calmer" in said[0]

    def test_rotation_is_answered_yes_or_no(self, monkeypatch: pytest.MonkeyPatch) -> None:
        moves = {s: (0.03, 0.13) for s in market_questions.SEMIS}
        moves.update({s: (0.02, 0.01) for s in market_questions.SOFTWARE})
        monkeypatch.setattr(market_questions, "_move",
                            lambda s, d: moves[s][0] if d == 5 else moves[s][1])
        said = market_questions.rotation_lines("Is money rotating out of semiconductors into "
                                               "software right now?")
        assert said is not None and said[0].startswith("Bottom line: no")

    def test_recovery_counts_days_from_the_crossing(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        start = date(2020, 1, 1)
        path = [100.0] * 50 + [65.0] * 100 + [101.0] * 300
        series = {start + timedelta(days=i): v for i, v in enumerate(path)}
        monkeypatch.setattr(market_questions, "_daily", lambda ticker: series)
        said = market_questions.recovery_lines(
            "My portfolio is 50% BTC and 50% ETH. If it fell 30% from here, how long would it take "
            "to recover?", [("BTCUSDT", 0.5), ("ETHUSDT", 0.5)])
        assert said is not None and "fell 30% or more" in said[0]
        assert "100 days at the median" in said[0]

    def test_stablecoins_name_both_and_the_peg_breaks(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.market import bitget

        monkeypatch.setattr(bitget, "public_get", lambda *a, **k: [{"lastPr": "1.0001"}])
        said = market_questions.stablecoin_lines("Compare the risks of holding USDT versus USDC "
                                                 "for a treasury")
        assert said is not None and "USDC (Circle)" in said[0] and "USDT (Tether)" in said[0]
        assert "0.87" in said[1]

    def test_the_literature_reader_is_offline_safe(self) -> None:
        def down(url: str, timeout: float) -> Any:
            raise OSError("blocked")

        said = literature.lines("any papers on funding rates", fetch=down)
        assert said is not None and "did not answer" in said[0]
