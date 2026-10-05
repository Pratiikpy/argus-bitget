"""Round 43 audits (judge, hostile, newcomer; Activity/audits/round43_*.md), fixed offline.

The builders' own modules carry their own test files (shareholder_metrics, token_supply,
futures_curve, pair_trade, market_breadth, crypto_structure, round43_newcomer). These pin the
readers and routes written alongside them: units and premises, views, setups, the diversifier
replay, and the routing guards that kept a question with the reader it belongs to."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest


def _ticker(last: float, bid: float | None = None, ask: float | None = None,
            change: float = 0.0, funding: float = 0.0) -> Any:
    return SimpleNamespace(last=last, bid=bid or last, ask=ask or last, change_24h=change,
                           funding_rate=funding)


class TestUnitsAsStated:
    def test_a_rate_per_hour_is_annualised_per_hour(self) -> None:
        from argus.lui.research.unit_checks import per_period

        said = per_period("Funding is 0.01% per hour on BTC. What is that per year?")
        assert said is not None and "87.60% a year simple" in said[0]
        assert "eight-hour" in said[1]

    def test_cents_are_dollars_over_a_hundred(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.lui.research import unit_checks

        monkeypatch.setattr(unit_checks, "_ticker", lambda s: _ticker(85_779.9))
        said = unit_checks.minor_units("BTC is 8576510 cents. Is that above the current price?")
        assert said is not None and "$85,765.10" in said[0] and "within 1%" in said[1]

    def test_pence_are_pounds_over_a_hundred(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.lui.research import parse, unit_checks

        monkeypatch.setattr(parse, "in_us_dollars",
                            lambda t: ("$33.06", ["£25.00 = about $33 at GBPUSD 1.3223"]))
        monkeypatch.setattr(unit_checks, "_ticker", lambda s: None)
        said = unit_checks.minor_units("Shell trades at 2,500p so that is 2,500 pounds a share, "
                                       "right? What is a 10 share position worth?")
        assert said is not None and "£25.00" in said[0] and "£250.00" in said[0]

    def test_p2p_is_not_two_pence(self) -> None:
        from argus.lui.research.unit_checks import minor_units

        assert minor_units("is P2P trading safe") is None

    def test_stated_fee_and_gas_are_used(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.lui.research import unit_checks

        monkeypatch.setattr(unit_checks, "_ticker", lambda s: _ticker(2_700.0))
        fee = unit_checks.stated_fee("A round trip in BTC with a 0.0001% taker fee: what move "
                                     "do I need to break even?")
        assert fee is not None and "0.000200%" in fee[0]
        gas = unit_checks.stated_gas("Gas is 30 gwei and a transfer uses 21000 gas. What does it "
                                     "cost in ETH?")
        assert gas is not None and "0.000630 ETH" in gas[0]

    def test_a_stop_or_target_on_the_wrong_side(self) -> None:
        from argus.lui.research.unit_checks import wrong_side

        short = wrong_side("Short BTC at 85,000 with take-profit 90,000. What do I make if it "
                           "hits?")
        assert short is not None and "a loss of 5,000.00" in short[0]
        long_ = wrong_side("Long ETH, entry 2,700, stop 2,900, size 5 ETH. What is my max loss?")
        assert long_ is not None and "take-profit, not a stop" in long_[0]
        assert wrong_side("I am long 1 BTC with a stop 5% below my entry.") is None

    def test_a_new_stop_across_turns_is_flagged(self) -> None:
        from argus.lui.research.unit_checks import stop_percent

        said = stop_percent("Actually my stop is 5% above entry on that long. What do I lose if "
                            "it is hit on 85,000 entry?",
                            ["I am long 1 BTC with a stop 5% below my entry."])
        assert said is not None and "89,250.00" in said[0] and "not a stop" in said[0]
        assert "changes what you said before" in said[1]

    def test_fractions_and_basis_points(self) -> None:
        from argus.lui.research.unit_checks import bps_versus_percent, fraction_or_percent

        frac = fraction_or_percent("Is 0.4 on the 24h change 40%? Explain.")
        assert frac is not None and "means 0.4%" in frac[0]
        bps = bps_versus_percent("How much does a fee of 0.1 bps cost on a 10,000 USDT trade "
                                 "versus 0.1%?")
        assert bps is not None and "$0.10" in bps[0] and "$10.00" in bps[0]


class TestPremises:
    def test_there_was_no_2025_halving(self) -> None:
        from argus.lui.research.unit_checks import false_premise

        said = false_premise("After the 2025 Bitcoin halving, how did BTC price react?")
        assert said is not None and "no Bitcoin halving in 2025" in said[0]

    def test_an_etf_ban_needs_a_source(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.lui.research import unit_checks
        from argus.market import etf_flows

        monkeypatch.setattr(etf_flows, "load", lambda: {"funds": {"BTC": {"date": "2026-10-02"}}})
        said = unit_checks.false_premise("Since spot Bitcoin ETFs were banned in the US in 2025, "
                                         "what has BTC done?")
        assert said is not None and "10 Jan 2024" in said[0] and "2026-10-02" in said[0]

    def test_percentage_points_are_not_basis_points(self, monkeypatch: pytest.MonkeyPatch
                                                    ) -> None:
        from argus.lui.research import macro, unit_checks

        monkeypatch.setattr(macro, "_fred", lambda s, d=45: [("2026-10-02", 5.28)])
        said = unit_checks.false_premise("The US 10-year yield rose 50 percentage points "
                                         "overnight to 4.5. Is that a 50bps move?")
        assert said is not None and "5,000 basis points" in said[0] and "5.28%" in said[1]

    def test_a_coin_has_no_pe(self) -> None:
        from argus.lui.research.unit_checks import crypto_equity_note

        assert "Ether has no earnings" in crypto_equity_note(
            "Give me BTC funding and also the P/E of ETH.", "")
        assert crypto_equity_note("What is NVDA P/E?", "") == ""


class TestRiskAndViews:
    def test_a_remembered_rule_sizes_the_position(self, monkeypatch: pytest.MonkeyPatch
                                                  ) -> None:
        from argus.lui.research import unit_checks

        monkeypatch.setattr(unit_checks, "_ticker", lambda s: _ticker(125.0))
        memory = ('[{"kind":"trade_risk","subject":"","value":"0.01","text":"risk 1% per trade",'
                  '"at":"2026-10-05","price_at":null,"replaces":""},{"kind":"capital",'
                  '"subject":"","value":"50000","text":"a $50,000 account","at":"2026-10-05",'
                  '"price_at":null,"replaces":""}]')
        said = unit_checks.risk_sizing("If I put a stop 8% below entry on SOL, how many coins "
                                       "does 1% risk allow?", memory)
        assert said is not None and "$6,250" in said[0] and "50.00 SOL" in said[0]
        assert said[-1].startswith("Remembered:")

    def test_a_bare_view_is_expressed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.lui.research import rule_test, view_expression

        stamps = [datetime(2022, 1, 1) + timedelta(days=i) for i in range(900)]
        closes = [100.0 + (i % 40) - 20 + i * 0.05 for i in range(900)]
        monkeypatch.setattr(rule_test, "daily_closes",
                            lambda s: (stamps, closes, "test closes"))
        said = view_expression.lines("I think oil is going higher into winter. How would I "
                                     "express it and what would prove me wrong?",
                                     today=date(2026, 10, 5))
        assert said is not None and said[0].startswith("Bottom line: express it as a long CL")
        assert any(line.startswith("Wrong if:") for line in said)

    def test_a_relative_view_is_a_ratio_trade(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.lui.research import rule_test, view_expression

        stamps = [datetime(2022, 1, 1) + timedelta(days=i) for i in range(900)]
        monkeypatch.setattr(rule_test, "daily_closes", lambda s: (
            stamps, [100.0 + i * (0.1 if s == "XAUUSDT" else 0.05) + (i % 7) for i in range(900)],
            "test closes"))
        said = view_expression.lines("I think the dollar is topping out and gold will outperform "
                                     "bitcoin into year end. How would I express it?",
                                     today=date(2026, 10, 5))
        assert said is not None and "long the XAU perpetual and short the BTC" in said[0]
        assert any(line.startswith("Not expressed: the dollar") for line in said)

    def test_a_setup_reads_the_tape_and_the_rule(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.lui.research import rule_test, setup_check

        stamps = [datetime(2022, 1, 1) + timedelta(days=i) for i in range(400)]
        closes = [50.0 + i * 0.2 + (i % 5) for i in range(400)]
        monkeypatch.setattr(rule_test, "daily_closes", lambda s: (stamps, closes, "test"))
        said = setup_check.lines("Is SOL a good candidate this week?",
                                 ["I am a swing trader with a 50k account, max risk 1% per "
                                  "trade."])
        assert said is not None and "trend up" in said[0]
        assert any(line.startswith("Your rule: 1.0% of $50,000") for line in said)

    def test_a_diversifier_is_replayed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.lui.research import diversifier, drawdown_sizing

        days = [datetime(2022, 1, 3) + timedelta(days=i) for i in range(600)]
        monkeypatch.setattr(drawdown_sizing, "_aligned", lambda symbols: (
            days, {"SPYUSDT": [100 + i * 0.05 + (i % 9) for i in range(600)],
                   "TLTUSDT": [100 - i * 0.01 + (i % 4) for i in range(600)],
                   "BTCUSDT": [100 + i * 0.3 + (i % 13) * 3 for i in range(600)]},
            ["test closes"]))
        said = diversifier.lines("what does that mean for using BTC as a diversifier in a "
                                 "60/40 book?")
        assert said is not None and "60%/40%" in said[0]
        assert [line.split(":")[0] for line in said[1:4]] == ["Without", "With 5%", "With 10%"]


class TestRoutingGuards:
    def test_arb_the_token_is_not_arbitrage(self) -> None:
        from argus.lui.research.venue_compare import ASKED

        assert not ASKED.search("What is the circulating versus max supply of SUI and ARB?")
        assert ASKED.search("is there an arb between Binance and Bitget")

    def test_a_hedge_is_not_the_basis(self) -> None:
        from argus.lui.research.desk_answers import perp_vs_spot_lines

        assert perp_vs_spot_lines("I hold 50% ETH, 30% SOL and 20% BTC. How would I hedge the "
                                  "downside using perps without selling spot?", []) is None

    def test_a_depth_question_skips_the_order_book_explainer(self) -> None:
        from argus.lui.newcomer import reply

        found = reply("How deep is the BTC order book on Bitget within 1% of mid?", named=False)
        assert found is None or "live list of everyone's waiting orders" not in " ".join(
            found.lines)

    def test_a_desk_question_is_not_a_macro_indicator(self) -> None:
        from argus.lui.research import macro_indicators

        assert macro_indicators.lines("why did the desk sit out the CPI print") is None

    def test_the_recall_of_what_was_given(self) -> None:
        from argus.lui.memory import recall_asked

        assert recall_asked("Remind me what account size and risk rule I gave you")

    def test_funding_names_the_higher_rate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.lui.research import unit_checks
        from argus.market import bitget

        monkeypatch.setattr(bitget, "fetch_tickers", lambda: {
            "BTCUSDT": _ticker(85_000, funding=0.000065), "ETHUSDT": _ticker(2_700,
                                                                             funding=0.000055)})
        said = unit_checks.funding_verdict("Compare BTC and ETH funding and say which has the "
                                           "higher funding rate")
        assert said is not None and said[0].startswith("Bottom line: BTC has the higher funding")


class TestDeepOrders:
    def test_a_large_order_is_swept_on_the_full_book(self, monkeypatch: pytest.MonkeyPatch
                                                     ) -> None:
        from decimal import Decimal

        from argus.lui.research import execution
        from argus.market import depth

        def book(symbol: str, *, limit: int = 50, category: str = "USDT-FUTURES") -> Any:
            levels = 50 if limit == 50 else 200
            return depth.OrderBook(
                symbol=symbol, fetched_at=datetime(2026, 10, 6),
                bids=tuple(depth.Level(Decimal(100 - i * 0.01), Decimal(10))
                           for i in range(levels)),
                asks=tuple(depth.Level(Decimal(100.01 + i * 0.01), Decimal(10))
                           for i in range(levels)))

        monkeypatch.setattr(depth, "fetch_orderbook", book)
        monkeypatch.setattr(execution, "_optimal_schedule", lambda *a, **k: None)
        monkeypatch.setattr(execution, "_cadence_line", lambda *a, **k: None)
        monkeypatch.setattr(execution, "_session_change", lambda *a, **k: None)
        plan = SimpleNamespace(slices=[SimpleNamespace(index=1, fraction=Decimal(1), style="market",
                                                       expected_cost_bps=Decimal(6))])
        said = execution._depth_lines("BTCUSDT", Decimal(100_000), Decimal(10_000_000), plan,
                                      "a $100k market buy")
        assert said and "beyond what can be seen" not in said[0]
