"""Round 45 audits (judge, hostile, newcomer, visual; Activity/audits/round45_*.md), fixed offline.

The builders' modules carry their own test files (trade_calc, portfolio_lab, fundamentals_lab,
round45_newcomer). These pin the readers and routes written alongside them: a restatement that
loops, corrections and additions to a stated position, the spot market when spot is asked, names
of two words, premises answered with "no", window and unit readers, past-day ranges and records,
desk limits, and the follow-ups that lost their subject."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


class TestRestatementLoop:
    def test_a_reading_that_changes_nothing_is_not_read_again(self) -> None:
        source = (Path(__file__).resolve().parents[1] / "src" / "argus" / "lui"
                  / "server.py").read_text(encoding="utf-8")
        assert "rewritten[0].strip() == text.strip()" in source
        assert "_RESTATE_DEPTH.get() >= 4" in source


class TestStatedPositions:
    def test_a_correction_rewrites_the_size(self) -> None:
        from argus.lui.server import _corrected_statement

        fixed = _corrected_statement("I bought 0.5 BTC at 80000 USDT. What is my profit now?",
                                     "Actually I meant 0.05 BTC, redo it.")
        assert fixed is not None
        assert fixed[0].startswith("I bought 0.05 BTC at 80000")
        assert "0.05 BTC, not 0.5" in fixed[1]
        assert _corrected_statement("I bought 0.5 BTC at 80000", "what is ETH doing") is None

    def test_return_on_margin(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.lui import server

        monkeypatch.setattr(server, "_price_now", lambda s: 86_000.0)
        said = server._return_on_margin("I bought 0.5 BTC at 80000 USDT.",
                                        "10x leverage on the original 0.5 BTC, return on margin?",
                                        (["x"], [], {}))
        assert said is not None and "$4,000.00" in said and "+75.0% on margin" in said

    def test_an_addition_is_not_an_order(self) -> None:
        from argus.lui.server import _ADDED_LEG

        m = _ADDED_LEG.search("Add 2 BTC at 90000 to that.")
        assert m is not None and (m.group("q"), m.group("s"), m.group("p")) == ("2", "BTC",
                                                                               "90000")

    def test_the_first_one_is_the_first_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.lui import server
        from argus.lui.research import parse

        names = {"What's the price of LINK, NEAR and GAS?": ("LINKUSDT", "NEARUSDT", "GASUSDT")}
        monkeypatch.setattr(parse, "research_symbols", lambda t: (names.get(t, ()), ()))
        import argus.lui.research as research

        monkeypatch.setattr(research, "research_symbols", parse.research_symbols)
        said = server._ordinal_name("And just the first one, in 3 days what is the typical move?",
                                    ["What's the price of LINK, NEAR and GAS?"])
        assert said.startswith("And LINK, in 3 days")


class TestNamesAndPremises:
    def test_two_word_coins(self) -> None:
        from argus.lui.research.parse import _TWO_WORD_COINS

        assert _TWO_WORD_COINS[r"\bbitcoin\s+cash\b"] == "BCH"
        assert _TWO_WORD_COINS[r"\bethereum\s+classic\b"] == "ETC"

    def test_a_ban_premise(self) -> None:
        from argus.lui.claims import ban_line

        said = ban_line("Ethereum fell to $3 yesterday after the Fed banned it", ("ETHUSDT",))
        assert said is not None and "does not ban assets" in said
        assert ban_line("ETH price now", ("ETHUSDT",)) is None

    def test_the_founder_and_launch_premise(self) -> None:
        from argus.lui.claims import founder_line

        said = founder_line("Bitcoin was created in 2012 by Vitalik. What's its price?",
                            ("BTCUSDT",))
        assert said is not None and "Satoshi Nakamoto" in said and "not 2012" in said

    def test_a_target_is_not_a_present_price(self) -> None:
        from argus.lui.server import _PRICE_CLAIM, _TARGET_PREMISE

        claim = _PRICE_CLAIM.search("will bitcoin hit 1 million this year?")
        assert claim is not None and (claim.group("k") or "").lower() == "million"
        target = _TARGET_PREMISE.search("will bitcoin hit 1 million this year?")
        assert target is not None and target.group("unit").lower() == "million"

    def test_a_stated_macro_figure_is_answered(self) -> None:
        from argus.lui.research.macro_indicators import _stated_figure

        assert _stated_figure("US CPI came in at 12% last month right?") == 12.0
        assert _stated_figure("What is the current Fed funds rate? I think it's 7.5%.") == 7.5
        assert _stated_figure("what is CPI now") is None

    def test_a_period_move_premise(self) -> None:
        from argus.lui.server import _period_move_premise

        said = _period_move_premise(
            "BTC up 1,000% this year, correct?",
            ["Bottom line: BTC is -2.2% so far this year (1 Jan to 06 Oct) — from 87,624.40"])
        assert said == "Bottom line: no — BTC is -2.2% this year, not +1000%."

    def test_finance_words_are_not_tickers(self) -> None:
        from argus.lui.question import _NOT_A_TICKER

        assert {"CAGR", "BUY", "PWNED", "EBITDA"} <= _NOT_A_TICKER
        assert "PUMP" not in _NOT_A_TICKER and "NOT" not in _NOT_A_TICKER


class TestUnitsAndArithmetic:
    def test_bps_and_a_stop(self) -> None:
        from argus.lui.research.unit_checks import move_and_stop

        said = move_and_stop("How many bps is a move from 2700 to 2727 in ETH, and is the 1% stop "
                             "I set at 2673 correct?")
        assert said is not None
        assert said[0].startswith("Bottom line: 2700 to 2727 is +100 bps (+1.00%).")
        assert "is right: exactly 1.00% below 2700" in said[0]

    def test_live_funding_beside_a_stated_rate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.lui.research import parse, unit_checks

        monkeypatch.setattr(parse, "research_symbols", lambda t: (("ETHUSDT",), ()))
        monkeypatch.setattr(unit_checks, "_ticker",
                            lambda s: SimpleNamespace(funding_rate=0.000046))
        said = unit_checks.per_period("Funding on ETHUSDT is 0.05 percent per hour right? "
                                      "Annualise it for me.")
        assert said is not None and said[0].startswith("Bottom line: no — ETH's funding")
        assert any("438.00% a year simple" in x for x in said)

    def test_the_dollar_cost_of_holding(self) -> None:
        from argus.lui.research.desk_math import funding_rate_lines

        said = funding_rate_lines("What does it cost to hold a 10,000 USDT BTC long for a year at "
                                  "0.01% funding every 8 hours? My exchange says it is free.")
        assert said is not None and "$1,095 a year" in said[0] and "not free" in said[0]


class TestWindowsAndRecords:
    def test_volatility_in_other_units(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.lui.research import quick_stats

        monkeypatch.setattr(quick_stats, "_named", lambda t, p: ["BTCUSDT"])
        monkeypatch.setattr(quick_stats, "_realised", lambda s, d: 0.365)
        said = quick_stats.vol_units_lines("Express that as daily volatility, and per hour.",
                                           ["Tell me BTC's 30 day volatility annualised."])
        assert said is not None and "1.91% a day" in said[0] and "0.390% an hour" in said[0]

    def test_the_chance_of_a_stated_fall(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.lui.research import quick_stats
        from argus.market import history

        closes = [100 * (1.01 if i % 2 else 0.99) ** (i % 3) for i in range(400)]
        monkeypatch.setattr(quick_stats, "_named", lambda t, p: ["BTCUSDT"])
        monkeypatch.setattr(history, "fetch_range", lambda *a, **k: [
            SimpleNamespace(close=c) for c in closes])
        said = quick_stats.drop_chance_lines("what's the chance BTC drops 50% tomorrow?", [])
        assert said is not None and "happened 0 times" in said[0] and "one-day" in said[0]

    def test_a_typical_move_over_days(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.lui.research import quick_stats
        from argus.market import history

        monkeypatch.setattr(quick_stats, "_named", lambda t, p: ["LINKUSDT"])
        monkeypatch.setattr(history, "fetch_range", lambda *a, **k: [
            SimpleNamespace(close=100 + (i % 7)) for i in range(400)])
        said = quick_stats.typical_move_lines("LINK, in 3 days what is the typical move?", [])
        assert said is not None and said[0].startswith("Bottom line: over 3 days, LINK")

    def test_a_named_days_high_and_low(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.lui.research import close_on, parse
        from argus.market import history

        day = date(2026, 10, 3)
        monkeypatch.setattr(parse, "research_symbols", lambda t: (("BTCUSDT",), ()))
        monkeypatch.setattr(parse, "is_us_equity", lambda s: False)
        monkeypatch.setattr(history, "fetch_window", lambda *a, **k: [SimpleNamespace(
            ts=datetime(2026, 10, 3, tzinfo=UTC), open=84_487.3, high=84_999.9, low=84_420.0,
            close=84_728.1)])
        said = close_on.close_lines("What were BTC's high and low on 2026-10-03 UTC?", [],
                                    now=datetime(2026, 10, 6, tzinfo=UTC))
        assert said is not None and "between 84,420.00 and 84,999.90" in said[0]
        assert day.isoformat() == "2026-10-03"


class TestDeskAnswers:
    def test_the_limits_come_from_the_policy(self) -> None:
        from argus.lui.answer import _risk_limits_lines

        said = _risk_limits_lines("What is the maximum position size the risk kernel allows per "
                                  "name, and who can override it?")
        assert said is not None and "$50,000 of notional" in said[0]
        assert any("25% of the book is held for a person" in x for x in said)
        assert _risk_limits_lines("how does your risk layer work") is None

    def test_what_rule_stopped_them(self, tmp_path: Path,
                                    monkeypatch: pytest.MonkeyPatch) -> None:
        import importlib

        answer = importlib.import_module("argus.lui.answer")
        path = tmp_path / "risk_records.jsonl"
        path.write_text("\n".join(json.dumps({"seq": s, "quantity_before": "0",
                                              "intervened": False}) for s in (1, 2)),
                        encoding="utf-8")
        monkeypatch.setattr(answer, "_risk_records_path", lambda: path)
        rows: Any = [SimpleNamespace(seq=1), SimpleNamespace(seq=2)]
        said = answer._rule_behind(rows)
        assert said is not None and said.startswith("No rule stopped them: in all 2")


class TestSmallFirstBuy:
    def test_simplified_keeps_the_first_clause(self) -> None:
        from argus.lui.server import _simplified

        assert _simplified("TSLA's worst 24 hours (-4.32%) would cost $432; over the year it "
                           "fell 14%") == "TSLA's worst 24 hours would cost $432."

    def test_the_chip_and_question_never_overflow(self) -> None:
        from argus.lui.server import PAGE

        assert "overflow-wrap:anywhere" in PAGE
        assert ".chip:hover, .chip:focus-visible { color:var(--ink); border-color:var(--accent);" \
               in PAGE

    def test_the_translation_note_shows_no_endpoint(self) -> None:
        from argus.lui.server import _TRANSLATION_FOLLOWS

        assert all("/translate" not in v and "token" not in v
                   for v in _TRANSLATION_FOLLOWS.values())


class TestSwappedFollowUp:
    def test_the_earlier_name_is_not_priced_again(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.lui import server
        from argus.market import bitget

        quote = SimpleNamespace(last="2700", change_24h="-0.01")
        monkeypatch.setattr(bitget, "fetch_tickers",
                            lambda: {"BTCUSDT": quote, "ETHUSDT": quote})
        lines = ["Bottom line: ETH last 2700 USDT on Bitget."]
        assert server._prices_left_out("bitcoin price?", lines, ("ETHUSDT",)) == []
        assert server._prices_left_out("bitcoin price?", lines)[0].startswith("BTC last")


class TestRound45Page:
    def test_an_empty_ask_says_so_under_the_box(self) -> None:
        from argus.lui.server import PAGE

        assert '<p class="empty" id="askerr" role="alert" hidden>' in PAGE
        assert "document.getElementById('askerr').hidden = false;" in PAGE

    def test_the_working_card_spins_unless_motion_is_reduced(self) -> None:
        from argus.lui.server import PAGE

        assert '<span class="spin" aria-hidden="true"></span>working' in PAGE
        assert "@media (prefers-reduced-motion: reduce) { .spin { animation:none } }" in PAGE

    def test_the_receipt_is_one_row_per_source_and_holds_the_coverage_note(self) -> None:
        from argus.lui.server import PAGE

        assert '<div class="rrow"><b>${esc(s.kind)}</b>: ${linked(s.ref)}' in PAGE
        assert r"/^(Sources reached|Did not answer)\b/.test(l)" in PAGE
        assert "a.lines.filter(l => inReceipt(a, l))" in PAGE
        assert ".src b { display:block" not in PAGE

    def test_no_text_under_twelve_pixels_on_the_console(self) -> None:
        import re

        from argus.lui.server import PAGE

        sizes = [float(x) for x in re.findall(r"(?<![\d.])(\d+(?:\.\d+)?)px/", PAGE)]
        sizes += [float(x) for x in re.findall(r"font-size:(\d+(?:\.\d+)?)px", PAGE)]
        assert sizes and min(sizes) >= 12


class TestFiledLabel:
    def test_a_filing_passage_is_filed_not_live(self) -> None:
        from argus.lui import provenance

        assert provenance.label("Revenue rose 12% on data-center demand. [1]") == "filed"
        assert provenance.label("Filed: revenue 46.7bn for the quarter") == "filed"
        assert provenance.label("Bottom line: Revenue rose 12% on demand. [1][2]") == "filed"
        assert provenance.label("BTC last 85,000 USDT on Bitget (-1.00% over 24h).") == "live"


class TestVenuePrices:
    def test_venue_prices_keep_enough_digits_for_a_bps_gap(self,
                                                           monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.lui.research import venue_compare
        from argus.market import bitget

        monkeypatch.setattr(bitget, "public_get", lambda *a, **k: [{"lastPr": "0.20641"}])
        monkeypatch.setattr(venue_compare, "_read",
                            lambda v, base: {"Coinbase": 0.20631, "Kraken": 0.20623}.get(v))
        said = venue_compare.lines("is there an arbitrage gap on ARB across exchanges",
                                   ("ARBUSDT",)) or []
        joined = " ".join(said)
        assert "Bitget spot 0.20641" in joined and "Coinbase 0.20631" in joined


def _unused(*_a: Any) -> timedelta:
    return timedelta(0)
