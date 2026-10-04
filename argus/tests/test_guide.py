"""The guided research task (`lui/guide.py`, build-list 4.3) and its step-four engine
(`lui/research/expressions.py`), offline: every market read stubbed, each figure checked by hand."""

from __future__ import annotations

import math
from typing import Any

import pytest

from argus.lui import guide, server
from argus.lui.research import expressions
from argus.truth import coverage


class TestCoverage:
    def test_a_nested_recording_reaches_the_outer_one(self) -> None:
        with coverage.recording() as outer:
            with coverage.recording() as inner:
                inner.note("Bitget tickers", True)
                inner.note("bitget-signal news_feed", False, "timed out")
                inner.note_model_call()
            outer.note("SEC EDGAR", True)
        assert outer.answered == {"Bitget tickers": True, "bitget-signal news_feed": False,
                                  "SEC EDGAR": True}
        assert outer.model_calls == 1 and inner.model_calls == 1
        assert "SEC EDGAR" not in inner.answered

    def test_the_cost_line(self) -> None:
        record = coverage.Record()
        record.note("Bitget tickers", True)
        record.note("bitget-signal news_feed", False, "timed out")
        record.note_model_call()
        record.note_model_call()
        said = guide.cost_line(record)
        assert said.startswith("What this step read: 1 source answered (Bitget tickers); 1 did "
                               "not (bitget-signal news_feed).")
        assert "2 calls to the language model (Qwen)" in said
        assert guide.cost_line(coverage.Record()).startswith("What this step read: nothing new")
        cached = coverage.Record()
        cached.note_model_call()
        assert guide.cost_line(cached).endswith("1 call to the language model (Qwen) on the "
                                                "hackathon key.")


class TestSteps:
    def test_each_step_offers_the_next(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.lui.research import quick_stats

        monkeypatch.setattr(quick_stats, "_realised", lambda s, d: 0.5)
        first = guide.envelope(1, "BTCUSDT")
        assert first["next"]["ask"] == ("I think BTC keeps rising over the next month — test "
                                        "that view")
        assert guide.envelope(3, "BTCUSDT")["next"]["ask"] == (
            "Compare spot, perpetual and options for $10,000 of BTC")
        # two standard deviations of a 50% year over 30 days: 1 - exp(-2 x 0.5 x sqrt(30/365))
        fall = (1 - math.exp(-2 * 0.5 * math.sqrt(30 / 365))) * 100
        assert guide.envelope(4, "BTCUSDT")["next"]["ask"] == (
            f"Stress test my book if BTC falls {fall:.0f}%")
        assert guide.envelope(5, "BTCUSDT")["next"] is None
        assert guide.envelope(1, None)["next"] is None

    def test_takeaways_carry_each_verdict(self) -> None:
        assert guide.takeaway(1, ["Bottom line: the thread to pull is BTC — its own move is the "
                                  "largest. Each holding's move is below."]) == (
            "the thread to pull is BTC — its own move is the largest.")
        thesis = ["Bottom line: your thesis on BTC, reason by reason — 1 supported.",
                  'Supported — "BTC keeps rising": The tape agrees: MACD +8.3.']
        assert guide.takeaway(2, thesis).startswith('Supported — "BTC keeps rising"')
        case = ["Bottom line: the case for going long BTC — no measured edge. More.",
                "Terms: bps = hundredths.", "Sizing: Do not add.",
                "BTC already carries 43% of the risk. Trim it to 23%."]
        assert guide.takeaway(3, case) == ("the case for going long BTC — no measured edge. "
                                           "Sizing: Do not add. BTC already carries 43% of the "
                                           "risk.")
        stress = ["Bottom line: BTC alone loses 8.00% of the book.", "Terms: beta.",
                  "If BTC moves -20%: your book moves about -16.50%, hardest hit ETH."]
        assert guide.takeaway(5, stress).startswith("If BTC moves -20%: your book moves about")

    def test_the_briefing_pulls_the_largest_own_move(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.lui import research
        from argus.lui.answer import Answer

        def run(text: str, request: Any, **_: Any) -> Answer:
            symbol = request.symbols[0]
            own = {"BTCUSDT": "+0.10", "ETHUSDT": "-1.20", "NVDAUSDT": "+0.40"}.get(symbol)
            if request.kind.value == "book":
                return Answer(question=None, lines=["Bottom line: risk.",  # type: ignore[arg-type]
                                                    "Where the risk sits: BTC 40% of the money"])
            return Answer(question=None, lines=[  # type: ignore[arg-type]
                "Bottom line: headlines.",
                f"{symbol} is +0.5% over 24 hours; the other {own}% is {symbol[:3]}'s own"],
                data={"news": {"change_24h_pct": 0.5, "headlines": [
                    {"title": "A headline", "feed": "decrypt", "link": "https://decrypt.co/x"}]}})

        monkeypatch.setattr(research, "run", run)
        lines, _, focus = guide.briefing({"BTCUSDT": 0.4, "ETHUSDT": 0.3, "NVDAUSDT": 0.3}) or (
            [], [], None)
        assert focus == "ETHUSDT"
        assert lines[0].startswith("Bottom line: the thread to pull in your book today is ETH")
        assert any(x.startswith("ETH: +0.50% in 24 hours (-1.20% of it its own") for x in lines)
        assert lines[-1].startswith("Where the risk sits:")


class TestExpressions:
    def test_liquidating_windows_by_hand(self) -> None:
        closes = [100, 90, 80, 60, 70, 100, 100, 100]
        # 3-day windows from each of the first five closes: 100 -> 60 is a 40% fall, 90 -> 60 a
        # 33% fall, 80 -> 60 25%; 60 and 70 never fall a third
        assert expressions.liquidating_windows(closes, 0.33, days=3) == (2, 5)

    def test_the_comparison_matches_a_hand_count(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.lui.research import parse
        from argus.market import bitget, crossasset_feed

        closes = [100.0 * math.exp(0.01 * ((-1) ** i)) for i in range(400)]
        monkeypatch.setattr(parse, "last_price", lambda s: 100.0)
        monkeypatch.setattr(parse, "is_us_equity", lambda s: False)
        monkeypatch.setattr(expressions, "_closes", lambda s, d: closes)
        monkeypatch.setattr(expressions, "_funding_month", lambda s: (0.0001, 90))
        monkeypatch.setattr(expressions, "_spot_taker", lambda s: 0.002)
        monkeypatch.setattr(crossasset_feed, "fetch_taker_bps", lambda s: 6.0)
        monkeypatch.setattr(bitget, "maintenance_margin_rate", lambda s, n: 0.004)
        said = expressions.lines("compare spot vs perp for $10,000 of BTC")
        assert said is not None
        lines, table = said
        # carry: 0.01% x 90 settlements x $10,000 = $90; perp fees 2 x 6bps = $12; spot 2 x 0.2%
        assert "BTC perpetual, 1x: capital $10,000; a month's carry -$90; round-trip fees -$12" \
            in lines[2]
        assert lines[1].startswith("BTC spot on Bitget: capital $10,000; a month's carry $0; "
                                   "round-trip fees -$40")
        assert "BTC perpetual, 3x: capital $3,333" in lines[3]
        assert "liquidated after a 32.9% fall; 0 of the last 370 thirty-day windows" in lines[3]
        assert lines[0].startswith("Bottom line: for $10,000 of BTC over a month, the cheapest to "
                                   "hold is BTC spot on Bitget (-$40 in carry and fees)")
        assert table["columns"][0] == "Way" and len(table["rows"]) == 3

    def test_a_stock_is_held_on_spot_as_its_rtoken(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.lui.research import parse
        from argus.market import bitget, crossasset_feed

        closes = [100.0 * math.exp(0.01 * ((-1) ** i)) for i in range(400)]
        monkeypatch.setattr(parse, "last_price", lambda s: 100.0)
        monkeypatch.setattr(parse, "is_us_equity", lambda s: True)
        monkeypatch.setattr(expressions, "_closes", lambda s, d: closes)
        monkeypatch.setattr(expressions, "_call", lambda *a: None)
        monkeypatch.setattr(expressions, "_funding_month", lambda s: (0.0001, 90))
        # Bitget lists RTSLAUSDT on spot at 0.1% taker (2026-10-04); a stock with no rToken has none
        monkeypatch.setattr(expressions, "_spot_taker",
                            lambda s: 0.001 if s == "RTSLAUSDT" else None)
        monkeypatch.setattr(crossasset_feed, "fetch_taker_bps", lambda s: 6.0)
        monkeypatch.setattr(bitget, "maintenance_margin_rate", lambda s, n: 0.005)
        said = expressions.lines("compare spot vs perp for $10,000 of TSLA")
        assert said is not None
        assert said[0][1].startswith("rTSLA spot on Bitget: capital $10,000; a month's carry $0; "
                                     "round-trip fees -$20")
        other = expressions.lines("compare spot vs perp for $10,000 of PLTR")
        assert other is not None and other[0][1].startswith("PLTR shares at a broker")
        assert "Bitget lists no RPLTRUSDT token on spot" in other[0][1]

    def test_not_every_or_is_a_comparison(self) -> None:
        assert expressions.lines("is BTC or ETH riskier?") is None
        for asked in ("what is the best way to express a bullish view on NVDA?",
                      "should I buy BTC spot or the perp?", "how best to play ETH, spot or perp?"):
            assert expressions.ASKED.search(asked), asked
        for other in ("how to trade BTC on bitget, is it safe?",
                      "is 10x leverage ok for a small account",
                      "whats the funding rate on the BTC perp"):
            assert not expressions.ASKED.search(other), other


class TestConsole:
    def test_a_briefing_is_routed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(guide, "briefing", lambda book: (
            ["Bottom line: the thread to pull in your book today is ETH."], [], "ETHUSDT"))
        got = server.handle_ask("Brief me on my book: what moved each holding?", [],
                                book="40% BTC, 60% ETH")
        assert got["lines"][0].startswith("Bottom line: the thread to pull")
        assert got["focus"] == "ETHUSDT"
        # a route that builds its payload by hand still says how long it took
        assert isinstance(got["elapsed_ms"], float)

    def test_the_page_carries_the_guide(self) -> None:
        assert 'id="guide-go"' in server.PAGE and "gnext" in server.PAGE
        assert "Your research, in" in server.PAGE
        # build-list 4.8: an answer can be pinned and is asked again on every visit
        assert "argus.pins" in server.PAGE and 'class="pinb"' in server.PAGE
