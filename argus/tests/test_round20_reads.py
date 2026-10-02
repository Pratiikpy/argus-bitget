"""Round 20 audit findings, each pinned to the reader that misread it. Offline."""

from __future__ import annotations

from argus.lui import memory, thesis
from argus.lui.research import sizing
from argus.lui.research.parse import detect, holding_shocks, with_book


class TestSidesAndShocks:
    def test_a_short_only_book_keeps_its_sign(self) -> None:
        request = detect("I am short 100% TSLA. What happens if TSLA rallies 20%?")
        assert request is not None and request.book == {"TSLAUSDT": -1.0}
        assert request.shock_on == "TSLAUSDT" and request.shock_pct == 20.0

    def test_named_and_joint_shocks(self) -> None:
        book = "50% NVDA, 50% TSLA"
        q = "What if not NVDA but TSLA falls 10%?"
        request = with_book(detect(q), book, q)
        assert request is not None and request.shock_on == "TSLAUSDT"
        assert holding_shocks("What if NVDA and TSLA both fall 10%?") == {
            "NVDAUSDT": -10.0, "TSLAUSDT": -10.0}
        assert holding_shocks("What if TSLA falls five percent and NVDA falls 0,5%?") == {
            "TSLAUSDT": -5.0, "NVDAUSDT": -0.5}

    def test_an_account_is_not_a_holding(self) -> None:
        q = "I have a $40k account. How much would I lose in dollars if NVDA fell 20%?"
        request = with_book(detect(q), "50% NVDA, 50% TSLA", q)
        assert request is not None and set(request.book) == {"NVDAUSDT", "TSLAUSDT"}


class TestAccounts:
    def test_grand_european_and_euro_accounts(self) -> None:
        lines, _s, _d = sizing.answer(
            "Size a short on TSLA: account 20 grand, risk 1%, stop 3% above entry.", None)
        assert "$200" in lines[0] and "$6,410" in lines[0]
        lines, _s, _d = sizing.answer("Risk 2% of 1.000.000 on BTC with a 1% stop.", None)
        assert "$20,000" in lines[0] and "$1,785,714" in lines[0]
        lines, _s, _d = sizing.answer("Risk 1.5% of 8.000 EUR on ETH with a 2% stop.", None)
        assert "€120" in lines[0] and "€5,660" in lines[0]

    def test_a_stop_from_structure_when_no_position_is_given(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        rows = [(110.0, 95.0 + i * 0.1, 100.0) for i in range(20)]
        monkeypatch.setattr(thesis, "_closes", lambda _s: rows)
        said = sizing.stop_for("My account is $25,000 and I can't lose more than $1,000 on NVDA. "
                               "Where should the stop go?", "NVDAUSDT", price=lambda _s: 100.0)
        assert said is not None and "lowest low" in said[0][0] and "$1,000 at risk" in said[0][0]


class TestMemory:
    def test_limits_and_dislikes_kept_as_said(self) -> None:
        facts = memory.extract("I have $3,000 total, I can only afford to lose $300, I hold for "
                               "a few months, and I don't want anything with leverage or meme "
                               "coins")
        kinds = {(f.kind, f.value) for f in facts}
        assert {("loss_usd", "300"), ("capital", "3000"), ("avoid", "leverage"),
                ("avoid", "meme coins")} <= kinds
        assert ("avoid", "meme coins") in {(f.kind, f.value)
                                          for f in memory.extract("I avoid meme coins")}

    def test_corrections_and_sales_keep_cash(self) -> None:
        facts = memory.extract("I hold 40% NVDA, 60% AAPL.")
        sold = memory.get(memory.apply_sales(facts, "I sold all my AAPL."), "book")
        assert sold is not None and sold.text == "I hold 40% NVDA, 60% cash"
        fixed = memory.get(memory.apply_corrections(facts, "Actually it's 30% NVDA, not 40%."),
                           "book")
        assert fixed is not None and fixed.text == "I hold 30% NVDA, 60% AAPL, 10% cash"
        assert ("capital", "25000") in {(f.kind, f.value) for f in memory.extract(
            "My account is 25.000 USD and I think NVDA will fall.")}


class TestThesisEvidence:
    def test_hyperscaler_is_not_hype_and_a_beat_is_an_earnings_claim(self) -> None:
        kinds = {r.text: r.kind for r in thesis.reasons(
            "I think NVDA will outperform because hyperscaler capex keeps growing and "
            "data-center revenue beat expectations")}
        assert kinds["hyperscaler capex keeps growing"] is thesis.Kind.DRIVER
        assert kinds["data-center revenue beat expectations"] is thesis.Kind.EARNINGS


class TestHoldingLimitsAndHedges:
    def test_how_much_can_i_hold_is_the_limit_not_a_default_add(self) -> None:
        from argus.lui.research.parse import HOLD_TO_LIMIT

        q = "How much TSLA can I hold?"
        request = with_book(detect(q), "40% NVDA, 30% TSLA, 30% COIN", q)
        assert request is not None and request.target == HOLD_TO_LIMIT

    def test_a_loss_limit_carries_its_span(self) -> None:
        from argus.lui.research.dispatch import _LOSS_LIMIT

        found = _LOSS_LIMIT.search("if I can only lose 5% in a bad month")
        assert found is not None and found.group("pct") == "5" and found.group("span") == "month"

    def test_shorting_a_held_name_is_a_cut_not_a_hedge(self) -> None:
        from dataclasses import replace

        from argus.lui.research.kinds import ResearchKind
        from argus.lui.research.parse import hedge_instruments, shorting_a_holding

        q = "Should I short NVDA to hedge?"
        assert hedge_instruments(q) == ("NVDAUSDT",)
        request = detect(q)
        assert request is not None
        request = replace(request, kind=ResearchKind.HEDGE,
                          book={"NVDAUSDT": 0.4, "TSLAUSDT": 0.3, "COINUSDT": 0.3})
        cut = shorting_a_holding(request, q)
        assert cut.kind is ResearchKind.IMPACT and cut.side == "short"
        assert cut.symbols[0] == "NVDAUSDT"
        assert any("is not a hedge of this book" in n for n in cut.notes)


class TestMemoryKeepsTheWholeView:
    def test_the_thesis_keeps_its_reason(self) -> None:
        facts = memory.extract("I'm bullish on NVDA because hyperscaler capex keeps growing.")
        kept = memory.get(facts, "thesis", "NVDAUSDT")
        assert kept is not None and "hyperscaler capex keeps growing" in kept.text

    def test_margin_is_kept_as_margin(self) -> None:
        facts = memory.extract("I have $5,000 of margin in my account.")
        capital = memory.get(facts, "capital")
        assert capital is not None and capital.value == "5000" and "margin" in capital.text

    def test_a_trade_against_the_stated_view_is_said(self) -> None:
        facts = memory.extract("I'm now bearish on NVDA.")
        request = detect("Should I buy NVDA?")
        line = memory.against_view(request, facts)
        assert line is not None and line.startswith("Against your own view")
        assert memory.against_view(detect("Should I short NVDA?"), facts) is None

    def test_hold_until_earnings_is_not_an_earnings_style(self) -> None:
        from argus.lui.memory_model import _valid

        assert _valid("style", "earnings", "", "I plan to hold until earnings") is None
        assert _valid("style", "earnings", "", "I mostly trade earnings") is not None


class TestEventsAndScenarios:
    def test_mean_for_my_book_is_not_a_definition(self) -> None:
        from argus.lui.concepts import concept_asked

        assert concept_asked("What does the CPI print next week mean for my book?") is None
        assert concept_asked("What does CPI mean?") is not None

    def test_an_earnings_miss_is_its_own_scenario(self) -> None:
        from argus.lui.research.dispatch import _EARNINGS_MISS

        assert _EARNINGS_MISS.search("What if it misses earnings badly?")
        assert _EARNINGS_MISS.search("what if TSLA has a terrible quarter")
        assert not _EARNINGS_MISS.search("what if QQQ falls 10%")

    def test_a_named_crisis_is_replayed_per_holding(self, monkeypatch) -> None:
        from datetime import date

        from argus.lui.research import episodes

        falls = {"QQQ": -53.4, "SPY": -55.2, "NVDA": -84.0}
        monkeypatch.setattr(episodes, "own_fall", lambda e, t, **_kw: (
            (falls[t], date(2007, 10, 31), date(2009, 3, 9)) if t in falls else None))
        line = episodes.replay_line(episodes.named("like 2008"), ("NVDAUSDT", "TSLAUSDT"),
                                    lambda s: s.removesuffix("USDT"))
        assert line is not None
        assert "S&P 500 ETF -55%" in line and "NVDA -84%" in line
        assert "TSLA did not trade through it" in line


class TestRecordPages:
    def test_elapsed_hours_are_not_hourly_marks(self) -> None:
        from argus.lui.agent_answer import _elapsed_hours

        summary = {"scoring_window": {"start": "2026-09-28T00:00:00Z"},
                   "generated_at": "2026-10-02T07:02:44Z"}
        assert _elapsed_hours(summary) == 103
        assert _elapsed_hours({}) == 0


class TestTheFirstChild:
    def test_the_preview_is_one_minute_of_the_first_hour(self, monkeypatch) -> None:
        from decimal import Decimal
        from types import SimpleNamespace

        from argus.lui.research import dispatch
        from argus.market import universe

        monkeypatch.setattr(universe, "contracts", lambda: {"COINUSDT": SimpleNamespace(
            size_step=Decimal("0.01"), min_qty="0.01")})
        request = detect("How should I buy $2 million of COIN?")
        assert request is not None and request.notional == 2_000_000
        plan = SimpleNamespace(slices=[SimpleNamespace(fraction=Decimal("0.5"), style="market")])
        ticker = SimpleNamespace(last=Decimal("200"), ask=Decimal("200.1"), bid=Decimal("199.9"))
        lines = dispatch._agent_hub_lines("COINUSDT", request, plan, ticker,
                                          "How should I buy $2 million of COIN?",
                                          adv=Decimal("24000000"))
        # an hour may take 10% of an hour's volume, $100,000; half of it as market, over sixty
        # one-minute children: about $833, 4.16 COIN — not half the whole order (row 715)
        assert lines and "--qty 4.16 " in lines[0]


class TestLiveReAsk:
    def test_an_avoided_class_is_said_under_the_lead(self) -> None:
        facts = memory.extract("I avoid meme coins.")
        line = memory.against_view(detect("Should I buy DOGE?"), facts)
        assert line is not None and line.startswith("Against your own rule")
        assert "meme coins" in line

    def test_the_track2_answer_counts_refused_orders(self) -> None:
        from argus.lui import agent_answer

        files = {"summary.json": {"metrics": {"n_hours": 2, "n_closed_trades": 0},
                                  "counts": {"orders_sent": 2, "fills": 0}},
                 "orders.json": {"counts": {"sent": 2}, "orders": [
                     {"symbol": "METAUSDT", "purpose": "protective_exit", "state": "rejected",
                      "submitted_at": "2026-09-28T09:16:00Z",
                      "rejection": {"message": "HTTP 400 from Bitget: Parameter X does not "
                                               "exist"}}] * 2}}
        lines, _s, _d = agent_answer.answer(files.get)
        assert any(line.startswith("The venue refused 2 of the 2 orders sent") for line in lines)
