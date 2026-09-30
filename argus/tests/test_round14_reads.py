"""Round 14 of the §27 audits (2026-09-30): a judge's pass on a trader who keeps a book, a loss
limit and a thesis across a session.

Each test pins one finding (tracker rows 563-575). Live sources are replaced by stubs or never
reached, so nothing here touches the network.
"""

from __future__ import annotations

import pytest

from argus.lui.research.kinds import ResearchKind, ResearchRequest


class TestSizingAndRiskShareQuestionsReachTheirReader:
    """A saved book and a stated limit must not turn a leg-sizing question into a hedge, or a
    risk-share question into an add-a-name test."""

    @pytest.mark.parametrize("q", [
        "How big should a SOL short leg be given my 4% drawdown limit?",
        "how large a position in ETH should I take",
        "What size position in BTC should I hold?"])
    def test_size_questions_match(self, q: str) -> None:
        from argus.lui.research.parse import SIZE_Q

        assert SIZE_Q.search(q)

    @pytest.mark.parametrize("q", ["how big is the BTC market", "what size is the account",
                                   "what is a position limit"])
    def test_other_questions_do_not(self, q: str) -> None:
        from argus.lui.research.parse import SIZE_Q

        assert not SIZE_Q.search(q)

    @pytest.mark.parametrize("q", ["Size the SOL short.", "size a BTC long leg"])
    def test_the_imperative_form_matches(self, q: str) -> None:
        from argus.lui.research.parse import SIZE_Q, detect

        assert SIZE_Q.search(q)
        request = detect(q)
        assert request is not None and request.kind is ResearchKind.IMPACT

    def test_the_imperative_form_is_not_greedy(self) -> None:
        from argus.lui.research.parse import SIZE_Q

        assert not SIZE_Q.search("Size the account settings page")

    def test_a_short_leg_question_is_a_short_impact(self) -> None:
        from argus.lui.research.parse import detect

        request = detect("How big should a SOL short leg be given my 4% drawdown limit?")
        assert request is not None
        assert request.kind is ResearchKind.IMPACT
        assert request.side == "short"
        assert request.symbols == ("SOLUSDT",)

    @pytest.mark.parametrize("q", [
        "What share of my book's risk does ETH carry, and why is it more than its 50% weight?",
        "Does ETH carry more risk than its weight in my book?"])
    def test_risk_share_questions_are_the_book(self, q: str) -> None:
        from argus.lui.research.parse import detect

        request = detect(q)
        assert request is not None and request.kind is ResearchKind.BOOK


class TestRiskPremiseIsChecked:
    _SITS = ("Where the risk sits: ETH 50% of the money, 49% of the risk; SOL 30% of the money, "
             "36% of the risk; TSLA 10% of the money, 4% of the risk")

    def test_a_false_premise_is_said_not_to_hold(self) -> None:
        from argus.lui.research.book import risk_premise_line

        line = risk_premise_line(
            "What share of my book's risk does ETH carry, and why is it more than its 50% weight?",
            [self._SITS])
        assert line is not None
        assert line.startswith("Premise check: ETH carries 49% of the risk on 50%")
        assert "less than its weight, not more" in line
        assert "SOL (36% of the risk on 30% of the money)" in line

    def test_a_true_premise_is_explained(self) -> None:
        from argus.lui.research.book import risk_premise_line

        line = risk_premise_line("why does SOL carry more risk than its weight?", [self._SITS])
        assert line is not None and line.startswith("Why: SOL carries 36% of the risk on 30%")

    def test_a_question_without_the_premise_gets_nothing(self) -> None:
        from argus.lui.research.book import risk_premise_line

        assert risk_premise_line("what share of my risk is ETH", [self._SITS]) is None


class TestTokenizedStocksKeepTheirWeights:
    def test_every_rtoken_named_is_read(self) -> None:
        from argus.lui.research.parse import rtokens_named

        found = [name for name, _ in rtokens_named("10% rTSLA and 10% rCOIN")]
        assert found == ["RTSLAUSDT", "RCOINUSDT"]

    def test_their_weights_pair_with_them(self) -> None:
        from argus.lui.research.parse import holding_pairs

        weights = {symbol: weight for _, symbol, weight in holding_pairs(
            "My book is $20,000: 50% ETH perp long, 30% SOL perp long, 10% rTSLA and 10% rCOIN.")}
        assert weights == {"ETHUSDT": 0.5, "SOLUSDT": 0.3, "TSLAUSDT": 0.1, "COINUSDT": 0.1}


class TestShortLegIsSizedOnTheLimit:
    def test_the_memory_line_sizes_the_leg_on_the_limit(self) -> None:
        from argus.lui import memory as mem

        facts = [
            mem.Fact(kind="capital", subject="", value="20000", text="My book is $20,000",
                     at="2026-09-30"),
            mem.Fact(kind="max_loss", subject="", value="0.04", text="my 4% drawdown limit",
                     at="2026-09-30")]
        lines = ["Bottom line: size SOL so that its worst observed 24 hours (-6.9%) as a short, "
                 "the loss on its biggest rally, is a loss you would accept — on a $10,000 "
                 "position that is about $688."]
        request = ResearchRequest(kind=ResearchKind.IMPACT, symbols=("SOLUSDT",), side="short")
        extra = " ".join(mem.after(lines, request, facts))
        assert "$800" in extra and "$11,594 leg (58% of the book)" in extra
        assert "the worst move above" not in extra


class TestRatioRangeIsTheRatiosOwn:
    def test_the_range_comes_from_the_ratio_not_one_leg(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        from datetime import UTC, datetime, timedelta
        from types import SimpleNamespace

        from argus.lui.research import quote
        from argus.market import history

        now = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
        hours = [now - timedelta(hours=h) for h in range(24 * 32, 0, -1)]

        def bars(symbol: str, **_: object) -> list[SimpleNamespace]:
            base = 2000.0 if symbol == "ETHUSDT" else 80000.0
            return [SimpleNamespace(ts=t, close=base * (1 + 0.0001 * i), high=0, low=0)
                    for i, t in enumerate(hours)]

        monkeypatch.setattr(history, "fetch_window", bars)
        text = quote._ratio_over_time("ETHUSDT", "BTCUSDT", 30)
        assert text is not None and "ETH/BTC ratio ran from" in text and "now" in text


    def test_ranged_reads_as_a_period_range_and_skips_the_gap_line(self) -> None:
        from argus.lui.research.parse import _PERIOD_MOVE, _RATIO_Q, _period_days

        text = "How has the ETH/BTC ratio ranged over the last 30 days?"
        assert _PERIOD_MOVE.search(text) and _RATIO_Q.search(text)
        assert _period_days(text) == 30


class TestThesisTurnsKeepTheirSide:
    def test_a_short_thesis_is_a_bear_case(self) -> None:
        from argus.lui.thesis_answer import _case

        assert _case("I am short TSLA because margins are falling") == "bear"
        assert _case("I think NVDA keeps running because capex is accelerating") == "bull"

    def test_a_revision_is_applied_to_the_reasons(self) -> None:
        from argus.lui.thesis_answer import _standing

        turns = [
            "I think NVDA keeps running because AI capex is accelerating, gross margins are "
            "expanding, and the data center bookings backlog is huge. Should I add 15% TSLA?",
            "Actually I was wrong about capex accelerating - it's slowing next quarter."]
        found = _standing(turns)
        assert found is not None
        _, stated, kept, dropped = found
        assert len(stated) == 4 and len(kept) == 3 and len(dropped) == 1


class TestMemoryReadsWhatTheTraderSaid:
    def test_a_book_is_not_a_bank_balance(self) -> None:
        from argus.lui.research.sizing import stated_capital

        assert stated_capital("My book is $20,000") == 20000.0

    def test_recall_lists_everything_with_its_date(self) -> None:
        from argus.lui import memory as mem

        facts = [mem.Fact(kind="max_loss", subject="", value="0.04", text="my 4% limit",
                          at="2026-09-30")]
        lines = mem.recall_lines(facts)
        assert lines[0].startswith("Bottom line: 1 thing remembered")
        assert any("my 4% limit" in line and "2026-09-30" in line for line in lines)
        assert mem.recall_asked("What do you remember about me?")
        assert mem.recall_lines([])[0].startswith("Bottom line: nothing is remembered")


class TestShortWindowReadsEveryBar:
    def test_the_short_worst_window_is_the_biggest_rally_over_the_full_history(self) -> None:
        from argus.desk.portfolio import worst_window

        quiet = [0.0] * 60
        rally = [0.004] * 24
        full = {"SOLUSDT": [*quiet, *rally, *quiet]}
        short = worst_window(weights={"SOLUSDT": -1.0}, columns=full)
        assert short.move_pct is not None and short.move_pct < -9.0
