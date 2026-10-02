"""Round 19 audit findings (judge, hostile reviewer, first-time user), each pinned to the reader
that misread it. Offline: no test here reaches a venue."""

from __future__ import annotations

from datetime import UTC, datetime

from argus.lui import intro, memory, multistep, thesis, thesis_answer
from argus.lui.question import Intent, classify
from argus.lui.research import sizing
from argus.lui.research.parse import _SELL_ALL, stated_direction
from argus.lui.server import _OWN_LOSS_Q, _statement_only, _without_others_holdings


class TestShockDirection:
    def test_a_negated_fall_followed_by_a_rally_is_a_rally(self) -> None:
        assert stated_direction("What if the Nasdaq does NOT drop, but rallies 8%?") == 1

    def test_rises_and_falls_read_their_own_clause(self) -> None:
        assert stated_direction("what if NVDA rises 15%") == 1
        assert stated_direction("what if the nasdaq drops 10%") == -1
        assert stated_direction("what if it does not rally but falls 5%") == -1

    def test_a_bare_signed_figure_states_no_word(self) -> None:
        assert stated_direction("QQQ -8%") is None


class TestShortsAndStops:
    def test_a_stated_trade_joins_the_question_after_it(self) -> None:
        found = multistep.parts(
            "I want to short TSLA. Where is my liquidation at 10x and what is my worst case?")
        assert found is None  # one question: the leverage reader answers it whole

    def test_stop_from_position_and_loss_limit(self) -> None:
        said = sizing.stop_for(
            "I have a $50k account and want to short 10% COIN, can't lose more than 3% of the "
            "account. Where should my stop go?", "COINUSDT", price=lambda _s: 200.0,
            atr=lambda _s: 0.02)
        assert said is not None
        lead = said[0][0]
        assert "29.9% above your entry" in lead and "about 259.76" in lead
        assert "$1,500 on the $5,000 short" in lead

    def test_european_figures(self) -> None:
        assert sizing.spelled_out("account 25.000, risk 0,5%") == "account 25000, risk 0.5%"
        assert sizing.spelled_out("price 3.14159") == "price 3.14159"


class TestOwnLossNotTheDesk:
    def test_the_traders_own_loss_is_not_the_desk_record(self) -> None:
        now = datetime.now(UTC)
        for q in ("how much can I lose", "how much could I lose in a bad week",
                  "I'm scared of losing money"):
            assert classify(q, now=now).intent is not Intent.PERFORMANCE, q
            assert _OWN_LOSS_Q.search(q), q

    def test_a_statement_is_told_apart_from_a_question(self) -> None:
        assert _statement_only("My max drawdown is 10% and my horizon is 3 months.")
        assert not _statement_only("what's my drawdown")


class TestThesisFollowUps:
    def test_case_against_is_not_the_strongest_reason_for(self) -> None:
        assert thesis_answer.AGAINST.search("What is the strongest case against my NVDA thesis?")
        assert thesis_answer.strongest("What is the strongest case against my NVDA thesis?",
                                       ["I think NVDA runs on AI capex, test my thesis"]) is None

    def test_a_challenge_with_nothing_new_is_a_follow_up(self) -> None:
        assert thesis_answer._follow_up_only(
            "Challenge my thesis: what is the strongest case against it?")
        assert not thesis_answer._follow_up_only(
            "I think NVDA runs on AI capex because hyperscalers keep spending, test my thesis")

    def test_what_would_make_you_change_your_mind(self) -> None:
        assert thesis_answer.FALSIFY.search("what would make you change your mind")

    def test_capex_keeps_rising_is_a_driver_not_a_trend(self) -> None:
        kinds = {r.text: r.kind for r in thesis.reasons(
            "I think semis will outperform because AI capex keeps rising")}
        assert kinds["AI capex keeps rising"] is thesis.Kind.DRIVER


class TestNewcomerTurns:
    def test_greeting_preamble_still_reads_as_the_intro(self) -> None:
        assert intro.INTRO_Q.search("hi, I'm new here. what is this and what can you do?")

    def test_no_name_buy_thanks_and_that_number(self) -> None:
        assert intro.NO_NAME_BUY_Q.search("what should I buy")
        assert intro.NO_NAME_BUY_Q.search("ok so should I?")
        assert intro.THANKS_Q.search("thanks, this was helpful")
        assert intro.THAT_NUMBER_Q.search("what does that number mean")
        lead = intro.no_name_buy_answer(1000.0, "DOGEUSDT")[0][0]
        assert "DOGE" in lead and "$1,000" in lead


class TestMemory:
    def test_stances_and_dislikes_are_kept(self) -> None:
        facts = memory.extract("I hate crypto and never trade it. I'm bearish on TSLA.")
        kinds = {(f.kind, f.subject, f.value) for f in facts}
        assert ("thesis", "TSLAUSDT", "bear") in kinds
        assert ("avoid", "", "crypto") in kinds

    def test_a_changed_view_replaces_the_old_and_is_recalled(self) -> None:
        old = memory.extract("I'm bearish on TSLA.")
        facts = memory.merge(old, memory.extract("Actually I'm now bullish on TSLA."))
        said = memory.recall_view("What's my view on TSLA?", facts)
        assert said is not None and "bullish" in said[0] and "replacing" in said[0]

    def test_a_sale_updates_the_remembered_book(self) -> None:
        facts = memory.extract("I hold 50% NVDA, 50% AAPL.")
        after = memory.apply_sales(facts, "Correction: I sold all my AAPL.")
        book = memory.get(after, "book")
        assert book is not None and book.text == "I hold 100% NVDA" and "AAPL" in book.replaces

    def test_several_facts_asked_back_and_the_missing_one_named(self) -> None:
        q = "What is my account size, what do I hold, and what is my max loss per trade?"
        assert memory.recall_asked(q)
        missing = memory.recall_missing(q, memory.extract("I have a $80,000 account."))
        assert missing and "loss limit" in missing[0] and "holdings" in missing[0]

    def test_someone_elses_holdings_are_not_the_traders(self) -> None:
        rest, note = _without_others_holdings(
            "My friend holds 100% COIN and my advisor says buy MSTR.")
        assert rest == "my advisor says buy MSTR." and "Not read as yours" in note
        rest, _ = _without_others_holdings(
            "my wife owns 50 NVDA, what if NVDA drops 10%? I hold 30% NVDA")
        assert rest == "what if NVDA drops 10%? I hold 30% NVDA"


class TestSales:
    def test_selling_a_held_name(self) -> None:
        assert _SELL_ALL.search("should I sell my SOL?")
        assert _SELL_ALL.search("what if I sold my ETH")
        assert not _SELL_ALL.search("should I buy SOL")


class TestRound19SecondPass:
    def test_an_order_to_sell_stays_an_order(self) -> None:
        from argus.lui.research.parse import selling_asked

        assert selling_asked("should I sell my SOL?")
        assert not selling_asked("Please simply sell all of my MSTR shares, I don't require any "
                                 "further explanation.")

    def test_a_short_is_read_on_the_short_side_by_both_readers(self) -> None:
        from argus.lui.research.parse import detect, with_book

        request = with_book(detect("Should I short TSLA?"), "40% NVDA, 60% TSLA",
                            "Should I short TSLA?")
        assert request is not None and request.side == "short"

    def test_a_follow_up_keeps_the_books_other_holdings(self) -> None:
        from argus.lui.research.parse import follow_up

        request = follow_up("What about NVDA?", ["Should I short TSLA?"], "40% NVDA, 60% TSLA")
        assert request is not None and set(request.symbols) == {"NVDAUSDT", "TSLAUSDT"}

    def test_two_names_and_a_comparison_are_compared(self) -> None:
        from argus.lui.research.kinds import ResearchKind
        from argus.lui.research.parse import detect

        request = detect("Compare NVDA and AMD — which has better momentum?")
        assert request is not None and request.kind is ResearchKind.COMPARE
        found = multistep.parts(
            "Compare NVDA and AMD: which is cheaper on valuation and which has better momentum?")
        assert found is not None and all(len(p.request.symbols) == 2 for p in found)

    def test_unknown_names_joined_in_other_languages(self) -> None:
        from argus.lui.research.parse import unread_names

        assert unread_names("¿Debería comprar FOOBARXYZ y AAPL?") == ["FOOBARXYZ"]
        assert unread_names("is AAPL or MSFT better") == []

    def test_a_saved_book_over_100_percent_is_said(self) -> None:
        from argus.lui.research.parse import detect, with_book

        request = with_book(detect("how risky is my book"), "60% AAPL, 30% NVDA, 30% MSFT",
                            "how risky is my book")
        assert request is not None and any("add up to 120%" in n for n in request.notes)

    def test_loss_limit_as_a_share_of_the_account_sizes(self) -> None:
        q = ("I have a $20,000 account and I cannot lose more than 5% of it. How big can my long "
             "MSTR position be with a stop?")
        assert sizing.asks_for_size(q)
        lines, _s, _d = sizing.answer(q, "MSTRUSDT", price=lambda _s: 100.0,
                                      worst_day=lambda _s: -0.05, atr=lambda _s: 0.02)
        assert "$1,000" in lines[0]
        said = sizing.stop_for("short 10% COIN on a $50,000 account with a 3% max loss. Where "
                               "should my stop go?", "COINUSDT", price=lambda _s: 100.0)
        assert said is not None and "$1,500 on the $5,000 short" in said[0][0]

    def test_profile_statements_are_not_thesis_reasons(self) -> None:
        q = ("I'm a swing trader, I can't lose more than 10%, and I think TSLA is overvalued "
             "because deliveries are falling.")
        kinds = {r.text: r.kind for r in thesis.reasons(q)}
        assert set(kinds) == {"TSLA is overvalued", "deliveries are falling"}
        assert kinds["deliveries are falling"] is thesis.Kind.DRIVER
        assert thesis.profile_parts(q) == ["I'm a swing trader", "I can't lose more than 10%"]

    def test_a_question_about_the_thesis_is_not_a_reason(self) -> None:
        q = ("I think NVDA keeps running because AI capex is still accelerating — I want to go "
             "long NVDA for the next 3 months. Is that a good thesis?")
        assert [r.text for r in thesis.reasons(q)] == ["NVDA keeps running",
                                                      "AI capex is still accelerating"]

    def test_the_breaker_reason_comes_from_the_record(self) -> None:
        from argus.lui import agent_answer

        rows = [{"seq": 2, "decided_at": "2026-10-02T04:00:00Z", "summary": "breaker on",
                 "mandate_response": "The circuit breaker is active (book.consecutive_losses = 4)"},
                {"seq": 1, "decided_at": "2026-10-01T04:00:00Z", "summary": "x",
                 "mandate_response": "circuit breaker (book.drawdown_pct = -0.05 exceeds "
                                     "kernel.breaker_reduce_only_drawdown_pct = 2.5, and "
                                     "kernel.breaker_losing_streak = 4)"}]
        lines, _s, _d = agent_answer.breaker_answer(lambda _name: rows)
        assert "4 consecutive losing closes against a 4-loss rule" in lines[0]
        assert agent_answer.asks_follow_up("Why is its circuit breaker on?",
                                           ["How is the Track 2 agent doing?"])

    def test_a_narrated_drawdown_the_record_does_not_hold_is_flagged(self) -> None:
        from argus.lui.agent_page import narration_check

        row = {"summary": "Circuit breaker active (4 consecutive losses, drawdown beyond 2.5%)",
               "mandate_response": "book.drawdown_pct = -0.0479769"}
        note = narration_check(row)
        assert note is not None and "0.048%" in note
        assert narration_check({**row, "mandate_response": "book.drawdown_pct = -3.1"}) is None

    def test_the_next_release_dates_come_from_the_calendar(self) -> None:
        from datetime import date

        from argus.lui.research.macro import _next_releases

        line = _next_releases("When is the next FOMC/CPI?", date(2026, 10, 2))
        assert line is not None and "next FOMC rate decision 2026-" in line
        assert "next CPI 2026-" in line
        assert _next_releases("what is CPI doing", date(2026, 10, 2)) is None

    def test_a_glossary_line_only_when_two_terms_are_used(self) -> None:
        from argus.lui.server import glossary_line

        assert glossary_line(["beta 1.2 and 14bps of cost"]) is not None
        assert glossary_line(["beta 1.2 only"]) is None
