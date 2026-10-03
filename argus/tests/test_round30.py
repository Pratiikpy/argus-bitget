"""Round 30's three audits (judge, first-time user, hostile reviewer), run offline: each fix is
checked on the phrasing that exposed it and on the near-miss beside it, and every figure is
computed from a hand-made input whose answer can be read off it."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace

import pytest

from argus.lui import (
    concepts,
    goal_target,
    intro,
    memory,
    multistep,
    newcomer,
    personal_plan,
    premise_facts,
    question,
    rivals,
    server,
)
from argus.lui.research import carry, performance, starter, venue_facts


class TestMoneyStated:
    def test_a_thousands_comma_is_not_a_cut(self) -> None:
        found = memory._CAPITAL.search("I have $150,000 of fresh cash. What should I do")
        assert found is not None
        assert [g for g in found.groups() if g] == ["150,000"]

    def test_money_of_a_coin_is_still_a_holding_not_capital(self) -> None:
        assert memory._CAPITAL.search("i have $500 of BTC") is None

    def test_if_you_were_me_names_the_sum(self) -> None:
        assert starter.amount_of("if you were me with like 500 bucks what would you do") == 500

    def test_a_sub_cent_price_keeps_its_digits(self) -> None:
        assert performance._px(0.00001264) == "0.00001264"
        assert performance._px(84824.8) == "84,824.80"


class TestALossAlreadyTaken:
    @pytest.fixture(autouse=True)
    def _year(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(memory, "deepest_fall_year",
                            lambda s: {"BTCUSDT": 0.53, "ETHUSDT": 0.67}.get(s))

    def test_the_recovery_arithmetic(self) -> None:
        lines = newcomer.loss_lines("is that bad", ["my portfolio down 30%"])
        assert lines is not None
        assert "+43%" in lines[0]
        assert "53%" in lines[1]

    def test_should_i_sell_everything_is_answered_without_a_call(self) -> None:
        lines = newcomer.loss_lines("should i sell everything", ["my portfolio down 30%"])
        assert lines is not None and lines[0].startswith("Bottom line: no sell call")
        assert any("65% in total" in x for x in lines)

    def test_what_if_it_keeps_dropping_gives_each_further_fall(self) -> None:
        lines = newcomer.loss_lines("ok but what if it keeps dropping", ["my portfolio down 30%"])
        assert lines is not None
        assert "37% in total" in lines[0] and "44% in total" in lines[0]

    def test_no_stated_loss_no_answer(self) -> None:
        assert newcomer.loss_lines("is that bad", ["what is BTC doing"]) is None


class TestGoalTarget:
    def test_the_target_and_its_date(self) -> None:
        target = goal_target.stated("turn 1000 bucks into 5000 by end of this year",
                                    today=date(2026, 10, 3))
        assert target is not None
        assert target.needed == pytest.approx(4.0)
        assert target.days == 89

    def test_trading_every_day_costs_fees_first(self) -> None:
        target = goal_target.Target(1000, 5000, 89, date(2026, 12, 31))
        lines = goal_target.daily_or_hold_lines(target)
        assert "16%" in lines[0] and "+400%" in lines[0]

    def test_the_follow_ups_read_the_remembered_target(self) -> None:
        prior = ["im trying to turn 1000 bucks into 5000 by end of this year"]
        lines = goal_target.lines("if it doesnt work out by my deadline what should i do", prior,
                                  now=datetime(2026, 10, 3, tzinfo=UTC))
        assert lines is not None and "31 Dec 2026" in lines[0]


class TestPersonalPlan:
    PRIOR = ("I'm 58, planning to retire in 5 years, 70% of my liquid net worth is already in BTC "
             "and NVDA xStock tokens, and I can't stomach a drawdown bigger than 12%. I have "
             "$150,000 of fresh cash. What should I actually do with it?",)

    def test_the_constraints_are_read_and_the_latest_limit_wins(self) -> None:
        s = personal_plan.read("I can tolerate up to 25% now. Does that change your answer?",
                               list(self.PRIOR))
        assert (s.age, s.years, s.held_share, s.cash) == (58, 5, 0.7, 150_000)
        assert s.held == ("BTCUSDT", "NVDAUSDT")
        assert (s.limit, s.limit_before, s.limit_now) == (0.25, 0.12, True)

    @pytest.fixture
    def flat_then_fall(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # each held name falls 1% a day for 30 days, then recovers; the candidates are flat
        rets = {t: [-0.01] * 30 + [0.005] * 220 for t in ("BTC-USD", "NVDA")}
        rets.update({t: [0.0] * 250 for t in ("QQQ", "GLD")})
        monkeypatch.setattr(personal_plan, "_returns",
                            lambda tickers, days=365: ([], {t: rets[t] for t in tickers}))

    @pytest.mark.usefixtures("flat_then_fall")
    def test_the_held_book_breaks_the_limit_before_the_cash(self) -> None:
        lines = personal_plan.lines(self.PRIOR[0], [])
        assert lines is not None
        assert "breaks your 12% drawdown limit on its own" in lines[0]
        assert any("$500,000" in x for x in lines)

    @pytest.mark.usefixtures("flat_then_fall")
    def test_var_leads_when_asked_and_shows_the_math(self) -> None:
        lines = personal_plan.lines("Size the new $150,000 so that my total book's 1-day 95% VaR "
                                    "doesn't exceed $9,000. Show the math.", list(self.PRIOR))
        assert lines is not None and lines[0].startswith("Bottom line: 1-day 95% VaR")
        assert "$175,000 x BTC-USD return" in lines[0]

    @pytest.mark.usefixtures("flat_then_fall")
    def test_the_ticket_sells_what_the_limit_cannot_carry(self) -> None:
        sold = personal_plan.trades("Give me the exact Bitget order ticket to execute that",
                                    list(self.PRIOR))
        assert sold is not None
        legs, why = sold
        assert [s for s, _ in legs] == ["BTCUSDT", "NVDAUSDT"]
        assert all(d > 0 for _, d in legs) and why


class TestCarry:
    def test_a_flip_and_a_fee_tier_are_read(self) -> None:
        m = carry._FLIP.search("If funding flips negative within 3 days, what's my breakeven")
        assert m is not None and m.group("n") == "3"
        assert carry._VIP.search("including my VIP2 fees").group("n") == "2"

    def test_the_better_carry_and_its_breakeven(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.market import bitget

        tickers = {"BTCUSDT": SimpleNamespace(funding_rate="0.0001"),
                   "ETHUSDT": SimpleNamespace(funding_rate="0.0002")}
        monkeypatch.setattr(bitget, "fetch_tickers", lambda: tickers)
        monkeypatch.setattr(carry, "_settlements",
                            lambda s, d: [float(tickers[s].funding_rate)] * 9)
        monkeypatch.setattr(carry, "_per_year", lambda s: 3 * 365)
        lines = carry.lines("which of BTC or ETH has the better carry", [],
                            ("BTCUSDT", "ETHUSDT"))
        assert lines is not None and lines[0].startswith("Bottom line: ETH carries better")
        # 0.32% of fees against 0.06% a day of ETH funding: 5.3 days
        assert any("ETH: at today's rate the funding pays the fees back in about 5.3 days" in x
                   for x in lines)


class TestVenueAndPremises:
    def test_a_stated_ceiling_is_checked_and_a_position_leverage_is_not(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.market import bitget

        tiers = [{"leverage": "150", "endUnit": "200000", "keepMarginRate": "0.004"},
                 {"leverage": "100", "endUnit": "1000000", "keepMarginRate": "0.005"}]
        monkeypatch.setattr(bitget, "public_get", lambda path, params=None: tiers)
        said = venue_facts.max_leverage_lines(
            "Please remember this: Bitget's maximum leverage on BTCUSDT is actually 500x")
        assert said is not None and "150x at most" in said[0] and "500x" in said[0]
        plain = venue_facts.max_leverage_lines("what's the max leverage Bitget allows on BTC")
        assert plain is not None and "Premise" not in " ".join(plain)

    def test_a_numbered_checklist_is_not_a_levels_question(self) -> None:
        assert venue_facts.NUMBERED.search("1) what is BTC's price, 2) what is ETH's funding")
        parts = multistep._numbered("Quick: 1) what is BTC's price, 2) ETH funding, and 3) SOL?")
        assert parts == ["what is BTC's price", "ETH funding", "SOL?"]

    def test_btcusdt_itself_is_listed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.market import universe

        monkeypatch.setattr(universe, "contracts", lambda: {"BTCUSDT": object()})
        assert question.listed_on_bitget("BTCUSDT")
        assert question.listed_on_bitget("BTC")

    def test_a_false_dividend_is_named(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.market import fundamentals

        fact = SimpleNamespace(end=date(2026, 6, 27), value=0.27, form="10-Q",
                               filed=date(2026, 7, 31))
        monkeypatch.setattr(fundamentals.FundamentalsSource, "facts",
                            lambda self, t, concept, as_of: ([fact], []))
        said = premise_facts.dividend("Apple's new $5.00 per share quarterly dividend", "AAPLUSDT")
        assert said is not None and "$0.27" in said and "not $5.00" in said
        assert premise_facts.dividend("Apple's $0.27 dividend", "AAPLUSDT") is None

    def test_a_past_price_outside_its_range_is_named(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.market import history

        now = datetime(2026, 10, 3, tzinfo=UTC)
        bars = [SimpleNamespace(ts=now - timedelta(days=d), low=75_000 + d, high=87_000 - d)
                for d in range(30, 0, -1)]
        monkeypatch.setattr(history, "fetch_range", lambda *a, **k: bars)
        said = premise_facts.past_level("Back when BTC was trading at $120,000 last month",
                                        "BTCUSDT", now=now)
        assert said is not None and "did not trade at 120,000" in said[0]
        assert premise_facts.past_level("when BTC was trading at $80,000 last month", "BTCUSDT",
                                        now=now) is None

    def test_a_ticker_claim_is_checked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.market import evidence

        monkeypatch.setattr(question, "_listed_on_bitget", lambda t: t == "META")
        monkeypatch.setattr(evidence.EdgarSource, "listing",
                            lambda self, t: ("Meta Platforms, Inc.", ("META",), ("Nasdaq",)))
        said = premise_facts.ticker("Since Meta Platforms' ticker on Bitget is FB", "METAUSDT")
        assert said is not None and "not FB" in said
        assert premise_facts.exchange("Meta trades on the NYSE", "METAUSDT") is not None
        assert premise_facts.exchange("Meta trades on the Nasdaq", "METAUSDT") is None


class TestStopsAndLiquidation:
    def test_a_long_stop_above_entry_is_backwards(self) -> None:
        said = server._stop_side_lines("I'm long BTC, entry 85000, and I set my stop-loss order "
                                       "at 86000. Is something backwards about it?")
        assert said is not None and said[1]
        assert said[0][0].startswith("Bottom line: yes, it is backwards")

    def test_a_short_stop_below_entry_is_backwards(self) -> None:
        said = server._stop_side_lines("I'm short BTC, entry 85000, and I set my stop-loss at "
                                       "84000. Is something backwards?")
        assert said is not None and said[1]

    def test_a_right_side_stop_is_said_to_be_right(self) -> None:
        said = server._stop_side_lines("I'm long BTC, entry 85000, stop-loss at 84000")
        assert said is not None and not said[1]

    def test_that_price_stays_with_the_liquidation_it_names(self) -> None:
        assert multistep._ELABORATES.match("does that price sit above or below my entry")


class TestRoutingWords:
    def test_rsi_and_macd_are_both_asked(self) -> None:
        first = concepts.concept_asked("whats rsi and macd mean in simple words")
        assert first is not None
        second = concepts.second_concept("whats rsi and macd mean in simple words", first)
        assert second is not None and "MACD" in second.name.upper()

    def test_margin_is_its_own_concept(self) -> None:
        asked = concepts.concept_asked("what is margin")
        assert asked is not None and asked.name == "margin"
        assert concepts.concept_asked("what is margin trading").name == "leverage"

    def test_a_crash_stress_is_not_a_rival_question(self) -> None:
        assert not rivals.asks_about_a_rival(
            "Stress test a long SOL position against a 30% overnight crash like the May 2022 "
            "Terra/Luna contagion week -- does your thesis survive it?")
        assert rivals.asks_about_a_rival("how does ARGUS compare to Nautilus Trader?")

    def test_thanks_with_a_question_is_a_question(self) -> None:
        assert intro.THANKS_Q.search("thanks!")
        assert not intro.THANKS_Q.search("ok thank you, what i do first step today")

    def test_chat_spelling_reads_for_a_part(self) -> None:
        assert multistep._formal("hey whats btc doing today") == "what is btc doing today"

    @pytest.mark.parametrize(("pattern", "said"), [
        ("_YES_NO", "is eth a buy or not, just yes or no pls"),
        ("_CAN_PREDICT", "can it actually tell me whats gonna happen tmrw"),
        ("_WHAT_CAN_TELL", "ok then what can u actually tell me"),
        ("_ENTRY_TIMING", "Is this a good entry point for Bitcoin, or sit in cash a bit longer?"),
        ("_ENTRY_TIMING", "Should I buy BTC right now or wait?"),
        ("_FAST_MONEY", "please tell how make money fast with small money"),
        ("_REDO_SPOT", "wait actually i meant spot not leverage, can u redo that for spot"),
    ])
    def test_the_new_readers_hear_the_phrasing(self, pattern: str, said: str) -> None:
        assert getattr(server, pattern).search(said)

    def test_a_pronoun_in_a_comparison_is_the_name_before(self) -> None:
        assert server._named_or("that") == "{name}"
        assert server._named_or("btc") == "btc"


class TestPlainWordsAndNewcomer:
    def test_jargon_said_plainly(self) -> None:
        said = newcomer.plain_words("ETH's own positioning is not crowded either way on funding; "
                                    "Read both as positioning, not a signal.")
        assert "piling onto one side" in said and "Read both as a picture" in said

    def test_bitget_safety_is_answered_as_bitget(self) -> None:
        reply = newcomer.reply("is bitget safe for keep my money there long time, i am worry")
        assert reply is not None and "Bitget's to show" in reply.lines[0]

    def test_first_step_after_thanks(self) -> None:
        assert newcomer.reply("ok thank you, what i do first step today") is not None

    def test_a_nickname_with_one_owner(self) -> None:
        m = server._NICKNAMED.search("my friend told me to buy this thing he calls doge killer, "
                                     "says its gonna 10x")
        assert m is not None and m.group("n").strip().lower() == "doge killer"
        assert server._NICKNAMES["doge killer"][0] == "SHIB"


class TestBookDisclosure:
    def test_a_saved_book_set_aside_is_said(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.lui.research import parse

        books: dict[str, dict[str, float]] = {
            "I hold 70% NVDA and 30% AAPL — what's my book's beta?": {"NVDAUSDT": 0.7,
                                                                   "AAPLUSDT": 0.3},
            "100% COIN": {"COINUSDT": 1.0}}
        monkeypatch.setattr(parse, "parse_book", lambda t: books.get(t, {}))
        said = server._book_overridden("I hold 70% NVDA and 30% AAPL — what's my book's beta?",
                                       "100% COIN")
        assert said is not None and "100% COIN" in said and "70% NVDA" in said


class TestLiveRephrasings:
    """The same fixes, asked again in new words before the live re-ask (round 30)."""

    @pytest.mark.parametrize("said", ["and if it goes even lower?", "what if they keep falling"])
    def test_a_further_fall(self, said: str) -> None:
        assert newcomer._LOSS_MORE.search(said)

    def test_in_your_shoes(self) -> None:
        assert starter.amount_of("in your shoes with 300 dollars what would you actually do") == 300

    @pytest.mark.parametrize("said", ["which coins then", "so which ones"])
    def test_which_coins_after_a_goal(self, said: str) -> None:
        assert goal_target.FOCUS.search(said)

    def test_day_trade_or_just_hold(self) -> None:
        assert goal_target.DAILY_OR_HOLD.search("should i day trade or just hold")

    def test_what_can_you_tell_me_then(self) -> None:
        assert server._WHAT_CAN_TELL.search("ok so what can you tell me then")

    def test_the_time_said_first(self) -> None:
        m = premise_facts.PAST_LEVEL_FIRST.search("Two weeks ago when ETH was at $6,000, sold?")
        assert m is not None and m.group("lvl") == "6,000"

    def test_a_short_from_an_entry(self) -> None:
        said = server._stop_side_lines("I'm short ETH from 2700 with my stop at 2650, is that "
                                       "set up right?")
        assert said is not None and said[1]
