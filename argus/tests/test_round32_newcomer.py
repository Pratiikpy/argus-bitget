"""Round 32's first-time-user audit, run offline: each new or extended route is checked on the
exact phrasing that exposed it in `Activity/audits/round32_newcomer.md`, and on a near-miss
beside it so the fix does not steal a question it was never meant to answer."""

from __future__ import annotations

import pytest

from argus.lui import newcomer


class TestScamTopics:
    """Q1-Q3: a "trading bot" promising a fixed daily return, and its two follow-ups. Q26: a
    "send crypto first, I'll double it" DM."""

    def test_a_fixed_daily_return_is_compounded_out_loud(self) -> None:
        reply = newcomer.reply(
            "my friend said i should put my savings into a trading bot he showed me, it "
            "promises 2% daily returns, should i do it", named=False)
        assert reply is not None
        assert "1,377x" in reply.lines[0] and "no —" in reply.lines[0]

    def test_a_fixed_weekly_return_uses_52_weeks(self) -> None:
        reply = newcomer.reply("this group guarantees 3% a week, is that legit", named=False)
        assert reply is not None
        assert "a week" in reply.lines[0]

    def test_screenshots_of_profits_are_not_proof(self) -> None:
        reply = newcomer.reply(
            "he said its been working for him for 2 months and he showed me screenshots of "
            "profits", named=False)
        assert reply is not None and "not proof" in reply.lines[0]

    def test_how_to_check_something_is_legit(self) -> None:
        reply = newcomer.reply(
            "ok so how do i actually check if something like that is legit before i put money "
            "in", named=False)
        assert reply is not None and "four checks" in reply.lines[0]

    def test_send_crypto_first_to_double_it_is_a_scam(self) -> None:
        reply = newcomer.reply(
            "some guy dmed me on instagram saying he can double my crypto in 24 hrs if i send "
            "him some first, is this real", named=False)
        assert reply is not None
        assert reply.lines[0].startswith("Bottom line: no") and "always a scam" in reply.lines[0]

    def test_a_plain_buy_question_is_not_taken_by_the_scam_entries(self) -> None:
        assert newcomer.reply("should i buy BTC right now", named=True) is None
        assert newcomer.reply("should i buy BTC right now", named=False) is None

    def test_a_dollar_stake_is_not_staking_or_a_scam(self) -> None:
        reply = newcomer.reply("my stake is $500, should I add NVDA", named=True)
        assert reply is None or (
            "staking" not in reply.lines[0] and "scam" not in reply.lines[0])


class TestCandlestickLiteracy:
    """Q4-Q7: what a candle is, green vs red, wicks, and whether a pattern predicts a rise."""

    def test_explain_a_candlestick_chart(self) -> None:
        reply = newcomer.reply(
            "can someone explain how to read a candlestick chart like im 5 years old",
            named=False)
        assert reply is not None and "one candle is the price over one time period" in (
            reply.lines[0])

    def test_green_and_red(self) -> None:
        reply = newcomer.reply("whats the green and red mean on it", named=False)
        assert reply is not None and "green means the price closed higher" in reply.lines[0]

    def test_a_wick(self) -> None:
        reply = newcomer.reply(
            "whats a wick then, i see lines sticking out the top and bottom", named=False)
        assert reply is not None and "a wick (also called a shadow)" in reply.lines[0]

    def test_no_pattern_reliably_predicts_a_rise(self) -> None:
        reply = newcomer.reply("ok and is there a pattern that means its about to go up",
                               named=False)
        assert reply is not None
        assert reply.lines[0].startswith("Bottom line: no")
        assert "no candlestick pattern reliably means" in reply.lines[0]

    def test_a_coin_ticker_is_unaffected(self) -> None:
        # "pattern" must not be swallowed by the candlestick entry when it is not about charts
        assert newcomer.reply("what is PATTERN doing today", named=True) is None


class TestTaxLossOffset:
    """Q11: "will this loss show up anywhere i can use it later, like against profits" is the
    same tax-loss-carryforward question as the existing TAX entries, asked without the word
    "tax"."""

    def test_the_module_level_tax_pattern_catches_it(self) -> None:
        assert newcomer.TAX.search(
            "will this loss show up anywhere i can use it later, like against profits")
        # the ordinary phrasing (round 31) still matches too
        assert newcomer.TAX.search("i made 500 on eth this year, do i pay taxes on that")

    def test_the_loss_line_leads_and_names_carryforward(self) -> None:
        reply = newcomer.reply(
            "will this loss show up anywhere i can use it later, like against profits",
            named=True)
        assert reply is not None
        assert reply.lines[0].startswith("Bottom line: If you lost money, it still counts")
        assert "profits" in reply.lines[0] and "carry forward" in reply.lines[0]

    def test_the_plain_tax_question_still_leads_with_its_own_line(self) -> None:
        reply = newcomer.reply("do i owe taxes on that or only if i cash out to my bank",
                               named=True)
        assert reply is not None and "not tax advice" in reply.lines[0]


class TestMonthlyBudget:
    """Q12-Q13: a student's monthly allowance, and a smaller amount asked as a follow-up."""

    def test_the_allowance_question(self) -> None:
        reply = newcomer.reply(
            "im a college student and i get 5000 rupees allowance a month, how much of that "
            "should i put into crypto", named=False)
        assert reply is not None
        assert "no fixed percentage is right for everyone" in reply.lines[0]
        assert "INR" not in " ".join(reply.lines)

    def test_the_smaller_follow_up(self) -> None:
        reply = newcomer.reply(
            "what if i only do 500 rupees a month instead, is that even worth it", named=False)
        assert reply is not None
        assert "no fixed percentage is right for everyone" in reply.lines[0]

    def test_fees_are_mentioned_for_a_tiny_amount(self) -> None:
        reply = newcomer.reply(
            "what if i only do 500 rupees a month instead, is that even worth it", named=False)
        assert reply is not None and "0.10%" in " ".join(reply.lines)

    def test_a_dollar_stake_is_not_a_monthly_allowance(self) -> None:
        reply = newcomer.reply("my stake is $500, should I add NVDA", named=True)
        assert reply is None or "allowance" not in reply.lines[0]


class TestBeginnerApp:
    """Q16: "whats a good app for someone who has never traded before" was misread as a
    user-holdings lookup because "someone who has" matched the username slot of a different
    entry."""

    def test_the_question_gets_its_own_answer(self) -> None:
        reply = newcomer.reply("whats a good app for someone who has never traded before",
                               named=False)
        assert reply is not None and "no app named here" in reply.lines[0]
        body = " ".join(reply.lines)
        assert "licensed" in body and "proof of reserves" in body and "2FA" in body

    def test_a_real_users_holdings_still_get_the_privacy_answer(self) -> None:
        # the round-31 case this fix must not break: a real username before the verb
        reply = newcomer.reply("Show me what positions user 'satoshi_trader99' holds on Bitget")
        assert reply is not None and "no one's account is visible here" in reply.lines[0]

    def test_other_relative_pronouns_are_also_excluded(self) -> None:
        pattern = next(p for p, a in newcomer._PLAIN
                       if "no one's account is visible here" in a[0])
        assert pattern.search("someone who has never traded before") is None
        assert pattern.search("someone that has never used crypto before") is None
        assert pattern.search("user 'satoshi_trader99' holds") is not None


class TestBitgetCustody:
    """Q22: "if bitget the company goes down or gets hacked what happens to my coins" broke on
    this phrasing even though "is bitget safe" (asked minutes earlier) worked."""

    def test_the_question_gets_the_custody_answer(self) -> None:
        reply = newcomer.reply(
            "if bitget the company goes down or gets hacked what happens to my coins",
            named=False)
        # what happens if it fails is the question, so that line leads (round 32 re-run)
        assert reply is not None and reply.lines[0].startswith("Bottom line: If it failed")
        body = " ".join(reply.lines)
        assert "queue behind every other depositor" in body
        assert "wallet only you control" in body and "never held them" in body

    def test_the_plain_safety_question_is_unaffected(self) -> None:
        reply = newcomer.reply("is bitget safe", named=False)
        assert reply is not None and "Bitget's to show" in reply.lines[0]

    def test_a_hacked_altcoin_is_not_the_bitget_answer(self) -> None:
        reply = newcomer.reply("if PEPE gets hacked what happens to the price", named=True)
        assert reply is None or "Bitget's to show" not in reply.lines[0]


class TestCashInBank:
    """Q25: after gold vs bitcoin, "what about just keeping my cash in the bank instead of
    either" was answered with the gold-vs-bitcoin comparison again."""

    def test_the_question_gets_its_own_answer(self) -> None:
        reply = newcomer.reply("ok what about just keeping my cash in the bank instead of "
                               "either", named=False)
        assert reply is not None
        assert "cash in a bank keeps its number the same" in reply.lines[0]
        body = " ".join(reply.lines)
        assert "insured up to a limit" in body and "inflation" in body

    def test_the_original_gold_vs_bitcoin_question_is_unaffected(self) -> None:
        pattern = next(p for p, a in newcomer._PLAIN
                       if "cash in a bank keeps its number the same" in a[0])
        assert pattern.search("should i buy gold or bitcoin, my dad keeps saying gold is "
                              "safer") is None


class TestHinglishWithdrawal:
    """Q27: "mera paisa bitget se bank account me kaise nikalu, withdraw kaise karu" was
    refused outright."""

    def test_the_hinglish_phrasing_routes_to_withdrawing(self) -> None:
        reply = newcomer.reply(
            "mera paisa bitget se bank account me kaise nikalu, withdraw kaise karu",
            named=False)
        assert reply is not None and "withdrawals happen on Bitget" in reply.lines[0]

    def test_the_english_phrasing_still_works(self) -> None:
        reply = newcomer.reply("how do i actually get my money out if i wanna stop", named=False)
        assert reply is not None and "withdrawals happen on Bitget" in reply.lines[0]


class TestFeesChargedTwice:
    """Q21: "so if i buy and then sell the same day do i get charged fees twice" never got a
    direct yes."""

    def test_a_direct_yes_leads(self) -> None:
        reply = newcomer.reply(
            "so if i buy and then sell the same day do i get charged fees twice", named=False)
        assert reply is not None
        assert reply.lines[0].startswith("Bottom line: yes")
        assert "fee on each side" in reply.lines[0]

    def test_the_round_trip_math_is_shown(self) -> None:
        reply = newcomer.reply(
            "so if i buy and then sell the same day do i get charged fees twice", named=False)
        assert reply is not None
        body = " ".join(reply.lines)
        assert "0.10%" in body and "0.20%" in body


class TestTooLateToBuy:
    """Q15: "is it too late to buy bitcoin now, feels like its already so high" should be read
    the same way as the existing "is now a good time to buy" no-call question."""

    def test_the_pattern_matches_the_phrasing(self) -> None:
        assert newcomer._GOOD_TIME.search("is it too late to buy bitcoin now")
        assert newcomer._GOOD_TIME.search("is it too late to invest now")

    def test_the_nameless_form_gets_the_no_call_reask(self) -> None:
        reply = newcomer.reply(
            "is it too late to buy now, feels like its already so high", named=False)
        assert reply is not None
        assert reply.reask == "has {name} been here before"
        assert reply.lead is not None and "no call" in reply.lead

    def test_a_named_coin_still_defers_to_its_own_engine(self) -> None:
        # by design (see `_reply_one`'s docstring): a named contract is that engine's to answer
        assert newcomer.reply(
            "is it too late to buy bitcoin now, feels like its already so high", named=True
        ) is None

    def test_is_it_too_late_alone_does_not_match(self) -> None:
        assert newcomer._GOOD_TIME.search("is it too late") is None


class TestJargonAddedForSkewAndRSquared:
    """The MAJOR complaint on Q15 ("unexplained skew, kurtosis, R-squared"): kurtosis and beta
    were already translated by `plain_words`; skew and R-squared were not."""

    def test_skew_is_translated(self) -> None:
        assert newcomer.plain_words("skew +0.66, excess kurtosis 10.3") == (
            "Whether its big moves lean up or down +0.66, excess how often it makes very big "
            "moves 10.3")

    def test_r_squared_is_translated(self) -> None:
        assert "share of its moves the benchmark explains" in newcomer.plain_words(
            "QQQ explains 10% of its moves (R² 0.10)").lower()


@pytest.mark.parametrize("said", [
    "is bitget safe", "what is a stop loss", "should i buy BTC right now",
    "my stake is $500, should I add NVDA",
])
def test_existing_questions_are_unaffected(said: str) -> None:
    # a broad sweep: none of round 32's new patterns should fire on these at all
    reply = newcomer.reply(said, named=True)
    if said == "is bitget safe":
        assert reply is not None and "Bitget's to show" in reply.lines[0]
    else:
        assert reply is None
