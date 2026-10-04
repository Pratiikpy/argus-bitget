"""Round 36's first-time-user audit, run offline: each new or extended route is checked on the
exact phrasing that exposed it in `Activity/audits/round36_newcomer.md`, and on a near-miss
beside it so the fix does not steal a question it was never meant to answer."""

from __future__ import annotations

from argus.lui import newcomer


class TestHowMuchToStartWith:
    """Conv A1: "how much money should i actually start with" put a filler word between "i"
    and the verb, which the gap-free match missed, and fell through to a ticker misread."""

    def test_actually_between_i_and_start_with_still_matches(self) -> None:
        reply = newcomer.reply(
            "hii so im totally new to this, how much money should i actually start with "
            "like is 500rs too less or what", named=False)
        assert reply is not None
        assert reply.reask == "I have $1,000, what should I do"

    def test_plain_form_without_filler_still_matches(self) -> None:
        reply = newcomer.reply("how much should i invest", named=False)
        assert reply is not None
        assert reply.reask == "I have $1,000, what should I do"

    def test_cost_to_trade_is_unaffected(self) -> None:
        # near-miss: a measured cost question, not a bare "how much to start with"
        assert newcomer.reply("how much does it cost to trade BTC", named=False) is None

    def test_how_much_can_i_lose_is_unaffected(self) -> None:
        # near-miss: a risk question, answered by a different engine entirely
        assert newcomer.reply("how much can I lose on BTC this week", named=False) is None


class TestBareDCA:
    """Conv A2, J2: "whats DCA i keep seeing it everywhere" has no pattern at all for a bare
    definition — only the lump-sum-comparison form is covered, and DCA read as an unknown
    ticker. Reproduced identically in Hinglish."""

    def test_bare_whats_dca(self) -> None:
        reply = newcomer.reply("whats DCA i keep seeing it everywhere", named=False)
        assert reply is not None
        assert "dollar-cost averaging" in reply.lines[0]
        assert "not a Bitget product or a coin" in " ".join(reply.lines)

    def test_hinglish_dca_wagera(self) -> None:
        reply = newcomer.reply(
            "accha and agar main thoda thoda paisa daalu time ke sath to usko kya bolte "
            "hain, DCA wagera?", named=False)
        assert reply is not None
        assert "dollar-cost averaging" in reply.lines[0]

    def test_dca_lump_sum_comparison_is_unaffected(self) -> None:
        # near-miss: the existing comparison answer must still win, not the new bare one
        reply = newcomer.reply("is DCA better than lump sum investing", named=False)
        assert reply is not None
        assert "lump sum" in reply.lines[0]
        assert "dollar-cost averaging (DCA) spreads" in " ".join(reply.lines)

    def test_how_is_that_different_from_lump_sum_is_unaffected(self) -> None:
        # near-miss: the existing follow-up form of the same comparison
        reply = newcomer.reply(
            "how is that different than just buying once with all my money at the start",
            named=False)
        assert reply is not None
        assert "lump sum" in reply.lines[0]


class TestBareETF:
    """Conv B1: "someone told me to just buy an etf instead of coins, what even is that"
    named the term earlier in the sentence, not right next to "what's", and got an
    incoherent template collision instead of a definition."""

    def test_referential_what_even_is_that(self) -> None:
        reply = newcomer.reply(
            "someone told me to just buy an etf instead of coins, what even is that",
            named=False)
        assert reply is not None
        assert "ETF (exchange-traded fund)" in reply.lines[0]

    def test_direct_form_is_unaffected(self) -> None:
        # near-miss: the original, already-working direct phrasing
        reply = newcomer.reply("whats an etf, is bitcoin an etf", named=False)
        assert reply is not None
        assert "ETF (exchange-traded fund)" in " ".join(reply.lines)

    def test_unrelated_what_even_is_that_is_unaffected(self) -> None:
        # near-miss: "what even is that" with no ETF mentioned must not match
        assert newcomer.reply("i saw this weird coin on twitter, what even is that",
                              named=False) is None


class TestBarePerpetual:
    """Conv C1: no bare "define perpetual" pattern existed at all — only the console's own
    suggested phrasing ("what is a perp") worked; a real person's own wording did not."""

    def test_bare_perpetual_definition(self) -> None:
        reply = newcomer.reply(
            "whats a perpetual, ppl keep saying perp this perp that", named=False)
        assert reply is not None
        assert "no expiry date" in reply.lines[0]
        body = " ".join(reply.lines)
        assert "funding" in body
        assert "liquidat" in body

    def test_what_is_a_perpetual_contract_variant(self) -> None:
        # a second, fully-spelled phrasing of the same bare definition
        reply = newcomer.reply("what is a perpetual contract", named=False)
        assert reply is not None
        assert "no expiry date" in reply.lines[0]

    def test_measured_funding_rate_question_is_unaffected(self) -> None:
        # near-miss: a measured, named-contract question must go to the market engine
        assert newcomer.reply("what is BTC's perpetual funding rate right now",
                              named=False) is None

    def test_cost_to_trade_a_perpetual_is_unaffected(self) -> None:
        # near-miss: a cost question, not a bare definition
        assert newcomer.reply("how much does it cost to trade a BTC perpetual",
                              named=False) is None


class TestBitgetOutageWording:
    """Conv D1, O1: the dedicated custody explainer exists and is correct, but its trigger
    missed a reversed effect-question-first order, a missing "crashes" verb, and a
    "where does my money go" phrasing with no "happens" at all."""

    def test_what_happens_to_my_money_reversed_order_with_crashes(self) -> None:
        reply = newcomer.reply(
            "ok dumb question but what happens to my money if bitget the app goes down or "
            "crashes", named=False)
        assert reply is not None
        body = " ".join(reply.lines)
        assert "proof-of-reserves" in body or "queue behind every other depositor" in body

    def test_where_does_my_money_go_if_bitget_shuts_down(self) -> None:
        reply = newcomer.reply(
            "if bitget shuts down one day where does my money actually go, is it stored on "
            "my phone or on their server", named=False)
        assert reply is not None
        body = " ".join(reply.lines)
        assert "queue behind every other depositor" in body or "proof-of-reserves" in body

    def test_should_i_move_it_to_a_wallet_follow_up(self) -> None:
        reply = newcomer.reply("so should i move it to a wallet instead then", named=False)
        assert reply is not None
        assert "wallet" in " ".join(reply.lines).lower()

    def test_price_crash_on_a_held_coin_is_unaffected(self) -> None:
        # near-miss: a price-crash question about a held asset, not a Bitget-outage question
        assert newcomer.reply(
            "if BTC goes down a lot what happens to my money, since I use bitget",
            named=False) is None

    def test_perpetual_price_crash_is_unaffected(self) -> None:
        # near-miss: a contract price move, not Bitget itself failing
        assert newcomer.reply(
            "bitget's btc perpetual crashed 10% today what happens to my position",
            named=False) is None


class TestBuyMeImperative:
    """Conv I2: "ok just buy me 100 dollars of bitcoin then" is a blunt imperative with no
    modal verb at all, and it was silently filed as a remembered position size with no
    refusal and no statement that no purchase happened — the most serious finding of the
    round."""

    def test_buy_me_dollar_amount(self) -> None:
        reply = newcomer.reply("ok just buy me 100 dollars of bitcoin then", named=False)
        assert reply is not None
        assert reply.lines[0].startswith("Bottom line: no")
        assert "never trades for anyone" in reply.lines[0]

    def test_buy_me_dollar_sign(self) -> None:
        reply = newcomer.reply("just buy me $500 of eth", named=False)
        assert reply is not None
        assert "never trades for anyone" in reply.lines[0]

    def test_sell_my_x_for_me(self) -> None:
        reply = newcomer.reply("sell my eth for me right now", named=False)
        assert reply is not None
        assert "never trades for anyone" in reply.lines[0]

    def test_go_buy_some_x_for_me(self) -> None:
        reply = newcomer.reply("just go buy some bitcoin for me", named=False)
        assert reply is not None
        assert "never trades for anyone" in reply.lines[0]

    def test_modal_form_is_unaffected(self) -> None:
        # near-miss: the original, already-working modal-question form
        reply = newcomer.reply("can u just buy it for me", named=False)
        assert reply is not None
        assert "never trades for anyone" in reply.lines[0]

    def test_get_me_the_price_is_unaffected(self) -> None:
        # near-miss: "get me" followed by information, not an amount or an object to trade
        assert newcomer.reply("get me the price of BTC", named=False) is None

    def test_whats_best_for_me_is_unaffected(self) -> None:
        # near-miss: a personalised-advice question, not an imperative to execute a trade
        assert newcomer.reply("what's the best strategy for me", named=False) is None

    def test_how_do_i_buy_bitcoin_is_unaffected(self) -> None:
        # near-miss: a how-to question with no "me" object at all
        reply = newcomer.reply("how do i buy bitcoin", named=False)
        assert reply is not None
        assert "never trades for anyone" not in reply.lines[0]


class TestGoodTimeCompressed:
    """Conv L1: "wassup is now good time buy" dropped both the article and the infinitive,
    and the strict match missed it in both the degraded and the LLM-available modes."""

    def test_compressed_good_time_to_buy(self) -> None:
        reply = newcomer.reply("wassup is now good time buy", named=False)
        assert reply is not None
        assert reply.reask == "has {name} been here before"
        assert reply.lead is not None and "no call" in reply.lead

    def test_full_form_is_unaffected(self) -> None:
        # near-miss: the original, already-working full phrasing
        reply = newcomer.reply("is now a good time to buy", named=False)
        assert reply is not None
        assert reply.reask == "has {name} been here before"

    def test_named_contract_is_unaffected(self) -> None:
        # near-miss: a named contract's own engine answers this, not the generic reply
        assert newcomer.reply("is this a good time to buy TSLA", named=True) is None

    def test_price_question_is_unaffected(self) -> None:
        # near-miss: a plain price question must not be read as a timing question
        assert newcomer.reply("what is the price of BTC right now", named=False) is None


class TestJustTellMeYesOrNo:
    """Conv L2: "idk man just tell me yes or no" pushed for a binary call and got nothing —
    not even a restatement of why this console will not give one."""

    def test_just_tell_me_yes_or_no(self) -> None:
        reply = newcomer.reply("idk man just tell me yes or no", named=False)
        assert reply is not None
        assert reply.lines[0].startswith("Bottom line: no yes or no here")

    def test_give_me_a_yes_or_no(self) -> None:
        reply = newcomer.reply("just give me a yes or no answer please", named=False)
        assert reply is not None
        assert reply.lines[0].startswith("Bottom line: no yes or no here")

    def test_bear_market_question_is_unaffected(self) -> None:
        # near-miss: an unrelated measured/analytic question
        assert newcomer.reply("how did BTC do in the 2022 bear market?", named=False) is None

    def test_price_question_is_unaffected(self) -> None:
        # near-miss: a plain price question ending differently
        assert newcomer.reply("what is the price of BTC right now", named=False) is None
