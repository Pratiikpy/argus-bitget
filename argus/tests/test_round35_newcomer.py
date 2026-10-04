"""Round 35's first-time-user audit, run offline: each new or extended route is checked on the
exact phrasing that exposed it in `Activity/audits/round35_newcomer.md`, and on a near-miss
beside it so the fix does not steal a question it was never meant to answer."""

from __future__ import annotations

import pytest

from argus.lui import newcomer


class TestEmotionalLoss:
    """Conv F2: "i lost half my savings on a trade last week i feel sick rn ngl" got the
    generic "I did not recognise that question" template — the most serious miss in the run."""

    def test_emotional_loss_message_gets_empathy_first(self) -> None:
        reply = newcomer.reply(
            "i lost half my savings on a trade last week i feel sick rn ngl", named=False)
        assert reply is not None
        assert reply.lines[0].startswith("Bottom line: that is a real loss")
        body = " ".join(reply.lines)
        assert "win it back" in body
        assert "talk to someone" in body

    def test_asking_to_get_it_back_fast_is_unaffected(self) -> None:
        # near-miss: round 11's revenge-trading answer, not the round-35 emotional one
        reply = newcomer.reply("is there any way to get it back fast", named=False)
        assert reply is not None
        assert "trading to win back a loss" in reply.lines[0]


class TestBearMarket:
    """Conv D1: "wats a bear market and how long do they usually last" was refused — a
    textbook glossary question."""

    def test_bear_market_definition_and_length(self) -> None:
        reply = newcomer.reply(
            "wats a bear market and how long do they usually last", named=False)
        assert reply is not None
        assert reply.lines[0].startswith("Bottom line: a bear market is a sustained fall")
        body = " ".join(reply.lines)
        assert "20%" in body
        assert "a year or more" in body

    def test_a_plain_recession_question_is_unaffected(self) -> None:
        reply = newcomer.reply(
            "what happens to crypto if theres a recession does it crash too", named=False)
        assert reply is not None
        assert reply.lines[0].startswith("Bottom line: crypto has mostly moved with the "
                                          "stock market")


class TestStopLossTakeProfit:
    """Conv A1, A4: the stop-loss/take-profit glossary pair, and the overnight-gap follow-up
    that was answered with an unrelated hedge-sizing pitch."""

    def test_stop_loss_and_take_profit_in_simple_words(self) -> None:
        reply = newcomer.reply(
            "wat is a stop loss and take profit in simple words", named=False)
        assert reply is not None
        assert reply.lines[0].startswith("Bottom line: a stop-loss automatically closes")
        body = " ".join(reply.lines)
        assert "take-profit" in body
        assert "Bitget's help centre" in body

    def test_stop_loss_overnight_gap_risk(self) -> None:
        reply = newcomer.reply(
            "what if price gaps past my stop loss overnight while im asleep does it still "
            "protect me", named=False)
        assert reply is not None
        assert reply.lines[0].startswith("Bottom line: not always")
        assert "fill lower than you set it" in " ".join(reply.lines)

    def test_bare_stop_loss_question_is_unaffected(self) -> None:
        # near-miss: "what is a stop loss" alone (no take-profit) is concepts.py's to answer
        assert newcomer.reply("what is a stop loss", named=True) is None


class TestPriceMismatch:
    """Conv C1-C2: "google" was read as the stock ticker GOOGL, and the follow-up "which one
    is the real price then" was read as an unknown ticker lookup. Both must answer even when a
    ticker is also detected in the text (named=True)."""

    def test_btc_price_vs_google_named_true(self) -> None:
        reply = newcomer.reply(
            "why is the btc price on bitget diff from wat google shows me rn", named=True)
        assert reply is not None
        assert reply.lines[0].startswith("Bottom line: Bitget and Google are not quoting "
                                          "the same thing")
        assert "GOOGL" not in " ".join(reply.lines)

    def test_which_one_is_the_real_price_then_named_true(self) -> None:
        reply = newcomer.reply("ok so which one is the real price then", named=True)
        assert reply is not None
        assert reply.lines[0].startswith("Bottom line: there is no single \"real\" price")

    def test_a_real_googl_stock_question_is_unaffected(self) -> None:
        # near-miss: an actual question about Google's own stock, not the price-source mix-up
        assert newcomer.reply("whats google stock doing today", named=True) is None


class TestReading24hChange:
    """Conv E3: "how do i even read the 24h change number on the app" was read as an unknown
    ticker lookup — a basic UI-literacy question, never answered."""

    def test_how_do_i_read_24h_change(self) -> None:
        reply = newcomer.reply(
            "how do i even read the 24h change number on the app", named=True)
        assert reply is not None
        assert reply.lines[0].startswith("Bottom line: the 24h change is how much the price "
                                          "has moved")


class TestBareFees:
    """Conv J2: "fees?" — the same one-word format as "ath?" one turn later, refused."""

    def test_bare_fees_question(self) -> None:
        reply = newcomer.reply("fees?", named=False)
        assert reply is not None
        assert reply.lines[0].startswith("Bottom line: Bitget's standard spot trading fee "
                                          "is 0.10%")

    def test_a_longer_fee_sentence_is_unaffected(self) -> None:
        reply = newcomer.reply("wat r the fees here", named=False)
        assert reply is not None
        assert "this console is free and takes no cut" in reply.lines[0]


class TestHinglishLeverage:
    """Conv I1: "bhai ye leverage wala cheez kitna risky hai seriously batao" got no
    translation line at all and fell through to the generic refusal, matched here directly on
    the Hinglish with no translation needed."""

    def test_hinglish_leverage_risk_question(self) -> None:
        reply = newcomer.reply(
            "bhai ye leverage wala cheez kitna risky hai seriously batao", named=False)
        assert reply is not None
        assert reply.lines[0].startswith("Bottom line: leverage is risky")

    def test_english_leverage_safety_is_unaffected(self) -> None:
        # near-miss: the already-working plain-English form of the same concept (conv F1)
        reply = newcomer.reply("whats a safe leverage to use as a beginner", named=False)
        assert reply is not None and "no leverage is safe" in reply.lines[0]


class TestFriendAllIn:
    """Conv K1: the run-on "...should i just trust him and go all in" latched onto only the
    slang word "moon" and dropped the far more important ask."""

    def test_friend_tip_all_in_run_on(self) -> None:
        reply = newcomer.reply(
            "ok so basically i downloaded the app yesterday and my friend said i should put "
            "like all my savings into one coin because its gonna moon soon and i dont really "
            "know what moon means but he seems confident and also he said something about "
            "leverage making it faster should i just trust him and go all in", named=False)
        assert reply is not None
        assert reply.lines[0].startswith("Bottom line: no — do not put all your savings "
                                          "into one coin")

    def test_a_plain_social_media_hype_coin_is_unaffected(self) -> None:
        reply = newcomer.reply(
            "i saw this coin called PEPE on my tiktok fyp, everyone was saying its gonna "
            "blow up", named=False)
        assert reply is not None
        assert "being sold to" in reply.lines[0]


class TestEarnVsFutures:
    """Conv G2: "whats the diff between the Earn tab and the Futures tab on bitget" was
    refused — a basic product-navigation question."""

    def test_earn_vs_futures_tabs(self) -> None:
        reply = newcomer.reply(
            "whats the diff between the Earn tab and the Futures tab on bitget", named=False)
        assert reply is not None
        assert reply.lines[0].startswith("Bottom line: Earn is for putting coins you "
                                          "already hold to work")

    def test_a_plain_earn_question_is_unaffected(self) -> None:
        reply = newcomer.reply("whats this bitget earn thing, is it safe", named=False)
        assert reply is not None and "Earn and staking pay you" in reply.lines[0]


class TestBitcoinForGrandma:
    """Conv G1: a good plain definition got live bid/ask spread and funding-rate figures
    tacked on — exactly the jargon a grandma-level explanation should omit."""

    def test_explain_bitcoin_to_grandma_has_no_jargon(self) -> None:
        reply = newcomer.reply(
            "can u explain bitcoin to me like im explaining to my grandma", named=False)
        assert reply is not None
        assert reply.lines[0].startswith("Bottom line: Bitcoin is digital money")
        body = " ".join(reply.lines)
        for jargon in ("bid", "ask", "spread", "funding"):
            assert jargon not in body.lower()


class TestFundingRateForMe:
    """Conv B1 passed on "wats funding rate mean like if im holding a long position"; the
    plainer "what does funding rate mean for me" form is asked just as often."""

    def test_funding_rate_what_does_it_mean_for_me(self) -> None:
        reply = newcomer.reply("funding rate, what does it mean for me", named=False)
        assert reply is not None
        assert reply.lines[0].startswith("Bottom line: the funding rate is a small payment")


class TestProfitTakingOnAPump:
    """Conv E1: "my coin pumped 50% today should i take profit now" silently substituted BTC
    for the user's own coin and never disclosed it."""

    def test_coin_pumped_50_percent_take_profit(self) -> None:
        reply = newcomer.reply(
            "my coin pumped 50% today should i take profit now", named=False)
        assert reply is not None
        assert reply.lines[0].startswith("Bottom line: no call here on whether to sell")
        assert "taking some profit, not all" in " ".join(reply.lines)


class TestSellRegret:
    """Conv E2: "but what if it keeps going up after i sell ill feel so dumb" was read as an
    unknown ticker symbol, dropping the emotional profit-taking framing entirely."""

    def test_what_if_it_keeps_going_up_after_i_sell(self) -> None:
        reply = newcomer.reply(
            "but what if it keeps going up after i sell ill feel so dumb", named=True)
        assert reply is not None
        assert reply.lines[0].startswith("Bottom line: it might keep going up after you sell")


class TestBitgetVsWallet:
    """Conv H1 (German, translated correctly): "What is the difference between Bitget and a
    wallet?" still got the generic refusal underneath the translation. Matched here directly
    on the English form, which the translation layer produces."""

    def test_difference_between_bitget_and_a_wallet(self) -> None:
        reply = newcomer.reply(
            "what is the difference between bitget and a wallet", named=False)
        assert reply is not None
        assert reply.lines[0].startswith("Bottom line: Bitget is an exchange")

    def test_money_safe_on_bitget_is_unaffected(self) -> None:
        # near-miss: round 11's custody answer, not the exchange-vs-wallet one
        reply = newcomer.reply("is my money safe on bitget", named=False)
        assert reply is not None
        assert "this console never holds, moves or touches money" in reply.lines[0]


@pytest.mark.parametrize("said", [
    "is bitget safe", "what is a stop loss", "should i buy BTC right now",
    "whats a memecoin", "what is a seed phrase", "how do i withdraw",
    "is 10x leverage ok for a small account", "should i sell my crypto now",
])
def test_round35_patterns_do_not_steal_existing_questions(said: str) -> None:
    reply = newcomer.reply(said, named=True)
    if said == "is bitget safe":
        assert reply is not None and "Bitget's to show" in reply.lines[0]
    elif said == "whats a memecoin":
        assert reply is not None and "coin built on a joke" in reply.lines[0]
    elif said == "what is a seed phrase":
        assert reply is not None and "master key to a self-custody wallet" in reply.lines[0]
    elif said == "how do i withdraw":
        assert reply is not None and "withdrawals happen on Bitget" in reply.lines[0]
    else:
        assert reply is None
