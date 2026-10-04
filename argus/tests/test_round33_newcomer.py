"""Round 33's first-time-user audit, run offline: each new or extended route is checked on the
exact phrasing that exposed it in `Activity/audits/round33_newcomer.md`, and on a near-miss
beside it so the fix does not steal a question it was never meant to answer."""

from __future__ import annotations

import pytest

from argus.lui import newcomer


class TestTwoFactorAndSeedPhrase:
    """Q1-Q4: what a wallet is, what a seed phrase is, losing 2FA, and forgetting a seed
    phrase — all four were either refused or answered a completely different question."""

    def test_what_is_a_crypto_wallet(self) -> None:
        reply = newcomer.reply(
            "what even is a crypto wallet lol do i need one to use bitget", named=False)
        assert reply is not None
        assert reply.lines[0].startswith("Bottom line: no")
        assert "not required to use Bitget" in reply.lines[0]

    def test_what_is_a_seed_phrase(self) -> None:
        reply = newcomer.reply(
            "ok and whats a seed phrase, why is everyone so scared about it", named=False)
        assert reply is not None and "master key to a self-custody wallet" in reply.lines[0]

    def test_lost_2fa_goes_to_account_recovery_not_revenge_trading(self) -> None:
        reply = newcomer.reply(
            "omg i think i lost access to my 2fa app how do i get back into my bitget "
            "account", named=False)
        assert reply is not None
        assert "Bitget's own account-recovery process" in reply.lines[0]
        assert "revenge" not in " ".join(reply.lines).lower()

    def test_forgot_seed_phrase_is_not_a_stablecoin_depeg(self) -> None:
        reply = newcomer.reply(
            "if i forgot my seed phrase too is my money just gone forever", named=False)
        assert reply is not None
        assert "not tied to a seed phrase at all" in reply.lines[0]
        body = " ".join(reply.lines)
        assert "Tether" not in body and "peg" not in body

    def test_a_plain_lost_password_still_routes_to_account_recovery(self) -> None:
        reply = newcomer.reply("i think im locked out of my bitget account", named=False)
        assert reply is not None and "account-recovery process" in reply.lines[0]


class TestNetworkMismatch:
    """Q6-Q7: ERC20 vs TRC20, and a USDT send on the wrong network."""

    def test_erc20_vs_trc20(self) -> None:
        reply = newcomer.reply(
            "im withdrawing usdt from bitget, which network should i pick erc20 or trc20 idk "
            "the diff", named=False)
        assert reply is not None
        assert "different blockchain networks" in reply.lines[0]
        assert "Ethereum" in reply.lines[0] and "TRON" in reply.lines[0]

    def test_sent_on_the_wrong_network(self) -> None:
        reply = newcomer.reply(
            "oh no i think i sent my usdt using the wrong network to my other wallet, is it "
            "gone", named=False)
        assert reply is not None
        assert reply.lines[0].startswith("Bottom line: not always gone")
        body = " ".join(reply.lines)
        assert "transaction hash" in body and "not guaranteed" in body

    def test_a_bare_withdraw_question_is_unaffected(self) -> None:
        reply = newcomer.reply("how do i actually get my money out if i wanna stop",
                               named=False)
        assert reply is not None and "withdrawals happen on Bitget" in reply.lines[0]


class TestEtfAndCheapCoin:
    """Q8, Q10: what an ETF is, and the low-unit-price "cheap coin" trap — the second one
    must be caught even when the server reads "coin" as the ticker COIN."""

    def test_what_is_an_etf(self) -> None:
        reply = newcomer.reply("whats an etf, is bitcoin an etf", named=False)
        assert reply is not None
        assert reply.lines[0].startswith("Bottom line:")
        body = " ".join(reply.lines)
        assert "exchange-traded fund" in body
        assert "Bitcoin itself is not an ETF" in body
        assert "spot bitcoin ETFs" in body

    def test_cheap_coin_trap_named_false(self) -> None:
        reply = newcomer.reply(
            "theres this coin at $0.0001 thats so cheap right i can buy millions of them",
            named=False)
        assert reply is not None
        assert "does not mean a coin is cheap" in reply.lines[0]
        assert "market cap" in reply.lines[0]

    def test_cheap_coin_trap_named_true(self) -> None:
        # the server may read "coin" as the ticker COIN (Coinbase) and pass named=True —
        # this must still be caught, since it is checked before the named gate
        reply = newcomer.reply(
            "theres this coin at $0.0001 thats so cheap right i can buy millions of them",
            named=True)
        assert reply is not None
        assert "does not mean a coin is cheap" in reply.lines[0]

    def test_a_plain_price_is_not_caught(self) -> None:
        # a real BTC price quote should not trip the cheap-coin trap
        assert newcomer.reply("what is BTC trading at", named=True) is None


class TestRetirementAndDca:
    """Q15: no alternative was ever named for "safer than crypto for retirement". Q17: the
    DCA-vs-lump-sum follow-up just repeated the DCA definition."""

    def test_safer_than_crypto_for_retirement(self) -> None:
        reply = newcomer.reply(
            "ok so whats actually safer than crypto for retirement money then", named=False)
        assert reply is not None
        body = " ".join(reply.lines)
        assert "government bonds" in body and "index funds" in body
        assert "no pick here" in reply.lines[0]

    def test_dca_vs_lump_sum_follow_up(self) -> None:
        reply = newcomer.reply(
            "how is that different than just buying once with all my money at the start",
            named=False)
        assert reply is not None
        assert "lump sum" in reply.lines[0] and "dollar-cost averaging" in reply.lines[0]

    def test_the_original_retirement_question_is_unaffected(self) -> None:
        reply = newcomer.reply(
            "my parents want to put their retirement savings into crypto should they do it",
            named=False)
        assert reply is not None and "licensed adviser" in reply.lines[0]


class TestSplittingAcrossCoins:
    """Q18: the "split it into different coins" half of a voice run-on was ignored entirely."""

    def test_splitting_into_different_coins(self) -> None:
        reply = newcomer.reply(
            "i have $500, should i put it all into bitcoin now or wait, or should i just "
            "split it into different coins to be safer", named=False)
        assert reply is not None
        assert "move together" in reply.lines[0] or "rise and fall together" in reply.lines[0]
        assert "tied to what bitcoin does" in reply.lines[0]

    def test_a_portfolio_rebalance_question_is_unaffected(self) -> None:
        # near-miss: no "coins" plural after the split/diversify verb
        assert newcomer.reply("should i diversify my portfolio", named=True) is None


class TestCustodyWithoutTheLanguageModel:
    """Q19/Q29-style custody questions must hit the custody answer from the plain-English
    router alone, since the hourly LLM allowance can run out (round 33's systemic finding)."""

    def test_keep_on_bitget_or_move_elsewhere(self) -> None:
        reply = newcomer.reply(
            "is it better to keep my coins on bitget or move them somewhere else", named=False)
        assert reply is not None
        assert reply.lines[0].startswith("Bottom line:")
        body = " ".join(reply.lines)
        assert "a claim on Bitget" in body
        assert "proof-of-reserves" in body
        assert "wallet only you control" in body

    def test_safe_to_keep_crypto_on_bitget_already_worked(self) -> None:
        reply = newcomer.reply("is it safe to keep crypto on bitget", named=False)
        assert reply is not None and "Bitget's to show" in reply.lines[0]

    def test_do_i_own_it_or_does_bitget_owe_me(self) -> None:
        reply = newcomer.reply(
            "if i buy bitcoin on bitget do i actually own it or does bitget just owe me",
            named=False)
        assert reply is not None
        assert "a claim on Bitget" in reply.lines[0]
        body = " ".join(reply.lines)
        assert "proof-of-reserves" in body and "self-custody wallet" in body


class TestMarketHoursAndSlang:
    """Q20, Q23, Q25: crypto never closes; "wen moon" went unrecognised even though "hodl?"
    worked a turn later; "should i buy rn" was declined because "rn" isn't "now"."""

    def test_crypto_never_closes(self) -> None:
        reply = newcomer.reply(
            "what time does the crypto market close like what time should i trade",
            named=False)
        assert reply is not None
        assert reply.lines[0].startswith("Bottom line: crypto never closes")
        assert "24 hours a day, 7 days a week" in reply.lines[0]

    def test_wen_moon(self) -> None:
        lines = newcomer.slang_lines("wen moon")
        assert lines is not None
        assert lines[0].startswith("Bottom line: WEN:")
        assert any(line.startswith("MOON:") for line in lines)

    def test_gm_everyone_still_returns_none(self) -> None:
        # the round-29 regression this fix must not break
        assert newcomer.slang_lines("gm everyone") is None

    def test_should_i_buy_rn(self) -> None:
        reply = newcomer.reply("ok but fr should i buy rn", named=False)
        assert reply is not None
        assert reply.reask == "has {name} been here before"
        assert reply.lead is not None and "no call" in reply.lead

    def test_should_i_buy_right_now_is_unaffected(self) -> None:
        # a named contract defers to its own engine either way (pre-existing behaviour)
        assert newcomer.reply("should i buy BTC right now", named=True) is None
        # the nameless phrasing without a ticker still gets the no-call reask as before
        reply = newcomer.reply("should i buy right now", named=False)
        assert reply is not None and reply.reask == "has {name} been here before"


class TestCoinVsToken:
    """Q30: "what's the difference between a coin and a token" was refused outright."""

    def test_coin_vs_token(self) -> None:
        reply = newcomer.reply("whats the difference between a coin and a token", named=False)
        assert reply is not None
        assert "its own blockchain" in reply.lines[0]
        body = " ".join(reply.lines)
        assert "gas" in body.lower()

    def test_a_memecoin_question_is_not_caught(self) -> None:
        reply = newcomer.reply("whats a memecoin", named=False)
        assert reply is not None
        assert "coin built on a joke" in reply.lines[0]
        assert "its own blockchain" not in reply.lines[0]


@pytest.mark.parametrize("said", [
    "is bitget safe", "what is a stop loss", "should i buy BTC right now",
    "whats a memecoin", "is 10x leverage ok for a small account", "how do i withdraw",
])
def test_round33_patterns_do_not_steal_existing_questions(said: str) -> None:
    reply = newcomer.reply(said, named=True)
    if said == "is bitget safe":
        assert reply is not None and "Bitget's to show" in reply.lines[0]
    elif said == "whats a memecoin":
        assert reply is not None and "coin built on a joke" in reply.lines[0]
    elif said == "how do i withdraw":
        assert reply is not None and "withdrawals happen on Bitget" in reply.lines[0]
    else:
        assert reply is None
