"""Round 34's first-time-user audit, run offline: each new or extended route is checked on the
exact phrasing that exposed it in `Activity/audits/round34_newcomer.md`, and on a near-miss
beside it so the fix does not steal a question it was never meant to answer."""

from __future__ import annotations

import pytest

from argus.lui import newcomer


class TestAirdropScams:
    """Conv1 Q1-Q3: whether airdrops are real, how to find legit ones, and the single most
    dangerous miss in the run — a DM airdrop-claim link asking to connect a wallet."""

    def test_dm_airdrop_link_never_connect_wallet(self) -> None:
        reply = newcomer.reply(
            "someone DMed me a link to claim an airdrop should i connect my wallet to it",
            named=False)
        assert reply is not None
        assert reply.lines[0].startswith("Bottom line: no")
        assert "never connect your wallet" in reply.lines[0]
        body = " ".join(reply.lines)
        assert "wallet drainer" in body.lower()

    def test_are_airdrops_real(self) -> None:
        reply = newcomer.reply("yo is airdrops real?... sounds like a scam ngl", named=False)
        assert reply is not None
        assert reply.lines[0].startswith("Bottom line: yes, real airdrops exist")

    def test_how_to_find_legit_airdrops(self) -> None:
        reply = newcomer.reply(
            "ok so if airdrops are real how do i find legit ones without getting scammed",
            named=False)
        assert reply is not None
        body = " ".join(reply.lines)
        assert "never asks for a fee" in body
        assert "project's own official site" in body

    def test_a_plain_airdrop_mention_elsewhere_is_unaffected(self) -> None:
        # near-miss: "airdrop" appears, but not in a scam-check shape
        assert newcomer.reply("what is the BTC price now", named=True) is None


class TestStablecoinYield:
    """Conv2 Q1, Q3: what a stablecoin yield is, why some apps offer 20% APY, and the general
    "is this APY too good to be true" heuristic."""

    def test_stablecoin_yield_and_20_percent_apy(self) -> None:
        reply = newcomer.reply(
            "whats a stablecoin yield and why do some apps offer 20% apy on usdt is that "
            "legit", named=False)
        assert reply is not None
        assert reply.lines[0].startswith("Bottom line: a stablecoin yield pays you")
        assert "20% APY" in reply.lines[0]

    def test_apy_too_good_to_be_true(self) -> None:
        reply = newcomer.reply("how do i tell if an apy is too good to be true", named=False)
        assert reply is not None
        assert "guaranteed, fixed return is the red flag" in " ".join(reply.lines)

    def test_a_real_bitget_earn_question_is_unaffected(self) -> None:
        reply = newcomer.reply("whats this bitget earn thing, is it safe", named=False)
        assert reply is not None and "Earn and staking pay you" in reply.lines[0]


class TestConvertVsTrade:
    """Conv5 Q1: Bitget's Convert versus its full Trade market, a basic newcomer concept
    refused outright."""

    def test_convert_vs_trade(self) -> None:
        reply = newcomer.reply(
            "whats the difference between convert and trade on the bitget app im confused",
            named=False)
        assert reply is not None
        assert reply.lines[0].startswith("Bottom line: on Bitget, Convert is a quick swap")
        body = " ".join(reply.lines)
        assert "limit order" in body and "market order" in body

    def test_placing_a_trade_is_unaffected(self) -> None:
        # near-miss: mentions "trade" but asks how to place one, not Convert-vs-Trade
        reply = newcomer.reply("how do i place a trade", named=False)
        assert reply is not None and "this console never places an order" in reply.lines[0]


class TestCoinLegitimacy:
    """Conv6 Q3: checking whether a coin itself is legit before buying — distinct from
    checking a service or a promised return, which round 32 already covers."""

    def test_how_do_i_know_a_coin_is_legit(self) -> None:
        reply = newcomer.reply(
            "how do i even know if a coin is legit before buying it", named=False)
        assert reply is not None
        assert reply.lines[0].startswith("Bottom line: no single check proves a coin is legit")
        body = " ".join(reply.lines)
        for check in ("team", "audit", "trading volume", "major exchange", "honeypot"):
            assert check in body.lower()

    def test_checking_a_service_is_still_unaffected(self) -> None:
        reply = newcomer.reply(
            "ok so how do i actually check if something like that is legit before i put "
            "money in", named=False)
        assert reply is not None and "four checks" in reply.lines[0]


class TestPresaleRisk:
    """Conv2 Q2: a presale is more than the generic shitcoin answer — there is no token or
    liquidity yet, and the team can vanish before anything lists."""

    def test_presale_specific_risk(self) -> None:
        reply = newcomer.reply(
            "my friend made bank on a new coin presale launch should i join presales too",
            named=False)
        assert reply is not None
        assert "riskier than buying an already-listed coin" in reply.lines[0]
        assert "exit scam" in " ".join(reply.lines)

    def test_a_plain_memecoin_pump_is_unaffected(self) -> None:
        reply = newcomer.reply("why would a coin with no real use even go up", named=False)
        assert reply is not None
        assert "a coin with no business behind it" in reply.lines[0]


class TestSlangTwoAtOnce:
    """Conv3 Q1: "what does rekt mean and also fomo" only defined REKT. Conv8 Q1's "diamond
    hands" was never defined anywhere, with or without a "what does" prefix."""

    def test_rekt_and_also_fomo_defines_both(self) -> None:
        lines = newcomer.slang_lines("what does rekt mean and also fomo")
        assert lines is not None
        assert lines[0].startswith("Bottom line: REKT:")
        assert any(line.startswith("FOMO:") for line in lines)

    def test_diamond_hands_defined_directly(self) -> None:
        lines = newcomer.slang_lines("what does diamond hands mean")
        assert lines is not None
        assert "diamond hands:" in lines[0].lower()

    def test_diamond_hands_via_wdym(self) -> None:
        lines = newcomer.slang_lines("my frend says diamond hands lol wdym by that")
        assert lines is not None
        assert "diamond hands:" in lines[0].lower()

    def test_gm_everyone_still_returns_none(self) -> None:
        # the round-29 regression this fix must not break
        assert newcomer.slang_lines("gm everyone") is None

    def test_a_single_slang_question_is_unaffected(self) -> None:
        lines = newcomer.slang_lines("what does rekt mean")
        assert lines is not None
        assert lines[0].startswith("Bottom line: REKT:")
        assert not any(line.startswith("FOMO:") for line in lines)


class TestFomoRektFollowUps:
    """Conv3 Q2-Q3: direct follow-ups about slang just defined were refused with the generic
    boilerplate instead of being answered."""

    def test_fomo_dump_same_as_rekt(self) -> None:
        reply = newcomer.reply(
            "wait so if i fomo into a coin and it dumps is that the same as getting rekt or "
            "different", named=False)
        assert reply is not None
        assert reply.lines[0].startswith("Bottom line: related, not identical")

    def test_can_i_get_my_money_back_from_rekt(self) -> None:
        reply = newcomer.reply(
            "ok so if i get rekt from fomoing into a coin can i like get my money bak "
            "somehow or its gone forever lol", named=False)
        assert reply is not None
        assert reply.lines[0].startswith("Bottom line: gone")
        assert "crypto transactions do not get reversed" in reply.lines[0].lower()


class TestGamblingConceptual:
    """Conv4 Q2-Q3: "is crypto basically just gambling then" chained two qualifiers the old
    regex couldn't take; "is there a way to trade without it being gambling" got an unsolicited
    $100k hedge pitch instead of a plain answer."""

    def test_is_crypto_basically_just_gambling(self) -> None:
        reply = newcomer.reply(
            "ok that's a lot tbh so is crypto basically just gambling then", named=False)
        assert reply is not None
        assert "a coin with no business behind it" in reply.lines[0]

    def test_trade_without_it_being_gambling(self) -> None:
        reply = newcomer.reply(
            "is there a way to trade without it basically being gambling", named=False)
        assert reply is not None
        assert reply.lines[0].startswith("Bottom line: yes")
        body = " ".join(reply.lines)
        assert "$100,000" not in body
        assert "will not size a position or pitch a hedge" in body.lower()

    def test_a_single_qualifier_gambling_question_is_unaffected(self) -> None:
        reply = newcomer.reply("so is it just gambling then", named=False)
        assert reply is not None and "a coin with no business behind it" in reply.lines[0]


class TestMinorTrading:
    """Conv6 Q2: "should a 16 year old be trading this stuff" repeated the prior turn's whale
    definition instead of answering the age question."""

    def test_should_a_16_year_old_trade(self) -> None:
        reply = newcomer.reply("should a 16 year old be trading this stuff", named=False)
        assert reply is not None
        assert reply.lines[0].startswith("Bottom line: not with an account of their own")
        assert "at least 18" in reply.lines[0]

    def test_first_person_sixteen_is_unaffected(self) -> None:
        reply = newcomer.reply("i'm 16 with $200 should i start trading crypto", named=False)
        assert reply is not None and "at least 18" in reply.lines[0]


class TestNftAndRecession:
    """Conv7 Q1-Q3: what an NFT is, the point of buying one that can also crash, and what a
    recession does to crypto — all three hit the console's hourly quota wall with zero plain-
    English fallback."""

    def test_what_is_an_nft(self) -> None:
        reply = newcomer.reply("what is an nft and should i buy one", named=False)
        assert reply is not None
        assert "non-fungible token" in reply.lines[0]

    def test_point_of_buying_an_nft_that_can_crash(self) -> None:
        reply = newcomer.reply(
            "if nfts crash too then whats even the point of buying one", named=False)
        assert reply is not None
        assert "honest point is not profit" in reply.lines[0]

    def test_crypto_in_a_recession(self) -> None:
        reply = newcomer.reply(
            "what happens to crypto if theres a recession does it crash too", named=False)
        assert reply is not None
        assert reply.lines[0].startswith("Bottom line: crypto has mostly moved with the "
                                          "stock market")

    def test_a_plain_etf_question_is_unaffected(self) -> None:
        reply = newcomer.reply("whats an etf, is bitcoin an etf", named=False)
        assert reply is not None and "exchange-traded fund" in " ".join(reply.lines)


@pytest.mark.parametrize("said", [
    "is bitget safe", "what is a stop loss", "should i buy BTC right now",
    "whats a memecoin", "what is a seed phrase", "how do i withdraw",
    "is 10x leverage ok for a small account",
])
def test_round34_patterns_do_not_steal_existing_questions(said: str) -> None:
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
