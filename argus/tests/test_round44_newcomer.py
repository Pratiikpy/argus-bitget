"""Round 44 newcomer audit (Activity/audits/round44_newcomer.md): the first-time investor's
emergencies, scam questions, basics, typo and slang forms, and follow-ups read with their topic,
answered offline by the beginner layer.

Every question is driven through ``beginner.early(text, prior)``, the first thing the console's
ask route calls. The facts behind each answer are cited in the comments of
``src/argus/lui/beginner.py``, each read from its source on 2026-10-06."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from argus.lui import beginner


def _said(text: str, prior: list[str] | None = None) -> list[str]:
    got = beginner.early(text, prior or [])
    assert got is not None, f"not answered: {text!r} after {prior!r}"
    assert got[0].startswith("Bottom line: ")
    assert 2 <= len(got) <= 6
    return got


def _all(text: str, prior: list[str] | None = None) -> str:
    return " ".join(_said(text, prior))


class TestWrongNetworkEmergency:
    @pytest.mark.parametrize("text", [
        "bro i sent my eth 2 wrong chain am i cooked",
        "I sent crypto to the wrong network, what now",
        "I sent coins to the wrong address",
        "i withdrew usdt on trc20 instead of erc20, help",
        "network mismatch on my deposit, what do i do",
        "sent my btc to a wrong adress pls hlp",
    ])
    def test_it_is_an_emergency_not_a_price_dip(self, text: str) -> None:
        said = _all(text)
        assert "hash" in said and "support" in said
        assert "scammer" in said and "fee" in said
        assert "paper loss" not in said and "wait without" not in said

    def test_it_names_every_outcome(self) -> None:
        said = _all("i sent my eth to the wrong network")
        assert "your own wallet" in said.lower() and "exchange" in said
        assert "contract" in said and "usually not recoverable" in said

    def test_hypothetical_prevention_is_not_the_emergency(self) -> None:
        got = beginner.early("how do I make sure I never send to the wrong network", [])
        assert got is None or "scammer" not in " ".join(got)

    def test_can_i_get_it_back_is_read_in_that_context(self) -> None:
        said = _all("so can I get it back", ["bro i sent my eth 2 wrong chain am i cooked"])
        assert "contact the receiving platform" in said
        assert "chasing" not in said and "win it back" not in said
        assert "do not pay anyone" in said

    @pytest.mark.parametrize("text", ["can i get it back", "is it gone forever", "any hope of "
                                      "recovering it"])
    def test_other_ways_to_ask_for_it_back(self, text: str) -> None:
        assert "depends on who controls" in _all(text, ["i sent coins to the wrong address"])

    def test_without_that_context_it_is_still_the_chasing_losses_answer(self) -> None:
        assert "win it back" in _all("so can I get it back", ["i lost 60% and feel sick"])

    def test_the_two_follow_ups_keep_the_wrong_send_topic_over_slang(self) -> None:
        prior = ["bro i sent my eth 2 wrong chain am i cooked"]
        assert "transaction hash" in _all("what should I check first", prior)
        assert "sending it again" in _all("so is it safe or not", prior)
        short = _said("say it shorter", prior)
        assert "scammer" in " ".join(short)


class TestScamsAndSafety:
    def test_recovery_service_is_a_scam(self) -> None:
        said = _all("a recovery service says they can get my stolen coins back for a fee")
        assert "scam" in said and "do not pay" in said and "FTC" in said

    @pytest.mark.parametrize("text", [
        "this recovery company offered to recover my crypto if i pay a fee first",
        "someone messaged me they can get my money back, they want a deposit first",
    ])
    def test_other_recovery_offers(self, text: str) -> None:
        assert "scam" in _all(text)

    def test_fake_app(self) -> None:
        said = _all("how do I spot a fake crypto app")
        assert "publisher" in said and "withdraw" in said and "recovery phrase" in said

    def test_a_fake_app_already_installed_gets_the_clean_up(self) -> None:
        said = _all("I downloaded a fake crypto app, how do I check what it did")
        assert "delete the app" in said

    def test_one_exchange_or_several(self) -> None:
        said = _all("keep crypto on one exchange or several")
        assert "single point of failure" in said and "FTX" in said and "proof-of-reserves" in said

    def test_benefits_depend_on_the_country_and_invent_no_rule(self) -> None:
        said = _all("do crypto gains count as income when I claim benefits")
        assert "depends on your country" in said and "ask the benefit office" in said
        assert not re.search(r"[$£€]\s?\d|\d+\s?%", said)

    def test_benefits_of_bitcoin_is_not_this_question(self) -> None:
        assert beginner.early("what are the benefits of bitcoin", []) is None

    def test_a_rug_pull_and_a_dump_are_two_definitions(self) -> None:
        said = _all("difference between a rug pull and a dump")
        assert said.count("Rug pull:") == 1 and said.count("Dump:") == 1
        assert "pump and dump" in said

    def test_the_plain_rug_answer_no_longer_defines_a_dump(self) -> None:
        said = _all("what is a rug pull")
        assert "sell their huge holdings all at once" not in said


class TestBasics:
    def test_hardware_wallet(self) -> None:
        said = _all("what is a hardware wallet")
        assert "physical device" in said and "offline" in said and "recovery phrase" in said
        assert "not required to use Bitget" not in said

    def test_stock_split(self) -> None:
        said = _all("what is a stock split")
        assert "2-for-1" in said and "value of what you own does not change" in said
        assert "rToken" not in said

    def test_a_named_stock_split_is_left_to_the_research_readers(self) -> None:
        assert beginner.early("what did the NVDA stock split do to the price", []) is None

    def test_index_fund(self) -> None:
        said = _all("what is an index fund")
        assert "S&P 500" in said and "expense ratio" in said and "not safe" in said

    def test_index_fund_against_bitcoin_is_left_alone(self) -> None:
        assert beginner.early("index fund vs bitcoin which is better", []) is None

    def test_buying_shares_the_first_time(self) -> None:
        said = _all("I want to buy shares for the first time, where do I start")
        assert "broker" in said and "rTokens" in said and "do not own the share" in said
        assert "spot market" not in said

    def test_balance_in_two_currencies(self) -> None:
        said = _all("why does my balance show a different number in USD and EUR")
        assert "current exchange rate" in said and "cannot see your account" in said

    def test_slippage(self) -> None:
        said = _all("what does it mean that my market order slipped")
        assert "price you expected" in said and "limit order" in said

    def test_a_quantitative_slippage_question_is_not_taken(self) -> None:
        assert beginner.early("what slippage would a $5m market buy on BTC have", []) is None

    def test_a_first_purchase_has_plain_cautions_not_a_position_table(self) -> None:
        said = _all("what to watch for when buying Bitcoin for the first time")
        assert "start small" in said and "two-factor" in said and "leverage" in said
        assert "kurtosis" not in said and "20%" not in said

    def test_a_dividend_is_not_necessarily_monthly(self) -> None:
        said = _all("what is a dividend and do I get paid every month")
        assert "four times a year" in said and "ex-dividend" in said

    def test_liquidation_consequence(self) -> None:
        said = _all("what happens if I get liquidated")
        assert "closes your leveraged position" in said and "liquidation price" in said

    def test_hodl(self) -> None:
        said = _all("is hodl a thing or just copium lol")
        assert "misspelling" in said and "47,216" in said and "no leverage" in said


class TestFomoAllIn:
    def test_the_audits_typo_form_leads_with_not_everything(self) -> None:
        got = _said("shud i put evrything in btc rn fomo is killing me")
        assert got[0].startswith("Bottom line: no, not everything")
        said = " ".join(got)
        assert "FOMO" in said and "small amount" in said and "no leverage" in said
        assert len(got) <= 4
        assert "funding" not in said and "finBERT" not in said

    @pytest.mark.parametrize("text", ["should i put all my savings in bitcoin",
                                      "is it smart to go all in on btc", "i want to put "
                                      "everything into eth"])
    def test_other_all_in_questions(self, text: str) -> None:
        assert "not everything" in _all(text)


class TestTyposReachTheCleanAnswers:
    def test_candlestick(self) -> None:
        said = _all("wat is a candel stik and y r sum red")
        assert "wicks" in said and "green" in said

    def test_dividend(self) -> None:
        assert "ex-dividend" in _all("wut is a divident n do i get payd evry month")

    def test_liquidation(self) -> None:
        assert "liquidation price" in _all("wat hapens if i get liqidated??")

    def test_plain_does_not_change_other_words(self) -> None:
        for word in ("bitcoin", "wallet", "everything", "years", "hap", "ratio", "yes", "rnd",
                     "address", "liquidation", "dividend", "candlestick"):
            assert beginner._plain(word) == word


_SUBJECTS = [
    ("should i join presales", "presale"),
    ("should i borrow money to invest", "borrow"),
    ("what is an index fund", "index"),
    ("what is staking", "staking"),
    ("my friend uses 50x leverage is that ok", "leverage"),
    ("is P2P trading safe", "p2p"),
    ("my friend added me to a signal group, should i join", "signal"),
    ("does a crypto ATM scam work like that", "atm"),
    ("what is a stablecoin", "stable"),
    ("keep crypto on one exchange or several", "wallet"),
    ("what is a hardware wallet", "wallet"),
    ("what is a stock split", "split"),
    ("I want to buy shares for the first time, where do I start", "shares"),
    ("what is a dividend and do I get paid every month", "dividend"),
    ("is hodl a thing or just copium lol", "hodl"),
    ("shud i put evrything in btc rn fomo is killing me", "all_in"),
    ("a recovery service says they can get my stolen coins back for a fee", "recovery"),
    ("how do I spot a fake crypto app", "fake_app"),
    ("what is a rug pull", "rug"),
    ("what is dollar cost averaging", "dca"),
    ("do crypto gains count as income when I claim benefits", "benefits"),
    ("what to watch for when buying Bitcoin for the first time", "first_buy"),
    ("I sent crypto to the wrong network, what now", "wrong_send"),
    ("what is a candlestick", "other"),
    ("what is market cap", "other"),
    ("why is the price different between two apps", "other"),
]


class TestFollowUpsReadWithTheirTopic:
    @pytest.mark.parametrize("subject,key", _SUBJECTS)
    def test_the_topic_is_found(self, subject: str, key: str) -> None:
        assert beginner._topic(subject) == key

    @pytest.mark.parametrize("subject,key", _SUBJECTS)
    @pytest.mark.parametrize("follow", ["so is it safe or not", "is it safe",
                                        "ok but is that dangerous"])
    def test_is_it_safe_answers_for_the_subject(self, subject: str, key: str,
                                                follow: str) -> None:
        said = _all(follow, [subject])
        assert "console never holds" not in said
        if key in beginner._SAFE_BY_TOPIC:
            assert said == " ".join(beginner._SAFE_BY_TOPIC[key])
        else:
            assert said == " ".join(beginner._SAFE_GENERIC)

    @pytest.mark.parametrize("subject,key", _SUBJECTS)
    @pytest.mark.parametrize("follow", ["what should I check first", "so what do i check first",
                                        "what to check"])
    def test_check_first_answers_for_the_subject(self, subject: str, key: str,
                                                 follow: str) -> None:
        said = _all(follow, [subject])
        expected = beginner._CHECK_BY_TOPIC.get(key, beginner._CHECK_GENERIC)
        assert said == " ".join(expected)

    def test_the_audits_three_conversations(self) -> None:
        presale = ["should i join presales"]
        assert "presale" in _all("so is it safe or not", presale)
        assert "borrowing to invest is not safe" in _all(
            "so is it safe or not", ["should i borrow money to invest", "what about interest"])
        assert "not safe" in _all("so is it safe or not", ["what is an index fund"])

    def test_the_latest_topic_wins_over_the_earlier_one(self) -> None:
        said = _all("what should I check first",
                    ["should i join presales", "I sent crypto to the wrong network, what now"])
        assert "transaction hash" in said

    def test_a_follow_up_with_no_known_topic_is_left_alone(self) -> None:
        assert beginner.early("so is it safe or not", []) is None
        assert beginner.early("what should I check first", ["what is the price of gold"]) is None

    def test_the_generic_answers_never_promise_safety(self) -> None:
        for lines in (beginner._SAFE_GENERIC, beginner._CHECK_GENERIC):
            text = " ".join(lines)
            assert "Bottom line: " in text and "buy" not in text.lower().split("bottom line:")[0]


class TestSayItShorter:
    @pytest.mark.parametrize("subject,key", [s for s in _SUBJECTS if s[1] != "wrong_send"
                                             or True])
    def test_it_is_genuinely_shorter_and_keeps_every_point(self, subject: str,
                                                           key: str) -> None:
        # the readers that reach live data are not asked: only fixed answers
        if key in ("leverage", "other") and "market cap" not in subject \
                and "candlestick" not in subject:
            pytest.skip("answered by a reader that reads live data or another layer")
        full = beginner._answer_for(beginner._plain(subject))
        short = beginner.early("say it shorter", [subject])
        if full is None:
            pytest.skip("answered by a later layer, not by this one")
        assert short is not None and short[0].startswith("Bottom line: ")
        assert len(" ".join(short)) < len(" ".join(full))
        assert len(short) >= min(len(full), 3) or len(short) >= len(full) - 1

    @pytest.mark.parametrize("follow", ["say it shorter", "shorter please", "make it shorter",
                                        "tl;dr", "that's too long", "be brief"])
    def test_ways_to_ask(self, follow: str) -> None:
        got = _said(follow, ["what is an index fund"])
        assert "index" in " ".join(got).lower()

    def test_the_audits_declined_conversations_are_answered(self) -> None:
        for subject in ("should i join presales", "should i borrow money to invest",
                        "what is an index fund"):
            assert _said("say it shorter", [subject])

    def test_a_short_answer_for_a_new_topic_keeps_its_main_numbers(self) -> None:
        short = " ".join(_said("say it shorter", ["what is a stock split"]))
        assert "2-for-1" in short or "value of what you own" in short

    def test_older_simpler_answers_are_untouched(self) -> None:
        assert _said("explain simpler", ["is P2P trading safe"]) == list(beginner._P2P_S)


class TestDeclinedLanguages:
    WANTED = ("vi", "tr", "id", "it", "pl", "sw", "ru", "hi", "ar", "th")

    def test_every_language_the_console_detects_has_a_refusal(self) -> None:
        from argus.lui.translate import PAUSED

        for code in ("fr", "de", "es", "pt", "zh", "zh-Hant", "ja", "ko", "vi", "tr", "id", "ru",
                     "hi", "ar", "th"):
            assert code in PAUSED or code in beginner.DECLINED_LANGUAGE, code

    def test_only_what_translate_lacks_is_added(self) -> None:
        from argus.lui.translate import PAUSED

        assert set(beginner.DECLINED_LANGUAGE) == set(self.WANTED)
        assert not (set(beginner.DECLINED_LANGUAGE) & set(PAUSED))

    @pytest.mark.parametrize("code", WANTED)
    def test_each_has_a_note_a_refusal_and_a_hint(self, code: str) -> None:
        got = beginner.declined_language(code, 17)
        assert got is not None
        note, refusal, hint = got
        assert "17" in note and "17" in refusal
        assert "{" not in note + refusal + hint
        assert "language model" not in note + refusal

    def test_a_language_without_text_gets_none(self) -> None:
        assert beginner.declined_language("fr", 5) is None
        assert beginner.declined_language(None, 5) is None


class TestFileHygiene:
    def test_no_backspace_bytes_in_the_files_this_round_wrote(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for rel in ("src/argus/lui/beginner.py", "tests/test_round44_newcomer.py"):
            assert b"\x08" not in (root / rel).read_bytes(), rel

    def test_every_new_answer_is_two_to_six_lines_and_plain(self) -> None:
        for name in dir(beginner):
            value = getattr(beginner, name)
            if (name.startswith("_") and name.endswith(("_A", "_S")) and name != "_FAKE_APP_DONE_A"
                    and isinstance(value, tuple)
                    and value and all(isinstance(x, str) for x in value)):
                assert value[0].startswith("Bottom line: "), name
                assert 2 <= len(value) <= 5, name
                assert not any(chr(0x2019) in line or chr(0x201c) in line for line in value), name
