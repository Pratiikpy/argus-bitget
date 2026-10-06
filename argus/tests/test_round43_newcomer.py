"""Round 43 newcomer audit (Activity/audits/round43_newcomer.md): first-time users' questions that
were refused or misread are answered by the beginner layer, offline.

Every question is driven through ``beginner.early(text, prior)``, the first thing the console's
ask route calls. The facts behind each answer are cited in the comments of
``src/argus/lui/beginner.py``, each read from its source on 2026-10-06."""

from __future__ import annotations

import re

import pytest

from argus.lui import beginner


def _said(text: str, prior: list[str] | None = None) -> list[str]:
    got = beginner.early(text, prior or [])
    assert got is not None, f"not answered: {text!r}"
    assert got[0].startswith("Bottom line: ")
    assert 2 <= len(got) <= 6
    return got


def _all(text: str, prior: list[str] | None = None) -> str:
    return " ".join(_said(text, prior))


class TestSafety:
    def test_p2p(self) -> None:
        said = _all("is P2P trading safe")
        assert "your own bank account" in said and "screenshot" in said and "escrow" in said

    def test_signal_group_from_a_friend(self) -> None:
        said = _all("my friend added me to a signal group, should i join")
        assert "scammed" in said and "never pay to join" in said.lower()

    def test_a_paid_signal_group_is_the_same_answer(self) -> None:
        assert "pump and dump" in _all("theres a telegram signal group that wants $50 a month, "
                                       "is it legit")

    def test_phishing_email_says_do_not_click(self) -> None:
        said = _all("I got an email saying it is from Bitget asking me to verify my account, "
                    "click link")
        assert "phishing" in said and "do not click" in said and "anti-phishing code" in said
        assert "KYC" not in said

    def test_forgot_2fa_is_not_a_ticker(self) -> None:
        said = _all("I forgot my 2FA, what now")
        assert "prove who you are" in said and "not listed" not in said
        assert "scam" in said

    @pytest.mark.parametrize("text", ["how do I know if a coin is a scam",
                                      "hw do i no if a cooin is a scam pls hlp"])
    def test_scam_coin_with_and_without_typos(self, text: str) -> None:
        said = _all(text)
        assert "listed on a major exchange" in said and "guaranteed return" in said

    def test_rug_pull(self) -> None:
        said = _all("bro whats a rugpull gm")
        assert "vanish" in said and "honeypot" in said

    def test_crypto_atm(self) -> None:
        said = _all("what is a crypto ATM and is it safe")
        assert "cannot be pulled back" in said and "scammer" in said

    def test_pay_a_fine_at_an_atm_is_called_a_scam_first(self) -> None:
        got = _said("a man phoned and said I must pay a fine at a crypto ATM")
        assert got[0].startswith("Bottom line: it is a scam")

    def test_sim_swap(self) -> None:
        said = _all("can I lose my account if someone gets my phone number")
        assert "SIM swap" in said and "authenticator app" in said and "whitelist" in said

    def test_report_a_scam(self) -> None:
        said = _all("how do I report a scam")
        assert "reportfraud.ftc.gov" in said and "Report Fraud" in said and "evidence" in said

    def test_trustworthy_exchange(self) -> None:
        said = _all("how do I know if an exchange is trustworthy")
        assert "regulator" in said and "proof-of-reserves" in said

    def test_pump_and_dump_says_how_to_spot_and_avoid(self) -> None:
        said = _all("what is a pump and dump")
        assert "How to spot" in said and "How to avoid" in said


class TestBasics:
    def test_read_a_chart(self) -> None:
        said = _all("how do I read a chart")
        assert "candle" in said and "volume" in said and "cannot tell you what comes next" in said

    def test_ten_dollars(self) -> None:
        got = _said("can i invest only 10 dollars")
        assert "$10" in got[0] and "$0.01" in " ".join(got) and "minimum order" in " ".join(got)

    def test_dca_versus_lump_sum(self) -> None:
        said = _all("what is dollar cost averaging vs lump sum")
        assert "two thirds" in said and "not crypto" in said

    def test_exchange_versus_wallet(self) -> None:
        said = _all("what is the difference between a crypto exchange and a wallet app")
        assert "recovery phrase" in said and "keeps your coins" in said

    def test_bull_trap(self) -> None:
        assert "breaks upward" in _all("what is a bull trap")

    def test_bitcoin_bubble_does_not_dump_statistics(self) -> None:
        said = _all("is bitcoin a bubble")
        assert "nobody knows" in said and "47,216" in said and "beta" not in said

    def test_inflation_and_bitcoin_is_honest_and_defined(self) -> None:
        said = _all("what is inflation and does bitcoin protect against it")
        assert "general rise in prices" in said and "not reliably" in said
        assert "9.1%" in said and "65%" in said
        assert "10-year" not in said and "curve" not in said

    def test_withdrawal_limit_invents_no_number(self) -> None:
        said = _all("what is a limit on how much I can withdraw each day")
        assert "Help Center" in said and "24 hours" in said
        assert not re.search(r"\d+\s*(?:million|USDT)", said)

    def test_buy_now_or_wait_for_a_dip(self) -> None:
        said = _all("should i buy now or wait for a dip")
        assert "nobody can time it" in said and "dollar-cost averaging" in said
        assert "Read as" not in said

    def test_the_app_shows_a_loss(self) -> None:
        said = _all("the app shows a loss, what do I do")
        assert "on paper" in said and "panic-sell" in said

    def test_bought_at_the_top(self) -> None:
        said = _all("i bought at the top lol am i cooked fr fr")
        assert "not necessarily cooked" in said and "sell" in said

    @pytest.mark.parametrize("text", ["is it too late to start investing im 40",
                                      "is it 2l8 to start inveesting im 40"])
    def test_too_late_at_forty(self, text: str) -> None:
        said = _all(text)
        assert "not too late" in said and "about 25 years" in said

    def test_wen_moon_ser(self) -> None:
        said = _all("wen moon ser, shud i ape in")
        assert "nobody knows" in said and "Never borrow" in said

    def test_two_apps_two_prices(self) -> None:
        said = _all("why is the price different on two apps")
        assert "order book" in said and "spread" in said

    def test_fraction_of_a_share_is_not_a_ticker(self) -> None:
        said = _all("can I buy a fraction of a share")
        assert "by amount" in said and "not a stock Bitget lists" not in said

    def test_fraction_of_a_coin(self) -> None:
        assert "100,000,000" in _all("can I buy a fraction of a bitcoin")

    def test_price_moving_fast_uses_live_closes_when_available(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        from datetime import datetime, timedelta

        start = datetime(2025, 1, 1)
        closes = [100.0 + i for i in range(60)]
        closes[40] = closes[39] * 0.90
        stamps = [start + timedelta(days=i) for i in range(60)]
        monkeypatch.setattr("argus.lui.research.rule_test.daily_closes",
                            lambda symbol: (stamps, closes, "stub"))
        said = _all("is it normal for the price to move this fast")
        assert "worst day was -10.0%" in said and "1 days" in said

    def test_price_moving_fast_survives_a_failed_source(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(symbol: str) -> tuple[list[object], list[float], str]:
            raise OSError("no network")

        monkeypatch.setattr("argus.lui.research.rule_test.daily_closes", boom)
        said = _all("is it normal for the price to move this fast")
        assert "normal" in said and "worst day" not in said

    def test_bitcoin_with_typos_is_defined_and_bought(self) -> None:
        said = _all("wat is bitcon n how do i buy sum plz")
        assert "digital currency" in said and "desk decides" not in said

    def test_whale_and_whether_to_follow(self) -> None:
        said = _all("what is a whale and should I follow them")
        assert "Usually not" in said and "bait" in said

    def test_market_cap_for_crypto(self) -> None:
        said = _all("what does market cap mean")
        assert "number of coins in circulation" in said and "does not mean cheap" in said

    def test_stablecoin_with_depeg_risk(self) -> None:
        said = _all("what is a stablecoin")
        assert "TerraUSD" in said and "not risk-free" in said


class TestCountryTax:
    def test_uk(self) -> None:
        said = _all("taxes on crypto in the UK")
        assert "HMRC" in said and "£3,000" in said and "Capital Gains Tax" in said
        assert not re.search(r"\bIRS\b", said)

    def test_us(self) -> None:
        said = _all("taxes on crypto in the US")
        assert "property" in said and "Form 8949" in said and "one year or less" in said

    def test_nigeria_names_the_authority_and_states_no_rate_it_cannot_source(self) -> None:
        said = _all("how are crypto gains taxed in Nigeria")
        assert "Nigeria Revenue Service" in said and "disagree" in said
        assert not re.search(r"\bIRS\b", said)

    def test_brazil(self) -> None:
        said = _all("how is crypto taxed in Brazil")
        assert "R$35,000" in said and "Receita Federal" in said
        assert not re.search(r"\bIRS\b", said)

    def test_a_pronoun_us_is_not_a_country(self) -> None:
        assert beginner.early("can you tax-check crypto for us", []) is None

    def test_an_unrelated_company_tax_question_is_left_alone(self) -> None:
        assert beginner.early("By how much did American Express's effective tax rate change "
                              "between FY2021 and FY2022?", []) is None

    def test_irs_remark_is_not_the_us_tax_answer(self) -> None:
        assert beginner.early("do you guys send anything to the irs or is that on me", []) is None


class TestFollowUps:
    def test_stablecoin_then_if_it_goes_down(self) -> None:
        said = _all("and if it goes down?", ["what is a stablecoin"])
        assert "lost its dollar peg" in said and "not a contract" not in said

    def test_loss_thread_then_if_it_goes_down(self) -> None:
        said = _all("and if it goes down?", ["the app shows a loss, what do I do",
                                             "how long should I hold"])
        assert "65% in 2022" in said and "leverage" in said

    def test_if_it_goes_down_with_no_subject_is_left_alone(self) -> None:
        assert beginner.early("and if it goes down?", ["what is the weather"]) is None

    def test_tax_then_what_about_the_us(self) -> None:
        said = _all("what about the US", ["taxes on crypto in the UK"])
        assert "IRS" in said and "Form 8949" in said

    def test_tax_then_what_about_nigeria(self) -> None:
        assert "Nigeria Revenue Service" in _all("what about Nigeria",
                                                 ["tax on crypto in Brazil"])

    def test_withdrawal_then_what_should_i_do_restates_the_steps(self) -> None:
        got = _said("ok so what should I do", ["how do i withdraw my money to my bank account"])
        text = " ".join(got)
        assert text.index("1. ") < text.index("2. ") < text.index("3. ") < text.index("4. ")
        assert "three checks" not in text

    def test_what_should_i_do_without_a_withdrawal_is_left_to_the_generic_answer(self) -> None:
        assert beginner.early("ok so what should I do",
                              ["should i buy now or wait for a dip"]) is None

    def test_simpler_after_withdrawal_keeps_every_step_in_shorter_words(self) -> None:
        full = _said("ok so what should I do", ["how do i withdraw my money to my bank account"])
        simple = _said("explain simpler", ["how do i withdraw my money to my bank account"])
        assert [x[:2] for x in simple[1:]] == ["1.", "2.", "3.", "4."]
        assert len(" ".join(simple)) < len(" ".join(full))
        assert not any(x.startswith("Said more simply") for x in simple)

    def test_simpler_after_the_what_now_list_keeps_item_one(self) -> None:
        simple = _said("explain simpler", ["what is staking", "ok so what should I do"])
        assert "1. " in " ".join(simple) and "3. " in " ".join(simple)

    def test_simpler_after_staking(self) -> None:
        simple = _all("explain simpler", ["what is staking"])
        assert "lock your coins" in simple

    def test_simpler_after_a_stablecoin_answer(self) -> None:
        full = _all("what is a stablecoin")
        simple = _all("explain simpler", ["what is a stablecoin"])
        assert len(simple) < len(full) and "ten cents" in simple

    def test_simpler_with_nothing_of_ours_before_it_is_left_alone(self) -> None:
        assert beginner.early("explain simpler", ["what is the price of nvda"]) is None


class TestDeclinedLanguage:
    @pytest.mark.parametrize("code,word", [("vi", "tiếng Anh"), ("tr", "İngilizce"),
                                           ("id", "bahasa Inggris")])
    def test_refusal_in_the_users_language(self, code: str, word: str) -> None:
        got = beginner.declined_language(code, 25)
        assert got is not None
        note, refusal, hint = got
        assert word in refusal and "25" in refusal and "{minutes}" not in refusal
        assert word in note and "25" in note
        assert "TSLA" in hint and "Without the language model" not in refusal

    def test_languages_the_server_already_covers_are_not_duplicated(self) -> None:
        from argus.lui.translate import PAUSED

        assert not set(beginner.DECLINED_LANGUAGE) & set(PAUSED)
        assert beginner.declined_language("fr", 10) is None
        assert beginner.declined_language(None, 10) is None

    def test_the_detector_picks_these_three_languages(self) -> None:
        from argus.lui.translate import target_language

        assert target_language("Làm sao để biết một nhóm tín "
                               "hiệu có phải lừng đảo không") == "vi"
        assert target_language("Apa itu staking dan apakah aman untuk pemula yang baru "
                               "mulai") == "id"


class TestNothingElseIsTaken:
    @pytest.mark.parametrize("text", [
        "what is the price of bitcoin",
        "how do i buy bitcoin with $100",
        "should I DCA into BTC with $1,000?",
        "dca into BTC starting 2023-02-29",
        "How should I buy $2 million of COIN?",
        "gold vs bitcoin as an inflation hedge",
        "BTC is in a bubble",
        "someone DMed me a link to claim an airdrop should i connect my wallet to it",
        "I want to risk $300, my stop is 3%, how many shares should I buy of a $50 stock",
        "what is Tesla's market cap",
        "how do I verify my account",
        "what is the stablecoin yield and why do some apps offer 20% apy",
    ])
    def test_questions_other_readers_own_pass_through(self, text: str) -> None:
        got = beginner.early(text, [])
        assert got is None or not any(
            marker in got[0] for marker in ("phishing", "dollar-cost averaging (DCA)", "bubble",
                                            "you can start with $"))

    def test_every_new_answer_is_short_plain_and_has_no_markup(self) -> None:
        asked = ["is P2P trading safe", "how do I report a scam", "what is a stablecoin",
                 "how do I read a chart", "can i invest only 10 dollars",
                 "taxes on crypto in the UK", "what does market cap mean"]
        for text in asked:
            for line in _said(text):
                assert len(line) < 700 and "**" not in line and chr(0x2212) not in line
