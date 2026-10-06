"""Round 45 newcomer audit (Activity/audits/round45_newcomer.md) and the round 45 hostile C8:
a first-time user's debt, scam, job, fear, privacy and leverage questions, answered offline by the
beginner layer.

Every question is driven through ``beginner.early(text, prior)``, the first thing the console's ask
route calls. The price history is replaced by a synthetic year with two 10-12% days, so every
figure asserted here is one the answer counted from that series, never a constant."""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta

import pytest

from argus.lui import beginner


def _series(n: int = 400) -> tuple[list[datetime], list[float], str]:
    closes = [100.0]
    for i in range(1, n):
        step = 0.01 if i % 2 else -0.01
        if i == 100:
            step = -0.12
        if i == 200:
            step = 0.11
        closes.append(closes[-1] * (1 + step))
    start = datetime(2025, 1, 1, tzinfo=UTC)
    return [start + timedelta(days=i) for i in range(n)], closes, "Bitget's daily closes"


@pytest.fixture(autouse=True)
def history(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("argus.lui.research.rule_test.daily_closes", lambda symbol: _series())


@pytest.fixture
def no_history(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(symbol: str) -> tuple[list[datetime], list[float], str]:
        raise OSError("offline")

    monkeypatch.setattr("argus.lui.research.rule_test.daily_closes", broken)


def _said(text: str, prior: list[str] | None = None) -> list[str]:
    got = beginner.early(text, prior or [])
    assert got is not None, f"not answered: {text!r} after {prior!r}"
    assert got[0].startswith("Bottom line: ")
    return got


def _all(text: str, prior: list[str] | None = None) -> str:
    said = " ".join(_said(text, prior))
    assert not re.search(r"\d[eE][+-]\d", said), "scientific notation"
    return said


class TestJob:
    def test_quit_job_after_a_price_target(self) -> None:
        said = _all("so should i quit my job?? be honest",
                    ["will bitcoin hit 1 million this year?"])
        assert said.startswith("Bottom line: no")
        assert "price target" in said and "kurtosis" not in said and "beta" not in said

    @pytest.mark.parametrize("text", [
        "should i quit my job and trade full time",
        "can i quit my job if bitcoin pays off, be honest",
        "is it a good idea to leave my job for crypto",
    ])
    def test_variants_are_a_plain_no(self, text: str) -> None:
        assert _all(text).startswith("Bottom line: no")

    def test_mixed_question_is_left_to_the_price_reader(self) -> None:
        text = "will bitcoin hit 1 million this year? i wanna quit my job if it does"
        assert beginner.early(text, []) is None
        line = beginner.income_line(text)
        assert line is not None and line.startswith("On your job: no")

    @pytest.mark.parametrize("text", [
        "i quit my job yesterday", "what time do i leave work", "the job market is bad",
        "will bitcoin hit 100k this year",
    ])
    def test_no_job_question_no_line(self, text: str) -> None:
        assert beginner.income_line(text) is None
        assert beginner.early(text, []) is None


class TestBorrowToBuy:
    @pytest.mark.parametrize("text", [
        "i borrowed 5000 from my bank to buy crypto cuz everyone says its gonna moon, good idea?",
        "i took a loan from my bank of 5000 dollars to buy bitcoin, is this a good idea or a bad "
        "idea",
        "should i put 3k on my credit card into bitcoin",
        "thinking of taking out a loan to invest in eth",
    ])
    def test_do_not_borrow(self, text: str) -> None:
        said = _all(text)
        assert said.startswith("Bottom line: no — do not borrow")
        assert "repaid" in said and "interest" in said

    def test_the_figures_are_counted_not_written_in(self) -> None:
        said = _all("i took a loan from my bank of 5000 dollars to buy bitcoin, is this good")
        stats = beginner._moves("BTCUSDT")
        assert stats is not None
        assert f"{abs(stats.change):.0%} over the last year" in said
        assert f"{stats.worst_day:.1%}" in said
        assert f"{abs(stats.drawdown):.0%} below its high" in said
        if stats.change < 0:
            assert f"{5000 * (1 + stats.change):,.0f} today" in said
        assert "Data: Bitcoin, the last" in said

    def test_already_borrowed_gets_the_repay_advice(self) -> None:
        said = _all("i borrowed 5000 from my bank to buy crypto, good idea?")
        assert "giving it back is the safe move" in said

    def test_without_history_it_says_so_and_keeps_the_warning(self, no_history: None) -> None:
        said = _all("i took a loan of 5000 dollars to buy bitcoin")
        assert "could not read the price history" in said
        assert said.startswith("Bottom line: no — do not borrow")

    @pytest.mark.parametrize("text", [
        "i have a mortgage should i buy bitcoin",
        "how do i get a loan against my bitcoin",
        "what is a flash loan in defi, can i buy crypto with it",
        "can you borrow on bitget margin to buy more bitcoin",
        "do i report crypto when i apply for a mortgage",
    ])
    def test_other_debt_questions_are_not_claimed(self, text: str) -> None:
        got = beginner.early(text, [])
        assert got is None or "do not borrow money to buy" not in got[0]


class TestStranger:
    @pytest.mark.parametrize("text", [
        "somebody on telegram says send him 500 usdt and he doubles it, should i",
        "a guy says send him my btc and he will flip it into 2x, legit?",
        "someone told me deposit 200 usdt to his wallet and get 3x back",
        "send her 100 usdt and she triples it, should i",
    ])
    def test_it_is_a_scam(self, text: str) -> None:
        said = _all(text)
        assert said.startswith("Bottom line: it is a scam — do not send")
        assert "cannot be pulled back" in said

    def test_a_growth_question_is_not_a_scam_warning(self) -> None:
        got = beginner.early("what if my 500 usdt doubles in a year", [])
        assert got is None or "it is a scam" not in got[0]


class TestRecoveryTriggerIsTight:
    """Round 45 hostile, C8: two arithmetic questions got the recovery-scam answer."""

    @pytest.mark.parametrize("text", [
        "What does my portfolio return if BTC falls 10%?",
        "BTC 1 unit at 90000; ETH -2 units at 2500. What does my portfolio return if BTC falls "
        "10%?",
        "A coin went from 100 to 150 and then back to 100 - what is the total return percent? "
        "And if I lose 50% then gain 50% where do I end?",
        "what is the return percent on 150 vs 100",
    ])
    def test_arithmetic_is_not_a_scam(self, text: str) -> None:
        got = beginner.early(text, [])
        assert got is None or "scam" not in " ".join(got)

    @pytest.mark.parametrize("text", [
        "a recovery service says they can recover my stolen crypto for a fee",
        "someone dm'd me offering to get my money back if i pay upfront",
        "they will return my funds for 10% of the amount, is it real",
        "recovery agency wants a deposit first to trace my stolen coins",
    ])
    def test_a_real_recovery_offer_still_is(self, text: str) -> None:
        assert "almost certainly a scam" in _all(text)


class TestLeverageWording:
    def test_ten_x_frequency_is_counted(self) -> None:
        said = _all("is 10x leverage ok for a small account")
        stats = beginner._moves("BTCUSDT")
        assert stats is not None and stats.at_least(10) == 2
        assert "in a day often" not in said
        assert f"on 2 of the last {stats.days} days" in said
        assert "worst 7-day stretch" in said and "Data: Bitcoin" in said

    def test_a_routine_move_is_called_routine(self) -> None:
        said = _all("is 100x ok for me")
        assert "a routine day" in said

    def test_no_frequency_is_made_up_without_data(self, no_history: None) -> None:
        said = _all("is 10x leverage ok for a small account")
        assert "could not be read just now" in said and "often" not in said

    def test_stake_at_ten_x(self) -> None:
        said = _all("ok so what happens if i put 20 bucks at 10x")
        stats = beginner._moves("BTCUSDT")
        assert stats is not None
        assert "an ordinary day for most coins" not in said
        assert "$20 holds a $200 position" in said and "10.0% move" in said
        assert f"typical day moves it about {stats.typical:.1%}" in said
        assert f"${20 * 10 * stats.typical:,.2f}" in said

    @pytest.mark.parametrize("text", [
        "if i put $20 at 10x and it drops 5% what happens",
        "how much could i lose with $20 at 10x in a week",
        "what happens if i put 20 bucks at 10x on nvda",
        "what is 10x",
    ])
    def test_stake_reader_leaves_other_questions(self, text: str) -> None:
        got = beginner.early(text, [])
        assert got is None or "holds a $200 position" not in got[0]


class TestLeverageInAnotherLanguage:
    @pytest.mark.parametrize("text", [
        "10x kaldıraç ne demek", "kaldıraç nedir", "¿qué es el apalancamiento?",  # noqa: RUF001
        "o que é alavancagem", "was ist der Hebel beim Trading",
    ])
    def test_it_is_the_leverage_definition(self, text: str) -> None:
        said = _all(text)
        assert "Leverage is holding a position larger than the money put up" in said
        assert "custody" not in said and "never holds" not in said

    def test_the_stated_multiple_is_worked(self) -> None:
        assert "At 10x, a move of about 10.0%" in _all("10x kaldıraç ne demek")  # noqa: RUF001

    def test_the_word_alone_is_not_a_definition_question(self) -> None:
        assert beginner.early("kaldıraç ile ilgili bir şey", []) is None  # noqa: RUF001


class TestNewcomerBasics:
    def test_whats_bitcoin_and_why_is_it_a_scam(self) -> None:
        said = _all("whats bitcoin and why do ppl say its a scam??")
        assert "digital currency" in said and "not a scam" in said
        assert "headlines" not in said and "beta" not in said

    def test_usdt_safe_and_what_is_a_stablecoin(self) -> None:
        said = _all("is usdt safe? what is a stablecoin")
        assert "not safe the way a bank deposit is" in said and "not insured" in said
        assert "TerraUSD" in said

    def test_stock_or_the_bitget_token(self) -> None:
        said = _all("so is it safer to buy tsla stock or the bitget tsla thing")
        assert "neither is safer" in said and "TSLA" in said
        assert "not a registered shareholder" in said and "broker" in said

    def test_privacy_and_who_runs_it(self) -> None:
        said = _all("do you save my data or my wallet? who runs this site")
        assert "no login and no wallet connection" in said
        assert "server keeps no chat history" in said
        assert "github.com/Pratiikpy/argus-bitget" in said
        assert "not Bitget's own site" in said

    @pytest.mark.parametrize("text", [
        "does this site store my questions", "who runs this website",
    ])
    def test_either_half_alone(self, text: str) -> None:
        assert _said(text)

    def test_too_late_to_buy_bitcoin_leads_with_the_answer(self) -> None:
        said = _all("is it too late to buy bitcoin?")
        stats = beginner._moves("BTCUSDT")
        assert stats is not None
        assert said.startswith("Bottom line: nobody can know")
        assert "kurtosis" not in said and "beta" not in said and "R²" not in said
        assert "has fallen as much as" in said and "Data: Bitcoin, daily closes" in said

    @pytest.mark.parametrize("text", [
        "What is Bitcoin and should I invest my money in it?",
        "whats bitcoin and is it a good investment",
        "explain bitcoin, should i put 2000 dollars in?",
    ])
    def test_what_it_is_and_whether_to_buy(self, text: str) -> None:
        # the Hindi question restated in English got beta, kurtosis and QQQ shocks (round 45)
        said = _all(text)
        assert said.startswith("Bottom line: Bitcoin is a digital currency")
        assert "nobody can honestly tell you" in said and "Over the last year it went" in said
        assert "kurtosis" not in said and "beta" not in said and "Not advice." in said
        assert ("$2,000" in said) == ("2000" in text)

    def test_a_stock_is_a_share_of_the_company(self) -> None:
        said = _all("what is NVDA and should I buy it")
        assert said.startswith("Bottom line: NVDA is a share of Nvidia")
        assert "broker" in said and "on spot" not in said

    def test_what_and_whether_without_history(self, no_history: None) -> None:
        said = _all("what is ethereum and should i buy some")
        assert "could not be read just now" in said and "Over the last year" not in said
        assert said.startswith("Bottom line: Ether is the coin of Ethereum")
        assert "Not advice." in said

    def test_how_much_profit_shows_the_spread_of_past_years(self) -> None:
        said = _all("if i buy $100 of bitcoin how much profit will i make")
        assert said.startswith("Bottom line: nobody can tell you the profit in advance")
        assert "ended up and" in said and "a bad one (1 in 10)" in said
        assert "For $100 put in" in said and "kurtosis" not in said and "beta" not in said
        assert "overlapping one-year holding periods" in said

    def test_how_much_profit_in_rupiah_is_said_in_rupiah(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.market import fx_rates

        monkeypatch.setattr(fx_rates, "usd_rate", lambda code: fx_rates.FxRate(
            code, 17_913.0, "the ECB reference rate", "2026-10-05"))
        said = _all("If I buy 1 million rupiah of bitcoin, how much profit will I make?")
        assert "Read in US dollars: 1,000,000 rupiah = about $55.83" in said
        assert "For 1,000,000 rupiah put in" in said and "rupiah ($" in said
        assert "For $1 put in" not in said

    def test_how_much_profit_without_history(self, no_history: None) -> None:
        said = _all("how much money can i make if i put 500 in eth")
        assert "could not be read just now" in said and "ended up" not in said

    def test_how_much_could_i_lose_with_a_sum(self) -> None:
        # the landing page's own example (round 45): worst day, week, month and year on the sum
        said = _all("how much could I lose with $200 in bitcoin")
        assert said.startswith("Bottom line: on spot, the most you can lose is the $200 itself")
        assert "its worst day, $" in said and "its worst year, $" in said
        assert "beta" not in said and "QQQ" not in said and "Not advice." in said

    @pytest.mark.parametrize("text", [
        "how much would I lose if NVDA fell 10% and I hold $5k",
        "how much could I lose with $200 in bitcoin at 10x",
        "how much could I lose with $200",
    ])
    def test_a_scenario_leverage_or_no_name_goes_elsewhere(self, text: str) -> None:
        assert beginner._loss_with_amount(text) is None

    def test_too_late_to_start_investing_keeps_its_own_answer(self) -> None:
        assert "it is not too late" in _all("am i too old to start investing, i'm 45")

    def test_without_history_there_is_no_invented_drop(self, no_history: None) -> None:
        said = _all("is it too late to buy bitcoin?")
        assert "has fallen as much as" not in said and "could not be read" in said

    @pytest.mark.parametrize("text", [
        "tenho medo de perder tudo", "비트코인 지금 사도 돼요? 처음이라 무서워요",
        "i'm scared i'll lose everything", "tengo miedo de perder mi dinero",
        "j'ai peur de tout perdre", "ich habe Angst alles zu verlieren",
        "her şeyi kaybetmekten korkuyorum", "takut kehilangan semua uang",
    ])
    def test_fear_gets_one_plain_paragraph(self, text: str) -> None:
        got = _said(text)
        assert len(got) == 1 and len(got[0]) < 900
        assert "sensible reaction" in got[0] and "kurtosis" not in got[0]
        assert not re.search(r"\d+%", got[0]), "no number wall"

    @pytest.mark.parametrize("text", [
        "im scared of missing out on bitcoin", "how do i buy bitcoin, im nervous",
        "i lost 3000 dollars yesterday on a coin called pepe and i feel sick",
    ])
    def test_other_feelings_keep_their_own_answers(self, text: str) -> None:
        got = beginner.early(text, [])
        assert got is None or "sensible reaction" not in got[0]

    def test_is_it_like_gambling_names_no_coin(self) -> None:
        said = _all("is it like gambling")
        assert "DOGE" not in said and "business behind" not in said
        assert "not the asset but how it is used" in said

    def test_named_memecoin_gambling_is_not_claimed(self) -> None:
        got = beginner.early("is dogecoin gambling", [])
        assert got is None or "not the asset but how it is used" not in got[0]


class TestFollowUps:
    def test_safest_wallet_for_a_noob(self) -> None:
        said = _all("which one is safest for a total noob", ["what is a wallet"])
        assert "exchange wallet" in said and "recovery phrase" in said

    def test_safest_without_a_wallet_topic_is_not_claimed(self) -> None:
        assert beginner.early("which one is safest for a total noob", ["bitcoin price"]) is None

    @pytest.mark.parametrize("text", [
        "what does that mean in normal words", "explícamelo más fácil",
        "say that in plain english", "i don't understand that",
    ])
    def test_normal_words_after_a_term(self, text: str) -> None:
        said = _all(text, ["what does beta mean"])
        assert "how strongly a price follows the overall market" in said
        assert "Example:" in said

    def test_normal_words_picks_the_term_asked(self) -> None:
        said = _all("what does that mean in normal words", ["what is kurtosis"])
        assert "surprisingly big move" in said

    def test_risky_or_not_keeps_the_subject(self) -> None:
        said = _all("so is it risky or not",
                    ["how risky is TSLA", "what does beta mean for tsla"])
        stats = beginner._moves("TSLAUSDT")
        assert stats is not None
        assert "TSLA's worst single day" in said and "bitcoin" not in said.lower()
        assert f"{stats.worst_day:.1%}" in said
        assert "Data: TSLA" in said

    def test_two_subjects_are_not_guessed(self) -> None:
        assert beginner.early("so is it risky or not", ["eth or sol"]) is None

    def test_a_topic_of_this_layer_keeps_its_own_safe_answer(self) -> None:
        said = _all("so is it safe", ["what is a stablecoin"])
        assert "TerraUSD" in said


class TestEverythingStillReachesItsOwnAnswer:
    @pytest.mark.parametrize("text", [
        "what is the price of bitcoin", "how is NVDA doing today",
        "what is my portfolio return if BTC falls 10%",
    ])
    def test_price_and_risk_questions_fall_through(self, text: str) -> None:
        assert beginner.early(text, []) is None


def test_should_i_buy_then_what_is_it() -> None:
    # the Hindi question restated by the model put the two questions the other way round
    said = " ".join(beginner._what_and_should("Should I buy Bitcoin? What is it?") or [])
    assert said.startswith("Bottom line: Bitcoin is a digital currency")
