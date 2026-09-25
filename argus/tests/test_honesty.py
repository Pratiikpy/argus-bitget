"""The honesty layer (`lui/honesty.py`): questions no console can answer as asked, detected
offline, and the answerable questions beside them left alone."""

from __future__ import annotations

from datetime import date

import pytest

from argus.lui import honesty

TODAY = date(2026, 9, 26)


@pytest.mark.parametrize(("question", "cause"), [
    ("what's my Bitget account balance", honesty.PRIVATE_ACCOUNT),
    ("show me my open orders on Bitget", honesty.PRIVATE_ACCOUNT),
    ("what's my unrealised P&L on my Bitget futures account", honesty.PRIVATE_ACCOUNT),
    ("how much margin do I have left on my account", honesty.PRIVATE_ACCOUNT),
    ("what did I pay for my NVDA shares", honesty.PRIVATE_ACCOUNT),
    ("我的比特币仓位现在盈利多少", honesty.PRIVATE_ACCOUNT),
    ("what are Citadel's exact positions in NVDA today", honesty.OTHERS_POSITIONS),
    ("which Bitget users are long TSLA right now", honesty.OTHERS_POSITIONS),
    ("what will NVDA close at next Friday", honesty.FUTURE_PRICE),
    ("what price will TSLA reach in 2030", honesty.FUTURE_PRICE),
    ("where will the S&P 500 be at year end", honesty.FUTURE_PRICE),
    ("will MSTR hit $1000 by December", honesty.FUTURE_PRICE),
    ("tell me the exact bottom for SOL this cycle", honesty.FUTURE_PRICE),
    ("比特币下周五的收盘价是多少", honesty.FUTURE_PRICE),
    ("what did the desk decide on 1 March 2026", honesty.BEFORE_DATA),
    ("how did the desk's calls do during the March 2020 crash", honesty.BEFORE_DATA),
    ("how risky is this stock?", honesty.AMBIGUOUS),
    ("which of the two is riskier?", honesty.AMBIGUOUS),
])
def test_an_unanswerable_question_is_named_by_its_true_cause(question: str, cause: str) -> None:
    found = honesty.detect(question, today=TODAY)
    assert found is not None and found.cause == cause, (question, found)


@pytest.mark.parametrize("question", [
    "what did Coinbase stock trade at in 2015",
    "what was the MSTR perp's price on Bitget on 1 June 2025",
    "COIN's average annual return over the last 15 years",
    "SOL's 10-year Sharpe ratio",
])
def test_a_past_date_or_a_long_span_is_caught_before_any_live_quote(question: str) -> None:
    found = honesty.detect(question, today=TODAY)
    assert found is not None and found.cause in (
        honesty.PAST_PRICE, honesty.BEFORE_DATA, honesty.HORIZON, honesty.LONG_HORIZON)


@pytest.mark.parametrize("question", [
    "what is NVDA trading at",
    "what's my P&L if NVDA drops 10%? I hold 60% NVDA 40% AAPL",
    "my book is 50% NVDA and 50% AAPL, how risky is it",
    "is my account too risky with 60% NVDA and 40% TSLA",
    "hedge my crypto",
    "how should I split a $50k order in NVDA",
    "what are the odds NVDA is up this week",
    "how has BTC done over the last year",
    "has COIN been here before",
    "how risky is JNJ",
    "is TSLA riskier than NVDA",
    "what did the desk decide on 2026-09-20",
    "how accurate were this desk's past predictions",
    "风险层拦截了什么",
    "英伟达现在值得买吗",
])
def test_an_answerable_question_is_left_alone(question: str) -> None:
    assert honesty.detect(question, today=TODAY) is None


def test_an_order_instruction_is_prefixed_and_a_question_about_one_is_not() -> None:
    assert honesty.order_prefix("set a limit order to buy gold at 4000")
    assert honesty.order_prefix("go long BTC with 5x leverage right now")
    assert honesty.order_prefix("how should I split a $50k order in NVDA") is None
    assert honesty.order_prefix("what would a 5x long in BTC risk") is None


def test_an_off_topic_question_with_a_pronoun_is_not_ambiguous() -> None:
    assert honesty.detect("help me write a sql query to join these two tables",
                          today=TODAY) is None
