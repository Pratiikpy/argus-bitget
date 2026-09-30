"""Round 12 of the §27 audits (2026-09-30): a judge's deep pass on the personalized thesis.

Each test pins one finding (tracker rows 530-535). Live sources are replaced by the rows they
return, so nothing here touches the network.
"""

from __future__ import annotations

from typing import Any

import pytest

_BULL = ("My bull case for BTC: ETF inflows are accelerating, the halving supply shock hasn't "
         "fully played out, and the Fed is about to ease. Which of these is strongest?")


def test_a_bull_case_is_split_into_its_reasons_without_the_asking() -> None:
    from argus.lui.thesis import Kind, reasons

    assert [(r.text, r.kind) for r in reasons(_BULL)] == [
        ("ETF inflows are accelerating", Kind.FLOWS),
        ("the halving supply shock hasn't fully played out", Kind.OTHER),
        ("the Fed is about to ease", Kind.MACRO)]


def test_a_bull_case_and_its_revision_reach_the_thesis_tester() -> None:
    from argus.lui.thesis_answer import REVISE, asks

    assert asks(_BULL)
    assert REVISE.search("Actually scratch the ETF point — assume ETF flows go negative instead. "
                         "Does that change your view?")


_FLOWS = {"funds": {"BTC": {"date": "2026-09-29", "net_inflow_usd": 66_194_702.4,
                            "five_day_usd": 769_353_764.43, "streak_days": 9}}}


@pytest.mark.parametrize(("claim", "verdict"), [
    ("ETF inflows are accelerating", "contradicted"),   # $66m against a $154m daily pace
    ("ETF inflows are strong", "supported"),            # five days net in
    ("ETF outflows are growing", "supported"),          # read as the latest day below the pace
    ("ETF outflows are hitting BTC", "contradicted"),   # five days were in, not out
])
def test_etf_flow_claims_read_the_spot_etf_record(claim: str, verdict: str) -> None:
    from argus.lui.thesis import Kind, Reason, _flows

    tested = _flows(Reason(claim, Kind.FLOWS), _FLOWS, "BTC")
    assert tested.result.value == verdict
    assert "SoSoValue" in tested.line
    assert _flows(Reason(claim, Kind.FLOWS), _FLOWS, "SOL").result.value == "not tested"


def test_a_stated_price_far_from_the_live_one_fails_as_a_premise() -> None:
    from argus.lui.thesis import price_premise

    failed = price_premise("I think BTC is cheap with spot around $109k", "BTC", 83_410.0)
    assert failed is not None and failed.result.value == "contradicted"
    assert "31% below the price the thesis states" in failed.line
    assert price_premise("BTC is cheap around $85k", "BTC", 83_410.0) is None  # within 10%
    assert price_premise("BTC is cheap", "BTC", 83_410.0) is None


def test_a_backward_looking_relative_claim_is_settled_by_the_record() -> None:
    from argus.lui.thesis import Kind, Reason, _relative

    found = {"names": ["SOLUSDT", "ETHUSDT"], "SOLUSDT": {7: 0.049, 30: 0.161, 89: 0.471},
             "ETHUSDT": {7: 0.006, 30: 0.084, 89: 0.544}}
    past = _relative(Reason("SOL has been outperforming ETH over the past month",
                            Kind.RELATIVE), found)
    assert past.result.value == "supported" and "Over 30 days" in past.line
    week = _relative(Reason("SOL beat ETH over the past week", Kind.RELATIVE), found)
    assert week.result.value == "supported" and "Over 7 days" in week.line
    forward = _relative(Reason("SOL will outrun ETH", Kind.RELATIVE), found)
    assert forward.result.value == "not measurable"


def test_a_coins_valuation_is_read_on_its_mayer_multiple(monkeypatch: Any) -> None:
    from argus.lui import thesis

    monkeypatch.setattr(thesis, "mayer_multiple",
                        lambda symbol: {"multiple": 0.82, "days": 700, "below": 0.12,
                                        "price": 60_000.0})
    cheap = thesis._crypto_valuation(thesis.Reason("BTC is cheap", thesis.Kind.VALUATION), "BTC")
    assert cheap.result is thesis.Result.SUPPORTED and "0.82 times its 200-day average" in \
        cheap.line
    rich = thesis._crypto_valuation(thesis.Reason("BTC is in a bubble", thesis.Kind.VALUATION),
                                    "BTC")
    assert rich.result is thesis.Result.CONTRADICTED


def test_a_macro_question_naming_a_contract_is_that_contracts_task() -> None:
    from argus.lui.task import Reading, read_question

    reading = read_question("Should I go long ETH into next week's FOMC decision?")
    assert isinstance(reading, Reading) and reading.name == "ETHUSDT"
    assert isinstance(read_question("what economic events matter this week"), str)


@pytest.mark.parametrize("asked", [
    "do you now think ETH is a stronger buy than SOL?",
    "is NVDA a better investment than AMD",
])
def test_a_better_buy_between_two_names_is_a_comparison(asked: str) -> None:
    from argus.lui.research import detect
    from argus.lui.research.kinds import ResearchKind

    found = detect(asked)
    assert found is not None and found.kind is ResearchKind.COMPARE


# --- trader memory, on the cases mem0 won (eval/memory_comparison.py) -----------------------------


@pytest.mark.parametrize(("said", "kinds"), [
    ("What is my risk tolerance? I think I can handle it but my max loss should probably be "
     "around 10%", [("max_loss", "0.1")]),
    ("How do I size this position? I usually hold for weeks and I cant lose more than 12% total",
     [("max_loss", "0.12"), ("horizon", "168")]),
    ("I am saving up for a house down payment next year so I want to be careful",
     [("goal", "a house down payment next year")]),
    ("Im trying to retire early so every trade needs to pull its weight",
     [("goal", "retire early")]),
    ("what if I cant lose more than 10%?", []),
])
def test_facts_after_a_question_and_life_goals_are_kept(said: str, kinds: list[Any]) -> None:
    from argus.lui.memory import extract

    assert [(f.kind, f.value) for f in extract(said)] == kinds


def test_a_correction_inside_one_message_keeps_the_later_fact_and_says_what_it_replaced() -> None:
    from argus.lui.memory import extract

    [fact] = extract("I hold for weeks normally -- well actually for this one Ill hold for months")
    assert (fact.kind, fact.value) == ("horizon", "720")
    assert "earlier in the same message" in fact.replaces


# --- the hostile review (tracker 537-544) ---------------------------------------------------------


def test_a_day_follow_up_swaps_the_day_in_the_question_it_follows() -> None:
    from argus.lui.server import _DAY_PHRASE

    assert _DAY_PHRASE.subn("on Thursday", "What happened in BTC on Monday?", count=1) == (
        "What happened in BTC on Thursday?", 1)


def test_no_reason_is_dropped_and_no_number_is_split() -> None:
    from argus.lui.thesis import Kind, reasons

    three = reasons("I think BTC rises because ETF inflows are accelerating, institutional "
                    "adoption is growing, and the halving cut new supply. Test my thesis. Which of "
                    "these is strongest?")
    assert [r.text for r in three] == ["ETF inflows are accelerating",
                                       "institutional adoption is growing",
                                       "the halving cut new supply"]
    priced = reasons("Bitcoin is at $200,000 right now. I think it keeps rising because ETF "
                     "inflows are accelerating.")
    assert [(r.text, r.kind) for r in priced] == [("it keeps rising", Kind.MOMENTUM),
                                                  ("ETF inflows are accelerating", Kind.FLOWS)]


def test_a_coins_institutional_adoption_reads_its_spot_etfs() -> None:
    from argus.lui.thesis import Kind, Reason, _etf_adoption

    tested = _etf_adoption(Reason("institutional adoption is growing", Kind.ACTIVITY),
                           {"funds": {"BTC": {"date": "2026-09-29", "five_day_usd": 7.69e8,
                                              "net_assets_usd": 1.08e11}}}, "BTC")
    assert tested.result.value == "supported" and "hold $108bn" in tested.line


@pytest.mark.parametrize("asked", ["How many Skills do you have?", "how many bitget skills"])
def test_a_count_of_the_skills_is_the_skills_answer(asked: str) -> None:
    from argus.lui import skills_explain

    assert skills_explain.asks(asked)


@pytest.mark.parametrize("asked", ["Is my BTC held on ARGUS insured?", "is my account insured",
                                   "is my money safe with you"])
def test_every_custody_phrasing_gets_the_custody_answer(asked: str) -> None:
    from argus.lui.newcomer import reply

    answered = reply(asked)
    assert answered is not None and "never holds, moves or touches money" in answered.lines[0]


def test_a_percentage_fall_is_set_against_a_dollar_loss() -> None:
    from argus.lui.research.sizing import loss_compare

    bare = loss_compare("If BTC drops 50 percent, is that worse than losing 10000 dollars?",
                        "BTCUSDT")
    assert bare is not None and "any holding above $20,000" in bare[0][0]
    held = loss_compare("I have $30,000 of BTC. If it falls 50%, is that worse than losing $10k?",
                        "BTCUSDT")
    assert held is not None and held[0][0] == ("Bottom line: on your $30,000, a 50% fall in BTC "
                                               "costs $15,000 — more than $10,000.")


# --- the first-time user (tracker 545-550) -------------------------------------------------------


@pytest.mark.parametrize(("asked", "name"), [
    ("what does slippage men", "slippage"),
    ("whats the diff between spot and futures", "spot and futures"),
])
def test_a_beginners_typos_still_reach_the_definition(asked: str, name: str) -> None:
    from argus.lui.concepts import concept_asked
    from argus.lui.research import research_symbols

    found = concept_asked(asked, research_symbols(asked)[0])
    assert found is not None and found.name == name


def test_withdrawing_is_answered_as_bitgets_not_the_consoles() -> None:
    from argus.lui.newcomer import reply

    answered = reply("how do I withdraw my money")
    assert answered is not None and "withdrawals happen on Bitget" in answered.lines[0]


def test_are_you_sure_is_a_follow_up() -> None:
    from argus.lui.server import _WHY_THAT

    for asked in ("are you sure?", "you sure", "is that right?", "how confident are you"):
        assert _WHY_THAT.match(asked), asked


def test_a_book_stated_in_dollars_is_sized_at_its_total() -> None:
    from decimal import Decimal

    from argus.lui.research import detect

    found = detect("I have $600 in SOL and $400 in TSLA, am I too risky?")
    assert found is not None and found.notional == Decimal("1000.0")


def test_the_same_question_asked_of_different_holdings() -> None:
    from argus.lui.server import _INSTEAD

    found = _INSTEAD.match("what if it was 600 in SOL and 400 in AAPL instead")
    assert found is not None and found.group("book") == "600 in SOL and 400 in AAPL"


# --- memory read by the model and checked by code (tracker 536) ----------------------------------


class _Model:
    def __init__(self, facts: list[dict[str, str]]) -> None:
        self.facts = facts
        self.calls = 0

    def complete_json(self, messages: list[dict[str, Any]], **_: Any) -> dict[str, Any]:
        self.calls += 1
        return {"facts": self.facts}


def test_the_models_facts_are_kept_only_when_the_message_carries_them() -> None:
    from argus.lui.memory_model import extract

    said = "honestly I cant go past a 12 percent drawdown and my friend never holds weekends"
    model = _Model([
        {"kind": "max_loss", "value": "12", "quote": "I cant go past a 12 percent drawdown"},
        {"kind": "max_loss", "value": "20", "quote": "I cant go past a 12 percent drawdown"},
        {"kind": "horizon", "value": "48", "quote": "my friend never holds weekends"},
        {"kind": "capital", "value": "50000", "quote": "I have $50,000"},
    ])
    kept = extract(said, model)
    assert [(f.kind, f.value) for f in kept] == [("max_loss", "0.12")]


def test_no_call_is_spent_on_a_plain_research_question() -> None:
    from argus.lui.memory_model import extract

    model = _Model([])
    assert extract("what is NVDA doing today", model) == [] and model.calls == 0
