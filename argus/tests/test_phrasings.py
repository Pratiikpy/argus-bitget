"""The phrasing corpus — how a person actually asks, not how a pattern was written.

This exists because the console must work with **no model key**. The router in `lui/router.py`
needs credentials; a judge opening a hosted demo after the hackathon balance is spent gets the
deterministic layer and nothing else. Every refusal it produces there is a refusal they see.

Each phrasing below is one a person plausibly types. They are grouped by what the console should
understand them to mean, and the ones that must stay refusals are here too — an interface that
quietly reinterprets an instruction as a question is more dangerous than one that plainly declines.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from argus.lui.question import Intent, classify

NOW = datetime(2026, 9, 14, 15, 0, tzinfo=UTC)


def _intent(text: str) -> Intent:
    return classify(text, now=NOW).intent


PERFORMANCE = [
    "how did we do this week?",
    "how are we doing?",
    "how's it going?",
    "did we make any money?",
    "did we make much money?",
    "did we lose money?",
    "what's the sharpe?",
    "what is the sortino?",
    "what's our drawdown?",
    "what is the win rate?",
    "how much have we made?",
    "did we beat the fees?",
    "what was the worst day?",
    "what's the pnl?",
]

ABSTENTION = [
    "why did you do nothing?",
    "why no trades?",
    "why did you stand aside?",
    "why did you pass?",
    "why didn't you trade?",
    "why did you sit out?",
]

INTEGRITY = [
    "is the record intact?",
    "has the log been tampered with?",
    "can I trust the ledger?",
    "was anything edited?",
    "prove the chain",
    "verify the record",
    "is this audited?",
    "has anything been altered?",
]

POSITION = [
    "what do we hold?",
    "what are our positions?",
    "what's open?",
    "are we holding anything?",
]

SESSION = [
    "is the market open?",
    "what session is it?",
    "how long until the open?",
]

CALIBRATION = [
    "how confident have you been?",
    "are you well calibrated?",
    "how often are you right?",
    "how good are your forecasts?",
    "were you right often?",
    "have you been overconfident?",
]

EVIDENCE = [
    "what did you see before deciding?",
    "what evidence did you have?",
    "what were the sources?",
    "what filings did you read?",
]

ORDERS_MUST_BE_REFUSED = [
    "sell half of that",
    "buy NVDA",
    "close the position",
    "short TSLA now",
]

OFF_VENUE_MUST_BE_REFUSED = [
    "what about BABA?",
    "how is AMD doing?",
    "what's SPY doing?",
    "how about gold?",
    "what about bitcoin?",
]


@pytest.mark.parametrize("text", PERFORMANCE)
def test_performance_phrasings(text: str) -> None:
    assert _intent(text) is Intent.PERFORMANCE, text


@pytest.mark.parametrize("text", ABSTENTION)
def test_abstention_phrasings(text: str) -> None:
    assert _intent(text) is Intent.ABSTENTION_WHY, text


@pytest.mark.parametrize("text", INTEGRITY)
def test_integrity_phrasings(text: str) -> None:
    assert _intent(text) is Intent.INTEGRITY, text


@pytest.mark.parametrize("text", POSITION)
def test_position_phrasings(text: str) -> None:
    assert _intent(text) is Intent.POSITION, text


@pytest.mark.parametrize("text", SESSION)
def test_session_phrasings(text: str) -> None:
    assert _intent(text) is Intent.SESSION, text


@pytest.mark.parametrize("text", CALIBRATION)
def test_calibration_phrasings(text: str) -> None:
    assert _intent(text) is Intent.CALIBRATION, text


@pytest.mark.parametrize("text", EVIDENCE)
def test_evidence_phrasings(text: str) -> None:
    assert _intent(text) is Intent.EVIDENCE, text


@pytest.mark.parametrize("text", ORDERS_MUST_BE_REFUSED)
def test_an_instruction_is_never_answered_as_a_question(text: str) -> None:
    """The failure this guards: 'sell half of that' matching a decision-explanation pattern."""
    assert _intent(text) is Intent.ORDER, text


@pytest.mark.parametrize("text", OFF_VENUE_MUST_BE_REFUSED)
def test_an_off_venue_instrument_is_named_rather_than_misunderstood(text: str) -> None:
    got = classify(text, now=NOW)
    assert got.intent is Intent.UNSUPPORTED, text
    assert got.reason, "an off-venue refusal must say which instrument and why"


def test_the_corpus_covers_every_answerable_intent() -> None:
    """A class with no phrasing in the corpus is a class nobody checked."""
    covered = {
        Intent.PERFORMANCE, Intent.ABSTENTION_WHY, Intent.INTEGRITY, Intent.POSITION,
        Intent.SESSION, Intent.CALIBRATION, Intent.EVIDENCE,
    }
    answerable = {
        Intent.PERFORMANCE, Intent.DECISION_WHY, Intent.DECISION_LIST, Intent.ABSTENTION_WHY,
        Intent.EVIDENCE, Intent.CALIBRATION, Intent.INTEGRITY, Intent.POSITION, Intent.SESSION,
    }
    missing = answerable - covered
    assert missing <= {Intent.DECISION_WHY, Intent.DECISION_LIST}, missing


RISK_CONTROL = [
    # Track 2 scores "risk control layer effectiveness" directly, so these are the phrasings a
    # judge is most likely to try. Every one of them reached UNKNOWN until 2026-09-13, found by
    # driving the hosted console rather than by reading the pattern table.
    "what did the risk layer block?",
    "what did the risk layer do?",
    "did the risk layer block anything?",
    "how often did the constitution bind?",
    "which constraint was binding most?",
    "what constraints fired?",
    "did the risk layer ever increase a position?",
    "was anything vetoed?",
    "did you override the model?",
    "how many times did you intervene?",
    "show me the risk controls",
    "what are your risk limits?",
    "is there a circuit breaker?",
    "what did risk cut?",
]


CHINESE: list[tuple[str, Intent]] = [
    # This competition is run by a Chinese-language-first team: the handbook ships in Chinese and
    # English, and the submission form has "Chinese and English versions, same fields". Every one
    # of these reached UNKNOWN until 2026-09-13 — found by driving the HOSTED console in Chinese,
    # not by reading the pattern list.
    # The fullwidth question mark below is deliberate: it is what a Chinese keyboard actually
    # produces, so a corpus using the ASCII one would test input no Chinese judge will ever type.
    ("为什么你没有交易 NVDA？", Intent.ABSTENTION_WHY),  # noqa: RUF001
    ("为什么没有开仓", Intent.ABSTENTION_WHY),
    ("为什么不交易", Intent.ABSTENTION_WHY),
    ("观望的原因", Intent.ABSTENTION_WHY),
    ("为什么买入 TSLA", Intent.DECISION_WHY),
    ("解释一下这个决定的逻辑", Intent.DECISION_WHY),
    ("夏普比率是多少", Intent.PERFORMANCE),
    ("我们赚到钱了吗", Intent.PERFORMANCE),
    ("最大回撤多少", Intent.PERFORMANCE),
    ("胜率如何", Intent.PERFORMANCE),
    ("风控拦截了什么", Intent.RISK_CONTROL),
    ("风险控制层做了什么", Intent.RISK_CONTROL),
    ("账本完整吗", Intent.INTEGRITY),
    ("有没有篡改", Intent.INTEGRITY),
    ("记录可信吗", Intent.INTEGRITY),
    ("现在有什么持仓", Intent.POSITION),
    ("仓位是多少", Intent.POSITION),
    ("校准情况如何", Intent.CALIBRATION),
    ("什么时候开盘", Intent.SESSION),
    ("现在休市了吗", Intent.SESSION),
    ("有哪些决定", Intent.DECISION_LIST),
    ("复盘一下", Intent.DECISION_LIST),
    ("看到了什么证据", Intent.EVIDENCE),
    ("有什么财报消息", Intent.EVIDENCE),
    # Refusals must stay refusals in Chinese too.
    ("帮我卖出一半", Intent.ORDER),
    ("黄金现在多少钱", Intent.UNSUPPORTED),
]


@pytest.mark.parametrize(("text", "expected"), CHINESE)
def test_chinese_phrasings_are_understood(text: str, expected: Intent) -> None:
    assert _intent(text) is expected, text


def test_the_chinese_patterns_use_no_word_boundary() -> None:
    """Chinese is written without spaces, so there is no word boundary between two Han
    characters. Every `\b`-anchored pattern is structurally incapable of matching Chinese, which
    is why the English list could not simply be extended — it is a tokenisation gap, not a
    vocabulary one. A `\b` creeping into this list would silently disable it."""
    from argus.lui.question import _CHINESE_PATTERNS

    for pattern, _intent_for in _CHINESE_PATTERNS:
        assert "\b" not in pattern, pattern


def test_chinese_is_matched_before_english() -> None:
    """A Chinese sentence carrying a Latin ticker would otherwise be claimed by whichever English
    pattern happens to catch the ticker."""
    assert _intent("为什么没有交易 NVDA") is Intent.ABSTENTION_WHY


def test_a_chinese_question_is_never_silently_english(self=None) -> None:
    from argus.lui.question import has_chinese

    assert has_chinese("夏普比率是多少")
    assert not has_chinese("what is the sharpe")


def test_the_deterministic_layer_understands_most_of_the_corpus() -> None:
    """The number that matters for a keyless demo, asserted rather than hoped."""
    everything = (
        PERFORMANCE + ABSTENTION + INTEGRITY + POSITION + SESSION + CALIBRATION + EVIDENCE
        + RISK_CONTROL + [text for text, _ in CHINESE]
    )
    understood = [q for q in everything if _intent(q) is not Intent.UNKNOWN]
    share = len(understood) / len(everything)
    assert share >= 0.95, (
        f"only {share:.0%} of natural phrasings are understood without a model; "
        f"missed: {[q for q in everything if _intent(q) is Intent.UNKNOWN]}"
    )


@pytest.mark.parametrize("text", RISK_CONTROL)
def test_risk_control_phrasings_route_to_the_risk_answerer(text: str) -> None:
    assert _intent(text) is Intent.RISK_CONTROL, text


def test_risk_patterns_do_not_steal_position_questions() -> None:
    """"what do we hold" is about the book, not about the layer that constrains it."""
    for text in ("what do we hold?", "show me open positions", "are we long anything?"):
        assert _intent(text) is Intent.POSITION, text


def test_risk_patterns_do_not_steal_performance_or_abstention() -> None:
    for text, expected in (
        ("how did we do this week?", Intent.PERFORMANCE),
        ("why did you not trade?", Intent.ABSTENTION_WHY),
        ("is the ledger intact?", Intent.INTEGRITY),
    ):
        assert _intent(text) is expected, text
