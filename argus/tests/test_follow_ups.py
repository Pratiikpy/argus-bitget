"""A follow-up is read against the question before it.

Gathered on 2026-09-27 from the three audit-round files (2026-09-25), which grouped tests by
the day a defect was found rather than by what they pin (audit finding 170).
"""

from __future__ import annotations

import pytest

from argus.lui import research
from argus.lui.research import ResearchKind


def test_a_new_book_reruns_the_previous_question() -> None:
    request = research.follow_up("actually make it 70% NVDA 30% AAPL",
                                 ["what does a 10% Nasdaq drop do to 50% NVDA 50% AAPL"])
    assert request is not None and request.kind is ResearchKind.STRESS
    assert request.book == {"NVDAUSDT": 0.7, "AAPLUSDT": 0.3}
    assert request.shock_pct == -10.0


def test_is_that_bullish_reads_the_previous_name() -> None:
    request = research.follow_up("is that bullish?", ["ETH open interest"])
    assert request is not None and request.kind is ResearchKind.SENTIMENT
    assert request.symbols == ("ETHUSDT",)


def test_bare_follow_ups_and_why_are_recognised() -> None:
    from argus.lui.server import _BARE_WHY

    assert research.BARE_FOLLOW.match("what about that one?")
    assert research.BARE_FOLLOW.match("tell me more")
    assert not research.BARE_FOLLOW.match("what about COIN?")
    assert _BARE_WHY.match("why?") and _BARE_WHY.match("how come")
    assert not _BARE_WHY.match("why did BTC fall?")


def test_a_ratio_rebooks_the_last_two_names() -> None:
    request = research.follow_up("actually make it 60/40",
                                 ["whats my drawdown if btc drops 20%", "and ETH?"])
    assert request is not None
    assert request.book == pytest.approx({"BTCUSDT": 0.6, "ETHUSDT": 0.4})


def test_compare_it_to_names_the_earlier_symbol() -> None:
    request = research.follow_up("compare it to BNB", ["sup with SOL today"])
    assert request is not None and request.kind is ResearchKind.COMPARE
    assert request.symbols == ("SOLUSDT", "BNBUSDT")


def test_a_chain_of_follow_ups_resolves_to_the_last_name() -> None:
    request = research.follow_up("is that good?", ["what is the BTC funding rate rn?", "and ETH?"])
    assert request is not None and request.symbols == ("ETHUSDT",)


def test_i_meant_swaps_the_name_and_keeps_the_question() -> None:
    found = research.resolved_previous(["price of bitcoin?", "and dogecoin?"], "")
    assert found is not None and found[0] == "price of bitcoin?"
    request = research.follow_up("actually i meant ethereum", ["price of bitcoin?"])
    assert request is not None and request.symbols == ("ETHUSDT",)


def test_a_new_shock_size_reruns_the_stress() -> None:
    request = research.follow_up("what about a 20% drop instead",
                                 ["whats my drawdown if btc drops 10%"])
    assert request is not None and request.kind is ResearchKind.STRESS
    assert request.shock_pct == -20.0


def test_an_asset_class_follow_up_on_macro() -> None:
    request = research.follow_up("how does that affect crypto",
                                 ["whats the fed gonna do with rates"])
    assert request is not None and request.kind is ResearchKind.MACRO
    assert request.symbols == ("BTCUSDT",)


def test_which_one_and_better_or_worse_are_follow_ups() -> None:
    from argus.lui.server import _BETTER_WORSE, _WHICH_ONE

    assert _WHICH_ONE.match("which one is more volatile")
    assert _BETTER_WORSE.match("is that better or worse than before")
