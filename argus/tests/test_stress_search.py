"""The most likely way a position breaks its loss limit (research/harvest/16-adaptive-stress-
testing.md): probabilities from observed moves and recorded books, the exact most likely failure
by enumeration, and the random baseline never beating it."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal as D

import pytest

from argus.desk.stress import Move, StressError
from argus.desk.stress_search import random_search, search
from argus.market.depth import Level, OrderBook

T0 = datetime(2026, 9, 1, tzinfo=UTC)


def moves(pcts: list[str]) -> list[Move]:
    return [Move(at=T0 + timedelta(hours=i), pct=D(p), phase="unknown")
            for i, p in enumerate(pcts)]


def book(i: int, bid_qty: str) -> OrderBook:
    return OrderBook(symbol="X", fetched_at=T0 + timedelta(days=i),
                     bids=(Level(D("99.9"), D(bid_qty)), Level(D("99"), D("1000"))),
                     asks=(Level(D("100.1"), D("1000")),))


# Ten moves, one of each size; two books: one deep (the whole order at the touch), one thin.
MOVES = moves(["-12", "-8", "-6", "-4.5", "-3", "-1", "0", "1", "2", "3"])
BOOKS = [book(0, "1000"), book(1, "1")]


def test_the_failure_probability_counts_failing_pairs() -> None:
    result = search("X", MOVES, BOOKS, quantity=D("10"), tolerance_pct=D("5"))
    assert result.evaluated == 20
    # Fails in both books: -12, -8, -6 (3 x 2). The thin book costs ~84bps + 6bps fee, so
    # -4.5% also fails there (4.5 + 0.955 x 0.90 > 5): 7 of 20.
    assert result.failure_probability == D("7") / D("20")
    assert result.most_likely is not None
    # The likeliest failure: the least extreme failing move with the likeliest book.
    assert result.most_likely.move_pct == D("-6")
    assert result.worst is not None and result.worst.move_pct == D("-12")


def test_a_book_that_cannot_absorb_the_exit_is_a_failure_on_its_own() -> None:
    result = search("X", moves(["1", "2"]), [book(0, "1")], quantity=D("5000"),
                    tolerance_pct=D("50"))
    assert result.failure_probability == 1
    assert result.most_likely is not None and not result.most_likely.exitable
    assert "cannot absorb" in result.sentence()


def test_a_short_fails_on_rises() -> None:
    result = search("X", MOVES, BOOKS[:1], quantity=D("10"), tolerance_pct=D("2.5"),
                    long=False)
    assert result.most_likely is not None and result.most_likely.move_pct == D("3")


def test_random_search_never_beats_enumeration() -> None:
    exact = search("X", MOVES, BOOKS, quantity=D("10"), tolerance_pct=D("5")).most_likely
    assert exact is not None
    for seed in range(50):
        found = random_search("X", MOVES, BOOKS, quantity=D("10"), tolerance_pct=D("5"),
                              budget=5, seed=seed)
        assert found is None or found.probability <= exact.probability


def test_nothing_to_search_refuses() -> None:
    with pytest.raises(StressError):
        search("X", [], BOOKS, quantity=D("1"), tolerance_pct=D("5"))
    with pytest.raises(StressError):
        search("X", MOVES, [], quantity=D("1"), tolerance_pct=D("5"))


def test_the_research_task_states_the_searched_probability() -> None:
    from types import SimpleNamespace

    from argus.desk.research import stress_step

    searched = search("X", MOVES, BOOKS, quantity=D("10"), tolerance_pct=D("5"))
    report = SimpleNamespace(results=[object()], failures=[], unexitable=[])
    finding = stress_step(report, searched)
    assert "Chance of losing more than 5% over the horizon: 35.0%" in finding.headline
    assert finding.detail["searched"]["failure_probability"] == "0.350000"
