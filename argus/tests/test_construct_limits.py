"""A book built under the limits a trader states in words: the reading of those limits, and the
book `lui/research/book._within_limits` returns under them
(research/harvest/46-cvxpy-allocation.md)."""

from __future__ import annotations

import random

import pytest

from argus.lui.research.book import _equal_risk_weights, _within_limits
from argus.lui.research.parse import stated_limits


@pytest.mark.parametrize(("text", "expected"), [
    ("build me a tech portfolio with at most 4 names, no more than 35% in any one",
     (0.35, 4, None)),
    ("Build a semis basket, max 30% each, only three stocks", (0.3, 3, None)),
    ("build a book, cap each at 20%, up to 5 positions", (0.2, 5, None)),
    ("make me a crypto portfolio where no single coin is above 50% of risk", (None, None, 0.5)),
    ("construct a portfolio of tech, no name more than 25% of the risk, 40% max per name",
     (0.4, None, 0.25)),
    ("give me a diversified portfolio with at most 40% of risk in any one holding",
     (None, None, 0.4)),
])
def test_stated_limits_are_read(text: str, expected: tuple[object, ...]) -> None:
    assert stated_limits(text) == expected


@pytest.mark.parametrize("text", [
    "Build me a portfolio of tech stocks",
    "build me a portfolio that returns 20% a year",
    "I have 5 stocks, build a 60/40 portfolio",
    "build me a tech portfolio under 10% drawdown",
])
def test_other_percentages_are_not_limits(text: str) -> None:
    assert stated_limits(text) == (None, None, None)


def columns(n: int = 6, seed: int = 9) -> dict[str, list[float]]:
    rng = random.Random(seed)
    common = [rng.gauss(0, 0.01) for _ in range(400)]
    return {f"S{i}USDT": [(0.5 + 0.15 * i) * c + rng.gauss(0, 0.004) for c in common]
            for i in range(n)}


def test_a_count_and_a_cap_are_both_met_and_reported() -> None:
    cols = columns()
    base = _equal_risk_weights(tuple(cols), cols)
    assert base is not None
    out = _within_limits(base, cols, 0.4, 3, None)
    assert out["status"] == "optimal"
    assert len(out["weights"]) <= 3 and max(out["weights"].values()) <= 0.4 + 1e-3
    assert abs(sum(out["weights"].values()) - 1) < 1e-9
    assert "at most 3 names" in out["headline"] and "What the limits cost" in out["limits_line"]
    assert "exact search" in out["limits_line"]


def test_limits_that_cannot_hold_say_why() -> None:
    cols = columns()
    base = _equal_risk_weights(tuple(cols), cols)
    assert base is not None
    out = _within_limits(base, cols, 0.3, 2, None)
    assert out["status"] == "infeasible" and "short of 100%" in out["plain"]


def test_a_risk_cap_is_met_and_labelled_local() -> None:
    cols = columns()
    base = {k: 1 / 6 for k in cols}  # equal money, unequal risk: the loud names carry most of it
    out = _within_limits(base, cols, None, None, 0.2)
    assert out["status"] == "locally optimal"
    assert "not convex" in out["limits_line"]
