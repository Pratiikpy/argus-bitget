"""Tail-risk contribution: the book's historical CVaR split exactly across positions.

Checked against skfolio itself on 20 random books (`data/tail_contribution_oracle.json`: CVaR to
1e-17, contributions to 3e-13, the rest being skfolio's finite-difference step). Pinned here on
inputs a reader can check by hand.
"""

from __future__ import annotations

import pytest

from argus.desk.portfolio import tail_contributions


def test_twenty_bars_at_95_percent_average_the_single_worst_bar() -> None:
    a = [0.01] * 19 + [-0.10]
    b = [0.00] * 19 + [-0.02]
    got = tail_contributions({"A": 0.5, "B": 0.5}, {"A": a, "B": b})
    assert got is not None
    # k = 1, ik = 0: CVaR is minus the worst bar, -(0.5 * -0.10 + 0.5 * -0.02)
    assert got.cvar == pytest.approx(0.06)
    assert dict(got.contributions) == pytest.approx({"A": 0.05, "B": 0.01})
    assert got.share("A") == pytest.approx(0.05 / 0.06)


def test_contributions_sum_to_the_cvar_with_a_fractional_bar() -> None:
    a = [0.001 * ((i * 7) % 11 - 5) for i in range(30)]
    b = [0.002 * ((i * 5) % 13 - 6) for i in range(30)]
    got = tail_contributions({"A": 0.3, "B": 0.7}, {"A": a, "B": b})
    assert got is not None
    assert sum(v for _, v in got.contributions) == pytest.approx(got.cvar)
    assert got.bars == 30


def test_too_short_or_empty_is_refused() -> None:
    assert tail_contributions({"A": 1.0}, {"A": [0.01] * 5}) is None
    assert tail_contributions({}, {"A": [0.01] * 30}) is None
