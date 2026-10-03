"""Win rates said with their uncertainty (argus.backtest.proportion), checked against hand
computations and against LuxAlgo edge-stats' own test values."""

from __future__ import annotations

import pytest

from argus.backtest.proportion import (
    rate_phrase,
    stability_phrase,
    stability_split,
    wilson,
)


def test_wilson_matches_a_hand_computed_interval() -> None:
    # 20 of 33: statsmodels' proportion_confint(20, 33, method="wilson") = (0.43683, 0.75317)
    w = wilson(20, 33)
    assert w is not None
    assert w.estimate == pytest.approx(20 / 33)
    assert w.lo == pytest.approx(0.4368344081994022, abs=1e-12)
    assert w.hi == pytest.approx(0.7531689256552292, abs=1e-12)
    assert not w.excludes(0.5)


def test_wilson_edges() -> None:
    assert wilson(0, 0) is None
    full = wilson(10, 10)
    assert full is not None
    assert full.hi == 1.0
    assert full.lo == pytest.approx(0.7224672001371106, abs=1e-12)  # statsmodels
    with pytest.raises(ValueError, match="impossible"):
        wilson(11, 10)


def test_rate_phrase_follows_the_sample_floors() -> None:
    assert rate_phrase(5, 9) == "5 of 9 cases — too few to state a rate"
    small = rate_phrase(20, 33)
    assert small.startswith("61% (20 of 33 cases; 95% range 44% to 75%")
    assert "not distinguishable from a coin flip" in small
    assert "small sample" not in small  # 33 is above the warning floor
    assert "a small sample" in rate_phrase(15, 20)
    assert "better than a coin flip" in rate_phrase(90, 100)
    assert "worse than a coin flip" in rate_phrase(10, 100)


def test_stability_split_agrees_and_disagrees() -> None:
    steady = [True, False] * 20
    assert stability_split(steady).agree is True
    assert "consistent" in (stability_phrase(steady) or "")
    shifted = [True] * 20 + [False] * 20
    assert stability_split(shifted).agree is False
    assert "not consistent" in (stability_phrase(shifted) or "")
    assert stability_phrase([True] * 12) is None  # a half below the refuse floor
