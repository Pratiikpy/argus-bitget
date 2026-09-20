"""Venue-survey tests — a profitable rule must be found and an unprofitable one must not.

This module makes an absolute claim ("this rule makes money on this venue") rather than a paired
one, so the way it fails is by calling noise profitable. The guard is that profitability requires a
dependence-aware interval to exclude zero, not a positive total — and the tests check that a series
with a large positive sum and a wide interval is reported as *not* profitable, which is the case a
naive implementation gets wrong.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta

import pytest

from argus.eval.incremental import DIRECTION_LONG, Instant, always_flat, always_long
from argus.eval.venue_rules import (
    MIN_INSTANTS,
    VenueRulesError,
    score_rule,
    survey,
)

START = datetime(2026, 4, 1, tzinfo=UTC)
HURDLE = 18.8


def _instants(
    n: int, *, drift: float, seed: int, spread: float = 150.0, symbols: int = 4
) -> list[Instant]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        out.append(Instant(
            symbol=f"S{i % symbols}",
            at=START + timedelta(hours=(i // symbols) * 6),
            trailing=tuple(rng.gauss(drift / 10_000, 0.01) for _ in range(72)),
            realised_bps=rng.gauss(drift, spread),
            hurdle_bps=HURDLE,
        ))
    return out


class TestScoringOneRule:
    def test_a_flat_rule_never_trades_and_earns_nothing(self) -> None:
        got = score_rule("flat", always_flat, _instants(300, drift=400.0, seed=1))
        assert got.trades == 0
        assert got.total_bps == 0.0
        assert got.accuracy is None

    def test_accuracy_is_measured_over_traded_instants(self) -> None:
        got = score_rule("long", always_long, _instants(300, drift=300.0, seed=2, spread=40.0))
        assert got.trades == 300
        assert got.accuracy is not None and got.accuracy > 0.9

    def test_accuracy_counts_direction_not_profit(self) -> None:
        """A trade can be directionally right and still lose to the hurdle. Conflating the two
        would report a rule as accurate because it was profitable, which is circular."""
        instants = [
            Instant("S", START, (0.001,) * 72, realised_bps=5.0, hurdle_bps=HURDLE),
        ]
        got = score_rule("long", always_long, instants)
        assert got.accuracy == 1.0
        assert got.total_bps < 0

    def test_a_rule_that_loses_has_a_negative_total(self) -> None:
        assert score_rule("long", always_long, _instants(300, drift=-300.0, seed=3)).total_bps < 0


class TestProfitabilityNeedsAnInterval:
    def test_a_clear_edge_is_reported_profitable(self) -> None:
        got = score_rule("long", always_long, _instants(400, drift=200.0, seed=4, spread=60.0))
        assert got.profitable is True

    def test_a_positive_total_with_a_wide_interval_is_not_profitable(self) -> None:
        """The case a naive implementation gets wrong. Summing to a positive number over a noisy
        series is what a coin does half the time; only the interval separates them."""
        got = score_rule("long", always_long, _instants(400, drift=20.0, seed=5, spread=600.0))
        assert got.total_bps != 0
        assert got.profitable is False

    def test_a_losing_rule_is_never_profitable(self) -> None:
        got = score_rule("long", always_long, _instants(400, drift=-200.0, seed=6, spread=60.0))
        assert got.profitable is False

    def test_a_rule_that_never_trades_has_no_interval(self) -> None:
        got = score_rule("flat", always_flat, _instants(400, drift=200.0, seed=7))
        assert got.profitable is None
        assert got.interval["available"] is False

    def test_both_interval_methods_are_reported(self) -> None:
        """They can disagree — the closed form assumes asymptotic normality and the bootstrap does
        not — and the disagreement is informative, so neither is dropped."""
        got = score_rule("long", always_long, _instants(400, drift=200.0, seed=8, spread=60.0))
        assert "hac" in got.interval
        assert "bootstrap" in got.interval

    def test_the_interval_reports_its_own_dependence_correction(self) -> None:
        got = score_rule("long", always_long, _instants(400, drift=100.0, seed=9))
        assert "eta" in got.interval["hac"]
        assert "block_length" in got.interval["bootstrap"]


class TestTheSurvey:
    def test_a_thin_sample_raises(self) -> None:
        """The floor is far above the paired comparison's, because this is an absolute claim."""
        with pytest.raises(VenueRulesError, match="below the"):
            survey(_instants(MIN_INSTANTS - 1, drift=0.0, seed=10))

    def test_the_floor_is_far_above_the_paired_one(self) -> None:
        from argus.eval.incremental import MIN_INSTANTS as PAIRED

        assert MIN_INSTANTS > PAIRED * 5

    def test_a_universe_with_no_edge_reports_no_profitable_rule(self) -> None:
        got = survey(_instants(600, drift=0.0, seed=11, spread=200.0))
        assert not got.any_profitable
        assert "NO RULE IS PROFITABLE" in got.verdict

    def test_it_states_the_defence_of_abstention_that_follows(self) -> None:
        got = survey(_instants(600, drift=0.0, seed=12, spread=200.0))
        assert "declining something that does not pay" in got.verdict

    def test_a_universe_with_an_edge_says_the_desk_has_a_case_to_answer(self) -> None:
        """The verdict must be able to come out against the desk, or it is not a measurement."""
        got = survey(_instants(600, drift=250.0, seed=13, spread=60.0))
        assert got.any_profitable
        assert "case to answer" in got.verdict

    def test_the_break_even_accuracy_is_computed_from_the_same_instants(self) -> None:
        got = survey(_instants(600, drift=0.0, seed=14, spread=200.0))
        assert 0.5 < got.break_even_accuracy < 1.0

    def test_the_verdict_compares_accuracy_against_break_even(self) -> None:
        got = survey(_instants(600, drift=0.0, seed=15))
        assert "against a break-even of" in got.verdict

    def test_every_rule_is_scored(self) -> None:
        from argus.eval.incremental import BASELINES

        got = survey(_instants(600, drift=50.0, seed=16))
        assert {r.name for r in got.results} == set(BASELINES)

    def test_the_best_rule_is_the_one_with_the_largest_total(self) -> None:
        got = survey(_instants(600, drift=200.0, seed=17, spread=60.0))
        assert got.best.total_bps == max(r.total_bps for r in got.results)

    def test_the_report_names_every_rule_and_its_verdict(self) -> None:
        text = survey(_instants(600, drift=0.0, seed=18)).render()
        assert "VENUE RULES" in text
        assert "profitable" in text

    def test_the_dict_carries_the_intervals_not_only_the_totals(self) -> None:
        got = survey(_instants(600, drift=200.0, seed=19, spread=60.0)).as_dict()
        traded = [r for r in got["results"] if r["trades"] > 0]
        assert traded
        assert all("interval" in r for r in traded)


class TestItNeverFabricatesADeskDecision:
    def test_the_survey_carries_no_desk_arm(self) -> None:
        """The whole reason this module is separate from `eval/incremental`. Growing the sample by
        assuming the desk would have abstained at instants it never saw would be inventing a
        decision record, which is the one thing the replay harness exists to prevent."""
        got = survey(_instants(600, drift=0.0, seed=20)).as_dict()
        assert "desk" not in got
        assert all(r["name"] != "desk" for r in got["results"])

    def test_the_instant_carries_no_decision_field(self) -> None:
        instant = _instants(1, drift=0.0, seed=21)[0]
        assert not hasattr(instant, "verdict")
        assert not hasattr(instant, "decision")

    def test_a_direction_comes_only_from_a_rule(self) -> None:
        """Every direction in this module is produced by a policy function from the trailing
        window. None is read from a ledger, so none can be mistaken for something the desk did."""
        instant = _instants(1, drift=100.0, seed=22)[0]
        assert always_long(instant) == DIRECTION_LONG
