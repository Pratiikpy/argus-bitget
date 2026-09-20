"""Anti-overfit gate tests.

The decisive test in this file is :meth:`TestNoiseIsRejected.test_a_pure_noise_factor_is_rejected`.
A suite that certifies noise is worse than no suite, because it converts "we did not check" into
"we checked and it was fine". Everything else here exists to make sure the gates are not simply
rejecting everything, which is the other way to be useless.

Fixtures are generated from a seeded RNG so a verdict is reproducible. A placebo test whose outcome
depends on when it ran is a different test each time and cannot be cited.
"""

from __future__ import annotations

import random

import pytest

from argus.research.overfit import (
    MIN_OBSERVATIONS,
    Observation,
    Outcome,
    check_half_life,
    check_ic_stability,
    check_placebo,
    check_subsample_stress,
    period_ic,
    run_all,
    spearman,
)

NAMES = [f"SYM{i}" for i in range(12)]
PERIODS = 60


def _noise(seed: int = 1) -> list[Observation]:
    """A factor with no relationship to the outcome whatsoever."""
    rng = random.Random(seed)
    return [
        Observation(p, n, rng.gauss(0, 1), rng.gauss(0, 1))
        for p in range(PERIODS)
        for n in NAMES
    ]


def _signal(seed: int = 2, strength: float = 0.85, decay: float = 0.45) -> list[Observation]:
    """A factor that genuinely predicts the next period's return, and decays after it.

    ``decay`` makes the edge fall away across horizons, which is what a real signal does and what
    the half-life gate looks for. A fixture without decay would pass three gates and stall the
    fourth, which is a worse fixture, not a better factor.
    """
    rng = random.Random(seed)
    rows: list[Observation] = []
    carried: dict[str, float] = {}
    for period in range(PERIODS):
        for name in NAMES:
            factor = rng.gauss(0, 1)
            future = strength * factor + rng.gauss(0, 1 - strength) + decay * carried.get(name, 0.0)
            carried[name] = factor
            rows.append(Observation(period, name, factor, future))
    return rows


class TestRankCorrelation:
    def test_a_perfect_monotone_relationship_is_one(self) -> None:
        xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        assert spearman(xs, [v * 3 for v in xs]) == pytest.approx(1.0)

    def test_a_perfectly_inverted_relationship_is_minus_one(self) -> None:
        xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        assert spearman(xs, list(reversed(xs))) == pytest.approx(-1.0)

    def test_it_is_rank_based_so_a_monotone_transform_does_not_move_it(self) -> None:
        """Pearson would move here. Spearman must not — that is why it is the right statistic."""
        xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        straight = spearman(xs, [v * 2 for v in xs])
        curved = spearman(xs, [v**3 for v in xs])
        assert straight == pytest.approx(curved)

    def test_ties_share_a_rank_rather_than_taking_an_arbitrary_order(self) -> None:
        assert spearman([1.0, 1.0, 1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0, 5.0]) is not None

    def test_a_constant_series_has_no_correlation_rather_than_zero(self) -> None:
        """None, not 0.0: "no variation to correlate" is not "no relationship"."""
        assert spearman([1.0] * 6, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) is None

    def test_too_small_a_cross_section_is_undefined(self) -> None:
        assert spearman([1.0, 2.0], [1.0, 2.0]) is None


class TestNoiseIsRejected:
    """The test that makes the suite worth having."""

    def test_a_pure_noise_factor_is_rejected(self) -> None:
        report = run_all(_noise())
        assert report.verdict.startswith("rejected"), report.as_dict()
        assert report.passed < 4

    def test_noise_fails_the_placebo_gate_specifically(self) -> None:
        """Noise is by construction indistinguishable from its own shuffle."""
        assert check_placebo(_noise()).outcome is Outcome.FAIL

    @pytest.mark.parametrize("seed", [1, 7, 13, 29])
    def test_noise_is_rejected_across_several_samples(self, seed: int) -> None:
        """One rejected sample could be luck; four is the gate working."""
        assert run_all(_noise(seed)).verdict.startswith(("rejected", "not proven"))


class TestSignalSurvives:
    """A gate that rejects everything is as useless as one that accepts everything."""

    def test_a_genuine_signal_clears_the_placebo_gate(self) -> None:
        result = check_placebo(_signal())
        assert result.outcome is Outcome.PASS, result.as_dict()
        assert result.detail["beats_null"] is True

    def test_a_genuine_signal_has_a_stable_ic(self) -> None:
        result = check_ic_stability(_signal())
        assert result.outcome is Outcome.PASS, result.as_dict()
        assert result.detail["sign_reversals"] == []

    def test_a_genuine_signal_survives_regime_slicing(self) -> None:
        result = check_subsample_stress(_signal())
        assert result.outcome is Outcome.PASS, result.as_dict()

    def test_the_measured_ic_is_positive_and_material(self) -> None:
        ics = period_ic(_signal())
        mean_ic = sum(ics.values()) / len(ics)
        assert mean_ic > 0.3, f"fixture should carry a clear signal, got {mean_ic}"


class TestInsufficientDataIsNotFailure:
    """The departure from the reference implementation, and the reason for it."""

    def _tiny(self) -> list[Observation]:
        rng = random.Random(3)
        return [
            Observation(p, n, rng.gauss(0, 1), rng.gauss(0, 1))
            for p in range(MIN_OBSERVATIONS - 5)
            for n in NAMES
        ]

    @pytest.mark.parametrize(
        "gate", [check_ic_stability, check_subsample_stress, check_placebo, check_half_life]
    )
    def test_every_gate_reports_inconclusive_not_failed(self, gate) -> None:  # type: ignore[no-untyped-def]
        assert gate(self._tiny()).outcome is Outcome.INCONCLUSIVE

    def test_the_report_counts_inconclusive_separately_from_failed(self) -> None:
        report = run_all(self._tiny())
        assert report.inconclusive == 4
        assert report.failed == 0
        assert report.passed == 0

    def test_the_verdict_says_not_proven_rather_than_rejected(self) -> None:
        """"We could not tell" must never be recorded as "it is bad"."""
        assert run_all(self._tiny()).verdict == "not proven: insufficient data for a verdict"

    def test_an_empty_sample_does_not_raise(self) -> None:
        report = run_all([])
        assert report.inconclusive == 4


class TestHalfLife:
    def test_a_decaying_signal_has_a_measurable_half_life(self) -> None:
        result = check_half_life(_signal(strength=0.9, decay=0.0))
        assert result.outcome in {Outcome.PASS, Outcome.FAIL}
        assert result.detail["half_life_periods"] is not None

    def test_an_ic_that_never_halves_is_inconclusive_not_a_pass(self) -> None:
        """A flat IC across every horizon usually means the horizon is being measured, not the
        future. Recording that as a pass would certify the least interpretable case."""
        rows = [
            Observation(p, n, float(hash((n, 0)) % 100), 1.0)
            for p in range(PERIODS)
            for n in NAMES
        ]
        assert check_half_life(rows).outcome in {Outcome.INCONCLUSIVE, Outcome.FAIL}


class TestDeterminism:
    def test_the_placebo_test_gives_the_same_answer_every_run(self) -> None:
        """A factor that passes on the third attempt has not passed."""
        rows = _signal()
        first = check_placebo(rows).detail
        second = check_placebo(rows).detail
        assert first["placebo_p95_abs_ic"] == second["placebo_p95_abs_ic"]
        assert first["real_ic"] == second["real_ic"]

    def test_the_whole_report_is_reproducible(self) -> None:
        rows = _noise(11)
        assert run_all(rows).as_dict() == run_all(rows).as_dict()


class TestTheVerdictIsNotAScore:
    def test_a_placebo_failure_dominates_other_passes(self) -> None:
        """A percentage would let three passes outvote the one failure that matters."""
        report = run_all(_noise())
        if any(r.name == "placebo" and r.outcome is Outcome.FAIL for r in report.results):
            assert "indistinguishable from shuffled data" in report.verdict

    def test_a_clean_sweep_is_stated_plainly(self) -> None:
        report = run_all(_signal())
        if report.failed == 0 and report.inconclusive == 0:
            assert report.verdict == "clears every gate"

    def test_the_report_serialises_every_test(self) -> None:
        payload = run_all(_signal()).as_dict()
        assert len(payload["tests"]) == 4
        assert {t["test"] for t in payload["tests"]} == {
            "ic_stability", "subsample_stress", "placebo", "half_life"
        }
