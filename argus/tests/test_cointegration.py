"""Cointegration tests — the first half is a reproduction, not an assertion of correctness.

A hand-rolled ADF is exactly the kind of code that runs, returns a plausible number, and is wrong:
an off-by-one in the lag matrix, an information criterion compared across different sample sizes, or
the ordinary critical-value table used on a fitted residual all produce output that looks fine and
rejects the null far too often.

So the statistics are not tested against our own expectations. They are tested against
**statsmodels 0.14.6**, run on this machine over the same fixed series, with its answers stored in
``tests/data/cointegration_expected.json``. Regenerate with:

    <trading venv>/Scripts/python.exe <oracle script>
        tests/data/cointegration_fixture.json

The fixture holds 500 hourly closes each of NVDAUSDT, COINUSDT and MSTRUSDT, a seeded random walk,
and a seeded stationary AR(1) — real data where the answer is unknown, and synthetic data where it
is known.

The second half tests what statsmodels does not do for us: point-in-time discipline, the four-leg
cost hurdle, and the multiple-testing arithmetic.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import pytest

from argus.research.cointegration import (
    LEGS_PER_ROUND_TRIP,
    MIN_OBSERVATIONS,
    TAKER_BPS_PER_LEG,
    CointegrationError,
    adf,
    engle_granger,
    half_life,
    mackinnon_crit,
    mackinnon_p,
    ols,
    pair_test,
    scan,
    zscores,
)

DATA = Path(__file__).resolve().parent / "data"
FIXTURE = json.loads((DATA / "cointegration_fixture.json").read_text(encoding="utf-8"))
EXPECTED = json.loads((DATA / "cointegration_expected.json").read_text(encoding="utf-8"))
SERIES: dict[str, list[float]] = FIXTURE["series"]


def _ou(n: int, *, phi: float, seed: int) -> list[float]:
    rng = random.Random(seed)
    out = [0.0]
    for _ in range(n - 1):
        out.append(phi * out[-1] + rng.gauss(0, 1))
    return out


class TestItReproducesStatsmodels:
    """Same inputs, same numbers. Anything else means we have invented a different test."""

    @pytest.mark.parametrize("name", sorted(SERIES))
    @pytest.mark.parametrize("regression", ["c", "n"])
    def test_the_adf_statistic_matches(self, name: str, regression: str) -> None:
        got = adf(SERIES[name], regression=regression)
        want = EXPECTED[f"adf_{regression}_{name}"]
        assert got.statistic == pytest.approx(want["stat"], abs=1e-9)
        assert got.pvalue == pytest.approx(want["p"], abs=1e-9)

    @pytest.mark.parametrize("name", sorted(SERIES))
    def test_the_selected_lag_and_sample_match(self, name: str) -> None:
        """The lag search is where a hand-rolled ADF usually diverges: statsmodels scores every
        candidate on the same rows and then refits the winner on the longer sample that lag allows.
        Getting one of those two steps wrong still produces a statistic — a different one."""
        got = adf(SERIES[name], regression="c")
        want = EXPECTED[f"adf_c_{name}"]
        assert got.usedlag == want["usedlag"]
        assert got.nobs == want["nobs"]

    @pytest.mark.parametrize("name", sorted(SERIES))
    def test_the_critical_values_match(self, name: str) -> None:
        got = adf(SERIES[name], regression="c")
        want = EXPECTED[f"adf_c_{name}"]["crit"]
        for level in ("1%", "5%", "10%"):
            assert got.critical_values[level] == pytest.approx(want[level], abs=1e-9)

    @pytest.mark.parametrize("label", sorted(FIXTURE["pairs"]))
    def test_engle_granger_matches(self, label: str) -> None:
        left, right = FIXTURE["pairs"][label]
        got = engle_granger(SERIES[left], SERIES[right])
        want = EXPECTED[f"coint_{label}"]
        assert got.statistic == pytest.approx(want["stat"], abs=1e-9)
        assert got.pvalue == pytest.approx(want["p"], abs=1e-9)
        assert got.critical_values["5%"] == pytest.approx(want["crit"][1], abs=1e-9)

    def test_the_residual_test_uses_the_two_series_table(self) -> None:
        """The classic error is to test a fitted residual against the ordinary ADF table. At the
        same sample size the N=2 critical value is materially harder to clear, and using N=1 would
        promote spurious pairs."""
        n1 = mackinnon_crit(regression="c", n=1, nobs=499)["5%"]
        n2 = mackinnon_crit(regression="c", n=2, nobs=499)["5%"]
        assert n2 < n1 - 0.4


class TestTheArithmeticUnderneath:
    def test_ols_recovers_a_known_relationship(self) -> None:
        x = [float(i) for i in range(50)]
        y = [3.0 + 2.5 * v for v in x]
        fit = ols(y, [x, [1.0] * 50])
        assert fit.params[0] == pytest.approx(2.5, abs=1e-9)
        assert fit.params[1] == pytest.approx(3.0, abs=1e-9)
        assert fit.ssr == pytest.approx(0.0, abs=1e-18)

    def test_a_singular_design_raises_rather_than_returning_noise(self) -> None:
        x = [1.0] * 40
        with pytest.raises(CointegrationError, match="singular"):
            ols([float(i) for i in range(40)], [x, [1.0] * 40])

    def test_the_p_value_is_clamped_outside_the_fitted_range(self) -> None:
        """MacKinnon's polynomials are fitted over a finite range; extrapolating a cubic past it
        returns probabilities outside [0, 1]. statsmodels clamps, so we clamp."""
        assert mackinnon_p(-50.0, regression="c", n=1) == 0.0
        assert mackinnon_p(50.0, regression="c", n=1) == 1.0

    def test_a_stationary_series_is_called_stationary_and_a_walk_is_not(self) -> None:
        assert adf(SERIES["stationary_ar1"], regression="c").stationary_at_5pct
        assert not adf(SERIES["nvdausdt"], regression="c").stationary_at_5pct


class TestHalfLife:
    def test_it_recovers_a_known_mean_reversion_speed(self) -> None:
        """For an AR(1) with coefficient phi, the OU half-life is ln(2)/-ln(phi). Estimated from
        1,500 points it should land within a few percent."""
        phi = 0.9
        got = half_life(_ou(1500, phi=phi, seed=11))
        assert got is not None
        assert got == pytest.approx(math.log(2) / -math.log(phi), rel=0.15)

    def test_a_diverging_series_has_no_half_life_rather_than_a_large_one(self) -> None:
        rising = [float(i) * 1.01 for i in range(200)]
        assert half_life(rising) is None

    def test_a_faster_reverter_has_a_shorter_half_life(self) -> None:
        fast = half_life(_ou(1500, phi=0.5, seed=3))
        slow = half_life(_ou(1500, phi=0.95, seed=3))
        assert fast is not None and slow is not None
        assert fast < slow

    def test_it_refuses_a_short_series(self) -> None:
        with pytest.raises(CointegrationError, match="below the"):
            half_life([1.0, 2.0, 3.0])


class TestPointInTimeDiscipline:
    def test_a_z_score_cannot_see_the_future(self) -> None:
        """The property the whole-sample z-score violates. Changing a later value must not move an
        earlier z by a single bit."""
        base = _ou(600, phi=0.8, seed=7)
        altered = list(base)
        altered[-1] = 500.0
        first = zscores(base, lookback=240)
        second = zscores(altered, lookback=240)
        assert first[:-1] == second[:-1]

    def test_early_entries_are_none_rather_than_computed_from_too_little(self) -> None:
        got = zscores(_ou(400, phi=0.8, seed=8), lookback=240)
        assert all(z is None for z in got[:240])
        assert got[240] is not None

    def test_a_tiny_lookback_is_refused(self) -> None:
        with pytest.raises(CointegrationError, match="at least 20"):
            zscores(_ou(200, phi=0.8, seed=9), lookback=5)

    def test_the_hedge_ratio_is_frozen_from_the_training_slice(self) -> None:
        """If the ratio were refitted on the full sample, the out-of-sample ADF would be a second
        in-sample test wearing a different label."""
        left, right = SERIES["nvdausdt"], SERIES["coinusdt"]
        result = pair_test("A", "B", left, right, train_fraction=0.6)
        split = int(len(left) * 0.6)
        train_only = engle_granger(left[:split], right[:split])
        assert result.in_sample.hedge_ratio == pytest.approx(train_only.hedge_ratio, abs=1e-12)
        full = engle_granger(left, right)
        assert result.in_sample.hedge_ratio != pytest.approx(full.hedge_ratio, abs=1e-9)


class TestCostsDecideIt:
    def test_a_round_trip_is_four_legs(self) -> None:
        result = pair_test("A", "B", SERIES["nvdausdt"], SERIES["coinusdt"])
        assert LEGS_PER_ROUND_TRIP == 4
        assert result.cost_bps == pytest.approx(TAKER_BPS_PER_LEG * 4)

    def test_a_narrow_spread_needs_an_impossible_entry(self) -> None:
        """Two nearly identical series have a tiny spread standard deviation, so the z-score needed
        to cover 24bps of gross notional is enormous — and the pair is correctly unusable however
        strongly it is cointegrated."""
        rng = random.Random(21)
        base = [100.0 + i * 0.01 for i in range(600)]
        noisy = [v + rng.gauss(0, 0.001) for v in base]
        result = pair_test("A", "B", noisy, base)
        assert result.required_z > 10
        assert result.extreme_share == 0.0
        assert not result.tradeable

    def test_tradeable_needs_out_of_sample_stationarity_too(self) -> None:
        result = pair_test("A", "B", SERIES["random_walk"], SERIES["nvdausdt"])
        if result.out_of_sample is not None and not result.out_of_sample.stationary_at_5pct:
            assert not result.tradeable


class TestMultipleTesting:
    def test_the_correction_is_the_projects_own_not_a_second_copy(self) -> None:
        """A scanner that carries its own copy of a correction is how two parts of one system come
        to disagree about what survived. This asserts the import, not a reimplementation."""
        import argus.research.cointegration as module

        assert not hasattr(module, "benjamini_hochberg")
        from argus.backtest.validation import benjamini_hochberg

        assert benjamini_hochberg([0.001, 0.02, 0.03, 0.9], fdr=0.05) == [True, True, True, False]

    def test_the_scan_counts_its_own_tests(self) -> None:
        report = scan({k: v for k, v in SERIES.items() if k != "stationary_ar1"}, lookback=100)
        assert report.tests == 4 * 3 // 2
        assert report.expected_false_positives == pytest.approx(0.05 * report.tests)
        assert report.bonferroni_threshold == pytest.approx(0.05 / report.tests)
        assert str(report.tests) in report.verdict

    def test_the_verdict_names_the_chance_rate_beside_the_hit_count(self) -> None:
        report = scan({k: v for k, v in SERIES.items() if k != "stationary_ar1"}, lookback=100)
        assert "expected by chance" in report.verdict
        assert "Benjamini-Hochberg" in report.verdict

    def test_bonferroni_is_never_more_permissive_than_fdr(self) -> None:
        report = scan(SERIES, lookback=100)
        assert len(report.survivors_bonferroni) <= len(report.survivors_fdr)

    def test_a_pair_too_short_to_test_is_dropped_not_counted(self) -> None:
        short = {"a": SERIES["nvdausdt"][:50], "b": SERIES["coinusdt"], "c": SERIES["mstrusdt"]}
        assert scan(short, lookback=100).tests == 1

    def test_the_report_serialises(self) -> None:
        blob = json.loads(json.dumps(scan(SERIES, lookback=100).as_dict()))
        assert blob["tests"] == 5 * 4 // 2
        assert "verdict" in blob and "pairs" in blob
        assert "required_z_to_clear_costs" in blob["pairs"][0]


class TestItRefuses:
    def test_mismatched_lengths_raise(self) -> None:
        with pytest.raises(CointegrationError, match="differ in length"):
            engle_granger(SERIES["nvdausdt"], SERIES["coinusdt"][:-1])

    def test_too_few_observations_raise(self) -> None:
        with pytest.raises(CointegrationError, match="below the"):
            adf([1.0] * (MIN_OBSERVATIONS - 1))

    def test_collinear_series_raise_rather_than_returning_infinity(self) -> None:
        base = SERIES["nvdausdt"]
        with pytest.raises(CointegrationError, match="collinear"):
            engle_granger(base, [v * 2.0 for v in base])

    def test_an_unknown_regression_raises(self) -> None:
        with pytest.raises(CointegrationError, match="must be"):
            adf(SERIES["nvdausdt"], regression="quadratic")
