"""Tests for the adversarial cost/concentration ablation of the allocation comparison.

Live-network tests, for the same reason the module itself is: the question is whether Riskfolio's
NCO still beats ARGUS's HRP on the *real* rToken candles once the fee is charged and the
concentration is capped, and a fixture would answer a different question. The fetch is expensive,
so it happens once per session in a module-scoped fixture.

These tests deliberately do NOT pin who wins. The winner is a measurement of live market data and
is allowed to move. What is pinned is that the ablation is a real ablation: that Riskfolio really
ran, that its `w_max` really constrained the weights rather than being silently ignored, that the
fee really equals ARGUS's own `turnover * taker_bps`, that the leverage comes from the estimation
window and never from the held-out one, and that every published boolean agrees with the numbers
it claims to summarise. A test asserting "ARGUS wins after costs" would be exactly the flattering
comparison this module exists to guard against.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from argus.desk.allocation import MIN_OBSERVATIONS, TAKER_BPS
from argus.eval.allocation_comparison import INVERSE_PAIR, load_returns
from argus.eval.allocation_cost_ablation import (
    MAX_LEVERAGE,
    SCOPE_STATEMENT,
    WEIGHT_CAP,
    _riskfolio_allocation,
    _two_sided_binomial_p,
    render,
    run_universe_ablation,
    run_walk_forward,
    summarise_verdict,
)

# Short grids. The module's own defaults produce 24 windows across five walk-forwards; these
# tests need the harness to be correct, not to be powerful, and a 480/240 grid gives four windows
# in a fraction of the time. Every invariant checked below is independent of window count.
TRAIN = 480
TEST = 240
STEP = 240


@pytest.fixture(scope="module")
def columns() -> dict[str, list[float]]:
    names, cols, failures = load_returns()
    if len(names) < 4 or len(cols[names[0]]) < MIN_OBSERVATIONS:
        pytest.skip(f"insufficient live rToken history to ablate allocators: {failures}")
    return cols


@pytest.fixture(scope="module")
def uncapped(columns: dict[str, list[float]]) -> dict[str, Any]:
    return run_walk_forward(columns, train_bars=TRAIN, test_bars=TEST, step_bars=STEP)


@pytest.fixture(scope="module")
def capped(columns: dict[str, list[float]]) -> dict[str, Any]:
    return run_walk_forward(
        columns, w_max=WEIGHT_CAP, train_bars=TRAIN, test_bars=TEST, step_bars=STEP,
    )


class TestTheBaselineActuallyRan:
    """The defect this whole line of work exists to correct is a baseline that was cloned and
    never run. An ablation that degraded to "Riskfolio would not import, so nothing changed"
    would reproduce that defect with a green suite on top of it."""

    def test_riskfolio_nco_produces_a_real_long_only_book(
        self, columns: dict[str, list[float]],
    ) -> None:
        weights = _riskfolio_allocation(
            columns, model="NCO", linkage="ward", leaf_order=True,
        )
        assert len(weights) == len(columns)
        assert all(w >= -1e-9 for w in weights.values())
        assert math.isclose(sum(weights.values()), 1.0, abs_tol=1e-6)

    def test_every_allocator_scored_every_window(self, uncapped: dict[str, Any]) -> None:
        assert uncapped["errors"] == {}
        counts = {row["n_windows"] for row in uncapped["results"].values()}
        assert counts == {uncapped["n_origins"]}
        assert uncapped["n_origins"] >= 2

    def test_the_riskfolio_twin_of_argus_is_a_null_control(
        self, uncapped: dict[str, Any],
    ) -> None:
        """Riskfolio's HRP with `leaf_order=False` is ARGUS's own algorithm. If the harness
        manufactures a difference between two identical allocations, every other row is worthless.
        This is the same control the sibling module uses, recomputed by this harness."""
        argus = uncapped["results"]["argus_hrp"]
        twin = uncapped["results"]["riskfolio_hrp_argus_equivalent"]
        assert math.isclose(
            twin["mean_oos_vol_bps"], argus["mean_oos_vol_bps"], rel_tol=1e-9
        )
        for criterion in ("oos_vol", "net_return", "levered_net_return"):
            assert twin[f"{criterion}_windows_compared"] == 0, criterion
            assert twin[f"{criterion}_windows_tied_with_argus"] == twin["n_windows"], criterion
        assert uncapped["parity_twin_null_control"]["behaves_as_a_null"] is True

    def test_the_cap_makes_the_twin_legitimately_stop_being_a_null(
        self, capped: dict[str, Any],
    ) -> None:
        """With `w_max` set, Riskfolio's bound-fitting loop moves ARGUS's own weights, so the twin
        is no longer ARGUS's allocation and must not be reported as a failed null."""
        assert capped["parity_twin_null_control"]["behaves_as_a_null"] is None


class TestTheWeightCapIsReal:
    """`w_max` had to be checked, not assumed: Riskfolio applies it in a post-model fitting loop
    (`HCPortfolio.py:1119-1140`), and a cap that were silently dropped for NCO would turn the
    concentration ablation into a second copy of the uncapped run."""

    def test_the_cap_binds_on_the_concentrated_allocator(
        self, columns: dict[str, list[float]],
    ) -> None:
        weights = _riskfolio_allocation(
            columns, w_max=WEIGHT_CAP, model="NCO", linkage="ward", leaf_order=True,
        )
        assert max(weights.values()) <= WEIGHT_CAP + 1e-6
        assert math.isclose(sum(weights.values()), 1.0, abs_tol=1e-6)

    def test_the_cap_changes_the_book_it_is_applied_to(
        self, columns: dict[str, list[float]],
    ) -> None:
        free = _riskfolio_allocation(columns, model="NCO", linkage="ward", leaf_order=True)
        bound = _riskfolio_allocation(
            columns, w_max=WEIGHT_CAP, model="NCO", linkage="ward", leaf_order=True,
        )
        assert max(free.values()) > WEIGHT_CAP, "NCO was already inside the cap; ablation is moot"
        assert max(abs(free[n] - bound[n]) for n in free) > 1e-6

    def test_the_capped_run_reports_weights_inside_the_cap(
        self, capped: dict[str, Any],
    ) -> None:
        for label, row in capped["results"].items():
            assert row["mean_largest_weight"] <= WEIGHT_CAP + 1e-6, label

    def test_the_capped_run_is_less_concentrated_than_the_uncapped_one(
        self, capped: dict[str, Any], uncapped: dict[str, Any],
    ) -> None:
        free = uncapped["results"]["riskfolio_nco_ward"]["mean_effective_positions"]
        bound = capped["results"]["riskfolio_nco_ward"]["mean_effective_positions"]
        assert bound > free


class TestTheFeeIsArgusOwnFee:
    """The cost model is `desk/allocation.py:441-451`'s, not a number invented for this test."""

    def test_cost_is_two_sided_turnover_times_the_taker_fee(
        self, uncapped: dict[str, Any],
    ) -> None:
        for label, row in uncapped["results"].items():
            expected = row["mean_turnover"] * TAKER_BPS
            assert math.isclose(row["mean_cost_bps"], expected, rel_tol=1e-9), label

    def test_net_return_is_gross_minus_cost(self, uncapped: dict[str, Any]) -> None:
        for label, row in uncapped["results"].items():
            assert math.isclose(
                row["mean_net_return_bps"],
                row["mean_gross_return_bps"] - row["mean_cost_bps"],
                rel_tol=1e-9, abs_tol=1e-9,
            ), label

    def test_the_concentrated_allocator_is_charged_more_than_argus(
        self, uncapped: dict[str, Any],
    ) -> None:
        """Not a result, a sanity check on the accounting: NCO refits to a far more concentrated
        book each window, so if it were not paying more turnover than ARGUS the fee would not be
        reaching it at all."""
        nco = uncapped["results"]["riskfolio_nco_ward"]
        argus = uncapped["results"]["argus_hrp"]
        assert nco["mean_turnover"] > argus["mean_turnover"]
        assert nco["mean_cost_bps"] > argus["mean_cost_bps"]


class TestNothingSeesTheFuture:
    """Leverage is the one place a lookahead could enter, so it is the one place checked hardest."""

    def test_leverage_never_exceeds_the_ceiling(self, uncapped: dict[str, Any]) -> None:
        for label, row in uncapped["results"].items():
            assert 0 < row["mean_leverage"] <= MAX_LEVERAGE + 1e-12, label

    def test_leverage_tracks_the_estimation_window_not_the_held_out_one(
        self, uncapped: dict[str, Any],
    ) -> None:
        """A leverage set from realised out-of-sample volatility would put every allocator's
        LEVERED realised volatility on top of the target. It does not, and the gap is the whole
        proof that the forecast is a forecast."""
        target = uncapped["target_vol_bps"]
        gaps = [
            abs(row["mean_levered_oos_vol_bps"] - target)
            for row in uncapped["results"].values()
        ]
        assert max(gaps) > 1e-6

    def test_the_levered_book_is_scaled_toward_the_common_risk_budget(
        self, uncapped: dict[str, Any],
    ) -> None:
        """Each allocator's levered volatility should sit far closer to the shared budget than its
        unlevered one does, or the common-risk comparison is not comparing at common risk."""
        target = uncapped["target_vol_bps"]
        nco = uncapped["results"]["riskfolio_nco_ward"]
        assert abs(nco["mean_levered_oos_vol_bps"] - target) < abs(
            nco["mean_oos_vol_bps"] - target
        )


class TestTheVerdictMatchesItsNumbers:
    def test_the_booleans_agree_with_the_means_they_summarise(
        self, uncapped: dict[str, Any],
    ) -> None:
        nco = uncapped["results"]["riskfolio_nco_ward"]
        argus = uncapped["results"]["argus_hrp"]
        assert uncapped["nco_beats_argus_on_oos_vol"] == (
            nco["mean_oos_vol_bps"] < argus["mean_oos_vol_bps"]
        )
        assert uncapped["nco_beats_argus_on_net_return"] == (
            nco["mean_net_return_bps"] > argus["mean_net_return_bps"]
        )
        assert uncapped["nco_beats_argus_on_levered_net_return"] == (
            nco["mean_levered_net_return_bps"] > argus["mean_levered_net_return_bps"]
        )

    def test_wins_and_losses_account_for_every_compared_window(
        self, uncapped: dict[str, Any],
    ) -> None:
        for label, row in uncapped["results"].items():
            if label == "argus_hrp":
                continue
            for criterion in ("oos_vol", "net_return", "levered_net_return"):
                compared = row[f"{criterion}_windows_compared"]
                wins = row[f"{criterion}_windows_beating_argus"]
                assert 0 <= wins <= compared <= uncapped["n_origins"], (label, criterion)

    def test_the_sign_test_matches_an_independent_recomputation(
        self, uncapped: dict[str, Any],
    ) -> None:
        for label, row in uncapped["results"].items():
            if label == "argus_hrp":
                continue
            for criterion in ("oos_vol", "net_return", "levered_net_return"):
                wins = row[f"{criterion}_windows_beating_argus"]
                trials = row[f"{criterion}_windows_compared"]
                assert row[f"{criterion}_sign_test_p"] == _two_sided_binomial_p(wins, trials)

    def test_the_verdict_counts_exactly_the_checks_it_lists(
        self, uncapped: dict[str, Any], capped: dict[str, Any],
    ) -> None:
        universes = {
            name: {
                "nco_beats_argus_on_oos_vol": flag,
                "nco_beats_argus_on_net_return": flag,
                "nco_beats_argus_on_levered_net_return": flag,
            }
            for name, flag in (("without_sqqq", True), ("without_qqq_and_sqqq", False))
        }
        verdict = summarise_verdict(
            {"uncapped": uncapped, "capped": capped, "universes": universes}
        )
        assert verdict["checks_total"] == len(verdict["checks"])
        assert verdict["checks_nco_still_wins"] == sum(
            1 for v in verdict["checks"].values() if v
        )
        assert verdict["published_verdict_survives_every_ablation"] == (
            verdict["checks_nco_still_wins"] == verdict["checks_total"]
        )
        # The universe block above was built with `without_qqq_and_sqqq` losing, so the "every
        # ablation" flag must be False no matter what the live data did.
        assert verdict["published_verdict_survives_every_ablation"] is False


class TestTheUniverseAblation:
    def test_it_really_removes_the_inverse_pair(
        self, columns: dict[str, list[float]],
    ) -> None:
        if INVERSE_PAIR[0] not in columns or INVERSE_PAIR[1] not in columns:
            pytest.skip("the inverse pair is not in this fetch; the ablation is not applicable")
        out = run_universe_ablation(
            columns, train_bars=TRAIN, test_bars=TEST, step_bars=STEP,
        )
        assert out["full_universe"]["n_instruments"] == len(columns)
        assert out["without_sqqq"]["n_instruments"] == len(columns) - 1
        assert out["without_qqq_and_sqqq"]["n_instruments"] == len(columns) - 2

    def test_every_universe_reports_a_ratio_consistent_with_its_two_means(
        self, columns: dict[str, list[float]],
    ) -> None:
        out = run_universe_ablation(
            columns, train_bars=TRAIN, test_bars=TEST, step_bars=STEP,
        )
        for label, row in out.items():
            assert math.isclose(
                row["nco_vol_ratio_vs_argus"],
                row["nco_mean_oos_vol_bps"] / row["argus_mean_oos_vol_bps"],
                rel_tol=1e-12,
            ), label
            assert row["nco_beats_argus_on_oos_vol"] == (row["nco_vol_ratio_vs_argus"] < 1.0)


class TestTheArithmeticItself:
    def test_the_binomial_p_value_is_exact(self) -> None:
        assert _two_sided_binomial_p(0, 0) is None
        assert _two_sided_binomial_p(2, 4) == 1.0
        assert _two_sided_binomial_p(4, 4) == pytest.approx(2 / 16)
        assert _two_sided_binomial_p(22, 24) == pytest.approx(3.5881996154785156e-05)

    def test_turnover_is_two_sided(self) -> None:
        from argus.eval.allocation_cost_ablation import _turnover

        # Moving half the book from A to B is 0.5 of weight sold and 0.5 bought: 1.0 two-sided,
        # which is `desk/allocation.py`'s own convention and twice the "one-way" figure.
        assert _turnover({"A": 0.0, "B": 1.0}, {"A": 0.5, "B": 0.5}) == pytest.approx(1.0)
        assert _turnover({"A": 0.5}, {"A": 0.5}) == 0.0

    def test_a_missing_name_counts_as_a_zero_holding(self) -> None:
        from argus.eval.allocation_cost_ablation import _turnover

        assert _turnover({"A": 1.0}, {}) == pytest.approx(1.0)


class TestReporting:
    def test_render_names_both_runs_the_universes_and_the_verdict(
        self, uncapped: dict[str, Any], capped: dict[str, Any],
    ) -> None:
        universes: dict[str, Any] = {
            name: {
                "n_instruments": 11,
                "argus_mean_oos_vol_bps": 14.0,
                "nco_mean_oos_vol_bps": 9.0,
                "nco_vol_ratio_vs_argus": 9.0 / 14.0,
                "nco_windows_beating_argus_on_vol": 3,
                "nco_windows_compared": 4,
                "nco_sign_test_p": 0.625,
                "nco_mean_effective_positions": 3.2,
                "nco_beats_argus_on_oos_vol": True,
                "nco_beats_argus_on_net_return": True,
                "nco_beats_argus_on_levered_net_return": True,
            }
            for name in ("full_universe", "without_sqqq", "without_qqq_and_sqqq")
        }
        report = {"uncapped": uncapped, "capped": capped, "universes": universes}
        report["verdict"] = summarise_verdict(report)
        text = render(report)
        assert "argus_hrp" in text
        assert "riskfolio_nco_ward" in text
        assert "without_qqq_and_sqqq" in text
        assert "VERDICT" in text

    def test_the_scope_statement_separates_claimed_from_not_claimed(self) -> None:
        assert "CLAIMED" in SCOPE_STATEMENT
        assert "NOT CLAIMED" in SCOPE_STATEMENT
        assert "Riskfolio-Lib 7.3.0" in SCOPE_STATEMENT


def test_the_module_reads_the_same_universe_the_comparison_did(
    columns: Mapping[str, Sequence[float]],
) -> None:
    """A comparison and its ablation must be about the same instruments or the ablation is a
    different experiment wearing the same name."""
    assert len(columns) >= 4
    assert all(len(v) == len(next(iter(columns.values()))) for v in columns.values())
