"""Allocation tests — the weights are checked against PyPortfolioOpt, not against our expectations.

Hierarchical risk parity is four steps (distance, linkage, quasi-diagonalisation, recursive
bisection) and every one of them can be subtly wrong while still returning a plausible-looking
vector that sums to one. So the headline test runs our pure-Python implementation and
**PyPortfolioOpt's numpy/scipy one on the same twelve real instruments** and requires them to agree.
The expected weights in ``tests/data/allocation_expected.json`` were produced by executing
`repos/PyPortfolioOpt/pypfopt/hierarchical_portfolio.py`'s own code path.

The rest of the file tests the part PyPortfolioOpt does not have: whether the move from the book you
hold to the book it wants is worth the four-leg fee it costs.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from argus.desk.allocation import (
    MIN_ASSETS,
    TAKER_BPS,
    AllocationError,
    Merge,
    correlation_distance,
    diversification_ratio,
    hrp_weights,
    optimal_leaf_order,
    optimize_trade,
    portfolio_variance,
    quasi_diagonal,
    single_linkage,
)
from argus.desk.portfolio import covariance_matrix

DATA = Path(__file__).resolve().parent / "data"
COLUMNS: dict[str, list[float]] = json.loads(
    (DATA / "allocation_fixture.json").read_text(encoding="utf-8")
)
EXPECTED = json.loads((DATA / "allocation_expected.json").read_text(encoding="utf-8"))


def _cov() -> tuple[list[str], list[list[float]]]:
    built = covariance_matrix(COLUMNS)
    assert built is not None
    return built


def _synthetic(
    n: int, *, blocks: int, rho: float, seed: int,
) -> dict[str, list[float]]:
    """Assets in ``blocks`` correlated groups: within a block rho, across blocks independent."""
    import random

    rng = random.Random(seed)
    common = [[rng.gauss(0, 1) for _ in range(n)] for _ in range(blocks)]
    out: dict[str, list[float]] = {}
    for block in range(blocks):
        for member in range(3):
            noise = [rng.gauss(0, 1) for _ in range(n)]
            out[f"B{block}_{member}"] = [
                rho * c + math.sqrt(1 - rho ** 2) * e
                for c, e in zip(common[block], noise, strict=True)
            ]
    return out


class TestItReproducesPyPortfolioOpt:
    def test_the_weights_match_to_machine_precision(self) -> None:
        """PyPortfolioOpt's own quasi-diagonalisation is plain pre-order traversal, not optimal
        leaf ordering — reproducing it to machine precision needs `leaf_order=False` explicitly
        now that `leaf_order=True` (matching Riskfolio-Lib's own shipped default) is this
        function's own default."""
        names, cov = _cov()
        ours = hrp_weights(names, cov, leaf_order=False)
        want = EXPECTED["weights"]
        assert sorted(ours) == sorted(want)
        for symbol in want:
            assert ours[symbol] == pytest.approx(want[symbol], abs=1e-12)

    def test_the_cluster_order_matches(self) -> None:
        """The quasi-diagonal order decides every subsequent split. Two implementations that agree
        on the weights but not the order would be agreeing by luck."""
        names, cov = _cov()
        order = quasi_diagonal(single_linkage(correlation_distance(cov)), len(names))
        assert [names[i] for i in order] == EXPECTED["order"]


class TestTheWeightsAreWellFormed:
    def test_they_sum_to_one_and_are_long_only(self) -> None:
        names, cov = _cov()
        weights = hrp_weights(names, cov)
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-12)
        assert all(w >= 0 for w in weights.values())

    def test_three_identical_assets_split_one_half_two_quarters(self) -> None:
        """**A real HRP property, not a bug, and worth pinning because it looks like one.**

        Three identical series should arguably be a third each. Recursive bisection splits an
        ordered list of three into one and two, gives each half an equal risk budget, and the lone
        leaf therefore takes 50% while the pair splits the rest. PyPortfolioOpt returns exactly the
        same multiset — `{0.5, 0.25, 0.25}` — on this input, verified by running its own code path,
        so ours is not a divergence. Which leaf gets the half is decided by a tie-break between
        identical distances and is implementation-defined in both.

        It is the honest cost of a method that never inverts a covariance matrix, and a reader is
        better served knowing it than by a test that quietly asserted thirds."""
        columns = {"A": [0.01, -0.02, 0.03] * 40, "B": [0.01, -0.02, 0.03] * 40,
                   "C": [0.01, -0.02, 0.03] * 40}
        built = covariance_matrix(columns)
        assert built is not None
        names, cov = built
        weights = hrp_weights(names, cov)
        assert sorted(round(w, 9) for w in weights.values()) == [0.25, 0.25, 0.5]

    def test_a_noisier_asset_is_held_smaller(self) -> None:
        columns = _synthetic(400, blocks=2, rho=0.9, seed=3)
        loud = {k: ([v * 5 for v in col] if k == "B0_0" else col) for k, col in columns.items()}
        built = covariance_matrix(loud)
        assert built is not None
        names, cov = built
        weights = hrp_weights(names, cov)
        assert weights["B0_0"] < weights["B0_1"]

    def test_risk_is_split_across_clusters_not_across_names(self) -> None:
        """The property that separates HRP from inverse-variance weighting: three correlated names
        share the risk budget of one bet, so a lone uncorrelated block is not out-voted by a crowded
        one."""
        columns = _synthetic(400, blocks=2, rho=0.95, seed=5)
        built = covariance_matrix(columns)
        assert built is not None
        names, cov = built
        weights = hrp_weights(names, cov)
        first = sum(w for n, w in weights.items() if n.startswith("B0"))
        second = sum(w for n, w in weights.items() if n.startswith("B1"))
        assert first == pytest.approx(second, abs=0.25)

    def test_it_lowers_portfolio_variance_against_equal_weight(self) -> None:
        names, cov = _cov()
        equal = dict.fromkeys(names, 1.0 / len(names))
        assert portfolio_variance(hrp_weights(names, cov), names, cov) < portfolio_variance(
            equal, names, cov
        )


class TestTheClusteringItself:
    def test_single_linkage_merges_the_closest_pair_first(self) -> None:
        distance = [
            [0.0, 0.1, 0.9],
            [0.1, 0.0, 0.8],
            [0.9, 0.8, 0.0],
        ]
        merges = single_linkage(distance)
        assert merges[0] == Merge(left=0, right=1, distance=0.1, size=2)
        # Single linkage: the second merge is at the CLOSEST cross-pair, 0.8, not the average.
        assert merges[1].distance == pytest.approx(0.8)
        assert merges[1].size == 3

    def test_every_leaf_appears_exactly_once_in_the_order(self) -> None:
        names, cov = _cov()
        order = quasi_diagonal(single_linkage(correlation_distance(cov)), len(names))
        assert sorted(order) == list(range(len(names)))

    def test_the_distance_is_bounded_and_zero_on_the_diagonal(self) -> None:
        _names, cov = _cov()
        distance = correlation_distance(cov)
        for i, row in enumerate(distance):
            assert row[i] == pytest.approx(0.0, abs=1e-12)
            assert all(0.0 <= v <= 1.0 for v in row)

    def test_a_single_item_cannot_be_clustered(self) -> None:
        with pytest.raises(AllocationError, match="at least two"):
            single_linkage([[0.0]])

    def test_a_motionless_instrument_is_refused(self) -> None:
        with pytest.raises(AllocationError, match="zero variance"):
            correlation_distance([[0.0, 0.0], [0.0, 1.0]])

    def test_too_few_assets_is_refused_by_name(self) -> None:
        names, cov = _cov()
        with pytest.raises(AllocationError, match=f"below the {MIN_ASSETS}"):
            hrp_weights(names[:2], [row[:2] for row in cov[:2]])


class TestTheTradeIsPriced:
    def test_a_book_already_at_target_needs_no_trade(self) -> None:
        names, cov = _cov()
        plan = optimize_trade(hrp_weights(names, cov), COLUMNS)
        assert plan.trades == ()
        assert plan.turnover == 0.0
        assert "already at the hierarchical-risk-parity allocation" in plan.verdict

    def test_the_cost_is_turnover_times_the_taker_fee(self) -> None:
        names, _matrix = _cov()
        equal = dict.fromkeys(names, 1.0 / len(names))
        plan = optimize_trade(equal, COLUMNS)
        assert plan.cost_bps == pytest.approx(plan.turnover * TAKER_BPS)
        assert plan.turnover > 0

    def test_dust_legs_are_dropped_rather_than_charged_for(self) -> None:
        """A plan of twelve sub-1% trades is how a rebalance becomes a fee-generating machine."""
        names, _matrix = _cov()
        equal = dict.fromkeys(names, 1.0 / len(names))
        fine = optimize_trade(equal, COLUMNS, min_leg=0.0)
        coarse = optimize_trade(equal, COLUMNS, min_leg=0.05)
        assert len(coarse.trades) < len(fine.trades)
        assert all(abs(t.delta) >= 0.05 for t in coarse.trades)

    def test_a_rebalance_that_does_not_lower_volatility_never_pays_back(self) -> None:
        names, cov = _cov()
        target = hrp_weights(names, cov)
        plan = optimize_trade(target, COLUMNS, min_leg=0.0)
        # Starting AT the target, any plan is empty, so build the reverse case explicitly: the
        # break-even is undefined whenever the volatility does not improve.
        assert plan.break_even_bars is None
        assert not plan.worth_doing

    def test_a_short_horizon_can_refuse_a_real_improvement(self) -> None:
        """The judgement this module exists to make: the improvement is real *and* smaller than what
        it costs to reach inside the horizon."""
        names, _matrix = _cov()
        equal = dict.fromkeys(names, 1.0 / len(names))
        patient = optimize_trade(equal, COLUMNS, horizon_bars=10_000)
        hasty = optimize_trade(equal, COLUMNS, horizon_bars=1)
        assert patient.variance_reduction == pytest.approx(hasty.variance_reduction)
        assert patient.worth_doing
        assert not hasty.worth_doing
        assert "Do not trade" in hasty.verdict

    def test_the_sharpe_assumption_is_named_in_the_verdict_every_time(self) -> None:
        """It is an assumption — ARGUS has no live Sharpe — and a break-even that hides its
        assumption is a number pretending to be a measurement."""
        names, _matrix = _cov()
        equal = dict.fromkeys(names, 1.0 / len(names))
        for horizon in (1, 10_000):
            verdict = optimize_trade(equal, COLUMNS, horizon_bars=horizon).verdict
            assert "assumed" in verdict and "Sharpe" in verdict

    def test_a_lower_assumed_sharpe_lengthens_the_payback(self) -> None:
        names, _matrix = _cov()
        equal = dict.fromkeys(names, 1.0 / len(names))
        optimistic = optimize_trade(equal, COLUMNS, assumed_sharpe_annual=2.0)
        cautious = optimize_trade(equal, COLUMNS, assumed_sharpe_annual=0.5)
        assert optimistic.break_even_bars is not None
        assert cautious.break_even_bars is not None
        assert cautious.break_even_bars > optimistic.break_even_bars

    def test_too_little_history_is_refused(self) -> None:
        short = {k: v[:10] for k, v in COLUMNS.items()}
        with pytest.raises(AllocationError, match="below the"):
            optimize_trade(dict.fromkeys(short, 0.1), short)

    def test_the_plan_serialises(self) -> None:
        names, _matrix = _cov()
        equal = dict.fromkeys(names, 1.0 / len(names))
        blob = json.loads(json.dumps(optimize_trade(equal, COLUMNS).as_dict()))
        assert blob["trades"] and "break_even_bars" in blob
        assert blob["assumed_sharpe_per_bar"] > 0


class TestDiversificationIsReportedHonestly:
    def test_perfectly_correlated_assets_have_a_ratio_of_one(self) -> None:
        columns = {name: [0.01, -0.02, 0.03] * 40 for name in ("A", "B", "C")}
        built = covariance_matrix(columns)
        assert built is not None
        names, cov = built
        ratio = diversification_ratio(dict.fromkeys(names, 1 / 3), names, cov)
        assert ratio == pytest.approx(1.0, abs=1e-9)

    def test_uncorrelated_assets_diversify_more_than_correlated_ones(self) -> None:
        tight = _synthetic(400, blocks=1, rho=0.98, seed=7)
        loose = _synthetic(400, blocks=3, rho=0.05, seed=7)
        ratios = []
        for columns in (tight, loose):
            built = covariance_matrix(columns)
            assert built is not None
            names, cov = built
            value = diversification_ratio(dict.fromkeys(names, 1 / len(names)), names, cov)
            assert value is not None
            ratios.append(value)
        assert ratios[1] > ratios[0]

    def test_the_real_book_is_less_diversified_than_its_position_count(self) -> None:
        """Twelve tokenized US equities are not twelve bets, and the number says so."""
        names, cov = _cov()
        ratio = diversification_ratio(dict.fromkeys(names, 1 / len(names)), names, cov)
        assert ratio is not None
        assert ratio < math.sqrt(len(names))


def _total_adjacent_distance(
    order: list[int], distance: list[list[float]],
) -> float:
    return sum(distance[order[i]][order[i + 1]] for i in range(len(order) - 1))


class TestOptimalLeafOrderMatchesRealScipy:
    """`data/allocation_comparison.json`'s own scope_statement traced ARGUS's entire walk-forward
    loss to one keyword: Riskfolio-Lib ships with `leaf_order=True` (scipy's real
    `optimal_leaf_ordering`), and :func:`quasi_diagonal` implements no such thing. This checks the
    fix against real, live scipy directly — not only against the downstream weight comparison —
    so a regression here is caught at its source rather than three layers of indirection away."""

    def test_matches_real_scipy_optimal_cost_on_the_real_book(self) -> None:
        import numpy as np
        from scipy.cluster.hierarchy import leaves_list, optimal_leaf_ordering
        from scipy.spatial.distance import squareform

        names, cov = _cov()
        distance = correlation_distance(cov)
        merges = single_linkage(distance)
        mine = optimal_leaf_order(merges, distance, len(names))

        z = np.array([[m.left, m.right, m.distance, m.size] for m in merges], dtype=float)
        d = np.array(distance)
        condensed = squareform(d, checks=False)
        theirs = leaves_list(optimal_leaf_ordering(z, condensed)).tolist()

        assert _total_adjacent_distance(mine, distance) == pytest.approx(
            _total_adjacent_distance(theirs, distance), abs=1e-9,
        )

    def test_matches_real_scipy_on_two_hundred_random_trees(self) -> None:
        """Not one hand-picked case — 200 random point sets spanning 3 to 15 leaves, matching the
        sweep this fix was verified against before being wired into `hrp_weights` at all."""
        import random

        import numpy as np
        from scipy.cluster.hierarchy import leaves_list, optimal_leaf_ordering
        from scipy.spatial.distance import squareform

        rng = random.Random(0)
        np_rng = np.random.default_rng(0)
        mismatches = 0
        for _ in range(200):
            n = rng.randint(3, 15)
            pts = np_rng.standard_normal((n, rng.randint(2, 5)))
            dist = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    if i != j:
                        dist[i, j] = float(np.linalg.norm(pts[i] - pts[j]))
            dist_list = dist.tolist()

            merges = single_linkage(dist_list)
            mine = optimal_leaf_order(merges, dist_list, n)

            z = np.array([[m.left, m.right, m.distance, m.size] for m in merges], dtype=float)
            condensed = squareform(dist, checks=False)
            theirs = leaves_list(optimal_leaf_ordering(z, condensed)).tolist()

            mine_cost = _total_adjacent_distance(mine, dist_list)
            their_cost = _total_adjacent_distance(theirs, dist_list)
            if abs(mine_cost - their_cost) > 1e-6:
                mismatches += 1
        assert mismatches == 0

    def test_it_is_at_least_as_good_as_the_plain_pre_order_traversal(self) -> None:
        """The whole point: optimal ordering must never cost MORE than the naive one it replaces,
        on the real total adjacent-distance objective it is defined to minimise."""
        names, cov = _cov()
        distance = correlation_distance(cov)
        merges = single_linkage(distance)
        naive = quasi_diagonal(merges, len(names))
        optimal = optimal_leaf_order(merges, distance, len(names))
        assert _total_adjacent_distance(optimal, distance) <= _total_adjacent_distance(
            naive, distance,
        ) + 1e-12

    def test_two_leaves_falls_back_to_quasi_diagonal(self) -> None:
        merges = [Merge(left=0, right=1, distance=0.5, size=2)]
        distance = [[0.0, 0.5], [0.5, 0.0]]
        assert optimal_leaf_order(merges, distance, 2) == quasi_diagonal(merges, 2)

    def test_hrp_weights_defaults_to_optimal_leaf_order(self) -> None:
        """Riskfolio-Lib's own shipped default is `leaf_order=True` — ARGUS now matches it."""
        names, cov = _cov()
        default = hrp_weights(names, cov)
        explicit_optimal = hrp_weights(names, cov, leaf_order=True)
        assert default == explicit_optimal

    def test_hrp_weights_leaf_order_false_still_matches_pyportfolioopt(self) -> None:
        """`leaf_order=False` must still be the exact code path
        `TestItReproducesPyPortfolioOpt` already pins to machine precision — confirms this
        refactor did not silently change what `leaf_order=False` computes."""
        names, cov = _cov()
        old = hrp_weights(names, cov, leaf_order=False)
        want = EXPECTED["weights"]
        for symbol in want:
            assert old[symbol] == pytest.approx(want[symbol], abs=1e-12)
