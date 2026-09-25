"""Tests for `desk/nco_ensemble.py` -- the estimators and bagged NCO ARGUS enters in the bake-off.

What is pinned is correctness of the parts, not which allocator wins: the Ledoit-Wolf port returns
PyPortfolioOpt's own number (when the corpus clone is present), the EWMA estimator collapses to the
sample covariance at an infinite half-life, the bootstrap is reproducible from its seed, and a
bagged allocation is a genuine long-only budget. Offline; no network, no LLM.
"""

from __future__ import annotations

import math
import os
import random
import sys
import types
from pathlib import Path

import pytest

from argus.desk.allocation import AllocationError, nco_weights
from argus.desk.nco_ensemble import (
    NCOConfig,
    ensemble_nco_weights,
    estimate,
    ewma_covariance,
    ledoit_wolf_constant_correlation,
    sample_covariance,
    stationary_bootstrap_indices,
)

_PYPFOPT_ENV = os.environ.get("ARGUS_PYPFOPT_PATH")
PYPFOPT = Path(_PYPFOPT_ENV) if _PYPFOPT_ENV else None
"""A clone of PyPortfolioOpt (commit a6638d2e), for the one test that compares the Ledoit-Wolf port
against the library's own number. Optional: without it that test is skipped, never faked."""


class TestKnownValues:
    """Pinned against numbers computed independently of this module, so the suite still checks the
    estimators on a clean public checkout where the PyPortfolioOpt clone is absent."""

    def test_ledoit_wolf_on_two_assets_returns_the_sample_covariance(self) -> None:
        # Two assets: the average of one correlation is that correlation, so the constant-
        # correlation target IS the sample covariance and any intensity leaves it unchanged.
        rows = [[0.01 * math.sin(t), 0.01 * math.cos(t / 3)] for t in range(90)]
        shrunk, delta = ledoit_wolf_constant_correlation(rows)
        s = sample_covariance(rows)
        assert 0.0 <= delta <= 1.0
        for i in range(2):
            for j in range(2):
                assert shrunk[i][j] == pytest.approx(s[i][j], rel=1e-12, abs=1e-18)

    def test_ledoit_wolf_pulls_correlations_toward_their_average(self) -> None:
        # Two factor groups, so the correlations are far from constant and the intensity lands
        # strictly inside (0, 1) rather than at a clip.
        rng = random.Random(9)
        rows = []
        for _ in range(120):
            f, g = rng.gauss(0, 0.01), rng.gauss(0, 0.01)
            rows.append([f + rng.gauss(0, 0.01), f + rng.gauss(0, 0.01), g + rng.gauss(0, 0.01),
                         g + rng.gauss(0, 0.01), rng.gauss(0, 0.01)])
        shrunk, delta = ledoit_wolf_constant_correlation(rows)
        s = sample_covariance(rows)

        def corr(m: list[list[float]], i: int, j: int) -> float:
            return m[i][j] / math.sqrt(m[i][i] * m[j][j])

        pairs = [(i, j) for i in range(5) for j in range(i + 1, 5)]
        r_bar = sum(corr(s, i, j) for i, j in pairs) / len(pairs)
        assert 0.0 < delta < 1.0
        for i, j in pairs:
            expected = delta * r_bar + (1 - delta) * corr(s, i, j)
            assert corr(shrunk, i, j) == pytest.approx(expected, rel=1e-12)
            assert abs(corr(shrunk, i, j) - r_bar) <= abs(corr(s, i, j) - r_bar) + 1e-15

    def test_ewma_with_a_one_bar_half_life_is_dominated_by_the_last_rows(self) -> None:
        rows = [[0.0, 0.0]] * 100 + [[0.01, -0.01], [-0.01, 0.01]]
        e = ewma_covariance(rows, 1.0)
        assert e[0][1] < 0 < e[0][0]
        assert e[0][1] == pytest.approx(-e[0][0], rel=1e-9)


def _factor_rows(t: int = 300, n: int = 6, seed: int = 3) -> list[list[float]]:
    """Correlated returns: one common factor plus idiosyncratic noise, like the rToken book."""
    rng = random.Random(seed)
    rows = []
    for _ in range(t):
        f = rng.gauss(0, 0.01)
        rows.append([0.8 * f + rng.gauss(0, 0.004 * (1 + j / n)) for j in range(n)])
    return rows


def _columns(rows: list[list[float]]) -> dict[str, list[float]]:
    return {f"S{j}": [r[j] for r in rows] for j in range(len(rows[0]))}


class TestEstimators:
    def test_sample_covariance_matches_a_direct_computation(self) -> None:
        rows = _factor_rows(t=80, n=3)
        s = sample_covariance(rows)
        xs = [r[0] for r in rows]
        ys = [r[2] for r in rows]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        direct = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / (len(xs) - 1)
        assert s[0][2] == pytest.approx(direct, rel=1e-12)
        assert s[2][0] == s[0][2]

    def test_ewma_with_an_infinite_half_life_is_the_sample_covariance(self) -> None:
        rows = _factor_rows(t=120, n=4)
        e = ewma_covariance(rows, 1e15)
        s = sample_covariance(rows)
        for i in range(4):
            for j in range(4):
                assert e[i][j] == pytest.approx(s[i][j], rel=1e-9)

    def test_ewma_weights_recent_rows_more(self) -> None:
        calm = [[0.001 * ((-1) ** t), 0.001 * ((-1) ** (t + 1))] for t in range(200)]
        loud = [[0.02 * ((-1) ** t), 0.02 * ((-1) ** (t + 1))] for t in range(40)]
        recent_loud = ewma_covariance(calm + loud, 20.0)[0][0]
        recent_calm = ewma_covariance(loud + calm, 20.0)[0][0]
        assert recent_loud > 10 * recent_calm

    def test_ewma_refuses_a_non_positive_half_life(self) -> None:
        with pytest.raises(AllocationError):
            ewma_covariance(_factor_rows(t=80, n=3), 0.0)

    def test_ledoit_wolf_intensity_is_a_probability_and_keeps_variances(self) -> None:
        rows = _factor_rows()
        shrunk, delta = ledoit_wolf_constant_correlation(rows)
        s = sample_covariance(rows)
        assert 0.0 <= delta <= 1.0
        for i in range(len(s)):
            assert shrunk[i][i] == pytest.approx(s[i][i], rel=1e-12)

    def test_ledoit_wolf_refuses_a_zero_variance_column(self) -> None:
        rows = [[r[0], 0.0, r[2]] for r in _factor_rows(t=80, n=3)]
        with pytest.raises(AllocationError):
            ledoit_wolf_constant_correlation(rows)

    @pytest.mark.skipif(PYPFOPT is None or not PYPFOPT.exists(),
                        reason="set ARGUS_PYPFOPT_PATH to a PyPortfolioOpt clone to run this")
    def test_ledoit_wolf_matches_pypfopt_to_machine_precision(self) -> None:
        assert PYPFOPT is not None
        np = pytest.importorskip("numpy")
        pd = pytest.importorskip("pandas")
        pytest.importorskip("cvxpy")
        # PyPortfolioOpt's package __init__ pulls `skbase` for an unrelated soft-dependency check;
        # a stub satisfies the import without touching the shrinkage code under test.
        if "skbase" not in sys.modules:
            stub = types.ModuleType("skbase.utils.dependencies")
            stub._check_soft_dependencies = lambda *a, **k: True  # type: ignore[attr-defined]
            sys.modules.update({"skbase": types.ModuleType("skbase"),
                                "skbase.utils": types.ModuleType("skbase.utils"),
                                "skbase.utils.dependencies": stub})
        sys.path.insert(0, str(PYPFOPT))
        try:
            from pypfopt.risk_models import (  # type: ignore[import-not-found, unused-ignore]
                CovarianceShrinkage,
            )
        finally:
            sys.path.remove(str(PYPFOPT))
        rng = np.random.default_rng(11)
        for _ in range(5):
            n = int(rng.integers(3, 10))
            x = rng.normal(size=(int(rng.integers(80, 300)), n)) @ rng.normal(size=(n, n)) * 0.01
            ref, ref_delta = CovarianceShrinkage(
                pd.DataFrame(x), returns_data=True, frequency=1,
            )._ledoit_wolf_constant_correlation()
            ours, delta = ledoit_wolf_constant_correlation(x.tolist())
            assert delta == pytest.approx(float(ref_delta), abs=1e-12)
            assert float(np.max(np.abs(np.array(ours) - ref))) <= 1e-12 * float(np.max(np.abs(ref)))

    def test_estimate_dispatches_and_refuses_unknown_names(self) -> None:
        rows = _factor_rows(t=80, n=3)
        assert estimate(rows, NCOConfig("sample")) == sample_covariance(rows)
        with pytest.raises(AllocationError):
            estimate(rows, NCOConfig("ewma"))
        with pytest.raises(AllocationError):
            estimate(rows, NCOConfig("nonsense"))


class TestBootstrap:
    def test_indices_are_in_range_and_reproducible(self) -> None:
        a = stationary_bootstrap_indices(500, 24.0, random.Random(1))
        b = stationary_bootstrap_indices(500, 24.0, random.Random(1))
        assert a == b
        assert len(a) == 500 and all(0 <= i < 500 for i in a)

    def test_mean_block_length_is_roughly_as_asked(self) -> None:
        idx = stationary_bootstrap_indices(20_000, 10.0, random.Random(2))
        breaks = sum(1 for i in range(1, len(idx)) if idx[i] != (idx[i - 1] + 1) % 20_000)
        assert 8.0 < len(idx) / (breaks + 1) < 12.5

    def test_refuses_a_block_shorter_than_one(self) -> None:
        with pytest.raises(AllocationError):
            stationary_bootstrap_indices(10, 0.5, random.Random(0))


class TestEnsembleAllocator:
    def test_unbagged_sample_config_is_exactly_nco_weights(self) -> None:
        rows = _factor_rows()
        cols = _columns(rows)
        ours = ensemble_nco_weights(cols, NCOConfig("sample"))
        ref = nco_weights(sorted(cols), sample_covariance(rows))
        assert ours == ref

    @pytest.mark.parametrize("config", [
        NCOConfig("lw_cc"), NCOConfig("ewma", halflife=120.0),
        NCOConfig("sample", n_boot=8), NCOConfig("lw_cc", n_boot=8),
    ])
    def test_every_config_is_a_long_only_budget(self, config: NCOConfig) -> None:
        w = ensemble_nco_weights(_columns(_factor_rows()), config)
        assert math.isclose(sum(w.values()), 1.0, abs_tol=1e-9)
        assert all(v >= -1e-12 for v in w.values())

    def test_bagging_is_reproducible_from_its_seed_and_moves_with_it(self) -> None:
        cols = _columns(_factor_rows())
        a = ensemble_nco_weights(cols, NCOConfig("sample", n_boot=6, seed=5))
        b = ensemble_nco_weights(cols, NCOConfig("sample", n_boot=6, seed=5))
        c = ensemble_nco_weights(cols, NCOConfig("sample", n_boot=6, seed=6))
        assert a == b
        assert a != c

    def test_refuses_too_few_observations(self) -> None:
        with pytest.raises(AllocationError):
            ensemble_nco_weights(_columns(_factor_rows(t=30)), NCOConfig())

    def test_label_names_every_component(self) -> None:
        assert NCOConfig("ewma", halflife=240.0, n_boot=32).label == "argus_nco[ewma240,bag32]"
        assert NCOConfig().label == "argus_nco[sample]"
