"""Tests for the general-purpose rival comparison of pairwise cointegration screening.

The rivals are real, installed libraries (statsmodels, arch, scipy), called here exactly as the
comparison calls them. No network: the real rToken universe is read from its frozen snapshot, and
the Monte Carlo result is read from the committed artefact rather than re-run (a full run is a
multi-hour computation; `python -m argus.eval.general_coint_comparison` reproduces it).
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import pytest

from argus.eval.artefact import is_strict
from argus.eval.general_coint_comparison import (
    ARTEFACT_PATH,
    CONFIRMED,
    PRODUCTION,
    SNAPSHOT_PATH,
    GeneralCointComparisonError,
    PipelineScore,
    build_universe,
    controls_fdr,
    decide,
    evaluate_series,
    load_snapshot,
    null_universe,
    paired_power_gain,
    planted_universe,
    reject,
    rival_pvalues,
    score_pipeline,
)


class TestUniverses:
    def test_null_has_no_true_pair_and_the_production_shape(self) -> None:
        u = null_universe(1)
        assert len(u.series) == 12
        assert {len(v) for v in u.series.values()} == {1440}
        assert not u.truth_train and not u.truth_forward

    def test_planted_truth_is_the_within_cluster_pairs(self) -> None:
        u = planted_universe(1)
        assert len(u.series) == 12
        assert len(u.truth_train) == 7
        assert u.truth_forward == u.truth_train
        assert all(a[0] == b[0] for a, b in u.truth_train)  # same cluster letter

    def test_broken_keeps_the_in_sample_truth_and_loses_four_pairs_forward(self) -> None:
        u = planted_universe(1, broken=True)
        assert len(u.truth_train) == 7
        assert len(u.truth_forward) == 3
        assert u.truth_forward < u.truth_train

    def test_same_seed_same_universe(self) -> None:
        assert build_universe("planted", 7).series == build_universe("planted", 7).series
        assert build_universe("null", 7).series != build_universe("null", 8).series

    def test_unknown_scenario_is_refused(self) -> None:
        with pytest.raises(GeneralCointComparisonError):
            build_universe("bogus", 1)


class TestCorrections:
    def test_argus_bh_matches_statsmodels_and_scipy_on_random_pvalues(self) -> None:
        rng = random.Random(3)
        for _ in range(200):
            m = rng.randint(1, 80)
            p = [rng.random() ** rng.choice((1, 3, 6)) for _ in range(m)]
            a = reject(p, "argus_bh")
            assert a == reject(p, "sm_bh") == reject(p, "scipy_bh")

    def test_by_is_never_more_permissive_than_bh(self) -> None:
        rng = random.Random(4)
        for _ in range(100):
            p = [rng.random() ** 4 for _ in range(40)]
            by, bh = reject(p, "sm_by"), reject(p, "sm_bh")
            assert all(not b or h for b, h in zip(by, bh, strict=True))
            assert reject(p, "scipy_by") == by

    def test_nan_is_never_rejected(self) -> None:
        assert reject([math.nan, 1e-9], "argus_bh") == [False, True]
        assert reject([math.nan, 1e-9], "sm_holm") == [False, True]

    def test_unknown_correction_is_refused(self) -> None:
        with pytest.raises(GeneralCointComparisonError):
            reject([0.1], "made_up")


def _result(found: dict[str, list[str]], truth: list[str], strength: dict[str, str] | None = None
            ) -> dict[str, Any]:
    return {"discoveries": found, "truth_forward": truth, "strength": strength or {}}


class TestScoring:
    def test_fdp_fwer_and_power_on_a_hand_built_case(self) -> None:
        results = [
            _result({"x": ["A/B", "Z/Y"]}, ["A/B", "C/D"], {"A/B": "s1", "C/D": "s2"}),
            _result({"x": []}, ["A/B", "C/D"], {"A/B": "s1", "C/D": "s2"}),
        ]
        s = score_pipeline(results, "x")
        assert s.fdr == pytest.approx(0.25)  # (1/2 + 0) / 2
        assert s.fwer == pytest.approx(0.5)
        assert s.power == pytest.approx(0.25)  # (1/2 + 0) / 2
        assert s.power_by_strength == {"s1": 0.5, "s2": 0.0}

    def test_null_scenario_has_no_power(self) -> None:
        s = score_pipeline([_result({"x": ["A/B"]}, [])], "x")
        assert s.power is None and s.fwer == 1.0 and s.fdr == 1.0

    def test_paired_gain_is_computed_on_the_same_universes(self) -> None:
        results = [
            _result({"a": ["A/B"], "b": ["A/B", "C/D"]}, ["A/B", "C/D"]),
            _result({"a": [], "b": ["A/B"]}, ["A/B", "C/D"]),
        ]
        gain, se = paired_power_gain(results, "b", "a")
        assert gain == pytest.approx(0.5)
        assert se == pytest.approx(0.0)

    def test_controls_fdr_allows_two_standard_errors(self) -> None:
        def score(fdr: float, se: float) -> PipelineScore:
            return PipelineScore("x", 10, fdr, se, fdr, None, None, {}, 0.0)

        assert controls_fdr(score(0.06, 0.01))
        assert not controls_fdr(score(0.08, 0.01))


def _scores(fdr: dict[str, float], power: dict[str, float]) -> dict[str, PipelineScore]:
    return {p: PipelineScore(p, 50, fdr[p], 0.005, fdr[p], power.get(p), 0.01, {}, 1.0)
            for p in fdr}


class TestDecide:
    PIPES = (PRODUCTION, CONFIRMED, "arch_po_pz|sm_bh", "argus_eg+pre05|argus_bh")

    def _planted(self, gains: dict[str, list[int]]) -> list[dict[str, Any]]:
        truth = ["A/B", "C/D"]
        out = []
        for i in range(50):
            found = {p: truth[:gains[p][i % len(gains[p])]] for p in self.PIPES}
            out.append(_result(found, truth))
        return out

    def test_a_rival_that_controls_fdr_and_finds_more_wins(self) -> None:
        fdr = dict.fromkeys(self.PIPES, 0.02)
        scores = {"null": _scores(fdr, {}), "planted": _scores(fdr, {})}
        planted = self._planted({PRODUCTION: [0], CONFIRMED: [0], "arch_po_pz|sm_bh": [2],
                                 "argus_eg+pre05|argus_bh": [0]})
        d = decide(scores, planted)
        assert d["verdict"] == "rival_wins"
        assert d["best_general_pipeline"]["pipeline"] == "arch_po_pz|sm_bh"

    def test_production_that_fails_fdr_loses_to_a_rival_that_holds_it(self) -> None:
        fdr = {PRODUCTION: 0.20, CONFIRMED: 0.01, "arch_po_pz|sm_bh": 0.03,
               "argus_eg+pre05|argus_bh": 0.2}
        scores = {"null": _scores(fdr, {}), "planted": _scores(fdr, {})}
        planted = self._planted({p: [1] for p in self.PIPES})
        d = decide(scores, planted)
        assert d["production_controls_fdr"] is False
        assert d["verdict"] == "rival_wins"

    def test_identical_power_and_fdr_is_a_tie(self) -> None:
        fdr = dict.fromkeys(self.PIPES, 0.02)
        scores = {"null": _scores(fdr, {}), "planted": _scores(fdr, {})}
        planted = self._planted({p: [1, 2] for p in self.PIPES})
        assert decide(scores, planted)["verdict"] == "tie"


def _four_series(seed: int, n: int = 400) -> dict[str, list[float]]:
    """Two series sharing one random-walk trend with a fast-reverting spread (phi 0.5), and two
    unrelated walks: the one true pair is unambiguous at this length."""
    rng = random.Random(seed)
    trend, a, b, z0, z1 = [0.0], [0.0], [0.0], [100.0], [120.0]
    for _ in range(n - 1):
        trend.append(trend[-1] + rng.gauss(0.0, 1.0))
        a.append(0.5 * a[-1] + rng.gauss(0.0, 1.0))
        b.append(0.5 * b[-1] + rng.gauss(0.0, 1.0))
        z0.append(z0[-1] + rng.gauss(0.0, 1.0))
        z1.append(z1[-1] + rng.gauss(0.0, 1.0))
    return {"A0": [100 + t + e for t, e in zip(trend, a, strict=True)],
            "A1": [80 + 1.5 * t + e for t, e in zip(trend, b, strict=True)],
            "Z0": z0, "Z1": z1}


@pytest.fixture(scope="module")
def small_result() -> dict[str, Any]:
    return evaluate_series(
        _four_series(5), train_tests=("argus_eg", "sm_coint", "arch_eg_bic", "arch_po_pz"),
        full_tests=("arch_eg_bic@full",), corrections=("argus_bh", "sm_bh", "scipy_bh"),
    )


class TestRivalsRunOnRealSeries:
    """Small universes, real library calls: the identities the comparison relies on."""

    @pytest.fixture
    def result(self, small_result: dict[str, Any]) -> dict[str, Any]:
        return small_result

    def test_argus_reproduces_statsmodels_coint(self, result: dict[str, Any]) -> None:
        assert result["checks"]["argus_vs_statsmodels_coint_max_abs_p_diff"] < 1e-8

    def test_production_pipeline_is_exactly_scan_survivors(self, result: dict[str, Any]) -> None:
        assert result["checks"]["scan_survivors_fdr_equals_production"] is True
        assert result["checks"]["bh_implementations_agree"] is True

    def test_misaligned_input_is_refused(self) -> None:
        with pytest.raises(GeneralCointComparisonError):
            evaluate_series({"A": [1.0] * 300, "B": [1.0] * 299})

    def test_every_pipeline_is_reported_and_no_rival_failed(self, result: dict[str, Any]) -> None:
        assert len(result["pairs"]) == 6
        assert "arch_po_pz+pre05|sm_bh" in result["discoveries"]
        assert "arch_eg_bic_both|argus_bh" in result["discoveries"]
        assert "argus_eg_both|argus_bh+oos" in result["discoveries"]
        assert CONFIRMED in result["discoveries"]
        assert not any(result["rival_failures"].values())

    def test_the_planted_pair_is_the_one_every_test_ranks_first(
        self, result: dict[str, Any]
    ) -> None:
        for test in ("argus_eg", "sm_coint", "arch_eg_bic", "arch_po_pz"):
            ps = result["pvalues"][test]
            assert result["pairs"][ps.index(min(ps))] == "A0/A1"

    def test_phillips_ouliaris_pz_does_not_depend_on_which_side_is_left(self) -> None:
        u = _four_series(9)
        a, b = u["A0"], u["A1"]
        forward = rival_pvalues(a, b, ["arch_po_pz"])["arch_po_pz"]
        backward = rival_pvalues(b, a, ["arch_po_pz"])["arch_po_pz"]
        assert forward == pytest.approx(backward, rel=1e-9)


class TestRealSnapshot:
    def test_snapshot_is_twelve_aligned_symbols(self) -> None:
        closes, meta = load_snapshot()
        assert len(closes) == 12
        assert {len(v) for v in closes.values()} == {meta["bars"]}
        assert meta["bars"] >= 1400
        assert not any(meta["dropped_unaligned_bars"].values())


class TestArtefact:
    @pytest.fixture(scope="class")
    def blob(self) -> dict[str, Any]:
        blob: dict[str, Any] = json.loads(Path(ARTEFACT_PATH).read_text(encoding="utf-8"))
        return blob

    def test_artefact_is_strict_json(self) -> None:
        assert is_strict(Path(ARTEFACT_PATH))

    def test_identity_checks_held_in_every_universe(self, blob: dict[str, Any]) -> None:
        checks = blob["identity_checks"]
        assert checks["scan_survivors_fdr_equals_production_in_every_universe"] is True
        assert checks["argus_bh_equals_statsmodels_and_scipy_bh_in_every_universe"] is True
        assert checks["argus_vs_statsmodels_coint_max_abs_p_diff"] < 1e-8

    def test_no_rival_raised_anywhere(self, blob: dict[str, Any]) -> None:
        assert not any(blob["rival_failures"].values())

    def test_verdict_is_one_of_the_four_and_names_its_evidence(self, blob: dict[str, Any]) -> None:
        d = blob["decision"]
        assert d["verdict"] in ("argus_wins", "tie", "rival_wins")
        assert d["best_general_pipeline"] is not None

    def test_real_universe_was_run_on_the_committed_snapshot(self, blob: dict[str, Any]) -> None:
        real = blob["real_rtoken_universe"]
        assert real["pairs_tested"] == 66
        _, meta = load_snapshot(SNAPSHOT_PATH)
        assert real["meta"]["sha256"] == meta["sha256"]

    def test_headline_rows_are_the_scored_rows(self, blob: dict[str, Any]) -> None:
        for scenario, rows in blob["headline"].items():
            everything = {r["pipeline"]: r for r in blob["all_scores"][scenario]}
            for row in rows:
                assert everything[row["pipeline"]] == row
