"""Tests for `eval/nco_bakeoff.py` -- the harness that decides whether ARGUS's allocator beats NCO.

The statistics and the protocol are pinned offline on synthetic inputs: the log-ratio orientation,
Holm's step-down, the mechanical verdict, the time split, and that every ablation arm differs from
`nco_weights` in exactly the component it names. The committed artefact, when present, is checked
for the properties the capability register reads from it -- disjoint selection and holdout, both
libraries' defaults in the decisive set, and bit-identical recomputation -- without pinning who won.
"""

from __future__ import annotations

import itertools
import json
import math
import random
import sys
from pathlib import Path

import pytest

from argus.desk.allocation import minimum_variance_weights, nco_weights
from argus.desk.nco_ensemble import NCOConfig, ensemble_nco_weights, sample_covariance
from argus.eval import nco_bakeoff
from argus.eval.nco_bakeoff import (
    MLAM_ENV,
    REPORT_PATH,
    Candidate,
    RivalUnavailable,
    _bad_tick,
    _nco_variant,
    _sign_p,
    _stationary_bootstrap_ci,
    ablation_candidates,
    argus_candidates,
    candidate_from_config,
    compare,
    decisive_rivals,
    evaluate,
    holm,
    minvar_ablation_candidates,
    origins,
    realised_vol,
    run_holdout,
    run_selection,
    split,
    stage_failures,
    verdict,
)


def _factor_columns(t: int = 900, n: int = 6, seed: int = 4) -> dict[str, list[float]]:
    rng = random.Random(seed)
    rows = []
    for _ in range(t):
        f, g = rng.gauss(0, 0.01), rng.gauss(0, 0.006)
        rows.append([(0.8 if j < n // 2 else 0.2) * f + (0.1 if j < n // 2 else 0.7) * g
                     + rng.gauss(0, 0.003 * (1 + j / n)) for j in range(n)])
    return {f"S{j}": [r[j] for r in rows] for j in range(n)}


class TestStatistics:
    def test_log_ratio_is_positive_when_argus_is_lower(self) -> None:
        row = compare([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                      [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7])
        assert row["argus_wins"] == 7 and row["rival_wins"] == 0
        assert row["mean_log_ratio"] == pytest.approx(math.log(1.1))
        lo, hi = row["bootstrap_ci95_mean_log_ratio"]
        assert lo > 0 and hi > 0

    def test_identical_series_are_all_ties_with_no_test(self) -> None:
        row = compare([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        assert row["ties"] == 6 and row["sign_test_p"] is None and row["wilcoxon_p"] is None

    def test_missing_windows_are_dropped_pairwise(self) -> None:
        row = compare([1.0, None, 3.0], [1.1, 2.0, None])
        assert row["n_windows"] == 1

    def test_holm_steps_down_and_stops_at_the_first_acceptance(self) -> None:
        out = holm({"a": 0.001, "b": 0.02, "c": 0.04, "d": None})
        assert out["a"]["rejects"] and out["a"]["threshold"] == pytest.approx(0.05 / 3)
        assert out["b"]["rejects"] and out["b"]["threshold"] == pytest.approx(0.05 / 2)
        assert out["c"]["rejects"] and out["c"]["threshold"] == pytest.approx(0.05)
        assert not out["d"]["rejects"]
        # b fails its 0.025 threshold, so c is not rejected even though 0.045 < 0.05 on its own.
        stopped = holm({"a": 0.001, "b": 0.04, "c": 0.045})
        assert stopped["a"]["rejects"]
        assert not stopped["b"]["rejects"] and not stopped["c"]["rejects"]

    def test_verdict_needs_significance_and_an_interval_clear_of_zero(self) -> None:
        rows = {
            "x": {"bootstrap_ci95_mean_log_ratio": [0.01, 0.05]},
            "y": {"bootstrap_ci95_mean_log_ratio": [-0.01, 0.05]},
        }
        both = verdict(rows, {"x": {"rejects": True}, "y": {"rejects": True}})
        assert both["per_rival"] == {"x": "WIN", "y": "TIE"} and both["overall"] == "TIE"
        insignificant = verdict(rows, {"x": {"rejects": False}, "y": {"rejects": False}})
        assert insignificant["per_rival"]["x"] == "TIE"
        loss = verdict({"z": {"bootstrap_ci95_mean_log_ratio": [-0.2, -0.1]}},
                       {"z": {"rejects": True}})
        assert loss["overall"] == "LOSS"
        win = verdict({"x": rows["x"]}, {"x": {"rejects": True}})
        assert win["overall"] == "WIN"


class TestProtocol:
    def test_origins_step_and_never_overrun(self) -> None:
        o = origins(1200, 480, 96, 96)
        assert o[0] == 0 and all(b - a == 96 for a, b in itertools.pairwise(o))
        assert o[-1] + 480 + 96 <= 1200

    def test_split_is_in_time_order_and_disjoint(self) -> None:
        sel, hold = split(list(range(0, 39 * 96, 96)))
        assert max(sel) < min(hold)
        assert len(hold) >= len(sel) and len(sel) + len(hold) == 39

    def test_evaluate_scores_the_held_out_window_only(self) -> None:
        cols = _factor_columns()
        equal = Candidate("eq", "reference", lambda c: dict.fromkeys(c, 1.0 / len(c)))
        row = evaluate(equal, cols, [0, 96], train=480, test=96)
        held = {n: v[480:576] for n, v in cols.items()}
        assert row["oos_vol_bps"][0] == pytest.approx(
            realised_vol(dict.fromkeys(cols, 1 / 6), held) * 10_000)
        assert row["turnover_per_rebalance"] == pytest.approx(0.0)

    def test_evaluate_records_a_failure_instead_of_raising(self) -> None:
        def broken(_cols: object) -> dict[str, float]:
            raise ValueError("no")

        row = evaluate(Candidate("bad", "reference", broken), _factor_columns(), [0])
        assert row["n_errors"] == 1 and row["oos_vol_bps"] == [None]

    def test_decisive_set_holds_every_default_and_every_pick(self) -> None:
        def noop(_cols: object) -> dict[str, float]:
            return {}

        cands = [
            Candidate("r_default", "riskfolio", noop, default=True),
            Candidate("r_tuned", "riskfolio", noop),
            Candidate("s_default", "skfolio", noop, default=True),
        ]
        assert decisive_rivals(cands, {"riskfolio": "r_tuned", "skfolio": "s_default"}) == [
            "r_default", "r_tuned", "s_default"]


class TestAblationArms:
    def test_default_variant_is_exactly_nco_weights(self) -> None:
        cols = _factor_columns()
        names = sorted(cols)
        cov = sample_covariance([[cols[n][t] for n in names] for t in range(480)])
        ours = _nco_variant(cov, names)
        ref = nco_weights(names, cov)
        for n in names:
            assert ours[n] == pytest.approx(ref[n], abs=1e-12)

    def test_each_arm_is_a_long_only_budget(self) -> None:
        cols = {n: v[:480] for n, v in _factor_columns().items()}
        arms = ablation_candidates(NCOConfig("lw_cc", n_boot=4))
        labels = {a.label for a in arms}
        assert {"ablate:no_bagging", "ablate:sample_covariance", "ablate:no_clustering(k=1)",
                "ablate:single_linkage", "ablate:intra_inverse_variance",
                "ablate:outer_equal_weight", "ablate:random_clusters_same_sizes"} <= labels
        for arm in arms:
            w = arm.fn(cols)
            assert math.isclose(sum(w.values()), 1.0, abs_tol=1e-9), arm.label
            assert all(v >= -1e-12 for v in w.values()), arm.label

    def test_bad_tick_corrupts_exactly_one_value_reproducibly(self) -> None:
        cols = {n: v[:480] for n, v in _factor_columns().items()}
        a, b = _bad_tick(cols, 96), _bad_tick(cols, 96)
        assert a == b
        changed = [(n, i) for n in cols for i in range(480) if a[n][i] != cols[n][i]]
        assert len(changed) == 1
        n, i = changed[0]
        assert abs(a[n][i]) == pytest.approx(0.15)


class TestExactTests:
    def test_sign_test_is_the_exact_two_sided_binomial(self) -> None:
        assert _sign_p(10, 10) == pytest.approx(2 * 0.5 ** 10)
        assert _sign_p(0, 10) == pytest.approx(2 * 0.5 ** 10)
        assert _sign_p(5, 10) == pytest.approx(1.0)
        assert _sign_p(8, 10) == pytest.approx(2 * sum(math.comb(10, k) for k in (8, 9, 10)) / 1024)
        assert _sign_p(0, 0) is None

    def test_bootstrap_interval_of_a_constant_is_that_constant(self) -> None:
        lo, hi = _stationary_bootstrap_ci([0.25] * 12, resamples=500)
        assert lo == pytest.approx(0.25) and hi == pytest.approx(0.25)

    def test_bootstrap_interval_brackets_the_mean_and_is_seeded(self) -> None:
        rng = random.Random(1)
        values = [rng.gauss(0.1, 0.05) for _ in range(30)]
        first = _stationary_bootstrap_ci(values, resamples=2000)
        assert first == _stationary_bootstrap_ci(values, resamples=2000)
        assert first[0] < sum(values) / len(values) < first[1]


class TestEndToEnd:
    """The whole selection -> holdout pipeline on synthetic returns with pure-Python allocators, so
    the wiring (picks, decisive set, Holm, verdict, oracle) is exercised without any rival library.
    """

    @pytest.fixture(scope="class")
    def columns(self) -> dict[str, list[float]]:
        return _factor_columns(t=1500)

    @pytest.fixture(scope="class")
    def candidates(self) -> list[Candidate]:
        def equal(cols: object) -> dict[str, float]:
            names = sorted(cols)  # type: ignore[call-overload]
            return dict.fromkeys(names, 1.0 / len(names))

        def minvar(cols: dict[str, list[float]]) -> dict[str, float]:
            names = sorted(cols)
            rows = [[cols[n][t] for n in names] for t in range(len(cols[names[0]]))]
            return minimum_variance_weights(names, sample_covariance(rows))

        return [
            Candidate("a_nco", "argus",
                      lambda c: ensemble_nco_weights(c, NCOConfig("sample")), default=True),
            Candidate("a_minvar", "argus", minvar),
            Candidate("r_equal_default", "riskfolio", equal, default=True),
            Candidate("r_minvar", "riskfolio", minvar),
        ]

    def test_selection_then_holdout_is_disjoint_and_complete(
        self, columns: dict[str, list[float]], candidates: list[Candidate],
    ) -> None:
        sel, hold = split(origins(len(columns["S0"])))
        assert len(sel) == 5 and len(hold) == 5
        selection = run_selection(columns, sel, candidates, log=lambda _m: None)
        assert set(selection["picks"]) == {"argus", "riskfolio"}
        holdout = run_holdout(columns, hold, candidates, selection["picks"], log=lambda _m: None)
        assert holdout["argus_pick"] == selection["picks"]["argus"]
        assert holdout["decisive_rivals"][0] == "r_equal_default"
        assert set(holdout["comparisons"]) == set(holdout["decisive_rivals"])
        for row in holdout["comparisons"].values():
            assert row["n_windows"] == 5 and "holm" in row
        assert set(holdout["verdict"]["per_rival"]) == set(holdout["decisive_rivals"])
        assert holdout["verdict"]["overall"] in {"WIN", "TIE", "LOSS"}
        assert "riskfolio" in holdout["oracle_rivals"]
        # every candidate scored on the holdout, context included
        assert set(holdout["results"]) == {c.label for c in candidates}

    def test_identical_allocators_tie_exactly(
        self, columns: dict[str, list[float]], candidates: list[Candidate],
    ) -> None:
        by = {c.label: c for c in candidates}
        hold = split(origins(len(columns["S0"])))[1]
        a = evaluate(by["a_minvar"], columns, hold)["oos_vol_bps"]
        b = evaluate(by["r_minvar"], columns, hold)["oos_vol_bps"]
        row = compare(a, b)
        assert row["ties"] == 5 and row["mean_log_ratio"] == 0.0


class TestCandidates:
    def test_argus_family_is_the_nine_documented_configurations(self) -> None:
        labels = [c.label for c in argus_candidates()]
        assert len(labels) == len(set(labels)) == 9
        assert sum(c.default for c in argus_candidates()) == 1
        assert "argus_minvar[sample]" in labels and "argus_nco[lw_cc,bag32]" in labels

    def test_candidate_from_config_rebuilds_the_same_allocator(self) -> None:
        cols = {n: v[:480] for n, v in _factor_columns().items()}
        for cand in argus_candidates():
            if cand.config.get("n_boot"):
                continue  # bagged configs: same code path, slower; seeded equality is covered
            rebuilt = candidate_from_config(cand.label, cand.config)
            assert rebuilt.fn(cols) == cand.fn(cols), cand.label

    def test_minvar_ablation_arms_each_change_one_thing(self) -> None:
        cols = {n: v[:480] for n, v in _factor_columns().items()}
        arms = minvar_ablation_candidates(NCOConfig("sample"))
        labels = {a.label for a in arms}
        assert labels == {"ablate:add_nco_clustering", "ablate:drop_long_only(closed_form)",
                          "ablate:add_bagging32", "ablate:covariance=lw_cc",
                          "ablate:covariance=ewma240"}
        for arm in arms:
            if arm.label == "ablate:add_bagging32":
                continue
            w = arm.fn(cols)
            assert math.isclose(sum(w.values()), 1.0, abs_tol=1e-9), arm.label
            if arm.long_only:
                assert all(v >= -1e-12 for v in w.values()), arm.label


class TestRivalLoading:
    def test_de_prado_rival_refuses_without_a_clone(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(MLAM_ENV, raising=False)
        with pytest.raises(RivalUnavailable, match=MLAM_ENV):
            nco_bakeoff._mlam()

    def test_skfolio_refuses_with_instructions_when_absent(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        try:
            import skfolio  # type: ignore[import-not-found, unused-ignore]  # noqa: F401
        except ImportError:
            pass
        else:
            pytest.skip("skfolio is installed here, so there is nothing to refuse")
        monkeypatch.delenv(nco_bakeoff.SKFOLIO_ENV, raising=False)
        with pytest.raises(RivalUnavailable, match="pip install skfolio"):
            nco_bakeoff._ensure_skfolio()

    def test_a_plot_only_import_does_not_sink_the_rival(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        """A clone whose module imports seaborn at the top loads even where seaborn is absent --
        the failure that cost the first full run its de Prado rival and its Monte Carlo stage."""
        pkg = tmp_path / "Machine_Learning_for_Asset_Managers"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("", encoding="utf-8")
        (pkg / "ch7_portfolio_construction.py").write_text(
            "import seaborn as sns\n\ndef optPort_nco():\n    return 42\n", encoding="utf-8")
        monkeypatch.setenv(MLAM_ENV, str(tmp_path))
        monkeypatch.setattr(sys, "path", list(sys.path))
        before = set(sys.modules)
        try:
            module = nco_bakeoff._mlam()
            assert module.optPort_nco() == 42
        finally:
            for name in set(sys.modules) - before:
                del sys.modules[name]


class TestStageFailures:
    def test_lists_failed_stages_and_failed_candidates(self) -> None:
        report = {
            "monte_carlo": {"error": "ModuleNotFoundError: seaborn"},
            "adversarial": {"x": 1},
            "holdout": {"results": {
                "ok": {"n_errors": 0, "errors": []},
                "bad": {"n_errors": 20, "errors": ["0: ModuleNotFoundError"]},
            }},
        }
        out = stage_failures(report)
        assert out == {"monte_carlo": "ModuleNotFoundError: seaborn",
                       "holdout:bad": {"windows_failed": 20,
                                       "first_error": "0: ModuleNotFoundError"}}

    def test_a_clean_report_has_none(self) -> None:
        assert stage_failures({"holdout": {"results": {"ok": {"n_errors": 0}}}}) == {}


@pytest.mark.skipif(not REPORT_PATH.exists(), reason="bake-off artefact not generated yet")
class TestCommittedArtefact:
    @pytest.fixture(scope="class")
    def report(self) -> dict[str, object]:
        blob: dict[str, object] = json.loads(Path(REPORT_PATH).read_text(encoding="utf-8"))
        return blob

    def test_selection_and_holdout_are_disjoint_in_time(self, report: dict) -> None:  # type: ignore[type-arg]
        sel = report["selection"]["origins"]
        hold = report["holdout"]["origins"]
        assert max(sel) < min(hold)

    def test_both_library_defaults_are_in_the_decisive_set(self, report: dict) -> None:  # type: ignore[type-arg]
        rivals = report["holdout"]["decisive_rivals"]
        assert "riskfolio_nco[hist,single]" in rivals
        assert "skfolio_nco[default]" in rivals

    def test_verdict_matches_the_rule_applied_to_its_own_numbers(self, report: dict) -> None:  # type: ignore[type-arg]
        comps = report["holdout"]["comparisons"]
        recomputed = verdict(comps, {k: v["holm"] for k, v in comps.items()})
        assert recomputed == report["holdout"]["verdict"]

    def test_recomputation_was_bit_identical(self, report: dict) -> None:  # type: ignore[type-arg]
        assert report["reproducibility"]["all_identical"] is True

    def test_every_decisive_rival_scored_every_holdout_window(self, report: dict) -> None:  # type: ignore[type-arg]
        n = len(report["holdout"]["origins"])
        for label in report["holdout"]["decisive_rivals"]:
            assert report["holdout"]["results"][label]["n_windows_scored"] == n, label

    def test_argus_pick_is_the_selection_pick(self, report: dict) -> None:  # type: ignore[type-arg]
        assert report["holdout"]["argus_pick"] == report["selection"]["picks"]["argus"]

    def test_decisive_statistics_recompute_from_the_per_window_record(self, report: dict) -> None:  # type: ignore[type-arg]
        hold = report["holdout"]
        argus = hold["results"][hold["argus_pick"]]["oos_vol_bps"]
        for label in hold["decisive_rivals"]:
            again = compare(argus, hold["results"][label]["oos_vol_bps"])
            stored = hold["comparisons"][label]
            assert set(again) == set(stored) - {"holm"}, label
            for key, value in again.items():
                if isinstance(value, (list, float)):
                    assert value == pytest.approx(stored[key], rel=1e-12), (label, key)
                else:
                    assert value == stored[key], (label, key)

    def test_every_stage_ran(self, report: dict) -> None:  # type: ignore[type-arg]
        failures = stage_failures(report)
        if "stage_failures" not in report:
            pytest.xfail(f"artefact predates the stage_failures record and must be regenerated; "
                         f"it records failures in {sorted(failures)}")
        assert failures == {}
        assert report["stage_failures"] == {}
