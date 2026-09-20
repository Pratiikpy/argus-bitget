"""Tests for the adversarial audit of `eval/regime_comparison.py`.

Offline and seeded, unlike `tests/test_regime_comparison.py`, and that is the point rather than a
convenience: the module under audit fetches live and says openly that its counts move between
passes, so a reviewer cannot re-run it and get the published numbers. Everything here is generated
from a fixed seed, so every assertion below is a claim a reviewer can reproduce exactly.

These pin findings, not shapes. Where an assertion could move, its docstring says what a failure
would mean — a failure here is a result to read, not a flake to retry. The two that would matter
most are the ones that cut against ARGUS: `test_the_argus_lead_over_ruptures_is_not_significant`
and `test_the_ordering_is_not_stable_across_every_margin`. If either starts passing in the other
direction, the verdict text in the module is wrong and must be rewritten before it ships.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from argus.eval.regime_groundtruth_audit import (
    ARMS,
    BASE_SEED,
    MARGIN_SWEEP,
    SCENARIOS,
    SERIES_LENGTH,
    TRUE_BREAKS,
    VERDICT_ON_THE_COMPARISON,
    check_artefact_consistency,
    family_placebo,
    incumbent_boundaries,
    make_series,
    render,
    run_determinism_check,
    run_ground_truth_benchmark,
    score,
    verify_baselines_are_real,
)


@pytest.fixture(scope="module")
def benchmark() -> dict[str, Any]:
    return run_ground_truth_benchmark()


@pytest.fixture(scope="module")
def baselines() -> dict[str, Any]:
    return verify_baselines_are_real()


@pytest.fixture(scope="module")
def integrity() -> dict[str, Any]:
    return check_artefact_consistency()


class TestTheBaselinesAreTheRealLibraries:
    """The single question this whole exercise exists to answer."""

    def test_stumpy_and_ruptures_are_installed_distributions_not_local_shims(
        self, baselines: dict[str, Any]
    ) -> None:
        """Version read from distribution metadata, path read off the imported module.

        `stumpy.__version__` is the literal string "Please install this project with setup.py" on
        this build, so a version taken from the attribute would have been meaningless — that is
        why `metadata.version` is used and why this asserts a real dotted version.
        """
        assert baselines["stumpy_version"].split(".")[0].isdigit()
        assert baselines["ruptures_version"].lstrip("v").split(".")[0].isdigit()
        assert "site-packages" in baselines["stumpy_path"]
        assert "site-packages" in baselines["ruptures_path"]

    def test_both_libraries_actually_executed_not_merely_imported(
        self, baselines: dict[str, Any]
    ) -> None:
        """An import proves installation; only a call proves the comparison went through it."""
        assert baselines["stumpy_executed"] is True
        assert baselines["ruptures_executed"] is True
        assert baselines["windows_compared"] > 900

    def test_argus_reproduces_stumpys_profile_exactly_on_input_it_never_saw(
        self, baselines: dict[str, Any]
    ) -> None:
        """`regime_comparison.py`'s Finding 3 claimed exact parity on 16,992 live windows. This
        re-checks it on a seeded series generated here, so the claim cannot be an artefact of the
        particular market data that module happened to fetch. A failure means the parity finding
        is data-dependent and Finding 3 is overstated."""
        assert baselines["argus_vs_stumpy_index_agreement"] == 1.0
        assert baselines["parity_reproduces_on_unseen_input"] is True


class TestGroundTruthExists:
    """`regime_comparison.py` built its referee on 'nobody knows where the real boundary is'."""

    def test_the_generators_put_the_breaks_where_every_arm_can_answer(self) -> None:
        """Both true breaks must clear the incumbent's 240-bar warmup and FLUSS's pinned head and
        tail, or a miss would be a structural inability rather than a failure to detect."""
        values, truth = make_series("variance_switch", BASE_SEED)
        assert len(values) == SERIES_LENGTH
        assert truth == list(TRUE_BREAKS)
        assert min(truth) > 241
        assert max(truth) < SERIES_LENGTH - 120
        assert all(v > 0 for v in values)

    def test_each_scenario_produces_a_distinct_series(self) -> None:
        paths = {s: make_series(s, BASE_SEED)[0] for s in SCENARIOS}
        assert len({tuple(v) for v in paths.values()}) == len(SCENARIOS)

    def test_an_unknown_scenario_is_refused_rather_than_silently_defaulted(self) -> None:
        with pytest.raises(ValueError, match="unknown scenario"):
            make_series("no_such_process", BASE_SEED)

    def test_scoring_uses_the_baselines_own_metrics(self, benchmark: dict[str, Any]) -> None:
        assert "ruptures.metrics" in benchmark["metric_source"]

    def test_an_arm_that_finds_nothing_scores_zero_rather_than_raising(self) -> None:
        """A live crash this test found, not a hypothetical.

        `ruptures.metrics.hausdorff` reduces over interior breakpoints only and raises
        `ValueError: zero-size array to reduction operation maximum` when one side is empty. An
        empty answer is exactly what `desk/regime.py`'s `find_boundaries` returns on a flat
        corrected arc curve — the one case `regime_comparison.py`'s Finding 5 credits ARGUS for —
        so scoring that case unguarded would have crashed on ARGUS's only win. It must score, and
        it must score badly rather than scoring 0.0 Hausdorff as if it were perfect.
        """
        empty = score([], TRUE_BREAKS, SERIES_LENGTH)
        assert empty["f1"] == 0.0
        assert empty["n_found"] == 0.0
        assert empty["mean_distance_to_truth"] == float(SERIES_LENGTH)
        assert empty["hausdorff"] == float(SERIES_LENGTH)

    def test_a_perfect_answer_scores_one(self) -> None:
        perfect = score(TRUE_BREAKS, TRUE_BREAKS, SERIES_LENGTH)
        assert perfect["f1"] == 1.0
        assert perfect["hausdorff"] == 0.0


class TestWhatGroundTruthSaysAboutTheVerdict:
    def test_every_arm_is_poor_in_absolute_terms(self, benchmark: dict[str, Any]) -> None:
        """The load-bearing result. `regime_comparison.py` records the capability as LOST because
        a specialist beat it; the stronger reading is that nothing here detects regimes well. If
        any arm's pooled f1 ever clears 0.6 this assertion should fail and that arm deserves a
        second look."""
        assert all(benchmark["pooled"][arm]["f1"] < 0.6 for arm in ARMS)

    def test_the_incumbent_buys_its_recall_with_precision(
        self, benchmark: dict[str, Any]
    ) -> None:
        """The two-line rule fires tens of times a series, so it covers nearly every true break and
        means nothing by it. This is why `regime_comparison.py`'s novelty test — which scores
        agreement with this rule — could never have been flattering to ARGUS."""
        incumbent = benchmark["pooled"]["incumbent"]
        assert incumbent["recall"] > 0.5
        assert incumbent["precision"] < 0.2
        assert incumbent["n_found"] > 10

    def test_argus_and_stumpy_score_near_identically_as_parity_predicts(
        self, benchmark: dict[str, Any]
    ) -> None:
        """Finding 3 says the two share the profile exactly and differ only in the arc curve, so
        their accuracy must be close. A large gap here would mean the ablation understated how
        load-bearing the parabola IAC is."""
        gap = abs(benchmark["pooled"]["argus"]["f1"] - benchmark["pooled"]["stumpy"]["f1"])
        assert gap < 0.1

    def test_ruptures_fails_the_pure_drift_switch(self, benchmark: dict[str, Any]) -> None:
        """The finding that breaks `regime_comparison.py`'s Finding 7. An rbf KernelCPD budgeted to
        two breaks scores zero f1 on a series whose only change is drift, while its Hausdorff is
        the best of the three — consistently placed, consistently wrong. A method can therefore
        have zero family spread and the worst accuracy, so 'spread is pure error measurement' is
        false as that finding states it."""
        drift = benchmark["per_scenario"]["drift_switch"]
        assert drift["ruptures"]["f1"] == 0.0
        assert drift["argus"]["f1"] > drift["ruptures"]["f1"]

    def test_the_argus_lead_over_ruptures_is_not_significant(
        self, benchmark: dict[str, Any]
    ) -> None:
        """Deliberately asserts the UNFLATTERING reading. ARGUS wins more replicates than it loses,
        but a paired sign test does not clear 5%. If this ever starts failing because p dropped
        below 0.05, the verdict text is too cautious and must be rewritten — which is the only
        acceptable reason for it to fail."""
        test = benchmark["paired_sign_tests"]["argus_vs_ruptures"]["f1"]
        assert test["wins"] > test["losses"]
        assert test["p_two_sided"] is not None
        assert test["p_two_sided"] >= 0.05

    def test_argus_does_not_significantly_beat_the_incumbent_either(
        self, benchmark: dict[str, Any]
    ) -> None:
        """The demotion survives ground truth. FLUSS wins more replicates than the two-line rule
        but not significantly, which is the same answer `regime_comparison.py` got on live data by
        a completely different route (novelty below its own null, p=0.954)."""
        test = benchmark["paired_sign_tests"]["argus_vs_incumbent"]["f1"]
        assert test["p_two_sided"] is not None
        assert test["p_two_sided"] >= 0.05

    def test_the_ordering_is_not_stable_across_every_margin(
        self, benchmark: dict[str, Any]
    ) -> None:
        """Also deliberately unflattering. ARGUS leads at 12/24/48/96 bars and loses at 192, so the
        headline is tolerance-dependent and the module says so rather than publishing the margin
        that suits. A failure here means the flip went away and the claim can be strengthened."""
        sweep = benchmark["margin_sweep"]
        assert sweep["ordering_stable"] is False
        assert sweep["argus_at_least_ruptures_at_every_margin"] is False
        assert sweep["smallest_margin_where_ruptures_leads_argus"] == max(MARGIN_SWEEP)

    def test_argus_leads_at_every_tolerance_tighter_than_the_flip(
        self, benchmark: dict[str, Any]
    ) -> None:
        table = benchmark["margin_sweep"]["pooled_f1_by_margin"]
        tight = [m for m in MARGIN_SWEEP if m < max(MARGIN_SWEEP)]
        assert all(table[str(m)]["argus"] > table[str(m)]["ruptures"] for m in tight)

    def test_every_budgeted_arm_was_held_to_the_same_boundary_count(
        self, benchmark: dict[str, Any]
    ) -> None:
        """A comparison where one arm may return more boundaries is not a comparison. Only the
        incumbent is budget-free, because that is how it ships."""
        for arm in ("argus", "stumpy", "ruptures"):
            assert benchmark["pooled"][arm]["n_found"] == pytest.approx(len(TRUE_BREAKS), abs=0.5)


class TestIncumbentArm:
    def test_the_real_exported_rule_is_driven_not_reimplemented(self) -> None:
        """`regime_comparison.py`'s Finding 1 was that `desk/regime.py` scored itself against a
        reimplementation of the incumbent that used a different statistic. This arm must not repeat
        that: it drives `rotation_regime_switch` over real `Bar` objects."""
        values, _ = make_series("variance_switch", BASE_SEED)
        flips = incumbent_boundaries(values)
        assert flips
        assert all(0 < f < len(values) for f in flips)
        assert flips == sorted(set(flips))

    def test_the_rule_is_silent_through_its_own_warmup(self) -> None:
        """`_vol`'s slow leg returns 0.0 until bar 241, so a flip before then would be a warmup
        artefact and would make the ground-truth score meaningless."""
        values, _ = make_series("both_switch", BASE_SEED + 3)
        assert min(incumbent_boundaries(values)) > 240


class TestFamilyPlacebo:
    """The control `regime_comparison.py`'s Finding 7 needed and did not run."""

    def test_the_family_is_an_outlier_for_every_method(self) -> None:
        """If three unrelated symbols spread as little as the QQQ family does, the referee measures
        the shared trading calendar rather than the shared underlying. They do not: the family is
        inside the tightest decile for all three methods."""
        report = json.loads(_comparison_artefact_text())
        placebo = family_placebo(report["base_case"]["per_symbol"])
        assert placebo["available"] is True
        assert placebo["family_is_outlier_for_every_method"] is True
        for method in ("argus", "stumpy", "ruptures"):
            assert placebo[method]["non_family_triples"] == 219
            assert placebo[method]["family_percentile"] <= 0.10

    def test_ruptures_is_the_most_coherent_and_that_is_still_true(self) -> None:
        """Finding 7's own result reproduces — ruptures really is tighter on the family than any
        of the 219 non-family triples. The audit does not dispute the measurement, only what it
        licenses: coherence without accuracy is consistency, not correctness."""
        placebo = family_placebo(
            json.loads(_comparison_artefact_text())["base_case"]["per_symbol"]
        )
        assert placebo["ruptures"]["family_spread_total"] < placebo["argus"]["family_spread_total"]
        assert placebo["ruptures"]["family_percentile"] == 0.0

    def test_a_universe_without_the_family_reports_unavailable(self) -> None:
        assert family_placebo([{"symbol": "AAPLUSDT", "argus_cuts": [1]}])["available"] is False


class TestArtefactIntegrity:
    def test_the_measured_fields_are_internally_consistent(
        self, integrity: dict[str, Any]
    ) -> None:
        """The speedup and the novelty rate are both derivable from other fields in the same file,
        so they are recomputed rather than trusted. Both check out — the measurements are sound."""
        assert integrity["speedup_self_consistent"] is True
        assert integrity["novelty_rate_self_consistent"] is True

    def test_the_narrated_scope_statement_has_drifted_from_the_measurements(
        self, integrity: dict[str, Any]
    ) -> None:
        """A real defect, found and reported rather than fixed here: the artefact's prose says 337x
        and a calendar bucket of 27 while the fields beside it read 310.36 and 38, so the file
        predates the current source. A failure here means the artefact was regenerated, which is
        the fix — update this test to assert the match at that point."""
        assert integrity["scope_statement_matches_current_source"] is False

    def test_the_artefact_is_not_strict_json(self, integrity: dict[str, Any]) -> None:
        """Bare `NaN` — an undefined arc-curve correlation on a half series. Python accepts it;
        `JSON.parse`, `encoding/json` and `serde_json` all reject it. Shared with several other
        `data/*.json`, so it is a house pattern and is reported, not patched from here."""
        assert integrity["parses_as_strict_json"] is False
        assert integrity["bare_nan_tokens"] > 0


class TestDeterminismAndOutput:
    def test_the_same_seed_gives_the_same_report(self) -> None:
        """The property the audited module cannot offer, because it fetches live. Both baselines
        have stochastic-looking internals, so this is a real question about them and not only
        about ARGUS's arithmetic."""
        assert run_determinism_check()["identical"] is True

    def test_render_names_the_finding_that_cuts_against_the_audit(
        self, benchmark: dict[str, Any], baselines: dict[str, Any], integrity: dict[str, Any]
    ) -> None:
        text = render(
            {
                "baselines_are_real": baselines,
                "ground_truth": benchmark,
                "family_placebo": family_placebo(
                    json.loads(_comparison_artefact_text())["base_case"]["per_symbol"]
                ),
                "artefact_integrity": integrity,
                "determinism": {"identical": True},
                "verdict_on_the_comparison": VERDICT_ON_THE_COMPARISON,
            }
        )
        assert "ordering stable" in text
        assert "sign test" in text
        assert "FAMILY PLACEBO" in text

    def test_the_verdict_states_what_was_not_verified(self) -> None:
        assert "NOT VERIFIED" in VERDICT_ON_THE_COMPARISON
        assert "p=0.057" in VERDICT_ON_THE_COMPARISON
        assert "ordering_stable is False" in VERDICT_ON_THE_COMPARISON


def _comparison_artefact_text() -> str:
    from argus.eval.regime_groundtruth_audit import COMPARISON_ARTEFACT

    return COMPARISON_ARTEFACT.read_text(encoding="utf-8")
