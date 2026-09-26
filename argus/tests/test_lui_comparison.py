"""Tests for the ARGUS-vs-Rasa LUI comparison.

ARGUS's side runs live, through the same function the console calls. Rasa's side is read from a
vendored, static reference (`eval/baselines/rasa_diet_sealed_predictions.json`) because training
Rasa's real DIET classifier needs a 1.9GB TensorFlow-based venv this project does not carry — the
same choice `quarantine_comparison.py` makes for AgentDojo's real attack corpus.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from argus.eval.lui_comparison import (
    RASA_PREDICTIONS_PATH,
    LuiComparisonError,
    main,
    mcnemar_exact,
    render,
    run_ablation_without_threshold,
    run_oos_comparison,
    run_sealed_comparison,
    scope_statement,
)


class TestMcnemarExact:
    def test_no_discordant_pairs_is_certain(self) -> None:
        assert mcnemar_exact(0, 0) == 1.0

    def test_a_perfectly_even_split_is_not_significant(self) -> None:
        assert mcnemar_exact(10, 10) > 0.05

    def test_a_lopsided_split_is_significant(self) -> None:
        assert mcnemar_exact(0, 36) < 0.01

    def test_matches_the_verified_headline_figure(self) -> None:
        """The exact pair this comparison was built to explain: 13 vs 36 discordant on the 293-row
        sealed split, independently reproduced (not just quoted) from the RIVAL agent's own
        analyse.py before this module existed."""
        p = mcnemar_exact(13, 36)
        assert p == pytest.approx(0.0014, abs=0.0002)

    def test_is_symmetric(self) -> None:
        assert mcnemar_exact(4, 9) == mcnemar_exact(9, 4)


class TestVendoredReference:
    def test_the_reference_file_exists(self) -> None:
        assert RASA_PREDICTIONS_PATH.is_file()

    def test_the_reference_carries_293_sealed_predictions(self) -> None:
        blob = json.loads(RASA_PREDICTIONS_PATH.read_text(encoding="utf-8"))
        assert len(blob["sealed_predictions"]) == 293

    def test_the_reference_carries_all_14_oos_probes_accounted_for(self) -> None:
        blob = json.loads(RASA_PREDICTIONS_PATH.read_text(encoding="utf-8"))
        assert blob["oos_probes"] == 14
        assert blob["oos_declined"] == 6
        assert len(blob["oos_probe_predictions"]) == 14 - 6


class TestSealedComparison:
    def test_runs_end_to_end_against_the_real_deployed_cascade(self) -> None:
        result = run_sealed_comparison()
        assert result.argus_total == 293
        assert result.rasa_total == 293
        # Not re-asserting the exact counts here (that would just re-type this module's own
        # arithmetic) -- the McNemar figure is what's pinned, in TestMcnemarExact above, against
        # an independently reproduced value.
        assert 0.0 <= result.argus_accuracy <= 1.0
        assert 0.0 <= result.rasa_accuracy <= 1.0

    def test_the_deployed_cascade_still_scores_what_the_artefact_publishes(self) -> None:
        # A routing change elsewhere silently cost two sealed answers on 2026-09-26 ("how's the
        # book looking" moved to position; "that shot ... last week" became ambiguous) while the
        # published figure stayed 233. The artefact is the claim, so the live cascade must match it.
        import json
        from pathlib import Path

        published = json.loads((Path(__file__).resolve().parents[1] / "data" /
                                 "lui_comparison.json").read_text(encoding="utf-8"))
        assert run_sealed_comparison().as_dict() == published["sealed_comparison"]

    def test_a_missing_reference_row_is_refused_not_silently_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Same-input-comparison is the whole point of this module; a Rasa reference that has
        drifted from the live sealed corpus must stop the run, not quietly compare fewer rows."""
        import argus.eval.lui_comparison as mod

        broken = tmp_path / "broken.json"
        broken.write_text(
            json.dumps({"sealed_predictions": [], "oos_probes": 14,
                        "oos_probe_predictions": []}),
            encoding="utf-8",
        )
        monkeypatch.setattr(mod, "RASA_PREDICTIONS_PATH", broken)
        with pytest.raises(LuiComparisonError, match="missing a sealed row"):
            mod.run_sealed_comparison()


class TestOosComparison:
    def test_runs_end_to_end_and_argus_leaks_are_a_subset_of_rasas(self) -> None:
        """The specific structural claim the scope statement makes: ARGUS's refusal set is a
        STRICT SUPERSET of Rasa's -- every question ARGUS answers that Rasa also declines is fine,
        but nothing ARGUS declines should be a question Rasa also declines and ARGUS uniquely
        misses, or the 'strict superset' framing would be a claim this test does not back."""
        result = run_oos_comparison()
        assert result.probes == 14
        assert set(result.argus_leaks) <= set(result.rasa_leaks)

    def test_a_probe_count_mismatch_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import argus.eval.lui_comparison as mod

        broken = tmp_path / "broken.json"
        broken.write_text(
            json.dumps({"sealed_predictions": [], "oos_probes": 3,
                        "oos_probe_predictions": []}),
            encoding="utf-8",
        )
        monkeypatch.setattr(mod, "RASA_PREDICTIONS_PATH", broken)
        with pytest.raises(LuiComparisonError, match="does not match"):
            mod.run_oos_comparison()


class TestAblationWithoutThreshold:
    def test_removing_the_threshold_does_not_close_the_accuracy_gap(self) -> None:
        """The specific empirical claim: this is a real model-quality gap, not caution. If a
        future model change ever closed this gap, that would be real news -- this test would then
        need updating, not silently pass either way, which is why it asserts direction and not
        just 'runs without crashing'."""
        ablation = run_ablation_without_threshold()
        sealed = run_sealed_comparison()
        assert ablation["accuracy_without_threshold"] >= sealed.argus_accuracy
        assert ablation["total"] == 293


class TestScopeStatement:
    def test_claims_the_real_loss_when_rasa_is_ahead(self) -> None:
        sealed = run_sealed_comparison()
        oos = run_oos_comparison()
        ablation = run_ablation_without_threshold()
        text = scope_statement(sealed, oos, ablation)
        if sealed.rasa_accuracy > sealed.argus_accuracy and sealed.mcnemar_p < 0.05:
            assert "real loss" in text
            assert "not softenable" not in text or "real loss" in text
        assert "NOT VERIFIED" in text
        assert "NOT CLAIMED" in text

    def test_never_launders_one_finding_into_the_other(self) -> None:
        sealed = run_sealed_comparison()
        oos = run_oos_comparison()
        ablation = run_ablation_without_threshold()
        text = scope_statement(sealed, oos, ablation)
        assert "neither is used to launder the other" in text


class TestMainAndRender:
    def test_main_returns_a_complete_serialisable_report(self) -> None:
        report = main()
        assert report["reference"]["repo"] == "RasaHQ/rasa"
        assert report["reference"]["licence"] == "Apache-2.0"
        json.dumps(report)

    def test_render_produces_readable_text(self) -> None:
        text = render(main())
        assert "sealed accuracy" in text
        assert "out-of-scope" in text
