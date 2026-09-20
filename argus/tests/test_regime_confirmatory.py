"""The exploratory regime finding, re-tested at a pre-committed size on a seed it never saw.

**One of the two exploratory conclusions reversed, and that is the reason this file exists.** At 24
replicates ARGUS appeared to lead the two-line incumbent rule 15-9 (p=0.307). At 120 the incumbent
wins **82 of 120** paired f1 comparisons at p=7.3e-05. The direction was never established at n=24;
the underpowered run simply happened to point the flattering way, which is the failure mode every
statistical guard in this project exists to catch and which caught us here.

What survives is narrow and real: **ARGUS beats ruptures on f1, significantly**, on a sample it had
not seen.

The seed is disjoint from the exploratory one on purpose. The exploratory run ended at p=0.057, and
the obvious move — add replicates until it crosses 0.05 — is optional stopping, which produces a
number that looks like a p-value and is not one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from argus.eval.regime_groundtruth_audit import (
    BASE_SEED,
    CONFIRMATORY_REPLICATES,
    CONFIRMATORY_SEED,
    SCENARIOS,
)

ARTEFACT = Path(__file__).resolve().parents[1] / "data" / "regime_confirmatory.json"


@pytest.fixture(scope="module")
def report() -> dict:
    if not ARTEFACT.exists():
        pytest.skip("run `python -m argus.eval.regime_groundtruth_audit` to produce the artefact")
    return json.loads(ARTEFACT.read_text(encoding="utf-8"))


class TestTheDesignIsHonest:
    def test_the_confirmatory_seed_is_not_the_exploratory_one(self) -> None:
        """The whole point. A confirmatory test on the same sample confirms nothing."""
        assert CONFIRMATORY_SEED != BASE_SEED

    def test_the_sample_size_was_fixed_in_the_source(self) -> None:
        """Pre-committed means it is a constant somebody can read, not an argument somebody tuned
        until the answer came out right."""
        assert CONFIRMATORY_REPLICATES == 40
        assert CONFIRMATORY_REPLICATES * len(SCENARIOS) == 120

    def test_the_artefact_records_the_design(self, report: dict) -> None:
        design = report["design"]
        assert design["series"] == 120
        assert design["independent_of_exploratory"] is True
        assert "optional stopping" in design["pre_committed"]


class TestWhatSurvived:
    def test_argus_beats_ruptures_on_f1_significantly(self, report: dict) -> None:
        """The earned claim, and the only one. ruptures is a 2,084-star specialist."""
        row = report["paired_sign_tests"]["argus_vs_ruptures"]["f1"]
        assert row["wins"] > row["losses"]
        assert row["significant_at_5pct"]
        assert row["p_two_sided"] < 0.05
        assert report["argus_beats_ruptures_on_f1"] is True

    def test_argus_is_indistinguishable_from_stumpy(self, report: dict) -> None:
        """Predicted by the exact matrix-profile parity, and therefore a check on the benchmark
        rather than on ARGUS: if these two ever separate, the harness is wrong."""
        row = report["paired_sign_tests"]["argus_vs_stumpy"]["f1"]
        assert not row["significant_at_5pct"]


class TestWhatReversed:
    def test_the_incumbent_beats_argus_on_f1_and_it_is_recorded(self, report: dict) -> None:
        """**Asserted so it cannot be quietly dropped.** `desk/regime.py` exists to see what the
        two-line volatility rule at `strategies/track1_suite.py:206-215` cannot. Properly powered,
        that rule wins the majority of paired f1 comparisons — and this test fails if the claim is
        ever softened without the measurement changing."""
        row = report["paired_sign_tests"]["argus_vs_incumbent"]["f1"]
        assert row["losses"] > row["wins"]
        assert row["significant_at_5pct"]
        assert report["argus_loses_to_incumbent_on_f1"] is True

    def test_both_framings_are_published_not_just_the_flattering_one(self, report: dict) -> None:
        """Pooled, ARGUS has the highest f1 of the four; paired, the incumbent wins more often.
        Both are true, and quoting either alone misleads — so the artefact carries both and the
        verdict names the tension instead of resolving it in our favour."""
        pooled = report["pooled_f1"]
        assert pooled["argus"] > pooled["incumbent"]
        assert report["paired_sign_tests"]["argus_vs_incumbent"]["f1"]["losses"] > 60
        verdict = report["verdict"]
        assert "REVERSED" in verdict
        assert "highest pooled f1" in verdict

    def test_the_verdict_leads_with_the_confirmation_and_states_the_reversal(
        self, report: dict
    ) -> None:
        assert report["verdict"].startswith("CONFIRMED")
        assert "does not survive power" in report["verdict"]
