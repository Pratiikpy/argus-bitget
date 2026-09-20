"""Factor memory tests.

Two properties, and the first is the reason the module exists.

**No outcome may reach the proposer.** `factor_lab.ProposerContext` has no field for performance by
construction; a memory is the obvious way to defeat that without touching the class. The tests here
score factors with distinctive numbers and then assert those numbers appear nowhere in what the
proposer is handed — not as a value, not as a grade, not as a ranking. This is the test FactorMiner
would fail: its generator receives ``=== RECOMMENDED DIRECTIONS (P_succ) ===`` with per-pattern
success grades (``factorminer/memory/retrieval.py:742-750``).

**A hypothesis is evaluated once.** Identity is the canonical form — expression and horizon, not the
name — so a renamed re-proposal is suppressed rather than paid for twice, and the trial count that
feeds the Deflated Sharpe gate is not inflated by repeats.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from argus.research.factor_lab import Factor, FactorRecord, Lifecycle
from argus.research.memory import (
    MAX_BARRED_IN_SIGNAL,
    MAX_TRIALS,
    PERFORMANCE_FIELDS,
    FactorMemory,
    MemorySignal,
    Rejection,
    Trial,
    assert_carries_no_outcome,
    canonical_form,
    certified_forms,
    classify,
)

AT = datetime(2026, 9, 12, 12, 0, tzinfo=UTC)

# A value chosen to be findable in any serialisation. If the proposer's view ever contains it, an
# outcome has leaked.
TELLTALE = 7.7777


def _record(
    expression: str = "slow_trend",
    *,
    name: str = "",
    horizon: int = 24,
    state: Lifecycle = Lifecycle.REJECTED,
    reason: str = "",
    trial: int = 1,
) -> FactorRecord:
    record = FactorRecord(
        factor=Factor(name=name or expression, expression=expression, rationale="t",
                      horizon_bars=horizon),
        trial_number=trial,
    )
    record.state = state
    record.rejection_reason = reason
    record.gross_sharpe = TELLTALE
    record.net_sharpe = TELLTALE
    record.oos_sharpe = TELLTALE
    record.dsr = TELLTALE
    return record


class TestNoOutcomeReachesTheProposer:
    """The property the whole module is shaped around."""

    def test_no_score_appears_in_the_signal(self) -> None:
        memory = FactorMemory()
        memory.remember(_record(reason="net Sharpe -0.4 after the 12bps round trip"), at=AT)
        memory.remember(_record("slow_fade", state=Lifecycle.CERTIFIED), at=AT)
        rendered = json.dumps(memory.signal().as_dict()) + " ".join(memory.signal().render())
        assert str(TELLTALE) not in rendered
        assert "7.7" not in rendered

    def test_a_performance_rejection_is_never_quoted(self) -> None:
        memory = FactorMemory()
        memory.remember(_record(reason="out-of-sample Sharpe -1.230"), at=AT)
        rendered = json.dumps(memory.signal().as_dict()) + " ".join(memory.signal().render())
        assert "Sharpe" not in rendered and "sample" not in rendered

    def test_certification_is_invisible_to_the_proposer(self) -> None:
        """Knowing which factor was certified is the single most contaminating fact there is."""
        memory = FactorMemory()
        memory.remember(_record("slow_trend", state=Lifecycle.CERTIFIED), at=AT)
        memory.remember(_record("slow_fade", state=Lifecycle.REJECTED, reason="DSR 0.2"), at=AT)
        signal = memory.signal()
        # Both appear, and nothing distinguishes the winner from the loser.
        assert set(signal.already_evaluated) == {"slow_trend@24", "slow_fade@24"}
        assert not signal.structurally_barred

    def test_the_signal_has_no_outcome_field(self) -> None:
        assert not set(MemorySignal.__dataclass_fields__) & PERFORMANCE_FIELDS

    def test_the_guard_runs_at_import_not_only_here(self) -> None:
        assert_carries_no_outcome()  # raises if MemorySignal ever grows one

    def test_the_deployment_path_can_still_see_outcomes(self) -> None:
        """Keeping it out of the proposer must not mean throwing it away."""
        memory = FactorMemory()
        memory.remember(_record("slow_trend", state=Lifecycle.CERTIFIED), at=AT)
        memory.remember(_record("slow_fade", reason="DSR 0.2"), at=AT)
        assert certified_forms(memory) == ("slow_trend@24",)


class TestTheStructuralLine:
    """Structural facts may cross; measured ones may not."""

    @pytest.mark.parametrize("reason", [
        "unscoreable: fewer than two returns",
        "no scoreable out-of-sample slice",
        "DSR uncomputable: variance is zero",
    ])
    def test_unmeasurable_is_structural(self, reason: str) -> None:
        assert classify(reason) is Rejection.STRUCTURAL

    @pytest.mark.parametrize("reason", [
        "net Sharpe -0.412 after the 12bps round trip (gross was 0.03)",
        "out-of-sample Sharpe -1.230",
        "DSR 0.41 over 8 trials - not distinguishable from the best of a random search",
        "anti-overfit: FAIL",
    ])
    def test_measured_and_found_wanting_is_performance(self, reason: str) -> None:
        assert classify(reason) is Rejection.PERFORMANCE

    def test_no_reason_is_neither(self) -> None:
        assert classify("") is Rejection.NONE
        assert classify("   ") is Rejection.NONE

    def test_an_unrecognised_reason_fails_closed(self) -> None:
        """A rejection reason added to the lab and not to this table must keep its number in."""
        assert classify("some future gate said no, 0.31") is Rejection.PERFORMANCE

    def test_only_structural_bars_are_published(self) -> None:
        memory = FactorMemory()
        memory.remember(_record("slow_trend", reason="no scoreable out-of-sample slice"), at=AT)
        memory.remember(_record("slow_fade", reason="net Sharpe -0.4"), at=AT)
        assert memory.signal().structurally_barred == ("slow_trend@24",)

    def test_the_bar_is_phrased_as_unmeasurable_not_as_bad(self) -> None:
        memory = FactorMemory()
        memory.remember(_record(reason="unscoreable: no bars"), at=AT)
        text = " ".join(memory.signal().render())
        assert "not a judgement of merit" in text


class TestIdentityIsTheHypothesisNotTheName:
    def test_two_names_for_one_expression_are_one_hypothesis(self) -> None:
        a = Factor(name="alpha", expression="slow_trend", rationale="t")
        b = Factor(name="beta", expression="slow_trend", rationale="t")
        assert canonical_form(a) == canonical_form(b)

    def test_the_horizon_is_part_of_the_identity(self) -> None:
        a = Factor(name="a", expression="slow_trend", rationale="t", horizon_bars=24)
        b = Factor(name="a", expression="slow_trend", rationale="t", horizon_bars=48)
        assert canonical_form(a) != canonical_form(b)

    def test_a_renamed_re_proposal_is_recognised(self) -> None:
        memory = FactorMemory()
        memory.remember(_record("slow_trend", name="alpha"), at=AT)
        assert memory.seen(Factor(name="beta", expression="slow_trend", rationale="t"))

    def test_the_first_evaluation_can_be_found_again(self) -> None:
        memory = FactorMemory()
        memory.remember(_record("slow_trend", name="alpha", trial=3), at=AT)
        memory.remember(_record("slow_trend", name="beta", trial=9), at=AT)
        found = memory.first_trial_of(Factor(name="x", expression="slow_trend", rationale="t"))
        assert found is not None and found.trial_number == 3

    def test_an_unseen_hypothesis_has_no_first_trial(self) -> None:
        assert FactorMemory().first_trial_of(
            Factor(name="x", expression="slow_trend", rationale="t")
        ) is None


class TestTheCapsAreEnforcedNotDeclared:
    """FactorMiner declares three caps and applies none of them."""

    def test_the_trial_cap_actually_evicts(self) -> None:
        memory = FactorMemory()
        for i in range(MAX_TRIALS + 5):
            memory.trials.append(Trial(
                canonical=f"f{i}@24", name=f"f{i}", trial_number=i,
                terminal_state="rejected", rejection_reason="", rejection=Rejection.NONE, at="",
            ))
        memory._enforce_cap()
        assert len(memory.trials) == MAX_TRIALS

    def test_an_eviction_is_reported_rather_than_silent(self) -> None:
        memory = FactorMemory()
        memory.trials = [
            Trial(canonical=f"f{i}@24", name="f", trial_number=i, terminal_state="rejected",
                  rejection_reason="", rejection=Rejection.NONE, at="")
            for i in range(MAX_TRIALS + 3)
        ]
        memory._enforce_cap()
        assert memory.dropped == 3

    def test_the_published_bar_list_is_bounded(self) -> None:
        memory = FactorMemory()
        for i in range(MAX_BARRED_IN_SIGNAL + 10):
            memory.remember(
                _record("slow_trend", horizon=i + 1, reason="unscoreable: none"), at=AT
            )
        assert len(memory.signal().structurally_barred) == MAX_BARRED_IN_SIGNAL

    def test_the_bar_list_keeps_the_most_recent(self) -> None:
        memory = FactorMemory()
        for i in range(MAX_BARRED_IN_SIGNAL + 5):
            memory.remember(
                _record("slow_trend", horizon=i + 1, reason="unscoreable: none"), at=AT
            )
        assert memory.signal().structurally_barred[0].endswith(f"@{MAX_BARRED_IN_SIGNAL + 5}")

    def test_the_bar_list_does_not_repeat_one_form(self) -> None:
        memory = FactorMemory()
        for _ in range(5):
            memory.remember(_record("slow_trend", reason="unscoreable: none"), at=AT)
        assert memory.signal().structurally_barred == ("slow_trend@24",)


class TestItSurvivesARestart:
    def test_a_saved_memory_reads_back_identically(self, tmp_path: Path) -> None:
        memory = FactorMemory()
        memory.remember(_record("slow_trend", reason="net Sharpe -0.4"), at=AT)
        memory.remember(_record("slow_fade", reason="unscoreable: none"), at=AT)
        path = memory.save(tmp_path / "m.json")
        back = FactorMemory.load(path)
        assert [t.as_dict() for t in back.trials] == [t.as_dict() for t in memory.trials]

    def test_saving_counts_a_run(self, tmp_path: Path) -> None:
        memory = FactorMemory()
        memory.save(tmp_path / "m.json")
        memory.save(tmp_path / "m.json")
        assert FactorMemory.load(tmp_path / "m.json").runs == 2

    def test_the_run_count_reaches_the_proposer_because_it_is_not_an_outcome(
        self, tmp_path: Path
    ) -> None:
        memory = FactorMemory()
        memory.save(tmp_path / "m.json")
        assert FactorMemory.load(tmp_path / "m.json").signal().runs == 1

    def test_a_missing_file_is_a_cold_start_not_an_error(self, tmp_path: Path) -> None:
        memory = FactorMemory.load(tmp_path / "absent.json")
        assert memory.trials == [] and memory.runs == 0

    def test_an_unreadable_shape_is_refused_rather_than_emptied(self, tmp_path: Path) -> None:
        path = tmp_path / "m.json"
        path.write_text(json.dumps({"schema_version": 99, "trials": []}), encoding="utf-8")
        with pytest.raises(ValueError, match="partially-understood"):
            FactorMemory.load(path)

    def test_the_structural_classification_survives_the_round_trip(self, tmp_path: Path) -> None:
        memory = FactorMemory()
        memory.remember(_record(reason="unscoreable: none"), at=AT)
        path = memory.save(tmp_path / "m.json")
        assert FactorMemory.load(path).trials[0].rejection is Rejection.STRUCTURAL


class TestGoingInCirclesIsVisible:
    def test_suppressed_duplicates_are_counted(self) -> None:
        memory = FactorMemory()
        memory.note_duplicate()
        memory.note_duplicate()
        assert memory.as_dict()["duplicates_suppressed"] == 2

    def test_the_count_persists(self, tmp_path: Path) -> None:
        memory = FactorMemory()
        memory.note_duplicate()
        path = memory.save(tmp_path / "m.json")
        assert FactorMemory.load(path).duplicates_suppressed == 1
