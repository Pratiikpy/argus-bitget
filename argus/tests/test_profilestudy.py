"""Profile-divergence study tests.

This study exists to answer a Track 3 judged criterion with a rate rather than an anecdote, and its
first run produced a number that was **wrong in the most dangerous way**: 83.3% divergence, which
looked like a strong result and was entirely a property of the declared size ladder. Every rung came
back at exactly 0% or 100%, so the real instruments and horizons moved nothing at all.

So the properties under test are the ones that stop that recurring:

* a rung where every frame diverges is **not discriminating** — it says nothing about the desk;
* the share attributable to real dimensions counts only discriminating rungs;
* real and declared inputs stay labelled and never merge into one number;
* an empty ledger yields UNDEFINED, never a divergence rate of zero.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from argus.desk.personalisation import Proposal, diverge, judge, standard_profiles
from argus.eval.profilestudy import (
    DECLARED_DIMENSIONS,
    REAL_DIMENSIONS,
    Frame,
    ProfileStudyError,
    RungResult,
    proposals_for,
    real_frames,
    study,
)

AT = datetime(2026, 9, 15, 13, 30, tzinfo=UTC)


def _ledger(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    import json

    path = tmp_path / "ledger.jsonl"
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + ("\n" if rows else ""), encoding="utf-8"
    )
    return path


class TestARungThatAlwaysDivergesProvesNothing:
    def test_a_rung_where_every_frame_diverges_is_not_discriminating(self) -> None:
        """100% divergence at a rung is the ladder talking, not the instruments."""
        assert RungResult("6000", "8", frames=195, diverged=195).discriminating is False

    def test_a_rung_where_no_frame_diverges_is_not_discriminating(self) -> None:
        assert RungResult("2000", "2", frames=195, diverged=0).discriminating is False

    def test_a_rung_that_splits_the_frames_is_discriminating(self) -> None:
        rung = RungResult("2000", "2", frames=195, diverged=15)
        assert rung.discriminating is True
        assert rung.rate == pytest.approx(15 / 195)

    def test_a_rung_with_no_frames_has_no_rate_rather_than_zero(self) -> None:
        rung = RungResult("2000", "2", frames=0, diverged=0)
        assert rung.rate is None
        assert rung.as_dict()["rate_pct"] is None


class TestTheHeadlineIsNotAllowedToStandAlone:
    def test_attributable_share_counts_only_discriminating_rungs(self, tmp_path: Path) -> None:
        rows = [
            {"symbol": "NVDAUSDT", "hours_to_discovery": 24.0},
            {"symbol": "SQQQUSDT", "hours_to_discovery": 24.0},
        ]
        result = study(ledger=_ledger(tmp_path, rows), now=AT)
        assert result.rate is not None
        attributable = result.attributable_to_real
        assert attributable is not None
        # The headline counts every diverging proposal; the attributable share counts only those
        # at rungs where the real frames changed the answer, so it must be the smaller number.
        assert attributable < result.rate

    def test_the_verdict_warns_when_no_rung_discriminates(self, tmp_path: Path) -> None:
        """One symbol means no frame can separate from another, whatever the ladder does."""
        rows = [{"symbol": "NVDAUSDT", "hours_to_discovery": 24.0}]
        verdict = study(ledger=_ledger(tmp_path, rows), now=AT).verdict
        assert "none of the declared rungs discriminates" in verdict
        assert "property of the size/loss ladder" in verdict

    def test_the_verdict_quantifies_the_real_share_when_a_rung_does_discriminate(
        self, tmp_path: Path
    ) -> None:
        rows = [
            {"symbol": "NVDAUSDT", "hours_to_discovery": 24.0},
            {"symbol": "TQQQUSDT", "hours_to_discovery": 24.0},
        ]
        verdict = study(ledger=_ledger(tmp_path, rows), now=AT).verdict
        assert "turns on a real dimension" in verdict
        assert "the ladder's doing" in verdict


class TestRealAndDeclaredNeverMerge:
    def test_both_dimension_lists_reach_the_artefact(self, tmp_path: Path) -> None:
        rows = [{"symbol": "NVDAUSDT", "hours_to_discovery": 24.0}]
        payload = study(ledger=_ledger(tmp_path, rows), now=AT).as_dict()
        assert payload["real_dimensions"] == list(REAL_DIMENSIONS)
        assert payload["declared_dimensions"] == list(DECLARED_DIMENSIONS)

    def test_the_notional_is_declared_not_read_from_the_log(self) -> None:
        """Every recorded decision carries quantity 0; a size here could only be invented."""
        assert "notional" in DECLARED_DIMENSIONS
        assert "notional" not in REAL_DIMENSIONS


class TestFramesComeFromTheRecord:
    def test_repeated_pairs_are_deduplicated_but_counted(self, tmp_path: Path) -> None:
        rows = [
            {"symbol": "NVDAUSDT", "hours_to_discovery": 24.0},
            {"symbol": "NVDAUSDT", "hours_to_discovery": 24.0},
            {"symbol": "NVDAUSDT", "hours_to_discovery": 2.0},
        ]
        frames = real_frames(_ledger(tmp_path, rows))
        assert len(frames) == 2
        assert {f.decisions for f in frames} == {1, 2}

    def test_a_decision_with_no_horizon_is_skipped_not_defaulted_to_zero(
        self, tmp_path: Path
    ) -> None:
        """A zero horizon means "trading right now", which is a claim, not an absence."""
        rows = [
            {"symbol": "NVDAUSDT", "hours_to_discovery": None},
            {"symbol": "NVDAUSDT", "hours_to_discovery": 24.0},
        ]
        frames = real_frames(_ledger(tmp_path, rows))
        assert [f.horizon_hours for f in frames] == [24.0]

    def test_an_absent_ledger_yields_no_frames(self, tmp_path: Path) -> None:
        assert real_frames(tmp_path / "absent.jsonl") == []

    def test_an_empty_record_reports_undefined_rather_than_a_rate_of_zero(
        self, tmp_path: Path
    ) -> None:
        result = study(ledger=_ledger(tmp_path, []), now=AT)
        assert result.frames == ()
        assert "UNDEFINED" in result.verdict
        assert "not zero" in result.verdict

    def test_every_frame_is_crossed_with_every_declared_rung(self) -> None:
        frames = [Frame("NVDAUSDT", 24.0, 1), Frame("TSLAUSDT", 2.0, 1)]
        assert len(proposals_for(frames)) == len(frames) * 3 * 2


class TestOneMandateCannotDiverge:
    def test_a_single_profile_is_refused_rather_than_scored(self, tmp_path: Path) -> None:
        with pytest.raises(ProfileStudyError):
            study(
                ledger=_ledger(tmp_path, [{"symbol": "NVDAUSDT", "hours_to_discovery": 24.0}]),
                profiles=(standard_profiles()[0],),
                now=AT,
            )


class TestTheExclusionThatWasNeverChecked:
    """`judge` checked horizon, loss and size, and skipped `excluded_symbols` entirely.

    It surfaced from this study: over 195 real frames every symbol diverged at exactly 83%,
    including the two the conservative mandate names in its exclusions. A forbidden instrument
    behaving identically to a permitted one is not a plausible result.
    """

    def test_an_excluded_symbol_is_refused_outright(self) -> None:
        conservative = standard_profiles()[0]
        assert "SQQQUSDT" in conservative.excluded_symbols
        verdict = judge(
            conservative,
            Proposal("SQQQUSDT", "technology", Decimal("2000"), 24.0, Decimal("2")),
        )
        assert verdict.outcome.value == "refused"
        assert verdict.permitted_notional == Decimal("0")
        assert "exclusions" in verdict.reasons[0]

    def test_the_exclusion_outranks_a_proposal_that_passes_every_other_limit(self) -> None:
        """Small, short, low-loss: clears horizon, tolerance and size, and is still refused."""
        conservative = standard_profiles()[0]
        tiny = Proposal("TQQQUSDT", "technology", Decimal("1"), 1.0, Decimal("0"))
        assert judge(conservative, tiny).outcome.value == "refused"

    def test_a_permitted_symbol_on_the_same_terms_is_taken(self) -> None:
        conservative = standard_profiles()[0]
        same = Proposal("NVDAUSDT", "technology", Decimal("2000"), 24.0, Decimal("2"))
        assert judge(conservative, same).outcome.value == "taken"

    def test_the_exclusion_makes_the_two_mandates_diverge(self) -> None:
        excluded = Proposal("SQQQUSDT", "technology", Decimal("2000"), 24.0, Decimal("2"))
        permitted = Proposal("NVDAUSDT", "technology", Decimal("2000"), 24.0, Decimal("2"))
        profiles = standard_profiles()
        assert diverge(excluded, profiles).diverged is True
        assert diverge(permitted, profiles).diverged is False


class TestTheStatedCeilingIsTheOneApplied:
    def test_a_taken_verdict_names_the_ceiling_including_slack(self) -> None:
        """A 52h proposal was reported TAKEN "within a 48h horizon", which reads as a bug."""
        aggressive = standard_profiles()[1]
        verdict = judge(
            aggressive,
            Proposal("NVDAUSDT", "technology", Decimal("2000"), 51.99, Decimal("2")),
        )
        assert verdict.outcome.value == "taken"
        assert "72h ceiling this mandate actually applies" in verdict.reasons[0]
        assert "slack" in verdict.reasons[0]
