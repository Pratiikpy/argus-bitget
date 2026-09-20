"""Cadence tests — the fix for a defect that was in the register's own opening.

All 36 founding claims were committed at one 24-hour horizon, so every one resolved the next day.
The handbook puts judge review at 9/22 to 10/7. A record that finished resolving a week before
anyone opens it is a record of something, and the entire argument for a register is that it is
*running*.

So the property under test is coverage over time, not correctness of a single claim: at any moment
inside a window there must be something pending and something recently resolved.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from argus.register.cadence import (
    ANCHOR_EVERY,
    HORIZON_LADDER,
    _symbol_for,
    build_cadence_batch,
    cycle_index,
    horizon_coverage,
)
from argus.register.claims import REGISTER_PATH, Predicate, register, validate

NOW = datetime(2026, 9, 14, 12, tzinfo=UTC)
WINDOW = (datetime(2026, 9, 22, tzinfo=UTC), datetime(2026, 10, 7, 23, 59, tzinfo=UTC))


class TestTheLadder:
    def test_it_spans_days_not_hours(self) -> None:
        """A single short horizon is what froze the founding batch. The ladder must reach past the
        far end of the judging window from the day it is registered."""
        assert min(HORIZON_LADDER) <= timedelta(hours=24)
        assert max(HORIZON_LADDER) >= timedelta(days=30)

    def test_every_horizon_is_distinct(self) -> None:
        assert len(set(HORIZON_LADDER)) == len(HORIZON_LADDER)

    def test_a_single_registration_covers_the_window_and_beyond(self) -> None:
        """Registered a week before the window opens, one cadence batch must still have claims
        resolving inside it and claims outstanding after it."""
        registered = datetime(2026, 9, 15, tzinfo=UTC)
        resolutions = [registered + h for h in HORIZON_LADDER]
        inside = [r for r in resolutions if WINDOW[0] <= r <= WINDOW[1]]
        after = [r for r in resolutions if r > WINDOW[1]]
        assert inside, "nothing resolves inside the judging window"
        assert after, "nothing is still pending when the window closes"


class TestNothingIsChosen:
    def test_the_subject_rotates_by_index(self) -> None:
        """A cadence that let a claimant pick today's subject would let them avoid the instruments
        they expect to be wrong about."""
        universe = ("A", "B", "C")
        assert [_symbol_for(i, universe) for i in range(6)] == ["A", "B", "C", "A", "B", "C"]

    def test_the_rotation_is_a_function_of_the_index_alone(self) -> None:
        universe = ("A", "B", "C")
        assert _symbol_for(4, universe) == _symbol_for(7, universe)

    def test_the_cycle_index_is_derived_from_the_record(self, tmp_path: Path) -> None:
        """Not stored separately: a second piece of state could disagree with the record, and the
        record is the thing that has to be right."""
        path = tmp_path / "register.jsonl"
        assert cycle_index(path) == 0
        from argus.register.claims import make_claim

        batch = [
            make_claim(
                claimant="argus", subject="NVDAUSDT", predicate=Predicate.ABS_MOVE_ABOVE_BPS,
                threshold=str(10 + i), resolves_at=NOW + timedelta(hours=24),
                source="bitget:1H:market", confidence=0.6, now=NOW,
            )
            for i in range(len(HORIZON_LADDER))
        ]
        register(batch, path=path)
        assert cycle_index(path) == 1


class TestCoverageIsMeasured:
    def test_an_empty_register_reports_the_freeze(self, tmp_path: Path) -> None:
        got = horizon_coverage(path=tmp_path / "absent.jsonl", window=WINDOW)
        assert got["resolving_inside_window"] == 0
        assert "frozen" in got["verdict"]

    def test_a_batch_that_all_resolves_early_reports_the_freeze(self, tmp_path: Path) -> None:
        """The exact shape of the original defect, asserted so it cannot return unnoticed."""
        from argus.register.claims import make_claim

        path = tmp_path / "register.jsonl"
        register(
            [
                make_claim(
                    claimant="argus", subject="NVDAUSDT",
                    predicate=Predicate.ABS_MOVE_ABOVE_BPS, threshold=str(10 + i),
                    resolves_at=NOW + timedelta(hours=24), source="bitget:1H:market",
                    confidence=0.6, now=NOW,
                )
                for i in range(3)
            ],
            path=path,
        )
        got = horizon_coverage(path=path, window=WINDOW)
        assert got["claims_total"] == 3
        assert got["resolving_inside_window"] == 0
        assert "frozen" in got["verdict"]

    def test_a_laddered_batch_covers_the_window(self, tmp_path: Path) -> None:
        from argus.register.claims import make_claim

        path = tmp_path / "register.jsonl"
        start = datetime(2026, 9, 15, tzinfo=UTC)
        register(
            [
                make_claim(
                    claimant="argus", subject="NVDAUSDT",
                    predicate=Predicate.ABS_MOVE_ABOVE_BPS, threshold=str(10 + i),
                    resolves_at=start + horizon, source="bitget:1H:market",
                    confidence=0.6, now=start,
                )
                for i, horizon in enumerate(HORIZON_LADDER)
            ],
            path=path,
        )
        got = horizon_coverage(path=path, window=WINDOW)
        assert got["resolving_inside_window"] >= 1
        assert got["still_pending_at_window_open"] >= 2
        assert "frozen" not in got["verdict"]


class TestTheBatchIsWellFormed:
    def test_every_claim_would_pass_registration(self) -> None:
        batch = build_cadence_batch(now=NOW, symbols=("NVDAUSDT",), index=0)
        if not batch:
            pytest.skip("no market history on this machine")
        for claim in batch:
            validate(claim)

    def test_one_claim_per_horizon(self) -> None:
        batch = build_cadence_batch(now=NOW, symbols=("NVDAUSDT",), index=0)
        if not batch:
            pytest.skip("no market history on this machine")
        assert len(batch) == len(HORIZON_LADDER)
        assert len({c.resolves_at for c in batch}) == len(HORIZON_LADDER)

    def test_the_threshold_grows_with_the_horizon(self) -> None:
        """Random-walk scaling: a 30-day move should have to clear far more than a 24-hour one, or
        the long claims would be trivially true and score nothing."""
        batch = build_cadence_batch(now=NOW, symbols=("NVDAUSDT",), index=0)
        if not batch:
            pytest.skip("no market history on this machine")
        thresholds = [float(c.threshold) for c in batch]
        assert thresholds == sorted(thresholds)
        assert thresholds[-1] > thresholds[0] * 4

    def test_the_cadence_makes_no_directional_claim(self) -> None:
        """Direction is committed once, at exactly 0.50, in the founding batch. Repeating a coin
        four times a day would pad the register with claims that cannot inform anyone."""
        batch = build_cadence_batch(now=NOW, symbols=("NVDAUSDT",), index=0)
        if not batch:
            pytest.skip("no market history on this machine")
        assert all(c.predicate is Predicate.ABS_MOVE_ABOVE_BPS for c in batch)


class TestTheLiveRecord:
    def test_the_register_is_no_longer_frozen_for_the_window(self) -> None:
        """The defect this module exists to fix, checked against the record that actually exists."""
        if not REGISTER_PATH.exists():
            pytest.skip("the register has not been opened on this machine")
        got = horizon_coverage()
        assert got["resolving_inside_window"] >= 1, got["verdict"]

    def test_anchoring_is_periodic_not_per_cycle(self) -> None:
        """A calendar submission is a round trip to four operators, and the chain already binds
        every earlier claim into its head."""
        assert ANCHOR_EVERY >= 2

    def test_the_cadence_runs_on_the_schedule(self) -> None:
        script = Path(__file__).resolve().parents[1] / "run_paper_cycle.ps1"
        if not script.exists():
            pytest.skip("the cycle script is not on this machine")
        text = script.read_text(encoding="utf-8", errors="replace")
        assert "argus.register.cadence" in text
        # Registration before resolution: a cycle that resolved first would leave the newest claims
        # unregistered until the next run.
        assert text.index("argus.register.cadence") < text.index("argus.register.resolve")
