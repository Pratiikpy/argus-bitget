"""Tests for the settlement-seal backfill, run against synthetic ledgers before this module is
ever pointed at the real one. The real ledger's own migration is a one-time, manually-run
operation (`python -m argus.paper.migrate_settlement_seals`), not something a test suite repeats.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from argus.paper.ledger import PaperLedger
from argus.paper.migrate_settlement_seals import Migration, MigrationError, migrate

T0 = datetime(2026, 3, 9, 14, 0, tzinfo=UTC)


def _pre_seal_ledger(path: Path, *, n_settled: int = 3, n_open: int = 1) -> PaperLedger:
    """A ledger with settled decisions and no seals — the exact shape every real row settled
    before 2026-09-22 is in, built here by settling normally and then stripping the seals a
    real pre-migration file would never have had."""
    led = PaperLedger(path=path)
    for i in range(n_settled + n_open):
        led.record(
            symbol="NVDAUSDT", verdict="trade", side="BUY",
            quantity=Decimal("10"), entry_price=Decimal("220"),
            stated_confidence=0.7, thesis=f"t{i}", invalidation=("x",),
            market_state_hash="m", approved_intent_hash="a",
            session_phase="weekend", hours_to_discovery=30.0,
            decided_at=T0 + timedelta(hours=i),
        )
    for i in range(n_settled):
        led.settle(i + 1, exit_price=Decimal("230"))

    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    stripped = [ln for ln in lines if json.loads(ln).get("kind", "decision") != "settlement_seal"]
    path.write_text("\n".join(stripped) + "\n", encoding="utf-8")
    return PaperLedger(path=path)


class TestTheBackfillItself:
    def test_every_unsealed_settlement_gets_a_seal(self, tmp_path: Path) -> None:
        path = tmp_path / "p.jsonl"
        _pre_seal_ledger(path, n_settled=3, n_open=1)
        result = migrate(path)
        assert len(result.backfilled) == 3
        assert result.backfilled == (1, 2, 3)

        after = PaperLedger(path=path)
        assert len(after.seals) == 3
        report = after.verify()
        assert report["chain_intact"] is True
        assert report["tampered_settlements"] == []
        assert report["unsealed_settlements"] == []

    def test_a_backup_is_written_before_any_change(self, tmp_path: Path) -> None:
        path = tmp_path / "p.jsonl"
        _pre_seal_ledger(path, n_settled=2, n_open=0)
        original_bytes = path.read_bytes()
        result = migrate(path)
        assert result.backup is not None
        assert result.backup.read_bytes() == original_bytes

    def test_open_positions_are_not_touched(self, tmp_path: Path) -> None:
        path = tmp_path / "p.jsonl"
        _pre_seal_ledger(path, n_settled=1, n_open=2)
        migrate(path)
        after = PaperLedger(path=path)
        assert after.seals[0].target_seq == 1
        assert len(after.seals) == 1

    def test_running_twice_is_a_no_op_the_second_time(self, tmp_path: Path) -> None:
        path = tmp_path / "p.jsonl"
        _pre_seal_ledger(path, n_settled=2, n_open=0)
        first = migrate(path)
        assert len(first.backfilled) == 2
        second = migrate(path)
        assert second.backfilled == ()
        assert second.already_sealed == 2

    def test_a_clean_ledger_with_no_settlements_needs_nothing(self, tmp_path: Path) -> None:
        path = tmp_path / "p.jsonl"
        led = PaperLedger(path=path)
        led.record(
            symbol="NVDAUSDT", verdict="trade", side="BUY",
            quantity=Decimal("10"), entry_price=Decimal("220"),
            stated_confidence=0.7, thesis="t", invalidation=(),
            market_state_hash="m", approved_intent_hash="a",
            session_phase="weekend", hours_to_discovery=30.0, decided_at=T0,
        )
        result = migrate(path)
        assert result.backfilled == ()
        assert result.backup is None


class TestDryRun:
    def test_dry_run_writes_nothing(self, tmp_path: Path) -> None:
        path = tmp_path / "p.jsonl"
        _pre_seal_ledger(path, n_settled=2, n_open=0)
        before_bytes = path.read_bytes()
        result = migrate(path, dry_run=True)
        assert path.read_bytes() == before_bytes
        assert result.backfilled == (1, 2)
        assert result.backup is None
        assert len(PaperLedger(path=path).seals) == 0


class TestItRefusesToGuess:
    def test_a_missing_file_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(MigrationError, match="does not exist"):
            migrate(tmp_path / "nope.jsonl")

    def test_a_seal_whose_hash_already_disagrees_is_refused_not_papered_over(
        self, tmp_path: Path
    ) -> None:
        """If a seal already exists but does not match its target's current fields, this
        migration must not silently add cover for it — that would launder a real defect into a
        clean-looking backfill."""
        path = tmp_path / "p.jsonl"
        led = PaperLedger(path=path)
        led.record(
            symbol="NVDAUSDT", verdict="trade", side="BUY",
            quantity=Decimal("10"), entry_price=Decimal("220"),
            stated_confidence=0.7, thesis="t", invalidation=(),
            market_state_hash="m", approved_intent_hash="a",
            session_phase="weekend", hours_to_discovery=30.0, decided_at=T0,
        )
        led.settle(1, exit_price=Decimal("230"))
        lines = path.read_text(encoding="utf-8").splitlines()
        seal_idx = next(i for i, ln in enumerate(lines) if json.loads(ln)["kind"] == "settlement_seal")
        seal = json.loads(lines[seal_idx])
        seal["settlement_seal_hash"] = "0" * 16
        lines[seal_idx] = json.dumps(seal, separators=(",", ":"))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        with pytest.raises(MigrationError, match="already fail"):
            migrate(path)


class TestRender:
    def test_render_names_the_honest_limitation_when_something_was_backfilled(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "p.jsonl"
        _pre_seal_ledger(path, n_settled=1, n_open=0)
        text = migrate(path).render()
        assert "not since the decision's real settlement date" in text

    def test_render_is_quiet_about_the_limitation_when_nothing_was_backfilled(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "p.jsonl"
        led = PaperLedger(path=path)
        led.record(
            symbol="NVDAUSDT", verdict="trade", side="BUY",
            quantity=Decimal("10"), entry_price=Decimal("220"),
            stated_confidence=0.7, thesis="t", invalidation=(),
            market_state_hash="m", approved_intent_hash="a",
            session_phase="weekend", hours_to_discovery=30.0, decided_at=T0,
        )
        text = migrate(path).render()
        assert "not since the decision's real settlement date" not in text


class TestAsDict:
    def test_as_dict_round_trips_the_key_fields(self, tmp_path: Path) -> None:
        path = tmp_path / "p.jsonl"
        _pre_seal_ledger(path, n_settled=1, n_open=0)
        result = migrate(path)
        blob = result.as_dict()
        assert blob["backfilled"] == [1]
        assert blob["already_sealed"] == 0
        assert isinstance(blob["backup"], str)
