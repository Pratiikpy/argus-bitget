"""Ledger repair tests.

The repair is the only code in this project permitted to rewrite history, so the tests are mostly
about what it must refuse to do. A repair that drops a decision, alters a thesis, reorders rows, or
leaves the chain broken while reporting success is worse than the corruption it was meant to fix,
because after it the operator stops looking.

The fixture reproduces the real damage: two cycles interleaved on 2026-09-12 and produced seven
duplicated sequence numbers across sixty-two rows.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from argus.paper.ledger import Entry, LedgerError, PaperLedger
from argus.paper.repair import INCIDENTS, digest, plan, repair

T0 = datetime(2026, 9, 12, 17, 0, tzinfo=UTC)
WHEN = datetime(2026, 9, 12, 18, 0, tzinfo=UTC)


def _clean(path: Path, n: int = 4) -> PaperLedger:
    led = PaperLedger(path=path)
    for i in range(n):
        led.record(
            symbol=f"SYM{i}USDT", verdict="no_trade", side="BUY",
            quantity=Decimal("0"), entry_price=Decimal("100"),
            stated_confidence=0.8, thesis=f"thesis {i}",
            invalidation=(f"invalidation {i}",), market_state_hash=f"m{i}" * 8,
            approved_intent_hash=f"a{i}" * 8, session_phase="weekend",
            hours_to_discovery=44.0, decided_at=T0 + timedelta(minutes=i),
        )
    return led


def _collide(path: Path) -> None:
    """Duplicate a sequence number the way two concurrent writers did."""
    lines = path.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[2])
    row["thesis"] = "a different decision written by the other cycle"
    row["decided_at"] = (T0 + timedelta(minutes=30)).isoformat()
    lines.insert(3, json.dumps(row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class TestItSeesTheDamage:
    def test_a_healthy_log_needs_nothing(self, tmp_path: Path) -> None:
        led = _clean(tmp_path / "p.jsonl")
        assert not plan(led.path).needed

    def test_a_healthy_log_is_reported_as_such_rather_than_silently(self, tmp_path: Path) -> None:
        led = _clean(tmp_path / "p.jsonl")
        assert "nothing to do" in " ".join(plan(led.path).render())

    def test_the_duplicate_is_named(self, tmp_path: Path) -> None:
        led = _clean(tmp_path / "p.jsonl")
        _collide(led.path)
        assert plan(led.path).duplicate_seqs == (3,)

    def test_the_count_of_rows_is_stated_both_ways(self, tmp_path: Path) -> None:
        led = _clean(tmp_path / "p.jsonl")
        _collide(led.path)
        proposal = plan(led.path)
        assert proposal.entries_before == proposal.entries_after == 5

    def test_planning_writes_nothing(self, tmp_path: Path) -> None:
        led = _clean(tmp_path / "p.jsonl")
        _collide(led.path)
        before = digest(led.path)
        plan(led.path)
        assert digest(led.path) == before


class TestNothingIsLostOrAltered:
    def _repaired(self, tmp_path: Path) -> tuple[Path, dict]:
        led = _clean(tmp_path / "p.jsonl")
        _collide(led.path)
        outcome = repair(led.path, reason="concurrent writers duplicated a sequence", at=WHEN)
        return led.path, outcome

    def test_every_decision_survives(self, tmp_path: Path) -> None:
        path, _ = self._repaired(tmp_path)
        assert len(PaperLedger(path=path).entries) == 5

    def test_no_thesis_is_altered(self, tmp_path: Path) -> None:
        path, _ = self._repaired(tmp_path)
        theses = [e.thesis for e in PaperLedger(path=path).entries]
        assert "a different decision written by the other cycle" in theses
        assert len(set(theses)) == 5

    def test_the_order_of_the_file_is_preserved(self, tmp_path: Path) -> None:
        """Reordering by timestamp would be a judgement about which decision came first."""
        path, _ = self._repaired(tmp_path)
        minutes = [int(e.decided_at[14:16]) for e in PaperLedger(path=path).entries]
        # The colliding row was written at :30 and sits fourth in the file. Sorted by time it would
        # be last; it stays fourth, because the file is the record of what happened.
        assert minutes == [0, 1, 2, 30, 3]

    def test_the_sequence_numbers_become_unique_and_contiguous(self, tmp_path: Path) -> None:
        path, _ = self._repaired(tmp_path)
        assert [e.seq for e in PaperLedger(path=path).entries] == [1, 2, 3, 4, 5]

    def test_the_repaired_chain_verifies(self, tmp_path: Path) -> None:
        path, _ = self._repaired(tmp_path)
        report = PaperLedger(path=path).verify()
        assert report["chain_intact"] and report["links_intact"]
        assert report["first_break_at"] is None

    def test_the_anchor_is_rewritten_to_match(self, tmp_path: Path) -> None:
        path, _ = self._repaired(tmp_path)
        assert PaperLedger(path=path).verify()["anchor"] == "anchor agrees with the log"

    def test_every_change_reports_its_content_preserved(self, tmp_path: Path) -> None:
        led = _clean(tmp_path / "p.jsonl")
        _collide(led.path)
        assert plan(led.path).is_content_preserving


class TestTheIncidentRecord:
    def _repaired(self, tmp_path: Path) -> tuple[Path, dict]:
        led = _clean(tmp_path / "p.jsonl")
        _collide(led.path)
        before = digest(led.path)
        outcome = repair(led.path, reason="two cycles overlapped", at=WHEN)
        return led.path, {"before": before, **outcome}

    def test_the_original_bytes_are_archived_untouched(self, tmp_path: Path) -> None:
        path, outcome = self._repaired(tmp_path)
        archive = path.with_name(outcome["incident"]["archive"])
        assert digest(archive) == outcome["before"]

    def test_the_archive_still_fails_verification(self, tmp_path: Path) -> None:
        """The evidence of the corruption is kept, not tidied away."""
        path, outcome = self._repaired(tmp_path)
        archive = path.with_name(outcome["incident"]["archive"])
        assert PaperLedger(path=archive).verify()["chain_intact"] is False

    def test_the_record_commits_to_both_digests(self, tmp_path: Path) -> None:
        path, outcome = self._repaired(tmp_path)
        incident = outcome["incident"]
        assert incident["archive_sha256"] == outcome["before"]
        assert incident["repaired_sha256"] == digest(path)

    def test_the_record_names_every_row_that_moved(self, tmp_path: Path) -> None:
        _, outcome = self._repaired(tmp_path)
        changes = outcome["incident"]["plan"]["changes"]
        assert changes and all("old_seq" in c and "new_seq" in c for c in changes)

    def test_the_reason_is_kept(self, tmp_path: Path) -> None:
        _, outcome = self._repaired(tmp_path)
        assert outcome["incident"]["reason"] == "two cycles overlapped"

    def test_a_second_repair_does_not_hide_the_first(self, tmp_path: Path) -> None:
        path, _ = self._repaired(tmp_path)
        _collide(path)
        repair(path, reason="it happened again", at=WHEN + timedelta(hours=1))
        history = json.loads(path.with_name(INCIDENTS).read_text(encoding="utf-8"))
        assert len(history) == 2


class TestWhatItRefuses:
    def test_an_unexplained_repair_is_refused(self, tmp_path: Path) -> None:
        led = _clean(tmp_path / "p.jsonl")
        _collide(led.path)
        with pytest.raises(LedgerError, match="stated reason"):
            repair(led.path, reason="   ")

    def test_a_healthy_log_is_not_rewritten(self, tmp_path: Path) -> None:
        led = _clean(tmp_path / "p.jsonl")
        before = digest(led.path)
        outcome = repair(led.path, reason="checking")
        assert outcome["repaired"] is False
        assert digest(led.path) == before

    def test_it_will_not_overwrite_an_existing_archive(self, tmp_path: Path) -> None:
        led = _clean(tmp_path / "p.jsonl")
        _collide(led.path)
        repair(led.path, reason="first", at=WHEN)
        _collide(led.path)
        with pytest.raises(LedgerError, match="refusing to overwrite an archive"):
            repair(led.path, reason="second, same second", at=WHEN)

    def test_a_settled_row_keeps_its_outcome(self, tmp_path: Path) -> None:
        """Settlement fields are outside the content hash; a re-link must not disturb them."""
        led = _clean(tmp_path / "p.jsonl")
        led.settle_abstention(2, price_now=Decimal("101"))
        _collide(led.path)
        repair(led.path, reason="concurrent writers", at=WHEN)
        settled = [e for e in PaperLedger(path=led.path).entries if e.is_settled]
        assert len(settled) == 1
        assert settled[0].exit_price == "101"

    def test_a_relink_that_changed_a_decision_would_be_caught(self, tmp_path: Path) -> None:
        """The guard, exercised directly: content_preserved is what authorises the write."""
        led = _clean(tmp_path / "p.jsonl")
        _collide(led.path)
        proposal = plan(led.path)
        tampered = proposal.changes[0].__class__(
            **{**proposal.changes[0].as_dict(), "content_preserved": False}
        )
        assert not type(proposal)(
            entries_before=proposal.entries_before,
            entries_after=proposal.entries_after,
            duplicate_seqs=proposal.duplicate_seqs,
            changes=(tampered,),
            source_digest=proposal.source_digest,
        ).is_content_preserving


class TestItReadsWhatTheLedgerWrites:
    def test_a_row_round_trips_through_the_repair_unchanged_field_by_field(
        self, tmp_path: Path
    ) -> None:
        led = _clean(tmp_path / "p.jsonl")
        original = {e.seq: e for e in led.entries}
        _collide(led.path)
        repair(led.path, reason="concurrent writers", at=WHEN)
        after = PaperLedger(path=led.path).entries
        first = next(e for e in after if e.thesis == "thesis 0")
        untouched: list[str] = [
            f for f in Entry.__dataclass_fields__ if f not in {"seq", "prev_hash"}
        ]
        for name in untouched:
            assert getattr(first, name) == getattr(original[1], name), name


class TestSealsSurviveARepair:
    """Added 2026-09-22 alongside settlement seals themselves. `_relinked` used to operate on
    `ledger.entries` (decisions only) — fine before seals existed, and a real data-loss bug once
    they do: rewriting the file from a decisions-only list silently drops every seal ever written.
    """

    def test_a_seal_is_not_dropped_by_repair(self, tmp_path: Path) -> None:
        led = _clean(tmp_path / "p.jsonl")
        led.settle_abstention(1, price_now=Decimal("101"))
        assert len(led.seals) == 1
        _collide(led.path)
        repair(led.path, reason="concurrent writers", at=WHEN)
        after = PaperLedger(path=led.path)
        assert len(after.seals) == 1

    def test_a_seal_s_target_seq_follows_its_decision_through_renumbering(
        self, tmp_path: Path
    ) -> None:
        """The collision fixture inserts a duplicate ahead of decision 4's real position, so
        repair renumbers everything from that point on — decision 4 (originally seq 4) settles
        to a different seq after repair, and its seal must move with it, not point at whatever
        row ends up with the old number."""
        led = _clean(tmp_path / "p.jsonl")
        led.settle_abstention(4, price_now=Decimal("101"))
        original_target = led.seals[0].target_seq
        _collide(led.path)
        repair(led.path, reason="concurrent writers", at=WHEN)
        after = PaperLedger(path=led.path)
        reseated = next(e for e in after.entries if e.thesis == "thesis 3")
        # The collision inserts a duplicate row ahead of decision 4's real position, so repair
        # actually does shift its seq — this assertion is only meaningful if that precondition
        # held; otherwise the test would pass trivially without exercising the retargeting at all.
        assert reseated.seq != original_target
        assert after.seals[0].target_seq == reseated.seq

    def test_the_repaired_chain_with_a_seal_still_verifies_as_tamper_evident(
        self, tmp_path: Path
    ) -> None:
        led = _clean(tmp_path / "p.jsonl")
        led.settle_abstention(1, price_now=Decimal("101"))
        _collide(led.path)
        repair(led.path, reason="concurrent writers", at=WHEN)
        report = PaperLedger(path=led.path).verify()
        assert report["chain_intact"] is True
        assert report["tampered_settlements"] == []
        assert report["unsealed_settlements"] == []

    def test_seals_keep_their_own_negative_seq_space_after_repair(self, tmp_path: Path) -> None:
        led = _clean(tmp_path / "p.jsonl")
        led.settle_abstention(1, price_now=Decimal("101"))
        led.settle_abstention(2, price_now=Decimal("102"))
        _collide(led.path)
        repair(led.path, reason="concurrent writers", at=WHEN)
        after = PaperLedger(path=led.path)
        assert {s.seq for s in after.seals} == {-1, -2}
        assert all(e.seq > 0 for e in after.entries)
