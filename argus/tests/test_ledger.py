"""Paper-trading ledger tests.

Fabricated or backfilled paper trading is an anti-pattern we named and banned. These tests assert
the structural defences actually hold, because "trust our log" is not a submission.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from argus.paper.ledger import GENESIS, Entry, LedgerError, PaperLedger

T0 = datetime(2026, 3, 9, 14, 0, tzinfo=UTC)


def _ledger(tmp_path: Path) -> PaperLedger:
    return PaperLedger(path=tmp_path / "paper.jsonl")


def _record(led: PaperLedger, *, qty: str = "10", price: str = "220", n: int = 0) -> Entry:
    return led.record(
        symbol="NVDAUSDT", verdict="trade", side="BUY",
        quantity=Decimal(qty), entry_price=Decimal(price),
        stated_confidence=0.7, thesis="guidance cut not priced",
        invalidation=("guidance reaffirmed",),
        market_state_hash="abc123", approved_intent_hash="def456",
        session_phase="weekend", hours_to_discovery=30.5,
        decided_at=T0 + timedelta(hours=n),
    )


class TestHashChain:
    def test_first_entry_links_to_genesis(self, tmp_path: Path) -> None:
        led = _ledger(tmp_path)
        assert _record(led).prev_hash == GENESIS

    def test_each_entry_links_to_the_last(self, tmp_path: Path) -> None:
        led = _ledger(tmp_path)
        a = _record(led, n=0)
        b = _record(led, n=1)
        assert b.prev_hash == a.content_hash
        assert led.verify()["chain_intact"] is True

    def test_editing_history_breaks_the_chain_at_that_point(self, tmp_path: Path) -> None:
        """A doctored log must fail its own verification."""
        led = _ledger(tmp_path)
        for i in range(4):
            _record(led, n=i)
        assert led.verify()["chain_intact"] is True

        # Tamper: rewrite a past decision's thesis on disk.
        raw = led.path.read_text(encoding="utf-8").splitlines()
        rows = [json.loads(x) for x in raw if x.strip()]
        rows[1]["thesis"] = "a story invented after the fact"
        led.path.write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
        )

        reloaded = PaperLedger(path=led.path)
        got = reloaded.verify()
        assert got["chain_intact"] is False
        assert got["first_break_at"] == 3  # the entry AFTER the edited one

    def test_a_reloaded_untampered_ledger_verifies(self, tmp_path: Path) -> None:
        led = _ledger(tmp_path)
        for i in range(3):
            _record(led, n=i)
        assert PaperLedger(path=led.path).verify()["chain_intact"] is True


class TestOutcomesAreWrittenOnce:
    def test_a_decision_is_recorded_without_an_outcome(self, tmp_path: Path) -> None:
        """There is no code path that writes a decision and its result together."""
        e = _record(_ledger(tmp_path))
        assert e.is_settled is False
        assert e.net_pnl is None
        assert e.direction_correct is None

    def test_settling_twice_is_refused(self, tmp_path: Path) -> None:
        led = _ledger(tmp_path)
        _record(led)
        led.settle(1, exit_price=Decimal("230"), settled_at=T0 + timedelta(days=1))
        with pytest.raises(LedgerError, match="written once"):
            led.settle(1, exit_price=Decimal("300"))

    def test_settling_an_unknown_entry_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(LedgerError, match="no entry"):
            _ledger(tmp_path).settle(99, exit_price=Decimal("1"))

    def test_settlement_does_not_change_the_decision_s_own_hash(self, tmp_path: Path) -> None:
        """Outcomes sit outside `content_hash`, so attaching one is distinguishable from tampering
        with the decision itself — the settled row's own `content_hash` is unchanged."""
        led = _ledger(tmp_path)
        _record(led, n=0)
        _record(led, n=1)
        before_hash = led.entries[-1].content_hash
        led.settle(1, exit_price=Decimal("230"))
        assert led.entries[-1].content_hash == before_hash
        assert led.verify()["chain_intact"] is True

    def test_settlement_extends_the_chain_with_a_seal_rather_than_leaving_it_unchanged(
        self, tmp_path: Path
    ) -> None:
        """**Updated 2026-09-22.** This used to assert `head_hash` was byte-identical before and
        after settling — true before the settlement-seal fix, and it was true for the wrong
        reason: nothing committed the outcome into the chain at all, which is the real gap
        `_check_settlement_seals` exists to close. `head_hash` moving forward here is the fix
        working, not a regression to guard against."""
        led = _ledger(tmp_path)
        _record(led, n=0)
        _record(led, n=1)
        before = led.verify()["head_hash"]
        led.settle(1, exit_price=Decimal("230"))
        after = led.verify()
        assert after["head_hash"] != before
        assert after["chain_intact"] is True
        assert len(led.seals) == 1
        assert led.seals[0].target_seq == 1


class TestCosts:
    def test_entry_cost_is_charged_at_decision_time(self, tmp_path: Path) -> None:
        e = _record(_ledger(tmp_path))
        assert Decimal(e.entry_cost_bps) > 0

    def test_net_is_below_gross_by_the_round_trip(self, tmp_path: Path) -> None:
        led = _ledger(tmp_path)
        _record(led, qty="10", price="220")
        got = led.settle(1, exit_price=Decimal("230"))
        assert Decimal(got.net_pnl or "0") < Decimal(got.gross_pnl or "0")

    def test_a_small_winner_can_be_a_net_loser(self, tmp_path: Path) -> None:
        """The measured reality on this venue: a 12bps round trip eats most small moves."""
        led = _ledger(tmp_path)
        _record(led, qty="10", price="220")
        got = led.settle(1, exit_price=Decimal("220.10"))   # ~4.5bps move
        assert Decimal(got.gross_pnl or "0") > 0
        assert Decimal(got.net_pnl or "0") < 0

    def test_performance_reports_when_cost_flipped_the_sign(self, tmp_path: Path) -> None:
        led = _ledger(tmp_path)
        _record(led, qty="10", price="220")
        led.settle(1, exit_price=Decimal("220.10"))
        assert led.performance()["cost_flipped_the_sign"] is True


class TestAbstentions:
    def test_abstentions_are_counted_separately_from_trades(self, tmp_path: Path) -> None:
        """A NO_TRADE is a decision, not a missing row — the log must show the agent chose."""
        led = _ledger(tmp_path)
        led.record(
            symbol="NVDAUSDT", verdict="no_trade", side="BUY",
            quantity=Decimal("0"), entry_price=Decimal("220"),
            stated_confidence=0.9, thesis="edge does not clear 12bps",
            invalidation=(), market_state_hash="x", approved_intent_hash="y",
            session_phase="weekend", hours_to_discovery=30.5, decided_at=T0,
        )
        _record(led, n=1)
        perf = led.performance()
        assert perf["abstentions"] == 1
        assert perf["decisions"] == 2

    def test_an_empty_ledger_says_so_rather_than_reporting_zeroes(self, tmp_path: Path) -> None:
        assert "nothing to report" in _ledger(tmp_path).performance()["note"]


class TestAbstentionsAreGradeable:
    """An abstention used to be permanently ungradeable.

    `_settle_due` skipped `is_abstention` outright, so a `no_trade` never received an outcome — and
    a desk that abstained correctly looked identical to one paralysed by its own hurdle. Every
    entry in the live ledger was in that state.
    """

    @staticmethod
    def _ledger(tmp_path):  # type: ignore[no-untyped-def]
        from argus.paper.ledger import PaperLedger

        return PaperLedger(path=tmp_path / "ledger.jsonl")

    @staticmethod
    def _abstain(ledger, symbol: str = "rNVDA"):  # type: ignore[no-untyped-def]
        # A lean is recorded because the scorecard grades a refusal against the trade it withheld
        # — its lean — and a refusal that states none is counted, not graded (2026-09-25).
        return ledger.record(
            symbol=symbol, verdict="no_trade", side="BUY", quantity=Decimal("0"),
            entry_price=Decimal("100"), stated_confidence=0.9, thesis="edge below the hurdle",
            invalidation=("a catalyst appears",), market_state_hash="h", approved_intent_hash="i",
            session_phase="weekend", hours_to_discovery=30.5, lean="up", lean_confidence=0.6,
        )

    def test_an_abstention_records_what_the_market_did(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        ledger = self._ledger(tmp_path)
        entry = self._abstain(ledger)
        assert entry.counterfactual_move_bps is None

        settled = ledger.settle_abstention(entry.seq, price_now=Decimal("101"))
        assert settled.is_settled
        assert settled.counterfactual_move_bps == "100.0000"

    def test_the_move_is_signed_as_it_happened_not_as_a_verdict(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """Pre-judging whether abstaining was right would bake the answer into the data."""
        ledger = self._ledger(tmp_path)
        down = ledger.settle_abstention(self._abstain(ledger).seq, price_now=Decimal("99"))
        assert down.counterfactual_move_bps.startswith("-")

    def test_a_position_cannot_be_settled_as_an_abstention(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        ledger = self._ledger(tmp_path)
        traded = ledger.record(
            symbol="rNVDA", verdict="trade", side="BUY", quantity=Decimal("10"),
            entry_price=Decimal("100"), stated_confidence=0.7, thesis="t",
            invalidation=("x",), market_state_hash="h", approved_intent_hash="i",
            session_phase="rth", hours_to_discovery=0.0,
        )
        with pytest.raises(LedgerError, match="carries a position"):
            ledger.settle_abstention(traded.seq, price_now=Decimal("101"))

    def test_an_abstention_settles_once(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        ledger = self._ledger(tmp_path)
        entry = self._abstain(ledger)
        ledger.settle_abstention(entry.seq, price_now=Decimal("101"))
        with pytest.raises(LedgerError, match="written once"):
            ledger.settle_abstention(entry.seq, price_now=Decimal("105"))

    def test_settling_does_not_break_the_chain(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """Outcome fields are not hashed, so attaching one cannot rewrite history."""
        ledger = self._ledger(tmp_path)
        for _ in range(3):
            self._abstain(ledger)
        ledger.settle_abstention(2, price_now=Decimal("102"))
        assert ledger.verify()["chain_intact"] is True

    def test_the_scorecard_now_grades_it(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """The point of the whole change: abstentions reach the Observatory."""
        from argus.eval.scorecard import score_ledger

        ledger = self._ledger(tmp_path)
        for _ in range(3):
            self._abstain(ledger)
        for seq, px in ((1, Decimal("101")), (2, Decimal("99")), (3, Decimal("100.02"))):
            ledger.settle_abstention(seq, price_now=px)

        score = score_ledger(ledger)
        assert len(score.abstention_outcomes) == 3
        assert score.ungradeable == 0

    def test_an_unsettled_abstention_stays_ungradeable(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """The honest default is preserved: no counterfactual means no grade, not a guess."""
        from argus.eval.scorecard import score_ledger

        ledger = self._ledger(tmp_path)
        self._abstain(ledger)
        score = score_ledger(ledger)
        assert score.abstention_outcomes == []
        assert score.ungradeable == 1

    def test_an_explicit_counterfactual_overrides_the_recorded_one(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """For replaying a scorecard against a corrected price history."""
        from argus.eval.scorecard import score_ledger

        ledger = self._ledger(tmp_path)
        entry = self._abstain(ledger)
        ledger.settle_abstention(entry.seq, price_now=Decimal("101"))
        score = score_ledger(ledger, counterfactuals={entry.seq: Decimal("-250")})
        assert score.abstention_outcomes[0].counterfactual_move_bps == Decimal("-250")


class TestTruncationIsDetected:
    """A hash chain proves what is present is unedited. It says nothing about what was removed.

    Verified against the real ledger before the anchor existed: deleting the last seven entries
    left ``chain_intact: True``. Since the rows most worth deleting are the most recent losses,
    this was the most valuable gap in the record's integrity story.
    """

    def _filled(self, tmp_path: Path) -> PaperLedger:
        led = PaperLedger(path=tmp_path / "anchor.jsonl")
        for _ in range(6):
            led.record(
                symbol="NVDAUSDT", verdict="no_trade", side="BUY", quantity=Decimal("0"),
                entry_price=Decimal("100"), stated_confidence=0.8, thesis="fixture",
                invalidation=(), market_state_hash="m" * 16, approved_intent_hash="a" * 16,
                session_phase="rth", hours_to_discovery=0.0,
            )
        return led

    def test_an_untouched_log_agrees_with_its_anchor(self, tmp_path: Path) -> None:
        report = self._filled(tmp_path).verify()
        assert report["chain_intact"] is True
        assert report["truncated"] is False
        assert report["anchor"] == "anchor agrees with the log"

    def test_removing_rows_from_the_end_is_caught(self, tmp_path: Path) -> None:
        led = self._filled(tmp_path)
        rows = led.path.read_text(encoding="utf-8").splitlines()
        led.path.write_text("\n".join(rows[:3]) + "\n", encoding="utf-8")

        report = PaperLedger(path=led.path).verify()
        assert report["chain_intact"] is False
        assert report["truncated"] is True
        assert report["links_intact"] is True, "the surviving links are valid; that is the point"
        assert "3 row(s) removed" in report["anchor"]

    def test_the_anchor_is_refreshed_by_settlement(self, tmp_path: Path) -> None:
        """Settlement rewrites the file, so an anchor left stale would read as truncation."""
        led = PaperLedger(path=tmp_path / "settled.jsonl")
        entry = led.record(
            symbol="NVDAUSDT", verdict="open_long", side="BUY", quantity=Decimal("1"),
            entry_price=Decimal("100"), stated_confidence=0.7, thesis="fixture",
            invalidation=(), market_state_hash="m" * 16, approved_intent_hash="a" * 16,
            session_phase="rth", hours_to_discovery=0.0,
        )
        led.settle(entry.seq, exit_price=Decimal("110"))
        assert led.verify()["chain_intact"] is True

    def test_a_log_with_no_anchor_says_so_rather_than_claiming_safety(
        self, tmp_path: Path
    ) -> None:
        """An absent anchor is an unknown, not a pass."""
        led = self._filled(tmp_path)
        led._anchor_path.unlink()
        report = PaperLedger(path=led.path).verify()
        assert "cannot be ruled out" in report["anchor"]


class TestConcurrentWriters:
    """Reproduces the corruption that happened to the live log, and proves it cannot recur.

    On 2026-09-12 the scheduled cycle and a manual one overlapped. Each had loaded the ledger when
    it ended at seq 40; each computed ``seq = len(entries) + 1`` from that stale head; each appended
    a different NVDAUSDT decision as seq 41, fifty-eight seconds apart. ``verify()`` caught it
    (``first_break_at: 41``) — the detection worked and the prevention did not exist.
    """

    def test_two_writers_holding_a_stale_head_do_not_both_write_the_same_seq(
        self, tmp_path: Path
    ) -> None:
        """The exact shape of the live failure: two objects, both loaded before either wrote."""
        path = tmp_path / "paper.jsonl"
        first, second = PaperLedger(path=path), PaperLedger(path=path)
        _record(first, n=1)
        _record(second, n=2)  # second still believes the log is empty

        seqs = [e.seq for e in PaperLedger(path=path).entries]
        assert seqs == [1, 2], f"expected a re-read head to renumber, got {seqs}"

    def test_the_chain_still_verifies_after_a_racing_write(self, tmp_path: Path) -> None:
        path = tmp_path / "paper.jsonl"
        first, second = PaperLedger(path=path), PaperLedger(path=path)
        _record(first, n=1)
        _record(second, n=2)
        report = PaperLedger(path=path).verify()
        assert report["chain_intact"] is True
        assert report["links_intact"] is True
        assert report["first_break_at"] is None

    def test_both_decisions_survive(self, tmp_path: Path) -> None:
        """Renumbering must not be a euphemism for dropping one."""
        path = tmp_path / "paper.jsonl"
        first, second = PaperLedger(path=path), PaperLedger(path=path)
        a = _record(first, n=1)
        b = _record(second, n=2)
        decided = {e.decided_at for e in PaperLedger(path=path).entries}
        assert decided == {a.decided_at, b.decided_at}

    def test_a_settlement_does_not_erase_a_concurrent_decision(self, tmp_path: Path) -> None:
        """A rewrite from a stale in-memory list would delete the other writer's row entirely."""
        path = tmp_path / "paper.jsonl"
        first, second = PaperLedger(path=path), PaperLedger(path=path)
        entry = _record(first, n=1)
        second._load()
        _record(second, n=2)
        first.settle(entry.seq, exit_price=Decimal("230"))  # `first` has never seen seq 2
        assert len(PaperLedger(path=path).entries) == 2

    def test_the_lock_is_released_when_a_write_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "paper.jsonl"
        led = _ledger(tmp_path)
        with pytest.raises(LedgerError):
            led.settle(99, exit_price=Decimal("1"))
        assert not path.with_suffix(path.suffix + ".lock").exists()
        _record(led)  # and the ledger is still usable

    def test_a_duplicate_seq_in_the_file_is_refused_rather_than_settled(
        self, tmp_path: Path
    ) -> None:
        """What the live log looked like. Settling positionally would have picked one at random."""
        path = tmp_path / "paper.jsonl"
        led = _ledger(tmp_path)
        _record(led, n=1)
        line = path.read_text(encoding="utf-8").splitlines()[0]
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        with pytest.raises(LedgerError, match="appears 2 times"):
            PaperLedger(path=path).settle(1, exit_price=Decimal("230"))


class TestTheLeanIsOnTheRecord:
    """The desk's directional view, written at decision time and never after it.

    Recorded because the `side` field turned out to carry nothing: BUY on all 53 settled
    abstentions, right 3.8% of the time against a 3.8% base rate. `argus.eval.shadow` grades this
    field instead.
    """

    def _lean(self, led: PaperLedger, *, lean: str, confidence: float, n: int = 0) -> Entry:
        return led.record(
            symbol="NVDAUSDT", verdict="no_trade", side="BUY",
            quantity=Decimal("0"), entry_price=Decimal("220"),
            stated_confidence=0.4, thesis="edge does not clear the hurdle",
            invalidation=("guidance reaffirmed",),
            market_state_hash="abc123", approved_intent_hash="def456",
            session_phase="weekend", hours_to_discovery=30.5,
            decided_at=T0 + timedelta(hours=n),
            lean=lean, lean_confidence=confidence,
        )

    def test_a_decision_records_the_lean_it_was_given(self, tmp_path: Path) -> None:
        got = self._lean(_ledger(tmp_path), lean="down", confidence=0.62)
        assert got.lean == "down"
        assert got.lean_confidence == 0.62

    def test_a_decision_with_no_lean_records_none_rather_than_a_direction(
        self, tmp_path: Path
    ) -> None:
        assert _record(_ledger(tmp_path)).lean == "none"

    def test_the_lean_survives_a_reload_from_disk(self, tmp_path: Path) -> None:
        led = _ledger(tmp_path)
        self._lean(led, lean="up", confidence=0.7)
        assert PaperLedger(path=led.path).entries[0].lean == "up"

    def test_a_row_written_before_the_field_existed_still_loads(self, tmp_path: Path) -> None:
        """The 161 rows already on the anchored chain have no lean key at all. If this raised, the
        whole history would be unreadable and the chain unverifiable."""
        led = _ledger(tmp_path)
        _record(led)
        raw = led.path.read_text(encoding="utf-8").splitlines()
        rows = [json.loads(x) for x in raw if x.strip()]
        for row in rows:
            row.pop("lean", None)
            row.pop("lean_confidence", None)
        led.path.write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
        )
        reloaded = PaperLedger(path=led.path)
        assert reloaded.entries[0].lean == "none"
        assert reloaded.entries[0].lean_confidence == 0.0

    def test_the_lean_is_outside_the_row_payload_so_the_anchor_survived(
        self, tmp_path: Path
    ) -> None:
        """Two rows identical but for the lean hash the same.

        That is the property that let the field be added at all: the live chain is anchored to
        Bitcoin at 161 rows, and putting the lean in `content_hash` would have rehashed every one of
        them and invalidated the anchor. Tamper-evidence comes from `approved_intent_hash` instead,
        which is in the payload — the test above proves that half.
        """
        led = _ledger(tmp_path)
        plain = self._lean(led, lean="none", confidence=0.0)
        doctored = Entry(**{**asdict(plain), "invalidation": tuple(plain.invalidation),
                            "lean": "up", "lean_confidence": 0.9})
        assert doctored.content_hash == plain.content_hash

    def test_a_lean_edited_on_disk_is_caught_by_the_intent_hash(self, tmp_path: Path) -> None:
        """The protection the docstring claims, tested rather than asserted. The lean is inside the
        Intent, the Intent's hash is in the row payload, so a lean rewritten after the move is known
        no longer agrees with the hash the row committed to."""
        from argus.decision.verdicts import Intent, Side, Verdict
        from argus.proof.autonomy import hash_intent

        intent = Intent(
            symbol="NVDAUSDT", side=Side.BUY, quantity=Decimal("0"), verdict=Verdict.NO_TRADE,
            stated_confidence=0.4, thesis="edge does not clear the hurdle",
            invalidation=("guidance reaffirmed",), lean="up", lean_confidence=0.7,
        )
        led = _ledger(tmp_path)
        entry = led.record(
            symbol="NVDAUSDT", verdict="no_trade", side="BUY",
            quantity=Decimal("0"), entry_price=Decimal("220"),
            stated_confidence=0.4, thesis="edge does not clear the hurdle",
            invalidation=("guidance reaffirmed",),
            market_state_hash="abc123", approved_intent_hash=hash_intent(intent),
            session_phase="weekend", hours_to_discovery=30.5, decided_at=T0,
            lean="up", lean_confidence=0.7,
        )
        forged = Intent(
            symbol="NVDAUSDT", side=Side.BUY, quantity=Decimal("0"), verdict=Verdict.NO_TRADE,
            stated_confidence=0.4, thesis="edge does not clear the hurdle",
            invalidation=("guidance reaffirmed",), lean="down", lean_confidence=0.7,
        )
        assert hash_intent(forged) != entry.approved_intent_hash

    def test_settling_an_abstention_leaves_the_lean_untouched(self, tmp_path: Path) -> None:
        """The outcome may not edit the call. If settlement could rewrite the lean, every accuracy
        computed from it would be a tautology."""
        led = _ledger(tmp_path)
        entry = self._lean(led, lean="down", confidence=0.55)
        settled = led.settle_abstention(entry.seq, price_now=Decimal("225"))
        assert settled.lean == "down"
        assert settled.lean_confidence == 0.55
        assert led.verify()["chain_intact"] is True


class TestThesisLength:
    """Found by driving the deployed console as a real user, not by scanning the code: two of
    three reasoning blocks in one live answer cut off mid-word, and the raw ledger showed why —
    `thesis[:500]` with no ellipsis, no comment, silently discarding whatever came after. 350 of
    519 rows on the live record hit exactly 500 characters. The historical rows cannot be
    repaired (append-only, hash-chained), but nothing protected this field from happening again,
    which is the gap these tests close."""

    def test_a_long_thesis_is_kept_up_to_the_new_limit(self, tmp_path: Path) -> None:
        from argus.paper.ledger import MAX_THESIS_LENGTH

        led = _ledger(tmp_path)
        long_thesis = "reasoning " * 200  # 2,000 chars, comfortably past the old 500-char cap
        assert len(long_thesis) > 500
        entry = led.record(
            symbol="NVDAUSDT", verdict="trade", side="BUY",
            quantity=Decimal("10"), entry_price=Decimal("220"),
            stated_confidence=0.7, thesis=long_thesis,
            invalidation=("guidance reaffirmed",),
            market_state_hash="abc123", approved_intent_hash="def456",
            session_phase="weekend", hours_to_discovery=30.5, decided_at=T0,
        )
        assert entry.thesis == long_thesis[:MAX_THESIS_LENGTH]
        assert len(entry.thesis) > 500, "the exact regression this test exists to catch"

    def test_a_thesis_past_the_new_limit_is_still_bounded_not_unbounded(
        self, tmp_path: Path
    ) -> None:
        """No cap at all would let one pathological generation bloat every row after it in an
        append-only log forever. A generous, documented ceiling is the right trade — not none."""
        from argus.paper.ledger import MAX_THESIS_LENGTH

        led = _ledger(tmp_path)
        pathological = "x" * 50_000
        entry = led.record(
            symbol="NVDAUSDT", verdict="trade", side="BUY",
            quantity=Decimal("10"), entry_price=Decimal("220"),
            stated_confidence=0.7, thesis=pathological,
            invalidation=("guidance reaffirmed",),
            market_state_hash="abc123", approved_intent_hash="def456",
            session_phase="weekend", hours_to_discovery=30.5, decided_at=T0,
        )
        assert len(entry.thesis) == MAX_THESIS_LENGTH

    def test_a_short_thesis_is_never_padded_or_altered(self, tmp_path: Path) -> None:
        led = _ledger(tmp_path)
        entry = _record(led)
        assert entry.thesis == "guidance cut not priced"


class TestSettlementSeals:
    """The fix for a real, working exploit found by adversarial testing 2026-09-22: `content_hash`
    deliberately excludes settlement fields (`net_pnl`, `direction_correct`, ...) so that attaching
    an outcome is not indistinguishable from tampering — correct, and documented at
    `Entry.content_hash`. But nothing else protected those fields either. A copy of the live ledger
    was tampered with directly on disk — bypassing `settle()` entirely, exactly what a compromised
    host or a malicious insider with file access would do — flipping a settled decision's `net_pnl`
    sign, and `verify()` still reported `chain_intact: True`. These tests replay that attack and
    confirm it is now caught, then check the fix does not introduce a new failure mode of its own.
    """

    def test_tampering_with_a_settled_outcome_is_now_detected(self, tmp_path: Path) -> None:
        """The exact exploit: edit `net_pnl` directly in the file, bypassing `settle()`."""
        led = _ledger(tmp_path)
        _record(led, n=0)
        led.settle(1, exit_price=Decimal("230"))
        assert led.verify()["chain_intact"] is True  # sealed correctly first

        lines = led.path.read_text(encoding="utf-8").splitlines()
        row = json.loads(lines[0])
        assert row["net_pnl"] is not None
        row["net_pnl"] = str(-abs(float(row["net_pnl"])) - 1)  # a different, wrong value
        row["direction_correct"] = not row["direction_correct"]
        lines[0] = json.dumps(row, separators=(",", ":"))
        led.path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        tampered = PaperLedger(path=led.path)
        report = tampered.verify()
        assert report["chain_intact"] is False
        assert report["tampered_settlements"] == [1]

    def test_tampering_with_the_seal_itself_is_also_detected(self, tmp_path: Path) -> None:
        """The other side of the same forgery: leave the decision alone, edit the seal's committed
        hash instead so it matches whatever fake outcome an attacker wants believed."""
        led = _ledger(tmp_path)
        _record(led, n=0)
        led.settle(1, exit_price=Decimal("230"))

        lines = led.path.read_text(encoding="utf-8").splitlines()
        seal_idx = next(
            i for i, ln in enumerate(lines) if json.loads(ln)["kind"] == "settlement_seal"
        )
        seal = json.loads(lines[seal_idx])
        seal["settlement_seal_hash"] = "0" * 16  # a fabricated hash matching nothing real
        lines[seal_idx] = json.dumps(seal, separators=(",", ":"))
        led.path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        tampered = PaperLedger(path=led.path)
        report = tampered.verify()
        assert report["chain_intact"] is False
        assert report["tampered_settlements"] == [1]

    def test_an_unsealed_settlement_is_flagged_not_silently_trusted(self, tmp_path: Path) -> None:
        """A decision marked settled with no seal at all — the shape a settlement performed by
        directly editing the file (never calling `settle()`) would take."""
        led = _ledger(tmp_path)
        _record(led, n=0)
        lines = led.path.read_text(encoding="utf-8").splitlines()
        row = json.loads(lines[0])
        row["settled_at"] = T0.isoformat()
        row["net_pnl"] = "500.00"
        row["direction_correct"] = True
        lines[0] = json.dumps(row, separators=(",", ":"))
        led.path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        bypassed = PaperLedger(path=led.path)
        report = bypassed.verify()
        assert report["chain_intact"] is False
        assert report["unsealed_settlements"] == [1]

    def test_a_correctly_settled_entry_passes_with_no_false_positive(self, tmp_path: Path) -> None:
        led = _ledger(tmp_path)
        _record(led, n=0)
        led.record(
            symbol="NVDAUSDT", verdict="no_trade", side="BUY",
            quantity=Decimal("0"), entry_price=Decimal("220"),
            stated_confidence=0.7, thesis="not enough edge",
            invalidation=(), market_state_hash="abc", approved_intent_hash="def",
            session_phase="weekend", hours_to_discovery=30.5, decided_at=T0 + timedelta(hours=1),
        )
        led.settle(1, exit_price=Decimal("230"))
        led.settle_abstention(2, price_now=Decimal("225"))
        report = led.verify()
        assert report["chain_intact"] is True
        assert report["tampered_settlements"] == []
        assert report["unsealed_settlements"] == []

    def test_seals_never_appear_in_entries(self, tmp_path: Path) -> None:
        """The property nine other modules read, assuming every row is a real decision — must
        never see a seal, or every one of them starts misreading the record."""
        led = _ledger(tmp_path)
        _record(led, n=0)
        led.settle(1, exit_price=Decimal("230"))
        assert len(led.entries) == 1
        assert all(e.kind == "decision" for e in led.entries)
        assert len(led.seals) == 1
        assert led.seals[0].kind == "settlement_seal"

    def test_settle_abstention_also_writes_a_seal(self, tmp_path: Path) -> None:
        led = _ledger(tmp_path)
        led.record(
            symbol="NVDAUSDT", verdict="no_trade", side="BUY",
            quantity=Decimal("0"), entry_price=Decimal("220"),
            stated_confidence=0.7, thesis="not enough edge",
            invalidation=(), market_state_hash="abc", approved_intent_hash="def",
            session_phase="weekend", hours_to_discovery=30.5, decided_at=T0,
        )
        led.settle_abstention(1, price_now=Decimal("225"))
        assert len(led.seals) == 1
        assert led.seals[0].target_seq == 1

    def test_the_chain_walk_does_not_false_positive_with_a_seal_between_two_decisions(
        self, tmp_path: Path
    ) -> None:
        """The bug caught before it shipped: `verify()`'s core link-check used to walk `entries`
        (decisions only), which skips seals entirely — comparing decision N+1's `prev_hash` against
        decision N's `content_hash` while a seal actually sits between them on disk. That mismatch
        would have made `verify()` cry tampering on a chain nothing had touched."""
        led = _ledger(tmp_path)
        _record(led, n=0)
        led.settle(1, exit_price=Decimal("230"))  # writes a seal between decision 1 and decision 2
        _record(led, n=1)
        report = led.verify()
        assert report["chain_intact"] is True
        assert report["links_intact"] is True
        assert report["first_break_at"] is None

    def test_removing_a_trailing_seal_is_caught_as_truncation(self, tmp_path: Path) -> None:
        """Deleting only the seal off the end (leaving the decision it sealed untouched) must not
        look like a clean, unmodified log — the anchor's `raw_entries` count exists for this."""
        led = _ledger(tmp_path)
        _record(led, n=0)
        led.settle(1, exit_price=Decimal("230"))
        lines = led.path.read_text(encoding="utf-8").splitlines()
        assert json.loads(lines[-1])["kind"] == "settlement_seal"
        led.path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")

        truncated = PaperLedger(path=led.path)
        report = truncated.verify()
        assert report["truncated"] is True
        assert report["chain_intact"] is False

    def test_a_seal_s_own_content_hash_is_not_affected_by_the_unhashed_seal_field(
        self, tmp_path: Path
    ) -> None:
        """`Entry.content_hash`'s own docstring promises the ORIGINAL 12-key payload is untouched
        by this feature — checked directly rather than trusted, since a silent change here would
        rehash all 566 real decisions and break a chain already anchored to Bitcoin."""
        plain = Entry(
            seq=1, decided_at=T0.isoformat(), symbol="NVDAUSDT", verdict="trade", side="BUY",
            quantity="10", entry_price="220", stated_confidence=0.7, thesis="t",
            invalidation=(), market_state_hash="a", approved_intent_hash="b",
            session_phase="weekend", hours_to_discovery=1.0, entry_cost_bps="1",
            prev_hash=GENESIS,
        )
        payload = {
            "seq": plain.seq, "decided_at": plain.decided_at, "symbol": plain.symbol,
            "verdict": plain.verdict, "side": plain.side, "quantity": plain.quantity,
            "entry_price": plain.entry_price, "stated_confidence": plain.stated_confidence,
            "thesis": plain.thesis, "market_state_hash": plain.market_state_hash,
            "approved_intent_hash": plain.approved_intent_hash, "prev_hash": plain.prev_hash,
        }
        import hashlib

        expected = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:16]
        assert plain.content_hash == expected
