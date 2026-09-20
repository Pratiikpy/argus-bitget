"""Pre-registration: the protocol must bind, and must not be amendable in silence."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from argus.decision.verdicts import Verdict
from argus.market.bitget import RTOKEN_SYMBOLS
from argus.paper.protocol import (
    STANDING,
    Commitment,
    Protocol,
    ProtocolError,
    audit,
    build,
    commit,
    enforce,
    governing,
    load,
    verify,
)
from argus.truth.clocks import SessionPhase

AT = datetime(2026, 9, 13, 12, 0, tzinfo=UTC)


def _protocol(**over: Any) -> Protocol:
    return replace(STANDING, **over)


def _commitment(entries: int = 94, **over: Any) -> Commitment:
    return build(_protocol(**over), committed_at=AT, ledger_head="head0", ledger_entries=entries)


class FakeEntry:
    """The ledger fields an audit reads. Deliberately not a real Entry: the audit must work on
    anything with these attributes, so a test never has to build a chain to check a rule."""

    def __init__(self, seq: int, symbol: str = "NVDAUSDT", verdict: str = "trade",
                 quantity: str = "1", session_phase: str = "rth") -> None:
        self.seq = seq
        self.symbol = symbol
        self.verdict = verdict
        self.quantity = quantity
        self.session_phase = session_phase


class TestTheProtocolRefusesToBeUncheckable:
    def test_the_standing_protocol_validates(self) -> None:
        STANDING.validate()

    def test_an_empty_universe_forbids_nothing(self) -> None:
        with pytest.raises(ProtocolError, match="permits nothing"):
            _protocol(universe=()).validate()

    def test_a_universe_the_venue_does_not_list_is_refused(self) -> None:
        with pytest.raises(ProtocolError, match="does not list"):
            _protocol(universe=("NOTAREALTOKEN",)).validate()

    def test_a_session_that_is_not_a_phase_is_refused(self) -> None:
        with pytest.raises(ProtocolError, match="not session phases"):
            _protocol(tradeable_sessions=("lunchtime",)).validate()

    def test_every_session_needs_a_declared_reasoning_budget(self) -> None:
        with pytest.raises(ProtocolError, match="prices the hurdle"):
            _protocol(thinking_by_session=((str(SessionPhase.RTH), "low"),)).validate()

    def test_a_verdict_the_system_cannot_produce_is_refused(self) -> None:
        with pytest.raises(ProtocolError, match="not verdicts"):
            _protocol(allowed_verdicts=("moon",)).validate()

    def test_a_zero_size_ceiling_permits_no_trade_at_all(self) -> None:
        with pytest.raises(ProtocolError, match="must be positive"):
            _protocol(max_quantity="0").validate()

    def test_a_non_positive_hold_is_refused(self) -> None:
        with pytest.raises(ProtocolError, match="hold_hours"):
            _protocol(hold_hours=0).validate()

    def test_a_preregistration_without_a_prediction_is_a_description(self) -> None:
        with pytest.raises(ProtocolError, match="description"):
            _protocol(hypothesis="   ").validate()

    def test_version_starts_at_one(self) -> None:
        with pytest.raises(ProtocolError, match="version starts at 1"):
            _protocol(version=0).validate()


class TestTheDigestCoversWhatItClaims:
    def test_the_digest_is_stable_across_rebuilds(self) -> None:
        assert _protocol().digest == _protocol().digest

    def test_changing_a_parameter_changes_the_digest(self) -> None:
        assert _protocol(max_quantity="2").digest != STANDING.digest

    def test_changing_the_stated_reason_also_changes_the_digest(self) -> None:
        """A protocol whose reasons can be rewritten while its numbers hold is still retellable."""
        assert _protocol(hypothesis="something else entirely").digest != STANDING.digest

    def test_known_weaknesses_are_part_of_the_commitment(self) -> None:
        assert _protocol(known_weaknesses=()).digest != STANDING.digest

    def test_the_canonical_form_is_key_sorted_and_compact(self) -> None:
        """Compactness is about the separators, not about prose: several committed values contain
        ', ' inside their own text, so the check is on the emitted structure."""
        text = STANDING.canonical()
        top = json.loads(text)
        assert list(top) == sorted(top)
        assert text.index('"amendment_rule"') < text.index('"universe"')
        assert '"version":1' in text
        assert '"version": 1' not in text

    def test_the_digest_names_its_algorithm(self) -> None:
        assert STANDING.digest.startswith("sha256:")


class TestVerificationIsThePathASkepticWalks:
    def test_a_fresh_commitment_verifies(self) -> None:
        ok, why = verify(_commitment())
        assert ok and "governs ledger seq 95 onward" in why

    def test_a_rewritten_protocol_body_fails_verification(self) -> None:
        c = _commitment()
        tampered = Commitment(
            protocol=replace(c.protocol, max_quantity="99"),
            committed_at=c.committed_at, ledger_head=c.ledger_head,
            ledger_entries=c.ledger_entries, protocol_digest=c.protocol_digest,
            commitment_digest=c.commitment_digest,
        )
        ok, why = verify(tampered)
        assert not ok and "does not match its recorded digest" in why

    def test_a_moved_ledger_position_fails_verification(self) -> None:
        c = _commitment()
        moved = Commitment(
            protocol=c.protocol, committed_at=c.committed_at, ledger_head=c.ledger_head,
            ledger_entries=5, protocol_digest=c.protocol_digest,
            commitment_digest=c.commitment_digest,
        )
        ok, why = verify(moved)
        assert not ok and "envelope does not match" in why

    def test_a_swapped_head_fails_verification(self) -> None:
        c = _commitment()
        swapped = Commitment(
            protocol=c.protocol, committed_at=c.committed_at, ledger_head="someotherhead",
            ledger_entries=c.ledger_entries, protocol_digest=c.protocol_digest,
            commitment_digest=c.commitment_digest,
        )
        assert not verify(swapped)[0]

    def test_a_negative_entry_count_is_refused_at_build(self) -> None:
        with pytest.raises(ProtocolError, match="negative"):
            build(STANDING, committed_at=AT, ledger_head="h", ledger_entries=-1)

    def test_an_unanchored_commitment_says_so_rather_than_hiding_it(self) -> None:
        assert _commitment().external_anchor is None
        assert "external_anchor" in _commitment().as_dict()


class TestCommitOnlyMovesForward:
    def test_it_appends_and_reloads(self, tmp_path: Path) -> None:
        path = tmp_path / "p.jsonl"
        c = _commitment()
        commit(c, path=path)
        back = load(path)
        assert len(back) == 1
        assert back[0].protocol_digest == c.protocol_digest
        assert back[0].protocol.universe == STANDING.universe
        assert verify(back[0])[0]

    def test_a_reloaded_commitment_is_byte_identical_in_digest(self, tmp_path: Path) -> None:
        path = tmp_path / "p.jsonl"
        c = commit(_commitment(), path=path)
        assert load(path)[0].commitment_digest == c.commitment_digest

    def test_a_missing_file_is_no_commitments_not_an_error(self, tmp_path: Path) -> None:
        assert load(tmp_path / "absent.jsonl") == ()

    def test_a_backward_commitment_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "p.jsonl"
        commit(_commitment(entries=94), path=path)
        with pytest.raises(ProtocolError, match="amended forward"):
            commit(_commitment(entries=50, version=2), path=path)

    def test_reusing_a_version_for_a_different_body_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "p.jsonl"
        commit(_commitment(entries=94), path=path)
        with pytest.raises(ProtocolError, match="needs a new version number"):
            commit(_commitment(entries=120, max_quantity="5"), path=path)

    def test_a_genuine_amendment_is_allowed_and_both_survive(self, tmp_path: Path) -> None:
        path = tmp_path / "p.jsonl"
        commit(_commitment(entries=94), path=path)
        commit(_commitment(entries=120, version=2, max_quantity="5"), path=path)
        got = load(path)
        assert [c.protocol.version for c in got] == [1, 2]
        assert [c.governs_from_seq for c in got] == [95, 121]

    def test_a_commitment_that_does_not_verify_is_never_written(self, tmp_path: Path) -> None:
        path = tmp_path / "p.jsonl"
        c = _commitment()
        broken = Commitment(
            protocol=c.protocol, committed_at=c.committed_at, ledger_head=c.ledger_head,
            ledger_entries=c.ledger_entries, protocol_digest="sha256:wrong",
            commitment_digest=c.commitment_digest,
        )
        with pytest.raises(ProtocolError, match="does not verify"):
            commit(broken, path=path)
        assert not path.exists()


class TestWhichProtocolGoverned:
    def test_decisions_before_any_commitment_are_ungoverned(self) -> None:
        c = _commitment(entries=94)
        assert governing((c,), 94) is None
        assert governing((c,), 1) is None

    def test_the_first_governed_seq_is_one_past_the_entry_count(self) -> None:
        c = _commitment(entries=94)
        assert governing((c,), 95) is c

    def test_the_later_commitment_wins_from_its_own_point(self) -> None:
        v1 = _commitment(entries=94)
        v2 = _commitment(entries=120, version=2)
        assert governing((v1, v2), 100) is v1
        assert governing((v1, v2), 121) is v2

    def test_order_on_disk_does_not_decide_which_governs(self) -> None:
        v1 = _commitment(entries=94)
        v2 = _commitment(entries=120, version=2)
        assert governing((v2, v1), 100) is v1

    def test_no_commitments_at_all_governs_nothing(self) -> None:
        assert governing((), 5000) is None


class TestTheAuditChecksRulesNotIntentions:
    def test_a_clean_ledger_is_compliant(self) -> None:
        c = _commitment(entries=0)
        report = audit([FakeEntry(1), FakeEntry(2, "AAPLUSDT")], (c,))
        assert report.compliant and report.governed == 2 and report.ungoverned == 0

    def test_ungoverned_decisions_are_counted_separately_not_called_compliant(self) -> None:
        c = _commitment(entries=94)
        report = audit([FakeEntry(1), FakeEntry(95)], (c,))
        assert report.ungoverned == 1 and report.governed == 1
        assert "predate any commitment" in " ".join(report.render())

    def test_a_symbol_outside_the_committed_universe_is_a_deviation(self) -> None:
        c = _commitment(entries=0, universe=("NVDAUSDT",))
        report = audit([FakeEntry(1, "AAPLUSDT")], (c,))
        assert not report.compliant
        assert report.deviations[0].rule == "universe"

    def test_opening_a_position_outside_the_committed_sessions_is_a_deviation(self) -> None:
        c = _commitment(entries=0)
        report = audit([FakeEntry(1, session_phase="weekend")], (c,))
        assert [d.rule for d in report.deviations] == ["session"]
        assert "weekend" in report.deviations[0].detail

    def test_abstaining_at_the_weekend_is_not_a_deviation(self) -> None:
        """The session rule restricts *opening* risk. The ninety-four weekend abstentions in the
        real ledger are exactly what the protocol expects, not violations of it."""
        c = _commitment(entries=0)
        report = audit(
            [FakeEntry(1, verdict="no_trade", quantity="0", session_phase="weekend")], (c,)
        )
        assert report.compliant

    def test_a_zero_quantity_trade_verdict_still_opens_nothing(self) -> None:
        c = _commitment(entries=0)
        report = audit([FakeEntry(1, verdict="trade", quantity="0", session_phase="weekend")], (c,))
        assert report.compliant

    def test_data_insufficient_is_an_abstention_however_it_is_sized(self) -> None:
        c = _commitment(entries=0)
        report = audit(
            [FakeEntry(1, verdict="data_insufficient", quantity="5", session_phase="weekend")], (c,)
        )
        assert report.compliant

    def test_oversize_is_a_deviation(self) -> None:
        c = _commitment(entries=0, max_quantity="1")
        report = audit([FakeEntry(1, quantity="2")], (c,))
        assert [d.rule for d in report.deviations] == ["size"]

    def test_an_unlisted_verdict_is_a_deviation_even_when_it_opens_nothing(self) -> None:
        c = _commitment(entries=0, allowed_verdicts=(str(Verdict.NO_TRADE),))
        report = audit([FakeEntry(1, verdict="trade", quantity="0")], (c,))
        assert [d.rule for d in report.deviations] == ["verdict"]

    def test_one_entry_can_break_several_rules_and_each_is_named(self) -> None:
        c = _commitment(entries=0, universe=("NVDAUSDT",), max_quantity="1")
        report = audit([FakeEntry(1, "AAPLUSDT", quantity="9", session_phase="weekend")], (c,))
        assert {d.rule for d in report.deviations} == {"universe", "session", "size"}

    def test_an_unparseable_quantity_never_crashes_the_audit(self) -> None:
        c = _commitment(entries=0)
        report = audit([FakeEntry(1, quantity="n/a", session_phase="weekend")], (c,))
        assert report.governed == 1

    def test_each_protocol_version_is_counted_on_its_own_decisions(self) -> None:
        v1 = _commitment(entries=0)
        v2 = _commitment(entries=2, version=2)
        report = audit([FakeEntry(1), FakeEntry(2), FakeEntry(3)], (v1, v2))
        assert report.by_protocol == {"argus-paper-v1 v1": 2, "argus-paper-v1 v2": 1}

    def test_the_rendered_report_names_every_deviation(self) -> None:
        c = _commitment(entries=0, universe=("NVDAUSDT",))
        text = " ".join(audit([FakeEntry(7, "AAPLUSDT")], (c,)).render())
        assert "1 deviation(s)" in text and "seq 7 AAPLUSDT" in text

    def test_as_dict_is_serialisable_and_states_compliance(self) -> None:
        c = _commitment(entries=0)
        got = audit([FakeEntry(1)], (c,)).as_dict()
        assert got["compliant"] is True and got["governed_decisions"] == 1


class TestTheStandingProtocolSaysWhatWeActuallyDo:
    def test_it_covers_the_whole_venue_universe(self) -> None:
        assert STANDING.universe == tuple(RTOKEN_SYMBOLS)
        assert len(STANDING.universe) == 12

    def test_it_will_not_open_risk_at_the_weekend(self) -> None:
        assert str(SessionPhase.WEEKEND) not in STANDING.tradeable_sessions
        assert str(SessionPhase.RTH) in STANDING.tradeable_sessions

    def test_the_hurdle_is_stated_as_a_rule_not_as_a_number(self) -> None:
        assert "deliberation" in STANDING.hurdle_rule
        assert "12.0bps" in STANDING.hurdle_rule

    def test_no_second_undeclared_bar_is_stacked_on_the_hurdle(self) -> None:
        assert STANDING.min_edge_over_hurdle_bps == "0"

    def test_settlement_is_declared_as_a_later_fetch(self) -> None:
        assert "LATER" in STANDING.settlement_rule
        assert "abstention" in STANDING.settlement_rule

    def test_the_hypothesis_is_falsifiable_and_names_the_failure_publication(self) -> None:
        assert "58-86%" in STANDING.hypothesis
        assert "publish" in STANDING.hypothesis

    def test_the_weaknesses_are_declared_in_advance(self) -> None:
        joined = " ".join(STANDING.known_weaknesses)
        assert "2026-09-12" in joined
        assert "no rTokens" in joined

    def test_amendment_is_forward_only_and_written_down(self) -> None:
        assert "new version" in STANDING.amendment_rule
        assert "never deleted" in STANDING.amendment_rule

    def test_the_real_weekend_ledger_would_be_compliant_under_it(self) -> None:
        """Ninety-four weekend abstentions, audited against the protocol that governs from 95."""
        c = _commitment(entries=94)
        entries = [
            FakeEntry(i, "NVDAUSDT", verdict="no_trade", quantity="0", session_phase="weekend")
            for i in range(1, 95)
        ]
        report = audit(entries, (c,))
        assert report.ungoverned == 94 and report.governed == 0 and report.compliant


class TestEnforcementMayOnlyReduce:
    def test_an_ungoverned_decision_passes_through_and_says_so(self) -> None:
        got = enforce(None, verdict="trade", quantity=Decimal("5"),
                      session_phase="rth", symbol="NVDAUSDT")
        assert not got.applied and got.quantity == Decimal("5")
        assert "no protocol governs" in got.reason

    def test_a_compliant_trade_is_untouched(self) -> None:
        got = enforce(_commitment(entries=0), verdict="trade", quantity=Decimal("1"),
                      session_phase="rth", symbol="NVDAUSDT")
        assert not got.applied and got.verdict == "trade" and got.quantity == Decimal("1")

    def test_it_refuses_a_trade_outside_the_committed_sessions(self) -> None:
        got = enforce(_commitment(entries=0), verdict="trade", quantity=Decimal("1"),
                      session_phase="weekend", symbol="NVDAUSDT")
        assert got.applied and got.verdict == str(Verdict.NO_TRADE)
        assert got.quantity == Decimal("0")
        assert "weekend" in got.reason

    def test_it_refuses_a_symbol_outside_the_committed_universe(self) -> None:
        got = enforce(_commitment(entries=0, universe=("NVDAUSDT",)), verdict="trade",
                      quantity=Decimal("1"), session_phase="rth", symbol="AAPLUSDT")
        assert got.applied and got.quantity == Decimal("0")
        assert "outside the committed universe" in got.reason

    def test_it_cuts_an_oversized_position_to_the_ceiling(self) -> None:
        got = enforce(_commitment(entries=0, max_quantity="1"), verdict="trade",
                      quantity=Decimal("7"), session_phase="rth", symbol="NVDAUSDT")
        assert got.applied and got.quantity == Decimal("1") and got.verdict == "trade"

    def test_an_abstention_is_never_touched_in_any_session(self) -> None:
        for phase in ("rth", "weekend", "overnight"):
            got = enforce(_commitment(entries=0), verdict="no_trade", quantity=Decimal("0"),
                          session_phase=phase, symbol="NVDAUSDT")
            assert not got.applied and got.verdict == "no_trade"

    def test_it_can_never_enlarge_a_position(self) -> None:
        c = _commitment(entries=0, max_quantity="100")
        for size in ("0.001", "0.5", "1", "99"):
            got = enforce(c, verdict="trade", quantity=Decimal(size),
                          session_phase="rth", symbol="NVDAUSDT")
            assert got.quantity <= Decimal(size)

    def test_it_can_never_create_a_trade_from_an_abstention(self) -> None:
        c = _commitment(entries=0)
        for verdict in (str(Verdict.NO_TRADE), str(Verdict.DATA_INSUFFICIENT)):
            got = enforce(c, verdict=verdict, quantity=Decimal("0"),
                          session_phase="rth", symbol="NVDAUSDT")
            assert got.verdict == verdict and got.quantity == Decimal("0")

    def test_the_rendered_line_distinguishes_acting_from_passing(self) -> None:
        c = _commitment(entries=0)
        acted = enforce(c, verdict="trade", quantity=Decimal("1"),
                        session_phase="weekend", symbol="NVDAUSDT")
        passed = enforce(c, verdict="trade", quantity=Decimal("1"),
                         session_phase="rth", symbol="NVDAUSDT")
        assert "refused" in acted.render()
        assert "nothing changed" in passed.render()
