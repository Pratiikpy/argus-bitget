"""Decision cards: the whole trail on one page, with absence printed rather than left blank."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from argus.eval.decisioncard import ABSENT, build, index, main

LEDGER: dict[str, Any] = {
    "seq": 7, "decided_at": "2026-09-12T18:13:31+00:00", "symbol": "NVDAUSDT",
    "verdict": "no_trade", "side": "BUY", "quantity": "0", "entry_price": "219.25",
    "stated_confidence": 0.92, "thesis": "nothing clears the hurdle",
    "invalidation": [], "market_state_hash": "aaa", "approved_intent_hash": "bbb",
    "session_phase": "weekend", "hours_to_discovery": 43.27, "entry_cost_bps": "0.000",
    "prev_hash": "ccc",
}

NOTES: dict[str, Any] = {
    "seq": 7, "symbol": "NVDAUSDT",
    "notes": ["[panel] 2 analysts", "[grounding] 1 of 3 figure(s) do not resolve to anything"],
    "flags": ["[grounding] 1 of 3 figure(s) do not resolve to anything"],
}

RISK: dict[str, Any] = {
    "seq": 7, "symbol": "NVDAUSDT", "intervened": False,
    "binding_constraint": "none", "reason": "no exposure proposed",
    "quantity_before": "0", "quantity_after": "0", "constitution_only_reduced": True,
}

COMMIT: dict[str, Any] = {
    "committed_at": "2026-09-13T05:00:00+00:00", "ledger_head": "head0", "ledger_entries": 5,
    "protocol_digest": "sha256:abc", "protocol": {"protocol_id": "argus-paper-v1", "version": 1},
}


def _trade(**over: Any) -> dict[str, Any]:
    row = dict(LEDGER)
    row.update({"verdict": "trade", "quantity": "1", "invalidation": ["thesis breaks"]})
    row.update(over)
    return row


class TestTheTrailIsJoined:
    def test_the_decision_itself_is_always_present(self) -> None:
        card = build(LEDGER)
        text = card.render()
        assert "Decision 7" in text and "NVDAUSDT" in text
        assert "nothing clears the hurdle" in text

    def test_checks_are_attached_by_sequence(self) -> None:
        card = build(LEDGER, notes=[NOTES])
        assert card.flags and "do not resolve" in card.render()

    def test_the_risk_ruling_is_attached(self) -> None:
        card = build(LEDGER, risk=[RISK])
        assert card.risk is not None and "nothing to narrow" in card.render()

    def test_the_governing_protocol_is_attached(self) -> None:
        card = build(LEDGER, commitments=[COMMIT])
        assert card.protocol is not None
        assert "argus-paper-v1" in card.render()

    def test_notes_for_a_different_decision_are_not_borrowed(self) -> None:
        other = dict(NOTES, seq=99)
        card = build(LEDGER, notes=[other])
        assert card.flags == [] and card.notes == []

    def test_the_newest_commitment_before_the_decision_governs(self) -> None:
        older = dict(COMMIT, ledger_entries=1, protocol_digest="sha256:old")
        card = build(LEDGER, commitments=[older, COMMIT])
        assert card.protocol is not None
        assert card.protocol["protocol_digest"] == "sha256:abc"

    def test_a_commitment_made_after_the_decision_does_not_govern_it(self) -> None:
        later = dict(COMMIT, ledger_entries=50)
        assert build(LEDGER, commitments=[later]).protocol is None

    def test_every_attached_part_names_where_it_came_from(self) -> None:
        card = build(LEDGER, notes=[NOTES], risk=[RISK], commitments=[COMMIT])
        text = card.render()
        for path in ("paper_ledger.jsonl", "desk_notes.jsonl", "risk_records.jsonl",
                     "protocol_commitments.jsonl"):
            assert path in text

    def test_only_attached_parts_are_cited(self) -> None:
        card = build(LEDGER)
        assert len(card.sources) == 1


class TestAbsenceIsPrinted:
    def test_a_missing_risk_record_says_so(self) -> None:
        assert "no risk record" in build(LEDGER).render()

    def test_missing_notes_say_so(self) -> None:
        assert "no notes were written" in build(LEDGER).render()

    def test_an_ungoverned_decision_is_named_ungoverned_not_compliant(self) -> None:
        text = build(LEDGER).render()
        assert "predates the pre-registered protocol" in text
        assert "ungoverned rather than as compliant" in text

    def test_an_unsettled_decision_never_reads_as_a_correct_one(self) -> None:
        text = build(LEDGER).render()
        assert "not settled yet" in text
        assert "right" not in text.split("## What happened next")[1].split("##")[0]

    def test_an_abstention_needs_no_falsifier_and_says_why(self) -> None:
        assert "opens no exposure, so it needs no falsifier" in build(LEDGER).render()

    def test_a_trade_without_invalidation_is_simply_marked_absent(self) -> None:
        card = build(_trade(invalidation=[]))
        assert ABSENT in card.render()

    def test_the_absent_marker_is_never_an_empty_string(self) -> None:
        assert ABSENT.strip()


class TestOutcomes:
    def test_a_settled_abstention_reports_the_counterfactual(self) -> None:
        row = dict(LEDGER, settled_at="2026-09-13T18:00:00+00:00",
                   counterfactual_move_bps="42.5")
        text = build(row).render()
        assert "42.5bps" in text and "gradeable rather than merely safe" in text

    def test_a_settled_abstention_without_a_counterfactual_says_that(self) -> None:
        row = dict(LEDGER, settled_at="2026-09-13T18:00:00+00:00")
        assert "counterfactual move was not recorded" in build(row).render()

    def test_a_settled_trade_reports_pnl_and_direction(self) -> None:
        row = _trade(settled_at="x", net_pnl="12.5", gross_pnl="15.0", direction_correct=True)
        text = build(row).render()
        assert "net 12.5" in text and "direction right" in text

    def test_a_wrong_direction_is_called_wrong(self) -> None:
        row = _trade(settled_at="x", net_pnl="-3", gross_pnl="-1", direction_correct=False)
        assert "direction wrong" in build(row).render()

    def test_a_settled_trade_with_no_grade_is_called_ungraded(self) -> None:
        row = _trade(settled_at="x", net_pnl="0", gross_pnl="0", direction_correct=None)
        assert "ungraded" in build(row).render()


class TestRiskSection:
    def test_an_intervention_states_the_constraint_and_the_sizes(self) -> None:
        risk = dict(RISK, intervened=True, binding_constraint="max_position",
                    reason="capped", quantity_before="100", quantity_after="50")
        text = build(_trade(), risk=[risk]).render()
        assert "max_position" in text and "100" in text and "50" in text

    def test_no_intervention_explains_that_the_layer_may_only_reduce(self) -> None:
        assert "may only reduce" in build(LEDGER, risk=[RISK]).render()


class TestTheIndex:
    def test_it_lists_every_card(self) -> None:
        cards = [build(LEDGER), build(dict(LEDGER, seq=8))]
        text = index(cards)
        assert "decision-7.md" in text and "decision-8.md" in text

    def test_it_reports_flag_counts_and_governance(self) -> None:
        cards = [build(LEDGER, notes=[NOTES], commitments=[COMMIT])]
        text = index(cards)
        assert "| 1 |" in text and "yes" in text

    def test_an_empty_index_still_renders(self) -> None:
        assert "0 decision(s)" in index([])


class TestTheIntegritySection:
    def test_the_hashes_are_shown(self) -> None:
        text = build(LEDGER).render()
        assert "aaa" in text and "bbb" in text and "ccc" in text


class TestSerialisation:
    def test_the_summary_carries_the_state_a_reader_filters_on(self) -> None:
        got = build(LEDGER, notes=[NOTES], risk=[RISK], commitments=[COMMIT]).as_dict()
        assert got["seq"] == 7 and got["flags"] == 1
        assert got["risk_recorded"] is True and got["governed"] is True
        assert got["settled"] is False


def test_the_cli_writes_cards_and_an_index(tmp_path: Path) -> None:
    out = tmp_path / "cards"
    assert main(["--all", "--out", str(out)]) == 0
    assert (out / "index.md").exists()
    written = list(out.glob("decision-*.md"))
    assert written
    assert "# Decision" in written[0].read_text(encoding="utf-8")


def test_the_cli_refuses_a_sequence_that_does_not_exist() -> None:
    assert main(["--seq", "99999999"]) == 1
