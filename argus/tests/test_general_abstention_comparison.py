"""Tests for ARGUS's abstention scoring vs. the general-purpose selective-prediction evaluators.

The rival numbers are recorded from real runs of fd-shifts (c4467aec) and torch-uncertainty
(3f82fe5d) in a scratch venv — see `RIVAL_PROVENANCE`. ARGUS's side is always called for real: the
live-record tests read the frozen rows of `data/paper_ledger.jsonl` and skip, with the reason, if
that ledger is absent.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from argus.eval import artefact
from argus.eval import general_abstention_comparison as module
from argus.eval.abstention_coverage import Extraction, LeanCall, score
from argus.eval.baselines import selective_rivals_runner as runner
from argus.eval.baselines.selective_rivals_runner import planted as runner_planted
from argus.eval.general_abstention_comparison import (
    FROZEN_COUNTS,
    LEDGER_PATH,
    REPORT_PATH,
    RIVAL_PROVENANCE,
    RIVAL_RECORDED,
    SCOPE_STATEMENT,
    criteria,
    degenerate_cases,
    direction_fidelity,
    frozen_extraction,
    frozen_settled_entries,
    live_parity,
    parity,
    planted_confidences,
    planted_skill,
    priced_working_point,
    rival_input,
    row_order_sensitivity,
    run,
)
from argus.paper.ledger import PaperLedger


def _calls(n: int = 90) -> tuple[LeanCall, ...]:
    out = []
    for i in range(n):
        move = ((i * 37) % 101 - 50) * 3.0
        at = datetime(2026, 9, 1, tzinfo=UTC) + timedelta(days=i % 5, minutes=i)
        out.append(LeanCall(seq=i, decided_at=at.isoformat(), symbol="X",
                            lean="up" if i % 3 else "down", confidence=0.55 + (i % 4) * 0.05,
                            move_bps=move, stated_confidence=0.6 + (i % 3) * 0.1))
    return tuple(out)


def _live_or_skip() -> Extraction:
    if not LEDGER_PATH.exists():
        pytest.skip(f"{LEDGER_PATH} absent — the live-record comparison needs the paper ledger")
    extraction = frozen_extraction()
    if len(extraction.calls) != FROZEN_COUNTS["lean_calls"]:
        pytest.fail(
            f"the frozen selection returned {len(extraction.calls)} calls, not "
            f"{FROZEN_COUNTS['lean_calls']}: a sealed settlement changed, which the chain forbids"
        )
    return extraction


class TestParity:
    def test_the_fixtures_match_fd_shifts_to_the_last_digit(self) -> None:
        got = parity()
        assert got["max_abs_diff"] < 1e-12
        assert all(row["working_point_matches"] for row in got["fixtures"])

    def test_the_live_record_matches_fd_shifts(self) -> None:
        extraction = _live_or_skip()
        for ranking in ("lean_confidence", "abstention_doubt"):
            report = score(extraction, ranking=ranking, resamples=20)
            assert live_parity(report, ranking)["max_abs_diff"] < 1e-12


class TestPlantedSkill:
    def test_abstention_quality_cannot_tell_a_perfect_gate_from_an_inverted_one(self) -> None:
        """The loss this comparison publishes: the old score never reads a confidence."""
        got = planted_skill(_calls())
        assert got["abstention_quality_is_identical_across_arms"] is True

    def test_the_adapted_score_orders_them(self) -> None:
        got = planted_skill(_calls())
        assert got["coverage_orders_oracle_random_inverted"] is True

    def test_fd_shifts_ordered_them_on_the_live_record(self) -> None:
        arms = RIVAL_RECORDED["planted_skill"]
        assert (arms["oracle"]["fd_shifts"]["aurc"] < arms["random"]["fd_shifts"]["aurc"]
                < arms["inverted"]["fd_shifts"]["aurc"])

    def test_the_adapted_score_matches_fd_shifts_on_every_live_arm(self) -> None:
        got = planted_skill(_live_or_skip().calls)
        for arm in ("oracle", "random", "inverted", "constant"):
            row = got["arms"][arm]
            assert row["argus_coverage_aurc"] == pytest.approx(row["fd_shifts_aurc"], abs=1e-12)
        assert got["max_abs_diff_argus_vs_fd_shifts"] < 1e-14
        assert got["fd_shifts_orders_oracle_random_inverted"] is True
        assert got["torch_uncertainty_orders_oracle_random_inverted"] is True

    def test_the_rival_runner_plants_the_same_four_arms(self) -> None:
        """Same input on both sides: the runner's construction is ARGUS's, arm for arm."""
        calls = _calls()
        ours = planted_confidences(calls)
        theirs = runner_planted([c.net_bps for c in calls])
        assert ours == theirs


class TestRowOrder:
    def test_argus_does_not_move(self) -> None:
        assert row_order_sensitivity(_calls(), permutations=30)["argus_aurc_spread"] == 0.0

    def test_torch_uncertainty_moved_more_than_the_effect_on_the_live_record(self) -> None:
        rec = RIVAL_RECORDED["row_order_sensitivity"]
        gap = abs(RIVAL_RECORDED["live"]["lean_confidence"]["fd_shifts"]["aurc"]
                  - 250 / 470)   # 250 of 470 leans would have lost after fees
        assert rec["fd_shifts_aurc"]["spread"] == 0.0
        assert rec["torch_uncertainty_aurc"]["spread"] > gap

    def test_the_error_count_behind_that_gap(self) -> None:
        calls = _live_or_skip().calls
        assert sum(c.error for c in calls) == 250


class TestPricedAndDegenerate:
    def test_fd_shifts_working_point_meets_its_target_and_loses_money(self) -> None:
        got = priced_working_point(_live_or_skip().calls)
        assert got["acted"] == 42
        assert got["meets_its_risk_target_and_loses_money"] is True
        assert got["total_net_bps"] == pytest.approx(-3344.1762, abs=1e-3)

    def test_where_fd_shifts_raised_argus_returns_none(self) -> None:
        got = degenerate_cases(_live_or_skip().calls)
        assert all(v["fd_shifts_raised"] for v in got.values())
        assert all(v["argus_working_point"] is None for v in got.values())


class TestDirectionFidelity:
    def _ledger(self, tmp_path: Path) -> PaperLedger:
        led = PaperLedger(path=tmp_path / "p.jsonl")
        for i, (lean, move) in enumerate((("down", "-80"), ("down", "-60"), ("up", "40"))):
            led.record(
                symbol="NVDAUSDT", verdict="no_trade", side="BUY", quantity=Decimal("0"),
                entry_price=Decimal("100"), stated_confidence=0.8, thesis="t", invalidation=(),
                market_state_hash="h", approved_intent_hash="a", session_phase="rth",
                hours_to_discovery=0.0, lean=lean, lean_confidence=0.6,
                decided_at=datetime(2026, 9, 1, tzinfo=UTC) + timedelta(hours=i),
            )
            led.settle_abstention(i + 1, price_now=Decimal("100") * (1 + Decimal(move) / 10000))
        return led

    def test_grading_by_side_flips_the_sign_where_the_leans_were_short(
        self, tmp_path: Path
    ) -> None:
        got = direction_fidelity([e for e in self._ledger(tmp_path).entries])
        assert got["graded_against_the_opposite_of_the_lean_before"] == 2
        # by side (all BUY): avoided 80+12 and 60+12, missed 40-12 -> +136
        # by lean (short, short, long): missed 68 and 48, missed 28 -> -144
        assert Decimal(got["before_side_based"]["total_value_bps"]) > 0
        assert Decimal(got["after_lean_based"]["total_value_bps"]) < 0
        assert got["sign_flipped"] is True

    def test_it_flipped_on_the_live_record(self) -> None:
        _live_or_skip()
        got = direction_fidelity(frozen_settled_entries())
        assert got["graded_before"] == FROZEN_COUNTS["settled"]
        assert got["graded_after"] == FROZEN_COUNTS["lean_calls"]
        assert got["graded_against_the_opposite_of_the_lean_before"] == 84
        assert got["no_lean_graded_as_a_side_before"] == FROZEN_COUNTS["no_lean"]
        assert got["sign_flipped"] is True


class TestArtefactAndScope:
    def test_the_artefact_is_strict_json_with_a_verdict(self) -> None:
        if not REPORT_PATH.exists():
            pytest.skip("artefact absent: python -m argus.eval.general_abstention_comparison")
        assert artefact.is_strict(REPORT_PATH)
        blob = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        assert blob["verdict"]["before_adaptation"].startswith("rival_wins")
        assert len(blob["criteria"]) == 6
        assert blob["input"]["matches_frozen_counts"] is True

    def test_provenance_pins_both_rivals(self) -> None:
        assert RIVAL_PROVENANCE["fd_shifts"]["commit"].startswith("c4467aec")
        assert RIVAL_PROVENANCE["torch_uncertainty"]["commit"].startswith("3f82fe5d")
        assert RIVAL_PROVENANCE["fd_shifts"]["licence"] == "Apache-2.0"

    def test_the_scope_names_what_is_not_claimed(self) -> None:
        assert "NOT claimed: that the desk's refusals were right" in SCOPE_STATEMENT
        assert "a loss, and it is published" in SCOPE_STATEMENT

    def test_provenance_names_a_runner_in_the_repository_not_a_local_path(self) -> None:
        text = json.dumps(RIVAL_PROVENANCE)
        assert "selective_rivals_runner.py" in text
        for marker in (artefact.SCRATCH_DIR, "C:/", "C:\\", "/Users/", "AppData"):
            assert marker not in text

    def test_no_local_path_reaches_the_code_or_the_artefact(self) -> None:
        paths = [Path(module.__file__), Path(runner.__file__)]
        if REPORT_PATH.exists():
            paths.append(REPORT_PATH)
        for path in paths:
            text = path.read_text(encoding="utf-8")
            for marker in (artefact.SCRATCH_DIR, "AppData", "C:/Users", "C:\\\\Users", "/c/Users"):
                assert marker not in text, f"{marker!r} in {path.name}"


class TestRivalInput:
    def test_it_carries_what_a_rival_needs_and_the_freeze(self) -> None:
        calls = _calls()
        blob = rival_input(Extraction(calls=calls, no_lean=0, unsettled=0))
        assert blob["upto_seq"] == 684 and blob["ledger_head_at_freeze"] == "5d144cbef0af9faf"
        assert len(blob["calls"]) == len(calls)
        first = blob["calls"][0]
        assert set(first) == {"seq", "confidence", "stated_confidence", "error", "net_bps"}
        assert first["error"] == calls[0].error and first["net_bps"] == calls[0].net_bps
        json.dumps(blob, allow_nan=False)

    def test_the_live_input_has_the_frozen_counts(self) -> None:
        blob = rival_input(_live_or_skip())
        assert len(blob["calls"]) == FROZEN_COUNTS["lean_calls"]
        assert sum(c["error"] for c in blob["calls"]) == 250


class TestCriteriaText:
    def test_the_priced_loss_reads_as_a_positive_loss(self) -> None:
        extraction = _live_or_skip()
        pw = priced_working_point(extraction.calls)
        assert pw["total_net_bps"] < 0
        row = criteria({
            "planted_skill": planted_skill(extraction.calls),
            "row_order": row_order_sensitivity(extraction.calls, permutations=5),
            "priced_working_point": pw,
            "uncertainty": {"fd_shifts_iid_width": 0.1, "argus_by_day_width": 0.2,
                            "icc_of_loss_within_day": 0.1, "design_effect": 5.0},
            "direction_fidelity": direction_fidelity(frozen_settled_entries()),
            "degenerate_cases": degenerate_cases(extraction.calls),
        })[2]
        assert "would have lost 3,344bps over 42 calls" in row["fd_shifts"]
        assert "lost -" not in row["fd_shifts"]


class TestArtefactReproduces:
    def test_every_number_but_timing_matches_a_fresh_run(self) -> None:
        """The saved artefact is what the code computes today, not what it computed once."""
        _live_or_skip()
        if not REPORT_PATH.exists():
            pytest.skip("artefact absent: python -m argus.eval.general_abstention_comparison")
        saved = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        fresh = json.loads(artefact.dumps(run()))
        saved.pop("costs")
        fresh.pop("costs")
        assert json.dumps(saved, sort_keys=True) == json.dumps(fresh, sort_keys=True)
