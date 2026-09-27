"""Self-running postmortems: they fire on the agreed triggers, and a remedy is tracked to closure.

The properties under test are the two the SRE postmortem practice asks for
(research/harvest/25-sre-postmortems.md): a review starts because a defined event happened, not
because somebody remembered; and an adopted fix is checked by whether the same diagnosis fires
again afterwards.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from argus.desk import postmortem
from argus.desk.postmortem import (
    CADENCE,
    Observation,
    Remedies,
    after_cycle,
    graded,
    remedy_id,
    triggers,
)
from argus.desk.rootcause import Diagnosis, DiagnosisReport

T0 = datetime(2026, 9, 28, 12, tzinfo=UTC)


@dataclass
class Row:
    """The ledger fields `desk/postmortem.py` reads, and nothing else."""

    seq: int
    side: str = "BUY"
    thesis: str = "fixture"
    entry_price: str = "100"
    exit_price: str | None = None
    stated_confidence: float = 0.8
    direction_correct: bool | None = None
    gross_pnl: str | None = None
    net_pnl: str | None = None
    settled: bool = True
    abstention: bool = False
    lean: str = "none"
    lean_confidence: float = 0.0
    counterfactual_move_bps: str | None = None
    kind: str = "decision"

    @property
    def is_settled(self) -> bool:
        return self.settled

    @property
    def is_abstention(self) -> bool:
        return self.abstention


def _trade(seq: int, pnl: str, *, right: bool | None = None) -> Row:
    won = Decimal(pnl) > 0 if right is None else right
    return Row(seq=seq, exit_price="101" if won else "99", direction_correct=won,
               gross_pnl=str(Decimal(pnl) + 1), net_pnl=pnl)


def _observe(activation: str = "active", escalations: int = 0, settlements: int = 0,
             trades: tuple[tuple[int, Decimal], ...] = ()) -> Observation:
    return Observation(activation=activation, escalations=escalations, settlements=settlements,
                       settled_trades=trades)


BASE = {"activation": "active", "escalations": 0, "settlements": 0, "last_settled_trade_seq": -1}


def test_the_first_cycle_runs_one() -> None:
    assert triggers(None, _observe()) == ["first run: no postmortem on record"]


def test_a_quiet_cycle_runs_none() -> None:
    assert triggers(BASE, _observe(settlements=CADENCE - 1)) == []


@pytest.mark.parametrize("state", ["halted", "reduce_only"])
def test_the_breaker_tripping_runs_one(state: str) -> None:
    assert triggers(BASE, _observe(activation=state)) == [f"circuit breaker active -> {state}"]


def test_staying_halted_does_not_run_one_every_cycle() -> None:
    assert triggers({**BASE, "activation": "halted"}, _observe(activation="halted")) == []


def test_recovery_is_not_an_incident() -> None:
    assert triggers({**BASE, "activation": "halted"}, _observe(activation="active")) == []


def test_an_escalation_runs_one() -> None:
    assert triggers(BASE, _observe(escalations=2)) == ["2 decision(s) escalated to a human"]


def test_a_worst_decile_loss_runs_one_once() -> None:
    trades = tuple((i, Decimal(i - 3)) for i in range(20))  # seq 0 and 1 are the worst two
    fresh = {**BASE, "last_settled_trade_seq": -1}
    assert triggers(fresh, _observe(trades=trades)) == [
        "settled trade(s) #0, #1 in the worst decile"]
    assert triggers({**BASE, "last_settled_trade_seq": 19}, _observe(trades=trades)) == []


def test_worst_decile_needs_a_sample() -> None:
    trades = tuple((i, Decimal(-5)) for i in range(postmortem.WORST_DECILE_MIN - 1))
    assert triggers(BASE, _observe(trades=trades)) == []


def test_the_cadence_runs_one() -> None:
    assert triggers(BASE, _observe(settlements=CADENCE)) == [
        f"{CADENCE} settlements since the last postmortem"]


def test_a_settled_trade_is_graded_on_its_side() -> None:
    [short] = graded([Row(seq=4, side="SELL", exit_price="98", direction_correct=True)])
    assert (short.predicted_direction, short.realised_direction) == ("down", "down")
    assert short.realised_magnitude_bps == 200 and short.predicted_magnitude_bps is None
    assert short.failure_mode == "correct"
    [loser] = graded([Row(seq=5, exit_price="99", direction_correct=False, stated_confidence=0.9)])
    assert loser.realised_direction == "down" and loser.failure_mode == "overconfident"


def test_an_abstention_is_graded_on_its_lean() -> None:
    rows = [Row(seq=1, abstention=True, lean="up", lean_confidence=0.6,
                counterfactual_move_bps="-35"),
            Row(seq=2, abstention=True, lean="none", counterfactual_move_bps="10"),
            Row(seq=3, abstention=True, lean="down", counterfactual_move_bps=None)]
    [only] = graded(rows)
    assert only.decision_id == "#1" and not only.direction_correct
    assert only.stated_confidence == 0.6


def test_unsettled_and_seal_rows_are_not_graded() -> None:
    assert graded([Row(seq=1, settled=False, direction_correct=True, exit_price="101"),
                   Row(seq=2, kind="settlement_seal", direction_correct=True,
                       exit_price="101")]) == []


def _report(*causes: str) -> DiagnosisReport:
    return DiagnosisReport(diagnoses=tuple(
        Diagnosis(cause=c, severity="primary", evidence=("#1",), detail="d", remedy=f"fix {c}")
        for c in causes), sample=5)


def test_a_remedy_that_fires_after_adoption_is_caught(tmp_path: Path) -> None:
    remedies = Remedies(tmp_path / "remedies.json")
    remedies.record(_report("overconfidence"), at=T0)
    rid = remedy_id("overconfidence", "fix overconfidence")
    remedies.adopt(rid, "sizing now uses the realised hit rate", at=T0 + timedelta(hours=1))
    assert remedies.record(_report("overconfidence"), at=T0 + timedelta(hours=2)) == [rid]
    row = remedies.all()[rid]
    assert row["times_fired"] == 2 and row["fired_after_adoption"] == 1
    assert row["adopted_note"] == "sizing now uses the realised hit rate"


def test_a_remedy_that_holds_stays_clean(tmp_path: Path) -> None:
    remedies = Remedies(tmp_path / "remedies.json")
    remedies.record(_report("fee drag"), at=T0)
    rid = remedy_id("fee drag", "fix fee drag")
    remedies.adopt(rid, "trade less often", at=T0 + timedelta(hours=1))
    assert remedies.record(_report("overconfidence"), at=T0 + timedelta(hours=2)) == []
    assert remedies.all()[rid]["fired_after_adoption"] == 0


def test_an_adoption_must_say_what_changed(tmp_path: Path) -> None:
    remedies = Remedies(tmp_path / "remedies.json")
    remedies.record(_report("fee drag"), at=T0)
    with pytest.raises(ValueError):
        remedies.adopt(remedy_id("fee drag", "fix fee drag"), "  ", at=T0)
    with pytest.raises(KeyError):
        remedies.adopt("nope", "x", at=T0)


def test_the_cycle_writes_stamped_artefacts_and_a_log(tmp_path: Path) -> None:
    rows: list[Any] = [_trade(i, "-40", right=False) for i in range(6)]
    summary = after_cycle(rows, [], root=tmp_path, activation="active", escalations=0, at=T0)
    assert summary is not None and summary["triggers"] == ["first run: no postmortem on record"]
    diagnosis = json.loads((tmp_path / "diagnosis.json").read_text(encoding="utf-8"))
    assert diagnosis["generated_at"] == T0.isoformat() and diagnosis["sample"] == 6
    assert "overconfidence" in [d["cause"] for d in diagnosis["diagnoses"]]
    assert (tmp_path / "remedies.json").exists()
    log = (tmp_path / "postmortems.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(log) == 1
    # Nothing new: the next cycle runs no postmortem and writes no log line.
    assert after_cycle(rows, [], root=tmp_path, activation="active", escalations=0,
                       at=T0 + timedelta(hours=1)) is None
    assert len((tmp_path / "postmortems.jsonl").read_text(encoding="utf-8").splitlines()) == 1


def test_a_halt_mid_run_triggers_and_is_remembered(tmp_path: Path) -> None:
    rows: list[Any] = [_trade(1, "5")]
    after_cycle(rows, [], root=tmp_path, activation="active", escalations=0, at=T0)
    fired = after_cycle(rows, [], root=tmp_path, activation="halted", escalations=0,
                        at=T0 + timedelta(hours=1))
    assert fired is not None and fired["triggers"] == ["circuit breaker active -> halted"]
    assert after_cycle(rows, [], root=tmp_path, activation="halted", escalations=0,
                       at=T0 + timedelta(hours=2)) is None
    after_cycle(rows, [], root=tmp_path, activation="active", escalations=0,
                at=T0 + timedelta(hours=3))
    again = after_cycle(rows, [], root=tmp_path, activation="reduce_only", escalations=0,
                        at=T0 + timedelta(hours=4))
    assert again is not None and again["triggers"] == ["circuit breaker active -> reduce_only"]


def test_a_failing_postmortem_never_stops_the_cycle(tmp_path: Path,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*a: Any, **k: Any) -> Any:
        raise RuntimeError("disk full")

    monkeypatch.setattr(postmortem, "diagnose", boom)
    summary = after_cycle([_trade(1, "5")], [], root=tmp_path, activation="active",
                          escalations=0, at=T0)
    assert summary is not None and summary["failed"] == "RuntimeError: disk full"
    assert "failed" in (tmp_path / "postmortems.jsonl").read_text(encoding="utf-8")


def test_too_few_graded_decisions_are_refused_not_diagnosed(tmp_path: Path) -> None:
    after_cycle([_trade(1, "-5")], [], root=tmp_path, activation="active", escalations=0, at=T0)
    diagnosis = json.loads((tmp_path / "diagnosis.json").read_text(encoding="utf-8"))
    assert diagnosis["refused"] and diagnosis["diagnoses"] == []


def test_the_paper_runner_runs_it_beside_its_own_ledger(tmp_path: Path) -> None:
    """`paper/runner._postmortem`, the hook every cycle calls: files land beside the ledger it
    was given, never in the live data directory, and an empty record is refused, not diagnosed."""
    from argus.decision.pause import PauseStore
    from argus.paper.ledger import PaperLedger
    from argus.paper.runner import _postmortem

    ledger_path = tmp_path / "paper_ledger.jsonl"
    summary = _postmortem(PaperLedger(path=ledger_path), ledger_path,
                          PauseStore(tmp_path / "pauses"), T0)
    assert summary is not None and summary["triggers"] == ["first run: no postmortem on record"]
    assert "refused" in summary["gate_autopsy"]  # type: ignore[operator]
    state = json.loads((tmp_path / "postmortem_state.json").read_text(encoding="utf-8"))
    assert state["activation"] == "active" and state["escalations"] == 0
    assert (tmp_path / "diagnosis.json").exists()


def test_the_gate_autopsy_runs_by_default_and_is_stamped(tmp_path: Path) -> None:
    """The analysis now lives in the risk layer (`risk/gate_autopsy.py`), so the postmortem runs
    it without anything handed in, and its artefact says when and why it was written."""
    rows: list[Any] = [Row(seq=i, abstention=True, settled=True) for i in range(3)]
    summary = after_cycle(rows, [], root=tmp_path, activation="active", escalations=0, at=T0)
    assert summary is not None and summary["gate_autopsy"]["decisions"] == 3
    written = json.loads((tmp_path / "abstention_autopsy.json").read_text(encoding="utf-8"))
    assert written["generated_at"] == T0.isoformat()
    assert written["triggered_by"] == ["first run: no postmortem on record"]
