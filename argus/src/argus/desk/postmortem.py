"""Postmortems that run themselves: the gate autopsy and the root-cause diagnosis, fired by events.

**Why this exists.** Two analyses already existed and neither ran on its own. `eval/autopsy.py`
explains why the desk took no position, and ran only when somebody typed its command — so the
cockpit panel reading its artefact could be showing a report from any date. `desk/rootcause.py`
turns graded decisions into named causes with checkable remedies, and had **no caller at all**
outside its tests (research/harvest/25-sre-postmortems.md). A diagnosis nobody runs is a claim the
product makes about itself and never checks.

**What was taken, and from where.** Google's *Site Reliability Engineering*, ch. 15 "Postmortem
Culture" (sre.google/sre-book/postmortem-culture/): a postmortem is **triggered by defined criteria
agreed in advance** — user-visible downtime past a threshold, data loss, on-call intervention,
resolution time past a threshold — rather than by whoever remembers, and every postmortem's action
items are **tracked to closure**. Both are processes in prose; there is no code to vendor. The
analogues here:

- *Triggers* (:func:`triggers`): the circuit breaker entering HALTED or REDUCE_ONLY (the desk's
  downtime), a pause escalating a decision to a human (on-call intervention), a settled trade
  landing in the worst decile of settled trades (the loss past a threshold), and every
  :data:`CADENCE` settlements, so a desk that never trips anything is still reviewed.
- *Action items tracked to closure* (:class:`Remedies`): each diagnosis gets a stable id from its
  cause and remedy. The remedies file records when it first and last fired, when a person marked it
  adopted and why, and — the check the book's review criteria ask for — whether the same diagnosis
  **fires again after adoption**, which is the difference between a remedy that worked and one that
  was only written down.

**What is graded** (:func:`graded`). A settled position grades its side against
``direction_correct``; a settled abstention grades its recorded lean against the counterfactual
move, the same pairs `paper/runner._graded_predictions` feeds the sizing calibration gate. No
decision in the ledger names a size, so ``predicted_magnitude_bps`` is ``None`` and the sizing
diagnosis cannot fire — by design, not by omission (`desk/workbench.Autopsy`).

**The gate autopsy is imported from the risk layer.** Its analysis moved from `eval/autopsy.py`
to `risk/gate_autopsy.py` (2026-09-28) so that this module, in layer 5, may run it: nothing below
the top layer imports the evaluation harness (`tests/test_eval_boundary.py`). :func:`run` still
accepts another examiner, which is how the tests exercise it.

A postmortem never stops a cycle: :func:`after_cycle` catches its own failures and records them in
the log, because a desk that stops trading because its self-review failed has inverted the order of
what matters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Protocol

from argus.desk.rootcause import DiagnosisReport, diagnose
from argus.desk.workbench import Autopsy

CADENCE = 10
"""Settlements between routine postmortems when nothing else has triggered one."""

WORST_DECILE_MIN = 10
"""Settled trades needed before "worst decile" means anything; below it every loss is a decile."""

ALARMS = ("halted", "reduce_only")
"""Breaker states whose arrival is a trigger. Returning to ACTIVE is recovery, not an incident."""


class GateExaminer(Protocol):
    """`eval/autopsy.examine`'s signature: the gate autopsy of a record."""

    def __call__(self, entries: Sequence[Any], risk_records: Sequence[dict[str, Any]], *,
                 hours_to_first_discovery: float | None = ...) -> Any: ...


@dataclass(frozen=True)
class Observation:
    """What a cycle knows that the triggers read. Plain values, so tests and replays build one."""

    activation: str
    escalations: int
    settlements: int
    settled_trades: tuple[tuple[int, Decimal], ...]
    """``(seq, net_pnl)`` for every settled position, in ledger order."""


def _paths(root: Path) -> dict[str, Path]:
    return {
        "state": root / "postmortem_state.json",
        "log": root / "postmortems.jsonl",
        "remedies": root / "remedies.json",
        "diagnosis": root / "diagnosis.json",
        "autopsy": root / "abstention_autopsy.json",
    }


def _write_json(path: Path, blob: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(handle, "w", encoding="utf-8") as out:
        json.dump(blob, out, indent=2, default=str)
    os.replace(name, path)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def triggers(previous: dict[str, Any] | None, now: Observation) -> list[str]:
    """Why a postmortem is due now, or ``[]``. ``previous`` is the state saved by the last run."""
    if previous is None:
        return ["first run: no postmortem on record"]
    out: list[str] = []
    before = str(previous.get("activation", "active"))
    if now.activation != before and now.activation in ALARMS:
        out.append(f"circuit breaker {before} -> {now.activation}")
    if now.escalations > int(previous.get("escalations", 0)):
        fresh = now.escalations - int(previous.get("escalations", 0))
        out.append(f"{fresh} decision(s) escalated to a human")
    last_seq = int(previous.get("last_settled_trade_seq", -1))
    if len(now.settled_trades) >= WORST_DECILE_MIN:
        pnls = sorted(p for _, p in now.settled_trades)
        cutoff = pnls[max(0, len(pnls) // 10 - 1)]
        worst = [s for s, p in now.settled_trades if s > last_seq and p <= cutoff and p < 0]
        if worst:
            out.append(f"settled trade(s) {', '.join(f'#{s}' for s in worst)} in the worst decile")
    if now.settlements - int(previous.get("settlements", 0)) >= CADENCE:
        out.append(f"{now.settlements - int(previous.get('settlements', 0))} settlements since "
                   f"the last postmortem")
    return out


def graded(entries: Sequence[Any]) -> list[Autopsy]:
    """Every settled decision the record can grade, as `desk/workbench.Autopsy`."""
    out: list[Autopsy] = []
    for entry in entries:
        if getattr(entry, "kind", "decision") != "decision" or not entry.is_settled:
            continue
        if not entry.is_abstention:
            if entry.direction_correct is None or entry.exit_price is None:
                continue
            predicted = "up" if str(entry.side).upper() == "BUY" else "down"
            wrong = "down" if predicted == "up" else "up"
            entry_price = Decimal(entry.entry_price)
            move = (Decimal(entry.exit_price) - entry_price) / entry_price * 10_000
            out.append(Autopsy(
                decision_id=f"#{entry.seq}", thesis=str(entry.thesis),
                predicted_direction=predicted,
                realised_direction=predicted if entry.direction_correct else wrong,
                predicted_magnitude_bps=None, realised_magnitude_bps=int(abs(move)),
                stated_confidence=float(entry.stated_confidence),
            ))
            continue
        lean = str(getattr(entry, "lean", "none")).lower()
        if lean not in ("up", "down") or entry.counterfactual_move_bps is None:
            continue
        move = Decimal(entry.counterfactual_move_bps)
        out.append(Autopsy(
            decision_id=f"#{entry.seq}", thesis=f"abstention lean: {entry.thesis}",
            predicted_direction=lean, realised_direction="up" if move > 0 else "down",
            predicted_magnitude_bps=None, realised_magnitude_bps=int(abs(move)),
            stated_confidence=float(getattr(entry, "lean_confidence", 0.0)),
        ))
    return out


def _costs(entries: Sequence[Any]) -> tuple[Decimal | None, Decimal | None]:
    """Gross P&L and fees over settled positions, or ``(None, None)`` when there are none."""
    gross = Decimal("0")
    net = Decimal("0")
    seen = False
    for entry in entries:
        if (getattr(entry, "kind", "decision") != "decision" or not entry.is_settled
                or entry.is_abstention or entry.gross_pnl is None or entry.net_pnl is None):
            continue
        seen = True
        gross += Decimal(entry.gross_pnl)
        net += Decimal(entry.net_pnl)
    return (gross, gross - net) if seen else (None, None)


def remedy_id(cause: str, remedy: str) -> str:
    """Stable across runs, so the same diagnosis is recognised when it fires again."""
    return hashlib.sha256(f"{cause}\n{remedy}".encode()).hexdigest()[:12]


class Remedies:
    """Every remedy a diagnosis has proposed, and what became of it."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def all(self) -> dict[str, dict[str, Any]]:
        blob = _read_json(self.path)
        return blob if isinstance(blob, dict) else {}

    def record(self, report: DiagnosisReport, *, at: datetime) -> list[str]:
        """Merge this run's diagnoses; returns the ids of adopted remedies that fired again."""
        rows = self.all()
        recurred: list[str] = []
        for diagnosis in report.diagnoses:
            rid = remedy_id(diagnosis.cause, diagnosis.remedy)
            row = rows.setdefault(rid, {
                "cause": diagnosis.cause, "remedy": diagnosis.remedy,
                "first_fired": at.isoformat(), "times_fired": 0,
                "adopted_at": None, "adopted_note": "", "fired_after_adoption": 0,
            })
            row["times_fired"] += 1
            row["last_fired"] = at.isoformat()
            row["severity"] = diagnosis.severity
            row["evidence"] = list(diagnosis.evidence)
            if row["adopted_at"] is not None and at.isoformat() > row["adopted_at"]:
                row["fired_after_adoption"] += 1
                recurred.append(rid)
        _write_json(self.path, rows)
        return recurred

    def adopt(self, rid: str, note: str, *, at: datetime) -> dict[str, Any]:
        rows = self.all()
        if rid not in rows:
            raise KeyError(f"no remedy {rid!r}; known: {', '.join(sorted(rows)) or 'none'}")
        if not note.strip():
            raise ValueError("say what was changed: an adoption with no note cannot be checked")
        rows[rid]["adopted_at"] = at.isoformat()
        rows[rid]["adopted_note"] = note.strip()
        rows[rid]["fired_after_adoption"] = 0
        _write_json(self.path, rows)
        return rows[rid]


def run(entries: Sequence[Any], risk_records: Sequence[dict[str, Any]], *, root: Path,
        reasons: Sequence[str], at: datetime,
        examine: GateExaminer | None = None) -> dict[str, Any]:
    """One postmortem: both analyses, their artefacts stamped with when and why, one log line."""
    from argus.risk.gate_autopsy import examine as gate_autopsy
    from argus.truth.clocks import DualClock

    paths = _paths(root)
    summary: dict[str, Any] = {"at": at.isoformat(), "triggers": list(reasons)}
    examine = examine or gate_autopsy
    try:
        state = DualClock().state(at, nav_age_seconds=60)
        gates = examine(entries, risk_records, hours_to_first_discovery=(
            0.0 if state.phase.has_price_discovery else state.hours_to_next_discovery))
        _write_json(paths["autopsy"], {**gates.as_dict(), "generated_at": at.isoformat(),
                                       "triggered_by": list(reasons)})
        summary["gate_autopsy"] = {"decisions": gates.decisions,
                                   "unreached": [g.gate.name for g in gates.unreached]}
    except ValueError as exc:  # gate_autopsy.AutopsyError: an empty or unreadable record
        summary["gate_autopsy"] = {"refused": str(exc)}
    gross, fees = _costs(entries)
    report = diagnose(graded(entries), gross_pnl=gross, fees=fees)
    _write_json(paths["diagnosis"], {**report.as_dict(), "generated_at": at.isoformat(),
                                     "triggered_by": list(reasons)})
    recurred = Remedies(paths["remedies"]).record(report, at=at)
    summary["diagnosis"] = {"sample": report.sample, "refused": report.refused,
                            "causes": [d.cause for d in report.diagnoses],
                            "adopted_remedies_that_fired_again": recurred}
    with paths["log"].open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(summary, default=str) + "\n")
    return summary


def observe(entries: Sequence[Any], *, activation: str, escalations: int) -> Observation:
    trades = tuple(
        (int(e.seq), Decimal(e.net_pnl)) for e in entries
        if getattr(e, "kind", "decision") == "decision" and e.is_settled
        and not e.is_abstention and e.net_pnl is not None
    )
    settled = sum(1 for e in entries
                  if getattr(e, "kind", "decision") == "decision" and e.is_settled)
    return Observation(activation=activation, escalations=escalations, settlements=settled,
                       settled_trades=trades)


def after_cycle(entries: Sequence[Any], risk_records: Sequence[dict[str, Any]], *, root: Path,
                activation: str, escalations: int, at: datetime | None = None,
                examine: GateExaminer | None = None) -> dict[str, Any] | None:
    """Run a postmortem if one is due; the paper runner calls this at the end of every cycle.
    Returns the run's summary, or ``None`` when nothing triggered one."""
    when = at or datetime.now(UTC)
    paths = _paths(root)
    now = observe(entries, activation=activation, escalations=escalations)
    previous = _read_json(paths["state"])
    reasons = triggers(previous if isinstance(previous, dict) else None, now)
    summary: dict[str, Any] | None = None
    try:
        if reasons:
            summary = run(entries, risk_records, root=root, reasons=reasons, at=when,
                          examine=examine)
    except Exception as exc:
        summary = {"at": when.isoformat(), "triggers": reasons,
                   "failed": f"{type(exc).__name__}: {exc}"}
        with paths["log"].open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(summary, default=str) + "\n")
    if reasons or previous is None:
        _write_json(paths["state"], {
            "activation": now.activation, "escalations": now.escalations,
            "settlements": now.settlements,
            "last_settled_trade_seq": max((s for s, _ in now.settled_trades), default=-1),
            "updated_at": when.isoformat(),
        })
    elif now.activation != previous.get("activation"):
        _write_json(paths["state"], {**previous, "activation": now.activation})
    return summary


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description="ARGUS postmortems and their remedies")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("run", help="run a postmortem now, whatever the triggers say")
    sub.add_parser("remedies", help="list every remedy and what became of it")
    adopt = sub.add_parser("adopt", help="mark a remedy adopted, saying what was changed")
    adopt.add_argument("remedy_id")
    adopt.add_argument("--note", required=True)
    args = parser.parse_args(argv)

    from argus.paper.ledger import PaperLedger
    from argus.paper.runner import LEDGER_PATH, RISK_PATH

    root = LEDGER_PATH.parent
    if args.command == "remedies":
        print(json.dumps(Remedies(_paths(root)["remedies"]).all(), indent=2))
        return 0
    if args.command == "adopt":
        row = Remedies(_paths(root)["remedies"]).adopt(args.remedy_id, args.note,
                                                       at=datetime.now(UTC))
        print(json.dumps(row, indent=2))
        return 0
    records: list[dict[str, Any]] = []
    if RISK_PATH.exists():
        records = [json.loads(x) for x in RISK_PATH.read_text(encoding="utf-8").splitlines()
                   if x.strip()]
    print(json.dumps(run(PaperLedger(path=LEDGER_PATH).entries, records, root=root,
                         reasons=["run by hand"], at=datetime.now(UTC)), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = [
    "ALARMS",
    "CADENCE",
    "WORST_DECILE_MIN",
    "GateExaminer",
    "Observation",
    "Remedies",
    "after_cycle",
    "graded",
    "observe",
    "remedy_id",
    "run",
    "triggers",
]
