"""Did the risk layer actually do anything? — the Track-2 judge question, answered with counts.

Track 2's judged half scores "risk control layer effectiveness"
(`BITGET_AI_BASE_CAMP_S2_HANDBOOK_EN.md`). ARGUS has a Constitution that may only reduce, a circuit
breaker, a drawdown ladder and calibration-gated sizing — and until this module, **nothing anywhere
could say how often any of it changed a decision.** `ConstitutionRuling` was designed to be
countable and its own docstring says so: *"which constraint bound, how often is the evidence that
the risk layer does real work, and free text cannot be aggregated."* Nothing was aggregating it.

**The two failure modes this report is built to expose, because both look like success.**

* A layer that **never fires** is not safe, it is untested. A clean record with zero interventions
  proves only that nothing has been asked of it, and a report that called that "0 violations" would
  be reading an absence of evidence as evidence.
* A layer that fires on **everything** has taken the decision away from the model. That is the
  positioning failure the whole architecture exists to avoid: if the Constitution binds every time,
  the model is not the decision-maker and the entry is mis-positioned for the track.

So :class:`RiskAudit` reports an :class:`Exercise` verdict — UNTESTED, LIGHT, ACTIVE, DOMINANT —
rather than a score, and the intervention rate is stated as a fraction of the decisions that could
have been intervened on.

**What this cannot answer yet, and says so.** Whether an intervention was *correct* needs the
outcome of the trade it reduced, and the log has no settled trades. The report returns the
effectiveness-of-outcome fields as ``None`` with the reason attached rather than printing a zero
that reads as "the risk layer saved nothing".
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Any

REPORT_PATH = Path(__file__).resolve().parents[3] / "data" / "risk_audit.json"
"""Where the report is written.

It had none. This module answers Track 2's judged line *"risk control layer effectiveness"*
(handbook line 254) and printed its answer to stdout only, so there was nothing on disk for a
submission to link or for `tests/test_artefacts_reproducible.py` to check. Every other eval module
here leaves an artefact; a finding that exists only in a terminal someone once ran is not evidence.
"""

MIN_DECISIONS_FOR_A_VERDICT = 10
"""Below this the exercise verdict is UNTESTED whatever the rate.

Two interventions in three decisions is not a 67% rate, it is three decisions. Reporting a
percentage over a handful of rows invites exactly the over-reading this module exists to prevent."""

DOMINANT_RATE = 0.8
"""Above this the layer, not the model, is effectively deciding."""

LIGHT_RATE = 0.05
"""Below this the layer is barely exercised, even when it has fired once or twice."""


class Exercise(StrEnum):
    """How hard the risk layer has actually been worked."""

    UNTESTED = "untested"
    """It has never bound, or there are too few decisions to say. Not a pass."""

    LIGHT = "light"
    ACTIVE = "active"
    DOMINANT = "dominant"
    """It binds on nearly everything, which means the model is not deciding."""

    @property
    def is_healthy(self) -> bool:
        """Only ACTIVE is evidence of a working layer. LIGHT is weak, the others are findings."""
        return self is Exercise.ACTIVE


@dataclass(frozen=True, slots=True)
class Intervention:
    """One decision the risk layer touched, and what it did to it."""

    seq: int
    symbol: str
    verdict: str
    binding_constraint: str
    reason: str
    quantity_before: str
    quantity_after: str
    model_changed_its_mind: bool
    """Did the model rewrite its intent in response, rather than repeat itself?"""

    @property
    def reduced(self) -> bool:
        try:
            return Decimal(self.quantity_after) < Decimal(self.quantity_before)
        except (ArithmeticError, ValueError):
            return False

    @property
    def created_exposure(self) -> bool:
        """The asymmetry violation. A risk layer may only reduce; this must never be true."""
        try:
            return Decimal(self.quantity_after) > Decimal(self.quantity_before)
        except (ArithmeticError, ValueError):
            return False

    def as_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "symbol": self.symbol,
            "verdict": self.verdict,
            "binding_constraint": self.binding_constraint,
            "reason": self.reason,
            "quantity_before": self.quantity_before,
            "quantity_after": self.quantity_after,
            "reduced": self.reduced,
            "model_changed_its_mind": self.model_changed_its_mind,
        }


@dataclass(frozen=True, slots=True)
class RiskAudit:
    """The risk layer's record over every decision that had one."""

    decisions: int
    interventions: tuple[Intervention, ...]
    positions_offered: int
    """Decisions carrying a quantity the layer could have reduced.

    The denominator that matters. A desk that abstained 79 times gave its Constitution nothing to
    bind on, and dividing by 79 would report a risk layer as idle when it was never invoked."""

    outcome_effect_note: str

    @property
    def rate(self) -> float | None:
        """Interventions as a fraction of decisions that offered a position to reduce."""
        if not self.positions_offered:
            return None
        return len(self.interventions) / self.positions_offered

    @property
    def exercise(self) -> Exercise:
        if self.decisions < MIN_DECISIONS_FOR_A_VERDICT or not self.interventions:
            return Exercise.UNTESTED
        rate = self.rate
        if rate is None:
            return Exercise.UNTESTED
        if rate >= DOMINANT_RATE:
            return Exercise.DOMINANT
        if rate <= LIGHT_RATE:
            return Exercise.LIGHT
        return Exercise.ACTIVE

    @property
    def asymmetry_violations(self) -> tuple[Intervention, ...]:
        """Any case where the layer increased exposure. The design forbids it; this checks."""
        return tuple(i for i in self.interventions if i.created_exposure)

    def by_constraint(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in self.interventions:
            counts[item.binding_constraint] = counts.get(item.binding_constraint, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))

    @property
    def responded(self) -> int:
        """Interventions the model answered by rewriting its intent rather than repeating it."""
        return sum(1 for i in self.interventions if i.model_changed_its_mind)

    def as_dict(self) -> dict[str, Any]:
        return {
            "decisions": self.decisions,
            "positions_offered": self.positions_offered,
            "interventions": len(self.interventions),
            "intervention_rate": None if self.rate is None else round(self.rate, 4),
            "exercise": str(self.exercise),
            "healthy": self.exercise.is_healthy,
            "by_constraint": self.by_constraint(),
            "model_responded": self.responded,
            "asymmetry_violations": [i.as_dict() for i in self.asymmetry_violations],
            "outcome_effect": None,
            "outcome_effect_note": self.outcome_effect_note,
            "detail": [i.as_dict() for i in self.interventions],
        }

    def render(self) -> list[str]:
        lines: list[str] = []
        if self.asymmetry_violations:
            # First, always. Everything else is a statistic; this is a broken invariant.
            lines.append(
                f"[risk] ASYMMETRY VIOLATED on {len(self.asymmetry_violations)} decision(s): the "
                f"risk layer increased exposure, which it is built to be incapable of"
            )
        if self.exercise is Exercise.UNTESTED:
            lines.append(
                f"[risk] the risk layer has not been exercised: {len(self.interventions)} "
                f"intervention(s) over {self.decisions} decision(s), "
                f"{self.positions_offered} of which offered a position to reduce. "
                f"This is untested, not safe"
            )
            return lines
        rate = self.rate or 0.0
        lines.append(
            f"[risk] {len(self.interventions)} intervention(s) on {self.positions_offered} "
            f"position-carrying decision(s) — {rate:.0%}, graded {self.exercise}"
        )
        if self.exercise is Exercise.DOMINANT:
            lines.append(
                "[risk] the layer binds on nearly every position, which means the model is not "
                "the decision-maker; that is a positioning failure, not a safety record"
            )
        for constraint, count in self.by_constraint().items():
            lines.append(f"[risk]   {constraint}: {count}")
        lines.append(
            f"[risk] the model rewrote its intent in response {self.responded} of "
            f"{len(self.interventions)} time(s)"
        )
        lines.append(f"[risk] {self.outcome_effect_note}")
        return lines


def from_records(
    records: Iterable[dict[str, Any]], *, settled_trades: int = 0
) -> RiskAudit:
    """Build the audit from the per-decision risk records written beside the ledger."""
    from argus.paper.corrections import is_voided

    rows = list(records)
    interventions: list[Intervention] = []
    offered = 0
    for row in rows:
        # **Voided rows carry a `quantity_before` that was never offered to the Constitution.**
        # Until 2026-09-20 `paper/runner.py` wrote that field from the model's first draft rather
        # than from the intent the Constitution actually ruled on, so seq 264 and 265 store
        # `quantity_before: "1"` beside `intervened: false` and `reason: "no exposure proposed;
        # nothing to narrow"` — three claims that cannot all be true. Counting them made this
        # module report *"2 of which offered a position to reduce"* while `eval/autopsy.py`
        # reported *"0 of 447 decision(s) proposed exposure"* off the same record.
        #
        # The rows are not edited — they are an audited artefact, and `paper/corrections.py`
        # already names them — so the correction lives in every reader, exactly as it does for the
        # ledger's own `is_abstention`.
        if is_voided(int(row.get("seq", 0))):
            continue
        if str(row.get("quantity_before", "0")) not in ("0", "0.0", ""):
            offered += 1
        if not row.get("intervened"):
            continue
        interventions.append(Intervention(
            seq=int(row.get("seq", 0)),
            symbol=str(row.get("symbol", "")),
            verdict=str(row.get("verdict", "")),
            binding_constraint=str(row.get("binding_constraint") or "unnamed"),
            reason=str(row.get("reason", "")),
            quantity_before=str(row.get("quantity_before", "0")),
            quantity_after=str(row.get("quantity_after", "0")),
            model_changed_its_mind=bool(row.get("model_changed_its_mind")),
        ))

    note = (
        f"whether an intervention was *correct* needs the outcome of the trade it reduced; "
        f"the log holds {settled_trades} settled trade(s), so that is undefined rather than zero"
    ) if settled_trades == 0 else (
        f"outcome effect is computable over {settled_trades} settled trade(s) and is not yet "
        f"implemented; it is reported as undefined rather than guessed"
    )
    return RiskAudit(
        decisions=len(rows),
        interventions=tuple(interventions),
        positions_offered=offered,
        outcome_effect_note=note,
    )


def read_records(path: Path) -> list[dict[str, Any]]:
    """Read the risk sidecar. A missing file is no records, which the audit reports as untested."""
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            out.append(row)
    return out


def audit(path: Path, *, settled_trades: int = 0) -> RiskAudit:
    return from_records(read_records(path), settled_trades=settled_trades)


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    from argus.paper.ledger import PaperLedger

    root = Path(__file__).resolve().parents[3] / "data"
    parser = argparse.ArgumentParser(description="Audit the risk layer over the decision record.")
    parser.add_argument("--records", type=Path, default=root / "risk_records.jsonl")
    parser.add_argument("--ledger", type=Path, default=root / "paper_ledger.jsonl")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    settled = 0
    if args.ledger.exists():
        settled = sum(1 for e in PaperLedger(path=args.ledger).entries if e.is_settled)

    report = audit(args.records, settled_trades=settled)
    if args.json:
        print(json.dumps(report.as_dict(), indent=2))
    else:
        for line in report.render():
            print(line)
    # Written on every run, not only under --json: the artefact is the thing a judge can open.
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report.as_dict(), indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nwritten to {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DOMINANT_RATE",
    "LIGHT_RATE",
    "MIN_DECISIONS_FOR_A_VERDICT",
    "Exercise",
    "Intervention",
    "RiskAudit",
    "audit",
    "from_records",
    "main",
    "read_records",
]
