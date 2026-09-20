"""Is the desk right about direction, while it is correctly refusing to trade?

The hurdle frontier put a number on the question that decides whether abstaining is wisdom or
paralysis: over the recorded instants, **trading beats abstaining at 56% directional accuracy**.
Nothing on record said whether this desk clears that bar, and the reason is not that the data was
missing — it is that the desk never committed to a view it could be graded on.

**That was assumed for weeks and then measured.** The decision contract has always required a
``side``, so every abstention carried one and it looked like a view. Over the 53 settled
abstentions carrying a counterfactual move it was **BUY in all 53**, directionally right 3.8% of
the time against a base rate of up-moves of exactly 3.8%. A field that matches the base rate to the
decimal carries no information: asked for a side on a decision it has declined to make, the model
fills the schema in.

So the desk now states a **lean** — the direction it would take if forced — on every decision
including the ones it refuses. It costs no risk, it is hashed with the decision through the intent
hash, and it is settled against the move that actually followed. This module grades it.

**What it will not do.** Until leans exist in the record, every figure here is UNDEFINED and says
so. The temptation is to fall back on ``side``, which is present on all 161 rows and would produce
a number immediately — a number that measures the tape and not the desk. That fallback is
deliberately absent, and :func:`grade` refuses rows whose lean is ``none`` rather than guessing one
from the side.

**Every settled decision, not only the refusals.** An abstention's move is written at settlement as
``counterfactual_move_bps``; a trade's is its exit price against its entry. :func:`move_of` reads
both, so the record is of the desk rather than of its abstentions — and it says which kind each call
came from, because a score built entirely out of one is a narrower claim than its headline.

**A lean of "none" is an answer, not a gap.** It means the desk could not call the direction, which
is different from calling it and being wrong. Both are recorded; only the second is scored for
accuracy, and the share of "none" is reported beside the accuracy because a desk that declines to
lean on everything has not earned a high score on the few it did.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from statistics import fmean
from typing import Any

DATA = Path(__file__).resolve().parents[3] / "data"
REPORT_PATH = DATA / "shadow_record.json"

MIN_GRADED = 20
"""Fewest graded leans before an accuracy is reported.

A directional accuracy from ten calls has a standard error near 16 percentage points, which is
wider than the entire gap between a useful desk and a coin. Below this the record says UNDEFINED.
"""

DEAD_ZONE_BPS = 5.0
"""Moves smaller than this are not scored either way.

A lean is a claim about direction, and a two-basis-point move is not a direction — it is the tape
being flat. Scoring those as wins or losses adds noise to the numerator and the denominator at once.
"""


class ShadowError(ValueError):
    """Raised rather than reporting a directional accuracy that has nothing behind it."""


@dataclass(frozen=True, slots=True)
class Call:
    """One graded lean: what the desk said, and what the market then did."""

    seq: int
    symbol: str
    lean: str
    confidence: float
    move_bps: float
    source: str
    decided_on: date | None = None
    """When the decision was made. Carried so the graded window can be bounded and handed to
    `eval/leakage.py:check_window` — a record cannot say whether a model had already read the
    period's news unless it knows which period it covers."""
    """Where the move came from: ``counterfactual`` on an abstention, ``fill`` on a trade.

    Recorded rather than assumed, because the two are not equally available. Only abstentions carry
    ``counterfactual_move_bps``; a settled trade carries an exit price instead, and a record that
    quietly graded one and not the other would be a record of abstentions calling itself a record of
    the desk.
    """

    @property
    def scored(self) -> bool:
        """Only a real lean against a real move is scored."""
        return self.lean in {"up", "down"} and abs(self.move_bps) > DEAD_ZONE_BPS

    @property
    def correct(self) -> bool | None:
        if not self.scored:
            return None
        return (self.move_bps > 0) if self.lean == "up" else (self.move_bps < 0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq, "symbol": self.symbol, "lean": self.lean,
            "confidence": round(self.confidence, 4), "move_bps": round(self.move_bps, 2),
            "source": self.source, "scored": self.scored, "correct": self.correct,
            "decided_on": None if self.decided_on is None else self.decided_on.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class ShadowRecord:
    """The desk's directional judgement while it was standing aside."""

    calls: tuple[Call, ...]
    break_even: float
    """The accuracy at which trading would have beaten abstaining, from the hurdle frontier."""

    @property
    def window(self) -> tuple[date, date] | None:
        """First and last decision date among the graded calls."""
        dates = sorted(c.decided_on for c in self.calls if c.decided_on is not None)
        return (dates[0], dates[-1]) if dates else None

    @property
    def leakage(self) -> Any:
        """Whether the model had already read the period this record grades.

        The gate `eval/leakage.py` was built to feed. It is **advisory and conservative**: what the
        instrument measures is recall of published *fundamentals*, and what this record grades is
        *price direction*. Overlap is the precondition for leakage, not proof of it — barj28's own
        finding is that capacity and realisation come apart. So the note is attached to the verdict
        rather than used to suppress it, and a reader can weigh both.
        """
        from argus.eval.leakage import check_window

        span = self.window
        if span is None:
            return None
        return check_window(span[0], span[1])

    @property
    def declined_to_lean(self) -> int:
        return sum(1 for c in self.calls if c.lean == "none")

    @property
    def flat_tape(self) -> int:
        return sum(
            1 for c in self.calls if c.lean in {"up", "down"} and abs(c.move_bps) <= DEAD_ZONE_BPS
        )

    @property
    def scored(self) -> tuple[Call, ...]:
        return tuple(c for c in self.calls if c.scored)

    @property
    def by_source(self) -> dict[str, int]:
        """How many scored calls came from an abstention and how many from a fill.

        Reported because a record made entirely of one kind is a narrower claim than it looks, and
        this desk abstains far more than it trades.
        """
        out: dict[str, int] = {}
        for call in self.scored:
            out[call.source] = out.get(call.source, 0) + 1
        return out

    @property
    def accuracy(self) -> float | None:
        rows = self.scored
        if len(rows) < MIN_GRADED:
            return None
        return sum(1 for c in rows if c.correct) / len(rows)

    @property
    def base_rate(self) -> float | None:
        """How often the move was up. An accuracy equal to this carries no information."""
        rows = self.scored
        if len(rows) < MIN_GRADED:
            return None
        return sum(1 for c in rows if c.move_bps > 0) / len(rows)

    @property
    def clears_break_even(self) -> bool | None:
        return None if self.accuracy is None else self.accuracy >= self.break_even

    @property
    def verdict(self) -> str:
        if not self.calls:
            return (
                "UNDEFINED: no settled decision is on the record yet, so there is nothing to grade "
                "a lean against. Unmeasured rather than zero"
            )
        if self.accuracy is None:
            return (
                f"UNDEFINED: {len(self.scored)} scored call(s) against a floor of {MIN_GRADED}. "
                f"{self.declined_to_lean} decision(s) declined to lean and {self.flat_tape} met a "
                f"tape flatter than {DEAD_ZONE_BPS:.0f}bps. An accuracy from this few has a "
                f"standard error wider than the gap it would be measuring"
            )
        rows = self.scored
        head = (
            f"{self.accuracy:.1%} directional accuracy over {len(rows)} scored lean(s), against a "
            f"break-even of {self.break_even:.1%} and a base rate of {self.base_rate:.1%} up-moves."
        )
        if self.base_rate is not None and abs(self.accuracy - self.base_rate) < 0.02:
            return (
                f"{head} That is indistinguishable from always calling the more common direction, "
                f"which is the same finding the `side` field produced and the reason the lean "
                f"exists"
            )
        if self.clears_break_even:
            return (
                f"{head} The desk clears the bar at which trading beats abstaining, so its "
                f"abstentions are costing its owner money and the confidence gate is too tight"
            )
        return (
            f"{head} The desk does not clear the bar at which trading would beat abstaining, so "
            f"standing aside is the correct policy on this evidence rather than merely a cautious "
            f"one"
        )

    def render(self) -> str:
        lines = [
            f"SHADOW RECORD — {len(self.calls)} decision(s), {len(self.scored)} scored",
            "",
            f"  declined to lean : {self.declined_to_lean}",
            f"  tape too flat    : {self.flat_tape}",
        ]
        if self.accuracy is not None:
            lines += [
                f"  accuracy         : {self.accuracy:.1%}",
                f"  base rate        : {self.base_rate:.1%}",
                f"  break-even       : {self.break_even:.1%}",
            ]
        lines += ["", f"  {self.verdict}"]
        check = self.leakage
        if check is not None:
            lines += ["", f"  [leakage] {check.status.value.upper()}: {check.note}"]
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "generated_at": datetime.now(UTC).isoformat(),
            "decisions": len(self.calls),
            "scored": len(self.scored),
            "declined_to_lean": self.declined_to_lean,
            "flat_tape": self.flat_tape,
            "accuracy": None if self.accuracy is None else round(self.accuracy, 5),
            "base_rate": None if self.base_rate is None else round(self.base_rate, 5),
            "break_even": round(self.break_even, 5),
            "by_source": self.by_source,
            "clears_break_even": self.clears_break_even,
            "mean_confidence": (
                round(fmean([c.confidence for c in self.scored]), 4) if self.scored else None
            ),
            "verdict": self.verdict,
            "window": (
                None if self.window is None
                else [self.window[0].isoformat(), self.window[1].isoformat()]
            ),
            "leakage": None if self.leakage is None else self.leakage.as_dict(),
            "calls": [c.as_dict() for c in self.calls],
        }


def move_of(row: Any) -> tuple[float, str] | None:
    """The move that followed a decision, and where the number came from.

    Two recorded shapes, because the ledger settles the two verdict classes differently:

    * an **abstention** carries ``counterfactual_move_bps``, written by ``settle_abstention`` at
      settlement — the move as it happened, signed, not as a judgement;
    * a **trade** carries an ``exit_price`` instead, and the same move is
      ``(exit - entry) / entry * 10000``.

    The second is computed here rather than stored, and that is the identical arithmetic
    ``ledger.settle_abstention`` performs (`paper/ledger.py:383`) on fields of equal standing — both
    prices are settlement fields, neither is hashed, and neither is re-fetched from a price series
    that may since have been revised. Restricting the record to the first would have made it a
    measurement of abstentions wearing the name of the desk.

    Returns ``None`` for anything unsettled, which is not a gap in the lean but an outcome that does
    not exist yet.
    """
    raw = getattr(row, "counterfactual_move_bps", None)
    if raw is not None:
        try:
            return float(raw), "counterfactual"
        except (TypeError, ValueError):
            return None
    exit_price = getattr(row, "exit_price", None)
    entry_price = getattr(row, "entry_price", None)
    if exit_price is None or entry_price is None:
        return None
    try:
        entry = float(entry_price)
        if entry <= 0:
            return None
        return (float(exit_price) - entry) / entry * 10_000.0, "fill"
    except (TypeError, ValueError):
        return None


def _decided_on(row: Any) -> date | None:
    """The decision's date, or ``None`` when the row cannot say.

    ``None`` rather than a fallback to today: a wrong date would place the graded window in a period
    the model may or may not have read, and the leakage gate would then answer a question about the
    wrong window with full confidence.
    """
    raw = getattr(row, "decided_at", None)
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw)).date()
    except ValueError:
        return None


def grade(rows: Sequence[Any], *, break_even: float) -> ShadowRecord:
    """Grade every recorded lean against the move that followed.

    ``rows`` are ledger entries. An unsettled row has no outcome and is skipped; a row whose lean is
    ``none`` is kept and counted as a declined lean, because a desk that declines to call direction
    on everything has not earned a score on the few it did.

    **The ``side`` field is never consulted.** It is present on every row and would produce a number
    immediately; that number was measured at 3.8% against a 3.8% base rate, so it measures the tape.
    """
    calls: list[Call] = []
    for row in rows:
        found = move_of(row)
        if found is None:
            continue
        move, source = found
        calls.append(Call(
            seq=int(getattr(row, "seq", 0)),
            symbol=str(getattr(row, "symbol", "?")),
            lean=str(getattr(row, "lean", "none")).strip().lower(),
            confidence=float(getattr(row, "lean_confidence", 0.0)),
            move_bps=move,
            source=source,
            decided_on=_decided_on(row),
        ))
    return ShadowRecord(calls=tuple(calls), break_even=break_even)


def main() -> int:  # pragma: no cover - CLI
    from argus.eval.hurdle import build as build_frontier
    from argus.paper.ledger import PaperLedger
    from argus.paper.runner import LEDGER_PATH

    ledger = PaperLedger(path=LEDGER_PATH)
    frontier = build_frontier()
    point = frontier.at_actual_hurdle
    break_even = point.break_even_accuracy or 0.5
    record = grade(ledger.entries, break_even=break_even)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(record.as_dict(), indent=2), encoding="utf-8")
    print(record.render())
    print(f"\nwritten to {REPORT_PATH}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = [
    "DEAD_ZONE_BPS",
    "MIN_GRADED",
    "Call",
    "ShadowError",
    "ShadowRecord",
    "grade",
    "move_of",
]
