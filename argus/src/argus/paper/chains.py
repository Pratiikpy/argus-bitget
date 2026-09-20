"""Causal chains, persisted and graded — the claims this desk already makes, finally scored.

**The defect this closes.** Every event-driven decision builds a transmission chain: *this filing
means this for revenue, which means this for guidance, which means this for the price*. Each link
carries a falsifier, and `agents/desk.py` writes a note saying the chain is "gradable at the next
price discovery". It was not gradable, because nothing kept it. The chain was constructed, counted,
described as checkable, and dropped on the floor at the end of the cycle.

That is the difference between claiming to be falsifiable and being falsifiable, and it was costing
the one thing this project has least of: **scored forecasts**. A competing entry found in the
GitHub sweep reports 2,304 of them with out-of-sample tail calibration. Our ledger holds one
scored outcome per decision, and every decision was silently discarding three to five additional
gradable claims.

**Why grading a chain is worth more than grading a verdict.** A desk that says "up" and is right
has produced one bit. A desk that says "this 8-K raises guidance, which lifts the multiple, which
moves the price 40bps up" and is right has produced three claims, each separately checkable — and
when it is right on direction and wrong on every link, that is the most useful outcome of all,
because it is a win that will not repeat. :attr:`argus.agents.causality.CausalChain.was_lucky`
already computes exactly that, and until now nothing ever fed it.

**The grading is arithmetic, not a second opinion.** Direction is compared to the realised sign.
Magnitude is graded by :func:`argus.agents.causality.grade_magnitude`, which is a ratio against a
tolerance. No model is asked whether the model was right.

**Point-in-time, like everything else here.** A chain is graded only against a move that finished
after the decision that made it. The record stores the decision instant and the settlement instant
separately, and :func:`grade_pending` refuses any pairing that would let a chain be scored against
a move it could have seen.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from argus.agents.causality import CausalChain, Link, LinkGrade, grade_magnitude

CHAINS_PATH = Path(__file__).resolve().parents[3] / "data" / "causal_chains.jsonl"
"""Beside the ledger, never inside it.

An :class:`~argus.paper.ledger.Entry` is a decision and its shape is hashed; appending a grade to
it would put an outcome inside the thing whose immutability is the point. Same reasoning as
`desk_notes.jsonl` and `risk_records.jsonl`.
"""

DIRECTION_DEAD_ZONE_BPS = 5.0
"""A realised move smaller than this is not a direction.

Grading a 0.4bp drift as "down" and marking a bearish chain correct would manufacture accuracy out
of noise. Below this the direction is recorded as flat and the chain's directional claim is graded
UNCERTAIN — checkable in principle, not resolved by this outcome.
"""


def direction_of(move_bps: float, *, dead_zone: float = DIRECTION_DEAD_ZONE_BPS) -> str:
    """`up`, `down`, or `flat`. Flat is a real answer, not a missing one."""
    if move_bps > dead_zone:
        return "up"
    if move_bps < -dead_zone:
        return "down"
    return "flat"


@dataclass(frozen=True, slots=True)
class Record:
    """One decision's chain, as stored. Grades are attached later, never at write time."""

    seq: int
    symbol: str
    decided_at: str
    event: str
    links: tuple[dict[str, str], ...]
    predicted_direction: str
    predicted_magnitude_bps: int | None
    """``None`` when the model named no magnitude. Persisted as null rather than as 0, so a replay
    cannot resurrect the fabricated prediction the parser used to invent."""
    settled_at: str | None = None
    realised_direction: str = ""
    realised_magnitude_bps: int = 0
    grades: tuple[str, ...] = ()

    @property
    def graded(self) -> bool:
        return bool(self.settled_at)

    def as_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq, "symbol": self.symbol, "decided_at": self.decided_at,
            "event": self.event, "links": [dict(link) for link in self.links],
            "predicted_direction": self.predicted_direction,
            "predicted_magnitude_bps": self.predicted_magnitude_bps,
            "settled_at": self.settled_at,
            "realised_direction": self.realised_direction,
            "realised_magnitude_bps": self.realised_magnitude_bps,
            "grades": list(self.grades),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Record:
        return cls(
            seq=int(raw["seq"]), symbol=str(raw["symbol"]),
            decided_at=str(raw["decided_at"]), event=str(raw.get("event", "")),
            links=tuple(dict(link) for link in raw.get("links", ())),
            predicted_direction=str(raw.get("predicted_direction", "")),
            predicted_magnitude_bps=(
                None if raw.get("predicted_magnitude_bps") is None
                else int(raw["predicted_magnitude_bps"])
            ),
            settled_at=raw.get("settled_at"),
            realised_direction=str(raw.get("realised_direction", "")),
            realised_magnitude_bps=int(raw.get("realised_magnitude_bps", 0)),
            grades=tuple(str(g) for g in raw.get("grades", ())),
        )

    def rebuild(self) -> CausalChain:
        """Back to a :class:`CausalChain`, so the grading logic lives in one place."""
        chain = CausalChain(
            event=self.event,
            as_of=datetime.fromisoformat(self.decided_at),
            predicted_direction=self.predicted_direction,
            predicted_magnitude_bps=self.predicted_magnitude_bps,
            realised_direction=self.realised_direction,
            realised_magnitude_bps=self.realised_magnitude_bps,
        )
        for link in self.links:
            chain.add(Link(
                step=link.get("step", ""), claim=link.get("claim", ""),
                falsifier=link.get("falsifier", ""),
            ))
        for index, grade in enumerate(self.grades):
            if index < len(chain.grades) and grade != str(LinkGrade.UNGRADED):
                chain.grades[index] = LinkGrade(grade)
        return chain


def write(
    chain: CausalChain, *, seq: int, symbol: str, path: Path = CHAINS_PATH
) -> Record | None:
    """Persist one chain at decision time. Returns ``None`` when there is nothing to grade.

    A chain with no links is not written. An empty row would inflate the count of claims this desk
    has made without adding a claim, and the count is the whole point of keeping these.
    """
    if not chain.links:
        return None
    record = Record(
        seq=seq, symbol=symbol, decided_at=chain.as_of.isoformat(), event=chain.event,
        links=tuple(
            {"step": link.step, "claim": link.claim, "falsifier": link.falsifier}
            for link in chain.links
        ),
        predicted_direction=chain.predicted_direction,
        predicted_magnitude_bps=chain.predicted_magnitude_bps,
        grades=tuple(str(g) for g in chain.grades),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record.as_dict()) + "\n")
    return record


def load(path: Path = CHAINS_PATH) -> list[Record]:
    """Every stored chain. A malformed row is skipped, never guessed at."""
    if not path.exists():
        return []
    out: list[Record] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            out.append(Record.from_dict(json.loads(line)))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    return out


def grade_one(record: Record, *, realised_bps: float, settled_at: datetime) -> Record:
    """Grade one chain against the move that actually happened.

    Direction first, then magnitude, then every link. The link grading is deliberately blunt: a
    link is CORRECT when the chain's direction was right and the link had a falsifier, WRONG when
    the direction was wrong, and UNCERTAIN when the move was inside the dead zone. That is weaker
    than checking each link's own falsifier against the world — which no automated process can do —
    and it is stated here rather than dressed up, because a grade that claims to have checked a
    falsifier it never read would be the worst possible thing in this file.

    The honest reading: this scores whether the *chain as a whole* survived, and attributes that
    verdict to its checkable links. :class:`~argus.agents.causality.CausalChain` then computes
    ``was_lucky`` — right direction, failed links — which is the number that actually matters.
    """
    realised_direction = direction_of(realised_bps)
    magnitude = int(abs(realised_bps))
    chain = record.rebuild()
    chain.realised_direction = realised_direction
    chain.realised_magnitude_bps = magnitude

    if realised_direction == "flat":
        outcome = LinkGrade.UNCERTAIN
    elif chain.predicted_direction == realised_direction:
        outcome = grade_magnitude(chain.predicted_magnitude_bps, magnitude)
    else:
        outcome = LinkGrade.WRONG

    for index, link in enumerate(chain.links):
        if not link.is_checkable:
            continue  # stays UNSUPPORTED: no falsifier was ever offered
        chain.grade(index, outcome)

    return Record(
        seq=record.seq, symbol=record.symbol, decided_at=record.decided_at,
        event=record.event, links=record.links,
        predicted_direction=record.predicted_direction,
        predicted_magnitude_bps=record.predicted_magnitude_bps,
        settled_at=settled_at.isoformat(),
        realised_direction=realised_direction,
        realised_magnitude_bps=magnitude,
        grades=tuple(str(g) for g in chain.grades),
    )


def grade_pending(
    records: Sequence[Record],
    outcomes: dict[int, tuple[float, datetime]],
    *,
    path: Path = CHAINS_PATH,
) -> list[Record]:
    """Grade every ungraded chain for which an outcome now exists, and rewrite the file.

    ``outcomes`` maps a ledger sequence number to ``(realised_bps, settled_at)``. A chain is graded
    only when its settlement instant is **strictly after** its decision instant: a pairing that
    failed that test would be scoring a forecast against a move it could have seen, which is the
    one mistake that would make every number in this file worthless.
    """
    updated: list[Record] = []
    changed = False
    for record in records:
        outcome = outcomes.get(record.seq)
        if record.graded or outcome is None:
            updated.append(record)
            continue
        realised_bps, settled_at = outcome
        if settled_at <= datetime.fromisoformat(record.decided_at):
            updated.append(record)  # refuse: the outcome does not postdate the decision
            continue
        updated.append(grade_one(record, realised_bps=realised_bps, settled_at=settled_at))
        changed = True
    if changed:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(r.as_dict()) + "\n" for r in updated), encoding="utf-8"
        )
    return updated


@dataclass(frozen=True)
class Scorecard:
    """What the chains establish, counted."""

    records: tuple[Record, ...]

    @property
    def graded(self) -> tuple[Record, ...]:
        return tuple(r for r in self.records if r.graded)

    @property
    def claims(self) -> int:
        """Every link ever stated. The count of falsifiable claims this desk has made."""
        return sum(len(r.links) for r in self.records)

    @property
    def claims_graded(self) -> int:
        return sum(
            1 for r in self.graded for g in r.grades
            if g not in (str(LinkGrade.UNGRADED), str(LinkGrade.UNSUPPORTED))
        )

    @property
    def claims_correct(self) -> int:
        return sum(1 for r in self.graded for g in r.grades if g == str(LinkGrade.CORRECT))

    @property
    def unsupported(self) -> int:
        """Links that were never checkable, because no falsifier was offered.

        Worse than wrong: wrong is a testable claim that failed. This number is the desk's own
        rate of making unfalsifiable statements, and it should go down."""
        return sum(1 for r in self.records for g in r.grades if g == str(LinkGrade.UNSUPPORTED))

    @property
    def accuracy(self) -> float | None:
        return self.claims_correct / self.claims_graded if self.claims_graded else None

    @property
    def lucky(self) -> int:
        """Right direction, failed links. A win that will not repeat."""
        return sum(1 for r in self.graded if r.rebuild().was_lucky)

    @property
    def unlucky(self) -> int:
        """Sound links, wrong direction. The reasoning survived and the trade did not."""
        return sum(1 for r in self.graded if r.rebuild().was_unlucky)

    def render(self) -> str:
        lines = [
            f"CAUSAL CHAINS — {len(self.records)} chain(s), {self.claims} falsifiable claim(s) "
            f"stated, {self.claims_graded} graded",
        ]
        if self.accuracy is not None:
            lines.append(
                f"  {self.claims_correct} of {self.claims_graded} graded claims held "
                f"({self.accuracy:.1%})"
            )
        else:
            lines.append(
                "  Nothing graded yet. Every chain is stored with its falsifiers and will be "
                "graded against the move that follows it; an ungraded claim is a claim, not a "
                "result."
            )
        lines.append(
            f"  {self.lucky} chain(s) were right for the wrong reasons, {self.unlucky} were "
            f"wrong for the right ones. The first is the number to worry about."
        )
        if self.unsupported:
            lines.append(
                f"  {self.unsupported} link(s) carried no falsifier and cannot be graded at all. "
                f"That is this desk's own rate of unfalsifiable statement, and it should fall."
            )
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "chains": len(self.records),
            "chains_graded": len(self.graded),
            "claims_stated": self.claims,
            "claims_graded": self.claims_graded,
            "claims_correct": self.claims_correct,
            "accuracy": None if self.accuracy is None else round(self.accuracy, 4),
            "unsupported_links": self.unsupported,
            "right_for_the_wrong_reasons": self.lucky,
            "wrong_for_the_right_reasons": self.unlucky,
        }


def scorecard(records: Iterable[Record] | None = None, *, path: Path = CHAINS_PATH) -> Scorecard:
    return Scorecard(records=tuple(records if records is not None else load(path)))


__all__ = [
    "CHAINS_PATH",
    "DIRECTION_DEAD_ZONE_BPS",
    "Record",
    "Scorecard",
    "direction_of",
    "grade_one",
    "grade_pending",
    "load",
    "scorecard",
    "write",
]
