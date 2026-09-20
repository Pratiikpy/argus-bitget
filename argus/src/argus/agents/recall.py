"""Episodic memory — what this desk did on this symbol before, and what it cost.

The agent-architecture audit (`research/architecture/agent-architecture-audit.md:207`) named this
the clearest gap in the reasoning layer: "no episodic retrieval", "each cycle is isolated", "every
trade on a repeated symbol starts from scratch". It also judged the payoff uncertain, on the
grounds that a two-week submission might never see a symbol twice. That reservation has expired —
the ledger holds 121 decisions across twelve instruments and NVDA alone appears dozens of times.

**Where this departs from the reference, and why it is better.** ``TauricResearch/TradingAgents``
implements the same idea as a reflection log: after a decision, a model writes a lesson in prose,
and later cycles read the prose back. That is memory of *what a model said about what happened*,
and it inherits every failure of the model that wrote it — an incorrect lesson is indistinguishable
from a correct one, and nothing can ever falsify it.

Ours reads the hash-chained ledger instead, and every lesson here is **arithmetic over graded
outcomes**. "You abstained on this symbol eleven times in weekend sessions; the median move you
passed on was -18bps and two of eleven exceeded the hurdle" is a computation anyone can redo from
`data/paper_ledger.jsonl`. No model writes it, so no model can be wrong about it.

Three guards, each protecting against a way this becomes a lie:

* **Point in time.** Only episodes *decided* strictly before the moment of recall are visible, and
  an episode's outcome counts only if it *settled* before that moment too. An episode decided last
  week but settled tomorrow contributes its decision and not its result. Without this split the
  memory would quietly leak the future into the past, which is the single most common way a
  backtested "learning" agent produces its improvement.
* **Only graded episodes become lessons.** An unsettled decision teaches nothing, however recent.
  It still appears as context, marked ungraded, because knowing the desk has looked at this symbol
  nine times this week is itself worth knowing.
* **A floor before any pattern is stated.** Below :data:`MIN_EPISODES_FOR_A_LESSON` the module
  reports the episodes and states no pattern — the same discipline the calibration scorecard
  already applies, and for the same reason: three observations wearing a percentage is not a rate.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any

MIN_EPISODES_FOR_A_LESSON = 5
"""Fewest graded episodes before this module will state a pattern.

Five, matching the calibration floor in `argus.eval.observatory`. Below it the episodes are listed
and nothing is concluded from them. A memory that generalises from two observations is worse than
no memory, because it produces a confident sentence the reasoning layer has no way to discount.
"""

RECALL_LIMIT = 8
"""How many episodes reach the prompt.

Bounded because the frame is bounded: a memory that grows without limit eventually crowds out the
market state it is supposed to contextualise. Graded episodes are preferred over ungraded ones
within the limit, since an ungraded episode carries strictly less information.
"""


def _decimal(raw: str | None) -> Decimal | None:
    if raw is None:
        return None
    try:
        return Decimal(raw)
    except (InvalidOperation, ValueError):
        return None


def _moment(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


@dataclass(frozen=True, slots=True)
class Episode:
    """One past decision on this symbol, as it may legitimately be seen from ``now``."""

    seq: int
    decided_at: datetime
    verdict: str
    session_phase: str
    stated_confidence: float
    thesis: str
    graded: bool
    """True only when this episode settled **before** the moment of recall."""

    net_pnl: Decimal | None = None
    direction_correct: bool | None = None
    counterfactual_move_bps: Decimal | None = None

    @property
    def is_abstention(self) -> bool:
        return self.verdict == "no_trade"

    @property
    def outcome_phrase(self) -> str:
        if not self.graded:
            return "not yet settled"
        if self.is_abstention:
            if self.counterfactual_move_bps is None:
                return "abstained, move not recorded"
            return f"abstained; the market then moved {self.counterfactual_move_bps:+.1f}bps"
        if self.net_pnl is None:
            return "traded, outcome not recorded"
        right = "right" if self.direction_correct else "wrong"
        return f"traded and was {right}; net {self.net_pnl:+}"

    def render(self) -> str:
        return (
            f"seq {self.seq} ({self.decided_at.date().isoformat()}, {self.session_phase}): "
            f"{self.verdict} at confidence {self.stated_confidence:.2f} — {self.outcome_phrase}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "decided_at": self.decided_at.isoformat(),
            "verdict": self.verdict,
            "session_phase": self.session_phase,
            "stated_confidence": self.stated_confidence,
            "graded": self.graded,
            "outcome": self.outcome_phrase,
        }


@dataclass(frozen=True)
class Recall:
    """What this desk can honestly remember about this symbol at this instant."""

    symbol: str
    as_of: datetime
    episodes: tuple[Episode, ...]
    considered: int
    """How many past decisions on this symbol existed, before the recall limit was applied."""

    @property
    def graded(self) -> tuple[Episode, ...]:
        return tuple(e for e in self.episodes if e.graded)

    @property
    def abstentions(self) -> tuple[Episode, ...]:
        return tuple(e for e in self.graded if e.is_abstention)

    @property
    def has_enough_to_generalise(self) -> bool:
        return len(self.graded) >= MIN_EPISODES_FOR_A_LESSON

    def passed_moves_bps(self) -> tuple[Decimal, ...]:
        return tuple(
            e.counterfactual_move_bps
            for e in self.abstentions
            if e.counterfactual_move_bps is not None
        )

    def median_passed_move_bps(self) -> Decimal | None:
        moves = sorted(abs(m) for m in self.passed_moves_bps())
        if not moves:
            return None
        mid = len(moves) // 2
        return moves[mid] if len(moves) % 2 else (moves[mid - 1] + moves[mid]) / 2

    def abstentions_that_beat(self, hurdle_bps: Decimal) -> int:
        """How many passes would have cleared the hurdle in absolute terms.

        Absolute, not signed: an abstention that avoided a large adverse move and one that missed
        a large favourable move are both evidence that the tape was moving, which is the question
        this number answers. Whether the pass was *correct* is the scorecard's job, not memory's.
        """
        return sum(1 for m in self.passed_moves_bps() if abs(m) > hurdle_bps)

    def lessons(self, *, hurdle_bps: Decimal) -> tuple[str, ...]:
        """Arithmetic statements about the record. No model writes these.

        Empty below the floor, deliberately: the caller then renders the episodes without a
        pattern, rather than a pattern the record cannot support.
        """
        if not self.has_enough_to_generalise:
            return ()
        out: list[str] = []
        passes = self.passed_moves_bps()
        if passes:
            median = self.median_passed_move_bps()
            beat = self.abstentions_that_beat(hurdle_bps)
            out.append(
                f"You stood aside on {self.symbol} {len(passes)} time(s) that have since been "
                f"graded. The median move you passed on was {median:.1f}bps in absolute terms, "
                f"and {beat} of {len(passes)} exceeded the {hurdle_bps}bps hurdle. "
                + (
                    "Most of those passes were on a tape that did not pay a round trip."
                    if beat * 2 <= len(passes) else
                    "More than half of those passes were on a tape that did move far enough to "
                    "pay a round trip, so the binding constraint was direction, not size."
                )
            )
        traded = [e for e in self.graded if not e.is_abstention and e.direction_correct is not None]
        if len(traded) >= MIN_EPISODES_FOR_A_LESSON:
            right = sum(1 for e in traded if e.direction_correct)
            out.append(
                f"Of {len(traded)} graded position(s) taken on {self.symbol}, "
                f"{right} went the stated way."
            )
        phases = {e.session_phase for e in self.graded}
        if len(phases) == 1:
            out.append(
                f"Every graded decision on {self.symbol} was made in a {phases.pop()} session, so "
                f"nothing here generalises to another session phase."
            )
        return tuple(out)

    def render(self, *, hurdle_bps: Decimal) -> str:
        if not self.episodes:
            return (
                f"MEMORY — no prior decision on {self.symbol} before this moment. "
                f"This is the first look, and nothing below is inherited."
            )
        lines = [
            f"MEMORY — {len(self.episodes)} prior decision(s) on {self.symbol} "
            f"(of {self.considered} on record before now; "
            f"{len(self.graded)} have since been graded):"
        ]
        lines.extend(f"  {e.render()}" for e in self.episodes)
        lessons = self.lessons(hurdle_bps=hurdle_bps)
        if lessons:
            lines.append("What the record says, computed from it rather than recalled:")
            lines.extend(f"  - {lesson}" for lesson in lessons)
        else:
            lines.append(
                f"  Fewer than {MIN_EPISODES_FOR_A_LESSON} graded episode(s), so no pattern is "
                f"stated. These are observations, not a lesson."
            )
        lines.append(
            "  Past decisions are context, never permission: an identical setup that worked twice "
            "is two observations, and this desk has abstained correctly far more often than it "
            "has traded."
        )
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "as_of": self.as_of.isoformat(),
            "considered": self.considered,
            "returned": len(self.episodes),
            "graded": len(self.graded),
            "episodes": [e.as_dict() for e in self.episodes],
        }


def recall(
    entries: Sequence[Any],
    *,
    symbol: str,
    now: datetime,
    limit: int = RECALL_LIMIT,
    session_phase: str = "",
) -> Recall:
    """Past decisions on ``symbol``, as they may legitimately be seen from ``now``.

    ``entries`` is any sequence of :class:`argus.paper.ledger.Entry`-shaped records; the ledger is
    not imported here so this module can be driven from a fixture without a file on disk.

    ``session_phase`` prefers episodes from the same phase when it is supplied. Prefers rather than
    filters: a weekend decision on this symbol is still worth seeing during regular hours, and
    hiding it would let a preference become a blind spot — the same rule the mandate applies to
    evidence.
    """
    candidates: list[Episode] = []
    for entry in entries:
        if entry.symbol != symbol:
            continue
        decided = _moment(entry.decided_at)
        if decided is None or decided >= now:
            continue  # strictly before: a decision cannot remember itself
        settled = _moment(entry.settled_at)
        graded = settled is not None and settled < now
        candidates.append(
            Episode(
                seq=entry.seq,
                decided_at=decided,
                verdict=entry.verdict,
                session_phase=entry.session_phase,
                stated_confidence=entry.stated_confidence,
                thesis=entry.thesis,
                graded=graded,
                # The outcome fields are read ONLY when the episode settled before `now`. Reading
                # them otherwise is the look-ahead this module exists to make impossible.
                net_pnl=_decimal(entry.net_pnl) if graded else None,
                direction_correct=entry.direction_correct if graded else None,
                counterfactual_move_bps=(
                    _decimal(entry.counterfactual_move_bps) if graded else None
                ),
            )
        )

    def rank(episode: Episode) -> tuple[int, int, datetime]:
        same_phase = bool(session_phase) and episode.session_phase == session_phase
        # Graded first, then same-phase, then most recent. Sorted descending on every key.
        return (int(episode.graded), int(same_phase), episode.decided_at)

    chosen = sorted(candidates, key=rank, reverse=True)[:limit]
    return Recall(
        symbol=symbol,
        as_of=now,
        episodes=tuple(sorted(chosen, key=lambda e: e.decided_at, reverse=True)),
        considered=len(candidates),
    )


__all__ = [
    "MIN_EPISODES_FOR_A_LESSON",
    "RECALL_LIMIT",
    "Episode",
    "Recall",
    "recall",
]
