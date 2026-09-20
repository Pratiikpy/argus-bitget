"""Is abstaining right? The question a desk with no trades has to answer, answered with arithmetic.

Every decision this desk has made is an abstention — 126 live, 30 replayed. The comfortable
explanation is that the deliberation-cost hurdle is too high for this universe, and the replay
harness was built to test exactly that by re-running the desk on regular-hours frames where the
anchor market was open. It abstained on all twenty of those too. So the pre-registered hypothesis
is refuted: the session phase was not the reason.

That leaves the real question, which no amount of further replaying answers: **was abstaining the
correct policy?** This module answers it without a single additional model call, because the answer
does not depend on the desk's opinion. It depends on two things that are already recorded — the
size of the moves that actually happened, and the size of the hurdle they had to clear.

**The oracle bound.** A trader with perfect foresight about direction still pays the hurdle. Their
net on an instant is ``|move| - hurdle``, and they simply decline when that is negative. Summed
over the record, that is the most any directional strategy could possibly have made. If it is
small, abstention costs almost nothing and the desk is right by construction.

**The break-even accuracy.** A trader who is right with probability ``p`` earns
``p(|r| - h) + (1 - p)(-|r| - h) = 2p|r| - |r| - h``, so break-even is at
``p = (|r| + h) / (2|r|)``. This is the number that settles the argument. It says what directional
accuracy the desk would need for trading to beat abstaining, given the moves that actually occurred
and the hurdle it actually faced — and it can be compared against the desk's *measured* accuracy
rather than an assumed one.

**What this found, and it is not what was expected.** Realised 24-hour absolute moves in the replay
average around 164bps against a total hurdle near 19bps. The hurdle is an order of magnitude
smaller than the moves. So the binding constraint is **not** the cost of deliberation — it is the
desk's own confidence gate. Abstention here is a statement about conviction, not about fees, and
saying otherwise would have been the comfortable answer rather than the true one. The verdict this
module emits names whichever constraint actually binds, and it is capable of naming ours.

**Sample size is handled, not assumed.** Thirty replayed instants across four symbols are not
thirty independent facts: instants sharing a timestamp share a market. The intraclass correlation
and Kish design effect from `eval.forecasts` are applied here rather than re-derived, so the
reported precision is the effective one.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean, median
from typing import Any

from argus.cost.model import CostModel
from argus.eval.forecasts import design_effect, intraclass_correlation

DATA = Path(__file__).resolve().parents[3] / "data"
REPLAY_PATH = DATA / "replay_ledger.jsonl"
LIVE_PATH = DATA / "paper_ledger.jsonl"
REPORT_PATH = DATA / "hurdle_frontier.json"

HURDLE_SWEEP_BPS = (0.0, 6.0, 12.0, 19.0, 30.0, 50.0, 100.0, 200.0, 400.0)
"""Hurdle levels to report the frontier at. 12 is the commission round trip, 19 the typical total
once deliberation is charged, and the tail extends past the largest observed move so the frontier
is shown all the way to the point where nothing clears."""

MIN_INSTANTS = 10
"""Below this the frontier is reported as INSUFFICIENT rather than drawn. A break-even accuracy
computed from five moves is a number with no sample behind it."""


class HurdleError(ValueError):
    """Raised rather than returning a frontier that cannot be computed."""


@dataclass(frozen=True, slots=True)
class Instant:
    """One decision point: what the hurdle was, and what the market then did."""

    symbol: str
    at: datetime
    move_bps: float
    """The realised move over the hold window, signed. Direction is the thing being tested, so the
    sign is kept even though the frontier uses the magnitude."""

    source: str
    """``replay`` or ``live``. Never mixed silently — the counts are reported separately."""


@dataclass(frozen=True, slots=True)
class FrontierPoint:
    """What every policy would have earned at one hurdle level."""

    hurdle_bps: float
    clearing_rate: float
    """Share of instants whose absolute move exceeded the hurdle. An instant below it cannot be
    traded profitably by anyone, in either direction."""

    oracle_net_bps: float
    """Mean net basis points for a perfect-foresight trader who declines when the move cannot clear
    the hurdle. The ceiling on any directional strategy."""

    oracle_total_bps: float
    break_even_accuracy: float | None
    """Directional accuracy needed to break even against abstaining, or None when no move clears
    the hurdle and no accuracy suffices."""

    coin_flip_net_bps: float
    """What a 50% trader earns: minus the hurdle, every time. Included because it is the honest
    null and it is always negative."""

    def as_dict(self) -> dict[str, Any]:
        return {
            "hurdle_bps": self.hurdle_bps,
            "clearing_rate": round(self.clearing_rate, 4),
            "oracle_net_bps": round(self.oracle_net_bps, 2),
            "oracle_total_bps": round(self.oracle_total_bps, 2),
            "break_even_accuracy": (
                None if self.break_even_accuracy is None else round(self.break_even_accuracy, 4)
            ),
            "coin_flip_net_bps": round(self.coin_flip_net_bps, 2),
        }


def break_even_accuracy(mean_abs_move_bps: float, hurdle_bps: float) -> float | None:
    """Directional accuracy at which trading stops losing to abstaining.

    ``p = (|r| + h) / (2|r|)``. Returns None when the hurdle is at least the mean move, because
    then the required accuracy is at or above 1.0 and no trader clears it — reporting 1.4 would
    imply a 140% hit rate is a thing that could be attempted.
    """
    if mean_abs_move_bps <= 0:
        return None
    p = (mean_abs_move_bps + hurdle_bps) / (2.0 * mean_abs_move_bps)
    return None if p >= 1.0 else p


def frontier_point(moves_bps: Sequence[float], hurdle_bps: float) -> FrontierPoint:
    """One hurdle level, evaluated against the moves that actually occurred."""
    if not moves_bps:
        raise HurdleError("no moves to evaluate; a frontier over an empty record is not a frontier")
    if hurdle_bps < 0:
        raise HurdleError("a negative hurdle is a subsidy, not a cost")
    magnitudes = [abs(m) for m in moves_bps]
    clears = [m for m in magnitudes if m > hurdle_bps]
    oracle = [max(0.0, m - hurdle_bps) for m in magnitudes]
    return FrontierPoint(
        hurdle_bps=hurdle_bps,
        clearing_rate=len(clears) / len(magnitudes),
        oracle_net_bps=fmean(oracle),
        oracle_total_bps=sum(oracle),
        break_even_accuracy=break_even_accuracy(fmean(magnitudes), hurdle_bps),
        coin_flip_net_bps=-hurdle_bps,
    )


@dataclass(frozen=True, slots=True)
class Frontier:
    """The whole answer, including the part that says how much to trust it."""

    instants: tuple[Instant, ...]
    points: tuple[FrontierPoint, ...]
    actual_hurdle_bps: float

    @property
    def moves(self) -> list[float]:
        return [i.move_bps for i in self.instants]

    @property
    def magnitudes(self) -> list[float]:
        return [abs(m) for m in self.moves]

    @property
    def clusters(self) -> list[list[float]]:
        """Instants grouped by timestamp. Two symbols scored on the same hour share a market."""
        by_time: dict[datetime, list[float]] = {}
        for item in self.instants:
            by_time.setdefault(item.at, []).append(abs(item.move_bps))
        return list(by_time.values())

    @property
    def icc(self) -> float:
        return intraclass_correlation(self.clusters)

    @property
    def effective_count(self) -> int:
        groups = [g for g in self.clusters if g]
        if not groups:
            return len(self.instants)
        average = sum(len(g) for g in groups) / len(groups)
        return max(1, int(len(self.instants) / design_effect(self.icc, average)))

    @property
    def at_actual_hurdle(self) -> FrontierPoint:
        return frontier_point(self.moves, self.actual_hurdle_bps)

    @property
    def binding_constraint(self) -> str:
        """Which thing actually stopped the desk trading — the hurdle, or its own conviction.

        This is the finding, and it is written so it can come out either way. The desk abstained on
        every instant. If the moves were small relative to the hurdle, the hurdle explains it and
        the desk was right for the stated reason. If the moves dwarf the hurdle, the stated reason
        is wrong whatever the decision's merits, and the confidence gate is what is binding.
        """
        if len(self.instants) < MIN_INSTANTS:
            return (
                f"INSUFFICIENT — {len(self.instants)} instant(s) is below the {MIN_INSTANTS} this "
                f"comparison needs; no constraint is named on this evidence"
            )
        point = self.at_actual_hurdle
        typical = median(self.magnitudes)
        ratio = typical / self.actual_hurdle_bps if self.actual_hurdle_bps > 0 else float("inf")
        if ratio < 2.0:
            return (
                f"THE HURDLE BINDS — the median absolute move of {typical:.0f}bps is only "
                f"{ratio:.1f}x the {self.actual_hurdle_bps:.0f}bps hurdle, so most instants cannot "
                f"be traded profitably in either direction and abstention is right for the reason "
                f"the desk gives"
            )
        accuracy = point.break_even_accuracy
        needed = (
            "no accuracy suffices" if accuracy is None
            else f"{accuracy:.1%} directional accuracy"
        )
        return (
            f"THE CONFIDENCE GATE BINDS, NOT THE HURDLE — the median absolute move of "
            f"{typical:.0f}bps is {ratio:.0f}x the {self.actual_hurdle_bps:.0f}bps hurdle, and "
            f"{point.clearing_rate:.0%} of instants clear it. Trading would beat abstaining at "
            f"{needed}. The desk's abstentions are therefore a statement about its conviction, "
            f"not about its costs, and any explanation that blames the fee is wrong"
        )

    def render(self) -> str:
        lines = [
            f"HURDLE FRONTIER — {len(self.instants)} instant(s), "
            f"{self.effective_count} effective after clustering (ICC {self.icc:.3f})",
            "",
            f"  median |move| {median(self.magnitudes):.0f}bps "
            f"· mean {fmean(self.magnitudes):.0f}bps "
            f"· max {max(self.magnitudes):.0f}bps · actual hurdle {self.actual_hurdle_bps:.1f}bps",
            "",
            f"{'hurdle':>9}{'clears':>9}{'oracle/instant':>17}"
            f"{'break-even acc':>17}{'coin flip':>12}",
        ]
        for point in self.points:
            accuracy = (
                "  none" if point.break_even_accuracy is None
                else f"{point.break_even_accuracy:.1%}"
            )
            lines.append(
                f"{point.hurdle_bps:>9.1f}{point.clearing_rate:>9.0%}"
                f"{point.oracle_net_bps:>16.0f}b{accuracy:>17}{point.coin_flip_net_bps:>11.0f}b"
            )
        lines += ["", f"  {self.binding_constraint}"]
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "generated_at": datetime.now(UTC).isoformat(),
            "instants": len(self.instants),
            "effective_instants": self.effective_count,
            "icc": round(self.icc, 5),
            "sources": {
                source: sum(1 for i in self.instants if i.source == source)
                for source in sorted({i.source for i in self.instants})
            },
            "actual_hurdle_bps": self.actual_hurdle_bps,
            "median_abs_move_bps": round(median(self.magnitudes), 2),
            "mean_abs_move_bps": round(fmean(self.magnitudes), 2),
            "max_abs_move_bps": round(max(self.magnitudes), 2),
            "binding_constraint": self.binding_constraint,
            "frontier": [p.as_dict() for p in self.points],
        }


def _load(path: Path, *, source: str, move_field: str) -> list[Instant]:
    if not path.exists():
        return []
    out: list[Instant] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        raw = row.get(move_field)
        if raw is None:
            continue
        try:
            move = float(raw)
        except (TypeError, ValueError):
            continue
        stamp = row.get("at") or row.get("decided_at")
        if not stamp:
            continue
        out.append(
            Instant(
                symbol=str(row.get("symbol", "?")),
                at=datetime.fromisoformat(str(stamp)),
                move_bps=move,
                source=source,
            )
        )
    return out


def load_instants(
    *, replay_path: Path = REPLAY_PATH, live_path: Path = LIVE_PATH
) -> list[Instant]:
    """Every decision point where a realised move is on record, from both ledgers.

    The live ledger stores the abstention's counterfactual under a different name than the replay
    ledger stores its realised move, which is deliberate — a counterfactual is what would have
    happened and a realised move is what did. For this measurement they answer the same question,
    and the source of each is carried through so the mix is never hidden.
    """
    return [
        *_load(replay_path, source="replay", move_field="realised_bps"),
        *_load(live_path, source="live", move_field="counterfactual_move_bps"),
    ]


def default_hurdle_bps(*, deliberation_bps: float = 6.8) -> float:
    """Commission round trip plus the typical deliberation charge.

    The deliberation term varies with session phase and volatility, so a single number is a
    representative value and not a constant of nature. 6.8 is the charge a full thinking budget
    incurs in regular hours at the 45% annualised volatility the desk assumes for a single-name
    rToken; off-hours it triples. Stated here rather than buried so the sensitivity is visible, and
    the frontier sweeps well past it in both directions anyway.
    """
    return float(CostModel.bitget_perp().round_trip_bps()) + deliberation_bps


def build(
    instants: Sequence[Instant] | None = None,
    *,
    hurdle_bps: float | None = None,
    sweep: Sequence[float] = HURDLE_SWEEP_BPS,
) -> Frontier:
    items = tuple(load_instants() if instants is None else instants)
    if not items:
        raise HurdleError("no instants with a recorded move; nothing to evaluate")
    actual = default_hurdle_bps() if hurdle_bps is None else hurdle_bps
    moves = [i.move_bps for i in items]
    # The actual hurdle is always on the frontier, but a sweep level within a tenth of a basis
    # point of it would print twice and read as two different answers to the same question.
    levels = sorted([actual, *(s for s in sweep if abs(s - actual) > 0.1)])
    return Frontier(
        instants=items,
        points=tuple(frontier_point(moves, level) for level in levels),
        actual_hurdle_bps=actual,
    )


def main() -> int:  # pragma: no cover - CLI
    frontier = build()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(frontier.as_dict(), indent=2), encoding="utf-8")
    print(frontier.render())
    print(f"\nwritten to {REPORT_PATH}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = [
    "HURDLE_SWEEP_BPS",
    "MIN_INSTANTS",
    "Frontier",
    "FrontierPoint",
    "HurdleError",
    "Instant",
    "break_even_accuracy",
    "build",
    "default_hurdle_bps",
    "frontier_point",
    "load_instants",
]
