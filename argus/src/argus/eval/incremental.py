"""Incremental value — what the desk added over a rule you could write on a napkin.

Track 2's Open Theme names "incremental value over a fixed-rule baseline or a Human+AI baseline" as
one of the five things an agent benchmark should measure. `eval.baseline` answers a version of that
question — the desk against holding its own picks — and it cannot answer it here, because it needs
settled trades and there are none. It returns UNDEFINED, correctly, and a criterion that returns
UNDEFINED is a criterion we do not yet score on.

This module answers the same question at the level where we *do* have a record: **the decision**.
Every instant the desk faced is replayed through a set of fixed rules — rules with no model, no
evidence and no reasoning — and each rule is scored on the move that actually followed, net of the
same hurdle the desk had to clear. The desk is then compared against each rule pair by pair, on the
identical instants, with an exact sign test.

**The baselines are chosen so that at least one of them should be embarrassing.** A benchmark whose
baselines are all weak measures nothing:

* ``always_flat`` — never trade. This is the honest null and, on the current record, it is also
  exactly what the desk does, so the comparison is a tie by construction. Reporting that tie is the
  point: it says in one line that the desk has not yet distinguished itself from doing nothing.
* ``always_long`` — buy every instant. Tests whether the universe simply drifted up over the
  window, which is the explanation that beats most trading systems.
* ``momentum`` and ``reversion`` — take the sign of the trailing return, or its opposite. Between
  them they cover both directional priors, so a desk that beats neither has no directional edge in
  either regime.
* ``volatility_gated`` — trade the momentum direction, but only when the trailing volatility
  implies a move large enough to clear the hurdle. This is the strongest baseline and the one that
  most resembles a competent human with no model: it is the rule the desk has to beat to be worth
  its deliberation cost.

**Why a sign test and not a t-test on the mean.** Net returns per instant are fat-tailed and the
sample is small, so a mean difference is driven by whichever instant moved most. The sign test asks
only how often one policy beat the other, which is the statement that survives an outlier. The
machinery is `eval.ablation`'s, reused rather than rewritten, including its refusal to state a
direction below thirty differing pairs.

**What this cannot say.** A rule that wins here has not been shown to be profitable — it has been
shown to beat the desk on this record. Every one of these baselines is also being selected from a
menu of five, so a winner among them is the best of five and is reported as such. And the record is
39 instants across four symbols, which the effective-sample correction cuts to roughly 25; that is
enough to notice a large difference and not enough to settle a small one, and the report says so
rather than leaving it to be inferred.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

from argus.eval.ablation import sign_test
from argus.eval.forecasts import design_effect, intraclass_correlation

DATA = Path(__file__).resolve().parents[3] / "data"
REPORT_PATH = DATA / "incremental_value.json"

MIN_INSTANTS = 20
"""Fewest instants before any comparison is reported.

Lower than `ablation.MIN_PAIRS` deliberately and for a stated reason: that floor governs a paired
ablation where both arms are stochastic, and these baselines are deterministic, so the only noise
is the market's. The direction of a comparison below thirty *differing* pairs is still reported as
inconclusive by the sign test itself.
"""

DIRECTION_NONE = 0
DIRECTION_LONG = 1
DIRECTION_SHORT = -1


class IncrementalError(ValueError):
    """Raised rather than reporting a comparison that has no sample behind it."""


@dataclass(frozen=True, slots=True)
class Instant:
    """One decision point, with everything a rule needs and nothing it does not.

    ``trailing`` are the returns strictly before the instant, so a rule cannot see its own answer.
    ``realised_bps`` is the move over the hold horizon, which is visible only to the scorer.
    """

    symbol: str
    at: datetime
    trailing: tuple[float, ...]
    realised_bps: float
    hurdle_bps: float

    def net_bps(self, direction: int) -> float:
        """What a policy taking ``direction`` earned here, net of the round trip.

        Flat earns exactly zero — not the move it declined. Crediting an abstention with the move
        it passed on is the single most common way a backtest of a selective strategy is inflated.
        """
        if direction == DIRECTION_NONE:
            return 0.0
        return direction * self.realised_bps - self.hurdle_bps


Policy = Callable[[Instant], int]
"""A rule: one instant in, one of -1, 0, +1 out. No state, so replay order cannot matter."""


def always_flat(instant: Instant) -> int:
    """Never trade. The null, and what the desk currently does on every instant."""
    return DIRECTION_NONE


def always_long(instant: Instant) -> int:
    """Buy everything. Beats most trading systems whenever the window happened to drift up."""
    return DIRECTION_LONG


def momentum(instant: Instant, *, lookback: int = 24) -> int:
    """The sign of the trailing return over ``lookback`` bars."""
    window = instant.trailing[-lookback:]
    if not window:
        return DIRECTION_NONE
    total = sum(window)
    if total > 0:
        return DIRECTION_LONG
    return DIRECTION_SHORT if total < 0 else DIRECTION_NONE


def reversion(instant: Instant, *, lookback: int = 24) -> int:
    """Momentum's opposite. Present so that no directional prior is left untested."""
    return -momentum(instant, lookback=lookback)


def volatility_gated(instant: Instant, *, lookback: int = 24) -> int:
    """Momentum, but only when trailing volatility implies the move can clear the hurdle.

    The strongest baseline here, and the one that stands in for a competent trader with no model:
    it declines the instants where the arithmetic says a round trip cannot be paid for. A desk
    whose whole contribution is deliberation has to beat this rule to be worth the deliberation.
    """
    window = instant.trailing[-lookback:]
    if len(window) < 2:
        return DIRECTION_NONE
    expected_move_bps = pstdev(window) * 10_000
    if expected_move_bps <= instant.hurdle_bps:
        return DIRECTION_NONE
    return momentum(instant, lookback=lookback)


BASELINES: dict[str, Policy] = {
    "always_flat": always_flat,
    "always_long": always_long,
    "momentum": momentum,
    "reversion": reversion,
    "volatility_gated": volatility_gated,
}


@dataclass(frozen=True, slots=True)
class PolicyScore:
    """One policy over the whole record."""

    name: str
    directions: tuple[int, ...]
    nets: tuple[float, ...]

    @property
    def trades(self) -> int:
        return sum(1 for d in self.directions if d != DIRECTION_NONE)

    @property
    def total_bps(self) -> float:
        return sum(self.nets)

    @property
    def mean_bps(self) -> float:
        return fmean(self.nets) if self.nets else 0.0

    @property
    def wins(self) -> int:
        return sum(1 for n in self.nets if n > 0)

    @property
    def win_rate(self) -> float | None:
        """Over the instants this policy actually traded. None when it never traded.

        A win rate computed over instants a policy declined would report a flat policy as 0% rather
        than as undefined, which reads as "it always loses" instead of "it never played".
        """
        traded = [n for n, d in zip(self.nets, self.directions, strict=True) if d != DIRECTION_NONE]
        if not traded:
            return None
        return sum(1 for n in traded if n > 0) / len(traded)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "trades": self.trades,
            "instants": len(self.nets),
            "total_bps": round(self.total_bps, 2),
            "mean_bps": round(self.mean_bps, 3),
            "win_rate": None if self.win_rate is None else round(self.win_rate, 4),
        }


def score(name: str, policy: Policy, instants: Sequence[Instant]) -> PolicyScore:
    directions = tuple(policy(i) for i in instants)
    return PolicyScore(
        name=name,
        directions=directions,
        nets=tuple(i.net_bps(d) for i, d in zip(instants, directions, strict=True)),
    )


@dataclass(frozen=True, slots=True)
class Comparison:
    """The desk against one baseline, pair by pair on identical instants."""

    baseline: str
    wins: int
    losses: int
    ties: int
    mean_difference_bps: float

    @property
    def differing(self) -> int:
        return self.wins + self.losses

    @property
    def p_value(self) -> float:
        return sign_test(self.wins, self.losses)

    @property
    def verdict(self) -> str:
        if self.differing == 0:
            return (
                f"IDENTICAL to {self.baseline} on every instant. The two policies made the same "
                f"call each time, so there is nothing to test and no value was added either way"
            )
        if self.p_value > 0.05:
            return (
                f"INCONCLUSIVE against {self.baseline}: {self.wins} wins, {self.losses} losses "
                f"over {self.differing} differing instant(s), p={self.p_value:.3f}"
            )
        better = "BEATS" if self.wins > self.losses else "LOSES TO"
        return (
            f"{better} {self.baseline}: {self.wins} wins, {self.losses} losses, "
            f"p={self.p_value:.3f}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline,
            "wins": self.wins,
            "losses": self.losses,
            "ties": self.ties,
            "differing": self.differing,
            "mean_difference_bps": round(self.mean_difference_bps, 3),
            "p_value": round(self.p_value, 6),
            "verdict": self.verdict,
        }


def compare(desk: PolicyScore, baseline: PolicyScore) -> Comparison:
    """Paired comparison. Both policies must have been scored on the same instants."""
    if len(desk.nets) != len(baseline.nets):
        raise IncrementalError(
            "the desk and the baseline must be scored on the same instants; comparing two "
            "different samples is not a paired test"
        )
    wins = losses = ties = 0
    differences = []
    for mine, theirs in zip(desk.nets, baseline.nets, strict=True):
        differences.append(mine - theirs)
        if mine > theirs:
            wins += 1
        elif mine < theirs:
            losses += 1
        else:
            ties += 1
    return Comparison(
        baseline=baseline.name,
        wins=wins,
        losses=losses,
        ties=ties,
        mean_difference_bps=fmean(differences) if differences else 0.0,
    )


@dataclass(frozen=True, slots=True)
class IncrementalReport:
    """Every baseline, the desk beside them, and what the sample can support."""

    instants: tuple[Instant, ...]
    desk: PolicyScore
    baselines: tuple[PolicyScore, ...]
    comparisons: tuple[Comparison, ...]

    @property
    def effective_instants(self) -> int:
        by_time: dict[datetime, list[float]] = {}
        for i in self.instants:
            by_time.setdefault(i.at, []).append(i.realised_bps)
        groups = [g for g in by_time.values() if g]
        if not groups:
            return len(self.instants)
        icc = intraclass_correlation(groups)
        average = sum(len(g) for g in groups) / len(groups)
        return max(1, int(len(self.instants) / design_effect(icc, average)))

    @property
    def best_baseline(self) -> PolicyScore:
        return max(self.baselines, key=lambda b: b.total_bps)

    @property
    def verdict(self) -> str:
        beaten = [c for c in self.comparisons if c.wins > c.losses and c.p_value <= 0.05]
        lost = [c for c in self.comparisons if c.losses > c.wins and c.p_value <= 0.05]
        best = self.best_baseline
        head = (
            f"Over {len(self.instants)} instant(s) ({self.effective_instants} effective), the desk "
            f"opened {self.desk.trades} position(s) for {self.desk.total_bps:+.0f}bps. The best "
            f"fixed rule was {best.name} at {best.total_bps:+.0f}bps over {best.trades} trade(s), "
            f"and it is the best of {len(self.baselines)} rules tried, which is itself a selection."
        )
        if lost:
            return (
                f"{head} NO INCREMENTAL VALUE DEMONSTRATED — the desk is beaten at 5% by "
                f"{', '.join(c.baseline for c in lost)}."
            )
        if beaten:
            return (
                f"{head} The desk beats {', '.join(c.baseline for c in beaten)} at 5%, and is not "
                f"significantly separated from the rest."
            )
        return (
            f"{head} NO INCREMENTAL VALUE DEMONSTRATED, and none disproved either: no baseline is "
            f"separated from the desk at 5% on this sample. With {self.effective_instants} "
            f"effective instant(s) this record can notice a large difference and cannot settle a "
            f"small one."
        )

    def render(self) -> str:
        lines = [
            f"INCREMENTAL VALUE — {len(self.instants)} instant(s), "
            f"{self.effective_instants} effective",
            "",
            f"{'policy':>18}{'trades':>8}{'total bps':>12}{'mean bps':>11}{'win rate':>10}",
        ]
        for policy in (self.desk, *self.baselines):
            rate = "  n/a" if policy.win_rate is None else f"{policy.win_rate:.0%}"
            lines.append(
                f"{policy.name:>18}{policy.trades:>8}{policy.total_bps:>12.0f}"
                f"{policy.mean_bps:>11.1f}{rate:>10}"
            )
        lines += ["", "  paired sign tests, desk against each rule:"]
        lines.extend(f"    {c.verdict}" for c in self.comparisons)
        lines += ["", f"  {self.verdict}"]
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "generated_at": datetime.now(UTC).isoformat(),
            "instants": len(self.instants),
            "effective_instants": self.effective_instants,
            "symbols": sorted({i.symbol for i in self.instants}),
            "desk": self.desk.as_dict(),
            "baselines": [b.as_dict() for b in self.baselines],
            "comparisons": [c.as_dict() for c in self.comparisons],
            "verdict": self.verdict,
        }


def evaluate(
    instants: Sequence[Instant],
    desk_directions: Sequence[int],
    *,
    baselines: dict[str, Policy] | None = None,
) -> IncrementalReport:
    """Score the desk and every baseline on the same instants and compare them pairwise."""
    if len(instants) < MIN_INSTANTS:
        raise IncrementalError(
            f"{len(instants)} instant(s) is below the {MIN_INSTANTS} this comparison needs; a "
            f"win-loss count from fewer is not a measurement of anything"
        )
    if len(desk_directions) != len(instants):
        raise IncrementalError("the desk must have a decision for every instant, and only those")
    if any(d not in (DIRECTION_LONG, DIRECTION_NONE, DIRECTION_SHORT) for d in desk_directions):
        raise IncrementalError("a direction must be -1, 0 or +1")

    rules = BASELINES if baselines is None else baselines
    desk = PolicyScore(
        name="desk",
        directions=tuple(desk_directions),
        nets=tuple(i.net_bps(d) for i, d in zip(instants, desk_directions, strict=True)),
    )
    scored = tuple(score(name, policy, instants) for name, policy in rules.items())
    return IncrementalReport(
        instants=tuple(instants),
        desk=desk,
        baselines=scored,
        comparisons=tuple(compare(desk, b) for b in scored),
    )


def main() -> int:  # pragma: no cover - CLI
    from argus.eval.collect import collect_instants

    instants, desk_directions = collect_instants()
    report = evaluate(instants, desk_directions)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")
    print(report.render())
    print(f"\nwritten to {REPORT_PATH}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = [
    "BASELINES",
    "DIRECTION_LONG",
    "DIRECTION_NONE",
    "DIRECTION_SHORT",
    "MIN_INSTANTS",
    "Comparison",
    "IncrementalError",
    "IncrementalReport",
    "Instant",
    "Policy",
    "PolicyScore",
    "always_flat",
    "always_long",
    "compare",
    "evaluate",
    "momentum",
    "reversion",
    "score",
    "volatility_gated",
]
