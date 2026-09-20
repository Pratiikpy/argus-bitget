"""Ablation — does this component earn its place, or is it decoration?

The project's own standard for calling a capability OWNED lists an ablation among the thirteen
conditions, and until now nothing here could run one. That gap is not cosmetic. A desk with nine
subsystems can look sophisticated while three of them contribute nothing, and the only way to tell
the difference is to remove one and measure. `research/architecture/sentiment-audit.md` reached
exactly that question about our own sentiment analyst and had no instrument to answer it with.

Two kinds of component exist here and they need different instruments, which is the whole design:

* **Deterministic components** — the Constitution, the execution guard, the hedgeability screen,
  the cost hurdle. Each is a pure function of the frame, so removing one and replaying the same
  frames gives an *exact* answer: these N decisions changed, in these ways. No statistics are
  needed and none are offered, because dressing an exact count in a p-value implies a sampling
  process that is not there. :func:`deterministic`.

* **Stochastic components** — anything whose output comes from a model. The same frame does not
  give the same answer twice, so a single paired run measures sampling noise as readily as effect.
  :func:`paired` therefore requires a minimum number of pairs and reports **INSUFFICIENT** below
  it rather than a direction. Refusing to answer is the point: a one-run ablation that reports
  "removing sentiment cost 4bps" is the kind of number that survives into a submission and is not
  true.

**Pairing is by frame, never by index.** Two arms run over the same market frames, and each frame's
two outcomes are compared against each other. Comparing arm means over differently-ordered runs
would let a fortunate ordering read as an effect — which is how ablations in this field usually go
wrong.

The statistic is an **exact two-sided sign test** (binomial, p=0.5, computed with
:func:`math.comb`, no SciPy) over the pairs where the arms actually differed. The sign test is
chosen deliberately over a paired t-test: trading outcome deltas are heavy-tailed and a t-test on
twenty pairs with one outlier reports significance that a resample will not reproduce. The sign
test throws away magnitude and keeps only direction, which is the part that survives fat tails.
Ties are excluded from the count rather than split, following the standard treatment — a component
that changed nothing on a frame is evidence about neither direction.

Read against ``microsoft/qlib``'s ``qlib/workflow/record_temp.py``, which records per-run metrics
for comparison but leaves the comparison to the reader, and against ``tau-bench``'s pass^k, which
makes the same point about single runs of a stochastic system: one sample of a noisy policy is not
a measurement of the policy.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

MIN_PAIRS = 30
"""Fewest differing pairs before :func:`paired` will state a direction.

Thirty is not a magic number and is not defended as one. It is the point at which the exact sign
test can reach two-sided p < 0.05 from a lopsided-but-realistic split (21 of 30 gives p = 0.043),
so below it the test cannot return a significant answer even when the component is genuinely
helping — and a test that cannot reject is not evidence of no effect. Reporting INSUFFICIENT there
is the honest reading. Raise it, never lower it, and say so in the report if you do.
"""

SIGNIFICANT = 0.05
"""Two-sided threshold. Reported alongside every verdict so the reader can disagree with it."""


class Verdict(StrEnum):
    """What the ablation established. Four states, and never a fifth dressed as one."""

    INSUFFICIENT = "insufficient"
    """Too few differing pairs to distinguish an effect from noise. Not 'no effect'."""

    NO_EFFECT = "no_effect"
    """Enough pairs, and the direction is not distinguishable from a coin. The component is not
    earning its cost on this evidence, which is a reason to demote it, not to keep it quietly."""

    HELPS = "helps"
    HURTS = "hurts"
    """A component that measurably makes outcomes worse. Worth its own name: the failure mode of
    an ablation harness is that it can only ever say 'keep', and then nothing is ever removed."""


def sign_test(wins: int, losses: int) -> float:
    """Exact two-sided binomial p-value for ``wins`` successes in ``wins + losses`` trials.

    Ties are not passed in — they are excluded upstream, per the standard treatment.

    Two-sided by doubling the smaller tail and clamping at 1.0, which is the conventional exact
    construction. Returns 1.0 for no trials rather than raising: an ablation with nothing to
    compare has not found evidence, and a caller checking the p-value should see "no evidence",
    not an exception it must special-case.
    """
    n = wins + losses
    if n <= 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * float(tail))


@dataclass(frozen=True, slots=True)
class Outcome:
    """One arm's result on one frame.

    ``value`` is whatever the experiment is scoring — realised basis points, a Sharpe
    contribution, a grading score. Its meaning is the caller's; the harness only ever compares two
    values produced by the same scorer on the same frame.
    """

    frame_id: str
    arm: str
    value: float
    detail: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Pair:
    """The same frame, both arms."""

    frame_id: str
    with_component: float
    without_component: float

    @property
    def delta(self) -> float:
        """Positive means the component helped on this frame."""
        return self.with_component - self.without_component


@dataclass(frozen=True, slots=True)
class Result:
    """What an ablation established, including what it could not."""

    component: str
    metric: str
    pairs: tuple[Pair, ...]
    verdict: Verdict
    p_value: float
    minimum_pairs: int
    unpaired: tuple[str, ...] = ()

    @property
    def n(self) -> int:
        return len(self.pairs)

    @property
    def wins(self) -> int:
        return sum(1 for p in self.pairs if p.delta > 0)

    @property
    def losses(self) -> int:
        return sum(1 for p in self.pairs if p.delta < 0)

    @property
    def ties(self) -> int:
        return sum(1 for p in self.pairs if p.delta == 0)

    @property
    def differing(self) -> int:
        return self.wins + self.losses

    @property
    def mean_delta(self) -> float:
        """Reported, but never the basis of the verdict.

        The verdict comes from the sign test precisely because this number is the one an outlier
        can carry. It is here so a reader can see the magnitude the direction implies, and see when
        the two disagree — a positive mean with a losing sign count means one frame is doing all
        the work, which is worth knowing before anything is promoted.
        """
        return sum(p.delta for p in self.pairs) / len(self.pairs) if self.pairs else 0.0

    @property
    def median_delta(self) -> float:
        if not self.pairs:
            return 0.0
        ordered = sorted(p.delta for p in self.pairs)
        mid = len(ordered) // 2
        return ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2

    @property
    def carried_by_one_frame(self) -> bool:
        """True when dropping the single largest-magnitude pair flips the sign of the mean.

        The check that would have caught our own Track-1 single-symbol result, where one instrument
        produced the whole return from four trades. A result this fragile is reported as such
        whatever the sign test says.
        """
        if len(self.pairs) < 2:
            return False
        deltas = [p.delta for p in self.pairs]
        biggest = max(range(len(deltas)), key=lambda i: abs(deltas[i]))
        remaining = deltas[:biggest] + deltas[biggest + 1:]
        full = sum(deltas) / len(deltas)
        trimmed = sum(remaining) / len(remaining)
        return full != 0 and (full > 0) != (trimmed > 0)

    def render(self) -> str:
        lines = [
            f"ABLATION — {self.component}, scored on {self.metric}",
            f"  verdict: {self.verdict.value.upper()}  (two-sided sign test p={self.p_value:.3f}, "
            f"threshold {SIGNIFICANT})",
            f"  pairs: {self.n}  ({self.wins} better with it, {self.losses} worse, "
            f"{self.ties} identical)",
            f"  delta with the component: mean {self.mean_delta:+.4f}, "
            f"median {self.median_delta:+.4f}",
        ]
        if self.verdict is Verdict.INSUFFICIENT:
            lines.append(
                f"  {self.differing} differing pair(s) is below the {self.minimum_pairs} this "
                f"harness needs to tell an effect from noise. This is NOT a finding of no effect; "
                f"it is the absence of a finding, and the component keeps whatever standing it "
                f"already had."
            )
        if self.carried_by_one_frame:
            lines.append(
                "  WARNING: removing the single largest pair flips the mean. One frame is "
                "carrying this result; do not promote on it."
            )
        if self.unpaired:
            lines.append(
                f"  {len(self.unpaired)} frame(s) ran in only one arm and were excluded: "
                f"{', '.join(self.unpaired[:5])}"
                + (" …" if len(self.unpaired) > 5 else "")
            )
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "metric": self.metric,
            "verdict": self.verdict.value,
            "p_value": round(self.p_value, 6),
            "n_pairs": self.n,
            "wins": self.wins,
            "losses": self.losses,
            "ties": self.ties,
            "mean_delta": round(self.mean_delta, 6),
            "median_delta": round(self.median_delta, 6),
            "minimum_pairs": self.minimum_pairs,
            "carried_by_one_frame": self.carried_by_one_frame,
            "unpaired": list(self.unpaired),
        }


def pair_outcomes(
    outcomes: Iterable[Outcome], *, with_arm: str, without_arm: str
) -> tuple[tuple[Pair, ...], tuple[str, ...]]:
    """Pair two arms by frame. Frames present in only one arm are returned separately, not dropped.

    Silently dropping them is the bug this signature exists to prevent: an arm that crashed on the
    hard frames and completed on the easy ones would otherwise look like an improvement.
    """
    with_by_frame: dict[str, float] = {}
    without_by_frame: dict[str, float] = {}
    for outcome in outcomes:
        if outcome.arm == with_arm:
            with_by_frame[outcome.frame_id] = outcome.value
        elif outcome.arm == without_arm:
            without_by_frame[outcome.frame_id] = outcome.value
    shared = sorted(set(with_by_frame) & set(without_by_frame))
    orphaned = sorted(set(with_by_frame) ^ set(without_by_frame))
    pairs = tuple(
        Pair(frame_id=f, with_component=with_by_frame[f], without_component=without_by_frame[f])
        for f in shared
    )
    return pairs, tuple(orphaned)


def paired(
    outcomes: Iterable[Outcome],
    *,
    component: str,
    metric: str,
    with_arm: str = "with",
    without_arm: str = "without",
    minimum_pairs: int = MIN_PAIRS,
    higher_is_better: bool = True,
) -> Result:
    """Ablate a stochastic component over paired runs.

    ``higher_is_better=False`` inverts the comparison for metrics where lower wins — realised cost,
    drawdown, time-to-decision. The inversion happens once, here, rather than being left to each
    caller to remember, because a caller that forgets it reports a harmful component as helpful.
    """
    pairs, orphaned = pair_outcomes(outcomes, with_arm=with_arm, without_arm=without_arm)
    if not higher_is_better:
        pairs = tuple(
            Pair(
                frame_id=p.frame_id,
                with_component=-p.with_component,
                without_component=-p.without_component,
            )
            for p in pairs
        )
    wins = sum(1 for p in pairs if p.delta > 0)
    losses = sum(1 for p in pairs if p.delta < 0)
    p_value = sign_test(wins, losses)
    if wins + losses < minimum_pairs:
        verdict = Verdict.INSUFFICIENT
    elif p_value >= SIGNIFICANT:
        verdict = Verdict.NO_EFFECT
    else:
        verdict = Verdict.HELPS if wins > losses else Verdict.HURTS
    return Result(
        component=component,
        metric=metric,
        pairs=pairs,
        verdict=verdict,
        p_value=p_value,
        minimum_pairs=minimum_pairs,
        unpaired=orphaned,
    )


@dataclass(frozen=True, slots=True)
class Change:
    """One frame where removing a deterministic component changed the answer."""

    frame_id: str
    with_component: Any
    without_component: Any

    def render(self) -> str:
        return (
            f"{self.frame_id}: with={self.with_component!r} without={self.without_component!r}"
        )


@dataclass(frozen=True, slots=True)
class DeterministicResult:
    """An exact count. No p-value, because there is no sampling here to be uncertain about."""

    component: str
    frames: int
    changes: tuple[Change, ...]

    @property
    def unchanged(self) -> int:
        return self.frames - len(self.changes)

    @property
    def share_changed(self) -> float:
        return len(self.changes) / self.frames if self.frames else 0.0

    @property
    def inert(self) -> bool:
        """The component changed no decision on any frame tested.

        Not proof it never would — a gate that never fired on calm frames is untested, not useless,
        and the report says which. But a component that is inert across a representative set is a
        component whose cost is being paid for nothing.
        """
        return not self.changes

    def render(self) -> str:
        lines = [
            f"ABLATION (deterministic) — {self.component}",
            f"  {len(self.changes)} of {self.frames} frame(s) decided differently without it "
            f"({self.share_changed:.1%}); no statistics, this is an exact replay.",
        ]
        if self.inert:
            lines.append(
                "  It changed nothing on any frame tested. Either the frames do not reach it, or "
                "it is not doing work — establish which before keeping it."
            )
        for change in self.changes[:10]:
            lines.append(f"  - {change.render()}")
        if len(self.changes) > 10:
            lines.append(f"  … and {len(self.changes) - 10} more")
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "frames": self.frames,
            "changed": len(self.changes),
            "share_changed": round(self.share_changed, 6),
            "inert": self.inert,
            "changes": [
                {
                    "frame_id": c.frame_id,
                    "with": str(c.with_component),
                    "without": str(c.without_component),
                }
                for c in self.changes
            ],
        }


def deterministic(
    frames: Sequence[tuple[str, Any]],
    *,
    component: str,
    with_component: Callable[[Any], Any],
    without_component: Callable[[Any], Any],
) -> DeterministicResult:
    """Replay every frame through both paths and report exactly which answers moved.

    ``frames`` is ``(frame_id, frame)`` so a change can be named, not just counted. Equality is
    ``!=`` on whatever the two callables return, so a caller comparing rich objects must give them
    a meaningful ``__eq__`` — dataclasses do, which is why the decision types here are dataclasses.
    """
    changes: list[Change] = []
    for frame_id, frame in frames:
        kept = with_component(frame)
        dropped = without_component(frame)
        if kept != dropped:
            changes.append(
                Change(frame_id=frame_id, with_component=kept, without_component=dropped)
            )
    return DeterministicResult(
        component=component, frames=len(frames), changes=tuple(changes)
    )


__all__ = [
    "MIN_PAIRS",
    "SIGNIFICANT",
    "Change",
    "DeterministicResult",
    "Outcome",
    "Pair",
    "Result",
    "Verdict",
    "deterministic",
    "pair_outcomes",
    "paired",
    "sign_test",
]
