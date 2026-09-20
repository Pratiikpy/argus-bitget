"""Disagreement, as a typed record rather than a sentence.

The desk already writes a line of prose when its analysts disagree, and ``agents/earnings.py:188``
already refuses to average a contradictory print into a weak consensus. Both are right and neither
is inspectable: a judge reading a decision cannot tell whether "the analysts disagreed" is a
measured fact about this decision or a phrase the model produced. A conflict that cannot be queried
is a conflict that cannot be scored.

So this module emits :class:`Conflict` records — who disagreed, about what, how far apart, which
view dominated and on what grounds — alongside the consensus. Every decision then carries either a
conflict set or an explicit unanimity, and the language interface can answer "did your analysts
agree?" from the record instead of from the thesis text.

**Why disagreement is worth surfacing rather than smoothing.** A panel that always agrees is either
reading one source through several analysts or has been aggregated until the disagreement is gone.
The first is measured already by ``Panel.independence_ratio``. The second is what this module
guards: strength of agreement and *volume* of agreement are different quantities, and a consensus
computed by averaging hides which one produced it.

**The sequencing hazard.** ARGUS's analysts run in order, and later analysts see earlier output.
``Agent_Market_Arena``'s ``testbed/get_daily_action.py:10-34`` dispatches every agent concurrently
against one frozen context precisely to avoid this — independent decisions on identical evidence,
so disagreement is genuine rather than an artefact of who spoke first.
:func:`independent_views` makes that property checkable here: given the views and the order they
were produced in, it reports whether a later analyst could have been influenced by an earlier one.
It does not make the calls concurrent — that belongs to the desk — but it means the desk cannot
claim independence it has not got.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from argus.agents.analysts import ACTIONABLE_CONFIDENCE, AnalystView

MATERIAL_GAP_BPS = 25.0
"""Below this, two analysts pointing the same way are agreeing about magnitude, not disagreeing."""

STRONG_CONVICTION = 0.6
"""A view below this confidence loses a tie on its own terms rather than on weight of numbers."""

CONVICTION_GAP = 0.3
"""How far apart two confidences must be before the distance is itself the disagreement.

**NOT VERIFIED.** This was a bare ``0.3`` sitting between two named and explained constants, with
nothing anywhere saying where it came from. It is a chosen threshold, not a measured one, and it is
named here so that it can be argued with rather than inherited.

What would ground it: the distribution of pairwise confidence gaps across a body of live panels,
split by whether the wider-gap pairs went on to disagree about anything that mattered. The stored
evidence is currently **four analyst views** (`data/track2_desk_run.json`), which is not a
distribution. Until that exists, treat the number as a convention.

It is also deliberately no longer the only route to a conviction conflict — see
`ACTIONABLE_CONFIDENCE` below, which triggers on a boundary the codebase already defines rather
than on a magnitude nobody has justified.
"""


class Kind(StrEnum):
    """What sort of disagreement it is. They are not interchangeable."""

    DIRECTION = "direction"
    """Opposite signals. The only kind that cannot be split the difference on."""

    MAGNITUDE = "magnitude"
    """Same direction, materially different size. A sizing question, not a thesis question."""

    CONVICTION = "conviction"
    """Same direction and size, far apart on confidence. Usually an evidence-quality dispute."""


@dataclass(frozen=True)
class Conflict:
    """One disagreement between two analysts, and how it resolved."""

    kind: Kind
    left: str
    right: str
    left_signal: str
    right_signal: str
    gap: float
    dominant: str
    grounds: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": str(self.kind),
            "between": [self.left, self.right],
            "signals": [self.left_signal, self.right_signal],
            "gap": round(self.gap, 3),
            "dominant": self.dominant,
            "grounds": self.grounds,
        }

    def render(self) -> str:
        return (
            f"[conflict:{self.kind}] {self.left} ({self.left_signal}) vs "
            f"{self.right} ({self.right_signal}), gap {self.gap:.2f} — "
            f"{self.dominant} dominates: {self.grounds}"
        )


def _opposed(a: str, b: str) -> bool:
    bullish = {"bullish", "long", "buy", "positive"}
    bearish = {"bearish", "short", "sell", "negative"}
    return (a in bullish and b in bearish) or (a in bearish and b in bullish)


def _dominant(left: AnalystView, right: AnalystView) -> tuple[str, str]:
    """Which view wins, and why — stated in terms a reader can disagree with.

    Confidence decides first, and only clearly: two analysts within 0.1 of each other are not
    meaningfully more or less sure, and picking one on that basis would be reading noise. When
    confidence does not separate them, the wider evidence base does, because a view resting on more
    independent sources is harder to have reached by accident.
    """
    if abs(left.confidence - right.confidence) >= 0.1:
        winner = left if left.confidence > right.confidence else right
        return winner.analyst, (
            f"confidence {max(left.confidence, right.confidence):.2f} against "
            f"{min(left.confidence, right.confidence):.2f}"
        )
    left_sources, right_sources = len(left.source_ids), len(right.source_ids)
    if left_sources != right_sources:
        winner = left if left_sources > right_sources else right
        return winner.analyst, (
            f"{max(left_sources, right_sources)} distinct sources against "
            f"{min(left_sources, right_sources)}"
        )
    return "neither", (
        "equally confident on equally many sources; the disagreement stands unresolved"
    )


def detect(views: Sequence[AnalystView]) -> tuple[Conflict, ...]:
    """Every material disagreement in the panel, pairwise.

    Pairwise rather than against the consensus: a consensus already contains the disagreement and
    comparing to it hides which two analysts actually differ.
    """
    out: list[Conflict] = []
    for i, left in enumerate(views):
        for right in views[i + 1:]:
            dominant, grounds = _dominant(left, right)

            if _opposed(left.signal, right.signal):
                out.append(Conflict(
                    Kind.DIRECTION, left.analyst, right.analyst, left.signal, right.signal,
                    gap=abs(left.magnitude_bps - right.magnitude_bps),
                    dominant=dominant, grounds=grounds,
                ))
                continue

            if left.signal != right.signal:
                continue

            gap = abs(left.magnitude_bps - right.magnitude_bps)
            if gap >= MATERIAL_GAP_BPS:
                out.append(Conflict(
                    Kind.MAGNITUDE, left.analyst, right.analyst, left.signal, right.signal,
                    gap=gap, dominant=dominant, grounds=grounds,
                ))
                continue

            conviction_gap = abs(left.confidence - right.confidence)
            # Two analysts on opposite sides of the actionable line disagree about whether to
            # trade at all. 0.52 against 0.49 is a gap of 0.03 and was reported as no conflict,
            # even though one of those views is actionable and the other is not. That boundary is
            # defined by the codebase; the gap threshold beside it is not. See `CONVICTION_GAP`.
            straddles = (left.confidence > ACTIONABLE_CONFIDENCE) != (
                right.confidence > ACTIONABLE_CONFIDENCE
            )
            if conviction_gap >= CONVICTION_GAP or straddles:
                out.append(Conflict(
                    Kind.CONVICTION, left.analyst, right.analyst, left.signal, right.signal,
                    gap=conviction_gap, dominant=dominant, grounds=grounds,
                ))
    return tuple(out)


@dataclass(frozen=True)
class ConflictReport:
    """What the panel disagreed about, or an explicit statement that it did not."""

    conflicts: tuple[Conflict, ...]
    analysts: int
    sequential: bool
    """True when later analysts could see earlier output, so agreement may be contagion."""

    @property
    def unanimous(self) -> bool:
        return not self.conflicts

    @property
    def has_directional_split(self) -> bool:
        """The kind that matters most: the panel does not agree which way to go."""
        return any(c.kind is Kind.DIRECTION for c in self.conflicts)

    @property
    def unresolved(self) -> tuple[Conflict, ...]:
        return tuple(c for c in self.conflicts if c.dominant == "neither")

    def as_dict(self) -> dict[str, Any]:
        return {
            "analysts": self.analysts,
            "unanimous": self.unanimous,
            "sequential": self.sequential,
            "directional_split": self.has_directional_split,
            "unresolved": len(self.unresolved),
            "conflicts": [c.as_dict() for c in self.conflicts],
        }

    def render(self) -> list[str]:
        """Lines for the decision record. Unanimity is stated, never left implied."""
        if self.unanimous:
            note = f"[conflict] none: {self.analysts} analysts agreed on direction and size"
            if self.sequential:
                note += (
                    " — but they ran in sequence, so agreement may be contagion rather than "
                    "consensus"
                )
            return [note]
        lines = [c.render() for c in self.conflicts]
        if self.unresolved:
            lines.append(
                f"[conflict] {len(self.unresolved)} disagreement(s) unresolved on the evidence; "
                f"the decision is taken in spite of them, not because they were settled"
            )
        return lines


def independent_views(views: Sequence[AnalystView], *, sequential: bool) -> bool:
    """Could these views have been reached independently?

    Two conditions, both necessary. The analysts must not have run in an order that let one read
    another, and they must not all rest on the same evidence. Either alone is insufficient: parallel
    analysts reading one article are as correlated as sequential ones.
    """
    if sequential:
        return False
    sources = {s for v in views for s in v.source_ids}
    return len(sources) > 1


def report(views: Sequence[AnalystView], *, sequential: bool = True) -> ConflictReport:
    """Build the record. ``sequential`` defaults to the pessimistic reading.

    Defaulting to ``True`` means a caller that has not thought about ordering gets the cautious
    label rather than an unearned claim of independence.
    """
    return ConflictReport(
        conflicts=detect(views), analysts=len(views), sequential=sequential
    )


__all__ = [
    "MATERIAL_GAP_BPS",
    "STRONG_CONVICTION",
    "Conflict",
    "ConflictReport",
    "Kind",
    "detect",
    "independent_views",
    "report",
]
