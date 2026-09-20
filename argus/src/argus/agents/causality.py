"""Event causality — graded link by link, not by whether the trade won.

Track 2's Event-Driven sub-theme asks how events "drive autonomous Agent trading". Every entry will
answer with a classifier and a P&L number. The problem with that answer is that **P&L cannot
distinguish a correct thesis from a lucky one**: a call that is right about direction while being
wrong about the mechanism, the affected variable, and the transmission path scores identically to
one that understood the event.

So the agent states its transmission chain in advance —

    event -> mechanism -> affected variable -> affected asset -> expected magnitude

— and each link is graded separately once the outcome is known. A chain that reached the right
direction through three broken links is recorded as a **failure**, which is the grade a P&L scorer
would have recorded as a success.

EventEdge is the strongest event system in the corpus (proprietary, so its event-study machinery is
re-derived from MacKinlay 1997 rather than copied). Its LLM is advisory-only and it runs once
daily. Nothing there, and nothing else we tore down, grades the chain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any


class LinkGrade(StrEnum):
    CORRECT = "correct"
    WRONG = "wrong"
    UNCERTAIN = "uncertain"
    """Checkable in principle, not yet resolved. Distinct from UNSUPPORTED."""

    UNSUPPORTED = "unsupported"
    """No evidence was ever offered for this link. Worse than wrong: wrong is a testable claim."""

    UNGRADED = "ungraded"


@dataclass(frozen=True, slots=True)
class Link:
    """One step in a transmission chain, with what would confirm or refute it."""

    step: str
    claim: str
    falsifier: str
    evidence_ids: tuple[str, ...] = ()

    @property
    def is_checkable(self) -> bool:
        """A link with no falsifier cannot be graded, so it cannot be a claim.

        This is the same rule the decision layer applies to theses, one level down.
        """
        return bool(self.falsifier.strip())


@dataclass
class CausalChain:
    """An event's stated transmission path, and its grades once the outcome is known."""

    event: str
    as_of: datetime
    links: list[Link] = field(default_factory=list)
    grades: list[LinkGrade] = field(default_factory=list)
    predicted_direction: str = ""
    predicted_magnitude_bps: int | None = None
    """How far the model said it would move, or ``None`` when it did not say.

    **``None``, not ``0``.** A missing ``magnitude_bps`` was coerced to zero, which is a real and
    confident prediction — *"this event moves the price not at all"* — attributed to a model that
    made no such claim. ``magnitude_error_bps`` then scored that invention against the realised
    move and folded the result into the accuracy record.
    """
    realised_direction: str = ""
    realised_magnitude_bps: int = 0

    def add(self, link: Link) -> None:
        self.links.append(link)
        self.grades.append(
            LinkGrade.UNGRADED if link.is_checkable else LinkGrade.UNSUPPORTED
        )

    def grade(self, index: int, grade: LinkGrade) -> None:
        if not 0 <= index < len(self.links):
            raise IndexError(f"no link {index} in a chain of {len(self.links)}")
        if self.grades[index] is LinkGrade.UNSUPPORTED:
            # An unfalsifiable link cannot be graded correct later. Allowing that would let a
            # chain launder an unsupported claim into a confirmed one after the fact.
            raise ValueError(
                f"link {index} ({self.links[index].step!r}) offered no falsifier; it cannot be "
                f"graded after the outcome is known"
            )
        self.grades[index] = grade

    # --- the scores ---------------------------------------------------------------------------

    @property
    def direction_correct(self) -> bool:
        return bool(self.realised_direction) and (
            self.predicted_direction == self.realised_direction
        )

    @property
    def links_correct(self) -> int:
        return sum(1 for g in self.grades if g is LinkGrade.CORRECT)

    @property
    def links_graded(self) -> int:
        return sum(1 for g in self.grades if g not in (LinkGrade.UNGRADED,))

    @property
    def chain_accuracy(self) -> float | None:
        """Fraction of graded links that held, or ``None`` when nothing has been graded.

        **``None``, not ``0.0``.** This returned zero for a chain with no graded links, which is
        indistinguishable from a chain that *was* graded and got every link wrong — and the two
        could not be more different: one is unmeasured, the other is refuted.

        The consequence was not cosmetic. ``was_lucky`` is ``direction_correct and chain_accuracy <
        0.5``, so **every ungraded chain that happened to call the direction right was labelled a
        lucky win** — the exact accusation this module exists to make carefully, made automatically
        against decisions nobody had examined. And `summary` averaged those zeros into
        ``mean_chain_accuracy_pct``, dragging a real figure down with fabricated ones.

        The metric a P&L scorer cannot produce is only worth having if it declines to produce one
        too.
        """
        graded = [g for g in self.grades if g is not LinkGrade.UNGRADED]
        if not graded:
            return None
        return self.links_correct / len(graded)

    @property
    def was_lucky(self) -> bool:
        """Right direction, broken reasoning.

        The case this whole module exists to name. A system that only tracks outcomes will bank
        this as a win and repeat the reasoning that produced it.
        """
        accuracy = self.chain_accuracy
        if accuracy is None:
            # An ungraded chain has not been shown lucky; it has not been shown anything.
            return False
        return self.direction_correct and accuracy < 0.5

    @property
    def was_unlucky(self) -> bool:
        """Sound reasoning, wrong outcome. Worth keeping — the process may still be right."""
        accuracy = self.chain_accuracy
        if accuracy is None:
            return False  # symmetrical: unmeasured is not a defence either
        return not self.direction_correct and accuracy >= 0.75

    @property
    def verdict(self) -> str:
        if not self.realised_direction:
            return "pending"
        if self.was_lucky:
            return "lucky"
        if self.was_unlucky:
            return "unlucky"
        return "sound" if self.direction_correct else "wrong"

    def magnitude_error_bps(self) -> int | None:
        """Absolute miss in basis points, or ``None`` when the model named no magnitude.

        Scoring an absent prediction against a realised move measures our own default, not the
        model.
        """
        if self.predicted_magnitude_bps is None:
            return None
        return abs(self.predicted_magnitude_bps - self.realised_magnitude_bps)

    def weakest_link(self) -> Link | None:
        """The first link that broke. Where the reasoning actually failed."""
        for link, grade in zip(self.links, self.grades, strict=True):
            if grade in (LinkGrade.WRONG, LinkGrade.UNSUPPORTED):
                return link
        return None

    def as_dict(self) -> dict[str, Any]:
        weakest = self.weakest_link()
        return {
            "event": self.event,
            "as_of": self.as_of.isoformat(),
            "chain": [
                {
                    "step": link.step,
                    "claim": link.claim,
                    "falsifier": link.falsifier,
                    "grade": str(grade),
                    "evidence": list(link.evidence_ids),
                }
                for link, grade in zip(self.links, self.grades, strict=True)
            ],
            "predicted": {
                "direction": self.predicted_direction,
                "magnitude_bps": self.predicted_magnitude_bps,
            },
            "realised": {
                "direction": self.realised_direction,
                "magnitude_bps": self.realised_magnitude_bps,
            },
            "direction_correct": self.direction_correct,
            "chain_accuracy": (
                None if self.chain_accuracy is None else round(self.chain_accuracy, 3)
            ),
            "links_correct": f"{self.links_correct}/{self.links_graded}",
            "magnitude_error_bps": self.magnitude_error_bps(),
            "verdict": self.verdict,
            "weakest_link": weakest.step if weakest else None,
        }


STANDARD_STEPS = (
    "what changed",
    "economic mechanism",
    "affected variable",
    "affected asset",
    "already priced?",
    "expected magnitude",
)



def _magnitude(value: object) -> int | None:
    """The model's stated magnitude, or ``None`` when it did not state one.

    ``int(float(value or 0))`` turned a missing field into a confident prediction of zero movement.
    An unparseable field is treated the same way as an absent one: unknown, not zero.
    """
    if value in (None, "", "null"):
        return None
    try:
        return int(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def chain_from_response(
    event: str, as_of: datetime, response: dict[str, Any]
) -> CausalChain:
    """Build a chain from an Event Analyst's JSON, without inventing links it did not state.

    A model that returns fewer links than the standard path gets a shorter chain, not a padded
    one. Filling the gaps would make every chain look complete and destroy the metric.
    """
    chain = CausalChain(
        event=event,
        as_of=as_of,
        predicted_direction=str(response.get("signal", "")).strip().lower(),
        predicted_magnitude_bps=_magnitude(response.get("magnitude_bps")),
    )
    raw_links = response.get("chain") or []
    falsifiers = response.get("chain_falsifiers") or []

    for i, text in enumerate(raw_links):
        step = STANDARD_STEPS[i] if i < len(STANDARD_STEPS) else f"link {i + 1}"
        chain.add(Link(
            step=step,
            claim=str(text).strip(),
            falsifier=str(falsifiers[i]).strip() if i < len(falsifiers) else "",
            evidence_ids=tuple(str(s) for s in response.get("source_ids", [])),
        ))
    return chain


@dataclass
class CausalLedger:
    """Chains across many events. The aggregate is what exposes systematic reasoning errors."""

    chains: list[CausalChain] = field(default_factory=list)

    def add(self, chain: CausalChain) -> None:
        self.chains.append(chain)

    def summary(self) -> dict[str, Any]:
        resolved = [c for c in self.chains if c.realised_direction]
        if not resolved:
            return {"chains": len(self.chains), "note": "no outcomes resolved yet"}

        lucky = [c for c in resolved if c.was_lucky]
        verdicts: dict[str, int] = {}
        for c in resolved:
            verdicts[c.verdict] = verdicts.get(c.verdict, 0) + 1

        # Which step in the standard path breaks most often — a systematic reasoning weakness
        # that neither P&L nor direction accuracy can surface.
        breaks: dict[str, int] = {}
        for c in resolved:
            weak = c.weakest_link()
            if weak:
                breaks[weak.step] = breaks.get(weak.step, 0) + 1

        # Only the chains that were actually graded. Including the ungraded ones as zeros — which
        # is what `chain_accuracy` used to return for them — understated this by however many
        # nobody had examined.
        _graded = [c.chain_accuracy for c in resolved if c.chain_accuracy is not None]

        return {
            "chains": len(self.chains),
            "resolved": len(resolved),
            "direction_accuracy_pct": round(
                100 * sum(1 for c in resolved if c.direction_correct) / len(resolved), 1
            ),
            # Averaged over the chains that were actually graded, with the count beside it.
            # Including ungraded chains as zeros understated this by however many nobody examined.
            "mean_chain_accuracy_pct": (
                round(100 * sum(_graded) / len(_graded), 1) if _graded else None
            ),
            "chains_graded": len(_graded),
            "chains_ungraded": len(resolved) - len(_graded),
            "verdicts": verdicts,
            "lucky_wins_pct": round(100 * len(lucky) / len(resolved), 1),
            "most_common_break": max(breaks, key=lambda k: breaks[k]) if breaks else None,
            "breaks_by_step": breaks,
            "note": (
                "lucky wins are right direction through broken reasoning; a P&L scorer records "
                "them as successes and repeats the reasoning that produced them"
            ),
        }


def grade_magnitude(
    predicted_bps: int | None, realised_bps: int, *, tolerance: float = 0.5
) -> LinkGrade:
    """Grade a magnitude claim. Generous by default — sizing is harder than direction.

    A prediction within ``tolerance`` of the realised move counts. Predicting 50bps and getting
    30 is a reasonable call; predicting 50 and getting 400 is not the same event.

    **An absent prediction grades UNCERTAIN, never WRONG.** The magnitude used to default to zero
    before reaching here, so a model that named no size was scored against the realised move and
    marked wrong for a number it never gave. Grading our own default is not grading the model.
    """
    if predicted_bps is None:
        return LinkGrade.UNCERTAIN
    if realised_bps == 0:
        return LinkGrade.UNCERTAIN
    ratio = abs(predicted_bps) / abs(realised_bps)
    return LinkGrade.CORRECT if (1 - tolerance) <= ratio <= (1 + tolerance) else LinkGrade.WRONG


__all__ = [
    "STANDARD_STEPS", "CausalChain", "CausalLedger", "Link", "LinkGrade",
    "chain_from_response", "grade_magnitude",
]
