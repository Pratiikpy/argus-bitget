"""Which analysts to run, decided from the evidence and priced against what running them costs.

Every multi-agent finance system in this corpus runs a fixed panel. TradingAgents takes a static
list from the caller and validates the names against a registry —
``tradingagents/graph/analyst_execution.py:56-69``, read, and its only dynamic filter is by asset
class (``cli/utils.py:90-99``: drop the fundamentals analyst for crypto), decided before any data is
fetched. FinRobot binds one agent per report section by name. RD-Agent runs its whole pipeline every
time. None of them reads the evidence to decide who should look at it, and **none skips an agent
because running it is not worth the cost** — that search returned nothing across the corpus.

The second half is the part ARGUS is positioned to do and the others are not: this project already
prices its own deliberation in basis points (`agents/meta_pm.py:65`), so "is this analyst worth
running" is an arithmetic question here and a matter of taste everywhere else. Off-hours a full
reasoning budget costs more than the 12bps round trip; an analyst asked to read one stale headline
is spending real money to produce a shrug.

**What this is not.** It is not an LLM router. A model choosing which models to run adds a call, a
schema to validate, and a second thing that can hallucinate, in front of a decision that is already
model-made. Selection here is deterministic Python over the evidence the desk holds, so the record
says exactly why each analyst ran or did not, and the same inputs always produce the same panel.

**The property that makes it safe.** A skipped analyst is *recorded as skipped*, with the evidence
it would have seen counted in the record. The failure mode being avoided is a router that quietly
drops evidence and leaves a decision looking as though nothing relevant existed. A skip is a
statement about cost, never a claim that the evidence was worthless, and :meth:`Selection.render`
says so in those words.

**A near miss still runs.** If nothing clears its hurdle but some analyst holds evidence it can
actually act on, that one runs anyway and :attr:`Selection.floor_applied` marks the case, so a panel
reached that way is never mistaken for a chosen one. The floor stops there: an analyst whose
evidence contains nothing it reads will return a shrug with certainty, and buying a certain shrug is
the exact cost this module exists to avoid. Selecting nobody is therefore a reachable outcome, and
it is not a desk that looked at nothing — the cross-asset analyst sits outside this selection and
runs on every cycle.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

from argus.truth.evidence import Evidence

MIN_EVIDENCE = 1
"""Below this an analyst has nothing to read and is skipped regardless of cost."""

STALE_AFTER = timedelta(hours=48)
"""Evidence older than this counts toward relevance at a discount rather than at full weight.

Not a hard cut: a Form 4 filed three days ago is still the most recent thing that insider chose to
do, and discarding it would be worse than weighting it down."""

STALE_WEIGHT = 0.4

MIN_RELEVANCE = 0.35
"""The relevance an analyst needs before its share of the deliberation cost is judged worth paying.

Calibrated against the live evidence mix rather than chosen for roundness: one 0.1-credibility feed
health line scores 0.1 and must not summon an analyst, while a single full-credibility filing scores
1.0 and must."""


# Which sources each analyst is entitled to read. Taken from `agents/desk.py:140-143` so the two
# cannot drift: selection that disagreed with routing would skip an analyst over evidence it was
# never going to see.
SOURCES: dict[str, frozenset[str]] = {
    "event": frozenset({"sec-edgar", "news", "macro"}),
    "sentiment": frozenset({"social"}),
    "earnings": frozenset({"filing", "transcript"}),
}

# Content an analyst needs before its evidence is worth its cost. A source match alone is weak: a
# generic headline arrives on the `news` channel and so does a merger announcement. Matched against
# the rendered claim, which is what the analyst itself would read.
_MATERIAL = re.compile(
    r"\b(?:8-K|10-Q|10-K|acquisition|merger|guidance|downgrade|upgrade|recall|resign|"
    r"investigation|lawsuit|bankrupt|dividend|buyback|split|halt|offering|insider|form\s*4|"
    r"results|earnings|revenue|margin|outlook|contract|approval|breach)\b",
    re.I,
)

_EARNINGS_CONTENT = re.compile(
    r"\b(?:revenue|earnings|eps|margin|guidance|quarter|fiscal|net\s+income|outlook|"
    r"beat|miss|consensus)\b",
    re.I,
)

_CONTENT: dict[str, re.Pattern[str] | None] = {
    "event": _MATERIAL,
    "earnings": _EARNINGS_CONTENT,
    # Sentiment reads mood, and mood has no vocabulary that separates signal from noise the way a
    # material-event word list does. Anything on the social channel is legitimately its business, so
    # it is gated on volume and credibility alone rather than on a keyword list that would encode a
    # guess about what sentiment looks like.
    "sentiment": None,
}


@dataclass(frozen=True, slots=True)
class Assessment:
    """One analyst, the evidence it would read, and what that is worth."""

    analyst: str
    evidence_count: int
    relevance: float
    """0..1. Credibility-weighted, discounted for staleness, gated on content where a content gate
    is meaningful for that analyst."""

    cost_bps: Decimal
    """This analyst's share of the cycle's deliberation cost."""

    run: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "analyst": self.analyst,
            "evidence_count": self.evidence_count,
            "relevance": round(self.relevance, 3),
            "cost_bps": str(self.cost_bps),
            "run": self.run,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class Selection:
    """The panel, and the full account of who was left out and why."""

    assessments: tuple[Assessment, ...]
    floor_applied: bool
    orphaned: tuple[str, ...] = ()
    """Evidence sources that no analyst is entitled to read, so nobody saw them.

    **Found by a demonstration abstaining for the wrong reason.** A scenario fixture wrote an 8-K
    with ``source="sec"`` instead of the canonical ``sec-edgar``. The item was gathered, counted in
    the evidence list and displayed in the trace — and routed to nobody, because
    :data:`SOURCES` maps channels and an unknown one matches no analyst. The panel note said
    "event not run - no evidence on its channels", which is a true statement about the *analyst* and
    says nothing about the *item*, and the decision that followed looked like judgement.

    :attr:`evidence_not_read` cannot catch this: it sums the evidence belonging to skipped analysts,
    and an orphan belongs to none of them, so it is invisible to a counter whose docstring promises
    it is never zero silently. The live sources are all routed
    (``sec-edgar``/``filing``/``social``), so this is a guard against the next source added rather
    than a live defect — which is exactly when it is cheap to add."""
    """The panel would have been empty and the strongest candidate was run anyway. A panel of one
    reached this way is a different thing from a panel of one that was chosen, and conflating them
    would let a desk report a considered decision it never made."""

    @property
    def run(self) -> tuple[str, ...]:
        return tuple(a.analyst for a in self.assessments if a.run)

    @property
    def skipped(self) -> tuple[Assessment, ...]:
        return tuple(a for a in self.assessments if not a.run)

    @property
    def bps_saved(self) -> Decimal:
        """Deliberation cost not spent. Real money on a venue where the fee is 12bps."""
        return sum((a.cost_bps for a in self.skipped), Decimal("0"))

    @property
    def evidence_not_read(self) -> int:
        """How many pieces of evidence no analyst looked at. Never zero silently."""
        return sum(a.evidence_count for a in self.skipped)

    def as_dict(self) -> dict[str, Any]:
        return {
            "run": list(self.run),
            "floor_applied": self.floor_applied,
            "bps_saved": str(self.bps_saved),
            "evidence_not_read": self.evidence_not_read,
            "orphaned_sources": list(self.orphaned),
            "assessments": [a.as_dict() for a in self.assessments],
        }

    def render(self) -> list[str]:
        lines = [
            f"[panel] {len(self.run)} of {len(self.assessments)} analysts run: "
            f"{', '.join(self.run) or 'none'}"
        ]
        if self.orphaned:
            # Loud, and before the per-analyst lines. An item nobody read is not a cost decision,
            # it is evidence that silently left the building.
            lines.append(
                f"[panel] WARNING {len(self.orphaned)} evidence source(s) reached no analyst: "
                f"{', '.join(self.orphaned)}. No analyst is entitled to read them, so they were "
                f"counted as evidence and used by nobody. Expected one of "
                f"{', '.join(sorted(set().union(*SOURCES.values())))}"
            )
        for skipped in self.skipped:
            # The "went unexamined" clause is only true when something was there to examine. On the
            # live cycle it read "0 piece(s) of evidence went unexamined; that is a judgement about
            # cost" — three claims in one sentence, none of them applicable. An analyst with an
            # empty channel was not skipped on cost; there was nothing to skip.
            if skipped.evidence_count:
                lines.append(
                    f"[panel] {skipped.analyst} not run — {skipped.reason}. "
                    f"{skipped.evidence_count} piece(s) of evidence it would have read went "
                    f"unexamined; that is a judgement about cost, not about their worth"
                )
            else:
                lines.append(f"[panel] {skipped.analyst} not run — {skipped.reason}")
        if self.bps_saved > 0:
            lines.append(
                f"[panel] {self.bps_saved}bps of deliberation not spent, against a "
                f"12bps round trip"
            )
        if self.floor_applied:
            lines.append(
                "[panel] nothing cleared its own cost; the strongest candidate was run anyway "
                "because it held evidence it could act on and a near miss is not a reason to "
                "look away"
            )
        elif not self.run:
            lines.append(
                "[panel] no analyst held evidence it could act on, so none was run; this is a "
                "statement about what arrived, and the cross-asset analyst still ran"
            )
        return lines


def _weight(evidence: Evidence, as_of: datetime) -> float:
    """One piece of evidence's contribution: its credibility, discounted if it is stale."""
    weight = max(0.0, min(1.0, evidence.credibility))
    if as_of - evidence.available_at > STALE_AFTER:
        weight *= STALE_WEIGHT
    return weight


def assess(
    analyst: str,
    evidence: Sequence[Evidence],
    *,
    as_of: datetime,
    cost_bps: Decimal,
) -> Assessment:
    """Score one analyst against the evidence it is entitled to read."""
    sources = SOURCES.get(analyst, frozenset())
    mine = [e for e in evidence if e.source in sources]

    if len(mine) < MIN_EVIDENCE:
        return Assessment(
            analyst=analyst, evidence_count=0, relevance=0.0, cost_bps=cost_bps, run=False,
            reason=f"no evidence on its channels ({', '.join(sorted(sources)) or 'none'})",
        )

    # Relevance counts only evidence whose content this analyst can actually act on, but the count
    # reported below is every piece it would have read — so a skip states the full extent of what
    # went unexamined, not the part that happened to match a keyword.
    pattern = _CONTENT.get(analyst)
    scoring = [e for e in mine if pattern.search(e.claim)] if pattern is not None else list(mine)

    relevance = min(1.0, sum(_weight(e, as_of) for e in scoring))

    if relevance < MIN_RELEVANCE:
        return Assessment(
            analyst=analyst, evidence_count=len(mine), relevance=relevance, cost_bps=cost_bps,
            run=False,
            reason=(
                f"relevance {relevance:.2f} below {MIN_RELEVANCE} — its evidence is stale, "
                f"low-credibility, or carries nothing this analyst reads"
            ),
        )
    return Assessment(
        analyst=analyst, evidence_count=len(mine), relevance=relevance, cost_bps=cost_bps,
        run=True,
        reason=f"relevance {relevance:.2f} on {len(mine)} piece(s) of evidence",
    )


def select(
    evidence: Sequence[Evidence],
    *,
    as_of: datetime,
    deliberation_bps: Decimal,
    analysts: Sequence[str] = ("event", "sentiment", "earnings"),
) -> Selection:
    """Choose the panel.

    ``deliberation_bps`` is the cycle's total reasoning cost from
    :func:`argus.agents.meta_pm.deliberation_cost_bps`; it is divided evenly across the candidates,
    because each analyst call carries the same reasoning budget. The division is stated rather than
    measured per-analyst: attributing latency to individual calls would need per-call timing the
    desk does not keep, and inventing a split would be a guess dressed as arithmetic.
    """
    share = (
        (deliberation_bps / Decimal(len(analysts))) if analysts else Decimal("0")
    )
    assessments = [
        assess(name, evidence, as_of=as_of, cost_bps=round(share, 3)) for name in analysts
    ]

    floor_applied = False
    if not any(a.run for a in assessments) and assessments:
        best = max(assessments, key=lambda a: (a.relevance, a.evidence_count))
        # `relevance > 0`, not `evidence_count > 0`. The floor exists to stop a *near miss* being
        # thrown away — an analyst holding evidence it can act on that fell just under the bar. An
        # analyst whose evidence contains nothing it reads at all will return a shrug with
        # certainty, and paying for a guaranteed shrug is the cost the module exists to avoid.
        # The panel is not left bare either way: the cross-asset analyst is outside this selection
        # and always runs (`agents/desk.py`), so "no analyst selected" means no *evidence* analyst,
        # never a desk that looked at nothing.
        if best.relevance > 0:
            floor_applied = True
            assessments = [
                (
                    Assessment(
                        analyst=a.analyst, evidence_count=a.evidence_count, relevance=a.relevance,
                        cost_bps=a.cost_bps, run=True,
                        reason=(
                            f"{a.reason}; run anyway as the strongest candidate, because a panel "
                            f"of nobody is not an abstention"
                        ),
                    )
                    if a is best
                    else a
                )
                for a in assessments
            ]

    known = set().union(*SOURCES.values())
    orphaned = tuple(sorted({str(e.source) for e in evidence if str(e.source) not in known}))
    return Selection(
        assessments=tuple(assessments), floor_applied=floor_applied, orphaned=orphaned,
    )


__all__ = [
    "MIN_EVIDENCE",
    "MIN_RELEVANCE",
    "SOURCES",
    "STALE_AFTER",
    "STALE_WEIGHT",
    "Assessment",
    "Selection",
    "assess",
    "select",
]
