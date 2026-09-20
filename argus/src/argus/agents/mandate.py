"""Mandate injection — the same evidence, two traders, two defensible answers.

Track 3 names "personalized thesis" as a scored criterion, and the submission form explicitly
refuses "all traders" as an answer. :class:`argus.desk.workbench.TraderProfile` already encodes who
the desk is for — capital, position limits, holding horizon, loss tolerance, preferred evidence —
and its own docstring states the test: *two profiles get different verdicts on identical market
state*. Nothing conditioned on it. The profile shaped the risk limits and never reached the
reasoning, so a conservative pensioner and an aggressive event trader received word-for-word the
same thesis and then had it clipped differently afterwards.

Clipping is not personalisation. A position halved after the fact still rests on reasoning written
for somebody else, and the trader reading it cannot tell which parts were meant for them.

This module turns a profile into the constraints that reach the PM *before* it reasons (wired at
``agents/desk.py``'s ``mandate_block`` and ``ordered_evidence``, both built ahead of
``MetaPM.decide()`` and read inside ``MarketFrame.to_prompt_block()``), and into an ordering over
the evidence the PM sees that reflects what this trader actually acts on. The per-sub-theme
analysts (event / sentiment / earnings) see a disjoint slice of evidence routed by source TYPE, not
by this trader's stated preference — reordering an already-homogeneous slice has little to order,
so the mandate's ordering promise is scoped to the PM's mixed evidence list, where preference
between sources is the thing being expressed. Two things follow, and both are testable:

* **The mandate is a hard frame, not a preference.** A 48-hour event trader is told that a thesis
  requiring a six-month hold is out of mandate and should be declined rather than resized. That is
  a different answer, not a smaller one.
* **Evidence is ordered, never filtered.** The conservative profile prefers filings; it does not
  become blind to news. Dropping evidence a profile does not favour would let a preference hide a
  catalyst, which is how personalisation becomes a blindfold. Preferred sources are surfaced first
  and the rest still arrive, marked.

A prior version of this docstring cited ``agno-agi/investment-team``'s ``teams/coordinate_team.py``
as prior art for constraining an agent's reasoning before it runs, from a ``context/`` directory of
per-agent mandates. Checked against the file directly (``gh api
repos/agno-agi/investment-team/contents/teams/coordinate_team.py``, 2026-09-15): the file is 63
lines total, has no ``context/`` directory reference and no per-agent mandate mechanism of any kind
— its only mandate-shaped content is one static line in a fixed instruction list, ``"Ensure all
decisions comply with the fund mandate."`` The citation was wrong and is withdrawn rather than
repeated. Three studied systems were checked directly for this specific mechanism — a real trader's
own constraints, reaching an agent's reasoning before it runs — and none of them do it:
``hkuds/vibe-trading`` (its mandate machinery, read in full across ``enforcement.py``,
``sdk_order_gate.py``, ``advisory/__init__.py`` and ``propose_mandate_tool.py``, is entirely
pre-broker order-gating and post-hoc risk commentary; nothing reaches an LLM prompt — see
``eval/mandate_comparison.py``), ``agno-agi/investment-team`` (above), and
``TauricResearch/TradingAgents`` (its Aggressive/Conservative/Neutral "risk profiles" are fixed
debate personas arguing with each other, not one real trader's own stated constraints —
``research/architecture/tradingagents.md:729``). This is **NOT VERIFIED** as true of the wider
corpus — only these three were checked by name — so the honest claim is narrower than "nobody does
this": nobody *checked* does this, and finding an accurate precedent (rather than replacing one
guess with another) was left for whoever checks a fourth.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from argus.desk.workbench import TraderProfile
from argus.truth.evidence import Evidence

HORIZON_SLACK = 1.5
"""A thesis may ask for up to half again the mandate horizon before it is out of mandate.

A hard equality would decline a 50-hour thesis for a 48-hour trader, which is a rounding error
being treated as a policy breach.
"""


@dataclass(frozen=True)
class Mandate:
    """What a profile permits, in the form the reasoning layer needs it."""

    profile: TraderProfile

    @property
    def max_position_notional(self) -> Decimal:
        return self.profile.capital * self.profile.max_position_pct / Decimal("100")

    @property
    def horizon_ceiling_hours(self) -> float:
        return self.profile.holding_horizon_hours * HORIZON_SLACK

    def permits_horizon(self, hours: float) -> bool:
        return hours <= self.horizon_ceiling_hours

    def refuses_symbol(self, symbol: str) -> bool:
        """A named exclusion is absolute. No thesis outvotes it."""
        return symbol in self.profile.excluded_symbols

    def out_of_mandate(
        self,
        *,
        horizon_hours: float,
        notional: Decimal,
        symbol: str = "",
        hedge_available: bool | None = None,
        confidence: float | None = None,
        open_positions: int | None = None,
    ) -> tuple[str, ...]:
        """Every reason this trade does not belong to this trader. Empty means it does.

        The later arguments default to ``None`` meaning "not supplied", which is different from
        "supplied and fine": a caller that cannot say whether a hedge exists must not have that
        read as a hedge existing. Only a supplied value can produce a breach.
        """
        reasons: list[str] = []
        if not self.permits_horizon(horizon_hours):
            reasons.append(
                f"the thesis needs {horizon_hours:.0f}h and the mandate horizon is "
                f"{self.profile.holding_horizon_hours}h; this is a different trader's trade, "
                f"not a smaller version of this one"
            )
        if notional > self.max_position_notional:
            reasons.append(
                f"notional {notional} exceeds the {self.profile.max_position_pct}% position "
                f"limit ({self.max_position_notional}) on {self.profile.capital} of capital"
            )
        if symbol and self.refuses_symbol(symbol):
            reasons.append(
                f"{symbol} is on this mandate's exclusion list and is not held at any size"
            )
        if self.profile.requires_hedge and hedge_available is False:
            reasons.append(
                "this mandate does not carry unhedgeable exposure, and no hedge is placeable in "
                "this session"
            )
        if confidence is not None and confidence < self.profile.min_confidence:
            reasons.append(
                f"stated confidence {confidence:.2f} is below this mandate's "
                f"{self.profile.min_confidence:.2f} floor"
            )
        if (
            self.profile.max_concurrent_positions
            and open_positions is not None
            and open_positions >= self.profile.max_concurrent_positions
        ):
            reasons.append(
                f"{open_positions} position(s) already open, at this mandate's cap of "
                f"{self.profile.max_concurrent_positions}"
            )
        return tuple(reasons)

    def render(self) -> str:
        """The frame the analysts and the PM reason inside."""
        preferred = ", ".join(self.profile.preferred_evidence) or "no stated preference"
        return (
            f"MANDATE — {self.profile.name}. "
            f"Capital {self.profile.capital}. "
            f"No single position above {self.profile.max_position_pct}% "
            f"({self.max_position_notional}); no sector above {self.profile.max_sector_pct}%. "
            f"Holding horizon {self.profile.holding_horizon_hours}h; a thesis that needs longer "
            f"than {self.horizon_ceiling_hours:.0f}h is out of mandate and should be declined "
            f"rather than resized. Loss tolerance {self.profile.loss_tolerance_pct}%. "
            f"This trader acts on: {preferred}. "
            + (
                "This mandate does NOT carry exposure it cannot hedge right now. "
                if self.profile.requires_hedge else ""
            )
            + (
                f"It never holds: {', '.join(self.profile.excluded_symbols)}. "
                if self.profile.excluded_symbols else ""
            )
            + (
                f"It needs at least {self.profile.min_confidence:.2f} confidence to act. "
                if self.profile.min_confidence else ""
            )
            +
            "Evidence outside that list is still shown and still counts; it is ordered lower "
            "because it is less often actionable for this trader, not because it is less true."
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile.name,
            "capital": str(self.profile.capital),
            "max_position_pct": str(self.profile.max_position_pct),
            "max_position_notional": str(self.max_position_notional),
            "horizon_hours": self.profile.holding_horizon_hours,
            "horizon_ceiling_hours": self.horizon_ceiling_hours,
            "loss_tolerance_pct": str(self.profile.loss_tolerance_pct),
            "preferred_evidence": list(self.profile.preferred_evidence),
            "requires_hedge": self.profile.requires_hedge,
            "excluded_symbols": list(self.profile.excluded_symbols),
            "max_concurrent_positions": self.profile.max_concurrent_positions,
            "min_confidence": self.profile.min_confidence,
        }


def order_evidence(
    evidence: Sequence[Evidence], *, mandate: Mandate
) -> tuple[tuple[Evidence, bool], ...]:
    """Order evidence by what this trader acts on. Nothing is removed.

    Returns each item with a flag saying whether it is one this profile prefers, so the caller can
    mark it rather than silently reordering. Within each group the original order is kept, which
    for the evidence feed means newest first.

    Filtering would be easier and is the wrong shape: a conservative profile that never sees a news
    item cannot decline it, and a preference that hides a catalyst has become a blindfold.
    """
    preferred = set(mandate.profile.preferred_evidence)
    first = [(e, True) for e in evidence if e.source in preferred]
    rest = [(e, False) for e in evidence if e.source not in preferred]
    return tuple(first + rest)


def frame_for(
    mandate: Mandate, evidence: Sequence[Evidence]
) -> tuple[str, tuple[str, ...]]:
    """The mandate line plus the evidence as the reasoning layer should see it."""
    ordered = order_evidence(evidence, mandate=mandate)
    aside = "  [outside this mandate's usual sources]"
    rendered = tuple(
        e.render() + ("" if is_preferred else aside) for e, is_preferred in ordered
    )
    return mandate.render(), rendered


__all__ = ["HORIZON_SLACK", "Mandate", "frame_for", "order_evidence"]
