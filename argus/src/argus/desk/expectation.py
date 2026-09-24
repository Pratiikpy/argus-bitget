"""The expectation gap — what the market expects against what the company has delivered.

Track 3's **Information Extraction & Signal Generation** sub-theme names "expectation gap detection"
as its example. Until now nothing in ARGUS computed one: `market/fundamentals.py` reads what was
**reported** and `market/estimates.py` reads what is **expected**, and the two had never been put
side by side. A reported number without an expectation is not a surprise, it is a fact with no
direction — which is the exact complaint the desk itself wrote into ledger entry 41.

**What cannot be computed, stated first.** A classic earnings surprise is the reported figure
against *the consensus that existed for that same quarter*. Yahoo's `earningsTrend` publishes only
forward periods — the current quarter, the next, the current fiscal year and the next — so **no
consensus for an already-reported quarter is available from any keyless source we found**. Seeking
Alpha's endpoint carries trailing periods and returns HTTP 403 to us (`market/estimates.py`). So
:class:`Gap` reports ``surprise = None`` with the reason, and never manufactures one by comparing a
reported quarter to a forward estimate that was never about it.

**What can be computed, and is more useful anyway.** Three things, all from real filings and a live
consensus:

* **implied growth** — the consensus for the coming quarter against what the company actually
  delivered in the *same quarter a year earlier*, matched on period end rather than on ordering;
* **delivered growth** — the same year-over-year comparison for each quarter already reported;
* **the direction of travel** — whether the growth being *expected* is faster or slower than the
  growth recently *delivered*, and which way analysts have been revising.

Measured on NVDA, 2026-09-13: delivered year-over-year growth ran +214% then +128%, while the
consensus for the coming quarter implies roughly +90% — **deceleration that analysts are still
revising upward**, 35 raises against 1 cut in thirty days. That pair is the signal. Neither half
says it alone, and no single figure anywhere in the filings or the estimates contains it.

**Fiscal calendars do not divide by 365.** NVDA's quarter ended 2026-07-26 and the consensus period
ends 2026-10-31; the year-ago comparable ends 2025-10-26, which is 370 days earlier, not 365.
Matching is therefore on a tolerance band around a year and refuses rather than guesses when nothing
falls inside it — pairing the wrong quarters would produce a growth rate that looks precise and
describes nothing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any, Protocol

YEAR_MATCH_DAYS = (330, 400)
"""How far from 365 days a "same quarter last year" match may sit.

Fiscal quarters drift against the calendar — NVDA's comparable is 370 days back — and a 52/53-week
retailer can move further. Wide enough to catch a real comparable, narrow enough that the adjacent
quarter (roughly 90 or 275 days away) can never be mistaken for one."""

MATERIAL_DECELERATION_PP = 20.0
"""Percentage points of growth change before the direction of travel is called a change at all.

Growth rates are noisy and a two-point move is not a deceleration. Chosen, not measured, and
labelled as a description of the arithmetic rather than a forecast."""


class Reported(Protocol):
    """A filed figure. Satisfied by :class:`argus.market.fundamentals.Fact`."""

    @property
    def concept(self) -> str: ...
    @property
    def value(self) -> float: ...
    @property
    def end(self) -> date: ...
    @property
    def filed(self) -> date: ...
    @property
    def is_quarterly(self) -> bool: ...


class Expected(Protocol):
    """A consensus estimate. Satisfied by :class:`argus.market.estimates.Consensus`."""

    @property
    def period(self) -> str: ...
    @property
    def end_date(self) -> str: ...
    @property
    def eps_avg(self) -> float | None: ...
    @property
    def analysts(self) -> int | None: ...


def _growth(now: float, before: float) -> float | None:
    """Year-over-year growth as a percentage. ``None`` when the base cannot carry one.

    A zero base makes growth undefined rather than infinite, and a **negative** base makes the sign
    meaningless: going from -1.00 to +0.50 is not "150% growth" in any sense a reader would accept.
    Both are refused instead of being dressed up.
    """
    if before == 0 or before < 0:
        return None
    return (now - before) / before * 100.0


def _match_year_ago(target: date, reported: Sequence[Reported]) -> Reported | None:
    """The quarter ending roughly a year before ``target``, or nothing."""
    low, high = YEAR_MATCH_DAYS
    best: tuple[int, Reported] | None = None
    for fact in reported:
        if not fact.is_quarterly:
            continue
        delta = (target - fact.end).days
        if not low <= delta <= high:
            continue
        distance = abs(delta - 365)
        if best is None or distance < best[0]:
            best = (distance, fact)
    return None if best is None else best[1]


@dataclass(frozen=True, slots=True)
class Gap:
    """What is expected, what was delivered, and the distance between them."""

    ticker: str
    concept: str
    expected_period_end: str
    expected_value: float | None
    analysts: int | None
    comparable_period_end: str | None
    comparable_value: float | None
    implied_growth_pct: float | None
    delivered_growth_pct: tuple[tuple[str, float], ...]
    """Year-over-year growth for each already-reported quarter that had a comparable."""

    revision_direction: str
    revisions_up_30d: int
    revisions_down_30d: int
    surprise: None = None
    """Always ``None``. A surprise needs the consensus that existed for a reported quarter, and no
    keyless source publishes it; the reason travels with the record rather than in a footnote."""

    surprise_reason: str = (
        "no consensus is published for an already-reported quarter by any keyless source we could "
        "reach, so a reported-against-expected surprise is unavailable rather than zero"
    )

    @property
    def latest_delivered_pct(self) -> float | None:
        return self.delivered_growth_pct[0][1] if self.delivered_growth_pct else None

    @property
    def direction_of_travel(self) -> str:
        """Whether expected growth runs ahead of, behind, or with recently delivered growth."""
        delivered = self.latest_delivered_pct
        if self.implied_growth_pct is None or delivered is None:
            return "unknown"
        change = self.implied_growth_pct - delivered
        if abs(change) < MATERIAL_DECELERATION_PP:
            return "steady"
        return "accelerating" if change > 0 else "decelerating"

    @property
    def expectations_rising_into_deceleration(self) -> bool:
        """Analysts revising up while the growth they expect is slowing.

        Worth naming because it is the configuration that most often ends in a de-rating: the
        estimate goes up, the growth rate comes down, and the two are reported in different places
        so nobody sees them together.
        """
        return self.direction_of_travel == "decelerating" and self.revision_direction == "up"

    def as_dict(self) -> dict[str, Any]:
        def r(x: float | None) -> float | None:
            return None if x is None else round(x, 2)
        return {
            "ticker": self.ticker,
            "concept": self.concept,
            "expected_period_end": self.expected_period_end,
            "expected_value": self.expected_value,
            "analysts": self.analysts,
            "comparable_period_end": self.comparable_period_end,
            "comparable_value": self.comparable_value,
            "implied_growth_pct": r(self.implied_growth_pct),
            "delivered_growth_pct": [
                {"period_end": p, "growth_pct": round(v, 2)} for p, v in self.delivered_growth_pct
            ],
            "direction_of_travel": self.direction_of_travel,
            "revision_direction": self.revision_direction,
            "revisions_up_30d": self.revisions_up_30d,
            "revisions_down_30d": self.revisions_down_30d,
            "surprise": self.surprise,
            "surprise_reason": self.surprise_reason,
        }

    def render(self) -> list[str]:
        lines: list[str] = []
        if self.expected_value is None:
            return [f"[expectation] no consensus {self.concept} published for {self.ticker}"]
        who = f"{self.analysts} analysts" if self.analysts else "an unstated number of analysts"
        lines.append(
            f"[expectation] {who} expect {self.concept} of {self.expected_value:.4f} for the "
            f"quarter ending {self.expected_period_end}"
        )
        if self.implied_growth_pct is not None and self.comparable_period_end:
            lines.append(
                f"[expectation] that implies {self.implied_growth_pct:+.1f}% against the "
                f"{self.comparable_value:.4f} actually delivered for the quarter ending "
                f"{self.comparable_period_end}"
            )
        if self.delivered_growth_pct:
            trail = ", ".join(f"{v:+.1f}%" for _, v in self.delivered_growth_pct[:3])
            lines.append(f"[expectation] delivered growth, newest first: {trail}")
        if self.direction_of_travel != "unknown":
            lines.append(
                f"[expectation] the growth being expected is {self.direction_of_travel} against "
                f"the growth recently delivered"
            )
        lines.append(
            f"[expectation] estimates revised {self.revision_direction}: "
            f"{self.revisions_up_30d} up / {self.revisions_down_30d} down in 30 days"
        )
        if self.expectations_rising_into_deceleration:
            lines.append(
                "[expectation] analysts are raising their numbers while the growth rate they "
                "expect is slowing — the two are published in different places and rarely read "
                "together"
            )
        lines.append(f"[expectation] {self.surprise_reason}")
        return lines


def detect(
    *,
    ticker: str,
    reported: Sequence[Reported],
    consensus: Sequence[Expected],
    revision_direction: str = "mixed",
    revisions_up_30d: int = 0,
    revisions_down_30d: int = 0,
    concept: str = "eps_diluted",
    period: str = "0q",
) -> Gap:
    """Put the coming quarter's consensus beside what was actually delivered a year earlier.

    ``reported`` are quarterly facts newest first, already point-in-time filtered by the caller —
    this function does no as-of gating of its own and must not be handed unfiltered history.
    """
    quarters = [f for f in reported if f.is_quarterly and f.concept == concept]
    forward = next((c for c in consensus if c.period == period), None)

    if forward is None or forward.eps_avg is None:
        return Gap(
            ticker=ticker, concept=concept, expected_period_end="", expected_value=None,
            analysts=None, comparable_period_end=None, comparable_value=None,
            implied_growth_pct=None, delivered_growth_pct=(),
            revision_direction=revision_direction,
            revisions_up_30d=revisions_up_30d, revisions_down_30d=revisions_down_30d,
        )

    target: date | None
    try:
        target = date.fromisoformat(forward.end_date)
    except ValueError:
        # A consensus with an unparseable period end cannot be matched to anything. Refusing here
        # is the difference between "no comparable" and a comparable picked from the wrong year.
        target = None

    comparable = _match_year_ago(target, quarters) if target else None
    implied = (
        _growth(forward.eps_avg, comparable.value) if comparable is not None else None
    )

    delivered: list[tuple[str, float]] = []
    for fact in quarters[:4]:
        prior = _match_year_ago(fact.end, quarters)
        if prior is None:
            continue
        value = _growth(fact.value, prior.value)
        if value is not None:
            delivered.append((fact.end.isoformat(), value))

    return Gap(
        ticker=ticker,
        concept=concept,
        expected_period_end=forward.end_date,
        expected_value=forward.eps_avg,
        analysts=forward.analysts,
        comparable_period_end=None if comparable is None else comparable.end.isoformat(),
        comparable_value=None if comparable is None else comparable.value,
        implied_growth_pct=implied,
        delivered_growth_pct=tuple(delivered),
        revision_direction=revision_direction,
        revisions_up_30d=revisions_up_30d,
        revisions_down_30d=revisions_down_30d,
    )


def main() -> int:
    import argparse
    from datetime import UTC, datetime

    from argus.market.estimates import EstimatesSource
    from argus.market.fundamentals import FundamentalsSource

    parser = argparse.ArgumentParser(
        description="Expectation gap: what is expected against what was delivered."
    )
    parser.add_argument("ticker", nargs="?", default="NVDA")
    args = parser.parse_args()

    now = datetime.now(UTC)
    facts, _ = FundamentalsSource().facts(args.ticker, concept="eps_diluted", as_of=now)
    estimates = EstimatesSource().fetch(args.ticker, as_of=now)
    current = next((c for c in estimates if c.period == "0q"), None)

    gap = detect(
        ticker=args.ticker,
        reported=facts,
        consensus=estimates,
        revision_direction="mixed" if current is None else current.revisions.direction,
        revisions_up_30d=0 if current is None else current.revisions.up_30d,
        revisions_down_30d=0 if current is None else current.revisions.down_30d,
    )
    for line in gap.render():
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "MATERIAL_DECELERATION_PP",
    "YEAR_MATCH_DAYS",
    "Expected",
    "Gap",
    "Reported",
    "detect",
    "main",
]
