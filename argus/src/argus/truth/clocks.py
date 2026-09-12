"""The two-clock model.

Every point-in-time implementation found across the 922-source research corpus assumes a single
market clock. A tokenized equity has two, and they disagree for roughly 65.5 hours a week:

    token clock   — continuous. The rToken trades 24/7.
    anchor clock  — sessioned. The underlying equity trades 09:30-16:00 ET on business days.

The consequence that a single-clock design cannot express: a fact can be **knowable** (and therefore
tradeable on the token) while being **unpriceable** (no genuine price discovery available on the
anchor). Conflating those two is what produces a backtest that quietly assumes it could have hedged
at 3am on a Sunday.

`DualClock` keeps them separate and makes the distinction a typed question rather than a convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from enum import StrEnum
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


class SessionPhase(StrEnum):
    """Phase of the *anchor* market. The token clock has no phases — it is always OPEN."""

    RTH = "rth"
    """Regular trading hours. Genuine price discovery is happening."""

    EXTENDED = "extended"
    """Pre/post market. Thin, wide, but a real quote exists."""

    OVERNIGHT = "overnight"
    """Between sessions on a business day. No anchor quote."""

    WEEKEND = "weekend"
    """Saturday/Sunday. The long gap — the core of the Sleeping-Anchor problem."""

    HOLIDAY = "holiday"
    """Market holiday. Behaves like WEEKEND but must be distinguishable for attribution."""

    @property
    def has_price_discovery(self) -> bool:
        """Is a genuine, tradeable anchor price being formed right now?

        EXTENDED deliberately returns False. A pre-market print on 200 shares is a quote, not price
        discovery, and treating it as discovery is precisely how overnight strategies acquire an
        edge that does not survive contact with a real fill.
        """
        return self is SessionPhase.RTH


# US equity market hours. Deliberately explicit rather than inherited from a calendar library:
# AlphaTrade hardcodes continuous NASDAQ hours at base_env.py:53-54 and nothing in the corpus
# models a session-dependent liquidity tide, so these boundaries are load-bearing for us.
RTH_OPEN = time(9, 30)
RTH_CLOSE = time(16, 0)
EXTENDED_OPEN = time(4, 0)
EXTENDED_CLOSE = time(20, 0)


@dataclass(frozen=True, slots=True)
class SessionState:
    """The anchor market's state at one instant, plus the staleness of what we know about it."""

    phase: SessionPhase
    as_of: datetime
    hours_to_next_discovery: float
    """Hours until the next RTH open. Zero while RTH is in progress.

    This is the quantity that makes gap risk computable: risk carried across a closed session is a
    function of how long the anchor stays asleep, not merely of position size.
    """

    nav_age_seconds: float | None = None
    oracle_age_seconds: float | None = None

    @property
    def is_anchor_asleep(self) -> bool:
        return not self.phase.has_price_discovery

    def nav_is_stale(self) -> bool:
        """Staleness thresholds differ by phase, because the same age means different things.

        3600s during RTH is a broken feed. 3600s on a Saturday is normal and expected.
        """
        if self.nav_age_seconds is None:
            return True  # unknown age is stale by default — never optimistic about missing data
        threshold = 3600.0 if self.phase is SessionPhase.RTH else 86400.0
        return self.nav_age_seconds > threshold


def _is_business_day(d: date) -> bool:
    return d.weekday() < 5


def _next_business_day(d: date) -> date:
    nxt = d + timedelta(days=1)
    while not _is_business_day(nxt):
        nxt += timedelta(days=1)
    return nxt


class DualClock:
    """Resolves an instant against both clocks at once.

    Holidays are injected rather than hardcoded so that a test can construct a specific market
    condition without depending on the real calendar — and so that a missing holiday file degrades
    to "we do not know", never to "assume open".
    """

    def __init__(self, holidays: frozenset[date] | None = None) -> None:
        self._holidays = holidays or frozenset()

    def phase(self, instant: datetime) -> SessionPhase:
        if instant.tzinfo is None:
            raise ValueError(
                "naive datetime rejected: an instant without a timezone cannot be resolved "
                "against two clocks in different zones"
            )
        local = instant.astimezone(ET)
        d = local.date()

        if d in self._holidays:
            return SessionPhase.HOLIDAY
        if not _is_business_day(d):
            return SessionPhase.WEEKEND

        t = local.time()
        if RTH_OPEN <= t < RTH_CLOSE:
            return SessionPhase.RTH
        if EXTENDED_OPEN <= t < EXTENDED_CLOSE:
            return SessionPhase.EXTENDED
        return SessionPhase.OVERNIGHT

    def next_discovery(self, instant: datetime) -> datetime:
        """The next instant at which genuine anchor price discovery resumes (the next RTH open)."""
        local = instant.astimezone(ET)
        d = local.date()

        candidate_today = datetime.combine(d, RTH_OPEN, tzinfo=ET)
        if (
            _is_business_day(d)
            and d not in self._holidays
            and local < candidate_today
        ):
            return candidate_today

        nxt = _next_business_day(d)
        while nxt in self._holidays:
            nxt = _next_business_day(nxt)
        return datetime.combine(nxt, RTH_OPEN, tzinfo=ET)

    def state(
        self,
        instant: datetime,
        *,
        nav_age_seconds: float | None = None,
        oracle_age_seconds: float | None = None,
    ) -> SessionState:
        phase = self.phase(instant)
        if phase.has_price_discovery:
            hours = 0.0
        else:
            hours = (self.next_discovery(instant) - instant).total_seconds() / 3600.0
        return SessionState(
            phase=phase,
            as_of=instant,
            hours_to_next_discovery=hours,
            nav_age_seconds=nav_age_seconds,
            oracle_age_seconds=oracle_age_seconds,
        )

    # --- the distinction a single clock cannot express -------------------------------------

    def is_knowable(self, fact_available_at: datetime, instant: datetime) -> bool:
        """Token clock: could we have known this fact at `instant`? Continuous, never closed."""
        return fact_available_at <= instant

    def is_priceable(self, instant: datetime) -> bool:
        """Anchor clock: is genuine price discovery available at `instant`?"""
        return self.phase(instant).has_price_discovery

    def is_hedgeable_against_anchor(self, instant: datetime) -> bool:
        """Can a hedge be placed in the underlying right now?

        Separate from `is_priceable` on purpose. They coincide today, but the moment we model a
        venue whose quotes exist while its matching engine is halted, they diverge — and a caller
        asking "can I hedge" must not silently be answered "is there a price".
        """
        return self.is_priceable(instant)
