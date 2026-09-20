"""The constructed-mandatory cost model.

Across 103 defects catalogued from 25 code-level teardowns, **cost-blindness is the single most
widespread failure class in the field** — ahead of look-ahead leakage, ahead of self-scoring. It is
also the one that most reliably converts a losing strategy into a winning backtest. Two production
engines fail it by default:

  * NautilusTrader backtests at **zero fees** unless a commission model is configured.
  * Qlib measures turnover as **gross notional rather than net delta**, overstating cost 2-3x on
    mean-reversion, and its defaults are A-share shaped (open 0.15%, close 0.25%,
    ``exchange.py:48-51``) — wrong for Bitget by a wide margin.
  * ``alphaagent`` computes both ``with_cost`` and ``without_cost`` metrics and its keep/reject
    logic reads only ``without_cost``; RD-Agent shows ``without_cost`` figures to both the proposer
    and the evaluator while feedback reads ``with_cost``.

The design response is structural rather than procedural: **there is no zero-fee constructor.**
``CostModel()`` cannot be instantiated without explicit rates, ``CostModel.zero()`` does not exist,
and a research-only frictionless model must be built through :func:`frictionless_for_research`,
which stamps the result so that any attempt to gate a decision on it raises.

The measured Bitget economics that motivate the defaults: **0.12% round-trip taker** versus roughly
0.00% measured intraday edge and ~0.10% overnight drift. The fee is larger than every effect we are
trying to capture, so a model that omits it is not approximate — it is inverted.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import ROUND_FLOOR, Decimal
from enum import StrEnum


class Liquidity(StrEnum):
    TAKER = "taker"
    MAKER = "maker"


class FrictionlessCostError(RuntimeError):
    """Raised when a frictionless cost model is used to justify a decision.

    A frictionless model is legitimate for isolating an effect in research. It is never legitimate
    as the basis for a keep/reject, a promotion, or an order. That distinction is enforced here
    rather than left to reviewer vigilance, because the field's record shows reviewer vigilance
    does not hold.
    """


@dataclass(frozen=True, slots=True)
class Fill:
    """One executed slice, in the units cost is actually charged in."""

    notional: Decimal
    liquidity: Liquidity
    spread_bps: Decimal = Decimal("0")
    adv_participation: Decimal = Decimal("0")
    """Fraction of average daily volume this slice represents. Drives the impact term."""


FUNDING_INTERVAL_HOURS = Decimal("8")
"""How often a Bitget perpetual settles funding. Read from the venue, not assumed.

`GET /api/v2/mix/market/current-fund-rate` returns ``fundingRateInterval: "8"`` for every rToken
checked on 2026-09-14.
"""

MAX_FUNDING_BPS = Decimal("500")
"""The venue's own per-settlement cap: ``minFundingRate: -0.005``, ``maxFundingRate: 0.005``.

A rate outside this is not a large cost, it is a unit error — the API returns a fraction (0.000341)
and this field wants basis points (3.41). Catching that at construction is the difference between a
3.41bps charge and a 34,100bps one.
"""


@dataclass(frozen=True, slots=True)
class CostBreakdown:
    """Every component, kept separate. A single total hides which assumption is load-bearing."""

    commission: Decimal
    spread: Decimal
    impact: Decimal
    borrow: Decimal
    funding: Decimal = Decimal("0")
    """Perpetual funding paid (positive) or received (negative) while the position was held.

    A separate channel from ``borrow`` on purpose. Borrow is an annualised rate on a short; funding
    settles on a fixed clock every ``FUNDING_INTERVAL_HOURS`` regardless of side, and its sign flips
    with the market rather than with your direction. Adding them would make a long paying funding
    indistinguishable from a short paying borrow, and only one of those is a function of the side
    you chose."""

    @property
    def total(self) -> Decimal:
        """Every channel, funding included.

        Funding was added to this dataclass and left out of this sum on the first pass: the charge
        was computed correctly and then discarded, so a position paying it looked identical to one
        that did not. A cost channel that exists but is not summed is worse than one that does not
        exist, because the number looks complete.
        """
        return self.commission + self.spread + self.impact + self.borrow + self.funding

    def bps_of(self, notional: Decimal) -> Decimal:
        if notional <= 0:
            raise ValueError("notional must be positive to express cost in bps")
        return self.total / notional * Decimal("10000")


@dataclass(frozen=True, slots=True)
class CostModel:
    """Transaction costs with no frictionless default anywhere in its construction.

    Market impact follows the square-root law with ``gamma`` defaulting to 1.5 — cvxportfolio's
    default at ``costs.py:826`` with convexity enforced at >= 1.0 (``costs.py:845-848``). That is a
    published, defensible exponent rather than an invented one. cvxportfolio itself is GPL, so this
    is a reimplementation of the published functional form, not vendored code.
    """

    taker_bps: Decimal
    maker_bps: Decimal
    impact_coefficient: Decimal = Decimal("10")
    gamma: Decimal = Decimal("1.5")
    borrow_bps_annual: Decimal = Decimal("0")

    funding_bps_per_interval: Decimal = Decimal("0")
    """Funding charged per settlement, in bps of notional. Pass the venue's live rate.

    Zero is the honest default and **not** a frictionless one: Bitget's rTokens settle at exactly
    zero most of the time. Measured from the venue's own history on 2026-09-14, 100 settlements per
    symbol: NVDAUSDT non-zero on 14, TSLAUSDT on 23, COINUSDT on 30, with means of 0.23, 0.36 and
    0.58 bps and a worst single settlement of 5.8bps. Over a 24-hour hold — three settlements — that
    is roughly 0.7 to 1.7bps typically and about 17bps at the tail, against an 18.8bps total hurdle.

    Small on average, and the tail is the whole hurdle. It was missing entirely until a competitive
    sweep flagged it and the venue's history was pulled to check: `Ticker.funding_rate` was already
    being fetched (`market/bitget.py:82`) and then thrown away before it reached any cost.
    """

    _frictionless: bool = False
    """Set only by :func:`frictionless_for_research`. Poisons any gating use of this model."""

    def __post_init__(self) -> None:
        if self._frictionless:
            return  # the research escape hatch validates differently; see the factory below
        if self.taker_bps <= 0:
            raise ValueError(
                "taker_bps must be positive. A zero-fee model is the most common defect in the "
                "field; if you genuinely need one for research, use frictionless_for_research()."
            )
        if self.maker_bps < 0:
            raise ValueError("maker_bps may be zero (rebate-neutral) but never negative here")
        if self.gamma < 1:
            raise ValueError("gamma must be >= 1 for the impact term to stay convex")
        if self.impact_coefficient < 0:
            raise ValueError("impact_coefficient must be non-negative")
        # Funding may be negative — a short is paid when the rate is positive, and pinning it to
        # non-negative would silently delete the only cost channel in this model that can be a
        # credit.
        if abs(self.funding_bps_per_interval) > MAX_FUNDING_BPS:
            raise ValueError(
                f"funding of {self.funding_bps_per_interval}bps per interval exceeds the venue cap "
                f"of {MAX_FUNDING_BPS}bps; check the units — the API returns a fraction, not bps"
            )

    @classmethod
    def bitget_perp(cls, *, funding_rate: Decimal | None = None) -> CostModel:
        """The measured Bitget round-trip: 0.06% per side taker, 0.12% round trip.

        ``funding_rate`` is the venue's rate **as a fraction** — exactly what
        ``/api/v2/mix/market/current-fund-rate`` returns and what `Ticker.funding_rate` already
        carries — and is converted to bps here. Passing it is how a caller stops paying an
        unmodelled cost; omitting it keeps the historical behaviour of charging none, which is
        correct for the majority of settlements on these instruments and wrong in the tail.
        """
        return cls(
            taker_bps=Decimal("6"), maker_bps=Decimal("2"),
            funding_bps_per_interval=(
                Decimal("0") if funding_rate is None else funding_rate * Decimal("10000")
            ),
        )

    def charge(
        self, fill: Fill, *, holding_days: Decimal = Decimal("0"), long: bool = True,
    ) -> CostBreakdown:
        """Every cost channel for one fill and the position it leaves behind.

        ``long`` sets the sign of funding only. It is a parameter here rather than a field on
        :class:`Fill` because a fill is one execution while funding is a property of the position
        that survives it — and because the sign flips a *cost* into a *credit*, which is not
        something a caller should be able to set by accident.
        """
        rate = self.taker_bps if fill.liquidity is Liquidity.TAKER else self.maker_bps
        commission = fill.notional * rate / Decimal("10000")

        # A taker crosses the full spread; a maker is paid it, but only when the fill was not
        # adversely selected. We deliberately do not credit the maker side: our own replay work
        # showed a +90% local sim become -0.45% once resting limit orders were modelled honestly,
        # because they fill when the market moves against you and miss when it does not.
        spread = (
            fill.notional * fill.spread_bps / Decimal("10000")
            if fill.liquidity is Liquidity.TAKER
            else Decimal("0")
        )

        impact = Decimal("0")
        if fill.adv_participation > 0:
            impact = (
                fill.notional
                * self.impact_coefficient
                * (fill.adv_participation ** self.gamma)
                / Decimal("10000")
            )

        borrow = (
            fill.notional * self.borrow_bps_annual / Decimal("10000")
            * holding_days / Decimal("365")
            if holding_days > 0
            else Decimal("0")
        )

        # Funding accrues per settlement, not per day: a position opened and closed inside one
        # interval pays nothing, and one held 24 hours crosses three. Floor division is deliberate —
        # charging a fraction of a settlement would invent a cost the venue never takes.
        settlements = (
            (holding_days * Decimal("24") / FUNDING_INTERVAL_HOURS).to_integral_value(
                rounding=ROUND_FLOOR
            )
            if holding_days > 0
            else Decimal("0")
        )
        funding = (
            fill.notional * self.funding_bps_per_interval / Decimal("10000") * settlements
            * (Decimal("1") if long else Decimal("-1"))
        )

        return CostBreakdown(
            commission=commission, spread=spread, impact=impact, borrow=borrow, funding=funding,
        )

    def round_trip_bps(self) -> Decimal:
        """Commission-only round trip, the number every edge must clear before anything else."""
        return self.taker_bps * 2

    def assert_gateable(self) -> None:
        """Call before any keep/reject, promotion, sizing or order decision.

        This is the guard that ``alphaagent`` and RD-Agent lack: both compute honest cost figures
        and then gate on the frictionless ones.
        """
        if self._frictionless:
            raise FrictionlessCostError(
                "a frictionless cost model cannot justify a decision. It exists to isolate an "
                "effect in research; gating on it reproduces the field's most common defect."
            )


def frictionless_for_research(reason: str) -> CostModel:
    """A zero-cost model, explicitly marked, for isolating an effect.

    The only way to obtain one. It carries a poison flag so that
    :meth:`CostModel.assert_gateable` refuses it, which means a frictionless number can be
    computed and reported but can never silently become the basis for a decision.
    """
    if not reason.strip():
        raise ValueError("a frictionless model requires a written reason")
    base = CostModel(taker_bps=Decimal("1"), maker_bps=Decimal("0"))
    return replace(
        base,
        taker_bps=Decimal("0"),
        maker_bps=Decimal("0"),
        impact_coefficient=Decimal("0"),
        borrow_bps_annual=Decimal("0"),
        _frictionless=True,
    )


def net_edge_bps(gross_edge_bps: Decimal, model: CostModel, *, round_trips: int = 1) -> Decimal:
    """Gross edge less the commission it must clear. Frequently negative — that is the point.

    Our own measurements: ~0.00% intraday edge and ~0.10% overnight drift against a 0.12%
    round-trip taker fee. Any strategy round-tripping daily loses before it begins, which is why
    this function exists as a first-class check rather than a footnote in a report.
    """
    model.assert_gateable()
    return gross_edge_bps - model.round_trip_bps() * Decimal(round_trips)
