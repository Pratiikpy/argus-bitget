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
from decimal import Decimal
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


@dataclass(frozen=True, slots=True)
class CostBreakdown:
    """Every component, kept separate. A single total hides which assumption is load-bearing."""

    commission: Decimal
    spread: Decimal
    impact: Decimal
    borrow: Decimal

    @property
    def total(self) -> Decimal:
        return self.commission + self.spread + self.impact + self.borrow

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

    @classmethod
    def bitget_perp(cls) -> CostModel:
        """The measured Bitget round-trip: 0.06% per side taker, 0.12% round trip."""
        return cls(taker_bps=Decimal("6"), maker_bps=Decimal("2"))

    def charge(self, fill: Fill, *, holding_days: Decimal = Decimal("0")) -> CostBreakdown:
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

        return CostBreakdown(commission=commission, spread=spread, impact=impact, borrow=borrow)

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
