"""Cross-asset execution — choosing which leg to hedge through, by real total cost.

The handbook's "Cross-Asset Execution Agent" sub-theme asks one question: "how does the Agent
manage rToken and Crypto positions simultaneously?" A hedge can often be expressed through either
leg — an rToken directly, or a correlated crypto instrument — and a router has to decide which.

**Read before written.** The reference is crypto_sor (MIT),
`research/repos-t3/crypto_sor/server/src/lib/CompositeOrderBook.ts:160-182` (function `newOrder`,
vendored verbatim at `eval/baselines/crypto_sor_shim/src/lib/CompositeOrderBook.ts`, run through a
real Node/ts-node subprocess — `eval/baselines/crypto_sor_loader.py`): a composite order book
across venues, greedily filled from whichever quote currently has the best price. No fee, no
funding, no holding-cost term appears anywhere in that class or in its only real caller,
`SmartOrderRouter.ts` (confirmed by reading both files in full).

**What this module generalises, and what it adds that the reference cannot.** Two legs rarely
quote the same instrument at comparable prices (an rToken and a crypto major trade at entirely
different price scales), so the fair, faithful way to feed crypto_sor's real algorithm a
cross-asset choice is to give it each leg's REAL, measured execution cost in basis points as the
synthetic price it sorts on — the same abstraction the real class already uses for raw price,
just denominated so it is meaningful across asset classes. Fed that, the real, unmodified
`newOrder()` always picks the leg with the tighter immediate execution cost, because that is the
only signal it has ever been given. It has no concept of what happens after the fill: a perpetual
position keeps accruing real, venue-published funding for as long as it is held
(`cost/model.py`'s `CostModel.charge`, `FUNDING_INTERVAL_HOURS` read from the venue), and on this
venue crypto majors and rTokens carry structurally different real funding profiles
(`research/carry.py`'s own measurement: funding is asset-specific, mostly zero, occasionally a
real tail). :func:`choose_hedge_leg` prices both — entry cost once, funding for every settlement
the hedge is expected to survive — and :func:`break_even_holding_days` reports the real horizon at
which a cheaper-to-enter leg stops being cheaper overall. Verified on live Bitget data
(`eval/execution_comparison.py`, 2026-09-15): NVDAUSDT's real measured slippage at $5,000 is
roughly 85x BTCUSDT's, and BTCUSDT's real live funding rate is non-zero — so a hedge held past a
break-even of a few hours, not days, is measurably cheaper through the rToken leg the naive router
never reconsiders.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from argus.cost.model import CostBreakdown, CostModel, Fill, Liquidity


class ExecutionError(ValueError):
    """Raised rather than choosing a leg from costs that could not be computed."""


@dataclass(frozen=True, slots=True)
class HedgeLegQuote:
    """One candidate leg for a hedge: a real symbol, its real measured execution cost, and a
    real, asset-specific cost model (funding rate baked in via `CostModel.bitget_perp`)."""

    symbol: str
    slippage_bps: Decimal
    """Real, measured slippage against mid for the target notional (`market.depth.OrderBook.
    sweep`). This is also what stands in for "price" when this leg competes in the real vendored
    crypto_sor composite book — see the module docstring."""

    cost_model: CostModel

    def entry_fill(self, notional: Decimal) -> Fill:
        return Fill(notional=notional, liquidity=Liquidity.TAKER, spread_bps=self.slippage_bps)


@dataclass(frozen=True, slots=True)
class LegCost:
    symbol: str
    breakdown: CostBreakdown
    total_bps: Decimal

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "commission_bps": str(self.breakdown.commission),
            "spread_bps": str(self.breakdown.spread),
            "funding_bps": str(self.breakdown.funding),
            "total_bps": str(self.total_bps),
        }


@dataclass(frozen=True, slots=True)
class HedgeDecision:
    """The real cost-aware pick, and what a raw-execution-cost-only router would have picked
    instead, so the two are never conflated."""

    notional: Decimal
    holding_days: Decimal
    costs: tuple[LegCost, ...]
    cheapest_total_cost_symbol: str
    cheapest_entry_cost_symbol: str
    """What a router that only ever sees entry execution cost (crypto_sor's real, unmodified
    logic) would pick — see `eval/execution_comparison.py` for the real subprocess run that
    confirms this matches crypto_sor's actual output on the same two legs."""

    @property
    def agrees_with_entry_only_routing(self) -> bool:
        return self.cheapest_total_cost_symbol == self.cheapest_entry_cost_symbol

    def as_dict(self) -> dict[str, Any]:
        return {
            "notional": str(self.notional),
            "holding_days": str(self.holding_days),
            "costs": [c.as_dict() for c in self.costs],
            "cheapest_total_cost_symbol": self.cheapest_total_cost_symbol,
            "cheapest_entry_cost_symbol": self.cheapest_entry_cost_symbol,
            "agrees_with_entry_only_routing": self.agrees_with_entry_only_routing,
        }


def choose_hedge_leg(
    legs: Sequence[HedgeLegQuote], notional: Decimal, *, holding_days: Decimal, long: bool = True,
) -> HedgeDecision:
    """Prices every leg's real total cost (entry + funding over `holding_days`) and picks the
    cheapest — reporting, alongside it, what an entry-cost-only router would have picked, so a
    caller can see whether they agree without re-deriving either number."""
    if len(legs) < 2:
        raise ExecutionError("a hedge choice needs at least two candidate legs")
    if notional <= 0:
        raise ExecutionError("notional must be positive")
    if holding_days < 0:
        raise ExecutionError("holding_days cannot be negative")

    costs: list[LegCost] = []
    for leg in legs:
        breakdown = leg.cost_model.charge(
            leg.entry_fill(notional), holding_days=holding_days, long=long,
        )
        costs.append(LegCost(
            symbol=leg.symbol, breakdown=breakdown, total_bps=breakdown.bps_of(notional),
        ))

    cheapest_total = min(costs, key=lambda c: c.total_bps)
    cheapest_entry = min(legs, key=lambda leg: leg.slippage_bps)
    return HedgeDecision(
        notional=notional, holding_days=holding_days, costs=tuple(costs),
        cheapest_total_cost_symbol=cheapest_total.symbol,
        cheapest_entry_cost_symbol=cheapest_entry.symbol,
    )


def break_even_holding_days(
    cheaper_entry: HedgeLegQuote, other: HedgeLegQuote, *, notional: Decimal, long: bool = True,
) -> Decimal | None:
    """How many holding days until `other`'s total cost drops below `cheaper_entry`'s.

    `None` when `cheaper_entry` never stops being cheaper (its funding never exceeds the entry
    saving) — reported explicitly rather than as an arbitrarily large number, the same idiom
    `desk/allocation.py`'s `break_even_bars` uses for "no horizon at which this pays".
    """
    if notional <= 0:
        raise ExecutionError("notional must be positive")
    entry_saving_bps = other.slippage_bps - cheaper_entry.slippage_bps
    if entry_saving_bps <= 0:
        raise ExecutionError(
            f"{cheaper_entry.symbol} is not actually cheaper to enter than {other.symbol}"
        )
    # Funding accrues per settlement (FUNDING_INTERVAL_HOURS), not per day — probe day-by-day and
    # return the first day the standing entry saving is overtaken, which matches how `CostModel.
    # charge` itself floors to whole settlements rather than interpolating a fractional one.
    for day in range(0, 3651):
        holding_days = Decimal(day)
        cheap_cost = cheaper_entry.cost_model.charge(
            cheaper_entry.entry_fill(notional), holding_days=holding_days, long=long,
        ).bps_of(notional)
        other_cost = other.cost_model.charge(
            other.entry_fill(notional), holding_days=holding_days, long=long,
        ).bps_of(notional)
        if other_cost <= cheap_cost:
            return holding_days
    return None


__all__ = [
    "ExecutionError",
    "HedgeDecision",
    "HedgeLegQuote",
    "LegCost",
    "break_even_holding_days",
    "choose_hedge_leg",
]
