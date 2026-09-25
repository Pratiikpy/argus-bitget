"""Net executable arbitrage across two real order books: the exact optimum, not a flat estimate.

**The question.** Two books quote the same underlying. On Bitget that is a spot rToken
(``RNVDAUSDT``, base coin ``rNVDA``) and the same company's USDT-margined perpetual
(``NVDAUSDT``); across venues it is the same asset on two exchanges. Each venue charges its own
taker fee. How much money can be taken *right now* by buying on one book and selling on the
other, at what size, and how much of it survives ARGUS's execution risk?

**Why this module exists.** `research/arbitrage_study.decompose()` answers a different question.
It takes one apparent spread, measured mid to mid, and subtracts flat constants: a 12bps
round-trip fee, 0.6bps of spread crossing and 2bps of slippage. Those constants were measured on
the stock perpetuals, and the decomposition has no way to see a book. It was run against a
general-purpose LP solver (HiGHS, through SciPy) on real, simultaneous Bitget books for every
spot rToken and its perpetual (`eval/general_arb_comparison.py`, `data/
general_arb_comparison.json`). On the 2026-09-25 regular-hours capture there were 240 two-sided
snapshots across 80 tickers. The optimum found 3 monetizable. ``decompose()`` accepted 52, and
49 of those lose money at the touch, a precision of 5.8%. A held-out capture of 160 snapshots,
taken after the code was frozen, gave 25 accepted and 24 wrong. The loss runs both ways:

* The spot rToken's quoted spread had a median of 9.2bps, not 0.6bps, and its taker fee is 10bps,
  not 6bps. A wide mid-to-mid gap that the touch cannot deliver is therefore called monetizable.
  Replacing the flat 0.6bps with the two books' measured half-spreads removes 48 of the 49 false
  accepts on the development capture and all 24 on the held-out capture. Replacing the fee
  alone removes 35 of 49.
* On the capability's own constructed two-exchange books, the apparent spread is already the gap
  between executable prices. Charging 0.6bps of spread crossing and 2bps of slippage on top
  refuses every spread from 12.1bps to 14.6bps, even though the touch clears the fee. One of the
  60 swept spreads in `eval/arbitrage_comparison.py` (13.47bps) falls in that band.

This module is ARGUS's answer to that loss. It ties HiGHS on every snapshot in both captures,
to within 4e-14 USDT (2.3e-14 on the development capture, 3.4e-14 held out). In the committed
artefact's timing of 40 snapshots it takes 18 microseconds per snapshot against HiGHS's 7
milliseconds, both directions, roughly 390 times faster; timings are machine-dependent and the
artefact's ``costs`` block is the figure to cite, not this sentence.

**The formulation, taken from the general-purpose view rather than from a trading repository.**
For one direction (buy on book A, sell on book B), with ask levels ``(a_i, q_i)`` ascending and bid
levels ``(b_j, r_j)`` descending::

    maximise   sum_j y_j * b_j * (1 - f_sell)  -  sum_i x_i * a_i * (1 + f_buy)
    subject to sum_i x_i = sum_j y_j,   0 <= x_i <= q_i,   0 <= y_j <= r_j
               (optionally) sum_i x_i * a_i * (1 + f_buy) <= max_notional, sum_i x_i <= max_qty

That is a linear programme, and `eval/general_arb_comparison.py` solves it with HiGHS on every
snapshot. It does not need a solver here: the marginal profit of the next unit,
``b_j * (1 - f_sell) - a_i * (1 + f_buy)``, can only fall as the walk moves to a higher ask or a
lower bid, and the budget each unit uses can only rise. So the optimum is the prefix of the walk
on which that marginal profit is still positive, cut at the first cap it meets. The comparison
checks that this walk and HiGHS agree on every real snapshot and every constructed one; a
disagreement fails the test suite.

**What was taken from where.**

* The per-leg fee, charged separately on each venue's own notional at that venue's own taker
  rate, follows Hummingbot's ``ArbProposal.profit_pct(account_for_fee=True)``, which builds one
  ``TradeFee`` per side (`hummingbot/strategy/amm_arb/data_types.py:76-121`, Apache-2.0, read at
  commit 2bfaccc). Only the behaviour is used. No code is copied.
* Rejected from Hummingbot: sizing at one fixed, user-chosen ``order_amount``
  (`amm_arb/utils.py:18-58`). The size is an output here, not an input. Also rejected:
  ``spot_perpetual_arbitrage``'s fee-blind ``profit_pct``, which stands a user-set
  ``min_opening_arbitrage_pct`` in for fees (`spot_perpetual_arbitrage/arb_proposal.py:56-64`,
  `spot_perpetual_arbitrage.py:204-207`).
* Execution probability and failed-leg survival are ARGUS's own, from `arbitrage_study.decompose`.
  They scale a positive net only. A losing trade that fails to execute is a saving, not a loss.

**What it does not model.** Queue position, latency between the two legs, book changes while an
order is in flight, and funding on the perpetual leg while the position stays open. A snapshot is
one instant (`market/depth.py`). The survival terms are the only allowance for the gap between a
visible book and a filled trade, and they are assumptions, not measurements.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal

from argus.market.depth import Level, OrderBook

_ONE = Decimal("1")
_BPS = Decimal("10000")

DEFAULT_EXECUTION_PROBABILITY = Decimal("0.92")
"""Both legs fill at the quoted prices. The same figure `arbitrage_study.decompose` uses."""

DEFAULT_FAILED_LEG_PROBABILITY = Decimal("0.05")
"""One leg fills and the other does not, which leaves naked exposure. Same source as above."""


class ExecutableArbError(ValueError):
    """Raised for inputs that cannot describe a real trade: a negative fee, a negative cap."""


@dataclass(frozen=True, slots=True)
class LegOptimum:
    """The best trade in one direction, buying on one book and selling on the other."""

    buy_venue: str
    sell_venue: str
    quantity: Decimal
    """Base units bought on ``buy_venue`` and sold on ``sell_venue``."""

    buy_notional: Decimal
    """Quote paid for the bought quantity, before the buy fee."""

    sell_notional: Decimal
    """Quote received for the sold quantity, before the sell fee."""

    fees: Decimal
    """Both legs' taker fees, in quote."""

    top_of_book_edge_bps: Decimal | None
    """Net edge of the first unit at the touch, in bps of the best ask. Negative when nothing
    clears the fees. ``None`` when either side is empty, because an empty side has no touch."""

    buy_levels: int
    sell_levels: int
    capped: bool
    """True when a quantity or notional cap stopped the walk before the edge ran out."""

    @property
    def net(self) -> Decimal:
        """Quote left after both fills and both fees. Never negative: the optimum can be to do
        nothing."""
        return self.sell_notional - self.buy_notional - self.fees

    @property
    def net_bps(self) -> Decimal:
        """Net as bps of the bought notional. Zero when no quantity was taken."""
        if self.buy_notional <= 0:
            return Decimal("0")
        return self.net / self.buy_notional * _BPS

    def as_dict(self) -> dict[str, object]:
        edge = self.top_of_book_edge_bps
        return {
            "buy_venue": self.buy_venue,
            "sell_venue": self.sell_venue,
            "quantity": str(self.quantity),
            "buy_notional": str(self.buy_notional),
            "sell_notional": str(self.sell_notional),
            "fees": str(self.fees),
            "net": str(self.net),
            "net_bps": str(round(self.net_bps, 4)),
            "top_of_book_edge_bps": None if edge is None else str(round(edge, 4)),
            "buy_levels": self.buy_levels,
            "sell_levels": self.sell_levels,
            "capped": self.capped,
        }


@dataclass(frozen=True, slots=True)
class ExecutableArb:
    """The better of the two directions, with ARGUS's execution risk applied on top."""

    best: LegOptimum
    other: LegOptimum
    execution_probability: Decimal
    failed_leg_probability: Decimal

    @property
    def monetizable(self) -> bool:
        return self.best.net > 0

    @property
    def expected_net(self) -> Decimal:
        """Net in expectation once both legs have to fill. Scales a positive net only."""
        if self.best.net <= 0:
            return self.best.net
        return self.best.net * self.execution_probability * (_ONE - self.failed_leg_probability)

    def as_dict(self) -> dict[str, object]:
        return {
            "monetizable": self.monetizable,
            "expected_net": str(self.expected_net),
            "execution_probability": str(self.execution_probability),
            "failed_leg_probability": str(self.failed_leg_probability),
            "best": self.best.as_dict(),
            "other": self.other.as_dict(),
        }


def _check_fee(name: str, fee: Decimal) -> None:
    if not Decimal("0") <= fee < _ONE:
        raise ExecutableArbError(f"{name} must be a fraction in [0, 1); got {fee}")


def leg_optimum(
    buy_asks: Sequence[Level],
    sell_bids: Sequence[Level],
    *,
    buy_fee: Decimal,
    sell_fee: Decimal,
    buy_venue: str = "a",
    sell_venue: str = "b",
    max_quantity: Decimal | None = None,
    max_notional: Decimal | None = None,
) -> LegOptimum:
    """The exact optimum of the linear programme in the module docstring, for one direction.

    ``buy_asks`` must be ascending by price and ``sell_bids`` descending, which is how
    :class:`~argus.market.depth.OrderBook` stores them. ``buy_fee`` and ``sell_fee`` are
    fractions (``Decimal("0.001")`` is 10bps). ``max_notional`` caps the quote spent on the buy leg,
    fee included.
    """
    _check_fee("buy_fee", buy_fee)
    _check_fee("sell_fee", sell_fee)
    for name, cap in (("max_quantity", max_quantity), ("max_notional", max_notional)):
        if cap is not None and cap < 0:
            raise ExecutableArbError(f"{name} must be non-negative; got {cap}")

    buy_mult, sell_mult = _ONE + buy_fee, _ONE - sell_fee
    edge: Decimal | None = None
    if buy_asks and sell_bids:
        best_ask, best_bid = buy_asks[0].price, sell_bids[0].price
        edge = (best_bid * sell_mult - best_ask * buy_mult) / best_ask * _BPS

    i = j = 0
    ask_left = buy_asks[0].quantity if buy_asks else Decimal("0")
    bid_left = sell_bids[0].quantity if sell_bids else Decimal("0")
    quantity = buy_notional = sell_notional = Decimal("0")
    buy_used: set[int] = set()
    sell_used: set[int] = set()
    capped = False

    while i < len(buy_asks) and j < len(sell_bids):
        ask, bid = buy_asks[i].price, sell_bids[j].price
        if bid * sell_mult - ask * buy_mult <= 0:
            break
        take = min(ask_left, bid_left)
        if max_quantity is not None and quantity + take >= max_quantity:
            take, capped = max_quantity - quantity, True
        if max_notional is not None:
            room = (max_notional - buy_notional * buy_mult) / (ask * buy_mult)
            if take >= room:
                take, capped = room, True
        if take > 0:
            quantity += take
            buy_notional += take * ask
            sell_notional += take * bid
            buy_used.add(i)
            sell_used.add(j)
            ask_left -= take
            bid_left -= take
        if capped:
            break
        if ask_left <= 0:
            i += 1
            ask_left = buy_asks[i].quantity if i < len(buy_asks) else Decimal("0")
        if bid_left <= 0:
            j += 1
            bid_left = sell_bids[j].quantity if j < len(sell_bids) else Decimal("0")

    fees = buy_notional * buy_fee + sell_notional * sell_fee
    return LegOptimum(
        buy_venue=buy_venue, sell_venue=sell_venue, quantity=quantity,
        buy_notional=buy_notional, sell_notional=sell_notional, fees=fees,
        top_of_book_edge_bps=edge, buy_levels=len(buy_used), sell_levels=len(sell_used),
        capped=capped,
    )


def best_arbitrage(
    book_a: OrderBook,
    book_b: OrderBook,
    *,
    fee_a: Decimal,
    fee_b: Decimal,
    max_quantity: Decimal | None = None,
    max_notional: Decimal | None = None,
    execution_probability: Decimal = DEFAULT_EXECUTION_PROBABILITY,
    failed_leg_probability: Decimal = DEFAULT_FAILED_LEG_PROBABILITY,
) -> ExecutableArb:
    """Both directions between two books; the one that leaves more money is ``best``.

    ``fee_a`` and ``fee_b`` are each venue's own taker rate as a fraction, read from the venue
    (`/api/v2/spot/public/symbols` and `/api/v2/mix/market/contracts` both publish
    ``takerFeeRate``) rather than assumed. On 2026-09-25 they were 0.001 for ``RNVDAUSDT`` and
    0.0006 for ``NVDAUSDT``.
    """
    for name, p in (("execution_probability", execution_probability),
                    ("failed_leg_probability", failed_leg_probability)):
        if not Decimal("0") <= p <= _ONE:
            raise ExecutableArbError(f"{name} must lie in [0, 1]; got {p}")
    a_to_b = leg_optimum(
        book_a.asks, book_b.bids, buy_fee=fee_a, sell_fee=fee_b,
        buy_venue=book_a.symbol, sell_venue=book_b.symbol,
        max_quantity=max_quantity, max_notional=max_notional,
    )
    b_to_a = leg_optimum(
        book_b.asks, book_a.bids, buy_fee=fee_b, sell_fee=fee_a,
        buy_venue=book_b.symbol, sell_venue=book_a.symbol,
        max_quantity=max_quantity, max_notional=max_notional,
    )

    def rank(leg: LegOptimum) -> tuple[Decimal, Decimal]:
        edge = leg.top_of_book_edge_bps
        return leg.net, Decimal("-Infinity") if edge is None else edge

    best, other = (a_to_b, b_to_a) if rank(a_to_b) >= rank(b_to_a) else (b_to_a, a_to_b)
    return ExecutableArb(
        best=best, other=other, execution_probability=execution_probability,
        failed_leg_probability=failed_leg_probability,
    )


__all__ = [
    "DEFAULT_EXECUTION_PROBABILITY",
    "DEFAULT_FAILED_LEG_PROBABILITY",
    "ExecutableArb",
    "ExecutableArbError",
    "LegOptimum",
    "best_arbitrage",
    "leg_optimum",
]
