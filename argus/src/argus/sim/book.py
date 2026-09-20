"""A price-time priority matching engine.

Rebuilt from ``jpmorganchase/abides-jpmc-public`` (**BSD 3-Clause**, © 2021 J.P. Morgan Chase;
original authors David Byrd and Dr. Tucker Balch, Georgia Institute of Technology). The matching
semantics follow ``order_book.py:75-150`` and are reproduced deliberately:

* **Price-time priority**, consuming all available shares at the best price before moving on,
  *without regard to order-size "fit"* and without trying to minimise the number of transactions.
* **Partial fills walk the book.** An inbound 100 can fill 30 at the touch, 50 one level worse and
  20 the level after that. Each match is its own execution, at its own price.
* **An unmatched remainder rests.** It does not evaporate and it does not fill at a worse price
  than its limit.

**What ARGUS adds, and it is the whole reason this module exists rather than a dependency.**
ABIDES models no fees at all — grep-proved absence across ``order_book.py`` and
``exchange_agent.py`` for "fee", and cash is adjusted by quantity x price only. Our own teardowns
put cost-blindness ahead of leakage as the most widespread defect in the corpus, so a simulator we
adopted wholesale would import the exact defect the rest of this codebase makes unconstructible.
Here every execution carries the liquidity flag that decides its fee, and the caller settles it
through :class:`~argus.cost.model.CostModel`.

Three further departures, each stated so ours and theirs are never ambiguous:

* **Tick size is enforced.** ABIDES leaves it to the agents (``7.7``); a book that accepts
  sub-tick prices silently produces spreads no venue would quote.
* **Self-trading is refused** rather than matched. An agent crossing its own resting order is a
  wash trade, and letting it match inflates simulated volume — which is the input a
  percent-of-volume market maker sizes against, so the error compounds.
* **Sessions are absent here on purpose.** The two-clock model lives above this layer; a matching
  engine should not know what a holiday is.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum
from itertools import count

_ZERO = Decimal("0")


class BookError(ValueError):
    """The book was given an order it cannot honestly accept."""


class Side(StrEnum):
    BUY = "buy"
    SELL = "sell"

    @property
    def opposite(self) -> Side:
        return Side.SELL if self is Side.BUY else Side.BUY


class Liquidity(StrEnum):
    """Which side of the fee schedule an execution lands on.

    The field ABIDES does not have. A fill is *taker* for the aggressor that crossed and *maker*
    for the order that was resting, which is the only basis on which a fee can be charged
    correctly — and charging both sides the taker rate is how a simulated maker strategy quietly
    becomes unprofitable in a way the operator cannot see.
    """

    MAKER = "maker"
    TAKER = "taker"


@dataclass(frozen=True, slots=True)
class Execution:
    """One match. Two of these are produced per trade — one per counterparty."""

    price: Decimal
    quantity: Decimal
    side: Side
    """The side the *owner of this execution* traded."""

    liquidity: Liquidity
    agent_id: str
    counterparty_id: str
    order_id: int
    sequence: int

    @property
    def notional(self) -> Decimal:
        return self.price * self.quantity


@dataclass
class Order:
    """A limit order. A market order is expressed as a limit at an unreachable price."""

    agent_id: str
    side: Side
    price: Decimal
    quantity: Decimal
    order_id: int = 0
    filled: Decimal = _ZERO

    @property
    def leaves(self) -> Decimal:
        return self.quantity - self.filled

    @property
    def is_complete(self) -> bool:
        return self.leaves <= _ZERO


@dataclass
class PriceLevel:
    """All orders resting at one price, in arrival order.

    A deque rather than a list because the only two operations that matter are "read the front"
    and "remove the front" — the definition of time priority.
    """

    price: Decimal
    orders: deque[Order] = field(default_factory=deque)

    @property
    def quantity(self) -> Decimal:
        return sum((o.leaves for o in self.orders), _ZERO)

    @property
    def is_empty(self) -> bool:
        return not self.orders


class OrderBook:
    """One symbol's book.

    Bids are held descending and asks ascending, each as a plain sorted list of levels. That is
    O(n) on insert and it is the right choice here: a simulated book runs tens of levels deep, and
    a heap or tree would trade readability for a speed-up nothing in this project can measure.
    """

    def __init__(self, symbol: str, *, tick_size: Decimal = Decimal("0.01")) -> None:
        if tick_size <= 0:
            raise BookError(f"tick_size={tick_size} must be positive")
        self.symbol = symbol
        self.tick_size = tick_size
        self._bids: list[PriceLevel] = []   # descending
        self._asks: list[PriceLevel] = []   # ascending
        self._ids = count(1)
        self._sequence = count(1)
        self.last_trade: Decimal | None = None
        self.traded_volume: Decimal = _ZERO

    # --- inspection ---------------------------------------------------------------------------

    @property
    def best_bid(self) -> Decimal | None:
        return self._bids[0].price if self._bids else None

    @property
    def best_ask(self) -> Decimal | None:
        return self._asks[0].price if self._asks else None

    @property
    def spread(self) -> Decimal | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return self.best_ask - self.best_bid

    @property
    def mid(self) -> Decimal | None:
        """Midpoint, or ``None`` when a side is empty.

        ``None`` rather than falling back to the last trade. A one-sided book has no midpoint, and
        inventing one is how a simulated market maker quotes around a price that does not exist.
        """
        if self.best_bid is None or self.best_ask is None:
            return None
        return (self.best_bid + self.best_ask) / 2

    def depth_at(self, side: Side, price: Decimal) -> Decimal:
        for level in self._side(side):
            if level.price == price:
                return level.quantity
        return _ZERO

    def levels(self, side: Side, depth: int = 5) -> list[tuple[Decimal, Decimal]]:
        """Top ``depth`` levels as (price, quantity) — the L2 snapshot agents subscribe to."""
        return [(lv.price, lv.quantity) for lv in self._side(side)[:depth]]

    def total_depth(self, side: Side, *, levels: int = 5) -> Decimal:
        return sum((q for _, q in self.levels(side, levels)), _ZERO)

    # --- the matching engine ------------------------------------------------------------------

    def submit(self, order: Order) -> list[Execution]:
        """Match what can be matched, rest the remainder. ``order_book.py:109-134``.

        Returns every execution the match produced, for **both** counterparties, so the caller can
        settle fees on each side. ABIDES notifies each agent separately over its message bus; here
        the executions are returned together because there is no bus, and losing the resting side
        would mean the maker never learns it was filled.
        """
        if order.quantity <= 0:
            raise BookError(f"quantity={order.quantity} must be positive")
        if order.price <= 0:
            raise BookError(f"price={order.price} must be positive")
        remainder = order.price % self.tick_size
        if remainder != 0:
            raise BookError(
                f"price={order.price} is not a multiple of tick_size={self.tick_size}. ABIDES "
                f"leaves tick size to the agents; we enforce it, because a sub-tick book quotes "
                f"spreads no venue would honour."
            )

        order.order_id = next(self._ids)
        executions: list[Execution] = []
        opposing = self._side(order.side.opposite)

        while not order.is_complete and opposing and self._crosses(order, opposing[0].price):
            level = opposing[0]
            while not order.is_complete and level.orders:
                resting = level.orders[0]

                if resting.agent_id == order.agent_id:
                    # A wash trade. ABIDES would match it; matching it here would inflate traded
                    # volume, which is exactly the input the PoV market maker sizes against.
                    raise BookError(
                        f"agent {order.agent_id!r} would cross its own resting order "
                        f"{resting.order_id} at {level.price}: a wash trade, refused"
                    )

                traded = min(order.leaves, resting.leaves)
                order.filled += traded
                resting.filled += traded
                seq = next(self._sequence)

                # The resting order set the price. The aggressor pays it — that is what makes one
                # side maker and the other taker.
                executions.append(Execution(
                    price=level.price, quantity=traded, side=order.side,
                    liquidity=Liquidity.TAKER, agent_id=order.agent_id,
                    counterparty_id=resting.agent_id, order_id=order.order_id, sequence=seq,
                ))
                executions.append(Execution(
                    price=level.price, quantity=traded, side=resting.side,
                    liquidity=Liquidity.MAKER, agent_id=resting.agent_id,
                    counterparty_id=order.agent_id, order_id=resting.order_id, sequence=seq,
                ))

                self.last_trade = level.price
                self.traded_volume += traded

                if resting.is_complete:
                    level.orders.popleft()

            if level.is_empty:
                opposing.pop(0)

        if not order.is_complete:
            self._rest(order)
        return executions

    def cancel(self, order_id: int) -> bool:
        """Pull a resting order. Returns whether it was there to pull."""
        for side in (Side.BUY, Side.SELL):
            levels = self._side(side)
            for i, level in enumerate(levels):
                for j, resting in enumerate(level.orders):
                    if resting.order_id == order_id:
                        del level.orders[j]
                        if level.is_empty:
                            levels.pop(i)
                        return True
        return False

    def cancel_all(self, agent_id: str) -> int:
        """Pull every resting order for one agent — how a ladder market maker requotes."""
        pulled = 0
        for side in (Side.BUY, Side.SELL):
            levels = self._side(side)
            for level in list(levels):
                keep = deque(o for o in level.orders if o.agent_id != agent_id)
                pulled += len(level.orders) - len(keep)
                level.orders = keep
                if level.is_empty:
                    levels.remove(level)
        return pulled

    # --- internals ----------------------------------------------------------------------------

    def _side(self, side: Side) -> list[PriceLevel]:
        return self._bids if side is Side.BUY else self._asks

    @staticmethod
    def _crosses(order: Order, level_price: Decimal) -> bool:
        if order.side is Side.BUY:
            return order.price >= level_price
        return order.price <= level_price

    def _rest(self, order: Order) -> None:
        levels = self._side(order.side)
        for level in levels:
            if level.price == order.price:
                level.orders.append(order)
                return
        level = PriceLevel(price=order.price, orders=deque([order]))
        levels.append(level)
        levels.sort(key=lambda lv: lv.price, reverse=order.side is Side.BUY)


__all__ = [
    "BookError",
    "Execution",
    "Liquidity",
    "Order",
    "OrderBook",
    "PriceLevel",
    "Side",
]
