"""Queue-position fill modelling — ported from hftbacktest (MIT).

Our own competitive assessment records this as a real gap: ARGUS models execution as a *cost*,
while hftbacktest models it as a *book*. A resting limit order does not fill because the price
touched it — it fills when the quantity ahead of it in the queue is gone. Ignoring that is how a
backtest awards itself maker fills it would never have received, and it is the mechanism behind the
measurement in our own standing rules: a local sim showed **+90% assuming limit fills; the real
replay gave -0.45%**.

Ported from ``nkaz001/hftbacktest`` (MIT, ``backtest/models/queue.rs``), reproduced faithfully
rather than approximated:

**RiskAdverse** (``queue.rs:44-96``) — only *trades* advance you.

    qty_ahead := bid_qty_at_tick(price)     on entry
    qty_ahead -= trade_qty                  on a trade at your price
    qty_ahead := min(qty_ahead, new_qty)    on a depth change; never rises
    exec      := round(-qty_ahead / lot) * lot

Conservative on purpose: it assumes no order ahead of you is ever cancelled. Real books lose orders
to cancellation constantly, so this *understates* your fill rate — the correct direction to be wrong
in, because the opposite manufactures an edge that does not exist.

**ProbQueue** (``queue.rs:124-217``) — when depth falls you cannot tell whether the cause was a
trade ahead of you (already counted) or a cancellation (which advances you), so it is apportioned:

    chg  = prev_qty - new_qty - cum_trade_qty
    back = prev_qty - front
    prob = P(the decrease was a cancellation *behind* you)
    est  = front - (1 - prob)·chg + min(back - prob·chg, 0)
    front := min(est, new_qty)

The orientation matters and is easy to invert: ``prob`` rises with **back**, not front. A level
that is mostly behind you loses its cancellations behind you, so your position barely improves;
a level that is mostly ahead of you loses them ahead, and you advance. Five probability functions
ship with hftbacktest (``queue.rs:221-330``) and all five are here, because the choice of function
is exactly the assumption a reader should be able to argue with.

**L3FIFO** (``queue.rs:481-1050``) — with a market-by-order feed there is nothing to estimate. The
queue is known, so it is kept explicitly and the order fills when everything ahead of it is gone.

**Why this matters here specifically.** Our measured environment is a book at roughly a third of
RTH depth for 65 hours a week. That is precisely where queue position dominates: on a thin book the
quantity ahead of you barely turns over, so a passive order a touch-based model fills may never fill
at all — and the once it does is when the market has already moved against you.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal
from math import log

_ZERO = Decimal("0")
_ONE = Decimal("1")
_HALF = Decimal("0.5")


class QueueError(ValueError):
    """The queue model was given something it cannot model honestly."""


# ---------------------------------------------------------------------------
# Probability functions — queue.rs:221-330
# ---------------------------------------------------------------------------


class Probability(ABC):
    """P(a depth decrease was a cancellation *behind* the order).

    Kept as its own type exactly as hftbacktest keeps it as its own trait, so the estimator and the
    assumption it rests on can be varied independently — and so the assumption has a name.
    """

    name: str = "probability"

    @abstractmethod
    def prob(self, front: Decimal, back: Decimal) -> Decimal:
        """Return a value in [0, 1], rising with ``back``."""

    def _clamp(self, value: Decimal) -> Decimal:
        # hftbacktest guards `if prob.is_infinite() { prob = 1.0 }`. Decimal cannot silently
        # produce an infinity here, but a degenerate level can still push the ratio out of range,
        # and a prob outside [0, 1] would let the estimator move the order backwards.
        return max(_ZERO, min(_ONE, value))


class PowerProbability(Probability):
    """``back^n / (back^n + front^n)`` — ``PowerProbQueueFunc``, ``queue.rs:221-240``.

    ``n=1`` is linear: cancellation probability is simply the share of quantity behind you. This is
    the most cited form in the microstructure literature and is our default.
    """

    def __init__(self, n: Decimal = _ONE) -> None:
        if n <= 0:
            raise QueueError(f"n={n} must be positive")
        self._n = n
        self.name = f"power(n={n})"

    def prob(self, front: Decimal, back: Decimal) -> Decimal:
        f, b = max(front, _ZERO), max(back, _ZERO)
        if f == 0 and b == 0:
            return _HALF
        fn, bn = f**self._n, b**self._n
        total = fn + bn
        return self._clamp(bn / total) if total > 0 else _HALF


class PowerProbability2(Probability):
    """``back^n / (back + front)^n`` — ``PowerProbQueueFunc2``, ``queue.rs:288-307``."""

    def __init__(self, n: Decimal = _ONE) -> None:
        if n <= 0:
            raise QueueError(f"n={n} must be positive")
        self._n = n
        self.name = f"power2(n={n})"

    def prob(self, front: Decimal, back: Decimal) -> Decimal:
        f, b = max(front, _ZERO), max(back, _ZERO)
        if f + b == 0:
            return _HALF
        return self._clamp(b**self._n / (f + b) ** self._n)


class PowerProbability3(Probability):
    """``1 - (front / (front + back))^n`` — ``PowerProbQueueFunc3``, ``queue.rs:311-330``."""

    def __init__(self, n: Decimal = _ONE) -> None:
        if n <= 0:
            raise QueueError(f"n={n} must be positive")
        self._n = n
        self.name = f"power3(n={n})"

    def prob(self, front: Decimal, back: Decimal) -> Decimal:
        f, b = max(front, _ZERO), max(back, _ZERO)
        if f + b == 0:
            return _HALF
        return self._clamp(_ONE - (f / (f + b)) ** self._n)


class LogProbability(Probability):
    """``ln(1+back) / (ln(1+back) + ln(1+front))`` — ``LogProbQueueFunc``, ``queue.rs:244-262``.

    Log-damped: as a level deepens, the advantage of having more quantity behind you diminishes.
    Use this when large book imbalances would otherwise drive the linear form to near-certainty.
    """

    name = "log"

    def prob(self, front: Decimal, back: Decimal) -> Decimal:
        f = Decimal(str(log(1 + float(max(front, _ZERO)))))
        b = Decimal(str(log(1 + float(max(back, _ZERO)))))
        if f + b == 0:
            return _HALF
        return self._clamp(b / (f + b))


class LogProbability2(Probability):
    """``ln(1+back) / ln(1+back+front)`` — ``LogProbQueueFunc2``, ``queue.rs:266-284``.

    Same numerator, a denominator over the whole level rather than the sum of two logs.

    It is *always at least* :class:`LogProbability`, never below it, and the proof is one line:
    ``ln(1+b) + ln(1+f) = ln((1+b)(1+f)) = ln(1+b+f+bf) >= ln(1+b+f)``, so this form's denominator
    is the smaller of the two. Our own architecture note calls it "less aggressive"; the note is
    wrong, and the arithmetic is what is implemented here.
    """

    name = "log2"

    def prob(self, front: Decimal, back: Decimal) -> Decimal:
        f, b = max(front, _ZERO), max(back, _ZERO)
        denom = Decimal(str(log(1 + float(f + b))))
        if denom == 0:
            return _HALF
        return self._clamp(Decimal(str(log(1 + float(b)))) / denom)


# ---------------------------------------------------------------------------
# Queue models — queue.rs:44-217
# ---------------------------------------------------------------------------


@dataclass
class QueuePosition:
    """Where a resting order sits, and what has happened in front of it."""

    front_qty: Decimal
    """Quantity ahead. Once it goes negative, that overshoot is what executes."""

    cum_trade_qty: Decimal = _ZERO
    """Trades counted since the last depth update, so the same volume is not counted twice — once
    as a trade and again as a cancellation when the level's total next changes."""

    filled_qty: Decimal = _ZERO

    @property
    def has_traded_through(self) -> bool:
        """Has volume passed the order's place in the queue?

        Reaching the *front* is not the same as filling: at ``front_qty == 0`` you are next, and
        nothing has yet traded through you.
        """
        return self.front_qty < _ZERO


class QueueModel(ABC):
    """How a resting order's position evolves as the book moves."""

    name: str = "queue"

    def new_order(self, level_qty: Decimal) -> QueuePosition:
        """Join the back of the queue: everything already resting is ahead of you.

        This is the only honest assumption for a backtest. Anything kinder amounts to claiming
        priority that was never earned.
        """
        return QueuePosition(front_qty=max(level_qty, _ZERO))

    @abstractmethod
    def on_trade(self, pos: QueuePosition, traded_qty: Decimal) -> None:
        """A trade printed at the order's price."""

    @abstractmethod
    def on_depth(self, pos: QueuePosition, prev_qty: Decimal, new_qty: Decimal) -> None:
        """The total quantity resting at the order's level changed."""

    def executable(
        self, pos: QueuePosition, leaves_qty: Decimal, *, lot_size: Decimal = Decimal("1")
    ) -> Decimal:
        """How much executes now — ``queue.rs:88-95``.

        hftbacktest computes ``round(-front_q_qty / lot_size) * lot_size``: the fill is the volume
        that traded *past* the order, not the whole order. A model that fills the full size the
        moment the queue empties is claiming depth the trade never had.
        """
        if lot_size <= 0:
            raise QueueError(f"lot_size={lot_size} must be positive")
        if not pos.has_traded_through:
            return _ZERO
        lots = (-pos.front_qty / lot_size).to_integral_value()
        return max(_ZERO, min(leaves_qty, lots * lot_size))

    def apply_fill(
        self, pos: QueuePosition, leaves_qty: Decimal, *, lot_size: Decimal = Decimal("1")
    ) -> Decimal:
        """Execute what is executable and record it."""
        got = self.executable(pos, leaves_qty, lot_size=lot_size)
        if got > 0:
            pos.filled_qty += got
            # The consumed depth is no longer ahead of anyone; leaving it negative would let the
            # same volume fill a later increment of the order for free.
            pos.front_qty += got
        return got


class RiskAdverseQueue(QueueModel):
    """Only trades advance the order. ``queue.rs:44-96``.

    Understates fills by construction — real books lose orders ahead of you to cancellation all
    day. Use it when a strategy's viability must not depend on an assumption about other people's
    cancellations.

    **RIVAL LENS check, 2026-09-22: read `queue.rs` again with fresh eyes looking for an hftbacktest
    model ARGUS had not ported.** This one was already ported, already wired into
    `eval/queueproof.py::_models()`, and already correctly labelled an ablation rather than a
    contender — verified rather than assumed. Its measured `queue_error`/`fill_error` are exactly
    identical, to five decimal places, to the `"every cancel is behind you"` constant-probability
    ablation in every regime tested — both synthetic (`data/queue_proof.json`, all five regimes,
    including the two where both score a perfect 0.0) and the real 2023-12-25 ESH4 MBO session
    (`data/mbo_queue_proof.json`: 0.05448 both). That is not a coincidence to re-derive next time:
    never attributing a cancellation to the front of the queue is mathematically the same policy
    as `const(1)`'s probability function, so the two models make identical predictions on every
    input this project's harness can construct. Neither beats the current best model
    (`LogProbability2`) on either the synthetic or the real data. A genuinely different algorithm
    from the rival's own source, checked and confirmed not to unlock a win here — logged so the
    next RIVAL LENS pass on this capability does not spend an afternoon re-confirming it.
    """

    name = "risk_adverse"

    def on_trade(self, pos: QueuePosition, traded_qty: Decimal) -> None:
        pos.front_qty -= max(traded_qty, _ZERO)

    def on_depth(self, pos: QueuePosition, prev_qty: Decimal, new_qty: Decimal) -> None:
        # Never rises: a level refilling behind you does not push you back.
        pos.front_qty = min(pos.front_qty, max(new_qty, _ZERO))


class ProbQueue(QueueModel):
    """Apportion a depth decrease between trades and cancellations. ``queue.rs:124-217``."""

    def __init__(self, probability: Probability | None = None) -> None:
        self._prob = probability or PowerProbability()
        self.name = f"prob[{self._prob.name}]"

    @property
    def probability(self) -> Probability:
        return self._prob

    def on_trade(self, pos: QueuePosition, traded_qty: Decimal) -> None:
        qty = max(traded_qty, _ZERO)
        pos.front_qty -= qty
        pos.cum_trade_qty += qty

    def on_depth(self, pos: QueuePosition, prev_qty: Decimal, new_qty: Decimal) -> None:
        chg = prev_qty - new_qty
        # Trades at this level were already applied by on_trade. Counting them again here would
        # advance the order twice for one event — the single subtlest bug in this model.
        chg -= pos.cum_trade_qty
        pos.cum_trade_qty = _ZERO

        if chg < 0:
            # The level grew. New quantity joins behind you, so your position can only improve to
            # the level's own size.
            pos.front_qty = min(pos.front_qty, max(new_qty, _ZERO))
            return

        front = pos.front_qty
        back = prev_qty - front
        prob = self._prob.prob(front, back)

        # queue.rs:212 — est_front = front - (1 - prob)*chg + min(back - prob*chg, 0)
        # The trailing min() term is what happens when the apportioned share exceeds the quantity
        # actually behind you: the excess has to come off the front.
        est_front = front - (_ONE - prob) * chg + min(back - prob * chg, _ZERO)
        pos.front_qty = min(est_front, max(new_qty, _ZERO))


# ---------------------------------------------------------------------------
# L3 — queue.rs:481-1050
# ---------------------------------------------------------------------------


@dataclass
class L3Order:
    """One order in a market-by-order queue."""

    order_id: str
    quantity: Decimal
    is_ours: bool = False


class L3FIFOQueue:
    """An explicit FIFO queue per price level. ``queue.rs:481-1050``.

    With a market-by-order feed the queue is *known*, so nothing is estimated: our order is appended
    behind everything resting, and it fills when everything ahead of it is gone. This is the
    ground truth the probabilistic models approximate, and having it here is what lets us say how
    wrong those approximations are rather than assuming they are close.
    """

    name = "l3_fifo"

    def __init__(self) -> None:
        self._levels: dict[Decimal, deque[L3Order]] = {}

    def add(self, price: Decimal, order: L3Order) -> None:
        self._levels.setdefault(price, deque()).append(order)

    def cancel(self, price: Decimal, order_id: str) -> bool:
        queue = self._levels.get(price)
        if queue is None:
            return False
        for i, existing in enumerate(queue):
            if existing.order_id == order_id:
                del queue[i]
                return True
        return False

    def ahead_of(self, price: Decimal, order_id: str) -> Decimal | None:
        """Quantity resting ahead of an order, or ``None`` if it is not in the book."""
        queue = self._levels.get(price)
        if queue is None:
            return None
        ahead = _ZERO
        for existing in queue:
            if existing.order_id == order_id:
                return ahead
            ahead += existing.quantity
        return None

    def reduce(self, price: Decimal, order_id: str, new_quantity: Decimal) -> bool:
        """Shrink a resting order in place, keeping its queue position.

        Added for real market-by-order feeds' Modify/Change message, which `queue.rs` itself
        never has to handle — the reference port only ever sees Add/Cancel/Trade, because
        hftbacktest's own examples build L2 from L3 rather than replaying a raw feed
        (`docs/order_fill.rst`). A real MBO feed does carry Modify, and CME's MDP3 — like every
        FIFO order-by-order venue this project has read about — treats a pure size *decrease* as
        priority-preserving: the order keeps its place and simply gets smaller, no requeue. This
        is standard across L3 feed protocols generally (ITCH, MDP3) but is NOT independently
        confirmed against CME's own primary spec, which sits behind a login this project could
        not reach — stated as unverified rather than silently assumed settled.

        A size *increase* or a price change is NOT this method's job: both lose priority on a
        real venue, and the caller must model that as `cancel` followed by `add` at the back, the
        same as a brand new order. Passing a `new_quantity` at or above the current one is a
        caller error, not something this method can safely interpret as "increase, keep
        priority" — so it does nothing and returns ``False`` rather than guess.
        """
        queue = self._levels.get(price)
        if queue is None:
            return False
        for existing in queue:
            if existing.order_id == order_id:
                if new_quantity <= _ZERO or new_quantity >= existing.quantity:
                    return False
                existing.quantity = new_quantity
                return True
        return False

    def on_trade(self, price: Decimal, traded_qty: Decimal) -> list[tuple[str, Decimal]]:
        """Consume the front of the queue and report what filled, in queue order."""
        queue = self._levels.get(price)
        if queue is None:
            return []
        remaining = max(traded_qty, _ZERO)
        fills: list[tuple[str, Decimal]] = []
        while remaining > 0 and queue:
            head = queue[0]
            taken = min(head.quantity, remaining)
            head.quantity -= taken
            remaining -= taken
            if head.is_ours:
                fills.append((head.order_id, taken))
            if head.quantity <= 0:
                queue.popleft()
        return fills

    def depth_at(self, price: Decimal) -> Decimal:
        return sum((o.quantity for o in self._levels.get(price, ())), _ZERO)


# ---------------------------------------------------------------------------
# The resting order the backtester actually holds
# ---------------------------------------------------------------------------


@dataclass
class RestingOrder:
    """A passive order and its queue position, advanced by book events.

    The invariant that makes this useful: a :class:`RestingOrder` cannot report a fill that the
    tape did not justify. There is no path from "the price touched my level" to ``filled``.
    """

    price: Decimal
    quantity: Decimal
    model: QueueModel
    level_qty: Decimal = _ZERO
    lot_size: Decimal = Decimal("1")
    position: QueuePosition = field(init=False)

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise QueueError(f"quantity={self.quantity} must be positive")
        self.position = self.model.new_order(self.level_qty)

    def on_trade_at_price(self, traded_qty: Decimal) -> Decimal:
        """A trade printed at our price. Returns the quantity that executed for us."""
        self.model.on_trade(self.position, traded_qty)
        return self.model.apply_fill(self.position, self.leaves, lot_size=self.lot_size)

    def on_depth_change(self, new_level_qty: Decimal) -> None:
        self.model.on_depth(self.position, self.level_qty, new_level_qty)
        self.level_qty = max(new_level_qty, _ZERO)

    @property
    def filled(self) -> Decimal:
        return self.position.filled_qty

    @property
    def leaves(self) -> Decimal:
        return self.quantity - self.position.filled_qty

    @property
    def is_complete(self) -> bool:
        return self.leaves <= _ZERO

    @property
    def queue_ahead(self) -> Decimal:
        return max(self.position.front_qty, _ZERO)


def fill_ratio(orders: list[RestingOrder]) -> Decimal:
    """Share of intended passive quantity that actually executed.

    This is the number that separates a maker strategy from a wish. Below roughly 0.5 on a
    realistic book, any result quoting maker fees is quoting fees it would not have paid.
    """
    intended = sum((o.quantity for o in orders), _ZERO)
    if intended <= 0:
        return _ZERO
    return sum((o.filled for o in orders), _ZERO) / intended


__all__ = [
    "L3FIFOQueue",
    "L3Order",
    "LogProbability",
    "LogProbability2",
    "PowerProbability",
    "PowerProbability2",
    "PowerProbability3",
    "ProbQueue",
    "Probability",
    "QueueError",
    "QueueModel",
    "QueuePosition",
    "RestingOrder",
    "RiskAdverseQueue",
    "fill_ratio",
]
