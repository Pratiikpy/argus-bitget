"""Passive execution — what a maker fee costs you in fills.

A backtest that charges the maker fee is making a claim about fills, not about fees, and the claim
is usually false. Our own standing rules record the measurement: a sim assuming limit fills returned
**+90%** where the replay returned **-0.45%**. The fee was never the point — the fills were.

This module makes that claim payable. A strategy may ask for passive execution, but it must then
supply, per bar, the depth resting at its price and the volume that printed there. The queue model
in :mod:`argus.execution.queue` decides how much actually fills. Whatever does not fill either goes
unexecuted or is chased across the spread at the taker fee, and the strategy is charged for what it
actually did.

The asymmetry is deliberate and mirrors the Constitution's: passive execution may **reduce** the
size you achieve and **raise** the fee you pay. There is no configuration in which asking for maker
treatment improves a result for free.

**Adverse selection is the reason the shortfall is not noise.** A resting bid fills when the market
is coming to meet it and misses when the market runs away, so the fills you get are systematically
the ones you would rather not have had. This module does not model that directly — it models only
whether the queue cleared — so a passive result here is still optimistic. That is stated rather than
buried, because an execution model that flatters itself while claiming realism is worse than none.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from argus.cost.model import CostModel
from argus.execution.queue import (
    ProbQueue,
    QueueError,
    QueueModel,
    RestingOrder,
)

_ZERO = Decimal("0")


class PassiveExecutionError(ValueError):
    """Passive execution was requested without the evidence it requires."""


@dataclass(frozen=True, slots=True)
class BookObservation:
    """What the book looked like over one bar, at the price we rested at.

    Both fields are required. There is no default that lets a strategy claim maker treatment
    without saying what the book was doing — that default *is* the defect.
    """

    level_qty: Decimal
    """Quantity resting at our price when the order was placed."""

    traded_qty: Decimal
    """Volume that printed at our price over the bar."""

    end_level_qty: Decimal | None = None
    """Quantity resting at our price at the end of the bar. Supplying it lets the queue model
    credit cancellations; omitting it means only trades advance us, which is stricter."""

    peak_level_qty: Decimal | None = None
    """Maximum quantity resting at our price during the bar.

    This is the only field that can put anything *behind* us, and without it the two queue models
    are indistinguishable: an order that joins at the back of a level and never sees the level grow
    has ``back = 0``, so every probability function returns zero and :class:`ProbQueue` degenerates
    to :class:`RiskAdverseQueue`. It is real feed data, not an assumption — but omitting it is the
    stricter choice, because quantity behind you slows your advance, never speeds it.
    """

    def __post_init__(self) -> None:
        for name in ("level_qty", "traded_qty", "end_level_qty", "peak_level_qty"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise PassiveExecutionError(f"{name}={value}: book quantities cannot be negative")
        if self.peak_level_qty is not None and self.peak_level_qty < self.level_qty:
            raise PassiveExecutionError(
                f"peak_level_qty={self.peak_level_qty} is below level_qty={self.level_qty}; "
                "a peak cannot be lower than a level observed during the same bar"
            )


@dataclass(frozen=True, slots=True)
class PassiveFill:
    """What a passive attempt actually achieved."""

    intended: Decimal
    filled_passive: Decimal
    chased_taker: Decimal
    fee_bps: Decimal

    @property
    def unexecuted(self) -> Decimal:
        return self.intended - self.filled_passive - self.chased_taker

    @property
    def fill_rate(self) -> Decimal:
        if self.intended <= 0:
            return _ZERO
        return self.filled_passive / self.intended

    @property
    def achieved(self) -> Decimal:
        return self.filled_passive + self.chased_taker

    @property
    def was_worth_resting(self) -> bool:
        """Did passive execution beat simply crossing?

        False whenever the whole order ended up chased anyway: the strategy paid the taker fee
        *and* took the delay. This is the outcome a maker-assuming backtest never reports.
        """
        return self.filled_passive > 0


@dataclass(frozen=True, slots=True)
class PassiveExecution:
    """Rest an order and find out what fills.

    ``chase`` decides what happens to the remainder. ``False`` leaves it unexecuted — the strategy
    simply does not reach its target size, which is the honest outcome for a patient strategy.
    ``True`` crosses the spread at the taker fee, which is what a strategy that actually needs the
    position must do, and it is charged accordingly.
    """

    cost: CostModel
    model: QueueModel | None = None
    chase: bool = False
    lot_size: Decimal = Decimal("1")

    def _queue(self) -> QueueModel:
        return self.model or ProbQueue()

    def execute(self, intended_qty: Decimal, book: BookObservation) -> PassiveFill:
        """Attempt ``intended_qty`` passively against one bar of book activity."""
        if intended_qty <= 0:
            raise PassiveExecutionError(f"intended_qty={intended_qty} must be positive")

        order = RestingOrder(
            price=Decimal("1"),  # price is irrelevant to queue position; quantity is not
            quantity=intended_qty,
            model=self._queue(),
            level_qty=book.level_qty,
            lot_size=self.lot_size,
        )
        # Order matters and mirrors the bar: others queue up behind us, the level then erodes,
        # and volume prints against whatever is left ahead.
        if book.peak_level_qty is not None:
            order.on_depth_change(book.peak_level_qty)
        if book.end_level_qty is not None:
            order.on_depth_change(book.end_level_qty)
        if book.traded_qty > 0:
            order.on_trade_at_price(book.traded_qty)

        filled = order.filled
        remainder = intended_qty - filled
        chased = remainder if (self.chase and remainder > 0) else _ZERO

        # A blended rate, because the order really did execute two ways. Reporting only the maker
        # rate on a half-filled order is precisely the flattery this module exists to prevent.
        executed = filled + chased
        if executed <= 0:
            fee_bps = _ZERO
        else:
            fee_bps = (
                filled * self.cost.maker_bps + chased * self.cost.taker_bps
            ) / executed

        return PassiveFill(
            intended=intended_qty,
            filled_passive=filled,
            chased_taker=chased,
            fee_bps=fee_bps,
        )

    def effective_one_way_bps(self, book: BookObservation, *, size: Decimal) -> Decimal:
        """The one-way fee a strategy of this size actually pays on this book.

        This is the number the backtest engine needs: not the maker rate the strategy asked for,
        but the rate its fills earned it.
        """
        return self.execute(size, book).fee_bps


def book_from_bar(extra: dict[str, object] | None) -> BookObservation | None:
    """Read a book observation off a bar's ``extra`` mapping, or return ``None``.

    Returning ``None`` rather than a permissive default is the point: a caller that wants passive
    execution and has no book data gets a refusal, not an optimistic guess.
    """
    if not extra:
        return None
    try:
        level = extra["level_qty"]
        traded = extra["traded_qty"]
    except (KeyError, TypeError):
        return None
    end = extra.get("end_level_qty")
    peak = extra.get("peak_level_qty")
    try:
        return BookObservation(
            level_qty=Decimal(str(level)),
            traded_qty=Decimal(str(traded)),
            end_level_qty=None if end is None else Decimal(str(end)),
            peak_level_qty=None if peak is None else Decimal(str(peak)),
        )
    except (ArithmeticError, PassiveExecutionError, QueueError, ValueError):
        return None


__all__ = [
    "BookObservation",
    "PassiveExecution",
    "PassiveExecutionError",
    "PassiveFill",
    "book_from_bar",
]
