"""The live order book, and what it actually costs to cross it.

Until this module existed ARGUS charged **0.6 basis points of spread on every paper fill**, entry
and exit, for every instrument, at every hour — a literal in `paper/ledger.py:311,427`. The number
came from a median of top-of-book quotes; it is not absurd, and it is also not a measurement of
the thing it is standing in for. The cost of a trade is not the quoted spread. It is what you pay
to take the size you actually want, which depends on how much is resting behind the touch — and
nothing in the system had ever looked.

`GET /api/v3/market/orderbook` is public, keyless and returns up to 200 levels per side (`agent-
sdk/src/generated/catalog.ts:115`: ``category`` and ``symbol`` required, ``limit`` default 5,
maximum 200). It was in Bitget's own SDK catalogue the whole time and `market/bitget.py` never
called it.

* What this module computes, and why each piece is not the one before it:**

* :meth:`OrderBook.spread_bps` — the quoted touch. What everyone reports.
* :meth:`OrderBook.sweep` — the volume-weighted price of *taking a given notional*, walking as
  many   levels as it takes, and the slippage against the mid that results. This is the number a
  trade   actually pays, and it is strictly worse than the quoted spread for any size that does
  not fit at   the touch.
* :meth:`OrderBook.executable_within` — how much notional can be taken before slippage exceeds a
  stated budget. The inverse question, and the one a risk layer should be asking: not "what does
  my   size cost" but "what size can this book carry".
* :meth:`OrderBook.imbalance` — resting bid notional against ask notional over the top levels. Not
  used to predict anything here; reported because a book that is 80% one-sided makes the *other*
  side's slippage a different number from the one measured a minute ago.

* It also gives the square-root impact law something to be checked against.** `cost/model.py`
  charges market impact as ``coefficient * participation ** gamma`` with cvxportfolio's gamma of
  1.5 and a coefficient of 10, neither of which had ever been compared with this venue's own book.
  :func:`impact_check` walks the real book at a series of sizes and reports the modelled cost
  beside the measured one. Where they disagree the model is wrong, and a model that cannot be
  wrong is not a model.

* A snapshot is a snapshot.** This is the book at one instant, and it is not a promise about the
  book a second later — cancellations are free and most of what is quoted is not there to trade.
  Everything here is therefore reported with its timestamp, and :func:`fetch_orderbook` never
  caches: a stale book that looks fresh is worse than no book at all, which is the same rule
  `risk/effectiveness.py` applies to a stale correlation.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from argus.market.bitget import BitgetError, _dec, _get

MAX_LEVELS = 200
"""The venue's own cap (`catalog.ts:115`: "Level Default: 5. Maximum: 200")."""

DEFAULT_LEVELS = 50
"""Enough to price any size this desk trades, and a fifth of the maximum payload."""

CATEGORY = "USDT-FUTURES"
"""rTokens are USDT-margined perpetuals. The endpoint requires the category explicitly."""


class DepthError(RuntimeError):
    """Raised rather than returning a book that cannot be priced."""


@dataclass(frozen=True, slots=True)
class Level:
    """One price level: what is quoted, and how much of it."""

    price: Decimal
    quantity: Decimal

    @property
    def notional(self) -> Decimal:
        return self.price * self.quantity


@dataclass(frozen=True, slots=True)
class Sweep:
    """The result of taking a given notional from one side of the book."""

    side: str
    requested_notional: Decimal
    filled_notional: Decimal
    average_price: Decimal
    levels_consumed: int
    slippage_bps: Decimal
    """Volume-weighted execution price against the mid, in basis points. Always positive: it is a
    cost whichever side you take."""

    complete: bool
    """False when the visible book could not absorb the size. A partial sweep's slippage describes
    the part that filled and says nothing about the rest, which is the honest report."""

    def as_dict(self) -> dict[str, Any]:
        return {
            "side": self.side,
            "requested_notional": str(self.requested_notional),
            "filled_notional": str(self.filled_notional.quantize(Decimal("0.01"))),
            "average_price": str(self.average_price.quantize(Decimal("0.0001"))),
            "levels_consumed": self.levels_consumed,
            "slippage_bps": float(round(self.slippage_bps, 4)),
            "complete": self.complete,
        }


@dataclass(frozen=True, slots=True)
class OrderBook:
    """One snapshot, with the arithmetic a trade needs and nothing it does not."""

    symbol: str
    fetched_at: datetime
    bids: tuple[Level, ...]
    """Descending by price: ``bids[0]`` is the best bid."""

    asks: tuple[Level, ...]
    """Ascending by price: ``asks[0]`` is the best ask."""

    def __post_init__(self) -> None:
        if not self.bids or not self.asks:
            raise DepthError(f"{self.symbol} returned a one-sided book; it cannot be priced")
        if self.asks[0].price <= self.bids[0].price:
            raise DepthError(
                f"{self.symbol} is crossed: best bid {self.bids[0].price} >= best ask "
                f"{self.asks[0].price}. A crossed book is a feed error, not an arbitrage"
            )

    @property
    def mid(self) -> Decimal:
        return (self.bids[0].price + self.asks[0].price) / 2

    @property
    def spread_bps(self) -> Decimal:
        """The quoted touch. The number every venue reports and no trade of size pays."""
        return (self.asks[0].price - self.bids[0].price) / self.mid * Decimal("10000")

    def side(self, direction: str) -> tuple[Level, ...]:
        """``BUY`` takes the asks; ``SELL`` hits the bids. Getting this backwards would report the
        cost of the trade you did not do, so it is one function rather than a comparison at each
        call site."""
        if direction.upper() == "BUY":
            return self.asks
        if direction.upper() == "SELL":
            return self.bids
        raise DepthError(f"side must be BUY or SELL, not {direction!r}")

    def sweep(self, notional: Decimal, *, direction: str = "BUY") -> Sweep:
        """Walk the book for ``notional`` and report what it would cost.

        Level by level rather than by a depth-weighted approximation: the approximation is exact
        only when the size fits inside one level, which is the case this measurement exists to stop
        assuming.
        """
        if notional <= 0:
            raise DepthError("a sweep needs a positive notional")
        levels = self.side(direction)
        remaining = notional
        spent = Decimal("0")
        quantity = Decimal("0")
        consumed = 0
        for level in levels:
            if remaining <= 0:
                break
            take_notional = min(remaining, level.notional)
            take_qty = take_notional / level.price
            spent += take_qty * level.price
            quantity += take_qty
            remaining -= take_notional
            consumed += 1
        if quantity <= 0:
            raise DepthError(f"{self.symbol} has no quoted size on the {direction} side")
        average = spent / quantity
        mid = self.mid
        # Positive whichever way the trade goes: buying above the mid and selling below it are both
        # costs, and signing them differently would let a sell look like a credit.
        slippage = abs(average - mid) / mid * Decimal("10000")
        return Sweep(
            side=direction.upper(), requested_notional=notional, filled_notional=spent,
            average_price=average, levels_consumed=consumed, slippage_bps=slippage,
            complete=remaining <= 0,
        )

    def executable_within(self, budget_bps: Decimal, *, direction: str = "BUY") -> Decimal:
        """Largest notional whose sweep slippage stays inside ``budget_bps``.

        Computed by accumulating levels rather than by bisection on :meth:`sweep`, so the answer is
        exact at a level boundary instead of converged-to. The honest inverse of "what does my size
        cost": what size will this book carry at a price I am willing to pay.
        """
        if budget_bps <= 0:
            raise DepthError("the slippage budget must be positive")
        mid = self.mid
        levels = self.side(direction)
        spent = Decimal("0")
        quantity = Decimal("0")
        best = Decimal("0")
        for level in levels:
            spent += level.notional
            quantity += level.quantity
            if quantity <= 0:
                continue
            slippage = abs(spent / quantity - mid) / mid * Decimal("10000")
            if slippage > budget_bps:
                break
            best = spent
        return best

    def imbalance(self, levels: int = 10) -> Decimal:
        """Resting bid notional as a share of both sides, over the top ``levels``.

        0.5 is balanced. Reported, never traded on: order-book imbalance predicts the next tick in
        the literature and the next tick is not a horizon this desk operates at. It is here so a
        reader can see whether a measured slippage came from a book that was already lopsided.
        """
        bid = sum((level.notional for level in self.bids[:levels]), Decimal("0"))
        ask = sum((level.notional for level in self.asks[:levels]), Decimal("0"))
        total = bid + ask
        return Decimal("0.5") if total <= 0 else bid / total

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "fetched_at": self.fetched_at.isoformat(),
            "mid": str(self.mid.quantize(Decimal("0.0001"))),
            "spread_bps": float(round(self.spread_bps, 4)),
            "bid_levels": len(self.bids),
            "ask_levels": len(self.asks),
            "imbalance": float(round(self.imbalance(), 4)),
        }


def _levels(rows: Any, *, descending: bool) -> tuple[Level, ...]:
    """Bitget returns ``[[price, size], ...]``. Sorted here rather than trusted.

    The endpoint documents no ordering guarantee, and every calculation in this module assumes the
    touch is at index zero. Sorting costs nothing at 200 levels and removes a silent dependency on
    an undocumented property of somebody else's feed.
    """
    out: list[Level] = []
    for row in rows or []:
        if not isinstance(row, (list, tuple)) or len(row) < 2:
            continue
        price, quantity = _dec(row[0]), _dec(row[1])
        if price > 0 and quantity > 0:
            out.append(Level(price=price, quantity=quantity))
    return tuple(sorted(out, key=lambda level: level.price, reverse=descending))


def fetch_orderbook(
    symbol: str, *, limit: int = DEFAULT_LEVELS, category: str = CATEGORY,
) -> OrderBook:
    """One live snapshot. Never cached, never retried silently.

    ``limit`` is capped at the venue's documented maximum rather than passed through: asking for 500
    levels returns an error whose message is about a parameter, not about depth, and a caller
    debugging slippage should not have to learn that.
    """
    if limit < 1:
        raise DepthError("limit must be at least one level")
    payload = _get(
        "/api/v3/market/orderbook",
        {"category": category, "symbol": symbol.upper(), "limit": str(min(limit, MAX_LEVELS))},
    )
    if not isinstance(payload, dict):
        raise DepthError(f"{symbol} returned no order book")
    return OrderBook(
        symbol=symbol.upper(),
        fetched_at=datetime.now(UTC),
        bids=_levels(payload.get("bids") or payload.get("b"), descending=True),
        asks=_levels(payload.get("asks") or payload.get("a"), descending=False),
    )


def measured_spread_bps(
    symbol: str, notional: Decimal, *, direction: str = "BUY", limit: int = DEFAULT_LEVELS,
) -> Decimal | None:
    """The slippage a trade of this size would pay right now, or ``None`` if the book cannot say.

    ``None`` rather than a fallback constant, so a caller must decide what to do about an absent
    measurement instead of being handed a plausible number with no provenance. `paper/ledger.py`
    then charges its documented 0.6bps default **and the record says the book was unavailable**.
    """
    try:
        book = fetch_orderbook(symbol, limit=limit)
        sweep = book.sweep(notional, direction=direction)
    except (BitgetError, DepthError):
        return None
    if not sweep.complete:
        # A size the visible book cannot absorb has no measured cost — only a lower bound. Returning
        # the partial figure would report the cheap half of a trade that could not be done.
        return None
    return sweep.slippage_bps


def impact_check(
    book: OrderBook, sizes: Sequence[Decimal], *, direction: str = "BUY",
    adv_notional: Decimal | None = None,
) -> list[dict[str, Any]]:
    """The square-root impact law against the real book, size by size.

    `cost/model.py` charges ``b * sigma * participation ** 0.5`` per dollar — the square-root law
    as cvxportfolio states it, with ``b`` fitted on these books (`eval/impact_calibration.py`).
    Until 2026-09-24 it charged ``10 * participation ** 1.5``, which this table showed was
    thousands of times too small. The comparison needs a participation rate, so an average daily
    notional is required; without one only the measured column is reported, which is still the
    more useful half.
    """
    from argus.cost.model import CostModel

    model = CostModel.bitget_perp()
    rows: list[dict[str, Any]] = []
    for size in sizes:
        try:
            sweep = book.sweep(size, direction=direction)
        except DepthError:
            continue
        row: dict[str, Any] = {
            "notional": str(size),
            "measured_slippage_bps": float(round(sweep.slippage_bps, 4)),
            "levels_consumed": sweep.levels_consumed,
            "complete": sweep.complete,
        }
        if adv_notional and adv_notional > 0:
            participation = size / adv_notional
            # `CostModel.impact_bps` is the square-root law per dollar, b * sigma * sqrt(p), with
            # b fitted on these same books (`eval/impact_calibration.py`).
            modelled = model.impact_bps(participation)
            row["participation"] = float(round(participation, 8))
            row["modelled_impact_bps"] = float(round(modelled, 4))
        rows.append(row)
    return rows


def main() -> int:  # pragma: no cover - CLI
    import argparse
    import sys
    from pathlib import Path

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from argus.market.bitget import RTOKEN_SYMBOLS

    parser = argparse.ArgumentParser(description="what does it cost to cross this book?")
    parser.add_argument("--symbols", default="")
    parser.add_argument("--limit", type=int, default=DEFAULT_LEVELS)
    parser.add_argument(
        "--sizes", default="1000,5000,25000,100000", help="notionals to price, comma separated",
    )
    args = parser.parse_args()

    wanted = [s.strip().upper() for s in args.symbols.split(",") if s.strip()] or list(
        RTOKEN_SYMBOLS
    )
    sizes = [Decimal(s.strip()) for s in args.sizes.split(",") if s.strip()]
    header = f"{'symbol':12} {'quoted':>8} " + " ".join(f"{'$' + str(int(s)):>10}" for s in sizes)
    # The two kinds of number in this table are not the same measurement, and the header used to
    # claim they were ("slippage in bps against mid, taking the ask"). `quoted` is the FULL
    # touch-to-touch spread (`Book.spread_bps`, line 134); the size columns are mid-relative
    # (`Sweep.slippage_bps`, line 185). A size that fits inside the top level pays the HALF spread,
    # so it reads exactly half `quoted` — QQQUSDT prints quoted 0.14 beside $1000 0.07. Anyone
    # taking a quoted-vs-executable ratio off the old header divided by a denominator twice too
    # large and understated the gap the table exists to show.
    print("EXECUTABLE COST — bps. 'quoted' is the full bid-ask spread; the size columns are")
    print("slippage against mid, taking the ask (a size inside the touch reads half 'quoted').\n")
    print(header)
    print("-" * len(header))
    rows: list[dict[str, Any]] = []
    for symbol in wanted:
        try:
            book = fetch_orderbook(symbol, limit=args.limit)
        except (BitgetError, DepthError) as exc:
            print(f"{symbol:12} unavailable ({exc})")
            continue
        cells: list[str] = []
        record: dict[str, Any] = {**book.as_dict(), "sweeps": []}
        for size in sizes:
            sweep = book.sweep(size, direction="BUY")
            cells.append(f"{float(sweep.slippage_bps):10.2f}" if sweep.complete else f"{'—':>10}")
            record["sweeps"].append(sweep.as_dict())
        rows.append(record)
        print(f"{symbol:12} {float(book.spread_bps):8.2f} " + " ".join(cells))

    if not rows:
        print("\nno book was readable; nothing written")
        return 1
    out = Path(__file__).resolve().parents[3] / "data" / "depth.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"generated_at": datetime.now(UTC).isoformat(), "books": rows}, indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n{len(rows)} book(s) written to {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = [
    "CATEGORY",
    "DEFAULT_LEVELS",
    "MAX_LEVELS",
    "DepthError",
    "Level",
    "OrderBook",
    "Sweep",
    "fetch_orderbook",
    "impact_check",
    "measured_spread_bps",
]
