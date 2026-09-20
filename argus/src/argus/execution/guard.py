"""Pre-trade validation — the checks a venue will enforce anyway, done before we ask.

Three independent sources named the same gap. Nautilus runs a `RiskEngine` that denies an order
before it leaves the process, with a named reason, on price precision, quantity precision, notional
bounds and submission rate. Hummingbot's `BudgetChecker` locks collateral for a hypothetical order
and refuses when the balance cannot cover it. And the owner's own Nomos hit three separate 10x sizing
bugs that were only fixed by reading tick size, contract multiplier and minimum notional *live from
the venue* rather than assuming them (`research/architecture/kairos-nomos.md`).

ARGUS had none of it. `execution/preflight.py` checks that the venue is reachable and the signature
is accepted — a session-level check, run once. Nothing validated an individual order, so an order
that violated a venue rule would be discovered by the venue, as a rejection, after the decision had
been recorded as taken.

**The specifications are fetched, never assumed.** Every field below came from
`GET /api/v3/market/instruments` for the real instrument, and reading it settled a question our own
cost model had only asserted: Bitget publishes `takerFeeRate: 0.0006`, so a round trip is exactly
the 12bps `CostModel.bitget_perp` charges. The same response carries `minOrderAmount: 5`,
`minOrderQty: 0.01`, `maxOrderQty: 52000`, `priceMultiplier: 0.01` and a ±2% price band — none of
which anything in this system enforced.

**Quantities round DOWN, never to nearest.** This is the Nomos lesson stated as code: rounding a
size to the nearest multiple can round *up*, and a size that rounds up can cross the very ceiling
that was just checked. Rounding down can only ever make a position smaller, which is the same
asymmetry the Constitution obeys — a guard may shrink an order or refuse it, never enlarge it.

    python -m argus.execution.guard --symbol NVDAUSDT
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from decimal import ROUND_DOWN, Decimal
from enum import StrEnum
from typing import Any

INSTRUMENTS_URL = "https://api.bitget.com/api/v3/market/instruments"
PRODUCT_TYPE = "USDT-FUTURES"
_TIMEOUT = 20

_ZERO = Decimal("0")


class GuardError(RuntimeError):
    """The guard could not do its job. Raised rather than waved through.

    A validation layer that fails open is worse than none: it produces the confidence of a check
    without the check.
    """


class Denial(StrEnum):
    """Why an order was refused. Named states, after Nautilus's own denial vocabulary.

    A generic rejection tells an operator that something is wrong. A named one tells them what.
    """

    NONE = "none"
    INSTRUMENT_UNKNOWN = "instrument_unknown"
    INSTRUMENT_OFFLINE = "instrument_offline"
    QUANTITY_BELOW_MINIMUM = "quantity_below_minimum"
    QUANTITY_ABOVE_MAXIMUM = "quantity_above_maximum"
    NOTIONAL_BELOW_MINIMUM = "notional_below_minimum"
    PRICE_OUTSIDE_BAND = "price_outside_band"
    PRICE_NOT_POSITIVE = "price_not_positive"
    RATE_LIMIT = "rate_limit"
    INSUFFICIENT_BALANCE = "insufficient_balance"

    @property
    def allows(self) -> bool:
        return self is Denial.NONE


@dataclass(frozen=True, slots=True)
class Instrument:
    """One instrument's trading rules, as the venue publishes them.

    Every field is read from the venue. Nothing here has a default that would let a missing value
    pass as a permissive one — :meth:`from_payload` raises instead, because a silently-defaulted
    minimum is how an order gets through a check that never happened.
    """

    symbol: str
    status: str
    price_precision: int
    quantity_precision: int
    price_multiplier: Decimal
    quantity_multiplier: Decimal
    min_order_qty: Decimal
    max_order_qty: Decimal
    min_order_amount: Decimal
    """Minimum notional in quote currency. 5 USDT on the rTokens."""

    buy_limit_ratio: Decimal
    sell_limit_ratio: Decimal
    """How far from the reference price an order may sit, as a fraction. 0.02 is ±2%."""

    maker_fee_rate: Decimal
    taker_fee_rate: Decimal
    is_rwa: bool
    """Bitget's own flag for a tokenised real-world asset. The rTokens carry ``isRwa: YES``."""

    @property
    def is_online(self) -> bool:
        return self.status.lower() == "online"

    @property
    def round_trip_taker_bps(self) -> Decimal:
        """What the venue says a round trip costs, in bps.

        Worth computing here rather than trusting the cost model: the two agreeing is a check, and
        the two disagreeing is a finding.
        """
        return self.taker_fee_rate * 2 * Decimal("10000")

    @classmethod
    def from_payload(cls, row: dict[str, Any]) -> Instrument:
        def need(key: str) -> str:
            value = row.get(key)
            if value in (None, ""):
                raise GuardError(
                    f"instrument {row.get('symbol', '?')} has no {key}; refusing to substitute a "
                    f"default for a venue rule"
                )
            return str(value)

        return cls(
            symbol=need("symbol"),
            status=need("status"),
            price_precision=int(need("pricePrecision")),
            quantity_precision=int(need("quantityPrecision")),
            price_multiplier=Decimal(need("priceMultiplier")),
            quantity_multiplier=Decimal(need("quantityMultiplier")),
            min_order_qty=Decimal(need("minOrderQty")),
            max_order_qty=Decimal(need("maxOrderQty")),
            min_order_amount=Decimal(need("minOrderAmount")),
            buy_limit_ratio=Decimal(need("buyLimitPriceRatio")),
            sell_limit_ratio=Decimal(need("sellLimitPriceRatio")),
            maker_fee_rate=Decimal(need("makerFeeRate")),
            taker_fee_rate=Decimal(need("takerFeeRate")),
            is_rwa=str(row.get("isRwa", "")).upper() == "YES",
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol, "status": self.status, "is_rwa": self.is_rwa,
            "price_precision": self.price_precision,
            "quantity_precision": self.quantity_precision,
            "price_multiplier": str(self.price_multiplier),
            "quantity_multiplier": str(self.quantity_multiplier),
            "min_order_qty": str(self.min_order_qty),
            "max_order_qty": str(self.max_order_qty),
            "min_order_amount": str(self.min_order_amount),
            "price_band_pct": str(self.buy_limit_ratio * 100),
            "taker_fee_rate": str(self.taker_fee_rate),
            "round_trip_taker_bps": str(self.round_trip_taker_bps),
        }


def quantise_down(value: Decimal, step: Decimal) -> Decimal:
    """Round to a multiple of ``step``, always downward.

    Downward, never to nearest. Rounding to nearest can round *up*, and a size that rounds up can
    cross the ceiling that was checked a moment earlier — which is how a validated order becomes an
    oversized one between the check and the wire.
    """
    if step <= 0:
        raise GuardError(f"step must be positive, got {step}")
    return (value / step).to_integral_value(rounding=ROUND_DOWN) * step


@dataclass
class RateLimiter:
    """A sliding window over submission times.

    Nautilus enforces a maximum order rate because a bug that submits in a loop is the failure that
    exhausts a quota or a balance before anyone notices. The window is checked, never merely
    counted: a fixed-bucket counter lets twice the limit through at a bucket boundary.
    """

    max_orders: int = 10
    window_seconds: float = 60.0
    _stamps: deque[float] = field(default_factory=deque)

    def would_exceed(self, *, now: float) -> bool:
        self._prune(now)
        return len(self._stamps) >= self.max_orders

    def record(self, *, now: float) -> None:
        self._prune(now)
        self._stamps.append(now)

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._stamps and self._stamps[0] <= cutoff:
            self._stamps.popleft()

    @property
    def in_window(self) -> int:
        return len(self._stamps)


@dataclass(frozen=True, slots=True)
class Ruling:
    """What the guard decided about one order."""

    denial: Denial
    reason: str
    quantity: Decimal
    price: Decimal | None
    adjusted: bool
    """Whether the guard changed the order rather than only judging it."""

    @property
    def allowed(self) -> bool:
        return self.denial.allows

    def render(self) -> str:
        if self.allowed:
            return (
                f"[guard] allowed: {self.quantity}"
                + (f" @ {self.price}" if self.price is not None else "")
                + (f" — {self.reason}" if self.adjusted else "")
            )
        return f"[guard] DENIED {self.denial}: {self.reason}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "denial": str(self.denial), "allowed": self.allowed, "reason": self.reason,
            "quantity": str(self.quantity),
            "price": None if self.price is None else str(self.price),
            "adjusted": self.adjusted,
        }


def validate(
    instrument: Instrument,
    *,
    quantity: Decimal,
    price: Decimal | None = None,
    reference_price: Decimal | None = None,
    side: str = "buy",
    available_balance: Decimal | None = None,
    limiter: RateLimiter | None = None,
    now: float | None = None,
) -> Ruling:
    """Check one order against the venue's own rules, shrinking it or refusing it.

    Ordered so the cheapest and most fundamental checks come first, and so a denial names the
    reason a person would want: an offline instrument is reported as offline rather than as a
    precision failure further down.

    The guard may **reduce** a quantity to the venue's step and ceiling. It never raises one — not
    even to reach ``min_order_qty``, because increasing an order to satisfy a minimum would be the
    guard authoring size, and a validation layer that can enlarge a position is a strategy.
    """
    moment = time.monotonic() if now is None else now

    if not instrument.is_online:
        return Ruling(
            Denial.INSTRUMENT_OFFLINE,
            f"{instrument.symbol} is {instrument.status}, not online", _ZERO, price, False,
        )

    if limiter is not None and limiter.would_exceed(now=moment):
        return Ruling(
            Denial.RATE_LIMIT,
            f"{limiter.in_window} order(s) already submitted in the last "
            f"{limiter.window_seconds:.0f}s, at the cap of {limiter.max_orders}",
            _ZERO, price, False,
        )

    if price is not None and price <= 0:
        return Ruling(Denial.PRICE_NOT_POSITIVE, f"price {price} is not positive",
                      _ZERO, price, False)

    adjusted = False
    final_price = price
    if price is not None:
        stepped = quantise_down(price, instrument.price_multiplier)
        if stepped != price:
            adjusted = True
            final_price = stepped
        if reference_price is not None and reference_price > 0:
            ratio = (
                instrument.buy_limit_ratio if side.lower() == "buy"
                else instrument.sell_limit_ratio
            )
            upper = reference_price * (Decimal("1") + ratio)
            lower = reference_price * (Decimal("1") - ratio)
            if not lower <= (final_price or _ZERO) <= upper:
                return Ruling(
                    Denial.PRICE_OUTSIDE_BAND,
                    f"{final_price} is outside the venue's ±{ratio * 100}% band around "
                    f"{reference_price} ({lower} to {upper})",
                    _ZERO, final_price, False,
                )

    stepped_qty = quantise_down(quantity, instrument.quantity_multiplier)
    if stepped_qty != quantity:
        adjusted = True

    if stepped_qty > instrument.max_order_qty:
        stepped_qty = quantise_down(instrument.max_order_qty, instrument.quantity_multiplier)
        adjusted = True
        if stepped_qty <= 0:
            return Ruling(
                Denial.QUANTITY_ABOVE_MAXIMUM,
                f"{quantity} exceeds the venue maximum {instrument.max_order_qty} and cannot be "
                f"reduced to a valid step", _ZERO, final_price, True,
            )

    if stepped_qty < instrument.min_order_qty:
        return Ruling(
            Denial.QUANTITY_BELOW_MINIMUM,
            f"{quantity} rounds down to {stepped_qty}, below the venue minimum "
            f"{instrument.min_order_qty}; the guard may not round a size up to reach it",
            _ZERO, final_price, adjusted,
        )

    mark = final_price if final_price is not None else reference_price
    if mark is not None and mark > 0:
        notional = stepped_qty * mark
        if notional < instrument.min_order_amount:
            return Ruling(
                Denial.NOTIONAL_BELOW_MINIMUM,
                f"notional {notional} is below the venue minimum "
                f"{instrument.min_order_amount}", _ZERO, final_price, adjusted,
            )
        if available_balance is not None and notional > available_balance:
            return Ruling(
                Denial.INSUFFICIENT_BALANCE,
                f"notional {notional} exceeds the available balance {available_balance}",
                _ZERO, final_price, adjusted,
            )

    reason = (
        "adjusted to the venue's step and limits" if adjusted
        else "within every published venue rule"
    )
    if limiter is not None:
        limiter.record(now=moment)
    return Ruling(Denial.NONE, reason, stepped_qty, final_price, adjusted)


def fetch_instruments(
    *, symbol: str | None = None, product_type: str = PRODUCT_TYPE, timeout: int = _TIMEOUT
) -> dict[str, Instrument]:
    """Read the venue's published specifications.

    Public and keyless. A failure raises: a guard that silently fell back to assumed limits would
    be exactly the thing this module exists to replace.
    """
    params = {"category": product_type}
    if symbol:
        params["symbol"] = symbol
    url = f"{INSTRUMENTS_URL}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": "argus/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode())
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise GuardError(f"could not read instrument specifications: {exc}") from exc

    if str(payload.get("code")) != "00000":
        raise GuardError(f"venue refused the instrument query: {payload.get('msg', payload)}")

    rows = payload.get("data") or []
    out: dict[str, Instrument] = {}
    for row in rows:
        try:
            instrument = Instrument.from_payload(row)
        except (GuardError, ValueError, ArithmeticError):
            # One malformed row must not deny the whole venue. It is simply absent, and an absent
            # instrument is denied by name at validation rather than waved through.
            continue
        out[instrument.symbol] = instrument
    if not out:
        raise GuardError("the venue returned no usable instrument specifications")
    return out


@dataclass
class Guard:
    """Instrument rules plus a rate limiter, for one trading session."""

    instruments: dict[str, Instrument]
    limiter: RateLimiter = field(default_factory=RateLimiter)

    def check(
        self, symbol: str, *, quantity: Decimal, price: Decimal | None = None,
        reference_price: Decimal | None = None, side: str = "buy",
        available_balance: Decimal | None = None, now: float | None = None,
    ) -> Ruling:
        instrument = self.instruments.get(symbol)
        if instrument is None:
            return Ruling(
                Denial.INSTRUMENT_UNKNOWN,
                f"{symbol} is not among the {len(self.instruments)} instruments the venue "
                f"published; no rules are known for it",
                _ZERO, price, False,
            )
        return validate(
            instrument, quantity=quantity, price=price, reference_price=reference_price,
            side=side, available_balance=available_balance, limiter=self.limiter, now=now,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ARGUS pre-trade guard")
    parser.add_argument("--symbol", default="NVDAUSDT")
    parser.add_argument("--quantity", type=Decimal, default=Decimal("1"))
    parser.add_argument("--price", type=Decimal, default=None)
    args = parser.parse_args(argv)

    try:
        instruments = fetch_instruments(symbol=args.symbol)
    except GuardError as exc:
        print(f"guard unavailable: {exc}")
        return 1

    guard = Guard(instruments=instruments)
    instrument = instruments[args.symbol]
    print(json.dumps(instrument.as_dict(), indent=2))
    ruling = guard.check(
        args.symbol, quantity=args.quantity, price=args.price,
        reference_price=args.price,
    )
    print(ruling.render())
    return 0 if ruling.allowed else 1


__all__ = [
    "INSTRUMENTS_URL",
    "Denial",
    "Guard",
    "GuardError",
    "Instrument",
    "RateLimiter",
    "Ruling",
    "fetch_instruments",
    "main",
    "quantise_down",
    "validate",
]


if __name__ == "__main__":
    raise SystemExit(main())
