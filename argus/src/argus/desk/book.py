"""Foundation 3 — the real portfolio. Positions, account state, and reconciliation.

**The gap this closes.** Named the single biggest structural gap in the product plan
(`Activity/08_TRADING_OS_PLAN.md`, "Foundation 3"): `desk/portfolio.py` computes risk from
caller-supplied *weights*, `risk/circuit.py`'s `BookState` carries only scalar equity/drawdown
figures, and nothing anywhere stores an actual position. Every proposal has therefore been
evaluated against an assumed book, never a real one. This module is the book.

**Read before written, per standing rule #3.** `nautilus_trader` (LGPLv3 — patterns read, nothing
vendored) was read directly and verified line-by-line, not taken from a summary:

* ``crates/model/src/position.rs:62-104`` — the ``Position`` struct: an append-only ``events``
  fill history, ``avg_px_open``/``avg_px_close``, signed quantity, realized PnL.
* ``crates/model/src/position.rs:972-1010`` — the weighted-average price update, verified directly:
  ``avg_px' = (avg_px * qty + last_px * last_qty) / (qty + last_qty)``. :func:`Position.apply_fill`
  below is this formula in ``Decimal``.
* ``crates/model/src/position.rs:519-551`` — a fill that reduces the position closes at the
  existing average and books realized PnL; a fill that *exceeds* the open quantity flips the
  position and the excess opens fresh at the fill price (``avg_px_open = last_px``, line 551).
  Reproduced in :func:`Position.apply_fill` for correctness in one-way mode, even though the
  primary target venue mode (below) does not naturally reach it.
* ``crates/model/src/accounts/base.rs`` and ``margin.rs`` — read for the margin-accounting shape,
  and **rejected**: see "What is deliberately not here" below.
* ``crates/execution/src/reconciliation/positions.rs:51-171`` — read for the reconciliation shape;
  its synthetic-fill backfill (line 90-112) is rejected, see below.

**What is deliberately not here, and why — this is not "less than Nautilus", it is a different
venue.** Nautilus reimplements margin arithmetic because it spans many venues with different rules
and must simulate them offline. ARGUS has one venue. Called live against the real account
(`mcp__bitget-agentic__account_overview`, category USDT-FUTURES, 2026-09-15) rather than assumed:

* ``accountMode: "unified"``, ``assetMode: "multi_assets"`` — margin is computed **across assets**
  by Bitget's own engine, not per instrument. Nautilus's ``MarginAccount.margins`` (per-instrument
  isolated margin, ``margin.rs:67-80``) does not describe this venue and is not built.
* The venue call returns ``imr``, ``mmr``, ``mgnRatio``, ``positionMgnRatio``, ``accountEquity``,
  ``unrealisedPnl`` directly. **Recomputing these locally, as the first-pass design brief for this
  module proposed ("CALCULATE locally... DO NOT rely on venue"), was rejected on reflection**: that
  guidance is right for a pre-trade buying-power *estimate* (:class:`OpenOrderReservation` does
  exactly that, locally, below) but wrong for the margin ratio that actually determines
  liquidation — Bitget's cross-asset engine is proprietary and not fully reproducible from a
  client-side model, so a locally recomputed figure could silently diverge from the number that
  actually triggers liquidation. :class:`VenueMarginSnapshot` below is a **pass-through of what the
  venue reports**, timestamped, never re-derived.
* **Direction and conceptual threshold of `mgnRatio`/`positionMgnRatio`: PRIMARY-VERIFIED,
  2026-09-15.** Bitget's own liquidation page
  (`bitget.com/futures/introduction/liquidation-summarize`, fetched and read directly) states it
  in the platform's own words: *"maintenance margin ratio = 100%"* triggers liquidation in
  cross-margin mode, escalating to 160% for Tier 2+ accounts with open orders. Higher is worse.
  **Numeric scale: STRONG EVIDENCE, resolved later the same day, short of a direct proof.**
  `ccxt` (read locally, not guessed) shows a real non-zero example of Bitget's raw V2 Mix
  **position-level** `marginRatio`, `"0.029599540355"` (`bitget.ts:8063`), passed through
  unchanged by `ccxt`'s own parser — plainly a fraction, not a percentage. The **account-level**
  `mgnRatio` this class actually carries is inferred by naming-family analogy, not by an
  independent non-zero example of that exact field (`ccxt` never maps it; its own worked example
  is `"0"`). `margin_usage` (built on this evidence, see below) is therefore built and gated, with
  the analogy-not-proof distinction stated in its own reason string, not hidden.
* **Hedge-mode changes what a "position" is, verified live, not assumed.** The account's
  ``holdMode`` is ``"hedge_mode"`` (confirmed live, matches the mode `execution/bitget_client.py`
  already detects and branches on via ``tradeSide``). In hedge mode Bitget holds a long and a short
  position **simultaneously in the same symbol** as two independent venue records. A position here
  is therefore keyed by ``(symbol, side)``, not by symbol alone — a design point neither Nautilus
  (single side per instrument by convention) nor a generic "positions" writeup would surface,
  because it is specific to this venue's hedge mode.
* Per-instrument leverage ratios, options/premium handling, and a betting account type: rejected
  outright — the categories do not exist on this venue (`USDT-FUTURES`/`COIN-FUTURES`/
  `USDC-FUTURES` only; no options product).
* Synthetic fill backfill on reconciliation (Nautilus, ``reconciliation/positions.rs:90-112``):
  rejected. A missing opening fill is a fact to surface, not a number to invent — see
  :class:`ReconciliationDiff`.

**Hedge relationships stay informational, following Nautilus's own explicit design choice**
(no file describes a hedge model; the *absence* is the finding — position linkage is strategy-level
knowledge, not a domain-model invariant, because there is no universal definition of "hedge" a
generic model could enforce). :class:`HedgeLink` exists for reporting only and is never read by
margin or PnL arithmetic.

**Cost basis: lazy weighted average, not FIFO lots.** Bitget futures settle mark-to-market, not by
matched lot; a FIFO cost-basis ledger answers a tax question this venue's product does not ask.
Every fill is still kept (:attr:`Position.lots`), append-only, so the audit trail exists even though
the position's headline numbers are the running weighted average.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any


class PositionSide(StrEnum):
    """One venue position record's side. Bitget hedge mode holds both per symbol at once."""

    LONG = "long"
    SHORT = "short"


class PositionError(RuntimeError):
    """A fill could not be applied without breaking an invariant this module guarantees."""


@dataclass(frozen=True, slots=True)
class Lot:
    """One fill, kept forever. The audit trail; not the pricing model.

    Bound to the order and Constitution verdict that authorised it, the same way
    `execution.orders.Order` is — a lot with no ``approved_intent_hash`` cannot be traced to a
    decision, which is the same defect that hash exists to prevent on the order itself.
    """

    order_id: str
    approved_intent_hash: str
    side: str
    quantity: Decimal
    price: Decimal
    commission: Decimal
    ts_filled: datetime

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise PositionError(f"lot quantity must be positive, got {self.quantity}")
        if self.price <= 0:
            raise PositionError(f"lot price must be positive, got {self.price}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "order_id": self.order_id,
            "side": self.side,
            "quantity": str(self.quantity),
            "price": str(self.price),
            "commission": str(self.commission),
            "ts_filled": self.ts_filled.isoformat(),
        }


@dataclass(slots=True)
class Position:
    """One ``(symbol, side)`` slot's state, rebuilt fill by fill.

    Never constructed with a starting quantity — a position exists because
    :meth:`apply_fill` was called with real fills, the same append-only discipline the order book
    already uses. There is no setter for ``quantity`` or ``entry_price``; both are derived.
    """

    symbol: str
    side: PositionSide
    quantity: Decimal = Decimal("0")
    entry_price: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    lots: list[Lot] = field(default_factory=list)
    ts_opened: datetime | None = None
    ts_last: datetime | None = None

    @property
    def is_flat(self) -> bool:
        return self.quantity == 0

    def apply_fill(self, lot: Lot) -> None:
        """Fold one fill into the running position.

        ``lot.side == self.side.value`` (e.g. a BUY into a LONG slot) **opens or adds**: the
        weighted average moves per the verified Nautilus formula
        (``position.rs:999-1010``, reproduced here in ``Decimal``). The opposite side **reduces**:
        it closes at the *existing* average and books realized PnL on the closed quantity; any
        excess beyond the open quantity flips the slot (``position.rs:545-551``) — reachable in
        one-way mode, not in the hedge-mode ``(symbol, side)`` keying this module defaults to, kept
        because the arithmetic must still be correct if this class is ever used that way.
        """
        self.lots.append(lot)
        opens = lot.side.lower() == ("buy" if self.side is PositionSide.LONG else "sell")

        if opens:
            total_cost = self.entry_price * self.quantity + lot.price * lot.quantity
            new_qty = self.quantity + lot.quantity
            self.entry_price = total_cost / new_qty
            self.quantity = new_qty
            if self.ts_opened is None:
                self.ts_opened = lot.ts_filled
        else:
            closing_qty = min(lot.quantity, self.quantity)
            direction = Decimal("1") if self.side is PositionSide.LONG else Decimal("-1")
            self.realized_pnl += direction * (lot.price - self.entry_price) * closing_qty
            self.quantity -= closing_qty
            excess = lot.quantity - closing_qty
            if self.quantity == 0:
                self.entry_price = Decimal("0")
            if excess > 0:
                # Flip: the excess opens fresh, in the opposite direction, at the fill price.
                self.side = (
                    PositionSide.SHORT if self.side is PositionSide.LONG else PositionSide.LONG
                )
                self.entry_price = lot.price
                self.quantity = excess

        self.ts_last = lot.ts_filled

    def unrealized_pnl(self, mark_price: Decimal) -> Decimal:
        """Mark-to-market PnL against a live price. ``0`` when flat, never ``None`` — a flat
        position genuinely carries no unrealized risk, which is a fact, not a missing one."""
        if self.is_flat:
            return Decimal("0")
        direction = Decimal("1") if self.side is PositionSide.LONG else Decimal("-1")
        return direction * (mark_price - self.entry_price) * self.quantity

    def realized_pnl_since(self, cutoff: datetime) -> Decimal:
        """Realized PnL from fills at or after ``cutoff`` only — the windowed figure
        :attr:`realized_pnl` deliberately is not (that field is a lifetime running total).

        **A full replay of this position's own lots, not a filtered sum.** ``entry_price`` is a
        mutable weighted average (:meth:`apply_fill`'s own docstring), and no historical snapshot
        of what it was at each past reducing fill survives anywhere except by replaying the fills
        that produced it — filtering ``self.lots`` by timestamp and summing would use today's
        entry_price against yesterday's fills, which is simply wrong.

        The replay is seeded with an arbitrary starting side (``LONG``) and self-corrects: when
        the temporary position is still flat, :meth:`apply_fill`'s own reversal-from-zero branch
        (``closing_qty = min(lot.quantity, 0) == 0``, a harmless no-op, followed by the "excess"
        branch) flips it to whatever side the first real lot actually represents — verified by
        test, not assumed, because a wrong seed silently producing a wrong first trade would be
        exactly the class of defect this project exists to catch.

        Built as the ARGUS-side counterpart to
        `eval.freqtrade_baseline.freqtrade_low_profit_pairs`: a windowed per-symbol realized PnL
        is exactly what a live ``per_symbol_underperformance`` Constitution gate needs, and
        nothing before this method could compute one.
        """
        replay = Position(symbol=self.symbol, side=PositionSide.LONG)
        windowed = Decimal("0")
        for lot in self.lots:
            before = replay.realized_pnl
            replay.apply_fill(lot)
            if lot.ts_filled >= cutoff:
                windowed += replay.realized_pnl - before
        return windowed

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": str(self.side),
            "quantity": str(self.quantity),
            "entry_price": str(self.entry_price),
            "realized_pnl": str(self.realized_pnl),
            "lots": [lot.as_dict() for lot in self.lots],
            "ts_opened": self.ts_opened.isoformat() if self.ts_opened else None,
            "ts_last": self.ts_last.isoformat() if self.ts_last else None,
        }


@dataclass(frozen=True, slots=True)
class AccountBalance:
    """One currency's balance. ``locked`` is ARGUS's own running reservation total, not a venue
    figure — see :class:`OpenOrderReservation` for why it is computed locally."""

    currency: str
    total: Decimal
    locked: Decimal = Decimal("0")

    @property
    def free(self) -> Decimal:
        return self.total - self.locked

    def as_dict(self) -> dict[str, Any]:
        return {
            "currency": self.currency, "total": str(self.total),
            "locked": str(self.locked), "free": str(self.free),
        }


@dataclass(frozen=True, slots=True)
class OpenOrderReservation:
    """Balance an open order holds against, computed locally on every order-state change.

    Computed here rather than trusted from the venue, unlike :class:`VenueMarginSnapshot` — this is
    a *pre-trade* estimate of what an order would cost if filled, which ARGUS must know before the
    venue has said anything at all (the whole point is to check headroom before submitting), and
    getting this conservative-and-local number wrong costs at most a rejected order. It is not the
    liquidation-relevant figure — that one comes from the venue, on purpose (see module docstring).
    """

    client_order_id: str
    symbol: str
    side: str
    reserved: Decimal
    """Notional the order would consume if fully filled at its limit/estimate price."""

    def as_dict(self) -> dict[str, Any]:
        return {
            "client_order_id": self.client_order_id, "symbol": self.symbol,
            "side": self.side, "reserved": str(self.reserved),
        }


def order_reservation(
    *, client_order_id: str, symbol: str, side: str, remaining_quantity: Decimal, price: Decimal
) -> OpenOrderReservation:
    """Notional an unfilled order would consume — the local pre-trade estimate.

    Deliberately simple: quantity times price, in quote currency, for both sides. Bitget futures
    are cash-margined in the quote currency (``marginCoin``, verified live as USDT on this
    account), unlike a spot buy/sell pair that locks different currencies per side
    (Nautilus's ``base_calculate_balance_locked``, ``accounts/base.rs:261-292``, rejected here for
    that reason — it models spot base/quote asymmetry this venue's futures product does not have).
    """
    if remaining_quantity <= 0:
        return OpenOrderReservation(
            client_order_id=client_order_id, symbol=symbol, side=side, reserved=Decimal("0")
        )
    return OpenOrderReservation(
        client_order_id=client_order_id, symbol=symbol, side=side,
        reserved=remaining_quantity * price,
    )


@dataclass(frozen=True, slots=True)
class VenueMarginSnapshot:
    """What Bitget's own margin engine reports, timestamped. Never re-derived — see module
    docstring for why recomputing this locally was considered and rejected."""

    account_equity: Decimal
    unrealised_pnl: Decimal
    imr: Decimal
    """Initial margin requirement, as reported."""

    mmr: Decimal
    """Maintenance margin requirement, as reported."""

    mgn_ratio: Decimal
    """Account-level margin ratio, as reported.

    **Direction and conceptual threshold PRIMARY-VERIFIED 2026-09-15**, fetched directly from
    Bitget's own liquidation page (``bitget.com/futures/introduction/liquidation-summarize``, read
    verbatim): *"maintenance margin ratio = 100%"* is the liquidation trigger in cross-margin mode,
    escalating to a forced close at 160% for accounts with open orders at Tier 2+. **Higher is
    worse.**

    **Numeric scale (fraction vs percentage) — STRONG EVIDENCE, not a direct proof for this exact
    field, checked rather than guessed.** `ccxt` (a well-maintained open-source exchange library,
    cloned locally, read directly) shows Bitget's V2 Mix **position-level** ``marginRatio`` as a
    real, non-zero worked example straight from its own source comment
    (``ccxt/ts/src/bitget.ts:8063``): ``"marginRatio": "0.029599540355"`` alongside
    ``"keepMarginRate": "0.004"`` — both plainly fractions (2.96% and 0.4%; a percentage-scaled
    API would read ``"2.9599540355"`` and ``"0.4"`` respectively, not these). `ccxt`'s own
    ``parsePosition`` passes the value through unchanged (``bitget.ts:8488``, no ``*100``/``/100``),
    confirming this is Bitget's raw wire scale, not a `ccxt` normalisation artefact.

    That evidence is for the **position-level** field on a different (V2 Mix, not V3 UTA) endpoint
    family, so it is not direct proof for the **account-level** ``mgnRatio``/``positionMgnRatio``
    this class actually carries — `ccxt`'s UTA balance parser shows only a zero worked example for
    those and does not map them at all (checked: no other reference in the file). Treated here as
    strong analogy — same field-naming family, sitting beside ``mmr``/``imr`` which are also
    fraction-styled — **not** as the same rigor as a direct non-zero example.
    `agents.desk.ConstitutionPolicy`'s ``margin_usage`` gate is built on this evidence and states
    the same caveat rather than repeating it as settled fact.
    """

    position_mgn_ratio: Decimal
    """Same evidence and same caveat as :attr:`mgn_ratio`, at position granularity."""

    leverage: Decimal
    fetched_at: datetime

    def as_dict(self) -> dict[str, Any]:
        return {
            "account_equity": str(self.account_equity),
            "unrealised_pnl": str(self.unrealised_pnl),
            "imr": str(self.imr), "mmr": str(self.mmr),
            "mgn_ratio": str(self.mgn_ratio), "position_mgn_ratio": str(self.position_mgn_ratio),
            "leverage": str(self.leverage), "fetched_at": self.fetched_at.isoformat(),
        }


def parse_venue_margin_snapshot(
    payload: dict[str, Any], *, fetched_at: datetime
) -> VenueMarginSnapshot:
    """Build a snapshot from ``account_overview``'s ``assets`` section.

    Field names verified against a live call (``mcp__bitget-agentic__account_overview``, category
    USDT-FUTURES, 2026-09-15): ``accountEquity``, ``unrealisedPnl``, ``imr``, ``mmr``, ``mgnRatio``,
    ``positionMgnRatio``, ``leverage`` — all present, all ``"0"`` on this unfunded account. Raises
    rather than defaulting on a missing key: a margin snapshot silently built from absent fields is
    the exact defect this module exists to prevent elsewhere.
    """
    try:
        return VenueMarginSnapshot(
            account_equity=Decimal(payload["accountEquity"]),
            unrealised_pnl=Decimal(payload["unrealisedPnl"]),
            imr=Decimal(payload["imr"]),
            mmr=Decimal(payload["mmr"]),
            mgn_ratio=Decimal(payload["mgnRatio"]),
            position_mgn_ratio=Decimal(payload["positionMgnRatio"]),
            leverage=Decimal(payload["leverage"]),
            fetched_at=fetched_at,
        )
    except KeyError as exc:
        raise PositionError(
            f"account_overview payload is missing {exc}; refusing to build a margin snapshot "
            f"from an incomplete venue response rather than defaulting a safety-relevant field"
        ) from exc


class ReconciliationStatus(StrEnum):
    MATCHED = "matched"
    WITHIN_TOLERANCE = "within_tolerance"
    MANUAL_REVIEW = "manual_review"


@dataclass(frozen=True, slots=True)
class ReconciliationDiff:
    """Local position state checked against what the venue reports. Never auto-corrected past
    tolerance — see module docstring on why synthetic fill backfill is rejected."""

    symbol: str
    side: PositionSide
    local_quantity: Decimal
    venue_quantity: Decimal
    tolerance: Decimal
    checked_at: datetime

    @property
    def diff(self) -> Decimal:
        return abs(self.local_quantity - self.venue_quantity)

    @property
    def status(self) -> ReconciliationStatus:
        if self.diff == 0:
            return ReconciliationStatus.MATCHED
        # Reached only when diff != 0, so at least one side is nonzero and this division is safe.
        # A local-is-zero-but-venue-isn't mismatch must NOT default to "within tolerance" — that
        # would silently hide a position the venue holds that nothing here is tracking at all.
        largest = max(self.local_quantity, self.venue_quantity)
        if self.diff / largest <= self.tolerance:
            return ReconciliationStatus.WITHIN_TOLERANCE
        return ReconciliationStatus.MANUAL_REVIEW

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol, "side": str(self.side),
            "local_quantity": str(self.local_quantity), "venue_quantity": str(self.venue_quantity),
            "diff": str(self.diff), "status": str(self.status),
            "checked_at": self.checked_at.isoformat(),
        }


DEFAULT_RECONCILIATION_TOLERANCE = Decimal("0.0001")
"""0.01% — the same tolerance Nautilus's reconciliation engine uses
(``reconciliation/positions.rs:39``), kept because it is a reasonable default and there was no
venue-specific reason found to pick a different one."""


def reconcile_position(
    *, symbol: str, side: PositionSide, local_quantity: Decimal, venue_quantity: Decimal,
    checked_at: datetime, tolerance: Decimal = DEFAULT_RECONCILIATION_TOLERANCE,
) -> ReconciliationDiff:
    return ReconciliationDiff(
        symbol=symbol, side=side, local_quantity=local_quantity, venue_quantity=venue_quantity,
        tolerance=tolerance, checked_at=checked_at,
    )


@dataclass(frozen=True, slots=True)
class HedgeLink:
    """A reported relationship between two positions. Informational only — see module docstring:
    Nautilus does not model this either, and for the same reason: there is no single definition of
    "hedge" a generic model could enforce without coupling strategy logic to position tracking."""

    source: tuple[str, PositionSide]
    target: tuple[str, PositionSide]
    relationship: str
    hedge_ratio: Decimal | None
    ts_created: datetime

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": {"symbol": self.source[0], "side": str(self.source[1])},
            "target": {"symbol": self.target[0], "side": str(self.target[1])},
            "relationship": self.relationship,
            "hedge_ratio": None if self.hedge_ratio is None else str(self.hedge_ratio),
            "ts_created": self.ts_created.isoformat(),
        }


class Book:
    """The real portfolio. Positions keyed by ``(symbol, side)``, balances by currency.

    Everything here is built from fills and venue reads that already happened — there is no method
    that invents a number. **Not yet wired anywhere**: nothing calls :meth:`apply_fill` from
    `execution.orders.OrderBook`'s fill events, and `desk.portfolio.assess` still takes
    caller-supplied weights rather than :meth:`weights`/:meth:`gross_weights` below. This class
    exists and is tested; the desk does not consume it yet — see `Activity/PROGRESS.md`.
    """

    def __init__(self) -> None:
        self.positions: dict[tuple[str, PositionSide], Position] = {}
        self.balances: dict[str, AccountBalance] = {}
        self.reservations: dict[str, OpenOrderReservation] = {}
        self.margin: VenueMarginSnapshot | None = None
        self.hedge_links: list[HedgeLink] = []

    def apply_fill(self, *, symbol: str, side: PositionSide, lot: Lot) -> Position:
        key = (symbol, side)
        position = self.positions.get(key)
        if position is None:
            position = Position(symbol=symbol, side=side)
            self.positions[key] = position
        position.apply_fill(lot)
        return position

    def reserve(self, reservation: OpenOrderReservation) -> None:
        self.reservations[reservation.client_order_id] = reservation

    def release(self, client_order_id: str) -> None:
        self.reservations.pop(client_order_id, None)

    def total_reserved(self) -> Decimal:
        """Sum of every open reservation, in quote currency (USDT on this venue — verified in
        `order_reservation`'s docstring). No per-currency split: :class:`OpenOrderReservation`
        does not carry one, because every reservation on this venue is already quote-denominated."""
        return sum((r.reserved for r in self.reservations.values()), Decimal("0"))

    def set_balance(self, balance: AccountBalance) -> None:
        self.balances[balance.currency] = balance

    def set_margin(self, snapshot: VenueMarginSnapshot) -> None:
        self.margin = snapshot

    def open_positions(self) -> list[Position]:
        return [p for p in self.positions.values() if not p.is_flat]

    def total_open_positions(self) -> int:
        return len(self.open_positions())

    def total_unrealized_pnl(self, marks: dict[str, Decimal]) -> Decimal:
        """Sum of unrealized PnL across open positions, given a mark price per symbol.

        Missing marks are excluded and the caller is not told they were — this mirrors
        `desk.portfolio`'s ``None``-on-insufficient-data doctrine but at the sum level: a partial
        sum silently presented as total would misstate risk, so a caller who needs to know
        coverage should check ``open_positions()`` against ``marks.keys()`` itself.
        """
        total = Decimal("0")
        for position in self.open_positions():
            mark = marks.get(position.symbol)
            if mark is not None:
                total += position.unrealized_pnl(mark)
        return total

    def symbol_realized_pnl_since(self, symbol: str, cutoff: datetime) -> Decimal:
        """Windowed realized PnL for one symbol, summed across both hedge-mode legs.

        Deliberately reads **all** of ``self.positions`` for this symbol, not
        :meth:`open_positions` — a symbol that lost money and was fully closed out is exactly the
        case a per-symbol underperformance gate (`agents.desk.ConstitutionPolicy`'s
        ``per_symbol_underperformance``, the ARGUS-native counterpart to freqtrade's
        ``LowProfitPairs``, `eval.freqtrade_baseline.freqtrade_low_profit_pairs`) needs to see;
        excluding flat positions would hide the very losses the gate exists to catch. Hedge mode
        keys a long and a short leg of the same symbol as two separate :class:`Position` objects
        (`Book.__init__`'s ``(symbol, PositionSide)`` keying) — both legs' realized PnL is real
        money and both are summed, matching :meth:`total_gross_notional`'s no-netting convention
        rather than :meth:`total_signed_notional`'s.
        """
        return sum(
            (
                position.realized_pnl_since(cutoff)
                for position in self.positions.values()
                if position.symbol == symbol
            ),
            Decimal("0"),
        )

    def total_gross_notional(self) -> Decimal:
        """Sum of ``quantity * entry_price`` across every open position, direction ignored.

        The figure `agents.desk.ConstitutionPolicy`'s ``gross_exposure`` gate compares against
        a flat cap — unlike :meth:`weights`, this deliberately does not net a hedge-mode long and
        short against each other, for the same reason :meth:`gross_weights` does not: both legs
        carry real margin and real liquidation risk regardless of whether they cancel in net terms.
        """
        return sum(
            (p.quantity * p.entry_price for p in self.open_positions()), Decimal("0")
        )

    def total_signed_notional(self) -> Decimal:
        """Net directional notional across the whole book — long positive, short negative,
        **netted within a symbol** (unlike :meth:`total_gross_notional`, which never nets).

        The whole-book counterpart to :meth:`weights`'s per-symbol netting: this is what a "how
        net-long or net-short is the desk overall" cap reads (Foundation 5's ``signed_exposure``),
        distinct from gross (margin/concentration) and distinct from a per-symbol weight
        (beta/factor exposure).
        """
        total = Decimal("0")
        for position in self.open_positions():
            signed = position.quantity * position.entry_price
            total += signed if position.side is PositionSide.LONG else -signed
        return total

    def weights(self, equity: Decimal) -> dict[str, float]:
        """Net signed weight per symbol — the shape `desk.portfolio.assess`/`decompose` consume.

        Long and short legs in the same symbol (hedge mode, verified live — see module docstring)
        **net against each other here**: a symbol held both long and short in equal size returns a
        weight of ``0``. That is correct for this function's purpose — beta and factor exposure are
        properties of *net* market exposure — and wrong for margin or concentration, which care
        about gross size regardless of direction. Use :meth:`gross_weights` for those. Returning
        both as separate methods rather than one blended number is deliberate: netting them into a
        single figure is exactly the kind of silent conflation Foundation 5's safety contract lists
        signed and gross exposure as two separate dimensions to prevent.

        ``0.0`` for every symbol when ``equity <= 0`` rather than dividing by it — an non-positive
        equity is a fact worth surfacing elsewhere, not a reason to raise or to return NaN-shaped
        weights here.
        """
        if equity <= 0:
            return {position.symbol: 0.0 for position in self.open_positions()}
        net: dict[str, Decimal] = {}
        for (symbol, side), position in self.positions.items():
            if position.is_flat:
                continue
            signed = position.quantity * position.entry_price
            if side is PositionSide.SHORT:
                signed = -signed
            net[symbol] = net.get(symbol, Decimal("0")) + signed
        return {symbol: float(notional / equity) for symbol, notional in net.items()}

    def gross_weights(self, equity: Decimal) -> dict[str, float]:
        """Absolute-value weight per ``(symbol, side)`` slot — never nets a hedge against itself.

        Keyed by ``f"{symbol}:{side}"`` rather than by symbol alone, because two entries can exist
        for one symbol in hedge mode and collapsing them would silently drop the gross size
        :meth:`weights` deliberately does not carry.
        """
        if equity <= 0:
            return {}
        out: dict[str, float] = {}
        for (symbol, side), position in self.positions.items():
            if position.is_flat:
                continue
            notional = position.quantity * position.entry_price
            out[f"{symbol}:{side}"] = float(notional / equity)
        return out

    def link_hedge(self, link: HedgeLink) -> None:
        self.hedge_links.append(link)

    def hedge_cluster(self, symbol: str, side: PositionSide) -> set[tuple[str, PositionSide]]:
        """Every position transitively linked to this one.

        Originally built for reporting only; `agents.desk.ConstitutionPolicy.rule`'s
        ``hedge_integrity`` gate (added 2026-09-15) is now a real consumer — it refuses an
        automatic reduction of a position while this cluster still contains another open one.
        The links themselves stay informational (see :class:`HedgeLink`'s docstring: no universal
        definition of "hedge" a generic model could enforce), but *whether a cluster member is
        still open* is a plain fact this method already computed correctly for the report, and
        reusing it is not the same as the model enforcing what a hedge means.
        """
        seen: set[tuple[str, PositionSide]] = {(symbol, side)}
        frontier = [(symbol, side)]
        while frontier:
            current = frontier.pop()
            for link in self.hedge_links:
                for a, b in ((link.source, link.target), (link.target, link.source)):
                    if a == current and b not in seen:
                        seen.add(b)
                        frontier.append(b)
        return seen

    def as_dict(self) -> dict[str, Any]:
        return {
            "positions": [p.as_dict() for p in self.positions.values()],
            "balances": [b.as_dict() for b in self.balances.values()],
            "reservations": [r.as_dict() for r in self.reservations.values()],
            "margin": None if self.margin is None else self.margin.as_dict(),
            "hedge_links": [h.as_dict() for h in self.hedge_links],
        }


__all__ = [
    "DEFAULT_RECONCILIATION_TOLERANCE",
    "AccountBalance",
    "Book",
    "HedgeLink",
    "Lot",
    "OpenOrderReservation",
    "Position",
    "PositionError",
    "PositionSide",
    "ReconciliationDiff",
    "ReconciliationStatus",
    "VenueMarginSnapshot",
    "order_reservation",
    "parse_venue_margin_snapshot",
    "reconcile_position",
]
