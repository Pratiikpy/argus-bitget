"""Pre-trade validation — the checks a venue will enforce anyway, done before we ask.

Three independent sources named the same gap. Nautilus runs a `RiskEngine` that denies an order
before it leaves the process, with a named reason, on price precision, quantity precision, notional
bounds and submission rate. Hummingbot's `BudgetChecker` locks collateral for a hypothetical order
and refuses when the balance cannot cover it. The author's own Nomos hit three separate 10x sizing
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

**Self-check (2026-09-25).** :func:`would_pass`, :meth:`Guard.self_check` and
:func:`self_check_request` answer "would this order pass?" with the same function that enforces
(:func:`_rule`) and return pass/fail only, after mle-bench's ``/validate`` endpoint. The identity
``would_pass(x) == validate(x).allowed`` is tested over a swept order space.

**Trace (2026-09-25, S20).** Every :class:`Ruling` now carries a :class:`PermissionCheckTrace`: the
whole gate stack in order, each with the input it read, what it did (pass, adjust, deny, skip, not
reached) and why. Taken from letta-code (Apache-2.0, ``src/permissions/checker.ts:202-229`` and
``src/permissions/types.ts:32-52``), whose permission checker pushes one ``{stage, matched,
pattern, message}`` event per stage it evaluates. Changed: letta-code records only the stages it
reached, and gates the trace behind an environment flag (``checker.ts:74-86``). Here the trace is
always attached, and the gates after a refusal are listed as ``not_reached`` rather than omitted, so
a reader sees the whole stack and what the refusal pre-empted. The trace is also what
:mod:`argus.risk.certify` checks, gate by gate, against the written specification.

**Three defects the certifier's specification exposed, fixed the same day.** Each is a violation
of this module's own stated contract; `eval/risk_shadow.py` lists every ruling the fixes change:

* **A quantity could be rounded up.** :func:`quantise_down` divided in the default 28-digit
  context, so ``0.0099999999999999999999999999999`` divided by the ``0.01`` step rounded to
  ``1.000…`` before truncation and came back as ``0.01`` — an order the guard *enlarged*, then
  allowed. The same context rounded products: a notional of ``4.99999…95`` became ``5`` and
  cleared a ``5`` USDT floor it does not meet. Arithmetic now runs in a context wide enough to be
  exact for the operands in hand.
* **A positive limit price below one tick was allowed at zero.** ``0.005`` steps down to ``0.00``
  on a ``0.01`` tick; with no reference price the band was skipped and the order passed with a
  limit price of zero. The stepped price must now stay positive.
* **An unknown side was allowed.** Bitget's order endpoint takes ``side`` from ``["buy", "sell"]``
  (``agent-sdk/src/tools/field-schemas.ts:57``); anything else was ruled against the *sell* band
  and passed. It is now refused by name, :attr:`Denial.SIDE_UNKNOWN`.

    python -m argus.execution.guard --symbol NVDAUSDT
    python -m argus.execution.guard --symbol NVDAUSDT --quantity 0.5 --self-check
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from dataclasses import dataclass, field, replace
from decimal import ROUND_DOWN, Decimal, getcontext, localcontext
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
    NOT_A_NUMBER = "not_a_number"
    """A quantity, price, reference or balance that is NaN or infinite.

    Added 2026-09-25, found by the self-check sweep (`eval/guard_selfcheck.py`): 13,800 of
    190,944 swept orders made the enforcing path *raise* ``decimal.InvalidOperation`` — a NaN
    compared with ``<=`` — instead of returning a ruling. Failing by exception is closed only if
    every caller catches that exception; a named denial is closed for all of them.

    The same sweep found the worse half: an **infinite quantity was allowed**. ``Infinity``
    stepped down is still ``Infinity``, which exceeds the ceiling, which the guard then shrinks to
    the venue maximum — so "buy infinity" became a 52,000-contract order ruled *allowed*. 1,935
    swept orders passed that way before this denial; 0 after."""

    SIDE_UNKNOWN = "side_unknown"
    """A side that is neither ``buy`` nor ``sell`` (case-insensitive).

    Added 2026-09-25 (S20). Bitget's order endpoint accepts exactly those two values
    (``agent-sdk/src/tools/field-schemas.ts:57``), so any other side is an order the venue will
    reject. Before this denial the guard ruled such an order against the *sell* band and allowed
    it, because the band lookup read ``side == "buy"`` and treated everything else as a sale."""

    RISK_MODE = "risk_mode"
    """Refused by the desk's active risk mode (:mod:`argus.risk.modes`), before any venue rule.

    Not raised by :func:`validate` itself: the mode gate sits in front of the venue gates and
    composes its refusal into the same :class:`Ruling` shape so one trace covers the whole path."""

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

    arithmetic_width: int = field(init=False, repr=False, compare=False)
    """The digit width of this instrument's venue numbers, summed once: its share of
    :func:`exact_precision`, cached because it is the same for every order on the instrument."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "arithmetic_width", sum(_digits_width(value) for value in (
            self.price_multiplier, self.quantity_multiplier, self.min_order_qty,
            self.max_order_qty, self.min_order_amount, self.buy_limit_ratio,
            self.sell_limit_ratio,
        )))

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


_PRECISION_MARGIN = 12
"""Guard digits beyond the operands' own width, so a non-terminating quotient cannot round across an
integer before it is truncated."""


def _digits_width(value: Decimal | None) -> int:
    """Coefficient digits plus places of exponent for one finite operand; 0 for ``None``, NaN or
    infinity, which are refused before any arithmetic."""
    if value is None or not value.is_finite():
        return 0
    _sign, digits, exponent = value.as_tuple()
    return len(digits) + (abs(exponent) if isinstance(exponent, int) else 0)


def exact_precision(*values: Decimal | None) -> int:
    """A context precision wide enough that arithmetic on these operands does not round.

    Decimal rounds every result to the context's precision, 28 significant digits by default. The
    guard multiplies prices by ratios and quantities by prices, and divides sizes by steps; with
    operands of more than a handful of digits a 28-digit result is rounded, and a rounded product
    can land on the wrong side of a venue limit. Every coefficient digit and every place of
    exponent is counted, which bounds the width of any product of two operands and of the integer
    part of any quotient, and :data:`_PRECISION_MARGIN` covers the fractional digits truncation
    needs to see.
    """
    return max(getcontext().prec, sum(_digits_width(v) for v in values) + _PRECISION_MARGIN)


def quantise_down(value: Decimal, step: Decimal) -> Decimal:
    """Round to a multiple of ``step``, always downward.

    Downward, never to nearest. Rounding to nearest can round *up*, and a size that rounds up can
    cross the ceiling that was checked a moment earlier — which is how a validated order becomes an
    oversized one between the check and the wire.

    **Computed in an exact context (2026-09-25).** In the default 28-digit context the quotient
    itself was rounded before it was truncated, so ``0.0099999999999999999999999999999 / 0.01``
    became ``1.000…`` and the result was ``0.01`` — larger than the value it was given. The
    context is now widened to :func:`exact_precision` of the two operands, and the result is checked
    against the one property this function exists to guarantee: its magnitude never exceeds the
    input's. The representation of every result that was already correct is unchanged, because the
    same operation runs; only its precision moved.
    """
    if step <= 0:
        raise GuardError(f"step must be positive, got {step}")
    with localcontext() as ctx:
        ctx.prec = exact_precision(value, step)
        stepped = (value / step).to_integral_value(rounding=ROUND_DOWN) * step
    if value.is_finite() and abs(stepped) > abs(value):
        raise GuardError(
            f"stepping {value} by {step} produced {stepped}, larger than its input; refusing a "
            f"rounding that would enlarge an order"
        )
    return stepped


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

    def would_exceed(self, *, now: float, prune: bool = True) -> bool:
        """Whether one more submission at ``now`` would cross the cap.

        ``prune=False`` answers the same question without touching the window, for
        :func:`would_pass`: a self-check that pruned would still be a *read*, but one whose
        answer to a later call at an earlier ``now`` could differ from the enforcing path's.
        Counting instead of pruning keeps the question a pure function of the stamps.
        """
        if prune:
            self._prune(now)
            return len(self._stamps) >= self.max_orders
        return self.count(now=now) >= self.max_orders

    def count(self, *, now: float) -> int:
        """Submissions inside the window ending at ``now``, without touching the window.

        What the trace and the rate denial report. :attr:`in_window` is the raw length, which on the
        self-check path (never pruned) can still include stamps that have aged out.
        """
        cutoff = now - self.window_seconds
        return sum(1 for stamp in self._stamps if stamp > cutoff)

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


class GateVerdict(StrEnum):
    """What one gate did with one order — the vocabulary of :class:`PermissionCheckTrace`."""

    PASS = "pass"
    ADJUST = "adjust"
    """Passed after the guard shrank the order to the venue's step or ceiling."""

    DENY = "deny"
    SKIP = "skip"
    """Not applicable to this order (no limit price, no reference, no limiter), and said so."""

    NOT_REACHED = "not_reached"
    """An earlier gate refused, so this one was never evaluated. Listed rather than omitted, so the
    trace always names the whole stack and a reader can see what the refusal pre-empted."""


RULE_GATES: tuple[str, ...] = (
    "finite_inputs", "side_known", "instrument_online", "rate_window", "price_positive",
    "price_step", "price_band", "quantity_step", "quantity_ceiling", "quantity_floor",
    "notional_floor", "balance",
)
"""The gates :func:`validate` runs, in the order it runs them. One trace event per gate, always."""

GUARD_GATES: tuple[str, ...] = ("instrument_known", *RULE_GATES)
"""The gates :meth:`Guard.check` runs: the instrument lookup first, then :data:`RULE_GATES`."""


@dataclass(frozen=True, slots=True)
class TraceEvent:
    """One gate's ruling on one order: which gate, what it read, what it did, and why."""

    gate: str
    input: str
    verdict: GateVerdict
    reason: str

    def as_dict(self) -> dict[str, str]:
        return {"gate": self.gate, "input": self.input, "verdict": str(self.verdict),
                "reason": self.reason}

    def render(self) -> str:
        shown = f" [{self.input}]" if self.input else ""
        return f"{self.gate}: {self.verdict}{shown} — {self.reason}"


@dataclass(frozen=True, slots=True)
class PermissionCheckTrace:
    """Every gate an order met, in order — the staged record behind a :class:`Ruling`.

    After letta-code's ``PermissionCheckTrace`` (Apache-2.0, ``src/permissions/types.ts:45-52``),
    which carries the ordered ``events`` a permission check produced. A ruling's one-line reason
    says what was decided; this says how, gate by gate, so "why was this exact size chosen" has an
    auditable answer rather than a sentence.
    """

    events: tuple[TraceEvent, ...] = ()

    @property
    def gates(self) -> tuple[str, ...]:
        return tuple(event.gate for event in self.events)

    @property
    def deciding(self) -> TraceEvent | None:
        """The event that refused the order, or ``None`` when every gate let it through."""
        for event in self.events:
            if event.verdict is GateVerdict.DENY:
                return event
        return None

    def prepend(self, event: TraceEvent) -> PermissionCheckTrace:
        """The same trace with a gate that ran in front of it, such as the risk-mode gate."""
        return PermissionCheckTrace((event, *self.events))

    def as_dict(self) -> list[dict[str, str]]:
        return [event.as_dict() for event in self.events]

    def render(self) -> str:
        return "\n".join(
            f"  {index:>2}. {event.render()}" for index, event in enumerate(self.events, start=1)
        )


def not_reached(gates: tuple[str, ...], refused_by: str) -> tuple[TraceEvent, ...]:
    """The events for gates an earlier refusal pre-empted."""
    return tuple(
        TraceEvent(gate, "", GateVerdict.NOT_REACHED, f"not evaluated: {refused_by} refused first")
        for gate in gates
    )


@dataclass(frozen=True, slots=True)
class Ruling:
    """What the guard decided about one order."""

    denial: Denial
    reason: str
    quantity: Decimal
    price: Decimal | None
    adjusted: bool
    """Whether the guard changed the order rather than only judging it."""

    trace: PermissionCheckTrace = field(default_factory=PermissionCheckTrace)
    """Every gate the order met, in order. See :class:`PermissionCheckTrace`."""

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

    def explain(self) -> str:
        """The one-line ruling followed by its whole gate trace."""
        return f"{self.render()}\n{self.trace.render()}" if self.trace.events else self.render()

    def as_dict(self) -> dict[str, Any]:
        return {
            "denial": str(self.denial), "allowed": self.allowed, "reason": self.reason,
            "quantity": str(self.quantity),
            "price": None if self.price is None else str(self.price),
            "adjusted": self.adjusted,
            "trace": self.trace.as_dict(),
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
    """Enforce: rule on one order and, if it is allowed, spend a slot in the rate window.

    The ruling itself is :func:`_rule`, shared word for word with :func:`would_pass`. See there.
    """
    return _rule(
        instrument, quantity=quantity, price=price, reference_price=reference_price, side=side,
        available_balance=available_balance, limiter=limiter, now=now, record=True,
    )


def would_pass(
    instrument: Instrument,
    *,
    quantity: Decimal,
    price: Decimal | None = None,
    reference_price: Decimal | None = None,
    side: str = "buy",
    available_balance: Decimal | None = None,
    limiter: RateLimiter | None = None,
    now: float | None = None,
) -> bool:
    """Self-check: would :func:`validate` allow this order right now? Pass or fail, nothing more.

    **Taken from mle-bench** (MIT; ``environment/grading_server.py:15-19`` and
    ``mlebench/grade.py:98-124``). Its ``/validate`` endpoint answers "is this submission valid?"
    by running the competition's *real* grader, ``competition.grader.grade_fn``, and returning
    only whether it raised — never the score. Two properties made that design worth copying:

    * **The check cannot drift from the enforcement, because it is the enforcement.** A separate
      "pre-check" re-implementation of the venue rules would be a second copy of eight rules that
      nobody keeps in step with the first; the day one gains a rule the other lacks, the check
      starts approving orders the guard refuses. Here both paths call :func:`_rule`, and the
      property ``would_pass(x) == validate(x).allowed`` is tested over a swept order space
      (``eval/guard_selfcheck.py``, ``tests/test_guard_selfcheck.py``).
    * **It answers only pass/fail.** mle-bench withholds the score so an agent cannot fit to the
      grader. The analogue here is the adjusted size: :func:`validate` may shrink an order to the
      venue step and ceiling, and a self-check that returned that size would turn a yes/no probe
      into a sizing oracle the caller could lean on instead of deciding its own size.

    **What differs from mle-bench, deliberately.** Its check is side-effect free because grading
    is. Ours has one side effect to suppress: enforcement records a submission in the rate window.
    A self-check that recorded would consume the budget it is asking about, so ``record=False``
    reaches the limiter as a non-mutating count (:meth:`RateLimiter.would_exceed` with
    ``prune=False``). Everything else — order of checks, rounding, bands, minima — is shared.

    Rejected: mle-bench's HTTP-500 path for a grader that raises. Here a check that cannot be
    evaluated is a ``False``, never an error the caller might read as "no objection".
    """
    try:
        return _rule(
            instrument, quantity=quantity, price=price, reference_price=reference_price,
            side=side, available_balance=available_balance, limiter=limiter, now=now,
            record=False,
        ).allowed
    except (GuardError, ArithmeticError):
        # The enforcing path raises on the same input, so it would not have allowed it either.
        return False


_SIDES = frozenset({"buy", "sell"})
"""The two sides Bitget's order endpoint accepts (``agent-sdk/src/tools/field-schemas.ts:57``)."""


class _Trail:
    """The trace under construction: one event per gate, recorded in :data:`RULE_GATES` order."""

    __slots__ = ("events",)

    def __init__(self) -> None:
        self.events: list[TraceEvent] = []

    def note(self, gate: str, verdict: GateVerdict, reason: str, given: str = "") -> None:
        self.events.append(TraceEvent(gate, given, verdict, reason))

    def refuse(
        self, gate: str, denial: Denial, reason: str, given: str, *, price: Decimal | None,
        adjusted: bool,
    ) -> Ruling:
        self.note(gate, GateVerdict.DENY, reason, given)
        return Ruling(denial, reason, _ZERO, price, adjusted, self.seal(refused_by=gate))

    def seal(self, *, refused_by: str | None = None) -> PermissionCheckTrace:
        if refused_by is None:
            return PermissionCheckTrace(tuple(self.events))
        rest = RULE_GATES[len(self.events):]
        return PermissionCheckTrace((*self.events, *not_reached(rest, refused_by)))


def _rule(
    instrument: Instrument,
    *,
    quantity: Decimal,
    price: Decimal | None,
    reference_price: Decimal | None,
    side: str,
    available_balance: Decimal | None,
    limiter: RateLimiter | None,
    now: float | None,
    record: bool,
) -> Ruling:
    """Check one order against the venue's own rules, shrinking it or refusing it.

    Ordered so the cheapest and most fundamental checks come first, and so a denial names the
    reason a person would want: an offline instrument is reported as offline rather than as a
    precision failure further down.

    The guard may **reduce** a quantity to the venue's step and ceiling. It never raises one — not
    even to reach ``min_order_qty``, because increasing an order to satisfy a minimum would be the
    guard authoring size, and a validation layer that can enlarge a position is a strategy.

    Every gate in :data:`RULE_GATES` writes exactly one event to the ruling's trace, in that order,
    including the gates a refusal pre-empted (``not_reached``) and the ones that do not apply to
    this order (``skip``). The trace records; it never decides — each gate's branch is the same
    comparison it was before the trace existed, and `eval/risk_shadow.py` replays the recorded
    order stream through the version without it to show that no ruling moved.

    The arithmetic half runs in a context of :func:`exact_precision` over every operand, so a
    product or a band edge is computed exactly rather than rounded to 28 digits first.
    """
    moment = time.monotonic() if now is None else now
    trail = _Trail()

    numbers = (("quantity", quantity), ("price", price), ("reference price", reference_price),
               ("available balance", available_balance))
    supplied = ", ".join(f"{label} {value}" for label, value in numbers if value is not None)
    for label, value in numbers:
        if value is not None and not value.is_finite():
            return trail.refuse(
                "finite_inputs", Denial.NOT_A_NUMBER, f"{label} {value} is not a finite number",
                supplied, price=None, adjusted=False,
            )
    trail.note("finite_inputs", GateVerdict.PASS, "every number supplied is finite", supplied)

    normalised = side.lower() if isinstance(side, str) else ""
    if normalised not in _SIDES:
        return trail.refuse(
            "side_known", Denial.SIDE_UNKNOWN,
            f"side {side!r} is neither buy nor sell, the only two the venue accepts",
            f"side {side!r}", price=price, adjusted=False,
        )
    trail.note("side_known", GateVerdict.PASS, f"{normalised} is a side the venue accepts",
               f"side {side}")

    status = f"status {instrument.status}"
    if not instrument.is_online:
        return trail.refuse(
            "instrument_online", Denial.INSTRUMENT_OFFLINE,
            f"{instrument.symbol} is {instrument.status}, not online", status,
            price=price, adjusted=False,
        )
    trail.note("instrument_online", GateVerdict.PASS, f"{instrument.symbol} is online", status)

    if limiter is None:
        trail.note("rate_window", GateVerdict.SKIP, "no rate limiter supplied for this call")
    else:
        exceeded = limiter.would_exceed(now=moment, prune=record)
        in_window = limiter.count(now=moment)
        window = (f"{in_window} in the last {limiter.window_seconds:.0f}s, "
                  f"cap {limiter.max_orders}")
        if exceeded:
            return trail.refuse(
                "rate_window", Denial.RATE_LIMIT,
                f"{in_window} order(s) already submitted in the last "
                f"{limiter.window_seconds:.0f}s, at the cap of {limiter.max_orders}",
                window, price=price, adjusted=False,
            )
        trail.note("rate_window", GateVerdict.PASS, "a slot is free in the window", window)

    with localcontext() as ctx:
        # exact_precision() over the order's four numbers and the instrument's seven, with the
        # instrument's share read from its cached width rather than recounted on every order.
        ctx.prec = max(getcontext().prec, instrument.arithmetic_width + _PRECISION_MARGIN + sum(
            _digits_width(value) for value in (quantity, price, reference_price, available_balance)
        ))
        return _size_and_value(
            instrument, trail, quantity=quantity, price=price, reference_price=reference_price,
            side=normalised, available_balance=available_balance, limiter=limiter,
            moment=moment, record=record,
        )


def _size_and_value(
    instrument: Instrument,
    trail: _Trail,
    *,
    quantity: Decimal,
    price: Decimal | None,
    reference_price: Decimal | None,
    side: str,
    available_balance: Decimal | None,
    limiter: RateLimiter | None,
    moment: float,
    record: bool,
) -> Ruling:
    """The arithmetic half of :func:`_rule`: price, band, size and value, in the exact context."""
    if price is None:
        trail.note("price_positive", GateVerdict.SKIP, "market order: no limit price to check")
    elif price <= 0:
        return trail.refuse(
            "price_positive", Denial.PRICE_NOT_POSITIVE, f"price {price} is not positive",
            f"price {price}", price=price, adjusted=False,
        )
    else:
        trail.note("price_positive", GateVerdict.PASS, f"price {price} is positive",
                   f"price {price}")

    adjusted = False
    final_price = price
    if price is None:
        trail.note("price_step", GateVerdict.SKIP, "market order: no limit price to step")
        trail.note("price_band", GateVerdict.SKIP,
                   "market order: no limit price to hold to a band")
    else:
        tick = instrument.price_multiplier
        limit = quantise_down(price, tick)
        stepping = f"price {price}, tick {tick}"
        if limit <= 0:
            return trail.refuse(
                "price_step", Denial.PRICE_NOT_POSITIVE,
                f"price {price} steps down to {limit} on the venue's {tick} tick; a limit price "
                f"must stay positive", stepping, price=limit, adjusted=False,
            )
        if limit != price:
            adjusted = True
            trail.note("price_step", GateVerdict.ADJUST,
                       f"stepped down to {limit} on the {tick} tick", stepping)
        else:
            trail.note("price_step", GateVerdict.PASS, f"already a multiple of the {tick} tick",
                       stepping)
        final_price = limit

        if reference_price is None or reference_price <= 0:
            trail.note(
                "price_band", GateVerdict.SKIP,
                "no positive reference price, so the venue's band cannot be checked here",
                f"price {limit}, reference {reference_price}",
            )
        else:
            ratio = (
                instrument.buy_limit_ratio if side == "buy" else instrument.sell_limit_ratio
            )
            upper = reference_price * (Decimal("1") + ratio)
            lower = reference_price * (Decimal("1") - ratio)
            band = f"price {limit}, {side} band {lower} to {upper}"
            if not lower <= limit <= upper:
                return trail.refuse(
                    "price_band", Denial.PRICE_OUTSIDE_BAND,
                    f"{limit} is outside the venue's ±{ratio * 100}% band around "
                    f"{reference_price} ({lower} to {upper})",
                    band, price=limit, adjusted=False,
                )
            trail.note("price_band", GateVerdict.PASS,
                       f"inside the ±{ratio * 100}% band around {reference_price}", band)

    step = instrument.quantity_multiplier
    stepped_qty = quantise_down(quantity, step)
    stepping = f"quantity {quantity}, step {step}"
    if stepped_qty != quantity:
        adjusted = True
        trail.note("quantity_step", GateVerdict.ADJUST, f"stepped down to {stepped_qty}",
                   stepping)
    else:
        trail.note("quantity_step", GateVerdict.PASS, f"already a multiple of the {step} step",
                   stepping)

    ceiling = f"quantity {stepped_qty}, maximum {instrument.max_order_qty}"
    if stepped_qty > instrument.max_order_qty:
        stepped_qty = quantise_down(instrument.max_order_qty, step)
        adjusted = True
        if stepped_qty <= 0:
            return trail.refuse(
                "quantity_ceiling", Denial.QUANTITY_ABOVE_MAXIMUM,
                f"{quantity} exceeds the venue maximum {instrument.max_order_qty} and cannot be "
                f"reduced to a valid step", ceiling, price=final_price, adjusted=True,
            )
        trail.note("quantity_ceiling", GateVerdict.ADJUST,
                   f"cut to {stepped_qty}, the largest step within the venue maximum", ceiling)
    else:
        trail.note("quantity_ceiling", GateVerdict.PASS, "within the venue maximum", ceiling)

    floor = f"quantity {stepped_qty}, minimum {instrument.min_order_qty}"
    if stepped_qty < instrument.min_order_qty:
        return trail.refuse(
            "quantity_floor", Denial.QUANTITY_BELOW_MINIMUM,
            f"{quantity} rounds down to {stepped_qty}, below the venue minimum "
            f"{instrument.min_order_qty}; the guard may not round a size up to reach it",
            floor, price=final_price, adjusted=adjusted,
        )
    trail.note("quantity_floor", GateVerdict.PASS, "at or above the venue minimum", floor)

    mark = final_price if final_price is not None else reference_price
    if mark is None or mark <= 0:
        unvalued = "no positive limit price or reference price to value the order at"
        trail.note("notional_floor", GateVerdict.SKIP, unvalued)
        trail.note("balance", GateVerdict.SKIP, unvalued)
    else:
        notional = stepped_qty * mark
        valued = (f"notional {notional} = {stepped_qty} x {mark}, "
                  f"minimum {instrument.min_order_amount}")
        if notional < instrument.min_order_amount:
            return trail.refuse(
                "notional_floor", Denial.NOTIONAL_BELOW_MINIMUM,
                f"notional {notional} is below the venue minimum "
                f"{instrument.min_order_amount}", valued, price=final_price, adjusted=adjusted,
            )
        trail.note("notional_floor", GateVerdict.PASS, "at or above the venue's minimum notional",
                   valued)
        if available_balance is None:
            trail.note("balance", GateVerdict.SKIP, "no available balance supplied")
        else:
            funded = f"notional {notional}, available {available_balance}"
            if notional > available_balance:
                return trail.refuse(
                    "balance", Denial.INSUFFICIENT_BALANCE,
                    f"notional {notional} exceeds the available balance {available_balance}",
                    funded, price=final_price, adjusted=adjusted,
                )
            trail.note("balance", GateVerdict.PASS, "covered by the available balance", funded)

    reason = (
        "adjusted to the venue's step and limits" if adjusted
        else "within every published venue rule"
    )
    if limiter is not None and record:
        limiter.record(now=moment)
    return Ruling(Denial.NONE, reason, stepped_qty, final_price, adjusted, trail.seal())


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
        """Enforce. Spends a rate-window slot when the order is allowed."""
        return self._judge(
            symbol, quantity=quantity, price=price, reference_price=reference_price, side=side,
            available_balance=available_balance, now=now, record=True,
        )

    def self_check(
        self, symbol: str, *, quantity: Decimal, price: Decimal | None = None,
        reference_price: Decimal | None = None, side: str = "buy",
        available_balance: Decimal | None = None, now: float | None = None,
    ) -> bool:
        """Would :meth:`check` allow this order right now? Pass or fail only; nothing recorded.

        The session-level twin of :func:`would_pass`, and it goes through the same
        :meth:`_judge` as :meth:`check`, so the unknown-instrument denial is shared as well.
        """
        try:
            return self._judge(
                symbol, quantity=quantity, price=price, reference_price=reference_price,
                side=side, available_balance=available_balance, now=now, record=False,
            ).allowed
        except (GuardError, ArithmeticError):
            return False

    def _judge(
        self, symbol: str, *, quantity: Decimal, price: Decimal | None,
        reference_price: Decimal | None, side: str, available_balance: Decimal | None,
        now: float | None, record: bool,
    ) -> Ruling:
        instrument = self.instruments.get(symbol)
        looked_up = f"symbol {symbol}"
        if instrument is None:
            reason = (
                f"{symbol} is not among the {len(self.instruments)} instruments the venue "
                f"published; no rules are known for it"
            )
            unknown = TraceEvent("instrument_known", looked_up, GateVerdict.DENY, reason)
            return Ruling(
                Denial.INSTRUMENT_UNKNOWN, reason, _ZERO, price, False,
                PermissionCheckTrace((unknown, *not_reached(RULE_GATES, "instrument_known"))),
            )
        ruling = _rule(
            instrument, quantity=quantity, price=price, reference_price=reference_price,
            side=side, available_balance=available_balance, limiter=self.limiter, now=now,
            record=record,
        )
        known = TraceEvent("instrument_known", looked_up, GateVerdict.PASS,
                           f"{symbol} has published venue rules")
        return replace(ruling, trace=ruling.trace.prepend(known))


_ORDER_FIELDS = frozenset(
    {"symbol", "quantity", "price", "reference_price", "side", "available_balance"}
)


def self_check_request(guard: Guard, body: Any) -> dict[str, bool]:
    """The durable self-check endpoint's whole contract: an order as JSON in, ``{"pass": bool}``
    out.

    mle-bench exposes its self-check as ``POST /validate`` inside the agent's container
    (``environment/grading_server.py:22-35``) so the agent can ask as often as it likes without
    touching the grader's private answers. This is the same shape for an agent that wants to know
    whether an order would clear the guard before it commits to one, and it is written as a plain
    function so any transport (the console's HTTP handler, the MCP server, a CLI) can carry it.

    Parsing is strict and every failure is a ``False``, never an exception and never a default:
    a quantity sent as a float, a missing symbol, an unknown field or a non-object body cannot be
    ruled on, and an order that cannot be ruled on does not pass. Floats are refused rather than
    converted because ``Decimal(0.1)`` is ``0.1000000000000000055511151231257827...`` — a binary
    approximation ruling on a size the caller never asked about. Numbers must arrive as strings or
    integers.
    """
    if not isinstance(body, dict) or not set(body) <= _ORDER_FIELDS:
        return {"pass": False}
    symbol = body.get("symbol")
    side = body.get("side", "buy")
    if not isinstance(symbol, str) or not symbol or not isinstance(side, str):
        return {"pass": False}

    def number(key: str) -> Decimal | None:
        raw = body.get(key)
        if raw is None:
            return None
        if isinstance(raw, bool) or not isinstance(raw, (str, int)):
            raise GuardError(f"{key} must be a decimal string or an integer")
        value = Decimal(str(raw).strip())
        if not value.is_finite():
            raise GuardError(f"{key} must be finite")
        return value

    try:
        quantity = number("quantity")
        price = number("price")
        reference_price = number("reference_price")
        available_balance = number("available_balance")
    except (GuardError, ArithmeticError, ValueError):
        return {"pass": False}
    if quantity is None:
        return {"pass": False}
    return {"pass": guard.self_check(
        symbol, quantity=quantity, price=price, reference_price=reference_price, side=side,
        available_balance=available_balance,
    )}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ARGUS pre-trade guard")
    parser.add_argument("--symbol", default="NVDAUSDT")
    parser.add_argument("--quantity", type=Decimal, default=Decimal("1"))
    parser.add_argument("--price", type=Decimal, default=None)
    parser.add_argument("--self-check", action="store_true",
                        help="answer PASS or FAIL only, recording nothing")
    args = parser.parse_args(argv)

    try:
        instruments = fetch_instruments(symbol=args.symbol)
    except GuardError as exc:
        print(f"guard unavailable: {exc}")
        return 1

    guard = Guard(instruments=instruments)
    if args.self_check:
        passed = guard.self_check(args.symbol, quantity=args.quantity, price=args.price,
                                  reference_price=args.price)
        print("PASS" if passed else "FAIL")
        return 0 if passed else 1
    instrument = instruments[args.symbol]
    print(json.dumps(instrument.as_dict(), indent=2))
    ruling = guard.check(
        args.symbol, quantity=args.quantity, price=args.price,
        reference_price=args.price,
    )
    print(ruling.explain())
    return 0 if ruling.allowed else 1


__all__ = [
    "GUARD_GATES",
    "INSTRUMENTS_URL",
    "RULE_GATES",
    "Denial",
    "GateVerdict",
    "Guard",
    "GuardError",
    "Instrument",
    "PermissionCheckTrace",
    "RateLimiter",
    "Ruling",
    "TraceEvent",
    "exact_precision",
    "fetch_instruments",
    "main",
    "not_reached",
    "quantise_down",
    "self_check_request",
    "validate",
    "would_pass",
]


if __name__ == "__main__":
    raise SystemExit(main())
