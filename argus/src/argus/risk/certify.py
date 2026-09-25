"""Certifying every risk gate from its written specification — a second implementation, not a rerun.

**Why this exists when the guard already has a self-check.** `execution/guard.py`'s self-check
(S13) proves ``would_pass(x) == validate(x).allowed``: the same function, asked twice. That is the
right property for a probe an agent may call, and it is silent on the question a judge of "risk
control layer effectiveness" actually asks — *is the function right?* A gate that allows an order
it should refuse agrees with itself perfectly. PlanBench (Valmeekam et al., NeurIPS 2023; MIT,
``karthikv792/LLMs-Planning``) is built on exactly that distinction: it never lets the planner, or
another model, judge a plan; it replays the plan through VAL, an external validator that knows only
the domain's preconditions and effects, and records which of the two kinds of correction signal is
sound (``llm_planning_analysis/back_prompting.py:355-359``).

This module is the VAL for ARGUS's risk layer. For each gate it holds a **written specification**
— what must be true for an order to pass the gate (its precondition) and what passing does to the
order (its effect) — and an evaluator written from that text alone. It replays the gate stack the
way ``full_validator/__init__.py:6-63`` (``get_all_errors``) replays a plan: walk the steps in
order, check each step's preconditions against the current state, apply its effects, and check the
goal at the end. The output is a :class:`Certificate`, and when an order stops it says where and
why in words, after ``utils/task_utils.py:627-668`` (``get_val_error_message``: "the following
action at step N has an unmet precondition: ..."):

    quantity_floor (step 10 of 12): unmet precondition — quantity 0.004 steps down to 0, below the
    venue minimum 0.01

**Independence is the whole point, so it is enforced, not promised.**

* No import from `execution/guard.py`, `risk/circuit.py`, `risk/sizing.py`,
  `risk/session_risk.py` or `risk/modes.py`. The certifier reads a ruling's public fields through
  the ``observe_*`` functions and never calls a gate. ``tests/test_risk_certify.py`` asserts the
  import list, so a later convenience import cannot quietly make the check circular.
* **Different arithmetic.** The guard computes in :class:`~decimal.Decimal` contexts; this computes
  in :class:`~fractions.Fraction`, which is exact by construction. Steps are truncated with
  ``int()`` on a rational, not ``to_integral_value``. The two agreeing is evidence; the day they
  disagree, one of them is wrong and the certificate names the gate.
* **Its own thresholds.** The circuit breaker's limits are restated here from its documentation,
  not imported. A threshold changed in `risk/circuit.py` without a matching change here is flagged
  on the next run — deliberately, because an unreviewed change to a halt threshold is exactly what a
  certifier exists to catch.

**What was taken, and what was not.** Taken: the step-replay structure (precondition check, effect
application, goal check), the localised unmet-precondition message, and the full-list mode
(``get_custom_validator_error_message``, ``task_utils.py:670-706``) — :attr:`Certificate.violations`
lists every departure, first one first. Not taken: the PDDL parser and the VAL binary (ARGUS's
gates are a dozen predicates; a STRIPS toolchain would be more code than the rules), and the
LLM-feedback path, which PlanBench itself documents as unsound.

**Three states, never two.** ``certified`` — the observed ruling is exactly what the specification
requires, trace included. ``violated`` — it is not, and :attr:`Certificate.violations` says how.
``uncertifiable`` — the record lacks what the specification needs (a pre-2026-09-13 decision with
no risk record, say). An uncertifiable ruling is not counted as a pass.

**Found by writing the specification (2026-09-25).** Stating the guard's rules as preconditions
exposed three orders it allowed against its own contract — a quantity rounded *up* by 28-digit
arithmetic, a sub-tick limit price allowed at zero, and a side that was neither buy nor sell. The
guard was fixed the same day; `eval/risk_shadow.py` lists every ruling that changed, and
`eval/risk_certify.py` shows this certifier flagging all three on the guard as it stood before.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal, localcontext
from enum import StrEnum
from fractions import Fraction
from typing import Any

# --- the certificate ------------------------------------------------------------------------------


class Status(StrEnum):
    """What a certificate concluded. Three states: an unreadable record is not a pass."""

    CERTIFIED = "certified"
    VIOLATED = "violated"
    UNCERTIFIABLE = "uncertifiable"


@dataclass(frozen=True, slots=True)
class StepCheck:
    """One gate, replayed: what the specification requires against what the ruling recorded."""

    step: int
    gate: str
    expected: str
    observed: str | None
    """``None`` when the ruling carries no trace to compare (an implementation without one)."""

    words: str

    @property
    def holds(self) -> bool:
        return self.observed is None or self.observed == self.expected

    def as_dict(self) -> dict[str, Any]:
        return {"step": self.step, "gate": self.gate, "expected": self.expected,
                "observed": self.observed, "holds": self.holds, "words": self.words}


@dataclass(frozen=True, slots=True)
class Certificate:
    """The certifier's finding on one ruling of one gate stack."""

    subject: str
    status: Status
    expected: str
    """The outcome the specification requires, in words."""

    observed: str
    first_unmet: str | None
    """Where and why the specification stops this order, localised to one gate; ``None`` when the
    specification lets it through. This is the independent explanation of a refusal."""

    violations: tuple[str, ...] = ()
    """Every way the observed ruling departs from the specification, first one first."""

    steps: tuple[StepCheck, ...] = ()

    @property
    def certified(self) -> bool:
        return self.status is Status.CERTIFIED

    def render(self) -> str:
        head = f"[certify] {self.subject}: {self.status} — expected {self.expected}"
        if self.observed != self.expected:
            head += f", observed {self.observed}"
        lines = [head]
        if self.first_unmet:
            lines.append(f"  first unmet precondition: {self.first_unmet}")
        lines.extend(f"  violation: {v}" for v in self.violations)
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject, "status": str(self.status), "expected": self.expected,
            "observed": self.observed, "first_unmet": self.first_unmet,
            "violations": list(self.violations), "steps": [s.as_dict() for s in self.steps],
        }


def _conclude(
    subject: str, expected: str, observed: str, first_unmet: str | None,
    violations: Sequence[str], steps: Sequence[StepCheck] = (),
) -> Certificate:
    return Certificate(
        subject=subject,
        status=Status.VIOLATED if violations else Status.CERTIFIED,
        expected=expected, observed=observed, first_unmet=first_unmet,
        violations=tuple(violations), steps=tuple(steps),
    )


# --- exact arithmetic -----------------------------------------------------------------------------


def exact(value: object) -> Fraction | None:
    """The exact rational value of a number, or ``None`` when it is not a finite number.

    ``Fraction`` refuses NaN (``ValueError``) and infinity (``OverflowError``) itself, which is the
    finiteness test used below — a different mechanism from the guard's ``Decimal.is_finite``.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        return Fraction(value if isinstance(value, (int, Fraction)) else Decimal(str(value)))
    except (ValueError, OverflowError, ArithmeticError):
        return None


def step_down(value: Fraction, step: Fraction) -> Fraction:
    """The multiple of ``step`` nearest zero from ``value``'s side: magnitude never increases."""
    return int(value / step) * step


def show(value: Fraction | None) -> str:
    """A rational as the decimal it is, or to twelve significant figures when it does not end."""
    if value is None:
        return "none"
    with localcontext() as ctx:
        ctx.prec = 80
        as_decimal = Decimal(value.numerator) / Decimal(value.denominator)
    if Fraction(as_decimal) == value:
        text = format(as_decimal.normalize(), "f")
        return "0" if text in ("-0", "0") else text
    return f"{float(value):.12g}"


# --- the venue gates ------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GateSpec:
    """One gate as written down: what it requires, what it does, and the denial it owns."""

    gate: str
    denials: tuple[str, ...]
    precondition: str
    effect: str

    def as_dict(self) -> dict[str, Any]:
        return {"gate": self.gate, "denials": list(self.denials),
                "precondition": self.precondition, "effect": self.effect}


VENUE_SPEC: tuple[GateSpec, ...] = (
    GateSpec("instrument_known", ("instrument_unknown",),
             "the symbol is one the venue published rules for (checked by the session guard only)",
             "none"),
    GateSpec("finite_inputs", ("not_a_number",),
             "the quantity, and the limit price, reference price and balance when given, are "
             "finite numbers", "none"),
    GateSpec("side_known", ("side_unknown",),
             "the side is buy or sell, ignoring case — the venue's own enum", "none"),
    GateSpec("instrument_online", ("instrument_offline",),
             "the instrument's published status is online, ignoring case", "none"),
    GateSpec("rate_window", ("rate_limit",),
             "fewer submissions than the cap sit strictly inside the window ending now; a stamp "
             "exactly one window old has left it. Skipped when no window is supplied",
             "none here; one slot is spent, at now, only if the whole order is allowed and the "
             "call enforces rather than self-checks"),
    GateSpec("price_positive", ("price_not_positive",),
             "a limit price, when given, is above zero. Skipped for a market order", "none"),
    GateSpec("price_step", ("price_not_positive",),
             "the limit price stepped down to the tick is still above zero",
             "the limit price becomes the largest tick multiple not above it"),
    GateSpec("price_band", ("price_outside_band",),
             "with a reference price above zero, the stepped limit price lies in reference x "
             "(1 - ratio) to reference x (1 + ratio) inclusive, with the buy ratio for a buy and "
             "the sell ratio for a sell. Skipped without a limit price or a positive reference",
             "none"),
    GateSpec("quantity_step", (),
             "none", "the quantity becomes the step multiple nearest zero on its own side"),
    GateSpec("quantity_ceiling", ("quantity_above_maximum",),
             "when the quantity exceeds the venue maximum, the maximum stepped down is above zero",
             "a quantity above the maximum becomes the maximum stepped down"),
    GateSpec("quantity_floor", ("quantity_below_minimum",),
             "the stepped quantity is at least the venue minimum; the guard may not raise it",
             "none"),
    GateSpec("notional_floor", ("notional_below_minimum",),
             "valued at the stepped limit price, or at the reference price for a market order, "
             "the notional is at least the venue minimum. Skipped when no positive value exists",
             "none"),
    GateSpec("balance", ("insufficient_balance",),
             "the same notional does not exceed the available balance. Skipped without a "
             "balance or without a value", "none"),
)
"""The venue gates in the order the specification requires them, one entry per gate."""

VENUE_GOALS: tuple[str, ...] = (
    "an allowed order is never larger than the order asked for",
    "an allowed quantity is a multiple of the step, within the venue minimum and maximum",
    "an allowed limit price is above zero, a multiple of the tick and not above the price asked",
    "an allowed limit price sits inside the band whenever a positive reference was given",
    "an allowed, valued order meets the minimum notional and fits the balance when one was given",
    "a refused order carries quantity zero and spends no rate-window slot",
)
"""What must hold of the final ruling, whatever path produced it — PlanBench's goal check."""

_SIDES = ("buy", "sell")


@dataclass(frozen=True, slots=True)
class VenueRules:
    """The venue's published numbers for one instrument, as the certifier reads them."""

    symbol: str
    status: str
    price_tick: Fraction
    quantity_step: Fraction
    min_quantity: Fraction
    max_quantity: Fraction
    min_notional: Fraction
    buy_ratio: Fraction
    sell_ratio: Fraction

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> VenueRules:
        """From the venue's own published row (``GET /api/v3/market/instruments``).

        The preferred source: it reads the venue's field names directly, so a defect in the
        guard's own parser (``Instrument.from_payload``) cannot leak into the specification.
        """
        def need(key: str) -> Fraction:
            got = exact(row.get(key))
            if got is None:
                raise ValueError(f"venue row field {key} is not a finite number: {row.get(key)!r}")
            return got

        return cls(
            symbol=str(row["symbol"]), status=str(row["status"]),
            price_tick=need("priceMultiplier"), quantity_step=need("quantityMultiplier"),
            min_quantity=need("minOrderQty"), max_quantity=need("maxOrderQty"),
            min_notional=need("minOrderAmount"), buy_ratio=need("buyLimitPriceRatio"),
            sell_ratio=need("sellLimitPriceRatio"),
        )

    @classmethod
    def read(cls, instrument: Any) -> VenueRules:
        """From anything carrying the guard's field names — its ``Instrument`` of any version.
        Only public fields are read. Prefer :meth:`from_row` where the raw row is at hand."""
        def need(value: object, name: str) -> Fraction:
            got = exact(value)
            if got is None:
                raise ValueError(f"venue field {name} is not a finite number: {value!r}")
            return got

        return cls(
            symbol=str(instrument.symbol), status=str(instrument.status),
            price_tick=need(instrument.price_multiplier, "price_multiplier"),
            quantity_step=need(instrument.quantity_multiplier, "quantity_multiplier"),
            min_quantity=need(instrument.min_order_qty, "min_order_qty"),
            max_quantity=need(instrument.max_order_qty, "max_order_qty"),
            min_notional=need(instrument.min_order_amount, "min_order_amount"),
            buy_ratio=need(instrument.buy_limit_ratio, "buy_limit_ratio"),
            sell_ratio=need(instrument.sell_limit_ratio, "sell_limit_ratio"),
        )


@dataclass(frozen=True, slots=True)
class VenueOrder:
    """The order exactly as the guard was asked about it."""

    quantity: object
    price: object = None
    reference_price: object = None
    side: object = "buy"
    available_balance: object = None
    symbol: str = ""


@dataclass(frozen=True, slots=True)
class Window:
    """The rate window as it stood when the order arrived."""

    stamps: tuple[float, ...]
    max_orders: int
    window_seconds: float
    now: float

    def inside(self, stamps: Iterable[float] | None = None) -> list[float]:
        cutoff = self.now - self.window_seconds
        return sorted(s for s in (self.stamps if stamps is None else stamps) if s > cutoff)


@dataclass(frozen=True, slots=True)
class ObservedVenue:
    """A venue ruling reduced to plain values, whichever implementation produced it."""

    denial: str
    quantity: Decimal
    price: Decimal | None
    adjusted: bool
    events: tuple[tuple[str, str, str], ...] | None
    """``(gate, verdict, reason)`` per trace event, or ``None`` when the ruling has no trace."""

    raised: str | None = None
    """The exception the implementation raised instead of ruling, if it did."""

    @property
    def allowed(self) -> bool:
        return self.raised is None and self.denial == "none"

    def outcome(self) -> str:
        if self.raised is not None:
            return f"raised {self.raised}"
        if self.allowed:
            priced = "" if self.price is None else f" @ {show(exact(self.price))}"
            return f"allowed {show(exact(self.quantity))}{priced}"
        return f"denied {self.denial}"


def observe_venue(ruling: Any) -> ObservedVenue:
    """Read a guard ruling's public fields. Works on the current guard, a baseline, or a mutant."""
    trace = getattr(ruling, "trace", None)
    events = None if trace is None else tuple(
        (str(event.gate), str(event.verdict), str(event.reason)) for event in trace.events
    )
    return ObservedVenue(
        denial=str(ruling.denial),
        quantity=Decimal(str(ruling.quantity)),
        price=None if ruling.price is None else Decimal(str(ruling.price)),
        adjusted=bool(ruling.adjusted),
        events=events,
    )


def raised(exc: BaseException) -> ObservedVenue:
    """The observation for an implementation that raised rather than ruling."""
    return ObservedVenue("none", Decimal(0), None, False, None, raised=type(exc).__name__)


@dataclass(frozen=True, slots=True)
class _Expected:
    steps: tuple[tuple[str, str, str], ...]
    """``(gate, verdict, words)`` for every gate in the stack."""

    denial: str
    quantity: Fraction
    price: Fraction | None
    adjusted: bool
    first_unmet: str | None


def expect_venue(
    order: VenueOrder, rules: VenueRules | None, *, lookup: bool = False,
    window: Window | None = None,
) -> _Expected:
    """What the specification says the gate stack must do with this order, gate by gate.

    ``lookup`` is true for a session-guard call, which begins with ``instrument_known``;
    ``rules`` is then ``None`` for a symbol the venue did not publish.
    """
    gates = [spec.gate for spec in VENUE_SPEC if lookup or spec.gate != "instrument_known"]
    steps: list[tuple[str, str, str]] = []

    def stop(gate: str, denial: str, words: str, *, price: Fraction | None = None) -> _Expected:
        steps.append((gate, "deny", words))
        index = len(steps)
        rest = gates[index:]
        steps.extend((later, "not_reached", f"{gate} refused first") for later in rest)
        return _Expected(tuple(steps), denial, Fraction(0), price, False,
                         f"{gate} (step {index} of {len(gates)}): unmet precondition — {words}")

    if lookup:
        if rules is None:
            return stop("instrument_known", "instrument_unknown",
                        f"{order.symbol or 'the symbol'} has no published venue rules")
        steps.append(("instrument_known", "pass", f"{rules.symbol} has published rules"))
    if rules is None:
        raise ValueError("a direct validate() call needs the instrument's rules")

    if order.quantity is None:
        return stop("finite_inputs", "not_a_number", "no quantity was supplied")
    numbers = (("quantity", order.quantity), ("price", order.price),
               ("reference price", order.reference_price),
               ("available balance", order.available_balance))
    values: dict[str, Fraction | None] = {}
    for label, raw in numbers:
        values[label] = None if raw is None else exact(raw)
        if raw is not None and values[label] is None:
            return stop("finite_inputs", "not_a_number", f"{label} {raw} is not a finite number")
    steps.append(("finite_inputs", "pass", "every number supplied is finite"))
    quantity = values["quantity"]
    assert quantity is not None
    price, reference, balance = (values["price"], values["reference price"],
                                 values["available balance"])

    side = order.side.lower() if isinstance(order.side, str) else None
    if side not in _SIDES:
        return stop("side_known", "side_unknown",
                    f"side {order.side!r} is not one of the venue's sides, buy or sell")
    steps.append(("side_known", "pass", f"{side} is a venue side"))

    if rules.status.lower() != "online":
        return stop("instrument_online", "instrument_offline",
                    f"{rules.symbol} is {rules.status}, not online")
    steps.append(("instrument_online", "pass", f"{rules.symbol} is online"))

    if window is None:
        steps.append(("rate_window", "skip", "no rate window supplied"))
    else:
        inside = len(window.inside())
        if inside >= window.max_orders:
            return stop("rate_window", "rate_limit",
                        f"{inside} submission(s) inside the {window.window_seconds:g}s window "
                        f"ending at {window.now:g}, at the cap of {window.max_orders}")
        steps.append(("rate_window", "pass", f"{inside} of {window.max_orders} slots in use"))

    adjusted = False
    limit: Fraction | None = None
    if price is None:
        steps.append(("price_positive", "skip", "market order"))
        steps.append(("price_step", "skip", "market order"))
        steps.append(("price_band", "skip", "market order"))
    else:
        if price <= 0:
            return stop("price_positive", "price_not_positive",
                        f"limit price {show(price)} is not above zero", price=price)
        steps.append(("price_positive", "pass", f"limit price {show(price)} is above zero"))
        limit = step_down(price, rules.price_tick)
        if limit <= 0:
            return stop("price_step", "price_not_positive",
                        f"limit price {show(price)} steps down to {show(limit)} on the "
                        f"{show(rules.price_tick)} tick", price=limit)
        if limit != price:
            adjusted = True
            steps.append(("price_step", "adjust", f"stepped down to {show(limit)}"))
        else:
            steps.append(("price_step", "pass", "already on the tick"))
        if reference is None or reference <= 0:
            steps.append(("price_band", "skip", "no positive reference price"))
        else:
            ratio = rules.buy_ratio if side == "buy" else rules.sell_ratio
            lower, upper = reference * (1 - ratio), reference * (1 + ratio)
            if not lower <= limit <= upper:
                return stop("price_band", "price_outside_band",
                            f"limit price {show(limit)} lies outside the {side} band "
                            f"{show(lower)} to {show(upper)} around {show(reference)}",
                            price=limit)
            steps.append(("price_band", "pass", f"inside {show(lower)} to {show(upper)}"))

    stepped = step_down(quantity, rules.quantity_step)
    if stepped != quantity:
        adjusted = True
        steps.append(("quantity_step", "adjust", f"stepped down to {show(stepped)}"))
    else:
        steps.append(("quantity_step", "pass", "already on the step"))

    if stepped > rules.max_quantity:
        ceiling = step_down(rules.max_quantity, rules.quantity_step)
        if ceiling <= 0:
            return stop("quantity_ceiling", "quantity_above_maximum",
                        f"quantity {show(quantity)} exceeds the maximum {show(rules.max_quantity)},"
                        f" which steps down to {show(ceiling)}, so no valid size exists",
                        price=limit)
        stepped, adjusted = ceiling, True
        steps.append(("quantity_ceiling", "adjust", f"cut to {show(ceiling)}"))
    else:
        steps.append(("quantity_ceiling", "pass", "within the maximum"))

    if stepped < rules.min_quantity:
        return stop("quantity_floor", "quantity_below_minimum",
                    f"quantity {show(quantity)} steps down to {show(stepped)}, below the venue "
                    f"minimum {show(rules.min_quantity)}", price=limit)
    steps.append(("quantity_floor", "pass", "at or above the minimum"))

    mark = limit if limit is not None else reference
    if mark is None or mark <= 0:
        steps.append(("notional_floor", "skip", "no positive value to price the order at"))
        steps.append(("balance", "skip", "no positive value to price the order at"))
    else:
        notional = stepped * mark
        if notional < rules.min_notional:
            return stop("notional_floor", "notional_below_minimum",
                        f"notional {show(notional)} ({show(stepped)} x {show(mark)}) is below the "
                        f"venue minimum {show(rules.min_notional)}", price=limit)
        steps.append(("notional_floor", "pass", f"notional {show(notional)}"))
        if balance is None:
            steps.append(("balance", "skip", "no balance supplied"))
        elif notional > balance:
            return stop("balance", "insufficient_balance",
                        f"notional {show(notional)} exceeds the available balance "
                        f"{show(balance)}", price=limit)
        else:
            steps.append(("balance", "pass", f"within the balance {show(balance)}"))

    return _Expected(tuple(steps), "none", stepped, limit, adjusted, None)


def _goals(order: VenueOrder, rules: VenueRules, seen: ObservedVenue) -> list[str]:
    """Goal conditions on an allowed ruling, checked on the ruling alone — the words a reader
    wants when an order got through that should not have."""
    unmet: list[str] = []
    asked = exact(order.quantity)
    got = exact(seen.quantity)
    if got is None:
        return [f"allowed quantity {seen.quantity} is not a finite number"]
    if asked is not None and abs(got) > abs(asked):
        unmet.append(f"goal unmet: allowed quantity {show(got)} is larger than the "
                     f"{show(asked)} asked for")
    if step_down(got, rules.quantity_step) != got:
        unmet.append(f"goal unmet: allowed quantity {show(got)} is not a multiple of the step "
                     f"{show(rules.quantity_step)}")
    if got < rules.min_quantity or got > rules.max_quantity:
        unmet.append(f"goal unmet: allowed quantity {show(got)} is outside "
                     f"{show(rules.min_quantity)} to {show(rules.max_quantity)}")
    limit = exact(seen.price) if seen.price is not None else None
    if order.price is not None:
        asked_price = exact(order.price)
        if limit is None or limit <= 0:
            unmet.append(f"goal unmet: allowed limit price {show(limit)} is not above zero")
        else:
            if step_down(limit, rules.price_tick) != limit:
                unmet.append(f"goal unmet: allowed limit price {show(limit)} is off the tick "
                             f"{show(rules.price_tick)}")
            if asked_price is not None and limit > asked_price:
                unmet.append(f"goal unmet: allowed limit price {show(limit)} is above the "
                             f"{show(asked_price)} asked for")
            reference = exact(order.reference_price)
            side = order.side.lower() if isinstance(order.side, str) else ""
            if reference is not None and reference > 0:
                ratio = rules.buy_ratio if side == "buy" else rules.sell_ratio
                if not reference * (1 - ratio) <= limit <= reference * (1 + ratio):
                    unmet.append(f"goal unmet: allowed limit price {show(limit)} is outside the "
                                 f"band around {show(reference)}")
    mark = limit if order.price is not None else exact(order.reference_price)
    if mark is not None and mark > 0:
        notional = got * mark
        if notional < rules.min_notional:
            unmet.append(f"goal unmet: allowed notional {show(notional)} is below the minimum "
                         f"{show(rules.min_notional)}")
        balance = exact(order.available_balance)
        if balance is not None and notional > balance:
            unmet.append(f"goal unmet: allowed notional {show(notional)} exceeds the balance "
                         f"{show(balance)}")
    return unmet


def certify_venue(
    order: VenueOrder, rules: VenueRules | None, seen: ObservedVenue, *, lookup: bool = False,
    window: Window | None = None, stamps_after: Sequence[float] | None = None,
    enforcing: bool = True, subject: str = "venue guard",
) -> Certificate:
    """Replay the venue gate stack from the specification and compare it with ``seen``.

    ``window`` and ``stamps_after`` are the rate window's stamps before and after the call; with
    both, the effect "one slot spent at now if and only if the order was allowed and the call
    enforced" is checked as well. ``enforcing`` is false for a self-check, which must spend none.
    """
    want = expect_venue(order, rules, lookup=lookup, window=window)
    expected = (f"denied {want.denial}" if want.denial != "none"
                else "allowed " + show(want.quantity)
                + ("" if want.price is None else f" @ {show(want.price)}"))
    violations: list[str] = []
    checks: list[StepCheck] = []

    if seen.raised is not None:
        violations.append(f"the implementation raised {seen.raised} instead of ruling; the "
                          f"specification requires a ruling for every order")
    else:
        if seen.denial != want.denial:
            violations.append(f"ruled {seen.denial}, but the specification requires {want.denial}"
                              + (f" because {want.first_unmet}" if want.first_unmet else ""))
        if seen.events is None:
            checks = [StepCheck(i, g, v, None, w) for i, (g, v, w) in enumerate(want.steps, 1)]
        else:
            seen_gates = [event[0] for event in seen.events]
            want_gates = [step[0] for step in want.steps]
            if seen_gates != want_gates:
                violations.append(f"trace lists gates {seen_gates}; the specification's stack "
                                  f"is {want_gates}")
            for index, ((gate, verdict, words), event) in enumerate(
                zip(want.steps, seen.events, strict=False), start=1,
            ):
                check = StepCheck(index, gate, verdict, event[1], words)
                checks.append(check)
                if event[0] == gate and not check.holds:
                    violations.append(
                        f"{gate} (step {index}): the trace records {event[1]} but the "
                        f"specification requires {verdict} — {words}"
                    )
        if seen.allowed:
            if rules is not None:
                violations.extend(_goals(order, rules, seen))
            got = exact(seen.quantity)
            if want.denial == "none" and got != want.quantity:
                violations.append(f"allowed quantity {show(got)}; the specification's effects "
                                  f"give {show(want.quantity)}")
            got_price = None if seen.price is None else exact(seen.price)
            if want.denial == "none" and got_price != want.price:
                violations.append(f"allowed limit price {show(got_price)}; the specification's "
                                  f"effects give {show(want.price)}")
            if want.denial == "none" and seen.adjusted != want.adjusted:
                violations.append(f"adjusted flag {seen.adjusted}; the specification's effects "
                                  f"{'changed' if want.adjusted else 'did not change'} the order")
        elif exact(seen.quantity) != 0:
            violations.append(f"refused order carries quantity {seen.quantity}, not zero")

    if window is not None and stamps_after is not None:
        spends = enforcing and seen.allowed
        before, after = window.inside(), window.inside(stamps_after)
        wanted = sorted([*before, window.now]) if spends else before
        if after != wanted:
            violations.append(
                f"rate window holds {len(after)} submission(s) after the call; the effect "
                f"requires {len(wanted)} ("
                + ("one slot spent at now" if spends else "none spent") + ")"
            )

    return _conclude(subject, expected, seen.outcome(), want.first_unmet, violations, checks)


# --- the circuit breaker --------------------------------------------------------------------------

TOTAL_DRAWDOWN_HALT = Fraction("0.10")
SESSION_DRAWDOWN_HALT = Fraction("0.04")
LADDER_FROM = Fraction("0.02")
LADDER_HALF = Fraction("0.06")
SHOCK_SIGMA = Fraction(3)
LOSING_STREAK = 4
STALE_AFTER = timedelta(hours=6)

CIRCUIT_SPEC: tuple[GateSpec, ...] = (
    GateSpec("total_drawdown", ("halted",),
             f"peak-to-trough drawdown is below {show(TOTAL_DRAWDOWN_HALT)}", "demands halted"),
    GateSpec("session_drawdown", ("halted",),
             f"drawdown since the session opened is below {show(SESSION_DRAWDOWN_HALT)}",
             "demands halted"),
    GateSpec("losing_streak", ("halted",),
             f"fewer than {LOSING_STREAK} consecutive losses", "demands halted"),
    GateSpec("shock", ("halted",),
             f"the signed move is above -{show(SHOCK_SIGMA)} sigma", "demands halted"),
    GateSpec("stale_evidence", ("reduce_only",),
             f"the evidence is no older than {STALE_AFTER}", "demands reduce_only"),
    GateSpec("drawdown_ladder", ("reduce_only",),
             f"drawdown is below {show(LADDER_FROM)} or already at the halt",
             "demands reduce_only; appetite scales 1 / 0.75 / 0.5 / 0 at 0, 2%, 6%, 10%"),
)
"""The breaker's six rules. Every rule that fires must be reported: the breaker's own contract is
that it evaluates all of them rather than stopping at the first (`risk/circuit.py` ``assess``)."""

_OPENS = frozenset({"trade", "hedge"})


@dataclass(frozen=True, slots=True)
class BookFacts:
    """The book the breaker judged, as plain values."""

    equity: Fraction
    peak_equity: Fraction
    session_open_equity: Fraction
    consecutive_losses: int
    open_positions: int
    evidence_age: timedelta
    move_sigma: Fraction

    @classmethod
    def read(cls, state: Any) -> BookFacts:
        def need(value: object) -> Fraction:
            got = exact(value)
            if got is None:
                raise ValueError(f"book field is not a finite number: {value!r}")
            return got

        return cls(
            equity=need(state.equity), peak_equity=need(state.peak_equity),
            session_open_equity=need(state.session_open_equity),
            consecutive_losses=int(state.consecutive_losses),
            open_positions=int(state.open_positions), evidence_age=state.evidence_age,
            move_sigma=need(state.realised_move_sigma),
        )

    @property
    def drawdown(self) -> Fraction:
        if self.peak_equity <= 0:
            return Fraction(0)
        return max(Fraction(0), (self.peak_equity - self.equity) / self.peak_equity)

    @property
    def session_drawdown(self) -> Fraction:
        if self.session_open_equity <= 0:
            return Fraction(0)
        return max(Fraction(0),
                   (self.session_open_equity - self.equity) / self.session_open_equity)


def expect_trips(book: BookFacts) -> list[tuple[str, str, str]]:
    """``(rule, demands, words)`` for every rule the specification says fires on this book."""
    fired: list[tuple[str, str, str]] = []
    dd = book.drawdown
    if dd >= TOTAL_DRAWDOWN_HALT:
        fired.append(("total_drawdown", "halted", f"drawdown {show(dd)} is at or past "
                                                  f"{show(TOTAL_DRAWDOWN_HALT)}"))
    if book.session_drawdown >= SESSION_DRAWDOWN_HALT:
        fired.append(("session_drawdown", "halted",
                      f"session drawdown {show(book.session_drawdown)} is at or past "
                      f"{show(SESSION_DRAWDOWN_HALT)}"))
    if book.consecutive_losses >= LOSING_STREAK:
        fired.append(("losing_streak", "halted", f"{book.consecutive_losses} consecutive losses"))
    if book.move_sigma <= -SHOCK_SIGMA:
        fired.append(("shock", "halted", f"move {show(book.move_sigma)} sigma"))
    if book.evidence_age > STALE_AFTER:
        fired.append(("stale_evidence", "reduce_only", f"evidence {book.evidence_age} old"))
    if LADDER_FROM <= dd < TOTAL_DRAWDOWN_HALT:
        fired.append(("drawdown_ladder", "reduce_only", f"drawdown {show(dd)} on the ladder"))
    return fired


def expect_multiplier(book: BookFacts) -> Fraction:
    dd = book.drawdown
    if dd >= TOTAL_DRAWDOWN_HALT:
        return Fraction(0)
    if dd >= LADDER_HALF:
        return Fraction(1, 2)
    if dd >= LADDER_FROM:
        return Fraction(3, 4)
    return Fraction(1)


def expect_narrowing(activation: str, verdict: str, open_positions: int) -> str:
    """The verdict the breaker may leave standing. It narrows and never widens."""
    fallback = "reduce" if open_positions > 0 else "no_trade"
    if activation == "halted" and verdict in _OPENS:
        return fallback
    if activation == "reduce_only" and verdict == "trade":
        return fallback
    return verdict


def certify_circuit(state: Any, verdict: str, ruling: Any) -> Certificate:
    """Certify one :func:`argus.risk.circuit.apply` ruling on one book."""
    book = BookFacts.read(state)
    fired = expect_trips(book)
    activation = ("halted" if any(d == "halted" for _, d, _ in fired)
                  else "reduce_only" if fired else "active")
    final = expect_narrowing(activation, verdict, book.open_positions)
    multiplier = expect_multiplier(book)
    expected = f"{activation}, {verdict} -> {final}, multiplier {show(multiplier)}"

    seen_trips = sorted((str(t.rule), str(t.demands)) for t in ruling.trips)
    seen_activation = str(ruling.activation)
    seen_final = str(ruling.verdict)
    seen_multiplier = exact(ruling.risk_multiplier)
    observed = f"{seen_activation}, {ruling.original_verdict} -> {seen_final}, multiplier " \
               f"{show(seen_multiplier)}"

    violations: list[str] = []
    want_trips = sorted((rule, demands) for rule, demands, _ in fired)
    for rule, demands, words in fired:
        if (rule, demands) not in seen_trips:
            violations.append(f"rule {rule} must fire and demand {demands} — {words} — but the "
                              f"ruling does not report it")
    for rule, demands in seen_trips:
        if (rule, demands) not in want_trips:
            violations.append(f"the ruling reports {rule} demanding {demands}, but its "
                              f"precondition for firing does not hold on this book")
    if seen_activation != activation:
        violations.append(f"activation {seen_activation}; the specification resolves the fired "
                          f"rules to {activation}")
    if str(ruling.original_verdict) != verdict:
        violations.append(f"the ruling records the original verdict as "
                          f"{ruling.original_verdict}, not {verdict}")
    if seen_final != final:
        violations.append(f"left {seen_final} standing; under {activation} the specification "
                          f"narrows {verdict} to {final}")
    if seen_final in _OPENS and verdict not in _OPENS:
        violations.append(f"widened {verdict} to {seen_final}: a breaker may never create "
                          f"exposure")
    if seen_multiplier != multiplier:
        violations.append(f"risk multiplier {show(seen_multiplier)}; the ladder gives "
                          f"{show(multiplier)} at drawdown {show(book.drawdown)}")
    first = fired[0] if fired else None
    first_unmet = (f"{first[0]}: unmet precondition — {first[2]}" if first else None)
    return _conclude("circuit breaker", expected, observed, first_unmet, violations)


_PROMOTIONS = {"halted": {"reduce_only"}, "reduce_only": {"halted", "active"},
               "active": {"halted", "reduce_only"}}


def certify_transition(before: str, demanded: str, after: str) -> Certificate:
    """Certify one :meth:`argus.risk.circuit.Breaker.evaluate` move between activations.

    Recovery is stepwise: a halted breaker asked to go straight to active must land on
    reduce_only, and no move may leave the permitted set.
    """
    if demanded == before:
        landed = before
    elif demanded in _PROMOTIONS[before]:
        landed = demanded
    else:
        landed = "reduce_only"
    violations = []
    if after != landed:
        violations.append(f"moved {before} -> {after} on a demand for {demanded}; the "
                          f"specification lands on {landed}")
    if after != before and after not in _PROMOTIONS[before]:
        violations.append(f"{before} -> {after} is not a permitted transition")
    return _conclude("breaker transition", f"{before} -> {landed}", f"{before} -> {after}",
                     None, violations)


# --- sizing and the session throttle --------------------------------------------------------------

MIN_GRADED = 20
MAX_ECE = Fraction("0.15")
MAX_FRACTION = Fraction("0.25")
FIXED_FRACTION = Fraction("0.05")
KELLY_SHARE = Fraction("0.5")
SIZING_TOLERANCE = Fraction(1, 10**24)
"""Sizing is specified in 28-significant-digit decimal arithmetic, not exact rationals; a fraction
within this of the exact value is the same instruction. Ten orders of magnitude below a
basis point."""

SIZING_SPEC: tuple[GateSpec, ...] = (
    GateSpec("calibration_sample", (), f"at least {MIN_GRADED} graded outcomes",
             "otherwise size at the fixed fraction"),
    GateSpec("calibration_error", (), f"expected calibration error at most {show(MAX_ECE)}",
             "otherwise size at the fixed fraction"),
    GateSpec("kelly_floor", (), "none", "a negative Kelly stake is zero, never a reversal"),
    GateSpec("kelly_share", (), "none", f"the Kelly stake is scaled by {show(KELLY_SHARE)}"),
    GateSpec("position_cap", (), "none", f"no fraction above {show(MAX_FRACTION)}"),
    GateSpec("reduce_only_multipliers", (),
             "both multipliers are non-negative and the session throttle is at most one",
             "otherwise refuse to size"),
)


def certify_sizing(
    *, win_probability: float, payoff: Decimal, graded: int, ece: float | None,
    risk_multiplier: Decimal, session_multiplier: Decimal, sizing: Any | None,
    refused: str | None = None,
) -> Certificate:
    """Certify one :func:`argus.risk.sizing.size` result. ``ece`` is the measured calibration
    error the gate reported — a measurement, not a gate, so it is an input here."""
    rm, sm = exact(risk_multiplier), exact(session_multiplier)
    passes = graded >= MIN_GRADED and ece is not None and Fraction(str(ece)) <= MAX_ECE
    p = Fraction(str(win_probability))
    b = exact(payoff) or Fraction(0)
    refusal: tuple[str, str] | None = None
    if rm is None or sm is None or rm < 0 or sm < 0 or sm > 1:
        refusal = ("refused: a multiplier may only reduce",
                   f"sized with multipliers {risk_multiplier} and {session_multiplier}; the "
                   f"specification requires a refusal, because a risk layer may not add exposure")
    elif passes and b > 0 and not 0 <= p <= 1:
        refusal = ("refused: the stated confidence is not a probability",
                   f"sized on win probability {win_probability}, which is not a probability")
    if refusal is not None:
        wanted, departure = refusal
        return _conclude("position sizing", wanted, refused or "sized", wanted,
                         [] if refused else [departure])
    assert rm is not None and sm is not None
    if refused:
        return _conclude("position sizing", "sized", f"refused ({refused})", None,
                         [f"refused ({refused}) where the specification sizes the position"])
    assert sizing is not None
    raw = Fraction(0) if b <= 0 else max(Fraction(0), (p * b - (1 - p)) / b)
    fraction = (min(MAX_FRACTION, raw * KELLY_SHARE * rm * sm) if passes
                else min(MAX_FRACTION, FIXED_FRACTION * rm * sm))
    expected = f"{'half-Kelly' if passes else 'fixed'} fraction {show(fraction)}"
    seen_passed = bool(sizing.gate.passed)
    seen_fraction = exact(sizing.fraction)
    observed = f"{'half-Kelly' if seen_passed else 'fixed'} fraction {show(seen_fraction)}"
    violations: list[str] = []
    if seen_passed != passes:
        why = (f"{graded} graded outcome(s), {MIN_GRADED} needed" if graded < MIN_GRADED
               else f"calibration error {ece} against a ceiling of {show(MAX_ECE)}")
        violations.append(f"calibration gate {'passed' if seen_passed else 'failed'}; the "
                          f"specification {'passes' if passes else 'fails'} it ({why})")
    if int(sizing.gate.graded) != graded:
        violations.append(f"gate reports {sizing.gate.graded} graded outcome(s), not {graded}")
    if seen_fraction is None or abs(seen_fraction - fraction) > SIZING_TOLERANCE:
        violations.append(f"fraction {show(seen_fraction)}; the specification gives "
                          f"{show(fraction)}")
    if seen_fraction is not None and not 0 <= seen_fraction <= MAX_FRACTION:
        violations.append(f"fraction {show(seen_fraction)} is outside 0 to {show(MAX_FRACTION)}")
    first = None if passes else (f"calibration_sample: {graded} of {MIN_GRADED} graded outcomes"
                                 if graded < MIN_GRADED
                                 else f"calibration_error: {ece} above {show(MAX_ECE)}")
    return _conclude("position sizing", expected, observed, first, violations)


THROTTLE_TOLERANCE = Fraction(1, 10**9)
"""The throttle's ratio is computed in binary floating point before it becomes a Decimal."""


def certify_throttle(
    *, measured: bool, baseline_bps: float, expected_bps: float, floor: Decimal,
    multiplier: Decimal,
) -> Certificate:
    """Certify one :func:`argus.risk.session_risk.throttle` result from the path figures it
    reported: unmeasured or quieter-than-baseline paths are not throttled, and a louder path is
    scaled to baseline/expected, never below ``floor`` and never above one."""
    low = exact(floor) or Fraction(0)
    if not measured or baseline_bps <= 0 or expected_bps <= baseline_bps:
        want = Fraction(1)
        why = ("not measured" if not measured else "no baseline" if baseline_bps <= 0
               else "the path is no louder than regular hours")
    else:
        want = max(low, min(Fraction(1), Fraction(baseline_bps) / Fraction(expected_bps)))
        why = f"scaled to {baseline_bps:.4g}/{expected_bps:.4g}"
    got = exact(multiplier)
    violations: list[str] = []
    if got is None or abs(got - want) > THROTTLE_TOLERANCE:
        violations.append(f"multiplier {show(got)}; the specification gives {show(want)} ({why})")
    if got is not None and not low <= got <= 1:
        violations.append(f"multiplier {show(got)} is outside {show(low)} to 1: the throttle may "
                          f"only reduce, and never below its floor")
    return _conclude("session throttle", f"multiplier {show(want)}", f"multiplier {show(got)}",
                     None, violations)


# --- the risk-mode gate ---------------------------------------------------------------------------

MODE_RANK = {"normal": 0, "paper": 1, "reduce_only": 2, "halted": 3}
LAYER_ORDER = ("operator", "venue", "circuit", "posture")

MODE_SPEC: tuple[GateSpec, ...] = (
    GateSpec("resolution", (),
             "every layer is present once, in the order operator, venue, circuit, posture, and "
             "the posture layer demands a mode",
             "the effective mode is the most restrictive demanded (normal < paper < reduce_only "
             "< halted); the binding layer is the first, in that order, to demand it"),
    GateSpec("risk_mode", ("risk_mode",),
             "the verdict carries a quantity (trade, hedge, reduce); a reduction is admitted in "
             "every mode and on either venue; otherwise the venue is paper unless the mode is "
             "normal; reduce_only refuses trade; halted refuses trade and hedge",
             "a refused trade or hedge is narrowed to reduce if positions are open, else "
             "no_trade"),
)


def expect_mode(demands: Sequence[tuple[str, str | None]]) -> tuple[str, str]:
    """(effective mode, binding layer) for demands given as (layer, mode or None) in order."""
    demanded = [(layer, mode) for layer, mode in demands if mode is not None]
    effective = max((mode for _, mode in demanded), key=MODE_RANK.__getitem__)
    binding = next(layer for layer, mode in demanded if mode == effective)
    return effective, binding


def expect_admission(mode: str, verdict: str, venue: str, open_positions: int) -> tuple[bool, str]:
    """(admitted, words) for one order under one effective mode."""
    if verdict not in ("trade", "hedge", "reduce"):
        return False, f"verdict {verdict} carries no quantity; there is no order to admit"
    if venue not in ("paper", "live"):
        return False, f"venue {venue!r} is neither paper nor live"
    if verdict == "reduce":
        return True, "a reduction is admitted in every mode"
    if venue == "live" and mode != "normal":
        return False, f"{mode} admits orders that open exposure on the paper venue only"
    fallback = "reduce" if open_positions > 0 else "no_trade"
    if mode == "halted":
        return False, f"halted refuses {verdict}; it narrows to {fallback}"
    if mode == "reduce_only" and verdict == "trade":
        return False, f"reduce_only refuses new directional risk; it narrows to {fallback}"
    return True, f"{mode} admits {verdict} on {venue}"


def certify_mode(
    demands: Sequence[tuple[str, str | None]], *, verdict: str, venue: str, open_positions: int,
    effective: str, binding: str, admitted: bool,
) -> Certificate:
    """Certify one risk-mode resolution and admission (:mod:`argus.risk.modes`)."""
    violations: list[str] = []
    layers = [layer for layer, _ in demands]
    if tuple(layers) != LAYER_ORDER:
        violations.append(f"layers {layers}; the specification requires {list(LAYER_ORDER)}")
    posture = dict(demands).get("posture")
    if posture is None:
        violations.append("the posture layer demands no mode; the desk has no baseline posture")
        return _conclude("risk mode", "unresolvable", f"{effective} by {binding}", None,
                         violations)
    want_mode, want_binding = expect_mode(demands)
    want_admit, words = expect_admission(want_mode, verdict, venue, open_positions)
    if effective != want_mode:
        violations.append(f"effective mode {effective}; the most restrictive demand is "
                          f"{want_mode}")
    if binding != want_binding:
        violations.append(f"binding layer {binding}; the first layer demanding {want_mode} is "
                          f"{want_binding}")
    if admitted != want_admit:
        violations.append(f"{'admitted' if admitted else 'refused'} {verdict} on {venue} under "
                          f"{want_mode}; the specification {'admits' if want_admit else 'refuses'}"
                          f" it — {words}")
    expected = f"{want_mode} by {want_binding}: {'admit' if want_admit else 'refuse'}"
    observed = f"{effective} by {binding}: {'admit' if admitted else 'refuse'}"
    return _conclude("risk mode", expected, observed,
                     None if want_admit else f"risk_mode: unmet precondition — {words}",
                     violations)


def expect_demands(
    *, operator: str | None, rules_loaded: bool, activation: str, live_enabled: bool,
) -> list[tuple[str, str | None]]:
    """What each layer must demand, from the facts it reads. Unknown or unreadable halts."""
    raw = (operator or "").strip().lower()
    wanted_operator = None if not raw else raw if raw in MODE_RANK else "halted"
    wanted_venue = None if rules_loaded else "halted"
    wanted_circuit = {"active": None, "reduce_only": "reduce_only",
                      "halted": "halted"}.get(activation, "halted")
    return [("operator", wanted_operator), ("venue", wanted_venue), ("circuit", wanted_circuit),
            ("posture", "normal" if live_enabled else "paper")]


def certify_layers(
    observed: Sequence[tuple[str, str | None]], *, operator: str | None, rules_loaded: bool,
    activation: str, live_enabled: bool,
) -> Certificate:
    """Certify the demand each layer derived from what it read (:mod:`argus.risk.modes`)."""
    wanted = expect_demands(operator=operator, rules_loaded=rules_loaded, activation=activation,
                            live_enabled=live_enabled)
    violations = [
        f"layer {layer} demands {seen}; reading operator={operator!r}, rules loaded="
        f"{rules_loaded}, breaker {activation!r}, live={live_enabled}, it must demand {want}"
        for (layer, want), (_, seen) in zip(wanted, observed, strict=False) if want != seen
    ]
    if len(observed) != len(wanted):
        violations.append(f"{len(observed)} layers observed, {len(wanted)} specified")
    shown = ", ".join(f"{layer}={mode}" for layer, mode in wanted)
    seen_shown = ", ".join(f"{layer}={mode}" for layer, mode in observed)
    return _conclude("risk-mode layers", shown, seen_shown, None, violations)


# --- the recorded decision chain ------------------------------------------------------------------

CHAIN_SPEC: tuple[GateSpec, ...] = (
    GateSpec("constitution", (),
             "a risk record exists for the decision",
             "the approved quantity is no larger than the proposed one"),
    GateSpec("protocol", (),
             "the protocol receives the Constitution's approved intent",
             "the protocol may shrink or refuse; it never enlarges"),
    GateSpec("venue_gate", (),
             "an order reaches the venue gate only if a positive quantity was approved, and never "
             "more than was approved",
             "the venue gate may shrink or refuse; it never enlarges"),
    GateSpec("ledger", (),
             "the ledger records exactly what the last gate let through, and a trade only with a "
             "positive quantity the venue gate allowed",
             "none"),
)
"""The live path's order of gates (`paper/runner.py`, the Constitution inside ``desk.run``, then
``enforce_protocol``, then ``guard.check``, then ``ledger.record``) as preconditions on the record
each one leaves. Certifies the *record*, not the code: a gate's output must be the next gate's
input, and nothing downstream may exceed what upstream approved."""


@dataclass(frozen=True, slots=True)
class RecordedChain:
    """What the record says each gate did for one decision. ``None`` where nothing was recorded."""

    seq: int
    symbol: str
    proposed: Decimal | None
    """The Constitution's input (``quantity_before`` in the risk record)."""

    approved: Decimal | None
    """The Constitution's output (``quantity_after``)."""

    constitution_verdict: str | None
    protocol_changed: bool | None
    """Whether the protocol note says it changed anything; ``None`` when there is no note."""

    venue_consulted: bool | None
    venue_allowed: Decimal | None
    """The quantity the venue gate allowed, zero if it refused, ``None`` if not consulted."""

    venue_adjusted: bool
    ledger_verdict: str
    ledger_quantity: Decimal


def certify_chain(chain: RecordedChain) -> Certificate:
    """Replay one recorded decision through :data:`CHAIN_SPEC`, step by step."""
    subject = f"decision seq {chain.seq} ({chain.symbol})"
    ledger_q = exact(chain.ledger_quantity) or Fraction(0)
    observed = f"ledger {chain.ledger_verdict} {show(ledger_q)}"
    if chain.approved is None or chain.proposed is None:
        return Certificate(subject, Status.UNCERTIFIABLE, "a risk record for the decision",
                           observed, None,
                           ("no risk record: the Constitution's step cannot be replayed",))
    proposed = exact(chain.proposed) or Fraction(0)
    approved = exact(chain.approved) or Fraction(0)
    violations: list[str] = []
    first_unmet: str | None = None

    def unmet(step: int, gate: str, words: str) -> None:
        nonlocal first_unmet
        text = f"{gate} (step {step} of 4): unmet precondition — {words}"
        violations.append(text)
        if first_unmet is None:
            first_unmet = text

    if approved > proposed:
        unmet(1, "constitution",
              f"approved {show(approved)} exceeds the {show(proposed)} proposed")
    # The protocol's output is what reaches the venue gate. With "nothing changed" it equals the
    # protocol's input, which the specification requires to be the Constitution's approval.
    if chain.venue_consulted:
        reached = exact(chain.venue_allowed) if not chain.venue_adjusted else None
        if approved <= 0:
            unmet(3, "venue_gate",
                  f"an order reached the venue gate although the Constitution approved "
                  f"{show(approved)} ({chain.constitution_verdict})")
        elif reached is not None and reached > approved:
            unmet(3, "venue_gate",
                  f"the venue gate was asked for {show(reached)}, more than the "
                  f"{show(approved)} the Constitution approved")
        allowed = exact(chain.venue_allowed) or Fraction(0)
        if ledger_q != allowed:
            unmet(4, "ledger", f"the ledger records {show(ledger_q)} but the venue gate let "
                               f"{show(allowed)} through")
    else:
        if ledger_q > 0:
            unmet(4, "ledger",
                  f"the ledger records {show(ledger_q)} although no venue ruling was recorded "
                  f"for this decision")
        if ledger_q > approved:
            unmet(4, "ledger", f"the ledger records {show(ledger_q)}, more than the "
                               f"{show(approved)} the Constitution approved")
    if chain.ledger_verdict in _OPENS and ledger_q <= 0:
        unmet(4, "ledger", f"the ledger records verdict {chain.ledger_verdict} with no quantity")
    if (chain.ledger_verdict in _OPENS and chain.constitution_verdict is not None
            and chain.constitution_verdict not in _OPENS):
        unmet(4, "ledger", f"the ledger records {chain.ledger_verdict} but the Constitution "
                           f"ruled {chain.constitution_verdict}")
    expected = f"ledger at most {show(approved)} ({chain.constitution_verdict})"
    return _conclude(subject, expected, observed, first_unmet, violations)


__all__ = [
    "CHAIN_SPEC",
    "CIRCUIT_SPEC",
    "LAYER_ORDER",
    "MODE_RANK",
    "MODE_SPEC",
    "SIZING_SPEC",
    "VENUE_GOALS",
    "VENUE_SPEC",
    "BookFacts",
    "Certificate",
    "GateSpec",
    "ObservedVenue",
    "RecordedChain",
    "Status",
    "StepCheck",
    "VenueOrder",
    "VenueRules",
    "Window",
    "certify_chain",
    "certify_circuit",
    "certify_layers",
    "certify_mode",
    "certify_sizing",
    "certify_throttle",
    "certify_transition",
    "certify_venue",
    "exact",
    "expect_admission",
    "expect_demands",
    "expect_mode",
    "expect_narrowing",
    "expect_trips",
    "expect_venue",
    "observe_venue",
    "raised",
    "show",
    "step_down",
]
