"""Does the self-check say exactly what enforcement does? Swept, not sampled.

`execution/guard.py` gained a self-check on 2026-09-25 — :func:`~argus.execution.guard.would_pass`,
after mle-bench's ``/validate`` endpoint (MIT, ``environment/grading_server.py:15-19``), which
answers "is this submission valid?" by running the *real* grader and returning only pass/fail. The
claim that makes such a check worth having is a single identity:

    would_pass(x) == validate(x).allowed        for every order x

and a second one that makes it safe to call as often as an agent likes:

    calling would_pass(x) changes nothing that a later validate(y) can observe.

Both are checked here over the full cross product of the axes on which the eight venue rules
branch — instrument state, quantity around every step, minimum and maximum, price around the step
and both band edges, the reference price, side, balance, and the rate window empty, one below the
cap, at the cap, and with stamps sitting exactly on the window boundary. Non-finite and signalling
decimals are included on purpose. **The first run found the enforcing path raising on them** —
13,800 of 190,944 orders ended in ``decimal.InvalidOperation`` rather than a ruling — which is why
``Denial.NOT_A_NUMBER`` now exists; ``enforcement_raised`` is reported so a regression shows.

**What this does not prove.** The sweep covers the branch points of the rules as written today. A
new rule added to :func:`~argus.execution.guard._rule` is covered automatically, because both
paths call it; a new *input axis* is not, until it is added to :func:`order_space`.

    python -m argus.eval.guard_selfcheck
"""

from __future__ import annotations

import copy
import itertools
from collections.abc import Iterator
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from argus.eval import artefact
from argus.execution.guard import (
    GuardError,
    Instrument,
    RateLimiter,
    validate,
    would_pass,
)

REPORT_PATH = Path(__file__).resolve().parents[3] / "data" / "guard_selfcheck.json"

NOW = 1_000.0
"""A fixed clock, so a rate-window state means the same thing on every run."""

NVDA_ROW: dict[str, Any] = {
    "symbol": "NVDAUSDT", "isRwa": "YES", "buyLimitPriceRatio": "0.02",
    "sellLimitPriceRatio": "0.02", "makerFeeRate": "0.0002", "takerFeeRate": "0.0006",
    "minOrderQty": "0.01", "maxOrderQty": "52000", "pricePrecision": "2",
    "quantityPrecision": "2", "priceMultiplier": "0.01", "quantityMultiplier": "0.01",
    "minOrderAmount": "5", "status": "online",
}
"""The real NVDAUSDT row read from ``GET /api/v3/market/instruments`` on 2026-09-13 (the same
fixture ``tests/test_guard.py`` pins)."""


def instruments() -> dict[str, Instrument]:
    """Four instruments, each reaching a branch the others do not."""
    def build(**over: str) -> Instrument:
        row = dict(NVDA_ROW)
        row.update(over)
        return Instrument.from_payload(row)

    return {
        "nvda": build(),
        "offline": build(status="limit_open"),
        "coarse": build(quantityMultiplier="1", minOrderQty="1", maxOrderQty="5",
                        minOrderAmount="50", buyLimitPriceRatio="0.05",
                        sellLimitPriceRatio="0.01"),
        # A ceiling below one step: stepping the maximum down reaches zero, the only way into the
        # QUANTITY_ABOVE_MAXIMUM denial.
        "sub_step_max": build(maxOrderQty="0.005"),
    }


QUANTITIES: tuple[str, ...] = (
    "-1", "0", "0.005", "0.01", "0.015", "0.05", "0.1", "1", "3.7", "5", "6",
    "51999.999", "52000", "52000.01", "1000000", "NaN", "Infinity",
)
PRICES: tuple[str | None, ...] = (
    None, "-1", "0", "97.99", "98", "99.999", "100", "101.99", "102", "102.01", "104.9",
    "1000000000", "sNaN",
)
REFERENCES: tuple[str | None, ...] = (None, "0", "100")
SIDES: tuple[str, ...] = ("buy", "sell", "SELL")
BALANCES: tuple[str | None, ...] = (None, "1", "1000", "1000000000000")


def limiter_states() -> dict[str, RateLimiter | None]:
    """The rate window's branch points, all at :data:`NOW` with a 60s window and a cap of 10."""
    def window(*stamps: float) -> RateLimiter:
        limiter = RateLimiter(max_orders=10, window_seconds=60.0)
        limiter._stamps.extend(stamps)
        return limiter

    return {
        "none": None,
        "empty": window(),
        "one_below_cap": window(*[NOW - 1.0] * 9),
        "at_cap": window(*[NOW - 1.0] * 10),
        "at_cap_but_half_expired": window(*[NOW - 120.0] * 5, *[NOW - 1.0] * 5),
        "at_cap_on_the_boundary": window(*[NOW - 60.0] * 10),
    }


@dataclass(frozen=True, slots=True)
class Order:
    instrument: str
    quantity: str
    price: str | None
    reference: str | None
    side: str
    balance: str | None
    window: str


def order_space() -> Iterator[Order]:
    for combo in itertools.product(
        instruments(), QUANTITIES, PRICES, REFERENCES, SIDES, BALANCES, limiter_states(),
    ):
        yield Order(*combo)


def _dec(raw: str | None) -> Decimal | None:
    return None if raw is None else Decimal(raw)


@dataclass(frozen=True, slots=True)
class Outcome:
    self_check: bool
    enforced: bool | None
    """``None`` when the enforcing path raised — a ruling nobody got."""

    window_untouched: bool

    @property
    def agrees(self) -> bool:
        return self.self_check == bool(self.enforced)


def judge(order: Order, *, specs: dict[str, Instrument] | None = None) -> Outcome:
    """Ask both paths about one order, each on its own copy of the same rate window."""
    instrument = (specs or instruments())[order.instrument]
    original = limiter_states()[order.window]
    probe, enforce = copy.deepcopy(original), copy.deepcopy(original)
    kwargs: dict[str, Any] = {
        "quantity": Decimal(order.quantity), "price": _dec(order.price),
        "reference_price": _dec(order.reference), "side": order.side,
        "available_balance": _dec(order.balance), "now": NOW,
    }
    passed = would_pass(instrument, limiter=probe, **kwargs)
    untouched = (probe is None and original is None) or (
        probe is not None and original is not None
        and list(probe._stamps) == list(original._stamps)
    )
    try:
        enforced: bool | None = validate(instrument, limiter=enforce, **kwargs).allowed
    except (GuardError, ArithmeticError):
        enforced = None
    return Outcome(passed, enforced, untouched)


def sweep() -> dict[str, Any]:
    """Every order in :func:`order_space`, both identities, and the counts."""
    total = passes = raised = disagreed = touched = 0
    disagreements: list[dict[str, Any]] = []
    mutated: list[dict[str, Any]] = []
    specs = instruments()
    for order in order_space():
        outcome = judge(order, specs=specs)
        total += 1
        passes += outcome.self_check
        raised += outcome.enforced is None
        if not outcome.agrees:
            disagreed += 1
            if len(disagreements) < 20:
                disagreements.append(_row(order))
        if not outcome.window_untouched:
            touched += 1
            if len(mutated) < 20:
                mutated.append(_row(order))
    return {
        "orders_swept": total,
        "self_check_pass": passes,
        "self_check_fail": total - passes,
        "enforcement_raised": raised,
        "disagreement_count": disagreed,
        "disagreements_first_20": disagreements,
        "self_check_mutated_window_count": touched,
        "self_check_mutated_window_first_20": mutated,
        "identity_holds": disagreed == 0,
        "self_check_is_read_only": touched == 0,
    }


def _row(order: Order) -> dict[str, Any]:
    return {
        "instrument": order.instrument, "quantity": order.quantity, "price": order.price,
        "reference": order.reference, "side": order.side, "balance": order.balance,
        "window": order.window,
    }


def slot_consumption() -> dict[str, Any]:
    """Why the self-check must not record: the counterfactual, measured.

    One slot left in the window. A self-check that went through the recording path would spend it,
    and the order it just approved would then be denied for the rate limit it consumed.
    """
    instrument = instruments()["nvda"]
    order: dict[str, Any] = {"quantity": Decimal("1"), "price": Decimal("100"),
                             "reference_price": Decimal("100"), "now": NOW}

    shared = limiter_states()["one_below_cap"]
    assert shared is not None
    ours_then_enforce = (would_pass(instrument, limiter=shared, **order),
                         validate(instrument, limiter=shared, **order).allowed)

    recording = limiter_states()["one_below_cap"]
    assert recording is not None
    naive_then_enforce = (validate(instrument, limiter=recording, **order).allowed,
                          validate(instrument, limiter=recording, **order).allowed)
    return {
        "window": "one slot left of 10",
        "self_check_then_enforce": list(ours_then_enforce),
        "recording_check_then_enforce": list(naive_then_enforce),
    }


def run(path: Path = REPORT_PATH) -> dict[str, Any]:
    report = {
        "what": "would_pass(x) == validate(x).allowed over the swept order space",
        "source": "mle-bench environment/grading_server.py:15-19, mlebench/grade.py:98-124 (MIT)",
        "sweep": sweep(),
        "slot_consumption": slot_consumption(),
    }
    artefact.write(path, report)
    return report


def main() -> int:  # pragma: no cover - CLI
    report = run()
    sweep_ = report["sweep"]
    print(f"orders swept: {sweep_['orders_swept']}  pass: {sweep_['self_check_pass']}  "
          f"fail: {sweep_['self_check_fail']}  enforcement raised: "
          f"{sweep_['enforcement_raised']}")
    print(f"identity holds: {sweep_['identity_holds']}  read-only: "
          f"{sweep_['self_check_is_read_only']}")
    print(f"slot consumption: {report['slot_consumption']}")
    return 0 if sweep_["identity_holds"] and sweep_["self_check_is_read_only"] else 1


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
