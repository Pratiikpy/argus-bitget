"""The guard's self-check: the same function that enforces, answering pass/fail only."""

from __future__ import annotations

import copy
from decimal import Decimal
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from argus.eval import guard_selfcheck
from argus.execution.guard import (
    Denial,
    Guard,
    RateLimiter,
    self_check_request,
    validate,
    would_pass,
)

NVDA = guard_selfcheck.instruments()["nvda"]


def test_self_check_equals_enforcement_over_the_whole_swept_order_space() -> None:
    report = guard_selfcheck.sweep()
    assert report["orders_swept"] == 190_944
    assert report["disagreement_count"] == 0, report["disagreements_first_20"]
    assert report["self_check_mutated_window_count"] == 0
    assert report["enforcement_raised"] == 0
    # Both outcomes are reached, so the identity is not trivially "always False".
    assert report["self_check_pass"] > 0 and report["self_check_fail"] > 0


decimals = st.one_of(
    st.none(),
    st.decimals(min_value=Decimal("-10"), max_value=Decimal("200000"), places=4),
    st.sampled_from([Decimal("NaN"), Decimal("Infinity"), Decimal("-Infinity")]),
)


@settings(max_examples=3000, deadline=None)
@given(
    quantity=st.decimals(allow_nan=True, allow_infinity=True, places=5),
    price=decimals, reference=decimals, balance=decimals,
    side=st.sampled_from(["buy", "sell", "SELL", "hold"]),
    stamps=st.lists(st.floats(min_value=900.0, max_value=1000.0), max_size=12),
)
def test_self_check_equals_enforcement_on_random_orders(
    quantity: Decimal, price: Decimal | None, reference: Decimal | None,
    balance: Decimal | None, side: str, stamps: list[float],
) -> None:
    limiter = RateLimiter(max_orders=10, window_seconds=60.0)
    limiter._stamps.extend(sorted(stamps))
    before = list(limiter._stamps)
    kwargs = {"quantity": quantity, "price": price, "reference_price": reference, "side": side,
              "available_balance": balance, "now": 1000.0}
    probe = would_pass(NVDA, limiter=limiter, **kwargs)  # type: ignore[arg-type]
    assert list(limiter._stamps) == before
    assert probe == validate(NVDA, limiter=copy.deepcopy(limiter), **kwargs).allowed  # type: ignore[arg-type]


def test_a_self_check_does_not_spend_the_slot_it_asks_about() -> None:
    measured = guard_selfcheck.slot_consumption()
    assert measured["self_check_then_enforce"] == [True, True]
    # The counterfactual: a check that recorded would deny the very order it approved.
    assert measured["recording_check_then_enforce"] == [True, False]


def test_non_finite_inputs_are_a_named_denial_not_an_exception_or_a_maximum_order() -> None:
    infinite = validate(NVDA, quantity=Decimal("Infinity"), reference_price=Decimal("100"))
    assert infinite.denial is Denial.NOT_A_NUMBER and infinite.quantity == 0
    nan_price = validate(NVDA, quantity=Decimal("1"), price=Decimal("NaN"))
    assert nan_price.denial is Denial.NOT_A_NUMBER
    assert would_pass(NVDA, quantity=Decimal("sNaN")) is False


def test_the_guard_level_self_check_shares_the_unknown_instrument_denial() -> None:
    guard = Guard(instruments={"NVDAUSDT": NVDA})
    assert guard.self_check("ZYXQUSDT", quantity=Decimal("1")) is False
    assert guard.self_check("NVDAUSDT", quantity=Decimal("1"),
                            reference_price=Decimal("100")) is True
    assert guard.limiter.in_window == 0
    assert guard.check("NVDAUSDT", quantity=Decimal("1"),
                       reference_price=Decimal("100")).allowed
    assert guard.limiter.in_window == 1


def test_the_endpoint_contract_is_pass_or_fail_and_nothing_else() -> None:
    guard = Guard(instruments={"NVDAUSDT": NVDA})
    ok = {"symbol": "NVDAUSDT", "quantity": "1", "price": "100", "reference_price": "100"}
    assert self_check_request(guard, ok) == {"pass": True}
    assert self_check_request(guard, {**ok, "quantity": "0.001"}) == {"pass": False}
    # Malformed requests fail closed, never raise and never default.
    bad: Any
    for bad in (None, [], "order", {}, {"symbol": "NVDAUSDT"},
                {**ok, "quantity": 1.0}, {**ok, "quantity": True}, {**ok, "quantity": "abc"},
                {**ok, "quantity": "NaN"}, {**ok, "leverage": "100"}, {**ok, "symbol": 7},
                {**ok, "side": ["buy"]}, {**ok, "price": {"x": 1}}):
        assert self_check_request(guard, bad) == {"pass": False}, bad
    assert guard.limiter.in_window == 0
