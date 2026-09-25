"""Pre-trade guard: the venue's own rules, enforced before we ask, and only ever downward."""

from __future__ import annotations

import os
from decimal import Decimal
from typing import Any

import pytest

from argus.cost.model import CostModel
from argus.execution.guard import (
    GUARD_GATES,
    RULE_GATES,
    Denial,
    GateVerdict,
    Guard,
    GuardError,
    Instrument,
    RateLimiter,
    exact_precision,
    fetch_instruments,
    quantise_down,
    validate,
)

LIVE = os.environ.get("ARGUS_LIVE_VENUE") == "1"
live_only = pytest.mark.skipif(not LIVE, reason="set ARGUS_LIVE_VENUE=1 to hit the venue")

# Verbatim from GET /api/v3/market/instruments?category=USDT-FUTURES&symbol=NVDAUSDT on
# 2026-09-13. Kept as a fixture so the tests pin the real shape, not an invented one.
NVDA_ROW: dict[str, Any] = {
    "symbol": "NVDAUSDT", "category": "USDT-FUTURES", "baseCoin": "NVDA", "quoteCoin": "USDT",
    "symbolType": "stock", "isRwa": "YES", "buyLimitPriceRatio": "0.02",
    "sellLimitPriceRatio": "0.02", "makerFeeRate": "0.0002", "takerFeeRate": "0.0006",
    "minOrderQty": "0.01", "maxOrderQty": "52000", "pricePrecision": "2",
    "quantityPrecision": "2", "priceMultiplier": "0.01", "quantityMultiplier": "0.01",
    "type": "perpetual", "minOrderAmount": "5", "status": "online",
}


def _instrument(**over: Any) -> Instrument:
    row = dict(NVDA_ROW)
    row.update(over)
    return Instrument.from_payload(row)


class TestTheSpecIsReadNotAssumed:
    def test_it_parses_the_real_venue_row(self) -> None:
        got = _instrument()
        assert got.symbol == "NVDAUSDT" and got.is_online and got.is_rwa
        assert got.min_order_amount == Decimal("5")
        assert got.max_order_qty == Decimal("52000")

    def test_a_missing_rule_raises_rather_than_defaulting(self) -> None:
        """A silently defaulted minimum is how an order passes a check that never happened."""
        for key in ("minOrderQty", "minOrderAmount", "priceMultiplier", "status"):
            row = dict(NVDA_ROW)
            row[key] = ""
            with pytest.raises(GuardError, match="refusing to substitute a default"):
                Instrument.from_payload(row)

    def test_the_venue_fee_confirms_our_cost_model(self) -> None:
        """The venue publishes takerFeeRate 0.0006; our model charges 12bps for a round trip."""
        assert _instrument().round_trip_taker_bps == CostModel.bitget_perp().round_trip_bps()

    def test_the_rwa_flag_is_read_from_the_venue(self) -> None:
        assert _instrument().is_rwa
        assert not _instrument(isRwa="NO").is_rwa

    def test_an_offline_instrument_is_not_online(self) -> None:
        assert not _instrument(status="offline").is_online


class TestQuantiseAlwaysRoundsDown:
    def test_it_rounds_down_not_to_nearest(self) -> None:
        """Nomos's lesson: rounding up can cross the ceiling that was just checked."""
        assert quantise_down(Decimal("1.999"), Decimal("0.01")) == Decimal("1.99")
        assert quantise_down(Decimal("1.991"), Decimal("0.01")) == Decimal("1.99")

    def test_an_exact_multiple_is_unchanged(self) -> None:
        assert quantise_down(Decimal("2.00"), Decimal("0.01")) == Decimal("2.00")

    def test_a_value_below_one_step_becomes_zero(self) -> None:
        assert quantise_down(Decimal("0.004"), Decimal("0.01")) == Decimal("0")

    def test_a_non_positive_step_is_refused(self) -> None:
        with pytest.raises(GuardError, match="step must be positive"):
            quantise_down(Decimal("1"), Decimal("0"))

    def test_it_never_returns_more_than_it_was_given(self) -> None:
        for value in ("0.001", "1", "1.4999", "51999.999", "52000"):
            got = quantise_down(Decimal(value), Decimal("0.01"))
            assert got <= Decimal(value)


class TestTheGuardOnlyEverShrinks:
    def test_a_clean_order_passes_untouched(self) -> None:
        got = validate(_instrument(), quantity=Decimal("1.00"), reference_price=Decimal("200"))
        assert got.allowed and got.quantity == Decimal("1.00") and not got.adjusted

    def test_a_sub_step_quantity_is_rounded_down_and_flagged(self) -> None:
        got = validate(_instrument(), quantity=Decimal("1.009"), reference_price=Decimal("200"))
        assert got.allowed and got.quantity == Decimal("1.00") and got.adjusted

    def test_an_oversized_order_is_cut_to_the_venue_maximum(self) -> None:
        got = validate(_instrument(), quantity=Decimal("99999"), reference_price=Decimal("200"))
        assert got.allowed and got.quantity == Decimal("52000") and got.adjusted

    def test_it_never_rounds_a_size_up_to_reach_the_minimum(self) -> None:
        """Enlarging an order to satisfy a minimum would be the guard authoring size."""
        got = validate(_instrument(), quantity=Decimal("0.004"), reference_price=Decimal("200"))
        assert not got.allowed
        assert got.denial is Denial.QUANTITY_BELOW_MINIMUM
        assert "may not round a size up" in got.reason

    def test_a_denied_order_carries_zero_quantity(self) -> None:
        got = validate(_instrument(), quantity=Decimal("0.001"), reference_price=Decimal("200"))
        assert got.quantity == Decimal("0")

    def test_the_output_quantity_never_exceeds_the_input(self) -> None:
        for value in ("0.01", "1", "1.337", "51999.99", "80000"):
            got = validate(
                _instrument(), quantity=Decimal(value), reference_price=Decimal("200")
            )
            assert got.quantity <= Decimal(value)


class TestVenueRulesAreEnforced:
    def test_an_offline_instrument_is_denied_by_name(self) -> None:
        got = validate(_instrument(status="offline"), quantity=Decimal("1"))
        assert got.denial is Denial.INSTRUMENT_OFFLINE

    def test_a_notional_below_the_venue_minimum_is_denied(self) -> None:
        got = validate(_instrument(), quantity=Decimal("0.01"), reference_price=Decimal("100"))
        assert got.denial is Denial.NOTIONAL_BELOW_MINIMUM
        assert "5" in got.reason

    def test_a_notional_at_the_minimum_passes(self) -> None:
        got = validate(_instrument(), quantity=Decimal("0.05"), reference_price=Decimal("100"))
        assert got.allowed

    def test_a_price_outside_the_band_is_denied(self) -> None:
        got = validate(
            _instrument(), quantity=Decimal("1"), price=Decimal("250"),
            reference_price=Decimal("200"), side="buy",
        )
        assert got.denial is Denial.PRICE_OUTSIDE_BAND
        assert "2.00%" in got.reason or "±" in got.reason

    def test_a_price_inside_the_band_passes(self) -> None:
        got = validate(
            _instrument(), quantity=Decimal("1"), price=Decimal("203"),
            reference_price=Decimal("200"),
        )
        assert got.allowed

    def test_a_price_is_snapped_to_the_venue_step(self) -> None:
        got = validate(
            _instrument(), quantity=Decimal("1"), price=Decimal("200.007"),
            reference_price=Decimal("200"),
        )
        assert got.price == Decimal("200.00") and got.adjusted

    def test_a_non_positive_price_is_denied(self) -> None:
        got = validate(_instrument(), quantity=Decimal("1"), price=Decimal("0"))
        assert got.denial is Denial.PRICE_NOT_POSITIVE

    def test_an_order_larger_than_the_balance_is_denied(self) -> None:
        got = validate(
            _instrument(), quantity=Decimal("10"), reference_price=Decimal("200"),
            available_balance=Decimal("100"),
        )
        assert got.denial is Denial.INSUFFICIENT_BALANCE

    def test_an_affordable_order_passes(self) -> None:
        got = validate(
            _instrument(), quantity=Decimal("0.1"), reference_price=Decimal("200"),
            available_balance=Decimal("100"),
        )
        assert got.allowed

    def test_the_sell_band_uses_the_sell_ratio(self) -> None:
        instrument = _instrument(sellLimitPriceRatio="0.5")
        got = validate(
            instrument, quantity=Decimal("1"), price=Decimal("250"),
            reference_price=Decimal("200"), side="sell",
        )
        assert got.allowed


class TestTheRateLimiter:
    def test_it_allows_up_to_the_cap(self) -> None:
        limiter = RateLimiter(max_orders=3, window_seconds=60)
        for i in range(3):
            got = validate(
                _instrument(), quantity=Decimal("1"), reference_price=Decimal("200"),
                limiter=limiter, now=float(i),
            )
            assert got.allowed, i

    def test_it_denies_the_one_past_the_cap(self) -> None:
        limiter = RateLimiter(max_orders=2, window_seconds=60)
        for i in range(2):
            validate(_instrument(), quantity=Decimal("1"), reference_price=Decimal("200"),
                     limiter=limiter, now=float(i))
        got = validate(_instrument(), quantity=Decimal("1"), reference_price=Decimal("200"),
                       limiter=limiter, now=3.0)
        assert got.denial is Denial.RATE_LIMIT
        assert "cap of 2" in got.reason

    def test_the_window_slides_rather_than_resetting_in_buckets(self) -> None:
        """A fixed bucket lets twice the limit through at a boundary."""
        limiter = RateLimiter(max_orders=2, window_seconds=10)
        validate(_instrument(), quantity=Decimal("1"), reference_price=Decimal("200"),
                 limiter=limiter, now=0.0)
        validate(_instrument(), quantity=Decimal("1"), reference_price=Decimal("200"),
                 limiter=limiter, now=1.0)
        assert limiter.would_exceed(now=5.0)
        assert not limiter.would_exceed(now=12.0)

    def test_a_denied_order_does_not_consume_a_slot(self) -> None:
        limiter = RateLimiter(max_orders=2, window_seconds=60)
        validate(_instrument(), quantity=Decimal("0.0001"), reference_price=Decimal("200"),
                 limiter=limiter, now=0.0)
        assert limiter.in_window == 0

    def test_the_limiter_counts_only_what_it_allowed(self) -> None:
        limiter = RateLimiter(max_orders=5, window_seconds=60)
        validate(_instrument(), quantity=Decimal("1"), reference_price=Decimal("200"),
                 limiter=limiter, now=0.0)
        assert limiter.in_window == 1


class TestTheGuardFacade:
    def test_an_unknown_symbol_is_denied_by_name(self) -> None:
        guard = Guard(instruments={"NVDAUSDT": _instrument()})
        got = guard.check("NOTLISTED", quantity=Decimal("1"))
        assert got.denial is Denial.INSTRUMENT_UNKNOWN
        assert "no rules are known" in got.reason

    def test_a_known_symbol_is_validated(self) -> None:
        guard = Guard(instruments={"NVDAUSDT": _instrument()})
        assert guard.check(
            "NVDAUSDT", quantity=Decimal("1"), reference_price=Decimal("200")
        ).allowed

    def test_the_facade_shares_one_limiter_across_symbols(self) -> None:
        guard = Guard(
            instruments={"NVDAUSDT": _instrument(), "TSLAUSDT": _instrument(symbol="TSLAUSDT")},
            limiter=RateLimiter(max_orders=1, window_seconds=60),
        )
        assert guard.check("NVDAUSDT", quantity=Decimal("1"),
                           reference_price=Decimal("200"), now=0.0).allowed
        second = guard.check("TSLAUSDT", quantity=Decimal("1"),
                             reference_price=Decimal("200"), now=1.0)
        assert second.denial is Denial.RATE_LIMIT

    def test_a_ruling_serialises(self) -> None:
        guard = Guard(instruments={"NVDAUSDT": _instrument()})
        got = guard.check("NVDAUSDT", quantity=Decimal("1"),
                          reference_price=Decimal("200")).as_dict()
        assert got["allowed"] is True and got["denial"] == "none"

    def test_every_denial_state_renders_its_name(self) -> None:
        for denial in Denial:
            assert str(denial)


class TestThePermissionCheckTrace:
    """Every ruling carries the whole gate stack, in order — letta-code's staged trace."""

    def test_an_allowed_order_lists_every_gate_once_in_order(self) -> None:
        got = validate(_instrument(), quantity=Decimal("1.009"), reference_price=Decimal("200"))
        assert got.trace.gates == RULE_GATES
        verdicts = {e.gate: e.verdict for e in got.trace.events}
        assert verdicts["quantity_step"] is GateVerdict.ADJUST
        assert verdicts["price_band"] is GateVerdict.SKIP
        assert got.trace.deciding is None
        assert GateVerdict.NOT_REACHED not in verdicts.values()

    def test_a_refusal_names_one_deciding_gate_and_the_rest_are_not_reached(self) -> None:
        got = validate(_instrument(), quantity=Decimal("0.004"), reference_price=Decimal("200"))
        deciding = got.trace.deciding
        assert deciding is not None and deciding.gate == "quantity_floor"
        assert deciding.reason == got.reason
        after = got.trace.events[RULE_GATES.index("quantity_floor") + 1:]
        assert after and all(e.verdict is GateVerdict.NOT_REACHED for e in after)
        assert sum(e.verdict is GateVerdict.DENY for e in got.trace.events) == 1

    def test_the_session_guard_puts_the_instrument_lookup_first(self) -> None:
        guard = Guard(instruments={"NVDAUSDT": _instrument()})
        known = guard.check("NVDAUSDT", quantity=Decimal("1"), reference_price=Decimal("200"))
        assert known.trace.gates == GUARD_GATES
        assert known.trace.events[0].verdict is GateVerdict.PASS
        unknown = guard.check("NOTLISTED", quantity=Decimal("1"))
        assert unknown.trace.gates == GUARD_GATES
        assert unknown.trace.deciding is not None
        assert unknown.trace.deciding.gate == "instrument_known"

    def test_the_trace_serialises_and_renders(self) -> None:
        got = validate(_instrument(), quantity=Decimal("1"), reference_price=Decimal("200"))
        blob = got.as_dict()["trace"]
        assert [row["gate"] for row in blob] == list(RULE_GATES)
        assert set(blob[0]) == {"gate", "input", "verdict", "reason"}
        assert got.explain().splitlines()[1].strip().startswith("1. finite_inputs: pass")

    def test_the_trace_records_and_never_decides(self) -> None:
        """Rulings are unchanged by the trace: the one-line render the runner writes is the same
        text it wrote before the trace existed, for an allowed and a refused order."""
        allowed = validate(_instrument(), quantity=Decimal("1"), reference_price=Decimal("200"))
        assert allowed.render() == "[guard] allowed: 1"
        refused = validate(_instrument(status="offline"), quantity=Decimal("1"))
        assert refused.render() == (
            "[guard] DENIED instrument_offline: NVDAUSDT is offline, not online"
        )


class TestTheDefectsTheCertifierFound:
    """Three orders the guard allowed against its own contract before 2026-09-25 (S20)."""

    def test_a_31_digit_quantity_is_never_rounded_up(self) -> None:
        tiny = Decimal("0.0099999999999999999999999999999")
        assert quantise_down(tiny, Decimal("0.01")) == 0
        got = validate(_instrument(), quantity=tiny, reference_price=Decimal("1000"))
        assert got.denial is Denial.QUANTITY_BELOW_MINIMUM
        near_max = Decimal("51999.999999999999999999999999999")
        capped = validate(_instrument(), quantity=near_max, reference_price=Decimal("100"))
        assert capped.allowed and capped.quantity == Decimal("51999.99") <= near_max

    def test_products_are_exact_at_the_notional_and_balance_limits(self) -> None:
        under = validate(_instrument(), quantity=Decimal("0.05"),
                         reference_price=Decimal("99.99999999999999999999999999999"))
        assert under.denial is Denial.NOTIONAL_BELOW_MINIMUM
        over = validate(_instrument(), quantity=Decimal("1"),
                        reference_price=Decimal("100.00000000000000000000000000001"),
                        available_balance=Decimal("100"))
        assert over.denial is Denial.INSUFFICIENT_BALANCE
        band = validate(_instrument(), quantity=Decimal("1"), price=Decimal("102"),
                        reference_price=Decimal("99.99999999999999999999999999999"))
        assert band.denial is Denial.PRICE_OUTSIDE_BAND

    def test_a_sub_tick_limit_price_is_refused_not_zeroed(self) -> None:
        got = validate(_instrument(), quantity=Decimal("1"), price=Decimal("0.005"))
        assert got.denial is Denial.PRICE_NOT_POSITIVE
        assert got.trace.deciding is not None and got.trace.deciding.gate == "price_step"

    def test_only_buy_and_sell_are_sides(self) -> None:
        for side in ("hold", "", "long", "buy "):
            got = validate(_instrument(), quantity=Decimal("1"), reference_price=Decimal("200"),
                           side=side)
            assert got.denial is Denial.SIDE_UNKNOWN, side
        for side in ("buy", "sell", "SELL", "Buy"):
            assert validate(_instrument(), quantity=Decimal("1"),
                            reference_price=Decimal("200"), side=side).allowed, side

    def test_ordinary_results_keep_their_representation(self) -> None:
        """The runner writes ``quantity`` into the hashed ledger as a string; exact arithmetic
        must not turn ``5`` into ``5.00``."""
        assert str(quantise_down(Decimal("5"), Decimal("0.01"))) == "5"
        assert str(quantise_down(Decimal("1.999"), Decimal("0.01"))) == "1.99"
        assert str(quantise_down(Decimal("0.004"), Decimal("0.01"))) == "0.00"

    def test_the_context_widens_with_the_operands(self) -> None:
        assert exact_precision(Decimal("1"), Decimal("0.01")) == 28
        assert exact_precision(Decimal("0.0099999999999999999999999999999")) > 31


@live_only
class TestAgainstTheLiveVenue:
    def test_the_venue_publishes_rules_for_every_rtoken(self) -> None:
        from argus.market.bitget import RTOKEN_SYMBOLS

        got = fetch_instruments()
        missing = [s for s in RTOKEN_SYMBOLS if s not in got]
        assert not missing, missing

    def test_the_live_fee_matches_our_cost_model(self) -> None:
        got = fetch_instruments(symbol="NVDAUSDT")["NVDAUSDT"]
        assert got.round_trip_taker_bps == CostModel.bitget_perp().round_trip_bps()

    def test_the_rtokens_are_flagged_as_rwa_by_the_venue(self) -> None:
        assert fetch_instruments(symbol="NVDAUSDT")["NVDAUSDT"].is_rwa
