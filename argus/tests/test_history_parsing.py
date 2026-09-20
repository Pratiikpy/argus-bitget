"""Candle parsing — and the regression that took the hedge layer down for a whole cycle.

`market/history.py` had no test file at all, which is how the defect below reached a scheduled run.

**What happened.** A correctness fix made every candle figure raise unless it was a *positive*
number, on the rule that "a missing price is not zero". That rule is right for a price. It was
applied to all four candle series, and `CandleType.PREMIUM` is a **signed fraction** — negative
whenever the token trades below its index. So `fetch_basis` raised on all twelve instruments, and
the 19:00 cycle's hedge-effectiveness step exited 1 with "no basis history" for the entire universe.

The two halves of the rule are separable and these tests keep them apart:

* **absence is not zero** — holds for *every* series, including premium;
* **a number must be positive** — a fact about prices only.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from argus.market.history import CandleType, HistoryError, _dec, _number, _px


class TestAbsenceIsNotZeroOnEverySeries:
    """The half of the rule that survived, and must keep applying to premium too."""

    @pytest.mark.parametrize("missing", [None, "", "null"])
    def test_a_missing_figure_raises(self, missing: object) -> None:
        with pytest.raises(HistoryError, match="is not zero"):
            _number(missing, field="close", symbol="NVDAUSDT")
        with pytest.raises(HistoryError, match="is not zero"):
            _px(missing, field="close", symbol="NVDAUSDT")

    @pytest.mark.parametrize("junk", ["abc", "1.2.3", object()])
    def test_an_unparseable_figure_raises(self, junk: object) -> None:
        with pytest.raises(HistoryError, match="is not a number"):
            _number(junk, field="close", symbol="NVDAUSDT")
        with pytest.raises(HistoryError, match="is not a number"):
            _px(junk, field="close", symbol="NVDAUSDT")

    def test_the_symbol_and_field_are_named_in_the_error(self) -> None:
        """A parser failure must say which instrument and which field, or a whole-universe outage
        reads as one undifferentiated wall of errors — which is exactly how it read."""
        with pytest.raises(HistoryError, match=r"SQQQUSDT.*high"):
            _number(None, field="high", symbol="SQQQUSDT")


class TestPricesMustBePositive:
    def test_a_positive_price_parses(self) -> None:
        assert _px("211.04", field="close", symbol="NVDAUSDT") == Decimal("211.04")

    @pytest.mark.parametrize("bad", ["0", "-1", "-0.0001"])
    def test_a_non_positive_price_raises(self, bad: str) -> None:
        with pytest.raises(HistoryError, match="not a positive price"):
            _px(bad, field="close", symbol="NVDAUSDT")


class TestPremiumIsSignedAndMustNotBeRejected:
    """The regression itself."""

    @pytest.mark.parametrize(
        "value",
        ["-0.000191185946", "-0.004093188555", "0", "-0.00063550279", "0.001009318244"],
    )
    def test_a_signed_premium_parses(self, value: str) -> None:
        """Every one of these is a real value from a live NVDAUSDT premium candle. Under the
        defect, the four non-positive ones raised and took `fetch_basis` down with them."""
        assert _number(value, field="close", symbol="NVDAUSDT") == Decimal(value)

    def test_the_price_parser_would_still_reject_them(self) -> None:
        """Pins the distinction rather than assuming it: the price guard is not merely unused on
        premium, it would actively refuse real premium data."""
        for value in ("-0.000191185946", "0", "-0.004093188555"):
            with pytest.raises(HistoryError):
                _px(value, field="close", symbol="NVDAUSDT")

    def test_fetch_routes_premium_to_the_signed_parser(self) -> None:
        """The wiring, checked at the source. A correct pair of parsers wired to the wrong series
        is the same outage."""
        import inspect

        from argus.market import history

        source = inspect.getsource(history.fetch)
        assert "_number if candle_type is CandleType.PREMIUM else _px" in source

    def test_every_other_series_keeps_the_price_guard(self) -> None:
        assert CandleType.PREMIUM is not CandleType.MARKET
        for series in (CandleType.MARKET, CandleType.INDEX, CandleType.MARK):
            assert series is not CandleType.PREMIUM


class TestVolumeKeepsItsDefault:
    """Volume is the one figure whose zero is a real reading — a quiet hour."""

    def test_absent_volume_is_zero(self) -> None:
        assert _dec(None) == Decimal("0")
        assert _dec("") == Decimal("0")

    def test_unparseable_volume_is_zero(self) -> None:
        assert _dec("abc") == Decimal("0")

    def test_real_volume_parses(self) -> None:
        assert _dec("1234.5") == Decimal("1234.5")
