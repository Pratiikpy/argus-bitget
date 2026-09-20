"""Instrument Master tests.

Foundation 1. Before this module, "TQQQUSDT is a leveraged ETF" was a claim resting on one
docstring, and the only thing distinguishing a declared identity from a measured behaviour was that
sentence. These tests check the properties that make the distinction real:

* the registry must cover exactly the rToken universe, no more and no less;
* an unregistered symbol is refused, never silently treated as a plain equity;
* a declared leverage target is checked against a measurement, and the check can actually fail —
  it is a real comparison, not a decoration that always reports agreement;
* an instrument with nothing declared reports ``None``, never a fabricated comparison.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from argus.market.bitget import RTOKEN_SYMBOLS
from argus.market.instruments import (
    LEVERAGE_TOLERANCE,
    REGISTRY,
    InstrumentIdentity,
    InstrumentKind,
    InstrumentMasterError,
    identity_of,
    leverage_check,
    registry_as_dict,
)


class TestTheRegistryCoversExactlyTheUniverse:
    def test_every_rtoken_symbol_is_declared(self) -> None:
        missing = [s for s in RTOKEN_SYMBOLS if s not in REGISTRY]
        assert missing == [], f"undeclared symbols: {missing}"

    def test_the_registry_declares_nothing_outside_the_universe(self) -> None:
        extra = [s for s in REGISTRY if s not in RTOKEN_SYMBOLS]
        assert extra == [], f"declared but not in RTOKEN_SYMBOLS: {extra}"

    def test_an_unregistered_symbol_is_refused_not_guessed(self) -> None:
        with pytest.raises(InstrumentMasterError, match="not in the Instrument Master"):
            identity_of("DOGEUSDT")


class TestEveryEntryCarriesItsSource:
    """A declared fact with no source is an assertion wearing a database's clothes."""

    @pytest.mark.parametrize("symbol", list(RTOKEN_SYMBOLS))
    def test_every_entry_names_a_source_and_a_fetch_date(self, symbol: str) -> None:
        identity = identity_of(symbol)
        assert identity.source.strip()
        assert isinstance(identity.source_fetched_at, date)

    def test_leveraged_entries_cite_the_issuer_by_name(self) -> None:
        for symbol in ("TQQQUSDT", "SQQQUSDT"):
            identity = identity_of(symbol)
            assert "proshares.com" in identity.source.lower()
            assert identity.issuer == "ProShares"


class TestKindIsNotInterchangeable:
    def test_single_company_stocks_declare_no_benchmark_and_no_multiple(self) -> None:
        nvda = identity_of("NVDAUSDT")
        assert nvda.kind is InstrumentKind.EQUITY
        assert nvda.benchmark is None
        assert nvda.target_multiple is None

    def test_the_index_etf_declares_a_1x_target_not_none(self) -> None:
        """1x is a real declared fact, distinct from 'nothing declared'."""
        qqq = identity_of("QQQUSDT")
        assert qqq.kind is InstrumentKind.INDEX_ETF
        assert qqq.target_multiple == Decimal("1")
        assert qqq.daily_reset is False

    def test_the_two_leveraged_funds_have_opposite_signed_targets(self) -> None:
        tqqq = identity_of("TQQQUSDT")
        sqqq = identity_of("SQQQUSDT")
        assert tqqq.target_multiple == Decimal("3")
        assert sqqq.target_multiple == Decimal("-3")
        assert tqqq.kind is sqqq.kind is InstrumentKind.LEVERAGED_ETF
        assert tqqq.daily_reset and sqqq.daily_reset

    def test_leveraged_funds_share_the_same_benchmark_as_the_index_fund(self) -> None:
        """TQQQ, SQQQ and QQQ are all declared against one Nasdaq-100 fact, not three."""
        assert (
            identity_of("TQQQUSDT").benchmark
            == identity_of("SQQQUSDT").benchmark
            == identity_of("QQQUSDT").benchmark
            == "Nasdaq-100 Index"
        )


class TestTheLeverageCheckIsARealComparison:
    """Behaviour under test: it must be able to fail, or it isn't a check."""

    def test_nothing_declared_reports_none_not_a_fabricated_agreement(self) -> None:
        assert leverage_check("NVDAUSDT", measured_beta=1.71) is None

    def test_a_measurement_matching_the_declared_target_is_within_tolerance(self) -> None:
        got = leverage_check("TQQQUSDT", measured_beta=2.912)
        assert got is not None
        assert got.declared == Decimal("3")
        assert got.within_tolerance is True

    def test_a_measurement_far_from_the_declared_target_disagrees(self) -> None:
        """The check must be able to say no. A beta of 1.5 against a declared 3x is not drift."""
        got = leverage_check("TQQQUSDT", measured_beta=1.5)
        assert got is not None
        assert got.within_tolerance is False
        assert "DISAGREES" in got.render()

    def test_the_sign_is_part_of_the_comparison(self) -> None:
        """A +2.9 measurement against a declared -3 must disagree, not read as 'close enough'."""
        got = leverage_check("SQQQUSDT", measured_beta=2.9)
        assert got is not None
        assert got.within_tolerance is False

    def test_daily_reset_funds_note_the_honest_cause_of_a_small_gap(self) -> None:
        got = leverage_check("SQQQUSDT", measured_beta=-2.885)
        assert got is not None
        assert "resets daily" in got.render()

    def test_the_tolerance_is_not_wide_enough_to_hide_a_materially_wrong_declaration(self) -> None:
        """A sanity bound on the constant itself: it must not swallow a full percentage point."""
        assert Decimal("1") > LEVERAGE_TOLERANCE

    def test_the_measured_side_of_the_current_live_artefact_is_within_tolerance(self) -> None:
        """Reproduces the actual figures this session measured, so a future drift is visible."""
        for symbol, measured in (("TQQQUSDT", 2.9118), ("SQQQUSDT", -2.8977)):
            got = leverage_check(symbol, measured)
            assert got is not None and got.within_tolerance


class TestSerialisation:
    def test_registry_as_dict_covers_every_symbol(self) -> None:
        payload = registry_as_dict()
        assert set(payload) == set(RTOKEN_SYMBOLS)

    def test_a_check_serialises_with_every_field_a_reader_needs(self) -> None:
        got = leverage_check("TQQQUSDT", measured_beta=2.9)
        assert got is not None
        payload = got.as_dict()
        assert {"symbol", "declared", "measured", "difference", "within_tolerance",
                "daily_reset"} <= set(payload)
        # Decimal must not leak into JSON-unsafe territory.
        assert isinstance(payload["declared"], str)

    def test_an_identity_round_trips_through_as_dict_without_losing_the_source(self) -> None:
        identity = identity_of("SQQQUSDT")
        payload = identity.as_dict()
        assert payload["source"] == identity.source
        assert payload["source_fetched_at"] == identity.source_fetched_at.isoformat()


class TestConstruction:
    def test_an_identity_can_be_built_directly_for_a_hypothetical_instrument(self) -> None:
        """Exercises the dataclass itself, independent of the shipped registry."""
        hypothetical = InstrumentIdentity(
            symbol="TESTUSDT", underlying="TEST", underlying_name="Test Fund",
            issuer="Test Sponsor", kind=InstrumentKind.LEVERAGED_ETF, benchmark="Test Index",
            target_multiple=Decimal("2"), daily_reset=True, source="test fixture",
            source_fetched_at=date(2026, 1, 1),
        )
        assert hypothetical.as_dict()["target_multiple"] == "2"
