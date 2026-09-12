"""Bitget market client tests.

Network tests are opt-in (ARGUS_LIVE_MARKET=1) so the suite stays fast and offline by default.
The parsing tests below run always, against response shapes recorded from the live API.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from argus.market.bitget import ANCHOR_OF, RTOKEN_SYMBOLS, Ticker, _dec
from argus.market.collector import Observation, session_summary
from argus.truth.clocks import DualClock

LIVE = os.environ.get("ARGUS_LIVE_MARKET") == "1"
live_only = pytest.mark.skipif(not LIVE, reason="set ARGUS_LIVE_MARKET=1")


def _ticker(symbol: str = "NVDAUSDT", bid: str = "219.36", ask: str = "219.38") -> Ticker:
    return Ticker(
        symbol=symbol, last=Decimal("219.37"), bid=Decimal(bid), ask=Decimal(ask),
        high_24h=Decimal("221"), low_24h=Decimal("217"), change_24h=Decimal("-0.00191"),
        base_volume=Decimal("49562.88"), funding_rate=Decimal("0"),
        fetched_at=datetime(2026, 3, 8, 7, 55, tzinfo=UTC),
    )


class TestParsing:
    def test_empty_string_becomes_default_not_an_error(self) -> None:
        """Bitget returns "" for absent numerics. A missing funding rate must not make a price
        unusable — but it must never silently become something plausible either."""
        assert _dec("") == Decimal("0")
        assert _dec(None) == Decimal("0")
        assert _dec("1.25") == Decimal("1.25")

    def test_garbage_does_not_raise(self) -> None:
        assert _dec("not-a-number") == Decimal("0")


class TestSpread:
    def test_spread_in_bps(self) -> None:
        """Measured live: NVDAUSDT quoted 0.46bps on a weekend."""
        assert _ticker().spread_bps == pytest.approx(Decimal("0.91"), abs=Decimal("0.01"))

    def test_missing_quotes_give_zero_not_a_crash(self) -> None:
        assert _ticker(bid="0", ask="0").spread_bps == Decimal("0")

    def test_anchor_mapping_is_explicit(self) -> None:
        """Regex matching swept in FARTCOINUSDT and MARSCOINUSDT on the first attempt."""
        assert _ticker().anchor == "NVDA"
        assert "FARTCOINUSDT" not in RTOKEN_SYMBOLS
        assert set(ANCHOR_OF) == set(RTOKEN_SYMBOLS)


class TestObservation:
    def test_observation_is_session_tagged(self) -> None:
        """The session join is the contribution — a price series without it cannot answer what an
        instrument does while its anchor is asleep."""
        obs = Observation.of(_ticker(), DualClock())
        assert obs.phase == "weekend"
        assert obs.anchor_asleep is True
        assert obs.hours_to_discovery > 0
        assert obs.anchor == "NVDA"

    def test_numbers_are_stored_as_strings(self) -> None:
        """These rows become Facts. A float that has been through JSON is not the number that
        was measured."""
        obs = Observation.of(_ticker(), DualClock())
        assert isinstance(obs.last, str)
        assert isinstance(obs.spread_bps, str)

    def test_summary_reports_fee_to_spread_ratio(self) -> None:
        """The headline the panel exists to produce."""
        rows = [Observation.of(_ticker(s), DualClock()) for s in ("NVDAUSDT", "TSLAUSDT")]
        got = session_summary(rows)
        assert got["round_trip_fee_bps"] == "12"
        assert Decimal(str(got["fee_to_median_spread_ratio"])) > 1


@live_only
class TestLiveMarket:
    def test_rtokens_are_live_on_bitget(self) -> None:
        from argus.market.bitget import fetch_rtokens
        got = fetch_rtokens()
        assert len(got) >= 10
        assert "NVDAUSDT" in got
        assert got["NVDAUSDT"].last > 0

    def test_candles_return_ordered_history(self) -> None:
        from argus.market.bitget import fetch_candles
        got = fetch_candles("NVDAUSDT", granularity="1H", limit=10)
        assert len(got) >= 5
        assert got[0]["ts"] < got[-1]["ts"]
