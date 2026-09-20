"""Tests for the funding carry study.

The study's whole value is that it refuses to recommend a trade on the strength of a funding rate,
so the properties worth pinning are the ones that let it say no: the sign convention that decides
who collects, the leg shape that can express a same-side basket, and the replay that outranks the
funding table. Network is avoided by seeding the price cache, so these run the real code path
against a series whose answer can be worked out by hand.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from argus.research import carry
from argus.research.carry import (
    BOOTSTRAP_SEED,
    BasketReplay,
    CarryPair,
    CarryUnavailable,
    FundingProfile,
    Settlement,
    _bootstrap_mean,
    effective_windows,
    profile,
    replay_basket,
)

START = datetime(2026, 6, 1, tzinfo=UTC)


def _flat_series(symbol: str, hours: int, price: float) -> list[tuple[datetime, float]]:
    return [(START + timedelta(hours=h), price) for h in range(hours)]


def _settlements(rates_bps: list[float], *, every_hours: int = 8) -> list[Settlement]:
    return [
        Settlement(at=START + timedelta(hours=every_hours * (i + 1)), rate_bps=r)
        for i, r in enumerate(rates_bps)
    ]


class TestTheFundingSignConvention:
    """Funding accrues as ``-weight x rate``. Getting this backwards would invert every result in
    the study while leaving every number looking plausible, so it is pinned on a flat market where
    the price leg contributes exactly nothing."""

    def _replay(self, weight: float, rate_bps: float) -> BasketReplay:
        carry._SERIES_CACHE.clear()
        carry._SERIES_CACHE[("AAA", 10)] = _flat_series("AAA", 24 * 5, 100.0)
        got = replay_basket(
            (("AAA", weight),), {"AAA": _settlements([rate_bps] * 30)},
            days=10, holding_days=1, round_trip_bps=0.0,
        )
        assert got is not None
        return got

    def test_a_short_collects_a_positive_rate(self) -> None:
        got = self._replay(-1.0, 10.0)
        assert got.mean_funding_pct > 0
        assert got.mean_price_pct == pytest.approx(0.0)

    def test_a_long_pays_a_positive_rate(self) -> None:
        got = self._replay(1.0, 10.0)
        assert got.mean_funding_pct < 0

    def test_a_short_pays_a_negative_rate(self) -> None:
        got = self._replay(-1.0, -10.0)
        assert got.mean_funding_pct < 0

    def test_the_magnitude_is_the_rate_times_the_settlements(self) -> None:
        """One day is three eight-hour settlements. 10bps each, short, is +30bps = +0.30%.

        The settlement fixture deliberately runs past the end of the price series. With it
        stopping short, the final windows contained two settlements instead of three and the
        mean came out at 0.275% — correct behaviour on a truncated funding record, and a
        fixture that would have quietly understated every figure in this class.
        """
        got = self._replay(-1.0, 10.0)
        assert got.mean_funding_pct == pytest.approx(0.30, abs=1e-9)


class TestTheReplayOutranksTheFunding:
    def test_a_losing_price_leg_overrides_a_paying_funding_leg(self) -> None:
        """The finding the module exists for: funding arrives and the position still loses."""
        carry._SERIES_CACHE.clear()
        falling = [(START + timedelta(hours=h), 100.0 * (1.0 + 0.001 * h)) for h in range(24 * 5)]
        carry._SERIES_CACHE[("AAA", 10)] = falling
        got = replay_basket(
            (("AAA", -1.0),), {"AAA": _settlements([10.0] * 30)},
            days=10, holding_days=1, round_trip_bps=0.0,
        )
        assert got is not None
        assert got.mean_funding_pct > 0, "the funding leg pays"
        assert got.mean_price_pct < 0, "the price leg costs more"
        assert not got.funding_survives_the_price

    def test_costs_are_subtracted_from_the_total_only(self) -> None:
        carry._SERIES_CACHE.clear()
        carry._SERIES_CACHE[("AAA", 10)] = _flat_series("AAA", 24 * 5, 100.0)
        free = replay_basket(
            (("AAA", -1.0),), {"AAA": _settlements([10.0] * 30)},
            days=10, holding_days=1, round_trip_bps=0.0,
        )
        costed = replay_basket(
            (("AAA", -1.0),), {"AAA": _settlements([10.0] * 30)},
            days=10, holding_days=1, round_trip_bps=20.0,
        )
        assert free is not None and costed is not None
        assert costed.mean_funding_pct == pytest.approx(free.mean_funding_pct)
        assert costed.mean_total_pct == pytest.approx(free.mean_total_pct - 0.20)

    def test_a_series_shorter_than_the_holding_period_returns_nothing(self) -> None:
        """Absence, not a window computed from too little data."""
        carry._SERIES_CACHE.clear()
        carry._SERIES_CACHE[("AAA", 10)] = _flat_series("AAA", 10, 100.0)
        assert replay_basket(
            (("AAA", -1.0),), {"AAA": []}, days=10, holding_days=14,
        ) is None


class TestTheLegShape:
    def test_a_same_side_basket_is_described_as_such(self) -> None:
        """The defect this shape replaced: a negative minimum-variance ratio puts both legs on the
        same side, and a long_leg/short_leg pair could only ever describe opposed legs."""
        pair = CarryPair(
            legs=(("QQQUSDT", -1.0), ("SQQQUSDT", -0.322)), structure="test",
            hedge_ratio=-0.322, r_squared=0.9, correlation_low=-0.9, observations=100,
            gross_notional=1.322, net_bps_per_settlement=0.5, net_low_bps=0.4,
            net_high_bps=0.6, round_trip_bps=16.0,
        )
        assert pair.description == "short 1.000 QQQUSDT + short 0.322 SQQQUSDT"
        assert "long" not in pair.description

    def test_an_opposed_basket_names_both_sides(self) -> None:
        pair = CarryPair(
            legs=(("QQQUSDT", -1.0), ("TQQQUSDT", 0.328)), structure="test",
            hedge_ratio=0.328, r_squared=0.9, correlation_low=0.9, observations=100,
            gross_notional=1.328, net_bps_per_settlement=0.5, net_low_bps=0.4,
            net_high_bps=0.6, round_trip_bps=16.0,
        )
        assert pair.description == "short 1.000 QQQUSDT + long 0.328 TQQQUSDT"


class TestTheArithmetic:
    def test_breakeven_is_the_round_trip_over_the_daily_carry(self) -> None:
        pair = CarryPair(
            legs=(("A", -1.0),), structure="t", hedge_ratio=1.0, r_squared=0.9,
            correlation_low=0.9, observations=100, gross_notional=1.0,
            net_bps_per_settlement=1.0, net_low_bps=0.5, net_high_bps=1.5, round_trip_bps=12.0,
        )
        # 1bps x 3 settlements a day = 3bps a day; 12bps takes four days.
        assert pair.breakeven_days == pytest.approx(4.0)

    def test_a_negative_carry_never_breaks_even(self) -> None:
        pair = CarryPair(
            legs=(("A", -1.0),), structure="t", hedge_ratio=1.0, r_squared=0.9,
            correlation_low=0.9, observations=100, gross_notional=1.0,
            net_bps_per_settlement=-1.0, net_low_bps=-1.5, net_high_bps=-0.5, round_trip_bps=12.0,
        )
        assert pair.breakeven_days == float("inf")
        assert pair.as_dict()["breakeven_days"] is None

    def test_effective_windows_collapses_overlap(self) -> None:
        """1,823 overlapping 14-day windows out of 90 days are about six independent holds."""
        assert effective_windows(1823, 14, 90.0) == pytest.approx(90 / 14)
        assert effective_windows(1823, 14, 90.0) < 7

    def test_effective_windows_never_claims_less_than_one(self) -> None:
        assert effective_windows(10, 60, 14.0) == 1.0


class TestTheInterval:
    def test_the_bootstrap_is_deterministic(self) -> None:
        values = [0.0] * 85 + [1.0] * 15
        first = _bootstrap_mean(values, seed=BOOTSTRAP_SEED)
        second = _bootstrap_mean(values, seed=BOOTSTRAP_SEED)
        assert first == second

    def test_an_all_zero_series_has_an_interval_at_zero(self) -> None:
        """The shape of this data is a point mass at zero with a thin tail. When the tail is absent
        the interval must collapse rather than manufacture width."""
        got = profile("TEST", _settlements([0.0] * 100))
        assert got.mean_bps == 0.0
        assert got.median_bps == 0.0
        assert (got.mean_low_bps, got.mean_high_bps) == (0.0, 0.0)
        assert not got.interval_excludes_zero

    def test_a_sparse_positive_series_is_still_distinguishable_from_zero(self) -> None:
        got = profile("TEST", _settlements([0.0] * 85 + [5.0] * 15))
        assert got.zero_share == pytest.approx(0.85)
        assert got.positive_share == pytest.approx(0.15)
        assert got.negative_share == 0.0
        assert got.interval_excludes_zero

    def test_the_median_being_zero_does_not_stop_the_mean_being_positive(self) -> None:
        """The central fact about this data: eighty-five per cent of settlements pay nothing and the
        mean is still positive, so reporting either number alone misleads."""
        got = profile("TEST", _settlements([0.0] * 85 + [5.0] * 15))
        assert got.median_bps == 0.0
        assert got.mean_bps > 0


class TestTheVerdict:
    def _pair(self, name: str, total: float, funding: float, price: float) -> CarryPair:
        return CarryPair(
            legs=((name, -1.0),), structure="t", hedge_ratio=1.0, r_squared=0.95,
            correlation_low=0.9, observations=2000, gross_notional=1.0,
            net_bps_per_settlement=0.5, net_low_bps=0.4, net_high_bps=0.6, round_trip_bps=12.0,
            replay=BasketReplay(
                holding_days=14, windows=1800, mean_total_pct=total, median_total_pct=total,
                mean_funding_pct=funding, mean_price_pct=price,
                share_positive=0.7 if total > 0 else 0.24, worst_pct=-3.0, best_pct=3.0,
            ),
        )

    def _profiles(self) -> list[FundingProfile]:
        return [profile("TEST", _settlements([0.0] * 85 + [5.0] * 15))]

    def test_a_basket_the_funding_recommends_and_the_replay_refuses_is_named(self) -> None:
        text = carry._verdict(self._profiles(), [self._pair("BAD", -0.1, 0.2, -0.3)])
        assert "loses money" in text
        assert "which is why there is a replay" in text

    def test_a_return_that_is_mostly_price_is_not_called_a_carry(self) -> None:
        text = carry._verdict(self._profiles(), [self._pair("DECAY", 0.22, 0.046, 0.295)])
        assert "should not be called a carry" in text
        assert "volatility decay" in text

    def test_a_genuine_carry_is_not_disclaimed_as_decay(self) -> None:
        text = carry._verdict(self._profiles(), [self._pair("REAL", 0.20, 0.30, -0.10)])
        assert "should not be called a carry" not in text

    def test_no_surviving_basket_proposes_nothing(self) -> None:
        text = carry._verdict(self._profiles(), [self._pair("BAD", -0.1, 0.2, -0.3)])
        assert "No basket survives being held" in text

    def test_the_overlap_caveat_is_always_stated_when_a_basket_survives(self) -> None:
        text = carry._verdict(self._profiles(), [self._pair("REAL", 0.20, 0.30, -0.10)])
        assert "independent holds" in text


class TestTheVenueIsNotGuessed:
    def test_a_bad_response_code_raises_rather_than_defaulting(self, monkeypatch) -> None:
        """A funding study that silently substitutes a zero rate for an unreachable venue would
        report 'no carry' for an outage, which is the same answer as a real finding."""
        import io as _io
        import json as _json

        class _Fake:
            def __enter__(self):
                return _io.StringIO(_json.dumps({"code": "40020", "msg": "Parameter limit error"}))

            def __exit__(self, *a):
                return False

        monkeypatch.setattr(carry.urllib.request, "urlopen", lambda *a, **k: _Fake())
        with pytest.raises(CarryUnavailable, match="40020"):
            carry.fetch_funding("NVDAUSDT")
