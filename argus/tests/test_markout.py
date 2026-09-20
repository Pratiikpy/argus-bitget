"""Markout tests — most of them exist to pin the sign convention.

A markout with the sign flipped is symmetric, plausible, and says the exact opposite of the truth:
it reports that resting orders are systematically picked *up* rather than picked off, which would
turn `cost/model.py`'s refusal to credit maker treatment into an apparent free lunch. So the sign is
asserted from both sides, on hand-built samples where the answer is obvious.

The rest cover the joins — a print with no later mid, a horizon longer than the window, a sample
that never overlaps the prints — because each of those silently produces a smaller, quieter number
rather than an error.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from argus.market.markout import (
    MAKER_BPS,
    MIN_PRINTS,
    MarkoutError,
    MidSample,
    Print,
    analyse,
)

NOW = datetime(2026, 9, 14, 12, tzinfo=UTC)
T0 = 1_789_000_000_000


def _samples(path: list[float], *, step_ms: int = 1000) -> list[MidSample]:
    return [
        MidSample(ts_ms=T0 + i * step_ms, mid=mid, spread_bps=1.0)
        for i, mid in enumerate(path)
    ]


def _prints(count: int, side: str, *, at_ms: int = T0, rpi: bool = False) -> list[Print]:
    return [
        Print(ts_ms=at_ms + i, price=100.0, size=1.0, side=side, rpi=rpi) for i in range(count)
    ]


def _report(path: list[float], side: str, count: int = MIN_PRINTS, **kwargs: object) -> object:
    return analyse(
        _samples(path), _prints(count, side), symbol="X", phase="rth", window_seconds=len(path),
        horizons=(5,), started_at=NOW, **kwargs,  # type: ignore[arg-type]
    )


class TestTheSignConvention:
    def test_a_rising_mid_after_an_aggressive_buy_is_a_loss_for_the_passive_seller(self) -> None:
        """`side="buy"` means somebody lifted the offer, so the counterparty was a resting SELL.
        The mid rising afterwards means that seller was picked off."""
        rising = [100.0 + i * 0.01 for i in range(10)]
        got = _report(rising, "buy")
        markout = got.markouts[0]  # type: ignore[attr-defined]
        assert markout.prints == MIN_PRINTS
        assert markout.median_bps < 0

    def test_a_rising_mid_after_an_aggressive_sell_is_a_gain_for_the_passive_buyer(self) -> None:
        rising = [100.0 + i * 0.01 for i in range(10)]
        got = _report(rising, "sell")
        assert got.markouts[0].median_bps > 0  # type: ignore[attr-defined]

    def test_the_two_sides_are_exact_mirrors_on_the_same_path(self) -> None:
        rising = [100.0 + i * 0.02 for i in range(10)]
        buy = _report(rising, "buy").markouts[0]  # type: ignore[attr-defined]
        sell = _report(rising, "sell").markouts[0]  # type: ignore[attr-defined]
        assert buy.median_bps == pytest.approx(-sell.median_bps, abs=1e-9)

    def test_a_flat_mid_costs_the_passive_side_nothing(self) -> None:
        got = _report([100.0] * 10, "buy")
        assert got.markouts[0].median_bps == pytest.approx(0.0, abs=1e-9)  # type: ignore[attr-defined]

    def test_the_magnitude_is_the_actual_move(self) -> None:
        """Five seconds after T0 the mid is 100.05, which is 5bps above 100. The passive seller
        behind an aggressive buy therefore loses exactly 5bps."""
        path = [100.0 + i * 0.01 for i in range(10)]
        got = _report(path, "buy")
        assert got.markouts[0].median_bps == pytest.approx(-5.0, abs=0.01)  # type: ignore[attr-defined]


class TestTheJoins:
    def test_a_print_with_no_later_mid_is_dropped_not_counted_as_flat(self) -> None:
        """Counting an unresolvable print as zero would pull every median toward nothing, which
        looks like a calm book rather than a short sample."""
        samples = _samples([100.0] * 3)  # only 3 seconds of mids
        got = analyse(
            samples, _prints(MIN_PRINTS, "buy"), symbol="X", phase="rth", window_seconds=3,
            horizons=(5,), started_at=NOW,
        )
        assert got.markouts[0].prints == 0

    def test_the_mid_is_taken_at_or_after_the_print_never_before(self) -> None:
        """The mid preceding a print is contaminated by whatever caused it."""
        samples = [
            MidSample(ts_ms=T0 - 5000, mid=90.0, spread_bps=1.0),
            MidSample(ts_ms=T0, mid=100.0, spread_bps=1.0),
            MidSample(ts_ms=T0 + 5000, mid=100.05, spread_bps=1.0),
        ]
        got = analyse(
            samples, _prints(MIN_PRINTS, "buy"), symbol="X", phase="rth", window_seconds=10,
            horizons=(5,), started_at=NOW,
        )
        # Entry mid is 100.0 (the sample at the print), not 90.0 (the one before it).
        assert got.markouts[0].median_bps == pytest.approx(-5.0, abs=0.01)

    def test_a_horizon_the_window_cannot_resolve_reports_zero_prints(self) -> None:
        got = analyse(
            _samples([100.0] * 10), _prints(MIN_PRINTS, "buy"), symbol="X", phase="rth",
            window_seconds=10, horizons=(5, 600), started_at=NOW,
        )
        by_horizon = {m.horizon_seconds: m for m in got.markouts}
        assert by_horizon[5].prints == MIN_PRINTS
        assert by_horizon[600].prints == 0

    def test_no_samples_at_all_is_an_error_not_an_empty_report(self) -> None:
        with pytest.raises(MarkoutError, match="no book samples"):
            analyse([], _prints(5, "buy"), symbol="X", phase="rth", window_seconds=10)


class TestTheHeadlineHorizon:
    def test_it_prefers_the_longest_horizon_with_enough_prints(self) -> None:
        got = analyse(
            _samples([100.0 + i * 0.01 for i in range(40)]), _prints(MIN_PRINTS, "buy"),
            symbol="X", phase="rth", window_seconds=40, horizons=(5, 15), started_at=NOW,
        )
        assert got.longest is not None
        assert got.longest.horizon_seconds == 15

    def test_an_empty_long_horizon_does_not_become_the_headline(self) -> None:
        """The bug this pins: a 90s window cannot resolve a 60s horizon for prints in its final
        minute, so the 60s row is empty while the 5s row is not. Reporting the empty row said
        "0 prints" and read as a quiet book rather than as a shorter usable horizon."""
        got = analyse(
            _samples([100.0 + i * 0.01 for i in range(20)]), _prints(MIN_PRINTS, "buy"),
            symbol="X", phase="rth", window_seconds=20, horizons=(5, 600), started_at=NOW,
        )
        assert got.longest is not None
        assert got.longest.horizon_seconds == 5
        assert got.longest.prints == MIN_PRINTS


class TestTheVerdict:
    def test_a_thin_sample_refuses_to_conclude(self) -> None:
        got = _report([100.0 + i * 0.01 for i in range(10)], "buy", count=MIN_PRINTS - 1)
        assert "below the" in got.verdict  # type: ignore[attr-defined]
        assert "not that resting here is safe" in got.verdict  # type: ignore[attr-defined]

    def test_adverse_selection_above_the_maker_fee_upholds_the_refusal(self) -> None:
        """A 10bps pick-off against a 2bps maker fee: passive execution is not free money."""
        violent = [100.0 + i * 0.02 for i in range(10)]
        got = _report(violent, "buy")
        assert "maker side stands" in got.verdict  # type: ignore[attr-defined]

    def test_adverse_selection_below_the_maker_fee_says_so_plainly(self) -> None:
        gentle = [100.0 + i * 0.000_1 for i in range(10)]
        got = _report(gentle, "buy")
        verdict = got.verdict  # type: ignore[attr-defined]
        assert "costing measurable edge" in verdict or "No measurable adverse selection" in verdict

    def test_the_maker_fee_it_is_compared_against_is_the_real_one(self) -> None:
        from decimal import Decimal

        from argus.cost.model import CostModel

        assert Decimal(str(MAKER_BPS)) == CostModel.bitget_perp().maker_bps

    def test_the_report_serialises(self) -> None:
        blob = json.loads(json.dumps(
            _report([100.0 + i * 0.01 for i in range(10)], "buy").as_dict(),  # type: ignore[attr-defined]
        ))
        assert blob["markouts"] and "verdict" in blob
        assert blob["maker_bps"] == MAKER_BPS

    def test_the_rendered_report_names_the_window_and_the_phase(self) -> None:
        text = _report([100.0 + i * 0.01 for i in range(10)], "buy").render()  # type: ignore[attr-defined]
        assert "MARKOUT" in text and "rth" in text
        assert "horizon" in text


class TestRetailFlowIsTracked:
    def test_the_rpi_share_is_reported(self) -> None:
        mixed = _prints(10, "buy", rpi=True) + _prints(10, "sell", at_ms=T0 + 100)
        got = analyse(
            _samples([100.0] * 10), mixed, symbol="X", phase="rth", window_seconds=10,
            horizons=(5,), started_at=NOW,
        )
        assert got.rpi_share == pytest.approx(0.5)

    def test_no_prints_is_a_zero_share_not_a_crash(self) -> None:
        got = analyse(
            _samples([100.0] * 10), [], symbol="X", phase="rth", window_seconds=10,
            horizons=(5,), started_at=NOW,
        )
        assert got.rpi_share == 0.0
        assert got.prints_seen == 0


class TestTheLiveFeed:
    def test_the_public_fills_endpoint_answers(self) -> None:
        from argus.market.bitget import BitgetError
        from argus.market.markout import fetch_prints

        try:
            prints = fetch_prints("NVDAUSDT", limit=20)
        except BitgetError as exc:
            pytest.skip(f"venue unreachable: {exc}")
        assert prints
        assert all(p.side in ("buy", "sell") for p in prints)
        assert all(p.price > 0 and p.size > 0 for p in prints)
        # Sorted oldest first, because the feed is not.
        assert [p.ts_ms for p in prints] == sorted(p.ts_ms for p in prints)

    def test_the_stored_report_is_readable(self) -> None:
        from pathlib import Path

        path = Path(__file__).resolve().parents[1] / "data" / "markout.json"
        if not path.exists():
            pytest.skip("no markout measurement on this machine")
        blob = json.loads(path.read_text(encoding="utf-8"))
        assert blob["symbol"].endswith("USDT")
        assert blob["window_seconds"] > 0
        assert blob["markouts"]
