"""Latency tests.

Two things are being guarded. First, that the interpolation reproduces hftbacktest's arithmetic
rather than something that merely looks like it. Second — and this is the part specific to us — that
deliberation time is treated as a cost. A desk whose model thinks for thirty seconds is trading a
thirty-second-old price, and a system that does not charge for that will always conclude that more
thinking is free.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from argus.execution.latency import (
    NANOS_PER_MILLI,
    NANOS_PER_SECOND,
    ConstantLatency,
    DecisionLatency,
    InterpolatedLatency,
    LatencyError,
    LatencyModel,
    LatencyRow,
    latency_slippage_bps,
    thinking_budget_cost_bps,
)
from argus.execution.latency_probe import (
    Sample,
    estimate_clock_offset_ns,
    load_rows,
    to_rows,
)

D = Decimal
MS = NANOS_PER_MILLI


def _row(req_ms: int, entry_ms: int, resp_ms: int) -> LatencyRow:
    req = req_ms * MS
    exch = req + entry_ms * MS
    return LatencyRow(req_ts=req, exch_ts=exch, resp_ts=exch + resp_ms * MS)


class TestConstantLatencyMustBeArgued:
    """hftbacktest calls a constant optimistic. So it is not available by default here."""

    def test_zero_latency_is_rejected(self) -> None:
        with pytest.raises(LatencyError, match="zero fee"):
            ConstantLatency(entry_ns=0, response_ns=MS, reason="fast")

    def test_negative_latency_is_rejected(self) -> None:
        with pytest.raises(LatencyError):
            ConstantLatency(entry_ns=-1, response_ns=MS, reason="fast")

    def test_a_reason_is_required(self) -> None:
        with pytest.raises(LatencyError, match="written reason"):
            ConstantLatency(entry_ns=MS, response_ns=MS, reason="")

    def test_whitespace_is_not_a_reason(self) -> None:
        with pytest.raises(LatencyError, match="written reason"):
            ConstantLatency(entry_ns=MS, response_ns=MS, reason="   ")

    def test_with_a_reason_it_works(self) -> None:
        model = ConstantLatency(
            entry_ns=5 * MS, response_ns=3 * MS,
            reason="daily-horizon strategy; latency is three orders below the holding period",
        )
        assert model.entry(0) == 5 * MS
        assert model.response(0) == 3 * MS
        assert isinstance(model, LatencyModel)


class TestInterpolation:
    """``latency.rs:170-273``."""

    def test_it_interpolates_linearly_between_observations(self) -> None:
        # Entry latency 10ms at t=0, 30ms at t=100. Halfway must read 20ms.
        model = InterpolatedLatency([_row(0, 10, 5), _row(100, 30, 5)])
        assert model.entry(50 * MS) == 20 * MS

    def test_it_lands_exactly_on_an_observation(self) -> None:
        model = InterpolatedLatency([_row(0, 10, 5), _row(100, 30, 5)])
        assert model.entry(0) == 10 * MS

    def test_before_the_record_it_holds_rather_than_extrapolates(self) -> None:
        """Extrapolating a latency curve invents the tail, and the tail is what costs money."""
        model = InterpolatedLatency([_row(100, 10, 5), _row(200, 30, 5)])
        assert model.entry(0) == 10 * MS

    def test_after_the_record_it_holds_too(self) -> None:
        model = InterpolatedLatency([_row(100, 10, 5), _row(200, 30, 5)])
        assert model.entry(10_000 * MS) == 30 * MS

    def test_response_is_keyed_on_the_exchange_clock_not_ours(self) -> None:
        """Two cursors over the same rows, because the two delays have different drivers."""
        rows = [_row(0, 10, 4), _row(100, 10, 20)]
        model = InterpolatedLatency(rows)
        # Response rows are bracketed by exch_ts: 10ms and 110ms.
        assert model.response(rows[0].exch_ts) == 4 * MS
        assert model.response(rows[1].exch_ts) == 20 * MS
        midpoint = (rows[0].exch_ts + rows[1].exch_ts) // 2
        assert model.response(midpoint) == 12 * MS

    def test_one_row_is_not_a_series(self) -> None:
        with pytest.raises(LatencyError, match="at least two"):
            InterpolatedLatency([_row(0, 10, 5)])

    def test_duplicate_request_timestamps_are_rejected(self) -> None:
        with pytest.raises(LatencyError, match="share req_ts"):
            InterpolatedLatency([_row(0, 10, 5), _row(0, 30, 5)])

    def test_rows_are_sorted_not_assumed_sorted(self) -> None:
        model = InterpolatedLatency([_row(100, 30, 5), _row(0, 10, 5)])
        assert model.entry(50 * MS) == 20 * MS

    def test_percentile_reads_the_tail(self) -> None:
        rows = [_row(i * 10, 10, 10) for i in range(9)] + [_row(90, 500, 500)]
        model = InterpolatedLatency(rows)
        assert model.percentile_ns(0.5) == 20 * MS
        assert model.percentile_ns(1.0) == 1000 * MS

    def test_percentile_outside_zero_to_one_is_rejected(self) -> None:
        model = InterpolatedLatency([_row(0, 10, 5), _row(100, 30, 5)])
        with pytest.raises(LatencyError):
            model.percentile_ns(1.5)


class TestRejections:
    """``latency.rs:199-214`` — exch_ts == 0 means the order never landed."""

    def test_a_rejection_returns_a_negative_entry_latency(self) -> None:
        rejected = LatencyRow(req_ts=0, exch_ts=0, resp_ts=40 * MS)
        model = InterpolatedLatency([rejected, _row(100, 10, 5)])
        assert rejected.was_rejected
        assert model.entry(0) == -40 * MS

    def test_rejected_rows_do_not_poison_the_response_index(self) -> None:
        """exch_ts == 0 would sort to the front and drag every response toward t=0."""
        rejected = LatencyRow(req_ts=0, exch_ts=0, resp_ts=40 * MS)
        good_a, good_b = _row(100, 10, 4), _row(200, 10, 20)
        model = InterpolatedLatency([rejected, good_a, good_b])
        assert model.response(good_a.exch_ts) == 4 * MS
        assert model.response(good_b.exch_ts) == 20 * MS

    def test_the_rejection_rate_is_reported(self) -> None:
        rows = [LatencyRow(req_ts=0, exch_ts=0, resp_ts=40 * MS), _row(100, 10, 5)]
        assert InterpolatedLatency(rows).rejection_rate == 0.5

    def test_an_all_rejected_series_refuses_to_answer(self) -> None:
        rows = [
            LatencyRow(req_ts=0, exch_ts=0, resp_ts=40 * MS),
            LatencyRow(req_ts=100 * MS, exch_ts=0, resp_ts=140 * MS),
        ]
        with pytest.raises(LatencyError, match="every recorded round trip was rejected"):
            InterpolatedLatency(rows).response(0)


class TestDeliberationIsACost:
    """The latency that actually binds on an LLM desk."""

    def test_thinking_dominates_the_network(self) -> None:
        fast = DecisionLatency(think_ms=0, network_ms=50, label="reflex")
        slow = DecisionLatency(think_ms=40_000, network_ms=50, label="full reasoning")
        assert slow.entry(0) > 500 * fast.entry(0)

    def test_the_response_leg_is_network_only(self) -> None:
        """We are not thinking while the venue answers."""
        model = DecisionLatency(think_ms=20_000, network_ms=50)
        assert model.response(0) == 50 * MS

    def test_zero_network_is_rejected(self) -> None:
        with pytest.raises(LatencyError, match="network never is"):
            DecisionLatency(think_ms=1000, network_ms=0)

    def test_it_satisfies_the_protocol(self) -> None:
        assert isinstance(DecisionLatency(think_ms=1000), LatencyModel)

    def test_slippage_scales_with_the_square_root_of_time(self) -> None:
        """A random walk's expected move grows with sqrt(t), so 4x the delay is 2x the cost."""
        vol = D("0.45")
        one = latency_slippage_bps(latency_ns=NANOS_PER_SECOND, annualised_vol=vol)
        four = latency_slippage_bps(latency_ns=4 * NANOS_PER_SECOND, annualised_vol=vol)
        assert four / one == pytest.approx(2.0, rel=1e-6)

    def test_a_thin_book_multiplies_the_cost(self) -> None:
        """Our measured off-hours regime runs near a third of RTH depth."""
        args = {"latency_ns": 10 * NANOS_PER_SECOND, "annualised_vol": D("0.45")}
        rth = latency_slippage_bps(**args)  # type: ignore[arg-type]
        overnight = latency_slippage_bps(**args, depth_multiplier=D("3"))  # type: ignore[arg-type]
        assert overnight == rth * 3

    def test_zero_volatility_costs_nothing(self) -> None:
        assert latency_slippage_bps(latency_ns=NANOS_PER_SECOND, annualised_vol=D("0")) == 0

    def test_a_rejected_order_cannot_be_priced(self) -> None:
        with pytest.raises(LatencyError, match="rejected order"):
            latency_slippage_bps(latency_ns=-1000, annualised_vol=D("0.45"))

    def test_negative_volatility_is_rejected(self) -> None:
        with pytest.raises(LatencyError):
            latency_slippage_bps(latency_ns=1000, annualised_vol=D("-0.1"))

    def test_a_thinking_budget_has_a_price_in_basis_points(self) -> None:
        """The number that makes "use the bigger budget" an argued choice, not a reflex."""
        vol = D("0.45")  # a plausible annualised vol for a single-name rToken
        cheap = thinking_budget_cost_bps(think_ms=8_000, annualised_vol=vol)
        dear = thinking_budget_cost_bps(think_ms=40_000, annualised_vol=vol)
        assert dear > cheap > 0
        # At RTH depth, 40s of thinking is 5.07bps against a 12bps round trip - large, but payable.
        assert Decimal("5") < dear < Decimal("6")

    def test_off_hours_deliberation_costs_more_than_the_trade(self) -> None:
        """The finding we expected to come out negligible, and did not.

        A full-reasoning budget off-hours costs more in expected adverse move than the entire
        round-trip fee. The intuition that "thinking is free because we hold for a day" is wrong on
        this venue, and this test is here so it cannot quietly become true again.
        """
        round_trip_bps = Decimal("12")
        off_hours = thinking_budget_cost_bps(
            think_ms=40_000, annualised_vol=D("0.45"), depth_multiplier=D("3")
        )
        assert off_hours > round_trip_bps
        assert Decimal("15") < off_hours < Decimal("16")

    def test_a_cheaper_budget_is_the_lever(self) -> None:
        """The trade the Meta-PM is now held to: accuracy against staleness."""
        vol, thin = D("0.45"), D("3")
        low = thinking_budget_cost_bps(think_ms=8_000, annualised_vol=vol, depth_multiplier=thin)
        full = thinking_budget_cost_bps(think_ms=40_000, annualised_vol=vol, depth_multiplier=thin)
        assert full - low > Decimal("8"), "the budget choice must be worth arguing about"

    def test_the_thin_book_version_is_charged_more(self) -> None:
        vol = D("0.45")
        assert thinking_budget_cost_bps(
            think_ms=40_000, annualised_vol=vol, depth_multiplier=D("3")
        ) > thinking_budget_cost_bps(think_ms=40_000, annualised_vol=vol)


class TestClockOffset:
    """Two clocks, an unknown constant between them, and a negative leg that would read as a
    rejection if the offset were ignored."""

    def test_the_midpoint_estimator_recovers_a_known_offset(self) -> None:
        offset_ns = 7_000 * MS
        samples = [
            Sample(req_ns=1_000 * MS, resp_ns=1_100 * MS,
                   venue_ms=(1_050 * MS - offset_ns) // 1_000_000, ok=True),
            Sample(req_ns=2_000 * MS, resp_ns=2_100 * MS,
                   venue_ms=(2_050 * MS - offset_ns) // 1_000_000, ok=True),
        ]
        assert estimate_clock_offset_ns(samples) == pytest.approx(offset_ns, abs=MS)

    def test_the_median_resists_one_stalled_request(self) -> None:
        good = [
            Sample(req_ns=t * MS, resp_ns=(t + 100) * MS, venue_ms=t + 50, ok=True)
            for t in (1_000, 2_000, 3_000)
        ]
        stalled = Sample(req_ns=4_000 * MS, resp_ns=14_000 * MS, venue_ms=4_050, ok=True)
        assert estimate_clock_offset_ns(good) == estimate_clock_offset_ns([*good, stalled])

    def test_no_venue_timestamp_means_no_offset_not_zero(self) -> None:
        samples = [Sample(req_ns=0, resp_ns=MS, venue_ms=None, ok=True)]
        assert estimate_clock_offset_ns(samples) is None

    def test_failed_samples_are_excluded(self) -> None:
        samples = [Sample(req_ns=0, resp_ns=MS, venue_ms=5, ok=False, error="URLError")]
        assert estimate_clock_offset_ns(samples) is None

    def test_an_impossible_row_is_dropped_not_written_as_a_rejection(self) -> None:
        """A negative entry leg is hftbacktest's rejection marker. It must never come from skew."""
        # venue timestamp lands outside the round trip: the offset did not hold for this sample.
        bad = Sample(req_ns=1_000 * MS, resp_ns=1_100 * MS, venue_ms=9_999, ok=True)
        good = Sample(req_ns=1_000 * MS, resp_ns=1_100 * MS, venue_ms=1_050, ok=True)
        rows = to_rows([bad, good], offset_ns=0)
        assert len(rows) == 1
        assert all(not r.was_rejected for r in rows)
        assert all(r.entry_ns > 0 and r.response_ns > 0 for r in rows)


class TestTheRecordedSeries:
    """The artefact on disk is a real measurement, and it must stay usable."""

    PATH = Path(__file__).resolve().parents[1] / "data" / "latency_probe.json"

    def test_the_probe_artefact_exists(self) -> None:
        assert self.PATH.exists(), "run `python -m argus.execution.latency_probe`"

    def test_it_records_a_passing_measurement(self) -> None:
        report = json.loads(self.PATH.read_text(encoding="utf-8"))
        assert report["verdict"].startswith("PASS")
        assert report["samples_ok"] >= 2

    def test_the_rows_drive_the_interpolator(self) -> None:
        rows = load_rows(self.PATH)
        model = InterpolatedLatency(rows)
        mid = (rows[0].req_ts + rows[-1].req_ts) // 2
        entry = model.entry(mid)
        assert entry > 0, "a real measured entry latency cannot be negative or a rejection"
        assert model.percentile_ns(0.99) >= model.percentile_ns(0.5)

    def test_the_measured_latency_is_priced(self) -> None:
        """What our actual connection costs on a single decision."""
        rows = load_rows(self.PATH)
        model = InterpolatedLatency(rows)
        cost = latency_slippage_bps(
            latency_ns=model.percentile_ns(0.99), annualised_vol=D("0.45")
        )
        assert cost > 0
        assert cost < Decimal("12"), "network latency alone should not exceed the round-trip fee"
