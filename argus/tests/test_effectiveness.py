"""Hedge-effectiveness tests — three constants in a risk layer, replaced by three measurements.

The values this module computes feed `risk/hedgeability.py`, where they are multiplied together to
price a hedge. Before it existed they were literals: ``risk_reduction=0.98``,
``correlation_confidence=0.98``, ``basis_stability=0.95``. Nothing checked them because there was
nothing to check them against.

So these tests do two jobs. The first is ordinary correctness on cases where the answer is known by
construction — a perfect hedge, an uncorrelated one, an anticorrelated one, one that needs sizing.
The second is the property that matters more: **every path that cannot measure must refuse**, and a
candidate built from a measurement must carry the sample that produced it.
"""

from __future__ import annotations

import json
import math
import random
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from argus.risk.effectiveness import (
    MIN_PAIRS,
    STALE_AFTER_HOURS,
    EffectivenessError,
    HedgeEffectiveness,
    Provenance,
    fisher_interval,
    load,
    lookup,
    measure,
)
from argus.risk.hedgeability import (
    HedgeabilitySurface,
    open_market_candidate,
    shut_market_candidate,
)

NOW = datetime(2026, 9, 14, 12, tzinfo=UTC)


def _walk(n: int, *, seed: int, start: float = 100.0, sigma: float = 0.01) -> list[float]:
    rng = random.Random(seed)
    out = [start]
    for _ in range(n - 1):
        out.append(out[-1] * math.exp(rng.gauss(0, sigma)))
    return out


def _measured(**kwargs: object) -> HedgeEffectiveness:
    base = {
        "spot": "NVDAUSDT", "hedge": "NVDA index", "phase": "rth", "observations": 251,
        "correlation": 0.999, "correlation_low": 0.998, "hedge_ratio": 0.997,
        "r_squared": 0.998, "unit_ratio_effectiveness": 0.997,
        "measured_at": NOW, "window_days": 60,
    }
    base.update(kwargs)
    return HedgeEffectiveness(**base)  # type: ignore[arg-type]


class TestKnownAnswers:
    def test_a_perfect_hedge_removes_everything(self) -> None:
        spot = _walk(400, seed=1)
        got = measure(spot, spot, spot="A", hedge="B")
        assert got.correlation == pytest.approx(1.0, abs=1e-9)
        assert got.r_squared == pytest.approx(1.0, abs=1e-9)
        assert got.unit_ratio_effectiveness == pytest.approx(1.0, abs=1e-9)
        assert got.hedge_ratio == pytest.approx(1.0, abs=1e-9)

    def test_an_unrelated_hedge_removes_nothing_and_gets_no_credit(self) -> None:
        got = measure(_walk(600, seed=2), _walk(600, seed=3), spot="A", hedge="B")
        assert abs(got.correlation) < 0.2
        assert got.r_squared < 0.05
        # The Fisher bound can sit below zero here, and the credit for "we cannot tell" is none.
        assert got.correlation_confidence >= Decimal("0")

    def test_an_anticorrelated_hedge_is_not_credited_as_confidence(self) -> None:
        """A -1 correlation hedges beautifully *if you flip the sign*, and the risk layer does not
        flip signs. Crediting |r| here would report a short hedge as a long one."""
        spot = _walk(400, seed=4)
        mirror = [2 * spot[0] - v for v in spot]
        got = measure(spot, mirror, spot="A", hedge="B")
        assert got.correlation < -0.9
        assert got.correlation_confidence == Decimal("0")

    def test_a_hedge_that_moves_twice_as_hard_needs_sizing(self) -> None:
        """The gap this module exists to expose: at the optimal ratio the hedge is perfect, and
        one-for-one it is worse than no hedge at all. A single 'basis stability' constant cannot
        say that; two measurements can."""
        spot = _walk(400, seed=5)
        doubled = [spot[0] * (v / spot[0]) ** 2 for v in spot]
        got = measure(spot, doubled, spot="A", hedge="B")
        assert got.r_squared == pytest.approx(1.0, abs=1e-6)
        assert got.hedge_ratio == pytest.approx(0.5, abs=1e-6)
        # For this seed the 1:1 hedge's residual variance lands within machine noise of the raw
        # spot variance (`unit_ratio_effectiveness` measured at ~2e-15 on one platform, ~-2e-15 on
        # another), because `math.exp`'s last-bit rounding differs between glibc (Linux CI) and
        # the Windows ucrt, and `_walk` compounds 400 of them. A strict `< 0` chases that rounding
        # noise instead of the property under test — that a naive 1:1 hedge on an instrument
        # moving twice as hard is no better than doing nothing — so this allows a hair of positive
        # slack no larger than a handful of ULPs at this magnitude.
        assert got.unit_ratio_effectiveness < 1e-9
        assert got.basis_stability == Decimal("0")

    def test_the_fisher_interval_matches_the_textbook(self) -> None:
        low, high = fisher_interval(0.5, 100)
        assert low == pytest.approx(0.337, abs=5e-4)
        assert high == pytest.approx(0.634, abs=5e-4)

    def test_more_observations_narrow_the_interval(self) -> None:
        narrow = fisher_interval(0.9, 1000)
        wide = fisher_interval(0.9, 30)
        assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])

    def test_a_perfect_correlation_does_not_blow_up_the_transform(self) -> None:
        low, high = fisher_interval(1.0, 200)
        assert low == pytest.approx(1.0, abs=1e-6)
        assert high == pytest.approx(1.0, abs=1e-6)


class TestItRefusesRatherThanDefaults:
    def test_too_few_observations_raise(self) -> None:
        with pytest.raises(EffectivenessError, match="below the"):
            measure(_walk(MIN_PAIRS, seed=6), _walk(MIN_PAIRS, seed=7), spot="A", hedge="B")

    def test_mismatched_lengths_raise(self) -> None:
        with pytest.raises(EffectivenessError, match="differ in length"):
            measure(_walk(200, seed=8), _walk(199, seed=9), spot="A", hedge="B")

    def test_a_non_positive_price_raises(self) -> None:
        bad = _walk(200, seed=10)
        bad[50] = 0.0
        with pytest.raises(EffectivenessError, match="positive"):
            measure(bad, _walk(200, seed=11), spot="A", hedge="B")

    def test_a_motionless_position_has_no_effectiveness_to_measure(self) -> None:
        with pytest.raises(EffectivenessError, match="no risk to hedge"):
            measure([100.0] * 200, _walk(200, seed=12), spot="A", hedge="B")

    def test_a_motionless_hedge_raises(self) -> None:
        with pytest.raises(EffectivenessError, match="does not move"):
            measure(_walk(200, seed=13), [100.0] * 200, spot="A", hedge="B")

    def test_a_tiny_sample_cannot_have_an_interval(self) -> None:
        with pytest.raises(EffectivenessError, match="more than three"):
            fisher_interval(0.5, 3)


class TestStalenessIsAbsence:
    def test_a_fresh_measurement_is_returned(self) -> None:
        table = {("NVDAUSDT", "rth"): _measured()}
        assert lookup("NVDAUSDT", "rth", table=table, now=NOW) is not None

    def test_a_stale_measurement_is_not_a_weaker_one(self) -> None:
        """A correlation from three weeks ago describes a different market, so it is treated as
        missing rather than as a slightly worse estimate."""
        table = {("NVDAUSDT", "rth"): _measured()}
        later = NOW + timedelta(hours=STALE_AFTER_HOURS + 1)
        assert lookup("NVDAUSDT", "rth", table=table, now=later) is None

    def test_an_unknown_phase_is_none(self) -> None:
        table = {("NVDAUSDT", "rth"): _measured()}
        assert lookup("NVDAUSDT", "weekend", table=table, now=NOW) is None

    def test_a_missing_file_is_an_empty_table_not_a_default(self, tmp_path: Path) -> None:
        assert load(tmp_path / "nothing.json") == {}

    def test_a_malformed_row_is_skipped_rather_than_guessed(self, tmp_path: Path) -> None:
        path = tmp_path / "measurements.json"
        good = _measured().as_dict()
        path.write_text(
            json.dumps({"measurements": [good, {"spot": "X"}]}), encoding="utf-8",
        )
        assert list(load(path)) == [("NVDAUSDT", "rth")]

    def test_it_round_trips(self) -> None:
        first = _measured()
        again = HedgeEffectiveness.from_dict(first.as_dict())
        assert again.correlation_low == pytest.approx(first.correlation_low)
        assert again.measured_at == first.measured_at


class TestTheRiskLayerRecordsWhereItsNumbersCameFrom:
    def test_a_measured_candidate_carries_its_sample(self) -> None:
        candidate = open_market_candidate("NVDA", measured=_measured())
        assert candidate.provenance is Provenance.MEASURED
        assert "n=251" in candidate.evidence and "rth" in candidate.evidence
        assert candidate.risk_reduction == Decimal("0.998")
        assert candidate.correlation_confidence == Decimal("0.998")
        assert candidate.basis_stability == Decimal("0.997")

    def test_an_unmeasured_candidate_is_stamped_assumed(self) -> None:
        candidate = open_market_candidate("NVDA", Decimal("0.98"))
        assert candidate.provenance is Provenance.ASSUMED
        assert candidate.evidence == ""

    def test_a_measured_candidate_without_a_sample_is_refused(self) -> None:
        """An unattributed measurement is an assertion with a better label, and this is the one
        place that distinction could be quietly lost."""
        from argus.risk.hedgeability import HedgeCandidate

        with pytest.raises(ValueError, match="must name its sample"):
            HedgeCandidate(
                instrument="NVDA", risk_reduction=Decimal("0.9"),
                correlation_confidence=Decimal("0.9"), liquidity_availability=Decimal("1"),
                execution_probability=Decimal("1"), basis_stability=Decimal("0.9"),
                execution_cost_bps=Decimal("5"), provenance=Provenance.MEASURED,
            )

    def test_a_shut_venue_zeroes_a_perfectly_good_hedge(self) -> None:
        """The Sleeping-Anchor sentence, kept separable: the statistics can be excellent and the
        hedge still unplaceable, and the record must say which of the two is true."""
        candidate = shut_market_candidate("NVDA", measured=_measured())
        assert candidate.provenance is Provenance.MEASURED
        assert candidate.risk_reduction > Decimal("0.9")
        assert candidate.effective_risk_reduction == Decimal("0")
        assert not candidate.is_placeable

    def test_the_measured_factors_differ_from_the_constants_they_replaced(self) -> None:
        """If they agreed there would be no point measuring. They do not: the constants understated
        the regular-hours hedge."""
        measured = open_market_candidate("NVDA", measured=_measured())
        assumed = open_market_candidate("NVDA", Decimal("0.98"))
        assert measured.risk_reduction != assumed.risk_reduction
        assert measured.basis_stability > assumed.basis_stability

    def test_the_surface_still_ranks_with_measured_candidates(self) -> None:
        surface = HedgeabilitySurface(
            (open_market_candidate(
                "NVDA", execution_probability=Decimal("1"), measured=_measured(),
            ),),
        )
        assert surface.menu
        assert surface.menu[0].risk_neutralisation_efficiency > 0


class TestTheLiveMeasurement:
    """The file the runner actually reads. Skipped where it has not been generated."""

    def test_every_stored_row_is_usable(self) -> None:
        from argus.risk.effectiveness import REPORT_PATH

        if not REPORT_PATH.exists():
            pytest.skip("no measurement file on this machine")
        table = load()
        assert table, "the measurement file exists but produced no usable rows"
        for (symbol, phase), row in table.items():
            assert symbol.endswith("USDT")
            assert row.phase == phase
            assert row.observations >= MIN_PAIRS
            assert Decimal("0") <= row.risk_reduction <= Decimal("1")
            assert Decimal("0") <= row.correlation_confidence <= Decimal("1")
            assert Decimal("0") <= row.basis_stability <= Decimal("1")

    def test_the_runner_stamps_provenance_from_it(self) -> None:
        from argus.paper.runner import _hedge_surface
        from argus.risk.effectiveness import REPORT_PATH

        if not REPORT_PATH.exists():
            pytest.skip("no measurement file on this machine")
        surface = _hedge_surface(False, "NVDAUSDT")
        candidate = surface.candidates[0]
        assert candidate.provenance in (Provenance.MEASURED, Provenance.ASSUMED)
        if candidate.provenance is Provenance.MEASURED:
            assert "n=" in candidate.evidence
            assert "measured" in surface.session_note.lower()
        else:
            assert "ASSUMED" in surface.session_note
