"""Gate-screen tests — and the reason this module's result is *worse* than the claim it replaced.

`data/overfit_gates.json` was cited in `README.md` for the project's strongest negative finding, and
**nothing could regenerate it.** Rebuilding the measurement did not reproduce the stored four-and-
four split: six primitives are refuted outright, two return no verdict for want of data, none clears
every gate, and all eight are net-negative after the fee.

The bucket size the gates need was not recoverable from the artefact, and it was **not tuned until
the old numbers came back** — calibrating a parameter to reproduce a stored answer is the exact
failure these gates exist to detect. `data/factor_lab.json` reaches the same all-eight verdict by an
independent route, which is the corroboration that matters.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from argus.backtest.engine import Bar
from argus.research.factor_lab import PRIMITIVES
from argus.research.overfit_gates import (
    FEE_BPS,
    PERIOD_BARS,
    REPORT_PATH,
    GatesError,
    observe,
    report_of,
    screen,
)


def _bars(n: int, *, start: float = 100.0, drift: float = 0.0) -> list[Bar]:
    at = datetime(2026, 6, 1, tzinfo=UTC)
    out: list[Bar] = []
    price = start
    for i in range(n):
        price *= 1 + drift
        out.append(
            Bar(ts=at + timedelta(hours=i), close=Decimal(str(round(price, 6))),
                extra={"volume": 1.0, "high": price, "low": price})
        )
    return out


class TestTheObservationsAreStrictlyForward:
    def test_the_signal_never_sees_its_own_return(self) -> None:
        bars = _bars(200, drift=0.001)
        got = observe(bars, "long_while_closed")
        assert len(got) == len(bars) - 1, "one observation per bar that has a successor"

    def test_periods_group_bars_so_the_rank_ic_has_a_cross_section(self) -> None:
        """`overfit.period_ic` correlates *across* the rows sharing a period. One row per period
        makes every IC undefined and every gate INCONCLUSIVE — which is why bars are bucketed."""
        bars = _bars(PERIOD_BARS * 3 + 1, drift=0.0005)
        periods = {o.period for o in observe(bars, "long_while_closed")}
        assert len(periods) >= 3
        assert max(periods) == (len(bars) - 2) // PERIOD_BARS

    def test_every_vetted_primitive_can_be_observed(self) -> None:
        bars = _bars(200, drift=0.0005)
        for name in PRIMITIVES:
            assert observe(bars, name), name


class TestItRefusesRatherThanGuessing:
    def test_an_unknown_factor_raises_and_names_the_known_ones(self) -> None:
        with pytest.raises(GatesError, match="not a vetted primitive"):
            observe(_bars(50), "made_up_factor")

    def test_too_little_history_raises_rather_than_returning_a_verdict(self) -> None:
        with pytest.raises(GatesError, match="below the"):
            screen(_bars(5), "long_while_closed")

    def test_a_non_positive_close_raises(self) -> None:
        bars = _bars(60)
        broken = list(bars)
        broken[0] = Bar(ts=bars[0].ts, close=Decimal("0"), extra=bars[0].extra)
        with pytest.raises(GatesError, match="not a price"):
            observe(broken, "long_while_closed")


class TestUnevaluatedIsNotRejected:
    """The absence-is-not-zero rule, applied to this module's own output.

    The first version folded *"not proven: insufficient data for a verdict"* in with the refuted
    factors and reported 8 noise primitives when 6 were refuted and 2 were merely unmeasured.
    `docclaims` caught it. A factor the gates could not evaluate has not been rejected.
    """

    def test_the_report_separates_refuted_from_unevaluated(self) -> None:
        if not REPORT_PATH.exists():
            pytest.skip("no live artefact on this machine")
        blob = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        assert "rejected_as_noise" in blob
        assert "not_proven_insufficient_data" in blob, "unevaluated needs its own bucket"
        overlap = set(blob["rejected_as_noise"]) & set(blob["not_proven_insufficient_data"])
        assert not overlap, f"a factor cannot be both refuted and unmeasured: {overlap}"

    def test_the_three_buckets_partition_the_primitives(self) -> None:
        if not REPORT_PATH.exists():
            pytest.skip("no live artefact on this machine")
        blob = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        covered = (
            set(blob["rejected_as_noise"])
            | set(blob["not_proven_insufficient_data"])
            | set(blob["cleared_every_gate"])
        )
        assert covered == set(PRIMITIVES), "every primitive must land in exactly one bucket"

    def test_the_finding_sentence_counts_what_the_buckets_hold(self) -> None:
        if not REPORT_PATH.exists():
            pytest.skip("no live artefact on this machine")
        blob = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        finding = blob["finding"]
        assert f"{len(blob['rejected_as_noise'])} of {len(PRIMITIVES)}" in finding
        assert "not a pass" in finding, "the unevaluated bucket must be stated, not implied"


class TestTheFeeIsChargedOnTurnover:
    def test_a_flat_signal_pays_once_not_every_bar(self) -> None:
        """A factor that holds the same position pays the round trip when it opens, not per bar."""
        bars = _bars(PERIOD_BARS * 4, drift=0.0004)
        got = screen(bars, "long_while_closed")
        gap = abs(got.gross_sharpe - got.net_sharpe)
        assert gap < 5.0, "a constant-position factor should not be crushed by turnover"

    def test_the_fee_only_ever_hurts(self) -> None:
        bars = _bars(PERIOD_BARS * 4, drift=0.0004)
        for name in ("long_while_closed", "closure_momentum", "slow_trend"):
            got = screen(bars, name)
            assert got.net_sharpe <= got.gross_sharpe + 1e-9, name

    def test_the_fee_is_the_one_the_rest_of_the_system_charges(self) -> None:
        assert FEE_BPS == 12.0


class TestTheArtefactIsReproducible:
    def test_report_of_emits_the_shape_the_readme_cites(self) -> None:
        bars = _bars(PERIOD_BARS * 4, drift=0.0004)
        outcomes = [screen(bars, n) for n in list(PRIMITIVES)[:3]]
        blob = report_of(outcomes, instrument="NVDAUSDT", bars=len(bars), interval="1H")
        assert set(blob) >= {
            "measured_at", "instrument", "bars", "interval", "fee_bps", "period_bars",
            "rejected_as_noise", "not_proven_insufficient_data", "cleared_every_gate",
            "cleared_but_unprofitable_after_fees", "net_negative_after_fees", "finding", "detail",
        }
        assert len(blob["detail"]) == 3

    def test_the_live_artefact_records_the_bucket_size_it_used(self) -> None:
        """`PERIOD_BARS` is inferred, not recovered. Storing it is what lets a later reader
        disagree with the inference instead of being unable to see it."""
        if not REPORT_PATH.exists():
            pytest.skip("no live artefact on this machine")
        blob = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        assert blob["period_bars"] == PERIOD_BARS

    def test_the_live_artefact_covers_every_vetted_primitive(self) -> None:
        if not REPORT_PATH.exists():
            pytest.skip("no live artefact on this machine")
        blob = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        assert {row["factor"] for row in blob["detail"]} == set(PRIMITIVES)
