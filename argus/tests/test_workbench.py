"""Track 3 workbench tests — all six sub-themes.

The defining property of this track is inverted from Track 2: the human decides. Nothing here may
place an order, and every claim must be traceable to a locator.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import ClassVar

import pytest

from argus.cost.model import CostModel
from argus.desk.workbench import (
    Autopsy,
    ClaimGraph,
    ErrorProfile,
    Extracted,
    Holding,
    Locator,
    TraderProfile,
    UnsourcedClaim,
    assess_trade,
    plan_execution,
    stress_position,
)
from argus.execution.passive import BookObservation

NOW = datetime(2026, 3, 9, 14, 0, tzinfo=UTC)


def _ext(label: str, value: str) -> Extracted:
    return Extracted(
        label=label, value=Decimal(value), unit="m",
        locator=Locator("NVDA 10-Q", page=84, bbox=(72, 331, 540, 388), cell="R4C2"),
        as_of=NOW,
    )


class TestInformationExtraction:
    """Sub-theme 1: a claim is never separable from its evidence."""

    def test_an_unsourced_claim_is_refused(self) -> None:
        g = ClaimGraph()
        with pytest.raises(UnsourcedClaim, match="does not appear"):
            g.assert_claim("margins are deteriorating", [])

    def test_trace_walks_from_claim_to_page_and_cell(self) -> None:
        g = ClaimGraph()
        g.assert_claim(
            "gross margin fell 250bps",
            [_ext("revenue", "18200"), _ext("cogs", "9100")],
            derivation="(revenue - cogs) / revenue vs prior quarter",
        )
        trace = g.trace("gross margin fell 250bps")
        assert "p84" in " ".join(trace)
        assert "R4C2" in " ".join(trace)
        assert "computed as" in " ".join(trace)

    def test_tracing_an_unknown_claim_raises(self) -> None:
        with pytest.raises(UnsourcedClaim, match="not on this graph"):
            ClaimGraph().trace("something nobody said")


class TestReviewAndSelfEvolution:
    """Sub-theme 2: 0/12 corpus sources do post-trade attribution."""

    def _autopsy(self, correct: bool, conf: float, mag_err: int = 0) -> Autopsy:
        return Autopsy(
            decision_id="d", thesis="t",
            predicted_direction="up", realised_direction="up" if correct else "down",
            predicted_magnitude_bps=50, realised_magnitude_bps=50 - mag_err,
            stated_confidence=conf,
        )

    def test_the_four_failure_modes_are_distinguished(self) -> None:
        assert self._autopsy(True, 0.8).failure_mode == "correct"
        assert self._autopsy(True, 0.8, mag_err=200).failure_mode == "magnitude"
        assert self._autopsy(False, 0.9).failure_mode == "overconfident"
        assert self._autopsy(False, 0.3).failure_mode == "thesis"

    def test_calibration_gap_detects_overconfidence(self) -> None:
        p = ErrorProfile()
        for _ in range(8):
            p.add(self._autopsy(False, 0.9))
        for _ in range(2):
            p.add(self._autopsy(True, 0.9))
        assert p.calibration_gap == pytest.approx(0.7, abs=0.01)
        assert any("overconfident" in f for f in p.findings())

    def test_small_samples_refuse_to_draw_a_pattern(self) -> None:
        p = ErrorProfile()
        p.add(self._autopsy(False, 0.9))
        assert "too few" in p.findings()[0]

    def test_repeated_sizing_errors_are_named_as_such(self) -> None:
        p = ErrorProfile()
        for _ in range(10):
            p.add(self._autopsy(True, 0.6, mag_err=300))
        assert any("sizing problem" in f for f in p.findings())


class TestStressTesting:
    """Sub-theme 3: exit cost must rise as liquidity falls."""

    def test_a_position_is_hardest_to_leave_when_you_most_want_to(self) -> None:
        rows = stress_position(
            quantity=Decimal("100"), entry_price=Decimal("200"),
            loss_tolerance_pct=Decimal("10"),
        )
        by_name = {r.scenario: r for r in rows}
        assert by_name["liquidity_collapse"].exit_cost_bps > by_name["base"].exit_cost_bps

    def test_a_large_gap_breaches_tolerance(self) -> None:
        rows = stress_position(
            quantity=Decimal("100"), entry_price=Decimal("200"),
            loss_tolerance_pct=Decimal("10"),
        )
        assert {r.scenario for r in rows if not r.survives} == {"gap_down"}


class TestPortfolioCopilot:
    """Sub-theme 6: the sentence a generic desk cannot produce."""

    # A diversified book. An undiversified one breaches every mandate for every trade, which
    # tests nothing — the first version of this fixture was 57% rNVDA and both profiles failed
    # identically.
    holdings: ClassVar[list[Holding]] = [
        Holding("rNVDA", Decimal("45"), Decimal("220"), "semis"),      # ~10k
        Holding("rAAPL", Decimal("60"), Decimal("330"), "hardware"),   # ~20k
        Holding("rMSFT", Decimal("60"), Decimal("495"), "software"),   # ~30k
        Holding("rAMZN", Decimal("78"), Decimal("256"), "retail"),     # ~20k
        Holding("rMETA", Decimal("31"), Decimal("648"), "media"),      # ~20k
    ]

    def test_a_good_trade_can_be_bad_for_the_book(self) -> None:
        got = assess_trade(
            holdings=self.holdings, symbol="rNVDA", sector="semis",
            trade_notional=Decimal("30000"), profile=TraderProfile.conservative(),
        )
        assert got.permitted is False
        assert "bad trade for your book" in got.verdict()
        assert any("single-name cap" in b for b in got.breaches)

    def test_the_same_trade_passes_for_a_different_mandate(self) -> None:
        """Personalisation is real only if the profile changes the verdict."""
        # 10k lifts rNVDA to ~18% of the book: past a 5% conservative cap, inside a 25%
        # aggressive one. The trade is identical; only the mandate differs.
        trade = dict(holdings=self.holdings, symbol="rNVDA", sector="semis",
                     trade_notional=Decimal("10000"))
        conservative = assess_trade(**trade, profile=TraderProfile.conservative())  # type: ignore[arg-type]
        aggressive = assess_trade(**trade, profile=TraderProfile.aggressive())  # type: ignore[arg-type]
        assert conservative.permitted != aggressive.permitted

    def test_both_sides_of_the_number_are_reported(self) -> None:
        got = assess_trade(
            holdings=self.holdings, symbol="rNVDA", sector="semis",
            trade_notional=Decimal("5000"), profile=TraderProfile.aggressive(),
        )
        assert got.position_pct_after > got.position_pct_before


class TestExecutionAssistance:
    """Sub-theme 5: a thin session is not a good time to be passive."""

    def test_thin_sessions_reduce_passive_weight(self) -> None:
        """hftbacktest's fill condition requires queue advance; on a third-depth book a resting
        order is adversely selected."""
        liquid = BookObservation(level_qty=Decimal("5000"), traded_qty=Decimal("40000"))
        asleep = plan_execution(
            symbol="rNVDA", notional=Decimal("20000"),
            adv_notional=Decimal("5000000"), anchor_asleep=True, book=liquid,
        )
        awake = plan_execution(
            symbol="rNVDA", notional=Decimal("20000"), adv_notional=Decimal("5000000"),
            book=liquid,
        )
        passive_asleep = sum(s.fraction for s in asleep.slices if "passive" in s.style)
        passive_awake = sum(s.fraction for s in awake.slices if "passive" in s.style)
        assert passive_asleep < passive_awake
        assert "moves against them" in asleep.rationale

    def test_a_passive_slice_without_a_book_is_quoted_at_taker(self) -> None:
        """The planner used to quote maker_bps on every slice it called "passive limit".

        That is an unevidenced claim about fills - the exact defect the backtest engine refuses
        outright - so with no book the honest quote is the taker rate, and the label says so.
        """
        model = CostModel.bitget_perp()
        plan = plan_execution(
            symbol="rNVDA", notional=Decimal("20000"), adv_notional=Decimal("5000000"),
        )
        assert all(s.expected_cost_bps == model.taker_bps for s in plan.slices)
        assert all("passive limit" not in s.style for s in plan.slices)
        assert "upper bound rather than a forecast" in plan.rationale

    def test_a_book_makes_the_passive_slices_real(self) -> None:
        """With a book, a passive slice is priced through the queue model, not assumed."""
        model = CostModel.bitget_perp()
        liquid = BookObservation(level_qty=Decimal("100"), traded_qty=Decimal("100000"))
        plan = plan_execution(
            symbol="rNVDA", notional=Decimal("20000"), adv_notional=Decimal("5000000"),
            book=liquid,
        )
        passive = [s for s in plan.slices if "passive limit" in s.style]
        assert passive, "a liquid book should produce passive slices"
        assert all(s.expected_cost_bps == model.maker_bps for s in passive)
        assert "fill 100%" in passive[0].style

    def test_an_illiquid_book_prices_passive_above_maker(self) -> None:
        """The cost of a passive slice that does not fill is the taker fee it gets chased at."""
        model = CostModel.bitget_perp()
        thin = BookObservation(level_qty=Decimal("500000"), traded_qty=Decimal("10"))
        plan = plan_execution(
            symbol="rNVDA", notional=Decimal("20000"), adv_notional=Decimal("5000000"),
            book=thin,
        )
        passive = [s for s in plan.slices if "passive limit" in s.style]
        assert passive
        assert passive[0].expected_cost_bps > model.maker_bps
        assert "fill 0%" in passive[0].style

    def test_large_footprint_forces_urgency(self) -> None:
        got = plan_execution(
            symbol="rNVDA", notional=Decimal("500000"), adv_notional=Decimal("1000000"),
        )
        assert got.slices[0].style == "market"

    def test_cost_rises_with_participation(self) -> None:
        small = plan_execution(symbol="rNVDA", notional=Decimal("1000"),
                               adv_notional=Decimal("10000000"))
        big = plan_execution(symbol="rNVDA", notional=Decimal("400000"),
                             adv_notional=Decimal("10000000"))
        assert big.expected_total_cost_bps > small.expected_total_cost_bps


class TestATinyOrderIsSentWhole:
    """Asking the hosted console how to sell $800 of AAPL returned a two-slice plan for an order
    worth 0.01% of the day's volume. Splitting cannot pay for itself at that size."""

    def test_below_the_threshold_there_is_one_slice(self) -> None:
        from decimal import Decimal

        from argus.desk.workbench import plan_execution

        plan = plan_execution(symbol="AAPLUSDT", notional=Decimal("800"),
                              adv_notional=Decimal("8000000"), anchor_asleep=True)
        assert len(plan.slices) == 1
        assert plan.slices[0].fraction == 1

    def test_a_large_order_is_still_split(self) -> None:
        from decimal import Decimal

        from argus.desk.workbench import plan_execution

        plan = plan_execution(symbol="AAPLUSDT", notional=Decimal("500000"),
                              adv_notional=Decimal("8000000"))
        assert len(plan.slices) > 1
