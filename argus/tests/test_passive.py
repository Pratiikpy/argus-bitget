"""Passive-execution tests.

One number motivates this whole file: a sim assuming limit fills returned +90% where the replay
returned -0.45%. These tests assert that the engine cannot reproduce that error — that asking for a
maker fee is a request to *simulate* fills, never a discount granted on request.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from argus.backtest.engine import Bar, run
from argus.cost.model import CostModel
from argus.execution.passive import (
    BookObservation,
    PassiveExecution,
    PassiveExecutionError,
    book_from_bar,
)
from argus.execution.queue import ProbQueue, RiskAdverseQueue

D = Decimal
COST = CostModel.bitget_perp()


# A fixed, uneven return pattern. Constant drift would give a strategy with no losing bar, and
# Sortino is undefined without downside - a degenerate fixture that tests nothing.
_PATTERN = ["0.011", "-0.004", "0.007", "0.019", "-0.013", "0.002", "-0.006", "0.015"]


def _bars(n: int, *, book: dict[str, object] | None) -> list[Bar]:
    """A trending-but-choppy series, so a long strategy has a real gross edge to lose."""
    start = datetime(2026, 3, 2, 14, 30, tzinfo=UTC)
    px = D("100")
    out: list[Bar] = []
    for i in range(n):
        px = px * (D("1") + D(_PATTERN[i % len(_PATTERN)]))
        out.append(Bar(ts=start + timedelta(days=i), close=px, extra=dict(book) if book else None))
    return out


def _alternating(bars: object, i: int) -> float:
    """Flip between flat and long every bar, so every bar is a rebalance."""
    return 1.0 if i % 2 == 0 else 0.0


class TestTheMakerClaimIsNotFree:
    """Requesting maker treatment requests a simulation, not a rebate."""

    def test_a_thin_book_filling_nothing_is_reported_as_the_finding(self) -> None:
        """Not a crash about zero variance - a statement that the queue never cleared."""
        deep = {"level_qty": "10000", "traded_qty": "5"}
        with pytest.raises(PassiveExecutionError, match="queue never cleared"):
            run(
                "passive-thin", "rNVDA", _bars(60, book=deep), _alternating,
                cost=COST, periods_per_year=252,
                passive=PassiveExecution(cost=COST), passive_size=D("100"),
            )

    def test_a_mostly_thin_book_scores_but_fails_the_maker_claim(self) -> None:
        """Some fills, nowhere near enough: the run scores, and says the claim does not hold."""
        result = run(
            "passive-sparse", "rNVDA",
            _bars(60, book={"level_qty": "180", "traded_qty": "200"}), _alternating,
            cost=COST, periods_per_year=252,
            passive=PassiveExecution(cost=COST), passive_size=D("100"),
        )
        assert result.passive is not None
        assert 0 < result.passive.fill_rate < 0.5
        assert not result.passive.maker_claim_holds

    def test_a_liquid_book_fills_and_pays_the_maker_rate(self) -> None:
        thin_queue = {"level_qty": "5", "traded_qty": "100000"}
        result = run(
            "passive-liquid", "rNVDA", _bars(60, book=thin_queue), _alternating,
            cost=COST, periods_per_year=252,
            passive=PassiveExecution(cost=COST), passive_size=D("100"),
        )
        assert result.passive is not None
        assert result.passive.fill_rate == pytest.approx(1.0)
        assert result.passive.maker_claim_holds
        assert result.passive.realised_fee_bps == pytest.approx(float(COST.maker_bps))

    def test_a_half_filled_order_is_charged_a_blended_rate(self) -> None:
        """Reporting the maker rate on a partly chased order is the flattery under test."""
        book = BookObservation(level_qty=D("100"), traded_qty=D("140"))
        fill = PassiveExecution(cost=COST, model=RiskAdverseQueue(), chase=True).execute(
            D("100"), book
        )
        assert fill.filled_passive == D("40")
        assert fill.chased_taker == D("60")
        blended = (D("40") * COST.maker_bps + D("60") * COST.taker_bps) / D("100")
        assert fill.fee_bps == blended
        assert fill.fee_bps > COST.maker_bps

    def test_chasing_everything_costs_the_full_taker_rate(self) -> None:
        book = BookObservation(level_qty=D("10000"), traded_qty=D("1"))
        fill = PassiveExecution(cost=COST, chase=True).execute(D("50"), book)
        assert fill.filled_passive == 0
        assert fill.chased_taker == D("50")
        assert fill.fee_bps == COST.taker_bps
        assert not fill.was_worth_resting

    def test_not_chasing_leaves_the_order_unexecuted(self) -> None:
        book = BookObservation(level_qty=D("10000"), traded_qty=D("1"))
        fill = PassiveExecution(cost=COST, chase=False).execute(D("50"), book)
        assert fill.unexecuted == D("50")
        assert fill.achieved == 0
        assert fill.fee_bps == 0

    def test_passive_never_beats_taker_on_the_fee_it_pays(self) -> None:
        """The asymmetry, asserted directly: passive can only reduce size or raise the rate."""
        for level, traded in [("0", "1000"), ("50", "100"), ("500", "100"), ("9999", "1")]:
            book = BookObservation(level_qty=D(level), traded_qty=D(traded))
            fill = PassiveExecution(cost=COST, chase=True).execute(D("100"), book)
            assert COST.maker_bps <= fill.fee_bps <= COST.taker_bps


class TestTheBookIsMandatory:
    """A maker claim with no book behind it is refused, not quietly granted."""

    def test_a_bar_without_book_data_raises(self) -> None:
        with pytest.raises(PassiveExecutionError, match="carries no"):
            run(
                "passive-blind", "rNVDA", _bars(60, book=None), _alternating,
                cost=COST, periods_per_year=252, passive=PassiveExecution(cost=COST),
            )

    def test_the_error_names_the_bar(self) -> None:
        with pytest.raises(PassiveExecutionError, match="2026-03-02"):
            run(
                "passive-blind", "rNVDA", _bars(60, book=None), _alternating,
                cost=COST, periods_per_year=252, passive=PassiveExecution(cost=COST),
            )

    def test_partial_book_data_is_not_enough(self) -> None:
        assert book_from_bar({"level_qty": "100"}) is None
        assert book_from_bar({"traded_qty": "100"}) is None
        assert book_from_bar(None) is None
        assert book_from_bar({}) is None

    def test_junk_book_data_is_refused_not_coerced(self) -> None:
        assert book_from_bar({"level_qty": "many", "traded_qty": "5"}) is None
        assert book_from_bar({"level_qty": "-5", "traded_qty": "5"}) is None

    def test_valid_book_data_reads_through(self) -> None:
        book = book_from_bar({"level_qty": "100", "traded_qty": "20", "end_level_qty": "80"})
        assert book == BookObservation(D("100"), D("20"), D("80"))

    def test_zero_passive_size_is_rejected(self) -> None:
        with pytest.raises(PassiveExecutionError, match="passive_size"):
            run(
                "passive", "rNVDA", _bars(60, book={"level_qty": "1", "traded_qty": "1"}),
                _alternating, cost=COST, periods_per_year=252,
                passive=PassiveExecution(cost=COST), passive_size=D("0"),
            )

    def test_negative_book_quantities_are_rejected(self) -> None:
        with pytest.raises(PassiveExecutionError, match="negative"):
            BookObservation(level_qty=D("-1"), traded_qty=D("10"))
        with pytest.raises(PassiveExecutionError, match="negative"):
            BookObservation(level_qty=D("10"), traded_qty=D("-1"))

    def test_a_zero_size_order_is_rejected(self) -> None:
        book = BookObservation(level_qty=D("10"), traded_qty=D("10"))
        with pytest.raises(PassiveExecutionError, match="must be positive"):
            PassiveExecution(cost=COST).execute(D("0"), book)


class TestAgainstTheTakerBaseline:
    """The comparison that would have caught the +90%/-0.45% error."""

    def test_passive_and_taker_diverge_on_a_realistic_book(self) -> None:
        book = {"level_qty": "150", "traded_qty": "400"}
        bars = _bars(80, book=book)
        taker = run("taker", "rNVDA", bars, _alternating, cost=COST, periods_per_year=252)
        passive = run(
            "passive", "rNVDA", bars, _alternating, cost=COST, periods_per_year=252,
            passive=PassiveExecution(cost=COST), passive_size=D("500"),
        )
        assert taker.passive is None
        assert passive.passive is not None
        # The taker run reaches full size every rebalance; the passive run does not.
        assert passive.passive.fill_rate < 1.0
        assert abs(passive.net.total_return) < abs(taker.net.total_return)

    def test_the_report_survives_serialisation(self) -> None:
        book = {"level_qty": "150", "traded_qty": "400"}
        result = run(
            "passive", "rNVDA", _bars(60, book=book), _alternating,
            cost=COST, periods_per_year=252,
            passive=PassiveExecution(cost=COST), passive_size=D("500"),
        )
        payload = result.as_dict()["passive"]
        assert payload is not None
        assert set(payload) >= {"fill_rate", "realised_fee_bps", "maker_claim_holds", "chase"}
        assert payload["chase"] is False

    def test_the_queue_model_is_a_stated_choice(self) -> None:
        """Two defensible models give two different answers, and neither is hidden.

        The divergence needs quantity *behind* us, which is what peak_level_qty supplies.
        """
        book = BookObservation(
            level_qty=D("100"), traded_qty=D("60"),
            peak_level_qty=D("400"), end_level_qty=D("150"),
        )
        conservative = PassiveExecution(cost=COST, model=RiskAdverseQueue()).execute(D("40"), book)
        probabilistic = PassiveExecution(cost=COST, model=ProbQueue()).execute(D("40"), book)
        assert probabilistic.filled_passive > conservative.filled_passive

    def test_without_a_peak_the_two_models_agree(self) -> None:
        """An honest limitation of bar granularity, asserted rather than left to be discovered.

        With no observed growth the order is the entire back of the level, back = 0, every
        probability function returns zero, and ProbQueue degenerates to RiskAdverseQueue.
        """
        book = BookObservation(level_qty=D("100"), traded_qty=D("30"), end_level_qty=D("50"))
        conservative = PassiveExecution(cost=COST, model=RiskAdverseQueue()).execute(D("40"), book)
        probabilistic = PassiveExecution(cost=COST, model=ProbQueue()).execute(D("40"), book)
        assert probabilistic.filled_passive == conservative.filled_passive

    def test_a_peak_below_the_level_is_rejected(self) -> None:
        with pytest.raises(PassiveExecutionError, match="below level_qty"):
            BookObservation(level_qty=D("100"), traded_qty=D("10"), peak_level_qty=D("50"))

    def test_accounting_closes(self) -> None:
        """intended = filled + chased + unexecuted, on every path."""
        for level, traded, chase in [
            ("0", "500", False), ("100", "50", True), ("900", "10", False), ("50", "60", True),
        ]:
            fill = PassiveExecution(cost=COST, chase=chase).execute(
                D("100"), BookObservation(level_qty=D(level), traded_qty=D(traded))
            )
            assert fill.filled_passive + fill.chased_taker + fill.unexecuted == fill.intended
            assert fill.filled_passive >= 0
            assert fill.unexecuted >= 0
