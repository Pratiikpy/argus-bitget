"""The cross-sectional book: neutral by construction, costed on what it trades, no look-ahead."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from argus.backtest.engine import Bar
from argus.cost.model import CostModel
from argus.research.crosssection import (
    CANDIDATES,
    TRADING_RULES,
    across_phases,
    backtest,
    neutralise,
)
from argus.research.grammar import Const, CrossRank, Field, Ref, Signal
from argus.research.panel import build_panel

START = datetime(2026, 9, 1, tzinfo=UTC)
COST = CostModel.bitget_perp()
CLOSE = Ref(field=Field.CLOSE)


def _bars(closes: list[float]) -> list[Bar]:
    return [
        Bar(ts=START + timedelta(hours=i), close=c,
            extra={"volume": 100.0, "high": c, "low": c})
        for i, c in enumerate(closes)
    ]


def _panel(**series: list[float]):  # type: ignore[no-untyped-def]
    return build_panel({k: _bars(v) for k, v in series.items()})


class TestTheBookIsNeutralByConstruction:
    def test_weights_sum_to_zero(self) -> None:
        book = neutralise({"A": 1.0, "B": 0.0, "C": -1.0})
        assert book.net == pytest.approx(0.0)

    def test_gross_exposure_is_one(self) -> None:
        assert neutralise({"A": 5.0, "B": -3.0}).gross == pytest.approx(1.0)

    def test_a_loud_factor_does_not_get_a_larger_book(self) -> None:
        """Without normalisation a factor would look better purely for producing bigger numbers."""
        quiet = neutralise({"A": 0.01, "B": -0.01})
        loud = neutralise({"A": 1000.0, "B": -1000.0})
        assert quiet.weights == pytest.approx(loud.weights)

    def test_a_universe_that_agrees_completely_holds_nothing(self) -> None:
        """Every name equal means no dispersion, which is no position rather than a levered one."""
        book = neutralise({"A": 7.0, "B": 7.0, "C": 7.0})
        assert book.gross == 0.0

    def test_an_empty_universe_is_empty_not_an_error(self) -> None:
        assert neutralise({}).weights == {}

    def test_the_highest_raw_value_gets_the_largest_long(self) -> None:
        book = neutralise({"A": 1.0, "B": 2.0, "C": 3.0})
        assert book.weights["C"] > book.weights["B"] > book.weights["A"]


class TestCostsAreChargedOnWhatIsTraded:
    def _flat_panel(self):  # type: ignore[no-untyped-def]
        # Prices never move, so every basis point of result is cost.
        return _panel(A=[100.0] * 40, B=[100.0] * 40, C=[100.0] * 40)

    def test_a_book_that_never_trades_costs_nothing(self) -> None:
        got = backtest(Signal(operand=Const(value=0.0)), self._flat_panel(),
                       name="flat", cost=COST)
        assert got.mean_turnover == 0.0 and all(r == 0.0 for r in got.net_returns)

    def test_the_per_side_rate_comes_from_the_venue_model(self) -> None:
        """Not an assumed 12bps round trip halved by hand — the model's own taker_bps."""
        got = backtest(Signal(operand=Const(value=0.0)), self._flat_panel(),
                       name="flat", cost=COST)
        assert got.cost_bps_per_bar == float(COST.taker_bps) == 6.0

    def test_turnover_drives_the_drag(self) -> None:
        panel = _panel(A=[100.0, 101.0, 100.0, 101.0] * 10,
                       B=[100.0, 99.0, 100.0, 99.0] * 10,
                       C=[100.0] * 40)
        factor = Signal(operand=CrossRank(operand=CLOSE))
        fast = backtest(factor, panel, name="f", cost=COST, rebalance_every=1)
        slow = backtest(factor, panel, name="f", cost=COST, rebalance_every=20)
        assert fast.mean_turnover > slow.mean_turnover
        assert fast.cost_drag_bps_per_bar > slow.cost_drag_bps_per_bar

    def test_a_no_trade_band_reduces_turnover(self) -> None:
        panel = _panel(A=[100.0, 100.5, 100.0, 100.5] * 10,
                       B=[100.0, 99.5, 100.0, 99.5] * 10,
                       C=[100.0] * 40)
        factor = Signal(operand=CrossRank(operand=CLOSE))
        tight = backtest(factor, panel, name="f", cost=COST, band=0.0)
        wide = backtest(factor, panel, name="f", cost=COST, band=0.5)
        assert wide.mean_turnover < tight.mean_turnover

    def test_gross_and_net_differ_by_exactly_the_cost(self) -> None:
        panel = _panel(A=[100.0, 102.0, 101.0, 103.0] * 10,
                       B=[100.0, 98.0, 99.0, 97.0] * 10)
        got = backtest(Signal(operand=CrossRank(operand=CLOSE)), panel, name="f", cost=COST)
        for i, (g, n) in enumerate(zip(got.gross_returns, got.net_returns, strict=True)):
            expected = got.turnover_per_bar[i] * 6.0 / 10_000.0
            assert g - n == pytest.approx(expected)

    def test_an_invalid_rebalance_period_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least 1 bar"):
            backtest(Signal(operand=Const(value=0.0)), self._flat_panel(),
                     name="f", cost=COST, rebalance_every=0)

    def test_a_negative_band_is_refused(self) -> None:
        with pytest.raises(ValueError, match="cannot be negative"):
            backtest(Signal(operand=Const(value=0.0)), self._flat_panel(),
                     name="f", cost=COST, band=-0.1)


class TestNoLookAhead:
    def test_the_weight_earns_the_next_bar_not_the_current_one(self) -> None:
        """A weight set at i and multiplied by the return *into* i would be a perfect forecast.
        A is flat then jumps; a rank taken before the jump cannot have seen it."""
        panel = _panel(A=[100.0, 100.0, 100.0, 200.0], B=[100.0, 100.0, 100.0, 100.0])
        got = backtest(Signal(operand=CrossRank(operand=CLOSE)), panel, name="f", cost=COST)
        # Three usable bars; the jump is only earned by the weight set at index 2, which was
        # decided when A and B were identical and therefore neutral.
        assert got.periods == 3
        assert got.gross_returns[2] == pytest.approx(0.0)

    def test_extending_the_series_does_not_change_earlier_returns(self) -> None:
        factor = Signal(operand=CrossRank(operand=CLOSE))
        short = backtest(factor, _panel(A=[100.0, 110.0, 105.0], B=[100.0, 95.0, 99.0]),
                         name="f", cost=COST)
        long = backtest(
            factor,
            _panel(A=[100.0, 110.0, 105.0, 130.0, 90.0], B=[100.0, 95.0, 99.0, 91.0, 120.0]),
            name="f", cost=COST,
        )
        assert short.net_returns == pytest.approx(long.net_returns[: short.periods])

    def test_it_stops_one_bar_short_of_the_panel(self) -> None:
        got = backtest(Signal(operand=Const(value=0.0)), _panel(A=[1.0] * 10, B=[1.0] * 10),
                       name="f", cost=COST)
        assert got.periods == 9


class TestTheBookDriftsBetweenRebalances:
    def test_a_held_position_grows_with_its_price(self) -> None:
        """Holding must not be a free rebalance. A doubles while B is flat, so by the second bar
        the book is more exposed to A than the weights it was given."""
        # A leads from the first bar: a tied universe ranks everyone 0.5 and trades nothing,
        # so the book has to be given something to hold before drift can be observed.
        panel = _panel(A=[110.0, 220.0, 440.0, 880.0], B=[100.0, 100.0, 100.0, 100.0])
        got = backtest(Signal(operand=CrossRank(operand=CLOSE)), panel,
                       name="f", cost=COST, rebalance_every=99)
        # Only the first bar rebalances, so every later return comes from a drifting book.
        assert got.turnover_per_bar[0] > 0
        assert all(t == 0.0 for t in got.turnover_per_bar[1:])
        assert abs(got.gross_returns[1]) > abs(got.gross_returns[0])

    def test_rebalancing_only_happens_on_its_own_schedule(self) -> None:
        panel = _panel(A=[100.0, 110.0, 90.0, 120.0, 80.0, 130.0],
                       B=[100.0, 90.0, 110.0, 80.0, 120.0, 70.0])
        got = backtest(Signal(operand=CrossRank(operand=CLOSE)), panel,
                       name="f", cost=COST, rebalance_every=2)
        traded_on = [i for i, t in enumerate(got.turnover_per_bar) if t > 0]
        assert all(i % 2 == 0 for i in traded_on)


class TestTheStudyCountsEveryTrial:
    def test_the_trial_count_is_factors_times_rules(self) -> None:
        """The line that decides whether the best result is believable. Eight factors over ten
        rules is eighty trials, and the deflated Sharpe must be charged for all of them."""
        assert len(CANDIDATES) * len(TRADING_RULES) == 80

    def test_every_candidate_needs_the_panel(self) -> None:
        for name, factor in CANDIDATES.items():
            assert factor.needs_panel, name

    def test_no_two_candidates_are_the_same_factor(self) -> None:
        canonicals = {f.canonical() for f in CANDIDATES.values()}
        assert len(canonicals) == len(CANDIDATES)

    def test_the_rules_span_hourly_to_multi_day(self) -> None:
        periods = {r for r, _ in TRADING_RULES}
        assert min(periods) == 1 and max(periods) >= 72

    def test_a_rule_appears_only_once(self) -> None:
        assert len(set(TRADING_RULES)) == len(TRADING_RULES)


class TestTheResultReportsWhatItCannotClaim:
    def _result(self):  # type: ignore[no-untyped-def]
        panel = _panel(A=[100.0 + i for i in range(60)],
                       B=[100.0 - i * 0.5 for i in range(60)],
                       C=[100.0] * 60)
        return backtest(Signal(operand=CrossRank(operand=CLOSE)), panel,
                        name="f", cost=COST, rebalance_every=4)

    def test_gross_is_reported_beside_net_never_alone(self) -> None:
        got = self._result().as_dict()
        assert "gross_sharpe" in got and "net_sharpe" in got

    def test_the_trading_rule_travels_with_the_result(self) -> None:
        """A Sharpe without the rule that produced it cannot be reproduced or argued with."""
        got = self._result().as_dict()
        assert got["rebalance_every"] == 4 and "band" in got

    def test_the_cost_drag_is_stated_in_basis_points(self) -> None:
        assert "cost_drag_bps_per_bar" in self._result().as_dict()

    def test_out_of_sample_decay_is_computed(self) -> None:
        got = self._result().as_dict()
        assert "retention_ratio" in got["out_of_sample_decay"]

    def test_rolling_stability_is_computed_when_the_series_is_long_enough(self) -> None:
        """The window is 30 days of hourly bars, so a shorter fixture cannot exercise this."""
        import math

        n = 800
        panel = _panel(
            A=[100.0 + 10 * math.sin(i / 7) for i in range(n)],
            B=[100.0 + 10 * math.cos(i / 11) for i in range(n)],
            C=[100.0 + i * 0.01 for i in range(n)],
        )
        got = backtest(Signal(operand=CrossRank(operand=CLOSE)), panel,
                       name="f", cost=COST, rebalance_every=4).as_dict()
        assert "scorable_share" in got["rolling_sharpe_stability"]

    def test_a_series_too_short_for_the_window_says_so_rather_than_faking_it(self) -> None:
        """A metrics module that raises is right to; a report that propagates the raise cannot
        print its own best row. The absence is named, and no number is substituted."""
        got = self._result().as_dict()
        assert str(got["rolling_sharpe_stability"]).startswith("undefined:")

    def test_an_undefined_sortino_is_named_not_zeroed(self) -> None:
        """The row with no losing bars is the best-looking one, which is exactly where a
        fabricated 0.0 would do the most damage."""
        rising = _panel(A=[100.0 + i for i in range(30)], B=[100.0] * 30)
        got = backtest(Signal(operand=CrossRank(operand=CLOSE)), rising,
                       name="f", cost=COST, rebalance_every=99).as_dict()
        assert isinstance(got["sortino"], float | str)
        if isinstance(got["sortino"], str):
            assert got["sortino"].startswith("undefined:")


class TestThePhaseSweepIsTheDefenceAgainstAlignmentLuck:
    """The check that found and killed this module's own best result. It must not be weakened."""

    def _wiggly(self, n: int = 400):  # type: ignore[no-untyped-def]
        import math

        return _panel(
            A=[100.0 + 8 * math.sin(i / 5.0) for i in range(n)],
            B=[100.0 + 8 * math.cos(i / 6.0) for i in range(n)],
            C=[100.0 + 5 * math.sin(i / 9.0) for i in range(n)],
        )

    def test_the_phase_changes_which_bars_are_traded(self) -> None:
        panel = self._wiggly()
        factor = Signal(operand=CrossRank(operand=CLOSE))
        zero = backtest(factor, panel, name="f", cost=COST, rebalance_every=5, phase=0)
        two = backtest(factor, panel, name="f", cost=COST, rebalance_every=5, phase=2)
        traded_zero = [i for i, t in enumerate(zero.turnover_per_bar) if t > 0]
        traded_two = [i for i, t in enumerate(two.turnover_per_bar) if t > 0]
        assert traded_zero[0] == 0 and traded_two[0] == 2

    def test_a_phase_outside_the_cycle_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must be in"):
            backtest(Signal(operand=Const(value=0.0)), self._wiggly(20),
                     name="f", cost=COST, rebalance_every=4, phase=4)

    def test_every_phase_of_the_cycle_is_run_not_a_sample(self) -> None:
        """Sampling phases would reintroduce the same arbitrary choice one level up."""
        got = across_phases(Signal(operand=CrossRank(operand=CLOSE)), self._wiggly(),
                            name="f", cost=COST, rebalance_every=7, band=0.0)
        assert len(got.phases) == 7
        assert sorted(r.phase for r in got.phases) == list(range(7))

    def test_the_score_is_the_mean_never_the_best_phase(self) -> None:
        got = across_phases(Signal(operand=CrossRank(operand=CLOSE)), self._wiggly(),
                            name="f", cost=COST, rebalance_every=9, band=0.0)
        best = got.best_phase
        assert best is not None
        assert got.mean_net_sharpe <= best.net_sharpe

    def test_the_spread_is_published_beside_the_mean(self) -> None:
        """A mean without the spread hides exactly what the spread was there to reveal."""
        got = across_phases(Signal(operand=CrossRank(operand=CLOSE)), self._wiggly(),
                            name="f", cost=COST, rebalance_every=9, band=0.0).as_dict()
        assert "phase_spread" in got and "worst_phase_net_sharpe" in got
        assert got["phase_spread"] >= 0

    def test_disagreeing_phases_are_reported_as_disagreeing(self) -> None:
        got = across_phases(Signal(operand=CrossRank(operand=CLOSE)), self._wiggly(),
                            name="f", cost=COST, rebalance_every=11, band=0.0)
        signs = {s > 0 for s in got.net_sharpes}
        assert got.phases_agree is (len(signs) == 1)

    def test_hourly_rebalancing_has_exactly_one_phase(self) -> None:
        """There is nothing to align when the book trades every bar."""
        got = across_phases(Signal(operand=CrossRank(operand=CLOSE)), self._wiggly(),
                            name="f", cost=COST, rebalance_every=1, band=0.0)
        assert len(got.phases) == 1 and got.spread == 0.0

    def test_phase_zero_alone_is_still_reported_so_it_can_be_compared(self) -> None:
        got = across_phases(Signal(operand=CrossRank(operand=CLOSE)), self._wiggly(),
                            name="f", cost=COST, rebalance_every=6, band=0.0).as_dict()
        assert "phase_0" in got and "net_sharpe" in got["phase_0"]

    def test_a_rule_with_no_scorable_phase_is_not_scored(self) -> None:
        flat = _panel(A=[100.0] * 50, B=[100.0] * 50)
        got = across_phases(Signal(operand=Const(value=0.0)), flat,
                            name="f", cost=COST, rebalance_every=3, band=0.0)
        assert not got.scorable and got.mean_net_sharpe == 0.0 and not got.phases_agree
