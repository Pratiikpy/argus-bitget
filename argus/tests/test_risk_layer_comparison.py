"""`eval/risk_layer_comparison.py` — the statistically-valid, OOS-split, adversarial, cost-
included run comparison the OWNED bar needs on top of the single hand-built scenario in
`test_freqtrade_baseline.py`. Deterministic pieces only here (no live venue fetch) — the live
`compare_real_symbols` happy path was run by hand against real Bitget data and its output is
`data/risk_layer_comparison.json`, matching how `research/track1_study.py`'s own network-dependent
orchestration is exercised (not unit-tested directly, only its pure helpers are)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from argus.backtest.engine import SyntheticTrade
from argus.eval.risk_layer_comparison import (
    Checkpoint,
    Contingency,
    RiskLayerComparisonError,
    _net_return_pct,
    _oos_boundary,
    _pearson,
    _precision_recall,
    _tally,
    adversarial_scenarios,
    argus_measures_ground_truth_directly,
    compare_combined_book,
    compare_real_symbols,
    measurement_architecture_analysis,
    protection_ablation,
    walk_combined_book,
    walk_trades,
)

AT = datetime(2026, 3, 1, tzinfo=UTC)


def _trade(
    exit_offset_days: int, return_pct: float, *, symbol: str = "NVDAUSDT",
) -> SyntheticTrade:
    return SyntheticTrade(
        symbol=symbol, entry_bar=0, exit_bar=1,
        entry_ts=AT + timedelta(days=exit_offset_days - 1),
        exit_ts=AT + timedelta(days=exit_offset_days),
        direction="long", weight=1.0,
        entry_price=Decimal("100"), exit_price=Decimal(str(100 * (1 + return_pct))),
    )


class TestNetReturnAppliesCostOnce:
    def test_cost_is_subtracted_from_the_raw_return_exactly_once(self) -> None:
        trade = _trade(1, 0.05)
        net = _net_return_pct(trade, cost_bps=Decimal("12"))
        assert net == pytest.approx(0.05 - 0.0012)

    def test_a_short_trades_raw_return_is_already_sign_adjusted_before_cost(self) -> None:
        trade = SyntheticTrade(
            symbol="NVDAUSDT", entry_bar=0, exit_bar=1, entry_ts=AT, exit_ts=AT,
            direction="short", weight=-1.0,
            entry_price=Decimal("100"), exit_price=Decimal("90"),
        )
        net = _net_return_pct(trade, cost_bps=Decimal("12"))
        assert net == pytest.approx(0.10 - 0.0012)


class TestOosBoundaryMatchesTheEnginesOwnSplit:
    def test_matches_the_documented_formula(self) -> None:
        # engine.py:283 — split = int(len(net_returns) * (1 - oos_fraction)); net_returns has
        # length len(bars) - 1.
        assert _oos_boundary(101, oos_fraction=0.35) == int(100 * 0.65)
        assert _oos_boundary(2159) == int(2158 * 0.65)


class TestContingencyTally:
    def test_a_perfectly_agreeing_pair_has_agreement_rate_one(self) -> None:
        tally = Contingency(both_locked=3, argus_only=0, freqtrade_only=0, neither=7)
        assert tally.total == 10
        assert tally.agreement_rate == 1.0

    def test_total_disagreement_has_agreement_rate_zero(self) -> None:
        tally = Contingency(both_locked=0, argus_only=5, freqtrade_only=5, neither=0)
        assert tally.agreement_rate == 0.0

    def test_an_empty_tally_reports_zero_not_a_division_error(self) -> None:
        assert Contingency().agreement_rate == 0.0


class TestWalkTradesReplaysChronologicallyAndAppliesBothSystems:
    def test_a_single_large_loss_locks_both_systems(self) -> None:
        trades = [_trade(1, -0.30)]
        boundary = AT + timedelta(days=10)
        checkpoints = walk_trades(trades, symbol="NVDAUSDT", oos_boundary_ts=boundary)
        assert len(checkpoints) == 1
        c = checkpoints[0]
        assert c.argus_locked
        assert c.freqtrade_risk_locked
        assert not c.is_oos

    def test_true_drawdown_pct_matches_the_actual_equity_move(self) -> None:
        """The field the combined-book comparison's decisive finding depends on: it must read
        the SAME peak-to-trough figure ARGUS's own breaker is judged against, not an
        independently-derived approximation that could silently drift from it."""
        trades = [_trade(1, -0.10)]
        boundary = AT + timedelta(days=10)
        checkpoints = walk_trades(trades, symbol="NVDAUSDT", oos_boundary_ts=boundary)
        # 10% loss minus ~0.12% cost, applied against full starting capital, no prior peak below.
        assert checkpoints[0].true_drawdown_pct == pytest.approx(0.1012, abs=1e-3)

    def test_true_drawdown_pct_never_exceeds_a_recovered_peak(self) -> None:
        """A loss followed by a recovery past the original peak resets the drawdown to zero —
        the same all-time-peak convention `risk.circuit.BookState.total_drawdown` itself uses."""
        checkpoints = walk_trades(
            [_trade(1, -0.10), _trade(2, 0.20)], symbol="NVDAUSDT",
            oos_boundary_ts=AT + timedelta(days=10),
        )
        assert checkpoints[0].true_drawdown_pct > 0
        assert checkpoints[1].true_drawdown_pct == 0.0

    def test_trades_out_of_input_order_are_still_replayed_chronologically(self) -> None:
        """Feeding trades reverse-chronologically must not change the accumulated equity path —
        `walk_trades` sorts by `exit_ts` itself rather than trusting caller order."""
        forward = walk_trades(
            [_trade(1, -0.05), _trade(2, -0.05), _trade(3, 0.10)],
            symbol="NVDAUSDT", oos_boundary_ts=AT + timedelta(days=100),
        )
        reversed_input = walk_trades(
            [_trade(3, 0.10), _trade(1, -0.05), _trade(2, -0.05)],
            symbol="NVDAUSDT", oos_boundary_ts=AT + timedelta(days=100),
        )
        assert [c.exit_ts for c in forward] == [c.exit_ts for c in reversed_input]
        assert [c.argus_activation for c in forward] == [
            c.argus_activation for c in reversed_input
        ]

    def test_is_oos_split_on_the_supplied_boundary(self) -> None:
        trades = [_trade(1, 0.01), _trade(5, 0.01), _trade(10, 0.01)]
        checkpoints = walk_trades(
            trades, symbol="NVDAUSDT", oos_boundary_ts=AT + timedelta(days=6),
        )
        assert [c.is_oos for c in checkpoints] == [False, False, True]

    def test_a_flat_untouched_book_reads_active_and_unlocked(self) -> None:
        trades = [_trade(1, 0.001)]
        checkpoints = walk_trades(
            trades, symbol="NVDAUSDT", oos_boundary_ts=AT + timedelta(days=10),
        )
        c = checkpoints[0]
        assert c.argus_activation == "active"
        assert not c.argus_locked
        assert not c.freqtrade_drawdown_locked
        assert not c.freqtrade_stoploss_locked

    def test_tallying_a_walked_sequence_is_internally_consistent(self) -> None:
        trades = [_trade(i, -0.03 if i % 2 else 0.02) for i in range(1, 8)]
        checkpoints = walk_trades(
            trades, symbol="NVDAUSDT", oos_boundary_ts=AT + timedelta(days=100),
        )
        tally = _tally(checkpoints)
        assert tally.total == len(checkpoints)
        assert tally.total == (
            tally.both_locked + tally.argus_only + tally.freqtrade_only + tally.neither
        )


def _checkpoint(
    *, drawdown: bool = False, stoploss: bool = False, low_profit: bool = False,
) -> Checkpoint:
    return Checkpoint(
        symbol="NVDAUSDT", exit_ts=AT, net_return_pct=0.0, is_oos=False,
        true_drawdown_pct=0.0, argus_activation="active", argus_trips=(),
        freqtrade_drawdown_locked=drawdown, freqtrade_stoploss_locked=stoploss,
        freqtrade_low_profit_locked=low_profit, freqtrade_cooldown_locked=False,
    )


class TestProtectionAblation:
    """The formal per-protection ablation: which of the three loss-severity protections fires
    ALONE (unique contribution) versus overlapping with the others."""

    def test_a_protection_that_only_ever_fires_alongside_another_has_zero_unique_share(
        self,
    ) -> None:
        checkpoints = [
            _checkpoint(drawdown=True, stoploss=True),
            _checkpoint(drawdown=True, stoploss=True),
        ]
        out = protection_ablation(checkpoints)
        assert out["fire_rate"]["drawdown"] == 1.0
        assert out["fire_rate"]["stoploss"] == 1.0
        assert out["unique_share"]["drawdown"] == 0.0
        assert out["unique_share"]["stoploss"] == 0.0

    def test_a_protection_that_always_fires_alone_has_unique_share_equal_to_fire_rate(
        self,
    ) -> None:
        checkpoints = [_checkpoint(low_profit=True), _checkpoint(low_profit=True), _checkpoint()]
        out = protection_ablation(checkpoints)
        assert out["fire_rate"]["low_profit"] == pytest.approx(2 / 3, abs=1e-4)
        assert out["unique_share"]["low_profit"] == pytest.approx(2 / 3, abs=1e-4)
        assert out["fire_rate"]["drawdown"] == 0.0

    def test_an_empty_checkpoint_list_reports_zero_not_a_division_error(self) -> None:
        out = protection_ablation([])
        assert out["n"] == 0
        assert out["fire_rate"] == {"drawdown": 0.0, "stoploss": 0.0, "low_profit": 0.0}
        assert out["unique_share"] == {"drawdown": 0.0, "stoploss": 0.0, "low_profit": 0.0}


class TestAdversarialScenarios:
    """Hand-built, deterministic fixtures — pinned exactly, not merely smoke-tested, since these
    are the scenarios this module's own OWNED-bar claim rests on."""

    def test_all_three_named_scenarios_are_present(self) -> None:
        out = adversarial_scenarios()
        assert set(out) == {
            "three_losses_no_streak_halt", "one_big_loss_then_many_small_wins",
            "rapid_fire_within_cooldown",
        }

    def test_three_losses_locks_argus_via_the_drawdown_ladder_not_the_streak(self) -> None:
        """Verified finding, not the original hypothesis (documented in the function's own
        docstring): the ladder — not the 4-loss streak halt — is what fires here."""
        out = adversarial_scenarios()
        checkpoints = out["three_losses_no_streak_halt"]["checkpoints"]
        assert checkpoints[0]["argus_activation"] != "active"
        assert "drawdown_ladder" in checkpoints[0]["argus_trips"]
        assert checkpoints[0]["freqtrade_low_profit_locked"]

    def test_rapid_fire_shows_the_real_asymmetry_argus_starts_unlocked(self) -> None:
        """The confirmed gap: freqtrade's per-symbol, zero-floor protections lock on the first
        trade while ARGUS's whole-book ladder does not — exactly what
        `ConstitutionPolicy.min_symbol_realized_pnl` (gate 12) exists to close one layer up."""
        out = adversarial_scenarios()
        checkpoints = out["rapid_fire_within_cooldown"]["checkpoints"]
        assert checkpoints[0]["argus_activation"] == "active"
        assert checkpoints[0]["freqtrade_low_profit_locked"]
        assert checkpoints[0]["freqtrade_cooldown_locked"]

    def test_every_scenario_reports_at_least_one_checkpoint(self) -> None:
        out = adversarial_scenarios()
        for name, row in out.items():
            assert row["checkpoints"], f"{name} produced no checkpoints"


class TestCompareRealSymbolsFailsHonestlyWithoutAStudyFile:
    def test_missing_track1_study_raises_rather_than_guessing_a_variant(
        self, tmp_path: Path,
    ) -> None:
        with pytest.raises(RiskLayerComparisonError):
            compare_real_symbols(
                symbols=("NVDAUSDT",), track1_study_path=tmp_path / "missing.json",
            )


class TestWalkCombinedBook:
    """The whole-book counterpart: capital split evenly across symbols, drawdown/stoploss-guard
    seeing every symbol's trades together, low-profit-pairs/cooldown staying per-symbol."""

    def test_no_symbols_raises(self) -> None:
        with pytest.raises(RiskLayerComparisonError):
            walk_combined_book([], symbols=(), oos_boundary_ts=AT + timedelta(days=100))

    def test_capital_is_split_evenly_a_loss_that_halts_alone_only_reduces_combined(
        self,
    ) -> None:
        """An 11% loss breaches ARGUS's 10% HALT cap on a full-capital single-symbol book but,
        split across two symbols, only moves the combined book ~5.5% — above the 2% REDUCE_ONLY
        threshold, but nowhere near the 10% halt one. Neither figure is "active": the point is
        the split changes which rung binds, not that it avoids the ladder outright."""
        trade = _trade(1, -0.11, symbol="AAA")
        boundary = AT + timedelta(days=100)

        solo = walk_trades([trade], symbol="AAA", oos_boundary_ts=boundary)
        assert solo[0].argus_activation == "halted"

        combined = walk_combined_book([trade], symbols=("AAA", "BBB"), oos_boundary_ts=boundary)
        assert combined[0].argus_activation == "reduce_only"

    def test_drawdown_and_stoploss_guard_see_every_symbols_trades_together(self) -> None:
        """A big loss on one symbol followed by a small WIN on a different symbol still reads
        the global window's drawdown — freqtrade's real default scope for these two
        protections — even though the second trade itself made money."""
        loss = _trade(1, -0.15, symbol="AAA")
        win = _trade(2, 0.05, symbol="BBB")
        checkpoints = walk_combined_book(
            [loss, win], symbols=("AAA", "BBB"), oos_boundary_ts=AT + timedelta(days=100),
        )
        assert checkpoints[0].symbol == "AAA"
        assert checkpoints[1].symbol == "BBB"
        assert checkpoints[1].freqtrade_drawdown_locked

    def test_low_profit_pairs_stays_scoped_to_its_own_symbol(self) -> None:
        """The other symbol's bad trade must not lock a symbol that has only ever won —
        `low_profit_pairs` sums only trades on the checkpoint's own symbol."""
        loss = _trade(1, -0.15, symbol="AAA")
        win = _trade(2, 0.05, symbol="BBB")
        checkpoints = walk_combined_book(
            [loss, win], symbols=("AAA", "BBB"), oos_boundary_ts=AT + timedelta(days=100),
        )
        assert not checkpoints[1].freqtrade_low_profit_locked

    def test_cooldown_fires_on_its_own_symbols_own_trade_not_the_others(self) -> None:
        """`cooldown_period` has no profit/loss test at all — it locks a symbol because a trade
        on THAT symbol just closed, regardless of direction. BBB's own win still locks BBB;
        AAA's loss plays no part in it."""
        loss = _trade(1, -0.15, symbol="AAA")
        win = _trade(2, 0.05, symbol="BBB")
        checkpoints = walk_combined_book(
            [loss, win], symbols=("AAA", "BBB"), oos_boundary_ts=AT + timedelta(days=100),
        )
        assert checkpoints[1].freqtrade_cooldown_locked  # BBB's own trade just closed
        assert checkpoints[0].freqtrade_cooldown_locked  # AAA's own trade just closed


class TestCompareCombinedBookFailsHonestly:
    def test_missing_track1_study_raises(self, tmp_path: Path) -> None:
        with pytest.raises(RiskLayerComparisonError):
            compare_combined_book(
                symbols=("NVDAUSDT",), track1_study_path=tmp_path / "missing.json",
            )


def _ma_checkpoint(
    *, true_drawdown_pct: float, freqtrade_drawdown_measured: float,
    argus_locked_via: str = "none", consecutive_losses: int = 0,
) -> Checkpoint:
    """A checkpoint built for the measurement-architecture analysis: `argus_locked_via` picks
    which real trigger (or none) `argus_activation` reflects, matching exactly what `assess()`
    would have produced for the given `true_drawdown_pct`/`consecutive_losses` combination —
    "drawdown" and "streak" are the two real triggers this comparison's own `BookState`
    construction can reach (see `argus_measures_ground_truth_directly`'s own docstring)."""
    activation = "active" if argus_locked_via == "none" else "reduce_only"
    return Checkpoint(
        symbol="NVDAUSDT", exit_ts=AT, net_return_pct=0.0, is_oos=False,
        true_drawdown_pct=true_drawdown_pct, argus_activation=activation, argus_trips=(),
        freqtrade_drawdown_locked=False, freqtrade_stoploss_locked=False,
        freqtrade_low_profit_locked=False, freqtrade_cooldown_locked=False,
        freqtrade_drawdown_measured=freqtrade_drawdown_measured,
        consecutive_losses=consecutive_losses,
    )


class TestPearson:
    def test_perfect_positive_correlation(self) -> None:
        assert _pearson([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]) == pytest.approx(1.0)

    def test_no_correlation_returns_near_zero(self) -> None:
        # A symmetric valley against a monotonic series has exactly zero linear relation.
        assert _pearson([1.0, 2.0, 3.0, 4.0], [1.0, 0.0, 0.0, 1.0]) == pytest.approx(0.0, abs=1e-9)

    def test_constant_series_is_zero_not_a_division_error(self) -> None:
        assert _pearson([5.0, 5.0, 5.0], [1.0, 2.0, 3.0]) == 0.0

    def test_fewer_than_two_points_is_zero(self) -> None:
        assert _pearson([1.0], [1.0]) == 0.0
        assert _pearson([], []) == 0.0


class TestPrecisionRecall:
    def test_perfect_predictor_scores_one_and_one(self) -> None:
        precision, recall = _precision_recall([True, False, True], [True, False, True])
        assert precision == 1.0
        assert recall == 1.0

    def test_a_predictor_that_never_fires_has_perfect_precision_and_zero_recall(self) -> None:
        precision, recall = _precision_recall([False, False], [True, False])
        assert precision == 1.0  # vacuous: no positive prediction was ever wrong
        assert recall == 0.0

    def test_a_predictor_that_always_fires_has_full_recall_and_low_precision(self) -> None:
        precision, recall = _precision_recall([True, True, True], [True, False, False])
        assert recall == 1.0
        assert precision == pytest.approx(1 / 3)

    def test_no_actual_positives_gives_vacuous_full_recall(self) -> None:
        _, recall = _precision_recall([True, False], [False, False])
        assert recall == 1.0


class TestArgusMeasuresGroundTruthDirectly:
    def test_true_when_every_checkpoint_reconstructs_from_drawdown_alone(self) -> None:
        checkpoints = [
            _ma_checkpoint(true_drawdown_pct=0.0, freqtrade_drawdown_measured=0.0),
            _ma_checkpoint(
                true_drawdown_pct=0.05, freqtrade_drawdown_measured=0.01,
                argus_locked_via="drawdown",
            ),
        ]
        assert argus_measures_ground_truth_directly(checkpoints) is True

    def test_true_when_the_streak_trigger_is_correctly_reflected(self) -> None:
        """The exact gap the real data caught this comparison's original single-signal
        reconstruction missing: a low-drawdown checkpoint locked purely by consecutive_losses."""
        checkpoints = [
            _ma_checkpoint(
                true_drawdown_pct=0.001, freqtrade_drawdown_measured=0.0,
                argus_locked_via="streak", consecutive_losses=4,
            ),
        ]
        assert argus_measures_ground_truth_directly(checkpoints) is True

    def test_false_when_argus_locked_disagrees_with_both_ground_truth_signals(self) -> None:
        """A checkpoint claiming ARGUS locked with neither drawdown nor streak past their real
        thresholds — the premise this whole analysis depends on failing, and it must be caught."""
        bad = Checkpoint(
            symbol="NVDAUSDT", exit_ts=AT, net_return_pct=0.0, is_oos=False,
            true_drawdown_pct=0.0, argus_activation="reduce_only", argus_trips=(),
            freqtrade_drawdown_locked=False, freqtrade_stoploss_locked=False,
            freqtrade_low_profit_locked=False, freqtrade_cooldown_locked=False,
            freqtrade_drawdown_measured=0.0, consecutive_losses=0,
        )
        assert argus_measures_ground_truth_directly([bad]) is False


class TestMeasurementArchitectureAnalysis:
    def test_refuses_to_run_when_its_own_premise_fails(self) -> None:
        bad = Checkpoint(
            symbol="NVDAUSDT", exit_ts=AT, net_return_pct=0.0, is_oos=False,
            true_drawdown_pct=0.0, argus_activation="reduce_only", argus_trips=(),
            freqtrade_drawdown_locked=False, freqtrade_stoploss_locked=False,
            freqtrade_low_profit_locked=False, freqtrade_cooldown_locked=False,
            freqtrade_drawdown_measured=0.0, consecutive_losses=0,
        )
        with pytest.raises(RiskLayerComparisonError, match="ground-truth-measurement"):
            measurement_architecture_analysis([bad])

    def test_a_perfectly_correlated_proxy_yields_strong_verdict_and_argus_still_wins_or_ties(
        self,
    ) -> None:
        """When freqtrade's proxy tracks the truth exactly, correlation should read STRONG and
        the verdict should say so plainly — the analysis does not manufacture a weak result."""
        checkpoints = [
            _ma_checkpoint(true_drawdown_pct=v, freqtrade_drawdown_measured=v)
            for v in (0.0, 0.005, 0.01, 0.015, 0.025, 0.03, 0.05)
        ]
        # Reflect ARGUS's real reduce-only trigger for each, matching drawdown >= 0.02.
        from argus.risk.circuit import REDUCE_ONLY_DRAWDOWN

        checkpoints = [
            Checkpoint(
                symbol=c.symbol, exit_ts=c.exit_ts, net_return_pct=0.0, is_oos=False,
                true_drawdown_pct=c.true_drawdown_pct,
                argus_activation=(
                    "reduce_only" if c.true_drawdown_pct >= float(REDUCE_ONLY_DRAWDOWN)
                    else "active"
                ),
                argus_trips=(),
                freqtrade_drawdown_locked=False, freqtrade_stoploss_locked=False,
                freqtrade_low_profit_locked=False, freqtrade_cooldown_locked=False,
                freqtrade_drawdown_measured=c.freqtrade_drawdown_measured,
                consecutive_losses=0,
            )
            for c in checkpoints
        ]
        result = measurement_architecture_analysis(checkpoints)
        assert result["freqtrade_proxy_vs_truth_correlation"] > 0.99
        assert "STRONG" in result["verdict"]

    def test_a_weakly_correlated_proxy_is_dominated_by_argus_real_point(self) -> None:
        """The shape of the actual finding on real data: freqtrade's proxy barely tracks truth,
        and ARGUS's real (drawdown-only here) decision dominates every swept threshold."""
        from argus.risk.circuit import REDUCE_ONLY_DRAWDOWN

        true_vals = [0.001, 0.003, 0.005, 0.008, 0.012, 0.018, 0.022, 0.028, 0.035, 0.04]
        # A proxy that is nearly random relative to the true ranking.
        proxy_vals = [0.03, 0.01, 0.04, 0.02, 0.005, 0.035, 0.015, 0.025, 0.045, 0.0]
        checkpoints = [
            Checkpoint(
                symbol="NVDAUSDT", exit_ts=AT, net_return_pct=0.0, is_oos=False,
                true_drawdown_pct=t,
                argus_activation=(
                    "reduce_only" if t >= float(REDUCE_ONLY_DRAWDOWN) else "active"
                ),
                argus_trips=(),
                freqtrade_drawdown_locked=False, freqtrade_stoploss_locked=False,
                freqtrade_low_profit_locked=False, freqtrade_cooldown_locked=False,
                freqtrade_drawdown_measured=p, consecutive_losses=0,
            )
            for t, p in zip(true_vals, proxy_vals, strict=True)
        ]
        result = measurement_architecture_analysis(checkpoints)
        assert abs(result["freqtrade_proxy_vs_truth_correlation"]) < 0.5
        assert result["argus_real_point"]["recall"] == 1.0

    def test_on_the_real_saved_risk_layer_comparison_data_shape(self) -> None:
        """Same structural shape the real 320-checkpoint run produced (weak correlation, ARGUS
        dominating every swept threshold) — pinned as a regression fixture so the analysis's own
        logic is exercised deterministically without a live fetch, matching this file's own
        stated policy of no live venue calls in the test suite."""
        import random

        from argus.risk.circuit import REDUCE_ONLY_DRAWDOWN

        rng = random.Random(7)
        checkpoints = []
        for _ in range(100):
            t = rng.uniform(0.0, 0.046)
            p = rng.uniform(0.0, 0.05)  # weakly related: independent draw, not derived from t
            checkpoints.append(Checkpoint(
                symbol="NVDAUSDT", exit_ts=AT, net_return_pct=0.0, is_oos=False,
                true_drawdown_pct=t,
                argus_activation=(
                    "reduce_only" if t >= float(REDUCE_ONLY_DRAWDOWN) else "active"
                ),
                argus_trips=(),
                freqtrade_drawdown_locked=False, freqtrade_stoploss_locked=False,
                freqtrade_low_profit_locked=False, freqtrade_cooldown_locked=False,
                freqtrade_drawdown_measured=p, consecutive_losses=0,
            ))
        result = measurement_architecture_analysis(checkpoints)
        assert result["argus_real_point"]["recall"] == 1.0
        assert result["argus_dominates_every_swept_threshold"] is True
