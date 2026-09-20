"""Incremental-value tests — the comparison must be able to say we lost.

A benchmark whose baselines are all weak measures nothing, and a comparison whose verdict can only
come out one way is not a comparison. So every test that touches the verdict is paired: a universe
where the desk should lose, and one where it should win. The scoring rules are pinned against
arithmetic rather than against what the implementation happens to produce — an abstention earns
exactly zero, a trade earns the move less the round trip, and a win rate over instants a policy
declined is undefined rather than nil.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta

import pytest

from argus.eval.collect import (
    MIN_TRAILING,
    OPENING_VERDICTS,
    TRAILING_BARS,
    build_instants,
    direction_of,
    trailing_returns,
)
from argus.eval.incremental import (
    BASELINES,
    DIRECTION_LONG,
    DIRECTION_NONE,
    DIRECTION_SHORT,
    MIN_INSTANTS,
    IncrementalError,
    Instant,
    always_flat,
    always_long,
    compare,
    evaluate,
    momentum,
    reversion,
    score,
    volatility_gated,
)

START = datetime(2026, 6, 1, tzinfo=UTC)
HURDLE = 18.8


def _instant(
    *, move: float = 100.0, trail: tuple[float, ...] | None = None, symbol: str = "NVDAUSDT",
    hour: int = 0,
) -> Instant:
    return Instant(
        symbol=symbol,
        at=START + timedelta(hours=hour),
        trailing=trail if trail is not None else tuple([0.001] * 72),
        realised_bps=move,
        hurdle_bps=HURDLE,
    )


def _universe(
    n: int, *, drift: float, seed: int, spread: float = 150.0, symbols: int = 1
) -> list[Instant]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        out.append(
            Instant(
                symbol=f"S{i % symbols}",
                at=START + timedelta(hours=(i // symbols) * 6),
                trailing=tuple(rng.gauss(drift / 10_000, 0.01) for _ in range(72)),
                realised_bps=rng.gauss(drift, spread),
                hurdle_bps=HURDLE,
            )
        )
    return out


class TestWhatADecisionEarns:
    def test_an_abstention_earns_exactly_zero_however_large_the_move(self) -> None:
        """Crediting an abstention with the move it declined is the commonest way a selective
        strategy's backtest is inflated."""
        assert _instant(move=900.0).net_bps(DIRECTION_NONE) == 0.0

    def test_a_long_earns_the_move_less_the_round_trip(self) -> None:
        assert _instant(move=100.0).net_bps(DIRECTION_LONG) == pytest.approx(100.0 - HURDLE)

    def test_a_short_earns_the_inverse_less_the_round_trip(self) -> None:
        assert _instant(move=-100.0).net_bps(DIRECTION_SHORT) == pytest.approx(100.0 - HURDLE)

    def test_a_correct_direction_can_still_lose_to_the_hurdle(self) -> None:
        """The whole reason the hurdle is charged.

        Being right by 5bps is a loss, not a small win.
        """
        assert _instant(move=5.0).net_bps(DIRECTION_LONG) < 0

    def test_the_hurdle_is_paid_on_a_losing_trade_too(self) -> None:
        assert _instant(move=-100.0).net_bps(DIRECTION_LONG) == pytest.approx(-100.0 - HURDLE)


class TestTheBaselinesDoWhatTheyAreNamed:
    def test_always_flat_never_trades(self) -> None:
        assert always_flat(_instant()) == DIRECTION_NONE

    def test_always_long_always_buys(self) -> None:
        assert always_long(_instant(move=-500.0)) == DIRECTION_LONG

    def test_momentum_follows_the_trailing_sign(self) -> None:
        assert momentum(_instant(trail=tuple([0.01] * 72))) == DIRECTION_LONG
        assert momentum(_instant(trail=tuple([-0.01] * 72))) == DIRECTION_SHORT

    def test_momentum_is_flat_on_an_exactly_flat_trail(self) -> None:
        """Not long. A tie is not a signal, and defaulting it to long would quietly make momentum
        part always_long."""
        assert momentum(_instant(trail=tuple([0.0] * 72))) == DIRECTION_NONE

    def test_momentum_reads_only_its_lookback(self) -> None:
        """A 24-bar rule must not be moved by bar 60. If it is, the window is not being applied."""
        trail = tuple([-0.05] * 48 + [0.01] * 24)
        assert momentum(_instant(trail=trail), lookback=24) == DIRECTION_LONG

    def test_reversion_is_exactly_the_opposite_of_momentum(self) -> None:
        for trail in (tuple([0.01] * 72), tuple([-0.01] * 72), tuple([0.0] * 72)):
            instant = _instant(trail=trail)
            assert reversion(instant) == -momentum(instant)

    def test_momentum_on_an_empty_trail_is_flat(self) -> None:
        assert momentum(_instant(trail=())) == DIRECTION_NONE

    def test_the_volatility_gate_declines_a_quiet_tape(self) -> None:
        """Trailing volatility far below the hurdle means no move can pay for the round trip, and
        the rule must decline rather than trade a direction it cannot monetise."""
        quiet = tuple([0.00001 * (1 if i % 2 else -1) for i in range(72)])
        assert volatility_gated(_instant(trail=quiet)) == DIRECTION_NONE

    def test_the_volatility_gate_trades_a_loud_tape_in_the_momentum_direction(self) -> None:
        loud = tuple([0.02 if i % 3 else 0.03 for i in range(72)])
        instant = _instant(trail=loud)
        assert volatility_gated(instant) == momentum(instant)
        assert volatility_gated(instant) != DIRECTION_NONE

    def test_the_volatility_gate_needs_two_bars_to_form_a_deviation(self) -> None:
        assert volatility_gated(_instant(trail=(0.02,))) == DIRECTION_NONE

    def test_every_baseline_returns_a_legal_direction(self) -> None:
        rng = random.Random(1)
        for _ in range(50):
            instant = _instant(trail=tuple(rng.gauss(0, 0.01) for _ in range(72)))
            for name, policy in BASELINES.items():
                assert policy(instant) in (-1, 0, 1), name

    def test_the_baseline_set_includes_one_that_should_be_embarrassing(self) -> None:
        """A benchmark whose baselines are all weak measures nothing."""
        assert "volatility_gated" in BASELINES
        assert "always_long" in BASELINES


class TestScoring:
    def test_a_flat_policy_scores_zero_over_any_record(self) -> None:
        scored = score("flat", always_flat, _universe(40, drift=400.0, seed=2))
        assert scored.total_bps == 0.0
        assert scored.trades == 0

    def test_a_win_rate_over_no_trades_is_undefined_not_zero(self) -> None:
        """Zero would read as "it always loses" rather than "it never played"."""
        assert score("flat", always_flat, _universe(30, drift=0.0, seed=3)).win_rate is None

    def test_the_win_rate_is_computed_over_traded_instants_only(self) -> None:
        # A constant trail has zero deviation however large its level, so the loud tape has to
        # alternate. That is not a quirk of the test: a market that moves the same amount every
        # bar is a market with no uncertainty, and the gate is right to decline it.
        loud = tuple(0.03 if i % 2 else -0.01 for i in range(72))
        instants = [
            _instant(move=500.0, trail=loud, hour=0),
            _instant(move=500.0, trail=tuple([0.0000001] * 72), hour=6),
        ]
        scored = score("gated", volatility_gated, instants)
        assert scored.trades == 1
        assert scored.win_rate == 1.0

    def test_always_long_wins_a_rising_universe(self) -> None:
        scored = score("long", always_long, _universe(40, drift=300.0, seed=4, spread=50.0))
        assert scored.total_bps > 0
        assert scored.win_rate is not None and scored.win_rate > 0.9

    def test_always_long_loses_a_falling_universe(self) -> None:
        assert score("long", always_long, _universe(40, drift=-300.0, seed=5)).total_bps < 0


class TestThePairedComparison:
    def test_identical_policies_have_no_differing_pairs(self) -> None:
        instants = _universe(40, drift=100.0, seed=6)
        desk = score("desk", always_flat, instants)
        got = compare(desk, score("always_flat", always_flat, instants))
        assert got.differing == 0
        assert "IDENTICAL" in got.verdict

    def test_an_identical_comparison_is_not_reported_as_a_win(self) -> None:
        """A tie is not value added. If this ever reported BEATS, the desk would appear to beat
        doing nothing by doing nothing."""
        instants = _universe(40, drift=100.0, seed=7)
        desk = score("desk", always_flat, instants)
        assert "BEATS" not in compare(desk, score("always_flat", always_flat, instants)).verdict

    def test_the_flat_desk_loses_to_always_long_in_a_rising_universe(self) -> None:
        instants = _universe(40, drift=250.0, seed=8, spread=60.0)
        got = compare(score("desk", always_flat, instants), score("long", always_long, instants))
        assert got.losses > got.wins
        assert got.p_value < 0.05
        assert "LOSES TO" in got.verdict

    def test_the_flat_desk_beats_always_long_in_a_falling_universe(self) -> None:
        instants = _universe(40, drift=-250.0, seed=9, spread=60.0)
        got = compare(score("desk", always_flat, instants), score("long", always_long, instants))
        assert got.wins > got.losses
        assert "BEATS" in got.verdict

    def test_a_marginal_difference_is_reported_as_inconclusive(self) -> None:
        instants = _universe(40, drift=0.0, seed=10, spread=200.0)
        got = compare(score("desk", always_flat, instants), score("long", always_long, instants))
        assert got.p_value > 0.05
        assert "INCONCLUSIVE" in got.verdict

    def test_comparing_different_samples_raises(self) -> None:
        a = score("a", always_flat, _universe(40, drift=0.0, seed=11))
        b = score("b", always_long, _universe(39, drift=0.0, seed=12))
        with pytest.raises(IncrementalError, match="not a paired test"):
            compare(a, b)

    def test_wins_losses_and_ties_account_for_every_instant(self) -> None:
        instants = _universe(40, drift=100.0, seed=13)
        got = compare(score("desk", always_flat, instants), score("m", momentum, instants))
        assert got.wins + got.losses + got.ties == 40


class TestTheReportRefusesToOverclaim:
    def test_a_thin_record_raises_rather_than_reporting_a_count(self) -> None:
        with pytest.raises(IncrementalError, match="below the"):
            evaluate(_universe(MIN_INSTANTS - 1, drift=0.0, seed=14), [0] * (MIN_INSTANTS - 1))

    def test_a_mismatched_decision_list_raises(self) -> None:
        instants = _universe(40, drift=0.0, seed=15)
        with pytest.raises(IncrementalError, match="every instant"):
            evaluate(instants, [0] * 39)

    def test_an_illegal_direction_raises(self) -> None:
        instants = _universe(40, drift=0.0, seed=16)
        with pytest.raises(IncrementalError, match="-1, 0 or"):
            evaluate(instants, [2] * 40)

    def test_it_names_the_best_rule_as_the_best_of_several_tried(self) -> None:
        """Selecting the best of five and quoting it without the five is the multiple-comparison
        error this whole project exists to avoid."""
        report = evaluate(_universe(40, drift=200.0, seed=17), [0] * 40)
        assert "best of 5 rules tried" in report.verdict
        assert "itself a selection" in report.verdict

    def test_a_beaten_desk_is_reported_as_having_added_nothing(self) -> None:
        report = evaluate(_universe(40, drift=250.0, seed=18, spread=60.0), [0] * 40)
        assert "NO INCREMENTAL VALUE DEMONSTRATED" in report.verdict

    def test_a_desk_that_beats_the_rules_is_reported_as_such(self) -> None:
        """The verdict must be able to come out in our favour, or it is not a measurement."""
        instants = _universe(40, drift=0.0, seed=19, spread=120.0)
        # A desk with perfect foresight on every instant.
        oracle = [DIRECTION_LONG if i.realised_bps > 0 else DIRECTION_SHORT for i in instants]
        report = evaluate(instants, oracle)
        assert "beats" in report.verdict
        assert report.desk.total_bps > report.best_baseline.total_bps

    def test_it_reports_the_sample_it_can_and_cannot_settle(self) -> None:
        report = evaluate(_universe(40, drift=0.0, seed=20, spread=300.0), [0] * 40)
        assert "cannot settle a small one" in report.verdict

    def test_clustered_symbols_reduce_the_effective_count(self) -> None:
        """Four symbols scored on one hour are not four independent instants."""
        rng = random.Random(21)
        instants = []
        for hour in range(10):
            shared = rng.gauss(150.0, 80.0)
            for s in range(4):
                instants.append(
                    Instant(
                        symbol=f"S{s}",
                        at=START + timedelta(hours=hour * 6),
                        trailing=tuple(rng.gauss(0.0, 0.01) for _ in range(72)),
                        realised_bps=shared + rng.gauss(0.0, 1.0),
                        hurdle_bps=HURDLE,
                    )
                )
        report = evaluate(instants, [0] * 40)
        assert report.effective_instants < 40

    def test_the_rendered_report_shows_every_policy_and_every_test(self) -> None:
        text = evaluate(_universe(40, drift=120.0, seed=22), [0] * 40).render()
        for name in BASELINES:
            assert name in text
        assert "paired sign tests" in text

    def test_the_dict_carries_the_comparisons_not_only_the_headline(self) -> None:
        got = evaluate(_universe(40, drift=120.0, seed=23), [0] * 40).as_dict()
        assert len(got["comparisons"]) == len(BASELINES)
        assert len(got["baselines"]) == len(BASELINES)
        assert got["effective_instants"] <= got["instants"]


class TestCollectingTheRealRecord:
    def test_an_abstention_maps_to_flat(self) -> None:
        for verdict in ("no_trade", "data_insufficient", "delay", "human_review"):
            assert direction_of(verdict, "buy", "1") == DIRECTION_NONE

    def test_a_trade_with_size_maps_to_its_side(self) -> None:
        assert direction_of("trade", "buy", "1") == DIRECTION_LONG
        assert direction_of("trade", "sell", "1") == DIRECTION_SHORT

    def test_a_trade_with_no_size_is_flat(self) -> None:
        """Every live row records side 'buy' with quantity zero. Reading that as a long position
        would invent 126 trades the desk never took."""
        assert direction_of("trade", "buy", "0") == DIRECTION_NONE
        assert direction_of("trade", "buy", None) == DIRECTION_NONE

    def test_an_unparseable_quantity_is_flat_not_an_error(self) -> None:
        assert direction_of("trade", "buy", "n/a") == DIRECTION_NONE

    def test_the_opening_verdicts_are_the_ones_that_put_exposure_on(self) -> None:
        assert frozenset({"trade", "reduce", "hedge"}) == OPENING_VERDICTS

    def test_the_trailing_window_stops_strictly_before_the_instant(self) -> None:
        """The off-by-one that separates a momentum baseline from a look-ahead."""
        stamps = [START + timedelta(hours=i) for i in range(100)]
        returns = [float(i) for i in range(100)]
        got = trailing_returns(stamps, returns, stamps[50], bars=5)
        assert got == (45.0, 46.0, 47.0, 48.0, 49.0)

    def test_extending_the_future_does_not_change_the_window(self) -> None:
        stamps = [START + timedelta(hours=i) for i in range(100)]
        returns = [float(i) for i in range(100)]
        short = trailing_returns(stamps[:60], returns[:60], stamps[50], bars=10)
        long = trailing_returns(stamps, returns, stamps[50], bars=10)
        assert short == long

    def test_an_instant_before_any_history_has_an_empty_window(self) -> None:
        stamps = [START + timedelta(hours=i) for i in range(10)]
        assert trailing_returns(stamps, [1.0] * 10, START - timedelta(hours=1)) == ()

    def test_an_instant_with_too_little_history_is_dropped_from_both_lists(self) -> None:
        """Dropping the instant but keeping its decision would misalign every later pair."""
        stamps = [START + timedelta(hours=i) for i in range(200)]
        returns = [0.001] * 200
        rows = [
            ("A", stamps[5], 100.0, "no_trade", "buy", "0"),
            ("A", stamps[150], 100.0, "no_trade", "buy", "0"),
        ]
        instants, directions = build_instants(rows, {"A": (stamps, returns)})
        assert len(instants) == len(directions) == 1
        assert instants[0].at == stamps[150]

    def test_a_symbol_with_no_history_is_dropped(self) -> None:
        stamps = [START + timedelta(hours=i) for i in range(200)]
        rows = [("MISSING", stamps[150], 100.0, "no_trade", "buy", "0")]
        instants, directions = build_instants(rows, {"A": (stamps, [0.001] * 200)})
        assert instants == [] and directions == []

    def test_the_trailing_length_is_capped_at_the_window(self) -> None:
        stamps = [START + timedelta(hours=i) for i in range(500)]
        rows = [("A", stamps[400], 100.0, "no_trade", "buy", "0")]
        instants, _ = build_instants(rows, {"A": (stamps, [0.001] * 500)})
        assert len(instants[0].trailing) == TRAILING_BARS

    def test_the_minimum_trailing_floor_is_the_stated_one(self) -> None:
        assert MIN_TRAILING == 24
        assert TRAILING_BARS >= MIN_TRAILING
