"""Decision stress testing: the shocks must come from history, and say how often they happened."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from argus.desk.stress import (
    MIN_OBSERVATIONS,
    SEVERITIES,
    STRUCTURAL,
    EmpiricalScenario,
    Move,
    StressError,
    assess,
    empirical_scenarios,
    horizon_moves,
    ordinal,
    quantile_move,
    reverse_stress_liquidity,
)

START = datetime(2026, 6, 1, tzinfo=UTC)


def _closes(values: list[str]) -> list[tuple[datetime, Decimal]]:
    return [(START + timedelta(hours=i), Decimal(v)) for i, v in enumerate(values)]


def _moves(pcts: list[str], phase: str = "rth", start: datetime = START) -> list[Move]:
    return [
        Move(at=start + timedelta(hours=i), pct=Decimal(p), phase=phase)
        for i, p in enumerate(pcts)
    ]


def _many(n: int = 100, phase: str = "rth") -> list[Move]:
    """A descending ramp from +5% to -5%, so percentiles are exactly predictable."""
    step = Decimal("10") / Decimal(n - 1)
    return [
        Move(at=START + timedelta(hours=i), pct=Decimal("5") - step * i, phase=phase)
        for i in range(n)
    ]


class TestHorizonMoves:
    def test_a_move_is_measured_across_the_whole_horizon(self) -> None:
        got = horizon_moves(_closes(["100", "101", "110"]), bars=2)
        assert len(got) == 1
        assert got[0].pct == Decimal("10")

    def test_windows_overlap_so_the_tail_is_not_thrown_away(self) -> None:
        got = horizon_moves(_closes(["100", "110", "120", "130"]), bars=2)
        assert len(got) == 2

    def test_a_move_is_stamped_at_the_end_of_its_window(self) -> None:
        got = horizon_moves(_closes(["100", "101", "110"]), bars=2)
        assert got[0].at == START + timedelta(hours=2)

    def test_a_one_bar_horizon_is_allowed(self) -> None:
        assert len(horizon_moves(_closes(["100", "110"]), bars=1)) == 1

    def test_a_zero_bar_horizon_is_refused(self) -> None:
        with pytest.raises(StressError, match="not a horizon"):
            horizon_moves(_closes(["100", "110"]), bars=0)

    def test_too_few_bars_for_the_horizon_gives_nothing(self) -> None:
        assert horizon_moves(_closes(["100", "110"]), bars=5) == []

    def test_a_zero_price_is_skipped_rather_than_dividing_by_zero(self) -> None:
        got = horizon_moves(_closes(["0", "100", "110"]), bars=1)
        assert [m.pct for m in got] == [Decimal("10")]

    def test_without_a_phase_function_the_phase_is_unknown_not_assumed(self) -> None:
        got = horizon_moves(_closes(["100", "110"]), bars=1)
        assert got[0].phase == "unknown"

    def test_the_phase_function_tags_each_move(self) -> None:
        got = horizon_moves(_closes(["100", "110", "120"]), bars=1, phase_of=lambda t: "rth")
        assert {m.phase for m in got} == {"rth"}


class TestQuantiles:
    def test_a_percentile_is_an_observation_that_really_occurred(self) -> None:
        moves = _many(101)
        got = quantile_move(moves, percentile=Decimal("50"))
        assert got in {m.pct for m in moves}

    def test_a_lower_percentile_is_a_worse_move(self) -> None:
        moves = _many()
        assert (
            quantile_move(moves, percentile=Decimal("1"))
            < quantile_move(moves, percentile=Decimal("50"))
        )

    def test_the_hundredth_percentile_is_the_best_observation(self) -> None:
        moves = _many()
        assert quantile_move(moves, percentile=Decimal("100")) == max(m.pct for m in moves)

    def test_too_few_observations_refuses_rather_than_estimating(self) -> None:
        with pytest.raises(StressError, match="below the 30"):
            quantile_move(_moves(["1", "-1", "2"]), percentile=Decimal("10"))

    def test_the_refusal_explains_why_a_tail_needs_a_sample(self) -> None:
        with pytest.raises(StressError, match="maximum, not an estimate"):
            quantile_move(_moves(["1"] * 5), percentile=Decimal("1"))

    def test_a_percentile_outside_the_range_is_refused(self) -> None:
        for bad in ("0", "-5", "101"):
            with pytest.raises(StressError, match="percentile must be"):
                quantile_move(_many(), percentile=Decimal(bad))


class TestEmpiricalScenarios:
    def test_every_named_severity_is_produced(self) -> None:
        got = empirical_scenarios(_many(), phase="rth", horizon_bars=2)
        assert [s.severity for s in got] == [name for name, _ in SEVERITIES]

    def test_each_scenario_carries_its_sample_size(self) -> None:
        got = empirical_scenarios(_many(80), phase="rth", horizon_bars=2)
        assert all(s.observations == 80 for s in got)

    def test_each_scenario_says_how_often_it_happened(self) -> None:
        got = empirical_scenarios(_many(100), phase="rth", horizon_bars=2)
        extreme = next(s for s in got if s.severity == "extreme")
        assert extreme.occurrences >= 1
        assert extreme.frequency_pct <= Decimal("5")

    def test_a_worse_severity_happens_less_often(self) -> None:
        got = {s.severity: s for s in empirical_scenarios(_many(200), phase="rth", horizon_bars=2)}
        assert got["extreme"].occurrences < got["severe"].occurrences
        assert got["severe"].occurrences < got["adverse"].occurrences

    def test_each_scenario_says_when_it_was_last_seen(self) -> None:
        got = empirical_scenarios(_many(), phase="rth", horizon_bars=2)
        assert all(s.last_seen is not None for s in got)

    def test_phase_filtering_uses_only_that_phase(self) -> None:
        pool = _many(60, phase="rth") + _many(60, phase="weekend")
        got = empirical_scenarios(pool, phase="rth", horizon_bars=2)
        assert all(s.observations == 60 for s in got)

    def test_a_phase_with_too_few_windows_yields_nothing_rather_than_blending(self) -> None:
        """The core honesty property: no percentile is better than a cross-session one."""
        pool = _many(60, phase="rth") + _moves(["1"] * 5, phase="weekend")
        assert empirical_scenarios(pool, phase="weekend", horizon_bars=2) == []

    def test_passing_no_phase_pools_everything_and_labels_it_all(self) -> None:
        pool = _many(40, phase="rth") + _many(40, phase="weekend")
        got = empirical_scenarios(pool, phase=None, horizon_bars=2)
        assert got and all(s.phase == "all" for s in got)
        assert all(s.observations == 80 for s in got)

    def test_the_horizon_is_carried_into_the_scenario(self) -> None:
        got = empirical_scenarios(_many(), phase="rth", horizon_bars=7)
        assert all(s.horizon_bars == 7 for s in got)

    def test_the_name_distinguishes_severity_and_phase(self) -> None:
        got = empirical_scenarios(_many(), phase="rth", horizon_bars=2)
        assert {s.name for s in got} == {f"{n}_rth" for n, _ in SEVERITIES}

    def test_it_converts_to_a_workbench_scenario_with_the_same_shock(self) -> None:
        got = empirical_scenarios(_many(), phase="rth", horizon_bars=2)[0]
        assert got.to_scenario().price_shock_pct == got.shock_pct

    def test_the_rendered_line_names_frequency_and_recency(self) -> None:
        got = empirical_scenarios(_many(), phase="rth", horizon_bars=2)[-1]
        text = got.render()
        assert "percentile of 100 observed windows" in text
        assert "time(s)" in text and "most recently" in text

    def test_it_serialises_as_measured(self) -> None:
        got = empirical_scenarios(_many(), phase="rth", horizon_bars=2)[0]
        assert got.as_dict()["measured"] is True

    def test_frequency_of_an_empty_sample_is_zero_not_a_crash(self) -> None:
        lone = EmpiricalScenario(
            "x", "severe", Decimal("5"), Decimal("-1"), 0, 0, None, "rth", 2
        )
        assert lone.frequency_pct == Decimal("0")


class TestStructuralScenarios:
    def test_every_structural_scenario_states_its_assumption(self) -> None:
        assert all(s.assumption.strip() for s in STRUCTURAL)

    def test_the_ones_that_are_guesses_say_not_measured(self) -> None:
        guessed = [s for s in STRUCTURAL if "NOT MEASURED" in s.assumption]
        assert {s.name for s in guessed} == {
            "hedge_unavailable", "spread_treble", "venue_outage"
        }

    def test_a_venue_outage_is_modelled_as_effectively_no_book(self) -> None:
        outage = next(s for s in STRUCTURAL if s.name == "venue_outage")
        assert outage.liquidity_multiplier <= Decimal("0.05")

    def test_stale_evidence_asserts_no_market_effect_at_all(self) -> None:
        """Listed precisely so a report cannot omit a failure mode with no price shock."""
        stale = next(s for s in STRUCTURAL if s.name == "stale_evidence")
        assert stale.price_shock_pct == Decimal("0")
        assert stale.liquidity_multiplier == Decimal("1")

    def test_they_serialise_as_not_measured(self) -> None:
        assert all(s.as_dict()["measured"] is False for s in STRUCTURAL)

    def test_they_convert_to_workbench_scenarios(self) -> None:
        for s in STRUCTURAL:
            assert s.to_scenario().name == s.name


class TestTheReport:
    def _report(self, **over: object):  # type: ignore[no-untyped-def]
        kwargs: dict[str, object] = {
            "symbol": "NVDAUSDT", "quantity": Decimal("1"),
            "entry_price": Decimal("200"), "loss_tolerance_pct": Decimal("5"),
            "moves": _many(120), "phase": "rth", "horizon_bars": 2, "simulate": False,
        }
        kwargs.update(over)
        return assess(**kwargs)  # type: ignore[arg-type]

    def test_it_tests_both_empirical_and_structural_scenarios(self) -> None:
        got = self._report()
        assert got.empirical and got.structural
        assert len(got.results) == len(got.empirical) + len(got.structural)

    def test_it_states_what_share_of_scenarios_is_measured(self) -> None:
        got = self._report()
        assert Decimal("0") < got.measured_share < Decimal("100")

    def test_with_no_history_it_says_every_shock_is_an_assumption(self) -> None:
        got = self._report(moves=[])
        assert got.empirical == []
        assert got.measured_share == Decimal("0")
        assert any("every shock in this report is an assumption" in n for n in got.notes)

    def test_with_no_history_the_render_explains_the_absence(self) -> None:
        text = " ".join(self._report(moves=[]).render())
        assert f"fewer than {MIN_OBSERVATIONS} observed windows" in text

    def test_a_tight_tolerance_produces_failures(self) -> None:
        got = self._report(loss_tolerance_pct=Decimal("0.5"))
        assert got.failures and not got.survives_all

    def test_a_loose_tolerance_survives(self) -> None:
        assert self._report(loss_tolerance_pct=Decimal("50")).survives_all

    def test_a_zero_position_is_refused(self) -> None:
        with pytest.raises(StressError, match="tests nothing"):
            self._report(quantity=Decimal("0"))

    def test_a_non_positive_price_is_refused(self) -> None:
        with pytest.raises(StressError, match="entry price"):
            self._report(entry_price=Decimal("0"))

    def test_the_phase_is_recorded_on_the_report(self) -> None:
        assert self._report(phase="rth").phase == "rth"
        assert self._report(phase=None).phase == "all"

    def test_the_render_marks_each_exit_cost_measured_or_assumed(self) -> None:
        text = " ".join(self._report().render())
        assert "assumed" in text

    def test_it_serialises_whole(self) -> None:
        got = self._report().as_dict()
        assert got["symbol"] == "NVDAUSDT"
        assert len(got["results"]) == len(got["empirical"]) + len(got["structural"])
        assert "measured_share_pct" in got

    def test_survives_all_is_false_when_there_is_nothing_to_test(self) -> None:
        got = assess(
            symbol="X", quantity=Decimal("1"), entry_price=Decimal("100"),
            loss_tolerance_pct=Decimal("5"), moves=[], structural=(), simulate=False,
        )
        assert got.results == [] and not got.survives_all


class TestOrdinals:
    @pytest.mark.parametrize(
        ("value", "text"),
        [("1", "1st"), ("2", "2nd"), ("3", "3rd"), ("4", "4th"), ("5", "5th"),
         ("11", "11th"), ("12", "12th"), ("13", "13th"), ("21", "21st"), ("50", "50th")],
    )
    def test_ordinals_read_as_english(self, value: str, text: str) -> None:
        assert ordinal(Decimal(value)) == text


def test_phase_conditioning_changes_the_answer() -> None:
    """The reason the module conditions at all, as a test rather than a claim.

    A calm regular session and a violent weekend produce very different tails, and a blended
    estimate describes neither.
    """
    calm = [
        Move(at=START + timedelta(hours=i), pct=Decimal("0.1"), phase="rth") for i in range(60)
    ]
    violent = [
        Move(at=START + timedelta(hours=i), pct=Decimal("-8"), phase="weekend") for i in range(60)
    ]
    pool = calm + violent
    rth = empirical_scenarios(pool, phase="rth", horizon_bars=2)
    weekend = empirical_scenarios(pool, phase="weekend", horizon_bars=2)
    blended = empirical_scenarios(pool, phase=None, horizon_bars=2)

    rth_extreme = next(s for s in rth if s.severity == "extreme").shock_pct
    weekend_extreme = next(s for s in weekend if s.severity == "extreme").shock_pct
    blended_extreme = next(s for s in blended if s.severity == "extreme").shock_pct

    assert rth_extreme > blended_extreme > weekend_extreme or rth_extreme > weekend_extreme
    assert rth_extreme != weekend_extreme


class TestReverseStressLiquidity:
    """Bisects the REAL simulated book (`sim.market.exit_cost_bps`) to find where a position
    stops being fully exitable — the measured answer `hedge_unavailable`'s flat 0.5x assumption
    admits it cannot give. Fixtures chosen from a direct empirical sweep of the simulator (done
    before writing the bisection, not assumed): qty=500 never breaks in [0.01, 1.0] at this
    price; qty=1500 breaks cleanly between depth 0.3 and 0.4; qty=5000 stays unexitable even at
    the healthiest depth tested."""

    def test_a_small_position_never_breaks_within_the_searched_range(self) -> None:
        result = reverse_stress_liquidity(
            quantity=Decimal("500"), entry_price=Decimal("100"), price_shock_pct=Decimal("-10"),
        )
        assert result.breaking_liquidity_multiplier is None
        assert not result.always_breaks

    def test_an_oversized_position_always_breaks_even_at_the_healthiest_depth(self) -> None:
        result = reverse_stress_liquidity(
            quantity=Decimal("5000"), entry_price=Decimal("100"), price_shock_pct=Decimal("-10"),
        )
        assert result.always_breaks
        assert result.breaking_liquidity_multiplier == result.searched_ceiling

    def test_a_position_with_a_genuine_breaking_point_is_found_by_bisection(self) -> None:
        result = reverse_stress_liquidity(
            quantity=Decimal("1500"), entry_price=Decimal("100"), price_shock_pct=Decimal("-10"),
        )
        assert not result.always_breaks
        assert result.breaking_liquidity_multiplier is not None
        # Verified empirically: breaks between 0.3 (not exitable) and 0.4 (exitable).
        assert Decimal("0.3") < result.breaking_liquidity_multiplier < Decimal("0.4")
        assert result.iterations > 0

    def test_the_result_is_reproducible(self) -> None:
        """The underlying simulator is seeded (`exit_cost_bps`'s default `seed=20260912`), so the
        same inputs must find the same breaking point every time."""
        kwargs = dict(quantity=Decimal("1500"), entry_price=Decimal("100"),
                      price_shock_pct=Decimal("-10"))
        first = reverse_stress_liquidity(**kwargs)
        second = reverse_stress_liquidity(**kwargs)
        assert first.breaking_liquidity_multiplier == second.breaking_liquidity_multiplier

    def test_render_names_the_measured_breaking_point(self) -> None:
        result = reverse_stress_liquidity(
            quantity=Decimal("1500"), entry_price=Decimal("100"), price_shock_pct=Decimal("-10"),
        )
        assert "breaks at" in result.render()
        assert "0.5x assumption" in result.render()

    def test_render_names_the_always_breaks_case_distinctly(self) -> None:
        result = reverse_stress_liquidity(
            quantity=Decimal("5000"), entry_price=Decimal("100"), price_shock_pct=Decimal("-10"),
        )
        assert "oversized for this book" in result.render()

    def test_render_names_the_never_breaks_case_distinctly(self) -> None:
        result = reverse_stress_liquidity(
            quantity=Decimal("500"), entry_price=Decimal("100"), price_shock_pct=Decimal("-10"),
        )
        assert "no breaking point found" in result.render()
