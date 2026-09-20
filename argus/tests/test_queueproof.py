"""Tests for the queue model's evidence package.

The module under test exists to decide whether a capability has earned OWNED, so the failure mode
that matters is not a crash — it is an experiment that cannot produce a negative result. Most of
what follows therefore checks that the harness *can* report the model losing, that the ground truth
is really ground truth, and that the numbers are the same on a second run.
"""

from __future__ import annotations

import pytest

from argus.eval.queueproof import (
    GRID,
    REGIMES,
    TOLERANCE,
    Reproduction,
    Scored,
    evaluate_regime,
    paired_interval,
    reproduce,
    score,
    simulate,
)

EPISODES = 250
SEED = 424242

BY_NAME = {r.name: r for r in REGIMES}
GAMMA_ONE = BY_NAME["gamma=1 - cancels proportional to quantity"]
FRONT_STICKY = BY_NAME["front-sticky - the front never cancels"]
ADVERSARIAL = BY_NAME["adversarial - front-sticky and replenishing"]


class TestReproduction:
    def test_every_function_matches_the_rust_formula(self) -> None:
        got = reproduce()
        assert len(got) == 8
        for row in got:
            assert row.reproduced, f"{row.model} deviates by {row.max_deviation:.2e}"
            assert row.cases == len(GRID)

    def test_the_worst_deviation_is_floating_point_noise(self) -> None:
        """Not merely 'within tolerance' — at the scale of double rounding, which is what a
        transcription of the same arithmetic should produce."""
        assert max(r.max_deviation for r in reproduce()) < 1e-15

    def test_the_tolerance_can_still_fail(self) -> None:
        """A tolerance nothing can fail is decoration. This is the guard against widening it."""
        assert not Reproduction(model="x", cases=9, max_deviation=TOLERANCE * 10).reproduced


class TestTheGroundTruthIsGroundTruth:
    def test_the_quantity_ahead_never_rises(self) -> None:
        """Joins go behind us and trades consume from the front, so nothing can push us back. If
        this ever fails, the simulator is not modelling a FIFO queue and every number is void."""
        for episode in simulate(60, seed=SEED, regime=GAMMA_ONE):
            aheads = [o.true_ahead for o in episode.observations]
            assert aheads == sorted(aheads, reverse=True)

    def test_a_fill_never_exceeds_the_order(self) -> None:
        for episode in simulate(60, seed=SEED, regime=GAMMA_ONE):
            assert 0.0 <= episode.true_filled <= 1.0

    def test_levels_are_never_negative(self) -> None:
        for episode in simulate(60, seed=SEED, regime=GAMMA_ONE):
            assert all(o.new_level >= -1e-9 for o in episode.observations)

    def test_the_front_sticky_regime_really_never_cancels_the_front(self) -> None:
        """The regime's whole claim, checked against the episodes rather than the docstring: the
        quantity ahead can only fall by exactly the quantity that traded."""
        for episode in simulate(60, seed=SEED, regime=FRONT_STICKY):
            previous = episode.entry_level
            for obs in episode.observations:
                assert obs.true_ahead >= previous - obs.traded - 1e-6
                previous = obs.true_ahead


class TestTheHarnessCanReportALoss:
    def test_a_perfect_assumption_scores_exactly_zero(self) -> None:
        """In the front-sticky regime 'every cancellation is behind you' is not an approximation,
        it is the generative rule. A harness that cannot score it at zero is measuring noise."""
        scored = score(simulate(EPISODES, seed=SEED, regime=FRONT_STICKY))
        perfect = next(s for s in scored if s.model == "ABLATION: every cancel is behind you")
        assert perfect.queue_error == pytest.approx(0.0, abs=1e-12)
        assert perfect.fill_error == pytest.approx(0.0, abs=1e-12)

    def test_the_model_loses_that_regime_and_says_so(self) -> None:
        result, _ = evaluate_regime(FRONT_STICKY, episodes=EPISODES, seed=SEED)
        assert not result.model_wins
        assert result.margin < 0

    def test_the_model_loses_the_adversarial_regime(self) -> None:
        result, _ = evaluate_regime(ADVERSARIAL, episodes=EPISODES, seed=SEED)
        assert not result.model_wins

    def test_at_least_one_regime_is_a_loss(self) -> None:
        """A sweep in which the model wins everywhere is a sweep whose regimes were chosen."""
        results = [evaluate_regime(r, episodes=120, seed=SEED)[0] for r in REGIMES]
        assert not all(r.model_wins for r in results)
        assert any(r.model_wins for r in results)


class TestTheAblation:
    def test_the_naive_assumptions_are_present_and_distinct(self) -> None:
        scored = score(simulate(120, seed=SEED, regime=GAMMA_ONE))
        ablations = {s.model: s.queue_error for s in scored if s.is_ablation}
        assert len(ablations) == 4
        # const(1) and RiskAdverse are the documented exception: assuming every cancellation is
        # behind you means the front only moves on trades, which is what RiskAdverse does.
        distinct = set(ablations.values())
        assert len(distinct) == 3

    def test_risk_adverse_is_exactly_the_always_behind_assumption(self) -> None:
        scored = {s.model: s for s in score(simulate(120, seed=SEED, regime=GAMMA_ONE))}
        assert scored["ABLATION: RiskAdverse (trades only)"].queue_error == pytest.approx(
            scored["ABLATION: every cancel is behind you"].queue_error
        )

    def test_a_real_model_beats_the_naive_ones_where_the_rule_is_in_the_family(self) -> None:
        result, _ = evaluate_regime(GAMMA_ONE, episodes=EPISODES, seed=SEED)
        assert result.model_wins
        assert result.margin_lo > 0, "the margin's interval must clear zero, not just its mean"


class TestTheStatistics:
    def test_an_identical_pair_has_a_zero_margin(self) -> None:
        a = Scored("a", 3, 0.1, 0.0, 0.0, (0.1, 0.2, 0.3))
        mean, lo, hi = paired_interval(a, a)
        assert (mean, lo, hi) == (0.0, 0.0, 0.0)

    def test_a_constant_difference_has_a_zero_width_interval(self) -> None:
        better = Scored("b", 3, 0.1, 0.0, 0.0, (0.1, 0.2, 0.3))
        worse = Scored("w", 3, 0.2, 0.0, 0.0, (0.2, 0.3, 0.4))
        mean, lo, hi = paired_interval(better, worse)
        assert mean == pytest.approx(0.1)
        assert lo == pytest.approx(0.1)
        assert hi == pytest.approx(0.1)

    def test_noise_around_zero_does_not_clear_the_interval(self) -> None:
        """The property that makes `model_wins` mean something: a difference indistinguishable
        from zero must not be reported as a win."""
        better = Scored("b", 4, 0.0, 0.0, 0.0, (0.10, 0.30, 0.10, 0.30))
        worse = Scored("w", 4, 0.0, 0.0, 0.0, (0.30, 0.10, 0.30, 0.10))
        mean, lo, _ = paired_interval(better, worse)
        assert mean == pytest.approx(0.0)
        assert lo < 0


class TestTheHeldOutRun:
    def test_the_pick_is_carried_over_rather_than_rechosen(self) -> None:
        """The whole point of the held-out run. If it re-picked the winner it would be running the
        selection again, and six candidates competing for 'best' inflate a margin on their own."""
        pick = ("LogProbability", "ABLATION: coin")
        result, _ = evaluate_regime(GAMMA_ONE, episodes=120, seed=SEED, pick=pick)
        assert (result.best_model, result.best_naive) == pick

    def test_the_in_sample_winner_survives_a_seed_it_never_saw(self) -> None:
        in_sample, _ = evaluate_regime(GAMMA_ONE, episodes=EPISODES, seed=SEED)
        held_out, _ = evaluate_regime(
            GAMMA_ONE, episodes=EPISODES, seed=SEED + 9_000,
            pick=(in_sample.best_model, in_sample.best_naive),
        )
        assert held_out.model_wins
        assert held_out.margin_lo > 0


class TestReproducibility:
    def test_the_same_seed_gives_the_same_episodes(self) -> None:
        first = simulate(40, seed=SEED, regime=GAMMA_ONE)
        second = simulate(40, seed=SEED, regime=GAMMA_ONE)
        assert first == second

    def test_the_same_seed_gives_the_same_scores(self) -> None:
        """Condition twelve. Pure Python, one seeded generator, no wall clock in the path."""
        first = [s.as_dict() for s in score(simulate(120, seed=SEED, regime=GAMMA_ONE))]
        second = [s.as_dict() for s in score(simulate(120, seed=SEED, regime=GAMMA_ONE))]
        assert first == second

    def test_a_different_seed_gives_different_episodes(self) -> None:
        assert simulate(40, seed=SEED, regime=GAMMA_ONE) != simulate(
            40, seed=SEED + 1, regime=GAMMA_ONE
        )


class TestCosts:
    def test_overstated_fills_carry_a_positive_cost(self) -> None:
        """The fill error is converted at the 4bps gap between our own maker and taker constants,
        so a model that awards itself fills it would not have got is charged for them."""
        scored = score(simulate(EPISODES, seed=SEED, regime=FRONT_STICKY))
        assert all(s.cost_bps >= 0 for s in scored)
        optimistic = next(s for s in scored if s.model == "ABLATION: every cancel is ahead of you")
        exact = next(s for s in scored if s.model == "ABLATION: every cancel is behind you")
        assert optimistic.cost_bps > exact.cost_bps == pytest.approx(0.0)

    def test_the_cheapest_model_is_not_always_the_most_accurate(self) -> None:
        """A finding, pinned so it cannot be quietly dropped: queue accuracy and execution cost are
        different objectives, and at least one regime ranks them differently."""
        scored = score(simulate(EPISODES, seed=SEED, regime=GAMMA_ONE))
        by_error = min(scored, key=lambda s: s.queue_error)
        by_cost = min(scored, key=lambda s: s.cost_bps)
        assert by_error.model != by_cost.model
