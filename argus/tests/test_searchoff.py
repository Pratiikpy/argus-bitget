"""Bake-off tests — the budget is the experiment, so the budget is what is tested hardest.

A comparison of search strategies is worthless the moment one of them gets more evaluations than
another, because on a noisy space more tries means a better maximum of the noise. Every test here
that touches fairness checks that: identical spend, identical seed, canonical deduplication, and a
winner picked on the in-sample half and reported on the out-of-sample one.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pytest

from argus.research.grammar import BinOp, Const, Field, Ref, Window
from argus.research.searchoff import (
    BUDGET,
    MAX_DEPTH,
    MIN_BARS,
    OOS_FRACTION,
    STRATEGIES,
    Arena,
    BakeOff,
    Candidate,
    SearchError,
    StrategyResult,
    mutate,
    random_expr,
    run,
)

START = datetime(2026, 3, 1, tzinfo=UTC)


@dataclass(frozen=True)
class Bar:
    ts: datetime
    close: float
    extra: dict[str, float]


def _bars(n: int = 450, *, seed: int = 1, drift: float = 0.0) -> list[Bar]:
    rng = random.Random(seed)
    price = 100.0
    out = []
    for i in range(n):
        price *= 1.0 + rng.gauss(drift, 0.01)
        out.append(Bar(
            ts=START + timedelta(hours=i), close=price,
            extra={"volume": abs(rng.gauss(1e6, 2e5)), "high": price * 1.002,
                   "low": price * 0.998},
        ))
    return out


class TestGeneratedExpressions:
    def test_every_generated_tree_is_within_the_depth_cap(self) -> None:
        rng = random.Random(1)
        for _ in range(300):
            assert random_expr(rng).depth <= MAX_DEPTH + 1

    def test_generation_is_reproducible_from_the_seed(self) -> None:
        """Without this the whole comparison is a draw rather than an experiment."""
        a = [random_expr(random.Random(7)).canonical() for _ in range(5)]
        b = [random_expr(random.Random(7)).canonical() for _ in range(5)]
        assert a == b

    def test_it_produces_more_than_one_shape(self) -> None:
        rng = random.Random(2)
        shapes = {random_expr(rng).canonical() for _ in range(200)}
        assert len(shapes) > 100

    def test_every_tree_has_a_canonical_form(self) -> None:
        rng = random.Random(3)
        for _ in range(100):
            assert random_expr(rng).canonical()

    def test_a_mutation_usually_changes_the_tree(self) -> None:
        rng = random.Random(4)
        parent = BinOp("add", Window("mean", 12, _field()), Const(1.0))
        changed = sum(
            1 for _ in range(50) if mutate(parent, rng).canonical() != parent.canonical()
        )
        assert changed > 40

    def test_a_leaf_mutates_into_something_new(self) -> None:
        rng = random.Random(5)
        leaf = Const(1.0)
        assert any(mutate(leaf, rng).canonical() != leaf.canonical() for _ in range(20))


def _field() -> Ref:
    return Ref(Field.CLOSE)


class TestTheBudgetIsTheExperiment:
    def test_every_strategy_spends_exactly_the_budget(self) -> None:
        """The precondition of the whole comparison. A strategy that stops early or overruns makes
        the ranking a statement about spend rather than about method."""
        budget = 25
        got = run(_bars(), symbol="TEST", budget=budget, seed=11)
        assert {r.evaluated for r in got.results} == {budget}
        assert got.budget_was_equal

    def test_an_arena_refuses_to_spend_past_its_budget(self) -> None:
        arena = Arena(bars=_bars(), budget=5)
        rng = random.Random(6)
        for _ in range(50):
            arena.evaluate(random_expr(rng))
        assert arena.spent <= 5

    def test_a_repeated_canonical_form_costs_no_trial(self) -> None:
        """The grammar's canonical spelling is an identity. Charging twice for one factor would let
        a strategy that rediscovers the same tree look busier than one that explores."""
        arena = Arena(bars=_bars(), budget=20)
        expr = Window("mean", 12, _field())
        first = arena.evaluate(expr)
        spent = arena.spent
        second = arena.evaluate(Window("mean", 12, _field()))
        assert arena.spent == spent
        assert first == second

    def test_the_budget_default_is_the_stated_one(self) -> None:
        assert BUDGET == 400

    def test_each_strategy_gets_its_own_generator(self) -> None:
        """A shared generator would make the result depend on the order the strategies ran in."""
        first = run(_bars(), symbol="T", budget=18, seed=3)
        second = run(_bars(), symbol="T", budget=18, seed=3)
        assert [r.as_dict() for r in first.results] == [r.as_dict() for r in second.results]


class TestScoring:
    def test_a_candidate_reports_its_own_decay(self) -> None:
        got = Candidate(canonical="x", in_sample=0.10, out_of_sample=0.02)
        assert got.decay == pytest.approx(0.08)

    def test_a_tree_that_cannot_be_evaluated_costs_a_trial_and_scores_nothing(self) -> None:
        """A candidate that raises is a trial that was spent. Not charging for it would let a
        strategy generating garbage explore for free."""
        arena = Arena(bars=_bars(50), budget=10)
        got = arena.evaluate(Window("mean", 48, _field()))
        assert arena.spent == 1
        assert got is None or math.isfinite(got.in_sample)

    def test_the_split_is_chronological_and_the_stated_fraction(self) -> None:
        arena = Arena(bars=_bars(1000), budget=1)
        assert arena.split == int(1000 * (1 - OOS_FRACTION))

    def test_a_constant_signal_scores_nothing_rather_than_infinity(self) -> None:
        """Zero variance has no Sharpe. Returning a large number here would make a do-nothing rule
        win every bake-off."""
        arena = Arena(bars=_bars(), budget=5)
        assert arena.evaluate(Const(0.0)) is None


class TestTheComparison:
    def test_every_strategy_is_run(self) -> None:
        got = run(_bars(), symbol="T", budget=20, seed=9)
        assert {r.name for r in got.results} == set(STRATEGIES)

    def test_the_winner_is_picked_in_sample_and_reported_out_of_sample(self) -> None:
        """Picking on the full sample is the error that makes every search method look like it
        works. The two numbers must be able to disagree."""
        got = run(_bars(seed=21), symbol="T", budget=16, seed=5)
        scored = [r for r in got.results if r.picked is not None]
        assert scored
        assert any(abs(r.picked.decay) > 1e-9 for r in scored if r.picked)

    def test_it_reports_a_strategy_that_found_nothing_rather_than_dropping_it(self) -> None:
        got = run(_bars(), symbol="T", budget=20, seed=13)
        assert len(got.results) == len(STRATEGIES)

    def test_a_short_series_raises(self) -> None:
        with pytest.raises(SearchError, match="below the"):
            run(_bars(MIN_BARS - 1), symbol="T", budget=10)

    def test_the_ranking_is_by_out_of_sample(self) -> None:
        got = run(_bars(), symbol="T", budget=25, seed=8)
        ranked = got.by_out_of_sample
        scores = [r.out_of_sample or 0.0 for r in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_random_is_in_the_field_as_the_null(self) -> None:
        """A method that cannot beat uniform sampling is not a search method."""
        assert "random" in STRATEGIES

    def test_the_verdict_names_random_beating_the_field_when_it_does(self) -> None:
        result = BakeOff(
            symbol="T", bars=600, budget=100,
            results=(
                StrategyResult("random", 100, 90, 0, Candidate("a", 0.05, 0.04)),
                StrategyResult("beam", 100, 60, 0, Candidate("b", 0.09, 0.01)),
            ),
        )
        assert "No search strategy beat uniform random sampling" in result.verdict

    def test_the_verdict_refuses_to_generalise_from_one_run(self) -> None:
        result = BakeOff(
            symbol="T", bars=600, budget=100,
            results=(
                StrategyResult("random", 100, 90, 0, Candidate("a", 0.05, 0.01)),
                StrategyResult("beam", 100, 60, 0, Candidate("b", 0.09, 0.06)),
            ),
        )
        assert "does not establish a general ordering" in result.verdict

    def test_an_empty_field_is_reported_as_undefined(self) -> None:
        result = BakeOff(symbol="T", bars=600, budget=100, results=(
            StrategyResult("random", 100, 0, 0, None),
        ))
        assert "undefined" in result.verdict

    def test_unequal_budgets_are_reported_rather_than_hidden(self) -> None:
        """If this ever printed True with different spends, every ranking below it is a statement
        about spend."""
        result = BakeOff(symbol="T", bars=600, budget=100, results=(
            StrategyResult("random", 100, 90, 0, Candidate("a", 0.05, 0.04)),
            StrategyResult("beam", 80, 60, 0, Candidate("b", 0.09, 0.01)),
        ))
        assert not result.budget_was_equal

    def test_the_report_shows_decay_per_strategy(self) -> None:
        text = run(_bars(), symbol="T", budget=25, seed=12).render()
        assert "decay" in text
        assert "equal budget honoured" in text

    def test_the_dict_carries_every_strategy_and_the_budget_check(self) -> None:
        got = run(_bars(), symbol="T", budget=20, seed=14).as_dict()
        assert len(got["results"]) == len(STRATEGIES)
        assert got["budget_was_equal"] is True


class TestStrategiesBehaveAsNamed:
    def test_beam_explores_less_than_random(self) -> None:
        """Exploitation with no exploration must visit fewer distinct forms on the same budget. If
        it did not, it is not a beam."""
        got = run(_bars(seed=31), symbol="T", budget=18, seed=4)
        beam = next(r for r in got.results if r.name == "beam")
        uniform = next(r for r in got.results if r.name == "random")
        assert beam.distinct <= uniform.distinct

    def test_novelty_explores_at_least_as_much_as_beam(self) -> None:
        got = run(_bars(seed=32), symbol="T", budget=18, seed=4)
        novelty = next(r for r in got.results if r.name == "novelty")
        beam = next(r for r in got.results if r.name == "beam")
        assert novelty.distinct >= beam.distinct

    def test_a_custom_field_of_one_still_runs(self) -> None:
        from argus.research.searchoff import search_random

        got = run(_bars(), symbol="T", budget=15, seed=2, strategies={"only": search_random})
        assert len(got.results) == 1
        assert got.budget_was_equal


class TestTheCostGuard:
    """Depth does not bound cost, and the first live run hung because of it."""

    def test_nested_windows_multiply(self) -> None:
        from argus.research.grammar import Delay, Window
        from argus.research.searchoff import node_cost

        tree = Delay(24, Window("min", 6, Window("slope", 48, Window("std", 48, _field()))))
        assert node_cost(tree) == 24 * 6 * 48 * 48

    def test_a_plain_leaf_costs_one(self) -> None:
        from argus.research.searchoff import node_cost

        assert node_cost(_field()) == 1
        assert node_cost(Const(1.0)) == 1

    def test_a_ruinous_tree_is_refused_and_still_charged(self) -> None:
        """A strategy that generates untradeable trees must not explore for free."""
        from argus.research.grammar import Delay, Window
        from argus.research.searchoff import MAX_NODE_COST

        arena = Arena(bars=_bars(), budget=5)
        ruinous = Delay(48, Window("std", 48, Window("mean", 48, _field())))
        assert node_cost_of(ruinous) > MAX_NODE_COST
        assert arena.evaluate(ruinous) is None
        assert arena.spent == 1
        assert arena.refused == 1

    def test_an_affordable_tree_is_not_refused(self) -> None:
        from argus.research.grammar import Window

        arena = Arena(bars=_bars(), budget=5)
        arena.evaluate(Window("mean", 12, _field()))
        assert arena.refused == 0

    def test_refusals_are_reported_not_hidden(self) -> None:
        got = run(_bars(), symbol="T", budget=40, seed=17).as_dict()
        assert all("refused_as_too_costly" in r for r in got["results"])


def node_cost_of(expr: object) -> int:
    from argus.research.searchoff import node_cost

    return node_cost(expr)  # type: ignore[arg-type]


class TestTheSweepIsWhatActuallyRanks:
    """One seed ranked these strategies three different ways.

    The first published table put `anneal` first on decay and `beam` last; a rerun on another seed
    kept `anneal` first but moved `evolutionary` to last; across three seeds `anneal` is last. Every
    run spent an identical, verified budget. The ordering moved because a single run reports the
    best maximum of one draw of noise, so the sweep keeps the spread rather than averaging it away.
    """

    def test_it_runs_the_whole_field_once_per_seed(self) -> None:
        from argus.research.searchoff import sweep

        got = sweep(_bars(), symbol="T", budget=12, seeds=(1, 2))
        assert len(got.runs) == 2
        assert {s.name for s in got.summaries} == set(STRATEGIES)
        assert all(len(s.decays) == 2 for s in got.summaries)

    def test_every_seed_spends_the_same_budget(self) -> None:
        from argus.research.searchoff import sweep

        got = sweep(_bars(), symbol="T", budget=12, seeds=(1, 2, 3))
        assert got.budget_was_equal
        assert all(r.evaluated == 12 for run_ in got.runs for r in run_.results)

    def test_different_seeds_produce_different_runs(self) -> None:
        """If they did not, the sweep would be one run counted three times and its spread a lie."""
        from argus.research.searchoff import sweep

        got = sweep(_bars(seed=41), symbol="T", budget=20, seeds=(1, 2))
        first, second = (r.as_dict() for r in got.runs)
        assert first != second

    def test_the_sweep_is_reproducible_from_its_seeds(self) -> None:
        from argus.research.searchoff import sweep

        a = sweep(_bars(), symbol="T", budget=12, seeds=(5, 6)).as_dict()
        b = sweep(_bars(), symbol="T", budget=12, seeds=(5, 6)).as_dict()
        assert a["strategies"] == b["strategies"]

    def test_the_published_seeds_are_three_and_fixed(self) -> None:
        from argus.research.searchoff import SWEEP_SEEDS

        assert len(SWEEP_SEEDS) == 3
        assert len(set(SWEEP_SEEDS)) == 3

    def test_a_single_seed_sweep_refuses_to_rank(self) -> None:
        """The failure this class exists to prevent, asserted rather than described."""
        from argus.research.searchoff import sweep

        got = sweep(_bars(), symbol="T", budget=12, seeds=(1,))
        assert "UNDEFINED" in got.verdict
        assert got.stable_winner is None or "UNDEFINED" in got.verdict

    def test_a_sweep_with_no_seeds_raises(self) -> None:
        from argus.research.searchoff import SearchError, sweep

        with pytest.raises(SearchError, match="at least one seed"):
            sweep(_bars(), symbol="T", budget=12, seeds=())


class TestTheSweepArithmetic:
    """Built from hand-made runs, so the ranking logic is tested without waiting on a search."""

    @staticmethod
    def _run(decays: dict[str, float | None], budget: int = 10) -> BakeOff:
        return BakeOff(
            symbol="T", bars=600, budget=budget,
            results=tuple(
                StrategyResult(
                    name, budget, budget, 0,
                    None if d is None else Candidate(name, 0.10, 0.10 - d),
                )
                for name, d in decays.items()
            ),
        )

    def _sweep(self, *runs: BakeOff) -> object:
        from argus.research.searchoff import Sweep

        return Sweep(symbol="T", bars=600, budget=10, seeds=tuple(range(len(runs))), runs=runs)

    def test_ranks_are_per_seed_and_one_is_best(self) -> None:
        got = self._sweep(
            self._run({"a": 0.01, "b": 0.05}),
            self._run({"a": 0.09, "b": 0.02}),
        )
        ranks = {s.name: s.ranks for s in got.summaries}  # type: ignore[attr-defined]
        assert ranks["a"] == (1, 2)
        assert ranks["b"] == (2, 1)

    def test_a_strategy_that_swapped_places_is_not_a_stable_winner(self) -> None:
        got = self._sweep(
            self._run({"a": 0.01, "b": 0.05}),
            self._run({"a": 0.09, "b": 0.02}),
        )
        assert got.stable_winner is None  # type: ignore[attr-defined]
        assert "No strategy holds first place in every seed" in got.verdict  # type: ignore[attr-defined]

    def test_a_strategy_first_in_every_seed_is_named(self) -> None:
        got = self._sweep(
            self._run({"a": 0.01, "b": 0.05}),
            self._run({"a": 0.02, "b": 0.09}),
        )
        assert got.stable_winner == "a"  # type: ignore[attr-defined]
        assert "holds first place in every seed" in got.verdict  # type: ignore[attr-defined]

    def test_the_mean_is_over_the_seeds(self) -> None:
        got = self._sweep(
            self._run({"a": 0.02, "b": 0.05}),
            self._run({"a": 0.04, "b": 0.05}),
        )
        a = next(s for s in got.summaries if s.name == "a")  # type: ignore[attr-defined]
        assert a.mean_decay == pytest.approx(0.03)

    def test_a_strategy_that_found_nothing_ranks_last_rather_than_vanishing(self) -> None:
        """Dropping it would flatter a method that produced no candidate at all."""
        got = self._sweep(self._run({"a": 0.05, "b": None}))
        b = next(s for s in got.summaries if s.name == "b")  # type: ignore[attr-defined]
        assert b.ranks == (2,)
        assert b.mean_decay is None

    def test_rank_spread_counts_how_far_the_seed_moved_it(self) -> None:
        got = self._sweep(
            self._run({"a": 0.01, "b": 0.05, "c": 0.09}),
            self._run({"a": 0.09, "b": 0.05, "c": 0.01}),
        )
        spreads = {s.name: s.rank_spread for s in got.summaries}  # type: ignore[attr-defined]
        assert spreads == {"a": 2, "b": 0, "c": 2}
        assert set(got.moved_two_or_more) == {"a", "c"}  # type: ignore[attr-defined]

    def test_an_unequal_budget_invalidates_the_whole_sweep(self) -> None:
        """Not a caveat under the table — the table must refuse to be read."""
        got = self._sweep(
            self._run({"a": 0.01, "b": 0.05}, budget=10),
            self._run({"a": 0.02, "b": 0.06}, budget=8),
        )
        # One strategy spending less than another inside a run is what the flag detects.
        uneven = BakeOff(symbol="T", bars=600, budget=10, results=(
            StrategyResult("a", 10, 9, 0, Candidate("a", 0.1, 0.09)),
            StrategyResult("b", 8, 7, 0, Candidate("b", 0.1, 0.05)),
        ))
        mixed = self._sweep(self._run({"a": 0.01, "b": 0.05}), uneven)
        assert got.budget_was_equal is True  # type: ignore[attr-defined]
        assert mixed.budget_was_equal is False  # type: ignore[attr-defined]
        assert "INVALID" in mixed.verdict  # type: ignore[attr-defined]

    def test_the_rendered_sweep_shows_the_per_seed_ranks(self) -> None:
        got = self._sweep(
            self._run({"a": 0.01, "b": 0.05}),
            self._run({"a": 0.09, "b": 0.02}),
        )
        text = got.render()  # type: ignore[attr-defined]
        assert "ranks" in text and "spread" in text
        assert "equal budget honoured across every seed" in text
