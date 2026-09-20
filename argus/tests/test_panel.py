"""Cross-sectional evaluation: aligned by time, resolved innermost-first, and no look-ahead."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pytest

from argus.research.grammar import (
    Const,
    CrossRank,
    CrossScale,
    Field,
    GrammarError,
    Ref,
    Signal,
    UnOp,
)
from argus.research.panel import (
    MIN_UNIVERSE,
    PanelError,
    build_panel,
    cross_sectional_nodes,
    evaluate_panel,
)

START = datetime(2026, 9, 1, tzinfo=UTC)


@dataclass(frozen=True)
class Bar:
    ts: datetime
    close: float
    extra: dict[str, float]


def _series(closes: list[float], *, skip: set[int] | None = None) -> list[Bar]:
    skip = skip or set()
    return [
        Bar(ts=START + timedelta(hours=i), close=c, extra={"volume": c * 10})
        for i, c in enumerate(closes)
        if i not in skip
    ]


def _panel(**series: list[float]):  # type: ignore[no-untyped-def]
    return build_panel({k: _series(v) for k, v in series.items()})


CLOSE = Ref(field=Field.CLOSE)


class TestAlignmentIsByTimestamp:
    def test_only_shared_timestamps_survive(self) -> None:
        panel = build_panel({
            "A": _series([1, 2, 3, 4]),
            "B": _series([1, 2, 3, 4], skip={2}),
        })
        assert panel.length == 3

    def test_the_dropped_bars_are_counted_not_hidden(self) -> None:
        panel = build_panel({
            "A": _series([1, 2, 3, 4]),
            "B": _series([1, 2, 3, 4], skip={2}),
        })
        assert panel.dropped["A"] == 1 and panel.total_dropped == 1
        assert "dropped as unshared" in panel.render()

    def test_a_gap_does_not_shift_one_series_against_another(self) -> None:
        """The bug this module exists to prevent: index 2 of A compared with index 2 of B when
        they are different hours."""
        panel = build_panel({
            "A": _series([10, 20, 30, 40]),
            "B": _series([10, 20, 30, 40], skip={1}),
        })
        for i, ts in enumerate(panel.timestamps):
            assert panel.bars["A"][i].ts == ts == panel.bars["B"][i].ts

    def test_series_that_never_overlap_are_refused(self) -> None:
        late = [Bar(ts=START + timedelta(days=30, hours=i), close=1.0, extra={})
                for i in range(3)]
        with pytest.raises(PanelError, match="do not overlap"):
            build_panel({"A": _series([1, 2, 3]), "B": late})

    def test_a_duplicate_timestamp_keeps_the_later_bar(self) -> None:
        """A venue restating a candle sends the corrected value second."""
        bars = [Bar(ts=START, close=1.0, extra={}), Bar(ts=START, close=9.0, extra={})]
        panel = build_panel({"A": bars, "B": [Bar(ts=START, close=5.0, extra={})]})
        assert panel.bars["A"][0].close == 9.0

    def test_an_empty_symbol_is_refused(self) -> None:
        with pytest.raises(PanelError, match="carries no bars"):
            build_panel({"A": _series([1, 2]), "B": []})

    def test_a_bar_without_a_timestamp_is_refused_by_name(self) -> None:
        class NoTs:
            close = 1.0

        with pytest.raises(PanelError, match="no `ts`"):
            build_panel({"A": [NoTs()]})

    def test_an_empty_panel_is_refused(self) -> None:
        with pytest.raises(PanelError, match="at least one symbol"):
            build_panel({})


class TestASingleSymbolCannotEvaluateACrossSectionalNode:
    def test_it_raises_rather_than_returning_a_neutral_value(self) -> None:
        bars = _series([1, 2, 3])
        with pytest.raises(GrammarError, match="cannot be evaluated from one symbol"):
            CrossRank(operand=CLOSE).evaluate(bars, 2)

    def test_the_refusal_says_where_to_go_instead(self) -> None:
        with pytest.raises(GrammarError, match=r"argus.research.panel"):
            CrossScale(operand=CLOSE).evaluate(_series([1, 2]), 1)

    def test_a_tree_reports_that_it_needs_a_panel(self) -> None:
        nested = Signal(operand=UnOp(op="neg", operand=CrossRank(operand=CLOSE)))
        assert nested.needs_panel

    def test_an_ordinary_tree_does_not(self) -> None:
        assert not Signal(operand=CLOSE).needs_panel

    def test_only_the_cross_sectional_node_itself_is_marked(self) -> None:
        node = CrossRank(operand=CLOSE)
        assert node.is_cross_sectional and not CLOSE.is_cross_sectional

    def test_a_one_symbol_universe_is_refused_not_silently_constant(self) -> None:
        panel = _panel(A=[1, 2, 3])
        with pytest.raises(PanelError, match="rank against oneself"):
            evaluate_panel(CrossRank(operand=CLOSE), panel)

    def test_the_minimum_universe_is_two(self) -> None:
        assert MIN_UNIVERSE == 2


class TestRankingAcrossTheUniverse:
    def test_the_highest_name_ranks_one_and_the_lowest_zero(self) -> None:
        panel = _panel(A=[1, 1, 1], B=[2, 2, 2], C=[3, 3, 3])
        got = evaluate_panel(CrossRank(operand=CLOSE), panel)
        assert got["A"][0] == 0.0 and got["C"][0] == 1.0

    def test_the_middle_name_ranks_in_the_middle(self) -> None:
        panel = _panel(A=[1, 1], B=[2, 2], C=[3, 3])
        assert evaluate_panel(CrossRank(operand=CLOSE), panel)["B"][0] == 0.5

    def test_ties_take_the_midrank(self) -> None:
        panel = _panel(A=[1, 1], B=[1, 1], C=[3, 3])
        got = evaluate_panel(CrossRank(operand=CLOSE), panel)
        assert got["A"][0] == got["B"][0] == 0.25

    def test_an_entirely_flat_universe_ranks_everyone_equally(self) -> None:
        panel = _panel(A=[5, 5], B=[5, 5], C=[5, 5])
        got = evaluate_panel(CrossRank(operand=CLOSE), panel)
        assert got["A"][0] == got["B"][0] == got["C"][0] == 0.5

    def test_the_rank_changes_with_the_instant(self) -> None:
        """The whole point: A leads at index 0 and trails at index 1."""
        panel = _panel(A=[9, 1], B=[1, 9])
        got = evaluate_panel(CrossRank(operand=CLOSE), panel)
        assert got["A"] == (1.0, 0.0) and got["B"] == (0.0, 1.0)

    def test_ranks_are_computed_on_the_operand_not_the_raw_close(self) -> None:
        negated = CrossRank(operand=UnOp(op="neg", operand=CLOSE))
        panel = _panel(A=[1, 1], B=[3, 3])
        got = evaluate_panel(negated, panel)
        assert got["A"][0] == 1.0 and got["B"][0] == 0.0

    def test_ranking_a_signal_is_refused(self) -> None:
        with pytest.raises(GrammarError, match="not meaningful"):
            CrossRank(operand=Signal(operand=CLOSE))


class TestScalingAcrossTheUniverse:
    def test_absolute_values_sum_to_one(self) -> None:
        panel = _panel(A=[1, 1], B=[3, 3])
        got = evaluate_panel(CrossScale(operand=CLOSE), panel)
        assert got["A"][0] + got["B"][0] == pytest.approx(1.0)

    def test_the_sign_is_preserved(self) -> None:
        scaled = CrossScale(operand=UnOp(op="neg", operand=CLOSE))
        panel = _panel(A=[1, 1], B=[3, 3])
        got = evaluate_panel(scaled, panel)
        assert got["A"][0] < 0 and got["B"][0] < 0

    def test_an_all_zero_universe_scales_to_zero_and_does_not_divide_by_it(self) -> None:
        scaled = CrossScale(operand=Const(value=0.0))
        panel = _panel(A=[1, 1], B=[3, 3])
        got = evaluate_panel(scaled, panel)
        assert got["A"][0] == 0.0 and got["B"][0] == 0.0

    def test_scaling_a_signal_is_refused(self) -> None:
        with pytest.raises(GrammarError, match="not meaningful"):
            CrossScale(operand=Signal(operand=CLOSE))


class TestNestingResolvesInnermostFirst:
    def test_a_rank_of_a_rank_evaluates(self) -> None:
        nested = CrossRank(operand=CrossRank(operand=CLOSE))
        panel = _panel(A=[1, 1], B=[2, 2], C=[3, 3])
        got = evaluate_panel(nested, panel)
        assert got["C"][0] == 1.0 and got["A"][0] == 0.0

    def test_nodes_are_ordered_innermost_first(self) -> None:
        inner = CrossRank(operand=CLOSE)
        outer = CrossScale(operand=inner)
        nodes = cross_sectional_nodes(outer)
        assert nodes[0].canonical() == inner.canonical()

    def test_the_same_subexpression_twice_is_one_computation(self) -> None:
        """Two spellings of one factor must not consume two trials, cross-sectional included."""
        from argus.research.grammar import BinOp

        rank = CrossRank(operand=CLOSE)
        both = BinOp(op="add", left=rank, right=CrossRank(operand=CLOSE))
        assert len(cross_sectional_nodes(both)) == 1

    def test_an_expression_with_no_cross_sectional_node_still_evaluates(self) -> None:
        panel = _panel(A=[1, 2], B=[3, 4])
        got = evaluate_panel(Signal(operand=Const(value=0.5)), panel)
        assert got["A"] == (0.5, 0.5)

    def test_a_single_symbol_panel_is_fine_without_cross_sectional_nodes(self) -> None:
        got = evaluate_panel(Signal(operand=Const(value=0.25)), _panel(A=[1, 2, 3]))
        assert got["A"] == (0.25, 0.25, 0.25)


class TestNoLookAhead:
    def test_index_i_sees_only_index_i_of_the_universe(self) -> None:
        """Ranking over the full sample and indexing back into it is the standard invisible leak.
        A's close is below B's for the first half and above it for the second; the rank at each
        index must follow that, never the full-sample ordering."""
        panel = _panel(A=[1, 1, 9, 9], B=[5, 5, 5, 5])
        got = evaluate_panel(CrossRank(operand=CLOSE), panel)
        assert got["A"] == (0.0, 0.0, 1.0, 1.0)

    def test_truncating_the_future_does_not_change_the_past(self) -> None:
        """The decisive test. If any future bar influenced an earlier value, cutting the series
        short would move it."""
        full = evaluate_panel(CrossRank(operand=CLOSE), _panel(A=[1, 4, 9, 2], B=[5, 5, 5, 5]))
        short = evaluate_panel(CrossRank(operand=CLOSE), _panel(A=[1, 4], B=[5, 5]))
        assert short["A"] == full["A"][:2]

    def test_the_same_holds_for_scale(self) -> None:
        full = evaluate_panel(CrossScale(operand=CLOSE), _panel(A=[1, 4, 9], B=[5, 5, 5]))
        short = evaluate_panel(CrossScale(operand=CLOSE), _panel(A=[1, 4], B=[5, 5]))
        assert short["A"] == pytest.approx(full["A"][:2])

    def test_the_same_holds_when_nested(self) -> None:
        nested = CrossScale(operand=CrossRank(operand=CLOSE))
        full = evaluate_panel(nested, _panel(A=[1, 4, 9], B=[5, 5, 5], C=[2, 2, 2]))
        short = evaluate_panel(nested, _panel(A=[1, 4], B=[5, 5], C=[2, 2]))
        assert short["A"] == pytest.approx(full["A"][:2])


class TestThePanelDescribesItself:
    def test_it_reports_its_width_and_length(self) -> None:
        panel = _panel(A=[1, 2, 3], B=[1, 2, 3])
        assert panel.width == 2 and panel.length == 3
        assert "2 symbol(s) aligned on 3" in panel.render()

    def test_symbols_come_back_sorted_so_output_is_deterministic(self) -> None:
        panel = build_panel({"Z": _series([1, 2]), "A": _series([1, 2])})
        assert panel.symbols == ("A", "Z")

    def test_it_serialises_with_its_window(self) -> None:
        got = _panel(A=[1, 2], B=[3, 4]).as_dict()
        assert got["length"] == 2 and got["first"] and got["last"]
