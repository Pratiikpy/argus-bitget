"""The second-order stress tree (`desk/stress_tree.py`) and the shock-frequency helpers it reads
from `desk/stress.py`, offline, on a seeded synthetic history."""

from __future__ import annotations

import math
import random
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from argus.desk import stress_tree as st
from argus.desk.stress import (
    MIN_OBSERVATIONS,
    Move,
    StressError,
    closes_from_returns,
    horizon_moves,
    shock_frequency,
)
from argus.eval.research_depth import recompute_tree

START = datetime(2026, 8, 1, tzinfo=UTC)
HOURS = 480


def _is_open(at: datetime) -> bool:
    return at.weekday() < 5 and 14 <= at.hour < 21


def _raw(seed: int = 7, hours: int = HOURS, crash_at: int | None = 300
         ) -> dict[str, dict[datetime, float]]:
    """QQQ plus three names that move with it at different betas, one with a fat idiosyncratic
    day, so every angle has something to find."""
    rng = random.Random(seed)
    stamps = [START + timedelta(hours=i) for i in range(hours)]
    qqq = [rng.gauss(0.0, 0.004) for _ in stamps]
    names = {"NVDAUSDT": (1.6, 0.006), "AAPLUSDT": (0.8, 0.003), "BTCUSDT": (0.5, 0.008)}
    out = {"QQQUSDT": dict(zip(stamps, qqq, strict=True))}
    for symbol, (b, noise) in names.items():
        series = [b * q + rng.gauss(0.0, noise) for q in qqq]
        if crash_at is not None and symbol == "NVDAUSDT":
            for i in range(crash_at, crash_at + 6):
                series[i] -= 0.012
        out[symbol] = dict(zip(stamps, series, strict=True))
    return out


BOOK = {"NVDAUSDT": 0.5, "AAPLUSDT": 0.3, "BTCUSDT": 0.2}


def _inputs(book: dict[str, float] | None = None, shock: float = -10.0,
            raw: dict[str, dict[datetime, float]] | None = None) -> st.StressInputs:
    return st.prepare(raw or _raw(), is_open=_is_open, weights=book or BOOK, shocked="QQQUSDT",
                      shock_pct=shock)


# --- desk/stress.py additions ---------------------------------------------------------------


def test_closes_from_returns_compounds_exactly() -> None:
    at = [START + timedelta(hours=i) for i in range(3)]
    closes = closes_from_returns(list(zip(at, [0.1, -0.5, 0.2], strict=True)))
    assert [c for _, c in closes] == [Decimal("110.0"), Decimal("55.00"), Decimal("66.000")]
    assert [t for t, _ in closes] == at  # no level is dated before the first return


def test_shock_frequency_counts_in_the_shock_direction() -> None:
    moves = [Move(at=START + timedelta(hours=i), pct=Decimal(p), phase="x")
             for i, p in enumerate(["-3", "-1", "2", "0.5"] * 10)]
    down = shock_frequency(moves, shock_pct=Decimal("-1"), horizon_bars=24)
    assert (down.occurrences, down.observations, down.extreme_pct) == (20, 40, Decimal("-3"))
    up = shock_frequency(moves, shock_pct=Decimal("5"), horizon_bars=24)
    assert up.beyond_history and up.extreme_pct == Decimal("2") and up.last_seen is None
    assert down.last_seen == START + timedelta(hours=37)


def test_shock_frequency_refuses_what_it_cannot_support() -> None:
    few = [Move(at=START, pct=Decimal("-1"), phase="x")] * (MIN_OBSERVATIONS - 1)
    with pytest.raises(StressError):
        shock_frequency(few, shock_pct=Decimal("-1"), horizon_bars=24)
    enough = [*few, Move(at=START, pct=Decimal("-1"), phase="x")]
    with pytest.raises(StressError):
        shock_frequency(enough, shock_pct=Decimal("0"), horizon_bars=24)


# --- the tree ----------------------------------------------------------------------------------


def test_prepare_refuses_what_it_cannot_grow_from() -> None:
    with pytest.raises(st.StressTreeError):
        _inputs(shock=0.0)
    with pytest.raises(st.StressTreeError):
        st.prepare(_raw(), is_open=_is_open, weights=BOOK, shocked="ETHUSDT", shock_pct=-5)
    with pytest.raises(st.StressTreeError):
        st.prepare(_raw(), is_open=_is_open, weights={"SOLUSDT": 1.0}, shocked="QQQUSDT",
                   shock_pct=-5)


def test_every_first_order_angle_answers_and_children_grow_from_results() -> None:
    tree = st.grow(_inputs())
    roots = [n for n in tree.nodes if n.depth == 1]
    assert [n.angle for n in roots] == list(st.ROOT_ANGLES)
    assert all(n.status == "answered" for n in roots)
    by_id = {n.id: n for n in tree.nodes}
    # beta is well above the hedge floor, so the hedge is tested through the realised window and
    # then through the tail; the holding the shock costs most is examined on its own worst day
    assert by_id["1.1"].angle == "hedge_residual" and by_id["1.1.1"].angle == "hedged_tail"
    assert by_id["1.2"].angle == "idiosyncratic" and by_id["1.2"].subject == "NVDAUSDT"
    assert by_id["2.1"].angle == "window_driver"
    # QQQ never fell 10% in a day in this history: beyond it, and re-run at what did happen
    assert "beyond everything in this history" in by_id["3"].text
    assert by_id["3.1"].angle == "history_shock"
    assert tree.levels == 3 and not tree.stops


def test_figures_agree_with_an_independent_recomputation() -> None:
    raw = _raw()
    tree = st.grow(st.prepare(raw, is_open=_is_open, weights=BOOK, shocked="QQQUSDT",
                              shock_pct=-10.0))
    checks = recompute_tree(tree, raw, _is_open)
    assert {c["node"] for c in checks} == {n.id for n in tree.answered}  # every node, checked
    assert len(checks) >= 2 * len(tree.answered) - 1
    assert [c for c in checks if not c["ok"]] == []


def test_rendered_text_states_the_computed_figures() -> None:
    tree = st.grow(_inputs())
    node = next(n for n in tree.answered if n.angle == "beta_shock")
    assert f"{node.figures['book_move_pct']:+.2f}%" in node.text
    window = next(n for n in tree.answered if n.angle == "realised_window")
    assert f"{window.figures['move_pct']:+.2f}%" in window.text
    lines = tree.render()
    assert lines[0].startswith("Scenario tree: ")
    assert any(line.startswith("  1.1. ") for line in lines)  # children indented under parents


def test_depth_and_breadth_bound_the_tree() -> None:
    shallow = st.grow(_inputs(), depth=1)
    assert {n.depth for n in shallow.nodes} == {1}
    narrow = st.grow(_inputs(), breadth=2)
    assert [n.angle for n in narrow.nodes if n.depth == 1] == ["beta_shock", "realised_window"]
    # the next level runs max(2, 2 // 2) = 2 children per parent at most
    per_parent: dict[str | None, int] = {}
    for n in narrow.nodes:
        per_parent[n.parent] = per_parent.get(n.parent, 0) + 1
    assert all(count <= 2 for parent, count in per_parent.items() if parent is not None)


def test_the_node_budget_names_what_it_did_not_run() -> None:
    tree = st.grow(_inputs(), max_nodes=6)
    skipped = [n for n in tree.nodes if n.status == "skipped"]
    assert len([n for n in tree.nodes if n.status != "skipped"]) == 6
    assert skipped and all("6-scenario budget" in n.reason for n in skipped)
    assert any("Not computed" in line for line in tree.render())


def test_one_failing_scenario_does_not_take_its_siblings(monkeypatch: pytest.MonkeyPatch
                                                         ) -> None:
    def broken(inputs: st.StressInputs, probe: st.Probe) -> tuple[str, dict[str, float]]:
        raise ZeroDivisionError("synthetic")

    angle = st.CATALOGUE["tail"]
    monkeypatch.setitem(st.CATALOGUE, "tail", st.Angle(angle.title, broken, angle.follow_ups,
                                                       angle.engine))
    tree = st.grow(_inputs())
    tail = next(n for n in tree.nodes if n.angle == "tail")
    assert tail.status == "failed" and "ZeroDivisionError" in tail.reason
    assert sum(1 for n in tree.nodes if n.depth == 1 and n.status == "answered") == 4
    assert not any(n.angle == "tail_trim" for n in tree.nodes)


def test_a_level_where_nothing_answers_stops_the_descent(monkeypatch: pytest.MonkeyPatch
                                                         ) -> None:
    def broken(inputs: st.StressInputs, probe: st.Probe) -> tuple[str, dict[str, float]]:
        raise RuntimeError("down")

    for name in st.ROOT_ANGLES:
        angle = st.CATALOGUE[name]
        monkeypatch.setitem(st.CATALOGUE, name, st.Angle(angle.title, broken, angle.follow_ups,
                                                         angle.engine))
    tree = st.grow(_inputs())
    assert len(tree.nodes) == len(st.ROOT_ANGLES) and not tree.answered
    assert any("#1579" in stop for stop in tree.stops)


def test_the_same_book_always_prints_the_same_tree() -> None:
    inputs = _inputs()
    assert st.grow(inputs, concurrency=1).render() == st.grow(inputs, concurrency=4).render()


def test_a_single_name_book_grows_no_arithmetic_only_scenarios() -> None:
    tree = st.grow(_inputs(book={"NVDAUSDT": 1.0}))
    angles = {n.angle for n in tree.nodes}
    assert "window_driver" not in angles and "tail_trim" not in angles
    assert "beta_shock" in angles


def test_the_shocked_instrument_is_never_its_own_idiosyncratic_scenario() -> None:
    raw = _raw()
    tree = st.grow(st.prepare(raw, is_open=_is_open, weights={"BTCUSDT": 0.6, "AAPLUSDT": 0.4},
                              shocked="BTCUSDT", shock_pct=-20.0))
    assert all(n.subject != "BTCUSDT" for n in tree.nodes if n.angle == "idiosyncratic")


def test_cash_dilutes_every_figure() -> None:
    full = st.grow(_inputs(book=BOOK))
    half = st.grow(_inputs(book={s: w / 2 for s, w in BOOK.items()}))
    move = {n.angle: n.figures for n in full.answered}["beta_shock"]["book_move_pct"]
    halved = {n.angle: n.figures for n in half.answered}["beta_shock"]["book_move_pct"]
    assert math.isclose(halved, move / 2, rel_tol=1e-9)


def test_research_lines_is_the_console_call() -> None:
    lines, sources, payload = st.research_lines(raw=_raw(), is_open=_is_open, book=BOOK,
                                                shocked="QQQUSDT", shock_pct=None,
                                                provenance="synthetic")
    assert payload["grown"] and payload["shock_pct"] == -10.0  # the default shock, stated
    assert lines[0].startswith("Scenario tree:") and "QQQ -10%" in lines[0]
    assert sources and sources[0].ref == "argus.desk.stress_tree"
    lines, sources, payload = st.research_lines(raw=_raw(), is_open=_is_open,
                                                book={"SOLUSDT": 1.0}, shocked="QQQUSDT",
                                                shock_pct=-5)
    assert lines == [] and sources == [] and not payload["grown"] and payload["reason"]


def test_a_session_gap_grows_the_shut_session_shock() -> None:
    rng = random.Random(11)
    stamps = [START + timedelta(hours=i) for i in range(HOURS)]
    qqq = [rng.gauss(0.0, 0.004) for _ in stamps]
    nvda = [(2.0 if _is_open(t) else 0.3) * q + rng.gauss(0.0, 0.002)
            for t, q in zip(stamps, qqq, strict=True)]
    raw = {"QQQUSDT": dict(zip(stamps, qqq, strict=True)),
           "NVDAUSDT": dict(zip(stamps, nvda, strict=True))}
    tree = st.grow(st.prepare(raw, is_open=_is_open, weights={"NVDAUSDT": 1.0},
                              shocked="QQQUSDT", shock_pct=-5.0))
    gap = next(n for n in tree.answered if n.angle == "session_gap")
    assert gap.figures["open_beta"] > 1.5 > 0.6 > gap.figures["shut_beta"]
    shut = next(n for n in tree.answered if n.angle == "shut_shock")
    assert shut.parent == gap.id
    assert shut.figures["shut_move_pct"] > shut.figures["open_move_pct"]
    assert [c for c in recompute_tree(tree, raw, _is_open) if not c["ok"]] == []


def test_a_history_that_did_see_the_shock_says_how_often() -> None:
    raw = _raw()
    tree = st.grow(st.prepare(raw, is_open=_is_open, weights=BOOK, shocked="QQQUSDT",
                              shock_pct=-0.5))
    node = next(n for n in tree.answered if n.angle == "shock_frequency")
    closes = closes_from_returns(sorted(raw["QQQUSDT"].items()))
    moves = horizon_moves(closes, bars=24)
    expected = sum(1 for m in moves if m.pct <= Decimal("-0.5"))
    assert node.figures["occurrences"] == expected > 0
    assert "most recently" in node.text


# --- the one-pass realised-window line, on the book as stated -----------------------------------


def test_the_book_worst_window_replays_the_weights_the_trader_gave() -> None:
    from argus.desk.portfolio import align, rebalance, worst_window
    from argus.eval.research_depth import Frame, np_worst

    raw = _raw()
    book = {"QQQUSDT": 0.5, "NVDAUSDT": 0.5}
    stated = st.book_worst_window(raw, book)
    assert stated.move_pct is not None
    independent, start = np_worst(book, Frame.build(raw, _is_open).cols)
    assert math.isclose(stated.move_pct, independent, rel_tol=1e-9)
    assert stated.start_index == start
    # the tree's own realised_window node reads the same figure, so one answer never shows two
    tree = st.grow(st.prepare(raw, is_open=_is_open, weights=book, shocked="QQQUSDT",
                              shock_pct=-10.0))
    node = next(n for n in tree.answered if n.angle == "realised_window")
    assert math.isclose(node.figures["move_pct"], stated.move_pct, rel_tol=1e-12)
    # the one-pass call's weights (the rest scaled by 1 - size) are a different book entirely
    _, columns = align(raw)
    used = rebalance({"NVDAUSDT": 0.5}, "QQQUSDT", 0.5)
    assert used == pytest.approx({"NVDAUSDT": 0.25, "QQQUSDT": 0.5})
    rebalanced = worst_window(weights=used, columns=columns).move_pct
    assert rebalanced is not None and not math.isclose(rebalanced, stated.move_pct, rel_tol=1e-3)


def test_the_book_worst_window_ignores_zero_weights_and_names_with_no_history() -> None:
    raw = _raw()
    full = st.book_worst_window(raw, {"NVDAUSDT": 0.6, "AAPLUSDT": 0.4})
    padded = st.book_worst_window(raw, {"NVDAUSDT": 0.6, "AAPLUSDT": 0.4, "BTCUSDT": 0.0,
                                        "NOPEUSDT": 0.1})
    assert full.move_pct is not None and full.move_pct == padded.move_pct
