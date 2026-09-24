"""The execution arena's referee: book walking, the depletion rule, and each arm's children.

The live run replays a converted Tardis day (`ARGUS_TARDIS_NPZ`); pinned here is what the verdict
rests on — a child pays for levels an earlier child of the same parent took until the venue
re-reports them, every arm spends exactly the parent's notional, and the Almgren-Chriss shape
front-loads.
"""

from __future__ import annotations

import pytest

from argus.eval.execution_arena import (
    HORIZON_MIN,
    Level,
    Snapshot,
    ac_fractions,
    execute,
    plans,
    sign_test,
    walk,
)


def _book(ts: int, updated: int = 0, mid: float = 100.0) -> Snapshot:
    asks = {mid + 0.5 + i: Level(10.0, updated) for i in range(5)}
    bids = {mid - 0.5 - i: Level(10.0, updated) for i in range(5)}
    return Snapshot(ts=ts, bids=bids, asks=asks)


def test_a_child_walks_levels_in_price_order() -> None:
    fill = walk(_book(1), "BUY", 100.5 * 10 + 101.5 * 5, {})
    assert fill.shares == pytest.approx(15.0)
    assert fill.mid_at_send == pytest.approx(100.0)


def test_what_a_child_takes_stays_taken_until_the_level_is_re_reported() -> None:
    taken: dict[tuple[str, float], tuple[float, int]] = {}
    walk(_book(1), "BUY", 100.5 * 10, taken)
    second = walk(_book(2), "BUY", 101.5 * 10, taken)  # 100.5 not re-reported: still empty
    assert second.notional / second.shares == pytest.approx(101.5)
    third = walk(_book(3, updated=3), "BUY", 100.5 * 10, taken)  # re-reported: back in play
    assert third.notional / third.shares == pytest.approx(100.5)


def test_an_order_deeper_than_the_book_is_refused() -> None:
    with pytest.raises(ValueError):
        walk(_book(1), "BUY", 1e9, {})


def test_every_arm_spends_the_whole_parent_and_ac_front_loads() -> None:
    for arm, children in plans(25_000.0).items():
        assert sum(n for _, n in children) == pytest.approx(25_000.0), arm
        assert all(0 <= m < HORIZON_MIN for m, _ in children), arm
    shape = ac_fractions(4)
    assert sum(shape) == pytest.approx(1.0)
    assert shape == sorted(shape, reverse=True)


def test_execute_scores_cost_against_the_mid_at_send_plus_fees() -> None:
    snaps = [_book(t) for t in range(HORIZON_MIN + 1)]
    got = execute(snaps, 0, "BUY", [(0, 100.5 * 5)])
    assert got is not None
    assert got["cost_bps"] == pytest.approx(0.5 / 100 * 1e4 + 6.0)
    assert execute(snaps, 0, "BUY", [(HORIZON_MIN + 5, 10.0)]) is None


def test_the_sign_test_is_two_sided() -> None:
    assert sign_test(0, 0) == 1.0
    assert sign_test(10, 0) == pytest.approx(2 / 1024)
