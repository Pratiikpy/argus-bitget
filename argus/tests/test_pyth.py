"""Pyth Pro (`market/pyth.py`) and its place in the overnight comparison (build-list 4.7).
Offline: the price rows are the shape Pyth's own MCP validates (``apps/mcp/src/clients/types.ts``,
its test fixture ``tests/clients/history.test.ts``); the live symbol list is a network test."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from argus.eval import overnight_comparison as oc
from argus.market import pyth


def test_a_price_row_is_scaled_by_its_exponent_and_dated() -> None:
    # Pyth's MCP fixture: 5_100_000_000_000 at exponent -8 is 51,000, published 2024-02-19
    rows = pyth.parse_prices([{"price_feed_id": 1, "price": 5_100_000_000_000, "exponent": -8,
                               "publish_time": 1_708_300_800, "channel": "fixed_rate@200ms",
                               "publisher_count": 9, "market_session": "over_night"},
                              {"price_feed_id": 2, "price": None, "publish_time": 1}])
    assert set(rows) == {1}
    assert rows[1].price == pytest.approx(51_000.0)
    assert rows[1].published == datetime(2024, 2, 19, tzinfo=UTC)
    assert rows[1].session == "over_night" and rows[1].publishers == 9


@pytest.mark.parametrize("stamp", [1_708_300_800, 1_708_300_800_000, 1_708_300_800_000_000])
def test_publish_time_is_read_in_any_of_its_units(stamp: int) -> None:
    assert pyth._moment(stamp) == datetime(2024, 2, 19, tzinfo=UTC)


def test_the_us_listing_is_chosen_over_an_index_or_a_token() -> None:
    feeds = [pyth.parse_feed({"pyth_lazer_id": 7, "symbol": "Equity.Index.NVDA/USD"}),
             pyth.parse_feed({"pyth_lazer_id": 1314, "symbol": "Equity.US.NVDA/USD",
                              "market_sessions": {"regular": {}, "over_night": {}}})]
    found = pyth.equity_feed("nvda", feeds)
    assert found is not None and found.lazer_id == 1314
    assert found.sessions == ("regular", "over_night")
    assert pyth.equity_feed("AAPL", feeds) is None


def test_no_key_means_no_price_read(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("PYTH_PRO_API_KEY", raising=False)
    monkeypatch.setattr(pyth, "SECRETS", tmp_path / "absent.env")
    with pytest.raises(pyth.NoKey, match="Pyth Terminal"):
        pyth.prices_at([1314], datetime(2026, 9, 24, 7, 30, tzinfo=UTC))
    (tmp_path / "pyth.env").write_text("OTHER=1\nPYTH_PRO_API_KEY='abc'\n", encoding="utf-8")
    monkeypatch.setattr(pyth, "SECRETS", tmp_path / "pyth.env")
    assert pyth.key() == "abc"


def test_a_stale_pyth_print_is_not_a_quote() -> None:
    moment = 1_000_000.0
    fresh = [[moment, 101.0, moment - 600, "over_night"]]
    stale = [[moment, 101.0, moment - 3 * 3600, "over_night"]]
    future = [[moment, 101.0, moment + 60, "over_night"]]
    assert oc.pyth_at(fresh, moment) == 101.0
    assert oc.pyth_at(stale, moment) is None
    assert oc.pyth_at(future, moment) is None
    assert oc.pyth_at(fresh, moment + 1) is None


def test_the_block_scores_only_rows_pyth_quoted() -> None:
    def row(i: int, gap: float, ours: float, theirs: float | None) -> dict[str, object]:
        estimates = {"argus_perp_0330": ours, "zero": 0.0}
        if theirs is not None:
            estimates["pyth_0330"] = theirs
        return {"stock": "NVDA", "opened": f"2026-09-{1 + i % 28:02d}", "gap": gap,
                "estimates": estimates}

    rows = [row(i, 0.01, 0.009, 0.0101) for i in range(24)] + [row(99, 0.02, 0.01, None)]
    block = oc.pyth_block([], rows)
    assert block["at_0330"]["rows"] == 24
    assert block["at_0330"]["rows_without_a_pyth_quote"] == 1
    summary = block["at_0330"]["summary"]
    assert summary["pyth_0330"]["mae_bps"] < summary["argus_perp_0330"]["mae_bps"]
    assert block["at_0900"]["rows"] == 0 and "summary" not in block["at_0900"]


@pytest.mark.network
def test_the_live_symbol_list_carries_the_overnight_session() -> None:
    feed = pyth.equity_feed("NVDA")
    assert feed is not None and "over_night" in feed.sessions
