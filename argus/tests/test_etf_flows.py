"""Spot ETF flows and Strategy's buys: creations, not price moves, and always dated."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any

import pytest

from argus.market import etf_flows as ef

NOW = datetime(2026, 9, 25, 12, tzinfo=UTC)


def _rows(values: list[float]) -> list[dict[str, Any]]:
    return [{"date": f"2026-08-{i + 1:02d}" if i < 31 else f"2026-09-{i - 30:02d}",
             "total_net_inflow": v, "total_net_assets": 1.0e11, "cum_net_inflow": 5.0e10}
            for i, v in enumerate(values)]


def test_the_latest_day_is_scored_against_its_own_month() -> None:
    flows = [10e6] * 29 + [-5e6, 1.2e9]
    out = ef.fund_flows("BTC", _rows(flows))
    assert out["net_inflow_usd"] == 1.2e9
    assert out["five_day_usd"] == pytest.approx(10e6 * 3 - 5e6 + 1.2e9)
    assert out["z_latest"] is not None and out["z_latest"] > 3
    assert out["streak_days"] == 1  # the day before was an outflow


def test_an_inflow_streak_is_counted() -> None:
    out = ef.fund_flows("ETH", _rows([-1e6] * 10 + [2e6, 3e6, 4e6, 5e6]))
    assert out["streak_days"] == 4


def test_no_rows_is_an_error_not_a_quiet_day() -> None:
    with pytest.raises(ef.FlowError):
        ef.fund_flows("BTC", [])


def _snapshot(fetch: Any) -> dict[str, Any]:
    return ef.sweep(get=fetch, now=NOW)


def _fake(path: str, query: Any) -> Any:
    if path == "/etfs/summary-history":
        return [{"date": "2026-09-23", "total_net_inflow": 346977216.17,
                 "total_net_assets": 108663294328.4, "cum_net_inflow": 5.7e10},
                {"date": "2026-09-22", "total_net_inflow": 714748985.5,
                 "total_net_assets": 1.1e11, "cum_net_inflow": 5.6e10}]
    return [{"date": "2026-09-21", "ticker": "MSTR", "btc_holding": "846000", "btc_acq": "950",
             "acq_cost": "75700000"}]


def test_the_console_lines_for_mstr_carry_flows_and_the_latest_buy() -> None:
    lines = ef.lines_for("MSTRUSDT", _snapshot(_fake), today=date(2026, 9, 25))
    assert lines[0].startswith("US spot BTC ETFs, 23 Sep: net +$347m")
    assert "creations less redemptions" in lines[0] and "stale" not in lines[0]
    assert lines[1].startswith("Strategy last bought 950 BTC on 2026-09-21 ($76m)")


def test_an_old_snapshot_says_stale_and_unlinked_names_get_nothing() -> None:
    snapshot = _snapshot(_fake)
    assert "stale" in ef.lines_for("BTCUSDT", snapshot, today=date(2026, 10, 5))[0]
    assert ef.lines_for("NVDAUSDT", snapshot) == []
    assert ef.lines_for("BTCUSDT", None) == []


def test_a_failing_endpoint_is_recorded_not_hidden() -> None:
    def broken(path: str, query: Any) -> Any:
        raise ef.FlowError("SoSoValue /etfs/summary-history unreachable: HTTP 403")

    snapshot = _snapshot(broken)
    assert snapshot["funds"] == {} and len(snapshot["errors"]) == 3
    assert ef.lines_for("BTCUSDT", snapshot) == []
