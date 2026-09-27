"""The Agent Hub dry-run preview of an execution plan's first child (`lui/agenthub.py`)."""

from __future__ import annotations

import json
from decimal import Decimal

import pytest

from argus.lui import agenthub


def test_the_child_is_floored_to_the_contract_step() -> None:
    order = agenthub.child_order("NVDAUSDT", "buy", Decimal("50000"), Decimal("224.65"),
                                 "0.01", "0.01")
    assert isinstance(order, agenthub.Order)
    assert order.qty == "222.56"
    assert Decimal(order.qty) * Decimal("224.65") <= Decimal("50000")


def test_a_child_below_the_minimum_or_without_a_step_is_refused_with_the_reason() -> None:
    small = agenthub.child_order("BTCUSDT", "buy", Decimal("5"), Decimal("80000"), "0.0001",
                                 "0.0001")
    assert isinstance(small, str) and "below the contract's minimum of 0.0001" in small
    unknown = agenthub.child_order("BTCUSDT", "buy", Decimal("5000"), Decimal("80000"), None, None)
    assert isinstance(unknown, str) and "size step could not be read" in unknown
    assert agenthub.lines_for(unknown)[0].startswith("Agent Hub preview: not given")


def test_the_command_is_the_request_body_and_never_sends() -> None:
    order = agenthub.Order("TSLAUSDT", "sell", "80.46")
    assert order.command() == ("bgc order --action place --category USDT-FUTURES --symbol "
                               "TSLAUSDT --side sell --orderType market --qty 80.46 "
                               "--paper-trading --dry-run")
    line = agenthub.lines_for(order)[0]
    assert '{"category":"USDT-FUTURES","symbol":"TSLAUSDT","side":"sell"' in line
    assert "--dry-run" in line and "sends nothing" in line


def test_the_mapping_is_the_one_bgc_itself_answered_with() -> None:
    """``data/agenthub_preview.json`` holds bgc 3.0.0's own dry-run replies; the module must
    still build exactly what bgc said it would send, or the record is out of date."""
    record = json.loads(agenthub.RECORD_PATH.read_text(encoding="utf-8"))
    assert record["cli"] == agenthub.CLI and record["all_match"] is True
    for row in record["orders"]:
        assert row["dry_run"] is True and row["path"] == agenthub.PLACE_PATH
        symbol, side, qty = (row["ours"][k] for k in ("symbol", "side", "qty"))
        assert agenthub.Order(symbol, side, qty).would_send() == row["bgc"]


@pytest.mark.parametrize("record", [None, {"all_match": False}])
def test_no_confirmation_is_claimed_without_a_matching_record(
        monkeypatch: pytest.MonkeyPatch, record: dict[str, bool] | None) -> None:
    monkeypatch.setattr(agenthub, "recorded", lambda: record)
    line = agenthub.lines_for(agenthub.Order("NVDAUSDT", "buy", "1"))[0]
    assert "matched bgc" not in line
