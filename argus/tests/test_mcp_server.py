"""ARGUS over the Model Context Protocol: handshake, tool list, and tool errors as results."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import pytest

from argus.lui import mcp_server as mcp
from argus.lui import research
from argus.market import history, universe


@pytest.fixture(autouse=True)
def frozen(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fail(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("live fetch disabled in tests")

    monkeypatch.setattr(research, "_fetch_live", _fail)
    monkeypatch.setattr(history, "fetch", _fail)
    monkeypatch.setattr(universe, "_fetch_live", _fail)
    monkeypatch.setattr(universe, "_CACHE", None)


def _rpc(method: str, params: Mapping[str, Any] | None = None, **kw: Any) -> dict[str, Any]:
    reply = mcp.handle({"jsonrpc": "2.0", "id": 7, "method": method, "params": params or {}}, **kw)
    assert reply is not None
    return reply


def test_initialize_negotiates_a_known_version_and_declares_tools() -> None:
    result = _rpc("initialize", {"protocolVersion": "2025-03-26"})["result"]
    assert result["protocolVersion"] == "2025-03-26"
    assert "tools" in result["capabilities"]
    assert result["serverInfo"]["name"] == "argus-research-desk"
    unknown = _rpc("initialize", {"protocolVersion": "1999-01-01"})["result"]
    assert unknown["protocolVersion"] == mcp.PROTOCOL_VERSIONS[0]


def test_a_notification_gets_no_response() -> None:
    assert mcp.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    status, body = mcp.handle_body(b'{"jsonrpc": "2.0", "method": "notifications/initialized"}')
    assert status == 202 and body == b""


def test_every_tool_has_a_schema() -> None:
    tools = _rpc("tools/list")["result"]["tools"]
    assert {t["name"] for t in tools} == {
        "argus_ask", "argus_quote", "argus_portfolio_impact", "argus_stress",
        "argus_execution_plan", "argus_scoreboard"}
    for tool in tools:
        assert tool["inputSchema"]["type"] == "object" and tool["description"]


def test_a_tool_failure_is_a_result_with_is_error_not_a_transport_error() -> None:
    def broken(name: str, args: Mapping[str, Any]) -> tuple[str, bool]:
        raise mcp.ToolError("'ZYXQ' is not a contract Bitget lists")

    reply = _rpc("tools/call", {"name": "argus_quote", "arguments": {"symbols": ["ZYXQ"]}},
                 tool=broken)
    assert "error" not in reply
    assert reply["result"]["isError"] is True
    assert "ZYXQ" in reply["result"]["content"][0]["text"]


def test_unknown_tool_and_unknown_method_are_json_rpc_errors() -> None:
    assert _rpc("tools/call", {"name": "delete_everything"})["error"]["code"] == -32602
    assert _rpc("resources/list")["error"]["code"] == -32601
    status, body = mcp.handle_body(b"not json")
    assert status == 400 and json.loads(body)["error"]["code"] == -32700


def test_a_book_is_validated_and_normalised() -> None:
    assert mcp._book({"NVDA": 60, "AAPL": 20}) == pytest.approx(
        {"NVDAUSDT": 0.75, "AAPLUSDT": 0.25})
    with pytest.raises(mcp.ToolError):
        mcp._book({"NVDA": -5})
    with pytest.raises(mcp.ToolError):
        mcp._symbol("ZYXQWV")


def test_the_stress_tool_runs_the_desks_engine_on_a_named_shock() -> None:
    text, is_error = mcp.call_tool("argus_stress", {
        "book": {"NVDA": 50, "COIN": 50}, "shock_percent": -30, "shocked": "MSTR"})
    assert not is_error
    assert "If MSTR moves -30%" in text and "Sources:" in text


def test_the_scoreboard_reads_the_register() -> None:
    text, is_error = mcp.call_tool("argus_scoreboard", {})
    assert not is_error and "States:" in text and "OWNED" in text


def test_a_batch_is_answered_as_a_batch() -> None:
    status, body = mcp.handle_body(json.dumps([
        {"jsonrpc": "2.0", "id": 1, "method": "ping"},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}]).encode())
    replies = json.loads(body)
    assert status == 200 and [r["id"] for r in replies] == [1, 2]
