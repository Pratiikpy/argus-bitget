"""ARGUS over the Model Context Protocol: handshake, tool list, and tool errors as results."""

from __future__ import annotations

import json
from collections.abc import Mapping
from types import SimpleNamespace
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
        "argus_execution_plan", "argus_scoreboard", "argus_research_task", "argus_week_ahead",
        "argus_exposures", "argus_review_trades"}
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


# --- hardening, 2026-09-25: the fuzz corpus, schema invariance, and the engine path ----------


def test_the_whole_fuzz_corpus_is_clean_with_zero_unstructured_exceptions() -> None:
    from argus.eval import mcp_fuzz

    result = mcp_fuzz.fuzz(mcp_fuzz.hand_rolled_transport, mcp.TOOLS)
    assert result["cases"] >= 145
    assert result["unstructured_exceptions"] == 0
    assert result["failures"] == []


def test_every_published_schema_is_unchanged() -> None:
    """The hardening validates against the schemas; it must not have edited them."""
    by_name = {t["name"]: t["inputSchema"] for t in mcp.TOOLS}
    assert set(by_name["argus_ask"]["properties"]) == {"question", "book", "memory"}
    assert by_name["argus_ask"]["required"] == ["question"]
    assert by_name["argus_quote"]["properties"]["symbols"] == {
        "type": "array", "items": {"type": "string"},
        "description": "Tickers or contracts, e.g. ['NVDA', 'BTCUSDT']."}
    assert by_name["argus_execution_plan"]["required"] == ["symbol", "usd"]
    assert by_name["argus_scoreboard"] == {"type": "object", "properties": {}}
    assert "additionalProperties" not in by_name["argus_stress"]


def test_invalid_arguments_are_a_tool_error_the_agent_can_read() -> None:
    reply = _rpc("tools/call", {"name": "argus_execution_plan",
                                "arguments": {"symbol": "NVDA", "usd": "lots"}})
    assert reply["result"]["isError"] is True
    assert reply["result"]["content"][0]["text"] == (
        "invalid arguments for argus_execution_plan: usd must be a number, not str")
    missing = _rpc("tools/call", {"name": "argus_quote", "arguments": {}})
    assert "symbols is required" in missing["result"]["content"][0]["text"]


def test_a_size_of_zero_is_refused_rather_than_read_as_twenty_percent() -> None:
    with pytest.raises(mcp.ToolError, match="size_percent must be above 0"):
        mcp.call_tool("argus_portfolio_impact", {
            "add": "TSLA", "size_percent": 0, "book": {"NVDA": 100}})
    reply = _rpc("tools/call", {"name": "argus_portfolio_impact", "arguments": {
        "add": "TSLA", "size_percent": 0, "book": {"NVDA": 100}}})
    assert reply["result"]["isError"] is True


def test_malformed_calls_through_the_real_engines_stay_structured() -> None:
    arguments: dict[str, Any]
    for arguments in ({"symbols": "NVDA"}, {"symbols": [1, 2]}, {"symbols": []},
                      {"symbols": ["ZYXQWV"]}):
        status, body = mcp.handle_body(json.dumps({
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {"name": "argus_quote", "arguments": arguments}}).encode())
        reply = json.loads(body)
        assert status == 200 and reply["result"]["isError"] is True, arguments


def test_names_are_validated_and_never_echoed_at_length() -> None:
    reply = _rpc("tools/call", {"name": "x" * 5000})
    assert reply["error"]["code"] == -32602 and len(reply["error"]["message"]) < 200
    assert _rpc("tools/call", {"name": "argus quote"})["error"]["code"] == -32602
    assert _rpc("tools/call", {"name": 7})["error"]["code"] == -32602


def test_envelope_rules_from_json_rpc() -> None:
    status, body = mcp.handle_body(b"[]")
    assert status == 400 and json.loads(body)["error"]["code"] == -32600
    status, body = mcp.handle_body(json.dumps(
        [{"jsonrpc": "2.0", "id": i, "method": "ping"} for i in range(mcp.MAX_BATCH + 1)]
    ).encode())
    assert status == 400 and json.loads(body)["error"]["code"] == -32600
    null_id = mcp.handle({"jsonrpc": "2.0", "id": None, "method": "ping"})
    assert null_id == {"jsonrpc": "2.0", "id": None, "result": {}}
    bad_id = mcp.handle({"jsonrpc": "2.0", "id": True, "method": "ping"})
    assert bad_id is not None and bad_id["error"]["code"] == -32600
    assert _rpc("ping", "x")["error"]["code"] == -32602  # type: ignore[arg-type]
    status, body = mcp.handle_body(b'{"jsonrpc": "2.0", "id": 1e999, "method": "ping"}')
    assert status == 400 and json.loads(body)["error"]["code"] == -32700


def test_nothing_raised_below_the_boundary_reaches_the_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def explode(message: Any, **kw: Any) -> Any:
        raise MemoryError("simulated")

    monkeypatch.setattr(mcp, "handle", explode)
    status, body = mcp.handle_body(b'{"jsonrpc": "2.0", "id": 1, "method": "ping"}')
    assert status == 500 and json.loads(body)["error"]["code"] == -32603


def test_argus_ask_carries_memory_and_the_labelled_lines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from argus.lui import server

    seen: dict[str, Any] = {}

    def fake_ask(question: str, history: Any, **kw: Any) -> dict[str, Any]:
        seen.update(kw, question=question)
        return {"refused": False, "lines": ["Assumed: 20%.", "Data: Bitget."],
                "sources": [{"ref": "bitget tickers"}], "memory": '["loss limit 5%"]'}

    monkeypatch.setattr(server, "handle_ask", fake_ask)
    reply = _rpc("tools/call", {"name": "argus_ask", "arguments": {
        "question": "should I add TSLA", "memory": '["loss limit 5%"]'}})
    text = reply["result"]["content"][0]["text"]
    assert text.startswith("[assumed] Assumed: 20%.\nData: Bitget.")
    assert "Memory (pass back as `memory` next time): [\"loss limit 5%\"]" in text
    assert seen["memory"] == '["loss limit 5%"]' and seen["visitor"] == "mcp"


class TestTheNewestEnginesAreReachable:
    """Audit, 2026-09-26: the research task, the week ahead, exposures and trade review had no MCP
    tool. Each is called here with its engine replaced, so what is pinned is the dispatch and the
    text a client receives, not the market that day."""

    def test_the_research_task_returns_the_reading_the_verdict_and_every_step(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.lui import task

        def run(question: str, request: Any, ledger: Any = None) -> Any:
            kind = request.kind.value
            return SimpleNamespace(lines=[f"{kind} reading", f"Bottom line: act on {kind}"],
                                   refused=False, data={})

        monkeypatch.setattr(task, "run", run)
        monkeypatch.setattr(task, "_price_path", lambda symbol: [])
        from argus.lui import exposures
        monkeypatch.setattr(exposures, "exposures_answer",
                            lambda before, after: (["exposure reading"], [], {}))
        text, error = mcp.call_tool("argus_research_task", {
            "question": "I hold 40% NVDA, 30% MSFT, 30% AAPL — should I add 15% TSLA?"})
        assert not error
        assert text.startswith("Read as: add 15% TSLA to 40% NVDA / 30% MSFT / 30% AAPL")
        assert "What to do, engine by engine:" in text
        assert text.count("\n\n") >= 8  # one block per step

    def test_a_question_the_task_cannot_read_says_why(self) -> None:
        text, error = mcp.call_tool("argus_research_task", {"question": "hello"})
        assert error and text

    def test_the_week_ahead_passes_the_book_and_the_window(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.lui import watchlist
        from argus.truth.source import Source

        seen: dict[str, Any] = {}

        def fake(question: str, book_text: str = "", **kwargs: Any) -> Any:
            seen.update(book=book_text, days=kwargs.get("days"))
            return ["Bottom line: CPI on Tuesday"], [Source("evidence", "bls.gov")], {}

        monkeypatch.setattr(watchlist, "watchlist", fake)
        text, error = mcp.call_tool("argus_week_ahead", {"book": "50% NVDA, 50% BTC", "days": 5})
        assert not error and "CPI on Tuesday" in text and "bls.gov" in text
        assert seen == {"book": "50% NVDA, 50% BTC", "days": 5}
        with pytest.raises(mcp.ToolError, match="days"):
            mcp.call_tool("argus_week_ahead", {"days": 40})

    def test_exposures_need_a_book(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.lui import exposures

        asked: list[str] = []

        def fake(raw_text: str, book_text: str = "", **kwargs: Any) -> Any:
            asked.append(raw_text)
            return SimpleNamespace(lines=["Bottom line: tech is 80% of the book"], sources=[],
                                   refused=False)

        monkeypatch.setattr(exposures, "answer", fake)
        text, _ = mcp.call_tool("argus_exposures",
                                {"book": "80% NVDA, 20% AAPL", "add": "TSLA 10%"})
        assert "tech is 80%" in text and asked == [
            "what are my sector and factor exposures if I add TSLA 10%"]
        with pytest.raises(mcp.ToolError, match="book is required"):
            mcp.call_tool("argus_exposures", {})

    def test_trade_review_refuses_text_with_no_trades_in_it(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.lui import journal

        monkeypatch.setattr(journal, "review_trades", lambda text, **k: None)
        with pytest.raises(mcp.ToolError, match="no trades could be read"):
            mcp.call_tool("argus_review_trades", {"fills": "just some words"})
        monkeypatch.setattr(journal, "review_trades",
                            lambda text, **k: (["2 trades reviewed"], [], {}))
        text, error = mcp.call_tool("argus_review_trades", {"fills": "2026-09-02,NVDA,buy,10,180"})
        assert not error and text == "2 trades reviewed"
