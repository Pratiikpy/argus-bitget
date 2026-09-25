"""The Skill-effectiveness matrix (`eval/skill_matrix.py`), exercised against scripted clients.

The live sweep is what publishes `data/skill_matrix.json`; these tests pin how a reply becomes a
column, so a change to the taxonomy that would silently move calls between columns fails here.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from argus.eval import skill_matrix as sm
from argus.eval.artefact import is_strict, write
from argus.market.rpc import ErrorKind, Outcome, RpcError, ToolResult
from argus.market.skills import SKILLS


def _text(payload: Any, *, is_error: bool = False) -> ToolResult:
    return ToolResult.from_result("t", {"isError": is_error, "content": [
        {"type": "text", "text": payload if isinstance(payload, str) else json.dumps(payload)}]})


def _structured(payload: Any) -> ToolResult:
    return ToolResult.from_result("do_query", {"content": [], "structuredContent": payload})


class TestAReplyLandsInExactlyOneColumn:
    def test_signal_replies(self) -> None:
        server = sm.SIGNAL_SERVER
        assert sm.classify_result(_text({"value": 44}), server=server)[0] is Outcome.ANSWERED
        assert sm.classify_result(_text({"alt_me_error": ""}), server=server)[0] is Outcome.EMPTY
        curve = {"2y": {"error": ""}, "10y": {"error": ""}, "spread_10y2y": 0.0}
        assert sm.classify_result(_text(curve), server=server)[0] is Outcome.EMPTY
        assert sm.classify_result(_text({"error": "Unknown action: x"}), server=server)[0] is \
            Outcome.DOMAIN_ERROR
        assert sm.classify_result(_text("Error executing tool", is_error=True),
                                  server=server)[0] is Outcome.DOMAIN_ERROR

    def test_mcp_server_replies(self) -> None:
        server = sm.MCP_SERVER
        rows = _structured({"success": True, "data": {"results": [{"symbol": "NVDA", "p": 1.5}]}})
        assert sm.classify_result(rows, server=server)[0] is Outcome.ANSWERED
        none = _structured({"success": True, "data": {"results": []}})
        assert sm.classify_result(none, server=server)[0] is Outcome.EMPTY
        down = _structured({"success": False, "status_code": 503, "data": "<html>"})
        outcome, kind, _ = sm.classify_result(down, server=server)
        assert (outcome, kind) == (Outcome.UPSTREAM_5XX, ErrorKind.UPSTREAM_5XX.value)

    def test_a_raised_failure_is_classified_and_the_legacy_reading_kept(self) -> None:
        def timeout() -> ToolResult:
            raise RpcError(ErrorKind.TIMEOUT, "timed out")

        def unknown_tool() -> ToolResult:
            raise RpcError(ErrorKind.PROTOCOL, "JSON-RPC error -32602: Unknown tool")

        slow = sm.attempt(timeout, {}, server=sm.SIGNAL_SERVER)
        assert (slow.outcome, slow.legacy) == (Outcome.TRANSPORT_ERROR, "timeout")
        refused = sm.attempt(unknown_tool, {}, server=sm.SIGNAL_SERVER)
        # The measured defect: the pre-migration client counted a protocol error as empty data.
        assert (refused.outcome, refused.legacy) == (Outcome.PROTOCOL_ERROR, "empty")


class TestTheCallsAreReadFromTheServers:
    def test_every_signal_tool_has_one_call_and_every_skill_is_covered(self) -> None:
        tools = [c.tool for c in sm.SIGNAL_CALLS]
        assert len(tools) == len(set(tools)) == 19
        assert set(SKILLS) <= {c.skill for c in sm.SIGNAL_CALLS}
        assert all(c.args.get("action") for c in sm.SIGNAL_CALLS)

    def test_entry_forms_follow_params_summary(self) -> None:
        required = {"params_summary": [{"name": "symbol", "required": True}]}
        assert sm.entry_forms(required, "crypto") == [{"symbol": "BTCUSDT"}, {"symbol": "BTC"}]
        optional = {"params_summary": [{"name": "symbol", "required": False}]}
        assert sm.entry_forms(optional, "equity") == [{"symbol": "NVDA"}, {}]
        labelled = {"params_summary": [{"name": "label", "required": True}]}
        assert [f["label"] for f in sm.entry_forms(labelled, "news")] == list(sm.NEWS_LABELS)
        typed = {"params_summary": [{"name": "label", "required": True, "type": "integer"}]}
        assert sm.entry_forms(typed, "news") == [{"label": 1}]  # the catalog's declared type
        assert sm.entry_forms({"params_summary": []}, "sentiment") == [{}]
        batch = {"params_summary": [{"name": "symbols", "required": True}]}
        assert sm.entry_forms(batch, "crypto")[0] == {"symbols": "BTCUSDT"}


class _Mcp:
    def __init__(self, replies: Mapping[str, list[Any]]) -> None:
        self.replies = {k: list(v) for k, v in replies.items()}
        self.calls: list[dict[str, Any]] = []

    def call_tool(self, name: str, arguments: Mapping[str, Any], *,
                  timeout: float | None = None) -> ToolResult:
        self.calls.append(dict(arguments))
        reply = self.replies[str(arguments["entry_id"])].pop(0)
        if isinstance(reply, RpcError):
            raise reply
        assert isinstance(reply, ToolResult)
        return reply


class _Signal:
    def __init__(self, reply: ToolResult) -> None:
        self.reply = reply
        self.tools: list[str] = []

    def call_tool(self, tool: str, args: dict[str, Any], *, timeout: int = 45) -> ToolResult:
        self.tools.append(tool)
        return self.reply


def _no_sleep(_: float) -> None:
    return None


class TestSweeps:
    def test_a_second_spelling_is_tried_after_an_empty_answer(self) -> None:
        entry = {"id": "crypto_x", "params_summary": [{"name": "symbol", "required": True}]}
        client = _Mcp({"crypto_x": [
            _structured({"success": True, "data": {"results": []}}),
            _structured({"success": True, "data": {"results": [{"v": 2}]}}),
        ]})
        rows: dict[str, sm.Row] = {}
        sm.sweep_mcp(client, [("crypto", entry)], rows, sleep=_no_sleep)
        row = rows["bitget-mcp-server:crypto_x"]
        assert [c["params"] for c in client.calls] == [{"symbol": "BTCUSDT"}, {"symbol": "BTC"}]
        assert row.outcomes[Outcome.ANSWERED.value] == 1
        assert len(row.attempts[0]) == 2  # both attempts are published

    def test_a_transport_failure_is_not_retried_with_another_spelling(self) -> None:
        entry = {"id": "crypto_y", "params_summary": [{"name": "symbol", "required": True}]}
        client = _Mcp({"crypto_y": [RpcError(ErrorKind.TRANSPORT, "reset")]})
        rows: dict[str, sm.Row] = {}
        sm.sweep_mcp(client, [("crypto", entry)], rows, sleep=_no_sleep)
        assert len(client.calls) == 1
        assert rows["bitget-mcp-server:crypto_y"].outcomes[Outcome.TRANSPORT_ERROR.value] == 1

    def test_the_report_tallies_by_skill_and_renders(self, tmp_path: Path) -> None:
        rows: dict[str, sm.Row] = {}
        signal = _Signal(_text({"value": 1}))
        sm.sweep_signal(signal, rows, sleep=_no_sleep)
        sm.sweep_signal(signal, rows, sleep=_no_sleep)
        entry = {"id": "equity_q", "params_summary": [{"name": "symbol", "required": True}]}
        mcp = _Mcp({"equity_q": [
            _structured({"success": False, "status_code": 503}),
            _structured({"success": False, "status_code": 503}),
        ]})
        sm.sweep_mcp(mcp, [("equity", entry)], rows, sleep=_no_sleep)
        sm.sweep_mcp(mcp, [("equity", entry)], rows, sleep=_no_sleep)
        matrix = sm.report(list(rows.values()), rounds=2, servers={
            sm.SIGNAL_SERVER: {"server": "fake", "negotiated": "2025-11-25",
                               "requested": "2025-11-25"}}, generated_at="2026-09-25T00:00:00")
        assert signal.tools == [c.tool for c in sm.SIGNAL_CALLS] * 2
        assert matrix["totals"]["calls"] == 19 * 2 + 2
        assert matrix["by_server"][sm.SIGNAL_SERVER]["answered"] == 38
        assert matrix["by_skill"]["bitget-mcp-server/equity"]["upstream_5xx"] == 2
        assert matrix["failure_kinds"] == {"upstream_5xx": 2}
        # The old datacoverage probe could only say "error" for the 503.
        assert matrix["taxonomy_vs_legacy"]["failures_legacy_left_as_one_word"] == 2
        lines = sm.render(matrix)
        assert lines[0].startswith("SKILL EFFECTIVENESS MATRIX — 2 round(s), 40 calls")
        assert any("upstream 5xx 2" in line for line in lines)
        assert any(line.strip().startswith("TOTAL") for line in lines)

        path = tmp_path / "skill_matrix.json"
        write(path, matrix)
        assert is_strict(path)
        assert sm.console_lines(path) == lines
        assert "has not been run" in sm.console_lines(tmp_path / "absent.json")[0]


def test_a_name_shared_by_both_servers_stays_two_rows() -> None:
    """`crypto_market` is a bitget-signal tool and a bitget-mcp-server catalog entry."""
    rows: dict[str, sm.Row] = {}
    sm.sweep_signal(_Signal(_text({"value": 1})), rows, sleep=_no_sleep)
    entry = {"id": "crypto_market", "params_summary": []}
    sm.sweep_mcp(_Mcp({"crypto_market": [_structured({"success": True, "data": {
        "results": [{"v": 1}]}})]}), [("crypto", entry)], rows, sleep=_no_sleep)
    assert len(rows) == 20
    assert {r.server for r in rows.values() if r.tool == "crypto_market"} == {
        sm.SIGNAL_SERVER, sm.MCP_SERVER}
