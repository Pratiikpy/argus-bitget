"""The shared MCP client and its failure taxonomy (`market/rpc.py`).

Every test here runs against a scripted transport, never the network: the claims are about how a
reply is *read* — which kind a failure is, whether it is retried, what version is negotiated — and a
live server cannot be made to produce a 429 or an expired session on demand. The live behaviour is
measured separately by `eval/skill_matrix.py`, which publishes what the real servers did.
"""

from __future__ import annotations

import email.message
import io
import json
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

import pytest

from argus.market import rpc
from argus.market.rpc import (
    ErrorKind,
    JsonRpcClient,
    Outcome,
    RpcError,
    ToolResult,
    classify_exception,
    classify_http,
    classify_rpc_error,
    payload_failure,
    reply_failure,
)

URL = "https://mcp.example.invalid/mcp"


class _Reply(io.BytesIO):
    def __init__(self, body: bytes, headers: dict[str, str] | None = None) -> None:
        super().__init__(body)
        message = email.message.Message()
        for key, value in (headers or {}).items():
            message[key] = value
        self.headers = message
        self.status = 200

    def __enter__(self) -> _Reply:
        return self


def _http_error(code: int, body: bytes = b"", headers: dict[str, str] | None = None) -> Exception:
    message = email.message.Message()
    for key, value in (headers or {}).items():
        message[key] = value
    return urllib.error.HTTPError(URL, code, "err", message, io.BytesIO(body))


def _sse(frame: dict[str, Any]) -> bytes:
    return f"event: message\ndata: {json.dumps(frame)}\n\n".encode()


class Server:
    """A scripted MCP endpoint. ``handlers`` maps a JSON-RPC method to a function of the request
    body returning bytes, a (bytes, headers) pair, or an exception to raise."""

    def __init__(self) -> None:
        self.seen: list[tuple[dict[str, Any], dict[str, str]]] = []
        self.handlers: dict[str, Callable[[dict[str, Any]], Any]] = {
            "server/discover": lambda b: _http_error(400, json.dumps({
                "jsonrpc": "2.0", "id": None,
                "error": {"code": -32600, "message": "Bad Request: Missing session ID"}}).encode()),
            "initialize": lambda b: (_sse({"jsonrpc": "2.0", "id": b["id"], "result": {
                "protocolVersion": "2025-11-25", "capabilities": {},
                "serverInfo": {"name": "fake", "version": "9"}}}), {"Mcp-Session-Id": "s1"}),
            "notifications/initialized": lambda b: b"",
        }

    def __call__(self, request: Any, *args: Any, **kwargs: Any) -> Any:
        if request.get_method() == "DELETE":
            self.seen.append(({"method": "DELETE"}, dict(request.header_items())))
            reply = self.handlers.get("DELETE", lambda b: b"")({})
            if isinstance(reply, Exception):
                raise reply
            return _Reply(reply)
        body = json.loads(request.data.decode())
        self.seen.append((body, dict(request.header_items())))
        reply = self.handlers[body["method"]](body)
        if isinstance(reply, Exception):
            raise reply
        if not isinstance(reply, bytes | tuple):
            return reply  # a scripted stream object, returned as is
        if isinstance(reply, tuple):
            return _Reply(reply[0], reply[1])
        return _Reply(reply)

    def methods(self) -> list[str]:
        return [b["method"] for b, _ in self.seen]


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch) -> Server:
    rpc._reset_discovery_cache()
    fake = Server()
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    return fake


def _client(**kwargs: Any) -> tuple[JsonRpcClient, list[float]]:
    slept: list[float] = []
    return JsonRpcClient(URL, sleep=slept.append, **kwargs), slept


def _tool_reply(result: dict[str, Any]) -> Callable[[dict[str, Any]], bytes]:
    return lambda b: _sse({"jsonrpc": "2.0", "id": b["id"], "result": result})


# --- classification --------------------------------------------------------------------------

class TestTheTaxonomyTellsFailuresApart:
    def test_http_statuses_map_to_owners(self) -> None:
        assert classify_http(429, b"", retry_after="3").kind is ErrorKind.RATE_LIMIT
        assert classify_http(429, b"", retry_after="3").retry_after == 3.0
        assert classify_http(401, b"").kind is ErrorKind.AUTH
        assert classify_http(403, b"").kind is ErrorKind.AUTH
        assert classify_http(404, b"").kind is ErrorKind.NOT_FOUND
        assert classify_http(503, b"").kind is ErrorKind.UPSTREAM_5XX

    def test_a_400_carrying_a_json_rpc_error_is_a_protocol_error(self) -> None:
        """Both Bitget servers answer a missing session with HTTP 400 and a JSON-RPC body,
        observed 2026-09-25. Reading only the status would call it a transport failure."""
        body = json.dumps({"jsonrpc": "2.0", "id": None, "error": {
            "code": -32600, "message": "Bad Request: Missing session ID"}}).encode()
        error = classify_http(400, body)
        assert error.kind is ErrorKind.PROTOCOL and error.code == -32600

    def test_network_failures(self) -> None:
        assert classify_exception(TimeoutError("x")).kind is ErrorKind.TIMEOUT
        assert classify_exception(urllib.error.URLError(TimeoutError())).kind is \
            ErrorKind.TIMEOUT
        assert classify_exception(
            urllib.error.URLError(ConnectionRefusedError())).kind is ErrorKind.TRANSPORT

    def test_json_rpc_codes_and_prose(self) -> None:
        assert classify_rpc_error({"code": -32022, "message": "x"}).kind is \
            ErrorKind.UNSUPPORTED_VERSION
        assert classify_rpc_error({"code": -32004, "message": "x"}).kind is \
            ErrorKind.UNSUPPORTED_VERSION  # the pre-2026-07-28 numbering
        assert classify_rpc_error({"code": -32000, "message": "Rate limit exceeded"}).kind is \
            ErrorKind.RATE_LIMIT  # ToolBench reads the words when the code says nothing
        assert classify_rpc_error({"code": -32602, "message": "Unknown tool"}).kind is \
            ErrorKind.PROTOCOL  # SEP-1303: failing to find a tool is a protocol error

    def test_retryable_is_derived_from_the_kind(self) -> None:
        retryable = {k for k in ErrorKind if RpcError(k, "").retryable}
        assert retryable == {ErrorKind.TIMEOUT, ErrorKind.TRANSPORT, ErrorKind.RATE_LIMIT,
                             ErrorKind.UPSTREAM_5XX}

    def test_every_kind_has_one_matrix_column_and_a_label(self) -> None:
        for kind in ErrorKind:
            assert isinstance(rpc.outcome_of(kind), Outcome)
            assert rpc.LABELS[kind]


class TestFailuresInsideAResult:
    def test_the_upstream_503_bitget_mcp_server_reports_inside_a_200(self) -> None:
        failed = payload_failure({"success": False, "status_code": 503, "data": "<html>"})
        assert failed is not None and failed[0] is ErrorKind.UPSTREAM_5XX

    def test_a_named_refusal_is_a_domain_error_and_an_empty_one_is_not(self) -> None:
        failed = payload_failure({"error": "Unknown action: latest"})
        assert failed is not None and failed[0] is ErrorKind.DOMAIN
        assert payload_failure({"error": ""}) is None  # the hollow envelope: empty, not refused
        assert payload_failure({"success": True, "data": {"results": [1]}}) is None

    def test_a_data_row_with_a_code_field_is_not_read_as_a_status(self) -> None:
        assert payload_failure({"code": 404, "name": "some instrument"}) is None

    def test_is_error_is_a_tool_execution_error(self) -> None:
        result = ToolResult.from_result("t", {"isError": True, "content": [
            {"type": "text", "text": "Error executing tool cross_asset"}]})
        failed = result.failure()
        assert failed is not None and failed[0] is ErrorKind.TOOL_EXECUTION

    def test_reply_failure_reads_raw_sse_bytes(self) -> None:
        escaped = _sse({"jsonrpc": "2.0", "id": 1, "result": {"content": [
            {"type": "text", "text": json.dumps({"success": False, "status_code": 503})}]}})
        assert reply_failure(escaped) == (ErrorKind.UPSTREAM_5XX, "upstream answered 503")
        protocol = _sse({"jsonrpc": "2.0", "id": 1, "error": {"code": -32602, "message": "x"}})
        failed = reply_failure(protocol)
        assert failed is not None and failed[0] is ErrorKind.PROTOCOL
        ok = _sse({"jsonrpc": "2.0", "id": 1, "result": {"content": [
            {"type": "text", "text": "{\"value\": 42}"}]}})
        assert reply_failure(ok) is None


# --- the client ------------------------------------------------------------------------------

class TestNegotiation:
    def test_the_servers_newest_version_is_used_and_sent_on_every_request(
        self, server: Server
    ) -> None:
        server.handlers["tools/call"] = _tool_reply({"content": [{"type": "text", "text": "1"}]})
        client, _ = _client()
        client.call_tool("x", {})
        assert client.negotiation is not None
        assert client.negotiation.negotiated == "2025-11-25"
        assert client.negotiation.server == "fake 9"
        assert "not implemented" in client.negotiation.discover
        _, headers = server.seen[-1]
        normalised = {k.lower(): v for k, v in headers.items()}
        assert normalised["mcp-protocol-version"] == "2025-11-25"
        assert normalised["mcp-session-id"] == "s1"
        assert server.methods() == ["server/discover", "initialize",
                                    "notifications/initialized", "tools/call"]

    def test_discover_is_probed_once_per_endpoint(self, server: Server) -> None:
        _client()[0].initialize()
        _client()[0].initialize()
        assert server.methods().count("server/discover") == 1

    def test_a_version_this_client_does_not_implement_is_refused(self, server: Server) -> None:
        server.handlers["initialize"] = lambda b: _sse(
            {"jsonrpc": "2.0", "id": b["id"], "result": {"protocolVersion": "2030-01-01"}})
        with pytest.raises(RpcError) as caught:
            _client()[0].initialize()
        assert caught.value.kind is ErrorKind.UNSUPPORTED_VERSION

    def test_unsupported_version_error_falls_back_to_the_newest_mutual_version(
        self, server: Server
    ) -> None:
        def initialize(body: dict[str, Any]) -> bytes:
            if body["params"]["protocolVersion"] != "2025-06-18":
                return _sse({"jsonrpc": "2.0", "id": body["id"], "error": {
                    "code": -32022, "message": "Unsupported protocol version",
                    "data": {"supported": ["2025-06-18", "2024-11-05"],
                             "requested": body["params"]["protocolVersion"]}}})
            return _sse({"jsonrpc": "2.0", "id": body["id"],
                         "result": {"protocolVersion": "2025-06-18"}})

        server.handlers["initialize"] = initialize
        negotiation = _client()[0].initialize()
        assert (negotiation.requested, negotiation.negotiated) == ("2025-06-18", "2025-06-18")


class TestRetryOnlyWhatARetryCanChange:
    def test_a_5xx_is_retried_and_succeeds(self, server: Server) -> None:
        replies = iter([_http_error(503), None])

        def call(body: dict[str, Any]) -> Any:
            nxt = next(replies)
            return nxt if nxt is not None else _tool_reply({"content": []})(body)

        server.handlers["tools/call"] = call
        client, slept = _client()
        client.call_tool("x", {})
        assert slept == [0.5]

    def test_a_protocol_error_is_not_retried(self, server: Server) -> None:
        server.handlers["tools/call"] = lambda b: _sse({"jsonrpc": "2.0", "id": b["id"], "error": {
            "code": -32602, "message": "Unknown tool: nope"}})
        client, slept = _client()
        with pytest.raises(RpcError) as caught:
            client.call_tool("nope", {})
        assert caught.value.kind is ErrorKind.PROTOCOL and slept == []
        assert server.methods().count("tools/call") == 1

    def test_a_tool_error_comes_back_in_the_result_not_as_an_exception(
        self, server: Server
    ) -> None:
        server.handlers["tools/call"] = _tool_reply({"isError": True, "content": [
            {"type": "text", "text": "validation error"}]})
        result = _client()[0].call_tool("x", {})
        assert result.is_error and server.methods().count("tools/call") == 1

    def test_rate_limit_honours_retry_after_within_a_cap(self, server: Server) -> None:
        replies = iter([_http_error(429, headers={"Retry-After": "60"}), None])

        def call(body: dict[str, Any]) -> Any:
            nxt = next(replies)
            return nxt if nxt is not None else _tool_reply({"content": []})(body)

        server.handlers["tools/call"] = call
        client, slept = _client()
        client.call_tool("x", {})
        assert slept == [10.0]  # capped, not a minute inside a live answer

    def test_timeouts_are_not_retried_when_the_client_says_so(self, server: Server) -> None:
        server.handlers["tools/call"] = lambda b: TimeoutError("read timed out")
        client, slept = _client(retry_timeouts=False)
        with pytest.raises(RpcError) as caught:
            client.call_tool("x", {})
        assert caught.value.kind is ErrorKind.TIMEOUT and slept == []
        retrying, slept_too = _client()
        with pytest.raises(RpcError) as again:
            retrying.call_tool("x", {})
        assert again.value.attempts == 2 and slept_too == [0.5]

    def test_an_expired_session_is_renewed_once(self, server: Server) -> None:
        replies = iter([_http_error(404), None])

        def call(body: dict[str, Any]) -> Any:
            nxt = next(replies)
            return nxt if nxt is not None else _tool_reply({"content": []})(body)

        server.handlers["tools/call"] = call
        client, _ = _client()
        client.call_tool("x", {})
        assert server.methods().count("initialize") == 2


class TestReplyCorrelation:
    def test_the_frame_with_our_id_is_chosen_among_notifications(self, server: Server) -> None:
        def call(body: dict[str, Any]) -> bytes:
            return (_sse({"jsonrpc": "2.0", "method": "notifications/progress", "params": {}})
                    + _sse({"jsonrpc": "2.0", "id": body["id"], "result": {"content": [
                        {"type": "text", "text": "mine"}]}}))

        server.handlers["tools/call"] = call
        assert _client()[0].call_tool("x", {}).text == "mine"

    def test_another_requests_reply_is_refused(self, server: Server) -> None:
        server.handlers["tools/call"] = lambda b: _sse({"jsonrpc": "2.0", "id": 99999,
                                                        "result": {"content": []}})
        with pytest.raises(RpcError) as caught:
            _client()[0].call_tool("x", {})
        assert caught.value.kind is ErrorKind.CORRELATION

    def test_list_tools_follows_the_cursor(self, server: Server) -> None:
        def listing(body: dict[str, Any]) -> bytes:
            cursor = body["params"].get("cursor")
            page = {"tools": [{"name": "b"}]} if cursor else \
                {"tools": [{"name": "a"}], "nextCursor": "p2"}
            return _sse({"jsonrpc": "2.0", "id": body["id"], "result": page})

        server.handlers["tools/list"] = listing
        assert [t["name"] for t in _client()[0].list_tools()] == ["a", "b"]


# --- the two migrated clients ----------------------------------------------------------------

class TestTheMigratedClientsKeepTheirContracts:
    def test_evidence_status_strings_still_read_right_in_skills(self, server: Server) -> None:
        from argus.market.evidence import BitgetSkillSource
        from argus.market.skills import Health, _classify

        source = BitgetSkillSource()
        source.client.url = URL
        server.handlers["tools/call"] = lambda b: _sse({"jsonrpc": "2.0", "id": b["id"], "error": {
            "code": -32602, "message": "Unknown tool: x"}})
        payload, status = source.call("x", {})
        assert "protocol error" in status
        assert _classify(payload, status)[0] is Health.UNAVAILABLE  # was EMPTY before 2026-09-25

        server.handlers["tools/call"] = lambda b: TimeoutError("slow")
        payload, status = source.call("x", {})
        assert _classify(payload, status)[0] is Health.TIMEOUT
        assert server.methods().count("tools/call") == 2  # no retry of a timeout here

        server.handlers["tools/call"] = _tool_reply({"isError": True, "content": [
            {"type": "text", "text": "Error executing tool cross_asset"}]})
        payload, status = source.call("x", {})
        assert _classify(payload, status)[0] is Health.TOOL_ERROR

        server.handlers["tools/call"] = _tool_reply({"content": [
            {"type": "text", "text": json.dumps({"error": "Unknown action: latest"})}]})
        payload, status = source.call("x", {})
        assert "returned no data (upstream refused the request" in status
        assert _classify(payload, status)[0] is Health.EMPTY

        server.handlers["tools/call"] = _tool_reply({"content": [
            {"type": "text", "text": json.dumps({"value": 44})}]})
        assert source.call("x", {}) == ({"value": 44}, "bitget:x: ok")

    def test_bitget_mcp_errors_carry_their_kind(self, server: Server) -> None:
        from argus.market.bitget_mcp import BitgetDataService, BitgetMcpError

        client = JsonRpcClient(URL, sleep=lambda s: None)
        service = BitgetDataService(client=client)
        assert service.server == "fake 9"

        server.handlers["tools/call"] = _tool_reply({"structuredContent": {
            "success": False, "status_code": 503, "data": "<html>503</html>"}, "content": []})
        with pytest.raises(BitgetMcpError) as down:
            service.query("equity_price_quote", symbol="NVDA")
        assert down.value.kind is ErrorKind.UPSTREAM_5XX
        assert isinstance(down.value, RuntimeError)

        server.handlers["tools/call"] = _tool_reply({"isError": True, "content": [
            {"type": "text", "text": "1 validation error for call[do_query]"}]})
        with pytest.raises(BitgetMcpError) as refused:
            service.query("equity_x", symbol="NVDA")
        assert refused.value.kind is ErrorKind.TOOL_EXECUTION
        assert "validation error" in str(refused.value)  # eval/datacoverage.py reads this

        server.handlers["tools/call"] = _tool_reply({"structuredContent": {
            "success": True, "data": {"results": [{"symbol": "NVDA"}]}}, "content": []})
        assert service.results("equity_price_quote", symbol="NVDA") == [{"symbol": "NVDA"}]


# --- coverage ---------------------------------------------------------------------------------

def test_coverage_names_the_kind_of_a_failed_mcp_call(monkeypatch: pytest.MonkeyPatch) -> None:
    from argus.truth import coverage

    rpc._reset_discovery_cache()
    fake = Server()
    fake.handlers["tools/call"] = _tool_reply({"content": [{"type": "text", "text": json.dumps(
        {"success": False, "status_code": 503})}]})
    monkeypatch.setattr(coverage, "_original", fake)
    monkeypatch.setattr(urllib.request, "urlopen", coverage._observed)
    client = JsonRpcClient("https://agent.bitget.com/mcp", sleep=lambda s: None)
    with coverage.recording() as record:
        client.call_tool("do_query", {"entry_id": "equity_calendar"})
    assert record.missing == ["bitget-mcp-server equity_calendar"]
    assert record.why["bitget-mcp-server equity_calendar"] == "upstream 5xx"
    assert record.as_dict()["failure_kinds"] == {
        "bitget-mcp-server equity_calendar": "upstream_5xx"}


class _Dripping:
    """A stream that sends an SSE keep-alive comment on every read and never the reply — what
    bitget-signal's ``cn_market`` did for 150 s on 2026-09-25."""

    def __init__(self) -> None:
        self.headers = email.message.Message()
        self.status = 200
        self.reads = 0

    def read1(self, amt: int = -1) -> bytes:
        import time

        self.reads += 1
        time.sleep(0.02)
        return b": ping - keepalive\r\n\r\n"

    def read(self, amt: int = -1) -> bytes:
        return self.read1(amt)

    def close(self) -> None:
        return None

    def __enter__(self) -> _Dripping:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def test_a_stream_held_open_by_pings_times_out_on_the_wall_clock(server: Server) -> None:
    drip = _Dripping()
    server.handlers["tools/call"] = lambda b: drip
    original = server.__call__

    def call(request: Any, *args: Any, **kwargs: Any) -> Any:
        reply = original(request, *args, **kwargs)
        return reply._payload if isinstance(reply, _Reply) and hasattr(reply, "_payload") \
            else reply

    client, slept = _client(retry_timeouts=False, timeout=0.2)
    with pytest.raises(RpcError) as caught:
        client.call_tool("cn_market", {})
    assert caught.value.kind is ErrorKind.TIMEOUT
    assert "keep-alive ping" in str(caught.value)
    assert drip.reads >= 3 and slept == []


def test_coverage_does_not_hang_on_a_pinging_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    from argus.truth import coverage

    rpc._reset_discovery_cache()
    fake = Server()
    fake.handlers["tools/call"] = lambda b: _Dripping()
    monkeypatch.setattr(coverage, "_original", fake)
    monkeypatch.setattr(urllib.request, "urlopen", coverage._observed)
    client = JsonRpcClient("https://datahub.noxiaohao.com/mcp", sleep=lambda s: None,
                           retry_timeouts=False, timeout=0.2)
    with coverage.recording() as record, pytest.raises(RpcError):
        client.call_tool("cn_market", {"action": "index"})
    assert record.why == {"bitget-signal cn_market": "timed out"}
    assert record.kinds == {"bitget-signal cn_market": "timeout"}


class TestTheSessionIsEnded:
    """No client ended its sessions until bitget-mcp-server refused new ones with "Too many open
    sessions" (2026-09-26)."""

    def test_close_sends_delete_with_the_session_id(self, server: Server) -> None:
        client, _ = _client()
        client.initialize()
        client.close()
        method, headers = server.seen[-1]
        assert method["method"] == "DELETE"
        assert {k.lower(): v for k, v in headers.items()}["mcp-session-id"] == "s1"
        assert client.session is None

    def test_close_without_a_session_sends_nothing(self, server: Server) -> None:
        client, _ = _client()
        client.close()
        assert server.seen == []

    def test_a_server_that_refuses_the_delete_is_not_an_error(self, server: Server) -> None:
        server.handlers["DELETE"] = lambda b: _http_error(405)
        client, _ = _client()
        client.initialize()
        client.close()
        assert client.session is None

    def test_the_context_manager_closes(self, server: Server) -> None:
        with _client()[0] as client:
            client.initialize()
        assert server.methods()[-1] == "DELETE"

