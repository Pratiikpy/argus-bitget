"""Every research answer says which sources it reached and which did not answer."""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.request
from typing import Any, ClassVar

import pytest

from argus.truth import coverage


class FakeResponse(io.BytesIO):
    headers: ClassVar[dict[str, str]] = {"Mcp-Session-Id": "s1"}
    status = 200

    def __enter__(self) -> FakeResponse:
        return self


@pytest.fixture
def transport(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """The wrapper installed over a fake transport: URL → bytes, or an exception to raise."""
    replies: dict[str, Any] = {}

    def fake(url: Any, *args: Any, **kwargs: Any) -> Any:
        target = url.full_url if isinstance(url, urllib.request.Request) else str(url)
        reply = replies[target]
        if isinstance(reply, Exception):
            raise reply
        return FakeResponse(reply)

    monkeypatch.setattr(coverage, "_original", fake)
    monkeypatch.setattr(urllib.request, "urlopen", coverage._observed)
    return replies


def _mcp(tool: str, arguments: dict[str, Any] | None = None) -> urllib.request.Request:
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                       "params": {"name": tool, "arguments": arguments or {}}}).encode()
    return urllib.request.Request("https://agent.bitget.com/mcp", data=body, method="POST")


def test_sources_are_named_for_a_reader() -> None:
    name = coverage.source_name
    assert name("https://api.bitget.com/api/v2/mix/market/tickers?x=1") == "Bitget tickers"
    assert name("https://api.bitget.com/api/v3/market/history-fund-rate") == \
        "Bitget funding history"
    assert name("https://gamma-api.polymarket.com/public-search?q=tesla") == "Polymarket"
    assert name("https://www.cnbc.com/id/100003114/device/rss/rss.html") == "news feed cnbc.com"
    assert name("https://data.sec.gov/submissions/CIK1.json") == "SEC XBRL"
    body = _mcp("do_query", {"entry_id": "equity_calendar"}).data
    assert isinstance(body, bytes)
    assert name("https://agent.bitget.com/mcp", body) == "bitget-mcp-server equity_calendar"
    handshake = json.dumps({"method": "initialize"}).encode()
    assert name("https://agent.bitget.com/mcp", handshake) == ""


def test_a_complete_answer_and_a_partial_one(transport: dict[str, Any]) -> None:
    transport["https://api.bitget.com/api/v2/mix/market/tickers"] = b"{}"
    transport["https://gamma-api.polymarket.com/public-search"] = urllib.error.HTTPError(
        "https://gamma-api.polymarket.com/public-search", 503, "down", {}, None)  # type: ignore[arg-type]
    with coverage.recording() as record:
        urllib.request.urlopen("https://api.bitget.com/api/v2/mix/market/tickers")
    assert record.line() == "Sources reached: 1 of 1 answered."
    with coverage.recording() as record:
        urllib.request.urlopen("https://api.bitget.com/api/v2/mix/market/tickers")
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen("https://gamma-api.polymarket.com/public-search")
    assert record.line() == ("Sources reached: 1 of 2 answered. Did not answer: Polymarket "
                             "(HTTP 503) — everything above is built without it.")


def test_an_mcp_tool_error_inside_a_200_is_not_answered(transport: dict[str, Any]) -> None:
    transport["https://agent.bitget.com/mcp"] = b'{"result": {"isError": true, "content": []}}'
    request = _mcp("do_query", {"entry_id": "equity_calendar"})
    with coverage.recording() as record, urllib.request.urlopen(request) as reply:
        assert json.load(reply)["result"]["isError"] is True  # the caller still reads it
    assert record.missing == ["bitget-mcp-server equity_calendar"]


def test_worker_threads_count_only_through_the_context_pool(transport: dict[str, Any]) -> None:
    from concurrent.futures import ThreadPoolExecutor

    transport["https://api.bitget.com/api/v2/mix/market/candles"] = b"[]"
    with coverage.recording() as record, coverage.ContextPool(max_workers=2) as pool:
        pool.submit(urllib.request.urlopen,
                    "https://api.bitget.com/api/v2/mix/market/candles").result()
    assert record.answered == {"Bitget candles": True}
    with coverage.recording() as record, ThreadPoolExecutor(max_workers=2) as plain:
        plain.submit(urllib.request.urlopen,
                     "https://api.bitget.com/api/v2/mix/market/candles").result()
    assert record.answered == {}  # why research.py uses ContextPool everywhere


def test_outside_a_recording_nothing_is_noted(transport: dict[str, Any]) -> None:
    transport["https://api.bitget.com/api/v2/mix/market/tickers"] = b"{}"
    urllib.request.urlopen("https://api.bitget.com/api/v2/mix/market/tickers")
    with coverage.recording() as record:
        pass
    assert record.line() is None
