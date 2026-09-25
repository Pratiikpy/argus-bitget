"""Anonymous usage events (`lui/usage.py`) and their report (`eval/usage_report.py`), offline."""

from __future__ import annotations

import json
import threading
import urllib.request
from datetime import UTC, datetime
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import pytest

from argus.eval import usage_report
from argus.lui import usage


def test_visitor_hash_rotates_daily_and_hides_the_address(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARGUS_USAGE_SALT", "s")
    one = datetime(2026, 9, 25, tzinfo=UTC)
    two = datetime(2026, 9, 26, tzinfo=UTC)
    assert usage.visitor_hash("1.2.3.4", one) == usage.visitor_hash("1.2.3.4", one)
    assert usage.visitor_hash("1.2.3.4", one) != usage.visitor_hash("1.2.3.4", two)
    assert usage.visitor_hash("1.2.3.4", one) != usage.visitor_hash("1.2.3.5", one)
    assert "1.2.3.4" not in usage.visitor_hash("1.2.3.4", one)
    monkeypatch.setenv("ARGUS_USAGE_SALT", "other")
    assert usage.visitor_hash("1.2.3.4", one) != "" and len(usage.visitor_hash("1.2.3.4")) == 12


@pytest.mark.parametrize(("agent", "kind"), [
    ("Mozilla/5.0 (Windows NT 10.0) Chrome/140", "browser"),
    ("Mozilla/5.0 HeadlessChrome/140", "automated"), ("curl/8.4", "automated"),
    ("Python-urllib/3.11", "automated"), ("", "automated")])
def test_client_class(agent: str, kind: str) -> None:
    assert usage.client_class(agent) == kind


def test_the_ask_event_carries_the_outcome_and_never_the_words(
        capsys: pytest.CaptureFixture[str]) -> None:
    payload = {"classified_by": "research", "intent": "research", "refused": False,
               "elapsed_ms": 812.4, "lines": ["Actionable: my secret NVDA thesis", "b"],
               "line_labels": ["computed", "missing"], "sources": [{}], "remembered": []}
    usage.ask_event(payload, answer="a" * 12, visitor="v" * 12, client="browser",
                    internal=False)
    line = capsys.readouterr().out.strip()
    event = json.loads(line)
    assert event["kind"] == "ask" and event["missing"] == 1 and event["lines"] == 2
    assert "secret" not in line and "NVDA" not in line


def test_a_malformed_feedback_id_is_refused() -> None:
    assert usage.feedback_event("../etc", True, visitor="v", client="browser",
                                internal=False) is None


def _event(kind: str, i: str, **extra: Any) -> dict[str, Any]:
    return {"argus_usage": 1, "kind": kind, "id": i, "at": "2026-09-25T10:00:00+00:00",
            "v": "v1", "client": "browser", "internal": False, **extra}


def test_report_counts_only_real_use_and_flags_thin_feedback(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(usage_report, "STORE", tmp_path / "events.jsonl")
    events = [_event("ask", "a1", refused=False, ms=500, route="research"),
              _event("ask", "a2", refused=True, ms=100, route="declined", v="v2"),
              _event("ask", "a3", refused=False, ms=300, route="research", internal=True),
              _event("ask", "a4", refused=False, ms=300, route="research", client="automated"),
              _event("feedback", "a1", useful=True),
              _event("feedback", "a3", useful=True, internal=True)]
    assert usage_report.store(events) == 6
    assert usage_report.store(events) == 0  # the store de-duplicates
    got = usage_report.summarise(usage_report.load())
    assert got["questions"] == 2 and got["answered"] == 1 and got["visitor_days"] == 2
    assert got["feedback_clicks"] == 1 and got["task_completion_rate"] == 1.0 and got["thin"]
    assert got["excluded"] == {"automated": 1, "internal": 1}


def test_events_are_read_out_of_a_vercel_log_record() -> None:
    event = _event("ask", "b" * 12)
    record = {"message": "GET /ask 200", "logs": [
        {"message": "noise"}, {"message": json.dumps(event)}]}
    assert usage_report.events_in(record) == [event]


@pytest.fixture
def base_url() -> Any:
    from argus.lui.server import Handler

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def _post(url: str, fields: dict[str, str]) -> tuple[int, dict[str, Any]]:
    request = urllib.request.Request(url, data=urlencode(fields).encode(), method="POST",
                                     headers={"User-Agent": "Mozilla/5.0 Chrome/140"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_the_page_routes_answer_by_post_and_log_an_event(
        base_url: str, capfd: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("BITGET_QWEN_API_KEY", "QWEN_API_KEY", "DASHSCOPE_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    status, answer = _post(base_url + "/ask", {"q": "what is the sharpe", "internal": "1"})
    assert status == 200 and answer["lines"] and len(answer["answer_id"]) == 12
    status, ok = _post(base_url + "/feedback", {"id": answer["answer_id"], "useful": "1"})
    assert status == 200 and ok == {"ok": True}
    assert _post(base_url + "/feedback", {"id": "nope", "useful": "1"})[0] == 400
    out = capfd.readouterr().out
    events = [json.loads(x) for x in out.splitlines() if x.startswith('{"argus_usage"')]
    assert [e["kind"] for e in events] == ["ask", "feedback"]
    assert events[0]["internal"] is True and events[0]["client"] == "browser"
    assert "sharpe" not in out.split('{"argus_usage"', 1)[1]


def test_mcp_refuses_a_foreign_origin_and_serves_its_own(base_url: str) -> None:
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}).encode()
    host = base_url.removeprefix("http://")
    for origin, status in (("https://evil.example", 403), (f"http://{host}", 200), ("", 200)):
        headers = {"Content-Type": "application/json", **({"Origin": origin} if origin else {})}
        request = urllib.request.Request(base_url + "/mcp", data=body, method="POST",
                                         headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                got = response.status
        except urllib.error.HTTPError as exc:
            got = exc.code
        assert got == status, origin
