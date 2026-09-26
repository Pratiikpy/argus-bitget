"""A browser that reaches a missing or broken page gets a page with a way back; an API client keeps
getting JSON. Until 2026-09-26 both got ``{"error":"not found"}`` (judge audit)."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from argus.lui import server
from argus.lui.server import Handler

HTML = {"Accept": "text/html,application/xhtml+xml"}


@pytest.fixture
def base_url() -> Iterator[str]:
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}"
    finally:
        httpd.shutdown()
        httpd.server_close()


def _get(url: str, headers: dict[str, str]) -> tuple[int, str, str]:
    try:
        with urlopen(Request(url, headers=headers)) as response:
            kind = response.headers.get("Content-Type", "")
            return response.status, kind, response.read().decode()
    except HTTPError as exc:
        return exc.code, exc.headers.get("Content-Type", ""), exc.read().decode()


def test_a_browser_gets_a_page_for_a_missing_path(base_url: str) -> None:
    status, kind, body = _get(f"{base_url}/judge", HTML)
    assert status == 404
    assert "text/html" in kind
    assert "No page here" in body
    assert "<code>/judge</code>" in body
    assert 'href="/research"' in body


def test_an_api_client_still_gets_json(base_url: str) -> None:
    status, kind, body = _get(f"{base_url}/judge", {"Accept": "application/json"})
    assert status == 404
    assert "application/json" in kind
    assert json.loads(body) == {"error": "not found"}


def test_a_page_that_fails_to_build_says_so_and_keeps_the_way_back(
    base_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(self: Handler, params: dict[str, list[str]]) -> None:
        raise KeyError("summary")

    monkeypatch.setattr(server.Handler, "_research_route", broken)
    status, kind, body = _get(f"{base_url}/research", HTML)
    assert status == 500
    assert "text/html" in kind
    assert "failed while it was being built (KeyError)" in body
    assert 'href="/"' in body


def test_a_missing_path_is_escaped() -> None:
    page = server.error_page(404, "/<script>alert(1)</script>")
    assert "<script>alert" not in page
    assert "&lt;script&gt;" in page
