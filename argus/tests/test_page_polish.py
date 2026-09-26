"""The judge-facing details a UI audit found missing (2026-09-26), pinned so they stay fixed.

Each test names the finding it closes. None needs the network: the pages are rendered from their
own functions, and the one live probe is replaced by a stub that never answers.
"""

from __future__ import annotations

import threading
import time
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from argus.lui import corrections_page, design, status_page
from argus.lui.corrections_page import Correction
from argus.lui.server import PAGE, Handler, mcp_page


class TestTheTopBar:
    def test_the_current_page_is_announced_not_only_coloured(self) -> None:
        """Finding 74: the active link carried a class and nothing a screen reader reads."""
        bar = design.nav("/status")
        assert '<a href="/status" class="on" aria-current="page">' in bar
        assert bar.count('aria-current="page"') == 1

    def test_a_keyboard_visitor_can_skip_the_bar(self) -> None:
        bar = design.nav("/")
        assert bar.startswith('<a class="skip" href="#main">')
        assert bar.endswith('<span id="main" tabindex="-1"></span>')

    def test_materials_is_in_the_bar_and_brand_moved_to_the_footer(self) -> None:
        """Finding 79: the page the submission form links to was missing from the bar."""
        assert ("/materials", "Materials") in design.LINKS
        assert all(path != "/brand" for path, _ in design.LINKS)
        assert 'href="/brand"' in design.footer()


class TestTheConsoleCard:
    def test_a_correct_answer_is_not_stamped_over_budget(self) -> None:
        """Findings 59 and 77: telemetry in the warning colour read as a failed answer."""
        assert "OVER BUDGET'" not in PAGE
        assert "answered in ${(a.elapsed_ms / 1000).toFixed(1)} s" in PAGE

    def test_answers_are_announced_as_they_arrive(self) -> None:
        assert '<div id="out" role="log" aria-live="polite"' in PAGE

    def test_a_translation_declares_its_language(self) -> None:
        """Finding 86: translated lines were rendered under lang="en"."""
        assert "lines.lang = a.translate.lang;" in PAGE


class TestTheMcpEndpointInABrowser:
    """Finding 80: /materials links to /mcp, and a browser that followed it met a JSON error."""

    def test_the_page_lists_the_tools_and_how_to_connect(self) -> None:
        from argus.lui.mcp_server import TOOLS

        page = mcp_page("example.test")
        assert "https://example.test/mcp" in page
        assert all(f"<code>{t['name']}</code>" in page for t in TOOLS)

    @pytest.fixture
    def base_url(self) -> object:
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        yield f"http://127.0.0.1:{server.server_port}"
        server.shutdown()
        server.server_close()

    def _get(self, url: str, accept: str) -> tuple[int, str]:
        try:
            with urlopen(Request(url, headers={"Accept": accept}), timeout=10) as r:
                return r.status, r.headers.get("Content-Type", "")
        except HTTPError as exc:
            return exc.code, exc.headers.get("Content-Type", "")

    def test_a_browser_gets_html_and_a_client_still_gets_json(self, base_url: str) -> None:
        assert self._get(f"{base_url}/mcp", "text/html") == (405, "text/html; charset=utf-8")
        assert self._get(f"{base_url}/mcp", "application/json") == (405, "application/json")


class TestTheCorrectionsPage:
    def test_an_artefact_path_is_a_link_to_the_file(self) -> None:
        """Finding 81: /wrong showed the path as dead text while /proof linked it."""
        page = corrections_page.render([Correction("h", "d", "data/x_comparison.json", "bug")])
        assert ("href='https://github.com/Pratiikpy/argus-bitget/blob/main/argus/"
                "data/x_comparison.json'") in page

    def test_something_that_is_not_a_path_stays_plain_text(self) -> None:
        page = corrections_page.render([Correction("h", "d", "the register", "open")])
        assert "<code>the register</code>" in page and "the register</code></a>" not in page


class TestTheStatusPage:
    def test_the_live_table_has_column_headers(self) -> None:
        """Finding 76: five bare cells per row and no header a screen reader can announce."""
        check = status_page.Check("Bitget market API", "all tickers", True, "ok", 120.0)
        page = status_page.render({"entries": 1, "chain_intact": True}, [check], 0.0, [], "")
        assert '<th scope="col">Surface</th>' in page
        assert "data-label='Result'" in page

    def test_a_probe_that_never_answers_cannot_hold_the_page(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Finding 75: leaving the probe pool's `with` block waited for every probe."""
        release = threading.Event()

        def hangs() -> str:
            release.wait(30)
            return "late"

        monkeypatch.setattr(status_page, "CHECKS", (("stub", "never answers", hangs),))
        monkeypatch.setattr(status_page, "PROBE_TIMEOUT_S", 0.5)
        monkeypatch.setattr(status_page, "_CACHE", {})
        began = time.monotonic()
        try:
            checks, _ = status_page.live_checks(force=True)
        finally:
            release.set()
        assert time.monotonic() - began < 5
        assert [c.ok for c in checks] == [False]
        assert "no answer within" in checks[0].detail
