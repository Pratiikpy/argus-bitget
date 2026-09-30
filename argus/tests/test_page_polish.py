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
from argus.lui.corrections import Correction
from argus.lui.server import PAGE, Handler, mcp_page


class TestTheTopBar:
    def test_the_current_page_is_announced_not_only_coloured(self) -> None:
        """Finding 74: the active link carried a class and nothing a screen reader reads."""
        bar = design.nav("/status")
        assert '<a href="/status" class="on" aria-current="page">' in bar
        # Once in the desktop row and once in the phone menu; one of the two is display:none at
        # any width, so a screen reader meets exactly one.
        assert bar.count('aria-current="page"') == 2

    def test_a_keyboard_visitor_can_skip_the_bar(self) -> None:
        bar = design.nav("/")
        assert bar.startswith('<a class="skip" href="#main">')
        assert bar.endswith('<span id="main" tabindex="-1"></span>')

    def test_materials_is_in_the_bar_and_brand_moved_to_the_footer(self) -> None:
        """Finding 79: the page the submission form links to was missing from the bar."""
        assert ("/materials", "Materials") in design.LINKS
        assert all(path != "/brand" for path, _ in design.LINKS)
        assert 'href="/brand"' in design.footer()

    def test_every_served_page_is_reachable_by_a_link(self) -> None:
        """Fresh-eyes audit, 2026-09-29: four pages answered 200 and nothing linked to them."""
        linked = {path for path, _ in design.LINKS} | {
            href for href in ("/architecture", "/policy", "/factors", "/agent", "/brand")
            if f'href="{href}"' in design.footer()}
        assert {"/", "/research", "/proof", "/wrong", "/status", "/materials", "/architecture",
                "/policy", "/factors", "/agent", "/brand"} <= linked

    def test_the_footer_names_the_track_2_agent_as_its_own_project(self) -> None:
        """First-user audit, 2026-09-29: "Track 2 agent" read as part of this console rather than
        a link away to a separate entry."""
        assert '<a href="/agent">Trading agent (separate project)</a>' in design.footer()
        assert "Track 2 agent" not in design.footer()

    def test_a_phone_gets_a_menu_button_with_every_link(self) -> None:
        """First-user audits, 2026-09-29 and round 11 (2026-09-30): the sideways-scrolling row hid
        four of seven links, and neither a fade nor a chevron over it read as "there is more". On a
        phone the bar is now a Menu disclosure holding every link — a <details>, so it works with
        no script (the proof pages ship none) and is announced as a disclosure."""
        bar = design.nav("/proof")
        menu = bar[bar.index('<details class="menu">'):bar.index("</details>")]
        assert "<summary>Menu</summary>" in menu
        for href, _ in design.LINKS:
            assert f'href="{href}"' in menu
        assert 'href="/proof" class="on" aria-current="page"' in menu
        css = design.BASE_CSS
        assert ".nav .menu { display:none }" in css.split("@media (max-width: 720px)", 1)[0]
        mobile = css.split("@media (max-width: 720px)", 1)[1]
        assert ".nav .links-wrap { display:none }" in mobile
        assert ".nav .menu { display:block" in mobile

    def test_the_menu_meets_wcag_aa_contrast(self) -> None:
        """The menu is drawn in the page's own ink-on-background pair, the same one every heading
        and body line already uses, checked here rather than assumed."""

        def luminance(hex_colour: str) -> float:
            r, g, b = (int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5))

            def lin(c: float) -> float:
                return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

            lr, lg, lb = lin(r), lin(g), lin(b)
            return 0.2126 * lr + 0.7152 * lg + 0.0722 * lb

        def ratio(a: str, b: str) -> float:
            la, lb = luminance(a) + 0.05, luminance(b) + 0.05
            return max(la, lb) / min(la, lb)

        palette = {name: hexcode for name, hexcode, *_ in design.PALETTE}
        assert ratio(palette["Ink"], palette["Paper"]) >= 4.5  # light mode: --ink on --bg
        # Dark mode swaps in near-white ink on near-black paper (design.TOKENS_CSS); Ink/Paper
        # inverted is the same pair read the other way, so the ratio is identical.
        assert ratio(palette["Paper"], palette["Ink"]) >= 4.5


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

    def test_the_record_line_says_what_it_is_before_the_number(self) -> None:
        """First-user audit, 2026-09-29: "836 decisions on record, chain intact" meant nothing to
        a newcomer — a decision by whom, with whose money, verified how. The plain sentence leads;
        the live figures (`${s.entries}`, chain state, age) are unchanged, not retyped."""
        assert "own paper desk" in PAGE
        assert "decides four times a day during US market hours, with " in PAGE
        assert "no real money, and every decision is written into a tamper-evident record" in PAGE
        assert "${s.entries} " in PAGE and "decisions logged so far, chain ${s.chain_intact" in PAGE
        # the age branches this sentence used to carry are still there, just not duplicated
        assert "Age unknown" in PAGE and "a scheduled cycle was missed" in PAGE

    def test_a_long_answer_collapses_behind_a_real_accessible_button(self) -> None:
        """First-user audit, 2026-09-29: a long answer (a research or portfolio question routinely
        runs well past eight lines) landed as one unbroken wall of text. Past eight lines only the
        first six show, behind a real <button> (reachable by keyboard, announced as expandable),
        never a link doing a button's job."""
        assert "divs.slice(0, 6)" in PAGE
        assert '`Show all ${divs.length} lines`' in PAGE
        assert '<button type="button" class="more" aria-expanded="false" aria-controls="${id}"' \
            in PAGE
        # a refusal, or an error, is never the thing this collapses
        assert "if (refused || divs.length <= 8) return divs.join('');" in PAGE
        assert "collapseLines(a.lines.map" in PAGE and "collapseLines(t.lines.map" in PAGE
        assert "a.refused)}</div>" in PAGE  # the initial render passes it
        assert "), a.refused);" in PAGE  # the translate swap passes it too
        # every line, hidden or shown, still carries its own source badge
        assert "pv((a.line_labels || [])[i])" in PAGE


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

    def test_a_browser_gets_a_page_and_a_client_still_gets_the_405(self, base_url: str) -> None:
        assert self._get(f"{base_url}/mcp", "text/html") == (200, "text/html; charset=utf-8")
        assert self._get(f"{base_url}/mcp", "application/json") == (405, "application/json")
        assert self._get(f"{base_url}/mcp", "text/event-stream")[0] == 405

    def test_no_description_ends_in_a_doubled_full_stop(self) -> None:
        assert ".." not in mcp_page("example.test")


class TestTheCorrectionsPage:
    def test_an_artefact_path_is_a_link_to_the_file(self) -> None:
        """Finding 81: /wrong showed the path as dead text while /proof linked it."""
        page = corrections_page.render([Correction("h", "d", "data/x_comparison.json", "bug")])
        assert ("href='https://github.com/Pratiikpy/argus-bitget/blob/main/argus/"
                "data/x_comparison.json'") in page

    def test_a_module_path_links_to_its_file_under_src(self) -> None:
        page = corrections_page.render(
            [Correction("h", "d", "argus/paper/corrections.py", "bug")])
        assert "blob/main/argus/src/argus/paper/corrections.py'" in page

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


class TestTheResearchPage:
    def test_the_weight_bars_are_visible_against_the_page(self) -> None:
        """Finding 71: the money bars were drawn in the hairline colour, about 1.2:1 on white;
        a chart's marks need 3:1 (WCAG 1.4.11). They take the secondary text colour now."""
        from argus.lui.task import unread_task
        from argus.lui.task_page import render_task

        page = render_task(unread_task("?", "no question"), "")
        assert ".chart rect.w, .key.w { fill:var(--dim); background:var(--dim) }" in page
        assert "fill:var(--line)" not in page


class TestTheBrandReceipt:
    def test_the_receipt_is_a_real_run_not_a_drawing(self) -> None:
        """Finding 78: /brand showed an engine path, a rival and a ledger hash no answer carries."""
        import json

        from argus.eval.research_task_record import REPORT_PATH
        from argus.lui import brand_page

        blob = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        page = brand_page.render()
        assert "seq 412 · 9f2c…a71e</dd>" not in page and "weekend-copilot" not in page
        assert f"<strong>{blob['task']['verdict']['call']}</strong>" in page
        assert "data/research_task_example.json" in page


class TestTheBaseStylesheet:
    """Finding 73: every page appends `design.BASE_CSS` after its own rules, so BASE wins a tie,
    and five pages carried `.wrap`, `h1`, `.sub`, `button` and `.card` values that never applied.
    The dead values were removed; this keeps a page from restating one BASE overrides."""

    @staticmethod
    def _rules(css: str) -> dict[str, dict[str, str]]:
        import re

        css = re.sub(r"@media[^{]*\{(?:[^{}]*\{[^{}]*\})*[^{}]*\}", "", css)
        out: dict[str, dict[str, str]] = {}
        for selectors, body in re.findall(r"([^{}]+)\{([^{}]*)\}", css):
            for selector in selectors.split(","):
                props = out.setdefault(" ".join(selector.split()), {})
                for decl in body.split(";"):
                    if ":" in decl:
                        name, value = decl.split(":", 1)
                        props[name.strip()] = value.strip()
        return out

    def test_no_page_restates_a_value_the_base_overrides(self) -> None:
        import re
        from pathlib import Path

        from argus.lui import design

        base = self._rules(design.BASE_CSS)
        clashes = []
        for path in sorted(Path(design.__file__).parent.glob("*.py")):
            source = path.read_text(encoding="utf-8")
            for block in re.findall(r"<style>(.*?)</style>", source, flags=re.S):
                if not re.search(r"(\{design\.BASE_CSS\}|__BASE__)\s*$", block):
                    continue  # BASE first (or absent): the page's own rules win, as written
                css = block.replace("{{", "{").replace("}}", "}")
                css = re.sub(r"\{design\.[A-Z_]+\}|__[A-Z]+__", "", css)
                for selector, props in self._rules(css).items():
                    for name, value in props.items():
                        if base.get(selector, {}).get(name, value) != value:
                            clashes.append(f"{path.name}: {selector} {{ {name}:{value} }}")
        assert clashes == []


class TestTheResearchFormSendsTheSavedBook:
    """A question asked on /research read the page's visible example book as "your saved book";
    the form now sends the browser's saved book and memory as their own fields."""

    @pytest.fixture
    def base_url(self) -> object:
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        yield f"http://127.0.0.1:{server.server_port}"
        server.shutdown()
        server.server_close()

    def test_the_saved_field_wins_over_the_visible_book(
            self, base_url: str, monkeypatch: pytest.MonkeyPatch) -> None:
        from urllib.parse import urlencode

        from argus.lui import task

        seen: dict[str, str] = {}

        def read(asked: str, saved: str = "") -> str:
            seen["saved"] = saved
            return "not read in this test"

        monkeypatch.setattr(task, "read_question", read)
        # An unread question now runs as the console answers it (`task.question_task`); this test
        # is about which book is read, so that answer is stubbed.
        monkeypatch.setattr(task, "question_task", lambda *a, **k: None)
        body = urlencode({"q": "should I add 15% TSLA?", "saved": "",
                          "memory": "[]", "book": task.DEFAULT_BOOK}).encode()
        with urlopen(Request(f"{base_url}/research", data=body), timeout=10) as reply:
            assert reply.status == 200
        assert seen["saved"] == ""

    def test_the_page_fills_both_fields_from_this_browser(self) -> None:
        from argus.lui.task import unread_task
        from argus.lui.task_page import render_task

        page = render_task(unread_task("?", "no question"), "")
        assert '<input type="hidden" name="saved"><input type="hidden" name="memory">' in page
        assert "localStorage.getItem('argus.book')" in page
        assert "localStorage.getItem('argus.memory')" in page
