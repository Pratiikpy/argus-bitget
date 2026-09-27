"""Demo-surface tests.

A broken demo is an invalidator on both tracks, so the properties here are the ones that would
actually break it in front of a judge rather than the ones that are pleasant to assert.

Two of these tests exist because the failures happened, on the first run of this server:

- the startup banner contained ``→`` and raised ``UnicodeEncodeError`` on a Windows cp1252
  console, so the demo did not start at all;
- the Content-Security-Policy header was ``default-src 'self'``, which covers ``script-src`` and so
  blocked the page's own inline script. The shell rendered and none of the behaviour did.

Both are invisible to a unit test that only calls the handler function, which is why these drive
the HTTP surface.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from datetime import UTC, datetime
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

import pytest

from argus.lui.server import PAGE, Handler, handle_ask


@pytest.fixture(scope="module")
def base_url() -> Iterator[str]:
    """A real server on an ephemeral port.

    Port 0 lets the OS choose, so a parallel run cannot collide with a fixed port.
    """
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()


def _get(url: str) -> tuple[int, bytes, dict[str, str]]:
    with urlopen(url) as response:
        return response.status, response.read(), dict(response.headers)


def _ask(base_url: str, question: str, turns: list[str] | None = None) -> dict[str, Any]:
    query = urlencode({"q": question, "turns": json.dumps(turns or [])})
    _, body, _ = _get(f"{base_url}/ask?{query}")
    result: dict[str, Any] = json.loads(body)
    return result


class TestTheDemoStarts:
    """Everything here failed once. None of it is hypothetical."""

    def test_console_output_is_ascii_only(self) -> None:
        """A Windows cp1252 console raises on anything else, and a demo that cannot print its own
        banner never gets as far as serving a page."""
        import inspect

        from argus.lui import cli, server

        for module in (server, cli):
            source = inspect.getsource(module)
            for lineno, line in enumerate(source.splitlines(), 1):
                if "print(" not in line:
                    continue
                offenders = sorted({c for c in line if ord(c) > 127})
                assert not offenders, f"{module.__name__}:{lineno} prints non-ASCII {offenders}"

    def test_the_policy_admits_the_page_s_own_inline_script(self, base_url: str) -> None:
        """`default-src 'self'` alone blocks an inline <script>, and the page is one file."""
        _, _, headers = _get(base_url + "/")
        policy = headers["Content-Security-Policy"]
        assert "script-src 'unsafe-inline'" in policy
        assert "style-src 'unsafe-inline'" in policy
        assert "default-src 'self'" in policy, "external origins must still be refused"

    def test_the_page_loads_nothing_from_any_other_origin(self) -> None:
        """The CSP allows inline; this asserts we never come to rely on more than that.

        **One narrow, understood exception, added 2026-09-22 with the favicon fix.** The inline
        SVG favicon's `xmlns='http://www.w3.org/2000/svg'` contains the substring "http://", and
        it is never fetched — it is the standard XML namespace identifier every standalone SVG
        document declares, required for the `data:` URI to render as SVG rather than as opaque
        XML. `test_favicon.py` (new) checks the favicon specifically; this test's job is unchanged
        for everything else on the page.
        """
        # **Two more understood exceptions, added 2026-09-24 with the brand redesign.** The two
        # typefaces load from Google Fonts — its stylesheet host and its font-file host, the only
        # external origins the CSP now names, for styles and fonts only — and the nav links to
        # the public GitHub repository, which is a link a reader follows, not a load. Every URL on
        # the page must be one of these; anything else fails.
        import re

        # **And the page's own public address, added 2026-09-26 with the link-preview card.**
        # og:url and og:image name the hosted console and its /og.png so a pasted link renders a
        # card; a crawler reads them, the page itself never fetches them.
        from argus.lui.design import PUBLIC_URL

        allowed = ("https://fonts.googleapis.com", "https://fonts.gstatic.com",
                   "https://github.com/Pratiikpy/argus-bitget", "http://www.w3.org/2000/svg",
                   PUBLIC_URL)
        urls = re.findall(r"https?://[^\s'\"<>)]+", PAGE)
        offenders = [u for u in urls if not u.startswith(allowed) and "127.0.0.1" not in u]
        assert not offenders, f"page references an external origin: {offenders[:3]}"
        assert 'src="//' not in PAGE

    def test_the_page_is_served_whole(self, base_url: str) -> None:
        status, body, headers = _get(base_url + "/")
        assert status == 200
        assert headers["Content-Type"].startswith("text/html")
        assert body.decode().rstrip().endswith("</html>")
        assert int(headers["Content-Length"]) == len(body)

    def test_status_reports_the_real_chain_state(
        self, base_url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A ledger with known contents, read through the endpoint: the count, the newest
        decision and the chain's state must be that ledger's, and an edit to a past decision must
        turn the reading to broken. Until 2026-09-27 this test asked only that the fields had the
        right types, which a hard-coded ``{"entries": 0, "chain_intact": true}`` passed (audit
        finding 146)."""
        from datetime import timedelta
        from decimal import Decimal

        from argus.lui import server
        from argus.paper.ledger import PaperLedger

        path = tmp_path / "paper.jsonl"
        ledger = PaperLedger(path=path)
        start = datetime(2026, 9, 21, 14, tzinfo=UTC)
        for n in range(3):
            ledger.record(
                symbol="NVDAUSDT", verdict="trade", side="BUY", quantity=Decimal("10"),
                entry_price=Decimal("180"), stated_confidence=0.6, thesis="test thesis",
                invalidation=("test invalidation",), market_state_hash="a" * 8,
                approved_intent_hash="b" * 8, session_phase="weekend",
                hours_to_discovery=30.0, decided_at=start + timedelta(hours=n))
        monkeypatch.setattr(server, "_ledger_path", lambda: path)

        payload = json.loads(_get(base_url + "/status")[1])
        assert payload["entries"] == 3
        assert payload["chain_intact"] is True
        assert payload["newest_decision_at"] == (start + timedelta(hours=2)).isoformat()
        assert payload["stale"] is True

        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        rows[0]["thesis"] = "a thesis written after the fact"
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        tampered = json.loads(_get(base_url + "/status")[1])
        assert tampered["entries"] == 3
        assert tampered["chain_intact"] is False


class TestTheAskEndpoint:
    def test_a_real_question_answers_with_sources(self, base_url: str) -> None:
        # A question with no time word, so the answer does not depend on today's date. It asked
        # "why did you do nothing all weekend" until 2026-09-27; that answer moves back at most
        # eight weekends to find a decision, so it failed CI on a Saturday morning and would have
        # failed on every clone once the shipped record was two months old (audit, 2026-09-26).
        payload = _ask(base_url, "how many decisions are on record")
        assert payload["refused"] is False
        assert payload["lines"]
        assert payload["sources"], "an assertion with no source is a defect"
        assert payload["elapsed_ms"] <= payload["budget_ms"]

    def test_a_weekend_question_is_answered_against_the_record_not_the_calendar(self) -> None:
        """The same weekend question, with the clock set an hour after the shipped record's newest
        decision: it answers whatever day the suite runs on."""
        from datetime import timedelta

        from argus.lui.server import _ledger_path
        from argus.paper.ledger import PaperLedger

        newest = max(datetime.fromisoformat(e.decided_at)
                     for e in PaperLedger(path=_ledger_path()).entries)
        payload = handle_ask("why did you do nothing all weekend", [],
                             now=newest + timedelta(hours=1))
        assert payload["refused"] is False, payload
        assert payload["sources"]

    def test_a_refusal_is_carried_as_a_refusal_not_an_error(self, base_url: str) -> None:
        payload = _ask(base_url, "what is XYZQ trading at")
        assert payload["refused"] is True
        assert "Bitget" in payload["reason"]

    def test_an_empty_question_is_rejected_rather_than_answered(self, base_url: str) -> None:
        import urllib.error

        with pytest.raises(urllib.error.HTTPError) as caught:
            _get(base_url + "/ask?q=%20")
        assert caught.value.code == 400

    def test_an_unknown_route_is_a_clean_404(self, base_url: str) -> None:
        import urllib.error

        with pytest.raises(urllib.error.HTTPError) as caught:
            _get(base_url + "/nope")
        assert caught.value.code == 404


class TestConversationIsHeldByTheClient:
    """Server-side session state would let two readers resolve each other's "that"."""

    def test_turns_come_back_so_the_client_can_return_them(self, base_url: str) -> None:
        first = _ask(base_url, "show me decision 25")
        assert first["turns"] == ["show me decision 25"]

    def test_a_reference_resolves_when_the_client_replays_its_turns(self, base_url: str) -> None:
        first = _ask(base_url, "show me decision 25")
        second = _ask(base_url, "what evidence backed that", first["turns"])
        assert second["intent"] == "evidence"
        assert not second["refused"]

    def test_the_same_question_without_history_is_ambiguous(self, base_url: str) -> None:
        """Proof that the resolution came from the replayed turns, not from server memory."""
        cold = _ask(base_url, "what evidence backed that", [])
        assert cold["intent"] == "ambiguous"

    def test_malformed_history_degrades_instead_of_failing(self, base_url: str) -> None:
        _, body, _ = _get(base_url + "/ask?" + urlencode({"q": "what is my position",
                                                          "turns": "not-json"}))
        assert json.loads(body)["refused"] is False

    def test_history_is_bounded_so_a_client_cannot_grow_it_without_limit(self) -> None:
        payload = handle_ask("what is my position", [f"question {i}" for i in range(200)])
        assert len(payload["turns"]) <= 12


class TestTheSurfaceIsReadOnly:
    def test_an_instruction_is_refused_over_http_too(self, base_url: str) -> None:
        payload = _ask(base_url, "sell half of that")
        assert payload["refused"] is True
        assert payload["intent"] == "order"

    def test_the_handler_exposes_no_write_verb(self) -> None:
        """No PUT, PATCH or DELETE handler exists. POST exists for one reason — the Model Context
        Protocol sends its JSON-RPC by POST (`lui/mcp_server.py`, 2026-09-25) — and it is held to
        the same property this test always guarded: nothing it can reach writes or trades."""
        for verb in ("do_PUT", "do_PATCH", "do_DELETE"):
            assert not hasattr(Handler, verb), f"{verb} would be a write path"

    def test_post_is_answered_only_at_mcp_the_webhook_and_the_page_routes(
            self, base_url: str) -> None:
        """Since 2026-09-25 the page also POSTs ``/ask``, ``/translate`` and ``/feedback`` — the
        same read-only answers, sent as a body so the host's access log never holds what a
        visitor typed (`lui/usage.py`). Every other path still refuses POST."""
        import urllib.error
        import urllib.request

        request = urllib.request.Request(base_url + "/status", data=b"{}", method="POST")
        try:
            urllib.request.urlopen(request, timeout=10)
            raise AssertionError("POST /status was accepted")
        except urllib.error.HTTPError as exc:
            assert exc.code == 405
        request = urllib.request.Request(base_url + "/ask", data=b"q=", method="POST")
        try:
            urllib.request.urlopen(request, timeout=10)
            raise AssertionError("POST /ask with no question was answered")
        except urllib.error.HTTPError as exc:
            assert exc.code == 400
        # The webhook answers nothing without Telegram's secret header (503 when the bot is not
        # configured on this server at all, 403 when it is and the secret is wrong).
        request = urllib.request.Request(base_url + "/telegram", data=b"{}", method="POST")
        try:
            urllib.request.urlopen(request, timeout=10)
            raise AssertionError("POST /telegram without the secret was accepted")
        except urllib.error.HTTPError as exc:
            assert exc.code in (403, 503)

    def test_an_order_through_telegram_is_refused_like_anywhere_else(self) -> None:
        from argus.lui.telegram_bot import ChatState, handle_update

        states: dict[int, ChatState] = {}
        replies = handle_update({"message": {"chat": {"id": 7}, "text": "sell half of NVDA now"}},
                                states)
        assert replies and "does not place" in replies[0][1]

    def test_no_mcp_tool_writes_and_an_order_through_it_is_refused(self) -> None:
        from argus.lui.mcp_server import TOOLS, call_tool

        for tool in TOOLS:
            words = f"{tool['name']} {tool['description']}".lower()
            assert not any(w in words for w in ("place an order", "submit", "cancel", "delete"))
        text, is_error = call_tool("argus_ask", {"question": "sell half of NVDA now"})
        assert is_error and "does not place" in text



def _post(base_url: str, path: str, body: bytes,
          headers: dict[str, str]) -> tuple[int, dict[str, Any]]:
    """A POST with the headers exactly as given: urllib would fix a bad Content-Length itself."""
    import http.client
    from urllib.parse import urlparse

    where = urlparse(base_url)
    conn = http.client.HTTPConnection(where.hostname, where.port, timeout=10)
    try:
        conn.putrequest("POST", path)
        for name, value in headers.items():
            conn.putheader(name, value)
        conn.endheaders(body)
        response = conn.getresponse()
        return response.status, json.loads(response.read() or b"{}")
    finally:
        conn.close()


class TestAPostBodyIsReadWhateverItsShape:
    """Found testing the hosted entry point (2026-09-27): a JSON body at /ask answered "empty
    question", and a Content-Length that was not a number dropped the connection unanswered."""

    def test_a_json_question_is_answered_like_a_form(self, base_url: str) -> None:
        body = json.dumps({"q": "sell half of that", "turns": []}).encode()
        status, payload = _post(base_url, "/ask", body, {
            "Content-Type": "application/json", "Content-Length": str(len(body))})
        assert status == 200
        assert payload["intent"] == "order" and payload["refused"] is True

    @pytest.mark.parametrize("body", [b"[1, 2]", b"{not json", b'"q"'])
    def test_a_json_body_that_is_not_an_object_is_a_400_that_says_so(
            self, base_url: str, body: bytes) -> None:
        status, payload = _post(base_url, "/ask", body, {
            "Content-Type": "application/json", "Content-Length": str(len(body))})
        assert status == 400 and "JSON body must be an object" in payload["error"]

    @pytest.mark.parametrize("length", ["abc", "-5"])
    @pytest.mark.parametrize("path", ["/ask", "/research", "/mcp", "/telegram"])
    def test_a_bad_content_length_is_a_400_on_every_post_route(
            self, base_url: str, path: str, length: str) -> None:
        status, payload = _post(base_url, path, b"q=x", {
            "Content-Type": "application/x-www-form-urlencoded", "Content-Length": length})
        assert status == 400 and "Content-Length" in payload["error"]

class TestThePageHoldsUpAtPhoneWidthAndInBothThemes:
    """Verified in a real browser on 2026-09-12 and pinned here so it cannot regress.

    Measured in a 400px iframe: no horizontal overflow, no element wider than the viewport, the
    search bar wraps rather than squashing, and an 18px side gutter. Contrast measured on the
    computed colours: body 17.1:1 light / 15.3:1 dark, muted text 5.6:1 light / 7.2:1 dark — both
    clear WCAG AA. These tests guard the CSS properties those measurements depend on.
    """

    def test_every_colour_token_is_defined_in_the_light_palette(self) -> None:
        """A token whose only definition sits inside the dark media block is undefined in light
        mode, which is the classic unreadable-page bug."""
        import re

        head, _, _unused = PAGE.partition("@media (prefers-color-scheme: dark)")
        bare_root = head[head.index(":root"):]
        defined = set(re.findall(r"(--[\w-]+)\s*:", bare_root))
        used = set(re.findall(r"var\((--[\w-]+)", PAGE))
        assert used <= defined, f"used only in dark: {sorted(used - defined)}"

    def test_the_dark_block_only_redefines_and_never_introduces(self) -> None:
        import re

        head, marker, dark = PAGE.partition("@media (prefers-color-scheme: dark)")
        assert marker, "the dark palette must exist"
        light = set(re.findall(r"(--[\w-]+)\s*:", head[head.index(":root"):]))
        in_dark = set(re.findall(r"(--[\w-]+)\s*:", dark[: dark.index("}")]))
        assert in_dark <= light, f"introduced only in dark: {sorted(in_dark - light)}"

    def test_the_body_paints_its_own_background(self) -> None:
        """A transparent body borrows the host's ground and inverts in the opposite theme."""
        assert "background:var(--bg)" in PAGE.replace(" ", "")

    def test_the_layout_can_wrap_rather_than_overflow(self) -> None:
        assert "flex-wrap:wrap" in PAGE.replace(" ", "")

    def test_the_input_may_shrink_below_its_basis(self) -> None:
        """Without min-width:0 a flex item refuses to shrink and forces a sideways scroll."""
        assert "min-width:0" in PAGE.replace(" ", "")

    def test_there_is_a_side_gutter_at_every_width(self) -> None:
        """The gutter is the base stylesheet's, which every page appends last; the console's own
        `.wrap` padding never applied and was removed (audit finding 73)."""
        from argus.lui.design import BASE_CSS

        assert BASE_CSS in PAGE
        assert ".wrap { max-width:1080px; margin:0 auto; padding:40px 24px 72px }" in BASE_CSS
        assert ".wrap { padding:28px 16px 56px }" in BASE_CSS

    def test_the_viewport_meta_is_present(self) -> None:
        assert 'name="viewport"' in PAGE and "width=device-width" in PAGE


class TestTheResearchTaskIsReachable:
    """**Track 3's required demo: one complete research task, question to actionable insight.**

    The route used to render a chain recorded on 2026-09-14 whose verdict its own allocation step
    contradicted. It now runs the task on request; these tests hold it to the three properties the
    handbook line asks for — the whole flow, an actionable end, and reachable by a judge.
    """

    @pytest.mark.network
    def test_the_page_runs_the_whole_chain(self, base_url: str) -> None:
        status, raw, headers = _get(base_url + "/research")
        body = raw.decode("utf-8")
        assert status == 200
        assert "text/html" in headers.get("Content-Type", "")
        assert "should I add 15% TSLA?" in body
        assert "What to do" in body
        assert body.count("<article") == 8

    @pytest.mark.network
    def test_every_step_names_its_engine_and_answers(self, base_url: str) -> None:
        _, body, _ = _get(base_url + "/research?format=json")
        payload = json.loads(body)
        assert len(payload["steps"]) == 8
        for step in payload["steps"]:
            assert step["engine"] and step["lines"], step

    @pytest.mark.network
    def test_it_ends_in_something_to_act_on(self, base_url: str) -> None:
        """The conclusion leads with what was asked: how much of the name the book can carry."""
        _, body, _ = _get(base_url + "/research?format=json")
        conclusion = json.loads(body)["conclusion"]
        assert conclusion, "a research task with no actionable end is not the task asked for"
        assert conclusion[0]["step"] == "What the trade does to your book"

    @pytest.mark.network
    def test_the_name_size_and_book_come_from_the_url(self, base_url: str) -> None:
        _, body, _ = _get(base_url + "/research?format=json&name=gold&size=10&book=50%25%20SPY"
                                     "%2C%2050%25%20QQQ")
        payload = json.loads(body)
        assert payload["name"] == "XAU"
        assert payload["size_pct"] == 10
        assert set(payload["book"]) == {"SPYUSDT", "QQQUSDT"}

    @pytest.mark.network
    def test_nonsense_parameters_still_answer(self, base_url: str) -> None:
        status, body, _ = _get(base_url + "/research?format=json&size=lots&name=")
        assert status == 200
        assert json.loads(body)["name"] == "TSLA"

    def test_the_console_links_to_it(self, base_url: str) -> None:
        """A route nobody can find is the same problem one step removed."""
        _, raw, _ = _get(base_url + "/")
        assert "/research" in raw.decode("utf-8")


class TestTheLossesPage:
    """**A losses page that silently shortens is the most flattering possible lie.**

    Every entry is read from an artefact at request time rather than written down once, so the
    failure mode that matters is not a wrong entry — it is a *missing* one disappearing quietly.
    """

    def test_it_is_reachable_and_linked(self, base_url: str) -> None:
        status, raw, _ = _get(base_url + "/wrong")
        assert status == 200
        assert "What we got wrong" in raw.decode("utf-8")
        _, home, _ = _get(base_url + "/")
        assert "/wrong" in home.decode("utf-8"), "a route nobody can find is not submitted"

    def test_it_records_real_findings(self, base_url: str) -> None:
        _, raw, _ = _get(base_url + "/wrong?format=json")
        rows = json.loads(raw)
        assert len(rows) >= 5
        kinds = {r["kind"] for r in rows}
        assert {"bug", "loss", "withdrawn"} <= kinds, (
            "the page must carry bugs we shipped, comparisons we lost, and claims we withdrew"
        )

    def test_every_entry_names_its_artefact(self, base_url: str) -> None:
        """An unsourced confession is just as unverifiable as an unsourced boast."""
        _, raw, _ = _get(base_url + "/wrong?format=json")
        for row in json.loads(raw):
            assert row["artefact"], row
            assert row["headline"] and row["detail"], row

    def test_a_missing_artefact_is_listed_not_dropped(self, tmp_path) -> None:
        """The one failure mode this page must not have."""
        from argus.lui.corrections_page import collect

        found = collect(tmp_path)
        assert found, "an empty data directory must still produce entries"
        assert any("unreadable" in c.headline for c in found), (
            "a finding whose artefact is gone must say so, not vanish"
        )

    def test_it_does_not_claim_completeness(self, base_url: str) -> None:
        _, raw, _ = _get(base_url + "/wrong")
        assert "not everything wrong" in raw.decode("utf-8").lower()

    def test_not_owned_counts_lost_and_tied_not_only_implemented(self, tmp_path) -> None:
        """Regression: the prior version read only `by_state["implemented"]` for the numerator, so
        a capability that had been measured and demonstrably LOST to a named specialist silently
        vanished from "cannot claim OWNED" instead of being the clearest possible example of it —
        on the one page whose entire purpose is not rounding up. Found 2026-09-22 driving the
        deployed page as a real user: it read "7 of 28" while the live register was "6 of 31"."""
        from argus.lui.corrections_page import collect

        (tmp_path / "standing.json").write_text(
            json.dumps({
                "by_state": {"owned": 2, "implemented": 1, "lost": 1, "tied": 0},
                "findings": [],
            }),
            encoding="utf-8",
        )
        found = collect(tmp_path)
        matches = [c for c in found if "cannot claim OWNED" in c.headline]
        assert len(matches) == 1
        assert "2 of 4 capabilities cannot claim OWNED" in matches[0].headline


class TestStalenessFollowsTheSchedule:
    """The desk decides at 13:30, 15:30, 17:30 and 19:30 UTC. The old 12-hour rule called the
    record stale every morning during the designed overnight gap."""

    def test_the_overnight_gap_is_not_stale(self) -> None:
        from argus.lui.server import _last_scheduled_cycle

        morning = datetime(2026, 9, 23, 10, 0, tzinfo=UTC)
        assert _last_scheduled_cycle(morning) == datetime(2026, 9, 22, 19, 30, tzinfo=UTC)

    def test_a_missed_afternoon_cycle_is_caught(self) -> None:
        from argus.lui.server import _last_scheduled_cycle

        after_two_runs = datetime(2026, 9, 23, 16, 45, tzinfo=UTC)
        assert _last_scheduled_cycle(after_two_runs) == datetime(2026, 9, 23, 15, 30, tzinfo=UTC)

    def test_a_cycle_still_inside_its_grace_is_not_yet_expected(self) -> None:
        from argus.lui.server import _last_scheduled_cycle

        just_after = datetime(2026, 9, 23, 13, 50, tzinfo=UTC)
        assert _last_scheduled_cycle(just_after) == datetime(2026, 9, 22, 19, 30, tzinfo=UTC)

    def test_next_cycle(self) -> None:
        from argus.lui.server import _next_scheduled_cycle

        assert _next_scheduled_cycle(datetime(2026, 9, 23, 20, 0, tzinfo=UTC)) == datetime(
            2026, 9, 24, 13, 30, tzinfo=UTC
        )


class TestAModelReadingIsHeldToTheWords:
    def test_a_leverage_reading_with_no_leverage_named_is_dropped(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The hosted model read "Long MSTR perp into earnings — funding looks cheap" as a 10x
        leverage question (live, 2026-09-25). "Perp" is not leverage."""
        from argus.lui import research, server

        class Planner:
            def complete_json(self, messages: list[dict[str, str]], **_: Any) -> dict[str, Any]:
                return {"kind": "leverage", "names": ["MSTR"], "candidate": "MSTR",
                        "confidence": 0.95, "why": "a perp position"}

        seen: list[Any] = []
        monkeypatch.setattr(server, "_model_for", lambda visitor: Planner())
        monkeypatch.setattr(server, "worth_asking_the_model", lambda text, **_: True)
        monkeypatch.setattr(server, "_research_payload",
                            lambda text, prior, request, *a, **k: seen.append(request) or {})
        handle_ask("Long MSTR perp into earnings — funding looks cheap", [])
        assert seen and seen[0].kind is not research.ResearchKind.LEVERAGE
        seen.clear()
        handle_ask("10x long MSTR over the weekend", [])
        assert seen and seen[0].kind is research.ResearchKind.LEVERAGE

    def test_an_add_the_model_reads_as_a_holding_keeps_the_patterns_add(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The hosted model read "add 5k of coin" as bitcoin with no candidate it could resolve,
        and the answer was about adding NVDA, already held (live, 2026-09-25)."""
        from argus.lui import research, server

        class Planner:
            def complete_json(self, messages: list[dict[str, str]], **_: Any) -> dict[str, Any]:
                return {"kind": "impact", "names": ["NVDA", "MSFT"], "candidate": "coin",
                        "holdings": {"NVDA": 40, "MSFT": 20}, "confidence": 0.85, "why": "add"}

        seen: list[Any] = []
        monkeypatch.setattr(server, "_model_for", lambda visitor: Planner())
        monkeypatch.setattr(server, "worth_asking_the_model", lambda text, **_: True)
        monkeypatch.setattr(server, "_research_payload",
                            lambda text, prior, request, *a, **k: seen.append(request) or {})
        handle_ask("portfolio is nvda 40%, msft 20%, cash rest — want to add 5k of coin, bad "
                   "idea?", [])
        assert seen and seen[0].kind is research.ResearchKind.IMPACT
        assert seen[0].symbols[0] == "COINUSDT"


def test_a_500_names_the_error_class_and_an_incident_but_never_its_message(
        caplog: pytest.LogCaptureFixture) -> None:
    """Audit finding 162: the body carried ``str(exc)``, which can hold a URL with its query or
    a local path. The message now goes to the server log only, under the id the body gives."""
    from argus.lui.server import server_error

    with caplog.at_level("ERROR", logger="argus.lui.server"):
        body = json.loads(server_error(ValueError("https://x.test/?token=s3cret"), "/ask"))
    assert body["error"] == "ValueError"
    assert "s3cret" not in json.dumps(body)
    assert body["incident"] in caplog.text and "s3cret" in caplog.text
