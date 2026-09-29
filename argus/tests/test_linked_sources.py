"""Addresses in answers are links (first-user audit, 2026-09-29: every citation was plain text)."""

from __future__ import annotations

from argus.lui.design import linked


def test_an_address_becomes_a_link_reading_its_host() -> None:
    out = linked("Bitcoin beats gold (coindesk) https://www.coindesk.com/daybook-us/2026/09/29/x.")
    assert ('<a href="https://www.coindesk.com/daybook-us/2026/09/29/x" rel="noopener noreferrer" '
            'target="_blank">coindesk.com &#8599;</a>.') in out


def test_text_around_it_is_still_escaped_and_the_query_survives() -> None:
    out = linked("<script>x</script> see https://sec.gov/a?b=1&c=2)")
    assert out.startswith("&lt;script&gt;x&lt;/script&gt; see ")
    assert 'href="https://sec.gov/a?b=1&amp;c=2"' in out and out.endswith("</a>)")


def test_a_quote_cannot_break_out_of_the_attribute() -> None:
    assert '"' not in linked('https://x.com/a"onmouseover="alert(1)').split('href="')[1].split(
        '" rel')[0]


def test_text_without_an_address_is_only_escaped() -> None:
    assert linked("P/E 28.6 & rising") == "P/E 28.6 &amp; rising"


def test_the_console_renders_answer_lines_and_receipts_through_the_same_rule() -> None:
    from argus.lui import server

    page = server.PAGE
    assert "const linked = s => esc(s)" in page
    assert "${linked(l)}" in page and "${linked(s.ref)}" in page


def test_a_question_in_flight_stays_on_screen_with_a_running_count() -> None:
    """First-user audit, 2026-09-29: a slow answer showed only a greyed-out button."""
    from argus.lui import server

    assert 'class="card pending" id="pending"' in server.PAGE
    assert "clearInterval(ticker)" in server.PAGE and "pending.remove()" in server.PAGE


def test_the_suggestions_are_buttons_a_keyboard_can_reach() -> None:
    """First-user audit, 2026-09-29: the 25 suggestion chips were spans, mouse-only."""
    from argus.lui import server

    assert '<button type="button" class="chip">' in server.PAGE
    assert '<span class="chip">' not in server.PAGE


def test_basis_points_carry_their_definition() -> None:
    """"bps" was on every cost line with no definition anywhere on the page (a first-user audit,
    2026-09-29): a figure in basis points says what they are on hover, and a word is left alone."""
    out = linked("a 12bps round trip; the bps word alone")
    assert out.count("<abbr") == 1
    assert "100bps is 1%" in out


def test_the_console_defines_basis_points_the_same_way() -> None:
    from argus.lui.server import PAGE

    assert "100bps is 1%" in PAGE
