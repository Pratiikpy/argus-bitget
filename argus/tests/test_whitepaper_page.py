"""The whitepaper and the proof deck, served from the site in the console's own tokens."""

from __future__ import annotations

from pathlib import Path

from argus.lui import whitepaper_page as wp

ROOT = Path(__file__).resolve().parents[1]


def test_the_served_copy_is_the_repository_copy() -> None:
    # GitHub readers get argus/WHITEPAPER.md; the hosted bundle serves the packaged copy
    assert wp.WHITEPAPER.read_bytes() == (ROOT / "WHITEPAPER.md").read_bytes()


def test_the_page_renders_every_section_with_contents() -> None:
    page = wp.render_whitepaper()
    assert page.startswith("<!doctype html>") and "<h1" in page
    assert '<nav class="toc" aria-label="Contents">' in page
    assert page.count("<h2 ") >= 6
    assert "The whitepaper could not be read on this server" not in page


def test_the_deck_is_served_whole() -> None:
    deck = wp.render_deck()
    assert deck.lstrip().lower().startswith("<!doctype html>")
    assert "The deck could not be read on this server" not in deck


def test_the_renderer_handles_what_the_paper_uses() -> None:
    body, toc = wp.render_markdown(
        "# T\n\n## 1. Problem\n\nA **bold** and *em* `code` [link](https://x.y).\n\n"
        "| a | b |\n|---|---|\n| 1 | 2 |\n\n- one\n- two\n\n```\nx < y\n```\n")
    assert '<h2 id="1-problem">1. Problem</h2>' in body
    assert "<strong>bold</strong>" in body and "<em>em</em>" in body
    assert "<code>code</code>" in body and '<a href="https://x.y">link</a>' in body
    assert "<td>1</td><td>2</td>" in body and "<li>one</li>" in body
    assert "x &lt; y" in body and toc == [(2, "1. Problem", "1-problem")]


def test_both_are_linked_from_the_site() -> None:
    from argus.lui import design

    assert ("/whitepaper", "Whitepaper") in design.LINKS
    assert 'href="/deck"' in wp.render_whitepaper()


def test_the_video_page_plays_the_served_file() -> None:
    page = wp.render_video()
    assert f'src="{wp.VIDEO_SRC}"' in page and "<video" in page and "controls" in page
