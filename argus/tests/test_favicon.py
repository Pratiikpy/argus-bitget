"""The inline SVG favicon — added 2026-09-22 after driving the live console as a first-time judge
would and finding one console error: a bare `/favicon.ico` 404, the only thing wrong on an
otherwise clean page. Fixing it introduced a second, different console error before this test
suite existed (the CSP's `default-src` fallback blocked the `data:` URI) — caught by actually
loading the page in a real browser and reading its console a second time, not assumed fixed from
the HTML alone. Both regressions are pinned here so neither can silently return.
"""

from __future__ import annotations

import threading
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from http.server import ThreadingHTTPServer
from urllib.parse import unquote
from urllib.request import urlopen

import pytest

from argus.lui.corrections_page import FAVICON as WRONG_PAGE_FAVICON
from argus.lui.corrections_page import render as render_wrong
from argus.lui.server import FAVICON, PAGE, Handler
from argus.lui.task import Step, Task, render_task


@pytest.fixture(scope="module")
def base_url() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()


class TestTheFaviconIsValid:
    def test_it_is_a_data_uri_svg(self) -> None:
        assert FAVICON.startswith("data:image/svg+xml,")

    def test_the_decoded_svg_parses_as_xml(self) -> None:
        decoded = unquote(FAVICON.removeprefix("data:image/svg+xml,"))
        ET.fromstring(decoded)  # raises ET.ParseError on malformed markup

    def test_it_contains_no_double_quote(self) -> None:
        """Every page embeds this inside `href="..."` — a literal `"` would truncate the tag,
        the same class of bug the single-quote version of this mistake already caused once."""
        assert '"' not in FAVICON

    def test_the_two_module_copies_stay_identical(self) -> None:
        """`corrections_page.py` deliberately duplicates this constant rather than importing it
        (see its own docstring) — the two are not the same object in memory, so nothing else
        would catch them drifting apart."""
        assert FAVICON == WRONG_PAGE_FAVICON


class TestEveryPageTemplateCarriesIt:
    def test_the_main_console_page(self) -> None:
        assert f'href="{FAVICON}"' in PAGE

    def test_the_research_task_page(self) -> None:
        task = Task(question="q", name="TSLA", size_pct=15, book={},
                    steps=[Step(title="t", engine="e", lines=["Actionable: x"])], seconds=0.1)
        assert f'href="{FAVICON}"' in render_task(task, FAVICON)

    def test_the_wrong_page(self) -> None:
        assert f'href="{FAVICON}"' in render_wrong([])


class TestTheRealServerServesItWithoutACspViolation:
    def test_img_src_admits_data_uris(self, base_url: str) -> None:
        """The regression this whole file exists to catch: a `data:` favicon added without also
        widening the CSP's `img-src` is fixed on paper and broken in every real browser."""
        with urlopen(base_url + "/") as response:
            policy = dict(response.headers)["Content-Security-Policy"]
        assert "img-src 'self' data:" in policy

    def test_default_src_self_is_unweakened(self, base_url: str) -> None:
        """The one property `img-src 'self' data:` must not cost: every other resource type is
        still refused from any origin but this server's own."""
        with urlopen(base_url + "/") as response:
            policy = dict(response.headers)["Content-Security-Policy"]
        assert "default-src 'self'" in policy
