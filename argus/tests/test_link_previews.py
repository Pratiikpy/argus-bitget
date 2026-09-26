"""A link to ARGUS pasted into X or a chat renders as a card with the mark, not a bare line.

The handbook makes an X post mandatory, and until 2026-09-26 only three of the console's pages set
a description and none set an image (judge audit). Every page now builds its head with
``design.head``, which carries the title, description, page URL and the preview image.
"""

from __future__ import annotations

import inspect
import threading
from collections.abc import Iterator
from http.server import ThreadingHTTPServer
from urllib.request import urlopen

import pytest

from argus.lui import corrections_page, design, proof_page, server, status_page, task
from argus.lui.server import PAGE, Handler


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


def test_the_preview_image_is_served_as_a_png(base_url: str) -> None:
    with urlopen(f"{base_url}/og.png") as response:
        body = response.read()
        assert response.headers.get("Content-Type") == "image/png"
    assert body.startswith(b"\x89PNG\r\n\x1a\n")
    width, height = int.from_bytes(body[16:20], "big"), int.from_bytes(body[20:24], "big")
    assert (width, height) == (1200, 630)


def test_the_shared_head_carries_a_large_image_card() -> None:
    head = design.head("T", "D", "/proof")
    assert f'<meta property="og:image" content="{design.OG_IMAGE}">' in head
    assert '<meta name="twitter:card" content="summary_large_image">' in head
    assert f'<meta property="og:url" content="{design.PUBLIC_URL}/proof">' in head
    assert '<meta name="description" content="D">' in head


def test_the_console_page_carries_it() -> None:
    assert design.OG_IMAGE in PAGE
    assert PAGE.count("<title>") == 1


@pytest.mark.parametrize(
    "render",
    [task.render_task, proof_page.render, status_page.render, corrections_page.render],
)
def test_every_page_builds_its_head_with_the_shared_one(render: object) -> None:
    source = inspect.getsource(render)  # type: ignore[arg-type]
    assert "design.head(" in source
    assert "<title>" not in source


def test_the_image_ships_with_the_package() -> None:
    assert server.OG_PNG.is_file()
