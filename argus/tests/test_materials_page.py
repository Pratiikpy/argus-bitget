"""``/materials``: every deliverable on one page, figures read live, no placeholder shown."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from argus.lui import materials_page
from argus.lui.materials_page import VIDEO_ENV, X_POST_ENV, collect, render

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(autouse=True)
def no_links_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(VIDEO_ENV, raising=False)
    monkeypatch.delenv(X_POST_ENV, raising=False)


class TestTheIndex:
    def test_every_page_a_judge_needs_is_listed(self) -> None:
        hrefs = {i.href for i in collect(DATA, "https://argus.example")}
        for path in ("/", "/research", "/proof", "/wrong", "/status", "/mcp"):
            assert f"https://argus.example{path}" in hrefs
        assert materials_page.REPOSITORY in hrefs
        assert materials_page.TELEGRAM in hrefs

    def test_the_register_line_is_the_register(self) -> None:
        report = json.loads((DATA / "standing.json").read_text(encoding="utf-8"))
        counts = report["by_state"]
        line = next(i.shows for i in collect(DATA) if i.label == "What we beat")
        assert f"{sum(counts.values())} capabilities" in line
        for state, n in counts.items():
            assert f"{n} {state.upper()}" in line

    def test_an_unreadable_register_says_so(self, tmp_path: Path) -> None:
        line = next(i.shows for i in collect(tmp_path) if i.label == "What we beat")
        assert "could not be read" in line


class TestNoPlaceholders:
    def test_a_missing_video_is_left_off(self) -> None:
        labels = {i.label for i in collect(DATA)}
        assert "Demo video" not in labels and "X post" not in labels
        page = render(collect(DATA))
        assert "placeholder" not in page.lower() and "yours" not in page.lower()

    def test_the_video_appears_once_it_exists(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(VIDEO_ENV, "https://youtu.be/example")
        monkeypatch.setenv(X_POST_ENV, "https://x.com/example/status/1")
        items = collect(DATA)
        video = next(i for i in items if i.label == "Demo video")
        assert video.href == "https://youtu.be/example" and video.group == "Use it"
        assert any(i.label == "X post" for i in items)


class TestThePage:
    def test_it_renders_every_item_escaped(self) -> None:
        page = render(collect(DATA, "https://argus.example"))
        assert page.startswith("<!doctype html>")
        assert "Everything, on one page." in page
        assert 'href="https://argus.example/research"' in page
        assert ">argus.example/research<" in page
        assert "?format=json" in page

    def test_the_server_answers_it_with_full_urls(self) -> None:
        """A real server on an ephemeral port: the page and its JSON, links built from the Host."""
        import threading
        from http.server import ThreadingHTTPServer
        from urllib.request import urlopen

        from argus.lui.server import Handler

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            with urlopen(f"{base}/materials") as response:
                page = response.read().decode("utf-8")
                assert response.status == 200
            with urlopen(f"{base}/materials?format=json") as response:
                rows = json.loads(response.read())
        finally:
            server.shutdown()
            server.server_close()
        assert f'href="{base}/research"' in page
        assert {r["label"] for r in rows} >= {"Console", "Research task", "What we beat"}
