"""The research page arrives as its steps finish (research/harvest/48-open-webui.md): the waiting
cards reach the reader before any engine has answered, each step's card follows its engine, and
the page the reader is left with is the one `render_task` draws."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from http.server import ThreadingHTTPServer
from typing import Any
from urllib.parse import quote
from urllib.request import urlopen

import pytest

from argus.lui import task as task_module
from argus.lui.server import Handler
from argus.lui.task import STEPS, Reading, Step, Task
from argus.lui.task_page import render_task, stream_end, stream_start, stream_step

READING = Reading(name="NVDAUSDT", size_pct=10.0, book={"AAPLUSDT": 1.0})


def fake_task(release: threading.Event) -> Any:
    def research_task(*, reading: Reading, asked: str, memory: str = "",
                      on_step: Any = None) -> Task:
        steps = []
        for i, (title, _, engine) in enumerate(STEPS):
            if i == 1:
                # The second engine answers only once the reader has the first bytes: a server
                # that buffered the response would never let this happen, and the test would hang
                # until its timeout.
                assert release.wait(10), "the page was not sent before the task finished"
            step = Step(title=title, engine=engine, lines=[f"line from step {i + 1}"],
                        seconds=0.1)
            steps.append(step)
            if on_step is not None:
                on_step(i, step)
        return Task(question=asked, name="NVDA", size_pct=10.0, book=dict(reading.book),
                    steps=steps, seconds=0.8, asked=asked, reading=reading)
    return research_task


@pytest.fixture
def base_url(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[str, threading.Event]]:
    release = threading.Event()
    monkeypatch.setattr(task_module, "research_task", fake_task(release))
    monkeypatch.setattr(task_module, "read_question", lambda asked, saved: READING)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}", release
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_the_waiting_page_arrives_before_the_engines_answer(
        base_url: tuple[str, threading.Event]) -> None:
    url, release = base_url
    with urlopen(f"{url}/research?q={quote('should I add 10% NVDA')}", timeout=20) as response:
        assert response.headers.get("Content-Length") is None
        first = b""
        # The waiting cards and step 1's card, all before step 2 is allowed to run.
        while b"slot-8" not in first or b"argusStep(1)" not in first:
            chunk = response.read1(65536)
            assert chunk, "the connection closed before the first step was sent"
            first += chunk
        assert b"running" in first and b"argusStep(2)" not in first
        release.set()
        rest = response.read()
    page = (first + rest).decode("utf-8")
    assert page.count("<template id='st-") == len(STEPS) + 1
    assert "argusDone()" in page and page.rstrip().endswith("</html>")


def test_the_page_left_behind_is_the_rendered_page() -> None:
    steps = [Step(title=t, engine=e, lines=["x"]) for t, _, e in STEPS]
    task = Task(question="q", name="NVDA", size_pct=10.0, book={"AAPLUSDT": 1.0}, steps=steps,
                seconds=1.0, asked="q", reading=READING)
    streamed = stream_start("q", "NVDA", 10.0, {"AAPLUSDT": 1.0},
                            [(s.title, s.engine) for s in steps])
    streamed += "".join(stream_step(n, s) for n, s in enumerate(steps, 1)) + stream_end(task)
    whole = render_task(task, "")
    # The finished content inside the last template is exactly the page's own main section.
    done = streamed.split("<template id='st-done'>", 1)[1].split("</template>", 1)[0]
    assert done in whole


def test_json_and_stream_off_keep_the_single_response(
        base_url: tuple[str, threading.Event]) -> None:
    url, release = base_url
    release.set()
    with urlopen(f"{url}/research?q=nvda&stream=0", timeout=20) as response:
        assert response.headers.get("Content-Length") is not None
        assert b"argusStep" not in response.read()
