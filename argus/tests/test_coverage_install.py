"""Installing the urlopen wrapper from many threads at once (`truth/coverage.py`)."""

from __future__ import annotations

import pytest


def test_racing_installs_keep_the_real_urlopen_as_the_original(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Audit finding 160: two threads installing at once must not wrap the wrapper."""
    import threading
    import urllib.request

    from argus.truth import coverage

    real = urllib.request.urlopen
    monkeypatch.setattr(coverage, "_original", None)
    monkeypatch.setattr(urllib.request, "urlopen", real)
    barrier = threading.Barrier(8)

    def race() -> None:
        barrier.wait()
        coverage.install()

    threads = [threading.Thread(target=race) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert coverage._original is real
    assert urllib.request.urlopen is coverage._observed
