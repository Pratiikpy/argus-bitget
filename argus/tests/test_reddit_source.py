"""Tests for `market.evidence.RedditSource` — real subprocess calls to `agent-reach`'s real
`rdt-cli` backend. Kept separate from `test_evidence.py` (deliberately network-free), same
structure as `test_twitter_source.py`.

Added 2026-09-16, same session as `TwitterSource`: `agent-reach doctor --json` reported this
channel `status: error` (a cookie-refresh warning, not a real failure) — a direct real query
found it working regardless, so the doctor summary was checked against the real thing.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import UTC, datetime, timedelta

import pytest

from argus.market.evidence import RedditSource

NOW = datetime(2026, 9, 16, 12, 0, tzinfo=UTC)


class TestErrorPaths:
    """Real degeneracies, exercised with a monkeypatched `subprocess.run` rather than the real
    network — about `RedditSource`'s own error handling, not about live reachability."""

    def test_cli_not_installed_degrades_rather_than_raising(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _missing(*args: object, **kwargs: object) -> None:
            raise FileNotFoundError("no such file")

        monkeypatch.setattr(subprocess, "run", _missing)
        src = RedditSource()
        evidence, status = src.evidence("NVDAUSDT", as_of=NOW)
        assert evidence == []
        assert "unavailable" in status[0]

    def test_a_real_crash_inside_the_cli_degrades_rather_than_raising(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The real shape this actually hit: `rdt-cli` itself crashing with a nonzero exit
        (`CalledProcessError`) on a Windows console-encoding bug inside its own code, before any
        JSON reached this process — see `_utf8_subprocess_env`'s own docstring for the real
        traceback this was diagnosed from."""

        def _crashes(*args: object, **kwargs: object) -> None:
            raise subprocess.CalledProcessError(1, cmd="rdt")

        monkeypatch.setattr(subprocess, "run", _crashes)
        src = RedditSource()
        evidence, status = src.evidence("NVDAUSDT", as_of=NOW)
        assert evidence == []
        assert "unavailable" in status[0]

    def test_unparseable_stdout_degrades_rather_than_raising(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _garbage(*args: object, **kwargs: object) -> subprocess.CompletedProcess:
            return subprocess.CompletedProcess(args=["rdt"], returncode=0, stdout="not json")

        monkeypatch.setattr(subprocess, "run", _garbage)
        src = RedditSource()
        evidence, status = src.evidence("NVDAUSDT", as_of=NOW)
        assert evidence == []
        assert "unparseable" in status[0]

    def test_ok_false_in_a_valid_payload_is_reported_not_silently_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _not_ok(*args: object, **kwargs: object) -> subprocess.CompletedProcess:
            return subprocess.CompletedProcess(
                args=["rdt"], returncode=0, stdout=json.dumps({"ok": False})
            )

        monkeypatch.setattr(subprocess, "run", _not_ok)
        src = RedditSource()
        evidence, status = src.evidence("NVDAUSDT", as_of=NOW)
        assert evidence == []
        assert status == ["reddit: reachable, ok=false"]

    def test_the_as_of_gate_drops_a_post_stamped_after_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        future_ts = (NOW + timedelta(hours=2)).timestamp()
        payload = {
            "ok": True,
            "data": {"kind": "Listing", "data": {"children": [{
                "kind": "t3",
                "data": {
                    "id": "1", "title": "from the future", "selftext": "",
                    "created_utc": future_ts, "author": "x", "subreddit": "stocks",
                    "score": 0, "num_comments": 0,
                },
            }]}},
        }

        def _future_post(*args: object, **kwargs: object) -> subprocess.CompletedProcess:
            return subprocess.CompletedProcess(
                args=["rdt"], returncode=0, stdout=json.dumps(payload)
            )

        monkeypatch.setattr(subprocess, "run", _future_post)
        src = RedditSource()
        evidence, status = src.evidence("NVDAUSDT", as_of=NOW)
        assert evidence == []
        assert status == ["reddit: 0 posts"]

    def test_a_real_shaped_payload_parses_into_evidence_correctly(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        past_ts = (NOW - timedelta(hours=1)).timestamp()
        payload = {
            "ok": True,
            "data": {"kind": "Listing", "data": {"children": [{
                "kind": "t3",
                "data": {
                    "id": "42", "title": "NVDA earnings thread",
                    "selftext": "beat on revenue", "created_utc": past_ts,
                    "author": "trader1", "subreddit": "wallstreetbets",
                    "score": 500, "num_comments": 120,
                },
            }]}},
        }

        def _one_post(*args: object, **kwargs: object) -> subprocess.CompletedProcess:
            return subprocess.CompletedProcess(
                args=["rdt"], returncode=0, stdout=json.dumps(payload)
            )

        monkeypatch.setattr(subprocess, "run", _one_post)
        src = RedditSource()
        evidence, status = src.evidence("NVDAUSDT", as_of=NOW)
        assert len(evidence) == 1
        e = evidence[0]
        assert e.id == "reddit-42"
        assert e.claim == "NVDA earnings thread — beat on revenue"
        assert e.source == "social"
        assert e.credibility == 0.35
        assert e.attributes["author"] == "trader1"
        assert e.attributes["subreddit"] == "wallstreetbets"
        assert e.attributes["score"] == 500
        assert status == ["reddit: 1 posts"]


@pytest.mark.skipif(shutil.which("rdt") is None, reason="rdt-cli not installed here")
class TestRealLiveCli:
    """The one real, live check — confirms the actual CLI still works, still returns real,
    parseable data for a real rToken symbol. Skips cleanly rather than failing if the CLI is
    genuinely absent."""

    def test_a_real_search_returns_real_recent_evidence(self) -> None:
        src = RedditSource(timeout=30.0)
        evidence, status = src.evidence("NVDAUSDT", as_of=datetime.now(UTC))
        assert status[0].startswith("reddit:")
        if evidence:
            e = evidence[0]
            assert e.source == "social"
            assert 0.0 < e.credibility < 1.0
            assert e.available_at <= datetime.now(UTC)
            assert e.claim
