"""Tests for `market.evidence.TwitterSource` — real subprocess calls to `agent-reach`'s real
`twitter-cli` backend. Kept separate from `test_evidence.py`, which is deliberately network-free
(every source there is driven by fixtures); this file is the one place the real CLI is exercised.

Added 2026-09-16 after re-checking a real, previously-disclosed blocker on the "Market sentiment"
capability (`eval/standing.py`, `t2-sentiment`: "the free replacements are blocked from this
network") and finding it stale — `agent-reach doctor --json` now reports Twitter/X `status: ok`.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import UTC, datetime, timedelta

import pytest

from argus.market.evidence import TwitterSource

NOW = datetime(2026, 9, 16, 12, 0, tzinfo=UTC)


class TestErrorPaths:
    """Real degeneracies, exercised with a monkeypatched `subprocess.run` rather than the real
    network — these are about `TwitterSource`'s own error handling, not about whether Twitter is
    reachable right now."""

    def test_cli_not_installed_degrades_rather_than_raising(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _missing(*args: object, **kwargs: object) -> None:
            raise FileNotFoundError("no such file")

        monkeypatch.setattr(subprocess, "run", _missing)
        src = TwitterSource()
        evidence, status = src.evidence("NVDAUSDT", as_of=NOW)
        assert evidence == []
        assert "unavailable" in status[0]

    def test_a_timeout_degrades_rather_than_raising(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _slow(*args: object, **kwargs: object) -> None:
            raise subprocess.TimeoutExpired(cmd="twitter", timeout=20.0)

        monkeypatch.setattr(subprocess, "run", _slow)
        src = TwitterSource()
        evidence, status = src.evidence("NVDAUSDT", as_of=NOW)
        assert evidence == []
        assert "unavailable" in status[0]

    def test_unparseable_stdout_degrades_rather_than_raising(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _garbage(*args: object, **kwargs: object) -> subprocess.CompletedProcess:
            return subprocess.CompletedProcess(args=["twitter"], returncode=0, stdout="not json")

        monkeypatch.setattr(subprocess, "run", _garbage)
        src = TwitterSource()
        evidence, status = src.evidence("NVDAUSDT", as_of=NOW)
        assert evidence == []
        assert "unparseable" in status[0]

    def test_ok_false_in_a_valid_payload_is_reported_not_silently_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _not_ok(*args: object, **kwargs: object) -> subprocess.CompletedProcess:
            return subprocess.CompletedProcess(
                args=["twitter"], returncode=0, stdout=json.dumps({"ok": False})
            )

        monkeypatch.setattr(subprocess, "run", _not_ok)
        src = TwitterSource()
        evidence, status = src.evidence("NVDAUSDT", as_of=NOW)
        assert evidence == []
        assert status == ["twitter: reachable, ok=false"]

    def test_the_as_of_gate_drops_a_tweet_stamped_after_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        future = (NOW + timedelta(hours=2)).isoformat()
        payload = {
            "ok": True,
            "data": [{
                "id": "1", "text": "from the future", "createdAtISO": future,
                "author": {"screenName": "x", "verified": False}, "metrics": {"likes": 0},
            }],
        }

        def _future_tweet(*args: object, **kwargs: object) -> subprocess.CompletedProcess:
            return subprocess.CompletedProcess(
                args=["twitter"], returncode=0, stdout=json.dumps(payload)
            )

        monkeypatch.setattr(subprocess, "run", _future_tweet)
        src = TwitterSource()
        evidence, status = src.evidence("NVDAUSDT", as_of=NOW)
        assert evidence == []
        assert status == ["twitter: 0 tweets"]

    def test_a_real_shaped_payload_parses_into_evidence_correctly(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        past = (NOW - timedelta(hours=1)).isoformat()
        payload = {
            "ok": True,
            "data": [{
                "id": "42", "text": "NVDA looking strong today",
                "createdAtISO": past,
                "author": {"screenName": "trader1", "verified": True},
                "metrics": {"likes": 100, "retweets": 10},
            }],
        }

        def _one_tweet(*args: object, **kwargs: object) -> subprocess.CompletedProcess:
            return subprocess.CompletedProcess(
                args=["twitter"], returncode=0, stdout=json.dumps(payload)
            )

        monkeypatch.setattr(subprocess, "run", _one_tweet)
        src = TwitterSource()
        evidence, status = src.evidence("NVDAUSDT", as_of=NOW)
        assert len(evidence) == 1
        e = evidence[0]
        assert e.id == "twitter-42"
        assert e.claim == "NVDA looking strong today"
        assert e.source == "social"
        assert e.credibility == 0.4
        assert e.attributes["author"] == "trader1"
        assert e.attributes["verified"] is True
        assert e.attributes["likes"] == 100
        assert status == ["twitter: 1 tweets"]


@pytest.mark.skipif(shutil.which("twitter") is None, reason="twitter-cli not installed here")
class TestRealLiveCli:
    """The one real, live check — confirms the actual CLI still works, still returns real,
    parseable data for a real rToken symbol. Skips cleanly rather than failing if the CLI is
    genuinely absent (a different machine, a fresh clone) — the offline tests above cover the
    logic either way."""

    def test_a_real_search_returns_real_recent_evidence(self) -> None:
        src = TwitterSource(timeout=30.0)
        evidence, status = src.evidence("NVDAUSDT", as_of=datetime.now(UTC))
        assert status[0].startswith("twitter:")
        if evidence:
            e = evidence[0]
            assert e.source == "social"
            assert 0.0 < e.credibility < 1.0
            assert e.available_at <= datetime.now(UTC)
            assert e.claim
