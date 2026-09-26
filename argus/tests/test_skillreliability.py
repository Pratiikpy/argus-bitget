"""A data source measured once is a coin toss written down.

**This file exists because the same sweep produced two different stories.** `market/skills.py`
recorded 6 ok / 10 empty / 3 tool_error on 2026-09-20 and 6 ok / 13 timeout on 2026-09-21 — same
code, same endpoint, incompatible descriptions of one service. Track 3 scores "data sources /
Skill integration count **and effectiveness**", so which snapshot gets quoted decides the claim,
and that is exactly the situation in which a claim should not be made.

Repeating the call settles it. The measured answer turned out to be *better* than either snapshot:
9 of 19 tools answer 3 for 3, and 3 of the 5 Skills have a reliable tool — not the 6 and 1 a single
sweep reported.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from argus.eval.skillreliability import SPACING, ToolReliability, measure

ARTEFACT = Path(__file__).resolve().parents[1] / "data" / "skill_reliability.json"


@pytest.fixture(scope="module")
def report() -> dict:
    if not ARTEFACT.exists():
        pytest.skip("run `python -m argus.eval.skillreliability` to produce the artefact")
    return json.loads(ARTEFACT.read_text(encoding="utf-8"))


class _Client:
    """A client with a scripted answer per attempt, so the classifier can be tested without
    depending on a third-party service being in any particular mood."""

    def __init__(self, statuses: list[str]) -> None:
        self._statuses = statuses
        self.calls = 0

    def call(self, tool: str, args: dict, *, timeout: int = 45) -> tuple[object, str]:
        status = self._statuses[min(self.calls, len(self._statuses) - 1)]
        self.calls += 1
        # A real payload with an "ok" status, and nothing with any other: the classifier reads
        # both, and an "ok" around an empty payload is not an answer.
        return ({"rsi": 48.2} if status.endswith(": ok") else {}), status


class TestTheThreeVerdictsAreDistinguishable:
    """The middle one is the whole point. A tool that answers sometimes is neither working nor
    down, and calling it either is a claim the evidence does not support."""

    def test_answering_every_time_is_reliable(self) -> None:
        row = ToolReliability("s", "t", "a", attempts=3, answered=3)
        assert row.verdict == "reliable"

    def test_answering_never_is_down(self) -> None:
        row = ToolReliability("s", "t", "a", attempts=3, answered=0)
        assert row.verdict == "down"

    def test_answering_sometimes_is_intermittent(self) -> None:
        row = ToolReliability("s", "t", "a", attempts=3, answered=1)
        assert row.verdict == "intermittent"

    def test_never_probed_is_not_reported_as_down(self) -> None:
        """Absent, not satisfied — and equally, absent is not failed."""
        assert ToolReliability("s", "t", "a").verdict == "not_probed"


class TestItActuallyRepeats:
    def test_every_probe_is_called_once_per_attempt(self) -> None:
        from argus.market.skills import Probe

        client = _Client(["bitget:t: ok"])
        probes = (Probe(skill="s", tool="t", action="a", args={}, yields="y"),)
        rows = measure(client, attempts=4, probes=probes)
        assert client.calls == 4
        assert rows[0].attempts == 4 and rows[0].answered == 4

    def test_a_flaky_tool_is_caught_as_intermittent(self) -> None:
        """The case a single sweep cannot see, and the reason the module exists."""
        from argus.market.skills import Probe

        client = _Client(["bitget:t: ok", "bitget:t: reachable, returned no data", "bitget:t: ok"])
        probes = (Probe(skill="s", tool="t", action="a", args={}, yields="y"),)
        rows = measure(client, attempts=3, probes=probes)
        assert rows[0].verdict == "intermittent"
        assert rows[0].answered == 2

    def test_a_raised_exception_counts_as_an_attempt_not_a_crash(self) -> None:
        """A transport failure is an observation about the service, not an error in the harness."""
        from argus.market.skills import Probe

        class Broken:
            def call(self, tool: str, args: dict, *, timeout: int = 45) -> tuple[object, str]:
                raise TimeoutError("upstream")

        probes = (Probe(skill="s", tool="t", action="a", args={}, yields="y"),)
        rows = measure(Broken(), attempts=2, probes=probes)
        assert rows[0].attempts == 2 and rows[0].answered == 0
        assert rows[0].failures

    def test_attempts_are_spaced_apart(self) -> None:
        """Back-to-back calls share whatever transient condition caused the failure, so three
        failures in three seconds is closer to one observation than to three."""
        assert SPACING > 0


class TestTheLiveMeasurement:
    def test_it_probed_every_tool_more_than_once(self, report: dict) -> None:
        assert report["attempts_per_tool"] >= 2
        assert all(r["attempts"] == report["attempts_per_tool"] for r in report["results"])

    def test_repeating_agrees_with_a_single_sweep_once_empty_envelopes_are_not_answers(
        self, report: dict
    ) -> None:
        """This test used to pin "repeating found 9 reliable against the sweep's 6". Three of the
        nine returned only an upstream error envelope under an "ok" status, which the health
        classifier counts as empty and this module, reading the status alone, counted as an
        answer (fixed 2026-09-26). Classified the same way, the repeated calls agree with the
        single sweep: the same tools answer every time, and no tool answers only sometimes."""
        reliable = {(r["tool"], r["action"]) for r in report["results"]
                    if r["verdict"] == "reliable"}
        assert reliable
        assert all(tool == "technical_analysis" for tool, _ in reliable)
        assert report["by_verdict"].get("intermittent", 0) == 0

    def test_it_says_how_many_skills_have_a_reliable_tool(self, report: dict) -> None:
        assert 1 <= report["skills_with_a_reliable_tool"] <= report["skills_total"]

    def test_the_scope_statement_refuses_to_blame_the_vendor(self, report: dict) -> None:
        """The failures carry Bitget's own error envelopes, including an explicit ConnectTimeout
        from its upstream. The most this can say is that the door was shut when we knocked."""
        scope = report["scope_statement"]
        assert "NOT CLAIMED" in scope
        assert "the door was shut when we knocked" in scope


def test_an_ok_status_around_an_error_envelope_is_not_an_answer() -> None:
    from argus.market.skills import Probe

    class Envelope:
        def call(self, tool: str, args: dict, *, timeout: int = 45) -> tuple[object, str]:
            return {"error": "", "url": "https://upstream"}, "bitget:defi_analytics: ok"

    probes = (Probe(skill="s", tool="t", action="a", args={}, yields="y"),)
    rows = measure(Envelope(), attempts=2, probes=probes)
    assert rows[0].answered == 0
    assert "error envelope" in rows[0].failures[0]

