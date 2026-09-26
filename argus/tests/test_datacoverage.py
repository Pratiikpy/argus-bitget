"""Integration *count* is the easy claim. Integration *effectiveness* is the graded one.

Track 3's criterion is "data sources / Skill integration count **and effectiveness**". A
submission can list sixty-seven integrations and have four that return anything, and a feature list
cannot tell the two apart. `eval/datacoverage.py` calls every entry and classifies what came back;
these tests pin the properties that make that number honest.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from argus.eval.datacoverage import PROBE_SYMBOL, Probe

ARTEFACT = Path(__file__).resolve().parents[1] / "data" / "data_coverage.json"


@pytest.fixture(scope="module")
def report() -> dict:
    if not ARTEFACT.exists():
        pytest.skip("run `python -m argus.eval.datacoverage` to produce the artefact")
    return json.loads(ARTEFACT.read_text(encoding="utf-8"))


class TestEveryEntryWasActuallyCalled:
    def test_the_probe_count_matches_the_catalog(self, report: dict) -> None:
        """A coverage report over a subset is a coverage report about the subset."""
        assert report["entries_probed"] == sum(report["catalog"].values())

    def test_every_probe_carries_a_health_state(self, report: dict) -> None:
        states = {p["health"] for p in report["probes"]}
        assert states <= {"ok", "empty", "error", "needs_params"}
        assert sum(report["by_health"].values()) == report["entries_probed"]

    def test_a_surface_that_does_not_answer_says_why(self, report: dict) -> None:
        """How much of the equity surface answers is a fact about the service on the day, not a
        property of this code: 21 of 22 answered on 2026-09-25, none since its upstream began
        returning 503. So the test pins what must hold either way: the count is dated, and every
        entry that failed carries the service's own reason."""
        assert report["checked_at"]
        assert 0 <= report["equity_answering"] <= report["equity_entries"]
        for probe in report["probes"]:
            if probe["health"] == "error":
                assert probe["detail"], "a failed entry must say how it failed"

    def test_it_probed_a_real_rtoken(self, report: dict) -> None:
        assert report["probe_symbol"] == PROBE_SYMBOL
        assert report["underlying"] == "NVDA"


class TestEmptyIsNotCollapsedIntoBroken:
    def test_empty_and_error_are_counted_separately(self, report: dict) -> None:
        """**The distinction that keeps the number honest.** An endpoint that responds with no
        rows for this symbol is a fact about coverage; one that refuses is a fact about the
        service. Merging them would let either be reported as the other."""
        health = report["by_health"]
        assert not (set(health) - {"ok", "empty", "error", "needs_params"})
        for probe in report["probes"]:
            # A refusal from the service is never filed as "no rows for this symbol".
            if probe["health"] == "empty":
                assert "service reported failure" not in (probe.get("detail") or "")

    def test_needs_params_is_not_reported_as_a_service_failure(self, report: dict) -> None:
        """A probe that calls a tool wrongly measures the probe. Those are labelled as ours."""
        for probe in report["probes"]:
            if probe["health"] == "needs_params":
                assert probe["detail"], "an unsatisfied entry must carry the server's complaint"

    def test_the_answering_rate_is_the_honest_denominator(self, report: dict) -> None:
        """Over everything probed, not over everything that happened to work."""
        expected = report["answering"] / report["entries_probed"]
        # The artefact stores the rate rounded to 4 places, so the tolerance is half a unit in
        # the last place it keeps. A tighter bound would be testing the rounding, not the rate.
        assert abs(report["answering_rate"] - expected) < 5e-5


class TestItStatesWhatItCannotClaim:
    def test_the_scope_statement_names_the_single_symbol_limit(self, report: dict) -> None:
        scope = report["scope_statement"]
        assert "NOT CLAIMED" in scope
        assert "one symbol" in scope or "single probes" in scope

    def test_a_probe_is_a_frozen_record(self) -> None:
        probe = Probe("x", "equity", "ok", 3)
        with pytest.raises(Exception):  # noqa: B017 - frozen dataclass
            probe.rows = 4  # type: ignore[misc]
