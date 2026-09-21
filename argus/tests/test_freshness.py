"""Is an artefact older than the record it describes?

`docclaims` compares a quoted figure against its artefact — and a stale artefact agrees with a
stale document perfectly. The pair is consistent and both describe a record that has moved on.
That is how the refusal study sat at 447 decisions while the ledger reached 495, and re-running it
moved the overnight horizon from 51.2% to **40.4%, below a coin flip**.

The tests that matter here are the ones asserting this check cannot be fooled into reporting
everything fresh.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from argus.eval.freshness import Artefact, FreshnessError, run, survey


class TestItCannotBeFooledIntoSayingFresh:
    def test_an_empty_ledger_raises_rather_than_passing(self, tmp_path, monkeypatch) -> None:
        """With no decisions, nothing can be behind — which would read as a clean bill."""
        import argus.eval.freshness as mod

        empty = tmp_path / "paper_ledger.jsonl"
        empty.write_text("", encoding="utf-8")
        monkeypatch.setattr(mod, "LEDGER", empty)
        with pytest.raises(FreshnessError, match="empty"):
            mod.run()

    def test_a_missing_ledger_raises(self, tmp_path, monkeypatch) -> None:
        import argus.eval.freshness as mod

        monkeypatch.setattr(mod, "LEDGER", tmp_path / "nope.jsonl")
        with pytest.raises(FreshnessError, match="nothing to be stale against"):
            mod.run()

    def test_an_undated_artefact_is_counted_separately_not_as_fresh(self) -> None:
        """No timestamp means staleness is unknowable, which is not the same as current."""
        undated = Artefact(name="x.json", owner="argus.eval.x", stamped_at=None,
                           behind_hours=None)
        assert not undated.dated
        assert not undated.stale
        report = run()
        assert report["undated"] == len(report["undated_names"])

    def test_the_check_finds_something(self) -> None:
        """A freshness check that reports zero ledger-derived artefacts has found nothing to
        check, and would pass forever."""
        assert len(survey()) > 0


class TestAttributionIsByOwnership:
    """**A first attempt attributed nineteen artefacts to docclaims and cockpit.**

    Those modules *read* almost every artefact in the repository and write almost none of them, so
    a reader chasing the re-run command would have found no code to run. Ownership is the module
    that declares the artefact as its own REPORT_PATH.
    """

    def test_no_artefact_is_owned_by_a_pure_reader(self) -> None:
        readers = {"argus.eval.docclaims", "argus.demo.cockpit", "argus.eval.themeaudit"}
        for artefact in survey():
            if artefact.name in {"doc_claims.json", "cockpit.json", "theme_audit.json"}:
                continue
            assert artefact.owner not in readers, (
                f"{artefact.name} attributed to {artefact.owner}, which only reads it"
            )

    def test_every_artefact_carries_a_runnable_command(self) -> None:
        for artefact in survey():
            row = artefact.as_dict()
            assert row["rerun"] and row["rerun"].startswith("python -m argus."), row


class TestTheReport:
    @pytest.fixture(scope="class")
    def report(self) -> dict[str, Any]:
        return run()

    def test_stale_artefacts_are_sorted_worst_first(self, report: dict[str, Any]) -> None:
        behind = [r["behind_hours"] for r in report["stale_detail"]]
        assert behind == sorted(behind, reverse=True)

    def test_it_does_not_call_stale_wrong(self, report: dict[str, Any]) -> None:
        """An earlier measurement of an earlier record is a legitimate thing to publish."""
        assert "NOT CLAIMED" in report["scope_statement"]
        assert "legitimate" in report["scope_statement"]

    def test_it_names_the_defect_that_motivated_it(self, report: dict[str, Any]) -> None:
        assert "40.4%" in report["why_this_matters"]

    def test_the_artefact_is_strict_json(self) -> None:
        from argus.eval.freshness import REPORT_PATH

        if REPORT_PATH.is_file():
            json.loads(REPORT_PATH.read_text(encoding="utf-8"))
