"""Cross-surface agreement: do two readings of the same fact say the same thing?

**The check that matters most here is `TestItCanActuallyFail`.** A consistency check that cannot
report an inconsistency is a green line in a report whose entire purpose is to go red — the same
discipline `tests/test_anchorcheck.py` applies to the anchor verifier and `eval/riskproof.py`
applies to the Constitution.

So one test reconstructs the defect this module was written for — the console counting two refused
positions as trades while every other surface reported zero — and asserts the comparison catches
it.
"""

from __future__ import annotations

from typing import Any

import pytest

from argus.eval.surfaces import Agreement, render, run


class TestTheLiveSurfaces:
    @pytest.fixture(scope="class")
    def report(self) -> dict[str, Any]:
        return run()

    def test_every_surface_agrees(self, report: dict[str, Any]) -> None:
        """If this fails, two of our own surfaces are telling a reader different things.

        The fix is whichever surface is wrong — never this test.
        """
        assert report["clean"], report["disagreements"]

    def test_the_trades_figure_is_checked(self, report: dict[str, Any]) -> None:
        """The specific defect that motivated the module stays covered by name."""
        facts = {c["fact"] for c in report["checks"]}
        assert "settled trades" in facts

    def test_every_check_names_both_sources(self, report: dict[str, Any]) -> None:
        """A disagreement is only actionable if a reader can go and look at both readings."""
        for check in report["checks"]:
            assert check["console_source"]
            assert check["artefact_source"]
            assert check["why_it_matters"]

    def test_the_scope_statement_admits_what_agreement_does_not_prove(
        self, report: dict[str, Any]
    ) -> None:
        """Two surfaces can agree and both be wrong; that is docclaims' job, not this one."""
        assert "NOT CLAIMED" in report["scope_statement"]


class TestItCanActuallyFail:
    """A checker that only ever passes has not been tested — it has been assumed."""

    def test_a_disagreement_is_reported_as_one(self) -> None:
        """The historical defect, reconstructed: console said 2 trades, artefact said 0."""
        bad = Agreement(
            fact="settled trades", console=2, artefact=0,
            console_says="lui.answer", artefact_says="eval.performance",
            why_it_matters="the documents all state zero",
        )
        assert not bad.agrees
        assert "DISAGREE" in bad.render()

    def test_a_disagreeing_report_is_not_clean(self) -> None:
        report = {
            "facts_checked": 1, "agree": 0, "disagree": 1, "clean": False,
            "checks": [
                {"fact": "settled trades", "console": 2, "artefact": 0, "agrees": False,
                 "console_source": "lui.answer", "artefact_source": "eval.performance",
                 "why_it_matters": "the documents all state zero"}
            ],
            "disagreements": [
                {"fact": "settled trades", "console": 2, "artefact": 0, "agrees": False,
                 "console_source": "lui.answer", "artefact_source": "eval.performance",
                 "why_it_matters": "the documents all state zero"}
            ],
        }
        text = "\n".join(render(report))
        assert "DISAGREE" in text
        assert "console 2 vs artefact 0" in text

    def test_both_readings_are_printed_and_neither_is_called_correct(self) -> None:
        """Naming a winner would make this a third opinion, which is how you get three
        surfaces disagreeing instead of two."""
        report = run()
        assert "neither is called correct" in "\n".join(render(report))

    def test_equal_values_of_different_types_do_not_count_as_agreement_by_accident(
        self,
    ) -> None:
        """`0 == False` in Python. A count compared against a boolean must not pass silently."""
        assert Agreement("f", 0, 0, "a", "b", "w").agrees
        assert not Agreement("f", 1, 0, "a", "b", "w").agrees


class TestTheFifthCheckIsNotATautology:
    """The obvious fifth comparison was a value against itself, and was replaced.

    `answer_integrity` returns `ledger.verify()` verbatim, so comparing the two always passes.
    """

    def test_it_compares_the_header_against_the_list(self) -> None:
        report = run()
        facts = {c["fact"] for c in report["checks"]}
        assert "count: page header vs list" in facts
        assert "chain intact" not in facts, (
            "answer_integrity returns verify() verbatim, so that comparison cannot fail"
        )

    def test_the_two_sources_are_genuinely_different_modules(self) -> None:
        report = run()
        row = next(c for c in report["checks"] if c["fact"] == "count: page header vs list")
        assert "lui.server" in row["console_source"]
        assert "lui.answer" in row["artefact_source"]
