"""Does the break-even heuristic reproduce a convex optimiser's rebalance decision? No.

**The published figure said it did, and could not have said anything else.**
`eval/allocation_comparison.run_convex_rebalance` reports `agreement_rate: 1.0` over four cells, all
on one equal-weight book, in which both sides said trade every time. A concordance with no
discordant cell available is not a measurement.

Run across a grid where both answers are reachable, agreement is **66.7%**, and the disagreement is
entirely one-directional: 24 cells where the convex program trades and ARGUS holds, and **zero** the
other way. ARGUS's heuristic is a strictly conservative approximation of the convex decision, not a
reproduction of it. Whether those 24 skipped trades would have paid is not measured here and is not
claimed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ARTEFACT = Path(__file__).resolve().parents[1] / "data" / "allocation_agreement.json"


@pytest.fixture(scope="module")
def report() -> dict:
    if not ARTEFACT.exists():
        pytest.skip("run `python -m argus.eval.allocation_agreement` to produce the artefact")
    return json.loads(ARTEFACT.read_text(encoding="utf-8"))


class TestTheGridCanActuallyDisagree:
    """Without this, the agreement rate is arithmetic. It is the whole reason this file exists."""

    def test_both_sides_say_no_somewhere(self, report: dict) -> None:
        assert report["discordant_cells_available"] is True
        assert report["argus_says_no_in"] > 0
        assert report["convex_says_no_in"] > 0

    def test_neither_side_answers_the_same_way_everywhere(self, report: dict) -> None:
        assert report["both_answers_observed_from_argus"] is True
        assert report["both_answers_observed_from_convex"] is True

    def test_the_grid_is_materially_larger_than_the_four_cells_it_replaces(
        self, report: dict
    ) -> None:
        assert report["cells"] >= 48


class TestTheHeuristicDoesNotReproduceTheConvexDecision:
    def test_agreement_is_well_below_one(self, report: dict) -> None:
        """Pinned as a bound rather than an equality: the claim being refuted is "it reproduces
        it", and any figure this far from 1.0 refutes it."""
        assert report["agreement_rate"] is not None
        assert report["agreement_rate"] < 0.9

    def test_the_disagreement_is_one_directional(self, report: dict) -> None:
        """**The actual finding, and it is more useful than the agreement rate.** ARGUS never
        trades when the convex program holds; it holds when the convex program trades. That makes
        it a strictly conservative approximation, which is a characterisation somebody can act on
        — unlike a bare percentage."""
        confusion = report["confusion"]
        assert confusion["argus_trades_convex_holds"] == 0
        assert confusion["convex_trades_argus_holds"] > 0

    def test_the_scope_statement_refuses_the_claim_it_cannot_support(self, report: dict) -> None:
        scope = report["scope_statement"]
        assert "NOT CLAIMED" in scope
        assert "either side is right" in scope
