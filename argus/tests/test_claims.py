"""Claim-grounding tests.

Built from one real failure. Ledger seq 41, 2026-09-12: the desk was shown two insider sales and
wrote *"insider sales are pre-arranged 10b5-1 plans already reflected in price"*. Both filings carry
``aff10b5One = 0``, and `argus.market.insider` forwards a sale only when it is **not**
pre-arranged — so the thesis asserted, about the two records in front of it, the exact property
that would have excluded them from ever being shown.

Numeric grounding cannot see that: no number was wrong.

The second property under test is restraint. This module detects contradictions against structured
fields, not hallucination in general. A claim it cannot settle must be reported as unexamined, never
as passing — a checker that says "sound" about something it never looked at is worse than no
checker.
"""

from __future__ import annotations

from typing import Any

import pytest

from argus.agents.claims import RULES, check

# The verbatim thesis from ledger seq 41.
SEQ_41 = (
    "Weekend session with 44 hours to genuine price discovery, no actionable edge, and consensus "
    "neutral at 0.83 across all panels. The RSS headlines are generic listicles, insider sales are "
    "pre-arranged 10b5-1 plans already reflected in price, and fundamentals cannot be assessed "
    "against expectations without consensus estimates."
)


def _record(
    ident: str = "form4-0001199039-26-000014-S",
    *,
    pre_arranged: bool = False,
    acquired: bool = False,
    conviction: bool = False,
) -> tuple[str, dict[str, Any]]:
    return ident, {
        "accession": "0001199039-26-000014",
        "insider": "STEVENS MARK A",
        "code": "S",
        "pre_arranged": pre_arranged,
        "acquired": acquired,
        "conviction": conviction,
    }


class TestTheRealFailure:
    """Seq 41, kept as a regression fixture."""

    def test_the_pre_arranged_claim_is_caught(self) -> None:
        report = check(SEQ_41, records=[_record()])
        assert not report.sound
        assert report.contradictions[0].rule == "pre_arranged"

    def test_it_names_the_evidence_that_refutes_it(self) -> None:
        report = check(SEQ_41, records=[_record()])
        contradiction = report.contradictions[0]
        assert contradiction.evidence_id == "form4-0001199039-26-000014-S"
        assert contradiction.actual is False

    def test_it_quotes_the_offending_text(self) -> None:
        report = check(SEQ_41, records=[_record()])
        assert "pre-arranged" in report.contradictions[0].claim_text

    def test_it_explains_why_the_claim_is_self_defeating(self) -> None:
        """The property claimed is the one that would have excluded the evidence."""
        report = check(SEQ_41, records=[_record()])
        assert "only reaches the desk when it is not pre-arranged" in (
            report.contradictions[0].explain
        )

    def test_the_verdict_is_stated_as_a_defect_not_a_disagreement(self) -> None:
        report = check(SEQ_41, records=[_record()])
        assert "defect in the reasoning" in " ".join(report.render())

    def test_the_same_thesis_is_sound_when_the_filing_agrees(self) -> None:
        """If the sales really had been 10b5-1, the thesis would have been right."""
        assert check(SEQ_41, records=[_record(pre_arranged=True)]).sound


class TestRestraint:
    """A checker that says "sound" about something it never looked at is worse than none."""

    def test_a_claim_with_no_structured_field_is_not_examined(self) -> None:
        report = check(SEQ_41, records=[("e1", {"insider": "someone"})])
        assert report.claims_examined == 0
        assert report.sound is True
        assert "no checkable claim" in " ".join(report.render())

    def test_no_evidence_at_all_is_reported_as_such(self) -> None:
        report = check(SEQ_41, records=[])
        assert report.records_checked == 0
        assert "no structured evidence" in " ".join(report.render())

    def test_a_thesis_with_no_claims_examines_nothing(self) -> None:
        report = check("Weekend session; no actionable edge.", records=[_record()])
        assert report.claims_examined == 0
        assert report.sound

    def test_the_counts_make_coverage_visible(self) -> None:
        report = check(SEQ_41, records=[_record()])
        assert report.claims_examined >= 1
        assert report.records_checked == 1
        assert report.as_dict()["claims_examined"] == report.claims_examined


class TestNegation:
    def test_a_negated_claim_is_not_treated_as_the_assertion(self) -> None:
        """"not pre-arranged" asserts the opposite and must not be flagged."""
        thesis = "These sales are not pre-arranged, so they carry information."
        assert check(thesis, records=[_record()]).sound

    @pytest.mark.parametrize(
        "phrase",
        ["are not pre-arranged", "were never pre-arranged", "no pre-arranged plan",
         "without a pre-arranged plan", "isn't pre-arranged"],
    )
    def test_the_common_negations_are_recognised(self, phrase: str) -> None:
        assert check(f"The insider sales {phrase} here.", records=[_record()]).sound

    def test_a_distant_negation_does_not_shield_the_claim(self) -> None:
        """A "not" forty words earlier is about something else."""
        thesis = (
            "There is not much news today. " + "Filler sentence to push distance. " * 3
            + "The insider sales are pre-arranged 10b5-1 plans."
        )
        assert not check(thesis, records=[_record()]).sound


class TestOneAgreeingRecordIsEnough:
    def test_a_thesis_may_be_about_a_subset_of_its_evidence(self) -> None:
        """Two sales, one genuinely pre-arranged: the claim is true of something."""
        records = [_record("a", pre_arranged=False), _record("b", pre_arranged=True)]
        assert check(SEQ_41, records=records).sound

    def test_it_contradicts_only_when_every_record_disagrees(self) -> None:
        records = [_record("a", pre_arranged=False), _record("b", pre_arranged=False)]
        assert not check(SEQ_41, records=records).sound


class TestTheOtherRules:
    def test_claiming_insider_buying_against_a_disposal_is_caught(self) -> None:
        report = check("Insider buying supports the long case.", records=[_record()])
        assert not report.sound
        assert report.contradictions[0].rule == "open_market_purchase"

    def test_claiming_conviction_against_a_sale_is_caught(self) -> None:
        report = check("A high-conviction insider signal here.", records=[_record()])
        assert not report.sound
        assert report.contradictions[0].rule == "conviction"

    def test_a_genuine_conviction_purchase_passes_both(self) -> None:
        record = _record(acquired=True, conviction=True)
        assert check("Insider buying, a high-conviction signal.", records=[record]).sound

    def test_every_rule_names_a_field_and_an_explanation(self) -> None:
        for rule in RULES:
            assert rule.field and rule.explain
            assert rule.regex.pattern


class TestTheFiledFigures:
    """The second structured source. Same machine, different evidence."""

    @staticmethod
    def _qoq(*, growing: bool) -> tuple[str, dict[str, Any]]:
        return "xbrl-NVDA-revenue-qoq", {
            "concept": "revenue",
            "change_pct": 12.4 if growing else -12.4,
            "growing": growing,
            "from_period": "2026-04-30",
            "to_period": "2026-07-31",
        }

    def test_claiming_growth_against_a_filed_decline_is_caught(self) -> None:
        report = check("Revenue growth supports the long case.", records=[self._qoq(growing=False)])
        assert not report.sound
        assert report.contradictions[0].rule == "fundamental_growth"

    def test_claiming_decline_against_filed_growth_is_caught(self) -> None:
        report = check("Revenue declined again this quarter.", records=[self._qoq(growing=True)])
        assert not report.sound
        assert report.contradictions[0].rule == "fundamental_decline"

    def test_each_claim_passes_against_the_figures_that_support_it(self) -> None:
        assert check("Revenue grew.", records=[self._qoq(growing=True)]).sound
        assert check("Revenue fell.", records=[self._qoq(growing=False)]).sound

    def test_a_per_period_fact_carries_no_field_and_is_left_alone(self) -> None:
        """One reported number cannot settle a claim about direction."""
        fact = ("xbrl-NVDA-revenue-2026-07-31", {"concept": "revenue", "value": 4.67e10})
        assert check("Revenue growth supports the long case.", records=[fact]).claims_examined == 0

    def test_the_two_directions_cannot_both_fire(self) -> None:
        """A thesis is allowed to be wrong once, not contradicted coming and going."""
        for growing in (True, False):
            report = check("Revenue grew.", records=[self._qoq(growing=growing)])
            assert len(report.contradictions) <= 1


class TestTheReportSerialises:
    def test_a_contradiction_carries_everything_needed_to_check_it(self) -> None:
        payload = check(SEQ_41, records=[_record()]).as_dict()
        entry = payload["contradictions"][0]
        for key in ("rule", "claim", "evidence", "field", "asserted", "actual", "explain"):
            assert key in entry

    def test_a_sound_report_says_how_much_it_checked(self) -> None:
        text = " ".join(check("Insider buying.", records=[_record(acquired=True)]).render())
        assert "checkable claim" in text and "structured record" in text
