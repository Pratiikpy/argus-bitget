"""The research chain: one question to an actionable insight, with absence printed."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pytest

from argus.desk.research import (
    MIN_STEPS_FOR_A_VERDICT,
    Finding,
    Verdict,
    analogue_step,
    beta_step,
    decide,
    evidence_step,
    execution_step,
    expectation_step,
    parse_book,
    portfolio_step,
    research,
    stress_step,
)

AT = datetime(2026, 9, 14, 15, 0, tzinfo=UTC)


class _Ev:
    def __init__(self, source: str) -> None:
        self.source = source


class _Impact:
    def __init__(self, **kw: Any) -> None:
        self.beta_before = kw.get("beta_before")
        self.beta_after = kw.get("beta_after")
        self.risk_share_before = kw.get("risk_share_before")
        self.risk_share_after = kw.get("risk_share_after")
        self.effective_positions_after = kw.get("effective_positions_after")
        self.max_correlation = kw.get("max_correlation")


class _Stress:
    def __init__(self, results: int = 9, failures: int = 0, unexitable: int = 0) -> None:
        self.results = list(range(results))
        self.failures = list(range(failures))
        self.unexitable = list(range(unexitable))


def _ok(step: str, **kw: Any) -> Finding:
    return Finding(step=step, available=True, headline="h", source="s", **kw)


class TestAbsenceIsNeverSilent:
    def test_no_evidence_is_an_absence_not_an_empty_finding(self) -> None:
        got = evidence_step([])
        assert not got.available and "no evidence" in got.headline

    def test_no_consensus_is_an_absence(self) -> None:
        assert not expectation_step(None).available

    def test_no_analogue_is_an_absence(self) -> None:
        assert not analogue_step(None).available

    def test_no_book_is_an_absence_for_both_portfolio_steps(self) -> None:
        assert not beta_step(None).available
        assert not portfolio_step(None).available

    def test_no_stress_is_an_absence(self) -> None:
        assert not stress_step(None).available

    def test_no_cost_model_is_an_absence(self) -> None:
        assert not execution_step(None, None).available

    def test_an_absent_step_renders_as_not_available(self) -> None:
        assert "NOT AVAILABLE" in evidence_step([]).render()

    def test_the_report_prints_every_absence_with_its_reason(self) -> None:
        report = research(question="q", symbol="NVDAUSDT", now=AT)
        text = report.render()
        assert "What could not be established" in text
        assert "absences, not zeros" in text

    def test_missing_steps_are_listed_by_name(self) -> None:
        report = research(question="q", symbol="NVDAUSDT", now=AT)
        steps = len(report.findings)
        assert len(report.missing) == steps and report.coverage == f"0/{steps}"


class TestTheStepsReadTheRealShapes:
    def test_evidence_counts_items_and_channels(self) -> None:
        got = evidence_step([_Ev("news"), _Ev("news"), _Ev("sec-edgar")])
        assert got.available and "3 item(s)" in got.headline and "2 channel(s)" in got.headline

    def test_beta_reports_the_before_and_after(self) -> None:
        got = beta_step(_Impact(beta_before=0.3, beta_after=0.45))
        assert got.available and "0.300" in got.headline and "0.450" in got.headline

    def test_a_material_beta_rise_is_a_concern(self) -> None:
        got = beta_step(_Impact(beta_before=0.3, beta_after=0.9))
        assert got.concern and "directional exposure" in got.concern

    def test_a_small_beta_rise_is_not_a_concern(self) -> None:
        assert not beta_step(_Impact(beta_before=0.30, beta_after=0.31)).concern

    def test_a_zero_beta_does_not_divide_by_zero(self) -> None:
        got = beta_step(_Impact(beta_before=0.0, beta_after=0.2))
        assert got.available and not got.concern

    def test_an_impact_without_beta_is_an_absence(self) -> None:
        assert not beta_step(_Impact()).available

    def test_portfolio_reports_risk_share_and_correlation(self) -> None:
        got = portfolio_step(_Impact(
            risk_share_before=0.1, risk_share_after=0.2,
            effective_positions_after=2.4, max_correlation=("MSFTUSDT", 0.1),
        ))
        assert got.available
        assert "10% → 20%" in got.headline or "10%" in got.headline
        assert "MSFTUSDT" in got.headline

    def test_a_dominant_position_is_a_concentration_concern(self) -> None:
        got = portfolio_step(_Impact(risk_share_before=0.1, risk_share_after=0.8))
        assert "concentration decision" in got.concern

    def test_a_highly_correlated_addition_is_a_diversification_concern(self) -> None:
        got = portfolio_step(_Impact(
            risk_share_before=0.1, risk_share_after=0.2, max_correlation=("AAPLUSDT", 0.93)
        ))
        assert "without gaining diversification" in got.concern

    def test_an_impact_with_no_figures_at_all_is_an_absence(self) -> None:
        assert not portfolio_step(_Impact()).available

    def test_stress_counts_scenarios_and_failures(self) -> None:
        got = stress_step(_Stress(results=9, failures=2))
        assert got.available and "9 scenario(s)" in got.headline
        assert "2 scenario(s) breach" in got.concern

    def test_an_unexitable_position_is_blocking(self) -> None:
        got = stress_step(_Stress(results=9, failures=9, unexitable=3))
        assert got.detail.get("blocking") is True
        assert "cannot be fully liquidated" in got.concern

    def test_a_cost_above_the_edge_is_flagged(self) -> None:
        got = execution_step(Decimal("12"), Decimal("5"))
        assert got.detail.get("cost_exceeds_edge") is True

    def test_an_edge_above_the_cost_is_not_flagged(self) -> None:
        got = execution_step(Decimal("12"), Decimal("40"))
        assert not got.concern


class TestTheVerdictIsAssembledNotNarrated:
    def _seven(self, available: int, **kw: Any) -> list[Finding]:
        out = [_ok(f"s{i}", **kw) for i in range(available)]
        out += [Finding(step=f"m{i}", available=False, headline="absent", source="")
                for i in range(7 - available)]
        return out

    def test_too_few_steps_is_insufficient_not_a_soft_decline(self) -> None:
        verdict, why = decide(self._seven(2))
        assert verdict is Verdict.INSUFFICIENT
        assert "below the" in why and "Missing" in why

    def test_a_clean_chain_recommends_adding(self) -> None:
        verdict, _ = decide(self._seven(7))
        assert verdict is Verdict.ADD

    def test_a_concern_argues_for_a_smaller_position(self) -> None:
        findings = self._seven(7)
        findings[0] = _ok("beta", concern="beta rises sharply")
        verdict, why = decide(findings)
        assert verdict is Verdict.ADD_SMALLER and "beta" in why

    def test_a_blocking_finding_declines_outright(self) -> None:
        findings = self._seven(7)
        findings[0] = _ok("stress", concern="cannot exit", detail={"blocking": True})
        verdict, why = decide(findings)
        assert verdict is Verdict.DECLINE and "disqualifying" in why

    def test_cost_above_edge_holds_off_rather_than_declining(self) -> None:
        """A hurdle problem is not an evidence problem, and the wording says so."""
        findings = self._seven(7)
        findings[0] = _ok("cost", concern="too expensive", detail={"cost_exceeds_edge": True})
        verdict, why = decide(findings)
        assert verdict is Verdict.HOLD_OFF and "hurdle problem" in why

    def test_blocking_outranks_a_cost_problem(self) -> None:
        findings = self._seven(7)
        findings[0] = _ok("stress", concern="cannot exit", detail={"blocking": True})
        findings[1] = _ok("cost", concern="expensive", detail={"cost_exceeds_edge": True})
        assert decide(findings)[0] is Verdict.DECLINE

    def test_the_same_findings_always_give_the_same_verdict(self) -> None:
        findings = self._seven(6)
        assert decide(findings) == decide(findings)

    def test_the_minimum_is_configurable_and_reported(self) -> None:
        verdict, why = decide(self._seven(3), minimum=6)
        assert verdict is Verdict.INSUFFICIENT and "below the 6" in why

    def test_the_default_minimum_is_four_of_seven(self) -> None:
        assert MIN_STEPS_FOR_A_VERDICT == 4
        assert decide(self._seven(4))[0] is not Verdict.INSUFFICIENT
        assert decide(self._seven(3))[0] is Verdict.INSUFFICIENT


class TestTheReport:
    def test_every_finding_names_the_module_that_produced_it(self) -> None:
        report = research(
            question="q", symbol="NVDAUSDT", evidence=[_Ev("news")],
            impact=_Impact(beta_before=0.3, beta_after=0.35, risk_share_before=0.1,
                           risk_share_after=0.2),
            stress=_Stress(), cost_bps=Decimal("12"), now=AT,
        )
        for finding in report.available:
            assert finding.source, finding.step

    def test_coverage_is_reported_as_a_fraction(self) -> None:
        report = research(question="q", symbol="X", evidence=[_Ev("news")], now=AT)
        assert report.coverage.endswith(f"/{len(report.findings)}")

    def test_the_report_states_that_nothing_was_summarised_by_a_model(self) -> None:
        assert "not summarised" in research(question="q", symbol="X", now=AT).render() or \
            "nothing here is summarised by a model" in research(
                question="q", symbol="X", now=AT).render()

    def test_concerns_get_their_own_section(self) -> None:
        report = research(
            question="q", symbol="X", evidence=[_Ev("news")],
            impact=_Impact(beta_before=0.2, beta_after=0.9, risk_share_before=0.1,
                           risk_share_after=0.2),
            stress=_Stress(), cost_bps=Decimal("12"), now=AT,
        )
        assert "What argues against it" in report.render()

    def test_it_serialises_whole(self) -> None:
        built = research(question="q", symbol="NVDAUSDT", now=AT)
        got = built.as_dict()
        assert got["symbol"] == "NVDAUSDT"
        assert len(got["findings"]) == len(built.findings)
        assert got["verdict"] == "insufficient"


class TestBookParsing:
    def test_a_book_is_parsed(self) -> None:
        assert parse_book("AAPLUSDT=0.5,MSFTUSDT=0.5") == {
            "AAPLUSDT": 0.5, "MSFTUSDT": 0.5
        }

    def test_whitespace_and_case_are_handled(self) -> None:
        assert parse_book(" aaplusdt = 1.0 ") == {"AAPLUSDT": 1.0}

    def test_a_malformed_entry_is_refused(self) -> None:
        with pytest.raises(ValueError, match="SYMBOL=weight"):
            parse_book("AAPLUSDT")

    def test_an_empty_book_is_empty_not_an_error(self) -> None:
        assert parse_book("") == {}


def test_the_handbook_deliverable_runs_end_to_end() -> None:
    """One question, chain, actionable insight — Track 3's required material."""
    report = research(
        question="Should I add NVDAUSDT to this book?",
        symbol="NVDAUSDT",
        evidence=[_Ev("news"), _Ev("sec-edgar")],
        impact=_Impact(beta_before=0.309, beta_after=0.457, risk_share_before=0.1,
                       risk_share_after=0.24, effective_positions_after=2.37,
                       max_correlation=("MSFTUSDT", 0.10)),
        stress=_Stress(results=9, failures=0),
        cost_bps=Decimal("12"), expected_edge_bps=Decimal("40"),
        now=AT,
    )
    assert report.verdict is Verdict.ADD_SMALLER
    assert len(report.available) == 5
    text = report.render()
    assert "Should I add NVDAUSDT" in text
    assert "ADD_SMALLER" in text
    assert "desk/portfolio.py:assess" in text

class TestAnAbsenceSaysWhichKindItIs:
    """The defect this closes put a false sentence in the Track 3 deliverable.

    The report said "no comparable historical state was found within the distance threshold" and
    "no consensus estimate was available for this name". Both assert that a search ran. Neither
    search had run: the orchestrator's CLI initialised both inputs to None and never assigned
    them. An absence is only honest if it distinguishes "looked and found nothing" from "never
    looked".
    """

    def test_a_step_that_was_never_run_says_so(self) -> None:
        from argus.desk.research import NOT_ATTEMPTED, analogue_step

        got = analogue_step(None, attempted=False)
        assert not got.available
        assert NOT_ATTEMPTED in got.headline

    def test_a_step_that_ran_and_found_nothing_says_that_instead(self) -> None:
        from argus.desk.research import NOT_ATTEMPTED, analogue_step

        got = analogue_step(None, attempted=True)
        assert not got.available
        assert NOT_ATTEMPTED not in got.headline
        assert "distance threshold" in got.headline

    def test_the_two_headlines_are_different_sentences(self) -> None:
        from argus.desk.research import analogue_step, expectation_step

        for step in (analogue_step, expectation_step):
            assert step(None, attempted=False).headline != step(None, attempted=True).headline

    def test_the_expectation_step_names_why_it_was_not_attempted(self) -> None:
        """And it must not claim the data does not exist.

        The first version of this message said "no consensus source is wired for rTokens", which
        was wrong: `market/estimates.py` fetches Yahoo consensus behind a cookie-and-crumb
        handshake and was built because the desk itself asked for it at ledger seq 41. Declaring a
        capability absent without checking the module list is the error this wording closes.
        """
        from argus.desk.research import expectation_step

        headline = expectation_step(None, attempted=False).headline
        assert "market/estimates.py" in headline
        assert "not that the data does not exist" in headline

    def test_a_report_defaults_to_attempted_so_silence_is_never_assumed(self) -> None:
        """If the default were "not attempted", a caller that genuinely searched and found nothing
        would have its real negative result mislabelled."""
        from argus.desk.research import research

        report = research(question="q", symbol="NVDAUSDT")
        analogue = next(f for f in report.findings if f.step == "historical analogue")
        assert "distance threshold" in analogue.headline
