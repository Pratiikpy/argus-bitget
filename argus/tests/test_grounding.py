"""Numeric-grounding tests.

The property under test is that a number appearing in a thesis can be traced to something that
produced it. Not that it is *correct* — correctness is the backtest's job and settlement's job —
but that it is attributable, which is the weaker property and the checkable one.

The second property is restraint about what counts as a claim. Years, ordinals, identifiers and
time-window labels are not assertions about the market, and flagging them would bury the real
findings under noise until a reader learned to ignore the report.
"""

from __future__ import annotations

import pytest

from argus.agents.grounding import TOLERANCE, check, extract

FACTS = {
    "hurdle_bps": 18.80,
    "move_24h_bps": 4.4,
    "round_trip_bps": 12.0,
    "hours_to_discovery": 46.0,
}

# Verbatim from the live ledger, seq 26.
REAL_THESIS = (
    "Weekend session with 46+ hours to genuine price discovery, no hedge menu available, and "
    "news mix is balanced with no surprise catalyst. The 24h change of +4.4bps is below the "
    "18.80bps total hurdle, making any directional bet a negative-expectancy proposition."
)


class TestAgainstTheRealRecord:
    """The only fixture that settles whether this is usable."""

    def test_a_real_thesis_grounds_completely(self) -> None:
        report = check(REAL_THESIS, facts=FACTS)
        assert report.grounded, [r.figure.raw for r in report.unresolved]
        assert report.coverage == 1.0

    def test_it_finds_the_figures_that_matter(self) -> None:
        raws = {f.raw for f in extract(REAL_THESIS)}
        assert "+4.4bps" in raws
        assert "18.80bps" in raws

    def test_the_time_window_label_is_not_treated_as_a_claim(self) -> None:
        """"the 24h change of +4.4bps" holds two numbers and only one is an assertion."""
        assert "24" not in {f.raw for f in extract(REAL_THESIS)}

    def test_a_thesis_with_no_figures_is_grounded_vacuously(self) -> None:
        report = check("No actionable edge exists; any token drift is noise.", facts=FACTS)
        assert report.grounded
        assert "states no figures" in report.render()[0]


class TestItCatchesAnUnsupportedNumber:
    def test_an_invented_figure_does_not_resolve(self) -> None:
        report = check("The spread is 250bps, far beyond the hurdle.", facts=FACTS)
        assert not report.grounded
        assert report.unresolved[0].figure.value == 250.0

    def test_the_report_names_the_offending_figure(self) -> None:
        report = check("Expected move is 97bps.", facts=FACTS)
        assert "97bps" in " ".join(report.render())

    def test_it_says_why_that_matters(self) -> None:
        report = check("Expected move is 97bps.", facts=FACTS)
        assert "unsupported fact to enter the record" in " ".join(report.render())

    def test_coverage_is_a_fraction_not_a_verdict(self) -> None:
        report = check("The hurdle is 18.80bps and the target is 900bps.", facts=FACTS)
        assert report.coverage == pytest.approx(0.5)


class TestTolerantButBounded:
    def test_a_rounded_quote_resolves(self) -> None:
        """A model writes 18.8 where the computed value is 18.80."""
        assert check("hurdle 18.8bps", facts=FACTS).grounded

    def test_a_materially_different_number_does_not_pass_as_rounding(self) -> None:
        assert not check("hurdle 25bps", facts=FACTS).grounded

    def test_the_tolerance_is_tight_enough_to_be_meaningful(self) -> None:
        assert TOLERANCE <= 0.05, "a loose tolerance makes the check decorative"

    def test_a_percentage_resolves_against_a_fraction(self) -> None:
        """0.0203 is quoted as "2%"; refusing that would flag correct prose."""
        assert check("the move was 2%", facts={"move": 0.0203}).grounded

    def test_bps_resolves_against_a_fraction(self) -> None:
        assert check("the move was 203bps", facts={"move": 0.0203}).grounded

    def test_a_zero_valued_fact_does_not_divide_by_zero(self) -> None:
        assert check("position is 0", facts={"position": 0.0}).grounded


class TestWhatIsNotAClaim:
    @pytest.mark.parametrize(
        "text",
        [
            "the 2026 filing season",
            "the 3rd consecutive session",
            "over the last 24h",
            "in the past 7 days",
            "a 15m candle",
        ],
    )
    def test_these_are_not_treated_as_market_claims(self, text: str) -> None:
        assert extract(text) == (), f"{text!r} produced a figure"

    def test_a_real_claim_beside_an_excluded_one_still_registers(self) -> None:
        figures = extract("over the last 24h the move was 44bps")
        assert [f.raw for f in figures] == ["44bps"]


class TestEvidenceBackedFigures:
    def test_a_number_carried_by_evidence_resolves_to_its_id(self) -> None:
        report = check(
            "the filing reports revenue of 35100 million",
            facts={},
            evidence_values=[("edgar-0001", 35_100.0)],
        )
        assert report.grounded
        assert report.resolutions[0].source == "edgar-0001"

    def test_the_resolution_names_which_source_matched(self) -> None:
        report = check("hurdle 18.80bps", facts=FACTS)
        assert report.resolutions[0].source == "hurdle_bps"

    def test_the_report_serialises_every_figure(self) -> None:
        payload = check(REAL_THESIS, facts=FACTS).as_dict()
        assert payload["figures"] == len(payload["detail"])
        for entry in payload["detail"]:
            assert {"raw", "value", "unit", "context", "resolved", "source"} <= set(entry)

    def test_every_figure_carries_its_surrounding_text(self) -> None:
        """A bare number in a report is as unauditable as a bare number in a thesis."""
        for figure in extract(REAL_THESIS):
            assert figure.context
