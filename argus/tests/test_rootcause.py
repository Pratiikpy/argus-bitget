"""Root-cause diagnosis tests.

The property under test is that a diagnosis is *checkable*. A cause with no decision ids behind it
cannot be verified and therefore cannot be wrong, which makes it worse than no diagnosis at all.
Every diagnosis here must name the decisions that support it and a remedy that can be evaluated
against a future one.

The second property is restraint: when the losses look like variance, the report must say so rather
than inventing a pattern. A system that always finds a cause is a horoscope.
"""

from __future__ import annotations

from decimal import Decimal

from argus.desk.rootcause import (
    MIN_FOR_DIAGNOSIS,
    diagnose,
)
from argus.desk.workbench import Autopsy


def _autopsy(
    ident: str,
    *,
    predicted: str = "up",
    realised: str = "up",
    predicted_bps: int = 40,
    realised_bps: int = 40,
    confidence: float = 0.7,
    ignored: tuple[str, ...] = (),
) -> Autopsy:
    return Autopsy(
        decision_id=ident,
        thesis="fixture",
        predicted_direction=predicted,
        realised_direction=realised,
        predicted_magnitude_bps=predicted_bps,
        realised_magnitude_bps=realised_bps,
        stated_confidence=confidence,
        evidence_ignored=ignored,
    )


def _overconfident(n: int = 10) -> list[Autopsy]:
    """States 0.9, right 30% of the time."""
    return [
        _autopsy(
            f"d{i}",
            predicted="up",
            realised="up" if i < int(n * 0.3) else "down",
            confidence=0.9,
        )
        for i in range(n)
    ]


def _sizing_errors(n: int = 10) -> list[Autopsy]:
    """Direction right every time, size badly wrong most of the time."""
    return [
        _autopsy(
            f"s{i}", predicted="up", realised="up",
            predicted_bps=100, realised_bps=100 if i < 3 else 15, confidence=0.6,
        )
        for i in range(n)
    ]


class TestTheFloor:
    def test_too_few_decisions_refuses_with_the_count(self) -> None:
        report = diagnose([_autopsy(f"d{i}") for i in range(MIN_FOR_DIAGNOSIS - 1)])
        assert report.usable is False
        assert "are needed" in report.refused

    def test_the_refusal_renders_as_a_recordable_line(self) -> None:
        report = diagnose([_autopsy("d1")])
        assert report.render()[0].startswith("[root-cause]")

    def test_an_empty_history_does_not_raise(self) -> None:
        assert diagnose([]).usable is False


class TestRestraint:
    """A system that always finds a cause is a horoscope."""

    def test_clean_decisions_produce_no_diagnosis(self) -> None:
        report = diagnose([_autopsy(f"d{i}", confidence=0.6) for i in range(10)])
        assert report.usable is True
        assert report.diagnoses == ()

    def test_it_says_the_losses_look_like_variance(self) -> None:
        report = diagnose([_autopsy(f"d{i}", confidence=0.6) for i in range(10)])
        assert "variance rather than a repeatable mistake" in " ".join(report.render())

    def test_a_hedged_loser_is_miscalibrated_not_overconfident(self) -> None:
        """Wrong and unsure is a different failure from wrong and certain, and needs a different
        remedy: the problem is that the numbers carry no information, not that they are too bold."""
        humble = [
            _autopsy(f"h{i}", predicted="up", realised="down", confidence=0.35)
            for i in range(10)
        ]
        causes = {d.cause for d in diagnose(humble).diagnoses}
        assert "overconfidence" not in causes
        assert "miscalibration" in causes

    def test_the_two_calibration_faults_carry_different_remedies(self) -> None:
        bold = next(
            d for d in diagnose(_overconfident()).diagnoses if d.cause == "overconfidence"
        )
        hedged = next(
            d for d in diagnose([
                _autopsy(f"h{i}", predicted="up", realised="down", confidence=0.35)
                for i in range(10)
            ]).diagnoses if d.cause == "miscalibration"
        )
        assert bold.remedy != hedged.remedy
        assert "unusable for sizing" in hedged.remedy


class TestOverconfidence:
    def test_it_is_detected(self) -> None:
        causes = {d.cause for d in diagnose(_overconfident()).diagnoses}
        assert "overconfidence" in causes

    def test_it_names_the_gap_with_numbers(self) -> None:
        diagnosis = next(
            d for d in diagnose(_overconfident()).diagnoses if d.cause == "overconfidence"
        )
        assert "%" in diagnosis.detail
        assert "realised hit rate" in diagnosis.detail

    def test_the_remedy_points_at_the_gate_that_already_enforces_it(self) -> None:
        diagnosis = next(
            d for d in diagnose(_overconfident()).diagnoses if d.cause == "overconfidence"
        )
        assert "argus.risk.sizing" in diagnosis.remedy


class TestSizingRatherThanThesis:
    def test_right_direction_wrong_size_is_diagnosed_separately(self) -> None:
        causes = {d.cause for d in diagnose(_sizing_errors()).diagnoses}
        assert "sizing, not thesis" in causes

    def test_the_remedy_says_to_keep_the_entry_rule(self) -> None:
        """The distinction is the point: the thesis process is working."""
        diagnosis = next(
            d for d in diagnose(_sizing_errors()).diagnoses if d.cause == "sizing, not thesis"
        )
        assert "thesis process is working" in diagnosis.remedy

    def test_direction_errors_do_not_produce_a_sizing_diagnosis(self) -> None:
        wrong_way = [
            _autopsy(f"w{i}", predicted="up", realised="down", confidence=0.5)
            for i in range(10)
        ]
        causes = {d.cause for d in diagnose(wrong_way).diagnoses}
        assert "sizing, not thesis" not in causes


class TestFeeDrag:
    def test_fees_eating_a_third_of_gross_is_diagnosed_as_turnover(self) -> None:
        report = diagnose(
            [_autopsy(f"d{i}", confidence=0.6) for i in range(10)],
            gross_pnl=Decimal("1000"), fees=Decimal("400"),
        )
        diagnosis = next(d for d in report.diagnoses if d.cause == "fee drag")
        assert "turnover is the cost" in diagnosis.detail

    def test_the_remedy_corrects_the_obvious_wrong_fix(self) -> None:
        """Holding longer does not help: the round trip costs the same whenever you pay it."""
        report = diagnose(
            [_autopsy(f"d{i}", confidence=0.6) for i in range(10)],
            gross_pnl=Decimal("1000"), fees=Decimal("600"),
        )
        remedy = next(d for d in report.diagnoses if d.cause == "fee drag").remedy
        assert "held twice as long still pays the same round trip" in remedy

    def test_modest_fees_are_not_diagnosed(self) -> None:
        report = diagnose(
            [_autopsy(f"d{i}", confidence=0.6) for i in range(10)],
            gross_pnl=Decimal("1000"), fees=Decimal("50"),
        )
        assert "fee drag" not in {d.cause for d in report.diagnoses}

    def test_missing_figures_drop_the_diagnosis_rather_than_guessing(self) -> None:
        report = diagnose([_autopsy(f"d{i}", confidence=0.6) for i in range(10)])
        assert "fee drag" not in {d.cause for d in report.diagnoses}

    def test_a_losing_book_does_not_produce_a_fee_share(self) -> None:
        """Fees as a share of a negative gross is not a meaningful ratio."""
        report = diagnose(
            [_autopsy(f"d{i}", confidence=0.6) for i in range(10)],
            gross_pnl=Decimal("-500"), fees=Decimal("100"),
        )
        assert "fee drag" not in {d.cause for d in report.diagnoses}


class TestRepeatedBlindSpot:
    def test_the_same_ignored_evidence_across_losses_is_a_process_gap(self) -> None:
        rows = [
            _autopsy(f"b{i}", predicted="up", realised="down", confidence=0.5,
                     ignored=("short interest",))
            for i in range(6)
        ]
        diagnosis = next(
            d for d in diagnose(rows).diagnoses if d.cause == "repeated blind spot"
        )
        assert "short interest" in diagnosis.detail
        assert "required field" in diagnosis.remedy

    def test_evidence_ignored_on_winners_does_not_count(self) -> None:
        """Skipping something and being right anyway is not the same failure."""
        rows = [
            _autopsy(f"g{i}", predicted="up", realised="up", confidence=0.6,
                     ignored=("short interest",))
            for i in range(10)
        ]
        assert "repeated blind spot" not in {d.cause for d in diagnose(rows).diagnoses}

    def test_one_off_omissions_are_not_a_pattern(self) -> None:
        rows = [
            _autopsy(f"o{i}", predicted="up", realised="down", confidence=0.5,
                     ignored=(f"thing {i}",))
            for i in range(8)
        ]
        assert "repeated blind spot" not in {d.cause for d in diagnose(rows).diagnoses}


class TestEveryDiagnosisIsCheckable:
    def test_each_one_names_the_decisions_behind_it(self) -> None:
        """A cause with nothing behind it cannot be wrong, which makes it useless."""
        report = diagnose(_overconfident(), gross_pnl=Decimal("1000"), fees=Decimal("500"))
        assert report.diagnoses
        for diagnosis in report.diagnoses:
            assert diagnosis.evidence, f"{diagnosis.cause} has no supporting decisions"

    def test_each_one_carries_a_remedy(self) -> None:
        report = diagnose(_overconfident(), gross_pnl=Decimal("1000"), fees=Decimal("500"))
        for diagnosis in report.diagnoses:
            assert len(diagnosis.remedy) > 20

    def test_primary_causes_sort_before_contributing_ones(self) -> None:
        report = diagnose(_overconfident(), gross_pnl=Decimal("1000"), fees=Decimal("900"))
        severities = [d.severity for d in report.diagnoses]
        assert severities == sorted(severities, key=lambda s: s != "primary")

    def test_the_report_serialises(self) -> None:
        payload = diagnose(_overconfident()).as_dict()
        assert payload["usable"] is True
        for entry in payload["diagnoses"]:
            assert {"cause", "severity", "decisions", "detail", "remedy"} <= set(entry)

    def test_a_rendered_line_carries_cause_evidence_and_remedy(self) -> None:
        line = diagnose(_overconfident()).render()[0]
        assert "decisions" in line and "->" in line
