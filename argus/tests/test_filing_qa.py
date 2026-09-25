"""The XBRL filing engine, replayed offline from the committed SEC snapshot.

The questions are written here, not copied from FinanceBench; the companies and years are ones
the snapshot holds, so every figure below comes from the company's own filed XBRL.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from argus.eval import financebench_xbrl as fb
from argus.market.statement_facts import CompanyFactsSource, ConceptSource, FaceStatementSource
from argus.research.filing_qa import CompanyResolver, FilingQA

SNAP = fb.SNAPSHOT_DIR


@pytest.fixture(scope="module")
def qa() -> FilingQA:
    return FilingQA(
        facts=CompanyFactsSource(snapshot_dir=SNAP / "facts", offline=True),
        companies=CompanyResolver(snapshot=SNAP / "companies.json", offline=True),
        faces=FaceStatementSource(snapshot_dir=SNAP / "faces", offline=True),
        concepts=ConceptSource(snapshot_dir=SNAP / "concepts", offline=True),
    )


class TestAFiledFigure:
    def test_a_line_item_in_the_unit_asked(self, qa: FilingQA) -> None:
        a = qa.answer("How much did 3M spend on capital expenditure in FY2018, in USD millions?")
        assert a.status == "answered"
        assert a.value == pytest.approx(1577.0)
        assert a.unit == "USD millions" and a.anchor_accn

    def test_a_rate_that_changed_is_given_in_points_with_both_years(self, qa: FilingQA) -> None:
        a = qa.answer("By how much did American Express's effective tax rate change between "
                      "FY2021 and FY2022?")
        assert a.status == "answered"
        assert a.unit == "percentage points" and a.value == pytest.approx(-3.0)
        assert "24.6% in FY2021" in a.text and "21.6% in FY2022" in a.text

    def test_did_it_grow_is_answered_as_the_change(self, qa: FilingQA) -> None:
        a = qa.answer("Did Pfizer's net PP&E grow from FY2020 to FY2021?")
        assert a.status == "answered" and a.fiscal_years == (2020, 2021)
        assert a.value is not None and a.value > 0
        assert "in FY2020" in a.text and "in FY2021" in a.text


class TestWhatItRefusesAndWhy:
    """Each of these was answered with a figure before 2026-09-26, and each figure was wrong."""

    def test_a_quarter_is_not_read_from_an_annual_filing(self, qa: FilingQA) -> None:
        a = qa.answer("Did CVS Health pay a dividend in Q2 of FY2022?")
        assert a.status == "abstained" and "10-Q" in a.reason

    def test_a_cause_is_prose_not_a_number(self, qa: FilingQA) -> None:
        a = qa.answer("What drove Ulta Beauty's higher inventories in FY2023?")
        assert a.status == "abstained" and "cause" in a.reason

    def test_an_unfiled_numerator_is_not_divided_by_itself(self, qa: FilingQA) -> None:
        a = qa.answer("Was Ulta Beauty's wages expense as a percent of net sales higher in "
                      "FY2023?")
        assert a.status == "abstained"
        assert "not a line this engine reads" in a.reason

    def test_an_arithmetic_phrase_the_formula_does_not_use_is_an_abstention(
            self, qa: FilingQA) -> None:
        a = qa.answer("What was 3M's FY2018 capital expenditure in USD millions, excluding "
                      "acquisitions?")
        assert a.status == "abstained" and "does not use" in a.reason


class TestTheEvaluation:
    def test_gold_figures_are_read_with_their_sign(self) -> None:
        assert fb.gold_numbers("$1577.00") == [1577.0]
        assert fb.gold_numbers("-$1,561M") == [-1561.0]
        assert fb.gold_numbers("from 24.6% to 21.6%")[::1] == [24.6, 21.6]

    def test_the_tolerance_rule(self) -> None:
        assert fb.close(8.74, 8.70) and not fb.close(8.9, 8.70)
        assert fb.close(0.39, 0.40) and not fb.close(0.37, 0.40)

    def test_a_non_figure_question_needs_a_written_grade(self) -> None:
        row = {"financebench_id": "x", "question_type": "novel-generated", "answer": "Yes."}
        assert fb.grade(row, "answered", 1.0, "1.0") == "ungraded"
        assert fb.grade(row, "abstained", None, "") == "abstained"

    def test_the_rivals_are_tallied_from_their_own_grades(self, tmp_path: Path) -> None:
        rows = [{"financebench_id": "a", "label": "Correct Answer"},
                {"financebench_id": "b", "label": "Refusal"},
                {"financebench_id": "c", "label": "Incorrect Answer"}]
        (tmp_path / "model_mode.jsonl").write_text("\n".join(json.dumps(r) for r in rows),
                                                   encoding="utf-8")
        got = fb.rival_tallies({"x": {"a", "b"}}, tmp_path)
        assert got == {"model_mode": {"x": {"correct": 1, "incorrect": 0, "refusal": 1}}}

    def test_a_missing_clone_is_said_not_skipped(self, tmp_path: Path) -> None:
        with pytest.raises(fb.BenchUnavailable):
            fb.load_questions(tmp_path / "nope.jsonl")


class TestTheCommittedArtefact:
    report = json.loads(fb.REPORT_PATH.read_text(encoding="utf-8"))

    def test_the_headline(self) -> None:
        head = self.report["headline"]
        assert head["argus_correct_of_50"] == 50
        assert head["best_rival_correct_of_50"] == 46
        assert head["rivals_with_more_correct"] == []
        assert self.report["argus"]["other"]["wrong"] == 0

    def test_it_says_it_is_not_held_out(self) -> None:
        assert "seen" in self.report["not_held_out"]

    def test_it_carries_no_gold_answer_text(self) -> None:
        text = json.dumps(self.report)
        # the questions and gold answers stay in FinanceBench's repository
        assert "financial analyst" not in text
        assert '"question"' not in text

    def test_the_guard_ablation_is_graded(self) -> None:
        extra = self.report["ablation_coverage_guard_off"]["extra_answers"]
        grades = [e["grade"] for e in extra]
        assert grades.count("wrong") == 5 and grades.count("correct") == 1


class TestTheConsoleRoute:
    """The console answers a fiscal-year figure from the engine, and leaves the rest alone."""

    @pytest.fixture(autouse=True)
    def offline_engine(self, qa: FilingQA, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.lui import server

        monkeypatch.setattr(server, "_XBRL", qa)

    def test_a_fiscal_year_figure_comes_with_its_formula_and_filed_lines(self) -> None:
        from argus.lui import server
        from argus.lui.provenance import labels

        p = server.handle_ask("What was 3M's capital expenditure in FY2018, in USD millions?",
                              [], visitor="test", book="")
        assert p["classified_by"] == "research-xbrl"
        lines = p["lines"]
        assert "$1,577.00 (USD millions)" in lines[0] and "no model wrote the figure" in lines[0]
        filed = [ln for ln in lines if ln.startswith("Filed: ")]
        assert filed and "us-gaap:PaymentsToAcquirePropertyPlantAndEquipment" in filed[0]
        assert labels(filed) == ["live"] * len(filed)

    def test_an_abstention_takes_the_usual_route(self) -> None:
        from argus.lui import server

        p = server.handle_ask("What drove Ulta Beauty's higher inventories in FY2023?", [],
                              visitor="test", book="")
        assert p.get("classified_by") != "research-xbrl"
