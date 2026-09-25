"""The factor grammar against general-purpose rivals (`argus.eval.general_grammar_comparison`).

Runs the ARGUS side live and the rivals from their write-through recording
(``data/general_grammar_rivals_recorded.json``, made by CEL / Polars / SymPy in an isolated venv).
The assertions pin what was measured — including the axes ARGUS lost — so a later change that
quietly flatters the comparison fails here.
"""

from __future__ import annotations

import ast
import json
from typing import Any

import pytest

from argus.eval import artefact
from argus.eval import general_grammar_comparison as ggc
from argus.research.grammar import EXPANDED, ORIGINAL_EIGHT
from argus.research.grammar_text import parse


@pytest.fixture(scope="module")
def report() -> dict[str, Any]:
    mp = pytest.MonkeyPatch()
    mp.delenv(ggc.RIVAL_PYTHON_ENV, raising=False)
    try:
        return ggc.main()
    finally:
        mp.undo()


class TestSameInput:
    def test_the_rivals_were_handed_the_shipped_factors_not_paraphrases(
        self, report: dict[str, Any]
    ) -> None:
        assert report["same_trees"] and all(report["same_trees"].values())

    def test_the_recording_matches_this_corpus(self, report: dict[str, Any]) -> None:
        assert report["rival_source"] == "recorded"
        assert report["corpus_digest"] == ggc.build_corpus()["digest"]
        assert set(report["rival_versions"]) == {"cel-expr-python", "polars", "sympy"}

    def test_a_stale_recording_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(ggc.RIVAL_PYTHON_ENV, raising=False)
        corpus = ggc.build_corpus()
        corpus["digest"] = "0" * 64
        with pytest.raises(ggc.GeneralGrammarComparisonError, match="different corpus"):
            ggc.load_rivals(corpus)

    def test_every_neutral_tree_is_valid_argus_text(self) -> None:
        for case in ggc.TYPECHECK_CASES:
            ggc.argus_text(case["tree"])
        for pair in ggc.DEDUP_PAIRS:
            parse(ggc.argus_text(pair["a"]))
            parse(ggc.argus_text(pair["b"]))
        shipped = {**ORIGINAL_EIGHT, **EXPANDED}
        for name, tree in ggc.TIMED_FACTORS:
            assert parse(ggc.argus_text(tree)).canonical() == shipped[name].canonical()


class TestTypeChecking:
    def test_argus_and_cel_reject_every_ill_typed_case_before_evaluation(
        self, report: dict[str, Any]
    ) -> None:
        tc = report["typecheck"]
        for system in ("argus", "cel"):
            assert tc[system]["ill_typed_rejected_statically"] == tc["ill_typed_total"] == 10
            assert tc[system]["controls_falsely_rejected"] == []

    def test_polars_evaluates_some_ill_typed_cases_silently(self, report: dict[str, Any]) -> None:
        pl = report["typecheck"]["polars"]
        assert pl["ill_typed_silently_evaluated"] == [
            "arithmetic_on_boolean", "branches_disagree", "compare_boolean_to_number"]
        assert pl["controls_falsely_rejected"] == []


class TestDuplicateIdentity:
    def test_canonical_missed_the_algebraic_identities_sympy_merges(
        self, report: dict[str, Any]
    ) -> None:
        dd = report["dedup"]
        assert dd["argus_canonical"]["equivalent_missed"] == [
            "associativity", "double_negation", "relational_flip", "subtract_negation"]
        assert dd["sympy"]["equivalent_merged"] == 7
        assert dd["sympy"]["equivalent_missed"] == ["symmetric_corr"]

    def test_normal_form_merges_all_and_falsely_merges_none(self, report: dict[str, Any]) -> None:
        dd = report["dedup"]
        assert dd["argus_normal_form"]["equivalent_merged"] == dd["equivalent_total"] == 8
        for system in ("argus_canonical", "argus_normal_form", "cel", "polars", "sympy"):
            assert dd[system]["distinct_falsely_merged"] == []

    def test_argus_merges_strictly_different_constants_by_its_own_convention(
        self, report: dict[str, Any]
    ) -> None:
        """Reported, not hidden: the 6-significant-digit rounding merges 0.1 and 0.1000004."""
        dd = report["dedup"]
        assert dd["argus_canonical"]["convention_pairs_merged"] == [
            "float_near_4e-7", "float_noise_1e-11"]
        assert dd["sympy"]["convention_pairs_merged"] == []


class TestNumerics:
    def test_polars_agrees_on_every_full_window_it_can_express(
        self, report: dict[str, Any]
    ) -> None:
        nm = report["numeric"]
        assert nm["full_window_agree"] == nm["full_window_expressible"] == 33
        assert nm["polars_inexpressible"] == [
            "argmax_24", "argmax_6", "argmax_96", "argmin_24", "argmin_6", "argmin_96"]
        assert nm["full_window_bars_undefined_in_polars"] == 0


class TestCostBudget:
    def test_argus_refuses_where_cel_refuses_and_the_ablation_accepts_everything(
        self, report: dict[str, Any]
    ) -> None:
        argus_rows = report["budget"]["argus"]["rows"]
        cel_rows = report["budget"]["cel"]
        assert [r["validate"] for r in argus_rows] == ["accepted"] + ["refused"] * 3
        assert [r["verdict"] for r in cel_rows] == ["evaluated"] + ["refused"] * 3
        assert all(r["validate_without_cost_bound"] == "accepted" for r in argus_rows)
        assert argus_rows[-1]["per_bar_path_estimated_seconds_per_bar"] > 3600

    def test_the_series_path_matches_polars_on_the_deepest_chain(
        self, report: dict[str, Any]
    ) -> None:
        for row in report["budget"]["argus"]["rows"]:
            assert row["abs_diff_vs_polars_last"] < 1e-9 * abs(row["series_last"])


class TestTiming:
    def test_series_is_bit_identical_and_polars_is_faster_everywhere(
        self, report: dict[str, Any]
    ) -> None:
        rows = report["timing"]["rows"]
        assert all(r["series_identical_to_per_bar"] for r in rows)
        expressible = [r for r in rows if "polars_seconds" in r]
        assert len(expressible) == 7
        assert all(r["signal_agreement"] == 1.0 for r in expressible)
        # The loss, pinned: Polars beats the columnar ARGUS path on every factor it can express.
        assert all(r["polars_speedup_over_argus_series"] > 1.0 for r in expressible)


def test_scope_statement_names_the_losses(report: dict[str, Any]) -> None:
    text = report["scope_statement"]
    assert "NOT CLAIMED: that ARGUS is faster than Polars" in text
    assert "NOT RUN" in text


def test_the_committed_artefacts_are_strict_json() -> None:
    assert artefact.is_strict(ggc.OUT)
    assert artefact.is_strict(ggc.RECORDED)


def test_the_committed_artefacts_carry_no_local_path() -> None:
    for path in (ggc.OUT, ggc.RECORDED):
        text = path.read_text(encoding="utf-8").lower()
        for marker in ("c:\\\\", "c:/", "/users/", "appdata", artefact.SCRATCH_DIR,
                       "site-packages"):
            assert marker not in text, (path.name, marker)


def test_the_rival_runner_imports_nothing_from_argus() -> None:
    """The runner executes under the rival's interpreter, where ARGUS is not installed; an import
    from ``argus`` would also let ARGUS code leak into what is reported as the rival's output."""
    tree = ast.parse(ggc.RUNNER.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module.split(".")[0])
    assert "argus" not in imported
    assert imported <= {"__future__", "importlib", "json", "statistics", "sys", "time",
                        "warnings", "collections", "typing"}


def test_the_argus_side_of_the_recording_is_rerun_not_copied(report: dict[str, Any]) -> None:
    """Every ARGUS number is computed in this process; only rival output comes from the file."""
    recorded = json.loads(ggc.RECORDED.read_text(encoding="utf-8"))
    assert "argus" not in recorded["typecheck"]
    assert set(recorded["dedup"]) == {"cel", "polars", "sympy"}
    assert report["typecheck"]["argus"]["rows"] == ggc.argus_typecheck()


def test_render(report: dict[str, Any]) -> None:
    text = ggc.render(report)
    assert "type checking" in text and "duplicate identity" in text
