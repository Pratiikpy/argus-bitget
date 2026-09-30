"""Offline arithmetic checks on `data/memory_comparison.json` — no network, no mem0 import.

`eval/memory_comparison.py` is the only place that calls real mem0ai (out of process, via
`eval/baselines/mem0_loader.py`) or the live Bitget-backed `argus.lui.server.handle_ask`. This file
never does either: it reads the JSON artefact those runs already wrote and checks that the numbers
inside it are internally consistent — the same discipline `test_mandate_comparison.py` and its
siblings apply to a live rerun, applied here to a static artefact instead, per this task's own
instruction that the test suite run offline.

**Four rounds, not two.** `tuning` is `eval/memory_eval.py`'s own fixtures. `held_out_1` (renamed
from plain `held_out` by `freeze_held_out_2()`) was genuinely held out for its own round, but
stopped being held-out the moment its structural cases informed a real edit to `lui/memory.py`
(2026-09-30) — it carries `held_out` and a `note` saying so, and is no longer compared as
out-of-sample. `held_out_2` is the second, genuinely blind round written without opening
`lui/memory.py` after that edit; it has no `argus_contradiction`/`mem0_contradiction` blocks (that
metric was not re-run this round) and its `structural_cases` carry raw extracted `Fact` lists
rather than a `argus_kinds_found` pass/fail count, per `BlindStructuralCase`'s own docstring.
`held_out_3` is the third, genuinely blind round, written without opening the new
`lui/memory_model.py` (or `lui/memory.py`) — ARGUS's side now spends real Qwen calls too
(`memory_model.combined()`), so its `statement_rows`/`not_about_me_rows` stay mem0-side (the
established convention every round shares) while its OWN per-item detail lives under
`argus_statement_rows`/`argus_not_about_me_rows`, and its `structural_cases` carry ARGUS's raw
facts under the key `argus_facts` (held_out_2 used `facts`) with the FULL `Fact` shape including
`at`/`price_at`. `held_out_3` itself stopped being held-out once its OWN rows informed a second
`lui/memory_model.py` change — `freeze_held_out_4()` relabels its `note` in place (no key rename,
unlike `held_out`→`held_out_1`, since `held_out_3` already had its own distinct name). `held_out_4`
is the fourth, genuinely blind round, same shape as `held_out_3`.

Skipped, not failed, when the artefact has not been generated yet (`python -m
argus.eval.memory_comparison`, or `argus.eval.memory_comparison.main()`/`main_held_out_2()`, which
need the isolated mem0 venv at `~/.venvs/mem0` and network access to the Qwen
endpoint and live Bitget candles) — a missing artefact is an environment fact, not a regression.
"""

from __future__ import annotations

import json

import pytest

from argus.eval.memory_comparison import (
    HELD_OUT_2_NOT_ABOUT_ME,
    HELD_OUT_2_STATEMENTS,
    HELD_OUT_2_STRUCTURAL,
    HELD_OUT_2_TASKS,
    HELD_OUT_3_NOT_ABOUT_ME,
    HELD_OUT_3_STATEMENTS,
    HELD_OUT_3_STRUCTURAL,
    HELD_OUT_3_TASKS,
    HELD_OUT_4_NOT_ABOUT_ME,
    HELD_OUT_4_STATEMENTS,
    HELD_OUT_4_STRUCTURAL,
    HELD_OUT_4_TASKS,
    HELD_OUT_NOT_ABOUT_ME,
    HELD_OUT_STATEMENTS,
    HELD_OUT_STRUCTURAL,
    HELD_OUT_TASKS,
    NOT_ABOUT_ME,
    REPORT_PATH,
    TUNING_STATEMENTS,
    TUNING_STRUCTURAL,
    TUNING_TASKS,
)

pytestmark = pytest.mark.skipif(
    not REPORT_PATH.is_file(),
    reason=f"{REPORT_PATH} not generated yet — run `python -m argus.eval.memory_comparison`",
)

_SPLITS = ("tuning", "held_out_1")
"""Splits that carry the full comparison, including contradiction — `held_out_2`/`held_out_3`/
`held_out_4` do not (see module docstring) and get their own test classes below."""
_ALL_SPLITS = ("tuning", "held_out_1", "held_out_2", "held_out_3", "held_out_4")
"""Splits that carry recall/false-positive/effect, the metrics every round has in common."""
_RECALL_SIDES = ("argus_recall", "mem0_recall")
_FP_SIDES = ("argus_false_positives", "mem0_false_positives")


@pytest.fixture(scope="module")
def report() -> dict:
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


class TestArtefactIsStrictJson:
    """The same class of defect `truth/artefact.py`'s own docstring describes finding elsewhere in
    this project: a bare `NaN`/`Infinity` token that only Python's own `json` module accepts."""

    def test_the_file_contains_no_nan_or_infinity_tokens(self) -> None:
        raw = REPORT_PATH.read_text(encoding="utf-8")
        assert "NaN" not in raw
        assert "Infinity" not in raw

    def test_the_file_parses_as_ordinary_json(self) -> None:
        json.loads(REPORT_PATH.read_text(encoding="utf-8"))  # raises if this is not valid JSON


class TestBudget:
    def test_qwen_calls_used_never_exceeds_what_was_planned(self, report: dict) -> None:
        assert report["qwen_calls_used"] <= report["qwen_calls_planned"]

    def test_qwen_calls_used_stays_within_the_400_call_cap(self, report: dict) -> None:
        assert report["qwen_calls_used"] <= 400

    def test_total_tokens_is_a_nonnegative_number(self, report: dict) -> None:
        assert report["qwen_total_tokens"] >= 0


_FIXTURES_BY_SPLIT = {
    "tuning": (TUNING_STATEMENTS, NOT_ABOUT_ME, TUNING_STRUCTURAL, TUNING_TASKS),
    "held_out_1": (HELD_OUT_STATEMENTS, HELD_OUT_NOT_ABOUT_ME, HELD_OUT_STRUCTURAL, HELD_OUT_TASKS),
    "held_out_2": (HELD_OUT_2_STATEMENTS, HELD_OUT_2_NOT_ABOUT_ME, HELD_OUT_2_STRUCTURAL,
                  HELD_OUT_2_TASKS),
    "held_out_3": (HELD_OUT_3_STATEMENTS, HELD_OUT_3_NOT_ABOUT_ME, HELD_OUT_3_STRUCTURAL,
                  HELD_OUT_3_TASKS),
    "held_out_4": (HELD_OUT_4_STATEMENTS, HELD_OUT_4_NOT_ABOUT_ME, HELD_OUT_4_STRUCTURAL,
                  HELD_OUT_4_TASKS),
}


class TestFixtureSizesMatchTheArtefact:
    """Pins the artefact's own counts to the real fixture sizes, so a silent fixture shrink — or
    a split whose chains never actually ran — shows up as a failing count, not a quieter total."""

    @pytest.mark.parametrize("split", _ALL_SPLITS)
    def test_recall_totals_match_statements(self, report: dict, split: str) -> None:
        statements, _nam, _struct, _tasks = _FIXTURES_BY_SPLIT[split]
        for side in _RECALL_SIDES:
            assert report[split][side]["total"] == len(statements)

    @pytest.mark.parametrize("split", _ALL_SPLITS)
    def test_false_positive_totals_match_not_about_me(self, report: dict, split: str) -> None:
        _stmt, nam, _struct, _tasks = _FIXTURES_BY_SPLIT[split]
        for side in _FP_SIDES:
            assert report[split][side]["total"] == len(nam)

    @pytest.mark.parametrize("split", _ALL_SPLITS)
    def test_structural_case_counts_match_the_fixtures(self, report: dict, split: str) -> None:
        _stmt, _nam, structural, _tasks = _FIXTURES_BY_SPLIT[split]
        assert len(report[split]["structural_cases"]) == len(structural)

    @pytest.mark.parametrize("split", _ALL_SPLITS)
    def test_task_counts_match_the_fixtures_on_both_sides(self, report: dict, split: str) -> None:
        _stmt, _nam, _struct, tasks = _FIXTURES_BY_SPLIT[split]
        assert report[split]["argus_effect"]["tasks"] == len(tasks)
        assert report[split]["mem0_effect"]["tasks"] == len(tasks)


class TestRecallArithmetic:
    @pytest.mark.parametrize("split", _ALL_SPLITS)
    @pytest.mark.parametrize("side", _RECALL_SIDES)
    def test_recall_rate_equals_recalled_over_total(self, report: dict, split: str,
                                                     side: str) -> None:
        d = report[split][side]
        expected = d["recalled"] / d["total"] if d["total"] else 0.0
        assert d["recall_rate"] == pytest.approx(expected, abs=1e-4)

    @pytest.mark.parametrize("split", _ALL_SPLITS)
    @pytest.mark.parametrize("side", _RECALL_SIDES)
    def test_recalled_is_between_zero_and_total(self, report: dict, split: str, side: str) -> None:
        d = report[split][side]
        assert 0 <= d["recalled"] <= d["total"]

    @pytest.mark.parametrize("split", _ALL_SPLITS)
    @pytest.mark.parametrize("side", _RECALL_SIDES)
    def test_missed_list_length_equals_total_minus_recalled(self, report: dict, split: str,
                                                             side: str) -> None:
        d = report[split][side]
        assert len(d["missed"]) == d["total"] - d["recalled"]


class TestFalsePositiveArithmetic:
    @pytest.mark.parametrize("split", _ALL_SPLITS)
    @pytest.mark.parametrize("side", _FP_SIDES)
    def test_rate_equals_count_over_total(self, report: dict, split: str, side: str) -> None:
        d = report[split][side]
        expected = d["false_positives"] / d["total"] if d["total"] else 0.0
        assert d["false_positive_rate"] == pytest.approx(expected, abs=1e-4)

    @pytest.mark.parametrize("split", _ALL_SPLITS)
    @pytest.mark.parametrize("side", _FP_SIDES)
    def test_examples_list_length_equals_the_false_positive_count(self, report: dict, split: str,
                                                                   side: str) -> None:
        d = report[split][side]
        assert len(d["examples"]) == d["false_positives"]


class TestStatementRowsReconcileWithTheSummary:
    """Per-statement detail and the summary counters must tell the same story — a summary that
    silently drifted from its own detail rows is worth catching even though both came from the same
    run, since a future edit to one path is easy to forget to mirror in the other."""

    @pytest.mark.parametrize("split", _ALL_SPLITS)
    def test_mem0_recalled_count_matches_fully_recalled_rows(
        self, report: dict, split: str,
    ) -> None:
        rows = report[split]["statement_rows"]
        recomputed = sum(1 for r in rows if r["fully_recalled"])
        assert recomputed == report[split]["mem0_recall"]["recalled"]

    @pytest.mark.parametrize("split", _ALL_SPLITS)
    def test_mem0_false_positive_count_matches_not_about_me_rows(self, report: dict,
                                                                  split: str) -> None:
        rows = report[split]["not_about_me_rows"]
        assert (sum(1 for r in rows if r["false_positive"])
               == report[split]["mem0_false_positives"]["false_positives"])

    @pytest.mark.parametrize("split", _ALL_SPLITS)
    def test_every_row_names_a_real_fixture_statement(self, report: dict, split: str) -> None:
        statements, _nam, _struct, _tasks = _FIXTURES_BY_SPLIT[split]
        expected_texts = {s.text for s in statements}
        row_texts = {r["text"] for r in report[split]["statement_rows"]}
        assert row_texts == expected_texts

    @pytest.mark.parametrize("split", _ALL_SPLITS)
    def test_facts_recalled_never_exceeds_facts_expected(self, report: dict, split: str) -> None:
        for row in report[split]["statement_rows"]:
            assert 0 <= row["facts_recalled"] <= row["facts_expected"]

    @pytest.mark.parametrize("split", _ALL_SPLITS)
    def test_fully_recalled_is_true_only_when_every_fact_was_found(self, report: dict,
                                                                    split: str) -> None:
        for row in report[split]["statement_rows"]:
            expected = row["facts_expected"] > 0 and row["facts_recalled"] == row["facts_expected"]
            assert row["fully_recalled"] == expected


class TestStructuralCases:
    """The three "mem0 should structurally win" cases per split — an unstructured fact, two facts
    of the same kind in one message, and a fact a leading question zeroes out entirely. Scoped to
    `tuning`/`held_out_1`: `held_out_2`'s structural rows carry a different shape (raw extracted
    facts, no `argus_kinds_found` pass/fail — see `TestHeldOut2StructuralCases` below)."""

    @pytest.mark.parametrize("split", _SPLITS)
    def test_argus_facts_found_equals_the_length_of_kinds_found(self, report: dict,
                                                                 split: str) -> None:
        for case in report[split]["structural_cases"]:
            assert case["argus_facts_found"] == len(case["argus_kinds_found"])

    @pytest.mark.parametrize("split", _SPLITS)
    def test_mem0_facts_recalled_is_between_zero_and_expected(self, report: dict,
                                                               split: str) -> None:
        for case in report[split]["structural_cases"]:
            assert 0 <= case["mem0_facts_recalled"] <= case["mem0_facts_expected"]

    @pytest.mark.parametrize("split", _SPLITS)
    def test_every_case_names_one_of_the_three_stated_gap_shapes(self, report: dict,
                                                                  split: str) -> None:
        allowed = {"leading_question_zeroes_extraction", "two_facts_same_kind_one_message",
                  "unstructured_fact_no_kind"}
        for case in report[split]["structural_cases"]:
            assert case["gap"] in allowed


class TestContradiction:
    @pytest.mark.parametrize("split", _SPLITS)
    def test_all_latest_win_matches_the_underlying_list(self, report: dict, split: str) -> None:
        d = report[split]["argus_contradiction"]
        assert d["all_latest_win"] == (all(d["latest_wins"]) if d["latest_wins"] else False)

    @pytest.mark.parametrize("split", _SPLITS)
    def test_all_replaces_recorded_matches_the_underlying_list(self, report: dict,
                                                                split: str) -> None:
        d = report[split]["argus_contradiction"]
        assert d["all_replaces_recorded"] == (
            all(d["replaces_recorded"]) if d["replaces_recorded"] else False
        )

    @pytest.mark.parametrize("split", _SPLITS)
    def test_kinds_checked_is_nonempty(self, report: dict, split: str) -> None:
        # Both P3's and P4's session-3 message state four facts; an empty list would mean the
        # contradiction check found nothing to compare, which is itself worth catching.
        assert len(report[split]["argus_contradiction"]["kinds_checked"]) > 0

    @pytest.mark.parametrize("split", _SPLITS)
    def test_latest_wins_and_replaces_recorded_are_the_same_length_as_kinds_checked(
        self, report: dict, split: str,
    ) -> None:
        d = report[split]["argus_contradiction"]
        assert len(d["latest_wins"]) == len(d["kinds_checked"])
        assert len(d["replaces_recorded"]) == len(d["kinds_checked"])

    @pytest.mark.parametrize("split", _SPLITS)
    def test_mem0_contradiction_never_reports_the_new_fact_missing_and_recalled_at_once(
        self, report: dict, split: str,
    ) -> None:
        d = report[split]["mem0_contradiction"]
        # `top_result_is_new_not_blended` can only be true when mem0 actually stored the new fact.
        if d["top_result_is_new_not_blended"]:
            assert d["new_fact_present"]


class TestEffect:
    @pytest.mark.parametrize("split", _ALL_SPLITS)
    def test_argus_changed_never_exceeds_task_count(self, report: dict, split: str) -> None:
        d = report[split]["argus_effect"]
        assert 0 <= d["changed_by_memory"] <= d["tasks"]

    @pytest.mark.parametrize("split", _ALL_SPLITS)
    def test_mem0_changed_never_exceeds_task_count(self, report: dict, split: str) -> None:
        d = report[split]["mem0_effect"]
        assert 0 <= d["changed_by_memory"] <= d["tasks"]

    @pytest.mark.parametrize("split", _ALL_SPLITS)
    def test_argus_memory_never_leaks_to_a_different_trader(self, report: dict, split: str) -> None:
        # ARGUS's memory is client-held and sent per request, never stored server-side — a second
        # visitor's turn must never carry the first visitor's remembered lines.
        assert report[split]["argus_effect"]["leaked_to_other_trader"] == 0

    @pytest.mark.parametrize("split", _ALL_SPLITS)
    def test_mem0_fresh_user_flag_matches_its_own_carried_count(self, report: dict,
                                                                 split: str) -> None:
        d = report[split]["mem0_effect"]
        assert d["fresh_user_never_carries"] == (d["fresh_user_carried_count"] == 0)

    @pytest.mark.parametrize("split", _ALL_SPLITS)
    def test_mem0_fresh_user_carried_count_never_exceeds_task_count(self, report: dict,
                                                                     split: str) -> None:
        d = report[split]["mem0_effect"]
        assert 0 <= d["fresh_user_carried_count"] <= d["tasks"]


class TestHeldOut1Relabelled:
    def test_the_old_held_out_key_no_longer_exists(self, report: dict) -> None:
        assert "held_out" not in report

    def test_held_out_1_carries_a_note_explaining_the_relabel(self, report: dict) -> None:
        note = report["held_out_1"].get("note", "")
        assert "tuning" in note.lower()


class TestHeldOut2FrozenRecord:
    """`held_out_2` must carry proof it was saved BEFORE either side ran: a timestamp, and the
    exact fixture texts, both written by `freeze_held_out_2()` before `main_held_out_2()` spent a
    single Qwen call or ran a single `extract()`."""

    def test_frozen_at_is_present_and_looks_like_a_timestamp(self, report: dict) -> None:
        frozen_at = report["held_out_2"]["frozen_at"]
        assert frozen_at and "T" in frozen_at and frozen_at.count("-") >= 2

    def test_frozen_statement_count_matches_the_live_fixture(self, report: dict) -> None:
        assert len(report["held_out_2"]["frozen_statements"]) == len(HELD_OUT_2_STATEMENTS)

    def test_frozen_not_about_me_count_matches_the_live_fixture(self, report: dict) -> None:
        assert len(report["held_out_2"]["frozen_not_about_me"]) == len(HELD_OUT_2_NOT_ABOUT_ME)

    def test_frozen_task_count_matches_the_live_fixture(self, report: dict) -> None:
        assert len(report["held_out_2"]["frozen_tasks"]) == len(HELD_OUT_2_TASKS)

    def test_frozen_structural_count_matches_the_live_fixture(self, report: dict) -> None:
        assert len(report["held_out_2"]["frozen_structural"]) == len(HELD_OUT_2_STRUCTURAL)

    def test_frozen_statement_texts_match_the_live_fixture_exactly(self, report: dict) -> None:
        frozen_texts = {s["text"] for s in report["held_out_2"]["frozen_statements"]}
        live_texts = {s.text for s in HELD_OUT_2_STATEMENTS}
        assert frozen_texts == live_texts


class TestHeldOut2StructuralCases:
    """`held_out_2`'s structural rows carry ARGUS's raw extracted facts (kind/value/text/replaces)
    rather than a pass/fail boolean — see `BlindStructuralCase`'s own docstring for why."""

    def test_every_case_names_one_of_the_three_stated_gap_shapes(self, report: dict) -> None:
        allowed = {"leading_question_zeroes_extraction", "two_facts_same_kind_one_message",
                  "unstructured_fact_no_kind"}
        for case in report["held_out_2"]["structural_cases"]:
            assert case["gap"] in allowed

    def test_four_cases_per_gap_shape(self, report: dict) -> None:
        gaps = [c["gap"] for c in report["held_out_2"]["structural_cases"]]
        for shape in ("leading_question_zeroes_extraction", "two_facts_same_kind_one_message",
                     "unstructured_fact_no_kind"):
            assert gaps.count(shape) == 4

    def test_every_fact_row_carries_the_full_fact_shape(self, report: dict) -> None:
        for case in report["held_out_2"]["structural_cases"]:
            for fact in case["facts"]:
                assert set(fact.keys()) == {"kind", "subject", "value", "text", "replaces"}

    def test_mem0_facts_recalled_is_between_zero_and_expected(self, report: dict) -> None:
        for case in report["held_out_2"]["structural_cases"]:
            assert 0 <= case["mem0_facts_recalled"] <= case["mem0_facts_expected"]


class TestHeldOut2Budget:
    """The coordinator's remaining-budget instruction: this round alone must stay within 275 Qwen
    calls, and the cumulative total across every round in this artefact must stay within 400."""

    def test_this_rounds_calls_stay_within_275(self, report: dict) -> None:
        assert report["held_out_2"]["qwen_calls_used_this_round"] <= 275

    def test_this_rounds_calls_never_exceed_what_was_planned(self, report: dict) -> None:
        d = report["held_out_2"]
        assert d["qwen_calls_used_this_round"] <= d["qwen_calls_planned_this_round"]

    def test_cumulative_calls_stay_within_the_400_call_cap(self, report: dict) -> None:
        assert report["qwen_calls_used"] <= 400

    def test_cumulative_calls_include_this_rounds_calls(self, report: dict) -> None:
        # tuning + held_out_1's round spent 125 (fixed history); this round's count must be
        # reflected in the cumulative total, not silently dropped.
        assert report["qwen_calls_used"] >= report["held_out_2"]["qwen_calls_used_this_round"]


class TestHeldOut3FrozenRecord:
    """Same discipline as `TestHeldOut2FrozenRecord`: `held_out_3` must carry proof it was saved
    before EITHER side ran — including before ARGUS's own Qwen-spending `combined()` calls, which
    is new this round."""

    def test_frozen_at_is_present_and_looks_like_a_timestamp(self, report: dict) -> None:
        frozen_at = report["held_out_3"]["frozen_at"]
        assert frozen_at and "T" in frozen_at and frozen_at.count("-") >= 2

    def test_frozen_statement_count_matches_the_live_fixture(self, report: dict) -> None:
        assert len(report["held_out_3"]["frozen_statements"]) == len(HELD_OUT_3_STATEMENTS)

    def test_frozen_not_about_me_count_matches_the_live_fixture(self, report: dict) -> None:
        assert len(report["held_out_3"]["frozen_not_about_me"]) == len(HELD_OUT_3_NOT_ABOUT_ME)

    def test_frozen_task_count_matches_the_live_fixture(self, report: dict) -> None:
        assert len(report["held_out_3"]["frozen_tasks"]) == len(HELD_OUT_3_TASKS)

    def test_frozen_structural_count_matches_the_live_fixture(self, report: dict) -> None:
        assert len(report["held_out_3"]["frozen_structural"]) == len(HELD_OUT_3_STRUCTURAL)

    def test_frozen_statement_texts_match_the_live_fixture_exactly(self, report: dict) -> None:
        frozen_texts = {s["text"] for s in report["held_out_3"]["frozen_statements"]}
        live_texts = {s.text for s in HELD_OUT_3_STATEMENTS}
        assert frozen_texts == live_texts


class TestHeldOut3StructuralCases:
    """`held_out_3`'s structural rows carry ARGUS's raw extracted facts under `argus_facts` (not
    `facts` — held_out_2's key — since this round also records `at`/`price_at`, the full `Fact`
    shape, not just kind/subject/value/text/replaces)."""

    def test_every_case_names_one_of_the_three_stated_gap_shapes(self, report: dict) -> None:
        allowed = {"leading_question_zeroes_extraction", "two_facts_same_kind_one_message",
                  "unstructured_fact_no_kind"}
        for case in report["held_out_3"]["structural_cases"]:
            assert case["gap"] in allowed

    def test_three_cases_per_gap_shape(self, report: dict) -> None:
        gaps = [c["gap"] for c in report["held_out_3"]["structural_cases"]]
        for shape in ("leading_question_zeroes_extraction", "two_facts_same_kind_one_message",
                     "unstructured_fact_no_kind"):
            assert gaps.count(shape) == 3

    def test_every_argus_fact_row_carries_the_full_fact_shape(self, report: dict) -> None:
        for case in report["held_out_3"]["structural_cases"]:
            for fact in case["argus_facts"]:
                assert set(fact.keys()) == {"kind", "subject", "value", "text", "at", "price_at",
                                            "replaces"}

    def test_mem0_facts_recalled_is_between_zero_and_expected(self, report: dict) -> None:
        for case in report["held_out_3"]["structural_cases"]:
            assert 0 <= case["mem0_facts_recalled"] <= case["mem0_facts_expected"]


class TestHeldOut3ArgusSpecificRows:
    """ARGUS's own per-item detail, distinct from the mem0-shaped `statement_rows`/
    `not_about_me_rows` every round shares (see module docstring)."""

    def test_argus_statement_rows_cover_every_fixture_statement(self, report: dict) -> None:
        rows = report["held_out_3"]["argus_statement_rows"]
        assert {r["text"] for r in rows} == {s.text for s in HELD_OUT_3_STATEMENTS}

    def test_argus_recall_recalled_count_matches_the_rows(self, report: dict) -> None:
        rows = report["held_out_3"]["argus_statement_rows"]
        recomputed = sum(1 for r in rows if r["recalled"])
        assert recomputed == report["held_out_3"]["argus_recall"]["recalled"]

    def test_argus_not_about_me_rows_cover_every_fixture_sentence(self, report: dict) -> None:
        rows = report["held_out_3"]["argus_not_about_me_rows"]
        assert {r["text"] for r in rows} == set(HELD_OUT_3_NOT_ABOUT_ME)

    def test_argus_false_positive_count_matches_the_rows(self, report: dict) -> None:
        rows = report["held_out_3"]["argus_not_about_me_rows"]
        recomputed = sum(1 for r in rows if r["false_positive"])
        assert recomputed == report["held_out_3"]["argus_false_positives"]["false_positives"]


class TestHeldOut3Budget:
    """The coordinator's remaining-budget instruction for this round: at most 150 Qwen calls
    across BOTH sides (ARGUS now spends real calls too, via `memory_model.combined()`), and the
    cumulative total across every round in this artefact must still stay within 400."""

    def test_this_rounds_calls_stay_within_150(self, report: dict) -> None:
        assert report["held_out_3"]["qwen_calls_used_this_round"] <= 150

    def test_this_rounds_calls_never_exceed_what_was_planned(self, report: dict) -> None:
        d = report["held_out_3"]
        assert d["qwen_calls_used_this_round"] <= d["qwen_calls_planned_this_round"]

    def test_argus_and_mem0_calls_sum_to_the_rounds_total(self, report: dict) -> None:
        d = report["held_out_3"]
        assert (d["argus_qwen_calls_used_this_round"] + d["mem0_qwen_calls_used_this_round"]
               == d["qwen_calls_used_this_round"])

    def test_cumulative_calls_stay_within_the_400_call_cap(self, report: dict) -> None:
        assert report["qwen_calls_used"] <= 400

    def test_cumulative_calls_include_this_rounds_calls(self, report: dict) -> None:
        assert report["qwen_calls_used"] >= report["held_out_3"]["qwen_calls_used_this_round"]

    def test_mem0_side_spent_exactly_one_call_per_message(self, report: dict) -> None:
        # mem0's cost model has been exactly 1 call/add() every round so far; held_out_3 reuses
        # the same add()-per-message chain shape, so this should still hold.
        d = report["held_out_3"]
        total_messages = (len(HELD_OUT_3_STATEMENTS) + len(HELD_OUT_3_NOT_ABOUT_ME)
                          + len(HELD_OUT_3_STRUCTURAL) + len(HELD_OUT_3_TASKS))
        assert d["mem0_qwen_calls_used_this_round"] == total_messages


class TestHeldOut3Relabelled:
    """`held_out_3`'s OWN rows informed a second `lui/memory_model.py` change (per the
    coordinator), so `freeze_held_out_4()` relabels it in place — same key, updated `note`, no
    rename (unlike `held_out`→`held_out_1`, since `held_out_3` already had its own distinct
    name)."""

    def test_held_out_3_carries_a_note_explaining_the_relabel(self, report: dict) -> None:
        note = report["held_out_3"].get("note", "")
        assert "tuning" in note.lower()

    def test_held_out_3_data_is_unchanged_by_the_relabel(self, report: dict) -> None:
        # Only `note` should differ from what held_out_3's own run produced — its fixture sizes
        # must still match the live HELD_OUT_3_* constants exactly.
        assert report["held_out_3"]["argus_recall"]["total"] == len(HELD_OUT_3_STATEMENTS)


class TestHeldOut4FrozenRecord:
    """Same discipline as `TestHeldOut3FrozenRecord`."""

    def test_frozen_at_is_present_and_looks_like_a_timestamp(self, report: dict) -> None:
        frozen_at = report["held_out_4"]["frozen_at"]
        assert frozen_at and "T" in frozen_at and frozen_at.count("-") >= 2

    def test_frozen_statement_count_matches_the_live_fixture(self, report: dict) -> None:
        assert len(report["held_out_4"]["frozen_statements"]) == len(HELD_OUT_4_STATEMENTS)

    def test_frozen_not_about_me_count_matches_the_live_fixture(self, report: dict) -> None:
        assert len(report["held_out_4"]["frozen_not_about_me"]) == len(HELD_OUT_4_NOT_ABOUT_ME)

    def test_frozen_task_count_matches_the_live_fixture(self, report: dict) -> None:
        assert len(report["held_out_4"]["frozen_tasks"]) == len(HELD_OUT_4_TASKS)

    def test_frozen_structural_count_matches_the_live_fixture(self, report: dict) -> None:
        assert len(report["held_out_4"]["frozen_structural"]) == len(HELD_OUT_4_STRUCTURAL)

    def test_frozen_statement_texts_match_the_live_fixture_exactly(self, report: dict) -> None:
        frozen_texts = {s["text"] for s in report["held_out_4"]["frozen_statements"]}
        live_texts = {s.text for s in HELD_OUT_4_STATEMENTS}
        assert frozen_texts == live_texts


class TestHeldOut4StructuralCases:
    """Same shape as `TestHeldOut3StructuralCases` — ARGUS's raw facts under `argus_facts`."""

    def test_every_case_names_one_of_the_three_stated_gap_shapes(self, report: dict) -> None:
        allowed = {"leading_question_zeroes_extraction", "two_facts_same_kind_one_message",
                  "unstructured_fact_no_kind"}
        for case in report["held_out_4"]["structural_cases"]:
            assert case["gap"] in allowed

    def test_three_cases_per_gap_shape(self, report: dict) -> None:
        gaps = [c["gap"] for c in report["held_out_4"]["structural_cases"]]
        for shape in ("leading_question_zeroes_extraction", "two_facts_same_kind_one_message",
                     "unstructured_fact_no_kind"):
            assert gaps.count(shape) == 3

    def test_every_argus_fact_row_carries_the_full_fact_shape(self, report: dict) -> None:
        for case in report["held_out_4"]["structural_cases"]:
            for fact in case["argus_facts"]:
                assert set(fact.keys()) == {"kind", "subject", "value", "text", "at", "price_at",
                                            "replaces"}

    def test_mem0_facts_recalled_is_between_zero_and_expected(self, report: dict) -> None:
        for case in report["held_out_4"]["structural_cases"]:
            assert 0 <= case["mem0_facts_recalled"] <= case["mem0_facts_expected"]


class TestHeldOut4ArgusSpecificRows:
    def test_argus_statement_rows_cover_every_fixture_statement(self, report: dict) -> None:
        rows = report["held_out_4"]["argus_statement_rows"]
        assert {r["text"] for r in rows} == {s.text for s in HELD_OUT_4_STATEMENTS}

    def test_argus_recall_recalled_count_matches_the_rows(self, report: dict) -> None:
        rows = report["held_out_4"]["argus_statement_rows"]
        recomputed = sum(1 for r in rows if r["recalled"])
        assert recomputed == report["held_out_4"]["argus_recall"]["recalled"]

    def test_argus_not_about_me_rows_cover_every_fixture_sentence(self, report: dict) -> None:
        rows = report["held_out_4"]["argus_not_about_me_rows"]
        assert {r["text"] for r in rows} == set(HELD_OUT_4_NOT_ABOUT_ME)

    def test_argus_false_positive_count_matches_the_rows(self, report: dict) -> None:
        rows = report["held_out_4"]["argus_not_about_me_rows"]
        recomputed = sum(1 for r in rows if r["false_positive"])
        assert recomputed == report["held_out_4"]["argus_false_positives"]["false_positives"]


class TestHeldOut4Budget:
    """The coordinator's remaining-budget instruction for this round: at most 110 Qwen calls
    across BOTH sides, and the cumulative total across every round must still stay within 400."""

    def test_this_rounds_calls_stay_within_110(self, report: dict) -> None:
        assert report["held_out_4"]["qwen_calls_used_this_round"] <= 110

    def test_this_rounds_calls_never_exceed_what_was_planned(self, report: dict) -> None:
        d = report["held_out_4"]
        assert d["qwen_calls_used_this_round"] <= d["qwen_calls_planned_this_round"]

    def test_argus_and_mem0_calls_sum_to_the_rounds_total(self, report: dict) -> None:
        d = report["held_out_4"]
        assert (d["argus_qwen_calls_used_this_round"] + d["mem0_qwen_calls_used_this_round"]
               == d["qwen_calls_used_this_round"])

    def test_cumulative_calls_stay_within_the_400_call_cap(self, report: dict) -> None:
        assert report["qwen_calls_used"] <= 400

    def test_cumulative_calls_include_this_rounds_calls(self, report: dict) -> None:
        assert report["qwen_calls_used"] >= report["held_out_4"]["qwen_calls_used_this_round"]

    def test_mem0_side_spent_exactly_one_call_per_message(self, report: dict) -> None:
        d = report["held_out_4"]
        total_messages = (len(HELD_OUT_4_STATEMENTS) + len(HELD_OUT_4_NOT_ABOUT_ME)
                          + len(HELD_OUT_4_STRUCTURAL) + len(HELD_OUT_4_TASKS))
        assert d["mem0_qwen_calls_used_this_round"] == total_messages


class TestLatencyAndCost:
    def test_argus_spends_zero_qwen_calls_and_tokens_per_turn(self, report: dict) -> None:
        a = report["latency_and_cost"]["argus"]
        assert a["qwen_calls_per_turn"] == 0
        assert a["qwen_tokens_per_turn"] == 0

    def test_mem0_spends_exactly_one_qwen_call_per_add(self, report: dict) -> None:
        assert report["latency_and_cost"]["mem0"]["qwen_calls_per_turn"] == 1.0

    def test_mem0_add_latency_stats_are_internally_ordered(self, report: dict) -> None:
        stats = report["latency_and_cost"]["mem0"]["add_latency_s"]
        assert stats["min"] <= stats["median"] <= stats["max"]
        assert stats["min"] <= stats["mean"] <= stats["max"]

    def test_mem0_search_latency_stats_are_internally_ordered(self, report: dict) -> None:
        stats = report["latency_and_cost"]["mem0"]["search_latency_s"]
        assert stats["min"] <= stats["median"] <= stats["max"]
        assert stats["min"] <= stats["mean"] <= stats["max"]

    def test_argus_mean_latency_is_negligible_next_to_mem0s(self, report: dict) -> None:
        # Pure regex and dict lookups versus a real network round trip to an LLM: the claim is
        # "orders of magnitude", not a specific ratio, so the bound is loose (100x) to avoid
        # flaking on an unusually fast mem0 response while still catching a real regression (e.g.
        # ARGUS's `extract()` accidentally gaining a network call).
        argus_s = report["latency_and_cost"]["argus"]["mean_elapsed_ms"] / 1000
        mem0_s = report["latency_and_cost"]["mem0"]["add_latency_s"]["mean"]
        if mem0_s > 0:
            assert argus_s < mem0_s / 100

    def test_mem0_search_and_get_all_calls_are_faster_than_its_own_add_calls(
        self, report: dict,
    ) -> None:
        # search()/get_all() make no LLM call (verified by reading mem0/memory/main.py, see this
        # module's own docstring); add() makes exactly one real HTTP round trip. If search ever
        # became as slow as add, that would mean it started doing real network work too.
        mem0 = report["latency_and_cost"]["mem0"]
        if mem0["search_or_get_all_calls"] and mem0["add_calls"]:
            assert mem0["search_latency_s"]["mean"] < mem0["add_latency_s"]["mean"]
