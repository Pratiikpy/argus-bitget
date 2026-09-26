"""The LUI rematch's frozen inputs and the pre-fix snapshot of ARGUS's router.

The inputs and the snapshot are committed artefacts that published code cites
(`lui/normalise.py`, `lui/question.py`), so each is checked against its own digest and the figures
quoted from them are recomputed here rather than trusted.
"""

# Full-width characters are the subject of these tests, not a typo.
# ruff: noqa: RUF001

from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path
from typing import Any

import pytest

from argus.eval import lui_rematch, lui_rematch_inputs
from argus.eval import lui_rematch as lr
from argus.eval.lui_rematch_inputs import (
    add_typos,
    clinc_probes,
    digest,
    hard_negatives,
    load,
    massive_probes,
    perturbations,
    strip_punctuation,
    to_fullwidth,
)

RUNNER = Path(__file__).resolve().parents[1] / "scripts" / "rasa_diet_runner.py"


@pytest.fixture(scope="module")
def inputs() -> dict[str, Any]:
    return lui_rematch_inputs.load()


@pytest.fixture(scope="module")
def snapshot() -> dict[str, Any]:
    return lui_rematch.load_snapshot()


def _cascade_correct(rows: list[dict[str, str]], predictions: dict[str, Any]) -> int:
    return sum(
        1 for r in rows
        if predictions[r["ask"]]["cascade"]
        == (None if r["expect"] == "out_of_scope" else r["expect"])
    )


class TestPerturbations:
    def test_typos_swap_adjacent_characters_and_reproduce_from_the_seed(self) -> None:
        once = add_typos("abcdef", 1, random.Random(7))
        assert once == add_typos("abcdef", 1, random.Random(7))
        assert sorted(once) == sorted("abcdef")
        assert sum(a != b for a, b in zip(once, "abcdef", strict=True)) in (0, 2)

    def test_typos_leave_a_single_character_alone(self) -> None:
        assert add_typos("a", 3, random.Random(1)) == "a"

    def test_fullwidth_maps_ascii_and_keeps_spaces(self) -> None:
        assert to_fullwidth("NVDA 30%") == "ＮＶＤＡ ３０％"

    def test_punctuation_is_stripped_in_both_scripts(self) -> None:
        assert strip_punctuation("why, NVDA? 为什么？") == "why NVDA 为什么"

    def test_an_unchanged_string_is_not_counted(self) -> None:
        out = perturbations([{"ask": "123", "expect": "session", "lang": "en"}],
                            to_traditional=None)
        assert out["upper"] == [] and out["no_punctuation"] == []
        assert [r["ask"] for r in out["fullwidth"]] == ["１２３"]

    def test_hard_negatives_cover_every_template_and_name(self) -> None:
        rows = hard_negatives()
        assert len(rows) == 120
        assert len({r["ask"] for r in rows}) == 120


class TestExternalProbes:
    def test_massive_keeps_ids_parallel_and_drops_finance(self) -> None:
        stock = lui_rematch_inputs.MASSIVE_INTENTS.index("qa_stock")
        joke = lui_rematch_inputs.MASSIVE_INTENTS.index("general_joke")
        en = [{"id": "1", "intent": joke, "utt": "tell me a joke"},
              {"id": "2", "intent": stock, "utt": "price of apple"},
              {"id": "3", "intent": joke, "utt": "a joke about my money"}]
        zh = [{"id": "1", "intent": joke, "utt": "讲个笑话"},
              {"id": "2", "intent": stock, "utt": "苹果股价"},
              {"id": "3", "intent": joke, "utt": "讲个笑话"}]
        tw = [{"id": "1", "intent": joke, "utt": "講個笑話"}]
        probes, excluded = massive_probes(en, zh, frozenset(), tw)
        assert [(p["id"], p["lang"]) for p in probes] == [("1", "en"), ("1", "zh"),
                                                          ("1", "zh-Hant")]
        assert excluded["finance_intent"] == 1 and excluded["finance_lexicon"] == 1

    def test_clinc_drops_finance_training_rows_and_duplicates(self) -> None:
        blob = {"oos_test": [["play some jazz", "oos"], ["PLAY SOME JAZZ", "oos"],
                             ["what is my bank balance", "oos"], ["seen before", "oos"]]}
        probes, excluded = clinc_probes(blob, frozenset({"seen before"}))
        assert [p["ask"] for p in probes] == ["play some jazz"]
        assert excluded == {"finance_lexicon": 1, "seen_in_training": 1, "duplicate": 1}


class TestCommittedInputs:
    def test_every_section_matches_its_digest(self, inputs: dict[str, Any]) -> None:
        for key, value in inputs["digests"].items():
            assert digest(inputs[key]) == value, key

    def test_sizes(self, inputs: dict[str, Any]) -> None:
        assert len(inputs["training"]) == 314
        assert len(inputs["sealed"]) == 293
        assert len(inputs["massive_oos"]) == 672
        assert len(inputs["clinc_oos"]) == 963
        assert len(inputs["confirmation_base"]) == 578

    def test_no_probe_was_a_training_row(self, inputs: dict[str, Any]) -> None:
        known = {r["text"].casefold() for r in inputs["training"]}
        for key in ("massive_oos", "clinc_oos", "hard_negatives"):
            assert not [r for r in inputs[key] if r["ask"].casefold() in known], key

    def test_both_sides_score_the_same_strings(self, inputs: dict[str, Any]) -> None:
        spec = importlib.util.spec_from_file_location("rasa_diet_runner", RUNNER)
        assert spec is not None and spec.loader is not None
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        assert lui_rematch.all_texts(inputs) == runner._all_texts(inputs)


class TestPreFixSnapshot:
    def test_it_matches_its_digest_and_the_inputs(
        self, inputs: dict[str, Any], snapshot: dict[str, Any],
    ) -> None:
        assert snapshot["inputs_digests"] == inputs["digests"]
        assert set(snapshot["predictions"]) == set(lui_rematch.all_texts(inputs))

    def test_a_second_snapshot_is_refused(self, tmp_path: Path) -> None:
        taken = tmp_path / "snapshot.json"
        taken.write_text("{}", encoding="utf-8")
        with pytest.raises(lui_rematch.RematchError):
            lui_rematch.snapshot_before_fixes(taken)

    def test_the_figures_published_code_quotes(
        self, inputs: dict[str, Any], snapshot: dict[str, Any],
    ) -> None:
        predictions = snapshot["predictions"]
        pert = inputs["perturbations"]
        assert _cascade_correct(inputs["sealed"], predictions) == 233
        assert (_cascade_correct(pert["fullwidth"], predictions), len(pert["fullwidth"])) == (
            15, 122)
        assert (_cascade_correct(pert["upper"], predictions), len(pert["upper"])) == (1, 105)


class TestTheFixesHold:
    def test_full_width_and_capitals_now_route(self, inputs: dict[str, Any]) -> None:
        pert = inputs["perturbations"]
        rows = pert["fullwidth"] + pert["upper"]
        live, _latencies = lui_rematch.predict_all([r["ask"] for r in rows])
        assert _cascade_correct(pert["fullwidth"], live) >= 80
        assert _cascade_correct(pert["upper"], live) >= 70


class TestTheReport:
    """The committed report against the vendored Rasa runs: verified, then pinned."""

    report = json.loads(lr.REPORT_PATH.read_text(encoding="utf-8"))

    def test_every_rasa_run_is_one_the_frozen_inputs_produced(self) -> None:
        vendored = json.loads(lr.RASA_PATH.read_text(encoding="utf-8"))
        texts = lr.all_texts(load())
        assert vendored["texts_digest"] == digest(texts)
        assert {r["config"] for r in vendored["runs"]} == {"rasa_default", "legacy_20260921"}
        assert sum(not r["repeat"] for r in vendored["runs"]) == 9
        assert all(len(r["labels"]) == len(texts) for r in vendored["runs"])

    def test_held_out_accuracy_is_a_tie_on_the_paired_interval(self) -> None:
        held = self.report["paired_cascade_vs_rasa_default"]["held_out"]
        assert held["verdict"] == "tie"
        low, high = held["ci95"]
        assert low < 0 < high

    def test_out_of_scope_declines_favour_argus_on_massive(self) -> None:
        massive = self.report["paired_cascade_vs_rasa_default"]["massive_oos"]
        assert massive["verdict"] == "argus_better"

    def test_the_losses_are_recorded_as_losses(self) -> None:
        paired = self.report["paired_cascade_vs_rasa_default"]
        assert paired["confirmation:no_punctuation"]["verdict"] == "rasa_better"
        assert paired["confirmation:typo2"]["verdict"] == "rasa_better"

    def test_the_repeat_is_not_claimed_as_a_cold_retrain(self) -> None:
        assert "cache" in self.report["reproducibility"]["rasa_repeat_note"]
