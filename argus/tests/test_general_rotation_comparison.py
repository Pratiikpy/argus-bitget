"""Tests for `eval/general_rotation_comparison.py` — the data-honest rotation against pandera,
pandera + pydantic and Great Expectations on the same frozen real Bitget corpus.

Offline by design: the corpus (`data/general_rotation_corpus.json`) and the rivals' recorded
outcomes (`data/general_rotation_rival_results.json`, produced by
`scripts/general_rotation_rival.py` in an interpreter that has the rival libraries) are committed
files, so everything here runs from a clean checkout. What is pinned: the ablation baseline's
bytes, that every contestant scored the byte-identical case, the scoring rules on hand-built
results, the headline scoreboard recomputed from the recorded rivals, and that the committed
artefact is what a fresh run produces.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pytest

from argus.eval import general_rotation_comparison as g
from argus.eval.baselines import argus_rotation_pre_contract

PRE_CONTRACT_SHA256 = "c22e86ab614ae4b417611b7442b3375bb861ebe37dc012d8866c63bfcb3e8fba"
MARKER = (
    b"# ============================== PRE-CONTRACT SOURCE FROM HERE "
    b"==============================\n"
)


@pytest.fixture(scope="module")
def cases() -> list[g.Case]:
    return g.build_cases(g.load_corpus())


@pytest.fixture(scope="module")
def report() -> dict[str, Any]:
    return g.run(cost_repeats=5)


def _accepted(weights: dict[str, float], scores: dict[str, float] | None = None) -> dict[str, Any]:
    return {"outcome": "accepted", "weights": weights, "scores": scores, "message": None}


class TestAblationBaselinePinned:
    def test_the_pre_contract_source_is_byte_identical_to_the_pinned_hash(self) -> None:
        raw = Path(argus_rotation_pre_contract.__file__).read_bytes()
        assert raw.count(MARKER) == 1
        body = raw.split(MARKER, 1)[1]
        assert hashlib.sha256(body).hexdigest() == PRE_CONTRACT_SHA256

    def test_the_pre_contract_rule_silently_accepts_a_negative_top_k(self) -> None:
        scores = {"A": 1.0, "B": 0.5, "S": 0.2}
        weights = argus_rotation_pre_contract.breadth_allocation(scores, ["A", "B"], ["S"],
                                                                 top_k=-1)
        assert weights == {"A": 0.0, "B": 0.0, "S": 0.0}  # a book that allocates nothing


class TestCorpusAndCases:
    def test_the_clean_universe_has_enough_real_history(self, cases: list[g.Case]) -> None:
        from argus.desk import rotation

        corpus = g.load_corpus()
        for symbol in (*g.RISK, *g.SAFE):
            months = rotation.monthly_closes_from_bars(corpus[symbol])
            assert len(months) >= rotation.MIN_MONTHLY_OBSERVATIONS, symbol

    def test_the_cli_default_short_history_is_real(self) -> None:
        from argus.desk import rotation

        corpus = g.load_corpus()
        counts = {s: len(rotation.monthly_closes_from_bars(corpus[s]))
                  for s in ("QQQUSDT", "XAUUSDT", "XAGUSDT")}
        assert counts == {"QQQUSDT": 12, "XAUUSDT": 10, "XAGUSDT": 9}

    def test_case_mix_and_determinism(self, cases: list[g.Case]) -> None:
        kinds = [c.kind for c in cases]
        assert (kinds.count("must_refuse"), kinds.count("tolerable"),
                kinds.count("control")) == (25, 3, 8)
        assert len({c.case_id for c in cases}) == len(cases)
        again = g.build_cases(g.load_corpus())
        assert [g.case_digest(c) for c in again] == [g.case_digest(c) for c in cases]

    def test_every_must_refuse_case_names_what_offends_or_is_a_parameter_fault(
        self, cases: list[g.Case]
    ) -> None:
        for case in cases:
            if case.kind == "must_refuse" and not case.offending:
                assert case.clause == "parameters", case.case_id

    def test_the_wire_survives_non_finite_values(self, cases: list[g.Case]) -> None:
        nan_case = next(c for c in cases if c.case_id == "nan_only_safe")
        wire = json.loads(json.dumps(nan_case.wire(), allow_nan=False))
        assert wire["scores"]["PAXGUSDT"] == "nan"


class TestScoringRules:
    def test_a_must_refuse_case_that_returns_a_book_is_a_silent_accept(
        self, cases: list[g.Case]
    ) -> None:
        case = next(c for c in cases if c.case_id == "gap_on_anchor")
        assert g.classify(case, _accepted({"X": 1.0}), None, None) == "silent_accept"
        refused = {"outcome": "refused", "weights": None, "scores": None, "message": "ETHUSDT"}
        assert g.classify(case, refused, None, None) == "refused"

    def test_a_tolerable_case_must_reproduce_the_contestants_own_clean_book(
        self, cases: list[g.Case]
    ) -> None:
        case = next(c for c in cases if c.case_id == "unsorted_bars")
        clean = _accepted({"A": 0.5, "B": 0.5}, {"A": 1.0, "B": 2.0})
        assert g.classify(case, _accepted({"A": 0.5, "B": 0.5}, {"A": 1.0, "B": 2.0}),
                          clean, None) == "correct"
        assert g.classify(case, _accepted({"A": 0.5, "B": 0.5}, {"A": 1.0, "B": 2.5}),
                          clean, None) == "silent_corruption"
        refused = {"outcome": "refused", "weights": None, "scores": None, "message": ""}
        assert g.classify(case, refused, clean, None) == "over_refusal"

    def test_a_control_book_must_hold_the_invariants(self, cases: list[g.Case]) -> None:
        case = next(c for c in cases if c.case_id == "scores_clean")
        universe = case.universe
        short = dict.fromkeys(universe, 0.0)
        short[universe[0]] = 0.75  # sums to 0.75: a quarter of the book nowhere
        assert g.classify(case, _accepted(short), None, None) == "wrong"
        full = dict.fromkeys(universe, 0.0)
        full[universe[0]] = 1.0
        assert g.classify(case, _accepted(full), None, None) == "correct"
        other = dict.fromkeys(universe, 0.0)
        other[universe[1]] = 1.0
        assert g.classify(case, _accepted(full), None, _accepted(other)) == (
            "disagrees_with_reference")


class TestSameInput:
    def test_every_rival_scored_the_byte_identical_case(self, cases: list[g.Case]) -> None:
        same = g.verify_same_input(cases, g.load_rival_results())
        assert same == {"cases": len(cases), "mismatched": [], "identical": True}

    def test_a_changed_case_is_caught(self, cases: list[g.Case]) -> None:
        rival = g.load_rival_results()
        digests = dict(rival["digests"])
        digests["clean"] = "0" * 64
        same = g.verify_same_input(cases, {**rival, "digests": digests})
        assert same["mismatched"] == ["clean"]


class TestHeadline:
    def test_argus_after_handles_every_case_and_names_every_offender(
        self, report: dict[str, Any]
    ) -> None:
        board = report["scoreboard"]["argus_after"]
        assert board["cases_handled_correctly"] == board["cases_total"] == 36
        assert board["offending_symbols_named"] == board["offending_symbols_total"]

    def test_the_pre_contract_ablation_loses_to_the_general_rival(
        self, report: dict[str, Any]
    ) -> None:
        board = report["scoreboard"]["argus_before"]
        assert board["cases_handled_correctly"] == 20
        assert len(board["silent_accepts"]) == 14
        assert report["verdict"]["pre_contract_argus_vs_best_general_rival"] == "rival_wins"

    def test_the_best_general_rival_ties_rather_than_loses(self, report: dict[str, Any]) -> None:
        verdict = report["verdict"]
        assert verdict["best_general_rival"] == "pandera_pydantic"
        assert verdict["argus_after_vs_best_general_rival"] == "tie"
        assert report["scoreboard"]["pytaa_raw"]["refused"] == 0

    def test_argus_is_cheaper_per_call_than_every_general_rival(
        self, report: dict[str, Any]
    ) -> None:
        costs = report["verdict"]["median_seconds_per_clean_call"]
        for rival in g.GENERAL_RIVALS:
            assert costs["argus_after"] < costs[rival], rival


class TestBusinessMonthDivergence:
    def test_scores_agree_exactly_when_no_anchor_month_ends_on_a_weekend(
        self, report: dict[str, Any]
    ) -> None:
        divergence = report["business_month_divergence"]
        assert divergence["identical_exactly_when_no_anchor_month_ends_on_a_weekend"]
        assert any(p["identical"] for p in divergence["points"])
        assert any(not p["identical"] for p in divergence["points"])
        for point in divergence["points"]:
            assert math.isfinite(point["difference"])


class TestArtefact:
    def test_the_committed_artefact_is_what_a_fresh_run_produces(
        self, report: dict[str, Any]
    ) -> None:
        committed = json.loads(g.ARTEFACT_PATH.read_text(encoding="utf-8"))
        for key in ("cases", "scoreboard", "grid", "same_input", "corpus",
                    "business_month_divergence", "scope_statement"):
            assert json.loads(json.dumps(report[key])) == committed[key], key

    def test_the_artefact_carries_no_machine_path(self) -> None:
        text = g.ARTEFACT_PATH.read_text(encoding="utf-8")
        assert "\\\\" not in text
        assert ":/" not in text.replace("https://", "")
        assert json.loads(text)["corpus"]["path"] == "data/general_rotation_corpus.json"

    def test_render_is_readable(self, report: dict[str, Any]) -> None:
        text = g.render(report)
        assert "best general rival: pandera_pydantic" in text
        assert "same input: True" in text


def test_scope_statement_names_what_is_not_claimed() -> None:
    assert "NOT claimed" in g.SCOPE_STATEMENT
    assert "forecasting value" in g.SCOPE_STATEMENT
