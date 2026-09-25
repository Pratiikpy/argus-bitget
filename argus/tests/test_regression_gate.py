"""The regression gate — SWE-bench's FAIL_TO_PASS / PASS_TO_PASS, applied to checklist rules.

`eval/regression_gate.py` admits a proposed rule only if it fires on the defective decision it was
written from, stays silent on the clean one it was contrasted with, fires on no previously-correct
decision in the fit window, and fires often enough to be a rule rather than an anecdote. These
tests pin every one of those refusals by name, the chronological split and its embargo, and the
end-to-end replay on a record with a planted cause — where the deterministic proposer must find the
cause, the gate must admit it, and the held-out window must confirm it.

The planted records are synthetic and labelled so: they test that the machinery can find a cause
that is there. What it finds on ARGUS's real record is `data/rule_proposals.json`, not this file.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from argus.desk.review import Defect, DefectKind, spec_from
from argus.desk.rule_proposer import (
    JOURNAL_FEATURES,
    ContrastProposer,
    FaultAuthor,
    Proposal,
    journal_records,
    pair_contrasts,
    parse_journal,
)
from argus.eval.regression_gate import (
    MAX_REGRESSIONS,
    REASON_CODES,
    RegressionGate,
    Resolution,
    Window,
    compute_fail_to_pass,
    compute_pass_to_pass,
    run,
    split_windows,
)

START = datetime(2026, 9, 12, tzinfo=UTC)


def _record(seq: int, *, sources: int, analysts: int, independence: float,
            symbol: str) -> dict[str, Any]:
    return {
        "seq": seq, "symbol": symbol, "at": (START + timedelta(hours=seq)).isoformat(),
        "notes": [f"panel: {analysts} analysts, {sources} distinct sources, independence "
                  f"{independence:.2f} -> bullish at 0.55 after provenance discount"],
        "flags": [],
    }


def planted(n: int = 200, seed: int = 7) -> tuple[list[dict[str, Any]], list[Defect]]:
    """A record where a grounding defect occurs exactly when sources < 6 AND analysts == 3."""
    rng = random.Random(seed)
    records: list[dict[str, Any]] = []
    defects: list[Defect] = []
    for seq in range(1, n + 1):
        sources = rng.randint(2, 20)
        analysts = rng.choice([2, 3, 4])
        rec = _record(seq, sources=sources, analysts=analysts,
                      independence=round(rng.uniform(1, 5), 2),
                      symbol=rng.choice(["NVDAUSDT", "TSLAUSDT", "AAPLUSDT"]))
        records.append(rec)
        if sources < 6 and analysts == 3:
            defects.append(Defect(seq, rec["symbol"], DefectKind.GROUNDING, "planted"))
    return records, defects


def _window(records: list[dict[str, Any]], defects: list[Defect]) -> Window:
    return Window(records=tuple(records), defects=tuple(defects))


def _proposal(conditions: list[dict[str, Any]], bad: dict[str, Any], good: dict[str, Any],
              kind: DefectKind = DefectKind.GROUNDING) -> Proposal:
    from argus.desk.rule_proposer import ContrastPair

    spec = spec_from({"name": "candidate-rule", "prompt": "Check the figures in the thesis.",
                      "rationale": "a fixed rationale for the test", "targets": [str(kind)],
                      "conditions": conditions})
    pair = ContrastPair(kind, bad, good, 0.1, ("sources",), "planted")
    return Proposal("test", pair, spec, FaultAuthor.DESK, "planted")


class TestSweBenchArithmetic:
    def test_ported_metrics_match_grading_py(self) -> None:
        report = {"FAIL_TO_PASS": {"success": [1], "failure": []},
                  "PASS_TO_PASS": {"success": [2, 3], "failure": [4]}}
        assert compute_fail_to_pass(report) == 1.0
        assert compute_pass_to_pass(report) == pytest.approx(2 / 3)

    def test_an_empty_pass_to_pass_set_is_not_scored_as_a_pass(self) -> None:
        # SWE-bench returns 1 here (grading.py:306-308, with its own TODO). Nothing is not evidence.
        report: dict[str, dict[str, list[int]]] = {
            "FAIL_TO_PASS": {"success": [1], "failure": []},
            "PASS_TO_PASS": {"success": [], "failure": []}}
        assert compute_pass_to_pass(report) is None


_RECORDS, _DEFECTS = planted()
_MARKED = {d.seq for d in _DEFECTS}


class TestGate:
    records, defects = _RECORDS, _DEFECTS
    bad = next(r for r in _RECORDS if r["seq"] in _MARKED)
    good = next(r for r in _RECORDS if r["seq"] not in _MARKED)

    def gate(self) -> RegressionGate:
        return RegressionGate(window=_window(self.records, self.defects))

    def test_the_true_cause_is_admitted_with_a_full_resolution(self) -> None:
        g = self.gate()
        result = g.judge(_proposal([{"feature": "sources", "op": "<", "value": 6},
                                    {"feature": "analysts", "op": "==", "value": 3}],
                                   self.bad, self.good))
        assert result.admitted and result.resolution is Resolution.FULL
        assert result.regressions == 0 and result.codes == ("admitted",)
        assert g.admitted and result.new_catches == len(self.defects)

    def test_half_the_cause_breaks_previously_correct_decisions_and_is_refused(self) -> None:
        result = self.gate().judge(_proposal([{"feature": "sources", "op": "<", "value": 6}],
                                             self.bad, self.good))
        assert not result.admitted and "regression" in result.codes
        assert result.regressions > MAX_REGRESSIONS
        assert result.resolution is Resolution.NO

    def test_a_rule_silent_on_its_own_target_is_refused(self) -> None:
        result = self.gate().judge(_proposal([{"feature": "sources", "op": ">", "value": 30}],
                                             self.bad, self.good))
        assert "target_not_fixed" in result.codes and not result.admitted

    def test_a_rule_that_fires_on_its_own_contrast_is_refused(self) -> None:
        result = self.gate().judge(_proposal([{"feature": "sources", "op": ">", "value": 0}],
                                             self.bad, self.good))
        assert "fires_on_contrast" in result.codes

    def test_a_rule_that_describes_only_its_target_is_an_anecdote(self) -> None:
        sources = float(self.bad["notes"][0].split(", ")[1].split()[0])
        independence = float(self.bad["notes"][0].split("independence ")[1].split()[0])
        result = self.gate().judge(_proposal(
            [{"feature": "sources", "op": "==", "value": sources},
             {"feature": "independence", "op": "==", "value": independence},
             {"feature": "analysts", "op": "==", "value": 3}], self.bad, self.good))
        assert "below_support" in result.codes and not result.admitted

    def test_a_second_rule_catching_nothing_new_is_redundant_and_its_pair_is_skipped(
        self,
    ) -> None:
        g = self.gate()
        cause = [{"feature": "sources", "op": "<", "value": 6},
                 {"feature": "analysts", "op": "==", "value": 3}]
        g.judge(_proposal(cause, self.bad, self.good))
        again = g.judge(_proposal([{"feature": "sources", "op": "<", "value": 5},
                                   {"feature": "analysts", "op": "==", "value": 3}],
                                  self.bad, self.good))
        assert "redundant" in again.codes
        pairs = pair_contrasts(self.records, self.defects, DefectKind.GROUNDING)
        assert all(g.already_fixed(p) for p in pairs)

    def test_a_refusal_from_the_proposer_is_recorded_not_admitted(self) -> None:
        from argus.desk.rule_proposer import ContrastPair

        pair = ContrastPair(DefectKind.GROUNDING, self.bad, self.good, 0.1, (), "x")
        result = self.gate().judge(Proposal("m", pair, None, FaultAuthor.DESK, "",
                                            refused="model declined"))
        assert result.codes == ("no_rule",) and not result.admitted

    def test_every_code_the_gate_emits_is_documented(self) -> None:
        g = self.gate()
        for conditions in ([{"feature": "sources", "op": ">", "value": 30}],
                           [{"feature": "sources", "op": "<", "value": 6}]):
            for code in g.judge(_proposal(conditions, self.bad, self.good)).codes:
                assert code in REASON_CODES


class TestSplit:
    def test_the_split_is_chronological_and_the_held_out_side_is_embargoed(self) -> None:
        records, defects = planted(40)
        graded = {r["seq"] for r in records}
        # Every fit-window lean settles 5 hours after it was taken: the last fit decision (hour
        # 20) settles at hour 25, so held-out decisions at hours 21-24 are embargoed.
        settled = {r["seq"]: START + timedelta(hours=r["seq"] + 5) for r in records}
        fit, held, embargoed = split_windows(records, defects, lean_graded=graded,
                                             settled_at=settled)
        assert max(int(r["seq"]) for r in fit.records) < min(int(r["seq"]) for r in held.records)
        assert embargoed == 4
        assert all(d.seq in {r["seq"] for r in fit.records} for d in fit.defects)

    def test_without_settlement_times_nothing_is_embargoed(self) -> None:
        records, defects = planted(40)
        _, held, embargoed = split_windows(records, defects, lean_graded=None)
        assert embargoed == 0 and len(held.records) == 20


class ScriptedModel:
    """Answers every pair with the planted cause; counts every call it receives."""

    def __init__(self) -> None:
        self.calls = 0

    def complete_json(self, messages: list[dict[str, Any]], *,
                      required_keys: tuple[str, ...] = (),
                      validate: Callable[[dict[str, Any]], str] | None = None,
                      max_tokens: int = 0, attempts: int = 3) -> dict[str, Any]:
        self.calls += 1
        answer = {"discussion": "fewer sources with a full panel", "author": "desk",
                  "description": "figures from nowhere",
                  "rule": {"name": "narrow-full-panel", "prompt": "Trace each figure.",
                           "rationale": "a full panel on thin evidence fills gaps",
                           "conditions": [{"feature": "sources", "op": "<", "value": 6},
                                          {"feature": "analysts", "op": "==", "value": 3}]}}
        assert validate is None or validate(answer) == ""
        return answer


class TestReplay:
    def test_the_planted_cause_is_found_admitted_and_confirmed_out_of_sample(self) -> None:
        records, defects = planted(300)
        report = run(records=records, defects=defects, lean_graded=None,
                     kinds=(DefectKind.GROUNDING,), hand_written=())
        induced = report["proposers"]["induction"]
        assert induced["admitted"] >= 1
        rule = next(r for r in induced["results"] if r["admitted"])["rule"]
        assert {c["feature"] for c in rule["conditions"]} == {"sources", "analysts"}
        score = report["checklist_comparison"][0]["induction"]
        assert score["precision"] == 1.0 and score["recall"] == 1.0
        assert report["held_out"]["induction"][0]["status"] == "active"

    def test_a_pair_only_proposer_is_reported_even_when_it_cannot_find_a_conjunction(
        self,
    ) -> None:
        # One pair shows one side of a two-feature cause; the gate refuses what that produces,
        # and the report carries the refusals rather than dropping the proposer.
        records, defects = planted(300)
        report = run(records=records, defects=defects, lean_graded=None,
                     kinds=(DefectKind.GROUNDING,), hand_written=())
        contrast = report["proposers"]["contrast"]
        assert contrast["pairs_tried"] > 0
        assert contrast["admitted"] + contrast["rejected"] == contrast["pairs_tried"]

    def test_the_model_cap_is_kept_and_the_calls_are_reported(self) -> None:
        records, defects = planted(300)
        model = ScriptedModel()
        report = run(records=records, defects=defects, lean_graded=None, model=model,
                     max_calls=4, kinds=(DefectKind.GROUNDING,), hand_written=())
        assert model.calls <= 4
        assert report["proposers"]["model"]["model_calls"] == model.calls
        # The first model rule is the planted cause, so every later pair is already fixed.
        assert report["proposers"]["model"]["admitted"] == 1
        assert report["proposers"]["model"]["pairs_skipped_already_fixed"] > 0
        assert {"contrast_all_pairs", "induction_all_pairs"} <= set(report["proposers"])

    def test_a_record_with_no_defects_of_a_kind_writes_no_rules_for_it(self) -> None:
        records, _ = planted(100)
        report = run(records=records, defects=[], lean_graded=None,
                     kinds=(DefectKind.GROUNDING,), hand_written=())
        assert report["pairs_available"] == {"grounding": 0}
        assert report["proposers"]["contrast"]["pairs_tried"] == 0


class TestTraderJournal:
    def test_revenge_trading_is_found_in_a_traders_own_journal(self) -> None:
        """A synthetic journal where every trade opened straight after a loss also loses."""
        import json

        rng = random.Random(3)
        lines: list[str] = []
        at = datetime(2026, 6, 1, 13, tzinfo=UTC)
        last_lost = False
        for _ in range(240):
            at += timedelta(hours=rng.choice([3, 4, 5]))
            lost = last_lost or rng.random() < 0.3
            outcome = -rng.uniform(20, 90) if lost else rng.uniform(20, 90)
            lines.append(json.dumps({
                "at": at.isoformat(), "symbol": rng.choice(["NVDA", "TSLA"]), "side": "long",
                "confidence": round(rng.uniform(0.4, 0.9), 2), "holding_hours": 2.0,
                "setup": rng.choice(["breakout", "pullback"]), "outcome_bps": outcome,
            }))
            last_lost = lost and not last_lost  # a revenge trade's own loss does not chain
        records, defects, graded = journal_records(parse_journal(lines))
        report = run(records=records, defects=defects, lean_graded=graded,
                     vocabulary=JOURNAL_FEATURES, kinds=(DefectKind.OUTCOME,), hand_written=(),
                     journal=True)
        admitted = [r for r in report["proposers"]["induction"]["results"] if r["admitted"]]
        assert admitted, report["proposers"]["induction"]["rejection_reasons"]
        assert any("after_a_loss" in r["rule"]["fires_when"] for r in admitted)
        assert all(r["taxonomy"]["author"] == "trader" for r in admitted)

    def test_the_contrast_proposer_uses_the_journal_vocabulary(self) -> None:
        records, defects, graded = journal_records(parse_journal([
            json_line(i, outcome=-50.0 if i % 2 else 50.0, confidence=0.9 if i % 2 else 0.5)
            for i in range(1, 12)
        ]))
        pairs = pair_contrasts(records, defects, DefectKind.OUTCOME, graded=graded,
                               vocabulary=JOURNAL_FEATURES)
        proposal = ContrastProposer(records=records, vocabulary=JOURNAL_FEATURES,
                                    journal=True).propose(pairs[0])
        assert proposal.spec is not None and proposal.author is FaultAuthor.TRADER
        assert {c.feature for c in proposal.spec.conditions} <= set(JOURNAL_FEATURES)


def json_line(i: int, *, outcome: float, confidence: float) -> str:
    import json

    return json.dumps({"at": (START + timedelta(hours=5 * i)).isoformat(), "symbol": "NVDA",
                       "side": "long", "confidence": confidence, "holding_hours": 1.0,
                       "outcome_bps": outcome})


class TestJournalEntryPoint:
    def test_a_short_journal_is_refused_with_the_number_it_needs(self) -> None:
        from argus.eval.regression_gate import MIN_JOURNAL_TRADES, journal_review

        lines = [json_line(i, outcome=-50.0, confidence=0.5) for i in range(1, 10)]
        out = journal_review(lines)
        assert "refused" in out and str(MIN_JOURNAL_TRADES) in out["refused"]

    def test_a_malformed_journal_is_refused_not_raised(self) -> None:
        from argus.eval.regression_gate import journal_review

        assert "line 1" in journal_review(["{not json"])["refused"]

    def test_the_admitted_checklist_lists_best_evidenced_rules_first(self) -> None:
        from argus.eval.regression_gate import admitted_checklist

        records, defects = planted(300)
        report = run(records=records, defects=defects, lean_graded=None,
                     kinds=(DefectKind.GROUNDING,), hand_written=())
        rows = admitted_checklist(report)
        assert rows and rows[0]["status"] == "active" and rows[0]["proposer"] == "induction"

