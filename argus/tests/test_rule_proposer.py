"""Rules a machine writes — the representation, the leakage boundary, the pairing and the proposers.

`desk/rule_proposer.py` turns a (defective, clean) pair of decisions into one rule, and
`desk/review.py` now carries the representation that rule has to fit: a conjunction of at most
three comparisons over a fixed vocabulary of decision features. These tests are weighted toward the
ways that goes wrong:

* **leakage** — a feature that reads a checker's verdict would let a rule "predict" a defect by
  restating it. Checked on every recorded decision, not on a fixture: stripping every checker line
  from a real record must leave its features unchanged;
* **untrusted output** — a model's rule is data, validated feature by feature and operator by
  operator before it can become a predicate, and every complaint is specific enough to feed back;
* **counting** — the model proposer reports the calls it actually spent, because the evaluation
  states a cap and must be able to show it kept it.

No test calls a real model. The scripted models below imitate `QwenClient.complete_json`'s retry
loop exactly: a response that fails ``validate`` is retried with the complaint, up to ``attempts``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from argus.desk.review import (
    DECISION_FIELDS,
    DESK_FEATURES,
    STANDING_RULES,
    DefectKind,
    RuleSpec,
    SpecError,
    attach_decisions,
    condition_from,
    defects_from_leans,
    defects_from_notes,
    features_of,
    lean_settled,
    spec_from,
)
from argus.desk.rule_proposer import (
    JOURNAL_FEATURES,
    ContrastPair,
    ContrastProposer,
    FaultAuthor,
    JournalError,
    ModelProposer,
    author_of,
    interleave,
    journal_records,
    pair_contrasts,
    parse_journal,
)

NOTES = Path(__file__).resolve().parents[1] / "data" / "desk_notes.jsonl"

CHECKER_PREFIXES = (
    "[grounding]", "[conflict", "[claim]", "[debate]", "[adversary]", "[constitution]",
    "entity gate", "  ok", "  an instrument", "  unsupported",
)
"""Every line a checker writes after the panel has spoken. A feature must not read any of them."""


def _record(seq: int, *, sources: int = 10, analysts: int = 3, independence: float = 4.0,
            symbol: str = "NVDAUSDT", stance: str = "bullish", concurrent: bool = True,
            at: datetime | None = None, extra: tuple[str, ...] = ()) -> dict[str, Any]:
    when = at or datetime(2026, 9, 12, tzinfo=UTC) + timedelta(hours=seq)
    notes = [
        f"panel: {analysts} analysts, {sources} distinct sources, independence "
        f"{independence:.2f} -> {stance} at 0.55 after provenance discount",
        "[panel] 3 analyst(s) ran concurrently in 20.0s; none could read another's answer, so "
        "agreement is consensus rather than contagion" if concurrent else
        "[conflict] none: 2 analysts agreed on direction and size — but they ran in sequence, so "
        "agreement may be contagion rather than consensus",
        *extra,
    ]
    return {"seq": seq, "symbol": symbol, "at": when.isoformat(), "notes": notes, "flags": []}


# --- the representation -----------------------------------------------------------------------


class TestSpecValidation:
    def _raw(self, **over: Any) -> dict[str, Any]:
        raw: dict[str, Any] = {
            "name": "narrow-and-thin", "prompt": "Ask where each figure came from.",
            "rationale": "fewer sources, more room for an unsupported number",
            "targets": ["grounding"],
            "conditions": [{"feature": "sources", "op": "<", "value": 6}],
        }
        raw.update(over)
        return raw

    def test_a_valid_rule_compiles_and_fires_only_where_its_conditions_hold(self) -> None:
        spec = spec_from(self._raw())
        assert spec.fires(_record(1, sources=4))
        assert not spec.fires(_record(2, sources=9))
        rule = spec.compile()
        assert rule.predicate(_record(3, sources=5)) and rule.targets == {DefectKind.GROUNDING}

    @pytest.mark.parametrize("over, complaint", [
        ({"conditions": [{"feature": "grounding_failures", "op": ">", "value": 0}]},
         "unknown feature"),
        ({"conditions": [{"feature": "concurrent_panel", "op": ">", "value": 1}]}, "flag"),
        ({"conditions": [{"feature": "sources", "op": "<", "value": "six"}]}, "is a number"),
        ({"conditions": [{"feature": "sources", "op": "<", "value": True}]}, "is a number"),
        ({"conditions": [{"feature": "panel_stance", "op": ">", "value": "x"}]}, "label"),
        ({"conditions": []}, "1 to 3"),
        ({"conditions": [{"feature": f, "op": "==", "value": True}
                         for f in ("has_news", "has_macro", "has_social", "has_filing")]},
         "1 to 3"),
        ({"conditions": [{"feature": "sources", "op": "<", "value": 6},
                         {"feature": "sources", "op": ">", "value": 2}]}, "at most one"),
        ({"targets": ["vibes"]}, "targets"),
        ({"targets": []}, "targets"),
        ({"name": "Has Spaces"}, "slug"),
        ({"prompt": "short"}, "prompt"),
    ])
    def test_malformed_rules_are_refused_with_a_message_that_can_be_fed_back(
        self, over: dict[str, Any], complaint: str,
    ) -> None:
        with pytest.raises(SpecError, match=complaint):
            spec_from(self._raw(**over))

    def test_targets_outside_the_allowed_kinds_are_refused(self) -> None:
        with pytest.raises(SpecError, match="within"):
            spec_from(self._raw(), allowed_targets=frozenset({DefectKind.LEAN}))

    def test_a_feature_the_record_cannot_answer_never_satisfies_a_comparison(self) -> None:
        # A missing measurement must not look like a passing one — in either direction.
        bare = {"seq": 1, "symbol": "X", "notes": []}
        for op in ("<", ">", "==", "!="):
            assert not condition_from({"feature": "sources", "op": op, "value": 5}).holds(
                features_of(bare))

    def test_labels_compare_case_insensitively_and_in_takes_a_list(self) -> None:
        cond = condition_from({"feature": "panel_stance", "op": "in", "value": ["Bullish", "x"]})
        assert cond.holds(features_of(_record(1, stance="bullish")))
        assert not cond.holds(features_of(_record(1, stance="bearish")))

    def test_a_flag_is_never_compared_as_a_number(self) -> None:
        cond = condition_from({"feature": "sources", "op": ">", "value": 0})
        assert not cond.holds({"sources": True})


class TestLeakageBoundary:
    def test_no_feature_reads_a_checker_line_on_any_recorded_decision(self) -> None:
        if not NOTES.exists():
            pytest.skip("no recorded desk notes on this machine")
        rows = [json.loads(x) for x in NOTES.read_text(encoding="utf-8").splitlines() if x.strip()]
        assert len(rows) > 100
        changed = []
        for row in rows:
            stripped = {
                **row, "flags": [],
                "notes": [n for n in row["notes"]
                          if not str(n).lower().startswith(CHECKER_PREFIXES)],
            }
            if features_of(row) != features_of(stripped):
                changed.append(row["seq"])
        assert changed == []

    def test_every_recorded_decision_answers_the_core_panel_features(self) -> None:
        if not NOTES.exists():
            pytest.skip("no recorded desk notes on this machine")
        rows = [json.loads(x) for x in NOTES.read_text(encoding="utf-8").splitlines() if x.strip()]
        for row in rows:
            got = features_of(row)
            assert {"analysts", "sources", "independence", "panel_stance"} <= set(got), row["seq"]

    def test_only_decision_time_ledger_fields_are_attached(self) -> None:
        entry = SimpleNamespace(
            seq=7, kind="decision", session_phase="rth", hours_to_discovery=0.0,
            stated_confidence=0.7, lean="up", lean_confidence=0.6, verdict="no_trade",
            counterfactual_move_bps="-120.0", exit_price="1", net_pnl="3",
            direction_correct=False, settled_at="2026-09-13T00:00:00+00:00",
        )
        merged = attach_decisions([_record(7)], [entry])[0]
        assert set(merged["decision"]) == set(DECISION_FIELDS)
        assert "counterfactual_move_bps" not in json.dumps(merged)
        got = features_of(merged)
        assert got["lean"] == "up" and got["session_phase"] == "rth"


class TestSequentialAgreementFix:
    rule = next(r for r in STANDING_RULES if r.name == "sequential-agreement")

    def test_a_concurrent_panel_saying_rather_than_contagion_does_not_fire_it(self) -> None:
        assert not self.rule.predicate(_record(1, concurrent=True))

    def test_a_sequential_panel_still_fires_it(self) -> None:
        assert self.rule.predicate(_record(1, concurrent=False))


# --- the lean as a defect ------------------------------------------------------------------------


def _entry(seq: int, lean: str, move: float | None, *, kind: str = "decision") -> SimpleNamespace:
    return SimpleNamespace(
        seq=seq, symbol="NVDAUSDT", kind=kind, lean=lean,
        counterfactual_move_bps=None if move is None else str(move),
        exit_price=None, entry_price="100",
    )


class TestLeanDefects:
    def test_a_lean_the_market_contradicted_is_a_defect_and_one_it_confirmed_is_not(self) -> None:
        found = defects_from_leans([_entry(1, "up", -40.0), _entry(2, "down", -40.0),
                                    _entry(3, "down", 25.0)])
        assert [(d.seq, d.kind) for d in found] == [(1, DefectKind.LEAN), (3, DefectKind.LEAN)]

    def test_no_lean_a_flat_tape_an_unsettled_row_and_a_seal_are_neither_right_nor_wrong(
        self,
    ) -> None:
        rows = [_entry(1, "none", -40.0), _entry(2, "up", -3.0), _entry(3, "up", None),
                _entry(4, "up", -40.0, kind="settlement_seal")]
        assert defects_from_leans(rows) == []
        assert lean_settled(rows) == set()
        assert lean_settled([_entry(5, "up", 30.0), _entry(6, "down", 30.0)]) == {5, 6}


# --- pairing -------------------------------------------------------------------------------------


def _marked(records: list[dict[str, Any]], bad: set[int]) -> list[Any]:
    from argus.desk.review import Defect

    return [Defect(seq=r["seq"], symbol=r["symbol"], kind=DefectKind.GROUNDING, detail="x")
            for r in records if r["seq"] in bad]


class TestPairing:
    def test_the_nearest_clean_decision_on_the_same_symbol_is_chosen(self) -> None:
        records = [
            _record(1, sources=3),
            _record(2, sources=4, symbol="TSLAUSDT"),    # nearer, but another instrument
            _record(3, sources=12),
            _record(4, sources=20),
        ]
        pairs = pair_contrasts(records, _marked(records, {1}), DefectKind.GROUNDING)
        assert [(p.bad_seq, p.good_seq) for p in pairs] == [(1, 3)]
        assert "sources" in pairs[0].differing

    def test_an_indistinguishable_bad_decision_is_dropped_rather_than_paired(self) -> None:
        at = datetime(2026, 9, 12, 15, tzinfo=UTC)
        records = [_record(1, sources=5, at=at), _record(2, sources=5, at=at)]
        assert pair_contrasts(records, _marked(records, {1}), DefectKind.GROUNDING) == []

    def test_graded_restricts_both_sides(self) -> None:
        records = [_record(1, sources=3), _record(2, sources=12), _record(3, sources=11)]
        pairs = pair_contrasts(records, _marked(records, {1}), DefectKind.GROUNDING,
                               graded={1, 3})
        assert [(p.bad_seq, p.good_seq) for p in pairs] == [(1, 3)]

    def test_interleave_spreads_a_budget_across_kinds_and_symbols(self) -> None:
        def pair(seq: int, kind: DefectKind, symbol: str) -> ContrastPair:
            r = _record(seq, symbol=symbol)
            return ContrastPair(kind, r, r, 0.1, ("sources",), "")

        g = [pair(1, DefectKind.GROUNDING, "A"), pair(2, DefectKind.GROUNDING, "A"),
             pair(3, DefectKind.GROUNDING, "B")]
        lean = [pair(9, DefectKind.LEAN, "A")]
        assert [p.bad_seq for p in interleave([g, lean])] == [1, 9, 3, 2]


# --- proposers ----------------------------------------------------------------------------------


class TestContrastProposer:
    def test_its_rule_separates_its_own_pair(self) -> None:
        records = [_record(1, sources=3, analysts=4), _record(2, sources=14, analysts=2),
                   _record(3, sources=9, analysts=3)]
        pair = pair_contrasts(records, _marked(records, {1}), DefectKind.GROUNDING)[0]
        proposal = ContrastProposer(records=records).propose(pair)
        assert proposal.spec is not None and proposal.calls == 0
        assert proposal.spec.fires(pair.bad) and not proposal.spec.fires(pair.good)
        assert len(proposal.spec.conditions) <= 2
        assert proposal.spec.targets == {DefectKind.GROUNDING}

    def test_the_author_follows_the_features_that_separate_the_pair(self) -> None:
        assert author_of(["sources", "has_filing"]) is FaultAuthor.ENVIRONMENT
        assert author_of(["independence", "sources"]) is FaultAuthor.DESK
        assert author_of(["sources"], journal=True) is FaultAuthor.TRADER


class ScriptedModel:
    """`complete_json`'s contract without a network: each scripted object is validated exactly as
    `QwenClient.complete_json` validates it, and a failure is retried up to ``attempts``."""

    def __init__(self, responses: list[dict[str, Any] | Exception]) -> None:
        self.responses = list(responses)
        self.calls = 0
        self.seen: list[list[dict[str, Any]]] = []

    def complete_json(
        self, messages: list[dict[str, Any]], *, required_keys: tuple[str, ...] = (),
        validate: Callable[[dict[str, Any]], str] | None = None, max_tokens: int = 0,
        attempts: int = 3,
    ) -> dict[str, Any]:
        last = ""
        for _ in range(attempts):
            self.seen.append(messages)
            self.calls += 1
            item = self.responses.pop(0)
            if isinstance(item, Exception):
                raise item
            missing = [k for k in required_keys if k not in item]
            last = f"missing {missing}" if missing else (validate(item) if validate else "")
            if not last:
                return item
        raise RuntimeError(f"could not obtain valid JSON after {attempts} attempts — {last}")


def _answer(rule: dict[str, Any] | None, **over: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"discussion": "fewer sources", "author": "environment",
                           "description": "a figure traced to nothing", "rule": rule}
    out.update(over)
    return out


_GOOD_RULE = {"name": "narrow-evidence", "prompt": "Ask where each figure came from.",
              "rationale": "few sources leave room for an unsupported number",
              "conditions": [{"feature": "sources", "op": "<", "value": 6}]}


def _pair() -> ContrastPair:
    records = [_record(1, sources=3), _record(2, sources=12)]
    return pair_contrasts(records, _marked(records, {1}), DefectKind.GROUNDING)[0]


class TestModelProposer:
    def test_a_valid_answer_becomes_a_spec_targeting_the_pairs_kind(self) -> None:
        model = ScriptedModel([_answer(_GOOD_RULE)])
        proposal = ModelProposer(model=model).propose(_pair())
        assert proposal.spec is not None and proposal.spec.targets == {DefectKind.GROUNDING}
        assert proposal.author is FaultAuthor.ENVIRONMENT and proposal.calls == 1

    def test_the_prompt_carries_the_pair_and_nothing_from_a_checker_verdict_as_a_feature(
        self,
    ) -> None:
        model = ScriptedModel([_answer(_GOOD_RULE)])
        ModelProposer(model=model).propose(_pair())
        user = json.loads(model.seen[0][1]["content"])
        assert user["defective_decision"]["seq"] == 1 and user["clean_decision"]["seq"] == 2
        assert set(user["defective_decision"]["features"]) <= set(DESK_FEATURES)

    def test_an_invalid_rule_is_retried_with_the_complaint_and_both_calls_are_counted(
        self,
    ) -> None:
        bad = {**_GOOD_RULE, "conditions": [{"feature": "grounding_failures", "op": ">",
                                             "value": 0}]}
        model = ScriptedModel([_answer(bad), _answer(_GOOD_RULE)])
        proposer = ModelProposer(model=model)
        proposal = proposer.propose(_pair())
        assert proposal.spec is not None
        assert proposal.calls == 2 and proposer.calls == 2 and model.calls == 2

    def test_a_decline_is_a_recorded_refusal_not_a_rule(self) -> None:
        model = ScriptedModel([_answer(None, no_rule="nothing here explains it")])
        proposal = ModelProposer(model=model).propose(_pair())
        assert proposal.spec is None and "nothing here explains it" in proposal.refused

    def test_a_failed_call_is_a_refusal_with_its_calls_counted(self) -> None:
        model = ScriptedModel([RuntimeError("HTTP 500")])
        proposer = ModelProposer(model=model)
        proposal = proposer.propose(_pair())
        assert proposal.spec is None and "HTTP 500" in proposal.refused
        assert proposer.calls == 1

    def test_the_author_is_validated_against_the_taxonomy(self) -> None:
        model = ScriptedModel([_answer(_GOOD_RULE, author="the market"), _answer(_GOOD_RULE)])
        proposal = ModelProposer(model=model).propose(_pair())
        assert proposal.spec is not None and model.calls == 2


# --- a trader's own journal ----------------------------------------------------------------------


def _trade(at: datetime, outcome: float | None, **over: Any) -> str:
    row: dict[str, Any] = {"at": at.isoformat(), "symbol": "nvda", "side": "long",
                           "confidence": 0.6, "holding_hours": 2.0, "outcome_bps": outcome}
    row.update(over)
    return json.dumps(row)


class TestJournal:
    def test_malformed_lines_are_refused_with_their_line_number(self) -> None:
        t = datetime(2026, 9, 1, tzinfo=UTC)
        with pytest.raises(JournalError, match="line 2"):
            parse_journal([_trade(t, 10.0), json.dumps({"at": t.isoformat(), "symbol": "X",
                                                        "side": "sideways"})])

    def test_after_a_loss_reads_only_trades_that_had_closed(self) -> None:
        t = datetime(2026, 9, 1, 9, tzinfo=UTC)
        entries = parse_journal([
            _trade(t, -80.0),                            # closes at 11:00
            _trade(t + timedelta(hours=1), 40.0, holding_hours=5.0),  # still open at 12:00
            _trade(t + timedelta(hours=3), -20.0),       # opens after it closed
        ])
        records, defects, graded = journal_records(entries)
        flags = [features_of(r, JOURNAL_FEATURES).get("after_a_loss") for r in records]
        assert flags == [None, None, True]
        assert [d.seq for d in defects] == [1, 3] and graded == {1, 2, 3}
        assert features_of(records[2], JOURNAL_FEATURES)["trades_earlier_same_day"] == 2

    def test_open_and_flat_trades_are_ungraded(self) -> None:
        t = datetime(2026, 9, 1, tzinfo=UTC)
        _, defects, graded = journal_records(parse_journal([_trade(t, None), _trade(t, 2.0)]))
        assert defects == [] and graded == set()


def test_rule_spec_round_trips_through_its_own_dict() -> None:
    spec = spec_from({
        "name": "stance-in", "prompt": "Check the stance before accepting it.",
        "rationale": "a fixed reason for the test",
        "targets": ["conflict"],
        "conditions": [{"feature": "panel_stance", "op": "in", "value": ["bullish"]}],
    })
    again = spec_from({**spec.as_dict()})
    assert isinstance(again, RuleSpec) and again == spec


def test_defects_from_notes_are_untouched_by_the_new_vocabulary() -> None:
    row = _record(1, extra=("[grounding] 1 of 3 figure(s) do not resolve to anything",))
    row["flags"] = [row["notes"][-1]]
    assert [d.kind for d in defects_from_notes([row])] == [DefectKind.GROUNDING]
