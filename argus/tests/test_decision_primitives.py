"""S23 decision-quality primitives: three-state support, m-of-K, the rubric pair, forced
reflection, the id remap and the prompt cache — and the evaluation that measures them.

Every model here is scripted. No test reaches the Qwen endpoint: the one test that exercises the
real client overrides ``_post``, the single place it makes a request.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from argus.agents import meta_pm as meta_pm_mod
from argus.agents import rubric
from argus.agents.claims import check as check_claims
from argus.agents.grounding import (
    PARTIAL_TOLERANCE,
    Support,
    check,
    combine,
    extract,
    fact_unit,
)
from argus.agents.meta_pm import (
    REFLECTION_PROMPT,
    SYSTEM_PROMPT,
    MarketFrame,
    MetaPM,
    reflection_complaint,
    shown_seqs,
)
from argus.cost.model import CostModel
from argus.decision.verdicts import (
    DEFAULT_AGREEMENT,
    Intent,
    Side,
    Verdict,
    agreement_ladder,
    agreement_note,
    backs,
    m_of_k,
)
from argus.desk.review import _MARKERS as REVIEW_MARKERS
from argus.eval import decision_primitives as dp
from argus.llm.cache import CACHE_SCHEMA, CachedModel, PromptCache, deterministic
from argus.llm.idmap import IdMap, restrict_to_shown
from argus.llm.ledger import CostLedger, Outcome
from argus.llm.qwen import Completion, QwenClient, Thinking, Usage
from argus.paper.runner import _FLAG_MARKERS
from argus.truth.clocks import ET, DualClock


def _flags(line: str) -> list[str]:
    """Every finding marker the runner or the review would read in ``line``."""
    low = line.lower()
    return [m for m in _FLAG_MARKERS if m in low] + [m for m, _ in REVIEW_MARKERS if m in low]


# --- three-state grounding --------------------------------------------------------------------


class TestGroundingSupport:
    def test_a_misquoted_hurdle_in_the_same_unit_is_a_near_miss(self) -> None:
        report = check("the hurdle is 20.80bps", facts={"total_hurdle_bps": 18.80})
        (res,) = report.resolutions
        assert not res.resolved and res.support is Support.PARTIAL
        assert res.near is not None and res.near.source == "total_hurdle_bps"
        assert res.near.distance == pytest.approx(2.0 / 20.80)

    def test_a_different_quantity_is_never_a_near_miss(self) -> None:
        # The coincidence that sank the first design: 20.80bps near 20 hours to discovery.
        report = check("the hurdle is 20.80bps", facts={"hours_to_discovery": 20.0})
        assert report.resolutions[0].support is Support.NONE
        assert report.support is Support.NONE

    def test_outside_the_band_or_across_a_sign_there_is_no_support(self) -> None:
        assert check("a 25bps hurdle", facts={"total_hurdle_bps": 18.80}).support is Support.NONE
        assert check("moved -5.2bps", facts={"move_bps": 5.0}).support is Support.NONE
        assert PARTIAL_TOLERANCE == 0.10

    def test_binary_meaning_is_unchanged_and_the_score_gives_half_credit(self) -> None:
        report = check(
            "fee 12bps, hurdle 20.80bps",
            facts={"round_trip_bps": 12.0, "total_hurdle_bps": 18.80},
        )
        assert not report.grounded  # a near miss is still unresolved
        assert report.coverage == pytest.approx(0.5)
        assert report.support_score == pytest.approx(0.75)
        assert report.support is Support.PARTIAL

    def test_thesis_level_combination(self) -> None:
        assert combine([]) is Support.FULL
        assert combine([Support.FULL, Support.FULL]) is Support.FULL
        assert combine([Support.NONE, Support.NONE]) is Support.NONE
        assert combine([Support.FULL, Support.NONE]) is Support.PARTIAL
        assert [s.score for s in Support] == [1.0, 0.5, 0.0]

    def test_units_are_read_from_fact_names_and_evidence_units_must_align(self) -> None:
        assert fact_unit("total_hurdle_bps") == "bps"
        assert fact_unit("change_pct") == "%"
        assert fact_unit("token_price") is None
        with pytest.raises(ValueError):
            check("x 3%", facts={}, evidence_values=[("e1", 3.0)], evidence_units=[])
        near = check("rose 3.2%", facts={}, evidence_values=[("e1", 3.0)], evidence_units=["%"])
        assert near.resolutions[0].support is Support.PARTIAL

    def test_the_support_line_is_not_a_second_finding(self) -> None:
        lines = check("fee 12bps, hurdle 20.80bps",
                      facts={"round_trip_bps": 12.0, "total_hurdle_bps": 18.80}).render()
        assert len(lines) == 3 and _flags(lines[0])  # the failure is still flagged, once
        assert lines[2].startswith("[grounding] support: partial (0.75)")
        assert "1 more misquotes one" in lines[2]
        assert _flags(lines[2]) == []
        none = check("a 25bps hurdle", facts={"total_hurdle_bps": 18.80}).render()
        assert none[2].startswith("[grounding] support: none") and _flags(none[2]) == []

    def test_a_grounded_thesis_renders_exactly_as_before(self) -> None:
        assert check("fee 12bps", facts={"round_trip_bps": 12.0}).render() == [
            "[grounding] all 1 figure(s) resolve to a computed value or a cited fact"
        ]


class TestClaimSupport:
    RECORDS = (("f1", {"pre_arranged": True}), ("f2", {"pre_arranged": False}))

    def test_a_claim_true_of_some_records_is_partial_and_not_a_contradiction(self) -> None:
        report = check_claims("the insider sales are pre-arranged", records=self.RECORDS)
        assert report.sound and report.claims_examined == 1
        assert report.support is Support.PARTIAL
        (partial,) = report.partial
        assert partial.agreeing == ("f1",) and partial.disagreeing == ("f2",)
        rendered = report.render()
        assert any("partially supported" in line for line in rendered)
        assert all(_flags(line) == [] for line in rendered)
        assert report.as_dict()["support"] == "partially_supported"

    def test_full_and_none_bracket_it(self) -> None:
        agree = check_claims("the sales are pre-arranged", records=self.RECORDS[:1])
        assert agree.support is Support.FULL and not agree.partial
        refuted = check_claims("the sales are pre-arranged", records=self.RECORDS[1:])
        assert refuted.support is Support.NONE and not refuted.sound

    def test_no_examined_claim_is_not_reported_as_support(self) -> None:
        assert check_claims("nothing checkable here", records=self.RECORDS).support is None

    def test_the_hurdle_sentence_no_longer_fires_the_conviction_rule(self) -> None:
        thesis = "the 20.80bps total hurdle requires a high-conviction directional call"
        report = check_claims(thesis, records=(("f1", {"conviction": False}),))
        assert report.claims_examined == 0 and report.sound


# --- m-of-K agreement ---------------------------------------------------------------------------


class TestAgreement:
    def test_the_stated_threshold_calls_only_real_agreement(self) -> None:
        assert DEFAULT_AGREEMENT == 2
        assert m_of_k(["bullish", "bullish", "neutral"]).call == "up"
        assert m_of_k(["bullish", "neutral", "insufficient_evidence"]).call is None
        assert m_of_k(["bullish", "neutral", "neutral"], 1).call == "up"
        assert m_of_k(["bearish", "short", "bullish"]).call == "down"

    def test_both_sides_reaching_the_threshold_is_a_split(self) -> None:
        vote = m_of_k(["bullish", "bearish"], 1)
        assert vote.split and vote.call is None and not vote.decisive
        assert vote.render() == "1-of-2: split"

    def test_a_threshold_below_one_is_refused(self) -> None:
        with pytest.raises(ValueError):
            m_of_k(["bullish"], 0)

    def test_the_ladder_reports_every_threshold(self) -> None:
        ladder = agreement_ladder(["bullish", "bullish", "bearish"])
        assert [v.call for v in ladder] == [None, "up", None]
        assert ladder[0].abstaining == 0 and ladder[2].voters == 3

    def test_backs_reads_the_intent_side_and_never_changes_it(self) -> None:
        trade = Intent(symbol="rNVDA", side=Side.SELL, quantity=Decimal("5"),
                       verdict=Verdict.TRADE, stated_confidence=0.6, thesis="t",
                       invalidation=("x",))
        assert backs(trade, m_of_k(["bearish", "bearish"])) is True
        assert backs(trade, m_of_k(["bullish", "bullish"])) is False
        abstain = Intent(symbol="rNVDA", side=Side.SELL, quantity=Decimal("0"),
                         verdict=Verdict.NO_TRADE, stated_confidence=0.6, thesis="t")
        assert backs(abstain, m_of_k(["bearish", "bearish"])) is None

    def test_the_note_marks_the_stated_threshold_and_flags_nothing(self) -> None:
        note = agreement_note(["bullish", "bullish", "neutral"])
        assert note.startswith("[agreement] 3 analyst(s): 2 up, 0 down, 1 neither")
        assert "2-of-3: up (stated threshold)" in note and _flags(note) == []
        assert "no analyst" in agreement_note([])


# --- the rubric pair ----------------------------------------------------------------------------

_RUBRIC = {"groundedness": 0.9, "evidence_breadth": 0.4, "independence": "0.5",
           "edge_over_hurdle": 0.2}


class TestRubric:
    def test_parse_is_all_or_nothing(self) -> None:
        assert rubric.parse({"self_rubric": _RUBRIC}) == {
            "groundedness": 0.9, "evidence_breadth": 0.4, "independence": 0.5,
            "edge_over_hurdle": 0.2,
        }
        for broken in ({**_RUBRIC, "independence": 1.5}, {**_RUBRIC, "groundedness": True},
                       {k: v for k, v in _RUBRIC.items() if k != "groundedness"},
                       {**_RUBRIC, "evidence_breadth": "high"}):
            assert rubric.parse({"self_rubric": broken}) is None
        assert rubric.parse({"confidence": 0.7}) is None

    def test_complaints_name_the_fault(self) -> None:
        assert "missing or is not an object" in rubric.complaint({})
        assert "missing ['groundedness']" in rubric.complaint(
            {"self_rubric": {k: v for k, v in _RUBRIC.items() if k != "groundedness"}})
        assert "0 to 1" in rubric.complaint({"self_rubric": {**_RUBRIC, "independence": 2}})
        assert rubric.complaint({"self_rubric": _RUBRIC}) == ""

    def test_each_score_is_paired_with_the_desks_measure_under_the_desks_bar(self) -> None:
        scores = rubric.parse({"self_rubric": _RUBRIC})
        assert scores is not None
        paired = {p.dimension: p for p in rubric.pairs(
            scores, grounding_support=0.75, distinct_sources=7, independence=2.0,
            settled_move_bps=30.0, hurdle_bps=20.0)}
        assert paired["groundedness"].meets_bar is False
        assert paired["evidence_breadth"].meets_bar is True
        assert paired["independence"].meets_bar is False
        assert paired["edge_over_hurdle"].measured == pytest.approx(1.5)
        assert paired["edge_over_hurdle"].meets_bar is True
        with pytest.raises(ValueError):
            rubric.meets_bar("vibes", 1.0)

    def test_the_desk_note_round_trips_and_leaves_edge_to_settlement(self) -> None:
        assert rubric.desk_note(None, grounding_support=1.0, distinct_sources=6,
                                independence=3.0) is None
        scores = rubric.parse({"self_rubric": _RUBRIC})
        line = rubric.desk_note(scores, grounding_support=1.0, distinct_sources=6,
                                independence=3.0)
        assert line is not None and _flags(line) == []
        back = rubric.parse_pair_note(line)
        assert back["groundedness"] == (0.9, 1.0)
        assert back["edge_over_hurdle"] == (0.2, None)
        assert rubric.parse_pair_note("[rubric] self-scored: groundedness 0.90") == {}

    def test_agreement_statistics(self) -> None:
        assert rubric.auc([0.1, 0.2, 0.8, 0.9], [False, False, True, True]) == 1.0
        assert rubric.auc([0.5, 0.5], [True, False]) == 0.5
        assert rubric.auc([0.5], [True]) is None
        assert rubric.spearman([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
        assert rubric.spearman([1, 1, 1], [1, 2, 3]) is None
        assert rubric.auc_p_value(0.5, 10, 10) == pytest.approx(1.0)


# --- forced reflection and the rubric flag on the Meta-PM ---------------------------------------

_MEMORY = "seq 12 (2026-09-13, weekend): NO_TRADE, lean UP, settled +40bps\n"


def _frame(memory: str = _MEMORY) -> MarketFrame:
    at = datetime(2026, 3, 8, 3, 0, tzinfo=ET)
    return MarketFrame(
        symbol="rNVDA", as_of=at, session=DualClock().state(at, nav_age_seconds=40_000),
        token_price=Decimal("118.40"), position_quantity=Decimal("0"),
        round_trip_bps=CostModel.bitget_perp().round_trip_bps(),
        evidence=("[e1] BTC -3.7% over 6h.",), memory_block=memory,
    )


def _answer(**extra: Any) -> dict[str, Any]:
    return {"verdict": "NO_TRADE", "side": "BUY", "quantity": 0, "confidence": 0.4,
            "thesis": "the edge does not clear the round trip", "invalidation": [],
            "counter_case": "a gap at the open", "lean": "NONE", "lean_confidence": 0.5, **extra}


_REFLECTION = {"memory": "stood aside; the lean was right",
               "next_goal": "a close above 120 with volume"}


class ScriptedPM:
    """Answers from a script, and applies ``required_keys`` and ``validate`` the way the real
    client's ``complete_json`` does: a refused answer is fed back and the next one is asked for."""

    budget = None

    def __init__(self, answers: list[dict[str, Any]]) -> None:
        self.answers = list(answers)
        self.complaints: list[str] = []
        self.systems: list[str] = []

    def complete(self, *_: Any, **__: Any) -> Completion:
        raise AssertionError("the Meta-PM asks for structured answers only")

    def complete_json(self, messages: list[dict[str, Any]], *,
                      required_keys: tuple[str, ...] = (),
                      validate: Callable[[dict[str, Any]], str] | None = None,
                      max_tokens: int = 0, attempts: int = 3,
                      thinking: Thinking = Thinking.LOW) -> dict[str, Any]:
        self.systems.append(str(messages[0]["content"]))
        for _ in range(attempts):
            answer = self.answers.pop(0)
            missing = [k for k in required_keys if k not in answer]
            fault = f"missing {missing}" if missing else (validate(answer) if validate else "")
            if not fault:
                return answer
            self.complaints.append(fault)
        raise AssertionError(f"no acceptable answer: {self.complaints}")


class TestForcedReflection:
    def test_the_default_prompt_and_validator_are_untouched(self) -> None:
        model = ScriptedPM([_answer()])
        pm = MetaPM(model, candidates=1, repair_rounds=0)
        assert pm.system_prompt() == SYSTEM_PROMPT
        assert pm._validator(_frame()) is meta_pm_mod._complain_about
        attempt = pm.deliberate(_frame()).committed_attempt.as_dict()
        assert "reflection" not in attempt and "rubric" not in attempt
        assert model.systems == [SYSTEM_PROMPT]

    def test_the_evaluation_must_name_a_seq_the_memory_listed(self) -> None:
        frame = _frame()
        assert shown_seqs(frame) == ("12",)
        good = _answer(evaluation_previous_decision="seq 12 proved right", **_REFLECTION)
        assert reflection_complaint(good, frame) == ""
        invented = _answer(evaluation_previous_decision="seq 99 proved wrong", **_REFLECTION)
        assert "seq 99" in reflection_complaint(invented, frame)
        assert "listed: seq 12" in reflection_complaint(invented, frame)
        unnamed = _answer(evaluation_previous_decision="it went fine", **_REFLECTION)
        assert "names no prior decision" in reflection_complaint(unnamed, frame)
        assert "\"memory\" is missing" in reflection_complaint(
            _answer(evaluation_previous_decision="seq 12 right", next_goal="x"), frame)

    def test_a_first_look_must_not_name_a_seq(self) -> None:
        first = _frame(memory="")
        ok = _answer(evaluation_previous_decision="FIRST LOOK", **_REFLECTION)
        assert reflection_complaint(ok, first) == ""
        bad = _answer(evaluation_previous_decision="seq 3 was right", **_REFLECTION)
        assert "FIRST LOOK" in reflection_complaint(bad, first)

    def test_both_flags_end_to_end_with_the_hallucinated_seq_sent_back(self) -> None:
        model = ScriptedPM([
            _answer(evaluation_previous_decision="seq 99 proved wrong", **_REFLECTION,
                    self_rubric=_RUBRIC),
            _answer(evaluation_previous_decision="seq 12 proved right", **_REFLECTION,
                    self_rubric=_RUBRIC),
        ])
        pm = MetaPM(model, candidates=1, repair_rounds=0, reflection=True, rubric=True)
        deliberation = pm.deliberate(_frame())
        assert len(model.complaints) == 1 and "seq 99" in model.complaints[0]
        assert REFLECTION_PROMPT in model.systems[0]
        assert rubric.RUBRIC_PROMPT in model.systems[0]
        committed = deliberation.committed_attempt
        assert committed.reflection is not None
        assert committed.reflection["evaluation_previous_decision"] == "seq 12 proved right"
        assert committed.rubric == rubric.parse({"self_rubric": _RUBRIC})
        rendered = deliberation.render()
        assert any(line.startswith("[reflection] evaluated: seq 12") for line in rendered)
        assert any(line.startswith("[rubric] self-scored: groundedness 0.90") for line in rendered)

    def test_a_missing_rubric_is_returned_to_the_model(self) -> None:
        model = ScriptedPM([_answer(), _answer(self_rubric=_RUBRIC)])
        MetaPM(model, candidates=1, repair_rounds=0, rubric=True).deliberate(_frame())
        assert model.complaints == ["missing ['self_rubric']"]


# --- id remap -----------------------------------------------------------------------------------

_IDS = ["twitter-2103455950178832631", "edgar-0001045810-26-000078", "cnbc-77"]


class TestIdMap:
    def test_handles_round_trip_and_unshown_ones_are_rejected_by_name(self) -> None:
        remap = IdMap([*_IDS, _IDS[0]])
        assert remap.shown == ("E0", "E1", "E2") and len(remap) == 3
        got = remap.resolve_all(["E1", " E0 ", "E1", "E7", "E01", "e2", "", None])
        assert got.ids == (_IDS[1], _IDS[0])
        assert got.rejected == ("E7", "E01", "e2")
        assert got.duplicates == 1 and not got.clean
        assert remap.resolve_all("E2").ids == (_IDS[2],)

    def test_expand_rewrites_whole_shown_handles_only(self) -> None:
        remap = IdMap(_IDS)
        assert remap.expand("per E1 and E12, not E2x") == f"per {_IDS[1]} and E12, not E2x"

    def test_handles_are_never_read_as_figures(self) -> None:
        assert extract("E0 and E12 both say so") == ()

    def test_identity_mode_and_the_guards(self) -> None:
        seqs = IdMap.identities(["12", "15"])
        assert seqs.resolve_all(["12", "99"]).rejected == ("99",)
        assert restrict_to_shown(["a", "b"], ["a"]).ids == ("a",)
        with pytest.raises(ValueError):
            IdMap(_IDS, prefix="7")
        with pytest.raises(KeyError):
            IdMap(_IDS).handle("never-registered")


# --- prompt cache -------------------------------------------------------------------------------


def _completion(text: str, *, tokens: int = 100, reported: bool = True) -> Completion:
    return Completion(content=text, reasoning="", finish_reason="stop",
                      usage=Usage(prompt_tokens=tokens - 10, completion_tokens=10,
                                  reasoning_tokens=0, total_tokens=tokens, reported=reported))


_MSGS = [{"role": "user", "content": "is 12bps cleared?"}]


def _ask_through(cache: PromptCache, calls: list[int], **params: Any) -> Completion:
    base: dict[str, Any] = {"temperature": 0.0, "max_tokens": 64, "json_mode": False,
                            "tools": None, "seed": None, "thinking": Thinking.LOW,
                            "stream": None}
    base.update(params)

    def ask() -> Completion:
        calls.append(1)
        return _completion(f"answer {len(calls)}")

    return cache.complete(ask, _MSGS, namespace="scripted", **base)


class TestPromptCache:
    def test_an_identical_deterministic_request_is_served_once(self) -> None:
        cache, calls = PromptCache(), list[int]()
        first = _ask_through(cache, calls)
        again = _ask_through(cache, calls)
        assert len(calls) == 1 and again.content == first.content
        _ask_through(cache, calls, max_tokens=65)  # any parameter is part of the key
        assert len(calls) == 2
        assert cache.stats.as_dict()["hit_rate"] == pytest.approx(1 / 3, abs=1e-4)
        assert cache.stats.tokens_saved == 100

    def test_a_sampled_request_is_never_reused(self) -> None:
        cache, calls = PromptCache(), list[int]()
        _ask_through(cache, calls, temperature=0.7)
        _ask_through(cache, calls, temperature=0.7)
        assert len(calls) == 2 and cache.stats.bypassed == 2 and cache.stats.lookups == 0
        _ask_through(cache, calls, temperature=0.7, seed=7)
        _ask_through(cache, calls, temperature=0.7, seed=7)
        assert len(calls) == 3
        assert deterministic(temperature=0.0, seed=None)
        assert not deterministic(temperature=0.2, seed=None)

    def test_answers_are_written_through_and_survive_a_new_process(self, tmp_path: Path) -> None:
        path = tmp_path / "cache.jsonl"
        cache, calls = PromptCache(path), list[int]()
        _ask_through(cache, calls)
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 1 and rows[0]["schema"] == CACHE_SCHEMA  # on disk before any close
        with path.open("a", encoding="utf-8") as handle:
            handle.write('{"schema": 99, "key": "k", "value": {}}\n{"torn')
        reloaded = PromptCache(path)
        assert len(reloaded) == 1 and reloaded.skipped_lines == 2
        _ask_through(reloaded, calls)
        assert len(calls) == 1 and reloaded.stats.hits == 1

    def test_a_hit_is_recorded_in_the_cost_ledger_and_unreported_usage_is_not_estimated(
        self,
    ) -> None:
        cache, ledger = PromptCache(), CostLedger(None)
        cache.put("k", {"content": "x", "usage": {"total_tokens": 50, "reported": False}})
        cache.count_hit(cache.get("k") or {}, namespace="scripted", thinking=Thinking.LOW,
                        stream=False, json_mode=False, tools=False, ledger=ledger)
        assert cache.stats.unmeasured_hits == 1 and cache.stats.tokens_saved == 0
        assert ledger.entries[-1].outcome == Outcome.CACHE_HIT


class _JsonModel:
    budget = None

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, *_: Any, **__: Any) -> Completion:
        raise AssertionError("not used")

    def complete_json(self, messages: list[dict[str, Any]], *,
                      required_keys: tuple[str, ...] = (),
                      validate: Callable[[dict[str, Any]], str] | None = None,
                      max_tokens: int = 2048, attempts: int = 3,
                      thinking: Thinking = Thinking.LOW) -> dict[str, Any]:
        self.calls += 1
        return {"verdict": "NO_TRADE", "call": self.calls}


class TestCachedModel:
    def test_a_stored_object_is_revalidated_by_each_caller(self) -> None:
        inner = _JsonModel()
        model = CachedModel(inner, namespace="scripted")
        assert model.complete_json(_MSGS, required_keys=("verdict",))["call"] == 1
        assert model.complete_json(_MSGS, required_keys=("verdict",))["call"] == 1
        stricter = model.complete_json(
            _MSGS, required_keys=("verdict",),
            validate=lambda obj: "" if obj["call"] > 1 else "stale",
        )
        assert stricter["call"] == 2 and inner.calls == 2
        assert model.cache.stats.revalidation_misses == 1


class _CountingQwen(QwenClient):
    """The real client with its one request method replaced: counts, never sends."""

    def __init__(self) -> None:
        super().__init__(api_key="offline-test", base_url="https://offline.invalid",
                         ledger=CostLedger(None))
        self.posts = 0

    def _post(self, payload: dict[str, Any]) -> Completion:
        self.posts += 1
        return _completion(f"post {self.posts}")


class TestQwenClientCache:
    def test_the_client_cache_serves_greedy_repeats_and_never_a_sample(self) -> None:
        client = _CountingQwen()
        client.complete(_MSGS, temperature=0.0)
        client.complete(_MSGS, temperature=0.0)
        assert client.posts == 1
        client.complete(_MSGS, temperature=0.7)
        client.complete(_MSGS, temperature=0.7)
        assert client.posts == 3


# --- the evaluation's readers, on synthetic rows ------------------------------------------------


class TestEvaluationReaders:
    def test_grounding_rows_read_the_rendered_lines(self) -> None:
        notes = [
            {"seq": 1, "notes": ["[grounding] 1 of 4 figure(s) do not resolve to anything the "
                                 "desk was given: 20.80bps"]},
            {"seq": 2, "notes": ["[grounding] all 3 figure(s) resolve to a computed value or a "
                                 "cited fact"]},
            {"seq": 3, "notes": ["panel: 0 analysts"]},
        ]
        rows = dp.grounding_rows(notes)
        assert [(r.seq, r.level) for r in rows] == [(1, Support.PARTIAL), (2, Support.FULL)]
        assert rows[0].named == ("20.80bps",)

    def test_a_panel_is_reconstructed_from_its_conflict_lines(self) -> None:
        panel_line = ("panel: 2 analysts, 3 distinct sources, independence 1.50 -> bullish at 0.55 "
                      "after provenance discount")
        reading = dp.read_panel({"seq": 9, "notes": [
            panel_line,
            "[conflict:direction] macro (bullish) vs flow (bearish) — opposite directions",
        ]})
        # Slots are the named analysts in sorted order (flow, macro), then any unnamed ones.
        assert reading is not None and reading.assignments == (("bearish", "bullish"),)
        assert reading.call_at(1) == (True, None)  # a split: determined, and no call
        silent = dp.read_panel({"seq": 10, "notes": [panel_line]})
        assert silent is not None and silent.skipped == "no conflict record"

    def test_the_live_three_state_lines_are_counted(self) -> None:
        notes = [
            {"seq": 5, "notes": ["[grounding] 1 of 2 figure(s) do not resolve to anything the "
                                 "desk was given: 9%"]},
            {"seq": 6, "notes": [
                "[grounding] 1 of 7 figure(s) do not resolve to anything the desk was given: 19bps",
                "[grounding] support: partial (0.93) — 6 of 7 figure(s) trace to a given value; "
                "1 more misquotes one by under 10%: 19bps is 5% off x_bps = 18",
                "[claim] pre_arranged: partially supported — \"...\" holds for 1 of 2",
            ]},
            {"seq": 7, "notes": ["[grounding] 1 of 2 figure(s) do not resolve to anything the "
                                 "desk was given: 9%"]},
        ]
        live = dp.three_state_live(notes)
        assert live["first_seq_with_a_support_line"] == 6
        assert live["failing_decisions_since"] == 2
        assert live["failing_decisions_missing_the_support_line"] == 1  # seq 7
        assert live["near_miss_figures_printed"] == 1
        assert live["partial_claims_by_rule"] == {"pre_arranged": 1}

    def test_rubric_pairs_are_scored_from_the_desks_lines(self) -> None:
        scores = rubric.parse({"self_rubric": _RUBRIC})
        lines = [
            rubric.desk_note(scores, grounding_support=g, distinct_sources=s, independence=i)
            for g, s, i in ((1.0, 8, 3.0), (0.5, 3, 1.0), (1.0, 9, 2.8))
        ]
        recorded = dp.rubric_pairs_on_record([{"seq": 1, "notes": lines}])
        assert recorded["pair_notes"] == 3
        assert recorded["axes"]["groundedness"]["n"] == 3
        assert "edge_over_hurdle" not in recorded["axes"]  # n/a until settlement

    def test_exact_tests(self) -> None:
        assert dp.fisher_two_sided(3, 1, 1, 3) == pytest.approx(0.4857, abs=1e-4)
        assert dp.wilson(0, 0) is None
        low, high = dp.wilson(5, 10) or (0.0, 0.0)
        assert low < 0.5 < high
