"""S4-S7: rank-before-commit, checker-triggered repair, the debate stall detector, and the
unused-evidence stress question — each tested with scripted models, never the live key."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, ClassVar

import pytest

from argus.agents import debate as debate_mod
from argus.agents.debate import (
    STALL_NOTICE,
    DebateBudget,
    Ending,
    Position,
    Side,
    hold,
    progress,
    round_stalled,
)
from argus.agents.meta_pm import (
    DEFAULT_CANDIDATES,
    DEFAULT_REPAIR_ROUNDS,
    MAX_REPAIR_ROUNDS,
    MarketFrame,
    MetaPM,
    check_response,
    given_values,
    rank_key,
)
from argus.desk.stress import (
    cited_evidence,
    lexical_encoder,
    split_evidence_line,
    unused_evidence,
    unused_evidence_stress,
)
from argus.eval import thesis_quality as tq
from argus.llm.qwen import Completion, QwenClient, QwenError, Usage
from argus.truth.clocks import DualClock

AT = datetime(2026, 9, 22, 15, 0, tzinfo=UTC)
EVIDENCE = (
    "[mkt-NVDAUSDT] (news, credibility 1.00, available 2026-09-22T15:00:00+00:00) "
    "NVDAUSDT last 222.88; 475 hourly bars of history to this instant",
    "[vix-1] (macro, credibility 0.90, available 2026-09-18T00:00:00+00:00) VIX 14.81, the 31% "
    "percentile of 9,276 daily closes",
    "[finra-1] (macro, credibility 0.95, available 2026-09-21T00:00:00+00:00) NVDA short volume "
    "37.8% of the session's prints",
)


def _frame(evidence: tuple[str, ...] = EVIDENCE) -> MarketFrame:
    return MarketFrame(
        symbol="NVDAUSDT", as_of=AT, session=DualClock().state(AT, nav_age_seconds=5.0),
        token_price=Decimal("222.88"), position_quantity=Decimal("0"),
        round_trip_bps=Decimal("12"), evidence=evidence, deliberation_bps=Decimal("3.02"),
    )


def _answer(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "verdict": "no_trade", "side": "buy", "quantity": 0, "confidence": 0.7,
        "thesis": "VIX at 14.81 and short volume of 37.8% give no edge over the 15.02 bps hurdle.",
        "invalidation": [], "counter_case": "A squeeze could start.", "lean": "none",
        "lean_confidence": 0.0,
    }
    base.update(over)
    return base


class ScriptedModel:
    """Answers `complete_json` from a list and records every message list it was sent."""

    def __init__(self, answers: list[dict[str, Any] | Exception]) -> None:
        self.answers = answers
        self.seen: list[list[dict[str, Any]]] = []
        self.budget = None

    def complete_json(self, messages: list[dict[str, Any]], **_: Any) -> dict[str, Any]:
        self.seen.append(messages)
        answer = self.answers[min(len(self.seen) - 1, len(self.answers) - 1)]
        if isinstance(answer, Exception):
            raise answer
        return answer

    def complete(self, *_: Any, **__: Any) -> Completion:  # pragma: no cover - unused
        raise NotImplementedError


# --- the checker --------------------------------------------------------------------------------


class TestTheChecker:
    def test_a_thesis_quoting_only_given_figures_is_clean(self) -> None:
        check = check_response(_answer(), _frame())
        assert check.defects == ()
        assert check.engages_evidence is True
        assert check.grounding_coverage == 1.0

    def test_an_invented_figure_is_named(self) -> None:
        check = check_response(_answer(thesis="Expected edge of 86.9bps beats the hurdle."),
                               _frame())
        assert "ungrounded_figure" in check.defects
        assert check.unresolved_figures == ("86.9bps",)

    def test_provenance_digits_cannot_ground_an_invented_figure(self) -> None:
        """The evidence prefix carries credibility 0.95 and timestamps; a thesis inventing "0.95"
        must not resolve against provenance."""
        _facts, values = given_values(_frame())
        assert 0.95 not in [v for _, v in values]
        assert ("vix-1", 14.81) in values

    def test_opening_exposure_without_a_falsifier_is_a_defect(self) -> None:
        check = check_response(_answer(verdict="trade", quantity=2, invalidation=[]), _frame())
        assert "no_falsifier" in check.defects

    def test_side_and_lean_pointing_opposite_ways_is_a_defect(self) -> None:
        check = check_response(
            _answer(verdict="trade", quantity=2, side="buy", lean="down",
                    invalidation=["VIX above 20"]), _frame(),
        )
        assert "side_contradicts_lean" in check.defects

    def test_a_missing_counter_case_is_a_defect(self) -> None:
        assert "no_counter_case" in check_response(_answer(counter_case=""), _frame()).defects

    def test_the_models_own_size_is_not_an_invented_figure(self) -> None:
        check = check_response(
            _answer(verdict="trade", quantity=5, invalidation=["VIX above 20"], lean="up",
                    thesis="Buy 5 units: VIX at 14.81 is calm."), _frame(),
        )
        assert "ungrounded_figure" not in check.defects

    def test_ranking_prefers_fewer_defects_then_score_then_the_first(self) -> None:
        clean = check_response(_answer(), _frame())
        dirty = check_response(_answer(counter_case=""), _frame())
        assert rank_key(clean, 1) < rank_key(dirty, 0)
        assert rank_key(clean, 0) < rank_key(clean, 1)


# --- S5: rank-before-commit ---------------------------------------------------------------------


class TestRankBeforeCommit:
    def test_the_default_is_one_call_exactly_as_before(self) -> None:
        model = ScriptedModel([_answer(counter_case="")])
        pm = MetaPM(model, repair_rounds=0)
        pm.decide(_frame(), decision_id="d")
        assert len(model.seen) == 1
        assert pm.candidates == DEFAULT_CANDIDATES

    def test_a_cleaner_second_candidate_is_committed(self) -> None:
        model = ScriptedModel([
            _answer(thesis="Edge 86.9bps.", counter_case=""),
            _answer(),
        ])
        pm = MetaPM(model, candidates=2, repair_rounds=0)
        proof = pm.decide(_frame(), decision_id="d")
        assert pm.last_deliberation is not None
        assert pm.last_deliberation.committed == 1
        assert "86.9" not in proof.llm_original_intent.thesis
        # The diversity instruction carried the first candidate.
        assert "not the first answer" in model.seen[1][-1]["content"]
        assert "86.9bps" in model.seen[1][-1]["content"]

    def test_on_a_tie_the_single_shot_answer_stands(self) -> None:
        model = ScriptedModel([_answer(), _answer(thesis="VIX at 14.81; no edge.")])
        pm = MetaPM(model, candidates=2, repair_rounds=0)
        pm.decide(_frame(), decision_id="d")
        assert pm.last_deliberation is not None
        assert pm.last_deliberation.committed == 0

    def test_a_failed_extra_candidate_does_not_take_the_decision_down(self) -> None:
        model = ScriptedModel([_answer(), QwenError("gateway 504")])
        pm = MetaPM(model, candidates=2, repair_rounds=0)
        proof = pm.decide(_frame(), decision_id="d")
        assert proof.llm_original_intent.thesis
        assert pm.last_deliberation is not None
        assert pm.last_deliberation.attempts[1].response is None
        assert "unavailable" in pm.last_deliberation.notes[0]

    def test_the_first_calls_failure_still_propagates(self) -> None:
        with pytest.raises(QwenError):
            MetaPM(ScriptedModel([QwenError("down")]), candidates=2).decide(
                _frame(), decision_id="d"
            )


# --- S6: checker-triggered repair ---------------------------------------------------------------


class TestRepair:
    def test_a_named_defect_is_shown_and_the_repair_committed(self) -> None:
        model = ScriptedModel([_answer(thesis="Edge 86.9bps."), _answer()])
        pm = MetaPM(model, repair_rounds=1)
        proof = pm.decide(_frame(), decision_id="d")
        assert len(model.seen) == 2
        instruction = model.seen[1][-1]["content"]
        assert "86.9bps" in instruction and "match no number you were given" in instruction
        assert pm.last_deliberation is not None and pm.last_deliberation.repaired
        assert "86.9" not in proof.llm_original_intent.thesis
        # Every attempt is kept, including the one that was replaced.
        assert "86.9" in str(pm.last_deliberation.attempts[0].response)

    def test_a_repair_that_adds_a_defect_does_not_win(self) -> None:
        model = ScriptedModel([
            _answer(thesis="Edge 86.9bps."),
            _answer(thesis="Edge 86.9bps and 42.0bps.", counter_case=""),
        ])
        pm = MetaPM(model, repair_rounds=1)
        proof = pm.decide(_frame(), decision_id="d")
        assert pm.last_deliberation is not None and pm.last_deliberation.committed == 0
        assert "42.0" not in proof.llm_original_intent.thesis

    def test_a_clean_answer_is_never_sent_back(self) -> None:
        model = ScriptedModel([_answer()])
        MetaPM(model, repair_rounds=2).decide(_frame(), decision_id="d")
        assert len(model.seen) == 1

    def test_rounds_are_capped(self) -> None:
        model = ScriptedModel([_answer(thesis="Edge 86.9bps.")])
        pm = MetaPM(model, repair_rounds=10)
        pm.decide(_frame(), decision_id="d")
        assert pm.repair_rounds == MAX_REPAIR_ROUNDS
        assert len(model.seen) == 1 + MAX_REPAIR_ROUNDS

    def test_the_default_matches_the_published_measurement(self) -> None:
        assert MetaPM(ScriptedModel([_answer()])).repair_rounds == DEFAULT_REPAIR_ROUNDS

    def test_the_deliberation_renders_what_changed(self) -> None:
        model = ScriptedModel([_answer(thesis="Edge 86.9bps."), _answer()])
        pm = MetaPM(model, repair_rounds=1)
        pm.decide(_frame(), decision_id="d")
        assert pm.last_deliberation is not None
        lines = pm.last_deliberation.render()
        assert "ungrounded_figure" in lines[1] and lines[1].endswith("none")
        assert json.dumps(pm.last_deliberation.as_dict())


# --- S7: the stall detector ---------------------------------------------------------------------


def _pos(side: Side, round_index: int, direction: str, magnitude: float | None,
         case: str = "RSI 18.5 is oversold") -> Position:
    return Position(side=side, round_index=round_index, direction=direction,
                    magnitude_bps=magnitude, case=case, strongest_opposing_point="their point")


class TestStallDetector:
    def test_a_restated_round_is_stalled(self) -> None:
        prior = [_pos(Side.BULL, 0, "up", 40), _pos(Side.BEAR, 0, "down", 30, "MACD -2.19")]
        reading = progress(prior, _pos(Side.BULL, 1, "up", 42,
                                       "Oversold at RSI 18.5, so a bounce is likely"))
        assert reading is not None and reading.stalled

    def test_a_new_figure_is_progress(self) -> None:
        prior = [_pos(Side.BULL, 0, "up", 40), _pos(Side.BEAR, 0, "down", 30, "MACD -2.19")]
        reading = progress(prior, _pos(Side.BULL, 1, "up", 40, "35 upgrades vs 1 downgrade"))
        assert reading is not None and not reading.stalled
        assert "#35" in reading.new_evidence

    def test_quoting_the_other_side_is_not_new_evidence(self) -> None:
        prior = [_pos(Side.BULL, 0, "up", 40), _pos(Side.BEAR, 0, "down", 30, "MACD -2.19")]
        reading = progress(prior, _pos(Side.BULL, 1, "up", 40, "MACD -2.19 is lagging"))
        assert reading is not None and reading.stalled

    def test_an_evidence_id_counts_as_a_citation(self) -> None:
        prior = [_pos(Side.BULL, 0, "up", 40)]
        reading = progress(prior, _pos(Side.BULL, 1, "up", 40, "see yahoo-tsla-441620102"))
        assert reading is not None and "yahoo-tsla-441620102" in reading.new_evidence

    def test_a_moved_magnitude_or_flipped_direction_is_progress(self) -> None:
        prior = [_pos(Side.BULL, 0, "up", 40)]
        moved = progress(prior, _pos(Side.BULL, 1, "up", 10))
        flipped = progress(prior, _pos(Side.BULL, 1, "unclear", 40))
        assert moved is not None and not moved.stalled
        assert flipped is not None and not flipped.stalled

    def test_a_missing_magnitude_is_never_counted_as_standing_still(self) -> None:
        reading = progress([_pos(Side.BULL, 0, "up", None)], _pos(Side.BULL, 1, "up", None))
        assert reading is not None and not reading.stalled

    def test_both_sides_must_stall(self) -> None:
        positions = [
            _pos(Side.BULL, 0, "up", 40), _pos(Side.BEAR, 0, "down", 30, "MACD -2.19"),
            _pos(Side.BULL, 1, "up", 40), _pos(Side.BEAR, 1, "down", 30, "new: 59.9% short"),
        ]
        assert not round_stalled(positions, 1)
        assert not round_stalled(positions, 0)


class Seat:
    def __init__(self, answers: list[dict[str, Any]]) -> None:
        self.answers = answers
        self.bodies: list[str] = []

    def complete_json(self, messages: list[dict[str, Any]], **_: Any) -> dict[str, Any]:
        self.bodies.append(str(messages[-1]["content"]))
        return self.answers[min(len(self.bodies) - 1, len(self.answers) - 1)]


def _say(direction: str, magnitude: float, case: str) -> dict[str, Any]:
    return {"direction": direction, "magnitude_bps": magnitude, "case": case,
            "strongest_opposing_point": "their point"}


RICH = DebateBudget(limit_bps=Decimal("100"))
EV = ("RSI 18.5", "MACD -2.19", "short volume 59.9%")


class TestTheDebateStops:
    def test_it_is_off_by_default_so_a_restating_debate_runs_its_rounds(self) -> None:
        got = hold(symbol="X", horizon_hours=24, evidence=EV,
                   bull_seat=Seat([_say("up", 40, "RSI 18.5")]),
                   bear_seat=Seat([_say("down", 40, "MACD -2.19")]), budget=RICH)
        assert got.ending is Ending.EXHAUSTED_ROUNDS and got.rounds == 3

    def test_on_it_stops_after_the_first_stalled_round_and_saves_the_rest(self) -> None:
        got = hold(symbol="X", horizon_hours=24, evidence=EV,
                   bull_seat=Seat([_say("up", 40, "RSI 18.5")]),
                   bear_seat=Seat([_say("down", 40, "MACD -2.19")]), budget=RICH,
                   stall_detection=True)
        assert got.ending is Ending.STALLED
        assert got.rounds == 2
        assert got.unresolved is True
        assert "STALLED" in got.render()
        assert got.as_dict()["progress"]

    def test_a_progressing_debate_is_not_stopped(self) -> None:
        bull = Seat([_say("up", 40, "RSI 18.5"), _say("up", 40, "35 upgrades"),
                     _say("up", 40, "revenue +17.9%")])
        bear = Seat([_say("down", 40, "MACD -2.19"), _say("down", 40, "59.9% short"),
                     _say("down", 40, "director sold 2.87 million")])
        got = hold(symbol="X", horizon_hours=24, evidence=EV, bull_seat=bull, bear_seat=bear,
                   budget=RICH, stall_detection=True)
        assert got.ending is Ending.EXHAUSTED_ROUNDS and got.rounds == 3

    def test_the_bounded_reset_issues_one_notice_then_stops(self) -> None:
        bull = Seat([_say("up", 40, "RSI 18.5")])
        bear = Seat([_say("down", 40, "MACD -2.19")])
        got = hold(symbol="X", horizon_hours=24, evidence=EV, bull_seat=bull, bear_seat=bear,
                   budget=RICH, stall_detection=True, reset_on_stall=True)
        assert got.resets == 1
        assert got.rounds == 3
        assert got.ending is Ending.STALLED
        assert STALL_NOTICE in bull.bodies[2] and STALL_NOTICE not in bull.bodies[1]


# --- S4: the unused-evidence stress question ----------------------------------------------------


def _toy_encoder(texts: list[str] | Any) -> list[list[float]]:
    """Three axes: rates, insiders, volatility. Deterministic, so the formula can be checked."""
    axes = (("yield", "rate", "curve"), ("insider", "sold", "director"), ("vix", "volatility"))
    return [[float(sum(t.lower().count(w) for w in axis)) + 0.01 for axis in axes]
            for t in texts]


class TestUnusedEvidence:
    EVID: ClassVar[list[tuple[str, str]]] = [
        ("vix-1", "VIX 14.81 volatility calm"),
        ("ins-1", "director sold 2.87 million shares; insider selling"),
        ("ust-1", "10y yield 5.18%, rate curve steep"),
    ]

    def test_cited_by_id_or_by_figure(self) -> None:
        assert cited_evidence("The vix-1 reading is calm", self.EVID) == {"vix-1"}
        assert cited_evidence("VIX sits at 14.81", self.EVID) == {"vix-1"}

    def test_the_storm_formula_surfaces_the_far_but_on_topic_item(self) -> None:
        ranked = unused_evidence(
            claim="volatility is calm and an insider director sold; rate curve matters too",
            queries=["a vix spike would invalidate this"],
            evidence=self.EVID, cited={"vix-1"}, encoder=_toy_encoder,
        )
        assert ranked[0].evidence_id in {"ins-1", "ust-1"}
        top = ranked[0]
        expected = ((1 - top.query_similarity) ** 0.5) * ((1 - top.cited_similarity) ** 0.5)
        assert top.score == pytest.approx(expected)

    def test_an_off_topic_item_is_gated_out(self) -> None:
        ranked = unused_evidence(
            claim="volatility vix volatility", queries=[],
            evidence=self.EVID, cited={"vix-1"}, encoder=_toy_encoder,
        )
        assert all(not u.gated_in and u.score == 0.0 for u in ranked)

    def test_the_entry_point_returns_nothing_when_everything_was_used(self) -> None:
        lines = ["[vix-1] (macro, credibility 0.90, available x) VIX 14.81"]
        assert unused_evidence_stress(thesis="VIX 14.81", evidence_lines=lines,
                                      encoder=lexical_encoder) is None

    def test_the_entry_point_asks_about_a_real_item(self) -> None:
        lines = [
            "[vix-1] (macro, credibility 0.90, available x) VIX 14.81 volatility calm",
            "[ins-1] (filing, credibility 1.00, available x) director sold shares; insider",
            "[ust-1] (macro, credibility 1.00, available x) rate curve 5.18%",
        ]
        got = unused_evidence_stress(
            thesis="VIX 14.81 volatility calm, while a director sold and the rate curve moved",
            invalidation=["a volatility spike"], evidence_lines=lines, encoder=_toy_encoder,
        )
        assert got is not None and got.evidence_id in {"ins-1", "ust-1"}
        assert got.question().startswith(f"The thesis did not use [{got.evidence_id}]")

    def test_split_keeps_the_claim_and_drops_provenance(self) -> None:
        eid, text = split_evidence_line(EVIDENCE[1])
        assert eid == "vix-1" and text.startswith("VIX 14.81") and "credibility" not in text

    def test_the_lexical_encoder_is_a_cosine_space(self) -> None:
        vectors = lexical_encoder(["insider sold shares", "insider sold stock", "treasury"])
        assert len({len(v) for v in vectors}) == 1


# --- the evaluation harness ---------------------------------------------------------------------


class TestTheHarness:
    def test_offline_replay_refuses_an_unrecorded_request(self) -> None:
        client = tq.CappedRecordingClient({}, live=False)
        with pytest.raises(QwenError, match="not in the recording"):
            client.complete([{"role": "user", "content": "json please"}])

    def test_a_recorded_response_is_replayed_at_zero_calls(self) -> None:
        client = tq.CappedRecordingClient({}, live=False)
        payload_messages = [{"role": "user", "content": "json please"}]
        payload = {"model": client._model, "messages": payload_messages, "temperature": 0.0,
                   "max_tokens": 2048}
        client._apply_thinking(payload, tq.Thinking.LOW)
        key = client._cache_key(payload)
        client.recording[key] = tq._completion_to(Completion(
            content='{"a": 1}', reasoning="", finish_reason="stop",
            usage=Usage(prompt_tokens=1, completion_tokens=1, reasoning_tokens=0,
                        total_tokens=2),
        ))
        got = client.complete(payload_messages)
        assert got.content == '{"a": 1}'
        assert client.total_posts == 0 and client.replayed == 1

    def test_the_cap_counts_every_http_attempt(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BITGET_QWEN_API_KEY", "test-only")
        monkeypatch.setenv("BITGET_QWEN_BASE_URL", "https://offline.invalid")
        sent: list[int] = []

        def fake_post(self: QwenClient, payload: dict[str, Any]) -> Completion:
            sent.append(1)
            return Completion(content='{"verdict": "x"}', reasoning="", finish_reason="stop",
                              usage=Usage(1, 1, 0, 2))

        monkeypatch.setattr(QwenClient, "_post", fake_post)
        client = tq.CappedRecordingClient({}, live=True, cap=3, stage_caps={"s5": 2, "s6": 5})
        client.complete([{"role": "user", "content": "a"}])
        client.complete([{"role": "user", "content": "b"}])
        with pytest.raises(QwenError, match="cap"):
            client.complete([{"role": "user", "content": "c"}])
        client.stage = "s6"
        client.complete([{"role": "user", "content": "d"}])
        with pytest.raises(QwenError, match="cap"):
            client.complete([{"role": "user", "content": "e"}])
        assert len(sent) == 3 and client.total_posts == 3

    def test_technicals_are_point_in_time(self) -> None:
        start = datetime(2026, 9, 1, tzinfo=UTC)
        closes = [(start + timedelta(hours=i), Decimal(100 + i % 7)) for i in range(200)]
        at = start + timedelta(hours=150)
        ev = tq.technical_evidence("X", at, closes)
        later = tq.technical_evidence("X", at, [*closes, (at + timedelta(hours=1),
                                                          Decimal("1000"))])
        assert ev is not None and later is not None and ev.claim == later.claim
        assert "RSI(14, hourly)" in ev.claim and ev.available_at == at

    def test_the_recorded_debate_log_is_read(self, tmp_path: Path) -> None:
        path = tmp_path / "notes.jsonl"
        path.write_text("\n".join(json.dumps({"notes": [n]}) for n in [
            "[debate] NVDAUSDT: 1 round(s), ended exhausted_budget, cost 2.0bps",
            "[debate] TSLAUSDT: 3 round(s), ended exhausted_rounds, cost 6.0bps",
            "[debate] not held — the panel agreed",
        ]), encoding="utf-8")
        got = tq.recorded_debate_log(path)
        assert got["debates_held"] == 2 and got["with_two_or_more_rounds"] == 1
        assert got["rounds_the_detector_could_have_saved"] == 1

    def test_the_stop_round_replays_the_counter(self) -> None:
        got = hold(symbol="X", horizon_hours=24, evidence=EV,
                   bull_seat=Seat([_say("up", 40, "RSI 18.5")]),
                   bear_seat=Seat([_say("down", 40, "MACD -2.19")]), budget=RICH)
        assert tq._stop_round(got) == 1
        assert debate_mod.MAX_STALL_COUNT == 0


@pytest.mark.skipif(not tq.REPORT_PATH.exists() or not tq.RECORDING_PATH.exists(),
                    reason="the live evaluation has not been run on this machine")
def test_the_published_report_reproduces_from_the_recording() -> None:
    """Every headline number in data/thesis_quality.json is recomputed offline, at zero calls."""
    published = json.loads(tq.REPORT_PATH.read_text(encoding="utf-8"))
    again = tq.evaluate(live=False)
    assert again["qwen_calls_made_this_run"] == 0
    for stage, keys in {
        "s5": ("arms", "ranking_changed_the_committed_answer"),
        "s6": ("after_one_round", "after_two_rounds", "verdict_changed_by_repair"),
        "s4": ("surfaced", "random", "paired"),
        "s7": ("rounds_saved", "wrong_detections", "proxy_vs_llm"),
    }.items():
        for key in keys:
            assert again[stage][key] == published[stage][key], (stage, key)
