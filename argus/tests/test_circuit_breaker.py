"""S17: the trajectory circuit-breaker, the retry policy, and the feed-sanity gate — in the desk.

Every model here is scripted; nothing reaches Qwen. The desk-level tests drive `TradingDesk.run`
end to end, because the property that matters is not that `agents/circuit.py` computes the right
answer in isolation but that **no stop escapes the decision loop as an exception** and every stop is
in the decision's record with its reason.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from random import Random
from typing import Any

import pytest

import argus.agents.desk as desk_mod
from argus.agents.circuit import (
    DEFAULT_THRESHOLDS,
    FaultKind,
    RetryPolicy,
    Step,
    Trip,
    assess,
    call_with_retry,
    classify,
    equivalent,
    nudge_for,
    trips_per_100,
)
from argus.agents.debate import Ending
from argus.agents.desk import ConstitutionPolicy, TradingDesk
from argus.desk.feed_sanity import (
    check_claim,
    check_panel_line,
    redact,
    screen_claims,
    screen_evidence,
)
from argus.risk.hedgeability import HedgeabilitySurface
from argus.truth.clocks import SessionPhase, SessionState
from argus.truth.evidence import Evidence

AT = datetime(2026, 9, 25, 15, 0, tzinfo=UTC)
MKT = "NVDAUSDT last 200, 24h change 0.01379, quoted spread 0.44bps, 24h base volume 60105.54"


# --- the retry policy ----------------------------------------------------------------------------


class TestClassify:
    def test_timeouts_and_connections_are_retryable(self) -> None:
        assert classify(TimeoutError("slow")) is FaultKind.TIMEOUT
        assert classify(ConnectionError("reset")) is FaultKind.TRANSIENT
        assert FaultKind.TIMEOUT.retryable and FaultKind.TRANSIENT.retryable

    def test_status_codes_decide_like_langgraph_default_retry_on(self) -> None:
        class Http(Exception):
            def __init__(self, code: int) -> None:
                super().__init__(f"http {code}")
                self.status_code = code

        assert classify(Http(503)) is FaultKind.TRANSIENT
        assert classify(Http(429)) is FaultKind.TRANSIENT
        assert classify(Http(401)) is FaultKind.PERMANENT
        assert not FaultKind.PERMANENT.retryable

    def test_a_malformed_answer_is_not_retried_at_this_layer(self) -> None:
        assert classify(ValueError("bad json")) is FaultKind.INVALID_OUTPUT
        assert not FaultKind.INVALID_OUTPUT.retryable


class TestRetryPolicy:
    def test_interval_is_langgraphs_formula_without_jitter(self) -> None:
        policy = RetryPolicy(jitter=False, max_interval=3.0)
        assert [policy.interval(n, Random(0)) for n in (1, 2, 3, 4)] == [0.5, 1.0, 2.0, 3.0]

    def test_jitter_adds_at_most_one_second(self) -> None:
        got = RetryPolicy().interval(1, Random(7))
        assert 0.5 <= got <= 1.5

    def test_transient_then_success_is_retried_and_succeeds(self) -> None:
        waits: list[float] = []
        calls = {"n": 0}

        def flaky() -> str:
            calls["n"] += 1
            if calls["n"] < 3:
                raise ConnectionError("reset by peer")
            return "ok"

        out = call_with_retry(flaky, name="venue", policy=RetryPolicy(jitter=False),
                              sleep=waits.append)
        assert out.ok and out.value == "ok"
        assert len(out.attempts) == 3 and waits == [0.5, 1.0]
        assert len(out.render()) == 2  # both failed attempts are visible, not silent

    def test_a_permanent_fault_is_never_resent(self) -> None:
        sent = {"n": 0}

        def rejected() -> str:
            sent["n"] += 1
            raise PermissionError("401 unauthorised")

        out = call_with_retry(rejected, name="venue.place_order", policy=RetryPolicy(),
                              sleep=lambda _s: None,
                              classifier=lambda _e: FaultKind.PERMANENT)
        assert not out.ok and sent["n"] == 1
        assert out.gave_up is not None and "gave up on venue.place_order" in out.gave_up
        assert "not retryable" in out.gave_up

    def test_a_timeout_gives_up_by_name_and_never_raises(self) -> None:
        def hung() -> str:
            time.sleep(0.5)
            return "late"

        started = time.monotonic()
        out = call_with_retry(
            hung, name="venue.place_order",
            policy=RetryPolicy(max_attempts=2, timeout_seconds=0.05, jitter=False,
                               initial_interval=0.0),
            sleep=lambda _s: None,
        )
        assert time.monotonic() - started < 0.45  # stopped waiting; did not sit out the hang
        assert not out.ok and out.fault is FaultKind.TIMEOUT
        assert out.gave_up is not None and "attempts exhausted" in out.gave_up


# --- the trajectory ------------------------------------------------------------------------------


def _step(qty: str, *, hours: float = 0, side: str = "buy", verdict: str = "trade",
          symbol: str = "NVDAUSDT", valid: bool = True) -> Step:
    return Step(symbol=symbol, at=AT + timedelta(hours=hours), verdict=verdict, side=side,
                quantity=Decimal(qty), valid=valid)


class TestEquivalence:
    def test_same_instrument_side_and_size_within_tolerance(self) -> None:
        assert equivalent(_step("50"), _step("54"))

    def test_size_outside_tolerance_is_a_different_order(self) -> None:
        assert not equivalent(_step("50"), _step("60"))

    def test_side_and_instrument_matter(self) -> None:
        assert not equivalent(_step("50"), _step("50", side="sell"))
        assert not equivalent(_step("50"), _step("50", symbol="TSLAUSDT"))

    def test_repeated_abstention_is_never_a_loop(self) -> None:
        assert not equivalent(_step("0", verdict="no_trade"), _step("0", verdict="no_trade"))


class TestAssess:
    def test_the_third_equivalent_order_in_the_window_trips(self) -> None:
        history = [_step("50", hours=0), _step("52", hours=2)]
        reading = assess(history, _step("49", hours=4))
        assert reading.trip is Trip.EQUIVALENT_ORDERS and "3 of 3" in reading.reason

    def test_the_second_does_not(self) -> None:
        assert assess([_step("50")], _step("50", hours=1)).trip is None

    def test_orders_outside_the_window_do_not_count(self) -> None:
        history = [_step("50", hours=0), _step("50", hours=25)]
        assert assess(history, _step("50", hours=50)).trip is None

    def test_three_invalid_answers_in_a_row_trip_across_symbols(self) -> None:
        history = [
            _step("0", verdict="human_review", valid=False, symbol="TSLAUSDT"),
            _step("0", verdict="human_review", valid=False, hours=0.1, symbol="METAUSDT"),
        ]
        reading = assess(history, _step("0", verdict="human_review", valid=False, hours=0.2))
        assert reading.trip is Trip.INVALID_OUTPUTS and reading.invalid_run == 3

    def test_a_valid_answer_breaks_the_invalid_run(self) -> None:
        history = [
            _step("0", verdict="human_review", valid=False),
            _step("0", verdict="no_trade", hours=0.1),
        ]
        assert assess(history, _step("0", verdict="human_review", valid=False, hours=0.2)).trip \
            is None

    def test_the_step_budget_stops_a_runaway_caller(self) -> None:
        history = [_step("0", verdict="no_trade", hours=i / 60)
                   for i in range(DEFAULT_THRESHOLDS.max_steps)]
        reading = assess(history, _step("0", verdict="no_trade", hours=1))
        assert reading.trip is Trip.STEP_BUDGET

    def test_naive_timestamps_do_not_raise_inside_the_breaker(self) -> None:
        naive = Step.from_record({"decided_at": "2026-09-25T14:00:00", "symbol": "NVDAUSDT",
                                  "verdict": "trade", "side": "BUY", "quantity": "50"})
        assert naive is not None and naive.at.tzinfo is not None
        assert assess([naive], _step("50")).trip is None

    def test_ledger_rows_are_read_and_seals_are_skipped(self) -> None:
        seal = {"kind": "settlement_seal", "decided_at": AT.isoformat(), "verdict": "x"}
        assert Step.from_record(seal) is None
        row = {"kind": "decision", "decided_at": AT.isoformat(), "symbol": "NVDAUSDT",
               "verdict": "human_review", "side": "BUY", "quantity": "0",
               "thesis": "[unparseable verdict 'sell'; routed to human review] ..."}
        step = Step.from_record(row)
        assert step is not None and not step.valid and step.side == "buy"

    def test_the_nudge_comes_one_repeat_before_the_trip(self) -> None:
        history = [_step("50", hours=0), _step("51", hours=1)]
        nudge = nudge_for(history, symbol="NVDAUSDT", now=AT + timedelta(hours=2))
        assert nudge is not None and "will be stopped" in nudge
        assert nudge_for(history[:1], symbol="NVDAUSDT", now=AT + timedelta(hours=2)) is None

    def test_trips_per_100(self) -> None:
        assert trips_per_100(3, 200) == 1.5 and trips_per_100(0, 0) is None


# --- the feed-sanity gate ------------------------------------------------------------------------


class TestFeedSanity:
    def test_a_nan_price_is_withheld_and_the_rest_of_the_line_kept(self) -> None:
        claim = MKT.replace("last 200", "last NaN")
        found = check_claim("mkt", claim, token_price=Decimal("200"))
        assert [f.rule for f in found] == ["non_numeric"]
        out = redact(claim, found)
        assert "NaN" not in out and "quoted spread 0.44bps" in out

    def test_a_price_that_disagrees_with_the_token_price(self) -> None:
        found = check_claim("mkt", MKT.replace("last 200", "last 210"), token_price=Decimal("200"))
        assert [f.rule for f in found] == ["price_mismatch"]

    def test_the_999_percent_day(self) -> None:
        found = check_claim("mkt", MKT.replace("0.01379", "9.99346"), token_price=Decimal("200"))
        assert [f.rule for f in found] == ["implausible_change"]

    def test_a_negative_spread(self) -> None:
        found = check_claim("mkt", MKT.replace("0.44bps", "-5bps"), token_price=Decimal("200"))
        assert [f.rule for f in found] == ["negative_spread"]

    def test_the_absurd_quant_edge(self) -> None:
        line = "[panel] quant: bullish 6895bps (confidence 0.80) — continuation clears costs"
        assert [f.rule for f in check_panel_line("panel", line, move_24h=None)] == ["absurd_edge"]

    def test_a_100x_unit_error_below_the_absurd_threshold(self) -> None:
        line = ("[panel] quant: bearish 45bps (confidence 0.80) — moved -0.00009 over 24h, which "
                "is -0.9bps; but at 90bps the move is 6.31x the hurdle")
        found = check_panel_line("panel", line, move_24h=Decimal("-0.00009"))
        assert [f.rule for f in found] == ["unit_error"]

    def test_clean_figures_pass(self) -> None:
        report = screen_claims(
            [("mkt", MKT), ("vix", "VIX 15.11 (-0.56 on the day), the 33% percentile of 9,276 "
                                   "daily closes — normal regime. Implies a one-day move of about "
                                   "95bps on US equities, which is what these tokens track.")],
            token_price=Decimal("200"),
        )
        assert report.clean and "none withheld" in report.render()[0]

    def test_screen_evidence_keeps_identity(self) -> None:
        items = [Evidence(id="mkt-NVDAUSDT", claim=MKT.replace("last 200", "last NaN"),
                          source="news", available_at=AT)]
        screened, report = screen_evidence(items, token_price=Decimal("200"))
        assert screened[0].id == "mkt-NVDAUSDT" and screened[0].source == "news"
        assert "[withheld by feed-sanity: non_numeric]" in screened[0].claim
        assert report.withheld == ("mkt-NVDAUSDT",)


# --- the desk ------------------------------------------------------------------------------------


class ScriptedModel:
    """Analysts answer neutral; the Meta-PM answers from ``pm`` (a dict, or a callable)."""

    def __init__(self, pm: Any, *, analyst_magnitude: int | None = None) -> None:
        self.pm = pm
        self.analyst_magnitude = analyst_magnitude
        self.pm_prompts: list[str] = []

    def complete_json(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        if "verdict" in tuple(kwargs.get("required_keys", ())):
            self.pm_prompts.append(str(messages[-1].get("content", "")))
            answer = self.pm() if callable(self.pm) else self.pm
            return dict(answer)
        view: dict[str, Any] = {"signal": "neutral", "confidence": 0.5, "reasoning": "swept",
                                "chain": ["a", "b"]}
        if self.analyst_magnitude is not None:
            view.update(signal="bullish", magnitude_bps=self.analyst_magnitude)
        return view

    def complete(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - unused
        raise NotImplementedError


NO_TRADE = {"verdict": "NO_TRADE", "side": "buy", "quantity": "0", "confidence": 0.5,
            "thesis": "nothing clears the hurdle", "invalidation": ["a catalyst appears"]}
BUY_50 = {"verdict": "TRADE", "side": "buy", "quantity": "50", "confidence": 0.7,
          "thesis": "the contract re-rates the name", "invalidation": ["the contract is denied"]}


def _run(model: ScriptedModel, *, history: list[Any] | None = None,
         evidence: list[Evidence] | None = None, **desk_kwargs: Any) -> Any:
    desk = TradingDesk(model, **desk_kwargs)  # type: ignore[arg-type]
    return desk.run(
        symbol="NVDAUSDT",
        session=SessionState(phase=SessionPhase.RTH, as_of=AT, hours_to_next_discovery=0.0,
                             nav_age_seconds=5.0),
        token_price=Decimal("200"), position=Decimal("0"),
        evidence=evidence if evidence is not None else [
            Evidence(id="mkt-NVDAUSDT", claim=MKT, source="news", available_at=AT),
            Evidence(id="e1", claim="NVDA announced a datacentre contract", source="news",
                     available_at=AT, credibility=0.9),
        ],
        hedges=HedgeabilitySurface(candidates=()), decision_id="test-circuit",
        constitution=ConstitutionPolicy(), history=history or [],
    )


@dataclass
class Row:
    """The attributes of a ledger `Entry` that memory recall and the breaker both read."""

    seq: int
    decided_at: str
    symbol: str
    verdict: str
    side: str
    quantity: str
    session_phase: str
    stated_confidence: float
    thesis: str
    settled_at: str | None = None
    net_pnl: str | None = None
    direction_correct: bool | None = None
    counterfactual_move_bps: str | None = None


def _ledger_row(hours_ago: float, *, verdict: str = "trade", quantity: str = "50",
                thesis: str = "prior") -> Row:
    return Row(seq=int(hours_ago * 10), decided_at=(AT - timedelta(hours=hours_ago)).isoformat(),
               symbol="NVDAUSDT", verdict=verdict, side="BUY", quantity=quantity,
               session_phase="rth", stated_confidence=0.6, thesis=thesis)


class TestTheDeskLoop:
    def test_a_clean_decision_records_no_trip_and_its_deliberation(self) -> None:
        run = _run(ScriptedModel(NO_TRADE))
        assert run.circuit == {"trips": [], "nudges": [], "faults": []}
        assert run.deliberation is not None and run.deliberation["calls"] == 1
        assert run.deliberation["extra_calls"] == 0
        assert any(n.startswith("[deliberation] 1 model call") for n in run.notes)
        assert run.as_dict()["deliberation"]["calls"] == 1

    def test_a_hung_model_degrades_to_a_safe_no_op_not_an_exception(self) -> None:
        def hang() -> dict[str, Any]:
            time.sleep(1.0)
            return BUY_50

        run = _run(ScriptedModel(hang),
                   pm_policy=RetryPolicy(max_attempts=1, timeout_seconds=0.1))
        intent = run.proof.llm_original_intent
        assert intent.verdict.value == "no_trade" and intent.quantity == 0
        assert intent.thesis.startswith("[circuit] no model decision")
        assert run.order is None
        assert run.circuit["trips"][0]["trip"] == "call_gave_up"
        assert "timeout" in run.circuit["trips"][0]["reason"]
        assert any("TRIP call_gave_up" in n for n in run.notes)

    def test_a_failing_model_degrades_too(self) -> None:
        def boom() -> dict[str, Any]:
            raise ConnectionError("endpoint unreachable")

        run = _run(ScriptedModel(boom))
        assert run.proof.llm_original_intent.verdict.value == "no_trade"
        assert run.circuit["faults"][0]["fault"] == "transient"

    def test_malformed_output_after_two_others_trips_invalid_outputs(self) -> None:
        bad = "[unparseable verdict 'sell'; routed to human review] x"
        history = [_ledger_row(2, verdict="human_review", quantity="0", thesis=bad),
                   _ledger_row(1, verdict="human_review", quantity="0", thesis=bad)]
        run = _run(ScriptedModel({**NO_TRADE, "verdict": "sell"}), history=history)
        assert run.proof.llm_original_intent.verdict.value == "human_review"
        assert [t["trip"] for t in run.circuit["trips"]] == ["invalid_outputs"]
        assert run.order is None

    def test_the_third_equivalent_order_is_stopped_before_the_venue(self) -> None:
        history = [_ledger_row(4, quantity="52"), _ledger_row(2, quantity="49")]
        run = _run(ScriptedModel(BUY_50), history=history)
        assert [t["trip"] for t in run.circuit["trips"]] == ["equivalent_orders"]
        assert run.order is None
        assert run.proof.llm_original_intent.quantity == 50  # the model's proposal is kept
        assert any("TRIP equivalent_orders" in n for n in run.notes)
        assert run.circuit["nudges"]  # the model was warned one repeat earlier, before deciding

    def test_the_nudge_reaches_the_model_prompt(self) -> None:
        model = ScriptedModel(NO_TRADE)
        _run(model, history=[_ledger_row(4), _ledger_row(2)])
        assert "will be stopped by the circuit breaker" in model.pm_prompts[0]

    def test_a_nan_price_never_reaches_the_model(self) -> None:
        model = ScriptedModel(NO_TRADE)
        evidence = [Evidence(id="mkt-NVDAUSDT", claim=MKT.replace("last 200", "last NaN"),
                             source="news", available_at=AT)]
        run = _run(model, evidence=evidence)
        assert "last NaN" not in model.pm_prompts[0]
        assert "[withheld by feed-sanity: non_numeric]" in model.pm_prompts[0]
        assert run.feed_sanity["evidence"]["withheld"] == ["mkt-NVDAUSDT"]

    def test_an_absurd_analyst_edge_is_dropped_before_consensus(self) -> None:
        model = ScriptedModel(NO_TRADE, analyst_magnitude=6895)
        run = _run(model)
        assert "6895bps" not in model.pm_prompts[0]
        assert all(v.magnitude_bps < 200 for v in run.panel.views)
        assert "absurd_edge" in run.feed_sanity["analysts"]["rules_fired"]


class TestAStalledDebateIsNotConverged:
    def test_stalled_reaches_escalation_as_not_converged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict[str, Any] = {}

        class Stalled:
            ending = Ending.STALLED
            cost_bps = Decimal("0")
            positions = ()

            def render(self) -> str:
                return "[debate] stalled"

            def as_dict(self) -> dict[str, Any]:
                return {"ending": "stalled"}

        from argus.decision.escalation import assess as real

        def spy(intent: Any, signals: Any) -> Any:
            seen["converged"] = signals.debate_converged
            return real(intent, signals)

        monkeypatch.setattr(desk_mod, "hold_debate", lambda **_k: Stalled())
        monkeypatch.setattr(desk_mod, "assess_escalation", spy)
        _run(ScriptedModel(NO_TRADE))
        assert seen["converged"] is False


class TestTheBreakerChangesTheDecision:
    def test_without_the_history_the_same_answer_reaches_the_venue(self) -> None:
        """The control: a breaker that only agrees with the decision it checks is a comment."""
        run = _run(ScriptedModel(BUY_50))
        assert run.circuit["trips"] == []
        assert run.order is not None and run.order.quantity == 50


class TestTheMeasurements:
    """The two artefacts are recomputed here from their own inputs, so a regression fails a test."""

    def test_every_recorded_corruption_is_caught_and_no_clean_input_is_flagged(self) -> None:
        import json

        from argus.eval.feed_sanity_gate import acceptance, clean_negatives
        from argus.eval.feedbugged import REPORT_PATH
        from argus.eval.perturbations import load_snapshots

        snapshots = load_snapshots()
        got = acceptance({s.id: s for s in snapshots},
                         json.loads(REPORT_PATH.read_text(encoding="utf-8")))
        assert got["detection"]["tp"] == 12 and got["detection"]["fn"] == 0
        assert got["detection"]["fp"] == 0
        assert clean_negatives(snapshots)["false_flags"] == 0

    def test_the_recorded_ledger_replays_with_no_trip_and_every_fault_is_caught(self) -> None:
        from argus.eval.circuit_replay import injections, load_trajectory, replay

        steps = load_trajectory()
        assert len(steps) > 600
        assert replay(steps) == []
        rows = injections(steps)
        assert all(r["caught_with_right_reason"] for r in rows), [
            (r["fault"], r["expected"], r["got"]) for r in rows
        ]


class TestASpentBudgetStillEndsTheCycle:
    def test_budget_exhausted_propagates_rather_than_recording_a_no_op(self) -> None:
        """The runner's token budget is its own control; the breaker must not absorb it."""
        from argus.llm.qwen import BudgetExhausted

        def spent() -> dict[str, Any]:
            raise BudgetExhausted("token budget spent")

        with pytest.raises(BudgetExhausted):
            _run(ScriptedModel(spent))
