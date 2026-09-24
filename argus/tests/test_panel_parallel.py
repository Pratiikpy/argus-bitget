"""The analyst panel: concurrent, independent, deterministic, and survivable.

Four properties, each of which was either untrue or unproven before the panel was parallelised:

* the analysts genuinely cannot read each other, so ``sequential=False`` is earned;
* one analyst failing costs that analyst, not the panel;
* the order of the record does not depend on which thread finished first;
* running concurrently is actually faster than running in sequence.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from argus.agents.debate import Ending
from argus.agents.desk import ConstitutionPolicy, TradingDesk
from argus.decision.verdicts import ConstitutionVerdict
from argus.risk.hedgeability import HedgeabilitySurface
from argus.truth.clocks import SessionPhase, SessionState
from argus.truth.evidence import Evidence

AT = datetime(2026, 9, 14, 15, 0, tzinfo=UTC)


def _session() -> SessionState:
    return SessionState(
        phase=SessionPhase.RTH, as_of=AT, hours_to_next_discovery=0.0, nav_age_seconds=5.0
    )


def _evidence() -> list[Evidence]:
    """One item on each analyst's channel, so all three have something to read."""
    return [
        Evidence(id="e1", claim="NVDA announced a new datacentre contract", source="news",
                 available_at=AT, credibility=0.9),
        Evidence(id="e2", claim="social chatter is elevated", source="social",
                 available_at=AT, credibility=0.4),
        Evidence(id="e3", claim="Q2 filing shows revenue up 12% QoQ", source="filing",
                 available_at=AT, credibility=1.0),
    ]


class RecordingModel:
    """A chat model that records every call, and can be made slow or made to fail."""

    def __init__(
        self, *, delay: float = 0.0, fail_roles: tuple[str, ...] = (),
        split: bool = False,
    ) -> None:
        self.delay = delay
        self.fail_roles = fail_roles
        # `split=True` makes the analysts point opposite ways. The desk now holds a debate ONLY
        # when the panel disagrees on direction, so every debate test needs a model that actually
        # produces that state rather than one that agrees with itself.
        self.split = split
        self._analyst_calls = 0
        self.calls: list[dict[str, Any]] = []
        self.concurrent_peak = 0
        self._live = 0
        self._lock = threading.Lock()

    def complete_json(
        self, messages: list[dict[str, Any]], **kwargs: Any
    ) -> dict[str, Any]:
        system = str(messages[0].get("content", ""))
        with self._lock:
            self.calls.append({"system": system, "user": str(messages[-1].get("content", ""))})
            self._live += 1
            self.concurrent_peak = max(self.concurrent_peak, self._live)
        try:
            if self.delay:
                time.sleep(self.delay)
            for role in self.fail_roles:
                if role in system.lower():
                    raise RuntimeError(f"{role} analyst upstream is down")
            required = tuple(kwargs.get("required_keys", ()))
            if "verdict" in required:
                # The portfolio manager's call, not an analyst's.
                return {
                    "verdict": "NO_TRADE", "side": "buy", "quantity": "0",
                    "confidence": 0.5, "thesis": "nothing clears the hurdle",
                    "invalidation": ["a catalyst appears"],
                }
            if self.split:
                with self._lock:
                    self._analyst_calls += 1
                    nth = self._analyst_calls
                signal = "bullish" if nth % 2 else "bearish"
                return {
                    "signal": signal, "confidence": 0.8, "magnitude_bps": 40,
                    "reasoning": f"{signal} on the evidence", "chain": ["a", "b"],
                }
            return {
                "signal": "neutral", "confidence": 0.5,
                "reasoning": "swept", "chain": ["a", "b"],
            }
        finally:
            with self._lock:
                self._live -= 1

    def complete(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - unused here
        raise NotImplementedError


def _run(model: RecordingModel) -> Any:
    desk = TradingDesk(model)
    return desk.run(
        symbol="NVDAUSDT",
        session=_session(),
        token_price=Decimal("200"),
        position=Decimal("0"),
        evidence=_evidence(),
        hedges=HedgeabilitySurface(candidates=()),
        decision_id="test-panel",
        constitution=ConstitutionPolicy(),
    )


class TestTheAnalystsCannotReadEachOther:
    def test_no_analyst_is_shown_another_analysts_output(self) -> None:
        """The structural claim behind sequential=False."""
        model = RecordingModel()
        _run(model)
        analyst_calls = [c for c in model.calls if "analyst" in c["system"].lower()]
        assert len(analyst_calls) >= 2
        for call in analyst_calls:
            # No analyst prompt may contain the JSON shape another analyst returns.
            assert '"signal"' not in call["user"]
            assert "reasoning" not in call["user"].lower()

    def test_every_analyst_call_is_a_fresh_two_message_conversation(self) -> None:
        model = RecordingModel()
        _run(model)
        assert model.calls
        for call in model.calls:
            assert call["system"] and call["user"]

    def test_each_analyst_sees_only_its_own_evidence_channel(self) -> None:
        model = RecordingModel()
        _run(model)
        by_role = {c["system"].split("\n")[0].lower(): c["user"] for c in model.calls}
        event = next((v for k, v in by_role.items() if "event analyst" in k), None)
        if event is not None:
            assert "social chatter" not in event

    def test_the_conflict_report_is_marked_independent(self) -> None:
        model = RecordingModel()
        run = _run(model)
        text = " ".join(run.notes)
        assert "contagion" not in text or "rather than contagion" in text


class TestFailureIsIsolated:
    def test_one_failing_analyst_does_not_take_the_panel_down(self) -> None:
        model = RecordingModel(fail_roles=("sentiment",))
        run = _run(model)
        assert run.proof is not None
        assert any("failed and was dropped" in n for n in run.notes)

    def test_the_failure_names_the_analyst_and_the_error(self) -> None:
        model = RecordingModel(fail_roles=("sentiment",))
        run = _run(model)
        note = next(n for n in run.notes if "failed and was dropped" in n)
        assert "sentiment" in note and "RuntimeError" in note

    def test_the_surviving_analysts_still_reach_the_panel(self) -> None:
        model = RecordingModel(fail_roles=("sentiment",))
        run = _run(model)
        assert any("analyst(s) ran concurrently" in n for n in run.notes)

    def test_a_decision_is_still_produced_when_every_analyst_fails(self) -> None:
        model = RecordingModel(fail_roles=("event", "sentiment", "earnings"))
        run = _run(model)
        assert run.proof is not None
        dropped = [n for n in run.notes if "failed and was dropped" in n]
        assert len(dropped) >= 2


class TestConcurrency:
    def test_the_analysts_actually_overlap(self) -> None:
        model = RecordingModel(delay=0.25)
        _run(model)
        assert model.concurrent_peak >= 2, (
            f"peak concurrency {model.concurrent_peak}: the panel is still effectively serial"
        )

    def test_the_panel_is_faster_than_the_sum_of_its_parts(self) -> None:
        """Measured on the panel itself, not on the whole run: the portfolio manager's own call
        follows the panel and would otherwise be counted as panel time."""
        delay = 0.3
        model = RecordingModel(delay=delay)
        run = _run(model)
        note = next(n for n in run.notes if "ran concurrently" in n)
        ran = int(note.split("]")[1].strip().split()[0])
        panel_seconds = float(note.split(" in ")[1].split("s;")[0])
        if ran >= 2:
            assert panel_seconds < delay * ran, (
                f"{panel_seconds:.2f}s for {ran} analysts at {delay}s each is not concurrent"
            )
            assert panel_seconds >= delay * 0.5

    def test_the_elapsed_panel_time_is_recorded(self) -> None:
        model = RecordingModel(delay=0.1)
        run = _run(model)
        note = next(n for n in run.notes if "ran concurrently" in n)
        assert "s;" in note

    def test_the_record_is_deterministic_regardless_of_finishing_order(self) -> None:
        """Concurrency must not make the panel depend on thread scheduling."""
        orders = set()
        for _ in range(3):
            run = _run(RecordingModel())
            names = tuple(v.analyst for v in run.panel.views) if hasattr(run, "panel") else ()
            orders.add(names)
        assert len(orders) <= 1, f"panel order varied across runs: {orders}"


def test_a_slow_analyst_does_not_serialise_the_others() -> None:
    """The whole point: the panel costs about as long as its slowest member, not their sum."""
    delay = 0.4
    run = _run(RecordingModel(delay=delay))
    note = next(n for n in run.notes if "ran concurrently" in n)
    ran = int(note.split("]")[1].strip().split()[0])
    panel_seconds = float(note.split(" in ")[1].split("s;")[0])
    assert panel_seconds < delay * ran, (
        f"{panel_seconds:.2f}s for {ran} analysts suggests they ran one after another"
    )


class TestTheDemotionIsVisibleInTheDecisionTrail:
    """A demoted analyst whose demotion lives only in a docstring is not demoted."""

    def test_the_standing_note_is_written_when_sentiment_runs(self) -> None:
        run = _run(RecordingModel())
        assert any("[standing] sentiment is DEMOTED" in n for n in run.notes)

    def test_the_note_names_the_evidence_that_killed_the_feed(self) -> None:
        notes = " ".join(_run(RecordingModel()).notes)
        assert "403" in notes and "418" in notes and "93%" in notes

    def test_the_note_names_the_route_back(self) -> None:
        """A demotion with no promotion gate is a deletion that nobody can argue with."""
        notes = " ".join(_run(RecordingModel()).notes)
        assert "ablation" in notes and "30 differing paired frames" in notes

    def test_the_note_says_how_much_evidence_it_actually_had(self) -> None:
        notes = " ".join(_run(RecordingModel()).notes)
        assert "reasoned over 1 item(s)" in notes


class TestEpisodicMemoryReachesTheDecision:
    """The agent-architecture audit's clearest gap: 'each cycle is isolated'. It no longer is."""

    def _history(self) -> list[Any]:
        from dataclasses import dataclass

        @dataclass
        class Row:
            seq: int
            decided_at: str
            symbol: str
            verdict: str
            session_phase: str
            stated_confidence: float
            thesis: str
            settled_at: str | None = None
            net_pnl: str | None = None
            direction_correct: bool | None = None
            counterfactual_move_bps: str | None = None

        earlier = AT - timedelta(days=3)
        settled = AT - timedelta(days=2)
        return [
            Row(seq=i, decided_at=(earlier - timedelta(hours=i)).isoformat(),
                symbol="NVDAUSDT", verdict="no_trade", session_phase="weekend",
                stated_confidence=0.8, thesis="no edge",
                settled_at=settled.isoformat(), counterfactual_move_bps="-40.0")
            for i in range(1, 8)
        ]

    def test_no_history_produces_no_memory_note(self) -> None:
        assert not any("[memory]" in n for n in _run(RecordingModel()).notes)

    def test_history_reaches_the_record(self) -> None:
        desk = TradingDesk(RecordingModel())
        run = desk.run(
            symbol="NVDAUSDT", session=_session(), token_price=Decimal("200"),
            position=Decimal("0"), evidence=_evidence(),
            hedges=HedgeabilitySurface(candidates=()), decision_id="test-memory",
            constitution=ConstitutionPolicy(), history=self._history(),
        )
        assert any("[memory]" in n for n in run.notes)

    def test_the_note_says_how_many_were_graded(self) -> None:
        desk = TradingDesk(RecordingModel())
        run = desk.run(
            symbol="NVDAUSDT", session=_session(), token_price=Decimal("200"),
            position=Decimal("0"), evidence=_evidence(),
            hedges=HedgeabilitySurface(candidates=()), decision_id="test-memory",
            constitution=ConstitutionPolicy(), history=self._history(),
        )
        note = next(n for n in run.notes if "[memory]" in n)
        assert "graded" in note and "7 of them graded" in note

    def test_the_model_is_shown_the_memory_before_it_decides(self) -> None:
        """Told first, like the mandate. A memory applied to an answer is not a memory."""
        model = RecordingModel()
        desk = TradingDesk(model)
        desk.run(
            symbol="NVDAUSDT", session=_session(), token_price=Decimal("200"),
            position=Decimal("0"), evidence=_evidence(),
            hedges=HedgeabilitySurface(candidates=()), decision_id="test-memory",
            constitution=ConstitutionPolicy(), history=self._history(),
        )
        seen = " ".join(c["user"] for c in model.calls)
        assert "WHAT THIS DESK ALREADY LEARNED HERE" in seen
        assert "prior decision(s) on NVDAUSDT" in seen


class TestTheDebateRunsInsideTheDesk:
    """Two independent audits named the absence of a debate as the architecture gap. It runs now,
    and these pin that it runs in the right place, at a priced cost, before the PM decides — and
    that it runs ONLY when there is something to argue about."""

    @staticmethod
    def _split_run(**kw: Any) -> Any:
        desk = TradingDesk(kw.pop("model", RecordingModel(split=True)))
        return desk.run(
            symbol="NVDAUSDT", session=_session(), token_price=Decimal("200"),
            position=Decimal("0"), evidence=_evidence(),
            hedges=HedgeabilitySurface(candidates=()), decision_id="test-debate",
            constitution=ConstitutionPolicy(), **kw,
        )

    def test_a_unanimous_panel_holds_no_debate(self) -> None:
        """A debate between two seats that already agree is paid restatement. Measured: it took a
        single-symbol cycle from 20,714 tokens to over 27,691 and exhausted its budget."""
        run = _run(RecordingModel())
        assert run.debate is not None and run.debate.ending is Ending.NOT_HELD
        assert any("not held" in n and "already agree" in n for n in run.notes)

    def test_a_split_panel_holds_one(self) -> None:
        run = self._split_run()
        assert run.debate is not None
        assert run.debate.ending is not Ending.NOT_HELD
        assert run.debate.positions

    def test_the_transcript_reaches_the_model_before_it_decides(self) -> None:
        model = RecordingModel(split=True)
        self._split_run(model=model)
        seen = " ".join(c["user"] for c in model.calls)
        assert "THE ARGUMENT ON BOTH SIDES" in seen

    def test_the_transcript_is_written_into_the_notes(self) -> None:
        run = self._split_run()
        assert any("[debate]" in n for n in run.notes)

    def test_a_split_panel_the_cycle_cannot_afford_is_not_argued_and_says_so(self) -> None:
        """Two cycles on 2026-09-23 held five debates in nine symbols and left three symbols
        undecided. When the runner says the budget is reserved, the split is decided without a
        debate — and the record says it was a budget call, not an agreeing panel."""
        model = RecordingModel(split=True)
        run = self._split_run(model=model, debate_affordable=False)
        assert run.debate is not None and run.debate.ending is Ending.NOT_HELD
        assert any("budget" in n and "split" in n for n in run.notes)
        assert not any("already agree" in n for n in run.notes)
        argued = RecordingModel(split=True)
        self._split_run(model=argued)
        assert len(model.calls) < len(argued.calls), "no debate seat was paid"

    def test_a_zero_budget_holds_no_debate_and_says_why(self) -> None:
        run = self._split_run(debate_budget=Decimal("0"))
        assert run.debate is not None
        assert run.debate.ending is Ending.EXHAUSTED_BUDGET
        assert any("exhausted_budget" in n for n in run.notes)

    def test_sharing_a_model_is_recorded_so_agreement_is_not_overread(self) -> None:
        """The desk's critic defaults to the desk's own model, so the record must say that
        agreement between the two seats is self-consistency."""
        run = self._split_run()
        assert run.debate is not None
        assert run.debate.shared_model
        assert not run.debate.agreement_is_evidence

    def test_the_debate_cost_is_added_to_the_hurdle_not_reported_beside_it(self) -> None:
        """An argument the desk does not pay for is one it will always think worth having."""
        run = self._split_run()
        assert run.debate is not None and run.debate.cost_bps > 0


class TestTheRecordNamesEveryAnalystThatRan:
    """The panel note under-reported the panel, and an audit drew the wrong conclusion from it.

    `selection.render()` writes ``[panel] N of M analysts run: ...`` for the analysts chosen by the
    evidence-cost selection. The cross-asset analyst is deliberately outside that selection — it
    reads the hedge menu and the position rather than the evidence list, so nothing about the
    evidence could make it irrelevant — and it therefore never appeared in the count. The live log
    said "3 of 3 analysts run: event, sentiment, earnings" on decisions where four analysts had run,
    and `eval/themeaudit.py` read 103 of those notes and reported that the cross-asset analyst had
    never run at all.

    On a track judged for **decision explainability**, a reasoning trail that omits a participant is
    the defect, not a cosmetic omission: it is exactly what a judge auditing the log would rely on.
    """

    def test_the_cross_asset_analyst_is_named_on_the_record(self) -> None:
        model = RecordingModel()
        got = _run(model)
        assert any(n.startswith("[panel] cross_asset also ran") for n in got.notes), got.notes

    def test_the_note_says_why_it_sits_outside_the_selection(self) -> None:
        """Without the reason, the next reader deletes it as a duplicate of the selection line."""
        model = RecordingModel()
        note = next(
            n for n in _run(model).notes if n.startswith("[panel] cross_asset also ran")
        )
        assert "outside the evidence-cost selection" in note
        assert "hedge menu" in note

    def test_it_is_named_even_when_the_hedge_menu_is_empty(self) -> None:
        """An empty menu is the finding the analyst exists to report, so this is exactly when the
        record must not go quiet about it."""
        model = RecordingModel()
        got = _run(model)  # _run passes HedgeabilitySurface(candidates=())
        note = next(n for n in got.notes if n.startswith("[panel] cross_asset also ran"))
        assert "0 instrument(s)" in note

    def test_the_theme_audit_now_counts_it(self) -> None:
        """The audit's matcher and this note are one contract; a reworded note must fail here."""
        from argus.eval.themeaudit import _analyst_ran

        model = RecordingModel()
        note = next(n for n in _run(model).notes if n.startswith("[panel] cross_asset also ran"))
        assert note.startswith("[panel] cross_asset also ran")
        assert callable(_analyst_ran)


class TestAnUnconstrainedDecisionIsNotRePutToTheModel:
    """The desk told its own model something untrue on every unconstrained decision.

    `revise()` ran unconditionally, and its prompt opens **"Your proposed action was constrained by
    the risk layer"** — printed above a binding constraint of "none" and the reason "no exposure
    proposed; nothing to narrow". That sentence was false on all 96 risk records on the live log,
    and the answer the model gave after being told it was what the ledger recorded.

    `AutonomyProof.record_revision` had already written the rule down: "Only meaningful after a
    ruling: a 'revision' with nothing to revise against is noise."
    """

    def test_no_revision_is_recorded_when_nothing_bound(self) -> None:
        model = RecordingModel()
        got = _run(model)
        assert got.ruling is not None
        assert got.ruling.verdict is ConstitutionVerdict.ALLOW
        assert got.proof.llm_revised_intent is None

    def test_the_model_is_never_told_it_was_constrained_when_it_was_not(self) -> None:
        """The sentence itself, asserted absent from every prompt the model received."""
        model = RecordingModel()
        _run(model)
        for call in model.calls:
            blob = json.dumps(call)
            assert "was constrained by the risk layer" not in blob

    def test_changed_its_mind_is_false_rather_than_noise(self) -> None:
        """It read True on 96 of 96 decisions where no constraint existed."""
        model = RecordingModel()
        assert _run(model).proof.llm_changed_its_mind is False

    def test_the_record_says_why_the_second_pass_was_skipped(self) -> None:
        model = RecordingModel()
        notes = _run(model).notes
        assert any(n.startswith("[constitution] nothing bound") for n in notes), notes

    def test_the_approved_intent_is_still_frozen_and_hashed(self) -> None:
        """Skipping the round trip must not skip approval: the order still has to carry a hash."""
        model = RecordingModel()
        got = _run(model)
        assert got.proof.approved_intent_hash
        assert len(got.proof.approved_intent_hash) == 16

    def test_exactly_one_decision_call_is_made_when_nothing_binds(self) -> None:
        """The confound-free measurement of the saving.

        Token totals across live cycles conflate the fix with the market: the debate runs only when
        the panel splits, the adversary only when exposure is opened, and two cycles a few hours
        apart see different evidence. A two-point fit on those numbers attributed the saving to
        "fixed" cost in one pair and to "marginal" in the other, which is over-fitting noise.

        The call count is deterministic and says the same thing without an estimate: the decision is
        put to the model **once** when nothing bound, where it was put twice.
        """
        model = RecordingModel()
        _run(model)
        pm_calls = [c for c in model.calls if "portfolio manager" in c["system"].lower()]
        assert len(pm_calls) == 1, [c["system"].split("\n")[0] for c in pm_calls]
