"""`decision/pause.py` and the desk's pause/resume: the typed pair, the store, the kill property.

The kill-and-resume property is tested with a real child process that is killed while it waits
(`eval/pausedrill.kill_while_paused`), not simulated by dropping an object. Every model here is
scripted; nothing calls Qwen.
"""

from __future__ import annotations

import base64
import io
import json
import os
import pickle
import subprocess
import sys
from contextlib import redirect_stdout
from dataclasses import replace
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from argus.agents.desk import (
    ConstitutionPolicy,
    DeskContinuation,
    TradingDesk,
    resume_signature,
)
from argus.decision.escalation import (
    Escalation,
    EscalationError,
    Trigger,
    takeover_rate,
)
from argus.decision.pause import (
    AlreadyResolved,
    HumanAction,
    HumanResponse,
    PauseError,
    PauseExpired,
    PauseIntegrityError,
    PauseRequest,
    PauseSignatureError,
    PauseStore,
    ResponseMismatch,
    check_response,
    main,
    resolve_intent,
    restricted_loads,
    state_hash,
    summarise,
)
from argus.decision.verdicts import Intent, Side, Verdict
from argus.eval.pausedrill import (
    AT,
    Scenario,
    ScriptedModel,
    answer_for,
    by_key,
    comparable,
    drill_one,
    run_paused_half,
)
from argus.proof.autonomy import hash_intent
from argus.risk.hedgeability import HedgeabilitySurface
from argus.truth.clocks import SessionPhase, SessionState
from argus.truth.evidence import Evidence


@pytest.fixture(autouse=True)
def _no_qwen(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BITGET_QWEN_API_KEY", raising=False)


HALTED_APPROVE = by_key("underlying_halted/approve")
HALTED_MODIFY = by_key("underlying_halted/modify_size")
HALTED_REJECT = by_key("underlying_halted/reject")


def _proposal(quantity: str = "60", side: Side = Side.BUY) -> Intent:
    return Intent(
        symbol="NVDAUSDT", side=side, quantity=Decimal(quantity), verdict=Verdict.TRADE,
        stated_confidence=0.9, thesis="guidance beat", invalidation=("guidance lower",),
    )


def _request(quantity: str = "60") -> PauseRequest:
    proposal = _proposal(quantity)
    return PauseRequest(
        request_id="d-1-p0", decision_id="d-1", sequence=0, symbol="NVDAUSDT", as_of=AT,
        expires_at=AT + timedelta(minutes=30), proposal=proposal,
        proposal_hash=hash_intent(proposal),
        escalation=Escalation(triggers=(Trigger.UNDERLYING_HALTED,), detail=("LUDP",)),
        state_hash="0" * 64, pipeline_signature=resume_signature(),
    )


def _response(
    action: HumanAction, quantity: str | None = None, *, request_id: str = "d-1-p0",
    reviewer: str = "reviewer-1", minutes: int = 2,
) -> HumanResponse:
    return HumanResponse(
        request_id=request_id, action=action, reviewer=reviewer,
        answered_at=AT + timedelta(minutes=minutes),
        quantity=None if quantity is None else Decimal(quantity),
    )


def _paused(tmp_path: Path, scenario: Scenario = HALTED_APPROVE, decision_id: str = "d-1") -> Any:
    store = PauseStore(tmp_path)
    run, _ = run_paused_half(scenario, store, decision_id=decision_id)
    return store, run


# --- the escalation record round trip -------------------------------------------------------------


class TestEscalationRecord:
    def test_round_trips(self) -> None:
        esc = Escalation(
            triggers=(Trigger.UNDERLYING_HALTED, Trigger.UNGROUNDED_FIGURE), detail=("a", "b"),
        )
        assert Escalation.from_dict(esc.as_dict()) == esc

    def test_unpaired_detail_is_refused(self) -> None:
        with pytest.raises(EscalationError):
            Escalation.from_dict({"triggers": ["underlying_halted"], "detail": []})


# --- the typed pair -------------------------------------------------------------------------------


class TestTypedPair:
    def test_each_valid_answer_is_accepted(self) -> None:
        request = _request()
        check_response(request, _response(HumanAction.APPROVE))
        check_response(request, _response(HumanAction.REJECT))
        check_response(request, _response(HumanAction.MODIFY_SIZE, "25"))
        check_response(request, _response(HumanAction.MODIFY_SIZE, "60"))

    @pytest.mark.parametrize(
        ("response", "fragment"),
        [
            (_response(HumanAction.APPROVE, request_id="other-p0"), "names"),
            (_response(HumanAction.APPROVE, "10"), "takes no quantity"),
            (_response(HumanAction.REJECT, "0"), "takes no quantity"),
            (_response(HumanAction.MODIFY_SIZE), "needs a quantity"),
            (_response(HumanAction.MODIFY_SIZE, "0"), "positive"),
            (_response(HumanAction.MODIFY_SIZE, "-5"), "positive"),
            (_response(HumanAction.MODIFY_SIZE, "61"), "may only reduce"),
            (_response(HumanAction.MODIFY_SIZE, "NaN"), "positive"),
            (_response(HumanAction.APPROVE, reviewer="  "), "reviewer"),
        ],
    )
    def test_an_answer_that_does_not_fit_is_refused(
        self, response: HumanResponse, fragment: str
    ) -> None:
        with pytest.raises(ResponseMismatch, match=fragment):
            check_response(_request(), response)

    def test_an_action_the_request_does_not_allow_is_refused(self) -> None:
        request = replace(_request(), allowed_actions=(HumanAction.REJECT,))
        with pytest.raises(ResponseMismatch, match="not an answer"):
            check_response(request, _response(HumanAction.APPROVE))

    def test_request_round_trips_and_detects_an_edited_proposal(self) -> None:
        request = _request()
        assert PauseRequest.from_dict(request.as_dict()) == request
        blob = request.as_dict()
        blob["proposal"]["quantity"] = "6000"  # a raised ceiling, written into the file
        with pytest.raises(PauseIntegrityError, match="does not match"):
            PauseRequest.from_dict(blob)

    def test_response_schema_names_the_ceiling(self) -> None:
        schema = _request("60").response_schema()
        assert schema["properties"]["request_id"]["const"] == "d-1-p0"
        assert "60" in schema["properties"]["quantity"]["description"]
        assert set(schema["properties"]["action"]["enum"]) == {a.value for a in HumanAction}


class TestResolveIntent:
    def test_the_three_answers(self) -> None:
        p = _proposal("60")
        assert resolve_intent(p, _response(HumanAction.APPROVE)) == p
        modified = resolve_intent(p, _response(HumanAction.MODIFY_SIZE, "25"))
        assert (modified.quantity, modified.verdict, modified.side) == (
            Decimal("25"), Verdict.TRADE, Side.BUY,
        )
        rejected = resolve_intent(p, _response(HumanAction.REJECT))
        assert (rejected.quantity, rejected.verdict) == (Decimal("0"), Verdict.NO_TRADE)

    @given(
        proposed=st.decimals(min_value=Decimal("0.001"), max_value=Decimal("1e6"), places=3),
        fraction=st.decimals(min_value=Decimal("0.001"), max_value=Decimal("1"), places=3),
        side=st.sampled_from(list(Side)),
    )
    def test_a_human_can_only_release_up_to_the_proposal(
        self, proposed: Decimal, fraction: Decimal, side: Side
    ) -> None:
        p = _proposal(str(proposed), side)
        quantity = max(Decimal("0.001"), (proposed * fraction).quantize(Decimal("0.001")))
        quantity = min(quantity, proposed)
        for response in (
            _response(HumanAction.APPROVE), _response(HumanAction.REJECT),
            _response(HumanAction.MODIFY_SIZE, str(quantity)),
        ):
            released = resolve_intent(p, response)
            assert released.side is p.side
            assert released.quantity <= p.quantity
            assert released.symbol == p.symbol


# --- hashing and the restricted unpickler ---------------------------------------------------------


class _Opaque:
    pass


class TestHashingAndUnpickling:
    def test_state_hash_refuses_an_object_with_no_canonical_form(self) -> None:
        with pytest.raises(TypeError, match="no canonical form"):
            state_hash({"x": _Opaque()})

    def test_state_hash_is_order_independent_and_exact(self) -> None:
        left = state_hash({"a": Decimal("1.0"), "b": AT})
        assert left == state_hash({"b": AT, "a": Decimal("1.0")})
        assert state_hash({"a": Decimal("1.0")}) != state_hash({"a": Decimal("1.00")})

    def test_the_unpickler_refuses_a_callable_outside_the_allowlist(self) -> None:
        class Exploit:
            def __reduce__(self) -> tuple[Any, ...]:
                return (os.system, ("echo pwned",))

        with pytest.raises(pickle.UnpicklingError, match="blocked"):
            restricted_loads(pickle.dumps(Exploit()))

    def test_the_unpickler_refuses_its_own_loader(self) -> None:
        with pytest.raises(pickle.UnpicklingError, match="blocked"):
            restricted_loads(pickle.dumps(restricted_loads))

    def test_the_unpickler_restores_argus_types(self) -> None:
        p = _proposal()
        assert restricted_loads(pickle.dumps(p)) == p


# --- the store ------------------------------------------------------------------------------------


class TestStoreIntegrity:
    def test_pause_writes_a_request_a_continuation_and_a_committing_event(
        self, tmp_path: Path
    ) -> None:
        store, run = _paused(tmp_path)
        assert run.pause is not None
        folder = tmp_path / "requests" / run.pause.request_id
        assert {p.name for p in folder.iterdir()} == {"request.json", "continuation.json"}
        kinds = [e["event"] for e in store.events()]
        assert kinds == ["decided", "paused"]
        assert store.status(run.pause.request_id) == "paused"

    def test_a_flipped_byte_is_refused_before_unpickling(self, tmp_path: Path) -> None:
        store, run = _paused(tmp_path)
        path = tmp_path / "requests" / run.pause.request_id / "continuation.json"
        blob = json.loads(path.read_text(encoding="utf-8"))
        raw = bytearray(base64.b64decode(blob["payload_b64"]))
        raw[len(raw) // 2] ^= 0xFF
        blob["payload_b64"] = base64.b64encode(bytes(raw)).decode()
        path.write_text(json.dumps(blob), encoding="utf-8")
        with pytest.raises(PauseIntegrityError, match="do not match their recorded hash"):
            store.load_continuation(run.pause.request_id, pipeline_signature=resume_signature())

    def test_a_torn_file_is_refused(self, tmp_path: Path) -> None:
        store, run = _paused(tmp_path)
        path = tmp_path / "requests" / run.pause.request_id / "continuation.json"
        text = path.read_text(encoding="utf-8")
        path.write_text(text[: len(text) // 2], encoding="utf-8")
        with pytest.raises(PauseIntegrityError):
            store.load_continuation(run.pause.request_id, pipeline_signature=resume_signature())

    def test_an_edited_state_hash_is_refused(self, tmp_path: Path) -> None:
        store, run = _paused(tmp_path)
        path = tmp_path / "requests" / run.pause.request_id / "continuation.json"
        blob = json.loads(path.read_text(encoding="utf-8"))
        blob["state_hash"] = "f" * 64
        path.write_text(json.dumps(blob), encoding="utf-8")
        with pytest.raises(PauseIntegrityError, match="restored state hashes"):
            store.load_continuation(run.pause.request_id, pipeline_signature=resume_signature())

    def test_a_changed_pipeline_is_refused(self, tmp_path: Path) -> None:
        store, run = _paused(tmp_path)
        with pytest.raises(PauseSignatureError, match="pipeline"):
            store.load_continuation(run.pause.request_id, pipeline_signature="0" * 64)

    def test_the_pipeline_signature_tracks_the_continuation_shape(self) -> None:
        assert len(resume_signature()) == 64
        assert resume_signature() == resume_signature()

    def test_a_torn_final_event_line_is_skipped_not_fatal(self, tmp_path: Path) -> None:
        store, run = _paused(tmp_path)
        with store.events_path.open("a", encoding="utf-8") as handle:
            handle.write('{"event": "answe')
        assert store.status(run.pause.request_id) == "paused"

    def test_a_path_traversal_request_id_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(PauseError, match="not a request id"):
            PauseStore(tmp_path).load_request("../../etc")


# --- the desk: pause, answer, resume --------------------------------------------------------------


class TestDeskPauses:
    def test_without_a_store_the_desk_behaves_as_before(self) -> None:
        run, _ = run_paused_half(HALTED_APPROVE, None, decision_id="d-1")
        assert run.pause is None and run.order is None
        assert run.ruled_intent is not None and run.ruled_intent.verdict is Verdict.HUMAN_REVIEW
        assert run.as_dict()["human_loop"] is None

    def test_an_escalated_decision_is_held_and_complete(self, tmp_path: Path) -> None:
        _, run = _paused(tmp_path)
        assert run.pause is not None and run.human is None and run.order is None
        assert run.ruled_intent.verdict is Verdict.HUMAN_REVIEW
        assert run.pause.proposal.quantity == Decimal("600")
        assert run.pause.escalation.triggers == (Trigger.UNDERLYING_HALTED,)
        assert run.as_dict()["human_loop"]["state"] == "paused"
        assert any(n.startswith("[pause] held for a human") for n in run.notes)

    def test_an_unescalated_decision_is_recorded_and_not_paused(self, tmp_path: Path) -> None:
        store = PauseStore(tmp_path)
        clean = replace(HALTED_MODIFY, halted=False)
        run, _ = run_paused_half(clean, store, decision_id="d-clean")
        assert run.pause is None and run.order is not None
        assert [e["event"] for e in store.events()] == ["decided"]

    def test_a_model_requested_review_is_counted_and_not_paused(self, tmp_path: Path) -> None:
        class NoFalsifier(ScriptedModel):
            def complete_json(self, messages: list[dict[str, Any]], **kw: Any) -> dict[str, Any]:
                out = super().complete_json(messages, **kw)
                out.pop("invalidation", None)
                if "verdict" in tuple(kw.get("required_keys", ())):
                    out["invalidation"] = []
                return out

        store = PauseStore(tmp_path)
        desk = TradingDesk(NoFalsifier(quantity="60", thesis="guidance beat"))
        run = desk.run(
            symbol="NVDAUSDT",
            session=SessionState(phase=SessionPhase.RTH, as_of=AT, hours_to_next_discovery=0.0,
                                 nav_age_seconds=5.0),
            token_price=Decimal("200"), position=Decimal("0"),
            evidence=[Evidence(id="e1", claim="NVDA contract", source="news", available_at=AT,
                               credibility=0.9)],
            hedges=HedgeabilitySurface(candidates=()), decision_id="d-model",
            constitution=ConstitutionPolicy(), underlying_halted=True, pause_store=store,
        )
        assert run.proof.llm_original_intent.verdict is Verdict.HUMAN_REVIEW
        assert run.pause is None
        summary = summarise(store.events())
        assert summary.model_requested_review == 1 and summary.paused == 0

    @pytest.mark.parametrize(
        ("scenario", "order_quantity", "constitution"),
        [
            (HALTED_APPROVE, Decimal("100"), "resize"),
            (HALTED_MODIFY, Decimal("25"), "allow"),
            (HALTED_REJECT, None, "allow"),
        ],
    )
    def test_an_answer_resumes_the_decision_through_the_risk_layer(
        self, tmp_path: Path, scenario: Scenario, order_quantity: Decimal | None,
        constitution: str,
    ) -> None:
        store, run = _paused(tmp_path, scenario)
        answer = answer_for(scenario, run.pause.request_id)
        store.answer(answer, now=answer.answered_at)
        model = ScriptedModel(quantity=scenario.quantity, thesis=scenario.thesis)
        resumed = TradingDesk(model).resume(
            store, run.pause.request_id, now=AT + timedelta(minutes=5)
        )
        assert resumed.human == answer
        assert resumed.ruling is not None and str(resumed.ruling.verdict) == constitution
        if order_quantity is None:
            assert resumed.order is None
        else:
            assert resumed.order is not None and resumed.order.quantity == order_quantity
        # From the paused point: the analysts and the first decision are not re-run. Only the
        # revision can call the model, and only when the Constitution bound.
        assert model.calls == (1 if constitution != "allow" else 0)
        assert store.status(run.pause.request_id) == "resumed"
        assert resumed.as_dict()["human_loop"]["state"] == "resumed"

    def test_resume_before_an_answer_is_refused(self, tmp_path: Path) -> None:
        store, run = _paused(tmp_path)
        with pytest.raises(PauseError, match="not been answered"):
            TradingDesk(ScriptedModel(quantity="600", thesis="x")).resume(
                store, run.pause.request_id, now=AT + timedelta(minutes=5)
            )

    def test_nothing_is_answered_or_resumed_twice(self, tmp_path: Path) -> None:
        store, run = _paused(tmp_path, HALTED_REJECT)
        rid = run.pause.request_id
        store.answer(_response(HumanAction.REJECT, request_id=rid), now=AT + timedelta(minutes=2))
        with pytest.raises(AlreadyResolved):
            store.answer(_response(HumanAction.APPROVE, request_id=rid),
                         now=AT + timedelta(minutes=3))
        desk = TradingDesk(ScriptedModel(quantity="60", thesis="x"))
        desk.resume(store, rid, now=AT + timedelta(minutes=5))
        with pytest.raises(AlreadyResolved):
            desk.resume(store, rid, now=AT + timedelta(minutes=6))

    def test_an_answer_after_the_window_is_refused_and_recorded(self, tmp_path: Path) -> None:
        store, run = _paused(tmp_path)
        rid = run.pause.request_id
        with pytest.raises(PauseExpired):
            store.answer(_response(HumanAction.APPROVE, request_id=rid, minutes=31),
                         now=AT + timedelta(minutes=31))
        assert store.status(rid) == "expired"
        assert summarise(store.events()).expired == 1

    def test_a_resume_after_the_window_is_refused(self, tmp_path: Path) -> None:
        store, run = _paused(tmp_path)
        rid = run.pause.request_id
        store.answer(_response(HumanAction.APPROVE, request_id=rid),
                     now=AT + timedelta(minutes=2))
        with pytest.raises(PauseExpired):
            TradingDesk(ScriptedModel(quantity="600", thesis="x")).resume(
                store, rid, now=AT + timedelta(minutes=45)
            )
        assert store.status(rid) == "expired"

    def test_resume_with_refuses_a_continuation_the_human_did_not_see(
        self, tmp_path: Path
    ) -> None:
        _, run = _paused(tmp_path)
        other = replace(run.continuation, proposed=_proposal("5"))
        answer = _response(HumanAction.APPROVE, request_id=run.pause.request_id)
        with pytest.raises(PauseError, match="not the one the human saw"):
            TradingDesk(ScriptedModel(quantity="600", thesis="x")).resume_with(
                other, run.pause, answer, now=AT + timedelta(minutes=5)
            )


# --- LangGraph's replay, keyed by proposal --------------------------------------------------------


class TestReplay:
    def test_a_rerun_while_pending_does_not_ask_twice(self, tmp_path: Path) -> None:
        store, first = _paused(tmp_path)
        second, _ = run_paused_half(HALTED_APPROVE, store, decision_id="d-1")
        assert second.pause is not None
        assert second.pause.request_id == first.pause.request_id
        assert second.continuation is None  # its snapshot is not the request's durable state
        assert [e["event"] for e in store.events()].count("paused") == 1

    def test_a_rerun_after_an_answer_replays_it(self, tmp_path: Path) -> None:
        store, first = _paused(tmp_path, HALTED_MODIFY)
        answer = answer_for(HALTED_MODIFY, first.pause.request_id)
        store.answer(answer, now=answer.answered_at)
        rerun, _ = run_paused_half(HALTED_MODIFY, store, decision_id="d-1")
        assert rerun.human == answer
        assert rerun.order is not None and rerun.order.quantity == Decimal("25")
        resumed = [e for e in store.events() if e["event"] == "resumed"]
        assert len(resumed) == 1 and resumed[0]["replayed"] is True

    def test_a_rerun_after_resumption_is_not_executed_again(self, tmp_path: Path) -> None:
        store, first = _paused(tmp_path, HALTED_MODIFY)
        answer = answer_for(HALTED_MODIFY, first.pause.request_id)
        store.answer(answer, now=answer.answered_at)
        TradingDesk(ScriptedModel(quantity="60", thesis="x")).resume(
            store, first.pause.request_id, now=AT + timedelta(minutes=5)
        )
        rerun, _ = run_paused_half(HALTED_MODIFY, store, decision_id="d-1")
        assert rerun.order is None
        assert any("already answered and acted on" in n for n in rerun.notes)

    def test_a_different_proposal_supersedes_and_asks_again(self, tmp_path: Path) -> None:
        store, first = _paused(tmp_path, HALTED_APPROVE)
        second, _ = run_paused_half(HALTED_MODIFY, store, decision_id="d-1")  # 60, not 600
        assert second.pause is not None
        assert second.pause.request_id != first.pause.request_id
        assert second.pause.sequence == 1
        assert store.status(first.pause.request_id) == "superseded"

    def test_an_answered_old_proposal_is_withdrawn_not_left_to_execute(
        self, tmp_path: Path
    ) -> None:
        store, first = _paused(tmp_path, HALTED_APPROVE)
        store.answer(_response(HumanAction.APPROVE, request_id=first.pause.request_id),
                     now=AT + timedelta(minutes=2))
        run_paused_half(HALTED_MODIFY, store, decision_id="d-1")  # a different proposal
        assert store.status(first.pause.request_id) == "superseded"
        with pytest.raises(AlreadyResolved):
            TradingDesk(ScriptedModel(quantity="600", thesis="x")).resume(
                store, first.pause.request_id, now=AT + timedelta(minutes=5)
            )


# --- the takeover rate from events ----------------------------------------------------------------


class TestSummary:
    def test_the_rate_from_events_is_the_escalation_module_rate(self, tmp_path: Path) -> None:
        store = PauseStore(tmp_path)
        outcomes = []
        for i, halted in enumerate((True, False, True, False)):
            scenario = replace(HALTED_MODIFY, halted=halted)
            run, _ = run_paused_half(scenario, store, decision_id=f"d-{i}")
            esc = run.pause.escalation if run.pause else Escalation(triggers=(), detail=())
            outcomes.append((True, esc))
        summary = summarise(store.events())
        assert summary.takeover.as_dict() == takeover_rate(outcomes).as_dict()
        assert summary.takeover.rate == 0.5
        assert summary.pending == 2

    def test_override_and_unanswered_rates(self, tmp_path: Path) -> None:
        store = PauseStore(tmp_path)
        plan = [HumanAction.APPROVE, HumanAction.MODIFY_SIZE, HumanAction.REJECT, None]
        for i, action in enumerate(plan):
            run, _ = run_paused_half(HALTED_MODIFY, store, decision_id=f"d-{i}")
            assert run.pause is not None
            rid = run.pause.request_id
            if action is None:
                with pytest.raises(PauseExpired):
                    store.answer(_response(HumanAction.APPROVE, request_id=rid, minutes=40),
                                 now=AT + timedelta(minutes=40))
                continue
            qty = "25" if action is HumanAction.MODIFY_SIZE else None
            store.answer(_response(action, qty, request_id=rid, minutes=2 + i),
                         now=AT + timedelta(minutes=2 + i))
        summary = summarise(store.events())
        assert (summary.approved, summary.modified, summary.rejected, summary.expired) == (
            1, 1, 1, 1,
        )
        assert summary.override_rate == pytest.approx(2 / 3)
        assert summary.unanswered_rate == pytest.approx(1 / 4)
        assert summary.median_answer_seconds == 180.0

    def test_nothing_answered_is_undefined_not_zero(self) -> None:
        summary = summarise([])
        assert summary.override_rate is None and summary.unanswered_rate is None
        assert summary.takeover.rate is None


# --- the kill-and-resume property -----------------------------------------------------------------


class TestKillAndResume:
    @pytest.mark.parametrize(
        "key", ["underlying_halted/approve", "halted_and_ungrounded/modify_size",
                "ungrounded_figure/reject"],
    )
    def test_killed_while_paused_resumes_to_the_same_state_and_decision(
        self, tmp_path: Path, key: str
    ) -> None:
        row = drill_one(by_key(key), tmp_path)
        assert row["child"]["alive_when_killed"] is True
        assert row["child"]["returncode"] != 0  # it was killed, not exited
        assert row["state_hash"]["identical"] is True
        assert row["decision"]["identical_to_no_kill"] is True, row["decision"]["difference_detail"]

    def test_the_comparison_can_fail(self, tmp_path: Path) -> None:
        """Negative control: the comparator is not blind. A different answer must differ."""
        _, run = _paused(tmp_path, HALTED_MODIFY)
        desk = TradingDesk(ScriptedModel(quantity="60", thesis=HALTED_MODIFY.thesis))
        rid = run.pause.request_id
        a = desk.resume_with(run.continuation, run.pause, _response(
            HumanAction.MODIFY_SIZE, "25", request_id=rid), now=AT + timedelta(minutes=5))
        b = desk.resume_with(run.continuation, run.pause, _response(
            HumanAction.MODIFY_SIZE, "24", request_id=rid), now=AT + timedelta(minutes=5))
        assert comparable(a) != comparable(b)

    def test_the_in_memory_snapshot_hashes_to_the_request(self, tmp_path: Path) -> None:
        _, run = _paused(tmp_path)
        assert isinstance(run.continuation, DeskContinuation)
        assert state_hash(run.continuation.signature_state()) == run.pause.state_hash


# --- the CLI --------------------------------------------------------------------------------------


def _cli(*args: str) -> tuple[int, str]:
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = main(list(args))
    return code, buf.getvalue()


class TestCli:
    def test_list_show_answer_rate(self, tmp_path: Path) -> None:
        _, run = _paused(tmp_path, HALTED_MODIFY)
        rid = run.pause.request_id
        root = str(tmp_path)
        at = (AT + timedelta(minutes=3)).isoformat()

        code, out = _cli("--root", root, "list", "--at", at)
        assert code == 0 and rid in out and "underlying_halted" in out

        code, out = _cli("--root", root, "show", rid)
        shown = json.loads(out)
        assert shown["status"] == "paused" and shown["request"]["request_id"] == rid

        code, out = _cli("--root", root, "answer", rid, "modify", "--quantity", "25",
                         "--reviewer", "reviewer-1", "--at", at)
        assert code == 0 and "recorded" in out
        assert PauseStore(tmp_path).status(rid) == "answered"

        code, out = _cli("--root", root, "list", "--at", at)
        assert "nothing is waiting" in out

        code, out = _cli("--root", root, "rate")
        assert json.loads(out)["modified"] == 1

    def test_answer_refusals_exit_non_zero(self, tmp_path: Path) -> None:
        _, run = _paused(tmp_path, HALTED_MODIFY)
        rid = run.pause.request_id
        at = (AT + timedelta(minutes=3)).isoformat()
        code, _ = _cli("--root", str(tmp_path), "answer", rid, "modify", "--quantity", "abc",
                       "--reviewer", "p", "--at", at)
        assert code == 2
        code, _ = _cli("--root", str(tmp_path), "answer", rid, "modify", "--quantity", "999",
                       "--reviewer", "p", "--at", at)
        assert code == 1
        late = (AT + timedelta(hours=2)).isoformat()
        code, _ = _cli("--root", str(tmp_path), "answer", rid, "approve", "--reviewer", "p",
                       "--at", late)
        assert code == 1
        assert PauseStore(tmp_path).status(rid) == "expired"

    def test_the_module_runs_as_a_script(self, tmp_path: Path) -> None:
        _paused(tmp_path, HALTED_MODIFY)
        done = subprocess.run(
            [sys.executable, "-m", "argus.decision.pause", "--root", str(tmp_path), "list",
             "--at", (AT + timedelta(minutes=3)).isoformat()],
            capture_output=True, text=True, timeout=120,
            env={**{k: v for k, v in os.environ.items() if k != "BITGET_QWEN_API_KEY"},
                 "PYTHONUTF8": "1"},
        )
        assert done.returncode == 0, done.stderr
        assert "d-1-p0" in done.stdout


def test_timestamps_are_never_naive() -> None:
    with pytest.raises(PauseError, match="timezone"):
        HumanResponse.from_dict({
            "format": "argus-pause/1", "request_id": "x-p0", "action": "approve",
            "reviewer": "p", "answered_at": datetime(2026, 9, 14, 15, 0).isoformat(),
        })
