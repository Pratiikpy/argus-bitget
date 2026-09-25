"""The pause drill: kill the desk while it waits for a human, resume it, and compare.

`decision/pause.py` claims the desk can stop on an escalation, be killed outright while it waits,
and continue from the paused point once a human answers — with the same state and the same
decision as if nothing had been killed. This module is the measurement of that claim, and nothing
here asks a reader to take it on trust:

- **The kill is real.** Each scenario's paused half runs in a *separate Python process*
  (``python -m argus.eval.pausedrill child ...``), which prints its request id and state hash once
  the pause has committed and then blocks. The parent confirms it is still alive, kills it with
  ``Popen.kill()`` — ``TerminateProcess`` on Windows, ``SIGKILL`` elsewhere, no cleanup handlers
  run — and only then answers and resumes, with a fresh desk object that shares nothing with the
  process that paused.
- **The baseline is the same pause without the kill.** The same scenario runs in-process, is
  answered with the same answer, and is resumed from the in-memory continuation
  (`TradingDesk.resume_with`) with no disk round trip.
- **The comparison is the whole record.** ``DeskRun.as_dict()`` of the two resumed runs is compared
  field for field. One note is normalised first — the panel's wall-clock duration ("ran
  concurrently in 0.0s"), which is the only nondeterministic value a desk run records and differs
  between any two runs whether or not one was killed. Nothing else is excluded.
- **The state hash is compared twice**: the hash the child printed before it died against the hash
  the parent recomputes from the restored continuation (the durability property), and the request's
  recorded hash against both.

The scripted model is deterministic and makes **no network call**, so this drill spends zero Qwen
calls. That is also its limit, stated rather than hidden: it proves the pause/resume machinery is
exact, not that a live model's second half behaves identically — a live model's revision call on
resume is a fresh call and can differ, which is the reason a resume restores the first half instead
of re-running it.

The "before" column is what the desk did with the same scenarios before pauses existed: an
escalated decision concluded as ``HUMAN_REVIEW`` with no order, and there was no operation by which
a human's answer could reach it. That is a capability comparison, not an accuracy one, and it is
labelled as such in the artefact.

Run: ``python -m argus.eval.pausedrill`` (writes ``data/pause_drill.json``).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from argus.agents.desk import ConstitutionPolicy, DeskRun, TradingDesk, resume_signature
from argus.decision.pause import (
    HumanAction,
    HumanResponse,
    PauseStore,
    state_hash,
    summarise,
)
from argus.eval.artefact import write
from argus.risk.hedgeability import HedgeabilitySurface
from argus.truth.clocks import SessionPhase, SessionState
from argus.truth.evidence import Evidence

AT = datetime(2026, 9, 14, 15, 0, tzinfo=UTC)
TOKEN_PRICE = Decimal("200")
REPORT_PATH = Path(__file__).resolve().parents[3] / "data" / "pause_drill.json"


class ScriptedModel:
    """A deterministic stand-in for the desk's model. Never touches the network.

    Every decision-shaped call (the Meta-PM's decision and its revision) returns the same TRADE;
    every other call returns one bullish analyst view. Deterministic by construction, so two runs
    of one scenario can differ only through the desk itself.
    """

    budget: Any = None
    """Part of the `ChatModel` protocol. A scripted model spends nothing, so there is none."""

    def __init__(self, *, quantity: str, thesis: str) -> None:
        self.quantity = quantity
        self.thesis = thesis
        self.calls = 0

    def complete_json(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        self.calls += 1
        required = tuple(kwargs.get("required_keys", ()))
        if "verdict" in required:
            return {
                "verdict": "TRADE", "side": "buy", "quantity": self.quantity,
                "confidence": 0.9, "thesis": self.thesis,
                "invalidation": ["guidance reaffirmed lower"],
            }
        return {
            "signal": "bullish", "confidence": 0.8, "magnitude_bps": 40,
            "reasoning": "guidance beat", "chain": ["contract signed", "revenue recognised"],
        }

    def complete(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("the drill's scripted model answers JSON calls only")


@dataclass(frozen=True, slots=True)
class Scenario:
    key: str
    trigger: str
    halted: bool
    thesis: str
    quantity: str
    action: HumanAction
    modify_to: Decimal | None = None


_GROUNDED = "guidance beat, not yet priced in"
_UNGROUNDED = "revenue grew 37% and margins reached 81%, which the price has not absorbed"


def scenarios() -> list[Scenario]:
    """Three triggers x three answers. The approve case proposes 600 units ($120,000 at $200) so
    the Constitution binds on resume and the model's revision call runs — the hardest path for the
    resume to reproduce, and the one that shows a human's approval does not release the risk
    layer."""
    triggers = (
        ("underlying_halted", True, _GROUNDED),
        ("ungrounded_figure", False, _UNGROUNDED),
        ("halted_and_ungrounded", True, _UNGROUNDED),
    )
    answers: tuple[tuple[HumanAction, str, Decimal | None], ...] = (
        (HumanAction.APPROVE, "600", None),
        (HumanAction.MODIFY_SIZE, "60", Decimal("25")),
        (HumanAction.REJECT, "60", None),
    )
    return [
        Scenario(
            key=f"{trigger}/{action.value}", trigger=trigger, halted=halted, thesis=thesis,
            quantity=quantity, action=action, modify_to=modify_to,
        )
        for trigger, halted, thesis in triggers
        for action, quantity, modify_to in answers
    ]


def by_key(key: str) -> Scenario:
    for s in scenarios():
        if s.key == key:
            return s
    raise KeyError(key)


def run_paused_half(
    scenario: Scenario, store: PauseStore | None, *, decision_id: str
) -> tuple[DeskRun, ScriptedModel]:
    model = ScriptedModel(quantity=scenario.quantity, thesis=scenario.thesis)
    desk = TradingDesk(model)
    run = desk.run(
        symbol="NVDAUSDT",
        session=SessionState(
            phase=SessionPhase.RTH, as_of=AT, hours_to_next_discovery=0.0, nav_age_seconds=5.0,
        ),
        token_price=TOKEN_PRICE,
        position=Decimal("0"),
        evidence=[
            Evidence(id="e1", claim="NVDA announced a new datacentre contract", source="news",
                     available_at=AT, credibility=0.9),
        ],
        hedges=HedgeabilitySurface(candidates=()),
        decision_id=decision_id,
        constitution=ConstitutionPolicy(),
        underlying_halted=scenario.halted,
        halt_reason="LUDP volatility halt" if scenario.halted else "",
        pause_store=store,
    )
    return run, model


def answer_for(scenario: Scenario, request_id: str) -> HumanResponse:
    return HumanResponse(
        request_id=request_id, action=scenario.action, reviewer="drill-reviewer",
        answered_at=AT + timedelta(minutes=4), quantity=scenario.modify_to,
        note=f"drill {scenario.key}",
    )


_TIMING = re.compile(r"ran concurrently in \d+(\.\d+)?s")


def comparable(run: DeskRun) -> dict[str, Any]:
    """The whole record, with the one wall-clock value normalised. See the module docstring."""
    record = run.as_dict()
    record["notes"] = [_TIMING.sub("ran concurrently in <t>s", n) for n in record["notes"]]
    return record


def _difference(no_kill: Any, killed: Any) -> dict[str, Any]:
    """What differs, small enough to read in a failure message. Lists report only their changes."""
    if isinstance(no_kill, list) and isinstance(killed, list):
        return {
            "only_no_kill": [str(x)[:300] for x in no_kill if x not in killed],
            "only_killed": [str(x)[:300] for x in killed if x not in no_kill],
        }
    return {"no_kill": str(no_kill)[:600], "killed": str(killed)[:600]}


def _child(scenario_key: str, root: Path) -> int:
    """The process that pauses and is then killed. Prints one line once the pause has committed."""
    scenario = by_key(scenario_key)
    run, _ = run_paused_half(scenario, PauseStore(root), decision_id=f"drill-{scenario.key}")
    if run.pause is None:
        sys.stdout.write(json.dumps({"error": "no pause"}) + "\n")
        sys.stdout.flush()
        return 3
    sys.stdout.write(json.dumps({
        "request_id": run.pause.request_id,
        "state_hash": run.pause.state_hash,
        "pid": os.getpid(),
    }) + "\n")
    sys.stdout.flush()
    while True:  # waiting for a human; the parent kills us here
        time.sleep(0.5)


def kill_while_paused(scenario: Scenario, root: Path) -> dict[str, Any]:
    """Run the paused half in a child process, kill it, and return what it said before it died."""
    env = {k: v for k, v in os.environ.items() if k != "BITGET_QWEN_API_KEY"}
    env["PYTHONUTF8"] = "1"
    proc = subprocess.Popen(
        [sys.executable, "-m", "argus.eval.pausedrill", "child", scenario.key, str(root)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL, env=env,
        text=True,
    )
    assert proc.stdout is not None
    line = proc.stdout.readline()
    alive_when_killed = proc.poll() is None
    proc.kill()
    proc.wait(timeout=30)
    stderr = proc.stderr.read() if proc.stderr is not None else ""
    if not line:
        raise RuntimeError(f"the paused child printed nothing; stderr:\n{stderr[-2000:]}")
    said = json.loads(line)
    if "error" in said:
        raise RuntimeError(f"the child did not pause: {said}")
    said["alive_when_killed"] = alive_when_killed
    said["returncode"] = proc.returncode
    return dict(said)


def drill_one(scenario: Scenario, workdir: Path) -> dict[str, Any]:
    decision_id = f"drill-{scenario.key}"
    resume_at = AT + timedelta(minutes=5)

    # Before pauses existed: the same decision with no store.
    before, _ = run_paused_half(scenario, None, decision_id=decision_id)

    # The no-kill baseline: pause, answer, resume from memory.
    store_a = PauseStore(workdir / "nokill")
    held, _ = run_paused_half(scenario, store_a, decision_id=decision_id)
    assert held.pause is not None and held.continuation is not None
    answer_a = answer_for(scenario, held.pause.request_id)
    store_a.answer(answer_a, now=answer_a.answered_at)
    resume_model_a = ScriptedModel(quantity=scenario.quantity, thesis=scenario.thesis)
    nokill = TradingDesk(resume_model_a).resume_with(
        held.continuation, held.pause, answer_a, store=store_a, now=resume_at,
    )

    # The killed run: pause in a child, kill it, answer and resume here from disk alone.
    store_b = PauseStore(workdir / "killed")
    said = kill_while_paused(scenario, store_b.root)
    answer_b = answer_for(scenario, said["request_id"])
    store_b.answer(answer_b, now=answer_b.answered_at)
    resume_model_b = ScriptedModel(quantity=scenario.quantity, thesis=scenario.thesis)
    restored = store_b.load_continuation(said["request_id"], pipeline_signature=resume_signature())
    recomputed = state_hash(restored.signature_state())
    killed = TradingDesk(resume_model_b).resume(store_b, said["request_id"], now=resume_at)
    request_b = store_b.load_request(said["request_id"])

    a, b = comparable(nokill), comparable(killed)
    differing = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    detail = {k: _difference(a.get(k), b.get(k)) for k in differing}
    return {
        "scenario": scenario.key,
        "trigger": scenario.trigger,
        "answer": scenario.action.value,
        "before_pauses": {
            "final_verdict": str(before.ruled_intent.verdict) if before.ruled_intent else None,
            "order": before.order is not None,
            "a_human_answer_could_reach_it": False,
        },
        "child": {
            "alive_when_killed": said["alive_when_killed"],
            "returncode": said["returncode"],
        },
        "state_hash": {
            "printed_before_kill": said["state_hash"],
            "recorded_on_request": request_b.state_hash,
            "recomputed_after_restore": recomputed,
            "identical": said["state_hash"] == request_b.state_hash == recomputed,
        },
        "decision": {
            "identical_to_no_kill": not differing,
            "differing_fields": differing,
            "difference_detail": detail,
            "constitution": None if killed.ruling is None else str(killed.ruling.verdict),
            "binding_constraint": None if killed.ruling is None
            else killed.ruling.binding_constraint,
            "revised_by_model": killed.proof.llm_revised_intent is not None,
            "approved_intent_hash": killed.proof.approved_intent_hash,
            "order_quantity": None if killed.order is None else str(killed.order.quantity),
            "order_client_id": None if killed.order is None else killed.order.client_order_id,
            "proposed_quantity": scenario.quantity,
        },
        "model_calls_on_resume": {"no_kill": resume_model_a.calls, "killed": resume_model_b.calls},
        "events_summary": summarise(store_b.events()).as_dict(),
    }


def run_drill(workdir: Path | None = None) -> dict[str, Any]:
    started = time.monotonic()
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="argus-pausedrill-") as tmp:
        base = workdir or Path(tmp)
        for i, scenario in enumerate(scenarios()):
            rows.append(drill_one(scenario, base / f"s{i}"))
    n = len(rows)
    hashes_ok = sum(1 for r in rows if r["state_hash"]["identical"])
    decisions_ok = sum(1 for r in rows if r["decision"]["identical_to_no_kill"])
    killed_alive = sum(1 for r in rows if r["child"]["alive_when_killed"])
    return {
        "what": (
            "Kill-and-resume drill for decision/pause.py: each scenario paused in a child "
            "process that was killed while waiting, then answered and resumed from disk by a "
            "fresh desk; compared with the same scenario paused and resumed in memory"
        ),
        "measured_at_as_of": AT.isoformat(),
        "qwen_calls": 0,
        "model": "scripted, deterministic, no network (argus.eval.pausedrill.ScriptedModel)",
        "scenarios": n,
        "children_alive_when_killed": killed_alive,
        "state_hash_identical": hashes_ok,
        "decision_identical_to_no_kill": decisions_ok,
        "comparison_kind": (
            "capability, not accuracy: before pauses, 0 of these escalated decisions could be "
            "resumed by a human answer; after, the count is decision_identical_to_no_kill"
        ),
        "before": {
            "resumable_by_a_human_answer": 0,
            "orders_placed": sum(1 for r in rows if r["before_pauses"]["order"]),
        },
        "after": {
            "resumed": n,
            "orders_placed": sum(1 for r in rows if r["decision"]["order_quantity"] is not None),
            "constitution_bound_on_resume": sum(
                1 for r in rows if r["decision"]["constitution"] not in (None, "allow")
            ),
        },
        "not_proven_here": [
            "a live model's revision call on resume is a fresh call and may differ from the one a "
            "no-kill run would make; this drill uses a deterministic model",
            "the drill's human is scripted; median answer time is therefore not a human's",
        ],
        "seconds": round(time.monotonic() - started, 2),
        "rows": rows,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "child":
        return _child(args[1], Path(args[2]))
    report = run_drill()
    write(REPORT_PATH, report)
    sys.stdout.write(
        f"{report['scenarios']} scenarios: state hash identical in "
        f"{report['state_hash_identical']}, decision identical to no-kill in "
        f"{report['decision_identical_to_no_kill']}, children "
        f"alive when killed {report['children_alive_when_killed']}; wrote {REPORT_PATH}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
