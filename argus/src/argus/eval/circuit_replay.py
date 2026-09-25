"""The trajectory breaker, replayed over the recorded ledger and over injected faults.

Two questions, answered without a model call:

1. **Does the breaker stay quiet on the real record?** Every decision in ``data/paper_ledger.jsonl``
   is replayed through :func:`argus.agents.circuit.assess` in sequence, each against the decisions
   before it — the same computation the desk now runs live. Every trip on this replay is a false
   trip unless it points at a real loop, and each one is listed with its reason. The result is
   reported as a human-takeover-rate style count: trips per hundred decisions.
2. **Does it catch what it exists to catch, with the right reason?** Faults are injected into a
   scripted copy of the same trajectory — malformed model outputs, repeated equivalent order
   intents (sizes jittered inside the tolerance, and one just outside it as a control), a runaway
   caller, and a venue call that times out on every attempt — and each is scored on whether it
   tripped, and tripped for the reason it was injected.

`Track 2` names *human-takeover rate* and *decision consistency* by title. Neither number exists in
any harness in the corpus (`research/architecture/agent-evaluation-harnesses.md`); this artefact is
where ARGUS's is recorded.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from random import Random
from typing import Any

from argus.agents.circuit import (
    DEFAULT_THRESHOLDS,
    FaultKind,
    RetryPolicy,
    Step,
    Thresholds,
    Trip,
    assess,
    call_with_retry,
    trips_per_100,
)
from argus.eval.artefact import write
from argus.eval.perturbations import DATA

LEDGER = DATA / "paper_ledger.jsonl"
OUT_PATH = DATA / "circuit_replay.json"


def load_trajectory(path: Path = LEDGER) -> list[Step]:
    steps: list[Step] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        step = Step.from_record(json.loads(line))
        if step is not None:
            steps.append(step)
    return sorted(steps, key=lambda s: s.at)


def replay(
    steps: Sequence[Step], thresholds: Thresholds = DEFAULT_THRESHOLDS
) -> list[tuple[int, Step, str, str]]:
    """``(index, step, trip, reason)`` for every decision the breaker would have stopped."""
    trips: list[tuple[int, Step, str, str]] = []
    for i, step in enumerate(steps):
        reading = assess(steps[:i], step, thresholds=thresholds)
        if reading.trip is not None:
            trips.append((i, step, reading.trip.value, reading.reason))
    return trips


def recorded_maxima(steps: Sequence[Step], window: timedelta) -> dict[str, Any]:
    """The largest per-symbol decision count in any window: the step budget's margin, visible."""
    worst = 0
    worst_at: str | None = None
    for i, s in enumerate(steps):
        n = sum(1 for o in steps[: i + 1] if o.symbol == s.symbol and s.at - window <= o.at)
        if n > worst:
            worst, worst_at = n, f"{s.symbol} at {s.at.isoformat()}"
    return {
        "max_decisions_per_symbol_in_window": worst, "where": worst_at,
        "exposure_opening_decisions": sum(1 for s in steps if s.opens_exposure),
        "invalid_outputs": sum(1 for s in steps if not s.valid),
        "verdicts": dict(Counter(s.verdict for s in steps)),
    }


def _trip_at(steps: Sequence[Step], index: int) -> str | None:
    reading = assess(steps[:index], steps[index])
    return None if reading.trip is None else reading.trip.value


def injections(clean: Sequence[Step]) -> list[dict[str, Any]]:
    """Each fault family planted after the end of the recorded run, scored on the right reason."""
    base = list(clean)
    t0 = base[-1].at + timedelta(hours=1) if base else datetime(2026, 9, 26, tzinfo=UTC)
    sym = "NVDAUSDT"
    out: list[dict[str, Any]] = []

    def trade(k: int, qty: str, side: str = "buy", hours: float = 0.0) -> Step:
        return Step(symbol=sym, at=t0 + timedelta(hours=hours + k), verdict="trade", side=side,
                    quantity=Decimal(qty))

    # 1. repeated equivalent order intent: sizes inside the 10% tolerance, not string-equal.
    seq = [*base, trade(0, "50"), trade(1, "52"), trade(2, "48")]
    out.append({"fault": "repeated_equivalent_order", "expected": Trip.EQUIVALENT_ORDERS.value,
                "got": _trip_at(seq, len(seq) - 1),
                "earlier_steps_tripped": [
                    _trip_at(seq, len(seq) - 3), _trip_at(seq, len(seq) - 2)
                ]})
    # 1b. control: the third order is 20% larger, outside tolerance — must NOT trip.
    seq = [*base, trade(0, "50"), trade(1, "52"), trade(2, "62")]
    out.append({"fault": "control_size_outside_tolerance", "expected": None,
                "got": _trip_at(seq, len(seq) - 1)})
    # 1c. control: the same orders spread over three days — outside the window, must NOT trip.
    seq = [*base, trade(0, "50"), trade(0, "50", hours=25), trade(0, "50", hours=50)]
    out.append({"fault": "control_outside_window", "expected": None,
                "got": _trip_at(seq, len(seq) - 1)})
    # 1d. control: alternating sides is not a repeated order.
    seq = [*base, trade(0, "50"), trade(1, "50", side="sell"), trade(2, "50")]
    out.append({"fault": "control_alternating_side", "expected": None,
                "got": _trip_at(seq, len(seq) - 1)})

    # 2. malformed model output three times in a row (across symbols: the model is shared).
    bad = [
        Step(symbol=s, at=t0 + timedelta(minutes=i), verdict="human_review", side="buy",
             quantity=Decimal("0"), valid=False)
        for i, s in enumerate(("NVDAUSDT", "TSLAUSDT", "METAUSDT"))
    ]
    seq = [*base, *bad]
    out.append({"fault": "malformed_output_x3", "expected": Trip.INVALID_OUTPUTS.value,
                "got": _trip_at(seq, len(seq) - 1),
                "earlier_steps_tripped": [
                    _trip_at(seq, len(seq) - 3), _trip_at(seq, len(seq) - 2)
                ]})
    # 2b. control: a deliberate human_review with a valid answer is a decision, not a fault.
    seq = [*base, *[replace(b, valid=True) for b in bad]]
    out.append({"fault": "control_deliberate_human_review", "expected": None,
                "got": _trip_at(seq, len(seq) - 1)})

    # 3. a runaway caller: one decision past the budget, two minutes apart, on one symbol.
    runaway = [
        Step(symbol=sym, at=t0 + timedelta(minutes=2 * i), verdict="no_trade", side="buy",
             quantity=Decimal("0"))
        for i in range(DEFAULT_THRESHOLDS.max_steps + 1)
    ]
    seq = [*base, *runaway]
    out.append({"fault": "runaway_caller", "expected": Trip.STEP_BUDGET.value,
                "got": _trip_at(seq, len(seq) - 1),
                "step_before_budget_tripped": _trip_at(seq, len(seq) - 2)})

    # 4. a venue call that never answers: timeout on every attempt, then a named give-up.
    waits: list[float] = []

    def hung() -> str:
        import time

        time.sleep(0.5)
        return "filled"

    outcome = call_with_retry(
        hung, name="venue.place_order",
        policy=RetryPolicy(max_attempts=3, timeout_seconds=0.02, jitter=False,
                           initial_interval=0.1),
        sleep=waits.append, rng=Random(0),
    )
    out.append({
        "fault": "venue_timeout_every_attempt", "expected": "gave_up:timeout",
        "got": None if outcome.ok else f"gave_up:{outcome.fault}",
        "attempts": len(outcome.attempts), "backoff_waits": waits, "give_up": outcome.gave_up,
    })

    # 5. a transient venue fault that clears: retried, then succeeds — must NOT give up.
    calls = {"n": 0}

    def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("connection reset by peer")
        return "filled"

    outcome = call_with_retry(flaky, name="venue.place_order",
                              policy=RetryPolicy(max_attempts=3, jitter=False),
                              sleep=lambda _s: None)
    out.append({"fault": "control_transient_then_success", "expected": None,
                "got": None if outcome.ok else f"gave_up:{outcome.fault}",
                "attempts": len(outcome.attempts)})

    # 6. a rejected order (4xx) is never re-sent.
    class Rejected(Exception):
        status_code = 400

    sent = {"n": 0}

    def rejected() -> str:
        sent["n"] += 1
        raise Rejected("order rejected: insufficient margin")

    outcome = call_with_retry(rejected, name="venue.place_order",
                              policy=RetryPolicy(max_attempts=3), sleep=lambda _s: None)
    out.append({"fault": "venue_rejection_not_resent", "expected": "gave_up:permanent",
                "got": None if outcome.ok else f"gave_up:{outcome.fault}", "times_sent": sent["n"]})

    for row in out:
        row["caught_with_right_reason"] = row["got"] == row["expected"]
    assert FaultKind.TIMEOUT.retryable  # the venue-timeout row depends on it
    return out


def run() -> dict[str, Any]:
    steps = load_trajectory()
    trips = replay(steps)
    injected = injections(steps)
    blob = {
        "generated_at": datetime.now(UTC).isoformat(),
        "item": "S17 trajectory circuit-breaker (agents/circuit.py), replayed",
        "ledger": str(LEDGER.name),
        "thresholds": {
            "max_steps_per_symbol_per_window": DEFAULT_THRESHOLDS.max_steps,
            "invalid_outputs": DEFAULT_THRESHOLDS.invalid_outputs,
            "equivalent_orders": DEFAULT_THRESHOLDS.equivalent_orders,
            "nudge_at": DEFAULT_THRESHOLDS.nudge_at,
            "size_tolerance": str(DEFAULT_THRESHOLDS.size_tolerance),
            "window_hours": DEFAULT_THRESHOLDS.window.total_seconds() / 3600,
        },
        "clean_replay": {
            "decisions": len(steps),
            "first": steps[0].at.isoformat() if steps else None,
            "last": steps[-1].at.isoformat() if steps else None,
            "trips": len(trips),
            "trips_per_100_decisions": trips_per_100(len(trips), len(steps)),
            "trip_rows": [
                {"index": i, "symbol": s.symbol, "at": s.at.isoformat(), "trip": t, "reason": r}
                for i, s, t, r in trips
            ],
            "recorded": recorded_maxima(steps, DEFAULT_THRESHOLDS.window),
        },
        "step_budget_sensitivity": {
            str(budget): len(replay(steps, replace(DEFAULT_THRESHOLDS, max_steps=budget)))
            for budget in (24, 32, 48)
        },
        "injected": injected,
        "injected_caught_with_right_reason": sum(r["caught_with_right_reason"] for r in injected),
        "injected_total": len(injected),
        "not_measured": (
            "the replay reads the ledger's recorded (post-Constitution) verdicts; the live desk "
            "assesses the intent before the Constitution, so a repeated proposal the Constitution "
            "blocked is invisible to this replay. The Meta-PM give-up path is exercised end to end "
            "in tests/test_circuit_breaker.py with a scripted model, not here."
        ),
        "qwen_requests": 0,
    }
    write(OUT_PATH, blob)
    return blob


def main() -> int:  # pragma: no cover - CLI
    blob = run()
    print(json.dumps({
        "clean": {k: blob["clean_replay"][k] for k in ("decisions", "trips",
                                                      "trips_per_100_decisions")},
        "recorded": blob["clean_replay"]["recorded"],
        "injected": [(r["fault"], r["expected"], r["got"]) for r in blob["injected"]],
        "caught": f"{blob['injected_caught_with_right_reason']}/{blob['injected_total']}",
    }, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = ["OUT_PATH", "injections", "load_trajectory", "recorded_maxima", "replay", "run"]
