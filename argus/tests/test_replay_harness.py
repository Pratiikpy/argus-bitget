"""The paper harness proved on its own: the real runner, driven by a scripted known-correct sequence
(teacher forcing), must report exactly the arithmetic done by hand; the same run with a planted
settlement defect must fail; and a person's decisions run through the same runner and scorer."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from argus.eval import replay_harness as rh
from argus.eval.replay_harness import (
    Decision,
    Market,
    ReplayError,
    Seat,
    oracle,
    replay,
    replay_market,
    validate,
)


@pytest.fixture(scope="module")
def market() -> Market:
    return replay_market()


@pytest.fixture(scope="module")
def instruments() -> dict[str, Any]:
    return rh._instruments()


@pytest.fixture(scope="module")
def teacher(market: Market, instruments: dict[str, Any]) -> dict[str, Decision]:
    return oracle(market, instruments, abstain=rh.TEACHER_ABSTAINS,
                  oversize=rh.TEACHER_OVERSIZES)


def test_the_replay_instant_follows_its_stated_rule(market: Market) -> None:
    assert market.decided_at.weekday() in (1, 2, 3) and market.decided_at.hour == 15
    assert market.marked_at - market.decided_at == rh.MARK_AFTER
    assert market.settled_at - market.decided_at == rh.SETTLE_AFTER
    assert len(market.prices) == 12


def test_the_gold_arithmetic_is_done_by_hand() -> None:
    # 10 bought at 100, settled at 110: 100 gross; 8bps a side (6 taker + 2 spread) on 1,000 in
    # and 1,100 out is 0.80 + 0.88.
    gross, net = rh._gold_net(Decimal("10"), "buy", Decimal("100"), Decimal("110"))
    assert (gross, net) == (Decimal("100"), Decimal("98.32"))
    gross, net = rh._gold_net(Decimal("10"), "sell", Decimal("100"), Decimal("110"))
    assert (gross, net) == (Decimal("-100"), Decimal("-101.68"))


def test_teacher_forcing_passes_every_check_on_the_real_runner(
        market: Market, instruments: dict[str, Any], teacher: dict[str, Decision]) -> None:
    from argus.paper import runner

    patched = ("fetch_rtokens", "QwenClient", "_settle_due", "datetime", "RISK_PATH")
    real = [getattr(runner, name) for name in patched]
    run = replay("teacher", teacher, market, instruments)
    checks = validate(run, market, oracle_run=True)
    failed = [f"{c.name}: {c.detail}" for c in checks if not c.passed]
    assert not failed, failed
    assert len(checks) == 16
    # the teacher's oversized order made the risk layer bind and the Meta-PM re-decide
    assert run.seat_calls.get("revision") == 1
    assert any(r.get("intervened") for r in run.risk)
    assert run.seat_calls.get("unscripted", 0) == 0 and not run.complaints
    # the runner is left exactly as it was found
    assert [getattr(runner, name) for name in patched] == real


def test_a_planted_settlement_defect_is_caught(
        market: Market, instruments: dict[str, Any], teacher: dict[str, Decision]) -> None:
    run = replay("sabotaged", teacher, market, instruments,
                 sabotage=rh.settle_on_decision_prices)
    failed = {c.name for c in validate(run, market, oracle_run=True) if not c.passed}
    assert "each trade settles at the gold gross and net P&L, 25 hours later" in failed
    assert "every oracle trade is scored the right way" in failed


def test_a_human_baseline_runs_on_the_same_harness(
        market: Market, instruments: dict[str, Any], tmp_path: Path) -> None:
    rows = [{"symbol": "AAPLUSDT", "verdict": "trade", "side": "buy", "quantity": "5",
             "invalidation": ["AAPL closes below its entry"]},
            {"symbol": "MSFTUSDT", "verdict": "no_trade", "lean": "up", "lean_confidence": 0.6},
            {"symbol": "NOTAVENUEUSDT", "verdict": "trade", "quantity": "1"}]
    path = tmp_path / "decisions.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    decisions = rh.human(path)(market, instruments)
    assert sorted(decisions) == ["AAPLUSDT", "MSFTUSDT"]  # a name not replayed is dropped
    run = replay("human", decisions, market, instruments)
    checks = validate(run, market, oracle_run=False)
    assert all(c.passed for c in checks), [c.detail for c in checks if not c.passed]
    assert run.performance["trades"] == 1 and run.performance["abstentions"] == 1


def test_a_decision_row_without_symbol_or_verdict_is_refused() -> None:
    with pytest.raises(ReplayError, match="symbol and verdict"):
        Decision.from_row({"symbol": "AAPLUSDT"})


def test_the_seat_never_answers_a_call_the_script_does_not_cover() -> None:
    seat = Seat({"AAPLUSDT": Decision("AAPLUSDT", "no_trade"),
                 "MSFTUSDT": Decision("MSFTUSDT", "no_trade")}, name="t")
    with pytest.raises(ReplayError, match="unscripted free-text"):
        seat.complete([{"role": "user", "content": "hello"}])
    with pytest.raises(ReplayError, match="unscripted call"):
        seat.complete_json([{"role": "user", "content": "x"}], required_keys=("summary",))
    two = [{"role": "user", "content": "SYMBOL: AAPLUSDT\nSYMBOL: MSFTUSDT"}]
    with pytest.raises(ReplayError, match="scripted decision for 2"):
        seat.complete_json(two, required_keys=("verdict",))
    one = [{"role": "user", "content": "SYMBOL: AAPLUSDT\nbenchmark QQQUSDT"}]
    answered = seat.complete_json(one, required_keys=("verdict",),
                                  validate=lambda r: "too vague")
    assert answered["verdict"] == "NO_TRADE"
    assert seat.complaints == ["too vague"] and seat.calls == {"decision": 1, "unscripted": 2}


def test_a_trade_a_gate_stopped_is_named_and_reported(
        market: Market, instruments: dict[str, Any]) -> None:
    """Twelve orders in one cycle meet the venue gate's ten-per-minute cap; the two it denies are
    booked as no trade, and the replay says which gate stopped them and that they are then graded
    as abstentions the decision-maker did not choose."""
    run = replay("buy-all", rh.buy_everything(market, instruments), market, instruments)
    stopped = rh.gate_reasons(run)
    assert len(stopped) == 2
    assert all(why and why[0].startswith("[guard] DENIED rate_limit") for why in stopped.values())
    assert all(c.passed for c in validate(run, market, oracle_run=False))
    finding = rh._gate_findings("buy-all", run)
    assert len(finding) == 1 and "2 scripted trade(s)" in finding[0]
    assert "2 graded with a counterfactual move" in finding[0]
    assert not run.data_writes
