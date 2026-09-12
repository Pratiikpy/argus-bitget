"""Scorecard tests — the loop only closes honestly if ungradeable rows stay ungraded."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from argus.eval.scorecard import score_ledger, scorecard
from argus.paper.ledger import PaperLedger

T0 = datetime(2026, 3, 9, 14, 0, tzinfo=UTC)


def _ledger(tmp_path: Path) -> PaperLedger:
    return PaperLedger(path=tmp_path / "p.jsonl")


def _trade(led: PaperLedger, *, n: int, conf: float, qty: str = "10") -> None:
    led.record(
        symbol="NVDAUSDT", verdict="trade", side="BUY",
        quantity=Decimal(qty), entry_price=Decimal("220"),
        stated_confidence=conf, thesis="t", invalidation=("x",),
        market_state_hash="h", approved_intent_hash="a",
        session_phase="rth", hours_to_discovery=0.0,
        decided_at=T0 + timedelta(hours=n),
    )


def _abstain(led: PaperLedger, *, n: int) -> None:
    led.record(
        symbol="NVDAUSDT", verdict="no_trade", side="BUY",
        quantity=Decimal("0"), entry_price=Decimal("220"),
        stated_confidence=0.9, thesis="edge does not clear 12bps",
        invalidation=(), market_state_hash="h", approved_intent_hash="a",
        session_phase="weekend", hours_to_discovery=30.0,
        decided_at=T0 + timedelta(hours=n),
    )


class TestOnlySettledRowsScore:
    def test_an_open_position_is_ungradeable_not_correct(self, tmp_path: Path) -> None:
        """Counting open positions lets a system flatter itself by holding losers."""
        led = _ledger(tmp_path)
        _trade(led, n=0, conf=0.9)
        got = score_ledger(led)
        assert got.predictions == []
        assert got.ungradeable == 1

    def test_settling_makes_it_gradeable(self, tmp_path: Path) -> None:
        led = _ledger(tmp_path)
        _trade(led, n=0, conf=0.9)
        led.settle(1, exit_price=Decimal("240"))
        got = score_ledger(led)
        assert len(got.predictions) == 1
        assert got.predictions[0].correct is True

    def test_a_losing_settled_trade_scores_as_incorrect(self, tmp_path: Path) -> None:
        led = _ledger(tmp_path)
        _trade(led, n=0, conf=0.9)
        led.settle(1, exit_price=Decimal("200"))
        assert score_ledger(led).predictions[0].correct is False


class TestAbstentions:
    def test_an_abstention_without_a_counterfactual_is_ungraded(self, tmp_path: Path) -> None:
        """Counting it as correct is the exact flattery this metric prevents."""
        led = _ledger(tmp_path)
        _abstain(led, n=0)
        got = score_ledger(led)
        assert got.abstention_outcomes == []
        assert got.abstentions == 1
        assert got.ungradeable == 1

    def test_a_counterfactual_makes_it_scoreable(self, tmp_path: Path) -> None:
        led = _ledger(tmp_path)
        _abstain(led, n=0)
        got = score_ledger(led, counterfactuals={1: Decimal("-40")})
        assert len(got.abstention_outcomes) == 1
        assert got.abstention_outcomes[0].was_right is True   # avoided a 40bps loss

    def test_abstentions_never_enter_calibration(self, tmp_path: Path) -> None:
        """A NO_TRADE has no direction to be right about."""
        led = _ledger(tmp_path)
        for i in range(4):
            _abstain(led, n=i)
        assert score_ledger(led).predictions == []


class TestFloorAndIntegrity:
    def test_below_five_graded_outcomes_no_calibration_is_reported(self, tmp_path: Path) -> None:
        """Calibration on four decisions is noise wearing a decimal point."""
        led = _ledger(tmp_path)
        for i in range(3):
            _trade(led, n=i, conf=0.8)
            led.settle(i + 1, exit_price=Decimal("240"))
        got = scorecard(led)
        assert "ece" not in got
        assert "below the floor of 5" in got["note"]

    def test_five_outcomes_produce_calibration(self, tmp_path: Path) -> None:
        led = _ledger(tmp_path)
        for i in range(6):
            _trade(led, n=i, conf=0.9)
            led.settle(i + 1, exit_price=Decimal("240") if i < 4 else Decimal("200"))
        got = scorecard(led)
        assert "ece" in got
        assert got["graded_predictions"] == 6
        assert got["accuracy_pct"] == round(100 * 4 / 6, 1)

    def test_a_tampered_ledger_is_refused_rather_than_scored(self, tmp_path: Path) -> None:
        """Scoring an edited ledger produces numbers indistinguishable from honest ones."""
        led = _ledger(tmp_path)
        for i in range(6):
            _trade(led, n=i, conf=0.8)
            led.settle(i + 1, exit_price=Decimal("240"))

        raw = led.path.read_text(encoding="utf-8").splitlines()
        rows = [json.loads(x) for x in raw if x.strip()]
        rows[1]["thesis"] = "rewritten after the fact"
        led.path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

        got = scorecard(PaperLedger(path=led.path))
        assert "error" in got
        assert "refusing to score" in got["error"]
        assert "ece" not in got

    def test_an_overconfident_run_shows_a_large_ece(self, tmp_path: Path) -> None:
        led = _ledger(tmp_path)
        for i in range(10):
            _trade(led, n=i, conf=0.95)
            led.settle(i + 1, exit_price=Decimal("240") if i < 5 else Decimal("200"))
        got = scorecard(led)
        assert got["ece"] > 0.4   # claimed 95%, landed 50%
