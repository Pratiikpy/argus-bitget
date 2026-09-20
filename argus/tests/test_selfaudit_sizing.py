"""Self-audit and position-sizing tests.

Two properties, one per module.

For the audit: an undersized sample must read as *inconclusive*, never as clean. "We looked and
found nothing wrong" and "there was not enough to look at" are different statements, and only one
of them is a reason to publish.

For sizing: confidence may not set position size until it has been shown to track outcomes. Sizing
on a miscalibrated probability does not express an edge, it levers a bias — so the gate is the
safety property and the Kelly arithmetic is the easy part.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from argus.eval.observatory import Prediction
from argus.eval.selfaudit import (
    Severity,
    audit,
    check_calibration_divergence,
    check_drawdown_widening,
    check_lookahead,
    check_sharpe_collapse,
)
from argus.paper.ledger import Entry, PaperLedger
from argus.risk.sizing import (
    FIXED_FRACTION,
    MAX_FRACTION,
    MIN_GRADED,
    calibration_gate,
    kelly_fraction,
    size,
)

START = datetime(2026, 9, 1, 14, 0, tzinfo=UTC)


def _entry(seq: int, *, decided: datetime, settled: datetime | None = None,
           net_pnl: str | None = None, confidence: float = 0.7) -> Entry:
    return Entry(
        seq=seq, decided_at=decided.isoformat(), symbol="NVDAUSDT", verdict="open_long",
        side="BUY", quantity="1", entry_price="100", stated_confidence=confidence,
        thesis="fixture", invalidation=(), market_state_hash="m" * 16,
        approved_intent_hash="a" * 16, session_phase="rth", hours_to_discovery=0.0,
        entry_cost_bps="6", prev_hash="0" * 16,
        settled_at=settled.isoformat() if settled else None,
        exit_price="101" if settled else None, gross_pnl=net_pnl, net_pnl=net_pnl,
        direction_correct=(Decimal(net_pnl) > 0) if (settled and net_pnl) else None,
    )


def _ledger(entries: list[Entry], tmp_path: Path) -> PaperLedger:
    import json
    from dataclasses import asdict

    path = tmp_path / "audit.jsonl"
    path.write_text(
        "\n".join(json.dumps(asdict(e), default=str) for e in entries) + "\n", encoding="utf-8"
    )
    return PaperLedger(path=path)


def _predictions(n: int, *, confidence: float, hit_rate: float) -> list[Prediction]:
    return [
        Prediction(confidence=confidence, correct=(i < int(n * hit_rate))) for i in range(n)
    ]


class TestAnUndersizedSampleIsInconclusiveNotClean:
    """The departure from the reference implementation."""

    def test_sharpe_collapse_on_a_short_series(self) -> None:
        finding = check_sharpe_collapse([0.01] * 6)
        assert finding.severity is Severity.INCONCLUSIVE
        assert "needed" in finding.message

    def test_drawdown_on_a_short_series(self) -> None:
        assert check_drawdown_widening([0.01] * 5).severity is Severity.INCONCLUSIVE

    def test_calibration_on_too_few_graded_outcomes(self) -> None:
        finding = check_calibration_divergence(_predictions(6, confidence=0.7, hit_rate=0.7))
        assert finding.severity is Severity.INCONCLUSIVE

    def test_the_report_refuses_to_call_that_publishable(self) -> None:
        """A result nobody could check is not a clean result."""
        from argus.eval.selfaudit import AuditReport, Finding

        report = AuditReport(findings=(
            Finding("X", Severity.INCONCLUSIVE, "not enough data"),
        ))
        assert report.publishable is False
        assert "not enough data to audit" in report.verdict


class TestItCatchesAFlatteredResult:
    def _collapsing(self) -> list[float]:
        """Strong and steady, then flat and noisy — the shape of a fitted result."""
        strong = [0.02, 0.021, 0.019, 0.022, 0.02, 0.021, 0.02, 0.019, 0.021, 0.02] * 3
        weak = [0.03, -0.031, 0.02, -0.025, 0.028, -0.03, 0.01, -0.012, 0.02, -0.022]
        return strong + weak

    def test_a_sharpe_collapse_is_flagged_critical(self) -> None:
        finding = check_sharpe_collapse(self._collapsing())
        assert finding.severity in {Severity.CRITICAL, Severity.WARNING}
        assert finding.detail["in_sample_sharpe"] > finding.detail["out_of_sample_sharpe"]

    def test_a_stable_series_is_clean(self) -> None:
        steady = [0.01, -0.004, 0.008, -0.003, 0.009, -0.005] * 8
        assert check_sharpe_collapse(steady).severity is Severity.CLEAN

    def test_widening_drawdown_is_flagged(self) -> None:
        calm = [0.004, -0.002] * 12
        rout = [-0.05, -0.04, -0.03, -0.05, -0.02, -0.04, -0.03, -0.05, -0.02, -0.03]
        finding = check_drawdown_widening(calm + rout)
        assert finding.severity is Severity.CRITICAL
        assert finding.detail["out_of_sample_drawdown"] > finding.detail["in_sample_drawdown"]

    def test_calibration_drift_is_flagged(self) -> None:
        honest = _predictions(30, confidence=0.7, hit_rate=0.7)
        deluded = _predictions(20, confidence=0.95, hit_rate=0.35)
        finding = check_calibration_divergence(honest + deluded)
        assert finding.severity is Severity.CRITICAL


class TestImpossibleOrderingVoidsTheRow:
    def test_settling_before_deciding_is_critical(self) -> None:
        rows = [_entry(1, decided=START, settled=START - timedelta(hours=3), net_pnl="10")]
        finding = check_lookahead(rows)
        assert finding.severity is Severity.CRITICAL
        assert "void" in finding.message

    def test_a_duplicate_sequence_number_is_caught(self) -> None:
        rows = [_entry(1, decided=START), _entry(1, decided=START + timedelta(hours=1))]
        assert check_lookahead(rows).severity is Severity.CRITICAL

    def test_a_well_ordered_log_is_clean(self) -> None:
        rows = [
            _entry(i, decided=START + timedelta(hours=i),
                   settled=START + timedelta(hours=i + 24), net_pnl="10")
            for i in range(1, 5)
        ]
        assert check_lookahead(rows).severity is Severity.CLEAN


class TestTheAuditRunsOnARealLedger:
    def test_an_all_abstention_log_audits_as_inconclusive_not_clean(
        self, tmp_path: Path
    ) -> None:
        """The live ledger's actual state: nothing settled, so nothing to audit."""
        rows = [_entry(i, decided=START + timedelta(hours=2 * i)) for i in range(1, 10)]
        report = audit(_ledger(rows, tmp_path))
        assert report.publishable is False
        assert report.inconclusive

    def test_the_report_serialises_every_finding(self, tmp_path: Path) -> None:
        rows = [_entry(i, decided=START + timedelta(hours=2 * i)) for i in range(1, 6)]
        payload = audit(_ledger(rows, tmp_path)).as_dict()
        assert len(payload["findings"]) == 4
        assert {"code", "severity", "message"} <= set(payload["findings"][0])


class TestTheCalibrationGate:
    """The safety property. The Kelly arithmetic is the easy part."""

    def test_too_few_graded_outcomes_refuses_the_gate(self) -> None:
        gate = calibration_gate(_predictions(MIN_GRADED - 1, confidence=0.7, hit_rate=0.7))
        assert gate.passed is False
        assert "needed before confidence" in gate.reason

    def test_poor_calibration_refuses_the_gate(self) -> None:
        """Says 0.95, right 35% of the time."""
        gate = calibration_gate(_predictions(40, confidence=0.95, hit_rate=0.35))
        assert gate.passed is False
        assert "lever a bias" in gate.reason

    def test_good_calibration_passes(self) -> None:
        gate = calibration_gate(_predictions(40, confidence=0.7, hit_rate=0.7))
        assert gate.passed is True
        assert gate.ece is not None and gate.ece <= 0.15

    def test_the_two_failure_modes_are_distinguishable(self) -> None:
        """Too little data is a wait; bad calibration is a fix. Different responses."""
        thin = calibration_gate(_predictions(5, confidence=0.7, hit_rate=0.7))
        bad = calibration_gate(_predictions(40, confidence=0.95, hit_rate=0.3))
        assert thin.ece is None, "no ECE is reported on a sample too small to compute one"
        assert bad.ece is not None


class TestSizing:
    def test_an_ungated_desk_falls_back_to_the_fixed_fraction(self) -> None:
        result = size(
            win_probability=0.9, payoff=Decimal("2"),
            predictions=_predictions(3, confidence=0.9, hit_rate=0.9),
        )
        assert result.fraction == FIXED_FRACTION
        assert "not usable" in result.basis

    def test_a_calibrated_desk_sizes_on_kelly(self) -> None:
        result = size(
            win_probability=0.7, payoff=Decimal("2"),
            predictions=_predictions(40, confidence=0.7, hit_rate=0.7),
        )
        assert "half-Kelly" in result.basis
        assert result.raw_kelly is not None and result.raw_kelly > 0

    def test_the_stake_is_capped_however_good_the_arithmetic_looks(self) -> None:
        result = size(
            win_probability=0.99, payoff=Decimal("50"),
            predictions=_predictions(40, confidence=0.7, hit_rate=0.7),
        )
        assert result.fraction <= MAX_FRACTION

    def test_the_breaker_multiplier_is_applied_last(self) -> None:
        """A halted book sizes to zero however good the edge looks."""
        result = size(
            win_probability=0.8, payoff=Decimal("3"),
            predictions=_predictions(40, confidence=0.7, hit_rate=0.7),
            risk_multiplier=Decimal("0"),
        )
        assert result.fraction == Decimal("0")

    def test_the_multiplier_also_binds_the_fallback(self) -> None:
        result = size(
            win_probability=0.8, payoff=Decimal("3"),
            predictions=[], risk_multiplier=Decimal("0"),
        )
        assert result.fraction == Decimal("0")


class TestKellyArithmetic:
    def test_a_fair_coin_at_even_money_stakes_nothing(self) -> None:
        assert kelly_fraction(0.5, Decimal("1")) == Decimal("0")

    def test_an_edge_produces_a_positive_stake(self) -> None:
        assert kelly_fraction(0.6, Decimal("1")) == pytest.approx(Decimal("0.2"))

    def test_a_negative_edge_floors_at_zero_rather_than_reversing(self) -> None:
        """A negative Kelly is an instruction to take the other side; this function sizes a
        decision that has already been made."""
        assert kelly_fraction(0.3, Decimal("1")) == Decimal("0")

    def test_a_non_positive_payoff_stakes_nothing(self) -> None:
        assert kelly_fraction(0.9, Decimal("0")) == Decimal("0")

    def test_an_impossible_probability_raises(self) -> None:
        with pytest.raises(ValueError, match="not a probability"):
            kelly_fraction(1.4, Decimal("2"))

    def test_more_edge_stakes_more(self) -> None:
        stakes = [kelly_fraction(p, Decimal("2")) for p in (0.4, 0.5, 0.6, 0.7, 0.8)]
        assert stakes == sorted(stakes)
