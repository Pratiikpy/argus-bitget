"""Tests for `argus.desk.rotation` — checked against pytaa's real, vendored VAA breadth rule.

No live network calls here — all cases use constructed monthly closes and momentum scores. The
live-data comparison (real Bitget cross-asset history, real pytaa code, run side by side) is
`tests/test_rotation_comparison.py`.
"""

from __future__ import annotations

import pytest

from argus.desk.allocation import TAKER_BPS
from argus.desk.rotation import (
    MIN_MONTHLY_OBSERVATIONS,
    MOMENTUM_HORIZONS,
    MOMENTUM_WEIGHT_SUM,
    RotationError,
    breadth_allocation,
    momentum_score,
    monthly_closes_from_bars,
    propose_rotation,
)


class TestMomentumScore:
    def test_refuses_below_the_minimum(self) -> None:
        with pytest.raises(RotationError, match="below the 13"):
            momentum_score([100.0] * (MIN_MONTHLY_OBSERVATIONS - 1))

    def test_accepts_exactly_the_minimum(self) -> None:
        closes = [100.0] * MIN_MONTHLY_OBSERVATIONS
        assert momentum_score(closes) == pytest.approx(0.0)

    def test_flat_prices_score_exactly_zero(self) -> None:
        """Every ratio in the formula is 1 at flat prices, and the weights sum to 19 by
        construction (`MOMENTUM_WEIGHT_SUM`), so the normalised score is exactly zero."""
        closes = [50.0] * 20
        assert momentum_score(closes) == pytest.approx(0.0)
        assert sum(w for w, _ in MOMENTUM_HORIZONS) == MOMENTUM_WEIGHT_SUM

    def test_uniform_growth_scores_positive(self) -> None:
        closes = [100.0 * (1.02 ** i) for i in range(20)]
        assert momentum_score(closes) > 0

    def test_uniform_decline_scores_negative(self) -> None:
        closes = [100.0 * (0.98 ** i) for i in range(20)]
        assert momentum_score(closes) < 0

    def test_refuses_a_zero_anchor(self) -> None:
        closes = [0.0] + [100.0] * (MIN_MONTHLY_OBSERVATIONS - 1)
        with pytest.raises(RotationError, match="zero monthly close"):
            momentum_score(closes)


class TestMonthlyClosesFromBars:
    def test_takes_the_last_close_of_each_month(self) -> None:
        from datetime import datetime

        bars = [
            (datetime(2026, 1, 5), 10.0),
            (datetime(2026, 1, 20), 11.0),
            (datetime(2026, 2, 3), 12.0),
            (datetime(2026, 2, 28), 13.0),
        ]
        assert monthly_closes_from_bars(bars) == [11.0, 13.0]

    def test_empty_input_gives_empty_output(self) -> None:
        assert monthly_closes_from_bars([]) == []


class TestBreadthAllocation:
    def test_reproduces_the_real_vendored_function_on_clean_input(self) -> None:
        """Pinned against `vigilant_allocation`'s real, measured output on this exact input
        (`eval/rotation_comparison.py`'s own smoke check)."""
        scores = {"A": 0.05, "B": -0.02, "C": 0.01, "SAFE": 0.03}
        out = breadth_allocation(scores, risk_assets=["A", "B", "C"], safe_assets=["SAFE"],
                                  top_k=2, step=0.25)
        assert out == pytest.approx({"A": 0.375, "B": 0.0, "C": 0.375, "SAFE": 0.25})

    def test_refuses_a_missing_safe_asset_score(self) -> None:
        scores = {"A": 0.1, "B": -0.1, "SAFE": float("nan")}
        with pytest.raises(RotationError, match="SAFE"):
            breadth_allocation(scores, risk_assets=["A", "B"], safe_assets=["SAFE"])

    def test_refuses_where_the_real_function_silently_underallocates(self) -> None:
        """The real vendored `vigilant_allocation`, fed this exact input, allocates 25% and
        leaves 75% of the book nowhere (see `eval/rotation_comparison.py`'s
        `run_silent_data_loss_cases`). This must raise instead."""
        scores = {"A": -0.1, "B": -0.2, "C": -0.05, "SAFE": float("nan")}
        with pytest.raises(RotationError):
            breadth_allocation(scores, risk_assets=["A", "B", "C"], safe_assets=["SAFE"], top_k=2)

    def test_refuses_empty_safe_assets(self) -> None:
        with pytest.raises(RotationError, match="safe asset"):
            breadth_allocation({"A": 0.1, "B": -0.1}, risk_assets=["A", "B"], safe_assets=[])

    def test_refuses_empty_risk_assets(self) -> None:
        with pytest.raises(RotationError, match="risk asset"):
            breadth_allocation({"SAFE": 0.1}, risk_assets=[], safe_assets=["SAFE"])

    def test_weights_sum_to_one_on_valid_input(self) -> None:
        scores = {"A": 0.3, "B": 0.2, "C": -0.1, "D": -0.4, "SAFE": 0.05}
        out = breadth_allocation(scores, risk_assets=["A", "B", "C", "D"], safe_assets=["SAFE"],
                                  top_k=3)
        assert sum(out.values()) == pytest.approx(1.0)

    def test_all_risk_positive_is_fully_risk_on(self) -> None:
        scores = {"A": 0.1, "B": 0.2, "SAFE": 0.05}
        out = breadth_allocation(scores, risk_assets=["A", "B"], safe_assets=["SAFE"], top_k=2)
        assert out["SAFE"] == 0.0
        assert sum(out.values()) == pytest.approx(1.0)


class TestProposeRotation:
    def _monthly(self, growth: float) -> list[float]:
        return [100.0 * ((1.0 + growth) ** i) for i in range(MIN_MONTHLY_OBSERVATIONS)]

    def test_proposes_trades_toward_the_breadth_target(self) -> None:
        monthly = {
            "RISK1": self._monthly(0.03),
            "RISK2": self._monthly(0.02),
            "SAFE": self._monthly(-0.01),
        }
        current = {"RISK1": 0.5, "RISK2": 0.5, "SAFE": 0.0}
        plan = propose_rotation(current, monthly, risk_assets=["RISK1", "RISK2"],
                                 safe_assets=["SAFE"], top_k=2)
        assert plan.universe_size == 3
        assert plan.cost_bps == pytest.approx(plan.turnover * TAKER_BPS)

    def test_no_trade_when_book_already_matches_target(self) -> None:
        monthly = {
            "RISK1": self._monthly(0.03),
            "RISK2": self._monthly(0.02),
            "SAFE": self._monthly(-0.01),
        }
        current = {"RISK1": 0.5, "RISK2": 0.5, "SAFE": 0.0}
        plan = propose_rotation(current, monthly, risk_assets=["RISK1", "RISK2"],
                                 safe_assets=["SAFE"], top_k=2)
        plan_again = propose_rotation(
            {t.symbol: t.weight_after for t in plan.trades} | {
                a: current.get(a, 0.0) for a in ("RISK1", "RISK2", "SAFE")
                if a not in {t.symbol for t in plan.trades}
            },
            monthly, risk_assets=["RISK1", "RISK2"], safe_assets=["SAFE"], top_k=2,
        )
        assert plan_again.trades == ()
        assert "no trade" in plan_again.verdict

    def test_refuses_when_a_series_is_missing(self) -> None:
        monthly = {"RISK1": self._monthly(0.01), "SAFE": self._monthly(-0.01)}
        with pytest.raises(RotationError, match="RISK2"):
            propose_rotation({}, monthly, risk_assets=["RISK1", "RISK2"], safe_assets=["SAFE"])

    def test_refuses_when_history_is_too_short(self) -> None:
        monthly = {
            "RISK1": self._monthly(0.01)[:-1],
            "SAFE": self._monthly(-0.01),
        }
        with pytest.raises(RotationError):
            propose_rotation({}, monthly, risk_assets=["RISK1"], safe_assets=["SAFE"])

    def test_regime_labels(self) -> None:
        monthly = {
            "RISK1": self._monthly(0.05),
            "RISK2": self._monthly(0.04),
            "SAFE": self._monthly(0.01),
        }
        plan = propose_rotation({}, monthly, risk_assets=["RISK1", "RISK2"], safe_assets=["SAFE"],
                                 top_k=2)
        assert "risk-on" in plan.regime

    def test_as_dict_is_json_serialisable(self) -> None:
        import json

        monthly = {
            "RISK1": self._monthly(0.03),
            "SAFE": self._monthly(-0.02),
        }
        plan = propose_rotation({}, monthly, risk_assets=["RISK1"], safe_assets=["SAFE"], top_k=1)
        json.dumps(plan.as_dict())
