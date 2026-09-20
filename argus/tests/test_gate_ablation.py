"""Population RCT (Track 2 §13.7a) — pins that each Foundation 5 dimension's ablation is isolated
(disturbs only itself), that the risk-violation reading follows the Constitution's narrow-only
invariant, and that `eval.ablation.deterministic` — not a second comparison engine — is what
actually runs underneath."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from argus.agents.desk import ConstitutionPolicy
from argus.decision.verdicts import Intent, Side, Verdict
from argus.desk.book import Book, HedgeLink, Lot, PositionSide
from argus.eval.gate_ablation import (
    DIMENSIONS,
    ablated_variants,
    population_rct,
    report_from_records,
)
from argus.risk.hedgeability import HedgeabilitySurface
from argus.truth.clocks import SessionPhase, SessionState

AT = datetime(2026, 9, 15, tzinfo=UTC)


def _session() -> SessionState:
    return SessionState(
        phase=SessionPhase.RTH, as_of=AT, hours_to_next_discovery=0.0, nav_age_seconds=10.0,
    )


def _hedges_available() -> HedgeabilitySurface:
    class _Available:
        is_empty = False
        menu: tuple[object, ...] = (object(),)
    return _Available()  # type: ignore[return-value]


def _trade(qty: str, *, symbol: str = "NVDAUSDT", side: Side = Side.BUY) -> Intent:
    return Intent(
        symbol=symbol, side=side, quantity=Decimal(qty), verdict=Verdict.TRADE,
        stated_confidence=0.9, thesis="t", invalidation=("x",),
    )


class TestAblatedVariantsAreIsolated:
    """Ablating one dimension must not change what any other gate reads — the exact defect
    `existing_gross`/`hedge_at_risk` reuse shipped in `eval/riskproof.py` earlier the same
    session, checked here at the Constitution-policy level instead of the sweep level."""

    def test_every_dimension_produces_a_distinct_policy(self) -> None:
        """Needs a baseline where every dimension is actually *active* — ablating a cap nobody
        configured (the bare default, e.g. `factor_exposure_limits={}`) is correctly a no-op, not
        a bug, and a from-defaults baseline caught exactly that for `hedge_integrity` (no book)
        and `factor_exposure` (no limits) while writing this test. Every field below is set to a
        real, non-default value for that reason."""
        from argus.desk.portfolio import FactorExposure, StressOutcome
        from argus.market.depth import Sweep

        book = Book()
        book.apply_fill(
            symbol="NVDAUSDT", side=PositionSide.LONG,
            lot=Lot(order_id="o1", approved_intent_hash="a1", side="buy",
                    quantity=Decimal("10"), price=Decimal("100"), commission=Decimal("0"),
                    ts_filled=AT),
        )
        book.link_hedge(HedgeLink(
            source=("NVDAUSDT", PositionSide.LONG), target=("SQQQUSDT", PositionSide.SHORT),
            relationship="HEDGE", hedge_ratio=None, ts_created=AT,
        ))
        baseline = ConstitutionPolicy(
            book=book,
            factor_exposures=(FactorExposure(factor="market", exposure=0.9, observations=100),),
            factor_exposure_limits={"market": 0.5},
            stress_outcomes=(
                StressOutcome(shock="-10%", portfolio_move_pct=-9.0, worst_position=None),
            ),
            max_scenario_loss_pct=-5.0,
            liquidation_cost_estimates={
                "NVDAUSDT": Sweep(
                    side="SELL", requested_notional=Decimal("10000"),
                    filled_notional=Decimal("10000"), average_price=Decimal("100"),
                    levels_consumed=5, slippage_bps=Decimal("900"), complete=True,
                ),
            },
            max_liquidation_cost_bps=Decimal("500"),
            min_symbol_realized_pnl=Decimal("-100"),
        )
        variants = ablated_variants(baseline)
        assert set(variants) == set(DIMENSIONS)
        for dimension, variant in variants.items():
            assert variant != baseline, f"{dimension} did not change the policy at all"

    def test_ablating_gross_exposure_leaves_signed_exposure_untouched(self) -> None:
        baseline = ConstitutionPolicy(max_signed_exposure_notional=Decimal("123"))
        variant = ablated_variants(baseline, dimensions=("gross_exposure",))["gross_exposure"]
        assert variant.max_signed_exposure_notional == Decimal("123")
        assert variant.max_gross_exposure_notional != baseline.max_gross_exposure_notional

    def test_ablating_hedge_integrity_keeps_the_positions_and_margin(self) -> None:
        book = Book()
        book.apply_fill(
            symbol="NVDAUSDT", side=PositionSide.LONG,
            lot=Lot(order_id="o1", approved_intent_hash="a1", side="buy",
                    quantity=Decimal("10"), price=Decimal("100"), commission=Decimal("0"),
                    ts_filled=AT),
        )
        book.link_hedge(HedgeLink(
            source=("NVDAUSDT", PositionSide.LONG), target=("SQQQUSDT", PositionSide.SHORT),
            relationship="HEDGE", hedge_ratio=None, ts_created=AT,
        ))
        baseline = ConstitutionPolicy(book=book)
        variant = ablated_variants(baseline, dimensions=("hedge_integrity",))["hedge_integrity"]
        assert variant.book.hedge_links == []
        assert variant.book.positions == book.positions
        assert variant.book.total_gross_notional() == book.total_gross_notional()
        # And the original is untouched — this is a copy, not a mutation in place.
        assert len(book.hedge_links) == 1

    def test_ablating_hedge_integrity_with_no_book_stays_none(self) -> None:
        variant = ablated_variants(ConstitutionPolicy(), dimensions=("hedge_integrity",))[
            "hedge_integrity"
        ]
        assert variant.book is None

    def test_an_unknown_dimension_raises_rather_than_silently_no_oping(self) -> None:
        with pytest.raises(ValueError, match="unknown dimension"):
            ablated_variants(ConstitutionPolicy(), dimensions=("not_a_real_gate",))


class TestPopulationRct:
    def test_order_size_shows_up_as_a_violation_when_it_is_the_only_binder(self) -> None:
        session, hedges = _session(), _hedges_available()
        baseline = ConstitutionPolicy(max_position_notional=Decimal("100"))
        frames = [("f1", _trade("101"))]
        results = population_rct(
            frames, session=session, hedges=hedges, baseline=baseline,
            dimensions=("order_size",),
        )
        assert results["order_size"].frames == 1
        assert results["order_size"].risk_violations == 1
        assert results["order_size"].violation_rate == 1.0
        assert not results["order_size"].inert

    def test_a_dimension_no_frame_ever_reaches_is_reported_inert_not_hidden(self) -> None:
        session, hedges = _session(), _hedges_available()
        baseline = ConstitutionPolicy(max_position_notional=Decimal("999999"))
        frames = [("f1", _trade("1"))]
        results = population_rct(
            frames, session=session, hedges=hedges, baseline=baseline,
            dimensions=("order_size",),
        )
        assert results["order_size"].inert
        assert results["order_size"].risk_violations == 0

    def test_every_dimension_runs_over_the_same_frames_independently(self) -> None:
        session, hedges = _session(), _hedges_available()
        frames = [("f1", _trade("100")), ("f2", _trade("50", side=Side.SELL))]
        results = population_rct(
            frames, session=session, hedges=hedges, baseline=ConstitutionPolicy(),
        )
        assert set(results) == set(DIMENSIONS)
        assert all(r.frames == 2 for r in results.values())

    def test_a_quantity_exactly_at_the_cap_does_not_bind_on_either_side(self) -> None:
        """`max_position` is a strict `>`, so the boundary value itself must not read as a
        violation on either the full policy or the ablated one."""
        session, hedges = _session(), _hedges_available()
        baseline = ConstitutionPolicy(max_position_notional=Decimal("50"))
        frames = [("f1", _trade("50"))]
        results = population_rct(
            frames, session=session, hedges=hedges, baseline=baseline,
            dimensions=("order_size",),
        )
        assert results["order_size"].inert

    def test_as_dict_carries_the_violation_rate(self) -> None:
        session, hedges = _session(), _hedges_available()
        baseline = ConstitutionPolicy(max_position_notional=Decimal("1"))
        frames = [("f1", _trade("2")), ("f2", _trade("1"))]
        results = population_rct(
            frames, session=session, hedges=hedges, baseline=baseline,
            dimensions=("order_size",),
        )
        record = results["order_size"].as_dict()
        assert record["risk_violations"] == 1
        assert record["violation_rate"] == 0.5


class TestReportFromRecords:
    """Reads `paper.runner._write_risk_record`'s persisted `ablated_rulings` — no re-ruling, the
    real and ablated `quantity_after`/`binding_constraint` are already on the row."""

    @staticmethod
    def _row(seq: int, *, real_qty: str, real_bind: str, ablated_qty: str,
              ablated_bind: str) -> dict[str, object]:
        return {
            "seq": seq,
            "binding_constraint": real_bind,
            "quantity_after": real_qty,
            "ablated_rulings": {
                "order_size": {"binding_constraint": ablated_bind, "quantity_after": ablated_qty},
            },
        }

    def test_a_dimension_with_no_rows_is_reported_as_zero_frames(self) -> None:
        results = report_from_records([], dimensions=("order_size",))
        assert results["order_size"].frames == 0
        assert results["order_size"].inert  # deliberate: see eval/ablation.py's own contract

    def test_older_rows_with_no_ablated_rulings_key_are_excluded_not_counted_as_clean(self) -> None:
        old_row = {"seq": 1, "binding_constraint": "none", "quantity_after": "10"}
        results = report_from_records([old_row], dimensions=("order_size",))
        assert results["order_size"].frames == 0

    def test_a_row_where_ablation_widened_the_quantity_counts_as_a_violation(self) -> None:
        row = self._row(
            1, real_qty="50", real_bind="max_position",
            ablated_qty="100", ablated_bind="none",
        )
        results = report_from_records([row], dimensions=("order_size",))
        assert results["order_size"].frames == 1
        assert results["order_size"].risk_violations == 1
        assert not results["order_size"].inert

    def test_a_row_where_nothing_changed_is_not_counted_as_a_change(self) -> None:
        row = self._row(
            1, real_qty="50", real_bind="none", ablated_qty="50", ablated_bind="none",
        )
        results = report_from_records([row], dimensions=("order_size",))
        assert results["order_size"].frames == 1
        assert results["order_size"].inert
        assert results["order_size"].risk_violations == 0

    def test_a_malformed_row_does_not_crash_the_aggregation(self) -> None:
        row = {
            "seq": 1, "binding_constraint": "none", "quantity_after": "not-a-number",
            "ablated_rulings": {
                "order_size": {"binding_constraint": "none", "quantity_after": "also-bad"},
            },
        }
        results = report_from_records([row], dimensions=("order_size",))
        assert results["order_size"].frames == 1
        assert results["order_size"].risk_violations == 0
