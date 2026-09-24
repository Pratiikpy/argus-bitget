"""Factor-lab tests — the separation must be structural, not a convention."""

from __future__ import annotations

from dataclasses import fields
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from argus.backtest.engine import Bar
from argus.research.factor_lab import (
    PRIMITIVES,
    Evaluator,
    Factor,
    FactorLab,
    FactorRecord,
    Lifecycle,
    LifecycleViolation,
    ProposerContext,
)

T0 = datetime(2026, 6, 1, tzinfo=UTC)


def _bars(n: int = 600) -> list[Bar]:
    """A steady up-drift.

    Chosen so the cost gate is exercised deterministically rather than by luck: any signal that
    holds through it has a positive gross Sharpe, and any signal that repeatedly enters and exits
    pays the 12bps round trip often enough to turn that negative. An earlier sawtooth fixture
    produced neither reliably, so the test that depended on it passed or failed on the shape of
    the toy data rather than on the mechanism.
    """
    out: list[Bar] = []
    price = Decimal("200")
    for i in range(n):
        # Drift plus a deterministic oscillation. Pure drift has almost no variance, which makes
        # Sharpe ratios enormous and unrepresentative; markets have both.
        drift = Decimal("1.0002")
        wobble = Decimal("1") + Decimal((i * 7) % 13 - 6) / Decimal("2000")
        price = price * drift * wobble
        out.append(Bar(ts=T0 + timedelta(hours=i), close=price.quantize(Decimal("0.0001"))))
    return out


class TestSeparationIsStructural:
    def test_the_proposer_context_has_no_field_for_performance(self) -> None:
        """Contaminating the loop must require editing a frozen dataclass, visible in a diff.

        This is the defect in mcts-llm-alpha, RD-Agent and FactorForge: all three route a score
        back to the thing that generated the candidate.
        """
        names = {f.name for f in fields(ProposerContext)}
        for forbidden in ("sharpe", "ic", "score", "performance", "pnl", "returns", "best"):
            assert not any(forbidden in n for n in names), f"{forbidden} reachable by the proposer"

    def test_context_carries_only_structure_and_a_count(self) -> None:
        lab = FactorLab(evaluator=Evaluator(_bars()))
        ctx = lab.context()
        assert ctx.trials_so_far == 0
        assert "attenuates" in ctx.market_structure
        assert ctx.already_proposed == ()

    def test_the_evaluator_refuses_unvetted_expressions(self) -> None:
        """Model-authored code in an evaluator is an arbitrary-execution surface."""
        with pytest.raises(ValueError, match="never model-authored code"):
            Factor(name="x", expression="__import__('os').system('rm -rf /')", rationale="r")


class TestLifecycle:
    def test_a_factor_cannot_skip_a_gate(self) -> None:
        r = FactorRecord(factor=Factor("f", "slow_trend", "r"), trial_number=1)
        with pytest.raises(LifecycleViolation, match="skips a gate"):
            r.advance(Lifecycle.CERTIFIED)

    def test_rejection_is_reachable_from_anywhere(self) -> None:
        r = FactorRecord(factor=Factor("f", "slow_trend", "r"), trial_number=1)
        r.reject("cost")
        assert r.state is Lifecycle.REJECTED

    def test_retire_exists_and_is_reachable(self) -> None:
        """A library that only ever grows is lying about decay."""
        assert Lifecycle.RETIRED in set(Lifecycle)
        r = FactorRecord(factor=Factor("f", "slow_trend", "r"), trial_number=1)
        r.advance(Lifecycle.RETIRED)
        assert r.state is Lifecycle.RETIRED

    def test_the_path_is_recorded(self) -> None:
        r = FactorRecord(factor=Factor("f", "slow_trend", "r"), trial_number=1)
        r.advance(Lifecycle.FORMALIZED)
        r.advance(Lifecycle.BACKTESTED)
        assert [f"{a}->{b}" for a, b in r.history] == [
            "proposed->formalized", "formalized->backtested",
        ]


class TestTrialCounting:
    def test_every_proposal_counts_including_the_failures(self) -> None:
        """A search that reports its best without its trial count has reported the maximum of a
        noise distribution."""
        lab = FactorLab(evaluator=Evaluator(_bars()))
        for name in list(PRIMITIVES)[:4]:
            lab.submit(Factor(name=name, expression=name, rationale="r"))
        assert lab.trials == 4
        assert lab.gate()["trials"] == 4

    def test_the_cemetery_records_what_killed_each_factor(self) -> None:
        lab = FactorLab(evaluator=Evaluator(_bars()))
        for name in list(PRIMITIVES)[:4]:
            lab.submit(Factor(name=name, expression=name, rationale="r"))
        lab.gate()
        dead = lab.cemetery()
        assert dead
        assert all(row["reason"] for row in dead)


class TestCostGate:
    def test_a_positive_gross_signal_dies_on_fees(self) -> None:
        """The whole point of the lab, made deterministic.

        On a steady up-drift, a signal that enters and exits at every session boundary has a
        positive gross Sharpe and pays the 12bps round trip every time. Measured on real NVDA
        data the same thing happens to `closure_reversion`: gross 2.433, net -4.003.
        """
        lab = FactorLab(evaluator=Evaluator(_bars()))
        for name in PRIMITIVES:
            lab.submit(Factor(name=name, expression=name, rationale="r"))

        scored = [r for r in lab.records if r.gross_sharpe is not None]
        assert scored, "nothing scored"

        killed = [
            r for r in scored
            if "after the 12bps round trip" in r.rejection_reason and (r.gross_sharpe or 0) > 0
        ]
        assert killed, (
            "expected a factor with positive gross and negative net; got "
            + str([(r.factor.name, r.gross_sharpe, r.net_sharpe) for r in scored])
        )

    def test_the_fee_always_reduces_total_return(self) -> None:
        """The real invariant, and not the one I first wrote.

        "Net Sharpe <= gross Sharpe" is **false**. Sharpe is mean/stdev, and trading cost adds
        variance as well as subtracting mean; for a strongly negative Sharpe the added variance
        pulls the ratio *toward* zero. Measured here: closure_reversion scored gross -171.159 and
        net -114.009 — the cost model was applied correctly and the ratio still rose.

        Total return carries no such subtlety: cost is money out, every time.
        """
        from argus.backtest.engine import run as run_backtest
        from argus.backtest.metrics import HOURLY_PER_YEAR
        from argus.cost.model import CostModel
        from argus.research.factor_lab import PRIMITIVES as P

        bars = _bars()
        cost = CostModel.bitget_perp()
        for name, fn in P.items():
            result = run_backtest(
                name, "lab", bars, fn, cost=cost, periods_per_year=HOURLY_PER_YEAR
            )
            assert result.net.total_return <= result.gross.total_return, (
                f"{name}: net {result.net.total_return} exceeded gross "
                f"{result.gross.total_return} — the fee was not charged"
            )
