"""The split-half reliability gate: noise must fail, a real edge must pass, and the PPCA must be
GoEmotions' PPCA.

The planted-factor tests are the gate's reason to exist. A reliability gate that passes coin flips
is decoration, and one that fails a genuine edge is a tax on the only thing the lab is looking for.
Both are checked across many seeds rather than one, because a statistical gate that passes a single
lucky draw has not been tested.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import numpy as np
import pytest

from argus.backtest.engine import Bar
from argus.research.factor_lab import (
    SPLIT_HALF_BLOCK_BARS,
    SPLIT_HALF_MIN_BLOCKS,
    Evaluator,
    Factor,
    FactorLab,
    FactorRecord,
    Lifecycle,
    payoff_halves,
    ppca,
    reliable_components,
    split_half,
    symmetric_eigh,
)
from argus.research.overfit import Outcome


def _returns(n: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    # fat-tailed: a Student-t-like mixture, the shape hourly returns actually have
    return [rng.gauss(0, 0.004) * (3.0 if rng.random() < 0.05 else 1.0) for _ in range(n)]


def _noise(returns: list[float], rng: random.Random) -> list[float]:
    return [1.0 if rng.random() < 0.5 else -1.0 for _ in returns]


def _real(returns: list[float], rng: random.Random, accuracy: float) -> list[float]:
    return [(1.0 if r > 0 else -1.0) * (1.0 if rng.random() < accuracy else -1.0)
            for r in returns]


class TestPlantedFactors:
    def test_noise_factors_fail(self) -> None:
        rng = random.Random(1)
        passed = 0
        trials = 100
        for seed in range(trials):
            returns = _returns(2160, seed)
            passed += split_half(_noise(returns, rng), returns, block=SPLIT_HALF_BLOCK_BARS,
                                 flips=400).passed
        # alpha is 5%: 5 expected in 100, and 11 is about three binomial standard deviations out
        assert passed <= 11

    def test_planted_real_factor_passes(self) -> None:
        rng = random.Random(2)
        passed = 0
        for seed in range(40):
            returns = _returns(2160, 1000 + seed)
            passed += split_half(_real(returns, rng, 0.60), returns,
                                 block=SPLIT_HALF_BLOCK_BARS, flips=400).passed
        # measured power at this edge and block is ~95%
        assert passed >= 34

    def test_constant_edge_is_not_demeaned_away(self) -> None:
        # Every block pays the same: a demeaned (GoEmotions-exact) statistic is identically zero,
        # which is why the gate is uncentred.
        values = [1.0] * 480
        returns = [0.001 + (0.0005 if i % 2 else -0.0005) for i in range(480)]
        result = split_half(values, returns, block=24, flips=400)
        assert result.passed
        xs, ys = payoff_halves(values, returns, block=24)
        demeaned, _ = ppca([[x] for x in xs], [[y] for y in ys], demean=True)
        assert demeaned[0] == pytest.approx(0.0, abs=1e-20)

    def test_too_few_blocks_is_inconclusive_not_a_failure(self) -> None:
        returns = _returns(24 * (SPLIT_HALF_MIN_BLOCKS - 1), 3)
        result = split_half([1.0] * len(returns), returns, block=24)
        assert result.outcome is Outcome.INCONCLUSIVE
        assert not result.passed

    def test_a_factor_that_never_trades_fails(self) -> None:
        returns = _returns(480, 4)
        result = split_half([0.0] * 480, returns, block=24)
        assert result.outcome is Outcome.FAIL

    def test_verdict_is_reproducible(self) -> None:
        returns = _returns(2160, 5)
        values = _noise(returns, random.Random(9))
        assert split_half(values, returns, block=24) == split_half(values, returns, block=24)


class TestPPCAIsGoEmotions:
    @pytest.mark.parametrize("seed", range(6))
    def test_eigh_matches_numpy(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        m = rng.normal(size=(7, 7))
        sym = m + m.T
        ours, vecs = symmetric_eigh(sym.tolist())
        theirs = np.flip(np.linalg.eigh(sym)[0])
        assert ours == pytest.approx(list(theirs), abs=1e-9)
        v = np.array(vecs)
        assert np.allclose(sym @ v, v * np.array(ours), atol=1e-8)

    def test_ppca_matches_the_source_function(self) -> None:
        """`ppca.py:87-99`, transcribed with numpy, against ours on the same halves."""
        rng = np.random.default_rng(7)
        x = rng.normal(size=(40, 5))
        y = x * 0.6 + rng.normal(size=(40, 5))
        xd, yd = x - x.mean(axis=0), y - y.mean(axis=0)
        crosscov = xd.T @ yd + yd.T @ xd
        v, _ = np.linalg.eigh(crosscov)
        ours, _ = ppca(x.tolist(), y.tolist())
        assert ours == pytest.approx(list(np.flip(v)), abs=1e-8)

    def test_rejects_mismatched_halves(self) -> None:
        with pytest.raises(ValueError):
            ppca([[1.0, 2.0]], [[1.0, 2.0], [3.0, 4.0]])

    def test_reliable_components_counts_planted_dimensions(self) -> None:
        rng = random.Random(8)
        x, y = [], []
        for _ in range(80):
            shared = [rng.gauss(0.5, 1), rng.gauss(0, 1)]
            # two real dimensions in the first two columns, three pure-noise columns
            x.append([shared[0] + rng.gauss(0, 0.3), shared[1] + rng.gauss(0, 0.3),
                      *(rng.gauss(0, 1) for _ in range(3))])
            y.append([shared[0] + rng.gauss(0, 0.3), shared[1] + rng.gauss(0, 0.3),
                      *(rng.gauss(0, 1) for _ in range(3))])
        out = reliable_components(x, y, flips=200)
        assert out["reliable_components"] == 2
        assert out["dimensions"] == 5


T0 = datetime(2026, 6, 1, tzinfo=UTC)


def _bars(n: int = 900) -> list[Bar]:
    rng = random.Random(21)
    price = Decimal("200")
    out = []
    for i in range(n):
        price *= Decimal(str(1 + rng.gauss(0.0002, 0.003)))
        out.append(Bar(ts=T0 + timedelta(hours=i), close=price))
    return out


class TestLabIntegration:
    def test_evaluator_series_matches_observations(self) -> None:
        bars = _bars()
        ev = Evaluator(bars)
        factor = Factor("f", "slow_trend", "r")
        values, returns = ev.payoff_series(factor)
        obs = ev.observations(FactorRecord(factor=factor, trial_number=1))
        assert values == [o.factor for o in obs]
        assert returns == [o.forward_return for o in obs]

    @pytest.mark.parametrize(("outcome", "certified"), [
        (Outcome.PASS, True), (Outcome.FAIL, False), (Outcome.INCONCLUSIVE, False),
    ])
    def test_split_half_decides_certification_of_a_survivor(
        self, monkeypatch: pytest.MonkeyPatch, outcome: Outcome, certified: bool,
    ) -> None:
        """A factor that has cleared DSR and the anti-overfit gates is certified only if it also
        reproduces across halves; the verdict is recorded either way."""
        from argus.research import factor_lab

        lab = FactorLab(evaluator=Evaluator(_bars()))
        record = FactorRecord(factor=Factor("f", "slow_trend", "r"), trial_number=1)
        for state in (Lifecycle.FORMALIZED, Lifecycle.BACKTESTED, Lifecycle.COST_CHECKED,
                      Lifecycle.OOS_TESTED):
            record.advance(state)
        record.net_sharpe = 3.0
        lab.records.append(record)
        monkeypatch.setattr(factor_lab, "deflated_sharpe", lambda *a, **k: 0.99)
        monkeypatch.setattr(Evaluator, "overfit_report", lambda self, r: None)
        verdict = factor_lab.SplitHalf(outcome, 18, 1.0, 0.5, 0.01, "planted")
        monkeypatch.setattr(Evaluator, "split_half", lambda self, f: verdict)
        gate = lab.gate()
        assert (record.state is Lifecycle.CERTIFIED) is certified
        assert gate["certified"] == (["f"] if certified else [])
        assert record.split_half == verdict.as_dict()
        if not certified:
            assert record.rejection_reason.startswith("split-half")

    def test_real_gate_records_split_half_on_every_survivor(self) -> None:
        lab = FactorLab(evaluator=Evaluator(_bars()))
        for name in ("slow_trend", "weekend_only", "long_while_closed"):
            lab.submit(Factor(name, name, "r"))
        lab.gate()
        for record in lab.records:
            assert "split_half" in record.as_dict()
            if record.state is Lifecycle.CERTIFIED:
                assert record.split_half is not None
                assert record.split_half["outcome"] == "pass"
