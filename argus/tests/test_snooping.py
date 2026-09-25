"""Hansen's SPA and Romano-Wolf's StepM (`argus.backtest.snooping`), the vendored `arch` excerpt
they are checked against (`argus.eval.baselines.arch_spa_stepm`), and the `searchoff` pool gate
built on them (`argus.research.discovery_gate`).

The port's claim is exact parity with `arch` 8.0.0's ``nested=True`` path when both read the same
resamples, so the parity tests compare p-values and StepM sets with ``==``. The excerpt's claim is
that every line under a VERBATIM marker is the named range of the published wheel; when `arch`
itself is importable that is checked line for line, and the excerpt is run against it.
"""

from __future__ import annotations

import importlib
import json
import random
import re
from datetime import datetime
from decimal import Decimal
from math import sqrt
from pathlib import Path
from typing import Any

import pytest

from argus.backtest.dependence import stationary_bootstrap_indices
from argus.backtest.engine import Bar
from argus.backtest.metrics import MetricError
from argus.backtest.snooping import (
    SnoopingBootstrap,
    expand_blocks,
    spa_test,
    stationary_bootstrap_blocks,
    stepm,
)
from argus.research.discovery_gate import (
    GateError,
    RecordingArena,
    gate,
    net_returns,
    pool_returns,
)
from argus.research.grammar import ORIGINAL_EIGHT

ROOT = Path(__file__).resolve().parents[1]
VENDORED = ROOT / "src" / "argus" / "eval" / "baselines" / "arch_spa_stepm.py"
CANDLES = ROOT / "data" / "regime_candles_fixture.json"


def _series(t: int, k: int, *, edges: dict[int, float], seed: int = 7) -> list[list[float]]:
    """``k`` unit-variance return series of length ``t``, AR(1)-dependent, with planted means."""
    rng = random.Random(seed)
    out: list[list[float]] = []
    for j in range(k):
        prev = 0.0
        column: list[float] = []
        for _ in range(t):
            prev = 0.3 * prev + rng.gauss(0.0, 1.0)
            column.append(prev + edges.get(j, 0.0))
        out.append(column)
    return out


# --- the resampling scheme -------------------------------------------------------------------


class TestBlocks:
    @pytest.mark.parametrize(("n", "block"), [(50, 1.0), (200, 7.5), (333, 40.0)])
    def test_blocks_expand_to_the_dependence_modules_own_indices(
        self, n: int, block: float
    ) -> None:
        for seed in (1, 2, 3):
            blocks = stationary_bootstrap_blocks(n, block_length=block, rng=random.Random(seed))
            indices = stationary_bootstrap_indices(
                n, block_length=block, rng=random.Random(seed)
            )
            assert expand_blocks(blocks, n) == indices
            assert sum(length for _, length in blocks) == n

    def test_degenerate_inputs_are_refused(self) -> None:
        with pytest.raises(MetricError):
            stationary_bootstrap_blocks(1, block_length=3.0, rng=random.Random(0))
        with pytest.raises(MetricError):
            stationary_bootstrap_blocks(10, block_length=0.0, rng=random.Random(0))


# --- the tests themselves --------------------------------------------------------------------


class TestSpaAndStepM:
    def test_a_planted_edge_is_found_and_noise_is_not(self) -> None:
        strategies = _series(400, 10, edges={3: 0.45})
        result = stepm(strategies, [0.0] * 400, resamples=400, block_length=5.0)
        assert result.superior == (3,)
        assert result.spa.rejects and result.spa.best_index == 3
        noise = stepm(_series(400, 10, edges={}), [0.0] * 400, resamples=400, block_length=5.0)
        assert noise.superior == ()

    def test_every_model_worse_than_the_benchmark_gives_a_p_value_of_one(self) -> None:
        # The studentized statistic is floored at zero (bsds.m): a family that is all losers has
        # nothing to be surprised by, so the p-value is exactly one rather than merely large.
        strategies = _series(300, 6, edges=dict.fromkeys(range(6), -0.8))
        result = spa_test(strategies, [0.0] * 300, resamples=300, block_length=4.0)
        assert result.p_consistent == 1.0 and result.p_lower == 1.0
        assert not result.rejects
        assert result.verdict.startswith("NOT ESTABLISHED")

    def test_stepm_stops_when_every_model_is_rejected(self) -> None:
        # arch 8.0.0 compares the latest round's count with k (bashtage/arch#862); the port uses
        # the running total, so a family of all-winners ends cleanly with every index named.
        strategies = _series(300, 5, edges={0: 1.5, 1: 1.2, 2: 1.0, 3: 0.9, 4: 0.8})
        result = stepm(strategies, [0.0] * 300, resamples=300, block_length=4.0)
        assert result.superior == (0, 1, 2, 3, 4)
        assert sum(len(s) for s in result.steps) == 5

    def test_one_set_of_draws_answers_both_variants(self) -> None:
        strategies = _series(250, 8, edges={1: 0.3})
        boot = SnoopingBootstrap(strategies, [0.0] * 250, resamples=200, block_length=5.0)
        a, b = boot.spa(studentize=True), boot.spa(studentize=True)
        assert a == b
        raw = boot.spa(studentize=False)
        assert raw.studentized is False and a.studentized is True
        assert json.dumps(boot.stepm().as_dict())  # the artefact form serialises

    def test_a_constant_excess_has_no_studentized_statistic(self) -> None:
        strategies = [[0.01] * 100, [0.02 * (i % 3) for i in range(100)]]
        boot = SnoopingBootstrap(strategies, [0.0] * 100, resamples=50, block_length=3.0)
        with pytest.raises(MetricError, match="no resampling variance"):
            boot.spa(studentize=True)
        assert boot.spa(studentize=False).strategies == 2

    @pytest.mark.parametrize(
        ("strategies", "benchmark", "match"),
        [
            ([], [0.0] * 50, "no strategies"),
            ([[0.1] * 10], [0.0] * 10, "below"),
            ([[0.1] * 50, [0.1] * 49], [0.0] * 50, "same periods"),
            ([[float("nan")] + [0.1] * 49], [0.0] * 50, "NaN"),
        ],
    )
    def test_malformed_families_are_refused(
        self, strategies: list[list[float]], benchmark: list[float], match: str
    ) -> None:
        with pytest.raises(MetricError, match=match):
            SnoopingBootstrap(strategies, benchmark, resamples=10, block_length=3.0)

    def test_size_outside_the_unit_interval_is_refused(self) -> None:
        boot = SnoopingBootstrap(_series(60, 2, edges={}), [0.0] * 60, resamples=20,
                                 block_length=3.0)
        with pytest.raises(MetricError):
            boot.stepm(size=1.0)


# --- parity with arch ------------------------------------------------------------------------


def _vendored() -> Any:
    pytest.importorskip("numpy")
    pytest.importorskip("pandas")
    return importlib.import_module("argus.eval.baselines.arch_spa_stepm")


def _arch_indices(vendored: Any, n: int, reps: int, seed: int, block: int) -> list[list[int]]:
    """The exact index draws arch's StationaryBootstrap makes for an integer seed."""
    import numpy as np

    rng = np.random.default_rng(seed)
    out: list[list[int]] = []
    for _ in range(reps):
        index = vendored._get_random_integers(rng, n, size=n).astype(np.int64)
        u = rng.random(n)
        out.append([int(i) for i in vendored.stationary_bootstrap_sample(index, u, 1.0 / block)])
    return out


class TestArchParity:
    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_the_port_reproduces_arch_nested_exactly_on_the_same_draws(self, seed: int) -> None:
        import numpy as np

        vendored = _vendored()
        t, reps = 240, 200
        columns = _series(t, 12, edges={2: 0.35, 7: 0.25, 9: 0.15}, seed=seed)
        y = np.array(columns).T
        bench = np.zeros(t)
        block = int(sqrt(t))
        spa = vendored.SPA(bench, -y, reps=reps, seed=seed, nested=True)
        spa.compute()
        step = vendored.StepM(bench, -y, reps=reps, seed=seed, nested=True)
        step.compute()
        boot = SnoopingBootstrap(
            columns, [0.0] * t, resamples=reps, block_length=float(block),
            indices=_arch_indices(vendored, t, reps, seed, block),
        )
        ours = boot.spa(studentize=False)
        assert [float(p) for p in spa.pvalues] == [ours.p_lower, ours.p_consistent, ours.p_upper]
        assert [int(i) for i in step.superior_models] == list(
            boot.stepm(studentize=False).superior
        )

    def test_the_excerpt_matches_the_installed_arch_line_for_line(self) -> None:
        arch = pytest.importorskip("arch")
        if arch.__version__ != "8.0.0":
            pytest.skip(f"the excerpt is of arch 8.0.0; {arch.__version__} is installed")
        root = Path(arch.__file__).parent.parent
        ours = VENDORED.read_text(encoding="utf-8").splitlines()
        marks = [(i, line) for i, line in enumerate(ours) if line.startswith("# ---- ")]
        checked = 0
        for n, (i, line) in enumerate(marks):
            found = re.search(r"VERBATIM: (\S+):(\d+)-(\d+)", line)
            if found is None:
                continue
            path, first, last = found.group(1), int(found.group(2)), int(found.group(3))
            upstream = (root / path).read_text(encoding="utf-8").splitlines()[first - 1:last]
            section = ours[i + 1:(marks[n + 1][0] if n + 1 < len(marks) else len(ours))]
            cursor = 0
            for wanted in upstream:  # every upstream line, in order; our lines are labelled
                while cursor < len(section) and section[cursor] != wanted:
                    cursor += 1
                assert cursor < len(section), f"{path}:{first}-{last} lost {wanted!r}"
                cursor += 1
            extra = [s for s in section if s.strip() and s not in upstream]
            assert all("ours" in s or s.endswith("falls back to this") or "=" in s
                       for s in extra), extra
            checked += 1
        assert checked == 6

    def test_the_excerpt_matches_the_installed_arch_numerically(self) -> None:
        import numpy as np

        pytest.importorskip("arch")
        from arch.bootstrap import SPA as RealSPA
        from arch.bootstrap import StepM as RealStepM

        vendored = _vendored()
        y = np.array(_series(200, 10, edges={4: 0.3, 8: 0.2})).T
        bench = np.zeros(200)
        for nested in (False, True):
            real = RealSPA(bench, -y, reps=150, seed=11, nested=nested)
            real.compute()
            ours = vendored.SPA(bench, -y, reps=150, seed=11, nested=nested)
            ours.compute()
            assert list(real.pvalues) == list(ours.pvalues)
            assert list(real.critical_values()) == list(ours.critical_values())
            real_step = RealStepM(bench, -y, reps=150, seed=11, nested=nested)
            real_step.compute()
            our_step = vendored.StepM(bench, -y, reps=150, seed=11, nested=nested)
            our_step.compute()
            assert list(real_step.superior_models) == list(our_step.superior_models)


# --- the searchoff pool gate -----------------------------------------------------------------


def _bars(n: int = 500) -> list[Bar]:
    series = json.loads(CANDLES.read_text(encoding="utf-8"))["series"]["NVDAUSDT"][:n]
    return [Bar(ts=datetime.fromisoformat(ts), close=Decimal(c)) for ts, c in series]


def _arena() -> RecordingArena:
    arena = RecordingArena(bars=_bars(), budget=20)
    for expr in ORIGINAL_EIGHT.values():
        arena.evaluate(expr)
    return arena


class TestDiscoveryGate:
    def test_recovered_series_reproduce_the_arenas_scores_bit_for_bit(self) -> None:
        arena = _arena()
        pool = pool_returns(arena)
        assert len(pool.canonicals) == len(arena.seen) > 0
        assert pool.in_sample_sharpes == tuple(c.in_sample for c in arena.seen.values())
        cut = arena.split
        for key, series in zip(pool.canonicals, pool.in_sample, strict=True):
            assert tuple(net_returns(arena.expressions[key], arena.bars, 0, cut,
                                     arena.cost_bps)) == series

    def test_a_drifted_score_is_refused_loudly(self) -> None:
        arena = _arena()
        key, found = next(iter(arena.seen.items()))
        arena.seen[key] = type(found)(canonical=found.canonical,
                                      in_sample=found.in_sample + 1e-12,
                                      out_of_sample=found.out_of_sample)
        with pytest.raises(GateError, match="drifted"):
            pool_returns(arena)

    def test_an_empty_arena_has_nothing_to_gate(self) -> None:
        with pytest.raises(GateError):
            pool_returns(RecordingArena(bars=_bars(), budget=5))

    def test_the_gate_reports_stepm_with_the_deflated_sharpe_beside_it(self) -> None:
        pool = pool_returns(_arena())
        verdict = gate(pool, resamples=200, seed=5)
        assert verdict.candidates == len(pool.canonicals)
        assert verdict.trials_spent == pool.trials_spent
        assert set(verdict.survivors) <= set(pool.canonicals)
        assert (verdict.deflated_sharpe_of_best is None) != (verdict.deflated_sharpe_error is None)
        body = verdict.as_dict()
        assert body["stepm"]["studentized"] is True
        assert json.dumps(body)
