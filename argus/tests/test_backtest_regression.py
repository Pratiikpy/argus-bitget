"""Strategy-level regression gate for the backtest engine: three strategies on frozen real bars,
their statistics pinned (research/harvest/56-freqtrade-lean.md, item 2).

`tests/test_backtest_engine.py` checks the engine's mechanics on synthetic series: lagged signals,
charged costs, the out-of-sample split. It never pins what the engine *reports* on real data. A
change to a metric, the cost charge, the split or the lag can therefore keep every mechanical test
green and still move every Sharpe, drawdown and win rate the desk publishes. QuantConnect Lean
closes that gap with `Tests/RegressionTests.cs` (`AlgorithmStatisticsRegression`) and
`Tests/AlgorithmRunner.cs:305` (`Assert.AreEqual(expected, result)`) over pinned statistics per
algorithm, run on every push (Apache-2.0; the pattern is taken, the code is ours).

The strategies are deliberately fixed and simple: buy and hold, a 24-hour momentum rule, a 6-hour
reversion rule. They are not under tuning, so the gate fires on an engine change, not on strategy
work. The bars are `data/general_factorsafety_bars.json` (NVDAUSDT, 1,487 Bitget hourly candles,
frozen 2026-09-25), checked by digest first so a changed fixture cannot pass as a changed engine.

**Tolerance.** Lean allows only ``-0`` against ``0`` (`AlgorithmRunner.cs:293-303`). Here the
allowance is a relative 1e-9: the engine's arithmetic is pure Python floats, but ``math.log`` and
``math.exp`` may differ in the last place between the Windows and Linux C libraries CI runs on. A
real change moves these figures by far more; :func:`test_the_gate_can_fail` shows one basis point
of fee moving the net Sharpe by more than 1e-4.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import datetime
from decimal import Decimal

import pytest

from argus.backtest.engine import BacktestResult, Bar, buy_and_hold, run
from argus.cost.model import CostModel
from argus.truth.paths import DATA_DIR

FIXTURE = DATA_DIR / "general_factorsafety_bars.json"

FIXTURE_DIGEST = "6b26985ceac9e1174f4478dcc39212123c2ae06e1696657f46dd2538cd07a809"
TOLERANCE = 1e-9


def momentum_24(bars: Sequence[Bar], i: int) -> float:
    return 1.0 if i >= 24 and bars[i].close > bars[i - 24].close else 0.0


def reversion_6(bars: Sequence[Bar], i: int) -> float:
    return 1.0 if i >= 6 and bars[i].close < bars[i - 6].close else 0.0


STRATEGIES = {"buy_and_hold": buy_and_hold, "momentum_24": momentum_24,
              "reversion_6": reversion_6}

# (total_return, sharpe, max_drawdown, win_rate, trades) per part; COSTS: (total_cost_bps, trades).
EXPECTED: dict[str, dict[str, tuple[float, float, float, float, int]]] = {
    "buy_and_hold": {
        "net": (0.08094331921006792, 1.4751194205071665, 0.10216428084526143,
                0.532967032967033, 1456),
        "gross": (0.08159136884693496, 1.4850870121560336, 0.10216428084526084,
                  0.532967032967033, 1456),
        "in_sample": (0.10226759291910215, 2.3957254056425605, 0.09736453319353792,
                      0.5334040296924708, 943),
        "out_of_sample": (-0.019345822961703063, -1.2747313781301555, 0.10216428084526229,
                          0.5321637426900585, 513),
    },
    "momentum_24": {
        "net": (0.009507779523209026, 0.34518734635019066, 0.07824859843349714,
                0.48926553672316386, 885),
        "gross": (0.09078124616791383, 2.123039581215434, 0.07380910168924379,
                  0.5378048780487805, 820),
        "in_sample": (0.006857949484004422, 0.35806658476218267, 0.07824859843349714,
                      0.4708904109589041, 584),
        "out_of_sample": (0.0026317814152052588, 0.3397369656128269, 0.0708634273688729,
                          0.5249169435215947, 301),
    },
    "reversion_6": {
        "net": (-0.17022571049691293, -4.267623245918978, 0.17866503689690008,
                0.39504563233376794, 767),
        "gross": (-0.013058668755709602, -0.18640813429960462, 0.07375281545023216,
                  0.5379644588045234, 619),
        "in_sample": (-0.09278406011563334, -2.9163630726627683, 0.1136996195850095,
                      0.3975409836065574, 488),
        "out_of_sample": (-0.08536187138769902, -9.526351441375553, 0.09694413950954234,
                          0.3906810035842294, 279),
    },
}
COSTS = {"buy_and_hold": (6.0, 1), "momentum_24": (774.0, 129), "reversion_6": (1734.0, 289)}


def load_bars() -> tuple[list[Bar], str]:
    """The frozen bars and the sha256 of their rows, read here rather than through the module
    that first used them, which is not public yet."""
    rows = json.loads(FIXTURE.read_text(encoding="utf-8"))["rows"]
    digest = hashlib.sha256(json.dumps(rows).encode("utf-8")).hexdigest()
    return [Bar(ts=datetime.fromisoformat(r[0]), close=Decimal(r[1]),
                extra={"volume": float(r[4]), "high": float(r[2]), "low": float(r[3])})
            for r in rows], digest


@pytest.fixture(scope="module")
def bars() -> list[Bar]:
    loaded, digest = load_bars()
    assert digest == FIXTURE_DIGEST, "the frozen bars changed; re-pin only on purpose"
    return loaded


def backtest(bars: Sequence[Bar], name: str, taker: str = "6") -> BacktestResult:
    cost = CostModel(taker_bps=Decimal(taker), maker_bps=Decimal("2"))
    return run(name, "NVDAUSDT", bars, STRATEGIES[name], cost=cost, periods_per_year=24 * 365)


@pytest.mark.parametrize("name", sorted(STRATEGIES))
def test_the_reported_statistics_have_not_moved(bars: list[Bar], name: str) -> None:
    result = backtest(bars, name)
    for part, expected in EXPECTED[name].items():
        got = getattr(result, part)
        assert got is not None, part
        observed = (got.total_return, got.sharpe, got.max_drawdown, got.win_rate, got.trades)
        for field, want, have in zip(("total_return", "sharpe", "max_drawdown", "win_rate"),
                                     expected[:4], observed[:4], strict=True):
            assert have == pytest.approx(want, rel=TOLERANCE, abs=1e-15), f"{name} {part} {field}"
        assert observed[4] == expected[4], f"{name} {part} trades"
    cost, trades = COSTS[name]
    assert result.total_cost_bps == pytest.approx(cost, rel=TOLERANCE)
    assert result.trades == trades


def test_the_gate_can_fail(bars: list[Bar]) -> None:
    """One basis point more fee per side moves the pinned net Sharpe far outside the tolerance."""
    dearer = backtest(bars, "momentum_24", taker="7")
    pinned = EXPECTED["momentum_24"]["net"][1]
    assert abs(dearer.net.sharpe - pinned) > 1e-4
