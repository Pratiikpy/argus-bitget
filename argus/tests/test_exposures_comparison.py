"""The Riskfolio-Lib comparison report (`eval/exposures_comparison.py`), offline.

`main()` delegates the regression itself to `argus.lui.exposures.ols` (pinned in
`tests/test_exposures.py`) and the competing fit to Riskfolio-Lib's own source, lifted from an
external, gitignored clone (`research/repos-t2/`, never committed — a hermetic test cannot depend
on it being present). So this file stubs both engines and pins what this module alone is
responsible for: which days it windows in, the return series it hands to the regression, the
rounding it applies to the report, and which factors it says Riskfolio-Lib zeroed out.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest

from argus.eval import exposures_comparison as ec
from argus.lui import exposures as ex


def test_the_audited_book_is_the_documented_weights_plus_named_add_candidates() -> None:
    held = {s: w for s, w in ec.BOOK.items() if w > 0}
    assert held == {"MSFTUSDT": 0.4, "METAUSDT": 0.3, "GOOGLUSDT": 0.3}
    assert sum(held.values()) == pytest.approx(1.0)
    assert set(ec.BOOK) - set(held) == {"NVDAUSDT", "BTCUSDT", "XAUUSDT"}


def _dates(n: int) -> list[date]:
    start = date(2026, 1, 1)
    return [start + timedelta(days=i) for i in range(n)]


def _hand_built_inputs() -> ex.Inputs:
    """One holding (AAAUSDT) plus two filler days that a correct window must exclude, and one
    symbol (BBBUSDT) whose closes could not be read at all."""
    d = _dates(6)
    filler = 1.0
    aaa = {d[0]: filler, d[1]: filler, d[2]: 100.0, d[3]: 110.0, d[4]: 99.0, d[5]: 108.9}
    spy = {d[0]: filler, d[1]: filler, d[2]: 100.0, d[3]: 101.0, d[4]: 98.98, d[5]: 101.9494}
    iwm = {d[0]: filler, d[1]: filler, d[2]: 100.0, d[3]: 103.0, d[4]: 101.97, d[5]: 104.0094}
    mtum = {d[0]: filler, d[1]: filler, d[2]: 100.0, d[3]: 100.0, d[4]: 100.0, d[5]: 103.0}
    btc = {d[0]: filler, d[1]: filler, d[2]: 100.0, d[3]: 105.0, d[4]: 101.85, d[5]: 105.924}
    return ex.Inputs(
        closes={"AAAUSDT": aaa},
        factors={"SPY": spy, "IWM": iwm, "MTUM": mtum, "BTC": btc},
        origins={"AAAUSDT": "test daily closes"},
        missing={"BBBUSDT": "ConnectionError"},
    )


def _fake_loadings_matrix(x: Any, frame: Any, **kwargs: Any) -> Any:
    import pandas as pd

    symbol = frame.columns[0]
    # market and crypto survive forward selection; size and momentum are zeroed out.
    return pd.DataFrame(
        {"market": [1.4], "size": [0.0], "momentum": [0.0], "crypto": [0.03]}, index=[symbol],
    )


def test_main_pins_the_window_the_return_series_and_the_report_rounding(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    inputs = _hand_built_inputs()
    monkeypatch.setattr(ec, "BOOK", {"AAAUSDT": 1.0, "BBBUSDT": 0.0})
    monkeypatch.setattr(ex, "WINDOW_DAYS", 3)  # window+1 = 4 of the 6 given days
    monkeypatch.setattr(ex, "sector_map", lambda: {"rows": {}})
    monkeypatch.setattr(ex, "fetch_inputs", lambda symbols, rows: inputs)
    monkeypatch.setattr(
        ec, "_riskfolio_functions", lambda: {"loadings_matrix": _fake_loadings_matrix},
    )
    out_path = tmp_path / "exposures_comparison.json"
    monkeypatch.setattr(ec, "OUT", out_path)

    calls: list[tuple[list[float], list[list[float]]]] = []

    def fake_ols(
        y: list[float], xs: list[list[float]],
    ) -> tuple[list[float], list[float], float]:
        calls.append((list(y), [list(x) for x in xs]))
        return [0.001, 1.5, -0.6, 0.2, 0.05], [0.0005, 0.5, 0.3, 0.2, 0.1], 0.75

    monkeypatch.setattr(ex, "ols", fake_ols)

    assert ec.main() == 0

    # the two filler days must not have reached the regression: only the last 4 of 6 days count,
    # giving 3 daily returns, worked out by hand from the closes above.
    (y, xs), = calls
    assert y == pytest.approx([0.10, -0.10, 0.10])
    assert xs[0] == pytest.approx([0.01, -0.02, 0.03])  # market: SPY's own return
    assert xs[1] == pytest.approx([0.02, 0.01, -0.01])  # size: IWM minus SPY
    assert xs[2] == pytest.approx([-0.01, 0.02, 0.0])  # momentum: MTUM minus SPY
    assert xs[3] == pytest.approx([0.05, -0.03, 0.04])  # crypto: BTC's own return

    report = json.loads(out_path.read_text(encoding="utf-8"))
    aaa = report["results"]["AAAUSDT"]
    assert aaa["n"] == 3
    assert aaa["argus"] == pytest.approx(
        {"market": 1.5, "size": -0.6, "momentum": 0.2, "crypto": 0.05},
    )
    assert aaa["argus_t"] == pytest.approx({"market": 3.0, "size": -2.0, "momentum": 1.0,
                                            "crypto": 0.5})
    assert aaa["argus_r2"] == pytest.approx(0.75)
    assert aaa["riskfolio_stepwise"] == pytest.approx(
        {"market": 1.4, "size": 0.0, "momentum": 0.0, "crypto": 0.03},
    )
    assert report["results"]["BBBUSDT"] == {"error": "ConnectionError"}
    assert report["riskfolio_zeroed"] == {"AAAUSDT": ["size", "momentum"]}
    assert "BBBUSDT" not in report["riskfolio_zeroed"]


def test_main_reports_an_unfittable_symbol_instead_of_crashing_the_whole_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    inputs = _hand_built_inputs()  # 3 daily returns: too few for a 4-factor OLS (needs > 6)
    monkeypatch.setattr(ec, "BOOK", {"AAAUSDT": 1.0})
    monkeypatch.setattr(ex, "WINDOW_DAYS", 3)
    monkeypatch.setattr(ex, "sector_map", lambda: {"rows": {}})
    monkeypatch.setattr(ex, "fetch_inputs", lambda symbols, rows: inputs)
    # riskfolio is read from a research clone a clean checkout does not have; the fit that fails
    # here is ARGUS's own, before riskfolio would be asked.
    monkeypatch.setattr(
        ec, "_riskfolio_functions", lambda: {"loadings_matrix": _fake_loadings_matrix},
    )
    monkeypatch.setattr(ec, "OUT", tmp_path / "exposures_comparison.json")

    assert ec.main() == 0
    report = json.loads((tmp_path / "exposures_comparison.json").read_text(encoding="utf-8"))
    assert "error" in report["results"]["AAAUSDT"]
