"""Four factors or eight, out of sample (`eval/factor_set_comparison.py`), offline."""

from __future__ import annotations

import random
from datetime import date, timedelta

import pytest

from argus.eval import factor_set_comparison as fsc
from argus.lui import exposures as ex

RAW = ("mkt", "smb", "mom", "val", "qual", "lowvol", "rates", "btc")


def _walk(rets: list[float], start: float = 100.0) -> list[float]:
    out = [start]
    for r in rets:
        out.append(out[-1] * (1 + r))
    return out


def _world(n: int = 300, seed: int = 4) -> tuple[list[date], dict[str, list[float]],
                                                  dict[str, dict[date, float]]]:
    rng = random.Random(seed)
    days = [date(2025, 1, 1) + timedelta(days=i) for i in range(n + 1)]
    f = {k: [rng.gauss(0, 0.01) for _ in range(n)] for k in RAW}

    def spread(key: str) -> list[float]:
        return _walk([m + s for m, s in zip(f["mkt"], f[key], strict=True)])

    series = {"SPY": _walk(f["mkt"]), "IWM": spread("smb"), "MTUM": spread("mom"),
              "VLUE": spread("val"), "QUAL": spread("qual"), "USMV": spread("lowvol"),
              "IEF": _walk(f["rates"]), "BTC": _walk(f["btc"])}
    return days, f, {k: dict(zip(days, v, strict=True)) for k, v in series.items()}


def _asset(days: list[date], f: dict[str, list[float]], loads: tuple[float, ...],
           seed: int = 1) -> dict[date, float]:
    rng = random.Random(seed)
    rets = [sum(w * f[k][t] for w, k in zip(loads, RAW, strict=True)) + rng.gauss(0, 0.004)
            for t in range(len(f["mkt"]))]
    return dict(zip(days, _walk(rets), strict=True))


def test_a_name_driven_by_the_added_factors_is_explained_better_out_of_sample() -> None:
    days, f, closes = _world()
    asset = _asset(days, f, (1.0, 0.0, 0.0, 0.6, -0.5, 0.8, -0.9, 0.0))
    got = fsc.compare(asset, closes)
    assert got is not None and got["n"] == 300
    assert got["eight"]["out_of_sample_r2"] > got["four"]["out_of_sample_r2"] + 0.1
    assert got["eight"]["adjusted_r2"] > got["four"]["adjusted_r2"]
    assert all(abs(got["added_t"][k]) > 5 for k in ("value", "quality", "low_vol", "rates"))


def test_a_name_the_added_factors_do_not_drive_gains_nothing_real() -> None:
    days, f, closes = _world(seed=6)
    asset = _asset(days, f, (1.2, -0.4, 0.3, 0.0, 0.0, 0.0, 0.0, 0.05), seed=2)
    got = fsc.compare(asset, closes)
    assert got is not None
    assert got["eight"]["out_of_sample_r2"] == pytest.approx(
        got["four"]["out_of_sample_r2"], abs=0.02)
    assert all(abs(t) < 3 for t in got["added_t"].values())


def test_too_little_history_is_not_compared() -> None:
    days, f, closes = _world(n=100)
    assert fsc.compare(_asset(days, f, (1, 0, 0, 0, 0, 0, 0, 0)), closes) is None


def test_the_summary_counts_only_names_that_were_fitted() -> None:
    results = {
        "A": {"four": {"adjusted_r2": 0.2, "out_of_sample_r2": 0.1},
              "eight": {"adjusted_r2": 0.3, "out_of_sample_r2": 0.2},
              "added_t": {"value": 2.5, "quality": 0.1, "low_vol": -3.0, "rates": 1.0}},
        "B": {"four": {"adjusted_r2": 0.4, "out_of_sample_r2": 0.3},
              "eight": {"adjusted_r2": 0.4, "out_of_sample_r2": 0.25},
              "added_t": {"value": 0.5, "quality": 2.0, "low_vol": 0.0, "rates": 0.0}},
        "C": {"error": "closes unreadable (HistoryError)"},
    }
    summary = fsc.summarise(results)
    assert summary["names_fitted"] == 2
    assert summary["mean_out_of_sample_r2"] == {"four": 0.2, "eight": 0.225}
    assert summary["eight_better_out_of_sample"] == 1 and summary["of"] == 2
    assert summary["added_factor_significant"] == {"value": 1, "quality": 1, "low_vol": 1,
                                                   "rates": 0}


def test_the_base_set_is_the_old_model_and_the_wide_set_the_current_one() -> None:
    assert fsc.BASE == ("market", "size", "momentum", "crypto")
    assert set(fsc.BASE) < set(ex.FACTORS) and len(ex.FACTORS) == 8
