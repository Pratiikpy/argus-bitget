"""The Decision Stress Testing head-to-head: the scorer that ranks ARGUS against AnalogDesk.

The comparison is only allowed to report when this scorer reproduces AnalogDesk's own numbers, so
the scorer's arithmetic is pinned here on hand-checkable inputs. The live run needs the rival's
clone and Node, and is exercised by `python -m argus.eval.analogstress_comparison`.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from argus.eval.analogstress_comparison import (
    compare,
    conformal_scale,
    diebold_mariano,
    matched_width,
    pit_chi_square,
    quantile,
    score,
)


def test_quantile_interpolates_linearly_between_order_statistics() -> None:
    assert quantile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    assert quantile([4.0, 1.0, 3.0, 2.0], 0.25) == pytest.approx(1.75)
    assert math.isnan(quantile([], 0.5))


def test_the_conformal_multiplier_uses_the_finite_sample_level() -> None:
    scores = [float(i) for i in range(1, 11)]  # n = 10, target 0.8 -> level ceil(8.8)/10 = 0.9
    assert conformal_scale(scores, 0.8) == pytest.approx(quantile(scores, 0.9))
    assert conformal_scale([float("nan"), 1.0], 0.8) == 1.0


def _row(era: str, date: str, sym: str, q: int, y: float, centre: float,
         half: float) -> dict[str, Any]:
    return {"era": era, "date": date, "sym": sym, "q": q, "y": y,
            "arm": {"centre": centre, "hw": half}}


def test_matched_width_lands_on_the_target_coverage() -> None:
    rows = [_row("test", f"d{i}", "A", i, y=i / 100, centre=0.0, half=0.01) for i in range(10)]
    width = matched_width(rows, "arm", 0.8)
    # coverage reaches 80% once the half-width covers |y| = 0.07: width 0.14
    assert width == pytest.approx(0.14, abs=1e-6)


def test_winkler_charges_width_plus_the_miss_scaled_by_two_over_alpha() -> None:
    calib = [_row("calibration", "c", "A", 0, 0.0, 0.0, 1.0)]
    test = [_row("test", "t", "A", 1, y=0.15, centre=0.0, half=0.10)]
    got = score(calib, test, "arm", 0.8)
    # frozen scale = 0 from a single perfect calibration hit; the band collapses to the centre
    assert got.scale == 0.0
    assert got.winkler == pytest.approx(0.0 + (2 / 0.2) * 0.15)


def test_pit_chi_square_is_zero_for_a_perfectly_uniform_pit() -> None:
    rows = [{"pit": (i + 0.5) / 100} for i in range(100)]
    assert pit_chi_square(rows) == 0.0


def test_the_paired_test_clusters_by_date() -> None:
    calib = [_row("calibration", "c", "A", 0, 0.0, 0.0, 1.0)]
    test = [_row("test", f"d{i % 5}", "A", i, y=0.05, centre=0.0, half=0.1) for i in range(20)]
    a = score(calib, test, "arm", 0.8)
    got = diebold_mariano(a, a, test)
    assert got["dates"] == 5 and got["mean_diff_pct"] == 0.0


def test_the_comparison_names_the_winner_only_with_significance() -> None:
    rows = []
    for i in range(60):
        era = "calibration" if i < 20 else "test"
        y = ((i * 37) % 11 - 5) / 100
        rows.append({"era": era, "date": f"d{i % 12}", "sym": "A", "q": i, "y": y,
                     "analogConformal": {"centre": 0.0, "hw": 0.03},
                     "argusAnalogue": {"centre": 0.0, "hw": 0.03}})
    grid = {"coverage": 0.8, "horizon": 5, "eras": {}, "rows": rows}
    report = compare(grid)
    assert report["verdict"].startswith("NO SIGNIFICANT DIFFERENCE")
    assert report["reproduced_rival"] == {"available": False}
    assert "limitations" in report
