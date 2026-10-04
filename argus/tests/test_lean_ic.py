"""The lean's information coefficient (`eval/lean_ic.py`, build-list 2.1) and the /status rows that
report the desk's audit of itself (`lui/status_page.self_audit_lines`)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from argus.eval import lean_ic
from argus.lui.status_page import self_audit_lines
from argus.paper.marks import Mark


def _mark(seq: int, cycle: str, lean: str, move: float, hours: float = 2.0) -> Mark:
    return Mark(seq=seq, symbol=f"S{seq}USDT", decided_at=f"2026-09-{cycle}T10:00:00+00:00",
                marked_at=f"2026-09-{cycle}T12:00:00+00:00", horizon_hours=hours,
                entry_price="100", mark_price="100", move_bps=str(move), lean=lean)


# Student-t two-sided p-values from scipy.stats.t.sf, so the pure-Python beta is held to scipy.
@pytest.mark.parametrize(("t", "df", "expected"), [
    (0.92, 57, 0.36144998670649),
    (-1.28, 18, 0.21679651712209),
    (2.5, 5, 0.05449009934238),
    (-4.0, 10, 0.00251833262474),
])
def test_the_p_value_is_scipys(t: float, df: int, expected: float) -> None:
    assert lean_ic.t_two_sided_p(t, df) == pytest.approx(expected, rel=1e-9)


def test_a_lean_that_ranks_every_cycle_scores_one_and_horizons_stay_apart() -> None:
    marks, conf, seq = [], {}, 0
    for day in ("10", "11", "12"):
        for k, move in enumerate((-30.0, -10.0, 10.0, 30.0)):
            seq += 1
            marks.append(_mark(seq, day, "up" if move > 0 else "down", move))
            conf[seq] = (0.2, 0.1, 0.1, 0.2)[k]  # more confident on the bigger move, both ways
    # an overnight mark that ranks the wrong way must not reach the 2h figure
    marks.append(_mark(99, "13", "up", -50.0, hours=18.0))
    conf[99] = 0.9
    marks.append(_mark(100, "13", "none", 5.0))
    blob = lean_ic.score(marks, conf)
    two = blob["horizons"]["about_2h"]
    assert two["calls"] == 12
    assert two["cycles_with_ic"] == 3
    assert two["ic_mean"] == pytest.approx(1.0)
    assert two["hit_rate"] == 1.0
    assert blob["horizons"]["overnight_12h_plus"]["calls"] == 1
    assert blob["unscored"]["lean_none"] == 1


def test_a_cycle_with_too_few_calls_is_left_out_of_the_series() -> None:
    marks = [_mark(i, "10", "up", float(i)) for i in range(1, lean_ic.MIN_CALLS_PER_CYCLE)]
    blob = lean_ic.score(marks, {m.seq: 0.1 * m.seq for m in marks})
    assert blob["horizons"]["about_2h"]["cycles_with_ic"] == 0
    assert blob["horizons"]["about_2h"]["t_stat"] is None


def test_the_status_line_says_no_skill_when_p_is_high(tmp_path: Path) -> None:
    (tmp_path / "lean_ic.json").write_text(json.dumps({
        "generated_at": "2026-10-05T00:00:00+00:00",
        "horizons": {"about_2h": {"calls": 628, "cycles_with_ic": 58, "hit_rate": 0.551,
                                  "ic_mean": 0.042, "t_stat": 0.92, "p_value": 0.36}}}),
        encoding="utf-8")
    assert "no evidence of skill" in (lean_ic.line(tmp_path / "lean_ic.json") or "")
    rows = dict(self_audit_lines(tmp_path))
    assert "no evidence the confidence ranks" in rows["Self-audit: does the lean rank the move?"]
    # artefacts that are absent produce no row rather than a guessed one
    assert len(rows) == 1


def test_the_live_artefacts_render_every_self_audit_row() -> None:
    from argus.truth.paths import DATA_DIR

    needed = ("lean_ic.json", "refusal_alpha.json", "perturbation_robustness.json",
              "consistency.json", "leakage.json")
    if not all((DATA_DIR / name).exists() for name in needed):
        pytest.skip("the measured artefacts are not in this checkout")
    rows = self_audit_lines(DATA_DIR)
    assert len(rows) == 5
    assert all(text and "None" not in text for _, text in rows)
