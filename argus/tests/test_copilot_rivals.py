"""The Portfolio Copilot head-to-head: the rival port and the scorer that ranks ARGUS against it.

The live run needs the rival's clone and a frozen Bitget snapshot and is exercised by
`python -m argus.eval.copilot_rivals`. What is pinned here is what the verdict rests on: the port
keeps weekend-copilot's quirks (the silent 1.0 beta, the undated tail slice) rather than quietly
fixing them, the exact test is exact, and a reproduction that does not match is reported as such.
"""

from __future__ import annotations

import itertools
import json
import math
import random
from pathlib import Path
from typing import Any

import pytest

from argus.eval.copilot_rivals import (
    NAMES,
    books,
    reproduction,
    rival_beta,
    rival_book_beta,
    rival_log_returns,
    rival_partner,
    score,
    verdict,
    wilcoxon_exact,
)


def test_the_rival_beta_slices_the_tail_without_aligning_dates() -> None:
    market = [0.01, -0.02, 0.03, -0.01]
    asset = [9.9, 0.02, -0.04, 0.06, -0.02]  # the leading 9.9 is dropped by the tail slice
    assert rival_beta(asset, market) == pytest.approx(2.0)
    assert rival_beta([0.1], [0.2]) == 1.0
    assert rival_beta([0.1, 0.2], [0.0, 0.0]) == 1.0


def test_the_rival_book_beta_falls_back_to_one_below_twenty_closes() -> None:
    rng = random.Random(1)
    market = [100.0]
    for _ in range(40):
        market.append(market[-1] * (1 + rng.gauss(0, 0.01)))
    doubled = [market[0]]
    for a, b in itertools.pairwise(market):
        doubled.append(doubled[-1] * math.exp(2 * math.log(b / a)))
    closes = {"QQQ": market, "TWICE": doubled, "SHORT": market[:19]}
    assert rival_book_beta({"TWICE": 1.0}, closes, "QQQ") == pytest.approx(2.0)
    assert rival_book_beta({"SHORT": 1.0}, closes, "QQQ") == 1.0
    assert rival_book_beta({"TWICE": 0.5, "SHORT": 0.5}, closes, "QQQ") == pytest.approx(1.5)


def test_log_returns_skip_non_positive_closes() -> None:
    assert rival_log_returns([1.0, 0.0, 2.0, 4.0]) == pytest.approx([math.log(2.0)])


def test_the_rival_partner_is_read_on_pairs_with_the_candidate() -> None:
    rng = random.Random(2)
    base = [100.0]
    for _ in range(40):
        base.append(base[-1] * (1 + rng.gauss(0, 0.01)))
    noise = [100.0]
    for _ in range(40):
        noise.append(noise[-1] * (1 + rng.gauss(0, 0.01)))
    closes = {"ADD": base, "TWIN": [c * 2 for c in base], "OTHER": noise}
    assert rival_partner("ADD", ["OTHER", "TWIN"], closes) == "TWIN"
    assert rival_partner("ADD", ["OTHER"], {"ADD": base[:5], "OTHER": noise}) is None


def test_the_exact_wilcoxon_matches_hand_counts() -> None:
    assert wilcoxon_exact([1.0, 2.0, 3.0, 4.0, 5.0]) == pytest.approx(2 / 32)
    assert wilcoxon_exact([0.0, 0.0]) == 1.0
    assert wilcoxon_exact([1.0, -1.0]) == 1.0


def test_books_are_seeded_normalised_and_add_a_new_name() -> None:
    first, second = books(random.Random(7)), books(random.Random(7))
    assert first == second
    for held, add, size in first:
        assert abs(sum(held.values()) - 1.0) < 1e-12
        assert add not in held and add in NAMES
        assert 3 <= len(held) <= 5 and size in (0.1, 0.2, 0.3)


def _row(origin: str, pred: dict[str, float], real: float, partner: str) -> dict[str, Any]:
    return {"origin": origin, "held": ["A", "B"], "pred": pred,
            "partner": {"argus_open": partner, "rival": "B"},
            "real": {"bitget_daily": real, "native_daily": real, "bitget_open_hourly": real},
            "real_partner": {"bitget_daily": "A", "native_daily": "A"},
            "rival_beta_one_fallbacks": 0}


def test_the_scorer_reports_significance_only_when_earned() -> None:
    rows = []
    for k in range(9):
        pred = {"argus_open": 1.0, "argus_blended": 1.1, "rival_spy": 1.6, "rival_qqq": 1.3,
                "naive_one": 1.0}
        rows.append(_row(f"2026-0{k + 1}-01", pred, 1.0 + 0.01 * k, "A"))
    scored = score(rows)
    primary = scored["comparisons"]["bitget_daily"]["argus_open vs rival_qqq"]
    assert primary["origins_better"] == 9 and primary["wilcoxon_p"] < 0.01
    assert verdict(scored).startswith("ARGUS WINS")
    assert scored["partner_hit_rate"]["bitget_daily"]["argus_open"] == 1.0
    assert scored["partner_hit_rate"]["bitget_daily"]["rival"] == 0.0


def test_a_reproduction_that_does_not_match_is_reported(tmp_path: Path) -> None:
    snaps = tmp_path / "data" / "snapshots"
    snaps.mkdir(parents=True)
    rng = random.Random(3)
    for name in ("SPY", "NVDA"):
        closes = [100.0]
        for _ in range(30):
            closes.append(closes[-1] * (1 + rng.gauss(0, 0.01)))
        (snaps / f"{name}.json").write_text(
            json.dumps({"bars": [{"date": "2026-01-01", "close": c} for c in closes]}),
            encoding="utf-8")
    delta = {"weightsBefore": {"RNVDA": 1.0}, "weightsAfter": {"RNVDA": 1.0},
             "betaBefore": 99.0, "betaAfter": 99.0,
             "maxCorrelation": {"pair": ["—", "—"], "value": 0.0}}
    (tmp_path / "argus-probe.json").write_text(
        json.dumps({"dataBuiltAt": "x", "cases": {"c": {"delta": delta}}}), encoding="utf-8")
    got = reproduction(tmp_path)
    assert got["available"] and not got["reproduced"]
    assert reproduction(tmp_path / "missing")["available"] is False


def test_the_console_quotes_the_measured_record_with_its_losses(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    from argus.lui import research

    answer = importlib.import_module("argus.lui.answer")

    errors = {"argus_open": 0.241, "rival_spy": 0.554, "rival_qqq": 0.36, "naive_one": 0.298}
    report = {"beta_error": {"bitget_daily": {"mean_abs_error": errors, "books_scored": 1800}},
              "comparisons": {"bitget_daily": {
                  "argus_open vs rival_spy": {"wilcoxon_p": 0.0117},
                  "argus_open vs rival_qqq": {"wilcoxon_p": 0.0977}}}}
    (tmp_path / "copilot_rivals.json").write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(answer, "_notes_path", lambda: tmp_path / "notes.jsonl")
    line = research._beta_track_record()
    assert line is not None
    assert "0.24" in line and "0.55" in line and "not significant" in line and "1.0" in line
    (tmp_path / "copilot_rivals.json").write_text("{}", encoding="utf-8")
    assert research._beta_track_record() is None
