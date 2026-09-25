"""The weekend reference price: nocturne's "fade the void" against ARGUS's perpetual-implied price.

nocturne (egbujor-emmanuel, a Season 2 desk, MIT) is the specialist for exactly this question:
*what is a tokenised stock worth while the US market is shut?* Its finding, on Bitget rToken spot
bars, is that the weekend move reverses — a slope of -1.003 on large caps — so the right
reference on Sunday evening is Friday's price, not the weekend's (`README.md`, `scripts/
study_v2.py`), and its walk-forward test prefers that full fade to every fitted model
(`scripts/fairvalue_v2.py`: large caps, MAE 1.922% against 2.117% for the current price).

ARGUS reads the stock's perpetual instead. The perpetual trades through the weekend on a much
deeper book than the rToken, and the console's implied open is the stock's last close carried by
the perpetual's move since (`lui/research.py::_implied_open_line`).

**Same input, same moments, same scorer.** nocturne's own `core.observations()` builds every
(symbol, weekend) record from its own hourly rToken bars: ``pf`` Friday 15:00 ET, ``pv`` Sunday
19:00 ET (the end of its "void"), ``pm`` Monday 10:00 ET, each the close of the bar opening at that
hour (`scripts/core.py:12-33`). Its walk-forward loop is reproduced here line for line with one
more column — ARGUS's prediction, the rToken's Friday price carried by the perpetual's Friday-to-
Sunday move, read at the same two bars — and scored with its own error and direction rules
(direction only where the actual move exceeds 0.2%, `fairvalue_v2.py:31`).

Scope: nocturne's large-cap universe restricted to names Bitget lists as perpetuals with history
over its weekends; the rows scored are the ones where both desks have a price.
"""

from __future__ import annotations

import ast
import datetime as dt
import json
import random
import statistics as st
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
NOCTURNE = ROOT.parent / "research" / "repos-rivals" / "nocturne"
DATA = ROOT / "data" / "h2h_nocturne"
REPORT = ROOT / "data" / "void_comparison.json"
MIN_TRAIN = 5
"""nocturne's own warm-up (`fairvalue_v2.py:8`) and training floor of 20 rows (`:15`)."""


def _nocturne() -> Any:
    scripts = str(NOCTURNE / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import core  # type: ignore[import-not-found]

    return core


def _large() -> set[str]:
    text = (NOCTURNE / "scripts" / "fairvalue_v2.py").read_text("utf-8")
    start = text.index("LARGE={") + len("LARGE=")
    names: set[str] = ast.literal_eval(text[start:text.index("}", start) + 1])
    return names


def collect(symbols: set[str]) -> dict[str, Any]:
    """The perpetuals' hourly bars over nocturne's weekends, saved in its own bar format (open
    time in ms, close in column 4) so its reader loads both series the same way."""
    from argus.market.history import fetch_window

    DATA.mkdir(parents=True, exist_ok=True)
    start = dt.datetime(2026, 6, 1, tzinfo=dt.UTC)
    found, failures = [], {}
    for rtoken in sorted(symbols):
        perp = f"{rtoken[1:]}USDT"
        try:
            bars = fetch_window(perp, start=start, interval="1H", pause=0.05)
        except Exception as exc:
            failures[perp] = f"{type(exc).__name__}: {exc}"[:160]
            continue
        if not bars:
            failures[perp] = "no bars"
            continue
        rows = [[int(b.ts.timestamp() * 1000), str(b.open), str(b.high), str(b.low),
                 str(b.close), str(b.volume), "0"] for b in bars]
        (DATA / f"{perp}.json").write_text(json.dumps(rows), "utf-8")
        found.append(perp)
    manifest = {"fetched": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
                "perpetuals": found, "failures": failures}
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=1), "utf-8")
    return manifest


def _perp_move(core: Any, sym: str, fri: dt.date) -> float | None:
    path = DATA / f"{sym[1:]}USDT.json"   # nocturne's "RNVDA" is the perpetual NVDAUSDT
    if not path.exists():
        return None
    bars = _BARS.get(path)
    if bars is None:
        bars = _BARS[path] = core.load_bars(str(path))
    sun = fri + dt.timedelta(days=2)
    pf, pv = core.near(bars, fri, 15), core.near(bars, sun, 19)
    return pv / pf - 1 if pf and pv else None


_BARS: dict[Path, Any] = {}


def predictions() -> list[dict[str, Any]]:
    """nocturne's walk-forward (`fairvalue_v2.py:10-26`) with ARGUS's column added."""
    core = _nocturne()
    large = _large()
    rec = [r for r in core.observations() if r["sym"] in large]
    weeks = sorted({r["fri"] for r in rec})
    preds = []
    for i, week in enumerate(weeks):
        if i < MIN_TRAIN:
            continue
        train = [r for r in rec if r["fri"] < week]
        test = [r for r in rec if r["fri"] == week]
        if len(train) < 20:
            continue
        m1 = core.ols([r["w"] for r in train], [r["m"] for r in train])
        m2 = core.ols2(train, "mkt", "res")
        for r in test:
            move = _perp_move(core, r["sym"], r["fri"])
            preds.append({
                "sym": r["sym"], "fri": str(week), "actual": r["m"], "M0": 0.0,
                "M1": (m1["alpha"] + m1["beta"] * r["w"]) if m1 else 0.0,
                "M2": (m2["a0"] + m2["b1"] * r["mkt"] + m2["b2"] * r["res"]) if m2 else 0.0,
                "MF": -r["w"],
                # The rToken's Friday price carried by the perpetual's Friday-to-Sunday move,
                # stated as nocturne's target: the return from the Sunday rToken price.
                "ARGUS": (r["pf"] * (1 + move) / r["pv"] - 1) if move is not None else None,
            })
    return preds


def _score(preds: list[dict[str, Any]], key: str) -> dict[str, Any]:
    """nocturne's own scorer (`fairvalue_v2.py:28-33`)."""
    errors = [p[key] - p["actual"] for p in preds]
    moved = [p for p in preds if abs(p["actual"]) > 0.002 and abs(p[key]) > 1e-9]
    hit = (sum(1 for p in moved if (p[key] > 0) == (p["actual"] > 0)) / len(moved)
           if moved else None)
    return {"mae_pct": round(st.mean(abs(e) for e in errors) * 100, 3),
            "rmse_pct": round(st.mean(e * e for e in errors) ** 0.5 * 100, 3),
            "direction_hit_rate": round(hit, 3) if hit is not None else None,
            "direction_scored": len(moved)}


def _paired(preds: list[dict[str, Any]], a: str, b: str, *, draws: int = 4000,
            seed: int = 11) -> dict[str, Any]:
    """|err_a| - |err_b| in percentage points, bootstrapped over weekends (the shared unit)."""
    by_week: dict[str, list[float]] = {}
    for p in preds:
        by_week.setdefault(p["fri"], []).append(
            (abs(p[a] - p["actual"]) - abs(p[b] - p["actual"])) * 100)
    keys = sorted(by_week)
    rng = random.Random(seed)
    means = sorted(st.mean([d for k in (rng.choice(keys) for _ in keys) for d in by_week[k]])
                   for _ in range(draws))
    point = st.mean(d for k in keys for d in by_week[k])
    low, high = means[int(0.025 * draws)], means[int(0.975 * draws) - 1]
    return {"a": a, "b": b, "mean_diff_pp": round(point, 3),
            "ci95_pp": [round(low, 3), round(high, 3)], "weekends": len(keys),
            "verdict": "a better" if high < 0 else "b better" if low > 0 else "not separable"}


def score() -> dict[str, Any]:
    preds = predictions()
    both = [p for p in preds if p["ARGUS"] is not None]
    models = ("M0", "MF", "M1", "M2")
    manifest = json.loads((DATA / "manifest.json").read_text("utf-8"))
    return {
        "generated": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "target": "rToken price Monday 10:00 ET, predicted from Sunday 19:00 ET (nocturne's own)",
        "baseline_reproduced": {
            "nocturne_published_large_caps": {"M0": 2.117, "MF": 1.922},
            "rerun_here": {m: _score(preds, m)["mae_pct"] for m in models},
        },
        "rows_nocturne": len(preds), "rows_both": len(both),
        "weekends": len({p["fri"] for p in both}),
        "symbols": sorted({p["sym"] for p in both}),
        "summary": {m: _score(both, m) for m in (*models, "ARGUS")},
        "paired": [_paired(both, "ARGUS", "MF"), _paired(both, "ARGUS", "M0"),
                   _paired(both, "MF", "M0")],
        "per_weekend": {week: {m: round(st.mean(abs(p[m] - p["actual"]) for p in both
                                                if p["fri"] == week) * 100, 3)
                               for m in ("M0", "MF", "ARGUS")}
                        for week in sorted({p["fri"] for p in both})},
        "failures": manifest.get("failures") or {},
        "limitations": ("nocturne's data ends where its clone was taken; the perpetuals are "
                        "fetched fresh over the same weekends. Symbols without a Bitget "
                        "perpetual are scored by nocturne alone and left out of the paired rows."),
    }


def main() -> int:  # pragma: no cover - CLI
    if "--collect" in sys.argv:
        collect(_large())
    report = score()
    REPORT.write_text(json.dumps(report, indent=2), "utf-8")
    print(json.dumps({k: report[k] for k in ("baseline_reproduced", "rows_both", "weekends",
                                             "summary", "paired")}, indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
