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

**The ARGUS column is the console's own line (changed 2026-09-26).** Until then this module
computed ``pf * (1 + move) / pv - 1`` itself while `eval/standing.py` credited the console, and the
harness-validity canary (`data/harness_validity.json`) found the console never ran at scoring
time. Now every row calls `lui/research._implied_open_line` at Sunday 20:00 ET (the close of the
bar nocturne reads at 19:00) with the perpetual's price then, through
`eval/overnight_comparison.console_feed`: the console's perpetual reader gets the saved Bitget bars
in ``data/h2h_nocturne/``, and its regular-close reader gets, for every session day, the rToken's
own price at 16:00 ET read by nocturne's ``near`` — the anchor this column has always used (a
Yahoo close is not in nocturne's data, and would add the rToken's basis to a target stated in
rToken prices). The session clock that picks which close to carry (a holiday Friday carries
Thursday's), the exact close-bar selection and the arithmetic are the console's. The implied
price is printed to the cent and read back from the sentence; the former in-module formula is
kept as ``console_replay``'s comparison so any difference is on the record.
"""

from __future__ import annotations

import ast
import datetime as dt
import json
import statistics as st
import sys
from pathlib import Path
from typing import Any

from argus.eval import artefact
from argus.eval.compare import ComparisonReport, finalise, legacy_outcome, paired_bootstrap
from argus.eval.evaluators import Evaluator, statistic

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


def _perp_bars(core: Any, sym: str) -> Any:
    """The perpetual's bars in nocturne's reader (``{open time in ET: (close, quote volume)}``),
    or None when no perpetual was saved. nocturne's "RNVDA" is the perpetual NVDAUSDT."""
    path = DATA / f"{sym[1:]}USDT.json"
    if not path.exists():
        return None
    if path not in _BARS:
        _BARS[path] = core.load_bars(str(path))
    return _BARS[path]


_BARS: dict[Path, Any] = {}


def _formula_move(core: Any, sym: str, fri: dt.date) -> float | None:
    """The in-module formula this harness scored before it called the console: the perpetual's
    Friday 15:00 to Sunday 19:00 ET bars through nocturne's ``near``. Kept only for the record of
    how it differs from the console (``console_replay``)."""
    bars = _perp_bars(core, sym)
    if bars is None:
        return None
    pf, pv = core.near(bars, fri, 15), core.near(bars, fri + dt.timedelta(days=2), 19)
    return pv / pf - 1 if pf and pv else None


def _console_inputs(core: Any, syms: set[str]) -> tuple[dict[str, list[tuple[float, float]]],
                                                        dict[str, list[tuple[dt.date, float]]]]:
    """What :func:`~argus.eval.overnight_comparison.console_feed` serves the console: each
    perpetual's hourly bars as ``(bar end, close)``, and each rToken's 16:00 ET price per session
    day (nocturne's ``pf`` reading) as the regular close the implied open is anchored on."""
    hourly: dict[str, list[tuple[float, float]]] = {}
    closes: dict[str, list[tuple[dt.date, float]]] = {}
    for sym in sorted(syms):
        perp = DATA / f"{sym[1:]}USDT.json"
        if perp.exists():
            rows = json.loads(perp.read_text("utf-8"))
            hourly[perp.stem] = sorted((int(r[0]) / 1000 + 3600, float(r[4])) for r in rows)
        rtoken = NOCTURNE / "data" / "1h" / f"{sym}USDT.json"
        if rtoken.exists():
            bars = core.load_bars(str(rtoken))
            days = sorted({t.date() for t in bars if t.weekday() < 5})
            closes[sym[1:]] = [(d, price) for d in days
                               if (price := core.near(bars, d, 15)) is not None]
    return hourly, closes


def predictions() -> list[dict[str, Any]]:
    """nocturne's walk-forward (`fairvalue_v2.py:10-26`) with ARGUS's column added: the console's
    implied open at Sunday 20:00 ET, stated as nocturne's target (the return from the Sunday
    rToken price). ``ARGUS`` is None where the console printed no line."""
    from argus.eval.overnight_comparison import console_feed, console_implied_open

    core = _nocturne()
    large = _large()
    rec = [r for r in core.observations() if r["sym"] in large]
    weeks = sorted({r["fri"] for r in rec})
    hourly, closes = _console_inputs(core, {r["sym"] for r in rec})
    preds = []
    with console_feed(hourly, closes):
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
                sun = r["fri"] + dt.timedelta(days=2)
                bars = _perp_bars(core, r["sym"])
                perp_last = core.near(bars, sun, 19) if bars is not None else None
                evening = dt.datetime.combine(sun, dt.time(20), core.ET)
                console = (console_implied_open(f"{r['sym'][1:]}USDT", perp_last, evening)
                           if perp_last else None)
                move = _formula_move(core, r["sym"], r["fri"])
                preds.append({
                    "sym": r["sym"], "fri": str(week), "actual": r["m"], "M0": 0.0,
                    "M1": (m1["alpha"] + m1["beta"] * r["w"]) if m1 else 0.0,
                    "M2": (m2["a0"] + m2["b1"] * r["mkt"] + m2["b2"] * r["res"]) if m2 else 0.0,
                    "MF": -r["w"],
                    "ARGUS": console[0] / r["pv"] - 1 if console else None,
                    "ARGUS_formula": ((r["pf"] * (1 + move) / r["pv"] - 1)
                                      if move is not None else None),
                    "console_anchor": console[1] if console else None, "pf": r["pf"],
                    "console_line": console[2] if console else None,
                })
    return preds


def _console_replay(preds: list[dict[str, Any]]) -> dict[str, Any]:
    """The console's line against the formula this module used to score in its place."""
    both = [p for p in preds if p["ARGUS"] is not None and p["ARGUS_formula"] is not None]
    diffs = [abs(p["ARGUS"] - p["ARGUS_formula"]) * 1e4 for p in both]
    return {
        "method": "lui/research._implied_open_line called at Sunday 20:00 ET per row, its "
                  "perpetual reader on data/h2h_nocturne/ and its close reader on the rToken's "
                  "16:00 ET price (overnight_comparison.console_feed)",
        "rows_offered": len(preds),
        "rows_the_console_answered": sum(p["ARGUS"] is not None for p in preds),
        "rows_the_former_formula_answered": sum(p["ARGUS_formula"] is not None for p in preds),
        "answered_by_one_only": [
            {"sym": p["sym"], "fri": p["fri"], "console": p["ARGUS"] is not None,
             "formula": p["ARGUS_formula"] is not None} for p in preds
            if (p["ARGUS"] is None) != (p["ARGUS_formula"] is None)],
        "anchor_differs_from_nocturnes_friday_price": [
            {"sym": p["sym"], "fri": p["fri"], "console_anchor": p["console_anchor"],
             "pf": p["pf"]} for p in preds if p["console_anchor"] is not None
            and abs(p["console_anchor"] / p["pf"] - 1) > 1e-5],
        "max_abs_diff_vs_former_formula_bps": round(max(diffs), 3) if diffs else None,
        "mean_abs_diff_vs_former_formula_bps": round(st.mean(diffs), 4) if diffs else None,
        "sample_line": next((p["console_line"] for p in preds if p["console_line"]), None),
    }


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
    """|err_a| - |err_b| in percentage points, bootstrapped over weekends (the shared unit).

    The resampling is the spine's (`eval/compare.py::paired_bootstrap`) since 2026-09-25: this
    function's former loop, moved there unchanged and pinned by `tests/test_eval_spine.py`."""
    by_week: dict[str, list[float]] = {}
    for p in preds:
        by_week.setdefault(p["fri"], []).append(
            (abs(p[a] - p["actual"]) - abs(p[b] - p["actual"])) * 100)
    boot = paired_bootstrap(by_week, draws=draws, seed=seed)
    return {"a": a, "b": b, "mean_diff_pp": round(boot.mean_diff, 3),
            "ci95_pp": [round(boot.low, 3), round(boot.high, 3)], "weekends": boot.units,
            "verdict": boot.legacy_verdict}


def score() -> dict[str, Any]:
    preds = predictions()
    both = [p for p in preds if p["ARGUS"] is not None]
    models = ("M0", "MF", "M1", "M2")
    manifest = json.loads((DATA / "manifest.json").read_text("utf-8"))
    report: dict[str, Any] = {
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
        "console_replay": _console_replay(preds),
        "limitations": ("nocturne's data ends where its clone was taken; the perpetuals are "
                        "fetched fresh over the same weekends. Symbols without a Bitget "
                        "perpetual are scored by nocturne alone and left out of the paired rows."),
    }
    report["comparison_reports"] = [r.to_dict() for r in comparison_reports(report)]
    return report


PUBLISHED = {"M0": 2.117, "MF": 1.922}
"""nocturne's own large-cap walk-forward figures (`README.md`, `scripts/fairvalue_v2.py`)."""


def _reproduces_nocturne(report: dict[str, Any]) -> Evaluator[ComparisonReport]:
    """The comparison only counts if nocturne's published baseline came out of its own code here."""
    rerun = report["baseline_reproduced"]["rerun_here"]
    return statistic("baseline_reproduced", lambda _r: (
        all(rerun.get(k) == v for k, v in PUBLISHED.items()),
        f"published {PUBLISHED}, re-run {({k: rerun.get(k) for k in PUBLISHED})}"))


def comparison_reports(report: dict[str, Any]) -> list[ComparisonReport]:
    """This harness's verdicts in the spine's shape, read from its own report dict.

    Pure, so a pre-migration artefact reads exactly like a fresh one. The outcome is the
    bootstrap's own verdict string (unrounded bounds), not re-derived from the rounded interval.
    Every report carries an extra STATISTIC-tier check, ANDed with the spine's rules: nocturne's
    published MAE must be reproduced from its clone, or no verdict against it counts.
    """
    paired = {(p["a"], p["b"]): p for p in report["paired"]}
    summary = report["summary"]
    # No per-weekend groups: a weekend holds a handful of rows, and naming a winner per weekend on
    # a point estimate would state more than seven weekends can show.
    out: list[ComparisonReport] = []
    for rival, label in (("MF", "nocturne's full fade (its best model)"),
                         ("M0", "the Sunday rToken price (no change)")):
        block = paired[("ARGUS", rival)]
        out.append(finalise(ComparisonReport(
            comparison="void", question="the rToken's Monday 10:00 ET price from Sunday 19:00",
            argus="ARGUS: the console's implied open (lui/research._implied_open_line), "
                  "anchored on the rToken's Friday price",
            rival=label, metric="mean absolute error of the Sunday-to-Monday return, pct",
            lower_is_better=True, argus_score=summary["ARGUS"]["mae_pct"],
            rival_score=summary[rival]["mae_pct"], n=block["weekends"], unit="weekend",
            outcome=legacy_outcome(block["verdict"], argus_is_a=True),
            basis="paired bootstrap over weekends, 4000 draws, seed 11; ARGUS better if the "
                  "whole 95% interval of (ARGUS - rival) absolute error is below zero",
            ci95=(block["ci95_pp"][0], block["ci95_pp"][1]),
            scored=report["rows_both"], total=report["rows_nocturne"],
            artefact="data/void_comparison.json", created_at=report["generated"]),
            extra=(_reproduces_nocturne(report),)))
    return out


def main() -> int:  # pragma: no cover - CLI
    if "--collect" in sys.argv:
        collect(_large())
    report = score()
    artefact.write(REPORT, report)
    print(json.dumps({k: report[k] for k in ("baseline_reproduced", "rows_both", "weekends",
                                             "summary", "paired")}, indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
