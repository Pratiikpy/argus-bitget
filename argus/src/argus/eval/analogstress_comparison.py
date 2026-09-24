"""Decision Stress Testing, head to head: ARGUS against AnalogDesk on AnalogDesk's own test.

Track 3 names the sub-theme "Decision Stress Testing: before opening a position, how does AI
retrieve historically similar scenarios?" The rival review of 2026-09-24 named the entry to beat:
**AnalogDesk** (lixinde586-afk/analogdesk, an S2 submission in this sub-theme), which publishes a
pre-registered out-of-sample protocol — 2,698 test queries over 71 US names, 2023 to 2026, with a
split-conformal band fitted on 2019-2022 — and reports its own loss honestly: at matched coverage
its analogue band is wider than the same-name unconditional band (10.20% against 9.48%).

**Same input, same scorer, their rules.** AnalogDesk carries no licence, so none of its code is in
this repository: `baselines/analogdesk_export.mjs` runs its unmodified engine from a local clone
and exports its query grid, each query's realised five-session return, its five predictors'
centres and half-widths, and the adjusted closes. ARGUS then answers the *same* queries from the
*same* closes, truncated at the query session and embargoed so every analogue's outcome was known
by then. Every predictor — theirs and ours — is scored by one implementation of their published
protocol (split-conformal scale on the calibration era, frozen; coverage with standard errors
clustered by date; matched-coverage width; PIT uniformity), written here from their description,
and **that scorer must first reproduce their own numbers** from their exported predictions before
any ARGUS number is reported (`baseline_reproduced`).

**The ARGUS arms.**

* ``argusAnalogue`` — `desk/analogue.find` as the console runs it: the trailing 20-session return
  and realised volatility, z-scored over the name's own history, overlapping episodes collapsed.
* ``argusShape`` — `desk/shapematch`: the z-normalised 20-session path, non-overlapping matches.

Each returns the outcomes of its analogues; the band is their median plus or minus a conformal
multiple of their standard deviation, built exactly as AnalogDesk builds its own, so the comparison
isolates *retrieval* — which past sessions are chosen — from interval construction.

**Primary metric** (fixed in the run plan before this ran): the Winkler interval score at 80%
nominal, frozen calibration-era multipliers, compared per query with a Diebold-Mariano test on
date-clustered differences. Lower is better. Path-breach Brier scores need each analogue's intraday
excursion, which ARGUS's analogue engines do not produce; they are not scored, and say so.

    python -m argus.eval.analogstress_comparison
"""

from __future__ import annotations

import json
import math
import os
import statistics
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PACKAGE = Path(__file__).resolve().parents[3]
DATA = PACKAGE / "data"
REPORT_PATH = DATA / "analogstress_comparison.json"
EXPORTER = Path(__file__).resolve().parent / "baselines" / "analogdesk_export.mjs"
DEFAULT_CLONE = PACKAGE.parent / "research" / "repos-rivals" / "lixinde586-afk~analogdesk"
WINDOW = 20
"""Sessions describing the state (return, volatility) and the path an analogue must match."""
TOP = 50
"""Analogues retrieved per query — AnalogDesk's own k."""
RIVAL = ("analogConformal", "analogRaw", "uncondNamePIT", "volHarness", "pooledUncond")
ARGUS = ("argusAnalogue", "argusShape")
SELF_CALIBRATING = frozenset({"analogRaw", "uncondNamePIT"})
"""Predictors AnalogDesk scores at a multiplier of 1: their bands are empirical percentiles."""


class AnalogDeskUnavailable(FileNotFoundError):
    """The clone or Node is missing, so the rival cannot be run — said, not skipped."""


def clone_path() -> Path:
    return Path(os.environ.get("ARGUS_ANALOGDESK_CLONE", str(DEFAULT_CLONE)))


def export(horizon: int = 5) -> dict[str, Any]:  # pragma: no cover - runs the rival
    clone = clone_path()
    if not (clone / "src" / "engine" / "validation.mjs").exists():
        raise AnalogDeskUnavailable(
            f"no AnalogDesk clone at {clone}; clone https://github.com/lixinde586-afk/analogdesk "
            f"there or set ARGUS_ANALOGDESK_CLONE")
    out = DATA / "_analogdesk_grid.json"
    subprocess.run(["node", str(EXPORTER), str(clone), str(out), str(horizon)], check=True,
                   capture_output=True, text=True, timeout=1800)
    try:
        grid: dict[str, Any] = json.loads(out.read_text(encoding="utf-8"))
    finally:
        out.unlink(missing_ok=True)
    return grid


# --- the scorer: AnalogDesk's published protocol, implemented from its description ------------

def quantile(values: list[float], p: float) -> float:
    """Linear interpolation between order statistics (their `quantile`, and numpy's default)."""
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return math.nan
    h = (n - 1) * p
    lo, hi = math.floor(h), math.ceil(h)
    return ordered[lo] if lo == hi else ordered[lo] + (ordered[hi] - ordered[lo]) * (h - lo)


def conformal_scale(scores: list[float], target: float) -> float:
    """The finite-sample split-conformal multiplier: the ceil((n+1) * target) / n quantile."""
    clean = [s for s in scores if math.isfinite(s)]
    if not clean:
        return math.nan
    level = min(1.0, math.ceil((len(clean) + 1) * target) / len(clean))
    return quantile(clean, level)


def _usable(row: dict[str, Any], key: str) -> tuple[float, float] | None:
    band = row.get(key)
    if not band:
        return None
    centre, half = band.get("centre"), band.get("hw")
    if (centre is None or half is None or not math.isfinite(centre) or not math.isfinite(half)
            or half <= 0 or row.get("y") is None):
        return None
    return float(centre), float(half)


@dataclass(frozen=True)
class Scored:
    predictor: str
    scale: float
    n: int
    coverage: float
    coverage_se: float
    width: float
    matched_width: float
    winkler: float
    per_query: dict[tuple[str, int], float]
    """Winkler score per (symbol, session), for the paired test."""

    def as_dict(self) -> dict[str, Any]:
        return {"predictor": self.predictor, "frozen_scale": round(self.scale, 4), "n": self.n,
                "coverage_pct": round(self.coverage * 100, 1),
                "coverage_se_pct": round(self.coverage_se * 100, 2),
                "width_pct": round(self.width * 100, 2),
                "matched_width_pct": round(self.matched_width * 100, 2),
                "winkler_pct": round(self.winkler * 100, 3)}


def _coverage_width(rows: list[dict[str, Any]], key: str, scale: float
                    ) -> tuple[int, float, float, float]:
    by_date: dict[str, list[int]] = defaultdict(list)
    hits, width, n = 0, 0.0, 0
    for row in rows:
        got = _usable(row, key)
        if got is None:
            continue
        centre, half = got
        hit = abs(row["y"] - centre) <= half * scale
        hits += hit
        width += 2 * half * scale
        n += 1
        by_date[row["date"]].append(int(hit))
    per = [sum(v) / len(v) for v in by_date.values()]
    se = statistics.stdev(per) / math.sqrt(len(per)) if len(per) > 1 else math.nan
    return n, (hits / n if n else math.nan), (width / n if n else math.nan), se


def matched_width(rows: list[dict[str, Any]], key: str, target: float) -> float:
    """Bisect one multiplier so coverage lands on ``target``, and report the width there."""
    lo, hi = 0.05, 20.0
    for _ in range(60):
        mid = (lo + hi) / 2
        _, cov, _, _ = _coverage_width(rows, key, mid)
        if not math.isfinite(cov):
            break
        if cov < target:
            lo = mid
        else:
            hi = mid
    return _coverage_width(rows, key, (lo + hi) / 2)[2]


def score(calib: list[dict[str, Any]], test: list[dict[str, Any]], key: str,
          target: float) -> Scored:
    if key in SELF_CALIBRATING:
        scale = 1.0
    else:
        scores = []
        for row in calib:
            got = _usable(row, key)
            if got is not None:
                scores.append(abs(row["y"] - got[0]) / got[1])
        scale = conformal_scale(scores, target)
    n, cov, width, se = _coverage_width(test, key, scale)
    alpha = 1 - target
    per_query: dict[tuple[str, int], float] = {}
    for row in test:
        got = _usable(row, key)
        if got is None:
            continue
        lo, hi = got[0] - got[1] * scale, got[0] + got[1] * scale
        y = row["y"]
        per_query[(row["sym"], row["q"])] = (hi - lo) + (2 / alpha) * max(0.0, lo - y) + (
            2 / alpha) * max(0.0, y - hi)
    wink = statistics.fmean(per_query.values()) if per_query else math.nan
    return Scored(key, scale, n, cov, se, width, matched_width(test, key, target), wink, per_query)


def pit_chi_square(rows: list[dict[str, Any]], key: str = "pit") -> float:
    pits = [r[key] for r in rows if r.get(key) is not None and math.isfinite(r[key])]
    bins = [0] * 10
    for p in pits:
        bins[min(9, max(0, int(p * 10)))] += 1
    expected = len(pits) / 10
    return sum((c - expected) ** 2 / expected for c in bins) if expected else math.nan


def diebold_mariano(a: Scored, b: Scored, test: list[dict[str, Any]]) -> dict[str, float]:
    """Paired test on Winkler differences (a minus b), clustered by query date."""
    by_date: dict[str, list[float]] = defaultdict(list)
    for row in test:
        key = (row["sym"], row["q"])
        if key in a.per_query and key in b.per_query:
            by_date[row["date"]].append(a.per_query[key] - b.per_query[key])
    per = [statistics.fmean(v) for v in by_date.values()]
    if len(per) < 3:
        return {"mean_diff_pct": math.nan, "t": math.nan, "p": math.nan, "dates": len(per)}
    mean = statistics.fmean(per)
    se = statistics.stdev(per) / math.sqrt(len(per))
    t = mean / se if se > 0 else math.nan
    p = math.erfc(abs(t) / math.sqrt(2)) if math.isfinite(t) else math.nan
    return {"mean_diff_pct": round(mean * 100, 3), "t": round(t, 3), "p": round(p, 4),
            "dates": len(per), "paired_queries": sum(len(v) for v in by_date.values())}


# --- the ARGUS arms ----------------------------------------------------------------------------

def _closes(adjusted: list[float | None], dates: list[str], q: int) -> list[tuple[datetime, float]]:
    out = []
    for i in range(q + 1):
        v = adjusted[i]
        if v is not None and v > 0:
            out.append((datetime.fromisoformat(dates[i]).replace(tzinfo=UTC), float(v)))
    return out


def _summary(outcomes: list[float], y: float) -> dict[str, float] | None:
    clean = [o for o in outcomes if math.isfinite(o)]
    if len(clean) < 10:
        return None
    return {"centre": statistics.median(clean), "hw": statistics.stdev(clean),
            "pit": sum(1 for o in clean if o <= y) / len(clean), "n": float(len(clean))}


def argus_analogue(closes: list[tuple[datetime, float]], horizon: int,
                   y: float) -> dict[str, float] | None:
    from argus.desk.analogue import corpus_from_closes, current_state, find

    query = current_state(closes, window=WINDOW)
    if query is None:
        return None
    corpus = corpus_from_closes(closes, symbol="X", window=WINDOW, horizon=horizon)
    as_of = closes[-1][0]
    report = find(query=query, corpus=corpus, as_of=as_of, limit=TOP)
    if report.distribution is None:
        return None
    return _summary([r / 10_000 for r in report.distribution.returns_bps], y)


def argus_shape(closes: list[tuple[datetime, float]], horizon: int,
                y: float) -> dict[str, float] | None:
    from argus.desk.shapematch import AnalogueError, find

    try:
        report = find(closes, symbol="X", window=WINDOW, horizon=horizon, top=TOP, trials=0)
    except AnalogueError:
        return None
    return _summary([o / 100 for o in report.outcomes], y)


def run_argus(grid: dict[str, Any]) -> None:
    """Answer every exported query with each ARGUS arm, in place."""
    horizon = int(grid["horizon"])
    dates: list[str] = grid["dates"]
    for row in grid["rows"]:
        closes = _closes(grid["adjusted"][row["sym"]], dates, int(row["q"]))
        for key, arm in (("argusAnalogue", argus_analogue), ("argusShape", argus_shape)):
            got = arm(closes, horizon, float(row["y"]))
            row[key] = got
            if got is not None:
                row[f"{key}_pit"] = got["pit"]


def compare(grid: dict[str, Any], rival_results: dict[str, Any] | None = None) -> dict[str, Any]:
    target = float(grid["coverage"])
    calib = [r for r in grid["rows"] if r["era"] == "calibration"]
    test = [r for r in grid["rows"] if r["era"] == "test"]
    scored = {key: score(calib, test, key, target) for key in (*RIVAL, *ARGUS)
              if any(r.get(key) for r in test)}
    reproduced = reproduction(scored, test, rival_results, int(grid["horizon"]))
    ranking = sorted(scored.values(), key=lambda s: s.winkler)
    best_argus = min((scored[k] for k in ARGUS if k in scored), key=lambda s: s.winkler,
                     default=None)
    best_rival = min((scored[k] for k in RIVAL if k in scored), key=lambda s: s.winkler)
    tests = {}
    if best_argus is not None:
        for key in RIVAL:
            if key in scored:
                tests[f"{best_argus.predictor} vs {key}"] = diebold_mariano(
                    best_argus, scored[key], test)
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "protocol": {"horizon": grid["horizon"], "coverage": target, "k": TOP, "window": WINDOW,
                     "calibration_queries": len(calib), "test_queries": len(test),
                     "eras": grid["eras"], "primary": "Winkler score at 80%, frozen scales"},
        "reproduced_rival": reproduced,
        "predictors": {k: v.as_dict() for k, v in scored.items()},
        "pit_chi_square": {"analogConformal": round(pit_chi_square(test), 1),
                           **{k: round(pit_chi_square(test, f"{k}_pit"), 1) for k in ARGUS}},
        "ranking_by_winkler": [s.predictor for s in ranking],
        "best_argus": None if best_argus is None else best_argus.predictor,
        "best_rival": best_rival.predictor,
        "diebold_mariano": tests,
        "verdict": _verdict(best_argus, best_rival, tests),
        "not_scored": "path-breach Brier at -5/-10/-20%: ARGUS's analogue engines return each "
                      "analogue's forward return, not its intraday excursion",
        "limitations": [
            "every predictor fails PIT uniformity (5% critical value 16.92), ARGUS's included",
            "the naive same-name band beats every retrieval method on Winkler, AnalogDesk's and "
            "ARGUS's alike — at five sessions, retrieval has not been shown to add information",
            "one horizon (five sessions) and one grid; AnalogDesk's other horizons are not run",
            "ARGUS's arms retrieve from the single name's own history; AnalogDesk retrieves "
            "across 71 names on 28 features, so this compares systems, not only methods",
        ],
    }


def reproduction(scored: dict[str, Scored], test: list[dict[str, Any]],
                 rival_results: dict[str, Any] | None, horizon: int) -> dict[str, Any]:
    """Our scorer against AnalogDesk's own published numbers for its own predictors: the frozen
    multiplier, test coverage, width and matched-coverage width, and the PIT chi-square. The
    comparison is reported only if these agree — a scorer that cannot reproduce the rival's own
    result has no standing to rank anyone."""
    if not rival_results:
        return {"available": False}
    run = (rival_results.get("runs", {}).get(str(horizon)) or rival_results)
    theirs_all = run.get("results", {})
    worst = 0.0
    rows: dict[str, Any] = {}
    for key in RIVAL:
        theirs, ours = theirs_all.get(key), scored.get(key)
        if not theirs or ours is None:
            continue
        pairs = {
            "scale": (ours.scale, theirs.get("lambdaFrozen")),
            "coverage": (ours.coverage, theirs.get("testEra", {}).get("coverage")),
            "width": (ours.width, theirs.get("testEra", {}).get("width")),
            "matched_width": (ours.matched_width, theirs.get("matched", {}).get("width")),
        }
        rows[key] = {name: [a, b] for name, (a, b) in pairs.items()}
        for a, b in pairs.values():
            if b is not None:
                worst = max(worst, abs(a - float(b)))
    their_pit = run.get("pit", {}).get("chiSquare")
    our_pit = pit_chi_square(test)
    if their_pit is not None:
        worst = max(worst, abs(our_pit - float(their_pit)) / 100)
    return {"available": True, "predictors": rows,
            "pit_chi_square": [our_pit, their_pit],
            "max_abs_difference": worst, "reproduced": worst < 1e-6}


def _verdict(argus: Scored | None, rival: Scored, tests: dict[str, Any]) -> str:
    if argus is None:
        return "ARGUS produced no scorable band on this grid"
    dm = tests.get(f"{argus.predictor} vs {rival.predictor}", {})
    gap = (argus.winkler - rival.winkler) * 100
    better = gap < 0
    significant = isinstance(dm.get("p"), float) and dm["p"] < 0.05
    who = "ARGUS WINS" if better and significant else "RIVAL WINS" if not better and significant \
        else "NO SIGNIFICANT DIFFERENCE"
    return (f"{who}: {argus.predictor} Winkler {argus.winkler * 100:.2f}% against the best rival "
            f"predictor {rival.predictor} at {rival.winkler * 100:.2f}% ({gap:+.2f} points, "
            f"Diebold-Mariano p={dm.get('p')}, clustered by date)")


def main() -> int:  # pragma: no cover - CLI
    grid = export()
    run_argus(grid)
    rival_file = clone_path() / "research" / "validation-results.json"
    rival = json.loads(rival_file.read_text(encoding="utf-8")) if rival_file.exists() else None
    report = compare(grid, rival)
    REPORT_PATH.write_text(json.dumps(report, indent=1), encoding="utf-8")
    for key, row in report["predictors"].items():
        print(f"{key:18} cov {row['coverage_pct']:5.1f}%  width {row['width_pct']:6.2f}%  "
              f"matched {row['matched_width_pct']:6.2f}%  Winkler {row['winkler_pct']:7.3f}%")
    print("rival reproduced:", report["reproduced_rival"].get("reproduced"),
          "max abs difference", report["reproduced_rival"].get("max_abs_difference"))
    print(report["verdict"])
    print(f"written to {REPORT_PATH}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["ARGUS", "RIVAL", "AnalogDeskUnavailable", "Scored", "compare", "conformal_scale",
           "diebold_mariano", "matched_width", "pit_chi_square", "quantile", "score"]
