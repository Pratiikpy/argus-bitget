"""Diversity-aware analogue retrieval, re-proved on the test the capability already stands on.

`desk/analogue.py` and `desk/shapematch.py` gained an opt-in Maximal Marginal Relevance re-ranking
on 2026-09-25 (paper-qa, Apache-2.0, `src/paperqa/llms.py:111-170`) and a mem0-shaped score
breakdown (Apache-2.0, `mem0/utils/scoring.py:60-140`). Both modules back a standing capability
whose comparison is :mod:`argus.eval.analogstress_comparison` — AnalogDesk's own pre-registered
out-of-sample protocol, 2,698 test queries over 71 US names. A retrieval change that has not been
run through that comparison has not been shown to help, so this module runs it, on the same grid,
with the same scorer, and publishes whichever way it comes out.

**Three questions, answered separately, because one number would hide which part moved.**

1. **Did the existing comparison survive?** The default path is unchanged, so the published
   ``argusAnalogue`` and ``argusShape`` rows must be reproduced exactly from a fresh export before
   anything new is reported. If they are not, the grid or the code has drifted and every other
   number here is void.
2. **Does diversity change what is retrieved?** Near-duplicate share among the retrieved set — for
   the state matcher, analogues from one episode (`analogue.OVERLAP_WINDOW`); for the path matcher,
   pairs of windows correlated at :data:`NEAR_DUPLICATE_RHO` or more, since its exclusion zone
   already forbids temporal overlap — and the spread of retrieved outcomes.
3. **Does it change the answer?** Split the way Mind2Web splits a web agent's score
   (MIT, `src/action_prediction/metric.py:99-113`): *retriever recall* first — its "Recall Cap @ k"
   is whether a positive candidate is ranked inside the top k, i.e. whether the reasoner was handed
   anything it could have got right — and *answer accuracy* after. Here the recall cap at k is
   whether the realised return lies inside the range of the first k retrieved outcomes: the band
   could have covered it without extrapolating. Answer accuracy is the frozen-scale 80% band's
   coverage, its Winkler score, and the sign of its centre. Coverage conditional on a recall hit and
   on a miss says whether a miss is the retriever's fault or the band's.

**Selection is pre-registered and blind to the test era.** Four lambdas (:data:`LAMBDAS`) plus
the unchanged default compete on the calibration era only — multipliers fitted 2019-2020, Winkler
scored 2021-2022 (`analogstress_comparison.select_variant`) — and the test era is scored once for
the one chosen. Every variant's test-era row is published too, labelled as not selected, so a
reader can see what picking on the test would have claimed.

    python -m argus.eval.retrieval_diversity [--grid exported_grid.json]
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from argus.eval.analogstress_comparison import (
    ARGUS,
    RIVAL,
    TOP,
    WINDOW,
    Scored,
    _closes,
    _summary,
    clone_path,
    compare,
    diebold_mariano,
    export,
    run_argus,
    score,
    select_variant,
)

PACKAGE = Path(__file__).resolve().parents[3]
DATA = PACKAGE / "data"
REPORT_PATH = DATA / "retrieval_diversity.json"
PUBLISHED = DATA / "analogstress_comparison.json"

LAMBDAS = (0.9, 0.7, 0.5, 0.3)
"""MMR lambdas tried, fixed before the run. 1.0 (no diversity) is the incumbent."""

RECALL_KS = (5, 10, 20, 50)
"""Mind2Web reports its recall cap at 5/10/20/50 (`metric.py:99`); the same cut-offs, and 50 is
also AnalogDesk's k."""

NEAR_DUPLICATE_RHO = 0.9
"""Two retrieved windows correlated at least this strongly are one shape counted twice."""

FAMILIES = {"argusAnalogue": "analogue", "argusShape": "shape"}


def arm_key(base: str, lam: float) -> str:
    return f"{base}MMR{round(lam * 100)}"


@dataclass(frozen=True)
class Retrieval:
    """What one arm retrieved for one query, before any band is built."""

    outcomes: tuple[float, ...]
    """Forward returns of the analogues, in pick order, as fractions."""
    near_duplicate_share: float
    effective_share: float
    """Independent episodes over analogues — 1.0 means no two share an episode."""


def recall_cap(outcomes: Sequence[float], y: float, k: int) -> bool:
    """Mind2Web's recall cap, transposed: is ``y`` inside the range of the first ``k`` outcomes?

    `metric.py:99-105` counts a query as recallable at k when any positive candidate ranks below k.
    A band built from analogues cannot cover a realised return outside the range they span without
    extrapolating, so that range is the retriever's ceiling on the answer.
    """
    head = [o for o in outcomes[:k] if math.isfinite(o)]
    return bool(head) and min(head) <= y <= max(head)


def _analogue(closes: list[tuple[datetime, float]], horizon: int, y: float,
              lams: Sequence[float | None]) -> dict[str, tuple[dict[str, float] | None,
                                                            Retrieval | None]]:
    from argus.desk.analogue import corpus_from_closes, current_state, find

    out: dict[str, tuple[dict[str, float] | None, Retrieval | None]] = {}
    query = current_state(closes, window=WINDOW)
    corpus = corpus_from_closes(closes, symbol="X", window=WINDOW, horizon=horizon)
    for lam in lams:
        key = "argusAnalogue" if lam is None else arm_key("argusAnalogue", lam)
        if query is None:
            out[key] = (None, None)
            continue
        report = find(query=query, corpus=corpus, as_of=closes[-1][0], limit=TOP, diversity=lam)
        if report.distribution is None:
            out[key] = (None, None)
            continue
        d = report.distribution
        outcomes = tuple(r / 10_000 for r in d.returns_bps)
        out[key] = (_summary(list(outcomes), y), Retrieval(
            outcomes=outcomes,
            near_duplicate_share=1.0 - d.effective_n / d.count,
            effective_share=d.effective_n / d.count,
        ))
    return out


def _shape(closes: list[tuple[datetime, float]], horizon: int, y: float,
           lams: Sequence[float | None]) -> dict[str, tuple[dict[str, float] | None,
                                                         Retrieval | None]]:
    from argus.desk.shapematch import AnalogueError, _correlation, find

    out: dict[str, tuple[dict[str, float] | None, Retrieval | None]] = {}
    prices = [c for _, c in closes]
    for lam in lams:
        key = "argusShape" if lam is None else arm_key("argusShape", lam)
        try:
            report = find(closes, symbol="X", window=WINDOW, horizon=horizon, top=TOP, trials=0,
                          diversity=lam)
        except AnalogueError:
            out[key] = (None, None)
            continue
        kept = [a for a in report.analogues if a.forward_pct is not None]
        outcomes = tuple(a.forward_pct / 100 for a in kept if a.forward_pct is not None)
        windows = [prices[a.start_index:a.start_index + WINDOW] for a in kept]
        dup = 0
        for i, w in enumerate(windows):
            if any(_correlation(w, windows[j]) >= NEAR_DUPLICATE_RHO for j in range(i)):
                dup += 1
        share = dup / len(windows) if windows else math.nan
        out[key] = (_summary(list(outcomes), y), Retrieval(outcomes, share, 1.0))
    return out


def run_arms(grid: dict[str, Any], lams: Sequence[float] = LAMBDAS
             ) -> dict[tuple[str, int], dict[str, Retrieval | None]]:
    """Every query answered by every arm, bands written into the rows in place (the shape
    `analogstress_comparison.score` reads), retrievals returned per query for the diagnostics."""
    horizon = int(grid["horizon"])
    dates: list[str] = grid["dates"]
    arms: list[float | None] = [None, *lams]
    retrieved: dict[tuple[str, int], dict[str, Retrieval | None]] = {}
    rows = grid["rows"]
    for n, row in enumerate(rows, start=1):
        closes = _closes(grid["adjusted"][row["sym"]], dates, int(row["q"]))
        y = float(row["y"])
        got = {**_analogue(closes, horizon, y, arms), **_shape(closes, horizon, y, arms)}
        per: dict[str, Retrieval | None] = {}
        for key, (band, retrieval) in got.items():
            row[key] = band
            if band is not None:
                row[f"{key}_pit"] = band["pit"]
            per[key] = retrieval
        retrieved[(row["sym"], int(row["q"]))] = per
        if n % 500 == 0:
            print(f"  {n}/{len(rows)} queries", file=sys.stderr)
    return retrieved


def diagnostics(test: list[dict[str, Any]], key: str, scored: Scored,
                retrieved: dict[tuple[str, int], dict[str, Retrieval | None]]) -> dict[str, Any]:
    """Retriever recall and answer accuracy for one arm, reported apart (Mind2Web's split)."""
    recall: dict[int, list[bool]] = {k: [] for k in RECALL_KS}
    cover_given: dict[bool, list[bool]] = {True: [], False: []}
    near_dup: list[float] = []
    spread: list[float] = []
    direction: list[bool] = []
    for row in test:
        retrieval = retrieved.get((row["sym"], int(row["q"])), {}).get(key)
        band = row.get(key)
        if retrieval is None or not band:
            continue
        y = float(row["y"])
        for k in RECALL_KS:
            recall[k].append(recall_cap(retrieval.outcomes, y, k))
        if math.isfinite(retrieval.near_duplicate_share):
            near_dup.append(retrieval.near_duplicate_share)
        if len(retrieval.outcomes) > 1:
            spread.append(statistics.stdev(retrieval.outcomes))
        half = float(band["hw"]) * scored.scale
        covered = abs(y - float(band["centre"])) <= half
        cover_given[recall_cap(retrieval.outcomes, y, TOP)].append(covered)
        if y != 0 and band["centre"] != 0:
            direction.append((y > 0) == (float(band["centre"]) > 0))

    def rate(xs: list[bool]) -> float | None:
        return round(sum(xs) / len(xs), 4) if xs else None

    return {
        "queries": len(near_dup),
        "retriever": {
            "recall_cap": {f"@{k}": rate(v) for k, v in recall.items()},
            "near_duplicate_share": round(statistics.fmean(near_dup), 4) if near_dup else None,
            "mean_outcome_stdev_pct": round(statistics.fmean(spread) * 100, 3) if spread else None,
        },
        "answer": {
            "coverage_pct": round(scored.coverage * 100, 2),
            "winkler_pct": round(scored.winkler * 100, 3),
            "direction_accuracy": rate(direction),
            "coverage_given_recall_hit": rate(cover_given[True]),
            "coverage_given_recall_miss": rate(cover_given[False]),
            "recall_misses": len(cover_given[False]),
        },
    }


def reproduce_published(grid: dict[str, Any]) -> dict[str, Any]:
    """Re-run the existing comparison end to end on this grid and diff it against the artefact.

    Uses `analogstress_comparison.run_argus` and `compare` unmodified, on a deep copy of the grid so
    the diversity arms cannot leak into it.
    """
    fresh = json.loads(json.dumps(grid))
    run_argus(fresh)
    rival_file = clone_path() / "research" / "validation-results.json"
    rival = json.loads(rival_file.read_text(encoding="utf-8")) if rival_file.exists() else None
    rerun = compare(fresh, rival)
    out: dict[str, Any] = {"verdict": rerun["verdict"],
                           "reproduced_rival": rerun["reproduced_rival"].get("reproduced")}
    if not PUBLISHED.exists():
        out["published_available"] = False
        return out
    published = json.loads(PUBLISHED.read_text(encoding="utf-8"))
    diffs: dict[str, float] = {}
    for key in (*ARGUS, *RIVAL):
        a, b = rerun["predictors"].get(key), published["predictors"].get(key)
        if a and b:
            diffs[key] = abs(float(a["winkler_pct"]) - float(b["winkler_pct"]))
    out.update({
        "published_available": True,
        "published_verdict": published["verdict"],
        "verdict_unchanged": rerun["verdict"] == published["verdict"],
        "max_abs_winkler_pct_difference": max(diffs.values()) if diffs else None,
        "argus_rows": {k: rerun["predictors"].get(k) for k in ARGUS},
    })
    return out


def evaluate(grid: dict[str, Any], retrieved: dict[tuple[str, int], dict[str, Retrieval | None]],
             lams: Sequence[float] = LAMBDAS) -> dict[str, Any]:
    target = float(grid["coverage"])
    rows = grid["rows"]
    calib = [r for r in rows if r["era"] == "calibration"]
    test = [r for r in rows if r["era"] == "test"]
    rival = {k: score(calib, test, k, target) for k in RIVAL if any(r.get(k) for r in test)}
    best_rival = min(rival.values(), key=lambda s: s.winkler)
    families: dict[str, Any] = {}
    for base, family in FAMILIES.items():
        keys = [base, *(arm_key(base, lam) for lam in lams)]
        selection = select_variant(rows, keys, target)
        chosen = selection["chosen"]
        scored = {k: score(calib, test, k, target) for k in keys}
        dm_vs_base = diebold_mariano(scored[chosen], scored[base], test)
        dm_vs_rival = diebold_mariano(scored[chosen], best_rival, test)
        gap = (scored[chosen].winkler - scored[base].winkler) * 100
        if chosen == base:
            verdict = (f"NO CHANGE: the calibration era selected no diversity for {family}; "
                       f"the default stays nearest-first")
        elif isinstance(dm_vs_base.get("p"), float) and dm_vs_base["p"] < 0.05:
            verdict = (f"{'DIVERSITY WINS' if gap < 0 else 'DIVERSITY LOSES'}: {chosen} Winkler "
                       f"{scored[chosen].winkler * 100:.3f}% against {base} "
                       f"{scored[base].winkler * 100:.3f}% ({gap:+.3f} points, DM p="
                       f"{dm_vs_base['p']})")
        else:
            verdict = (f"NO SIGNIFICANT DIFFERENCE: {chosen} Winkler "
                       f"{scored[chosen].winkler * 100:.3f}% against {base} "
                       f"{scored[base].winkler * 100:.3f}% ({gap:+.3f} points, DM p="
                       f"{dm_vs_base.get('p')})")
        families[family] = {
            "incumbent": base,
            "selection": selection,
            "chosen": chosen,
            "verdict": verdict,
            "diebold_mariano_vs_incumbent": dm_vs_base,
            f"diebold_mariano_vs_best_rival_{best_rival.predictor}": dm_vs_rival,
            "arms": {k: {"selected": k == chosen, **scored[k].as_dict(),
                         **diagnostics(test, k, scored[k], retrieved)} for k in keys},
        }
    return {
        "best_rival": best_rival.as_dict(),
        "families": families,
    }


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI, runs the rival
    from argus.eval.artefact import write

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--grid", type=Path, help="a grid exported by analogdesk_export.mjs")
    args = parser.parse_args(argv)
    grid = (json.loads(args.grid.read_text(encoding="utf-8")) if args.grid else export())
    published = reproduce_published(grid)
    print("existing comparison rerun:", published.get("verdict"))
    print("unchanged:", published.get("verdict_unchanged"),
          "max |ΔWinkler|:", published.get("max_abs_winkler_pct_difference"))
    retrieved = run_arms(grid)
    result = evaluate(grid, retrieved)
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "protocol": {
            "grid": "AnalogDesk validation grid, horizon 5 sessions, 80% nominal",
            "test_queries": sum(1 for r in grid["rows"] if r["era"] == "test"),
            "calibration_queries": sum(1 for r in grid["rows"] if r["era"] == "calibration"),
            "lambdas": list(LAMBDAS), "k": TOP, "window": WINDOW,
            "selection": "calibration era only: multipliers 2019-2020, Winkler 2021-2022",
            "near_duplicate": {
                "analogue": "share of analogues sharing an episode (OVERLAP_WINDOW)",
                "shape": f"share of windows correlated >= {NEAR_DUPLICATE_RHO} with an earlier "
                         f"pick",
            },
            "recall_cap": "realised return inside the range of the first k retrieved outcomes "
                          "(Mind2Web metric.py:99-105, transposed)",
        },
        "existing_comparison_rerun": published,
        **result,
        "qwen_calls": 0,
    }
    undefined = write(REPORT_PATH, report)
    for family, block in result["families"].items():
        print(f"{family}: {block['verdict']}")
        for key, arm in block["arms"].items():
            print(f"  {key:22} Winkler {arm['winkler_pct']:7.3f}%  cov {arm['coverage_pct']:5.1f}%"
                  f"  near-dup {arm['retriever']['near_duplicate_share']}  "
                  f"recall@50 {arm['retriever']['recall_cap']['@50']}")
    if undefined:
        print("undefined values written as null:", undefined)
    print(f"written to {REPORT_PATH}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["LAMBDAS", "NEAR_DUPLICATE_RHO", "RECALL_KS", "Retrieval", "arm_key", "diagnostics",
           "evaluate", "recall_cap", "reproduce_published", "run_arms"]
