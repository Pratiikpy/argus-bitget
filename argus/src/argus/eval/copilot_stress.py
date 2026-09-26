"""Portfolio Copilot stress, head to head: "if QQQ falls this much, what does my book do?"

Every ARGUS add-to-book answer prints the sentence "if QQQ falls 5%, the book moves about -6.2%
through beta alone" (`desk/portfolio.stress_by_beta`). This harness scores that sentence on the
days the market actually fell, against the method rival the review of 2026-09-24 named for the
stress half of the sub-theme: **skfolio** (BSD-3, 1.3.1), whose regular-vine copula samples the
book conditional on the benchmark's move and whose Entropy Pooling reweights history to the same
view. Both skfolio engines run unmodified in their own interpreter
(`baselines/skfolio_stress.py`); ARGUS's arm is the production function.

**Question scored.** On each held-out UTC day on which QQQUSDT fell at least 1%, each system is
given that day's realised QQQUSDT move and everything known before the day began, and states the
book's move. The realised move of the same book is the target. This is how a stress sentence is
validated: conditional on the shock, not forecasting the shock.

**Arms.**

* ``argus_beta`` — production: open-session hourly betas over the 30 days before the day.
* ``argus_blended`` — the same over every hourly bar.
* ``daily_beta`` — each name's daily beta over every prior day: the same window skfolio sees, so
  window length and method can be told apart.
* ``skfolio_vine`` — mean of 2,000 vine samples with QQQ pinned to the shock; also its 10% quantile.
* ``skfolio_ep`` — Entropy Pooling posterior mean with the view QQQ == shock; also its weighted 10%
  quantile.

**Pre-registered** (:data:`PREREG`, hashed into the report): primary = mean absolute error of the
book's move in percentage points on every day with QQQUSDT down 1% or more, ``argus_beta`` against
``skfolio_vine``, Wilcoxon signed-rank over per-day mean errors (the 200 books on one day share one
market path). Secondary: the 2% subset, every other pair, pinball loss at q = 0.1 for the two
distributional arms (ARGUS states a point, so it is scored as a point, and said so).

    python -m argus.eval.copilot_stress
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import random
import statistics
import subprocess
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any

from argus.desk import portfolio
from argus.eval.copilot_rivals import BENCHMARK, NAMES, books, load_bitget

PACKAGE = Path(__file__).resolve().parents[3]
REPORT_PATH = PACKAGE / "data" / "copilot_stress.json"
RUNNER = Path(__file__).resolve().parent / "baselines" / "skfolio_stress.py"
DOWN_DAY = -0.01
SEVERE_DAY = -0.02
MIN_HISTORY_DAYS = 60
LOOKBACK_DAYS = 30
BOOKS = 200
N_SAMPLES = 2000
QUANTILE = 0.1
SEED = 20260925

PREREG: dict[str, Any] = {
    "question": "given the day's realised QQQUSDT move, how far is each system's stated book move "
                "from the book's realised move",
    "event_rule": f"UTC daily QQQUSDT return <= {DOWN_DAY}, at least {MIN_HISTORY_DAYS} prior days",
    "stale_rule": "days on which any instrument's close is unchanged are removed from every arm's "
                  "daily history and from the events",
    "books": BOOKS, "seed": SEED, "names": list(NAMES), "benchmark": BENCHMARK,
    "arms": ["argus_beta", "argus_blended", "daily_beta", "skfolio_vine", "skfolio_ep"],
    "primary": {"metric": "mean absolute error of the book's daily move, percentage points",
                "comparison": "argus_beta vs skfolio_vine",
                "test": "Wilcoxon signed-rank over per-day mean errors"},
    "secondary": [f"days with QQQUSDT <= {SEVERE_DAY}", "every other arm pair",
                  f"pinball loss at q = {QUANTILE} for skfolio_vine and skfolio_ep"],
    "n_samples": N_SAMPLES,
}


def prereg_hash() -> str:
    return hashlib.sha256(json.dumps(PREREG, sort_keys=True).encode()).hexdigest()


def daily_returns(bitget: Mapping[str, Sequence[tuple[datetime, float]]]
                  ) -> tuple[list[date], dict[str, list[float]]]:
    """UTC-day close-to-close returns for every name and the benchmark, on days all of them have."""
    closes: dict[str, dict[date, float]] = {}
    for name, series in bitget.items():
        per_day: dict[date, float] = {}
        for t, c in series:
            per_day[t.date()] = c
        closes[name] = per_day
    days = sorted(set.intersection(*(set(v) for v in closes.values())))
    stamps = days[1:]
    return stamps, {n: [closes[n][b] / closes[n][a] - 1 for a, b in itertools.pairwise(days)]
                    for n in closes}


def stale(rets: Mapping[str, Sequence[float]], i: int) -> bool:
    """A day on which some instrument did not trade: its close is exactly the previous close.

    **Found by the rival, not by us.** Bitget's stock perpetuals did not trade over weekends and
    US nights from November 2025 to January 2026 (30-39% of hourly bars flat, against 1% from
    March), so those UTC days carry a return of exactly zero. The first run fed them to every arm;
    skfolio's vine fitted a Student-t with 0.13 degrees of freedom and a scale of 1e-14 to them and
    sampled returns of 10^8. Unchanged prices are not observations of how the market co-moves, so
    they are removed from every arm's history alike."""
    return any(series[i] == 0.0 for series in rets.values())


def events(stamps: Sequence[date], rets: Mapping[str, Sequence[float]]) -> list[int]:
    return [i for i in range(MIN_HISTORY_DAYS, len(stamps))
            if rets[BENCHMARK][i] <= DOWN_DAY and not stale(rets, i)]


def history(rets: Mapping[str, Sequence[float]], i: int) -> list[int]:
    """Indices of the traded days before day ``i``."""
    return [t for t in range(i) if not stale(rets, t)]


def residual_quantile(weights: Mapping[str, float], daily: Mapping[str, Sequence[float]],
                      betas: Mapping[str, float], q: float = QUANTILE) -> float:
    """The ``q`` quantile of the book's daily move left unexplained by its beta, centred.

    **Added after the first scored run, and not pre-registered.** That run showed skfolio's vine
    beating ARGUS on the tail (pinball at q = 0.1) for the plain reason that ARGUS stated no tail at
    all. This is the smallest honest tail a linear stress can carry: the point from beta, plus how
    far below its beta line this book has actually landed one day in ten."""
    n = len(daily[BENCHMARK])
    book_beta = sum(w * betas[s] for s, w in weights.items())
    resid = [sum(w * daily[s][t] for s, w in weights.items()) - book_beta * daily[BENCHMARK][t]
             for t in range(n)]
    centre = statistics.fmean(resid)
    return weighted_quantile([r - centre for r in resid], [1.0] * n, q)


def pinball(pred: float, real: float, q: float) -> float:
    diff = real - pred
    return max(q * diff, (q - 1) * diff)


def weighted_quantile(values: Sequence[float], weights: Sequence[float], q: float) -> float:
    pairs = sorted(zip(values, weights, strict=True))
    total = sum(w for _, w in pairs)
    acc = 0.0
    for v, w in pairs:
        acc += w
        if acc >= q * total:
            return v
    return pairs[-1][0]


def wilcoxon(diffs: Sequence[float]) -> float:
    """Two-sided Wilcoxon signed-rank p: exact up to 20 non-zero differences, normal approximation
    with tie correction above (the exact enumeration is 2**n)."""
    from argus.eval.copilot_rivals import wilcoxon_exact

    d = [x for x in diffs if x != 0]
    n = len(d)
    if n <= 20:
        return wilcoxon_exact(d)
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks = [0.0] * n
    ties: list[int] = []
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        ties.append(j - i + 1)
        i = j + 1
    w = sum(r for r, x in zip(ranks, d, strict=True) if x > 0)
    mean = n * (n + 1) / 4
    var = n * (n + 1) * (2 * n + 1) / 24 - sum(t ** 3 - t for t in ties) / 48
    z = (abs(w - mean) - 0.5) / math.sqrt(var)
    return math.erfc(max(z, 0.0) / math.sqrt(2))


def run_skfolio(stamps: Sequence[date], rets: Mapping[str, Sequence[float]],
                picks: Sequence[int]) -> dict[str, Any]:  # pragma: no cover - rival interpreter
    python = os.environ.get("ARGUS_SKFOLIO_PYTHON")
    if not python:
        raise RuntimeError("set ARGUS_SKFOLIO_PYTHON to an interpreter with skfolio==1.3.1")
    names = [*NAMES, BENCHMARK]
    payload = {"n_samples": N_SAMPLES, "events": [
        {"id": stamps[i].isoformat(), "names": names, "benchmark": BENCHMARK,
         "shock": rets[BENCHMARK][i], "seed": SEED + k,
         "returns": [[rets[n][t] for n in names] for t in history(rets, i)]}
        for k, i in enumerate(picks)]}
    done = subprocess.run([python, str(RUNNER)], input=json.dumps(payload), capture_output=True,
                          text=True, check=True)
    result: dict[str, Any] = json.loads(done.stdout)
    return result


def _per_day(rows: Sequence[dict[str, Any]], arms: Sequence[str]) -> dict[str, dict[str, float]]:
    """Mean absolute error per day and arm, over the days on which every arm in ``arms`` answered.

    A day one system could not answer is dropped from *that pair's* comparison only, and the
    refusal is counted separately. Dropping it from every comparison would remove the most extreme
    days from ARGUS's record whenever a rival fails on them."""
    per_day: dict[str, dict[str, list[float]]] = {}
    for r in rows:
        if any(r["pred"].get(a) is None for a in arms):
            continue
        slot = per_day.setdefault(r["day"], {a: [] for a in arms})
        for a in arms:
            slot[a].append(abs(r["pred"][a] - r["real"]) * 100)
    return {d: {a: statistics.fmean(v) for a, v in s.items()} for d, s in per_day.items()}


def _per_day_each(rows: Sequence[dict[str, Any]], arms: Sequence[str]
                  ) -> dict[str, dict[str, float]]:
    """Each arm's mean absolute error per day over the books it answered; an arm that answered no
    book on a day is absent from that day."""
    out: dict[str, dict[str, float]] = {}
    for a in arms:
        for day, m in _per_day(rows, [a]).items():
            out.setdefault(day, {})[a] = m[a]
    return out


def score(rows: Sequence[dict[str, Any]], arms: Sequence[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for label, threshold in (("down_1pct", DOWN_DAY), ("down_2pct", SEVERE_DAY)):
        subset = [r for r in rows if r["shock"] <= threshold]
        days = sorted({r["day"] for r in subset})
        block: dict[str, Any] = {"days": len(days), "book_days": len(subset),
                                 "mean_abs_error_pp": {}, "days_answered": {},
                                 "comparisons": {}}
        for a in arms:
            means = _per_day(subset, [a])
            block["days_answered"][a] = len(means)
            if means:
                block["mean_abs_error_pp"][a] = round(
                    statistics.fmean(m[a] for m in means.values()), 4)
        for mine, theirs in itertools.permutations(arms, 2):
            if mine not in ("argus_beta", "argus_blended") and not (
                    mine == "daily_beta" and theirs.startswith("skfolio")):
                continue
            means = _per_day(subset, [mine, theirs])
            diffs = [m[mine] - m[theirs] for _, m in sorted(means.items())]
            if diffs:
                block["comparisons"][f"{mine} vs {theirs}"] = {
                    "mean_difference_pp": round(statistics.fmean(diffs), 4),
                    "days_better": sum(x < 0 for x in diffs), "days": len(diffs),
                    "wilcoxon_p": round(wilcoxon(diffs), 4)}
        pin = [r for r in subset if r.get("q10")]
        if pin:
            block["pinball_q10_pp"] = {
                arm: round(statistics.fmean(pinball(r["q10"][arm], r["real"], QUANTILE) * 100
                                            for r in pin), 4)
                for arm in ("skfolio_vine", "skfolio_ep", "argus_band")}
            block["pinball_q10_pp"]["argus_beta_as_point"] = round(statistics.fmean(
                pinball(r["pred"]["argus_beta"], r["real"], QUANTILE) * 100 for r in pin), 4)
            block["q10_coverage"] = {
                arm: round(statistics.fmean(r["real"] < r["q10"][arm] for r in pin), 4)
                for arm in ("skfolio_vine", "skfolio_ep", "argus_band")}
            by_day: dict[str, list[float]] = {}
            for r in pin:
                by_day.setdefault(r["day"], []).append(
                    pinball(r["q10"]["argus_band"], r["real"], QUANTILE)
                    - pinball(r["q10"]["skfolio_vine"], r["real"], QUANTILE))
            diffs = [statistics.fmean(v) for _, v in sorted(by_day.items())]
            block["pinball_argus_band_vs_skfolio_vine"] = {
                "days_better": sum(x < 0 for x in diffs), "days": len(diffs),
                "wilcoxon_p": round(wilcoxon(diffs), 4), "not_preregistered": True}
        out[label] = block
    return out


def verdict(scored: dict[str, Any]) -> str:
    block = scored["down_1pct"]
    comp = block["comparisons"].get("argus_beta vs skfolio_vine")
    err = block["mean_abs_error_pp"]
    if comp is None:
        return "NOT SCORED: skfolio's vine produced no usable prediction"
    lead = (f"mean absolute error {err['argus_beta']} pp (ARGUS) vs {err['skfolio_vine']} pp "
            f"(skfolio vine) over {block['days']} days with QQQUSDT down 1% or more, ARGUS better "
            f"on {comp['days_better']}/{comp['days']} days, Wilcoxon p={comp['wilcoxon_p']}")
    if comp["wilcoxon_p"] < 0.05:
        who = "ARGUS" if comp["mean_difference_pp"] < 0 else "SKFOLIO"
        return f"{who} WINS the primary: {lead}"
    return f"NO SIGNIFICANT DIFFERENCE on the primary: {lead}"


def main() -> int:  # pragma: no cover - CLI
    from argus.truth.clocks import DualClock

    clock = DualClock()
    bitget, snapshot = load_bitget()
    stamps, rets = daily_returns(bitget)
    picks = events(stamps, rets)
    rival = run_skfolio(stamps, rets, picks)
    names = [*NAMES, BENCHMARK]
    rng = random.Random(SEED)
    book_list = [portfolio.rebalance(b, add, size) for b, add, size in books(rng)][:BOOKS]
    rows: list[dict[str, Any]] = []
    open_cache: dict[datetime, bool] = {}

    def is_open(t: datetime) -> bool:
        if t not in open_cache:
            open_cache[t] = clock.phase(t).has_price_discovery
        return open_cache[t]

    for k, i in enumerate(picks):
        day = stamps[i]
        start = datetime.combine(day, time(), tzinfo=UTC)
        raw = {n: portfolio.returns([(t, c) for t, c in bitget[n]
                                     if start - timedelta(days=LOOKBACK_DAYS) <= t < start])
               for n in names}
        hourly_stamps, cols = portfolio.align(raw)
        open_rows = [j for j, t in enumerate(hourly_stamps) if is_open(t)]
        open_cols = {n: [v[j] for j in open_rows] for n, v in cols.items()}
        shock = rets[BENCHMARK][i]
        shocks = (portfolio.Shock("realised", shock * 100),)
        traded = history(rets, i)
        daily_hist = {n: [rets[n][t] for t in traded] for n in names}
        record = rival["events"][day.isoformat()]
        samples = record.get("vine_samples")
        ep_w = record.get("ep_weights")
        for b, weights in enumerate(book_list):
            def book_move(per_name: Mapping[str, float], w: Mapping[str, float] = weights
                          ) -> float:
                return sum(wt * per_name[n] for n, wt in w.items())

            def stress(columns: Mapping[str, Sequence[float]],
                       w: Mapping[str, float] = weights,
                       shock_set: tuple[portfolio.Shock, ...] = shocks) -> float | None:
                got = portfolio.stress_by_beta(weights=w, columns=columns,
                                               benchmark=columns[BENCHMARK], shocks=shock_set)[0]
                return None if got.portfolio_move_pct is None else got.portfolio_move_pct / 100

            daily_b = {n: portfolio.beta(daily_hist[n], daily_hist[BENCHMARK]) or 0.0
                       for n in weights}
            pred: dict[str, float | None] = {
                "argus_beta": stress(open_cols),
                "argus_blended": stress(cols),
                "daily_beta": book_move({n: daily_b[n] * shock for n in weights}),
                "skfolio_vine": None, "skfolio_ep": None,
            }
            q10: dict[str, float] = {}
            if samples:
                moves = [sum(weights[n] * s[names.index(n)] for n in weights) for s in samples]
                pred["skfolio_vine"] = statistics.fmean(moves)
                q10["skfolio_vine"] = weighted_quantile(moves, [1.0] * len(moves), QUANTILE)
            if ep_w:
                hist = [sum(weights[n] * daily_hist[n][t] for n in weights)
                        for t in range(len(traded))]
                pred["skfolio_ep"] = sum(p * m for p, m in zip(ep_w, hist, strict=True))
                q10["skfolio_ep"] = weighted_quantile(hist, ep_w, QUANTILE)
            point = pred["argus_beta"]
            if point is not None:
                q10["argus_band"] = point + residual_quantile(weights, daily_hist, daily_b)
            rows.append({"day": day.isoformat(), "event": k, "book": b, "shock": shock,
                         "real": book_move({n: rets[n][i] for n in weights}),
                         "pred": pred, "q10": q10 if len(q10) == 3 else None})
    arms = PREREG["arms"]
    scored = score(rows, arms)
    shock_by_day = {r["day"]: r["shock"] for r in rows}
    report = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "prereg": PREREG, "prereg_sha256": prereg_hash(),
        "rival": {"name": "skfolio", "version": rival.get("skfolio_version"),
                  "licence": "BSD-3-Clause", "engines": ["VineCopula", "EntropyPooling"],
                  "errors": {d: {k: v for k, v in e.items() if k.endswith("error")}
                             for d, e in rival["events"].items()
                             if any(k.endswith("error") for k in e)}},
        "bitget_snapshot": snapshot,
        "event_days": [stamps[i].isoformat() for i in picks],
        "stale_days_removed": sum(stale(rets, t) for t in range(len(stamps))),
        "prereg_revisions": [
            "2026-09-24: stale_rule added after the first run exposed unchanged weekend closes "
            "(see `stale`); arms, metric, primary and test unchanged"],
        "ep_solvers": {d: e.get("ep_solver") for d, e in rival["events"].items()},
        **scored,
        "verdict": verdict(scored),
        "limitations": [
            "the shock is given, not forecast: this scores the stress sentence, not a prediction "
            "of the next fall",
            "one eleven-month period of Bitget stock perpetuals; down days are few and clustered",
            "ARGUS states a point, so its tail is not scored; skfolio's quantile is",
        ],
        "rows_sample": rows[:BOOKS],
        # One row per event day and arm (added 2026-09-26): the unit the primary Wilcoxon test is
        # over, so the groupwise audit can break the result down by day and by shock size.
        "per_day": [{"day": day, "shock": shock_by_day[day],
                     "mae_pp": {a: round(m[a], 6) for a in m}}
                    for day, m in sorted(_per_day_each(rows, arms).items())],
    }
    REPORT_PATH.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(report["verdict"])
    print(json.dumps(scored, indent=1)[:3500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
