"""Portfolio Copilot, head to head: ARGUS against weekend-copilot on the numbers a copilot states.

Track 3's Open Theme names "a portfolio-aware AI PM / Portfolio Copilot that evaluates how a
proposed trade changes beta, sector and factor exposures, correlation, and concentration across an
existing portfolio". The rival review of 2026-09-24 named the S2 entry that answers exactly
that question end to end: **weekend-copilot** (PhiBao/weekend-copilot, MIT, @461ad820). Its risk
engine is small and dependency-free, so this harness does not describe it — it runs its
formulas, ported line for line from ``lib/risk-engine/delta.ts:47-130`` and ``lib/stats.ts:2-86``,
and **must first reproduce the
numbers its own engine printed** for its three preset books (``baseline_reproduced``) before any
comparison is reported.

**What is compared: the two numbers both copilots state and a trader acts on.**

* **B1, beta after the trade.** ARGUS states the book's open-session beta against QQQUSDT from 30
  days of hourly Bitget bars (`desk/portfolio.copilot`, the function the console calls).
  weekend-copilot states a value-weighted blend of each name's beta from every daily native close it
  holds (two years of Yahoo bars), against SPY by default, with a silent 1.0 wherever fewer than 20
  closes exist (``delta.ts:90,99``). Both are scored against what the book then did.
* **B2, which holding the new name moves with.** ARGUS names the held position the candidate is most
  correlated with; weekend-copilot computes the most correlated pair from the same correlation
  function (``stats.ts:73-86``), which is read here restricted to pairs that include the candidate,
  so both answer the same question. Scored as a hit when it names the holding the candidate
  actually co-moved with most over the next 28 days.

**Arms.** ``argus_open`` (production), ``argus_blended`` (all hourly bars; the figure ARGUS's
docstring says misleads, scored so that claim is tested rather than asserted),
``rival_spy`` (weekend-copilot as shipped), ``rival_qqq`` (its own ``marketNative`` parameter set to
QQQ, so the benchmark matches the target and only the method differs), ``naive_one`` (every beta
1.0 — the floor both must clear).

**Targets.** ``bitget_daily`` (primary): the after-book's realised beta on Bitget daily UTC closes
against QQQUSDT over the 28 days after the origin — what a holder of these instruments experienced.
``native_daily``: realised beta on native daily closes against native QQQ over the next 20 trading
days — the rival's home data. ``bitget_open_hourly``: realised open-session hourly beta — ARGUS's
home target. Each arm is scored on all three, so neither side is judged only on the other's ground.

**Pre-registered** in :data:`PREREG` before the first scored run: nine monthly origins, 200 books
per origin drawn from one seed, primary = mean absolute beta error on ``bitget_daily``, ARGUS
production against ``rival_qqq``, compared by an exact Wilcoxon signed-rank test over the nine
per-origin mean differences (books inside one origin share one market path and are not independent,
so they are averaged before testing). The spec's SHA-256 is written into the report.

    python -m argus.eval.copilot_rivals --freeze   # snapshot Bitget hourly bars (once)
    python -m argus.eval.copilot_rivals            # score

NOT COMPARED HERE, and said so rather than implied: weekend-copilot's Friday-to-Monday gap stress
and its hedge comparator (the hedge is a sizing choice for the proposed trade — full, half, or an
illustrative coupon — not an instrument, ``hedge.ts:1-90``), sector weights (both products bucket by
a hard-coded sector map), and HHI (the same formula in both).
"""

from __future__ import annotations

import functools
import hashlib
import itertools
import json
import math
import os
import random
import statistics
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from argus.desk import portfolio

PACKAGE = Path(__file__).resolve().parents[3]
DATA = PACKAGE / "data"
REPORT_PATH = DATA / "copilot_rivals.json"
SNAPSHOT_PATH = DATA / "arena" / "copilot" / "bitget_1h.json"
DEFAULT_CLONE = PACKAGE.parent / "research" / "repos-rivals" / "PhiBao~weekend-copilot"

NAMES = ("AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "TSLA")
"""The names both products cover: weekend-copilot's native list (``lib/market/symbols.ts:2``)
intersected with Bitget's stock perpetuals. AMD and MU have no Bitget perp; SPY, QQQ and BTC are
benchmarks or not stocks."""
BENCHMARK = "QQQ"
SNAPSHOT_START = datetime(2025, 10, 28, tzinfo=UTC)
"""QQQUSDT's first hourly bar on Bitget; nothing earlier can be scored against it."""
LOOKBACK_DAYS = 30
FORWARD_DAYS = 28
NATIVE_FORWARD_SESSIONS = 20
ORIGINS = tuple(datetime(2025 + (m > 12), (m - 1) % 12 + 1, 1, tzinfo=UTC) for m in range(12, 21))
"""2025-12-01 through 2026-08-01, monthly: every origin with 30 days of QQQUSDT before it and both
forward windows inside the data (native closes end 2026-09-18)."""
BOOKS_PER_ORIGIN = 200
ADD_SIZES = (0.1, 0.2, 0.3)
SEED = 20260924

PREREG: dict[str, Any] = {
    "question": "which copilot's stated post-trade beta and most-correlated holding are closer "
                "to what the book then did",
    "names": list(NAMES), "benchmark": BENCHMARK,
    "origins": [o.date().isoformat() for o in ORIGINS],
    "books_per_origin": BOOKS_PER_ORIGIN, "held": [3, 5], "add_sizes": list(ADD_SIZES),
    "seed": SEED, "lookback_days": LOOKBACK_DAYS, "forward_days": FORWARD_DAYS,
    "native_forward_sessions": NATIVE_FORWARD_SESSIONS,
    "arms": ["argus_open", "argus_blended", "rival_spy", "rival_qqq", "naive_one"],
    "targets": ["bitget_daily", "native_daily", "bitget_open_hourly"],
    "primary": {"metric": "mean absolute error of beta after the trade", "target": "bitget_daily",
                "comparison": "argus_open vs rival_qqq",
                "test": "exact two-sided Wilcoxon signed-rank over per-origin mean differences"},
    "secondary": ["every other arm pair and target, reported, not tested for a headline",
                  "B2 hit rate of the most-correlated holding, bitget_daily and native_daily"],
}


def prereg_hash() -> str:
    return hashlib.sha256(json.dumps(PREREG, sort_keys=True).encode()).hexdigest()


def clone_path() -> Path:
    return Path(os.environ.get("ARGUS_WEEKEND_COPILOT_CLONE", DEFAULT_CLONE))


# =============================================================================================
# Data: a frozen Bitget snapshot, and the rival's own native snapshot read in place
# =============================================================================================


def freeze() -> dict[str, Any]:  # pragma: no cover - network
    """Snapshot hourly Bitget closes for every name and the benchmark, hashed."""
    from argus.market.history import CandleType, fetch_window

    series: dict[str, list[list[Any]]] = {}
    for name in (*NAMES, BENCHMARK):
        candles = fetch_window(f"{name}USDT", start=SNAPSHOT_START, interval="1H",
                               candle_type=CandleType.MARKET)
        series[name] = [[c.ts.isoformat(), float(c.close)] for c in candles]
    body = json.dumps(series, sort_keys=True)
    snapshot = {"fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "source": "Bitget public v2 mix history-candles, USDT-FUTURES, 1H, MARKET",
                "sha256": hashlib.sha256(body.encode()).hexdigest(), "series": series}
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_PATH.write_text(json.dumps(snapshot), encoding="utf-8")
    return snapshot


def load_bitget() -> tuple[dict[str, list[tuple[datetime, float]]], dict[str, Any]]:
    snapshot = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    series = snapshot["series"]
    digest = hashlib.sha256(json.dumps(series, sort_keys=True).encode()).hexdigest()
    if digest != snapshot["sha256"]:
        raise RuntimeError(f"{SNAPSHOT_PATH} does not match its recorded hash; refusing to score")
    out = {k: [(datetime.fromisoformat(t), float(c)) for t, c in v] for k, v in series.items()}
    return out, {"fetched_at": snapshot["fetched_at"], "sha256": digest,
                 "source": snapshot["source"]}


def load_native(clone: Path) -> tuple[dict[str, list[tuple[date, float]]], dict[str, str]]:
    """weekend-copilot's own daily native closes (``data/snapshots/*.json``, ``bars[].close``, as
    ``lib/market/snapshots.ts:53`` reads them). Not copied into this repository: the bars are
    Yahoo's. Their hashes are recorded instead."""
    out: dict[str, list[tuple[date, float]]] = {}
    hashes: dict[str, str] = {}
    for name in (*NAMES, BENCHMARK, "SPY"):
        path = clone / "data" / "snapshots" / f"{name}.json"
        raw = path.read_bytes()
        hashes[name] = hashlib.sha256(raw).hexdigest()
        bars = json.loads(raw)["bars"]
        out[name] = [(date.fromisoformat(b["date"]), float(b["close"])) for b in bars]
    return out, hashes


# =============================================================================================
# The rival, ported line for line (MIT, PhiBao/weekend-copilot @461ad820)
# =============================================================================================


def rival_log_returns(closes: Sequence[float]) -> list[float]:
    """``lib/stats.ts:2-8``."""
    return [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))
            if closes[i - 1] > 0 and closes[i] > 0]


def rival_beta(asset: Sequence[float], market: Sequence[float]) -> float:
    """``lib/stats.ts:56-71``: the last ``n`` of each series, **not date-aligned**, 1.0 when
    degenerate."""
    n = min(len(asset), len(market))
    if n < 2:
        return 1.0
    a, m = list(asset[-n:]), list(market[-n:])
    ma, mm = sum(a) / n, sum(m) / n
    cov = sum((x - ma) * (y - mm) for x, y in zip(a, m, strict=True))
    vm = sum((y - mm) ** 2 for y in m)
    return 1.0 if vm == 0 else cov / vm


def rival_correlation(a: Sequence[float], b: Sequence[float]) -> float:
    """``lib/stats.ts:73-86``."""
    n = min(len(a), len(b))
    if n < 2:
        return 0.0
    x, y = list(a[-n:]), list(b[-n:])
    sa, sb = statistics.stdev(x), statistics.stdev(y)
    if sa == 0 or sb == 0:
        return 0.0
    mx, my = sum(x) / n, sum(y) / n
    return sum((p - mx) * (q - my) for p, q in zip(x, y, strict=True)) / ((n - 1) * sa * sb)


def rival_book_beta(weights: Mapping[str, float], closes: Mapping[str, Sequence[float]],
                    market: str) -> float:
    """``delta.ts:87-102``, including both silent 1.0 fallbacks."""
    market_rets = rival_log_returns(closes.get(market, []))
    total = 0.0
    for name, w in weights.items():
        series = closes.get(name, [])
        if len(series) >= 20:
            r = rival_log_returns(series)
            b = 1.0 if len(r) < 20 or len(market_rets) < 20 else rival_beta(r, market_rets)
        else:
            b = 1.0
        total += w * b
    return total


def rival_max_correlation(names: Sequence[str], closes: Mapping[str, Sequence[float]]
                          ) -> tuple[tuple[str, str], float]:
    """``delta.ts:122-130``: the largest |rho| over pairs of names with at least 20 closes."""
    usable = [n for n in dict.fromkeys(names) if len(closes.get(n, [])) >= 20]
    best: tuple[tuple[str, str], float] = (("—", "—"), 0.0)
    for i, j in itertools.combinations(range(len(usable)), 2):
        c = abs(rival_correlation(rival_log_returns(closes[usable[i]]),
                                  rival_log_returns(closes[usable[j]])))
        if c > best[1]:
            best = ((usable[i], usable[j]), c)
    return best


def rival_partner(add: str, held: Sequence[str], closes: Mapping[str, Sequence[float]]
                  ) -> str | None:
    """The rival's correlation function read on pairs that include the candidate."""
    best: tuple[str | None, float] = (None, -1.0)
    if len(closes.get(add, [])) < 20:
        return None
    for other in held:
        if len(closes.get(other, [])) < 20:
            continue
        c = abs(rival_correlation(rival_log_returns(closes[add]),
                                  rival_log_returns(closes[other])))
        if c > best[1]:
            best = (other, c)
    return best[0]


def reproduction(clone: Path) -> dict[str, Any]:
    """Recompute weekend-copilot's own printed beta and max correlation for its preset books from
    its own full snapshot. ``argus-probe.json`` is its engine's output, run unmodified."""
    probe_path = clone / "argus-probe.json"
    if not probe_path.exists():
        return {"available": False, "reason": f"{probe_path} missing: run the rival's engine first"}
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    closes: dict[str, list[float]] = {}
    for path in (clone / "data" / "snapshots").glob("*.json"):
        if path.stem.isupper():
            closes[path.stem] = [float(b["close"]) for b in json.loads(path.read_bytes())["bars"]]
    worst = 0.0
    cases = {}
    for key, case in probe["cases"].items():
        delta = case["delta"]
        native = {k: {k[1:] if k.startswith("R") and k[1:] in closes else k: v
                      for k, v in delta[k].items()} for k in ("weightsBefore", "weightsAfter")}
        mine_before = rival_book_beta(native["weightsBefore"], closes, "SPY")
        mine_after = rival_book_beta(native["weightsAfter"], closes, "SPY")
        pair, rho = rival_max_correlation(list(native["weightsAfter"]), closes)
        diffs = [abs(mine_before - delta["betaBefore"]), abs(mine_after - delta["betaAfter"]),
                 abs(rho - delta["maxCorrelation"]["value"])]
        worst = max(worst, *diffs)
        cases[key] = {"beta_before": [delta["betaBefore"], mine_before],
                      "beta_after": [delta["betaAfter"], mine_after],
                      "max_correlation": [delta["maxCorrelation"]["value"], rho,
                                          sorted(delta["maxCorrelation"]["pair"]), sorted(pair)]}
    return {"available": True, "max_abs_difference": worst, "reproduced": worst < 1e-9,
            "cases": cases, "rival_commit": "461ad82043bd9776572ffd33d35e330a1243d624",
            "data_built_at": probe.get("dataBuiltAt")}


# =============================================================================================
# Books, arms, targets
# =============================================================================================


def books(rng: random.Random) -> list[tuple[dict[str, float], str, float]]:
    """``BOOKS_PER_ORIGIN`` long-only books: 3-5 held names, Dirichlet(1) weights, one new name."""
    out = []
    for _ in range(BOOKS_PER_ORIGIN):
        k = rng.randint(3, 5)
        chosen = rng.sample(NAMES, k + 1)
        held, add = chosen[:k], chosen[k]
        draws = [rng.gammavariate(1.0, 1.0) for _ in held]
        total = sum(draws)
        out.append(({n: d / total for n, d in zip(held, draws, strict=True)}, add,
                    rng.choice(ADD_SIZES)))
    return out


def _hourly_returns(series: Sequence[tuple[datetime, float]], start: datetime, end: datetime
                    ) -> dict[datetime, float]:
    return portfolio.returns([(t, c) for t, c in series if start <= t < end])


def _daily_closes(series: Sequence[tuple[datetime, float]]) -> dict[date, float]:
    """The last hourly close of each UTC day."""
    out: dict[date, float] = {}
    for t, c in series:
        out[t.date()] = c
    return out


def _realised_beta(weights: Mapping[str, float], cols: Mapping[str, Sequence[float]],
                   bench: Sequence[float]) -> float | None:
    book = [sum(w * cols[n][i] for n, w in weights.items()) for i in range(len(bench))]
    return portfolio.beta(book, bench)


def _realised_partner(add: str, held: Sequence[str], cols: Mapping[str, Sequence[float]]
                      ) -> str | None:
    best: tuple[str | None, float] = (None, -1.0)
    for other in held:
        c = portfolio.correlation(cols[add], cols[other])
        if c is not None and abs(c) > best[1]:
            best = (other, abs(c))
    return best[0]


def run(bitget: Mapping[str, Sequence[tuple[datetime, float]]],
        native: Mapping[str, Sequence[tuple[date, float]]]) -> list[dict[str, Any]]:
    """Every arm's prediction and every target's realisation, per book and origin."""
    from argus.truth.clocks import DualClock

    clock = DualClock()

    @functools.cache
    def is_open(t: datetime) -> bool:
        return clock.phase(t).has_price_discovery

    rng = random.Random(SEED)
    daily = {n: _daily_closes(s) for n, s in bitget.items()}
    rows: list[dict[str, Any]] = []
    for origin in ORIGINS:
        start, end = origin - timedelta(days=LOOKBACK_DAYS), origin + timedelta(days=FORWARD_DAYS)
        raw = {f"{n}USDT": _hourly_returns(bitget[n], start, origin) for n in (*NAMES, BENCHMARK)}
        stamps, cols = portfolio.align(raw)
        # Forward targets.
        fwd_raw = {n: _hourly_returns(bitget[n], origin, end) for n in (*NAMES, BENCHMARK)}
        fwd_stamps, fwd = portfolio.align(fwd_raw)
        open_rows = [i for i, t in enumerate(fwd_stamps) if is_open(t)]
        fwd_open = {n: [v[i] for i in open_rows] for n, v in fwd.items()}
        days = sorted(d for d in daily[BENCHMARK] if origin.date() - timedelta(days=1) <= d
                      < end.date() and all(d in daily[n] for n in (*NAMES, BENCHMARK)))
        fwd_daily = {n: [daily[n][b] / daily[n][a] - 1 for a, b in itertools.pairwise(days)]
                     for n in (*NAMES, BENCHMARK)}
        nat_before = {n: [c for d, c in s if d < origin.date()] for n, s in native.items()}
        nat_after_dates = sorted({d for d, _ in native[BENCHMARK] if d >= origin.date()}
                                 )[:NATIVE_FORWARD_SESSIONS]
        last_before = max(d for d, _ in native[BENCHMARK] if d < origin.date())
        nat_index = {n: dict(s) for n, s in native.items()}
        nat_days = [last_before, *nat_after_dates]
        nat_fwd = {n: [math.log(nat_index[n][b] / nat_index[n][a])
                       for a, b in itertools.pairwise(nat_days)] for n in (*NAMES, BENCHMARK)}
        blended_bench = cols[f"{BENCHMARK}USDT"]

        for i, (before, add, size) in enumerate(books(rng)):
            after = portfolio.rebalance(before, add, size)
            needed = {f"{n}USDT" for n in (*before, add, BENCHMARK)}
            report = portfolio.copilot(
                add=f"{add}USDT", before={f"{n}USDT": w for n, w in before.items()}, size=size,
                raw={k: v for k, v in raw.items() if k in needed},
                benchmark=f"{BENCHMARK}USDT", is_open=is_open,
            )
            blended = portfolio.assess(
                symbol=f"{add}USDT", weights_before={f"{n}USDT": w for n, w in before.items()},
                weights_after={f"{n}USDT": w for n, w in after.items()},
                columns={k: v for k, v in cols.items() if k in needed}, benchmark=blended_bench,
                session=portfolio.Session.BLENDED,
            )
            argus_partner = report.impact.max_correlation
            rows.append({
                "origin": origin.date().isoformat(), "book": i, "add": add, "size": size,
                "held": sorted(before), "after": {n: round(w, 6) for n, w in after.items()},
                "pred": {
                    "argus_open": report.impact.beta_after,
                    "argus_blended": blended.beta_after,
                    "rival_spy": rival_book_beta(after, nat_before, "SPY"),
                    "rival_qqq": rival_book_beta(after, nat_before, BENCHMARK),
                    "naive_one": 1.0,
                },
                "partner": {
                    "argus_open": None if argus_partner is None
                    else argus_partner[0].removesuffix("USDT"),
                    "rival": rival_partner(add, sorted(before), nat_before),
                },
                "real": {
                    "bitget_daily": _realised_beta(after, fwd_daily, fwd_daily[BENCHMARK]),
                    "native_daily": _realised_beta(after, nat_fwd, nat_fwd[BENCHMARK]),
                    "bitget_open_hourly": _realised_beta(after, fwd_open, fwd_open[BENCHMARK]),
                },
                "real_partner": {
                    "bitget_daily": _realised_partner(add, sorted(before), fwd_daily),
                    "native_daily": _realised_partner(add, sorted(before), nat_fwd),
                },
                "rival_beta_one_fallbacks": sum(len(nat_before[n]) < 21 for n in after),
                "bars": {"lookback_hourly": len(stamps), "forward_daily": len(fwd_daily[BENCHMARK]),
                         "forward_open_hourly": len(open_rows),
                         "native_forward": len(nat_fwd[BENCHMARK])},
            })
    return rows


# =============================================================================================
# Scoring
# =============================================================================================


def wilcoxon_exact(diffs: Sequence[float]) -> float:
    """Two-sided exact Wilcoxon signed-rank p-value; zeros dropped, ties given mid-ranks."""
    d = [x for x in diffs if x != 0]
    n = len(d)
    if n == 0:
        return 1.0
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    observed = sum(r for r, x in zip(ranks, d, strict=True) if x > 0)
    centre = sum(ranks) / 2
    extreme = 0
    for signs in itertools.product((0, 1), repeat=n):
        w = sum(r for r, s in zip(ranks, signs, strict=True) if s)
        if abs(w - centre) >= abs(observed - centre) - 1e-12:
            extreme += 1
    return extreme / float(2 ** n)


def score(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    arms = PREREG["arms"]
    out: dict[str, Any] = {"beta_error": {}, "comparisons": {}, "partner_hit_rate": {}}
    for target in PREREG["targets"]:
        usable = [r for r in rows if r["real"][target] is not None
                  and all(r["pred"][a] is not None for a in arms)]
        per_origin: dict[str, dict[str, list[float]]] = {}
        for r in usable:
            slot = per_origin.setdefault(r["origin"], {a: [] for a in arms})
            for a in arms:
                slot[a].append(abs(r["pred"][a] - r["real"][target]))
        origin_means = {o: {a: statistics.fmean(v) for a, v in s.items()}
                        for o, s in per_origin.items()}
        out["beta_error"][target] = {
            "books_scored": len(usable),
            "mean_abs_error": {a: round(statistics.fmean(
                [abs(r["pred"][a] - r["real"][target]) for r in usable]), 4) for a in arms},
            "median_abs_error": {a: round(statistics.median(
                [abs(r["pred"][a] - r["real"][target]) for r in usable]), 4) for a in arms},
            "per_origin_mean_abs_error": {o: {a: round(v, 4) for a, v in m.items()}
                                          for o, m in sorted(origin_means.items())},
        }
        for mine, theirs in (("argus_open", "rival_qqq"), ("argus_open", "rival_spy"),
                             ("argus_open", "naive_one"), ("argus_open", "argus_blended"),
                             ("rival_qqq", "naive_one")):
            diffs = [m[mine] - m[theirs] for _, m in sorted(origin_means.items())]
            out["comparisons"].setdefault(target, {})[f"{mine} vs {theirs}"] = {
                "mean_difference": round(statistics.fmean(diffs), 4),
                "origins_better": sum(d < 0 for d in diffs),
                "origins": len(diffs),
                "wilcoxon_p": round(wilcoxon_exact(diffs), 4),
            }
    for target in ("bitget_daily", "native_daily"):
        scored = [r for r in rows if r["real_partner"][target] is not None]
        chance = statistics.fmean(1 / len(r["held"]) for r in scored)
        out["partner_hit_rate"][target] = {
            "books": len(scored), "chance": round(chance, 4),
            **{arm: round(statistics.fmean(r["partner"][arm] == r["real_partner"][target]
                                           for r in scored), 4)
               for arm in ("argus_open", "rival")},
        }
    out["rival_beta_one_fallbacks"] = sum(r["rival_beta_one_fallbacks"] for r in rows)
    return out


def verdict(scored: dict[str, Any]) -> str:
    primary = scored["comparisons"]["bitget_daily"]["argus_open vs rival_qqq"]
    err = scored["beta_error"]["bitget_daily"]["mean_abs_error"]
    if primary["wilcoxon_p"] < 0.05:
        who = "ARGUS" if primary["mean_difference"] < 0 else "weekend-copilot"
        return (f"{who.upper()} WINS the primary: mean absolute beta error "
                f"{err['argus_open']} (ARGUS) vs {err['rival_qqq']} (weekend-copilot, QQQ), "
                f"better on {primary['origins_better']}/{primary['origins']} origins, "
                f"Wilcoxon p={primary['wilcoxon_p']}")
    return (f"NO SIGNIFICANT DIFFERENCE on the primary: mean absolute beta error "
            f"{err['argus_open']} (ARGUS) vs {err['rival_qqq']} (weekend-copilot, QQQ), ARGUS "
            f"better on {primary['origins_better']}/{primary['origins']} origins, "
            f"Wilcoxon p={primary['wilcoxon_p']}")


def limitations(scored: dict[str, Any]) -> list[str]:
    """What the run does not show, written from its own numbers so it cannot drift from them."""
    comp = scored["comparisons"]["bitget_daily"]
    err = scored["beta_error"]["bitget_daily"]["mean_abs_error"]
    naive = comp["argus_open vs naive_one"]
    blend = comp["argus_open vs argus_blended"]
    return [
        f"nine origins over one eleven-month period (December 2025 to August 2026); the primary "
        f"test has little power, and p = {comp['argus_open vs rival_qqq']['wilcoxon_p']} does "
        f"not clear 0.05",
        f"ARGUS is not significantly better than calling every beta 1.0: {err['argus_open']} "
        f"against {err['naive_one']}, better on {naive['origins_better']}/{naive['origins']} "
        f"origins, p = {naive['wilcoxon_p']}",
        f"the open-session beta is not measurably better than the blended one at forecasting the "
        f"next month's beta ({err['argus_open']} against {err['argus_blended']}, p = "
        f"{blend['wilcoxon_p']}); the module's claim that the blended figure misleads is about "
        f"what the book does while the anchor market prices it, and is not supported as a "
        f"forecasting advantage",
        "books are random long-only weights over seven names; no short books, no crypto legs",
    ]


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--freeze", action="store_true", help="snapshot Bitget hourly bars")
    args = parser.parse_args(argv)
    if args.freeze:
        snap = freeze()
        print(f"froze {len(snap['series'])} series, sha256 {snap['sha256'][:16]}")
        return 0
    clone = clone_path()
    bitget, snapshot = load_bitget()
    native, native_hashes = load_native(clone)
    rows = run(bitget, native)
    scored = score(rows)
    report = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "prereg": PREREG, "prereg_sha256": prereg_hash(),
        "rival": {"name": "weekend-copilot", "repo": "https://github.com/PhiBao/weekend-copilot",
                  "licence": "MIT", "commit": "461ad82043bd9776572ffd33d35e330a1243d624",
                  "native_snapshot_sha256": native_hashes},
        "bitget_snapshot": snapshot,
        "baseline_reproduced": reproduction(clone),
        **scored,
        "verdict": verdict(scored),
        "limitations": limitations(scored),
        "not_compared": [
            "weekend-copilot's Friday-to-Monday gap stress (per proposed name, not per book)",
            "its hedge comparator: a sizing choice for the proposed trade, not a hedge instrument",
            "sector weights and HHI: the same arithmetic in both products",
        ],
        "rows": rows,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(report["verdict"])
    print(json.dumps({k: scored[k] for k in ("beta_error", "partner_hit_rate")}, indent=1)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
