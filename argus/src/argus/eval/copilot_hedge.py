"""Portfolio Copilot hedge, head to head: ARGUS against Ballast (S2) on an rToken holder's night.

The sub-theme asks for "stress tests or hedge suggestions". The rival review of 2026-09-24 named
the S2 entry that owns the hedge half: **Ballast** (Ritapossible/Ballast, MIT, @5cf6759), which
hedges a spot rToken (``RTSLAUSDT``) with the same company's stock perpetual (``TSLAUSDT``) across
the hours the US market is shut. It was run unmodified (``research/hedge_study.py`` then
``research/oos.py``) and reproduced its README on every figure checked: held-out median R² 0.981
before and 0.997 after, tail cut 86.5% then 95.1%, 12 of 12 names holding out of sample. Its
per-night series — close to next open, spot and perp, as ``ballast/overnight.py`` builds them — is
frozen at ``data/arena/copilot/ballast_nights.json`` so both systems are scored on identical nights.

**The comparison, on Ballast's own protocol** (``research/oos.py:63-87``): per name, the hedge ratio
is fitted on the first 70% of nights and applied unrefitted to the last 30%.

* ``ballast_same_name`` — short the same-name perp, ratio by Ballast's own OLS
  (``ballast/stats.py:8-20``, ported) on the fit nights, residual as ``oos.py:37-48`` computes it.
* ``argus_index`` — what ARGUS's hedge answer offered a holder before 2026-09-24: its
  minimum-variance hedge on QQQUSDT, the index leg its console ranked first.
* ``argus_same_name`` — what ARGUS's console answers now: `desk/rtoken_hedge.overnight_hedge`,
  the production function, on the same nights.

**Scored:** held-out share of overnight variance removed, held-out p95 tail cut, and the gross cost
of carrying the hedge one night (|ratio| x Ballast's 12 bp taker round trip, ``ballast/costs.py``).
Paired across names with an exact Wilcoxon test.

**Stated before the numbers.** ``argus_same_name`` and ``ballast_same_name`` are the same
arithmetic — ``minimum_variance_hedge``'s ratio is minus the OLS slope Ballast fits — so the most
ARGUS can reach on this measure is parity, and the report checks that parity per name rather than
asserting it. The first run of this module (2026-09-24) was a clean loss: ARGUS had no model of the
spot rToken, only the perpetual, and could not tell a holder of ``RTSLAUSDT`` that ``TSLAUSDT``
existed as its hedge. `market/rtoken_spot.py` and `desk/rtoken_hedge.py` were built from that loss.

    python -m argus.eval.copilot_hedge
"""

from __future__ import annotations

import hashlib
import json
import os
import statistics
from collections.abc import Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from argus.desk.diversification import minimum_variance_hedge
from argus.eval.copilot_rivals import wilcoxon_exact

PACKAGE = Path(__file__).resolve().parents[3]
REPORT_PATH = PACKAGE / "data" / "copilot_hedge.json"
NIGHTS_PATH = PACKAGE / "data" / "arena" / "copilot" / "ballast_nights.json"
DEFAULT_CLONE = PACKAGE.parent / "research" / "repos-rivals" / "Ritapossible~Ballast"
FIT_SHARE = 0.7
ROUND_TRIP_BP = 12.0
INDEX = "QQQ"


def clone_path() -> Path:
    return Path(os.environ.get("ARGUS_BALLAST_CLONE", DEFAULT_CLONE))


def freeze() -> dict[str, Any]:  # pragma: no cover - reads the rival's run
    raw = (clone_path() / "argus-repro-series.json").read_bytes()
    NIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    NIGHTS_PATH.write_bytes(raw)
    return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def _p95_abs(xs: Sequence[float]) -> float:
    ordered = sorted(abs(x) for x in xs)
    rank = 0.95 * (len(ordered) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


def hedged(spot: Sequence[float], leg: Sequence[float], ratio: float) -> list[float]:
    return [s + ratio * h for s, h in zip(spot, leg, strict=True)]


def ballast_ols(x: Sequence[float], y: Sequence[float]) -> float:
    """``ballast/stats.py:8-20``: the univariate OLS slope of ``y`` on ``x``."""
    mx, my = statistics.fmean(x), statistics.fmean(y)
    sxx = sum((u - mx) ** 2 for u in x)
    return sum((u - mx) * (v - my) for u, v in zip(x, y, strict=True)) / sxx


def ballast_arm(spot: Sequence[float], perp: Sequence[float]) -> dict[str, float]:
    """Ballast's held-out protocol, ``research/oos.py:63-87``: slope frozen on the first 70%,
    residual ``y - beta * x`` on the rest, variance explained by population variances."""
    cut = int(len(spot) * FIT_SHARE)
    beta = ballast_ols(perp[:cut], spot[:cut])
    resid = [v - beta * u for u, v in zip(perp[cut:], spot[cut:], strict=True)]
    return {
        "ratio": round(-beta, 6),
        "held_out_variance_removed": round(
            1 - statistics.pvariance(resid) / statistics.pvariance(spot[cut:]), 5),
        "held_out_tail_cut": round(1 - _p95_abs(resid) / _p95_abs(spot[cut:]), 5),
        "cost_bp_per_night": round(abs(beta) * ROUND_TRIP_BP, 3),
        "fit_nights": cut, "held_out_nights": len(spot) - cut,
    }


def argus_arm(spot_symbol: str, perp_symbol: str, days: Sequence[date],
              spot: Sequence[float], perp: Sequence[float]) -> dict[str, float]:
    """The console's own function on the same nights. Its live ratio is fitted on every night; the
    held-out figures come from its own 70/30 split, which is what is compared."""
    from argus.desk.rtoken_hedge import overnight_hedge

    h = overnight_hedge(spot_symbol, perp_symbol, dict(zip(days, spot, strict=True)),
                        dict(zip(days, perp, strict=True)))
    return {
        "ratio": round(h.ratio, 6),
        "held_out_variance_removed": round(h.held_out_variance_removed, 5),
        "held_out_tail_cut": round(h.held_out_tail_cut, 5),
        "cost_bp_per_night": round(h.cost_bp_per_night, 3),
        "fit_nights": h.nights - h.held_out_nights, "held_out_nights": h.held_out_nights,
    }


def evaluate(spot: Sequence[float], leg: Sequence[float], name: str) -> dict[str, float]:
    """Fit on the first :data:`FIT_SHARE` of nights, score the rest unrefitted."""
    cut = int(len(spot) * FIT_SHARE)
    fit = minimum_variance_hedge(spot[:cut], leg[:cut], instrument=name)
    held_spot, held_leg = spot[cut:], leg[cut:]
    resid = hedged(held_spot, held_leg, fit.ratio)
    var_before = statistics.variance(held_spot)
    return {
        "ratio": round(fit.ratio, 6),
        "in_sample_variance_removed": round(fit.variance_reduction, 5),
        "held_out_variance_removed": round(1 - statistics.variance(resid) / var_before, 5),
        "held_out_tail_cut": round(1 - _p95_abs(resid) / _p95_abs(held_spot), 5),
        "cost_bp_per_night": round(abs(fit.ratio) * ROUND_TRIP_BP, 3),
        "fit_nights": cut, "held_out_nights": len(held_spot),
    }


def compare(series: dict[str, Any]) -> dict[str, Any]:
    index = {n["date"]: n["perp_overnight_log_return"] for n in series[INDEX]["nights"]}
    per_name: dict[str, Any] = {}
    for name, body in series.items():
        if name == INDEX:
            continue  # its index hedge is its same-name hedge
        nights = [n for n in body["nights"] if n["date"] in index]
        spot = [n["spot_overnight_log_return"] for n in nights]
        perp = [n["perp_overnight_log_return"] for n in nights]
        qqq = [index[n["date"]] for n in nights]
        if len(nights) < 40:
            continue
        days = [date.fromisoformat(n["date"]) for n in nights]
        per_name[name] = {
            "spot": body["spot_symbol"], "perp": body["perp_symbol"], "nights": len(nights),
            "first": nights[0]["date"], "last": nights[-1]["date"],
            "ballast_same_name": ballast_arm(spot, perp),
            "argus_index": evaluate(spot, qqq, f"{INDEX}USDT"),
            "argus_same_name": argus_arm(body["spot_symbol"], body["perp_symbol"], days, spot,
                                         perp),
        }
    diffs = [v["argus_index"]["held_out_variance_removed"]
             - v["ballast_same_name"]["held_out_variance_removed"] for v in per_name.values()]
    parity = max(abs(v["argus_same_name"]["held_out_variance_removed"]
                     - v["ballast_same_name"]["held_out_variance_removed"])
                 for v in per_name.values())
    arms = ("ballast_same_name", "argus_index", "argus_same_name")
    summary = {
        arm: {k: round(statistics.median(v[arm][k] for v in per_name.values()), 5)
              for k in ("held_out_variance_removed", "held_out_tail_cut", "cost_bp_per_night")}
        for arm in arms}
    return {
        "names": len(per_name), "per_name": per_name, "median": summary,
        "argus_same_name_vs_ballast_max_abs_difference": parity,
        "comparison": {"argus_index vs ballast_same_name": {
            "names_better": sum(d > 0 for d in diffs), "names": len(diffs),
            "median_difference": round(statistics.median(diffs), 5),
            "wilcoxon_p": round(wilcoxon_exact(diffs), 5)}},
    }


def verdict(result: dict[str, Any]) -> str:
    med = result["median"]
    comp = result["comparison"]["argus_index vs ballast_same_name"]
    parity = result["argus_same_name_vs_ballast_max_abs_difference"]
    same = med["argus_same_name"]["held_out_variance_removed"]
    rival = med["ballast_same_name"]["held_out_variance_removed"]
    head = "TIED" if parity < 1e-3 else "ARGUS AHEAD" if same > rival else "BALLAST AHEAD"
    return (f"{head}: held-out overnight variance removed, median over {result['names']} names "
            f"— ARGUS's console now {same:.1%}, Ballast's same-name perp {rival:.1%} (largest "
            f"per-name gap {parity:.1e}; the same arithmetic). Before the spot rToken was "
            f"modelled ARGUS offered its index hedge, which removes "
            f"{med['argus_index']['held_out_variance_removed']:.1%} and lost on "
            f"{comp['names'] - comp['names_better']}/{comp['names']} names "
            f"(Wilcoxon p={comp['wilcoxon_p']})")


def main() -> int:  # pragma: no cover - CLI
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--freeze", action="store_true")
    if parser.parse_args().freeze:
        print(freeze())
        return 0
    raw = NIGHTS_PATH.read_bytes()
    data = json.loads(raw)
    result = compare(data["series"])
    report = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "rival": {"name": "Ballast", "repo": "https://github.com/Ritapossible/Ballast",
                  "licence": "MIT", "commit": data.get("commit"),
                  "baseline_reproduced": "README figures reproduced by running hedge_study.py "
                                         "and oos.py unmodified: held-out R2 0.981 -> 0.997, "
                                         "tail cut 86.5% -> 95.1%, 12/12 names"},
        "nights_sha256": hashlib.sha256(raw).hexdigest(),
        "protocol": {"split": f"first {FIT_SHARE:.0%} of each name's nights fit, rest held out",
                     "source": "research/oos.py:63-87"},
        **result,
        "verdict": verdict(result),
        "limitations": [
            "parity is the ceiling on this measure: both sides fit the same slope, so ARGUS's "
            "claim here is only that its console now offers the hedge, tested, with the index "
            "alternative and the weekend split beside it",
            "nights only (close to next open), Ballast's own window; intraday hedging not scored",
            "gross taker cost; Ballast's funding credit (0.7 bp a night on average) not applied",
        ],
    }
    REPORT_PATH.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(report["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
