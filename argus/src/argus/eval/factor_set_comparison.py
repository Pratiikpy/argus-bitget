"""Four factors or eight: which set explains a holding's daily returns better, out of sample.

The exposures answer (`lui/exposures.py`) regressed each holding on market, size, momentum and
crypto. The audit of 2026-09-26 asked for the style and rates factors a risk desk reads — value,
quality, low volatility and interest rates (finding 26). More regressors always raise the in-sample
fit, so that is not the test. This module fits both sets on the first half of the same aligned
daily returns and scores each on the second half it never saw; a factor set that only overfits
loses there.

For each name: the adjusted R² of both sets over the whole sample, the out-of-sample R² of both
(fitted on the first half, predicting the second), and the t-statistic of each added factor in the
eight-factor fit. The names are the US stocks among Bitget's rToken perpetuals that a desk would
hold, fixed here before the run and not chosen by result.

    python -m argus.eval.factor_set_comparison      # needs the network (Yahoo, Bitget)
"""

from __future__ import annotations

import itertools
import json
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from typing import Any

from argus.lui import exposures as ex
from argus.truth.paths import DATA_DIR

REPORT_PATH = DATA_DIR / "factor_set_comparison.json"

NAMES: tuple[str, ...] = (
    "MSFTUSDT", "METAUSDT", "GOOGLUSDT", "NVDAUSDT", "TSLAUSDT", "AAPLUSDT", "AMZNUSDT",
    "COINUSDT", "MSTRUSDT", "HOODUSDT", "JPMUSDT", "XOMUSDT", "JNJUSDT", "KOUSDT", "PGUSDT",
)
"""Fixed 2026-09-27 before the first run: the seven largest US technology names, four
crypto-linked ones, and four from banks, energy, health care and staples so the book is not all
one style. A name whose closes cannot be read is reported, not replaced."""

BASE: tuple[str, ...] = ("market", "size", "momentum", "crypto")
"""The factor set the exposures answer used until 2026-09-27."""

MAX_DAYS = 2 * ex.WINDOW_DAYS
"""Up to two years of aligned days; in practice Bitget's daily BTC history bounds it."""


def adjusted_r2(r2: float, n: int, k: int) -> float:
    return 1.0 - (1.0 - r2) * (n - 1) / (n - k - 1)


def out_of_sample_r2(y: Sequence[float], xs: Sequence[Sequence[float]],
                     split: int) -> float | None:
    """Fit on ``[:split]``, predict ``[split:]``, and score against the test half's own mean."""
    fitted = ex.ols(list(y[:split]), [list(x[:split]) for x in xs])
    if fitted is None:
        return None
    coef = fitted[0]
    test = list(y[split:])
    predicted = [coef[0] + sum(c * x[split + i] for c, x in zip(coef[1:], xs, strict=True))
                 for i in range(len(test))]
    mean = sum(test) / len(test)
    sst = sum((a - mean) ** 2 for a in test)
    if sst <= 0:
        return None
    return 1.0 - sum((a - p) ** 2 for a, p in zip(test, predicted, strict=True)) / sst


def compare(closes: Mapping[date, float], factor_closes: Mapping[str, Mapping[date, float]],
            *, max_days: int = MAX_DAYS) -> dict[str, Any] | None:
    """Both factor sets on one holding's aligned returns, or None with too little history."""
    days = sorted(set(closes).intersection(*(set(factor_closes[k]) for k in ex.FACTOR_SERIES))
                  )[-(max_days + 1):]
    if len(days) - 1 < 2 * ex.MIN_OBS:
        return None
    _, factors = ex.factor_returns({k: {d: v[d] for d in days} for k, v in factor_closes.items()})
    y = [closes[b] / closes[a] - 1.0 for a, b in itertools.pairwise(days)]
    n, split = len(y), len(y) // 2
    out: dict[str, Any] = {"n": n, "start": days[1].isoformat(), "end": days[-1].isoformat()}
    for label, names in (("four", BASE), ("eight", ex.FACTORS)):
        xs = [factors[f] for f in names]
        fitted = ex.ols(y, xs)
        if fitted is None:
            return None
        coef, ses, r2 = fitted
        out[label] = {"adjusted_r2": round(adjusted_r2(r2, n, len(names)), 4),
                      "out_of_sample_r2": _round(out_of_sample_r2(y, xs, split))}
        if label == "eight":
            out["added_t"] = {f: round(coef[i + 1] / ses[i + 1], 2)
                              for i, f in enumerate(names) if f not in BASE and ses[i + 1] > 0}
    return out


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


def summarise(results: Mapping[str, Any]) -> dict[str, Any]:
    rows = [r for r in results.values() if isinstance(r, dict) and "four" in r]
    paired = [r for r in rows if r["four"]["out_of_sample_r2"] is not None
              and r["eight"]["out_of_sample_r2"] is not None]

    def mean(values: Sequence[float]) -> float | None:
        return round(sum(values) / len(values), 4) if values else None

    added = [f for f in ex.FACTORS if f not in BASE]
    return {
        "names_fitted": len(rows),
        "mean_adjusted_r2": {"four": mean([r["four"]["adjusted_r2"] for r in rows]),
                             "eight": mean([r["eight"]["adjusted_r2"] for r in rows])},
        "mean_out_of_sample_r2": {
            "four": mean([r["four"]["out_of_sample_r2"] for r in paired]),
            "eight": mean([r["eight"]["out_of_sample_r2"] for r in paired])},
        "eight_better_out_of_sample": sum(
            1 for r in paired if r["eight"]["out_of_sample_r2"] > r["four"]["out_of_sample_r2"]),
        "of": len(paired),
        "added_factor_significant": {
            f: sum(1 for r in rows if abs(r.get("added_t", {}).get(f, 0.0)) >= 2.0)
            for f in added},
    }


def run() -> dict[str, Any]:
    factor_closes = ex.factor_closes()
    results: dict[str, Any] = {}
    for symbol in NAMES:
        try:
            closes, origin = ex.closes_for(symbol)
        except Exception as exc:
            results[symbol] = {"error": f"closes unreadable ({type(exc).__name__})"}
            continue
        compared = compare(closes, factor_closes)
        results[symbol] = ({"error": "too few aligned days"} if compared is None
                           else {**compared, "origin": origin})
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "method": ("each name's daily returns regressed by least squares on the four factors the "
                   "exposures answer used until 2026-09-27 and on the eight it uses now; adjusted "
                   "R2 over the whole sample, and out-of-sample R2 of a fit on the first half "
                   "scored on the second half against that half's own mean"),
        "factors": {"four": list(BASE), "eight": list(ex.FACTORS)},
        "summary": summarise(results),
        "results": results,
    }


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI, live network
    report = run()
    REPORT_PATH.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report["summary"], indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))


__all__ = ["BASE", "NAMES", "REPORT_PATH", "adjusted_r2", "compare", "main", "out_of_sample_r2",
           "run", "summarise"]
