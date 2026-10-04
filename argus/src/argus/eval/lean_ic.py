"""Information coefficient of the desk's lean — does its forced-direction call rank the next move?

Build-list 2.1 (agent self-audit). Every abstention records the direction the desk would take if
forced (``lean``) and how sure it was (``lean_confidence``). The score here is the signed
confidence, ``+confidence`` for up and ``-confidence`` for down; the outcome is the move the
decision was later marked at (`paper/marks.py`). Leans of ``"none"`` are not scored: they are an
answer, not a missing value, and `eval/refusal.py` already reports how often they occur.

**Method, taken from alphalens** (``research/repos/alphalens-reloaded``, Apache-2.0):
``performance.factor_information_coefficient`` takes a Spearman rank correlation between the
factor and the forward return *per date*, across the assets of that date, and
``plotting.plot_information_table`` (``plotting.py:182-186``) summarises the per-date series as
IC mean, IC std, risk-adjusted IC (mean / std) and ``scipy.stats.ttest_1samp`` against zero. Here
a "date" is one decision cycle, which is where alphalens's independence assumption actually holds.
The pooled correlation over every call is reported next to it, never in place of it: calls in one
cycle share one market move, so the pooled ``n`` overstates the evidence.

**What is different, and why.**

* **Pure Python.** The hosted console ships no scipy, so the one-sample t is computed directly and
  its two-sided p-value comes from the Student-t distribution by the regularised incomplete beta
  function (Numerical Recipes' continued fraction), not a normal approximation.
* **Horizons are never pooled**, for the reason `eval/refusal.py` gives: a decision in the last
  cycle before the overnight gap is next seen about 18h later, and averaging that with the 2h marks
  would measure the recording schedule. The ``about_2h`` bucket is the one whose windows do not
  overlap; it is the headline.
* **The settled ledger is not used.** Abstentions settle once, at 24h or more, while decisions are
  two hours apart, so consecutive settlement windows share most of their span. Computed on it (2026-
  10-05) the per-cycle IC came out -0.092 with t = -2.14 — a significant-looking number built from
  windows that overlap by ~92%, which is exactly the defect refusal.py documents. On the
  non-overlapping 2h marks the same desk scores +0.04 with t near 0.9. Recorded so the first number
  is not rediscovered and published.

    python -m argus.eval.lean_ic       # writes data/lean_ic.json
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from argus.agents.rubric import spearman
from argus.eval.refusal import HORIZONS
from argus.paper.marks import Mark, read_marks
from argus.truth import artefact
from argus.truth.paths import DATA_DIR

LEDGER_PATH = DATA_DIR / "paper_ledger.jsonl"
OUT = DATA_DIR / "lean_ic.json"
MIN_CALLS_PER_CYCLE = 4
"""A cycle's rank correlation over fewer calls is mostly noise and is left out of the series."""


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta (Numerical Recipes 3rd ed., ``betacf``)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1 - qab * x / qap
    d = 1 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d
        d = 1 / (d if abs(d) > tiny else tiny)
        c = 1 + aa / c if abs(1 + aa / c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d
        d = 1 / (d if abs(d) > tiny else tiny)
        c = 1 + aa / c if abs(1 + aa / c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1) < 3e-14:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta ``I_x(a, b)``."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                     + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1) / (a + b + 2):
        return front * _betacf(a, b, x) / a
    return 1 - front * _betacf(b, a, 1 - x) / b


def t_two_sided_p(t: float, df: int) -> float:
    """Two-sided p-value of a Student-t statistic, as ``scipy.stats.ttest_1samp`` reports it."""
    return _betai(df / 2, 0.5, df / (df + t * t))


def ttest_1samp(values: Sequence[float]) -> tuple[float, float] | None:
    """``(t, p)`` of the mean against zero; ``None`` below three values or with no spread."""
    n = len(values)
    if n < 3:
        return None
    mean = sum(values) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
    if sd == 0:
        return None
    t = mean / (sd / math.sqrt(n))
    return t, t_two_sided_p(t, n - 1)


def _score(lean: str, confidence: float) -> float:
    return confidence if lean == "up" else -confidence


def horizon_ic(calls: Sequence[tuple[str, float, float]]) -> dict[str, Any]:
    """IC summary for ``(cycle, score, move_bps)`` calls of one horizon."""
    scores, moves = [s for _, s, _ in calls], [m for _, _, m in calls]
    by_cycle: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for cycle, s, m in calls:
        by_cycle[cycle].append((s, m))
    series = []
    for pairs in by_cycle.values():
        if len(pairs) < MIN_CALLS_PER_CYCLE:
            continue
        value = spearman([s for s, _ in pairs], [m for _, m in pairs])
        if value is not None:
            series.append(value)
    out: dict[str, Any] = {"calls": len(calls), "cycles_with_ic": len(series),
                           "hit_rate": None, "pooled_ic": None, "ic_mean": None, "ic_std": None,
                           "risk_adjusted_ic": None, "t_stat": None, "p_value": None}
    if calls:
        out["hit_rate"] = round(sum(1 for s, m in zip(scores, moves, strict=True) if s * m > 0)
                                / len(calls), 4)
    pooled = spearman(scores, moves)
    if pooled is not None:
        out["pooled_ic"] = round(pooled, 4)
    if len(series) >= 2:
        mean = sum(series) / len(series)
        sd = math.sqrt(sum((v - mean) ** 2 for v in series) / (len(series) - 1))
        out["ic_mean"], out["ic_std"] = round(mean, 4), round(sd, 4)
        out["risk_adjusted_ic"] = round(mean / sd, 4) if sd else None
    test = ttest_1samp(series)
    if test is not None:
        out["t_stat"], out["p_value"] = round(test[0], 3), round(test[1], 4)
    return out


def _confidences(path: Path) -> dict[int, float]:
    found: dict[int, float] = {}
    if not path.exists():
        return found
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "seq" in row and row.get("lean_confidence") is not None:
            found[int(row["seq"])] = float(row["lean_confidence"])
    return found


def score(marks: Sequence[Mark], confidences: Mapping[int, float]) -> dict[str, Any]:
    buckets: dict[str, list[tuple[str, float, float]]] = {name: [] for name, _, _ in HORIZONS}
    unscored = {"lean_none": 0, "no_confidence": 0, "outside_horizons": 0}
    for mark in marks:
        if mark.lean not in ("up", "down"):
            unscored["lean_none"] += 1
            continue
        confidence = confidences.get(mark.seq)
        if confidence is None:
            unscored["no_confidence"] += 1
            continue
        name = next((n for n, lo, hi in HORIZONS if lo <= mark.horizon_hours < hi), None)
        if name is None:
            unscored["outside_horizons"] += 1
            continue
        buckets[name].append((mark.decided_at[:16], _score(mark.lean, confidence), mark.move))
    return {"horizons": {name: horizon_ic(calls) for name, calls in buckets.items()},
            "unscored": unscored, "headline_horizon": "about_2h",
            "min_calls_per_cycle": MIN_CALLS_PER_CYCLE}


def line(path: Path = OUT) -> str | None:
    """One sentence for /status, or ``None`` when the artefact is absent or has no headline."""
    if not path.exists():
        return None
    blob = json.loads(path.read_text(encoding="utf-8"))
    head = blob.get("horizons", {}).get("about_2h", {})
    if head.get("ic_mean") is None or head.get("t_stat") is None:
        return None
    verdict = "no evidence of skill" if head["p_value"] >= 0.05 else (
        "a ranking skill" if head["ic_mean"] > 0 else "a contrary ranking")
    return (f"Lean IC at ~2h: mean {head['ic_mean']:+.3f} over {head['cycles_with_ic']} cycles "
            f"(t {head['t_stat']:+.2f}, p {head['p_value']:.2f}; {head['calls']} calls, hit rate "
            f"{100 * head['hit_rate']:.1f}%) — {verdict}.")


def run(*, out: Path = OUT, marks_path: Path | None = None,
        ledger_path: Path = LEDGER_PATH) -> dict[str, Any]:
    marks = read_marks(marks_path) if marks_path is not None else read_marks()
    blob = {"generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            **score(marks, _confidences(ledger_path))}
    artefact.write(out, blob)
    return blob


def main() -> int:  # pragma: no cover - CLI
    blob = run()
    print(json.dumps(blob, indent=1))
    print(line())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
