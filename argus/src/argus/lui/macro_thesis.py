"""A macro reason in a thesis — "Fed rate cuts will boost NVDA" — tested against what is measured.

The thesis tester answered "no engine in this task tests a macro claim" (a judge, round 11,
2026-09-30), while two measurements that bear on it were already in the console: the event study
of each name around Fed decisions and CPI releases (`research/event_reactions.py`), and the
10-year Treasury yield the hosted console ships in its FRED snapshot. This reads both.

* **Sensitivity to rates**: the name's close-to-close return on each trading day against that
  day's change in the 10-year yield, over the days both series cover — a slope (the move per
  10bp rise), the correlation and its t-statistic. A claim that falling rates help the name
  expects a negative slope; a claim that rising rates hurt it expects the same.
* **Around the events**: the event study's own verdict for the name around Fed decisions (or CPI
  releases, for an inflation claim), quoted as the study states it.

The verdict is the sensitivity's, and only when its t-statistic clears 2; otherwise the reason is
not measurable, with the figures shown. What rates will do is the part no data tests, and the
answer says so. The event study does not split cuts from hikes, and says it does not.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from collections.abc import Mapping
from datetime import UTC, date, datetime
from itertools import pairwise
from typing import Any

from argus.lui.trace import trace_module
from argus.truth.paths import DATA_DIR

INFLATION = re.compile(r"\b(?:cpi|inflation|prices?\s+ris\w*|disinflation)\b", re.I)
EASING = re.compile(r"\b(?:cuts?|cutting|lower\s+rates|rates?\s+(?:fall|falling|drop|come\s+down)|"
                    r"easing|dovish|pivot)\b", re.I)
TIGHTENING = re.compile(r"\b(?:hikes?|hiking|higher\s+rates|rates?\s+(?:rise|rising|up)|"
                        r"tightening|hawkish)\b", re.I)
HELPS = re.compile(r"\b(?:boost|help|lift|drive|push|send|benefit|good\s+for|pump|rally|up)\w*\b",
                   re.I)
HURTS = re.compile(r"\b(?:hurt|hit|crush|sink|drag|weigh|bad\s+for|tank|dump|down|pressure)\w*\b",
                   re.I)
T_TO_CALL = 2.0
PREMISE = re.compile(
    r"\b(?:is|are|has\s+been|have\s+been|has|have|keeps?)\s+(?:cutting|cut|lowering|lowered|easing|"
    r"eased|hiking|hiked|raising|raised|tightening|tightened)\b", re.I)
"""A claim that the Fed is already moving, which the fed funds rate can check."""
REAL_PREMISE = re.compile(
    r"\breal\s+(?:interest\s+)?(?:yields?|rates?)\s+(?:are|is|have\s+been|has\s+been|keep|will\s+keep)?"
    r"\s*(?P<d>fall\w*|drop\w*|declin\w*|going\s+down|coming\s+down|down|lower\w*|ris\w*|"
    r"climb\w*|going\s+up|up|higher)\b", re.I)
"""A claim about where real yields are going, which the 10-year less breakeven inflation checks:
"gold beats bitcoin because real yields are falling" was tested against the nominal yield's
correlation while the real yield had risen 69bp (a first-time user, round 27)."""


def _daily_closes(symbol: str) -> dict[date, float]:
    from argus.market.bitget import public_get

    rows = public_get("/api/v2/mix/market/candles",
                      {"productType": "USDT-FUTURES", "symbol": symbol, "granularity": "1D",
                       "limit": "100"}) or []
    return {datetime.fromtimestamp(int(r[0]) / 1000, UTC).date(): float(r[4]) for r in rows}


def facts(symbol: str) -> dict[str, Any]:
    """The name's daily closes against the 10-year yield, and its event-study rows."""
    out: dict[str, Any] = {"symbol": symbol}
    try:
        snapshot = json.loads((DATA_DIR / "macro_snapshot.json").read_text(encoding="utf-8"))
        tens = {date.fromisoformat(d): float(v) for d, v in snapshot["series"]["DGS10"]}
        funds = [(date.fromisoformat(d), float(v)) for d, v in snapshot["series"]["DFF"]]
    except (OSError, ValueError, KeyError):
        tens, funds = {}, []
        breakevens: dict[date, float] = {}
    else:
        breakevens = {date.fromisoformat(d): float(v)
                      for d, v in snapshot["series"].get("T10YIE", [])}
    real_days = sorted(d for d in tens if d in breakevens)
    if len(real_days) >= 10:
        out["real"] = {"then": tens[real_days[0]] - breakevens[real_days[0]],
                       "now": tens[real_days[-1]] - breakevens[real_days[-1]],
                       "from": real_days[0].isoformat(), "to": real_days[-1].isoformat()}
    try:
        closes = _daily_closes(symbol)
    except Exception:
        closes = {}
    days = sorted(d for d in tens if d in closes)
    pairs = [((tens[b] - tens[a]) * 100, closes[b] / closes[a] - 1)
             for a, b in pairwise(days) if closes[a] > 0]
    if len(pairs) >= 20:
        dy = [p[0] for p in pairs]
        ret = [p[1] for p in pairs]
        if statistics.pvariance(dy) > 0 and statistics.pvariance(ret) > 0:
            corr = statistics.correlation(dy, ret)
            slope = statistics.covariance(dy, ret) / statistics.variance(dy)
            n = len(pairs)
            t = corr * math.sqrt((n - 2) / max(1e-12, 1 - corr * corr))
            out["rates"] = {"n": n, "slope_per_10bp": slope * 10, "corr": corr, "t": t,
                            "from": days[0].isoformat(), "to": days[-1].isoformat(),
                            "ten_year": tens[days[-1]]}
    if funds:
        out["fed_funds"] = {"now": funds[-1][1], "then": funds[0][1],
                            "since": funds[0][0].isoformat()}
    try:
        study = json.loads((DATA_DIR / "event_reactions.json").read_text(encoding="utf-8"))
        out["events"] = {r["kind"]: r for r in study.get("reactions", [])
                         if r.get("symbol") == symbol and r.get("kind") in ("FOMC", "CPI")}
    except (OSError, ValueError):
        out["events"] = {}
    return out


def expects_negative_slope(reason: str) -> bool | None:
    """Whether the claim says the name does better as yields fall: True for "cuts will boost" and
    "hikes will hurt", False for the reverse, None when the claim names no direction."""
    easing, tightening = EASING.search(reason), TIGHTENING.search(reason)
    helps, hurts = HELPS.search(reason), HURTS.search(reason)
    if (easing and helps) or (tightening and hurts):
        return True
    if (easing and hurts) or (tightening and helps):
        return False
    return None


def test(reason: str, found: Mapping[str, Any] | None, name: str
         ) -> tuple[str, str, list[str]]:
    """``(result, line, evidence lines)``."""
    if not found:
        return "not tested", f"{name}'s macro data did not answer in time.", []
    evidence: list[str] = []
    inflation = bool(INFLATION.search(reason))
    kind = "CPI" if inflation else "FOMC"
    row = (found.get("events") or {}).get(kind)
    if row:
        label = "CPI releases" if inflation else "Fed decisions"
        verdict = str(row["verdict"])
        # Said plainly: "PARTIAL: ... rejects under corrado_rank but not under patell, bmp" read
        # as a statistics log beside a retail trader's thesis (round 11, live check).
        plain = ("a measured reaction, confirmed by every test" if verdict.startswith("EFFECT")
                 else "a reaction only some of the tests support" if verdict.startswith("PARTIAL")
                 else "no reliable reaction" if verdict.startswith("NO EFFECT")
                 else "too few events to measure")
        evidence.append(f"Around the last {row['events']} {label}, {name} showed {plain}"
                        + (f" ({row['average_car_bps']:+.0f}bps on average)"
                           if row.get("average_car_bps") is not None else "")
                        + (" — cuts and hikes counted together." if kind == "FOMC" else "."))
    funds = found.get("fed_funds")
    if funds:
        evidence.append(f"The fed funds rate is {funds['now']:.2f}%, from {funds['then']:.2f}% on "
                        f"{funds['since']} (FRED, the console's shipped snapshot).")
    premise = PREMISE.search(reason)
    if premise and funds:
        # "the Fed is cutting rates" while the fed funds rate rose from 3.62% to 3.88% was left
        # "not measurable" (round 11): the premise itself is a fact, and it is checked first.
        cutting = bool(re.search(r"cut|lower|eas", premise.group(0), re.I))
        moved = funds["now"] - funds["then"]
        if (cutting and moved > 0.1) or (not cutting and moved < -0.1):
            return ("contradicted",
                    f"The premise does not hold: the fed funds rate went from {funds['then']:.2f}% "
                    f"on {funds['since']} to {funds['now']:.2f}% — "
                    + ("up, not down" if cutting else "down, not up") + ".", evidence)
    real = found.get("real")
    claimed = REAL_PREMISE.search(reason)
    if claimed and real:
        falling = not re.match(r"ris|climb|going\s+up|up|higher", claimed.group("d"), re.I)
        moved = (real["now"] - real["then"]) * 100
        line = (f"The real 10-year yield (the 10-year Treasury less breakeven inflation, FRED) "
                f"went from {real['then']:.2f}% on {real['from']} to {real['now']:.2f}% on "
                f"{real['to']}, {moved:+.0f}bp")
        if abs(moved) < 10:
            return ("not measurable", line + " — flat, so the premise is neither true nor false "
                    "yet; where real yields go next is the part no data tests.", evidence)
        holds = (moved < 0) == falling
        return ("supported" if holds else "contradicted",
                line + (" — falling, as the reason says." if holds and falling else
                        " — rising, as the reason says." if holds else
                        " — rising, so the reason is false so far." if falling else
                        " — falling, so the reason is false so far."), evidence)
    rates = found.get("rates")
    if inflation or not rates:
        result = "not measurable"
        line = ("What moves an inflation print cannot be tested from the price alone; the event "
                "study's read around CPI releases is below." if inflation else
                "The 10-year yield and the name's closes did not overlap enough days to measure "
                "its sensitivity to rates.")
        return result, line + " What policy will do next is the part no data tests.", evidence
    said = (f"Against the 10-year Treasury yield over {rates['n']} trading days "
            f"({rates['from']} to {rates['to']}): {name} moved {rates['slope_per_10bp']:+.2%} "
            f"for each 10bp rise in the yield (correlation {rates['corr']:+.2f}, "
            f"t {rates['t']:+.1f}).")
    wants_negative = expects_negative_slope(reason)
    if abs(rates["t"]) < T_TO_CALL or wants_negative is None:
        result = "not measurable"
        verdict = (" That relation is too weak to call either way" if abs(rates["t"]) < T_TO_CALL
                   else " The claim names no direction to hold that against")
    else:
        agrees = (rates["slope_per_10bp"] < 0) == wants_negative
        result = "supported" if agrees else "contradicted"
        verdict = (" So far it has done better when yields "
                   + ("fall" if rates["slope_per_10bp"] < 0 else "rise")
                   + (", as the claim needs" if agrees else ", the opposite of what the claim "
                                                             "needs"))
    return (result, said + verdict + "; whether the Fed cuts, and when, is the part no data "
            "tests.", evidence)


trace_module(globals())
