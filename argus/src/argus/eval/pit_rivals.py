"""Point-in-time SEC fundamentals: ARGUS against three agent toolkits, on the same frozen filings.

A research agent asked "what was NVDA's latest quarterly revenue as of Tuesday 14:00 UTC" must not
answer with a 10-Q EDGAR accepted at 20:36 that evening, and must not keep serving a number the
company has since restated. `market/pit.py` gates every XBRL row to the second EDGAR accepted its
filing and resolves each period to the latest filing public at the cutoff. The rivals, each read
from its source and run here unmodified:

* **Vibe-Trading** (HKUDS/Vibe-Trading, MIT) — its agent tool ``get_fundamentals`` in both modes:
  ``pit=True`` aligns a value to its *filed date* and keeps the *first* filing of a period;
  ``pit=False`` keeps the *last* (`backtest/loaders/fundamentals_loader.py:111`).
* **OpenBB ODP** (AGPL-3.0, its own environment) — its SEC income-statement fetcher in default and
  ``pit_mode`` (`openbb_sec/utils/statement_schema/_extraction.py:677-682`), the path
  ``obb.equity.fundamental.income(provider="sec")`` takes.
* **LangAlpha** (Apache-2.0) — its keyless yfinance fundamentals tool, which takes no date at all.

The runners (`scripts/pit_runners/`) serve all of them the same SEC snapshot and record what their
code returned. This module scores those records and ARGUS on the same questions.

**The questions.** For every 10-Q and 10-K that carries a quarterly revenue fact, in 2017-2026, for
ten tickers: the latest quarter's revenue as of one hour *before* EDGAR accepted it, and one hour
*after*. Before, the right answer is the previous quarter; after, it is the new one.

**Truth** is computed here from the snapshot by the rule itself, independently of `market/pit.py`:
a row counts once its accession's acceptance instant has passed; each period takes the best-ranked
tag with a visible row, and within it the most recently accepted value.

**What each system is scored on.**

* *Leak*: it answers with a row not yet accepted at the cutoff. A filed-date gate leaks on every
  filing accepted during the day it is dated; no gate leaks on every one.
* *Stale after restatement*: once a restatement is public, it still serves the first-reported
  value.
* *Wrong derived per-share figure*: a fiscal-Q4 EPS made by subtraction across a year in which the
  share count moved.

The rivals' own outputs are checked against the behaviour this module attributes to them before
any score uses it: their first-or-last choice on every restated period, and their date alignment.

    python -m argus.eval.pit_rivals        # scores research/pit_rivals/outputs/*.json
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from typing import Any

from argus.market.fundamentals import CONCEPTS
from argus.market.pit import (
    AcceptanceIndex,
    PitFact,
    facts_from_units,
    parse_acceptance,
    resolve,
)

PACKAGE = Path(__file__).resolve().parents[3]
WORKSPACE = PACKAGE.parent
SNAPSHOT = WORKSPACE / "research" / "pit_rivals" / "snapshot"
OUTPUTS = WORKSPACE / "research" / "pit_rivals" / "outputs"
ARTEFACT = PACKAGE / "data" / "pit_rivals.json"

TICKERS = ("NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "AMD", "COIN", "MSTR")
FROM_YEAR = 2017
OFFSET = timedelta(hours=1)
QUARTERLY_FORMS = frozenset({"10-Q", "10-K", "10-Q/A", "10-K/A"})

# (gate, supersede) per system, as each one's code chooses them; checked against their outputs
CONFIGS: dict[str, tuple[str, str]] = {
    "argus": ("accepted", "latest"),
    "vibe_trading_pit": ("filed-date", "first"),
    "vibe_trading_research": ("none", "latest"),
    "openbb_pit": ("none", "first"),
    "openbb_default": ("none", "latest"),
    "langalpha_yfinance": ("none", "latest"),
}


# --- the snapshot as facts --------------------------------------------------------------------


def acceptance_index(ticker: str, snapshot: Path = SNAPSHOT) -> AcceptanceIndex:
    raw = json.loads((snapshot / f"{ticker}.acceptance.json").read_text(encoding="utf-8"))
    accepted: dict[str, datetime] = {}
    for accn, row in raw["accessions"].items():
        when = parse_acceptance(str(row.get("accepted", "")))
        if when is not None:
            accepted[accn] = when
    return AcceptanceIndex(cik=int(raw["cik"]), accepted=accepted,
                           fetched_at=datetime(2026, 9, 25, tzinfo=UTC), pages=int(raw["n_pages"]))


def load_facts(ticker: str, concept: str, snapshot: Path = SNAPSHOT) -> list[PitFact]:
    """Every row of every alias tag of ``concept`` for ``ticker``, stamped with its acceptance."""
    facts = json.loads((snapshot / f"{ticker}.companyfacts.json").read_text(encoding="utf-8"))
    gaap = facts.get("facts", {}).get("us-gaap", {})
    index = acceptance_index(ticker, snapshot)
    out: list[PitFact] = []
    for rank, tag in enumerate(CONCEPTS[concept]):
        units = (gaap.get(tag) or {}).get("units") or {}
        out.extend(facts_from_units(concept, tag, rank, units, index, observed_at=None))
    return out


# --- truth, from the rule, independently of market/pit.py -------------------------------------


def truth_latest_quarter(facts: Sequence[PitFact], as_of: datetime) -> tuple[date, float] | None:
    """The newest quarter knowable at ``as_of`` and its current value, from first principles."""
    periods: dict[date, list[PitFact]] = {}
    for f in facts:
        if f.available_at <= as_of and f.is_quarterly:
            periods.setdefault(f.end, []).append(f)
    if not periods:
        return None
    end = max(periods)
    rows = periods[end]
    best = min(f.tag_rank for f in rows)
    latest = max((f for f in rows if f.tag_rank == best), key=lambda f: (f.available_at, f.accn))
    return end, latest.value


def answer(facts: Sequence[PitFact], as_of: datetime, config: str) -> PitFact | None:
    """The latest quarter a system configured as ``config`` shows at ``as_of``."""
    gate, supersede = CONFIGS[config]
    view = resolve(facts, concept=facts[0].concept if facts else "revenue", as_of=as_of,
                   gate=gate, supersede=supersede, derive_q4=False)  # type: ignore[arg-type]
    quarterly = [f for f in view.facts if f.is_quarterly]
    return quarterly[0] if quarterly else None


# --- probes -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Probe:
    ticker: str
    accn: str
    accepted: datetime
    side: str  # "before" | "after"

    @property
    def as_of(self) -> datetime:
        return self.accepted - OFFSET if self.side == "before" else self.accepted + OFFSET


def probes_for(ticker: str, facts: Sequence[PitFact]) -> list[Probe]:
    seen: dict[str, datetime] = {}
    for f in facts:
        if (f.form in QUARTERLY_FORMS and f.basis == "accepted" and f.is_quarterly
                and f.available_at.year >= FROM_YEAR):
            seen.setdefault(f.accn, f.available_at)
    return [Probe(ticker, accn, when, side) for accn, when in sorted(seen.items(),
            key=lambda kv: kv[1]) for side in ("before", "after")]


def score_config(facts: Sequence[PitFact], probes: Sequence[Probe], config: str,
                 as_of_now: datetime) -> dict[str, Any]:
    """Leak and staleness of one configuration over the probes. A system with no date parameter
    (``gate == "none"``) is asked at ``as_of_now`` whatever the probe says, because that is the
    only question its tool can answer."""
    leak = stale = right = late = 0
    rows = []
    for p in probes:
        truth = truth_latest_quarter(facts, p.as_of)
        shown = answer(facts, p.as_of if CONFIGS[config][0] != "none" else as_of_now, config)
        if truth is None or shown is None:
            continue
        leaked = shown.available_at > p.as_of
        is_right = (not leaked) and shown.end == truth[0] and shown.value == truth[1]
        is_stale = (not leaked) and shown.end == truth[0] and shown.value != truth[1]
        is_late = (not leaked) and shown.end < truth[0]
        leak += leaked
        stale += is_stale
        right += is_right
        late += is_late
        rows.append({"ticker": p.ticker, "accn": p.accn, "side": p.side,
                     "as_of": p.as_of.isoformat(), "leak": leaked, "stale": is_stale,
                     "late": is_late, "right": is_right})
    n = len(rows)
    return {"questions": n, "right": right, "leak": leak, "late": late,
            "stale_after_restatement": stale,
            "right_rate": round(right / n, 4) if n else None, "rows": rows}


def score_restatements(facts: Sequence[PitFact], config: str, as_of_now: datetime
                       ) -> dict[str, Any]:
    """The value of each restated quarter, asked an hour before and an hour after the
    restatement was accepted. Before, the right answer is the first-reported value (the restated
    one is not public yet); after, it is the restated value."""
    by_end: dict[date, list[PitFact]] = {}
    for f in facts:
        if f.is_quarterly and f.basis == "accepted":
            by_end.setdefault(f.end, []).append(f)
    gate, supersede = CONFIGS[config]
    right = leak = stale = asked = asked_after = right_after = 0
    for end, rows in sorted(by_end.items()):
        best = min(f.tag_rank for f in rows)
        same = sorted((f for f in rows if f.tag_rank == best),
                      key=lambda f: (f.available_at, f.accn))
        changes = [f for prev, f in pairwise(same) if f.value != prev.value]
        for restated in changes:
            for side, as_of in (("before", restated.available_at - OFFSET),
                                ("after", restated.available_at + OFFSET)):
                public = [f for f in same if f.available_at <= as_of]
                if not public:
                    continue
                truth = public[-1].value
                ask_at = as_of if gate != "none" else as_of_now
                view = resolve(facts, concept=restated.concept, as_of=ask_at, gate=gate,  # type: ignore[arg-type]
                               supersede=supersede, derive_q4=False)  # type: ignore[arg-type]
                shown = view.by_end().get(end)
                if shown is None:
                    continue
                asked += 1
                asked_after += side == "after"
                if shown.available_at > as_of:
                    leak += 1
                elif shown.value == truth:
                    right += 1
                    right_after += side == "after"
                elif side == "after":
                    stale += 1
    return {"questions": asked, "right": right, "leak": leak, "stale_after_restatement": stale,
            "questions_after": asked_after, "right_after": right_after}


# --- the rivals scored on what their own code returned ---------------------------------------


@dataclass(frozen=True)
class Shown:
    """One period and value a rival's output offers."""

    end: date
    value: float
    filed: date | None = None


def rival_rows(system: str, ticker: str, outputs: Mapping[str, Any]) -> list[Shown]:
    """Every quarterly revenue row the rival's own output holds for ``ticker``."""
    if system.startswith("vibe_trading"):
        mode = "pit" if system.endswith("pit") else "research"
        series = (outputs["vibe"]["tickers"].get(ticker, {}).get(mode, {})
                  .get("series", {}).get("revenue", []))
        return [Shown(date.fromisoformat(str(end)), float(value), date.fromisoformat(str(filed)))
                for end, filed, value in series]
    if system.startswith("openbb"):
        mode = "pit" if system.endswith("pit") else "default"
        rows = outputs["openbb"]["tickers"].get(ticker, {}).get(mode, {}).get("rows", [])
        return [Shown(date.fromisoformat(str(r["period_ending"])), float(r["total_revenue"]))
                for r in rows if r.get("total_revenue") is not None]
    if system.startswith("openalice"):
        rows = (outputs["openalice"]["tickers"].get(ticker, {}).get("response", {})
                .get("rows", []))
        return [Shown(date.fromisoformat(str(r["period_ending"])), float(r["revenue"]))
                for r in rows if r.get("revenue") is not None and r.get("period_type") == "3M"]
    data = (outputs["langalpha"]["tickers"].get(ticker, {}).get("response", {}).get("data", {})
            .get("Total Revenue") or {})
    out = []
    for k, v in data.items():
        try:
            out.append(Shown(date.fromisoformat(str(k)[:10]), float(v)))
        except (TypeError, ValueError):
            continue   # Yahoo reports a missing quarter as "N/A"
    return out


DATE_ALIGNED = frozenset({"vibe_trading_pit"})
"""The one rival whose output carries a date to gate by: Vibe-Trading's point-in-time series,
aligned to each value's filed date. Every other rival's tool returns today's rows whatever date it
is asked about."""

RIVALS = ("vibe_trading_pit", "vibe_trading_research", "openbb_pit", "openbb_default",
          "langalpha_yfinance", "openalice_yfinance")

SNAP = timedelta(days=7)
"""Yahoo labels a fiscal quarter by its calendar month-end (NVDA's quarter ending 2026-07-26
arrives as 2026-07-31). A rival row is matched to the SEC quarter whose end is within a week of
its label; quarters are ninety days apart, so no row can match two. Without this, every
Yahoo-backed rival's answer for a 52/53-week filer scored as a leak (fixed 2026-09-26)."""


def snapped(rows: Sequence[Shown], ends: Sequence[date]) -> list[Shown]:
    """``rows`` relabelled to the SEC quarter end each one reports, where one is within `SNAP`."""
    out = []
    for r in rows:
        near = min(ends, key=lambda e: abs((e - r.end).days), default=None)
        if near is not None and abs(near - r.end) <= SNAP:
            r = Shown(near, r.value, r.filed)
        out.append(r)
    return out


def first_public(facts: Sequence[PitFact]) -> dict[date, datetime]:
    """When each quarter first became public: its earliest accepted row."""
    out: dict[date, datetime] = {}
    for f in facts:
        if f.is_quarterly and f.basis == "accepted":
            out[f.end] = min(out.get(f.end, f.available_at), f.available_at)
    return out


def rival_outcomes(system: str, ticker: str, facts: Sequence[PitFact], probes: Sequence[Probe],
                   outputs: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Each latest-quarter question the rival's own rows can answer, with its outcome: ``right``,
    ``leak`` (a quarter not yet public), ``late`` (an older quarter although a newer one was
    public) or ``stale`` (the right quarter with a superseded value)."""
    public = first_public(facts)
    rows = snapped(rival_rows(system, ticker, outputs), sorted(public))
    out = []
    for p in probes:
        truth = truth_latest_quarter(facts, p.as_of)
        if truth is None:
            continue
        if system in DATE_ALIGNED:
            usable = [r for r in rows if r.filed is not None and r.filed <= p.as_of.date()]
        else:
            usable = list(rows)
        if not usable:
            continue
        shown = max(usable, key=lambda r: r.end)
        became = public.get(shown.end)
        if became is None or became > p.as_of:
            outcome = "leak"
        elif shown.end < truth[0]:
            outcome = "late"
        elif shown.end == truth[0] and shown.value == truth[1]:
            outcome = "right"
        else:
            outcome = "stale"
        out.append({"ticker": p.ticker, "accn": p.accn, "side": p.side,
                    "as_of": p.as_of.isoformat(), "outcome": outcome})
    return out


def score_rival(system: str, ticker: str, facts: Sequence[PitFact], probes: Sequence[Probe],
                outputs: Mapping[str, Any]) -> dict[str, int]:
    """The latest-quarter questions, answered from the rival's own rows."""
    rows = rival_outcomes(system, ticker, facts, probes, outputs)
    count = {k: sum(r["outcome"] == k for r in rows) for k in ("right", "leak", "late", "stale")}
    return {"questions": len(rows), "right": count["right"], "leak": count["leak"],
            "late": count["late"], "stale_after_restatement": count["stale"]}


def exact_mcnemar(argus_only: int, rival_only: int) -> float:
    """Two-sided exact McNemar p-value: the discordant pairs against a fair coin."""
    n = argus_only + rival_only
    if n == 0:
        return 1.0
    k = min(argus_only, rival_only)
    tail = float(Fraction(sum(math.comb(n, i) for i in range(k + 1)), 1 << n))
    return min(1.0, 2 * tail)


def wilson(right: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for a proportion."""
    if n == 0:
        return (0.0, 1.0)
    phat = right / n
    centre = (phat + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (round(centre - half, 4), round(centre + half, 4))


DESIGN_TICKER = "NVDA"
"""The gate was designed on one filing, NVDA's 10-Q dated 2026-08-26 and accepted 20:36 UTC.
The other nine tickers were never looked at while it was built: they are the held-out set."""


def paired(argus_rows: Sequence[Mapping[str, Any]], rival_rows_: Sequence[Mapping[str, Any]]
           ) -> dict[str, Any]:
    """ARGUS against one rival on the questions both answered, paired by (accession, side)."""
    mine = {(r["accn"], r["side"]): bool(r["right"]) for r in argus_rows}
    both = [(mine[(r["accn"], r["side"])], r["outcome"] == "right") for r in rival_rows_
            if (r["accn"], r["side"]) in mine]
    argus_only = sum(a and not b for a, b in both)
    rival_only = sum(b and not a for a, b in both)
    n = len(both)
    return {"n": n, "argus_right": sum(a for a, _ in both), "rival_right": sum(b for _, b in both),
            "argus_only_right": argus_only, "rival_only_right": rival_only,
            "p_value": exact_mcnemar(argus_only, rival_only)}


def score_rival_restatements(system: str, ticker: str, facts: Sequence[PitFact],
                             outputs: Mapping[str, Any]) -> dict[str, int]:
    """Each restated quarter an hour after the restatement: the rival's value for it against the
    restated value. (Before the restatement, a date-less rival cannot be asked.)"""
    ends = sorted(first_public(facts))
    rows = {r.end: r for r in snapped(rival_rows(system, ticker, outputs), ends)}
    right = stale = other = asked = 0
    for end, first, latest in restated_periods(facts):
        shown = rows.get(end)
        if shown is None:
            continue
        asked += 1
        if shown.value == latest:
            right += 1
        elif shown.value == first:
            stale += 1
        else:
            other += 1
    return {"questions": asked, "right": right, "stale_after_restatement": stale,
            "neither_value": other}


# --- the rivals' own outputs, checked against the behaviour attributed to them ---------------


def restated_periods(facts: Sequence[PitFact]) -> list[tuple[date, float, float]]:
    """Quarterly periods whose best tag carries two different values: (end, first, latest)."""
    by_end: dict[date, list[PitFact]] = {}
    for f in facts:
        if f.is_quarterly:
            by_end.setdefault(f.end, []).append(f)
    out = []
    for end, rows in sorted(by_end.items()):
        best = min(f.tag_rank for f in rows)
        same = sorted((f for f in rows if f.tag_rank == best),
                      key=lambda f: (f.available_at, f.accn))
        if len({f.value for f in same}) > 1:
            out.append((end, same[0].value, same[-1].value))
    return out


def vibe_keeps(series: Iterable[Sequence[Any]], restated: Sequence[tuple[date, float, float]]
               ) -> dict[str, int]:
    """For each restated period, whether Vibe-Trading's own series kept the first or last value."""
    values = {str(r[0]): float(r[2]) for r in series}
    first = last = other = 0
    for end, v_first, v_last in restated:
        got = values.get(end.isoformat())
        if got is None:
            continue
        if got == v_first:
            first += 1
        elif got == v_last:
            last += 1
        else:
            other += 1
    return {"first": first, "last": last, "neither": other}


def vibe_date_leaks(series: Iterable[Sequence[Any]], facts: Sequence[PitFact]) -> dict[str, Any]:
    """Vibe-Trading aligns each value to its filed *date*. For each quarter it shows, the hours on
    that date before EDGAR accepted the filing are hours its tool served a number not yet public."""
    accepted: dict[tuple[str, str], datetime] = {}
    for f in facts:
        if f.is_quarterly and f.basis == "accepted":
            accepted.setdefault((f.end.isoformat(), f.filed.isoformat()), f.available_at)
    hours = []
    for end, filed, _value in series:
        when = accepted.get((str(end), str(filed)))
        if when is None:
            continue
        start_of_day = datetime.combine(date.fromisoformat(str(filed)), datetime.min.time(),
                                        tzinfo=UTC)
        hours.append((when - start_of_day).total_seconds() / 3600)
    return {"filings": len(hours), "hours_early_median": round(statistics.median(hours), 2)
            if hours else None, "hours_early_max": round(max(hours), 2) if hours else None}


def openbb_keeps(rows: Sequence[Mapping[str, Any]], restated: Sequence[tuple[date, float, float]]
                 ) -> dict[str, int]:
    values = {str(r.get("period_ending")): r.get("total_revenue") for r in rows}
    first = last = other = 0
    for end, v_first, v_last in restated:
        got = values.get(end.isoformat())
        if got is None:
            continue
        if float(got) == v_first:
            first += 1
        elif float(got) == v_last:
            last += 1
        else:
            other += 1
    return {"first": first, "last": last, "neither": other}


def openbb_q4_eps(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """OpenBB's fiscal-Q4 diluted EPS against its own Q4 net income over its own Q4 diluted share
    count: a subtraction across a share-count change shows up as a gap between the two."""
    out = []
    for r in rows:
        if r.get("fiscal_period") != "Q4":
            continue
        eps, ni, shares = r.get("diluted_eps"), r.get("net_income"), r.get(
            "weighted_ave_diluted_shares_os")
        if eps is None or not ni or not shares:
            continue
        implied = float(ni) / float(shares)
        out.append({"period_ending": r.get("period_ending"), "diluted_eps": float(eps),
                    "net_income_over_shares": round(implied, 4),
                    "off_by_more_than_20pct": abs(float(eps) - implied) > 0.2 * abs(implied)})
    return out


def run(snapshot: Path = SNAPSHOT, outputs: Path = OUTPUTS) -> dict[str, Any]:
    vibe = json.loads((outputs / "vibe.json").read_text(encoding="utf-8"))
    openbb = json.loads((outputs / "openbb.json").read_text(encoding="utf-8"))
    langalpha = json.loads((outputs / "langalpha.json").read_text(encoding="utf-8"))
    openalice = json.loads((outputs / "openalice.json").read_text(encoding="utf-8"))
    as_of_now = datetime(2026, 9, 25, 23, 59, tzinfo=UTC)
    per_config: dict[str, dict[str, Any]] = {c: {"questions": 0, "right": 0, "leak": 0,
                                                 "late": 0, "stale_after_restatement": 0,
                                                 "rows": []}
                                             for c in CONFIGS}
    restatement: dict[str, dict[str, int]] = {c: {"questions": 0, "right": 0, "leak": 0,
                                                  "stale_after_restatement": 0,
                                                  "questions_after": 0, "right_after": 0}
                                              for c in CONFIGS}
    behaviour: dict[str, Any] = {}
    eps_checks: dict[str, Any] = {}
    for ticker in TICKERS:
        facts = load_facts(ticker, "revenue", snapshot)
        probes = probes_for(ticker, facts)
        for config in CONFIGS:
            s = score_config(facts, probes, config, as_of_now)
            agg = per_config[config]
            for key in ("questions", "right", "leak", "late", "stale_after_restatement"):
                agg[key] += s[key]
            agg["rows"].extend(s["rows"])
            r = score_restatements(facts, config, as_of_now)
            for key, value in r.items():
                restatement[config][key] += value
        restated = restated_periods(facts)
        vt = vibe["tickers"].get(ticker, {})
        ob = openbb["tickers"].get(ticker, {})
        behaviour[ticker] = {
            "restated_quarters": len(restated),
            "vibe_pit_keeps": vibe_keeps(vt.get("pit", {}).get("series", {}).get("revenue", []),
                                         restated),
            "vibe_research_keeps": vibe_keeps(
                vt.get("research", {}).get("series", {}).get("revenue", []), restated),
            "vibe_pit_filed_date_alignment": vibe_date_leaks(
                vt.get("pit", {}).get("series", {}).get("revenue", []), facts),
            "openbb_pit_keeps": openbb_keeps(ob.get("pit", {}).get("rows", []), restated),
            "openbb_default_keeps": openbb_keeps(ob.get("default", {}).get("rows", []), restated),
            "langalpha_takes_a_date": False,
            "langalpha_periods_returned": sorted(
                (langalpha["tickers"].get(ticker, {}).get("response", {}).get("data", {})
                 .get("Total Revenue") or {}).keys()),
        }
        eps_checks[ticker] = openbb_q4_eps(ob.get("default", {}).get("rows", []))
    for agg in per_config.values():
        n = agg["questions"]
        agg["right_rate"] = round(agg["right"] / n, 4) if n else None
    outputs_all = {"vibe": vibe, "openbb": openbb, "langalpha": langalpha,
                   "openalice": openalice}
    direct: dict[str, dict[str, int]] = {r: {} for r in RIVALS}
    direct_restated: dict[str, dict[str, int]] = {r: {} for r in RIVALS}
    rows_rivals: dict[str, list[dict[str, Any]]] = {r: [] for r in RIVALS}
    for ticker in TICKERS:
        facts = load_facts(ticker, "revenue", snapshot)
        probes = probes_for(ticker, facts)
        for rival in RIVALS:
            for key, value in score_rival(rival, ticker, facts, probes, outputs_all).items():
                direct[rival][key] = direct[rival].get(key, 0) + value
            rows_rivals[rival].extend(rival_outcomes(rival, ticker, facts, probes, outputs_all))
            for key, value in score_rival_restatements(rival, ticker, facts, outputs_all).items():
                direct_restated[rival][key] = direct_restated[rival].get(key, 0) + value
    argus_rows = per_config["argus"]["rows"]
    held_argus = [r for r in argus_rows if r["ticker"] != DESIGN_TICKER]
    significance = {
        "test": "exact two-sided McNemar on the questions both systems answered",
        "argus_right_rate_ci95": wilson(sum(bool(r["right"]) for r in argus_rows),
                                        len(argus_rows)),
        "paired": {rival: paired(argus_rows, rows_rivals[rival]) for rival in RIVALS},
    }
    held_out = {
        "design_ticker": DESIGN_TICKER,
        "held_out_tickers": [t for t in TICKERS if t != DESIGN_TICKER],
        "argus": {"n": len(held_argus), "right": sum(bool(r["right"]) for r in held_argus),
                  "leak": sum(bool(r["leak"]) for r in held_argus)},
        "paired": {rival: paired(held_argus, [r for r in rows_rivals[rival]
                                              if r["ticker"] != DESIGN_TICKER])
                   for rival in RIVALS},
    }
    return {
        "snapshot": {"tickers": list(TICKERS), "from_year": FROM_YEAR,
                     "probe_offset_hours": OFFSET.total_seconds() / 3600},
        "configurations": {c: {"gate": g, "supersede": s} for c, (g, s) in CONFIGS.items()},
        "rivals_from_their_own_outputs": {"latest_quarter": direct,
                                          "restated_quarter_after": direct_restated},
        "argus": {"latest_quarter": {k: v for k, v in per_config["argus"].items() if k != "rows"},
                  "restated_quarter": restatement["argus"]},
        "mechanism_ablation": {
            "note": "ARGUS's resolver run with each rival's gate and supersede rule, to show "
                    "which mechanism causes which error; the rivals' own scores are above",
            "latest_quarter": {c: {k: v for k, v in s.items() if k != "rows"}
                               for c, s in per_config.items()},
            "restated_quarter": restatement},
        "significance": significance,
        "held_out": held_out,
        "rows_argus": per_config["argus"]["rows"],
        "rows_rivals": rows_rivals,
        "rival_behaviour_as_their_outputs_show_it": behaviour,
        "openbb_q4_eps": eps_checks,
        "rival_runs": {"vibe": vibe.get("clone"), "openbb": openbb.get("openbb_sec_dir"),
                       "langalpha_fetched_at": langalpha.get("fetched_at"),
                       "yfinance": langalpha.get("yfinance_version"),
                       "openalice_fetched_at": openalice.get("fetched_at")},
        "not_scored": {
            "lumen_terminal": "its only fundamentals path (src/adapters/equity.ts:405-560, "
                              "EquityFundamentalsAdapter) reads Yahoo quoteSummary's "
                              "financialData.totalRevenue, a trailing-twelve-month figure with "
                              "no period or date, so it cannot answer a quarter's revenue at "
                              "any cutoff; read, not run on these questions"},
    }


def main(argv: Sequence[str] | None = None) -> int:
    del argv
    report = run()
    ARTEFACT.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    a = report["argus"]
    print(f"{'argus':24s} latest {a['latest_quarter']['right']}/{a['latest_quarter']['questions']}"
          f" leak {a['latest_quarter']['leak']} late {a['latest_quarter']['late']} | restated "
          f"(after) {a['restated_quarter']['right_after']}/"
          f"{a['restated_quarter']['questions_after']}")
    for rival in RIVALS:
        s = report["rivals_from_their_own_outputs"]["latest_quarter"][rival]
        r = report["rivals_from_their_own_outputs"]["restated_quarter_after"][rival]
        print(f"{rival:24s} latest {s['right']}/{s['questions']} leak {s['leak']} late "
              f"{s['late']} stale {s['stale_after_restatement']} | restated {r['right']}/"
              f"{r['questions']} stale {r['stale_after_restatement']} other {r['neither_value']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
