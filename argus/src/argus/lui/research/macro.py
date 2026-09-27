"""Macro answers: FRED series, CPI and PCE releases, rate sensitivity, through bitget-signal's
macro Skill first."""

from __future__ import annotations

import itertools
import json
import re
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from argus.desk.portfolio import (
    beta,
    correlation,
)
from argus.lui.answer import Source, unlead
from argus.lui.research.kinds import (
    BENCHMARK,
    _t,
)
from argus.lui.research.parse import (
    _DOLLAR_FOCUS,
)
from argus.lui.research.riskmath import (
    FRED_MONTHLY,
    FRED_SERIES,
    _event_lines,
)
from argus.lui.skillroute import Routed
from argus.lui.skillroute import route as skill_route
from argus.lui.trace import trace_module
from argus.market.skills import Health
from argus.truth import http
from argus.truth.coverage import ContextPool
from argus.truth.paths import DATA_DIR

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}&cosd={start}"


FRED_SNAPSHOT = DATA_DIR / "macro_snapshot.json"
"""FRED series as last read by the desk cycle, shipped with the console. FRED did not answer from
the hosted console's network on 2026-09-24 (three macro questions timed out after about 20s), so
the console reads FRED live when it can within :data:`FRED_TIMEOUT_S` and otherwise answers from
this file, dated, and says which it used."""


FRED_TIMEOUT_S = 5.0


FRED_BACKOFF_S = 600.0


_FRED_DOWN_UNTIL = 0.0


_FRED_USED_SNAPSHOT: dict[str, str] = {}


def _fred_live(series: str, days: int) -> list[tuple[str, float]]:
    from datetime import timedelta as _td

    start = (datetime.now(UTC) - _td(days=days)).date().isoformat()
    text = http.fetch_text(FRED_CSV.format(series=series, start=start), timeout=FRED_TIMEOUT_S)
    rows: list[tuple[str, float]] = []
    for line in text.splitlines()[1:]:
        day, _, value = line.partition(",")
        try:
            rows.append((day, float(value)))
        except ValueError:
            continue
    return rows


def _fred(series: str, days: int = 45) -> list[tuple[str, float]]:
    """One FRED series as (date, value) rows, oldest first, missing days ('.') dropped — live when
    FRED answers, else from :data:`FRED_SNAPSHOT` (recorded in ``_FRED_USED_SNAPSHOT``)."""
    from datetime import timedelta as _td

    global _FRED_DOWN_UNTIL
    if time.monotonic() >= _FRED_DOWN_UNTIL:
        try:
            rows = _fred_live(series, days)
            if rows:
                return rows
        except Exception:
            # Once FRED has failed, stop waiting on it for a while: every series in one answer
            # would otherwise sit out its own timeout (15-19s per macro answer, measured live).
            _FRED_DOWN_UNTIL = time.monotonic() + FRED_BACKOFF_S
    if not FRED_SNAPSHOT.exists():
        return []
    snap = json.loads(FRED_SNAPSHOT.read_text(encoding="utf-8"))
    cutoff = (datetime.now(UTC) - _td(days=days)).date().isoformat()
    rows = [(d, float(v)) for d, v in snap.get("series", {}).get(series, []) if d >= cutoff]
    if not rows:  # an old snapshot still beats nothing; keep its tail
        rows = [(d, float(v)) for d, v in snap.get("series", {}).get(series, [])][-days:]
    if rows:
        _FRED_USED_SNAPSHOT[series] = str(snap.get("generated_at", ""))[:10]
    return rows


def write_macro_snapshot(days: int = 120) -> int:
    """Read every series the console uses from FRED and write them to :data:`FRED_SNAPSHOT`.
    Run by the desk cycle, so the hosted console's fallback is never more than a cycle old."""
    series = {sid: _fred_live(sid, days) for sid in FRED_SERIES}
    series.update({sid: _fred_live(sid, 480) for sid in FRED_MONTHLY})
    FRED_SNAPSHOT.write_text(json.dumps({
        "generated_at": datetime.now(UTC).isoformat(),
        "source": "https://fred.stlouisfed.org (FRED, Federal Reserve Bank of St. Louis)",
        "series": series,
    }, indent=1) + "\n", encoding="utf-8", newline="\n")
    return sum(len(v) for v in series.values())


def _book_rate_lines(book: Mapping[str, float], dollar_first: bool) -> tuple[list[str], str | None,
                                                                            dict[str, Any]]:
    """Each holding's measured sensitivity to the 10-year yield and the dollar, weighted into the
    book's. The same regression the single-name line uses, so the book figure is the sum of the
    lines a reader can check one by one."""
    with ContextPool(max_workers=max(1, len(book))) as pool:
        jobs = {s: pool.submit(_rate_sensitivity, s) for s in book}
        per: dict[str, dict[str, Any]] = {}
        for s, job in jobs.items():
            try:
                found = job.result()
            except Exception:
                found = None
            if found is not None:
                per[s] = found
    if not per:
        return [], None, {}
    covered = sum(w for s, w in book.items() if s in per)
    rate = sum(book[s] * per[s]["pct_per_10bp"] for s in per)
    dollar_parts = {s: per[s]["corr_dollar"] for s in per if per[s].get("corr_dollar") is not None}
    lines = [f"{_t(s)} ({book[s]:.0%}): {per[s]['pct_per_10bp']:+.2f}% per +10bp in the 10-year, "
             f"correlation {per[s]['corr_10y']:+.2f}"
             + (f"; {per[s]['corr_dollar']:+.2f} with the dollar" if s in dollar_parts else "")
             for s in sorted(per, key=lambda s: -abs(book[s] * per[s]["pct_per_10bp"]))]
    driver = max(per, key=lambda s: abs(book[s] * per[s]["pct_per_10bp"]))
    missing = [s for s in book if s not in per]
    if missing:
        lines.append(f"Not measured (too little shared history): "
                     f"{', '.join(_t(s) for s in missing)} — the book figure covers {covered:.0%} "
                     f"of it.")
    if dollar_first and dollar_parts:
        weighted = sum(book[s] * c for s, c in dollar_parts.items())
        relation = ("with" if weighted > 0.1 else "against" if weighted < -0.1
                    else "barely with")
        head = (f"Bottom line: your book has moved {relation} the dollar — a weighted "
                f"correlation of {weighted:+.2f} over the last three months — so a stronger "
                f"dollar has "
                + ("helped it" if weighted > 0.1 else "hurt it" if weighted < -0.1 else
                   "not been what moves it")
                + f"; {_t(driver)} carries the most rate exposure.")
    else:
        cut = -rate * 1.0  # a 10bp fall in the 10-year
        head = (f"Bottom line: your book has moved about {rate:+.2f}% for each +10bp in the "
                f"10-year "
                f"(weighted from each holding's last three months), so if a Fed cut took the "
                f"10-year down 10bp the measured relationship says about {cut:+.2f}% — "
                f"{_t(driver)} is the biggest part of it. The 10-year does not have to follow the "
                f"Fed; this is sensitivity, not a forecast.")
    return lines, head, {"per_symbol": per, "book_pct_per_10bp": rate, "covered": covered}


def _fred_is_the_mirror(ident: str, args: dict[str, Any]
                        ) -> tuple[Health, str, Any, str, bool]:
    """The mirror for bitget-signal's rates_yields in the macro answer: FRED, fetched by the
    answer itself, so the route only records which of the two the figures stand on."""
    del ident, args
    return Health.OK, "read by the macro answer", None, "FRED (St. Louis Fed)", True


def _macro(symbol: str | None, book: Mapping[str, float] | None = None,
           raw_text: str = "") -> tuple[list[str], list[Source], dict[str, Any]]:
    """Rates, the Fed, inflation and the dollar from FRED, and how ``symbol`` (QQQ when none) — or
    the whole ``book`` when one is held — has traded against the 10-year yield and the dollar."""
    from argus.market.evidence import RSS_FEEDS, RssSource

    target = symbol or BENCHMARK
    with ContextPool(max_workers=len(FRED_SERIES) + 3) as pool:
        jobs = {sid: pool.submit(_fred, sid) for sid in FRED_SERIES}
        fed_job = pool.submit(lambda: RssSource().headlines("fed", RSS_FEEDS["fed"][0]))
        # bitget-signal's macro-analyst Skill is asked for the curve first; its tool names FRED,
        # which is read beside it, so the receipt says which of the two the figures came from.
        curve_job = pool.submit(skill_route, "rates_yields", "yield_curve", None, wait=4.0,
                                mirror_call=_fred_is_the_mirror)
        series = {}
        for sid, job in jobs.items():
            try:
                series[sid] = job.result()
            except Exception:
                series[sid] = []
        try:
            fed = fed_job.result()
        except Exception:
            fed = []
    lines: list[str] = []
    readings: dict[str, Any] = {}
    for sid, label in FRED_SERIES.items():
        rows = series.get(sid) or []
        if not rows:
            continue
        day, last = rows[-1]
        first = rows[0][1]
        unit = "" if sid == "DTWEXBGS" else "%"
        change = last - first
        change_text = (f"{change:+.2f}" if sid == "DTWEXBGS" else f"{change * 100:+.0f}bp")
        readings[sid] = {"date": day, "value": last, "change_45d": change}
        lines.append(f"{label}: {last:.2f}{unit} on {day} ({change_text} over the last "
                     f"{len(rows)} readings, about six weeks).")
    ten = readings.get("DGS10", {}).get("value")
    two = readings.get("DGS2", {}).get("value")
    breakeven = readings.get("T10YIE", {}).get("value")
    if ten is not None and two is not None:
        spread = (ten - two) * 100
        lines.append(f"Curve: 10-year minus 2-year is {spread:+.0f}bp — "
                     + ("inverted." if spread < 0 else "positively sloped."))
    if ten is not None and breakeven is not None:
        lines.append(f"Real 10-year yield (nominal less breakeven inflation): about "
                     f"{ten - breakeven:.2f}%.")
    dollar_first = bool(_DOLLAR_FOCUS.search(raw_text))
    book_head: str | None = None
    if book and len(book) > 1:
        book_lines, book_head, book_readings = _book_rate_lines(book, dollar_first)
        lines.extend(book_lines)
        if book_readings:
            readings["book"] = book_readings
    try:
        sensitivity = None if book_head else _rate_sensitivity(target)
    except Exception:
        sensitivity = None  # the backdrop still stands without the co-movement line
    if sensitivity is not None:
        readings["sensitivity"] = sensitivity
        name = "tech (QQQ)" if target == BENCHMARK else _t(target)
        rho = sensitivity["corr_10y"]
        strength = ("strongly" if abs(rho) >= 0.5 else "moderately" if abs(rho) >= 0.25
                    else "barely")
        text = (f"{name} has moved {strength} with rates over {sensitivity['days']} trading days — "
                f"correlation {rho:+.2f} between its daily return and the daily change in the "
                f"10-year yield, about {sensitivity['pct_per_10bp']:+.2f}% for each +10bp")
        if sensitivity.get("corr_dollar") is not None:
            usd = sensitivity["corr_dollar"]
            text += (f"; {usd:+.2f} with the broad dollar index, so a stronger dollar has "
                     + ("helped" if usd > 0.1 else "hurt" if usd < -0.1 else "barely moved")
                     + f" {name}")
        lines.append(text + ". Correlation, not a cause.")
    policy = re.compile(r"\b(?:FOMC|monetary\s+policy|federal\s+funds|minutes|statement|"
                        r"Powell|rate|speech|testimony|economic\s+projections)\b", re.I)
    fed = [h for h in fed if policy.search(h.title)]
    latest_fed = sorted(fed, key=lambda h: h.published, reverse=True)[:2]
    for h in latest_fed:
        lines.append(f"Federal Reserve, {h.published:%d %b}: {h.title} {h.link}")
    if not readings:
        return [], [], {}
    if book_head is not None:
        lines.insert(0, book_head)
    elif (dollar_first and sensitivity is not None
          and sensitivity.get("corr_dollar") is not None):
        usd = sensitivity["corr_dollar"]
        name = "tech (QQQ)" if target == BENCHMARK else _t(target)
        lines.insert(0, (
            f"Bottom line: {name} has moved "
            + ("with" if usd > 0.1 else "against" if usd < -0.1 else "independently of")
            + f" the dollar (correlation {usd:+.2f} over {sensitivity['days']} trading days), so "
            + ("a stronger dollar has come with a stronger " + name if usd > 0.1 else
               "a stronger dollar has come with a weaker " + name if usd < -0.1 else
               "the dollar has not been what moves " + name)
            + " — the lines below give the rates side."))
    elif ten is not None:
        head = (f"Bottom line: the 10-year is {ten:.2f}%"
                + (f" and the curve {'inverted' if two is not None and ten < two else 'upward'}"
                   if two is not None else ""))
        if sensitivity is not None:
            rho = sensitivity["corr_10y"]
            head += (" — and rates have been driving it: rising yields have come with falling "
                     "prices, so size it against the rates calendar." if rho <= -0.25 else
                     " — and it has been rising with yields, trading on growth rather than "
                     "rates." if rho >= 0.25 else
                     " — but it has not been trading on rates; the backdrop is context, not the "
                     "driver.")
        else:
            head += "."
        lines.insert(0, head)
    lines.extend(_event_lines(raw_text, always=True)[0])
    try:
        from argus.market.bitget_positioning import us_stock_brief

        brief = us_stock_brief()
    except Exception:
        brief = None
    if brief:
        lines.append(brief)
    if _FRED_USED_SNAPSHOT:
        dated = sorted(set(_FRED_USED_SNAPSHOT.values()))
        # Each series line says it is the shipped reading, so it is not read — or labelled — as
        # this minute's (the audit's round 3 found the fallback figures tagged live)
        names = tuple(form for name in FRED_SERIES.values()
                      for form in (f"{name}:", f"{name[0].upper()}{name[1:]}:"))
        lines = [line.replace(" on 20", " (the shipped reading) on 20", 1)
                 if line.startswith(names) else line for line in lines]
        lines.append(f"FRED did not answer from here just now, so these series are the desk "
                     f"cycle's last reading ({', '.join(dated)}), not this minute's.")
        _FRED_USED_SNAPSHOT.clear()
    macro_sources = [Source(kind="venue", ref="FRED (St. Louis Fed) + Bitget TLTUSDT/EURUSDUSDT",
                            detail="FRED series DGS10, DGS2, DFF, T10YIE, DTWEXBGS; Bitget hourly "
                                   "candles; Federal Reserve press feed, live")]
    try:
        curve = curve_job.result(timeout=8.0)
    except Exception:
        curve = None
    if curve is not None and curve.via == "skill":
        macro_sources.append(curve.source())
        lines.append("bitget-signal's macro-analyst Skill answered the yield curve as well "
                     "(rates_yields); the figures above are FRED's, the source that tool names.")
    elif curve is not None:
        macro_sources[0] = Source(kind="venue", ref=macro_sources[0].ref,
                                  detail=macro_sources[0].detail + "; FRED is the source "
                                  "bitget-signal's rates_yields names, read because the Skill "
                                  + curve.skill_said)
    if brief:
        macro_sources.append(Source(kind="venue", ref="bitget-mcp-server news_label_search",
                                    detail="Bitget UEX Daily, the latest US-stock brief"))
    if _CPI_Q.search(raw_text) or _PCE_Q.search(raw_text):
        cpi, cpi_sources = _cpi_lines("PCE" if _PCE_Q.search(raw_text) else "CPI")
        if cpi:
            lines = [cpi[0], *(unlead(x)
                               for x in lines), *cpi[1:]]
            macro_sources.extend(cpi_sources)
    return lines, macro_sources, readings


_PCE_Q = re.compile(r"\bpce\b|personal\s+consumption", re.I)


_CPI_Q = re.compile(r"\bcpi\b|\binflation\s+(?:print|number|data|rate|reading|looking|now|today)\b|"
                    r"\bhow\s+(?:high|hot)\s+is\s+inflation|\binflation\s*\??\s*$|"
                    r"\bconsumer\s+prices?\b|通胀|通脹|消费者物价|物価", re.I)


def _cpi_lines(kind: str = "CPI") -> tuple[list[str], list[Source]]:
    """The latest CPI print from FRED (all items and core, year on year and on the month) and how
    QQQ has reacted on CPI days in the desk's own event study. "what's the latest CPI print and how
    did stocks react" was answered without a CPI figure (2026-09-25 audit): the macro answer carried
    rates and breakevens only."""
    from argus.lui.answer import desk_notes_path

    def yoy(series: str) -> tuple[str, float, float | None] | None:
        # Matched by date, not by position: FRED leaves a month blank when a release is missed
        # (October 2025 is empty in CPIAUCSL), so "13 rows back" is not always a year back.
        try:
            rows = dict(_fred(series, 480))
        except Exception:
            return None
        if not rows:
            return None
        day = max(rows)
        year, month = int(day[:4]), int(day[5:7])
        year_ago = f"{year - 1}-{month:02d}-01"
        prior = f"{year if month > 1 else year - 1}-{(month - 2) % 12 + 1:02d}-01"
        if year_ago not in rows:
            return None
        last = rows[day]
        on_month = (last / rows[prior] - 1) * 100 if prior in rows else None
        return day, (last / rows[year_ago] - 1) * 100, on_month

    pce = kind == "PCE"
    # bitget-signal's macro-analyst Skill is asked for the same release beside FRED, which its
    # macro_indicators tool names as its source; the receipt says which one the figures stand on
    # (audit finding 109).
    with ContextPool(max_workers=1) as pool:
        skill_job = pool.submit(skill_route, "macro_indicators", "latest_release",
                                {"indicator": "core_pce" if pce else "cpi"}, wait=4.0,
                                mirror_call=_fred_is_the_mirror)
        headline, core = (yoy("PCEPI"), yoy("PCEPILFE")) if pce else (yoy("CPIAUCSL"),
                                                                      yoy("CPILFESL"))
        try:
            routed: Routed | None = skill_job.result(timeout=8.0)
        except Exception:
            routed = None
    if headline is None:
        return [], []
    shipped = {k: v for k, v in _FRED_USED_SNAPSHOT.items() if k in FRED_MONTHLY}

    def pair(reading: tuple[str, float, float | None]) -> str:
        month_part = (f" and {reading[2]:+.1f}% on the month" if reading[2] is not None else
                      " (the month before was not published, so no monthly change)")
        return f"{reading[1]:+.1f}% on the year{month_part}"

    month = datetime.fromisoformat(headline[0]).strftime("%b %Y")
    text = (f"Bottom line: US {'PCE inflation' if pce else 'CPI'} for {month}: "
            f"{pair(headline)}"
            + (f"; core, excluding food and energy, {pair(core)}" if core else "") + "."
            + (f" (FRED's last reading shipped with the console on "
               f"{sorted(set(shipped.values()))[-1]}; FRED did not answer just now.)"
               if shipped else ""))
    lines = [text]
    skill_sources: list[Source] = []
    if routed is not None and routed.via == "skill":
        skill_sources.append(routed.source())
        lines.append("bitget-signal's macro-analyst Skill answered the same release "
                     "(macro_indicators); the figures above are FRED's, the source that tool "
                     "names.")
    why = ("" if routed is None or routed.via == "skill" else
           f"; FRED is the source bitget-signal's macro_indicators names, read because the "
           f"Skill {routed.skill_said}")
    if pce:
        return lines, [Source(kind="venue", ref="FRED PCEPI, PCEPILFE",
                              detail=f"BEA personal consumption expenditures price index, "
                                     f"{headline[0]}{why}"), *skill_sources]
    try:
        study = json.loads((desk_notes_path().parent / "event_reactions.json").read_text("utf-8"))
        qqq = next(r for r in study["reactions"] if r.get("symbol") == "QQQUSDT"
                   and r.get("kind") == "CPI")
        lines.append(f"How stocks reacted: on QQQ's last {qqq['events']} CPI days the average "
                     f"abnormal move was {qqq['average_car_bps']:+.1f}bps — "
                     f"{str(qqq['verdict']).split('.')[0].lower()}. Ask \"how does QQQ react to "
                     f"CPI\" for every release and test.")
    except (OSError, ValueError, KeyError, StopIteration, TypeError):
        pass
    return lines, [Source(kind="venue", ref="FRED CPIAUCSL, CPILFESL",
                          detail=f"BLS consumer price index, {headline[0]}{why}"),
                   *skill_sources]


def _rate_sensitivity(symbol: str, days: int = 90) -> dict[str, Any] | None:
    """How ``symbol``'s daily return has co-moved with the daily change in the 10-year yield (and
    the broad dollar), on dates both series report. Bitget daily candles against FRED's DGS10 and
    DTWEXBGS; the standard equity-rates sensitivity, stated as % per +10bp."""
    from argus.market.history import CandleType, fetch

    # The price at the US close (4pm New York, 20:00 UTC in daylight time): the 4h candle that
    # opens at 16:00 UTC closes there, on the same clock as the Treasury's daily yield. Bitget's own
    # daily candles open at 16:00 UTC, so a "daily" return straddles two US sessions — measured on
    # QQQ against the 10-year: correlation 0.00 same-day and -0.32 at a one-day lag, an artefact of
    # the boundary, not a finding.
    bars = fetch(symbol, interval="4H", candle_type=CandleType.MARKET, recent=True,
                 limit=min(1000, days * 6))
    closes = {c.ts.date().isoformat(): float(c.close) for c in bars if c.ts.hour == 16}
    ordered = sorted(closes)
    rets = {d: closes[d] / closes[p] - 1 for p, d in itertools.pairwise(ordered)
            if closes[p] > 0}

    def changes(series: str) -> dict[str, float]:
        rows = _fred(series, days=days + 10)
        return {d: v - prev for (_, prev), (d, v) in itertools.pairwise(rows)}

    ten = changes("DGS10")
    common = sorted(set(rets) & set(ten))
    if len(common) < 20:
        return None
    r = [rets[d] * 100 for d in common]
    y = [ten[d] * 100 for d in common]  # basis points
    rho = correlation(r, y)
    slope = beta(r, y)
    if rho is None or slope is None:
        return None
    out: dict[str, Any] = {"days": len(common), "corr_10y": rho, "pct_per_10bp": slope * 10}
    try:
        dollar = changes("DTWEXBGS")
        both = sorted(set(rets) & set(dollar))
        if len(both) >= 20:
            out["corr_dollar"] = correlation([rets[d] for d in both], [dollar[d] for d in both])
    except Exception:
        pass
    return out


# Every engine here is a traced step from import on (lui/trace.py, trace_module).
trace_module(globals())
