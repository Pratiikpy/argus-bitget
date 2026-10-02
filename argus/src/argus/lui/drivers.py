"""A thesis's demand driver, tested against what the companies filed.

"I think NVDA runs on AI capex through 2027 — test my thesis" was answered with the next day's
price base rates (the hosted model's reading) or with the desk's own decision on NVDA (the
patterns'), and the claim itself — that the company's business rides its customers' capital
spending — was never looked at (judge audit, 2026-09-30). The thesis tester (`lui/thesis.py`) had
no kind for it: "capex" matched none of its patterns, so the reason was not even listed.

What is measured, all from SEC XBRL (``data.sec.gov`` companyconcept, keyless):

* **the company's revenue**, quarter by quarter, and its growth on the same quarter a year before;
* **the spending named as the driver**: for AI or data-centre capex, the capital expenditure of the
  four largest US cloud builders — Microsoft, Alphabet, Amazon and Meta — summed quarter by
  quarter. That is a definition, stated on the answer, not a claim about who the company sells to;
* **how large the company is beside that spending**: its revenue as a share of the same quarter's
  capex, now and a year and two years before — a scale, not evidence that one drives the other.

Cash-flow lines are filed year-to-date in 10-Qs (six months, nine months) and as a whole year in
the 10-K, so a quarter is the difference of two consecutive year-to-date figures with the same
start; income-statement lines are usually filed as three-month figures and are used as filed.

The verdict is about **today**: whether the driver is still growing and the company with it. A
claim that runs forward ("through 2027") cannot be tested from filings, and the answer says which
part was tested, which was not, and what would break it.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date
from itertools import pairwise
from typing import Any

from argus.lui.trace import trace_module

AI_CAPEX = re.compile(
    r"\bcapex\b|capital\s+(?:spend\w*|expenditure\w*)|data[\s-]*cent(?:er|re)s?|hyperscal\w*|"
    r"\bai\s+(?:spend\w*|demand|build[\s-]*out|boom|infrastructure|investment|chips?|servers?)|"
    r"cloud\s+(?:spend\w*|capex|build\w*)", re.I)
"""A driver that names AI or cloud capital spending."""

BUILDERS: tuple[tuple[str, str], ...] = (("MSFT", "Microsoft"), ("GOOGL", "Alphabet"),
                                         ("AMZN", "Amazon"), ("META", "Meta"))
"""The spending measured for "AI capex": the four largest US cloud builders' own capex."""

REVENUE_TAGS = ("RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues",
                "SalesRevenueNet")
CAPEX_TAGS = ("PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets")
QUARTER_DAYS = (80, 100)
YEAR_APART_DAYS = (350, 380)
ALIGN_DAYS = 45
"""Fiscal quarters end on different days (NVIDIA's in late January, April, July and October); two
quarters are the same calendar quarter when they end within 45 days of each other."""


@dataclass(frozen=True)
class Quarter:
    end: date
    value: float


def quarters(rows: Sequence[Any]) -> list[Quarter]:
    """Three-month values from companyconcept rows, oldest first: as filed where a row spans a
    quarter, otherwise the difference of two year-to-date rows sharing a start date."""
    latest: dict[tuple[date | None, date], Any] = {}
    for row in rows:
        if row.start is None or row.unit != "USD":
            continue
        key = (row.start, row.end)
        if key not in latest or row.filed > latest[key].filed:
            latest[key] = row
    found: dict[date, float] = {}
    by_start: dict[date, list[Any]] = {}
    for (start, _end), row in latest.items():
        assert start is not None
        by_start.setdefault(start, []).append(row)
        if QUARTER_DAYS[0] <= (row.end - start).days <= QUARTER_DAYS[1]:
            found[row.end] = row.value
    for runs in by_start.values():
        runs.sort(key=lambda r: r.end)
        for earlier, later in pairwise(runs):
            if (QUARTER_DAYS[0] <= (later.end - earlier.end).days <= QUARTER_DAYS[1]
                    and later.end not in found):
                found[later.end] = later.value - earlier.value
    return [Quarter(end, value) for end, value in sorted(found.items())]


def growth(series: Sequence[Quarter]) -> list[tuple[date, float]]:
    """Each quarter's change on the quarter a year before it, where that quarter was filed."""
    out: list[tuple[date, float]] = []
    for q in series:
        before = next((p for p in series
                       if YEAR_APART_DAYS[0] <= (q.end - p.end).days <= YEAR_APART_DAYS[1]), None)
        if before is not None and before.value > 0:
            out.append((q.end, q.value / before.value - 1))
    return out


def _combined(per_builder: Mapping[str, Sequence[Quarter]]) -> list[Quarter]:
    """The builders' capex summed by calendar quarter, only where every builder filed it."""
    anchor = per_builder[BUILDERS[0][0]]
    out: list[Quarter] = []
    for q in anchor:
        total = q.value
        for ticker, _ in BUILDERS[1:]:
            match = next((p for p in per_builder[ticker]
                          if abs((p.end - q.end).days) <= ALIGN_DAYS), None)
            if match is None:
                break
            total += match.value
        else:
            out.append(Quarter(q.end, total))
    return out


def _scale(revenue: Sequence[Quarter], capex: Sequence[Quarter]) -> list[tuple[str, float]]:
    """The company's revenue as a share of the builders' capex in the same calendar quarter —
    latest, a year before and two years before, where both were filed.

    Not a correlation of the two growth rates: that was tried on NVIDIA's filings and swung from
    +0.56 over eight quarters to -0.82 over twelve, because the base effects of a company tripling
    dominate it. A share of the same quarter's spending is stable to read and says what it is."""
    out: list[tuple[str, float]] = []
    for years_back, label in ((0, "this quarter"), (1, "a year before"), (2, "two years before")):
        if len(revenue) <= 4 * years_back:
            break
        q = revenue[-1 - 4 * years_back]
        match = next((c for c in capex if abs((c.end - q.end).days) <= ALIGN_DAYS), None)
        if match is None or match.value <= 0:
            break
        out.append((label, q.value / match.value))
    return out


Rows = Callable[[int, str], Sequence[Any]]


def _series(cik: int, tags: Sequence[str], rows: Rows) -> tuple[list[Quarter], str]:
    """The tag, of those given, whose quarters run latest — companies change tags over time."""
    best: tuple[list[Quarter], str] = ([], tags[0])
    for tag in tags:
        found = quarters(rows(cik, tag))
        if found and (not best[0] or found[-1].end > best[0][-1].end):
            best = (found, tag)
    return best


def facts(ticker: str, reason: str, *, cik_for: Callable[[str], int | None] | None = None,
          rows: Rows | None = None) -> dict[str, Any]:
    """Everything the driver test reads, fetched in parallel; ``error`` names what did not
    answer."""
    if cik_for is None or rows is None:
        from argus.market.evidence import EdgarSource
        from argus.market.statement_facts import ConceptSource

        cik_for = cik_for or EdgarSource().cik_for
        rows = rows or ConceptSource().rows
    names = [ticker, *([t for t, _ in BUILDERS] if AI_CAPEX.search(reason) else [])]
    try:
        ciks = {name: cik_for(name) for name in names}
    except Exception as exc:  # the SEC's ticker map did not answer
        return {"error": f"the SEC's ticker map did not answer ({type(exc).__name__})"}
    fund = ciks.get(ticker) is None and len(names) > 1
    if ciks.get(ticker) is None and not fund:
        return {"error": f"{ticker} files no XBRL with the SEC under that ticker"}
    get = rows

    def one(name: str) -> tuple[str, tuple[list[Quarter], str]]:
        cik = ciks[name]
        if cik is None:
            return name, ([], "")
        # A fund (SMH) files no statements of its own; "AI capex keeps rising" is still a claim
        # about the builders' spending, which their own filings settle (round 18, row 616).
        tags = REVENUE_TAGS if name == ticker else CAPEX_TAGS
        return name, _series(cik, tags, get)

    with ThreadPoolExecutor(max_workers=len(names)) as pool:
        got = dict(pool.map(one, names))
    out: dict[str, Any] = {"ticker": ticker, "cik": ciks[ticker], "revenue": got[ticker][0],
                           "revenue_tag": got[ticker][1], "fund": fund}
    if len(names) > 1:
        out["builders"] = {name: got[name][0] for name, _ in BUILDERS}
        out["builder_tags"] = {name: got[name][1] for name, _ in BUILDERS}
        out["builder_ciks"] = {name: ciks[name] for name, _ in BUILDERS}
    return out


def _pct(x: float) -> str:
    return f"{x:+.0%}"


def _bn(x: float) -> str:
    return f"${x / 1e9:,.1f}bn"


FORWARD = re.compile(r"\b(?:through|until|till|by|into|in)\s+(?:20\d\d|next\s+year)\b|"
                     r"\b(?:for\s+(?:years|the\s+next)|long[\s-]+term|next\s+\d+\s+years)\b", re.I)


BEARISH = re.compile(r"\b(?:slow\w*|fall\w*|declin\w*|weak\w*|shrink\w*|peak\w*|drop\w*|"
                     r"fad\w*|roll\w*\s+over|cut\w*|decelerat\w*|dry\w*\s+up|"
                     r"(?:is|are)\s+over\b)", re.I)
"""A driver claimed to be turning down ("demand is slowing", "AI capex is peaking")."""


def _slope(trend: Sequence[float]) -> str:
    """Growth over the last quarters, read as a direction: five points either way, or steady."""
    if len(trend) >= 3 and trend[-1] > trend[0] + 0.05:
        return "accelerating"
    if len(trend) >= 3 and trend[-1] < trend[0] - 0.05:
        return "slowing"
    return "steady"


def test(reason: str, found: Mapping[str, Any]) -> tuple[str, str, list[tuple[str, str, str]]]:
    """``(result, line, evidence)``: result is ``supported``, ``contradicted``, ``not measurable``
    or ``not tested``; evidence is ``(text, source, url)`` triples."""
    from argus.market.statement_facts import CONCEPT_URL

    if found.get("error"):
        return "not tested", f"The filings could not be read just now: {found['error']}.", []
    ticker = str(found["ticker"])
    if found.get("fund"):
        return _spending_only(reason, ticker, found)
    revenue: list[Quarter] = list(found.get("revenue") or [])
    rev_growth = growth(revenue)
    if not rev_growth:
        return ("not tested", f"{ticker}'s quarterly revenue could not be read from its XBRL "
                              f"filings, so the driver could not be tested.", [])
    evidence: list[tuple[str, str, str]] = []
    last_end, last_g = rev_growth[-1]
    trend = [g for _, g in rev_growth[-4:]]
    slope = _slope(trend)
    rev_url = CONCEPT_URL.format(cik=int(found["cik"]), tag=found["revenue_tag"])
    evidence.append((f"{ticker} revenue, quarter to {last_end:%d %b %Y}: "
                     f"{_bn(revenue[-1].value)}, {_pct(last_g)} on the year; the last "
                     f"{len(trend)} quarters' growth " + ", ".join(_pct(g) for g in trend)
                     + f" ({slope}).", "SEC XBRL, company filings", rev_url))
    builders = found.get("builders")
    line = ""
    if builders is not None:
        if not all(builders.get(t) for t, _ in BUILDERS):
            missing = [n for t, n in BUILDERS if not builders.get(t)]
            return ("not tested", f"The capex of {', '.join(missing)} could not be read from "
                                  f"their filings just now, so the spending named as the driver "
                                  f"is not measured.", evidence)
        combined = _combined(builders)
        capex_growth = growth(combined)
        if not capex_growth:
            return ("not tested", "The cloud builders' quarters could not be lined up a year "
                                  "apart, so their capex growth is not measured.", evidence)
        cap_end, cap_g = capex_growth[-1]
        per = []
        for t, name in BUILDERS:
            g = growth(builders[t])
            if g:
                per.append(f"{name} {_pct(g[-1][1])}")
        evidence.append((f"Microsoft, Alphabet, Amazon and Meta together spent "
                         f"{_bn(combined[-1].value)} on capex in the quarter to about "
                         f"{cap_end:%b %Y}, {_pct(cap_g)} on the year (" + ", ".join(per) + ").",
                         "SEC XBRL, company filings",
                         CONCEPT_URL.format(cik=int(found["builder_ciks"]["MSFT"]),
                                            tag=found["builder_tags"]["MSFT"])))
        scale = _scale(revenue, combined)
        if scale:
            said = [f"{share:.0%}" + (" of their capex " if i == 0 else " ") + label
                    for i, (label, share) in enumerate(scale)]
            evidence.append((f"{ticker}'s quarterly revenue came to "
                             + (", ".join(said[:-1]) + " and " + said[-1] if len(said) > 1
                                else said[0])
                             + " — how large the company is beside the spending, not proof "
                               "that one "
                  "drives the other.", "computed from the two series above", ""))
        cap_trend = [g for _, g in capex_growth[-4:]]
        cap_slope = _slope(cap_trend)
        if BEARISH.search(reason):
            # "AI capex is peaking": the spending's growth, not its sign, is the claim.
            result = ("supported" if cap_g < 0 or cap_slope == "slowing" else
                      "contradicted" if cap_slope == "accelerating" else "not measurable")
            line = (f"The claim is that the spending is turning: the cloud builders' capex growth "
                    f"over the last {len(cap_trend)} quarters ran " + ", ".join(
                        _pct(g) for g in cap_trend) + f" ({cap_slope}), and {ticker}'s revenue "
                    f"growth {_pct(last_g)} ({slope}).")
        else:
            holds = last_g > 0 and cap_g > 0
            broken = last_g < 0 or cap_g < 0
            result = "supported" if holds else "contradicted" if broken else "not measurable"
            line = (f"Today it holds: the cloud builders' capex is {_pct(cap_g)} on the year and "
                    f"{ticker}'s revenue {_pct(last_g)} ({slope})." if holds else
                    f"Today it does not hold: the cloud builders' capex is {_pct(cap_g)} on the "
                    f"year and {ticker}'s revenue {_pct(last_g)}.")
    elif BEARISH.search(reason):
        result = ("supported" if last_g < 0 or slope == "slowing" else
                  "contradicted" if slope == "accelerating" else "not measurable")
        line = (f"The claim is that demand is turning: {ticker}'s revenue growth over the last "
                f"{len(trend)} quarters ran " + ", ".join(_pct(g) for g in trend)
                + f" ({slope}) — all of its revenue, not only the product or segment the claim may "
                  "name, which is not read here.")
    else:
        result = ("supported" if last_g > 0 else "contradicted" if last_g < 0
                  else "not measurable")
        line = (f"{ticker}'s revenue is {_pct(last_g)} on the year to {last_end:%b %Y} "
                f"({slope}) — the growth the driver implies is " +
                ("there" if last_g > 0 else "not there") + " — all of its revenue, not only the "
                "product or segment the claim may name, which is not read here.")
    if FORWARD.search(reason):
        line += (" What the filings cannot test is the part still ahead: they show the quarters "
                 "reported so far, and the claim runs past them. It breaks the quarter the "
                 "builders' capex growth turns negative" if builders is not None else
                 " What the filings cannot test is the part still ahead: it breaks the quarter "
                 "revenue growth turns negative")
        line += " — each report is the next check."
    return result, line, evidence


def _spending_only(reason: str, ticker: str,
                   found: Mapping[str, Any]) -> tuple[str, str, list[tuple[str, str, str]]]:
    """The driver claim on a name with no filings of its own, read from the spending alone."""
    from argus.market.statement_facts import CONCEPT_URL

    builders = found.get("builders") or {}
    if not all(builders.get(t) for t, _ in BUILDERS):
        missing = [n for t, n in BUILDERS if not builders.get(t)]
        return ("not tested", f"The capex of {', '.join(missing)} could not be read from their "
                              f"filings just now, so the spending named as the driver is not "
                              f"measured.", [])
    combined = _combined(builders)
    capex_growth = growth(combined)
    if not capex_growth:
        return ("not tested", "The cloud builders' quarters could not be lined up a year apart, "
                              "so their capex growth is not measured.", [])
    cap_end, cap_g = capex_growth[-1]
    cap_trend = [g for _, g in capex_growth[-4:]]
    cap_slope = _slope(cap_trend)
    per = [f"{name} {_pct(g[-1][1])}" for t, name in BUILDERS if (g := growth(builders[t]))]
    evidence = [(f"Microsoft, Alphabet, Amazon and Meta together spent {_bn(combined[-1].value)} "
                 f"on capex in the quarter to about {cap_end:%b %Y}, {_pct(cap_g)} on the year ("
                 + ", ".join(per) + ").", "SEC XBRL, company filings",
                 CONCEPT_URL.format(cik=int(found["builder_ciks"]["MSFT"]),
                                    tag=found["builder_tags"]["MSFT"]))]
    if BEARISH.search(reason):
        result = ("supported" if cap_g < 0 or cap_slope == "slowing" else
                  "contradicted" if cap_slope == "accelerating" else "not measurable")
    else:
        result = ("supported" if cap_g > 0 else "contradicted" if cap_g < 0
                  else "not measurable")
    line = (f"{ticker} is a fund and files no statements of its own, so the spending is read "
            f"directly: the cloud builders' capex growth over the last {len(cap_trend)} quarters "
            f"ran " + ", ".join(_pct(g) for g in cap_trend) + f" ({cap_slope}). Whether it "
            f"reaches {ticker}'s holdings' revenue is not read here — ask it of one of them "
            f"(NVDA, AMD, AVGO) to test that link too.")
    if FORWARD.search(reason):
        line += (" What the filings cannot test is the part still ahead: it breaks the quarter "
                 "the builders' capex growth turns negative — each report is the next check.")
    return result, line, evidence


trace_module(globals())
