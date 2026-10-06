"""Company research questions that came back with a sub-question dropped (round 45 judge, M6-M10).

Each answer below combines figures the console already reads (SEC XBRL company facts and inline
XBRL in `filing_figures`, `revenue_split`/`market/ixbrl`, the 8-K report dates in `earnings_move`,
Yahoo's quoteSummary in `fundamentals`); what was missing was the combination. Every part the
question asks is answered, or the line says which part could not be read and why.

- **M6, results plus reaction.** "What did Nvidia report for revenue and data center growth last
  quarter, and how did the stock react?" gave revenue alone. Now: total revenue and growth, the
  named segment's revenue and growth (the 10-Q's inline XBRL segment facts, this quarter against
  the same quarter a year earlier), and the stock's move on the first session after the release.
- **M7, free cash flow margin.** "Which of the two has the better free cash flow margin?" (after
  "NVDA vs AMD") named one company and no margin. Now: (operating cash flow - capital spending) /
  revenue for each company over the trailing four quarters, and which is higher. The two names come
  from the earlier question when this one does not name them.
- **M8, risk factors against the prior year.** "Summarize the main risk factors in Microsoft's
  latest 10-K, and flag anything new versus the prior year" dropped the second half silently.
  Now: Item 1A's headings in the latest 10-K, and those present only in it (new) or only in the
  prior year's (removed), from the two filings. A heading reworded between years can show in both
  lists; the answer says so. If the prior 10-K cannot be read, one line says that.
- **M9, an eight-quarter margin trend.** "Apple's gross margin and operating margin trends over the
  last 8 quarters" gave one year-over-year pair. Now: one point per quarter (quarters rebuilt from
  year-to-date XBRL facts where the filing gives a nine-month figure), the direction, and the
  latest against the first of the quarters shown.
- **M10, valuation.** "Compare Coinbase and Robinhood on revenue growth and valuation" gave growth
  only; "Is Tesla's P/E justified given its latest quarterly net income and revenue growth?" gave
  no quarter and no view. Now: trailing-four-quarter revenue growth, market cap, trailing and
  forward P/E, price/sales and EV/sales with the cheaper company on each; for a single stock, the
  growth the multiple assumes (trailing P/E over forward P/E, the analysts' next-year EPS growth)
  set against the growth delivered. No buy or sell call is made.

Entry point: :func:`lines`. It returns None for any question it does not own.
"""

from __future__ import annotations

import difflib
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from html.parser import HTMLParser
from typing import Any, Final

from argus.lui.trace import trace_module

LINE_WIDTH: Final = 100

_REACTION: Final = re.compile(
    r"\breact(?:ed|s|ion|ions)?\b|\b(?:stock|shares?|share\s+price|price)\s+(?:move[ds]?|jump(?:ed)?|"
    r"fell|fall|drop(?:ped)?|rose|rall(?:y|ied)|gain(?:ed)?|sank)\b|\bmarket\s+(?:response|"
    r"reaction)\b|\bwhat\s+did\s+the\s+(?:stock|shares?)\s+do\b", re.I)
_RESULTS: Final = re.compile(r"\b(?:report(?:ed|s)?|results?|earnings|release[sd]?|"
                             # "what did NVDA's latest 10-Q say about data center revenue" opened
                             # on a list of filing dates (round 45 visual, M4)
                             r"10-?q|10-?k|quarterly\s+filing)\b", re.I)
_REVENUE: Final = re.compile(r"\b(?:revenues?|sales|top[\s-]line)\b", re.I)
_FCF_MARGIN: Final = re.compile(
    r"\bfree\s+cash\s*flow\s+margins?\b|\bfcf\s+margins?\b|\bfree[\s-]cash[\s-]flow[\s-]margins?\b",
    re.I)
_MARGIN: Final = re.compile(r"\b(gross|operating|net|profit)\s+margins?\b", re.I)
_TREND: Final = re.compile(
    r"\btrends?\b|\b(?:over|across|for|in)\s+the\s+(?:last|past|previous)\s+(?:\w+\s+)?quarters\b|"
    r"\b(?:last|past|previous)\s+(?:\w+\s+)?quarters\b|\bquarter[\s-]by[\s-]quarter\b|"
    r"\bquarterly\s+(?:margins?|history|trend)\b|\bover\s+time\b|\bhistory\b", re.I)
_RISK: Final = re.compile(r"\brisk\s+factors?\b", re.I)
_DIFF: Final = re.compile(
    r"\bnew(?:ly)?\b|\bprior[\s-]year\b|\bprevious\s+(?:year|10-?k|filing)\b|\blast\s+year\b|"
    r"\bversus\b|\bvs\.?\b|\bcompared?\b|\bchang(?:e|ed|es)\b|\badded\b|\bremoved\b|\bdiffer\w*\b|"
    r"\byear[\s-]over[\s-]year\b|\bsince\b", re.I)
_TENK: Final = re.compile(r"\b10-?k\b|\bannual\s+report\b|\blatest\s+filing\b", re.I)
_VALUATION: Final = re.compile(
    r"\bvaluations?\b|\bvalued\b|\bp\s*/\s*e\b|\bpe\s+ratio\b|\bprice[\s-]to[\s-]earnings\b|"
    r"\bprice[\s-]to[\s-]sales\b|\bp\s*/\s*s\b|\bev\s*/\s*(?:sales|revenue)\b|\bmultiples?\b|"
    r"\bcheap(?:er)?\b|\bexpensive\b|\bmarket\s+cap\w*\b", re.I)
_JUSTIFIED: Final = re.compile(
    r"\bjustif\w+|\bwarrant\w*|\bdeserv\w+|\bover[\s-]?valued\b|\bunder[\s-]?valued\b|\btoo\s+"
    r"(?:high|expensive|rich)\b|\bpriced\s+in\b|\breasonable\b|\bfairly\s+valued\b|\bworth\b", re.I)
_COMPARE: Final = re.compile(r"\bcompar\w+|\bvs\.?\b|\bversus\b|\bwhich\b|\bcheaper\b|\bbetter\b|"
                             r"\band\b|\bor\b|\bagainst\b", re.I)
_GROWTH: Final = re.compile(r"\bgrowth\b|\bgrew\b|\bgrowing\b", re.I)
_PAIR_REFERENCE: Final = re.compile(
    r"\b(?:the\s+two|both|the\s+pair|of\s+them|between\s+them|those\s+two|these\s+two|the\s+other|"
    r"which\s+(?:one|of))\b", re.I)
_OPTIONS: Final = re.compile(r"\b(?:implied|expected|priced|options?)\b", re.I)
_WORD_COUNT: Final = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
                      "eight": 8, "nine": 9, "ten": 10, "twelve": 12}


def _short(text: str, width: int = 150) -> str:
    """A heading cut at a word, without its closing full stop, so a list of them reads cleanly."""
    text = text.rstrip(". ")
    return text if len(text) <= width else text[: width - 1].rsplit(" ", 1)[0] + "…"


def _x(value: float) -> str:
    """A multiple to one decimal: 350.7x, 7.9x."""
    return f"{value:.1f}x"


# ----------------------------------------------------------------------------- data readers
# Every network read is one small function so a test can replace it.


def _facts(ticker: str) -> dict[str, Any] | None:
    from argus.lui.research.filing_figures import company_facts

    return company_facts(ticker)


def _yahoo(ticker: str) -> dict[str, Any]:
    from argus.lui.research.fundamentals import yahoo_summary

    return yahoo_summary(ticker)


def _document(ticker: str) -> tuple[str, str, str, dict[str, str]] | None:
    from argus.market.ixbrl import latest_filing_document

    return latest_filing_document(ticker)


def _reports(ticker: str) -> list[tuple[date, bool]]:
    from argus.lui.research.earnings_move import report_history

    return report_history(ticker)


def _closes(ticker: str) -> list[tuple[date, float]]:
    from argus.market.equity_history import daily

    return [(r.day, float(r.close)) for r in daily(ticker)]


def _tenk_filings(ticker: str) -> list[tuple[date, date, str]]:
    """(filed, period end, document URL) of each 10-K on EDGAR, newest first."""
    from argus.market.evidence import EdgarSource

    edgar = EdgarSource()
    cik = edgar.cik_for(ticker)
    if cik is None:
        return []
    recent = edgar._get(edgar.SUBMISSIONS_URL.format(cik=cik))["filings"]["recent"]
    out: list[tuple[date, date, str]] = []
    for i, form in enumerate(recent.get("form", [])):
        if form != "10-K" or not recent["reportDate"][i]:
            continue
        accession = recent["accessionNumber"][i].replace("-", "")
        out.append((date.fromisoformat(recent["filingDate"][i]),
                    date.fromisoformat(recent["reportDate"][i]),
                    f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/"
                    f"{recent['primaryDocument'][i]}"))
    return out


def _fetch(url: str) -> str:
    from argus.market.evidence import FEED_USER_AGENT
    from argus.truth import http

    return http.fetch_text(url, timeout=120.0, headers={"User-Agent": FEED_USER_AGENT})


# ----------------------------------------------------------------------------- who is asked about


def _stocks(text: str) -> list[str]:
    """The US-listed tickers a text names, in the order named."""
    from argus.lui.research.parse import is_us_equity, research_symbols

    symbols, _ = research_symbols(text)
    seen: list[str] = []
    for symbol in symbols:
        if is_us_equity(symbol):
            ticker = symbol.removesuffix("USDT")
            if ticker not in seen:
                seen.append(ticker)
    return seen


def _tickers(text: str, prior: Sequence[str], limit: int = 4) -> list[str]:
    """The companies asked about: those the text names, else those of the latest earlier question
    that named two or more ("Which of the two ...", after "NVDA vs AMD"), else one it named."""
    own = _stocks(text)
    if own:
        return own[:limit]
    single: list[str] = []
    for earlier in reversed(prior):
        found = _stocks(earlier)
        if len(found) >= 2:
            return found[:limit]
        if found and not single:
            single = found
    return single[:limit]


# ----------------------------------------------------------------------------- shared arithmetic


def _load(ticker: str) -> dict[str, Any]:
    from argus.lui.research.filing_figures import _company

    return _company(ticker, (), lambda t: _facts(t))


def _recent_run(ends: Sequence[date]) -> list[date]:
    """The latest unbroken run of quarter ends, oldest first (a gap over 105 days is a missing
    quarter, and a run does not cross it)."""
    ordered = sorted(ends)
    run: list[date] = []
    for day in reversed(ordered):
        if run and not 75 <= (run[0] - day).days <= 105:
            break
        run.insert(0, day)
    return run


def _sum(series: dict[date, float], ends: Sequence[date]) -> float | None:
    if not ends or any(d not in series for d in ends):
        return None
    return float(sum(series[d] for d in ends))


def _trailing(c: dict[str, Any], key: str) -> tuple[float | None, list[date]]:
    """The last four quarters' sum of one series, and the four quarter ends."""
    run = _recent_run(list(c["series"]["revenue"]))
    ends = run[-4:]
    return (_sum(c["series"][key], ends) if len(ends) == 4 else None), ends


def _prior_four(c: dict[str, Any], ends: Sequence[date]) -> float | None:
    """The four quarters a year before ``ends`` (each within ten days of 364 days earlier)."""
    revenue: dict[date, float] = c["series"]["revenue"]
    total = 0.0
    for end in ends:
        match = next((d for d in revenue if abs((end - d).days - 364) <= 10), None)
        if match is None:
            return None
        total += revenue[match]
    return total


def _gap(now: float, then: float) -> str:
    """A change said so it cannot be misread across a sign (as filing_figures does)."""
    from argus.lui.research.filing_figures import _change

    return _change(now, then)


def _bn(x: float) -> str:
    from argus.lui.research.filing_figures import money

    return money(x)


def _errors(c: dict[str, Any]) -> str | None:
    return f"{c['ticker']}: {c['error']}." if "error" in c else None


# ----------------------------------------------------------------------------- M6: results + move


@dataclass(frozen=True)
class _Segment:
    member: str
    label: str
    axis: str
    now: float
    then: float | None
    end: date


def segments(document: str, names: dict[str, str] | None = None) -> list[_Segment]:
    """Revenue by single-dimension member for the document's latest quarter, each with the same
    quarter a year earlier where the filing tags it (a 10-Q carries both)."""
    from argus.market import ixbrl

    rows: dict[str, dict[date, float]] = {}
    axes: dict[str, str] = {}
    labels: dict[str, str] = {}
    for line in ixbrl.dimensioned_revenue(document, names):
        if not 80 <= (line.end - line.start).days <= 100:
            continue
        rows.setdefault(line.member, {})[line.end] = line.value
        axes[line.member] = line.axis
        labels[line.member] = line.label
    if not rows:
        return []
    latest = max(end for series in rows.values() for end in series)
    out: list[_Segment] = []
    for member, series in rows.items():
        if latest not in series or member == "us-gaap:OperatingSegmentsMember":
            continue
        before = next((d for d in series if abs((latest - d).days - 364) <= 10), None)
        out.append(_Segment(member, labels[member], axes[member], series[latest],
                            series[before] if before else None, latest))
    return sorted(out, key=lambda s: -s.now)


def _plain(text: str) -> str:
    return " " + re.sub(r"[^a-z0-9]+", " ", text.lower()).strip() + " "


_SEGMENT_ASK: Final = re.compile(
    r"\b(?:revenues?|sales)\s*(?:,|and|&)\s+(?P<seg>[a-z][a-z &-]{2,30}?)\s+(?:revenues?\s+)?"
    r"(?:growth|grew|rose|sales)\b|"
    # "what did NVDA's latest 10-Q say about data center revenue" (round 45 visual, M4)
    r"\b(?:about|on|for)\s+(?:its\s+|the\s+)?(?P<seg2>[a-z][a-z &-]{2,30}?)\s+"
    r"(?:revenues?|sales)\b", re.I)
_GENERIC: Final = {"total", "overall", "net", "quarterly", "year", "yearly", "annual", "revenue",
                   "profit", "earnings", "income", "its", "the", "their", "company"}


def _asked_segments(text: str, found: list[_Segment]) -> tuple[list[_Segment], str | None]:
    """The segment lines the text names, and the phrase it named when none matches."""
    spaced = _plain(text)
    squeezed = spaced.replace(" ", "")
    picked: list[_Segment] = []
    for seg in found:
        label = _plain(seg.label)
        if len(label.strip()) < 4 or label.strip() in ("other", "all other"):
            continue
        if label in spaced or (len(label.strip().replace(" ", "")) >= 8
                               and label.replace(" ", "") in squeezed):
            picked.append(seg)
    if picked:
        return picked, None
    m = _SEGMENT_ASK.search(text)
    if m:
        words = [w for w in re.findall(r"[a-z]+", (m.group("seg") or m.group("seg2")).lower())
                 if w not in _GENERIC]
        if words:
            return [], " ".join(words)
    return [], None


def _reaction(ticker: str, end: date) -> tuple[date, bool, date, float] | None:
    """(report day, after the close?, first session reacting, move) for the report that carried
    the quarter ending ``end``: the first results 8-K filed more than twelve days and at most a
    hundred after the quarter end. The move is the close before the news to the first close after
    it, as `earnings_move.realised_moves` takes it."""
    from datetime import timedelta

    reports = _reports(ticker)
    candidates = [(d, after) for d, after in reports
                  if end + timedelta(days=12) < d <= end + timedelta(days=100)]
    if not candidates:
        return None
    day, after = min(candidates)
    closes = _closes(ticker)
    days = [d for d, _ in closes]
    k = next((i for i, d in enumerate(days) if d >= day), None)
    if k is None:
        return None
    if after:
        if days[k] != day or k + 1 >= len(days):
            return None
        return day, True, days[k + 1], closes[k + 1][1] / closes[k][1] - 1
    if k == 0:
        return None
    return day, False, days[k], closes[k][1] / closes[k - 1][1] - 1


def _results_and_reaction(text: str, ticker: str, *, reaction_asked: bool) -> list[str]:
    c = _load(ticker)
    if (bad := _errors(c)) is not None:
        return [f"Bottom line: {bad}"]
    from argus.lui.research.filing_figures import _value

    end, year_ago = c["end"], c["prior"]
    revenue, then = _value(c, "revenue", end), _value(c, "revenue", year_ago)
    parts: list[str] = []
    detail: list[str] = []
    if revenue is not None:
        parts.append(f"revenue of {_bn(revenue)}" + (f" ({_gap(revenue, then)})"
                                                     if then is not None else ""))
    else:
        parts.append("its revenue could not be read from the filings")
    # the segment the question names, from the 10-Q's inline XBRL
    try:
        found = _document(ticker)
    except Exception:
        found = None
    asked: list[_Segment] = []
    named: str | None = None
    if found is not None:
        asked, named = _asked_segments(text, segments(found[0], found[3]))
    for seg in asked:
        move = f" ({_gap(seg.now, seg.then)})" if seg.then is not None else (
            " (no figure for the same quarter a year earlier is tagged)")
        name = seg.label if re.search(r"revenues?$", seg.label, re.I) else f"{seg.label} revenue"
        parts.append(f"{name} of {_bn(seg.now)}{move}")
        if seg.end != end:
            detail.append(f"{seg.label} is for the quarter to {seg.end:%d %b %Y}, the filing's "
                          f"latest; the totals above are to {end:%d %b %Y}.")
    if named is not None:
        top = (segments(found[0], found[3]) if found else [])[:5]
        tagged = ", ".join(f"{s.label} {_bn(s.now)}" for s in top) or "none"
        parts.append(f"no '{named}' revenue line is tagged in its latest 10-Q (it tags: {tagged})")
    elif found is None and _SEGMENT_ASK.search(text):
        parts.append("its latest 10-Q could not be read, so no segment revenue is given")
    reaction = ""
    if reaction_asked:
        try:
            move_read = _reaction(ticker, end)
            reaction_error = None
        except Exception:
            move_read = None
            reaction_error = "Yahoo's daily closes or the 8-K list could not be read"
        if move_read is not None:
            day, after, session, price_move = move_read
            timing = "after the close" if after else "before the open"
            reaction = (f"; the stock moved {price_move:+.1%} on {session:%d %b %Y}, its first "
                        f"session after the release ({day:%d %b %Y}, {timing})")
        else:
            reaction = ("; the stock's reaction could not be read ("
                        + (reaction_error or "no results 8-K for that quarter was found on EDGAR, "
                                             "or its first trading session has no close yet")
                        + ")")
    out = [f"Bottom line: {ticker} reported, for the quarter to {end:%d %b %Y}: "
           + "; ".join(parts) + reaction + "."]
    out += detail
    if revenue is not None and then is not None:
        out.append(f"Revenue a year earlier: {_bn(then)} (quarter to {year_ago:%d %b %Y}).")
    out.append("Data: SEC EDGAR XBRL company facts (total revenue), the latest 10-Q's inline XBRL "
               "(segment revenue, this quarter and the same quarter a year earlier)"
               + (", 8-K item 2.02 filing times for the release, Yahoo Finance daily closes "
                  "(close before the news to the first close after)" if reaction_asked else "")
               + ". Not advice.")
    return out


# ----------------------------------------------------------------------------- M7: FCF margin


def _fcf_margin(text: str, prior: Sequence[str]) -> list[str]:
    tickers = _tickers(text, prior)
    if not tickers:
        return ["Bottom line: no company is named — say which ones, for example \"free cash flow "
                "margin of NVDA vs AMD\"."]
    rows: list[dict[str, Any]] = []
    notes: list[str] = []
    for ticker in tickers:
        c = _load(ticker)
        if (bad := _errors(c)) is not None:
            notes.append(bad)
            continue
        revenue, ends = _trailing(c, "revenue")
        cfo, _ = _trailing(c, "cfo")
        capex, _ = _trailing(c, "capex")
        if revenue is None or not revenue:
            notes.append(f"{ticker}: fewer than four consecutive quarters of revenue are tagged.")
            continue
        if cfo is None:
            notes.append(f"{ticker}: operating cash flow is not tagged for each of the last four "
                         "quarters, so no free cash flow margin is computed.")
            continue
        if capex is None:
            notes.append(f"{ticker}: capital spending is not tagged for each of the last four "
                         "quarters under the property-and-equipment tags, so no free cash flow "
                         "margin is computed.")
            continue
        last = ends[-1]
        series = c["series"]
        quarter = ((series["cfo"][last] - series["capex"][last]) / series["revenue"][last]
                   if last in series["cfo"] and last in series["capex"] and series["revenue"][last]
                   else None)
        rows.append({"ticker": ticker, "revenue": revenue, "cfo": cfo, "capex": capex,
                     "fcf": cfo - capex, "margin": (cfo - capex) / revenue, "end": last,
                     "quarter": quarter})
    if not rows:
        return ["Bottom line: " + " ".join(notes)]
    ranked = sorted(rows, key=lambda r: -r["margin"])
    if len(ranked) == 1:
        r = ranked[0]
        head = (f"{r['ticker']}'s free cash flow margin over the four quarters to "
                f"{r['end']:%d %b %Y} is {r['margin']:.1%}")
    else:
        top, second = ranked[0], ranked[1]
        gap = (top["margin"] - second["margin"]) * 100
        head = (f"{top['ticker']} has the better free cash flow margin: {top['margin']:.1%} "
                f"against {second['ticker']}'s {second['margin']:.1%} ({gap:.1f} points "
                f"higher), over the four quarters to {top['end']:%d %b %Y}")
        if top["end"] != second["end"]:
            head += f" ({second['ticker']}'s to {second['end']:%d %b %Y})"
    out = ["Bottom line: " + head + "."]
    for r in ranked:
        line = (f"{r['ticker']}: free cash flow {_bn(r['fcf'])} on revenue {_bn(r['revenue'])} = "
                f"{r['margin']:.1%} (operating cash flow {_bn(r['cfo'])} less capital spending "
                f"{_bn(r['capex'])})")
        if r["quarter"] is not None:
            line += f"; the latest quarter alone {r['quarter']:.1%}"
        out.append(line + ".")
    out += notes
    out.append("Free cash flow here is operating cash flow less cash paid for property and "
               "equipment; leases and acquired intangibles are not counted as capital spending.")
    out.append("Data: each company's 10-Q and 10-K XBRL company facts on SEC EDGAR; quarters are "
               "rebuilt from year-to-date figures where the filing gives only those. Not advice.")
    return out


# ----------------------------------------------------------------------------- M8: risk factors


_BLOCK_TAGS: Final = frozenset({"p", "div", "li", "tr", "table", "h1", "h2", "h3", "h4", "h5", "h6",
                                "br", "ul", "ol", "td"})
_BOLD_STYLE: Final = re.compile(r"font-weight:\s*(?:bold|[6-9]00)", re.I)
_ITEM_1A: Final = re.compile(r"^item\s*1a\.?(?:\s*[^\w\s]?\s*risk\s+factors)?\.?$", re.I)
_ITEM_1B: Final = re.compile(r"^item\s*1b\b", re.I)
_NOT_A_HEADING: Final = re.compile(r"^(?:item\s*1a|part\s+i\b|table\s+of\s+contents)", re.I)


class _Blocks(HTMLParser):
    """The document as text blocks, each with its leading run of bold text: a risk factor's
    heading is bold, either a paragraph of its own or the first sentence of its paragraph."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[tuple[str, str]] = []
        self._parts: list[tuple[str, bool]] = []
        self._stack: list[tuple[str, bool]] = []

    def _flush(self) -> None:
        text = re.sub(r"\s+", " ", "".join(t for t, _ in self._parts)).strip()
        if text:
            lead: list[str] = []
            for chunk, bold in self._parts:
                if chunk.strip() and not bold:
                    break
                lead.append(chunk)
            self.blocks.append((text, re.sub(r"\s+", " ", "".join(lead)).strip()))
        self._parts = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _BLOCK_TAGS:
            self._flush()
        style = dict(attrs).get("style") or ""
        self._stack.append((tag, bool(_BOLD_STYLE.search(style)) or tag in ("b", "strong")))

    def handle_endtag(self, tag: str) -> None:
        if tag in _BLOCK_TAGS:
            self._flush()
        for i in range(len(self._stack) - 1, -1, -1):
            if self._stack[i][0] == tag:
                del self._stack[i:]
                break

    def handle_data(self, data: str) -> None:
        self._parts.append((data, any(b for _, b in self._stack)))

    def close(self) -> None:
        super().close()
        self._flush()


def risk_headings(document: str) -> list[str]:
    """The Item 1A headings of a 10-K's HTML, in filing order. The section runs from the "Item 1A"
    heading to the next "Item 1B"; the table-of-contents entry and the page headers that repeat
    "Item 1A" are skipped by taking the pair that spans the most text."""
    parser = _Blocks()
    parser.feed(document)
    parser.close()
    blocks = parser.blocks
    starts = [i for i, (t, _) in enumerate(blocks) if _ITEM_1A.match(t)]
    ends = [i for i, (t, _) in enumerate(blocks) if _ITEM_1B.match(t)]
    best: tuple[int, int] | None = None
    for s in starts:
        e = next((x for x in ends if x > s), None)
        if e is not None and (best is None or e - s > best[1] - best[0]):
            best = (s, e)
    if best is None:
        return []
    found: list[str] = []
    for _, lead in blocks[best[0] + 1: best[1]]:
        if (len(lead.split()) >= 4 and not _NOT_A_HEADING.match(lead) and not lead.endswith(":")
                and lead not in found):
            found.append(lead)
    return found


_STOP: Final = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "could", "for", "from", "has", "have",
    "in", "is", "it", "its", "may", "of", "on", "or", "our", "that", "the", "their", "to", "we",
    "which", "will", "with", "would", "company"})


def _tokens(heading: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", heading.lower()) if w not in _STOP}


def _similar(a: str, b: str) -> bool:
    """Whether two risk headings are the same risk: close in text, or sharing most of their
    meaningful words (a company rewords a heading each year)."""
    plain_a, plain_b = re.sub(r"[^a-z0-9 ]", "", a.lower()), re.sub(r"[^a-z0-9 ]", "", b.lower())
    if difflib.SequenceMatcher(None, plain_a, plain_b).ratio() >= 0.7:
        return True
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return False
    shared = len(ta & tb)
    return shared / len(ta | tb) >= 0.55 or (min(len(ta), len(tb)) >= 4
                                              and shared / min(len(ta), len(tb)) >= 0.85)


def risk_changes(now: Sequence[str], before: Sequence[str]) -> tuple[list[str], list[str]]:
    """(headings only in ``now``, headings only in ``before``)."""
    new = [h for h in now if not any(_similar(h, p) for p in before)]
    gone = [h for h in before if not any(_similar(h, n) for n in now)]
    return new, gone


def _risk_factors(ticker: str) -> list[str]:
    try:
        filings = _tenk_filings(ticker)
    except Exception:
        filings = []
    if not filings:
        return [f"Bottom line: {ticker}'s 10-K filings could not be read from SEC EDGAR just now."]
    filed, period, url = filings[0]
    try:
        now = risk_headings(_fetch(url))
    except Exception:
        now = []
    if len(now) < 5:
        return [f"Bottom line: {ticker}'s latest 10-K (filed {filed:%d %b %Y}) could not be read "
                f"for its Item 1A risk factors just now. Source: {url}"]
    out: list[str] = []
    prior = next(((f, p, u) for f, p, u in filings[1:] if 340 <= (period - p).days <= 390), None)
    before: list[str] = []
    why = ""
    if prior is None:
        why = "the prior year's 10-K is not among the company's recent EDGAR filings"
    else:
        try:
            before = risk_headings(_fetch(prior[2]))
        except Exception:
            before = []
        if len(before) < 5:
            why = (f"the prior year's 10-K (filed {prior[0]:%d %b %Y}) could not be read for its "
                   "Item 1A just now")
    long_form = [h for h in now if len(h.split()) >= 8]
    lead = (f"{ticker}'s latest 10-K (fiscal year to {period:%d %b %Y}, filed {filed:%d %b %Y}) "
            f"lists {len(now)} risk-factor headings in Item 1A")
    new: list[str] = []
    gone: list[str] = []
    if why:
        out.append(f"Bottom line: {lead}; no comparison with the prior year is made, because "
                   f"{why}.")
    else:
        assert prior is not None
        new, gone = risk_changes(now, before)
        out.append(f"Bottom line: {lead}; against the prior year's 10-K (filed "
                   f"{prior[0]:%d %b %Y}, {len(before)} headings), {len(new)} new and "
                   f"{len(gone)} of its headings no longer there.")
    out.append("Main risks, in the order the 10-K gives them: "
               + " | ".join(_short(h, 140) for h in long_form[:5]) + ".")
    if not why:
        for label, group in (("New in this year's 10-K (Item 1A)", new),
                             ("In the prior year's 10-K (Item 1A), not in this year's", gone)):
            if group:
                out.append(f"{label}: " + " | ".join(_short(h, 140) for h in group[:6])
                           + (f" | and {len(group) - 6} more" if len(group) > 6 else "") + ".")
            else:
                out.append(f"{label}: none.")
        out.append("A heading the company reworded can appear in both lists; read the filings' "
                   "text before treating one as a new exposure.")
    out.append(f"Data: Item 1A of the 10-K filed {filed:%d %b %Y} ({url})"
               + (f" and of the 10-K filed {prior[0]:%d %b %Y} ({prior[2]})"
                  if prior is not None and not why else "")
               + ", headings read from the filings' bold text. Not advice.")
    return out


# ----------------------------------------------------------------------------- M9: margin trend


def _quarter_count(text: str) -> int:
    m = re.search(r"\b(?:last|past|previous)\s+(\d+|[a-z]+)\s+quarters\b", text, re.I)
    if m:
        raw = m.group(1).lower()
        n = int(raw) if raw.isdigit() else _WORD_COUNT.get(raw, 0)
        if n:
            return max(2, min(n, 16))
    return 8


def _slope(values: Sequence[float]) -> float:
    """Least-squares change per step."""
    n = len(values)
    mean_x, mean_y = (n - 1) / 2, sum(values) / n
    denominator = sum((i - mean_x) ** 2 for i in range(n))
    return sum((i - mean_x) * (v - mean_y) for i, v in enumerate(values)) / denominator


def _margin_series(c: dict[str, Any], kind: str) -> dict[date, float]:
    series = c["series"]
    revenue: dict[date, float] = series["revenue"]
    out: dict[date, float] = {}
    for end, rev in revenue.items():
        if not rev:
            continue
        if kind == "gross":
            if end in series["gross"]:
                out[end] = series["gross"][end] / rev
            elif end in series["cost"]:
                out[end] = 1 - series["cost"][end] / rev
        elif kind == "operating" and end in series["operating"]:
            out[end] = series["operating"][end] / rev
        elif kind == "net" and end in series["net"]:
            out[end] = series["net"][end] / rev
    return out


def _margin_trend(text: str, prior: Sequence[str]) -> list[str]:
    tickers = _tickers(text, prior, limit=3)
    if not tickers:
        return ["Bottom line: no company is named — say which one, for example \"Apple's gross "
                "margin over the last 8 quarters\"."]
    wanted = [k for k in ("gross", "operating", "net")
              if re.search(rf"\b{k}\s+margins?\b", text, re.I)]
    if re.search(r"\bprofit\s+margins?\b", text, re.I) and "net" not in wanted:
        wanted.append("net")
    wanted = wanted or ["gross", "operating"]
    count = _quarter_count(text)
    out: list[str] = []
    heads: list[str] = []
    for ticker in tickers:
        c = _load(ticker)
        if (bad := _errors(c)) is not None:
            out.append(bad)
            continue
        for kind in wanted:
            margins = _margin_series(c, kind)
            run = _recent_run(list(margins))[-count:]
            label = f"{ticker} {kind} margin"
            if len(run) < 2:
                out.append(f"{label}: fewer than two consecutive quarters are tagged in its "
                           "filings, so no trend is computed.")
                continue
            values = [margins[d] for d in run]
            change = (values[-1] - values[0]) * 100
            per_quarter = _slope(values)
            span = per_quarter * (len(values) - 1) * 100
            direction = ("widening" if span > 1.0 else "narrowing" if span < -1.0
                         else "broadly flat")
            short = (f" ({len(run)} of the {count} quarters asked for are tagged)"
                     if len(run) < count else "")
            heads.append(f"{ticker}'s {kind} margin is {direction}, {values[-1]:.1%} in the "
                         f"quarter to {run[-1]:%b %Y} against {values[0]:.1%} in {run[0]:%b %Y} "
                         f"({change:+.1f} points)")
            out.append(f"{label}, {len(run)} quarters" + short + ": "
                       + ", ".join(f"{d:%b %Y} {v:.1%}" for d, v in zip(run, values, strict=True))
                       + f". Trend {per_quarter * 100:+.2f} points a quarter; high "
                       f"{max(values):.1%}, low {min(values):.1%}.")
    if not heads:
        return ["Bottom line: " + (" ".join(out) if out else "no margin could be read just now.")]
    out.insert(0, "Bottom line: " + "; ".join(heads) + ".")
    out.append("Direction: the fitted trend over the quarters shown, widening or narrowing when it "
               "moves more than one point in total, else broadly flat; the latest quarter is set "
               "against the first quarter shown.")
    out.append("Data: each company's 10-Q and 10-K XBRL company facts on SEC EDGAR; a fiscal "
               "fourth quarter is the year less the nine months, and margins are the quarter's "
               "profit line over its revenue. Not advice.")
    return out


# ----------------------------------------------------------------------------- M10: valuation


def _number(node: Any) -> float | None:
    from argus.lui.research.fundamentals import raw_number

    value = raw_number(node)
    return value if value is not None and math.isfinite(value) else None


def _valuation_row(ticker: str) -> dict[str, Any]:
    c = _load(ticker)
    row: dict[str, Any] = {"ticker": ticker, "notes": []}
    if (bad := _errors(c)) is not None:
        row["notes"].append(bad)
        return row
    revenue, ends = _trailing(c, "revenue")
    row["revenue"], row["end"], row["c"] = revenue, (ends[-1] if ends else c["end"]), c
    before = _prior_four(c, ends) if revenue is not None else None
    row["growth"] = (revenue / before - 1) if revenue and before and before > 0 else None
    if row["growth"] is None:
        row["notes"].append(f"{ticker}: eight quarters of revenue are not all tagged, so the "
                            "trailing growth rate is not computed.")
    try:
        summary = _yahoo(ticker)
    except Exception:
        row["notes"].append(f"{ticker}: Yahoo Finance's valuation fields could not be read just "
                            "now.")
        return row
    detail = summary.get("summaryDetail") or {}
    stats = summary.get("defaultKeyStatistics") or {}
    row["cap"] = _number(detail.get("marketCap"))
    row["trailing_pe"] = _number(detail.get("trailingPE"))
    row["forward_pe"] = _number(detail.get("forwardPE")) or _number(stats.get("forwardPE"))
    row["eps"] = _number(stats.get("trailingEps"))
    row["ev"] = _number(stats.get("enterpriseValue"))
    row["ps"] = row["cap"] / revenue if row.get("cap") and revenue else None
    row["evs"] = row["ev"] / revenue if row.get("ev") and revenue else None
    return row


def _multiple(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if value is not None and value > 0:
        return _x(value)
    if key == "trailing_pe" and row.get("eps") is not None and row["eps"] <= 0:
        return f"none (loss-making, trailing EPS {row['eps']:.2f})"
    return "not available"


def _cheaper(rows: list[dict[str, Any]], key: str, label: str) -> str | None:
    pool = [r for r in rows if (r.get(key) or 0) > 0]
    if len(rows) < 2 or not pool:
        return None
    if len(pool) == 1:
        gone = ", ".join(r["ticker"] for r in rows if r not in pool)
        return (f"{pool[0]['ticker']} is the only one with a meaningful {label} "
                f"({_x(pool[0][key])}; {gone} has none)")
    pool.sort(key=lambda r: r[key])
    return (f"{pool[0]['ticker']} is cheaper on {label} ({_x(pool[0][key])} against "
            f"{_x(pool[-1][key])} for {pool[-1]['ticker']})")


def _compare_valuation(tickers: list[str]) -> list[str]:
    rows = [_valuation_row(t) for t in tickers[:4]]
    good = [r for r in rows if "c" in r]
    notes = [n for r in rows for n in r["notes"]]
    if len(good) < 2:
        return ["Bottom line: fewer than two of the companies could be read just now. "
                + " ".join(notes)]
    out: list[str] = []
    verdicts: list[str] = []
    grown = [r for r in good if r["growth"] is not None]
    if len(grown) >= 2:
        grown.sort(key=lambda r: -r["growth"])
        verdicts.append(f"{grown[0]['ticker']} is growing faster ({grown[0]['growth']:+.1%} "
                        f"against {grown[-1]['growth']:+.1%} for {grown[-1]['ticker']}, trailing "
                        "four quarters)")
    for key, label in (("trailing_pe", "trailing P/E"), ("forward_pe", "forward P/E"),
                       ("ps", "price/sales"), ("evs", "EV/sales")):
        said = _cheaper(good, key, label)
        if said:
            verdicts.append(said)
    out.append("Bottom line: " + "; ".join(verdicts) + "." if verdicts else
               "Bottom line: the valuation fields could not be read for these companies just now.")
    for r in good:
        growth = (f"{r['growth']:+.1%}" if r["growth"] is not None else "not computed")
        cap = _bn(r["cap"]) if r.get("cap") else "not available"
        out.append(f"{r['ticker']}: revenue growth {growth} (four quarters to {r['end']:%d %b %Y}"
                   f"{', ' + _bn(r['revenue']) if r['revenue'] else ''}); market cap {cap}; "
                   f"trailing P/E {_multiple(r, 'trailing_pe')}; forward P/E "
                   f"{_multiple(r, 'forward_pe')}; price/sales {_multiple(r, 'ps')}; EV/sales "
                   f"{_multiple(r, 'evs')}.")
    out += notes
    out.append("Cheaper on a multiple is not the same as better value: a multiple is the price "
               "paid for a dollar of sales or earnings, and says nothing about the growth "
               "behind it.")
    out.append("Data: revenue from each company's 10-Q and 10-K XBRL facts on SEC EDGAR (four "
               "quarters against the four a year earlier); market cap, enterprise value and P/E "
               "from Yahoo Finance quoteSummary, price/sales and EV/sales over that revenue. "
               "Not advice.")
    return out


def _justified(ticker: str) -> list[str]:
    row = _valuation_row(ticker)
    if "c" not in row:
        return ["Bottom line: " + " ".join(row["notes"])]
    from argus.lui.research.filing_figures import _value

    c = row["c"]
    end, year_ago = c["end"], c["prior"]
    net, net_then = _value(c, "net income", end), _value(c, "net income", year_ago)
    rev, rev_then = _value(c, "revenue", end), _value(c, "revenue", year_ago)
    quarter_growth = rev / rev_then - 1 if rev and rev_then and rev_then > 0 else None
    trailing: float | None = row.get("trailing_pe")
    forward: float | None = row.get("forward_pe")
    delivered = quarter_growth if quarter_growth is not None else row["growth"]
    if net is not None and net_then is not None and net_then > 0:
        had = (f"net income {'fell' if net < net_then else 'rose'} "
               f"{abs(net / net_then - 1):.1%} in the latest quarter")
    else:
        had = "latest-quarter net income growth is not available"
    implied = trailing / forward - 1 if trailing and forward and forward > 0 else None
    if trailing is None or forward is None or implied is None:
        headline = (f"{ticker}'s P/E cannot be set against its growth here: it needs both a "
                    "trailing and a forward P/E, and "
                    + ("the trailing one does not exist because earnings over the last year are "
                       "negative" if trailing is None else "the forward one is missing"))
    elif implied <= 0:
        headline = (f"{ticker}'s forward P/E ({_x(forward)}) is not below its trailing one "
                    f"({_x(trailing)}): analysts expect no earnings growth over the next year "
                    f"({implied:+.0%}), so the multiple is not resting on expected growth; {had}")
    elif delivered is None or delivered <= 0:
        headline = (f"{ticker}'s multiple ({_x(trailing)} trailing, {_x(forward)} forward) assumes "
                    f"{implied:+.0%} earnings growth next year, and revenue did not grow in the "
                    f"latest quarter; {had}")
    elif implied / delivered > 1.5:
        headline = (f"{ticker}'s multiple is priced for more growth than the latest quarter "
                    f"delivered: {_x(trailing)} trailing and {_x(forward)} forward assume "
                    f"{implied:+.0%} earnings growth next year, about {implied / delivered:.1f} "
                    "times the "
                    f"{delivered:.1%} revenue growth delivered, and {had}")
    else:
        headline = (f"{ticker}'s multiple ({_x(trailing)} trailing, {_x(forward)} forward) assumes "
                    f"{implied:+.0%} earnings growth next year, within 1.5 times the "
                    f"{delivered:.1%} revenue growth delivered; {had}")
    out = [f"Bottom line: {headline}."]
    delivered_said: list[str] = []
    if net is not None:
        delivered_said.append(f"net income {_bn(net)}"
                              + (f", {_gap(net, net_then)}" if net_then is not None else ""))
    else:
        delivered_said.append("net income is not tagged for it")
    if rev is not None:
        delivered_said.append(f"revenue {_bn(rev)}"
                              + (f", {_gap(rev, rev_then)}" if rev_then is not None else ""))
    if row["growth"] is not None:
        delivered_said.append(f"revenue over the last four quarters {row['growth']:+.1%}")
    out.append(f"Delivered, latest quarter to {end:%d %b %Y}: " + "; ".join(delivered_said) + ".")
    out.append("The multiple: trailing P/E " + (_x(trailing) if trailing else "none (loss-making)")
               + ", forward P/E " + (_x(forward) if forward else "not available")
               + (f"; trailing over forward is the +{implied:.0%} next-year EPS growth analysts "
                  "expect" if implied is not None and implied > 0 else "") + ".")
    if trailing is not None and delivered is not None and delivered > 0:
        years = math.log(2) / math.log(1 + delivered)
        out.append(f"Arithmetic: at {delivered:.1%} a year with margins unchanged, earnings take "
                   f"{years:.1f} years to double, which is what halves a {_x(trailing)} trailing "
                   "P/E.")
    out.append("This sets the growth the multiple assumes against the growth delivered; it is not "
               "a call on whether the price is right, and no buy or sell view is given.")
    out += row["notes"]
    out.append("Data: latest-quarter and trailing figures from SEC EDGAR XBRL company facts; "
               "trailing and forward P/E from Yahoo Finance quoteSummary (trailing is GAAP EPS, "
               "forward the analysts' consensus). Not advice.")
    return out


# ----------------------------------------------------------------------------- the entry point


def lines(text: str, prior: Sequence[str] = ()) -> list[str] | None:
    """The answer to a company-research question this module owns, or None.

    Owns: results plus the stock's reaction (revenue and a named segment); a free cash flow margin
    comparison; a 10-K's risk factors set against the prior year; a quarterly margin trend; a
    valuation comparison of two or more stocks; and whether one stock's P/E is justified."""
    if not text or not text.strip():
        return None
    margin_trend = _MARGIN.search(text) and _TREND.search(text)
    # 10-K risk factors against the prior year
    if _RISK.search(text) and _DIFF.search(text) and (_TENK.search(text) or _stocks(text)):
        stocks = _stocks(text)
        if len(stocks) == 1:
            return _risk_factors(stocks[0])
        return None
    # free cash flow margin
    if _FCF_MARGIN.search(text):
        return _fcf_margin(text, prior)
    # a margin trend over several quarters
    if margin_trend and not _VALUATION.search(text):
        if _stocks(text) or _PAIR_REFERENCE.search(text) or prior:
            return _margin_trend(text, prior)
        return None
    # valuation and growth
    if _VALUATION.search(text):
        stocks = _stocks(text)
        if len(stocks) >= 2 and (_GROWTH.search(text) or _COMPARE.search(text)):
            return _compare_valuation(stocks)
        if len(stocks) == 1 and _JUSTIFIED.search(text) and re.search(
                r"\bp\s*/\s*e\b|\bpe\s+ratio\b|\bmultiple\b|\bvaluation\b|\bprice[\s-]to[\s-]"
                r"earnings\b", text, re.I):
            return _justified(stocks[0])
        return None
    # results and the stock's reaction
    reaction_asked = bool(_REACTION.search(text))
    if ((reaction_asked or _SEGMENT_ASK.search(text)) and _RESULTS.search(text)
            and _REVENUE.search(text) and not _OPTIONS.search(text)):
        stocks = _stocks(text)
        if len(stocks) == 1:
            return _results_and_reaction(text, stocks[0], reaction_asked=reaction_asked)
    return None


__all__ = ["lines", "risk_changes", "risk_headings", "segments"]

# Every engine here is a traced step from import on (lui/trace.py, trace_module).
trace_module(globals())
