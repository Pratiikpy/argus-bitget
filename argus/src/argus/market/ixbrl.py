"""Revenue by product line, read from the inline XBRL of a company's latest 10-Q or 10-K.

SEC EDGAR's company-facts API carries only a filing's undimensioned totals; revenue split by
product or service — Robinhood's "Cryptocurrencies" transaction revenue, Coinbase's consumer and
institutional transaction revenue — is tagged in the filing itself, as a fact whose context names
a member of ``srt:ProductOrServiceAxis`` (or a company axis). Round 41's judge (M10) asked for
Coinbase's and Robinhood's crypto revenue and got their total revenue only.

This module reads the 10-Q's inline XBRL: every ``xbrli:context`` (its period and its explicit
dimension members) and every ``ix:nonFraction`` revenue fact, applying each fact's ``scale`` and
``sign``. A fact is kept when its context has exactly one dimension and its period is the latest
quarter (a duration of 80 to 100 days), so year-to-date figures and cross-tabulated cells do not
leak into the split.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from datetime import date
from typing import Final

REVENUE_TAGS: Final = ("us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
                       "us-gaap:Revenues")
_CONTEXT: Final = re.compile(r"<xbrli:context\b[^>]*\bid=\"(?P<id>[^\"]+)\"[^>]*>(?P<body>.*?)"
                             r"</xbrli:context>", re.S)
_MEMBER: Final = re.compile(r"<xbrldi:explicitMember\b[^>]*dimension=\"(?P<axis>[^\"]+)\"[^>]*>"
                            r"(?P<member>[^<]+)</xbrldi:explicitMember>")
_FACT: Final = re.compile(r"<ix:nonFraction\b(?P<attrs>[^>]*)>"
                          r"(?P<value>.*?)</ix:nonFraction>", re.S)
_ATTR: Final = re.compile(r"(\w[\w:.-]*)=\"([^\"]*)\"")


@dataclass(frozen=True)
class Line:
    member: str
    label: str
    value: float
    start: date
    end: date
    axis: str = ""
    """The dimension the member belongs to (``srt:ProductOrServiceAxis``, a segment or a
    geography axis), so a split is read one axis at a time."""


def _label(member: str) -> str:
    name = member.split(":")[-1].removesuffix("Member")
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)


def labels(linkbase: str) -> dict[str, str]:
    """``prefix:Name`` -> the company's own label for it, from the filing's label linkbase
    (``*_lab.xml``): the terse label where one is given, else the standard label without its
    " [Member]" suffix. Coinbase tags its transaction revenue with the standard
    ``us-gaap:BankServicingMember`` and labels it "Transaction revenue" there."""
    found: dict[str, tuple[int, str]] = {}
    for m in re.finditer(r"<link:label\b([^>]*)>([^<]*)</link:label>", linkbase):
        attrs = dict(_ATTR.findall(m.group(1)))
        key = attrs.get("xlink:label", "").removeprefix("lab_")
        if "_" not in key:
            continue
        prefix, name = key.split("_", 1)
        role = attrs.get("xlink:role", "")
        rank = 0 if role.endswith("/terseLabel") else 1 if role.endswith("/label") else 2
        text = html.unescape(m.group(2)).replace(" [Member]", "").strip()
        concept = f"{prefix}:{name}"
        if text and (concept not in found or rank < found[concept][0]):
            found[concept] = (rank, text)
    return {k: v for k, (_, v) in found.items()}


def dimensioned_revenue(document: str, names: dict[str, str] | None = None) -> list[Line]:
    """Every revenue fact in ``document`` with one dimension member and a quarter's or a year's
    period, each period kept: a 10-Q tags the same quarter a year earlier beside the latest, which
    is what a segment's year-on-year growth is read from."""
    contexts: dict[str, tuple[date, date, list[tuple[str, str]]]] = {}
    for m in _CONTEXT.finditer(document):
        body = m.group("body")
        began = re.search(r"<xbrli:startDate>([^<]+)</xbrli:startDate>", body)
        ended = re.search(r"<xbrli:endDate>([^<]+)</xbrli:endDate>", body)
        if not began or not ended:
            continue
        members = [(x.group("axis"), x.group("member").strip()) for x in _MEMBER.finditer(body)]
        contexts[m.group("id")] = (date.fromisoformat(began.group(1).strip()),
                                   date.fromisoformat(ended.group(1).strip()), members)
    found: list[Line] = []
    for fact in _FACT.finditer(document):
        attrs = dict(_ATTR.findall(fact.group("attrs")))
        if attrs.get("name") not in REVENUE_TAGS:
            continue
        context = contexts.get(attrs.get("contextRef", ""))
        if context is None:
            continue
        start, end, members = context
        days = (end - start).days
        if len(members) != 1 or not (80 <= days <= 100 or 350 <= days <= 380):
            continue
        raw = re.sub(r"<[^>]+>", "", fact.group("value"))
        raw = html.unescape(raw).replace(",", "").strip()
        try:
            value = float(raw) * 10 ** int(attrs.get("scale", "0") or 0)
        except ValueError:
            continue
        if attrs.get("sign") == "-":
            value = -value
        member = members[0][1]
        found.append(Line(member=member, label=(names or {}).get(member) or _label(member),
                          value=value, start=start, end=end, axis=members[0][0]))
    return found


def revenue_lines(document: str, names: dict[str, str] | None = None) -> list[Line]:
    """Revenue facts with one dimension member, for the latest quarter in ``document``."""
    found = dimensioned_revenue(document, names)
    if not found:
        return []
    # the quarter when the filing tags one; a 10-K that splits revenue for the year only (as
    # Microsoft's does) gives the year, and the line's start says which
    quarterly = [x for x in found if (x.end - x.start).days <= 100]
    found = quarterly or found
    latest = max(x.end for x in found)
    seen: set[str] = set()
    out = []
    for line in found:
        if line.end == latest and line.member not in seen:
            seen.add(line.member)
            out.append(line)
    return out


def latest_filing_document(ticker: str) -> tuple[str, str, str, dict[str, str]] | None:
    """(document text, form, period, member labels) of the latest 10-Q or 10-K, or None."""
    from argus.market.evidence import FEED_USER_AGENT, EdgarSource
    from argus.truth import http

    edgar = EdgarSource()
    cik = edgar.cik_for(ticker)
    if cik is None:
        return None
    recent = edgar._get(edgar.SUBMISSIONS_URL.format(cik=cik))["filings"]["recent"]
    for i, form in enumerate(recent.get("form", [])):
        if form in ("10-Q", "10-K"):
            accession = recent["accessionNumber"][i].replace("-", "")
            url = (f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/"
                   f"{recent['primaryDocument'][i]}")
            text = http.fetch_text(url, timeout=60.0, headers={"User-Agent": FEED_USER_AGENT})
            names: dict[str, str] = {}
            try:
                names = labels(http.fetch_text(url.rsplit(".", 1)[0] + "_lab.xml", timeout=60.0,
                                               headers={"User-Agent": FEED_USER_AGENT}))
            except Exception:
                names = {}
            return text, form, str(recent["reportDate"][i]), names
    return None


__all__ = ["Line", "labels", "latest_filing_document", "revenue_lines"]
