"""Form 4 insider transactions — and the codes that decide whether one means anything.

Bitget's Event-Driven Agent sub-theme asks how announcements drive autonomous trading. Insider
filings are the densest announcement stream a US issuer produces: NVDA alone filed 555 Form 4s in
its recent-submissions window, against 63 8-Ks. Ingesting them naively is therefore an excellent
way to drown a desk in noise, and most systems that read Form 4 do exactly that — they count buys
and sells.

**The first real Form 4 this module was written against is why.** NVDA accession
``0002152188-26-000005``, filed 2026-09-11 by an EVP: 172,507 shares acquired at a price of
**zero**. Counted as a purchase it is the largest insider buy of the quarter. It is an RSU vest.
Transaction code ``A`` — a grant. The executive expressed no view about anything.

So the code is the signal, not the direction:

* ``P`` — open-market purchase. The one transaction an insider makes with their own money, at a
  price they chose, when they did not have to. This is the case the literature finds informative.
* ``S`` — open-market sale. Weakly informative at best: insiders sell for houses, divorces, tax
  bills and diversification, none of which is a view on the business.
* ``A`` / ``F`` / ``M`` — grant, tax withholding on vest, option exercise. Mechanical. An ``F`` is
  generated *by* a vest the insider did not choose the timing of.
* ``aff10b5One`` — the filing's own flag saying the trade was pre-arranged under a Rule 10b5-1
  plan, often months earlier. A pre-arranged trade carries no information about today, and the
  filing tells you so; ignoring that field is choosing not to know.

:class:`InsiderTrade` therefore exposes :attr:`is_informative`, and the evidence builder emits only
those — with the discarded count stated, so a quiet feed is distinguishable from a filtered one.

**Parsing.** ``xml.etree`` with a size cap. SEC XML is a trusted, well-formed source and these
documents are small; the cap exists because "trusted" is a statement about intent, not about what
arrives on the socket.
"""

from __future__ import annotations

import re
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any

from argus.market.evidence import _UA, EdgarSource
from argus.truth.evidence import Evidence

MAX_DOCUMENT_BYTES = 2_000_000
"""A Form 4 is a few kilobytes. This is four orders of magnitude of headroom and still a bound."""

TIMEOUT = 15

INFORMATIVE_CODES = frozenset({"P", "S"})
"""Open-market transactions. Everything else is compensation machinery."""

CONVICTION_CODES = frozenset({"P"})
"""The insider spent their own money at a price they chose. The only strongly studied case."""

CODE_MEANING: dict[str, str] = {
    "P": "open-market purchase",
    "S": "open-market sale",
    "A": "grant or award",
    "F": "shares withheld for tax on vesting",
    "M": "option or derivative exercise",
    "G": "gift",
    "C": "conversion",
    "D": "disposition to the issuer",
}


@dataclass(frozen=True)
class InsiderTrade:
    """One transaction line from one Form 4."""

    issuer: str
    ticker: str
    insider: str
    title: str
    is_director: bool
    is_officer: bool
    is_ten_percent_owner: bool
    transaction_date: datetime
    accepted_at: datetime
    code: str
    acquired: bool
    shares: Decimal
    price: Decimal
    shares_after: Decimal
    pre_arranged: bool
    accession: str

    @property
    def meaning(self) -> str:
        return CODE_MEANING.get(self.code, f"code {self.code}")

    @property
    def notional(self) -> Decimal:
        return self.shares * self.price

    @property
    def is_informative(self) -> bool:
        """Did this person choose to trade, at a price they chose, for their own reasons?

        A pre-arranged 10b5-1 trade fails this even when the code is ``P``: the decision was taken
        months ago under different information, and the filing says so in its own field.
        """
        return self.code in INFORMATIVE_CODES and not self.pre_arranged and self.price > 0

    @property
    def is_conviction(self) -> bool:
        return self.is_informative and self.code in CONVICTION_CODES

    @property
    def role(self) -> str:
        if self.is_officer and self.title:
            return self.title
        if self.is_officer:
            return "officer"
        if self.is_director:
            return "director"
        if self.is_ten_percent_owner:
            return "10% owner"
        return "insider"

    def render(self) -> str:
        direction = "acquired" if self.acquired else "disposed"
        return (
            f"{self.insider} ({self.role}) {direction} {self.shares:,.0f} {self.ticker} "
            f"at {self.price} — {self.meaning}"
            + (" under a pre-arranged 10b5-1 plan" if self.pre_arranged else "")
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "insider": self.insider,
            "role": self.role,
            "code": self.code,
            "meaning": self.meaning,
            "acquired": self.acquired,
            "shares": str(self.shares),
            "price": str(self.price),
            "notional": str(self.notional),
            "pre_arranged": self.pre_arranged,
            "informative": self.is_informative,
            "conviction": self.is_conviction,
            "transaction_date": self.transaction_date.isoformat(),
            "accepted_at": self.accepted_at.isoformat(),
            "accession": self.accession,
        }


def _text(node: ET.Element | None, path: str, default: str = "") -> str:
    if node is None:
        return default
    found = node.find(path)
    return (found.text or default).strip() if found is not None else default


def _decimal(raw: str) -> Decimal:
    try:
        return Decimal(raw or "0")
    except InvalidOperation:
        return Decimal("0")


def _flag(node: ET.Element | None, path: str) -> bool:
    return _text(node, path, "0").strip() in {"1", "true", "TRUE"}


def parse_form4(xml_text: str, *, accepted_at: datetime, accession: str) -> list[InsiderTrade]:
    """Turn one Form 4 document into its transaction lines.

    A malformed document yields nothing rather than raising: one unparseable filing must not take
    the evidence feed down, and the feed reports its own health separately.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []

    issuer = root.find("issuer")
    owner = root.find("reportingOwner")
    relationship = owner.find("reportingOwnerRelationship") if owner is not None else None

    ticker = _text(issuer, "issuerTradingSymbol").upper()
    name = _text(issuer, "issuerName")
    insider = _text(owner, "reportingOwnerId/rptOwnerName") if owner is not None else ""
    pre_arranged = _flag(root, "aff10b5One")

    out: list[InsiderTrade] = []
    for table in ("nonDerivativeTable/nonDerivativeTransaction",
                  "derivativeTable/derivativeTransaction"):
        for tx in root.findall(table):
            code = _text(tx, "transactionCoding/transactionCode").upper()
            if not code:
                continue
            raw_date = _text(tx, "transactionDate/value")
            try:
                when = datetime.fromisoformat(raw_date).replace(tzinfo=UTC)
            except ValueError:
                continue
            out.append(
                InsiderTrade(
                    issuer=name,
                    ticker=ticker,
                    insider=insider,
                    title=_text(relationship, "officerTitle"),
                    is_director=_flag(relationship, "isDirector"),
                    is_officer=_flag(relationship, "isOfficer"),
                    is_ten_percent_owner=_flag(relationship, "isTenPercentOwner"),
                    transaction_date=when,
                    accepted_at=accepted_at,
                    code=code,
                    acquired=_text(
                        tx, "transactionAmounts/transactionAcquiredDisposedCode/value"
                    ).upper() == "A",
                    shares=_decimal(_text(tx, "transactionAmounts/transactionShares/value")),
                    price=_decimal(
                        _text(tx, "transactionAmounts/transactionPricePerShare/value")
                    ),
                    shares_after=_decimal(
                        _text(tx, "postTransactionAmounts/sharesOwnedFollowingTransaction/value")
                    ),
                    pre_arranged=pre_arranged,
                    accession=accession,
                )
            )
    return out


@dataclass(frozen=True)
class InsiderDecision:
    """One insider's transaction in one filing, however many price tiers it filled across.

    A market order large enough to walk the book is reported as several lines with different prices.
    Emitting each as separate evidence makes one decision look like five events, which is what the
    first live run of this module did: a director's single disposal in NVDA accession
    ``0001199039-26-000014`` arrived as five lines and five pieces of evidence. The trade was one
    decision and the desk should see one.
    """

    accession: str
    ticker: str
    insider: str
    role: str
    code: str
    meaning: str
    acquired: bool
    shares: Decimal
    notional: Decimal
    accepted_at: datetime
    tiers: int
    is_conviction: bool
    pre_arranged: bool = False
    """Always ``False`` for a decision built from informative trades, and carried anyway.

    Aggregation runs over trades that already passed :attr:`InsiderTrade.is_informative`, which
    excludes pre-arranged plans, so this field is constant by construction. It is kept because a
    claim cannot be checked against a field that is not there: ledger seq 41 asserted these sales
    were 10b5-1 plans and :mod:`argus.agents.claims` could not contradict it, because the record
    the claim was about had dropped the only attribute that settles it."""

    @property
    def average_price(self) -> Decimal:
        """Volume-weighted. A plain mean across tiers would misprice an uneven fill."""
        return (self.notional / self.shares) if self.shares else Decimal("0")

    def render(self) -> str:
        direction = "acquired" if self.acquired else "disposed"
        tail = f" across {self.tiers} price tiers" if self.tiers > 1 else ""
        return (
            f"{self.insider} ({self.role}) {direction} {self.shares:,.0f} {self.ticker} "
            f"at an average {self.average_price:.2f}{tail} — {self.meaning}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "accession": self.accession,
            "ticker": self.ticker,
            "insider": self.insider,
            "role": self.role,
            "code": self.code,
            "acquired": self.acquired,
            "shares": str(self.shares),
            "average_price": str(self.average_price),
            "notional": str(self.notional),
            "tiers": self.tiers,
            "conviction": self.is_conviction,
            "pre_arranged": self.pre_arranged,
            "accepted_at": self.accepted_at.isoformat(),
        }


def aggregate(trades: list[InsiderTrade]) -> list[InsiderDecision]:
    """Collapse transaction lines into decisions, one per (filing, insider, code, direction)."""
    buckets: dict[tuple[str, str, str, bool], list[InsiderTrade]] = {}
    for trade in trades:
        buckets.setdefault(
            (trade.accession, trade.insider, trade.code, trade.acquired), []
        ).append(trade)

    out: list[InsiderDecision] = []
    for (accession, insider, code, acquired), lines in buckets.items():
        shares = sum((t.shares for t in lines), Decimal("0"))
        notional = sum((t.notional for t in lines), Decimal("0"))
        first = lines[0]
        out.append(
            InsiderDecision(
                accession=accession,
                ticker=first.ticker,
                insider=insider,
                role=first.role,
                code=code,
                meaning=first.meaning,
                acquired=acquired,
                shares=shares,
                notional=notional,
                accepted_at=max(t.accepted_at for t in lines),
                tiers=len(lines),
                is_conviction=any(t.is_conviction for t in lines),
                pre_arranged=any(t.pre_arranged for t in lines),
            )
        )
    out.sort(key=lambda d: d.accepted_at, reverse=True)
    return out


class InsiderSource:
    """Fetches and parses Form 4 filings for one ticker."""

    ARCHIVE = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/"

    def __init__(self, *, user_agent: str = _UA, edgar: EdgarSource | None = None) -> None:
        self._ua = user_agent
        self._edgar = edgar or EdgarSource(user_agent=user_agent)

    def _fetch(self, url: str) -> str:
        request = urllib.request.Request(
            url, headers={"User-Agent": self._ua, "Accept": "*/*"}
        )
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            body: bytes = response.read(MAX_DOCUMENT_BYTES)
        return body.decode("utf-8", errors="replace")

    def _document_url(self, cik: int, accession: str) -> str | None:
        """Find the raw XML in the accession directory.

        The submissions index names an ``xslF345X06/...`` path, which is the *rendered* view. The
        machine-readable document sits beside it without the XSL prefix, so the directory listing
        is read rather than the path being constructed from a pattern that may change.
        """
        listing = self._fetch(self.ARCHIVE.format(cik=cik, accession=accession.replace("-", "")))
        candidates = [
            href for href in re.findall(r'href="([^"]+\.xml)"', listing)
            if "xsl" not in href.lower()
        ]
        if not candidates:
            return None
        href = candidates[0]
        return f"https://www.sec.gov{href}" if href.startswith("/") else href

    def trades(
        self, ticker: str, *, since: datetime, limit: int = 12
    ) -> tuple[list[InsiderTrade], list[str]]:
        """Recent Form 4 transactions, newest first, with a per-source status list."""
        status: list[str] = []
        cik = self._edgar.cik_for(ticker)
        if cik is None:
            return [], [f"insider:{ticker}: no CIK on EDGAR"]

        try:
            submissions = self._edgar._get(self._edgar.SUBMISSIONS_URL.format(cik=cik))
        except Exception as exc:  # network, JSON, or SEC rate limit
            return [], [f"insider:{ticker}: submissions unavailable ({type(exc).__name__})"]

        recent = submissions.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        out: list[InsiderTrade] = []
        looked = 0
        for i, form in enumerate(forms):
            if form != "4" or looked >= limit:
                continue
            try:
                accepted = datetime.fromisoformat(
                    recent["acceptanceDateTime"][i].replace("Z", "+00:00")
                )
            except (ValueError, KeyError, IndexError):
                continue
            if accepted < since:
                continue
            looked += 1
            accession = recent["accessionNumber"][i]
            try:
                url = self._document_url(cik, accession)
                if url is None:
                    status.append(f"insider:{accession}: no machine-readable document")
                    continue
                out.extend(
                    parse_form4(self._fetch(url), accepted_at=accepted, accession=accession)
                )
            except Exception as exc:
                status.append(f"insider:{accession}: unreadable ({type(exc).__name__})")

        out.sort(key=lambda t: t.accepted_at, reverse=True)
        status.append(f"insider:{ticker}: {looked} Form 4(s) read, {len(out)} transaction line(s)")
        return out, status

    def evidence(
        self, ticker: str, *, as_of: datetime, lookback: timedelta = timedelta(days=30)
    ) -> tuple[list[Evidence], list[str]]:
        """Informative insider transactions only, as evidence the event analyst can route on.

        The discarded count is reported rather than dropped. A feed that returns nothing because
        every filing was an RSU vest is a different state from a feed that returned nothing because
        it was down, and the desk must be able to tell them apart.
        """
        trades, status = self.trades(ticker, since=as_of - lookback)
        trades = [t for t in trades if t.accepted_at <= as_of]
        informative = [t for t in trades if t.is_informative]
        mechanical = len(trades) - len(informative)
        if mechanical:
            status.append(
                f"insider:{ticker}: {mechanical} line(s) discarded as compensation machinery "
                f"(grants, tax withholding, option exercises, pre-arranged plans)"
            )

        out = [
            Evidence(
                id=f"form4-{g.accession}-{g.code}",
                claim=g.render(),
                source="sec-edgar",
                available_at=g.accepted_at,
                credibility=1.0 if g.is_conviction else 0.6,
                # The structured record behind the sentence. `argus.agents.claims` settles a thesis
                # about pre-arrangement, direction or conviction against these fields; without them
                # the only thing a checker could compare is one model's prose against another's.
                attributes=g.as_dict(),
            )
            for g in aggregate(informative)
        ]
        if len(informative) > len(out):
            status.append(
                f"insider:{ticker}: {len(informative)} transaction line(s) aggregated into "
                f"{len(out)} decision(s); a sale filled across price tiers is one decision"
            )
        return out, status


__all__ = [
    "CODE_MEANING",
    "CONVICTION_CODES",
    "INFORMATIVE_CODES",
    "MAX_DOCUMENT_BYTES",
    "InsiderDecision",
    "InsiderSource",
    "InsiderTrade",
    "aggregate",
    "parse_form4",
]
