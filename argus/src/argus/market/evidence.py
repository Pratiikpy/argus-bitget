"""Evidence feeds — what the desk has been missing.

The paper desk abstained on every one of its first eighteen decisions, and its own thesis said why:
*"no catalyst, no hedge, consensus neutral — the -3.54% move is a price observation, not
actionable information."* The runner fed it price facts only. The event, sentiment and earnings
analysts had never been shown an event. A desk that abstains on no catalyst is behaving correctly;
the missing piece was never the hurdle. It was this module.

Five sources, in order of trust:

1. **SEC EDGAR** — the primary event feed. Public API, no key, and every filing carries
   ``acceptanceDateTime``: the exact instant the fact became public, which is the timestamp the
   as-of store needs and the one most feeds do not provide. 8-K items are exposed so the event
   analyst can tell a 2.02 (results) from a 5.02 (an officer leaving).
2. **Direct RSS** — nine outlets verified to return live items with ``pubDate`` on 2026-09-12,
   plus a per-symbol Yahoo Finance feed. Two candidates (WSJ Markets, Blockworks) were dropped
   because their newest item was months old; a feed that is alive but stale is worse than an
   absent one, since it would enter the record with a timestamp that looks current.
3. **Bitget's own skill server** — ``datahub.noxiaohao.com/mcp``, sanctioned (hardcoded at
   ``bitget-signal/scripts/install.js:30``), keyless, nineteen tools. On probe **every
   upstream-fetching tool returned empty data**. It is integrated anyway, because it is the
   ecosystem's own reference set and a real integration counts — but behind a client that
   **degrades visibly**. An empty response becomes a note in the evidence, never a silent gap.
4. **Twitter/X**, via ``agent-reach``'s real ``twitter-cli`` backend — opt-in (see
   :class:`TwitterSource`), because it is a real subprocess call to an external CLI and LIVE
   ONLY. Added 2026-09-16 after re-checking a real, previously-disclosed blocker
   ("the free replacements are blocked from this network," ``eval/standing.py``'s "Market
   sentiment" capability) and finding it stale: this machine's real network access to Twitter/X
   now works, verified with a real query before writing a line of integration code.
5. **Reddit**, via ``agent-reach``'s real ``rdt-cli`` backend — same shape as Twitter above (see
   :class:`RedditSource`): opt-in, LIVE ONLY. ``agent-reach doctor`` reported this channel
   ``status: error`` (a cookie-refresh warning, not a real failure); a direct real query found it
   working regardless, so the doctor summary was checked against the real thing rather than
   trusted on its own.

**The two-clock discipline is unchanged.** Every item enters with ``available_at`` set to its
publish time, and :func:`gather` refuses anything stamped after ``as_of``. A feed cannot leak the
future into a decision by being fast.

**Routing.** ``Evidence.source`` is the routing key the desk already uses
(``desk.py``: ``sec-edgar``/``news``/``macro`` → event analyst; ``social`` → sentiment;
``filing``/``transcript`` → earnings). This module emits those strings and nothing else, so the
analysts that were starved are the ones that get fed.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from enum import StrEnum
from typing import Any

from argus.market.rpc import LABELS, ErrorKind, JsonRpcClient, RpcError, ToolResult, payload_failure
from argus.truth.evidence import Evidence

# The SEC requires a User-Agent that identifies the requester and rejects anonymous clients (403).
# The contact is read from the environment so no personal address is committed to source.
#
# **The fallback used to contain the repository URL, and that was a live outage.** SEC's filter
# rejects any User-Agent carrying a bare domain: probed on 2026-09-13,
# "ARGUS research (github.com/Pratiikpy/argus-bitget)" and the same string without parentheses both
# returned 403, while "ARGUS research", "ARGUS-research/0.1" and "ARGUS research
# contact@argus.invalid" all returned 200. It is the domain, not the missing email.
#
# Every EDGAR call was therefore failing — 8-K and 10-Q filings, Form 4 insider transactions and
# XBRL fundamentals — and the live cycle had been recording `edgar: unavailable (HTTPError)` in its
# status for as long as nobody read it. The pipeline reported the outage correctly and no one was
# looking, which is the failure mode this project builds status lines to prevent.
#
# **It broke again, the same way, and the fallback was the cause.** Probed 2026-09-13 against both
# SEC hosts with six agents:
#
#   "ARGUS research"                              www.sec.gov 403   data.sec.gov 200
#   "ARGUS research contact@argus.invalid"        www.sec.gov 200   data.sec.gov 200
#   "ARGUS research desk (contact@argus.invalid)" www.sec.gov 200   data.sec.gov 200
#   "ARGUS-research/0.1"                          www.sec.gov 200   data.sec.gov 200
#
# The two hosts filter differently. `data.sec.gov` accepts the bare string, `www.sec.gov` does not,
# and `www.sec.gov` is where `company_tickers.json` lives — so with no contact email set, the CIK
# lookup 403s and *every* EDGAR path fails at its first step while the submissions endpoint keeps
# answering, which is why the earlier fix looked complete. A partial outage is harder to see than a
# total one.
#
# The fallback therefore carries a contact token. `argus.invalid` is a reserved TLD (RFC 2606): it
# cannot receive mail and names no real person, so it satisfies SEC's fair-access filter without
# publishing anyone's address. Set ARGUS_CONTACT_EMAIL in a deployment to be a better citizen.
# `tests/test_evidence.py` pins the shape so a future edit cannot quietly remove the token again.
_FALLBACK_CONTACT = "contact@argus.invalid"
_UA = (
    f"ARGUS research {os.environ.get('ARGUS_CONTACT_EMAIL', '')}".strip()
    if os.environ.get("ARGUS_CONTACT_EMAIL")
    else f"ARGUS research desk ({_FALLBACK_CONTACT})"
)
_TIMEOUT = 15


class EvidenceError(RuntimeError):
    """A source was asked for something it cannot supply honestly."""


# --- symbols -----------------------------------------------------------------------------------

def underlying_ticker(symbol: str) -> str:
    """``NVDAUSDT`` → ``NVDA``. The anchor the token tracks, and the name filings are under."""
    s = symbol.upper()
    for suffix in ("USDT", "USDC", "USD"):
        if s.endswith(suffix) and len(s) > len(suffix):
            return s[: -len(suffix)]
    return s


# --- SEC EDGAR ---------------------------------------------------------------------------------

EVENT_FORMS = frozenset({"8-K", "8-K/A"})
FILING_FORMS = frozenset({"10-Q", "10-K", "10-Q/A", "10-K/A"})

# The complete 8-K item taxonomy, transcribed from the SEC's own Form 8-K instructions
# (https://www.sec.gov/files/form8-k.pdf, fetched 2026-09-13). Thirty-three items.
#
# This table held twelve of them until that fetch. The twenty-one missing ones were not obscure:
# 1.03 bankruptcy, 1.05 material cybersecurity incident, 2.04 an accelerated debt obligation, 2.06
# a material impairment, 3.01 a delisting notice and 4.01 an auditor change were all invisible to
# the event analyst, which saw a bare code it had no label for. A filing the desk cannot name is a
# filing the desk cannot weigh.
ITEM_LABELS = {
    "1.01": "material agreement",
    "1.02": "termination of material agreement",
    "1.03": "bankruptcy or receivership",
    "1.04": "mine safety shutdown",
    "1.05": "material cybersecurity incident",
    "2.01": "acquisition or disposition completed",
    "2.02": "results of operations",
    "2.03": "new direct financial obligation",
    "2.04": "triggering event accelerating an obligation",
    "2.05": "exit or disposal costs",
    "2.06": "material impairment",
    "3.01": "delisting notice or listing-rule failure",
    "3.02": "unregistered equity sale",
    "3.03": "material modification to shareholder rights",
    "4.01": "change of certifying accountant",
    "4.02": "non-reliance on prior financials",
    "5.01": "change in control",
    "5.02": "officer or director change",
    "5.03": "charter, bylaw or fiscal-year change",
    "5.04": "trading suspension under an employee benefit plan",
    "5.05": "code of ethics amendment or waiver",
    "5.06": "change in shell company status",
    "5.07": "shareholder vote",
    "5.08": "shareholder director nominations",
    "6.01": "ABS informational material",
    "6.02": "change of servicer or trustee",
    "6.03": "change in credit enhancement",
    "6.04": "failure to make a required distribution",
    "6.05": "Securities Act updating disclosure",
    "6.06": "static pool",
    "7.01": "Regulation FD disclosure",
    "8.01": "other events",
    "9.01": "financial statements and exhibits",
}


class Materiality(StrEnum):
    """How much a filing's item code says on its own, before anyone reads the document.

    The distinction the desk was missing entirely: a 9.01 exhibit list and a 4.02 non-reliance on
    previously issued financial statements arrived as the same kind of object. The second is a
    restatement — among the strongest negative signals in equities — and it was being handed to the
    analyst with no more weight than a list of attachments.

    The tiers are a property of the *code*, not of the document. That is the limit of what a code
    can establish and the class says so rather than implying it read the filing.
    """

    SEVERE = "severe"
    """Repricing on its own. A reader who saw only this code would already act."""

    MATERIAL = "material"
    """Usually moves price, but the direction needs the document."""

    ROUTINE = "routine"
    """Procedural. Absence of news rather than news."""

    OPAQUE = "opaque"
    """A catch-all whose contents the code does not constrain.

    7.01 and 8.01 exist precisely so a company can disclose something that fits nowhere else, so
    they carry anything from a conference schedule to a CEO's resignation. Classifying them as
    ROUTINE would be the comfortable reading and is wrong; classifying them as MATERIAL would cry
    wolf on every investor-day announcement. They get their own name, meaning **the code tells you
    nothing and the document must be read**."""


ITEM_MATERIALITY: dict[str, Materiality] = {
    # Severe: each is, by itself, a reason to reprice.
    "1.03": Materiality.SEVERE,   # bankruptcy
    "1.05": Materiality.SEVERE,   # cybersecurity incident — mandatory since the 2023 rule
    "2.04": Materiality.SEVERE,   # debt acceleration; a covenant breach reaching the surface
    "2.06": Materiality.SEVERE,   # material impairment
    "3.01": Materiality.SEVERE,   # delisting notice
    "4.01": Materiality.SEVERE,   # auditor change — frequently precedes 4.02
    "4.02": Materiality.SEVERE,   # non-reliance: the prior numbers were wrong
    "5.01": Materiality.SEVERE,   # change in control
    # Material: moves price, direction needs reading.
    "1.01": Materiality.MATERIAL,
    "1.02": Materiality.MATERIAL,
    "2.01": Materiality.MATERIAL,
    "2.02": Materiality.MATERIAL,  # earnings
    "2.03": Materiality.MATERIAL,
    "2.05": Materiality.MATERIAL,
    "3.02": Materiality.MATERIAL,
    "3.03": Materiality.MATERIAL,
    "5.02": Materiality.MATERIAL,  # a CEO departure and a board appointment share this code
    "5.06": Materiality.MATERIAL,
    # Opaque: the code constrains nothing.
    "7.01": Materiality.OPAQUE,
    "8.01": Materiality.OPAQUE,
    # Routine: procedural.
    "1.04": Materiality.ROUTINE,
    "5.03": Materiality.ROUTINE,
    "5.04": Materiality.ROUTINE,
    "5.05": Materiality.ROUTINE,
    "5.07": Materiality.ROUTINE,
    "5.08": Materiality.ROUTINE,
    "6.01": Materiality.ROUTINE,
    "6.02": Materiality.ROUTINE,
    "6.03": Materiality.ROUTINE,
    "6.04": Materiality.ROUTINE,
    "6.05": Materiality.ROUTINE,
    "6.06": Materiality.ROUTINE,
    "9.01": Materiality.ROUTINE,
}


def materiality_of(items: Sequence[str]) -> Materiality:
    """The strongest claim this filing's codes support.

    Maximum rather than average: a filing carrying both a restatement and an exhibit list is a
    restatement. Averaging would let a company dilute a severe item by attaching routine ones,
    which is a thing that actually happens.

    An unknown code is OPAQUE, never ROUTINE. The SEC adds items — 1.05 arrived in 2023 — and a
    table written today will be incomplete tomorrow, so the unknown case must mean "go and read
    it", not "nothing to see".
    """
    if not items:
        return Materiality.ROUTINE
    order = (Materiality.SEVERE, Materiality.MATERIAL, Materiality.OPAQUE, Materiality.ROUTINE)
    seen = [ITEM_MATERIALITY.get(i.strip(), Materiality.OPAQUE) for i in items if i.strip()]
    if not seen:
        return Materiality.ROUTINE
    for level in order:
        if level in seen:
            return level
    return Materiality.ROUTINE


@dataclass(frozen=True, slots=True)
class Filing:
    form: str
    filed: datetime
    accepted: datetime
    accession: str
    items: tuple[str, ...]
    description: str

    @property
    def is_event(self) -> bool:
        return self.form in EVENT_FORMS

    @property
    def is_periodic(self) -> bool:
        return self.form in FILING_FORMS

    @property
    def item_summary(self) -> str:
        labelled = [f"{i} {ITEM_LABELS[i]}" if i in ITEM_LABELS else i for i in self.items]
        return ", ".join(labelled) if labelled else "no items listed"

    @property
    def materiality(self) -> Materiality:
        """What the item codes establish on their own. Only meaningful for an 8-K.

        A periodic report has no item codes, and calling a 10-K routine because it carries none
        would be exactly backwards — so the periodic forms are reported as MATERIAL, which is what
        a full financial statement is.
        """
        if self.is_periodic:
            return Materiality.MATERIAL
        return materiality_of(self.items)

    @property
    def demands_a_read(self) -> bool:
        """True when the code alone cannot settle it: either severe, or a catch-all."""
        return self.materiality in (Materiality.SEVERE, Materiality.OPAQUE)


class EdgarSource:
    """SEC EDGAR submissions, keyed on the anchor ticker.

    Two public endpoints, both keyless and both requiring a descriptive ``User-Agent`` (the SEC
    blocks anonymous clients):

    * ``https://www.sec.gov/files/company_tickers.json`` — ticker → CIK
    * ``https://data.sec.gov/submissions/CIK{cik:010d}.json`` — recent filings with
      ``acceptanceDateTime``

    The CIK map is fetched once and cached for the process; it changes rarely.
    """

    TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
    SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"

    def __init__(self, *, user_agent: str = _UA) -> None:
        self._ua = user_agent
        self._ciks: dict[str, int] | None = None

    def _get(self, url: str) -> Any:
        headers = {"User-Agent": self._ua, "Accept": "application/json"}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
            return json.loads(r.read().decode())

    def cik_for(self, ticker: str) -> int | None:
        if self._ciks is None:
            raw = self._get(self.TICKERS_URL)
            self._ciks = {v["ticker"].upper(): int(v["cik_str"]) for v in raw.values()}
        return self._ciks.get(ticker.upper())

    def filings(self, ticker: str, *, since: datetime, limit: int = 20) -> list[Filing]:
        """Recent 8-K / 10-Q / 10-K filings accepted after ``since``, newest first."""
        cik = self.cik_for(ticker)
        if cik is None:
            return []
        sub = self._get(self.SUBMISSIONS_URL.format(cik=cik))
        rf = sub.get("filings", {}).get("recent", {})
        out: list[Filing] = []
        n = len(rf.get("form", []))
        for i in range(n):
            form = rf["form"][i]
            if form not in EVENT_FORMS and form not in FILING_FORMS:
                continue
            accepted = _parse_iso(rf["acceptanceDateTime"][i])
            if accepted is None or accepted < since:
                continue
            filed = _parse_iso(rf["filingDate"][i] + "T00:00:00Z") or accepted
            raw_items = (rf.get("items", [""] * n)[i] or "").split(",")
            items = tuple(s.strip() for s in raw_items if s.strip())
            desc = (rf.get("primaryDocDescription", [""] * n)[i] or "").strip()
            out.append(Filing(form, filed, accepted, rf["accessionNumber"][i], items, desc))
            if len(out) >= limit:
                break
        return out

    def evidence(self, symbol: str, *, as_of: datetime, lookback: timedelta) -> list[Evidence]:
        ticker = underlying_ticker(symbol)
        out: list[Evidence] = []
        for f in self.filings(ticker, since=as_of - lookback):
            if f.accepted > as_of:
                continue  # the as-of rule, applied at the source as well as at the gate
            level = f.materiality
            if f.is_event:
                # The materiality travels in the claim text, not only in an attribute. The analyst
                # reads prose, and a code it cannot weigh is a code it will average away.
                claim = f"{ticker} filed {f.form} [{level.value.upper()}] — {f.item_summary}"
                if level is Materiality.SEVERE:
                    claim += (
                        ". This item class reprices on its own; treat it as the dominant fact "
                        "unless the document says otherwise."
                    )
                elif level is Materiality.OPAQUE:
                    claim += (
                        ". This is a catch-all item: the code constrains nothing about the "
                        "contents, so it is neither news nor the absence of news until read."
                    )
                source = "sec-edgar"
            else:
                suffix = f" — {f.description}" if f.description else ""
                claim = f"{ticker} filed {f.form}{suffix}"
                source = "filing"
            out.append(Evidence(
                id=f"edgar-{f.accession}",
                claim=claim,
                source=source,
                available_at=f.accepted,
                credibility=1.0,  # a filing is the issuer's own statement under liability
                attributes={
                    "materiality": level.value,
                    "items": ",".join(f.items),
                    "demands_a_read": str(f.demands_a_read).lower(),
                },
            ))
        return out


# --- RSS ---------------------------------------------------------------------------------------

# Verified 2026-09-12: each returned live items with a parseable publish time. Feeds whose newest
# item was months old (WSJ Markets, Blockworks) are deliberately absent.
RSS_FEEDS: dict[str, tuple[str, str]] = {
    # key: (url, routing source)
    "coindesk":      ("https://www.coindesk.com/arc/outboundfeeds/rss/", "news"),
    "cointelegraph": ("https://cointelegraph.com/rss", "news"),
    "decrypt":       ("https://decrypt.co/feed", "news"),
    "theblock":      ("https://www.theblock.co/rss.xml", "news"),
    "cnbc":          ("https://www.cnbc.com/id/100003114/device/rss/rss.html", "news"),
    "marketwatch":   ("https://feeds.content.dowjones.io/public/rss/mw_topstories", "news"),
    "fed":           ("https://www.federalreserve.gov/feeds/press_all.xml", "macro"),
    "sec-press":     ("https://www.sec.gov/news/pressreleases.rss", "macro"),
}
YAHOO_SYMBOL_FEED = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"

_ATOM = "{http://www.w3.org/2005/Atom}"
_DC = "{http://purl.org/dc/elements/1.1/}"


@dataclass(frozen=True, slots=True)
class Headline:
    feed: str
    title: str
    link: str
    published: datetime


class RssSource:
    """Direct RSS/Atom, with the publish time carried through.

    Each outlet's feed is the same one Bitget's ``news_feed`` tool aggregates; pulling it directly
    removes the dependency on their server without changing what a reader would see.
    """

    def __init__(
        self, feeds: dict[str, tuple[str, str]] | None = None, *, user_agent: str = _UA
    ) -> None:
        self._feeds = feeds if feeds is not None else RSS_FEEDS
        self._ua = user_agent

    def _fetch(self, url: str) -> bytes:
        req = urllib.request.Request(url, headers={"User-Agent": self._ua})
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
            return bytes(r.read())

    def headlines(self, key: str, url: str) -> list[Headline]:
        root = ET.fromstring(self._fetch(url))
        items = root.findall(".//item") or root.findall(f".//{_ATOM}entry")
        out: list[Headline] = []
        for it in items:
            title = _text(it, "title") or _text(it, f"{_ATOM}title") or ""
            link = _text(it, "link") or ""
            if not link:
                a = it.find(f"{_ATOM}link")
                link = a.get("href", "") if a is not None else ""
            pub = None
            for tag in ("pubDate", f"{_DC}date", f"{_ATOM}published", f"{_ATOM}updated"):
                pub = _parse_any(_text(it, tag))
                if pub:
                    break
            if not title or pub is None:
                continue  # an item with no publish time cannot be placed on the clock
            out.append(Headline(key, title.strip(), link.strip(), pub))
        return out

    def evidence(
        self, symbol: str, *, as_of: datetime, lookback: timedelta, keywords: Iterable[str] = ()
    ) -> tuple[list[Evidence], list[str]]:
        """Headlines mentioning the symbol's underlying, plus a per-feed status list.

        Returns the status alongside the evidence so a caller can *show* which feeds answered —
        the visible-error-recovery trait, made mandatory rather than optional.
        """
        ticker = underlying_ticker(symbol)
        words = {ticker.lower(), *[k.lower() for k in keywords]}
        pattern = re.compile(r"\b(" + "|".join(re.escape(w) for w in sorted(words)) + r")\b", re.I)
        since = as_of - lookback

        feeds = dict(self._feeds)
        feeds[f"yahoo-{ticker.lower()}"] = (YAHOO_SYMBOL_FEED.format(ticker=ticker), "news")

        out: list[Evidence] = []
        status: list[str] = []
        for key, (url, source) in feeds.items():
            try:
                hs = self.headlines(key, url)
            except (urllib.error.URLError, ET.ParseError, TimeoutError, OSError) as exc:
                status.append(f"{key}: unavailable ({type(exc).__name__})")
                continue
            kept = 0
            for h in hs:
                if h.published > as_of or h.published < since:
                    continue
                if not key.startswith("yahoo") and not pattern.search(h.title):
                    continue
                out.append(Evidence(
                    id=f"rss-{key}-{abs(hash(h.link or h.title)) % 10**10}",
                    claim=f"{h.title} ({key})",
                    source=source,
                    available_at=h.published,
                    credibility=0.7,  # secondary reporting; a filing is 1.0
                ))
                kept += 1
            status.append(f"{key}: {len(hs)} items, {kept} relevant")
        return out, status


# --- Bitget skill server -----------------------------------------------------------------------

class BitgetSkillSource:
    """Bitget's sanctioned MCP server, wrapped so it can fail loudly.

    Keyless, nineteen tools, and Cloudflare rejects a bare Python User-Agent. An earlier note here
    said "every upstream-fetching tool returned empty"; re-probed on 2026-09-13 that is too strong —
    ``technical_analysis`` returns real readings — and the rest is measured tool by tool in
    :mod:`argus.market.skills` rather than asserted in a docstring.

    **The response-correlation bug this class used to have.** Every call was sent with the literal
    JSON-RPC id ``9`` and the reply was taken as the *last* SSE frame in the body. JSON-RPC uses the
    id to match a response to its request, and with one id for every call there was nothing to match
    on: a live probe had ``sentiment_index`` come back holding ``defi_analytics``'s payload and
    ``derivatives_sentiment`` return the string ``Error executing tool cross_asset``. Evidence
    attributed to the wrong tool is worse than no evidence, because its provenance reads as sound.
    Ids are now monotonic per instance and the frame is selected by id — a reply that does not match
    is refused rather than returned.
    """

    URL = "https://datahub.noxiaohao.com/mcp"

    def __init__(self, *, user_agent: str = "claude-code/2.0") -> None:
        self._ua = user_agent
        # The shared client (`market/rpc.py`, 2026-09-25): ids are monotonic per instance and the
        # reply frame is chosen by id, which is the fix described above, now in one place for both
        # Bitget servers. Timeouts are not retried here: `market/skills.py` measured the slow tools
        # at 15-31 s every time, so a tool that missed a 10 s deadline once misses it twice, and a
        # live answer would pay for both. A rate limit, a 5xx or a dropped connection is retried.
        self.client = JsonRpcClient(
            self.URL, headers={"User-Agent": user_agent}, timeout=_TIMEOUT,
            retry_timeouts=False, client_info={"name": "argus", "version": "0.1"},
        )

    def call_tool(self, tool: str, args: dict[str, Any], *,
                  timeout: int = _TIMEOUT) -> ToolResult:
        """One tool call, typed: a protocol or transport failure raises :class:`RpcError` with its
        :class:`~argus.market.rpc.ErrorKind`; a failure inside the tool comes back in the result
        (:meth:`ToolResult.failure`). What ``eval/skill_matrix.py`` counts."""
        return self.client.call_tool(tool, args, timeout=timeout)

    def call(self, tool: str, args: dict[str, Any], *, timeout: int = _TIMEOUT) -> tuple[Any, str]:
        """Invoke one tool. Returns ``(payload, status)``; never raises on an empty payload.

        The status wording is a contract `market/skills.py:_classify` reads, so every typed failure
        is rendered into the phrase that already means the same thing there — and the kind is named
        inside the parentheses, so a reader of the status sees which failure it was. One behaviour
        changed on purpose: a JSON-RPC ``error`` frame (an unknown tool, a missing session) used to
        fall through as an empty payload and read "reachable, returned no data". It is a protocol
        error and now says so.
        """
        try:
            result = self.call_tool(tool, args, timeout=timeout)
        except RpcError as exc:
            if exc.kind is ErrorKind.CORRELATION:
                # A mismatched reply is a correctness failure, not a network one, and is named so.
                return None, f"bitget:{tool}: response correlation failed ({exc})"
            if exc.kind is ErrorKind.TIMEOUT:
                return None, f"bitget:{tool}: unavailable (TimeoutError)"
            return None, f"bitget:{tool}: unavailable ({exc.label}: {str(exc)[:120]})"
        # The first content block, as before the migration: both servers put the payload there.
        text = str(result.content[0].get("text", "")) if result.content else ""
        # Bitget's own MCP server sets `isError` on a failed call and keeps the payload in the
        # text channel (`agent-mcp/src/server.ts:119-124`). Reading the text without checking
        # the flag turns "Error executing tool cross_asset" into a piece of evidence.
        if result.is_error:
            return None, f"bitget:{tool}: tool reported an error ({text[:120]})"
        try:
            payload: Any = json.loads(text)
        except json.JSONDecodeError:
            payload = text
        if _is_empty(payload):
            failed = payload_failure(payload)
            why = f" ({LABELS[failed[0]]}: {failed[1][:100]})" if failed else ""
            return payload, f"bitget:{tool}: reachable, returned no data{why}"
        return payload, f"bitget:{tool}: ok"

    def evidence(self, symbol: str, *, as_of: datetime) -> tuple[list[Evidence], list[str]]:
        """Fear & Greed as a dated social fact, when the server has it."""
        payload, status = self.call("sentiment_index", {"action": "current"})
        out: list[Evidence] = []
        if isinstance(payload, dict) and "value" in payload:
            out.append(Evidence(
                id=f"bitget-fng-{int(as_of.timestamp())}",
                claim=(
                    f"Crypto Fear & Greed index {payload.get('value')} "
                    f"({payload.get('value_classification', '')})"
                ),
                source="social",
                # A snapshot at fetch time; the index carries no earlier stamp.
                available_at=as_of,
                credibility=0.6,
            ))
        return out, [status]


def _utf8_subprocess_env() -> dict[str, str]:
    """The environment `TwitterSource`/`RedditSource` launch their real CLI subprocess with.

    **Two real encoding bugs, both found the same way (running it, not assumed safe), on the
    CHILD's own side rather than this process's.** Both `rdt-cli` and (potentially) `twitter-cli`
    are themselves Python programs; on Windows, anything they write without an explicit encoding
    — stdout via `click.echo`, or a local cache file via `Path.write_text()` — inherits the
    platform default (cp1252), not UTF-8. Both were caught crashing on real data: `rdt-cli`'s own
    `click.echo(json.dumps(...))` on a real post containing ``≈``, and separately its own
    `index_cache.py::save_index()` writing a local cache file on a real post containing an emoji
    (``\U0001f4c8``) — two different real code paths inside the same real dependency, hitting the
    same underlying platform-default-encoding class of bug. `PYTHONIOENCODING=utf-8` alone fixed
    the first (it only governs stdin/stdout/stderr); the second needed the broader
    `PYTHONUTF8=1` interpreter mode, which makes UTF-8 the default text encoding everywhere in
    the child interpreter, including `open()`/`Path.write_text()` calls with no explicit
    encoding. Both are set, since either code path can be hit depending on what a search returns.
    """
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


# --- Twitter/X ------------------------------------------------------------------------------

class TwitterSource:
    """Real social-sentiment text via the `agent-reach` router's real `twitter-cli` backend.

    **Why this exists now, and did not before.** The `BitgetSkillSource` above is the ecosystem's
    own sanctioned feed, and its own `sentiment_index` tool is a crypto-wide Fear & Greed number,
    not per-symbol text — `agents/analysts.py`'s own real, disclosed blocker on the "Market
    sentiment" capability (`eval/standing.py`, `t2-sentiment`) says social evidence for tokenised
    equities is empty ~93% of the time and names the reason: "the free replacements are blocked
    from this network." Re-checked directly on 2026-09-16, not carried forward from that earlier
    finding: `agent-reach doctor --json` now reports Twitter/X `status: ok`, backend `twitter-cli`
    — a real change in this machine's own network access since that blocker was written, verified
    with a real live query (`twitter search NVDA --json`) returning real, current, per-symbol
    tweets with real engagement metrics and real ISO timestamps.

    **LIVE ONLY — not a point-in-time replay source.** Every other source in this module accepts
    an `as_of` in the past and returns only what was knowable by then. Twitter's real search API
    has no such parameter — a query returns whatever is CURRENTLY indexed, not a historical
    snapshot as of an arbitrary past date. `evidence()` still enforces the `available_at <= as_of`
    gate as a safety net (a tweet whose own real timestamp is after `as_of` is dropped), but
    calling this with an `as_of` meaningfully in the past will simply starve it, not backfill it
    correctly — this source belongs in live gathering, never in a backtest/paper-replay path.

    **Credibility set low, deliberately.** Unverified social text is the least trustworthy source
    this module carries — lower than the Bitget Fear & Greed snapshot (0.6) and far below EDGAR
    filings (1.0). `agents/analysts.py`'s own coordination-defense finding (repetition does not
    raise actionable-ness) is exactly the property this feed needs a caller that respects it.
    """

    CLI = "twitter"

    def __init__(self, *, cli: str = CLI, timeout: float = 20.0) -> None:
        self._cli = cli
        self._timeout = timeout

    def evidence(
        self, symbol: str, *, as_of: datetime, max_tweets: int = 10,
    ) -> tuple[list[Evidence], list[str]]:
        query = underlying_ticker(symbol)
        try:
            # A real bug, found by running this on a real tweet, not assumed safe: `text=True`
            # with no explicit `encoding` decodes with the platform default (cp1252 on Windows),
            # which cannot represent real tweet content carrying emoji or other non-Latin-1
            # characters and crashed the subprocess's own stdout reader thread mid-read, leaving
            # `result.stdout` as `None`. UTF-8 is what the real CLI actually emits.
            result = subprocess.run(
                [self._cli, "search", query, "-n", str(max_tweets), "--json", "--lang", "en"],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=self._timeout, check=True, env=_utf8_subprocess_env(),
            )
        except FileNotFoundError:
            return [], ["twitter: unavailable (cli not installed)"]
        except (OSError, subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
            return [], [f"twitter: unavailable ({type(exc).__name__})"]

        try:
            payload = json.loads(result.stdout)
        except (json.JSONDecodeError, TypeError):
            return [], ["twitter: unavailable (unparseable output)"]

        if not payload.get("ok"):
            return [], ["twitter: reachable, ok=false"]

        tweets = payload.get("data") or []
        out: list[Evidence] = []
        for tweet in tweets:
            stamp = tweet.get("createdAtISO")
            if not isinstance(stamp, str):
                continue
            try:
                created = datetime.fromisoformat(stamp)
            except ValueError:
                continue
            if created.tzinfo is None:
                created = created.replace(tzinfo=UTC)
            if created > as_of:
                continue
            text = str(tweet.get("text", "")).strip()
            if not text:
                continue
            author = tweet.get("author") or {}
            metrics = tweet.get("metrics") or {}
            out.append(Evidence(
                id=f"twitter-{tweet.get('id', created.timestamp())}",
                claim=text[:280],
                source="social",
                available_at=created,
                credibility=0.4,
                attributes={
                    "author": author.get("screenName", ""),
                    "verified": bool(author.get("verified", False)),
                    "likes": metrics.get("likes", 0),
                    "retweets": metrics.get("retweets", 0),
                },
            ))
        return out, [f"twitter: {len(out)} tweets"]


# --- Reddit ---------------------------------------------------------------------------------

class RedditSource:
    """Real social-sentiment text via the `agent-reach` router's real `rdt-cli` backend.

    **Found the same way `TwitterSource` was, same day.** `agent-reach doctor --json` reports
    Reddit `status: error` (a cookie-refresh warning, not a real failure) — read as `"off"` at a
    glance and it would have been wrong. A real, direct query (`rdt search NVDA --json`) returned
    real, current r/wallstreetbets posts with real titles, scores and usernames, so the doctor
    summary was checked against the real thing rather than trusted on its own.

    **LIVE ONLY**, same reasoning and same real constraint as `TwitterSource`: Reddit's real
    search has no point-in-time parameter, so this source belongs in live gathering only.

    **Credibility set slightly below Twitter's (0.35 vs 0.4), stated why.** Not a claim that
    Reddit is categorically less trustworthy than Twitter — both are unverified social text — but
    the real query above surfaced r/wallstreetbets specifically, a subreddit whose own real
    culture is explicitly meme/noise-heavy by design (`link_flair_text: "YOLO"` on the very first
    real result), and this module's own credibility numbers are meant to reflect what a caller
    should weigh a claim by, not just its source TYPE.
    """

    CLI = "rdt"

    def __init__(self, *, cli: str = CLI, timeout: float = 20.0) -> None:
        self._cli = cli
        self._timeout = timeout

    def evidence(
        self, symbol: str, *, as_of: datetime, max_posts: int = 10, newest: bool = False,
    ) -> tuple[list[Evidence], list[str]]:
        """``newest`` asks for the latest posts of the past week instead of the most relevant of
        all time. Relevance is the CLI's default, and for a live crowd read it returns threads
        weeks old — every one fell outside a 48-hour window (`market/social_pulse.py`,
        2026-09-25). `rdt-cli` takes ``--sort new --time week`` for this, as the Track 2 agent's
        own collector already does."""
        query = underlying_ticker(symbol)
        order = ["--sort", "new", "--time", "week"] if newest else []
        try:
            result = subprocess.run(
                [self._cli, "search", query, *order, "-n", str(max_posts), "--json"],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=self._timeout, check=True, env=_utf8_subprocess_env(),
            )
        except FileNotFoundError:
            return [], ["reddit: unavailable (cli not installed)"]
        except (OSError, subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
            return [], [f"reddit: unavailable ({type(exc).__name__})"]

        try:
            payload = json.loads(result.stdout)
        except (json.JSONDecodeError, TypeError):
            return [], ["reddit: unavailable (unparseable output)"]

        if not payload.get("ok"):
            return [], ["reddit: reachable, ok=false"]

        # The real listing shape rdt-cli passes through unmodified from Reddit's own API:
        # {"data": {"kind": "Listing", "data": {"children": [{"kind": "t3", "data": {...}}]}}}.
        listing = payload.get("data") or {}
        posts = ((listing.get("data") or {}).get("children")) or []
        out: list[Evidence] = []
        for post in posts:
            fields = post.get("data") or {}
            stamp = fields.get("created_utc")
            if not isinstance(stamp, (int, float)):
                continue
            try:
                created = datetime.fromtimestamp(stamp, tz=UTC)
            except (ValueError, OSError, OverflowError):
                continue
            if created > as_of:
                continue
            title = str(fields.get("title", "")).strip()
            if not title:
                continue
            selftext = str(fields.get("selftext", "")).strip()
            claim = f"{title} — {selftext}" if selftext else title
            out.append(Evidence(
                id=f"reddit-{fields.get('id', created.timestamp())}",
                claim=claim[:280],
                source="social",
                available_at=created,
                credibility=0.35,
                attributes={
                    "author": fields.get("author", ""),
                    "subreddit": fields.get("subreddit", ""),
                    "score": fields.get("score", 0),
                    "num_comments": fields.get("num_comments", 0),
                },
            ))
        return out, [f"reddit: {len(out)} posts"]


# --- the gather --------------------------------------------------------------------------------

@dataclass
class Gathered:
    evidence: list[Evidence]
    status: list[str] = field(default_factory=list)

    @property
    def catalysts(self) -> int:
        """Items that are events, not price. What the desk was missing."""
        return sum(1 for e in self.evidence if e.source in ("sec-edgar", "filing", "macro"))


def gather(
    symbol: str,
    *,
    as_of: datetime,
    lookback: timedelta = timedelta(hours=72),
    filing_lookback: timedelta = timedelta(days=7),
    edgar: EdgarSource | None = None,
    rss: RssSource | None = None,
    bitget: BitgetSkillSource | None = None,
    twitter: TwitterSource | None = None,
    reddit: RedditSource | None = None,
    insider: Any | None = None,
    insider_lookback: timedelta = timedelta(days=30),
    fundamentals: Any | None = None,
    keywords: Iterable[str] = (),
    max_headlines: int = 6,
) -> Gathered:
    """Everything knowable about ``symbol`` at ``as_of``, and nothing after it.

    Two windows on purpose. Headlines decay in hours; a 72h window keeps them current. Filings
    are sparse — a 14-day sweep of all twelve rTokens found eight 8-Ks — so ``filing_lookback``
    is a week, or a weekend cycle would see none and conclude there was never a catalyst.

    The status list is part of the result, not a log line: it is rendered into the evidence the
    desk sees, so a feed that went dark is a fact the decision-maker was told about.
    """
    if as_of.tzinfo is None:
        raise EvidenceError(
            "as_of must be timezone-aware; a naive instant cannot be placed on the clock"
        )
    edgar = edgar or EdgarSource()
    rss = rss or RssSource()
    bitget = bitget or BitgetSkillSource()

    evidence: list[Evidence] = []
    status: list[str] = []

    try:
        got = edgar.evidence(symbol, as_of=as_of, lookback=filing_lookback)
        evidence.extend(got)
        status.append(f"edgar: {len(got)} filings")
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError, KeyError) as exc:
        status.append(f"edgar: unavailable ({type(exc).__name__})")

    got, st = rss.evidence(symbol, as_of=as_of, lookback=lookback, keywords=keywords)
    evidence.extend(got)
    status.extend(st)

    got, st = bitget.evidence(symbol, as_of=as_of)
    evidence.extend(got)
    status.extend(st)

    # Real per-symbol social text, when a source is supplied. Opt-in for the same reason as
    # insider/fundamentals below — a real subprocess call to an external CLI should never fire
    # because a caller forgot to disable it, and it is LIVE ONLY (see `TwitterSource`'s own
    # docstring): a caller replaying a past `as_of` should not silently get today's tweets.
    if twitter is not None:
        try:
            got, st = twitter.evidence(symbol, as_of=as_of)
            evidence.extend(got)
            status.extend(st)
        except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
            status.append(f"twitter: unavailable ({type(exc).__name__})")

    # Same reasoning as twitter immediately above — opt-in, LIVE ONLY (see `RedditSource`'s own
    # docstring).
    if reddit is not None:
        try:
            got, st = reddit.evidence(symbol, as_of=as_of)
            evidence.extend(got)
            status.extend(st)
        except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
            status.append(f"reddit: unavailable ({type(exc).__name__})")

    # Insider transactions, when a source is supplied. A month rather than a week: Form 4 must be
    # filed within two business days of the trade, but the trades themselves are sparse for any one
    # name, and a purchase made three weeks ago is still the most recent thing that insider chose
    # to say.
    #
    # Unlike the three sources above this one does **not** default to a live client. Each Form 4
    # costs two more requests to SEC (a directory listing and a document) on top of the submissions
    # call, so a caller that has not asked for insider data should not pay for it — and a gather
    # that silently reaches the network because a caller forgot to disable it is a test hazard as
    # much as a cost one. `argus.paper.runner` supplies it; a unit test that does not, gets none.
    if insider is not None:
        try:
            got, st = insider.evidence(
                underlying_ticker(symbol), as_of=as_of, lookback=insider_lookback
            )
            evidence.extend(got)
            status.extend(st)
        except (urllib.error.URLError, TimeoutError, OSError,
                json.JSONDecodeError, KeyError) as exc:
            status.append(f"insider: unavailable ({type(exc).__name__})")

    # Reported financials, when a source is supplied. Opt-in for the same reason as the insider
    # feed: one request per concept, and a gather that reaches the network because a caller forgot
    # to disable it is a hazard rather than a convenience.
    if fundamentals is not None:
        try:
            got, st = fundamentals.evidence(underlying_ticker(symbol), as_of=as_of)
            evidence.extend(got)
            status.extend(st)
        except (urllib.error.URLError, TimeoutError, OSError,
                json.JSONDecodeError, KeyError) as exc:
            status.append(f"xbrl: unavailable ({type(exc).__name__})")

    # The gate. Sources apply it too, but this is the line a test can point at.
    leaked = [e for e in evidence if e.available_at > as_of]
    if leaked:
        raise EvidenceError(f"{len(leaked)} items stamped after as_of reached the gather; refusing")

    evidence.sort(key=lambda e: e.available_at, reverse=True)

    # Cost control, stated as such. Filings are sparse and always kept; headlines are capped at the
    # newest ``max_headlines`` after de-duplication by title, because an evidence-fed cycle cost
    # roughly twice a price-only one and the hackathon key has a finite balance. A desk shown six
    # current headlines and every filing has what it needs; shown thirty, it has a bigger bill.
    filings = [e for e in evidence if e.source in ("sec-edgar", "filing")]
    others: list[Evidence] = []
    seen: set[str] = set()
    for e in evidence:
        if e.source in ("sec-edgar", "filing"):
            continue
        key = e.claim.split(" (")[0].strip().lower()
        if key in seen:
            continue
        seen.add(key)
        others.append(e)
    social_tail = [e for e in others[max_headlines:] if e.source == "social"]
    kept = filings + others[:max_headlines] + social_tail
    kept.sort(key=lambda e: e.available_at, reverse=True)
    if len(kept) < len(evidence):
        status.append(f"cap: {len(evidence) - len(kept)} older/duplicate headlines withheld")
    return Gathered(evidence=kept, status=status)


# --- helpers -----------------------------------------------------------------------------------

def _text(el: ET.Element, tag: str) -> str:
    e = el.find(tag)
    return (e.text or "").strip() if e is not None and e.text else ""


def _parse_iso(s: str) -> datetime | None:
    try:
        d = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return d if d.tzinfo else d.replace(tzinfo=UTC)
    except (ValueError, AttributeError):
        return None


def _parse_any(s: str) -> datetime | None:
    if not s:
        return None
    d = _parse_iso(s)
    if d:
        return d.astimezone(UTC)
    try:
        d = parsedate_to_datetime(s)
        return (d if d.tzinfo else d.replace(tzinfo=UTC)).astimezone(UTC)
    except (TypeError, ValueError):
        return None


def _is_empty(payload: Any) -> bool:
    if payload in (None, "", [], {}):
        return True
    if isinstance(payload, dict):
        if set(payload) <= {"error", "alt_me_error"} and not any(payload.values()):
            return True
        if payload.get("error"):
            return True
    return isinstance(payload, list) and all(
        isinstance(x, dict) and not x.get("items") for x in payload
    )


__all__ = [
    "RSS_FEEDS",
    "BitgetSkillSource",
    "EdgarSource",
    "EvidenceError",
    "Filing",
    "Gathered",
    "Headline",
    "RssSource",
    "gather",
    "underlying_ticker",
]
