"""What management reported and guided, read from the company's own earnings release.

"Summarize NVDA's last earnings call and what management guided" was answered with an earnings date
and a SUE figure (a critic's probe, 2026-09-24): nothing in the console read what the company said.
Every US issuer files its earnings press release with the SEC as exhibit 99.1 to an 8-K under
item 2.02 ("results of operations") — the same numbers and, for companies that give it, the same
outlook management then discusses on the call. This module reads that exhibit, the company's own
filing. Call transcripts are available from keyless sources too (an earlier version of this
paragraph said they were not free; the rival review of 2026-09-24 found otherwise); they are not
read here yet, which is why the answer says it reads the release rather than the call.

**Extraction is deterministic and quoted.** Figures are taken only from sentences in the release,
and the outlook is returned as the release's own sentences, verbatim. Nothing is paraphrased by a
model, so a number in an answer can be found word for word in the filing it cites.

**The expectation gap is measured against the company's own guidance.** The prior release's revenue
outlook ("expected to be $X, plus or minus Y%") is compared with the revenue this release reports:
the one expectation that is public, dated and not a vendor's consensus. Where a company does not
guide, the answer says so rather than inventing a benchmark.
"""

from __future__ import annotations

import html
import re
import urllib.request
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from argus.market.evidence import EdgarSource

_UA = "ARGUS research desk research@argus-desk.example"
_TIMEOUT = 30.0
ARCHIVE = "https://www.sec.gov/Archives/edgar/data/{cik}/{folder}/"
_BILLION = re.compile(r"\$\s?([\d,]+(?:\.\d+)?)\s*(billion|million)", re.I)


@dataclass(frozen=True)
class Release:
    ticker: str
    filed: date
    accession: str
    url: str
    headline: str
    revenue: float | None            # in dollars
    revenue_sentence: str
    outlook: tuple[str, ...]
    guided_revenue: float | None     # midpoint, in dollars
    guided_band_pct: float | None
    extras: dict[str, str] = field(default_factory=dict)


def _fetch(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": _UA, "Accept": "*/*"})
    with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:
        return str(response.read(4_000_000).decode("utf-8", errors="replace"))


def clean(document: str) -> str:
    text = html.unescape(re.sub(r"<[^>]+>", " ", document))
    return re.sub(r"\s+", " ", text).strip()


def _dollars(amount: str, unit: str) -> float:
    return float(amount.replace(",", "")) * (1e9 if unit.lower() == "billion" else 1e6)


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z•“\"])|\s+•\s+", text)
    return [p.strip(" •") for p in parts if len(p.strip()) > 20]


def _prose(sentence: str) -> bool:
    """A real sentence, not a table-of-contents line, a heading or letter-spaced slide text."""
    if sentence.rstrip().endswith(":") or re.search(r"(?:\b[A-Z]\s){4,}", sentence):
        return False
    if re.search(r"\b\d{1,3}\s+[A-Z][a-z]+(?:\s+&\s+[A-Z][a-z]+)?\s+\d{1,3}\b", sentence):
        return False
    return len(sentence) < 600


def parse(text: str) -> dict[str, Any]:
    """Headline, reported revenue and the outlook from an earnings release's plain text."""
    sentences = _sentences(text)
    revenue, revenue_sentence = None, ""
    for sentence in sentences:
        if re.search(r"\b(?:total\s+)?revenue\b", sentence, re.I) and re.search(
                r"\bquarter\b|\breported\b|\bup\b|\bdown\b", sentence, re.I):
            found = _BILLION.search(sentence)
            if found:
                revenue, revenue_sentence = _dollars(*found.groups()), sentence
                break
    outlook: list[str] = []
    start = re.search(r"\b(?:Outlook|Guidance|Business Outlook|Financial Outlook)\b", text)
    if start:
        window = text[start.start(): start.start() + 2500]
        stop = re.search(r"\b(?:Highlights|CFO Commentary|Conference Call|Forward-Looking|"
                         r"About [A-Z])", window[20:])
        window = window[: stop.start() + 20] if stop else window
        outlook = [s for s in _sentences(window) if re.search(
            r"\bexpect|\bguid|\banticipat|\bproject|\bwill provide", s, re.I) and _prose(s)]
    guided, band = None, None
    for sentence in outlook:
        if re.search(r"\brevenue\b", sentence, re.I):
            found = _BILLION.search(sentence)
            if found:
                guided = _dollars(*found.groups())
                pm = re.search(r"plus or minus\s+([\d.]+)\s*%", sentence, re.I)
                band = float(pm.group(1)) if pm else None
                break
    headline = sentences[0][:300] if sentences else ""
    return {"headline": headline, "revenue": revenue, "revenue_sentence": revenue_sentence,
            "outlook": outlook[:6], "guided_revenue": guided, "guided_band_pct": band}


class EarningsReleaseSource:
    """Finds and reads a company's recent earnings releases on EDGAR."""

    def __init__(self, edgar: EdgarSource | None = None) -> None:
        self._edgar = edgar or EdgarSource(user_agent=_UA)

    def _exhibit_url(self, cik: int, accession: str) -> str | None:
        folder = accession.replace("-", "")
        index = _fetch(ARCHIVE.format(cik=cik, folder=folder) + f"{accession}-index.htm")
        for row in re.findall(r"<tr[^>]*>(.*?)</tr>", index, re.S):
            cells = [re.sub(r"<[^>]+>", "", c).strip()
                     for c in re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)]
            links = re.findall(r'href="([^"]+)"', row)
            if any(c.upper().startswith("EX-99.1") for c in cells) and links:
                href = links[0]
                return f"https://www.sec.gov{href}" if href.startswith("/") else href
        return None

    def releases(self, ticker: str, count: int = 2) -> list[Release]:
        """The latest ``count`` earnings releases, newest first."""
        cik = self._edgar.cik_for(ticker)
        if cik is None:
            return []
        recent = self._edgar._get(self._edgar.SUBMISSIONS_URL.format(cik=cik))
        rf = recent.get("filings", {}).get("recent", {})
        out: list[Release] = []
        for i, form in enumerate(rf.get("form", [])):
            if form != "8-K" or "2.02" not in (rf.get("items", [""] * (i + 1))[i] or ""):
                continue
            accession = rf["accessionNumber"][i]
            url = self._exhibit_url(cik, accession)
            if url is None:
                continue
            fields = parse(clean(_fetch(url)))
            out.append(Release(
                ticker=ticker.upper(), filed=date.fromisoformat(rf["filingDate"][i]),
                accession=accession, url=url, headline=fields["headline"],
                revenue=fields["revenue"], revenue_sentence=fields["revenue_sentence"],
                outlook=tuple(fields["outlook"]), guided_revenue=fields["guided_revenue"],
                guided_band_pct=fields["guided_band_pct"]))
            if len(out) >= count:
                break
        return out


def _money(value: float) -> str:
    return f"${value / 1e9:,.1f}bn" if value >= 1e9 else f"${value / 1e6:,.0f}m"


def answer_lines(ticker: str, source: EarningsReleaseSource | None = None) -> list[str]:
    """The console's lines: what was reported, against the company's own prior guidance, and what
    it guides now — every figure from the releases, each cited."""
    found = (source or EarningsReleaseSource()).releases(ticker, count=2)
    if not found:
        return []
    latest = found[0]
    prior = found[1] if len(found) > 1 else None
    lines: list[str] = []
    lead = (f"Actionable: no call transcript is read here — {ticker}'s own earnings release "
            f"(8-K exhibit 99.1, filed {latest.filed:%d %b %Y})")
    if latest.revenue is not None:
        lead += f" reports revenue of {_money(latest.revenue)}"
        if prior is not None and prior.guided_revenue:
            gap = latest.revenue / prior.guided_revenue - 1
            band = (f" ± {prior.guided_band_pct:g}%" if prior.guided_band_pct is not None else "")
            verdict = ("above the top of" if prior.guided_band_pct is not None
                       and gap * 100 > prior.guided_band_pct else
                       "below the bottom of" if prior.guided_band_pct is not None
                       and gap * 100 < -prior.guided_band_pct else "within")
            lead += (f", {gap:+.1%} against the {_money(prior.guided_revenue)}{band} it guided "
                     f"a quarter earlier — {verdict} its own range")
    if latest.guided_revenue is not None:
        lead += f"; for next quarter it guides {_money(latest.guided_revenue)}"
        if latest.guided_band_pct is not None:
            lead += f" ± {latest.guided_band_pct:g}%"
        if latest.revenue:
            lead += f" ({latest.guided_revenue / latest.revenue - 1:+.1%} on this quarter)"
    elif latest.outlook == ():
        lead += "; the release gives no forward outlook"
    if latest.revenue is None:
        lead += " — its revenue is not stated in a sentence this reader can quote; open the source"
    lines.append(lead + ".")
    if latest.revenue_sentence:
        lines.append(f"Reported, in the release's words: “{latest.revenue_sentence[:400]}”")
    for sentence in latest.outlook[:4]:
        lines.append(f"Outlook, in the release's words: “{sentence[:300]}”")
    lines.append(f"Source: {latest.url}")
    return lines


__all__ = ["EarningsReleaseSource", "Release", "answer_lines", "clean", "parse"]
