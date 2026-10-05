"""Shareholder returns and valuation for one to four US-listed companies, compared from their own
filings: dividends, payout, buybacks and shares retired, margins and their trend, revenue growth,
free cash flow, and trailing and forward P/E.

Round 43 asked comparisons of this kind ("which has the better outlook for margins, Exxon or
Chevron, and how do their dividend yields compare?", "rank Exxon, Shell and BP by dividend yield
and payout ratio", "does Microsoft have a buyback and how many shares has it retired", "how much
has Circle paid out versus Coinbase", "what is the P/E and dividend yield of Bitcoin") and no
reader took them whole: `filing_figures` reads one latest quarter, not a trailing year, and has
no dividends, buybacks, share counts or P/E. Every source below was read live on 2026-10-06.

**Filings, trailing four quarters.** SEC EDGAR's XBRL company facts
(``data.sec.gov/api/xbrl/companyfacts``), through `filing_figures.company_document`. A quarter is
a three-month row, or the difference of two year-to-date rows with the same start
(`market/hyperscaler_capex.quarters`); the trailing year is the latest four quarters on the
revenue calendar, and the year before is the four quarters before those. Where a quarter is not
tagged on its own, the year is assembled as fiscal year + year to date - year to date a year
earlier. Coinbase tags no nine months for 2025, but does tag its year and its half: its
repurchases are 790 + 1,243 - 0 = $2.03bn. One tag per measure, the one with the newest period:
Ford's ``Revenues`` and ``RevenueFromContractWithCustomer...`` are different totals, so they are
never spliced.

**Hand checks against the raw rows.** Exxon: dividends paid 17,231 (FY2025) + 8,633 (H1 2026) -
8,623 (H1 2025) = $17,241m; net income 28,844 + 18,708 - 14,795 = $32,757m; payout 52.63%
(Yahoo's own ``payoutRatio``: 52.51%). Apple: repurchases 90,711 (FY2025) + 62,094 (nine months
to 27 Jun 2026) - 70,579 (nine months to 28 Jun 2025) = $82,226m, which the module reports as
$82.23bn.

**What EDGAR does not serve, found by reading it.**

- *ExxonMobil Holdings Corp* took the XOM ticker under a new CIK (2115436) whose facts begin with
  the 2025 comparatives. The earlier quarters are under Exxon Mobil Corporation (CIK 34088), so
  :data:`PREDECESSORS` merges the two.
- *Shell and BP* (CIKs 1306965 and 313807) tag the ``ifrs-full`` taxonomy only, in 20-F annual
  facts: no quarters, so their figures are the latest fiscal year against the one before. Shell
  tags no gross profit, operating profit or capital-spending line (its free cash flow is not
  computed); BP tags no gross profit, EPS or share count (its ``dei`` taxonomy is absent). BP's
  2025 profit attributable to shareholders is $55m against $5.06bn of dividends paid, a 9,198%
  payout that the answer prints and marks as not useful. Their dividends per share are per
  ordinary share (an ADR is 2 for Shell, 6 for BP), so the per-ADR dividend comes from Yahoo
  and is never mixed with the 20-F's.
- *SEC's feed trails the filing.* Ford filed its 10-Q for the quarter to 30 Jun 2026 on 29 Jul;
  the facts feed still ended at 31 Mar on 6 Oct. `filing_figures.latest_periodic_report` reads
  the submissions record and the answer says so rather than passing March off as June.
- Energy companies tag no operating income (Exxon, Chevron) and Exxon no gross profit. A gross
  margin built as revenue less a cost-of-sales line (Chevron, Alphabet, Ford) is labelled as
  derived.

**Share count.** The balance sheet's ``CommonStockSharesOutstanding`` at the quarter end and a
year earlier; else the three-month diluted weighted average; else the cover page. Microsoft: 7,427m
against 7,434m. The change is net of shares issued to staff, and the gross repurchased shares
(``StockRepurchased...Shares``) are added where tagged (Microsoft 36m).

**Yahoo Finance** (`fundamentals.yahoo_summary`, quoteSummary) for the price, market cap,
indicated dividend rate, forward P/E and enterprise value; checked 2026-10-06 for XOM (price
$164.00, dividend rate $4.12, forward P/E 14.45), SHEL ($96.52, $3.12, 9.30), BP ($44.62, $2.02,
8.46), MSFT ($525.18, $3.92, 22.21), COIN and CRCL (no dividend rate). A dividend yield is the
filing's latest declared (or paid) dividend times four over that price; a company paying less
than every quarter gets the four quarters added up instead. Where Yahoo's indicated rate differs
by more than 10%, the answer says so. Trailing P/E is price over four quarters' diluted EPS from
the filing (Exxon $7.78, Yahoo's own $7.77). A quarter whose EPS is more than twice the average
of the other three is flagged, because it distorts the trailing P/E and net margin: Exxon's
$3.48 against $1.43, Alphabet's $9.11 against $3.60 on 6 Oct 2026.

**Not a forecast.** "Which has the better outlook for margins" is answered with the trend (this
trailing year against the one before, in points) and says that a filing carries no margin
guidance. Forward P/E is Yahoo's, on analysts' consensus EPS, and is named as such.

**Not companies.** Bitcoin, Ether and other crypto assets, and gold, silver and crude oil, have
no earnings, dividend or shares: a P/E, EPS, dividend yield, buyback or margin asked of one is
answered with that fact, as a stated answer rather than a missing figure. Ether's staking
rewards are named as protocol rewards, not a dividend, and no number is given for them here.

Entry point: :func:`lines`. :data:`CUE` and :func:`asks` say when the reader takes a question;
margins, growth, free cash flow and EPS asked of one company with a filing word ("10-Q",
"latest quarter") stay with `filing_figures`.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Final

from argus.lui.research import filing_figures
from argus.lui.research.fundamentals import raw_number, yahoo_summary
from argus.market.hyperscaler_capex import quarters

PREDECESSORS: Final[dict[str, tuple[int, ...]]] = {"XOM": (34088,)}
SHORT_NAMES: Final[dict[str, str]] = {
    "XOM": "Exxon", "CVX": "Chevron", "SHEL": "Shell", "BP": "BP", "MSFT": "Microsoft",
    "GOOGL": "Alphabet", "AAPL": "Apple", "TSLA": "Tesla", "F": "Ford", "COIN": "Coinbase",
    "CRCL": "Circle", "KO": "Coca-Cola", "PEP": "PepsiCo"}
ADR_RATIOS: Final[dict[str, int]] = {"SHEL": 2, "BP": 6}
"""Ordinary shares behind one ADR (Shell 2, BP 6). A 20-F reports per ordinary share, Yahoo and
the NYSE price per ADR; the answer takes the per-ADR dividend from Yahoo and never mixes them."""
FINANCE_ARMS: Final[dict[str, str]] = {"F": "Ford Credit", "GM": "GM Financial"}
MAX_COMPANIES: Final = 4
_US_FORMS: Final = frozenset({"10-Q", "10-K"})
_IFRS_FORMS: Final = frozenset({"20-F", "40-F", "20-F/A"})

_DIV = re.compile(r"\bdividends?\b|\bpay\s?outs?\b|\bdistribut\w*\s+(?:to\s+)?shareholders\b", re.I)
_PAYOUT = re.compile(r"\bpay\s?out\s+ratios?\b|\bhow\s+much\s+of\s+(?:its\s+)?(?:profit|earnings|"
                     r"income)\b[^?]{0,30}\b(?:pay|paid|return)", re.I)
_BUYBACK = re.compile(
    r"\bbuy[\s-]?backs?\b|\brepurchas\w+|\bbought\s+back\b|\bretire[sd]?\b[^?]{0,30}\bshares?\b|"
    r"\bshares?\b[^?]{0,30}\bretire[sd]?\b|\bhow\s+many\s+shares\b|\bshare\s+count\b|"
    r"\bdiluti\w+", re.I)
_PAID_OUT = re.compile(r"\bpaid\s+out\b|\breturn(?:ed|s)?\s+(?:to\s+)?(?:shareholders|capital)\b|"
                       r"\bshareholder\s+(?:returns?|yield)\b|\bcapital\s+returns?\b", re.I)
_BOTH = re.compile(r"\bdividends?\s+(?:or|and|plus|\+)\s+(?:share\s+)?(?:buy[\s-]?backs?|"
                   r"repurchas\w+)|\b(?:buy[\s-]?backs?|repurchas\w+)\s+(?:or|and|plus|\+)\s+"
                   r"dividends?", re.I)
_MARGIN = re.compile(r"\b(?:gross|operating|net|profit|ebit)\s+margins?\b|\bmargins?\b", re.I)
_GROWTH = re.compile(r"\b(?:revenue|sales|top[\s-]line)\s+(?:growth|grew|growing)\b|"
                     r"\bgrowth\s+(?:in|of)\s+(?:revenue|sales)\b", re.I)
_FCF = re.compile(r"\bfree\s+cash\s*flow\b|\bfcf\b", re.I)
_PE = re.compile(r"\bp\s?/\s?e\b|\bpe\s+ratios?\b|\bprice[\s-]to[\s-]earnings\b|"
                 r"\bearnings\s+multiple\b|\bforward\s+earnings\b", re.I)
_FORWARD = re.compile(r"\bforward\b|\bnext[\s-]twelve\b|\bfwd\b", re.I)
_TRAILING = re.compile(r"\btrailing\b|\bttm\b|\bcurrent\s+p/?e\b", re.I)
_EVS = re.compile(r"\bev\s*/\s*(?:sales|revenue)\b|\benterprise\s+value\s+(?:to|/|over)\s+"
                  r"(?:sales|revenue)\b", re.I)
_EPS = re.compile(r"\beps\b|\bearnings\s+per\s+share\b", re.I)
_GENERIC = re.compile(r"\bfundamentals?\b", re.I)
_RANK = re.compile(r"\brank\w*\b|\border\s+(?:them|these)\b|\bhighest\b|\blowest\b|\bwhich\s+"
                   r"(?:one\s+)?(?:pays?|has|offers?)\s+(?:the\s+)?(?:most|more|bigger|larger|"
                   r"better|higher)\b", re.I)
_COMPARE = re.compile(r"\bcompar\w+\b|\bversus\b|\bvs\.?\b|\bagainst\b|\bor\b|\band\b|\bbetter\b|"
                      r"\bwhich\b|\brank\w*\b", re.I)
_FILING_WORDS = re.compile(
    r"\b10-?[qk]\b|\bquarterly\s+report|\blatest\s+(?:quarter|filing|results)\b|\blast\s+"
    r"(?:reported\s+)?quarter\b|\bmost\s+recent\s+quarter\b|\bthis\s+quarter\b|\bq[1-4]\b", re.I)
_WHY = re.compile(r"\b(?:why|explain\w*|reasons?|drove|caused?)\b", re.I)
CUE: Final = re.compile("|".join(f"(?:{p.pattern})" for p in (
    _DIV, _BUYBACK, _MARGIN, _GROWTH, _FCF, _PE, _EVS, _EPS, _GENERIC, _PAID_OUT)), re.I)
"""The metric words this reader answers. Dividends, payout, buybacks, shares retired, P/E and
EV/sales are its own; margins, revenue growth, free cash flow and EPS are shared with
`filing_figures`, so a question with only those, no comparison and a filing word ("10-Q",
"latest quarter") is left to that reader."""

_ALIASES: Final[tuple[tuple[str, str], ...]] = (
    (r"\bexxon(?:\s?mobil)?\b|(?-i:\bXOM\b)", "XOM"),
    (r"\bchevron\b|(?-i:\bCVX\b)", "CVX"),
    (r"(?-i:\bShell(?:\s+plc)?\b|\bSHEL\b)", "SHEL"),
    (r"(?-i:\bBP(?:\s+plc)?\b)", "BP"),
    (r"\bmicrosoft\b|(?-i:\bMSFT\b)", "MSFT"),
    (r"\balphabet\b|\bgoogle\b|(?-i:\bGOOGL?\b)", "GOOGL"),
    (r"\bapple\b|(?-i:\bAAPL\b)", "AAPL"),
    (r"\btesla\b|(?-i:\bTSLA\b)", "TSLA"),
    (r"(?-i:\bFord\b|\$F\b)", "F"),
    (r"\bcoinbase\b|(?-i:\bCOIN\b)", "COIN"),
    (r"(?-i:\bCircle\b|\bCRCL\b)", "CRCL"),
    (r"\bcoca[\s-]?cola\b|(?-i:\bKO\b)", "KO"),
    (r"\bpepsi(?:co)?\b|(?-i:\bPEP\b)", "PEP"),
)
"""The names the brief lists, matched where two readings are possible only in the capitalisation
that means the company ("Shell", "BP", "Circle", "Ford"; "bp" is a basis point)."""

_ASSETS: Final[tuple[tuple[str, str, str], ...]] = (
    (r"\bbitcoin\b|(?-i:\bBTC\b)", "Bitcoin", "crypto"),
    (r"\bethereum\b|\bether\b|(?-i:\bETH\b)", "Ether", "crypto"),
    (r"\bsolana\b|(?-i:\bSOL\b)", "Solana's SOL", "crypto"),
    (r"\bripple\b|(?-i:\bXRP\b)", "XRP", "crypto"),
    (r"\bdogecoin\b|(?-i:\bDOGE\b)", "Dogecoin", "crypto"),
    (r"\bcardano\b|(?-i:\bADA\b)", "Cardano's ADA", "crypto"),
    (r"\blitecoin\b|(?-i:\bLTC\b)", "Litecoin", "crypto"),
    (r"(?-i:\bBNB\b)", "BNB", "crypto"),
    (r"\bgold\b|(?-i:\bXAU\b)", "Gold", "metal"),
    (r"\bsilver\b|(?-i:\bXAG\b)", "Silver", "metal"),
    (r"\bplatinum\b|\bpalladium\b|\bcopper\b", "That metal", "metal"),
    (r"\bcrude(?:\s+oil)?\b|\bbrent\b|\bwti\b", "Crude oil", "metal"),
)
_NOT_COMPANIES: Final = frozenset({
    "EPS", "FCF", "ROE", "ROA", "TTM", "EBIT", "YOY", "ETF", "IPO", "CEO", "USD", "SEC", "GAAP",
    "DPS", "US", "UK", "EU", "GDP", "CPI", "FED", "PE", "EV", "AI", "ALL", "FOR", "ARE", "THE",
    "NEW", "OUT", "ONE", "TWO", "BUY", "SELL", "HOLD", "TOP", "LOW", "HIGH", "NOW", "CAN", "HAS",
    "HOW", "WHO", "WHY", "ANY", "OFF", "RANK", "ITS", "NYSE", "BEST", "DIV"})

_ASSET_TOKENS: Final = frozenset({"BTC", "ETH", "SOL", "XRP", "DOGE", "ADA", "LTC", "BNB", "XAU",
                                  "XAG", "WTI"})
_FACTS_CACHE: dict[str, tuple[float, dict[str, Any] | None]] = {}


@dataclass
class Company:
    """Everything read for one company. ``None`` means "not available", never zero."""

    ticker: str
    name: str
    kind: str = "us"
    error: str | None = None
    end: date | None = None
    period: str = ""
    filed: str = ""
    notes: list[str] = field(default_factory=list)
    # Yahoo
    yahoo: bool = False
    price: float | None = None
    mcap: float | None = None
    y_rate: float | None = None
    y_forward_pe: float | None = None
    y_trailing_pe: float | None = None
    y_eps: float | None = None
    y_ev: float | None = None
    # filings, trailing four quarters (US) or the fiscal year (IFRS), and the period before
    rev: float | None = None
    rev_prior: float | None = None
    gross_derived: bool = False
    gross: float | None = None
    gross_prior: float | None = None
    operating: float | None = None
    operating_prior: float | None = None
    net: float | None = None
    net_prior: float | None = None
    cfo: float | None = None
    cfo_prior: float | None = None
    capex: float | None = None
    capex_prior: float | None = None
    div_paid: float | None = None
    buyback: float | None = None
    buyback_shares: float | None = None
    eps: float | None = None
    eps_prior: float | None = None
    eps_quarters: list[tuple[date, float]] = field(default_factory=list)
    dps: float | None = None
    dps_annual: float | None = None
    dps_kind: str = ""
    dps_end: date | None = None
    dps_note: str = ""
    shares_now: float | None = None
    shares_then: float | None = None
    shares_basis: str = ""

    @property
    def label(self) -> str:
        return f"{self.name} ({self.ticker})" if self.name != self.ticker else self.ticker

    @property
    def annual_dividend(self) -> float | None:
        """Per share, per year: the filings' latest declared figure annualised, else Yahoo's."""
        if self.dps_annual is not None:
            return self.dps_annual
        return self.y_rate if self.y_rate else None

    @property
    def yield_(self) -> float | None:
        rate = self.annual_dividend
        return rate / self.price if rate is not None and self.price else None

    @property
    def payout(self) -> float | None:
        if self.div_paid is None or self.net is None or self.net <= 0:
            return None
        return self.div_paid / self.net

    @property
    def fcf(self) -> float | None:
        return None if self.cfo is None or self.capex is None else self.cfo - self.capex

    @property
    def fcf_prior(self) -> float | None:
        if self.cfo_prior is None or self.capex_prior is None:
            return None
        return self.cfo_prior - self.capex_prior

    @property
    def pays(self) -> bool:
        return bool((self.div_paid or 0) > 0 or (self.dps_annual or 0) > 0
                    or (self.y_rate or 0) > 0)

    @property
    def share_change(self) -> float | None:
        if not self.shares_now or not self.shares_then:
            return None
        return self.shares_now / self.shares_then - 1

    @property
    def trailing_pe(self) -> float | None:
        if self.kind == "us" and self.eps is not None and self.price:
            return self.price / self.eps if self.eps > 0 else None
        if self.kind == "ifrs":
            return self.y_trailing_pe if self.y_trailing_pe and self.y_trailing_pe > 0 else None
        return None

    @property
    def returned(self) -> float | None:
        """Dividends paid plus buybacks, both over the period; a company that pays no dividend
        counts $0 for that half, and None when neither half is known."""
        dividends = self.div_paid if self.div_paid is not None else (None if self.pays else 0.0)
        if dividends is None and self.buyback is None:
            return None
        return (dividends or 0.0) + (self.buyback or 0.0)

    @property
    def buyback_share_of_cap(self) -> float | None:
        return self.buyback / self.mcap if self.buyback is not None and self.mcap else None

    def margin(self, which: str, prior: bool = False) -> float | None:
        top = self.rev_prior if prior else self.rev
        value = {"gross": self.gross_prior if prior else self.gross,
                 "operating": self.operating_prior if prior else self.operating,
                 "net": self.net_prior if prior else self.net}[which]
        return value / top if value is not None and top and top > 0 else None

    @property
    def growth(self) -> float | None:
        if self.rev is None or not self.rev_prior or self.rev_prior <= 0:
            return None
        return self.rev / self.rev_prior - 1

    @property
    def fcf_margin(self) -> float | None:
        return self.fcf / self.rev if self.fcf is not None and self.rev and self.rev > 0 else None

    @property
    def eps_growth(self) -> float | None:
        if self.eps is None or self.eps_prior is None or self.eps_prior <= 0 or self.eps <= 0:
            return None
        return self.eps / self.eps_prior - 1

    @property
    def ev_sales(self) -> float | None:
        return self.y_ev / self.rev if self.y_ev and self.rev and self.rev > 0 else None


# --- what was asked ----------------------------------------------------------------------------


def asked_metrics(text: str) -> list[str]:
    """The metric keys the question names, in a fixed order. ``div``, ``payout``, ``buyback``,
    ``shares``, ``gross``/``operating``/``net`` (margins), ``growth``, ``fcf``, ``pe``, ``fpe``,
    ``evs``, ``eps``. Plain "margins" asks for all three; "P/E" for trailing and forward, "forward
    P/E" for forward only."""
    found: list[str] = []
    if _DIV.search(text) or _PAID_OUT.search(text):
        found.append("div")
        if _PAYOUT.search(text) or re.search(r"\bpay\s?out\b", text, re.I):
            found.append("payout")
    if _BUYBACK.search(text) or _PAID_OUT.search(text):
        found.append("buyback")
        if re.search(r"\bretire|\bhow\s+many\s+shares\b|\bshare\s+count\b|\bdiluti", text, re.I):
            found.append("shares")
    if _MARGIN.search(text):
        for key in ("gross", "operating", "net"):
            if re.search(rf"\b{'(?:operating|ebit)' if key == 'operating' else key}\s+margins?\b",
                         text, re.I):
                found.append(key)
        if not any(k in found for k in ("gross", "operating", "net")):
            found += ["gross", "operating", "net"]
    if _GROWTH.search(text):
        found.append("growth")
    if _FCF.search(text):
        found.append("fcf")
    if _PE.search(text):
        forward, trailing = bool(_FORWARD.search(text)), bool(_TRAILING.search(text))
        if forward and not trailing:
            found.append("fpe")
        else:
            found += ["pe", "fpe"]
    if _EVS.search(text):
        found.append("evs")
    if _EPS.search(text):
        found.append("eps")
    if "div" in found and "buyback" in found and (_PAID_OUT.search(text) or _BOTH.search(text)):
        found.append("returned")
    if not found and _GENERIC.search(text):
        found += ["gross", "operating", "net", "growth", "pe", "fpe"]
    return list(dict.fromkeys(found))


def _distinctive(metrics: Sequence[str]) -> bool:
    """Whether the question asks for something only this reader answers."""
    return any(m in metrics for m in ("div", "payout", "buyback", "shares", "pe", "fpe", "evs"))


def _positions(text: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for pattern, ticker in _ALIASES:
        m = re.search(pattern, text, re.I)
        if m:
            out[ticker] = m.start()
    return out


def _listed(text: str) -> list[str]:
    """US tickers of Bitget-listed equities the text names (``NVDAUSDT`` as NVDA)."""
    from argus.lui.research.parse import is_us_equity, research_symbols

    try:
        symbols = research_symbols(text)[0]
    except Exception:
        return []
    return [s.removesuffix("STOCKUSDT").removesuffix("USDT") for s in symbols if is_us_equity(s)]


def _named(text: str) -> list[tuple[str, str]]:
    """Companies named in full or by first word ("Nvidia"), as (name as typed, ticker); SEC's
    register, through `market.company_names`."""
    from argus.market.company_names import us_ticker

    found: list[tuple[str, str]] = []
    rest = text
    for _ in range(MAX_COMPANIES):
        hit = us_ticker(rest)
        if hit is None:
            break
        found.append(hit)
        rest = rest.replace(hit[0], " ", 1)
    return found


def resolve(text: str) -> list[str]:
    """The tickers the question names, in the order named, at most :data:`MAX_COMPANIES`."""
    where = _positions(text)
    for said, ticker in _named(text):
        if ticker not in where:
            m = re.search(re.escape(said), text)
            where[ticker] = m.start() if m else len(text)
    for ticker in _listed(text):
        if ticker in where or ticker in _NOT_COMPANIES:
            continue
        m = re.search(rf"(?-i:\b{re.escape(ticker)}\b)", text)
        if m:
            where[ticker] = m.start()
    from argus.market.company_names import sec_registered

    for m in re.finditer(r"(?-i:\b[A-Z]{2,5}\b)", text):
        token = m.group(0)
        if (token not in where and token not in _NOT_COMPANIES and token not in _ASSET_TOKENS
                and sec_registered(token)):
            where[token] = m.start()
    return [t for t, _ in sorted(where.items(), key=lambda kv: kv[1])]


def _assets(text: str) -> list[tuple[str, str]]:
    found = [(m.start(), name, kind) for pattern, name, kind in _ASSETS
             if (m := re.search(pattern, text, re.I))]
    return [(name, kind) for _, name, kind in sorted(found)]


# --- reading the filings -----------------------------------------------------------------------


def _document(ticker: str) -> dict[str, Any] | None:
    """SEC's company facts, kept ten minutes."""
    hit = _FACTS_CACHE.get(ticker)
    if hit is not None and time.monotonic() - hit[0] < 600:
        return hit[1]
    doc = filing_figures.company_document(ticker, PREDECESSORS.get(ticker, ()))
    _FACTS_CACHE[ticker] = (time.monotonic(), doc)
    return doc


def _tag_spans(doc: dict[str, Any], taxonomy: str, tag: str, unit: str,
               forms: frozenset[str]) -> dict[tuple[date, date], float]:
    rows = doc.get(taxonomy, {}).get(tag, {}).get("units", {}).get(unit, [])
    out: dict[tuple[date, date], float] = {}
    for row in sorted((r for r in rows if r.get("form") in forms and r.get("start")),
                      key=lambda r: str(r.get("filed", ""))):
        out[(date.fromisoformat(row["start"]), date.fromisoformat(row["end"]))] = float(row["val"])
    return out


def _best(doc: dict[str, Any], taxonomy: str, tags: Sequence[str], unit: str = "USD",
          forms: frozenset[str] = _US_FORMS) -> tuple[str, dict[tuple[date, date], float]]:
    """The tag with the most recent period (the first listed on a tie) and its duration facts,
    later filings overriding earlier ones for the same period. One tag only: Ford's ``Revenues``
    and ``RevenueFromContractWithCustomer...`` are different totals and must not be spliced."""
    name = ""
    best: dict[tuple[date, date], float] = {}
    for tag in tags:
        spans = _tag_spans(doc, taxonomy, tag, unit, forms)
        if spans and (not best or max(e for _, e in spans) > max(e for _, e in best)):
            name, best = tag, spans
    return name, best


def _spans(doc: dict[str, Any], taxonomy: str, tags: Sequence[str], unit: str = "USD",
           forms: frozenset[str] = _US_FORMS) -> dict[tuple[date, date], float]:
    return _best(doc, taxonomy, tags, unit, forms)[1]


def _quarter_values(spans: dict[tuple[date, date], float]) -> dict[date, float]:
    return quarters([{"form": "10-Q", "start": s.isoformat(), "end": e.isoformat(), "val": v}
                     for (s, e), v in spans.items()])


def _ttm(spans: dict[tuple[date, date], float], by_quarter: dict[date, float],
         ends: Sequence[date]) -> float | None:
    """The four quarters ending ``ends`` summed. Where a quarter is not tagged on its own the
    year is assembled from what is: a fiscal year, plus the year to date, less the year to date
    a year earlier (Coinbase tags no nine months for 2025, but does tag the year and the half)."""
    if len(ends) != 4:
        return None
    if all(e in by_quarter for e in ends):
        return sum(by_quarter[e] for e in ends)
    last = ends[-1]
    for (start, end), value in spans.items():
        if end == last and 350 <= (end - start).days <= 380:
            return value
    for (start, end), value in spans.items():
        if end in ends[:3] and 350 <= (end - start).days <= 380:
            now = next((v for (s2, e2), v in spans.items()
                        if e2 == last and abs((s2 - end).days - 1) <= 4), None)
            before = next((v for (s3, e3), v in spans.items()
                           if s3 == start and abs((last - e3).days - 365) <= 10), None)
            if now is not None and before is not None:
                return value + now - before
    return None


def _flow(doc: dict[str, Any], tags: Sequence[str], ends: Sequence[date], unit: str = "USD"
          ) -> tuple[float | None, float | None, dict[date, float]]:
    spans = _spans(doc, "us-gaap", tags, unit)
    if not spans:
        return None, None, {}
    by_q = _quarter_values(spans)
    now = _ttm(spans, by_q, ends[-4:])
    prior = _ttm(spans, by_q, ends[-8:-4]) if len(ends) >= 8 else None
    return now, prior, by_q


def _calendar(by_quarter: dict[date, float]) -> list[date]:
    """The latest run of consecutive quarter ends (at most eight), oldest first."""
    ends = sorted(by_quarter)
    if not ends:
        return []
    chain = [ends[-1]]
    for e in reversed(ends[:-1]):
        gap = (chain[0] - e).days
        if gap < 80:
            continue
        if gap > 100:
            break
        chain.insert(0, e)
    return chain[-8:]


def _instants(doc: dict[str, Any], taxonomy: str, tag: str, forms: frozenset[str]
              ) -> dict[date, float]:
    rows = doc.get(taxonomy, {}).get(tag, {}).get("units", {}).get("shares", [])
    out: dict[date, float] = {}
    for row in sorted((r for r in rows if r.get("form") in forms and not r.get("start")),
                      key=lambda r: str(r.get("filed", ""))):
        out[date.fromisoformat(row["end"])] = float(row["val"])
    return out


def _near(values: dict[date, float], target: date, days: int) -> tuple[date, float] | None:
    close = [(abs((d - target).days), d, v) for d, v in values.items()
             if abs((d - target).days) <= days]
    if not close:
        return None
    _, day, value = min(close)
    return day, value


def _share_count(doc: dict[str, Any], anchor: date) -> tuple[float, float, str] | None:
    """Shares now and a year earlier, from the best source the filings carry: the balance
    sheet's shares outstanding at the quarter ends, else the three-month diluted weighted average,
    else the cover page's count."""
    year_ago = date(anchor.year - 1, anchor.month, min(anchor.day, 28))
    held = _instants(doc, "us-gaap", "CommonStockSharesOutstanding", _US_FORMS)
    now, then = _near(held, anchor, 5), _near(held, year_ago, 10)
    if now and then:
        return now[1], then[1], "shares outstanding on the balance sheet at the quarter ends"
    weighted = _tag_spans(doc, "us-gaap", "WeightedAverageNumberOfDilutedSharesOutstanding",
                          "shares", _US_FORMS)
    three = {e: v for (s, e), v in weighted.items() if 80 <= (e - s).days <= 100}
    now, then = _near(three, anchor, 5), _near(three, year_ago, 10)
    if now and then:
        return now[1], then[1], "diluted weighted-average shares for the quarter"
    cover = _instants(doc, "dei", "EntityCommonStockSharesOutstanding", _US_FORMS)
    if cover:
        last = max(cover)
        then = _near(cover, date(last.year - 1, last.month, min(last.day, 28)), 40)
        if then and then[0] != last:
            return cover[last], then[1], "shares outstanding on the filings' cover pages"
    return None


def _filed_of(doc: dict[str, Any], taxonomy: str, tags: Sequence[str], end: date,
              forms: frozenset[str]) -> str:
    best: tuple[str, str] | None = None
    for tag in tags:
        for rows in doc.get(taxonomy, {}).get(tag, {}).get("units", {}).values():
            for row in rows:
                newer = best is None or str(row.get("filed", "")) > best[1]
                if row.get("end") == end.isoformat() and row.get("form") in forms and newer:
                    best = (str(row["form"]), str(row.get("filed", "")))
    return f"{best[0]} filed {date.fromisoformat(best[1]):%d %b %Y}" if best else ""


_REVENUE: Final = ("RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues",
                   "SalesRevenueNet")
_COST: Final = ("CostOfGoodsAndServicesSold", "CostOfRevenue")
_NET: Final = ("NetIncomeLoss", "ProfitLoss")
_CFO: Final = ("NetCashProvidedByUsedInOperatingActivities",
               "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations")
_CAPEX: Final = ("PaymentsToAcquirePropertyPlantAndEquipment",
                 "PaymentsToAcquireProductiveAssets")
_DIV_PAID: Final = ("PaymentsOfDividendsCommonStock", "PaymentsOfDividends")
_BUY: Final = ("PaymentsForRepurchaseOfCommonStock", "PaymentsForRepurchaseOfEquity")
_BUY_SHARES: Final = ("StockRepurchasedAndRetiredDuringPeriodShares",
                      "StockRepurchasedDuringPeriodShares", "TreasuryStockSharesAcquired")
_DPS: Final = ("CommonStockDividendsPerShareDeclared", "CommonStockDividendsPerShareCashPaid")


def _read_us(c: Company, doc: dict[str, Any]) -> None:
    rev_spans = _spans(doc, "us-gaap", _REVENUE)
    by_rev = _quarter_values(rev_spans)
    ends = _calendar(by_rev)
    if len(ends) < 4:
        c.error = "its filings tag too few consecutive quarters of revenue to total four"
        return
    c.end = ends[-1]
    c.period = f"four quarters to {c.end:%d %b %Y}"
    c.filed = _filed_of(doc, "us-gaap", _REVENUE, c.end, _US_FORMS)
    c.rev = _ttm(rev_spans, by_rev, ends[-4:])
    c.rev_prior = _ttm(rev_spans, by_rev, ends[-8:-4]) if len(ends) >= 8 else None
    c.gross, c.gross_prior, _ = _flow(doc, ("GrossProfit",), ends)
    if c.gross is None:
        cost, cost_prior, _ = _flow(doc, _COST, ends)
        if cost is not None and c.rev is not None:
            c.gross_derived = True
            c.gross = c.rev - cost
            c.gross_prior = (c.rev_prior - cost_prior if cost_prior is not None
                             and c.rev_prior is not None else None)
    c.operating, c.operating_prior, _ = _flow(doc, ("OperatingIncomeLoss",), ends)
    c.net, c.net_prior, _ = _flow(doc, _NET, ends)
    c.cfo, c.cfo_prior, _ = _flow(doc, _CFO, ends)
    c.capex, c.capex_prior, _ = _flow(doc, _CAPEX, ends)
    c.div_paid, _, _ = _flow(doc, _DIV_PAID, ends)
    c.buyback, _, _ = _flow(doc, _BUY, ends)
    if c.buyback is None:
        _, held = _best(doc, "us-gaap", _BUY)
        last = max((e, v) for (_, e), v in held.items()) if held else None
        if last and last[1] == 0 and 0 < (c.end - last[0]).days <= 420:
            c.buyback = 0.0
            c.notes.append(f"{c.ticker}: its filings tag $0 of repurchases for the period to "
                           f"{last[0]:%d %b %Y} and nothing after, which is how a company with "
                           "no buyback files; treated as no buyback.")
    c.buyback_shares, _, _ = _flow(doc, _BUY_SHARES, ends, unit="shares")
    c.eps, c.eps_prior, eps_q = _flow(doc, ("EarningsPerShareDiluted",), ends, unit="USD/shares")
    c.eps_quarters = [(e, eps_q[e]) for e in ends[-4:] if e in eps_q]
    if len(c.eps_quarters) < 4:
        c.eps_quarters = []
    dps_tag, dps_spans = _best(doc, "us-gaap", _DPS, "USD/shares")
    dps_q = _quarter_values(dps_spans)
    recent = [dps_q.get(e) for e in ends[-4:]]
    if recent[-1] is not None:
        c.dps, c.dps_end = recent[-1], ends[-1]
        c.dps_kind = "paid" if dps_tag.endswith("CashPaid") else "declared"
        if recent[-1] > 0 and (all(v is None for v in recent[:-1])
                               or all(v is not None and v > 0 for v in recent)
                               or any(v is None for v in recent)):
            c.dps_annual = recent[-1] * 4
            c.dps_note = "the latest quarter's dividend times four"
        else:
            c.dps_annual = sum(v for v in recent if v is not None)
            c.dps_note = "the four quarters' dividends added up (not paid every quarter)"
    counted = _share_count(doc, c.end)
    if counted:
        c.shares_now, c.shares_then, c.shares_basis = counted


_IFRS_REV: Final = ("Revenue", "RevenueFromContractsWithCustomers")
_IFRS_NET: Final = ("ProfitLossAttributableToOwnersOfParent",
                    "ProfitLossAttributableToOrdinaryEquityHoldersOfParentEntity", "ProfitLoss")
_IFRS_CAPEX: Final = (
    "PurchaseOfPropertyPlantAndEquipmentIntangibleAssetsOtherThanGoodwillInvestmentPropertyAnd"
    "OtherNoncurrentAssets", "PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities",
    "PaymentsToAcquirePropertyPlantAndEquipment")
_IFRS_DIV: Final = ("DividendsPaidToEquityHoldersOfParentClassifiedAsFinancingActivities",
                    "DividendsPaidOrdinaryShares", "DividendsPaid")
_IFRS_BUY: Final = ("PaymentsToAcquireOrRedeemEntitysShares", "PurchaseOfTreasuryShares")


def _year(spans: dict[tuple[date, date], float], end: date) -> float | None:
    for (s, e), v in spans.items():
        if 350 <= (e - s).days <= 380 and abs((e - end).days) <= 5:
            return v
    return None


def _read_ifrs(c: Company, doc: dict[str, Any]) -> None:
    def pair(tags: Sequence[str], unit: str = "USD") -> tuple[float | None, float | None]:
        spans = _spans(doc, "ifrs-full", tags, unit, _IFRS_FORMS)
        if c.end is None or not spans:
            return None, None
        return (_year(spans, c.end),
                _year(spans, date(c.end.year - 1, c.end.month, min(c.end.day, 28))))

    rev_spans = _spans(doc, "ifrs-full", _IFRS_REV, "USD", _IFRS_FORMS)
    years = [e for (s, e) in rev_spans if 350 <= (e - s).days <= 380]
    if not years:
        c.error = "its 20-F tags no annual revenue EDGAR could read"
        return
    c.end = max(years)
    c.period = f"fiscal {c.end.year} (year to {c.end:%d %b %Y}), the latest annual report"
    c.filed = _filed_of(doc, "ifrs-full", _IFRS_REV, c.end, _IFRS_FORMS)
    c.rev, c.rev_prior = pair(_IFRS_REV)
    c.gross, c.gross_prior = pair(("GrossProfit",))
    c.operating, c.operating_prior = pair(("ProfitLossFromOperatingActivities",))
    c.net, c.net_prior = pair(_IFRS_NET)
    c.cfo, c.cfo_prior = pair(("CashFlowsFromUsedInOperatingActivities",))
    c.capex, c.capex_prior = pair(_IFRS_CAPEX)
    c.div_paid, _ = pair(_IFRS_DIV)
    c.buyback, _ = pair(_IFRS_BUY)
    c.eps, c.eps_prior = pair(("DilutedEarningsLossPerShare",
                               "DilutedEarningsLossPerShareFromContinuingOperations"),
                              "USD/shares")
    now, then = pair(("WeightedAverageShares", "AdjustedWeightedAverageShares"), "shares")
    if now and then:
        c.shares_now, c.shares_then = now, then
        c.shares_basis = "weighted-average ordinary shares for the year"


def _read_yahoo(c: Company, summary: dict[str, Any]) -> None:
    detail = summary.get("summaryDetail", {}) or {}
    stats = summary.get("defaultKeyStatistics", {}) or {}
    fin = summary.get("financialData", {}) or {}
    c.price = raw_number(fin.get("currentPrice")) or raw_number(detail.get("previousClose"))
    c.mcap = raw_number(detail.get("marketCap"))
    c.y_rate = raw_number(detail.get("dividendRate"))
    c.y_forward_pe = raw_number(detail.get("forwardPE")) or raw_number(stats.get("forwardPE"))
    c.y_trailing_pe = raw_number(detail.get("trailingPE"))
    c.y_eps = raw_number(stats.get("trailingEps"))
    c.y_ev = raw_number(stats.get("enterpriseValue"))
    c.yahoo = c.price is not None


def read_company(ticker: str, *, facts_of: Callable[[str], dict[str, Any] | None] = _document,
                 summary_of: Callable[[str], dict[str, Any]] = yahoo_summary,
                 report_of: Callable[[str], tuple[str, date, date] | None]
                 = filing_figures.latest_periodic_report) -> Company:
    """All the figures for one ticker; ``error`` set when its filings cannot be read."""
    c = Company(ticker=ticker, name=SHORT_NAMES.get(ticker, ticker))
    try:
        doc = facts_of(ticker)
    except Exception:
        doc = None
    if not doc:
        c.error = "its filings could not be read from SEC EDGAR just now"
        return c
    if c.name == ticker:
        entity = str(doc.get("entityName") or "")
        c.name = entity.title() if entity else ticker
    has_us = bool(_spans(doc, "us-gaap", _REVENUE))
    if has_us:
        _read_us(c, doc)
    elif _spans(doc, "ifrs-full", _IFRS_REV, "USD", _IFRS_FORMS):
        c.kind = "ifrs"
        _read_ifrs(c, doc)
    else:
        c.error = "its filings carry no revenue figure EDGAR could read in a form this reader uses"
        return c
    if c.error:
        return c
    try:
        _read_yahoo(c, summary_of(ticker))
    except Exception:
        c.yahoo = False
    try:
        report = report_of(ticker)
    except Exception:
        report = None
    if c.dps_annual and c.y_rate and abs(c.y_rate / c.dps_annual - 1) > 0.1:
        c.notes.append(
            f"{c.ticker}: Yahoo Finance's indicated dividend is ${c.y_rate:,.2f} a year against "
            f"${c.dps_annual:,.2f} annualised from the filing; the yield above uses the filing.")
    if report and c.end and report[1] > c.end and (report[1] - c.end).days > 20:
        c.notes.append(
            f"{c.label}: its {report[0]} for the period to {report[1]:%d %b %Y} (filed "
            f"{report[2]:%d %b %Y}) is not yet in SEC's XBRL feed, so the figures here run to "
            f"{c.end:%d %b %Y}.")
    return c


# --- saying it ---------------------------------------------------------------------------------


def _usd(x: float) -> str:
    if x == 0:
        return "$0"
    if abs(x) >= 1e12:
        return f"${x / 1e12:,.2f}tn"
    return filing_figures.money(x)


def _pct(x: float, places: int = 1) -> str:
    return f"{x:.{places}%}"


def _pts(now: float, then: float) -> str:
    return f"{(now - then) * 100:+.1f} pts"


def _shares(x: float) -> str:
    return f"{x / 1e6:,.0f}m"


def _eps_text(x: float) -> str:
    return f"-${-x:.2f}" if x < 0 else f"${x:.2f}"


def _price(c: Company) -> str:
    return f"${c.price:,.2f}" if c.price is not None else "an unavailable price"


def _p_div(c: Company) -> str:
    if not c.pays:
        why = ("none in its filings and none indicated by Yahoo Finance" if c.yahoo
               else "none in its filings (Yahoo Finance did not answer to confirm)")
        return f"pays no dividend ({why})"
    parts: list[str] = []
    if c.dps is not None and c.dps_annual is not None and c.dps_end is not None:
        parts.append(f"${c.dps:,.2f} a share {c.dps_kind} for the quarter to {c.dps_end:%d %b %Y}"
                     f", ${c.dps_annual:,.2f} a year ({c.dps_note})")
    elif c.y_rate:
        per = " per ADR" if c.ticker in ADR_RATIOS else " per share"
        parts.append(f"Yahoo Finance's indicated dividend is ${c.y_rate:,.2f} a year{per}")
    else:
        parts.append("it pays a dividend, but no per-share figure could be read")
    if c.yield_ is not None:
        parts.append(f"yield {_pct(c.yield_, 2)} at {_price(c)}")
    if c.div_paid is not None:
        parts.append(f"{_usd(c.div_paid)} of dividends paid in the {_span_word(c)}")
    return "; ".join(parts)


def _span_word(c: Company) -> str:
    return "year" if c.kind == "ifrs" else "four quarters"


def _p_payout(c: Company) -> str | None:
    if c.div_paid is None:
        return ("payout ratio not computed: no dividends-paid line tagged" if not c.pays
                else "payout ratio not computed: no dividends-paid line tagged in its filings")
    if c.net is None:
        return "payout ratio not computed: net income not tagged"
    if c.net <= 0:
        return (f"payout ratio not meaningful: net income was a {_usd(-c.net)} loss against "
                f"{_usd(c.div_paid)} of dividends paid")
    ratio = c.div_paid / c.net
    tail = (f"{_usd(c.div_paid)} of dividends paid against {_usd(c.net)} of net income in the "
            f"{_span_word(c)}")
    if ratio > 3:
        return (f"payout ratio {ratio * 100:,.0f}%, which is not a useful measure here: {tail}, "
                "because profit attributable to shareholders was small")
    return f"payout ratio {_pct(ratio)} ({tail})"


def _p_buyback(c: Company) -> str:
    if c.buyback is None:
        return "no repurchase line tagged in its filings for the period"
    if c.buyback <= 0:
        return f"no buyback: $0 repurchased in the {_span_word(c)}"
    text = f"bought back {_usd(c.buyback)} of its shares in the {_span_word(c)}"
    if c.buyback_share_of_cap is not None and c.mcap:
        text += f", {_pct(c.buyback_share_of_cap, 2)} of its {_usd(c.mcap)} market cap"
    return text


def _p_returned(c: Company) -> str:
    if c.returned is None:
        return "cash returned to shareholders could not be totalled from its filings"
    parts = [f"{_usd(c.div_paid)} of dividends" if c.div_paid else "no dividends",
             f"{_usd(c.buyback)} of buybacks" if c.buyback else "no buybacks"]
    text = f"returned {_usd(c.returned)} to shareholders in the {_span_word(c)}: " + " and ".join(
        parts)
    if c.mcap and c.returned:
        text += f" ({_pct(c.returned / c.mcap, 2)} of its {_usd(c.mcap)} market cap)"
    return text


def _p_shares(c: Company) -> str:
    if c.shares_now is None or c.shares_then is None:
        return "share count over the year is not tagged in its filings"
    change = c.shares_now - c.shares_then
    word = "fell" if change < 0 else "rose"
    text = (f"{c.shares_basis}: {_shares(c.shares_now)} against {_shares(c.shares_then)} a year "
            f"earlier, so the count {word} by {_shares(abs(change))} ({c.share_change:+.2%}), "
            "net of any shares issued to staff" if c.share_change is not None else "")
    if c.buyback_shares:
        text += (f"; it repurchased {_shares(c.buyback_shares)} shares gross in the "
                 f"{_span_word(c)}")
    return text


def _p_margin(c: Company, which: str) -> str:
    label = {"gross": "gross margin", "operating": "operating margin", "net": "net margin"}[which]
    now, then = c.margin(which), c.margin(which, prior=True)
    if now is None:
        reasons = {"gross": "gross profit is not tagged and no cost of sales to derive it",
                   "operating": "operating income is not tagged",
                   "net": "net income is not tagged"}
        return f"{label} not available ({reasons[which]} in its filings)"
    base = f"{label} {_pct(now)}"
    if which == "gross" and c.gross_derived:
        base += " (revenue less its cost-of-sales line; no gross profit is tagged)"
    return base + (f" against {_pct(then)} a year earlier ({_pts(now, then)})"
                   if then is not None else "")


def _p_growth(c: Company) -> str:
    if c.rev is None or c.growth is None:
        return (f"revenue {_usd(c.rev)}, with no prior period tagged to compare"
                if c.rev is not None else "revenue growth not available")
    before = "prior year" if c.kind == "ifrs" else "prior four quarters"
    return f"revenue {_usd(c.rev)}, {c.growth:+.1%} on the {before} ({_usd(c.rev_prior or 0)})"


def _p_fcf(c: Company) -> str:
    if c.fcf is None:
        missing = "capital spending" if c.cfo is not None else "cash from operations"
        return f"free cash flow not computed: {missing} is not tagged for the period"
    text = (f"free cash flow {_usd(c.fcf)} ({_usd(c.cfo or 0)} from operations less "
            f"{_usd(c.capex or 0)} of capital spending)")
    if c.fcf_margin is not None:
        text += f", {_pct(c.fcf_margin)} of revenue"
    if c.fcf_prior is not None:
        text += f"; the period before {_usd(c.fcf_prior)}"
    if c.ticker in FINANCE_ARMS:
        text += (f" (the cash flow statement includes {FINANCE_ARMS[c.ticker]}, so this is not "
                 "the company's own adjusted free cash flow)")
    return text


def _one_off(c: Company, metrics: Sequence[str] = ("pe",)) -> str | None:
    """A quarter whose EPS is more than twice the average of the other three, which distorts the
    trailing P/E (and the net margin) built on the four."""
    hit = [name for key, name in (("pe", "the trailing P/E"), ("eps", "EPS growth"),
                                  ("net", "the net margin")) if key in metrics]
    hurt = " and ".join(hit) if len(hit) < 3 else ", ".join(hit[:-1]) + " and " + hit[-1]
    q = [v for _, v in c.eps_quarters]
    for i, (end, value) in enumerate(c.eps_quarters):
        others = q[:i] + q[i + 1:]
        if len(others) == 3 and value > 0 and sum(others) / 3 > 0 and value > 2 * sum(others) / 3:
            verb = "is" if len(hit) == 1 else "are"
            tail = "; the forward P/E is the cleaner comparison" if "pe" in metrics else ""
            return (f"{c.ticker}'s EPS of ${value:.2f} for the quarter to {end:%d %b %Y} is more "
                    f"than twice the average of the other three (${sum(others) / 3:.2f}). A "
                    f"one-off may be inflating the year's earnings, so {hurt} {verb} "
                    f"distorted{tail}")
    return None


def _p_pe(c: Company) -> str:
    if c.kind == "us" and c.eps is not None and c.price:
        if c.eps <= 0:
            return (f"trailing P/E not meaningful: {_span_word(c)} diluted EPS is "
                    f"{_eps_text(c.eps)} (a loss)")
        text = (f"trailing P/E {c.trailing_pe:.1f} ({_price(c)} over {_span_word(c)} diluted "
                f"EPS ${c.eps:.2f})")
        if c.y_eps and abs(c.y_eps - c.eps) / abs(c.eps) > 0.1:
            text += f"; Yahoo Finance's own trailing EPS is {_eps_text(c.y_eps)}"
        return text
    if c.kind == "ifrs" and c.trailing_pe is not None:
        return (f"trailing P/E {c.trailing_pe:.1f} (Yahoo Finance's; a 20-F filer's per-share "
                "figures are per ordinary share, not per ADR, so it is not rebuilt from the "
                "filing)")
    if not c.yahoo:
        return "trailing P/E not available: Yahoo Finance did not answer for a price"
    return "trailing P/E not available: no diluted EPS tagged in its filings"


def _p_fpe(c: Company) -> str:
    if c.y_forward_pe is not None and c.y_forward_pe > 0:
        return f"forward P/E {c.y_forward_pe:.1f} (Yahoo Finance, on analysts' consensus EPS)"
    return ("forward P/E not available from Yahoo Finance"
            + ("" if c.yahoo else " (it did not answer)"))


def _p_evs(c: Company) -> str:
    if c.ev_sales is None:
        return "EV/sales not available (Yahoo Finance's enterprise value or revenue missing)"
    return (f"EV/sales {c.ev_sales:.2f} (Yahoo Finance's enterprise value {_usd(c.y_ev or 0)} "
            f"over {_usd(c.rev or 0)} of revenue)")


def _p_eps(c: Company) -> str:
    if c.eps is None:
        return "diluted EPS not tagged in its filings"
    if c.eps_prior is None:
        return f"diluted EPS ${c.eps:.2f}, with no prior period tagged"
    if c.eps_growth is None:
        return (f"diluted EPS ${c.eps:.2f} against ${c.eps_prior:.2f} the period before (a "
                "percentage is not given across a loss)")
    return f"diluted EPS ${c.eps:.2f}, {c.eps_growth:+.1%} on ${c.eps_prior:.2f} the period before"


def phrase(c: Company, metric: str) -> str | None:
    """What one company's figures say for one metric, or None for a metric that has no line."""
    if metric == "div":
        return _p_div(c)
    if metric == "payout":
        return _p_payout(c)
    if metric == "buyback":
        return _p_buyback(c)
    if metric == "returned":
        return _p_returned(c)
    if metric == "shares":
        return _p_shares(c)
    if metric in ("gross", "operating", "net"):
        return _p_margin(c, metric)
    if metric == "growth":
        return _p_growth(c)
    if metric == "fcf":
        return _p_fcf(c)
    if metric == "pe":
        return _p_pe(c)
    if metric == "fpe":
        return _p_fpe(c)
    if metric == "evs":
        return _p_evs(c)
    if metric == "eps":
        return _p_eps(c)
    return None


# metric -> (name in a ranking, value, format, highest first, note)
def _value(c: Company, metric: str) -> float | None:
    return {
        "div": c.yield_, "payout": c.payout, "buyback": c.buyback, "returned": c.returned,
        "shares": c.share_change, "gross": c.margin("gross"), "operating": c.margin("operating"),
        "net": c.margin("net"), "growth": c.growth, "fcf": c.fcf, "pe": c.trailing_pe,
        "fpe": c.y_forward_pe if c.y_forward_pe and c.y_forward_pe > 0 else None,
        "evs": c.ev_sales, "eps": c.eps_growth}.get(metric)


_RANK_NAME: Final[dict[str, tuple[str, str, bool]]] = {
    "div": ("dividend yield", "pct2", True), "payout": ("payout ratio", "pct0", True),
    "buyback": ("buybacks over the period", "usd", True),
    "returned": ("dividends plus buybacks over the period", "usd", True),
    "shares": ("change in share count", "pct2s", False),
    "gross": ("gross margin", "pct1", True), "operating": ("operating margin", "pct1", True),
    "net": ("net margin", "pct1", True), "growth": ("revenue growth", "pct1s", True),
    "fcf": ("free cash flow", "usd", True), "pe": ("trailing P/E", "x1", False),
    "fpe": ("forward P/E", "x1", False), "evs": ("EV/sales", "x2", False),
    "eps": ("EPS growth", "pct1s", True)}


def _fmt(kind: str, v: float) -> str:
    return {"pct2": f"{v:.2%}", "pct0": f"{v:.0%}", "pct1": f"{v:.1%}", "pct1s": f"{v:+.1%}",
            "pct2s": f"{v:+.2%}", "usd": _usd(v), "x1": f"{v:.1f}", "x2": f"{v:.2f}"}[kind]


def _ranking(cos: list[Company], metric: str) -> str | None:
    name, kind, high_first = _RANK_NAME[metric]
    values = [(c.ticker, v) for c in cos if (v := _value(c, metric)) is not None]
    if not values:
        return None
    ordered = sorted(values, key=lambda tv: tv[1], reverse=high_first)
    order = ("highest first" if high_first else
             "lowest first" if metric != "shares" else "biggest reduction first")
    nets = {c.ticker: c.net for c in cos}

    def said(t: str, v: float) -> str:
        if metric == "payout" and v > 3:
            return f"{t} {v * 100:,.0f}% (net income only {_usd(nets[t] or 0)})"
        return f"{t} {_fmt(kind, v)}"

    body = ", ".join(said(t, v) for t, v in ordered)
    absent = [c.ticker for c in cos if _value(c, metric) is None]
    tail = f"; no figure for {', '.join(absent)}" if absent else ""
    return f"{name} ({order}): {body}{tail}" if len(values) > 1 else f"{name}: {body}{tail}"


_PRIMARY_MARGIN: Final = ("operating", "gross", "net")


def _margin_trend(cos: list[Company], keys: Sequence[str]) -> str | None:
    """Which company's margin improved more, from the metrics asked, never a forecast."""
    for key in [k for k in _PRIMARY_MARGIN if k in keys]:
        rows = [(c, c.margin(key), c.margin(key, prior=True)) for c in cos]
        have = [(c, n, p) for c, n, p in rows if n is not None and p is not None]
        if len(have) < 2:
            continue
        best = max(have, key=lambda r: r[1] - r[2])
        body = "; ".join(f"{c.ticker} {_pct(n)} from {_pct(p)} ({_pts(n, p)})" for c, n, p in have)
        return (f"{key} margin trend: {body}; {best[0].ticker}'s improved the most. That is the "
                "record of the last period, not a forecast: a filing carries no margin guidance")
    return None


def _crypto_line(name: str, kind: str, metrics: Sequence[str]) -> str:
    """One stated answer per non-company asset: the figures an equity metric would need do not
    exist, and that is the answer, not a gap."""
    asked = set(metrics)
    parts: list[str] = []
    if asked & {"pe", "fpe", "eps", "evs"}:
        parts.append(f"{name} has no earnings, so it has no P/E ratio and no EPS to grow"
                     if kind == "crypto" else
                     f"{name} is a commodity that produces no earnings, so it has no P/E ratio "
                     "and no EPS")
    if asked & {"div", "payout", "returned"}:
        parts.append(f"{name} pays no dividend, so its dividend yield is nil: nothing is "
                     "distributed to holders" if kind == "crypto" else
                     f"{name} pays no dividend: a holder is paid only through the price")
    if asked & {"buyback", "shares"}:
        parts.append(f"{name} has no company behind it, so there is no buyback and no share "
                     "count to retire" if kind == "crypto" else
                     f"{name} has no shares to buy back")
    if asked & {"gross", "operating", "net", "growth", "fcf"}:
        parts.append(f"{name} has no income statement, so no margins, revenue or free cash flow")
    text = ". ".join(parts)
    if text and name == "Ether":
        text += (". Staking rewards are paid by the protocol to validators, not by a company out "
                 "of profit, so they are not a dividend and are not counted here")
    return text


def _ifrs_gaps(c: Company) -> list[str]:
    gaps = [name for name, value in (("gross profit", c.gross), ("operating profit", c.operating),
                                     ("capital spending", c.capex),
                                     ("a share count", c.shares_now), ("diluted EPS", c.eps),
                                     ("share repurchases", c.buyback))
            if value is None]
    return gaps


def _crypto_lines(assets: Sequence[tuple[str, str]], metrics: Sequence[str]) -> list[str]:
    return [line for name, kind in assets if (line := _crypto_line(name, kind, metrics))]


def asks(text: str) -> bool:
    """Whether this reader would take the question (no network)."""
    return _plan(text) is not None


def _plan(text: str) -> tuple[list[str], list[str], list[tuple[str, str]]] | None:
    if _WHY.search(text) or not CUE.search(text):
        return None
    metrics = asked_metrics(text)
    if not metrics:
        return None
    assets = _assets(text)
    tickers = resolve(text)
    if not tickers and not assets:
        return None
    if not _distinctive(metrics):
        # margins, growth, FCF and EPS are `filing_figures`'s when asked of one company's latest
        # filing; compared across companies, or with no filing word, they are this reader's
        if _FILING_WORDS.search(text) and len(tickers) < 2:
            return None
        if len(tickers) < 2 and not assets and not _COMPARE.search(text):
            return None
    return metrics, tickers[:MAX_COMPANIES], assets


def lines(text: str, *, facts_of: Callable[[str], dict[str, Any] | None] = _document,
          summary_of: Callable[[str], dict[str, Any]] = yahoo_summary,
          report_of: Callable[[str], tuple[str, date, date] | None]
          = filing_figures.latest_periodic_report) -> list[str] | None:
    """The comparison asked, or None when the question is not one of this reader's."""
    plan = _plan(text)
    if plan is None:
        return None
    metrics, tickers, assets = plan
    stated = _crypto_lines(assets, metrics)
    if not tickers:
        if not stated:
            return None
        return ["Bottom line: " + stated[0] + ".", *[s + "." for s in stated[1:]],
                "Data: no source was needed: these are not companies, so they file no financial "
                "statements and declare no dividends. Not advice."]
    cos = [read_company(t, facts_of=facts_of, summary_of=summary_of, report_of=report_of)
           for t in tickers]
    good = [c for c in cos if c.error is None]
    out: list[str] = []
    if not good:
        names = ", ".join(c.label for c in cos)
        return [f"Bottom line: {names}: the figures could not be read just now ("
                f"{cos[0].error}); ask again in a minute.",
                "Data: SEC EDGAR XBRL company facts and Yahoo Finance were tried. Not advice."]
    fragments: list[str] = []
    solo_cut = 3
    if len(good) > 1:
        trend = _margin_trend(good, metrics)
        if trend:
            fragments.append(trend)
        for m in metrics:
            if (m in ("gross", "operating", "net") and trend) or (
                    m in ("div", "buyback") and "returned" in metrics):
                continue
            ranked = _ranking(good, m)
            if ranked:
                fragments.append(ranked)
    else:
        solo = [p for m in metrics[:solo_cut] if (p := phrase(good[0], m))]
        fragments.append(f"{good[0].label}, {good[0].period}: " + "; ".join(solo))
    out.append("Bottom line: " + "; ".join(fragments[:4]) + ".")
    for c in cos:
        if c.error:
            out.append(f"{c.label}: {c.error}.")
            continue
        if len(good) == 1:
            rest = [p for m in metrics[solo_cut:] if (p := phrase(c, m))]
            if rest:
                out.append(f"{c.label}, also: " + "; ".join(rest) + ".")
            continue
        parts = [p for m in metrics if (p := phrase(c, m))]
        out.append(f"{c.label}, {c.period}: " + "; ".join(parts) + ".")
    for c in good:
        flag = _one_off(c, metrics) if {"pe", "eps", "net"} & set(metrics) else None
        if flag:
            out.append(flag + ".")
        out.extend(c.notes)
    if len(tickers) < len(resolve(text)):
        out.append(f"Only the first {MAX_COMPANIES} companies named are compared.")
    ifrs = [c for c in good if c.kind == "ifrs"]
    if ifrs:
        names = " and ".join(c.label for c in ifrs)
        gaps = "; ".join(f"{c.ticker} does not tag {', '.join(g)}" for c in ifrs
                         if (g := _ifrs_gaps(c)))
        out.append(
            f"{names} file{'s' if len(ifrs) == 1 else ''} a 20-F under IFRS, which carries annual "
            "figures only: the latest fiscal year against the one before, not trailing four "
            "quarters. " + (f"In the XBRL: {gaps}." if gaps else ""))
    if not all(c.yahoo for c in good):
        out.append("Yahoo Finance did not answer for every company, so price-based figures "
                   "(yield, P/E, buyback against market cap) are missing where noted.")
    out.extend(s + "." for s in stated)
    sources = [f"{c.ticker} {c.filed}" for c in good if c.filed]
    out.append("Data: each company's own XBRL filings via SEC EDGAR company facts ("
               + ("; ".join(sources) if sources else "10-Q and 10-K")
               + "), quarters rebuilt from year-to-date figures where the filing gives only those, "
               "and Yahoo Finance's quoteSummary for the price, market cap, indicated dividend "
               "and forward P/E. Not advice.")
    return out


__all__ = ["CUE", "Company", "asked_metrics", "asks", "lines", "phrase", "read_company",
           "resolve"]
