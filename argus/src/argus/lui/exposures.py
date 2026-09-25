"""Book exposures — sector, industry and factor — before and after a proposed trade.

**Why this module exists, measured before it was written.** The readiness audit of 2026-09-25 put
"what are my sector and factor exposures?" to the hosted console with a book of 40% MSFT, 30% META
and 30% GOOGL, and it was refused. The handbook's own Open Theme example asks for exactly this: "how
a proposed trade changes beta, sector and factor exposures, correlation, and concentration". The
console could already say beta to one index (`research._book_beta_line`) and risk share
(`desk.portfolio.copilot`); it could not say which sectors a book is in, which styles it leans on,
or how a trade moves either.

**Sector and industry — a frozen map, read, not remembered.** ``data/sector_map.json`` holds one
row for every stock and ETF perpetual Bitget lists (the venue's own ``isRwa`` flag, via
:mod:`argus.market.universe`), written by :func:`freeze_sector_map` from Yahoo Finance's
``quoteSummary`` (``assetProfile`` for a company's sector and industry; ``fundProfile`` and
``topHoldings`` for an ETF's category and the sector split of what it holds). Three decisions:

* **The sector names are GICS's, the classification is not.** GICS is licensed by MSCI and S&P and
  no public endpoint serves it. Yahoo's eleven sectors are Morningstar's scheme, and each maps one
  to one onto a GICS sector by name (Technology → Information Technology, Consumer Cyclical →
  Consumer Discretionary, Financial Services → Financials, ...: :data:`YAHOO_TO_GICS`). Membership
  can differ at the edges, so every answer says "Yahoo's classification, GICS names" rather than
  claiming GICS. SEC SIC codes were the alternative the brief named and were rejected: SIC puts
  Alphabet and Meta under 7370 "services-computer programming" beside Microsoft, which is the
  distinction a sector question is asking for.
* **A ticker match is checked against the price.** Bitget's symbol is not always the US ticker
  (``SPXUSDT`` is a memecoin, ``CVXUSDT`` is Convex — see :mod:`argus.market.universe`). A row is
  marked ``check: "price"`` only when Yahoo's last price for the ticker is within
  :data:`PRICE_CHECK_TOLERANCE` of Bitget's last for the perpetual; a listing outside the US
  (Samsung on the KRX, Tencent in Hong Kong) is mapped by :data:`YAHOO_TICKERS`, quoted in its own
  currency, and marked ``check: "name"`` with Yahoo's long name stored so a reader can see what
  it was matched to. A name that matched neither — a private company such as OpenAI, a GPU-rental
  index — is ``unclassified``, said as such, never guessed.
* **An ETF is looked through, not labelled.** QQQ is not a sector; its holdings are. An ETF row
  carries Yahoo's ``sectorWeightings`` scaled by its stock position, with bonds and cash as their
  own buckets, so a book half in QQQ reads about a quarter Information Technology. A leveraged or
  inverse fund's look-through is its holdings' split, not scaled by its leverage — the answer says
  so and the factor loadings (which see the leverage in the returns) carry it.

Crypto (every contract the venue does not flag ``isRwa``) and gold (``XAUUSDT``, ``XAUTUSDT``,
``PAXGUSDT``) are their own asset classes; other commodities, FX pairs and non-US index products
are named as such. ``SP500USDT`` and ``NDX100USDT`` are looked through SPY's and QQQ's rows.

**Factors — a time-series factor model on traded proxies.** Each holding's daily returns are
regressed on four factor returns at once, over the last :data:`WINDOW_DAYS` aligned US trading
days (at least :data:`MIN_OBS`):

* **market** — SPY, the S&P 500 (the console's ``SPYUSDT`` beta, `research._book_beta_line`);
* **size** — IWM minus SPY: small caps over large, so a negative loading is a large-cap tilt;
* **momentum** — MTUM (iShares MSCI USA Momentum) minus SPY;
* **crypto** — Bitget's BTCUSDT.

This is the Fama-French/Carhart time-series regression with ETF spreads standing in for the
academic long-short portfolios, which are published monthly with a lag and so cannot describe a
book today. It is FactorAnalytics' ``fitTsfm`` with ``fit.method="LS"`` and
``variable.selection="none"`` (`research/repos-themed/braverock~FactorAnalytics/R/fitTsfm.R:166`,
GPL-2: the method is used, none of its code). The book's loading on each factor is the weighted sum
of its holdings' loadings, which for least squares on one factor set is the book's own regression
— FactorAnalytics' ``portSdDecomp`` builds a portfolio's exposures the same way.

**Compared with the best general tool on this machine, and rejected in part.** Riskfolio-Lib's
``loadings_matrix``
(`research/repos-t2/dcajasn~riskfolio-lib/riskfolio/src/ParamsEstimation.py:779`,
BSD-3) fits the same regression but keeps a factor only if forward stepwise selection admits it at
p < 0.05 (`forward_regression`, line 354), setting every other loading to exactly zero. Run by
`argus.eval.exposures_comparison` on the same book and the same returns
(``data/exposures_comparison.json``, 252 days to 2026-09-25), its loadings equal ours to the
fourth decimal where it keeps all four factors (MSFT: market 1.2495, size -0.6269 in both); where
it drops one, it reports "no exposure" for a loading the data merely could not pin down (GOOGL's
size, -0.27 at t -1.8, becomes 0.00), and the loadings it keeps are refitted without the dropped
factor, so they absorb it (GOOGL's market beta moves from 1.67 to 1.59). A trader asking "am I
exposed to X" needs the estimate and its uncertainty, not a zero,
so ARGUS keeps every factor and prints each loading's t-statistic instead. Its OLS is re-implemented
here in pure Python because shipped ARGUS code carries no numpy (``pyproject.toml``).

**Stated limits, on the answer and here.** Crypto's daily bar on Bitget closes at 16:00 UTC, four
hours before the US close (20:00 UTC in summer), and a Korean or Japanese listing closes before New
York opens; loadings across those clocks are biased toward zero, and the lines say which closes
were used. Returns are raw, not in excess of cash: at about 0.02% a day, cash moves no loading at
the precision shown.
"""

from __future__ import annotations

import itertools
import json
import math
import re
import sys
import threading
import time
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from argus.lui.answer import Answer, Source

SECTOR_MAP_PATH = Path(__file__).resolve().parents[3] / "data" / "sector_map.json"
TICKERS_URL = "https://api.bitget.com/api/v2/mix/market/tickers?productType=USDT-FUTURES"

WINDOW_DAYS = 252
"""One year of US trading days: long enough that a loading's standard error is a few hundredths,
short enough that a company that changed its business two years ago is read as it is now."""

MIN_OBS = 60
"""Fewer aligned days than this and a four-factor loading is noise; the holding is reported as
having too little history rather than given a number."""

PRICE_CHECK_TOLERANCE = 0.15
"""Yahoo's last price and Bitget's last price may differ by this much and still be the same
instrument: rTokens track their stock within a few percent even off-hours, and a ticker collision
(a memecoin at 0.50 against a stock at 300) misses by orders of magnitude."""

FETCH_DEADLINE_S = 25.0

FACTORS = ("market", "size", "momentum", "crypto")
FACTOR_WORDS = {
    "market": "Market beta (S&P 500, SPY)",
    "size": "Size beta (IWM minus SPY; above 0 leans small-cap, below 0 large-cap)",
    "momentum": "Momentum beta (MTUM minus SPY; above 0 leans to recent winners)",
    "crypto": "Crypto beta (Bitget BTCUSDT)",
}

YAHOO_TO_GICS: dict[str, str] = {
    "Technology": "Information Technology",
    "Communication Services": "Communication Services",
    "Consumer Cyclical": "Consumer Discretionary",
    "Consumer Defensive": "Consumer Staples",
    "Financial Services": "Financials",
    "Healthcare": "Health Care",
    "Basic Materials": "Materials",
    "Industrials": "Industrials",
    "Energy": "Energy",
    "Utilities": "Utilities",
    "Real Estate": "Real Estate",
}
"""Yahoo's (Morningstar's) eleven sectors onto the GICS sector of the same scope, by name."""

_WEIGHT_KEYS: dict[str, str] = {
    "technology": "Information Technology",
    "communication_services": "Communication Services",
    "consumer_cyclical": "Consumer Discretionary",
    "consumer_defensive": "Consumer Staples",
    "financial_services": "Financials",
    "healthcare": "Health Care",
    "basic_materials": "Materials",
    "industrials": "Industrials",
    "energy": "Energy",
    "utilities": "Utilities",
    "realestate": "Real Estate",
}
"""The keys of Yahoo's ``topHoldings.sectorWeightings`` for an ETF, onto the same GICS names."""

YAHOO_TICKERS: dict[str, str] = {
    # Korea (KRX), quoted in KRW
    "SAMSUNG": "005930.KS", "SAMSUNGEM": "009150.KS", "SKHYNIX": "000660.KS",
    "HYUNDAI": "005380.KS", "NAVER": "035420.KS", "LGELECTRONICS": "066570.KS",
    "HANMI": "042700.KS", "DOOSBOT": "454910.KS", "DOOSENER": "034020.KS",
    # Hong Kong (HKEX), quoted in HKD
    "TENCENT": "0700.HK", "TENCENTHKD": "0700.HK", "XIAOMI": "1810.HK", "XIAOMIHKD": "1810.HK",
    "MEITUAN": "3690.HK", "KUAISHOU": "1024.HK", "NETEASE": "9999.HK", "BYD": "1211.HK",
    "POPMART": "9992.HK", "LENOVOHKD": "0992.HK", "SMIC": "0981.HK",
    # Japan (TSE), quoted in JPY
    "SOFTBANK": "9984.T", "TOKYOEL": "8035.T", "ADVANTEST": "6857.T", "LASERTEC": "6920.T",
    "KIOXIA": "285A.T", "SUMIELEC": "5802.T",
    # Mainland China (SSE / SZSE), quoted in CNY
    "GIGADEVICE": "603986.SS", "ZHONGJI": "300308.SZ",
    # found by Yahoo's own search (``/v1/finance/search``) on 2026-09-26 and checked by name
    "MINIMAX": "0100.HK", "MINIMAXHKD": "0100.HK", "SHEIN": "0625.HK", "SHEINHKD": "0625.HK",
    "CSOPSK2LHKD": "7709.HK", "CSOPSS2LHKD": "7747.HK", "BRKB": "BRK-B",
}
"""Bitget names that are not a US ticker, onto the home listing's Yahoo symbol. Every one is
checked by name when the map is frozen: the row stores Yahoo's long name, and
`tests/test_exposures.py` asserts the ones a wrong mapping would most likely hit."""

GOLD = frozenset({"XAUUSDT", "XAUTUSDT", "PAXGUSDT"})
INDEX_LOOK_THROUGH = {"SP500USDT": "SPYUSDT", "NDX100USDT": "QQQUSDT"}
"""An index product is looked through the ETF that tracks the same index."""

INDEX_PRICE_PROXY = {"SP500USDT": "SPY", "NDX100USDT": "QQQ", "DIASTOCKUSDT": "DIA"}


# --- the sector map ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Classification:
    """What one contract is, for exposure purposes."""

    symbol: str
    kind: str
    """``stock`` | ``etf`` | ``crypto`` | ``gold`` | ``commodity`` | ``fx`` | ``index`` |
    ``unclassified``."""

    weights: Mapping[str, float]
    """Sector or asset-class buckets, summing to one."""

    industry: str | None = None
    name: str | None = None
    ticker: str | None = None
    """The Yahoo symbol whose daily closes stand for this contract, when it is an equity."""

    leveraged: bool = False
    note: str = ""


_MAP_CACHE: dict[str, Any] | None = None
_MAP_LOCK = threading.Lock()


def sector_map(path: Path = SECTOR_MAP_PATH) -> dict[str, Any]:
    """The frozen map: ``{"generated_at", "source", "rows": {symbol: row}}``."""
    global _MAP_CACHE
    with _MAP_LOCK:
        if _MAP_CACHE is None or path != SECTOR_MAP_PATH:
            loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
            if path != SECTOR_MAP_PATH:
                return loaded
            _MAP_CACHE = loaded
        return _MAP_CACHE


def _is_rwa(symbol: str) -> bool | None:
    from argus.market.universe import contracts

    contract = contracts().get(symbol)
    return None if contract is None else contract.rwa


def classify(symbol: str, rows: Mapping[str, Any] | None = None) -> Classification:
    """One contract's sector or asset-class buckets, from the frozen map or the venue's flags."""
    from argus.market.universe import NOT_EQUITY

    rows = sector_map()["rows"] if rows is None else rows
    if symbol in GOLD:
        return Classification(symbol, "gold", {"Gold": 1.0}, name="gold")
    if symbol in INDEX_LOOK_THROUGH and INDEX_LOOK_THROUGH[symbol] in rows:
        proxy = classify(INDEX_LOOK_THROUGH[symbol], rows)
        return Classification(symbol, "index", proxy.weights, name=proxy.name,
                              ticker=INDEX_PRICE_PROXY[symbol],
                              note=f"looked through {proxy.ticker}, which tracks the same index")
    kind = NOT_EQUITY.get(symbol)
    if kind == "commodity":
        return Classification(symbol, "commodity", {"Commodities": 1.0})
    if kind == "fx":
        return Classification(symbol, "fx", {"FX": 1.0})
    if kind == "index":
        return Classification(symbol, "index", {"Non-US equity index": 1.0})
    row = rows.get(symbol)
    if row is None:
        rwa = _is_rwa(symbol)
        if rwa is False:
            return Classification(symbol, "crypto", {"Crypto": 1.0})
        return Classification(symbol, "unclassified", {"Unclassified": 1.0},
                              note="not in the sector map")
    weights = {str(k): float(v) for k, v in (row.get("weights") or {}).items() if float(v) > 0}
    if not weights:
        weights = {"Unclassified": 1.0}
    return Classification(symbol, str(row.get("kind") or "unclassified"), weights,
                          industry=row.get("industry"), name=row.get("name"),
                          ticker=row.get("ticker"), leveraged=bool(row.get("leveraged")),
                          note=str(row.get("note") or ""))


def _raw(node: Any) -> float | None:
    value = node.get("raw") if isinstance(node, dict) else node
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def row_from_summary(summary: Mapping[str, Any], *, ticker: str,
                     bitget_last: float | None) -> dict[str, Any]:
    """One map row from a Yahoo ``quoteSummary`` result. Pure: the tests feed it recorded
    payloads."""
    price = summary.get("price") or {}
    profile = summary.get("assetProfile") or {}
    quote_type = str((summary.get("quoteType") or {}).get("quoteType") or "")
    currency = str(price.get("currency") or "")
    last = _raw(price.get("regularMarketPrice"))
    name = price.get("longName") or price.get("shortName")
    row: dict[str, Any] = {"ticker": ticker, "name": name, "yahoo_type": quote_type,
                           "currency": currency, "yahoo_last": last, "bitget_last": bitget_last}
    if currency == "USD" and last and bitget_last:
        gap = abs(last / bitget_last - 1.0)
        row["check"] = "price" if gap <= PRICE_CHECK_TOLERANCE else "failed"
        if row["check"] == "failed":
            row.update(kind="unclassified", weights={"Unclassified": 1.0},
                       note=f"Yahoo's {ticker} is at {last:g} {currency}, Bitget's contract at "
                            f"{bitget_last:g}: not the same instrument")
            return row
    else:
        row["check"] = "name"
    if quote_type == "EQUITY":
        sector = YAHOO_TO_GICS.get(str(profile.get("sector") or ""))
        row.update(kind="stock", industry=profile.get("industry") or None,
                   yahoo_sector=profile.get("sector") or None,
                   weights={sector: 1.0} if sector else {"Unclassified": 1.0})
        if not sector:
            row["note"] = "Yahoo gives no sector for it"
        return row
    if quote_type == "ETF":
        fund = summary.get("fundProfile") or {}
        holdings = summary.get("topHoldings") or {}
        category = str(fund.get("categoryName") or "")
        stock = _raw(holdings.get("stockPosition")) or 0.0
        bond = _raw(holdings.get("bondPosition")) or 0.0
        weights: dict[str, float] = {}
        for item in holdings.get("sectorWeightings") or []:
            for key, node in item.items():
                share = _raw(node)
                if key in _WEIGHT_KEYS and share:
                    weights[_WEIGHT_KEYS[key]] = weights.get(_WEIGHT_KEYS[key], 0.0) + share * stock
        if bond > 0.005:
            weights["Bonds"] = bond
        lowered = f"{category} {name or ''}".lower()
        # "Ultrashort Bond" is SGOV's category (T-bills), not a leveraged short: word-bounded
        leveraged = bool(re.search(r"trading--|leverag|inverse|\b-?[23]x\b|\bultra(?:pro)?\b|"
                                   r"\bultrashort\s+(?!bond)|\bbull\b|\bbear\b|daily\s+max",
                                   lowered))
        if "digital assets" in lowered or re.search(r"\bbitcoin\b|\bether\b", lowered):
            weights = {"Crypto": 1.0}
        total = sum(weights.values())
        if total < 0.5:
            # a fund whose look-through Yahoo does not publish (volatility futures, swaps)
            weights = {f"ETF: {category or 'no category'}": 1.0}
        else:
            weights = {k: v / total for k, v in weights.items()}
        row.update(kind="etf", category=category or None, leveraged=leveraged, weights=weights)
        if leveraged:
            row["note"] = ("a leveraged or inverse fund: the split is its holdings', not scaled by "
                           "its leverage")
        return row
    row.update(kind="unclassified", weights={"Unclassified": 1.0},
               note=f"Yahoo types {ticker} as {quote_type or 'nothing'}")
    return row


def _bitget_lasts() -> dict[str, float]:
    request = urllib.request.Request(TICKERS_URL, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=10) as response:
        payload = json.loads(response.read().decode("utf-8"))
    out: dict[str, float] = {}
    for item in payload.get("data") or []:
        try:
            out[str(item["symbol"])] = float(item["lastPr"])
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _yahoo_ticker(symbol: str) -> str:
    from argus.market.universe import contracts

    contract = contracts().get(symbol)
    base = contract.name if contract is not None else symbol.removesuffix("USDT")
    return YAHOO_TICKERS.get(base, base.replace(".", "-"))


def freeze_sector_map(path: Path = SECTOR_MAP_PATH, *, workers: int = 4) -> dict[str, int]:
    """Write the map for every stock and ETF perpetual Bitget lists, and count what came back."""
    from argus.market.estimates import EstimatesSource
    from argus.market.universe import contracts, is_equity, origin

    listed = contracts()
    symbols = sorted(s for s in listed if is_equity(s))
    lasts = _bitget_lasts()
    local = threading.local()

    def one(symbol: str) -> tuple[str, dict[str, Any]]:
        source = getattr(local, "source", None)
        if source is None:
            source = local.source = EstimatesSource()
        ticker = _yahoo_ticker(symbol)
        try:
            summary = source.summary(ticker, "assetProfile,price,quoteType,fundProfile,topHoldings")
        except Exception as exc:
            return symbol, {"ticker": ticker, "kind": "unclassified", "check": "none",
                            "weights": {"Unclassified": 1.0},
                            "note": f"Yahoo has no {ticker}: {type(exc).__name__}"}
        return symbol, row_from_summary(summary, ticker=ticker, bitget_last=lasts.get(symbol))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = dict(pool.map(one, symbols))
    counts: dict[str, int] = {}
    for row in rows.values():
        key = f"{row.get('kind')}/{row.get('check')}"
        counts[key] = counts.get(key, 0) + 1
    path.write_text(json.dumps({
        "generated_at": datetime.now(UTC).isoformat(),
        "source": ("Yahoo Finance quoteSummary (assetProfile, price, quoteType, fundProfile, "
                   "topHoldings) per contract; Bitget USDT-futures contracts (isRwa) and tickers"),
        "universe": f"{len(symbols)} stock and ETF perpetuals, registry {origin()}",
        "sector_names": "Yahoo's (Morningstar's) sectors under their GICS names; not GICS itself",
        "counts": counts,
        "rows": rows,
    }, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    global _MAP_CACHE
    with _MAP_LOCK:
        _MAP_CACHE = None
    return counts


# --- the factor model -------------------------------------------------------------------------


@dataclass(frozen=True)
class Fit:
    """One holding's regression on the four factors."""

    loadings: Mapping[str, float]
    t_stats: Mapping[str, float]
    r_squared: float
    n: int
    start: date
    end: date


def _invert(matrix: list[list[float]]) -> list[list[float]] | None:
    """Gauss-Jordan with partial pivoting; None for a singular matrix."""
    size = len(matrix)
    work = [row[:] + [1.0 if i == j else 0.0 for j in range(size)] for i, row in enumerate(matrix)]
    for col in range(size):
        pivot = max(range(col, size), key=lambda r: abs(work[r][col]))
        if abs(work[pivot][col]) < 1e-18:
            return None
        work[col], work[pivot] = work[pivot], work[col]
        lead = work[col][col]
        work[col] = [v / lead for v in work[col]]
        for r in range(size):
            if r != col and work[r][col] != 0.0:
                factor = work[r][col]
                work[r] = [a - factor * b for a, b in zip(work[r], work[col], strict=True)]
    return [row[size:] for row in work]


def ols(y: Sequence[float], xs: Sequence[Sequence[float]]
        ) -> tuple[list[float], list[float], float] | None:
    """Least squares of ``y`` on a constant and ``xs``: (coefficients, standard errors, R²), the
    constant first. None when the design is singular or leaves no degrees of freedom."""
    n, k = len(y), len(xs) + 1
    if n <= k + 1 or any(len(x) != n for x in xs):
        return None
    cols: list[Sequence[float]] = [[1.0] * n, *xs]
    xtx = [[math.fsum(a * b for a, b in zip(ci, cj, strict=True)) for cj in cols] for ci in cols]
    xty = [math.fsum(a * b for a, b in zip(ci, y, strict=True)) for ci in cols]
    inverse = _invert(xtx)
    if inverse is None:
        return None
    coef = [math.fsum(inverse[i][j] * xty[j] for j in range(k)) for i in range(k)]
    resid = [y[t] - math.fsum(coef[i] * cols[i][t] for i in range(k)) for t in range(n)]
    sse = math.fsum(r * r for r in resid)
    mean = math.fsum(y) / n
    sst = math.fsum((v - mean) ** 2 for v in y)
    s2 = sse / (n - k)
    ses = [math.sqrt(max(inverse[i][i] * s2, 0.0)) for i in range(k)]
    return coef, ses, (1.0 - sse / sst) if sst > 0 else 0.0


Closes = Mapping[date, float]


def factor_returns(factor_closes: Mapping[str, Closes]
                   ) -> tuple[list[date], dict[str, list[float]]]:
    """The four factor return series on the dates all four share, closes aligned before returns
    are taken so a weekend is one return on every series (Friday to Monday)."""
    spy, iwm, mtum, btc = (factor_closes[k] for k in ("SPY", "IWM", "MTUM", "BTC"))
    days = sorted(set(spy) & set(iwm) & set(mtum) & set(btc))
    out: dict[str, list[float]] = {f: [] for f in FACTORS}
    stamps: list[date] = []
    for before, after in itertools.pairwise(days):
        r_spy = spy[after] / spy[before] - 1.0
        out["market"].append(r_spy)
        out["size"].append(iwm[after] / iwm[before] - 1.0 - r_spy)
        out["momentum"].append(mtum[after] / mtum[before] - 1.0 - r_spy)
        out["crypto"].append(btc[after] / btc[before] - 1.0)
        stamps.append(after)
    return stamps, out


def fit_holding(closes: Closes, factor_closes: Mapping[str, Closes], *,
                window: int = WINDOW_DAYS) -> Fit | None:
    """One holding's loadings on the last ``window`` dates it shares with every factor."""
    spy, iwm, mtum, btc = (factor_closes[k] for k in ("SPY", "IWM", "MTUM", "BTC"))
    days = sorted(set(closes) & set(spy) & set(iwm) & set(mtum) & set(btc))[-(window + 1):]
    if len(days) - 1 < MIN_OBS:
        return None
    sub = {k: {d: v[d] for d in days} for k, v in factor_closes.items()}
    _, factors = factor_returns(sub)
    y = [closes[b] / closes[a] - 1.0 for a, b in itertools.pairwise(days)]
    fitted = ols(y, [factors[f] for f in FACTORS])
    if fitted is None:
        return None
    coef, ses, r2 = fitted
    loadings = {f: coef[i + 1] for i, f in enumerate(FACTORS)}
    t_stats = {f: coef[i + 1] / ses[i + 1] if ses[i + 1] > 0 else 0.0
               for i, f in enumerate(FACTORS)}
    return Fit(loadings, t_stats, r2, len(y), days[1], days[-1])


# --- data -------------------------------------------------------------------------------------


_BITGET_CACHE: dict[str, tuple[float, dict[date, float]]] = {}
_BITGET_LOCK = threading.Lock()
CACHE_TTL_S = 6 * 3600.0


def _bitget_daily(symbol: str) -> dict[date, float]:
    """Bitget daily closes keyed by the UTC date on which the bar closes (16:00 UTC)."""
    from argus.market.history import CandleType, fetch_window

    now = time.monotonic()
    with _BITGET_LOCK:
        hit = _BITGET_CACHE.get(symbol)
        if hit and now - hit[0] < CACHE_TTL_S:
            return hit[1]
    # the recent endpoint returns about 90 daily bars; a year of them needs the paged history
    bars = fetch_window(symbol, start=datetime.now(UTC) - timedelta(days=400), interval="1D",
                        candle_type=CandleType.MARKET, pause=0.05)
    out = {(b.ts + timedelta(days=1)).date(): float(b.close) for b in bars if float(b.close) > 0}
    if len(out) < 30:
        raise RuntimeError(f"only {len(out)} daily bars for {symbol}")
    with _BITGET_LOCK:
        _BITGET_CACHE[symbol] = (time.monotonic(), out)
    return out


def _yahoo_daily(ticker: str) -> dict[date, float]:
    from argus.market import equity_history

    return {d.day: d.close for d in equity_history.daily(ticker)}


def closes_for(symbol: str, rows: Mapping[str, Any] | None = None) -> tuple[dict[date, float], str]:
    """Daily closes for one contract and whose they are: an equity's own listing on Yahoo
    (split-adjusted; the rToken tracks it, and it reaches back years), anything else Bitget's
    daily candles — the same split `research._daily_closes` makes, keyed by date so series can
    be aligned."""
    info = classify(symbol, rows)
    if info.ticker and info.kind in ("stock", "etf", "index"):
        try:
            return _yahoo_daily(info.ticker), f"{info.ticker} daily closes (Yahoo)"
        except Exception:
            pass  # a listing younger than a year: the perpetual's own candles below
    return _bitget_daily(symbol), f"{symbol} daily candles (Bitget, 16:00 UTC close)"


def factor_closes() -> dict[str, dict[date, float]]:
    return {"SPY": _yahoo_daily("SPY"), "IWM": _yahoo_daily("IWM"),
            "MTUM": _yahoo_daily("MTUM"), "BTC": _bitget_daily("BTCUSDT")}


# --- the answer -------------------------------------------------------------------------------


def after_trade(book: Mapping[str, float], proposed: Mapping[str, float]) -> dict[str, float]:
    """The book once each proposed name sits at its stated weight, the rest scaled pro rata.

    Generalises `desk.portfolio.rebalance` (adds only) to trims and exits: a name already held at
    w0 and moved to w scales every other holding by (1 - w) / (1 - w0)."""
    out = {s: float(w) for s, w in book.items()}
    for symbol, target in proposed.items():
        held = out.get(symbol, 0.0)
        rest = 1.0 - held
        scale = (1.0 - target) / rest if rest > 1e-12 else 0.0
        out = {s: w * scale for s, w in out.items() if s != symbol}
        if target > 0:
            out[symbol] = target
        if not out:
            return {}
    return {s: w for s, w in out.items() if abs(w) > 1e-9}


def sector_weights(book: Mapping[str, float], rows: Mapping[str, Any] | None = None
                   ) -> dict[str, float]:
    out: dict[str, float] = {}
    for symbol, weight in book.items():
        for bucket, share in classify(symbol, rows).weights.items():
            out[bucket] = out.get(bucket, 0.0) + weight * share
    return dict(sorted(out.items(), key=lambda kv: -abs(kv[1])))


def effective_number(weights: Mapping[str, float]) -> float:
    """1 / Σw² over absolute weights scaled to one: 1 for a single name, N for N equal names."""
    gross = sum(abs(w) for w in weights.values())
    if gross <= 0:
        return 0.0
    return 1.0 / sum((abs(w) / gross) ** 2 for w in weights.values())


def book_loadings(book: Mapping[str, float], fits: Mapping[str, Fit | None]
                  ) -> tuple[dict[str, float], float]:
    """The weighted sum of holding loadings, and the share of the book it covers."""
    covered = sum(abs(w) for s, w in book.items() if fits.get(s) is not None)
    out = {f: math.fsum(w * fit.loadings[f] for s, w in book.items()
                        if (fit := fits.get(s)) is not None) for f in FACTORS}
    return out, covered


def _book_series(book: Mapping[str, float], closes: Mapping[str, Closes]
                 ) -> dict[date, float]:
    held = [s for s in book if s in closes]
    if not held:
        return {}
    days = sorted(set.intersection(*(set(closes[s]) for s in held)))[-(WINDOW_DAYS + 1):]
    out: dict[date, float] = {}
    for a, b in itertools.pairwise(days):
        out[b] = math.fsum(book[s] * (closes[s][b] / closes[s][a] - 1.0) for s in held)
    return out


def _correlation(xs: Mapping[date, float], ys: Mapping[date, float]) -> tuple[float, int] | None:
    from argus.desk.portfolio import correlation

    days = sorted(set(xs) & set(ys))
    if len(days) < MIN_OBS:
        return None
    value = correlation([xs[d] for d in days], [ys[d] for d in days])
    return None if value is None else (value, len(days))


def _t(symbol: str) -> str:
    base = symbol.removesuffix("USDT")
    return base.removesuffix("STOCK") if base.endswith("STOCK") and len(base) > 5 else base


def _tstat(value: float) -> str:
    """A t-statistic as read: a holding that is the factor itself (BTC on BTC) has no residual."""
    return f"{value:+.1f}" if abs(value) < 100 else ("above +99" if value > 0 else "below -99")


def _pct(value: float) -> str:
    return f"{value:.0%}"


@dataclass
class Inputs:
    """Everything an answer reads, fetched once; injected in tests."""

    closes: dict[str, dict[date, float]]
    factors: dict[str, dict[date, float]]
    origins: dict[str, str]
    missing: dict[str, str] = field(default_factory=dict)


def fetch_inputs(symbols: Sequence[str], rows: Mapping[str, Any] | None = None) -> Inputs:
    """Daily closes for every name and factor, fetched concurrently under a deadline."""
    out: dict[str, dict[date, float]] = {}
    origins: dict[str, str] = {}
    missing: dict[str, str] = {}
    pool = ThreadPoolExecutor(max_workers=4)
    try:
        factors_future = pool.submit(factor_closes)
        futures = {s: pool.submit(closes_for, s, rows) for s in symbols}
        deadline = time.monotonic() + FETCH_DEADLINE_S
        for symbol, future in futures.items():
            try:
                series, origin = future.result(timeout=max(0.1, deadline - time.monotonic()))
                out[symbol], origins[symbol] = series, origin
            except Exception as exc:
                missing[symbol] = type(exc).__name__
        factors = factors_future.result(timeout=max(0.1, deadline - time.monotonic()))
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    return Inputs(out, factors, origins, missing)


def _signed(value: float) -> str:
    return f"{value:+.2f}".replace("+", "+" if value >= 0 else "")


def _move(before: float, after: float | None) -> str:
    return f"{before:.2f}" if after is None else f"{before:.2f} → {after:.2f}"


def exposures_answer(
    book: Mapping[str, float],
    proposed: Mapping[str, float] | None = None,
    *,
    inputs: Inputs | None = None,
    rows: Mapping[str, Any] | None = None,
    notes: Sequence[str] = (),
) -> tuple[list[str], list[Source], dict[str, Any]]:
    """Sector weights and factor loadings of ``book``, and of the book after ``proposed`` (each
    name to its stated target weight, the rest pro rata), with the biggest change, concentration
    and one ``Actionable:`` lead. Returns (lines, sources, data) like every research engine.

    ``book`` maps Bitget symbols to weights summing to one; ``proposed`` maps symbols to target
    weights in [0, 1]."""
    if rows is None:
        frozen = sector_map()
        rows, map_date = frozen["rows"], str(frozen.get("generated_at", ""))[:10]
    else:
        map_date = "the given map"
    before = {s: float(w) for s, w in book.items() if abs(w) > 1e-9}
    after = after_trade(before, proposed) if proposed else None
    names = list(dict.fromkeys([*before, *(proposed or {})]))
    if inputs is None:
        inputs = fetch_inputs(names, rows)

    fits: dict[str, Fit | None] = {}
    for symbol in names:
        series = inputs.closes.get(symbol)
        fits[symbol] = None if series is None else fit_holding(series, inputs.factors)
    load_b, cover_b = book_loadings(before, fits)
    load_a, cover_a = book_loadings(after, fits) if after is not None else ({}, 0.0)
    sec_b = sector_weights(before, rows)
    sec_a = sector_weights(after, rows) if after is not None else {}
    eff_b = effective_number(before)
    eff_a = effective_number(after) if after is not None else None
    eff_sec_b = effective_number(sec_b)
    eff_sec_a = effective_number(sec_a) if after is not None else None

    trade_words = ""
    if proposed:
        parts = []
        for symbol, target in proposed.items():
            held = before.get(symbol, 0.0)
            if target <= 0:
                parts.append(f"selling all {_t(symbol)}")
            elif held <= 0:
                parts.append(f"adding {_pct(target)} {_t(symbol)}")
            else:
                parts.append(f"moving {_t(symbol)} from {_pct(held)} to {_pct(target)}")
        trade_words = " and ".join(parts)

    lines: list[str] = []
    buckets = list(dict.fromkeys([*sec_b, *sec_a]))
    changes = sorted(((b, sec_a.get(b, 0.0) - sec_b.get(b, 0.0)) for b in buckets),
                     key=lambda kv: -abs(kv[1]))
    factor_changes = sorted(((f, load_a[f] - load_b[f]) for f in FACTORS),
                            key=lambda kv: -abs(kv[1])) if after is not None else []

    # the lead
    top_b = next(iter(sec_b.items()), ("nothing", 0.0))
    if after is not None and changes:
        bucket = changes[0][0]
        factor, fdelta = factor_changes[0]
        direction = "more" if (eff_a or 0) > eff_b else "less"
        lead = (f"Actionable: {trade_words} moves {bucket} from {_pct(sec_b.get(bucket, 0.0))} to "
                f"{_pct(sec_a.get(bucket, 0.0))} of the book — the biggest sector change — and "
                f"market beta from {load_b['market']:.2f} to {load_a['market']:.2f}; the largest "
                f"factor change is {factor} ({_signed(fdelta)}), and the book becomes {direction} "
                f"spread by name ({eff_b:.1f} → {(eff_a or 0):.1f} effective positions).")
        if (eff_sec_a or 0) < eff_sec_b and max(sec_a.values(), default=0.0) >= 0.5:
            heavy = max(sec_a.items(), key=lambda kv: kv[1])
            lead += (f" {heavy[0]} would be {_pct(heavy[1])} of the book; size the trade down if "
                     f"that is more than you mean to hold in one sector.")
    else:
        tilt = max(("size", "momentum", "crypto"), key=lambda f: abs(load_b[f]))
        lead = (f"Actionable: your book is {_pct(top_b[1])} {top_b[0]}"
                + (f" across {eff_sec_b:.1f} effective sectors" if len(sec_b) > 1 else "")
                + f"; its market beta is {load_b['market']:.2f} to the S&P 500 and its largest "
                  f"style tilt is {tilt} at {_signed(load_b[tilt])}.")
        if top_b[1] >= 0.5 and len(before) > 1:
            lead += (" Half or more of it rides on one sector, so the names do not diversify "
                     "each other much; a holding from another sector would.")
    lines.append(lead)

    # sectors
    shown = ", ".join(f"{b} {_pct(w)}" for b, w in sec_b.items() if abs(w) >= 0.005)
    lines.append(f"Sector weights of the book now (Yahoo's classification, GICS names): {shown}.")
    if after is not None:
        moved = ", ".join(f"{b} {_pct(sec_b.get(b, 0.0))} → {_pct(sec_a.get(b, 0.0))}"
                          for b, d in changes if abs(d) >= 0.005)
        lines.append(f"After the trade: {moved or 'no sector weight moves by half a point'}.")
    industries: dict[str, list[str]] = {}
    for symbol in names:
        info = classify(symbol, rows)
        category = (rows.get(symbol) or {}).get("category") if info.kind == "etf" else None
        label = info.industry or (f"ETF, {category}" if category else info.kind)
        industries.setdefault(label, []).append(_t(symbol))
    lines.append("Industries: " + "; ".join(f"{k} ({', '.join(v)})"
                                            for k, v in industries.items()) + ".")

    # factors
    for factor in FACTORS:
        t_bits = [f"{_t(s)} t {_tstat(fit.t_stats[factor])}" for s in names
                  if (fit := fits.get(s)) is not None]
        lines.append(f"{FACTOR_WORDS[factor]}: "
                     f"{_move(load_b[factor], load_a[factor] if after is not None else None)} "
                     f"for the book; per holding "
                     + ", ".join(f"{_t(s)} {fit.loadings[factor]:.2f}" for s in names
                                 if (fit := fits.get(s)) is not None)
                     + (f" ({'; '.join(t_bits)})" if factor != "market" else "") + ".")
    fitted = [f for f in fits.values() if f is not None]
    if fitted:
        n_lo, n_hi = min(f.n for f in fitted), max(f.n for f in fitted)
        start = min(f.start for f in fitted)
        end = max(f.end for f in fitted)
        lines.append(
            f"Assumed: loadings fitted over the last {WINDOW_DAYS} US trading days available "
            f"({start.isoformat()} to {end.isoformat()}, {n_lo}"
            + (f" to {n_hi}" if n_hi != n_lo else "")
            + " daily returns per holding), all four factors at once; |t| above 2 is a loading "
              "the data pins down, below 2 one it cannot tell from zero.")
    thin = [s for s in names if fits.get(s) is None and s in inputs.closes]
    lost = [s for s in names if s not in inputs.closes]
    if thin:
        lines.append(f"Missing: {', '.join(_t(s) for s in thin)} has too few daily closes "
                     f"aligned with the factors (under {MIN_OBS}) for a loading.")
    if lost:
        lines.append(f"Missing: {', '.join(_t(s) for s in lost)} — daily closes could not be "
                     f"read, so the book's loadings cover {_pct(cover_b)} of it.")

    # concentration and correlation
    conc = (f"effective positions {eff_b:.1f}"
            + (f" → {eff_a:.1f}" if eff_a is not None else "")
            + f" (1/Σw²), effective sectors {eff_sec_b:.1f}"
            + (f" → {eff_sec_a:.1f}" if eff_sec_a is not None else ""))
    largest_b = max(before.items(), key=lambda kv: abs(kv[1]))
    conc += f"; largest holding {_t(largest_b[0])} {_pct(largest_b[1])}"
    if after:
        largest_a = max(after.items(), key=lambda kv: abs(kv[1]))
        conc += f" → {_t(largest_a[0])} {_pct(largest_a[1])}"
    if after is not None:
        lines.append(f"After the trade, concentration: {conc}.")
    else:
        lines.append(f"Concentration: {conc}.")
    if proposed:
        book_series = _book_series(before, inputs.closes)
        for symbol in proposed:
            other = inputs.closes.get(symbol)
            if symbol in before and len(before) == 1:
                continue
            rest = {s: w for s, w in before.items() if s != symbol}
            total = sum(rest.values())
            series = (_book_series({s: w / total for s, w in rest.items()}, inputs.closes)
                      if symbol in before and total > 0 else book_series)
            pair = None if other is None or not series else _correlation(
                {d: other[d] / other[p] - 1.0
                 for p, d in itertools.pairwise(sorted(other))}, series)
            if pair is not None:
                lines.append(f"{_t(symbol)}'s daily correlation {pair[0]:+.2f} with the rest of "
                             f"the book over {pair[1]} shared days: "
                             + ("it moves with what you hold, so it adds to the same risk."
                                if pair[0] >= 0.6 else
                                "it moves partly apart from what you hold." if pair[0] >= 0.2
                                else "it barely moves with what you hold, so it diversifies."))

    lever = [s for s in names if classify(s, rows).leveraged]
    if lever:
        lines.append(f"Caveat: {', '.join(_t(s) for s in lever)} is a leveraged or inverse fund; "
                     f"its sector split is its holdings', not scaled by its leverage, while its "
                     f"betas above carry the leverage.")
    unclassified = [s for s in names if "Unclassified" in classify(s, rows).weights]
    if unclassified:
        lines.append(f"Missing: {', '.join(_t(s) for s in unclassified)} has no sector in the map "
                     f"— " + "; ".join(classify(s, rows).note or "no public classification"
                                       for s in unclassified) + ".")
    for note in notes:
        lines.append(f"Assumed: {note}.")
    origins = sorted(set(inputs.origins.values()))
    lines.append(
        f"Data: sectors from data/sector_map.json (Yahoo quoteSummary, frozen {map_date}); "
        f"factor returns from SPY, IWM and MTUM daily closes (Yahoo) and BTCUSDT daily candles "
        f"(Bitget, 16:00 UTC close, four hours before the US close); holdings from "
        + "; ".join(origins) + ".")
    lines.append("Method: one least-squares regression per holding of its daily returns on SPY, "
                 "IWM minus SPY, MTUM minus SPY and BTC together; the book's loading is the "
                 "weight-sum of its "
                 "holdings' (argus.lui.exposures).")

    sources = [
        Source(kind="computation", ref="argus.lui.exposures.exposures_answer",
               detail=f"4-factor OLS, {WINDOW_DAYS}d window, weights summed"),
        Source(kind="computation", ref="data/sector_map.json",
               detail=f"Yahoo quoteSummary sectors, frozen {map_date}"),
        Source(kind="evidence", ref="https://query1.finance.yahoo.com/v8/finance/chart",
               detail="SPY, IWM, MTUM and the holdings' daily closes"),
        Source(kind="venue", ref="bitget /api/v3/market/candles", detail="BTCUSDT 1D"),
    ]
    data: dict[str, Any] = {
        "book_before": before, "book_after": after,
        "sectors_before": sec_b, "sectors_after": sec_a or None,
        "loadings_before": load_b, "loadings_after": load_a or None,
        "coverage_before": cover_b, "coverage_after": cover_a if after is not None else None,
        "effective_positions": [eff_b, eff_a], "effective_sectors": [eff_sec_b, eff_sec_a],
        "fits": {s: None if f is None else {"loadings": dict(f.loadings),
                                             "t": dict(f.t_stats), "r2": f.r_squared, "n": f.n,
                                             "start": f.start.isoformat(),
                                             "end": f.end.isoformat()}
                 for s, f in fits.items()},
        "missing": dict(inputs.missing),
    }
    return lines, sources, data


# --- the question ----------------------------------------------------------------------------


EXPOSURES_QUESTION = re.compile(
    r"\b(?:sector|sectoral|industry|industries|factor|style)\s+(?:and\s+(?:sector|factor|"
    r"industry|style)\s+)?(?:exposures?|tilts?|loadings?|breakdown|split|mix|allocation|weights?|"
    r"weightings?|concentration|bets?)\b"
    r"|\bexposures?\s+(?:(?:by|per|across)\s+|to\s+(?:each\s+)?)(?:sector|industry|industries|"
    r"factor|style)s?\b"
    r"|\b(?:sectors?|industries)\s+(?:am\s+i|is\s+my\s+(?:book|portfolio)|does\s+my\s+(?:book|"
    r"portfolio))\b"
    r"|\bwhat\s+sectors?\b[^?.]{0,40}\b(?:book|portfolio|hold|holdings|own)\b"
    r"|\b(?:size|momentum|value|growth)\s+(?:factor\s+)?(?:tilt|loading|exposure)s?\b"
    r"|\bfactor\s+(?:model|betas?)\b[^?.]{0,40}\b(?:book|portfolio)\b"
    r"|\bcrypto\s+beta\s+(?:of|in|for)\s+(?:my|the)\s+(?:book|portfolio|stocks|holdings)\b"
    r"|(?:行业|板块|因子|风格|市值|动量)(?:的)?(?:敞口|暴露|分布|配置|占比|权重|载荷|倾斜)"
    r"|(?:敞口|暴露)(?:在)?(?:哪些)?(?:行业|板块|因子)",
    re.I,
)
"""A question that asks for the book's sector, industry or factor make-up. Kept narrow: "exposure"
alone ("my exposure to NVDA", "what's my crypto exposure") is a weight or a beta the book answer
already gives, and "beta" alone is `research._book_beta_line`'s."""


_SHARE_UNITS = re.compile(r"\d[\d,]*\s*(?:shares?\b|股)", re.I)
"""A trade stated in shares ("add 500 shares of NVDA", "加入200股TSLA") needs prices and a book
value to become a weight, which `research`'s IMPACT kind already does (`_amount_pairs`). Two such
questions in the held-out corpus (``lui_heldout_corpus_2026-09-25.jsonl``, expected "impact")
also ask about sector concentration or factor exposure; they stay with IMPACT."""


def asks_exposures(text: str) -> bool:
    return EXPOSURES_QUESTION.search(text) is not None and not _SHARE_UNITS.search(text)


DEFAULT_ADD = 0.20
"""The target weight of a name the question adds without a size, as `research.DEFAULT_SIZE`."""


def parse(text: str, book_text: str = "") -> tuple[dict[str, float], dict[str, float],
                                                  list[str]]:
    """(book, proposed, notes) from a question and the saved book.

    Weights stated before an add verb are the book; a weighted name after it is the trade ("add
    20% NVDA"), and a bare name after it is added at :data:`DEFAULT_ADD`, said so in a note. With
    no holdings in the question, the saved book is used and the note says that too."""
    from argus.lui import research

    notes: list[str] = []
    verb = research._ADD_VERB.search(text)
    cut = verb.start() if verb is not None else len(text)
    pairs = research._pairs(text)
    book: dict[str, float] = {}
    proposed: dict[str, float] = {}
    for position, symbol, weight in pairs:
        if position < cut:
            book[symbol] = book.get(symbol, 0.0) + weight
        else:
            proposed[symbol] = abs(weight)
    if verb is not None and not proposed:
        named, _ = research.research_symbols(text[verb.end():])
        for symbol in named[:1]:
            proposed[symbol] = DEFAULT_ADD
            notes.append(f"no size was stated for {_t(symbol)}, so it is added at "
                         f"{_pct(DEFAULT_ADD)} of the book after the trade, the rest scaled down "
                         f"pro rata")
    if book:
        total = sum(abs(w) for w in book.values())
        if abs(total - 1.0) > 0.02:
            notes.append(f"your holdings add up to {_pct(total)}, so they were scaled to 100%")
            book = {s: w / total for s, w in book.items()}
    elif book_text.strip():
        saved = research.parse_book(book_text)
        if saved:
            book = dict(saved)
            notes.append("used your saved book (" + ", ".join(f"{_pct(w)} {_t(s)}"
                                                               for s, w in book.items()) + ")")
    if not book:
        named, _ = research.research_symbols(text[:cut])
        named = tuple(s for s in named if s not in proposed)
        if named:
            book = {s: 1.0 / len(named) for s in named}
            notes.append("no weights were stated, so the names are read as equal weight")
    if proposed and verb is not None:
        for symbol, target in proposed.items():
            if symbol not in book and not any("no size" in n for n in notes):
                notes.append(f"\"{verb.group(0)} {_pct(target)} {_t(symbol)}\" is read as "
                             f"{_t(symbol)} at {_pct(target)} of the book after the trade, the "
                             f"rest scaled down pro rata")
    return book, proposed, notes


def answer(raw_text: str, book_text: str = "", *,
           inputs_for: Callable[[Sequence[str]], Inputs] | None = None) -> Answer | None:
    """THE call for the console: an Answer when the question asks for sector or factor exposures,
    else None so the caller carries on. A question with no book at all is answered with what to
    send, not refused."""
    if not asks_exposures(raw_text):
        return None
    from argus.lui.question import Intent, Question, Speed, Tense

    book, proposed, notes = parse(raw_text, book_text)
    symbols = tuple(dict.fromkeys([*book, *proposed]))
    question = Question(raw=raw_text, intent=Intent.RESEARCH, speed=Speed.SLOW,
                        tense=Tense.FUTURE, symbols=symbols,
                        matched="research:exposures:patterns")
    if not book and proposed:
        book, proposed = dict(proposed), {}
        book = {s: 1.0 / len(book) for s in book}
        notes = [n for n in notes if "pro rata" not in n]
        notes.append("no book was stated or saved, so the named contract is read on its own")
    if not book:
        return Answer(question=question, lines=[
            "Actionable: tell me what you hold to read its exposures — for example \"40% MSFT, "
            "30% META, 30% GOOGL: what are my sector and factor exposures if I add 20% NVDA?\" — "
            "or save it once in My book.",
        ], sources=[Source(kind="computation", ref="argus.lui.exposures",
                           detail="no book stated or saved")],
            data={"kind": "exposures", "book": {}})
    inputs = inputs_for(symbols) if inputs_for is not None else None
    lines, sources, data = exposures_answer(book, proposed or None, inputs=inputs, notes=notes)
    data["kind"] = "exposures"
    return Answer(question=question, lines=lines, sources=sources, data=data)


def main(argv: Sequence[str] = ()) -> int:
    args = list(argv) or sys.argv[1:]
    if args[:1] == ["--freeze"]:
        counts = freeze_sector_map()
        print(json.dumps(counts, indent=1))
        return 0
    text = " ".join(args) or "40% MSFT, 30% META, 30% GOOGL: sector and factor exposures?"
    result = answer(text)
    if result is None:
        print("not an exposures question")
        return 1
    print("\n".join(result.lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "EXPOSURES_QUESTION",
    "FACTORS",
    "Classification",
    "Fit",
    "Inputs",
    "after_trade",
    "answer",
    "asks_exposures",
    "book_loadings",
    "classify",
    "effective_number",
    "exposures_answer",
    "factor_returns",
    "fit_holding",
    "freeze_sector_map",
    "ols",
    "parse",
    "row_from_summary",
    "sector_weights",
]
