"""Scenario and research-brief readers for the questions round 44's judge found unanswered.

Round 44's live audit (Activity/audits/round44_judge.md) put nine questions to the console that it
answered with a neighbouring fact, or not at all. Each reader below is strict: it takes a question
only when the shape is unmistakable, and :func:`lines` returns None for everything else so the
console's other readers keep their questions. Every answer opens "Bottom line: ", names its sources
in a last "Data: ... Not advice." line, and says in words what could not be read or did not exist;
a source that fails gives one honest "could not be read just now" line, never an invented figure.

Everything below was read live on 2026-10-06, and each figure quoted was recomputed by a second
route (raw Yahoo chart JSON, FRED CSV and SEC company facts fetched directly, numpy and hand
arithmetic) before it was written here.

1. **Macro scenarios on a stated book (MAJOR 2, 3)** - "How would a stagflation scenario hit a book
   of 50% SPY, 30% TLT, 20% gold?" and "What would a recession do to a book of 40% QQQ, 40% BTC, 20%
   T-bills?" were answered with a QQQ beta shock and a 3-month bill rate. A scenario here is a
   past window measured from adjusted daily closes (`market/equity_history.daily`, Yahoo, since
   listing: SPY 1993-01-29, QQQ 1999-03-10, gold futures GC=F 2000-08-30, TLT 2002-07-30, GLD
   2004-11-18, SHV 2007-01-11, BTC-USD 2014-09-17). Stagflation is 3 Jan to 12 Oct 2022 (the S&P
   500's record close and its bear-market low, both confirmed on SPY's adjusted closes); recession
   is 19 Feb to 23 Mar 2020 and 12 Sep 2008 to 9 Mar 2009 (the low confirmed on SPY). 1973-74 is
   said to be unavailable. An asset with no price at a window's start (Bitcoin in 2008) makes the
   book's P&L "cannot be computed" for that window, with the rest of the book's contribution
   given, and the 2022 window is added for reference. Bitcoin comes from Yahoo's BTC-USD, not
   `rule_test.daily_closes`, because that reader keeps five years and 2020 is outside it. T-bills
   are SHV (listed 2007): SGOV began trading in May 2020 and ^IRX is a yield, not a price. Checked:
   2022 window SPY -24.5%, TLT -29.3%, gold -6.8%, so 50/30/20 is -22.4%; 2020 QQQ -27.9%, BTC
   -33.4%, SHV +0.6%, so 40/40/20 is -24.4%.
2. **Risk parity (MAJOR 5)** - two plain sentences on what it is, then inverse-volatility weights
   from 60- and 252-day realised volatility (daily log returns, sample standard deviation times
   the square root of 252, on the days every asset traded) and each asset's share of risk from the
   252-day covariance matrix, with the long-only equal-risk-contribution weights solved on it
   (:func:`equal_risk_weights`). The default assets are SPY, TLT and GLD. Checked on 2026-10-05
   closes: volatilities SPY 13.0% / 11.1%, TLT 9.4% / 10.0%, GLD 29.8% / 24.9% (252 / 60 days),
   weights 35.4%, 49.1%, 15.5%; the risk shares come out 35.9%, 31.3%, 32.8%. A bare "what is risk
   parity?" is left to the concept reader.
3. **Gold against real yields (MAJOR 4)** - gold futures against FRED DFII10 (10-year TIPS yield,
   through `macro._fred`), regression of ln(gold) on the yield over the 5,800 days FRED is asked
   for (3,963 common days, 18 Nov 2010 to 2 Oct 2026; fewer than ten years answers "not read").
   Result: ln(gold) = 7.325 + 0.192 x yield, R-squared 0.25, so the fit puts gold at $2,660
   against $4,162 (+1.4 standard deviations of the fit's error). The slope is positive over the
   whole span because the link broke after 2022: fitted to 2010-2021 alone it is -19% per point of
   real yield (R-squared 0.66) and correlates -0.81 on levels, against +0.52 since 2022 and -0.26
   over the last 252 days. Those are measured and printed; none is assumed.
4. **Kelly with a stated payoff (CRITICAL 1)** - :func:`kelly_payoff` reads "average win 1.5R,
   average loss 1R", "my average winner is 1.5 times my average loser", "payoff 1.5:1",
   "win/loss ratio of 1.5" and the older forms; `server._kelly_lines` calls it, so 55% at 1.5R
   gives 0.55 - 0.45 / 1.5 = 25.0% full, 12.5% half, 6.25% quarter, instead of 10.0% on an assumed
   1:1.
5. **Earnings quality (MAJOR 11)** - four-quarter operating cash flow against net income, the
   accruals ratio (net income - operating cash flow) / average total assets, and free-cash-flow
   conversion, from SEC XBRL through `shareholder_metrics.read_company` and the balance sheet's
   ``Assets`` instants. Microsoft, four quarters to 30 Jun 2026: net income $133.749bn, operating
   cash flow $182.935bn (1.37x), assets $619.003bn then $758.376bn, accruals -7.14%; a year
   earlier 1.34x. All four figures are the 10-K's own rows.
6. **Insider selling against the buyback (MAJOR 10)** - Form 4 open-market sales and purchases over
   90 days (`insider_flow`'s grouping) against the four-quarter repurchases in
   `shareholder_metrics`, with the windows' difference said and the buyback scaled to 90 days.
   Meta: $105m sold (21 decisions, 19 under 10b5-1 plans) against $3.33bn repurchased; Apple:
   $92m against $82.23bn.
7. **Analyst revisions (MAJOR 7)** - Yahoo's quoteSummary ``earningsTrend`` answers through
   `market.estimates.EstimatesSource.summary` (the call `fundamentals.yahoo_summary` makes, with
   its fixed module list swapped for this one): ``epsTrend`` carries current, 7, 30, 60 and 90
   days ago, ``epsRevisions`` the up and down counts. Tesla: current-quarter EPS $0.55 to $0.44,
   this year $2.11 to $1.74, next year $2.56 to $2.14 over 90 days; 1 up and 4 down revisions in
   30 days for the year. If it does not answer after two tries the reader says revisions are not
   read and that reported EPS is not a revision. A move through zero is given as two figures,
   never as a percentage of a negative number.
8. **Guidance and the reaction (MAJOR 1)** - the latest 8-K item 2.02 exhibit 99.1 through
   `market.earnings_release` (outlook sentences, guided revenue and band) and the move from the
   close before the release to the first close after it through `earnings_moves.reactions`.
   NVIDIA, release of 26 Aug 2026: revenue guided to $108.0bn plus or minus 2% (+12.3% on $96.2bn
   reported), stock +8.7% on 27 Aug against the S&P 500 +0.7%; the quarter before it guided
   $91.0bn and reported $96.2bn. Apple, 30 Jul 2026: no outlook section in the release, stock
   -7.4% on 31 Jul, EPS $2.02 against $1.89 (Yahoo ``earningsHistory``). Microsoft's release
   says guidance is given on the call, which is quoted and not read.
9. **Research brief (MAJOR 8)** - "Bull case:", "Bear case:" and "What to watch:" built from
   measured points, each with its figure and source: price against its 200-day average, three- and
   twelve-month returns, a 20% fall from the 12-month high, 30-day volatility, revenue growth,
   net-margin change, free-cash-flow sign, trailing P/E against 25, buyback size, share-count
   growth, Form 4 buying or selling and 90-day estimate revisions; the watch list is the next
   earnings date (`watchlist.earnings_date`) with the typical move on results, an ex-dividend date,
   recent 8-K items, and for a coin the DeFiLlama unlock calendar. The follow-up "What would change
   your mind on the bear case?" recomputes the brief named in ``prior`` and gives the reading
   that would undo each point. A side with no point crossing its bar says so.
"""

from __future__ import annotations

import bisect
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any, Final

from argus.lui.research import filing_figures

# --------------------------------------------------------------------------- shared helpers


def _pct(x: float, places: int = 1) -> str:
    return f"{x:+.{places}%}"


def _plain(x: float, places: int = 1) -> str:
    return f"{x:.{places}%}"


def _usd(x: float) -> str:
    return filing_figures.money(x)


def _px(x: float) -> str:
    return f"${x:,.2f}" if abs(x) < 1000 else f"${x:,.0f}"


def _d(day: date) -> str:
    return f"{day:%d %b %Y}"


def _close(out: list[str], data: str) -> list[str]:
    """The answer's first line leads with the verdict, and the last names the sources."""
    if not out[0].startswith("Bottom line: "):
        out[0] = "Bottom line: " + out[0]
    out.append(f"Data: {data} Not advice.")
    return out


def _failed(what: str, why: str, data: str) -> list[str]:
    return [f"Bottom line: {what} could not be read just now ({why}); no figure is given and none "
            "is estimated.", f"Data: {data}. Not advice."]


def _series(ticker: str) -> dict[date, float]:
    """Yahoo Finance's whole daily history of ``ticker`` since listing, as date -> adjusted close
    (dividends and splits included), through `market/equity_history.daily`."""
    from argus.market.equity_history import daily

    return {row.day: float(row.close) for row in daily(ticker)}


def _tickers(text: str, limit: int = 3) -> list[str]:
    """US-listed companies a question names, in the order the console's own readers find them."""
    from argus.lui.research import research_symbols, shareholder_metrics
    from argus.lui.research.parse import is_us_equity

    found = [s.removesuffix("USDT") for s in research_symbols(text)[0] if is_us_equity(s)]
    try:
        named = shareholder_metrics.resolve(text)
    except Exception:
        named = []
    out: list[str] = []
    for ticker in [*found, *named]:
        if ticker not in out:
            out.append(ticker)
    return out[:limit]


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs)


def _sd(xs: Sequence[float]) -> float:
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _cov(a: Sequence[float], b: Sequence[float]) -> float:
    ma, mb = _mean(a), _mean(b)
    return sum((x - ma) * (y - mb) for x, y in zip(a, b, strict=True)) / (len(a) - 1)


def _corr(a: Sequence[float], b: Sequence[float]) -> float:
    return _cov(a, b) / (_sd(a) * _sd(b))


def _log_returns(closes: Sequence[float]) -> list[float]:
    return [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]


# --------------------------------------------------------------------------- the assets a book nam

_ALIASES: Final[dict[str, tuple[str | None, str]]] = {
    "spy": ("SPY", "S&P 500 (SPY)"), "s&p": ("SPY", "S&P 500 (SPY)"),
    "s&p 500": ("SPY", "S&P 500 (SPY)"), "sp500": ("SPY", "S&P 500 (SPY)"),
    "stock": ("SPY", "US stocks (SPY)"), "stocks": ("SPY", "US stocks (SPY)"),
    "equities": ("SPY", "US stocks (SPY)"), "equity": ("SPY", "US stocks (SPY)"),
    "qqq": ("QQQ", "Nasdaq-100 (QQQ)"), "nasdaq": ("QQQ", "Nasdaq-100 (QQQ)"),
    "nasdaq-100": ("QQQ", "Nasdaq-100 (QQQ)"), "nasdaq 100": ("QQQ", "Nasdaq-100 (QQQ)"),
    "tlt": ("TLT", "20+ year Treasuries (TLT)"), "bond": ("TLT", "long bonds (TLT)"),
    "bonds": ("TLT", "long bonds (TLT)"), "treasuries": ("TLT", "long bonds (TLT)"),
    "treasury": ("TLT", "long bonds (TLT)"), "long bonds": ("TLT", "long bonds (TLT)"),
    "gold": ("GC=F", "gold futures (GC=F)"), "xau": ("GC=F", "gold futures (GC=F)"),
    "gc=f": ("GC=F", "gold futures (GC=F)"), "gld": ("GLD", "gold (GLD)"),
    "btc": ("BTC-USD", "Bitcoin (BTC-USD)"), "bitcoin": ("BTC-USD", "Bitcoin (BTC-USD)"),
    "eth": ("ETH-USD", "Ether (ETH-USD)"), "ethereum": ("ETH-USD", "Ether (ETH-USD)"),
    "ether": ("ETH-USD", "Ether (ETH-USD)"),
    "t-bills": ("SHV", "T-bills (SHV)"), "t-bill": ("SHV", "T-bills (SHV)"),
    "tbills": ("SHV", "T-bills (SHV)"), "t bills": ("SHV", "T-bills (SHV)"),
    "treasury bills": ("SHV", "T-bills (SHV)"), "bills": ("SHV", "T-bills (SHV)"),
    "sgov": ("SHV", "T-bills (SHV)"), "bil": ("SHV", "T-bills (SHV)"),
    "shv": ("SHV", "T-bills (SHV)"),
    "cash": (None, "cash"),
}
"""Name -> (Yahoo ticker, label). "Gold" is gold futures (history from 2000, so it reaches every
window below); T-bills are SHV, the iShares 0-1 year Treasury ETF (listed 2007), because SGOV only
began trading in May 2020 and ^IRX is a yield, not a price."""

_STOP_NAMES: Final = frozenset({"a", "an", "the", "my", "of", "in", "into", "to", "and", "or",
                                "book", "portfolio", "it", "is", "at", "on", "for", "with", "more",
                                "less"})
_WEIGHT: Final = re.compile(
    r"(?P<w>\d{1,3}(?:\.\d+)?)\s*%\s*(?:(?:in|of|into)\s+)?"
    r"(?P<n>[A-Za-z][A-Za-z&.=-]*(?:\s+(?:500|100|bills?|stocks?|bonds?|treasur\w+))?)")


@dataclass(frozen=True)
class _Holding:
    label: str
    ticker: str | None
    weight: float


@dataclass(frozen=True)
class _Book:
    holdings: tuple[_Holding, ...]
    unknown: tuple[str, ...]

    @property
    def total(self) -> float:
        return sum(h.weight for h in self.holdings)


def _asset(name: str) -> tuple[str | None, str] | None:
    key = re.sub(r"\s+", " ", name.strip().lower().rstrip(".,"))
    if key in _ALIASES:
        return _ALIASES[key]
    head = key.split(" ")[0]
    if head in _ALIASES and head not in ("treasury", "t"):
        return _ALIASES[head]
    if re.fullmatch(r"[A-Z]{1,5}", name.strip()):
        return name.strip(), name.strip()
    return None


def _book(text: str) -> _Book | None:
    """The weights a question states ("50% SPY, 30% TLT, 20% gold"), or None with fewer than two."""
    holdings: list[_Holding] = []
    unknown: list[str] = []
    for m in _WEIGHT.finditer(text):
        name = m.group("n")
        if name.lower() in _STOP_NAMES:
            continue
        found = _asset(name)
        if found is None:
            unknown.append(name)
            continue
        holdings.append(_Holding(found[1], found[0], float(m.group("w")) / 100))
    if len(holdings) + len(unknown) < 2:
        return None
    return _Book(tuple(holdings), tuple(unknown))


# --------------------------------------------------------------------------- 1. macro scenarios


@dataclass(frozen=True)
class _Window:
    label: str
    start: date
    end: date
    why: str


_RATE_SHOCK: Final = _Window(
    "the 2022 rate shock", date(2022, 1, 3), date(2022, 10, 12),
    "from the S&P 500's record close to its bear-market closing low, with inflation at "
    "multi-decade highs and the Fed raising rates throughout")
STAGFLATION_WINDOWS: Final = (_RATE_SHOCK,)
RECESSION_WINDOWS: Final = (
    _Window("the 2020 Covid crash", date(2020, 2, 19), date(2020, 3, 23),
            "from the S&P 500's record close to its closing low, 23 trading days"),
    _Window("the 2008-09 financial crisis", date(2008, 9, 12), date(2009, 3, 9),
            "from the last close before Lehman Brothers failed to the S&P 500's closing low"),
)
_SCENARIO: Final = re.compile(r"\b(stagflation(?:ary)?|recession(?:ary)?)\b", re.I)
MAX_GAP_DAYS: Final = 7


def _move(series: dict[date, float], window: _Window) -> float | None:
    """The asset's total return from the last close on or before the window's start to the last
    close on or before its end; None when it had no price at the start (it did not exist yet)."""
    days = sorted(series)
    if not days or days[0] > window.start + timedelta(days=MAX_GAP_DAYS):
        return None
    i, j = bisect.bisect_right(days, window.start) - 1, bisect.bisect_right(days, window.end) - 1
    if i < 0 or j <= i or (window.end - days[j]).days > MAX_GAP_DAYS:
        return None
    return series[days[j]] / series[days[i]] - 1


def _window_line(window: _Window, book: _Book, moves: dict[str, float | None]) -> tuple[str, str]:
    """(the window's detail line, the same result as a clause for the bottom line)."""
    parts: list[str] = []
    each: list[str] = []
    pnl, covered, missing = 0.0, 0.0, []
    for h in book.holdings:
        move = 0.0 if h.ticker is None else moves.get(h.ticker)
        if move is None:
            missing.append(h.label)
            parts.append(f"{h.label} had no price then")
            continue
        pnl += h.weight * move
        covered += h.weight
        parts.append(f"{h.label} {_pct(move)} x {h.weight:.0%} = {_pct(h.weight * move)}")
        each.append(f"{h.label} {_pct(move)}")
    span = f"{window.start:%d %b %Y} to {window.end:%d %b %Y}"
    if missing:
        head = (f"book P&L not computable ({', '.join(missing)} had no price; the holdings that "
                f"did, {covered:.0%} of the book, contributed {_pct(pnl)} of it)")
        clause = (f"cannot be computed for {window.label} ({span}) because "
                  f"{', '.join(missing)} had no price then")
    else:
        head = f"book {_pct(pnl)}"
        clause = f"{_pct(pnl)} over {window.label} ({span}): " + ", ".join(each)
    line = (f"{window.label[0].upper()}{window.label[1:]}, {span} ({window.why}): {head}. "
            "By holding: " + "; ".join(parts) + ".")
    return line, clause


def _scenario(text: str, prior: Sequence[str]) -> list[str] | None:
    said = _SCENARIO.search(text)
    book = _book(text) if said else None
    if said is None or book is None:
        return None
    stag = said.group(1).lower().startswith("stag")
    kind = "stagflation" if stag else "recession"
    data = ("Yahoo Finance adjusted daily closes (dividends included) for each asset; the windows "
            "are past analogues, each buy-and-hold at the weights stated.")
    if book.unknown:
        return _failed(f"the {kind} scenario on this book",
                       "I did not recognise " + ", ".join(f"\"{u}\"" for u in book.unknown)
                       + "; name each holding as a ticker (SPY, TLT), gold, bitcoin or T-bills",
                       data.split(";")[0])
    if book.total > 1.005:
        return [f"Bottom line: the weights add up to {book.total:.0%}, over 100%, so the book "
                f"cannot be run through the {kind} windows; restate them to total 100% or less.",
                f"Data: {data} Not advice."]
    windows = STAGFLATION_WINDOWS if stag else RECESSION_WINDOWS
    tickers = sorted({h.ticker for h in book.holdings if h.ticker})
    closes: dict[str, dict[date, float]] = {}
    for ticker in tickers:
        try:
            closes[ticker] = _series(ticker)
        except Exception:
            return _failed(f"the {kind} scenario on this book",
                           f"Yahoo Finance's history for {ticker} did not arrive",
                           data.split(";")[0])
    extra: tuple[_Window, ...] = ()
    if not stag and any(
            h.ticker and _move(closes[h.ticker], windows[-1]) is None for h in book.holdings):
        extra = (_RATE_SHOCK,)
    out: list[str] = []
    shorts: list[str] = []
    side: list[str] = []
    for window in (*windows, *extra):
        moves = {t: _move(closes[t], window) for t in tickers}
        line, clause = _window_line(window, book, moves)
        if window in extra:
            line = ("Because an asset here did not trade in 2008-09, the other risk-off window it "
                    "has is " + line[0].lower() + line[1:] + " That year was a bear market for "
                    "stocks and bonds together, not a recession.")
            side.append(clause)
        else:
            shorts.append(clause)
        out.append(line)
    cash = 1 - book.total
    if stag:
        lead = ("a stagflation-type shock would have cost this book " + shorts[0]
                + " - the nearest modern analogue, with every price available.")
    else:
        lead = ("a recession would have cost this book " + "; ".join(shorts)
                + (". For reference, the other risk-off window every asset has, which was a "
                   "bear market and not a recession: " + "; ".join(side) if side else "") + ".")
        lead = lead.replace("..", ".")
    if stag:
        out.append("1973-74, the textbook stagflation, is not available: the earliest daily "
                   "history behind these assets starts in 1993 (SPY), 2000 (gold futures) and "
                   "2002 (TLT), so 2022 is the only stagflation-type window with every price.")
    else:
        late = sorted({h.label for h in book.holdings if h.ticker and _move(
            closes[h.ticker], RECESSION_WINDOWS[-1]) is None})
        if late:
            out.append(f"{', '.join(late)} did not exist as a priced asset in 2008-09, so that "
                       "window cannot be run for it; 2020 and 2022 are the windows it has.")
    if cash > 0.005:
        out.append(f"The weights add up to {book.total:.0%}; the other {cash:.0%} is treated as "
                   "cash that neither gains nor loses.")
    out.append("A window is one history, not a model: the next stagflation or recession can hit "
               "each asset differently, and the book here is held unrebalanced from the first "
               "day to the last.")
    out.insert(0, lead[0].upper() + lead[1:])
    return _close(out, data)


# --------------------------------------------------------------------------- 2. risk parity

_PARITY: Final = re.compile(r"\brisk[\s-]+parity\b|\binverse[\s-]+vol(?:atility)?\b", re.I)
_BUILD: Final = re.compile(
    r"\b(?:build|construct|compute|calculate|work\s+out|weights?|weighting|allocat\w+|"
    r"\d{2,3}\s*-?\s*day|realised|realized|using\s+vol\w*|for\s+(?:stocks|equities|spy|qqq|"
    r"bonds?|gold|btc|bitcoin|tlt|gld))\b", re.I)
_PARITY_NAME: Final = re.compile(
    r"\b(s&p\s*500|s&p|sp500|spy|qqq|nasdaq|tlt|treasur(?:y|ies)|bonds?|gold|gld|xau|btc|bitcoin|"
    r"eth|ethereum|ether|stocks?|equities)\b|(?-i:\b[A-Z]{2,5}\b)", re.I)
_NOT_TICKERS: Final = frozenset({"ETF", "USD", "US", "AI", "AND", "THE", "FOR", "VOL", "ERC", "CPI",
                                 "GDP", "FED", "RISK", "I", "SD", "ETFS", "OK", "TIPS"})
PARITY_DEFAULT: Final = (("SPY", "stocks (SPY)"), ("TLT", "bonds (TLT)"), ("GLD", "gold (GLD)"))
TRADING_DAYS: Final = 252
MIN_PARITY_RETURNS: Final = 130


def _parity_assets(text: str) -> tuple[list[tuple[str, str]], bool]:
    """The assets named, as (Yahoo ticker, label), and whether the default three were used."""
    out: list[tuple[str, str]] = []
    for m in _PARITY_NAME.finditer(text):
        word = m.group(0)
        found = _asset(word) if m.group(1) else (
            (word, word) if word not in _NOT_TICKERS else None)
        if found is None or found[0] is None:
            continue
        ticker, label = found[0], found[1]
        if ticker == "GC=F":
            ticker, label = "GLD", "gold (GLD)"
        if ticker == "SHV":
            continue
        if all(ticker != t for t, _ in out):
            out.append((ticker, label))
    if len(out) < 2:
        return list(PARITY_DEFAULT), True
    return out[:6], False


def _vol(rets: Sequence[float], window: int) -> float:
    return _sd(list(rets[-window:])) * math.sqrt(TRADING_DAYS)


def _risk_shares(weights: Sequence[float], cov: Sequence[Sequence[float]]) -> list[float]:
    """Each asset's share of the portfolio's variance: w_i (Cov w)_i over w'Cov w."""
    marginal = [sum(row[j] * weights[j] for j in range(len(weights))) for row in cov]
    parts = [w * m for w, m in zip(weights, marginal, strict=True)]
    total = sum(parts)
    return [p / total for p in parts]


def equal_risk_weights(cov: Sequence[Sequence[float]]) -> list[float] | None:
    """The long-only weights at which every asset carries the same share of risk, by a
    multiplicative fixed-point iteration (w_i <- w_i * sqrt(target / contribution_i)); None when it
    has not converged to within 0.0001 of an equal share after 5,000 steps."""
    n = len(cov)
    w = [1.0 / n] * n
    for _ in range(5000):
        shares = _risk_shares(w, cov)
        if max(abs(s - 1.0 / n) for s in shares) < 1e-4:
            return w
        w = [wi * math.sqrt((1.0 / n) / s) for wi, s in zip(w, shares, strict=True)]
        total = sum(w)
        w = [wi / total for wi in w]
    return None


def _parity(text: str, prior: Sequence[str]) -> list[str] | None:
    if not _PARITY.search(text) or not _BUILD.search(text):
        return None
    assets, default = _parity_assets(text)
    data = ("Yahoo Finance adjusted daily closes; volatility is the standard deviation of daily "
            "log returns times the square root of 252, on the days every asset traded.")
    series: dict[str, dict[date, float]] = {}
    unread: list[str] = []
    for ticker, _label in assets:
        try:
            series[ticker] = _series(ticker)
        except Exception:
            unread.append(ticker)
    assets = [(t, label) for t, label in assets if t in series]
    if len(assets) < 2:
        return _failed("the risk-parity weights", "Yahoo Finance's history did not arrive for "
                       + ", ".join(unread or [t for t, _ in PARITY_DEFAULT]), data)
    common = sorted(set.intersection(*(set(series[t]) for t, _ in assets)))
    closes = {t: [series[t][d] for d in common] for t, _ in assets}
    rets = {t: _log_returns(closes[t]) for t, _ in assets}
    if len(rets[assets[0][0]]) < MIN_PARITY_RETURNS:
        return _failed("the risk-parity weights", "the assets share too little price history",
                       data)
    vols = {w: [_vol(rets[t], w) for t, _ in assets]
            for w in (60, TRADING_DAYS) if len(rets[assets[0][0]]) >= w}
    weights = {w: [(1 / v) / sum(1 / x for x in vs) for v in vs] for w, vs in vols.items()}
    long = TRADING_DAYS if TRADING_DAYS in vols else max(vols)
    cov = [[_cov(rets[a][-long:], rets[b][-long:]) * TRADING_DAYS for b, _ in assets]
           for a, _ in assets]
    shares = _risk_shares(weights[long], cov)
    erc = equal_risk_weights(cov)
    names = [label for _, label in assets]
    named = ", ".join(f"{n} {w:.1%}" for n, w in zip(names, weights[long], strict=True))
    out = [f"Bottom line: inverse-volatility (risk-parity) weights on {long}-day volatility are "
           f"{named}, as of {common[-1]:%d %b %Y}."
           + (" No assets were named, so stocks, bonds and gold are used." if default else ""),
           "Risk parity sizes each asset by how much risk it brings rather than by how many "
           "dollars it holds, so that every asset contributes the same share of the portfolio's "
           "swings; a calm asset such as bonds therefore gets a bigger weight than a volatile one "
           "such as stocks. The simple version used here weights each asset by 1 divided by its "
           "volatility, which equalises risk exactly only when the assets are uncorrelated, so "
           "the covariance check below shows how close it gets."]
    for window in sorted(vols):
        out.append(f"{window}-day realised volatility (annualised) and the weight it gives: "
                   + "; ".join(f"{n} {v:.1%} -> {w:.1%}" for n, v, w in
                               zip(names, vols[window], weights[window], strict=True)) + ".")
    out.append(f"Share of portfolio risk at the {long}-day weights, from the {long}-day covariance "
               "matrix (each asset's weight times its covariance with the whole, over the total): "
               + "; ".join(f"{n} {s:.1%}" for n, s in zip(names, shares, strict=True))
               + f" (equal would be {1 / len(names):.1%} each).")
    if erc is not None:
        out.append("Weights that make those shares exactly equal, solved on the same covariance "
                   "matrix: " + ", ".join(f"{n} {w:.1%}" for n, w in zip(names, erc, strict=True))
                   + ".")
    out.append("Risk-parity funds typically lever the bond leg until its dollar risk matches the "
               "stocks'; this does not. These weights sum to 100% with no borrowing, so the whole "
               "portfolio is calmer than an all-stock one but its return is lower for the same "
               "reason. Volatilities move: the "
               f"{min(vols)}-day and {max(vols)}-day figures above are the same asset on two "
               "different recent histories.")
    return _close(out, data)


# --------------------------------------------------------------------------- 3. gold vs real yields

_GOLD: Final = re.compile(r"\bgold\b|\bxau\b", re.I)
_REAL: Final = re.compile(r"\breal\s+(?:yields?|rates?|interest\s+rates?)\b|\btips\s+yields?\b|"
                          r"\bdfii10\b|\binflation[\s-]protected\b", re.I)
_REL: Final = re.compile(r"\bcheap\b|\bexpensive\b|\brich\b|\bvalued?\b|\bversus\b|\bvs\.?\b|"
                         r"\bagainst\b|\brelative\b|\brelationship\b|\bcorrelat\w+|\bfollow\w*|"
                         r"\btrack\w*|\bexplain\w*|\bfair\b|\bovervalued\b|\bundervalued\b|"
                         r"\bdriven\b|\bdecoupl\w+|\bbroke\w*", re.I)
REAL_YIELD_DAYS: Final = 5800
MIN_SPAN_DAYS: Final = 3650
MIN_POINTS: Final = 2000
RECENT_POINTS: Final = 252
BREAK_YEAR: Final = 2022


def _fred_series(series: str, days: int) -> list[tuple[str, float]]:
    from argus.lui.research.macro import _fred

    return _fred(series, days)


def _ols(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    """(intercept, slope) of y on x."""
    slope = _cov(x, y) / _cov(x, x)
    return _mean(y) - slope * _mean(x), slope


def _gold_real_yield(text: str, prior: Sequence[str]) -> list[str] | None:
    if not (_GOLD.search(text) and _REAL.search(text) and _REL.search(text)):
        return None
    data = ("Yahoo Finance gold futures (GC=F) daily closes; FRED DFII10 (10-year Treasury "
            "inflation-indexed constant-maturity yield, daily).")
    try:
        gold = _series("GC=F")
        real = dict(_fred_series("DFII10", REAL_YIELD_DAYS))
    except Exception:
        return _failed("gold against the 10-year real yield", "Yahoo Finance or FRED did not "
                       "answer", data)
    gold_by_iso = {d.isoformat(): p for d, p in gold.items()}
    days = sorted(set(gold_by_iso) & set(real))
    if len(days) < MIN_POINTS or (date.fromisoformat(days[-1]) - date.fromisoformat(
            days[0])).days < MIN_SPAN_DAYS:
        return _failed("gold against the 10-year real yield", "fewer than ten years of both "
                       "series came back", data)
    x = [real[d] for d in days]
    lg = [math.log(gold_by_iso[d]) for d in days]
    price = [gold_by_iso[d] for d in days]
    a, b = _ols(x, lg)
    resid = [lg[i] - (a + b * x[i]) for i in range(len(days))]
    s = math.sqrt(sum(r * r for r in resid) / (len(resid) - 2))
    r2 = _corr(x, lg) ** 2
    fit_now = math.exp(a + b * x[-1])
    z = resid[-1] / s
    gap = price[-1] - fit_now
    first, last = date.fromisoformat(days[0]), date.fromisoformat(days[-1])
    full_says = ("about in line" if abs(z) < 0.5 else "expensive" if z > 0 else "cheap")
    out = [""]
    older = ""
    cut = next((i for i, d in enumerate(days) if d >= f"{BREAK_YEAR}-01-01"), None)
    old_says = None
    if cut is not None and cut >= 1000:
        a0, b0 = _ols(x[:cut], lg[:cut])
        old = math.exp(a0 + b0 * x[-1])
        s0 = math.sqrt(sum((lg[i] - (a0 + b0 * x[i])) ** 2 for i in range(cut)) / (cut - 2))
        z0 = (lg[-1] - (a0 + b0 * x[-1])) / s0
        old_says = ("about in line" if abs(z0) < 0.5 else "expensive" if z0 > 0 else "cheap")
        older = (f"Fitted only on {_d(first)} to {_d(date.fromisoformat(days[cut - 1]))} (before "
                 f"the {BREAK_YEAR} break; R-squared {_corr(x[:cut], lg[:cut]) ** 2:.2f}, "
                 f"gold {_pct(math.exp(b0) - 1, 0)} per point of real yield, real yields from "
                 f"{min(x[:cut]):.2f}% to "
                 f"{max(x[:cut]):.2f}%), the line gives {_px(old)} at today's {x[-1]:.2f}% real "
                 f"yield and gold is {_pct(price[-1] / old - 1, 0)} against it"
                 + (" - but today's real yield is outside the range that fit ever saw, so read "
                    "that as the size of the break, not as a fair value."
                    if x[-1] > max(x[:cut]) or x[-1] < min(x[:cut]) else "."))
    if old_says is None:
        verdict = f"{full_says} on the full-span fit"
    elif old_says == full_says:
        verdict = f"{full_says} on both the full-span and the pre-{BREAK_YEAR} fit"
    else:
        verdict = (f"{full_says} on the full-span fit but {old_says} on the pre-{BREAK_YEAR} "
                   "one")
    corr_levels_year = _corr(x[-RECENT_POINTS:], lg[-RECENT_POINTS:])
    corr_levels = _corr(x, lg)
    quality = []
    if r2 < 0.5:
        quality.append("weak")
    if abs(corr_levels - corr_levels_year) > 0.3 or corr_levels * corr_levels_year < 0:
        quality.append("unstable")
    link = (f"the link is {' and '.join(quality)}" if quality else "the link has been steady")
    out[0] = (f"Bottom line: gold at {_px(price[-1])} looks {verdict} against the 10-year real "
              f"yield ({x[-1]:.2f}%, FRED DFII10, {_d(last)}): the {first.year}-{last.year} fit "
              f"puts it at {_px(fit_now)}, so gold is {_px(abs(gap))} "
              f"{'above' if gap > 0 else 'below'} that ({z:+.1f} standard deviations of the fit's "
              f"own error) - {link}: R-squared {r2:.2f}, and the correlation of gold's level with "
              f"the real yield is {corr_levels:+.2f} over the full span against "
              f"{corr_levels_year:+.2f} over the last {RECENT_POINTS} days.")
    out.append(f"The full-span fit: ln(gold) = {a:.3f} {'-' if b < 0 else '+'} {abs(b):.3f} x real "
               f"yield over {len(days):,} common days ({_d(first)} to {_d(last)}), R-squared "
               f"{r2:.2f}; on it one more point of real yield goes with gold "
               f"{_pct(abs(math.exp(b) - 1), 0).lstrip('+')} {'lower' if b < 0 else 'higher'}. "
               f"Today's gold price is what that fit expects at a real yield of "
               f"{(lg[-1] - a) / b:.2f}%, against the {x[-1]:.2f}% actual.")
    if older:
        out.append(older)
    if cut is not None and len(days) - cut > 250:
        out.append(f"Since {BREAK_YEAR} alone ({len(days) - cut:,} days) the correlation of the "
                   f"levels is {_corr(x[cut:], lg[cut:]):+.2f}; before it, "
                   f"{_corr(x[:cut], lg[:cut]):+.2f}.")
    if len(days) > RECENT_POINTS + 5:
        dx = [x[i] - x[i - 1] for i in range(1, len(x))]
        dg = [lg[i] - lg[i - 1] for i in range(1, len(lg))]
        out.append(
            "Correlation of daily changes (gold's log change against the yield's point change): "
            f"{_corr(dx, dg):+.2f} over the full span, "
            f"{_corr(dx[-RECENT_POINTS:], dg[-RECENT_POINTS:]):+.2f} over the last "
            f"{RECENT_POINTS} days (from {_d(date.fromisoformat(days[-RECENT_POINTS]))}).")
    out.append("A fit on levels says where gold would sit if a past relationship held, not that it "
               "must return there: the textbook link is gold falling as real yields rise, and "
               "the lines above show how much of it has survived since the break.")
    return _close(out, data)


# --------------------------------------------------------------------------- 4. Kelly payoff

_N: Final = r"(?P<{name}>\d+(?:\.\d+)?)(?!\.?\d)"


def _n(name: str) -> str:
    return _N.format(name=name)


_AVG_WIN: Final = (r"\b(?:average|avg\.?|mean|typical)\s+(?:win(?:ner|ners|s)?|gain|profit)\b"
                   r"\s*(?:of|is|are|=|:)?\s*\$?")
_AVG_LOSS: Final = (r"\b(?:average|avg\.?|mean|typical)\s+(?:loss(?:es)?|loser|losers)\b"
                    r"\s*(?:of|is|are|=|:)?\s*\$?")
_TIMES: Final = re.compile(
    r"\b(?:(?:average|avg\.?|typical)\s+)?(?:win(?:ner|ners|s)?|gain)\s+(?:is|are|=)\s+"
    + _n("t") + r"\s*(?:times|x)\s+(?:(?:as\s+(?:big|large)\s+as|bigger\s+than|larger\s+than)\s+)?"
    r"(?:my\s+|the\s+|our\s+)?(?:(?:average|avg\.?|typical)\s+)?(?:loss(?:es)?|loser|losers)\b",
    re.I)
_UNIT: Final = r"\s*(?:R\b|%|x\b)?"
_PAIR_WL: Final = re.compile(_AVG_WIN + _n("aw") + _UNIT + r".{0,40}?" + _AVG_LOSS + _n("al"),
                             re.I | re.S)
_PAIR_LW: Final = re.compile(_AVG_LOSS + _n("bl") + _UNIT + r".{0,40}?" + _AVG_WIN + _n("bw"),
                             re.I | re.S)
_RATIO: Final = re.compile(
    r"\bwin\s*(?:/|-|to|:)\s*loss\s+ratio\s+(?:of\s+|is\s+|=\s*)?" + _n("r") + r"|"
    r"\b(?:reward|gain|profit)\s*(?:/|-|to|:)\s*(?:risk|loss)(?:\s+ratio)?\s+(?:of\s+|is\s+|"
    r"=\s*)?" + _n("r2") + r"|\bpayoff\s+(?:of\s+|ratio\s+(?:of\s+)?|is\s+)?" + _n("r3")
    + r"(?!\s*:)|" + _n("r4") + r"\s*(?:R|x)\s*(?:payoff|reward|win)\b", re.I)
_COLON: Final = re.compile(r"\b" + _n("w") + r"\s*:\s*" + _n("l"))
_EVEN: Final = re.compile(r"\beven\s+money\b", re.I)


def kelly_payoff(text: str) -> float | None:
    """The payoff ratio b (average win over average loss) a question states, or None when it states
    none. Forms read: "average win 1.5R, average loss 1R" (in either order, with or without R, $
    or %), "my average winner is 1.5 times my average loser", "payoff 1.5:1", "1.5:1", "payoff
    ratio of 1.5", "win/loss ratio of 1.5", "reward-to-risk 1.5", "1.5R payoff" and "even money".

    Round 44's judge put "55% win rate, average win 1.5R, average loss 1R" to the console and got
    "a 1:1 payoff (no payoff was given ...)", a 10.0% Kelly fraction for what is 25.0%: only the
    ratio form had been read."""
    if (m := _TIMES.search(text)):
        return float(m.group("t"))
    if (m := _PAIR_WL.search(text)) and float(m.group("al")) > 0:
        return float(m.group("aw")) / float(m.group("al"))
    if (m := _PAIR_LW.search(text)) and float(m.group("bl")) > 0:
        return float(m.group("bw")) / float(m.group("bl"))
    if (m := _RATIO.search(text)):
        return float(next(v for k, v in m.groupdict().items() if v and k.startswith("r")))
    if (m := _COLON.search(text)) and float(m.group("l")) > 0:
        return float(m.group("w")) / float(m.group("l"))
    if _EVEN.search(text):
        return 1.0
    return None


# --------------------------------------------------------------------------- 5. earnings quality

_QUALITY: Final = re.compile(
    r"\bearnings\s+quality\b|\bquality\s+of\s+(?:[\w&.'\u2019-]+\s+){0,3}?earnings\b|\baccruals?\b|"
    r"\bcash\s+conversion\b|\bcash\s*flow\s+(?:versus|vs\.?|against|compared\s+(?:to|with)|"
    r"relative\s+to)\s+(?:net\s+)?(?:income|profit|earnings)\b|\bnet\s+income\s+(?:versus|vs\.?|"
    r"against)\s+(?:operating\s+)?cash\s*flow\b|"
    # "Does Amazon's cash flow back up its reported profit?" (round 44 re-ask)
    r"\bcash\s*flows?\b[^?]{0,30}\b(?:back(?:s|ing)?\s+up|support\w*|match\w*|cover\w*|keep\w*\s+"
    r"up\s+with|justif\w*)\b[^?]{0,30}\b(?:profits?|earnings|net\s+income)\b|\b(?:profits?|"
    r"earnings)\b[^?]{0,30}\bbacked\s+by\s+(?:real\s+)?cash\b", re.I)


def _quality_inputs(ticker: str) -> dict[str, Any]:
    """Four quarters of net income, operating cash flow and capital spending for the year and the
    year before, and total assets at the three year-ends they need, from SEC XBRL company facts."""
    from argus.lui.research import shareholder_metrics as sm

    c = sm.read_company(ticker)
    if c.error or c.end is None:
        raise RuntimeError(c.error or "no filing period")
    doc = sm._document(ticker) or {}
    assets = filing_figures._instants(doc.get("us-gaap", {}), ("Assets",))
    near = [sm._near(assets, c.end - timedelta(days=365 * k), 12) for k in (0, 1, 2)]
    return {"company": c, "assets": [n[1] if n else None for n in near]}


def _earnings_quality(text: str, prior: Sequence[str]) -> list[str] | None:
    if not _QUALITY.search(text):
        return None
    tickers = _tickers(text, 2)
    if not tickers:
        return None
    data = ("each company's own 10-Q and 10-K, read from SEC EDGAR's XBRL company facts "
            "(net income, operating cash flow, capital spending, total assets); four-quarter "
            "totals.")
    out: list[str] = []
    for ticker in tickers:
        try:
            got = _quality_inputs(ticker)
        except Exception as exc:
            out.append(f"{ticker}: its filings could not be read from SEC EDGAR just now "
                       f"({type(exc).__name__}), so no earnings-quality figure is given.")
            continue
        c = got["company"]
        a0, a1, a2 = got["assets"]
        if c.net is None or c.cfo is None:
            out.append(f"{c.label}: its filings do not tag net income and operating cash flow for "
                       "four quarters, so earnings quality is not read.")
            continue
        ratio = c.cfo / c.net if c.net > 0 else None
        accrual = ((c.net - c.cfo) / ((a0 + a1) / 2)) if a0 and a1 else None
        prior_ratio = (c.cfo_prior / c.net_prior
                       if c.cfo_prior is not None and c.net_prior and c.net_prior > 0 else None)
        prior_accrual = (((c.net_prior - c.cfo_prior) / ((a1 + a2) / 2))
                         if c.net_prior is not None and c.cfo_prior is not None and a1 and a2
                         else None)
        fcf_conv = c.fcf / c.net if c.fcf is not None and c.net > 0 else None
        sense = ("operating cash flow is larger than reported profit, so the profit is backed by "
                 "cash" if ratio is not None and ratio >= 1 else
                 "operating cash flow is below reported profit, so part of the profit has not yet "
                 "arrived as cash" if ratio is not None else
                 "net income is not positive, so a cash-to-profit ratio is not meaningful")
        lead = (f"{c.label}, {c.period}: operating cash flow {_usd(c.cfo)} against net income "
                f"{_usd(c.net)}"
                + (f", {ratio:.2f}x" if ratio is not None else "")
                + (f"; accruals ratio {accrual:+.1%} of average total assets" if accrual is not None
                   else "") + f" - {sense}.")
        out.append(lead)
        if accrual is not None:
            out.append(f"{c.ticker} accruals ratio = (net income {_usd(c.net)} - operating cash "
                       f"flow {_usd(c.cfo)}) / average total assets "
                       f"{_usd((a0 + a1) / 2)} ({_usd(a1)} a year earlier, {_usd(a0)} now) = "
                       f"{accrual:+.2%}; negative means cash earnings exceed accounting "
                       "earnings, positive means profit has run ahead of cash.")
        if c.capex is not None and c.fcf is not None:
            out.append(f"{c.ticker} free-cash-flow conversion: operating cash flow less capital "
                       f"spending ({_usd(c.capex)} cash paid for property and equipment) is "
                       f"{_usd(c.fcf)}"
                       + (f", {fcf_conv:.0%} of net income" if fcf_conv is not None else "")
                       + "; finance leases are not in that tag.")
        if prior_ratio is not None or prior_accrual is not None:
            out.append(f"{c.ticker} a year earlier (the four quarters before): "
                       + (f"cash flow {prior_ratio:.2f}x net income" if prior_ratio is not None
                          else "cash flow to net income not computable")
                       + (f", accruals ratio {prior_accrual:+.1%}" if prior_accrual is not None
                          else "") + ".")
    if not out:
        return None
    out[0] = "Bottom line: " + out[0]
    return _close(out, data)


# --------------------------------------------------------------------------- 6. insiders vs buyback

_INSIDER: Final = re.compile(r"\binsiders?\b|\bform\s*4\b|\bexecutives?\s+(?:sell\w*|sold)\b", re.I)
_BUYBACK_WORD: Final = re.compile(r"\bbuy[\s-]?backs?\b|\brepurchas\w+|\bbuying\s+back\b|"
                                  r"\bbought\s+back\b", re.I)
INSIDER_DAYS: Final = 90


def _insider_totals(ticker: str, days: int = INSIDER_DAYS) -> dict[str, Any]:
    """Open-market Form 4 sales (code S) and purchases (code P) over the last ``days`` days,
    grouped one decision per filing and insider as `insider_flow` groups them."""
    from argus.lui.research.insider_flow import _grouped
    from argus.market.insider import InsiderSource

    since = datetime.now(UTC) - timedelta(days=days)
    trades, status = InsiderSource().trades(ticker, since=since, limit=200)
    read = next((s for s in status if "Form 4(s) read" in s), "")
    filings = int(m.group(1)) if (m := re.search(r"(\d+) Form 4", read)) else 0
    if not trades and not filings:
        raise RuntimeError(status[-1].split(": ", 1)[-1] if status else "no reply")
    sales, buys = _grouped(trades, "S"), _grouped(trades, "P")
    return {"days": days, "filings": filings, "sales_usd": sum(s["usd"] for s in sales),
            "sales_n": len(sales), "planned": sum(1 for s in sales if s["planned"]),
            "buys_usd": sum(b["usd"] for b in buys), "buys_n": len(buys),
            "capped": len(trades) >= 200}


def _buyback(ticker: str) -> Any:
    from argus.lui.research import shareholder_metrics as sm

    return sm.read_company(ticker)


def _insider_buyback(text: str, prior: Sequence[str]) -> list[str] | None:
    if not (_INSIDER.search(text) and _BUYBACK_WORD.search(text)):
        return None
    tickers = _tickers(text, 2)
    if not tickers:
        return None
    data = ("SEC EDGAR Form 4 filings (open-market sales, code S, and purchases, code P, over "
            f"the last {INSIDER_DAYS} days); the company's own 10-Q and 10-K cash-flow statement "
            "(repurchases of common stock, four quarters).")
    out: list[str] = []
    for ticker in tickers:
        try:
            ins = _insider_totals(ticker)
        except Exception as exc:
            ins = None
            why_ins = str(exc)[:80] or type(exc).__name__
        try:
            c = _buyback(ticker)
            if c.error:
                raise RuntimeError(c.error)
        except Exception:
            c = None
        if ins is None and c is None:
            out.append(f"{ticker}: neither its Form 4 filings nor its cash-flow statement could "
                       "be read just now, so no net figure is given.")
            continue
        if ins is None:
            out.append(f"{ticker}: insider sales could not be read just now ({why_ins}); the "
                       f"company bought back "
                       f"{_usd(c.buyback) if c.buyback else 'nothing it tags'} in the "
                       f"{c.period}.")
            continue
        sold = f"insiders sold {_usd(ins['sales_usd'])} on the open market"
        if ins["buys_n"]:
            sold += f" and bought {_usd(ins['buys_usd'])}"
        else:
            sold += " and bought nothing"
        sold += f" over the last {ins['days']} days ({ins['sales_n']} sale decisions"
        sold += f", {ins['planned']} under 10b5-1 plans)" if ins["sales_n"] else ")"
        if c is None or c.buyback is None:
            out.append(f"{ticker}: {sold}; its repurchases could not be read from its filings "
                       "just now, so no net figure is given.")
            continue
        net_insiders = ins["buys_usd"] - ins["sales_usd"]
        pro_rata = c.buyback * ins["days"] / 365
        net = c.buyback + net_insiders
        out.append(f"{ticker}: {sold}; the company bought back {_usd(c.buyback)} in the "
                   f"{c.period}; net of the two, {_usd(abs(net))} more stock was "
                   f"{'bought' if net >= 0 else 'sold'} than "
                   f"{'sold' if net >= 0 else 'bought'} ({_usd(c.buyback)} bought back "
                   f"{'less' if net_insiders < 0 else 'plus'} {_usd(abs(net_insiders))} of "
                   f"insider {'sales' if net_insiders < 0 else 'purchases'}), on windows that "
                   "differ.")
        out.append(f"{ticker} on a like-for-like window: scaled to {ins['days']} days the buyback "
                   f"is about {_usd(pro_rata)}, so insiders sold "
                   f"{_plain(ins['sales_usd'] / pro_rata) if pro_rata else 'n/a'} of what the "
                   "company bought back in the same span.")
        out.append(f"{ticker}: the windows differ - Form 4 sales cover {ins['days']} days, the "
                   f"buyback the {c.period} - so the net line mixes a short window with a long "
                   "one, and the second line shows it scaled. Insider sales are weak evidence "
                   "(pay, taxes and diversification explain most), and a buyback offsets share "
                   "issuance to staff before it shrinks the share count."
                   + (" The Form 4 read hit its 200-line cap, so the sales total is a floor."
                      if ins["capped"] else ""))
    if not out:
        return None
    out[0] = "Bottom line: " + out[0]
    return _close(out, data)


# --------------------------------------------------------------------------- 7. analyst revisions

_REVISION: Final = re.compile(
    r"\b(?:revis\w+|rais(?:e|ed|ing)|cut(?:ting|s)?|lower(?:ed|ing)|upgrad\w+|downgrad\w+|"
    r"trend(?:ing)?)\b", re.I)
_ESTIMATE: Final = re.compile(r"\b(?:eps|earnings)\s+(?:estimates?|forecasts?|expectations?)\b|"
                              r"\b(?:estimates?|forecasts?|consensus)\b|\banalysts?\b", re.I)
_EPS_WORD: Final = re.compile(r"\beps\b|\bearnings\b|\bestimates?\b|\bforecasts?\b|\bconsensus\b",
                              re.I)
_PERIODS: Final = (("0q", "the quarter to"), ("0y", "the year to"), ("+1y", "the year to"))


def _trend(ticker: str) -> dict[str, dict[str, Any]]:
    """Yahoo's earningsTrend for the periods 0q, +1q, 0y and +1y: the consensus EPS now and
    7, 30, 60 and 90 days ago, and the revision counts, through `market/estimates`."""
    from argus.lui.research.fundamentals import raw_number
    from argus.market.estimates import EstimatesSource

    summary = EstimatesSource().summary(ticker, "earningsTrend")
    rows = (summary.get("earningsTrend") or {}).get("trend") or []
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        period = str(row.get("period"))
        trend, rev = row.get("epsTrend") or {}, row.get("epsRevisions") or {}
        out[period] = {
            "end": row.get("endDate"), "now": raw_number(trend.get("current")),
            "d7": raw_number(trend.get("7daysAgo")), "d30": raw_number(trend.get("30daysAgo")),
            "d60": raw_number(trend.get("60daysAgo")), "d90": raw_number(trend.get("90daysAgo")),
            "up7": raw_number(rev.get("upLast7days")),
            "down7": raw_number(rev.get("downLast7Days")),
            "up30": raw_number(rev.get("upLast30days")),
            "down30": raw_number(rev.get("downLast30days")),
            "analysts": raw_number((row.get("earningsEstimate") or {}).get("numberOfAnalysts"))}
    return out


def _period_name(period: str, end: Any) -> str:
    try:
        when = date.fromisoformat(str(end))
    except ValueError:
        return {"0q": "the current quarter", "0y": "this year", "+1y": "next year"}[period]
    if period == "0q":
        return f"the quarter to {when:%d %b %Y}"
    return f"{'next' if period == '+1y' else 'this'} year (to {when:%b %Y})"


def _eps(x: float) -> str:
    return f"-${abs(x):.2f}" if x < 0 else f"${x:.2f}"


def _eps_change(now: float, then: float) -> tuple[float, str]:
    """(a signed size for ranking, how the move reads). A percentage only when both estimates are
    positive; a move through zero or between losses is given as the two figures, ranked as +-1."""
    if then > 0 and now > 0:
        change = now / then - 1
        return change, f"from {_eps(then)} to {_eps(now)} ({_pct(change)})"
    sign = 1.0 if now > then else -1.0 if now < then else 0.0
    return sign, f"from {_eps(then)} to {_eps(now)}"


def _revisions(text: str, prior: Sequence[str]) -> list[str] | None:
    if not (_REVISION.search(text) and _ESTIMATE.search(text) and _EPS_WORD.search(text)):
        return None
    tickers = _tickers(text, 1)
    if not tickers:
        return None
    ticker = tickers[0]
    data = ("Yahoo Finance quoteSummary earningsTrend (the analysts' consensus EPS now and 7, 30, "
            "60 and 90 days ago, and the number of analysts revising it up or down).")
    trend: dict[str, dict[str, Any]] = {}
    for _attempt in range(2):
        try:
            trend = _trend(ticker)
        except Exception:
            trend = {}
        if trend:
            break
    usable = {p: v for p, v in trend.items() if p in dict(_PERIODS)
              and v.get("now") is not None and v.get("d90")}
    if not usable:
        return [f"Bottom line: analyst EPS revisions for {ticker} are not read just now - Yahoo "
                "Finance's earnings-trend feed did not return a 90-day estimate history. What this "
                "console does read is reported EPS from the company's filings, and that is not a "
                "revision: it says what was earned, not how forecasts moved.",
                f"Data: {data} Not advice."]
    pieces: list[str] = []
    detail: list[str] = []
    for period, _ in _PERIODS:
        v = usable.get(period)
        if v is None:
            continue
        _, said = _eps_change(v["now"], v["d90"])
        name = _period_name(period, v["end"])
        pieces.append(f"{name} {said}")
        steps = " -> ".join(f"{_eps(v[k])} ({label})" for k, label in
                            (("d90", "90 days ago"), ("d60", "60"), ("d30", "30"), ("d7", "7"))
                            if v.get(k) is not None)
        counts = ""
        if v.get("up30") is not None and v.get("down30") is not None:
            counts = (f"; revisions in the last 30 days: {v['up30']:.0f} up, {v['down30']:.0f} down"
                      + (f" ({v['up7']:.0f} up, {v['down7']:.0f} down in the last 7)"
                         if v.get("up7") is not None and v.get("down7") is not None else ""))
        crowd = f"; {v['analysts']:.0f} analysts" if v.get("analysts") else ""
        detail.append(f"{name[0].upper()}{name[1:]}: {steps} -> {_eps(v['now'])} now"
                      f"{counts}{crowd}.")
    lead = usable.get("0y") or next(iter(usable.values()))
    move = _eps_change(lead["now"], lead["d90"])[0]
    word = "raised" if move > 0.01 else "cut" if move < -0.01 else "left little changed"
    from argus.lui.research.shareholder_metrics import SHORT_NAMES

    name = SHORT_NAMES.get(ticker, ticker)
    out = [f"Bottom line: over the last 90 days analysts have {word} {name}'s ({ticker}) EPS "
           f"estimates: " + "; ".join(pieces) + ".", *detail,
           "These are changes in the consensus forecast (analysts' adjusted EPS), not reported "
           "EPS, and the 90-day figure is Yahoo's own snapshot of the consensus 90 days ago."]
    return _close(out, data)


# --------------------------------------------------------------------------- 8. guidance and react

_GUIDE: Final = re.compile(r"\bguid(?:e|ed|es|ing|ance)\b", re.I)
_GUIDE_CONTEXT: Final = re.compile(r"\b(?:quarter|results|earnings|report\w*|beat|miss\w*|"
                                   r"react\w*|stock|shares)\b", re.I)
_OUTLOOK: Final = re.compile(r"\boutlook\b[^?]{0,40}\b(?:next|coming|upcoming|this)\s+quarter\b|"
                             r"\b(?:next|coming|upcoming)\s+quarter\b[^?]{0,40}\boutlook\b|"
                             r"\boutlook\b[^?]{0,30}\b(?:earnings|results|release)\b", re.I)
_BEATMISS: Final = re.compile(r"\bbeat\b|\bmiss(?:ed|es)?\b|\bsurpris\w+|\bin\s+line\b", re.I)


def _releases(ticker: str) -> list[Any]:
    from argus.market.earnings_release import EarningsReleaseSource

    return EarningsReleaseSource().releases(ticker, count=2)


def _reactions(ticker: str) -> list[Any]:
    from argus.lui.research import earnings_moves

    return earnings_moves.reactions(ticker, 2)


def _last_surprise(ticker: str) -> tuple[date, float, float] | None:
    """The newest quarter's EPS actual and consensus estimate, from Yahoo's earningsHistory."""
    from argus.market.estimates import EstimatesSource

    rows = (EstimatesSource().summary(ticker, "earningsHistory") or {}).get(
        "earningsHistory", {}).get("history") or []
    best: tuple[date, float, float] | None = None
    for row in rows:
        try:
            end = datetime.fromtimestamp(int(row["quarter"]["raw"]), UTC).date()
            actual, estimate = float(row["epsActual"]["raw"]), float(row["epsEstimate"]["raw"])
        except (KeyError, TypeError, ValueError):
            continue
        if best is None or end > best[0]:
            best = (end, actual, estimate)
    return best


def _quote(sentence: str) -> str:
    return sentence.replace("\ufffd", "'").strip()[:280]


def _guidance(text: str, prior: Sequence[str]) -> list[str] | None:
    if not ((_GUIDE.search(text) and _GUIDE_CONTEXT.search(text)) or _OUTLOOK.search(text)):
        return None
    tickers = _tickers(text, 1)
    if not tickers:
        return None
    ticker = tickers[0]
    from argus.lui.research.shareholder_metrics import SHORT_NAMES

    name = SHORT_NAMES.get(ticker, ticker)
    data = ("the company's own results release (8-K item 2.02, exhibit 99.1) on SEC EDGAR; the "
            "move is the close before the release to the first close after it (Yahoo Finance "
            "split-adjusted daily closes, 8-K acceptance times), with the S&P 500 (SPY) beside it.")
    try:
        releases = _releases(ticker)
    except Exception:
        releases = []
    if not releases:
        return _failed(f"{name}'s latest guidance", "its results release on SEC EDGAR could not "
                       "be read", data)
    latest = releases[0]
    out: list[str] = []
    reaction = None
    try:
        for r in _reactions(ticker):
            if abs((r.released.date() - latest.filed).days) <= 1:
                reaction = r
                break
    except Exception:
        reaction = None
    if latest.guided_revenue:
        band = (f", plus or minus {latest.guided_band_pct:g}%"
                if latest.guided_band_pct is not None else "")
        growth = (f" ({_pct(latest.guided_revenue / latest.revenue - 1)} on the "
                  f"{_usd(latest.revenue)} it just reported)" if latest.revenue else "")
        guide = (f"{name} guided next-quarter revenue to {_usd(latest.guided_revenue)}{band}"
                 f"{growth} in its release of {_d(latest.filed)}")
    elif latest.outlook:
        guide = (f"{name}'s release of {_d(latest.filed)} gives no numeric revenue guidance; its "
                 "outlook section says: \u201c" + _quote(latest.outlook[0]) + "\u201d")
    else:
        guide = (f"{name}'s release of {_d(latest.filed)} contains no outlook or guidance "
                 "section; whatever management says about the next quarter is said on the call, "
                 "which is not read here")
    if reaction is not None:
        spy = (f", the S&P 500 {_pct(reaction.market)} that day"
               if reaction.market is not None else "")
        guide += (f"; the stock moved {_pct(reaction.move)} by the close of "
                  f"{_d(reaction.moved_day)}, the first session after it{spy}")
    else:
        guide += "; the price reaction could not be matched to that release just now"
    out.append(guide + ".")
    beat = ""
    if _BEATMISS.search(text):
        try:
            surprise = _last_surprise(ticker)
        except Exception:
            surprise = None
        if surprise is not None and surprise[0] < latest.filed:
            end, actual, estimate = surprise
            word = ("beat" if actual > estimate else "missed" if actual < estimate
                    else "matched")
            gap = f", {actual / estimate - 1:+.0%}" if estimate and word != "matched" else ""
            beat = (f"{name} {word} the consensus for the quarter to {_d(end)}: EPS "
                    f"${actual:.2f} against ${estimate:.2f} expected{gap} (Yahoo Finance's "
                    "earnings history, the analysts' adjusted basis)")
            data += " Beat or miss: Yahoo Finance earnings history."
        else:
            beat = (f"{name}'s beat or miss against the consensus could not be read just now "
                    "(Yahoo Finance's earnings history did not return the latest quarter)")
    if beat:
        # two answers, two lines: the beat leads, the guidance follows with its name's capital
        out[0:1] = [f"{beat}.", f"Guidance: {out[0]}"]
    if latest.guided_revenue:
        first = next((i for i, s in enumerate(latest.outlook)
                      if re.search(r"\brevenue\b", s, re.I)), 0)
        rest = [s for i, s in enumerate(latest.outlook) if i != first]
        out.extend(f"Rest of the outlook, in the release's words: \u201c{_quote(s)}\u201d"
                   for s in rest[:3])
        if len(releases) > 1 and releases[1].guided_revenue and latest.revenue:
            before = releases[1]
            gap = latest.revenue / before.guided_revenue - 1
            band = before.guided_band_pct
            where = ("above" if band is not None and gap * 100 > band else
                     "below" if band is not None and gap * 100 < -band else "within")
            out.append(f"Against its own previous guide: {name} had guided "
                       f"{_usd(before.guided_revenue)}"
                       + (f" plus or minus {band:g}%" if band is not None else "")
                       + f" and reported {_usd(latest.revenue)} ({_pct(gap)}), {where} that "
                         "range.")
    elif latest.outlook:
        out.extend(f"Outlook section, in the release's words: \u201c{_quote(s)}\u201d"
                   for s in latest.outlook[:2])
    out.append(f"Source: {latest.url}")
    out[0] = "Bottom line: " + out[0]
    out.append("A guide is the company's own forecast and a reaction is one day's price move on "
               "the whole release (results and outlook together), so the move does not isolate "
               "the guidance.")
    return _close(out, data)


# --------------------------------------------------------------------------- 9. research brief

_BRIEF: Final = re.compile(
    r"\bresearch\s+(?:brief|note|summary|memo)\b|\bbull(?:ish)?\s*(?:/|&|and|vs\.?|versus)\s*"
    r"bear(?:ish)?\b|\bbull\s+case\s+(?:and|&|/)\s+(?:the\s+)?bear\s+case\b", re.I)
_FALSIFY: Final = re.compile(
    r"\bchange\s+(?:your|ur)\s+mind\b|\bfalsif\w+|\bprove\s+(?:the\s+|that\s+)?(?:bear|bull|it)\b|"
    r"\binvalidat\w+|\bwhat\s+would\s+(?:make|show|prove)\b|\bwrong\s+about\b|"
    r"\bwhat\s+(?:would|could)\s+(?:flip|reverse|break|kill)\b|\bwhat\s+am\s+i\s+missing\b", re.I)
_BEAR: Final = re.compile(r"\bbear(?:ish)?\b", re.I)
_BULL: Final = re.compile(r"\bbull(?:ish)?\b", re.I)
TREND_SCALE: Final = 0.20
MIN_SCORE: Final = 0.1
"""A reading weaker than this (a price 2% above its 200-day average) is neutral, not a point."""
MIN_BRIEF_BARS: Final = 260
ITEM_NAMES: Final = {
    "1.01": "a material agreement", "1.05": "a cybersecurity incident", "2.02": "results",
    "2.03": "new debt", "3.02": "an unregistered share sale", "5.02": "a change of "
    "officers or directors", "7.01": "a Regulation FD disclosure", "8.01": "other events",
    "9.01": "financial statements and exhibits", "5.07": "a shareholder vote",
    "1.02": "the end of a material agreement", "2.01": "a completed acquisition or sale"}


@dataclass(frozen=True)
class _Point:
    side: str
    score: float
    short: str
    text: str
    falsifier: str


def _clamp(x: float) -> float:
    return max(-1.0, min(1.0, x))


def _brief_closes(symbol: str) -> tuple[list[date], list[float], str]:
    from argus.lui.research.rule_test import daily_closes

    stamps, closes, where = daily_closes(symbol)
    return [t.date() for t in stamps], closes, where


def _next_report(ticker: str) -> Any:
    from argus.lui.watchlist import earnings_date

    return earnings_date(ticker, datetime.now(UTC).date())


def _ex_dividend(ticker: str) -> str | None:
    from argus.market.bitget_positioning import next_ex_dividend

    return next_ex_dividend(ticker)


def _typical_move(ticker: str) -> tuple[float, int] | None:
    from argus.lui.research import earnings_moves

    found = earnings_moves.reactions(ticker, 4)
    if not found:
        return None
    return sum(abs(r.move) for r in found) / len(found), len(found)


def _recent_8k(ticker: str) -> list[tuple[date, str]]:
    from argus.lui.watchlist import recent_8k

    since = datetime.now(UTC) - timedelta(days=21)
    return [(f.day, f.items) for f in recent_8k([ticker], since).get(ticker, [])]


def _unlock_note(ticker: str) -> str | None:
    """A dated token unlock inside 90 days for a coin, from DeFiLlama's emissions index."""
    from argus.lui.research import token_supply

    coins = token_supply.coin_ids(ticker)
    if not coins:
        return None
    _tick, gecko, name = coins[0]
    now = datetime.now(UTC).timestamp()
    for row in token_supply.unlock_calendar(token_supply.llama_index(), now, 90):
        if row["gecko_id"] == gecko:
            big = row["biggest"]
            when = (f"; the largest dated cliff is {big[1]:,.0f} tokens on "
                    f"{datetime.fromtimestamp(big[0], UTC):%d %b %Y}" if big else "")
            return (f"{name} unlocks about {row['total']:,.0f} tokens ({_usd(row['value'])}, "
                    f"{_plain(row['share'])} of circulating supply) in the next 90 days{when} "
                    "(DeFiLlama emissions index).")
    return f"DeFiLlama's emissions index lists no unlock for {name} in the next 90 days."


@dataclass
class _Facts:
    symbol: str
    ticker: str
    equity: bool
    days: list[date]
    closes: list[float]
    where: str
    company: Any = None
    insider: dict[str, Any] | None = None
    trend: dict[str, dict[str, Any]] | None = None
    report: Any = None
    ex_div: str | None = None
    move: tuple[float, int] | None = None
    filings: list[tuple[date, str]] | None = None
    unlock: str | None = None
    missing: list[str] | None = None


def _gather(symbol: str) -> _Facts:
    from argus.lui.research.parse import is_us_equity
    from argus.truth.coverage import ContextPool

    ticker = symbol.removesuffix("USDT")
    equity = is_us_equity(symbol)
    days, closes, where = _brief_closes(symbol)
    facts = _Facts(symbol, ticker, equity, days, closes, where, missing=[])
    jobs: dict[str, Callable[[], Any]] = {}
    if equity:
        jobs = {"company": lambda: _buyback(ticker), "insider": lambda: _insider_totals(ticker),
                "trend": lambda: _trend(ticker), "report": lambda: _next_report(ticker),
                "ex_div": lambda: _ex_dividend(ticker), "move": lambda: _typical_move(ticker),
                "filings": lambda: _recent_8k(ticker)}
    else:
        jobs = {"unlock": lambda: _unlock_note(ticker)}
    with ContextPool(max_workers=max(1, len(jobs))) as pool:
        futures = {k: pool.submit(fn) for k, fn in jobs.items()}
    for key, future in futures.items():
        try:
            value = future.result()
        except Exception:
            value = None
            assert facts.missing is not None
            facts.missing.append(key)
        if key == "company" and value is not None and getattr(value, "error", None):
            value = None
            assert facts.missing is not None
            facts.missing.append(key)
        setattr(facts, key, value)
    return facts


def _price_points(f: _Facts) -> list[_Point]:
    out: list[_Point] = []
    if len(f.closes) < MIN_BRIEF_BARS:
        return out
    last = f.closes[-1]
    per_year = 252 if f.equity else 365
    month, quarter = (21, 63) if f.equity else (30, 90)
    sma = _mean(f.closes[-200:])
    if len(f.closes) <= per_year + 1:
        return out
    r1, r3, r12 = (last / f.closes[-k - 1] - 1 for k in (month, quarter, per_year))
    gap = last / sma - 1
    out.append(_Point(
        "bull" if gap >= 0 else "bear", _clamp(gap / TREND_SCALE),
        "the price is below its 200-day average" if gap < 0
        else "the price is above its 200-day average",
        f"Price {_px(last)} is {_pct(gap)} against its 200-day average ({_px(sma)}) "
        "(Yahoo Finance and Bitget daily closes).",
        f"a daily close above the 200-day average ({_px(sma)}, {_pct(sma / last - 1)} from here)"
        if gap < 0 else f"a daily close below the 200-day average ({_px(sma)})"))
    out.append(_Point(
        "bull" if r3 >= 0 else "bear", _clamp(r3 / 0.30),
        f"the price is {_pct(r3, 0)} over three months",
        f"Three-month return {_pct(r3, 0)} (one month {_pct(r1, 0)}, twelve months "
        f"{_pct(r12, 0)}; daily closes).",
        "a three-month return turning negative" if r3 >= 0
        else "a three-month return turning positive"))
    if abs(r12) >= 0.20:
        out.append(_Point(
            "bull" if r12 > 0 else "bear", _clamp(r12 / 0.60),
            f"the price is {_pct(r12, 0)} over twelve months",
            f"Twelve-month return {_pct(r12, 0)} (daily closes).",
            "a twelve-month return turning negative" if r12 > 0
            else "a twelve-month return turning positive"))
    high = max(f.closes[-per_year:])
    dd = last / high - 1
    if dd <= -0.20:
        out.append(_Point(
            "bear", _clamp(dd / 0.5), f"the price is {abs(dd):.0%} below its 12-month high",
            f"Price is {_pct(dd, 0)} from its 12-month closing high of {_px(high)}.",
            f"a recovery to within 10% of that high ({_px(high * 0.9)}, "
            f"{_pct(high * 0.9 / last - 1, 0)} from here)"))
    vol = _sd(_log_returns(f.closes[-31:])) * math.sqrt(per_year)
    if vol >= 0.60:
        out.append(_Point(
            "bear", _clamp(-(vol - 0.4) / 0.6), f"30-day volatility is {vol:.0%} a year",
            f"Thirty-day realised volatility is {vol:.0%} a year, so a one-standard-deviation "
            f"month is about {vol / math.sqrt(12):.0%} either way.",
            "30-day realised volatility falling under 40% a year"))
    return out


def _filing_points(f: _Facts) -> list[_Point]:
    c = f.company
    out: list[_Point] = []
    if c is None or c.kind != "us":
        return out
    src = f"{c.ticker} filings, {c.period}, SEC XBRL"
    if c.growth is not None:
        g = c.growth
        out.append(_Point(
            "bull" if g >= 0 else "bear", _clamp(g / 0.25),
            "revenue is shrinking" if g < 0 else "revenue is growing",
            f"Revenue {_usd(c.rev)}, {_pct(g)} on the four quarters before ({src}).",
            "revenue growth above zero in the next 10-Q" if g < 0
            else "revenue growth falling below zero in the next 10-Q"))
    m, m0 = c.margin("net"), c.margin("net", prior=True)
    if m is not None and m0 is not None:
        d_pts = m - m0
        out.append(_Point(
            "bull" if d_pts >= 0 else "bear", _clamp(d_pts / 0.05),
            "the net margin is narrowing" if d_pts < 0 else "the net margin is widening",
            f"Net margin {m:.1%} against {m0:.1%} the year before, {d_pts * 100:+.1f} points "
            f"({src}).",
            (f"net margin back above {m0:.1%} over the next four quarters" if 0 < m0 <= 0.5
             else "four-quarter net income turning positive") if d_pts < 0
            else (f"net margin falling back under {m0:.1%}" if m0 > 0
                  else "four-quarter net income turning negative")))
    if c.fcf is not None and c.fcf_margin is not None:
        fm = c.fcf_margin
        out.append(_Point(
            "bull" if fm >= 0 else "bear", _clamp(fm / 0.15),
            "free cash flow is negative" if fm < 0 else "free cash flow is positive",
            f"Free cash flow {_usd(c.fcf)} ({fm:.0%} of revenue): operating cash flow "
            f"{_usd(c.cfo)} less capital spending {_usd(c.capex)} ({src}).",
            "free cash flow turning positive in the next 10-Q" if fm < 0
            else "free cash flow falling below zero in the next 10-Q"))
    pe = c.trailing_pe
    if pe is not None and c.price and c.eps:
        out.append(_Point(
            "bear" if pe > 25 else "bull", _clamp((25 - pe) / 50),
            f"the stock trades at {pe:.0f}x trailing earnings",
            f"Trailing P/E {pe:.1f} (price {_px(c.price)} over four quarters' diluted EPS "
            f"${c.eps:.2f}; Yahoo Finance price, SEC XBRL EPS).",
            f"four-quarter EPS reaching ${c.price / 25:.2f} ({c.price / 25 / c.eps - 1:+.0%} on "
            "today's), a 25x multiple at today's price" if pe > 25
            else "four-quarter EPS falling, or the price rising, until the multiple passes 25"))
    elif c.eps is not None and c.eps <= 0:
        out.append(_Point(
            "bear", -0.6, "there are no trailing earnings",
            f"Four-quarter diluted EPS is {_eps(c.eps)}, so there is no trailing P/E ({src}).",
            "four-quarter diluted EPS turning positive"))
    if c.buyback_share_of_cap is not None and c.buyback_share_of_cap >= 0.01:
        out.append(_Point(
            "bull", _clamp(c.buyback_share_of_cap / 0.04),
            f"the company bought back {c.buyback_share_of_cap:.1%} of its market cap",
            f"The company bought back {_usd(c.buyback)}, {c.buyback_share_of_cap:.1%} of its "
            f"market cap ({src}).",
            "repurchases stopping or the share count rising in the next 10-Q"))
    if c.share_change is not None and c.share_change >= 0.02:
        out.append(_Point(
            "bear", _clamp(-c.share_change / 0.05), "the share count is rising",
            f"Shares outstanding are {_pct(c.share_change)} on a year earlier "
            f"({c.shares_basis or 'SEC XBRL'}).",
            "a flat or falling share count in the next 10-Q"))
    return out


def _flow_points(f: _Facts) -> list[_Point]:
    out: list[_Point] = []
    ins = f.insider
    if ins is not None:
        if ins["buys_n"]:
            out.append(_Point(
                "bull", 0.6, "insiders are buying with their own money",
                f"{ins['buys_n']} open-market insider purchases worth {_usd(ins['buys_usd'])} "
                f"in {ins['days']} days (SEC Form 4, code P).",
                "open-market purchases stopping and sales by officers starting"))
        elif ins["sales_usd"] >= 1e6:
            out.append(_Point(
                "bear", -0.3, "insiders are selling and none are buying",
                f"Insiders sold {_usd(ins['sales_usd'])} on the open market in {ins['days']} "
                f"days ({ins['sales_n']} decisions, {ins['planned']} under 10b5-1 plans) and "
                "bought nothing (SEC Form 4); weak evidence, since pay and taxes drive most "
                "sales.",
                "an open-market insider purchase (Form 4 code P) by an officer"))
    trend = (f.trend or {}).get("0y")
    if trend and trend.get("now") is not None and trend.get("d90") is not None:
        change, said = _eps_change(trend["now"], trend["d90"])
        if abs(change) >= 0.03:
            out.append(_Point(
                "bull" if change > 0 else "bear", _clamp(change / 0.10),
                "analysts have been raising estimates" if change > 0
                else "analysts have been cutting estimates",
                f"Consensus EPS for {_period_name('0y', trend.get('end'))} moved {said} in 90 "
                f"days; {trend.get('up30') or 0:.0f} up and {trend.get('down30') or 0:.0f} down "
                "revisions in the last 30 (Yahoo Finance).",
                "estimate revisions turning net upward over a 30-day window" if change < 0
                else "estimate revisions turning net downward over a 30-day window"))
    return out


def _watch(f: _Facts) -> list[str]:
    out: list[str] = []
    if f.equity:
        r = f.report
        if r is not None:
            when = f"{r.day:%d %b %Y}" + (f" ({r.timing})" if r.timing else "")
            est = ", date estimated" if getattr(r, "estimated", False) else ""
            typical = (f"; on its last {f.move[1]} results the stock moved {f.move[0]:.1%} either "
                       "way on average (SEC 8-K times, Yahoo closes)" if f.move else "")
            out.append(f"{when}: next earnings report ({r.source}{est}){typical}.")
        else:
            out.append("The next earnings date could not be read just now.")
        if f.ex_div:
            out.append(f.ex_div)
        for day, items in (f.filings or [])[:2]:
            codes = [c.strip() for c in items.split(",") if c.strip()]
            said = ", ".join(f"item {c} ({ITEM_NAMES[c]})" if c in ITEM_NAMES else f"item {c}"
                             for c in codes)
            out.append(f"{_d(day)}: an 8-K was filed covering {said} (SEC EDGAR) - read it for "
                       "what it changes.")
        if f.trend and f.trend.get("0q", {}).get("end"):
            out.append("The consensus for the current quarter keeps moving until the report: "
                       f"{_period_name('0q', f.trend['0q']['end'])} (Yahoo Finance).")
    else:
        out.append(f.unlock or "No token-unlock schedule could be read just now (DeFiLlama's "
                   "emissions index); coins also reprice on funding and open-interest "
                   "swings, which this brief does not read.")
    return out


def _brief_points(f: _Facts) -> tuple[list[_Point], list[_Point]]:
    every = [*_price_points(f), *_filing_points(f), *_flow_points(f)]
    bull = sorted((p for p in every if p.side == "bull" and p.score >= MIN_SCORE),
                  key=lambda p: -p.score)[:4]
    bear = sorted((p for p in every if p.side == "bear" and p.score <= -MIN_SCORE),
                  key=lambda p: p.score)[:4]
    return bull, bear


def _brief_ticker(text: str) -> str | None:
    from argus.lui.research import research_symbols

    symbols = research_symbols(text)[0]
    if symbols:
        return symbols[0]
    return None


def _brief(text: str, prior: Sequence[str]) -> list[str] | None:
    follow = bool(_FALSIFY.search(text)) and (_BEAR.search(text) or _BULL.search(text)) is not None
    carried = False
    symbol: str | None
    if follow:
        symbol = _brief_ticker(text)
        if symbol is None:
            source = next((q for q in reversed(prior[-4:]) if _BRIEF.search(q)), None)
            symbol = _brief_ticker(source) if source is not None else None
            carried = True
    elif _BRIEF.search(text):
        symbol = _brief_ticker(text)
    else:
        return None
    if symbol is None:
        return None
    name = symbol.removesuffix("USDT")
    data = ("Yahoo Finance and Bitget daily closes; SEC EDGAR 10-Q/10-K XBRL facts and Form 4 "
            "filings; Yahoo Finance consensus EPS trend and earnings calendar.")
    try:
        facts = _gather(symbol)
    except Exception:
        return _failed(f"the research brief on {name}", "its price history did not arrive", data)
    data = data.replace("Yahoo Finance and Bitget daily closes", facts.where[:1].upper()
                        + facts.where[1:] if facts.where else "daily closes")
    bull, bear = _brief_points(facts)
    if follow:
        side = "bull" if _BULL.search(text) and not _BEAR.search(text) else "bear"
        points = bull if side == "bull" else bear
        if not points:
            return [f"Bottom line: the {side} case on {name} has no measured point to test: none "
                    f"of the readings in the brief crosses its bar on that side.",
                    f"Data: {data} Not advice."]
        out = [f"Bottom line: what would change the {side} case on {name}: {len(points)} measured "
               f"point{'s' if len(points) != 1 else ''} stand behind it, and each is undone by a "
               "reading you can check."]
        out.extend(f"{i}. {p.short[0].upper()}{p.short[1:]} ({p.text.rstrip('.')}). It would "
                   f"change if: {p.falsifier}." for i, p in enumerate(points, 1))
        dated = _watch(facts)
        if dated:
            out.append("The first dated test of several of these: " + dated[0])
        if carried:
            out.append(f"This is the {side} case from the research brief on {name} above, "
                       "recomputed on today's data.")
        return _close(out, data)
    lead = (f"{name}, on price history to {_d(facts.days[-1])}: the measured evidence has "
            f"{len(bull)} point{'s' if len(bull) != 1 else ''} for the bull case and "
            f"{len(bear)} for the bear case"
            + (f"; the strongest against is that {bear[0].short}" if bear else "")
            + (f", the strongest for is that {bull[0].short}" if bull else "") + ".")
    out = [lead, "Bull case:"]
    out.extend(f"{i}. {p.text}" for i, p in enumerate(bull, 1))
    if not bull:
        out.append("None of the measured readings crosses its bar on the bull side.")
    out.append("Bear case:")
    out.extend(f"{i}. {p.text}" for i, p in enumerate(bear, 1))
    if not bear:
        out.append("None of the measured readings crosses its bar on the bear side.")
    out.append("What to watch:")
    out.extend(f"- {w}" for w in _watch(facts))
    if facts.missing:
        out.append("Not read just now: " + ", ".join(sorted(facts.missing)) + " (the source did "
                   "not answer), so no point from them is included.")
    if not facts.equity:
        out.append("A coin has no earnings, filings or insiders, so the cases rest on price, "
                   "volatility and its unlock schedule only.")
    out.append("A point counts when its reading crosses a stated bar (price against its 200-day "
               "average, a 20% fall from the 12-month high, 30-day volatility over 60%, revenue "
               "and margin direction, free cash flow sign, trailing P/E against 25, share-count "
               "growth over 2%, a buyback over 1% of market cap, estimate revisions of 3% or "
               "more, open-market insider buying or over $1m of selling); ask \"what would "
               "change your mind on the bear case?\" for the readings that would undo each "
               "bear point.")
    return _close(out, data)


# --------------------------------------------------------------------------- entry point


def lines(text: str, prior: Sequence[str] = ()) -> list[str] | None:
    """The answer to a scenario, risk-parity, gold-versus-real-yield, earnings-quality,
    insider-versus-buyback, estimate-revision, guidance or research-brief question, or None when
    ``text`` is none of them (every reader here is strict, so the console's other readers keep
    their questions). ``prior`` is the conversation's earlier questions, oldest first."""
    readers: tuple[tuple[str, Callable[[str, Sequence[str]], Any]], ...] = (
        ("the research brief", _brief), ("the scenario", _scenario),
        ("the risk-parity weights", _parity), ("gold against real yields", _gold_real_yield),
        ("the earnings-quality figures", _earnings_quality),
        ("the insider and buyback figures", _insider_buyback),
        ("the analyst revisions", _revisions), ("the guidance", _guidance))
    for name, reader in readers:
        try:
            found = reader(text, prior)
        except Exception as exc:
            return _failed(name, f"{type(exc).__name__}", "see the sources named by each figure")
        if found is not None:
            return list(found)
    return None


__all__ = ["equal_risk_weights", "kelly_payoff", "lines"]
