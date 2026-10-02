"""Book arithmetic the answers share: dollar lines, beta, value at risk, the size that stays inside
a risk budget."""

from __future__ import annotations

import itertools
import math
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from argus.desk.portfolio import (
    PortfolioError,
    align,
    beta,
    decompose,
    rebalance,
)
from argus.lui.answer import Source
from argus.lui.question import (
    TRADED_SYMBOLS,
)
from argus.lui.research.data import (
    _FETCH_SLOTS,
    MarketData,
)
from argus.lui.research.kinds import (
    BENCHMARK,
    RISK_BUDGET,
    ResearchRequest,
    _t,
)
from argus.lui.research.text import sentence_cut
from argus.lui.trace import trace_module


def _book_dollar_lines(request: ResearchRequest, data: MarketData, is_open: Any,
                       worth: float) -> list[str]:
    """The saved book in dollars at a stated book value, with each name's ceiling under the risk
    budget in dollars too — what "the dollar risk budget for each name if my book is worth
    $250,000" asks (2026-09-25 audit, round 2)."""
    columns = _open_columns(data.raw, is_open)
    parts = []
    for sym, weight in sorted(request.book.items(), key=lambda kv: -kv[1]):
        if weight < 0:
            # A short is said as one, with no long-side ceiling: "TSLA $-2,000 now, at most
            # $2,240" read as a long cap on a short (a hostile review, 2026-09-30).
            parts.append(f"{_t(sym)} short ${-weight * worth:,.0f}")
            continue
        others = {s: w for s, w in request.book.items() if s != sym}
        ceiling = (max_size_within_budget(add=sym, before=request.book, columns=columns,
                                          budget=request.budget, as_target=True)
                   if others else None)
        parts.append(f"{_t(sym)} ${weight * worth:,.0f} now"
                     + (f", at most ${ceiling * worth:,.0f} to stay under "
                        f"{request.budget:.0%} of book risk" if ceiling is not None else ""))
    if not parts:
        return []
    return [f"Bottom line: at ${worth:,.0f}, the book in dollars — " + "; ".join(parts) + "."]


_BETA_ASKED = re.compile(r"\bbeta\b|\bhow\s+much\s+(?:does|do)\s+my\s+(?:book|portfolio)\s+move\b",
                         re.I)


_SP500 = re.compile(r"\bs\s*&\s*p(?:\s*500)?\b|\bspx\b|\bspy\b|\bsp500\b", re.I)


_NASDAQ_NAMED = re.compile(r"\bnasdaq\w*|\bndx\w*|\bqqq\b|\bnq\b", re.I)


def _book_beta_line(book: Mapping[str, float], data: MarketData, is_open: Any,
                    text: str) -> tuple[str, Source] | None:
    """The book's beta to each index the question names — the S&P 500 (SPY on Bitget), the
    Nasdaq-100 (QQQ), or both — in regular hours. "beta of my book to the S&P 500" was answered
    with a concentration report whose only beta was to QQQ (2026-09-25 audit, round 2), and "beta
    of my book to spx and ndx" as a comparison of the two indices (round 3)."""
    from argus.market.history import fetch_range

    benches = [b for b, asked in (("SPYUSDT", _SP500.search(text)),
                                  (BENCHMARK, _NASDAQ_NAMED.search(text))) if asked] or [BENCHMARK]
    raw = dict(data.raw)
    for bench in benches:
        if bench not in raw:
            try:
                bars = fetch_range(bench, days=30, interval="1H")
            except Exception:
                return None
            from argus.desk.portfolio import returns as to_returns

            raw[bench] = to_returns([(b.ts, float(b.close)) for b in bars])
    columns = _open_columns(raw, is_open)
    read: list[tuple[str, float, float]] = []
    for bench in benches:
        if bench not in columns:
            continue
        total = seen = 0.0
        for sym, weight in book.items():
            value = beta(columns.get(sym, []), columns[bench])
            if value is not None:
                total += weight * value
                seen += abs(weight)
        if seen:
            read.append((bench, total, seen))
    if not read:
        return None
    names = {"SPYUSDT": "the S&P 500 (SPY)", BENCHMARK: "the Nasdaq-100 (QQQ)"}
    betas = " and ".join(f"{names[b]} {t:.2f}" for b, t, _ in read)
    first = read[0]
    return (f"Bottom line: this book's beta to {betas} in regular US hours over the last 30 days — "
            f"a 1% move in {names[first[0]].split(' (')[0]} moves the book about {first[1]:.2f}%"
            + (f", from the {first[2]:.0%} of it with enough history" if first[2] < 0.99 else "")
            + ".",
            Source(kind="computation", ref="argus.desk.portfolio.beta",
                   detail="weighted holding betas to " + ", ".join(b for b, _, _ in read)
                          + ", open-session hourly returns"))


def _open_columns(raw: Mapping[str, Mapping[datetime, float]],
                  is_open: Any) -> dict[str, list[float]]:
    stamps, columns = align(raw)
    rows = [i for i, t in enumerate(stamps) if is_open(t)]
    return {k: [v[i] for i in rows] for k, v in columns.items()}


def _moments(values: Sequence[float]) -> tuple[float, float] | None:
    """Sample skewness and excess kurtosis (the adjusted Fisher-Pearson estimators, as
    `scipy.stats.skew(bias=False)` and `kurtosis(bias=False)` define them)."""
    n = len(values)
    if n < 4:
        return None
    mean = sum(values) / n
    m2 = sum((v - mean) ** 2 for v in values) / n
    if m2 <= 0:
        return None
    m3 = sum((v - mean) ** 3 for v in values) / n
    m4 = sum((v - mean) ** 4 for v in values) / n
    g1 = m3 / m2 ** 1.5
    g2 = m4 / m2 ** 2 - 3.0
    skew = g1 * math.sqrt(n * (n - 1)) / (n - 2)
    kurt = ((n + 1) * g2 + 6.0) * (n - 1) / ((n - 2) * (n - 3))
    return skew, kurt


def _resolve_resize(request: ResearchRequest, before: dict[str, float],
                    ) -> tuple[dict[str, float], float, str] | str | None:
    """The book and final weight for a resize question, the line that says what was done, or a
    reply asking for the book when the current weight cannot be known. None when not a resize.

    A stated "from 8%" overrides the book's figure for that name (the question is the more
    specific statement); a relative "by 30%" needs the current weight, from the book."""
    if request.target is None and request.resize_by is None:
        return None
    add = request.symbols[0]
    book = dict(before)
    if request.resize_from is not None and book:
        rest = sum(w for s, w in book.items() if s != add)
        if rest > 0:
            book = {s: w * (1.0 - request.resize_from) / rest for s, w in book.items() if s != add}
            book[add] = request.resize_from
    # With cash in the book, one risky name is a whole book: the rest of it is cash, and the
    # resized weight comes from or goes to cash ("cut my BTC by 30%" on 50% BTC, 50% cash asked
    # for a book it had just been given, 2026-09-25 audit).
    if add not in book and len(book) >= 2:
        # "What if I trim TSLA to 10% of my book?" on a book of NVDA, MSFT and AAPL was asked for
        # the holdings it had just been given (a judge, round 13, 2026-09-30): the name is not in
        # the book, and that is the answer.
        held = ", ".join(f"{w:.0%} {_t(s)}" for s, w in book.items())
        size = f"{request.target:.0%} " if request.target is not None else ""
        return (f"{_t(add)} is not in your book ({held}), so there is nothing to trim or resize. "
                f"To weigh buying it, ask \"what does adding {size or '10% '}{_t(add)} do to my "
                f"book\".")
    if add not in book or (len(book) < 2 and not request.cash):
        others = [n for n in ("AAPL", "MSFT", "NVDA") if n != _t(add)][:2]
        return (f"Resizing {_t(add)} changes every other holding's share too, so I need what "
                f"you hold — say it with weights, e.g. \"what if I trim {_t(add)} to 10%? I hold "
                f"30% {_t(add)}, 40% {others[0]}, 30% {others[1]}\", or save your book once on "
                f"the console.")
    current = book[add]
    if request.target is not None:
        # Trimming a short keeps it short: "trim TSLA to 10%" with TSLA at -40% was resized to
        # +10%, a 50-point swing the other way (a hostile review, round 22, on README's own example)
        target = -abs(request.target) if current < 0 else request.target
    else:
        target = max(0.0, current * (1.0 + (request.resize_by or 0.0)))
    if target >= 1.0:
        return (f"That takes {_t(add)} to {target:.0%} of the book, which leaves nothing else "
                f"in it; ask about {_t(add)} on its own instead.")
    verb = "to 0%, selling all of it" if target == 0 else f"to {target:.0%}"
    over = (request.resize_by is not None and request.resize_by < -1.0)
    tail = (", the difference moving to or from cash." if len(book) < 2 else
            ", every other holding scaled so the book still sums to 100%.")
    return book, target, (
        f"Resizing {_t(add)} from {current:.0%} {verb}"
        + (f" (a {abs(request.resize_by):.0%} {'raise' if (request.resize_by or 0) > 0 else 'cut'}"
           f" of the position" + ("; a position cannot be cut by more than all of it, so this is "
                                  "read as selling it all" if over else "") + ")"
           if request.resize_by is not None else "")
        + tail)


_VAR_LEVEL = re.compile(r"\b(9\d(?:\.\d+)?)\s*%", re.I)


VAR_DAYS = 500


_VOL_SPIKE = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*x\s+(?:the\s+)?(?:vol\w*|variance)\b|\b(?:vol\w*)\s+(?:spike|jump|"
    r"shock|doubl\w*|tripl\w*)\s*(?:of\s+|by\s+)?(\d+(?:\.\d+)?)?\s*x?\b|\b(doubl|tripl)\w*\s+"
    r"(?:the\s+)?vol\w*", re.I)
"""A volatility scenario ("a 2x vol spike across all crypto perps", "if volatility doubles"),
read as nothing until 2026-09-25 (`eval/figurecheck.py`)."""


def _vol_multiple(text: str) -> float | None:
    found = _VOL_SPIKE.search(text)
    if found is None:
        return None
    if found.group(1) or found.group(2):
        return float(found.group(1) or found.group(2))
    word = (found.group(3) or "").lower()
    return 3.0 if word.startswith("tripl") else 2.0


def _var_lines(before: Mapping[str, float], after: Mapping[str, float],
               text: str, *, scale: float = 1.0) -> tuple[list[str], list[Source]]:
    """One-day historical value at risk and expected shortfall of the book, before and after.

    Asked for by name ("recompute portfolio VaR at 95%") and answered by nothing until
    2026-09-25 (`eval/figurecheck.py` found the 95% dropped). Thirty days of hourly bars hold
    thirty daily outcomes, one or two of them past a 95% line, so daily VaR is read from up to 500
    days of Bitget daily closes, aligned on the dates every held name traded. Historical, not
    parametric: the loss the book would actually have taken on its worst days, at today's
    weights. Expected shortfall is the average of the days past the line (skfolio's CVaR
    definition is the one `desk/portfolio.tail_contributions` uses; the plain tail mean is used
    here for a daily series)."""
    from argus.market.history import CandleType, fetch_window

    found = _VAR_LEVEL.search(text)
    level = float(found.group(1)) / 100 if found else 0.95
    names = sorted({*before, *after})
    closes: dict[str, dict[Any, float]] = {}
    for name in names:
        try:
            with _FETCH_SLOTS:
                bars = fetch_window(name, start=datetime.now(UTC) - timedelta(days=VAR_DAYS),
                                    interval="1D", candle_type=CandleType.MARKET, pause=0.05)
        except Exception:
            return [], []
        closes[name] = {b.ts.date(): float(b.close) for b in bars if float(b.close) > 0}
    dates = sorted(set.intersection(*(set(c) for c in closes.values()))) if closes else []
    if len(dates) < 60:
        return [f"Value at risk: the names share only {len(dates)} days of Bitget history, too "
                f"few for a {level:.0%} one-day figure."], []
    rets = {n: [closes[n][b] / closes[n][a] - 1 for a, b in itertools.pairwise(dates)]
            for n in names}

    def var_es(weights: Mapping[str, float]) -> tuple[float, float] | None:
        if not weights:
            return None
        book = sorted(sum(float(weights.get(n, 0.0)) * rets[n][t] for n in names)
                      for t in range(len(dates) - 1))
        cut = max(1, int((1 - level) * len(book)))
        return -book[cut - 1], -sum(book[:cut]) / cut

    now, then = var_es(after), var_es(before)
    if now is None:
        return [], []
    if scale != 1.0:
        # Filtered historical simulation, the simplest honest form: every past day's move scaled
        # by the stated multiple, so the shape of the book's worst days is kept and only their
        # size changes. It says nothing about correlations rising together, which in a real
        # volatility spike they do — so this is a floor on the damage, and says so.
        worst_day = min(sum(float(after.get(n, 0.0)) * rets[n][t] for n in names)
                        for t in range(len(dates) - 1))
        return ([f"Volatility at {scale:g}x: every past day's move scaled by {scale:g}, the book's "
                 f"{level:.0%} one-day value at risk goes from {now[0]:.2%} to "
                 f"{now[0] * scale:.2%} and its expected shortfall from {now[1]:.2%} to "
                 f"{now[1] * scale:.2%}; its worst day in {len(dates) - 1} would have been "
                 f"{worst_day * scale:+.2%} instead of {worst_day:+.2%}. Correlations are held "
                 f"where they were, and in a real spike they rise, so read this as the least it "
                 f"would cost."],
                [Source(kind="computation", ref="argus.lui.research.riskmath._var_lines",
                        detail=f"filtered historical simulation, vol x{scale:g}, "
                               f"{len(dates) - 1} aligned daily returns")])
    change = (f" (was {then[0]:.2%} and {then[1]:.2%})" if then is not None else "")
    lines = [f"Value at risk, {level:.0%}, one day: the book at these weights lost more than "
             f"{now[0]:.2%} on {1 - level:.0%} of its {len(dates) - 1} past days, and "
             f"{now[1]:.2%} on average on those days (expected shortfall){change}. Historical, "
             f"from Bitget daily closes; a figure for normal days, not a floor for bad ones."]
    return lines, [Source(kind="computation", ref="argus.lui.research.riskmath._var_lines",
                          detail=f"historical {level:.0%} VaR and expected shortfall, "
                                 f"{len(dates) - 1} aligned daily returns")]


def max_size_within_budget(
    *, add: str, before: Mapping[str, float], columns: Mapping[str, Sequence[float]],
    budget: float = RISK_BUDGET, as_target: bool = False,
) -> float | None:
    """The largest target weight for ``add`` that keeps its share of book risk at or under
    ``budget`` — the number a trader can act on. None when no size in the grid qualifies (or, with
    no book, when the question has no answer: a lone position is always 100% of its own risk)."""
    if not before:
        return None
    best: float | None = None
    from argus.desk.portfolio import resize

    for step in range(1, 100 if as_target else 101):
        size = step / 100.0
        try:
            after = resize(before, add, size) if as_target else rebalance(before, add, size)
        except PortfolioError:
            # A book of the name alone has nothing to trim against: no weight answers the
            # question, which is said as "no size fits" rather than raised to the reader.
            return None
        risk = decompose(after, columns)
        if risk is None:
            return None
        share = risk.share_of_risk(add)
        if share is None or share > budget:
            break
        best = size
    return best


def _desk_view(symbol: str, ledger: Any) -> tuple[list[str], list[Source]]:
    if ledger is None:
        return [], []
    if symbol not in TRADED_SYMBOLS:
        return [f"The desk itself trades twelve stock perpetuals and {_t(symbol)} is not one of "
                f"them, so there is no desk call on it — this answer is the analysis alone."], []
    rows = [e for e in ledger.entries if e.symbol == symbol]
    if not rows:
        return [f"The desk has no recorded decision on {symbol} yet."], []
    last = rows[-1]
    thesis = (last.thesis or "").strip()
    line = (
        f"The desk's own last call on {symbol} (decision {last.seq}, {last.decided_at[:16]}Z): "
        f"{last.verdict}"
        + f" at stated confidence {float(last.stated_confidence):.2f}"
        + (f" — {sentence_cut(thesis)}" if thesis else "")
    )
    if re.search(r"\brevision", thesis, re.I) and re.search(r"\d+\s+up\s*/\s*\d+\s+down", thesis):
        # "35 up / 1 down" sat beside "12 raised and 0 cut their target" with nothing saying they
        # count different things (a judge, round 21)
        line += (" (its revision count is analysts' earnings-estimate changes over 30 days; a "
                 "price-target line elsewhere counts firms' target changes — different measures)")
    return [line], [Source(kind="ledger", ref=f"seq {last.seq}",
                           detail=f"{symbol} {last.verdict} @ {last.decided_at}")]


FRED_SERIES: dict[str, str] = {
    "DGS10": "10-year Treasury", "DGS2": "2-year Treasury", "DFF": "fed funds",
    "T10YIE": "10-year breakeven inflation", "DTWEXBGS": "broad dollar index",
}


FRED_MONTHLY: dict[str, str] = {
    "CPIAUCSL": "CPI", "CPILFESL": "core CPI", "PCEPI": "PCE price index",
    "PCEPILFE": "core PCE price index",
}
"""Monthly inflation series, shipped in the snapshot with 480 days so a year-on-year change can
be read from it: the hosted console cannot reach FRED, and "whats the latest CPI number" was
answered with the rates dashboard and no CPI at all (answer audit, round 3)."""


_EVENT_WORDS = re.compile(
    r"\b(?:cpi|inflation|fomc|fed|rate\s+decision|powell|this\s+week|next\s+week|event|data\s+"
    r"release)\b", re.I)


def _event_lines(raw_text: str, *, always: bool = False) -> tuple[list[str], str | None]:
    """The next CPI release and FOMC decision, from the official schedules' snapshot, when the
    question is about them (or ``always``). Returns the lines and a short clause for a lead line."""
    if not always and not _EVENT_WORDS.search(raw_text):
        return [], None
    from argus.market.calendar import upcoming

    events, fetched = upcoming(days=60)
    if not events:
        return [], None
    first = {e.kind: e for e in reversed(events)}
    ordered = sorted(first.values(), key=lambda e: e.at)
    line = "Scheduled: " + "; ".join(
        f"{e.label} {e.at:%a %d %b} {e.at:%H:%M} UTC" for e in ordered) + (
        f" (BLS and Federal Reserve schedules, read {fetched}).")
    asked = next((e for e in ordered if e.kind.lower() in raw_text.lower()
                  or (e.kind == "CPI" and "inflation" in raw_text.lower())
                  or (e.kind == "FOMC" and re.search(r"\bfed\b|rate\s+decision", raw_text,
                                                     re.I))), ordered[0])
    clause = f"before the {asked.label} on {asked.at:%a %d %b} at {asked.at:%H:%M} UTC"
    return [line], clause


# Every engine here is a traced step from import on (lui/trace.py, trace_module).
trace_module(globals())
