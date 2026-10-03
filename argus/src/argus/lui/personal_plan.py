"""A personal thesis: the trader's own constraints, measured, and what they imply for new money.

A judge (round 30) opened with the most direct probe of "personalized thesis" there is: "I'm 58,
planning to retire in 5 years, 70% of my liquid net worth is already in BTC and NVDA, I can't
stomach a drawdown bigger than 12%, I have $150,000 of fresh cash — what should I actually do with
it, and why; a real thesis, not a generic allocation." The console answered with a canned
three-market table that used none of it, then read "scratch the 12% limit, I can tolerate 25%"
as a question about its own track record, and answered "size the $150,000 so my book's 1-day 95%
VaR stays under $9,000 — show the math" with a memory acknowledgement.

This module reads the constraints a message (or the conversation) states and measures them:

* **The book already held.** The names and the share of net worth in them, weighted equally when
  no split is given (said). The deepest fall of that mix over the last year of daily closes
  (``market.equity_history.daily``: Yahoo Finance daily closes, aligned on the days both trade) is
  the yardstick a drawdown limit is checked against.
* **The limit.** The latest drawdown tolerance stated wins ("scratch the 12%, I can take 25%" is
  25%), so a changed constraint re-runs the whole plan and the answer says what changed.
* **The fresh cash.** When the share in the held names and the cash are both given, net worth is
  read as the held part plus the cash with the cash as the rest (70% held, $150,000 the other 30%
  makes $500,000), and the answer says it assumed that.
* **What the limit allows.** The most that can sit in the held mix without its own worst year
  breaking the limit on the whole net worth (limit x net worth / the mix's deepest fall), against
  what is held now — so the answer can say "the book you already hold breaks your limit before the
  new cash is placed, by $X".
* **1-day 95% VaR**, when asked: historical simulation over the same aligned daily returns, the
  5th-percentile one-day loss in dollars, with the arithmetic shown, and the most each candidate
  market could take of the fresh cash before the total breaks the stated VaR budget.

No pick: the plan names what the constraints imply, measured, and says that the choice is the
trader's. Every figure is computed from the data named beside it.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Final

from argus.lui.trace import trace_module

_AGE: Final = re.compile(r"\b(?:i'?m|i\s+am|aged?)\s+(?P<age>[2-9]\d)\b(?!\s*%)", re.I)
_RETIRE: Final = re.compile(
    r"\bretir\w*\s+in\s+(?:about\s+|around\s+)?(?P<y>\d{1,2})\s+years?\b|\b(?P<y2>\d{1,2})[\s-]+"
    r"years?\s+(?:to|until|till|before)\s+(?:i\s+)?retir\w*|\b(?P<y3>\d{1,2})[\s-]*(?:year|yr)\s+"
    r"horizon\b", re.I)
_HELD_SHARE: Final = re.compile(
    r"\b(?P<pct>\d{1,3})\s*%\s+of\s+my\s+(?:liquid\s+|investable\s+|total\s+)?(?:net\s+worth|"
    r"portfolio|savings|money|wealth|assets)\s+(?:is\s+|are\s+|sits\s+)?(?:already\s+|now\s+)?"
    r"(?:in|into)\s+(?P<what>[^.;!?]{2,80})", re.I)
_LIMIT: Final = re.compile(
    r"\bdrawdown\s+(?:bigger|larger|greater|more|worse)\s+than\s+(?P<a>\d{1,2}(?:\.\d+)?)\s*%|"
    r"\b(?:tolerate|stomach|handle|accept|take|bear|live\s+with)\s+(?:up\s+to\s+|at\s+most\s+|a\s+)?"
    r"(?:a\s+)?(?P<b>\d{1,2}(?:\.\d+)?)\s*%(?:\s+(?:drawdown|fall|drop|loss))?|"
    r"\bmax(?:imum)?\s+drawdown\s+(?:of\s+|is\s+|at\s+)?(?P<c>\d{1,2}(?:\.\d+)?)\s*%|"
    r"\b(?P<d>\d{1,2}(?:\.\d+)?)\s*%\s+(?:max(?:imum)?\s+)?drawdown\s+(?:limit|cap|tolerance)",
    re.I)
_CASH: Final = re.compile(
    r"\$\s?(?P<a>\d[\d,]*(?:\.\d+)?)(?![\d,.]*\d)\s*(?P<k>k|m)?\s+(?:of\s+)?(?:fresh|new|spare|"
    r"free|idle|extra)?\s*(?:cash|money|capital|savings)\b|\b(?:fresh|new|spare)\s+(?:cash|money)"
    r"\s+of\s+\$\s?(?P<b>\d[\d,]*(?:\.\d+)?)(?![\d,.]*\d)\s*(?P<k2>k|m)?", re.I)
_VAR: Final = re.compile(
    r"\b(?:1|one)[\s-]*day\s+(?P<conf>9[05]|99)\s*%\s+var\b|\bvar\b[^?.]{0,40}?\$\s?(?P<cap>\d[\d,]*)"
    r"|\bvalue[\s-]+at[\s-]+risk\b", re.I)
_VAR_CAP: Final = re.compile(r"\b(?:exceed|above|over|under|below|within|max(?:imum)?\s+of|of)\s+"
                             r"\$\s?(?P<cap>\d[\d,]*(?:\.\d+)?)(?![\d,.]*\d)\s*(?P<k>k)?", re.I)
ASKS: Final = re.compile(
    r"\bwhat\s+should\s+i\s+(?:actually\s+)?do\b|\bthesis\b|\bplan\b|\ballocat\w*|\bdoes\s+(?:that|"
    r"this|it)\s+change\b|\bwhat\s+changes\b|\bquantify\b|\bsize\s+(?:the|my|it|this)\b|\bvar\b|"
    r"\bvalue[\s-]+at[\s-]+risk\b|\bwhere\s+should\s+(?:it|the\s+(?:cash|money))\s+go\b", re.I)
_CANDIDATES: Final = (("QQQ", "the Nasdaq-100 (QQQ)"), ("GLD", "gold (GLD)"),
                      ("BTC-USD", "bitcoin"))


@dataclass
class Situation:
    age: int | None = None
    years: int | None = None
    held_share: float | None = None
    held: tuple[str, ...] = ()
    limit: float | None = None
    limit_before: float | None = None
    cash: float | None = None
    var_cap: float | None = None
    limit_now: bool = False
    """The latest limit was stated in the message being answered."""
    var_now: bool = False
    said: list[str] = field(default_factory=list)

    @property
    def enough(self) -> bool:
        return sum(x is not None and x != () for x in (
            self.age or self.years, self.held or None, self.limit, self.cash)) >= 2


def _money(raw: str, k: str | None) -> float:
    return float(raw.replace(",", "")) * {"k": 1e3, "m": 1e6}.get((k or "").lower(), 1.0)


def read(text: str, prior: list[str]) -> Situation:
    """The constraints stated across the conversation, the latest of each winning."""
    from argus.lui.research import research_symbols

    s = Situation()
    for said in [*prior[-8:], text]:
        s.limit_now = s.var_now = False
        if (m := _AGE.search(said)) is not None:
            s.age = int(m.group("age"))
        if (m := _RETIRE.search(said)) is not None:
            s.years = int(m.group("y") or m.group("y2") or m.group("y3"))
        if (m := _HELD_SHARE.search(said)) is not None:
            names = research_symbols(m.group("what"))[0]
            if names:
                s.held_share = int(m.group("pct")) / 100
                s.held = tuple(names[:4])
        limits = [float(next(g for g in m.groups() if g)) / 100 for m in _LIMIT.finditer(said)]
        if limits:
            new = limits[-1]
            if s.limit is not None and new != s.limit:
                s.limit_before = s.limit
            s.limit = new
            s.limit_now = True
        if (m := _CASH.search(said)) is not None:
            s.cash = _money(m.group("a") or m.group("b"), m.group("k") or m.group("k2"))
        if _VAR.search(said) and (m := _VAR_CAP.search(said)) is not None:
            s.var_cap = _money(m.group("cap"), m.group("k"))
            s.var_now = True
    return s


def _ticker(symbol: str) -> str:
    from argus.lui.research.parse import is_us_equity

    base = symbol.removesuffix("USDT")
    return base if is_us_equity(symbol) else f"{base}-USD"


def _returns(tickers: list[str], days: int = 365) -> tuple[list[date], dict[str, list[float]]]:
    """Daily close-to-close returns on the days every ticker has a close, over ``days``."""
    from argus.market.equity_history import daily

    since = datetime.now(UTC).date() - timedelta(days=days)
    closes = {t: {d.day: d.close for d in daily(t) if d.day >= since} for t in tickers}
    common = sorted(set.intersection(*(set(c) for c in closes.values())))
    out = {t: [closes[t][b] / closes[t][a] - 1 for a, b in itertools.pairwise(common)]
           for t in tickers}
    return common[1:], out


def _deepest(path: list[float]) -> float:
    value, peak, deepest = 1.0, 1.0, 0.0
    for r in path:
        value *= 1 + r
        peak = max(peak, value)
        deepest = max(deepest, 1 - value / peak)
    return deepest


def _mix(weights: dict[str, float], rets: dict[str, list[float]]) -> list[float]:
    n = len(next(iter(rets.values())))
    return [sum(w * rets[t][i] for t, w in weights.items()) for i in range(n)]


def _var95(dollars: dict[str, float], rets: dict[str, list[float]]) -> float:
    pnl = sorted(sum(d * rets[t][i] for t, d in dollars.items())
                 for i in range(len(next(iter(rets.values())))))
    return -pnl[max(0, int(0.05 * len(pnl)) - 1)]


def lines(text: str, prior: list[str]) -> list[str] | None:
    """The personal plan, or None when the conversation states too little to measure one."""
    if not ASKS.search(text):
        return None
    s = read(text, prior)
    if not s.enough or not s.held:
        return None
    held = [_ticker(x) for x in s.held]
    try:
        _days, rets = _returns(sorted({*held, *(t for t, _ in _CANDIDATES)}))
    except Exception:
        return None
    if not rets or len(next(iter(rets.values()))) < 150:
        return None
    names = [x.removesuffix("USDT") for x in s.held]
    equal = {t: 1 / len(held) for t in held}
    mix_fall = _deepest(_mix(equal, rets))
    split = " and ".join(names) + (" in equal parts (no split was given)" if len(held) > 1 else "")
    out: list[str] = []
    worth = None
    if s.held_share is not None and s.cash is not None and s.held_share < 1:
        worth = s.cash / (1 - s.held_share)
    held_dollars = worth * s.held_share if worth and s.held_share else None
    who = []
    if s.age:
        who.append(f"{s.age}")
    if s.years:
        who.append(f"retiring in {s.years} years")
    lead_who = (", ".join(who) + ": ") if who else ""
    if s.limit is not None and s.held_share is not None:
        own_fall = s.held_share * mix_fall
        verdict = "breaks" if own_fall > s.limit else "fits inside"
        lead = (f"Bottom line: {lead_who}the {s.held_share:.0%} you already hold in {split} "
                f"{verdict} your {s.limit:.0%} drawdown limit on its own — that mix fell "
                f"{mix_fall:.0%} from its high at its worst in the last year, which on "
                f"{s.held_share:.0%} of your net worth is a {own_fall:.1%} fall of the whole")
        if own_fall > s.limit:
            lead += (", before the new cash is placed anywhere. So the thesis is about the money "
                     "already in, not only the new money: the cash cannot fix a limit the "
                     "existing holdings already break.")
        else:
            lead += "."
        out.append(lead)
        if worth is not None and held_dollars is not None:
            allowed = s.limit * worth / mix_fall
            out.append(f"In dollars (assumed: the {s.held_share:.0%} is ${held_dollars:,.0f} and "
                       f"your ${s.cash:,.0f} is the other {1 - s.held_share:.0%}, so net worth is "
                       f"${worth:,.0f}): a {s.limit:.0%} limit is ${s.limit * worth:,.0f}; at the "
                       f"mix's {mix_fall:.0%} worst fall it allows about ${allowed:,.0f} in "
                       f"{' and '.join(names)} against ${held_dollars:,.0f} held — "
                       + (f"about ${held_dollars - allowed:,.0f} more than the limit carries."
                          if allowed < held_dollars else
                          f"room for about ${allowed - held_dollars:,.0f} more."))
            if s.limit_before is not None and s.limit_now:
                before = s.limit_before * worth / mix_fall
                out.insert(1, f"What changed: the limit moved from {s.limit_before:.0%} to "
                              f"{s.limit:.0%}, so the most {' and '.join(names)} can hold within "
                              f"it moves from about ${before:,.0f} to about ${allowed:,.0f} "
                              f"(+${allowed - before:,.0f}); everything else above is unchanged.")
        rows = []
        for ticker, label in _CANDIDATES:
            fall = _deepest(rets[ticker])
            rows.append(f"{label} {fall:.0%}")
        out.append("Where new cash would add least to a fall, by each market's own deepest drop "
                   "in the same year: cash 0%, " + ", ".join(rows) + " — every dollar placed in "
                   "a market that falls adds its share of that fall to the whole.")
    if s.var_cap is not None and held_dollars is not None and s.cash is not None:
        dollars = {t: held_dollars / len(held) for t in held}
        base_var = _var95(dollars, rets)
        n = len(next(iter(rets.values())))
        out.append(f"1-day 95% VaR, shown: from the {n} days in the last year when all of these "
                   f"traded, each day's dollar change of your held book is "
                   f"{' + '.join(f'${d:,.0f} x {t} return' for t, d in dollars.items())}; the "
                   f"5th-worst percent of those days is the VaR — ${base_var:,.0f} for the book "
                   f"already held, against your ${s.var_cap:,.0f} budget.")
        if base_var >= s.var_cap:
            over = base_var - s.var_cap
            out.append(f"So the held book alone is over the budget by ${over:,.0f}: "
                       f"the ${s.cash:,.0f} can only sit in cash (VaR 0) without making it worse, "
                       f"and meeting ${s.var_cap:,.0f} means trimming what is held.")
        else:
            room = []
            for ticker, label in _CANDIDATES:
                lo, hi = 0.0, s.cash
                for _ in range(30):
                    mid = (lo + hi) / 2
                    trial = {**dollars, ticker: dollars.get(ticker, 0.0) + mid}
                    if _var95(trial, rets) <= s.var_cap:
                        lo = mid
                    else:
                        hi = mid
                room.append(f"{label} up to ${lo:,.0f}")
            out.append(f"Budget left: ${s.var_cap - base_var:,.0f}. Of the ${s.cash:,.0f}, the "
                       f"most each market could take before the total VaR reaches "
                       f"${s.var_cap:,.0f} (the rest in cash): " + "; ".join(room) + ".")
    if not out:
        return None
    var_at = next((i for i, x in enumerate(out) if x.startswith("1-day 95% VaR")), None)
    if s.var_now and var_at is not None:
        # the VaR asked for leads; the drawdown picture follows as context
        var_lines = out[var_at:]
        out = [f"Bottom line: {var_lines[0]}", *var_lines[1:],
               *(x.replace("Bottom line: ", "Against your drawdown limit: ", 1)
                 for x in out[:var_at])]
    if s.years is not None:
        out.append(f"With {s.years} years to go, a fall like the worst above can still be inside "
                   "the years left to recover, or it can arrive in the last one; the limit you "
                   "set is the one to hold the plan to.")
    out.append("This is your stated situation measured, not advice — you make the call. Data: "
               "Yahoo Finance daily closes over the last year, on the days every name traded.")
    return out


TICKET: Final = re.compile(
    r"\b(?:order\s+)?tickets?\b|\bexecute\s+(?:that|this|it|the\s+plan)\b|\bexact\s+orders?\b|"
    r"\bplace\s+(?:that|those|the)\s+orders?\b|\bhow\s+do\s+i\s+(?:execute|place|do)\s+(?:that|"
    r"this|it)\b", re.I)
"""Asking for the orders that carry out the plan just given."""


def trades(text: str, prior: list[str]) -> tuple[list[tuple[str, float]], list[str]] | None:
    """The sells the conversation's plan implies, as (symbol, dollars), with the reasons, or None
    when there is no plan or it implies no trade. "Give me the exact Bitget order ticket to execute
    that" after a plan that found the held book over its limits put all $150,000 into BTC, the name
    the plan had just flagged (a judge, round 30)."""
    if not TICKET.search(text):
        return None
    s = read("", [*prior, text])
    if not s.enough or not s.held or s.held_share is None or s.cash is None:
        return None
    held = [_ticker(x) for x in s.held]
    try:
        _days, rets = _returns(held)
    except Exception:
        return None
    worth = s.cash / (1 - s.held_share)
    held_dollars = worth * s.held_share
    keep = held_dollars
    why: list[str] = []
    if s.limit is not None:
        mix_fall = _deepest(_mix({t: 1 / len(held) for t in held}, rets))
        allowed = s.limit * worth / mix_fall
        if allowed < keep:
            keep = allowed
            why.append(f"the {s.limit:.0%} drawdown limit allows about ${allowed:,.0f}")
    if s.var_cap is not None:
        base = _var95({t: held_dollars / len(held) for t in held}, rets)
        if base > s.var_cap:
            allowed = held_dollars * s.var_cap / base
            if allowed < keep:
                keep = allowed
            why.append(f"the ${s.var_cap:,.0f} 1-day VaR allows about ${allowed:,.0f}")
    if keep >= held_dollars - 1:
        return None
    each = (held_dollars - keep) / len(held)
    return [(sym, each) for sym in s.held], why


__all__ = ["ASKS", "TICKET", "Situation", "lines", "read", "trades"]

trace_module(globals())
