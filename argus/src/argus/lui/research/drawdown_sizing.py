"""How much of a name a book can add and stay inside a stated maximum drawdown — sized against the
book's own worst historical fall, not against a default risk budget.

Two round-41 findings were the same miss on two doors:

- **M8** (Portuguese, the judge's q21): "60% in tech, a 10% maximum drawdown limit — how much gold
  should I add?" was sized on a $10,000 worked example against a 2% one-day rule; the 10% limit
  and the 60% weight were filed as memory and never used.
- **M13** (MCP ``argus_research_task``): "60% BTC, 40% MSFT, 15% max drawdown limit, add 20% COIN"
  came back "Add smaller: at most 12%" from a default 25% risk budget; the 15% limit was ignored.

**What is measured.** Daily closes over the last five years (`rule_test.daily_closes`: Bitget's
for a coin, Yahoo's split-adjusted for a stock, gold futures for gold), on the days every name has
a close; the book rebalanced to its weights each day; its maximum drawdown, peak to trough. "Tech
stocks" with no names is read as the Nasdaq-100 (QQQ), and the answer says so. A book whose
weights sum below 100% holds the rest in cash, and an add is paid for from that cash; a book that
is fully invested pays for the add by trimming every holding in proportion.

**What is answered.** The book's worst fall as it stands; then the largest weight of the new name
that keeps the worst fall inside the limit — scanned in 1% steps — or, when the book already
breaches the limit, that it does, by how much, and what the add does to it. The asked weight, when
one is stated, is tested as asked. A drawdown that happened is not the worst that can happen; the
answer says which years produced it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Final

_LIMIT: Final = re.compile(
    r"\b(?P<a>\d{1,2}(?:\.\d+)?)\s*%\s+(?:max(?:imum)?\s+)?(?:drawdown|loss)\s+(?:limit|tolerance|"
    r"cap)\b|\b(?P<b>\d{1,2}(?:\.\d+)?)\s*%\s+max(?:imum)?\s+(?:drawdown|loss)\b|\bmax(?:imum)?\s+"
    r"(?:drawdown|loss)\s+(?:limit\s+|tolerance\s+)?(?:is\s+|of\s+)?(?P<c>\d{1,2}(?:\.\d+)?)\s*%|"
    r"\b(?:drawdown|loss)\s+limit\s+(?:is\s+|of\s+)?(?P<d>\d{1,2}(?:\.\d+)?)\s*%", re.I)
_ADD: Final = re.compile(
    r"\b(?:add(?:ing)?|buy(?:ing)?|put(?:ting)?|allocat\w*)\s+(?:(?:a|another)\s+)?"
    r"(?:(?P<w>\d{1,2}(?:\.\d+)?)\s*%\s+(?:of\s+\w+\s+)?(?:in(?:to)?\s+|of\s+)?)?(?P<name>[A-Za-z][\w&.-]*)"
    r"|\bhow\s+much\s+(?P<name2>[A-Za-z][\w&.-]*)\s+(?:should|can|could|do)\s+i\s+(?:add|buy|"
    r"hold|own|put)", re.I)
_HOLD: Final = re.compile(
    r"(?P<w>\d{1,3}(?:\.\d+)?)\s*%\s+(?:of\s+(?:my\s+)?(?:portfolio|book|money|account)\s+)?"
    r"(?:(?:is\s+)?in\s+)?(?P<name>[A-Za-z][\w&.-]*(?:\s+stocks?)?)", re.I)
_TECH: Final = re.compile(r"^(?:tech|technology|tech\s+stocks?|big\s+tech)$", re.I)
_NOT_NAMES: Final = frozenset({"max", "maximum", "drawdown", "loss", "of", "my", "the", "in", "a",
                               "limit", "cash", "more", "it", "some", "position", "stocks"})


@dataclass(frozen=True)
class Asked:
    book: dict[str, float]
    add: str
    weight: float | None
    limit: float
    proxies: tuple[str, ...]


def _symbol(word: str) -> tuple[str | None, str | None]:
    from argus.lui.research import research_symbols

    word = word.strip()
    if _TECH.match(word):
        return "QQQUSDT", f"\"{word}\" is read as the Nasdaq-100 (QQQ)"
    if word.lower() in _NOT_NAMES:
        return None, None
    found = research_symbols(word)[0]
    return (found[0], None) if found else (None, None)


def read(text: str) -> Asked | None:
    """The book, the add, its weight and the limit, or None when any of the four is missing."""
    limit = _LIMIT.search(text)
    add = _ADD.search(text)
    if limit is None or add is None:
        return None
    add_symbol, add_note = _symbol(add.group("name") or add.group("name2"))
    if add_symbol is None:
        return None
    book: dict[str, float] = {}
    proxies = [add_note] if add_note else []
    span = add.span()
    for m in _HOLD.finditer(text):
        if span[0] <= m.start() < span[1] or re.match(r"\s*(?:max|drawdown|loss)",
                                                      text[m.end("w") + 1:], re.I):
            continue
        symbol, note = _symbol(m.group("name"))
        if symbol is None or symbol == add_symbol:
            continue
        book[symbol] = book.get(symbol, 0.0) + float(m.group("w")) / 100
        if note:
            proxies.append(note)
    if not book or sum(book.values()) > 1.0001:
        return None
    value = next(v for k, v in limit.groupdict().items() if v)
    return Asked(book=book, add=add_symbol,
                 weight=float(add.group("w")) / 100 if add.group("w") else None,
                 limit=float(value) / 100, proxies=tuple(proxies))


def _aligned(symbols: list[str]) -> tuple[list[datetime], dict[str, list[float]], list[str]]:
    from argus.lui.research.rule_test import daily_closes

    series, said = {}, []
    for symbol in symbols:
        stamps, closes, where = daily_closes(symbol)
        series[symbol] = {t.date(): c for t, c in zip(stamps, closes, strict=True)}
        said.append(f"{symbol.removesuffix('USDT')}: {where}")
    days = sorted(set.intersection(*(set(s) for s in series.values())))
    return ([datetime(d.year, d.month, d.day) for d in days],
            {s: [series[s][d] for d in days] for s in symbols}, said)


def max_drawdown(weights: dict[str, float], closes: dict[str, list[float]]
                 ) -> tuple[float, int, int]:
    """(worst fall, index of the peak, index of the trough) of the book rebalanced daily."""
    n = len(next(iter(closes.values())))
    value, peak, peak_at = 1.0, 1.0, 0
    worst, worst_peak, worst_at = 0.0, 0, 0
    for i in range(1, n):
        value *= 1 + sum(w * (closes[s][i] / closes[s][i - 1] - 1) for s, w in weights.items())
        if value > peak:
            peak, peak_at = value, i
        fall = value / peak - 1
        if fall < worst:
            worst, worst_peak, worst_at = fall, peak_at, i
    return worst, worst_peak, worst_at


def _with(book: dict[str, float], add: str, weight: float) -> dict[str, float]:
    cash = max(0.0, 1.0 - sum(book.values()))
    out = dict(book)
    if weight <= cash + 1e-9:
        out[add] = out.get(add, 0.0) + weight
        return out
    scale = (1.0 - weight) / sum(book.values())
    out = {s: w * scale for s, w in book.items()}
    out[add] = out.get(add, 0.0) + weight
    return out


def _name(symbol: str) -> str:
    return {"XAUUSDT": "gold", "QQQUSDT": "the Nasdaq-100 (QQQ)"}.get(
        symbol, symbol.removesuffix("USDT"))


def call(text: str) -> str | None:
    """The one-line sizing call for a research task's verdict, from the same measurement as
    :func:`lines`: add, add smaller, or do not add, against the stated drawdown limit."""
    asked = read(text)
    if asked is None:
        return None
    try:
        _days, closes, _said = _aligned([*asked.book, asked.add])
    except Exception:
        return None
    base = max_drawdown(asked.book, closes)[0]
    inside = [w / 100 for w in range(0, 51)
              if max_drawdown(_with(asked.book, asked.add, w / 100), closes)[0] >= -asked.limit]
    top = max(inside) if inside else None
    if top is None or top == 0:
        return (f"Do not add: the book is already past your {asked.limit:.0%} drawdown limit"
                if base < -asked.limit else
                f"Do not add: any {_name(asked.add)} takes the book past your "
                f"{asked.limit:.0%} limit")
    if asked.weight is not None and asked.weight > top + 1e-9:
        return f"Add smaller: at most {top:.0%} inside your {asked.limit:.0%} drawdown limit"
    return f"Add, at most {top:.0%} inside your {asked.limit:.0%} drawdown limit"


def lines(text: str) -> list[str] | None:
    """The drawdown-limited size, or None when the question states no book, add and limit."""
    asked = read(text)
    if asked is None:
        return None
    symbols = [*asked.book, asked.add]
    try:
        days, closes, said = _aligned(symbols)
    except Exception:
        return ["Bottom line: the daily history for that book could not be read just now, so the "
                "add cannot be sized against your drawdown limit; ask again in a minute."]
    if len(days) < 250:
        return ["Bottom line: the names in that book share under a year of daily history here — "
                "too little to size an add against a drawdown limit."]
    limit, add = asked.limit, _name(asked.add)
    base, b_peak, b_at = max_drawdown(asked.book, closes)
    holding = ", ".join(f"{w:.0%} {_name(s)}" for s, w in asked.book.items())
    cash = 1.0 - sum(asked.book.values())
    sizes = [(w / 100, max_drawdown(_with(asked.book, asked.add, w / 100), closes)[0])
             for w in range(0, 51)]
    inside = [w for w, dd in sizes if dd >= -limit]
    best_w, best_dd = max(sizes, key=lambda r: r[1])
    out: list[str] = []
    span = f"{days[0]:%b %Y} to {days[-1]:%b %Y}"
    if base < -limit:
        lead = (f"your book ({holding}" + (f", {cash:.0%} cash" if cash > 0.005 else "")
                + f") already fell {base:.1%} at its worst ({days[b_peak]:%b %Y} to "
                  f"{days[b_at]:%b %Y}), past your {limit:.0%} limit before any {add} is added")
        if inside:
            lead += (f"; adding {add} up to {max(inside):.0%} keeps the worst fall inside it, "
                     f"because {add} cushioned the falls the book took")
        elif best_dd > base + 0.005:
            lead += (f"; {add} helps — its best weight, {best_w:.0%}, cuts the worst fall to "
                     f"{best_dd:.1%} — but no weight up to 50% brings the book inside "
                     f"{limit:.0%}, so the limit needs a smaller position in what you hold")
        else:
            lead += f"; adding {add} at any weight up to 50% does not bring it inside"
        out.append(lead + ".")
        scales = [k / 100 for k in range(100, 0, -5)
                  if max_drawdown({sym: w * k / 100 for sym, w in asked.book.items()},
                                  closes)[0] >= -limit]
        if scales:
            keep = scales[0]
            out.append(f"What the limit allows: holding the same names at {keep:.0%} of their "
                       f"present size, the rest in cash, kept the worst fall inside "
                       f"{limit:.0%} — that is the size the limit sets before any add.")
    elif inside and max(inside) > 0:
        top = max(inside)
        lead = (f"with {holding}" + (f" and {cash:.0%} cash" if cash > 0.005 else "")
                + f", you can add {add} up to about {top:.0%} of the book and keep the worst fall "
                  f"over {span} inside your {limit:.0%} limit")
        out.append(lead + f"; as the book stands its worst fall was {base:.1%}.")
    else:
        out.append(f"with {holding}, any {add} at all takes the book's worst fall over {span} "
                   f"past your {limit:.0%} limit (it is {base:.1%} without it).")
    if asked.weight is not None:
        at = max_drawdown(_with(asked.book, asked.add, asked.weight), closes)[0]
        out.append(f"At the {asked.weight:.0%} you asked: the worst fall would have been {at:.1%} "
                   + ("— inside the limit." if at >= -limit else
                      f"— {(-limit - at) * 100:.1f} points past the {limit:.0%} limit."))
    recent = [i for i, d in enumerate(days) if (days[-1] - d).days <= 365]
    if len(recent) > 150:
        tail = {sym: c[recent[0]:] for sym, c in closes.items()}
        now_dd = max_drawdown(asked.book, tail)[0]
        fits = [w / 100 for w in range(0, 51)
                if max_drawdown(_with(asked.book, asked.add, w / 100), tail)[0] >= -limit]
        out.append(f"Over the last 12 months alone the book's worst fall was {now_dd:.1%}; "
                   + (f"any weight of {add} up to 50% kept it inside {limit:.0%} in that year."
                      if fits and max(fits) >= 0.5 else
                      f"{add} up to {max(fits):.0%} kept it inside {limit:.0%} in that year."
                      if fits and max(fits) > 0 else
                      f"no weight of {add} kept it inside {limit:.0%} even in that year."
                      if not fits else f"no {add} at all kept it inside {limit:.0%} in that year."))
    out.append(f"The weight that gave the shallowest worst fall was {best_w:.0%} {add}, at "
               f"{best_dd:.1%}; at 20% it was "
               f"{dict(sizes)[0.2]:.1%}, at 50% {dict(sizes)[0.5]:.1%}.")
    funded = ("from the cash" if cash > 0.005 else "by trimming every holding in proportion")
    out.append(f"How: daily closes {span} on the {len(days)} days every name traded, the book "
               f"rebalanced daily to its weights, the add paid for {funded} (beyond the cash, by "
               "trimming); worst fall is peak to trough. "
               + (" ".join(f"{p[:1].upper()}{p[1:]}." for p in asked.proxies) + " "
                  if asked.proxies else "")
               + "A past drawdown is not the worst possible one. Not advice.")
    out[0] = "Bottom line: " + out[0]
    out.append("Data: " + "; ".join(said) + ".")
    return out


__all__ = ["Asked", "call", "lines", "max_drawdown", "read"]
