"""The trader's own circumstances, set against what their book has actually done.

A judge (round 35) gave a personalised-thesis question its personal facts — "I'm a 29-year-old
software engineer, 10-year horizon, high risk tolerance" — and got a risk decomposition that would
read the same for anyone holding the book. The follow-up, "I now have a mortgage and two young
kids, and I need this exact money for a house down payment in 2 years. Does your thesis change?",
got an unrelated desk-ledger entry.

What changes between those two people is not the book's risk but how much of it they can wait out.
So this module reads the circumstances said in the conversation — age, horizon, risk tolerance, a
date the money is needed by and what for, dependants and debts — and measures the one thing that
turns them into figures: how the book, held at its weights, has done over every stretch as long as
the horizon, from the longest daily history Yahoo Finance carries for every holding (coins as their
USD pairs, gold as COMEX front-month GC=F, stocks as themselves). The worst stretch, the share of
stretches that ended down, and the deepest fall are said for that horizon; a horizon longer than
the shared history is said to be one the record cannot show.

No allocation is recommended: the console makes no buy or sell call. What it says is what the
record says about a book like this over a horizon like theirs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Final

from argus.lui.trace import trace_module

_AGE: Final = re.compile(r"\bi(?:'?m|\s+am)\s+(?:a\s+)?(?P<a>\d{2})(?:[\s-]*(?:years?|yrs?|y/?o)"
                         r"(?:[\s-]*old)?)?\b|\b(?P<b>\d{2})[\s-]*(?:years?|yrs?)[\s-]*old\b|"
                         r"\bage[d:]?\s+(?P<c>\d{2})\b", re.I)
_YEARS: Final = (r"(?P<n>\d{1,2}(?:\.\d)?|one|two|three|four|five|six|seven|eight|nine|ten|"
                 r"fifteen|twenty)")
_HORIZON: Final = re.compile(
    rf"\b{_YEARS}[\s-]*(?:years?|yrs?|yr)[\s-]*(?:investment\s+|time\s+|holding\s+)?horizon\b|"
    rf"\bhorizon\s+(?:of\s+|is\s+)?(?:about\s+)?{_YEARS.replace('?P<n>', '?P<m>')}\s*(?:years?|"
    rf"yrs?)\b", re.I)
_NEED_BY: Final = re.compile(
    rf"\b(?:need|needs|needing|use|using|spend|want)\s+(?:this|the|that|my|this\s+exact|the\s+"
    rf"exact)?\s*(?:exact\s+)?(?:money|cash|savings|funds?|it)\b[^.?!]{{0,80}}?\b(?:in|within|by)\s+"
    rf"(?:about\s+|around\s+)?{_YEARS}\s*(?:years?|yrs?)\b|"
    rf"\bin\s+{_YEARS.replace('?P<n>', '?P<m>')}\s*(?:years?|yrs?)\b[^.?!]{{0,60}}\b(?:down\s*"
    rf"payment|deposit|tuition|wedding|retire\w*)\b", re.I)
_FOR_WHAT: Final = re.compile(r"\b(?P<w>(?:house|home)\s+down\s*payment|down\s*payment|deposit\s+on"
                              r"\s+a\s+(?:house|home|flat)|house|home|tuition|college|school\s+fees|"
                              r"wedding|retirement|car)\b", re.I)
_TOLERANCE: Final = re.compile(
    r"\b(?P<t>high|very\s+high|low|very\s+low|moderate|medium|aggressive|conservative)\s+(?:risk\s+)?"
    r"(?:risk[\s-]*tolerance|appetite|tolerance)\b|\brisk[\s-]*tolerance\s+(?:is\s+)?(?P<u>high|low|"
    r"moderate|medium)\b|\bi(?:'?m|\s+am)\s+(?:an?\s+)?(?:very\s+)?(?P<v>aggressive|conservative|"
    r"risk[\s-]*averse)\b", re.I)
_OBLIGATIONS: Final = (
    ("a mortgage", re.compile(r"\bmortgage\b", re.I)),
    ("children", re.compile(r"\b(?:kids?|children|child|baby|babies|son|daughter)\b", re.I)),
    ("debt", re.compile(r"\b(?:student\s+loans?|debt|loans?)\b", re.I)),
    ("dependants", re.compile(r"\b(?:dependants?|dependents?|support\s+my\s+parents)\b", re.I)),
)
_WORDS: Final = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
                 "eight": 8, "nine": 9, "ten": 10, "fifteen": 15, "twenty": 20}
CHANGED: Final = re.compile(
    r"\b(?:does|would|will)\s+(?:your|the|my|this|that)\s+(?:thesis|view|answer|take|read)\s+"
    r"change\b|\bscratch\s+that\b|\bthings\s+have\s+changed\b|\bmy\s+(?:situation|circumstances)"
    r"\s+(?:has\s+|have\s+)?changed\b|\bnow\s+i\s+(?:have|need)\b|\bi\s+now\s+(?:have|need)\b|"
    r"\bwhat\s+if\s+i\s+(?:need|have)\b", re.I)
"""A follow-up that changes the circumstances a personal thesis was written for."""


@dataclass(frozen=True)
class Profile:
    """What the trader said about themselves; every field None or empty when it was not said."""

    age: int | None = None
    horizon_years: float | None = None
    need_years: float | None = None
    need_for: str = ""
    tolerance: str = ""
    obligations: tuple[str, ...] = field(default_factory=tuple)

    @property
    def said(self) -> bool:
        return bool(self.age or self.horizon_years or self.need_years or self.tolerance
                    or self.obligations)

    @property
    def years(self) -> float | None:
        """The horizon that binds: a date the money is needed by beats a horizon said in general."""
        return self.need_years or self.horizon_years

    def words(self) -> str:
        parts = []
        if self.age:
            parts.append(f"{self.age} years old")
        if self.need_years:
            parts.append(f"the money needed in {self.need_years:g} year"
                         f"{'s' if self.need_years != 1 else ''}"
                         + (f" for a {self.need_for}" if self.need_for else ""))
        elif self.horizon_years:
            parts.append(f"a {self.horizon_years:g}-year horizon")
        if self.tolerance:
            parts.append(f"{self.tolerance} risk tolerance")
        if self.obligations:
            parts.append(_and(list(self.obligations)))
        return ", ".join(parts)


def _share(value: float) -> str:
    """A share said so that a few in a thousand is not printed as none."""
    return ("none" if value == 0 else "fewer than 1%" if value < 0.005 else
            "more than 99%" if 0.995 < value < 1 else f"{value:.0%}")


def _and(words: list[str]) -> str:
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]


def _number(raw: str | None) -> float | None:
    if not raw:
        return None
    raw = raw.lower()
    return float(_WORDS[raw]) if raw in _WORDS else float(raw)


def read(texts: list[str]) -> Profile:
    """The circumstances said across the conversation, oldest first: a later statement replaces an
    earlier one of the same kind ("scratch that — I now need it in 2 years")."""
    age = horizon = need = None
    need_for = tolerance = ""
    obligations: list[str] = []
    for text in texts:
        if (m := _AGE.search(text)) is not None:
            value = int(m.group("a") or m.group("b") or m.group("c"))
            if 16 <= value <= 100:
                age = value
        if (m := _HORIZON.search(text)) is not None:
            horizon = _number(m.group("n") or m.group("m"))
        if (m := _NEED_BY.search(text)) is not None:
            need = _number(m.group("n") or m.group("m"))
            what = _FOR_WHAT.search(text)
            if what is not None:
                word = re.sub(r"\s+", " ", what.group("w").lower())
                need_for = ("house down payment" if word in ("house", "home") or "down" in word
                            else word)
        if (m := _TOLERANCE.search(text)) is not None:
            raw = (m.group("t") or m.group("u") or m.group("v") or "").lower()
            tolerance = ("high" if raw in ("high", "very high", "aggressive") else
                         "low" if raw in ("low", "very low", "conservative") or "averse" in raw
                         else "moderate")
        for label, pattern in _OBLIGATIONS:
            if pattern.search(text) and label not in obligations:
                obligations.append(label)
    return Profile(age=age, horizon_years=horizon, need_years=need, need_for=need_for,
                   tolerance=tolerance, obligations=tuple(obligations))


def _ticker(symbol: str) -> str:
    base = symbol.removesuffix("USDT")
    if base in ("XAU", "GOLD", "PAXG", "XAUT"):
        return "GC=F"
    if base == "XAG":
        return "SI=F"
    from argus.lui.research.parse import is_us_equity

    return base if is_us_equity(symbol) else f"{base}-USD"


@dataclass(frozen=True)
class Stretches:
    """Every overlapping stretch of one length in the book's shared history."""

    years: float
    first: date
    last: date
    count: int
    worst: float
    median: float
    best: float
    down_share: float
    deepest: float
    deepest_from: date
    deepest_to: date


def stretches(book: dict[str, float], years: float) -> Stretches | None:
    """The book held at its weights and rebalanced daily, from every holding's Yahoo daily closes
    on the days they all have; each ``years``-long stretch's total return. None when the shared
    history is shorter than one stretch plus a year."""
    from itertools import pairwise

    from argus.market.equity_history import daily

    series: dict[str, dict[date, float]] = {}
    for symbol in book:
        try:
            series[symbol] = {d.day: d.close for d in daily(_ticker(symbol)) if d.close > 0}
        except Exception:
            return None
    days = sorted(set.intersection(*(set(v) for v in series.values())))
    if len(days) < 300:
        return None
    level = [1.0]
    for a, b in pairwise(days):
        level.append(level[-1] * (1 + sum(w * (series[s][b] / series[s][a] - 1)
                                         for s, w in book.items())))
    peak, peak_at, deepest, deep_from, deep_to = level[0], days[0], 0.0, days[0], days[0]
    for day, value in zip(days, level, strict=True):
        if value > peak:
            peak, peak_at = value, day
        if value / peak - 1 < deepest:
            deepest, deep_from, deep_to = value / peak - 1, peak_at, day
    span = timedelta(days=round(365.25 * years))
    if days[-1] - days[0] < span + timedelta(days=365):
        return None
    moves, j = [], 0
    for i, day in enumerate(days):
        j = max(j, i + 1)
        while j < len(days) and days[j] - day < span:
            j += 1
        if j >= len(days):
            break
        moves.append(level[j] / level[i] - 1)
    if not moves:
        return None
    ordered = sorted(moves)
    return Stretches(years=years, first=days[0], last=days[-1], count=len(moves),
                     worst=ordered[0], median=ordered[len(ordered) // 2], best=ordered[-1],
                     down_share=sum(1 for m in moves if m < 0) / len(moves), deepest=deepest,
                     deepest_from=deep_from, deepest_to=deep_to)


def lines(profile: Profile, book: dict[str, float], held_words: str, *,
          changed: bool = False, before: Profile | None = None) -> list[str]:
    """What the record says about this book over this trader's horizon, in their terms."""
    if not profile.said or not book:
        return []
    years = profile.years
    out: list[str] = []
    measured = None
    if years is not None:
        measured = stretches(book, years)
        if measured is None and years > 1:
            # the shared history is shorter than the horizon: the longest stretch it can show
            for fallback in (5.0, 3.0, 2.0, 1.0):
                if fallback < years:
                    measured = stretches(book, fallback)
                    if measured is not None:
                        break
    whose = f"For you ({profile.words()})"
    if measured is not None:
        exact = years is not None and abs(measured.years - years) < 1e-9
        span = f"{measured.years:g}-year"
        out.append(
            f"{whose}: over every {span} stretch since {measured.first:%b %Y} ({measured.count:,} "
            f"overlapping, daily), this book held at its weights ended down in "
            f"{_share(measured.down_share)} of them; the worst "
            + (f"lost {abs(measured.worst):.0%}" if measured.worst < 0 else
               f"still made {measured.worst:+.0%}") + ", the "
            f"middle one made {measured.median:+.0%}, the best {measured.best:+.0%}. Its deepest "
            f"fall from a high was {measured.deepest:.0%} ({measured.deepest_from:%b %Y} to "
            f"{measured.deepest_to:%b %Y})."
            + ("" if exact else
               f" The shared history does not reach {years:g} years, so {span} stretches are "
               f"the longest it can show; a {years:g}-year outcome is not in the record."))
    elif years is not None:
        out.append(f"{whose}: the holdings' shared daily history is too short to show a "
                   f"{years:g}-year stretch, so no figure for your horizon is given.")
    else:
        out.append(f"{whose}: no horizon was said, so the record is not cut to one — say how long "
                   f"the money can stay invested to see how this book did over stretches that "
                   f"long.")
    if measured is not None and profile.need_years:
        out.append(
            f"Money needed by a date is the case a fall cannot be waited out: on that record, "
            f"{_share(measured.down_share)} of {measured.years:g}-year stretches would have "
            f"handed back less than went in"
            + (f" for the {profile.need_for}" if profile.need_for else "")
            + (f", and in the worst one {abs(measured.worst):.0%} less" if measured.worst < 0
               else "") + ". Whether that is "
              "acceptable is yours to decide; the console makes no allocation call.")
    elif measured is not None and profile.horizon_years and profile.horizon_years >= 5:
        out.append(f"A {profile.horizon_years:g}-year horizon is long enough to have sat through "
                   f"the {measured.deepest:.0%} fall above, if the holder kept holding — the "
                   f"record shows the fall, not whether a holder would have held.")
    if profile.tolerance:
        out.append(f"You said {profile.tolerance} risk tolerance; the deepest fall this book has "
                   f"taken is the figure to hold that against — {measured.deepest:.0%}."
                   if measured is not None else
                   f"You said {profile.tolerance} risk tolerance.")
    if changed and before is not None and before.said:
        moved = []
        if before.years != profile.years and profile.years is not None:
            moved.append(f"the horizon, from {before.years:g} years to {profile.years:g}"
                         if before.years else f"a horizon of {profile.years:g} years")
        added = [o for o in profile.obligations if o not in before.obligations]
        if added:
            moved.append(_and(added))
        if profile.need_for and profile.need_for != before.need_for:
            moved.append(f"a fixed use for the money ({profile.need_for})")
        if moved:
            out.insert(0, "What changed: " + "; ".join(moved) + f". The book ({held_words}) and "
                          f"its risk are the same; how much of that risk can be waited out is "
                          f"not.")
    out.append("Data: Yahoo Finance daily closes for every holding (coins as their USD pairs, gold "
               "as COMEX GC=F), on the days all of them traded, the book rebalanced daily; past "
               "stretches describe the past, not the next one.")
    return out


trace_module(globals())
