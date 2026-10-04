"""A money target with a date: "turn $1,000 into $5,000 by the end of this year".

A first-time user stated exactly that, asked whether it was realistic, then asked which coins to
focus on, whether to trade every day or hold, and what to do if the date passed without it (round
30). The first got two coins' year-to-date returns and no verdict; the other three got the desk's
own track record and its open positions. Each is answered here from the target itself:

* **Realistic?** The multiple asked for is a return over a stated number of days. Every window of
  that length in about three years of Bitget daily closes (``dispatch.span_moves``, 00:00 UTC
  days) is counted, and the answer says how many reached it. That is a base rate over overlapping
  windows, not a forecast, and it says so.
* **Which coins?** The same count for five large coins Bitget lists, with each one's worst window,
  so the coins that cleared the bar more often are shown beside how far they fell.
* **Every day or hold?** What trading once a day costs at Bitget's standard taker fee over the
  days left (``desk_answers.SPOT_TAKER_VIP0`` and ``PERP_TAKER``, read from Bitget's contract
  list), against the return the target needs.
* **If it does not work out?** The decision to make before the date, not at it.

No pick and no call: each answer measures the goal against what these markets have done.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Final

from argus.lui.trace import trace_module

TARGET: Final = re.compile(
    r"\b(?:turn|grow|make|get|flip|build)\s+(?:my\s+|this\s+|the\s+|a\s+|only\s+|just\s+)?\$?\s*"
    r"(?P<a>\d[\d,]*(?:\.\d+)?)(?![\d,.]*\d)\s*(?P<ka>k|grand|thousand)?\s*(?:bucks|dollars?|usd|"
    r"usdt)?\s+(?:in)?to\s+\$?\s*(?P<b>\d[\d,]*(?:\.\d+)?)(?![\d,.]*\d)\s*(?P<kb>k|grand|"
    r"thousand)?\b", re.I)
"""A sum and the sum it should become."""
_BY_YEAR_END: Final = re.compile(
    r"\b(?:by\s+|before\s+)?(?:the\s+)?end\s+of\s+(?:this|the)\s+year\b|\bby\s+(?:december|dec|"
    r"new\s+year'?s?|christmas)\b|\b(?:this|by\s+the\s+end\s+of\s+the)\s+year\b|\beoy\b", re.I)
_IN_SPAN: Final = re.compile(
    r"\b(?:in|within|over|by)\s+(?:the\s+next\s+)?(?:like\s+|about\s+|around\s+)?"
    r"(?P<n>\d+|a|one|two|three|six)\s+(?P<u>days?|"
    r"weeks?|months?|years?)\b", re.I)
_WORDS: Final = {"a": 1, "one": 1, "two": 2, "three": 3, "six": 6}
_UNIT_DAYS: Final = {"day": 1, "week": 7, "month": 30, "year": 365}

FOCUS: Final = re.compile(
    r"\bwhat\s+(?:coins?|crypto|tokens?|stocks?|markets?)\s+(?:should|do|would|can)\s+i\s+"
    r"(?:actually\s+|even\s+)?(?:focus\s+on|buy|pick|look\s+at|trade|get)\b|\bwhich\s+(?:coins?|"
    r"tokens?|ones?)\s+(?:should|would|could)\b|\bwhat\s+(?:should|do)\s+i\s+focus\s+on\b|"
    r"^\W*(?:so\s+|ok(?:ay)?\s+|and\s+)?(?:which|what)\s+(?:coins?|ones?|tokens?)(?:\s+then)?\W*$",
    re.I)
DAILY_OR_HOLD: Final = re.compile(
    r"\b(?:trad(?:e|ing)|day\s*trad(?:e|ing))\s+every\s*day\b[^?]{0,40}\bhold|\bhold\w*\b[^?]{0,40}"
    r"\btrad(?:e|ing)\s+every\s*day\b|\bday\s*trad\w*\s+or\s+(?:just\s+)?hold|\bhold\s+or\s+"
    r"(?:just\s+)?(?:day\s*)?trad",
    re.I)
MISSED: Final = re.compile(
    r"\bif\s+it\s+(?:does\s*n[o']?t|doesnt|does\s+not|wont|won'?t|never)\s+(?:work(?:\s+out)?|"
    r"happen|get\s+there|hit|reach)\b|\bif\s+i\s+(?:miss|don'?t\s+(?:hit|reach|make))\b|\bwhat\s+if\s+i\s+"
    r"(?:miss|fail|don'?t\s+make\s+it)\b", re.I)
REALISTIC: Final = re.compile(
    r"\b(?:is\s+(?:that|this|it)\s+(?:even\s+)?(?:realistic|possible|doable|achievable|"
    r"reasonable|crazy)|realistic(?:ally)?|possible|doable|achievable|can\s+i\s+(?:do|make)\s+"
    r"(?:that|it|this))\b", re.I)
COINS: Final = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT")


@dataclass(frozen=True, slots=True)
class Target:
    start: float
    goal: float
    days: int
    deadline: date | None
    """None when no date was said and a year is assumed."""

    @property
    def needed(self) -> float:
        return self.goal / self.start - 1


def _money(value: str, k: str | None) -> float:
    return float(value.replace(",", "")) * (1000 if k else 1)


_TARGET_IT: Final = re.compile(
    r"\b(?:turn|grow|make|get|flip)\s+(?:it|that|this|them|my\s+money)\s+(?:in)?to\s+(?:like\s+|"
    r"about\s+|around\s+)?\$?\s*(?P<b>\d[\d,]*(?:\.\d+)?)(?![\d,.]*\d)\s*(?P<kb>k|grand|thousand)?"
    r"\b", re.I)
_HAVE: Final = re.compile(
    r"\b(?:got|have|has|holding|saved)\s+(?:only\s+|just\s+|like\s+|about\s+)*\$?\s*"
    r"(?P<a>\d[\d,]*(?:\.\d+)?)(?![\d,.]*\d)\s*(?P<ka>k|grand|thousand)?\s*(?:bucks|dollars?|usd|"
    r"usdt)?", re.I)
""""i only got 50 bucks to my name, i need to turn it into like 300 for a concert ticket in 3
weeks, is that even possible" was filed as a note and never answered (a first-time user, round
31): the sum held and the sum wanted, said apart."""


_DOUBLE: Final = re.compile(
    r"\b(?P<m>double|triple|quadruple|10x|5x|3x|2x)\s+(?:my\s+|the\s+|this\s+)?\$?\s*(?P<a>\d[\d,]*"
    r"(?:\.\d+)?)(?![\d,.]*\d)\s*(?P<ka>k|grand|thousand)?\s*(?:bucks|dollars?|usd|usdt)?\b", re.I)
_MULTIPLE: Final = {"double": 2.0, "2x": 2.0, "triple": 3.0, "3x": 3.0, "quadruple": 4.0,
                    "5x": 5.0, "10x": 10.0}
"""A target said as a multiple: "my goal is to double 1000 dollars in like 6 months" was filed as a
note (a first-time user, round 31)."""
NEW_DATE: Final = re.compile(
    r"\b(?:in|is|by)\s+(?P<n>\d+|a|one|two|three|six)\s+(?P<u>days?|weeks?|months?)\s+(?:not|"
    r"instead\s+of)\s+\d+|\b(?:not|instead\s+of)\s+\d+\s*(?:days?|weeks?|months?)?[^?]{0,20}?\b(?:"
    r"in|is|it'?s)\s+(?P<n2>\d+)\s+(?P<u2>days?|weeks?|months?)\b", re.I)
"""A deadline moved: "actually wait the concert is in 5 weeks not 3"."""
SAFER: Final = re.compile(r"\b(?:safer|less\s+risky|riskier|more\s+dangerous)\b", re.I)
_LEVERAGE_SAID: Final = re.compile(r"\bleverage|\bfutures\b|\bperps?\b|\bmargin\b", re.I)


_NEED: Final = re.compile(
    r"\b(?:need|needs|want|gotta\s+(?:get|have)|have\s+to\s+(?:get|have|make))\s+(?:like\s+|about\s+|"
    r"around\s+)?\$?\s*(?P<b>\d[\d,]*(?:\.\d+)?)(?![\d,.]*\d)\s*(?P<kb>k|grand|thousand)?\s*"
    r"(?:bucks|dollars?|usd|usdt)?\b", re.I)
"""A sum needed by a date, said before what is held: "i need 400 dollars for a festival in 4 weeks
and i have 80" was filed as a note (a live re-ask, round 31)."""


def stated(text: str, *, today: date) -> Target | None:
    """The target ``text`` states, with the days it allows."""
    m = TARGET.search(text)
    it = _TARGET_IT.search(text) if m is None else None
    if it is None and m is None and (_IN_SPAN.search(text) or _BY_YEAR_END.search(text)):
        it = _NEED.search(text)
    have = _HAVE.search(text) if it is not None else None
    doubled = _DOUBLE.search(text) if m is None else None
    if m is not None:
        start, goal = _money(m.group("a"), m.group("ka")), _money(m.group("b"), m.group("kb"))
    elif doubled is not None:
        start = _money(doubled.group("a"), doubled.group("ka"))
        goal = start * _MULTIPLE[doubled.group("m").lower()]
    elif it is not None and have is not None:
        start, goal = (_money(have.group("a"), have.group("ka")),
                       _money(it.group("b"), it.group("kb")))
    else:
        return None
    if start <= 0 or goal <= start:
        return None
    if _BY_YEAR_END.search(text):
        deadline = date(today.year, 12, 31)
        if deadline <= today:
            deadline = date(today.year + 1, 12, 31)
        return Target(start, goal, (deadline - today).days, deadline)
    span = _IN_SPAN.search(text)
    if span is not None:
        n = span.group("n").lower()
        count = int(n) if n.isdigit() else _WORDS[n]
        days = count * _UNIT_DAYS[span.group("u").lower().rstrip("s")]
        return Target(start, goal, max(1, days), today + timedelta(days=days))
    return Target(start, goal, 365, None)


def remembered(text: str, prior: list[str], *, today: date) -> Target | None:
    """The target this message or one of the last few states, with any later moved date applied.

    "the concert is in 5 weeks not 3" moved the date, and the next turn still answered with the old
    one (a first-time user, round 31): a date moved after the target was set now carries."""
    turns = [*prior[-6:], text]
    for at in range(len(turns) - 1, -1, -1):
        target = stated(turns[at], today=today)
        if target is not None:
            for later in turns[at + 1:-1]:
                target = moved_target(target, later, today=today) or target
            return target
    return None


def moved_target(target: Target, text: str, *, today: date) -> Target | None:
    """``target`` with the deadline ``text`` moves it to, or None when ``text`` moves nothing."""
    moved = NEW_DATE.search(text)
    if moved is None:
        return None
    n = (moved.group("n") or moved.group("n2")).lower()
    unit = (moved.group("u") or moved.group("u2")).lower().rstrip("s")
    days = (int(n) if n.isdigit() else _WORDS[n]) * _UNIT_DAYS[unit]
    return Target(target.start, target.goal, days, today + timedelta(days=days))


def _count(symbol: str, days: int, needed: float) -> tuple[int, int, float, float] | None:
    """(windows that reached ``needed``, windows, best, worst) over ``days``-day windows."""
    from argus.lui.research.dispatch import span_moves

    spans = span_moves(symbol, days)
    if spans is None:
        return None
    moves = spans[0]
    return sum(1 for m in moves if m >= needed), len(moves), max(moves), min(moves)


def _when(target: Target) -> str:
    if target.deadline is None:
        return "in a year (no date was given, so a year is assumed)"
    return f"by {target.deadline:%d %b %Y}, {target.days} days away"


def realism_lines(target: Target) -> list[str] | None:
    rows = []
    for symbol in ("BTCUSDT", "ETHUSDT"):
        got = _count(symbol, target.days, target.needed)
        if got is not None:
            rows.append((symbol.removesuffix("USDT"), *got))
    if not rows:
        return None
    hits = sum(r[1] for r in rows)
    verdict = ("not realistic without leverage" if hits == 0 else
               "possible but rare" if hits / sum(r[2] for r in rows) < 0.05 else
               "within what these markets have done, though not the usual outcome")
    per = "; ".join(f"{name} reached it in {hit:,} of {n:,} {target.days}-day windows (best "
                    f"{best:+.0%}, worst {worst:+.0%})" for name, hit, n, best, worst in rows)
    lev = 5.0
    under, wipe = target.needed / lev, 1 / lev - 0.004
    from argus.lui.research.dispatch import span_moves

    falls = span_moves("BTCUSDT", 30)
    wiped = (f", and BTC ended {sum(1 for m in falls[0] if m <= -wipe) / len(falls[0]):.0%} of its "
             f"30-day windows down that far or more" if falls else "")
    return [f"Bottom line: turning ${target.start:,.0f} into ${target.goal:,.0f} {_when(target)}, "
            f"needs +{target.needed:.0%} — {verdict}. Over about three years of Bitget daily "
            f"closes, {per}.",
            f"With {lev:g}x leverage the coin itself would need about +{under:.0%} (before fees "
            f"and funding) — and at {lev:g}x a fall of about {wipe:.0%} against you wipes the "
            f"stake{wiped}.",
            "Overlapping windows share most of their days, so these counts describe what happened, "
            "not the odds of the next stretch.",
            "A goal this far above what the market has done usually ends with a larger size or "
            "more leverage to catch up; deciding the most you are willing to lose first is what "
            "keeps it a goal and not a wipe-out."]


def focus_lines(target: Target) -> list[str] | None:
    rows = []
    for symbol in COINS:
        got = _count(symbol, target.days, target.needed)
        if got is not None:
            rows.append((symbol.removesuffix("USDT"), *got))
    if not rows:
        return None
    rows.sort(key=lambda r: (-r[1] / r[2], r[0]))
    lines = [f"Bottom line: no pick — but your goal sets the bar at +{target.needed:.0%} "
             f"{_when(target)}, and here is how often five large coins on Bitget cleared it over "
             f"windows that long in about three years, beside how far each fell:"]
    lines += [f"{name}: reached +{target.needed:.0%} in {hit / n:.1%} of {n:,} windows (best "
              f"{best:+.0%}); its worst window was {worst:+.0%}."
              for name, hit, n, best, worst in rows]
    lines.append("The coins that cleared the bar more often are the ones that also fell furthest: "
                 "the same swings that make a goal like this possible make losing most of it "
                 "possible. Overlapping windows describe the past, not the next stretch.")
    return lines


def daily_or_hold_lines(target: Target | None) -> list[str]:
    from argus.lui.research.desk_answers import PERP_TAKER, SPOT_TAKER_VIP0

    days = target.days if target is not None else 90
    spot = 1 - (1 - 2 * SPOT_TAKER_VIP0) ** days
    perp = 1 - (1 - 2 * PERP_TAKER) ** days
    need = (f" — on top of the +{target.needed:.0%} your goal needs" if target is not None else "")
    span = f"the {days} days to your date" if target is not None else f"{days} days"
    return [f"Bottom line: holding costs almost nothing; trading every day has a cost before any "
            f"result. One round trip a day for {span} at Bitget's standard taker fee takes about "
            f"{spot:.0%} of the money on spot ({SPOT_TAKER_VIP0:.2%} a side) or {perp:.0%} on a "
            f"perpetual ({PERP_TAKER:.2%} a side){need}.",
            "That is the fee alone; the spread and any losing trades come on top, and most "
            "short-term traders lose to fees and mistakes before they find an edge.",
            "Holding a perpetual pays or earns funding every few hours instead; holding spot "
            "pays nothing while you hold.",
            "Ask \"what does it cost to trade BTC\" for the fee arithmetic on one trade."]


def missed_lines(target: Target | None, text: str = "") -> list[str]:
    when = (f" ({target.deadline:%d %b %Y})" if target is not None and target.deadline else "")
    decide = (f"decide it now, not on the date{when} — the date is yours, the market's moves are "
              "not, and a missed target is the moment people double the size or add leverage to "
              "catch up, which is how a missed goal becomes a lost stake.")
    rest = ["The three honest options when the date comes: keep what is there and move the date; "
            "take it out and stop; or accept the result and change nothing. None of them is "
            "\"bet bigger\".",
            "Write down today the loss at which you stop — for example 30% of what you started "
            "with — and keep to it whatever the date.",
            "Ask \"how much could I lose on BTC in a bad month\" with your amount to see that loss "
            "in dollars."]
    if re.search(r"\bborrow\w*|\bloan\b", text, re.I):
        # The borrowing question is the one asked, so it leads (a first-time user, round 31).
        return ["Bottom line: no — borrowing to make up a miss is the most expensive version of "
                "\"bet bigger\": a loss then leaves a debt as well, with interest.",
                f"{decide[:1].upper()}{decide[1:]}", *rest]
    return [f"Bottom line: {decide}", *rest]


def safer_lines(target: Target) -> list[str]:
    """Holding against leverage for the same target, said for someone who has never traded."""
    lev = 5.0
    return [f"Bottom line: buying and holding without leverage is the safer of the two — the most "
            f"you can lose is the ${target.start:,.0f} you put in, and only if the coin goes to "
            f"zero; with {lev:g}x leverage a fall of about {1 / lev - 0.004:.0%} closes the "
            f"position and the ${target.start:,.0f} is gone.",
            f"Leverage makes the +{target.needed:.0%} target closer and the total loss far more "
            f"likely at the same time; for someone who has never traded, the first mistake with "
            f"leverage is usually the whole stake.",
            "Ask \"what happens on a 3x long on BTC\" to see the line for any leverage."]


def lines(text: str, prior: list[str], *, now: datetime | None = None) -> list[str] | None:
    """The answer when ``text`` is about a stated money target, or None."""
    today = (now or datetime.now(UTC)).date()
    here = stated(text, today=today)
    # a goal said in full — the sum, the target and the date — is a question in itself
    if here is not None and (REALISTIC.search(text) or "?" in text or here.deadline is not None
                             or re.search(r"\b(?:should|can|could|would)\s+i\b|\bhow\s+(?:do|"
                                          r"can|should)\s+i\b", text, re.I)):
        return realism_lines(here)
    target = remembered(text, prior, today=today)
    if target is None:
        return daily_or_hold_lines(None) if DAILY_OR_HOLD.search(text) else None
    moved = moved_target(target, text, today=today)
    if moved is not None:
        before, days = target.days, moved.days
        target = moved
        again = realism_lines(target)
        if again is not None:
            return [again[0], f"What changed: {before} days became {days}; the rise needed is "
                              f"the same +{target.needed:.0%}, measured now over {days}-day "
                              f"windows.", *again[1:]]
    if SAFER.search(text) and any(_LEVERAGE_SAID.search(q) for q in [*prior[-3:], text]):
        return safer_lines(target)
    if FOCUS.search(text):
        return focus_lines(target)
    if DAILY_OR_HOLD.search(text):
        return daily_or_hold_lines(target)
    if MISSED.search(text):
        return missed_lines(target, text)
    return None


__all__ = ["DAILY_OR_HOLD", "FOCUS", "MISSED", "NEW_DATE", "REALISTIC", "SAFER", "TARGET",
           "Target", "daily_or_hold_lines", "focus_lines", "lines", "missed_lines",
           "moved_target", "realism_lines", "remembered", "safer_lines", "stated"]

trace_module(globals())
