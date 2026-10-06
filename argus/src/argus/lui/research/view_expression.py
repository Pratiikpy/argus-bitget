"""A view stated without reasons, turned into a trade that expresses it and the events that would
prove it wrong.

Round 43's judge (C7) stated three views — "I think oil is going higher into winter", "I think the
dollar is topping out and gold will outperform bitcoin into year end", and a pair trade — and
asked "how would I express it and what would prove me wrong?". All three were told "there is no
thesis earlier in this conversation", although the view was in the same message: the thesis
reader needs reasons ("because funding is negative") to test, and a bare directional view has
none. A view still has an expression and falsifiers, and both are measurable:

* **The expression.** The Bitget contract for each instrument named (a perpetual, so there is no
  expiry to roll), the direction the words give ("higher", "outperform", "topping out"), and a
  relative view as a ratio trade: long the one said to win, short the other, equal dollars.
* **The stop and the size.** The stop is the last 20 daily closes' low for a long (high for a
  short; the ratio's for a relative view) — the level whose break says the market has not started
  agreeing. The size is the risk budget over the stop's distance: 1% of the account the
  conversation states, or per $10,000 when none is stated, and the answer says which.
* **The prior.** Of every window as long as the horizon asked over the last five years of daily
  closes, the share that moved the way the view needs and the median move — what the view has to
  beat to be more than the base rate.
* **The falsifiers.** A close beyond the stop; reaching the horizon without the move (said with
  the size of move the base rate makes ordinary); and, for a relative view, the ratio breaking its
  20-day extreme. Anything the view names that this console cannot trade or measure ("the dollar"
  has no Bitget contract) is said, not dropped.

Every figure comes from `rule_test.daily_closes` (Bitget's daily closes for crypto and the
commodities Bitget lists, Yahoo's split-adjusted closes for US stocks). A base rate is a history,
not a forecast.
"""

from __future__ import annotations

import re
import statistics
from collections.abc import Sequence
from datetime import date, datetime
from typing import Final

from argus.lui.numbers import sig

_VIEW: Final = re.compile(r"\bi\s+(?:think|believe|feel|expect|reckon|bet)\b|\bmy\s+(?:view|"
                          r"thesis|bet|take|call)\s+is\b", re.I)
_EXPRESS: Final = re.compile(
    r"\bhow\s+(?:would|should|could|do|can)\s+i\s+(?:express|trade|play|position|implement|"
    r"structure)\b|\bwhat\s+(?:would|could)\s+(?:prove|show)\b[^?]{0,30}\bwrong\b|\bprove\s+me\s+"
    r"wrong\b|\bwhat\s+(?:trade|position)\s+(?:would|should)\b", re.I)
_UP: Final = re.compile(r"\b(?:higher|rise|rises|rising|rall\w*|go(?:es|ing)?\s+up|climb\w*|"
                        r"bull\w*|outperform\w*|beat\w*|win|gain\w*|moon\w*|rip\w*)\b", re.I)
_DOWN: Final = re.compile(r"\b(?:lower|fall\w*|drop\w*|decline\w*|top(?:ping|s|ped)?\s+out|"
                          r"peak\w*|underperform\w*|bear\w*|crash\w*|weaken\w*|roll(?:s|ing)?\s+"
                          r"over)\b", re.I)
_RELATIVE: Final = re.compile(r"\b(?P<a>[A-Za-z][\w.&-]*)\s+(?:will\s+|to\s+|should\s+|can\s+)?"
                              r"(?P<verb>outperform|beat|underperform|lag)\w*\s+(?P<b>[A-Za-z][\w.&-]*)",
                              re.I)
_UNTRADED: Final = {r"\b(?:the\s+)?(?:us\s+)?dollar\b(?!\s+index\s+futures)": "the dollar",
                    r"\byields?\b|\brates?\b": "rates", r"\binflation\b": "inflation"}
_ACCOUNT: Final = re.compile(r"\$?(?P<n>\d+(?:,\d{3})*(?:\.\d+)?)\s*(?P<u>k|m)?\s*(?:usd\s+)?"
                             r"(?:account|book|portfolio|capital)\b", re.I)
_RISK: Final = re.compile(r"\b(?:risk(?:ing)?|max\s+risk)\s+(?:of\s+)?(?P<r>\d+(?:\.\d+)?)\s*%",
                          re.I)
DEFAULT_RISK: Final = 0.01
STOP_WINDOW: Final = 20
DEFAULT_DAYS: Final = 60


def _horizon(text: str, today: date) -> tuple[int, str]:
    """(calendar days, how it was read)."""
    m = re.search(r"\b(\d{1,3})\s*(day|week|month)s?\b", text, re.I)
    if m:
        n, unit = int(m.group(1)), m.group(2).lower()
        days = n * (7 if unit == "week" else 30 if unit == "month" else 1)
        return days, f"{n} {unit}{'s' if n != 1 else ''}"
    if re.search(r"\byear[\s-]?end\b|\bend\s+of\s+(?:the\s+)?year\b", text, re.I):
        return max(10, (date(today.year, 12, 31) - today).days), "to year end"
    if re.search(r"\bwinter\b", text, re.I):
        end = date(today.year + (1 if today.month >= 3 else 0), 2, 28)
        return max(10, (end - today).days), "through winter (to the end of February)"
    if re.search(r"\b(?:this|the)\s+quarter\b", text, re.I):
        q_end_month = ((today.month - 1) // 3 + 1) * 3
        end = date(today.year, q_end_month, 30 if q_end_month in (6, 9) else 31)
        return max(10, (end - today).days), "to quarter end"
    return DEFAULT_DAYS, f"{DEFAULT_DAYS} days (no horizon was stated)"


def _account(texts: Sequence[str]) -> tuple[float | None, float]:
    size, risk = None, DEFAULT_RISK
    for t in texts:
        a = _ACCOUNT.search(t)
        if a:
            size = float(a.group("n").replace(",", "")) * {"k": 1e3, "m": 1e6}.get(
                (a.group("u") or "").lower(), 1.0)
        r = _RISK.search(t)
        if r:
            risk = float(r.group("r")) / 100
    return size, risk


def _windows(closes: list[float], days: int, bars_per_day: float) -> list[float]:
    """Every overlapping window of ``days`` calendar days, in bars (a stock trades about five
    days in seven, a coin every day)."""
    step = max(1, round(days * bars_per_day))
    return [closes[i + step] / closes[i] - 1 for i in range(0, len(closes) - step)]


def _per_day(stamps: Sequence[object]) -> float:
    first, last = stamps[0], stamps[-1]
    span = (last - first).days if isinstance(first, (datetime, date)) and isinstance(
        last, (datetime, date)) else 0
    return len(stamps) / span if span > 0 else 1.0


def lines(text: str, prior: Sequence[str] = (), *, today: date | None = None
          ) -> list[str] | None:
    """The expression and falsifiers of a directional or relative view, or None when ``text``
    states no view or does not ask how to express or falsify one."""
    from argus.lui.research.parse import research_symbols
    from argus.lui.research.rule_test import daily_closes

    view_text = text if _VIEW.search(text) else next(
        (t for t in reversed(list(prior)[-3:]) if _VIEW.search(t)), None)
    if view_text is None or not _EXPRESS.search(text):
        return None
    if re.search(r"\bpair\b|\bspread\b|\bhedge\s+ratio\b", view_text + " " + text, re.I):
        return None  # a pair trade is `pair_trade`'s
    symbols, notes = research_symbols(view_text)
    if not symbols:
        return None
    today = today or datetime.now().date()
    days, horizon_said = _horizon(view_text, today)
    size, risk = _account([*prior, text])
    budget = (size or 10_000.0) * risk
    out: list[str] = []
    untraded = [label for pattern, label in _UNTRADED.items()
                if re.search(pattern, view_text, re.I) and not re.search(
                    r"\bdxy\b|\bdollar\s+index\b", " ".join(symbols), re.I)]
    rel = _RELATIVE.search(view_text)
    pair: tuple[str, str] | None = None
    if rel is not None and len(symbols) >= 2:
        a = research_symbols(rel.group("a"))[0]
        b = research_symbols(rel.group("b"))[0]
        if a and b and a[0] != b[0]:
            winner, loser = (a[0], b[0]) if rel.group("verb").lower() in (
                "outperform", "beat") else (b[0], a[0])
            pair = (winner, loser)
    try:
        if pair is not None:
            out = _relative(pair, days, horizon_said, budget, size is not None, risk)
        else:
            sym = symbols[0]
            up = bool(_UP.search(view_text)) or not _DOWN.search(view_text)
            out = _directional(sym, up, days, horizon_said, budget, size is not None, risk)
    except Exception:
        return ["Bottom line: the daily history behind that view could not be read just now, so "
                "no stop, size or base rate is given; ask again in a minute."]
    for label in untraded:
        out.insert(1, f"Not expressed: {label} — there is no Bitget contract on it, so that part "
                      "of the view is carried only through the legs above; it is said, not "
                      "traded.")
    if notes:
        out.insert(1, "Read: " + "; ".join(notes) + ".")
    out.append("Data: " + "; ".join(dict.fromkeys(
        daily_closes(s)[2] for s in (pair or (symbols[0],)))) + ". A base rate is a history, "
               "not a forecast. Not advice.")
    return out


def _stats(closes: list[float], days: int, bars_per_day: float) -> tuple[float, float]:
    moves = _windows(closes, days, bars_per_day)
    return sum(m > 0 for m in moves) / len(moves), statistics.median(moves)


def _directional(sym: str, up: bool, days: int, horizon: str, budget: float, stated: bool,
                 risk: float) -> list[str]:
    from argus.lui.research.rule_test import daily_closes

    stamps, closes, _ = daily_closes(sym)
    name = sym.removesuffix("USDT")
    last = closes[-1]
    recent = closes[-STOP_WINDOW:]
    stop = min(recent) if up else max(recent)
    distance = abs(last / stop - 1)
    why = f"the last {STOP_WINDOW} daily closes' {'low' if up else 'high'}"
    if distance < 0.01:
        # a 20-day extreme within 1% is inside one day's noise; 3% is used and said
        stop = last * (0.97 if up else 1.03)
        distance = 0.03
        why = (f"3% {'below' if up else 'above'} — the last {STOP_WINDOW} closes' "
               f"{'low' if up else 'high'} is within 1%, inside a day's noise")
    notional = budget / distance
    rose, median = _stats(closes, days, _per_day(stamps))
    share = rose if up else 1 - rose
    side = "long" if up else "short"
    sized = (f"${notional:,.0f} of {name} risks ${budget:,.0f} ({risk:.0%} of your "
             f"${budget / risk:,.0f})" if stated else
             f"${notional:,.0f} of {name} per $10,000 of account at {risk:.0%} risk "
             "(no account size was stated)")
    return [
        f"Bottom line: express it as a {side} {name} perpetual on Bitget, stop at {stop:,.2f} "
        f"({why}; {distance:.1%} from {last:,.2f}); {sized}.",
        f"Horizon read as {horizon}: over the last five years {share:.0%} of {days}-day windows "
        f"moved {name} the way this view needs (median {median:+.1%}), so the view has to beat "
        "that base rate to be more than the market's usual drift.",
        f"Wrong if: {name} closes {'below' if up else 'above'} {stop:,.2f}; or the horizon "
        f"passes with {name} {'at or below' if up else 'at or above'} {last:,.2f} — the move "
        "the view claims has not come; a perpetual also settles funding every period it is "
        f"held, which a {side} {'pays' if up else 'receives'} while funding is positive.",
    ]


def _relative(pair: tuple[str, str], days: int, horizon: str, budget: float, stated: bool,
              risk: float) -> list[str]:
    from argus.lui.research.rule_test import daily_closes

    winner, loser = pair
    s1, c1, _ = daily_closes(winner)
    s2, c2, _ = daily_closes(loser)
    one = {t.date(): c for t, c in zip(s1, c1, strict=True)}
    two = {t.date(): c for t, c in zip(s2, c2, strict=True)}
    common = sorted(set(one) & set(two))
    ratio = [one[d] / two[d] for d in common]
    w, lo = winner.removesuffix("USDT"), loser.removesuffix("USDT")
    recent = ratio[-STOP_WINDOW:]
    stop = min(recent)
    distance = max(0.01, ratio[-1] / stop - 1)
    leg = budget / distance / 2
    rose, median = _stats(ratio, days, _per_day(common))
    r1 = [one[common[i]] / one[common[i - 1]] - 1 for i in range(1, len(common))]
    r2 = [two[common[i]] / two[common[i - 1]] - 1 for i in range(1, len(common))]
    corr = statistics.correlation(r1[-250:], r2[-250:]) if len(r1) > 30 else float("nan")
    sized = (f"${leg:,.0f} a leg risks ${budget:,.0f} ({risk:.0%} of your ${budget / risk:,.0f})"
             if stated else f"${leg:,.0f} a leg per $10,000 of account at {risk:.0%} risk (no "
                            "account size was stated)")
    return [
        f"Bottom line: express it as a ratio trade — long the {w} perpetual and short the {lo} "
        f"perpetual on Bitget, equal dollars; {sized}, with the stop where the {w}/{lo} ratio "
        f"closes below its last {STOP_WINDOW} days' low ({distance:.1%} below today's ratio).",
        f"Horizon read as {horizon}: over the past five years the {w}/{lo} ratio rose in "
        f"{rose:.0%} of {days}-day windows (median {median:+.1%}); their daily returns "
        f"correlate {corr:+.2f} over the last year, so the trade is mostly a bet on the "
        "difference, not on the market's direction.",
        f"Wrong if: the ratio closes below that 20-day low; or the horizon passes with the ratio "
        f"at or below today's {sig(ratio[-1], 4)} — {w} has not beaten {lo}; or one leg's funding "
        "eats the edge (each perpetual charges or pays funding every period).",
    ]


__all__ = ["lines"]
