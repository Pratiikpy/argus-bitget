"""The next Bank of Japan meeting, and what a yen-carry unwind has done to BTC on the record.

"When is the next BOJ policy meeting, and what would a yen-carry-trade unwind do to bitcoin's
price?" (asked in Japanese) was matched to the console's own crypto "carry" — spot against a
perpetual short — and answered with funding and fees (round 40 judge, Q24). The yen carry trade is
borrowing in yen to hold higher-yielding assets; it unwinds when the yen jumps, as on 5 August
2024. Both halves are public:

- **The meeting.** The Bank of Japan's own schedule page
  (``boj.or.jp/en/mopo/mpmsche_minu/index.htm``), read now: two-day meetings, the decision on the
  second day.
- **The unwind's footprint.** Days the yen strengthened sharply — USD/JPY (Yahoo's "JPY=X") down
  1.5% or more in a day — over the last five years, and BTC's move from the close before to the
  close two days after (Yahoo's BTC-USD). Against it, BTC's average move over any such window.
"""

from __future__ import annotations

import html
import re
from datetime import date, timedelta
from itertools import pairwise
from statistics import mean, median
from typing import Final

ASKED: Final = re.compile(
    r"\b(?:yen\s+carry|carry[\s-]+trade\b[^?]{0,40}\b(?:yen|japan|jpy|boj)|boj|bank\s+of\s+japan|"
    r"japan\w*\s+(?:central\s+bank|rate\s+hike))\b", re.I)
_BOJ_PAGE: Final = "https://www.boj.or.jp/en/mopo/mpmsche_minu/index.htm"
_MONTHS: Final = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "june": 6, "jul": 7,
                  "july": 7, "aug": 8, "sept": 9, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
_MEETING: Final = re.compile(r"\b(?P<mon>Jan|Feb|Mar|Apr|May|June|July|Aug|Sept|Oct|Nov|Dec)\.?\s+"
                             r"(?P<d1>\d{1,2})\s*\((?:Mon|Tues|Wed|Thurs|Fri)\.\),\s*(?P<d2>\d{1,2})"
                             r"\s*\((?:Mon|Tues|Wed|Thurs|Fri)\.\)")


def boj_meetings(today: date) -> list[date]:
    """Every decision day on the Bank of Japan's published schedule (the second day of each
    two-day meeting), oldest first."""
    from argus.truth import http

    page = http.fetch_text(_BOJ_PAGE, timeout=20.0)
    text = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", page)))
    first_year = re.search(r"Table\s*:\s*(20\d\d)", text)
    year = int(first_year.group(1)) if first_year else today.year
    out: list[date] = []
    last_month = 0
    for m in _MEETING.finditer(text):
        month = _MONTHS[m.group("mon").lower()]
        if month < last_month:
            year += 1
        last_month = month
        out.append(date(year, month, int(m.group("d2"))))
    return out


def lines(text: str) -> list[str] | None:
    """The BOJ date and the yen-spike record for BTC, or None when ``text`` does not ask it."""
    if not ASKED.search(text):
        return None
    from argus.market.equity_history import daily

    out: list[str] = []
    today = date.today()
    try:
        upcoming = [d for d in boj_meetings(today) if d >= today]
    except Exception:
        upcoming = []
    meeting = (f"the Bank of Japan's next policy decision is on {upcoming[0]:%a %d %b %Y}, the "
               f"second day of its two-day meeting (its own published schedule)"
               if upcoming else "the Bank of Japan's schedule page did not answer just now")
    try:
        yen = {d.day: float(d.close) for d in daily("JPY=X")}
        btc = {d.day: float(d.close) for d in daily("BTC-USD")}
    except Exception:
        return [f"Bottom line: {meeting}; the yen and BTC price histories did not answer, so no "
                f"record of past unwinds is given."]
    days = sorted(d for d in yen if d >= today - timedelta(days=5 * 365))

    def btc_at(day: date) -> float | None:
        for back in range(5):
            found = btc.get(day - timedelta(days=back))
            if found:
                return found
        return None

    spikes = []
    for prev, day in pairwise(days):
        drop = yen[day] / yen[prev] - 1
        if drop <= -0.015:
            before, after = btc_at(prev), btc_at(day + timedelta(days=2))
            if before and after:
                spikes.append((day, drop, after / before - 1))
    any_window = []
    for prev in days[::5]:
        before, after = btc_at(prev), btc_at(prev + timedelta(days=3))
        if before and after:
            any_window.append(after / before - 1)
    if spikes:
        falls = sum(m < 0 for *_, m in spikes)
        lean = ("a sharp yen jump has more often come with BTC falling" if falls * 2 > len(spikes)
                else "a sharp yen jump has not, on its own, reliably moved BTC")
        out.append(f"Bottom line: {meeting}. On the record, {lean}: on the {len(spikes)} days in "
                   f"five years that USD/JPY fell 1.5% or more, BTC moved a median "
                   f"{median(m for *_, m in spikes):+.1%} from the close before to two days after, "
                   f"and fell in {falls} of {len(spikes)}.")
        worst = min(spikes, key=lambda s: s[2])
        out.append(f"The sharpest: {worst[0]:%d %b %Y}, the yen up {-worst[1]:.1%} in a day and "
                   f"BTC {worst[2]:+.1%} over the window — "
                   + ("the August 2024 unwind, when the trade was crowded." if worst[0].year == 2024
                      and worst[0].month == 8 else "one episode, not a rule."))
        if any_window:
            out.append(f"For comparison, any three-day window in the same five years: BTC averaged "
                       f"{mean(any_window):+.1%}. {len(spikes)} episodes are few; the link runs "
                       f"through leveraged positions being cut everywhere at once, so it is "
                       f"strongest when markets are crowded.")
    else:
        out.append(f"Bottom line: {meeting}. No day in five years saw USD/JPY fall 1.5% or more, "
                   f"so there is no record of a yen spike to read.")
    out.append("A carry unwind is the yen strengthening as borrowed yen is repaid; it is not the "
               "crypto \"carry\" of spot against a perpetual short. Data: Bank of Japan schedule, "
               "Yahoo Finance daily closes (JPY=X, BTC-USD). Not advice.")
    return out
