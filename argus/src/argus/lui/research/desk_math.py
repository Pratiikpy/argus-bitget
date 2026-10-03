"""Four questions a trader works out on paper, answered from the trader's own figures.

A hostile review in round 25 asked each and got an unrelated answer or a refusal:

- "BTC spot is 64,000 and the quarterly future is 65,600 with 90 days to expiry. What's the basis
  and annualised yield?" — Treasury yields. The basis is future minus spot; its yield is that over
  spot, annualised simply (x 365 / days, the convention exchanges quote a cash-and-carry in) and
  compounded beside it.
- "I opened a short at 07:59 UTC and closed at 08:01 UTC with $1,000,000 notional and +0.05%
  funding. Did I pay or receive?" and "from Friday 16:00 UTC to Monday 16:00 UTC, funding +0.01%
  every 8h. How many payments?" — a decline and the next settlement time. Funding is exchanged by
  whoever holds the position at a settlement instant (Bitget: 00:00, 08:00 and 16:00 UTC on the
  usual 8-hour schedule), so the count is the settlements after the open and up to the close; a
  positive rate is paid by longs to shorts.
- "The Fed decision is at 2pm New York time on 28 Oct. What time is that in UTC and in Mumbai?" —
  a rates answer. The IANA time-zone database (`zoneinfo`) gives the offset on that date, so a
  daylight-saving change between now and then is applied.
- "I'm long $500k of SOL. Its beta to BTC is 1.4. How much BTC do I short?" — the console's own
  beta. A beta the trader states is the one used: hedge = position x beta.
"""

from __future__ import annotations

import re
from datetime import UTC, date, datetime, time, timedelta
from itertools import pairwise
from zoneinfo import ZoneInfo

from argus.lui.trace import trace_module

_NUM = r"\$?\s?(?:\d[\d,]*(?:\.\d+)?)(?:\s?[kKmM](?![A-Za-z]))?"


def _n(raw: str) -> float:
    s = raw.replace("$", "").replace(",", "").strip()
    scale = 1.0
    if s[-1:] in "kK":
        s, scale = s[:-1], 1e3
    elif s[-1:] in "mM":
        s, scale = s[:-1], 1e6
    return float(s) * scale


# --- basis ---------------------------------------------------------------------------------------

_SPOT = re.compile(rf"\bspot\s+(?:price\s+)?(?:is\s+|at\s+|=\s*|of\s+)?(?P<v>{_NUM})|"
                   rf"(?P<v2>{_NUM})\s+spot\b", re.I)
_FUTURE = re.compile(rf"\b(?:quarterly|dated|december|march|june|september|front[\s-]month)?\s*"
                     rf"(?:futures?|perp(?:etual)?s?)\s+(?:price\s+)?(?:is\s+|at\s+|=\s*|of\s+)?"
                     rf"(?P<v>{_NUM})|(?P<v2>{_NUM})\s+(?:futures?|perp)\b", re.I)
_DAYS_TO = re.compile(r"\b(?P<d>\d+)\s+days?\b", re.I)


def basis_lines(text: str) -> list[str] | None:
    if not re.search(r"\b(?:basis|carry|annuali[sz]ed|contango|backwardation|premium)\b", text,
                     re.I):
        return None
    # the spot price is the number after "spot" when one is written so ("BTC future at 59000,
    # spot 60000" read 59000 as spot, a round-26 check); "60000 spot" otherwise
    s = re.search(rf"\bspot\s+(?:price\s+)?(?:is\s+|at\s+|=\s*|of\s+)?(?P<v>{_NUM})", text, re.I) \
        or _SPOT.search(text)
    f, d = _FUTURE.search(text), _DAYS_TO.search(text)
    if s is None or f is None or d is None:
        return None
    spot, future, days = _n(s.group("v") or s.group("v2")), _n(f.group("v") or f.group("v2")), \
        int(d.group("d"))
    if spot <= 0:
        return None
    if days <= 0:
        # "Annualised basis over 0 days?" got a ticker (a hostile review, round 26)
        return [f"Bottom line: it cannot be annualised — with 0 days to expiry the future must "
                f"already equal spot, so a {future - spot:,.2f} gap is a price to check, not a "
                f"yield. Give the days left."]
    gap = future - spot
    period = gap / spot
    simple = period * 365 / days
    compound = (future / spot) ** (365 / days) - 1
    shape = "contango (the future above spot)" if gap > 0 else \
        "backwardation (the future below spot)" if gap < 0 else "flat"
    return [f"Bottom line: the basis is {gap:+,.2f} ({period:+.2%} of spot), {shape} — "
            f"{simple:+.1%} a year annualised simply ({period:+.2%} x 365 / {days}), "
            f"{compound:+.1%} compounded.",
            (f"That is the cash-and-carry yield: buy spot at {spot:,.2f}, sell the future at "
             f"{future:,.2f}, and the gap converges by expiry if both are held to it" if gap >= 0
             else "Reversed, it is what a short-spot, long-future position earns as the gap "
                  "closes; a long spot holder gives it up by holding spot instead")
            + " — before fees, funding on any perpetual leg, and the cost of the capital tied up."]


# --- funding across settlement times -------------------------------------------------------

_DAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
_AT = (r"(?:(?P<{p}day>mon|tue|wed|thu|fri|sat|sun)[a-z]*\s+)?(?P<{p}h>\d{{1,2}}):"
       r"(?P<{p}m>\d\d)\s*(?:utc)?")
_WINDOW = re.compile(
    r"\b(?:opened|open|from|entered|held\s+from|holding[^?]{0,60}?from)\b[^?]{0,40}?"
    + _AT.format(p="a") + r"[^?]{0,40}?\b(?:closed|close|to|until|till)\b[^?]{0,10}?"
    + _AT.format(p="b"), re.I)
_RATE = re.compile(r"(?P<sign>[+-]?)(?P<r>\d+(?:\.\d+)?)\s*%\s*(?:funding|(?:every|per|each)\s+"
                   r"(?P<h>\d+)\s*h)|funding\s+(?:rate\s+)?(?:of\s+|is\s+|at\s+)?(?P<sign2>[+-]?)"
                   r"(?P<r2>\d+(?:\.\d+)?)\s*%", re.I)
_NOTIONAL = re.compile(rf"(?P<v>{_NUM})\s*(?:notional|position|of\s+(?:btc|eth|\w+))|"
                       rf"(?P<v3>{_NUM})\s+(?:\w+\s+){{0,3}}?(?:perp|position|long|short)\b|"
                       rf"\bperp(?:etual)?\s+(?:of\s+|worth\s+)?(?P<v4>{_NUM})|"
                       rf"(?:notional|position|long|short)\s+(?:of\s+|worth\s+)?(?P<v2>{_NUM})",
                       re.I)


def _instant(day: str | None, hour: str, minute: str, anchor: date) -> datetime:
    when = anchor
    if day:
        target = next(i for i, d in enumerate(_DAYS) if d.startswith(day.lower()[:3]))
        when = anchor + timedelta(days=(target - anchor.weekday()) % 7)
    return datetime.combine(when, time(int(hour), int(minute)), tzinfo=UTC)


_LISTED_HOURS = re.compile(r"\b(?:at|settles?\s+at|paid\s+at)\s+(?P<l>\d{1,2}(?::00)?"
                           r"(?:\s*(?:,|and)\s*\d{1,2}(?::00)?)+)\s*(?:utc)?", re.I)


def _settle_hours(text: str, every: int) -> list[int]:
    listed = _LISTED_HOURS.search(text)
    if listed is not None:
        hours = sorted({int(h) for h in re.findall(r"\d{1,2}(?=(?::00)?\b)", listed.group("l"))
                        if int(h) < 24})
        if len(hours) >= 2:
            return hours
    return list(range(0, 24, every))


def funding_rate_lines(text: str) -> list[str] | None:
    """"Funding is 0.01% every 8 hours. What is that annualised?" was declined (a hostile
    review, round 26); "funding 0.01% every 8 hours for 3 days" on a stated notional was priced
    at the live rate."""
    r = _RATE.search(text)
    if r is None or not re.search(r"\bfunding\b", text, re.I) or _WINDOW.search(text):
        return None
    rate = float(r.group("r") or r.group("r2")) / 100
    if (r.group("sign") or r.group("sign2")) == "-":
        rate = -rate
    every_said = re.search(r"\b(?:every|per|each)\s+(?P<h>\d+)\s*(?:h|hours?)\b", text, re.I)
    every = int(every_said.group("h")) if every_said else 8
    per_day = rate * 24 / every
    span = re.search(r"\bfor\s+(?P<n>\d+)\s+(?P<u>days?|weeks?|hours?)\b", text, re.I)
    n = _NOTIONAL.search(text)
    if span is not None and n is not None:
        notional = _n(n.group("v") or n.group("v2") or n.group("v3") or n.group("v4"))
        unit = span.group("u").lower()
        hours = int(span.group("n")) * (24 if unit.startswith("day") else 168
                                        if unit.startswith("week") else 1)
        count = hours // every
        short = re.search(r"\bshort\b", text, re.I) is not None
        pays = (rate > 0) != short
        return [f"Bottom line: the {'short' if short else 'long'} "
                f"{'pays' if pays else 'receives'} about ${abs(rate) * notional * count:,.2f} — "
                f"{count} settlements in {span.group(0)[4:]} at one every {every} hours, each "
                f"{abs(rate):.4%} of ${notional:,.0f}.",
                f"At the rate you gave, held the whole time; the real rate resets every "
                f"{every} hours."]
    if re.search(r"\bannuali[sz]ed|\ba\s+year\b|\bper\s+year\b|\byearly\b|\bapr\b|\bapy\b", text,
                 re.I):
        simple = per_day * 365
        compound = (1 + rate) ** (365 * 24 / every) - 1
        return [f"Bottom line: {rate:+.4%} every {every} hours is {per_day:+.3%} a day, "
                f"{simple:+.2%} a year simple ({rate:+.4%} x {24 // every} x 365) and "
                f"{compound:+.2%} compounded.",
                "Paid by longs to shorts while positive; the rate resets every settlement, so "
                "a year at one rate is an illustration, not a forecast."]
    return None


def funding_window_lines(text: str) -> list[str] | None:
    if not re.search(r"\bfunding\b", text, re.I):
        return None
    w, r, n = _WINDOW.search(text), _RATE.search(text), _NOTIONAL.search(text)
    if w is None or r is None or n is None:
        return None
    anchor = date(2026, 1, 5)  # a Monday: the weekdays named are laid out from it
    opened = _instant(w.group("aday"), w.group("ah"), w.group("am"), anchor)
    closed = _instant(w.group("bday"), w.group("bh"), w.group("bm"), anchor)
    if closed <= opened:
        closed += timedelta(days=7 if w.group("bday") else 1)
    every = int(r.group("h")) if r.group("h") else 8
    settle_hours = _settle_hours(text, every)
    count, t = 0, opened.replace(minute=0) + timedelta(hours=1)
    while t <= closed:
        if t.hour in settle_hours and t > opened:
            count += 1
        t += timedelta(hours=1)
    rate = float(r.group("r") or r.group("r2")) / 100
    if (r.group("sign") or r.group("sign2")) == "-":
        rate = -rate
    notional = _n(n.group("v") or n.group("v2") or n.group("v3") or n.group("v4"))
    short = re.search(r"\bshort\b", text, re.I) is not None
    each = notional * rate
    total = each * count
    pays = (rate > 0) != short
    side = "short" if short else "long"
    if count == 0:
        return [f"Bottom line: neither — no settlement falls while the {side} is open "
                f"({opened:%a %H:%M} to {closed:%a %H:%M} UTC), so no funding changes hands."]
    return [f"Bottom line: the {side} {'pays' if pays else 'receives'} {count} funding "
            f"payment{'s' if count != 1 else ''} — {count} x {abs(rate):.4%} x ${notional:,.0f} = "
            f"${abs(total):,.2f}.",
            f"Funding is exchanged by whoever holds the position at each settlement "
            f"({', '.join(f'{h:02d}:00' for h in settle_hours)} UTC); held from "
            f"{opened:%a %H:%M} to {closed:%a %H:%M} UTC, the position is open at {count} of "
            f"them. A positive rate is paid by longs to shorts; the rate can change at each "
            f"settlement, so this assumes it stays at {rate:+.4%}."]


# --- time zones ------------------------------------------------------------------------------

_ZONES = {"new york": "America/New_York", "ny": "America/New_York", "et": "America/New_York",
          "est": "America/New_York", "edt": "America/New_York", "eastern": "America/New_York",
          "utc": "UTC", "gmt": "UTC", "london": "Europe/London", "mumbai": "Asia/Kolkata",
          "india": "Asia/Kolkata", "ist": "Asia/Kolkata", "delhi": "Asia/Kolkata",
          "tokyo": "Asia/Tokyo", "singapore": "Asia/Singapore", "hong kong": "Asia/Hong_Kong",
          "dubai": "Asia/Dubai", "sydney": "Australia/Sydney", "berlin": "Europe/Berlin",
          "paris": "Europe/Paris", "frankfurt": "Europe/Berlin", "chicago": "America/Chicago",
          "san francisco": "America/Los_Angeles", "los angeles": "America/Los_Angeles",
          "beijing": "Asia/Shanghai", "shanghai": "Asia/Shanghai", "seoul": "Asia/Seoul"}
_ZONE = "|".join(sorted((re.escape(z) for z in _ZONES), key=len, reverse=True))
_MONTHS = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")
_TIME_SAID = re.compile(
    rf"\b(?P<h>\d{{1,2}})(?::(?P<m>\d\d))?\s*(?P<ap>am|pm)?\s+(?:in\s+)?(?P<z>{_ZONE})"
    rf"(?:\s+time)?\b", re.I)
_DATE_SAID = re.compile(
    r"\b(?:(?:mon|tue|wed|thu|fri|sat|sun)[a-z]*,?\s+)?(?P<d>\d{1,2})(?:st|nd|rd|th)?\s+"
    r"(?P<mo>jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?(?:,?\s+(?P<y>(?:19|20)\d\d))?|"
    r"\b(?P<mo2>jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(?P<d2>\d{1,2})"
    r"(?:st|nd|rd|th)?(?:,?\s+(?P<y2>(?:19|20)\d\d))?", re.I)


def time_zone_lines(text: str, today: date | None = None) -> list[str] | None:
    said = _TIME_SAID.search(text)
    if said is None or not re.search(r"\bwhat\s+time\b|\bin\s+(?:" + _ZONE + r")\b[^?]*\?",
                                     text[said.end():], re.I):
        return None
    targets = [z.lower() for z in re.findall(rf"\b({_ZONE})\b", text[said.end():], re.I)]
    targets = [z for z in dict.fromkeys(targets) if _ZONES[z] != _ZONES[said.group("z").lower()]]
    if not targets:
        return None
    hour = int(said.group("h")) % 12 + (12 if (said.group("ap") or "").lower() == "pm" else 0) \
        if said.group("ap") else int(said.group("h"))
    day = today or datetime.now(UTC).date()
    # the date may sit anywhere in the question ("2pm ET on Wednesday 28 October 2026"); it was
    # ignored and today used, which also lost the daylight-saving change (a hostile review,
    # round 26)
    dated = _DATE_SAID.search(text)
    if dated is not None:
        month = _MONTHS.index((dated.group("mo") or dated.group("mo2"))[:3].lower()) + 1
        year_said = dated.group("y") or dated.group("y2")
        number_said = int(dated.group("d") or dated.group("d2"))
        year = int(year_said) if year_said else (
            day.year if (month, number_said) >= (day.month, day.day) else day.year + 1)
        try:
            day = date(year, month, number_said)
        except ValueError:
            return [f"Bottom line: {number_said} {dated.group('mo') or dated.group('mo2')} "
                    f"{year} is not a date on the calendar, so there is no time to convert."]
    source = ZoneInfo(_ZONES[said.group("z").lower()])
    at = datetime.combine(day, time(hour, int(said.group("m") or 0)), tzinfo=source)
    converted = [f"{z.title() if len(z) > 3 else z.upper()} "
                 f"{at.astimezone(ZoneInfo(_ZONES[z])):%H:%M on %a %d %b}" for z in targets]
    offset = at.strftime("%z")
    return [f"Bottom line: {at:%H:%M} {said.group('z')} time on {at:%a %d %b %Y} is "
            + " and ".join(converted) + ".",
            f"Offsets for that date from the IANA time-zone database, so daylight saving in "
            f"force on {at:%d %b} is applied ({said.group('z')} is UTC{offset[:3]}:{offset[3:]} "
            f"then)."]


# --- futures hedges, factor stress, price paths ---------------------------------------------------

_FUTURE_SPECS = {"ES": ("the S&P 500", 50.0), "MES": ("the S&P 500", 5.0),
                 "NQ": ("the Nasdaq-100", 20.0), "MNQ": ("the Nasdaq-100", 2.0)}
_BOOK_VALUE = re.compile(rf"(?P<v>{_NUM})\s+(?:\w+\s+){{0,2}}?(?:book|portfolio|of\s+stock|stock|"
                         rf"account|position)|(?:book|portfolio|account)\s+(?:is\s+|of\s+)?"
                         rf"(?P<v2>{_NUM})", re.I)
_BETA_SAID = re.compile(r"\bbeta\s+(?:of\s+|is\s+|=\s*)?(?P<b>-?\d+(?:\.\d+)?)", re.I)


def futures_hedge_lines(text: str) -> list[str] | None:
    """"$500k portfolio, beta 1.2. How many ES at 5000 to hedge? And MES?", "Hedge a $1M book with
    beta -0.5 to the S&P using ES at 5000" (answered with the console's own QQQ hedge, on the wrong
    side; a hostile review, round 26). Contracts = beta x value / (index level x multiplier), CME
    multipliers; a positive beta is hedged by selling, a negative one by buying."""
    if not re.search(r"\bhedg", text, re.I):
        return None
    value_m, beta_m = _BOOK_VALUE.search(text), _BETA_SAID.search(text)
    codes = re.findall(r"\b(ES|MES|NQ|MNQ)\b", text)
    level_m = re.search(r"\b(?:ES|MES|NQ|MNQ)(?:\s+futures?)?\s+(?:at|@|is\s+at|=)\s*"
                        r"(?P<l>\d[\d,]*(?:\.\d+)?)", text)
    if value_m is None or beta_m is None or not codes or level_m is None:
        return None
    value = _n(value_m.group("v") or value_m.group("v2"))
    beta = float(beta_m.group("b"))
    level = float(level_m.group("l").replace(",", ""))
    if value <= 0 or level <= 0:
        return None
    fraction = 0.5 if re.search(r"\bhalf\b|\b50\s*%", text, re.I) else 1.0
    exposure = beta * value * fraction
    side = "sell" if exposure > 0 else "buy"
    rows = []
    for code in dict.fromkeys(codes):
        _index, mult = _FUTURE_SPECS[code]
        count = abs(exposure) / (level * mult)
        rows.append(f"{side} {count:,.1f} {code} ({abs(exposure):,.0f} / ({level:,.0f} x "
                    f"${mult:g}))")
    return [f"Bottom line: {' or '.join(rows)} — the book's {beta:+g} beta on ${value:,.0f}"
            + (" hedged by half" if fraction < 1 else "")
            + f" is {'-' if exposure < 0 else '+'}${abs(exposure):,.0f} of index "
            f"exposure, and a {'positive' if exposure > 0 else 'negative'} exposure is offset by "
            f"{'selling' if exposure > 0 else 'buying'} futures.",
            "Round to whole contracts (or mix in micros for the remainder); the hedge holds only "
            "as long as the stated beta does, and it covers the market's part of the moves, not "
            "the book's own."]


_LOADING = re.compile(r"(?:beta\s+(?:of\s+)?)?(?P<b>-?\d+(?:\.\d+)?)\s+(?:beta\s+)?"
                      r"(?:to|on|vs\.?)\s+(?P<f>[A-Za-z&\d]{2,12})", re.I)
_SHOCK = re.compile(r"\b(?P<f>[A-Za-z&\d]{2,12})\s+(?:falls?|drops?|is\s+down|down|declines?|"
                    r"rises?|is\s+up|up|gains?)?\s*(?:by\s+)?(?P<s>[-+\u2212]?\d+(?:\.\d+)?)\s*%",
                    re.I)
_FACTOR_ALIAS = {"SPX": "S&P", "S&P": "S&P", "SP500": "S&P", "SPY": "S&P", "ES": "S&P",
                 "BTC": "BTC", "BITCOIN": "BTC", "NASDAQ": "NDX", "NDX": "NDX", "QQQ": "NDX",
                 "ETH": "ETH", "GOLD": "GOLD"}


def stated_factor_stress_lines(text: str) -> list[str] | None:
    """"$200k book, beta 1.2 to SPX and 0.5 to BTC. SPX falls 10% and BTC falls 20%. Loss?" lost
    the SPX leg and treated BTC as the whole book (a hostile review, round 26). With loadings and
    shocks stated, the move is the sum of each loading times its factor's shock."""
    value_m = _BOOK_VALUE.search(text)
    loads: dict[str, float] = {_FACTOR_ALIAS[m.group("f").upper()]: float(m.group("b"))
                               for m in _LOADING.finditer(text)
                               if m.group("f").upper() in _FACTOR_ALIAS}
    if value_m is None or len(loads) < 2:
        return None
    shocks: dict[str, float] = {}
    for m in _SHOCK.finditer(text):
        factor_name = _FACTOR_ALIAS.get(m.group("f").upper())
        if factor_name is None or factor_name not in loads:
            continue
        size = float(m.group("s").replace("\u2212", "-"))
        down = re.search(r"falls?|drops?|down|declines?", m.group(0), re.I) is not None
        shocks[factor_name] = -abs(size) / 100 if down or size < 0 else size / 100
    if set(shocks) != set(loads):
        return None
    value = _n(value_m.group("v") or value_m.group("v2"))
    move = sum((loads[f] * shocks[f] for f in loads), 0.0)
    parts = " + ".join(f"{loads[f]:g} x {shocks[f]:+.0%} ({f})" for f in loads)
    return [f"Bottom line: {'a loss' if move < 0 else 'a gain'} of about ${abs(value * move):,.0f} "
            f"({move:+.2%} of ${value:,.0f}) — {parts} = {move:+.2%}.",
            "Each factor's loading times its shock, added: the loadings you gave, applied as "
            "stated; anything the factors do not explain is left out."]


_PATH_POINT = re.compile(
    rf"(?:\b(?:was|at|closed\s+(?:at\s+)?|is|of)\s+|(?<![\w.$,]))(?P<p>{_NUM})\s+(?:on|by)\s+"
    rf"(?P<when>(?:\d{{1,2}}\s+)?"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*(?:\s+\d{1,2})?)", re.I)


def price_path_lines(text: str) -> list[str] | None:
    """Returns over a path of prices the question states: "100 on Jan 1, 120 on Jun 30, 90 on Dec
    31", "bought at 100, sold at 80, then bought at 80 and sold at 100", "closed at 100 before
    earnings, opened at 90 the next day and closed at 95", "closed 230 the day before and 220 on
    the day". Each was misrouted (a hostile review, round 26)."""
    if not re.search(r"\b(?:return|reaction|move|moved|gain|loss|change|performance|up|down)\b",
                     text, re.I):
        return None
    points = [(m.group("when").strip(), _n(m.group("p"))) for m in _PATH_POINT.finditer(text)]
    if len(points) >= 2 and all(p > 0 for _w, p in points):
        steps = [f"{a[0]} to {b[0]}: {b[1] / a[1] - 1:+.2%}" for a, b in pairwise(points)]
        total = points[-1][1] / points[0][1] - 1
        return [f"Bottom line: {points[0][0]} to {points[-1][0]}: {total:+.2%} in all — "
                + "; ".join(steps) + ".",
                "Price returns from the prices you gave, without dividends."]
    trades = re.findall(rf"\bbought\s+(?:at\s+)?(?P<b>{_NUM})[^.;]*?\bsold\s+(?:at\s+)?"
                        rf"(?P<s>{_NUM})", text, re.I)
    if len(trades) >= 2:
        legs = [_n(s) / _n(b) - 1 for b, s in trades]
        growth = 1.0
        for leg in legs:
            growth *= 1 + leg
        return [f"Bottom line: {growth - 1:+.2%} in all — "
                + "; ".join(f"trade {i}: {leg:+.2%}" for i, leg in enumerate(legs, 1))
                + " — returns compound, so a fall and an equal-looking rise do not cancel.",
                "Each trade is sell over buy, less one; the total multiplies them."]
    gap = re.search(rf"\bclosed\s+(?:at\s+)?(?P<c0>{_NUM})\s+(?:the\s+day\s+)?before[^.;]*?"
                    rf"\bopened\s+(?:at\s+)?(?P<o>{_NUM})[^.;]*?\bclosed\s+(?:at\s+)?"
                    rf"(?P<c1>{_NUM})", text, re.I)
    if gap is not None:
        c0, o, c1 = _n(gap.group("c0")), _n(gap.group("o")), _n(gap.group("c1"))
        return [f"Bottom line: it gapped {o / c0 - 1:+.2%} at the open and closed "
                f"{c1 / c0 - 1:+.2%} on the day ({c0:,.2f} to {o:,.2f}, then {c1:,.2f}); the "
                f"move inside the day was {c1 / o - 1:+.2%}.",
                "From the prices you gave: the gap is open over the prior close, the day's "
                "reaction close over close."]
    day = re.search(rf"\bclosed\s+(?:at\s+)?(?P<a>{_NUM})\s+the\s+day\s+before\s+and\s+(?:at\s+)?"
                    rf"(?P<b>{_NUM})\s+on\s+the\s+day", text, re.I)
    if day is not None:
        a, b = _n(day.group("a")), _n(day.group("b"))
        return [f"Bottom line: {b / a - 1:+.2%} on the day ({a:,.2f} to {b:,.2f}), close to "
                f"close."]
    return None


# --- a hedge from a stated beta -------------------------------------------------------------------

_STATED_BETA = re.compile(
    rf"\b(?:long|short|holding|hold|have|own|bought)\s+(?P<v>{_NUM})\s+(?:of\s+|in\s+)?(?P<a>[A-Za-z]{{2,10}})"
    rf"\b[^?]{{0,80}}?\bbeta\s+(?:to|vs\.?|versus|against|on)\s+(?P<b>[A-Za-z]{{2,10}})\s+"
    rf"(?:is|of|=|at)\s+(?P<beta>-?\d+(?:\.\d+)?)", re.I)


def stated_beta_hedge_lines(text: str) -> list[str] | None:
    m = _STATED_BETA.search(text)
    if m is None or not re.search(r"\bhedge|\bshort\b[^?]*\bhow\s+much|\bhow\s+much\b", text,
                                  re.I):
        return None
    value, beta = _n(m.group("v")), float(m.group("beta"))
    asset, base = m.group("a").upper(), m.group("b").upper()
    hedge = value * beta
    short_side = re.search(rf"\bshort\s+{re.escape(m.group('v'))}", text, re.I) is not None
    return [f"Bottom line: {'buy' if short_side else 'short'} about ${abs(hedge):,.0f} of {base} "
            f"— ${value:,.0f} of {asset} x the beta of {beta:g} you gave.",
            f"A beta hedge offsets the part of {asset}'s moves that follows {base}; whatever "
            f"{asset} does on its own news stays unhedged, and a beta measured in calm markets "
            f"often rises in a selloff. Ask for {asset}'s beta to {base} to see the measured one."]


trace_module(globals())
