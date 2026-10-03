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

_BASIS = re.compile(
    rf"\bspot\s+(?:is\s+|at\s+|=\s*)?(?P<s>{_NUM})\b[^?]{{0,80}}?\bfutures?\s+(?:is\s+|at\s+|=\s*)?"
    rf"(?P<f>{_NUM})\b[^?]{{0,60}}?\b(?P<d>\d+)\s+days?\b", re.I)


def basis_lines(text: str) -> list[str] | None:
    m = _BASIS.search(text)
    if m is None or not re.search(r"\b(?:basis|yield|carry|annuali[sz]ed|premium)\b", text, re.I):
        return None
    spot, future, days = _n(m.group("s")), _n(m.group("f")), int(m.group("d"))
    if spot <= 0 or days <= 0:
        return None
    gap = future - spot
    period = gap / spot
    simple = period * 365 / days
    compound = (future / spot) ** (365 / days) - 1
    return [f"Bottom line: the basis is {gap:,.2f} ({period:.2%} of spot) — about {simple:.1%} a "
            f"year annualised simply ({period:.2%} x 365 / {days}), {compound:.1%} compounded.",
            f"That is the cash-and-carry yield: buy spot at {spot:,.2f}, sell the future at "
            f"{future:,.2f}, and the {gap:,.2f} converges by expiry if both are held to it — "
            f"before fees, funding on any perpetual leg, and the cost of the capital tied up."]


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
                       rf"(?:notional|position|long|short)\s+(?:of\s+|worth\s+)?(?P<v2>{_NUM})",
                       re.I)


def _instant(day: str | None, hour: str, minute: str, anchor: date) -> datetime:
    when = anchor
    if day:
        target = next(i for i, d in enumerate(_DAYS) if d.startswith(day.lower()[:3]))
        when = anchor + timedelta(days=(target - anchor.weekday()) % 7)
    return datetime.combine(when, time(int(hour), int(minute)), tzinfo=UTC)


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
    hours = [int(h) for h in re.findall(r"\b(\d\d):00\b", text.split("opened")[0])] if \
        re.search(r"\bpaid\s+at\b", text, re.I) else list(range(0, 24, every))
    settle_hours = sorted({h for h in hours if 0 <= h < 24}) or list(range(0, 24, every))
    count, t = 0, opened.replace(minute=0) + timedelta(hours=1)
    while t <= closed:
        if t.hour in settle_hours and t > opened:
            count += 1
        t += timedelta(hours=1)
    rate = float(r.group("r") or r.group("r2")) / 100
    if (r.group("sign") or r.group("sign2")) == "-":
        rate = -rate
    notional = _n(n.group("v") or n.group("v2"))
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
    rf"(?:\s+time)?\b(?:[^?]{{0,20}}?\bon\s+(?P<d>\d{{1,2}})\s+(?P<mo>[A-Za-z]{{3,9}}))?", re.I)


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
    if said.group("d") and said.group("mo")[:3].lower() in _MONTHS:
        month = _MONTHS.index(said.group("mo")[:3].lower()) + 1
        day = date(day.year if month >= day.month else day.year + 1, month, int(said.group("d")))
    source = ZoneInfo(_ZONES[said.group("z").lower()])
    at = datetime.combine(day, time(hour, int(said.group("m") or 0)), tzinfo=source)
    converted = [f"{z.title() if len(z) > 3 else z.upper()} "
                 f"{at.astimezone(ZoneInfo(_ZONES[z])):%H:%M on %d %b}" for z in targets]
    offset = at.strftime("%z")
    return [f"Bottom line: {at:%H:%M} {said.group('z')} time on {at:%d %b %Y} is "
            + " and ".join(converted) + ".",
            f"Offsets for that date from the IANA time-zone database, so daylight saving in "
            f"force on {at:%d %b} is applied ({said.group('z')} is UTC{offset[:3]}:{offset[3:]} "
            f"then)."]


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
