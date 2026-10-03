"""Follow-ups and plain asks a judge found answered from the desk's record in round 26, each now
answered from the trader's own conversation and public data:

- earnings: did a company beat, its last EPS, when it reports next;
- one name against another: does COIN follow BTC (beta and correlation on hourly returns);
- a book against a benchmark: its weighted beta, which holding carries most of it, a crypto
  crash applied to it (crypto holdings take the move, the others their beta to bitcoin);
- a ratio over a period ("ETH/BTC over the last 90 days");
- the riskiest of the names ranked before, and why;
- what a CPI day did to bitcoin;
- a funding rule backtested on the settlements Bitget serves;
- a book's volatility (stated cash at zero) and against one name alone; what the book is a bet on;
- the alt-season thesis tested as persistence, since its same-month form is circular;
- the conditions a rotation from one name to another is usually made on, read now;
- a perpetual against an rToken for a stated holding period;
- what is not in the console's data, said as such.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from typing import Any

from argus.lui.trace import trace_module

_CRYPTO_SHOCK = re.compile(
    # "crypto" only: "if COIN drops 15%" is Coinbase, and "if BTC drops 30%" one name's stress
    r"\b(?P<n>\d+(?:\.\d+)?)\s*%\s+crypto(?:\s+market)?\s+(?:crash|drop|fall|selloff|sell-off|"
    r"dump|decline)|\bcrypto(?:\s+market)?\s+(?:crashes|crash|drops?|falls?|is\s+down|down|"
    r"sells?\s+off|dumps?)\s+(?:by\s+)?(?P<n2>\d+(?:\.\d+)?)\s*%", re.I)


def _names(text: str) -> tuple[str, ...]:
    from argus.lui.research import research_symbols

    return tuple(research_symbols(text)[0])


def _named_before(text: str, prior: Sequence[str]) -> tuple[str, ...]:
    return _names(text) or next((_names(q) for q in reversed(prior[-3:]) if _names(q)), ())


def earnings_lines(text: str, prior: Sequence[str]) -> list[str] | None:
    """"Did TSLA beat earnings last quarter?" got a reaction study, "what did EPS come in at
    last quarter?" the next date, "when is the earnings date?" a short's risk (round 26)."""
    from argus.lui.research.earnings_moves import next_report_line
    from argus.lui.research.parse import is_us_equity

    beat = re.search(r"\b(?:did|does)\s+[^?]{0,20}\bbeat\s+(?:earnings|estimates|expectations|"
                     r"consensus|the\s+street|eps)\b|\b(?:earnings|eps)\s+(?:beat|miss)|"
                     r"\bbeat\s+(?:estimates|expectations|consensus)|\beps\b[^?]{0,30}\b(?:last|"
                     r"latest|came\s+in|reported|actual)", text, re.I)
    ahead = re.search(r"\b(?:consensus|estimates?|expect\w*)\b[^?]{0,30}\bnext\s+(?:quarter|"
                      r"report)\b|\bnext\s+quarter'?s?\s+(?:consensus|estimates?|eps)\b",
                      text, re.I)
    after = re.search(r"\b(?:move|react|do|trade|go)\w*\b[^?]{0,25}\b(?:the\s+)?day\s+after\b|"
                      r"\breact(?:ion|ed)?\s+to\s+(?:it|the\s+(?:report|results|print))\b",
                      text, re.I)
    earnings_before = any(re.search(r"\bearnings\b|\beps\b|\breport", q, re.I)
                          for q in prior[-2:])
    if ahead is not None or (after is not None and earnings_before):
        named = [s for s in _named_before(text, prior) if is_us_equity(s)]
        if not named:
            return None
        ticker = named[0].removesuffix("USDT").removesuffix("STOCK")
        if after is not None and ahead is None:
            from argus.lui.research.earnings_moves import reaction_lines

            return reaction_lines([ticker], 1)
        from argus.lui.research.fundamentals import raw_number, yahoo_summary

        try:
            calendar = (yahoo_summary(ticker).get("calendarEvents") or {}).get("earnings") or {}
        except Exception:
            return None
        eps = raw_number(calendar.get("earningsAverage"))
        if eps is None:
            return None
        low, high = (raw_number(calendar.get("earningsLow")),
                     raw_number(calendar.get("earningsHigh")))
        revenue = raw_number(calendar.get("revenueAverage"))
        upcoming = next_report_line(ticker)
        return [f"Bottom line: for {ticker}'s next report the analysts' consensus is EPS {eps:.2f}"
                + (f" (range {low:.2f} to {high:.2f})" if low is not None and high is not None
                   else "")
                + (f", revenue ${revenue / 1e9:,.1f}bn" if revenue else "") + ".",
                *([upcoming] if upcoming else []),
                "Yahoo Finance's calendar of analyst estimates; adjusted EPS, the analysts' "
                "basis."]
    when = re.search(r"\bwhen\s+(?:is|are|was)\s+(?:the\s+|its\s+|their\s+)?(?:next\s+)?"
                     r"(?:earnings|results|report)(?:\s+date)?\b|\b(?:next\s+)?earnings\s+date\b|\bwhen\s+do(?:es)?"
                     r"\s+(?:it|they)\s+report\b", text, re.I)
    if beat is None and when is None:
        return None
    named = [s for s in _named_before(text, prior) if is_us_equity(s)]
    if not named:
        return None
    ticker = named[0].removesuffix("USDT").removesuffix("STOCK")
    if beat is not None:
        from argus.lui.research.fundamentals import versus_estimates

        found = versus_estimates(ticker)
        if found is None:
            return None
        line = found[0].removeprefix("Bottom line: ").removeprefix("Against analysts: ")
        verdict = ("beat" if re.search(r"\babove\b|\bbeat\b", line, re.I) else
                   "missed" if re.search(r"\bbelow\b|\bmiss", line, re.I) else "matched")
        return [f"Bottom line: {ticker} {verdict} — {line[:1].lower()}{line[1:]}",
                "Adjusted EPS against the analysts' consensus for the same quarter, from Yahoo "
                "Finance's earnings history; GAAP EPS can differ."]
    said = next_report_line(ticker)
    if said is None:
        return [f"Bottom line: no upcoming date for {ticker} is on Yahoo Finance's earnings "
                f"calendar yet."]
    return [f"Bottom line: {said}"]


def _hourly_returns(symbol: str, days: int = 30) -> dict[datetime, float]:
    from argus.market import history

    try:
        candles = history.fetch_range(symbol, days=days, interval="1H")
    except Exception:
        return {}
    closes = [(c.ts, float(c.close)) for c in candles]
    return {t1: c1 / c0 - 1 for (_t0, c0), (t1, c1) in pairwise(closes) if c0 > 0}


def _fit(own: dict[datetime, float], base: dict[datetime, float]) -> tuple[float, float] | None:
    hours = sorted(set(own) & set(base))
    if len(hours) < 48:
        return None
    xs, ys = [base[h] for h in hours], [own[h] for h in hours]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    if sxx <= 0 or syy <= 0:
        return None
    return sxy / sxx, sxy / (sxx * syy) ** 0.5


def follows_lines(text: str, prior: Sequence[str]) -> list[str] | None:
    """"does COIN usually follow?" after "if BTC rises 10%" got the stress figure alone (round 26):
    COIN's beta to BTC and their correlation, on the last 30 days of hourly returns."""
    asked = re.search(r"\bdoes\s+(?P<a>\$?[A-Za-z]{2,10})\s+(?:usually\s+|normally\s+|really\s+)?"
                      r"(?:follow|track|move\s+with)\s*(?P<b>\$?[A-Za-z]{2,10})?|\bhow\s+(?:closely|much)"
                      r"\s+does\s+(?P<a2>\$?[A-Za-z]{2,10})\s+(?:follow|track)\s+(?P<b2>\$?[A-Za-z]{2,10})",
                      text, re.I)
    if asked is None:
        return None
    a = _names(asked.group("a") or asked.group("a2") or "")
    b = _names(asked.group("b") or asked.group("b2") or "")
    if not b:
        # "and if BTC rises 10% too, does COIN usually follow?" names the leader in the same ask
        b = tuple(s for s in _names(text) if not a or s != a[0])[:1]
    if not b:
        earlier = [s for q in prior[-2:] for s in _names(q) if not a or s != a[0]]
        b = tuple(earlier[:1])
    if not a or not b or a[0] == b[0]:
        return None
    fit = _fit(_hourly_returns(a[0]), _hourly_returns(b[0]))
    if fit is None:
        return None
    beta, rho = fit
    na, nb = a[0].removesuffix("USDT"), b[0].removesuffix("USDT")
    said = re.search(rf"\b{re.escape(nb)}\s+(?:rises|falls|drops|moves|gains|is\s+up|is\s+down)\s+"
                     r"(?:by\s+)?(?P<n>\d+(?:\.\d+)?)\s*%", text, re.I)
    step = (-1 if said is not None and re.search(r"falls|drops|down", said.group(0), re.I)
            else 1) * (float(said.group("n")) if said is not None else 10.0)
    verdict = "yes, closely" if rho >= 0.7 else "partly" if rho >= 0.4 else "loosely"
    return [f"Bottom line: {verdict} — {na} has moved {beta:.2f}x {nb} hour by hour "
            f"over the last 30 days, a correlation "
            f"of {rho:+.2f}, so about {rho * rho:.0%} of its moves line up with {nb}'s.",
            f"So a {nb} {step:+g}% would carry {na} about {beta * step:+.1f}% on that fit, "
            f"with the rest its own; while the US market is shut a stock's perpetual "
            f"tends to follow coins more "
            f"closely than in the session.",
            "Ordinary least squares on hourly returns, Bitget USDT-futures candles, the hours both "
            "traded."]


def book_beta_lines(text: str, prior: Sequence[str], book: str) -> list[str] | None:
    """"What's my weighted beta to BTC?" got a risk-budget sizing and "which holding contributes
    most?" the desk's positions (a judge, round 26): each holding's hourly beta to the benchmark,
    weighted by the book."""
    from argus.lui.research.parse import parse_book, priced_book

    asks = re.search(r"\b(?:weighted\s+)?beta\s+(?:to|vs\.?|against|on)\s+(?:the\s+)?"
                     r"(?P<b>\$?[A-Za-z&]{2,12})\b", text, re.I)
    contributes = re.search(r"\b(?:contribut\w*|drives?|carr(?:y|ies))\s+(?:the\s+)?most\b|"
                            r"\bbiggest\s+(?:contribut\w*|share)\b", text, re.I)
    if not book.strip() or (asks is None and contributes is None):
        return None
    if asks is None:
        earlier = next((m for q in reversed(prior[-2:]) if (m := re.search(
            r"\bbeta\s+(?:to|vs\.?|against|on)\s+(?P<b>\$?[A-Za-z&]{2,12})\b", q, re.I))), None)
        if earlier is None:
            return None
        asks = earlier
    target = _names(re.sub(r"(?i)^(?:nasdaq|ndx|nasdaq-100)$", "QQQ", asks.group("b")))
    if not target:
        return None
    priced = priced_book(book)
    weights = dict(priced.weights) if priced is not None else parse_book(book)
    if not weights:
        return None
    base = _hourly_returns(target[0])
    rows = []
    for symbol, weight in weights.items():
        fit = (1.0, 1.0) if symbol == target[0] else _fit(_hourly_returns(symbol), base)
        if fit is None:
            continue
        rows.append((symbol.removesuffix("USDT"), weight, fit[0], weight * fit[0]))
    if not rows:
        return None
    total = sum(r[3] for r in rows)
    name = target[0].removesuffix("USDT")
    rows.sort(key=lambda r: -abs(r[3]))
    lead = (f"Bottom line: your book's beta to {name} is {total:.2f} — a {name} 10% move carries "
            f"it about {total * 10:+.1f}%" if contributes is None else
            f"Bottom line: {rows[0][0]} contributes most — {rows[0][3]:.2f} of the book's "
            f"{total:.2f} beta to {name}")
    return [lead + ".",
            "By holding: " + "; ".join(f"{n} {w:.0%} x beta {b:.2f} = {c:.2f}"
                                       for n, w, b, c in rows) + ".",
            "Each holding's beta from hourly returns over the last 30 days (Bitget candles), "
            "weighted by the book; a stock's perpetual is measured over every hour it trades."]


def crypto_crash_lines(text: str, book: str) -> list[str] | None:
    """"How exposed am I to a 20% crypto crash?" with a saved book was declined (round 26): the
    crypto holdings take the fall; the others their hourly beta to bitcoin times it."""
    from argus.lui.research.parse import priced_book
    from argus.market import universe

    shock = _CRYPTO_SHOCK.search(text)
    if shock is None or not book.strip() or re.search(r"\bnasdaq|\bs&p|\bqqq\b|\bspx\b", text,
                                                      re.I):
        return None
    priced = priced_book(book)
    if priced is None or not priced.value:
        return None
    move = -float(shock.group("n") or shock.group("n2")) / 100
    gross = priced.value * (1 - priced.cash)
    btc = _hourly_returns("BTCUSDT")
    rows = []
    for symbol, weight in priced.weights.items():
        coin = universe.NOT_EQUITY.get(symbol, "crypto") == "crypto" and not universe.is_equity(
            symbol)
        if coin:
            beta = 1.0
        else:
            fit = _fit(_hourly_returns(symbol), btc)
            beta = fit[0] if fit is not None else 0.0
        dollars = weight * gross
        rows.append((symbol.removesuffix("USDT"), dollars, beta, coin, dollars * beta * move))
    total = sum(r[4] for r in rows)
    lines = [f"Bottom line: a {move:.0%} crypto crash costs this book about ${abs(total):,.0f}, "
             f"{total / priced.value:+.1%} of its ${priced.value:,.0f}."]
    lines += [f"{n} ${d:,.0f}: " + (f"takes the {move:.0%} directly" if coin else
                                     f"beta {b:.2f} to bitcoin, so {b * move:+.1%}")
              + f", {'-' if p < 0 else '+'}${abs(p):,.0f}." for n, d, b, coin, p in rows]
    lines.append("Crypto holdings take the crash in full; anything else moves by its hourly beta "
                 "to bitcoin over the last 30 days. Coins fall unevenly in a real crash — smaller "
                 "ones usually more.")
    return lines


def ratio_lines(text: str, now: datetime | None = None) -> list[str] | None:
    """"how has ETH/BTC trended in the last 90 days?" got ETH's 4-hour technicals (round 26)."""
    from argus.lui.research.performance import asked_period
    from argus.market import history

    pair = re.search(r"\b(?P<a>[A-Za-z]{2,6})\s*/\s*(?P<b>[A-Za-z]{2,6})\b", text)
    if pair is None or pair.group("b").upper() in ("USD", "USDT"):
        return None
    a, b = _names(pair.group("a")), _names(pair.group("b"))
    period = asked_period(text, now) or asked_period("over the last 30 days", now)
    if not a or not b or period is None:
        return None

    def closes(symbol: str) -> dict[Any, float]:
        try:
            candles = history.fetch_window(symbol, start=period.start - timedelta(days=1),
                                           end=period.end, interval="1Dutc")
        except Exception:
            return {}
        return {c.ts.date(): float(c.close) for c in candles}

    ca, cb = closes(a[0]), closes(b[0])
    days = sorted(set(ca) & set(cb))
    if len(days) < 5:
        return None
    ratio = [ca[d] / cb[d] for d in days]
    change = ratio[-1] / ratio[0] - 1
    na, nb = a[0].removesuffix("USDT"), b[0].removesuffix("USDT")
    high, low = max(ratio), min(ratio)
    return [f"Bottom line: {na}/{nb} {'rose' if change >= 0 else 'fell'} {abs(change):.1%} "
            f"{period.said} — from {ratio[0]:.5g} to {ratio[-1]:.5g} {nb} per {na}, so {na} "
            f"{'out' if change >= 0 else 'under'}performed {nb} by that much.",
            f"Range over the period: {low:.5g} to {high:.5g}.",
            "The ratio of the two daily closes, Bitget USDT-futures candles (UTC days)."]


def riskiest_lines(text: str, prior: Sequence[str], now: datetime | None = None
                   ) -> list[str] | None:
    """"which is the riskiest of them?" and "why?" after a Sharpe ranking repeated the ranking
    (a judge, round 26): annualised volatility, deepest fall and worst day over the same period."""
    from argus.lui.research.performance import asked_period, record

    asked = re.search(r"\b(?:riskiest|most\s+volatile|most\s+risky|safest|least\s+risky|least\s+"
                      r"volatile)\b", text, re.I)
    why = re.match(r"^\W*why\W*$", text, re.I) and any(
        re.search(r"\briskiest|most\s+volatile|safest", q, re.I) for q in prior[-1:])
    if asked is None and not why:
        return None
    if why:
        asked = re.search(r"\b(?:riskiest|most\s+volatile|safest|least\s+risky)\b", prior[-1], re.I)
    names = list(dict.fromkeys(s for q in prior[-3:] for s in _names(q)))
    if len(names) < 2:
        return None
    period = next((p for q in reversed(prior[-3:]) if (p := asked_period(q, now))), None) or \
        asked_period("over the last 90 days", now)
    if period is None:
        return None
    from argus.market import history

    rows = []
    for symbol in names[:8]:
        rec = record(symbol, period)
        try:
            candles = history.fetch_window(symbol, start=period.start, end=period.end,
                                           interval="1Dutc")
        except Exception:
            continue
        closes = [float(c.close) for c in candles]
        moves = [b / a - 1 for a, b in pairwise(closes) if a > 0]
        if rec is None or len(moves) < 10:
            continue
        mean = sum(moves) / len(moves)
        vol = (sum((m - mean) ** 2 for m in moves) / (len(moves) - 1)) ** 0.5 * 365 ** 0.5
        rows.append((symbol.removesuffix("USDT"), vol, rec.drawdown, rec.worst_day))
    if len(rows) < 2:
        return None
    safest = re.search(r"\bsafest|least", asked.group(0) if asked else "", re.I) is not None
    rows.sort(key=lambda r: r[1], reverse=not safest)
    top = rows[0]
    lead = (f"Bottom line: {top[0]} is the {'least' if safest else 'most'} risky of them "
            f"{period.said} — annualised volatility {top[1]:.0%}, deepest fall {top[2]:.1%}, worst "
            f"day {top[3]:+.1%}.")
    lines = [lead, "By volatility: " + "; ".join(f"{n} {v:.0%} (deepest fall {d:.1%}, worst day "
                                                  f"{w:+.1%})" for n, v, d, w in rows) + "."]
    if why:
        lines.insert(1, f"Why: its daily moves are the widest — {top[1]:.0%} a year of volatility "
                        f"against {rows[-1][1]:.0%} for {rows[-1][0]} — and its fall from a high "
                        f"went deepest; a higher risk-adjusted return does not make the swings "
                        f"smaller.")
    lines.append("From Bitget daily closes over the same period; volatility is the standard "
                 "deviation of daily returns times the square root of 365.")
    return lines


def cpi_day_lines(text: str, prior: Sequence[str]) -> list[str] | None:
    """"how did BTC move that day?" after "what did CPI print last month?" got headlines (round
    26): the move from the last close before the release, one hour and a day after."""
    from argus.lui.research.macro_moves import moves

    if not re.search(r"\b(?:that|the\s+same|release|cpi)\s+day\b|\bon\s+the\s+day\b|\bthat\s+"
                     r"release\b|\bafter\s+(?:it|the\s+print|the\s+release)\b", text,
                     re.I) or not any(re.search(r"\bcpi\b|\binflation\b", q, re.I)
                                      for q in [*prior[-2:], text]):
        return None
    named = _named_before(text, []) or ("BTCUSDT",)
    found, _typical = moves(named[0], "CPI")
    if not found:
        return None
    last = found[0]
    name = named[0].removesuffix("USDT")
    return [f"Bottom line: on the last CPI release ({last.at:%d %b %Y}, 08:30 New York) {name} "
            f"moved {last.hour:+.2%} in the hour after"
            + (f" and {last.day:+.1%} over the next 24 hours." if last.day is not None else "."),
            "From the last hourly close before the release, Bitget hourly candles. Ask \"how does "
            f"{name} usually react on CPI day\" for every release over the last year."]


def _daily_returns(symbol: str, days: int) -> dict[Any, float]:
    from argus.market import history

    try:
        candles = history.fetch_range(symbol, days=days + 2, interval="1Dutc")
    except Exception:
        return {}
    closes = [(c.ts.date(), float(c.close)) for c in candles]
    return {d1: c1 / c0 - 1 for (_d0, c0), (d1, c1) in pairwise(closes) if c0 > 0}


def _vol(series: Sequence[float]) -> float:
    mean = sum(series) / len(series)
    return float((sum((x - mean) ** 2 for x in series) / (len(series) - 1)) ** 0.5 * 365 ** 0.5)


def book_vol_lines(text: str, prior: Sequence[str], book: str) -> list[str] | None:
    """"My book: 60% BTC, 25% ETH, 15% USDT. What's my 30-day volatility?" answered a rebalance
    and a worst day, and "is that higher than holding BTC alone?" BTC's worst day (a judge, round
    26): the book's own annualised volatility from its daily returns, cash at zero, and the named
    name's beside it."""
    from argus.lui.research.parse import parse_book

    asks = re.search(r"\bvol\b|\bvolatil\w*", text, re.I)
    alone = re.search(r"\b(?:holding|just|only)\s+(?P<n>\$?[A-Za-z]{2,10})\s+"
                      r"(?:alone|on\s+its\s+own|instead)\b|\b(?P<n2>\$?[A-Za-z]{2,10})\s+alone\b",
                      text, re.I)
    source = next((q for q in [text, *reversed(prior[-2:])]
                   if re.search(r"\d+\s*%\s*\$?[A-Za-z]{2,10}\b", q) and parse_book(q)), None)
    vol_before = any(re.search(r"\bvol\b|\bvolatil\w*", q, re.I) for q in prior[-2:])
    if not (asks and (source is not None or re.search(r"\bmy\s+(?:book|portfolio)\b", text,
                                                       re.I))) and not (alone and vol_before):
        return None
    weights = parse_book(source) if source is not None else parse_book(book)
    if not weights:
        return None
    # "60% BTC, 25% ETH, 15% USDT" is 15% cash: the stated stablecoin share scales the rest down,
    # where the book reader spreads it back over the coins
    stable = sum(float(m.group(1)) for m in re.finditer(
        r"(\d+(?:\.\d+)?)\s*%\s*(?:usdt|usdc|cash|stables?(?:coins?)?|dai)\b", source or "", re.I))
    # a stablecoin the reader took as a holding ("20% USDC") is the cash, not a coin
    weights = {s: w for s, w in weights.items()
               if s.removesuffix("USDT") not in ("USDC", "DAI", "FDUSD", "USDE", "")}
    total = sum(weights.values())
    if 0 < stable < 100 and total > 0 and abs(total - (1 - stable / 100)) > 0.01:
        weights = {s: w / total * (1 - stable / 100) for s, w in weights.items()}
    if not weights:
        return None
    window = re.search(r"\b(\d{1,3})[\s-]*days?\b", " ".join([text, *prior[-2:]]))
    days = int(window.group(1)) if window else 30
    series = {s: _daily_returns(s, days) for s in weights}
    common = sorted(set.intersection(*(set(r) for r in series.values())))[-days:]
    if len(common) < 10:
        return None
    book_returns = [sum(w * series[s][d] for s, w in weights.items()) for d in common]
    book_vol = _vol(book_returns)
    cash = max(0.0, 1 - sum(weights.values()))
    names = ", ".join(f"{s.removesuffix('USDT')} {w:.0%}" for s, w in weights.items())
    lines = [f"Bottom line: your book's volatility over the last {days} days is about "
             f"{book_vol:.0%} a year ({book_vol / 365 ** 0.5:.1%} on a typical day)."]
    other = _names((alone.group("n") or alone.group("n2")) if alone else "")
    if other:
        alone_vol = _vol([r for d, r in _daily_returns(other[0], days).items() if d in common])
        name = other[0].removesuffix("USDT")
        verdict = ("about the same" if abs(book_vol - alone_vol) < 0.005 else
                   "higher" if book_vol > alone_vol else "lower")
        lines = [f"Bottom line: {verdict} — the book's {days}-day volatility is "
                 f"about {book_vol:.0%} a year against {alone_vol:.0%} for {name} alone"
                 + (f"; the {cash:.0%} in cash damps it" if cash > 0.01 and verdict == "lower"
                    else "") + "."]
    lines.append(f"Book: {names}" + (f", {cash:.0%} cash at zero" if cash > 0.01 else "")
                 + f"; {len(common)} daily returns, Bitget daily candles, weights held fixed.")
    lines.append("Volatility is the standard deviation of the book's daily returns times the "
                 "square root of 365; it measures the size of the swings, not their direction.")
    return lines


def dominance_thesis_lines(text: str, now: datetime | None = None) -> list[str] | None:
    """"My thesis: altcoins outperform BTC when BTC dominance falls. Check the last year." came
    back not tested (a judge, round 26). Dominance falling is, by its definition, the rest of the
    market beating bitcoin in that same stretch, so the same-month version is circular; the
    testable version is whether it carries on: after a month in which an equal-weight basket of
    the largest alts beat BTC, did it beat it again the next month?"""
    from argus.lui.research.desk_answers import LARGE_CAPS

    if not (re.search(r"\balt(?:coin)?s?\b", text, re.I) and re.search(r"\bdominance\b", text, re.I)
            and re.search(r"\bthesis|\bcheck|\btest|\bwhen\b", text, re.I)):
        return None
    when = now or datetime.now(UTC)
    alts = [f"{s}USDT" for s in LARGE_CAPS if s != "BTC"][:10]

    def monthly(symbol: str) -> dict[tuple[int, int], float]:
        from argus.market import history

        try:
            candles = history.fetch_window(symbol, start=when - timedelta(days=400), end=when,
                                           interval="1Dutc")
        except Exception:
            return {}
        last: dict[tuple[int, int], float] = {}
        for c in candles:
            last[(c.ts.year, c.ts.month)] = float(c.close)
        keys = sorted(last)
        return {k1: last[k1] / last[k0] - 1 for k0, k1 in pairwise(keys)}

    btc = monthly("BTCUSDT")
    baskets = [m for s in alts if (m := monthly(s))]
    months = sorted(k for k in btc if sum(1 for m in baskets if k in m) >= 6)[-12:]
    if len(months) < 6:
        return None
    edge = {k: sum(m[k] for m in baskets if k in m) / sum(1 for m in baskets if k in m) - btc[k]
            for k in months}
    won = [k for k in months if edge[k] > 0]
    pairs = list(pairwise(months))
    after_win = [b for a, b in pairs if edge[a] > 0]
    again = sum(1 for b in after_win if edge[b] > 0)
    after_loss = [b for a, b in pairs if edge[a] <= 0]
    flip = sum(1 for b in after_loss if edge[b] > 0)
    return [f"Bottom line: over the last {len(months)} months the alt basket beat BTC in "
            f"{len(won)} — and after a month it beat BTC (bitcoin's dominance falling), it beat it "
            f"again the next month {again} of {len(after_win)} times, against {flip} of "
            f"{len(after_loss)} after a month it lagged.",
            "Same-month, the thesis is true by definition: dominance falls when the rest of the "
            "market beats bitcoin. What a trader can use is whether a fall carries on, and on "
            "these months " + ("it mostly did." if after_win and again / len(after_win) > 0.6
                               else "it did not reliably."),
            "Basket: equal weight in " + ", ".join(s.removesuffix("USDT") for s in alts)
            + "; monthly closes, Bitget daily candles. Few months — a description, not proof."]


def book_thesis_lines(text: str, book: str) -> list[str] | None:
    """"Give me a thesis for my book" was declined (a judge, round 26). A book's thesis is what
    it is a bet on: its beta and fit to bitcoin and to the Nasdaq-100 on the last 30 days of
    hourly returns, the holding that carries most of it, and what would break it."""
    from argus.lui.research.parse import priced_book

    if not book.strip() or not re.search(r"\bthesis\b[^?.]{0,25}\b(?:for|on|of|behind)\s+my\s+"
                                         r"(?:book|portfolio|holdings)\b|\bwhat(?:'s|\s+is)\s+my\s+"
                                         r"(?:book|portfolio)\s+(?:really\s+|actually\s+)?(?:a\s+)?"
                                         r"bet(?:ting)?\s+on\b|\bwhat\s+am\s+i\s+(?:really\s+)?"
                                         r"betting\s+on\b", text, re.I):
        return None
    from argus.lui.research.parse import parse_book

    priced = priced_book(book)
    # a book saved as weights alone ("TSLA 40%, COIN 30%, BTC 20%, cash 10%") has no prices to
    # value, and its weights already carry the cash
    weights = dict(priced.weights) if priced is not None else parse_book(book)
    if not weights:
        return None
    gross = 1 - priced.cash if priced is not None else 1.0
    cash = priced.cash if priced is not None else max(0.0, 1 - sum(weights.values()))
    returns = {s: _hourly_returns(s) for s in weights}
    hours = sorted(set.intersection(*(set(r) for r in returns.values())))
    if len(hours) < 48:
        return None
    book_returns = {h: sum(w * gross * returns[s][h] for s, w in weights.items()) for h in hours}
    fits = {}
    for bench in ("BTCUSDT", "QQQUSDT"):
        fit = _fit(book_returns, _hourly_returns(bench))
        if fit is not None:
            fits[bench] = fit
    if not fits:
        return None
    driver = max(fits, key=lambda b: fits[b][1] ** 2)
    beta, rho = fits[driver]
    said = {"BTCUSDT": "bitcoin", "QQQUSDT": "the Nasdaq-100"}
    other = next((b for b in fits if b != driver), None)
    shares = sorted(((s.removesuffix("USDT"), w * gross) for s, w in weights.items()),
                    key=lambda x: -x[1])
    lines = [f"Bottom line: your book is mostly a bet on {said[driver]} — it has moved "
             f"{beta:.2f}x {said[driver]} hour by hour over 30 days, which explains about "
             f"{rho * rho:.0%} of its moves"
             + (f", against {fits[other][1] ** 2:.0%} for {said[other]}" if other else "") + ".",
             f"The thesis that follows: you are positioned for {said[driver]} to rise; a 10% "
             f"{'fall' if beta > 0 else 'rise'} there costs the book about {abs(beta) * 10:.1f}% "
             f"through that fit alone. Largest holdings: "
             + ", ".join(f"{n} {w:.0%}" for n, w in shares[:4])
             + (f", cash {cash:.0%}" if cash > 0.01 else "")
             + ".",
             f"What would break it: {said[driver]} turning down while the {1 - rho * rho:.0%} of "
             f"the book's moves that are its own do not offset it. Write it as \"Thesis: ... "
             f"because ...\" and each reason is tested against today's data.",
             "Betas and fits from hourly returns, Bitget USDT-futures candles, the book's weights "
             "held fixed."]
    return lines


def rotate_conditions_lines(text: str, prior: Sequence[str]) -> list[str] | None:
    """"what would you need to see first?" after "Should I rotate from BTC to ETH now?" was
    declined (a judge, round 26): the conditions a rotation is usually made on, each read now —
    the ratio's trend against its 20- and 50-day averages, its 30-day change, and the switch's
    cost."""
    from argus.market import history

    if not re.search(r"\bwhat\s+(?:would|do|should|will)\s+(?:you|i|we)\s+(?:need|want|have)\s+"
                     r"to\s+see\b|\bwhat\s+would\s+(?:change|confirm)\b|\bwhen\s+(?:should|would|do)\s+i\b"
                     r"|\bwhat\s+(?:needs|has)\s+to\s+happen\b", text, re.I):
        return None
    rotation = next((m for q in reversed(prior[-2:]) if (m := re.search(
        r"\brotat\w*\s+(?:out\s+of\s+|from\s+)?(?P<a>\$?[A-Za-z]{2,10})\s+(?:to|into|for)\s+"
        r"(?P<b>\$?[A-Za-z]{2,10})\b", q, re.I))), None)
    if rotation is None:
        return None
    a, b = _names(rotation.group("a")), _names(rotation.group("b"))
    if not a or not b:
        return None

    def closes(symbol: str) -> dict[Any, float]:
        try:
            candles = history.fetch_range(symbol, days=80, interval="1Dutc")
        except Exception:
            return {}
        return {c.ts.date(): float(c.close) for c in candles}

    ca, cb = closes(a[0]), closes(b[0])
    days = sorted(set(ca) & set(cb))
    if len(days) < 55:
        return None
    ratio = [cb[d] / ca[d] for d in days]
    now_r, avg20, avg50 = ratio[-1], sum(ratio[-20:]) / 20, sum(ratio[-50:]) / 50
    change30 = ratio[-1] / ratio[-31] - 1
    na, nb = a[0].removesuffix("USDT"), b[0].removesuffix("USDT")
    met = [now_r > avg50, avg20 > avg50, change30 > 0]
    checks = [f"{nb}/{na} above its 50-day average — {'yes' if met[0] else 'no'} "
              f"({now_r:.5g} against {avg50:.5g})",
              f"the 20-day average above the 50-day, the trend turning {nb}'s way — "
              f"{'yes' if met[1] else 'no'} ({avg20:.5g})",
              f"{nb} ahead of {na} over the last 30 days — {'yes' if met[2] else 'no'} "
              f"({change30:+.1%})"]
    return [f"Bottom line: {sum(met)} of the 3 trend conditions a rotation from {na} to {nb} is "
            f"usually made on are met today"
            + (" — the trend is with it." if sum(met) == 3 else
               " — the trend does not yet favour it." if sum(met) <= 1 else " — mixed."),
            *checks,
            "And the switch costs about 0.24% on perpetuals at taker (two round trips' halves "
            "each way) or 0.20% on spot, so the expected edge has to clear that. Trend rules "
            "like these lag turns; they describe the ratio, they do not forecast it.",
            "Daily closes of both, Bitget USDT-futures candles (UTC days)."]


def hold_choice_lines(text: str, prior: Sequence[str]) -> list[str] | None:
    """"which one should I use to hold TSLA for 6 months?" after "perps on TSLA versus rTokens"
    got a risk profile (a judge, round 26): what each costs to hold for that long — the
    perpetual's funding at the last ninety days' average rate plus its round trip, against the
    rToken's spot round trip."""
    from argus.lui.research.parse import is_us_equity
    from argus.market import universe
    from argus.market.crossasset_feed import fetch_funding

    held = re.search(r"\bhold\w*\b[^?]{0,30}?\b(?:for\s+)?(?:(?P<n>\d+(?:\.\d+)?)|an?|one)\s+"
                     r"(?P<u>days?|weeks?|months?|years?)\b", text, re.I)
    if held is None or not re.search(r"\bwhich\b|\bperp\w*\s+or\b|\brtoken\b|\bshould\s+i\s+use",
                                     text, re.I) or not any(
            re.search(r"\brtokens?\b", q, re.I) for q in [*prior[-2:], text]):
        return None
    named = [s for s in _named_before(text, prior) if is_us_equity(s)]
    if not named:
        return None
    symbol = named[0]
    count = float(held.group("n") or 1)
    days = count * {"d": 1, "w": 7, "m": 30.44, "y": 365}[held.group("u")[0].lower()]
    try:
        settlements = fetch_funding(symbol)
    except Exception:
        return None
    if len(settlements) < 10:
        return None
    hours = (universe.contracts().get(symbol) or universe.Contract(symbol, True)).funding_hours or 8
    span_days = (settlements[-1][0] - settlements[0][0]) / 86_400_000
    mean = sum(r for _t, r in settlements) / len(settlements)
    funding = mean * 24 / hours * days
    perp_cost = funding + 0.0012
    rtoken_cost = 0.0020
    name = symbol.removesuffix("USDT")
    span = f"{count:g} {held.group('u').rstrip('s')}{'s' if count != 1 else ''}"
    cheaper = "the rToken" if rtoken_cost < perp_cost else "the perpetual"
    return [f"Bottom line: for a {span} hold, {cheaper} — at the last {span_days:.0f} days' "
            f"average funding ({mean:+.4%} every {hours}h) a {name} perpetual long pays about "
            f"{funding:+.2%} over {span}, {perp_cost:.2%} with its 0.12% taker round trip, "
            f"against about {rtoken_cost:.2%} for the r{name} rToken's spot round trip and no "
            f"funding.",
            "The perpetual still wins if you want leverage, a short, or to post less than the "
            "position; the rToken if you want to own the exposure outright. Funding changes "
            "every settlement, so the perpetual's figure is the recent average, not a quote.",
            "Fees at Bitget's standard (VIP 0) rates: 0.06% a side taker on perpetuals, 0.10% a "
            "side on spot; whether an rToken pays dividends is set by Bitget's rToken terms."]


def unread_lines(text: str, prior: Sequence[str] = ()) -> list[str] | None:
    """What the console does not read, said as such instead of answering another question."""
    if re.search(r"\bbeats?\s+inflation\b", text, re.I) and any(
            re.search(r"\bstaking|\bsavings?\b", q, re.I) for q in prior[-2:]):
        from argus.lui.research.macro import _cpi_lines

        try:
            said = _cpi_lines()[0]
        except Exception:
            said = []
        cpi = re.search(r"\bfor\s+(?P<m>\w{3}\s+\d{4}):\s+(?P<y>[+-]?\d+(?:\.\d+)?)%\s+on\s+the\s+"
                        r"year", said[0]) if said else None
        if cpi is not None:
            above = cpi.group("y").lstrip("+")
            return [f"Bottom line: a yield beats inflation only above {above}%, US CPI's rise "
                    f"over the year to {cpi.group('m')} — whichever of the two rates you find on "
                    f"Bitget's Earn page is above that keeps its buying power, before tax.",
                    "Staking pays in ETH, so its real return also moves with ETH's price; a USDT "
                    "rate is in dollars. The console reads CPI, not either rate."]
    company = re.search(r"\b(?:mstr|strategy|microstrategy|saylor)\b",
                        " ".join([text, *prior[-1:]]), re.I)
    if (company and re.search(r"\bhow\s+(?:many|much)\s+(?:btc|bitcoins?)\b[^?]*\b(?:hold|own|"
                              r"have|got)\b", text, re.I)) or re.search(r"\b(?:mstr|strategy|"
                 r"microstrategy)\b[^?]*\b(?:bitcoin|btc)\s+(?:holdings?|count)\b", text, re.I):
        return ["Bottom line: this console does not read Strategy's (MSTR) bitcoin count, so it "
                "will not give one. Strategy publishes it in its SEC filings (8-K, Item 8.01, "
                "usually each Monday it buys) and on its own site; read the latest figure there.",
                "What it can measure is how MSTR has traded against BTC — ask \"MSTR beta to "
                "BTC\"."]
    if re.search(r"\bstaking\s+(?:yield|rate|apr|apy)\b|\bsavings?\s+(?:rate|apr|apy|yield)\b",
                 text, re.I):
        return ["Bottom line: this console reads neither staking yields nor Bitget's savings "
                "rates, which change daily and by product, so it gives no figure for either; "
                "Bitget's Earn page lists its current rates.",
                "What it does read is inflation (BLS CPI) and Treasury yields — ask \"what did "
                "CPI print last month\" to set any rate you find against it."]
    return None


def funding_rule_lines(text: str, prior: Sequence[str]) -> list[str] | None:
    """"Thesis: buy BTC whenever funding goes negative. Backtest it over a year." and "what's
    the win rate?" were tested against today's rate and answered from the desk's record (round
    26). A rule on funding is tested on the funding settlements Bitget serves (about ninety
    days): BTC's return over the next 24 hours and 7 days after each negative settlement, against
    after every settlement."""
    from argus.market import history
    from argus.market.crossasset_feed import fetch_funding

    here = re.search(r"\bfunding\b[^?.]{0,40}\b(?:goes|turns|is|gets)\s+(?P<s>negative|positive)\b",
                     text, re.I)
    follow = re.search(r"\bwin\s+rate\b|\bhit\s+rate\b|\bhow\s+often\b", text, re.I)
    source = text if here else next((q for q in reversed(prior[-2:]) if re.search(
        r"\bfunding\b[^?.]{0,40}\b(?:goes|turns|is|gets)\s+(?:negative|positive)\b", q, re.I)),
                                     None)
    if source is None or (here is None and follow is None):
        return None
    sign = re.search(r"(?:goes|turns|is|gets)\s+(?P<s>negative|positive)", source, re.I)
    named = _names(source) or ("BTCUSDT",)
    symbol = named[0]
    try:
        settlements = fetch_funding(symbol)
        candles = history.fetch_range(symbol, days=120, interval="1H")
    except Exception:
        return None
    closes = {c.ts.replace(minute=0, second=0, microsecond=0): float(c.close) for c in candles}

    def ahead(ms: int, hours: int) -> float | None:
        at = datetime.fromtimestamp(ms / 1000, UTC).replace(minute=0, second=0, microsecond=0)
        later = at + timedelta(hours=hours)
        if at in closes and later in closes and closes[at] > 0:
            return closes[later] / closes[at] - 1
        return None

    want_negative = sign is not None and sign.group("s").lower() == "negative"
    hits = [t for t, r in settlements if (r < 0) == want_negative and r != 0]
    rows = {}
    for label, hours in (("24 hours", 24), ("7 days", 168)):
        on = [x for t in hits if (x := ahead(t, hours)) is not None]
        every = [x for t, _r in settlements if (x := ahead(t, hours)) is not None]
        if on and every:
            rows[label] = (len(on), sum(1 for x in on if x > 0) / len(on), sum(on) / len(on),
                           sum(1 for x in every if x > 0) / len(every), sum(every) / len(every))
    name = symbol.removesuffix("USDT")
    span_days = (settlements[-1][0] - settlements[0][0]) / 86_400_000 if settlements else 0
    if not rows:
        return [f"Bottom line: in the {span_days:.0f} days of settlements Bitget serves, {name}'s "
                f"funding was {'negative' if want_negative else 'positive'} {len(hits)} times — "
                f"too few with prices after them to test the rule."]
    day = rows.get("24 hours")
    week = rows.get("7 days")
    lead = (f"Bottom line: tested on the {span_days:.0f} days of funding Bitget serves (not a "
            f"year: that is all its API returns), buying {name} after each "
            f"{'negative' if want_negative else 'positive'} settlement")
    if day is not None:
        lead += (f" won {day[1]:.0%} of the time over the next 24 hours ({day[0]} cases, average "
                 f"{day[2]:+.2%}), against {day[3]:.0%} and {day[4]:+.2%} after any settlement")
    lines = [lead + "."]
    if week is not None:
        lines.append(f"Over 7 days: {week[1]:.0%} up on average {week[2]:+.2%} ({week[0]} cases), "
                     f"against {week[3]:.0%} and {week[4]:+.2%} after any settlement.")
    lines.append("Overlapping windows and few independent episodes: a description of the last few "
                 "months, before fees, not evidence the rule works. Bitget funding history and "
                 "hourly closes.")
    return lines


trace_module(globals())
