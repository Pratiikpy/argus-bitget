"""Where a name trades right now, what a round trip costs, and the quote's extras."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from argus.lui.answer import Source, unlead
from argus.lui.question import (
    TRADED_SYMBOLS,
)
from argus.lui.research.anchor import (
    _last_regular_close,
    _perp_at_close,
    _rtoken_tracking,
    _yahoo_close,
)
from argus.lui.research.data import (
    _FETCH_SLOTS,
)
from argus.lui.research.execution import (
    _daily_volatility_bps,
)
from argus.lui.research.kinds import (
    _t,
)
from argus.lui.research.parse import (
    _DAILY_TA,
    _FUNDING_EXPLAIN_Q,
    _FUNDING_WORDS,
    _HOLD_PERIOD,
    _HOW_MANY,
    _LIQUIDITY_TIME_Q,
    _PERIOD_MOVE,
    _RATIO_Q,
    _ROUND_TRIP,
    _SHARE_OF_BOOK,
    _SINCE_HIGH_Q,
    _SPREAD_WIDEN_Q,
    _WEEKEND_TRADING_Q,
    _number,
    _period_days,
    is_us_equity,
    parse_notional,
)
from argus.lui.research.venue import (
    _funding_meaning,
)
from argus.lui.trace import trace_module

_MARK_Q = re.compile(r"\bmark(?:\s+price)?\b|\bindex\s+price\b", re.I)


_NEXT_FUNDING_Q = re.compile(
    r"\bnext\s+funding\b|\bfunding\s+(?:time|settlement|countdown|payment)s?\b|"
    r"\bwhen\s+(?:is|does)\s+(?:the\s+)?(?:next\s+)?funding\b", re.I)


_GAP_Q = re.compile(r"\b(?:over|under|vs\.?|versus|against|premium|discount|basis|gap|"
                    r"difference|spread\s+between)\b", re.I)


_PRICE_ASKED = re.compile(
    r"\bprice[ds]?\b|\bpricing\b|\btrading\s+at\b|\btrades?\s+at\b|\bquote\b|\bhow\s+much\s+is\b|"
    r"\bwhere\s+is\s+\S+\s+(?:trading|now|at)\b|\bwhat(?:'s|\s+is)\s+\S+\s+at\b|\blast\s+price\b|"
    r"\bcurrent(?:ly)?\s+(?:price|trading)\b|\bpreis\b|\bprecio\b|\bprix\b|价格|現価|価格|가격",
    re.I)


def _lead_with(lines: list[str], prefix: str) -> list[str]:
    """Move the line starting with ``prefix`` to the front as the answer's lead, demoting the
    current lead to a plain line. Unchanged when no such line exists."""
    plain = [unlead(line)
             for line in lines]
    hit = next((i for i, line in enumerate(plain) if line.startswith(prefix)), None)
    if hit is None:
        return lines
    return [f"Bottom line: {plain[hit]}", *plain[:hit], *plain[hit + 1:]]


def _rtoken_market_lines(spot: str, perp: str, perp_ticker: Any,
                         text: str) -> tuple[list[str], list[Source]]:
    """The spot rToken's own market: last, bid/ask and spread, the visible depth near the price,
    and its gap to the same company's perpetual — plus, when the question asks for a premium over
    a period, how that gap has behaved hour by hour. Bitget's spot ticker also reports a 24h
    volume, which is left out: for RNVDAUSDT on 2026-09-25 it read $11.8bn, about 850 times the
    perpetual's, which no reading of the units makes plausible."""
    from argus.market.bitget import public_get

    base = "r" + spot.removeprefix("R").removesuffix("USDT")
    try:
        row = (public_get("/api/v2/spot/market/tickers", {"symbol": spot}) or [{}])[0]
        last, bid, ask = float(row["lastPr"]), float(row["bidPr"]), float(row["askPr"])
    except Exception:
        return [f"{base} ({spot}): Bitget's spot ticker did not answer just now."], []
    spread = (ask - bid) / ((ask + bid) / 2) * 10_000 if ask > 0 and bid > 0 else None
    depth_text = ""
    try:
        book = public_get("/api/v2/spot/market/orderbook", {"symbol": spot, "limit": "50"}) or {}
        mid = (ask + bid) / 2
        near = [(float(pr), float(sz)) for side in ("bids", "asks") for pr, sz in book.get(side, [])
                if abs(float(pr) / mid - 1) <= 0.005]
        depth = sum(pr * sz for pr, sz in near)
        depth_text = f"; about ${depth:,.0f} of orders rest within 0.5% of the price"
    except Exception:
        depth_text = ""
    perp_last = float(perp_ticker.last)
    gap = (last / perp_last - 1) * 10_000 if perp_last > 0 else None
    lines = [
        f"Bottom line: {base}, the Bitget spot rToken, last {last:g}; bid {bid:g} / ask {ask:g}"
        + (f", spread {spread:.1f}bps" if spread is not None else "") + depth_text + "."
        + (f" It trades {abs(gap):.1f}bps {'over' if gap >= 0 else 'under'} the "
           f"{_t(perp)} perpetual ({perp_last:g})." if gap is not None else ""),
    ]
    sources = [Source(kind="venue", ref="bitget /api/v2/spot/market/tickers + orderbook",
                      detail=f"{spot} last, bid, ask, 50 levels")]
    days = _period_days(text) if re.search(r"premium|discount|gap|track", text, re.I) else None
    if days:
        tracked = _rtoken_tracking(spot, perp, days)
        if tracked is not None:
            lines.append(tracked[0])
            sources.append(tracked[1])
    return lines, sources


def _sized_round_trip(symbol: str, notional: Decimal,
                      fee_bps: Any) -> tuple[str, Source] | None:
    """A round trip of a stated size: the asks walked for the entry, the bids for the exit, plus
    both taker fees. "Round trip cost of trading 500 btc perp" got the flat 12bps a one-lot trade
    pays (answer audit, round 3); at size the book is the cost."""
    from argus.market.depth import fetch_orderbook

    try:
        with _FETCH_SLOTS:
            book = fetch_orderbook(symbol, limit=50)
        entry = book.sweep(notional, direction="BUY")
        exit_ = book.sweep(notional, direction="SELL")
    except Exception:
        return None
    total = float(fee_bps) + float(entry.slippage_bps) + float(exit_.slippage_bps)
    thin = not (entry.complete and exit_.complete)
    if thin:
        # Past the visible book the walk understates the cost; each leg is priced instead by the
        # square-root impact law the execution plan uses, on the day's volume and volatility.
        try:
            from argus.cost.model import CostModel
            from argus.market.bitget import fetch_tickers

            ticker = fetch_tickers()[symbol]
            adv = ticker.base_volume * ticker.last
            leg = float(CostModel.bitget_perp().impact_bps(notional / adv,
                                                           _daily_volatility_bps(symbol)))
            law_total = float(fee_bps) + 2 * leg
            return (f"Bottom line: a round trip of ${float(notional):,.0f} in {_t(symbol)} is "
                    f"{float(notional / adv):.1%} of its 24h volume each way, more than the 50 "
                    f"visible levels hold — the square-root impact law puts each leg near "
                    f"{leg:.1f}bps, so about {law_total:.1f}bps "
                    f"(${float(notional) * law_total / 10_000:,.0f}) with {float(fee_bps):.0f}bps "
                    f"of taker fees; ask how to split it to bring that down.",
                    Source(kind="computation", ref="argus.cost.model.CostModel.impact_bps",
                           detail=f"{symbol} square-root impact on 24h volume, both legs"))
        except Exception:
            pass
    text = (f"Bottom line: a round trip of ${float(notional):,.0f} in {_t(symbol)} costs about "
            f"{total:.1f}bps (${float(notional) * total / 10_000:,.0f}) on the book right now — "
            f"{float(entry.slippage_bps):.1f}bps walking the asks to get in, "
            f"{float(exit_.slippage_bps):.1f}bps walking the bids to get out, and "
            f"{float(fee_bps):.0f}bps in taker fees"
            + (" — more than the 50 visible levels hold, so the real cost is higher; ask how to "
               "split it for the impact law's estimate beyond the book." if thin else "."))
    return text, Source(kind="venue", ref="bitget /api/v2/mix/market/merge-depth",
                        detail=f"{symbol} both sides walked for ${float(notional):,.0f}")


def _quote_extras(raw_text: str, quoted: list[tuple[str, Any]],
                  fee: float) -> tuple[list[str], list[Source], str | None]:
    """The quote figures a question asks for by name, beyond the standard quote: mark and index
    price, the next funding settlement, funding over a stated holding period and size, and the gap
    between two quoted instruments. Each was asked for in the 2026-09-25 answer audit and answered
    with the plain quote, which carried none of them (the XAUT and XAU prices were even printed
    side by side without their gap). Returns the lines, their sources, and the lead line when the
    question asked for exactly one of these."""
    from argus.market import universe
    from argus.market.bitget import public_get

    symbol, ticker = quoted[0]
    base = _t(symbol)
    lines: list[str] = []
    sources: list[Source] = []
    lead: str | None = None
    contract = universe.contracts().get(symbol) or universe.Contract(symbol, False)
    hours = contract.funding_hours or 8
    rate = float(ticker.funding_rate) * 100.0

    if _MARK_Q.search(raw_text):
        try:
            row = (public_get("/api/v2/mix/market/ticker",
                        {"symbol": symbol, "productType": "usdt-futures"}) or [{}])[0]
            mark, index = float(row["markPrice"]), float(row["indexPrice"])
            gap = (float(ticker.last) / mark - 1.0) * 10_000
            text = (f"{base} mark price {mark:g} and index {index:g}, against a last trade of "
                    f"{ticker.last}: the last trade is {gap:+.1f}bps from the mark. Unrealised P&L "
                    f"and liquidation run on the mark, not the last trade.")
            lines.append(text)
            lead = lead or text
            sources.append(Source(kind="venue", ref="bitget /api/v2/mix/market/ticker",
                                  detail=f"{symbol} markPrice, indexPrice"))
        except Exception:
            lines.append(f"{base}'s mark price did not arrive from Bitget just now.")

    if _NEXT_FUNDING_Q.search(raw_text):
        try:
            row = (public_get("/api/v2/mix/market/current-fund-rate",
                        {"symbol": symbol, "productType": "usdt-futures"}) or [{}])[0]
            at = datetime.fromtimestamp(int(row["nextUpdate"]) / 1000, tz=UTC)
            wait = at - datetime.now(UTC)
            h, m = divmod(max(0, int(wait.total_seconds() // 60)), 60)
            text = (f"{base}'s next funding settlement is {at:%a %H:%M} UTC, in {h}h {m:02d}m, at "
                    f"the current {float(row['fundingRate']) * 100:+.4f}% (settled every "
                    f"{row.get('fundingRateInterval') or hours}h).")
            lines.append(text)
            lead = lead or text
            sources.append(Source(kind="venue", ref="bitget /api/v2/mix/market/current-fund-rate",
                                  detail=f"{symbol} nextUpdate"))
        except Exception:
            lines.append(f"{base}'s next funding time did not arrive from Bitget just now.")

    period = _HOLD_PERIOD.search(raw_text)
    if period is not None and (re.search(r"fund", raw_text, re.I) or _ROUND_TRIP.search(raw_text)
                               or re.search(r"\bcost\b", raw_text, re.I)):
        if period.group(0).lower() == "overnight":
            held_hours = 16.0
        else:
            amount = float(period.group(1) or 1)
            unit = period.group(2).lower()
            held_hours = amount * (1 if unit.startswith("h") else 24 if unit.startswith("d")
                                   else 168 if unit.startswith("w") else 720)
        settlements = int(held_hours // hours)
        short = bool(re.search(r"\bshort\b", raw_text, re.I))
        paid = rate * settlements * (-1 if short else 1)
        size = parse_notional(raw_text)
        dollars = (f" (about ${float(size) * paid / 100:,.2f} on ${float(size):,.0f})"
                   if size else "")
        side = "short" if short else "long"
        verb = "pays" if paid > 0 else "receives" if paid < 0 else "pays nothing in"
        spread = float(ticker.spread_bps)
        total = fee + spread + paid * 100
        text = (f"Held {held_hours:g}h as a {side}: {settlements} funding settlement(s) at today's "
                f"{rate:+.4f}% — the {side} {verb} {abs(paid):.3f}% of the position{dollars}; with "
                f"the {fee + spread:.1f}bps round trip that is about {total:.1f}bps in "
                f"all, if the rate holds (it resets every {hours}h).")
        lines.append(text)
        lead = lead or text

    if _FUNDING_EXPLAIN_Q.search(raw_text):
        per_year = 24 / hours * 365
        text = (f"Funding here is quoted per settlement interval — every {hours}h for {base} — and "
                f"the yearly figure multiplies it by the {per_year:.0f} settlements in a year: "
                f"{rate:+.4f}% per interval is about {rate * per_year:+.1f}% a year if it held.")
        lines.append(text)
        lead = lead or text

    if _HOW_MANY.search(raw_text):
        share = _SHARE_OF_BOOK.search(raw_text)
        worth: float | None = None
        how = ""
        if share is not None:
            book_value = _number(share.group(2)) * {"k": 1e3, "m": 1e6}.get(
                (share.group(3) or "").lower(), 1.0)
            worth = float(share.group(1)) / 100 * book_value
            how = f"{float(share.group(1)):g}% of ${book_value:,.0f} is ${worth:,.0f}, and "
        else:
            size = parse_notional(raw_text)
            worth = float(size) if size else None
        if worth:
            units = worth / float(ticker.last)
            text = (f"{how}${worth:,.0f} of {base} at Bitget's last {ticker.last} is about "
                    f"{units:,.4g} {base} — on the perpetual, where a size is a quantity of the "
                    f"underlying, not whole shares.")
        else:
            text = (f"{base} is {ticker.last} on Bitget; name the amount — \"how many {base} is "
                    f"$10,000\" — and the count follows.")
        lines.append(text)
        lead = lead or text
    days = _period_days(raw_text) if _PERIOD_MOVE.search(raw_text) else None
    if days:
        from argus.market.history import fetch_window

        # Hourly bars up to 30 days so the start sits within an hour of the cutoff; daily beyond.
        step = timedelta(hours=1) if days <= 30 else timedelta(days=1)
        try:
            bars = fetch_window(symbol, start=datetime.now(UTC) - timedelta(days=days + 2),
                                interval="1H" if days <= 30 else "1D", pause=0.05)
        except Exception:
            bars = []
        cutoff = datetime.now(UTC) - timedelta(days=days)
        # A bar closes at its open time plus its length; the start is the last close at or before
        # the cutoff, and the range covers the bars that followed it.
        before = [b for b in bars if b.ts + step <= cutoff]
        if before:
            start = float(before[-1].close)
            window = [b for b in bars if b.ts > before[-1].ts]
            high = max(float(b.high) for b in window)
            low = min(float(b.low) for b in window)
            move = (float(ticker.last) / start - 1) * 100
            closed = before[-1].ts + step
            text = (f"Over the last {days} day(s) {base} moved {move:+.2f}%, from {start:g} at the "
                    f"{closed:%d %b %H:%M} UTC close to {ticker.last} now; its range in that "
                    f"time was {low:g} to {high:g}.")
            lines.append(text)
            lead = lead or text
            sources.append(Source(kind="venue", ref="bitget /api/v3/market/candles",
                                  detail=f"{symbol} hourly, last {days + 2} days"))
    if _WEEKEND_TRADING_Q.search(raw_text):
        kind_of = universe.NOT_EQUITY.get(symbol)
        underlying = ("its stock trades only on weekdays, 09:30 to 16:00 New York time"
                      if symbol in TRADED_SYMBOLS or universe.is_equity(symbol) else
                      "the FX market it tracks is shut from Friday 22:00 to Sunday 22:00 UTC"
                      if kind_of == "fx" else
                      "the futures it tracks mostly pause from Friday evening to Sunday evening "
                      "New York time" if kind_of == "commodity" else
                      "crypto itself never closes")
        text = (f"Yes — {base}'s Bitget perpetual trades through the weekend, like every Bitget "
                f"contract; {underlying}, so weekend prices "
                + ("move on Bitget's own book with no outside market to anchor them."
                   if kind_of in ("fx", "commodity") or symbol in TRADED_SYMBOLS
                   or universe.is_equity(symbol) else "are ordinary crypto prices."))
        lines.append(text)
        lead = lead or text
    if len(quoted) >= 2 and _RATIO_Q.search(raw_text):
        (a_sym, a), (b_sym, b) = quoted[0], quoted[1]
        ratio = float(a.last) / float(b.last)
        text = (f"The {_t(a_sym)}/{_t(b_sym)} ratio is {ratio:,.4g} on Bitget right now "
                f"({a.last} over {b.last}).")
        lines.append(text)
        lead = lead or text
    if len(quoted) >= 2 and _SINCE_HIGH_Q.search(raw_text):
        from argus.market.history import fetch_window

        (subject, sub_ticker), (anchor, _anchor_ticker) = quoted[0], quoted[1]
        try:
            anchor_bars = fetch_window(anchor, start=datetime.now(UTC) - timedelta(days=366),
                                       interval="1D", pause=0.05)
            subject_bars = fetch_window(subject, start=datetime.now(UTC) - timedelta(days=366),
                                        interval="1D", pause=0.05)
        except Exception:
            anchor_bars, subject_bars = [], []
        this_year = bool(re.search(r"\bthis\s+year\b|\bytd\b|year[\s-]to[\s-]date", raw_text,
                                   re.I))
        if this_year:
            anchor_bars = [b for b in anchor_bars if b.ts.year == datetime.now(UTC).year]
        if anchor_bars:
            top = max(anchor_bars, key=lambda b: float(b.high))
            closes = {b.ts: float(b.close) for b in subject_bars}
            base_close = closes.get(top.ts)
            via = "Bitget's daily closes"
            if base_close is None and (subject in TRADED_SYMBOLS or universe.is_equity(subject)):
                # The perpetual may have listed after the anchor's high (SQQQ's history on Bitget
                # starts in March 2026, gold's 2026 high was in January): the stock's own close
                # that day stands in, and the answer says so.
                from argus.market import equity_history

                try:
                    stock_closes = {d.day: d.close for d in equity_history.daily(_t(subject))}
                except Exception:
                    stock_closes = {}
                base_close = stock_closes.get(top.ts.date())
                via = (f"{_t(subject)}'s own close that day (Yahoo), as Bitget's perpetual listed "
                       f"later")
            span_label = "this year" if this_year else "the past year"
            if base_close:
                move = (float(sub_ticker.last) / base_close - 1) * 100
                text = (f"{_t(anchor)} made its high of {span_label}, {float(top.high):,.6g}, on "
                        f"{top.ts:%d %b %Y}; since then {_t(subject)} has moved {move:+.2f}%, from "
                        f"{base_close:,.6g} to {sub_ticker.last} ({via}).")
                sources.append(Source(kind="venue", ref="bitget /api/v3/market/candles",
                                      detail=f"{anchor}, {subject} daily, one year"))
            else:
                text = (f"{_t(anchor)} made its high of {span_label} on {top.ts:%d %b %Y}, before "
                        f"any "
                        f"price history this desk has for {_t(subject)}, so the move since cannot "
                        f"be stated.")
            lines.append(text)
            lead = lead or text
    if len(quoted) >= 2 and _GAP_Q.search(raw_text):
        (a_sym, a), (b_sym, b) = quoted[0], quoted[1]
        gap = (float(a.last) / float(b.last) - 1.0) * 10_000
        text = (f"{_t(a_sym)} trades {abs(gap):.1f}bps {'over' if gap >= 0 else 'under'} "
                f"{_t(b_sym)} on Bitget ({a.last} against {b.last}).")
        lines.append(text)
        lead = lead or text
    if re.search(r"\b(?:52|fifty[\s-]two)\s*-?\s*w|\byear(?:ly)?\s+(?:high|low|range)|"
                 r"\b(?:1|one)[\s-]*year\s+(?:high|low|range)", raw_text, re.I) \
            and is_us_equity(symbol):
        # "tsla 52 week high and low" was answered with the 24h range (answer audit, round 3);
        # the figure a trader means is the share's own closing high and low over 52 weeks
        try:
            from argus.market import equity_history

            sessions = equity_history.daily(base)[-252:]
            peak = max(sessions, key=lambda d: d.close)
            trough = min(sessions, key=lambda d: d.close)
            last_close = sessions[-1].close
            text = (f"{base}'s 52-week closing high is {peak.close:,.2f} ({peak.day:%d %b %Y}) and "
                    f"low {trough.close:,.2f} ({trough.day:%d %b %Y}), split-adjusted (Yahoo "
                    f"Finance); the last close, {last_close:,.2f}, is "
                    f"{(last_close / peak.close - 1):.0%} from the high and "
                    f"{(last_close / trough.close - 1):+.0%} from the low.")
            lines.append(text)
            lead = text  # the 52-week figures are what was asked
            sources.append(Source(kind="venue", ref="Yahoo Finance daily chart",
                                  detail=f"{base} split-adjusted closes, 252 sessions"))
        except Exception:
            lines.append(f"{base}'s 52-week closing range did not arrive from Yahoo just now; "
                         f"the perpetual's own range over the year is given below.")
    if _LIQUIDITY_TIME_Q.search(raw_text):
        from argus.market import liquidity_profile
        from argus.market.history import fetch_range

        try:
            with _FETCH_SLOTS:
                hourly = fetch_range(symbol, days=30, interval="1H")
            prof = liquidity_profile.profile(
                symbol, [(b.ts, float(b.volume), float(b.close)) for b in hourly])
        except Exception:
            prof = None
        if prof is not None:
            found = [line.replace("Bottom line: ", "", 1)
                     for line in liquidity_profile.lines(prof, base, is_us_equity(symbol))]
            lead = next((line for line in found if line.startswith("Weekends")), found[0]) \
                if re.search(r"weekend", raw_text, re.I) else found[0]
            lines.extend(line for line in found if line is not lead)
            sources.append(Source(kind="computation", ref="argus.market.liquidity_profile",
                                  detail=f"{symbol} hourly volume x close, "
                                         f"{prof.hours_read} hours"))
    if _SPREAD_WIDEN_Q.search(raw_text):
        spread = float(ticker.spread_bps)
        wider = [(w, fee + w) for w in (max(1.0, spread * 5), 10.0, 25.0)]
        text = (f"Yes — a taker round trip pays the whole spread once (half going in, half coming "
                f"out) on top of {fee:.0f}bps of fees: at {base}'s spread now, {spread:.1f}bps, "
                f"that is {fee + spread:.1f}bps; "
                + "; ".join(f"at {w:.0f}bps it is {total:.0f}bps" for w, total in wider)
                + ". Every basis point of spread is a basis point of round-trip cost, and a size "
                  "larger than the touch also walks deeper into a book that thinned as the spread "
                  "widened — ask the round-trip cost of your size for that part.")
        lines.append(text)
        lead = text
    if lead is None and (_FUNDING_WORDS.search(raw_text)
                         or re.search(r"\bfunding\b", raw_text, re.I)):
        # "BTC funding rate" opened on the round-trip cost with the rate inside the quote line
        # (2026-09-25 audit, round 2): the rate asked for leads, with what it means.
        text = (f"{base} funding is {rate:+.4f}% per {hours}h settlement"
                + (f". {_funding_meaning(symbol, rate)}" if rate else
                   " — flat, so holding costs nothing beyond fees."))
        lines.append(text)
        lead = text
    return lines, sources, lead


def _session_line(symbol: str, *, anchor_open: bool, us_listed: bool) -> str:
    """What the clock means for this contract's price — which depends on what it tracks.

    The twelve stock perpetuals and every other US-listed stock have a US anchor whose hours are
    modelled (`argus.truth.clocks`). A crypto contract has no anchor at all. Gold, oil, FX, index
    products and Asian equities have anchors on other clocks that this console does not model, and
    saying "the US market is shut" about gold would be a confident sentence about the wrong market.
    ``us_listed`` is whether `bitget-mcp-server` recognised the underlying as a US ticker.
    """
    from argus.market import universe

    contract = universe.contracts().get(symbol)
    if symbol in TRADED_SYMBOLS or (us_listed and universe.is_equity(symbol)):
        return ("The US anchor market is open, so this price has price discovery behind it."
                if anchor_open else
                "The US anchor market is shut: this stock perpetual is trading without its "
                "anchor, so its price is being discovered on a thinner book and can gap at the "
                "open.")
    if contract is not None and not contract.rwa:
        return ("A crypto contract: it trades around the clock on its own market, with no "
                "off-chain anchor to gap against.")
    return (f"{_t(symbol)} tracks an off-chain market on its own trading hours, which this console "
            f"does not model — check that market's session before reading a quiet-hours price as "
            f"real price discovery.")


def _daily_closes(symbol: str) -> tuple[list[float], str]:
    """Daily closes, oldest first, and whose they are. A stock's own split-adjusted closes (Yahoo)
    are what a "200-day average" means to a trader and reach back decades; the perpetual has only
    existed a year or so. Crypto reads Bitget's daily candles."""
    if is_us_equity(symbol):
        from argus.market import equity_history

        try:
            days = equity_history.daily(_t(symbol))
            if len(days) >= 30:
                return [d.close for d in days], (f"{_t(symbol)}'s split-adjusted daily closes "
                                                 f"(Yahoo)")
        except Exception:
            pass
    from argus.market.history import CandleType, fetch_window

    with _FETCH_SLOTS:
        bars = fetch_window(symbol, start=datetime.now(UTC) - timedelta(days=500),
                            interval="1D", candle_type=CandleType.MARKET, pause=0.05)
    return [float(b.close) for b in bars if float(b.close) > 0], "Bitget daily candles"


def _daily_technicals(symbol: str, question: str) -> tuple[list[str], list[Source]]:
    """The moving average and RSI a trader reads on the daily chart, computed from daily closes:
    each N-day average asked for (simple, or exponential when "EMA" is said), where the price sits
    against it, the daily Wilder RSI(14), and for a cross the 50-day against the 200-day."""
    from argus.market.skills import rsi

    try:
        closes, whose = _daily_closes(symbol)
    except Exception:
        return [], []
    if len(closes) < 15:
        return [], []
    ticker = _t(symbol)
    last = closes[-1]
    wanted = sorted({int(m.group(1) or m.group(2)) for m in _DAILY_TA.finditer(question)
                     if (m.group(1) or m.group(2)) and 2 <= int(m.group(1) or m.group(2)) <= 400})
    if re.search(r"\b(?:golden|death)\s+cross", question, re.I):
        wanted = sorted({*wanted, 50, 200})
    exponential = bool(re.search(r"\bema\b|exponential", question, re.I))
    lines: list[str] = []
    averages: dict[int, float] = {}
    for n in wanted:
        if len(closes) < n:
            lines.append(f"Missing: the {n}-day average needs {n} daily closes and {whose} hold "
                         f"{len(closes)}, so it is not given.")
            continue
        if exponential:
            k = 2 / (n + 1)
            value = sum(closes[:n]) / n
            for close in closes[n:]:
                value = close * k + value * (1 - k)
        else:
            value = sum(closes[-n:]) / n
        averages[n] = value
        gap = (last / value - 1) * 100
        above = [closes[i] > sum(closes[i - n + 1:i + 1]) / n
                 for i in range(max(n - 1, len(closes) - 20), len(closes))]
        lines.append(f"{ticker}'s {n}-day {'exponential' if exponential else 'simple'} moving "
                     f"average is {value:,.2f}; the last daily close, {last:,.2f}, is "
                     f"{abs(gap):.1f}% {'above' if gap >= 0 else 'below'} it"
                     + ("" if exponential else
                        f", and closed above it on {sum(above)} of the last {len(above)} days")
                     + ".")
    daily_rsi = rsi(closes[-250:])
    if daily_rsi is not None:
        state = ("overbought" if daily_rsi >= 70 else "oversold" if daily_rsi <= 30
                 else "neutral")
        lines.append(f"RSI(14, 1D) {daily_rsi:.1f} — {state}, on the daily chart.")
    cross_line = None
    if 50 in averages and 200 in averages and len(closes) >= 201 and not exponential:
        # the 50-day less the 200-day on each of the last days, to date the most recent cross
        spread = [sum(closes[i - 49:i + 1]) / 50 - sum(closes[i - 199:i + 1]) / 200
                  for i in range(199, len(closes))]
        now = spread[-1]
        since = next((len(spread) - 1 - i for i in range(len(spread) - 1, 0, -1)
                      if (spread[i] > 0) != (spread[i - 1] > 0)), None)
        kind = "golden cross (the 50-day rising through the 200-day)" if now > 0 else \
            "death cross (the 50-day falling through the 200-day)"
        cross_line = (f"{ticker}'s 50-day average is {abs(now / averages[200] * 100):.1f}% "
                      f"{'above' if now > 0 else 'below'} its 200-day"
                      + (f"; the last cross was a {kind} {since} trading day(s) ago"
                         if since is not None else
                         "; the two have not crossed in the history read here")
                      + ".")
        lines.append(cross_line)
    if not lines:
        return [], []
    lead = next((line for line in lines if not line.startswith("Missing")), lines[0])
    if cross_line is not None and re.search(r"\bcross", question, re.I):
        lead = cross_line
        asked = re.search(r"\b(golden|death)\s+cross", question, re.I)
        if asked is not None:
            # "is SPY in a death cross?" is a yes-or-no question; the answer says which first
            golden_now = averages[50] > averages[200]
            yes = golden_now == (asked.group(1).lower() == "golden")
            lead = (f"{'Yes' if yes else 'No'} — {ticker} is in "
                    f"{'golden' if golden_now else 'death'}-cross territory. {cross_line}")
    elif not wanted and daily_rsi is not None:
        lead = next(line for line in lines if line.startswith("RSI(14, 1D)"))
    lines.remove(cross_line if cross_line is not None and lead.endswith(cross_line) else lead)
    lines.insert(0, f"Bottom line: {lead}" if not lead.startswith("Missing") else lead)
    lines.append(f"Computed by ARGUS from {whose}, {len(closes)} days"
                 + ("; a moving average describes where price has been, not where it goes."
                    if averages else "."))
    return lines, [Source(kind="computation", ref="argus.lui.research.quote._daily_technicals",
                          detail=f"{whose}; SMA/EMA and Wilder RSI(14) on daily closes")]


def _technicals_computed(symbol: str) -> tuple[list[str], list[Source]]:
    """The same reading as :func:`_technicals`, from :func:`_indicators` alone — used when the
    Skill has no series for a contract."""
    from argus.market import skills
    from argus.market.skills import indicators

    with _FETCH_SLOTS:
        got = indicators(symbol)
    if got is None:
        return [], []
    lines: list[str] = []
    reading: list[str] = []
    if "rsi" in got:
        value = float(got["rsi"])
        state = "overbought" if value >= 70 else "oversold" if value <= 30 else "neutral"
        lines.append(f"RSI(14, 4h) {value:.1f} — {state}.")
        if state != "neutral":
            reading.append(f"RSI is {state}")
    histogram = float(got["histogram"])
    cross = str(got["cross"])
    lines.append(f"MACD {float(got['dif']):.3f} vs signal {float(got['dea']):.3f} (histogram "
                 f"{histogram:+.3f})" + (f", {cross}" if cross else "") + ".")
    reading.append("momentum turning up" if histogram > 0 else "momentum turning down")
    atr = float(got["atr"])
    lines.append(f"ATR {atr:.3f} (4h), {atr / float(got['close']) * 100:.2f}% of price.")
    bars = int(float(got["bars"]))
    if bars < skills.SHORT_HISTORY_BARS:
        lines.append(f"Short history: {_t(symbol)} has {bars} four-hour bars on Bitget, since "
                     f"{str(got['since'])[:10]}, so these readings rest on little data.")
    lines.insert(0, "Bottom line: " + "; ".join(reading) + ".")
    lines.append("Caveat: technicals describe the tape, not an edge — the systematic signals this "
                 "desk tested on these names did not clear costs.")
    return lines, [Source(kind="computation", ref="argus.lui.research._indicators",
                          detail=f"{symbol} 4H, {int(float(got['bars']))} bars from Bitget")]


def _overnight_record(symbol: str) -> dict[str, Any] | None:
    """The implied open's record on this stock when it was scored, else across all scored stocks.
    Per stock because the pooled figure would overstate the lead where it is small: on QQQ the
    index futures gloaming reads are nearly the same instrument, and the two came out level."""
    from argus.lui.answer import desk_notes_path

    try:
        report = json.loads((desk_notes_path().parent / "overnight_comparison.json")
                            .read_text(encoding="utf-8"))
        rival = report["best_gloaming"]
        own = report["per_stock"].get(_t(symbol))
        if own:
            paired = own["argus_perp_vs_best_gloaming"]
            return {"scope": f"{_t(symbol)} on {own['nights']} nights",
                    "argus": own["argus_perp"], "hit": own["argus_perp_direction_hit_rate"],
                    "rival": own[rival], "zero": own["zero"],
                    "separable": paired["verdict"] != "not separable"}
        summary = report["summary"]
        return {"scope": f"{report['nights']} nights across {len(report['per_stock'])} stocks",
                "argus": summary["argus_perp"]["mae_bps"],
                "hit": summary["argus_perp"]["direction_hit_rate"],
                "rival": summary[rival]["mae_bps"], "zero": summary["zero"]["mae_bps"],
                "separable": True}
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _implied_open_line(symbol: str, perp_last: Decimal,
                       now: datetime | None = None) -> tuple[str, Source] | None:
    """While the US market is shut: where the stock should open, read from its own perpetual's
    move since the last regular close, with that reading's measured record.

    `eval/overnight_comparison.py` scored it on every night since April across eight names
    against gloaming's overnight fair value (index futures, BTC/ETH and the dollar, blended as it
    ships and OLS-refit) and against assuming no gap; the perpetual's own move was the closest to
    the real open, and adding the futures to it did not improve it, so it is used alone."""
    from argus.market import universe

    if symbol not in TRADED_SYMBOLS and not universe.is_equity(symbol):
        return None
    instant = now or datetime.now(UTC)
    close = _last_regular_close(instant)
    if close is None:
        return None
    # The regular close only: an extended-hours quote is not a close (see `_shut_close`), so
    # without the daily bar there is no line rather than one read off a pre-market print.
    stock = _yahoo_close(symbol, close)
    if stock is None:
        return None
    perp_close = _perp_at_close(symbol, close)
    if perp_close is None:
        return None
    move = float(perp_last) / perp_close - 1.0
    implied = stock * (1.0 + move)
    text = (f"Implied open: {_t(symbol)}'s perpetual has moved {move * 100:+.2f}% since the "
            f"{close.astimezone(UTC):%a %H:%M} UTC close, which puts the stock near "
            f"{implied:,.2f} at the next open (last close {stock:g}).")
    record = _overnight_record(symbol)
    if record:
        tie = "" if record["separable"] else " (a difference too small to call)"
        text += (f" Read this way on {record['scope']}, the open was missed by "
                 f"{record['argus']:.0f}bps on average, with the direction right "
                 f"{record['hit'] * 100:.0f}% of the time; gloaming, an S2 desk that estimates "
                 f"overnight fair value from index futures, crypto and the dollar, missed by "
                 f"{record['rival']:.0f}bps{tie}, and assuming no gap by {record['zero']:.0f}bps.")
    return text, Source(kind="computation", ref="argus.eval.overnight_comparison",
                        detail=f"{symbol} 1H close at {close.astimezone(UTC):%Y-%m-%d %H:%M} "
                               f"UTC {perp_close:g}; stock last {stock:g}")


# Every engine here is a traced step from import on (lui/trace.py, trace_module).
trace_module(globals())
