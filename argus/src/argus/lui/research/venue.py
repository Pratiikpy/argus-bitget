"""Where a name trades and what holding it costs: the venue, leverage, funding and closure risk."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from argus.desk.portfolio import (
    align,
    beta,
    correlation,
    variance,
)
from argus.lui.answer import Source
from argus.lui.question import (
    TRADED_SYMBOLS,
)
from argus.lui.research.anchor import (
    _premium_line,
    _rtoken_tracking,
)
from argus.lui.research.data import (
    _FETCH_SLOTS,
)
from argus.lui.research.kinds import (
    BENCHMARK,
    CRYPTO_ANCHOR,
    CRYPTO_LINKED,
    _t,
)
from argus.lui.research.parse import (
    is_us_equity,
)
from argus.lui.research.riskmath import (
    _moments,
)
from argus.lui.trace import trace_module


def _crypto_anchor_line(symbol: str, raw: Mapping[str, Mapping[datetime, float]],
                        columns: Mapping[str, Sequence[float]]) -> str | None:
    """BTC beta and correlation over every hour, beside how much QQQ explains in open hours.

    Both trade around the clock on Bitget, so every aligned hour counts; QQQ's figure is the
    open-session one the rest of the answer uses. Says which of the two prices the name."""
    if symbol not in CRYPTO_LINKED or CRYPTO_ANCHOR not in raw:
        return None
    _stamps, both = align({k: v for k, v in raw.items() if k in (symbol, CRYPTO_ANCHOR)})
    b = beta(both.get(symbol, []), both.get(CRYPTO_ANCHOR, []))
    rho = correlation(both.get(symbol, []), both.get(CRYPTO_ANCHOR, []))
    if b is None or rho is None:
        return None
    qqq = correlation(columns.get(symbol, []), columns.get(BENCHMARK, []))
    verdict = ""
    if qqq is not None:
        verdict = (f"; QQQ explains {qqq * qqq:.0%} in open hours, so BTC, not QQQ, is the "
                   f"market that prices it and the hedge to size" if rho * rho > qqq * qqq else
                   f"; QQQ explains {qqq * qqq:.0%} in open hours, so the equity market prices it "
                   f"more than BTC does")
    return (f"{_t(symbol)} against bitcoin: {b:.2f}x BTC's hourly move across all "
            f"{len(both.get(symbol, []))} aligned hours, correlation {rho:+.2f} (R² "
            f"{rho * rho:.0%}: the share of its moves bitcoin's explain){verdict}.")


def _distribution_line(symbol: str, raw: Mapping[str, Mapping[datetime, float]],
                       columns: Mapping[str, Sequence[float]]) -> str | None:
    """The shape of a single name's returns: annualised volatility, skew, excess kurtosis, and how
    much of its open-session move QQQ explains (beta's R²). A trader asking "how risky is it" is
    also asking how fat its tails are, and a beta with an R² of 0.1 is not a hedge ratio anyone
    should lean on — both were asked for on the held-out corpus and neither was answered."""
    hourly = list(raw.get(symbol, {}).values())
    shape = _moments(hourly)
    var = variance(hourly)
    if shape is None or var is None:
        return None
    vol = math.sqrt(var) * math.sqrt(24 * 365)
    rho = correlation(columns.get(symbol, []), columns.get(BENCHMARK, []))
    r2 = "" if rho is None else (
        f"; QQQ explains {rho * rho:.0%} of its open-session moves (R²), so a QQQ hedge "
        + ("covers most of its risk" if rho * rho >= 0.5 else "leaves most of its risk in place"))
    tails = ("fat-tailed — large hourly moves are far more common than a normal curve implies"
             if shape[1] > 3 else "close to normal in its tails" if shape[1] < 1 else
             "moderately fat-tailed")
    # Each statistic with its meaning beside it: a first-time user met "skew +0.81" and "R²" with
    # no word on what either says (the round-8 first-user audit, 2026-09-30).
    lean = ("the large moves have been upward more often than downward" if shape[0] > 0.3 else
            "the large moves have been downward more often than upward" if shape[0] < -0.3 else
            "large moves have come about as often up as down")
    return (f"Return shape over {len(hourly)} hourly bars: realised volatility {vol:.0%} a year "
            f"(how far it typically swings), skew {shape[0]:+.2f} ({lean}), excess kurtosis "
            f"{shape[1]:.1f} ({tails}){r2}.")


def _venue(symbol: str, is_open: Any, spot: str | None = None
           ) -> tuple[list[str], list[Source]]:
    """Bitget's real-world-asset book as it stands, and one name worked through."""
    from argus.cost.model import CostModel
    from argus.market import universe
    from argus.market.bitget import fetch_tickers

    listed = universe.contracts()
    kinds = {"equity": 0, "commodity": 0, "fx": 0, "index": 0}
    for sym, contract in listed.items():
        if contract.rwa:
            kinds[universe.NOT_EQUITY.get(sym, "equity")] += 1
    crypto = sum(1 for c in listed.values() if not c.rwa)
    model = CostModel.bitget_perp()
    base = symbol.removesuffix("USDT")
    spot_count: int | None = None
    try:
        from argus.market.rtoken_spot import spot_rtokens

        spot_count = len(spot_rtokens())
    except Exception:
        spot_count = None
    lines = [
        "Bitget offers a US stock two ways. The rToken is a spot token (R" + base + "USDT for "
        + _t(symbol) + ") that you own outright: no leverage and no funding. The stock perpetual ("
        + symbol + ") is a USDT-margined contract: leverage, shorting and a funding payment "
        "every few hours. Both trade around the clock, including when the stock's own market "
        "is shut. ARGUS's own paper desk decides on the perpetuals and places no real orders; "
        "this console holds no money and never trades for you.",
        (f"Listed now: {spot_count:,} spot rTokens, and " if spot_count else "Listed now: ")
        + f"{sum(kinds.values())} real-world-asset perpetuals beside {crypto} crypto ones — "
        f"{kinds['equity']} stocks and ETFs, {kinds['commodity']} commodities, {kinds['fx']} "
        f"currency pairs and {kinds['index']} index products.",
        f"A perpetual trade costs {model.taker_bps:.0f}bps a side as taker "
        f"({model.round_trip_bps():.0f}bps a round trip).",
    ]
    ticker = None
    try:
        ticker = fetch_tickers().get(symbol)
    except Exception:
        ticker = None
    carry = None
    if ticker is not None:
        rate = float(ticker.funding_rate) * 100
        hours = (listed.get(symbol) or universe.Contract(symbol, True)).funding_hours or 8
        carry = rate * 24 / hours * 365
        lines.append(f"Holding costs funding: {_t(symbol)} is at {rate:+.4f}% every {hours}h right "
                     f"now, about {carry:+.1f}% a year for a long if it held — owning the share "
                     f"costs nothing to hold.")
        premium = _premium_line(symbol, ticker.last, is_open(datetime.now(UTC)))
        if premium is not None:
            lines.append(premium[0])
    if spot:
        tracked = _rtoken_tracking(spot, symbol, 30)
        if tracked is not None:
            lines.append(f"How closely it tracks: {tracked[0][0].lower()}{tracked[0][1:]}")
        lines.append("Whether an rToken pays dividends, is backed one-to-one by the share or can "
                     "be redeemed for it is set by Bitget's rToken terms, which this desk has not "
                     "verified — read them on Bitget before relying on any of it. What is measured "
                     "here is the price.")
    lines.append("Neither is the share: no vote, and outside US hours both are priced on "
                 "Bitget's own books, not the exchange's. An rToken holder who wants protection "
                 "while the US market is shut can short the same company's perpetual — ask "
                 f"\"how do I hedge my r{base} over the weekend\" for the tested ratio.")
    lines.insert(0, (
        "Bottom line: use the perpetual for leverage, shorting, or to hedge a book around the "
        "clock; the spot rToken to hold exposure without funding; for a long buy-and-hold "
        "position the share itself costs least to carry"
        + (f" — at today's funding a perpetual long pays about {carry:.1f}% a year."
           if carry is not None and carry > 0 else ".")))
    return lines, [Source(kind="venue", ref="bitget contracts + tickers + bitget-mcp-server",
                          detail=f"{len(listed)} contracts; {_t(symbol)} as the worked example")]


def _leverage(symbol: str, multiple: float, side: str, *, closure: str | None = None,
              notional: float | None = None) -> tuple[list[str], list[Source], dict[str, Any]]:
    """How a ``multiple``-times position on ``side`` in ``symbol`` fares against the last 30 days of
    hourly highs and lows: the liquidation distance, how often a day's adverse move reached it, the
    worst adverse day and the largest multiple that survives it, and funding at that multiple."""
    from argus.market.bitget import fetch_tickers
    from argus.market.history import CandleType, fetch

    try:
        with _FETCH_SLOTS:
            bars = fetch(symbol, interval="1H", candle_type=CandleType.MARKET, recent=True,
                         limit=720)
    except Exception:
        return [], [], {}
    if len(bars) < 48 or multiple <= 0:
        return [], [], {}
    # Liquidation distance: the margin (1/leverage) less Bitget's maintenance margin at this
    # position's tier, before fees. At 20x the whole margin is gone after an adverse 5%.
    from argus.market.bitget import maintenance_margin_rate

    mmr = maintenance_margin_rate(symbol, notional or 1000.0)
    distance = 1.0 / multiple - (mmr or 0.0)
    from argus.market.bitget import max_leverage

    allowed = max_leverage(symbol, notional or 1000.0)
    if distance <= 0 or (allowed is not None and multiple > allowed):
        # "I'm 500x long ETH" was given a liquidation price above its own entry and never told
        # the position cannot be opened (a hostile review, round 22)
        why = (f"its {1 / multiple:.2%} margin is at or below Bitget's {mmr:.2%} maintenance "
               f"margin, so it would be liquidated the moment it opened"
               if distance <= 0 and mmr is not None else
               f"Bitget allows at most {allowed:g}x on {_t(symbol)} at this size")
        cap = (f" The most Bitget allows here is {allowed:g}x, where a "
               f"{1 / allowed - (mmr or 0.0):.2%} move against it wipes the margin."
               if allowed is not None else "")
        return ([f"Bottom line: a {multiple:g}x {side} on {_t(symbol)} cannot be opened — "
                 f"{why}.{cap} So yes: at {multiple:g}x, any move against it is a "
                 f"liquidation.",
                 "Data: Bitget's public position tier table (query-position-lever) for "
                 f"{_t(symbol)}. This is analysis, not advice — you make the call."],
                [Source(kind="venue", ref="bitget /api/v2/mix/market/query-position-lever",
                        detail=f"{_t(symbol)} tiers")],
                {"leverage": multiple, "max_leverage": allowed, "maintenance_margin_rate": mmr})
    worst = 0.0
    hits = 0
    windows = 0
    for i in range(len(bars) - 24):
        entry = float(bars[i].close)
        if entry <= 0:
            continue
        ahead = bars[i + 1:i + 25]
        if side == "short":
            adverse = max(float(b.high) for b in ahead) / entry - 1
        else:
            adverse = 1 - min(float(b.low) for b in ahead) / entry
        worst = max(worst, adverse)
        windows += 1
        hits += adverse >= distance
    days = windows / 24
    # The largest multiple whose liquidation line (1/L less the maintenance margin) sits beyond
    # the worst day. Computed without the maintenance margin until 2026-09-25, which let "10x
    # would have survived" sit beside a 9.9% worst day and an 8.0% line.
    cushion = worst + (mmr or 0.0)
    survive = int(1 / cushion + 1e-9) if cushion > 0 else None  # 1/0.10000000000000009 is 9.999…
    verb = "rise" if side == "short" else "fall"
    ticker = _t(symbol)
    lines = [
        f"Bottom line: at {multiple:g}x {side} a {distance:.1%} {verb} in {ticker} wipes the margin"
        # "from 0% of the hourly entry points" was unreadable to a newcomer (round 22)
        f" — of {side}s opened at each hour of the last {days:.0f} days, {hits / windows:.0%} "
        f"would have been wiped out within 24 hours. Its worst 24 hours against a "
        # from hourly highs and lows, which is wider than the close-to-close worst day other
        # answers quote; a first-time user saw 5.4% here and -4.2% there with no reason given
        f"{side} was {worst:.1%}, measured from hourly highs and lows (wider than a close-to-close "
        f"figure)"
        + ("." if not survive else
           # "10x would have survived every day (anything up to 17x did)" read as a green light to
           # a newcomer (round 19, row 640): it is what one month allowed, and it says so
           f": {multiple:g}x stayed clear of liquidation in this one month — up to {survive}x "
           f"would have — which says nothing about the next month's worst day." if multiple <=
           survive else
           f", which only {survive}x or less would have survived.")
    ]
    try:
        ticker_row = fetch_tickers().get(symbol)
    except Exception:
        ticker_row = None
    if ticker_row is not None and float(ticker_row.last) > 0:
        # The price itself: "where is my liquidation price on a 10x BTC long?" was answered with
        # a distance only (2026-09-25 audit, round 2). Isolated margin, opened at the last price.
        entry_price = float(ticker_row.last)
        liq = entry_price * (1 - distance if side == "long" else 1 + distance)
        lines.append(f"Liquidation price: a {multiple:g}x {side} opened at the last price, "
                     f"{entry_price:,.6g}, is liquidated near {liq:,.6g} on isolated margin, "
                     f"{distance:.1%} {'below' if side == 'long' else 'above'} entry; cross "
                     f"margin moves it by whatever else the account holds.")
        payload_liq: float | None = liq
    else:
        payload_liq = None
    if ticker_row is not None:
        rate = float(ticker_row.funding_rate) * 100
        pays = (rate > 0 and side == "long") or (rate < 0 and side == "short")
        lines.append(
            "Funding: flat right now, so holding costs nothing beyond fees." if rate == 0 else
            f"Funding: {rate:+.4f}% of the position per interval; a {side} "
            + ("pays" if pays else "receives")
            + f" it, which at {multiple:g}x is {abs(rate) * multiple:.3f}% of your margin each "
              f"interval.")
    lines.append(
        (f"The liquidation distance is the {1 / multiple:.1%} margin less Bitget's {mmr:.2%} "
         f"maintenance margin at this position's tier, before fees, which bring it closer"
         if mmr is not None else
         "The liquidation distance is before fees and Bitget's maintenance margin (its tier "
         "table did not answer), both of which move it closer")
        + "; hourly highs and lows understate the extremes inside each hour.")
    payload: dict[str, Any] = {
        "leverage": multiple, "side": side, "liquidation_distance": distance,
        "maintenance_margin_rate": mmr, "hit_rate_24h": hits / windows if windows else None,
        "worst_adverse_24h": worst, "survivable_leverage": survive,
        "liquidation_price": payload_liq}
    sources = [Source(kind="computation", ref="argus.lui.research.venue._leverage",
                      detail=f"{symbol} 1H highs/lows, {len(bars)} bars")]
    if mmr is not None:
        sources.append(Source(kind="venue", ref="bitget /api/v2/mix/market/query-position-lever",
                              detail=f"{symbol} maintenance margin tier"))
    if closure is not None:
        closure_lines, closure_sources, closure_payload = _closure_risk(
            symbol, multiple, side, distance, closure, ticker_row)
        # The closure's own funding line replaces the per-interval one.
        lines = [lines[0], *closure_lines,
                 *(line for line in lines[1:] if not line.startswith("Funding:"))]
        sources.extend(closure_sources)
        payload["closure"] = closure_payload
    return lines, sources, payload


def _closure_risk(symbol: str, multiple: float, side: str, distance: float, closure: str,
                  ticker_row: Any) -> tuple[list[str], list[Source], dict[str, Any]]:
    """What a leveraged position faces across the hours the stock does not trade.

    Two records, because they answer different halves. The stock's own history since listing
    (`market/equity_history.py`, Yahoo daily, split-adjusted) says how far the price has jumped
    from Friday's close to Monday's open — decades of weekends. But a Bitget perpetual keeps
    trading through the weekend and is liquidated on its own path, not on Monday's open, so the
    perpetual's own worst point inside each past weekend (`desk/odds.py` adverse excursion on
    Bitget's daily lows and highs) is the half that decides a liquidation. baserate reads only the
    first; its idea of a long weekend record is taken, none of its code."""
    from argus.desk.odds import directional_odds
    from argus.market import equity_history
    from argus.market.history import CandleType, fetch_window

    lines: list[str] = []
    sources: list[Source] = []
    payload: dict[str, Any] = {"closure": closure}
    name = _t(symbol)
    if closure == "weekend" and (symbol in TRADED_SYMBOLS or is_us_equity(symbol)):
        try:
            stock_days = equity_history.daily(name)
            gaps = equity_history.closure_gaps(stock_days)
            rec = equity_history.record(gaps, side=side, adverse=distance)
            matched = equity_history.matched_record(stock_days, gaps, side=side)
            band = equity_history.weekend_band(stock_days)
        except equity_history.HistoryError:
            rec, matched, band = None, None, None
        if rec is not None:
            worst_move = rec.worst.move * (-1 if side == "short" else 1)
            lines.append(
                f"{name}'s own weekends since {rec.since.year} ({rec.n:,} of them, Friday close "
                f"to Monday open, split-adjusted): {rec.up:.0%} opened more than 1% in a "
                f"{side}'s favour, {rec.flat:.0%} within 1%, {rec.down_1_5:.0%} 1-5% against, "
                f"{rec.down_5:.0%} more than 5% against. The worst was {worst_move:+.1%} "
                f"({rec.worst.closed:%d %b %Y}); {rec.beyond} of {rec.n:,} opened past the "
                f"{distance:.1%} liquidation line.")
            sources.append(Source(kind="venue", ref="Yahoo Finance daily chart",
                                  detail=f"{name} since {rec.since.isoformat()}, adjusted"))
            payload["stock_weekends"] = rec.as_dict()
        if band is not None:
            # Scaled to the volatility the stock has now and held to its stated coverage, the
            # forecast `eval/weekend_quantiles.py` found best of six on 17,044 weekends (audit
            # finding 27): baserate's trend-and-momentum match is shown below, and scored worse.
            crosses = band.p10 <= -distance if side != "short" else band.p90 >= distance
            lines.append(
                f"Scaled to {name}'s volatility now ({band.ewma_vol:.1%} a day, RiskMetrics' "
                f"exponentially weighted measure), its past weekends put Monday's open between "
                f"{band.p10:+.1%} and {band.p90:+.1%} of Friday's close in eight weekends out of "
                f"ten (middle {band.p50:+.1%}); the {distance:.1%} liquidation line is "
                + ("inside that band." if crosses else "outside it.")
                + f" Bands built this way covered {band.covered:.0%} of the {band.tracked:,} past "
                  f"{name} weekends they were tested on, out of sample.")
            sources.append(Source(kind="computation",
                                  ref="argus.market.equity_history:weekend_band",
                                  detail="EWMA-scaled weekends, adaptive conformal band; chosen "
                                         "by data/weekend_quantiles.json"))
            payload["weekend_band"] = band.as_dict()
        if matched is not None:
            # baserate matches the current regime to past weekends and filters; the filter is
            # kept here and tested: the matched share is shown with its interval, and only
            # called different when the interval excludes the share across all weekends.
            signed = -1 if side == "short" else 1
            worst_note = ("" if matched.worst is None else
                          f" The worst of them was {matched.worst.move * signed:+.1%}"
                          f" ({matched.worst.closed:%d %b %Y}).")
            lines.append(
                f"Going into this weekend {name} is {matched.state.words()}. In the "
                f"{matched.n} past weekends that began that way, {matched.against_share:.0%} "
                f"opened more than 1% against a {side} (95% interval {matched.against_low:.0%} to "
                f"{matched.against_high:.0%}), against {matched.all_against_share:.0%} across all "
                f"weekends — "
                + ("a measurable difference, found in-sample." if matched.differs else
                   "no measurable difference, so the state adds nothing here.")
                + worst_note
                + (" Scored out of sample on 13 stocks, this kind of match forecast weekends only "
                   "marginally better than the plain record, and worse than the volatility-scaled "
                   "band above." if band is not None else ""))
            payload["matched_weekends"] = matched.as_dict()
    try:
        with _FETCH_SLOTS:
            daily = fetch_window(symbol, start=datetime.now(UTC) - timedelta(days=500),
                                 interval="1D", candle_type=CandleType.MARKET, pause=0.05)
    except Exception:
        daily = []
    kept = [b for b in daily if float(b.close) > 0]
    odds = directional_odds(
        [(b.ts + timedelta(days=1), float(b.close)) for b in kept], 1, cost_bps=0.0, side=side,
        weekend=closure == "weekend",
        extremes=[(float(b.low), float(b.high)) for b in kept]) if closure == "weekend" else None
    if odds is not None and odds.adverse_p90_bps is not None:
        breached = 0
        index = {(b.ts + timedelta(days=1)).date(): k for k, b in enumerate(kept)}
        for k, b in enumerate(kept):
            closed = b.ts + timedelta(days=1)
            if closed.weekday() != 4:
                continue
            j = index.get((closed + timedelta(days=3)).date())
            if j is None:
                continue
            start = float(b.close)
            window = kept[k + 1:j + 1]
            worst = (max(float(x.high) for x in window) / start - 1 if side == "short"
                     else 1 - min(float(x.low) for x in window) / start)
            breached += worst >= distance
        lines.append(
            f"On the perpetual, which trades through the weekend and is liquidated on its own "
            f"path: in 90% of its last {odds.windows} weekends the worst point was within "
            f"{abs(odds.adverse_p90_bps) / 100:.1f}% of Friday's close, and {breached} of "
            f"{odds.windows} touched the {distance:.1%} line.")
        sources.append(Source(kind="venue", ref="bitget /api/v3/market/history-candles",
                              detail=f"{symbol} 1D lows/highs, {odds.windows} weekends"))
        payload["perp_weekends"] = {"windows": odds.windows, "p90_adverse_bps":
                                    odds.adverse_p90_bps, "touched_liquidation": breached}
    if ticker_row is not None:
        from argus.market import universe

        listed = universe.contracts().get(symbol) or universe.Contract(symbol, False)
        interval = listed.funding_hours or 8
        hours = 72 if closure == "weekend" else 16
        rate = float(ticker_row.funding_rate)
        carry = rate * (hours / interval) * multiple * (1 if side == "long" else -1)
        lines.append(
            f"Funding across the {closure} ({hours}h, {hours // interval} settlements) at "
            f"today's rate: " + ("nothing — the rate is flat." if carry == 0 else
                                 f"{'costs' if carry > 0 else 'pays you'} about "
                                 f"{abs(carry):.2%} of your margin."))
        payload["funding_carry_of_margin"] = carry
    return lines, sources, payload


def _funding_meaning(symbol: str, rate_pct: float) -> str | None:
    """What a funding rate means for someone holding the contract: who pays whom, and what it
    costs over a day and a year at the contract's own settlement interval."""
    from argus.market import universe

    hours = (universe.contracts().get(symbol) or universe.Contract(symbol, False)).funding_hours
    if not hours or rate_pct == 0:
        return (f"Funding: flat, so holding {_t(symbol)} costs nothing beyond fees."
                if rate_pct == 0 else None)
    per_day = rate_pct * 24 / hours
    payer, payee = ("longs", "shorts") if rate_pct > 0 else ("shorts", "longs")
    crowd = ("a crowded long" if rate_pct > 0.03 else "a crowded short" if rate_pct < -0.03
             else "no crowding either way")
    return (f"Funding means {payer} pay {payee} every {hours}h: {abs(per_day):.3f}% of the "
            f"position a day, about {abs(per_day) * 365:.1f}% a year if it held — {crowd}.")


# Every engine here is a traced step from import on (lui/trace.py, trace_module).
trace_module(globals())
