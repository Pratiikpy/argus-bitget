"""Sentiment answers: Fear & Greed through bitget-signal's Skill first, funding, long/short and the
crowd."""

from __future__ import annotations

from typing import Any

from argus.lui.answer import Source
from argus.lui.question import (
    TRADED_SYMBOLS,
)
from argus.lui.research.evidence import (
    CRYPTO_SENTIMENT,
    _coordination_test,
    _desk_integrity_read,
    _skill_btc_line,
    _skill_fear_greed,
)
from argus.lui.research.kinds import (
    BENCHMARK,
    _t,
)
from argus.lui.research.news import (
    NEWS_LOOKBACK_HOURS,
    _headlines_naming,
)
from argus.lui.research.parse import (
    is_us_equity,
)
from argus.lui.research.venue import (
    _funding_meaning,
)
from argus.lui.skillroute import Routed
from argus.lui.skillroute import route as skill_route
from argus.lui.trace import trace_module
from argus.market.skills import Health
from argus.truth import http
from argus.truth.coverage import ContextPool


def _sentiment(symbol: str | None = None) -> tuple[list[str], list[Source], dict[str, Any]]:
    """Whether the talk about a name is information or repetition, and how it is positioned.

    For a named contract: its headlines grouped into stories so a syndicated article counts once
    (`market/stories.py`), the desk's own last integrity read of it, the measured coordinated-
    posting result, and its funding and 24-hour run on Bitget. For the market: the crypto fear &
    greed index and its week, with BTC and ETH funding as positioning. The Market Sentiment
    capability is OWNED for the integrity layer — "five accounts repeating one article is one
    source" — and until 2026-09-24 the console showed none of it (a judge audit's finding)."""

    from argus.market.bitget import fetch_tickers
    from argus.market.stories import group

    def alternative_week(ident: str, args: dict[str, Any]
                         ) -> tuple[Health, str, Any, str, bool]:
        # The mirror for bitget-signal's sentiment_index: alternative.me, the upstream the tool's
        # own description names, read for its last eight days so the answer keeps its week.
        del ident, args
        upstream = "alternative.me Fear & Greed"
        try:
            week = list(http.fetch_json("https://api.alternative.me/fng/?limit=8",
                                        timeout=10).get("data") or [])
        except Exception as exc:
            return Health.UNAVAILABLE, type(exc).__name__, None, upstream, True
        return (Health.OK if week else Health.EMPTY), "ok", week, upstream, True

    def fear_greed() -> tuple[list[dict[str, Any]], Routed | None]:
        # Bitget's data service first; then bitget-signal's sentiment Skill; then alternative.me,
        # the source that Skill names, labelled as such (lui/skillroute.py).
        try:
            from argus.market.bitget_positioning import crypto_mood

            mood = crypto_mood()
        except Exception:
            mood = None
        if mood is not None:
            value, label, week = mood
            return ([{"value": v, "value_classification": label if i == 0 else "",
                      "via": "bitget"} for i, v in enumerate(week)] or [
                {"value": value, "value_classification": label, "via": "bitget"}]), None
        routed = skill_route("sentiment_index", "current", mirror_call=alternative_week)
        if routed.via == "skill":
            reading = _skill_fear_greed(routed.payload)
            if reading is not None:
                return [{**reading, "via": "bitget-signal"}], routed
            # The Skill answered in a shape this does not read: take the mirror, and say so.
            health, _, week, upstream, same = alternative_week("", {})
            routed = Routed(routed.tool, routed.action, routed.skill, Health.EMPTY,
                            "answered in an unrecognised shape", "mirror" if week else "none",
                            week, upstream, same, routed.asked_at)
            return (list(week or []) if health is Health.OK else []), routed
        return (list(routed.payload or []) if routed.via == "mirror" else []), routed

    from argus.market import bitget_positioning

    named = symbol is not None and symbol != BENCHMARK and (
        symbol in TRADED_SYMBOLS or is_us_equity(symbol))
    crypto_side = symbol is None or symbol in CRYPTO_SENTIMENT or not named
    base = "ETH" if symbol == "ETHUSDT" else "BTC"
    with ContextPool(max_workers=5) as pool:
        index_job = pool.submit(fear_greed)
        # bitget-signal's crypto_derivatives answers in under a second (data/skill_matrix.json,
        # 2026-09-26), so the BTC side of the answer reads it and checks it against Bitget's own
        # ticker, the way the RSI is checked against Bitget's candles.
        skill_btc_job = (pool.submit(skill_route, "crypto_derivatives", "ticker_24h",
                                     {"symbol": "BTC/USDT", "exchange": "bitget"}, wait=4.0)
                         if crypto_side else None)
        news_job = pool.submit(_headlines_naming, symbol) if named and symbol else None
        mood_job = pool.submit(bitget_positioning.stock_mood) if (named or symbol is None) \
            else None
        crowd_job = (pool.submit(bitget_positioning.positioning, base)
                     if crypto_side and (symbol in (None, "BTCUSDT", "ETHUSDT")
                                         or symbol in CRYPTO_SENTIMENT) else None)
        rows, index_route = index_job.result()
        try:
            headlines, names = news_job.result() if news_job is not None else ([], set())
        except Exception:
            headlines, names = [], set()
        try:
            stock_line = mood_job.result() if mood_job is not None else None
        except Exception:
            stock_line = None
        try:
            crowd_lines = crowd_job.result().lines() if crowd_job is not None else []
        except Exception:
            crowd_lines = []
    extra = [line for line in (stock_line, *crowd_lines) if line]
    oi_symbol = symbol or "BTCUSDT"
    try:
        # Open interest set against its own volume and the whole venue (`market/open_interest`);
        # "ETH open interest" had no answer at all until 2026-09-25.
        from argus.market import open_interest

        reading = open_interest.read(oi_symbol)
        oi_lines = open_interest.lines(reading, _t(oi_symbol)) if reading else []
    except Exception:
        oi_lines = []
    extra.extend(oi_lines)
    try:
        # Bitget's own account and position long/short split, for every perpetual — the ratios
        # above are Binance's and cover BTC and ETH only (2026-09-25 audit, round 2).
        from argus.market import long_short

        split = long_short.read(oi_symbol)
        extra.extend(long_short.lines(split, _t(oi_symbol)) if split else [])
    except Exception:
        pass
    integrity = _desk_integrity_read(symbol) if symbol else None
    tested = _coordination_test(symbol)
    story_line = None
    stories = group(headlines, names) if headlines else []
    if named and symbol:
        ticker = _t(symbol)
        if not headlines:
            story_line = (f"News: nothing in {NEWS_LOOKBACK_HOURS}h names {ticker} in the outlets "
                          f"read here, so there is no chatter to weigh.")
        elif len(stories) == len(headlines):
            story_line = (f"News: {len(headlines)} headline(s) name {ticker} in "
                          f"{NEWS_LOOKBACK_HOURS}h, each a different story — none is carried "
                          f"twice, so the count is not inflated by syndication.")
        else:
            loud = stories[0]
            story_line = (f"News: {len(headlines)} headlines name {ticker} in "
                          f"{NEWS_LOOKBACK_HOURS}h but they are {len(stories)} stories — "
                          f"\u201c{loud.title}\u201d was carried {loud.copies} times "
                          f"({', '.join(loud.outlets)}) and counts once.")
    if not rows and story_line is None and integrity is None:
        return [], [], {}
    now_value = int(rows[0]["value"]) if rows else None
    label = str(rows[0]["value_classification"]) if rows else ""
    week = [int(r["value"]) for r in rows[:8]]
    lines = ([f"Crypto fear & greed: {now_value} ({label}); over the last {len(week)} days it "
              f"ranged {min(week)} to {max(week)}."] if rows else [])
    own: str | None = None
    crowd = ""
    skill_sources: list[Source] = []
    try:
        tickers = fetch_tickers()
        if symbol is not None and symbol in tickers:
            own_ticker = tickers[symbol]
            rate = float(own_ticker.funding_rate) * 100
            meaning = _funding_meaning(symbol, rate)
            change: float | None = float(own_ticker.change_24h) * 100
            crowd = ("crowded long" if rate > 0.03 else "crowded short" if rate < -0.03
                     else "not crowded either way")
            own = (f"{_t(symbol)}'s own positioning is {crowd}"
                   + (f" after a {change:+.1f}% day" if change is not None else ""))
            lines.append(f"{_t(symbol)} on Bitget: funding {rate:+.4f}% per interval"
                         + (f", {change:+.2f}% over 24h" if change is not None else "") + ".")
            if meaning:
                lines.append(meaning)
        cross = _skill_btc_line(skill_btc_job, tickers.get("BTCUSDT"))
        if cross is not None:
            lines.append(cross[0])
            skill_sources.append(cross[1])
        for sym in ("BTCUSDT", "ETHUSDT"):
            if sym == symbol:
                continue
            t = tickers.get(sym)
            if t is not None:
                rate = float(t.funding_rate) * 100
                side = "longs pay shorts" if rate > 0 else "shorts pay longs"
                lines.append(f"{_t(sym)} funding {rate:+.4f}% per interval — {side}, "
                             f"{'a crowded long' if rate > 0.03 else 'no crowding'} on Bitget.")
    except Exception:
        pass
    stretch = ("unavailable" if now_value is None else
               "stretched toward greed — the side of the index where crypto has historically "
               "been more exposed to pullbacks" if now_value >= 70 else
               "stretched toward fear" if now_value <= 30 else "in its middle range")
    if rows and rows[0].get("via") == "bitget":
        sources = [Source(kind="venue", ref="bitget-mcp-server crypto fear & greed + Bitget "
                                           "funding",
                          detail="crypto fear & greed from Bitget's data service and Bitget "
                                 "funding rates, live")]
    else:
        # Which of bitget-signal's Skill or the source it names gave the index, and why.
        sources = ([index_route.source()] if index_route is not None and index_route.answered
                   else [])
        sources.append(Source(kind="venue", ref="Bitget funding",
                              detail="funding rates, live; Bitget's data service did not answer "
                                     "the index"))
    sources.extend(skill_sources)
    if named and symbol:
        ticker = _t(symbol)
        if symbol not in CRYPTO_SENTIMENT:
            # The crypto index and BTC/ETH funding say nothing about how a stock is talked about;
            # for NVDA they were three lines of someone else's market.
            lines = [line for line in lines
                     if not line.startswith(("Crypto fear", "BTC funding", "ETH funding"))]
            sources = [Source(kind="venue", ref="Bitget funding",
                              detail=f"{ticker}'s funding rate and 24h move, live")]
        read = ""
        if integrity is not None:
            at = str(integrity["at"])[:16].replace("T", " ")
            read = (f"the desk's own panel last read it {integrity['signal']} at "
                    f"{integrity['confidence']:.2f} once agreement that shared sources was "
                    f"discounted ({integrity['analysts']} analysts, "
                    f"{integrity['sources']} distinct sources; decision {integrity['seq']}, "
                    f"{at} UTC)")
            sources.append(Source(kind="ledger", ref=f"desk notes, decision {integrity['seq']}",
                                  detail="the analyst panel after the provenance discount"))
        repeated = bool(stories) and len(stories) < len(headlines)
        signal = (integrity is not None and integrity["signal"] in ("bullish", "bearish")
                  and integrity["confidence"] >= 0.5)
        verdict = ("a position signal worth weighing" if signal else "no position signal")
        parts = [p for p in (crowd and f"{ticker}'s funding is {crowd}", read,
                             "its headlines repeat one story" if repeated else "") if p]
        lead = f"Bottom line: {verdict} on {ticker}"
        lead += (" — " + "; ".join(parts) + "." if parts else ".")
        lead += (" Loud and repeated is treated as one source here, not as confirmation."
                 if repeated else "")
        lines.insert(0, lead)
        if story_line:
            lines.insert(1, story_line)
            sources.append(Source(kind="computation", ref="argus.market.stories",
                                  detail=f"{len(headlines)} headline(s) into {len(stories)} "
                                         f"story(ies)"))
        for offset, line in enumerate(extra):
            lines.insert(2 + offset, line)
        if extra:
            sources.append(Source(kind="venue", ref="bitget-mcp-server",
                                  detail="sentiment_market_fear_greed, long/short ratios, "
                                         "Hyperliquid whale positions, liquidations"))
        if tested is not None:
            lines.append(tested[0])
            sources.append(Source(kind="computation", ref="data/sentiment_comparison.json",
                                  detail="finBERT vs the desk's sentiment analyst, coordinated "
                                         "posting"))
        return lines, sources, {"index": now_value, "label": label, "week": week,
                                "headlines": len(headlines), "stories": len(stories),
                                "integrity": integrity}
    lines[1:1] = extra
    if tested is not None:
        lines.append(tested[0])
    if own is not None:
        lines.insert(0, f"Bottom line: {own}; the crypto market backdrop is {stretch}. Read both "
                        f"as positioning, not a signal — this desk's own sentiment work found the "
                        f"index adds nothing on its own, and funding is a cost before it is a "
                        f"view.")
    else:
        lines.insert(0, f"Bottom line: sentiment is {stretch}; read it as positioning, not a "
                        f"signal — this desk's own sentiment work found the index adds nothing on "
                        f"its own.")
    if extra:
        sources.append(Source(kind="venue", ref="bitget-mcp-server",
                              detail="stock-market fear & greed, long/short ratios, Hyperliquid "
                                     "whale positions, liquidations"))
    return lines, sources, {"index": now_value, "label": label, "week": week}


# Every engine here is a traced step from import on (lui/trace.py, trace_module).
trace_module(globals())
