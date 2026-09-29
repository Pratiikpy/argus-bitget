"""News answers: headlines that name the instrument, filings, and bitget-signal's news Skill
first."""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from argus.desk.portfolio import (
    beta,
)
from argus.lui.answer import Source, unlead
from argus.lui.question import (
    TRADED_SYMBOLS,
)
from argus.lui.research.data import (
    load,
)
from argus.lui.research.kinds import (
    BENCHMARK,
    _t,
)
from argus.lui.research.parse import (
    is_us_equity,
)
from argus.lui.research.riskmath import (
    _open_columns,
)
from argus.lui.skillroute import route as skill_route
from argus.lui.trace import trace_module
from argus.market.skills import Health
from argus.truth.coverage import ContextPool

NEWS_LOOKBACK_HOURS = 48


FILING_LOOKBACK_DAYS = 7


MARKET_FEEDS = ("cnbc", "marketwatch", "fed")
"""The outlets read for a market-wide question; for a single name every feed is read and only
headlines that name it are kept."""


def _news_feeds(symbol: str) -> tuple[dict[str, tuple[str, str]], set[str], re.Pattern[str]]:
    """The feeds read for ``symbol``, the names it goes by, and the pattern a headline must match
    to be about it. For QQQ, the market-wide feeds and no name filter."""
    from argus.market.evidence import RSS_FEEDS, YAHOO_SYMBOL_FEED
    from argus.market.universe import ALIASES

    ticker = _t(symbol)
    names = {ticker.lower()}
    for alias, target in ALIASES.items():
        if target == symbol and len(alias) > 3:
            names.add(alias.lower())
    feeds = ({k: v for k, v in RSS_FEEDS.items() if k in MARKET_FEEDS} if symbol == BENCHMARK
             else {**RSS_FEEDS, f"yahoo-{ticker.lower()}": (
                 YAHOO_SYMBOL_FEED.format(ticker=ticker), "news")})
    pattern = re.compile(r"\b(" + "|".join(re.escape(n) for n in sorted(names)) + r")\b", re.I)
    return feeds, names, pattern


def _headlines_naming(symbol: str) -> tuple[list[Any], set[str]]:
    """Headlines of the last ``NEWS_LOOKBACK_HOURS`` that name ``symbol``, newest first, one per
    exact title — the same set the news answer reads."""
    from datetime import timedelta as _td

    from argus.market.evidence import RssSource

    feeds, names, pattern = _news_feeds(symbol)
    rss = RssSource()
    now = datetime.now(UTC)

    def read(item: tuple[str, tuple[str, str]]) -> list[Any]:
        try:
            return rss.headlines(item[0], item[1][0])
        except Exception:
            return []

    with ContextPool(max_workers=len(feeds)) as pool:
        headlines = [h for found in pool.map(read, feeds.items()) for h in found]
    seen: set[str] = set()
    kept = []
    for h in sorted(headlines, key=lambda h: h.published, reverse=True):
        if (h.published < now - _td(hours=NEWS_LOOKBACK_HOURS) or h.published > now
                or h.title in seen or not pattern.search(h.title)):
            continue
        seen.add(h.title)
        kept.append(h)
    return kept, names


TONE_ASKED = (
    ("negative", re.compile(r"\b(?:negative|bad|bearish|worrying|concerning|downbeat|grim)\s+"
                            r"(?:news|headlines?|stories|press)\b|\bany\s+bad\b", re.I)),
    ("positive", re.compile(r"\b(?:positive|good|bullish|upbeat|encouraging)\s+"
                            r"(?:news|headlines?|stories|press)\b|\bany\s+good\b", re.I)),
)
"""A question that asks for one side of the news ("any negative news on NVDA") is answered with
that side: every headline is scored and the ones on the asked side lead. Before 2026-09-29 it got
the same list as any news question (stranger QA)."""


def tone_asked(text: str) -> str | None:
    return next((tone for tone, pattern in TONE_ASKED if pattern.search(text)), None)


def _tone_lines(kept: Sequence[Any], ticker: str, tone: str,
                now: datetime) -> tuple[list[str], dict[str, Any]] | None:
    """Every kept headline scored with VADER (the vendored lexicon the crowd-tone line uses), and
    the ones on the asked side listed first. A lexicon reads words, not events, so the line says
    so; ``None`` when VADER cannot be loaded, and the plain list stands."""
    from argus.market import vader
    from argus.market.social_pulse import vader_tone

    try:
        score = vader_tone()
    except vader.VaderLoadError:
        return None
    scored = [(h, score(h.title)) for h in kept]
    side = [(h, v) for h, v in scored if (v <= vader.NEGATIVE_AT if tone == "negative"
                                          else v >= vader.POSITIVE_AT)]
    side.sort(key=lambda hv: hv[1] if tone == "negative" else -hv[1])
    head = (f"Bottom line: of {len(kept)} headline(s) naming {ticker} in {NEWS_LOOKBACK_HOURS}h, "
            f"{len(side)} read {tone} by VADER's word lexicon"
            + (" — none do, so nothing on the record reads that way." if not side else
               ", listed first below. A lexicon scores wording, not what the news means for "
               "the stock; read them."))
    lines = [head]
    for h, v in side[:5]:
        hours = (now - h.published).total_seconds() / 3600
        outlet = "Yahoo Finance" if h.feed.startswith("yahoo") else h.feed
        lines.append(f"{tone.capitalize()} ({v:+.2f}), {hours:.0f}h ago — {h.title} ({outlet}) "
                     f"{h.link}")
    return lines, {"asked": tone, "scorer": "VADER compound", "on_side": len(side),
                   "scored": len(scored)}


def _news(symbol: str, is_open: Any,
          tone: str | None = None) -> tuple[list[str], list[Source], dict[str, Any]]:
    """Headlines that name ``symbol``, its SEC filings this week, and its 24-hour move split into
    the market's part and its own. For QQQ, the market-wide headlines instead."""
    from datetime import timedelta as _td

    from argus.market.bitget import fetch_tickers
    from argus.market.evidence import EdgarSource, RssSource

    ticker = _t(symbol)
    now = datetime.now(UTC)
    since = now - _td(hours=NEWS_LOOKBACK_HOURS)
    market = symbol == BENCHMARK
    files_with_sec = not market and (symbol in TRADED_SYMBOLS or is_us_equity(symbol))
    feeds, _, pattern = _news_feeds(symbol)
    rss = RssSource()

    def read(item: tuple[str, tuple[str, str]]) -> list[Any]:
        try:
            return rss.headlines(item[0], item[1][0])
        except Exception:
            return []

    with ContextPool(max_workers=len(feeds) + 2) as pool:
        pending = [pool.submit(read, item) for item in feeds.items()]
        tickers_job = pool.submit(fetch_tickers)
        filings_job = None if not files_with_sec else pool.submit(lambda: EdgarSource().filings(
                ticker, since=now - _td(days=FILING_LOOKBACK_DAYS)))
        # bitget-signal's news-briefing Skill is asked beside the feeds read here, and the
        # receipt says which of the two the headlines came from (audit finding 109).
        skill_job = pool.submit(skill_route, "news_feed", "latest",
                                {"keyword": "" if market else ticker, "limit": 5}, wait=4.0,
                                mirror_call=_own_feeds_are_the_mirror)
        headlines = [h for job in pending for h in job.result()]
        try:
            tickers = tickers_job.result()
        except Exception:
            tickers = {}
        try:
            filings = filings_job.result() if filings_job is not None else []
        except Exception:
            filings = []
    seen: set[str] = set()
    kept = []
    for h in sorted(headlines, key=lambda h: h.published, reverse=True):
        if h.published < since or h.published > now or h.title in seen:
            continue
        if not market and not pattern.search(h.title):
            continue
        seen.add(h.title)
        kept.append(h)
    lines: list[str] = []
    move = tickers.get(symbol)
    bench = tickers.get(BENCHMARK)
    change = None if move is None else float(move.change_24h) * 100
    split = ""
    if not market and change is not None and bench is not None:
        data = load((symbol,))
        columns = _open_columns(data.raw, is_open)
        symbol_beta = beta(columns.get(symbol, []), columns.get(BENCHMARK, []))
        if symbol_beta is not None:
            market_part = symbol_beta * float(bench.change_24h) * 100
            own = change - market_part
            split = (f" QQQ moved {float(bench.change_24h) * 100:+.2f}%, which at {ticker}'s "
                     f"beta of {symbol_beta:.2f} explains {market_part:+.2f}%; the other "
                     f"{own:+.2f}% is {ticker}'s own.")
    events = [f for f in filings if f.is_event]
    if change is not None:
        lead = f"{ticker} is {change:+.2f}% over 24 hours on Bitget."
        if market:
            lead = f"The Nasdaq-100 (QQQ on Bitget) is {change:+.2f}% over 24 hours."
        lines.append(lead + split)
    if events:
        f = events[0]
        lines.insert(0, f"Bottom line: {ticker} filed an 8-K on {f.filed:%d %b} ({f.item_summary}) "
                        f"— a company event is on the record; read it before trading the move.")
    elif not market and kept:
        count = "1 headline names" if len(kept) == 1 else f"{len(kept)} headlines name"
        lines.insert(0, f"Bottom line: {count} {ticker} in {NEWS_LOOKBACK_HOURS}h"
                        + (" and no 8-K was filed this week — no company event on the record, so "
                           "the move is sentiment or the market, not a disclosed fact."
                           if files_with_sec else
                           " — read them against the split below: the market explains the part "
                           "its beta carries, the rest is its own."))
    elif not market:
        lines.insert(0, f"Bottom line: nothing in {NEWS_LOOKBACK_HOURS}h names {ticker}"
                        + (" and no 8-K was filed this week" if files_with_sec else "")
                        + " — whatever moved it is not in the news sources read here; the split "
                          "below says how much of it the market explains.")
    else:
        lines.insert(0, "Bottom line: the headlines below are what the market is reading; none of "
                        "them is a cause until a price reaction can be tied to it.")
    toned = _tone_lines(kept, ticker, tone, now) if tone and kept and not market else None
    if toned is not None:
        # The asked side leads; the 8-K lead, when there is one, still comes first.
        at = 1 if events else 0
        if not events:
            lines[0] = unlead(lines[0])
        lines[at:at] = toned[0]
    listed = {line.split(" — ", 1)[1] for line in (toned[0][1:] if toned else [])
              if " — " in line}
    for h in kept[:5]:
        hours = (now - h.published).total_seconds() / 3600
        outlet = "Yahoo Finance" if h.feed.startswith("yahoo") else h.feed
        if f"{h.title} ({outlet}) {h.link}" in listed:
            continue
        lines.append(f"{hours:.0f}h ago — {h.title} ({outlet}) {h.link}")
    for f in filings[:2]:
        lines.append(f"SEC filing: {f.form} filed {f.filed:%Y-%m-%d} — "
                     f"{f.item_summary if f.is_event else f.description or f.form}.")
    payload = {"change_24h_pct": change, "headlines": [
        {"title": h.title, "feed": h.feed, "link": h.link, "published": h.published.isoformat()}
        for h in kept[:10]], "filings": [f.form for f in filings],
        "tone": toned[1] if toned is not None else None}
    sources = [Source(kind="venue", ref="RSS + Yahoo Finance + SEC EDGAR",
                      detail=f"{len(kept)} headline(s), {len(filings)} filing(s)")]
    try:
        routed = skill_job.result(timeout=8.0)
    except Exception:
        routed = None
    if routed is not None and routed.via == "skill":
        skill_lines = _skill_headlines(routed.payload, {h.title for h in kept})
        lines.extend(skill_lines)
        sources.append(routed.source())
        payload["skill_headlines"] = len(skill_lines)
    elif routed is not None:
        sources[0] = Source(kind="venue", ref=sources[0].ref,
                            detail=sources[0].detail + "; publishers' feeds read directly because "
                            "bitget-signal's news_feed " + routed.skill_said)
    return lines, sources, payload


def _own_feeds_are_the_mirror(ident: str, args: dict[str, Any]
                              ) -> tuple[Health, str, Any, str, bool]:
    """The news answer reads publishers' feeds itself; the route records only which of the Skill
    and those feeds the headlines stand on."""
    del ident, args
    return Health.OK, "read by the news answer", None, "publishers' RSS feeds", True


def _skill_headlines(payload: Any, already: set[str]) -> list[str]:
    """Headlines bitget-signal's news_feed returned that the feeds read here did not.

    The reply is a list of ``{"feed", "error", "items"}`` per feed (read from the live tool on
    2026-09-27, when every ``items`` list was empty). The fields of an item are not documented in
    ``bitget-signal/skills/news-briefing/SKILL.md`` and the server's source is not public, so an
    item is shown only when it carries a ``title``; NOT VERIFIED against a non-empty reply."""
    out: list[str] = []
    for feed in payload if isinstance(payload, list) else []:
        for item in (feed.get("items") or []) if isinstance(feed, dict) else []:
            title = str(item.get("title") or "").strip() if isinstance(item, dict) else ""
            if not title or title in already:
                continue
            already.add(title)
            link = str(item.get("link") or item.get("url") or "")
            out.append(f"bitget-signal news_feed ({feed.get('feed')}): {title}"
                       + (f" {link}" if link else ""))
            if len(out) >= 3:
                return out
    return out


# Every engine here is a traced step from import on (lui/trace.py, trace_module).
trace_module(globals())
