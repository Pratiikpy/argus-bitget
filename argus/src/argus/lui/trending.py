"""What the crowd is looking at right now, and which of it a Bitget trader can actually trade.

**Why this exists.** "What's trending?" and "what is the crowd talking about?" had no answer: the
console routed them to a research engine that needs a named instrument and refused. bitget-signal
carries two tools for exactly this — ``crypto_market.trending`` (the market-intel Skill, CoinGecko's
search-trending list) and ``social_trending`` (hot lists from Chinese social platforms, of which
Xueqiu is the investing one) — and until 2026-09-27 the console asked neither (audit findings 108
and 109).

**What the answer is.** The coins CoinGecko's users are searching for most, each checked against
Bitget's own USDT perpetual board: whether it is listed, its 24-hour move and its funding rate.
Trending is attention, not a forecast, and the answer says so; the useful fact is which of the
names drawing the crowd a trader can act on here, and whether the crowd has already moved the
price. The Xueqiu hot list is asked too and shown when it answers.

**Order and labels** are `lui/skillroute.py`'s: the Skill first, the source it names second
(CoinGecko, via `market/skill_mirror.trending`), and the receipt says which one spoke. Xueqiu has
no public mirror, so when the Skill does not answer, the answer says that and shows nothing in its
place.

**NOT VERIFIED:** the Skill's own reply shapes. On 2026-09-27 ``crypto_market.trending`` timed out
and ``social_trending`` returned ``{"provider": "all_failed", "items": []}``. The Skill's reply is
read as CoinGecko's documented ``/search/trending`` shape (``coins[].item``), which its tool
description names as its source, or as the mirror's normalised ``trending`` list; a social item
is shown only when it carries a ``title``.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from typing import Any

from argus.lui.answer import Source

_ASKS = re.compile(
    r"\b(?:what(?:'?s|\s+is|\s+are)\s+(?:the\s+)?(?:trending|hot|hottest)|trending\s+"
    r"(?:coins?|tokens?|names?|now|today|right\s+now)|what\s+(?:is|are)\s+(?:the\s+)?(?:crowd|"
    r"people|everyone|retail)\s+(?:talking|buzzing|excited)\s+about|what(?:'?s|\s+is)\s+"
    r"everyone\s+buying|hot\s+coins?)\b|热门|热搜|大家在(?:讨论|聊)", re.I)

TOP = 10

PRICE_TOLERANCE = 0.15
"""CoinGecko's price and Bitget's last may differ by this much and still be one coin; a ticker
collision misses by far more (the same check `lui/exposures.py` makes for stock tickers)."""


_STOCKS = re.compile(r"\b(?:stocks?|equit(?:y|ies)|shares|tech|nasdaq|s&p|dow|rtokens?|"
                     r"market\s+for\s+stocks)\b|股票|美股|A股", re.I)
"""This answer is CoinGecko's crypto list; a question about what is hot in stocks is not it."""


def asks_for_trending(text: str) -> bool:
    return bool(_ASKS.search(text)) and not _STOCKS.search(text)


def coins_from(payload: Any) -> list[dict[str, Any]]:
    """Trending coins from either reply shape: the mirror's ``{"trending": [...]}`` or
    CoinGecko's own ``{"coins": [{"item": {...}}]}``. Unknown shapes give nothing."""
    if not isinstance(payload, Mapping):
        return []
    rows = payload.get("trending")
    if not isinstance(rows, list):
        rows = [c.get("item") for c in payload.get("coins") or [] if isinstance(c, Mapping)]
    out = []
    for row in rows:
        if isinstance(row, Mapping) and row.get("symbol"):
            data = row.get("data")
            price = row.get("price_usd", data.get("price") if isinstance(data, Mapping) else None)
            out.append({"symbol": str(row["symbol"]).upper(), "name": str(row.get("name") or ""),
                        "rank": row.get("market_cap_rank"),
                        "price": float(price) if isinstance(price, (int, float)) else None})
    return out[:TOP]


def social_titles(payload: Any, limit: int = 5) -> list[str]:
    items = payload.get("items") if isinstance(payload, Mapping) else None
    return [str(i["title"]).strip() for i in items or []
            if isinstance(i, Mapping) and str(i.get("title") or "").strip()][:limit]


Route = Callable[..., Any]


def trending(text: str, *, route: Route | None = None,
             tickers: Callable[[], Mapping[str, Any]] | None = None,
             mirror: Callable[[Mapping[str, Any]], dict[str, Any]] | None = None,
             ) -> tuple[list[str], list[Source], dict[str, Any]]:
    """The answer. ``route``, ``tickers`` and ``mirror`` default to the live Skill route,
    Bitget's ticker board and CoinGecko; tests pass fakes."""
    from argus.lui.skillroute import route as live_route
    from argus.market.bitget import fetch_tickers
    from argus.market.skill_mirror import trending as coingecko
    from argus.truth.coverage import ContextPool

    ask: Route = route or live_route
    board_of: Callable[[], Mapping[str, Any]] = tickers or fetch_tickers
    direct: Callable[[Mapping[str, Any]], dict[str, Any]] = mirror or coingecko

    with ContextPool(max_workers=3) as pool:
        coin_job = pool.submit(ask, "crypto_market", "trending", None, wait=6.0)
        social_job = pool.submit(ask, "social_trending", "trending",
                                 {"platform": "xueqiu", "limit": 10}, wait=6.0)
        board_job = pool.submit(board_of)
        routed = coin_job.result()
        social = social_job.result()
        try:
            board = board_job.result()
        except Exception:
            board = {}

    coins = coins_from(routed.payload) if routed.answered else []
    sources: list[Source] = []
    if routed.via == "skill" and not coins:
        # the Skill answered in a shape this reader does not know: CoinGecko directly, said so
        try:
            coins = coins_from(direct({}))
            sources.append(Source(kind="evidence", ref="CoinGecko /search/trending",
                                  detail="read directly: bitget-signal crypto_market answered "
                                         "in a shape this console does not read"))
        except Exception:
            coins = []
    elif routed.answered:
        sources.append(routed.source())

    lines: list[str] = []
    rows: list[dict[str, Any]] = []
    collided: list[str] = []
    for coin in coins:
        ticker = board.get(f"{coin['symbol']}USDT")
        if (ticker is not None and coin.get("price") and float(ticker.last) > 0
                and abs(float(ticker.last) / coin["price"] - 1) > PRICE_TOLERANCE):
            # the same letters, a different coin: Bitget's contract is not the one trending
            collided.append(f"{coin['symbol']} (CoinGecko ${coin['price']:,.6g}, Bitget "
                            f"{float(ticker.last):,.6g})")
            ticker = None
        row = {**coin, "listed": ticker is not None}
        if ticker is not None:
            row.update(change_24h=float(ticker.change_24h),
                       funding_rate=float(ticker.funding_rate))
        rows.append(row)
    listed = [r for r in rows if r["listed"]]
    if not coins:
        lines.append("Missing: the trending list could not be read just now — bitget-signal's "
                     f"crypto_market {routed.skill_said}, and CoinGecko, the source it names, "
                     "did not answer either. Ask again in a minute.")
    else:
        if listed:
            mover = max(listed, key=lambda r: abs(r["change_24h"]))
            lead = (f"Bottom line: {len(listed)} of the {len(rows)} coins most searched on "
                    f"CoinGecko trade as USDT perpetuals on Bitget; the biggest 24-hour move "
                    f"among them is {mover['symbol']} at {mover['change_24h']:+.1%}. Trending is "
                    f"attention, not a forecast; each 24-hour move below shows how far the price "
                    f"has already gone.")
        else:
            lead = (f"Bottom line: none of the {len(rows)} coins most searched on CoinGecko has "
                    f"a USDT perpetual on Bitget, so the crowd's attention is elsewhere than "
                    f"this board.")
        lines.append(lead)
        for r in rows:
            where = (f"on Bitget {r['change_24h']:+.1%} in 24h, funding "
                     f"{r['funding_rate'] * 1e4:+.1f}bp" if r["listed"]
                     else "no Bitget USDT perpetual")
            rank = f", market-cap rank {r['rank']}" if r.get("rank") else ""
            lines.append(f"{r['symbol']} ({r['name']}{rank}): {where}.")
    if collided:
        lines.append("Not matched: " + "; ".join(collided) + " — Bitget lists a contract with the "
                     "same ticker at a different price, so it is not the coin that is trending.")
    titles = social_titles(social.payload) if social.answered else []
    if titles:
        lines.append("Xueqiu's hot list (bitget-signal social_trending): " + "; ".join(titles)
                     + ".")
        sources.append(social.source())
    else:
        lines.append(f"Xueqiu's hot list: bitget-signal's social_trending {social.skill_said}; "
                     f"it has no public mirror, so nothing stands in for it.")
    if board:
        sources.append(Source(kind="venue", ref="bitget /api/v2/mix/market/tickers",
                              detail="USDT perpetual board, 24h change and funding"))
    else:
        lines.append("Missing: Bitget's ticker board did not answer, so listing and moves are "
                     "not shown.")
    lines.append("Data: " + ", ".join(s.ref for s in sources) + ". Analysis, not advice.")
    return lines, sources, {"trending": rows, "listed": len(listed), "collided": collided,
                            "coins_via": routed.via, "social_via": social.via}


__all__ = ["TOP", "asks_for_trending", "coins_from", "social_titles", "trending"]
