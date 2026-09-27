"""Checking a claim the question makes — a funding level, a move, liquidity, a session — against
the live figure."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from argus.lui.answer import Source
from argus.lui.question import (
    TRADED_SYMBOLS,
)
from argus.lui.research.anchor import (
    _premium_line,
)
from argus.lui.research.kinds import _t
from argus.lui.research.parse import (
    _DIRECTIONAL,
    _SHORT,
    parse_notional,
)
from argus.lui.research.session import (
    anchor_is_open,
)
from argus.lui.trace import trace_module

_FUNDING_CLAIM = re.compile(
    r"\bfunding\s+(?:rate\s+)?(?:looks?|is|seems?|feels?|being|still|now|so|pretty|very|too|"
    r"really|quite|\s)*\s*(?P<a>cheap|low|negative|free|expensive|high|rich|elevated|hot|"
    r"crowded|stretched|positive)\b"
    r"|\b(?P<b>cheap|low|negative|expensive|high|rich|elevated)\s+funding\b"
    # 资金费率贵吗 / 资金费率很低 / 资金费便宜: the same premise asked in Chinese.
    r"|\u8d44\u91d1\u8d39(?:\u7387)?(?:\u5f88|\u592a|\u6bd4\u8f83|\u633a|\u662f\u5426|"
    r"\u662f\u4e0d\u662f)?"
    r"(?P<c>\u8d35|\u4fbf\u5b9c|\u9ad8|\u4f4e)",
    re.IGNORECASE)


_CJK_FUNDING_WORD = {"\u8d35": "expensive", "\u4fbf\u5b9c": "cheap", "\u9ad8": "high",
                     "\u4f4e": "low"}


_PREMIUM_CLAIM = re.compile(
    r"\b(?:perp\w*|contract|token|rtoken)\b[^.?!]{0,40}?\b(?:at\s+an?\s+|"
    r"trad\w+\s+(?:at\s+)?an?\s+)?"
    r"(?P<c>premium|discount)\b|\bbasis\s+(?:looks?\s+|is\s+)?(?P<d>wide|rich|tight|cheap)\b"
    # "NVDA trades at a premium": the ticker is the subject. Only read for a stock, where the
    # premium has a stock to be measured against (see `_claim_check`).
    r"|\btrad\w*\s+(?:at\s+)?an?\s+(?P<e>premium|discount)\b(?!\s+to\s+(?:its\s+)?peers)",
    re.IGNORECASE)


_CHEAP_WORDS = frozenset({"cheap", "low", "negative", "free"})


_MOVE_UP = (r"up|higher|green|rall(?:y|ying|ied)|pump(?:ing|ed)?|surg(?:e|ing|ed)|"
             r"ris(?:e|ing)|rose|climb(?:ing|ed)?|moon(?:ing|ed)?|rip(?:ping|ped)?")


_MOVE_DOWN = (r"down|lower|red|drop(?:ping|ped)?|fall(?:ing)?|fell|dump(?:ing|ed)?|"
               r"crash(?:ing|ed)?|tank(?:ing|ed)?|plung(?:e|ing|ed)|sink(?:ing)?|sank|"
               r"slid(?:e|ing)?|bleed(?:ing)?")


_MOVE_CLAIM = re.compile(
    rf"\bwhy\s+(?:did|is|was|has|are)\s+(?:\w+\s+){{0,3}}?(?P<why>{_MOVE_UP}|{_MOVE_DOWN})\b"
    rf"|\b(?:is|was|has\s+been|are|been)\s+(?:so\s+|really\s+|still\s+)?(?P<state>{_MOVE_UP}|"
    rf"{_MOVE_DOWN})\b(?!\s+(?:to\b|in\s+\d))"
    r"|\b(?P<past>rallied|rose|jumped|surged|soared|climbed|spiked|popped|pumped|gained|mooned|"
    r"dropped|fell|slid|sank|plunged|tanked|crashed|dumped|slumped)\b"
    r"(?!\s+(?:to\b|in\s+\d|if\b))",
    re.IGNORECASE)
"""A claim about the move that already happened ("why did TSLA drop today", "NVDA is pumping").
Future tense is `_DIRECTIONAL`'s; "is going to drop" never reaches here because "going" is not a
move word. The idea of refusing to let a price move confirm its stated cause is MirrorLine's
(PinnacleCryptNG, a Season 2 desk with no licence file): rebuilt from its described behaviour, no
code taken."""


_STATED_SIZE = re.compile(
    r"\s*(?:by\s+|about\s+|around\s+|over\s+|nearly\s+|almost\s+|like\s+|some\s+)?[+-]?"
    r"(\d+(?:\.\d+)?)\s*(?:%|percent\b|pct\b)", re.I)


def _stated_move_size(text: str, after: int) -> float | None:
    """The size a move claim states right after its verb ("up 5%", "rallied by 8 percent")."""
    found = _STATED_SIZE.match(text, after)
    return float(found.group(1)) if found else None


_LIQUIDITY_CLAIM = re.compile(
    r"\b(?:is|looks?|seems?|are|it's|plenty|very|super|quite|pretty|really|too)\s+"
    r"(?:(?:very|super|quite|pretty|really|too|plenty)\s+)?(?P<word>liquid|illiquid|deep|thin|"
    r"shallow)\b(?!\s+(?:into|in\s+the\s+money))"
    r"|\b(?P<word2>deep|thin|shallow)\s+(?:order\s+)?book\b|\b(?:can|will)\s+(?:easily\s+)?fill\b"
    r"|\bis\s+[A-Za-z]{2,12}\s+(?:still\s+)?(?P<word3>liquid|illiquid|deep|thin)\b",
    re.IGNORECASE)
"""A claim about the order book — "NVDA is liquid enough for $50k", "the book is thin". MirrorLine
reads these and ARGUS did not (2026-09-25); the live book is one call away, so it is measured."""


LIQUIDITY_DEFAULT_USD = Decimal(50_000)


_HYPOTHETICAL = re.compile(r"\b(?:if|what\s+if|suppose|assuming)\b", re.IGNORECASE)
""""If NVDA dropped 10%" is a scenario, not a claim about the tape; the stress engine owns it."""


_PAST_UP = frozenset({"rallied", "rose", "jumped", "surged", "soared", "climbed", "spiked",
                      "popped", "pumped", "gained", "mooned"})


SESSION_CLAIM = re.compile(
    r"\b(?:us|u\.s\.|american|stock|equity|nyse|nasdaq|wall\s+street)\s+(?:stock\s+)?"
    r"(?:market|session|exchange)s?\s+(?:is|are)\s+(?:now\s+|still\s+|currently\s+)?"
    r"(?P<state>open|closed|shut)\b"
    r"|\b(?:trading|traded|trades|is|it'?s|it\s+is)\s+(?:now\s+|still\s+)?"
    r"(?P<after>after[- ]hours|pre[- ]?market|during\s+(?:us\s+)?market\s+hours)\b",
    re.IGNORECASE)
"""A claim about whether the anchor market is trading — "the US market is closed right now",
"NVDA is trading after hours". MirrorLine checks this one and ARGUS did not (head-to-head,
2026-09-25): the clock is in `truth/clocks`, so the claim is checkable and was simply unread."""


_CAUSAL = re.compile(r"\b(?:because(?:\s+of)?|due\s+to|driven\s+by|on\s+the\s+back\s+of|"
                     r"thanks\s+to|caused\s+by|after\s+the)\b", re.IGNORECASE)


def _claim_check(raw_text: str, symbol: str) -> tuple[list[str], list[Source]] | None:
    """Premises stated in the question, each measured and given a verdict.

    Funding "cheap" or "expensive" is judged against the contract's own last settlements
    (`research/carry.fetch_funding`, Bitget `/api/v3/market/history-fund-rate`), because a rate is
    only cheap relative to what this contract usually pays: +0.005% is ordinary for BTC and dear
    for a stock perpetual that sits at zero most of the time. A premium claim is checked against
    the stock's own quote through `_premium_line`.
    """
    # A claim about the session needs no instrument; every other premise is about one.
    funding = _FUNDING_CLAIM.search(raw_text) if symbol else None
    premium = _PREMIUM_CLAIM.search(raw_text) if symbol else None
    if premium is not None and premium.group("e"):
        from argus.market import universe

        if symbol not in TRADED_SYMBOLS and not universe.is_equity(symbol):
            premium = None  # "BTC trades at a premium" has no stock to measure it against
    move = None if not symbol or (_DIRECTIONAL.search(raw_text) and not re.search(
        r"\bwhy\b", raw_text, re.I)) or _HYPOTHETICAL.search(raw_text) \
        else _MOVE_CLAIM.search(raw_text)
    session = SESSION_CLAIM.search(raw_text)
    liquidity = _LIQUIDITY_CLAIM.search(raw_text) if symbol else None
    if not funding and not premium and not move and not session and not liquidity:
        return None
    lines: list[str] = []
    sources: list[Source] = []
    ticker = None
    if funding or premium or move:
        try:
            from argus.market.bitget import fetch_tickers

            ticker = fetch_tickers().get(symbol)
        except Exception:
            ticker = None
    if liquidity:
        measured = _liquidity_claim_line(symbol, liquidity, raw_text)
        if measured:
            lines.append(measured[0])
            sources.append(measured[1])
    if session:
        lines.append(session_claim_line(session))
        sources.append(Source(kind="computation", ref="argus.truth.clocks",
                              detail="NYSE regular session, holidays included"))
    if move and ticker is not None:
        word = (move.group("why") or move.group("state") or move.group("past") or "").lower()
        claimed_up = (word in _PAST_UP if move.group("past")
                      else re.fullmatch(_MOVE_UP, word, re.I) is not None)
        change = float(ticker.change_24h) * 100
        if abs(change) < 0.3 and (change > 0) == claimed_up:
            verdict = f"barely — {_t(symbol)} is {change:+.2f}% over 24 hours, essentially flat"
        elif abs(change) < 0.3:
            verdict = (f"does not hold — {_t(symbol)} is {change:+.2f}% over 24 hours, essentially "
                       f"flat and if anything {'up' if change > 0 else 'down'}")
        elif (change > 0) == claimed_up:
            verdict = f"holds — {_t(symbol)} is {change:+.2f}% over 24 hours"
            stated = _stated_move_size(raw_text, move.end())
            if stated is not None and abs(change) < stated / 2:
                # "NVDA is up 5% today" with NVDA +1.18%: the direction holds and the size does
                # not, and saying "holds" alone let a four-fold overstatement stand (found in a
                # sample of answers, 2026-09-25).
                verdict = (f"holds in direction, not in size — {_t(symbol)} is {change:+.2f}% over "
                           f"24 hours, not the {stated:g}% stated")
            elif stated is not None and abs(change) > stated * 2:
                verdict = (f"holds, and understates it — {_t(symbol)} is {change:+.2f}% over 24 "
                           f"hours, more than the {stated:g}% stated")
        else:
            verdict = (f"does not hold — {_t(symbol)} is {change:+.2f}% over 24 hours, "
                       f"{'up' if change > 0 else 'down'}, not {'up' if claimed_up else 'down'}")
        cause = ("; the reason you give is not something a price can confirm — the lines below "
                 "say whether a filing or headline backs it" if _CAUSAL.search(raw_text) else "")
        lines.append(f"Your premise that {_t(symbol)} is {'up' if claimed_up else 'down'}: "
                     f"{verdict}{cause}.")
        sources.append(Source(kind="venue", ref="bitget /api/v2/mix/market/tickers",
                              detail=f"{symbol} 24h change, live"))
    if funding:
        word = (funding.group("a") or funding.group("b")
                or _CJK_FUNDING_WORD.get(funding.group("c") or "", "")).lower()
        line = _funding_claim_line(symbol, word, ticker)
        if line:
            lines.append(line[0])
            sources.append(line[1])
    if premium and ticker is not None:
        word = (premium.group("c") or premium.group("d") or premium.group("e") or "").lower()
        measured = _premium_line(symbol, ticker.last, _anchor_open_now())
        if measured is None:
            lines.append(f"Your premise that the contract trades at a {word}: not checkable — the "
                         f"stock's own quote did not arrive, so there is no gap to measure.")
        else:
            text = measured[0]
            is_premium = " premium " in text
            claimed_premium = word in ("premium", "wide", "rich")
            size = re.search(r"(\d+(?:\.\d+)?)bps", text)
            small = size is not None and float(size.group(1)) < 5
            verdict = ("unclear — the gap is under 5bps, inside the spread" if small else
                       "holds" if is_premium == claimed_premium else "does not hold")
            lines.append(f"Your premise that the contract trades at a {word}: {verdict}. "
                         + text.replace("Versus the stock: ", ""))
            sources.append(measured[1])
    return (lines, sources) if lines else None


def _liquidity_claim_line(symbol: str, found: re.Match[str],
                          raw_text: str) -> tuple[str, Source] | None:
    """The claim against the live 50-level book: what the stated size (or $50,000) costs to cross.

    "Liquid" holds when the whole size fills inside the visible book for less than one 6bps taker
    fee of slippage, does not hold when the book cannot absorb it or it costs over 20bps, and is
    unclear in between — said with the number, never as a bare adjective."""
    from argus.market.bitget import BitgetError
    from argus.market.depth import DepthError, fetch_orderbook

    word = (found.group("word") or found.group("word2") or found.group("word3")
            or "fill").lower()
    claims_liquid = word in ("liquid", "deep", "fill")
    size = parse_notional(raw_text) or LIQUIDITY_DEFAULT_USD
    stated = parse_notional(raw_text) is not None
    direction = "SELL" if _SHORT.search(raw_text) else "BUY"
    try:
        book = fetch_orderbook(symbol)
        sweep = book.sweep(size, direction=direction)
    except (BitgetError, DepthError):
        return (f"Your premise that {_t(symbol)}'s book is {word}: not checkable — the order book "
                f"did not arrive.",
                Source(kind="venue", ref="bitget /api/v3/market/orderbook", detail="unavailable"))
    slip = float(sweep.slippage_bps)
    liquid = sweep.complete and slip < 6.0
    illiquid = (not sweep.complete) or slip > 20.0
    what = "liquid" if claims_liquid else "thin"
    verdict = ("holds" if (liquid if claims_liquid else illiquid) else
               "does not hold" if (illiquid if claims_liquid else liquid) else
               "unclear — it fills, but for more than one taker fee of slippage")
    size_words = f"${size:,.0f}" + ("" if stated else " (no size was stated)")
    fill = (f"a {size_words} {'buy' if direction == 'BUY' else 'sell'} fills across "
            f"{sweep.levels_consumed} level{'s' if sweep.levels_consumed != 1 else ''} for "
            f"{slip:.1f}bps of slippage"
            if sweep.complete else
            f"the visible 50 levels cannot absorb a {size_words} "
            f"{'buy' if direction == 'BUY' else 'sell'}")
    lead = ((f"Is {_t(symbol)} liquid enough? " if claims_liquid
             else f"Is {_t(symbol)}'s book thin? ")
            + {"holds": "Yes", "does not hold": "No"}.get(verdict, "Borderline")
            if raw_text.rstrip().endswith("?") and found.group("word3")
            else f"Your premise that {_t(symbol)} is {what}: {verdict}")
    return (f"{lead} — {fill}, against a 6bps taker fee each way.",
            Source(kind="venue", ref="bitget /api/v3/market/orderbook",
                   detail=f"{symbol} 50 levels, {book.fetched_at:%H:%M:%S} UTC"))


def session_claim_line(found: re.Match[str]) -> str:
    """The verdict on a claim about whether the US market is trading, read from the clock."""
    now = datetime.now(UTC)
    is_open = _anchor_open_now()
    if found.group("state"):
        claimed_open = found.group("state").lower() == "open"
        what = f"the US market is {'open' if claimed_open else 'closed'}"
    else:
        phrase = found.group("after").lower()
        claimed_open = phrase.startswith("during")
        what = ("it is US market hours" if claimed_open
                else f"it is {phrase.replace('-', ' ')} in the US")
    verdict = "holds" if claimed_open == is_open else "does not hold"
    return (f"Your premise that {what}: {verdict} — at {now:%H:%M} UTC the NYSE regular session "
            f"is {'open' if is_open else 'closed'}"
            + ("" if is_open else ", so the stock's last price is its last close and the "
               "perpetual is the only thing trading") + ".")


def _anchor_open_now() -> bool:
    try:
        return bool(anchor_is_open()(datetime.now(UTC)))
    except Exception:
        return False


def _funding_claim_line(symbol: str, word: str, ticker: Any) -> tuple[str, Source] | None:
    from argus.research import carry

    if ticker is None:
        return (f"Your premise that funding looks {word}: not checkable — {_t(symbol)}'s live "
                f"funding rate did not arrive.",
                Source(kind="venue", ref="bitget /api/v2/mix/market/tickers",
                       detail=f"{symbol} funding unavailable"))
    now_bps = float(ticker.funding_rate) * 10_000
    try:
        history = [s.rate_bps for s in carry.fetch_funding(symbol, pages=1)]
    except Exception:
        history = []
    from argus.market import universe

    listed = universe.contracts().get(symbol) or universe.Contract(symbol, False)
    hours = listed.funding_hours or 8
    yearly = now_bps / 100 * (24 / hours) * 365
    cost = (f"a long pays about {yearly:.1f}% a year at this rate" if yearly > 0 else
            f"a long is paid about {-yearly:.1f}% a year at this rate" if yearly < 0 else
            "holding costs nothing in funding at this rate")
    claims_cheap = word in _CHEAP_WORDS
    if len(history) < 20:
        verdict = "unclear — the contract's settlement history did not arrive to compare against"
        context = ""
    else:
        below = sum(1 for r in history if r < now_bps)
        equal = sum(1 for r in history if r == now_bps)
        rank = (below + equal / 2) / len(history)
        ordered = sorted(history)
        median = ordered[len(ordered) // 2]
        context = (f" — higher than {rank:.0%} of its last {len(history)} settlements "
                   f"(median {median / 100:+.4f}%)")
        # Most stock perpetuals settle at exactly zero most of the time, so a rank computed with
        # ties is fragile: two contracts both at zero, one with a few more negative settlements,
        # got "holds" and "unclear" on the same run (2026-09-25). A rate at or below zero is
        # cheap for a long whatever its rank; beside that, the comparison is with the median.
        if claims_cheap and now_bps <= 0:
            verdict = ("holds — a long pays nothing, and less than this contract usually charges"
                       if now_bps < median else
                       "holds in absolute terms — a long pays nothing — which is this contract's "
                       "usual rate")
        elif not claims_cheap and now_bps <= median:
            verdict = "does not hold — the rate is at or below this contract's usual"
        elif 0.35 <= rank <= 0.65:
            verdict = "unclear — the rate is ordinary for this contract"
        elif (rank < 0.35) == claims_cheap:
            verdict = "holds"
        else:
            verdict = "does not hold"
        if verdict == "does not hold" and abs(yearly) < 3:
            verdict += (f" relative to its own history, though at {abs(yearly):.1f}% a year the "
                        f"cost is small either way")
    return (
        f"Your premise that funding looks {word}: {verdict}. {_t(symbol)} funding is "
        f"{now_bps / 100:+.4f}% per {hours}h{context}; {cost}.",
        Source(kind="venue", ref="bitget /api/v3/market/history-fund-rate",
               detail=f"{symbol}; last {len(history)} settlements and the live rate"),
    )


# Every engine here is a traced step from import on (lui/trace.py, trace_module).
trace_module(globals())
