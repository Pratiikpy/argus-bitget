"""Honest answers to questions no console can answer as asked — and the true reason, said plainly.

**What was found.** INFEASIBILITY-BENCH (`eval/infeasibilitybench.py`, artefact
``data/infeasibility_bench.json``, run ``20260925T160539Z``) put 73 such questions to the offline
console. It declined 23 with the true reason, 21 with a reason that was not the true one ("I did
not recognise that question" to "what will NVDA close at next Friday"), and answered 29 *with
figures*: Coinbase's price "in 2015" with today's COIN quote, bitcoin's 25-year drawdown with a
30-day risk profile, "what are Citadel's exact positions in NVDA today" with the desk's own "No
open positions", "what did the desk decide on 1 March 2026" with decision 684. Every non-right row
of that artefact was read before this module was written; the phrasings below are theirs.

**What this module does.** One offline detector per cause, each reading only the question's words,
the turn history, the saved book, the local ledger file and the frozen or already-cached contract
registry — never the network. Only two answer builders read data, and only after detection: a
price on a past date (:func:`_past_price_lines`) and an N-year statistic longer than the
instrument's life (:func:`_horizon_lines`). Both read with a timeout and, when the read fails, fall
back to the plain true reason rather than a number from nothing.

* ``private_account`` — the trader's own Bitget account. The console holds no key to anyone's
  account; it says so and offers what it can do (state the book, paste trades for a review). It
  does not fire on a question that states its own holdings or asks a hypothetical.
* ``others_positions`` — a named institution's or other users' live positions. Not public; 13F
  filings (quarterly, delayed) and Bitget's aggregate long/short split are, and are offered.
* ``order_instruction`` — handled by :func:`order_prefix`, which reuses the console's own order
  classifier (`lui/question.is_order_instruction`, `lui/research._is_an_order`, the Chinese
  ORDER patterns) and adds the three imperatives the bench found it missed. The console's analysis
  of such an order is kept; the prefix says nothing was sent above it.
* ``future_price`` — extends `lui/research.price_forecast_asked` with the six phrasings it missed
  (listed at :data:`_FUTURE_EXTRA`), keeping its `_PAST_PREDICTION` exclusion.
* ``before_data`` — the desk's record before its first decision (read from the ledger), and a
  price on a past date, answered from Yahoo daily (stocks) or Bitget daily candles (perps), with
  the day the history starts named when the date is before it.
* ``horizon_exceeds_data`` — the statistic over the longest span that exists, with that span
  stated and the N-year figure said not to exist.
* ``unlisted`` — a company whose tickers match no Bitget contract, checked against the registry
  (live when the answer is built, frozen or cached during detection), with the count checked.
* ``ambiguous`` — no resolvable name and nothing earlier to refer to: asks which one.

**Why a separate module.** `lui/research.py` and `lui/server.py` belong to other builders; this
module is called from `lui/server._answer` (see the report in ``data/honesty_eval.json``).
"""

from __future__ import annotations

import itertools
import json
import math
import re
import statistics
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, TypeVar

ROOT = Path(__file__).resolve().parents[3]
LEDGER_PATH = ROOT / "data" / "paper_ledger.jsonl"

READ_TIMEOUT_S = 25.0
"""The most a data read behind an answer may take before the plain reason is given instead."""

PRIVATE_ACCOUNT = "private_account"
OTHERS_POSITIONS = "others_positions"
ORDER_INSTRUCTION = "order_instruction"
FUTURE_PRICE = "future_price"
BEFORE_DATA = "before_data"
PAST_PRICE = "past_price"
"""A price on a past date the data does cover: answered, not declined."""
HORIZON = "horizon_exceeds_data"
LONG_HORIZON = "long_horizon"
"""An N-year statistic the history is long enough for: answered as asked."""
UNLISTED = "unlisted"
AMBIGUOUS = "ambiguous"

_T = TypeVar("_T")
_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="honesty")


def _with_timeout(read: Callable[[], _T], timeout: float = READ_TIMEOUT_S) -> _T | None:
    """``read()`` or None if it raised or ran past ``timeout``. A late thread is abandoned."""
    future = _POOL.submit(read)
    try:
        return future.result(timeout=timeout)
    except FutureTimeout:
        return None
    except Exception:  # any failed read degrades to the plain reason
        return None


# --- the registry, offline --------------------------------------------------------------------


def _registry() -> Mapping[str, Any]:
    """Bitget's USDT-futures contracts without touching the network: the registry
    `market/universe.contracts` already cached in this process, else its frozen snapshot."""
    from argus.market import universe

    cached = universe._CACHE
    if cached is not None:
        return cached[1]
    try:
        return universe._from_snapshot()[0]
    except (OSError, ValueError, KeyError):
        return {}


def _contract_for(token: str, registry: Mapping[str, Any]) -> str | None:
    """A ticker or name to a listed contract symbol, read from ``registry`` (no network)."""
    from argus.market import universe

    upper = token.strip().upper().removesuffix("'S").removesuffix("’S")
    if not upper:
        return None
    alias = universe.ALIASES.get(upper)
    if alias and alias in registry:
        return alias
    for candidate in (upper if upper.endswith("USDT") else "", f"{upper}STOCKUSDT",
                      f"{upper}USDT"):
        if candidate and candidate in registry:
            return candidate
    return None


_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9&]*(?:['’]s)?")
_WORD_TICKER = frozenset({
    "A", "I", "AM", "AN", "AS", "AT", "BE", "BY", "DO", "GO", "IF", "IN", "IS", "IT", "ME", "MY",
    "NO", "OF", "ON", "OR", "SO", "TO", "UP", "US", "WE", "ALL", "AND", "ANY", "ARE", "BUT",
    "CAN", "FOR", "HAS", "HOW", "NOW", "OUT", "THE", "TOP", "WHY", "YOU", "BIG", "LOW", "HIGH",
    "ONE", "TWO", "NEW", "OLD", "DAY", "AI", "CEO", "USD", "USDT", "PNL", "ETF", "IPO", "EPS",
    "RSI", "MACD", "ATH", "YTD", "YOY", "EOD", "EOY", "GDP", "CPI", "FED", "FOMC", "OK", "VS",
    "PM", "UTC", "ET", "EST", "NYSE", "NASDAQ", "BITGET", "SEC", "LONG", "SHORT", "BUY",
    "SELL", "HOLD", "WILL", "WHAT", "WHEN", "WHERE", "WHO", "DID", "WAS", "NOT", "ITS", "PERP",
    "GIVE", "SHOW", "TELL", "DESK", "LAST", "PAST", "NEXT", "OVER", "YEAR", "YEARS", "WEEK",
    "MONTH", "BEST", "WORST", "RISK", "PRICE", "STOCK", "TOKEN", "MAX", "MIN",
})
"""Capitalised words that are words before they are tickers, when written in capitals."""


def names_in(text: str, registry: Mapping[str, Any] | None = None) -> tuple[str, ...]:
    """Listed contracts ``text`` names, offline. Lower-case words resolve only as a known name
    ("bitcoin", "tesla", "gold"); a ticker must be written in capitals ("COIN", "SOL")."""
    from argus.lui.question import resolve_symbol
    from argus.market import universe

    listed = registry if registry is not None else _registry()
    found: list[str] = []

    def add(symbol: str | None) -> None:
        if symbol and symbol not in found:
            found.append(symbol)

    for name, symbol in universe.CJK_ALIASES.items():
        if name in text:
            add(symbol)
    if re.search(r"\bS\s*&\s*P(?:\s*500)?\b", text, re.I):
        add("SP500USDT")
    for raw in _TOKEN.findall(text):
        word = re.sub(r"['’]s$", "", raw)
        upper = word.upper()
        if upper in universe.ALIASES and not (word.isupper() and len(word) <= 3
                                              and upper in _WORD_TICKER):
            add(_contract_for(upper, listed))
            continue
        if word.isupper() and len(word) >= 2 and upper not in _WORD_TICKER:
            add(resolve_symbol(word) or _contract_for(upper, listed))
            continue
        if not word.isupper():
            add(resolve_symbol(word) if upper in _NAME_WORDS else None)
    return tuple(found)


_NAME_WORDS = frozenset({"TESLA", "APPLE", "MICROSOFT", "GOOGLE", "NVIDIA", "COINBASE", "META",
                         "AMAZON", "NETFLIX", "PALANTIR", "MICROSTRATEGY", "ALPHABET", "CIRCLE",
                         "ROBINHOOD"})
"""Lower- or title-case company names `lui/question.resolve_symbol` knows."""


# --- orders: a prefix, never a replacement ---------------------------------------------------

_ORDER_EXTRA = re.compile(
    # "set a limit order to buy gold at 4000" (ord-05): `set` is not among the console's order
    # verbs, so the analysis came back with no word that nothing was placed.
    r"^\s*(?:(?:ok(?:ay)?|please|pls|now|just|then)[\s,]+)*"
    r"(?:set|put\s+in|enter|place|submit|create)\s+(?:me\s+)?(?:a|an|my|the)?\s*"
    r"(?:(?:buy|sell|limit|market|stop|stop[\s-]loss|take[\s-]profit|trailing|bracket|oco|"
    r"conditional|trigger)\s+)*(?:orders?|stops?|stop[\s-]loss(?:es)?|take[\s-]profits?)\b"
    # "move my stop on MSTR up to breakeven" (ord-09) was read as a question about prices.
    r"|^\s*(?:(?:ok(?:ay)?|please|pls|now|just|then)[\s,]+)*"
    r"(?:move|raise|lower|tighten|trail|widen|drop|pull|shift|adjust|update|bump)\s+"
    r"(?:my|the|our)\s+(?:\w+\s+){0,2}(?:stops?|stop[\s-]loss|take[\s-]profit|tp|sl|limit|"
    r"orders?|targets?)\b",
    re.I)
"""Imperatives the console's order classifier did not know. ``go long BTC with 5x leverage right
now`` (ord-07) it already classified as an order; the gap there was only the missing statement."""


def order_prefix(text: str) -> str | None:
    """The line to put above whatever the console says to an instruction to trade, or None.

    Detection reuses the console's own classifier — `lui/question.is_order_instruction` (the
    English, request, Chinese-helper and ten-language forms), the Chinese ORDER patterns in
    `lui/question._CHINESE_COMPILED`, and `lui/research._is_an_order` — and adds only
    :data:`_ORDER_EXTRA`. A question ("how should I split a $50k order in NVDA") is not an order.
    """
    from argus.lui.normalise import fold
    from argus.lui.question import _CHINESE_COMPILED, Intent, is_order_instruction
    from argus.lui.research import _is_an_order

    raw = fold(text).strip()
    if not raw or re.search(r"\?|？|\bhow\b|\bshould\b|\bwhat\b|\bwhich\b|\bshow\s+me\b|"
                            r"\b(?:and|then)\s+(?:show|tell|give)\b|\bresulting\b|"
                            r"怎么|怎样|如何|应该|是否|要不要|吗|呢|影响|什么|哪", raw, re.I):
        return None  # a question about an order, or a request for its analysis (`_is_an_order`)
    if re.match(r"^\s*(?:please\s+|pls\s+)?hedge\b", raw, re.I):
        # a hedge request is answered as a hedge analysis (`research._IMPERATIVE_HEDGE`)
        return None
    chinese = any(i is Intent.ORDER and p.search(raw) for p, i in _CHINESE_COMPILED)
    if not (is_order_instruction(raw) or _is_an_order(raw) or chinese
            or _ORDER_EXTRA.search(raw)):
        return None
    if chinese or re.search(r"[一-鿿]", raw):
        return ("没有发送任何订单：本控制台不下单、不改单、不撤单。以下是这笔订单的成本与风险，"
                "下单由你自己在 Bitget 完成。")
    return ("Nothing was sent: this console never places, changes or cancels orders. What follows "
            "is what that order would cost or risk — placing it is yours to do on Bitget.")


# --- future prices -------------------------------------------------------------------------------

_FUTURE_EXTRA = re.compile(
    # fut-01 "what will NVDA close at next Friday", fut-08 "what will AAPL trade at right after
    # its next earnings": `research._PRICE_FORECAST` knows "be worth/at/trading", not these verbs.
    r"\bwhat\s+will\s+(?:[\w&.$'’-]+\s+){1,4}(?:close|open|trade|settle|end|finish|print|"
    r"sit)\s+(?:at|on|the)\b"
    # fut-03 "what price will TSLA reach in 2030".
    r"|\bwhat\s+(?:price|level|value)\s+will\b"
    # fut-05 "where will the S&P 500 be at year end": the old pattern allowed one word of name.
    r"|\bwhere\s+will\s+(?:[\w&.$'’-]+\s+){1,4}(?:be|go|trade|close|end|land|finish|top|"
    r"bottom)\b"
    # fut-06 "will MSTR hit $1000 by December" — a yes/no on a price level at a future time.
    r"|^\s*will\s+(?:[\w&.$'’-]+\s+){1,3}(?:hit|reach|touch|break|cross|get\s+to|top|"
    r"surpass|exceed|go\s+(?:above|below|past|over|under)|fall\s+(?:to|below|under)|"
    r"drop\s+(?:to|below|under))\s+\$?\d"
    # fut-10 "tell me the exact bottom for SOL this cycle".
    r"|\b(?:exact|precise|precisely)\s+(?:\w+\s+){0,2}(?:bottom|top|peak|low|high|close|"
    r"target)\b|\b(?:the\s+)?(?:bottom|top|peak)\s+(?:for|of|in)\s+[\w&.$-]+\s+this\s+cycle\b"
    r"|\bwhen\s+will\s+(?:[\w&.$'’-]+\s+){1,3}(?:bottom|top|peak)\b"
    # fut-09 "比特币下周五的收盘价是多少": 收盘价 is not 价格, and 下周五 is not in the old list.
    r"|(?:明天|明日|后天|下周|下週|下个?月|下個月|明年|年底|年末|将来|將來|未来|未來)"
    r"[^?？]{0,12}(?:收盘价|收盤價|开盘价|開盤價|价格|價格|价位|價位|点位|點位|会涨到|会跌到)",
    re.I)
"""Phrasings of a future price that `lui/research.price_forecast_asked` missed on the bench."""


_NOT_A_FORECAST = re.compile(
    r"\b(?:past|previous|earlier|its|your|their|the\s+desk'?s?|desk'?s)\s+(?:\w+\s+)?"
    r"predictions?\b|\bcalibrat\w*|\bprediction\s+(?:accuracy|record|track)", re.I)
"""About predictions already made: "the calibration accuracy of this desk's past predictions"
matched `research._PRICE_FORECAST` on its bare "predict" (held-out corpus)."""


def future_price_asked(text: str) -> bool:
    """`research.price_forecast_asked`, widened by :data:`_FUTURE_EXTRA`, with the same
    `_PAST_PREDICTION` exclusion (whether a pattern predicted anything; analysts' targets) and
    :data:`_NOT_A_FORECAST`."""
    from argus.lui.research import _PAST_PREDICTION, price_forecast_asked

    if _NOT_A_FORECAST.search(text):
        return False
    if price_forecast_asked(text):
        return True
    return bool(_FUTURE_EXTRA.search(text)) and not _PAST_PREDICTION.search(text)


def _future_lines(text: str, names: Sequence[str]) -> list[str]:
    name = _short(names[0]) if names else "it"
    example = _short(names[0]) if names else "NVDA"
    return [
        f"Actionable: no one can know that — a future price cannot be known, and this console "
        f"does not forecast prices, so there is no figure for {name} at that date.",
        f"What it can give is the historical range: ask \"what are the odds {example} is up "
        f"this week\" for how often it has risen over such a stretch and how far it usually "
        f"moves, or \"has {example} been here before\" for what followed similar setups.",
    ]


# --- private data ---------------------------------------------------------------------------------

_STATED_OR_HYPOTHETICAL = re.compile(
    r"\bif\b|\bsuppose\b|\bassum\w+|\bwhat\s+happens\b|\bi\s+(?:hold|own|have|am\s+(?:long|short))"
    r"\s+(?:about\s+|roughly\s+)?(?:\$?\d|(?-i:[A-Z]{2,6})\b)|\bi\s+(?:bought|sold|paid|entered|got\s+in)"
    r"\b[^?]{0,40}\d|\bmy\s+(?:book|portfolio|holdings)\s+(?:is|are|=|:)|"
    r"\d+(?:\.\d+)?\s*%\s*(?:of\s+)?[A-Za-z]|\bat\s+\$?\d|如果|假如|假设|我持有|我有\d",
    re.I)
"""The question brings its own facts — holdings, an entry, a hypothetical — so it is answerable."""

_PRIVATE = re.compile(
    # A bare "my account" is not here: "is my account too risky now", said with a saved book, is
    # a risk question the book answers (held-out corpus); nor is "my order of 2 BTC", which asks
    # how to split one, or "my equity longs", which are holdings.
    r"\bmy\s+bitget\s+(?:futures\s+|spot\s+)?account\b|\bwhat'?s\s+in\s+my\s+(?:account|wallet)\b"
    r"|\bmy\s+(?:bitget\s+|futures\s+|spot\s+|trading\s+|exchange\s+)*(?:(?:account|wallet)\s+"
    r"(?:balance|equity|value|margin|holdings)|balance|available\s+(?:balance|margin|funds)|"
    r"margin(?:\s+(?:left|level|ratio|balance))?|"
    r"open\s+orders?|pending\s+orders?|orders\b|fills?|order\s+(?:history|status)|trade\s+history|trading\s+history|"
    r"transaction\s+history|deposits?|withdrawals?|cost\s+basis|average\s+(?:cost|entry|price)|"
    r"entry\s+price|liquidation\s+price|(?:unreali[sz]ed|reali[sz]ed|open)\s+(?:p&l|pnl|"
    r"profits?|loss(?:es)?|gains?))\b"
    r"|\b(?:unreali[sz]ed|reali[sz]ed)\s+(?:p&l|pnl|profits?|loss(?:es)?|gains?)\s+on\s+my\b"
    r"|\bwhat\s+did\s+i\s+(?:pay|trade|buy|sell|spend)\b|\bwhat\s+have\s+i\s+(?:traded|bought|"
    r"sold)\b|\bhow\s+much\s+(?:margin|money|cash|usdt|funds?|balance)\s+(?:do\s+i|have\s+i|"
    r"i)\b|\bmy\s+(?:last|recent|latest)\s+(?:trades?|orders?|fills?)\b"
    r"|我的[^?？，,。]{0,8}(?:余额|餘額|账户|賬戶|订单|訂單|成交|保证金|保證金|盈亏|盈虧|盈利|"
    r"亏损|虧損|持仓成本|成本价)"
    r"|我的[^?？，,。]{0,8}(?:仓位|倉位|持仓|持倉)[^?？，,。]{0,8}(?:盈利|盈亏|亏损|赚|亏|多少)",
    re.I)
"""The asker's own account: balance, margin, orders, fills, history, cost basis, account P&L."""

_OTHERS = re.compile(
    r"\bwhich\s+(?:bitget\s+|other\s+)?(?:users|traders|accounts|people|clients|customers|"
    r"whales|funds)\s+(?:are|were|is|have|hold|own|bought|sold)\b"
    r"|\bwho\s+(?:is|are|was|were)\s+(?:long|short|buying|selling|holding)\b"
    r"|\b(?P<who>(?-i:[A-Z][\w&.-]+(?:\s+[A-Z][\w&.-]+){0,2}))(?:'s|’s)\s+(?:exact\s+|current\s+|live\s+|"
    r"latest\s+|real[\s-]time\s+|actual\s+)*(?:positions?|holdings|book|portfolio|trades|"
    r"orders)\b[^?.]{0,40}\b(?:today|right\s+now|now|live|currently|this\s+(?:morning|week))\b"
    r"|\b(?P<who2>(?-i:[A-Z][\w&.-]+(?:\s+[A-Z][\w&.-]+){0,2}))(?:'s|’s)\s+(?:exact|live|real[\s-]time)"
    r"\s+(?:positions?|holdings|book|portfolio|trades|orders)\b",
    re.I)
"""Other people's live positions: named users, or a named firm's positions *now*. A firm's 13F
holdings ("what did Berkshire buy last quarter") are public and do not match."""


def _private_lines() -> list[str]:
    return [
        "Actionable: this console cannot see your Bitget account — it holds no key to anyone's "
        "account, so nothing was read: no balance, margin, orders, fills or P&L.",
        "What it can do: state your book in My book (\"I hold 60% NVDA, 40% AAPL\") and it "
        "answers risk, stress and hedging for it; or paste your trades (date, symbol, side, "
        "size, price) and it reviews them.",
    ]


def _others_lines(text: str, names: Sequence[str]) -> list[str]:
    found = _OTHERS.search(text)
    who = ""
    if found is not None:
        who = found.group("who") or found.group("who2") or ""
    subject = f"{who}'s live positions are" if who else "Other traders' positions are"
    symbol = _short(names[0]) if names else "NVDA"
    return [
        f"Actionable: {subject} not public — no one outside the account can see them, and this "
        f"console has no access to anyone's account.",
        f"What is public: 13F filings (quarterly, filed up to 45 days after the quarter ends) "
        f"for a named manager's US holdings, and Bitget's aggregate long/short split, which "
        f"shows how the crowd leans without naming anyone — ask \"what's the long/short ratio "
        f"on {symbol}\".",
    ]


# --- dates ----------------------------------------------------------------------------------------

_MONTHS = {m: i for i, m in enumerate(
    ("january", "february", "march", "april", "may", "june", "july", "august", "september",
     "october", "november", "december"), start=1)}
_MONTH_RE = (r"(?P<mon>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|"
             r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)")


@dataclass(frozen=True, slots=True)
class Span:
    start: date
    end: date
    label: str


def _month(token: str) -> int:
    token = token.lower()
    return next(i for name, i in _MONTHS.items() if name.startswith(token[:3]))


def _month_end(year: int, month: int) -> date:
    return (date(year + (month == 12), month % 12 + 1, 1) - timedelta(days=1))


def spans_in(text: str) -> list[Span]:
    """Calendar dates, months and years a question names, as closed spans."""
    out: list[Span] = []
    taken: list[tuple[int, int]] = []

    def free(match: re.Match[str]) -> bool:
        a, b = match.span()
        return all(b <= x or a >= y for x, y in taken)

    patterns: tuple[tuple[str, str], ...] = (
        ("day", rf"\b(?P<d>\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?{_MONTH_RE}\s*,?\s+"
                r"(?P<y>(?:19|20)\d{2})\b"),
        ("day", rf"\b{_MONTH_RE}\s+(?P<d>\d{{1,2}})(?:st|nd|rd|th)?\s*,?\s+"
                r"(?P<y>(?:19|20)\d{2})\b"),
        ("xmas_eve", r"\bchristmas\s+eve\s*,?\s+(?P<y>(?:19|20)\d{2})\b"),
        ("xmas", r"\bchristmas(?:\s+day)?\s*,?\s+(?P<y>(?:19|20)\d{2})\b"),
        ("iso", r"\b(?P<y>(?:19|20)\d{2})-(?P<m>\d{2})-(?P<d>\d{2})\b"),
        ("month", rf"\b{_MONTH_RE}\s*,?\s+(?P<y>(?:19|20)\d{{2}})\b"),
        ("year", r"(?<![\d$.,])(?P<y>(?:19|20)\d{2})(?![\d%])(?!\s*(?:bps|x\b|shares|usd|usdt|"
                 r"dollars|points|contracts))"),
        ("zh_year", r"(?P<y>(?:19|20)\d{2})\s*年(?!\s*\d{1,2}\s*月)"),
    )
    for kind, pattern in patterns:
        for match in re.finditer(pattern, text, re.I):
            if not free(match):
                continue
            try:
                year = int(match.group("y"))
                if kind == "day":
                    day = date(year, _month(match.group("mon")), int(match.group("d")))
                    out.append(Span(day, day, f"{day.day} {day:%b %Y}"))
                elif kind == "iso":
                    day = date(year, int(match.group("m")), int(match.group("d")))
                    out.append(Span(day, day, day.isoformat()))
                elif kind == "xmas_eve":
                    out.append(Span(date(year, 12, 24), date(year, 12, 24),
                                    f"Christmas Eve {year}"))
                elif kind == "xmas":
                    out.append(Span(date(year, 12, 25), date(year, 12, 25),
                                    f"Christmas {year}"))
                elif kind == "month":
                    month = _month(match.group("mon"))
                    out.append(Span(date(year, month, 1), _month_end(year, month),
                                    f"{date(year, month, 1):%B %Y}"))
                else:
                    out.append(Span(date(year, 1, 1), date(year, 12, 31), str(year)))
            except (ValueError, StopIteration):
                continue
            taken.append(match.span())
    return out


# --- the desk's own record --------------------------------------------------------------------


def _ledger_span(path: Path = LEDGER_PATH) -> tuple[datetime, datetime, int] | None:
    """First and last ``decided_at`` and the decision count, read from the ledger file."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    stamps: list[datetime] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("kind", "decision") == "decision" and row.get("decided_at"):
            stamps.append(datetime.fromisoformat(str(row["decided_at"])))
    if not stamps:
        return None
    return min(stamps), max(stamps), len(stamps)


_DESK = re.compile(r"\b(?:the\s+desk|desk'?s?|argus|the\s+agent|your\s+(?:calls|decisions|"
                   r"record|ledger|trades|p&l|pnl)|you\s+(?:decide|do|trade|call|skip|pass|"
                   r"make)\w*|did\s+you)\b|\bthe\s+(?:record|ledger)\b", re.I)
_DESK_HORIZON = re.compile(
    r"\byear[\s-]+(?:over|on)[\s-]+year\b|\byoy\b|\bover\s+the\s+(?:last|past)\s+"
    r"(?P<n>\d+|two|three|four|five|six|nine|twelve|eighteen|twenty[\s-]four)\s+"
    r"(?P<unit>months?|years?|quarters?)\b|\bmonth[\s-]+by[\s-]+month\b[^?]{0,40}\b(?P<n2>\d+|"
    r"six|twelve)\s+months\b", re.I)
_NUMBER_WORDS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
                 "nine": 9, "ten": 10, "twelve": 12, "fifteen": 15, "eighteen": 18, "twenty": 20,
                 "twenty-four": 24, "twenty four": 24, "twenty-five": 25, "thirty": 30,
                 "fifty": 50, "one": 1, "a": 1}


def _number(token: str) -> int | None:
    token = token.lower().strip()
    if token.isdigit():
        return int(token)
    return _NUMBER_WORDS.get(token)


def _desk_days_asked(text: str) -> float | None:
    found = _DESK_HORIZON.search(text)
    if found is None:
        return None
    if found.group("n") or found.group("n2"):
        n = _number(found.group("n") or found.group("n2") or "") or 0
        unit = (found.group("unit") or "months").lower()
        per = 365.25 if unit.startswith("year") else 91.3 if unit.startswith("quarter") else 30.44
        return n * per
    return 2 * 365.25  # year over year needs at least two years


def _desk_before_lines(span: Span, ledger: tuple[datetime, datetime, int]) -> list[str]:
    first, last, count = ledger
    return [
        f"Actionable: the desk's record starts on {first.day} {first:%b %Y} — its first decision "
        f"— so there are no decisions, calls or P&L before that, and none for {span.label}.",
        f"What it holds: {count} decisions from {first:%Y-%m-%d} to {last:%Y-%m-%d}; ask "
        f"\"what did the desk decide on {first:%Y-%m-%d}\" or \"show me the latest decisions\".",
        "Data: the decision ledger (data/paper_ledger.jsonl), first and last decided_at.",
    ]


def _desk_horizon_lines(days: float, ledger: tuple[datetime, datetime, int]) -> list[str]:
    first, last, count = ledger
    span_days = (last - first).total_seconds() / 86400.0
    asked = f"{days / 365.25:.0f} years" if days >= 365 else f"{days / 30.44:.0f} months"
    return [
        f"Actionable: the desk's record spans only {span_days:.0f} days — it starts on "
        f"{first:%Y-%m-%d} — so {asked} of it do not exist yet, and neither does any comparison "
        f"across them.",
        f"What it holds: {count} decisions over those {span_days:.0f} days; ask \"what is the "
        f"sharpe\" or \"are you well calibrated\" for the whole record as it stands.",
        "Data: the decision ledger (data/paper_ledger.jsonl), first and last decided_at.",
    ]


# --- instruments and their history ---------------------------------------------------------------


def _short(symbol: str) -> str:
    base = symbol.removesuffix("USDT")
    return base.removesuffix("STOCK") if base.endswith("STOCK") and len(base) > 5 else base


_YAHOO_INDEX = {"SP500USDT": "^GSPC", "NDX100USDT": "^NDX", "XAUUSDT": "GC=F",
                "XAGUSDT": "SI=F", "CLUSDT": "CL=F", "BZUSDT": "BZ=F"}


def _is_crypto(symbol: str) -> bool:
    contract = _registry().get(symbol)
    if contract is not None:
        return not bool(getattr(contract, "rwa", False))
    return symbol in {"BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT"}


def _yahoo_ticker(symbol: str) -> str:
    return _YAHOO_INDEX.get(symbol) or _short(symbol)


@dataclass(frozen=True, slots=True)
class Series:
    closes: tuple[tuple[date, float], ...]
    source: str
    per_year: int
    """Bars a year: 252 for an exchange-traded name, 365 for a 24/7 perp."""


def _yahoo_series(ticker: str) -> Series:
    from argus.market import equity_history

    days = equity_history.daily(ticker)
    return Series(tuple((d.day, d.close) for d in days),
                  f"Yahoo Finance daily closes for {ticker}, split-adjusted", 252)


def _bitget_series(symbol: str, since: datetime | None = None) -> Series:
    from argus.market import history

    start = since or datetime(2017, 1, 1, tzinfo=UTC)
    candles = history.fetch_window(symbol, start=start, interval="1D", pause=0.25)
    if not candles:
        raise RuntimeError(f"no daily candles for {symbol}")
    return Series(tuple((c.ts.date(), float(c.close)) for c in candles),
                  f"Bitget {symbol} daily candles (history-candles, 1D)", 365)


def _series(symbol: str, *, perp: bool) -> Series | None:
    if perp or _is_crypto(symbol):
        return _with_timeout(lambda: _bitget_series(symbol))
    return _with_timeout(lambda: _yahoo_series(_yahoo_ticker(symbol)))


def _years(a: date, b: date) -> float:
    return (b - a).days / 365.25


# --- a price on a past date ---------------------------------------------------------------------

_PRICE_WORDS = re.compile(
    r"\b(?:trade[ds]?|trading|price[ds]?|priced|clos(?:e|ed|ing)|open(?:ed|ing)?|worth|quote[ds]?|"
    r"funding|level|valued|sell(?:ing)?\s+for|cost)\b|价格|價格|收盘|收盤|股价|股價", re.I)
_PERP = re.compile(r"\bperp(?:etual)?s?\b|\bon\s+bitget\b|\bfunding\b|合约|永续", re.I)
_FUNDING = re.compile(r"\bfunding\b|资金费率", re.I)


def _past_price_lines(symbol: str, span: Span, text: str, today: date) -> tuple[str, list[str]]:
    """The close on or around ``span`` from the data that exists, or where that data starts."""
    name = _short(symbol)
    perp_asked = bool(_PERP.search(text)) or _is_crypto(symbol)
    funding = bool(_FUNDING.search(text))
    lines: list[str] = []
    data: list[str] = []
    cause = PAST_PRICE
    if perp_asked:
        series = _series(symbol, perp=True)
        if series is None:
            return BEFORE_DATA, [
                f"Actionable: a price for {span.label} needs {symbol}'s daily history on Bitget, "
                f"and it did not arrive — so no figure is given rather than today's quote in its "
                f"place.",
                f"Ask again shortly, or ask \"where is {name} trading right now\" for today's.",
            ]
        first = series.closes[0][0]
        data.append(series.source)
        if span.end < first:
            cause = BEFORE_DATA
            what = "funding rate" if funding else "price"
            lines.append(
                f"Actionable: Bitget's {symbol} history starts on {first.day} {first:%b %Y} — its "
                f"first daily bar — so the perpetual did not exist in {span.label} and there is "
                f"no {what} for it then.")
        else:
            cause = PAST_PRICE
            lines.extend(_describe(series, span, f"Bitget's {symbol}", funding=funding))
    if not perp_asked or not _is_crypto(symbol):
        stock = _with_timeout(lambda: _yahoo_series(_yahoo_ticker(symbol)))
        if stock is not None:
            data.append(stock.source)
            first = stock.closes[0][0]
            if span.end < first:
                cause = BEFORE_DATA if cause != PAST_PRICE or not lines else cause
                lines.append(
                    f"{'' if lines else 'Actionable: '}{_yahoo_ticker(symbol)} was not yet "
                    f"listed in {span.label} — its daily history starts on "
                    f"{first.day} {first:%b %Y}, so it has no {span.label} price.")
            else:
                lead = "" if lines else "Actionable: "
                described = _describe(stock, span, f"{_yahoo_ticker(symbol)} (the stock)",
                                      funding=False)
                lines.extend([lead + described[0], *described[1:]] if described else [])
        elif not lines:
            return BEFORE_DATA, [
                f"Actionable: a price for {span.label} needs {name}'s daily history, and it did "
                f"not arrive — so no figure is given rather than today's quote in its place.",
                f"Ask again shortly, or ask \"where is {name} trading right now\" for today's.",
            ]
    lines.append(f"Today's quote is not an answer to a past date; ask \"where is {name} "
                 f"trading right now\" for that.")
    if data:
        lines.append("Data: " + "; ".join(data) + ".")
    return cause, lines


def _describe(series: Series, span: Span, label: str, *, funding: bool) -> list[str]:
    inside = [(d, c) for d, c in series.closes if span.start <= d <= span.end]
    if funding:
        return [f"{label} was trading in {span.label}, but this console reads only the current "
                f"funding rate, not its history, so it has no funding figure for that date."]
    if not inside:
        before = [(d, c) for d, c in series.closes if d <= span.end]
        if not before:
            return []
        d, c = before[-1]
        return [f"{label} closed at {c:,.2f} on {d:%Y-%m-%d}, the last session on or before "
                f"{span.label}."]
    if span.start == span.end:
        d, c = inside[0]
        return [f"{label} closed at {c:,.2f} on {d:%Y-%m-%d}."]
    lows = min(inside, key=lambda x: x[1])
    highs = max(inside, key=lambda x: x[1])
    return [f"{label} in {span.label}: first close {inside[0][1]:,.2f} ({inside[0][0]:%Y-%m-%d}), "
            f"last {inside[-1][1]:,.2f} ({inside[-1][0]:%Y-%m-%d}); lowest close "
            f"{lows[1]:,.2f} on {lows[0]:%Y-%m-%d}, highest {highs[1]:,.2f} on "
            f"{highs[0]:%Y-%m-%d}."]


# --- an N-year statistic -------------------------------------------------------------------------

_N_YEARS = re.compile(
    r"\b(?P<n>\d{1,3}|two|three|four|five|six|seven|eight|nine|ten|twelve|fifteen|twenty|"
    r"twenty[\s-]five|thirty|fifty)[\s-]+(?:years?|yr)\b|\b(?P<n2>\d{1,3})\s*y\b", re.I)
_STAT_WORDS = re.compile(
    r"\b(?:returns?|cagr|performance|perform(?:ed)?|sharpe|sortino|beta|drawdowns?|dd|"
    r"volatil\w*|vol|std|deviation|annuali[sz]ed|history|funding|correlation|gain(?:ed)?|"
    r"growth|how\s+(?:has|did|volatile)|track\s+record)\b", re.I)


def _stat(text: str) -> str:
    lower = text.lower()
    for key, words in (("beta", ("beta", "correlat")), ("sharpe", ("sharpe", "sortino")),
                       ("drawdown", ("drawdown", " dd")), ("volatility", ("volatil", " vol",
                                                                          "std", "deviation")),
                       ("funding", ("funding",))):
        if any(w in f" {lower}" for w in words):
            return key
    return "return"


def _benchmark(text: str) -> str:
    return "SPY" if re.search(r"s\s*&\s*p|\bspx\b|\bspy\b|\bmarket\b", text, re.I) else "QQQ"


def _returns(closes: Sequence[tuple[date, float]]) -> list[float]:
    return [b[1] / a[1] - 1 for a, b in itertools.pairwise(closes) if a[1] > 0]


def _max_drawdown(closes: Sequence[tuple[date, float]]) -> tuple[float, date, date]:
    peak, peak_day = closes[0][1], closes[0][0]
    worst, at_peak, at_trough = 0.0, closes[0][0], closes[0][0]
    for day, close in closes:
        if close > peak:
            peak, peak_day = close, day
        dd = close / peak - 1
        if dd < worst:
            worst, at_peak, at_trough = dd, peak_day, day
    return worst, at_peak, at_trough


def statistics_over(series: Series, *, years: float | None,
                    against: Series | None = None) -> dict[str, float | str]:
    """Annualised return, volatility, max drawdown, Sharpe (rf 0%) and beta, over the last
    ``years`` of ``series`` (all of it when None). Pure arithmetic, no reads."""
    closes = list(series.closes)
    if years is not None:
        cutoff = closes[-1][0] - timedelta(days=round(years * 365.25))
        closes = [(d, c) for d, c in closes if d >= cutoff]
    rets = _returns(closes)
    span = _years(closes[0][0], closes[-1][0])
    out: dict[str, float | str] = {
        "start": closes[0][0].isoformat(), "end": closes[-1][0].isoformat(), "years": span,
        "total_return": closes[-1][1] / closes[0][1] - 1,
    }
    out["annual_return"] = ((closes[-1][1] / closes[0][1]) ** (1 / span) - 1) if span > 0 else 0.0
    vol = statistics.pstdev(rets) if len(rets) > 1 else 0.0
    out["annual_volatility"] = vol * math.sqrt(series.per_year)
    out["sharpe_rf0"] = (statistics.fmean(rets) / vol * math.sqrt(series.per_year)
                         if vol > 0 else 0.0)
    dd, peak, trough = _max_drawdown(closes)
    out.update(max_drawdown=dd, drawdown_peak=peak.isoformat(),
               drawdown_trough=trough.isoformat())
    if against is not None:
        theirs = dict(against.closes)
        paired = [(d, c, theirs[d]) for d, c in closes if d in theirs]
        mine = [b[1] / a[1] - 1 for a, b in itertools.pairwise(paired)]
        bench = [b[2] / a[2] - 1 for a, b in itertools.pairwise(paired)]
        if len(mine) > 20 and statistics.pvariance(bench) > 0:
            out["beta"] = (statistics.covariance(mine, bench) / statistics.variance(bench))
            out["beta_days"] = float(len(mine))
    return out


def _stat_sentence(stat: str, s: Mapping[str, float | str], benchmark: str) -> str:
    def pct(key: str) -> str:
        return f"{float(s[key]) * 100:+.1f}%"

    if stat == "beta" and "beta" in s:
        return (f"beta to {benchmark} {float(s['beta']):.2f} on {int(float(s['beta_days']))} "
                f"daily returns")
    if stat == "sharpe":
        return (f"Sharpe {float(s['sharpe_rf0']):.2f} (risk-free rate taken as 0%; annualised "
                f"return {pct('annual_return')}, volatility "
                f"{float(s['annual_volatility']) * 100:.1f}%)")
    if stat == "drawdown":
        return (f"max drawdown {pct('max_drawdown')} (peak {s['drawdown_peak']}, trough "
                f"{s['drawdown_trough']})")
    if stat == "volatility":
        return f"annualised volatility {float(s['annual_volatility']) * 100:.1f}%"
    return (f"annualised return {pct('annual_return')} (total {pct('total_return')}), max "
            f"drawdown {pct('max_drawdown')}")


def _horizon_lines(symbol: str, years: float, text: str) -> tuple[str, list[str]]:
    name = _short(symbol)
    stat = _stat(text)
    asked = f"{years:g}-year"
    if stat == "funding":
        series = _series(symbol, perp=True)
        if series is None:
            return HORIZON, [
                f"Actionable: {years:g} years of {symbol} funding need the perpetual's history, "
                f"and Bitget's daily candles did not arrive — so nothing is computed.",
                f"Ask \"what's the funding on {name}\" for the current rate.",
            ]
        first = series.closes[0][0]
        have = _years(first, series.closes[-1][0])
        if have >= years:
            return LONG_HORIZON, [
                f"Actionable: {symbol} has traded on Bitget since {first:%Y-%m-%d}, so {years:g} "
                f"years exist, but this console reads only the current funding rate, not its "
                f"history.", f"Data: {series.source}."]
        return HORIZON, [
            f"Actionable: the {symbol} perpetual has only existed since {first:%Y-%m-%d} — its "
            f"history starts there, {have:.1f} years ago — so a {asked} funding history does "
            f"not exist.",
            f"Ask \"what's the funding on {name}\" for the current rate and what it costs to "
            f"hold.", f"Data: {series.source}."]
    series = _series(symbol, perp=False)
    if series is None:
        return HORIZON, [
            f"Actionable: a {asked} figure for {name} needs its full daily history, and the read "
            f"did not arrive — so nothing is computed rather than a shorter window passed off as "
            f"{years:g} years.",
            f"Ask again shortly, or ask \"how risky is {name}\" for its recent risk profile.",
        ]
    benchmark = _benchmark(text)
    against = (_with_timeout(lambda: _yahoo_series(benchmark)) if stat == "beta" else None)
    first, last = series.closes[0][0], series.closes[-1][0]
    have = _years(first, last)
    data = [series.source] + ([against.source] if against is not None else [])
    if have >= years:
        s = statistics_over(series, years=years, against=against)
        return LONG_HORIZON, [
            f"Actionable: {name} over the last {years:g} years ({s['start']} to {s['end']}): "
            f"{_stat_sentence(stat, s, benchmark)}.",
            "Data: " + "; ".join(data) + ".",
        ]
    s = statistics_over(series, years=None, against=against)
    source = "Bitget's daily history" if series.per_year == 365 else "its daily history"
    return HORIZON, [
        f"Actionable: {name} has only traded since {first:%Y-%m-%d} in the data — {source} starts "
        f"there, {have:.1f} years ago — so a {asked} figure does not exist.",
        f"Over the {have:.1f} years that do exist ({s['start']} to {s['end']}): "
        f"{_stat_sentence(stat, s, benchmark)}.",
        "Data: " + "; ".join(data) + ".",
    ]


# --- unlisted names -------------------------------------------------------------------------------

COMPANY_TICKERS: dict[str, tuple[str, ...]] = {
    "deutsche bank": ("DB", "DBK"), "saudi aramco": ("ARAMCO", "2222"), "aramco": ("ARAMCO",),
    "totalenergies": ("TTE", "TOTAL"), "total energies": ("TTE",), "nestle": ("NSRGY", "NESN"),
    "nestlé": ("NSRGY", "NESN"), "reliance industries": ("RELIANCE", "RELI"),
    "spacex": ("SPCX", "SPACEX"), "credit suisse": ("CS", "CSGN"), "twitter": ("TWTR",),
    "hsbc": ("HSBC",), "stripe": ("STRIPE",), "openai": ("OPENAI",), "anthropic": ("ANTHROPIC",),
    "bytedance": ("BYTEDANCE",), "shein": ("SHEIN",), "tencent": ("TCEHY", "TENCENT"),
    "samsung": ("SSNLF", "SAMSUNG"), "toyota": ("TM", "TOYOTA"), "lvmh": ("LVMUY", "LVMH"),
    "unilever": ("UL",), "siemens": ("SIEGY", "SIEMENS"), "novartis": ("NVS",),
    "roche": ("RHHBY",), "shell": ("SHEL",), "bp": ("BP",), "sony": ("SONY",),
    "nintendo": ("NTDOY",), "volkswagen": ("VWAGY",), "bmw": ("BMWYY",), "ubs": ("UBS",),
    "barclays": ("BCS",), "santander": ("SAN",), "tata motors": ("TTM",),
    "infosys": ("INFY",), "petrobras": ("PBR",), "gazprom": ("GAZP",),
    "berkshire hathaway": ("BRK", "BRKB"), "jpmorgan": ("JPM",), "jp morgan": ("JPM",),
    "goldman sachs": ("GS",), "morgan stanley": ("MS",), "bank of america": ("BAC",),
    "exxon": ("XOM",), "exxonmobil": ("XOM",), "walmart": ("WMT",), "disney": ("DIS",),
    "boeing": ("BA",), "pfizer": ("PFE",), "johnson & johnson": ("JNJ",),
    "coca-cola": ("KO",), "mcdonald's": ("MCD",), "visa": ("V",), "mastercard": ("MA",),
    "micron": ("MU",), "strategy": ("MSTR",), "microstrategy": ("MSTR",),
    "microsoft": ("MSFT",), "circle": ("CRCL",), "tsmc": ("TSM",), "asml": ("ASML",),
    "novo nordisk": ("NVO",), "eli lilly": ("LLY",), "alibaba": ("BABA",),
}
"""Company names people type, to the tickers they trade under (US listing first). This is a fact
about the companies, not about Bitget: whether Bitget lists any of the tickers is read from the
registry every time, so SpaceX, whose SPCX Bitget does list, never reads as unlisted."""

COMPANY_STATUS: dict[str, str] = {
    "twitter": "taken private in October 2022 (now X Corp.), so it has no listed shares at all",
    "credit suisse": "acquired by UBS in June 2023, so its shares no longer trade",
    "stripe": "privately held", "openai": "privately held", "anthropic": "privately held",
    "bytedance": "privately held", "shein": "privately held",
    "saudi aramco": "listed only on the Saudi exchange (Tadawul, 2222)",
    "aramco": "listed only on the Saudi exchange (Tadawul, 2222)",
    "reliance industries": "listed in India (NSE: RELIANCE)",
}

_COMPANY_RE = re.compile(
    r"(?<![\w&])(" + "|".join(sorted((re.escape(n) for n in COMPANY_TICKERS), key=len,
                                     reverse=True)) + r")(?![\w&])", re.I)
_UNKNOWN_NAME = re.compile(
    r"\b(?P<name>[A-Z][a-zA-Z&.]+(?:\s+[A-Z][a-zA-Z&.]+){0,2})(?:'s|’s)?\s+(?:stock|shares|"
    r"share\s+price)\b|\bshares\s+of\s+(?P<name2>[A-Z][a-zA-Z&.]+(?:\s+[A-Z][a-zA-Z&.]+){0,2})")
_NOT_A_COMPANY = frozenset({"The", "A", "My", "Our", "This", "That", "Which", "What", "Is", "US",
                            "Tech", "Chip", "Bank", "Oil", "Energy", "Growth", "Value", "Meme",
                            "Tokenized", "Tokenised", "Bitget", "Nasdaq", "NYSE", "Any", "Each",
                            "Every", "Top", "Best", "Big", "China", "Chinese", "Japanese",
                            "European", "Indian", "American", "Should", "Would", "How", "Why",
                            "Will", "Can", "Does", "Do", "I"})


def _listed_tickers(tickers: Sequence[str], registry: Mapping[str, Any],
                    spot: frozenset[str] = frozenset()) -> list[str]:
    hits: list[str] = []
    for ticker in tickers:
        for symbol in (f"{ticker}USDT", f"{ticker}STOCKUSDT"):
            if symbol in registry:
                hits.append(symbol)
        for symbol in (f"R{ticker}USDT", f"{ticker}USDT"):
            if symbol in spot:
                hits.append(f"{symbol} (spot)")
    return hits


def _unlisted_candidate(text: str, registry: Mapping[str, Any]) -> tuple[str, tuple[str, ...],
                                                                         str] | None:
    """(name as typed, tickers checked, status) for a company Bitget lists nothing for."""
    for match in _COMPANY_RE.finditer(text):
        key = match.group(1).lower()
        tickers = COMPANY_TICKERS[key]
        if key in ("strategy", "circle", "visa", "shell") and not match.group(1)[0].isupper():
            continue  # the ordinary words
        if not _listed_tickers(tickers, registry):
            return match.group(1), tickers, COMPANY_STATUS.get(key, "")
    for match in _UNKNOWN_NAME.finditer(text):
        name = (match.group("name") or match.group("name2") or "").strip()
        words = name.split()
        while words and words[0] in _NOT_A_COMPANY:
            words = words[1:]
        if not words:
            continue
        name = " ".join(words)
        if re.fullmatch(r"[A-Z]{1,6}", name):
            continue  # a ticker: `lui/question.extract_symbols` and the spot list own those
        if any(_contract_for(w, registry) for w in words) or names_in(name, registry):
            continue
        squashed = re.sub(r"[^A-Za-z]", "", name).upper()
        if name.lower() in COMPANY_TICKERS or _contract_for(squashed, registry):
            continue
        return name, (squashed,), ""
    return None


def _spot_symbols() -> frozenset[str]:
    import urllib.request

    request = urllib.request.Request("https://api.bitget.com/api/v2/spot/public/symbols",
                                     headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=8.0) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return frozenset(str(r.get("symbol")) for r in payload.get("data") or [])


def _unlisted_lines(name: str, tickers: Sequence[str], status: str) -> tuple[str,
                                                                          list[str]] | None:
    from argus.market import universe

    live = _with_timeout(universe.contracts, 10.0)
    registry: Mapping[str, Any] = live if live is not None else _registry()
    origin = universe.origin() if live is not None else "frozen snapshot"
    spot = _with_timeout(_spot_symbols, 10.0) or frozenset()
    hits = _listed_tickers(tickers, registry, spot)
    if hits:
        return None  # it is listed after all; the console's own answer stands
    checked = f"{len(registry)} USDT-futures contracts ({origin})"
    if spot:
        checked += f" and {len(spot)} spot symbols (live)"
    said = " or ".join(tickers)
    example = next((s for s in ("JPMUSDT", "GSUSDT", "XOMUSDT", "NVDAUSDT") if s in registry),
                   "NVDAUSDT")
    lead = (f"Actionable: {name} is not listed on Bitget — no perpetual and no spot or rToken "
            f"market under {said} among the {checked} checked — so there is no Bitget data to "
            f"answer from.")
    lines = [lead]
    if status:
        lines.append(f"{name} is {status}.")
    lines.append(f"Any contract Bitget lists can be asked about — e.g. \"how risky is "
                 f"{_short(example)}\"; if {name} trades under another ticker, name the ticker.")
    lines.append(f"Data: Bitget contract list ({checked}).")
    return UNLISTED, lines


# --- ambiguous ------------------------------------------------------------------------------------

_DEICTIC = re.compile(
    r"\b(?:this|that|the)\s+(?:(?:chip|tech|bank|oil|ai|ev|energy|meme|crypto|semiconductor|"
    r"mining|software|big)\s+)?(?:stock|coin|token|name|company|ticker|contract|asset|share)s?\b"
    r"|\bthe\s+perp(?:etual)?\b|\b(?:the\s+two|those\s+two|these\s+two|both\s+of\s+them|"
    r"either\s+one|the\s+other\s+one)\b|\bwhich\s+of\s+(?:the|those|these)\s+(?:two|three)\b")
_ZH_TOPIC = re.compile(r"^(?:它的|这个的|那个的)?(?:技术面|基本面|资金费率|走势|估值|超买|超卖)"
                       r"(?:怎么样|如何|呢|吗|多少)?[?？]?$")
"""A bare Chinese research topic with no name ("技术面怎么样"). "风险层最近拦截了哪些交易" (what the
risk layer blocked) names the desk and must not match (blind corpus)."""
_PARTIAL = re.compile(r"\b(?P<w>[A-Z][a-z]{3,})\b")


def _partial_matches(word: str) -> list[str]:
    """Listed names that start with ``word`` ("Micro" → Microsoft, Micron, MicroStrategy)."""
    from argus.market import universe

    lower = word.lower()
    names = {n.lower(): t[0] for n, t in COMPANY_TICKERS.items()}
    names.update({a.lower(): _short(s) for a, s in universe.ALIASES.items() if len(a) > 3})
    hits = sorted({t for n, t in names.items() if n.startswith(lower) and n != lower})
    return hits if len(hits) >= 2 else []


_TRADING_WORD = re.compile(
    r"\b(?:stocks?|shares?|perps?|perpetuals?|coins?|tokens?|prices?|risk\w*|volatil\w*|"
    r"technicals?|overbought|oversold|funding|trad\w*|buy|sell|hedge\w*|chart|rsi|macd|beta|"
    r"drawdown|returns?|doing|moving|up|down|safer|cheaper|valuation|earnings)\b", re.I)


def _ambiguous(text: str, prior: Sequence[str], names: Sequence[str]) -> str | None:
    if names or [p for p in prior if p.strip()]:
        return None
    deictic = _DEICTIC.search(text.lower())
    if deictic and _TRADING_WORD.search(text):
        # "these two" is only a missing instrument when the question is about markets: "help me
        # write a sql query to join these two tables" is off-topic, not ambiguous (held-out corpus)
        return deictic.group(0)
    for match in _PARTIAL.finditer(text):
        if match.start() == 0:
            continue
        hits = _partial_matches(match.group("w"))
        if hits:
            return f"{match.group('w')}|{','.join(hits)}"
    if _ZH_TOPIC.search(text.strip()):
        return "zh"
    return None


def _ambiguous_lines(found: str) -> list[str]:
    if "|" in found:
        word, hits = found.split("|", 1)
        options = hits.replace(",", ", ")
        return [f"Actionable: which one? \"{word}\" could mean more than one listed name — "
                f"{options} — so nothing was looked up.",
                f"Name the ticker: e.g. \"how is {hits.split(',')[0]} doing today\"."]
    if found == "zh":
        return ["Actionable: 哪一个？问题里没有点名任何标的，也没有之前的问题可以指代 — "
                "which one: no instrument was named and there is no earlier question to refer to.",
                "请带上代码或名称，例如：\"英伟达技术面怎么样\" 或 \"is NVDA overbought\"。"]
    return [f"Actionable: which one? \"{found}\" names no instrument, and there is no earlier "
            f"question for it to refer to — so nothing was looked up.",
            "Name the ticker — e.g. \"how risky is NVDA?\" — or ask about two by name: \"is TSLA "
            "riskier than NVDA\"."]


# --- the entry point ----------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Detected:
    cause: str
    symbol: str = ""
    span: Span | None = None
    years: float = 0.0
    detail: str = ""


def detect(text: str, *, prior: Sequence[str] = (), book: str = "",
           today: date | None = None) -> Detected | None:
    """Which cause makes ``text`` unanswerable as asked, or None. Never touches the network: it
    reads the words, the turns, the ledger file and the offline contract registry."""
    from argus.lui.normalise import fold

    raw = fold(text).strip()
    if not raw or order_prefix(raw) is not None:
        return None
    now = today or datetime.now(UTC).date()
    registry = _registry()
    names = names_in(raw, registry)
    spans = spans_in(raw)
    future_span = any(s.start > now for s in spans)
    if future_price_asked(raw) or (future_span and _PRICE_WORDS.search(raw)
                                   and re.search(r"\b(?:exact|will|be)\b", raw, re.I)):
        return Detected(FUTURE_PRICE, names[0] if names else "")
    stated = bool(_STATED_OR_HYPOTHETICAL.search(raw))
    if _OTHERS.search(raw):
        return Detected(OTHERS_POSITIONS, names[0] if names else "")
    if _PRIVATE.search(raw) and not stated:
        return Detected(PRIVATE_ACCOUNT)
    past = [s for s in spans if s.end < now - timedelta(days=1)]
    if _DESK.search(raw) and not names[1:]:
        ledger = _ledger_span()
        if ledger is not None:
            early = [s for s in past if s.end < ledger[0].date()]
            if early:
                return Detected(BEFORE_DATA, span=early[0], detail="desk")
            days = _desk_days_asked(raw)
            span_days = (ledger[1] - ledger[0]).total_seconds() / 86400.0
            if days is not None and days > span_days and not names:
                return Detected(HORIZON, years=days / 365.25, detail="desk")
    if names and past and _PRICE_WORDS.search(raw) and not _N_YEARS.search(raw) \
            and not re.search(r"\b(?:since|from)\b", raw, re.I):
        return Detected(BEFORE_DATA, names[0], span=past[0], detail="price")
    years_match = _N_YEARS.search(raw)
    if names and years_match and _STAT_WORDS.search(raw) and not stated:
        n = _number(years_match.group("n") or years_match.group("n2") or "")
        if n is not None and n >= 2:
            return Detected(HORIZON, names[0], years=float(n), detail="instrument")
    if not names:
        candidate = _unlisted_candidate(raw, registry)
        if candidate is not None:
            return Detected(UNLISTED, detail=json.dumps(
                [candidate[0], list(candidate[1]), candidate[2]]))
    found = _ambiguous(raw, prior, names)
    if found is not None and not book.strip():
        return Detected(AMBIGUOUS, detail=found)
    if found is not None and book.strip() and "|" in found:
        return Detected(AMBIGUOUS, detail=found)
    return None


def honest_answer(text: str, *, prior: list[str], book: str,
                  today: date | None = None) -> tuple[str, list[str]] | None:
    """``(cause, lines)`` for a question no console can answer as asked, else None.

    Lines are in the console's style: an ``Actionable:`` lead stating the true reason plainly,
    then what the console can do instead, then a ``Data:`` line when anything was read. Only the
    past-price and N-year builders read data, with :data:`READ_TIMEOUT_S`, and both fall back to
    the plain reason when the read fails. An order is not answered here — :func:`order_prefix`
    puts its statement above the console's own analysis.
    """
    found = detect(text, prior=prior, book=book, today=today)
    if found is None:
        return None
    now = today or datetime.now(UTC).date()
    names = (found.symbol,) if found.symbol else ()
    if found.cause == FUTURE_PRICE:
        return FUTURE_PRICE, _future_lines(text, names)
    if found.cause == PRIVATE_ACCOUNT:
        return PRIVATE_ACCOUNT, _private_lines()
    if found.cause == OTHERS_POSITIONS:
        return OTHERS_POSITIONS, _others_lines(text, names)
    if found.cause == BEFORE_DATA and found.detail == "desk" and found.span is not None:
        ledger = _ledger_span()
        return (BEFORE_DATA, _desk_before_lines(found.span, ledger)) if ledger else None
    if found.cause == HORIZON and found.detail == "desk":
        ledger = _ledger_span()
        return (HORIZON, _desk_horizon_lines(found.years * 365.25, ledger)) if ledger else None
    if found.cause == BEFORE_DATA and found.span is not None:
        return _past_price_lines(found.symbol, found.span, text, now)
    if found.cause == HORIZON:
        return _horizon_lines(found.symbol, found.years, text)
    if found.cause == UNLISTED:
        name, tickers, status = json.loads(found.detail)
        return _unlisted_lines(str(name), [str(t) for t in tickers], str(status))
    if found.cause == AMBIGUOUS:
        return AMBIGUOUS, _ambiguous_lines(found.detail)
    return None


__all__ = [
    "AMBIGUOUS", "BEFORE_DATA", "COMPANY_TICKERS", "FUTURE_PRICE", "HORIZON", "LONG_HORIZON",
    "ORDER_INSTRUCTION", "OTHERS_POSITIONS", "PAST_PRICE", "PRIVATE_ACCOUNT", "UNLISTED",
    "Detected", "Series", "Span", "detect", "future_price_asked", "honest_answer", "names_in",
    "order_prefix", "spans_in", "statistics_over",
]
