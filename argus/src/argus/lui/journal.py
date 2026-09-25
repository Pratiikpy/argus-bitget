"""A review of the trader's OWN trades: what each one did, the habits they share, and a checklist.

**Why this module exists, measured before it was written.** The 2026-09-25 readiness audit pasted
"bought NVDA at 188 sold at 176, bought COIN at 330 sold at 301 ... what bad patterns do I have and
give me a checklist" into the hosted console, and it answered with the *desk's* statistics. Nothing
in the console read a trader's trades out of a message: `eval/regression_gate.journal_review` takes
a JSON-lines file with an ``at`` timestamp and an ``outcome_bps`` on every row and refuses anything
under :data:`~argus.eval.regression_gate.MIN_JOURNAL_TRADES` (40) graded trades, which is right for
a rule that must survive a held-out half and useless for the six trades a trader types from memory.
This module reads what a trader actually pastes, computes what can be computed honestly for each
trade, counts the recurring habits, and says plainly when a handful of trades is an anecdote.

**What was taken, from where.**

* `hkuds/vibe-trading` (MIT), ``agent/src/tools/trade_journal_tool.py``:
  - ``pair_trades_fifo`` (``:72-213``): fills are paired per symbol, first in first out, and a sell
    that finds no open long opens a short (a buy first covers open shorts). Taken for pasted fills.
  - ``_disposition_effect`` (``:314-347``): losers held longer than winners, as the ratio of mean
    holding times, with 1.2 as the cut-off where it starts to count. Taken, cut-off included, and
    applied a second time to *size* (mean loss against mean win), which is the half of the
    disposition effect a trader who gives no dates can still be tested on.
  - ``_BUY_TOKENS``/``_SELL_TOKENS`` in ``trade_journal_parsers.py:34-61``: Chinese side words
    (买入, 卖出, 做多, 做空) beside the English ones. Taken as vocabulary.
* Rejected from the same file: ``_chasing_momentum`` (``:407-436``) calls a buy a chase when the
  *trader's own* previous fills in the name were 3% lower, which is a statement about the trader's
  fills, not the market; here a chase is an entry after the name's own top-decile day, read from
  its price history. ``_severity``'s low/medium/high labels (``:303-311``) are not used: a count
  with its denominator says more and cannot be mistaken for a test. ``_overtrading`` and
  ``_anchoring`` need at least four trading days and five fills per name, which a pasted handful
  never has; they are left out rather than run on nothing.
* Bitget's fill fields, read from source, never guessed. Bitget's own toolkit documents the fills
  *request* (``agent-skill/references/commands.md:587-606``, ``GET /api/v3/trade/fills``;
  ``agent-sdk/src/generated/catalog.ts:134``) but not the response. The response fields are from
  ``tiagosiebler/bitget-api`` (MIT): ``src/types/response/v3/trade.ts:83-99`` (``FillV3``: execId,
  orderId, category, symbol, orderType, side, execPrice, execQty, execValue, tradeScope, feeDetail,
  execPnl, tradeSide, createdTime, updatedTime) and ``src/types/response/v2/futures.ts:414-433``
  (``FuturesOrderFillV2``: tradeId, symbol, orderId, price, baseVolume, feeDetail, side,
  quoteVolume, profit, tradeSide, posMode, tradeScope, cTime). The column names of the CSV that
  Bitget's *website* exports are NOT VERIFIED — no copy was available — so the header reader also
  accepts the plain English and Chinese names a spreadsheet uses (Date, Pair, Direction, Price,
  Amount, Fee; 时间, 交易对, 方向, 成交价, 数量, 手续费) and reports which columns it read.
* ``tradeSide`` "close" is read as *reduces the open position, whichever way it is*, not as a
  direction. Bitget's hedge mode reports a closing fill with the position's side rather than the
  trade's, and the documentation available here does not pin the convention down; a "close" row
  must shrink an open position under either reading, so that is all it is used for. An "open" row
  opens in the direction of ``side`` under both.
* The console's own engines, reused rather than copied: the session calendar with Lean's US
  holidays (`truth/clocks.py`, `eval/baselines/lean_market_holidays_loader.py`), the daily-close
  sources `research._daily_closes` chooses (Yahoo split-adjusted closes for a stock, Bitget daily
  candles otherwise; `research.py:6900-6919`), the earnings release times from SEC EDGAR 8-K item
  2.02 acceptance (`research/event_reactions.earnings_releases`, `market/evidence.EdgarSource`), and
  the window distribution of `desk/odds.directional_odds` for "was this move inside the name's
  ordinary noise at that horizon". EDGAR's ``acceptanceDateTime`` was checked to be true UTC on
  NVDA's six 2.02 filings (2026-09-26: 20:21 UTC in daylight-saving months, 21:31 in winter, both
  16:2x-16:3x New York), so a release is placed before or after the close from its own time.
* `eval/regression_gate.journal_review` is called, unchanged, when a pasted journal is large and
  timed enough for it (:data:`GATE_MIN_TRADES` graded trades, every one with timestamps): the
  held-out rule gate then says which habits survive out of sample. Below that it is not run,
  because it refuses, and the count-based review below is what the trader gets, labelled as such.

**The honesty rules.** Every figure comes from the trader's own prices or from a named price
history; a default the reader applied is an ``Assumed:`` line; anything that could not be read or
checked is a ``Missing:`` line. With fewer than :data:`MIN_STATISTICAL_TRADES` closed trades the
answer says the patterns are anecdotes, not statistics — a habit seen twice in four trades is worth
checking, not a finding. Fees and funding are not in a typed trade, so returns are before costs and
say so; a pasted fill file with fees is net of them.
"""

from __future__ import annotations

import csv
import itertools
import json
import re
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from datetime import time as clock_time
from typing import Any
from zoneinfo import ZoneInfo

from argus.lui.answer import Source
from argus.truth.trace import every, traced

NEW_YORK = ZoneInfo("America/New_York")

MIN_STATISTICAL_TRADES = 5
"""Closed trades below which every pattern is called an anecdote. The task's own floor ("with
fewer than ~5 trades, say the patterns are anecdotes"); review's rule floor is far higher
(`desk/review.MIN_DECISIONS` = 20) and is what the held-out gate uses."""

GATE_MIN_TRADES = 40
"""`eval/regression_gate.MIN_JOURNAL_TRADES` (2 x `desk/review.MIN_DECISIONS`), restated so this
module does not import the replay machinery for a six-trade question; checked equal in the tests."""

DISPOSITION_RATIO = 1.2
"""vibe-trading's cut-off for the disposition effect (``trade_journal_tool.py:337``,
``_severity(ratio, (1.2, 1.5))``): losers held (here also: lost) 1.2x the winners' before it
counts."""

NOISE_HORIZONS = (1, 5, 20)
"""Sessions tried, shortest first, when a trade gives no dates: a move is described by the shortest
horizon whose ordinary range contains it."""

BAND = "10th-90th percentile"
"""What "ordinary noise" means here: inside the middle 80% of the name's past moves over windows of
the same length (`desk/odds.directional_odds` p10/p90)."""

LARGE_DAY_SHARE = 0.10
"""An entry "after a large up day" is one whose previous session was in the name's top 10% of daily
gains over the year before (bottom 10% of daily moves for a short)."""

HISTORY_SESSIONS = 500
RANK_SESSIONS = 250
FETCH_DEADLINE_S = 20.0
MAX_TEXT = 60_000
MAX_LEGS = 2_000

YAHOO_REF = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
EDGAR_REF = "https://data.sec.gov/submissions/ (8-K item 2.02, acceptanceDateTime)"


# --- the pieces of a trade ----------------------------------------------------------------------


@dataclass
class Leg:
    """One fill or one typed statement ("bought NVDA at 188"). ``action`` is buy, sell, short,
    cover or close; a close shrinks whatever is open."""

    action: str
    symbol: str | None
    price: float | None
    qty: float | None = None
    day: date | None = None
    at: datetime | None = None
    fee: float | None = None
    venue_pnl: float | None = None
    open_close: str | None = None
    flags: set[str] = field(default_factory=set)
    text: str = ""


@dataclass(frozen=True)
class Trade:
    """A closed round trip: an entry (possibly several lots) and the exit that closed it."""

    symbol: str
    side: str
    entry: float
    exit: float
    opened: date | None
    closed: date | None
    qty: float | None
    lots: int
    averaged_down: bool
    flags: frozenset[str]
    fees: float | None = None
    venue_pnl: float | None = None
    opened_at: datetime | None = None
    closed_at: datetime | None = None
    order: int = 0

    @property
    def return_pct(self) -> float:
        """The trade's return on its own entry price, in percent, before costs."""
        move = self.exit / self.entry - 1.0
        return 100.0 * (move if self.side == "long" else -move)

    @property
    def price_move_pct(self) -> float:
        return 100.0 * (self.exit / self.entry - 1.0)

    @property
    def notional(self) -> float | None:
        return None if self.qty is None else self.qty * self.entry

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol, "side": self.side, "entry": self.entry, "exit": self.exit,
            "opened": None if self.opened is None else self.opened.isoformat(),
            "closed": None if self.closed is None else self.closed.isoformat(),
            "qty": self.qty, "lots": self.lots, "averaged_down": self.averaged_down,
            "flags": sorted(self.flags), "fees": self.fees, "venue_pnl": self.venue_pnl,
            "return_pct": round(self.return_pct, 3),
        }


@dataclass
class Check:
    """What was computed for one trade. None means it could not be checked, never "no"."""

    trade: Trade
    sessions: int | None = None
    closures: int | None = None
    earnings: bool | None = None
    earnings_day: date | None = None
    noise_sessions: int | None = None
    band: tuple[float, float] | None = None
    inside_noise: bool | None = None
    prior_move_pct: float | None = None
    prior_rank: float | None = None
    chased: bool | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "trade": self.trade.as_dict(), "sessions": self.sessions, "closures": self.closures,
            "earnings": self.earnings,
            "earnings_day": None if self.earnings_day is None else self.earnings_day.isoformat(),
            "noise_sessions": self.noise_sessions,
            "band_pct": None if self.band is None else [round(b, 2) for b in self.band],
            "inside_noise": self.inside_noise, "prior_move_pct": self.prior_move_pct,
            "prior_rank": self.prior_rank, "chased": self.chased,
        }


@dataclass(frozen=True)
class Pattern:
    """A habit found in the trades: which ones show it, of how many it could be checked on, and
    what it cost in summed trade return (percentage points)."""

    key: str
    name: str
    members: tuple[int, ...]
    of: int
    cost: float
    finding: str
    check: str

    def as_dict(self) -> dict[str, Any]:
        return {"key": self.key, "name": self.name, "count": len(self.members), "of": self.of,
                "cost_pct_points": round(self.cost, 2), "finding": self.finding,
                "check": self.check, "members": list(self.members)}


# --- reading free text ----------------------------------------------------------------------------

_NUM = r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"

_VERBS: tuple[tuple[str, str], ...] = (
    # order matters: the longer phrase first, so "sold short" is not read as "sold"
    ("short", r"\bsold\s+short\b|\bshorted\b|\bwent\s+short\b|\bshorting\b|\bopened\s+(?:a\s+)?"
              r"short\b|做空|开空|空了"),
    ("cover", r"\bcovered\b|\bbought\s+(?:it\s+|them\s+)?back\b|平空|回补"),
    ("close", r"\bstopped\s+out\b|\bgot\s+stopped\b|\bstop(?:\s*-?\s*loss)?\s+(?:got\s+)?"
              r"(?:hit|triggered|filled)\b|\bhit\s+my\s+stop\b|\bexited\b|\bexit(?:ed)?\s+(?=@|at\b|"
              r"\$?\d)|\bclosed\b|\bgot\s+out\b|\btook\s+(?:the\s+)?(?:profits?|loss)\b|"
              r"\bcut\s+(?:it|the\s+loss|my\s+loss)\b|平仓|平多|止损|止盈|清仓|出掉|出了|割肉"),
    ("sell", r"\bsold\b|\bdumped\b|卖出|卖了|卖掉|卖的"),
    ("buy", r"\bbought\b|\bwent\s+long\b|\blonged\b|\bgot\s+(?:long|in)\b|\bentered\b|"
            r"\bentry\s+(?=@|at\b|\$?\d)|\badded\b|\baveraged\s+down\b|买入|买了|买进|买的|建仓|"
            r"开多|做多|加仓|补仓|抄底"),
)
_VERB = re.compile("|".join(f"(?P<{name}>{pattern})" for name, pattern in _VERBS), re.I)

_ROUND_TRIP = re.compile(
    rf"(?:(?P<sym1>\$?[A-Za-z][A-Za-z0-9.]{{0,14}}|[一-鿿]{{2,5}})\s+)?"
    rf"(?P<side>long\b|short\b|做多|做空)\s*(?:(?P<sym2>\$?[A-Za-z][A-Za-z0-9.]{{0,14}}|"
    rf"[一-鿿]{{2,5}})\s*)?(?:from\s+|at\s+|@\s*|在\s*)?\$?(?P<p1>{_NUM})\s*"
    rf"(?:->|→|=>|to|-|–|到|至)\s*\$?(?P<p2>{_NUM})", re.I)  # noqa: RUF001 (a pasted en dash)
""""long NVDA from 188 to 176", "NVDA short 250 -> 270", "做多英伟达 188到176": a whole round trip
in one phrase."""

_MONTHS = {m: i for i, m in enumerate(
    ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"), start=1)}
_DATE_ISO = re.compile(r"(?<!\d)(20\d\d)[-/.](\d{1,2})[-/.](\d{1,2})(?!\d)")
"""Digit lookarounds, not ``\\b``: a CJK character is a word character, so "在2026-08-20以180"
has no word boundary before the year and read 2026 as the entry price (live run, 2026-09-26)."""
_DATE_CJK = re.compile(r"(?:(20\d\d)\s*年\s*)?(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]")
_DATE_MON_D = re.compile(
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+(\d{1,2})(?:st|nd|rd|th)?"
    r"\b(?:,?\s*(20\d\d)\b)?", re.I)
_DATE_D_MON = re.compile(
    r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*"
    r"\.?(?:,?\s*(20\d\d)\b)?", re.I)
_DATE_SLASH = re.compile(r"(?<![\d/.])(\d{1,2})/(\d{1,2})(?![\d/])")

_QTY = re.compile(
    rf"{_NUM}\s*(k)?\s*(?:shares?|sh\b|units?|contracts?|lots?\b|股|个|手|张|枚)", re.I)
_QTY_COIN = re.compile(rf"{_NUM}\s*(btc|eth|sol|bitcoin|比特币|以太坊)\b", re.I)
_PRICE_AT = re.compile(
    rf"(?:\bat\b|@|\bfor\b|\baround\b|\bnear\b|~|\bprice\s*(?:of\s+)?|\bavg\b|\baverage\b|"
    rf"在|以|价格?|价位|成本|均价)\s*(?:是|为|:|：)?\s*\$?\s*{_NUM}\s*(k\b)?",  # noqa: RUF001
    re.I)
_PRICE_BEFORE_CJK_VERB = re.compile(
    rf"{_NUM}\s*(?:元|美元|刀|u\b|usdt\b)?\s*(?=买|卖|做|平|止|开|建|加|补|出|割|抄|的价)", re.I)
_BARE_NUMBER = re.compile(
    rf"(?<![A-Za-z0-9_.%/]){_NUM}\s*(k\b)?(?!\s*(?:%|x\b|percent|pct|倍|个百分点))", re.I)

_FLAGS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("stopped", re.compile(r"\bstopped\s+out\b|\bgot\s+stopped\b|\bstop(?:\s*-?\s*loss)?\s+"
                           r"(?:got\s+)?(?:hit|triggered|filled)\b|\bhit\s+my\s+stop\b|止损",
                           re.I)),
    ("weekend", re.compile(r"\b(?:over|through|across|into)\s+(?:the\s+)?weekend\b|过周末|周末",
                           re.I)),
    ("earnings", re.compile(r"\b(?:over|through|into|across|before|ahead\s+of|during)\s+(?:the\s+)?"
                            r"(?:earnings|er\b|report|results|print)|财报", re.I)),
    ("chased", re.compile(r"\bfomo\b|\bchas(?:ed|ing)\b|追高|追涨", re.I)),
    ("revenge", re.compile(r"\brevenge\b|报复", re.I)),
)

_REVIEW_CUE = re.compile(
    r"\bwhat\s+(?:did\s+i\s+do\s+wrong|am\s+i\s+doing\s+wrong|went\s+wrong)\b|"
    r"\b(?:bad|losing|recurring|my)\s+(?:trading\s+)?(?:habits?|patterns?)\b|\bpatterns?\b|"
    r"\bmistakes?\b|\bchecklist\b|\breview\b|\bfeedback\b|\bcritique\b|\bgrade\b|\blessons?\b|"
    r"\bhabits?\b|\bjournal\b|\bpost[\s-]?mortem\b|\bwhere\s+(?:do|did)\s+i\s+(?:go\s+wrong|lose)\b|"
    r"复盘|毛病|坏习惯|错在|教训|清单|总结|问题出在|哪里做错|什么问题", re.I)
"""A request to review what was done. With one completed trade it makes the message a journal;
with two or more the trades alone do."""

_REVIEW_REQUEST = re.compile(
    r"\b(?:review|analy[sz]e|critique|grade|audit|go\s+(?:through|over)|look\s+(?:at|over)|"
    r"assess|evaluate)\s+(?:my|these|the\s+following)\s+(?:(?:past|last|recent|own|\d+)\s+)*"
    r"(?:trades|trading(?:\s+history)?|fills|trade\s+history|trade\s+log|journal|executions)\b|"
    r"\bmy\s+(?:trading|trade)\s+(?:journal|log|history|habits|mistakes|patterns)\b|"
    r"\bwhat\s+(?:bad\s+)?(?:habits|patterns)\s+do\s+i\s+have\b|"
    r"复盘\s*(?:一下\s*)?(?:我的|我)\s*(?:交易|操作)|我的\s*(?:交易|操作)\s*(?:记录|习惯|日志|问题|毛病)",
    re.I)
"""A request to review the trader's own trades with none pasted. Answered with what to paste,
never with the desk's record."""

_DESK = re.compile(r"\bthe\s+desk\b|\bdesk's\b|\byour\s+(?:own\s+)?trades\b|\b(?:did|have)\s+you\b",
                   re.I)
_HYPOTHETICAL = re.compile(r"\b(?:if|should|would|could|shall|will|gonna|going\s+to|want\s+to|"
                           r"planning|plan\s+to|thinking\s+of)\b|如果|要是|假如|想要|打算|应该|要不要",
                           re.I)
_CJK = re.compile(r"[一-鿿]")
_CLAUSE_SPLIT = re.compile(r"[,，;；。\n]+|\s+(?:then|and\s+then|after\s+that|later)\s+|然后|之后|"  # noqa: RUF001
                           r"后来|接着", re.I)
_CRYPTO_LOWER = frozenset({"BTC", "ETH", "SOL", "XRP", "DOGE", "BNB", "ADA", "AVAX", "LTC", "SUI"})
_NOT_SYMBOL = frozenset({"AT", "ON", "IN", "TO", "FOR", "THE", "AND", "OUT", "BACK", "STOP", "LONG",
                         "SHORT", "BUY", "SELL", "SOLD", "EXIT", "ENTRY", "FROM", "USDT", "USD",
                         "I", "A", "MY", "IT", "ER", "PNL", "ME", "US", "NOW", "OK", "AM", "PM"})


def _number(text: str, k: str | None = None) -> float:
    value = float(text.replace(",", ""))
    return value * 1000.0 if k else value


def _resolve_word(word: str) -> str | None:
    """A word in a trade statement to a listed contract, by the console's own reader: the twelve
    stock perpetuals and their company names in any case, any other ticker in capitals, a full
    ``...USDT`` symbol, and the common coins in lower case (a journal line carries a price, which is
    the market cue `research._LOWERCASE_CRYPTO` asks for)."""
    from argus.lui import research
    from argus.market import universe

    token = word.strip().lstrip("$").rstrip(".")
    if not token:
        return None
    if _CJK.search(token):
        for name, symbol in universe.CJK_ALIASES.items():
            if name in token:
                return symbol
        return None
    if token.upper() in _NOT_SYMBOL:
        return None
    upper = token.upper()
    if not token.isupper() and (upper in _CRYPTO_LOWER or upper in research._LOWERCASE_CRYPTO):
        found = universe.resolve(upper)
        return found[0] if found else None
    try:
        found = research._resolve(token)
    except Exception:
        return None
    return found[0] if found else None


def _symbols_in(text: str) -> list[tuple[int, str]]:
    """Every instrument named in ``text`` with its position, in order."""
    from argus.market import universe

    out: list[tuple[int, str]] = []
    for m in re.finditer(r"\$?[A-Za-z][A-Za-z0-9.]{0,14}", text):
        symbol = _resolve_word(m.group(0))
        if symbol:
            out.append((m.start(), symbol))
    for name, symbol in universe.CJK_ALIASES.items():
        for m in re.finditer(re.escape(name), text):
            out.append((m.start(), symbol))
    return sorted(out)


def _date_from(y: str | None, m: int, d: int, now: date, notes: set[str]) -> date | None:
    try:
        if y:
            return date(int(y), m, d)
        found = date(now.year, m, d)
    except ValueError:
        return None
    if found > now:
        found = date(now.year - 1, m, d)
    notes.add("dates without a year were read as the most recent such date")
    return found


def _find_date(text: str, now: date, notes: set[str]) -> tuple[date | None, str]:
    """The first date in ``text`` and the text with every date removed (so "2026-08-20" is never
    read as three prices)."""
    found: date | None = None
    for pattern in (_DATE_ISO, _DATE_CJK, _DATE_MON_D, _DATE_D_MON, _DATE_SLASH):
        for m in pattern.finditer(text):
            if found is None:
                if pattern is _DATE_ISO or pattern is _DATE_CJK:
                    found = _date_from(m.group(1), int(m.group(2)), int(m.group(3)), now, notes)
                elif pattern is _DATE_MON_D:
                    found = _date_from(m.group(3), _MONTHS[m.group(1)[:3].lower()],
                                       int(m.group(2)), now, notes)
                elif pattern is _DATE_D_MON:
                    found = _date_from(m.group(3), _MONTHS[m.group(2)[:3].lower()],
                                       int(m.group(1)), now, notes)
                else:
                    found = _date_from(None, int(m.group(1)), int(m.group(2)), now, notes)
                    if found is not None:
                        notes.add("a date written 8/20 was read month first (US order)")
        text = pattern.sub(" ", text)
    return found, text


def _leg_from(action: str, window: str, context_symbol: str | None, now: date,
              notes: set[str]) -> Leg:
    day, rest = _find_date(window, now, notes)
    symbols = _symbols_in(rest)
    symbol = symbols[0][1] if symbols else context_symbol
    qty: float | None = None
    spans: list[tuple[int, int]] = []
    for pattern in (_QTY, _QTY_COIN):
        m = pattern.search(rest)
        if m is not None:
            qty = _number(m.group(1), m.group(2) if pattern is _QTY else None)
            spans.append(m.span())
            break
    if qty is None and symbols:
        # "bought 10 NVDA at 188": a number written straight before the name is a size
        before = re.search(rf"{_NUM}\s*(k)?\s+(?:of\s+)?\$?$", rest[: symbols[0][0]])
        if before is not None and _PRICE_AT.search(rest[symbols[0][0]:]):
            qty = _number(before.group(1), before.group(2))
            spans.append(before.span())
    masked = rest
    for a, b in spans:
        masked = masked[:a] + " " * (b - a) + masked[b:]
    price: float | None = None
    m = _PRICE_AT.search(masked) or _PRICE_BEFORE_CJK_VERB.search(masked)
    if m is not None:
        price = _number(m.group(1), m.group(2) if m.re is _PRICE_AT else None)
    else:
        bare = [b for b in _BARE_NUMBER.finditer(masked)]
        if bare:
            price = _number(bare[0].group(1), bare[0].group(2))
    flags = {name for name, pattern in _FLAGS if pattern.search(window)}
    if action == "close" and "stopped" in flags:
        notes.add("a stop that was hit was read as closing the position")
    return Leg(action=action, symbol=symbol, price=price, qty=qty, day=day, flags=flags,
               text=window.strip())


def parse_text(text: str, *, now: date | None = None) -> tuple[list[Leg], list[str]]:
    """Trade statements in English or Chinese, in the order written, and the assumptions made.

    A statement is a past-tense trade verb (bought, sold, shorted, covered, stopped out, 买入,
    卖出, 做空, 止损 ...) with its price, and optionally a size, a date and a note ("held over
    earnings", "FOMO"). English puts a verb before what it governs and Chinese usually after
    ("188买入英伟达"), so each clause is cut at its verbs in the matching direction. A clause
    with no verb ("held it over the weekend") belongs to the statement before it.
    """
    today = now or datetime.now(UTC).date()
    notes: set[str] = set()
    legs: list[Leg] = []
    text = text[:MAX_TEXT]
    # whole round trips written as one phrase first, then blanked so their verbs are not re-read
    for m in _ROUND_TRIP.finditer(text):
        symbol = None
        for raw in (m.group("sym1"), m.group("sym2")):
            if raw and symbol is None:
                symbol = _resolve_word(raw)
        if symbol is None:
            continue
        short = m.group("side").lower() in ("short", "做空")
        tail = text[m.end(): m.end() + 80]
        flags = {name for name, pattern in _FLAGS if pattern.search(m.group(0) + tail)}
        legs.append(Leg("short" if short else "buy", symbol, _number(m.group("p1")), flags=flags,
                        text=m.group(0)))
        legs.append(Leg("cover" if short else "close", symbol, _number(m.group("p2")), flags=set(),
                        text=m.group(0)))
        text = text[: m.start()] + " " * (m.end() - m.start()) + text[m.end():]
    context: str | None = legs[-1].symbol if legs else None
    for clause in _CLAUSE_SPLIT.split(text):
        if not clause.strip():
            continue
        verbs = list(_VERB.finditer(clause))
        if not verbs:
            found = _symbols_in(clause)
            if found:
                context = found[-1][1]
            if legs and clause.strip():
                extra = {name for name, pattern in _FLAGS if pattern.search(clause)}
                legs[-1].flags |= extra
                if legs[-1].day is None:
                    day, _ = _find_date(clause, today, notes)
                    legs[-1].day = day
            continue
        cjk_first = bool(_CJK.search(clause)) and bool(
            re.search(r"\d", clause[: verbs[0].start()]))
        windows: list[tuple[str, str]] = []
        for i, verb in enumerate(verbs):
            action = verb.lastgroup or "buy"
            if cjk_first:
                start = 0 if i == 0 else verbs[i - 1].end()
                end = verb.end() if i < len(verbs) - 1 else len(clause)
            else:
                start = 0 if i == 0 else verb.start()
                end = verbs[i + 1].start() if i < len(verbs) - 1 else len(clause)
            windows.append((action, clause[start:end]))
        for action, window in windows:
            leg = _leg_from(action, window, context, today, notes)
            if leg.symbol:
                context = leg.symbol
            legs.append(leg)
            if len(legs) >= MAX_LEGS:
                break
    return legs, sorted(notes)


# --- reading a pasted fill file -------------------------------------------------------------------

_COLUMNS: dict[str, tuple[str, ...]] = {
    "symbol": ("symbol", "pair", "tradingpair", "contract", "instrument", "futures", "market",
               "coin", "交易对", "币对", "合约", "币种", "标的"),
    "side": ("side", "direction", "type", "tradetype", "bs", "buysell", "方向", "买卖方向",
             "交易方向", "买卖"),
    "trade_side": ("tradeside", "posside", "positionside", "开平", "开平方向"),
    "price": ("execprice", "price", "priceavg", "avgprice", "fillprice", "averageprice",
              "dealprice", "filledprice", "成交价", "成交均价", "价格", "均价"),
    "qty": ("execqty", "basevolume", "size", "qty", "quantity", "filled", "filledamount",
            "amount", "volume", "数量", "成交数量", "成交量"),
    "value": ("execvalue", "quotevolume", "value", "total", "turnover", "成交额", "成交金额"),
    "time": ("createdtime", "ctime", "updatedtime", "utime", "time", "date", "datetime",
             "timestamp", "filltime", "tradetime", "dateutc", "timeutc", "时间", "成交时间",
             "日期"),
    "fee": ("fee", "fees", "totalfee", "feedetail", "commission", "手续费"),
    "pnl": ("execpnl", "profit", "realizedpnl", "realisedpnl", "closedpnl", "pnl", "已实现盈亏",
            "收益", "盈亏"),
}
"""Header names read, normalised (lower case, letters and CJK only). The first entry of each is the
Bitget v3 ``FillV3`` field and the second the v2 ``FuturesOrderFillV2`` one where they differ
(`tiagosiebler/bitget-api` ``src/types/response/v3/trade.ts:83-99``,
``src/types/response/v2/futures.ts:414-433``); the rest are spreadsheet names, NOT VERIFIED against
Bitget's website export."""


def _norm(header: str) -> str:
    return re.sub(r"[^a-z一-鿿]", "", header.lower())


def _header_map(cells: Sequence[str]) -> dict[str, int]:
    found: dict[str, int] = {}
    for index, cell in enumerate(cells):
        key = _norm(cell)
        for name, aliases in _COLUMNS.items():
            if name not in found and key in aliases:
                found[name] = index
                break
    return found


def _csv_time(raw: str) -> datetime | None:
    raw = raw.strip()
    if re.fullmatch(r"\d{13}", raw):
        return datetime.fromtimestamp(int(raw) / 1000.0, tz=UTC)
    if re.fullmatch(r"\d{10}", raw):
        return datetime.fromtimestamp(int(raw), tz=UTC)
    cleaned = re.sub(r"\s*\((?:UTC)?[+-]?\d*\)|\s*UTC[+-]?\d*$", "", raw).replace("/", "-")
    try:
        found = datetime.fromisoformat(cleaned.replace("Z", "+00:00"))
    except ValueError:
        return None
    return found if found.tzinfo else found.replace(tzinfo=UTC)


def _csv_action(side: str, trade_side: str) -> tuple[str, str | None] | None:
    """A fill's side words to an action and whether it opens or closes. Bitget's API sends ``side``
    buy/sell with ``tradeSide`` open/close; a spreadsheet often says "Open long" or "平空"."""
    text = f"{side} {trade_side}".lower()
    open_close: str | None = None
    if re.search(r"\bclose\b|\breduce\b|平", text):
        open_close = "close"
    elif re.search(r"\bopen\b|开", text):
        open_close = "open"
    if re.search(r"\bshort\b|空", text) and not re.search(r"\bbuy\b|买", side.lower()):
        return ("cover" if open_close == "close" else "short"), open_close
    if re.search(r"\blong\b|多", text) and open_close == "close":
        return "close", open_close
    if re.search(r"\bbuy\b|买|\blong\b|多", text):
        return "buy", open_close
    if re.search(r"\bsell\b|卖", text):
        return "sell", open_close
    return None


def _fee_value(raw: str) -> float | None:
    raw = raw.strip()
    if not raw:
        return None
    if raw.startswith("["):
        try:
            rows = json.loads(raw)
        except ValueError:
            return None
        total = 0.0
        for row in rows if isinstance(rows, list) else []:
            if isinstance(row, dict) and str(row.get("feeCoin", "USDT")).upper() in (
                    "USDT", "USDC", "USD"):
                total += abs(float(row.get("fee") or 0.0))
        return total
    try:
        return abs(_number(re.sub(r"[^\d.,-]", "", raw).lstrip("-")))
    except ValueError:
        return None


def parse_fills(text: str) -> tuple[list[Leg], list[str]] | None:
    """A pasted table of fills (comma, tab or semicolon separated), oldest first, or None when the
    text holds no such table. A table is a header line naming at least a side and a price, followed
    by rows."""
    lines = [line for line in text[:MAX_TEXT].splitlines() if line.strip()]
    for start, line in enumerate(lines):
        delimiter = "\t" if "\t" in line else ";" if line.count(";") >= 2 else ","
        cells = next(csv.reader([line], delimiter=delimiter))
        columns = _header_map(cells)
        if not {"side", "price"} <= columns.keys() or len(columns) < 3:
            continue
        legs: list[Leg] = []
        notes = ["read a fill table with columns: "
                 + ", ".join(cells[i].strip() for i in sorted(columns.values()))]
        for row in csv.reader(lines[start + 1:], delimiter=delimiter):
            if len(row) <= max(columns.values()):
                continue

            def cell(name: str, row: list[str] = row,
                     columns: dict[str, int] = columns) -> str:
                return row[columns[name]].strip() if name in columns else ""

            action = _csv_action(cell("side"), cell("trade_side"))
            price_text = re.sub(r"[^\d.,]", "", cell("price"))
            if action is None or not price_text:
                continue
            symbol = _resolve_word(cell("symbol")) if cell("symbol") else None
            qty_text = re.sub(r"[^\d.,]", "", cell("qty"))
            at = _csv_time(cell("time")) if cell("time") else None
            pnl_text = re.sub(r"[^\d.,-]", "", cell("pnl"))
            try:
                legs.append(Leg(
                    action=action[0], open_close=action[1], symbol=symbol,
                    price=_number(price_text), qty=_number(qty_text) if qty_text else None,
                    at=at, day=None if at is None else at.date(), fee=_fee_value(cell("fee")),
                    venue_pnl=float(pnl_text) if pnl_text not in ("", "-", ".") else None,
                    text=",".join(row)))
            except ValueError:
                continue
            if len(legs) >= MAX_LEGS:
                break
        if not legs:
            return None
        if all(leg.at is not None for leg in legs):
            legs.sort(key=lambda leg: leg.at or datetime.min.replace(tzinfo=UTC))
        return legs, notes
    return None


# --- legs into round trips ------------------------------------------------------------------------


@dataclass
class _Lot:
    price: float
    qty: float | None
    day: date | None
    at: datetime | None
    fee: float | None
    flags: set[str]


def pair(legs: Sequence[Leg]) -> tuple[list[Trade], list[str], list[str]]:
    """Round trips from legs, first in first out per symbol (vibe-trading's ``pair_trades_fifo``),
    the positions still open, and what could not be placed.

    A typed "sold" with no size closes the whole position; a fill with a size closes that much, the
    oldest lots first. Several entries into one position are averaged (by size when every lot has
    one, equally otherwise) and a position whose later entry was worse than its first is marked as
    averaged down."""
    book: dict[str, tuple[str, list[_Lot]]] = {}
    trades: list[Trade] = []
    dropped: list[str] = []
    order = 0
    for leg in legs:
        if leg.price is None or leg.price <= 0:
            dropped.append(f"{leg.text[:60]!r} carries no price")
            continue
        symbol = leg.symbol
        if symbol is None:
            open_names = [s for s, (_, lots) in book.items() if lots]
            symbol = open_names[-1] if len(open_names) == 1 else None
        if symbol is None:
            dropped.append(f"{leg.text[:60]!r} names no instrument")
            continue
        side, lots = book.get(symbol, ("", []))
        lot = _Lot(leg.price, leg.qty, leg.day, leg.at, leg.fee, set(leg.flags))
        closing = (
            (leg.open_close == "close" and bool(lots))
            or leg.action == "close"
            or (leg.action == "cover" and side == "short")
            or (leg.action == "buy" and side == "short" and bool(lots))
            or (leg.action == "sell" and side == "long" and bool(lots))
            or (leg.action == "sell" and side == "short" and bool(lots)
                and leg.open_close is None and leg.qty is None)
        )
        if closing and lots:
            want = leg.qty
            taken: list[_Lot] = []
            if want is None or any(x.qty is None for x in lots):
                taken, lots = lots, []
            else:
                remaining = want
                keep: list[_Lot] = []
                for x in lots:
                    q = x.qty or 0.0
                    if remaining <= 1e-12:
                        keep.append(x)
                    elif q <= remaining + 1e-12:
                        taken.append(x)
                        remaining -= q
                    else:
                        taken.append(_Lot(x.price, remaining, x.day, x.at,
                                          None if x.fee is None else x.fee * remaining / q,
                                          x.flags))
                        keep.append(_Lot(x.price, q - remaining, x.day, x.at,
                                         None if x.fee is None else x.fee * (q - remaining) / q,
                                         x.flags))
                        remaining = 0.0
                lots = keep
            sized = all(x.qty is not None for x in taken)
            weights = [x.qty or 0.0 for x in taken] if sized else [1.0] * len(taken)
            total = sum(weights) or 1.0
            entry = sum(x.price * w for x, w in zip(taken, weights, strict=True)) / total
            worse = any((x.price < taken[0].price) if side == "long" else (x.price > taken[0].price)
                        for x in taken[1:])
            fees = [x.fee for x in taken] + [leg.fee]
            order += 1
            trades.append(Trade(
                symbol=symbol, side=side, entry=entry, exit=leg.price,
                opened=taken[0].day, closed=leg.day, qty=total if sized else None,
                lots=len(taken), averaged_down=worse,
                flags=frozenset(set().union(*(x.flags for x in taken)) | leg.flags),
                fees=None if all(f is None for f in fees) else sum(f or 0.0 for f in fees),
                venue_pnl=leg.venue_pnl, opened_at=taken[0].at, closed_at=leg.at, order=order))
            book[symbol] = (side if lots else "", lots)
            continue
        if leg.action in ("cover", "close"):
            dropped.append(f"{leg.text[:60]!r} closes a position that was never opened")
            continue
        new_side = "short" if leg.action in ("short", "sell") else "long"
        if lots and side != new_side:
            dropped.append(f"{leg.text[:60]!r} flips an open position; read as a new one")
            lots = []
        book[symbol] = (new_side, [*lots, lot])
    still_open = [f"{_short(s)} {side} from {lots[0].price:g}" for s, (side, lots) in book.items()
                  if lots]
    return trades, still_open, dropped


def _short(symbol: str) -> str:
    from argus.lui.research import _t

    return _t(symbol)


# --- market facts per trade -----------------------------------------------------------------------

History = tuple[list[tuple[date, float]], str]
"""Daily closes (oldest first) and whose they are."""
Releases = tuple[list[datetime], str]
"""Earnings release times (UTC) and where they were read."""


def _is_equity(symbol: str) -> bool:
    from argus.lui.research import _is_equity as equity

    try:
        return bool(equity(symbol))
    except Exception:
        from argus.lui.question import TRADED_SYMBOLS

        return symbol in TRADED_SYMBOLS


def daily_history(symbol: str) -> History:
    """The closes `research._daily_closes` would read (`research.py:6900-6919`), with their dates:
    a stock's split-adjusted closes from Yahoo, else Bitget's daily candles."""
    if _is_equity(symbol):
        from argus.market import equity_history

        try:
            days = equity_history.daily(_short(symbol))
            if len(days) >= 30:
                return ([(d.day, d.close) for d in days],
                        f"{_short(symbol)}'s split-adjusted daily closes (Yahoo)")
        except Exception:
            pass
    from argus.market.history import CandleType, fetch_window

    bars = fetch_window(symbol, start=datetime.now(UTC) - timedelta(days=HISTORY_SESSIONS + 400),
                        interval="1D", candle_type=CandleType.MARKET, pause=0.05)
    return ([(b.ts.date(), float(b.close)) for b in bars if float(b.close) > 0],
            f"{_short(symbol)} Bitget daily candles")


def earnings_releases(symbol: str, since: date) -> Releases:
    """Every 8-K item 2.02 acceptance time on EDGAR since ``since`` (the issuer's own record, as
    `research/event_reactions.earnings_releases` reads it), for a stock only."""
    from argus.market.evidence import EdgarSource

    start = datetime.combine(since - timedelta(days=5), clock_time(0), tzinfo=UTC)
    found = [f.accepted for f in EdgarSource().filings(_short(symbol), since=start, limit=400)
             if f.form == "8-K" and "2.02" in f.items]
    return sorted(found), f"SEC EDGAR 8-K item 2.02 for {_short(symbol)}"


def _holidays() -> frozenset[date] | None:
    try:
        from argus.eval.baselines.lean_market_holidays_loader import load_usa_equity_holidays

        return load_usa_equity_holidays()
    except Exception:
        return None


def _trading_day(d: date, holidays: frozenset[date]) -> bool:
    return d.weekday() < 5 and d not in holidays


def reaction_day(accepted: datetime, holidays: frozenset[date]) -> date:
    """The first US session that could price a release: the same day for one before 09:30 New York
    on a trading day or during the session, the next trading day for one after the close."""
    local = accepted.astimezone(NEW_YORK)
    d = local.date()
    if _trading_day(d, holidays) and local.time() < clock_time(16, 0):
        return d
    d += timedelta(days=1)
    while not _trading_day(d, holidays):
        d += timedelta(days=1)
    return d


def _band(closes: list[tuple[date, float]], sessions: int) -> tuple[float, float] | None:
    """The middle 80% of the name's ``sessions``-long moves, in percent (`desk/odds.py`)."""
    from argus.desk.odds import directional_odds

    stamped = [(datetime.combine(d, clock_time(0), tzinfo=UTC), c) for d, c in closes]
    odds = directional_odds(stamped, sessions, cost_bps=0.0)
    if odds is None or odds.windows < 30:
        return None
    return odds.p10_bps / 100.0, odds.p90_bps / 100.0


def check_trade(trade: Trade, history: History | None, releases: Releases | None,
                holidays: frozenset[date] | None) -> Check:
    """Everything that can be computed for one trade from its prices, its dates and the name's own
    history; what cannot is left None."""
    out = Check(trade=trade)
    equity = _is_equity(trade.symbol)
    dated = trade.opened is not None and trade.closed is not None
    if dated and holidays is not None and equity:
        assert trade.opened is not None and trade.closed is not None
        span = [trade.opened + timedelta(days=k)
                for k in range(1, (trade.closed - trade.opened).days + 1)]
        out.closures = sum(1 for d in span if not _trading_day(d, holidays))
    if dated and releases is not None and equity and holidays is not None:
        assert trade.opened is not None and trade.closed is not None
        days = [reaction_day(at, holidays) for at in releases[0]]
        crossed = [d for d in days if trade.opened < d <= trade.closed]
        out.earnings = bool(crossed)
        out.earnings_day = crossed[0] if crossed else None
    if history is None or len(history[0]) < 60:
        return out
    closes = history[0]
    if trade.opened is not None:
        before = [c for c in closes if c[0] < trade.opened]
    else:
        before = list(closes)
    before = before[-HISTORY_SESSIONS:]
    if dated:
        assert trade.opened is not None and trade.closed is not None
        opened, closed = trade.opened, trade.closed
        out.sessions = max(1, sum(1 for d, _ in closes if opened < d <= closed))
        band = _band(before, out.sessions)
        if band is not None:
            out.noise_sessions, out.band = out.sessions, band
            out.inside_noise = band[0] <= trade.price_move_pct <= band[1]
    else:
        for sessions in NOISE_HORIZONS:
            band = _band(before, sessions)
            if band is None:
                continue
            out.noise_sessions, out.band = sessions, band
            out.inside_noise = band[0] <= trade.price_move_pct <= band[1]
            if out.inside_noise:
                break
    if trade.opened is not None and len(before) >= RANK_SESSIONS // 2:
        moves = [before[i][1] / before[i - 1][1] - 1.0 for i in range(1, len(before))]
        last = moves[-1]
        past = moves[-RANK_SESSIONS - 1:-1]
        if past:
            if trade.side == "long":
                out.prior_rank = sum(1 for m in past if m >= last) / len(past)
            else:
                out.prior_rank = sum(1 for m in past if m <= last) / len(past)
            out.prior_move_pct = 100.0 * last
            out.chased = out.prior_rank <= LARGE_DAY_SHARE
    return out


# --- patterns -------------------------------------------------------------------------------------


def _pct(value: float) -> str:
    return f"{value:+.1f}%"


def _names(checks: Sequence[Check], members: Sequence[int]) -> str:
    shown = [f"{_short(checks[i].trade.symbol)} {_pct(checks[i].trade.return_pct)}"
             for i in members[:4]]
    return ", ".join(shown) + (f" and {len(members) - 4} more" if len(members) > 4 else "")


def find_patterns(checks: Sequence[Check]) -> list[Pattern]:
    """Each habit tested on the trades it can be tested on, with its count and its cost. Ranked by
    cost: the percentage points of summed trade return lost on the trades that show it."""
    out: list[Pattern] = []
    rets = [c.trade.return_pct for c in checks]
    wins = [i for i, r in enumerate(rets) if r > 0]
    losses = [i for i, r in enumerate(rets) if r < 0]
    n = len(checks)

    if wins and losses:
        avg_win = sum(rets[i] for i in wins) / len(wins)
        avg_loss = -sum(rets[i] for i in losses) / len(losses)
        if avg_loss >= DISPOSITION_RATIO * avg_win:
            big = tuple(i for i in losses if -rets[i] > avg_win)
            cost = sum(-rets[i] - avg_win for i in big)
            out.append(Pattern(
                "losers_bigger", "losses larger than wins", big, len(losses), cost,
                f"your average loss ({avg_loss:.1f}%) is {avg_loss / avg_win:.1f}x your average "
                f"win ({avg_win:.1f}%); {len(big)} of {len(losses)} losses ran past the average "
                f"win ({_names(checks, big)})",
                f"Set the exit that caps a loss at your typical win (about {avg_win:.1f}% here) "
                f"before entering, and close when it is hit rather than waiting for a recovery."))
    elif losses and not wins and len(losses) >= 2:
        cost = -sum(rets[i] for i in losses)
        out.append(Pattern(
            "all_losers", "every trade lost", tuple(losses), n, cost,
            f"all {len(losses)} trades lost ({_names(checks, losses)})",
            "Before the next entry, write down the price that would prove the idea wrong and "
            "place the exit there; no winner here offsets a loss."))

    held = [(i, c) for i, c in enumerate(checks) if c.trade.opened and c.trade.closed]
    win_hold = [(c.trade.closed - c.trade.opened).days for i, c in held  # type: ignore[operator]
                if rets[i] > 0]
    loss_hold = [(c.trade.closed - c.trade.opened).days for i, c in held  # type: ignore[operator]
                 if rets[i] < 0]
    if win_hold and loss_hold:
        mean_win = sum(win_hold) / len(win_hold)
        mean_loss = sum(loss_hold) / len(loss_hold)
        if mean_loss >= DISPOSITION_RATIO * max(mean_win, 0.5):
            slow = tuple(i for i, c in held if rets[i] < 0
                         and (c.trade.closed - c.trade.opened).days > mean_win)  # type: ignore[operator]
            out.append(Pattern(
                "held_losers_longer", "holding losers longer than winners", slow, len(loss_hold),
                -sum(rets[i] for i in slow),
                f"losers were held {mean_loss:.1f} days on average and winners {mean_win:.1f} "
                f"(the disposition effect); {len(slow)} loser(s) outlasted the average winner",
                f"Give a losing trade no more time than a winning one: if it has not worked in "
                f"about {max(mean_win, 1.0):.0f} day(s), close it."))

    earn = tuple(i for i, c in enumerate(checks)
                 if c.earnings or (c.earnings is None and "earnings" in c.trade.flags))
    earn_lost = tuple(i for i in earn if rets[i] < 0)
    earn_of = sum(1 for c in checks if c.earnings is not None or "earnings" in c.trade.flags)
    if earn_lost:
        out.append(Pattern(
            "earnings_losers", "holding through earnings", earn_lost, earn_of,
            -sum(rets[i] for i in earn_lost),
            f"{len(earn)} of {earn_of} checkable trades were held through an earnings release "
            f"and {len(earn_lost)} of those lost ({_names(checks, earn_lost)})",
            "Check the next earnings date (SEC 8-K / the company calendar) before entering; "
            "decide in advance whether to hold through it, at a size you can take a gap on."))

    closure = tuple(i for i, c in enumerate(checks)
                    if (c.closures or 0) > 0 or (c.closures is None and "weekend" in c.trade.flags))
    closure_lost = tuple(i for i in closure if rets[i] < 0)
    closure_of = sum(1 for c in checks if c.closures is not None or "weekend" in c.trade.flags)
    if closure_lost:
        out.append(Pattern(
            "closure_losers", "holding through a closed US session", closure_lost, closure_of,
            -sum(rets[i] for i in closure_lost),
            f"{len(closure)} of {closure_of} checkable stock trades were held across a weekend "
            f"or US holiday, when the stock does not trade but the perpetual does, and "
            f"{len(closure_lost)} of those lost ({_names(checks, closure_lost)})",
            "Before a Friday or a holiday, decide whether the position is worth a gap with no "
            "stock market to anchor it; cut it to a size that survives the gap if it is."))

    chased = tuple(i for i, c in enumerate(checks)
                   if c.chased or (c.chased is None and "chased" in c.trade.flags))
    chased_of = sum(1 for c in checks if c.chased is not None or "chased" in c.trade.flags)
    chased_lost = tuple(i for i in chased if rets[i] < 0)
    if chased_lost:
        out.append(Pattern(
            "chasing", "entering after a large move", chased_lost, chased_of,
            -sum(rets[i] for i in chased_lost),
            f"{len(chased)} of {chased_of} checkable entries came the session after one of the "
            f"name's top-10% days in the trade's direction (or were called FOMO), and "
            f"{len(chased_lost)} lost ({_names(checks, chased_lost)})",
            "After a name's biggest day of the month, wait one session before entering, or "
            "enter at half size."))

    tight = tuple(i for i, c in enumerate(checks)
                  if "stopped" in c.trade.flags and rets[i] < 0 and c.inside_noise)
    stopped_of = sum(1 for c in checks if "stopped" in c.trade.flags and c.inside_noise is not None)
    if tight:
        out.append(Pattern(
            "stop_inside_noise", "stops inside ordinary noise", tight, stopped_of,
            -sum(rets[i] for i in tight),
            f"{len(tight)} of {stopped_of} stop-outs were moves the name makes routinely over "
            f"that many sessions ({_names(checks, tight)})",
            "Place a stop outside the name's ordinary range for the time you plan to hold, and "
            "size down so that stop is affordable."))

    beyond = tuple(i for i, c in enumerate(checks)
                   if rets[i] < 0 and c.inside_noise is False and "stopped" not in c.trade.flags
                   and c.band is not None)
    band_of = sum(1 for i, c in enumerate(checks) if rets[i] < 0 and c.band is not None)
    if beyond:
        out.append(Pattern(
            "loss_beyond_noise", "losses past the ordinary range", beyond, band_of,
            -sum(rets[i] for i in beyond),
            f"{len(beyond)} of {band_of} measured losses went past the name's {BAND} range for "
            f"the holding period — moves it rarely makes — with no stop recorded "
            f"({_names(checks, beyond)})",
            "Have an exit at the edge of the name's ordinary range for your holding period; a "
            "move past it means the idea was wrong, not noisy."))

    averaged = tuple(i for i, c in enumerate(checks) if c.trade.averaged_down and rets[i] < 0)
    if averaged:
        out.append(Pattern(
            "averaged_down", "adding to losing positions", averaged,
            sum(1 for c in checks if c.trade.lots > 1), -sum(rets[i] for i in averaged),
            f"{len(averaged)} losing positions were added to at a worse price "
            f"({_names(checks, averaged)})",
            "Do not add to a position that is below its entry; add only when it is working."))

    ordered = sorted(range(n), key=lambda i: checks[i].trade.order)
    sized = [i for i in ordered if checks[i].trade.notional is not None]
    up_after_loss = tuple(b for a, b in itertools.pairwise(sized)
                          if rets[a] < 0 and (checks[b].trade.notional or 0.0)
                          > 1.2 * (checks[a].trade.notional or 0.0))
    if up_after_loss and any(rets[i] < 0 for i in up_after_loss):
        out.append(Pattern(
            "size_after_loss", "sizing up after a loss", up_after_loss,
            sum(1 for a in sized[:-1] if rets[a] < 0),
            -sum(rets[i] for i in up_after_loss if rets[i] < 0),
            f"{len(up_after_loss)} trades were more than 1.2x the size of the losing trade "
            f"before them ({_names(checks, up_after_loss)})",
            "After a loss, keep the next trade at or below your usual size."))

    by_name: dict[str, list[int]] = {}
    for i, c in enumerate(checks):
        by_name.setdefault(c.trade.symbol, []).append(i)
    for symbol, members in by_name.items():
        lost = [i for i in members if rets[i] < 0]
        if len(members) >= 3 and len(lost) * 3 >= 2 * len(members):
            out.append(Pattern(
                f"repeat_{symbol}", f"repeated losses in {_short(symbol)}", tuple(lost),
                len(members), -sum(rets[i] for i in lost),
                f"{len(lost)} of {len(members)} {_short(symbol)} trades lost",
                f"Before another {_short(symbol)} trade, write what is different from the last "
                f"{len(lost)} that lost."))
    return sorted(out, key=lambda p: (-p.cost, -len(p.members)))


# --- the answer -----------------------------------------------------------------------------------


@dataclass
class Journal:
    legs: list[Leg]
    notes: list[str]
    parsed_from: str


def read_journal(text: str, *, now: date | None = None) -> Journal | None:
    """The trades in a message when it is a journal to review, else None.

    A message is a journal when it holds a fill table, or two or more completed round trips, or one
    completed round trip and a request for review ("what did I do wrong", "复盘"). A message about
    the desk's own trades, an order still being planned ("if I buy at 180 and sell at 200"), or a
    position still open ("I bought at 188, should I sell?") is not.
    """
    if not text or not text.strip():
        return None
    table = parse_fills(text)
    if table is not None:
        trades, _, _ = pair(table[0])
        if trades or len(table[0]) >= 2:
            return Journal(table[0], table[1], "fills")
    if _DESK.search(text):
        return None
    if not _VERB.search(text) and not _ROUND_TRIP.search(text):
        return None
    legs, notes = parse_text(text, now=now)
    trades, _, _ = pair(legs)
    if not trades:
        return None
    if len(trades) == 1:
        if not _REVIEW_CUE.search(text):
            return None
        if _HYPOTHETICAL.search(text) and not re.search(r"\bwhat\s+did\s+i\b|复盘", text, re.I):
            return None
    return Journal(legs, notes, "text")


def review_requested(text: str) -> bool:
    """A request to review the trader's own trades that pastes none."""
    return bool(_REVIEW_REQUEST.search(text)) and not _DESK.search(text)


def _fetch_market(symbols: Sequence[str], earliest: date | None,
                  history: Callable[[str], History],
                  releases: Callable[[str, date], Releases] | None,
                  ) -> tuple[dict[str, History], dict[str, Releases], list[str]]:
    """Daily histories and earnings releases for every name, concurrently, under one deadline. A
    name that does not answer in time is named as missing, never silently skipped."""
    histories: dict[str, History] = {}
    found: dict[str, Releases] = {}
    failed: list[str] = []
    pool = ThreadPoolExecutor(max_workers=4)
    try:
        jobs: dict[Any, tuple[str, str]] = {}
        for s in symbols:
            jobs[pool.submit(history, s)] = ("history", s)
            if releases is not None and earliest is not None and _is_equity(s):
                jobs[pool.submit(releases, s, earliest)] = ("earnings", s)
        done, pending = wait(jobs, timeout=FETCH_DEADLINE_S)
        for future in done:
            kind, s = jobs[future]
            try:
                value = future.result()
            except Exception as exc:
                failed.append(f"{_short(s)} {kind} ({type(exc).__name__})")
                continue
            if kind == "history":
                histories[s] = value
            else:
                found[s] = value
        for future in pending:
            kind, s = jobs[future]
            failed.append(f"{_short(s)} {kind} (no answer in {FETCH_DEADLINE_S:.0f}s)")
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    return histories, found, sorted(failed)


@traced(declaration=every("computed"))
def trade_lines(checks: Sequence[Check]) -> list[str]:
    """One line per closed trade: its return on the trader's own prices, and each check that could
    be made."""
    out: list[str] = []
    for n, c in enumerate(checks, start=1):
        t = c.trade
        when = ""
        if t.opened and t.closed:
            when = f" ({t.opened:%Y-%m-%d} to {t.closed:%Y-%m-%d})"
        lots = f", {t.lots} entries averaged" if t.lots > 1 else ""
        parts = [f"Trade {n}: {_short(t.symbol)} {t.side} {t.entry:g} to {t.exit:g}{when}{lots}: "
                 f"{_pct(t.return_pct)}"]
        if t.fees:
            parts.append(f"fees {t.fees:.2f}")
        if c.earnings is True and c.earnings_day is not None:
            parts.append(f"held through earnings (reaction session {c.earnings_day:%Y-%m-%d})")
        elif c.earnings is False:
            parts.append("no earnings release inside the hold")
        if c.closures:
            parts.append(f"held across {c.closures} weekend/holiday day(s) with the US market shut")
        if c.band is not None and c.noise_sessions is not None:
            where = "inside" if c.inside_noise else "outside"
            parts.append(f"a {_pct(t.price_move_pct)} price move is {where} {_short(t.symbol)}'s "
                         f"ordinary {c.noise_sessions}-session range "
                         f"({_pct(c.band[0])} to {_pct(c.band[1])}, {BAND})")
        if c.prior_move_pct is not None and c.prior_rank is not None:
            tail = "top" if t.side == "long" else "bottom"
            share = max(1, round(100 * c.prior_rank))
            parts.append(f"entered after a {_pct(c.prior_move_pct)} session"
                         + (f" ({tail} {share}% of its past year)" if c.chased else ""))
        stated = sorted(t.flags & {"stopped", "weekend", "earnings", "chased", "revenge"})
        if stated:
            parts.append("you noted: " + ", ".join(stated))
        out.append("; ".join(parts) + ".")
    return out


@traced(declaration=every("computed"))
def summary_lines(checks: Sequence[Check], patterns: Sequence[Pattern]) -> list[str]:
    """The lead (the one habit most worth fixing, with its count), the totals and the patterns."""
    rets = [c.trade.return_pct for c in checks]
    wins = sum(1 for r in rets if r > 0)
    losses = sum(1 for r in rets if r < 0)
    out: list[str] = []
    if patterns:
        top = patterns[0]
        out.append(f"Actionable: the habit most worth fixing is {top.name} — {len(top.members)} "
                   f"of {top.of} trades, costing {top.cost:.1f} percentage points of summed trade "
                   f"return. {top.check}")
    else:
        worst = min(range(len(rets)), key=lambda i: rets[i])
        out.append(f"Actionable: none of the checked habits shows up in these {len(rets)} trades; "
                   f"the largest loss was {_short(checks[worst].trade.symbol)} "
                   f"{_pct(rets[worst])}, so review that one entry and exit by hand.")
    total = sum(rets)
    out.append(f"{len(rets)} closed trade(s): {wins} won, {losses} lost, summed return "
               f"{_pct(total)} (each trade's % on its own entry price, equal-weighted).")
    dollars = [c.trade for c in checks if c.trade.qty is not None]
    if dollars and len(dollars) == len(checks):
        pnl = sum((t.exit - t.entry) * (t.qty or 0.0) * (1 if t.side == "long" else -1)
                  - (t.fees or 0.0) for t in dollars)
        out.append(f"On the sizes given, the trades made {pnl:+,.2f} in the quote currency"
                   + (" after the fees listed." if any(t.fees for t in dollars) else "."))
    for rank, p in enumerate(patterns, start=1):
        out.append(f"Pattern {rank} — {p.name} ({len(p.members)} of {p.of}, "
                   f"{p.cost:.1f} points): {p.finding}.")
    return out


@traced(declaration=every("computed"))
def checklist_lines(patterns: Sequence[Pattern]) -> list[str]:
    """The reusable checklist: one check per habit found, in the order of what it cost."""
    if not patterns:
        return []
    out = ["Checklist, from your own trades (run it before every entry):"]
    for n, p in enumerate(patterns, start=1):
        out.append(f"{n}. {p.check} (from {len(p.members)} of {p.of} trades)")
    return out


def gate_lines(trades: Sequence[Trade]) -> tuple[list[str], dict[str, Any] | None]:
    """The held-out rule gate on a large, fully timed journal (`eval/regression_gate`), or nothing.
    It is run unchanged and without a model; its refusal is reported rather than hidden."""
    timed = [t for t in trades if t.opened_at is not None and t.closed_at is not None]
    if len(timed) < GATE_MIN_TRADES:
        return [], None
    from argus.eval.regression_gate import admitted_checklist, journal_review

    rows = [json.dumps({
        "at": t.opened_at.isoformat() if t.opened_at else "", "symbol": _short(t.symbol),
        "side": t.side, "outcome_bps": round(t.return_pct * 100.0, 2),
        "holding_hours": round(((t.closed_at or t.opened_at or datetime.now(UTC))
                                - (t.opened_at or datetime.now(UTC))).total_seconds() / 3600.0,
                               3),
    }) for t in timed]
    report = journal_review(rows)
    if "refused" in report:
        refusal = f"Missing: the held-out rule gate refused this journal — {report['refused']}."
        return [refusal], report
    admitted = admitted_checklist(report)
    if not admitted:
        return ["Tested on the later half of your trades, held out: no rule written on the earlier "
                "half survived, so none is added to the checklist."], report
    lines = [f"Tested on the later half of your trades, held out: {len(admitted)} rule(s) from the "
             f"earlier half survived the gate."]
    for row in admitted[:5]:
        lines.append(f"Held-out rule ({row.get('status')}): {row.get('rule') or row.get('name')}; "
                     f"lift {row.get('lift')}.")
    return lines, report


def review_trades(text: str, *, now: datetime | None = None,
                  history: Callable[[str], History] | None = daily_history,
                  releases: Callable[[str, date], Releases] | None = earnings_releases,
                  ) -> tuple[list[str], list[Source], dict[str, Any]] | None:
    """Review the trader's own trades in ``text``: the one function the console calls.

    Returns None when the message is not a journal to review (so the caller routes it elsewhere),
    else ``(lines, sources, data)`` like the research engines. ``history`` and ``releases`` read the
    network by default; pass None (or a stub) to review from the trader's prices alone.
    """
    clock = now or datetime.now(UTC)
    journal = read_journal(text, now=clock.date())
    if journal is None:
        if not review_requested(text):
            return None
        return ([
            "Missing: no trades were found in the message, and the console does not read your "
            "Bitget account — paste them and the review runs on them.",
            "Paste one trade per clause, e.g. \"bought NVDA at 188 on 2026-08-20, sold at 176 "
            "on 2026-08-27; shorted TSLA at 250, covered at 270\" (中文也可以: \"8月20日188买入"
            "英伟达, 8月27日176卖出\"), or paste a fill export with columns such as symbol, "
            "side, execPrice, execQty, createdTime.",
        ], [Source("computation", "argus.lui.journal:review_trades", "no trades in the message")],
            {"journal": {"trades": [], "refused": "no trades pasted"}})
    trades, still_open, dropped = pair(journal.legs)
    lines: list[str] = []
    sources: list[Source] = [Source(
        "computation", "argus.lui.journal:review_trades",
        f"{len(trades)} round trip(s) read from {journal.parsed_from}")]
    if not trades:
        lines.append("Missing: the fills pasted do not close any position, so there is no "
                     "finished trade to review yet"
                     + (f" (open: {'; '.join(still_open[:6])})." if still_open else "."))
        return lines, sources, {"journal": {"trades": [], "open": still_open,
                                            "dropped": dropped}}

    symbols = sorted({t.symbol for t in trades})
    opened = [t.opened for t in trades if t.opened is not None]
    earliest = min(opened) if opened else None
    histories: dict[str, History] = {}
    found: dict[str, Releases] = {}
    failed: list[str] = []
    if history is not None:
        histories, found, failed = _fetch_market(symbols, earliest, history, releases)
    holidays = _holidays()
    checks = [check_trade(t, histories.get(t.symbol), found.get(t.symbol), holidays)
              for t in trades]
    patterns = find_patterns(checks)

    summary = summary_lines(checks, patterns)
    lines.append(summary[0])
    if len(trades) < MIN_STATISTICAL_TRADES:
        lines.append(f"Too few trades to call any of this a statistic: {len(trades)} closed "
                     f"trade(s), under {MIN_STATISTICAL_TRADES}, so each pattern below is an "
                     f"anecdote worth checking, not a measured habit.")
    lines.extend(summary[1:])
    lines.extend(checklist_lines(patterns))
    lines.extend(trade_lines(checks))
    gate, gate_report = gate_lines(trades)
    lines.extend(gate)

    assumed: list[str] = []
    if journal.parsed_from == "text":
        assumed.append("returns are before fees and funding, which a typed trade does not carry")
    if not all(t.qty is not None for t in trades):
        assumed.append("trades without a size count equally in the summed return")
    if not opened and histories:
        assumed.append("no dates were given, so each move is set against the shortest horizon "
                       "(1, 5 or 20 sessions) whose ordinary range contains it")
    assumed.extend(journal.notes if journal.parsed_from == "text" else [])
    if any(c.sessions == 1 and c.trade.opened == c.trade.closed for c in checks):
        assumed.append("a same-day trade is set against a full session's range, the finest the "
                       "daily history allows")
    for note in assumed:
        lines.append(f"Assumed: {note}.")
    if journal.parsed_from == "fills":
        lines.append(f"Data: {journal.notes[0]}.")
    missing: list[str] = []
    if not opened:
        missing.append("whether a trade crossed earnings, a weekend or a large up day was not "
                       "checked: no dates were given")
    elif holidays is None:
        missing.append("the US holiday calendar could not be read, so closed sessions were not "
                       "counted")
    undated = sum(1 for t in trades if t.opened is None or t.closed is None)
    if opened and undated:
        missing.append(f"{undated} trade(s) without both dates were not checked for earnings, "
                       f"closures or chasing")
    no_edgar = [s for s in symbols if _is_equity(s) and s not in found and releases is not None
                and earliest is not None]
    if no_edgar:
        missing.append("earnings releases could not be read for "
                       + ", ".join(_short(s) for s in no_edgar))
    if failed:
        missing.append("could not fetch " + "; ".join(failed))
    if still_open:
        missing.append("still open, not graded: " + "; ".join(still_open[:6]))
    if dropped:
        missing.append("not read as a trade: " + "; ".join(dropped[:4]))
    for note in missing:
        lines.append(f"Missing: {note}.")

    for s, (_, whose) in sorted(histories.items()):
        ref = (YAHOO_REF.format(ticker=_short(s)) if "Yahoo" in whose
               else "bitget /api/v3/market/candles 1D")
        sources.append(Source("venue" if "Bitget" in whose else "evidence", ref, whose))
    for _s, (times, whose) in sorted(found.items()):
        sources.append(Source("evidence", EDGAR_REF, f"{whose}: {len(times)} release(s)"))
    if histories:
        sources.append(Source("computation", "argus.desk.odds:directional_odds",
                              f"{BAND} of same-length windows before each entry"))
    if holidays is not None and any(c.closures is not None for c in checks):
        sources.append(Source("computation", "lean_market_holidays_usa.json",
                              "QuantConnect Lean US equity holidays, via truth/clocks.py"))
    data_parts = ["your own entry and exit prices"]
    if histories:
        data_parts.append("daily closes: " + "; ".join(
            whose for _, (_, whose) in sorted(histories.items())))
    if found:
        data_parts.append("earnings: SEC EDGAR 8-K item 2.02 acceptance times")
    lines.append("Data: " + "; ".join(data_parts) + f"; read {clock:%Y-%m-%d %H:%M} UTC.")

    data = {"journal": {
        "parsed_from": journal.parsed_from,
        "trades": [c.as_dict() for c in checks],
        "patterns": [p.as_dict() for p in patterns],
        "checklist": [p.check for p in patterns],
        "anecdotal": len(trades) < MIN_STATISTICAL_TRADES,
        "open": still_open, "dropped": dropped, "fetch_failed": failed,
        "gate": None if gate_report is None else {
            k: gate_report[k] for k in ("refused", "windows") if k in gate_report},
    }}
    return lines, sources, data


__all__ = [
    "BAND",
    "DISPOSITION_RATIO",
    "GATE_MIN_TRADES",
    "MIN_STATISTICAL_TRADES",
    "Check",
    "Journal",
    "Leg",
    "Pattern",
    "Trade",
    "check_trade",
    "daily_history",
    "earnings_releases",
    "find_patterns",
    "pair",
    "parse_fills",
    "parse_text",
    "reaction_day",
    "read_journal",
    "review_requested",
    "review_trades",
]
