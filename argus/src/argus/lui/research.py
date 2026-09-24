"""Research questions — the part of the console a Track 3 judge actually types.

**Why this module exists, measured before it was written.** Driving the hosted console on
2026-09-23 with five questions a trader would naturally ask, it answered none of them correctly.
"How would adding TSLA change my portfolio risk" — the exact worked example the handbook gives for
the Track 3 Open Theme — was classified as a performance question and answered with "Sharpe not
available". Everything needed to answer it properly already existed: :func:`argus.desk.portfolio.
copilot` computes session-aware beta, risk share, diversification, beta-propagated stress and the
realised worst window from live Bitget candles, and :func:`argus.desk.workbench.plan_execution`
splits an order with every slice's cost stated. None of it was reachable by asking. The console
was an explainer of its own ledger, which is one feature of a research workbench, not the thing
itself.

**The design keeps the property the rest of the console is built on: the model never says a
number.** A research question is parsed into a :class:`ResearchRequest` — a kind, the symbols, the
book, a size, a shock — by a model asked to fill in that structure and nothing else
(:func:`plan_with_model`), or, with no model available, by deterministic patterns (:func:`detect`:
instant, free, auditable). The model reads first because it was measured to read better: on two
corpora written blind by agents that never saw this file, the patterns alone got 73% and 58% of
questions right, and the model recovered most of the rest. Either way the request is validated
against the traded universe and then answered by the desk's own engines. Every figure a reader sees
comes from arithmetic over candles, carries a :class:`~argus.lui.answer.Source`, and says where the
candles came from.

**Where the data comes from, stated on every answer.** Live Bitget candles are fetched first,
concurrently, under a hard deadline. If any symbol cannot be fetched in time the whole answer falls
back to `data/risk_layer_candles_fixture.json` — real Bitget history frozen on a named date — rather
than mixing live and frozen series, because two series from different clocks do not align and a
risk share computed across them would be wrong in a way nobody could see. The fallback is named in
the answer with its date; it is never silent.

**What the answer is for.** Track 3 asks for a flow "from question to actionable insight", with a
human making the final call. So each answer leads with the conclusion a trader can act on — how
much of the book's risk the trade would carry, and the largest size that keeps it inside a stated
risk budget — then the evidence, then what the desk itself last concluded about the name. It never
tells anyone to buy: the console reads and computes; the trader decides.
"""

from __future__ import annotations

import json
import math
import re
import threading
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Any

from argus.desk.portfolio import (
    CopilotReport,
    PortfolioError,
    Shock,
    align,
    beta,
    copilot,
    correlation,
    decompose,
    rebalance,
    returns,
    stress_by_beta,
    variance,
)
from argus.lui.answer import Answer, Source
from argus.lui.question import (
    TRADED_SYMBOLS,
    Intent,
    Question,
    Speed,
    Tense,
    resolve_symbol,
)

BENCHMARK = "QQQUSDT"
LOOKBACK_DAYS = 30
FETCH_DEADLINE_S = 12.0
"""Total wall-clock allowed for the live fetch. Chosen against the hosting budget, not tuned: a
serverless request that runs past its limit returns nothing at all, and a frozen answer with its
date stated beats no answer."""

DEFAULT_SIZE = 0.20
"""Target weight when the question names none. Stated in the answer whenever it is used."""

RISK_BUDGET = 0.25
"""The single-name risk share the sizing guidance is written against: no one position carrying more
than a quarter of the book's volatility. A common desk convention rather than a law — it is shown
in the answer as the assumption it is, so a trader with a different budget can read the table."""

CACHE_TTL_S = 600.0

FIXTURE_PATH = Path(__file__).resolve().parents[3] / "data" / "risk_layer_candles_fixture.json"


class ResearchKind(StrEnum):
    IMPACT = "impact"
    """What a proposed trade does to a book (or, with no book, to a standalone position)."""

    STRESS = "stress"
    """What a benchmark move does to a book, through each position's beta, plus the realised worst
    window."""

    COMPARE = "compare"
    """Two or more names side by side: session betas, stress sensitivity, correlation."""

    EXECUTION = "execution"
    """How to split an order of a stated size, with the cost of each slice."""

    QUOTE = "quote"
    """Where a name trades right now on Bitget, what a round trip costs, and whether its anchor
    market is open. A research workbench that refuses to say a price is not one — the old console
    refused on the grounds that a live quote is not part of the ledger, which was right for a
    ledger explainer and wrong for this."""

    TECHNICALS = "technicals"
    """Momentum, trend and structure from Bitget's own `bitget-signal` technical-analysis Skill —
    RSI, MACD, support/resistance and ATR — read, not recomputed, and cross-checked elsewhere
    (`market/skills.cross_check_rsi`)."""

    FUNDAMENTALS = "fundamentals"
    """The anchor company from Bitget's `bitget-mcp-server`: its scheduled report, the analyst
    consensus when it is fresh enough to use, institutional (13F) holders and valuation, plus the
    rToken's premium to the stock."""

    ANALOGUE = "analogue"
    """Decision Stress Testing's own question — *has this setup happened before, and what followed?*
    — answered as a distribution over the name's past states, not an anecdote
    (`desk/analogue.find`)."""


@dataclass(frozen=True)
class ResearchRequest:
    """A research question reduced to exactly what the engines need. Nothing else survives."""

    kind: ResearchKind
    symbols: tuple[str, ...]
    """The names the question is about. For IMPACT and EXECUTION, the first is the candidate."""

    book: Mapping[str, float] = field(default_factory=dict)
    """Current holdings as weights summing to one, or empty when none were stated."""

    size: float = DEFAULT_SIZE
    size_stated: bool = False
    shock_pct: float | None = None
    notional: Decimal | None = None
    urgent: bool = False
    parsed_by: str = "patterns"
    notes: tuple[str, ...] = ()
    """Every assumption the parser made, shown to the reader verbatim."""

    budget: float = RISK_BUDGET
    """The largest share of book risk the trader will let one name carry. Their own number when
    they state one ("keep any name under 15% of my risk"), else the stated default."""

    budget_stated: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": str(self.kind),
            "symbols": list(self.symbols),
            "book": dict(self.book),
            "size": self.size,
            "size_stated": self.size_stated,
            "shock_pct": self.shock_pct,
            "notional": None if self.notional is None else str(self.notional),
            "urgent": self.urgent,
            "parsed_by": self.parsed_by,
            "notes": list(self.notes),
            "budget": self.budget,
            "budget_stated": self.budget_stated,
        }


# --- parsing --------------------------------------------------------------------------------

_WORDISH = r"[A-Za-z][A-Za-z0-9]{1,15}"
_PCT = r"(\d+(?:\.\d+)?)\s*(?:%|percent\b|pct\b)"
_PAIR_PCT_FIRST = re.compile(rf"{_PCT}\s*(?:of\s+|in\s+|into\s+)?(?:my\s+)?({_WORDISH})", re.I)
_PAIR_NAME_FIRST = re.compile(rf"({_WORDISH})\s*(?:at|=|:|-)?\s*{_PCT}", re.I)
_PAIR_FRACTION = re.compile(rf"({_WORDISH})\s*=\s*(0?\.\d+|1(?:\.0+)?)\b", re.I)

_ADD_VERB = re.compile(
    r"\b(?:add(?:ing)?|buy(?:ing)?|include|including|put(?:ting)?|allocat\w*|"
    r"throw(?:ing)?|toss(?:ing)?|pick(?:ing)?\s+up|swap(?:ping)?\s+in|"
    r"get(?:ting)?\s+into|enter(?:ing)?|go(?:ing)?\s+long|rotat\w*(?:\s+\d+(?:\.\d+)?%)?\s+"
    r"into|mov(?:e|ing)\s+(?:\d+(?:\.\d+)?%\s+)?into|switch(?:ing)?\s+into)\b",
    re.I,
)
_HOLDINGS = re.compile(
    r"\b(?:i\s+(?:hold|own|have|am\s+holding)|my\s+(?:book|portfolio|holdings?|positions?)\s+"
    r"(?:is|are|=|:)|holding|hedged\s+with|currently\s+(?:hold|own)|i'?m\s+(?:mostly|mainly|all|heavy|"
    r"heavily|long|overweight|loaded)(?:\s+(?:in|on|up\s+on|with))?|i\s+am\s+(?:mostly|"
    r"mainly|heavy|heavily|long)(?:\s+(?:in|on))?|(?:a\s+)?(?:bunch|lot|ton)\s+of|"
    r"heavy\s+on)\b",
    re.I,
)
_PORTFOLIO_WORDS = re.compile(
    r"\b(?:portfolio|book|holdings?|risk|diversif\w*|exposure|beta|concentrat\w*|correlat\w*|"
    r"hedge|volatil\w*|drawdown|position\s+siz\w*|how\s+much)\b",
    re.I,
)
_SHOULD_I = re.compile(
    r"\b(?:should\s+i|is\s+it\s+(?:a\s+good\s+idea|smart|wise)\s+to|worth\s+(?:buying|adding)|"
    r"what\s+if\s+i|would\s+it\s+make\s+sense\s+to|can\s+i|dumb|stupid|smart|wise|"
    r"(?:good|bad)\s+idea|worth\s+it|make\s+(?:things|it)\s+(?:worse|better)|thinking\s+"
    r"(?:about|of))\b",
    re.I,
)
_STRESS = re.compile(
    r"\b(?:stress|worst[\s-]case|bear\s+case|what\s+(?:happens|would\s+happen)|what\s+if|if|"
    r"scenario|shock|when|(?:impact|effect)\s+of)\b[^?]*?\b(?:market|qqq|nasdaq|ndx|benchmark|index|stocks?|tech|"
    r"semis?|semiconductors?|chips?|sector|equities|it|everything)\b"
    r"[^?]*?\b(?:drop\w*|fall\w*|fell|crash\w*|dump\w*|tank\w*|sell[\s-]?off|sells?\s+off|"
    r"selling\s+off|down|declin\w*|"
    r"rall\w*|ris\w*|jump\w*|up|surg\w*|gain\w*|rips?|ripping|moon\w*|pump\w*|"
    r"-\s?\d+(?:\.\d+)?\s*%)",
    re.I,
)
_STRESS_BARE = re.compile(
    r"\bstress[\s-]?test\w*\b|\bworst[\s-]case\b|\bbear\s+case\b|\bhow\s+bad\b|"
    r"\bwhat'?s\s+the\s+damage\b|\bp\s?&\s?l\s+on\b|\bhow\s+exposed\b|\bhow\s+(?:bad|much)\s+"
    r"(?:would|could|will)\s+i\s+lose\b",
    re.I,
)
_SHOCK_NUMBER = re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:%|percent\b|pct\b)", re.I)
_DOWN_WORDS = re.compile(
    r"\b(?:drop\w*|fall\w*|fell|crash\w*|dump\w*|tank\w*|sell[\s-]?off|sells?\s+off|"
    r"selling\s+off|down|declin\w*)\b", re.I
)
_COMPARE = re.compile(
    r"\b(?:compare|comparison|versus|vs\.?|or|against|relative\s+to|more\s+risky|riskier|"
    r"safer|which\s+is\s+(?:better|riskier|safer)|difference\s+between|correlat\w*|rank\w*|"
    r"sort\w*|by\s+(?:realis|realiz)\w*\s+vol\w*|by\s+vol\w*|most\s+volatile|least\s+volatile)\b",
    re.I,
)
_PROFILE = re.compile(
    r"\b(?:beta|how\s+risky|risky|riskier|safe|risk\s+profile|volatil\w*|sensitiv\w*|exposure|how\s+correlated|"
    r"correlat\w*)\b",
    re.I,
)
_EXECUTION = re.compile(
    r"\b(?:split|slice|execute|execution|work(?:ing)?\s+(?:an?\s+)?order|fill|slippage|"
    r"how\s+(?:should|do|would)\s+i\s+(?:buy|sell|enter|get\s+into|exit)|best\s+way\s+to\s+"
    r"(?:buy|sell|exit|enter|get)|market\s+or\s+limit|limit\s+or\s+market|twap|vwap|"
    r"in\s+one\s+(?:clip|go|shot)|minimi[sz]e\s+(?:the\s+)?(?:impact|slippage|cost)|"
    r"cost\s+of\s+(?:buying|selling)|(?:sell|selling|exit|exiting|unload\w*|dump\w*)\s+"
    r"(?:\$|\d))",
    re.I,
)
_NOTIONAL = re.compile(
    r"(?:\$|usd\s*|usdt\s*)?\s*(\d+(?:[.,]\d+)*)\s*(k|m|thousand|million)?\s*"
    r"(?:\$|usd|usdt|dollars?)?",
    re.I,
)
_QUOTE = re.compile(
    r"\b(?:price|priced|trading\s+at|trade\s+at|trades\s+at|quote\w*|how\s+much\s+is|"
    r"going\s+for|spread|funding|last\s+print|what'?s\s+(?:the\s+)?\w+\s+at|"
    r"where\s+is\s+(?:the\s+)?\w+\s+trading|where'?s\s+(?:the\s+)?\w+\s+trading)\b",
    re.I,
)
_BUDGET = re.compile(
    r"(?:risk\s+budget|max(?:imum)?\s+(?:risk|share\s+of\s+risk)|no\s+(?:single\s+)?name\s+"
    r"(?:above|over|more\s+than)|(?:any|each|one|a\s+single)\s+name\s+(?:under|below|at\s+"
    r"most)|cap\s+(?:each|any|per)\s+name\s+at)\D{0,30}?(\d+(?:\.\d+)?)\s*%",
    re.I,
)
"""A trader's own single-name risk cap. Kept deliberately narrow: a percentage near these words is
a budget, and every other percentage in the question is a weight, a size or a shock."""


def parse_budget(text: str) -> float | None:
    match = _BUDGET.search(text)
    if match is None:
        return None
    value = float(match.group(1)) / 100.0
    return value if 0.0 < value < 1.0 else None


def _strip_budget(text: str) -> str:
    """The question with its budget phrase removed, so "15%" is never also read as a weight."""
    return _BUDGET.sub(" ", text)


_ANALOGUE = re.compile(
    r"\b(?:been\s+here\s+before|similar\s+(?:setups?|situations?|times|periods|conditions|"
    r"moments)|like\s+this\s+before|historically|history\s+(?:says|shows|suggests)|"
    r"what\s+happened\s+(?:next|after|last\s+time)|past\s+(?:times|instances)|analog\w*|"
    r"last\s+time\s+it|base\s+rate|(?:ever|previously)\s+(?:done|did|been|had|dropped|"
    r"rallied|pumped|looked)|(?:prior|previous|past)\s+(?:instances?|episodes?|occasions?)|"
    r"(?:comparable|similar)\s+(?:\w+\s+){0,3}(?:pattern|setup|move|drawdown|situation)|"
    r"did\s+(?:that|this)\s+pattern|pehle\s+bhi|pichli\s+baar|alguna\s+vez)\b",
    re.I,
)
_FUNDAMENTALS = re.compile(
    r"\b(?:earnings|reports?\s+(?:next|on|when)|when\s+does\s+\w+\s+report|eps|guidance|"
    r"analysts?|price\s+target|consensus|(?:analyst|eps|earnings|revenue|consensus)\s+"
    r"estimates?|13f|institution\w*|who\s+owns|holders?|"
    r"fundamental\w*|revenue|valuation|market\s+cap|p/?e\b|premium\s+to|discount\s+to|"
    r"vs\.?\s+the\s+stock|against\s+the\s+stock)\b",
    re.I,
)
_TECHNICALS = re.compile(
    r"\b(?:rsi|macd|technical\w*|overbought|oversold|support|resistance|momentum|trend\w*|"
    r"chart\w*|bollinger|atr|moving\s+average|ta\b|breakout|levels?)\b",
    re.I,
)
_FORECAST = re.compile(
    r"\b(?:tomorrow|tonight|next\s+(?:week|month|year|quarter)|by\s+(?:monday|tuesday|wednesday|"
    r"thursday|friday|the\s+close|eod|end\s+of)|will\s+\w+\s+(?:be|go|close|hit|reach|trade)|"
    r"going\s+to\s+(?:be|go|hit)|predict\w*|forecast\w*|guess|kal|kitna\s+hoga)\b",
    re.I,
)
"""A price asked for at a future time. Only consulted for a quote: "if the market drops tomorrow,
how bad does my MSTR get hit" is a stress question and stays one. Found on the blind corpus C,
where "price kal subah kitna hoga BTC ka, guess kar lo" (tomorrow morning's BTC price) was quoted
as today's."""
_PRICE_TARGET = re.compile(r"\b(?:price\s+target|target\s+price|in\s+20\d\d)\b", re.I)


def _is_equity(symbol: str) -> bool:
    from argus.market import universe

    return universe.is_equity(symbol)


_URGENT = re.compile(r"\b(?:urgent\w*|asap|right\s+now|immediately|fast|quickly)\b", re.I)


_NOT_A_NAME = frozenset({
    "I", "A", "AI", "PM", "AM", "ET", "UTC", "US", "USD", "USDT", "EPS", "PE", "ROI", "NAV", "LLM",
    "OK", "NO", "YES", "WHY", "HOW", "AND", "OR", "THE", "ARGUS", "IT", "WE", "Q", "FY", "ETF",
    "CEO", "CFO", "IPO", "GDP", "CPI", "PPI", "FOMC", "FED", "SEC", "RSI", "MACD", "ATR", "ATH",
    "ATL", "EMA", "SMA", "VWAP", "TWAP", "PNL", "DCA", "TP", "SL", "IV", "YTD", "QOQ", "YOY", "EOD",
    "ASAP", "FYI", "IMO", "EU", "UK", "HK", "EV", "API", "NOW", "ME", "MY", "IS", "AT", "ON",
    "TO", "IN", "OF", "BE", "DO", "GO", "SO", "UP", "IF", "BY", "AN", "AS", "ALL", "ANY", "BUY",
    "SELL", "HOLD", "LONG", "SHORT", "WHAT", "WHEN", "RISK", "TOP", "NEW", "ONE", "BIG", "HIGH",
    "LOW", "OPEN", "CLOSE", "SAFE", "GOOD", "BAD", "BEST", "HODL", "FOMO", "FUD", "NFA", "DYOR",
    "OTC", "TA", "FA", "YOLO", "BTFD", "LOL", "OMG", "WTF", "TLDR", "MAX", "MIN", "AVG", "VS",
})
"""Capitalised words that are prose, units or trading vocabulary, never a contract — several of
them ("US", "ME", "NOW") are also Bitget tickers, which is exactly why they are listed. ServiceNow
is still reachable as ``NOWUSDT`` or ``ServiceNow``."""


def _resolve(raw: str, *, trust_case: bool = True) -> tuple[str, str] | None:
    """One token to a listed contract and a note on how it was read, or None.

    The twelve rTokens and their names ("tesla") resolve in any case, as do the unambiguous names
    in `argus.market.universe.ALIASES` ("gold", "bitcoin") and a full symbol ("pltrusdt"). Any
    other listed ticker must be written the way tickers are written — in capitals — because Bitget
    lists tokens called US, ME and NOW, and "what should I buy now" is not a question about
    ServiceNow. ``trust_case=False`` (a question typed entirely in capitals) turns that last rule
    off rather than reading every shouted word as an instrument.
    """
    from argus.market import universe

    traded = resolve_symbol(raw)
    if traded is not None:
        return traded, ""
    upper = raw.strip().upper()
    if not upper or upper in _NOT_A_NAME:
        return None
    if upper in universe.ALIASES or upper.endswith("USDT"):
        return universe.resolve(upper)
    if trust_case and raw.strip().isupper() and len(upper) >= 2:
        return universe.resolve(upper)
    if upper in _NAMED_COMPANIES:
        return universe.resolve(_NAMED_COMPANIES[upper])
    return None


_NAMED_COMPANIES = {"SERVICENOW": "NOW"}


def _shouting(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return len(letters) >= 12 and sum(c.isupper() for c in letters) / len(letters) > 0.7


def _read(text: str) -> dict[str, str]:
    """Every listed contract the text names, in the order named, each with the note on how it was
    read ("" when it was read literally)."""
    trust = not _shouting(text)
    found: dict[str, str] = {}
    for match in re.finditer(r"[A-Za-z][A-Za-z0-9&]{1,17}", text):
        hit = _resolve(match.group(0).replace("&", ""), trust_case=trust)
        if hit is not None and hit[0] not in found:
            found[hit[0]] = hit[1]
    return found


def worth_asking_the_model(text: str, *, now: datetime | None = None) -> bool:
    """Whether the model should read ``text`` before the patterns do.

    Not when the question is about the desk's own record — the ledger answerers own those. Yes
    whenever a word could be a listed contract (:func:`may_name_a_contract`). And yes whenever the
    ledger patterns cannot place the question at all: a name the registry cannot spell ("intel",
    "aaple", "oro", "natural gas") is exactly what the model can, and on the fourth blind corpus
    (2026-09-23) four of six misses were questions the model was simply never shown. A question the
    ledger patterns *do* place, with no contract in it, is left to them — asking the model there
    would only add latency to a question already answered.
    """
    from argus.lui.question import classify

    if about_the_record(text):
        return False
    if may_name_a_contract(text):
        return True
    placed = classify(text, now=now or datetime.now(UTC)).intent
    # POSITION and MARKET are claimed on single words ("hold", "price") that a research question
    # about an unrecognised name uses too: "is intel a scary stock to hold" was read as a question
    # about the desk's open positions. The model reads those as well; a genuine position or market
    # question comes back as `record` and falls through to the ledger unchanged.
    return placed in (Intent.UNKNOWN, Intent.AMBIGUOUS, Intent.UNSUPPORTED, Intent.POSITION,
                      Intent.MARKET)


def may_name_a_contract(text: str) -> bool:
    """Whether any word in ``text``, in any case, could be a listed contract — the gate for
    consulting the model, which is deliberately looser than :func:`research_symbols`.

    `research_symbols` only reads an uncapitalised word as a ticker when it is one of the desk's
    names or an alias, because Bitget lists tokens called US, ME and NOW. That is right for the
    patterns, which cannot tell "hood" the ticker from "hood" the word. The model can, so it is
    asked whenever a word *could* be a contract, and its answer goes through the same resolver.
    Measured on a corpus written blind (2026-09-23): "is amd overbought rn", "whats the spread on
    hood" and "is gme really risky" never reached the model before this gate.
    """
    from argus.market import universe

    if research_symbols(text)[0]:
        return True
    listed = universe.contracts()
    for match in re.finditer(r"[A-Za-z][A-Za-z0-9]{1,15}", text):
        word = match.group(0).upper()
        if word in _NOT_A_NAME or len(word) < 2:
            continue
        if word in universe.ALIASES or f"{word}USDT" in listed or f"{word}STOCKUSDT" in listed:
            return True
    return False


def research_symbols(text: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Every listed contract the text names, in the order named, and a note for each one read
    other than literally (SPX as the S&P 500, CVX as Chevron)."""
    found = _read(text)
    return tuple(found), tuple(n for n in found.values() if n)


def _pairs(text: str) -> list[tuple[int, str, float]]:
    """Every (position, symbol, weight) the text states, weights as fractions of one."""
    found: list[tuple[int, str, float]] = []
    taken: set[tuple[int, int]] = set()

    trust = not _shouting(text)

    def keep(span: tuple[int, int], name: str, value: float) -> None:
        hit = _resolve(name, trust_case=trust)
        symbol = None if hit is None else hit[0]
        if symbol is None or value <= 0:
            return
        if any(not (span[1] <= a or span[0] >= b) for a, b in taken):
            return
        taken.add(span)
        found.append((span[0], symbol, value))

    for match in _PAIR_FRACTION.finditer(text):
        keep(match.span(), match.group(1), float(match.group(2)))
    for match in _PAIR_PCT_FIRST.finditer(text):
        keep(match.span(), match.group(2), float(match.group(1)) / 100.0)
    for match in _PAIR_NAME_FIRST.finditer(text):
        keep(match.span(), match.group(1), float(match.group(2)) / 100.0)
    found.sort()
    return found


def _normalise(book: dict[str, float], notes: list[str]) -> dict[str, float]:
    total = sum(book.values())
    if not book or total <= 0:
        return {}
    if abs(total - 1.0) > 0.02:
        notes.append(
            f"your holdings add up to {total:.0%}, so they were scaled to 100% before any risk "
            f"figure was computed — say the rest if it is cash or another name"
        )
        return {s: w / total for s, w in book.items()}
    return book


def _parse_notional(text: str) -> Decimal | None:
    best: Decimal | None = None
    for match in _NOTIONAL.finditer(text):
        digits, unit = match.group(1), (match.group(2) or "").lower()
        whole = match.group(0)
        if not unit and "$" not in whole and not re.search(r"usd|dollar", whole, re.I):
            continue
        try:
            value = Decimal(digits.replace(",", ""))
        except ArithmeticError:
            continue
        if unit in ("k", "thousand"):
            value *= 1000
        elif unit in ("m", "million"):
            value *= 1_000_000
        if value > 0 and (best is None or value > best):
            best = value
    return best


_ABOUT_THE_RECORD = re.compile(
    r"\b(?:you|the\s+desk|argus|we)\s+(?:did|do|traded|pass\w*|skip\w*|decide\w*|bought|sold|"
    r"stood|abstain\w*)\b|\bwhy\s+did\b|\bdecision\s+\d+\b",
    re.I,
)


def about_the_record(text: str) -> bool:
    """A question about what the desk itself did — the ledger's to answer, never research's."""
    return bool(_ABOUT_THE_RECORD.search(text))


def detect(text: str) -> ResearchRequest | None:
    """A research request, or None when the question is not one — see :func:`_detect`.

    Wraps it to attach how each analysed name was read, and only for names the request actually
    analyses: in "what if the Nasdaq drops 10%? I hold 40% gold" the Nasdaq is the shock, not a
    holding, and a note saying it was read as NDX100USDT would describe a reading never used.
    """
    request = _detect(text)
    if request is None:
        return None
    read_as = [n for s, n in _read(text).items() if n and s in request.symbols
               and n not in request.notes]
    return replace(request, notes=(*request.notes, *read_as)) if read_as else request


_SHOCK_SUBJECTS = frozenset({"NDX100USDT", "SP500USDT", "DIASTOCKUSDT"})
"""Index products a trader names as the market in a scenario. Since the console began resolving
every Bitget contract, "if the Nasdaq drops 10%" resolved "Nasdaq" to NDX100USDT and answered for a
book of 100% NDX100 — a confident answer about a portfolio nobody holds."""


def _detect(text: str) -> ResearchRequest | None:
    """A research request, or None when the question is not one.

    Conservative on purpose. A question about the desk's own record ("why did you pass on NVDA")
    must fall through to the ledger answerers untouched, so a request is only built when the words
    that make it a research question are actually present, and a named symbol is always required.
    """
    raw = text.strip()
    if not raw:
        return None
    budget = parse_budget(raw)
    if budget is not None:
        request = detect(_strip_budget(raw))
        return None if request is None else replace(request, budget=budget, budget_stated=True)
    symbols, _ = research_symbols(raw)
    if _STRESS.search(raw) or _STRESS_BARE.search(raw):
        # "if the Nasdaq drops 10%" names the market being shocked, not a holding. An index
        # product stays in the request only when it is given a weight ("30% SP500").
        weighted = {symbol for _, symbol, _ in _pairs(raw)}
        symbols = tuple(s for s in symbols if s not in _SHOCK_SUBJECTS or s in weighted)
    if not symbols:
        if (_STRESS.search(raw) or _STRESS_BARE.search(raw)) and re.search(
            r"\b(?:my|i'?m|i\s+am|i\s+hold|i\s+own)\b", raw, re.I
        ):
            return ResearchRequest(kind=ResearchKind.STRESS, symbols=())
        return None
    # Questions about what the desk did are the ledger's, whatever else they mention.
    if about_the_record(raw):
        return None

    notes: list[str] = []
    pairs = _pairs(raw)
    urgent = bool(_URGENT.search(raw))

    simple = (not _ADD_VERB.search(raw) and not _STRESS.search(raw) and not pairs
              and not _EXECUTION.search(raw))
    if simple and _ANALOGUE.search(raw):
        return ResearchRequest(kind=ResearchKind.ANALOGUE, symbols=symbols[:1])
    if simple and _FUNDAMENTALS.search(raw):
        if _PRICE_TARGET.search(raw) and not (symbols[0] in TRADED_SYMBOLS
                                              or _is_equity(symbols[0])):
            return None  # a price target for crypto or a commodity is a forecast; refused
        return ResearchRequest(kind=ResearchKind.FUNDAMENTALS, symbols=symbols[:1])
    if simple and _TECHNICALS.search(raw):
        return ResearchRequest(kind=ResearchKind.TECHNICALS, symbols=symbols[:1])
    if simple and _QUOTE.search(raw):
        if _FORECAST.search(raw):
            return None  # a price asked for a future time is a forecast; refused, not quoted
        return ResearchRequest(kind=ResearchKind.QUOTE, symbols=symbols[:4])

    if _EXECUTION.search(raw):
        notional = _parse_notional(raw)
        if notional is not None:
            return ResearchRequest(
                kind=ResearchKind.EXECUTION, symbols=symbols[:1], notional=notional,
                urgent=urgent, notes=tuple(notes),
            )

    add_match = _ADD_VERB.search(raw)
    holdings_match = _HOLDINGS.search(raw)
    candidate: str | None = None
    size: float | None = None
    book: dict[str, float] = {}

    if add_match:
        after_verb = [p for p in pairs if p[0] >= add_match.start()]
        before_verb = [p for p in pairs if p[0] < add_match.start()]
        # The candidate is the first name named after the verb; its paired weight, if any, is
        # the size. Everything stated before the verb is the book.
        tail_symbols, _ = research_symbols(raw[add_match.start():])
        if tail_symbols:
            candidate = tail_symbols[0]
        for _, symbol, weight in after_verb:
            if symbol == candidate and size is None:
                size = weight
            else:
                book[symbol] = book.get(symbol, 0.0) + weight
        for _, symbol, weight in before_verb:
            book[symbol] = book.get(symbol, 0.0) + weight
        if size is None and not tail_symbols:
            candidate = None
    else:
        for _, symbol, weight in pairs:
            book[symbol] = book.get(symbol, 0.0) + weight

    # "I hold NVDA and AAPL" — names without weights — is an equal-weight book, said out loud.
    if holdings_match and not book:
        clause = raw[holdings_match.end(): add_match.start() if add_match and
                     add_match.start() > holdings_match.end() else len(raw)]
        held, _ = research_symbols(clause)
        held = tuple(s for s in held if s != candidate)
        if held:
            book = {s: 1.0 / len(held) for s in held}
            notes.append(
                "no weights were given for your holdings, so they were read as equal weight"
            )

    if candidate and candidate in book and size is None:
        # "add more NVDA" with NVDA already held and no size: the default applies, stated.
        pass
    book = _normalise(book, notes)

    stress = _STRESS.search(raw) or _STRESS_BARE.search(raw)
    if stress and not add_match:
        shock: float | None = None
        for match in _SHOCK_NUMBER.finditer(raw):
            # A percentage that is a holding weight is not the shock size.
            if any(abs(pos - match.start()) < 2 for pos, _, _ in pairs):
                continue
            value = abs(float(match.group(1)))
            shock = -value if _DOWN_WORDS.search(raw) or match.group(1).startswith("-") else value
        return ResearchRequest(
            kind=ResearchKind.STRESS,
            symbols=tuple(book) or symbols,
            book=book or {s: 1.0 / len(symbols) for s in symbols},
            shock_pct=shock,
            notes=tuple(notes if book else [
                *notes,
                "no book was stated, so " + (
                    f"{symbols[0]} is stressed on its own" if len(symbols) == 1
                    else "the named symbols are stressed as an equal-weight book"
                ),
            ]),
        )

    if candidate and (add_match and (_PORTFOLIO_WORDS.search(raw) or _SHOULD_I.search(raw)
                                     or book or size is not None)):
        if size is None:
            notes.append(
                f"no size was given, so {candidate} is assessed at a {DEFAULT_SIZE:.0%} target "
                f"weight — say the size you have in mind to change it"
            )
        if not book:
            notes.append(
                "no current holdings were stated, so this is the risk of the position on its "
                "own — tell me what you hold (\"I hold 50% NVDA, 50% AAPL\") to see what it "
                "does to your book"
            )
        return ResearchRequest(
            kind=ResearchKind.IMPACT,
            symbols=(candidate, *[s for s in book if s != candidate]),
            book=book, size=min(max(size or DEFAULT_SIZE, 0.01), 1.0),
            size_stated=size is not None, notes=tuple(notes),
        )

    if len(symbols) >= 2 and (_COMPARE.search(raw) or _PROFILE.search(raw)):
        return ResearchRequest(kind=ResearchKind.COMPARE, symbols=symbols[:4], notes=tuple(notes))

    if _PROFILE.search(raw) or _SHOULD_I.search(raw):
        # The name being judged is one the trader does not already hold. When every named symbol
        # is a holding, the question is about the book itself, and a trade of a name into its own
        # position would report "100% of risk (was 100%)" — arithmetic about nothing.
        fresh = [s for s in symbols if s not in book]
        if book and not fresh:
            return ResearchRequest(kind=ResearchKind.STRESS, symbols=tuple(book), book=book,
                                   notes=tuple(notes))
        if book:
            notes.append(
                f"no size was given, so {fresh[0]} is assessed at a {DEFAULT_SIZE:.0%} target "
                f"weight"
            )
        else:
            notes.append(
                "no current holdings were stated, so this is the risk of the position on its own"
            )
        return ResearchRequest(
            kind=ResearchKind.IMPACT, symbols=(fresh[0], *book), book=book,
            size=DEFAULT_SIZE, notes=tuple(notes),
        )
    return None


def parse_book(text: str) -> dict[str, float]:
    """A saved book like "40% NVDA, 30% MSFT, 30% AAPL" (or bare names, read as equal weight)."""
    book: dict[str, float] = {}
    for _, symbol, weight in _pairs(text):
        book[symbol] = book.get(symbol, 0.0) + weight
    if not book:
        named, _ = research_symbols(text)
        book = {s: 1.0 / len(named) for s in named} if named else {}
    return _normalise(book, [])


def with_book(request: ResearchRequest | None, book_text: str) -> ResearchRequest | None:
    """Apply the visitor's saved book to a request that did not state one.

    This is the personalised half of the workbench: a trader who has said once what they hold
    should not have to repeat it in every question. A book written into the question itself always
    wins over the saved one — the question is the more specific statement of intent — and whenever
    the saved book is used, the answer says so.
    """
    if request is None or not book_text.strip():
        return request
    saved_budget = parse_budget(book_text)
    if saved_budget is not None and not request.budget_stated:
        request = replace(request, budget=saved_budget, budget_stated=True)
        book_text = _strip_budget(book_text)
    if request.book:
        return request
    book = parse_book(book_text)
    if not book:
        return request
    note = f"used your saved book ({', '.join(f'{w:.0%} {_t(s)}' for s, w in book.items())})"
    kept = tuple(n for n in request.notes
                 if "no current holdings" not in n and "no book was stated" not in n)
    if request.kind is ResearchKind.IMPACT:
        candidate = request.symbols[0]
        others = [s for s in book if s != candidate]
        return replace(request, book=book, symbols=(candidate, *others), notes=(*kept, note))
    if request.kind is ResearchKind.STRESS:
        return replace(request, book=book, symbols=tuple(book), notes=(*kept, note))
    return request


# --- the model fallback ---------------------------------------------------------------------

PLANNER_PROMPT = """You turn a trader's research question into a structured request for a \
portfolio-risk engine. You never answer the question and never state a number of your own; the \
engine computes everything.

Kinds:
- impact: what adding/buying a name does to their risk, whether a name is risky to own, or a \
single name's risk profile (volatility, beta, skew, kurtosis, R-squared)
- stress: what a market (QQQ) move would do to their holdings, worst case
- compare: two or more names side by side
- execution: how to split or execute an order of a stated size
- quote: where a name trades right now, its price, spread or funding
- technicals: RSI, MACD, trend, momentum, support/resistance, overbought/oversold
- fundamentals: earnings dates, analyst estimates or targets, institutional holders, valuation
- analogue: has this name been in a similar situation before and what followed — "has it ever \
done this", "prior instances of a comparable pattern", "last time it looked like this", in any \
language ("pehle bhi", "pichli baar", "alguna vez"). Choose this over technicals or quote whenever \
the question asks what happened AFTER a similar past setup
- record: a question about the trading desk's OWN activity — its decisions, why it traded or \
stood aside, its performance, positions, evidence, risk layer, calibration
- none: anything else — chit-chat, jokes, weather, news, orders to trade, and any request to \
predict, guess or target a future price ("where will it be tomorrow", "price target for ETH in \
2027", "kal kitna hoga"), in any language

Names: any contract Bitget lists — US stocks and ETFs, commodities, index products, FX pairs and \
crypto. Write every name as its ticker, correcting spelling and translating company or asset names \
from any language: Intel -> INTC, Apple/aaple -> AAPL, oro/gold -> XAU, silver -> XAG, \
WTI oil -> CL, Brent -> BZ, natural gas -> NATGAS, copper -> COPPER, S&P 500 -> SP500, \
Nasdaq-100 -> NDX100, \
bitcoin -> BTC. The engine checks each ticker against the venue and drops anything Bitget does not \
list.

confidence: your honest probability, between 0 and 1, that you chose the right kind. A clear \
research question you understood is about 0.9 even when holdings or sizes are missing — the \
engine handles missing inputs itself. Never leave it at 0 when you did pick a kind.

Holdings stated in dollars are converted to percent of their stated total. "A little bit" or "a \
chunk" with no number means size_percent null. Holdings named without weights go in holdings \
with equal percentages.

Reply with only this JSON (fill every field; null where a value was not stated):
{"kind": "...", "candidate": "...", "names": ["TICKER", "TICKER"], "holdings": {"TICKER": 40}, \
"size_percent": 10, "shock_percent": -10, "order_usd": 50000, "confidence": 0.9, "why": "..."}

"names" lists every instrument the question is about, as tickers, in the order asked — for a \
compare it holds every name being compared."""

MIN_PLAN_CONFIDENCE = 0.6


def plan_with_model(text: str, client: Any) -> tuple[ResearchRequest | None, dict[str, Any]]:
    """Ask the model to fill in a :class:`ResearchRequest`, and validate every field it returns.

    Returns the request (or None) and an audit record of what the model said. The model's output is
    treated as untrusted input: kinds outside the enum, names outside the traded universe, weights
    that are not positive numbers — all are dropped, not coerced. It can choose *which* engine to
    run and with *which* inputs; it cannot put a figure in the answer.
    """
    audit: dict[str, Any] = {"attempted": client is not None, "applied": False}
    if client is None:
        audit["detail"] = "no model is configured"
        return None, audit
    try:
        from argus.llm.qwen import Thinking

        # Thinking off: this is triage into a fixed schema, not reasoning. Measured on the first
        # run at the default (LOW), a planner call took 10-28s — longer than a trader waits.
        raw = client.complete_json(
            [{"role": "system", "content": PLANNER_PROMPT}, {"role": "user", "content": text}],
            required_keys=("kind", "confidence"),
            max_tokens=300,
            thinking=Thinking.OFF,
        )
    except Exception as exc:  # any transport failure leaves the deterministic answer standing
        audit["detail"] = f"planner unavailable ({type(exc).__name__})"
        return None, audit
    audit["model"] = {k: raw.get(k) for k in ("kind", "confidence", "why")}
    if str(raw.get("kind", "")).strip().lower() in ("none", "record"):
        audit["detail"] = "the model read this as a question about the desk's record" if str(
            raw.get("kind", "")).strip().lower() == "record" else "the model found no research " \
            "question here"
        return None, audit
    try:
        kind = ResearchKind(str(raw.get("kind", "")).strip().lower())
    except ValueError:
        audit["detail"] = "the model found no research question here"
        return None, audit
    try:
        confidence = float(raw.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    if confidence < MIN_PLAN_CONFIDENCE:
        audit["detail"] = f"confidence {confidence:.2f} below {MIN_PLAN_CONFIDENCE}"
        return None, audit

    notes: list[str] = []
    book: dict[str, float] = {}
    holdings = raw.get("holdings") or {}
    if isinstance(holdings, dict):
        for name, value in holdings.items():
            hit = _resolve(str(name), trust_case=False) or _resolve(str(name).upper())
            symbol = None if hit is None else hit[0]
            try:
                weight = float(value) / 100.0
            except (TypeError, ValueError):
                continue
            if symbol and weight > 0:
                book[symbol] = book.get(symbol, 0.0) + weight
    book = _normalise(book, notes)
    named_by_model = str(raw.get("candidate") or "")
    candidate_hit = (_resolve(named_by_model, trust_case=False)
                     or _resolve(named_by_model.upper())) if named_by_model else None
    candidate = None if candidate_hit is None else candidate_hit[0]
    size: float | None = None
    try:
        if raw.get("size_percent") is not None:
            size = float(raw["size_percent"]) / 100.0
    except (TypeError, ValueError):
        size = None
    if size is not None and not 0.0 < size <= 1.0:
        size = None
    shock: float | None = None
    try:
        if raw.get("shock_percent") is not None:
            shock = float(raw["shock_percent"])
    except (TypeError, ValueError):
        shock = None
    read_as = _read(text)
    named: list[str] = []
    listed_names = raw.get("names")
    for name in listed_names if isinstance(listed_names, list) else []:
        # The model's list of every instrument asked about — the only way a compare reaches a
        # name the text resolver cannot spell ("comparame oro y bitcoin": oro -> XAU).
        hit = _resolve(str(name), trust_case=False) or _resolve(str(name).upper())
        if hit is not None:
            named.append(hit[0])
    symbols = tuple(dict.fromkeys(
        [s for s in (candidate, *named, *book, *read_as) if s is not None]
    ))
    if not symbols:
        audit["detail"] = "the model named no instrument this desk can analyse"
        return None, audit
    if candidate is not None and set(book) == {candidate}:
        # "Is Intel a scary stock to hold" came back as holdings {INTC: 100} with INTC as the
        # candidate — a book made only of the name being asked about. That is a question about the
        # name on its own; sized against itself it answered "even a 1% position would carry more
        # than 25% of this book's risk" (fourth blind corpus, 2026-09-23).
        book = {}
    if kind in (ResearchKind.QUOTE, ResearchKind.FUNDAMENTALS) and (
            _FORECAST.search(text) or (_PRICE_TARGET.search(text) and not all(
                s in TRADED_SYMBOLS or _is_equity(s) for s in symbols))):
        # A price for a future time is a forecast, and a "price target" for anything but a
        # company's shares has no analyst consensus behind it — both refused, not answered. The
        # model read "price target for ETH in 2027" as a fundamentals question.
        audit["detail"] = "a forecast of a future price, which is refused"
        return None, audit

    request: ResearchRequest | None
    if kind is ResearchKind.EXECUTION:
        try:
            notional = Decimal(str(raw.get("order_usd"))) if raw.get("order_usd") else None
        except ArithmeticError:
            notional = None
        request = None if notional is None or notional <= 0 else ResearchRequest(
            kind=kind, symbols=symbols[:1], notional=notional,
            urgent=bool(_URGENT.search(text)), parsed_by="model", notes=tuple(notes),
        )
    elif kind is ResearchKind.STRESS:
        # Holdings as stated; failing that, the names in the question read as an equal-weight
        # book ("my QQQ and META weights") — except an index product, which in a stress question
        # is the market being shocked ("if the Nasdaq drops 10%"), not something held. With
        # nothing left the request goes out empty and is answered by asking for the holdings, or
        # filled from the visitor's saved book.
        if not book:
            held = [s for s in symbols if s not in _SHOCK_SUBJECTS]
            if held:
                book = {s: 1.0 / len(held) for s in held}
                notes.append("no weights were given for your holdings, so they were read as "
                             "equal weight")
        request = ResearchRequest(kind=kind, symbols=tuple(book), book=book, shock_pct=shock,
                                  parsed_by="model", notes=tuple(notes))
    elif kind is ResearchKind.QUOTE or (kind is ResearchKind.COMPARE and len(symbols) >= 2):
        request = ResearchRequest(kind=kind, symbols=symbols[:4], parsed_by="model")
    elif kind in (ResearchKind.TECHNICALS, ResearchKind.FUNDAMENTALS, ResearchKind.ANALOGUE):
        request = ResearchRequest(kind=kind, symbols=symbols[:1], parsed_by="model")
    else:
        lead = candidate or symbols[0]
        if size is None:
            notes.append(f"no size was given, so {lead} is assessed at {DEFAULT_SIZE:.0%}")
        request = ResearchRequest(
            kind=ResearchKind.IMPACT,
            symbols=(lead, *[s for s in book if s != lead]),
            book=book, size=size or DEFAULT_SIZE, size_stated=size is not None,
            parsed_by="model", notes=tuple(notes),
        )
    budget = parse_budget(text)
    if request is not None and budget is not None:
        request = replace(request, budget=budget, budget_stated=True)
    if request is not None:
        extra = [n for s, n in read_as.items()
                 if n and s in request.symbols and n not in request.notes]
        if extra:
            request = replace(request, notes=(*request.notes, *extra))
    audit["applied"] = request is not None
    return request, audit


# --- data -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class MarketData:
    """Per-bar returns for the requested names, and an honest account of where they came from."""

    raw: Mapping[str, Mapping[datetime, float]]
    provenance: str
    source: Source
    live: bool


_SERIES_CACHE: dict[tuple[str, int], tuple[float, dict[datetime, float]]] = {}
_CACHE_LOCK = threading.Lock()
_FETCH_SLOTS = threading.Semaphore(3)
"""At most three candle requests in flight. Bitget's history endpoint returns HTTP 429 when a
burst arrives at once — measured on the first run of this module, when eight concurrent fetches
sent a two-name question to the frozen fallback — so concurrency is capped rather than maximised."""


def _candles(symbol: str, days: int) -> list[Any]:
    """Hourly candles covering ``days``, in as few venue calls as the venue allows.

    Up to 1,000 bars (41 days) is one call to the recent-candles endpoint; history-candles pages at
    100, so the same thirty days used to be eight sequential calls — measured 4.2s per name on
    2026-09-23, which put a three-name question past :data:`FETCH_DEADLINE_S` and sent a BTC
    impact question to a frozen file that has no BTC in it. Longer ranges fall back to paging.
    """
    from argus.market.history import RECENT_LIMIT, CandleType, fetch, fetch_range

    with _FETCH_SLOTS:
        if days * 24 <= RECENT_LIMIT:
            return fetch(symbol, interval="1H", candle_type=CandleType.MARKET, recent=True,
                         limit=days * 24)
        return fetch_range(symbol, days=days, interval="1H", candle_type=CandleType.MARKET,
                           pause=0.1)


def _fetch_one(symbol: str, days: int, deadline: float) -> dict[datetime, float]:
    now = time.monotonic()
    with _CACHE_LOCK:
        hit = _SERIES_CACHE.get((symbol, days))
        if hit and now - hit[0] < CACHE_TTL_S:
            return hit[1]
    wait = 0.5
    while True:
        try:
            candles = _candles(symbol, days)
            break
        except Exception as exc:
            if "429" not in str(exc) or time.monotonic() + wait > deadline:
                raise
            time.sleep(wait)
            wait *= 2
    series = returns([(c.ts, float(c.close)) for c in candles])
    if len(series) < 50:
        raise PortfolioError(f"{symbol}: only {len(series)} bars came back")
    with _CACHE_LOCK:
        _SERIES_CACHE[(symbol, days)] = (time.monotonic(), series)
    return series


def _fetch_live(symbols: Sequence[str], days: int) -> dict[str, dict[datetime, float]]:
    deadline = time.monotonic() + FETCH_DEADLINE_S
    out: dict[str, dict[datetime, float]] = {}
    pool = ThreadPoolExecutor(max_workers=min(4, len(symbols)))
    try:
        futures = {pool.submit(_fetch_one, s, days, deadline): s for s in symbols}
        for future in as_completed(futures, timeout=FETCH_DEADLINE_S):
            out[futures[future]] = future.result()
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    return out


def _from_fixture(symbols: Sequence[str], days: int) -> tuple[dict[str, dict[datetime, float]],
                                                               str]:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    candles = fixture.get("candles", {})
    out: dict[str, dict[datetime, float]] = {}
    for symbol in symbols:
        rows = candles.get(symbol)
        if not rows:
            raise PortfolioError(f"{symbol} is not in the frozen history either")
        prices = [(datetime.fromisoformat(ts), float(close)) for ts, close in rows]
        cutoff = prices[-1][0] - timedelta(days=days)
        out[symbol] = returns([p for p in prices if p[0] >= cutoff])
    return out, str(fixture.get("generated_at", "an unrecorded date"))[:10]


def load(symbols: Sequence[str], *, days: int = LOOKBACK_DAYS) -> MarketData:
    """Returns for ``symbols`` plus the benchmark: live if every name arrives in time, else frozen.

    All-or-nothing on purpose — see the module docstring on why live and frozen series are never
    mixed in one answer.
    """
    wanted = tuple(sorted({*symbols, BENCHMARK}))
    try:
        raw = _fetch_live(wanted, days)
        data = MarketData(
            raw=raw, live=True,
            provenance=f"live Bitget hourly candles, last {days} days, fetched just now",
            source=Source(kind="venue", ref="bitget /api/v3/market/candles",
                          detail=f"{', '.join(wanted)}; {days}d hourly; live"),
        )
    except Exception as exc:
        raw, frozen_on = _from_fixture(wanted, days)
        data = MarketData(
            raw=raw, live=False,
            provenance=(
                f"Bitget hourly candles frozen on {frozen_on} (the live fetch did not complete: "
                f"{type(exc).__name__}) — the figures are real, but not as of this minute"
            ),
            source=Source(kind="computation", ref="data/risk_layer_candles_fixture.json",
                          detail=f"real Bitget history frozen {frozen_on}; {days}d window"),
        )
    return data


# --- answering ------------------------------------------------------------------------------


def _is_open() -> Any:
    from argus.truth.clocks import DualClock

    clock = DualClock()
    return lambda t: clock.phase(t).has_price_discovery


def _open_columns(raw: Mapping[str, Mapping[datetime, float]],
                  is_open: Any) -> dict[str, list[float]]:
    stamps, columns = align(raw)
    rows = [i for i, t in enumerate(stamps) if is_open(t)]
    return {k: [v[i] for i in rows] for k, v in columns.items()}


def _moments(values: Sequence[float]) -> tuple[float, float] | None:
    """Sample skewness and excess kurtosis (the adjusted Fisher-Pearson estimators, as
    `scipy.stats.skew(bias=False)` and `kurtosis(bias=False)` define them)."""
    n = len(values)
    if n < 4:
        return None
    mean = sum(values) / n
    m2 = sum((v - mean) ** 2 for v in values) / n
    if m2 <= 0:
        return None
    m3 = sum((v - mean) ** 3 for v in values) / n
    m4 = sum((v - mean) ** 4 for v in values) / n
    g1 = m3 / m2 ** 1.5
    g2 = m4 / m2 ** 2 - 3.0
    skew = g1 * math.sqrt(n * (n - 1)) / (n - 2)
    kurt = ((n + 1) * g2 + 6.0) * (n - 1) / ((n - 2) * (n - 3))
    return skew, kurt


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
    return (f"Return shape over {len(hourly)} hourly bars: realised volatility {vol:.0%} a year, "
            f"skew {shape[0]:+.2f}, excess kurtosis {shape[1]:.1f} ({tails}){r2}.")


def max_size_within_budget(
    *, add: str, before: Mapping[str, float], columns: Mapping[str, Sequence[float]],
    budget: float = RISK_BUDGET,
) -> float | None:
    """The largest target weight for ``add`` that keeps its share of book risk at or under
    ``budget`` — the number a trader can act on. None when no size in the grid qualifies (or, with
    no book, when the question has no answer: a lone position is always 100% of its own risk)."""
    if not before:
        return None
    best: float | None = None
    for step in range(1, 101):
        size = step / 100.0
        after = rebalance(before, add, size)
        risk = decompose(after, columns)
        if risk is None:
            return None
        share = risk.share_of_risk(add)
        if share is None or share > budget:
            break
        best = size
    return best


def _desk_view(symbol: str, ledger: Any) -> tuple[list[str], list[Source]]:
    if ledger is None:
        return [], []
    if symbol not in TRADED_SYMBOLS:
        return [f"The desk itself trades twelve rTokens and {_t(symbol)} is not one of them, so "
                f"there is no desk call on it — this answer is the analysis alone."], []
    rows = [e for e in ledger.entries if e.symbol == symbol]
    if not rows:
        return [f"The desk has no recorded decision on {symbol} yet."], []
    last = rows[-1]
    thesis = (last.thesis or "").strip()
    line = (
        f"The desk's own last call on {symbol} (decision {last.seq}, {last.decided_at[:16]}Z): "
        f"{last.verdict}"
        + f" at stated confidence {float(last.stated_confidence):.2f}"
        + (f" — {_sentence_cut(thesis)}" if thesis else "")
    )
    return [line], [Source(kind="ledger", ref=f"seq {last.seq}",
                           detail=f"{symbol} {last.verdict} @ {last.decided_at}")]


def _question(raw: str, request: ResearchRequest) -> Question:
    return Question(raw=raw, intent=Intent.RESEARCH, speed=Speed.SLOW, tense=Tense.FUTURE,
                    symbols=request.symbols, matched=f"research:{request.kind}:{request.parsed_by}")


def _t(symbol: str) -> str:
    """The name a trader reads: NVDA, not NVDAUSDT; CVX, not CVXSTOCKUSDT."""
    base = symbol.removesuffix("USDT")
    return base.removesuffix("STOCK") if base.endswith("STOCK") and len(base) > 5 else base


def _clean(line: str) -> str:
    for tag in ("[portfolio] ", "[stress] "):
        line = line.replace(tag, "")
    return re.sub(r"\b([A-Z][A-Z0-9]*?)(?:STOCK)?USDT\b", r"\1", line)


def _sentence_cut(text: str, limit: int = 240) -> str:
    """Shorten at a sentence end rather than mid-word; a thesis cut at "The memory rec" reads as
    a bug, because it is one."""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    end = max(cut.rfind(". "), cut.rfind("; "))
    return (cut[: end + 1] if end > 60 else cut.rsplit(" ", 1)[0] + "...").strip()


def _hedge_line(book_beta: float | None) -> str | None:
    """The hedge the Open Theme asks for, as a number: a QQQ short sized to the book's open-session
    beta neutralises the market-driven part of the risk. Stated with its limit — beta hedges the
    market component only; what a name does on its own news is untouched by it."""
    if book_beta is None or abs(book_beta) < 0.05:
        return None
    side = "short" if book_beta > 0 else "long"
    return (
        f"Hedge: {side} QQQ worth about {abs(book_beta):.0%} of the book's value neutralises its "
        f"market exposure (book beta {book_beta:.2f}); it does nothing for single-name news risk."
    )


def _impact_lines(report: CopilotReport, request: ResearchRequest,
                  columns: Mapping[str, Sequence[float]]) -> list[str]:
    add = report.symbol
    impact = report.impact
    lines: list[str] = []
    standalone = not request.book
    if standalone:
        betas = {str(k): v.beta for k, v in report.session_beta.items()}
        o, s = betas.get("open"), betas.get("shut")
        if o is not None:
            lines.append(
                f"{add} moves {o:.2f}x the Nasdaq-100 (QQQ) while US markets are open"
                + (f" and {s:.2f}x while they are shut" if s is not None else "")
                + " — the open-session figure is the one that prices it."
            )
        worst_day = report.worst.move_pct
        if worst_day is not None and worst_day < 0:
            # With no book there is no risk share to budget against; the sizing rule a trader can
            # act on is the one the realised record gives — the worst day it has actually had.
            lines.append(
                f"Actionable: size {add} so that its worst observed day ({worst_day:+.1f}%) is a "
                f"loss you would accept — on a $10,000 position that is about "
                f"${abs(worst_day) * 100:,.0f}; tell me what you hold to see its share of your "
                f"risk."
            )
    else:
        share = impact.risk_share_after
        if share is not None:
            lines.append(
                f"At {request.size:.0%} of the book, {add} would carry {share:.0%} of your total "
                f"risk" + (f" (was {impact.risk_share_before:.0%})" if impact.risk_share_before
                           else "") + "."
            )
        ceiling = max_size_within_budget(add=add, before=request.book, columns=columns,
                                         budget=request.budget)
        if ceiling is not None:
            verdict = ("inside" if (share or 0.0) <= request.budget else "over")
            lines.append(
                f"Actionable: to keep {add} under {request.budget:.0%} of book risk"
                + (" (your budget)" if request.budget_stated else "")
                + ", size it at no "
                f"more than {ceiling:.0%} — the {request.size:.0%} proposed is {verdict} that "
                f"budget."
            )
        elif share is not None:
            lines.append(
                f"Actionable: even a 1% position in {add} would carry more than "
                f"{request.budget:.0%} of this book's risk — it dominates what you hold."
            )
        # The engine's own render repeats the risk-share sentence written just above; keep the
        # rest (beta shift, diversification, closest existing holding).
        lines.extend(line for line in impact.render() if "of total portfolio risk" not in line)
        hedge = _hedge_line(impact.beta_after)
        if hedge:
            lines.append(hedge)
    if report.risk_after is not None and not standalone:
        parts = []
        for item in sorted(report.risk_after.contributions, key=lambda c: -c.contribution)[:4]:
            parts.append(f"{item.symbol.removesuffix('USDT')} {item.weight:.0%} weight / "
                         f"{item.contribution / report.risk_after.volatility:.0%} risk")
        lines.append("After the trade: " + "; ".join(parts) + ".")
    for outcome in report.stress:
        if outcome.portfolio_move_pct is not None and outcome.shock in (
            "benchmark -5%", "benchmark -10%"
        ):
            lines.append(
                f"If QQQ falls {outcome.shock.split('-')[-1]}, "
                + ("this position" if standalone else "the book")
                + f" moves about {outcome.portfolio_move_pct:+.1f}% through beta alone."
            )
    worst = report.worst.render().replace("[stress] ", "Realised worst case: ")
    if standalone:
        worst = worst.replace("this book", "this position")
    lines.append(worst)
    return lines


def _skill_calls(calls: Sequence[tuple[str, dict[str, Any]]],
                 timeout: int = 10) -> list[tuple[Any, str]]:
    """Several `bitget-signal` Skill calls at once, in order. Each gets its own session, because
    the transport is a stateful JSON-RPC session and sharing one across threads would interleave
    requests on it."""
    from argus.market.evidence import BitgetSkillSource

    def one(call: tuple[str, dict[str, Any]]) -> tuple[Any, str]:
        return BitgetSkillSource().call(call[0], call[1], timeout=timeout)

    with ThreadPoolExecutor(max_workers=len(calls)) as pool:
        return list(pool.map(one, calls))


def _technicals(symbol: str) -> tuple[list[str], list[Source]]:
    """RSI, MACD, support/resistance and ATR as `bitget-signal` reports them, and what they say
    together. Nothing here is recomputed: the figures are the Skill's, and each is sourced to it."""
    calls = [("technical_analysis", {"action": a, "symbol": symbol})
             for a in ("rsi", "macd", "support_resistance", "atr")]
    results = dict(zip(("rsi", "macd", "sr", "atr"), _skill_calls(calls), strict=True))
    lines: list[str] = []
    sources: list[Source] = []
    reading: list[str] = []

    rsi, status = results["rsi"]
    if isinstance(rsi, dict) and rsi.get("rsi") is not None:
        value = float(rsi["rsi"])
        state = ("overbought" if value >= 70 else "oversold" if value <= 30 else "neutral")
        lines.append(f"RSI({rsi.get('period', 14)}, {rsi.get('timeframe', '')}) {value:.1f} — "
                     f"{state}.")
        sources.append(Source(kind="venue", ref="bitget-signal technical_analysis.rsi",
                              detail=status))
        if state != "neutral":
            reading.append(f"RSI is {state}")
    from argus.market.skills import indicators, macd_fields

    macd, status = results["macd"]
    if isinstance(macd, dict) and macd.get("histogram") is not None:
        mine = indicators(symbol)
        checked = macd_fields(macd, None if mine is None else float(mine["dea"]))
        if checked is None:
            lines.append(f"MACD line {float(macd['macd']):.3f}; its signal line could not be "
                         f"checked against Bitget's candles just now, so it is not quoted.")
        else:
            line, hist, swapped = checked
            cross = (str(mine["cross"]) if swapped and mine is not None else
                     str(macd.get("cross") or "").replace("_", " "))
            lines.append(f"MACD {float(macd['macd']):.3f} vs signal {line:.3f} (histogram "
                         f"{hist:+.3f})" + (f", {cross}" if cross else "") + ".")
            if swapped:
                lines.append("Corrected: bitget-signal returns MACD's signal line and histogram "
                             "in each other's fields; recomputed from Bitget's 4h candles, the "
                             "figures above are the right way round and its cross flag is "
                             "replaced.")
            reading.append("momentum turning up" if hist > 0 else "momentum turning down")
            sources.append(Source(kind="venue", ref="bitget-signal technical_analysis.macd",
                                  detail=status + ("; signal/histogram swapped by the Skill, "
                                                   "corrected against Bitget 4h candles"
                                                   if swapped else "; checked against Bitget "
                                                   "4h candles")))
    sr, status = results["sr"]
    if isinstance(sr, dict) and sr.get("current_price") is not None:
        price = float(sr["current_price"])
        above = [float(x) for x in sr.get("resistances") or [] if float(x) > price]
        below = [float(x) for x in sr.get("supports") or [] if float(x) < price]
        parts = []
        if above:
            r = min(above)
            parts.append(f"nearest resistance {r:g} ({(r / price - 1) * 100:.1f}% above)")
            if (r / price - 1) * 100 < 1.0:
                reading.append(f"price is within 1% of resistance at {r:g}")
        if below:
            sp = max(below)
            parts.append(f"nearest support {sp:g} ({(1 - sp / price) * 100:.1f}% below)")
        lines.append(f"Price {price:g}: " + ("; ".join(parts) if parts else
                                              "no support or resistance level within range") + ".")
        sources.append(Source(kind="venue",
                              ref="bitget-signal technical_analysis.support_resistance",
                              detail=status))
    atr, status = results["atr"]
    if isinstance(atr, dict) and atr.get("atr") is not None:
        extra = f"; suggested stop distance {atr['stop_distance']}" if atr.get(
            "stop_distance") is not None else ""
        lines.append(f"ATR {float(atr['atr']):.3f} ({atr.get('timeframe', '')}){extra}.")
        sources.append(Source(kind="venue", ref="bitget-signal technical_analysis.atr",
                              detail=status))
    if lines:
        lines.insert(0, "Actionable: " + (
            "; ".join(reading) if reading else "no technical extreme — the setup is neutral"
        ) + ".")
        lines.append("Caveat: technicals describe the tape, not an edge — the systematic signals "
                     "this desk tested on these names did not clear costs.")
    return lines, sources


def _session_line(symbol: str, *, anchor_open: bool, us_listed: bool) -> str:
    """What the clock means for this contract's price — which depends on what it tracks.

    The twelve rTokens and every other US-listed stock have a US anchor whose hours are modelled
    (`argus.truth.clocks`). A crypto contract has no anchor at all. Gold, oil, FX, index products
    and Asian equities have anchors on other clocks that this console does not model, and saying
    "the US market is shut" about gold would be a confident sentence about the wrong market.
    ``us_listed`` is whether `bitget-mcp-server` recognised the underlying as a US ticker.
    """
    from argus.market import universe

    contract = universe.contracts().get(symbol)
    if symbol in TRADED_SYMBOLS or (us_listed and universe.is_equity(symbol)):
        return ("The US anchor market is open, so this price has price discovery behind it."
                if anchor_open else
                "The US anchor market is shut: this rToken is trading without its anchor, so its "
                "price is being discovered on a thinner book and can gap at the open.")
    if contract is not None and not contract.rwa:
        return ("A crypto contract: it trades around the clock on its own market, with no "
                "off-chain anchor to gap against.")
    return (f"{_t(symbol)} tracks an off-chain market on its own trading hours, which this console "
            f"does not model — check that market's session before reading a quiet-hours price as "
            f"real price discovery.")


PREMIUM_SANITY_BPS = 500.0


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
    lines.insert(0, "Actionable: " + "; ".join(reading) + ".")
    lines.append("Caveat: technicals describe the tape, not an edge — the systematic signals this "
                 "desk tested on these names did not clear costs.")
    return lines, [Source(kind="computation", ref="argus.lui.research._indicators",
                          detail=f"{symbol} 4H, {int(float(got['bars']))} bars from Bitget")]


def _premium_line(symbol: str, rtoken_last: Decimal,
                  anchor_open: bool) -> tuple[str, Source] | None:
    """The rToken's premium or discount to the stock it tracks, from `bitget-mcp-server`'s quote
    of the underlying. The number that makes an rToken different from its stock, stated with the
    clock: while the anchor is shut the stock's price is its last close, so the gap is partly the
    overnight move the rToken has priced and the stock has not."""
    from argus.market import universe
    from argus.market.bitget_mcp import BitgetDataService

    if symbol not in TRADED_SYMBOLS and not universe.is_equity(symbol):
        return None
    try:
        stock = BitgetDataService().quote(_t(symbol))
        last = float(stock.get("last_price") or 0)
    except Exception:
        return None
    if last <= 0:
        return None
    premium = (float(rtoken_last) / last - 1.0) * 10_000
    if abs(premium) > PREMIUM_SANITY_BPS:
        # A tokenised share trades within basis points of its stock. A gap this wide means the
        # two tickers are not the same company, or one price is bad — not a premium to report.
        return None
    clock = ("both live" if anchor_open else
             "the stock's price is its last close, so this includes the move since")
    return (
        f"Versus the stock: {_t(symbol)} {last:g} → the rToken trades at a "
        f"{abs(premium):.1f}bps {'premium' if premium >= 0 else 'discount'} ({clock}).",
        Source(kind="venue", ref="bitget-mcp-server quote", detail=f"{_t(symbol)} "
               f"last {last:g}"),
    )


EARNINGS_NEAR_DAYS = 7
"""A report this close is the first thing a position has to price, so it leads the answer."""

CONSENSUS_MAX_AGE_DAYS = 120
"""An analyst consensus older than this is withheld and its age stated. Measured on 2026-09-23:
`bitget-mcp-server` returned NVDA's consensus scraped 2024-11-21, for fiscal 2025 — quoting it as
current would have put a two-year-old EPS estimate in front of a trader as today's expectation."""


PRICE_TARGET_WINDOW_DAYS = 90
"""Analyst targets older than this are left out of the summary: a target is a view at a date, and
a 2016 target on NVDA says nothing about today. The window keeps each firm's latest view only."""


def _price_target_line(ticker: str, rows: Sequence[Mapping[str, Any]], stock: Any,
                       today: Any) -> str | None:
    """Where analysts' current price targets sit against the stock, from the last 90 days only,
    one target per firm (its latest). The consensus row the same server returns is often years
    old and is withheld; these rows are dated, so they can be used."""
    from datetime import date, timedelta

    from argus.market.bitget_mcp import RATING_STANCE

    cutoff = today - timedelta(days=PRICE_TARGET_WINDOW_DAYS)
    latest: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        try:
            when = date.fromisoformat(str(row.get("published_date"))[:10])
        except ValueError:
            continue
        firm = str(row.get("analyst_firm") or row.get("rating_org") or "")
        if when < cutoff or not firm or row.get("price_target") is None:
            continue
        if firm not in latest or str(row.get("published_date")) > str(
                latest[firm].get("published_date")):
            latest[firm] = row
    if not latest:
        return None
    targets = sorted(float(v["price_target"]) for v in latest.values())
    mid = targets[len(targets) // 2] if len(targets) % 2 else (
        targets[len(targets) // 2 - 1] + targets[len(targets) // 2]) / 2
    stances = [RATING_STANCE.get(str(v.get("latest_rating_cn") or v.get("rating_current")))
               for v in latest.values()]
    raised = sum(1 for v in latest.values() if v.get("price_target_previous") is not None
                 and float(v["price_target"]) > float(v["price_target_previous"]))
    cut = sum(1 for v in latest.values() if v.get("price_target_previous") is not None
              and float(v["price_target"]) < float(v["price_target_previous"]))
    last = float(stock.get("last_price") or 0) if isinstance(stock, dict) else 0.0
    upside = f", {(mid / last - 1) * 100:+.0f}% from the stock's {last:g}" if last > 0 else ""
    return (
        f"Analyst price targets, last {PRICE_TARGET_WINDOW_DAYS} days ({len(latest)} firms, each "
        f"firm's latest): median {mid:g}, range {targets[0]:g} to {targets[-1]:g}{upside}; "
        f"{stances.count('buy')} buy, {stances.count('hold')} hold, {stances.count('sell')} sell; "
        f"{raised} raised and {cut} cut their target."
    )


def _earnings_surprise(ticker: str) -> tuple[str, Source] | None:
    """The last quarter's earnings surprise from the company's own SEC filings, as SUE.

    Standardized Unexpected Earnings — the year-over-year EPS change over the company's own
    trailing volatility of that change — is the formula `research/sue.py` reproduces to
    floating-point identity against QuantConnect's reference, on point-in-time, restatement-aware
    XBRL facts (`market/fundamentals.py`). The desk already reads it; this puts it in front of the
    trader who asks about earnings. No return-predictiveness is claimed: the desk's own PEAD study
    (`research/pead_study.py`) is what would license that, and the sentence stays descriptive.
    """
    from argus.market.fundamentals import FundamentalsSource
    from argus.research.sue import MIN_QUARTERS
    from argus.research.sue import read as sue_read

    facts, _ = FundamentalsSource().facts(ticker, concept="eps_diluted", as_of=datetime.now(UTC))
    if len(facts) < MIN_QUARTERS:
        return None
    window = facts[:MIN_QUARTERS]
    sue = sue_read(ticker, [f.value for f in window])
    size = ("a large" if abs(sue.sue) >= 2 else "a moderate" if abs(sue.sue) >= 1 else "a small")
    direction = "beat" if sue.sue > 0 else "shortfall" if sue.sue < 0 else "in-line print"
    return (
        f"Earnings surprise: the quarter ending {window[0].end.isoformat()} was {size} {direction} "
        f"— SUE {sue.sue:+.2f}, i.e. diluted EPS changed {sue.eps_change:+.2f} year over year "
        f"against a usual swing of {sue.eps_std:.2f} (SEC filing, filed "
        f"{window[0].filed.isoformat()}).",
        Source(kind="computation", ref="argus.research.sue via SEC XBRL",
               detail=f"{ticker} eps_diluted, {sue.quarters_used} quarters"),
    )


def _fundamentals(symbol: str) -> tuple[list[str], list[Source]]:
    from datetime import date

    from argus.market import universe
    from argus.market.bitget_mcp import BitgetDataService

    ticker = _t(symbol)
    if symbol not in TRADED_SYMBOLS and not universe.is_equity(symbol):
        kind = universe.NOT_EQUITY.get(symbol, "crypto")
        what = {"commodity": "a commodity", "fx": "a currency pair", "index": "an index product",
                "crypto": "a crypto contract"}[kind]
        return ([f"Actionable: {ticker} is {what}, so there is no earnings calendar, analyst "
                 f"consensus or 13F filing to report — ask for its technicals, a quote, or what "
                 f"it does to your book instead."], [])
    lines: list[str] = []
    sources: list[Source] = []
    today = datetime.now(UTC).date()

    # Five independent lookups, run side by side: one after another they took ~6s. Each gets its
    # own client, because the data server is a stateful JSON-RPC session and sharing one across
    # threads would interleave requests on it (the fault found in the Skill client, see
    # `argus.market.evidence`).
    def ask(method: str) -> Any:
        return getattr(BitgetDataService(), method)(ticker)

    with ThreadPoolExecutor(max_workers=5) as pool:
        pending = {name: pool.submit(ask, name) for name in
                   ("next_earnings", "consensus", "institutional_holdings", "quote",
                    "price_targets")}
        pending["surprise"] = pool.submit(_earnings_surprise, ticker)

    def safe(name: str) -> Any:
        try:
            return pending[name].result()
        except Exception:
            return None

    earnings = safe("next_earnings")
    if isinstance(earnings, dict) and earnings.get("report_date"):
        when = date.fromisoformat(str(earnings["report_date"])[:10])
        period = earnings.get("period_ending")
        session = {"盘后": "after the close", "盘前": "before the open"}.get(
            str(earnings.get("is_trading_time") or ""), "")
        if when >= today:
            days = (when - today).days
            lines.append(f"Next report: {when.isoformat()}{', ' + session if session else ''} — "
                         f"{days} day(s) away (period ending {period}).")
            if days <= EARNINGS_NEAR_DAYS:
                lines.insert(0, f"Actionable: {ticker} reports in {days} day(s) — an rToken "
                                f"position held through it carries the earnings gap, and the "
                                f"rToken can reprice before the stock does.")
            else:
                lines.insert(0, f"Actionable: no {ticker} report inside {EARNINGS_NEAR_DAYS} "
                                f"days, so a position opened now does not carry an earnings gap "
                                f"this week.")
        else:
            likely = when + timedelta(days=91)
            lines.append(f"Most recent report on file: {when.isoformat()} (period ending "
                         f"{period}); the source has not yet published the next date.")
            lines.insert(0, f"Actionable: {ticker}'s next report date is not published yet. It "
                            f"reports quarterly, so expect it around {likely:%d %b} — an "
                            f"estimate, not a date; confirm it before holding a position into "
                            f"that window.")
        sources.append(Source(kind="venue", ref="bitget-mcp-server equity_calendar",
                              detail=f"{ticker} report {when.isoformat()}"))

    consensus = safe("consensus")
    if isinstance(consensus, dict) and consensus.get("fore_mean") is not None:
        scraped = str(consensus.get("scraped_date") or "")
        try:
            age = (today - date.fromisoformat(scraped[:10])).days
        except ValueError:
            age = None
        if age is not None and age <= CONSENSUS_MAX_AGE_DAYS:
            lines.append(
                f"Analyst consensus ({consensus.get('fore_indicator_name')}, "
                f"{consensus.get('fore_org_num')} analysts): {consensus['fore_mean']} "
                f"(range {consensus.get('min_fore_value')} to {consensus.get('high_fore_value')})."
            )
        else:
            lines.append(
                f"The analyst consensus the source holds was scraped {scraped or 'on an unknown '}"
                f"{'' if scraped else 'date'} — too old to present as today's expectation, so "
                f"it is withheld rather than quoted."
            )
        sources.append(Source(kind="venue", ref="bitget-mcp-server consensus",
                              detail=f"scraped {scraped}"))

    holders = safe("institutional_holdings")
    if isinstance(holders, list) and holders:
        latest = max(str(h.get("period_ending") or "") for h in holders)
        filed = [h for h in holders if str(h.get("period_ending") or "") == latest]
        top = sorted(filed, key=lambda h: -float(h.get("principal_amount") or 0))[:3]
        names = ", ".join(f"{h.get('org_name')} ({float(h.get('principal_amount') or 0):,.0f} sh)"
                          for h in top)
        count = (f"{len(filed)} 13F records" if len(filed) > 1 else "one 13F record")
        lines.append(f"Institutional holders: the source returned {count} for the period ending "
                     f"{latest} — {names}. A sample of filers, not a full ownership table.")
        sources.append(Source(kind="venue", ref="bitget-mcp-server 13F holdings",
                              detail=f"{ticker} period {latest}"))

    targets = safe("price_targets")
    target_line = _price_target_line(ticker, targets, safe("quote"), today) if isinstance(
        targets, list) else None
    if target_line is not None:
        lines.append(target_line)
        sources.append(Source(kind="venue", ref="bitget-mcp-server equity_estimates_price_target",
                              detail=f"{ticker} analyst targets, last "
                                     f"{PRICE_TARGET_WINDOW_DAYS} days"))

    surprise = safe("surprise")
    if surprise is not None:
        lines.append(surprise[0])
        sources.append(surprise[1])

    stock = safe("quote")
    if isinstance(stock, dict) and stock.get("total_market_cap"):
        cap = float(stock["total_market_cap"])
        pb = stock.get("pb")
        lines.append(f"{ticker} market cap ${cap / 1e9:,.0f}bn"
                     + (f", price-to-book {float(pb):.1f}" if pb else "") + ".")
        sources.append(Source(kind="venue", ref="bitget-mcp-server quote",
                              detail=f"{ticker} fundamentals snapshot"))
    if not lines:
        lines.append(f"bitget-mcp-server covers US stocks and ETFs, and returned nothing for "
                     f"{ticker} just now — a non-US listing has no US filings to report.")
    return lines, sources


ANALOGUE_HORIZON_BARS = 24
ANALOGUE_DAYS = 90
"""Bitget's history endpoint documents a 90-day maximum range, so this is as far back as the venue
lets the search look. Thirty days — the default for everything else — gave three or four
independent episodes, too few to call anything a base rate."""

MIN_INDEPENDENT_EPISODES = 8
"""Below this many independent episodes the answer says the base rate is thin, however many raw
matches there are. `desk/analogue` already collapses overlapping windows and reports the effective
count; this is where the console acts on it. Measured on the first live run: 50 raw matches for
COIN were 4 independent episodes, and the answer led with a 78% hit rate as though it were one."""


def _analogue(symbol: str, data: MarketData) -> tuple[list[str], list[Source], dict[str, Any]]:

    from argus.desk.analogue import corpus_from_closes, current_state, find

    series = sorted(data.raw.get(symbol, {}).items())
    span = max(1, (series[-1][0] - series[0][0]).days) if series else 0
    closes: list[tuple[datetime, float]] = []
    level = 100.0
    for stamp, ret in series:
        level *= 1.0 + ret
        closes.append((stamp, level))
    query = current_state(closes)
    corpus = corpus_from_closes(closes, symbol=symbol, horizon=ANALOGUE_HORIZON_BARS)
    if query is None or not corpus:
        return ([f"Not enough {_t(symbol)} history to describe the current state."], [], {})
    as_of = closes[-1][0] if closes[-1][0].tzinfo else closes[-1][0].replace(tzinfo=UTC)
    report = find(query=query, corpus=corpus, as_of=as_of)
    lines = [
        f"Now: {_t(symbol)} is {query['trailing_return']:+.0f}bps over the last 24h with hourly "
        f"volatility of {query['volatility_bps']:.0f}bps."
    ]
    dist = report.distribution
    if not report.usable or dist is None:
        lines.append(f"Refused to generalise: {report.refused}. Too few comparable past states "
                     f"for a distribution — an anecdote with error bars is not an answer.")
    elif dist.effective_n < MIN_INDEPENDENT_EPISODES:
        lines.insert(0, (
            f"Actionable: treat this as thin — the {dist.count} past states that resemble now come "
            f"from only {dist.effective_n} independent episodes in the last {span} days, "
            f"too few for a base rate. For what it is worth, the next 24h median was "
            f"{dist.median:+.0f}bps and it rose {dist.hit_rate:.0%} of the time."
        ))
    else:
        lines.insert(0, (
            f"Actionable: in {dist.count} comparable past states ({dist.effective_n} independent "
            f"episodes over {span} days), the next 24h median was {dist.median:+.0f}bps "
            f"and it rose {dist.hit_rate:.0%} of the time — against a ~12bps round trip."
        ))
        lines.append(f"Spread of outcomes: 25th percentile {dist.quantile(25):+.0f}bps, 75th "
                     f"{dist.quantile(75):+.0f}bps — the range, not the median, is the risk.")
    return lines, [Source(kind="computation", ref="argus.desk.analogue.find",
                          detail="state = trailing 24h return + realised vol; outcome = next "
                                 "24h; overlapping episodes collapsed")], report.as_dict()


ANALOGUE_OLDER_WAIT_S = 9.0
"""How long an analogue answer waits for the oldest stretch of history (55 to 90 days back), which
only history-candles serves, 100 bars a page. If it has not arrived, the answer runs on the 55 days
it has and says so rather than holding the visitor for the last few pages."""


def _analogue_data(symbol: str) -> MarketData:
    """Ninety days of hourly bars for one name, fetched side by side.

    Three stretches, each from the endpoint that serves it fastest: the latest 1,000 bars and the
    window behind them back to day 55 from the recent-candles endpoint (one call each), and days 55
    to 90 from history-candles. For the twelve rTokens the oldest stretch comes from the frozen
    history file instead, which is legitimate here in a way it is not for the risk answers: a
    closed hourly candle never changes, so extending a live series backwards with closed candles
    recorded earlier is the same series, not a mix of two. Every join must be contiguous — a gap
    would put a fake return across it — or the older side is dropped and the answer says how many
    days it covers. If the live call fails, the answer is frozen and labelled that way, exactly as
    :func:`load` does.
    """
    from concurrent.futures import wait

    from argus.market.history import (
        RECENT_REACH_DAYS,
        CandleType,
        fetch,
        fetch_window,
    )

    frozen: list[tuple[datetime, float]] = []
    frozen_on = "an unrecorded date"
    try:
        fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        frozen = [(datetime.fromisoformat(ts), float(close))
                  for ts, close in fixture.get("candles", {}).get(symbol, [])]
        frozen_on = str(fixture.get("generated_at", frozen_on))[:10]
    except (OSError, ValueError):
        frozen = []

    now = datetime.now(UTC)
    reach = now - timedelta(days=RECENT_REACH_DAYS)
    oldest = now - timedelta(days=ANALOGUE_DAYS)

    def recent() -> list[tuple[datetime, float]]:
        with _FETCH_SLOTS:
            bars = fetch(symbol, interval="1H", candle_type=CandleType.MARKET, recent=True,
                         limit=1000)
        return [(c.ts, float(c.close)) for c in bars]

    def behind(until: datetime) -> list[tuple[datetime, float]]:
        with _FETCH_SLOTS:
            bars = fetch(symbol, interval="1H", candle_type=CandleType.MARKET, recent=True,
                         limit=1000, start=reach, end=until)
        return [(c.ts, float(c.close)) for c in bars]

    def older() -> list[tuple[datetime, float]]:
        with _FETCH_SLOTS:
            # Overlap the window it meets by a day: the two endpoints disagree by an hour or
            # so about where a boundary bar falls, and a gap there would drop this stretch.
            bars = fetch_window(symbol, start=oldest, end=reach + timedelta(days=1),
                                interval="1H", candle_type=CandleType.MARKET, pause=0.1)
        return [(c.ts, float(c.close)) for c in bars]

    def joined(early: list[tuple[datetime, float]],
               late: list[tuple[datetime, float]]) -> list[tuple[datetime, float]] | None:
        """``early`` then ``late`` if they meet within an hour, else None."""
        before = [p for p in early if p[0] < late[0][0]]
        if before and late[0][0] - before[-1][0] <= timedelta(hours=1):
            return [*before, *late]
        return None

    pool = ThreadPoolExecutor(max_workers=3)
    try:
        latest = pool.submit(recent)
        from_file = bool(frozen) and frozen[0][0] <= oldest + timedelta(days=1)
        paged = None if from_file else pool.submit(older)
        try:
            live = latest.result(timeout=FETCH_DEADLINE_S)
            if len(live) < 50:
                raise PortfolioError(f"{symbol}: only {len(live)} bars came back")
        except Exception as exc:
            if not frozen:
                raise
            cutoff = frozen[-1][0] - timedelta(days=ANALOGUE_DAYS)
            return MarketData(
                raw={symbol: returns([p for p in frozen if p[0] >= cutoff])}, live=False,
                provenance=(f"Bitget hourly candles frozen on {frozen_on} (the live fetch did "
                            f"not complete: {type(exc).__name__}) — the figures are real, but not "
                            f"as of this minute"),
                source=Source(kind="computation", ref="data/risk_layer_candles_fixture.json",
                              detail=f"real Bitget history frozen {frozen_on}; {ANALOGUE_DAYS}d"),
            )
        closes = live
        extended = ""
        if from_file:
            longer = joined(frozen, closes)
            if longer is not None:
                closes, extended = longer, f"closed candles recorded on {frozen_on}"
        else:
            try:
                middle = pool.submit(behind, live[0][0]).result(timeout=FETCH_DEADLINE_S)
                longer = joined(middle, closes) if middle else None
                if longer is not None:
                    closes = longer
            except Exception:
                pass
            if paged is not None:
                done, _ = wait([paged], timeout=ANALOGUE_OLDER_WAIT_S)
                try:
                    early = paged.result() if done else []
                except Exception:
                    early = []
                longer = joined(early, closes) if early else None
                if longer is not None:
                    closes, extended = longer, "history-candles"
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    cutoff = closes[-1][0] - timedelta(days=ANALOGUE_DAYS)
    closes = [p for p in closes if p[0] >= cutoff]
    span = (closes[-1][0] - closes[0][0]).days
    live_days = (live[-1][0] - live[0][0]).days
    if extended.startswith("closed candles"):
        provenance = (f"live Bitget hourly candles for the last {live_days} days, extended back "
                      f"to {span} days with {extended}")
    else:
        provenance = f"live Bitget hourly candles, last {span} days" + (
            "" if span >= ANALOGUE_DAYS - 1 else
            f" (the venue's older history did not arrive in time, so the search covers {span} "
            f"days rather than {ANALOGUE_DAYS})")
    return MarketData(
        raw={symbol: returns(closes)}, live=True, provenance=provenance,
        source=Source(kind="venue", ref="bitget /api/v3/market/candles + history-candles",
                      detail=f"{symbol}; 1H; {len(closes)} bars over {span} days"),
    )


def _no_candles(provenance: str, ref: str, detail: str) -> MarketData:
    return MarketData(raw={}, live=True, provenance=provenance,
                      source=Source(kind="venue", ref=ref, detail=detail))


_NO_CANDLES: dict[ResearchKind, MarketData] = {
    ResearchKind.EXECUTION: _no_candles("Bitget live 24h ticker",
                                        "bitget /api/v2/mix/market/tickers", "24h volume"),
    ResearchKind.QUOTE: _no_candles("Bitget live ticker, plus the stock's quote from "
                                    "bitget-mcp-server", "bitget /api/v2/mix/market/tickers",
                                    "last, bid, ask, 24h range, funding"),
    ResearchKind.TECHNICALS: _no_candles("Bitget's bitget-signal technical-analysis Skill, live",
                                         "bitget-signal technical_analysis",
                                         "rsi, macd, support/resistance, atr"),
    ResearchKind.FUNDAMENTALS: _no_candles("Bitget's bitget-mcp-server (US equity data) and the "
                                           "company's SEC filings, live",
                                           "bitget-mcp-server", "earnings calendar, consensus, "
                                           "13F, quote"),
}
"""The kinds answered without thirty days of candles, each with the source it really uses — the
data line once said "Bitget live 24h ticker" under a technical-analysis answer."""


def run(raw_text: str, request: ResearchRequest, *, ledger: Any = None) -> Answer:
    """Answer a research request from the desk's engines. Refuses by name rather than guessing."""
    question = _question(raw_text, request)
    if not request.symbols:
        return Answer(
            question=question, refused=True,
            reason="the scenario needs your holdings, and none were named",
            lines=[
                "I can run that the moment I know what you hold — say it with weights, e.g. "
                "\"what if the Nasdaq drops 10%? I hold 50% NVDA, 30% MSFT, 20% AAPL\".",
                "Any contract Bitget lists can be named — the twelve rTokens the desk trades, "
                "other US stocks and ETFs, gold, oil, index products and crypto.",
            ],
        )
    try:
        data = (_NO_CANDLES[request.kind] if request.kind in _NO_CANDLES
                else _analogue_data(request.symbols[0])
                if request.kind is ResearchKind.ANALOGUE
                else load(request.symbols))
    except Exception as exc:
        return Answer(
            question=question, refused=True,
            reason=f"market data for {', '.join(request.symbols)} could not be loaded "
                   f"({type(exc).__name__})",
            lines=[f"I could not load market data for {', '.join(request.symbols)} just now, live "
                   f"or frozen, so I will not guess at the risk. Try again in a minute."],
        )
    is_open = _is_open()
    lines: list[str] = []
    sources: list[Source] = [data.source]
    payload: dict[str, Any] = {"request": request.as_dict(), "live_data": data.live}

    try:
        if request.kind is ResearchKind.IMPACT:
            add = request.symbols[0]
            before = {s: w for s, w in request.book.items()}
            if add in before and len(before) == 1:
                before = {}
            report = copilot(add=add, before=before, size=request.size, raw=data.raw,
                             benchmark=BENCHMARK, is_open=is_open)
            columns = _open_columns(data.raw, is_open)
            lines = _impact_lines(report, request, columns)
            payload["report"] = report.as_dict()
            sources.append(Source(kind="computation", ref="argus.desk.portfolio.copilot",
                                  detail="session beta, Euler risk decomposition, beta stress, "
                                         "realised worst 24h window"))
            if not before:
                profile = _distribution_line(add, data.raw, columns)
                if profile is not None:
                    lines.append(profile)
            desk_lines, desk_sources = _desk_view(add, ledger)
            lines.extend(desk_lines)
            sources.extend(desk_sources)

        elif request.kind is ResearchKind.STRESS:
            columns = _open_columns(data.raw, is_open)
            shocks = [Shock("benchmark -5%", -5.0), Shock("benchmark -10%", -10.0)]
            if request.shock_pct is not None and request.shock_pct not in (-5.0, -10.0):
                shocks.insert(0, Shock(f"benchmark {request.shock_pct:+g}%", request.shock_pct))
            outcomes = stress_by_beta(weights=request.book, columns=columns,
                                      benchmark=columns[BENCHMARK], shocks=shocks)
            for outcome in outcomes:
                if outcome.portfolio_move_pct is None:
                    lines.append(f"{outcome.shock}: unavailable — {outcome.reason}")
                    continue
                worst = outcome.worst_position
                lines.append(
                    f"If QQQ moves {outcome.shock.removeprefix('benchmark ')}: your book moves "
                    f"about {outcome.portfolio_move_pct:+.2f}%"
                    + (f", hardest hit {worst[0].removesuffix('USDT')} {worst[1]:+.2f}%"
                       if worst else "") + " (market-driven part only, through each beta)."
                )
            book_beta = 0.0
            driven: dict[str, float] = {}
            for symbol, weight in request.book.items():
                symbol_beta = beta(columns.get(symbol, []), columns[BENCHMARK])
                if symbol_beta is not None:
                    book_beta += weight * symbol_beta
                    driven[symbol] = weight * symbol_beta
            if len(driven) > 1 and book_beta > 0:
                # The holding to name is the one carrying the most loss RELATIVE to its weight —
                # the biggest position is usually the biggest contributor and naming it tells the
                # trader nothing. A live answer named QQQ (60% of the book, 52% of the loss) while
                # COIN carried 48% of the loss on 40% of the weight.
                top = max(driven, key=lambda s: driven[s] / book_beta - request.book[s])
                share = driven[top] / book_beta
                if share - request.book[top] >= 0.02:
                    lines.append(
                        f"Actionable: {_t(top)} is {request.book[top]:.0%} of the book but "
                        f"{share:.0%} of its market-driven loss in a sell-off — trimming it cuts "
                        f"the drawdown fastest; the QQQ hedge below is the other lever."
                    )
                else:
                    lines.append(
                        "Actionable: the sell-off loss is spread roughly in line with your "
                        "weights, so trimming any one name barely helps — the QQQ hedge below is "
                        "the lever."
                    )
            hedge = _hedge_line(book_beta)
            if hedge:
                lines.append(hedge)
            add = next(iter(request.book))
            report = copilot(add=add, before={s: w for s, w in request.book.items() if s != add}
                             or {}, size=request.book[add] if len(request.book) > 1 else 1.0,
                             raw=data.raw, benchmark=BENCHMARK, is_open=is_open)
            lines.append(report.worst.render().replace(
                "[stress] ", "What actually happened, not a model — "))
            payload["stress"] = [o.as_dict() for o in outcomes]
            payload["worst_window"] = report.worst.as_dict()
            sources.append(Source(kind="computation", ref="argus.desk.portfolio.stress_by_beta",
                                  detail="beta-propagated shock + realised worst window"))

        elif request.kind is ResearchKind.COMPARE:
            columns = _open_columns(data.raw, is_open)
            rows: list[dict[str, Any]] = []
            for symbol in request.symbols:
                rep = copilot(add=symbol, before={}, size=1.0, raw=data.raw,
                              benchmark=BENCHMARK, is_open=is_open)
                betas = {str(k): v.beta for k, v in rep.session_beta.items()}
                ten = next((o.portfolio_move_pct for o in rep.stress
                            if o.shock == "benchmark -10%"), None)
                hourly = list(data.raw.get(symbol, {}).values())
                var = variance(hourly)
                # Annualised on the hours an rToken actually trades: 24 x 365, not 252 sessions.
                vol = None if var is None else math.sqrt(var) * math.sqrt(24 * 365)
                rows.append({"symbol": symbol, "beta_open": betas.get("open"),
                             "beta_shut": betas.get("shut"), "qqq_minus_10": ten,
                             "worst_24h": rep.worst.move_pct, "realised_vol": vol})
            for row in sorted(rows, key=lambda r: -(r["realised_vol"] or 0.0)):
                vol_text = ("n/a" if row["realised_vol"] is None
                            else f"{row['realised_vol']:.0%}")
                lines.append(
                    f"{row['symbol'].removesuffix('USDT')}: realised vol {vol_text} a year; beta "
                    f"{row['beta_open'] or 0:.2f} open / {row['beta_shut'] or 0:.2f} shut; QQQ "
                    f"-10% implies {(row['qqq_minus_10'] or 0):+.1f}%; worst realised day "
                    f"{(row['worst_24h'] or 0):+.1f}%."
                )
            a, b = request.symbols[0], request.symbols[1]
            rho = correlation(columns.get(a, []), columns.get(b, []))
            if rho is not None:
                read = ("mostly the same bet" if abs(rho) >= 0.7 else
                        "a genuinely different bet" if abs(rho) <= 0.3 else "partly the same bet")
                lines.insert(0, f"{a.removesuffix('USDT')} and {b.removesuffix('USDT')} move "
                                f"together at {rho:+.2f} in the open session — {read}.")
            riskiest = max(rows, key=lambda r: abs(r["beta_open"] or 0))
            wildest = max(rows, key=lambda r: r["realised_vol"] or 0.0)
            actionable = (
                f"Actionable: {_t(riskiest['symbol'])} carries the most market risk per dollar of "
                f"the {len(rows)}"
            )
            if wildest["symbol"] != riskiest["symbol"]:
                actionable += (
                    f", but {_t(wildest['symbol'])} swings the most on its own — most of its risk "
                    f"is its own news, which a QQQ hedge will not touch"
                )
            lines.insert(1, actionable + ". Listed from most to least volatile.")
            payload["compare"] = rows
            sources.append(Source(kind="computation", ref="argus.desk.portfolio",
                                  detail="session betas, beta stress, realised worst window"))

        elif request.kind is ResearchKind.QUOTE:
            from argus.cost.model import CostModel
            from argus.market.bitget import fetch_tickers

            try:
                tickers = fetch_tickers()
            except Exception as exc:
                return Answer(
                    question=question, refused=True,
                    reason=f"the venue did not answer ({type(exc).__name__})",
                    lines=["Bitget did not return a quote just now, and a stale price presented "
                           "as current is worse than none. Try again in a minute."],
                )
            now = datetime.now().astimezone()
            anchor_open = is_open(now)
            fee = CostModel.bitget_perp().round_trip_bps()
            quoted = []
            for symbol in request.symbols:
                ticker = tickers.get(symbol)
                if ticker is None:
                    lines.append(f"{_t(symbol)}: no quote returned by Bitget just now.")
                    continue
                change = float(ticker.change_24h) * 100.0
                funding = float(ticker.funding_rate) * 100.0
                volume = ticker.base_volume * ticker.last
                lines.append(
                    f"{_t(symbol)} last {ticker.last} USDT on Bitget ({change:+.2f}% over 24h); "
                    f"bid {ticker.bid} / ask {ticker.ask}, spread {ticker.spread_bps:.1f}bps; "
                    f"24h range {ticker.low_24h} to {ticker.high_24h}; funding {funding:+.4f}% per "
                    f"interval; 24h volume about ${volume:,.0f}."
                )
                quoted.append((symbol, ticker))
            if quoted:
                symbol, ticker = quoted[0]
                all_in = fee + ticker.spread_bps
                lines.insert(0, (
                    f"Actionable: a round trip in {_t(symbol)} costs about {all_in:.1f}bps "
                    f"({fee:.0f}bps taker fees + the {ticker.spread_bps:.1f}bps spread) — a trade "
                    f"needs a move bigger than that just to break even."
                ))
                premium = _premium_line(symbol, ticker.last, anchor_open)
                if premium is None:
                    data = _no_candles("Bitget live ticker", "bitget /api/v2/mix/market/tickers",
                                       "last, bid, ask, 24h range, funding")
                lines.append(_session_line(symbol, anchor_open=anchor_open,
                                           us_listed=premium is not None))
                if premium is not None:
                    lines.append(premium[0])
                    sources.append(premium[1])
                stamp = quoted[0][1].fetched_at.strftime("%Y-%m-%d %H:%M:%S UTC")
                lines.append(f"Quoted {stamp}.")
            payload["quotes"] = {
                s: {"last": str(t.last), "bid": str(t.bid), "ask": str(t.ask),
                    "change_24h": str(t.change_24h), "funding_rate": str(t.funding_rate),
                    "fetched_at": t.fetched_at.isoformat()}
                for s, t in quoted
            }
            sources.append(Source(kind="computation", ref="argus.cost.model.CostModel.bitget_perp",
                                  detail="round-trip taker fee"))

        elif request.kind is ResearchKind.TECHNICALS:
            lines, extra = _technicals(request.symbols[0])
            sources.extend(extra)
            if not lines:
                # The Skill has no series for some listed contracts (CVXSTOCKUSDT, SP500USDT —
                # "No OHLCV data", measured 2026-09-23). The same indicators are computed from
                # Bitget's own 4h candles instead, and the answer says whose numbers they are.
                lines, extra = _technicals_computed(request.symbols[0])
                sources.extend(extra)
                if lines:
                    data = _no_candles("RSI, MACD and ATR computed by ARGUS from Bitget's live 4h "
                                       "candles, because bitget-signal has no series for this "
                                       "contract", "bitget /api/v3/market/candles",
                                       "4H; Wilder RSI(14), MACD(12,26,9), Wilder ATR(14)")
            if not lines:
                return Answer(
                    question=question, refused=True,
                    reason="bitget-signal's technical-analysis Skill did not answer and Bitget's "
                           "candles could not be read",
                    lines=["Neither Bitget's technical-analysis Skill nor Bitget's own candles "
                           "answered just now, so there is nothing to read the tape from. Try "
                           "again shortly."],
                )

        elif request.kind is ResearchKind.FUNDAMENTALS:
            lines, extra = _fundamentals(request.symbols[0])
            sources.extend(extra)
            if not extra:
                data = _no_candles("Bitget's contract list, which flags this contract as not a "
                                   "company's shares — no equity data was requested",
                                   "bitget /api/v2/mix/market/contracts", "isRwa + asset class")

        elif request.kind is ResearchKind.ANALOGUE:
            lines, extra, report_dict = _analogue(request.symbols[0], data)
            sources.extend(extra)
            payload["analogue"] = report_dict

        elif request.kind is ResearchKind.EXECUTION:
            from argus.desk.workbench import plan_execution
            from argus.market.bitget import fetch_tickers

            symbol = request.symbols[0]
            assert request.notional is not None
            adv: Decimal | None = None
            try:
                ticker = fetch_tickers().get(symbol)
                if ticker is not None:
                    adv = ticker.base_volume * ticker.last
            except Exception:
                adv = None
            asleep = not is_open(datetime.now().astimezone())
            if adv is None or adv <= 0:
                return Answer(
                    question=question, refused=True,
                    reason="the venue's 24h volume could not be read, and a plan without it "
                           "would state a participation rate it does not know",
                    lines=[f"I could not read {symbol}'s 24h volume from Bitget just now, and "
                           f"splitting an order without it means guessing its footprint. Try "
                           f"again in a minute."],
                )
            plan = plan_execution(symbol=symbol, notional=request.notional, adv_notional=adv,
                                  urgency="high" if request.urgent else "low",
                                  anchor_asleep=asleep)
            lines = [
                f"A ${request.notional:,.0f} order is {plan.participation_rate:.2%} of "
                f"{symbol.removesuffix('USDT')}'s 24h volume (${adv:,.0f}); expected all-in "
                f"cost about {plan.expected_total_cost_bps:.1f}bps.",
                f"Plan: {plan.rationale}.",
                *[f"Slice {s.index}: {s.fraction:.0%} as {s.style}, ~{s.expected_cost_bps:.1f}bps"
                  for s in plan.slices],
            ]
            payload["execution"] = {
                "participation": str(plan.participation_rate),
                "expected_cost_bps": str(plan.expected_total_cost_bps),
                "slices": [{"fraction": str(s.fraction), "style": s.style,
                            "cost_bps": str(s.expected_cost_bps)} for s in plan.slices],
            }
            sources.append(Source(kind="computation", ref="argus.desk.workbench.plan_execution",
                                  detail="superlinear impact; passive slices quoted at taker"))
    except PortfolioError as exc:
        return Answer(question=question, refused=True, reason=str(exc),
                      lines=[f"The risk engine refused this one: {exc}"], sources=[data.source])

    # The conclusion a trader acts on leads; the evidence follows it. Engine lines are written
    # lower-case for the CLI's indented layout, so they are sentence-cased for reading here.
    lines = [_clean(line) for line in lines]
    lines = [line[:1].upper() + line[1:] for line in lines]
    lines.sort(key=lambda line: 0 if line.startswith("Actionable:") else 1)
    # A "read as" note names both contracts on purpose (CVXSTOCKUSDT vs CVXUSDT); stripping the
    # suffixes there would make the two identical and the note meaningless.
    lines.extend(f"Assumed: {note if ' read as ' in note else _clean(note)}."
                 for note in request.notes)
    lines.append(f"Data: {data.provenance}. This is analysis, not advice — you make the call.")
    return Answer(question=question, lines=lines, sources=sources, data=payload)


__all__ = [
    "BENCHMARK",
    "DEFAULT_SIZE",
    "PLANNER_PROMPT",
    "RISK_BUDGET",
    "MarketData",
    "ResearchKind",
    "ResearchRequest",
    "about_the_record",
    "detect",
    "load",
    "max_size_within_budget",
    "parse_book",
    "parse_budget",
    "plan_with_model",
    "run",
    "with_book",
]
