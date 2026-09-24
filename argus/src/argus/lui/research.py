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

import itertools
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
    worst_window,
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

    HEDGE = "hedge"
    """Which instrument to hedge a book with, measured: for each candidate (index, sector or crypto
    perpetual) the share of the book's variance a beta-sized short removes, and what it costs to
    enter on today's order book and to hold through funding. The funding-aware leg choice beat
    crypto_sor's entry-only router on five live leg pairs (`eval/execution_comparison.py`); until
    this kind it was reachable only from the desk's record, never from a question."""

    VENUE = "venue"
    """Bitget's tokenised-stock offering itself, from live data: what it lists by kind, what trading
    it costs, what holding it costs in funding, and how far a token trades from its stock right now
    — and the honest comparison with owning the share. "What is Bitget's tokenized stock offering
    and should I use it instead of buying stocks directly?" hit a pronoun parser bug before."""

    CONSTRUCT = "construct"
    """Build a book: the names given, or a theme's names, weighted so each carries an equal share of
    the risk (correlation included), with the resulting book's volatility, beta, stress and worst
    realised day. "Build me a portfolio of tech stocks" was refused before this kind."""

    LEVERAGE = "leverage"
    """What leverage does to a position in this name: the adverse move that liquidates it, how often
    the name has moved that far against the side within a day, its worst such day, the largest
    leverage that would have survived it, and what funding costs at that leverage. Before this kind
    "explain the risk in shorting DOGE with 20x" got a generic sizing line (a judge's probe)."""

    MACRO = "macro"
    """The rates, Fed, inflation and dollar backdrop from FRED's public series, and how tech (or a
    named name) has actually traded against long bonds and the dollar this month on Bitget's own
    contracts. Bitget's macro Skill returns empty fields today (measured 2026-09-24), so the
    series are read from the Federal Reserve Bank of St. Louis directly and sourced as such."""

    SENTIMENT = "sentiment"
    """The crypto fear & greed index and its week, from alternative.me — the source Bitget's own
    `sentiment_index` Skill wraps and today fails to reach — beside BTC and ETH funding as a read
    of positioning."""

    NEWS = "news"
    """What is being said about a name now, and why it moved: its 24-hour move split into the part
    the market explains (beta times QQQ's move) and the part that is its own, beside the headlines
    that name it and any SEC filing this week. Before this kind "what's the news on NVDA today?" was
    answered "no decision in the record matches that" and "why did the market drop today?" with the
    session clock (a judge's live probe, 2026-09-24). Attribution, not causation, and said so."""

    BOOK = "book"
    """The book the trader already holds, on its own: where its risk sits, what a sell-off does to
    it, its worst realised day, and which holding to trim to bring every name inside the risk
    budget. Before this kind "I hold 60% BTC, 30% ETH, 10% SOL — how risky is my portfolio?" was
    answered as a proposal to add *another* 20% of BTC (fourth blind corpus and a judge's live
    probe, 2026-09-24)."""

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
    leverage: float | None = None
    side: str = "long"
    cash: float = 0.0
    """Fraction of the book's value held as cash or stablecoins. It carries no risk and is not a
    position, so it dilutes every risk figure rather than being scaled away."""

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
            "cash": self.cash,
            "leverage": self.leverage,
            "side": self.side,
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
    r"(?:about|of))\b|"
    # Chinese has no word boundaries: 值得买 (worth buying), 能买吗 (can I buy), 该不该买 (should I
    # buy), 风险大吗 (is it risky) — "英伟达现在值得买吗?" reached the session clock before.
    r"值得(?:买|入手|投资)|能不能买|能买吗|该不该(?:买|卖)|要不要(?:买|卖)|风险(?:大|高)吗|"
    r"可以买吗",
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
    r"\b(?:split|slice|execute|execution|work(?:ing)?\s+(?:an?\s+)?(?:\w+\s+){0,2}order|fill|"
    r"slippage|without\s+moving\s+(?:the\s+)?(?:price|market)|"
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
    r"\b(?:earn\w*|reports?\s+(?:next|on|when)|when\s+does\s+\w+\s+report|eps|guidance|"
    r"(?:beat|miss(?:ed)?)\b[^?]{0,30}\b(?:quarter|q[1-4]|estimates?|expectations?|consensus|"
    r"street|numbers)|did\s+\w+\s+(?:beat|miss)|quarterly\s+results|last\s+quarter|"
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


_NEWS = re.compile(
    r"\b(?:news|headlines?|catalysts?|press\s+release|8-k|what'?s\s+(?:going\s+on|happening)\s+"
    r"with|what\s+happened\s+(?:to|with)\s+\w+\s+(?:today|this\s+week|yesterday)|why\s+(?:is|did|"
    r"has|was|are|were)\s+(?:\S+\s+){1,4}?(?:drop|fall|fell|dump|crash|tank|rall|jump|pump|"
    r"surg|spik|soar|sink|slid|slump|plung|mov|up|down|red|green)\w*)",
    re.I,
)
"""News and "why did it move" questions."""
_VENUE = re.compile(
    r"\bbitget'?s?\b[^?]{0,40}\b(?:tokeni[sz]ed|rtokens?|stock\s+(?:offering|perps?|futures|"
    r"contracts?|tokens?)|offering)|\brtokens?\b[^?]{0,40}\b(?:what|how|work|instead|versus|vs)\b|"
    r"\b(?:tokeni[sz]ed|rtoken)\s+stocks?\b[^?]{0,40}\b(?:instead|versus|vs|or)\b",
    re.I,
)
_CONSTRUCT = re.compile(
    r"\b(?:build|construct|design|create|make|suggest|give\s+me|put\s+together)\s+(?:me\s+)?(?:an?\s+)?"
    r"(?:\w+\s+){0,3}(?:portfolio|book|basket|allocation)|how\s+(?:should|would|do)\s+i\s+"
    r"(?:allocate|weight|divide)\b",
    re.I,
)
THEMES: dict[str, tuple[str, ...]] = {
    "semis": ("NVDAUSDT", "AMDUSDT", "AVGOUSDT", "MUUSDT", "TSMUSDT"),
    "magnificent": ("AAPLUSDT", "MSFTUSDT", "NVDAUSDT", "GOOGLUSDT", "AMZNUSDT", "METAUSDT",
                    "TSLAUSDT"),
    "tech": ("NVDAUSDT", "AAPLUSDT", "MSFTUSDT", "METAUSDT", "GOOGLUSDT", "AMZNUSDT"),
    "crypto": ("BTCUSDT", "ETHUSDT", "SOLUSDT"),
    "commodit": ("XAUUSDT", "XAGUSDT", "CLUSDT", "COPPERUSDT"),
    "diversif": ("QQQUSDT", "XAUUSDT", "BTCUSDT", "CLUSDT"),
    "defensive": ("SPYUSDT", "XAUUSDT", "KOUSDT", "WMTUSDT"),
}
"""Theme words to their names on Bitget, the ones a trader means by the word. Checked in order, so
"semis" and "magnificent 7" win over the broader "tech"."""
_THEME_WORDS = re.compile(r"\b(semi\w*|chip\w*|magnificent|mag\s*7|tech\w*|crypto\w*|coins?|"
                          r"commodit\w*|diversif\w*|all[\s-]weather|conservative|defensive|"
                          r"low[\s-]risk|safe\w*|cautious)\b", re.I)


def _theme(text: str) -> tuple[str, tuple[str, ...]] | None:
    match = _THEME_WORDS.search(text)
    if match is None:
        return None
    word = match.group(1).lower()
    key = ("semis" if word.startswith(("semi", "chip")) else
           "magnificent" if word.startswith(("magnificent", "mag")) else
           "crypto" if word.startswith(("crypto", "coin")) else
           "diversif" if word.startswith(("diversif", "all")) else
           "defensive" if word.startswith(("conservative", "defensive", "low", "safe",
                                           "cautious")) else
           "commodit" if word.startswith("commodit") else "tech")
    return key, THEMES[key]


_OUTLOOK = re.compile(r"\b(?:outlook|next\s+(?:week|month|quarter))\b", re.I)
"""A soft question about what lies ahead — answered with base rates, labelled as not a forecast."""
_PRICE_FORECAST = re.compile(
    r"\b(?:forecast\w*|predict\w*|price\s+target|where\s+will\s+\w+\s+(?:be|go|trade|close)|"
    r"give\s+me\s+a\s+number|how\s+(?:high|low|far)\s+will|kitna\s+hoga|gonna\s+moon)\b", re.I)
"""A request for a price at a future time. Refused — the blind corpora mark these must-refuse, and a
number here would be the one thing in the console not computed from data."""
_LEVERAGE = re.compile(r"\b(\d+(?:\.\d+)?)\s*x\b|\bleverage\w*|\bliquidat\w*|\bmargin\b", re.I)
_SHORT = re.compile(r"\bshort\w*\b|\bsell(?:ing)?\s+short\b|\bbearish\s+bet\b", re.I)
_MACRO = re.compile(
    r"\b(?:macro\w*|fed|fomc|federal\s+reserve|powell|interest\s+rates?|rate\s+(?:cuts?|hikes?)|"
    r"yields?|treasur\w*|10[\s-]?y(?:ea)?r|2[\s-]?y(?:ea)?r|inflation|cpi|dollar|dxy|recession|"
    r"bond\s+market)\b",
    re.I,
)
_CRYPTO_WORD = re.compile(r"\b(?:crypto\w*|coins?|bitcoin)\b", re.I)
_DOLLAR_FOCUS = re.compile(r"\b(?:dollar|dxy|usd|greenback)\b", re.I)

_SENTIMENT = re.compile(
    r"\b(?:fear\s*(?:&|and)?\s*greed|sentiment|market\s+mood|fomo|euphori\w*|"
    r"crowded|crowding|positioning|how\s+(?:bullish|bearish|greedy|fearful)\s+is|"
    r"(?:is|are)\s+(?:people|traders|everyone|the\s+market)\s+(?:bullish|bearish)|"
    r"(?:bullish|bearish)\s+(?:on|about)|overheat\w*|froth\w*)\b",
    re.I,
)
_MARKET_WIDE = re.compile(r"\b(?:the\s+market|markets|stocks|equities|nasdaq|s&p|wall\s+street|"
                          r"tech\s+stocks|crypto\s+market)\b", re.I)
_BOOK_RISK = re.compile(
    r"\b(?:how\s+risky\s+is\s+my|risk\s+(?:of|in|on)\s+my|my\s+(?:portfolio|book|holdings)\s+"
    r"(?:risk|look|safe)|rebalanc\w*|diversif\w*|concentrat\w*|review\s+my\s+(?:portfolio|"
    r"book|holdings|positions)|is\s+my\s+(?:portfolio|book)\s+(?:safe|ok|okay|too))\b",
    re.I,
)
"""A question about the book already held, with no new name being added."""
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
    from argus.market.universe import CJK_ALIASES

    trust = not _shouting(text)
    found: dict[str, str] = {}
    for name, symbol in sorted(CJK_ALIASES.items(), key=lambda kv: text.find(kv[0])):
        if name in text and symbol not in found:
            found[symbol] = ""
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


_DEPTH = re.compile(
    r"\border\s*-?\s*book\b|\b(?:market\s+)?depth\b|\bliquid(?:ity)?\b|\bhow\s+(?:thin|deep)\b",
    re.I,
)
""""What is the order book depth on NVDA?" and "show me the order book for NVDA" reached the
decision log and the Sharpe answer (a judge's probe, 2026-09-24); the order book is read by the
execution plan, which walks it."""

_HEDGE = re.compile(
    r"\bhedg\w*\b|\bprotect\s+(?:my|the|this)\s+(?:book|portfolio|downside|positions?)\b",
    re.I,
)
HEDGE_HOLDING_DAYS = 7
HEDGE_BOOK_VALUE = Decimal("100000")
HEDGE_CANDIDATES_EQUITY = ("QQQUSDT", "SPYUSDT", "SMHUSDT")
HEDGE_CANDIDATES_CRYPTO = ("BTCUSDT", "ETHUSDT")

_ORDER_WORDS = re.compile(
    r"\b(?:orders?|slippage|twap|vwap|clips?|block\s+trade|market\s+or\s+limit|limit\s+or\s+market|"
    r"(?:split|slic)\w*\s+(?:it|up|(?:an?|the|my|this|that)\s+(?:\w+\s+){0,3}"
    r"(?:order|trade|buy|sell|position|purchase|exit))|"
    r"minimi[sz]e\s+(?:the\s+)?(?:impact|slippage|cost))\b",
    re.I,
)
"""Words that make an execution-shaped question about working an order, as opposed to "how should
I buy NVDA", which asks whether to, not how to."""

_ORDER_STATUS = re.compile(
    r"\b(?:did|has|have|is|was|were)\b[^?]{0,30}\b(?:orders?|trades?)\b[^?]{0,20}"
    r"\b(?:fill\w*|execut\w*|go(?:ne)?\s+through|done|placed)\b|\border\s+status\b|"
    r"\b(?:was|were|did|had)\b[^?]{0,40}\b(?:slippage|execution|fill)|"
    r"\blast\s+(?:\w+\s+){0,2}(?:execution|fill|order|trade)s?\b|"
    r"\b(?:the\s+desk|you)\s+(?:ran|made|executed|placed|filled|did)\b",
    re.I,
)
""""Did my order fill?" and "what was the slippage on the last GOOGL execution the desk ran?" ask
about orders that exist, which is the record's to answer, not a plan's."""

EXECUTION_EXAMPLE_SYMBOL = "NVDAUSDT"
EXECUTION_DEFAULT_NOTIONAL = Decimal("50000")
EXECUTION_LARGE_NOTIONAL = Decimal("250000")
EXECUTION_BOOK_VALUE = Decimal("100000")
_LARGE_ORDER = re.compile(r"\b(?:large|big|huge|block|size(?:able)?|whale)\b", re.I)


def _is_an_order(raw: str) -> bool:
    """An instruction to trade ("sell 10 ETH at 5000") is refused as an order by the console; it
    must never be answered as a question about how to trade."""
    from argus.lui.question import _INTERROGATIVE, _ORDER_VERB

    return bool(_ORDER_VERB.match(raw)) and not _INTERROGATIVE.match(raw)


def _execution_request(raw: str, symbols: tuple[str, ...], *, urgent: bool,
                       notes: tuple[str, ...] = ()) -> ResearchRequest:
    """An execution plan, with any missing instrument or size defaulted and the default said."""
    stated = list(notes)
    notional = _parse_notional(raw)
    if notional is None:
        notional = (EXECUTION_LARGE_NOTIONAL if _LARGE_ORDER.search(raw)
                    else EXECUTION_DEFAULT_NOTIONAL)
        stated.append(f"no size was stated, so ${notional:,.0f} is worked — give the size you "
                      f"have in mind for its own plan")
    if not symbols:
        symbols = (EXECUTION_EXAMPLE_SYMBOL,)
        stated.append("no instrument was named, so NVDA is the worked example — name yours")
    return ResearchRequest(kind=ResearchKind.EXECUTION, symbols=symbols[:1], notional=notional,
                           urgent=urgent, notes=tuple(stated))


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
    r"stood|abstain\w*)\b|\bdecision\s+\d+\b|"
    # "why did" is the desk's own record — unless what follows is a price moving. "Why did the
    # market drop today?" is news, and was answered with the session clock because this bare
    # alternative claimed every "why did" (a judge's live probe, 2026-09-24).
    r"\bwhy\s+did\b(?![^?]*\b(?:drop|fall|fell|dump|crash|tank|rall|jump|pump|surg|spik|soar|"
    r"sink|slid|slump|plung|mov|go\s+(?:up|down)|went\s+(?:up|down))\w*)",
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


def _forecast_note(text: str) -> tuple[str, ...]:
    """A question that asks what WILL happen is answered with the backdrop it lands in, and says
    so — "what will the S&P do after the FOMC" gets the rates, not a prediction."""
    if _FORECAST.search(text) or re.search(r"\bwill\b[^?]*\b(?:do|be|go|react|move)\b", text, re.I):
        return ("a forecast was asked for; this console does not forecast prices, so this is the "
                "backdrop the event lands in, measured, not a prediction",)
    return ()


_CJK = re.compile(r"[\u4e00-\u9fff]")
_CJK_KINDS: tuple[tuple[re.Pattern[str], ResearchKind], ...] = (
    (re.compile(r"财报|业绩|盈利|每股收益|分析师|目标价|季报|营收"), ResearchKind.FUNDAMENTALS),
    (re.compile(r"超买|超卖|技术面|均线|支撑|阻力|RSI|MACD", re.I), ResearchKind.TECHNICALS),
    (re.compile(r"情绪|恐慌|贪婪|拥挤"), ResearchKind.SENTIMENT),
    (re.compile(r"新闻|消息|为什么(?:跌|涨)|怎么(?:跌|涨)"), ResearchKind.NEWS),
    (re.compile(r"盘口|深度|拆单|滑点|大单"), ResearchKind.EXECUTION),
    (re.compile(r"价格|多少钱|报价|现价"), ResearchKind.QUOTE),
)
"""Chinese questions name the research kind in a handful of words. "英伟达下个季度财报什么时候公布"
(when does Nvidia report next?) reached an unrelated decision (a judge's probe, 2026-09-24)."""


def _cjk_request(raw: str, symbols: tuple[str, ...]) -> ResearchRequest | None:
    if not symbols or not _CJK.search(raw):
        return None
    for pattern, kind in _CJK_KINDS:
        if pattern.search(raw):
            if kind is ResearchKind.EXECUTION:
                return _execution_request(raw, symbols, urgent=False)
            return ResearchRequest(kind=kind, symbols=symbols[:1])
    return None


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
    cjk = _cjk_request(raw, symbols)
    if cjk is not None:
        return cjk
    if _VENUE.search(raw) and not about_the_record(raw):
        return ResearchRequest(kind=ResearchKind.VENUE, symbols=symbols[:1] or ("NVDAUSDT",),
                               notes=() if symbols else ("NVDA used as the worked example",))
    if _CONSTRUCT.search(raw) and not about_the_record(raw) and not _EXECUTION.search(raw):
        named = symbols if len(symbols) >= 2 else ()
        theme = None if named else _theme(raw)
        if named or theme:
            chosen = named or (theme[1] if theme else ())
            return ResearchRequest(
                kind=ResearchKind.CONSTRUCT, symbols=tuple(chosen),
                notes=() if named else (f"the {theme[0] if theme else ''} theme read as "
                                        + ", ".join(_t(s) for s in chosen),))
    if (_HEDGE.search(raw) and not about_the_record(raw) and not _is_an_order(raw)
            and not _COMPARE.search(raw) and not _ADD_VERB.search(raw)
            and not (_STRESS.search(raw) or _STRESS_BARE.search(raw))
            and not re.search(r"\bhedged\s+with\b|\bas\s+a\s+hedge\b", raw, re.I)):
        # A hedge already chosen ("adding SQQQ as a hedge", "long NVDA hedged with SQQQ") is a
        # question about that position or scenario, answered by the impact and stress engines.
        hedge_pairs = _pairs(raw)
        hedge_book: dict[str, float] = {}
        for _, symbol, weight in hedge_pairs:
            hedge_book[symbol] = hedge_book.get(symbol, 0.0) + weight
        hedge_notes: list[str] = []
        if hedge_book:
            hedge_book = _normalise(hedge_book, hedge_notes)
        elif symbols:
            hedge_book = {symbols[0]: 1.0}
        elif _CRYPTO_WORD.search(raw):
            hedge_book = {"BTCUSDT": 1.0}
            hedge_notes.append("crypto read as bitcoin")
        notional = _parse_notional(raw)
        return ResearchRequest(kind=ResearchKind.HEDGE, symbols=tuple(hedge_book),
                               book=hedge_book, notional=notional,
                               notes=(*hedge_notes, *_forecast_note(raw)))
    if _SENTIMENT.search(raw) and not about_the_record(raw):
        # "What is the sentiment on COIN right now?" was answered "no open positions" (a judge's
        # probe, 2026-09-24): a named contract gets its own positioning beside the backdrop.
        return ResearchRequest(kind=ResearchKind.SENTIMENT, symbols=symbols[:1])
    if (_MACRO.search(raw) and not about_the_record(raw)
            and not (_STRESS.search(raw) or _STRESS_BARE.search(raw))
            and not _FUNDAMENTALS.search(raw)):
        macro_pairs = _pairs(raw)
        if macro_pairs:
            # "I hold 40% NVDA, 30% MSFT, 30% GOOGL — what does a Fed rate cut do to my book?"
            # went to the decision log (a judge's probe, 2026-09-24): a macro question with a book
            # in it was not a macro question to the patterns. It is the macro question asked of
            # that book.
            macro_book: dict[str, float] = {}
            for _, symbol, weight in macro_pairs:
                macro_book[symbol] = macro_book.get(symbol, 0.0) + weight
            book_notes: list[str] = []
            macro_book = _normalise(macro_book, book_notes)
            return ResearchRequest(kind=ResearchKind.MACRO, symbols=tuple(macro_book),
                                   book=macro_book,
                                   notes=(*_forecast_note(raw), *book_notes))
        if not symbols and _CRYPTO_WORD.search(raw):
            return ResearchRequest(kind=ResearchKind.MACRO, symbols=("BTCUSDT",),
                                   notes=(*_forecast_note(raw), "crypto read as bitcoin"))
        return ResearchRequest(kind=ResearchKind.MACRO, symbols=symbols[:1],
                               notes=_forecast_note(raw))
    if _NEWS.search(raw) and not symbols and _MARKET_WIDE.search(raw) and not about_the_record(raw):
        return ResearchRequest(kind=ResearchKind.NEWS, symbols=(BENCHMARK,),
                               notes=("the market read as the Nasdaq-100 through QQQ",))
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
        if (_EXECUTION.search(raw) and _ORDER_WORDS.search(raw) and not about_the_record(raw)
                and not _ORDER_STATUS.search(raw) and not _is_an_order(raw)):
            # "How should I split a large sell order to minimise slippage?" names no instrument
            # and was refused, though it is the Execution Assistance sub-theme's own question
            # (a judge's probe, 2026-09-24). The plan is worked on a stated example instead.
            return _execution_request(raw, (), urgent=bool(_URGENT.search(raw)))
        return None
    # Questions about what the desk did are the ledger's, whatever else they mention.
    if about_the_record(raw):
        return None

    notes: list[str] = []
    pairs = _pairs(raw)
    urgent = bool(_URGENT.search(raw))

    if (pairs and _EXECUTION.search(raw) and _ORDER_WORDS.search(raw) and not _is_an_order(raw)
            and not _ORDER_STATUS.search(raw) and _parse_notional(raw) is None):
        # "I hold 50% NVDA and 50% AAPL — how should I split a 20% TSLA order?" sizes the order as
        # a share of a book whose value is not stated. It read as a stress test. The order is the
        # name given a weight last; its size is that share of a stated-default book.
        _, target, weight = pairs[-1]
        order_value = (EXECUTION_BOOK_VALUE * Decimal(str(weight))).quantize(Decimal("1"))
        return ResearchRequest(
            kind=ResearchKind.EXECUTION, symbols=(target,), notional=order_value, urgent=urgent,
            notes=(f"a {weight:.0%} order is sized on a ${EXECUTION_BOOK_VALUE:,.0f} book, since "
                   f"the book's value was not given — say it in dollars for your own plan",))

    simple = (not _ADD_VERB.search(raw) and not _STRESS.search(raw) and not pairs
              and not _EXECUTION.search(raw))
    if symbols and not pairs and _OUTLOOK.search(raw) and not _PRICE_FORECAST.search(raw):
        # "What is the outlook for gold over the next month?" and "forecast BTC for next week" were
        # refused as questions about the desk's record. A forecast is not something this console
        # makes; what it can say is what followed the name's similar past states — base rates,
        # labelled as such.
        return ResearchRequest(kind=ResearchKind.ANALOGUE, symbols=symbols[:1],
                               notes=_forecast_note(raw) or ("a forecast was asked for; this is "
                                                             "what followed similar past states, "
                                                             "not a prediction",))
    lever = _LEVERAGE.search(raw)
    if lever and symbols and not pairs:
        amount = re.search(r"\b(\d+(?:\.\d+)?)\s*x\b", raw, re.I)
        return ResearchRequest(
            kind=ResearchKind.LEVERAGE, symbols=symbols[:1],
            leverage=float(amount.group(1)) if amount else None,
            side="short" if _SHORT.search(raw) else "long",
            notes=() if amount else ("no leverage was stated, so 10x is assessed — say the "
                                     "multiple you have in mind",),
        )
    if _DEPTH.search(raw) and not _ORDER_STATUS.search(raw):
        request = _execution_request(raw, symbols, urgent=urgent, notes=tuple(notes))
        read_at = request.notional or EXECUTION_DEFAULT_NOTIONAL
        return replace(request, notes=tuple(
            f"the order book is read for a ${read_at:,.0f} order — name your size for its own plan"
            if n.startswith("no size was stated") else n for n in request.notes))
    if simple and _NEWS.search(raw) and not _ANALOGUE.search(raw):
        return ResearchRequest(kind=ResearchKind.NEWS, symbols=symbols[:1])
    if simple and _ANALOGUE.search(raw):
        return ResearchRequest(kind=ResearchKind.ANALOGUE, symbols=symbols[:1])
    if simple and _FUNDAMENTALS.search(raw):
        if _PRICE_TARGET.search(raw) and not (symbols[0] in TRADED_SYMBOLS
                                              or _is_equity(symbols[0])):
            return None  # a price target for crypto or a commodity is a forecast; refused
        return ResearchRequest(kind=ResearchKind.FUNDAMENTALS, symbols=symbols[:2])
    if simple and _TECHNICALS.search(raw):
        return ResearchRequest(kind=ResearchKind.TECHNICALS, symbols=symbols[:1])
    if simple and _QUOTE.search(raw):
        if _FORECAST.search(raw):
            return None  # a price asked for a future time is a forecast; refused, not quoted
        return ResearchRequest(kind=ResearchKind.QUOTE, symbols=symbols[:4])

    if _EXECUTION.search(raw) and not _ORDER_STATUS.search(raw) and (
            _parse_notional(raw) is not None
            or (_ORDER_WORDS.search(raw) and not pairs and not _is_an_order(raw))):
        # With no size stated, "how do I split an order for BTC" used to fall through to the
        # portfolio-impact answer. It is an execution question; the size is defaulted and said.
        return _execution_request(raw, symbols, urgent=urgent, notes=tuple(notes))

    add_match = _ADD_VERB.search(raw)
    holdings_match = _HOLDINGS.search(raw)
    if pairs and not add_match and _BOOK_RISK.search(raw) and not (
            _STRESS.search(raw) or _STRESS_BARE.search(raw)):
        owned: dict[str, float] = {}
        for _, symbol, weight in pairs:
            owned[symbol] = owned.get(symbol, 0.0) + weight
        owned = _normalise(owned, notes)
        if owned:
            return ResearchRequest(kind=ResearchKind.BOOK, symbols=tuple(owned), book=owned,
                                   notes=tuple(notes))
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


def _states_holdings(question: str) -> bool:
    """Does the question itself say what the trader holds? A weight given to the name being added
    ("adding 20% TSLA") is the size of the trade, not a holding."""
    if _HOLDINGS.search(question):
        return True
    pairs = _pairs(question)
    verb = _ADD_VERB.search(question)
    if verb is None:
        return bool(pairs)
    return any(position < verb.start() for position, _, _ in pairs)


def with_book(request: ResearchRequest | None, book_text: str,
              question: str = "") -> ResearchRequest | None:
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
    if request.book and question and not _states_holdings(question):
        # The question states no holdings, so a book on the request was supplied by the model, not
        # the trader. In one of three identical calls it read "what does adding 20% TSLA do to my
        # risk" as a book of TSLA alone and answered "100% of your risk" (a judge's probe,
        # 2026-09-24). The saved book is the trader's own statement and it wins.
        request = replace(request, book={})
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
    if request.kind in (ResearchKind.STRESS, ResearchKind.BOOK):
        return replace(request, book=book, symbols=tuple(book), notes=(*kept, note))
    if request.kind is ResearchKind.MACRO and not request.symbols:
        return replace(request, book=book, symbols=tuple(book), notes=(*kept, note))
    if request.kind is ResearchKind.HEDGE and not request.symbols:
        return replace(request, book=book, symbols=tuple(book), notes=(*kept, note))
    return request


_FOLLOW_UP = re.compile(
    r"^\s*(?:and\s+)?(?:what|how)\s+(?:about|abt|bout)\b|^\s*and\s+(?:for\s+)?\S+\s*\??\s*$|"
    r"^\s*(?:same|now|ok(?:ay)?|also)\b[^?]{0,40}\b(?:for|with)\b|^\s*(?:do|try)\s+(?:the\s+)?"
    r"same\b|^\s*(?:swap|switch|replace)\b",
    re.I,
)


def follow_up(text: str, prior: list[str], book_text: str = "") -> ResearchRequest | None:
    """"And what about COIN?" after a research question: the same question, asked of COIN.

    Returned an unrelated session-phase blurb 2 times of 2 (a judge's probe, 2026-09-24). The
    earlier question is re-read from the client's own turn history — the server keeps none — and
    only its instrument is replaced; everything else the trader said still applies."""
    if not prior or len(text) > 80 or not _FOLLOW_UP.search(text):
        return None
    named, _ = research_symbols(text)
    if not named:
        return None
    for earlier in reversed(prior[-4:]):
        base = with_book(detect(earlier), book_text, earlier)
        if base is None:
            continue
        new = named[0]
        if base.kind is ResearchKind.IMPACT and base.book:
            others = tuple(s for s in base.symbols[1:] if s != new)
            symbols: tuple[str, ...] = (new, *others)
        elif base.kind in (ResearchKind.STRESS, ResearchKind.BOOK):
            return None  # a book question does not take a single name
        else:
            symbols = (new, *base.symbols[1:]) if base.kind is ResearchKind.COMPARE else (new,)
        return replace(base, symbols=symbols, notes=(
            *base.notes, f"read as the previous question — \"{earlier[:60]}\" — asked of "
                         f"{_t(new)}"))
    return None


# --- the model fallback ---------------------------------------------------------------------

PLANNER_PROMPT = """You turn a trader's research question into a structured request for a \
portfolio-risk engine. You never answer the question and never state a number of your own; the \
engine computes everything.

Kinds:
- impact: what adding/buying a name does to their risk, whether a name is risky to own, or a \
single name's risk profile (volatility, beta, skew, kurtosis, R-squared)
- stress: what a market (QQQ) move would do to their holdings, worst case
- venue: what Bitget's tokenized stocks (rTokens) are, how they work, or whether to use them \
instead of buying the stock directly
- hedge: how or with what to hedge a holding or a book, or protect it against a fall
- construct: build or allocate a new portfolio from named names or a theme (tech, semis, crypto, \
commodities, diversified); put every name in "names"
- leverage: the risk of a leveraged long or short in a name — liquidation, margin, "20x"
- macro: the rates, Fed, inflation, bond or dollar backdrop and how it affects stocks or a name
- sentiment: the fear & greed index or overall market sentiment
- news: recent news, headlines or filings about a name, or why a name or the market moved today \
(for the market as a whole, name QQQ)
- book: how risky the portfolio they already hold is, or how to rebalance or diversify it — no \
new name being added
- compare: two or more names side by side on risk — volatility, beta, correlation (a comparison \
of earnings, analyst targets or valuation is fundamentals, not compare)
- execution: how to split or execute an order, what an order of some size would cost, or a \
name's order-book depth or liquidity
- quote: where a name trades right now, its price, spread or funding (not order-book depth)
- technicals: RSI, MACD, trend, momentum, support/resistance, overbought/oversold
- fundamentals: earnings dates, analyst estimates or targets, institutional holders, valuation \
(for one or two names; put both in "names" when comparing)
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

Holdings go in the field matching how they were stated, never converted by you: percentages in \
"holdings", dollar amounts in "holdings_usd" ("5k in TSLA" -> {"TSLA": 5000}), quantities in \
"holdings_units" ("2 BTC" -> {"BTC": 2}, "100 shares of AAPL" -> {"AAPL": 100}). Cash and \
stablecoins (USD, USDT, USDC) go in "cash_usd", never in holdings. "A little bit" or "a chunk" \
with no number means size_percent null. Holdings named without weights go in holdings with equal \
percentages.

Reply with only this JSON (fill every field; null where a value was not stated):
{"kind": "...", "candidate": "...", "names": ["TICKER", "TICKER"], "holdings": {"TICKER": 40}, \
"holdings_usd": {"TICKER": 5000}, "holdings_units": {"TICKER": 2}, "cash_usd": 10000, \
"leverage": 20, "side": "long", "size_percent": 10, "shock_percent": -10, "order_usd": 50000, \
"confidence": 0.9, "why": "..."}

"names" lists every instrument the question is about, as tickers, in the order asked — for a \
compare it holds every name being compared."""

MIN_PLAN_CONFIDENCE = 0.6


def _value_holdings(raw: Mapping[str, Any], notes: list[str]) -> tuple[dict[str, float], float]:
    """Holdings the model reported in dollars or units, as fractions of the whole account, and the
    cash fraction. Units are priced at Bitget's live last price; every conversion is written into
    ``notes`` so the reader sees the number it rests on. Empty when nothing was stated that way.

    Found by a judge's probe (2026-09-24): "I have 10k USDT, 5k in TSLA and 2 BTC" was read as
    TSLA 71% / BTC 29% — two bitcoin counted as two dollars, the cash dropped — when BTC is about
    92% of that account at the live price."""
    usd: dict[str, float] = {}
    for field_name in ("holdings_usd", "holdings_units"):
        stated = raw.get(field_name)
        if not isinstance(stated, dict):
            continue
        for name, value in stated.items():
            hit = _resolve(str(name), trust_case=False) or _resolve(str(name).upper())
            try:
                amount = float(value)
            except (TypeError, ValueError):
                continue
            if hit is None or amount <= 0:
                continue
            if field_name == "holdings_units":
                price = _last_price(hit[0])
                if price is None:
                    notes.append(f"no live price for {_t(hit[0])}, so its {amount:g} units were "
                                 f"left out")
                    continue
                notes.append(f"{amount:g} {_t(hit[0])} valued at ${amount * price:,.0f} "
                             f"(Bitget last {price:,.2f})")
                amount *= price
            usd[hit[0]] = usd.get(hit[0], 0.0) + amount
    try:
        cash_usd = max(0.0, float(raw.get("cash_usd") or 0.0))
    except (TypeError, ValueError):
        cash_usd = 0.0
    total = sum(usd.values()) + cash_usd
    if not usd or total <= 0:
        return {}, 0.0
    if cash_usd:
        notes.append(f"${cash_usd:,.0f} held as cash is {cash_usd / total:.0%} of the account and "
                     f"carries no risk")
    return {s: v / total for s, v in usd.items()}, cash_usd / total


def _last_price(symbol: str) -> float | None:
    from argus.market.bitget import fetch_tickers

    try:
        ticker = fetch_tickers().get(symbol)
    except Exception:
        return None
    return None if ticker is None else float(ticker.last)


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
    valued, cash = _value_holdings(raw, notes)
    if valued:
        # Dollar or unit holdings were stated: the book is their live value over the whole
        # account, cash included, and is deliberately NOT rescaled to 100% — cash is part of it.
        book = valued
    else:
        book = _normalise(book, notes)
        cash = 0.0
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
    if kind is ResearchKind.VENUE:
        audit["applied"] = True
        return ResearchRequest(kind=kind, symbols=symbols[:1] or ("NVDAUSDT",),
                               parsed_by="model"), audit
    if kind is ResearchKind.MACRO and len(book) > 1:
        # A macro question asked of a stated book is measured on the book, as the patterns do.
        audit["applied"] = True
        return ResearchRequest(kind=kind, symbols=tuple(book), book=book, parsed_by="model",
                               notes=_forecast_note(text)), audit
    if kind in (ResearchKind.MACRO, ResearchKind.SENTIMENT):
        # The backdrop needs no instrument; a named one is the name to measure against it.
        audit["applied"] = True
        return ResearchRequest(kind=kind, symbols=symbols[:1], parsed_by="model",
                               notes=_forecast_note(text)), audit
    if kind is ResearchKind.HEDGE:
        audit["applied"] = True
        hedged = book or ({symbols[0]: 1.0} if symbols else {})
        return ResearchRequest(kind=kind, symbols=tuple(hedged), book=hedged,
                               notional=_parse_notional(text), parsed_by="model"), audit
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
    if kind is ResearchKind.CONSTRUCT:
        chosen = symbols if len(symbols) >= 2 else ()
        theme = None if chosen else _theme(text)
        chosen = chosen or (theme[1] if theme else ())
        request = None if len(chosen) < 2 else ResearchRequest(
            kind=kind, symbols=tuple(chosen)[:10], parsed_by="model", notes=tuple(notes))
    elif kind is ResearchKind.LEVERAGE:
        try:
            multiple = float(raw.get("leverage")) if raw.get("leverage") else None
        except (TypeError, ValueError):
            multiple = None
        side = "short" if str(raw.get("side") or "").lower() == "short" else "long"
        request = ResearchRequest(kind=kind, symbols=symbols[:1], leverage=multiple, side=side,
                                  parsed_by="model", notes=tuple(notes) if multiple else (
                                      *notes, "no leverage was stated, so 10x is assessed"))
    elif kind is ResearchKind.BOOK:
        request = None if not book else ResearchRequest(
            kind=kind, symbols=tuple(book), book=book, cash=cash, parsed_by="model",
            notes=tuple(notes))
    elif kind is ResearchKind.EXECUTION:
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
    elif kind in (ResearchKind.TECHNICALS, ResearchKind.FUNDAMENTALS, ResearchKind.ANALOGUE,
                  ResearchKind.NEWS):
        # Fundamentals compares two names side by side ("compare AAPL and MSFT fundamentals"
        # answered AAPL alone and dropped MSFT silently); the others read one name.
        width = 2 if kind is ResearchKind.FUNDAMENTALS else 1
        request = ResearchRequest(kind=kind, symbols=symbols[:width], parsed_by="model")
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


def _venue(symbol: str, is_open: Any) -> tuple[list[str], list[Source]]:
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
    lines = [
        f"Bitget lists {sum(kinds.values())} real-world-asset contracts beside {crypto} crypto "
        f"ones: {kinds['equity']} stocks and ETFs, {kinds['commodity']} commodities, "
        f"{kinds['fx']} currency pairs and {kinds['index']} index products — all USDT-margined "
        f"perpetuals that trade around the clock, including when the stock's own market is shut.",
        f"Trading costs {model.taker_bps:.0f}bps a side as taker ({model.round_trip_bps():.0f}bps "
        f"a round trip).",
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
    lines.append("What the token is not: a perpetual contract, not the share — no dividends, no "
                 "vote, and outside US hours its price is discovered on Bitget's own book, not "
                 "the exchange's.")
    lines.insert(0, (
        "Actionable: use the token for exposure outside US market hours, for leverage, or to "
        "hedge a book around the clock; for a buy-and-hold position the share is cheaper"
        + (f" — at today's funding a long pays about {carry:.1f}% a year to hold."
           if carry is not None and carry > 0 else ".")))
    return lines, [Source(kind="venue", ref="bitget contracts + tickers + bitget-mcp-server",
                          detail=f"{len(listed)} contracts; {_t(symbol)} as the worked example")]


def _leverage(symbol: str, multiple: float,
              side: str) -> tuple[list[str], list[Source], dict[str, Any]]:
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
    # Liquidation distance before fees and the maintenance margin, which both bring it closer:
    # at 20x the whole margin is gone after an adverse 5%.
    distance = 1.0 / multiple
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
    survive = int(1 / worst + 1e-9) if worst > 0 else None  # 1/0.10000000000000009 is 9.999…
    verb = "rise" if side == "short" else "fall"
    ticker = _t(symbol)
    lines = [
        f"Actionable: at {multiple:g}x {side} a {distance:.1%} {verb} in {ticker} wipes the margin"
        f" — and {ticker} moved that far against a {side} within 24 hours from {hits / windows:.0%}"
        f" of the hourly entry points in the last {days:.0f} days. Its worst 24 hours against a "
        f"{side} was {worst:.1%}"
        + ("." if not survive else
           f", so {multiple:g}x would have survived every day in the window (anything up to "
           f"{survive}x did)." if multiple <= survive else
           f", which only {survive}x or less would have survived.")
    ]
    try:
        ticker_row = fetch_tickers().get(symbol)
    except Exception:
        ticker_row = None
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
        "The liquidation distance is before fees and Bitget's maintenance margin, both of which "
        "move it closer; hourly highs and lows understate the extremes inside each hour."
    )
    payload = {"leverage": multiple, "side": side, "liquidation_distance": distance,
               "hit_rate_24h": hits / windows if windows else None, "worst_adverse_24h": worst,
               "survivable_leverage": survive}
    return lines, [Source(kind="computation", ref="argus.lui.research._leverage",
                          detail=f"{symbol} 1H highs/lows, {len(bars)} bars")], payload


FRED_SERIES: dict[str, str] = {
    "DGS10": "10-year Treasury", "DGS2": "2-year Treasury", "DFF": "fed funds",
    "T10YIE": "10-year breakeven inflation", "DTWEXBGS": "broad dollar index",
}
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}&cosd={start}"


FRED_SNAPSHOT = Path(__file__).resolve().parents[3] / "data" / "macro_snapshot.json"
"""FRED series as last read by the desk cycle, shipped with the console. FRED did not answer from
the hosted console's network on 2026-09-24 (three macro questions timed out after about 20s), so
the console reads FRED live when it can within :data:`FRED_TIMEOUT_S` and otherwise answers from
this file, dated, and says which it used."""
FRED_TIMEOUT_S = 5.0
FRED_BACKOFF_S = 600.0
_FRED_DOWN_UNTIL = 0.0
_FRED_USED_SNAPSHOT: dict[str, str] = {}


def _fred_live(series: str, days: int) -> list[tuple[str, float]]:
    import urllib.request
    from datetime import timedelta as _td

    start = (datetime.now(UTC) - _td(days=days)).date().isoformat()
    req = urllib.request.Request(FRED_CSV.format(series=series, start=start),
                                 headers={"User-Agent": "argus-research/1.0"})
    with urllib.request.urlopen(req, timeout=FRED_TIMEOUT_S) as resp:
        text = resp.read().decode("utf-8")
    rows: list[tuple[str, float]] = []
    for line in text.splitlines()[1:]:
        day, _, value = line.partition(",")
        try:
            rows.append((day, float(value)))
        except ValueError:
            continue
    return rows


def _fred(series: str, days: int = 45) -> list[tuple[str, float]]:
    """One FRED series as (date, value) rows, oldest first, missing days ('.') dropped — live when
    FRED answers, else from :data:`FRED_SNAPSHOT` (recorded in ``_FRED_USED_SNAPSHOT``)."""
    from datetime import timedelta as _td

    global _FRED_DOWN_UNTIL
    if time.monotonic() >= _FRED_DOWN_UNTIL:
        try:
            rows = _fred_live(series, days)
            if rows:
                return rows
        except Exception:
            # Once FRED has failed, stop waiting on it for a while: every series in one answer
            # would otherwise sit out its own timeout (15-19s per macro answer, measured live).
            _FRED_DOWN_UNTIL = time.monotonic() + FRED_BACKOFF_S
    if not FRED_SNAPSHOT.exists():
        return []
    snap = json.loads(FRED_SNAPSHOT.read_text(encoding="utf-8"))
    cutoff = (datetime.now(UTC) - _td(days=days)).date().isoformat()
    rows = [(d, float(v)) for d, v in snap.get("series", {}).get(series, []) if d >= cutoff]
    if not rows:  # an old snapshot still beats nothing; keep its tail
        rows = [(d, float(v)) for d, v in snap.get("series", {}).get(series, [])][-days:]
    if rows:
        _FRED_USED_SNAPSHOT[series] = str(snap.get("generated_at", ""))[:10]
    return rows


def write_macro_snapshot(days: int = 120) -> int:
    """Read every series the console uses from FRED and write them to :data:`FRED_SNAPSHOT`.
    Run by the desk cycle, so the hosted console's fallback is never more than a cycle old."""
    series = {sid: _fred_live(sid, days) for sid in FRED_SERIES}
    FRED_SNAPSHOT.write_text(json.dumps({
        "generated_at": datetime.now(UTC).isoformat(),
        "source": "https://fred.stlouisfed.org (FRED, Federal Reserve Bank of St. Louis)",
        "series": series,
    }, indent=1) + "\n", encoding="utf-8", newline="\n")
    return sum(len(v) for v in series.values())


def _book_rate_lines(book: Mapping[str, float], dollar_first: bool) -> tuple[list[str], str | None,
                                                                            dict[str, Any]]:
    """Each holding's measured sensitivity to the 10-year yield and the dollar, weighted into the
    book's. The same regression the single-name line uses, so the book figure is the sum of the
    lines a reader can check one by one."""
    with ThreadPoolExecutor(max_workers=max(1, len(book))) as pool:
        jobs = {s: pool.submit(_rate_sensitivity, s) for s in book}
        per: dict[str, dict[str, Any]] = {}
        for s, job in jobs.items():
            try:
                found = job.result()
            except Exception:
                found = None
            if found is not None:
                per[s] = found
    if not per:
        return [], None, {}
    covered = sum(w for s, w in book.items() if s in per)
    rate = sum(book[s] * per[s]["pct_per_10bp"] for s in per)
    dollar_parts = {s: per[s]["corr_dollar"] for s in per if per[s].get("corr_dollar") is not None}
    lines = [f"{_t(s)} ({book[s]:.0%}): {per[s]['pct_per_10bp']:+.2f}% per +10bp in the 10-year, "
             f"correlation {per[s]['corr_10y']:+.2f}"
             + (f"; {per[s]['corr_dollar']:+.2f} with the dollar" if s in dollar_parts else "")
             for s in sorted(per, key=lambda s: -abs(book[s] * per[s]["pct_per_10bp"]))]
    driver = max(per, key=lambda s: abs(book[s] * per[s]["pct_per_10bp"]))
    missing = [s for s in book if s not in per]
    if missing:
        lines.append(f"Not measured (too little shared history): "
                     f"{', '.join(_t(s) for s in missing)} — the book figure covers {covered:.0%} "
                     f"of it.")
    if dollar_first and dollar_parts:
        weighted = sum(book[s] * c for s, c in dollar_parts.items())
        relation = ("with" if weighted > 0.1 else "against" if weighted < -0.1
                    else "barely with")
        head = (f"Actionable: your book has moved {relation} the dollar — a weighted "
                f"correlation of {weighted:+.2f} over the last three months — so a stronger "
                f"dollar has "
                + ("helped it" if weighted > 0.1 else "hurt it" if weighted < -0.1 else
                   "not been what moves it")
                + f"; {_t(driver)} carries the most rate exposure.")
    else:
        cut = -rate * 1.0  # a 10bp fall in the 10-year
        head = (f"Actionable: your book has moved about {rate:+.2f}% for each +10bp in the 10-year "
                f"(weighted from each holding's last three months), so if a Fed cut took the "
                f"10-year down 10bp the measured relationship says about {cut:+.2f}% — "
                f"{_t(driver)} is the biggest part of it. The 10-year does not have to follow the "
                f"Fed; this is sensitivity, not a forecast.")
    return lines, head, {"per_symbol": per, "book_pct_per_10bp": rate, "covered": covered}


def _macro(symbol: str | None, book: Mapping[str, float] | None = None,
           raw_text: str = "") -> tuple[list[str], list[Source], dict[str, Any]]:
    """Rates, the Fed, inflation and the dollar from FRED, and how ``symbol`` (QQQ when none) — or
    the whole ``book`` when one is held — has traded against the 10-year yield and the dollar."""
    from argus.market.evidence import RSS_FEEDS, RssSource

    target = symbol or BENCHMARK
    with ThreadPoolExecutor(max_workers=len(FRED_SERIES) + 2) as pool:
        jobs = {sid: pool.submit(_fred, sid) for sid in FRED_SERIES}
        fed_job = pool.submit(lambda: RssSource().headlines("fed", RSS_FEEDS["fed"][0]))
        series = {}
        for sid, job in jobs.items():
            try:
                series[sid] = job.result()
            except Exception:
                series[sid] = []
        try:
            fed = fed_job.result()
        except Exception:
            fed = []
    lines: list[str] = []
    readings: dict[str, Any] = {}
    for sid, label in FRED_SERIES.items():
        rows = series.get(sid) or []
        if not rows:
            continue
        day, last = rows[-1]
        first = rows[0][1]
        unit = "" if sid == "DTWEXBGS" else "%"
        change = last - first
        change_text = (f"{change:+.2f}" if sid == "DTWEXBGS" else f"{change * 100:+.0f}bp")
        readings[sid] = {"date": day, "value": last, "change_45d": change}
        lines.append(f"{label}: {last:.2f}{unit} on {day} ({change_text} over the last "
                     f"{len(rows)} readings, about six weeks).")
    ten = readings.get("DGS10", {}).get("value")
    two = readings.get("DGS2", {}).get("value")
    breakeven = readings.get("T10YIE", {}).get("value")
    if ten is not None and two is not None:
        spread = (ten - two) * 100
        lines.append(f"Curve: 10-year minus 2-year is {spread:+.0f}bp — "
                     + ("inverted." if spread < 0 else "positively sloped."))
    if ten is not None and breakeven is not None:
        lines.append(f"Real 10-year yield (nominal less breakeven inflation): about "
                     f"{ten - breakeven:.2f}%.")
    dollar_first = bool(_DOLLAR_FOCUS.search(raw_text))
    book_head: str | None = None
    if book and len(book) > 1:
        book_lines, book_head, book_readings = _book_rate_lines(book, dollar_first)
        lines.extend(book_lines)
        if book_readings:
            readings["book"] = book_readings
    try:
        sensitivity = None if book_head else _rate_sensitivity(target)
    except Exception:
        sensitivity = None  # the backdrop still stands without the co-movement line
    if sensitivity is not None:
        readings["sensitivity"] = sensitivity
        name = "tech (QQQ)" if target == BENCHMARK else _t(target)
        rho = sensitivity["corr_10y"]
        strength = ("strongly" if abs(rho) >= 0.5 else "moderately" if abs(rho) >= 0.25
                    else "barely")
        text = (f"{name} has moved {strength} with rates over {sensitivity['days']} trading days — "
                f"correlation {rho:+.2f} between its daily return and the daily change in the "
                f"10-year yield, about {sensitivity['pct_per_10bp']:+.2f}% for each +10bp")
        if sensitivity.get("corr_dollar") is not None:
            usd = sensitivity["corr_dollar"]
            text += (f"; {usd:+.2f} with the broad dollar index, so a stronger dollar has "
                     + ("helped" if usd > 0.1 else "hurt" if usd < -0.1 else "barely moved")
                     + f" {name}")
        lines.append(text + ". Correlation, not a cause.")
    policy = re.compile(r"\b(?:FOMC|monetary\s+policy|federal\s+funds|minutes|statement|"
                        r"Powell|rate|speech|testimony|economic\s+projections)\b", re.I)
    fed = [h for h in fed if policy.search(h.title)]
    latest_fed = sorted(fed, key=lambda h: h.published, reverse=True)[:2]
    for h in latest_fed:
        lines.append(f"Federal Reserve, {h.published:%d %b}: {h.title} {h.link}")
    if not readings:
        return [], [], {}
    if book_head is not None:
        lines.insert(0, book_head)
    elif (dollar_first and sensitivity is not None
          and sensitivity.get("corr_dollar") is not None):
        usd = sensitivity["corr_dollar"]
        name = "tech (QQQ)" if target == BENCHMARK else _t(target)
        lines.insert(0, (
            f"Actionable: {name} has moved "
            + ("with" if usd > 0.1 else "against" if usd < -0.1 else "independently of")
            + f" the dollar (correlation {usd:+.2f} over {sensitivity['days']} trading days), so "
            + ("a stronger dollar has come with a stronger " + name if usd > 0.1 else
               "a stronger dollar has come with a weaker " + name if usd < -0.1 else
               "the dollar has not been what moves " + name)
            + " — the lines below give the rates side."))
    elif ten is not None:
        head = (f"Actionable: the 10-year is {ten:.2f}%"
                + (f" and the curve {'inverted' if two is not None and ten < two else 'upward'}"
                   if two is not None else ""))
        if sensitivity is not None:
            rho = sensitivity["corr_10y"]
            head += (" — and rates have been driving it: rising yields have come with falling "
                     "prices, so size it against the rates calendar." if rho <= -0.25 else
                     " — and it has been rising with yields, trading on growth rather than "
                     "rates." if rho >= 0.25 else
                     " — but it has not been trading on rates; the backdrop is context, not the "
                     "driver.")
        else:
            head += "."
        lines.insert(0, head)
    if _FRED_USED_SNAPSHOT:
        dated = sorted(set(_FRED_USED_SNAPSHOT.values()))
        lines.append(f"FRED did not answer from here just now, so these series are the desk "
                     f"cycle's last reading ({', '.join(dated)}), not this minute's.")
        _FRED_USED_SNAPSHOT.clear()
    return lines, [Source(kind="venue", ref="FRED (St. Louis Fed) + Bitget TLTUSDT/EURUSDUSDT",
                          detail="FRED series DGS10, DGS2, DFF, T10YIE, DTWEXBGS; Bitget hourly "
                                 "candles; Federal Reserve press feed, live")], readings


def _rate_sensitivity(symbol: str, days: int = 90) -> dict[str, Any] | None:
    """How ``symbol``'s daily return has co-moved with the daily change in the 10-year yield (and
    the broad dollar), on dates both series report. Bitget daily candles against FRED's DGS10 and
    DTWEXBGS; the standard equity-rates sensitivity, stated as % per +10bp."""
    from argus.market.history import CandleType, fetch

    # The price at the US close (4pm New York, 20:00 UTC in daylight time): the 4h candle that
    # opens at 16:00 UTC closes there, on the same clock as the Treasury's daily yield. Bitget's own
    # daily candles open at 16:00 UTC, so a "daily" return straddles two US sessions — measured on
    # QQQ against the 10-year: correlation 0.00 same-day and -0.32 at a one-day lag, an artefact of
    # the boundary, not a finding.
    bars = fetch(symbol, interval="4H", candle_type=CandleType.MARKET, recent=True,
                 limit=min(1000, days * 6))
    closes = {c.ts.date().isoformat(): float(c.close) for c in bars if c.ts.hour == 16}
    ordered = sorted(closes)
    rets = {d: closes[d] / closes[p] - 1 for p, d in itertools.pairwise(ordered)
            if closes[p] > 0}

    def changes(series: str) -> dict[str, float]:
        rows = _fred(series, days=days + 10)
        return {d: v - prev for (_, prev), (d, v) in itertools.pairwise(rows)}

    ten = changes("DGS10")
    common = sorted(set(rets) & set(ten))
    if len(common) < 20:
        return None
    r = [rets[d] * 100 for d in common]
    y = [ten[d] * 100 for d in common]  # basis points
    rho = correlation(r, y)
    slope = beta(r, y)
    if rho is None or slope is None:
        return None
    out: dict[str, Any] = {"days": len(common), "corr_10y": rho, "pct_per_10bp": slope * 10}
    try:
        dollar = changes("DTWEXBGS")
        both = sorted(set(rets) & set(dollar))
        if len(both) >= 20:
            out["corr_dollar"] = correlation([rets[d] for d in both], [dollar[d] for d in both])
    except Exception:
        pass
    return out


def _sentiment(symbol: str | None = None) -> tuple[list[str], list[Source], dict[str, Any]]:
    """The crypto fear & greed index and its week, with BTC and ETH funding as positioning — and,
    for a named contract, that contract's own positioning on Bitget: who pays funding, how crowded
    it is, and how far it has run in 24 hours."""
    import json as _json
    import urllib.request

    from argus.market.bitget import fetch_tickers

    try:
        req = urllib.request.Request("https://api.alternative.me/fng/?limit=8",
                                     headers={"User-Agent": "argus-research/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            rows = _json.loads(resp.read().decode("utf-8")).get("data") or []
    except Exception:
        rows = []
    if not rows:
        return [], [], {}
    now_value = int(rows[0]["value"])
    label = str(rows[0]["value_classification"])
    week = [int(r["value"]) for r in rows[:8]]
    lines = [f"Crypto fear & greed: {now_value} ({label}); over the last {len(week)} days it "
             f"ranged {min(week)} to {max(week)}."]
    own: str | None = None
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
    stretch = ("stretched toward greed — the side of the index where crypto has historically "
               "been more exposed to pullbacks" if now_value >= 70 else
               "stretched toward fear" if now_value <= 30 else "in its middle range")
    if own is not None:
        lines.insert(0, f"Actionable: {own}; the crypto market backdrop is {stretch}. Read both "
                        f"as positioning, not a signal — this desk's own sentiment work found the "
                        f"index adds nothing on its own, and funding is a cost before it is a "
                        f"view.")
    else:
        lines.insert(0, f"Actionable: sentiment is {stretch}; read it as positioning, not a "
                        f"signal — this desk's own sentiment work found the index adds nothing on "
                        f"its own.")
    return lines, [Source(kind="venue", ref="alternative.me Fear & Greed + Bitget funding",
                          detail="alternative.me fear & greed (the source Bitget's sentiment "
                                 "Skill wraps; the Skill returned no data today) and Bitget "
                                 "funding rates, live")], {"index": now_value, "label": label,
                                                          "week": week}


NEWS_LOOKBACK_HOURS = 48
FILING_LOOKBACK_DAYS = 7
MARKET_FEEDS = ("cnbc", "marketwatch", "fed")
"""The outlets read for a market-wide question; for a single name every feed is read and only
headlines that name it are kept."""


def _news(symbol: str, is_open: Any) -> tuple[list[str], list[Source], dict[str, Any]]:
    """Headlines that name ``symbol``, its SEC filings this week, and its 24-hour move split into
    the market's part and its own. For QQQ, the market-wide headlines instead."""
    from datetime import timedelta as _td

    from argus.market.bitget import fetch_tickers
    from argus.market.evidence import RSS_FEEDS, YAHOO_SYMBOL_FEED, EdgarSource, RssSource
    from argus.market.universe import ALIASES

    ticker = _t(symbol)
    now = datetime.now(UTC)
    since = now - _td(hours=NEWS_LOOKBACK_HOURS)
    market = symbol == BENCHMARK
    files_with_sec = not market and (symbol in TRADED_SYMBOLS or _is_equity(symbol))
    names = {ticker.lower()}
    for alias, target in ALIASES.items():
        if target == symbol and len(alias) > 3:
            names.add(alias.lower())
    feeds = ({k: v for k, v in RSS_FEEDS.items() if k in MARKET_FEEDS} if market
             else {**RSS_FEEDS, f"yahoo-{ticker.lower()}": (
                 YAHOO_SYMBOL_FEED.format(ticker=ticker), "news")})
    rss = RssSource()
    pattern = re.compile(r"\b(" + "|".join(re.escape(n) for n in sorted(names)) + r")\b", re.I)

    def read(item: tuple[str, tuple[str, str]]) -> list[Any]:
        try:
            return rss.headlines(item[0], item[1][0])
        except Exception:
            return []

    with ThreadPoolExecutor(max_workers=len(feeds) + 2) as pool:
        pending = [pool.submit(read, item) for item in feeds.items()]
        tickers_job = pool.submit(fetch_tickers)
        filings_job = None if not files_with_sec else pool.submit(lambda: EdgarSource().filings(
                ticker, since=now - _td(days=FILING_LOOKBACK_DAYS)))
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
        lines.insert(0, f"Actionable: {ticker} filed an 8-K on {f.filed:%d %b} ({f.item_summary}) "
                        f"— a company event is on the record; read it before trading the move.")
    elif not market and kept:
        count = "1 headline names" if len(kept) == 1 else f"{len(kept)} headlines name"
        lines.insert(0, f"Actionable: {count} {ticker} in {NEWS_LOOKBACK_HOURS}h"
                        + (" and no 8-K was filed this week — no company event on the record, so "
                           "the move is sentiment or the market, not a disclosed fact."
                           if files_with_sec else
                           " — read them against the split below: the market explains the part "
                           "its beta carries, the rest is its own."))
    elif not market:
        lines.insert(0, f"Actionable: nothing in {NEWS_LOOKBACK_HOURS}h names {ticker}"
                        + (" and no 8-K was filed this week" if files_with_sec else "")
                        + " — whatever moved it is not in the news sources read here; the split "
                          "below says how much of it the market explains.")
    else:
        lines.insert(0, "Actionable: the headlines below are what the market is reading; none of "
                        "them is a cause until a price reaction can be tied to it.")
    for h in kept[:5]:
        hours = (now - h.published).total_seconds() / 3600
        outlet = "Yahoo Finance" if h.feed.startswith("yahoo") else h.feed
        lines.append(f"{hours:.0f}h ago — {h.title} ({outlet}) {h.link}")
    for f in filings[:2]:
        lines.append(f"SEC filing: {f.form} filed {f.filed:%Y-%m-%d} — "
                     f"{f.item_summary if f.is_event else f.description or f.form}.")
    payload = {"change_24h_pct": change, "headlines": [
        {"title": h.title, "feed": h.feed, "link": h.link, "published": h.published.isoformat()}
        for h in kept[:10]], "filings": [f.form for f in filings]}
    return lines, [Source(kind="venue", ref="RSS + Yahoo Finance + SEC EDGAR",
                          detail=f"{len(kept)} headline(s), {len(filings)} filing(s)")], payload


def _book_report(request: ResearchRequest, data: MarketData,
                 is_open: Any) -> tuple[list[str], list[Source], dict[str, Any]]:
    """The whole book as held: risk by holding against weight, volatility, beta and how much of the
    book's movement QQQ explains, the stress and the worst realised 24 hours, and the one trim that
    brings every holding inside the risk budget."""
    columns = _open_columns(data.raw, is_open)
    weights = {s: w for s, w in request.book.items() if w > 0}
    risk = decompose(weights, columns)
    if risk is None:
        return ([f"Not enough shared history across {', '.join(_t(s) for s in weights)} to "
                 f"decompose the book's risk."], [], {})
    shares = {c.symbol: c.contribution / risk.volatility for c in risk.contributions}
    lines: list[str] = []
    budget = request.budget
    over = sorted((s for s in shares if shares[s] > budget), key=lambda s: -shares[s])
    # With N holdings the fairest a book can be is 1/N of the risk each, so a 25% budget cannot be
    # met by four or fewer names without parking the rest in cash. A trader asking how risky a
    # three-coin book is was told to go 43% cash. For such a book, unless the trader set the budget
    # themselves, the actionable is the equal-risk rebalance instead: every name carrying the same
    # share, the money fully invested, correlation included.
    balanced = (_equal_risk_weights(tuple(weights), columns)
                if len(weights) >= 2 and not request.budget_stated
                and len(weights) * budget <= 1.0 else None)
    if balanced is not None:
        top = max(shares, key=lambda s: shares[s])
        lines.append(
            f"Actionable: the risk is concentrated in {_t(top)} ({weights[top]:.0%} of the money, "
            f"{shares[top]:.0%} of the risk). An equal-risk rebalance holds "
            + ", ".join(f"{_t(s)} {balanced[s]:.0%}"
                        for s in sorted(balanced, key=lambda s: -balanced[s]))
            + f" — each name then carries about {1 / len(weights):.0%} of the risk, fully "
              f"invested."
        )
    elif over:
        top = over[0]
        target = _trim_to_budget(top, weights, columns, budget)
        freed = weights[top] - target if target is not None else None
        lines.append(
            f"Actionable: {_t(top)} is {weights[top]:.0%} of the book but {shares[top]:.0%} of its "
            f"risk — over your {budget:.0%} budget"
            + (f"; trimming it to about {target:.0%} (freeing {freed:.0%} of the book, held as "
               f"cash here) brings it inside." if target is not None and freed is not None else
               "; no trim of it alone brings it inside while the rest is unchanged.")
            + (f" {len(over) - 1} other holding(s) are also over budget." if len(over) > 1 else "")
        )
    else:
        top = max(shares, key=lambda s: shares[s])
        lines.append(
            f"Actionable: no holding carries more than your {budget:.0%} risk budget — the largest "
            f"is {_t(top)} at {shares[top]:.0%} of the book's risk, so nothing needs trimming on "
            f"risk grounds."
        )
    lines.append("Where the risk sits: " + "; ".join(
        f"{_t(s)} {weights[s]:.0%} of the money, {shares[s]:.0%} of the risk"
        for s in sorted(shares, key=lambda s: -shares[s])) + ".")
    hourly_vol = risk.volatility
    annual = hourly_vol * math.sqrt(24 * 365)
    spread = risk.effective_positions
    lines.append(
        f"Book volatility about {annual:.0%} a year on open-session hours"
        + (f"; risk is spread across {spread:.1f} effective position(s) of {len(weights)}"
           if spread is not None else "")
        + (f"; {request.cash:.0%} cash dilutes all of it" if request.cash else "") + "."
    )
    bench = columns.get(BENCHMARK, [])
    complete = bool(bench) and all(s in columns for s in weights)
    book_series = [sum(weights[s] * columns[s][i] for s in weights)
                   for i in range(len(bench))] if complete else []
    book_beta = beta(book_series, bench) if book_series else None
    rho = correlation(book_series, bench) if book_series else None
    r2 = None if rho is None else rho * rho
    if book_beta is not None:
        shocks = [Shock("benchmark -5%", -5.0), Shock("benchmark -10%", -10.0)]
        for outcome in stress_by_beta(weights=weights, columns=columns, benchmark=bench,
                                      shocks=shocks):
            if outcome.portfolio_move_pct is not None:
                lines.append(f"If QQQ moves {outcome.shock.removeprefix('benchmark ')}: the book "
                             f"moves about {outcome.portfolio_move_pct:+.2f}% through beta "
                             f"(book beta {book_beta:.2f}).")
    hedge = _hedge_line(book_beta, r2)
    if hedge:
        lines.append(hedge)
    worst = worst_window(weights=weights, columns=columns)
    lines.append(worst.render().replace("[stress] ", "What actually happened, not a model — "))
    payload = {"risk": risk.as_dict(), "shares": shares, "book_beta": book_beta,
               "r_squared_vs_qqq": r2, "cash": request.cash, "worst_window": worst.as_dict()}
    return lines, [Source(kind="computation", ref="argus.desk.portfolio.decompose",
                          detail="Euler risk decomposition, beta stress, realised worst window")
                   ], payload


def _equal_risk_weights(names: Sequence[str], columns: Mapping[str, Sequence[float]],
                        rounds: int = 300) -> dict[str, float] | None:
    """Weights, summing to one, at which every name carries the same share of the book's risk —
    equal risk contribution, correlation included, found by the standard multiplicative update
    (each weight scaled by the square root of target over current share, then renormalised).
    None if the covariance cannot be built or the iteration does not settle within 1 point."""
    weights = {n: 1.0 / len(names) for n in names}
    target = 1.0 / len(names)
    for _ in range(rounds):
        risk = decompose(weights, columns)
        if risk is None:
            return None
        shares = {c.symbol: c.contribution / risk.volatility for c in risk.contributions}
        if any(shares.get(n, 0.0) <= 0 for n in names):
            return None
        weights = {n: weights[n] * math.sqrt(target / shares[n]) for n in names}
        total = sum(weights.values())
        weights = {n: w / total for n, w in weights.items()}
    risk = decompose(weights, columns)
    if risk is None:
        return None
    worst = max(abs(c.contribution / risk.volatility - target) for c in risk.contributions)
    return weights if worst <= 0.01 else None


def _trim_to_budget(symbol: str, weights: Mapping[str, float],
                    columns: Mapping[str, Sequence[float]], budget: float) -> float | None:
    """The largest weight for ``symbol`` — the rest unchanged, the difference held as cash — at
    which its share of the book's risk is inside ``budget``. None if even a 1% holding is over."""
    best: float | None = None
    step = 0.01
    weight = weights[symbol]
    while weight > step / 2:
        weight = round(weight - step, 6)
        trial = {**weights, symbol: weight}
        risk = decompose(trial, columns)
        share = None if risk is None else risk.share_of_risk(symbol)
        if share is not None and share <= budget:
            best = weight
            break
    return best


def _hedge_plan(book: Mapping[str, float], value: Decimal,
                raw_text: str) -> tuple[list[str], list[Source], dict[str, Any]]:
    """Every candidate hedge for ``book``, measured the same way, and the one to use.

    For each candidate perpetual: the book's beta to it and the R² of that fit over thirty days of
    hourly returns (the share of the book's variance a beta-sized short removes), then the cost of
    that short — entry on today's order book at the hedge's own size, plus funding for
    :data:`HEDGE_HOLDING_DAYS` at the current rate, signed for a short. The pick is the candidate
    that removes the most variance per unit of cost, stated with the runner-up so the trade-off is
    visible rather than decided silently."""
    from argus.cost.model import CostModel
    from argus.desk.execution import HedgeLegQuote
    from argus.market.bitget import fetch_tickers
    from argus.market.depth import fetch_orderbook

    crypto = [s for s in book if not _is_equity(s) and s not in TRADED_SYMBOLS]
    equity = [s for s in book if s not in crypto]
    candidates = [c for c in ((HEDGE_CANDIDATES_EQUITY if equity else ())
                              + (HEDGE_CANDIDATES_CRYPTO if crypto else ())) if c not in book]
    if not candidates:
        return [], [], {}
    data = load([*book, *candidates])
    series = data.raw
    stamps = sorted(set.intersection(*(set(series.get(s, {})) for s in (*book, *candidates))))
    if len(stamps) < 100:
        return [], [], {}
    book_returns = [sum(book[s] * series[s][t] for s in book) for t in stamps]
    tickers = fetch_tickers()
    rows: list[dict[str, Any]] = []
    for leg in candidates:
        leg_returns = [series[leg][t] for t in stamps]
        slope = beta(book_returns, leg_returns)
        rho = correlation(book_returns, leg_returns)
        if slope is None or rho is None or slope <= 0:
            continue
        size = (value * Decimal(str(round(slope, 4)))).quantize(Decimal("1"))
        try:
            with _FETCH_SLOTS:
                sweep = fetch_orderbook(leg, limit=50).sweep(size, direction="SELL")
            ticker = tickers[leg]
        except Exception:
            continue
        quote = HedgeLegQuote(symbol=leg, slippage_bps=sweep.slippage_bps,
                              cost_model=CostModel.bitget_perp(funding_rate=ticker.funding_rate))
        cost = quote.cost_model.charge(quote.entry_fill(size),
                                       holding_days=Decimal(HEDGE_HOLDING_DAYS), long=False)
        rows.append({"leg": leg, "beta": slope, "r2": rho * rho, "size": size,
                     "entry_bps": float((cost.commission + cost.spread) / size * 10000),
                     "funding_bps": float(cost.funding / size * 10000),
                     "total_bps": float(cost.bps_of(size)),
                     "complete": sweep.complete})
    if not rows:
        return [], [], {}
    rows.sort(key=lambda r: -r["r2"])
    best = max(rows, key=lambda r: r["r2"] / max(r["total_bps"], 1.0) if r["r2"] >= 0.3
               else -1.0)
    lines = []
    for r in rows:
        funding = r["funding_bps"]
        lines.append(
            f"{_t(r['leg'])}: short ${r['size']:,.0f} (beta {r['beta']:.2f}) removes about "
            f"{r['r2']:.0%} of the book's variance; entry {r['entry_bps']:.1f}bps"
            + (" (more than the visible book)" if not r["complete"] else "")
            + (f", funding over {HEDGE_HOLDING_DAYS} days {abs(funding):.1f}bps "
               + ("paid" if funding > 0 else "received") if abs(funding) >= 0.05
               else ", no funding at the current rate")
            + f" — {r['total_bps']:.1f}bps all in.")
    if best["r2"] < 0.3:
        head = (f"Actionable: no listed hedge explains much of this book — the best, "
                f"{_t(rows[0]['leg'])}, removes {rows[0]['r2']:.0%} of its variance — so the "
                f"risk is mostly its own; reduce the largest holding rather than hedge it.")
    else:
        runner = next((r for r in rows if r is not best), None)
        head = (f"Actionable: hedge with a short in {_t(best['leg'])} of about "
                f"${best['size']:,.0f} "
                f"— it removes about {best['r2']:.0%} of the book's variance for "
                f"{best['total_bps']:.1f}bps to put on and hold {HEDGE_HOLDING_DAYS} days")
        if runner is not None:
            head += (f"; {_t(runner['leg'])} would remove {runner['r2']:.0%} for "
                     f"{runner['total_bps']:.1f}bps")
        head += ". A beta hedge covers the market's part only, not single-name news."
    lines.insert(0, head)
    lines.append(f"Sized on a ${value:,.0f} book" + (" (stated)" if value != HEDGE_BOOK_VALUE
                                                      else " — say your book's value for its "
                                                           "own sizes") + ".")
    return lines, [data.source, Source(kind="venue", ref="bitget order books + funding rates",
                                       detail=f"{', '.join(_t(r['leg']) for r in rows)}; live")], \
        {"legs": rows, "pick": best["leg"], "live": data.live}


def _hedge_line(book_beta: float | None, r_squared: float | None = None) -> str | None:
    """The hedge the Open Theme asks for, as a number: a QQQ short sized to the book's open-session
    beta neutralises the market-driven part of the risk. Stated with its limit — beta hedges the
    market component only; what a name does on its own news is untouched by it."""
    if book_beta is None or abs(book_beta) < 0.05:
        return None
    side = "short" if book_beta > 0 else "long"
    if r_squared is not None and r_squared < 0.3:
        # A crypto book with a 0.9 beta and an R² of 0.1 would be "hedged" by a QQQ short that
        # removes a tenth of its risk; saying "neutralises its market exposure" there misleads.
        return (
            f"Hedge: QQQ explains only {r_squared:.0%} of this book's moves, so a QQQ {side} "
            f"(about {abs(book_beta):.0%} of the book, beta {book_beta:.2f}) would remove little "
            f"of its risk — it is mostly its own; reduce the largest holding instead."
        )
    return (
        f"Hedge: {side} QQQ worth about {abs(book_beta):.0%} of the book's value neutralises its "
        f"market exposure (book beta {book_beta:.2f}"
        + (f", QQQ explains {r_squared:.0%} of its moves" if r_squared is not None else "")
        + "); it does nothing for single-name news risk."
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
                f"Actionable: size {add} so that its worst observed 24 hours ({worst_day:+.1f}%) "
                f"is a "
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


MAX_HOURLY_PARTICIPATION = Decimal("0.10")
"""A slice is spaced so the order never takes more than this share of an hour's volume."""

_SELL_WORDS = re.compile(r"\b(?:sell\w*|liquidat\w*|exit\w*|dump\w*|unload\w*|trim\w*|"
                         r"reduce\w*|close\s+(?:out|my))\b", re.I)


def _depth_lines(symbol: str, notional: Decimal, adv: Decimal, plan: Any,
                 raw_text: str) -> list[str]:
    """The live order book behind the plan: what taking the whole order at once costs, what each
    slice costs when swept alone, and how far apart to space the slices. The plan above prices a
    slice by its style alone, which is why every slice of a $50k NVDA order read "6bps" — the book
    says what the size itself does (a judge's probe, 2026-09-24)."""
    from argus.market.depth import fetch_orderbook

    side = "SELL" if _SELL_WORDS.search(raw_text) else "BUY"
    try:
        with _FETCH_SLOTS:
            book = fetch_orderbook(symbol, limit=50)
        whole = book.sweep(notional, direction=side)
    except Exception:
        return [f"Slice {s.index}: {s.fraction:.0%} as {s.style}, ~{s.expected_cost_bps:.1f}bps "
                f"(the live order book did not answer, so size impact is not included)"
                for s in plan.slices]
    lines: list[str] = []
    fee = float(plan.slices[0].expected_cost_bps) if plan.slices else 6.0
    reach = ("" if whole.complete else
             f" — the visible 50 levels hold only ${float(whole.filled_notional):,.0f}, so the "
             f"rest would fill beyond what can be seen")
    lines.append(
        f"Order book ({side.lower()} side, live): taking all ${float(notional):,.0f} at once "
        f"walks {whole.levels_consumed} level(s) for {float(whole.slippage_bps):.1f}bps of "
        f"slippage on top of the fee{reach}.")
    hourly = adv / Decimal(24)
    for part in plan.slices:
        size = notional * part.fraction
        try:
            swept = book.sweep(size, direction=side)
            impact = (f"{float(swept.slippage_bps):.1f}bps book impact" if swept.complete else
                      f"at least {float(swept.slippage_bps):.1f}bps book impact — larger than the "
                      f"visible book")
        except Exception:
            impact = "book impact unreadable"
        lines.append(f"Slice {part.index}: {part.fraction:.0%} (${float(size):,.0f}) as "
                     f"{part.style} — about {float(part.expected_cost_bps):.1f}bps fee plus "
                     f"{impact}.")
    total_hours = (float(notional / (hourly * MAX_HOURLY_PARTICIPATION)) if hourly > 0
                   else 0.0)
    if total_hours > 1:
        # Bigger than an hour of fair participation: the slices are a style mix worked across a
        # schedule of small child orders, not three big blocks — each block alone would sweep
        # past the visible book.
        child = hourly * MAX_HOURLY_PARTICIPATION / 12
        lines.append(
            f"Schedule: at no more than {MAX_HOURLY_PARTICIPATION:.0%} of an hour's volume the "
            f"order takes about "
            + ("an hour" if total_hours < 1.5 else f"{total_hours:.0f} hours" if total_hours < 48
               else f"{total_hours / 24:.1f} days")
            + f" — work it as child orders of about ${float(child):,.0f} every 5 minutes, "
              f"applying the style mix above across the run.")
    elif len(plan.slices) > 1:
        gap = total_hours * 60 / len(plan.slices)
        lines.append(f"Schedule: space the {len(plan.slices)} slices about {max(gap, 1):.0f} "
                     f"minute(s) apart, so the order never takes more than "
                     f"{MAX_HOURLY_PARTICIPATION:.0%} of an hour's volume.")
    one_shot = fee + float(whole.slippage_bps)
    at_least = "" if whole.complete else "at least "
    # The verdict follows the plan, not a threshold of its own. It used to say "splitting on the
    # schedule below is worth doing" whenever the sweep cost 2bps or more — beside a plan that
    # had just said one order is cheapest (a $15k TSLA order, 2026-09-24).
    if len(plan.slices) <= 1:
        verdict = (" — one order is the plan; the book impact is the price of filling now, and a "
                   "limit at the touch saves it if the fill can wait."
                   if float(whole.slippage_bps) >= 2 else
                   " — the book absorbs it; one order is the plan.")
    else:
        verdict = (" — the book absorbs it, so splitting saves little beyond the passive fills."
                   if float(whole.slippage_bps) < 2 else
                   " — enough book impact that splitting on the schedule below is worth doing.")
    lines.insert(0, (
        f"Actionable: in one market order this costs {at_least or 'about '}{one_shot:.1f}bps "
        f"({fee:.1f}bps fee + {at_least}{float(whole.slippage_bps):.1f}bps book impact)"
        + verdict))
    return lines


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


def _lead_with_what_was_asked(lines: list[str], question: str) -> list[str]:
    """Put the line that answers the question first. "What are analysts' price targets for AAPL?"
    used to lead with the earnings date; the target line was fifth."""
    focus = (r"Analyst price targets" if re.search(r"\btarget|analyst|rating", question, re.I)
             else r"^Institutions:|Institutional holders" if re.search(
                 r"\b13f|institution|holders?|who\s+owns|"
                                                        r"funds?\s+(?:bought|own|hold)",
                                                        question, re.I)
             else r"Earnings surprise" if re.search(r"\bsurprise|beat|miss", question, re.I)
             else None)
    if focus is None:
        return lines
    # Alternatives in order of preference: the institutional summary before one 13F sample.
    hit = next((i for pattern in focus.split("|") for i, line in enumerate(lines)
                if re.search(pattern, line)), None)
    if hit is None:
        return lines
    chosen = lines[hit]
    # One lead line: the answer's own "Actionable:" line steps down behind the one asked for.
    rest = [re.sub(r"^Actionable:\s*(\w)", lambda m: m.group(1).upper(), line)
            for i, line in enumerate(lines) if i != hit]
    body = re.sub(r"^[A-Z]{1,6} — ", "", chosen)
    return [f"Actionable: {body[0].lower() + body[1:]}", *rest]


STALE_QUARTER_DAYS = 135
"""A quarter that ended longer ago than this is not the company's latest: with results due about
45 days after quarter end, the next one has been reported. Measured on MSFT (2026-09-24): SUE read
the March quarter, because its June quarter is fiscal Q4 and sits only in the 10-K."""


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
        f"{window[0].filed.isoformat()})."
        + (" Not the latest quarter: a fiscal fourth quarter is reported only inside the annual "
           "10-K, which carries no separate quarterly figure, so this is the one before it."
           if (datetime.now(UTC).date() - window[0].end).days > STALE_QUARTER_DAYS else ""),
        Source(kind="computation", ref="argus.research.sue via SEC XBRL",
               detail=f"{ticker} eps_diluted, {sue.quarters_used} quarters"),
    )


def _fundamentals(symbol: str) -> tuple[list[str], list[Source]]:
    from datetime import date

    from argus.market import universe
    from argus.market.bitget_mcp import BitgetDataService

    ticker = _t(symbol)
    listed = universe.contracts()
    if listed and symbol not in listed and symbol not in TRADED_SYMBOLS:
        # "Reliance Industries is not a company's shares" was the old answer for a real company
        # Bitget does not list (a judge's probe, 2026-09-24). The true statement is about the venue.
        return ([f"Actionable: {ticker} is not listed on Bitget — none of its {len(listed)} "
                 f"contracts tracks it — so this console has no quote, filings feed or analyst "
                 f"data for it. Ask about a listed name instead."], [])
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

    def entry(entry_id: str) -> Any:
        return BitgetDataService().results(entry_id, symbol=ticker)

    with ThreadPoolExecutor(max_workers=8) as pool:
        pending = {name: pool.submit(ask, name) for name in
                   ("next_earnings", "consensus", "institutional_holdings", "quote",
                    "price_targets")}
        pending["surprise"] = pool.submit(_earnings_surprise, ticker)
        # Three more of the server's equity entries, unused until an audit of the toolkit counted
        # 5 of 22 in use (2026-09-24): valuation ratios, the institutional position summary and
        # insider (Form 4) filings. Each is read only where its fields mean one thing.
        for entry_id in ("equity_fundamental_ratios", "equity_ownership_inst_position_summary",
                         "equity_ownership_insider_trading"):
            pending[entry_id] = pool.submit(entry, entry_id)

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
    ratios = safe("equity_fundamental_ratios")
    ratio_row = (max(ratios, key=lambda r: str(r.get("period_ending") or ""))
                 if isinstance(ratios, list) and ratios else None)
    if isinstance(stock, dict) and stock.get("total_market_cap"):
        cap = float(stock["total_market_cap"])
        pb = stock.get("pb")
        lines.append(f"{ticker} market cap ${cap / 1e9:,.0f}bn"
                     + (f", price-to-book {float(pb):.1f}" if pb and ratio_row is None else "")
                     + ".")
        sources.append(Source(kind="venue", ref="bitget-mcp-server quote",
                              detail=f"{ticker} fundamentals snapshot"))
    if ratio_row is not None:
        parts = [(label, ratio_row.get(key)) for label, key in (
            ("P/E (trailing 12m)", "pe_ttm_ed"), ("P/S (trailing 12m)", "ps_ttm_ed"),
            ("P/B (latest quarter)", "pb_mrq"), ("EV/EBITDA", "ent_multi"))]
        shown = [f"{label} {float(value):.1f}" for label, value in parts
                 if isinstance(value, (int, float)) and value > 0]
        dividend = ratio_row.get("div_yield_12m")
        if isinstance(dividend, (int, float)) and dividend > 0:
            shown.append(f"dividend yield {float(dividend):.2f}%")
        if shown:
            lines.append(f"Valuation on {ratio_row.get('period_ending')}: {', '.join(shown)}.")
            sources.append(Source(kind="venue", ref="bitget-mcp-server equity_fundamental_ratios",
                                  detail=f"{ticker} {ratio_row.get('period_ending')}"))
    positions = safe("equity_ownership_inst_position_summary")
    if isinstance(positions, list) and positions:
        now_row = max(positions, key=lambda r: str(r.get("chg_date") or ""))
        holders_n, share = now_row.get("holder_num"), now_row.get("org_postion_calc")
        if isinstance(holders_n, (int, float)) and isinstance(share, (int, float)):
            lines.append(f"Institutions: {holders_n:,.0f} holders own {share:.1f}% of the shares "
                         f"(as of {now_row.get('chg_date')}).")
            sources.append(Source(kind="venue",
                                  ref="bitget-mcp-server equity_ownership_inst_position_summary",
                                  detail=f"{ticker} {now_row.get('chg_date')}"))
    insiders = safe("equity_ownership_insider_trading")
    if isinstance(insiders, list) and insiders:
        recent = sorted(insiders, key=lambda r: str(r.get("filing_date") or ""), reverse=True)[:3]
        cutoff = (today - timedelta(days=30)).isoformat()
        fresh = [r for r in recent if str(r.get("filing_date") or "") >= cutoff]
        if fresh:
            who = "; ".join(f"{r.get('owner_name')} ({r.get('ownership_type') or 'insider'}), "
                            f"filed {r.get('filing_date')}" for r in fresh)
            lines.append(f"Insider filings (SEC Form 4) in the last 30 days: {who} — read the "
                         f"filing for direction and size: {fresh[0].get('filing_url')}")
            sources.append(Source(kind="venue",
                                  ref="bitget-mcp-server equity_ownership_insider_trading",
                                  detail=f"{ticker} Form 4, last 30 days"))
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
    ResearchKind.EXECUTION: _no_candles("Bitget live 24h ticker and order book (50 levels)",
                                        "bitget /api/v2/mix/market/tickers + orderbook",
                                        "24h volume, depth"),
    ResearchKind.QUOTE: _no_candles("Bitget live ticker, plus the stock's quote from "
                                    "bitget-mcp-server", "bitget /api/v2/mix/market/tickers",
                                    "last, bid, ask, 24h range, funding"),
    ResearchKind.TECHNICALS: _no_candles("Bitget's bitget-signal technical-analysis Skill, live",
                                         "bitget-signal technical_analysis",
                                         "rsi, macd, support/resistance, atr"),
    ResearchKind.VENUE: _no_candles("Bitget's live contract list and tickers, and the stock's own "
                                    "quote from bitget-mcp-server",
                                    "bitget contracts + tickers + bitget-mcp-server",
                                    "listings, fees, funding, premium"),
    ResearchKind.LEVERAGE: _no_candles("live Bitget hourly candles (highs and lows), last 30 days, "
                                       "and the live funding rate", "bitget /api/v3/market/candles",
                                       "1H high/low, funding"),
    ResearchKind.NEWS: _no_candles("Bitget live tickers and hourly candles, headlines from Yahoo "
                                   "Finance and eight outlet feeds, SEC EDGAR filings, live",
                                   "bitget + RSS + SEC EDGAR", "headlines, 8-K, 24h move"),
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
    if request.kind in (ResearchKind.MACRO, ResearchKind.SENTIMENT):
        found, extra, backdrop = (_macro(request.symbols[0] if request.symbols else None,
                                         request.book or None, raw_text)
                                  if request.kind is ResearchKind.MACRO
                                  else _sentiment(request.symbols[0] if request.symbols
                                                  else None))
        if not found:
            return Answer(question=question, refused=True,
                          reason="the backdrop sources did not answer",
                          lines=["Neither FRED nor the sentiment source answered just now, so "
                                 "there is nothing honest to report. Try again shortly."])
        found.extend(f"Assumed: {note}." for note in request.notes)
        found.append(f"Data: {extra[0].detail if extra else 'public sources'}. This is analysis, "
                     f"not advice — you make the call.")
        return Answer(question=question, lines=found, sources=extra,
                      data={"request": request.as_dict(), request.kind.value: backdrop})
    if request.kind is ResearchKind.HEDGE and request.book:
        value = request.notional or HEDGE_BOOK_VALUE
        found, extra, hedge_found = _hedge_plan(request.book, value, raw_text)
        if not found:
            return Answer(question=question, refused=True,
                          reason="the hedge candidates could not be measured",
                          lines=["The candidate hedges' prices or order books did not arrive, so "
                                 "no hedge can be measured just now. Try again shortly."])
        found.extend(f"Assumed: {note}." for note in request.notes)
        found.append("Data: live Bitget hourly candles (30 days), order books and funding rates. "
                     "This is analysis, not advice — you make the call.")
        return Answer(question=question, lines=found, sources=extra,
                      data={"request": request.as_dict(), "hedge": hedge_found})
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

        elif request.kind is ResearchKind.VENUE:
            lines, extra = _venue(request.symbols[0], is_open)
            sources.extend(extra)

        elif request.kind is ResearchKind.CONSTRUCT:
            columns = _open_columns(data.raw, is_open)
            names = tuple(s for s in request.symbols if s in columns)
            weights = _equal_risk_weights(names, columns) if len(names) >= 2 else None
            if weights is None:
                return Answer(question=question, refused=True,
                              reason="the names share too little history to weight by risk",
                              lines=["Not enough shared hourly history across those names to "
                                     "balance their risk. Try names that trade on Bitget every "
                                     "hour, or fewer of them."])
            built = replace(request, kind=ResearchKind.BOOK, book=weights, budget_stated=True,
                            budget=1.0)
            book_lines, extra, book_payload = _book_report(built, data, is_open)
            lines = [
                "Actionable: an equal-risk book of these names holds "
                + ", ".join(f"{_t(s)} {weights[s]:.0%}"
                            for s in sorted(weights, key=lambda s: -weights[s]))
                + f" — each carries about {1 / len(weights):.0%} of the risk, the quieter names "
                  f"held larger so no single one dominates.",
                *[line for line in book_lines if not line.startswith("Actionable")],
            ]
            sources.extend(extra)
            payload["construct"] = {"weights": weights, **book_payload}

        elif request.kind is ResearchKind.LEVERAGE:
            lines, extra, lever_payload = _leverage(request.symbols[0], request.leverage or 10.0,
                                                    request.side)
            sources.extend(extra)
            payload["leverage"] = lever_payload
            if not lines:
                return Answer(question=question, refused=True,
                              reason="Bitget's candles for this name could not be read",
                              lines=["Bitget's hourly candles did not come back just now, so the "
                                     "liquidation odds cannot be measured. Try again shortly."])

        elif request.kind is ResearchKind.NEWS:
            lines, extra, news_payload = _news(request.symbols[0], is_open)
            sources.extend(extra)
            payload["news"] = news_payload

        elif request.kind is ResearchKind.BOOK:
            lines, extra, book_payload = _book_report(request, data, is_open)
            sources.extend(extra)
            payload["book"] = book_payload

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
                    f"-10% implies {(row['qqq_minus_10'] or 0):+.1f}%; worst 24-hour window "
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
                meaning = _funding_meaning(symbol, funding)
                if meaning:
                    lines.append(meaning)
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
            if len(request.symbols) > 1:
                with ThreadPoolExecutor(max_workers=len(request.symbols)) as pool:
                    parts = list(pool.map(_fundamentals, request.symbols))
                lines, extra = [], []
                for symbol, (part_lines, part_sources) in zip(request.symbols, parts,
                                                               strict=True):
                    lines.extend(line.replace("Actionable: ", f"Actionable ({_t(symbol)}): ", 1)
                                 if line.startswith("Actionable:") else f"{_t(symbol)} — {line}"
                                 for line in part_lines)
                    extra.extend(part_sources)
            else:
                lines, extra = _fundamentals(request.symbols[0])
            lines = _lead_with_what_was_asked(lines, raw_text)
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
                f"{symbol.removesuffix('USDT')}'s 24h volume (${adv:,.0f}); the fees alone come "
                f"to about {plan.expected_total_cost_bps:.1f}bps, and the book below adds what "
                f"the size itself costs.",
                f"Plan: {plan.rationale}.",
            ]
            lines.extend(_depth_lines(symbol, request.notional, adv, plan, raw_text))
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
    lines.sort(key=lambda line: 0 if line.startswith("Actionable") else 1)
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
