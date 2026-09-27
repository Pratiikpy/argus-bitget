"""What a research question can be about, and the request a question is read into."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum
from typing import Any

from argus.lui.trace import trace_module
from argus.truth.paths import DATA_DIR

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


FIXTURE_PATH = DATA_DIR / "risk_layer_candles_fixture.json"


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

    EVENT = "event"
    """How a name has reacted to a scheduled event type — CPI releases, Fed decisions, its own
    earnings — from `research/event_reactions.py`: the MacKinlay event study with the four tests
    and the clustering correction (`research/eventstudy.py`, OWNED against whale-signals), and how
    large its move on those days is against its ordinary 24 hours."""

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

    spot: str | None = None
    """The spot rToken held (``RTSLAUSDT``) when the question is about owning the token rather than
    trading the perpetual. Its hedge is a different answer: the same company's perpetual."""

    shock_on: str | None = None
    """For STRESS, the instrument the shock hits when it is not the Nasdaq: "oil -20%", "if ETH
    drops 25%", "MSTR craters 30%". Every holding then moves through its beta to that instrument.
    None means QQQ. Before 2026-09-25 every stress was a QQQ shock, and "if oil drops 20% how does
    that hit my book" was answered as a question about adding oil."""

    horizon_hours: int | None = None
    """For ANALOGUE, the horizon of a directional question ("will MSTR be higher in 48 hours?"),
    answered with the contract's record over windows of that length (`desk/odds.py`). None for
    the plain "has it been here before" question."""

    weekend: bool = False
    """The directional horizon is the weekend: Friday's close to Monday's."""

    target: float | None = None
    """For IMPACT, the weight a HELD name ends at ("trim TSLA to 10%", "NVDA from 8% to 20%",
    "cut QQQ by 30%" once its current weight is known) — `desk/portfolio.resize`, not an add."""

    resize_from: float | None = None
    """The current weight the question states for the resized name ("from 8%"), which overrides
    the book's own figure for it; None when not stated."""

    resize_by: float | None = None
    """A relative cut or raise ("by 30%", "close 30% of my position"): the target is this fraction
    of the current weight, known only once the book is."""

    level: float | None = None
    """A price level a directional question names ("closes above 70000 this week")."""

    mandate_text: str = ""
    """The trader's own earlier words about their mandate ("I'm conservative", "I can't lose more
    than 8%"), from `lui/memory.py`, read by the mandate check when the question states none."""

    mandate_capital: Decimal | None = None
    """The account size the trader stated earlier, which the mandate sizes against."""

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": str(self.kind),
            "symbols": list(self.symbols),
            "book": dict(self.book),
            "size": self.size,
            "size_stated": self.size_stated,
            "shock_on": self.shock_on,
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
            "spot": self.spot,
            "horizon_hours": self.horizon_hours,
            "weekend": self.weekend,
            "target": self.target,
            "resize_from": self.resize_from,
            "resize_by": self.resize_by,
            "level": self.level,
        }


CRYPTO_ANCHOR = "BTCUSDT"


CRYPTO_LINKED = frozenset({"MSTRUSDT", "COINUSDT", "HOODUSDT"})
"""Stocks whose price is mostly a claim on crypto: Strategy holds bitcoin, Coinbase and Robinhood
earn on crypto volume. "Should I buy MSTR" was benchmarked to QQQ alone, although MSTR's hourly
correlation with BTC was +0.81 (readiness backlog L42)."""


def _t(symbol: str) -> str:
    """The name a trader reads: NVDA, not NVDAUSDT; CVX, not CVXSTOCKUSDT."""
    base = symbol.removesuffix("USDT")
    return base.removesuffix("STOCK") if base.endswith("STOCK") and len(base) > 5 else base


bare_symbol = _t
"""The public name for other packages (`lui/journal.py`, `lui/memory.py`, ...). The engines here
write `_t` inside their f-strings, where eleven letters would break a hundred lines."""


# Every engine here is a traced step from import on (lui/trace.py, trace_module).
trace_module(globals())
