"""Paper-trading performance — the three numbers Track 2 is actually scored on.

The handbook names them exactly: **"Paper trading Sharpe, max drawdown, win rate."** That is the
50% quantitative half of Track 2, and until this module existed nothing computed them from the
ledger. :mod:`argus.eval.scorecard` produced calibration (Brier, ECE, reliability, abstention
quality) — better than the field, and not one of the three. :mod:`argus.backtest.metrics` produced
all three — but only for the backtest engine's synthetic return stream, never for the real log.
This is the bridge, and it reuses those primitives rather than growing a second implementation.

**Why a daily series and not a per-trade one.** A Sharpe over per-trade returns answers "how good
is the average trade", which is not what "paper trading Sharpe" means to anybody scoring a log. It
also silently rewards trading less: fewer, luckier trades raise it. The honest construction is a
calendar series over the whole paper-trading window, in which a day with no settlement contributes
exactly ``0.0`` — because that is what the account did that day. Flat days drag the mean and shrink
the standard deviation, and they should: a desk that abstains for a fortnight and then wins once
has not earned a fortnight's Sharpe.

**Annualisation is 365, not 252.** rTokens are perpetuals on a 7x24 venue; there are no market
holidays and no weekends off. Annualising a 7-day-a-week series by ``sqrt(252)`` understates it by
``sqrt(365/252)`` ≈ 20%. The constant is named and exported so a reader can disagree with it in one
place rather than discovering it inlined at a call site.

**What this module refuses to do.** It never returns 0.0 for an undefined statistic. A ledger of
pure abstentions has no Sharpe — its daily series is identically zero, the standard deviation is
floating-point noise, and :func:`argus.backtest.metrics.sharpe` raises rather than dividing by it.
That refusal is carried through to the report as an explicit ``None`` plus the reason, because a
printed ``"sharpe": 0.0`` is indistinguishable from a real zero and would be read as one.

Read against ``agent-backtest-lab`` (Apache-2.0), whose ``abl/scorecard/drawdown.py:39-80``
computes drawdown from the net-of-cost equity curve with a running peak, and whose
``abl/multipletest/psr.py:52-84`` derives the Sharpe moments. Both agree with
:mod:`argus.backtest.metrics`; what they do not carry is the 7x24 annualisation or the
zero-variance refusal, so ours departs on those two points deliberately.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any

from argus.backtest.metrics import MetricError, max_drawdown, sharpe, sortino
from argus.paper.corrections import is_voided
from argus.paper.ledger import Entry, PaperLedger

PERIODS_PER_YEAR = 365
"""Calendar days. rTokens trade 7x24; there is no holiday calendar to thin the series."""

MIN_DAYS_FOR_SHARPE = 30
"""A *statistical* minimum, not the arithmetic one — and the difference was a real shipped bug.

This was 2, justified as "below this a standard deviation is not defined at all". That guards
against a **crash**, not against a **meaningless number**, and the distinction stayed invisible for
as long as the desk had never traded. The moment it did — two settled fills on 2026-09-15, both
winners — this printed **Sharpe 6.75** onto the cockpit from two daily returns, next to prose still
saying the figure was undefined. An absurd Sharpe is worse than an absent one: it is precisely the
overfit-looking number this project exists to refuse, and it would have shipped to judges.

30 is not invented here — it is this codebase's own already-considered threshold
(`backtest.dependence.MIN_OBSERVATIONS`, "fewest observations before a dependence correction means
anything"). Using one number for both keeps a single standard rather than two that can drift.
"""

MIN_TRADES_TO_REPORT = 1
"""Win rate over zero trades is not 0%, it is undefined. One trade is enough to state a fact.

Deliberately left at 1, and it is a different kind of claim from the Sharpe above: a win rate is a
**descriptive count** of what actually happened ("2 of 2 settled trades were profitable"), not an
inferential statistic that extrapolates. It is reported with its own `n` beside it so a reader
cannot mistake the sample for a track record — see :attr:`Performance.as_dict`.
"""

MIN_TRADES_FOR_DRAWDOWN = 5
"""Max drawdown needs enough trades for a peak-to-trough to mean something.

Separated from `MIN_TRADES_TO_REPORT` for the same reason the Sharpe constant was raised. With two
winning trades and no losing one, the arithmetic returns **0.00%** — which reads as *"this desk
took risk and never lost"* when the truth is *"this desk has taken two positions"*. The existing
code already worried about exactly this failure ("the best-looking risk number on the scorecard —
earned by not participating"); it simply guarded it with a threshold of 1, which could not catch it.
"""


def _as_date(stamp: str) -> date:
    return datetime.fromisoformat(stamp).date()


@dataclass(frozen=True)
class DailyReturn:
    """One calendar day of the paper-trading window."""

    day: date
    net_pnl: Decimal
    ret: float
    trades_settled: int

    @property
    def is_flat(self) -> bool:
        return self.trades_settled == 0


@dataclass
class SymbolContribution:
    """Per-symbol split, because a single instrument carrying the whole result is the classic trap.

    A cross-sectional strategy that looks strong in aggregate and turns out to be one name with
    four lucky trades has been produced on this project before. Reporting the split makes that
    visible in the same object as the headline number rather than in a follow-up investigation.
    """

    symbol: str
    trades: int
    wins: int
    net_pnl: Decimal

    @property
    def win_rate(self) -> float:
        return self.wins / self.trades if self.trades else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "trades": self.trades,
            "wins": self.wins,
            "net_pnl": str(self.net_pnl),
            "win_rate_pct": round(100 * self.win_rate, 1),
        }


@dataclass
class PaperPerformance:
    """The scored half of Track 2, computed from settled ledger rows and nothing else."""

    window_days: int
    first_day: date | None
    last_day: date | None
    capital: Decimal

    trades: int
    """Settled, non-abstention entries. The only rows that can win or lose money."""

    open_positions: int
    abstentions: int
    flat_days: int

    net_pnl: Decimal
    total_return: float

    sharpe: float | None
    sortino: float | None
    max_drawdown: float | None
    """Worst peak-to-trough fall in equity, or ``None`` when nothing was ever at risk.

    **``None``, not ``0.0``, and this one is scored.** Bitget's Track 2 judging focus is *"Paper
    trading Sharpe, max drawdown, win rate"* (handbook line 254). With zero settled trades the
    equity series never moves, so the arithmetic honestly returns 0.0% — and 0.0% max drawdown is
    an *outstanding* risk number. It would be read as excellent risk control by anyone scoring that
    line, when what it actually measures is an account that never took a position.

    `sharpe`, `sortino` and `win_rate` were already reported as ``None`` here with stated reasons.
    Drawdown was the one that still produced a flattering number from the same absence, and it was
    the most flattering of the four.

    A drawdown over a never-invested account is not comparable to a traded strategy's drawdown, so
    it is not published as one.
    """
    win_rate: float | None

    undefined: dict[str, str] = field(default_factory=dict)
    """Statistic name -> why it could not be computed. Never silently replaced with a zero."""

    by_symbol: list[SymbolContribution] = field(default_factory=list)
    daily: list[DailyReturn] = field(default_factory=list)

    @property
    def largest_symbol_share(self) -> float:
        """Fraction of gross absolute PnL from the single biggest contributor."""
        gross = sum(abs(c.net_pnl) for c in self.by_symbol)
        if not gross:
            return 0.0
        return float(max(abs(c.net_pnl) for c in self.by_symbol) / gross)

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "window_days": self.window_days,
            "first_day": self.first_day.isoformat() if self.first_day else None,
            "last_day": self.last_day.isoformat() if self.last_day else None,
            "capital": str(self.capital),
            "periods_per_year": PERIODS_PER_YEAR,
            "trades": self.trades,
            "open_positions": self.open_positions,
            "abstentions": self.abstentions,
            "flat_days": self.flat_days,
            "net_pnl": str(self.net_pnl),
            "total_return_pct": round(100 * self.total_return, 4),
            "max_drawdown_pct": (
                None if self.max_drawdown is None else round(100 * self.max_drawdown, 4)
            ),
        }
        out["sharpe"] = round(self.sharpe, 3) if self.sharpe is not None else None
        out["sortino"] = round(self.sortino, 3) if self.sortino is not None else None
        out["win_rate_pct"] = (
            round(100 * self.win_rate, 2) if self.win_rate is not None else None
        )
        if self.undefined:
            out["undefined"] = dict(self.undefined)
        if self.by_symbol:
            out["by_symbol"] = [c.as_dict() for c in self.by_symbol]
            out["largest_symbol_share_pct"] = round(100 * self.largest_symbol_share, 1)
        return out


def _settled_trades(ledger: PaperLedger) -> list[Entry]:
    """Settled positions the desk actually took.

    **Voided rows are excluded, and that exclusion is the whole reason this function has a
    docstring.** `paper/corrections.py` lists ledger rows that record a fill the Constitution had
    refused — a `paper/runner.py` defect fixed on 2026-09-20. They are left in the chain unedited
    (deleting them would make the record look better than the system behaved), so every consumer
    must filter them instead. Without this filter, Sharpe, win rate, max drawdown and net P&L are
    all computed over positions that were never opened.
    """
    return [
        e for e in ledger.entries
        if e.is_settled and not e.is_abstention and not is_voided(e.seq)
    ]


def daily_series(
    ledger: PaperLedger, *, capital: Decimal, through: date | None = None
) -> list[DailyReturn]:
    """Build the calendar return series over the paper-trading window.

    Every day between the first decision and ``through`` (default: the last settlement, or the
    last decision if nothing has settled) appears exactly once, including days on which the desk
    did nothing. A trade lands on the day it **settled**, not the day it was decided, because that
    is when the profit or loss became real.
    """
    if not ledger.entries:
        return []
    if capital <= 0:
        raise MetricError("capital must be positive to express PnL as a return")

    settled = _settled_trades(ledger)
    by_day: dict[date, list[Entry]] = defaultdict(list)
    for entry in settled:
        assert entry.settled_at is not None  # is_settled guarantees this
        by_day[_as_date(entry.settled_at)].append(entry)

    first = min(_as_date(e.decided_at) for e in ledger.entries)
    candidates = [max(by_day) if by_day else first]
    candidates.append(max(_as_date(e.decided_at) for e in ledger.entries))
    last = through or max(candidates)
    if last < first:
        last = first

    out: list[DailyReturn] = []
    day = first
    while day <= last:
        todays = by_day.get(day, [])
        pnl = sum((Decimal(e.net_pnl or "0") for e in todays), Decimal("0"))
        out.append(
            DailyReturn(
                day=day,
                net_pnl=pnl,
                ret=float(pnl / capital),
                trades_settled=len(todays),
            )
        )
        day += timedelta(days=1)
    return out


def evaluate_ledger(
    ledger: PaperLedger, *, capital: Decimal = Decimal("10000"), through: date | None = None
) -> PaperPerformance:
    """Compute Sharpe, max drawdown and win rate from the paper log.

    ``capital`` is the notional the returns are expressed against. It is a stated denominator, not
    a measured one: the ledger records PnL in quote currency and a return needs a base. Changing it
    scales Sharpe not at all (mean and standard deviation scale together) and total return
    proportionally, which is exactly the behaviour a reader should expect.
    """
    daily = daily_series(ledger, capital=capital, through=through)
    trades = _settled_trades(ledger)
    undefined: dict[str, str] = {}

    contributions: dict[str, SymbolContribution] = {}
    for entry in trades:
        pnl = Decimal(entry.net_pnl or "0")
        c = contributions.setdefault(
            entry.symbol, SymbolContribution(entry.symbol, 0, 0, Decimal("0"))
        )
        c.trades += 1
        c.net_pnl += pnl
        if pnl > 0:
            c.wins += 1

    returns = [d.ret for d in daily]
    equity = [1.0]
    for r in returns:
        equity.append(equity[-1] * (1 + r))

    sharpe_value: float | None = None
    sortino_value: float | None = None
    if len(returns) < MIN_DAYS_FOR_SHARPE:
        undefined["sharpe"] = (
            f"the window is {len(returns)} day(s); a standard deviation needs at least "
            f"{MIN_DAYS_FOR_SHARPE}"
        )
        undefined["sortino"] = undefined["sharpe"]
    else:
        try:
            sharpe_value = sharpe(returns, periods_per_year=PERIODS_PER_YEAR)
        except MetricError as exc:
            undefined["sharpe"] = str(exc)
        try:
            sortino_value = sortino(returns, periods_per_year=PERIODS_PER_YEAR)
        except MetricError as exc:
            undefined["sortino"] = str(exc)

    win_rate: float | None = None
    if len(trades) < MIN_TRADES_TO_REPORT:
        undefined["win_rate"] = (
            "no settled trades; a win rate over zero trades is undefined, not zero per cent"
        )
    else:
        win_rate = sum(1 for e in trades if Decimal(e.net_pnl or "0") > 0) / len(trades)

    # Drawdown measures how far capital fell from a peak. With nothing ever deployed there is no
    # peak and no fall, and the 0.0% the arithmetic returns would be the best-looking risk number
    # on the scorecard — earned by not participating. Reported as undefined for the same reason
    # win_rate is, and named explicitly because this is one of the three figures Track 2 scores.
    drawdown: float | None = None
    if not trades:
        undefined["max_drawdown"] = (
            "no settled trades; equity never moved, so the 0.0% this would otherwise report "
            "measures an account that never took a position rather than the desk's risk control, "
            "and is not comparable to a traded strategy's drawdown"
        )
    elif len(trades) < MIN_TRADES_FOR_DRAWDOWN:
        undefined["max_drawdown"] = (
            f"{len(trades)} settled trade(s); a peak-to-trough needs at least "
            f"{MIN_TRADES_FOR_DRAWDOWN} to describe risk rather than luck. With every settled "
            f"trade a winner the arithmetic returns 0.00%, which reads as 'took risk, never lost' "
            f"when the truth is 'has taken {len(trades)} position(s)'"
        )
    else:
        drawdown = max_drawdown(equity)

    net = sum((Decimal(e.net_pnl or "0") for e in trades), Decimal("0"))
    return PaperPerformance(
        window_days=len(daily),
        first_day=daily[0].day if daily else None,
        last_day=daily[-1].day if daily else None,
        capital=capital,
        trades=len(trades),
        open_positions=sum(
            1 for e in ledger.entries if not e.is_settled and not e.is_abstention
        ),
        abstentions=sum(1 for e in ledger.entries if e.is_abstention),
        flat_days=sum(1 for d in daily if d.is_flat),
        net_pnl=net,
        total_return=equity[-1] - 1.0,
        sharpe=sharpe_value,
        sortino=sortino_value,
        max_drawdown=drawdown,
        win_rate=win_rate,
        undefined=undefined,
        by_symbol=sorted(contributions.values(), key=lambda c: -abs(c.net_pnl)),
        daily=daily,
    )


__all__ = [
    "MIN_DAYS_FOR_SHARPE",
    "MIN_TRADES_TO_REPORT",
    "PERIODS_PER_YEAR",
    "DailyReturn",
    "PaperPerformance",
    "SymbolContribution",
    "daily_series",
    "evaluate_ledger",
]
