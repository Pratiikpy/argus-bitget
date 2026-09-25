"""XA arena: ARGUS's cross-asset hedge router against the systems that lead the sub-theme.

`eval/execution_comparison.py` compared `desk/execution.py` with crypto_sor, a same-instrument,
cross-*venue* router, and never asked whether the leg it picked hedged anything; the 2026-09-24
rival review re-graded the row to IMPLEMENTED for exactly that reason and named the rivals that
actually lead Cross-Asset Execution. This module runs them, unmodified, on the same input as ARGUS.

**One book, one tape, one scorer.** Every contestant manages the same starting book — $100,000:
rNVDA, rAAPL and rTSLA spot (one sixth each), BTC and ETH USDT-M perpetual longs ($25,000 and
$15,000, fully collateralised), $10,000 of free USDT — on the frozen XA-TAPE (`eval/xa_tape.py`).
At the close of every hourly bar a contestant sees only data up to that close (history is streamed
bar by bar, never handed over whole), returns orders, and they fill at the next bar's open. Every
fill pays the venue's taker fee and slippage walked on a recorded order book; every perpetual pays
or receives the venue's real funding at each settlement. Nothing is charged differently for anyone.

**The contestants.**

* ARGUS — `desk/crossasset.py`, plus ablations that remove one mechanism each.
* Five rivals, run from their own clones in a separate process (`eval/baselines/xa_rivals/`):
  Triad (danielamodu, S2), Omni (Jayanng, S2, MIT), Crossfire (CryptoCT01, S2), VIGIL
  (norbert351, S2) and HedgeAgents (the JansenAnalytics replication of arXiv 2502.13165). Each is
  run in the mode it ships without an LLM key; what that leaves out is stated per rival in
  :data:`RIVAL_NOTES`, never hidden.
* Naive baselines a hedger must beat to mean anything: hold, a same-name perpetual hedge over
  every shut session, the same over weekends only, and a volatility-target overlay.

**The scorer, pre-registered in :data:`PREREG` before any arm ran.** Primary: the annualised
certainty equivalent ``mean - (gamma/2) var`` of hourly book returns net of every cost at
gamma = 5, compared pairwise with a stationary bootstrap over hours (Politis & Romano 1994,
mean block 24 h) and Holm-corrected across rivals. Guard: CVaR95 of hourly returns inside the
pre-registered risk windows (macro releases, amplitude shocks by HedgeAgents' own EMC rule, rToken
divergence events by Triad's own threshold). Reported beside them: Sharpe, drawdown, costs by
channel, turnover, weekend window losses, and the timing alpha against the held book (Newey-West),
which separates hedging skill from simply holding less.

**Periods.** Period B (2026-06-28 → 2026-09-25) is primary: funding is observed for every
settlement. Period A (2026-03-29 → 2026-06-28) is an earlier, independent window where the venue no
longer serves funding history; it is run with funding recorded as unobserved (not zero-cost by
assumption — the column is marked) and is the check that ARGUS's constants were not tuned to B.
"""

from __future__ import annotations

import json
import math
import os
import random
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from argus.desk.crossasset import (
    PERP_OF_SPOT,
    SHUT,
    CostCurve,
    RiskModel,
    RouterConfig,
    RouterInput,
    estimate,
    phase_calendar,
    route,
    settlements_between,
)
from argus.eval import artefact
from argus.eval.xa_tape import HOUR_MS, Tape, load, tape_digest

REPORT_PATH = Path(__file__).resolve().parents[3] / "data" / "xa_arena.json"
RUNNER_DIR = Path(__file__).resolve().parent / "baselines" / "xa_rivals"

START_NAV = 100_000.0
BOOK_SPOT = {"RNVDAUSDT": 1 / 6, "RAAPLUSDT": 1 / 6, "RTSLAUSDT": 1 / 6}
BOOK_PERP = {"BTCUSDT": 0.25, "ETHUSDT": 0.15}
HEDGE_PERPS: tuple[str, ...] = (
    "NVDAUSDT", "AAPLUSDT", "TSLAUSDT", "QQQUSDT", "SPYUSDT", "SQQQUSDT", "BTCUSDT", "ETHUSDT",
)
SPOT_TRADEABLE: tuple[str, ...] = (
    "RNVDAUSDT", "RAAPLUSDT", "RTSLAUSDT", "RQQQUSDT", "RSPYUSDT",
)
MODEL_KEYS: tuple[str, ...] = (
    "spot:RNVDAUSDT", "spot:RAAPLUSDT", "spot:RTSLAUSDT",
    *(f"perp:{s}" for s in HEDGE_PERPS),
)

PERIODS: dict[str, tuple[datetime, datetime]] = {
    "B": (datetime(2026, 6, 28, tzinfo=UTC), datetime(2026, 9, 25, tzinfo=UTC)),
    "A": (datetime(2026, 3, 29, tzinfo=UTC), datetime(2026, 6, 28, tzinfo=UTC)),
}
PRIMARY_PERIOD = "B"
MAX_GROSS_PERP_NAV = 5.0

PREREG: dict[str, Any] = {
    "key": "t2-4-cross-asset-xa-arena",
    "registered": "2026-09-25, before any arm was scored",
    "book": {"nav": START_NAV, "spot": BOOK_SPOT, "perp": BOOK_PERP, "free_usdt_share": 0.1},
    "periods": {k: [a.isoformat(), b.isoformat()] for k, (a, b) in PERIODS.items()},
    "primary_period": PRIMARY_PERIOD,
    "execution": "decide at the close of bar i on data <= close i; fill at the open of bar i+1",
    "costs": "venue perp taker fee (instruments endpoint), 10 bps spot taker, slippage walked on "
             "a 2026-09-25 book, real funding at each settlement where the venue serves it",
    "primary": {"metric": "annualised certainty equivalent of hourly net returns, gamma=5",
                "test": "paired stationary bootstrap over hours, mean block 24h, B=2000, seed 7",
                "correction": "Holm across rivals", "alpha": 0.05, "direction": "greater"},
    "guard": "CVaR95 of hourly net returns inside the union of pre-registered risk windows",
    "risk_windows": {
        "macro": "FOMC 14:00 ET and CPI 08:30 ET releases: [release-1h, release+24h]",
        "shock": "HedgeAgents EMC rule on 00:00 UTC closes of BTC, ETH, rNVDA, rAAPL, rTSLA "
                 "(|1d|>5% or |3d|>10%): [trigger, trigger+120h]",
        "divergence": "Triad's 24h gap |rToken - BTC| >= 3pp (its HEDGE threshold, "
                      "src/decision/engine.py:25 x 5pp scale): [t, t+24h]",
        "weekend": "shut sessions longer than 24h (reported separately, not in the guard union)",
    },
    "argus_config": "desk.crossasset.RouterConfig() defaults, gamma=5, fixed before scoring",
    "verdict_rule": "WIN if primary one-sided Holm p<0.05 for ARGUS; LOSS if the reverse test "
                    "is significant; otherwise TIE. Stated per rival.",
    "deviations": [
        "The hedge-only bound in desk/crossasset.py:route was added after a first period-B run "
        "(2026-09-25) showed the unbounded router levering the crypto sleeve in calm shut "
        "sessions. Period B is therefore not a clean holdout for that choice; period A is.",
        "2026-09-26: route() gained a golden-section search along its own step "
        "(_best_on_segment) after the constructed funding-spike case showed it discarding a "
        "profitable move. A correctness fix found by a constructed input, not by a scored period, "
        "but made after period-B numbers had been seen.",
    ],
}

RIVAL_NOTES: dict[str, str] = {
    "triad": "danielamodu/Triad @d70c67b, no licence file: run from the clone, never vendored. "
             "Keyless path: Groq absent so engine.decide falls back to weighted_decision "
             "(its own documented fallback). Signals read through its own get_divergence / "
             "get_event / get_sentiment with the bgc/RSS transport replaced by the tape: RSS and "
             "bgc skill text are unavailable historically, so the event signal comes from its "
             "own expansion_event fallback and sentiment from its funding z + basis fallback. "
             "Live cadence is 5 min; replayed hourly. Sizes as configured ($500/$1,000 per "
             "decision, $1,000 cap per bot leg); a x10 variant matches its own replay's $10,000 "
             "book to ours.",
    "omni": "Jayanng/Omni @c50d566, MIT. Keyless path: llm.decide returns fallback_decision; "
            "policy.validate with its forced minimum protection and session hard-veto; margin "
            "model risk.evaluate with the venue discount rate and MMR ladder frozen on the tape; "
            "scenario shocks by its own empirical_tail_shock on tape daily closes. Governs rNVDA "
            "(its demo portfolio). Daemon cadence 60 s; replayed hourly. Hedge cap $5,000 as "
            "configured.",
    "crossfire": "CryptoCT01/Crossfire @7a3bdfa, no licence file: run from the clone. Keyless "
                 "path is policy_decide, which observes the Mag7-vs-BTC divergence and suppresses "
                 "every open ('policy does not open risk'); its LLM (DeepSeek via OpenRouter) is "
                 "the only path that trades. The would-have-fired count is reported.",
    "vigil": "norbert351/vigil @3bfad2d, package.json says MIT but no licence file: run only. "
             "Its own replay path (src/backtest.js): planOrders + crossAssetRegime on daily "
             "00:00 UTC closes and the historical Fear & Greed, with its risk-off slice. Targets "
             "set to the arena book's weights (a user input in its config). Crypto keys map to "
             "the perpetual sleeve; it never shorts.",
    "hedgeagents": "JansenAnalytics/hedge-agents @8b87aaf (replication of arXiv 2502.13165; "
                   "package.json says MIT, no licence file): run only. Keyless: every LLM call "
                   "fails over to its own coded fallback, so the Budget Allocation Conference "
                   "returns equal weight (bac.cjs _fallbackOttoDecision) every 30 days and the "
                   "Extreme Market Conference returns Hold. A second arm uses its deterministic "
                   "optimiser (math.cjs optimizePortfolio via domain.runPortfolioOptimizer, "
                   "neutral 5% forecasts). Sleeves: Dave = crypto perps, Bob = rToken basket, "
                   "Emily (FX) = USDT.",
}


# --- the book and the simulator ---------------------------------------------------------------


@dataclass
class Book:
    cash: float
    spot: dict[str, float]
    perp: dict[str, float]
    base_perp_qty: dict[str, float]
    peak_nav: float = START_NAV
    day_start_nav: float = START_NAV
    day: str = ""

    def nav(self, px: Callable[[str], float]) -> float:
        return self.cash + sum(q * px(f"spot:{s}") for s, q in self.spot.items())

    def as_msg(self, px: Callable[[str], float]) -> dict[str, Any]:
        nav = self.nav(px)
        return {
            "cash": self.cash, "nav": nav, "peak_nav": self.peak_nav,
            "day_start_nav": self.day_start_nav, "start_nav": START_NAV,
            "spot": {s: q for s, q in self.spot.items() if q},
            "perp": {s: q for s, q in self.perp.items() if q},
            "base_perp": dict(self.base_perp_qty),
        }


@dataclass
class Ledger:
    nav: list[float] = field(default_factory=list)
    fees: float = 0.0
    slippage: float = 0.0
    funding: float = 0.0
    traded: float = 0.0
    fills: int = 0
    stale_fills: int = 0
    refused_orders: int = 0
    actions: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    net_exposure: list[float] = field(default_factory=list)


class Contestant(Protocol):
    name: str

    def start(self, init: dict[str, Any]) -> None: ...
    def bar(self, msg: dict[str, Any]) -> None: ...
    def step(self, msg: dict[str, Any]) -> dict[str, Any]: ...
    def finish(self) -> dict[str, Any]: ...


def cost_curves(tape: Tape) -> dict[str, CostCurve]:
    out: dict[str, CostCurve] = {}
    perp_fee = tape.fees.get("perp_taker_bps", {})
    spot_fee = float(tape.fees.get("spot_taker_bps", 10.0))
    for k, meta in tape.slippage.items():
        kind, sym = k.split(":", 1)
        fee = round(float(perp_fee.get(sym, 6.0)), 6) if kind == "perp" else spot_fee
        pts = tuple((float(p["notional"]), float(p["slippage_bps"]))
                    for p in meta.get("curve", []) if p.get("slippage_bps") is not None)
        out[k] = CostCurve(fee, pts)
    return out


def _bar_payload(tape: Tape, i: int, keys: Sequence[str]) -> dict[str, list[float]]:
    return {k: [tape.open[k][i], tape.high[k][i], tape.low[k][i], tape.close[k][i],
                tape.quote_volume[k][i], 1.0 if tape.stale[k][i] else 0.0]
            for k in keys if not math.isnan(tape.close[k][i])}


def _settled(tape: Tape, lo_ms: int, hi_ms: int, observed_from: int | None) -> list[list[Any]]:
    """Funding settlements with ``lo_ms < ts <= hi_ms`` (none when funding is unobserved)."""
    if observed_from is None:
        return []
    out: list[list[Any]] = []
    for sym, rows in tape.funding.items():
        for ts, rate in rows:
            if lo_ms < ts <= hi_ms and ts >= observed_from:
                out.append([sym, ts, rate])
    return out


@dataclass(frozen=True)
class RunResult:
    name: str
    period: str
    start_i: int
    end_i: int
    ledger: Ledger
    finish: dict[str, Any]
    seconds: float


def run(tape: Tape, contestant: Contestant, period: str, *,
        funding_observed: bool | None = None) -> RunResult:
    """One contestant over one period, with every fill, fee and settlement applied here."""
    t0 = time.time()
    p_start, p_end = PERIODS[period]
    s = tape.index_of(int(p_start.timestamp() * 1000))
    e = min(len(tape.hours), tape.index_of(int(p_end.timestamp() * 1000) - HOUR_MS) + 1)
    observed_from = tape.funding_observed_from
    fund_ok = observed_from <= tape.hours[s] if funding_observed is None else funding_observed
    fund_from = observed_from if fund_ok else None
    keys = sorted(tape.close)
    curves = cost_curves(tape)
    contestant.start({
        "type": "init", "period": period, "start_i": s, "end_i": e,
        "hours": [tape.hours[0], HOUR_MS, len(tape.hours)],
        "keys": keys, "funding_observed": fund_ok,
        "fees": tape.fees, "slippage": tape.slippage, "venue_risk": tape.venue_risk,
        "reality": tape.reality, "macro": tape.macro,
        "book_spec": {"nav": START_NAV, "spot": BOOK_SPOT, "perp": BOOK_PERP},
    })
    fng_sent = 0
    for i in range(0, s):
        msg: dict[str, Any] = {"type": "bar", "i": i, "ts": tape.hours[i] + HOUR_MS,
               "bars": _bar_payload(tape, i, keys),
               "funding": _settled(tape, tape.hours[i - 1] if i else -1, tape.hours[i],
                                   fund_from),
               "fng": [p for p in tape.fng[fng_sent:] if p[0] <= tape.hours[i]]}
        fng_sent += len(msg["fng"])
        contestant.bar(msg)

    def px(k: str, idx: int, field_: str = "close") -> float:
        arr: Sequence[float] = getattr(tape, field_)[k]
        return float(arr[idx])

    def closes_at(idx: int) -> Callable[[str], float]:
        """The close of every key at bar ``idx``, bound now rather than when it is called."""
        return lambda k: px(k, idx)

    book = Book(cash=START_NAV, spot={}, perp={}, base_perp_qty={})
    for sym, w in BOOK_SPOT.items():
        book.spot[sym] = START_NAV * w / px(f"spot:{sym}", s, "open")
    for sym, w in BOOK_PERP.items():
        q = START_NAV * w / px(f"perp:{sym}", s, "open")
        book.perp[sym] = q
        book.base_perp_qty[sym] = q
    book.cash = START_NAV * (1 - sum(BOOK_SPOT.values()))
    led = Ledger()
    pending: list[dict[str, Any]] = []
    prev_close: dict[str, float] = {f"perp:{sym}": px(f"perp:{sym}", s, "open")
                                    for sym in book.perp}

    for i in range(s, e):
        ts_open = tape.hours[i]
        # 1. funding at this bar's open, on the position held into it
        if fund_from is not None:
            for sym, _ts, rate in _settled(tape, ts_open - HOUR_MS, ts_open, fund_from):
                q = book.perp.get(sym, 0.0)
                if q:
                    pay = q * px(f"perp:{sym}", i, "open") * rate
                    book.cash -= pay
                    led.funding += pay
        # 2. fills at the open
        q_before = dict(book.perp)
        for order in pending:
            _fill(book, order, tape, i, curves, led)
        pending = []
        # 3. perpetual variation margin
        for sym in set(q_before) | set(book.perp):
            k = f"perp:{sym}"
            o, c = px(k, i, "open"), px(k, i)
            pc = prev_close.get(k, o)
            book.cash += q_before.get(sym, 0.0) * (o - pc) + book.perp.get(sym, 0.0) * (c - o)
            prev_close[k] = c
        for sym in list(book.perp):
            prev_close[f"perp:{sym}"] = px(f"perp:{sym}", i)
        nav = book.nav(closes_at(i))
        led.nav.append(nav)
        led.net_exposure.append(_net_exposure(book, closes_at(i)) / nav if nav else 0.0)
        book.peak_nav = max(book.peak_nav, nav)
        day = datetime.fromtimestamp((ts_open + HOUR_MS) / 1000, UTC).date().isoformat()
        if day != book.day:
            book.day = day
            book.day_start_nav = nav
        # 4. decision at the close
        msg = {"type": "step", "i": i, "ts": ts_open + HOUR_MS,
               "bars": _bar_payload(tape, i, keys),
               "funding": _settled(tape, ts_open - HOUR_MS, ts_open, fund_from),
               "fng": [p for p in tape.fng[fng_sent:] if p[0] <= ts_open],
               "book": book.as_msg(closes_at(i))}
        fng_sent += len(msg["fng"])
        reply: dict[str, Any] = (contestant.step(msg) if i + 1 < e
                                 else {"orders": [], "action": "END"})
        action = str(reply.get("action", "HOLD"))
        led.actions[action] = led.actions.get(action, 0) + 1
        if reply.get("note") and len(led.notes) < 12:
            led.notes.append(f"{datetime.fromtimestamp(msg['ts'] / 1000, UTC):%Y-%m-%d %H:%M} "
                             f"{reply['note']}"[:400])
        pending = [o for o in reply.get("orders", []) if abs(float(o.get("usd", 0.0))) > 0]

    finish = contestant.finish()
    return RunResult(contestant.name, period, s, e, led, finish, time.time() - t0)


INVERSE_LEVERAGE = {"SQQQUSDT": -3.0}
"""SQQQ is the inverse-3x Nasdaq fund, so a long in it is short exposure."""


def _net_exposure(book: Book, px: Callable[[str], float]) -> float:
    """Signed market exposure in USD (a diagnostic, not a risk model)."""
    spot = sum(q * px(f"spot:{s}") for s, q in book.spot.items())
    perp = sum(q * px(f"perp:{s}") * INVERSE_LEVERAGE.get(s, 1.0) for s, q in book.perp.items())
    return spot + perp


def _fill(book: Book, order: dict[str, Any], tape: Tape, i: int,
          curves: dict[str, CostCurve], led: Ledger) -> None:
    kind = str(order.get("kind", "perp"))
    sym = str(order["symbol"]).upper()
    usd = float(order["usd"])
    k = f"{kind}:{sym}"
    if k not in tape.open or math.isnan(tape.open[k][i]) or k not in curves:
        led.refused_orders += 1
        return
    price = tape.open[k][i]
    nav = book.nav(lambda kk: tape.open[kk][i] if not math.isnan(tape.open[kk][i]) else 0.0)
    if kind == "spot":
        if usd < 0:
            held = book.spot.get(sym, 0.0) * price
            usd = -min(-usd, held)
        else:
            rate = curves[k].bps(usd) / 1e4
            usd = min(usd, max(0.0, book.cash) / (1 + rate))
        if abs(usd) < 1.0:
            led.refused_orders += 1
            return
        rate = curves[k].bps(usd) / 1e4
        book.spot[sym] = book.spot.get(sym, 0.0) + usd / price
        book.cash -= usd
    else:
        gross_after = sum(abs(q) * tape.open[f"perp:{s}"][i] for s, q in book.perp.items()
                          if s != sym) + abs(book.perp.get(sym, 0.0) * price + usd)
        if nav > 0 and gross_after > MAX_GROSS_PERP_NAV * nav:
            led.refused_orders += 1
            return
        rate = curves[k].bps(usd) / 1e4
        book.perp[sym] = book.perp.get(sym, 0.0) + usd / price
        if abs(book.perp[sym]) * price < 1e-6:
            book.perp.pop(sym)
    fee = abs(usd) * curves[k].fee_bps / 1e4
    slip = abs(usd) * rate - fee
    book.cash -= fee + slip
    led.fees += fee
    led.slippage += slip
    led.traded += abs(usd)
    led.fills += 1
    if tape.stale[k][i]:
        led.stale_fills += 1


# --- history shared by the in-process contestants --------------------------------------------


class History:
    """Bars streamed so far, per key, aligned to tape indices. Only ever holds the past."""

    def __init__(self) -> None:
        self.hours: list[int] = []
        self.open: dict[str, list[float]] = {}
        self.close: dict[str, list[float]] = {}
        self.real: dict[str, list[bool]] = {}
        self.funding: dict[str, list[tuple[int, float]]] = {}
        self.first: dict[str, int] = {}
        self.keys: list[str] = []

    def start(self, init: dict[str, Any]) -> None:
        self.keys = list(init["keys"])
        for k in self.keys:
            self.open[k], self.close[k], self.real[k] = [], [], []

    def push(self, msg: dict[str, Any]) -> None:
        i = int(msg["i"])
        if len(self.hours) > i:
            return
        self.hours.append(int(msg["ts"]) - HOUR_MS)
        for k in self.keys:
            b = msg["bars"].get(k)
            if b is None:
                self.open[k].append(float("nan"))
                self.close[k].append(float("nan"))
                self.real[k].append(False)
                continue
            if k not in self.first:
                self.first[k] = i
            self.open[k].append(b[0])
            self.close[k].append(b[3])
            self.real[k].append(not b[5])
        for sym, ts, rate in msg.get("funding", []):
            self.funding.setdefault(sym, []).append((int(ts), float(rate)))


class _InProcess:
    """A contestant that runs inside this process on streamed history (ARGUS and baselines)."""

    name = "base"

    def __init__(self, history: History | None = None, *, owns_history: bool = True) -> None:
        self.h = history if history is not None else History()
        self.owns = owns_history if history is not None else True
        self.curves: dict[str, CostCurve] = {}
        self.decisions = 0

    def start(self, init: dict[str, Any]) -> None:
        if self.owns:
            self.h.start(init)
        self.curves = {k: CostCurve(v.fee_bps, v.slippage) for k, v in
                       _curves_from_init(init).items()}

    def bar(self, msg: dict[str, Any]) -> None:
        if self.owns:
            self.h.push(msg)

    def step(self, msg: dict[str, Any]) -> dict[str, Any]:
        if self.owns:
            self.h.push(msg)
        return self.decide(msg)

    def decide(self, msg: dict[str, Any]) -> dict[str, Any]:
        return {"orders": [], "action": "HOLD"}

    def finish(self) -> dict[str, Any]:
        return {}


def _curves_from_init(init: dict[str, Any]) -> dict[str, CostCurve]:
    perp_fee = init["fees"].get("perp_taker_bps", {})
    spot_fee = float(init["fees"].get("spot_taker_bps", 10.0))
    out: dict[str, CostCurve] = {}
    for k, meta in init["slippage"].items():
        kind, sym = k.split(":", 1)
        fee = round(float(perp_fee.get(sym, 6.0)), 6) if kind == "perp" else spot_fee
        out[k] = CostCurve(fee, tuple((float(p["notional"]), float(p["slippage_bps"]))
                                      for p in meta.get("curve", [])
                                      if p.get("slippage_bps") is not None))
    return out


class Hold(_InProcess):
    name = "hold"


class SessionHedge(_InProcess):
    """Short the same-name perpetual for the full rToken holding over every shut session (or only
    weekends), flat while the cash market is open. The mechanism Omni and Ballast use, with no
    trigger and no cost check — what a hedger must beat to show its judgement is worth anything."""

    def __init__(self, *, weekends_only: bool, **kw: Any) -> None:
        super().__init__(**kw)
        self.weekends_only = weekends_only
        self.name = "static_weekend_hedge" if weekends_only else "static_session_hedge"
        self._phases: list[str] = []

    def start(self, init: dict[str, Any]) -> None:
        super().start(init)
        base, step, n = init["hours"]
        self._phases = phase_calendar([base + j * step for j in range(n)])

    def decide(self, msg: dict[str, Any]) -> dict[str, Any]:
        i = int(msg["i"])
        nxt = i + 1 if i + 1 < len(self._phases) else i
        want = self._phases[nxt] == SHUT
        if want and self.weekends_only:
            run_len = 0
            j = nxt
            while j < len(self._phases) and self._phases[j] == SHUT and run_len <= 30:
                run_len += 1
                j += 1
            back = nxt
            while back > 0 and self._phases[back - 1] == SHUT and run_len <= 30:
                run_len += 1
                back -= 1
            want = run_len > 24
        book = msg["book"]
        orders = []
        for spot_sym, qty in book["spot"].items():
            perp = PERP_OF_SPOT.get(spot_sym)
            if not perp or f"perp:{perp}" not in msg["bars"]:
                continue
            spot_px = msg["bars"][f"spot:{spot_sym}"][3]
            perp_px = msg["bars"][f"perp:{perp}"][3]
            target = -qty * spot_px if want else 0.0
            cur = book["perp"].get(perp, 0.0) * perp_px
            if abs(target - cur) >= 50:
                orders.append({"kind": "perp", "symbol": perp, "usd": target - cur})
        return {"orders": orders, "action": "HEDGE" if orders and want else
                ("UNHEDGE" if orders else "HOLD")}


class VolTarget(_InProcess):
    """Scale the crypto sleeve by ``min(1, normal vol / forecast vol)`` (EWMA, lambda 0.97): the
    textbook response to a volatility shock, and the naive form of "reallocate after a shock"."""

    name = "vol_target_overlay"

    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        self.ew: dict[str, float] = {}

    def decide(self, msg: dict[str, Any]) -> dict[str, Any]:
        i = int(msg["i"])
        book = msg["book"]
        orders = []
        for sym, base_q in book["base_perp"].items():
            k = f"perp:{sym}"
            c = self.h.close[k]
            lo = max(1, i - 60 * 24)
            rets = [math.log(c[j] / c[j - 1]) for j in range(lo, i + 1) if c[j - 1] > 0]
            if len(rets) < 200:
                continue
            normal = statistics.fmean(r * r for r in rets)
            ew = normal
            for r in rets[-240:]:
                ew = 0.97 * ew + 0.03 * r * r
            scale = min(1.0, math.sqrt(normal / ew)) if ew > 0 else 1.0
            target = base_q * scale * c[i]
            cur = book["perp"].get(sym, 0.0) * c[i]
            if abs(target - cur) >= max(50.0, 0.1 * abs(base_q * c[i])):
                orders.append({"kind": "perp", "symbol": sym, "usd": target - cur})
        return {"orders": orders, "action": "REBALANCE" if orders else "HOLD"}


class Argus(_InProcess):
    """ARGUS's router (`desk/crossasset.py`) as an arena contestant."""

    def __init__(self, cfg: RouterConfig | None = None, *, name: str = "argus",
                 model_cache: dict[tuple[Any, ...], RiskModel] | None = None, **kw: Any) -> None:
        super().__init__(**kw)
        self.cfg = cfg or RouterConfig()
        self.name = name
        self.cache = model_cache if model_cache is not None else {}
        self._phases: list[str] = []
        self.explained: list[str] = []
        self.hedge_hours = 0
        self.refusals = 0
        self.gated: dict[str, int] = {}

    def start(self, init: dict[str, Any]) -> None:
        super().start(init)
        base, step, n = init["hours"]
        self._phases = phase_calendar([base + j * step for j in range(n)])

    def _model(self, i: int, keys: tuple[str, ...]) -> RiskModel:
        c = self.cfg
        sig = (i, keys, c.lookback_hours, tuple(sorted(c.block_hours.items())),
               c.normal_block_hours, c.ewma_lambda, c.regime_clip, c.shrink, c.min_blocks,
               c.stability_min_abs_corr, c.horizon_cap_hours, c.use_stability_gate,
               c.use_regime, c.use_phase)
        if sig not in self.cache:
            self.cache[sig] = estimate(self.h.close, keys, self._phases, i, c, real=self.h.real)
        return self.cache[sig]

    def decide(self, msg: dict[str, Any]) -> dict[str, Any]:
        i = int(msg["i"])
        book = msg["book"]
        lo = i - self.cfg.lookback_hours // 2
        keys = tuple(k for k in MODEL_KEYS if k in self.h.first and self.h.first[k] <= lo)
        model = self._model(i, keys)
        px = {k: self.h.close[k][i] for k in keys}
        nav = float(book["nav"])
        spot_usd = {f"spot:{s}": q * px.get(f"spot:{s}", 0.0) for s, q in book["spot"].items()}
        perp_usd = {f"perp:{s}": book["perp"].get(s, 0.0) * px[f"perp:{s}"]
                    for s in HEDGE_PERPS if f"perp:{s}" in px}
        base = {f"perp:{s}": q * px[f"perp:{s}"] for s, q in book["base_perp"].items()
                if f"perp:{s}" in px}
        ts = int(msg["ts"])
        f_now: dict[str, float] = {}
        f_norm: dict[str, float] = {}
        for sym, rows in self.h.funding.items():
            k = f"perp:{sym}"
            past = [r for t_, r in rows if t_ <= ts - HOUR_MS]
            if past:
                f_now[k] = statistics.fmean(past[-3:])
                window = [r for t_, r in rows
                          if ts - self.cfg.lookback_hours * HOUR_MS <= t_ <= ts - HOUR_MS]
                f_norm[k] = statistics.fmean(window) if window else 0.0
        inp = RouterInput(
            nav=nav, spot_usd=spot_usd, perp_usd=perp_usd, base_perp_usd=base,
            funding_now=f_now, funding_normal=f_norm,
            settlements_ahead=settlements_between(ts, model.horizon_hours),
            costs={k: self.curves[k] for k in perp_usd if k in self.curves},
        )
        tradeable = [k for k in perp_usd if k in self.curves]
        decision = route(model, inp, self.cfg, tradeable=tradeable)
        self.decisions += 1
        if decision.refused:
            self.refusals += 1
        for a, b in decision.gated_pairs:
            key = f"{a}~{b}"
            self.gated[key] = self.gated.get(key, 0) + 1
        overlay = sum(abs(decision.target_usd.get(k, 0.0) - base.get(k, 0.0))
                      for k in decision.target_usd)
        if overlay > 100:
            self.hedge_hours += 1
        orders = [{"kind": "perp", "symbol": k.split(":", 1)[1], "usd": v}
                  for k, v in decision.orders_usd.items()]
        note = decision.explain() if orders else ""
        if orders and len(self.explained) < 20:
            self.explained.append(
                f"{datetime.fromtimestamp(ts / 1000, UTC):%Y-%m-%d %H:%M} {note}")
        return {"orders": orders, "action": "REFUSE" if decision.refused else
                ("HEDGE" if orders else "HOLD"), "note": note}

    def finish(self) -> dict[str, Any]:
        top_gated = sorted(self.gated.items(), key=lambda kv: -kv[1])[:8]
        return {"decisions": self.decisions, "refusals": self.refusals,
                "hours_with_overlay": self.hedge_hours, "sample_explanations": self.explained,
                "most_gated_pairs": top_gated}


# --- rivals, in their own process -------------------------------------------------------------


class Subprocess:
    """A rival run from its own clone in a child process, speaking one JSON object per line."""

    def __init__(self, name: str, argv: list[str], cwd: Path, *,
                 env: dict[str, str] | None = None) -> None:
        self.name = name
        self.argv = argv
        self.cwd = cwd
        self.env = env
        self.proc: subprocess.Popen[str] | None = None
        self._err: Any = None

    def _send(self, msg: dict[str, Any], *, reply: bool) -> dict[str, Any]:
        assert self.proc is not None and self.proc.stdin is not None
        assert self.proc.stdout is not None
        self.proc.stdin.write(json.dumps(msg, separators=(",", ":")) + "\n")
        self.proc.stdin.flush()
        if not reply:
            return {}
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"{self.name} runner exited: {self._stderr_tail()}")
        out: dict[str, Any] = json.loads(line)
        if "error" in out:
            raise RuntimeError(f"{self.name} runner error: {out['error']} "
                               f"{out.get('trace', '')}")
        return out

    def _stderr_tail(self) -> str:
        if self._err is None:
            return ""
        self._err.seek(0)
        return str(self._err.read().decode("utf-8", "replace"))[-3000:]

    def start(self, init: dict[str, Any]) -> None:
        env = dict(os.environ)
        for secret in ("BITGET_QWEN_API_KEY", "OPENROUTER_API_KEY", "GROQ_API_KEY",
                       "OMNI_LLM_API_KEY", "VIGIL_QWEN_API_KEY", "ANTHROPIC_API_KEY",
                       "OPENAI_API_KEY", "BITGET_API_KEY", "BITGET_SECRET_KEY",
                       "BITGET_PASSPHRASE"):
            env.pop(secret, None)
        env["PYTHONUTF8"] = "1"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env.update(self.env or {})
        # stderr goes to a temporary file, not a pipe: a chatty rival would otherwise fill the
        # pipe buffer and block while this process waits on its stdout.
        # closed in finish(), which outlives this method: a with-block would close it here
        self._err = tempfile.TemporaryFile()  # noqa: SIM115
        self.proc = subprocess.Popen(
            self.argv, cwd=self.cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self._err, text=True, encoding="utf-8", env=env, bufsize=1)
        self._send(init, reply=True)

    def bar(self, msg: dict[str, Any]) -> None:
        self._send(msg, reply=False)

    def step(self, msg: dict[str, Any]) -> dict[str, Any]:
        return self._send(msg, reply=True)

    def finish(self) -> dict[str, Any]:
        out = self._send({"type": "finish"}, reply=True)
        assert self.proc is not None
        if self.proc.stdin is not None:
            self.proc.stdin.close()
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        if self._err is not None:
            self._err.close()
        return out


def rival_contestants() -> list[Subprocess]:
    from argus.eval.baselines.xa_rivals import RivalUnavailable, commands

    try:
        cmds = commands()
    except RivalUnavailable as exc:
        UNAVAILABLE["all_rivals"] = str(exc)
        return []
    return [Subprocess(name, argv, cwd, env=env) for name, argv, cwd, env in cmds]


# --- metrics ----------------------------------------------------------------------------------


def returns(nav: Sequence[float], start: float = START_NAV) -> list[float]:
    out = []
    prev = start
    for v in nav:
        out.append(v / prev - 1.0)
        prev = v
    return out


def cvar(xs: Sequence[float], q: float = 0.05) -> float:
    """Expected shortfall of the worst ``q`` share, as a positive loss."""
    ordered = sorted(xs)
    n = max(1, math.floor(q * len(ordered)))
    return -statistics.fmean(ordered[:n]) if ordered else float("nan")


def ce(xs: Sequence[float], gamma: float, *, per_year: float = 8760.0) -> float:
    if len(xs) < 2:
        return float("nan")
    return (statistics.fmean(xs) - gamma / 2 * statistics.pvariance(xs)) * per_year


def max_drawdown(nav: Sequence[float], start: float = START_NAV) -> float:
    peak = start
    worst = 0.0
    for v in nav:
        peak = max(peak, v)
        worst = max(worst, (peak - v) / peak)
    return worst


def sharpe(xs: Sequence[float], *, per_year: float = 8760.0) -> float:
    sd = statistics.pstdev(xs) if len(xs) > 1 else 0.0
    return statistics.fmean(xs) / sd * math.sqrt(per_year) if sd > 0 else float("nan")


def newey_west_alpha(y: Sequence[float], x: Sequence[float], lags: int = 24) -> dict[str, float]:
    """OLS ``y = a + b x`` with a Newey-West (Bartlett) standard error on ``a``, annualised."""
    n = len(y)
    mx, my = statistics.fmean(x), statistics.fmean(y)
    sxx = sum((xi - mx) ** 2 for xi in x)
    b = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y, strict=True)) / sxx if sxx else 0.0
    a = my - b * mx
    res = [yi - a - b * xi for xi, yi in zip(x, y, strict=True)]
    # HAC variance of the mean of residual-weighted intercept moment (x demeaned regressor).
    zbar = [1.0 - mx * (xi - mx) * n / sxx if sxx else 1.0 for xi in x]
    g = [z * e for z, e in zip(zbar, res, strict=True)]
    s = sum(v * v for v in g) / n
    for lag in range(1, lags + 1):
        w = 1 - lag / (lags + 1)
        s += 2 * w * sum(g[t] * g[t - lag] for t in range(lag, n)) / n
    se = math.sqrt(max(s, 0.0) / n)
    return {"alpha_annual": a * 8760, "beta": b, "t_alpha": a / se if se > 0 else float("nan"),
            "resid_sd_annual": statistics.pstdev(res) * math.sqrt(8760)}


# --- risk windows (pre-registered rules, applied to the tape) ---------------------------------


def _et_to_utc(day: str, hh: int, mm: int) -> int:
    from zoneinfo import ZoneInfo

    local = datetime.fromisoformat(day).replace(hour=hh, minute=mm, tzinfo=ZoneInfo(
        "America/New_York"))
    return int(local.astimezone(UTC).timestamp() * 1000)


def risk_windows(tape: Tape, s: int, e: int) -> dict[str, set[int]]:
    """Hour indices in ``[s, e)`` inside each pre-registered window family."""
    out: dict[str, set[int]] = {"macro": set(), "shock": set(), "divergence": set(),
                                "weekend": set()}
    hrs = tape.hours
    releases = ([_et_to_utc(d, 14, 0) for d in tape.macro.get("fomc", [])]
                + [_et_to_utc(d, 8, 30) for d in tape.macro.get("cpi", [])])
    for r in releases:
        for i in range(s, e):
            if r - HOUR_MS <= hrs[i] <= r + 24 * HOUR_MS:
                out["macro"].add(i)
    daily_keys = ("perp:BTCUSDT", "perp:ETHUSDT", "spot:RNVDAUSDT", "spot:RAAPLUSDT",
                  "spot:RTSLAUSDT")
    for i in range(max(s, 72), e):
        if (hrs[i] + HOUR_MS) // HOUR_MS % 24:
            continue
        for k in daily_keys:
            c = tape.close[k]
            d1 = c[i] / c[i - 24] - 1
            d3 = c[i] / c[i - 72] - 1
            if abs(d1) > 0.05 or abs(d3) > 0.10:
                out["shock"].update(range(i + 1, min(e, i + 121)))
                break
    for i in range(max(s, 24), e):
        btc = tape.close["spot:BTCUSDT"][i] / tape.close["spot:BTCUSDT"][i - 24] - 1
        for k in ("spot:RAAPLUSDT", "spot:RNVDAUSDT", "spot:RTSLAUSDT"):
            gap = tape.close[k][i] / tape.close[k][i - 24] - 1 - btc
            if abs(gap) >= 0.03:
                out["divergence"].update(range(i + 1, min(e, i + 25)))
                break
    phases = phase_calendar(list(hrs))
    j = s
    while j < e:
        if phases[j] != SHUT:
            j += 1
            continue
        stop = j
        while stop < e and phases[stop] == SHUT:
            stop += 1
        if stop - j > 24:
            out["weekend"].update(range(j, stop))
        j = stop
    return out


def window_losses(nav: Sequence[float], s: int, members: set[int]) -> list[float]:
    """NAV change over each contiguous run of hours in ``members`` (fractional)."""
    out: list[float] = []
    runs: list[list[int]] = []
    for i in sorted(members):
        if runs and i == runs[-1][-1] + 1:
            runs[-1].append(i)
        else:
            runs.append([i])
    for r in runs:
        first, last = r[0] - s, r[-1] - s
        if last >= len(nav) or first < 0:
            continue
        before = nav[first - 1] if first >= 1 else START_NAV
        out.append(nav[last] / before - 1)
    return out


# --- statistics -------------------------------------------------------------------------------


def stationary_bootstrap(n: int, *, mean_block: float, reps: int, seed: int) -> list[list[int]]:
    """Politis & Romano (1994) resampled index paths (wrapping), one list per replicate."""
    rng = random.Random(seed)
    p = 1.0 / mean_block
    paths: list[list[int]] = []
    for _ in range(reps):
        idx: list[int] = []
        j = rng.randrange(n)
        while len(idx) < n:
            idx.append(j)
            j = rng.randrange(n) if rng.random() < p else (j + 1) % n
        paths.append(idx)
    return paths


def paired_test(a: Sequence[float], b: Sequence[float], stat: Callable[[Sequence[float]], float],
                paths: Sequence[Sequence[int]]) -> dict[str, float]:
    """Bootstrap distribution of ``stat(a) - stat(b)`` on shared resampled hours; one-sided p for
    the difference being <= 0 (i.e. evidence that ``a`` is better) and the reverse."""
    obs = stat(a) - stat(b)
    diffs = []
    for path in paths:
        ra = [a[j] for j in path]
        rb = [b[j] for j in path]
        diffs.append(stat(ra) - stat(rb))
    diffs.sort()
    m = len(diffs)
    # Centred bootstrap: the null distribution is the resampled differences shifted to mean 0.
    mean_d = statistics.fmean(diffs)
    p_greater = (1 + sum(1 for d in diffs if d - mean_d >= obs)) / (m + 1)
    p_less = (1 + sum(1 for d in diffs if d - mean_d <= obs)) / (m + 1)
    return {"diff": obs, "ci_lo": diffs[int(0.025 * m)], "ci_hi": diffs[int(0.975 * m) - 1],
            "p_a_better": p_greater, "p_b_better": p_less}


def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, float] = {}
    running = 0.0
    for rank, (k, p) in enumerate(items):
        adj = min(1.0, (m - rank) * p)
        running = max(running, adj)
        out[k] = running
    return out


# --- scoring ----------------------------------------------------------------------------------


def score(res: RunResult, hold_rets: Sequence[float], windows: dict[str, set[int]],
          *, funding_observed: bool) -> dict[str, Any]:
    led = res.ledger
    r = returns(led.nav)
    s = res.start_i
    guard = windows["macro"] | windows["shock"] | windows["divergence"]
    in_guard = [r[i - s] for i in sorted(guard) if s <= i < res.end_i]
    wk = window_losses(led.nav, s, windows["weekend"])
    alpha = newey_west_alpha(r, hold_rets) if res.name != "hold" else None
    return {
        "name": res.name, "period": res.period, "hours": len(r),
        "total_return": led.nav[-1] / START_NAV - 1,
        "ce_gamma": {str(g): ce(r, g) for g in (2, 5, 10, 20)},
        "sharpe": sharpe(r), "ann_vol": statistics.pstdev(r) * math.sqrt(8760),
        "max_drawdown": max_drawdown(led.nav),
        "cvar95_hourly_all": cvar(r), "cvar95_risk_windows": cvar(in_guard),
        "risk_window_hours": len(in_guard),
        "cvar95_by_family": {k: cvar([r[i - s] for i in sorted(v) if s <= i < res.end_i])
                             for k, v in windows.items() if v},
        "weekend_windows": len(wk), "weekend_mean": statistics.fmean(wk) if wk else None,
        "weekend_worst": min(wk) if wk else None,
        "costs_usd": {"fees": led.fees, "slippage": led.slippage,
                      "funding": led.funding if funding_observed else None,
                      "funding_observed": funding_observed},
        "turnover_x_nav": led.traded / START_NAV, "fills": led.fills,
        "stale_fills": led.stale_fills, "refused_orders": led.refused_orders,
        "mean_net_exposure": statistics.fmean(led.net_exposure) if led.net_exposure else None,
        "timing_vs_hold": alpha, "actions": led.actions, "notes": led.notes[:6],
        "diagnostics": res.finish, "seconds": round(res.seconds, 1),
    }


def compare(results: dict[str, RunResult], *, target: str, rivals: Sequence[str],
            reps: int = 2000, seed: int = 7, gamma: float = 5.0) -> dict[str, Any]:
    ra = returns(results[target].ledger.nav)
    n = len(ra)
    paths = stationary_bootstrap(n, mean_block=24.0, reps=reps, seed=seed)
    out: dict[str, Any] = {}
    raw: dict[str, float] = {}
    rev: dict[str, float] = {}
    for name in rivals:
        rb = returns(results[name].ledger.nav)
        t_ce = paired_test(ra, rb, lambda xs: ce(xs, gamma), paths)
        t_sh = paired_test(ra, rb, sharpe, paths)
        t_cv = paired_test(ra, rb, lambda xs: -cvar(xs), paths)
        out[name] = {"ce5": t_ce, "sharpe": t_sh, "neg_cvar95_all_hours": t_cv}
        raw[name] = t_ce["p_a_better"]
        rev[name] = t_ce["p_b_better"]
    adj = holm(raw)
    adj_rev = holm(rev)
    for name in rivals:
        verdict = ("WIN" if adj[name] < 0.05 else "LOSS" if adj_rev[name] < 0.05 else "TIE")
        out[name]["holm_p_argus_better"] = adj[name]
        out[name]["holm_p_rival_better"] = adj_rev[name]
        out[name]["verdict"] = verdict
    return out


# --- the whole run ----------------------------------------------------------------------------


ABLATIONS: dict[str, RouterConfig] = {
    "argus": RouterConfig(),
    "argus-no_funding": RouterConfig(use_funding=False),
    "argus-no_costs_in_objective": RouterConfig(use_costs=False),
    "argus-no_stability_gate": RouterConfig(use_stability_gate=False),
    "argus-same_name_only": RouterConfig(cross_asset=False),
    "argus-no_regime": RouterConfig(use_regime=False),
    "argus-no_phase": RouterConfig(use_phase=False),
    "argus-gamma2": RouterConfig(gamma=2.0),
    "argus-gamma10": RouterConfig(gamma=10.0),
    "argus-gamma20": RouterConfig(gamma=20.0),
}


def contestants(*, rivals: bool = True, ablations: bool = True) -> list[Any]:
    shared: dict[tuple[Any, ...], RiskModel] = {}
    out: list[Any] = [Hold(), SessionHedge(weekends_only=False),
                      SessionHedge(weekends_only=True), VolTarget()]
    names = ABLATIONS if ablations else {"argus": ABLATIONS["argus"]}
    for name, cfg in names.items():
        out.append(Argus(cfg, name=name, model_cache=shared))
    if rivals:
        out.extend(rival_contestants())
    return out


UNAVAILABLE: dict[str, str] = {}
"""``period:contestant`` -> why it could not be run, filled by :func:`run_period`."""


def run_period(tape: Tape, period: str, *, rivals: bool = True,
               ablations: bool = True, log: Callable[[str], None] = print) -> dict[str, RunResult]:
    results: dict[str, RunResult] = {}
    for c in contestants(rivals=rivals, ablations=ablations):
        try:
            res = run(tape, c, period)
        except RuntimeError as exc:
            # A rival that cannot run (missing clone requirement, runner error) is recorded as
            # unavailable by the caller, never silently dropped and never scored as flat.
            UNAVAILABLE[f"{period}:{c.name}"] = str(exc)[:600]
            proc = getattr(c, "proc", None)
            if proc is not None and proc.poll() is None:
                proc.kill()
            log(f"  {period} {c.name:<32} UNAVAILABLE {str(exc)[:160]}")
            continue
        results[res.name] = res
        log(f"  {period} {res.name:<32} NAV {res.ledger.nav[-1]:>11,.2f}  fills "
            f"{res.ledger.fills:>5}  {res.seconds:6.1f}s")
    return results


def reproducibility(tape: Tape, period: str) -> dict[str, Any]:
    """Run ARGUS twice from a cold start; the NAV paths must be bit-identical."""
    a = run(tape, Argus(RouterConfig()), period)
    b = run(tape, Argus(RouterConfig()), period)
    return {"identical_nav_path": a.ledger.nav == b.ledger.nav,
            "final_nav": [a.ledger.nav[-1], b.ledger.nav[-1]], "fills": [a.ledger.fills,
                                                                          b.ledger.fills]}


def _case_model(keys: tuple[str, ...], sd: Sequence[float], corr: Sequence[Sequence[float]],
                *, scale: float = 1.0, ok: bool = True) -> RiskModel:
    """A hand-built forecast: ``cov_normal_h`` from ``sd``/``corr``, ``cov_h`` = ``scale`` x it."""
    n = len(keys)
    cov = [[corr[a][b] * sd[a] * sd[b] for b in range(n)] for a in range(n)]
    return RiskModel(keys, 8, SHUT, [[scale * v for v in row] for row in cov], cov,
                     {k: scale for k in keys}, (), 40 if ok else 3, ok,
                     "" if ok else "3 shut-phase blocks in the lookback, 20 required")


def adversarial_cases() -> dict[str, dict[str, Any]]:
    """Constructed inputs to the router, each with the pass condition stated before it is run.

    These isolate the router's decision rule from the tape: every forecast, funding rate and cost
    is set by hand, so each case asks one question about behaviour, not about a market. A rToken
    long of $30,000, a BTC perpetual long of $25,000, the NVDA perpetual unheld; ``perp:NVDAUSDT``
    and ``perp:BTCUSDT`` are the tradeable legs.
    """
    keys = ("spot:RNVDAUSDT", "perp:NVDAUSDT", "perp:BTCUSDT")
    sd = (0.02, 0.02, 0.025)
    corr = ((1.0, 0.9, 0.3), (0.9, 1.0, 0.3), (0.3, 0.3, 1.0))
    cheap = CostCurve(6.0, ((1_000.0, 1.0), (40_000.0, 5.0)))
    prohibitive = CostCurve(5_000.0, ((1_000.0, 1.0), (40_000.0, 5.0)))
    cfg = RouterConfig()

    def inp(*, f_now: float = 0.0001, cost: CostCurve = cheap) -> RouterInput:
        return RouterInput(
            nav=100_000.0, spot_usd={"spot:RNVDAUSDT": 30_000.0},
            perp_usd={"perp:NVDAUSDT": 0.0, "perp:BTCUSDT": 25_000.0},
            base_perp_usd={"perp:BTCUSDT": 25_000.0},
            funding_now={"perp:NVDAUSDT": 0.0001, "perp:BTCUSDT": f_now},
            funding_normal={"perp:NVDAUSDT": 0.0001, "perp:BTCUSDT": 0.0001},
            settlements_ahead=3, costs={"perp:NVDAUSDT": cost, "perp:BTCUSDT": cost})

    tradeable = ["perp:NVDAUSDT", "perp:BTCUSDT"]
    out: dict[str, dict[str, Any]] = {}

    def record(name: str, condition: str, passed: bool, d: Any) -> None:
        out[name] = {"condition": condition, "passed": passed, "decision": d.as_dict()}

    d = route(_case_model(keys, sd, corr), inp(), cfg, tradeable=tradeable)
    record("normal_risk_normal_carry", "forecast equals normal and funding equals normal: "
           "no order (the holder's own book is optimal)", not d.orders_usd and not d.refused, d)

    d = route(_case_model(keys, sd, corr, scale=4.0), inp(), cfg, tradeable=tradeable)
    record("risk_shock", "forecast variance x4 on every leg: sells at least one perpetual, buys "
           "none, and modelled risk falls",
           bool(d.orders_usd) and all(v < 0 for v in d.orders_usd.values())
           and d.risk_after_usd < d.risk_before_usd, d)

    d = route(_case_model(keys, sd, corr), inp(f_now=0.003), cfg, tradeable=tradeable)
    record("funding_spike", "BTC funding 30x its normal rate over 3 settlements, risk normal: "
           "reduces the BTC long", d.orders_usd.get("perp:BTCUSDT", 0.0) < 0, d)

    d = route(_case_model(keys, sd, corr, scale=4.0), inp(cost=prohibitive), cfg,
              tradeable=tradeable)
    record("prohibitive_costs", "the same x4 shock with a 50% taker fee: no order",
           not d.orders_usd, d)

    d = route(_case_model(keys, sd, corr, scale=0.25), inp(), cfg, tradeable=tradeable)
    record("calm_forecast", "forecast variance x0.25: never adds exposure (hedge-only bound) — "
           "no buy of either perpetual", all(v <= 0 for v in d.orders_usd.values()), d)

    d = route(_case_model(keys, sd, corr, ok=False), inp(), cfg, tradeable=tradeable)
    record("thin_history", "too few phase blocks to estimate: refuses and keeps the book",
           d.refused and not d.orders_usd, d)
    return out


RIVAL_NAMES = ("triad", "triad-x10", "omni", "crossfire", "vigil", "hedgeagents",
               "hedgeagents-optimizer")
BASELINE_NAMES = ("hold", "static_session_hedge", "static_weekend_hedge", "vol_target_overlay")


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI
    args = list(argv if argv is not None else sys.argv[1:])
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    tape = load()
    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "tape": {"digest": tape_digest(tape.manifest), "window": [
            tape.manifest["window_start"], tape.manifest["window_end"]],
            "funding_observed_from": datetime.fromtimestamp(
                tape.funding_observed_from / 1000, UTC).isoformat()},
        "prereg": PREREG, "rival_notes": RIVAL_NOTES, "periods": {},
    }
    periods = [p for p in ("B", "A") if not args or p in args]
    for period in periods:
        print(f"period {period}")
        results = run_period(tape, period)
        s = next(iter(results.values())).start_i
        e = next(iter(results.values())).end_i
        windows = risk_windows(tape, s, e)
        hold_r = returns(results["hold"].ledger.nav)
        fund = tape.funding_observed_from <= tape.hours[s]
        scores = {n: score(r, hold_r, windows, funding_observed=fund) for n, r in results.items()}
        rivals_present = [n for n in (*RIVAL_NAMES, *BASELINE_NAMES) if n in results]
        comp = compare(results, target="argus", rivals=rivals_present)
        ablation_comp = compare(results, target="argus",
                                rivals=[n for n in results if n.startswith("argus-")])
        report["periods"][period] = {
            "hours": e - s, "funding_observed": fund,
            "risk_window_hours": {k: len(v) for k, v in windows.items()},
            "scores": scores, "argus_vs": comp, "argus_vs_ablations": ablation_comp,
        }
    if "B" in periods:
        report["reproducibility"] = reproducibility(tape, "B")
    report["adversarial_cases"] = adversarial_cases()
    report["unavailable"] = dict(UNAVAILABLE)
    undefined = artefact.write(REPORT_PATH, report)
    print(f"written {REPORT_PATH} ({len(undefined)} undefined values recorded as null)")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = [
    "ABLATIONS", "PERIODS", "PREREG", "RIVAL_NOTES", "UNAVAILABLE", "Argus", "Book", "Contestant",
    "History",
    "Hold", "Ledger", "SessionHedge", "Subprocess", "VolTarget", "adversarial_cases", "ce",
    "compare", "contestants",
    "cvar", "holm", "main", "max_drawdown", "newey_west_alpha", "paired_test", "returns",
    "risk_windows", "run", "run_period", "score", "sharpe", "stationary_bootstrap",
    "window_losses",
]
