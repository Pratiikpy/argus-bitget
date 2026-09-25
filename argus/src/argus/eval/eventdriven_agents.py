"""Event -> decision -> trade: a real S2 Event-Driven agent's own events, gated four ways, traded.

`eval/eventdriven_rivals.py` asks which significance test is honest when events cluster. This
module asks the sub-theme's question — *how do events drive autonomous trading?* — on the real
event stream of a real Season-2 Event-Driven entry, and scores the trades.

**The shared input is slimon's own perception, run unmodified.** JohnboscoE/slimon trades
Bitget's US-stock perpetuals on the demo venue; its perception layer turns closed 5-minute candles
into ``price_move`` / ``range_expansion`` / ``volume_spike`` events with the thresholds in its
committed ``config/agent.toml``, and its LLM decides from there. That perception is deterministic,
so it is replayed here — their real ``Perception.detect`` on real Bitget 5-minute candles
(`data/eventdriven_rivals_5m.json.gz`, 95 days, its 12-symbol watchlist). The replay is checked
against slimon's own published event log first: driven at the exact tick times their agent ran,
it must reproduce the events they logged, or nothing downstream is their perception.

**Four gates decide which event classes are worth trading**, each on a training half, each with
its own real test, each held to the same 12bps round-trip hurdle (slimon's own 0.06% taker fee,
twice; ARGUS's own round trip). Then every gate trades the other half, and the trades are scored:

* **ARGUS** — `research/eventstudy.py`: trade a class only if all four tests reject at 5% after
  the Kolari-Pynnonen clustering deflation and the average abnormal return clears the hurdle.
* **Vibe-Trading** — its real `quantlib.event_study`, BMP p <= 0.05 (its own advice: "When the
  three disagree, BMP is the one to report"), same hurdle on its CAAR.
* **Ballast** — its real pooled ``t_stat`` on raw returns, |t| >= 2 as its Gate 1b reads it.
* **whale-signals** — its real ``compute_hit_rates``: fixed-50% binomial on direction.
* Two gate-free baselines — follow every event in its own direction, and fade every event.

Every gate's trades are the same instrument: the single name itself, unhedged, entered at the event
candle's close and closed six hours later, one position per name at a time, net of the hurdle. The
gates that test abnormal returns (ARGUS, Vibe-Trading) therefore decide on a market-relative effect
and are paid on the raw one; a class they pass on a market-neutral effect can still lose on beta.

slimon's own LLM decisions are not re-bought (the Qwen balance is capped and a handful of calls
proves nothing); its real, published decisions are scored directly instead, on the same clock.

**Nicholas-03/trading-bot's hard-catalyst gate** runs unmodified on slimon's 1,448 real logged
headlines: which ones it passes, and whether the ones it passes move the name more than the ones
it rejects — measured with ARGUS's abnormal-return machinery.
"""

from __future__ import annotations

import contextlib
import gzip
import io
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from math import sqrt
from pathlib import Path
from typing import Any

from argus.eval.baselines.eventdriven_agents_loader import (
    SLIMON_CLONE,
    load_nicholas_filters,
    load_slimon,
    provenance,
    slimon_config,
)
from argus.eval.baselines.vibe_trading_eventstudy_loader import (
    load_ballast_stats,
    load_vibe_eventstudy,
)
from argus.eval.baselines.whale_signals_event_study_loader import load_event_study_module
from argus.research.eventstudy import (
    EventStudyError,
    EventWindow,
    build_window,
    returns_from,
    study,
)

DATA = Path(__file__).resolve().parents[3] / "data"
SNAPSHOT_5M = DATA / "eventdriven_rivals_5m.json.gz"
REPORT_PATH = DATA / "eventdriven_agents.json"
CHAINS_PATH = DATA / "causal_chains.jsonl"

MARKET = "NDX100USDT"
INDEX_PERPS = frozenset({"NDX100USDT", "SP500USDT"})
BAR_MS = 300_000
HOLD_BARS = 72
"""Six hours of 5-minute bars: the event window, the holding period, and the whale-signals
horizon (its ``fwd_return_6h``). One horizon, fixed before any result was seen."""
ESTIMATION_BARS = 2016
"""A week of 5-minute bars, 24/7. Long enough for a stable beta against NDX100USDT."""
GAP_BARS = 72
COST_BPS = 12.0
DIRECTIONAL_TYPES = ("price_move", "range_expansion", "volume_spike")


class AgentStudyError(RuntimeError):
    """The comparison could not be run as specified."""


# --- the tape ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Tape:
    """Real 5-minute candles on a grid every watched symbol shares."""

    ms: tuple[int, ...]
    rows: dict[str, list[list[float]]]
    closes: dict[str, list[float]]
    returns: dict[str, list[float]]
    stamps: tuple[datetime, ...]
    """``stamps[k]`` is the bar whose return is ``returns[s][k]`` (close k -> close k+1)."""
    index: dict[int, int]
    source_sha256: str
    fetched_at: str


def load_tape(path: Path = SNAPSHOT_5M) -> Tape:
    import hashlib

    raw = path.read_bytes()
    blob = json.loads(gzip.decompress(raw))
    candles: dict[str, list[list[float]]] = blob["candles"]
    common = sorted(set.intersection(*({int(r[0]) for r in rows} for rows in candles.values())))
    if len(common) < ESTIMATION_BARS + GAP_BARS + HOLD_BARS + 1000:
        raise AgentStudyError(f"only {len(common)} aligned 5m bars in {path}")
    keep = set(common)
    rows = {s: sorted((r for r in v if int(r[0]) in keep), key=lambda r: r[0])
            for s, v in candles.items()}
    closes = {s: [float(r[4]) for r in v] for s, v in rows.items()}
    return Tape(
        ms=tuple(common), rows=rows, closes=closes,
        returns={s: returns_from(c) for s, c in closes.items()},
        stamps=tuple(datetime.fromtimestamp(m / 1000, tz=UTC) for m in common[1:]),
        index={m: i for i, m in enumerate(common)},
        source_sha256=hashlib.sha256(raw).hexdigest(),
        fetched_at=str(blob.get("fetched_at", "")),
    )


# --- slimon's perception, replayed -------------------------------------------------------------


def _iso_ms(text: str) -> int:
    return int(datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp() * 1000)


def replay_perception(tape: Tape, tick_ms: Sequence[int]) -> list[dict[str, Any]]:
    """slimon's real ``Perception.detect`` at each tick, with only closed candles visible.

    Mirrors ``slimon/market.py:Market.snapshot``: a candle is visible once ``ts + step <= now``,
    and each view holds the last ``lookback_candles + 1`` of them. The agent's portfolio is flat
    — the position-review events a held book would add are not a function of the tape.
    """
    mods = load_slimon()
    cfg = slimon_config()["perception"]
    lookback = int(cfg["lookback_candles"])
    Candle = mods["market"].Candle
    SymbolView = mods["market"].SymbolView
    MarketSnapshot = mods["market"].MarketSnapshot
    us_session = mods["market"].us_session
    state = mods["journal"].State("argus-replay-never-saved.json")
    state.data = {}
    perception = mods["perception"].Perception(cfg, state)
    portfolio = mods["broker"].Portfolio(source="argus-replay", equity=0.0, unrealized_pnl=0.0,
                                         positions=[])
    symbols = list(tape.rows)
    built = {s: [Candle(int(r[0]), r[1], r[2], r[3], r[4], r[5]) for r in tape.rows[s]]
             for s in symbols}
    pointer = dict.fromkeys(symbols, 0)
    events: list[dict[str, Any]] = []
    for now_ms in sorted(tick_ms):
        now = datetime.fromtimestamp(now_ms / 1000, tz=UTC)
        views = {}
        for s in symbols:
            seq = built[s]
            p = pointer[s]
            while p < len(seq) and seq[p].ts + BAR_MS <= now_ms:
                p += 1
            pointer[s] = p
            views[s] = SymbolView(s, seq[max(0, p - (lookback + 1)):p], None, None, None)
        snap = MarketSnapshot(now, views, [])
        events.extend(perception.detect(snap, portfolio, now, us_session(now)))
    return events


def every_tick(tape: Tape) -> list[int]:
    """One tick 15 seconds after every candle close, as slimon's loop schedules them."""
    return [m + BAR_MS + 15_000 for m in tape.ms]


def slimon_logged(kind: str) -> list[dict[str, Any]]:
    out = []
    for path in sorted((SLIMON_CLONE / "logs" / kind).glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                with contextlib.suppress(json.JSONDecodeError):
                    out.append(json.loads(line))
    return out


def _event_key(e: dict[str, Any]) -> tuple[str, str, str]:
    if e.get("symbol"):
        return (e["type"], e["symbol"], str(e.get("source_ts") or ""))
    return (e["type"], "", str(e.get("received_at", ""))[:10])


def validate_replay(tape: Tape) -> dict[str, Any]:
    """Drive the replay at the tick times slimon's own agent ran, and compare with its log.

    Candle-derived and session events only: news and position reviews are not a function of the
    tape. A tick is counted only when the tape covers its whole 4-hour lookback.
    """
    decisions = slimon_logged("decisions")
    first_ok = tape.ms[0] + 60 * BAR_MS
    ticks = sorted({_iso_ms(r["ts"]) for r in decisions if r.get("ts")})
    ticks = [t for t in ticks if first_ok <= t <= tape.ms[-1] + BAR_MS + 60_000]
    if not ticks:
        raise AgentStudyError("no slimon tick falls inside the tape")
    lo, hi = ticks[0], ticks[-1]
    kinds = (*DIRECTIONAL_TYPES, "us_regular_open", "us_regular_close")
    theirs = {_event_key(e) for e in slimon_logged("events")
              if e.get("type") in kinds and lo <= _iso_ms(e["received_at"]) <= hi}
    ours = {_event_key(e) for e in replay_perception(tape, ticks) if e.get("type") in kinds}
    matched = theirs & ours
    by_type: dict[str, dict[str, int]] = {}
    for kind in kinds:
        t = {k for k in theirs if k[0] == kind}
        o = {k for k in ours if k[0] == kind}
        by_type[kind] = {"logged_by_slimon": len(t), "replayed": len(o), "matched": len(t & o)}
    return {
        "ticks_replayed": len(ticks),
        "window": [datetime.fromtimestamp(lo / 1000, tz=UTC).isoformat(),
                   datetime.fromtimestamp(hi / 1000, tz=UTC).isoformat()],
        "logged_by_slimon": len(theirs),
        "replayed": len(ours),
        "matched": len(matched),
        "recall_of_their_log": round(len(matched) / len(theirs), 4) if theirs else None,
        "precision_against_their_log": round(len(matched) / len(ours), 4) if ours else None,
        "by_type": by_type,
        "unmatched_examples": sorted(theirs - ours)[:5],
        "extra_examples": sorted(ours - theirs)[:5],
    }


# --- event classes -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Trigger:
    """One directional event on a single name, placed on the tape."""

    symbol: str
    kind: str
    direction: int
    j: int
    """Index of the event candle in the tape: entry at its close, window starts at return j."""

    @property
    def label(self) -> str:
        return f"{self.kind}_{'up' if self.direction > 0 else 'down'}"


def _direction(event: dict[str, Any]) -> int:
    payload = event.get("payload") or {}
    if event["type"] == "price_move":
        value = float(payload.get("return_pct", 0.0))
    elif event["type"] == "range_expansion":
        candle = payload.get("candle") or {}
        value = float(candle.get("c", 0.0)) - float(candle.get("o", 0.0))
    else:
        value = float(payload.get("candle_return_pct", 0.0))
    return (value > 0) - (value < 0)


def triggers_from(tape: Tape, events: Sequence[dict[str, Any]]) -> list[Trigger]:
    out = []
    first = ESTIMATION_BARS + GAP_BARS + 2
    last = len(tape.ms) - HOLD_BARS - 2
    for e in events:
        if e.get("type") not in DIRECTIONAL_TYPES or not e.get("symbol"):
            continue
        if e["symbol"] in INDEX_PERPS:
            continue
        d = _direction(e)
        j = tape.index.get(_iso_ms(e["source_ts"])) if e.get("source_ts") else None
        if d == 0 or j is None or not first <= j <= last:
            continue
        out.append(Trigger(e["symbol"], e["type"], d, j))
    out.sort(key=lambda t: (t.j, t.symbol, t.kind))
    return out


def deoverlap(triggers: Sequence[Trigger]) -> list[Trigger]:
    """Per symbol and class, drop a trigger whose window overlaps the previous one kept.

    Overlapping windows of the same name are the same observation counted twice, and no test
    here — ARGUS's included — corrects for that; removing them is the only honest option.
    """
    last: dict[tuple[str, str], int] = {}
    out = []
    for t in triggers:
        key = (t.symbol, t.label)
        if key in last and t.j < last[key] + HOLD_BARS:
            continue
        last[key] = t.j
        out.append(t)
    return out


def raw_return(tape: Tape, symbol: str, j: int) -> float:
    closes = tape.closes[symbol]
    return closes[j + HOLD_BARS] / closes[j] - 1.0


# --- the gates ---------------------------------------------------------------------------------


@dataclass
class GateCall:
    trade: bool
    direction: int
    events: int
    effect_bps: float | None
    detail: dict[str, Any] = field(default_factory=dict)


def _windows(tape: Tape, triggers: Sequence[Trigger]) -> list[EventWindow]:
    out = []
    for t in triggers:
        w = build_window(t.symbol, tape.stamps[t.j - 1], tape.stamps, tape.returns[t.symbol],
                         tape.returns[MARKET], event_bars=HOLD_BARS,
                         estimation_bars=ESTIMATION_BARS, gap_bars=GAP_BARS)
        if w is not None:
            out.append(w)
    return out


def gate_argus(tape: Tape, triggers: Sequence[Trigger], *, adjusted: bool = True) -> GateCall:
    windows = _windows(tape, triggers)
    try:
        result = study("gate", windows, event_bars=HOLD_BARS)
        stats = result.statistics
    except EventStudyError as exc:
        return GateCall(False, 0, len(windows), None, {"refused": str(exc)})
    key = "adjusted_p_value" if adjusted else "p_value"
    worst = max(row[key] for row in stats.values())
    effect = result.average_car_bps
    trade = worst <= 0.05 and abs(effect) > COST_BPS
    return GateCall(trade, (effect > 0) - (effect < 0), len(windows), round(effect, 2), {
        "largest_p": round(worst, 6), "cross_correlation": round(result.correlation, 5),
        "p_values": {n: round(row[key], 6) for n, row in stats.items()},
    })


class VibeGate:
    def __init__(self, tape: Tape) -> None:
        import pandas as pd

        self.vibe = load_vibe_eventstudy()
        index = pd.DatetimeIndex(list(tape.stamps))
        self.index = index
        self.frame = pd.DataFrame({s: r for s, r in tape.returns.items() if s != MARKET},
                                  index=index)
        self.market = pd.Series(tape.returns[MARKET], index=index)

    def __call__(self, triggers: Sequence[Trigger]) -> GateCall:
        if len(triggers) < 2:
            return GateCall(False, 0, len(triggers), None, {"refused": "fewer than 2 events"})
        try:
            r = self.vibe.event_study(
                self.frame, self.market, [(t.symbol, self.index[t.j]) for t in triggers],
                event_window=(0, HOLD_BARS - 1), estimation_window=ESTIMATION_BARS,
                estimation_gap=GAP_BARS, model="market")
        except ValueError as exc:
            return GateCall(False, 0, len(triggers), None, {"refused": str(exc)})
        effect = float(r.caar_total) * 10_000
        p = float(r.bmp_p_value)
        trade = p <= 0.05 and abs(effect) > COST_BPS
        return GateCall(trade, (effect > 0) - (effect < 0), len(r.events), round(effect, 2), {
            "bmp_p": round(p, 6), "t_p": round(float(r.t_p_value), 6),
            "flagged_shared_dates": len(r.shared_event_dates), "dropped": len(r.dropped),
        })


def gate_ballast(tape: Tape, triggers: Sequence[Trigger]) -> GateCall:
    raws = [raw_return(tape, t.symbol, t.j) for t in triggers]
    mean, t = load_ballast_stats().t_stat(raws)
    if t != t:  # their function returns NaN below eight observations
        return GateCall(False, 0, len(raws), None, {"refused": "fewer than 8 events"})
    effect = mean * 10_000
    trade = abs(t) >= 2.0 and abs(effect) > COST_BPS
    return GateCall(trade, (effect > 0) - (effect < 0), len(raws), round(effect, 2),
                    {"t": round(float(t), 4)})


def gate_whale(tape: Tape, triggers: Sequence[Trigger], direction: int) -> GateCall:
    """Their real ``compute_hit_rates``: an 'up' class is a buy signal (exchange_withdrawal, a hit
    when price rises), a 'down' class a sell signal (exchange_deposit, a hit when it falls)."""
    import pandas as pd

    if not triggers:
        # Their function indexes a `tx_category` column an empty frame does not have; with no
        # events there is nothing for it to decide, so the refusal is ours, not a crash in theirs.
        return GateCall(False, 0, 0, None, {"refused": "no events in this class"})
    category = "exchange_withdrawal" if direction > 0 else "exchange_deposit"
    rows: list[dict[str, Any]] = []
    for t in triggers:
        closes = tape.closes[t.symbol]
        rows.append({"tx_category": category,
                     "fwd_return_1h": closes[t.j + 12] / closes[t.j] - 1.0,
                     "fwd_return_6h": closes[t.j + HOLD_BARS] / closes[t.j] - 1.0,
                     "fwd_return_24h": closes[min(t.j + 288, len(closes) - 1)] / closes[t.j] - 1.0})
    with contextlib.redirect_stdout(io.StringIO()):
        got: dict[int, dict[str, float]] = load_event_study_module().compute_hit_rates(
            pd.DataFrame(rows)).get(category, {})
    row = got.get(6)
    if not row:
        return GateCall(False, 0, len(rows), None, {"refused": "fewer than 30 events"})
    raw = sum(r["fwd_return_6h"] for r in rows) / len(rows) * 10_000
    # Their hit rate and p-value are numpy scalars; plain Python values go into the artefact.
    hit_rate, p = float(row["hit_rate"]), float(row["pvalue"])
    trade_dir = direction if hit_rate > 0.5 else -direction
    trade = p < 0.05 and trade_dir * raw > COST_BPS
    return GateCall(trade, trade_dir, len(rows), round(raw, 2),
                    {"hit_rate": round(hit_rate, 4), "p": round(p, 6)})


# --- the trades --------------------------------------------------------------------------------


def trade(tape: Tape, triggers: Sequence[Trigger], plan: dict[str, int]) -> list[dict[str, Any]]:
    """Every trigger whose class the plan trades, one position per name at a time."""
    busy_until: dict[str, int] = {}
    out = []
    for t in triggers:
        side = plan.get(t.label, 0)
        if not side or t.j < busy_until.get(t.symbol, -1):
            continue
        busy_until[t.symbol] = t.j + HOLD_BARS
        net = side * raw_return(tape, t.symbol, t.j) * 10_000 - COST_BPS
        out.append({"symbol": t.symbol, "class": t.label, "side": side,
                    "at": datetime.fromtimestamp(tape.ms[t.j] / 1000, tz=UTC).isoformat(),
                    "net_bps": round(net, 3)})
    return out


def score_trades(trades: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Net result, with a t clustered by UTC day: same-day trades share one market."""
    if not trades:
        return {"trades": 0, "net_bps_total": 0.0, "net_bps_mean": None, "hit_rate": None,
                "day_clustered_t": None, "days": 0}
    nets = [t["net_bps"] for t in trades]
    days: dict[str, float] = {}
    for t in trades:
        days[t["at"][:10]] = days.get(t["at"][:10], 0.0) + t["net_bps"]
    daily = list(days.values())
    tstat = None
    if len(daily) > 2:
        mean = sum(daily) / len(daily)
        var = sum((x - mean) ** 2 for x in daily) / (len(daily) - 1)
        tstat = round(mean / sqrt(var / len(daily)), 3) if var > 0 else None
    return {"trades": len(nets), "net_bps_total": round(sum(nets), 2),
            "net_bps_mean": round(sum(nets) / len(nets), 3),
            "hit_rate": round(sum(1 for x in nets if x > 0) / len(nets), 4),
            "day_clustered_t": tstat, "days": len(daily)}


GATES = ("argus", "argus_without_clustering_adjustment", "vibe_trading_bmp",
         "ballast_pooled_t", "whale_signals_fixed_null", "no_gate_follow", "no_gate_fade")


def walk_forward(tape: Tape, triggers: Sequence[Trigger]) -> dict[str, Any]:
    """Two folds: learn on one half of the calendar, trade the other, then swap."""
    vibe = VibeGate(tape)
    classes = sorted({t.label for t in triggers})
    mid = (triggers[0].j + triggers[-1].j) // 2
    halves = {"first": [t for t in triggers if t.j < mid - HOLD_BARS],
              "second": [t for t in triggers if t.j >= mid]}
    folds = {}
    all_trades: dict[str, list[dict[str, Any]]] = {g: [] for g in GATES}
    for name, (train_key, test_key) in {"train_first_trade_second": ("first", "second"),
                                        "train_second_trade_first": ("second", "first")}.items():
        train = deoverlap(halves[train_key])
        calls: dict[str, dict[str, GateCall]] = {g: {} for g in GATES}
        for label in classes:
            members = [t for t in train if t.label == label]
            direction = members[0].direction if members else 0
            calls["argus"][label] = gate_argus(tape, members)
            calls["argus_without_clustering_adjustment"][label] = gate_argus(
                tape, members, adjusted=False)
            calls["vibe_trading_bmp"][label] = vibe(members)
            calls["ballast_pooled_t"][label] = gate_ballast(tape, members)
            calls["whale_signals_fixed_null"][label] = gate_whale(tape, members, direction)
            calls["no_gate_follow"][label] = GateCall(True, direction, len(members), None)
            calls["no_gate_fade"][label] = GateCall(True, -direction, len(members), None)
        fold: dict[str, Any] = {"train_triggers": len(train),
                                "test_triggers": len(halves[test_key])}
        for gate in GATES:
            plan = {label: c.direction for label, c in calls[gate].items() if c.trade}
            trades = trade(tape, halves[test_key], plan)
            all_trades[gate].extend(trades)
            fold[gate] = {
                "classes_traded": plan,
                "calls": {label: {"trade": c.trade, "direction": c.direction, "events": c.events,
                                  "effect_bps": c.effect_bps, **c.detail}
                          for label, c in calls[gate].items()},
                "test": score_trades(trades),
            }
        folds[name] = fold
    combined = {g: score_trades(all_trades[g]) for g in GATES}
    return {"classes": classes, "folds": folds, "out_of_sample_combined": combined,
            "test_trades": {g: all_trades[g] for g in GATES}}


# --- slimon's own LLM decisions ----------------------------------------------------------------


def slimon_llm_decisions(tape: Tape) -> dict[str, Any]:
    """Every OPEN proposal slimon's model made, scored on the same clock and hurdle.

    Uses their real ``report.summary`` for their own headline, unmodified, beside ARGUS's study of
    the single-name proposals. Index perps are scored on raw return only: they are the market.
    """
    mods = load_slimon()
    rows = slimon_logged("decisions")
    rows.sort(key=lambda r: r.get("ts", ""))
    with contextlib.redirect_stdout(io.StringIO()):
        own = mods["report"].summary(rows)
    own.pop("trips", None)
    proposals = []
    single: list[Trigger] = []
    for r in rows:
        d = r.get("decision") or {}
        if d.get("action") not in ("OPEN_LONG", "OPEN_SHORT"):
            continue
        tick = _iso_ms(r["ts"])
        candle = (tick - 15_000) - BAR_MS
        j = tape.index.get(candle - candle % BAR_MS)
        side = 1 if d["action"] == "OPEN_LONG" else -1
        symbol = d.get("instrument", "")
        verdict = (r.get("risk") or {}).get("verdict")
        row: dict[str, Any] = {"at": r["ts"], "symbol": symbol, "side": side,
                               "confidence": d.get("confidence"), "gate": verdict,
                               "executed": bool((r.get("execution") or {}).get("status"))}
        if j is not None and j + HOLD_BARS < len(tape.ms) and symbol in tape.closes:
            row["net_bps_6h"] = round(side * raw_return(tape, symbol, j) * 10_000 - COST_BPS, 2)
            if symbol not in INDEX_PERPS and side > 0:
                single.append(Trigger(symbol, "llm_open", side, j))
        proposals.append(row)
    scored = [p for p in proposals if "net_bps_6h" in p]
    verdict_text = None
    study_row: dict[str, Any] = {}
    windows = _windows(tape, single)
    try:
        result = study("slimon-llm", windows, event_bars=HOLD_BARS)
        verdict_text = result.verdict
        study_row = {n: {"adjusted_p_value": row["adjusted_p_value"]}
                     for n, row in result.statistics.items()}
        study_row["average_car_bps"] = round(result.average_car_bps, 2)
    except EventStudyError as exc:
        verdict_text = f"refused: {exc}"
    return {
        "their_own_report_summary": own,
        "proposals": proposals,
        "scored_on_tape": len(scored),
        "net_bps_6h_mean": round(sum(p["net_bps_6h"] for p in scored) / len(scored), 2)
        if scored else None,
        "argus_study_of_single_name_long_proposals": {"events": len(windows),
                                                      "verdict": verdict_text, **study_row},
    }


# --- Nicholas-03's hard-catalyst gate on slimon's real headlines -------------------------------


def nicholas_gate(tape: Tape) -> dict[str, Any]:
    """Their real filters, unmodified, on every headline slimon logged; then ARGUS measures how
    far each group of headlines moved its name over six hours, abnormal to NDX100USDT."""
    f = load_nicholas_filters()
    news = [e for e in slimon_logged("events") if e.get("type") == "news"]
    passed: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    reasons: dict[str, int] = {}
    for e in news:
        payload = e.get("payload") or {}
        headline = str(payload.get("headline") or "")
        rejections = [name for name, hit in (
            ("retrospective", f.is_retrospective_headline(headline)),
            ("routine", f.is_routine_news(headline)),
            ("vague_or_analyst", f.is_vague_or_analyst_news(headline)),
            ("soft_partnership", f.is_soft_partnership_without_materiality(headline)),
            ("no_hard_catalyst", not f.is_hard_catalyst_news(headline)),
        ) if hit]
        for name in rejections:
            reasons[name] = reasons.get(name, 0) + 1
        (failed if rejections else passed).append(e)

    def moves(group: Sequence[dict[str, Any]]) -> dict[str, Any]:
        seen: set[tuple[str, int]] = set()
        sizes = []
        for e in group:
            symbol = e.get("symbol")
            if not symbol or symbol in INDEX_PERPS or symbol not in tape.closes:
                continue
            published = _iso_ms(e["source_ts"])
            j = tape.index.get(published - published % BAR_MS)
            if j is None or (symbol, j) in seen:
                continue
            seen.add((symbol, j))
            w = build_window(symbol, tape.stamps[j - 1], tape.stamps, tape.returns[symbol],
                             tape.returns[MARKET], event_bars=HOLD_BARS,
                             estimation_bars=ESTIMATION_BARS, gap_bars=GAP_BARS)
            if w is None:
                continue
            blocks = [sum(w.estimation_abnormal[k:k + HOLD_BARS])
                      for k in range(0, len(w.estimation_abnormal) - HOLD_BARS + 1, HOLD_BARS)]
            usual = sum(abs(b) for b in blocks) / len(blocks)
            if usual > 0:
                sizes.append(abs(w.car) / usual)
        if not sizes:
            return {"events": 0, "mean_size_ratio": None}
        mean = sum(sizes) / len(sizes)
        sd = sqrt(sum((x - mean) ** 2 for x in sizes) / (len(sizes) - 1)) if len(sizes) > 1 else 0
        return {"events": len(sizes), "mean_size_ratio": round(mean, 3),
                "stderr": round(sd / sqrt(len(sizes)), 3) if len(sizes) > 1 else None}

    return {"headlines": len(news), "passed": len(passed), "rejected": len(failed),
            "rejection_reasons": reasons,
            "passed_examples": [str((e.get("payload") or {}).get("headline", ""))
                                for e in passed[:8]],
            "abnormal_move_size_ratio": {"passed": moves(passed), "rejected": moves(failed)}}


# --- the chain-falsifier defect, re-measured ---------------------------------------------------


FIX_DATE = "2026-09-24"


def chain_defect_evidence(path: Path = CHAINS_PATH) -> dict[str, Any]:
    """The two EventAnalyst defects the 2026-09-24 review found, counted before and after the fix
    (`agents/analysts.py`: ``chain_falsifiers`` in the prompt, ``chain_event`` skipping the
    ``mkt-`` price line), on the live desk's own recorded chains."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    out: dict[str, Any] = {}
    for name, after in (("before_fix", False), ("after_fix", True)):
        group = [r for r in rows if (str(r.get("decided_at", ""))[:10] >= FIX_DATE) is after]
        links = [link for r in group for link in (r.get("links") or [])]
        price_line = sum(1 for r in group if str(r.get("event", "")).startswith(
            f"{r.get('symbol', '')} last "))
        out[name] = {
            "chains": len(group),
            "links": len(links),
            "links_with_a_falsifier": sum(1 for x in links if str(x.get("falsifier", "")).strip()),
            "chains_every_link_falsifiable": sum(
                1 for r in group if r.get("links") and all(
                    str(x.get("falsifier", "")).strip() for x in r["links"])),
            "event_field_is_the_price_line": price_line,
            "graded_links": sum(1 for r in group for g in (r.get("grades") or [])
                                if g not in ("ungraded", "unsupported")),
        }
    return out


# --- assembly ----------------------------------------------------------------------------------


def run(tape: Tape | None = None) -> dict[str, Any]:
    tape = tape or load_tape()
    started = time.perf_counter()
    validation = validate_replay(tape)
    replay_start = time.perf_counter()
    events = replay_perception(tape, every_tick(tape))
    replay_seconds = time.perf_counter() - replay_start
    triggers = triggers_from(tape, events)
    by_class: dict[str, int] = {}
    for t in triggers:
        by_class[t.label] = by_class.get(t.label, 0) + 1
    gate_start = time.perf_counter()
    forward = walk_forward(tape, triggers)
    gate_seconds = time.perf_counter() - gate_start
    return {
        "generated_from": {
            "snapshot": SNAPSHOT_5M.name, "snapshot_sha256": tape.source_sha256,
            "fetched_at": tape.fetched_at, "bars": len(tape.ms),
            "first_bar": datetime.fromtimestamp(tape.ms[0] / 1000, tz=UTC).isoformat(),
            "last_bar": datetime.fromtimestamp(tape.ms[-1] / 1000, tz=UTC).isoformat(),
            "market": MARKET, "hold_bars": HOLD_BARS, "estimation_bars": ESTIMATION_BARS,
            "gap_bars": GAP_BARS, "cost_bps_round_trip": COST_BPS,
        },
        "rivals": provenance(),
        "replay_reproduces_their_log": validation,
        "perception_events": len(events),
        "triggers_by_class": by_class,
        "walk_forward": forward,
        "slimon_llm_decisions": slimon_llm_decisions(tape),
        "nicholas_03_gate_on_slimon_headlines": nicholas_gate(tape),
        "chain_falsifier_defect": chain_defect_evidence(),
        "costs": {"replay_seconds": round(replay_seconds, 1),
                  "gates_and_trading_seconds": round(gate_seconds, 1),
                  "total_seconds": round(time.perf_counter() - started, 1)},
    }


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI
    """``python -m argus.eval.eventdriven_agents`` — needs the two rival clones (the loader names
    the clone commands) and writes ``data/eventdriven_agents.json``. No model is called."""
    import argparse

    from argus.eval.artefact import write as write_artefact

    argparse.ArgumentParser(description="slimon's own events, gated four ways, traded").parse_args(
        argv)
    report = run()
    non_finite = write_artefact(REPORT_PATH, report)
    replay = report["replay_reproduces_their_log"]
    print(f"[eventdriven-agents] replay matched {replay['matched']} of "
          f"{replay['logged_by_slimon']} logged events (recall {replay['recall_of_their_log']})")
    for gate, row in report["walk_forward"]["out_of_sample_combined"].items():
        print(f"[eventdriven-agents] {gate}: {row['trades']} trades, "
              f"{row['net_bps_total']:+.1f}bps net, day-clustered t {row['day_clustered_t']}")
    if non_finite:
        print(f"[eventdriven-agents] non-finite values written as null at: {non_finite}")
    print(f"written to {REPORT_PATH}")
    return 0


__all__ = [
    "COST_BPS",
    "GATES",
    "HOLD_BARS",
    "Tape",
    "Trigger",
    "chain_defect_evidence",
    "deoverlap",
    "gate_argus",
    "gate_ballast",
    "gate_whale",
    "load_tape",
    "main",
    "nicholas_gate",
    "replay_perception",
    "run",
    "score_trades",
    "slimon_llm_decisions",
    "trade",
    "triggers_from",
    "validate_replay",
    "walk_forward",
]


if __name__ == "__main__":
    raise SystemExit(main())
