"""Triad (danielamodu/Triad @d70c67b) in the XA arena — its own keyless tick.

Run as ``python triad_runner.py [scale]`` with the Triad clone as working directory. Each hour it
runs the sequence of `main.py:tick` (lines 674-870) with Triad's own, unmodified functions:

* signals — `src/signals/price_divergence.get_divergence`, `event_signal.get_event`,
  `sentiment_signal.get_sentiment`. Their transport (the ``bgc`` CLI, RSS feeds, the
  bitget-signal skill subprocess) is replaced by the tape: `src/cli.tickers`, `cli.candles` and
  `cli.funding_rate_history` answer from bars up to the decision, and the RSS / skill-text readers
  return nothing, because historical headlines for these hours are not available. Their own code
  then falls through to its coded fallbacks: the 1H volume-and-range expansion event
  (`event_signal.py:184-210`) and the funding z-score plus perp basis (`sentiment_signal.py:
  128-157`).
* decision — `src/decision/engine.decide`; with no ``GROQ_API_KEY`` it returns
  `weighted_decision` (engine.py:246-278), Triad's own fallback.
* risk — `src/risk/cage.validate` with a ledger kept by `src/risk/state.py`'s own functions, and
  the per-leg brackets of `config.STOP_PCT` / `config.TAKE_PCT` forcing EXIT
  (`main.py:758-771`).
* sizing — `src/execution/executor.size_for_confidence` capped at `config.RISK_MAX_POSITION_USD`
  (executor.py:345-352).

Order semantics are Triad's own (`executor.py:1-6`): LONG_RTOKEN buys the widest-gap rToken,
HEDGE_CRYPTO sells BTC, EXIT sells the selected rToken and BTC. ``scale`` multiplies every size and
cap (1 = as configured; 10 = its own replay's $10,000 book scaled to the arena's $100,000).
"""

from __future__ import annotations

import importlib
import os
import sys
from datetime import UTC, datetime
from typing import Any

sys.path.insert(0, os.getcwd())
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))

from _proto import Tape, serve  # type: ignore[import-not-found, unused-ignore]

SCALE = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0



def _load(name: str) -> Any:
    """One of the rival's own modules, from its clone on ``sys.path``."""
    return importlib.import_module(name)

class Triad:
    def __init__(self, init: dict[str, Any], tape: Tape) -> None:
        self.tape = tape
        self.config: Any = _load("config")
        self.cli: Any = _load("src.cli")
        self.div: Any = _load("src.signals.price_divergence")
        self.event: Any = _load("src.signals.event_signal")
        self.sent: Any = _load("src.signals.sentiment_signal")
        self.engine: Any = _load("src.decision.engine")
        self.cage: Any = _load("src.risk.cage")
        self.state: Any = _load("src.risk.state")
        self.executor: Any = _load("src.execution.executor")
        # Transport only: every read answers from the tape, up to the decision.
        self.cli.tickers = self._tickers
        self.cli.candles = self._candles
        self.cli.funding_rate_history = self._funding_history
        self.event._fetch_rss = lambda: ""
        self.event._fetch_raw = lambda: ""
        self.sent._fetch_raw = lambda: ""
        if SCALE != 1.0:
            self.config.RISK_MAX_POSITION_USD = self.config.RISK_MAX_POSITION_USD * SCALE
        self.kill = os.path.exists(self.config.KILL_FILE)
        self.rstate = self.state.fresh_state()
        self.legs: dict[str, dict[str, float]] = {}
        self.realized = 0.0
        self.counts: dict[str, int] = {}
        self.engines: dict[str, int] = {}
        self.blocked = 0

    # --- transport -----------------------------------------------------------------------------

    def _tickers(self, category: str, symbol: str = "") -> list[dict[str, Any]]:
        kind = "spot" if category.upper() == "SPOT" else "perp"
        last = self.tape.close(f"{kind}:{symbol}")
        ch = self.tape.change(f"{kind}:{symbol}", 24)
        if last is None or ch is None:
            return []
        return [{"symbol": symbol, "lastPrice": str(last), "price24hPcnt": str(ch)}]

    def _candles(self, category: str, symbol: str, interval: str, limit: int) -> list[list[str]]:
        kind = "spot" if category.upper() == "SPOT" else "perp"
        if interval != "1H":
            raise self.cli.BgcError(f"arena serves 1H candles, not {interval}")
        return [[str(int(r[0])), *(str(v) for v in r[1:])]
                for r in self.tape.hourly_rows(f"{kind}:{symbol}", limit=limit)]

    def _funding_history(self, category: str, symbol: str, limit: int = 100,
                         cursor: str = "") -> list[dict[str, Any]]:
        rows = self.tape.funding.get(symbol, [])
        newest = sorted(rows, key=lambda r: -r[0])[:limit]
        return [{"symbol": symbol, "fundingRate": str(rate), "fundingRateTimestamp": str(ts)}
                for ts, rate in newest]

    # --- the tick -----------------------------------------------------------------------------

    def _price(self, symbol: str) -> float:
        return self.tape.close(f"spot:{symbol}") or 0.0

    def _bot_pnl(self) -> float:
        unreal = sum(leg["qty"] * self._price(sym) - leg["cost"] for sym, leg in self.legs.items())
        return self.realized + unreal

    def step(self, msg: dict[str, Any], tape: Tape) -> dict[str, Any]:
        cfg = self.config
        price = _safe(self.div.get_divergence, {"signal": "STABLE", "direction": "FLAT",
                                                "divergence_score": 0.0})
        event = _safe(self.event.get_event, {"signal": "NEUTRAL", "confidence": 0.5})
        sentiment = _safe(self.sent.get_sentiment, {"sentiment": "neutral", "score": 0.5})
        selected = str(price.get("selected_rtoken") or cfg.RTOKEN_SYMBOL).upper()

        gate_pnl = self._bot_pnl()
        today = datetime.fromtimestamp(int(msg["ts"]) / 1000, UTC).date().isoformat()
        self.state.roll_day(self.rstate, today, gate_pnl)
        ctx = {"drawdown_pct": self.state.drawdown_pct(self.rstate, gate_pnl),
               "day_loss_pct": self.state.day_loss_pct(self.rstate, gate_pnl),
               "exposure": dict(self.rstate.get("exposure", {})), "corrupt": False,
               "broker_dead": False, "broker_streak": 0}
        ctx["daily_halted"] = ctx["day_loss_pct"] > cfg.RISK_MAX_DAILY_LOSS_PCT
        decision = self.engine.decide({"price": price, "event": event, "sentiment": sentiment},
                                      {}, [])
        self.engines[decision.get("engine_used", "?")] = \
            self.engines.get(decision.get("engine_used", "?"), 0) + 1
        for sym, leg in self.legs.items():
            px = self._price(sym)
            move = px / leg["entry"] - 1 if leg["entry"] > 0 else 0.0
            if move <= -cfg.STOP_PCT or move >= cfg.TAKE_PCT:
                decision = {"decision": "EXIT", "confidence": 1.0,
                            "reasoning": f"bracket on {sym} ({move:+.2%})"}
                break
        name = str(decision.get("decision", "HOLD")).upper()
        trade_symbol = (selected if name == "LONG_RTOKEN" else
                        cfg.CRYPTO_SYMBOL if name == "HEDGE_CRYPTO" else "")
        verdict = self.cage.validate(decision, {}, ctx, trade_symbol)
        if not verdict.get("approved"):
            self.blocked += 1
            name = "HOLD"
        self.counts[name] = self.counts.get(name, 0) + 1
        if name == "HOLD":
            return {"orders": [], "action": "HOLD"}
        size = self.executor.size_for_confidence(decision.get("confidence", 0)) * SCALE
        notional = min(cfg.RISK_MAX_POSITION_USD, size)
        if notional <= 0:
            return {"orders": [], "action": f"{name}_LOW_CONFIDENCE"}
        orders: list[dict[str, Any]] = []
        book = msg["book"]
        if name == "LONG_RTOKEN":
            orders.append({"kind": "spot", "symbol": selected, "usd": notional})
            px = self._price(selected)
            leg = self.legs.setdefault(selected, {"qty": 0.0, "cost": 0.0, "entry": px})
            leg["qty"] += notional / px if px else 0.0
            leg["cost"] += notional
            self.state.record_fills(self.rstate, {"executed": True, "details": {
                "symbol": selected, "side": "buy", "notional_usdt": notional,
                "executed": True}})
        else:
            symbols = [selected, cfg.CRYPTO_SYMBOL] if name == "EXIT" else [cfg.CRYPTO_SYMBOL]
            for sym in symbols:
                if sym == cfg.CRYPTO_SYMBOL:
                    held = book["perp"].get("BTCUSDT", 0.0) * (tape.close("perp:BTCUSDT") or 0)
                    usd = min(notional, max(held, 0.0))
                    if usd > 0:
                        orders.append({"kind": "perp", "symbol": "BTCUSDT", "usd": -usd})
                else:
                    held = book["spot"].get(sym, 0.0) * self._price(sym)
                    usd = min(notional, held)
                    if usd > 0:
                        orders.append({"kind": "spot", "symbol": sym, "usd": -usd})
                        open_leg = self.legs.get(sym)
                        if open_leg:
                            worth = open_leg["qty"] * self._price(sym)
                            frac = min(1.0, usd / max(worth, 1e-9))
                            self.realized += frac * (worth - open_leg["cost"])
                            open_leg["qty"] *= 1 - frac
                            open_leg["cost"] *= 1 - frac
                            if open_leg["qty"] <= 1e-12:
                                self.legs.pop(sym)
                self.state.record_fills(self.rstate, {"executed": True, "details": {
                    "symbol": sym, "side": "sell", "notional_usdt": notional,
                    "executed": True}})
        return {"orders": orders, "action": name,
                "note": str(decision.get("reasoning", ""))[:240]}

    def finish(self) -> dict[str, Any]:
        return {"actions": self.counts, "engine_used": self.engines,
                "cage_blocked": self.blocked, "scale": SCALE, "kill_file_present": self.kill,
                "bot_pnl": self._bot_pnl()}


def _safe(fn: Any, fallback: dict[str, Any]) -> dict[str, Any]:
    """`main.py:505-510`'s own ``safe`` wrapper: a failing signal becomes its neutral fallback."""
    try:
        out = fn()
        return out if isinstance(out, dict) else dict(fallback)
    except Exception:
        return dict(fallback)


if __name__ == "__main__":
    serve(Triad)
