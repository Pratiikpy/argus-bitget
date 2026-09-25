"""Omni (Jayanng/Omni @c50d566, MIT) in the XA arena — its own keyless governance cycle.

Run as ``python omni_runner.py`` with the Omni clone as working directory. Each hour it runs the
same sequence as `omni/cli.py:run_cycle` (lines 263-560) on the arena book, calling Omni's own,
unmodified functions:

1. `omni/session.py:classify` on the Reality payloads frozen on the tape, at the decision time.
2. `omni/scenario.py:empirical_tail_shock` for the governing rToken and the largest crypto perp —
   with `omni/bitget_public.candles` answered from the tape (daily rows up to the decision) instead
   of the network. The quantile arithmetic is theirs.
3. `omni/risk.py:evaluate` with the venue discount rate picked by `omni/venue_risk.py:_pick_tier`
   and MMR by `mmr_for_notional`, both on the tables frozen on the tape.
4. `omni/llm.py:decide` with no API key — which returns `fallback_decision`, its own coded path.
5. `omni/policy.py:validate` with its default `PolicyConfig` (hedge cap 5,000 USDT).

Execution follows `omni/executor.py:113-284`: ``HEDGE_STOCK_PERP`` opens an *additional* short of
the requested notional on the mapped perpetual; ``REDUCE_PERP`` and ``CLOSE_PERP`` both call
``close_position`` on the whole symbol — a ``reduce_pct`` of 50 still closes everything
(`executor.py:257-270`). That is how Omni behaves, so that is what the arena does.
"""

from __future__ import annotations

import importlib
import os
import sys
from datetime import UTC, datetime
from typing import Any

sys.path.insert(0, os.getcwd())
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))

from _proto import HOUR_MS, Tape, serve  # type: ignore[import-not-found, unused-ignore]

GOVERNING = "RNVDAUSDT"
CODES = {"RNVDAUSDT": "NVDA", "RAAPLUSDT": "AAPL", "RTSLAUSDT": "TSLA"}



def _load(name: str) -> Any:
    """One of the rival's own modules, from its clone on ``sys.path``."""
    return importlib.import_module(name)

class Omni:
    def __init__(self, init: dict[str, Any], tape: Tape) -> None:
        self.bp: Any = _load("omni.bitget_public")
        self.risk: Any = _load("omni.risk")
        self.llm: Any = _load("omni.llm")
        self.policy: Any = _load("omni.policy")
        self.session: Any = _load("omni.session")
        self.scenario: Any = _load("omni.scenario")
        self.coll: Any = _load("omni.collateral")
        self.cfg: Any = _load("omni.config")
        self.vr: Any = _load("omni.venue_risk")
        self.init = init
        self.tape = tape
        self.stock_perps = {"NVDAUSDT", "AAPLUSDT", "TSLAUSDT", "QQQUSDT", "SPYUSDT",
                            "SQQQUSDT", "METAUSDT", "AMZNUSDT", "MSFTUSDT", "GOOGLUSDT"}
        # The candle read Omni makes (`bitget_public.candles`) answered from the tape.
        self.bp.candles = self._candles
        vr = init["venue_risk"]
        self.discount = {str(e.get("coin", "")).upper(): e.get("list") or []
                         for e in vr.get("discount_rate", [])}
        self.tiers = {sym: [self.vr.TierBand(tier=int(float(b.get("tier") or 0)),
                                             min_usdt=float(b.get("minTierValue") or 0),
                                             max_usdt=float(b.get("maxTierValue") or 0),
                                             mmr=float(b.get("mmr") or 0),
                                             max_leverage=float(b.get("leverage") or 0))
                            for b in (bands or [])]
                      for sym, bands in vr.get("position_tier", {}).items()}
        self.counts: dict[str, int] = {}
        self.needs_attention = 0
        self.overrides = 0
        self.shock_samples: list[tuple[float, float]] = []

    def _candles(self, symbol: str, interval: str = "1H", limit: int = 24,
                 timeout: int = 30) -> list[list[str]]:
        if interval != "1D":
            raise RuntimeError(f"arena serves only the daily candles Omni's scenario reads, "
                               f"not {interval}")
        rows = self.tape.daily_rows(f"spot:{symbol}", limit=limit)
        return [[str(int(r[0])), str(r[1]), str(r[2]), str(r[3]), str(r[4])] for r in rows]

    def _haircut(self, value: float) -> float:
        band = self.vr._pick_tier(self.discount.get(self.vr._discount_coin(GOVERNING).upper(), []),
                                  value)
        if band is None:
            return 1.0
        d = float(band.get("discountRate") or 0.0)
        return 1.0 - d if 0.0 < d <= 1.0 else 1.0

    def step(self, msg: dict[str, Any], tape: Tape) -> dict[str, Any]:
        book = msg["book"]
        now = datetime.fromtimestamp(int(msg["ts"]) / 1000, UTC)
        reality = self.init["reality"]
        stock = (reality.get("stock_info", {}).get(GOVERNING) or [{}])[0]
        sess = self.session.classify(reality.get("states", {}), stock,
                                     reality.get("calendar", {}).get(CODES[GOVERNING], {}),
                                     now=now)
        marks = {s: tape.close(f"spot:{s}") or 0.0 for s in book["spot"]}
        rpos = [self.coll.RTokenPosition(symbol=s, code=CODES.get(s, ""), qty=q, avg_price=0.0)
                for s, q in book["spot"].items() if s in CODES]
        futures = []
        for sym, q in book["perp"].items():
            mark = tape.close(f"perp:{sym}") or 0.0
            mmr = self.vr.mmr_for_notional(self.tiers.get(sym, []), abs(q) * mark) or 0.0
            futures.append(self.risk.FuturesPosition(
                symbol=sym, pos_side="long" if q > 0 else "short", total=abs(q),
                mark_price=mark, avg_price=mark, unrealised_pnl=0.0, mmr=mmr, leverage=1.0,
                asset_class="stock" if sym in self.stock_perps else "crypto"))
        holding = sum(p.qty * marks.get(p.symbol, 0.0) for p in rpos)
        haircut = self._haircut(holding)
        est = self.scenario.empirical_tail_shock(GOVERNING, interval="1D")
        rtoken_shock = est.shock_pct if est is not None else 0.0
        proxy = self.scenario.crypto_proxy_symbol(futures)
        cest = self.scenario.empirical_tail_shock(proxy, interval="1D") if proxy else None
        crypto_shock = cest.shock_pct if cest is not None else 0.0
        self.shock_samples.append((rtoken_shock, crypto_shock))
        rd = self.risk.evaluate(
            as_of=now.isoformat(), observed_effective_equity=float(book["cash"]),
            rtoken_positions=rpos, rtoken_marks=marks, futures_positions=futures,
            haircut_pct=haircut, rtoken_shock_pct=rtoken_shock, crypto_shock_pct=crypto_shock,
            hedge_symbol=self.cfg.stock_perp_for(GOVERNING),
        ).to_dict()
        if rd["results"].get("needs_attention"):
            self.needs_attention += 1
        decision = self.llm.decide(rd, sess.to_dict(), "", "", "")
        pol = self.policy.validate(decision.action, decision.params, rd, sess.to_dict(),
                                   self.policy.PolicyConfig())
        if pol.override:
            self.overrides += 1
        action = pol.action if pol.approved else "HOLD"
        self.counts[action] = self.counts.get(action, 0) + 1
        orders: list[dict[str, Any]] = []
        if action == "HEDGE_STOCK_PERP":
            sym = self.cfg.stock_perp_for(str(pol.params.get("symbol") or GOVERNING))
            notional = float(pol.params.get("notional_usdt") or 0.0)
            if sym and notional > 0:
                orders.append({"kind": "perp", "symbol": sym, "usd": -notional})
        elif action in ("REDUCE_PERP", "CLOSE_PERP"):
            sym = str(pol.params.get("symbol") or "").upper()
            q = book["perp"].get(sym, 0.0)
            mark = tape.close(f"perp:{sym}") or 0.0
            if q and mark:
                orders.append({"kind": "perp", "symbol": sym, "usd": -q * mark})
        note = ""
        if orders:
            note = f"{action} {pol.reason[:200]}"
        return {"orders": orders, "action": action, "note": note}

    def finish(self) -> dict[str, Any]:
        rs = [a for a, _ in self.shock_samples]
        cs = [b for _, b in self.shock_samples]
        return {"actions": self.counts, "needs_attention_hours": self.needs_attention,
                "policy_overrides": self.overrides,
                "rtoken_shock_range": [min(rs), max(rs)] if rs else None,
                "crypto_shock_range": [min(cs), max(cs)] if cs else None,
                "hour_ms": HOUR_MS}


if __name__ == "__main__":
    serve(Omni)
