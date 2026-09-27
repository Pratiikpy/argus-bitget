"""Should a book that holds rToken spot and crypto hedge now — through which perpetual, how much?

The console's door to `desk/crossasset.py` (audit finding 164). The router answers exactly the
handbook's Cross-Asset Execution question for a book holding both: it hedges only when the risk
forecast for the hours to the next session boundary, or a perpetual's funding, departs from normal
by more than the fees, slippage and funding the hedge would cost — and otherwise says to hold. Its
inputs are the sixty-day hourly snapshot `market/crossasset_feed.py` keeps; the answer states how
old that is.

**Reading the book.** Spot rTokens are read from the question or the saved book by name and weight
("40% RNVDAUSDT") or dollars ("$30k RTSLA"); crypto and stock perpetuals through the console's own
book reader. With dollar amounts the book's value is their sum; with weights only, the answer
prices a $100,000 book and says so, because the router's fees and slippage depend on size.
"""

from __future__ import annotations

import re
import statistics
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from argus.desk.crossasset import (
    CostCurve,
    RouterConfig,
    RouterInput,
    estimate,
    phase_calendar,
    route,
    settlements_between,
)
from argus.lui.answer import Source
from argus.market.crossasset_feed import HOUR_MS, PERPS, SPOT, FeedError, Snapshot, load

DEFAULT_NAV = 100_000.0
STALE_HOURS = 6.0
"""An older snapshot is still used, and the answer leads with its age."""

_RTOKEN = r"R(NVDA|AAPL|TSLA|QQQ|SPY)(?:USDT)?"
_PCT_FIRST = re.compile(rf"(\d+(?:\.\d+)?)\s*%\s*(?:of\s+)?{_RTOKEN}\b", re.I)
_NAME_FIRST = re.compile(rf"\b{_RTOKEN}(?:\s+spot)?\s*(?:at\s+)?(\d+(?:\.\d+)?)\s*%", re.I)
_DOLLARS = re.compile(rf"\$\s*(\d+(?:\.\d+)?)\s*([km])?\s*(?:of\s+)?{_RTOKEN}\b", re.I)
_ANY_RTOKEN = re.compile(rf"\b{_RTOKEN}\b", re.I)
_HEDGE = re.compile(r"\bhedg\w*|\bprotect\w*|\bcover\b|\boffset\w*|\breduce\s+(?:my\s+)?risk\b|"
                    r"\bweekend\s+risk\b|对冲|保护", re.I)
_CRYPTO = re.compile(r"\b(?:BTC|ETH|bitcoin|ether(?:eum)?)\b", re.I)


def asks_for_cross_asset(text: str, book_text: str = "") -> bool:
    """A hedge question from a book that holds rToken spot and crypto at once."""
    both = f"{text} {book_text}"
    return bool(_HEDGE.search(text) and _ANY_RTOKEN.search(both) and _CRYPTO.search(both))


def _scale(unit: str | None) -> float:
    return {"k": 1e3, "m": 1e6}.get((unit or "").lower(), 1.0)


def read_book(text: str, holdings: Callable[[str], list[tuple[int, str, float]]]
              ) -> tuple[dict[str, float], dict[str, float], float, bool]:
    """``(spot weights, perp weights, NAV, NAV was stated)`` from the words."""
    dollars = {f"R{m.group(3).upper()}USDT": float(m.group(1)) * _scale(m.group(2))
               for m in _DOLLARS.finditer(text)}
    spot: dict[str, float] = {}
    for m in _PCT_FIRST.finditer(text):
        spot[f"R{m.group(2).upper()}USDT"] = float(m.group(1)) / 100
    for m in _NAME_FIRST.finditer(text):
        spot.setdefault(f"R{m.group(1).upper()}USDT", float(m.group(2)) / 100)
    perps: dict[str, float] = {}
    for _, symbol, weight in holdings(_ANY_RTOKEN.sub(" ", text)):
        if symbol in PERPS:
            perps[symbol] = perps.get(symbol, 0.0) + weight
    perp_dollars = {f"{m.group(3).upper()}USDT": float(m.group(1)) * _scale(m.group(2))
                    for m in re.finditer(r"\$\s*(\d+(?:\.\d+)?)\s*([km])?\s*(?:of\s+)?"
                                         r"(BTC|ETH)\b", text, re.I)}
    if dollars or perp_dollars:
        nav = sum(dollars.values()) + sum(perp_dollars.values())
        return ({s: v / nav for s, v in dollars.items()},
                {s: v / nav for s, v in perp_dollars.items()}, nav, True)
    total = sum(spot.values()) + sum(perps.values())
    if total > 1.0 + 1e-9:
        spot = {s: w / total for s, w in spot.items()}
        perps = {s: w / total for s, w in perps.items()}
    return spot, perps, DEFAULT_NAV, False


def _funding(snapshot: Snapshot, symbol: str, now_ms: int, lookback_hours: int
             ) -> tuple[float, float]:
    rows = snapshot.funding.get(symbol, [])
    past = [r for t, r in rows if t <= now_ms]
    if not past:
        return 0.0, 0.0
    window = [r for t, r in rows if now_ms - lookback_hours * HOUR_MS <= t <= now_ms]
    return statistics.fmean(past[-3:]), (statistics.fmean(window) if window else 0.0)


def answer(text: str, book_text: str = "", *, snapshot: Snapshot | None = None,
           holdings: Callable[[str], list[tuple[int, str, float]]] | None = None,
           now: datetime | None = None) -> tuple[list[str], list[Source], dict[str, Any]]:
    """The router's decision for the trader's book, in words, with its inputs named."""
    if holdings is None:
        from argus.lui.research.parse import holding_pairs
        holdings = holding_pairs
    at = now or datetime.now(UTC)
    try:
        snap = snapshot or load()
    except FeedError as exc:
        return ([f"Bottom line: the cross-asset router's inputs are not on this server ({exc}); "
                 "ask again once the data job has run."], [], {"refused": True})
    spot, perps, nav, stated = read_book(f"{text} {book_text}", holdings)
    if not spot:
        return (["Bottom line: name the rToken spot you hold (\"40% RNVDAUSDT, 60% BTC\", or "
                 "\"$30k RTSLA, $20k ETH\") and the router will say whether to hedge it."],
                [], {"refused": True})
    cfg = RouterConfig()
    i = len(snap.hours) - 1
    future = [snap.hours[-1] + (j + 1) * HOUR_MS for j in range(cfg.horizon_cap_hours + 1)]
    phases = phase_calendar([*snap.hours, *future])
    keys = tuple(k for k in (*(f"spot:{s}" for s in SPOT if s in spot),
                             *(f"perp:{s}" for s in PERPS))
                 if k in snap.close and sum(snap.real[k][-cfg.lookback_hours:]) > 24 * 20)
    model = estimate(snap.close, keys, phases, i, cfg, real=snap.real)
    now_ms = snap.hours[-1]
    f_now: dict[str, float] = {}
    f_norm: dict[str, float] = {}
    for symbol in PERPS:
        f_now[f"perp:{symbol}"], f_norm[f"perp:{symbol}"] = _funding(
            snap, symbol, now_ms, cfg.lookback_hours)
    perp_usd = {f"perp:{s}": perps.get(s, 0.0) * nav for s in PERPS if f"perp:{s}" in keys}
    costs = {f"perp:{s}": CostCurve(snap.fees_bps.get(s, 6.0), tuple(snap.slippage.get(s, ())))
             for s in PERPS if f"perp:{s}" in keys}
    inp = RouterInput(
        nav=nav, spot_usd={f"spot:{s}": w * nav for s, w in spot.items()}, perp_usd=perp_usd,
        base_perp_usd={k: v for k, v in perp_usd.items() if v}, funding_now=f_now,
        funding_normal=f_norm, settlements_ahead=settlements_between(now_ms, model.horizon_hours),
        costs=costs)
    decision = route(model, inp, cfg, tradeable=list(costs))
    age = snap.age_hours(at)
    lines = [f"Bottom line: {decision.explain()}"]
    if not decision.refused:
        lines.append(f"Risk cost over the horizon: ${decision.risk_before_usd:,.2f} held as it is"
                     f", ${decision.risk_after_usd:,.2f} after the router's change "
                     f"(certainty equivalent at risk aversion {cfg.gamma:g}, on a ${nav:,.0f} "
                     f"book).")
        unstable: dict[str, list[str]] = {}
        for a, b in decision.gated_pairs:
            held, other = (a, b) if a.startswith("spot:") else (b, a)
            if held.startswith("spot:") and other.startswith("perp:"):
                unstable.setdefault(held.split(":")[1], []).append(
                    other.split(":")[1].removesuffix("USDT"))
        for held, others in unstable.items():
            lines.append(f"Not usable to hedge {held}: " + ", ".join(others) + " — the "
                         "correlation changed sign or fell below 0.2 in one half of the "
                         "lookback.")
    book_line = ", ".join([*(f"{s} spot {w:.0%}" for s, w in spot.items()),
                           *(f"{s} {w:.0%}" for s, w in perps.items())])
    lines.append(f"Read as: {book_line}" + ("" if stated else
                 f"; priced as a ${DEFAULT_NAV:,.0f} book, since no dollar amounts were given")
                 + ".")
    lines.append(f"Inputs: sixty days of hourly Bitget closes, funding settlements and live "
                 f"order-book slippage, read {age:.1f} hours ago"
                 + (" — older than usual, so treat the session read with care" if
                    age > STALE_HOURS else "") + ". A hedge overlay the router proposes; no "
                 "order is sent.")
    sources = [Source(kind="computation", ref="argus.desk.crossasset.route",
                      detail="certainty-equivalent hedge router"),
               Source(kind="artefact", ref="data/crossasset_snapshot.json",
                      detail=f"written {snap.written_at:%Y-%m-%d %H:%M} UTC")]
    return lines, sources, {"crossasset": decision.as_dict(), "snapshot_age_hours": round(age, 2)}
