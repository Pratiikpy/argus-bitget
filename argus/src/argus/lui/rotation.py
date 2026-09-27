"""Risk on or risk off, across US stocks, crypto and gold — the console's rotation answer.

**Why this exists.** `desk/rotation.py` implements the breadth-momentum rotation the handbook's
Cross-Asset Allocation sub-theme names (Keller and Keuning's Vigilant Asset Allocation, verified
against pytaa's own `vigilant_allocation` in `eval/rotation_comparison.py`), and until 2026-09-27
nothing a trader could ask reached it: only its evaluation arena did (audit finding 164). This
module is the question that reaches it.

**What history it reads, and why not Bitget's.** The rule scores each asset on twelve months of
monthly closes. Bitget's stock perpetuals and its gold contract are younger than that — gold had
nine months on 2026-09-15 — and the rule rightly refuses a series it cannot score. So the scores
are computed on the instruments the perpetuals track, from Yahoo's daily history (NVDA, AAPL and
QQQ themselves; BTC-USD and ETH-USD; GLD for gold), and every line says so. The trades are then
named as the Bitget contracts a trader would use. Where a series is still too short or stale, the
engine's refusal is the answer, clause by clause; it never proposes a partial rotation.

**What the answer is not.** A momentum rule's regime reading and the book it implies, priced at the
taker fee. It is not a forecast, and a trader's own view of any one name is not in it.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime, time
from typing import Any

from argus.desk.rotation import RotationError, RotationPlan, propose_rotation_from_bars
from argus.lui.answer import Source

RISK: dict[str, str] = {"NVDAUSDT": "NVDA", "AAPLUSDT": "AAPL", "QQQUSDT": "QQQ",
                        "BTCUSDT": "BTC-USD", "ETHUSDT": "ETH-USD"}
"""The risk universe the rule was evaluated on (`eval/rotation_comparison.py`), Bitget contract →
the instrument whose history scores it."""

SAFE: dict[str, str] = {"XAUUSDT": "GLD"}
"""The safe asset: gold, scored on the GLD fund's history."""

_ASKS = re.compile(
    r"\brisk[\s-]?(?:on|off)\b|\brotat(?:e|ion|ing)\b.*\b(?:stocks?|crypto|gold|asset\s+class)"
    r"|\b(?:stocks?|equities)\s*(?:,|or|vs\.?|versus)\s*(?:crypto|bitcoin|gold)\b.*\b(?:gold|crypto|"
    r"which|where|rotate|allocat\w*)\b|\bwhich\s+asset\s+class\b|\bdefensive\s+(?:allocation|"
    r"rotation)\b|\bbreadth\s+momentum\b|\bvigilant\s+asset\s+allocation\b|风险偏好|避险轮动",
    re.I)


def asks_for_rotation(text: str) -> bool:
    """A question about rotating between asset classes, or about risk-on versus risk-off."""
    return bool(_ASKS.search(text))


def _bars(ticker: str, daily: Callable[[str], Sequence[Any]]) -> list[tuple[datetime, float]]:
    return [(datetime.combine(d.day, time(21), tzinfo=UTC), float(d.close)) for d in daily(ticker)]


def _current(book: Mapping[str, float]) -> tuple[dict[str, float], list[str]]:
    """The book's weights on the rule's universe, and the holdings outside it."""
    universe = {**RISK, **SAFE}
    inside = {s: w for s, w in book.items() if s in universe}
    outside = sorted(s.removesuffix("USDT") for s in book if s not in universe)
    return inside, outside


def rotation(text: str, book_text: str = "", *,
             daily: Callable[[str], Sequence[Any]] | None = None,
             parse_book: Callable[[str], Mapping[str, float]] | None = None,
             ) -> tuple[list[str], list[Source], dict[str, Any]]:
    """The rotation answer: the regime, each asset's score, the target book and its cost."""
    if daily is None:
        from argus.market.equity_history import daily as yahoo_daily
        daily = yahoo_daily
    if parse_book is None:
        from argus.lui.research import parse_book as read_book
        parse_book = read_book
    book = dict(parse_book(book_text)) if book_text.strip() else {}
    current, outside = _current(book)
    bars: dict[str, list[tuple[datetime, float]]] = {}
    unread: list[str] = []
    for symbol, ticker in {**RISK, **SAFE}.items():
        try:
            bars[symbol] = _bars(ticker, daily)
        except Exception as exc:  # one series failing is named, and the rule then refuses
            unread.append(f"{ticker} ({type(exc).__name__})")
    sources = [Source(kind="venue", ref="Yahoo Finance daily history",
                      detail=", ".join(sorted({*RISK.values(), *SAFE.values()}))),
               Source(kind="computation", ref="argus.desk.rotation.propose_rotation_from_bars",
                      detail="breadth momentum, Keller & Keuning 2017")]
    try:
        plan = propose_rotation_from_bars(current, bars, tuple(RISK), tuple(SAFE))
    except RotationError as exc:
        lines = ["Bottom line: the rotation rule will not give a reading today — it needs twelve "
                 "months of closes for every asset and refuses a partial answer rather than "
                 "guess one."]
        lines += [f"Refused: {v.symbol or 'universe'} — {v.detail}"
                  for v in (exc.violations or ())][:6]
        if unread:
            lines.append("Not read: " + ", ".join(unread) + ".")
        return lines, sources, {"refused": True}
    return _lines(plan, current, outside, bool(book)), sources, {"rotation": plan.as_dict()}


def _lines(plan: RotationPlan, current: Mapping[str, float], outside: Sequence[str],
           has_book: bool) -> list[str]:
    named = {**RISK, **SAFE}
    order = sorted(plan.scores, key=lambda s: -plan.scores[s])
    target = {t.symbol: t.weight_after for t in plan.trades}
    held_target = {s: target.get(s, current.get(s, 0.0)) for s in named}
    lead = (f"Bottom line: {plan.regime}. The rule's book: "
            + ", ".join(f"{s.removesuffix('USDT')} {w:.0%}" for s, w in
                        sorted(held_target.items(), key=lambda kv: -kv[1]) if w > 0)
            + ".")
    lines = [lead,
             "Momentum scores (weighted 1, 3, 6 and 12-month returns; below zero counts against "
             "risk): " + ", ".join(f"{s.removesuffix('USDT')} {plan.scores[s]:+.3f}" for s in order)
             + "."]
    if has_book:
        if plan.trades:
            moves = "; ".join(f"{t.symbol.removesuffix('USDT')} {t.weight_before:.0%} → "
                              f"{t.weight_after:.0%}" for t in plan.trades)
            lines.append(f"From your book: {moves} — turnover {plan.turnover:.0%}, about "
                         f"{plan.cost_bps:.1f} bps of the book at the taker fee.")
        else:
            lines.append("Your book already sits at the rule's allocation; nothing to trade.")
        if outside:
            lines.append("Held outside the rule's universe, and left as they are: "
                         + ", ".join(outside) + ".")
    else:
        lines.append("No book is saved, so this is the rule's allocation from cash; save your "
                     "holdings to see the trades from where you are.")
    lines.append("Scored on the instruments the Bitget contracts track (" + ", ".join(
        f"{s.removesuffix('USDT')} on {t}" for s, t in named.items())
        + "), because the perpetuals have less than the twelve months the rule needs. A regime "
        "reading and an allocation rule, not a forecast.")
    return lines


def answer(text: str, book_text: str = "") -> tuple[list[str], list[Source], dict[str, Any]]:
    """The console's entry point."""
    return rotation(text, book_text)
