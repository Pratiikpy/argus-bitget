"""Momentum, trend and structure from bitget-signal's technical-analysis Skill, read and cross-
checked."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from argus.lui.answer import LEAD, Source
from argus.lui.research.kinds import _t
from argus.lui.trace import trace_module
from argus.truth.coverage import ContextPool


def _skill_calls(calls: Sequence[tuple[str, dict[str, Any]]],
                 timeout: int = 10) -> list[tuple[Any, str]]:
    """Several `bitget-signal` Skill calls at once, in order. Each gets its own session, because
    the transport is a stateful JSON-RPC session and sharing one across threads would interleave
    requests on it."""
    from argus.market.evidence import BitgetSkillSource

    def one(call: tuple[str, dict[str, Any]]) -> tuple[Any, str]:
        return BitgetSkillSource().call(call[0], call[1], timeout=timeout)

    with ContextPool(max_workers=len(calls)) as pool:
        return list(pool.map(one, calls))


_STATE_ASKED = re.compile(r"\b(overbought|oversold)\b|超买|超卖|과매수|과매도|"
                          r"買われすぎ|売られすぎ|sobrecompra|sobreventa|sobrecomprad|sobrevendid",
                          re.I)


_RSI_LINE = re.compile(r"RSI\(14[^)]*\)\s+([\d.]+)")


def _answer_the_state_asked(question: str, symbol: str, lines: list[str]) -> list[str]:
    """Lead with a yes or no when the question asks whether a name is overbought or oversold.

    "is TSLA overbought" was answered "Bottom line: momentum turning down." followed by "RSI 63.4 —
    neutral" — the answer was there, but a trader had to infer it (a judge-style pass,
    2026-09-25). The verdict is read from the RSI line already computed; nothing new is fetched."""
    asked = _STATE_ASKED.search(question)
    reading = next((_RSI_LINE.search(line) for line in lines if _RSI_LINE.search(line)), None)
    if asked is None or reading is None:
        return lines
    word = (asked.group(1) or "").lower()
    wants_oversold = word == "oversold" or any(t in question for t in (
        "超卖", "과매도", "売られすぎ", "sobreventa", "sobrevendid"))
    rsi = float(reading.group(1))
    if wants_oversold:
        verdict = (f"Yes — {_t(symbol)} is oversold: RSI {rsi:.1f}, under 30." if rsi <= 30 else
                   f"No — {_t(symbol)} is not oversold: RSI {rsi:.1f}, above the 30 line.")
    else:
        verdict = (f"Yes — {_t(symbol)} is overbought: RSI {rsi:.1f}, over 70." if rsi >= 70 else
                   f"No — {_t(symbol)} is not overbought: RSI {rsi:.1f}, below the 70 line.")
    lead = next((i for i, line in enumerate(lines) if LEAD.match(line)), None)
    if lead is None:
        return [f"Bottom line: {verdict}", *lines]
    rest = LEAD.sub("", lines[lead], count=1).strip()
    merged = f"Bottom line: {verdict.rstrip('.')}; {rest}" if rest else f"Bottom line: {verdict}"
    return [*lines[:lead], merged, *lines[lead + 1:]]


def _technicals(symbol: str, *, found: dict[str, Any] | None = None,
                ) -> tuple[list[str], list[Source]]:
    """RSI, MACD, support/resistance and ATR as `bitget-signal` reports them, and what they say
    together. Nothing here is recomputed: the figures are the Skill's, and each is sourced to it.

    ``found``, when given, receives the figures stated — ``rsi``, ``macd_histogram``, ``price``,
    ``support`` and ``resistance`` (the nearest of each, when in range), ``atr`` and ``timeframe``
    — for a caller that reasons with them rather than with the sentences."""
    if found is None:
        found = {}
    calls = [("technical_analysis", {"action": a, "symbol": symbol})
             for a in ("rsi", "macd", "support_resistance", "atr")]
    results = dict(zip(("rsi", "macd", "sr", "atr"), _skill_calls(calls), strict=True))
    lines: list[str] = []
    sources: list[Source] = []
    reading: list[str] = []

    rsi, status = results["rsi"]
    if isinstance(rsi, dict) and rsi.get("rsi") is not None:
        value = float(rsi["rsi"])
        found.update(rsi=value, timeframe=str(rsi.get("timeframe") or ""))
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
            found["macd_histogram"] = float(hist)
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
        found["price"] = price
        above = [float(x) for x in sr.get("resistances") or [] if float(x) > price]
        below = [float(x) for x in sr.get("supports") or [] if float(x) < price]
        parts = []
        if above:
            r = min(above)
            found["resistance"] = r
            parts.append(f"nearest resistance {r:g} ({(r / price - 1) * 100:.1f}% above)")
            if (r / price - 1) * 100 < 1.0:
                reading.append(f"price is within 1% of resistance at {r:g}")
        if below:
            sp = max(below)
            found["support"] = sp
            parts.append(f"nearest support {sp:g} ({(1 - sp / price) * 100:.1f}% below)")
        lines.append(f"Price {price:g}: " + ("; ".join(parts) if parts else
                                              "no support or resistance level within range") + ".")
        sources.append(Source(kind="venue",
                              ref="bitget-signal technical_analysis.support_resistance",
                              detail=status))
    atr, status = results["atr"]
    if isinstance(atr, dict) and atr.get("atr") is not None:
        found["atr"] = float(atr["atr"])
        extra = f"; suggested stop distance {atr['stop_distance']}" if atr.get(
            "stop_distance") is not None else ""
        lines.append(f"ATR {float(atr['atr']):.3f} ({atr.get('timeframe', '')}){extra}.")
        sources.append(Source(kind="venue", ref="bitget-signal technical_analysis.atr",
                              detail=status))
    if lines:
        lines.insert(0, "Bottom line: " + (
            "; ".join(reading) if reading else "no technical extreme — the setup is neutral"
        ) + ".")
        lines.append("Caveat: technicals describe the tape, not an edge — the systematic signals "
                     "this desk tested on these names did not clear costs.")
    return lines, sources


# Every engine here is a traced step from import on (lui/trace.py, trace_module).
trace_module(globals())
