"""Bitget's official research Skills as a measured perception layer.

Track 3 is judged on "Feature depth (**data sources / Skill integration count and effectiveness**)"
(`BITGET_AI_BASE_CAMP_S2_HANDBOOK_EN.md:281`), and the handbook says plainly that an AI Trading Desk
"can combine `bitget-signal`'s research Skills as its perception layer" (`:425`). Five Skills ship
keyless — `macro-analyst`, `market-intel`, `news-briefing`, `sentiment-analyst`,
`technical-analysis` — served by one MCP endpoint exposing nineteen tools.

Before this module ARGUS called **one action of one tool**.

**Count without effectiveness would be a lie, so this module measures rather than claims.** A live
sweep on 2026-09-13 found most upstream fetchers behind these tools timing out or returning empty
from this network, while `technical_analysis` answers reliably — including on the exact rToken
symbols ARGUS trades. Wiring nineteen tools in and letting the dead ones return nothing would look
like nineteen integrations and behave like one. So every tool is described here, every call is
health-classified into a state that cannot be confused with "the market is quiet", and the report
states how many actually answered.

**The Skill's numbers are checked, not trusted.** :func:`cross_check_rsi` recomputes RSI(14) from
Bitget's own candles through `argus.market.history` and reports the gap. Measured on 2026-09-13
across four rTokens: NVDA 30.83 vs 28.47, TSLA 57.43 vs 58.27, MSFT 54.17 vs 53.81, COIN 46.68 vs
45.92 — agreement within 2.4 points, so the Skill is returning genuine per-symbol readings and not a
constant. That check is the difference between integrating a data source and believing one.

**Two client defects were found getting here**, both in `argus.market.evidence.BitgetSkillSource`
and both fixed there: every call was sent with the same JSON-RPC id so replies could not be matched
to requests (a probe had `sentiment_index` return `defi_analytics`'s payload), and the `isError`
flag Bitget's own server sets on a failed call (`agent-mcp/src/server.ts:119-124`) was ignored, so
`Error executing tool cross_asset` would have been parsed as data.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from argus.truth.evidence import Evidence

RSI_AGREEMENT_POINTS = 5.0
"""How far the Skill's RSI may sit from our own before the cross-check calls it a disagreement.

Set from the measured spread (0.36-2.36 across four rTokens on 2026-09-13) with room for a different
upstream exchange and bar alignment. Tighter would flag ordinary provenance differences; looser
would accept a genuinely wrong reading."""

DEFAULT_TIMEOUT = 45
"""Seconds per tool call during a sweep.

Measured on 2026-09-13: ``technical_analysis`` answers in under two seconds, and the tools of the
other four Skills answer in 15-31 seconds — ``sentiment_index`` 31.0, ``crypto_market`` 30.3,
``news_feed`` 20.7, ``macro_indicators`` 20.3, ``rates_yields`` 20.3, ``global_assets`` 15.3. The
earlier 12- and 20-second defaults classified all of those as TIMEOUT, which read as "the upstream
is dead" when the truth was "the upstream is slow and then hollow". Forty-five covers the slowest
measured reply with margin; the sweep is a separate command precisely so this wait is affordable."""


class Health(StrEnum):
    """What a tool did when it was called. Five states, none of them ambiguous.

    The distinction that matters is between :attr:`EMPTY` and every kind of failure. A tool that is
    reachable and returned nothing may genuinely mean "no news"; a tool that timed out means we do
    not know. Collapsing the two is how a desk ends up treating silence as information.
    """

    OK = "ok"
    EMPTY = "empty"
    """Reachable, answered, carried no data. Says nothing about the market."""

    TOOL_ERROR = "tool_error"
    """The tool itself reported failure through ``isError``."""

    TIMEOUT = "timeout"
    UNAVAILABLE = "unavailable"
    """Network or transport failure, including a reply that could not be correlated."""

    @property
    def answered(self) -> bool:
        return self is Health.OK


class SkillClient(Protocol):
    """The one method this module needs. Keeps the MCP transport swappable in tests."""

    def call(
        self, tool: str, args: dict[str, Any], *, timeout: int = ...
    ) -> tuple[Any, str]: ...


@dataclass(frozen=True, slots=True)
class Probe:
    """One tool call this desk knows how to make, and what it is for."""

    skill: str
    """Which of the five official Skills this call belongs to."""

    tool: str
    action: str
    args: dict[str, Any]
    yields: str
    """Plain-language description of the evidence it produces when it answers."""

    symbol_key: str | None = None
    """The argument name that takes an instrument, when the call is per-symbol."""

    def for_symbol(self, symbol: str) -> dict[str, Any]:
        args = {"action": self.action, **self.args}
        if self.symbol_key:
            args[self.symbol_key] = symbol
        return args

    @property
    def ident(self) -> str:
        return f"{self.tool}.{self.action}"


# The five official Skills and what each is for. Taken from `bitget-signal/skills/*/SKILL.md`
# rather than from the handbook's prose summary, because the SKILL.md is what the Skill runs.
SKILLS: dict[str, str] = {
    "technical-analysis": "Technical analysis: 23 indicators across 6 categories",
    "sentiment-analyst": "Sentiment and positioning: Fear & Greed, long/short, funding, OI",
    "macro-analyst": "Macro and cross-asset: Fed policy, yields, BTC vs DXY / Nasdaq / Gold",
    "news-briefing": "News aggregation and narrative synthesis across 44 feeds",
    "market-intel": "On-chain and institutional intelligence: ETF flows, DeFi TVL, DEX activity",
}

# Ordered so the per-symbol technical calls — the ones measured to answer — come first. A sweep cut
# short still produces the part of the report that carries information.
PROBES: tuple[Probe, ...] = (
    Probe("technical-analysis", "technical_analysis", "rsi", {},
          "momentum: RSI(14) with an overbought/oversold reading", symbol_key="symbol"),
    Probe("technical-analysis", "technical_analysis", "macd", {},
          "trend: MACD line, signal line and the cross", symbol_key="symbol"),
    Probe("technical-analysis", "technical_analysis", "atr", {},
          "volatility: ATR and a suggested stop distance", symbol_key="symbol"),
    Probe("technical-analysis", "technical_analysis", "bollinger", {},
          "volatility envelope: Bollinger bands", symbol_key="symbol"),
    Probe("technical-analysis", "technical_analysis", "support_resistance", {},
          "structure: nearby support and resistance levels", symbol_key="symbol"),
    Probe("technical-analysis", "technical_analysis", "ma", {},
          "trend: moving averages", symbol_key="symbol"),
    Probe("sentiment-analyst", "sentiment_index", "current", {},
          "crowd mood: the Fear & Greed index"),
    Probe("sentiment-analyst", "derivatives_sentiment", "long_short",
          {"symbol": "BTCUSDT", "period": "4h"}, "positioning: the long/short account ratio"),
    Probe("sentiment-analyst", "derivatives_sentiment", "open_interest",
          {"symbol": "BTCUSDT", "period": "1h"}, "positioning: open interest"),
    Probe("sentiment-analyst", "derivatives_sentiment", "taker_ratio",
          {"symbol": "BTCUSDT", "period": "4h"}, "pressure: the taker buy/sell ratio"),
    Probe("macro-analyst", "macro_indicators", "latest_release", {},
          "macro: the most recent scheduled economic release"),
    Probe("macro-analyst", "rates_yields", "yield_curve", {},
          "macro: the US Treasury yield curve"),
    Probe("macro-analyst", "cross_asset", "correlation", {},
          "macro: rolling correlation of BTC against equities, gold and the dollar"),
    Probe("macro-analyst", "global_assets", "price", {},
          "the anchor equity's own price, which is what an rToken tracks", symbol_key="symbol"),
    Probe("news-briefing", "news_feed", "latest", {},
          "narrative: aggregated headlines across 44 feeds"),
    Probe("news-briefing", "tradfi_news", "earnings", {},
          "the earnings calendar, which dates the next scheduled catalyst"),
    Probe("market-intel", "defi_analytics", "tvl_rank", {},
          "capital: DeFi TVL rankings"),
    Probe("market-intel", "network_status", "eth_gas", {},
          "chain load: Ethereum gas"),
    Probe("market-intel", "crypto_market", "trending", {},
          "attention: trending coins"),
)


def _classify(payload: Any, status: str) -> tuple[Health, str]:
    """Turn one call's outcome into a state that cannot be mistaken for market information."""
    low = status.lower()
    if "tool reported an error" in low:
        return Health.TOOL_ERROR, status
    if "correlation failed" in low:
        return Health.UNAVAILABLE, status
    if "timeouterror" in low.replace(" ", ""):
        return Health.TIMEOUT, status
    if "unavailable" in low:
        return Health.UNAVAILABLE, status
    # An upstream that answers with its own error envelope is EMPTY, not healthy: the tool worked
    # and the data did not arrive. Both shapes seen live are this — {"error": ""} carrying only a
    # source url, and {"alt_me_error": ""}.
    if isinstance(payload, dict) and payload and all(
        key.endswith("error") or key == "url" for key in payload
    ):
        return Health.EMPTY, f"{status}; upstream returned only an error envelope"
    if "returned no data" in low or payload in (None, {}, [], ""):
        return Health.EMPTY, status
    if hollow(payload):
        return Health.EMPTY, f"{status}; answered with a hollow payload (every leg error or zero)"
    return Health.OK, status


_HOLLOW_KEYS = frozenset({"url", "note", "source", "feed", "symbol", "timeframe", "period",
                          "platform", "provider"})
"""Keys that describe the request, or which backend was tried, rather than answer it. A payload
made only of these carries nothing about the market. ``platform`` and ``provider`` added
2026-09-26: ``social_trending`` answered ``{"platform": "xueqiu", "provider": "all_failed",
"items": []}`` on every platform and was counted reliable, three attempts of three."""


def hollow(payload: Any) -> bool:
    """True when a payload has the *shape* of an answer and none of the substance.

    Seen live on 2026-09-13 with a 60-second timeout: ``rates_yields(yield_curve)`` returned every
    tenor as ``{"error": ""}`` and then ``spread_10y2y: 0.0, inverted: false`` — numbers derived
    from legs that never arrived. ``news_feed`` returned seven feeds, each ``{"error": "",
    "items": []}``. The flat error-envelope check above catches neither, and both were classified
    OK, so an all-zero yield curve and an empty news day were one step from becoming evidence.

    The rule: containers are hollow when every child container is hollow and every scalar is a
    zero, a false, an empty string, a None, or sits under a key that only names the request.
    A single real number under a real key makes the payload substantive, so a genuinely flat
    spread reported next to real tenors still counts as data.
    """
    if payload is None or payload == "" or payload == [] or payload == {}:
        return True
    if isinstance(payload, dict):
        if not payload:
            return True
        for key, value in payload.items():
            if key.endswith("error") or key in _HOLLOW_KEYS:
                continue
            if isinstance(value, dict | list):
                if not hollow(value):
                    return False
            elif value not in (0, 0.0, False, "", None):
                return False
        return True
    if isinstance(payload, list):
        return all(hollow(item) for item in payload)
    return payload in (0, 0.0, False)


@dataclass(frozen=True, slots=True)
class ToolHealth:
    """One probe's outcome, with enough detail to be argued with."""

    probe: Probe
    health: Health
    detail: str
    payload: Any
    checked_at: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "skill": self.probe.skill,
            "tool": self.probe.tool,
            "action": self.probe.action,
            "yields": self.probe.yields,
            "health": str(self.health),
            "detail": self.detail,
            "checked_at": self.checked_at,
            "sample": (
                None if self.payload is None
                else json.dumps(self.payload, default=str)[:240]
            ),
        }


@dataclass(frozen=True, slots=True)
class SkillReport:
    """What the official Skills actually gave us, per Skill and per tool."""

    symbol: str
    checked_at: str
    results: tuple[ToolHealth, ...]

    @property
    def answered(self) -> tuple[ToolHealth, ...]:
        return tuple(r for r in self.results if r.health.answered)

    @property
    def skills_reached(self) -> tuple[str, ...]:
        """Official Skills for which at least one tool answered. The honest integration count."""
        return tuple(sorted({r.probe.skill for r in self.answered}))

    @property
    def skills_reachable(self) -> tuple[str, ...]:
        """Official Skills whose server responded at all, data or not.

        Kept separate from :attr:`skills_reached` because the two tell different stories: a Skill
        that is reachable and hollow is Bitget's upstream problem; a Skill that is unreachable
        could be ours. Only ``reached`` may be quoted as integration."""
        live = {Health.OK, Health.EMPTY, Health.TOOL_ERROR}
        return tuple(sorted({r.probe.skill for r in self.results if r.health in live}))

    def by_health(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in self.results:
            counts[str(row.health)] = counts.get(str(row.health), 0) + 1
        return counts

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "checked_at": self.checked_at,
            "skills_available": sorted(SKILLS),
            "skills_reached": list(self.skills_reached),
            "skills_reachable": list(self.skills_reachable),
            "tools_probed": len(self.results),
            "tools_answered": len(self.answered),
            "by_health": self.by_health(),
            "results": [r.as_dict() for r in self.results],
        }

    def render(self) -> list[str]:
        lines = [
            f"[skills] {len(self.answered)} of {len(self.results)} official-Skill calls answered "
            f"for {self.symbol}; {len(self.skills_reached)} of {len(SKILLS)} Skills reached "
            f"({', '.join(self.skills_reached) or 'none'}); "
            f"{len(self.skills_reachable)} of {len(SKILLS)} reachable"
        ]
        for state, count in sorted(self.by_health().items()):
            if state != str(Health.OK):
                lines.append(
                    f"[skills] {count} call(s) {state} — recorded as absent data, never as a "
                    f"quiet market"
                )
        return lines

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2), encoding="utf-8")
        return path


def load(path: Path) -> SkillReport | None:
    """Read a saved health report. A missing or unreadable file is "unknown", not "healthy".

    Returning ``None`` rather than an empty report matters: an empty report would say every tool
    failed, and the caller would stop calling tools that have never been tried.
    """
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    by_ident = {p.ident: p for p in PROBES}
    rows: list[ToolHealth] = []
    for row in payload.get("results", []):
        item = by_ident.get(f"{row.get('tool')}.{row.get('action')}")
        if item is None:
            continue  # a probe this build no longer knows; skipped rather than guessed at
        try:
            health = Health(row.get("health", "unavailable"))
        except ValueError:
            continue
        rows.append(ToolHealth(
            probe=item, health=health, detail=str(row.get("detail", "")),
            payload=None, checked_at=str(row.get("checked_at", "")),
        ))
    return SkillReport(
        symbol=str(payload.get("symbol", "")),
        checked_at=str(payload.get("checked_at", "")),
        results=tuple(rows),
    )


def usable_probes(
    report: SkillReport | None, *, probes: Sequence[Probe] = PROBES
) -> tuple[Probe, ...]:
    """The calls worth making on a live cycle.

    A full sweep takes minutes, because a dead upstream costs the whole timeout before it admits
    anything, and a decision cycle cannot pay that on every symbol. So the cycle calls only the
    tools the last sweep measured as answering.

    **With no report at all, every probe is returned.** Absence of a measurement is not a
    measurement of absence: a build that has never swept should try everything once rather than
    silently call nothing forever.
    """
    if report is None:
        return tuple(probes)
    healthy = {row.probe.ident for row in report.answered}
    return tuple(p for p in probes if p.ident in healthy)


def probe(
    client: SkillClient,
    *,
    symbol: str,
    at: datetime,
    probes: Sequence[Probe] = PROBES,
    timeout: int = DEFAULT_TIMEOUT,
) -> SkillReport:
    """Call every known Skill tool once and classify what came back."""
    stamp = at.isoformat()
    results: list[ToolHealth] = []
    for item in probes:
        payload, status = client.call(item.tool, item.for_symbol(symbol), timeout=timeout)
        health, detail = _classify(payload, status)
        results.append(ToolHealth(
            probe=item, health=health, detail=detail,
            payload=payload if health.answered else None, checked_at=stamp,
        ))
    return SkillReport(symbol=symbol, checked_at=stamp, results=tuple(results))


def evidence(report: SkillReport, *, as_of: datetime) -> list[Evidence]:
    """Turn the calls that answered into evidence the desk can read.

    Only :attr:`Health.OK` becomes evidence. A timeout produces nothing, and the fact that it
    produced nothing lives in the report rather than in the evidence list — a desk handed "no data"
    as a piece of evidence will reason about it as though it were a finding.

    ``available_at`` is the moment of the call. These are snapshots carrying no earlier stamp, so
    dating them earlier would invent a point-in-time guarantee the source does not give.
    """
    out: list[Evidence] = []
    for row in report.answered:
        payload = row.payload
        if (row.probe.tool, row.probe.action) == ("technical_analysis", "macd") and isinstance(
                payload, dict):
            payload = correct_macd(payload)
        out.append(Evidence(
            id=f"skill-{row.probe.tool}-{row.probe.action}-{int(as_of.timestamp())}",
            claim=(
                f"[{row.probe.skill}] {row.probe.yields}: "
                f"{json.dumps(payload, default=str)[:300]}"
            ),
            source="macro" if row.probe.skill == "macro-analyst" else "social",
            available_at=as_of,
            # A vendor-computed indicator is a real reading of a real series, but it is one
            # provider's arithmetic on one provider's bars — below a filing, above a headline.
            credibility=0.75,
            attributes=payload if isinstance(payload, dict) else {},
        ))
    return out


def macd_fields(payload: Mapping[str, Any],
                 recomputed_signal: float | None) -> tuple[float, float, bool] | None:
    """(signal line, histogram, swapped) from a bitget-signal MACD reading, checked against a
    recomputation from Bitget's own candles; None when it cannot be checked.

    The Skill documents MACD(12,26,9) with DIF, DEA and HIST (`bitget-signal/skills/
    technical-analysis/references/indicators.md:45-53`). Measured live on 2026-09-23 its ``signal``
    field carries the histogram and its ``histogram`` field the signal line: BTCUSDT came back
    ``macd 1568.0, signal -148.3, histogram 1716.3`` while Bitget's 4h candles give DIF 1568, DEA
    1715 and a DIF path of 1567 to 1924 over the last twelve bars — a signal line of -148 is
    impossible there. NVDA, TSLA and META showed the same exchange. Arithmetic cannot catch it,
    because DIF - DEA = HIST and DIF - HIST = DEA are both true either way round, so the check is
    the recomputation: whichever field sits nearer the recomputed signal line is the signal line.
    The Skill's ``cross`` flag is built from the swapped pair and is replaced whenever the swap is
    seen. The day the Skill is fixed this reads it straight with no change here.
    """
    if recomputed_signal is None:
        return None
    signal = float(payload.get("signal") or 0.0)
    histogram = float(payload.get("histogram") or 0.0)
    if abs(histogram - recomputed_signal) < abs(signal - recomputed_signal):
        return histogram, signal, True
    return signal, histogram, False


MIN_INDICATOR_BARS = 36
"""MACD(12,26,9) needs 26 bars for its slow average and nine more for the signal line. A contract
listed a week ago (CVXSTOCKUSDT: 49 four-hour bars on 2026-09-23) still gets a reading, with its
short history stated; below this there is nothing honest to compute."""

SHORT_HISTORY_BARS = 120
"""Under twenty days of 4h bars the answer says the readings rest on a short history."""


def indicators(symbol: str) -> dict[str, float | str] | None:
    """RSI(14), MACD(12,26,9) and ATR(14) on Bitget's 4h candles — the settings the Skill reports
    (`timeframe: 4h`, `period: 14`), so the two describe the same thing and can be compared.

    RSI is :func:`rsi`, Wilder's definition, already cross-checked against the Skill
    to within 2.4 points on four rTokens (see that module). MACD seeds each EMA with the simple mean
    of its first window, as TA-Lib does; ATR uses Wilder's smoothing, as he defined it.
    """
    from argus.market.history import CandleType, fetch

    try:
        bars = fetch(symbol, interval="4H", candle_type=CandleType.MARKET, recent=True, limit=300)
    except Exception:
        return None
    if len(bars) < MIN_INDICATOR_BARS:
        return None
    closes = [float(b.close) for b in bars]

    def ema(values: list[float], n: int) -> list[float]:
        alpha = 2.0 / (n + 1)
        out = [sum(values[:n]) / n]
        for v in values[n:]:
            out.append(out[-1] + alpha * (v - out[-1]))
        return out

    fast, slow = ema(closes, 12), ema(closes, 26)
    dif = [f - sl for f, sl in zip(fast[26 - 12:], slow, strict=False)]
    dea = ema(dif, 9)
    histogram = dif[-1] - dea[-1]
    previous = dif[-2] - dea[-2]
    ranges = [max(float(b.high) - float(b.low), abs(float(b.high) - float(a.close)),
                  abs(float(b.low) - float(a.close))) for a, b in itertools.pairwise(bars)]
    atr = sum(ranges[:14]) / 14
    for r in ranges[14:]:
        atr = (atr * 13 + r) / 14
    value = rsi(closes, 14)
    out: dict[str, float | str] = {
        "dif": dif[-1], "dea": dea[-1], "histogram": histogram, "atr": atr,
        "close": closes[-1], "bars": float(len(bars)), "since": bars[0].ts.isoformat(),
        "cross": ("golden cross" if previous <= 0 < histogram else
                  "death cross" if previous >= 0 > histogram else ""),
    }
    if value is not None:
        out["rsi"] = value
    return out


def correct_macd(payload: dict[str, Any]) -> dict[str, Any]:
    """A `technical_analysis.macd` payload with its fields checked before anything reads it.

    The desk hands Skill readings to the model as evidence, and on 2026-09-23 decision 586's thesis
    cited "golden cross MACD" on NVDA — the Skill's cross, built from its swapped fields (see
    :func:`macd_fields`). Every MACD payload is recomputed from Bitget's 4h candles here: the
    fields are put the right way round and the cross replaced, with ``corrected`` saying so; if the
    candles cannot be read, the unverifiable signal line, histogram and cross are removed rather
    than passed on.
    """
    symbol = str(payload.get("symbol") or "")
    if not symbol or payload.get("macd") is None:
        return payload
    mine = indicators(symbol)
    checked = macd_fields(payload, None if mine is None else float(mine["dea"]))
    out = dict(payload)
    if checked is None or mine is None:
        for key in ("signal", "histogram", "cross"):
            out.pop(key, None)
        out["unverified"] = "signal line and cross removed: could not be recomputed from candles"
        return out
    line, hist, swapped = checked
    out["signal"], out["histogram"] = line, hist
    if swapped:
        out["cross"] = str(mine["cross"]).replace(" ", "_") or "none"
        out["corrected"] = ("the Skill returned signal and histogram in each other's fields; "
                            "restored and cross recomputed from Bitget 4h candles")
    return out


@dataclass(frozen=True, slots=True)
class CrossCheck:
    """The Skill's number against ours, on the same instrument."""

    symbol: str
    theirs: float | None
    ours: float | None
    agrees: bool | None
    """``None`` when one side could not be computed — unknown, not agreement."""

    note: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol, "skill_rsi": self.theirs, "our_rsi": self.ours,
            "agrees": self.agrees, "note": self.note,
        }


def rsi(closes: Sequence[float], period: int = 14) -> float | None:
    """Wilder's RSI, the definition the indicator is named for.

    Wilder smooths with a running average seeded on the first ``period`` changes, which is not a
    simple mean of the last ``period`` and gives a different number. Getting this wrong would make
    the cross-check below fail against a correct Skill reading, and we would then "discover" a
    problem that was ours.
    """
    if len(closes) < period + 1:
        return None
    gains = losses = 0.0
    for i in range(1, period + 1):
        change = closes[i] - closes[i - 1]
        gains += max(change, 0.0)
        losses += max(-change, 0.0)
    avg_gain, avg_loss = gains / period, losses / period
    for i in range(period + 1, len(closes)):
        change = closes[i] - closes[i - 1]
        avg_gain = (avg_gain * (period - 1) + max(change, 0.0)) / period
        avg_loss = (avg_loss * (period - 1) + max(-change, 0.0)) / period
    if avg_loss == 0:
        return 100.0
    return 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)


def cross_check_rsi(
    client: SkillClient,
    *,
    symbol: str,
    closes: Sequence[float],
    tolerance: float = RSI_AGREEMENT_POINTS,
) -> CrossCheck:
    """Recompute the Skill's RSI from our own candles and report the gap.

    This is what separates integrating a data source from believing one. ``closes`` are 4-hour
    closes for ``symbol``, matching the timeframe the Skill reports, newest last.
    """
    payload, status = client.call("technical_analysis", {"action": "rsi", "symbol": symbol})
    theirs: float | None = None
    if isinstance(payload, dict):
        value = payload.get("rsi")
        if isinstance(value, int | float) and not isinstance(value, bool):
            theirs = float(value)
    ours = rsi(closes)

    if theirs is None or ours is None:
        missing = "the Skill did not answer" if theirs is None else "we have too few bars"
        return CrossCheck(symbol, theirs, ours, None, f"not comparable: {missing} ({status})")

    gap = abs(theirs - ours)
    agrees = gap <= tolerance
    return CrossCheck(
        symbol, round(theirs, 2), round(ours, 2), agrees,
        (
            f"gap {gap:.2f} points against a {tolerance:.1f} tolerance; "
            f"{'agrees' if agrees else 'disagrees'} with our own computation on Bitget candles"
        ),
    )


def main(argv: list[str] | None = None) -> int:
    """Sweep the official Skill surface and write the health report beside the ledger.

    Separate from the decision cycle on purpose: a sweep pays a full timeout for every dead
    upstream, and the cycle reads the result rather than repeating the measurement.

    ``argv`` defaults to ``None`` (real ``sys.argv``) so tests can pass ``[]`` and assert on the
    parsed ``--timeout`` default without touching the network — the reason this parameter exists
    at all is that the default silently drifted to a stale, too-short 12 once before (this
    module's own :data:`DEFAULT_TIMEOUT` was raised to 45 and documented why, but nothing kept
    this CLI's own default in sync with it), and an untestable ``main()`` is exactly how that kind
    of drift hides.
    """
    import argparse
    from datetime import UTC

    from argus.market.evidence import BitgetSkillSource

    parser = argparse.ArgumentParser(description="Probe Bitget's official research Skills.")
    parser.add_argument("--symbol", default="NVDAUSDT")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument(
        "--out", type=Path,
        default=Path(__file__).resolve().parents[3] / "data" / "bitget_skills_health.json",
    )
    args = parser.parse_args(argv)

    report = probe(
        BitgetSkillSource(), symbol=args.symbol, at=datetime.now(UTC), timeout=args.timeout
    )
    for line in report.render():
        print(line)
    for row in report.results:
        print(f"  {row.probe.skill:20} {row.probe.ident:38} {row.health}")
    print(f"saved -> {report.save(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_TIMEOUT",
    "PROBES",
    "RSI_AGREEMENT_POINTS",
    "SKILLS",
    "CrossCheck",
    "Health",
    "Probe",
    "SkillClient",
    "SkillReport",
    "ToolHealth",
    "correct_macd",
    "cross_check_rsi",
    "evidence",
    "hollow",
    "indicators",
    "load",
    "macd_fields",
    "main",
    "probe",
    "rsi",
    "usable_probes",
]
