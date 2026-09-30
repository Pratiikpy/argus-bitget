"""What the listed options market says about an underlying, read from Cboe's free delayed quotes.

The perception comparison against OpenBB's keyless providers (`eval/perception_breadth.py`,
2026-09-28) found options chains answering for all ten underlyings there and for none here: the
desk and the workbench could see the perpetual's funding and the stock's short volume, and not the
one market where traders state, with money, how far and which way they expect the stock to move.
OpenBB's ``cboe`` provider (``openbb_cboe/models/options_chains.py``, AGPL-3.0) reads the same
public file; its behaviour was read, none of its code is used here.

**The source.** ``cdn.cboe.com/api/global/delayed_quotes/options/{SYMBOL}.json`` needs no key and
returns the whole listed chain with bid, ask, implied volatility, delta, open interest and volume
per contract, plus the underlying's price and Cboe's own 30-day implied volatility. It is delayed
(about fifteen minutes in session). Its ``timestamp`` is UTC, not Eastern: on 2026-09-28 it read
21:55:18 at 21:56 UTC (17:56 in New York), so it is converted before it is shown.

**What is computed, and why each one.**

* **Put/call ratios**, by volume (today's flow) and by open interest (standing positions). A reader
  of an equity desk asks "is the crowd hedged", and these are the numbers that answer it.
* **The implied move to the nearest expiry at least five days out**: the at-the-money straddle's mid
  price over spot. It is the market's own price for the size of the move, which is what a hurdle
  or a stop should be read against, and it needs no model: it is a price.
* **The 25-delta skew** at that expiry: the put's implied volatility minus the call's, at the
  contracts whose deltas are nearest -0.25 and +0.25. Positive means downside protection is bid.

Contracts with no two-sided quote or a zero implied volatility are left out of the straddle and the
skew rather than read as zero, and the count kept is reported. The put/call ratios count every
listed contract: a trade that printed or a position that is open is real whether or not the
contract is quoted at this moment.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any
from zoneinfo import ZoneInfo

from argus.truth import http
from argus.truth.evidence import Evidence

CHAIN_URL = "https://cdn.cboe.com/api/global/delayed_quotes/options/{symbol}.json"
INDEX_URL = "https://cdn.cboe.com/api/global/delayed_quotes/options/_{symbol}.json"
"""Cboe prefixes index roots (SPX, VIX, NDX) with an underscore. None of the twelve rTokens'
underlyings is an index, so the plain file is the one read; the constant is kept so the rule is
visible where the URL is built."""

MIN_DAYS_TO_EXPIRY = 5
"""The implied move is read at the first expiry at least this many calendar days out. A same-day
or next-day straddle prices one session's gap, which is a different question."""

TIMEOUT_S = 20.0
NEW_YORK = ZoneInfo("America/New_York")
FLAT_SKEW_VOL_POINTS = 0.5
"""Below half a vol point the two wings are priced alike for any reading a trader would act on."""

_OCC = re.compile(r"^(?P<root>[A-Z.]+)(?P<ymd>\d{6})(?P<right>[CP])(?P<strike>\d{8})$")


class OptionsError(RuntimeError):
    """The chain could not be read or does not support the figure asked for."""


@dataclass(frozen=True)
class Contract:
    root: str
    expiry: date
    right: str
    strike: float
    bid: float
    ask: float
    iv: float
    delta: float
    open_interest: float
    volume: float

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def quoted(self) -> bool:
        return self.bid > 0 and self.ask >= self.bid and self.iv > 0


def parse_contract(row: Mapping[str, Any]) -> Contract | None:
    """One Cboe row, or ``None`` when its option symbol is not an OCC symbol."""
    match = _OCC.match(str(row.get("option", "")))
    if match is None:
        return None
    ymd = match.group("ymd")
    return Contract(
        root=match.group("root"),
        expiry=date(2000 + int(ymd[:2]), int(ymd[2:4]), int(ymd[4:])),
        right=match.group("right"),
        strike=int(match.group("strike")) / 1000.0,
        bid=float(row.get("bid") or 0.0), ask=float(row.get("ask") or 0.0),
        iv=float(row.get("iv") or 0.0), delta=float(row.get("delta") or 0.0),
        open_interest=float(row.get("open_interest") or 0.0),
        volume=float(row.get("volume") or 0.0),
    )


@dataclass(frozen=True)
class OptionsSummary:
    symbol: str
    quoted_at: str
    """Cboe's snapshot time, converted from the UTC it is served in to New York time."""
    spot: float
    iv30: float | None
    """Cboe's 30-day implied volatility, in percent."""
    contracts: int
    quoted_contracts: int
    put_call_volume: float | None
    put_call_open_interest: float | None
    expiry: date | None
    days_to_expiry: int | None
    atm_strike: float | None
    implied_move_pct: float | None
    skew_25d: float | None
    """Put IV minus call IV at the contracts nearest -0.25 and +0.25 delta, in vol points."""

    def claim(self) -> str:
        parts = [f"{self.symbol} listed options (Cboe delayed, {self.quoted_at} New York): spot "
                 f"{self.spot:.2f}"]
        if self.iv30 is not None:
            parts.append(f"30-day implied vol {self.iv30:.1f}%")
        if self.implied_move_pct is not None and self.expiry is not None:
            parts.append(f"at-the-money straddle prices a {self.implied_move_pct:.1f}% move to "
                         f"{self.expiry.isoformat()} ({self.days_to_expiry} days)")
        if self.skew_25d is not None:
            lean = ("flat" if abs(self.skew_25d) < FLAT_SKEW_VOL_POINTS else
                    "puts bid" if self.skew_25d > 0 else "calls bid")
            parts.append(f"25-delta skew {self.skew_25d:+.1f} vol points ({lean})")
        if self.put_call_volume is not None:
            parts.append(f"put/call {self.put_call_volume:.2f} by volume")
        if self.put_call_open_interest is not None:
            parts.append(f"{self.put_call_open_interest:.2f} by open interest")
        parts.append(f"{self.quoted_contracts} of {self.contracts} contracts two-sided")
        return "; ".join(parts)


def _ratio(num: float, den: float) -> float | None:
    return round(num / den, 3) if den > 0 else None


def _new_york(stamp: str) -> str:
    """Cboe's UTC ``YYYY-MM-DD HH:MM:SS`` as New York wall time; unparseable text is kept."""
    try:
        at = datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
    except ValueError:
        return stamp
    return at.astimezone(NEW_YORK).strftime("%Y-%m-%d %H:%M")


def summarise(payload: Mapping[str, Any], *, today: date,
              after: date | None = None) -> OptionsSummary:
    """Reduce one Cboe chain to the figures above. Pure: no network.

    ``after``, when given, picks the first expiry on or after that day rather than the nearest:
    the move priced around an earnings report is the straddle that spans it. The nearest expiry,
    16 days before TSLA's report, was quoted as "the move around its next earnings" (a judge's
    audit, 2026-09-30)."""
    data = payload.get("data") or {}
    symbol = str(data.get("symbol") or "")
    spot = float(data.get("current_price") or data.get("close") or 0.0)
    if not symbol or spot <= 0:
        raise OptionsError("the chain carries no underlying symbol or price")
    contracts = [c for c in (parse_contract(r) for r in data.get("options") or []) if c]
    if not contracts:
        raise OptionsError(f"{symbol}: the chain lists no contracts")
    puts = [c for c in contracts if c.right == "P"]
    calls = [c for c in contracts if c.right == "C"]
    quoted = [c for c in contracts if c.quoted]
    later = sorted({c.expiry for c in quoted if (c.expiry - today).days >= MIN_DAYS_TO_EXPIRY
                    and (after is None or c.expiry >= after)})
    expiry = later[0] if later else None
    atm = move = skew = None
    if expiry is not None:
        at = [c for c in quoted if c.expiry == expiry]
        strikes = sorted({c.strike for c in at if c.right == "C"}
                         & {c.strike for c in at if c.right == "P"})
        if strikes:
            atm = min(strikes, key=lambda k: abs(k - spot))
            call = next(c for c in at if c.right == "C" and c.strike == atm)
            put = next(c for c in at if c.right == "P" and c.strike == atm)
            move = round((call.mid + put.mid) / spot * 100.0, 2)
        wing_put = min((c for c in at if c.right == "P"), key=lambda c: abs(c.delta + 0.25),
                       default=None)
        wing_call = min((c for c in at if c.right == "C"), key=lambda c: abs(c.delta - 0.25),
                        default=None)
        if wing_put is not None and wing_call is not None:
            skew = round((wing_put.iv - wing_call.iv) * 100.0, 2)
    iv30 = data.get("iv30")
    return OptionsSummary(
        symbol=symbol, quoted_at=_new_york(str(payload.get("timestamp") or "")), spot=spot,
        iv30=float(iv30) if iv30 is not None else None,
        contracts=len(contracts), quoted_contracts=len(quoted),
        put_call_volume=_ratio(sum(c.volume for c in puts), sum(c.volume for c in calls)),
        put_call_open_interest=_ratio(sum(c.open_interest for c in puts),
                                      sum(c.open_interest for c in calls)),
        expiry=expiry, days_to_expiry=(expiry - today).days if expiry else None,
        atm_strike=atm, implied_move_pct=move, skew_25d=skew,
    )


def fetch_chain(symbol: str, *, timeout: float = TIMEOUT_S) -> dict[str, Any]:
    """The raw delayed chain for ``symbol``. Raises :class:`argus.truth.http.RpcError`."""
    blob = http.fetch_json(CHAIN_URL.format(symbol=symbol.upper()), timeout=timeout)
    if not isinstance(blob, dict):
        raise OptionsError(f"{symbol}: Cboe returned {type(blob).__name__}, not an object")
    return blob


def options_summary(symbol: str, *, now: datetime | None = None,
                    timeout: float = TIMEOUT_S, after: date | None = None) -> OptionsSummary:
    now = now or datetime.now(UTC)
    # Days to expiry are counted on the exchange's calendar: after 20:00 in New York the UTC date
    # is already tomorrow's.
    return summarise(fetch_chain(symbol, timeout=timeout), today=now.astimezone(NEW_YORK).date(),
                     after=after)


def options_evidence(symbol: str, *, as_of: datetime,
                     timeout: float = TIMEOUT_S) -> tuple[list[Evidence], str]:
    """The summary as one evidence item, and a status line either way. LIVE ONLY: Cboe serves
    today's chain, so a replayed ``as_of`` gets nothing rather than today's options."""
    if abs((datetime.now(UTC) - as_of).total_seconds()) > 3600:
        return [], f"options:{symbol}: not replayable (Cboe serves the current chain only)"
    try:
        summary = options_summary(symbol, now=as_of, timeout=timeout)
    except (http.RpcError, OptionsError, ValueError, KeyError) as exc:
        return [], f"options:{symbol}: unavailable ({http.reason_of(exc)})"
    return [Evidence(id=f"options-{symbol}-{as_of.date().isoformat()}", claim=summary.claim(),
                     source="news", available_at=as_of, credibility=0.9)], (
        f"options:{symbol}: {summary.quoted_contracts} quoted contracts")


def summaries(symbols: Sequence[str]) -> dict[str, OptionsSummary | str]:
    """Each symbol's summary, or the reason it has none."""
    out: dict[str, OptionsSummary | str] = {}
    for symbol in symbols:
        try:
            out[symbol] = options_summary(symbol)
        except (http.RpcError, OptionsError, ValueError, KeyError) as exc:
            out[symbol] = http.reason_of(exc)
    return out


__all__ = [
    "CHAIN_URL",
    "Contract",
    "OptionsError",
    "OptionsSummary",
    "fetch_chain",
    "options_evidence",
    "options_summary",
    "parse_contract",
    "summarise",
]
