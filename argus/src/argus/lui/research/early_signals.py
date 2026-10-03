"""A daily list of what is waking up on Bitget, with a score anyone can recompute by hand.

"What should I look at today?" deserves a short list, not a pick. Each liquid USDT perpetual on
Bitget is scored out of 100 on four things a trader reads first, each from a public source and each
shown as its own share of the score:

* **Unusual volume (35):** the last 24 hours' traded value against the average day of the 20 before
  it (Bitget daily candles). 1x is nothing; 3x or more is the full 35.
* **Momentum (25):** the move over the last five daily closes, either way; 10% or more is the full
  25. The sign is shown, because a fall that is waking up is as worth reading as a rise.
* **Crowded funding (20):** the size of the current funding rate per settlement; 0.05% or more is
  the full 20 — the crowd paying up to hold one side.
* **Attention (20):** whether the coin is on CoinGecko's search-trending list now (the market-intel
  Skill's source, `lui/trending.py`).

Taken from an early-signal scanner guide shared with the team on 2026-10-03 (volume, momentum,
news and attention into one transparent 100-point score, reviewed by hand), with Bitget's funding
rate in place of a news count: a count of headlines measures noise, and funding measures money.

The score is descriptive. It ranks what to *read*, never what to buy: the answer says so, shows
every component, and names the hour it was computed. Volume and attention can be manufactured;
a high score is a reason to look, nothing more.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final

from argus.lui.trace import trace_module

ASKED: Final = re.compile(
    r"\bearly[\s-]?signals?\b|\bunusual\s+(?:volume|activity)\b|\bscan\s+(?:the\s+)?market\b|"
    r"\bwhat(?:'s|\s+is)\s+(?:waking\s+up|heating\s+up|stirring)\b|"
    r"\bwhat\s+should\s+i\s+(?:look\s+at|watch|keep\s+an\s+eye\s+on)\s+(?:today|now|this\s+week)\b|"
    r"\b(?:daily|today'?s)\s+(?:scan|screen|movers\s+list)\b",
    re.I,
)
MIN_TURNOVER: Final = 20_000_000.0
"""USD traded in 24 hours for a contract to be scanned: thin books move on little."""
MAX_SCANNED: Final = 40
STABLES: Final = frozenset({"USDCUSDT", "USDEUSDT", "FDUSDUSDT", "DAIUSDT", "TUSDUSDT"})
WEIGHTS: Final = {"volume": 35.0, "momentum": 25.0, "funding": 20.0, "attention": 20.0}


@dataclass(frozen=True, slots=True)
class Scored:
    symbol: str
    turnover: float
    volume_ratio: float | None
    move_5d: float | None
    funding: float
    trending: bool

    @property
    def parts(self) -> dict[str, float]:
        def clamp(x: float) -> float:
            return max(0.0, min(1.0, x))

        return {
            "volume": WEIGHTS["volume"] * clamp(((self.volume_ratio or 1.0) - 1.0) / 2.0),
            "momentum": WEIGHTS["momentum"] * clamp(abs(self.move_5d or 0.0) / 0.10),
            "funding": WEIGHTS["funding"] * clamp(abs(self.funding) / 0.0005),
            "attention": WEIGHTS["attention"] if self.trending else 0.0,
        }

    @property
    def score(self) -> float:
        return sum(self.parts.values())


def _history(symbol: str) -> tuple[float | None, float | None]:
    """(average daily traded value over the 20 days before the last, the 5-day move)."""
    from argus.market import history

    try:
        candles = history.fetch_range(symbol, days=26, interval="1Dutc")
    except Exception:
        return None, None
    closes = [float(c.close) for c in candles]
    values = [float(c.volume) * float(c.close) for c in candles]
    prior = values[-21:-1]
    average = sum(prior) / len(prior) if len(prior) >= 10 else None
    move = closes[-1] / closes[-6] - 1 if len(closes) >= 6 and closes[-6] > 0 else None
    return average, move


def scan(*, trending_symbols: frozenset[str] | None = None) -> list[Scored]:
    """Every liquid perpetual scored, highest first."""
    from argus.lui.research.desk_answers import _tickers

    tickers = _tickers()
    liquid = sorted(
        ((s, float(t.base_volume) * float(t.last), t) for s, t in tickers.items()
         if s.endswith("USDT") and s not in STABLES and float(t.last) > 0),
        key=lambda row: -row[1])
    liquid = [row for row in liquid if row[1] >= MIN_TURNOVER][:MAX_SCANNED]
    if trending_symbols is None:
        trending_symbols = _trending()
    with ThreadPoolExecutor(max_workers=8) as pool:
        histories = list(pool.map(lambda row: _history(row[0]), liquid))
    out = []
    for (symbol, turnover, ticker), (average, move) in zip(liquid, histories, strict=True):
        out.append(Scored(
            symbol=symbol, turnover=turnover,
            volume_ratio=turnover / average if average else None,
            move_5d=move, funding=float(ticker.funding_rate),
            trending=symbol.removesuffix("USDT") in trending_symbols))
    return sorted(out, key=lambda s: -s.score)


def _trending() -> frozenset[str]:
    from argus.lui.trending import coins_from
    from argus.market.skill_mirror import trending

    try:
        return frozenset(c["symbol"] for c in coins_from(trending({})))
    except Exception:
        return frozenset()


def lines(*, now: datetime | None = None, top: int = 8) -> list[str] | None:
    when = now or datetime.now(UTC)
    try:
        scored = scan()
    except Exception:
        return None
    if not scored:
        return None
    # a ticker in another script reads as a bug to an English reader (round 27); it is counted
    shown = [s for s in scored if _latin(s.symbol)]
    unshown = len(scored) - len(shown)
    best = shown[:top]
    if not best:
        return None

    def row(s: Scored) -> str:
        p = s.parts
        ratio = f"{s.volume_ratio:.1f}x" if s.volume_ratio is not None else "n/a"
        move = f"{s.move_5d:+.1%}" if s.move_5d is not None else "n/a"
        return (f"{s.symbol.removesuffix('USDT')} {s.score:.0f}/100 — volume {ratio} its 20-day "
                f"average ({p['volume']:.0f}), 5 days {move} ({p['momentum']:.0f}), funding "
                f"{s.funding:+.4%} ({p['funding']:.0f})"
                + (", on CoinGecko's trending list (20)" if s.trending else "") + ".")

    lead = best[0]
    return [f"Bottom line: of the {len(scored)} Bitget perpetuals trading over "
            f"${MIN_TURNOVER / 1e6:.0f}m a day, "
            f"{lead.symbol.removesuffix('USDT')} scores highest today at {lead.score:.0f}/100 — "
            f"a list of what to read, not what to buy.",
            *(f"{i}. {row(s)}" for i, s in enumerate(best, 1)),
            "Score out of 100: unusual volume 35 (24h traded value against the 20 days before, 3x "
            "or more is full), 5-day move 25 (10% either way is full), funding 20 (0.05% a "
            "settlement is full), CoinGecko trending 20. Volume and attention can be "
            "manufactured; a high score is a reason to look, nothing more."
            + (f" {unshown} contract{'s' if unshown != 1 else ''} named in another script "
               f"{'are' if unshown != 1 else 'is'} left off this list." if unshown else ""),
            f"Computed {when:%d %b %H:%M} UTC from Bitget's public tickers and daily candles and "
            "CoinGecko's trending list. Ask about any name on it for its full picture."]


TODAY_ASKED: Final = re.compile(
    r"\bwhat\s+should\s+i\s+(?:look\s+at|watch|keep\s+an\s+eye\s+on)\s+(?:today|now|this\s+week)\b",
    re.I)
"""The newcomer's "what should I look at today?", which wants the day before it wants a scan."""


def _latin(symbol: str) -> bool:
    return symbol.isascii()


def today_lines(*, now: datetime | None = None) -> list[str] | None:
    """"what should i look at today?" led a newcomer with an unexplained Chinese-character ticker
    and two pump-shaped small caps (a first-time user, round 27): the day first — BTC and ETH
    over 24 hours and the next scheduled US release — then the three Latin-named contracts
    scoring highest on the scan, said as a list to read."""
    from argus.lui.research.desk_answers import _tickers

    when = now or datetime.now(UTC)
    try:
        tickers = _tickers()
    except Exception:
        return None
    majors = []
    for symbol in ("BTCUSDT", "ETHUSDT"):
        ticker = tickers.get(symbol)
        change = getattr(ticker, "change_24h", None) if ticker is not None else None
        if change is not None:
            majors.append(f"{symbol.removesuffix('USDT')} {float(change):+.1%}")
    if not majors:
        return None
    event = None
    try:
        from argus.lui.watchlist import watchlist

        week = watchlist("what is on the calendar this week", now=when)
        event = next((str(x).removeprefix("Scheduled: ") for x in week[0]
                      if str(x).startswith("Scheduled:")), None)
    except Exception:
        event = None
    try:
        scored = [s for s in scan() if _latin(s.symbol)]
    except Exception:
        scored = []
    out = [f"Bottom line: today, {' and '.join(majors)} over 24 hours"
           + (f"; the next scheduled US release this week is {event.rstrip('.')}" if event else
              "; nothing major is on the US calendar this week")
           + ". Those, and anything you already hold, are the place to start."]
    if scored:
        out.append("Moving unusually beyond the big names — a list to read, not to buy, since "
                   "volume like this is often a pump that fades:")
        for s in scored[:3]:
            move = f"{s.move_5d:+.1%}" if s.move_5d is not None else "n/a"
            ratio = f"{s.volume_ratio:.1f}x" if s.volume_ratio is not None else "n/a"
            out.append(f"{s.symbol.removesuffix('USDT')}: {move} over 5 days, trading {ratio} its "
                       f"usual daily value.")
    out.append(f"Computed {when:%d %b %H:%M} UTC from Bitget's public tickers and the BLS and Fed "
               "calendars. Ask \"early signals\" for the full scored scan, or about any name for "
               "its picture.")
    return out


__all__ = ["ASKED", "TODAY_ASKED", "WEIGHTS", "Scored", "lines", "scan", "today_lines"]

trace_module(globals())
