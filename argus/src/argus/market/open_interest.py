"""Open interest: how much of a contract is held open, set against its own trading and the market.

Bitget publishes open interest as a current figure only — ``holdingAmount`` on every USDT-futures
ticker (`/api/v2/mix/market/tickers`, the same number `/api/v3/market/open-interest` returns
per symbol, checked 2026-09-25) — with no history endpoint in its public API or in the agent
toolkit (`agent-sdk/src/generated/catalog.ts` lists `open-interest` and `open-interest-limit`,
both current). `bitget-signal`'s ``derivatives_sentiment`` offers an open-interest history, but
it reads Binance and returned an empty error for every call made from here on 2026-09-25.

So the reading is built from what is measured now, and history is recorded from now on:

* **Notional held open** — holding times last price, in USDT.
* **Open interest over 24h volume** — how many days of today's trading are held open. A contract
  whose positions sit far longer than its turnover is held by people who are not trading it,
  which is where forced unwinds come from. Ranked against every USDT perpetual with more than
  $1m of daily volume, so "high" means high on this venue today.
* **Change** — against the snapshot nearest 24 hours ago in `data/open_interest_history.jsonl`,
  written hourly by the desk's refresh loop (`record`). Stated only when such a snapshot exists;
  never backfilled or estimated.

Funding says which side is paying to hold, and with a change in open interest it says whether new
money is arriving long or short; without the change, the answer says the change is not yet
known rather than guessing it.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

MIN_VOLUME_USDT = 1_000_000.0
HISTORY_HOURS = 24
WINDOW_HOURS = 3
"""A snapshot within three hours of 24 hours ago counts as "a day ago"; outside that, none does."""


def _history_path() -> Path:
    override = os.environ.get("ARGUS_DATA_DIR", "").strip()
    base = Path(override) if override else Path(__file__).resolve().parents[3] / "data"
    return base / "open_interest_history.jsonl"


@dataclass(frozen=True)
class Reading:
    symbol: str
    holding: float
    """Contracts held open, in the base asset."""
    notional: float
    volume_24h: float
    funding: float
    turnover_days: float | None
    """Open interest over 24h volume: days of today's trading held open."""
    rank_pct: float | None
    """Share of liquid USDT perpetuals with a lower turnover_days, 0-100."""
    peers: int
    change_24h: float | None = None
    """Fractional change in holding against the snapshot a day ago, when one exists."""
    change_from: str | None = None


def snapshot(fetch: Any = None) -> dict[str, dict[str, float]]:
    """Every USDT perpetual's holding, last price, 24h USDT volume and funding, in one call."""
    if fetch is None:
        from argus.market.bitget import _get

        rows = _get("/api/v2/mix/market/tickers", {"productType": "usdt-futures"}) or []
    else:
        rows = fetch()
    out: dict[str, dict[str, float]] = {}
    for row in rows:
        try:
            holding = float(row.get("holdingAmount") or 0)
            last = float(row.get("lastPr") or 0)
            volume = float(row.get("usdtVolume") or row.get("quoteVolume") or 0)
            funding = float(row.get("fundingRate") or 0)
        except (TypeError, ValueError):
            continue
        if row.get("symbol") and holding > 0 and last > 0:
            out[str(row["symbol"])] = {"holding": holding, "last": last, "volume": volume,
                                       "funding": funding}
    return out


def _past(symbol: str, now: datetime, path: Path | None = None) -> tuple[float, str] | None:
    path = path or _history_path()
    if not path.exists():
        return None
    target = now - timedelta(hours=HISTORY_HOURS)
    best: tuple[float, float, str] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
            at = datetime.fromisoformat(row["at"])
        except (ValueError, KeyError, TypeError):
            continue
        held = (row.get("holding") or {}).get(symbol)
        if held is None:
            continue
        gap = abs((at - target).total_seconds()) / 3600
        if gap <= WINDOW_HOURS and (best is None or gap < best[0]):
            best = (gap, float(held), row["at"])
    return (best[1], best[2]) if best else None


def read(symbol: str, *, now: datetime | None = None, fetch: Any = None,
         history: Path | None = None) -> Reading | None:
    book = snapshot(fetch)
    own = book.get(symbol)
    if own is None:
        return None
    notional = own["holding"] * own["last"]
    turnover = notional / own["volume"] if own["volume"] > 0 else None
    ratios = [v["holding"] * v["last"] / v["volume"] for v in book.values()
              if v["volume"] >= MIN_VOLUME_USDT]
    rank = (100.0 * sum(r < turnover for r in ratios) / len(ratios)
            if turnover is not None and ratios and own["volume"] >= MIN_VOLUME_USDT else None)
    past = _past(symbol, now or datetime.now(UTC), history)
    return Reading(symbol=symbol, holding=own["holding"], notional=notional,
                   volume_24h=own["volume"], funding=own["funding"], turnover_days=turnover,
                   rank_pct=rank, peers=len(ratios),
                   change_24h=(own["holding"] / past[0] - 1) if past else None,
                   change_from=past[1] if past else None)


def lines(reading: Reading, ticker: str) -> list[str]:
    """The reading in words. Every figure is one of the fields above."""
    out = [f"Open interest: {reading.holding:,.2f} {ticker} held open on Bitget, about "
           f"${reading.notional:,.0f}."]
    if reading.turnover_days is not None:
        where = ("" if reading.rank_pct is None else
                 f" — higher than {reading.rank_pct:.0f}% of the {reading.peers} USDT perpetuals "
                 f"trading over $1m a day")
        out.append(f"That is {reading.turnover_days:.2f} days of today's volume "
                   f"(${reading.volume_24h:,.0f}) held open{where}. Positions that sit far longer "
                   f"than the contract trades are where forced unwinds come from.")
    side = ("longs pay shorts" if reading.funding > 0 else
            "shorts pay longs" if reading.funding < 0 else "neither side pays")
    if reading.change_24h is not None:
        arriving = ("new positions opened" if reading.change_24h > 0 else "positions closed")
        out.append(f"Over the last day open interest changed {reading.change_24h * 100:+.1f}% "
                   f"(against {reading.change_from}): {arriving}, while funding is "
                   f"{reading.funding * 100:+.4f}% — {side}.")
    else:
        out.append(f"Funding is {reading.funding * 100:+.4f}% — {side}. The day's change in open "
                   f"interest is not known yet: Bitget publishes only the current figure, and "
                   f"this desk's own hourly record of it has no reading from a day ago.")
    return out


def record(path: Path | None = None, *, fetch: Any = None,
           now: datetime | None = None) -> int:
    """Append one snapshot of every contract's holding. Returns the number recorded."""
    book = snapshot(fetch)
    path = path or _history_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"at": (now or datetime.now(UTC)).isoformat(timespec="seconds"),
           # Liquid contracts only: 805 contracts an hour would grow the file by 0.7MB a day
           # for names nobody asks about.
           "holding": {s: v["holding"] for s, v in sorted(book.items())
                       if v["volume"] >= MIN_VOLUME_USDT}}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
    return len(row["holding"])


def main() -> int:  # pragma: no cover - CLI
    print(f"recorded {record()} contracts")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
