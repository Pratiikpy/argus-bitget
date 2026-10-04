"""Pyth Pro: US equities priced through the night, from the venues that trade them (build-list 4.7).

Pyth's equity feeds carry four sessions — regular, pre-market (04:00-09:30 New York), post-market
(16:00-20:00) and overnight (20:00-04:00) — each with its own publisher minimum
(``GET /v1/symbols``, read live 2026-10-05: ``Equity.US.NVDA/USD`` has ``over_night`` with
``min_pub`` 2). That is the quote ``eval/overnight_comparison.py`` names as not run ("Blue Ocean ATS
overnight quotes, a paid feed"): the one rival the console's perpetual-implied open has never been
scored against at 03:30, the hour its claim lives in.

**Taken from** Pyth's own MCP server (``pyth-network/pyth-crosschain`` ``apps/mcp``, Apache-2.0):
``src/clients/history.ts`` — ``/v1/symbols`` unauthenticated; ``/v1/{channel}/price`` with ``ids``
repeated and ``timestamp`` in microseconds, ``Authorization: Bearer <token>``; and
``src/utils/display-price.ts`` — the price is an integer, the display price ``price * 10^exponent``.
The channel is ``fixed_rate@200ms``, the one its tests and the History API docs use.

**What needs a key, checked 2026-10-05.** Every price read answers 401 without one: Hermes
``/v2/updates/price/latest`` and the legacy ``/api/latest_price_feeds``, ``hermes-beta``,
Benchmarks ``/v1/updates/price/{t}``, and the History API's ``/price`` and ``/history`` (its docs
mark them "Auth: Yes"); the hosted MCP (``mcp.pyth.network/mcp``) takes the same token as a tool
argument. The key is free with a Pyth Terminal account (``docs.pyth.network/price-feeds/pro/
acquire-api-key``). It is read from ``PYTH_PRO_API_KEY`` in the environment or
``.secrets/pyth.env``, never from code. Symbols need none and are tested live.

**NOT VERIFIED until a key exists:** the live ``/price`` response. Its shape here is the one Pyth's
MCP validates (``src/clients/types.ts`` ``HistoricalPriceResponseSchema``: a list of objects with
``price_feed_id``, ``price``, ``exponent``, ``publish_time``); ``publish_time`` is read as seconds,
milliseconds or microseconds by its size, since the fixture uses seconds and the API's other
timestamps are microseconds.
"""

from __future__ import annotations

import os
import urllib.parse
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from argus.truth import http

HISTORY_URL: Final = "https://pyth.dourolabs.app"
CHANNEL: Final = "fixed_rate@200ms"
TIMEOUT_S: Final = 20.0
SESSIONS: Final = ("regular", "pre_market", "post_market", "over_night")
SECRETS: Final = Path(__file__).resolve().parents[4] / ".secrets" / "pyth.env"


class PythError(RuntimeError):
    """A Pyth read that could not be completed; the message says which and why."""


class NoKey(PythError):
    """No Pyth Pro key is configured, so no price can be read."""


@dataclass(frozen=True)
class Feed:
    lazer_id: int
    symbol: str
    asset_type: str
    exponent: int
    sessions: tuple[str, ...]
    """The market sessions the feed publishes in, of :data:`SESSIONS`."""


@dataclass(frozen=True)
class Price:
    lazer_id: int
    price: float
    published: datetime
    session: str | None
    publishers: int | None


def key() -> str | None:
    """The configured Pyth Pro key, or ``None``: the environment first, then ``.secrets``."""
    found = os.environ.get("PYTH_PRO_API_KEY", "").strip()
    if found:
        return found
    try:
        for line in SECRETS.read_text(encoding="utf-8").splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "PYTH_PRO_API_KEY" and value.strip():
                return value.strip().strip('"').strip("'")
    except OSError:
        return None
    return None


def _get(path: str, params: Sequence[tuple[str, str]], *, token: str | None) -> Any:
    # ``ids`` repeats, so the query is encoded here rather than from a mapping
    url = f"{HISTORY_URL}{path}?{urllib.parse.urlencode(params)}"
    headers = {"Authorization": f"Bearer {token}"} if token else None
    try:
        return http.fetch_json(url, headers=headers, timeout=TIMEOUT_S)
    except http.RpcError as exc:
        status = http.status_of(exc)
        raise PythError(f"Pyth {path} answered HTTP {status}" if status is not None
                        else f"Pyth {path} did not answer ({http.reason_of(exc)})") from exc


def parse_feed(raw: Mapping[str, Any]) -> Feed:
    sessions = raw.get("market_sessions") or {}
    return Feed(lazer_id=int(raw["pyth_lazer_id"]), symbol=str(raw["symbol"]),
                asset_type=str(raw.get("asset_type", "")), exponent=int(raw.get("exponent", 0)),
                sessions=tuple(s for s in SESSIONS if s in sessions))


def symbols(query: str | None = None, asset_type: str | None = None) -> list[Feed]:
    """Pyth's feed list (no key needed)."""
    params = [(k, v) for k, v in (("query", query), ("asset_type", asset_type)) if v]
    raw = _get("/v1/symbols", params, token=None)
    if not isinstance(raw, list):
        raise PythError("Pyth /v1/symbols did not return a list")
    return [parse_feed(r) for r in raw]


def equity_feed(ticker: str, feeds: Iterable[Feed] | None = None) -> Feed | None:
    """The US listing's feed for ``ticker`` (``Equity.US.<T>/USD``), not an index or token."""
    wanted = f"Equity.US.{ticker.upper()}/USD"
    pool = feeds if feeds is not None else symbols(query=ticker, asset_type="equity")
    return next((f for f in pool if f.symbol == wanted), None)


def _moment(value: float) -> datetime:
    """``publish_time`` in seconds, milliseconds or microseconds, read by its size."""
    seconds = value / 1e6 if value > 1e14 else value / 1e3 if value > 1e11 else value
    return datetime.fromtimestamp(seconds, UTC)


def parse_prices(raw: Any) -> dict[int, Price]:
    """``/v1/{channel}/price``'s list, per feed id; a row without a price is left out."""
    if not isinstance(raw, list):
        raise PythError("Pyth /price did not return a list")
    out: dict[int, Price] = {}
    for row in raw:
        if row.get("price") is None or row.get("publish_time") is None:
            continue
        exponent = int(row.get("exponent") or 0)
        count = row.get("publisher_count")
        out[int(row["price_feed_id"])] = Price(
            lazer_id=int(row["price_feed_id"]), price=float(row["price"]) * 10.0 ** exponent,
            published=_moment(float(row["publish_time"])),
            session=row.get("market_session"), publishers=None if count is None else int(count))
    return out


def prices_at(ids: Sequence[int], moment: datetime, *,
              token: str | None = None) -> dict[int, Price]:
    """Each feed's price at ``moment`` (History API, needs a key)."""
    token = token or key()
    if not token:
        raise NoKey("no Pyth Pro key: set PYTH_PRO_API_KEY or .secrets/pyth.env (free with a "
                    "Pyth Terminal account)")
    params = [("ids", str(i)) for i in ids]
    params.append(("timestamp", str(int(moment.timestamp() * 1_000_000))))
    return parse_prices(_get(f"/v1/{CHANNEL}/price", params, token=token))


def main() -> int:  # pragma: no cover - CLI, live network
    feeds = symbols(asset_type="equity")
    overnight = [f for f in feeds
                 if "over_night" in f.sessions and f.symbol.startswith("Equity.US.")]
    print(f"{len(feeds)} equity feeds; {len(overnight)} US listings publish overnight")
    print("key configured" if key() else "no key configured: prices cannot be read")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
