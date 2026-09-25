"""Money actually arriving in the US spot bitcoin and ether ETFs, and Strategy's bitcoin buying.

**Why this source.** `market/bitget_positioning.py` documents why Bitget's own ``crypto_etf_flows``
is not quoted: its ``net_flow_1d`` is the day's change in each fund's dollar holding, so it followed
bitcoin's price and read zero on weekends — a price move dressed as money arriving. SoSoValue's
``/etfs/summary-history`` reports the aggregate daily **net inflow** (creations less redemptions)
per asset, with traded value and net assets, published after the US close. That is the series the
gap was waiting for. Found through the owner's earlier SoSoValue project (POD), whose SDK
(`packages/sosovalue-sdk`, MIT) gave the endpoint, the ``x-soso-api-key`` header and the response
shape; this module calls the API directly with the standard library.

**Why a snapshot.** The key stays on the desk's machine (`.secrets/sosovalue.env`); the hosted
console reads the dated artefact, like every other slow sweep, and says how old it is.

**What is said, and what is not.** The net inflow, its five-day sum, and how unusual the latest
day is against the last thirty (a z-score) — facts about positioning in a regulated product, not a
forecast. Strategy's purchase history adds the dates and sizes of its bitcoin buys, which move MSTR
and are public filings.

    python -m argus.market.etf_flows           # writes data/etf_flows.json
"""

from __future__ import annotations

import json
import statistics
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

DATA = Path(__file__).resolve().parents[3] / "data"
FLOWS_PATH = DATA / "etf_flows.json"
SECRETS = Path(__file__).resolve().parents[4] / ".secrets" / "sosovalue.env"
BASE = "https://openapi.sosovalue.com/openapi/v1"
STALE_AFTER = timedelta(days=4)
"""Flows publish once per US trading day; older than a long weekend is stale."""
LINKED = {"BTCUSDT": ("BTC",), "ETHUSDT": ("ETH",), "COINUSDT": ("BTC", "ETH"),
          "MSTRUSDT": ("BTC",)}
"""Which fund flows belong in which name's answer: the asset itself, and the two stocks whose
price follows crypto (an exchange and a bitcoin treasury company)."""

Get = Callable[[str, Mapping[str, Any] | None], Any]


class FlowError(RuntimeError):
    """SoSoValue did not answer usefully. Carries its own reason."""


def _key() -> tuple[str, str]:
    values: dict[str, str] = {}
    for line in SECRETS.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            name, value = line.split("=", 1)
            values[name.strip()] = value.strip().strip('"').strip("'")
    key = values.get("SOSOVALUE_API_KEY")
    if not key:
        raise FlowError("no SOSOVALUE_API_KEY in .secrets/sosovalue.env")
    return key, values.get("SOSOVALUE_BASE_URL") or BASE


def get_live(path: str, query: Mapping[str, Any] | None = None) -> Any:
    key, base = _key()
    url = base.rstrip("/") + path + ("?" + urllib.parse.urlencode(query) if query else "")
    request = urllib.request.Request(url, headers={"x-soso-api-key": key,
                                                   "User-Agent": "ARGUS research desk"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read())
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        raise FlowError(f"SoSoValue {path} unreachable: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("code") != 0:
        raise FlowError(f"SoSoValue {path} answered {str(payload)[:120]}")
    return payload.get("data")


def fund_flows(asset: str, rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Latest day, five-day sum and a z-score of the latest day against the prior thirty."""
    ordered = sorted(rows, key=lambda r: str(r["date"]))
    flows = [float(r["total_net_inflow"]) for r in ordered]
    if not flows:
        raise FlowError(f"no {asset} ETF flow rows")
    latest = ordered[-1]
    prior = flows[-31:-1]
    z = None
    if len(prior) >= 10 and statistics.pstdev(prior) > 0:
        z = (flows[-1] - statistics.fmean(prior)) / statistics.pstdev(prior)
    streak = 0
    for value in reversed(flows):
        if value == 0 or (streak and (value > 0) != (flows[-1] > 0)):
            break
        streak += 1
    return {
        "asset": asset,
        "date": str(latest["date"]),
        "net_inflow_usd": flows[-1],
        "five_day_usd": sum(flows[-5:]),
        "z_latest": None if z is None else round(z, 2),
        "streak_days": streak,
        "net_assets_usd": float(latest.get("total_net_assets") or 0.0),
        "cumulative_usd": float(latest.get("cum_net_inflow") or 0.0),
        "days": len(flows),
    }


def treasury(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda r: str(r["date"]), reverse=True)
    buys = [{"date": str(r["date"]), "btc": float(r["btc_acq"]),
             "usd": float(r["acq_cost"])} for r in ordered[:5]]
    return {"ticker": "MSTR", "holding_btc": float(ordered[0]["btc_holding"]) if ordered else None,
            "recent": buys}


def sweep(get: Get = get_live, now: datetime | None = None) -> dict[str, Any]:
    stamp = now or datetime.now(UTC)
    out: dict[str, Any] = {"generated_at": stamp.isoformat(),
                           "source": "SoSoValue /etfs/summary-history and /btc-treasuries "
                                     "(US spot ETFs; creations less redemptions)",
                           "funds": {}, "errors": []}
    for asset in ("BTC", "ETH"):
        try:
            # No start_date: SoSoValue answers 403 when one is sent (2026-09-25); the default
            # window is its most recent month, newest first.
            rows = get("/etfs/summary-history", {"symbol": asset, "country_code": "US",
                                                 "limit": 50})
            out["funds"][asset] = fund_flows(asset, list(rows or []))
        except (FlowError, KeyError, TypeError, ValueError) as exc:
            out["errors"].append(f"{asset}: {exc}")
    try:
        out["treasury"] = treasury(list(get("/btc-treasuries/MSTR/purchase-history",
                                            {"limit": 5}) or []))
    except (FlowError, KeyError, TypeError, ValueError) as exc:
        out["errors"].append(f"MSTR treasury: {exc}")
    return out


def load(path: Path = FLOWS_PATH) -> dict[str, Any] | None:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _usd(value: float) -> str:
    sign = "+" if value > 0 else "-" if value < 0 else ""
    size = abs(value)
    return f"{sign}${size / 1e9:,.2f}bn" if size >= 1e9 else f"{sign}${size / 1e6:,.0f}m"


def lines_for(symbol: str, snapshot: dict[str, Any] | None,
              today: date | None = None) -> list[str]:
    """The console's sentences for a crypto-linked name; empty for anything else."""
    if not snapshot or symbol not in LINKED:
        return []
    out = []
    for asset in LINKED[symbol]:
        fund = (snapshot.get("funds") or {}).get(asset)
        if not fund:
            continue
        day = date.fromisoformat(fund["date"])
        stale = (today or datetime.now(UTC).date()) - day > STALE_AFTER
        unusual = ""
        if fund.get("z_latest") is not None:
            z = fund["z_latest"]
            unusual = (f", {abs(z):.1f} standard deviations {'above' if z > 0 else 'below'} its "
                       f"30-day norm" if abs(z) >= 1 else ", an ordinary day for it")
        streak = fund.get("streak_days", 0)
        run = (f"; {streak} {'inflow' if fund['net_inflow_usd'] > 0 else 'outflow'} days running"
               if streak >= 3 else "")
        out.append(f"US spot {asset} ETFs, {day:%d %b}: net {_usd(fund['net_inflow_usd'])} "
                   f"(creations less redemptions){unusual}; five days {_usd(fund['five_day_usd'])}"
                   f"{run}; ${fund['net_assets_usd'] / 1e9:,.0f}bn held"
                   + (" — stale" if stale else "") + " (SoSoValue).")
    held = snapshot.get("treasury") or {}
    if symbol == "MSTRUSDT" and held.get("recent"):
        last = held["recent"][0]
        verb = "bought" if last["btc"] > 0 else "sold"
        out.append(f"Strategy last {verb} {abs(last['btc']):,.0f} BTC on {last['date']} "
                   f"(${abs(last['usd']) / 1e6:,.0f}m), holding {held['holding_btc']:,.0f} BTC; "
                   f"its buys are filed with the SEC and move MSTR on the day (SoSoValue).")
    return out


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI, live network
    snapshot = sweep()
    FLOWS_PATH.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    for symbol in LINKED:
        for line in lines_for(symbol, snapshot):
            print(f"{symbol}: {line}")
    print("errors:", snapshot["errors"] or "none")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["FLOWS_PATH", "FlowError", "fund_flows", "lines_for", "load", "sweep", "treasury"]
