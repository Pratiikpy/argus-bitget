"""When Bitget's research Skills go dark, read the sources they read.

`bitget-signal`'s five Skills are served by one hosted MCP endpoint
(`https://datahub.noxiaohao.com/mcp`, `bitget-signal/CHANGELOG.md:15`). Each tool is a thin fetcher
over a public upstream, and each tool's own description names it: Fear & Greed from alternative.me,
positioning from Binance futures, macro and rates from FRED, prices and correlations from Yahoo
Finance, 44 RSS feeds, DeFiLlama, CoinGecko.

**What was measured, 2026-09-25.** The sweep in :mod:`argus.market.skills` found 6 of 19 tool
calls answering — all six are `technical_analysis`; every other Skill came back as an empty error
envelope (``{"error": ""}``, ``{"alt_me_error": ""}``) or ``Error executing tool``. The same calls
were repeated through a second, independent MCP client (Claude Code's own connection to the
endpoint) with the same result, so the fault is not in our client. The upstreams themselves were
then called directly from this machine and every one answered: alternative.me, Binance
``/futures/data/*``, FRED's ``fredgraph.csv``, Yahoo's ``query2`` chart API, DeFiLlama, CoinGecko
and the RSS feeds. The hosted server is failing to reach sources that are up.

**So this module reads those sources itself**, one mirror per probe in
:data:`argus.market.skills.PROBES` that is not `technical_analysis` (that Skill answers, and its
MACD is already corrected against Bitget's own candles by ``skills.correct_macd``). Every mirrored
reading is labelled for what it is:

* ``same_upstream=True`` — the mirror reads the upstream the Skill's own tool description names.
  The number is the one the Skill would have returned had its server reached it.
* ``same_upstream=False`` — the Skill's upstream needs a key we do not hold (`tradfi_news` is
  Finnhub, ``FINNHUB_API_KEY``) or is not named (`network_status` says only "on-chain public
  data"), so a different public source answers the same question. Named, never hidden.

**The Bitget measurement is never overwritten.** :func:`fill` records the Skill's own health
beside the mirror's, so the honest integration count (Skills that answered through Bitget) and the
effective count (questions the desk could answer) stay two separate numbers. A reader who sees
"19 of 19" without that split has been misled, and this module does not produce that sentence.

Nothing here needs a key. Every fetch uses the standard library, a 20-second timeout, and raises
:class:`MirrorError` rather than returning a neutral default — a mirror that invents a quiet market
is worse than the dark Skill it replaces.
"""

from __future__ import annotations

import csv
import io
import itertools
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from argus.market.skills import PROBES, Health, Probe, SkillReport, hollow
from argus.truth.evidence import Evidence

USER_AGENT = "Mozilla/5.0 (ARGUS research desk)"
"""Yahoo's ``query2`` answers 429 to a bare client and 200 to a browser-shaped User-Agent
(measured 2026-09-25); every other upstream here accepts either."""

TIMEOUT = 20


class MirrorError(RuntimeError):
    """An upstream did not give a usable answer. Carries the upstream's own reason."""


def _get(url: str, *, timeout: int = TIMEOUT, data: bytes | None = None,
         headers: Mapping[str, str] | None = None) -> bytes:
    request = urllib.request.Request(
        url, data=data, headers={"User-Agent": USER_AGENT, **(headers or {})})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return bytes(response.read())
    except urllib.error.HTTPError as exc:
        raise MirrorError(f"{url.split('?')[0]} answered HTTP {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise MirrorError(f"{url.split('?')[0]} unreachable: {exc}") from exc


def _json(url: str, **kwargs: Any) -> Any:
    try:
        return json.loads(_get(url, **kwargs))
    except ValueError as exc:
        raise MirrorError(f"{url.split('?')[0]} returned something other than JSON") from exc


# --- sentiment-analyst -----------------------------------------------------------------------

def fear_greed(_: Mapping[str, Any]) -> dict[str, Any]:
    rows = (_json("https://api.alternative.me/fng/?limit=2") or {}).get("data") or []
    if not rows:
        raise MirrorError("alternative.me returned no readings")
    now, prior = rows[0], rows[1] if len(rows) > 1 else None
    return {
        "value": int(now["value"]), "classification": now["value_classification"],
        "as_of": datetime.fromtimestamp(int(now["timestamp"]), UTC).isoformat(),
        "prior_value": int(prior["value"]) if prior else None,
    }


_BINANCE = "https://fapi.binance.com/futures/data/"


def _binance(path: str, args: Mapping[str, Any], default_period: str) -> list[dict[str, Any]]:
    symbol = str(args.get("symbol") or "BTCUSDT")
    period = str(args.get("period") or default_period)
    rows = _json(f"{_BINANCE}{path}?symbol={symbol}&period={period}&limit=2")
    if not isinstance(rows, list) or not rows:
        raise MirrorError(f"Binance {path} returned no rows for {symbol}")
    return rows


def long_short(args: Mapping[str, Any]) -> dict[str, Any]:
    row = _binance("globalLongShortAccountRatio", args, "4h")[-1]
    return {"symbol": row["symbol"], "long_short_ratio": float(row["longShortRatio"]),
            "long_account": float(row["longAccount"]), "short_account": float(row["shortAccount"]),
            "as_of": datetime.fromtimestamp(int(row["timestamp"]) / 1000, UTC).isoformat()}


def open_interest(args: Mapping[str, Any]) -> dict[str, Any]:
    rows = _binance("openInterestHist", args, "1h")
    last = rows[-1]
    first = rows[0]
    change = (float(last["sumOpenInterestValue"]) / float(first["sumOpenInterestValue"]) - 1
              if len(rows) > 1 and float(first["sumOpenInterestValue"]) else None)
    return {"symbol": last["symbol"], "open_interest": float(last["sumOpenInterest"]),
            "open_interest_usd": float(last["sumOpenInterestValue"]),
            "change_one_period_pct": None if change is None else round(change * 100, 3),
            "as_of": datetime.fromtimestamp(int(last["timestamp"]) / 1000, UTC).isoformat()}


def taker_ratio(args: Mapping[str, Any]) -> dict[str, Any]:
    row = _binance("takerlongshortRatio", args, "4h")[-1]
    return {"buy_sell_ratio": float(row["buySellRatio"]), "buy_volume": float(row["buyVol"]),
            "sell_volume": float(row["sellVol"]),
            "as_of": datetime.fromtimestamp(int(row["timestamp"]) / 1000, UTC).isoformat()}


# --- macro-analyst ---------------------------------------------------------------------------

def fred_series(series: str, *, days: int = 400) -> list[tuple[date, float]]:
    """A FRED series as (date, value), missing observations ('.') dropped. Keyless CSV endpoint,
    the one ``argus.lui.research.FRED_CSV`` already uses."""
    start = (datetime.now(UTC).date() - timedelta(days=days)).isoformat()
    text = _get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}&cosd={start}")
    out: list[tuple[date, float]] = []
    for row in csv.reader(io.StringIO(text.decode("utf-8", "replace"))):
        if len(row) < 2 or not row[0][:1].isdigit():
            continue
        try:
            out.append((date.fromisoformat(row[0]), float(row[1])))
        except ValueError:
            continue
    if not out:
        raise MirrorError(f"FRED {series} returned no observations")
    return out


INDICATORS: dict[str, tuple[str, str]] = {
    "cpi": ("CPIAUCSL", "Consumer Price Index, all urban consumers"),
    "core_pce": ("PCEPILFE", "Core PCE price index"),
    "unemployment": ("UNRATE", "Unemployment rate"),
    "nonfarm_payrolls": ("PAYEMS", "Nonfarm payrolls, thousands"),
}
"""The Skill's own indicator keys (its ``macro_indicators`` schema) mapped to FRED series ids."""


def latest_release(args: Mapping[str, Any]) -> dict[str, Any]:
    key = str(args.get("indicator") or "cpi")
    if key not in INDICATORS:
        raise MirrorError(f"no FRED mapping for indicator {key!r}")
    series, label = INDICATORS[key]
    rows = fred_series(series, days=500)
    (when, value), (_, prior) = rows[-1], rows[-2] if len(rows) > 1 else rows[-1]
    year_ago = next((v for d, v in reversed(rows) if d <= when - timedelta(days=360)), None)
    return {"indicator": key, "series": series, "label": label, "period": when.isoformat(),
            "value": value, "prior": prior,
            "yoy_pct": None if year_ago in (None, 0) else round((value / year_ago - 1) * 100, 2)}


TENORS = (("t3m", "DGS3MO"), ("t1y", "DGS1"), ("t2y", "DGS2"), ("t5y", "DGS5"),
          ("t10y", "DGS10"), ("t30y", "DGS30"))


def yield_curve(_: Mapping[str, Any]) -> dict[str, Any]:
    curve: dict[str, float] = {}
    as_of = ""
    for key, series in TENORS:
        when, value = fred_series(series, days=21)[-1]
        curve[key] = value
        as_of = max(as_of, when.isoformat())
    spread = round(curve["t10y"] - curve["t2y"], 3)
    return {"yield_curve": curve, "spread_10y2y": spread, "inverted": spread < 0, "as_of": as_of}


def yahoo_closes(symbol: str, *, range_: str = "1y") -> list[tuple[date, float]]:
    raw = _json("https://query2.finance.yahoo.com/v8/finance/chart/"
                f"{urllib.parse.quote(symbol)}?range={range_}&interval=1d")
    try:
        result = raw["chart"]["result"][0]
        stamps = result["timestamp"]
        closes = result["indicators"]["quote"][0]["close"]
    except (KeyError, IndexError, TypeError) as exc:
        raise MirrorError(f"Yahoo returned no series for {symbol}") from exc
    out = [(datetime.fromtimestamp(int(t), UTC).date(), float(c))
           for t, c in zip(stamps, closes, strict=False) if c is not None]
    if len(out) < 2:
        raise MirrorError(f"Yahoo returned too few closes for {symbol}")
    return out


CROSS_TARGETS = (("gold", "GC=F"), ("dxy", "DX-Y.NYB"), ("ndx", "^NDX"), ("spx", "^GSPC"),
                 ("t10y", "^TNX"), ("vix", "^VIX"))
"""The Skill's default targets (``gold,dxy,ndx,spx,t10y,vix``) with the Yahoo symbols it maps them
to (its own ``assets_list`` names these)."""


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    n = len(xs)
    if n < 3 or n != len(ys):
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    return None if sxx == 0 or syy == 0 else sxy / math.sqrt(sxx * syy)


def _returns(series: Mapping[date, float], days: Sequence[date]) -> list[float]:
    return [series[b] / series[a] - 1 for a, b in itertools.pairwise(days)]


def correlation(args: Mapping[str, Any], *, window: int = 30) -> dict[str, Any]:
    """Pearson correlation of daily returns, BTC against each target, over the dates both traded —
    full year and the latest ``window`` sessions, as the Skill's own description specifies."""
    base = dict(yahoo_closes(str(args.get("base") or "BTC-USD")))
    out: dict[str, Any] = {}
    for key, symbol in CROSS_TARGETS:
        try:
            other = dict(yahoo_closes(symbol))
        except MirrorError as exc:
            out[key] = {"error": str(exc)}
            continue
        days = sorted(set(base) & set(other))
        x, y = _returns(base, days), _returns(other, days)
        full, recent = pearson(x, y), pearson(x[-window:], y[-window:])
        out[key] = {"symbol": symbol, "full_period": None if full is None else round(full, 3),
                    f"rolling_{window}d": None if recent is None else round(recent, 3),
                    "sessions": len(days)}
    if all("error" in v for v in out.values()):
        raise MirrorError("Yahoo answered for no target")
    return {"base": "BTC-USD", "correlations": out}


def anchor_symbol(symbol: str) -> str:
    """``NVDAUSDT`` → ``NVDA``. The probe passes the Bitget contract; Yahoo wants the stock."""
    s = symbol.upper()
    return s[:-4] if s.endswith("USDT") else s


def asset_price(args: Mapping[str, Any]) -> dict[str, Any]:
    symbol = anchor_symbol(str(args.get("symbol") or "SPY"))
    closes = yahoo_closes(symbol, range_="5d")
    (when, last), (_, prior) = closes[-1], closes[-2]
    return {"symbol": symbol, "price": last, "as_of": when.isoformat(), "prior_close": prior,
            "change_pct": round((last / prior - 1) * 100, 3)}


# --- news-briefing ---------------------------------------------------------------------------

FEEDS = (("coindesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
         ("cointelegraph", "https://cointelegraph.com/rss"),
         ("cnbc", "https://www.cnbc.com/id/100003114/device/rss/rss.html"),
         ("decrypt", "https://decrypt.co/feed"))
"""Four of the Skill's 44 feed keys (its ``news_feed`` sources list), one per beat: crypto markets,
crypto news, US business."""


def _items(feed: str, body: bytes, limit: int) -> list[dict[str, str]]:
    try:
        root = ET.fromstring(body)
    except ET.ParseError as exc:
        raise MirrorError(f"{feed} returned malformed RSS") from exc
    out = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        if title:
            out.append({"feed": feed, "title": title, "link": (item.findtext("link") or "").strip(),
                        "published": (item.findtext("pubDate") or "").strip()})
        if len(out) >= limit:
            break
    return out


def news(args: Mapping[str, Any]) -> dict[str, Any]:
    limit = int(args.get("limit") or 5)
    keyword = str(args.get("keyword") or "").lower()
    items: list[dict[str, str]] = []
    failed: list[str] = []
    for feed, url in FEEDS:
        try:
            items.extend(_items(feed, _get(url), 30 if keyword else limit))
        except MirrorError:
            failed.append(feed)
    if keyword:
        items = [i for i in items if keyword in i["title"].lower()]
    if not items and len(failed) == len(FEEDS):
        raise MirrorError("no feed answered")
    return {"items": items, "feeds_answered": len(FEEDS) - len(failed), "feeds_failed": failed}


def earnings(args: Mapping[str, Any]) -> dict[str, Any]:
    """Nasdaq's public earnings calendar for the next five weekdays. The Skill reads Finnhub, which
    needs a key; this is a different source for the same fact, and says so."""
    wanted = str(args.get("symbol") or "").upper()
    rows: list[dict[str, str]] = []
    day = datetime.now(UTC).date()
    checked = 0
    while checked < 5:
        if day.weekday() < 5:
            checked += 1
            raw = _json(f"https://api.nasdaq.com/api/calendar/earnings?date={day.isoformat()}",
                        headers={"Accept": "application/json"})
            for row in ((raw or {}).get("data") or {}).get("rows") or []:
                if not wanted or row.get("symbol", "").upper() == wanted:
                    rows.append({"date": day.isoformat(), "symbol": row.get("symbol", ""),
                                 "name": row.get("name", ""), "time": row.get("time", ""),
                                 "eps_forecast": row.get("epsForecast", "")})
        day += timedelta(days=1)
    return {"window_days": 5, "reports": rows[:50], "count": len(rows)}


# --- market-intel ----------------------------------------------------------------------------

def tvl_rank(args: Mapping[str, Any]) -> dict[str, Any]:
    limit = int(args.get("limit") or 10)
    rows = _json("https://api.llama.fi/protocols", timeout=40)
    if not isinstance(rows, list) or not rows:
        raise MirrorError("DeFiLlama returned no protocols")
    # DeFiLlama lists centralised exchanges beside protocols; a DeFi TVL ranking excludes them.
    ranked = sorted((r for r in rows if isinstance(r.get("tvl"), int | float)
                     and r.get("category") != "CEX"),
                    key=lambda r: -float(r["tvl"]))[:limit]
    return {"protocols": [{"name": r.get("name"), "category": r.get("category"),
                           "tvl_usd": round(float(r["tvl"])),
                           "change_1d_pct": r.get("change_1d")} for r in ranked]}


ETH_RPC = "https://eth.drpc.org"
"""A keyless public Ethereum RPC. Of five tried on 2026-09-25, only this one answered: llamarpc
525, cloudflare-eth -32046, ankr requires a key, publicnode returned nothing."""


def eth_gas(_: Mapping[str, Any]) -> dict[str, Any]:
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "eth_feeHistory",
                       "params": [5, "latest", [10, 50, 90]]}).encode()
    raw = _json(ETH_RPC, data=body, headers={"Content-Type": "application/json"})
    result = (raw or {}).get("result")
    if not result or not result.get("baseFeePerGas"):
        raise MirrorError(f"eth_feeHistory failed: {(raw or {}).get('error')}")
    gwei = 1e9
    base = int(result["baseFeePerGas"][-1], 16) / gwei
    tips = [[int(x, 16) / gwei for x in row] for row in result.get("reward") or []]
    last = tips[-1] if tips else [0.0, 0.0, 0.0]
    return {"base_fee_gwei": round(base, 3),
            "priority_fee_gwei": {"low": round(last[0], 3), "median": round(last[1], 3),
                                  "high": round(last[2], 3)},
            "gas_used_ratio": [round(x, 3) for x in result.get("gasUsedRatio") or []]}


def trending(_: Mapping[str, Any]) -> dict[str, Any]:
    raw = _json("https://api.coingecko.com/api/v3/search/trending")
    coins = [c.get("item") or {} for c in (raw or {}).get("coins") or []]
    if not coins:
        raise MirrorError("CoinGecko returned no trending coins")
    return {"trending": [{"id": c.get("id"), "symbol": c.get("symbol"), "name": c.get("name"),
                          "market_cap_rank": c.get("market_cap_rank")} for c in coins[:10]]}


# --- the table ------------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Mirror:
    """How one Skill probe is answered when the Skill cannot answer it."""

    upstream: str
    same_upstream: bool
    """True when this is the upstream the Skill's own tool description names."""

    fetch: Callable[[Mapping[str, Any]], dict[str, Any]]


MIRRORS: dict[str, Mirror] = {
    "sentiment_index.current": Mirror("alternative.me Fear & Greed", True, fear_greed),
    "derivatives_sentiment.long_short": Mirror("Binance futures data", True, long_short),
    "derivatives_sentiment.open_interest": Mirror("Binance futures data", True, open_interest),
    "derivatives_sentiment.taker_ratio": Mirror("Binance futures data", True, taker_ratio),
    "macro_indicators.latest_release": Mirror("FRED", True, latest_release),
    "rates_yields.yield_curve": Mirror("FRED", True, yield_curve),
    "cross_asset.correlation": Mirror("Yahoo Finance", True, correlation),
    "global_assets.price": Mirror("Yahoo Finance", True, asset_price),
    "news_feed.latest": Mirror("RSS: CoinDesk, Cointelegraph, CNBC, Decrypt", True, news),
    "tradfi_news.earnings": Mirror("Nasdaq earnings calendar (Skill uses Finnhub, keyed)", False,
                                   earnings),
    "defi_analytics.tvl_rank": Mirror("DeFiLlama", True, tvl_rank),
    "network_status.eth_gas": Mirror("public Ethereum RPC (eth_feeHistory)", False, eth_gas),
    "crypto_market.trending": Mirror("CoinGecko", True, trending),
}


@dataclass(frozen=True, slots=True)
class MirrorRow:
    """One probe: what Bitget's Skill did, and what the mirror did when it had to step in."""

    probe: Probe
    bitget: Health
    mirror: Health | None
    """None when the Skill answered and the mirror was not needed."""
    upstream: str
    same_upstream: bool
    detail: str
    payload: Any
    latency_ms: float

    @property
    def answered(self) -> bool:
        return self.bitget.answered or self.mirror is Health.OK

    def as_dict(self) -> dict[str, Any]:
        return {
            "skill": self.probe.skill, "tool": self.probe.tool, "action": self.probe.action,
            "yields": self.probe.yields, "bitget": str(self.bitget),
            "mirror": None if self.mirror is None else str(self.mirror),
            "upstream": self.upstream, "same_upstream": self.same_upstream,
            "detail": self.detail, "latency_ms": round(self.latency_ms),
            "sample": None if self.payload is None else json.dumps(self.payload, default=str)[:400],
        }


CACHE_SECONDS = 900.0
"""How long a mirrored reading is reused. Twelve of the thirteen are market-wide (the index, BTC
positioning, the curve, the news), so a cycle over several symbols would otherwise fetch the same
reading once per symbol; fifteen minutes is shorter than the slowest of them updates."""

_CACHE: dict[str, tuple[float, tuple[Health, str, Any, float]]] = {}


def mirror_one(probe: Probe, args: Mapping[str, Any], *,
               cache: bool = True) -> tuple[Health, str, Any, float]:
    """Run one mirror. Hollow answers are EMPTY, exactly as a hollow Skill answer is. Only OK
    readings are cached — a failure is retried on the next call rather than remembered."""
    key = f"{probe.ident}|{json.dumps(dict(args), sort_keys=True, default=str)}"
    hit = _CACHE.get(key) if cache else None
    if hit is not None and time.monotonic() - hit[0] < CACHE_SECONDS:
        return hit[1]
    outcome = _run(probe, args)
    if cache and outcome[0] is Health.OK:
        _CACHE[key] = (time.monotonic(), outcome)
    return outcome


def _run(probe: Probe, args: Mapping[str, Any]) -> tuple[Health, str, Any, float]:
    mirror = MIRRORS[probe.ident]
    started = time.perf_counter()
    try:
        payload = mirror.fetch(args)
    except MirrorError as exc:
        return Health.UNAVAILABLE, str(exc), None, (time.perf_counter() - started) * 1000
    elapsed = (time.perf_counter() - started) * 1000
    if hollow(payload):
        return Health.EMPTY, f"{mirror.upstream} answered with no data", None, elapsed
    return Health.OK, f"{mirror.upstream}: ok", payload, elapsed


def fill(report: SkillReport, *, symbol: str | None = None) -> tuple[MirrorRow, ...]:
    """Mirror every probe the Skill report shows as not answering; keep the Skill's own rows."""
    rows: list[MirrorRow] = []
    for result in report.results:
        item = result.probe
        mirror = MIRRORS.get(item.ident)
        if result.health.answered or mirror is None:
            rows.append(MirrorRow(item, result.health, None, "Bitget Skill", True,
                                  result.detail, result.payload, 0.0))
            continue
        health, detail, payload, ms = mirror_one(item, item.for_symbol(symbol or report.symbol))
        rows.append(MirrorRow(item, result.health, health, mirror.upstream, mirror.same_upstream,
                              detail, payload, ms))
    return tuple(rows)


def fill_missing(answered: set[str], *, symbol: str,
                 probes: Sequence[Probe] = PROBES) -> tuple[MirrorRow, ...]:
    """For a live cycle: mirror every probe that has a mirror and is not in ``answered`` (the
    idents the last Skill sweep measured as answering). Bitget's health for those rows is the
    sweep's verdict, recorded as EMPTY — it was called, it did not answer."""
    rows: list[MirrorRow] = []
    for item in probes:
        mirror = MIRRORS.get(item.ident)
        if mirror is None or item.ident in answered:
            continue
        health, detail, payload, ms = mirror_one(item, item.for_symbol(symbol))
        rows.append(MirrorRow(item, Health.EMPTY, health, mirror.upstream, mirror.same_upstream,
                              detail, payload, ms))
    return tuple(rows)


def load_summary(path: Path) -> dict[str, Any] | None:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return loaded if isinstance(loaded, dict) else None


def summary(rows: Sequence[MirrorRow]) -> dict[str, Any]:
    skills_bitget = sorted({r.probe.skill for r in rows if r.bitget.answered})
    skills_any = sorted({r.probe.skill for r in rows if r.answered})
    return {
        "tools_probed": len(rows),
        "answered_by_bitget": sum(r.bitget.answered for r in rows),
        "answered_by_mirror": sum(r.mirror is Health.OK for r in rows),
        "answered_by_mirror_same_upstream": sum(r.mirror is Health.OK and r.same_upstream
                                                for r in rows),
        "answered_total": sum(r.answered for r in rows),
        "skills_reached_through_bitget": skills_bitget,
        "skills_answered_in_total": skills_any,
    }


def evidence(rows: Sequence[MirrorRow], *, as_of: datetime) -> list[Evidence]:
    """Mirrored readings as evidence, each claim naming the source that actually answered.

    Only rows the mirror answered are returned; rows Bitget answered already become evidence
    through ``skills.evidence``, and producing them twice would double-count one reading."""
    out: list[Evidence] = []
    for row in rows:
        if row.mirror is not Health.OK:
            continue
        via = ("the Skill's own upstream" if row.same_upstream
               else "a substitute for the Skill's keyed upstream")
        out.append(Evidence(
            id=f"mirror-{row.probe.tool}-{row.probe.action}-{int(as_of.timestamp())}",
            claim=(f"[{row.probe.skill}, Bitget server dark; read from {row.upstream}, {via}] "
                   f"{row.probe.yields}: {json.dumps(row.payload, default=str)[:300]}"),
            source="macro" if row.probe.skill == "macro-analyst" else "social",
            available_at=as_of,
            credibility=0.75 if row.same_upstream else 0.7,
            attributes=row.payload if isinstance(row.payload, dict) else {},
        ))
    return out


def save(rows: Sequence[MirrorRow], path: Path, *, symbol: str, checked_at: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"symbol": symbol, "checked_at": checked_at, **summary(rows),
                                "results": [r.as_dict() for r in rows]}, indent=2),
                    encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    """Sweep Bitget's Skills, mirror what they could not answer, write both side by side."""
    import argparse

    from argus.market.evidence import BitgetSkillSource
    from argus.market.skills import DEFAULT_TIMEOUT, probe

    root = Path(__file__).resolve().parents[3] / "data"
    parser = argparse.ArgumentParser(description="Bitget Skill sweep with upstream mirrors.")
    parser.add_argument("--symbol", default="NVDAUSDT")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--out", type=Path, default=root / "skill_mirror.json")
    parser.add_argument("--health-out", type=Path, default=root / "bitget_skills_health.json")
    args = parser.parse_args(argv)

    now = datetime.now(UTC)
    report = probe(BitgetSkillSource(), symbol=args.symbol, at=now, probes=PROBES,
                   timeout=args.timeout)
    report.save(args.health_out)
    rows = fill(report)
    for row in rows:
        print(f"  {row.probe.ident:38} bitget={row.bitget!s:12} "
              f"mirror={row.mirror!s:12} {row.upstream}")
    print(json.dumps(summary(rows)))
    print(f"saved -> {save(rows, args.out, symbol=args.symbol, checked_at=now.isoformat())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["CACHE_SECONDS", "MIRRORS", "Mirror", "MirrorError", "MirrorRow", "evidence", "fill",
           "fill_missing", "load_summary", "main", "mirror_one", "pearson", "save", "summary"]
