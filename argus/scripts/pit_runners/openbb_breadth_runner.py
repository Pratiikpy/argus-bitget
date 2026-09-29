"""Measure, on real live calls, which OpenBB Platform (ODP) providers answer WITHOUT any API
key, for a fixed list of US tickers.

Runs only inside the ``research/_venvs/pit-rivals`` venv, which has the keyless OpenBB provider
and extension packages editable-installed from the local OpenBB-upstream clone (AGPL-3.0; never
imported from ARGUS). This file contains no OpenBB code of its own — it only calls the public
``obb.*`` Python interface that OpenBB's own package build generates, exactly as a real user of
the SDK would.

Endpoint -> provider mapping below was derived by (1) reading each provider's
``credentials=`` kwarg in ``openbb_<provider>/__init__.py`` to exclude keyed providers, and
(2) enumerating every ``obb.equity.*`` / ``obb.news.*`` / ``obb.derivatives.*`` / ``obb.etf.*``
command's docstring ("Default priority: ...") to find which of the keyless providers actually
back it, then inspecting each command's signature to classify it as per-symbol (has a leading
``symbol`` parameter) or market-wide.

Usage::

    python openbb_breadth_runner.py --tickers NVDA,TSLA,... --out out.json --timeout 30
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# scripts/pit_runners/ -> scripts/ -> argus/ -> the workspace, where research/repos holds the
# clones.
OPENBB_CLONE = Path(__file__).resolve().parents[3] / "research" / "repos" / "OpenBB-upstream"

# provider name -> credentials field found in openbb_<provider>/__init__.py (None = no
# credentials kwarg at all / literal None = keyless).
ALL_PROVIDERS_CREDENTIALS: dict[str, Any] = {
    "alpha_vantage": ["api_key"],
    "benzinga": ["api_key"],
    "biztoc": ["api_key"],
    "bls": ["api_key"],
    "cboe": None,
    "cftc": ["app_token (optional; throttled without it)"],
    "congress_gov": ["api_key"],
    "deribit": None,
    "ecb": None,
    "econdb": ["api_key (optional; falls back to a temporary token if unset)"],
    "eia": ["api_key (optional; only required for functions beyond the Weekly Petroleum "
            "Status Report)"],
    "famafrench": None,
    "federal_reserve": None,
    "finra": None,
    "finviz": None,
    "fmp": ["api_key"],
    "fred": ["api_key"],
    "government_us": None,
    "imf": None,
    "intrinio": ["api_key"],
    "multpl": None,
    "nasdaq": ["api_key"],
    "oecd": None,
    "sec": None,
    "seeking_alpha": None,
    "stockgrid": None,
    "tiingo": ["token"],
    "tmx": None,
    "tradier": ["account_type + api_key/token (has a shared demo token but still a required "
                "credential)"],
    "tradingeconomics": ["api_key"],
    "wsj": None,
    "yfinance": None,
}

# Providers actually used for the per-symbol US-equity/ETF breadth test. deribit (crypto
# derivatives only), famafrench/federal_reserve/government_us/imf/oecd/ecb/multpl/cftc/eia
# (macro/market-wide, no per-symbol equity endpoints) are keyless but out of scope for this
# ticker-breadth measurement and are not installed as extra weight.
KEYLESS_TESTED_PROVIDERS = [
    "cboe", "finra", "finviz", "sec", "tmx", "wsj", "yfinance", "stockgrid", "seeking_alpha",
]

KEYLESS_NOT_TESTED_NO_PER_SYMBOL_ENDPOINT = [
    "deribit", "famafrench", "federal_reserve", "government_us", "imf", "oecd", "ecb", "multpl",
    "cftc", "eia",
]

EXCLUDED_KEYED = {
    k: v for k, v in ALL_PROVIDERS_CREDENTIALS.items() if v is not None
    # these three are "optional key" nuances, listed separately
    and k not in ("cftc", "econdb", "eia")
} | {
    "cftc": ALL_PROVIDERS_CREDENTIALS["cftc"],
    "econdb": ALL_PROVIDERS_CREDENTIALS["econdb"],
    "eia": ALL_PROVIDERS_CREDENTIALS["eia"],
}

# path -> list of keyless providers that back it (subset of KEYLESS_TESTED_PROVIDERS),
# established live against obb's own docstrings ("Default priority: ...").
# path -> extra fixed kwargs a real caller must supply beyond symbol/provider (required params
# with no usable default).
PER_SYMBOL_EXTRA_KWARGS: dict[str, dict[str, Any]] = {
    # openbb_sec's own docstring: "Defaults to 'Revenues' ... AAPL, MSFT, GOOG, BRK-A currently
    # report revenue as 'RevenueFromContractWithCustomerExcludingAssessedTax'" — the code default
    # is actually '' (fails validation), so pass the field the docstring recommends.
    "equity.compare.company_facts": {"fact": "RevenueFromContractWithCustomerExcludingAssessedTax"},
}

PER_SYMBOL_ENDPOINTS: dict[str, list[str]] = {
    "equity.compare.company_facts": ["sec"],
    "equity.darkpool.otc": ["finra"],
    "equity.estimates.consensus": ["tmx", "yfinance"],
    "equity.estimates.forward_eps": ["seeking_alpha"],
    "equity.estimates.forward_sales": ["seeking_alpha"],
    "equity.estimates.price_target": ["finviz"],
    "equity.fundamental.balance": ["sec", "yfinance"],
    "equity.fundamental.balance_growth": ["sec"],
    "equity.fundamental.cash": ["sec", "yfinance"],
    "equity.fundamental.cash_growth": ["sec"],
    "equity.fundamental.dividends": ["tmx", "yfinance"],
    "equity.fundamental.filings": ["sec", "tmx"],
    "equity.fundamental.income": ["sec", "yfinance"],
    "equity.fundamental.income_growth": ["sec"],
    "equity.fundamental.management": ["yfinance"],
    "equity.fundamental.management_discussion_analysis": ["sec"],
    "equity.fundamental.metrics": ["finviz", "yfinance"],
    "equity.ownership.form_13f": ["sec"],
    "equity.ownership.insider_trading": ["sec", "tmx"],
    "equity.ownership.share_statistics": ["yfinance"],
    "equity.price.historical": ["cboe", "tmx", "yfinance"],
    "equity.price.performance": ["finviz"],
    "equity.price.quote": ["cboe", "tmx", "yfinance"],
    "equity.profile": ["finviz", "tmx", "yfinance"],
    "equity.shorts.fails_to_deliver": ["sec"],
    "equity.shorts.short_interest": ["finra"],
    "equity.shorts.short_volume": ["stockgrid"],
    "news.company": ["tmx", "yfinance"],
    "derivatives.futures.curve": ["cboe", "yfinance"],
    "derivatives.options.chains": ["cboe", "tmx", "yfinance"],
    "etf.countries": ["tmx"],
    "etf.historical": ["cboe", "tmx", "yfinance"],
    "etf.holdings": ["tmx"],
    "etf.info": ["tmx", "yfinance"],
    "etf.nport_disclosure": ["sec"],
    "etf.price_performance": ["finviz"],
    "etf.sectors": ["tmx"],
}

# path -> (list of keyless providers, kwargs to call with). Called ONCE per provider, not once
# per ticker — these are market-wide / discovery / search endpoints, not per-symbol lookups.
MARKET_WIDE_ENDPOINTS: dict[str, tuple[list[str], dict[str, Any]]] = {
    "equity.discovery.active": (["yfinance"], {}),
    "equity.discovery.aggressive_small_caps": (["yfinance"], {}),
    "equity.discovery.gainers": (["tmx", "yfinance"], {}),
    "equity.discovery.growth_tech": (["yfinance"], {}),
    "equity.discovery.latest_financial_reports": (["sec"], {}),
    "equity.discovery.losers": (["yfinance"], {}),
    "equity.discovery.undervalued_growth": (["yfinance"], {}),
    "equity.discovery.undervalued_large_caps": (["yfinance"], {}),
    "equity.screener": (["finviz", "yfinance"], {}),
    "equity.search": (["cboe", "sec", "tmx"], {"query": "Apple"}),
    "equity.compare.groups": (["finviz"], {}),
    "equity.calendar.earnings": (["seeking_alpha", "tmx"], {}),
    "etf.discovery.active": (["wsj"], {}),
    "etf.discovery.gainers": (["wsj"], {}),
    "etf.discovery.losers": (["wsj"], {}),
    "etf.search": (["tmx"], {"query": "QQQ"}),
}


def _get_obj(path: str):
    from openbb import obb

    obj = obb
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def _call_with_timeout(fn, kwargs: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(fn, **kwargs)
        try:
            result = future.result(timeout=timeout_s)
        except concurrent.futures.TimeoutError:
            elapsed = time.perf_counter() - started
            return {
                "answered": False, "rows": 0, "latency_s": round(elapsed, 3),
                "error": f"TimeoutError: exceeded {timeout_s}s",
            }
        except Exception as exc:
            elapsed = time.perf_counter() - started
            msg = f"{type(exc).__name__}: {exc}"
            return {
                "answered": False, "rows": 0, "latency_s": round(elapsed, 3),
                "error": msg[:300],
            }
    elapsed = time.perf_counter() - started
    rows = _count_rows(getattr(result, "results", None))
    return {
        "answered": rows >= 1, "rows": rows, "latency_s": round(elapsed, 3), "error": None,
    }


def _count_rows(results: Any) -> int:
    """Row-count that also handles OpenBB's columnar RootModel results.

    Some endpoints (options chains being the clear case) return a single pydantic RootModel
    object per call rather than a list of per-row records — the "rows" live as parallel list
    attributes inside that one object (e.g. ``.contract_symbol`` has one entry per contract).
    ``len(obj)`` on that object raises ``TypeError`` and was silently counted as 0 rows in an
    earlier version of this script, which understated cboe/tmx/yfinance options-chain coverage.
    This counts the dumped root as a list when it is one, else treats a non-empty single record
    as 1 row, matching how a real caller would read the response.
    """
    if results is None:
        return 0
    if isinstance(results, list):
        return len(results)
    try:
        return len(results)
    except TypeError:
        pass
    try:
        dumped = results.model_dump()
    except Exception:
        return 1  # a single, non-list, non-dumpable-but-present result: still an answer
    if isinstance(dumped, list):
        return len(dumped)
    if isinstance(dumped, dict):
        return 1 if dumped else 0
    return 1


def run(tickers: list[str], timeout_s: float) -> dict[str, Any]:
    per_symbol_results: list[dict[str, Any]] = []
    for path, providers in PER_SYMBOL_ENDPOINTS.items():
        obj = _get_obj(path)
        extra = PER_SYMBOL_EXTRA_KWARGS.get(path, {})
        for provider in providers:
            for ticker in tickers:
                call_kwargs = {"symbol": ticker, "provider": provider, **extra}
                outcome = _call_with_timeout(obj, call_kwargs, timeout_s)
                row = {"provider": provider, "endpoint": path, "ticker": ticker, **outcome}
                per_symbol_results.append(row)
                tag = "OK" if row["answered"] else "--"
                print(f"[{tag}] {provider:14s} {path:45s} {ticker:6s} rows={row['rows']:>4} "
                      f"{row['latency_s']:>6.2f}s {row['error'] or ''}", flush=True)

    market_wide_results: list[dict[str, Any]] = []
    for path, (providers, extra_kwargs) in MARKET_WIDE_ENDPOINTS.items():
        obj = _get_obj(path)
        for provider in providers:
            kwargs = {"provider": provider, **extra_kwargs}
            outcome = _call_with_timeout(obj, kwargs, timeout_s)
            row = {"provider": provider, "endpoint": path, **outcome}
            market_wide_results.append(row)
            tag = "OK" if row["answered"] else "--"
            print(f"[{tag}] {provider:14s} {path:45s} {'(market-wide)':6s} rows={row['rows']:>4} "
                  f"{row['latency_s']:>6.2f}s {row['error'] or ''}", flush=True)

    per_symbol_results.extend(_earnings_by_symbol(tickers, timeout_s))
    return {"per_symbol": per_symbol_results, "market_wide": market_wide_results}


EARNINGS_HORIZON_DAYS = 120
"""How far ahead the earnings calendar is asked, so a stock's next report is inside it whichever
day of its quarter the run falls on. Scored per symbol on both sides of the comparison: a stock
counts only when an upcoming report date is listed for it."""


def _earnings_by_symbol(tickers: list[str], timeout_s: float) -> list[dict[str, Any]]:
    """The keyless calendars' upcoming report dates, turned into one row per ticker: answered
    when the ticker is listed with a date from today on. The calendar is market-wide, so it is
    called once per provider over the horizon, and the ticker is looked for in what came back."""
    from datetime import timedelta

    from openbb import obb

    today = datetime.now(UTC).date()
    rows: list[dict[str, Any]] = []
    for provider in ("tmx", "seeking_alpha"):
        started = time.perf_counter()
        error = None
        listed: dict[str, str] = {}
        # Asked a week at a time: TMX's fetcher makes one request per day with a three-second
        # timeout each, and a 120-day window in one call timed out as a whole (2026-09-29).
        failures: list[str] = []
        for offset in range(0, EARNINGS_HORIZON_DAYS, 7):
            start = today + timedelta(days=offset)
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    result = pool.submit(
                        obb.equity.calendar.earnings, provider=provider,
                        start_date=start.isoformat(),
                        end_date=(start + timedelta(days=6)).isoformat(),
                    ).result(timeout=max(timeout_s, 60.0))
            except Exception as exc:
                failures.append(f"{start}: {type(exc).__name__}")
                continue
            for item in getattr(result, "results", None) or []:
                sym = str(getattr(item, "symbol", "") or "").upper()
                when = str(getattr(item, "report_date", "") or "")[:10]
                if sym in tickers and when >= today.isoformat():
                    listed.setdefault(sym, when)
        if failures:
            weeks = len(range(0, EARNINGS_HORIZON_DAYS, 7))
            error = f"{len(failures)} of {weeks} weeks failed: " + "; ".join(failures[:3])
        latency = round(time.perf_counter() - started, 3)
        for ticker in tickers:
            rows.append({"provider": provider, "endpoint": "equity.calendar.earnings",
                         "ticker": ticker, "answered": ticker in listed,
                         "rows": 1 if ticker in listed else 0, "latency_s": latency,
                         "error": error, "report_date": listed.get(ticker)})
        print(f"[cal] {provider:14s} upcoming reports for {sorted(listed)} {error or ''}",
              flush=True)
    return rows


def _pkg_versions() -> dict[str, str]:
    import importlib.metadata as md

    out = {}
    for dist in md.distributions():
        name = dist.metadata.get("Name", "")
        if name and name.lower().startswith("openbb"):
            out[name] = dist.version
    return dict(sorted(out.items()))


def _openbb_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(OPENBB_CLONE), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception as exc:
        return f"UNVERIFIED: {exc}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tickers", default="NVDA,TSLA,AAPL,MSFT,META,GOOGL,AMZN,COIN,MSTR,QQQ,TQQQ,SQQQ")
    parser.add_argument("--out", required=True)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    started_wall = time.time()
    try:
        results = run(tickers, args.timeout)
        run_error = None
    except Exception:
        results = {"per_symbol": [], "market_wide": []}
        run_error = traceback.format_exc()

    out = {
        "as_of_utc": datetime.now(UTC).isoformat(),
        "openbb_commit": _openbb_commit(),
        "python": sys.version,
        "installed_openbb_packages": _pkg_versions(),
        "tickers": tickers,
        "timeout_s": args.timeout,
        "keyless_tested_providers": KEYLESS_TESTED_PROVIDERS,
        "keyless_not_tested_no_per_symbol_endpoint": KEYLESS_NOT_TESTED_NO_PER_SYMBOL_ENDPOINT,
        "excluded_keyed_providers": EXCLUDED_KEYED,
        "wall_time_s": round(time.time() - started_wall, 1),
        "run_error": run_error,
        **results,
    }
    Path(args.out).write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
