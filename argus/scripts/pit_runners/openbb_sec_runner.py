"""Run OpenBB ODP's real SEC income-statement fetcher on a frozen SEC snapshot, in its own process.

Not imported by ARGUS. OpenBB ODP is AGPL-3.0, so it is installed in a separate virtual
environment (``research/_venvs/pit-rivals``, editable from the local clone of
``OpenBB-finance/OpenBB`` at commit 3e071fcc2cd9f891cac6040ae60296dba76dab46) and driven here as
a subprocess by :mod:`argus.eval.pit_rivals`. This file contains no OpenBB code.

What runs is OpenBB's own ``SecIncomeStatementFetcher.fetch_data`` — ``transform_query``,
``aextract_data`` → ``get_standardized_financials`` → ``resolve_company_facts``, then
``transform_data`` — exactly as ``obb.equity.fundamental.income(provider="sec")`` and the
openbb-mcp-server ``equity_fundamental_income`` tool call it. Only the HTTP transport is replaced:
``openbb_core.provider.utils.helpers.amake_request`` is pointed at the snapshot's
``companyfacts`` file and ``openbb_sec.utils.helpers.symbol_map`` at the snapshot's ticker map,
so every rival and ARGUS read byte-identical SEC data. ``--live`` skips both patches and fetches
from SEC, which is how parity with the snapshot run is established.

Usage::

    python openbb_sec_runner.py --snapshot DIR --tickers NVDA,AAPL --out out.json [--live]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any

FIELDS = (
    "period_ending", "fiscal_period", "fiscal_year", "total_revenue", "net_income",
    "net_income_to_common", "diluted_eps", "weighted_ave_diluted_shares_os",
)
SOURCE_FIELDS = ("total_revenue", "net_income", "diluted_eps")


def _patch_transport(snapshot: Path) -> None:
    import openbb_core.provider.utils.helpers as helpers
    import openbb_sec.utils.helpers as sec_helpers

    tickers = json.loads((snapshot / "company_tickers.json").read_text(encoding="utf-8"))
    cik_by_ticker = {str(v["ticker"]).upper(): str(v["cik_str"]).zfill(10)
                     for v in tickers.values()}
    ticker_by_cik = {cik: t for t, cik in cik_by_ticker.items()}

    async def fake_request(url: str, **_kwargs: Any) -> Any:
        if "/api/xbrl/companyfacts/CIK" not in url:
            raise RuntimeError(f"snapshot runner refused an unexpected request: {url}")
        cik = url.rsplit("CIK", 1)[1].split(".json")[0].zfill(10)
        ticker = ticker_by_cik[cik]
        return json.loads((snapshot / f"{ticker}.companyfacts.json").read_text(encoding="utf-8"))

    async def fake_symbol_map(symbol: str, use_cache: bool = True) -> str:
        return cik_by_ticker.get(symbol.upper(), "")

    helpers.amake_request = fake_request  # type: ignore[assignment]
    sec_helpers.symbol_map = fake_symbol_map  # type: ignore[assignment]


async def _run_one(ticker: str, pit_mode: bool) -> dict[str, Any]:
    from openbb_sec.models.income_statement import SecIncomeStatementFetcher

    started = time.perf_counter()
    result = await SecIncomeStatementFetcher.fetch_data(
        {"symbol": ticker, "period": "quarterly", "pit_mode": pit_mode, "use_cache": False}, {}
    )
    elapsed = time.perf_counter() - started
    rows = []
    for record in result.result:  # type: ignore[union-attr]
        dumped = record.model_dump()
        row = {k: dumped.get(k) for k in FIELDS}
        row["period_ending"] = str(row["period_ending"])
        rows.append(row)
    fields_meta = (result.metadata or {}).get("fields", {})  # type: ignore[union-attr]
    sources = {f: fields_meta.get(f, {}).get("sources", {}) for f in SOURCE_FIELDS}
    return {"rows": rows, "sources": sources, "elapsed_s": elapsed}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--tickers", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--modes", default="default,pit")
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    warnings.simplefilter("ignore")

    snapshot = Path(args.snapshot)
    if not args.live:
        _patch_transport(snapshot)
    import openbb_core
    import openbb_sec

    out: dict[str, Any] = {
        "runner": "openbb_sec_runner",
        "live": bool(args.live),
        # Package directory names only: an absolute install path would publish the machine's
        # user directory inside a committed artefact.
        "openbb_core_dir": Path(openbb_core.__file__).parent.name,
        "openbb_sec_dir": Path(openbb_sec.__file__).parent.name,
        "python": sys.version,
        "tickers": {},
    }
    for ticker in [t.strip().upper() for t in args.tickers.split(",") if t.strip()]:
        per: dict[str, Any] = {}
        for mode in args.modes.split(","):
            try:
                per[mode] = asyncio.run(_run_one(ticker, pit_mode=(mode == "pit")))
            except Exception as exc:  # a rival's failure is a result to record
                per[mode] = {"error": f"{type(exc).__name__}: {exc}"}
        out["tickers"][ticker] = per
    Path(args.out).write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
