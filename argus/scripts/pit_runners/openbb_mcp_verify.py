"""Verify the capability TOML's claim about openbb-mcp-server: that it "serves ODP's own FastAPI
routes as MCP tools" (``openbb_mcp_server/app/app.py:348``, ``from openbb_core.api.rest_api import
app``), so its numbers are exactly the ODP output ``openbb_sec_runner.py`` already scored.

Not imported by ARGUS. Runs in the same venv and against the same monkeypatched transport as
``openbb_sec_runner.py`` (``research/_venvs/pit-rivals``, editable OpenBB-upstream clone), so both
read byte-identical SEC data.

**What actually blocked this before it was run.** ``import openbb`` and
``from openbb_core.api.rest_api import app`` both raised
``AttributeError: '_IncludedRouter' object has no attribute 'path'`` inside
``openbb_core/app/router.py:429`` (``CommandMap.get_command_map``). Root cause, found by
inspection: the venv is ``--system-site-packages`` and inherited ``fastapi==0.139.0`` /
``starlette==1.3.1`` from the global user site-packages (installed there for other tools), while
``openbb_platform/core/pyproject.toml`` pins ``fastapi = "0.136.3"`` (Poetry caret: <0.137). A local
``pip install "fastapi==0.136.3"`` inside the venv (this file's own doing, run once, shadowing the
global copy without touching it — confirmed by pip's own refusal to uninstall the outside-venv
copy) fixed both imports; ``starlette>=0.46.0`` was already satisfied by 1.3.1. This is exactly the
kind of dependency wall Standing Rule "test environments, not production" and "exhaust what is in
reach before naming a blocker" says to open, not report as blocked.

**What this script actually calls.** ``openbb_core.api.rest_api.app`` — literally the FastAPI
``app`` object ``openbb_mcp_server/app/app.py`` imports at that line and wraps as MCP tools
(``create_mcp_server`` -> ``process_fastapi_routes_for_mcp``) — through a ``TestClient`` hitting
``GET /api/v1/equity/fundamental/income?symbol=...&provider=sec&period=quarterly&pit_mode=...``,
the same route ``obb.equity.fundamental.income(provider="sec")`` and the MCP tool both dispatch
through (``openbb_equity/fundamental/fundamental_router.py::income`` ->
``OBBject.from_query`` -> the same ``SecIncomeStatementFetcher.fetch_data``
``openbb_sec_runner.py`` calls directly). Row-for-row equality against ``openbb.json``'s already-
recorded ODP output is the claim being checked, not assumed.

Usage::

    python openbb_mcp_verify.py --snapshot DIR --tickers NVDA,AAPL --odp openbb.json --out out.json
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Any

FIELDS = (
    "period_ending", "fiscal_period", "fiscal_year", "total_revenue", "net_income",
    "net_income_to_common", "diluted_eps", "weighted_ave_diluted_shares_os",
)


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


def _differs(a: Any, b: Any) -> bool:
    """True field difference, not float64-last-bit JSON round-trip noise (the REST route's
    response is re-parsed from serialized JSON; the runner's direct fetcher call is not — a
    diluted_eps of 1.7600000000000007 vs 1.7600000000000002 is the same reported number)."""
    if isinstance(a, int | float) and isinstance(b, int | float):
        return not math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-9)
    return a != b


def _rows_equal(mcp_rows: list[dict[str, Any]], odp_rows: list[dict[str, Any]]) -> dict[str, Any]:
    mcp_by_end = {r["period_ending"]: r for r in mcp_rows}
    odp_by_end = {r["period_ending"]: r for r in odp_rows}
    mismatches = []
    for end, odp_row in odp_by_end.items():
        mcp_row = mcp_by_end.get(end)
        if mcp_row is None:
            mismatches.append({"period_ending": end, "issue": "missing from mcp-route output"})
            continue
        diffs = {f: (odp_row.get(f), mcp_row.get(f)) for f in FIELDS
                 if _differs(odp_row.get(f), mcp_row.get(f))}
        if diffs:
            mismatches.append({"period_ending": end, "diffs": diffs})
    extra = sorted(set(mcp_by_end) - set(odp_by_end))
    return {"odp_rows": len(odp_rows), "mcp_route_rows": len(mcp_rows),
            "identical": not mismatches and not extra, "mismatches": mismatches,
            "mcp_route_only_periods": extra}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--tickers", required=True)
    parser.add_argument("--odp", required=True, help="already-recorded openbb.json (ODP output)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--modes", default="default,pit")
    args = parser.parse_args()
    warnings.simplefilter("ignore")

    snapshot = Path(args.snapshot)
    _patch_transport(snapshot)

    # The literal import openbb_mcp_server/app/app.py performs at its top (line 348 in the clone
    # read for this capability) — proof the fix above makes the MCP server's own module importable,
    # not just the bare FastAPI app.
    import openbb_mcp_server.app.app as mcp_app_module
    from fastapi.testclient import TestClient
    from openbb_core.api.rest_api import app as rest_app

    client = TestClient(rest_app)
    odp = json.loads(Path(args.odp).read_text(encoding="utf-8"))

    out: dict[str, Any] = {
        "runner": "openbb_mcp_verify",
        "mcp_server_module_imports": bool(mcp_app_module),
        "route": "/api/v1/equity/fundamental/income",
        "tickers": {},
    }
    for ticker in [t.strip().upper() for t in args.tickers.split(",") if t.strip()]:
        per: dict[str, Any] = {}
        odp_ticker = odp.get("tickers", {}).get(ticker, {})
        for mode in args.modes.split(","):
            try:
                resp = client.get("/api/v1/equity/fundamental/income", params={
                    "symbol": ticker, "provider": "sec", "period": "quarterly",
                    "pit_mode": "true" if mode == "pit" else "false", "use_cache": "false",
                })
            except Exception as exc:  # the same real failure a rival's own runner would record
                per[mode] = {"route_error": f"{type(exc).__name__}: {exc}",
                            "odp_also_errored": "error" in odp_ticker.get(mode, {}),
                            "odp_error": odp_ticker.get(mode, {}).get("error")}
                continue
            if resp.status_code != 200:
                per[mode] = {"error": f"HTTP {resp.status_code}: {resp.text[:500]}"}
                continue
            results = resp.json().get("results", [])
            mcp_rows = [{k: r.get(k) for k in FIELDS} for r in results]
            odp_rows = odp_ticker.get(mode, {}).get("rows", [])
            per[mode] = _rows_equal(mcp_rows, odp_rows)
            if "error" in odp_ticker.get(mode, {}):
                per[mode]["odp_had_recorded_error_but_route_succeeded"] = True
        out["tickers"][ticker] = per
    Path(args.out).write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    checked = [m for t in out["tickers"].values() for m in t.values()
               if "route_error" not in m and "error" not in m]
    all_identical = bool(checked) and all(m.get("identical") for m in checked)
    print(f"all comparable ticker/mode rows identical to ODP: {all_identical} "
          f"({len(checked)} comparable of {sum(len(t) for t in out['tickers'].values())})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
