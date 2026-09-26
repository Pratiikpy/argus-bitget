"""Run Vibe-Trading's real ``get_fundamentals`` tool on a frozen SEC snapshot, in its own process.

Not imported by ARGUS. Vibe-Trading (HKUDS/Vibe-Trading, MIT) is run from its own clone at
``research/repos-t2/hkuds~vibe-trading`` (commit a65427a3d587e43cd5c7541948b36524e8fd2f12), with
``agent/`` on ``sys.path`` exactly as its own agent runs it. Nothing of theirs is edited.

Two things are recorded per ticker and PIT mode, both from their code:

* ``panel`` — what their agent-facing tool ``GetFundamentalsTool.execute`` returns
  (``src/tools/get_fundamentals_tool.py``): a daily panel, one value per date, "PIT-safe ...
  aligned by filed date". Stored as change points to keep the artefact small.
* ``series`` — their loader's own per-period extraction ``_extract_concept_series``
  (``backtest/loaders/fundamentals_loader.py:48-134``), the sparse ``period_end/filed/value``
  table the panel is forward-filled from. The tool never exposes the period a panel value belongs
  to; this is used only to label it, and the label is cross-checked against the panel.

Only the transport is replaced: ``backtest.loaders.sec_edgar_client.get_company_facts`` and
``cik_for`` serve the snapshot. ``--live`` leaves both alone and fetches from SEC (parity check).
Their opt-in loader cache stays off (its default).

Usage::

    python vibe_trading_runner.py --clone DIR --snapshot DIR --tickers NVDA --out out.json [--live]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any

FIELDS = ("revenue", "net_income")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clone", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--tickers", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--start", default="2017-01-01")
    parser.add_argument("--end", default="2026-09-25")
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    warnings.simplefilter("ignore")

    sys.path.insert(0, str(Path(args.clone) / "agent"))
    from backtest.loaders import _fundamental_schema as schema
    from backtest.loaders import fundamentals_loader as loader
    from backtest.loaders import sec_edgar_client as client
    from src.tools.get_fundamentals_tool import GetFundamentalsTool

    snapshot = Path(args.snapshot)
    tickers_map = json.loads((snapshot / "company_tickers.json").read_text(encoding="utf-8"))
    cik_by_ticker = {str(v["ticker"]).upper(): str(v["cik_str"]).zfill(10)
                     for v in tickers_map.values()}
    # One CIK can carry several tickers (Alphabet's GOOGL, GOOG and others); the one the snapshot
    # holds a file for is the one to serve.
    ticker_by_cik = {cik: t for t, cik in cik_by_ticker.items()
                     if (snapshot / f"{t}.companyfacts.json").exists()}

    def snapshot_facts(cik: str | int) -> dict[str, Any]:
        ticker = ticker_by_cik[str(cik).zfill(10)]
        loaded: dict[str, Any] = json.loads(
            (snapshot / f"{ticker}.companyfacts.json").read_text(encoding="utf-8"))
        return loaded

    if not args.live:
        client.get_company_facts = snapshot_facts  # type: ignore[assignment]
        client.cik_for = lambda t: cik_by_ticker.get(str(t).upper().removesuffix(".US"))  # type: ignore[assignment]

    out: dict[str, Any] = {"runner": "vibe_trading_runner", "live": bool(args.live),
                           "clone": Path(args.clone).resolve().name, "tickers": {}}
    for ticker in [t.strip().upper() for t in args.tickers.split(",") if t.strip()]:
        per: dict[str, Any] = {}
        for pit in (True, False):
            mode = "pit" if pit else "research"
            started = time.perf_counter()
            envelope = json.loads(GetFundamentalsTool().execute(
                symbols=[ticker], fields=list(FIELDS), start=args.start, end=args.end,
                freq="quarterly", pit=pit,
            ))
            elapsed = time.perf_counter() - started
            if not envelope.get("ok"):
                per[mode] = {"error": envelope.get("error"), "elapsed_s": elapsed}
                continue
            panel: dict[str, list[list[Any]]] = {}
            for field, records in envelope["data"].items():
                points: list[list[Any]] = []
                previous: Any = object()
                for record in records:
                    value = record.get(ticker)
                    if value != previous:
                        points.append([str(record["date"])[:10], value])
                        previous = value
                panel[field] = points
            facts = client.get_company_facts(cik_by_ticker[ticker])
            series: dict[str, list[list[Any]]] = {}
            for field in FIELDS:
                sparse = loader._extract_concept_series(
                    facts, list(schema.SEC_CONCEPT_MAP[field]), "quarterly", pit=pit)
                series[field] = [
                    [str(r.period_end.date()), str(r.filed.date()), float(r.value)]
                    for r in sparse.itertuples()
                ]
            per[mode] = {"panel": panel, "series": series, "elapsed_s": elapsed}
        out["tickers"][ticker] = per
    Path(args.out).write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
