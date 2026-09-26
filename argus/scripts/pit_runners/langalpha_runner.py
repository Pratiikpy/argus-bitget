"""Call LangAlpha's real yfinance fundamentals MCP tool, live, in its own process.

Not imported by ARGUS. LangAlpha (ginlix-ai/LangAlpha, Apache-2.0) is run from its clone at
``research/repos-themed/ginlix-ai~LangAlpha`` (commit 12472ac8d7c3a79ae86affd479ef5e839ab651bd).
The function called is ``get_income_statement`` in
``plugins/yfinance/yf_fundamentals_mcp_server.py`` — the tool its agent reaches for "income
statement" on the keyless path (its FMP path needs a paid key). It has no date, period or as-of
parameter; it returns whatever Yahoo reports now. There is no snapshot mode because the upstream is
Yahoo, not SEC: the response is saved to disk the moment it arrives and every later score is
computed from that file.

Usage::

    python langalpha_runner.py --clone DIR --tickers NVDA,AAPL --out out.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clone", required=True)
    parser.add_argument("--tickers", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    warnings.simplefilter("ignore")

    clone = Path(args.clone)
    sys.path.insert(0, str(clone))
    import importlib

    module = importlib.import_module("plugins.yfinance.yf_fundamentals_mcp_server")
    tool = module.get_income_statement
    fn = getattr(tool, "fn", tool)

    out: dict[str, Any] = {"runner": "langalpha_runner", "clone": clone.resolve().name,
                           "fetched_at": datetime.now(UTC).isoformat(), "tickers": {}}
    import yfinance

    out["yfinance_version"] = getattr(yfinance, "__version__", "unknown")
    out_path = Path(args.out)
    for ticker in [t.strip().upper() for t in args.tickers.split(",") if t.strip()]:
        started = time.perf_counter()
        try:
            response = fn(ticker, quarterly=True)
        except Exception as exc:  # a rival's failure is a result to record
            response = {"error": f"{type(exc).__name__}: {exc}"}
        out["tickers"][ticker] = {"response": response,
                                  "elapsed_s": time.perf_counter() - started}
        # Write-through: a live answer is saved before the next request is made.
        out_path.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
        time.sleep(0.5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
