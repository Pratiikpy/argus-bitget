"""ARGUS's factor loadings against Riskfolio-Lib's ``loadings_matrix``, same book, same returns.

Riskfolio-Lib (BSD-3, `research/repos-t2/dcajasn~riskfolio-lib`) is the most complete general
factor-model library on this machine. Its ``loadings_matrix``
(`riskfolio/src/ParamsEstimation.py:779`) regresses each asset on the factors after forward
stepwise selection at p < 0.05 (`forward_regression`, line 354). The package itself does not import
here (it needs cvxpy, arch and a compiled extension), so its two functions are lifted **unmodified**
from the file by ``ast`` and executed against numpy, pandas and statsmodels, which is everything
they use. `argus.lui.exposures.ols` is run on the identical return matrix.

Dev-only: pandas and statsmodels are dev dependencies; shipped ARGUS code never imports this.
Run: ``PYTHONPATH=src python -m argus.eval.exposures_comparison``.
"""

from __future__ import annotations

import ast
import itertools
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from argus.lui import exposures as ex

RISKFOLIO = (Path(__file__).resolve().parents[4] / "research" / "repos-t2" / "dcajasn~riskfolio-lib"
             / "riskfolio" / "src" / "ParamsEstimation.py")
OUT = Path(__file__).resolve().parents[3] / "data" / "exposures_comparison.json"
BOOK = {"MSFTUSDT": 0.4, "METAUSDT": 0.3, "GOOGLUSDT": 0.3, "NVDAUSDT": 0.0, "BTCUSDT": 0.0,
        "XAUUSDT": 0.0}
"""The audit's book, plus the names a trade on it would add; each is fitted on its own."""


def _riskfolio_functions() -> dict[str, Any]:
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    tree = ast.parse(RISKFOLIO.read_text(encoding="utf-8"))
    wanted: list[ast.stmt] = [node for node in tree.body if isinstance(node, ast.FunctionDef)
              and node.name in ("forward_regression", "loadings_matrix")]
    namespace: dict[str, Any] = {"np": np, "pd": pd, "sm": sm}
    exec(compile(ast.Module(body=wanted, type_ignores=[]), str(RISKFOLIO), "exec"), namespace)
    return namespace


def main() -> int:
    import pandas as pd

    rows = ex.sector_map()["rows"]
    inputs = ex.fetch_inputs(list(BOOK), rows)
    functions = _riskfolio_functions()
    results: dict[str, Any] = {}
    for symbol in BOOK:
        closes = inputs.closes.get(symbol)
        if closes is None:
            results[symbol] = {"error": inputs.missing.get(symbol, "no closes")}
            continue
        f = inputs.factors
        days = sorted(set(closes) & set(f["SPY"]) & set(f["IWM"]) & set(f["MTUM"])
                      & set(f["BTC"]))[-(ex.WINDOW_DAYS + 1):]
        sub = {k: {d: v[d] for d in days} for k, v in f.items()}
        _, factors = ex.factor_returns(sub)
        y = [closes[b] / closes[a] - 1.0 for a, b in itertools.pairwise(days)]
        fitted = ex.ols(y, [factors[k] for k in ex.FACTORS])
        assert fitted is not None
        coef, ses, r2 = fitted
        x = pd.DataFrame({k: factors[k] for k in ex.FACTORS})
        frame = pd.DataFrame({symbol: y})
        full = functions["loadings_matrix"](x, frame, feature_selection="stepwise",
                                            stepwise="Forward", criterion="pvalue",
                                            threshold=0.05)
        results[symbol] = {
            "n": len(y),
            "argus": {k: round(coef[i + 1], 4) for i, k in enumerate(ex.FACTORS)},
            "argus_t": {k: round(coef[i + 1] / ses[i + 1], 2) for i, k in enumerate(ex.FACTORS)},
            "argus_r2": round(r2, 3),
            "riskfolio_stepwise": {k: round(float(full.loc[symbol, k]), 4) for k in ex.FACTORS},
            "origin": inputs.origins.get(symbol),
        }
    dropped = {s: [k for k in ex.FACTORS if r.get("riskfolio_stepwise", {}).get(k) == 0.0]
               for s, r in results.items() if "argus" in r}
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "method": "same daily returns, same four factors; ARGUS OLS keeps all four, Riskfolio-Lib "
                  "loadings_matrix(feature_selection='stepwise', stepwise='Forward', "
                  "criterion='pvalue', threshold=0.05) keeps those forward selection admits",
        "riskfolio_source": "research/repos-t2/dcajasn~riskfolio-lib/riskfolio/src/"
                            "ParamsEstimation.py:354,779 (BSD-3), executed unmodified",
        "results": results,
        "riskfolio_zeroed": dropped,
    }
    OUT.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
