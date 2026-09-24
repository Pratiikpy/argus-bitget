"""skfolio's conditional stress engines, run unmodified, for `eval/copilot_stress.py`.

skfolio (BSD-3, skfolio/skfolio 1.3.1) is the method rival the review of 2026-09-24 named for the
stress half of the Portfolio Copilot sub-theme: a book's move *given* a benchmark move, from

* ``VineCopula(central_assets=[benchmark]).sample(conditioning={benchmark: shock})`` — a regular
  vine fitted to the joint daily returns, sampled with the benchmark pinned to the shock
  (``distribution/multivariate/_vine_copula.py:518-595``), and
* ``EntropyPooling(mean_views=[f"{benchmark} == {shock}"])`` — the historical scenarios reweighted
  with the least relative entropy that makes the benchmark's mean equal the shock
  (``prior/_entropy_pooling.py:436-590``), read from ``return_distribution_``.

It runs in its own interpreter (``ARGUS_SKFOLIO_PYTHON``; skfolio pulls numpy, scipy and
scikit-learn, which the stdlib-only desk does not carry). Input on stdin, output on stdout, both
JSON::

    {"events": [{"id": str, "names": [...], "returns": [[...], ...], "benchmark": str,
                 "shock": float, "seed": int}], "n_samples": int}

Returns, per event, the vine's conditional samples (``n_samples`` x names) and Entropy Pooling's
posterior scenario weights (one per row of ``returns``). The harness turns both into a book's move
and quantiles, so the rival's numbers are the rival's and only the weighting of assets is ours.
"""

from __future__ import annotations

import json
import sys
from typing import Any


def run(payload: dict[str, Any]) -> dict[str, Any]:
    import numpy as np  # type: ignore[import-not-found, unused-ignore]
    from skfolio.distribution import VineCopula  # type: ignore[import-not-found, unused-ignore]
    from skfolio.prior import EntropyPooling  # type: ignore[import-not-found, unused-ignore]

    out: dict[str, Any] = {}
    for event in payload["events"]:
        names = list(event["names"])
        x = np.asarray(event["returns"], dtype=float)
        bench = event["benchmark"]
        record: dict[str, Any] = {"names": names}
        try:
            import pandas as pd  # type: ignore[import-not-found, unused-ignore]

            frame = pd.DataFrame(x, columns=names)
            vine = VineCopula(central_assets=[bench], random_state=int(event["seed"]))
            vine.fit(frame)
            samples = vine.sample(n_samples=int(payload["n_samples"]),
                                  conditioning={bench: float(event["shock"])})
            record["vine_samples"] = np.asarray(samples, dtype=float).round(6).tolist()
        except Exception as exc:  # the rival's own failure is recorded, not hidden
            record["vine_error"] = f"{type(exc).__name__}: {exc}"
        try:
            import pandas as pd  # type: ignore[import-not-found, unused-ignore]

            frame = pd.DataFrame(x, columns=names)
            view = [f"{bench} == {float(event['shock'])!r}"]
            try:
                ep = EntropyPooling(mean_views=view).fit(frame)
                record["ep_solver"] = "TNC"
            except Exception as first:
                # skfolio's own error text names this remedy: "try another solver such as
                # 'CLARABEL'". Followed, and recorded, rather than scoring the default's failure.
                ep = EntropyPooling(mean_views=view, solver="CLARABEL").fit(frame)
                record["ep_solver"] = f"CLARABEL (TNC failed: {type(first).__name__})"
            weights = np.asarray(ep.return_distribution_.sample_weight, dtype=float)
            record["ep_weights"] = weights.tolist()
            record["ep_effective_scenarios"] = float(ep.effective_number_of_scenarios_)
        except Exception as exc:
            record["ep_error"] = f"{type(exc).__name__}: {exc}"
        out[event["id"]] = record
    import skfolio  # type: ignore[import-not-found, unused-ignore]

    return {"skfolio_version": skfolio.__version__, "events": out}


def main() -> int:  # pragma: no cover - runs in the rival's interpreter
    payload = json.load(sys.stdin)
    json.dump(run(payload), sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
