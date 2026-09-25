"""Runs the two general-purpose selective-prediction evaluators on ARGUS's frozen abstention input.

This is the rival half of `eval/general_abstention_comparison.py`, kept as a standalone script
because the rivals cannot share ARGUS's environment: fd-shifts' ``rc_stats.py`` calls ``np.trapz``
and ``np.infty``, both gone from NumPy 2, and torch-uncertainty needs ``torchmetrics``. It imports
nothing from ``argus``, so it runs in any Python 3.11 environment holding ``numpy<2``, ``torch``,
``torchmetrics``, ``scipy``, ``scikit-learn`` and ``matplotlib``.

Both rivals are run from their own clones, unmodified:

* **fd-shifts** (`IML-DKFZ/fd-shifts` @ c4467aec, Apache-2.0). Its package ``__init__`` imports
  loguru, omegaconf and faiss for unrelated experiment code, so the two risk-coverage files
  (``fd_shifts/analysis/rc_stats.py`` and ``rc_stats_utils.py``) are loaded as a synthetic package
  straight from the clone, byte for byte.
* **torch-uncertainty** (`torch-uncertainty/torch-uncertainty` @ 3f82fe5d, Apache-2.0). Its
  ``metrics/classification/risk_coverage.py`` is loaded straight from the clone for the same
  reason: the package ``__init__`` pulls Lightning and the model zoo, and the metric file needs
  only torch, torchmetrics, numpy and matplotlib.

The file hashes of all three are written into the output, so a reader can check they ran the same
bytes. Usage::

    python -m argus.eval.general_abstention_comparison --write-rival-input frozen.json
    python src/argus/eval/baselines/selective_rivals_runner.py \\
        --fd-shifts <fd-shifts clone> --torch-uncertainty <torch-uncertainty clone> \\
        --input frozen.json --out rival_results.json

The planted-skill construction below is the same as
`general_abstention_comparison.planted_confidences`, including the ``random.Random(7)`` shuffle,
so every arm is the same input on both sides; a test pins the two against each other.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import sys
import types
from pathlib import Path
from typing import Any

FD_SHIFTS_COMMIT = "c4467aec134e99691359da209f811d91283fc1e3"
TORCH_UNCERTAINTY_COMMIT = "3f82fe5d15a7bf877a821731baadef1e4731c31d"

FIXTURES: dict[str, tuple[list[float], list[int]]] = {
    "ties": ([0.9, 0.8, 0.8, 0.7, 0.6, 0.6, 0.6, 0.5], [0, 1, 0, 0, 1, 1, 0, 1]),
    "all_tied": ([0.6] * 6, [1, 0, 1, 1, 0, 0]),
    "perfect": ([0.9, 0.8, 0.7, 0.6], [0, 0, 1, 1]),
    "inverted": ([0.9, 0.8, 0.7, 0.6], [1, 1, 0, 0]),
}
"""Small inputs whose rival output ARGUS's parity tests pin to the last digit."""

PERMUTATIONS = 200
PERMUTATION_SEED = 20260925


def planted(nets: list[float]) -> dict[str, list[float]]:
    """The real outcomes with confidences re-assigned: perfect, shuffled, inverted, constant."""
    n = len(nets)
    rank = sorted(range(n), key=lambda i: nets[i])
    oracle = [0.0] * n
    for r, i in enumerate(rank):
        oracle[i] = 0.05 + 0.9 * r / (n - 1)
    shuffled = list(oracle)
    random.Random(7).shuffle(shuffled)
    return {"oracle": oracle, "random": shuffled, "inverted": [1.0 - x for x in oracle],
            "constant": [0.6] * n}


def _sha16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _load(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_fd_shifts(clone: Path) -> Any:
    """fd-shifts' ``rc_stats`` module, its relative import of ``rc_stats_utils`` intact."""
    analysis = clone / "fd_shifts" / "analysis"
    package = types.ModuleType("fdrc")
    package.__path__ = [str(analysis)]
    sys.modules["fdrc"] = package
    _load("fdrc.rc_stats_utils", analysis / "rc_stats_utils.py")
    return _load("fdrc.rc_stats", analysis / "rc_stats.py")


def load_torch_uncertainty(clone: Path) -> Any:
    return _load("tu_risk_coverage", clone / "src" / "torch_uncertainty" / "metrics"
                 / "classification" / "risk_coverage.py")


class Rivals:
    def __init__(self, fd_clone: Path, tu_clone: Path) -> None:
        self.rc = load_fd_shifts(fd_clone)
        self.tu = load_torch_uncertainty(tu_clone)

    def fd(self, conf: list[float], res: list[int], *, ci: bool = False) -> dict[str, Any]:
        """fd-shifts' own quantities, divided by its x1000 display scale."""
        import numpy as np

        s = self.rc.RiskCoverageStats(confids=np.asarray(conf, dtype=float),
                                      residuals=np.asarray(res, dtype=float))
        out: dict[str, Any] = {
            "aurc": float(s.aurc) / 1000, "augrc": float(s.augrc) / 1000,
            "aurc_optimal": float(s.aurc_optimal) / 1000,
            "augrc_optimal": float(s.augrc_optimal) / 1000,
            "eaurc": float(s.eaurc) / 1000, "n_rc_points": len(s.coverages),
        }
        try:
            cov, risk, thr = s.get_working_point(risk="selective-risk", target_risk=0.5)
            out["working_point"] = {"coverage": float(cov), "risk": float(risk),
                                    "threshold": float(thr)}
        except ValueError as exc:  # recorded: no admissible point is the finding
            out["working_point"] = f"error: {exc}"
        if ci:
            np.random.seed(0)
            lo, hi = s.evaluate_ci(risk="selective-risk", n_bs=2000)
            out["aurc_ci95_iid"] = [float(lo) / 1000, float(hi) / 1000]
        return out

    def torch_uncertainty(self, conf: list[float], res: list[int]) -> dict[str, Any]:
        """AURC, AUGRC, RiskAtxCov(0.5) and CovAtxRisk(0.5), NaN recorded as ``None``.

        A binary classifier is encoded with ``p = 0.5 + confidence/2`` for the predicted class
        (monotone, so the ranking is the confidence's) and the target is the predicted class
        exactly when the call would have netted a profit.
        """
        import math

        import torch

        p = torch.tensor([0.5 + c / 2 for c in conf], dtype=torch.float64)
        probs = torch.stack([1 - p, p], dim=-1)
        targets = torch.tensor([1 - int(r) for r in res])
        out: dict[str, Any] = {}
        for name, metric in (("aurc", self.tu.AURC()), ("augrc", self.tu.AUGRC()),
                             ("risk_at_50cov", self.tu.RiskAtxCov(0.5)),
                             ("cov_at_50risk", self.tu.CovAtxRisk(0.5))):
            metric.update(probs, targets)
            value = float(metric.compute().reshape(-1)[0])
            out[name] = None if math.isnan(value) else value
        return out


def run(blob: dict[str, Any], fd_clone: Path, tu_clone: Path) -> dict[str, Any]:
    import numpy as np
    import torch

    rivals = Rivals(fd_clone, tu_clone)
    calls = sorted(blob["calls"], key=lambda c: c["seq"])
    conf = [float(c["confidence"]) for c in calls]
    doubt = [1.0 - float(c["stated_confidence"]) for c in calls]
    err = [int(c["error"]) for c in calls]
    nets = [float(c["net_bps"]) for c in calls]
    n = len(calls)
    analysis = fd_clone / "fd_shifts" / "analysis"

    results: dict[str, Any] = {
        "input": {k: blob[k] for k in ("upto_seq", "settled_by", "ledger_head_at_freeze")}
        | {"calls": n},
        "provenance": {
            "fd_shifts_commit": FD_SHIFTS_COMMIT,
            "fd_shifts_rc_stats_sha256_16": _sha16(analysis / "rc_stats.py"),
            "fd_shifts_rc_stats_utils_sha256_16": _sha16(analysis / "rc_stats_utils.py"),
            "torch_uncertainty_commit": TORCH_UNCERTAINTY_COMMIT,
            "torch_uncertainty_risk_coverage_sha256_16": _sha16(
                tu_clone / "src" / "torch_uncertainty" / "metrics" / "classification"
                / "risk_coverage.py"),
            "python": sys.version.split()[0], "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "live": {
            "lean_confidence": {"fd_shifts": rivals.fd(conf, err, ci=True),
                                "torch_uncertainty": rivals.torch_uncertainty(conf, err)},
            "abstention_doubt": {"fd_shifts": rivals.fd(doubt, err, ci=True),
                                 "torch_uncertainty": rivals.torch_uncertainty(doubt, err)},
        },
    }

    rng = np.random.default_rng(PERMUTATION_SEED)
    fd_a: list[float] = []
    tu_a: list[float] = []
    tu_g: list[float] = []
    for _ in range(PERMUTATIONS):
        perm = rng.permutation(n)
        c = [conf[i] for i in perm]
        e = [err[i] for i in perm]
        fd_a.append(rivals.fd(c, e)["aurc"])
        t = rivals.torch_uncertainty(c, e)
        tu_a.append(t["aurc"])
        tu_g.append(t["augrc"])

    def spread(xs: list[float]) -> dict[str, float]:
        return {"min": min(xs), "max": max(xs), "spread": max(xs) - min(xs)}

    results["row_order_sensitivity"] = {
        "permutations": PERMUTATIONS, "fd_shifts_aurc": spread(fd_a),
        "torch_uncertainty_aurc": spread(tu_a), "torch_uncertainty_augrc": spread(tu_g),
    }
    results["planted_skill"] = {
        label: {"fd_shifts": rivals.fd(cs, err),
                "torch_uncertainty": rivals.torch_uncertainty(cs, err)}
        for label, cs in planted(nets).items()
    }
    results["fixtures"] = {
        label: {"confids": c, "residuals": e, "fd_shifts": rivals.fd(c, e),
                "torch_uncertainty": rivals.torch_uncertainty(c, e),
                "torch_uncertainty_reversed_rows": rivals.torch_uncertainty(c[::-1], e[::-1])}
        for label, (c, e) in FIXTURES.items()
    }
    return results


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI, rival environment
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--fd-shifts", type=Path, required=True, help="fd-shifts clone")
    parser.add_argument("--torch-uncertainty", type=Path, required=True,
                        help="torch-uncertainty clone")
    parser.add_argument("--input", type=Path, required=True,
                        help="written by general_abstention_comparison --write-rival-input")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    blob = json.loads(args.input.read_text(encoding="utf-8"))
    results = run(blob, args.fd_shifts, args.torch_uncertainty)
    args.out.write_text(json.dumps(results, indent=1, allow_nan=False), encoding="utf-8")
    print(json.dumps(results["provenance"], indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
