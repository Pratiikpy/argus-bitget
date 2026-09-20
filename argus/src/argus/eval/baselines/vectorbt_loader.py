"""Make the vendored vectorbt DSR metrics importable without editing a line of them.

``vectorbt_dsr_metrics.py`` contains, verbatim, ``from vectorbt import _typing as tp`` — real code
inside vectorbt's own repo layout, where ``vectorbt._typing`` is a large module pulling in
``plotly``/``numba``/``mypy_extensions`` for dozens of type aliases. This file uses exactly one of
them, ``tp.Array1d`` (which vectorbt itself defines as a plain ``np.ndarray`` alias), purely as a
parameter/return annotation. This module registers a minimal stand-in exposing just that one name
under the exact dotted path vectorbt itself uses, so the vendored file's own import resolves
without installing plotly/numba to satisfy an annotation, and without editing a single line of the
vendored source.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol

import numpy as np

_BASELINES_DIR = Path(__file__).resolve().parent
_SHIM_MARKER = "_argus_vectorbt_baseline_shim"


class VectorbtBaselineLoadError(RuntimeError):
    """The vendored vectorbt baseline could not be loaded, or a name it needs is already taken."""


def _shim_module(dotted_name: str, **attrs: Any) -> ModuleType:
    existing = sys.modules.get(dotted_name)
    if existing is not None:
        if not getattr(existing, _SHIM_MARKER, False):
            raise VectorbtBaselineLoadError(
                f"sys.modules[{dotted_name!r}] already holds a module this loader did not "
                f"create; refusing to overwrite it (fail-closed, matching the vendored code's "
                f"own posture)"
            )
        return existing
    module = types.ModuleType(dotted_name)
    for k, v in attrs.items():
        setattr(module, k, v)
    setattr(module, _SHIM_MARKER, True)
    sys.modules[dotted_name] = module
    return module


def load_metrics_module() -> ModuleType:
    """Load the vendored ``vectorbt_dsr_metrics.py`` with its own unedited import satisfied.

    Returns:
        The executed module object — carries ``deflated_sharpe_ratio`` and
        ``approx_exp_max_sharpe``.
    """
    _shim_module("vectorbt")
    _shim_module("vectorbt._typing", Array1d=np.ndarray)

    dotted = "argus_baseline_vectorbt_dsr_metrics"
    if dotted in sys.modules and getattr(sys.modules[dotted], _SHIM_MARKER, False):
        return sys.modules[dotted]

    spec = importlib.util.spec_from_file_location(
        dotted, _BASELINES_DIR / "vectorbt_dsr_metrics.py"
    )
    if spec is None or spec.loader is None:
        raise VectorbtBaselineLoadError(
            "could not build an import spec for vectorbt_dsr_metrics.py"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[dotted] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        del sys.modules[dotted]
        raise VectorbtBaselineLoadError(
            f"vendored vectorbt baseline failed to load: {exc}"
        ) from exc
    setattr(module, _SHIM_MARKER, True)
    return module


class VectorbtDsrBaseline(Protocol):
    """The subset of the vendored module's surface this comparison actually calls."""

    def deflated_sharpe_ratio(self, *args: Any, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class VectorbtDsrSymbols:
    """Typed handle onto the vendored module's names, built once by
    :func:`load_vectorbt_baseline`."""

    deflated_sharpe_ratio: Any
    approx_exp_max_sharpe: Any


def load_vectorbt_baseline() -> VectorbtDsrSymbols:
    """Load the vendored file and return the names ``dsr_comparison.py`` needs.

    Raises:
        VectorbtBaselineLoadError: A name this loader needs was already taken, or the vendored
            file could not be executed.
    """
    module = load_metrics_module()
    return VectorbtDsrSymbols(
        deflated_sharpe_ratio=module.deflated_sharpe_ratio,
        approx_exp_max_sharpe=module.approx_exp_max_sharpe,
    )


def vectorbt_var_sharpe(sharpe_ratios: list[float], *, ddof: int = 1) -> float:
    """The exact computation at ``vectorbt/returns/accessors.py:596``: ``np.var(sharpe_ratio,
    ddof=ddof)``. Not vendored — it is already a single, bare numpy call with nothing else
    around it (the surrounding method is the heavy pandas-accessor code this loader is
    deliberately not pulling in) — reproduced here as exactly that one call, cited to the exact
    line, so :mod:`argus.eval.dsr_comparison` runs the real computation rather than describing it.
    """
    return float(np.var(np.asarray(sharpe_ratios, dtype=float), ddof=ddof))


__all__ = [
    "VectorbtBaselineLoadError",
    "VectorbtDsrBaseline",
    "VectorbtDsrSymbols",
    "load_metrics_module",
    "load_vectorbt_baseline",
    "vectorbt_var_sharpe",
]
