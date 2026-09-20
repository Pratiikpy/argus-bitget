"""Make the vendored `stumpy_squared_distance.py` (STUMPY's real `_calculate_squared_distance`)
importable and CALLABLE without editing it.

Like `qlib_eval_surface.py`, this is a decisive EXCERPT (one function, not a whole file) with its
own imports stripped — so its free names (`np`, `config`, `njit`) are injected directly into the
loaded module's `__dict__` before `exec_module` runs, the same technique
`qlib_eval_surface_loader.py` uses. Three real dependencies, satisfied narrowly:

1. `numpy` — real, already installed; the function's own `np.isinf`/`np.abs` calls run against it
   unmodified.
2. `numba.njit` — real, already installed (`numba==0.67.0`, confirmed); the function keeps its own
   real `@njit(fastmath=...)` decorator, so it genuinely JIT-compiles exactly as STUMPY's own code
   does, not a stripped-down "looks like it" version.
3. `config.STUMPY_DENOM_THRESHOLD` / `config.STUMPY_FASTMATH_FLAGS` — the real VALUES from
   `stumpy/config.py:13,21` at the vendored commit (`1e-14` and
   `{"nsz", "arcp", "contract", "afn", "reassoc"}`), supplied as a small namespace rather than by
   vendoring the whole config module, since this excerpt reads only these two names from it.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import ModuleType

_BASELINES_DIR = Path(__file__).resolve().parent
_SHIM_MARKER = "_argus_stumpy_squared_distance_shim"


class StumpySquaredDistanceLoadError(RuntimeError):
    """The vendored STUMPY squared-distance function could not be loaded."""


def load_squared_distance_module() -> ModuleType:
    """Load `stumpy_squared_distance.py` with `_calculate_squared_distance` fully callable.

    Returns the loaded module — call `._calculate_squared_distance(m, QT, mu_Q, sigma_Q, M_T,
    Sigma_T, Q_subseq_isconstant, T_subseq_isconstant)` exactly as STUMPY's real code does.
    """
    dotted = "argus_eval_stumpy_squared_distance"
    existing = sys.modules.get(dotted)
    if existing is not None:
        if not getattr(existing, _SHIM_MARKER, False):
            raise StumpySquaredDistanceLoadError(
                f"sys.modules[{dotted!r}] already holds a module this loader did not create"
            )
        return existing

    import numpy as np
    from numba import njit

    path = _BASELINES_DIR / "stumpy_squared_distance.py"
    spec = importlib.util.spec_from_file_location(dotted, path)
    if spec is None or spec.loader is None:
        raise StumpySquaredDistanceLoadError(f"could not build an import spec for {path}")
    module = importlib.util.module_from_spec(spec)

    module.np = np  # type: ignore[attr-defined]
    module.njit = njit  # type: ignore[attr-defined]
    # Real values from stumpy/config.py:13,21 at the vendored commit — see this module's own
    # docstring for why only the values, not the config module, are carried over. A real
    # `types.ModuleType`, not `SimpleNamespace`: numba's `@njit` nopython-mode global-resolution
    # freezes module-attribute constants at compile time but does not know how to type a
    # `SimpleNamespace` at all (`TypingError: Cannot determine Numba type of
    # <class 'types.SimpleNamespace'>`, hit directly before this fix) — found by actually running
    # the real decorator, not assumed from reading numba's docs.
    config_module = types.ModuleType("argus_eval_stumpy_config_values")
    config_module.STUMPY_DENOM_THRESHOLD = 1e-14  # type: ignore[attr-defined]
    config_module.STUMPY_FASTMATH_FLAGS = {"nsz", "arcp", "contract", "afn", "reassoc"}  # type: ignore[attr-defined]
    module.config = config_module  # type: ignore[attr-defined]
    setattr(module, _SHIM_MARKER, True)

    sys.modules[dotted] = module
    spec.loader.exec_module(module)
    return module


__all__ = ["StumpySquaredDistanceLoadError", "load_squared_distance_module"]
