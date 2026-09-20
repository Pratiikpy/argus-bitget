"""Make the vendored qlib expression engine (`qlib_expression_base.py` + `qlib_expression_ops.py`)
importable and CALLABLE without editing a line of either — including the deferred cache import
inside `Expression.load()`, which an earlier session wrongly concluded was unavoidable (see
`qlib_expression_base.py`'s own header for the correction).

Three real dependency layers are satisfied here, each with the narrowest stand-in that lets the
real code run unmodified:

1. Module-level imports `ops.py` itself makes (`._libs.rolling`, `._libs.expanding` — the
   Cython-accelerated trend operators `Slope`/`Rsquare`/`Resi`/`WMA` use, never constructed by
   this comparison; `.base`; `..log`; `..utils`) — shimmed so the file imports at all.
2. The DEFERRED `from .cache import H` inside `Expression.load()` (only reached when `.load()` is
   actually CALLED) — shimmed with a plain `{"f": {}}` dict, matching the real code's own
   `H["f"][cache_key]` usage exactly, never the real redis-backed cache.
3. `Feature._load_internal`'s own `from .data import FeatureD` (a real data-provider needing a
   full qlib calendar/instrument setup this comparison has no need to stand up) is never reached
   at all — this comparison's own `SyntheticLeaf` (below) is an `Expression` subclass with its
   own `_load_internal` returning a caller-supplied `pandas.Series` directly, so real qlib
   operators (`Rank`, `Mean`, `Std`, `Corr`, ...) compose around a leaf that never calls `Feature`
   or its data-provider chain at all.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd

_BASELINES_DIR = Path(__file__).resolve().parent
_SHIM_MARKER = "_argus_qlib_expression_shim"

# `qlib`, `qlib.data` and `qlib.utils` are also shimmed by `qlib_loader.py` (this package's other
# qlib-vendoring loader, used for `qlib_cs_processor.py`) — real qlib puts `Expression`/`D`/
# `get_callable_kwargs` under the SAME three dotted names, so whichever loader runs first in a
# process legitimately creates the shared namespace the other must reuse, not treat as a
# collision. This marker (duplicated, not imported, in each loader file — matching
# `tradingagents_loader.py`'s identical precedent) identifies those three specifically; every
# other shimmed name here (`qlib.data._libs`, `qlib.log`, `qlib.data.cache`, ...) stays private.
_SHARED_QLIB_NAMESPACE_MARKER = "_argus_qlib_namespace_shim"
_SHARED_QLIB_NAMESPACE_NAMES = frozenset({"qlib", "qlib.data", "qlib.utils"})


class QlibExpressionLoadError(RuntimeError):
    """The vendored qlib expression engine could not be loaded, or a name it needs is taken."""


def _load_module_from_file(dotted_name: str, path: Path, *, register: bool = False) -> ModuleType:
    spec = importlib.util.spec_from_file_location(dotted_name, path)
    if spec is None or spec.loader is None:
        raise QlibExpressionLoadError(f"could not build an import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    if register:
        sys.modules[dotted_name] = module
    spec.loader.exec_module(module)
    return module


def _shim_package(dotted_name: str) -> ModuleType:
    shared = dotted_name in _SHARED_QLIB_NAMESPACE_NAMES
    existing = sys.modules.get(dotted_name)
    if existing is not None:
        is_ours = getattr(existing, _SHIM_MARKER, False)
        is_shared = shared and getattr(existing, _SHARED_QLIB_NAMESPACE_MARKER, False)
        if not (is_ours or is_shared):
            raise QlibExpressionLoadError(
                f"sys.modules[{dotted_name!r}] already holds a module this loader did not "
                f"create; refusing to overwrite it (fail-closed, matching the vendored code's "
                f"own posture)"
            )
        return existing
    module = types.ModuleType(dotted_name)
    setattr(module, _SHARED_QLIB_NAMESPACE_MARKER if shared else _SHIM_MARKER, True)
    sys.modules[dotted_name] = module
    return module


def _make_unreachable(name: str) -> object:
    def _unreachable(*args: object, **kwargs: object) -> object:
        raise NotImplementedError(
            f"{name} is a stub for a Cython-accelerated trend operator "
            f"(Slope/Rsquare/Resi/WMA) — this comparison never constructs those operators, so "
            f"being called means a test forgot to substitute a real one"
        )

    _unreachable.__name__ = name
    return _unreachable


def _shim_cache_module() -> None:
    """`Expression.load()`'s deferred `from .cache import H` — a plain in-memory dict matching
    the real `H["f"][cache_key]` usage exactly, never the real redis-backed cache."""
    dotted = "qlib.data.cache"
    existing = sys.modules.get(dotted)
    if existing is not None and getattr(existing, _SHIM_MARKER, False):
        return
    if existing is not None:
        raise QlibExpressionLoadError(f"sys.modules[{dotted!r}] already holds a module")
    module = types.ModuleType(dotted)
    setattr(module, _SHIM_MARKER, True)
    module.H = {"f": {}}  # type: ignore[attr-defined]
    sys.modules[dotted] = module


def load_expression_module() -> tuple[ModuleType, ModuleType]:
    """Load the vendored `qlib_expression_base.py` and `qlib_expression_ops.py` with every one
    of their own unedited imports satisfied.

    Returns:
        (base_module, ops_module) — `ops_module` carries `Ref`, `Mean`, `Std`, `Max`, `Min`,
        `Rank`, `Corr` and the rest of the real operator classes.
    """
    _shim_package("qlib")
    _shim_package("qlib.data")
    _shim_package("qlib.data._libs")

    for sub, fn_names in (
        ("rolling", ("rolling_slope", "rolling_rsquare", "rolling_resi")),
        ("expanding", ("expanding_slope", "expanding_rsquare", "expanding_resi")),
    ):
        dotted = f"qlib.data._libs.{sub}"
        existing = sys.modules.get(dotted)
        if existing is not None and getattr(existing, _SHIM_MARKER, False):
            continue
        if existing is not None:
            raise QlibExpressionLoadError(f"sys.modules[{dotted!r}] already holds a module")
        module = types.ModuleType(dotted)
        setattr(module, _SHIM_MARKER, True)
        for fname in fn_names:
            setattr(module, fname, _make_unreachable(fname))
        sys.modules[dotted] = module

    log_dotted = "qlib.log"
    if log_dotted not in sys.modules or not getattr(sys.modules[log_dotted], _SHIM_MARKER, False):
        import logging

        log_module = types.ModuleType(log_dotted)
        setattr(log_module, _SHIM_MARKER, True)
        log_module.get_module_logger = logging.getLogger  # type: ignore[attr-defined]
        sys.modules[log_dotted] = log_module

    utils_dotted = "qlib.utils"

    def _get_callable_kwargs(*args: object, **kwargs: object) -> tuple[Any, dict[str, Any]]:
        raise NotImplementedError(
            "get_callable_kwargs stub — this comparison never constructs a custom operator"
        )

    existing_utils = sys.modules.get(utils_dotted)
    if existing_utils is not None and (
        getattr(existing_utils, _SHIM_MARKER, False)
        or getattr(existing_utils, _SHARED_QLIB_NAMESPACE_MARKER, False)
    ):
        # Another qlib-vendoring loader (or an earlier call in this one) already created this —
        # merge in the one attr this loader's own consumers need, if it is not already there,
        # rather than clobbering a module `qlib.utils.data`/`.serial`/`.paral` may already be
        # nested under.
        if not hasattr(existing_utils, "get_callable_kwargs"):
            existing_utils.get_callable_kwargs = _get_callable_kwargs  # type: ignore[attr-defined]
    elif existing_utils is not None:
        raise QlibExpressionLoadError(
            f"sys.modules[{utils_dotted!r}] already holds a module this loader did not create; "
            f"refusing to overwrite it (fail-closed, matching the vendored code's own posture)"
        )
    else:
        utils_module = types.ModuleType(utils_dotted)
        # "qlib.utils" is one of the three shared namespace names (see the module-level
        # docstring) — tagged with the SHARED marker, not this loader's private one, so
        # `qlib_loader.py` recognizes and reuses it (merging in its own needed attrs) rather than
        # refusing it as an unrecognized module if it loads second.
        setattr(utils_module, _SHARED_QLIB_NAMESPACE_MARKER, True)
        utils_module.get_callable_kwargs = _get_callable_kwargs  # type: ignore[attr-defined]
        sys.modules[utils_dotted] = utils_module

    _shim_cache_module()

    base_dotted = "qlib.data.base"
    if base_dotted in sys.modules and getattr(sys.modules[base_dotted], _SHIM_MARKER, False):
        base_module = sys.modules[base_dotted]
    else:
        base_module = _load_module_from_file(
            base_dotted, _BASELINES_DIR / "qlib_expression_base.py", register=True,
        )
        setattr(base_module, _SHIM_MARKER, True)

    ops_dotted = "qlib.data.ops"
    if ops_dotted in sys.modules and getattr(sys.modules[ops_dotted], _SHIM_MARKER, False):
        return base_module, sys.modules[ops_dotted]

    ops_module = _load_module_from_file(
        ops_dotted, _BASELINES_DIR / "qlib_expression_ops.py", register=True,
    )
    setattr(ops_module, _SHIM_MARKER, True)
    return base_module, ops_module


class SyntheticLeaf:
    """An `Expression`-compatible leaf whose `_load_internal` returns a caller-supplied
    `pandas.Series` directly — never touches `Feature`/`FeatureD` or any real data provider.
    Real qlib operators (`Rank(leaf, N)`, `Mean(leaf, N)`, ...) compose around this exactly as
    they would around a real `Feature`, calling `self.feature.load(...)` on it, which resolves
    to THIS class's own `.load()` (inherited from the real, vendored `Expression`) — the real
    cache-check path runs for real, against the shimmed in-memory `H`, not bypassed."""

    def __init__(self, series: pd.Series, *, name: str = "synthetic") -> None:
        self._series = series
        self._name = name

    def __str__(self) -> str:
        return f"${self._name}"

    def _load_internal(
        self, instrument: str, start_index: object, end_index: object, *args: object,
    ) -> pd.Series:
        return self._series.copy()

    def get_longest_back_rolling(self) -> int:
        return 0

    def get_extended_window_size(self) -> tuple[int, int]:
        return 0, 0


def make_synthetic_leaf(series: pd.Series, *, name: str = "synthetic") -> Any:
    """Build a real `SyntheticLeaf`, but inheriting from the REAL vendored `Expression` class
    (not just duck-typed) so `Expression.load()`'s own cache-check logic runs for real."""
    base_module, _ = load_expression_module()

    # SyntheticLeaf MUST come first in the MRO: Expression declares _load_internal etc. as
    # @abc.abstractmethod, and Python's attribute lookup for __abstractmethods__ resolution
    # stops at the FIRST class in the MRO that defines a name — Expression-first would find its
    # own abstract stub before ever reaching SyntheticLeaf's concrete override.
    class _RealSyntheticLeaf(
        SyntheticLeaf, base_module.Expression  # type: ignore[misc, name-defined]
    ):
        def __init__(self, series: pd.Series, *, name: str = "synthetic") -> None:
            SyntheticLeaf.__init__(self, series, name=name)

    return _RealSyntheticLeaf(series, name=name)


__all__ = [
    "QlibExpressionLoadError",
    "SyntheticLeaf",
    "load_expression_module",
    "make_synthetic_leaf",
]
