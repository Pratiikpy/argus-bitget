"""Make the vendored qlib cross-sectional processor importable without editing a line of it.

``qlib_cs_processor.py`` is one file lifted out of a much larger package — its own unedited
imports (``qlib.utils.data``, ``qlib.constant``, ``qlib.data.dataset.utils``, ``qlib.utils.serial``,
``qlib.utils.paral``, ``qlib.data.inst_processor``, ``qlib.data``) pull in qlib's data-provider
framework, which this comparison never touches (it calls exactly two names from the vendored file:
``get_group_columns`` and ``CSRankNorm``). This module registers small stand-ins for every one of
those import targets in :data:`sys.modules` under the exact dotted names qlib itself would use, so
the vendored file's relative imports resolve without a real qlib installation and without editing a
single line of the vendored source.

Each stand-in raises :class:`NotImplementedError` if anything actually calls it, rather than
returning a plausible-looking wrong answer — the same fail-loud-not-fail-silent posture as the
vendored code it stands in for, and a real test (not just an absence of errors) that this
comparison genuinely never exercises the parts it isn't vendoring.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol

_BASELINES_DIR = Path(__file__).resolve().parent
_SHIM_MARKER = "_argus_qlib_baseline_shim"

# `qlib`, `qlib.data` and `qlib.utils` are also shimmed by `qlib_expression_loader.py` and
# `qlib_eval_surface_loader.py` (this package's other qlib-vendoring loaders) — real qlib puts
# `Expression`/`Rank`/... and `D`/`get_callable_kwargs` under the SAME three dotted names, so
# whichever of these loaders runs first in a process legitimately creates the shared namespace the
# others must reuse, not treat as a collision. This marker (duplicated, not imported, in each
# loader file — matching `tradingagents_loader.py`'s identical precedent) identifies those three
# specifically; every other shimmed name here stays private to this loader.
_SHARED_QLIB_NAMESPACE_MARKER = "_argus_qlib_namespace_shim"
_SHARED_QLIB_NAMESPACE_NAMES = frozenset({"qlib", "qlib.data", "qlib.utils"})


class QlibBaselineLoadError(RuntimeError):
    """The vendored qlib baseline could not be loaded, or a name it needs is already taken."""


def _unreachable(name: str) -> Any:
    """Return a stand-in that raises if ever actually called — used for shimmed qlib internals
    this comparison's own two call sites (``get_group_columns``, ``CSRankNorm``) never reach."""

    def _raise(*_args: Any, **_kwargs: Any) -> Any:
        raise NotImplementedError(
            f"{name} is a stand-in for real qlib infrastructure this comparison does not use "
            f"(only get_group_columns/CSRankNorm are exercised) — reaching this means a code "
            f"path outside that scope ran, which is itself the bug to fix, not this stub"
        )

    return _raise


def _shim_module(dotted_name: str, **attrs: Any) -> ModuleType:
    shared = dotted_name in _SHARED_QLIB_NAMESPACE_NAMES
    existing = sys.modules.get(dotted_name)
    if existing is not None:
        is_ours = getattr(existing, _SHIM_MARKER, False)
        is_shared = shared and getattr(existing, _SHARED_QLIB_NAMESPACE_MARKER, False)
        if not (is_ours or is_shared):
            raise QlibBaselineLoadError(
                f"sys.modules[{dotted_name!r}] already holds a module this loader did not "
                f"create; refusing to overwrite it (fail-closed, matching the vendored code's "
                f"own posture)"
            )
        if is_shared:
            # Another qlib-vendoring loader created this shared namespace entry first — merge in
            # any attrs THIS loader's own consumers need that are not already present, rather
            # than silently returning a module missing them (e.g. `qlib.data.D`, which
            # `qlib_cs_processor.py`'s own `from qlib.data import D` needs at import time).
            for k, v in attrs.items():
                if not hasattr(existing, k):
                    setattr(existing, k, v)
        return existing
    module = types.ModuleType(dotted_name)
    for k, v in attrs.items():
        setattr(module, k, v)
    setattr(module, _SHARED_QLIB_NAMESPACE_MARKER if shared else _SHIM_MARKER, True)
    sys.modules[dotted_name] = module
    return module


class _SerializableStub:
    """Stand-in for ``qlib.utils.serial.Serializable`` — ``Processor``'s base class.

    ``Processor.config()`` calls ``super().config(**kwargs)``; ``CSRankNorm.__call__`` never calls
    ``config()``, so a real implementation is never needed for this comparison, only a name that
    satisfies ``class Processor(Serializable):`` and would raise loudly if actually used."""

    def config(self, **kwargs: Any) -> None:
        raise NotImplementedError(
            "Serializable.config is a stand-in — CSRankNorm.__call__ never calls it"
        )


class _InstProcessorStub:
    """Stand-in for ``qlib.data.inst_processor.InstProcessor`` — only ``TimeRangeFlt``'s base
    class, a class this comparison never constructs."""


def _install_shims() -> None:
    # Every package a shimmed module here belongs to must be a plain empty package too (e.g.
    # `qlib.data` is both a namespace package AND, below, a module exposing `D`) — `_shim_module`
    # only sets attributes the FIRST time a dotted name is created, so each name gets exactly one
    # call, with every attribute it needs, rather than an empty pass followed by a second call
    # that would silently no-op against the already-shimmed module.
    for name in ("qlib", "qlib.data.dataset", "qlib.utils"):
        _shim_module(name)

    # `D` is qlib's global data-provider singleton; only `TimeRangeFlt.__init__` touches it
    # (`D.calendar(...)`), and this comparison never constructs a `TimeRangeFlt`.
    _shim_module("qlib.data", D=_unreachable("qlib.data.D"))
    _shim_module("qlib.utils.data", zscore=_unreachable("qlib.utils.data.zscore"),
                 robust_zscore=_unreachable("qlib.utils.data.robust_zscore"))
    _shim_module("qlib.constant", EPS=1e-12)
    _shim_module("qlib.data.dataset.utils",
                 fetch_df_by_index=_unreachable("qlib.data.dataset.utils.fetch_df_by_index"))
    _shim_module("qlib.utils.serial", Serializable=_SerializableStub)
    _shim_module("qlib.utils.paral",
                 datetime_groupby_apply=_unreachable("qlib.utils.paral.datetime_groupby_apply"))
    _shim_module("qlib.data.inst_processor", InstProcessor=_InstProcessorStub)


def load_processor_module() -> ModuleType:
    """Load the vendored ``qlib_cs_processor.py`` with its own unedited imports satisfied.

    Returns:
        The executed module object — carries ``get_group_columns``, ``CSRankNorm``,
        ``CSZScoreNorm`` and every other top-level name from the vendored file.
    """
    _install_shims()
    # Named to match qlib's own real dotted path (not a flat name like the shims above) because
    # the vendored file's own `from ...constant import EPS`-style relative imports resolve
    # against `__package__`, which importlib derives from this name's own dot structure — a flat
    # name has no parent package for a relative import to climb to.
    dotted = "qlib.data.dataset.processor"
    if dotted in sys.modules and getattr(sys.modules[dotted], _SHIM_MARKER, False):
        return sys.modules[dotted]

    spec = importlib.util.spec_from_file_location(dotted, _BASELINES_DIR / "qlib_cs_processor.py")
    if spec is None or spec.loader is None:
        raise QlibBaselineLoadError("could not build an import spec for qlib_cs_processor.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[dotted] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        del sys.modules[dotted]
        raise QlibBaselineLoadError(f"vendored qlib baseline failed to load: {exc}") from exc
    setattr(module, _SHIM_MARKER, True)
    return module


class QlibCrossSectionalBaseline(Protocol):
    """The subset of the vendored module's surface this comparison actually calls."""

    def get_group_columns(self, *args: Any, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class QlibCrossSectionSymbols:
    """Typed handle onto the vendored module's names, built once by :func:`load_qlib_baseline`."""

    cs_rank_norm: Any
    cs_zscore_norm: Any
    """Not used by ``crosssection_comparison.py`` (ARGUS has no zscore-normalisation operator to
    compare it against — see that module's own docstring) — exposed for
    ``test_baselines.py`` to confirm the shimmed ``zscore``/``robust_zscore`` dependencies fail
    loudly if ever actually reached, rather than trusting they are merely unused."""
    get_group_columns: Any


def load_qlib_baseline() -> QlibCrossSectionSymbols:
    """Load the vendored file and return the names ``crosssection_comparison.py`` needs.

    Raises:
        QlibBaselineLoadError: A name this loader needs was already taken, or the vendored file
            could not be executed.
    """
    module = load_processor_module()
    return QlibCrossSectionSymbols(
        cs_rank_norm=module.CSRankNorm,
        cs_zscore_norm=module.CSZScoreNorm,
        get_group_columns=module.get_group_columns,
    )


__all__ = [
    "QlibBaselineLoadError",
    "QlibCrossSectionSymbols",
    "QlibCrossSectionalBaseline",
    "load_processor_module",
    "load_qlib_baseline",
]
