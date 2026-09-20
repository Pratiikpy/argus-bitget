"""Make the vendored `qlib_eval_surface.py` (qlib's real `parse_field` + `ExpressionProvider`)
importable and CALLABLE without editing a line of it.

The vendored file is an EXCERPT, not a whole module — unlike `qlib_expression_base.py`/
`qlib_expression_ops.py`, it carries no import statements of its own (the two snippets are 26 and
25 lines respectively, copied byte-for-byte from `qlib/utils/__init__.py` and `qlib/data/data.py`,
starting mid-file). So the six names its code references but never defines itself (`re`, `abc`,
`Feature`, `PFeature`, `Operators`, `get_module_logger`) cannot be satisfied by pre-registering
`sys.modules` entries the way `qlib_expression_loader.py` does for real `import` statements —
there is nothing here for Python's import machinery to intercept. Instead this loader injects
those six names directly into the vendored module's `__dict__` BEFORE `exec_module` runs, so that:

- `abc` resolves for `class ExpressionProvider(abc.ABC):`, evaluated at class-definition time
  (i.e. during `exec_module` itself, not deferred to first use).
- `re`, `Feature`, `PFeature`, `Operators`, `get_module_logger` resolve at call time, when
  `parse_field()` / `get_expression_instance()` actually run — including inside the bare
  `eval(parse_field(field))` call, which (per Python's `eval(expr)` one-argument form) runs with
  the CALLING FRAME's globals, i.e. this module's own `__dict__` — exactly the same dict these six
  names were injected into, so a field string's `Operators.Ref(...)` / `Feature("close")` text
  resolves against the real, vendored `qlib_expression_ops.py` operator registry.

`Feature`/`PFeature`/`Operators` are the REAL ones from the already-vendored
`qlib_expression_ops.py` (via `qlib_expression_loader.load_expression_module()`), not stand-ins —
so a field string this
loader evaluates constructs the same real operator objects `eval/grammar_comparison.py`'s numeric
comparison already exercises directly.
"""

from __future__ import annotations

import abc
import importlib.util
import re
import sys
import types
from pathlib import Path
from types import ModuleType

from argus.eval.baselines.qlib_expression_loader import load_expression_module

_BASELINES_DIR = Path(__file__).resolve().parent
_SHIM_MARKER = "_argus_qlib_eval_surface_shim"


class QlibEvalSurfaceLoadError(RuntimeError):
    """The vendored qlib eval-surface excerpt could not be loaded, or a name it needs is taken."""


def load_eval_surface_module() -> ModuleType:
    """Load `qlib_eval_surface.py` with `parse_field`/`ExpressionProvider` fully callable.

    Returns the loaded module — call `.parse_field(field)` or build an
    `.ExpressionProvider()` and call `.get_expression_instance(field)` on it, exactly as real
    qlib code does.
    """
    dotted = "argus_eval_qlib_eval_surface"
    existing = sys.modules.get(dotted)
    if existing is not None:
        if not getattr(existing, _SHIM_MARKER, False):
            raise QlibEvalSurfaceLoadError(
                f"sys.modules[{dotted!r}] already holds a module this loader did not create"
            )
        return existing

    _base_module, ops_module = load_expression_module()

    # Idempotent: OpsWrapper.register() overwrites by class name and only logs a warning on
    # collision, so calling this every load (rather than caching a "registered once" flag) is
    # harmless and keeps this function stateless between test runs.
    ops_module.Operators.register(ops_module.OpsList)

    path = _BASELINES_DIR / "qlib_eval_surface.py"
    spec = importlib.util.spec_from_file_location(dotted, path)
    if spec is None or spec.loader is None:
        raise QlibEvalSurfaceLoadError(f"could not build an import spec for {path}")
    module = importlib.util.module_from_spec(spec)

    # Injected BEFORE exec_module: `abc` is read at class-definition time
    # (`class ExpressionProvider(abc.ABC):`), which happens inside exec_module itself.
    module.re = re  # type: ignore[attr-defined]
    module.abc = abc  # type: ignore[attr-defined]
    module.Feature = ops_module.Feature  # type: ignore[attr-defined]
    module.PFeature = ops_module.PFeature  # type: ignore[attr-defined]
    module.Operators = ops_module.Operators  # type: ignore[attr-defined]

    def _get_module_logger(name: str) -> types.SimpleNamespace:
        import logging

        return logging.getLogger(name)  # type: ignore[return-value]

    module.get_module_logger = _get_module_logger  # type: ignore[attr-defined]
    setattr(module, _SHIM_MARKER, True)

    sys.modules[dotted] = module
    spec.loader.exec_module(module)
    return module


__all__ = ["QlibEvalSurfaceLoadError", "load_eval_surface_module"]
