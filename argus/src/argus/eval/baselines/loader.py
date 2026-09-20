"""Make the vendored Vibe-Trading files importable without editing a line of them.

``vibe_trading_enforcement.py`` contains, verbatim, the line ``from src.live.mandate.model
import (AssetClass, InstrumentType, Mandate)`` — correct inside Vibe-Trading's own repo layout
(``agent/`` as the import root), meaningless here. The alternative to this loader is editing that
import line, which would mean the vendored copy is no longer what `baselines/__init__.py` and
`vibe_trading_enforcement.py`'s own header claim it is: unmodified. Instead this module registers
the vendored model file in :data:`sys.modules` under the exact dotted name the enforcement file
expects, so Python's own import machinery resolves it and the vendored file never needs to know
it is not running inside Vibe-Trading's repo.

Failing loudly on a name collision (rather than silently reusing whatever ``sys.modules["src"]``
already holds) matches the fail-closed posture of the code being loaded.
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
_SHIM_MARKER = "_argus_vibe_trading_baseline_shim"


class BaselineLoadError(RuntimeError):
    """The vendored baseline could not be loaded, or a name it needs is already taken."""


def _load_module_from_file(dotted_name: str, path: Path, *, register: bool = False) -> ModuleType:
    """Execute ``path`` as a module named ``dotted_name``.

    ``register=True`` puts the module into :data:`sys.modules` *before* executing its body, which
    ``dataclasses`` needs: its ``@dataclass`` decorator resolves ``sys.modules[cls.__module__]``
    while the class body is still executing, so the entry must exist first, not after
    ``exec_module`` returns. Both vendored files define frozen dataclasses, so both need this.
    """
    spec = importlib.util.spec_from_file_location(dotted_name, path)
    if spec is None or spec.loader is None:
        raise BaselineLoadError(f"could not build an import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    if register:
        sys.modules[dotted_name] = module
    spec.loader.exec_module(module)
    return module


def _shim_package(dotted_name: str) -> ModuleType:
    """Return a placeholder package module registered in ``sys.modules`` under ``dotted_name``.

    Reuses one we registered ourselves on a prior call (idempotent); refuses to touch a
    same-named module we did not create, so this never silently shadows something unrelated.
    """
    existing = sys.modules.get(dotted_name)
    if existing is not None:
        if not getattr(existing, _SHIM_MARKER, False):
            raise BaselineLoadError(
                f"sys.modules[{dotted_name!r}] already holds a module this loader did not "
                f"create; refusing to overwrite it (fail-closed, matching the baseline's own "
                f"posture)"
            )
        return existing
    module = types.ModuleType(dotted_name)
    setattr(module, _SHIM_MARKER, True)
    sys.modules[dotted_name] = module
    return module


def load_enforcement_module() -> ModuleType:
    """Load the vendored ``enforcement.py`` with its own unedited imports satisfied.

    Registers ``src`` / ``src.live`` / ``src.live.mandate`` / ``src.live.mandate.model`` in
    :data:`sys.modules` (the last pointing at our vendored, byte-verified copy of Vibe-Trading's
    model file) before executing the vendored enforcement file, so its verbatim
    ``from src.live.mandate.model import (...)`` line resolves exactly as it does upstream.

    Returns:
        The executed module object — carries ``check_mandate``, ``OrderIntent``,
        ``BreachEvent``, ``BREACH_KIND_*`` and every other top-level name from the vendored file.
    """
    for name in ("src", "src.live", "src.live.mandate"):
        _shim_package(name)

    model_dotted = "src.live.mandate.model"
    if model_dotted not in sys.modules or not getattr(
        sys.modules[model_dotted], _SHIM_MARKER, False
    ):
        model_module = _load_module_from_file(
            model_dotted, _BASELINES_DIR / "vibe_trading_mandate_model.py", register=True
        )
        setattr(model_module, _SHIM_MARKER, True)

    # Same idempotency reasoning as the model module just above: without this check, a second
    # `load_baseline()` call in the same process (e.g. two test files each calling it) would
    # re-execute and re-register a second, distinct enforcement module — same behaviour, but a
    # different object identity every time, which is surprising for callers that cache what this
    # returns and breaks an `is`-identity check across calls (caught by
    # `test_loading_twice_in_one_process_is_idempotent`).
    enforcement_dotted = "argus_baseline_vibe_trading_enforcement"
    if enforcement_dotted not in sys.modules or not getattr(
        sys.modules[enforcement_dotted], _SHIM_MARKER, False
    ):
        enforcement_module = _load_module_from_file(
            enforcement_dotted, _BASELINES_DIR / "vibe_trading_enforcement.py", register=True
        )
        setattr(enforcement_module, _SHIM_MARKER, True)

    return sys.modules[enforcement_dotted]


class VibeTradingBaseline(Protocol):
    """The subset of the vendored module's surface this comparison actually calls.

    A ``Protocol``, not a re-export of the dataclasses themselves, because the vendored types are
    defined at runtime by :func:`load_enforcement_module` / :func:`load_model_module` — mypy
    cannot see them statically. Callers get real static checking against this shape; the loader
    functions return ``ModuleType`` (unavoidably dynamic) and callers narrow through here.
    """

    def check_mandate(self, *args: Any, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class VibeTradingSymbols:
    """Typed handle onto the vendored module's callable/constructible names.

    Built once by :func:`load_baseline` so the rest of ``mandate_comparison.py`` never touches
    ``sys.modules`` or ``getattr`` directly.
    """

    check_mandate: Any
    order_intent: Any
    breach_event: Any
    breach_kind_universe: str
    breach_kind_instrument: str
    breach_kind_quantitative: str
    mandate: Any
    hard_caps: Any
    universe_constraint: Any
    consent_meta: Any
    instrument_type: Any
    asset_class: Any


def load_baseline() -> VibeTradingSymbols:
    """Load both vendored files and return every name ``mandate_comparison.py`` needs.

    Raises:
        BaselineLoadError: A name this loader needs was already taken by something else, or the
            vendored files could not be executed (e.g. moved or corrupted).
    """
    try:
        enforcement = load_enforcement_module()
    except Exception as exc:  # pragma: no cover - exercised via BaselineLoadError subtype tests
        if isinstance(exc, BaselineLoadError):
            raise
        raise BaselineLoadError(f"vendored baseline failed to load: {exc}") from exc

    model = sys.modules["src.live.mandate.model"]
    return VibeTradingSymbols(
        check_mandate=enforcement.check_mandate,
        order_intent=enforcement.OrderIntent,
        breach_event=enforcement.BreachEvent,
        breach_kind_universe=enforcement.BREACH_KIND_UNIVERSE,
        breach_kind_instrument=enforcement.BREACH_KIND_INSTRUMENT,
        breach_kind_quantitative=enforcement.BREACH_KIND_QUANTITATIVE,
        mandate=model.Mandate,
        hard_caps=model.HardCaps,
        universe_constraint=model.UniverseConstraint,
        consent_meta=model.ConsentMeta,
        instrument_type=model.InstrumentType,
        asset_class=model.AssetClass,
    )


__all__ = ["BaselineLoadError", "VibeTradingBaseline", "VibeTradingSymbols", "load_baseline"]
