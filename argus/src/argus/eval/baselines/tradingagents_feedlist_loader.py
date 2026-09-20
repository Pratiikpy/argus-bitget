"""Make the vendored TradingAgents data-vendor router importable without editing a line of it.

``tradingagents_feedlist_interface.py`` contains, verbatim, seven sibling imports
(``from .alpha_vantage import ...``, ``from .config import get_config``, ``from .errors import
...``, ``from .fred import ...``, ``from .polymarket import ...``, ``from .y_finance import
...``, ``from .yfinance_news import ...``) — correct inside TradingAgents' own package layout
(``tradingagents.dataflows`` as a real package), meaningless here.

Two of those siblings are real, vendored dependencies of the code under comparison
(``config.py``, which resolves the vendor chain; ``errors.py``, whose exception classes
``route_to_vendor()`` actually catches) and are loaded for real. The other five
(``alpha_vantage``, ``fred``, ``polymarket``, ``y_finance``, ``yfinance_news``) are the actual
network-calling vendor implementations — never exercised by this comparison, which tests
``route_to_vendor()``'s own fallback/dispatch LOGIC by substituting synthetic vendor functions
directly into its real ``VENDOR_METHODS`` dict at runtime (exactly how a real caller configures a
vendor). They are stubbed here with placeholder functions under the exact names ``interface.py``
imports, each raising ``NotImplementedError`` if ever actually called — fail loud, not silent, the
same posture used for every other baseline's unused shim surface in this package.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import ModuleType
from typing import Any

_BASELINES_DIR = Path(__file__).resolve().parent
_SHIM_MARKER = "_argus_tradingagents_feedlist_shim"

# The bare "tradingagents" top-level package is also shimmed by `tradingagents_loader.py` (the
# episodic-memory baseline, a different subpackage of the same real repo:
# `tradingagents.agents.utils`). A loader-private marker on that one shared name would make
# whichever loader runs second see "a module I did not create" and refuse it, even though both
# loaders' placeholder is equally empty and equally harmless. This marker is therefore shared by
# name (duplicated as a literal in each file, not imported, to keep each loader independently
# readable) — every other shim this file creates below keeps its own private `_SHIM_MARKER`,
# since `tradingagents.dataflows.*` is not shared with any other loader.
_SHARED_TOP_LEVEL_MARKER = "_argus_tradingagents_toplevel_shim"

_STUB_VENDOR_FUNCTIONS: dict[str, tuple[str, ...]] = {
    "alpha_vantage": (
        "get_balance_sheet", "get_cashflow", "get_fundamentals", "get_global_news",
        "get_income_statement", "get_indicator", "get_insider_transactions", "get_news",
        "get_stock",
    ),
    "fred": ("get_macro_data",),
    "polymarket": ("get_prediction_markets",),
    "y_finance": (
        "get_balance_sheet", "get_cashflow", "get_fundamentals", "get_income_statement",
        "get_insider_transactions", "get_stock_stats_indicators_window", "get_YFin_data_online",
    ),
    "yfinance_news": ("get_global_news_yfinance", "get_news_yfinance"),
}


class TradingAgentsFeedlistLoadError(RuntimeError):
    """The vendored TradingAgents feed-list router could not be loaded, or a name it needs is
    already taken."""


def _load_module_from_file(dotted_name: str, path: Path, *, register: bool = False) -> ModuleType:
    spec = importlib.util.spec_from_file_location(dotted_name, path)
    if spec is None or spec.loader is None:
        raise TradingAgentsFeedlistLoadError(f"could not build an import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    if register:
        sys.modules[dotted_name] = module
    spec.loader.exec_module(module)
    return module


def _shim_package(dotted_name: str, *, marker: str = _SHIM_MARKER) -> ModuleType:
    existing = sys.modules.get(dotted_name)
    if existing is not None:
        if not getattr(existing, marker, False):
            raise TradingAgentsFeedlistLoadError(
                f"sys.modules[{dotted_name!r}] already holds a module this loader did not "
                f"create; refusing to overwrite it (fail-closed, matching the vendored code's "
                f"own posture)"
            )
        return existing
    module = types.ModuleType(dotted_name)
    setattr(module, marker, True)
    sys.modules[dotted_name] = module
    return module


def _make_unreachable(name: str) -> object:
    def _unreachable(*args: object, **kwargs: object) -> object:
        raise NotImplementedError(
            f"{name} is a stub for a real network vendor call — this comparison substitutes "
            f"its own synthetic vendor functions directly into VENDOR_METHODS and never reaches "
            f"the stub; being called means a test forgot to substitute it"
        )

    _unreachable.__name__ = name
    return _unreachable


def _shim_vendor_module(dotted_name: str, function_names: tuple[str, ...]) -> ModuleType:
    existing = sys.modules.get(dotted_name)
    if existing is not None and getattr(existing, _SHIM_MARKER, False):
        return existing
    if existing is not None:
        raise TradingAgentsFeedlistLoadError(
            f"sys.modules[{dotted_name!r}] already holds a module this loader did not create"
        )
    module = types.ModuleType(dotted_name)
    setattr(module, _SHIM_MARKER, True)
    for fname in function_names:
        setattr(module, fname, _make_unreachable(fname))
    sys.modules[dotted_name] = module
    return module


def load_interface_module() -> ModuleType:
    """Load the vendored ``interface.py`` with every one of its own unedited imports satisfied.

    Returns:
        The executed module object — carries ``route_to_vendor``, ``VENDOR_METHODS``,
        ``TOOLS_CATEGORIES``, ``OPTIONAL_CATEGORIES``, ``get_category_for_method``, ``get_vendor``.
    """
    _shim_package("tradingagents", marker=_SHARED_TOP_LEVEL_MARKER)
    _shim_package("tradingagents.dataflows")

    default_config_dotted = "tradingagents.default_config"
    if default_config_dotted not in sys.modules or not getattr(
        sys.modules[default_config_dotted], _SHIM_MARKER, False
    ):
        dc_module = _load_module_from_file(
            default_config_dotted,
            _BASELINES_DIR / "tradingagents_feedlist_default_config.py",
            register=True,
        )
        setattr(dc_module, _SHIM_MARKER, True)

    config_dotted = "tradingagents.dataflows.config"
    if config_dotted not in sys.modules or not getattr(
        sys.modules[config_dotted], _SHIM_MARKER, False
    ):
        config_module = _load_module_from_file(
            config_dotted, _BASELINES_DIR / "tradingagents_feedlist_config.py", register=True
        )
        setattr(config_module, _SHIM_MARKER, True)

    errors_dotted = "tradingagents.dataflows.errors"
    if errors_dotted not in sys.modules or not getattr(
        sys.modules[errors_dotted], _SHIM_MARKER, False
    ):
        errors_module = _load_module_from_file(
            errors_dotted, _BASELINES_DIR / "tradingagents_feedlist_errors.py", register=True
        )
        setattr(errors_module, _SHIM_MARKER, True)

    for vendor_name, fn_names in _STUB_VENDOR_FUNCTIONS.items():
        _shim_vendor_module(f"tradingagents.dataflows.{vendor_name}", fn_names)

    interface_dotted = "tradingagents.dataflows.interface"
    if interface_dotted in sys.modules and getattr(
        sys.modules[interface_dotted], _SHIM_MARKER, False
    ):
        return sys.modules[interface_dotted]

    module = _load_module_from_file(
        interface_dotted, _BASELINES_DIR / "tradingagents_feedlist_interface.py", register=True
    )
    setattr(module, _SHIM_MARKER, True)
    return module


def load_set_config() -> Any:
    """The real, vendored `set_config()` — returned as a callable rather than requiring callers
    to `import tradingagents.dataflows.config` directly, since that dotted path exists only as a
    runtime `sys.modules` shim this loader creates, not a real package on disk mypy can resolve."""
    load_interface_module()  # ensures tradingagents.dataflows.config is registered
    config_module = sys.modules["tradingagents.dataflows.config"]
    result: Any = config_module.set_config
    return result


__all__ = ["TradingAgentsFeedlistLoadError", "load_interface_module", "load_set_config"]
