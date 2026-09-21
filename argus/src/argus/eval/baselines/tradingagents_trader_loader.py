"""Make the vendored `tradingagents_trader.py` (TradingAgents' real `TraderProposal`) importable.

Registered in `sys.modules` before execution (`register=True`), matching every other loader in
this package — `TraderProposal`'s fields use `from __future__ import annotations` (string
annotations), and pydantic resolves those against `sys.modules[cls.__module__].__dict__` on
first validation; a module never registered under its own name resolves against nothing and
raises `PydanticUndefinedAnnotation` on the very first real construction.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_BASELINES_DIR = Path(__file__).resolve().parent
_SHIM_MARKER = "_argus_tradingagents_trader_shim"
_DOTTED = "argus_baseline_tradingagents_trader"


class TradingAgentsTraderLoadError(RuntimeError):
    """The vendored TradingAgents `TraderProposal` excerpt could not be loaded."""


def load_trader_module() -> ModuleType:
    """Load `tradingagents_trader.py`. Carries `TraderAction`, `TraderProposal`,
    `render_trader_proposal` — construct `.TraderProposal(action=..., reasoning=..., ...)`
    exactly as TradingAgents' real `trader_node` does via its structured-output call."""
    existing = sys.modules.get(_DOTTED)
    if existing is not None:
        if not getattr(existing, _SHIM_MARKER, False):
            raise TradingAgentsTraderLoadError(
                f"sys.modules[{_DOTTED!r}] already holds a module this loader did not create"
            )
        return existing

    path = _BASELINES_DIR / "tradingagents_trader.py"
    spec = importlib.util.spec_from_file_location(_DOTTED, path)
    if spec is None or spec.loader is None:
        raise TradingAgentsTraderLoadError(f"could not build an import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    setattr(module, _SHIM_MARKER, True)
    sys.modules[_DOTTED] = module
    spec.loader.exec_module(module)
    return module


__all__ = ["TradingAgentsTraderLoadError", "load_trader_module"]
