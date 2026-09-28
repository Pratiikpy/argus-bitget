"""The risk policy as data: every threshold the Constitution enforces, in one readable file.

`risk/constitution.ConstitutionPolicy` holds the desk's limits as dataclass defaults, each with a
long rationale. That is where they are enforced, and it is the wrong place to *read* them: a judge
asking "what are this agent's risk limits, and which are switched on?" had to read Python. Open
Policy Agent's central idea is that policy is data, versioned and inspectable apart from the
program that applies it (research/harvest/26-opa-rule-gate.md, §5 item 2). OPA's Rego engine is
not adopted — a Datalog evaluator for a dozen numeric ceilings is out of proportion, and OPA
decides booleans where the Constitution computes ceilings and takes the least — so the idea is
rebuilt as a plain manifest:

* ``data/risk_policy.json`` holds every threshold with its unit and meaning, and a digest. The
  paper runner builds its policy from this file each cycle, so editing the file changes the live
  limits, and nothing else does.
* The file must name exactly the thresholds the Constitution has — a missing or unknown key
  refuses to load rather than quietly falling back to a default — and each value is parsed to the
  field's own type.
* Each cycle also writes ``data/risk_policy_live.json``: the same thresholds, the digest they
  were loaded under, and **which gates had their input in that cycle**. A threshold whose input
  is never supplied cannot fire, and saying so is the difference between a policy and a list of
  numbers.

    python -m argus.risk.policy_manifest --write    # regenerate from the code defaults
    python -m argus.risk.policy_manifest --seal     # after editing a value: re-stamp the digest
    python -m argus.risk.policy_manifest            # does it load, and does it match the code?
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Mapping
from dataclasses import fields
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from argus.risk.constitution import ConstitutionPolicy
from argus.truth.paths import DATA_DIR

POLICY_PATH = DATA_DIR / "risk_policy.json"
LIVE_PATH = DATA_DIR / "risk_policy_live.json"
SCHEMA = 1

THRESHOLDS: dict[str, tuple[str, str]] = {
    "max_position_notional": ("USD", "largest notional one order may carry"),
    "max_unhedged_notional": ("USD", "largest exposure carried while no hedge is available"),
    "min_confidence_to_trade": ("probability", "stated confidence below which no order is sent"),
    "max_gross_exposure_notional": ("USD", "total gross notional across the book, order included"),
    "max_signed_exposure_notional": ("USD", "net long-minus-short notional across the book"),
    "max_margin_usage_ratio": ("fraction", "venue margin ratio cap; 1.0 is the liquidation line"),
    "factor_exposure_limits": ("per factor", "cap on the book's exposure to each named factor"),
    "max_scenario_loss_pct": ("percent", "floor on the book's move in the worst benchmark shock"),
    "max_liquidation_cost_bps": ("bps", "ceiling on the worst position's forced-exit slippage"),
    "assumed_payoff": ("ratio", "win/loss ratio used by Kelly sizing once calibration passes"),
    "session_horizon_bars": ("hours", "assumed holding period for the volatility throttle"),
    "min_symbol_realized_pnl": ("USD", "floor on one symbol's realised PnL over the window"),
    "symbol_underperformance_window_minutes": ("minutes", "window for the per-symbol PnL floor"),
}
"""Every threshold the Constitution enforces, with its unit and a one-line meaning. The fields not
listed are state the caller injects each cycle (prices, the book, measured risks), not policy."""

GATE_INPUTS: dict[str, str] = {
    "max_gross_exposure_notional": "book",
    "max_signed_exposure_notional": "book",
    "max_margin_usage_ratio": "book.margin",
    "factor_exposure_limits": "factor_exposures",
    "max_scenario_loss_pct": "stress_outcomes",
    "max_liquidation_cost_bps": "liquidation_cost_estimates",
    "min_symbol_realized_pnl": "book",
}
"""The injected input a limit needs before its gate can fire. A limit not listed needs only the
order itself and the reference price."""

PARAMETERS: dict[str, str] = {
    "assumed_payoff": "Kelly sizing, used only once calibration passes",
    "session_horizon_bars": "the session-volatility throttle, which needs measured session risk",
    "symbol_underperformance_window_minutes": "the per-symbol PnL floor",
}
"""Thresholds that shape a gate rather than being one: they never refuse an order themselves,
so they are listed with the mechanism they feed and are not counted among the limits."""


class PolicyError(ValueError):
    """The manifest is unreadable, incomplete, names a threshold that does not exist, or carries a
    value that is not of the threshold's type."""


def _field_types() -> dict[str, Any]:
    defaults = ConstitutionPolicy()
    return {f.name: getattr(defaults, f.name) for f in fields(ConstitutionPolicy)}


def _encode(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): float(v) for k, v in dict(value).items()}
    return value


def thresholds_of(policy: ConstitutionPolicy) -> dict[str, Any]:
    """The policy's thresholds, JSON-ready (decimals as strings, so nothing is rounded)."""
    return {name: _encode(getattr(policy, name)) for name in THRESHOLDS}


def digest(thresholds: dict[str, Any]) -> str:
    canonical = json.dumps(thresholds, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def manifest(policy: ConstitutionPolicy | None = None) -> dict[str, Any]:
    values = thresholds_of(policy or ConstitutionPolicy())
    return {
        "schema": SCHEMA,
        "digest": digest(values),
        "thresholds": {name: {"value": values[name], "unit": unit, "meaning": meaning}
                       for name, (unit, meaning) in THRESHOLDS.items()},
    }


def _parse(name: str, raw: Any, default: Any) -> Any:
    try:
        if raw is None:
            if default is not None and name != "factor_exposure_limits":
                raise PolicyError(f"{name} may not be null: the Constitution needs a value")
            return None if name != "factor_exposure_limits" else {}
        if name == "factor_exposure_limits":
            if not isinstance(raw, dict):
                raise PolicyError(f"{name} must map factor names to limits")
            return {str(k): float(v) for k, v in raw.items()}
        if isinstance(default, bool):
            raise PolicyError(f"{name}: boolean thresholds are not supported")
        if isinstance(default, int):
            if not isinstance(raw, int) or isinstance(raw, bool):
                raise PolicyError(f"{name} must be a whole number, got {raw!r}")
            return raw
        if isinstance(default, float) or (default is None and name == "max_scenario_loss_pct"):
            return float(raw)
        return Decimal(str(raw))
    except (InvalidOperation, TypeError, ValueError) as exc:
        if isinstance(exc, PolicyError):
            raise
        raise PolicyError(f"{name}: {raw!r} is not a number") from None


def load(path: Path = POLICY_PATH, *, sealed: bool = True) -> tuple[dict[str, Any], str]:
    """The thresholds as ConstitutionPolicy keyword arguments, and the digest they hash to.

    ``sealed`` requires the file's stated digest to match its values: an edit that was never
    sealed with ``--seal`` is refused, so a limit cannot change by a stray keystroke. The paper
    runner treats a refusal as a reason to decide nothing that cycle, never as a reason to fall
    back to the code defaults."""
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PolicyError(f"{path} cannot be read: {exc}") from None
    if not isinstance(blob, dict) or blob.get("schema") != SCHEMA:
        raise PolicyError(f"{path} is not a schema-{SCHEMA} risk policy")
    table = blob.get("thresholds")
    if not isinstance(table, dict):
        raise PolicyError(f"{path} has no thresholds table")
    missing = sorted(set(THRESHOLDS) - set(table))
    unknown = sorted(set(table) - set(THRESHOLDS))
    if missing or unknown:
        raise PolicyError(f"{path}: missing {missing or 'none'}, unknown {unknown or 'none'}")
    types = _field_types()
    kwargs = {}
    for name in THRESHOLDS:
        entry = table[name]
        if not isinstance(entry, dict) or "value" not in entry:
            raise PolicyError(f"{name} needs a value")
        kwargs[name] = _parse(name, entry["value"], types[name])
    values = thresholds_of(ConstitutionPolicy(**kwargs))
    stated = blob.get("digest")
    actual = digest(values)
    if sealed and stated != actual:
        raise PolicyError(f"{path}: digest {str(stated)[:12]} does not match its values "
                          f"({actual[:12]}); seal the edit with --seal")
    return kwargs, actual


def policy(path: Path = POLICY_PATH, **injected: Any) -> tuple[ConstitutionPolicy, str]:
    """The live policy: thresholds from the manifest, state from the caller."""
    kwargs, version = load(path)
    return ConstitutionPolicy(**kwargs, **injected), version


def live_record(active: ConstitutionPolicy, version: str, *, at: datetime | None = None
                ) -> dict[str, Any]:
    """What one cycle ran under: the thresholds, their digest, and which gates could fire."""
    armed = {}
    for name in THRESHOLDS:
        if name in PARAMETERS:
            continue
        value = getattr(active, name)
        needs = GATE_INPUTS.get(name)
        supplied: Any = active
        for part in (needs or "").split("."):
            supplied = getattr(supplied, part, None) if part else supplied
        has_input = needs is None or supplied not in (None, (), [])
        is_set = value is not None and value != {}
        armed[name] = {"set": is_set, "input": needs, "input_present": has_input,
                       "can_fire": is_set and has_input}
    return {"at": (at or datetime.now(UTC)).isoformat(timespec="seconds"), "digest": version,
            "thresholds": thresholds_of(active), "gates": armed,
            "can_fire": sorted(k for k, v in armed.items() if v["can_fire"]),
            "cannot_fire": sorted(k for k, v in armed.items() if not v["can_fire"])}


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI
    args = sys.argv[1:] if argv is None else argv
    if "--write" in args:
        POLICY_PATH.write_text(json.dumps(manifest(), indent=2) + "\n",
                               encoding="utf-8", newline="\n")
        print(f"written to {POLICY_PATH}")
        return 0
    if "--seal" in args:
        edited, _ = load(sealed=False)
        text = json.dumps(manifest(ConstitutionPolicy(**edited)), indent=2)
        POLICY_PATH.write_text(text + "\n", encoding="utf-8", newline="\n")
        print(f"sealed {POLICY_PATH}")
        return 0
    kwargs, version = load()
    same = thresholds_of(ConstitutionPolicy(**kwargs)) == thresholds_of(ConstitutionPolicy())
    print(f"{POLICY_PATH.name}: digest {version[:12]}, "
          f"{'equal to' if same else 'DIFFERENT from'} the code defaults")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["GATE_INPUTS", "LIVE_PATH", "PARAMETERS", "POLICY_PATH", "THRESHOLDS", "PolicyError",
           "digest",
           "live_record", "load", "manifest", "policy", "thresholds_of"]
