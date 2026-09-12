"""Runtime build status — what works right now, checked rather than claimed.

Exists because "it's built" and "it runs" are different statements, and a status document that is
written by hand drifts from the code within a day. This checks.
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from typing import Any

DATA = Path(__file__).resolve().parents[2] / "data"

MODULES = (
    "argus.truth.clocks", "argus.truth.facts", "argus.cost.model",
    "argus.decision.verdicts", "argus.execution.orders", "argus.execution.bitget_client",
    "argus.execution.preflight", "argus.execution.queue", "argus.execution.passive",
    "argus.execution.latency", "argus.execution.latency_probe", "argus.execution.schedule",
    "argus.risk.hedgeability", "argus.llm.qwen",
    "argus.llm.provider", "argus.agents.meta_pm", "argus.agents.analysts", "argus.agents.desk",
    "argus.proof.autonomy", "argus.market.bitget", "argus.market.history",
    "argus.market.validation", "argus.backtest.engine", "argus.backtest.metrics",
    "argus.strategies.session_alpha", "argus.strategies.track1_suite",
    "argus.desk.workbench", "argus.paper.ledger", "argus.paper.runner",
    "argus.eval.observatory", "argus.eval.challenge", "argus.eval.bakeoff",
)

ARTEFACTS = (
    "arbitrage_study.json", "gap_study.json", "weekend_significance.json",
    "track1_study.json", "rtoken_panel.csv", "paper_ledger.jsonl", "bakeoff.json",
    "latency_probe.json",
)


# The eighteen sub-themes the three tracks name, each mapped to the symbol that implements it and
# the test file that exercises it. This is checked at runtime rather than asserted in a document,
# because a coverage table in prose is a claim and an import that resolves is evidence.
#
# Format: sub-theme -> (track, "module:symbol", test file)
SUBTHEMES: dict[str, tuple[int, str, str]] = {
    # Track 1 — Alpha Factory
    "arbitrage": (1, "argus.research.arbitrage_study:study", "test_market.py"),
    "after-hours": (1, "argus.strategies.track1_suite:afterhours_attenuation_fade",
                    "test_workbench.py"),
    "cross-market": (1, "argus.strategies.track1_suite:crossmarket_basis_reversion",
                     "test_workbench.py"),
    "rtoken-factors": (1, "argus.strategies.track1_suite:rtoken_attenuation_carry",
                       "test_workbench.py"),
    "cross-asset-rotation": (1, "argus.strategies.track1_suite:rotation_regime_switch",
                             "test_workbench.py"),
    "t1-open-execution-aware": (1, "argus.strategies.track1_suite:execution_aware_alpha",
                                "test_passive.py"),
    # Track 2 — Agentic Trading
    "event-driven": (2, "argus.agents.analysts:EventAnalyst", "test_analysts.py"),
    "sentiment": (2, "argus.agents.analysts:SentimentAnalyst", "test_analysts.py"),
    "earnings": (2, "argus.agents.earnings:decompose", "test_earnings.py"),
    "cross-asset-execution": (2, "argus.agents.analysts:CrossAssetAnalyst", "test_hedgeability.py"),
    "factor-discovery": (2, "argus.research.factor_lab:ProposerContext", "test_factor_lab.py"),
    "t2-open-evaluation": (2, "argus.eval.observatory:ModelScorecard", "test_observatory.py"),
    # Track 3 — AI Trading Desk
    "info-extraction": (3, "argus.desk.workbench:ClaimGraph", "test_workbench.py"),
    "review-self-evolution": (3, "argus.desk.workbench:ErrorProfile", "test_workbench.py"),
    "stress-testing": (3, "argus.desk.workbench:stress_position", "test_workbench.py"),
    "personal-workbench": (3, "argus.desk.workbench:TraderProfile", "test_workbench.py"),
    "execution-assistance": (3, "argus.execution.schedule:trajectory", "test_schedule.py"),
    "portfolio-copilot": (3, "argus.desk.workbench:assess_trade", "test_workbench.py"),
}

TESTS = Path(__file__).resolve().parents[2] / "tests"


def subtheme_coverage() -> dict[str, Any]:
    """Resolve every sub-theme's implementing symbol and its test file.

    A sub-theme counts as covered only when the symbol actually imports *and* the named test file
    exists. Either half missing is reported by name — a renamed function or a deleted test file
    turns a green claim red here rather than at submission.
    """
    covered: dict[str, str] = {}
    missing: dict[str, str] = {}
    for theme, (track, target, test_file) in SUBTHEMES.items():
        module_name, _, symbol = target.partition(":")
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            missing[theme] = f"import failed: {str(exc)[:80]}"
            continue
        if not hasattr(module, symbol):
            missing[theme] = f"{module_name} has no {symbol!r}"
            continue
        if not (TESTS / test_file).exists():
            missing[theme] = f"test file {test_file} is absent"
            continue
        covered[theme] = f"T{track} · {target} · {test_file}"
    return {
        "covered": f"{len(covered)}/{len(SUBTHEMES)}",
        "by_track": {
            f"track_{t}": sum(1 for k in covered if SUBTHEMES[k][0] == t) for t in (1, 2, 3)
        },
        "detail": covered,
        "uncovered": missing,
    }


def check() -> dict[str, Any]:
    importable, broken = [], {}
    for name in MODULES:
        try:
            importlib.import_module(name)
            importable.append(name)
        except Exception as exc:
            broken[name] = str(exc)[:120]

    artefacts = {
        name: (DATA / name).stat().st_size if (DATA / name).exists() else None
        for name in ARTEFACTS
    }

    from argus.execution.bitget_client import credentials_present

    return {
        "modules_importable": f"{len(importable)}/{len(MODULES)}",
        "modules_broken": broken,
        "subthemes": subtheme_coverage(),
        "artefacts_on_disk": {k: v for k, v in artefacts.items() if v},
        "artefacts_missing": [k for k, v in artefacts.items() if v is None],
        "credentials": {
            "qwen": bool(os.environ.get("BITGET_QWEN_API_KEY")),
            "nvidia": bool(os.environ.get("NVIDIA_API_KEY")),
            "bitget_trading": credentials_present(),
        },
        "blocked": (
            [] if credentials_present()
            else ["bitget demo order placement — see STATUS.md; needs a key only you can create"]
        ),
    }


def main() -> int:
    print(json.dumps(check(), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
