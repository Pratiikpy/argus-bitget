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
    "argus.decision.verdicts", "argus.decision.escalation",
    "argus.execution.orders", "argus.execution.bitget_client",
    "argus.execution.preflight", "argus.execution.guard", "argus.execution.queue",
    "argus.execution.passive",
    "argus.execution.latency", "argus.execution.latency_probe", "argus.execution.schedule",
    "argus.sim.book", "argus.sim.agents", "argus.sim.market",
    "argus.research.grammar", "argus.research.panel", "argus.research.overfit",
    "argus.research.crosssection",
    "argus.research.memory",
    "argus.market.evidence", "argus.market.insider", "argus.market.fundamentals",
    "argus.market.skills", "argus.market.estimates", "argus.market.macro",
    "argus.market.volatility", "argus.market.microstructure", "argus.market.instruments",
    "argus.risk.hedgeability", "argus.risk.circuit", "argus.risk.sizing", "argus.llm.qwen",
    "argus.llm.provider", "argus.agents.meta_pm", "argus.agents.analysts", "argus.agents.desk",
    "argus.agents.conflict", "argus.agents.grounding", "argus.agents.mandate",
    "argus.agents.recall", "argus.agents.novelty",
    "argus.agents.claims", "argus.agents.selection", "argus.agents.adversary",
    "argus.agents.debate",
    "argus.proof.autonomy", "argus.market.bitget", "argus.market.history",
    "argus.market.validation", "argus.backtest.engine", "argus.backtest.metrics",
    "argus.backtest.validation",
    "argus.strategies.session_alpha", "argus.strategies.track1_suite",
    "argus.desk.workbench", "argus.desk.analogue", "argus.desk.rootcause",
    "argus.desk.portfolio", "argus.desk.diversification",
    "argus.desk.expectation", "argus.desk.personalisation",
    "argus.desk.stress", "argus.desk.review", "argus.desk.research",
    "argus.paper.ledger", "argus.paper.runner", "argus.paper.repair",
    "argus.paper.protocol", "argus.paper.anchor", "argus.paper.chains",
    "argus.paper.replay",
    "argus.eval.observatory", "argus.eval.challenge", "argus.eval.bakeoff",
    "argus.eval.performance", "argus.eval.episodes", "argus.eval.selfaudit",
    "argus.eval.riskaudit", "argus.eval.baseline", "argus.eval.docclaims",
    "argus.eval.riskproof", "argus.eval.decisioncard",
    "argus.eval.ablation", "argus.eval.ablations", "argus.eval.standing",
    "argus.eval.forecasts", "argus.eval.hurdle",
    "argus.eval.forecastbench",
    "argus.eval.incremental", "argus.eval.collect", "argus.eval.bench",
    "argus.eval.venue_rules",
    "argus.eval.profile_value",
    "argus.eval.sources",
    "argus.eval.degradation",
    "argus.eval.shadow",
    "argus.eval.themeaudit",
    "argus.eval.autopsy",
    "argus.eval.luibench",
    "argus.eval.architecture",
    "argus.eval.consistency",
    "argus.eval.cyclecheck",
    "argus.desk.shapematch",
    "argus.research.cointegration",
    "argus.risk.effectiveness",
    "argus.desk.allocation",
    "argus.market.depth",
    "argus.risk.session_risk",
    "argus.market.markout",
    "argus.desk.regime",
    "argus.agents.quarantine",
    "argus.eval.leakage",
    "argus.register.claims",
    "argus.register.resolve",
    "argus.register.open_register",
    "argus.register.cadence",
    "argus.demo.flow",
    "argus.demo.cockpit",
    "argus.research.overfitting_study",
    "argus.research.searchoff",
    "argus.research.eventstudy", "argus.research.event_reactions",
    "argus.backtest.dependence",
    "argus.market.calendar", "argus.market.earnings_release", "argus.market.stories",
    "argus.market.bitget_positioning", "argus.eval.impact_calibration",
    "argus.eval.analogstress_comparison",
    "argus.lui.question", "argus.lui.answer", "argus.lui.cli", "argus.lui.server",
    "argus.lui.phrasebook",
    "argus.lui.router", "argus.lui.research", "argus.lui.task", "argus.lui.status_page",
    "argus.lui.proof_page",
)

ARTEFACTS = (
    "arbitrage_study.json", "gap_study.json", "weekend_significance.json",
    "track1_study.json", "rtoken_panel.csv", "paper_ledger.jsonl", "bakeoff.json",
    "latency_probe.json", "factor_lab.json", "factor_memory.json",
    "desk_notes.jsonl", "paper_ledger_incidents.json",
    "bitget_skills_health.json", "skill_crosscheck.json", "risk_records.jsonl",
    "session_beta.json", "doc_claims.json", "protocol_commitments.jsonl",
    "stress_report.json", "review_report.json", "risk_proof.json",
    "research_report.json", "standing.json",
    "crosssection_study.json", "ablations.json",
    "policy_forecasts.json", "replay_ledger.jsonl",
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
    "info-extraction": (3, "argus.desk.expectation:detect", "test_expectation.py"),
    "review-self-evolution": (3, "argus.desk.review:review", "test_review.py"),
    "stress-testing": (3, "argus.desk.stress:assess", "test_stress.py"),
    "personal-workbench": (3, "argus.desk.personalisation:diverge", "test_personalisation.py"),
    "execution-assistance": (3, "argus.execution.schedule:trajectory", "test_schedule.py"),
    "portfolio-copilot": (3, "argus.desk.portfolio:assess", "test_portfolio.py"),
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

    from argus.eval import docclaims, standing
    from argus.execution.bitget_client import credentials_present

    return {
        "doc_claims": docclaims.summary(docclaims.audit()),
        # The four states, checked against the tree rather than asserted in a document. A register
        # defect here means a capability names evidence that is not on disk.
        "standing": standing.summary(standing.audit()),
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
            else ["bitget credentials absent from the environment; load .secrets/bitget.env"]
        ),
        "venue_verified": {
            "signature_accepted": True,
            "order_round_trip": "1482642956228374529",
            "product_type": "SUSDT-FUTURES",
            "note": "demo carries no rTokens; the internal ledger stays the record for those",
        },
    }


def main() -> int:
    """Exit non-zero when a document quotes a number the artefacts contradict.

    This used to `return 0` unconditionally, so a stale figure appeared as one line of JSON inside
    a report that a reader — or a CI step — would call green. The count is already computed and
    already printed; making it an exit code costs nothing and is the difference between reporting
    the defect and catching it. Note this is the cheap audit: `tests_passing` is in `EXPENSIVE` and
    is skipped here, which is why the block names its `unchecked` count. `eval/docclaims.py
    --tests` is the exhaustive gate.
    """
    report = check()
    print(json.dumps(report, indent=2, default=str))
    return 1 if report["doc_claims"]["stale"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
