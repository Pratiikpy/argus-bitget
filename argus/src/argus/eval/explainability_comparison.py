"""Same-input comparison: `argus.agents.grounding.check` vs TradingAgents' real, unmodified
`TraderProposal` — the T2 judging criterion "decision explainability" put to a concrete test:
when a trading decision states a number, does anything check it came from somewhere real?

Runs TradingAgents' real, vendored `TraderProposal` Pydantic model
(`eval/baselines/tradingagents_trader.py`, `agents/schemas.py` upstream — executed unmodified via
`eval/baselines/tradingagents_trader_loader.py`) and ARGUS's real `agents.grounding.check` on the
same real facts and the same constructed, out-of-range numbers.

**The central, measured finding.** TradingAgents' Trader prompt instructs the model to "ground
concrete price levels... in the technical market report's price structure" (`agents/trader/
trader.py:36-39`), but nothing downstream ever checks that it did: `TraderProposal`'s own real
`field_validator` (`_nullish_float_to_none` / `_coerce_optional_float`) only normalises STRING
FORMAT — placeholder text, a trailing "%", a currency symbol — never the VALUE, and reading every
real consumer of `trader_investment_plan` (`portfolio_manager.py`, all three `risk_mgmt/*.py`
debators, `reporting.py`, `trading_graph.py`) finds none that compares `entry_price`/`stop_loss`
against `market_report`'s real figures. A price with no relationship to any fact the desk was
given validates and flows downstream exactly like a correct one. ARGUS's `grounding.check`
resolves every number in a thesis against the real facts the desk actually had, within a stated
tolerance, and reports precisely which ones do not resolve — run here on the same real market
data and the same fabricated figures, it flags every one.

SCOPE, stated explicitly:

* This tests one specific, narrow property — whether a stated price is traceable to a real fact —
  not whether TradingAgents' overall system ever catches bad output by other means (a human
  reading the saved markdown report, a downstream risk check this comparison does not exercise).
  No claim is made about TradingAgents' full pipeline; the claim is scoped to the real code path
  read here: `TraderProposal`'s own validator and every real consumer of its output.
* `TraderProposal.reasoning` (free text) is not checked here — only `entry_price`/`stop_loss`,
  the two numeric fields the schema itself defines and the ones `trader.py`'s own grounding
  instruction names. TradingAgents may catch numeric hallucination in prose elsewhere; this
  comparison only tests the structured fields its own schema makes checkable.
* ARGUS's `grounding.check` proves a figure is traceable to a known value, never that the value
  itself is correct — a correctly-cited but wrong fact would still resolve. That is a narrower,
  checkable property, stated in `grounding.py`'s own module docstring, not claimed as more here.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from argus.agents.grounding import check as grounding_check
from argus.eval.baselines.tradingagents_trader_loader import load_trader_module
from argus.market.bitget import fetch_tickers

REAL_SYMBOLS: tuple[str, ...] = ("NVDAUSDT", "AAPLUSDT", "MSFTUSDT")

# Multiplicative factors applied to a real current price to build a figure with no relationship
# to any fact the desk was given — several magnitudes and directions, not one hand-picked case.
HALLUCINATION_FACTORS: tuple[float, ...] = (7.3, 0.03, -1.0, 1000.0)


def _real_facts(symbol: str) -> dict[str, float]:
    """Real facts the desk actually had for `symbol`: current price and the real 24h
    high/low this venue reports — the same class of figure TradingAgents' own `market_report`
    names (current price, support/resistance)."""
    ticker = fetch_tickers()[symbol]
    return {
        "current_price": float(ticker.last),
        "resistance_24h": float(ticker.high_24h),
        "support_24h": float(ticker.low_24h),
    }


def run_baseline_reproduced() -> dict[str, Any]:
    """TradingAgents' real, unmodified `TraderProposal` validates a wildly fabricated
    `entry_price` (no relationship to any real fact) without error — proves the schema's own
    real validator performs no value-level check, only string-format normalisation."""
    mod = load_trader_module()
    proposal = mod.TraderProposal(
        action=mod.TraderAction.BUY,
        reasoning="ATR breakout confirms bullish continuation near the stated level.",
        entry_price=999_999.0,
        stop_loss=1.0,
    )
    return {
        "real_entry_price_validated": proposal.entry_price,
        "real_stop_loss_validated": proposal.stop_loss,
        "real_validator_raised": False,
        "real_rendered": mod.render_trader_proposal(proposal),
    }


def run_same_input_comparison() -> dict[str, Any]:
    """The measured comparison: for each real symbol and each fabricated entry_price, run
    TradingAgents' real `TraderProposal` validator and ARGUS's real `grounding.check` on the
    same real facts and the same fabricated number."""
    mod = load_trader_module()
    cases: list[dict[str, Any]] = []
    for symbol in REAL_SYMBOLS:
        facts = _real_facts(symbol)
        for factor in HALLUCINATION_FACTORS:
            fabricated = round(facts["current_price"] * factor, 4)
            ta_raised = False
            try:
                mod.TraderProposal(
                    action=mod.TraderAction.BUY,
                    reasoning=f"Entry near {fabricated} on {symbol}.",
                    entry_price=fabricated,
                )
            except Exception:  # recording whether the real validator raises at all
                ta_raised = True

            thesis = f"Entry near {fabricated} on {symbol}."
            report = grounding_check(thesis, facts=facts)
            argus_flags_it = any(
                r.figure.value == fabricated and not r.resolved for r in report.resolutions
            )
            cases.append({
                "symbol": symbol,
                "factor": factor,
                "fabricated_entry_price": fabricated,
                "real_current_price": facts["current_price"],
                "tradingagents_real_validator_raised": ta_raised,
                "argus_grounding_flags_it": argus_flags_it,
            })

    n = len(cases)
    n_ta_catches = sum(1 for c in cases if c["tradingagents_real_validator_raised"])
    n_argus_catches = sum(1 for c in cases if c["argus_grounding_flags_it"])
    return {
        "cases": cases,
        "n_cases": n,
        "n_tradingagents_catches": n_ta_catches,
        "n_argus_catches": n_argus_catches,
    }


def run_positive_control() -> dict[str, Any]:
    """The mechanism is not a blanket flag: a thesis stating the REAL current price must
    resolve. Checked on the same real symbols, so a pass here is not free."""
    results = []
    for symbol in REAL_SYMBOLS:
        facts = _real_facts(symbol)
        thesis = f"Entry near {facts['current_price']} on {symbol}."
        report = grounding_check(thesis, facts=facts)
        results.append({
            "symbol": symbol,
            "real_price": facts["current_price"],
            "argus_resolves_the_real_figure": report.grounded,
        })
    return {
        "results": results,
        "all_real_figures_resolve": all(r["argus_resolves_the_real_figure"] for r in results),
    }


def run_ablation() -> dict[str, Any]:
    """Isolates the exact mechanism `grounding.check` relies on: a figure just inside
    `TOLERANCE` (a model's honest paraphrase, e.g. "189.5" for a computed 189.4991) resolves; the
    same figure moved just past `TOLERANCE` does not, on the same real fact."""
    from argus.agents.grounding import TOLERANCE

    real_value = 189.4991
    facts = {"current_price": real_value}

    just_inside = round(real_value * (1 + TOLERANCE * 0.5), 4)
    inside_report = grounding_check(f"Entry near {just_inside}.", facts=facts)
    inside_resolved = any(
        r.figure.value == just_inside and r.resolved for r in inside_report.resolutions
    )

    just_outside = round(real_value * (1 + TOLERANCE * 3), 4)
    outside_report = grounding_check(f"Entry near {just_outside}.", facts=facts)
    outside_resolved = any(
        r.figure.value == just_outside and r.resolved for r in outside_report.resolutions
    )

    return {
        "real_value": real_value,
        "just_inside_tolerance": just_inside,
        "just_inside_resolves": inside_resolved,
        "just_outside_tolerance": just_outside,
        "just_outside_resolves": outside_resolved,
        "tolerance_is_the_load_bearing_boundary": inside_resolved and not outside_resolved,
    }


def run_failure_cases() -> dict[str, Any]:
    """Real, measured behaviour of the real `TraderProposal` validator on edge-case input, found
    by running it, not assumed."""
    mod = load_trader_module()
    findings: dict[str, Any] = {}

    negative_price = mod.TraderProposal(
        action=mod.TraderAction.SELL, reasoning="Short entry.", entry_price=-50.0,
    )
    findings["negative_price"] = {
        "real_value_accepted": negative_price.entry_price,
        "real_validator_raised": False,
    }

    percent_string = mod.TraderProposal(
        action=mod.TraderAction.BUY, reasoning="Percent distance stated instead of a price.",
        entry_price="15%",
    )
    findings["percent_string_dropped_to_none"] = {
        "real_input": "15%",
        "real_result": percent_string.entry_price,
        "real_dropped_to_none_not_converted": percent_string.entry_price is None,
    }

    placeholder_string = mod.TraderProposal(
        action=mod.TraderAction.HOLD, reasoning="No specific level stated.", entry_price="N/A",
    )
    findings["placeholder_string_dropped_to_none"] = {
        "real_input": "N/A",
        "real_result": placeholder_string.entry_price,
    }
    return findings


def measure_costs(repeats: int = 200) -> dict[str, Any]:
    """Real wall-clock cost of both real, pure-Python checks on the same case — no subprocess
    or network call in either path once the real facts are already fetched."""
    facts = {"current_price": 189.5, "resistance_24h": 195.0, "support_24h": 182.0}
    mod = load_trader_module()
    thesis = "Entry near 999999.0 on NVDAUSDT."

    start = time.perf_counter()
    for _ in range(repeats):
        mod.TraderProposal(action=mod.TraderAction.BUY, reasoning="x", entry_price=999_999.0)
    ta_elapsed = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(repeats):
        grounding_check(thesis, facts=facts)
    argus_elapsed = time.perf_counter() - start

    return {
        "repeats": repeats,
        "tradingagents_validator_seconds_per_call": ta_elapsed / repeats,
        "argus_grounding_seconds_per_call": argus_elapsed / repeats,
    }


def run_reproducibility_check() -> dict[str, Any]:
    facts = {"current_price": 189.5, "resistance_24h": 195.0, "support_24h": 182.0}
    thesis = "Entry near 999999.0 on NVDAUSDT."
    first = grounding_check(thesis, facts=facts).grounded
    second = grounding_check(thesis, facts=facts).grounded
    return {"identical": first == second}


SCOPE_STATEMENT = (
    "TradingAgents' real, unmodified TraderProposal (agents/schemas.py) validates a fabricated "
    "entry_price with no relationship to any real fact the desk was given, in every case tested "
    "across three real symbols and four magnitudes/directions -- its field_validator only "
    "normalises string FORMAT (placeholder text, a trailing '%', a currency symbol), never the "
    "VALUE, and no downstream consumer (portfolio_manager.py, the three risk_mgmt debators, "
    "reporting.py, trading_graph.py -- all read directly) compares entry_price/stop_loss against "
    "market_report's real figures. ARGUS's real grounding.check, run on the same real facts and "
    "the same fabricated figures, flags every one, and a positive control on the same real "
    "symbols confirms it is not a blanket flag -- the real current price resolves cleanly. NOT "
    "claimed TradingAgents' full pipeline never catches bad output by any means -- only that the "
    "one real, checkable code path (the schema's own validator plus every real consumer of its "
    "output) performs no value-level check. NOT claimed about TraderProposal.reasoning (free "
    "text) -- only the two numeric fields the schema itself defines. NOT claimed ARGUS's "
    "mechanism proves a figure is CORRECT, only that it is traceable to a known value -- stated "
    "in grounding.py's own module docstring, not stronger here."
)


def main() -> int:  # pragma: no cover - CLI
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    report = {
        "baseline_reproduced": run_baseline_reproduced(),
        "same_input_comparison": run_same_input_comparison(),
        "positive_control": run_positive_control(),
        "ablation": run_ablation(),
        "failure_cases": run_failure_cases(),
        "costs": measure_costs(),
        "reproducibility": run_reproducibility_check(),
        "scope_statement": SCOPE_STATEMENT,
    }
    print(render(report))
    out = Path(__file__).resolve().parents[3] / "data" / "explainability_comparison.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nwritten to {out}")
    return 0


def render(report: dict[str, Any]) -> str:
    same = report["same_input_comparison"]
    control = report["positive_control"]
    costs = report["costs"]
    lines = ["DECISION EXPLAINABILITY vs TradingAgents' real TraderProposal\n"]
    lines.append(
        f"  {same['n_cases']} fabricated-figure case(s) across {len(REAL_SYMBOLS)} real "
        f"symbols: TradingAgents catches {same['n_tradingagents_catches']}/{same['n_cases']}, "
        f"ARGUS grounding catches {same['n_argus_catches']}/{same['n_cases']}"
    )
    lines.append(f"  positive control (real prices resolve): {control['all_real_figures_resolve']}")
    lines.append(
        f"  costs: tradingagents validator "
        f"{costs['tradingagents_validator_seconds_per_call']*1e6:.1f}us vs argus grounding "
        f"{costs['argus_grounding_seconds_per_call']*1e6:.1f}us per call"
    )
    lines.append(f"  reproducible: {report['reproducibility']['identical']}")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = [
    "HALLUCINATION_FACTORS",
    "REAL_SYMBOLS",
    "SCOPE_STATEMENT",
    "main",
    "measure_costs",
    "render",
    "run_ablation",
    "run_baseline_reproduced",
    "run_failure_cases",
    "run_positive_control",
    "run_reproducibility_check",
    "run_same_input_comparison",
]
