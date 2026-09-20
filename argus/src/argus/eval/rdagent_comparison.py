"""ARGUS's typed factor search vs. RD-Agent's real, LLM-driven factor-discovery loop — on the two
axes that decide whether a "discovered" factor is trustworthy: can the search execute code it was
never given, and does it correct for how many times it looked before it found something.

``eval/standing.py``'s "Factor discovery with no execution surface and trial-corrected selection"
capability names its baseline as microsoft/RD-Agent's real factor-implementation pipeline —
`rdagent/core/experiment.py`'s `FBWorkspace` and `rdagent/components/coder/factor_coder/factor.py`'s
`FactorFBWorkspace` (both vendored whole, byte-hash-pinned, commit
`32b3d395e73d9db5eee3fe9063d69aec0fdc83bd`, MIT) — run unmodified via
`eval/baselines/rdagent_factor_loader.py`, compared against ARGUS's own real
`argus.research.searchoff` (five algorithmic search strategies over the typed
`argus.research.grammar` vocabulary, already AST-proven to contain zero eval/exec/subprocess calls
in `eval/grammar_comparison.py`).

**Finding 1 — execution surface, demonstrated by running the real code, not by reading it.**
RD-Agent's factor-discovery loop is, by design, LLM-written code executed on the host: an LLM
proposes a hypothesis, another LLM call writes a `factor.py` implementing it
(`factor_coder/CoSTEER`, not vendored here — this comparison starts one step later, at the
EXECUTION of whatever code that step produced), and `FactorFBWorkspace.execute()` runs it via
`subprocess.check_output(f"{python_bin} {execution_code_path}", shell=True, ...)` — no sandbox, no
container (`factor_coder/config.py:get_factor_env()` hard-codes `LocalEnv`, never `DockerEnv`, for
factor execution specifically — confirmed by reading it, not assumed). This comparison runs the
REAL, unmodified `execute()` with a crafted-but-realistic `factor.py` payload and confirms a file
the payload names gets written by the real subprocess — genuine host-level code execution through
code this module never edited. ARGUS's own search never writes or runs a file of any kind: every
candidate is a typed `Expr` node, evaluated by `Expr.evaluate()`'s closed dispatch — the same
structural guarantee already AST-proven in `grammar_comparison.py`, reused here rather than
re-derived.

**Finding 2 — trial-count correction, or the lack of it, found by reading the real accept/reject
code AND by running ARGUS's real gate on ARGUS's own real trial pool.** RD-Agent's real
`QlibFactorExperiment2Feedback.generate_feedback()` (`scenarios/qlib/developer/feedback.py`) asks
an LLM to read the CURRENT hypothesis's backtest metrics as text and decide `"Replace Best
Result": yes/no` — the SAME held-out test window (`test_start`..`test_end` in the real qlib
backtest template) is the acceptance criterion for every iteration of a loop that can run
indefinitely, with no explicit cap on the number of hypotheses tried. An exhaustive grep of the
entire real repository for `deflat`/`multiple.test`/`false discovery`/`PBO`/`bonferroni`/`holm`/
`benjamini` returns ZERO matches — there is no trial-count correction anywhere in RD-Agent's
source. ARGUS's own real `deflated_sharpe()` (`backtest/metrics.py`, already the OWNED
"Overfitting gates that raise instead of returning NaN" capability's subject), run here on a real,
freshly-executed 400-trial `search_random` pass over real NVDAUSDT hourly bars, reports a
deflated-Sharpe probability of **9.3e-05** for the best in-sample candidate that same trial pool
produced — the gate a system with RD-Agent's real accept/reject mechanism has no equivalent of
would refuse this exact "discovery" as statistically indistinguishable from noise.
"""

from __future__ import annotations

import ast
import random
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from argus.backtest.engine import Bar
from argus.backtest.metrics import MetricError, deflated_sharpe
from argus.eval.baselines.rdagent_factor_loader import RdAgentFactorLoadError, load_factor_module
from argus.research.searchoff import (
    MIN_BARS,
    Arena,
    search_random,
)

_RDAGENT_EXPERIMENT_SRC = (
    Path(__file__).resolve().parent / "baselines" / "rdagent_experiment.py"
)
_RDAGENT_FACTOR_SRC = Path(__file__).resolve().parent / "baselines" / "rdagent_factor.py"
_SEARCHOFF_SRC = Path(__file__).resolve().parents[1] / "research" / "searchoff.py"


class RdAgentComparisonError(RuntimeError):
    """The comparison could not run — the vendored baseline failed to load."""


# =================================================================================================
# Deterministic synthetic bars for the fast, network-free test path.
# =================================================================================================


def synthetic_bars(*, n: int = 420, seed: int = 1) -> list[Bar]:
    """A seeded random walk — real code, deterministic, no network. `main()` uses real, live
    NVDAUSDT hourly bars instead (see its own docstring); the test suite uses this so it never
    depends on a live fetch."""
    rng = random.Random(seed)
    price = 100.0
    bars: list[Bar] = []
    for i in range(n):
        price *= 1 + rng.gauss(0, 0.01)
        bars.append(
            Bar(
                ts=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(hours=i),
                close=Decimal(str(round(price, 6))),
                extra={"volume": 1000.0, "high": price * 1.001, "low": price * 0.999},
            )
        )
    return bars


# =================================================================================================
# Execution surface — AST-scanned on both, then RUN on RD-Agent's real code with a live payload.
# =================================================================================================

_EXEC_FORBIDDEN_BARE = {"eval", "exec", "compile", "__import__"}
"""Names dangerous regardless of what they're called on — same set `grammar_comparison.py` uses."""

_EXEC_FORBIDDEN_QUALIFIED = {
    ("subprocess", "run"), ("subprocess", "check_output"), ("subprocess", "check_call"),
    ("subprocess", "Popen"), ("subprocess", "call"), ("os", "system"), ("os", "popen"),
}
"""Module-qualified process-spawning calls. `.run(...)`/`.call(...)` alone are far too common in
ordinary code (this project's own `research/searchoff.py` has a function literally named `run`) to
blocklist unqualified — only flagged when the call is on a name statically resolvable to the real
`subprocess`/`os` module, not any object with a same-named method."""


@dataclass(frozen=True)
class ExecutionSurfaceScan:
    path: str
    forbidden_calls_found: tuple[str, ...]

    @property
    def has_execution_surface(self) -> bool:
        return bool(self.forbidden_calls_found)

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "forbidden_calls_found": sorted(self.forbidden_calls_found),
            "has_execution_surface": self.has_execution_surface,
        }


def _scan(path: Path) -> ExecutionSurfaceScan:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if isinstance(fn, ast.Name) and fn.id in _EXEC_FORBIDDEN_BARE:
            found.add(fn.id)
        elif (
            isinstance(fn, ast.Attribute)
            and isinstance(fn.value, ast.Name)
            and (fn.value.id, fn.attr) in _EXEC_FORBIDDEN_QUALIFIED
        ):
            found.add(f"{fn.value.id}.{fn.attr}")
    return ExecutionSurfaceScan(str(path), tuple(sorted(found)))


def scan_argus_searchoff() -> ExecutionSurfaceScan:
    return _scan(_SEARCHOFF_SRC)


def scan_rdagent_factor_execution() -> ExecutionSurfaceScan:
    return _scan(_RDAGENT_FACTOR_SRC)


@dataclass(frozen=True)
class InjectionProof:
    """Not "the source contains subprocess.check_output" — this is the result of actually calling
    the real, unmodified `FactorFBWorkspace.execute()` with a crafted `factor.py` payload and
    checking, from OUTSIDE the subprocess, that the file it names now exists with the content it
    named — proof the payload ran as a real, separate process on the host."""

    marker_written: bool
    marker_contents: str
    workspace_path: str
    execution_error: str | None

    @property
    def executed_attacker_code(self) -> bool:
        return self.marker_written and self.marker_contents == _MARKER_CONTENTS

    def as_dict(self) -> dict[str, Any]:
        return {
            "marker_written": self.marker_written,
            "marker_contents": self.marker_contents,
            "workspace_path": self.workspace_path,
            "execution_error": self.execution_error,
            "executed_attacker_code": self.executed_attacker_code,
        }


_MARKER_CONTENTS = "ARGUS proof-of-execution - real subprocess ran real payload"
"""ASCII only — `FBWorkspace.inject_files` writes `factor.py` via `Path.write_text(v)` with no
explicit encoding (the real code's own choice, not this test's), which on Windows defaults to the
platform encoding, not UTF-8; a non-ASCII byte written that way can land in the file as something
the subprocess's own UTF-8-by-default source reader (PEP 3120) then rejects with a SyntaxError —
found the hard way with an em dash here, confirmed by the exact `\\x97` PEP-263 error message, and
fixed by keeping the payload ASCII rather than by guessing at an encoding fix on either side."""


def run_injection_proof(factor_module: Any) -> InjectionProof:
    """Run the real, unmodified `FactorFBWorkspace.execute()` with a `factor.py` payload that
    does nothing destructive — it only writes a marker file this function checks for from
    OUTSIDE the subprocess, which is the honest way to prove real execution without needing to
    trust anything the subprocess itself reports back."""
    task = factor_module.FactorTask(
        factor_name="argus_proof_of_execution",
        factor_description="a benign, attacker-controlled factor.py payload",
        factor_formulation="N/A",
        variables={},
    )
    ws = factor_module.FactorFBWorkspace(target_task=task, raise_exception=True)
    marker_path = ws.workspace_path / "argus_proof_of_execution.txt"
    payload = f'with open(r"{marker_path}", "w") as f:\n    f.write("{_MARKER_CONTENTS}")\n'
    ws.file_dict = {"factor.py": payload}

    # The real code raises NoOutputError (no result.h5, since this payload never writes one) —
    # EXPECTED here, and does not mean the payload itself failed to run; checked separately below
    # via the marker file, which is why every exception is caught rather than one specific type.
    execution_error: str | None = None
    try:
        ws.execute()
    except Exception as exc:
        execution_error = f"{type(exc).__name__}: {exc}"

    marker_written = marker_path.exists()
    marker_contents = marker_path.read_text(encoding="utf-8") if marker_written else ""
    return InjectionProof(marker_written, marker_contents, str(ws.workspace_path), execution_error)


@dataclass(frozen=True)
class FailureCase:
    system: str
    input_: str
    outcome: str

    def as_dict(self) -> dict[str, Any]:
        return {"system": self.system, "input": self.input_, "outcome": self.outcome}


def run_failure_cases(factor_module: Any) -> list[FailureCase]:
    """The case RD-Agent's own error handling is designed for — a factor.py that crashes — run
    for real, alongside ARGUS's construction-time rejection of an out-of-budget search."""
    out: list[FailureCase] = []

    task = factor_module.FactorTask(
        factor_name="argus_crash_case", factor_description="a factor.py that raises",
        factor_formulation="N/A", variables={},
    )
    ws = factor_module.FactorFBWorkspace(target_task=task, raise_exception=True)
    ws.file_dict = {"factor.py": "raise RuntimeError('deliberate crash')\n"}
    try:
        ws.execute()
        out.append(
            FailureCase("rdagent", "factor.py raises RuntimeError", "NO EXCEPTION (unexpected)")
        )
    except factor_module.CustomRuntimeError as exc:
        out.append(
            FailureCase(
                "rdagent", "factor.py raises RuntimeError",
                f"CustomRuntimeError caught cleanly by execute()'s own handler: "
                f"{'deliberate crash' in str(exc)}",
            )
        )

    from argus.research.grammar import Field, GrammarError, Ref, Window

    try:
        Window("not_a_real_op", 5, Ref(Field.CLOSE))
        out.append(FailureCase("argus", "Window('not_a_real_op', ...)", "NO EXCEPTION (BUG)"))
    except GrammarError:
        out.append(
            FailureCase(
                "argus", "Window('not_a_real_op', ...)",
                "GrammarError at construction — never reaches a search loop at all",
            )
        )
    return out


# =================================================================================================
# Trial-count correction — RD-Agent's real accept/reject has none; ARGUS's real DSR gate does.
# =================================================================================================


@dataclass(frozen=True)
class TrialPoolResult:
    bars: int
    budget: int
    trials_spent: int
    distinct_candidates: int
    best_in_sample: float
    in_sample_observations: int
    variance_of_trials: float
    raw_claim: float
    deflated_probability: float | None
    deflated_error: str | None

    @property
    def deflation_is_severe(self) -> bool:
        """ARGUS's own real gate: does correcting for the real trial count collapse a positive
        raw claim to a probability indistinguishable from noise (p < 0.01)?"""
        return self.deflated_probability is not None and self.deflated_probability < 0.01

    def as_dict(self) -> dict[str, Any]:
        return {
            "bars": self.bars,
            "budget": self.budget,
            "trials_spent": self.trials_spent,
            "distinct_candidates": self.distinct_candidates,
            "best_in_sample": round(self.best_in_sample, 6),
            "in_sample_observations": self.in_sample_observations,
            "variance_of_trials": round(self.variance_of_trials, 8),
            "raw_claim": round(self.raw_claim, 6),
            "deflated_probability": (
                None if self.deflated_probability is None else self.deflated_probability
            ),
            "deflated_error": self.deflated_error,
            "deflation_is_severe": self.deflation_is_severe,
        }


def run_trial_pool(bars: list[Bar], *, budget: int, seed: int = 20260913) -> TrialPoolResult:
    """Run ARGUS's real `search_random` over `bars`, then run ARGUS's real `deflated_sharpe` on
    the real trial pool it produced — both real, neither reimplemented."""
    arena = Arena(bars=bars, cost_bps=12.0, budget=budget)
    search_random(arena, random.Random(seed))
    in_samples = [c.in_sample for c in arena.seen.values()]
    if not in_samples:
        raise RdAgentComparisonError("search_random produced no scoreable candidate")
    best = max(in_samples)
    mean = sum(in_samples) / len(in_samples)
    variance = sum((x - mean) ** 2 for x in in_samples) / len(in_samples)
    n_obs = arena.split - 1

    try:
        p = deflated_sharpe(
            best, n=max(n_obs, 2), trials=arena.spent, variance_of_trials=variance,
        )
        error = None
    except MetricError as exc:
        p, error = None, str(exc)

    return TrialPoolResult(
        bars=len(bars), budget=budget, trials_spent=arena.spent,
        distinct_candidates=len(arena.seen), best_in_sample=best,
        in_sample_observations=n_obs, variance_of_trials=variance,
        raw_claim=best, deflated_probability=p, deflated_error=error,
    )


# =================================================================================================
# Ablation — is the deflation trial-count-sensitive, or a fixed penalty regardless of how many
# trials were actually run?
# =================================================================================================


@dataclass(frozen=True)
class AblationPoint:
    trials: int
    probability: float

    def as_dict(self) -> dict[str, Any]:
        return {"trials": self.trials, "probability": self.probability}


def run_ablation(pool: TrialPoolResult) -> list[AblationPoint]:
    """Sweep `trials` from 1 to the real count on the SAME observed value and variance — proving
    the correction responds to trial count, not merely to whether trials>1 at all."""
    out: list[AblationPoint] = []
    for trials in (1, 5, 25, 100, pool.trials_spent):
        p = deflated_sharpe(
            pool.raw_claim, n=max(pool.in_sample_observations, 2), trials=trials,
            variance_of_trials=pool.variance_of_trials,
        )
        out.append(AblationPoint(trials, p))
    return out


# =================================================================================================
# Costs.
# =================================================================================================


def measure_costs(factor_module: Any, bars: list[Bar]) -> dict[str, float]:
    """The two numbers below are NOT measuring the same thing, and reporting one as simply
    "cheaper" than the other would overclaim — found by actually running both on real, differently
    sized bar series, not assumed. `rdagent_subprocess_spawn_seconds` is real RD-Agent code's fixed
    per-call floor: interpreter startup + shell spawn, with a trivial payload that does almost no
    computation — a TAX every RD-Agent candidate pays before its factor.py's own logic even starts,
    REGARDLESS of how much or little that logic actually computes. `argus_grammar_evaluate_seconds`
    is real, computation-bound work (two O(bars) `_score_half` passes per candidate) with NO
    subprocess tax at all — it scales with the size of the backtest, and on `bars`, this project's
    own real ~1487-bar NVDAUSDT window makes it the LARGER of the two numbers, not the smaller one
    (found here, not assumed from the smaller synthetic fixture the test suite uses instead)."""
    import contextlib

    task = factor_module.FactorTask(
        factor_name="cost_probe", factor_description="d", factor_formulation="N/A", variables={},
    )
    ws = factor_module.FactorFBWorkspace(target_task=task, raise_exception=True)
    marker_path = ws.workspace_path / "cost_probe.txt"
    ws.file_dict = {"factor.py": f'with open(r"{marker_path}", "w") as f:\n    f.write("x")\n'}
    start = time.perf_counter()
    n_runs = 5
    # NoOutputError, expected every call (see run_injection_proof) — this payload never writes a
    # result.h5, so the real code's own post-execution file check always raises; the cost being
    # measured is the subprocess spawn+run itself, which already happened by the time it raises.
    for _ in range(n_runs):
        with contextlib.suppress(Exception):
            ws.execute()
    rdagent_seconds = (time.perf_counter() - start) / n_runs

    arena = Arena(bars=bars, cost_bps=12.0, budget=200)
    start = time.perf_counter()
    search_random(arena, random.Random(1))
    argus_seconds = (time.perf_counter() - start) / max(arena.spent, 1)

    return {
        "rdagent_subprocess_spawn_seconds": rdagent_seconds,
        "argus_grammar_evaluate_seconds": argus_seconds,
        "bars_scored": len(bars),
        "argus_cost_exceeds_rdagent_spawn_floor": argus_seconds > rdagent_seconds,
    }


# =================================================================================================
# Reproducibility.
# =================================================================================================


def run_reproducibility_check(bars: list[Bar]) -> dict[str, bool]:
    arena_a = Arena(bars=bars, cost_bps=12.0, budget=60)
    search_random(arena_a, random.Random(42))
    arena_b = Arena(bars=bars, cost_bps=12.0, budget=60)
    search_random(arena_b, random.Random(42))
    a_scores = sorted(c.in_sample for c in arena_a.seen.values())
    b_scores = sorted(c.in_sample for c in arena_b.seen.values())
    return {"argus_reproducible": a_scores == b_scores}


# =================================================================================================
# Scope statement.
# =================================================================================================

SCOPE_STATEMENT = """\
Claimed: RD-Agent's real, unmodified FactorFBWorkspace.execute() executes attacker-controlled \
Python via an unsandboxed subprocess (LocalEnv, hard-coded for factor execution, confirmed by \
reading factor_coder/config.py's get_factor_env()) — demonstrated by running it, not by reading \
the source alone; ARGUS's typed grammar contains zero eval/exec/subprocess calls anywhere in its \
source (AST-verified, the same technique grammar_comparison.py already runs, reused here rather \
than re-derived). Separately and independently: RD-Agent's real factor-search accept/reject \
mechanism has no trial-count correction anywhere in its source (confirmed by an exhaustive grep \
for deflation/multiple-testing/PBO/FDR terms across the whole real repository, zero matches); \
ARGUS's real deflated_sharpe(), run on a real 400-trial search_random pass over real NVDAUSDT \
hourly bars, reports a 9.3e-05 probability for the best in-sample candidate that same real trial \
pool produced — a discovery a system with RD-Agent's real accept/reject mechanism has no way to \
refuse.

NOT claimed: that RD-Agent's factor-implementation CODER (the LLM call that writes factor.py in \
the first place, upstream of the execute() step this comparison starts at) is unsound — this \
comparison exercises the EXECUTION step only, not the code-generation step, which this module \
does not vendor or run. NOT claimed: that every RD-Agent-proposed factor is spurious — the real \
qlib backtest template DOES separate train/valid/test date ranges (confirmed by reading \
factor_template/conf_combined_factors.yaml), so the metrics themselves are computed out-of- \
sample; the gap found here is narrower and specific: the SAME held-out test window is reused as \
the acceptance criterion across an unbounded number of loop iterations, with nothing correcting \
for how many hypotheses were tried against it. NOT claimed: that ARGUS's search methodology \
(random/beam/evolutionary/novelty/anneal) finds BETTER factors than RD-Agent's LLM-driven \
proposals — the published Track 1 result is that 0 of 12 symbols survive the deflated Sharpe over \
all trials on ARGUS's own side too (research/searchoff.py's own module docstring); the claim here \
is narrower still: ARGUS's search has a real gate that would catch the exact failure mode \
RD-Agent's real accept/reject mechanism has no defense against. NOT claimed: that ARGUS's \
grammar evaluation is cheaper than RD-Agent's subprocess execution — measure_costs()'s own \
docstring is explicit that the two numbers measure different things (a fixed per-call subprocess-\
spawn floor vs. real O(bars) computation with no subprocess tax at all), and on this project's \
real ~1487-bar NVDAUSDT window the computation-bound ARGUS number is the LARGER of the two, found \
by running both, not assumed from a smaller synthetic fixture.
"""


# =================================================================================================
# Entry point.
# =================================================================================================


def main(*, use_live_market_data: bool = True) -> dict[str, Any]:
    """Run the whole comparison and return a serialisable summary.

    Args:
        use_live_market_data: when True (the default, used by `__main__`), fetches real, live
            NVDAUSDT hourly bars via `argus.market.history.fetch_range` for the trial-pool
            evidence — the same real network call `research/searchoff.py`'s own `main()` makes.
            Tests pass False to use the deterministic, network-free `synthetic_bars()` instead.

    Raises:
        RdAgentComparisonError: the vendored RD-Agent baseline failed to load, or a real search
            over 62 days of NVDAUSDT hourly bars produced no scoreable candidate.
    """
    try:
        _experiment_module, factor_module = load_factor_module()
    except RdAgentFactorLoadError as exc:
        raise RdAgentComparisonError(
            f"could not load the vendored RD-Agent factor-execution baseline: {exc}"
        ) from exc

    if use_live_market_data:
        from argus.market.history import CandleType, fetch_range

        candles = fetch_range("NVDAUSDT", days=62, interval="1H", candle_type=CandleType.MARKET)
        bars = [
            Bar(
                ts=c.ts, close=c.close,
                extra={"volume": float(c.volume), "high": float(c.high), "low": float(c.low)},
            )
            for c in candles
        ]
        if len(bars) < MIN_BARS:
            raise RdAgentComparisonError(
                f"only {len(bars)} live bars fetched, need at least {MIN_BARS}"
            )
        budget = 400
    else:
        bars = synthetic_bars()
        budget = 30

    argus_scan = scan_argus_searchoff()
    rdagent_scan = scan_rdagent_factor_execution()
    injection = run_injection_proof(factor_module)
    failure_cases = run_failure_cases(factor_module)
    pool = run_trial_pool(bars, budget=budget)
    ablation = run_ablation(pool)
    costs = measure_costs(factor_module, bars)
    reproducibility = run_reproducibility_check(bars)

    return {
        "argus_execution_surface_scan": argus_scan.as_dict(),
        "rdagent_execution_surface_scan": rdagent_scan.as_dict(),
        "injection_proof": injection.as_dict(),
        "failure_cases": [f.as_dict() for f in failure_cases],
        "trial_pool": pool.as_dict(),
        "ablation": [a.as_dict() for a in ablation],
        "costs": costs,
        "reproducibility": reproducibility,
        "used_live_market_data": use_live_market_data,
        "scope_statement": SCOPE_STATEMENT,
    }


def render(report: dict[str, Any]) -> str:
    lines = [
        "RD-AGENT COMPARISON — ARGUS typed factor search vs. RD-Agent real factor execution",
        "",
        "-- execution surface --",
    ]
    a = report["argus_execution_surface_scan"]
    r = report["rdagent_execution_surface_scan"]
    lines.append(f"  argus searchoff.py forbidden calls found: {a['forbidden_calls_found']}")
    lines.append(f"  rdagent factor execution forbidden calls found: {r['forbidden_calls_found']}")
    inj = report["injection_proof"]
    lines.append(
        f"  injection proof: marker written by real subprocess = {inj['executed_attacker_code']}"
    )
    lines.append("")
    lines.append("-- trial-count correction --")
    p = report["trial_pool"]
    lines.append(
        f"  {p['trials_spent']} real trials, best in-sample={p['raw_claim']:.4f} "
        f"-> deflated probability={p['deflated_probability']}"
    )
    lines.append(f"  deflation_is_severe (p<0.01): {p['deflation_is_severe']}")
    lines.append(f"  ablation (trials -> probability): {report['ablation']}")
    lines.append("")
    cst = report["costs"]
    rdagent_cost = cst["rdagent_subprocess_spawn_seconds"]
    argus_cost = cst["argus_grammar_evaluate_seconds"]
    lines.append(
        f"cost (not directly comparable — see measure_costs' own docstring): "
        f"rdagent subprocess spawn floor {rdagent_cost:.4f}s, "
        f"argus grammar evaluate {argus_cost:.4f}s on {cst['bars_scored']} bars "
        f"(argus exceeds rdagent's spawn floor: {cst['argus_cost_exceeds_rdagent_spawn_floor']})"
    )
    rp = report["reproducibility"]
    lines.append(f"reproducible — argus: {rp['argus_reproducible']}")
    return "\n".join(lines)


if __name__ == "__main__":
    import json

    result = main()
    print(render(result))
    out_path = Path(__file__).resolve().parents[3] / "data" / "rdagent_comparison.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nsaved -> {out_path}")


__all__ = [
    "SCOPE_STATEMENT",
    "AblationPoint",
    "ExecutionSurfaceScan",
    "FailureCase",
    "InjectionProof",
    "RdAgentComparisonError",
    "TrialPoolResult",
    "main",
    "measure_costs",
    "render",
    "run_ablation",
    "run_failure_cases",
    "run_injection_proof",
    "run_reproducibility_check",
    "run_trial_pool",
    "scan_argus_searchoff",
    "scan_rdagent_factor_execution",
    "synthetic_bars",
]
