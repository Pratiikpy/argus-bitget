"""Does each comparison harness actually run the code it is cited for? A canary, per harness.

``eval/standing.py`` credits a capability to a module (``module="argus/desk/odds.py"``) and backs it
with an artefact a harness wrote (``artefact="data/stopquality_comparison.json"``). Nothing checked
that the harness which wrote the artefact *executes* that module. A harness can score a copy of the
formula written inside itself, or outputs recorded weeks ago, or swallow an exception from the code
under test and still print a verdict — and every one of those passes a review that reads the
artefact. This module is the check.

**What was taken, from where.** AgentDojo's ``TaskSuite.check`` proves its benchmark is sound before
any model is scored (``src/agentdojo/task_suite/task_suite.py:422-479``, ethz-spylab/agentdojo,
**MIT**): it fills every injection vector with a unique ``---CANARY_<vector>---`` string, runs the
*ground-truth* pipeline, and fails any task whose canary never appears in a tool output
(``is_task_injectable``, ``:482-491``) or whose ground truth does not solve it (``:437-443``). Both
halves carry over:

1. **Reachability** (the injectability half). The modules ``standing.py`` credits for a harness's
   artefact must execute when the harness's entry point runs. This is read from the sabotage run
   below: a canary that fires *during the entry point* proves its module was called. Where a canary
   fires already while the harness is being imported, the entry point never runs in that process,
   so reachability is measured instead by a separate profiled run that records every ARGUS function
   executed (module-level import code does not count; only function calls do). Profiling is not
   the default because it slowed the heaviest harnesses (allocation, analogstress, regime) past a
   15-minute budget on the first full run, 2026-09-25.
2. **Sabotage canary** (the canary-string half, turned around). The run is repeated with every
   function and method of the credited modules replaced by one that raises
   ``CanaryError("---CANARY_<module>.<function>---")``. The canary must *surface*: the harness
   either fails, or its output carries the token. A harness that completes with the canary nowhere
   in its output has swallowed a failure of the code it scores — exactly the silent-failure mode
   mle-bench's grader refuses (``mlebench/grade_helpers.py:36-55``).
3. **Oracle canary** (the ground-truth half), for the harnesses migrated onto the spine
   (``claimcheck``, ``void``, ``overnight``): the harness's own scorer is handed a perfect desk and
   a wrong one. The perfect desk must score perfectly and the wrong one must score worse; if not,
   the scorer is not measuring what it says.

Each harness's checks are combined with AND through ``eval/evaluators.py`` (all rules, run to the
end so every failure is listed), and the fleet verdict is all-or-nothing: it exists only if every
harness was actually run.

**Isolation, so the canary cannot do harm.** Every run is a subprocess with no Qwen key in its
environment, with every socket connection refused (``NetworkBlockedError``, an ``OSError`` so a
harness's own offline handling is exercised as it would be on a plane), and with every file write
under the repository redirected to a scratch directory — a canary run never rewrites a published
artefact. A harness that needs the network is reported as not reproducible offline, which is a
finding about the artefact rather than a defect in the canary.

**Why not a hand-written list of targets per harness.** The targets come from ``standing.py``'s own
register. That is the claim a judge reads; checking a list written here would check this module's
opinion of what each harness is for.
"""

from __future__ import annotations

import argparse
import builtins
import contextlib
import importlib
import inspect
import io
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import FrameType, ModuleType
from typing import Any

from argus.eval.evaluators import Evaluator, Grade, all_or_nothing, combine, rule, statistic

PACKAGE = Path(__file__).resolve().parents[3]          # .../bitget/argus
SRC = PACKAGE / "src" / "argus"
EVAL = SRC / "eval"
DATA = PACKAGE / "data"
REPORT_PATH = DATA / "harness_validity.json"
PROTECTED = PACKAGE.parent                             # the workspace: never written by a canary
TIMEOUT_S = 900
CANARY = "---CANARY_{}---"

SKIP: dict[str, str] = {
    "sentiment_comparison": "main() builds a live QwenClient and scores the model's own answers; "
                            "this task's Qwen cap is 0, so it is not run",
}
"""Harnesses deliberately not executed, with the reason. They count as NOT RUN, never as passed."""

ENTRY: dict[str, str] = {
    "claimcheck_comparison": "score", "void_comparison": "score",
    "overnight_comparison": "score", "stopquality_comparison": "score",
}
"""Entry points other than ``main``: the scorer that produces the artefact, without its CLI."""

HARNESS_CODE = re.compile(r"^argus/eval/(?:[a-z_]+_comparison|baselines/[a-z_0-9]+)\.py$")
"""Files that are harness or rival code, never the ARGUS code a capability is credited to."""


class CanaryError(RuntimeError):
    """Raised by a sabotaged function of a credited module."""


class NetworkBlockedError(ConnectionRefusedError):
    """Every socket connection in a canary run. An OSError, so offline handling is exercised."""


# ---------------------------------------------------------------------------------------------
# Which modules does standing.py credit for each harness's artefact?


def _home_as_tilde(text: str) -> str:
    """A traceback with the machine's home directory written as ``~``: the path says nothing a
    reader needs, and an artefact meant for publishing must not carry a personal path."""
    home = str(Path.home())
    return text.replace(home, "~").replace(home.replace("\\", "/"), "~")

def harnesses() -> list[str]:
    """Every ``eval/*_comparison.py`` module, by module name."""
    return sorted(p.stem for p in EVAL.glob("*_comparison.py"))


def claimed_modules(harness: str) -> dict[str, list[str]]:
    """``{capability name: [credited module paths]}`` for every standing capability tied to this
    harness: a proof cites its artefact (``data/<harness>.json``) or its source
    (``src/argus/eval/<harness>.py``), or the capability's own ``module`` field lists the harness.
    All three forms occur in the register — the first full run read only the first and found
    nothing credited for most harnesses, a defect in this function, not in the harnesses."""
    from argus.eval.standing import REGISTER

    artefact = f"data/{harness}.json"
    source = f"argus/eval/{harness}.py"
    out: dict[str, list[str]] = {}
    for capability in REGISTER:
        credited = re.findall(r"argus/[A-Za-z0-9_/]+\.py", capability.module)
        cited = any(proof.artefact == artefact or proof.artefact.endswith(source)
                    for proof in capability.proofs)
        if cited or source in credited:
            out[capability.name] = credited
    return out


def code_under_test(harness: str, credited: Sequence[str]) -> list[str]:
    """The credited paths that are ARGUS code rather than harness or rival code."""
    return sorted({p for p in credited
                   if not HARNESS_CODE.match(p) and p != f"argus/eval/{harness}.py"})


def _dotted(path: str) -> str:
    return path.removesuffix(".py").replace("/", ".")


def _relative(filename: str) -> str | None:
    try:
        rel = Path(filename).resolve().relative_to(SRC.parent)
    except (ValueError, OSError):
        return None
    return rel.as_posix()


# ---------------------------------------------------------------------------------------------
# The child process: guards, profiler, sabotage, then the harness's entry point.

@dataclass
class _Guards:
    network_attempts: int = 0
    writes_redirected: list[str] = field(default_factory=list)
    fired: list[str] = field(default_factory=list)
    executed: set[str] = field(default_factory=set)


def _install_guards(guards: _Guards, scratch: Path) -> None:
    def refuse(*_a: Any, **_k: Any) -> Any:
        guards.network_attempts += 1
        raise NetworkBlockedError("network is blocked in a harness-validity run")

    socket.socket.connect = refuse  # type: ignore[method-assign]
    socket.socket.connect_ex = refuse  # type: ignore[method-assign]
    socket.create_connection = refuse
    socket.getaddrinfo = refuse

    real_open = io.open

    def contained_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        if isinstance(file, (str, os.PathLike)) and any(c in mode for c in "wax+"):
            target = Path(os.fspath(file)).resolve()
            if target.is_relative_to(PROTECTED):
                redirected = scratch / re.sub(r"[^A-Za-z0-9_.-]", "_", target.as_posix())[-180:]
                guards.writes_redirected.append(target.relative_to(PROTECTED).as_posix())
                return real_open(redirected, mode, *args, **kwargs)
        return real_open(file, mode, *args, **kwargs)

    builtins.open = contained_open
    io.open = contained_open

    real_replace, real_rename = os.replace, os.rename

    def guarded(real: Callable[..., None]) -> Callable[..., None]:
        def move(src: Any, dst: Any, *a: Any, **k: Any) -> None:
            target = Path(os.fspath(dst)).resolve()
            if target.is_relative_to(PROTECTED):
                guards.writes_redirected.append(
                    f"(move refused) {target.relative_to(PROTECTED).as_posix()}")
                return
            real(src, dst, *a, **k)
        return move

    os.replace = guarded(real_replace)
    os.rename = guarded(real_rename)


def _sabotage(module: ModuleType, guards: _Guards) -> int:
    """Replace every function and method defined in ``module`` with one raising a canary."""
    def canary(qualname: str) -> Callable[..., Any]:
        token = CANARY.format(f"{module.__name__}.{qualname}")

        def fire(*_a: Any, **_k: Any) -> Any:
            guards.fired.append(token)
            raise CanaryError(token)
        return fire

    count = 0
    for name, obj in list(vars(module).items()):
        if inspect.isfunction(obj) and obj.__module__ == module.__name__:
            setattr(module, name, canary(name))
            count += 1
        elif inspect.isclass(obj) and obj.__module__ == module.__name__:
            for attr, member in list(vars(obj).items()):
                if attr.startswith("__"):
                    continue
                if inspect.isfunction(member):
                    setattr(obj, attr, canary(f"{name}.{attr}"))
                elif isinstance(member, (staticmethod, classmethod)):
                    wrapped = canary(f"{name}.{attr}")
                    setattr(obj, attr, type(member)(wrapped))
                else:
                    continue
                count += 1
    return count


def _entry_args(fn: Callable[..., Any]) -> tuple[Any, ...]:
    params = inspect.signature(fn).parameters
    return ([],) if "argv" in params else ()


def _child(harness: str, mode: str, targets: Sequence[str], out: Path, *,
           profiled: bool = False) -> None:
    guards = _Guards()
    scratch = Path(tempfile.mkdtemp(prefix=f"hv_{harness}_"))
    real_open = io.open
    _install_guards(guards, scratch)
    sys.argv = [f"{harness}.py"]
    result: dict[str, Any] = {"harness": harness, "mode": mode}
    sabotaged: dict[str, int] = {}
    stdout = io.StringIO()

    def profile(frame: FrameType, event: str, _arg: Any) -> None:
        if event == "call" and frame.f_code.co_name != "<module>":
            guards.executed.add(frame.f_code.co_filename)

    started = time.perf_counter()
    stage = "import"
    fired_before = 0
    try:
        if mode == "canary":
            for target in targets:
                sabotaged[target] = _sabotage(importlib.import_module(_dotted(target)), guards)
        fired_before = len(guards.fired)
        module = importlib.import_module(f"argus.eval.{harness}")
        result["fired_at_import"] = len(guards.fired) > fired_before
        entry = getattr(module, ENTRY.get(harness, "main"))
        stage = "entry"
        fired_before = len(guards.fired)
        if profiled:
            sys.setprofile(profile)
            threading.setprofile(profile)
        with contextlib.redirect_stdout(stdout):
            returned = entry(*_entry_args(entry))
        result["status"] = "ok"
        result["returned"] = json.dumps(returned, default=str)[:2_000_000]
        result["fired_in_entry"] = sorted(set(guards.fired[fired_before:]))
    except BaseException as exc:  # the canary run must report every failure, KeyboardInterrupt too
        result["status"] = f"raised during {stage}"
        result["error"] = _home_as_tilde("".join(traceback.format_exception(exc))[-6000:])
        result["returned"] = ""
        result["fired_in_entry"] = (sorted(set(guards.fired[fired_before:]))
                                    if stage == "entry" else [])
        if stage == "import":
            # A canary fired while a credited module or the harness was being imported (a
            # module-level call such as a constant built from a sabotaged function).
            result["fired_at_import"] = bool(guards.fired)
    finally:
        sys.setprofile(None)
        threading.setprofile(None)
    result.update({
        "elapsed_s": round(time.perf_counter() - started, 2),
        "stdout_tail": stdout.getvalue()[-4000:],
        "network_attempts": guards.network_attempts,
        "writes_redirected": sorted(set(guards.writes_redirected)),
        "profiled": profiled,
        "executed": sorted(r for f in guards.executed if (r := _relative(f)) is not None),
        "sabotaged_functions": sabotaged,
        "fired": sorted(set(guards.fired)),
    })
    with real_open(out, "w", encoding="utf-8") as handle:
        json.dump(result, handle)


# ---------------------------------------------------------------------------------------------
# The parent: run both modes per harness, grade, aggregate.

def _run_child(harness: str, mode: str, targets: Sequence[str], timeout: int = TIMEOUT_S, *,
               profiled: bool = False) -> dict[str, Any]:
    env = {k: v for k, v in os.environ.items() if "QWEN" not in k.upper()}
    env["PYTHONUTF8"] = "1"
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "result.json"
        cmd = [sys.executable, "-m", "argus.eval.harness_validity", "--child", harness,
               "--mode", mode, "--out", str(out), *[f"--target={t}" for t in targets],
               *(["--profile"] if profiled else [])]
        try:
            proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout,
                                  cwd=str(PACKAGE), check=False)
        except subprocess.TimeoutExpired:
            return {"harness": harness, "mode": mode, "status": f"timed out after {timeout}s"}
        if not out.exists():
            return {"harness": harness, "mode": mode, "status": "child produced no result",
                    "error": _home_as_tilde((proc.stderr or proc.stdout)[-4000:])}
        blob: dict[str, Any] = json.loads(out.read_text("utf-8"))
        return blob


@dataclass(frozen=True)
class HarnessRun:
    """Everything measured about one harness."""

    harness: str
    claims: dict[str, list[str]]
    under_test: list[str]
    skipped: str | None
    baseline: dict[str, Any]
    canary: dict[str, Any]
    oracle: dict[str, Any] | None = None
    profiled: dict[str, Any] | None = None

    def reached(self) -> list[str]:
        """Credited modules shown to execute in the entry point: by a canary that fired there, or,
        when the canary fired at import, by the profiled run's executed list."""
        fired = " ".join(self.canary.get("fired_in_entry") or [])
        by_canary = {p for p in self.under_test if f"---CANARY_{_dotted(p)}." in fired}
        executed = set((self.profiled or {}).get("executed") or [])
        return sorted(by_canary | (set(self.under_test) & executed))

    def surfaced(self) -> bool | None:
        """Did a fired canary surface? None when nothing fired or it fired only at import."""
        fired = self.canary.get("fired") or []
        if not fired or self.canary.get("fired_at_import"):
            return None
        if self.canary.get("status") != "ok":
            return True
        visible = f"{self.canary.get('returned', '')}{self.canary.get('stdout_tail', '')}"
        return any(token in visible for token in fired)


def _checks(run: HarnessRun) -> list[Evaluator[HarnessRun]]:
    reached = run.reached()
    checks: list[Evaluator[HarnessRun]] = [
        rule("ran_offline", lambda r: (r.baseline.get("status") == "ok",
                                       str(r.baseline.get("status")))),
        rule("reproducible_without_network", lambda r: (
            r.baseline.get("network_attempts") == 0,
            f"{r.baseline.get('network_attempts')} connection attempt(s) refused")),
        rule("cited_by_standing", lambda r: (bool(r.claims),
                                             f"{len(r.claims)} capability(ies) cite its artefact")),
        rule("credits_code_outside_the_harness", lambda r: (
            bool(r.under_test),
            f"credited: {sorted({p for v in r.claims.values() for p in v})}")),
        # Any credited module executing passes; the ones that did not are named in the detail. A
        # capability that credits a types-only module alongside the logic would otherwise fail on
        # a file with no functions to call.
        rule("reaches_the_credited_code", lambda r: (
            bool(r.under_test) and bool(reached),
            f"executed {reached} of {r.under_test}")),
        rule("sabotage_canary_surfaces", lambda r: (
            r.surfaced() is True,
            f"canary {'surfaced' if r.surfaced() else 'fired only at import' if r.canary.get('fired_at_import') else 'swallowed' if r.surfaced() is False else 'never fired'}; "  # noqa: E501
            f"{len(r.canary.get('fired') or [])} distinct canary(ies) fired")),
    ]
    if run.oracle is not None:
        checks.append(statistic("oracle_scores_perfectly", lambda r: (
            bool(r.oracle and r.oracle["oracle_perfect"]), json.dumps(r.oracle)[:300])))
        checks.append(statistic("wrong_desk_scores_worse", lambda r: (
            bool(r.oracle and r.oracle["anti_oracle_worse"]), json.dumps(r.oracle)[:300])))
    return checks


def grade(run: HarnessRun) -> Grade:
    """AND of every check, all run (``stop_at_first_failure=False``) so each failure is listed."""
    return combine(_checks(run), run, stop_at_first_failure=False)


# ---------------------------------------------------------------------------------------------
# Oracle canaries for the three harnesses migrated onto the spine.

def oracle_claimcheck() -> dict[str, Any]:
    """Hand ``claimcheck.grade`` a desk right on every claim, and one that is always wrong."""
    from argus.eval import claimcheck_comparison as cc

    claims = json.loads((cc.DATA / "claims.json").read_text("utf-8"))
    tape = json.loads((cc.DATA / "tape.json").read_text("utf-8"))
    mirror = json.loads((cc.DATA / "mirrorline_raw.json").read_text("utf-8"))
    us_open = cc.session_open_at_start(tape)

    def desk(right: bool) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, sentences in claims.items():
            truth_dir = cc._truth_direction(name, tape)
            out[name] = []
            for sentence in sentences:
                kind, asserted = cc._claim_kind(sentence)
                holds = ((asserted == "open") == us_open if kind == "session"
                         else truth_dir == asserted)
                says = holds if right else not holds
                subject = "US market" if kind == "session" else name
                line = f"Your premise ({subject}): {'holds' if says else 'does not hold'}"
                if kind == "causal" and right:
                    line += "; the stated cause is not something a price can confirm"
                out[name].append({"lines": [line]})
        return out

    perfect = cc.grade(claims, tape, desk(True), mirror, us_open=us_open)
    wrong = cc.grade(claims, tape, desk(False), mirror, us_open=us_open)
    return {"graded": perfect["graded"], "oracle_correct": perfect["argus_correct"],
            "wrong_desk_correct": wrong["argus_correct"],
            "oracle_perfect": perfect["argus_correct"] == perfect["graded"] > 0,
            "anti_oracle_worse": wrong["argus_correct"] < perfect["argus_correct"]}


def oracle_void() -> dict[str, Any]:
    """nocturne's own walk-forward with ARGUS's column replaced by the realised return, and by its
    negation."""
    from argus.eval import void_comparison as vc

    preds = [p for p in vc.predictions() if p["ARGUS"] is not None]
    oracle = [{**p, "ARGUS": p["actual"]} for p in preds]
    wrong = [{**p, "ARGUS": -3 * p["actual"]} for p in preds]
    perfect, bad = vc._score(oracle, "ARGUS"), vc._score(wrong, "ARGUS")
    vs_fade = vc._paired(oracle, "ARGUS", "MF")
    return {"rows": len(preds), "oracle_mae_pct": perfect["mae_pct"],
            "wrong_desk_mae_pct": bad["mae_pct"], "oracle_vs_full_fade": vs_fade["verdict"],
            "oracle_perfect": perfect["mae_pct"] == 0 and vs_fade["verdict"] == "a better",
            "anti_oracle_worse": bad["mae_pct"] > perfect["mae_pct"]}


def oracle_overnight() -> dict[str, Any]:
    """The overnight harness's rows with an ``argus_perp`` that knew the gap, and one that
    predicted it backwards."""
    from argus.eval import overnight_comparison as oc

    inputs = json.loads((oc.DATA / "inputs.json").read_text("utf-8"))
    rows = [r for r in oc.predict(oc.nights(inputs)) if r["fitted"]]
    oracle = [{**r, "estimates": {**r["estimates"], "argus_perp": r["gap"]}} for r in rows]
    wrong = [{**r, "estimates": {**r["estimates"], "argus_perp": -r["gap"]}} for r in rows]
    perfect, bad = oc._summary(oracle, "argus_perp"), oc._summary(wrong, "argus_perp")
    vs_best = oc._paired(oracle, "argus_perp", "gloaming_ols")
    return {"rows": len(rows), "oracle_mae_bps": perfect["mae_bps"],
            "wrong_desk_mae_bps": bad["mae_bps"], "oracle_vs_gloaming": vs_best["verdict"],
            "oracle_perfect": perfect["mae_bps"] == 0 and vs_best["verdict"] == "a better",
            "anti_oracle_worse": bad["mae_bps"] > perfect["mae_bps"]}


ORACLES: dict[str, Callable[[], dict[str, Any]]] = {
    "claimcheck_comparison": oracle_claimcheck,
    "void_comparison": oracle_void,
    "overnight_comparison": oracle_overnight,
}


def measure(harness: str, *, timeout: int = TIMEOUT_S) -> HarnessRun:
    """Run one harness's baseline and sabotage canary (and oracle, where one exists)."""
    claims = claimed_modules(harness)
    under_test = code_under_test(harness, [p for v in claims.values() for p in v])
    if harness in SKIP:
        return HarnessRun(harness, claims, under_test, SKIP[harness],
                          {"status": "not run: " + SKIP[harness]}, {"status": "not run"})
    baseline = _run_child(harness, "baseline", [], timeout)
    canary = (_run_child(harness, "canary", under_test, timeout) if under_test
              else {"status": "not run: nothing outside the harness is credited"})
    profiled = (_run_child(harness, "baseline", [], timeout, profiled=True)
                if canary.get("fired_at_import") else None)
    oracle: dict[str, Any] | None = None
    if harness in ORACLES:
        try:
            oracle = ORACLES[harness]()
        except Exception as exc:
            oracle = {"error": f"{type(exc).__name__}: {exc}", "oracle_perfect": False,
                      "anti_oracle_worse": False}
    print(f"measured {harness}: baseline {baseline.get('status')}, canary "
          f"{canary.get('status')}", file=sys.stderr, flush=True)
    return HarnessRun(harness, claims, under_test, None, baseline, canary, oracle, profiled)


def _summarise(run: HarnessRun) -> dict[str, Any]:
    graded = grade(run) if run.skipped is None else None
    reached = run.reached()
    return {
        "harness": run.harness, "skipped": run.skipped,
        "capabilities_citing_it": run.claims, "code_under_test": run.under_test,
        "reached": reached,
        "baseline": {k: run.baseline.get(k) for k in (
            "status", "elapsed_s", "network_attempts", "writes_redirected", "error")},
        "canary": {k: run.canary.get(k) for k in (
            "status", "elapsed_s", "fired_at_import", "sabotaged_functions", "error")}
        | {"fired": len(run.canary.get("fired") or []), "surfaced": run.surfaced()},
        "oracle": run.oracle,
        "grade": graded.as_dict() if graded is not None else None,
        "valid": graded.passed if graded is not None else None,
    }


def run_all(names: Sequence[str] | None = None, *, workers: int = 3,
            timeout: int = TIMEOUT_S) -> dict[str, Any]:
    """Measure every harness (in parallel subprocesses) and publish the artefact."""
    names = list(names or harnesses())
    with ThreadPoolExecutor(max_workers=workers) as pool:
        runs = list(pool.map(lambda n: measure(n, timeout=timeout), names))
    rows = [_summarise(r) for r in runs]
    fleet = all_or_nothing({r["harness"]: None if r["valid"] is None else float(r["valid"])
                            for r in rows})
    failures: dict[str, int] = {}
    for r in rows:
        for check in (r["grade"] or {}).get("checks", []):
            if check["passed"] is not True:
                failures[check["name"]] = failures.get(check["name"], 0) + 1
    return {
        "generated": datetime.now(UTC).isoformat(timespec="seconds"),
        "method": "AgentDojo-style canary per harness: reachability of the code standing.py "
                  "credits (read from sabotage canaries firing in the entry point; a profiled "
                  "run where they fire at import), a sabotage canary that must surface, and an "
                  "oracle/anti-oracle desk where the harness is on the eval spine; network "
                  "refused and writes contained in every run",
        "harnesses": len(rows),
        "run": sum(1 for r in rows if r["skipped"] is None),
        "valid": sum(1 for r in rows if r["valid"] is True),
        "invalid": sum(1 for r in rows if r["valid"] is False),
        "not_run": [r["harness"] for r in rows if r["valid"] is None],
        "fleet_valid_share": fleet.as_dict(),
        "failures_by_check": dict(sorted(failures.items(), key=lambda kv: -kv[1])),
        "rows": rows,
        "qwen_calls": 0,
    }


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description="harness-validity canary")
    parser.add_argument("--child")
    parser.add_argument("--mode", default="baseline")
    parser.add_argument("--out")
    parser.add_argument("--target", action="append", default=[])
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args(argv)
    if args.child:
        _child(args.child, args.mode, args.target, Path(args.out), profiled=args.profile)
        return 0
    from argus.eval import artefact

    report = run_all(args.only or None, workers=args.workers)
    artefact.write(REPORT_PATH, report)
    for row in report["rows"]:
        failed = [c["name"] for c in (row["grade"] or {}).get("checks", [])
                  if c["passed"] is not True]
        print(f"{row['harness']:32s} valid={row['valid']} failed={failed or '-'}")
    print(f"valid {report['valid']} / run {report['run']} / harnesses {report['harnesses']}")
    return 0



if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
