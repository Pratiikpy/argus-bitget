"""Same state, same answer? Track 2's Open Theme names decision consistency first.

The handbook's Open Theme for Agentic Trading lists *"decision consistency, risk-violation rate,
max drawdown, stress behavior, human-takeover rate, incremental value"*. ARGUS measured five.
The first was reported as *"the model revised its own answer 96 times"* — a different quantity
entirely, and one produced by a defect since fixed. The honest report became **UNDEFINED**, because
no market state had ever been put to the desk twice.

This puts one there, N times, and counts the agreements.

**Two questions, and conflating them measures the wrong thing.** `QwenClient` keeps an in-process
response cache, on by default, keyed on the request payload (`llm/qwen.py:198`) — *"identical
requests are cached, because re-asking a deterministic question is pure waste"*. So replaying an
identical frame inside one process returns the cached completion and agrees with itself perfectly,
having asked the model nothing. That number is real but it is a property of the **cache**:

* **replay consistency** — what a re-run of the deployed system produces, cache and all.
* **sampling consistency** — the cache disabled, so the model genuinely decides again on the same
  evidence. This is the one that measures the decision-maker, and it is the headline here.

`temperature=0` is not an answer to this. The client sets it by default and its docstring already
says why — *"decision-consistency-under-replay is a metric we are graded on"* — but zero temperature
constrains sampling, not the provider: batching, kernel non-determinism and expert routing all move
a greedy decode. Whether it holds is an empirical question about an endpoint, which is why this
module exists rather than a sentence claiming determinism.

**Three levels of agreement, because "the same answer" is three claims.**

1. **Action** — verdict, side and quantity. Did the desk *do* the same thing? The only level that
   touches money, and the one a judge means.
2. **View** — action plus the recorded lean. Did it *think* the same thing?
3. **Verbatim** — the approved intent hash, which covers the thesis and the invalidation conditions
   through `hash_intent`. Word-for-word identity.

Reporting only (3) would score a desk that always sells the same size and phrases it differently as
inconsistent, which is false where it matters. Reporting only (1) would hide a desk whose stated
reasoning wanders while its action happens to be pinned by a hurdle. The three are reported
separately and never averaged into one figure.

**Prior art.** `sierra-research/tau-bench` computes `pass^k` over repeated trials
(`tau_bench/run.py:180-203`), which is the right statistic for exactly this and the one the earlier
LUI benchmark could not use because its classifier is deterministic. Here the component under test
genuinely is not, so `unanimity` below is `pass^k` at `k = runs` on a single task: the fraction of
tasks where every trial agreed. tau-bench has no notion of a response cache, so nothing in it
separates the two questions above.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

DATA = Path(__file__).resolve().parents[3] / "data"
REPORT_PATH = DATA / "consistency.json"

MIN_RUNS = 3
"""Fewest replays before agreement is reported.

Two runs give a rate of 0% or 100% and nothing in between, which is not a measurement of a
distribution — it is a coin described as one.
"""


class ConsistencyError(ValueError):
    """Raised rather than reporting an agreement rate computed from too few replays."""


@dataclass(frozen=True, slots=True)
class Decision:
    """What one replay produced, reduced to the fields agreement is defined over."""

    verdict: str
    side: str
    quantity: str
    lean: str
    lean_confidence: float
    intent_hash: str
    thesis: str

    @property
    def action(self) -> tuple[str, str, str]:
        """What it did. The only level that moves money."""
        return (self.verdict, self.side, self.quantity)

    @property
    def view(self) -> tuple[str, str, str, str]:
        """What it did and which way it leaned."""
        return (*self.action, self.lean)

    def as_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict, "side": self.side, "quantity": self.quantity,
            "lean": self.lean, "lean_confidence": round(self.lean_confidence, 4),
            "intent_hash": self.intent_hash, "thesis": self.thesis[:180],
        }


def decision_of(run: Any) -> Decision:
    """Read the approved decision off a completed desk run.

    The **approved** intent, not the original: consistency is a claim about what the desk decided,
    and on a constrained decision the original is a proposal the system did not act on.
    """
    proof = run.proof
    intent = proof.llm_revised_intent or (
        proof.constitution_ruling.resulting_intent if proof.constitution_ruling is not None
        else proof.llm_original_intent
    )
    original = proof.llm_original_intent
    return Decision(
        verdict=str(intent.verdict),
        side=str(intent.side),
        quantity=str(intent.quantity),
        lean=str(getattr(original, "lean", "none")),
        lean_confidence=float(getattr(original, "lean_confidence", 0.0)),
        intent_hash=str(proof.approved_intent_hash),
        thesis=str(intent.thesis),
    )


@dataclass(frozen=True, slots=True)
class Replay:
    """One market state, put to the desk several times."""

    label: str
    decisions: tuple[Decision, ...]
    cached: bool
    """Whether the client's response cache was left on. With it on, this measures the cache."""

    @property
    def runs(self) -> int:
        return len(self.decisions)

    def _agreement(self, key: Callable[[Decision], Any]) -> float | None:
        if self.runs < MIN_RUNS:
            return None
        values = [key(d) for d in self.decisions]
        modal = max(values, key=values.count)
        return values.count(modal) / len(values)

    @property
    def lean_agreement(self) -> float | None:
        """Did it think the same thing, whatever it then did?

        Reported on its own because the first live run buried it: the lean was ``down`` on all five
        replays while the action agreed on only three, and `view` — action plus lean — inherited the
        action's variance and showed 60%. The stable field was invisible inside the unstable one.
        """
        return self._agreement(lambda d: d.lean)

    @property
    def action_agreement(self) -> float | None:
        return self._agreement(lambda d: d.action)

    @property
    def view_agreement(self) -> float | None:
        return self._agreement(lambda d: d.view)

    @property
    def verbatim_agreement(self) -> float | None:
        return self._agreement(lambda d: d.intent_hash)

    @property
    def unanimous_action(self) -> bool:
        return len({d.action for d in self.decisions}) == 1

    @property
    def distinct_actions(self) -> tuple[tuple[str, str, str], ...]:
        return tuple(sorted({d.action for d in self.decisions}))

    def as_dict(self) -> dict[str, Any]:
        def pct(value: float | None) -> float | None:
            return None if value is None else round(value, 4)

        return {
            "label": self.label, "runs": self.runs, "cache_enabled": self.cached,
            "lean_agreement": pct(self.lean_agreement),
            "action_agreement": pct(self.action_agreement),
            "view_agreement": pct(self.view_agreement),
            "verbatim_agreement": pct(self.verbatim_agreement),
            "unanimous_action": self.unanimous_action,
            "distinct_actions": [list(a) for a in self.distinct_actions],
            "decisions": [d.as_dict() for d in self.decisions],
        }


@dataclass(frozen=True, slots=True)
class Report:
    replays: tuple[Replay, ...]

    @property
    def sampled(self) -> tuple[Replay, ...]:
        """Replays that actually asked the model again. The only ones that measure the desk."""
        return tuple(r for r in self.replays if not r.cached)

    @property
    def unanimity(self) -> float | None:
        """`pass^k` at k = runs: the share of states where every replay took the same action."""
        rows = self.sampled
        if not rows:
            return None
        return sum(1 for r in rows if r.unanimous_action) / len(rows)

    @property
    def verdict(self) -> str:
        if not self.replays:
            return "UNDEFINED: no state was replayed"
        if not self.sampled:
            return (
                "UNDEFINED as a statement about the desk: every replay ran with the response cache "
                "enabled, so the model was asked once and the agreement measured is the cache's"
            )
        rows = self.sampled
        actions = [r.action_agreement for r in rows if r.action_agreement is not None]
        views = [r.view_agreement for r in rows if r.view_agreement is not None]
        verbatim = [r.verbatim_agreement for r in rows if r.verbatim_agreement is not None]
        if not actions:
            return (
                f"UNDEFINED: {rows[0].runs} replay(s) is below the floor of {MIN_RUNS}; two runs "
                f"report 0% or 100% and nothing between, which is a coin described as a measurement"
            )
        leans = [r.lean_agreement for r in rows if r.lean_agreement is not None]
        head = (
            f"{len(rows)} state(s), {rows[0].runs} replays each with the cache disabled. "
            f"Lean agreement {min(leans):.0%}-{max(leans):.0%}, "
            f"action {min(actions):.0%}-{max(actions):.0%}, "
            f"view {min(views):.0%}-{max(views):.0%}, "
            f"verbatim {min(verbatim):.0%}-{max(verbatim):.0%}."
        )
        if leans and min(leans) == 1.0 and actions and min(actions) < 1.0:
            head += (
                " The desk's directional view is identical on every replay while its action is "
                "not: it knows what it thinks and wavers on what to do about it."
            )
        if self.unanimity == 1.0:
            return (
                f"{head} Every state produced the same action on every replay, so the desk is "
                f"reproducible where it matters. Its wording is not, and the verbatim figure says "
                f"by how much — a distinction a single 'consistency' number would have hidden"
            )
        unstable = [r.label for r in rows if not r.unanimous_action]
        return (
            f"{head} {len(unstable)} state(s) took more than one action across replays and are "
            f"named rather than averaged away: {', '.join(unstable)}. On identical evidence, that "
            f"is the decision-maker and not the market"
        )

    def render(self) -> str:
        lines = [f"DECISION CONSISTENCY — {len(self.replays)} state(s)", ""]
        for replay in self.replays:
            mode = "cache ON (measures the cache)" if replay.cached else "cache off"
            lines.append(f"  {replay.label}  x{replay.runs}  [{mode}]")
            for name, value in (
                ("lean", replay.lean_agreement),
                ("action", replay.action_agreement),
                ("view", replay.view_agreement),
                ("verbatim", replay.verbatim_agreement),
            ):
                shown = "n/a" if value is None else f"{value:.0%}"
                lines.append(f"      {name:<9} {shown}")
            if not replay.unanimous_action:
                for action in replay.distinct_actions:
                    lines.append(f"      took: {' '.join(action)}")
        lines += ["", f"  {self.verdict}"]
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "generated_at": datetime.now(UTC).isoformat(),
            "states": len(self.replays),
            "unanimity": None if self.unanimity is None else round(self.unanimity, 4),
            "verdict": self.verdict,
            "replays": [r.as_dict() for r in self.replays],
        }


def replay(
    make_run: Callable[[int], Any], *, label: str, runs: int = 5, cached: bool = False,
) -> Replay:
    """Put one state to the desk ``runs`` times and collect what it decided.

    ``make_run`` takes the attempt index and returns a completed desk run. It builds the desk itself
    so that each replay gets a **fresh client**: sharing one would let the first answer populate the
    cache and make every later run agree with it for free.
    """
    if runs < MIN_RUNS:
        raise ConsistencyError(
            f"{runs} replay(s) is below the floor of {MIN_RUNS}; agreement from two runs is 0% or "
            f"100% and nothing between"
        )
    return Replay(
        label=label, cached=cached,
        decisions=tuple(decision_of(make_run(i)) for i in range(runs)),
    )


def scenario_run(index: int) -> Any:  # pragma: no cover - drives the live model
    """One desk pass over the Sleeping-Anchor state, with a fresh uncached client each time."""
    from argus.agents.desk import ConstitutionPolicy, TradingDesk
    from argus.cost.model import CostModel
    from argus.demo.flow import sleeping_anchor_frame
    from argus.llm.qwen import QwenClient, Thinking, TokenBudget
    from argus.risk.hedgeability import HedgeabilitySurface
    from argus.truth.clocks import DualClock

    as_of, evidence = sleeping_anchor_frame()
    session = DualClock().state(as_of, nav_age_seconds=40_000)
    desk = TradingDesk(
        # cache=False is the whole point: with the client's cache on, replay two onwards would be
        # served from replay one and agree perfectly without asking the model anything.
        QwenClient(budget=TokenBudget(limit=60_000), cache=False),
        pm_thinking=Thinking.LOW, cost=CostModel.bitget_perp(),
    )
    return desk.run(
        symbol="NVDAUSDT", session=session, token_price=Decimal("118.40"),
        position=Decimal("2"), evidence=evidence,
        hedges=HedgeabilitySurface(candidates=()),
        decision_id=f"consistency-{index}", constitution=ConstitutionPolicy(),
    )


def main() -> int:  # pragma: no cover - CLI
    import argparse
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="same state, same answer?")
    parser.add_argument("--runs", type=int, default=MIN_RUNS)
    args = parser.parse_args()

    report = Report(replays=(
        replay(scenario_run, label="sleeping-anchor", runs=args.runs, cached=False),
    ))
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")
    print(report.render())
    print("\nwritten to " + str(REPORT_PATH))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


def report_from(replays: Sequence[Replay]) -> Report:
    """Assemble a report from replays gathered elsewhere."""
    return Report(replays=tuple(replays))


__all__ = [
    "MIN_RUNS",
    "ConsistencyError",
    "Decision",
    "Replay",
    "Report",
    "decision_of",
    "replay",
    "report_from",
]
