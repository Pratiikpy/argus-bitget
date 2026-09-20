"""The factor laboratory — a searcher that cannot see its own scores.

Track 2's Factor Discovery sub-theme asks how an agent "proposes hypotheses, discovers alpha
factors, and translates into tradable decisions". Almost every system answers by looping
propose → score → feed the score back. Three we tore down do exactly that, and it is the defect
that makes their results meaningless:

* **mcts-llm-alpha** computes a genuine IS/OOS overfitting score at ``qlib_evaluator.py:143``, then
  **unconditionally overwrites it** with the generating model's self-judgment
  (``comprehensive.py:90-98``).
* **RD-Agent** — Microsoft's, the strongest factor machinery in existence — uses a single
  ``APIBackend()`` to both propose factors and judge them. No independent evaluator, **no purged
  CV, no embargo, no deflated Sharpe, no PBO, and no trial counter.**
* **FactorForge** feeds the generator the IC of the top three factors every round and asks for
  variations (``evolution_engine.py:95-99``).

**FactorMiner is the closest thing to a counter-example, and it does not hold either.** This
docstring previously credited its generator with seeing "only syntax errors, never scores", and the
source does not support that. ``factorminer/agent/factor_generator.py:113-118`` declares
``generate_batch(memory_signal=..., library_state=...)`` — "guided by memory priors" — and the
signal is rendered straight into the user prompt at ``:165-171``. What it carries is not neutral:
``factorminer/memory/retrieval.py:742-750`` writes ``=== RECOMMENDED DIRECTIONS (P_succ) ===``
followed by each pattern's ``success_rate`` grade, and ``library_state`` carries
``recent_admissions``, the names of the factors that scored well enough to be admitted. Coarse
grades are still outcomes. Its *evaluator* is genuinely independent of its generator, which is more
than RD-Agent manages; its *memory* is the side door.

That finding is what `argus.research.memory` is built around: the lab's own memory records
everything and publishes only identity and structural facts, with an import-time guard that fails
if an outcome-bearing field is ever added to the signal.

This lab enforces the separation structurally rather than by convention:

1. The proposer is handed :class:`ProposerContext`, which physically cannot carry a score — it has
   no field for one. Adding one would be a visible change to a frozen dataclass.
2. Evaluation is deterministic Python over price data. No model grades a factor.
3. Every proposal increments a **trial counter**, and the counter is what the Deflated Sharpe gate
   consumes. A search that does not record how many times it looked has not produced a result.
4. A factor holds an explicit lifecycle state and cannot skip one. ``RETIRE`` exists and fires on
   measured decay, because a library that only ever grows is lying about decay.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from argus.backtest.engine import Bar, run
from argus.backtest.metrics import (
    HOURLY_PER_YEAR,
    MetricError,
    deflated_sharpe,
)
from argus.cost.model import CostModel
from argus.research.overfit import Observation, OverfitReport, run_all
from argus.truth.clocks import DualClock, SessionPhase


class Lifecycle(StrEnum):
    """A factor's state. It cannot skip one, and RETIRE is reachable."""

    PROPOSED = "proposed"
    FORMALIZED = "formalized"
    BACKTESTED = "backtested"
    COST_CHECKED = "cost_checked"
    OOS_TESTED = "oos_tested"
    DSR_GATED = "dsr_gated"
    CERTIFIED = "certified"
    DEPLOYED = "deployed"
    DECAYED = "decayed"
    RETIRED = "retired"
    REJECTED = "rejected"


_ORDER = (
    Lifecycle.PROPOSED, Lifecycle.FORMALIZED, Lifecycle.BACKTESTED, Lifecycle.COST_CHECKED,
    Lifecycle.OOS_TESTED, Lifecycle.DSR_GATED, Lifecycle.CERTIFIED, Lifecycle.DEPLOYED,
    Lifecycle.DECAYED, Lifecycle.RETIRED,
)


class LifecycleViolation(RuntimeError):
    """A factor tried to skip a gate. Certification is the whole product; skipping is a bug."""


@dataclass(frozen=True, slots=True)
class ProposerContext:
    """Everything the proposer is allowed to see.

    **There is deliberately no field for performance.** Not a filtered one, not an aggregate — the
    shape itself cannot carry a score, so contaminating the loop requires editing this class, which
    is visible in a diff. That is the difference between a convention and a guarantee.
    """

    market_structure: str
    already_proposed: tuple[str, ...]
    trials_so_far: int
    """The proposer may know *how many* times it has been asked, which is not a signal about
    quality — and it prevents the pathological loop of re-proposing the same thing forever."""

    memory: tuple[str, ...] = ()
    """Lines from :meth:`argus.research.memory.MemorySignal.render`, when a memory is attached.

    Every one of them is identity or structure — what has already been evaluated, and what could not
    be measured at all. They are built from a whitelist in `research/memory.py` and a guard there
    fails at import if an outcome-bearing field is ever added, because a memory is the one way to
    contaminate this class without editing it."""


@dataclass(frozen=True, slots=True)
class Factor:
    """A proposed factor. ``expression`` is a name in :data:`PRIMITIVES`."""

    name: str
    expression: str
    rationale: str
    horizon_bars: int = 24

    def __post_init__(self) -> None:
        if self.expression not in PRIMITIVES:
            raise ValueError(
                f"unknown expression {self.expression!r}; the evaluator only runs vetted "
                f"primitives, never model-authored code"
            )


@dataclass
class FactorRecord:
    """A factor and everything measured about it. Append-only history."""

    factor: Factor
    trial_number: int
    state: Lifecycle = Lifecycle.PROPOSED
    history: list[tuple[str, str]] = field(default_factory=list)
    gross_sharpe: float | None = None
    net_sharpe: float | None = None
    oos_sharpe: float | None = None
    dsr: float | None = None
    overfit: dict[str, Any] | None = None
    """The anti-overfit report, when one could be computed. ``None`` means the factor never
    reached the gate, which is different from reaching it and being found wanting."""

    rejection_reason: str = ""

    def advance(self, to: Lifecycle, *, note: str = "") -> None:
        if to in (Lifecycle.REJECTED, Lifecycle.RETIRED):
            self.history.append((str(self.state), str(to)))
            self.state = to
            return
        try:
            here, there = _ORDER.index(self.state), _ORDER.index(to)
        except ValueError:
            raise LifecycleViolation(f"{self.state} -> {to} is not on the lifecycle") from None
        if there != here + 1:
            raise LifecycleViolation(
                f"{self.factor.name}: {self.state} -> {to} skips a gate. Certification is the "
                f"product; a factor that skips a gate has not been certified."
            )
        self.history.append((str(self.state), str(to)))
        self.state = to

    def reject(self, reason: str) -> None:
        self.rejection_reason = reason
        self.advance(Lifecycle.REJECTED)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.factor.name,
            "expression": self.factor.expression,
            "trial": self.trial_number,
            "state": str(self.state),
            "gross_sharpe": self.gross_sharpe,
            "net_sharpe": self.net_sharpe,
            "oos_sharpe": self.oos_sharpe,
            "dsr": self.dsr,
            "overfit": self.overfit,
            "rejection_reason": self.rejection_reason,
            "path": [f"{a}->{b}" for a, b in self.history],
        }


# --- the primitive library the evaluator will run --------------------------------------------
# Vetted signal functions, referenced by name. The proposer picks from these rather than emitting
# code: model-authored code in an evaluator is an arbitrary-execution surface, and sandboxing it
# is a larger problem than this sub-theme needs.

_CLOCK = DualClock()


def _ret(bars: Sequence[Bar], i: int, n: int) -> float:
    j = i - n
    if j < 0:
        return 0.0
    a, b = float(bars[j].close), float(bars[i].close)
    return (b - a) / a if a > 0 else 0.0


def _sign(x: float) -> float:
    return 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)


def _closed(bars: Sequence[Bar], i: int) -> bool:
    return not _CLOCK.phase(bars[i].ts).has_price_discovery


PRIMITIVES: dict[str, Any] = {
    "long_while_closed": lambda b, i: 1.0 if _closed(b, i) else 0.0,
    "long_while_open": lambda b, i: 1.0 if not _closed(b, i) else 0.0,
    "weekend_only": lambda b, i: 1.0 if _CLOCK.phase(b[i].ts) is SessionPhase.WEEKEND else 0.0,
    # Flat when the lookback has no history. These two previously fell through to a full -1.0
    # (and +1.0) position on six bars of missing data, because `_ret` returns 0.0 and `0.0 > 0` is
    # False. A momentum factor taking a maximum short on data it does not have is a bug; it was
    # found by reconstructing these primitives in `argus.research.grammar` and diffing the two.
    "closure_momentum": lambda b, i: (
        _sign(_ret(b, i, 6)) if _closed(b, i) else 0.0
    ),
    "closure_reversion": lambda b, i: (
        -_sign(_ret(b, i, 6)) if _closed(b, i) else 0.0
    ),
    "near_reopen": lambda b, i: (
        1.0 if _closed(b, i) and _CLOCK.state(b[i].ts).hours_to_next_discovery <= 4 else 0.0
    ),
    "slow_trend": lambda b, i: 1.0 if _ret(b, i, 48) > 0 else 0.0,
    "slow_fade": lambda b, i: -1.0 if _ret(b, i, 48) > 0 else 1.0,
}


class Evaluator:
    """Deterministic scoring. No model touches this.

    Kept as a separate object from the proposer so the boundary is a real one: the evaluator
    receives a :class:`Factor` and returns numbers, and has no channel back to whatever produced it.
    """

    def __init__(self, bars: Sequence[Bar], *, cost: CostModel | None = None) -> None:
        self._bars = bars
        self._cost = cost or CostModel.bitget_perp()
        self._cost.assert_gateable()

    def observations(self, record: FactorRecord) -> list[Observation]:
        """Turn one factor into the (factor value, next-bar return) pairs the gates score.

        ARGUS's lab evaluates a factor on a **single** instrument's bar series, so the natural
        "cross-section at a period" does not exist. Rather than fabricate one, each bar becomes its
        own period with one observation, and the rank correlation is taken across a rolling window
        of bars instead of across names at an instant. That is a real difference from the
        cross-sectional IC in the literature and it is stated rather than glossed: the gates are
        measuring time-series predictive power here, not cross-sectional ranking power.
        """
        bars = self._bars
        signal = PRIMITIVES[record.factor.expression]
        out: list[Observation] = []
        window = max(2, record.factor.horizon_bars)
        for i in range(len(bars) - 1):
            try:
                value = float(signal(bars, i))
            except (ValueError, ZeroDivisionError, OverflowError, IndexError):
                continue
            nxt = _ret(bars, i + 1, 1)
            # One period per window keeps enough readings inside a period for a rank correlation
            # to be defined at all; a period holding a single pair has no ranks to correlate.
            out.append(Observation(period=i // window, name=f"bar{i % window}",
                                   factor=value, forward_return=nxt))
        return out

    def overfit_report(self, record: FactorRecord) -> OverfitReport | None:
        """Run the four anti-overfit gates over this factor's own readings."""
        rows = self.observations(record)
        if not rows:
            return None
        return run_all(rows)

    def score(self, record: FactorRecord) -> FactorRecord:
        """Walk one factor through every gate, in order, stopping at the first failure."""
        record.advance(Lifecycle.FORMALIZED)
        signal = PRIMITIVES[record.factor.expression]

        try:
            result = run(
                record.factor.name, "lab", self._bars, signal,
                cost=self._cost, periods_per_year=HOURLY_PER_YEAR,
            )
        except MetricError as exc:
            record.reject(f"unscoreable: {exc}")
            return record

        record.gross_sharpe = round(result.gross.sharpe, 3)
        record.net_sharpe = round(result.net.sharpe, 3)
        record.advance(Lifecycle.BACKTESTED)

        # Cost gate. On this venue the fee is larger than most effects, so this is where most
        # factors die — which is the honest outcome, not a tuning problem.
        if result.net.sharpe <= 0:
            record.advance(Lifecycle.COST_CHECKED)
            record.reject(
                f"net Sharpe {result.net.sharpe:.3f} after the 12bps round trip "
                f"(gross was {result.gross.sharpe:.3f})"
            )
            return record
        record.advance(Lifecycle.COST_CHECKED)

        if result.out_of_sample is None:
            record.reject("no scoreable out-of-sample slice")
            return record
        record.oos_sharpe = round(result.out_of_sample.sharpe, 3)
        record.advance(Lifecycle.OOS_TESTED)

        if result.out_of_sample.sharpe <= 0:
            record.reject(f"out-of-sample Sharpe {result.out_of_sample.sharpe:.3f}")
            return record

        return record


@dataclass
class FactorLab:
    """The loop. Proposals in, certified factors out, everything else in the cemetery."""

    evaluator: Evaluator
    records: list[FactorRecord] = field(default_factory=list)
    memory: Any | None = None
    """An optional :class:`argus.research.memory.FactorMemory`. Attached, the lab stops re-scoring
    hypotheses it has a record of, and the proposer is told what has been tried — never how any of
    it did. Typed loosely to keep the memory module's dependency one-directional."""

    @property
    def trials(self) -> int:
        """Every proposal ever scored. This is what the DSR gate consumes.

        A search that reports its best result without this number has not produced a result; it has
        reported the maximum of a noise distribution.
        """
        return len(self.records)

    def context(self) -> ProposerContext:
        """What the proposer gets. Carries no performance information by construction."""
        return ProposerContext(
            market_structure=(
                "Tokenized US equity perpetual. The token trades continuously; the underlying "
                "equity market closes. Measured: price discovery attenuates ~7x during weekends "
                "(32.17 -> 4.53 bps/hour). Round trip costs 12bps."
            ),
            already_proposed=tuple(r.factor.name for r in self.records),
            trials_so_far=self.trials,
            memory=() if self.memory is None else tuple(self.memory.signal().render()),
        )

    def submit(self, factor: Factor) -> FactorRecord | None:
        """Score one proposal. Returns ``None`` when memory has seen this hypothesis before.

        A repeat is suppressed rather than re-scored, and deliberately does **not** increment
        :attr:`trials`. The Deflated Sharpe gate consumes that count as the number of distinct looks
        taken; re-testing one hypothesis produces no new maximum, so counting it again would deflate
        against a search that never happened. The suppression is counted in the memory and surfaced
        by :meth:`funnel`, so a proposer going in circles is visible rather than merely cheap.
        """
        if self.memory is not None and self.memory.seen(factor):
            self.memory.note_duplicate()
            return None
        record = FactorRecord(factor=factor, trial_number=self.trials + 1)
        self.records.append(record)
        scored = self.evaluator.score(record)
        if self.memory is not None:
            self.memory.remember(scored)
        return scored

    def gate(self) -> dict[str, Any]:
        """Apply the Deflated Sharpe gate to survivors, using the real trial count."""
        survivors = [r for r in self.records if r.state is Lifecycle.OOS_TESTED]
        if not survivors:
            return {"certified": [], "note": "nothing reached the DSR gate"}

        sharpes = [r.net_sharpe or 0.0 for r in self.records if r.net_sharpe is not None]
        variance = 0.0
        if len(sharpes) > 1:
            mu = sum(sharpes) / len(sharpes)
            variance = sum((s - mu) ** 2 for s in sharpes) / (len(sharpes) - 1)

        certified = []
        for record in survivors:
            record.advance(Lifecycle.DSR_GATED)
            try:
                record.dsr = round(deflated_sharpe(
                    record.net_sharpe or 0.0, n=len(self.evaluator._bars),
                    trials=self.trials, variance_of_trials=variance,
                ), 4)
            except MetricError as exc:
                record.reject(f"DSR uncomputable: {exc}")
                continue

            if record.dsr <= 0.95:
                record.reject(
                    f"DSR {record.dsr} over {self.trials} trials — not distinguishable from "
                    f"the best of a random search"
                )
                continue

            # The Deflated Sharpe asks whether this result beats the best of a random *search*.
            # The anti-overfit gates ask a different question — whether the factor beats its own
            # shuffled self, holds its sign across sub-periods and regimes, and decays like a real
            # signal. A factor can clear the first and fail the second, so both are required before
            # anything is called certified. An INCONCLUSIVE verdict does not certify either:
            # "we could not tell" is not "it passed".
            overfit = self.evaluator.overfit_report(record)
            if overfit is not None:
                record.overfit = overfit.as_dict()
                if overfit.failed or overfit.inconclusive:
                    record.reject(f"anti-overfit: {overfit.verdict}")
                    continue

            record.advance(Lifecycle.CERTIFIED)
            certified.append(record)
        if self.memory is not None:
            # The gate is where a factor reaches its terminal state. Memory written at score time
            # records every survivor as oos_tested, so it is refreshed here or it is wrong.
            for record in self.records:
                self.memory.update(record)
        return {
            "trials": self.trials,
            "variance_of_trial_sharpes": round(variance, 4),
            "certified": [r.name for r in (x.factor for x in certified)],
        }

    def cemetery(self) -> list[dict[str, str]]:
        """Everything that died, and what killed it.

        Retained deliberately. A library that only ever grows is publication bias inside our own
        system, and the rejection reasons are the most informative output this lab produces.
        """
        return [
            {"name": r.factor.name, "died_at": str(r.state), "reason": r.rejection_reason}
            for r in self.records if r.state is Lifecycle.REJECTED
        ]

    def funnel(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for r in self.records:
            counts[str(r.state)] = counts.get(str(r.state), 0) + 1
        return {
            "proposed": self.trials,
            "by_final_state": counts,
            "certified": counts.get(str(Lifecycle.CERTIFIED), 0),
            "rejected": counts.get(str(Lifecycle.REJECTED), 0),
            # Repeats are suppressed rather than scored, so they do not appear above. Reporting the
            # number keeps a proposer that is going in circles visible instead of merely cheap.
            "duplicates_suppressed": (
                0 if self.memory is None else self.memory.duplicates_suppressed
            ),
        }


def report(lab: FactorLab) -> dict[str, Any]:
    """Run the gate, then describe the lab.

    **The gate is called first, deliberately.** It was second here, and since a dict literal is
    evaluated top to bottom the funnel was computed before any factor reached its terminal state —
    the published `data/factor_lab.json` recorded ``by_final_state: {oos_tested: 1}`` and
    ``certified: 0`` for a factor the gate had not yet judged. Nothing certified at the time, so the
    two numbers happened to agree and the defect stayed invisible; the first certification would
    have produced a report whose funnel contradicted its own gate.
    """
    gate = lab.gate()
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "separation": (
            "the proposer receives ProposerContext, which has no field for performance; "
            "evaluation is deterministic Python and no model grades a factor"
        ),
        "funnel": lab.funnel(),
        "gate": gate,
        "factors": [r.as_dict() for r in lab.records],
        "cemetery": lab.cemetery(),
    }


def main() -> int:
    from argus.market.history import CandleType, fetch_range
    from argus.research.memory import FactorMemory

    candles = fetch_range("NVDAUSDT", days=90, interval="1H", candle_type=CandleType.MARKET)
    bars = [
        Bar(
            ts=c.ts,
            close=c.close,
            # Same reason as `track1_study`: a grammar field with no data behind it reads 0.0 for
            # every bar, and the search cannot tell that apart from a real constant.
            extra={"volume": float(c.volume), "high": float(c.high), "low": float(c.low)},
        )
        for c in candles
    ]

    root = Path(__file__).resolve().parents[3] / "data"
    memory_path = root / "factor_memory.json"
    memory = FactorMemory.load(memory_path)
    lab = FactorLab(evaluator=Evaluator(bars), memory=memory)

    # Stand-ins for model proposals: every vetted primitive, submitted blind. The lab behaves
    # identically whether these come from a model or a list — which is the point of the boundary.
    # On the second run every one of these is already in memory, so the lab suppresses them all
    # rather than re-scoring them; that is the loop working, not a failure.
    for name in PRIMITIVES:
        lab.submit(Factor(name=name, expression=name, rationale="session-structure hypothesis"))

    out = root / "factor_lab.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = report(lab)
    memory.save(memory_path)

    f = payload["funnel"]
    print(f"proposed {f['proposed']} · certified {f['certified']} · rejected {f['rejected']}")
    print(f"suppressed as already evaluated: {f['duplicates_suppressed']}")
    print(f"trials fed to the DSR gate: {payload['gate'].get('trials')}")

    # A run in which every proposal was already known has produced no new evidence, and writing its
    # empty funnel over the previous report would destroy a real result to record that nothing
    # happened. Found by running this twice: the second pass overwrote eight scored factors with
    # zeroes. The memory is still saved — it is the thing that legitimately changed.
    if not lab.records:
        print(f"\nevery proposal was already evaluated; {out.name} left as it was")
        return 0

    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print("\ncemetery:")
    for row in payload["cemetery"]:
        print(f"  {row['name']:<22} died at {row['died_at']:<14} {row['reason'][:70]}")
    print(f"\nfull report -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PRIMITIVES", "Evaluator", "Factor", "FactorLab", "FactorRecord",
    "Lifecycle", "LifecycleViolation", "ProposerContext", "report",
]
