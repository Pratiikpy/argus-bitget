"""The ablations this desk can actually run today, run.

:mod:`argus.eval.ablation` is the instrument. This is the experiment. The capability register's
standing complaint about the perception layer was that "source COUNT is no longer the gap; source
EFFECT is untested", and the same was true of every deterministic component in the decision path:
each one was argued for and none was measured.

**Only the deterministic components are here, and that is the honest boundary.** Removing an
analyst changes what a model reasons over, and a model's answer differs between two runs on the
same input, so that ablation needs the paired protocol in
:func:`argus.eval.ablation.paired` and a budget of real cycles. The components below are pure
functions of the frame — given the same evidence they always produce the same panel — so removing
one and replaying gives an exact count, with no statistics and nothing to be uncertain about.

**The frames are gathered live rather than replayed from the ledger, because the ledger does not
hold them.** A decision record carries the verdict, the thesis and the hashes of the state; it
does not carry the evidence list, by design — the evidence is large, and hashing the state was
judged enough. That means a replay of a past cycle's *selection* is not reconstructible, and
pretending otherwise by re-gathering today's evidence and calling it last week's would be the
worst kind of backtest. So each run of this module gathers the current evidence for the whole
universe and ablates against that, and the artefact records the moment it was gathered.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from argus.agents.selection import select
from argus.eval.ablation import DeterministicResult, deterministic
from argus.market.bitget import RTOKEN_SYMBOLS
from argus.market.evidence import gather
from argus.truth.clocks import DualClock
from argus.truth.evidence import Evidence

OUT_PATH = Path(__file__).resolve().parents[3] / "data" / "ablations.json"

LOOKBACK = timedelta(days=14)
"""How far back evidence is gathered for each frame. Matches the desk's own cycle window."""

ANALYSTS = ("event", "sentiment", "earnings")

MACRO_FEEDS = ("treasury_curve", "fear_greed", "implied_volatility")
"""The instrument-independent items `argus.paper.runner` gives every panel.

Named as data because the artefact used to *assert* that every frame carries all three, in prose,
while three ``except Exception: pass`` blocks could silently remove any of them. A claim that a
failure path can falsify has to be computed from what actually happened, not written in advance.
"""


@dataclass(frozen=True, slots=True)
class MacroCoverage:
    """Which instrument-independent feeds were obtained, and which were not, with the reason.

    **This is the fix for a silent failure that could invalidate a whole run.** Every frame in an
    ablation shares these items, so one dead feed does not shrink the universe — it thins *every*
    frame at once. The result would still be produced, still look complete, and be measuring a
    desk that sees less than the real one, while the artefact's own `frame_completeness` note went
    on claiming all three were present.
    """

    obtained: tuple[str, ...]
    failed: tuple[tuple[str, str], ...]
    """``(feed, error)`` for each feed that did not answer. Never discarded."""

    @property
    def complete(self) -> bool:
        return not self.failed

    def as_dict(self) -> dict[str, Any]:
        return {
            "expected": list(MACRO_FEEDS),
            "obtained": list(self.obtained),
            "failed": [{"feed": f, "error": e} for f, e in self.failed],
            "complete": self.complete,
        }


@dataclass(frozen=True, slots=True)
class FrameSet:
    """The frames an ablation will run on, and everything that did not make it into them."""

    rows: tuple[tuple[str, tuple[Evidence, ...]], ...]
    macro: MacroCoverage

    dropped: tuple[tuple[str, str], ...]
    """``(symbol, reason)`` for each symbol excluded.

    The count alone was reported before, which told a reader the universe had shrunk but not which
    instruments left it or why — and a universe that loses its liquid names is a different
    experiment from one that loses its illiquid ones.
    """

    empty: tuple[str, ...]
    """Symbols whose feeds answered but carried no evidence.

    Distinct from `dropped`: a feed that said nothing and a feed that failed are not the same
    observation, and collapsing them would hide which one happened.
    """

    def __len__(self) -> int:
        return len(self.rows)

    def as_dict(self) -> dict[str, Any]:
        return {
            "frames": len(self.rows),
            "symbols": [s for s, _ in self.rows],
            "evidence_per_frame": {s: len(e) for s, e in self.rows},
            "macro": self.macro.as_dict(),
            "dropped": [{"symbol": s, "reason": r} for s, r in self.dropped],
            "empty": list(self.empty),
        }


def frames(
    *, symbols: tuple[str, ...] = RTOKEN_SYMBOLS, now: datetime | None = None
) -> FrameSet:
    """One frame per symbol: its live evidence, as the desk would see it this instant.

    A symbol whose feeds all fail is excluded rather than entered as an empty frame. An empty
    frame would make every component look inert on it, which is a statement about the feed and
    would be read as a statement about the component.

    Returns a :class:`FrameSet` rather than a bare list so that what is *missing* travels with
    what is present. Every exclusion here — a dead macro feed, a symbol that raised, a symbol that
    returned nothing — used to be swallowed, and an ablation reads as a statement about a
    component when it may be a statement about a feed.
    """
    at = now or datetime.now(UTC)
    shared, coverage = _instrument_independent(at)
    out: list[tuple[str, tuple[Evidence, ...]]] = []
    dropped: list[tuple[str, str]] = []
    empty: list[str] = []
    for symbol in symbols:
        try:
            gathered = gather(symbol, as_of=at, lookback=LOOKBACK)
        except Exception as exc:
            # One dead feed must not cost the whole experiment, but which symbol left and why is
            # recorded rather than left to be inferred from a shrunken count.
            dropped.append((symbol, f"{type(exc).__name__}: {exc}"))
            continue
        if gathered.evidence:
            out.append((symbol, tuple(gathered.evidence) + shared))
        else:
            empty.append(symbol)
    return FrameSet(tuple(out), coverage, tuple(dropped), tuple(empty))


def _instrument_independent(at: datetime) -> tuple[tuple[Evidence, ...], MacroCoverage]:
    """The macro items every frame carries: Treasury curve, risk appetite, implied volatility.

    Included because `argus.paper.runner` gives them to every symbol's panel, and a frame missing
    them would under-count the evidence the selection layer weighs. What is still missing is named
    in the artefact rather than glossed: the Bitget Skills, consensus estimates, insider Form 4s
    and XBRL fundamentals are per-symbol and each costs a live call per instrument, so a frame here
    is a **lower bound** on what the desk really sees and the result is read accordingly.
    """
    out: list[Evidence] = []
    obtained: list[str] = []
    failed: list[tuple[str, str]] = []

    def attempt(feed: str, fetch: Callable[[], list[Evidence]]) -> None:
        """Run one feed and record the outcome either way.

        The three blocks this replaces were ``except Exception: pass``. A feed that failed left no
        trace anywhere — not in the return value, not in the artefact, not in a log — while every
        frame in the run quietly lost the items it would have carried.
        """
        try:
            items = fetch()
        except Exception as exc:
            failed.append((feed, f"{type(exc).__name__}: {exc}"))
            return
        out.extend(items)
        obtained.append(feed)

    def curve() -> list[Evidence]:
        from argus.market.macro import evidence as curve_evidence
        from argus.market.macro import latest as latest_curve

        return list(curve_evidence(latest_curve(), as_of=at))

    def appetite() -> list[Evidence]:
        from argus.market.macro import fear_greed_evidence, fetch_fear_greed

        return list(fear_greed_evidence(fetch_fear_greed(), as_of=at))

    def vol() -> list[Evidence]:
        from argus.market.volatility import evidence as vix_evidence
        from argus.market.volatility import fetch as fetch_vix

        return list(vix_evidence(fetch_vix(), as_of=at))

    attempt("treasury_curve", curve)
    attempt("fear_greed", appetite)
    attempt("implied_volatility", vol)
    return tuple(out), MacroCoverage(tuple(obtained), tuple(failed))


def selection_ablation(
    frames_in: list[tuple[str, tuple[Evidence, ...]]],
    *,
    now: datetime,
    deliberation_bps: Decimal,
) -> DeterministicResult:
    """Does evidence-conditioned analyst selection change the panel, or is it decoration?

    With the component: `argus.agents.selection.select` decides who runs from what the evidence
    actually carries. Without it: everyone runs, which is what the desk did before the layer
    existed and what most systems in the corpus still do.

    The compared value is the **set of analysts that ran**, sorted, so the answer is "these frames
    produced a different panel" rather than a difference in some downstream number that a model
    also touched.
    """
    def with_selection(evidence: tuple[Evidence, ...]) -> tuple[str, ...]:
        return tuple(sorted(
            select(
                list(evidence), as_of=now, deliberation_bps=deliberation_bps, analysts=ANALYSTS
            ).run
        ))

    def without_selection(evidence: tuple[Evidence, ...]) -> tuple[str, ...]:
        return tuple(sorted(ANALYSTS))

    return deterministic(
        [(symbol, ev) for symbol, ev in frames_in],
        component="analyst selection (agents/selection.py)",
        with_component=with_selection,
        without_component=without_selection,
    )


def as_of_gate_ablation(
    frames_in: list[tuple[str, tuple[Evidence, ...]]], *, now: datetime
) -> DeterministicResult:
    """Does the point-in-time gate remove anything, or is every feed already behaving?

    The most load-bearing correctness control in the system and the one least likely to be
    noticed when it stops working: it drops evidence timestamped after the decision instant. If it
    never fires on live data that is worth knowing — it means the gate is untested in production
    and its protection is theoretical, which is a different statement from "the gate is working".
    """
    def gated(evidence: tuple[Evidence, ...]) -> int:
        return sum(1 for e in evidence if e.available_at <= now)

    def ungated(evidence: tuple[Evidence, ...]) -> int:
        return len(evidence)

    return deterministic(
        [(symbol, ev) for symbol, ev in frames_in],
        component="as-of evidence gate",
        with_component=gated,
        without_component=ungated,
    )


def run(*, out: Path = OUT_PATH, now: datetime | None = None) -> dict[str, Any]:
    """Run every deterministic ablation and write the artefact."""
    at = now or datetime.now(UTC)
    session = DualClock().state(at)
    gathered = frames(now=at)
    # The desk's own per-cycle reasoning cost, so selection is ablated at the price it really pays.
    from argus.agents.meta_pm import deliberation_cost_bps
    from argus.llm.qwen import Thinking

    deliberation = round(
        deliberation_cost_bps(session, thinking=Thinking.FULL, annualised_vol=Decimal("0.45")), 2
    )
    results = [
        selection_ablation(list(gathered.rows), now=at, deliberation_bps=deliberation),
        as_of_gate_ablation(list(gathered.rows), now=at),
    ]
    report = {
        "generated_at": at.isoformat(),
        "session_phase": str(session.phase),
        "deliberation_bps": str(deliberation),
        **gathered.as_dict(),
        "note": (
            "Deterministic components only. Ablating an analyst needs the paired protocol in "
            "argus.eval.ablation.paired and a budget of real cycles, because a model's answer "
            "differs between two runs on the same input."
        ),
        "frame_completeness": _completeness(gathered),
        "results": [r.as_dict() for r in results],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _completeness(gathered: FrameSet) -> str:
    """Say what the frames actually carried — read from the run, not asserted in advance.

    The sentence this replaces stated flatly that every frame carried all three macro items. It
    was written once and could not notice a dead feed, which is exactly the failure the coverage
    record now makes impossible to publish silently.
    """
    base = (
        "Each frame carries the RSS and EDGAR evidence for its symbol. It does NOT carry the "
        "Bitget Skills, consensus estimates, Form 4 insider records or XBRL fundamentals, each "
        "of which costs a live call per instrument, so a frame here is a lower bound on what the "
        "desk sees and a component shown to change a decision here would change it at least as "
        "often in production. "
    )
    macro = gathered.macro
    if macro.complete:
        return base + (
            "All three instrument-independent macro items the runner gives every panel "
            f"({', '.join(MACRO_FEEDS)}) were present in this run."
        )
    missing = ", ".join(f"{feed} ({error})" for feed, error in macro.failed)
    return base + (
        f"**This run is thinner than that.** {len(macro.failed)} of {len(MACRO_FEEDS)} "
        f"instrument-independent macro feeds did not answer: {missing}. Those items are shared by "
        "every frame, so their absence thins the whole experiment at once rather than shrinking "
        "the universe, and a component that looks inert here may only be missing its input."
    )


def main() -> int:  # pragma: no cover - CLI
    import contextlib
    import sys

    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    report = run()
    print(f"{report['frames']} live frame(s), {report['session_phase']} session")
    for row in report["results"]:
        print()
        print(f"{row['component']}: {row['changed']} of {row['frames']} frame(s) differ "
              f"({row['share_changed']:.1%})")
        if row["inert"]:
            print("  inert on every frame tested")
        for change in row["changes"][:6]:
            print(f"  {change['frame_id']}: with={change['with']} without={change['without']}")
    return 0


__all__ = ["ANALYSTS", "LOOKBACK", "OUT_PATH", "as_of_gate_ablation", "frames", "run",
           "selection_ablation"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
