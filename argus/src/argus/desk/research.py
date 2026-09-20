"""One question, end to end — the research task the handbook requires as a deliverable.

Track 3's required materials are an accessible demo **and** "one complete research task (full flow
from question to actionable insight)". The audit in
``research/architecture/research-quality-audit.md`` found the honest position: ARGUS had six strong
subsystems — the earnings expectation gap, historical analogue retrieval, session-conditional beta,
portfolio impact, stress from realised history, execution cost — and nothing that ran them in order.
A person wanting to answer *"should I add NVDA to this book?"* had to call seven modules by hand and
assemble the answer themselves. Six good components and no seventh step is not a research desk; it
is a toolbox.

This is the seventh step. It takes one question, runs the chain in the order an analyst actually
works, and returns a recommendation in which **every number carries the module that produced it**.

    evidence → expectation gap → historical analogue → session beta
             → portfolio impact → stress → execution cost → recommendation

Three rules make the output worth reading, and each exists because its absence is the normal
failure of generated research:

**Every step may be absent, and absence is printed.** A step that could not run says so, with the
reason, and the recommendation is explicitly qualified by what is missing. The alternative — a
report whose sections silently disappear — reads as though the analysis was complete and narrower
than it was. :attr:`Finding.available` is false, never merely empty.

**The recommendation is assembled, not narrated.** The verdict comes from deterministic rules over
the findings, and the prose quotes the findings rather than restating them. Nothing here asks a
model to summarise, because a summary is where an unsourced number enters a document that otherwise
carries citations. Every figure in the report can be traced to the module and artefact that
computed it.

**A thin answer is a stated conclusion.** When too few steps produced anything, the verdict is
``INSUFFICIENT`` with the count — not a confident recommendation resting on two of seven inputs.

    python -m argus.desk.research --symbol NVDAUSDT --book "NVDAUSDT=0.3,AAPLUSDT=0.7"
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from itertools import pairwise
from typing import Any

MIN_STEPS_FOR_A_VERDICT = 4
"""Findings required before a directional recommendation is offered at all.

Four of seven. Below it the report says INSUFFICIENT and names what was missing, because a
recommendation resting on two inputs is a guess with a citation list."""


class Verdict(StrEnum):
    """What the research concluded. Deliberately not a price target."""

    ADD = "add"
    """The evidence supports the trade and the portfolio can carry it."""

    ADD_SMALLER = "add_smaller"
    """Supported, but the portfolio or the stress test argues for less than was proposed."""

    HOLD_OFF = "hold_off"
    """Nothing is wrong with the idea; the edge does not clear what it costs to take it."""

    DECLINE = "decline"
    """A specific finding argues against it."""

    INSUFFICIENT = "insufficient"
    """Too little of the chain ran to say. Not a soft decline — an absence of an answer."""


@dataclass(frozen=True, slots=True)
class Finding:
    """One step's contribution, with where it came from.

    ``available`` is the field that matters. A step that did not run is not an empty finding; it is
    a named absence, and the report prints it as one.
    """

    step: str
    available: bool
    headline: str
    source: str
    """The module and artefact that produced it, so a reader can re-derive the number."""

    detail: dict[str, Any] = field(default_factory=dict)
    concern: str = ""
    """Set when this step argues *against* the trade. Collected into the verdict."""

    def render(self) -> str:
        if not self.available:
            return f"  {self.step}: NOT AVAILABLE — {self.headline}"
        flag = "  ⚠ " if self.concern else "    "
        line = f"  {self.step}: {self.headline}\n{flag}source: {self.source}"
        return line + (f"\n      concern: {self.concern}" if self.concern else "")

    def as_dict(self) -> dict[str, Any]:
        return {
            "step": self.step, "available": self.available, "headline": self.headline,
            "source": self.source, "detail": self.detail, "concern": self.concern,
        }


@dataclass
class Report:
    """One research question, answered."""

    question: str
    symbol: str
    asked_at: datetime
    findings: list[Finding] = field(default_factory=list)
    verdict: Verdict = Verdict.INSUFFICIENT
    rationale: str = ""

    @property
    def available(self) -> list[Finding]:
        return [f for f in self.findings if f.available]

    @property
    def missing(self) -> list[Finding]:
        return [f for f in self.findings if not f.available]

    @property
    def concerns(self) -> list[Finding]:
        return [f for f in self.available if f.concern]

    @property
    def coverage(self) -> str:
        return f"{len(self.available)}/{len(self.findings)}"

    def render(self) -> str:
        lines = [
            f"# {self.question}",
            "",
            f"**{self.verdict.upper()}** — {self.rationale}",
            "",
            f"Chain coverage {self.coverage}. Every figure below names the module that computed "
            f"it; nothing here is summarised by a model.",
            "",
            "## What the chain found",
        ]
        lines.extend(f.render() for f in self.available)
        if self.concerns:
            lines += ["", "## What argues against it"]
            lines.extend(f"  - {f.step}: {f.concern}" for f in self.concerns)
        if self.missing:
            lines += ["", "## What could not be established"]
            lines.extend(f"  - {f.step}: {f.headline}" for f in self.missing)
            lines.append(
                "  These are absences, not zeros. The recommendation above is qualified by them."
            )
        lines += ["", f"_Asked {self.asked_at.isoformat()}._"]
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "question": self.question, "symbol": self.symbol,
            "asked_at": self.asked_at.isoformat(),
            "verdict": str(self.verdict), "rationale": self.rationale,
            "coverage": self.coverage,
            "findings": [f.as_dict() for f in self.findings],
            "concerns": [f.step for f in self.concerns],
            "missing": [f.step for f in self.missing],
        }


def decide(findings: Sequence[Finding], *, minimum: int = MIN_STEPS_FOR_A_VERDICT) -> tuple[
    Verdict, str
]:
    """Turn the findings into a verdict by rule, not by narration.

    Deterministic on purpose: the same findings must always produce the same recommendation, and a
    reader must be able to check the reasoning without re-running a model.
    """
    got = [f for f in findings if f.available]
    if len(got) < minimum:
        absent = [f.step for f in findings if not f.available]
        return Verdict.INSUFFICIENT, (
            f"only {len(got)} of {len(findings)} steps produced anything, below the {minimum} "
            f"needed for a recommendation. Missing: {', '.join(absent) or 'none named'}"
        )

    concerns = [f for f in got if f.concern]
    blocking = [f for f in concerns if f.detail.get("blocking")]
    if blocking:
        return Verdict.DECLINE, (
            f"{blocking[0].step} is disqualifying: {blocking[0].concern}"
        )
    if any(f.detail.get("cost_exceeds_edge") for f in got):
        return Verdict.HOLD_OFF, (
            "the idea is sound and the cost of taking it exceeds the edge it is expected to "
            "produce; this is a hurdle problem, not an evidence problem"
        )
    if concerns:
        return Verdict.ADD_SMALLER, (
            f"{len(concerns)} step(s) argue for less than was proposed: "
            + "; ".join(f"{f.step} — {f.concern}" for f in concerns[:3])
        )
    return Verdict.ADD, (
        f"{len(got)} of {len(findings)} steps ran and none argued against the trade"
    )


# --- the steps -----------------------------------------------------------------------------
#
# Each returns a Finding and never raises: a step that cannot run reports that it could not, and
# the chain continues. One dead upstream must not cost the whole answer.

NOT_ATTEMPTED = "not attempted"
"""Marker for a step the orchestrator never ran, as distinct from one that ran and found nothing.

**These were conflated and it put a false sentence in the flagship Track 3 artefact.** The report
said *"no comparable historical state was found within the distance threshold"* and *"no consensus
estimate was available for this name"* — both of which assert that a search was performed. Neither
search was performed: `main` initialised ``gap`` and ``analogue`` to None and never assigned them,
so the absence reason described work that had not taken place. An absence is only honest if it says
which kind it is.
"""


def _not_attempted(step: str, reason: str) -> Finding:
    """A step that was never run. Never phrased as though a search came back empty."""
    return Finding(
        step=step, available=False,
        headline=f"{NOT_ATTEMPTED} — {reason}",
        source="", concern="",
    )


def _absent(step: str, why: str) -> Finding:
    return Finding(step=step, available=False, headline=why, source="")


def evidence_step(evidence: Sequence[Any]) -> Finding:
    """How much evidence there is, counted once per story rather than once per copy.

    **The raw item count was the headline here and it is the wrong number when a feed
    syndicates.** Six wire copies of one story are six items and one piece of evidence, and every
    downstream confidence built on the count inherits the error. `agents/novelty.py` clusters
    near-duplicates and the corrected count is reported beside the raw one whenever they differ.
    """
    if not evidence:
        return _absent("evidence", "no evidence was gathered for this instrument")
    sources = sorted({str(getattr(e, "source", "?")) for e in evidence})
    detail: dict[str, Any] = {"items": len(evidence), "channels": sources}
    headline = f"{len(evidence)} item(s) across {len(sources)} channel(s): {', '.join(sources)}"
    concern = ""
    try:
        from argus.agents.novelty import cluster as cluster_evidence

        novelty = cluster_evidence(evidence)
    except Exception:
        novelty = None
    if novelty is not None:
        detail["distinct_stories"] = novelty.distinct_stories
        detail["duplication_ratio"] = round(novelty.duplication_ratio, 3)
        if novelty.distinct_stories < novelty.items:
            headline = (
                f"{len(evidence)} item(s) carrying {novelty.distinct_stories} distinct story(ies) "
                f"across {len(sources)} channel(s): {', '.join(sources)}"
            )
            concern = (
                f"the item count overstates the evidence by "
                f"{novelty.duplication_ratio:.1f}x; any confidence built on it is overstated by "
                f"the same factor"
            )
        if novelty.coordinated:
            concern = (
                f"{len(novelty.coordinated)} story(ies) arrived from several distinct sources "
                f"within two hours, which is a promoted narrative rather than corroboration"
            )
    return Finding(
        step="evidence", available=True,
        headline=headline[:300],
        source="market/evidence.py:gather + agents/novelty.py:cluster",
        concern=concern,
        detail=detail,
    )


def expectation_step(gap: Any, *, attempted: bool = True) -> Finding:
    if gap is None:
        if not attempted:
            return _not_attempted(
                "expectation gap",
                "the consensus lookup was not run for this question. It is available — "
                "market/estimates.py fetches it keylessly — so an absence here means the caller "
                "skipped the step, not that the data does not exist",
            )
        return _absent("expectation gap", "no consensus estimate was available for this name")
    try:
        headline = gap.render()
    except Exception:
        headline = str(gap)
    concern = ""
    if getattr(gap, "decelerating", False):
        concern = "growth is decelerating against expectations"
    return Finding(
        step="expectation gap", available=True, headline=headline[:300],
        source="desk/expectation.py:detect", concern=concern,
    )


def analogue_step(report: Any, *, attempted: bool = True) -> Finding:
    if report is None and not attempted:
        return _not_attempted(
            "historical analogue", "the analogue search was not run for this question"
        )
    if report is None or not getattr(report, "matches", ()):
        return _absent(
            "historical analogue",
            "no comparable historical state was found within the distance threshold",
        )
    dist = getattr(report, "distribution", None)
    hit = getattr(dist, "hit_rate", None) if dist else None
    headline = f"{len(report.matches)} comparable state(s)"
    if hit is not None:
        headline += f"; the move that followed was in the same direction {hit:.0%} of the time"
    concern = ""
    if hit is not None and hit < 0.5:
        concern = f"history went the other way {1 - hit:.0%} of the time after states like this"
    return Finding(
        step="historical analogue", available=True, headline=headline,
        source="desk/analogue.py:find", concern=concern,
        detail={"matches": len(report.matches), "hit_rate": hit},
    )


def shape_step(report: Any, *, attempted: bool = True) -> Finding:
    """The path-shape retrieval, which answers a different question from :func:`analogue_step`.

    Kept as its own step rather than folded into the analogue finding because the two can disagree,
    and a disagreement is information: a state-vector match says the market is of this *kind*, a
    shape match says the last day has *traced this line before*. Collapsing them into one headline
    would hide the case where one finds precedent and the other does not.
    """
    if report is None:
        if not attempted:
            return _not_attempted(
                "path shape", "the shape search was not run for this question"
            )
        return _absent("path shape", "no price history was available to match a shape against")
    if not getattr(report, "has_precedent", False):
        return Finding(
            step="path shape", available=True,
            headline=report.verdict[:300],
            source="desk/shapematch.py:find",
            concern=(
                "the present has no precedent here that noise on this same series does not "
                "also produce"
            ),
            detail={
                "analogues": len(report.analogues),
                "has_precedent": False,
                "null_p": report.null_p,
            },
        )
    share = report.upside_share
    headline = (
        f"{len(report.analogues)} non-overlapping precedent(s), nearest distance "
        f"{report.best.distance:.2f}; the next {report.horizon} bar(s) moved a median of "
        f"{report.median_outcome:+.2f}%"
    )
    concern = ""
    if share is not None and share < 0.5:
        concern = (
            f"the path went the other way {1 - share:.0%} of the time after shapes like this"
        )
    return Finding(
        step="path shape", available=True, headline=headline,
        source="desk/shapematch.py:find", concern=concern,
        detail={
            "analogues": len(report.analogues),
            "median_outcome_pct": report.median_outcome,
            "upside_share": share,
            "null_p": report.null_p,
        },
    )


def cointegration_step(report: Any, *, attempted: bool = True) -> Finding:
    """Is there a partner whose spread against this instrument is stationary and worth four legs?

    Reported even when the answer is no, and especially then: "we looked for a statistical-arbitrage
    relationship and there is none" is a result a reader can act on, while silence reads as though
    the question was never asked. The count of tests travels with the answer because at 11 candidate
    partners, one apparent hit at the 5% level is what chance alone produces.
    """
    if report is None:
        if not attempted:
            return _not_attempted(
                "cointegration", "no pairwise search was run for this instrument"
            )
        return _absent("cointegration", "no partner had enough overlapping history to test")
    tradeable = report.tradeable
    if not tradeable:
        naive = sum(1 for p in report.pairs if p.in_sample.pvalue <= 0.05)
        return Finding(
            step="cointegration", available=True,
            headline=(
                f"{report.tests} partner(s) tested, {naive} significant naively against "
                f"{report.expected_false_positives:.1f} expected by chance; none survives "
                f"multiple-testing correction, so no pair trade is available here"
            ),
            source="research/cointegration.py:scan",
            concern="",
            detail={"tests": report.tests, "survivors": 0},
        )
    best = min(tradeable, key=lambda p: p.in_sample.pvalue)
    partner = best.right if best.left.startswith(report.pairs[0].left) else best.left
    return Finding(
        step="cointegration", available=True,
        headline=(
            f"{len(tradeable)} partner(s) survive correction and clear the four-leg cost; "
            f"nearest is {partner} (p={best.in_sample.pvalue:.4f}, half-life "
            f"{best.half_life_bars:.0f} bars, entry needs |z| >= {best.required_z:.2f}, now "
            f"{best.z_now:+.2f})" if best.z_now is not None and best.half_life_bars is not None
            else f"{len(tradeable)} partner(s) survive correction"
        ),
        source="research/cointegration.py:scan",
        concern=(
            "a cointegrated pair is a bet that a relationship persists; it breaks without warning "
            "and the loss is unbounded on both legs"
        ),
        detail={"tests": report.tests, "survivors": len(tradeable)},
    )


def allocation_step(plan: Any, *, attempted: bool = True) -> Finding:
    """Not "is this trade acceptable?" but "is there a better book, and is it worth reaching?"

    Every other step grades the trade that was proposed. This one proposes a different one, which
    is the only step that can find a decision the model never considered — and then prices the
    move, because a better allocation reached by paying more than it is worth is not better.
    """
    if plan is None:
        if not attempted:
            return _not_attempted(
                "allocation", "no book was supplied, so no target allocation was computed"
            )
        return _absent("allocation", "the covariance needed to allocate could not be built")
    if not plan.trades:
        return Finding(
            step="allocation", available=True,
            headline="the book already sits at the hierarchical-risk-parity allocation",
            source="desk/allocation.py:optimize_trade",
        )
    concern = ""
    if plan.worth_doing:
        concern = (
            f"a different book is materially better: {plan.variance_reduction:+.0%} of variance "
            f"for {plan.cost_bps:.1f}bps, repaying in {plan.break_even_bars:.0f} bars"
        )
    return Finding(
        step="allocation", available=True,
        headline=plan.verdict[:300],
        source="desk/allocation.py:optimize_trade",
        concern=concern,
        detail={
            "legs": len(plan.trades),
            "turnover": plan.turnover,
            "cost_bps": plan.cost_bps,
            "variance_reduction": plan.variance_reduction,
            "worth_doing": plan.worth_doing,
        },
    )


def beta_step(impact: Any) -> Finding:
    if impact is None:
        return _absent(
            "session beta", "no return history was available to estimate session-conditional beta"
        )
    before = getattr(impact, "beta_before", None)
    after = getattr(impact, "beta_after", None)
    if before is None or after is None:
        return _absent(
            "session beta",
            "beta could not be estimated — the aligned return history was too short",
        )
    concern = ""
    if abs(before) > 1e-9 and after > before * 1.25:
        concern = (
            f"open-session beta rises from {before:.3f} to {after:.3f}, a material increase in "
            f"directional exposure during the only session that prices the underlying"
        )
    return Finding(
        step="session beta", available=True,
        headline=f"open-session beta {before:.3f} → {after:.3f}",
        source="desk/portfolio.py:assess (session=OPEN)", concern=concern,
        detail={"beta_before": before, "beta_after": after},
    )


def diversification_step(result: Any, hedges: Any = None) -> Finding:
    """How many bets the book really is, and what would hedge it.

    The two items on Track 3's Open Theme list that `desk/portfolio.py` could not supply: its
    inverse-Herfindahl count is not correlation-aware — its own docstring says so — and
    `risk/hedgeability.py` measures whether a hedge is *possible* rather than naming one.
    """
    if result is None:
        return _not_attempted(
            "diversification",
            "no aligned book history was available, so no rotation could be computed",
        )
    parts = [
        f"{result.effective_bets:.2f} effective bet(s) across {result.positions} position(s)"
    ]
    if result.torsion_bets is not None:
        parts.append(f"minimum torsion reads {result.torsion_bets:.2f}")
    concern = ""
    if result.concentration_ratio < 0.5:
        overstated = result.positions - result.effective_bets
        concern = (
            f"{result.largest_share:.0%} of the book's variance sits in one direction; the "
            f"position count overstates the diversification by {overstated:.1f} bets"
        )
    if hedges:
        best = hedges[0]
        parts.append(
            f"best hedge {best.instrument} at {best.ratio:+.3f} removes "
            f"{best.variance_reduction:.0%} of the variance"
            if best.useful
            else f"no candidate hedge removes more than {best.variance_reduction:.0%}"
        )
    return Finding(
        step="diversification", available=True, headline="; ".join(parts)[:300],
        source="desk/diversification.py:effective_bets", concern=concern,
        detail=result.as_dict(),
    )


def portfolio_step(impact: Any) -> Finding:
    if impact is None:
        return _absent("portfolio impact", "no book was supplied, so nothing could be compared")
    share_before = getattr(impact, "risk_share_before", None)
    share_after = getattr(impact, "risk_share_after", None)
    positions_after = getattr(impact, "effective_positions_after", None)
    correlated = getattr(impact, "max_correlation", None)

    parts: list[str] = []
    if share_before is not None and share_after is not None:
        parts.append(f"this name carries {share_before:.0%} → {share_after:.0%} of book risk")
    if positions_after is not None:
        parts.append(f"{positions_after:.2f} effective position(s) after")
    if correlated:
        name, rho = correlated
        parts.append(f"most correlated with {name} at {rho:.2f}")
    if not parts:
        return _absent(
            "portfolio impact",
            "the book produced no comparable risk figures — too few aligned observations",
        )

    concern = ""
    if share_after is not None and share_after > 0.5:
        concern = (
            f"this one name would carry {share_after:.0%} of the book's risk; that is a "
            f"concentration decision, not a position-sizing one"
        )
    elif correlated and correlated[1] > 0.85:
        concern = (
            f"it is {correlated[1]:.2f} correlated with {correlated[0]}, which is already held — "
            f"the book gains exposure without gaining diversification"
        )
    return Finding(
        step="portfolio impact", available=True, headline="; ".join(parts)[:300],
        source="desk/portfolio.py:assess", concern=concern,
        detail={"risk_share_after": share_after, "effective_positions_after": positions_after},
    )


def stress_step(report: Any) -> Finding:
    if report is None or not getattr(report, "results", ()):
        return _absent("stress", "no stress scenarios could be built for this instrument")
    failures = getattr(report, "failures", [])
    unexitable = getattr(report, "unexitable", [])
    headline = (
        f"{len(report.results)} scenario(s) tested, {len(failures)} breached the loss tolerance"
    )
    concern = ""
    detail: dict[str, Any] = {"scenarios": len(report.results), "failures": len(failures)}
    if unexitable:
        concern = (
            f"{len(unexitable)} scenario(s) leave a position that cannot be fully liquidated — "
            f"mark-to-market survival is not survival"
        )
        detail["blocking"] = True
    elif failures:
        concern = f"{len(failures)} scenario(s) breach the stated loss tolerance"
    return Finding(
        step="stress", available=True, headline=headline,
        source="desk/stress.py:assess", concern=concern, detail=detail,
    )


def execution_step(
    cost_bps: Decimal | None, expected_edge_bps: Decimal | None
) -> Finding:
    if cost_bps is None:
        return _absent("execution cost", "no cost model was supplied")
    headline = f"round trip plus deliberation costs {cost_bps}bps"
    concern = ""
    detail: dict[str, Any] = {"cost_bps": str(cost_bps)}
    if expected_edge_bps is not None:
        headline += f" against an expected edge of {expected_edge_bps}bps"
        detail["edge_bps"] = str(expected_edge_bps)
        if expected_edge_bps <= cost_bps:
            concern = (
                f"the expected edge ({expected_edge_bps}bps) does not clear what it costs to "
                f"take it ({cost_bps}bps)"
            )
            detail["cost_exceeds_edge"] = True
    return Finding(
        step="execution cost", available=True, headline=headline,
        source="cost/model.py + agents/meta_pm.py:deliberation_cost_bps",
        concern=concern, detail=detail,
    )


def research(
    *,
    question: str,
    symbol: str,
    evidence: Sequence[Any] = (),
    gap: Any = None,
    analogue: Any = None,
    shape: Any = None,
    pairs: Any = None,
    allocation: Any = None,
    diversification: Any = None,
    hedges: Any = None,
    attempted_gap: bool = True,
    attempted_analogue: bool = True,
    attempted_shape: bool = True,
    attempted_pairs: bool = True,
    attempted_allocation: bool = True,
    impact: Any = None,
    stress: Any = None,
    cost_bps: Decimal | None = None,
    expected_edge_bps: Decimal | None = None,
    now: datetime | None = None,
    minimum: int = MIN_STEPS_FOR_A_VERDICT,
) -> Report:
    """Run the chain over whatever was supplied and answer the question.

    Every input is optional. The orchestrator's contract is that a missing input becomes a named
    absence in the report rather than a silently narrower analysis, so a caller that can only
    supply four steps gets an answer that says it rests on four steps.
    """
    findings = [
        evidence_step(evidence),
        expectation_step(gap, attempted=attempted_gap),
        analogue_step(analogue, attempted=attempted_analogue),
        shape_step(shape, attempted=attempted_shape),
        cointegration_step(pairs, attempted=attempted_pairs),
        allocation_step(allocation, attempted=attempted_allocation),
        beta_step(impact),
        portfolio_step(impact),
        diversification_step(diversification, hedges),
        stress_step(stress),
        execution_step(cost_bps, expected_edge_bps),
    ]
    verdict, rationale = decide(findings, minimum=minimum)
    return Report(
        question=question, symbol=symbol, asked_at=now or datetime.now(UTC),
        findings=findings, verdict=verdict, rationale=rationale,
    )


def parse_book(text: str) -> Mapping[str, float]:
    """``NVDAUSDT=0.3,AAPLUSDT=0.7`` to a weight map. Raises on anything else."""
    out: dict[str, float] = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        name, _, weight = part.partition("=")
        if not weight:
            raise ValueError(f"{part!r} is not SYMBOL=weight")
        out[name.strip().upper()] = float(weight)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ARGUS research desk — one question, end to end")
    parser.add_argument("--symbol", default="NVDAUSDT")
    parser.add_argument("--question", default="")
    parser.add_argument("--book", default="", help="SYM=weight,SYM=weight — the current book")
    parser.add_argument("--size", type=float, default=0.1, help="proposed weight for the symbol")
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--save", default="")
    args = parser.parse_args(argv)

    # A Windows console defaults to cp1252 and cannot encode the arrows and warning marks this
    # report uses, so printing it raised UnicodeEncodeError on a stock terminal — a demo that
    # crashes on the machine a judge is most likely to run it on. Forcing UTF-8 on the way out
    # rather than stripping the characters: the report is also written to a file and read in a
    # browser, where they belong.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            with contextlib.suppress(ValueError, OSError):
                reconfigure(encoding="utf-8", errors="replace")

    question = args.question or f"Should I add {args.symbol} to this book?"
    now = datetime.now(UTC)

    evidence: list[Any] = []
    gap = analogue = impact = stress = shape = pairs = allocation = None
    diversification = None
    hedges: Any = None
    cost_bps: Decimal | None = None

    try:
        from argus.market.evidence import gather

        evidence = list(gather(args.symbol, as_of=now).evidence)
    except Exception:
        evidence = []

    try:
        from argus.cost.model import CostModel

        cost_bps = CostModel.bitget_perp().round_trip_bps()
    except Exception:
        cost_bps = None

    try:
        from argus.desk.stress import assess as stress_assess
        from argus.desk.stress import horizon_moves, phase_of_timestamp
        from argus.market.bitget import fetch_rtokens
        from argus.market.history import CandleType, fetch_range

        price = fetch_rtokens()[args.symbol].last
        candles = fetch_range(args.symbol, interval="1H", days=args.days,
                              candle_type=CandleType.MARKET)
        moves = horizon_moves([(c.ts, c.close) for c in candles], bars=2,
                              phase_of=phase_of_timestamp)
        stress = stress_assess(
            symbol=args.symbol, quantity=Decimal("1"), entry_price=price,
            loss_tolerance_pct=Decimal("5"), moves=moves, phase="rth", horizon_bars=2,
        )
    except Exception:
        stress = None

    # Portfolio impact and session beta, from real aligned history. Both come from one call:
    # `assess` returns the before/after of the same book, so asking twice would be two different
    # alignments of the same data and could disagree with itself.
    attempted_allocation = False
    if args.book:
        try:
            from argus.desk.portfolio import Session, align, assess, returns
            from argus.market.history import CandleType, fetch_range

            book = dict(parse_book(args.book))
            names = sorted(set(book) | {args.symbol})
            # Every series is aligned on the timestamps they all share, benchmark included.
            # Aligning per-pair instead would let beta and correlation be computed over different
            # bar counts and quietly disagree with each other.
            series: dict[str, dict[datetime, float]] = {}
            for name in [*names, "QQQUSDT"]:
                bars = fetch_range(name, interval="1H", days=args.days,
                                   candle_type=CandleType.MARKET)
                series[name] = returns([(c.ts, float(c.close)) for c in bars])
            _stamps, columns = align(series)
            benchmark = columns.pop("QQQUSDT")
            after = dict(book)
            after[args.symbol] = after.get(args.symbol, 0.0) + args.size
            total = sum(after.values()) or 1.0
            after = {k: v / total for k, v in after.items()}
            impact = assess(
                symbol=args.symbol, weights_before=book, weights_after=after,
                columns=columns, benchmark=benchmark, session=Session.OPEN,
            )

            # The two Open Theme items `desk/portfolio.py` could not supply: a correlation-aware
            # bet count, and a hedge named and sized rather than merely declared possible. Built
            # from the same aligned columns, so they cannot describe a different alignment.
            from argus.desk.diversification import effective_bets, suggest_hedges
            from argus.desk.portfolio import covariance_matrix

            built = covariance_matrix({k: v for k, v in columns.items() if after.get(k)})
            if built is not None:
                cov_names, cov = built
                diversification = effective_bets(after, cov_names, cov)
                # The book's own return path, then every name not in it as a hedge candidate.
                length = min(len(v) for v in columns.values())
                book_series = [
                    sum(after.get(n, 0.0) * columns[n][t] for n in cov_names)
                    for t in range(length)
                ]
                candidates = {
                    n: list(columns[n][:length]) for n in columns if not after.get(n)
                }
                candidates["QQQUSDT"] = list(benchmark[:length])
                hedges = suggest_hedges(book_series, candidates)

            # The allocation the book *could* hold, priced against the turnover to reach it. Uses
            # the same aligned columns as everything above, so the covariance it clusters on is the
            # covariance the other steps report.
            from argus.desk.allocation import optimize_trade

            attempted_allocation = True
            allocation = optimize_trade({k: v for k, v in after.items() if v}, columns)
        except Exception:
            impact = None
            allocation = None

    # The expectation gap, actually run. Until 2026-09-13 this step reported "no consensus
    # estimate was available for this name" without a lookup, and then — worse — reported "no
    # consensus source is wired for rTokens" after the absence kinds were separated. Both were
    # wrong: `market/estimates.py` fetches Yahoo consensus behind a cookie-and-crumb handshake and
    # `market/fundamentals.py` reads what was actually filed, and the module docstring of the
    # former records that it was built because ledger seq 41 asked for exactly this. Declaring a
    # capability absent without checking our own module list is the error this closes.
    #
    # The rToken is a claim on an underlying equity, so the consensus is looked up under the
    # underlying's ticker: NVDAUSDT is priced against NVDA, and NVDAUSDT has no analysts.
    attempted_gap = False
    underlying = args.symbol.removesuffix("USDT")
    try:
        from argus.desk.expectation import detect as detect_gap
        from argus.market.estimates import EstimatesSource
        from argus.market.fundamentals import FundamentalsSource

        consensus = EstimatesSource().fetch(underlying, as_of=now)
        reported, _notes = FundamentalsSource().facts(
            underlying, concept="eps_diluted", as_of=now
        )
        if consensus and reported:
            attempted_gap = True
            # **The revision counts were being dropped here.** `detect` takes
            # `revision_direction`, `revisions_up_30d` and `revisions_down_30d`, all defaulting to
            # "mixed"/0/0, and this call passed none of them — so the chain printed "estimates
            # revised mixed: 0 up / 0 down in 30 days" while the fetched consensus carried 35 up
            # and 1 down. The data was there, gathered, and thrown away one line before it was
            # needed; the default made the loss look like a finding rather than an omission.
            revisions = getattr(consensus[0], "revisions", None)
            gap = detect_gap(
                ticker=underlying, reported=reported, consensus=consensus,
                revision_direction=getattr(revisions, "direction", "mixed"),
                revisions_up_30d=int(getattr(revisions, "up_30d", 0) or 0),
                revisions_down_30d=int(getattr(revisions, "down_30d", 0) or 0),
            )
        else:
            # A lookup that ran and came back empty is a different statement from one that never
            # ran, and the step now distinguishes them.
            attempted_gap = True
    except Exception:
        gap = None

    # The analogue search, actually run. Until 2026-09-13 this was left at None and the report
    # said "no comparable historical state was found within the distance threshold" — a sentence
    # asserting a search that had never happened. The corpus is built from the same hourly history
    # the rest of the chain uses: each bar becomes a state described by its trailing returns and
    # realised volatility, paired with the move that followed it. The forward return is never a
    # feature; it is the answer.
    attempted_analogue = False
    attempted_shape = False
    try:
        from argus.desk.analogue import Observation
        from argus.desk.analogue import find as find_analogue
        from argus.desk.shapematch import find as find_shape
        from argus.market.history import CandleType, fetch_range

        bars = fetch_range(args.symbol, interval="1H", days=args.days,
                           candle_type=CandleType.MARKET)
        closes = [(c.ts, float(c.close)) for c in bars]
        corpus: list[Observation] = []
        window, horizon = 24, 24
        for i in range(window, len(closes) - horizon):
            past = [closes[j][1] for j in range(i - window, i + 1)]
            rets = [b / a - 1.0 for a, b in pairwise(past)]
            mean = sum(rets) / len(rets)
            vol = (sum((r - mean) ** 2 for r in rets) / len(rets)) ** 0.5
            # Not `price`: that name already holds a Decimal from the stress block above, and
            # reusing it made mypy infer this whole arithmetic as Decimal-over-float. The second
            # name collision of this kind in the project; both were caught by the type checker
            # rather than at runtime, which is the argument for running it in strict mode.
            close_now = closes[i][1]
            if close_now <= 0:
                continue
            corpus.append(Observation(
                as_of=closes[i][0], symbol=args.symbol,
                features={
                    "trailing_return": (close_now / past[0] - 1.0) * 10_000,
                    "volatility_bps": vol * 10_000,
                },
                forward_return_bps=(closes[i + horizon][1] / close_now - 1.0) * 10_000,
            ))
        if len(corpus) > horizon:
            attempted_analogue = True
            # The query is the most recent reconstructible state, and every observation in the
            # corpus is strictly older than it — the same point-in-time rule the rest of the desk
            # uses, enforced here by `find`'s own as_of gate rather than by trimming the list.
            latest = corpus[-1]
            analogue = find_analogue(
                query=dict(latest.features), corpus=corpus[:-1], as_of=latest.as_of
            )

        # The second retrieval, over the same bars: the state vector above asks what *kind* of
        # market this is, this asks whether the path itself has been traced before. They are run
        # together because they share a fetch, and reported apart because they can disagree.
        if len(closes) > window + horizon * 2:
            attempted_shape = True
            shape = find_shape(closes, symbol=args.symbol, window=window, horizon=horizon)
    except Exception:
        analogue = None
        shape = None

    # The pairwise search. Run over the other rTokens rather than the whole venue: these twelve are
    # the instruments this desk can actually trade, so a cointegrated partner outside that list
    # would be a relationship we could observe and never act on.
    attempted_pairs = False
    try:
        from argus.market.bitget import RTOKEN_SYMBOLS
        from argus.market.history import CandleType, fetch_range
        from argus.research.cointegration import MIN_OBSERVATIONS, narrow, scan

        universe: dict[str, list[float]] = {}
        for name in RTOKEN_SYMBOLS:
            try:
                rows = fetch_range(name, interval="1H", days=args.days,
                                   candle_type=CandleType.MARKET)
            except Exception:
                continue
            if len(rows) >= MIN_OBSERVATIONS * 2:
                universe[name] = [float(c.close) for c in rows]
        # Only pairs involving the instrument under question: testing all 66 and reporting the one
        # that happens to involve this symbol would be selecting on the answer.
        if args.symbol in universe and len(universe) >= 2:
            attempted_pairs = True
            pairs = narrow(
                scan(universe, lookback=min(240, len(universe[args.symbol]) // 3)), args.symbol,
            )
    except Exception:
        pairs = None

    report = research(
        question=question, symbol=args.symbol, evidence=evidence, gap=gap,
        analogue=analogue, shape=shape, pairs=pairs, allocation=allocation, impact=impact,
        stress=stress,
        cost_bps=cost_bps, now=now,
        attempted_gap=attempted_gap, attempted_analogue=attempted_analogue,
        attempted_shape=attempted_shape, attempted_pairs=attempted_pairs,
        attempted_allocation=attempted_allocation,
        diversification=diversification, hedges=hedges,
    )
    print(report.render())
    if args.save:
        from pathlib import Path

        Path(args.save).write_text(
            json.dumps(report.as_dict(), indent=2, default=str), encoding="utf-8"
        )
        print(f"\nsaved -> {args.save}")
    return 0


__all__ = [
    "MIN_STEPS_FOR_A_VERDICT",
    "Finding",
    "Report",
    "Verdict",
    "allocation_step",
    "analogue_step",
    "beta_step",
    "cointegration_step",
    "decide",
    "evidence_step",
    "execution_step",
    "expectation_step",
    "main",
    "parse_book",
    "portfolio_step",
    "research",
    "shape_step",
    "stress_step",
]


if __name__ == "__main__":
    raise SystemExit(main())
