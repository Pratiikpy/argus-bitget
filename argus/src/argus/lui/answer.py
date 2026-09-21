"""Answering — every reply resolves to a ledger row, a computation, or a refusal.

**The rule this module exists to enforce.** An answer is only allowed to contain a fact that came
from the record. Not "the desk was cautious because markets were quiet" — *"seq 25, NVDAUSDT,
2026-09-12T15:24Z, no_trade, confidence 0.92, thesis: …, entry hash 9ec58de4"*. Every
:class:`Answer` carries its sources, and an answer with no sources is a bug, not a style choice.

That is a deliberate inversion of how the nine systems in ``research/subthemes/_CENSUS-lui.md``
work. They generate prose and then attach citations to it where they can. Here the citation comes
first: the answerer reads rows, and the sentence is constructed from what the rows say. A fact that
cannot be traced cannot be said, which is why :class:`Answer` has no free-text field a model could
fill in unsupervised.

**Refusal is an answer.** "I cannot tell you that, and here is the specific reason" is a correct
outcome and is rendered as one. The judge questions this is built against include several that are
*supposed* to be refused — an instrument the venue does not carry, a reference with nothing to bind
to, a statistic the sample cannot support. Answering those anyway is the failure being tested for.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from argus.eval.performance import evaluate_ledger
from argus.lui.phrasebook import t
from argus.lui.question import TRADED_SYMBOLS, Intent, Question
from argus.paper.ledger import Entry, PaperLedger


@dataclass(frozen=True)
class Source:
    """Where a fact came from. Enough for a reader to go and check it."""

    kind: str
    """``ledger`` | ``computation`` | ``evidence`` | ``venue``."""

    ref: str
    """A ledger seq, an evidence id, a module path, or a URL."""

    detail: str = ""

    def as_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "ref": self.ref, "detail": self.detail}

    def __str__(self) -> str:
        return f"{self.kind}:{self.ref}" + (f" ({self.detail})" if self.detail else "")


@dataclass
class Answer:
    """One reply. ``lines`` is what a person reads; ``sources`` is how they check it."""

    question: Question
    lines: list[str] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)
    refused: bool = False
    reason: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def is_grounded(self) -> bool:
        """An answer that asserts something must say where it came from."""
        return self.refused or bool(self.sources)

    def as_dict(self) -> dict[str, Any]:
        return {
            "intent": str(self.question.intent),
            "speed": str(self.question.speed),
            "refused": self.refused,
            "reason": self.reason,
            "lines": list(self.lines),
            "sources": [s.as_dict() for s in self.sources],
            "data": self.data,
        }


def _refuse(question: Question, reason: str, *, suggestion: str = "") -> Answer:
    lines = [reason]
    if suggestion:
        lines.append(suggestion)
    return Answer(question=question, lines=lines, refused=True, reason=reason)


def _in_window(entry: Entry, question: Question) -> bool:
    if question.window is None:
        return True
    return question.window.contains(datetime.fromisoformat(entry.decided_at))


def _matching(ledger: PaperLedger, question: Question) -> list[Entry]:
    rows = [e for e in ledger.entries if _in_window(e, question)]
    if question.symbols:
        rows = [e for e in rows if e.symbol in question.symbols]
    if question.seq is not None:
        rows = [e for e in rows if e.seq == question.seq]
    return rows


def _src(entry: Entry) -> Source:
    return Source(
        kind="ledger",
        ref=f"seq {entry.seq}",
        detail=f"{entry.symbol} {entry.verdict} @ {entry.decided_at}",
    )


# --- answerers ------------------------------------------------------------------------------


def answer_performance(ledger: PaperLedger, question: Question) -> Answer:
    """The three scored numbers, or a named reason why they do not exist yet."""
    perf = evaluate_ledger(ledger)
    lines: list[str] = []
    sources = [
        Source("computation", "argus.eval.performance:evaluate_ledger",
               f"over {perf.window_days} day(s), {perf.trades} settled trade(s)")
    ]

    lang = question.language
    # max_drawdown joins the guard rather than being asserted non-None below it. It is undefined
    # on exactly the same condition as win_rate (no settled trades), and stating that here means the
    # console can never render a 0.0% drawdown for an account that never took a position — the
    # best-looking risk number on the page, earned by not participating.
    if perf.sharpe is None or perf.win_rate is None or perf.max_drawdown is None:
        for stat, why in perf.undefined.items():
            lines.append(t("perf.undefined_stat", lang, stat=stat, why=why))
        lines.append(
            t("perf.no_trades", lang, abstentions=perf.abstentions, trades=perf.trades,
              days=perf.window_days)
        )
    else:
        lines.append(
            t("perf.headline", lang, sharpe=perf.sharpe, drawdown=100 * perf.max_drawdown,
              win_rate=100 * perf.win_rate, trades=perf.trades, days=perf.window_days)
        )
        if perf.by_symbol:
            top = perf.by_symbol[0]
            lines.append(
                t("perf.largest", lang, symbol=top.symbol,
                  share=perf.as_dict()["largest_symbol_share_pct"], trades=top.trades)
            )
    lines.append(t("perf.net_pnl", lang, net=perf.net_pnl, capital=perf.capital))
    for entry in [e for e in ledger.entries if e.is_settled and not e.is_abstention][:3]:
        sources.append(_src(entry))
    return Answer(question=question, lines=lines, sources=sources, data=perf.as_dict())


def answer_decision_why(ledger: PaperLedger, question: Question) -> Answer:
    """Reconstruct one decision from the row that recorded it."""
    rows = _matching(ledger, question)
    if not rows:
        return _refuse(
            question,
            "No decision in the record matches that.",
            suggestion=(
                f"The log holds {len(ledger.entries)} decision(s)"
                + (f" on {', '.join(sorted({e.symbol for e in ledger.entries}))}"
                   if ledger.entries else "")
                + "."
            ),
        )

    entry = rows[-1]
    lang = question.language
    lines = [
        t("why.header", lang, seq=entry.seq, symbol=entry.symbol, verdict=entry.verdict,
          at=entry.decided_at, phase=entry.session_phase),
        t("why.confidence", lang, confidence=entry.stated_confidence,
          hours=entry.hours_to_discovery),
        t("why.thesis", lang, thesis=entry.thesis),
    ]
    if entry.invalidation:
        lines.append(t("why.invalidation", lang, conditions="; ".join(entry.invalidation)))
    if entry.is_settled:
        direction = t(
            "why.direction_correct" if entry.direction_correct else "why.direction_wrong", lang
        )
        lines.append(
            t("why.settled", lang, at=entry.settled_at, net=entry.net_pnl, direction=direction)
        )
    else:
        lines.append(t("why.unsettled", lang))
    lines.append(
        t("why.hash", lang, hash=entry.content_hash[:16], prev=entry.prev_hash[:8])
    )

    if len(rows) > 1:
        lines.append(t("why.multiple", lang, count=len(rows)))
    return Answer(question=question, lines=lines, sources=[_src(e) for e in rows[-3:]],
                  data={"seq": entry.seq, "verdict": entry.verdict})


def answer_abstention_why(ledger: PaperLedger, question: Question) -> Answer:
    """Why the desk stood aside — the most common correct answer on this venue."""
    rows = [e for e in _matching(ledger, question) if e.is_abstention]
    if not rows:
        window = question.window.label if question.window else "the record"
        return _refuse(question, f"No abstentions in {window}.")

    phases = sorted({e.session_phase for e in rows})
    symbols = sorted({e.symbol for e in rows})
    lang = question.language
    lines = [
        t("abst.header", lang,
          count=len(rows),
          window=(f" in {question.window.label}" if question.window else ""),
          symbols=len(symbols), phases="/".join(phases)),
        t("abst.explain", lang),
    ]
    for entry in rows[-3:]:
        lines.append(t("abst.row", lang, seq=entry.seq, symbol=entry.symbol, thesis=entry.thesis))
    settled = [e for e in rows if e.counterfactual_move_bps is not None]
    lines.append(
        t("abst.settled", lang, count=len(settled)) if settled else t("abst.unsettled", lang)
    )
    return Answer(question=question, lines=lines, sources=[_src(e) for e in rows[-3:]],
                  data={"abstentions": len(rows), "symbols": symbols})


def answer_decision_list(ledger: PaperLedger, question: Question) -> Answer:
    rows = _matching(ledger, question)
    if not rows:
        return _refuse(question, "No decisions match that window or symbol.")
    # **A voided row records something the desk did not do, and counting it as a trade made the
    # console contradict every other surface.** Two rows on the live record booked positions the
    # Constitution had explicitly refused (`paper/corrections.py`), and every consumer of the
    # ledger excludes them — `ledger.performance`, `eval/performance`, the cockpit, the scorecard.
    # This one did not: it counted `entry.verdict` raw, so the console answered *"493 no_trade,
    # 2 trade"* while the submission, the README and the cockpit all said zero trades.
    #
    # Both statements were built from the same chain, which is exactly how a tamper-evident log
    # launders a mistake: the chain verifies, so the wrong number looks authenticated. The rows
    # still appear in the listing below — deleting them is what `corrections.py` refuses — but
    # they are counted as what they were, not as what they claimed.
    verdicts: dict[str, int] = {}
    voided = 0
    for entry in rows:
        if entry.is_void:
            voided += 1
            continue
        verdicts[entry.verdict] = verdicts.get(entry.verdict, 0) + 1
    lang = question.language
    lines = [
        t("list.header", lang, count=len(rows),
          window=(f" in {question.window.label}" if question.window else ""),
          breakdown=", ".join(f"{n} {v}" for v, n in sorted(verdicts.items()))),
    ]
    if voided:
        lines.append(t("list.voided", lang, voided=voided))
    for entry in rows[-5:]:
        lines.append(
            t("list.row", lang, seq=entry.seq, symbol=entry.symbol, verdict=entry.verdict,
              confidence=entry.stated_confidence, at=entry.decided_at)
        )
    return Answer(question=question, lines=lines, sources=[_src(e) for e in rows[-5:]],
                  data={"count": len(rows), "verdicts": verdicts, "voided": voided})


def answer_integrity(ledger: PaperLedger, question: Question) -> Answer:
    """Whether the record can be trusted — verified, not asserted."""
    report = ledger.verify()
    intact = bool(report["chain_intact"])
    lang = question.language
    state = t("integ.state_intact" if intact else "integ.state_broken", lang)
    lines = [t("integ.header", lang, state=state, count=len(ledger.entries))]
    if not intact:
        lines.append(t("integ.break_at", lang, at=report.get("first_break_at")))
    else:
        lines.append(t("integ.explain", lang))
    if ledger.entries:
        lines.append(t("integ.head", lang, hash=ledger.entries[-1].content_hash[:16]))
    return Answer(question=question, lines=lines,
                  sources=[Source("computation", "argus.paper.ledger:PaperLedger.verify")],
                  data={k: str(v) for k, v in report.items()})


def answer_position(ledger: PaperLedger, question: Question) -> Answer:
    open_rows = [e for e in ledger.entries if not e.is_settled and not e.is_abstention]
    if not open_rows:
        total = len(ledger.entries)
        return Answer(
            question=question,
            lines=[
                t("pos.none", question.language),
                t("pos.none_detail", question.language, total=total,
                  abstentions=sum(1 for e in ledger.entries if e.is_abstention)),
            ],
            sources=[Source("ledger", "all rows", f"{total} entries scanned")],
            data={"open": 0, "entries": total},
        )
    lang = question.language
    lines = [t("pos.header", lang, count=len(open_rows))]
    for entry in open_rows:
        lines.append(
            t("pos.row", lang, seq=entry.seq, symbol=entry.symbol, side=entry.side,
              quantity=entry.quantity, price=entry.entry_price)
        )
    return Answer(question=question, lines=lines, sources=[_src(e) for e in open_rows],
                  data={"open": len(open_rows)})


def answer_calibration(ledger: PaperLedger, question: Question) -> Answer:
    """Calibration needs a run of graded outcomes; below the floor it is noise."""
    from argus.eval.scorecard import scorecard

    card = scorecard(ledger)
    graded = int(card.get("graded_predictions", 0))
    sources = [Source("computation", "argus.eval.scorecard:scorecard",
                      f"{graded} graded prediction(s)")]
    if "ece" not in card:
        return Answer(
            question=question,
            lines=[
                t("calib.insufficient", question.language, graded=graded, floor=5),
                t("calib.insufficient_why", question.language),
            ],
            sources=sources,
            data={"graded_predictions": graded},
        )
    return Answer(
        question=question,
        lines=[
            t("calib.headline", question.language, ece=card["ece"], brier=card["brier"],
              accuracy=card.get("accuracy_pct"), graded=graded),
            t("calib.deterministic", question.language),
        ],
        sources=sources,
        data={k: card[k] for k in ("ece", "brier") if k in card},
    )


def answer_session(ledger: PaperLedger, question: Question) -> Answer:
    """Session state, read from the most recent decision rather than recomputed."""
    if not ledger.entries:
        return _refuse(question, "No decisions recorded, so no session state to report.")
    latest = ledger.entries[-1]
    lang = question.language
    lines = [
        t("sess.header", lang, at=latest.decided_at, phase=latest.session_phase,
          hours=latest.hours_to_discovery),
        t("sess.explain", lang),
    ]
    return Answer(question=question, lines=lines, sources=[_src(latest)],
                  data={"phase": latest.session_phase,
                        "hours_to_discovery": latest.hours_to_discovery})


def answer_evidence(ledger: PaperLedger, question: Question) -> Answer:
    """What the desk could see. Reads the decision's own record, never re-fetches.

    Re-fetching would answer a different question: what is visible *now*, not what was visible when
    the decision was taken. Those diverge the moment a headline lands, and conflating them is how a
    post-hoc justification gets mistaken for a contemporaneous reason.
    """
    rows = _matching(ledger, question)
    if not rows:
        return _refuse(question, "No decision matches, so there is no evidence set to show.")
    entry = rows[-1]
    lang = question.language
    lines = [
        t("ev.header", lang, seq=entry.seq, symbol=entry.symbol, at=entry.decided_at),
        t("ev.thesis", lang, thesis=entry.thesis),
        t("ev.hash", lang, hash=entry.market_state_hash[:16]),
        t("ev.feed", lang),
    ]
    return Answer(
        question=question, lines=lines,
        sources=[_src(entry),
                 Source("computation", "argus.market.evidence:gather", "as-of gated")],
        data={"seq": entry.seq, "market_state_hash": entry.market_state_hash},
    )


def _risk_records_path() -> Path:
    """Where the risk records live, honouring the same override the ledger uses.

    `argus.eval.riskaudit` derives its default from the source tree's layout and takes the path as
    an argument, so the console has to supply one. Deriving it here rather than importing a
    constant keeps the hosted bundle working: the deployment copies the package away from its
    checkout, and a path computed from `__file__` would point at a directory that does not exist.
    """
    override = os.environ.get("ARGUS_DATA_DIR", "").strip()
    if override:
        return Path(override) / "risk_records.jsonl"

    from argus.paper.runner import RISK_PATH

    return RISK_PATH


def answer_risk_control(ledger: PaperLedger, question: Question) -> Answer:
    """What the risk layer actually did — counted from its own records, never asserted.

    Track 2 scores "risk control layer effectiveness" directly, and the honest answer here is
    usually uncomfortable: on a desk that abstains most of the time, the Constitution is handed
    almost nothing to bind on. Reporting an intervention rate against every decision would make an
    untested layer look restrained, so the denominator is decisions that actually offered a
    position to reduce, and the answer says which denominator it used.
    """
    from argus.eval.riskaudit import audit as risk_audit

    path = _risk_records_path()
    if not path.exists():
        return _refuse(
            question,
            "No risk records are bundled with this console, so there is nothing to count.",
            suggestion="Run a decision cycle; every decision writes one risk record.",
        )
    report = risk_audit(path, settled_trades=sum(1 for e in ledger.entries if e.is_settled))
    lang = question.language
    lines = [
        t("risk.header", lang, decisions=report.decisions, offered=report.positions_offered),
    ]
    if not report.positions_offered:
        lines.append(t("risk.untested", lang))
    else:
        rate = report.rate
        lines.append(
            f"It intervened {len(report.interventions)} time(s) — "
            f"{'unavailable' if rate is None else format(rate, '.1%')} of the decisions that "
            f"offered exposure. Exercise: {report.exercise}."
        )
        by_constraint = report.by_constraint()
        if by_constraint:
            lines.append(
                "Binding constraints: "
                + ", ".join(f"{k} x{v}" for k, v in by_constraint.items())
                + "."
            )
        lines.append(
            f"The model rewrote its own intent after {report.responded} of those interventions "
            f"rather than repeating it."
        )
    violations = report.asymmetry_violations
    lines.append(
        f"Asymmetry: {len(violations)} case(s) where the layer increased exposure. "
        + (
            "The design forbids it — the layer may only reduce, never create, enlarge or reverse "
            "— and this is the check, not the claim."
            if not violations else
            "This is a design violation and is reported rather than suppressed."
        )
    )
    return Answer(
        question=question,
        lines=lines,
        sources=[
            Source("computation", "argus.eval.riskaudit:audit"),
            Source("artefact", path.name, f"{report.decisions} record(s)"),
        ],
        data=report.as_dict(),
    )


_ANSWERERS = {
    Intent.PERFORMANCE: answer_performance,
    Intent.DECISION_WHY: answer_decision_why,
    Intent.DECISION_LIST: answer_decision_list,
    Intent.ABSTENTION_WHY: answer_abstention_why,
    Intent.EVIDENCE: answer_evidence,
    Intent.CALIBRATION: answer_calibration,
    Intent.INTEGRITY: answer_integrity,
    Intent.POSITION: answer_position,
    Intent.SESSION: answer_session,
    Intent.RISK_CONTROL: answer_risk_control,
}


def answer(ledger: PaperLedger, question: Question) -> Answer:
    """Route a classified question to its answerer, or refuse by name."""
    if question.intent is Intent.UNSUPPORTED:
        return _refuse(
            question, question.reason,
            suggestion="ARGUS decides on: " + ", ".join(TRADED_SYMBOLS) + ".",
        )
    if question.intent is Intent.AMBIGUOUS:
        return _refuse(question, question.reason,
                       suggestion="Name a symbol or a decision number.")
    if question.intent is Intent.ORDER:
        return _refuse(
            question, question.reason,
            suggestion=(
                "Orders are placed by the desk itself, through the risk layer, and recorded in "
                "the ledger. Ask why a decision was taken, or what the desk is holding."
            ),
        )
    if question.intent is Intent.MARKET:
        return _refuse(
            question,
            "A live quote is not part of the record, so answering it here would mix a fact from "
            "now into a report about then.",
            suggestion="Ask about a decision, its evidence, or the performance of the log.",
        )
    handler = _ANSWERERS.get(question.intent)
    if handler is None:
        return _refuse(
            question,
            "I did not recognise that question well enough to answer it from the record.",
            suggestion=(
                "Answerable today: performance, why a decision was taken, why the desk stood "
                "aside, the evidence behind a decision, calibration, chain integrity, open "
                "positions, session state, what the risk layer did."
            ),
        )
    return handler(ledger, question)


__all__ = ["Answer", "Source", "answer"]
