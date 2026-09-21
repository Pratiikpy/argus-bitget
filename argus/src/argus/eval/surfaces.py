"""Do our surfaces agree with each other? — the check `eval/docclaims.py` cannot make.

`docclaims` compares every figure in the documents against the artefact that produced it, and it
works: 95 claims, 0 stale. It is also, structurally, unable to catch the defect that motivated
this module.

**On 2026-09-21 the console answered "493 no_trade, 2 trade" while the README, the submission and
the cockpit all said zero trades.** Every surface was individually defensible. The console was
reading the ledger correctly — two rows really do carry ``verdict: trade``. The documents were
reading `eval/performance` correctly — it excludes them, because those two rows booked positions
the Constitution had refused (`paper/corrections.py`). Each surface passed its own check against
the data, and they contradicted each other anyway.

**That is the shape of the problem.** A per-surface check asks *"does this number match its
source?"*. It never asks *"do two surfaces that describe the same fact say the same thing?"* — and
a reader comparing the console to the README is asking exactly that. Worse, both numbers came from
the same hash chain, so the contradiction arrived wearing a verification: the chain proves a record
was not *altered*, never that two readings of it agree.

So this module asks the second question. It drives the **console's own answerer** — not a
reimplementation of it — and compares what a visitor is told against what the artefacts say, one
named fact at a time. A disagreement is a failure with both readings printed, because reporting
which one is *right* would require this module to be a third opinion, and a third opinion is how
you get three surfaces disagreeing instead of two.

    python -m argus.eval.surfaces
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from argus.eval.artefact import write

DATA = Path(__file__).resolve().parents[3] / "data"
REPORT_PATH = DATA / "surface_agreement.json"

AT = datetime(2026, 9, 21, 12, 0, tzinfo=UTC)
"""A fixed clock, so a window-sensitive question does not change answer overnight."""


class SurfaceError(RuntimeError):
    """The comparison cannot be made. Raised rather than reported as agreement."""


@dataclass(frozen=True, slots=True)
class Agreement:
    """One fact, read off two surfaces."""

    fact: str
    console: Any
    artefact: Any
    console_says: str
    artefact_says: str
    why_it_matters: str

    @property
    def agrees(self) -> bool:
        return bool(self.console == self.artefact)

    def as_dict(self) -> dict[str, Any]:
        return {
            "fact": self.fact,
            "console": self.console,
            "artefact": self.artefact,
            "console_source": self.console_says,
            "artefact_source": self.artefact_says,
            "agrees": self.agrees,
            "why_it_matters": self.why_it_matters,
        }

    def render(self) -> str:
        mark = "ok  " if self.agrees else "DISAGREE"
        body = (
            f"{self.console}" if self.agrees
            else f"console says {self.console!r}, artefact says {self.artefact!r}"
        )
        return f"  {mark}  {self.fact:34} {body}"


def _ask(question: str) -> Any:
    """One question, through the console's real answerer.

    Deliberately the production path — `classify` then `reclassify` then `answer` — rather than a
    call into the answer function with a hand-built `Question`. A check that bypasses the
    classifier cannot catch a disagreement the classifier causes.
    """
    from argus.lui.answer import answer
    from argus.lui.ngram import reclassify
    from argus.lui.question import classify
    from argus.paper.ledger import PaperLedger
    from argus.paper.runner import LEDGER_PATH

    asked = classify(question, now=AT)
    asked, _source = reclassify(asked)
    return answer(PaperLedger(path=LEDGER_PATH), asked)


def _checks() -> list[Agreement]:
    """Every fact that appears on more than one surface, read off both."""
    from argus.eval.performance import evaluate_ledger
    from argus.paper.ledger import PaperLedger
    from argus.paper.runner import LEDGER_PATH

    ledger = PaperLedger(path=LEDGER_PATH)
    perf = evaluate_ledger(ledger)
    out: list[Agreement] = []

    # 1. Trades. **The one that was wrong.** The console counted two refused positions as trades
    #    while every other consumer excluded them.
    listed = _ask("show me every decision")
    console_trades = int(listed.data.get("verdicts", {}).get("trade", 0))
    out.append(Agreement(
        fact="settled trades",
        console=console_trades, artefact=int(perf.trades),
        console_says="lui.answer:answer_decision_list verdict breakdown",
        artefact_says="eval.performance:evaluate_ledger.trades",
        why_it_matters=(
            "the submission, the README and the cockpit all state zero trades; a console saying "
            "otherwise makes our own record contradict our own claim on the surface built to "
            "show the record can be trusted"
        ),
    ))

    # 2. Decision count — the number a visitor reads first, on both the page header and the list.
    out.append(Agreement(
        fact="decisions on record",
        console=int(listed.data.get("count", -1)), artefact=len(ledger.entries),
        console_says="lui.answer:answer_decision_list count",
        artefact_says="paper.ledger entries",
        why_it_matters=(
            "a stale snapshot verifies its chain perfectly, so a wrong count here looks healthy"
        ),
    ))

    # 3. Whether the headline statistics exist at all. A console that reports a Sharpe the
    #    performance module refuses to compute is the more dangerous direction of this error.
    perf_answer = _ask("what is the sharpe")
    console_has_sharpe = perf_answer.data.get("sharpe") is not None
    out.append(Agreement(
        fact="sharpe is computable",
        console=console_has_sharpe, artefact=perf.sharpe is not None,
        console_says="lui.answer:answer_performance",
        artefact_says="eval.performance:evaluate_ledger.sharpe",
        why_it_matters=(
            "Track 2 is scored on Sharpe; a console that shows one where the maths refuses is the "
            "worst available failure and would be found by the first judge who asked twice"
        ),
    ))

    # 4. Abstentions, which is the number the whole thesis rests on.
    out.append(Agreement(
        fact="abstentions",
        console=int(listed.data.get("verdicts", {}).get("no_trade", -1)),
        artefact=int(perf.abstentions) - int(listed.data.get("voided", 0)),
        console_says="lui.answer:answer_decision_list verdict breakdown",
        artefact_says="eval.performance abstentions, less the voided rows",
        why_it_matters=(
            "'refused 495 times out of 495' is the headline claim; the console must be able to "
            "show the same arithmetic"
        ),
    ))

    # 5. The count as the PAGE HEADER reports it, against the count the LIST reports.
    #
    # **Not chain integrity.** The obvious fifth check was the console's tamper-evidence answer
    # against `ledger.verify()` — and `answer_integrity` returns that verify() result verbatim, so
    # the comparison is a value against itself. A tautology that always passes is worse than no
    # check: it adds a green line to a report whose whole purpose is to go red.
    #
    # `lui.server._status` and `answer_decision_list` genuinely reach the ledger by different
    # paths, and the header is the first number a visitor reads. On 2026-09-15 the hosted page
    # rendered 126 decisions against a live 219, and the chain of that stale snapshot verified
    # perfectly.
    from argus.lui.server import _status

    status = _status()
    out.append(Agreement(
        fact="count: page header vs list",
        console=int(status.get("entries", -1)),
        artefact=int(listed.data.get("count", -2)),
        console_says="lui.server:_status (the page header)",
        artefact_says="lui.answer:answer_decision_list (the list itself)",
        why_it_matters=(
            "the header is the first number read and the list is the one checked; a stale "
            "snapshot verifies its chain perfectly, so disagreement here is the only signal"
        ),
    ))
    return out


def run() -> dict[str, Any]:
    checks = _checks()
    disagreements = [c for c in checks if not c.agrees]
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "facts_checked": len(checks),
        "agree": len(checks) - len(disagreements),
        "disagree": len(disagreements),
        "clean": not disagreements,
        "checks": [c.as_dict() for c in checks],
        "disagreements": [c.as_dict() for c in disagreements],
        "scope_statement": (
            "Each fact is read off the CONSOLE's production answer path (classify -> reclassify "
            "-> answer) and off the artefact the documents quote, and the two are compared. NOT "
            "CLAIMED: that agreement means either is correct — two surfaces can agree and both be "
            "wrong, which is what `eval/docclaims.py` checks separately by comparing each against "
            "its source. NOT CLAIMED: that this covers every figure; it covers the ones that "
            "appear on more than one surface, which is where a contradiction can hide."
        ),
    }


def render(report: dict[str, Any]) -> list[str]:
    lines = [
        f"SURFACE AGREEMENT — {report['facts_checked']} fact(s) read off two surfaces each",
        f"  {report['agree']} agree, {report['disagree']} disagree",
        "",
    ]
    for row in report["checks"]:
        mark = "ok      " if row["agrees"] else "DISAGREE"
        body = (
            str(row["console"]) if row["agrees"]
            else f"console {row['console']!r} vs artefact {row['artefact']!r}"
        )
        lines.append(f"  {mark}  {row['fact']:28} {body}")
    if report["disagree"]:
        lines.append("")
        for row in report["disagreements"]:
            lines.append(f"  {row['fact']}: {row['why_it_matters']}")
            lines.append(f"    console  <- {row['console_source']}")
            lines.append(f"    artefact <- {row['artefact_source']}")
    lines.append(
        "  Both readings are printed and neither is called correct. Deciding which one is right "
        "would make this a third opinion, and a third opinion is how you get three surfaces "
        "disagreeing instead of two."
    )
    return lines


def main() -> int:  # pragma: no cover - CLI
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    report = run()
    for line in render(report):
        print(line)
    write(REPORT_PATH, report)
    print(f"\nwritten to {REPORT_PATH}")
    return 0 if report["clean"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["REPORT_PATH", "Agreement", "SurfaceError", "main", "render", "run"]
