"""One decision, on one page — the whole trail from evidence to outcome.

Track 2 scores "decision explainability". Everything needed to explain a decision here already
exists and is scattered across four files written by four different parts of the system: the
ledger holds the decision and its hashes, ``desk_notes.jsonl`` holds what the checkers found,
``risk_records.jsonl`` holds what the Constitution did, and ``protocol_commitments.jsonl`` holds
the rules that governed it. A reader who wants to understand decision 64 has to join them by hand.

That join is this module. A :class:`Card` is one decision with every part of its trail attached, in
the order a person actually asks about it: *what was decided, why, what was it checked against,
what did the risk layer do, what rules were in force, and what happened next.*

Two properties matter more than the formatting.

**Nothing on a card is computed here.** Every field is copied from the artefact that produced it,
and the card names which file each part came from. A page that recomputed a number would be a
second opinion wearing the ledger's authority, and where the two disagreed the reader would have no
way to tell which was the record.

**Absence is printed.** A decision with no risk record, no notes, or no outcome says so in those
words. The failure this avoids is the one every dashboard has: a missing section renders as empty
space, empty space reads as "nothing to report", and "nothing to report" is indistinguishable from
"nobody looked". An unsettled decision in particular must never look like a correct one.

    python -m argus.eval.decisioncard --seq 64
    python -m argus.eval.decisioncard --all --out data/cards
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ABSENT = "— not recorded —"
"""What is printed where a section has no data. Never an empty string: a blank looks like a
finding of nothing, and this is a finding of no data."""


def _lean_line(entry: dict[str, Any]) -> str:
    """The direction the desk would have taken, and how firmly.

    **A recorded ``none`` is ambiguous on this log, and the card says so rather than choosing.** The
    intended reading is three states — UP, DOWN, and a deliberate decline to call it — with rows
    written before the field existed as a fourth, printed as absent. That fourth state is not
    recoverable here: settling any entry rewrites the whole file through ``asdict``
    (`paper/ledger.py:_persist_settlement`), which stamps the dataclass default onto every row it
    rewrites. So the 161 decisions that predate the field now carry ``lean: "none"`` on disk,
    indistinguishable from a desk that looked and declined.

    Printing them as "declined to call the direction" would credit them with a judgement they never
    made; printing them as absent would deny a genuine decline. The honest line names both, and it
    disappears from new rows as the log grows.
    """
    if "lean" not in entry:
        return ABSENT + " (decided before the desk recorded a lean)"
    lean = str(entry.get("lean", "none")).lower()
    if lean not in {"up", "down"}:
        return (
            "none — the desk declined to call the direction, or this decision predates the field. "
            "A settled row cannot tell the two apart"
        )
    return f"{lean.upper()} at {float(entry.get('lean_confidence', 0.0)):.2f} confidence"


@dataclass(frozen=True, slots=True)
class Source:
    """Where one part of a card came from, so every line can be traced back."""

    section: str
    path: str

    def render(self) -> str:
        return f"{self.section}: {self.path}"


@dataclass
class Card:
    """One decision and its whole trail."""

    seq: int
    entry: dict[str, Any]
    notes: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    risk: dict[str, Any] | None = None
    protocol: dict[str, Any] | None = None
    sources: list[Source] = field(default_factory=list)

    @property
    def symbol(self) -> str:
        return str(self.entry.get("symbol", "?"))

    @property
    def is_abstention(self) -> bool:
        return str(self.entry.get("verdict", "")) in {"no_trade", "data_insufficient"}

    @property
    def is_settled(self) -> bool:
        return self.entry.get("settled_at") is not None

    @property
    def outcome_line(self) -> str:
        """What happened, or the precise reason nothing is known yet."""
        if not self.is_settled:
            return (
                "not settled yet — it is held for 24 hours and graded against a price fetched in "
                "a later cycle, never one available when it was decided"
            )
        if self.is_abstention:
            moved = self.entry.get("counterfactual_move_bps")
            if moved is None:
                return "settled as an abstention, but the counterfactual move was not recorded"
            return (
                f"stood aside; the price then moved {moved}bps over the horizon — which is what "
                f"makes a refusal gradeable rather than merely safe"
            )
        correct = self.entry.get("direction_correct")
        return (
            f"net {self.entry.get('net_pnl', ABSENT)} "
            f"(gross {self.entry.get('gross_pnl', ABSENT)}), direction "
            f"{'right' if correct else 'wrong' if correct is not None else 'ungraded'}"
        )

    def _void_banner(self) -> list[str]:
        """Say so, at the top, when this card describes a row that records a fill never taken.

        **This card is how the defect was found**, so it is the last place that should render the
        contradiction without naming it. Before 2026-09-20 seq 264 printed `verdict: trade`,
        `Size 1 SELL`, `no order: final verdict human_review with quantity 0` and `net 9.6521` on
        one page, and left the reader to notice. The fields below are still shown verbatim — the
        row is not edited and not hidden — but the banner states which of them are the defect.
        """
        from argus.paper.corrections import VOIDED

        void = next((v for v in VOIDED if v.seq == self.seq), None)
        if void is None:
            return []
        return [
            "> ⚠️ **VOID — this decision was never taken.** The fields below are reproduced "
            "verbatim from the ledger and are NOT corrected in place; the row stays in the hash "
            "chain unedited. What is wrong with them is stated here instead.",
            ">",
            f"> {void.reason}",
            ">",
            f"> Evidence: {void.evidence}",
            ">",
            "> The stored `verdict`, `quantity`, `entry_price` and every P&L figure on this card "
            "are artefacts of a `paper/runner.py` defect fixed on 2026-09-20 "
            "(`paper/runner.py::governed_intent`). The desk refused this position. Every derived "
            "figure excludes it — see `paper/corrections.py`.",
            "",
        ]

    def render(self) -> str:
        e = self.entry
        lines = [
            f"# Decision {self.seq} — {self.symbol} — {e.get('verdict', ABSENT)}",
            "",
            *self._void_banner(),
            f"**Decided** {e.get('decided_at', ABSENT)} during the "
            f"{e.get('session_phase', ABSENT)} session, "
            f"{e.get('hours_to_discovery', ABSENT)}h from the next price discovery.",
            f"**Size** {e.get('quantity', ABSENT)} {e.get('side', '')} at "
            f"{e.get('entry_price', ABSENT)}; entry cost {e.get('entry_cost_bps', ABSENT)}bps.",
            f"**Stated confidence** {e.get('stated_confidence', ABSENT)}",
            # The lean is the view the desk held while refusing to act, and on an abstention it is
            # the only part of the card that can ever be graded. `side` sits above it and is not a
            # substitute: measured over 53 settled rows it was BUY in all 53, right 3.8% of the
            # time against a 3.8% base rate. A card that showed only the side would be showing the
            # field with no information in it.
            f"**Lean** {_lean_line(e)}",
            "",
            "## Why",
            str(e.get("thesis") or ABSENT),
            "",
            "## What would make this wrong",
        ]
        invalidation = e.get("invalidation") or []
        lines.extend(
            [f"- {c}" for c in invalidation] if invalidation
            else [
                ABSENT + " (an abstention opens no exposure, so it needs no falsifier)"
                if self.is_abstention else ABSENT
            ]
        )

        lines += ["", "## What the checkers found"]
        if self.flags:
            lines.append(f"{len(self.flags)} finding(s):")
            lines.extend(f"- {f}" for f in self.flags)
        elif self.notes:
            lines.append("no check reported a problem. Every check that ran:")
            lines.extend(f"- {n}" for n in self.notes)
        else:
            lines.append(ABSENT + " (no notes were written for this decision)")

        if self.flags and self.notes:
            lines += ["", "<details><summary>every check that ran</summary>", ""]
            lines.extend(f"- {n}" for n in self.notes)
            lines += ["", "</details>"]

        lines += ["", "## What the risk layer did"]
        if self.risk is None:
            lines.append(ABSENT + " (no risk record for this sequence)")
        elif not self.risk.get("intervened"):
            lines.append(
                f"nothing to narrow — {self.risk.get('reason') or 'no binding constraint'}. "
                f"The Constitution may only reduce, so an untouched decision is the model's own."
            )
        else:
            lines.append(
                f"**{self.risk.get('binding_constraint')}** bound: "
                f"{self.risk.get('reason')}. Size {self.risk.get('quantity_before')} → "
                f"{self.risk.get('quantity_after')}; reduced only: "
                f"{self.risk.get('constitution_only_reduced')}."
            )

        lines += ["", "## The rules in force"]
        if self.protocol is None:
            lines.append(
                ABSENT + " — this decision predates the pre-registered protocol and is reported "
                "as ungoverned rather than as compliant"
            )
        else:
            p = self.protocol.get("protocol", {})
            lines.append(
                f"`{p.get('protocol_id')}` v{p.get('version')}, digest "
                f"`{self.protocol.get('protocol_digest')}`, committed "
                f"{self.protocol.get('committed_at')} against ledger head "
                f"`{self.protocol.get('ledger_head')}` at "
                f"{self.protocol.get('ledger_entries')} entries — so it was frozen before this "
                f"decision existed."
            )

        lines += [
            "", "## What happened next", self.outcome_line,
            "", "## Integrity",
            f"market state `{e.get('market_state_hash', ABSENT)}`, approved intent "
            f"`{e.get('approved_intent_hash', ABSENT)}`, chained to "
            f"`{e.get('prev_hash', ABSENT)}`.",
            "", "## Where each part came from",
        ]
        lines.extend(f"- {s.render()}" for s in self.sources)
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq, "symbol": self.symbol, "verdict": self.entry.get("verdict"),
            "settled": self.is_settled, "flags": len(self.flags), "notes": len(self.notes),
            "risk_recorded": self.risk is not None,
            "governed": self.protocol is not None,
            "outcome": self.outcome_line,
        }


def build(
    entry: dict[str, Any],
    *,
    notes: Sequence[dict[str, Any]] = (),
    risk: Sequence[dict[str, Any]] = (),
    commitments: Sequence[dict[str, Any]] = (),
    paths: dict[str, str] | None = None,
) -> Card:
    """Join one decision to everything written about it.

    Matching is by sequence number throughout, which is the only identifier every writer shares.
    A part with no match is left ``None`` and printed as absent rather than filled in.
    """
    seq = int(entry.get("seq", 0))
    where = paths or {}
    note_row = next((n for n in notes if int(n.get("seq", -1)) == seq), None)
    risk_row = next((r for r in risk if int(r.get("seq", -1)) == seq), None)

    governing: dict[str, Any] | None = None
    for c in commitments:
        entries_at = int(c.get("ledger_entries", 0))
        if seq > entries_at and (
            governing is None or entries_at > int(governing.get("ledger_entries", 0))
        ):
            governing = c

    sources = [Source("the decision", where.get("ledger", "data/paper_ledger.jsonl"))]
    if note_row is not None:
        sources.append(Source("the checks", where.get("notes", "data/desk_notes.jsonl")))
    if risk_row is not None:
        sources.append(Source("the risk ruling", where.get("risk", "data/risk_records.jsonl")))
    if governing is not None:
        sources.append(
            Source("the rules", where.get("protocol", "data/protocol_commitments.jsonl"))
        )

    return Card(
        seq=seq,
        entry=entry,
        notes=[str(n) for n in (note_row or {}).get("notes", [])],
        flags=[str(f) for f in (note_row or {}).get("flags", [])],
        risk=risk_row,
        protocol=governing,
        sources=sources,
    )


def index(cards: Sequence[Card]) -> str:
    """A table of every decision, so a reader can find the interesting ones."""
    lines = [
        "# Decision cards",
        "",
        f"{len(cards)} decision(s). Every row links to the full trail.",
        "",
        "| # | symbol | verdict | checks flagged | risk bound | governed | outcome |",
        "|---|--------|---------|----------------|-----------|----------|---------|",
    ]
    for c in cards:
        bound = (
            "—" if c.risk is None
            else (c.risk.get("binding_constraint") if c.risk.get("intervened") else "none")
        )
        outcome = "settled" if c.is_settled else "open"
        lines.append(
            f"| [{c.seq}](decision-{c.seq}.md) | {c.symbol} | {c.entry.get('verdict')} | "
            f"{len(c.flags)} | {bound} | {'yes' if c.protocol else 'no'} | {outcome} |"
        )
    return "\n".join(lines)


def _load(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def main(argv: list[str] | None = None) -> int:
    from argus.paper.protocol import PROTOCOL_PATH
    from argus.paper.runner import LEDGER_PATH, NOTES_PATH, RISK_PATH

    parser = argparse.ArgumentParser(description="ARGUS decision cards")
    parser.add_argument("--seq", type=int, default=0, help="one decision")
    parser.add_argument("--all", action="store_true", help="every decision")
    parser.add_argument("--out", default="", help="directory to write cards into")
    args = parser.parse_args(argv)

    entries = _load(LEDGER_PATH)
    notes = _load(NOTES_PATH)
    risk = _load(RISK_PATH)
    commitments = _load(PROTOCOL_PATH)

    if not args.all and not args.seq:
        args.seq = int(entries[-1]["seq"]) if entries else 0

    chosen = (
        entries if args.all
        else [e for e in entries if int(e.get("seq", 0)) == args.seq]
    )
    if not chosen:
        print(f"no decision {args.seq} in the ledger")
        return 1

    cards = [
        build(e, notes=notes, risk=risk, commitments=commitments) for e in chosen
    ]
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        for card in cards:
            (out / f"decision-{card.seq}.md").write_text(card.render(), encoding="utf-8")
        (out / "index.md").write_text(index(cards), encoding="utf-8")
        print(f"wrote {len(cards)} card(s) -> {out}")
        return 0

    for card in cards:
        print(card.render())
        print()
    return 0


__all__ = ["ABSENT", "Card", "Source", "build", "index", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
