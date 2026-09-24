"""RESEARCH-BENCH: does the console understand the research questions a trader actually types?

`lui/research.py` turns a question like *"what would adding 20% TSLA do to my risk?"* into a request
the desk's engines answer. Whether that works cannot be judged from phrasings its author wrote,
because an author writes the phrasings their parser already expects — this project has been caught
by that exact gap before (`eval/obliquebench.py`: 95% on the author's corpus, 9.5% on a held-out
one). So the two corpora here were written by **agents that were forbidden to read this
repository**, before any score was taken:

* ``data/research_bench_corpus_a.json`` — 60 questions, one writer, varied tone.
* ``data/research_bench_corpus_b.json`` — 60 questions from six personas a trader desk actually
  meets: a nervous beginner, a terse PM, a crypto-native writing slang, a non-native English
  speaker, a careful retiree and a quant.

Each question carries the kind of answer it needs (impact, stress, compare, execution, profile) or
``none`` — a question that must NOT be analysed: an instrument off the venue, a plain order, a
question about the desk's own past decisions, chit-chat. Refusing those is scored as a success;
analysing them is scored as a failure, because a confident research answer to a question nobody
asked is the worst thing this layer can do.

**Stated contamination, so the numbers are read correctly.** Every corpus was written by an agent
forbidden to read this repository, and every one stops being held out the moment its misses inform
a fix. The held-out figure for each is therefore its *first* full-pipeline run:

* **A** (60): 59/60 (98%). Its misses then informed the deterministic patterns.
* **B** (60, six personas): 57/60 (95%). Its misses exposed the planner copying a literal ``0.0``
  confidence from the prompt template, and later asked for skew, kurtosis and R², now computed.
* **D** (100, ten labels, six personas incl. code-switching): **94/100**, patterns alone 55/100,
  must-refuse 9/10 — scored once, before any fix. Its misses were names the model was never shown
  (company names, misspellings, other languages), a book made only of the name asked about, and a
  crypto price target read as fundamentals; all fixed, so D is no longer held out.
* **C** (90, all ten labels including quote, technicals, fundamentals and analogue): **86/90
  (96%)**, patterns alone 66/90, must-refuse 8/9. Its misses were three lowercase tickers the model
  was never asked about ("amd", "hood", "gme") and one Hinglish price forecast read as a quote;
  both fixed, so C is no longer held out either.

Four questions A and B labelled must-refuse named AMD, PLTR, SPY and gold, refused at the time as
off-venue; Bitget lists all four, the console now answers any listed contract, and each is
relabelled in its corpus with the reason recorded. Figures after the fixes are in
``data/research_bench.json`` and are not held out. A fourth corpus is the only way to get a fresh
held-out number, and it should be written before anything else changes.

    python -m argus.eval.researchbench            # patterns only, no key needed
    python -m argus.eval.researchbench --model    # full pipeline, ~450 tokens per question
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from argus.eval import artefact
from argus.lui.research import (
    ResearchKind,
    ResearchRequest,
    detect,
    plan_with_model,
    worth_asking_the_model,
)

DATA = Path(__file__).resolve().parents[3] / "data"
CORPORA = (DATA / "research_bench_corpus_a.json", DATA / "research_bench_corpus_b.json",
           DATA / "research_bench_corpus_c.json", DATA / "research_bench_corpus_d.json")
REPORT_PATH = DATA / "research_bench.json"

EXPECTED: dict[str, ResearchKind | None] = {
    "impact": ResearchKind.IMPACT,
    "profile": ResearchKind.IMPACT,
    "stress": ResearchKind.STRESS,
    "compare": ResearchKind.COMPARE,
    "execution": ResearchKind.EXECUTION,
    "quote": ResearchKind.QUOTE,
    "technicals": ResearchKind.TECHNICALS,
    "fundamentals": ResearchKind.FUNDAMENTALS,
    "analogue": ResearchKind.ANALOGUE,
    "none": None,
}
"""A single-name profile is answered by the impact engine with no book — the standalone position."""


@dataclass(frozen=True)
class Case:
    text: str
    label: str
    symbols: tuple[str, ...]

    @property
    def wanted_symbols(self) -> set[str]:
        from argus.market import universe

        out = set()
        for raw in self.symbols:
            upper = raw.upper()
            upper = {"GOOG": "GOOGL", "GOOGLE": "GOOGL"}.get(upper, upper)
            # The corpus author writes tickers as a trader would ("XAU", "CVX"); the venue's
            # contract for one can carry a suffix (CVXSTOCKUSDT), which the registry resolves.
            hit = universe.resolve(upper)
            out.add(hit[0] if hit is not None else
                    upper if upper.endswith("USDT") else upper + "USDT")
        return out


def load(path: Path) -> list[Case]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [Case(q["text"], q["label"], tuple(q.get("symbols", []))) for q in raw["questions"]]


def read(case: Case, client: Any = None) -> tuple[ResearchRequest | None, str]:
    """The request the console would build, in the console's own order: model first when there
    is one and the question names a traded instrument and is not about the desk's record."""
    if client is not None and worth_asking_the_model(case.text):
        planned, _ = plan_with_model(case.text, client)
        if planned is not None:
            return planned, "model"
    request = detect(case.text)
    return request, "patterns" if request is not None else "-"


def correct(case: Case, request: ResearchRequest | None) -> bool:
    wanted = EXPECTED[case.label]
    if wanted is None:
        return request is None
    if request is None or request.kind is not wanted:
        return False
    return not case.wanted_symbols or bool(set(request.symbols) & case.wanted_symbols)


def score(cases: Sequence[Case], client: Any = None) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        request, by = read(case, client)
        rows.append({
            "text": case.text, "label": case.label, "read_by": by,
            "got": None if request is None else str(request.kind),
            "correct": correct(case, request),
        })
    by_label: dict[str, dict[str, int]] = {}
    for row in rows:
        bucket = by_label.setdefault(row["label"], {"correct": 0, "total": 0})
        bucket["total"] += 1
        bucket["correct"] += int(row["correct"])
    must_refuse = [r for r in rows if r["label"] == "none"]
    return {
        "correct": sum(r["correct"] for r in rows),
        "total": len(rows),
        "by_label": by_label,
        "must_refuse_refused": sum(r["correct"] for r in must_refuse),
        "must_refuse_total": len(must_refuse),
        "misses": [r for r in rows if not r["correct"]],
    }


TOKENS_PER_QUESTION = 1_500
"""Model allowance per scored question: a planner call measures about 450 tokens; the margin keeps
a long question from exhausting the allowance before the corpus ends."""


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description="Research-question understanding, held out")
    parser.add_argument("--model", action="store_true",
                        help="use the configured model first, as the hosted console does")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args(argv)

    client = None
    if args.model:
        from argus.lui.router import build_router

        client = build_router(budget_tokens=TOKENS_PER_QUESTION)
        if client is None:
            print("no model key configured; scoring patterns only")
    report: dict[str, Any] = {"mode": "model+patterns" if client else "patterns"}
    for path in CORPORA:
        cases = load(path)
        if client is not None:
            # A fresh allowance per corpus, sized to it. One shared 150,000-token allowance ran out
            # two-thirds of the way through the fourth corpus (2026-09-24) and scored the rest on
            # the patterns alone — 57/100 against 98/100 an hour earlier on the same code. A
            # figure that depends on the order corpora are scored in is not a measurement.
            client = build_router(budget_tokens=TOKENS_PER_QUESTION * len(cases))
        result = score(cases, client)
        report[path.stem] = result
        print(f"{path.stem}: {result['correct']}/{result['total']} "
              f"({result['correct'] / result['total']:.0%}); must-refuse refused "
              f"{result['must_refuse_refused']}/{result['must_refuse_total']}")
        for miss in result["misses"]:
            print(f"   miss [{miss['label']}, got {miss['got']}] {miss['text']}")
    if args.save:
        artefact.write(REPORT_PATH, report)
        print(f"saved -> {REPORT_PATH}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = ["CORPORA", "EXPECTED", "Case", "correct", "load", "read", "score"]
