"""Does the console split a multi-part question into its parts, and leave single questions whole?

Two measurements, both offline (decomposition only; no engine or model is called):

1. **Recall on multi-part questions.** 25 questions written for this module on 2026-09-25, before
   `lui/multistep.py` was run on them, each labelled with the engine every part needs. A question
   scores when every labelled part is found with the right kind, in order.
2. **False splits on single questions.** Every question in the three held-out routing corpora
   (`data/lui_*_2026-09-25.jsonl`, 680 rows written by writers who never read this repository)
   asks one thing. A split there is a defect: one answer cut into pieces.

    python -m argus.eval.multistep_eval
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from argus.eval.artefact import write

DATA = Path(__file__).resolve().parents[3] / "data"
REPORT_PATH = DATA / "multistep_eval.json"
CORPORA = ("lui_final_heldout_2026-09-25.jsonl", "lui_heldout_corpus_2026-09-25.jsonl",
           "lui_blind_corpus_2026-09-25.jsonl")

MULTI: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Is NVDA overbought, and what would adding 20% of it do to my book? Then how do I split a "
     "$50k buy?", ("technicals", "impact", "execution")),
    ("BTC funding rate and ETH open interest", ("quote", "sentiment")),
    ("compare NVDA and AMD risk; hedge my book; when does NVDA report earnings",
     ("compare", "hedge", "fundamentals")),
    ("where is NVDA trading? and when does it report earnings?", ("quote", "fundamentals")),
    ("I hold 40% NVDA 60% AAPL. what if the nasdaq drops 10%? and how do I hedge it?",
     ("stress", "hedge")),
    ("what is the TSLA funding rate? is TSLA overbought?", ("quote", "technicals")),
    ("how has BTC done over the last week, and what are the odds it's up this week?",
     ("quote", "analogue")),
    ("when does AAPL report earnings? what are analysts' targets? is it overbought?",
     ("fundamentals", "technicals")),
    ("what's the long/short ratio on SOL; how liquid is SOL on weekends",
     ("sentiment", "quote")),
    ("ETH price, then how should I split a $100k ETH buy", ("quote", "execution")),
    ("is MSTR riskier than COIN? and what is MSTR's liquidation price at 5x long?",
     ("compare", "leverage")),
    ("what's the CPI print? and how does QQQ usually react on CPI days?", ("macro", "event")),
    ("where will NVDA open? then what's its 200-day moving average?", ("quote", "technicals")),
    ("what's the gold price; what does a 10% drop in gold do to 50% XAU 50% NVDA",
     ("quote", "stress")),
    ("is the hype on NVDA real? and what's the news on it?", ("sentiment", "news")),
    ("what are rTokens? and what is the price of RNVDAUSDT?", ("venue", "quote")),
    ("take profit for a TSLA long? and where should my stop go?", ("analogue",)),
    ("TQQQ decay if QQQ goes sideways for a month; and what is QQQ's 52 week high and low",
     ("quote", "quote")),
    ("NVDA spread and depth; then split a $250k NVDA sell", ("execution",)),
    ("how risky is 60% NVDA 40% MSFT? also what is its beta to the S&P?", ("book",)),
    ("BTC sentiment and ETH sentiment", ("sentiment", "sentiment")),
    ("what does adding 10% COIN do to 50% NVDA 50% AAPL? and is COIN overbought?",
     ("impact", "technicals")),
    ("should I hold NVDA for a week? and what's the weekend gap risk on it?",
     ("analogue", "analogue")),
    ("what's the fed doing with rates; how does BTC react to that", ("macro", "macro")),
    ("gold vs bitcoin risk; then how do I hedge 50% BTC 50% ETH", ("compare", "hedge")),
)
"""Multi-part questions and the engine each part needs, in order. Where two parts need the same
engine on the same name, one label is listed and the question must be answered whole (the loop
breaker runs that engine once over both clauses).

First run, 2026-09-25, before any fix: 18/25 split correctly and 3/680 single held-out questions
split by mistake. Three of the seven misses were this set's own labelling (a single-label row
scored as "not split", which is the behaviour the label means); the rest were fixed in
`lui/multistep.py` and `lui/research.py`, and one row was reworded ("how does that affect BTC"
became "how does BTC react to that") when pronoun-led clauses were made to elaborate the clause
before them. The figures this module reports now are therefore not held out."""


def recall() -> dict[str, Any]:
    from argus.lui.multistep import parts

    rows = []
    for question, kinds in MULTI:
        found = parts(question) or []
        got = tuple(p.request.kind.value for p in found)
        # one label means both parts need one engine on one name: answered whole, not split
        ok = (not found) if len(kinds) == 1 else got == kinds
        rows.append({"question": question, "expected": list(kinds), "got": list(got), "ok": ok})
    return {"questions": len(rows), "split_correctly": sum(r["ok"] for r in rows), "rows": rows}


def false_splits() -> dict[str, Any]:
    from argus.lui.multistep import parts

    split: list[dict[str, Any]] = []
    total = 0
    for name in CORPORA:
        path = DATA / name
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            total += 1
            found = parts(str(row["text"]))
            if found is not None:
                split.append({"corpus": name, "text": row["text"],
                              "parts": [p.text for p in found]})
    return {"single_questions": total, "split": len(split), "rows": split}


def run() -> dict[str, Any]:
    report = {"generated_at": datetime.now(UTC).isoformat(), "multi": recall(),
              "single": false_splits()}
    write(REPORT_PATH, report)
    return report


def main() -> int:  # pragma: no cover - CLI
    report = run()
    multi, single = report["multi"], report["single"]
    print(f"multi-part: {multi['split_correctly']}/{multi['questions']} split into the right parts")
    for row in multi["rows"]:
        if not row["ok"]:
            print("  miss", row["expected"], row["got"], row["question"][:80])
    print(f"single questions split by mistake: {single['split']}/{single['single_questions']}")
    for row in single["rows"][:30]:
        print("  split", row["parts"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
