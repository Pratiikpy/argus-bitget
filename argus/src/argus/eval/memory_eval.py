"""Does the console's memory change later answers, and only when it should?

`lui/memory.py` claims three things, each tested here on a fixed set written before the first run:

1. **Extraction precision and recall.** 30 statements a trader makes about themselves, each with
   the fact it states, and 20 sentences that state nothing about the trader but share its words
   ("I think the Fed meets next week", "what if I can't lose more than 10%?" asked as a question).
   A fact extracted from the second set is a false memory.
2. **Effect.** 20 two-session tasks: a statement in session one, a question in session two. With
   the memory, the session-two answer must carry the remembered fact's line; with memory off (the
   ablation) it must not, and the rest of the answer must be the same engine's.
3. **Scope.** The memory is sent by the client and never stored, so one trader's facts cannot
   reach another's answer: the second trader's session two is run with an empty memory and must
   carry none of the first trader's lines.

The effect set calls the live engines (Bitget candles and tickers); it runs in a few minutes and
spends no model tokens (the planner is not used: the patterns route every question in the set).

    python -m argus.eval.memory_eval
"""

from __future__ import annotations

import argparse
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from argus.eval.artefact import write

REPORT_PATH = Path(__file__).resolve().parents[3] / "data" / "memory_eval.json"

STATEMENTS: tuple[tuple[str, str], ...] = (
    ("I can't lose more than 10% of my account", "max_loss"),
    ("i cant afford to lose more than 5%", "max_loss"),
    ("my max drawdown is 15%", "max_loss"),
    ("my loss limit is 8%", "max_loss"),
    ("I never lose more than 20% on anything", "max_loss"),
    ("my risk budget is 20%", "budget"),
    ("no single name above 30% of my risk", "budget"),
    ("I'm a swing trader", "horizon"),
    ("Im a day trader", "horizon"),
    ("i am a long-term investor", "horizon"),
    ("I usually hold for weeks", "horizon"),
    ("I hold for months", "horizon"),
    ("I'm conservative", "style"),
    ("im pretty aggressive", "style"),
    ("I am risk-averse", "style"),
    ("I mostly trade earnings", "style"),
    ("I trade momentum", "style"),
    ("I only trade crypto", "style"),
    ("my account is 50k", "capital"),
    ("my portfolio is $250,000", "capital"),
    ("I have $20k to invest", "capital"),
    ("I think NVDA will outperform on AI capex", "thesis"),
    ("i believe btc is going to crash", "thesis"),
    ("I expect TSLA to miss", "thesis"),
    ("I think gold will rally into year end", "thesis"),
    ("I reckon ETH can double", "thesis"),
    ("I don't trade TSLA", "avoid"),
    ("I never touch meme coins like DOGE", "avoid"),
    ("i dont want any MSTR", "avoid"),
    ("I won't hold COIN", "avoid"),
)
"""Each statement and the kind of fact it states."""

NOT_ABOUT_ME: tuple[str, ...] = (
    "I think the Fed meets next week",
    "what if I can't lose more than 10%?",
    "is a 20% risk budget normal?",
    "what does a swing trader usually hold?",
    "how do day traders manage risk",
    "is momentum working this year",
    "what is a conservative allocation",
    "how big is a 50k account in BTC",
    "does TSLA trade on weekends",
    "why did NVDA fall",
    "I think",
    "what do you think about NVDA",
    "should I trade earnings",
    "what is my max drawdown",
    "how much of my risk is NVDA",
    "is DOGE a meme coin",
    "what if BTC crashes 20%",
    "can ETH double",
    "how much is 20k of gold",
    "do rTokens pay dividends",
)
"""Sentences that share the statements' words and state nothing about the trader."""

TASKS: tuple[tuple[str, str, str], ...] = (
    ("I can't lose more than 10%", "what if the nasdaq drops 10%? I hold 60% NVDA 40% AAPL",
     "Remembered:"),
    ("I can't lose more than 5%", "what if BTC drops 15%? I hold 50% BTC 50% ETH", "Remembered:"),
    ("my max drawdown is 15%", "stress 70% TSLA 30% NVDA for a 10% nasdaq drop", "Remembered:"),
    ("my loss limit is 8%", "what if the nasdaq drops 20%? I hold 100% QQQ", "Remembered:"),
    ("I never lose more than 20% on anything", "what if gold drops 10%? I hold 50% XAU 50% NVDA",
     "Remembered:"),
    ("I'm a swing trader", "will BTC go up?", "Remembered:"),
    ("Im a day trader", "will NVDA be higher?", "Remembered:"),
    ("I usually hold for weeks", "will ETH go up?", "Remembered:"),
    ("I hold for months", "will gold be higher?", "Remembered:"),
    ("I'm a swing trader", "odds TSLA is up", "Remembered:"),
    ("I think NVDA will outperform on AI capex", "where is NVDA trading", "Your thesis on NVDA"),
    ("i believe btc is going to crash", "BTC price", "Your thesis on BTC"),
    ("I expect TSLA to miss", "TSLA funding rate", "Your thesis on TSLA"),
    ("I think gold will rally into year end", "gold price", "Your thesis on XAU"),
    ("I reckon ETH can double", "is ETH overbought", "Your thesis on ETH"),
    ("I don't trade TSLA", "where is TSLA trading", "Remembered:"),
    ("i dont want any MSTR", "MSTR price", "Remembered:"),
    ("I won't hold COIN", "COIN funding rate", "Remembered:"),
    ("my risk budget is 20%",
     "I hold 50% NVDA, 50% AAPL — what does adding 20% TSLA do to my risk?", "Remembered:"),
    ("no single name above 30% of my risk", "how risky is 60% NVDA 40% MSFT", "Remembered:"),
)
"""(session-one statement, session-two question, the line the memory must add)."""


def extraction() -> dict[str, Any]:
    from argus.lui.memory import extract

    hits = [(text, kind, [f.kind for f in extract(text)]) for text, kind in STATEMENTS]
    recalled = sum(1 for _, kind, got in hits if kind in got)
    false = [(text, [f.kind for f in extract(text)]) for text in NOT_ABOUT_ME]
    false = [(text, kinds) for text, kinds in false if kinds]
    return {"statements": len(STATEMENTS), "recalled": recalled,
            "missed": [(t, k, g) for t, k, g in hits if k not in g],
            "not_about_me": len(NOT_ABOUT_ME), "false_memories": false}


def effect() -> dict[str, Any]:
    for key in list(os.environ):
        if "QWEN" in key:
            del os.environ[key]
    from argus.lui.server import handle_ask

    rows = []
    for statement, question, marker in TASKS:
        first = handle_ask(statement, [], visitor="memory-eval-a")
        remembered = str(first.get("memory") or "")
        with_memory = handle_ask(question, [], visitor="memory-eval-a", memory=remembered)
        without = handle_ask(question, [], visitor="memory-eval-a", memory="")
        other = handle_ask(question, [], visitor="memory-eval-b", memory="")

        def carries(payload: dict[str, Any], marker: str = marker) -> bool:
            return any(str(line).startswith(marker) for line in payload.get("lines") or [])

        rows.append({
            "statement": statement, "question": question, "marker": marker,
            "stored": remembered != "[]" and bool(remembered),
            "with_memory_carries": carries(with_memory),
            "without_memory_carries": carries(without),
            "other_trader_carries": carries(other),
            "same_engine": with_memory.get("classified_by") == without.get("classified_by"),
        })
    changed = sum(1 for r in rows if r["with_memory_carries"] and not r["without_memory_carries"])
    return {"tasks": len(rows), "changed_by_memory": changed,
            "leaked_to_other_trader": sum(1 for r in rows if r["other_trader_carries"]),
            "ablation_carried": sum(1 for r in rows if r["without_memory_carries"]),
            "rows": rows}


def run(*, online: bool = True) -> dict[str, Any]:
    report: dict[str, Any] = {"generated_at": datetime.now(UTC).isoformat(),
                              "extraction": extraction()}
    if online:
        report["effect"] = effect()
    write(REPORT_PATH, report)
    return report


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description="measure the console's memory")
    parser.add_argument("--offline", action="store_true", help="extraction only, no live engines")
    args = parser.parse_args(argv)
    report = run(online=not args.offline)
    ex = report["extraction"]
    print(f"extraction: {ex['recalled']}/{ex['statements']} recalled, "
          f"{len(ex['false_memories'])}/{ex['not_about_me']} false memories")
    for miss in ex["missed"]:
        print("  missed", miss)
    for false in ex["false_memories"]:
        print("  false", false)
    if "effect" in report:
        ef = report["effect"]
        print(f"effect: {ef['changed_by_memory']}/{ef['tasks']} answers changed by memory; "
              f"ablation carried {ef['ablation_carried']}; leaked to another trader "
              f"{ef['leaked_to_other_trader']}")
        for row in ef["rows"]:
            if not row["with_memory_carries"]:
                print("  no effect:", row["statement"], "->", row["question"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
