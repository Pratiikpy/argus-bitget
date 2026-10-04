"""Each part of the console measured on its own, laid out as JPMorgan's "Ask David" (5.1).

Ask David (JPMorgan Private Bank's research assistant, presented by its engineers at LangChain
Interrupt, 2025-05-29; notes in ``research/s2-field/youtube/research-workbench/
08_jpmorgan-ask-david-langgraph.md``) is a supervisor that routes to specialists — structured data,
unstructured documents, analytics — then a personalisation node and a reflection node before the
answer, and its team's stated practice is to **evaluate each sub-agent independently, not only end
to end, to find the weak link**. ARGUS's console has the same parts under its own names. This reads
each part's own measurement from the dated artefact that holds it and puts them side by side; it
renames nothing and runs nothing.

The weak link, read from the numbers rather than chosen: personalisation's recall on the last blind
set is the lowest figure on the card (it trails mem0 on recall while making no false memories), and
the reflection node is deterministic where Ask David's is a model judge — it can mark an untraced
figure but cannot judge whether a sentence without one makes sense.

    python -m argus.eval.node_scorecard       # writes data/node_scorecard.json
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final

from argus.truth import artefact
from argus.truth.paths import DATA_DIR

OUT: Final = DATA_DIR / "node_scorecard.json"


def _load(data: Path, name: str) -> dict[str, Any] | None:
    try:
        blob = json.loads((data / name).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return blob if isinstance(blob, dict) else None


def _row(node: str, ask_david: str, argus: str, measure: str, value: str, artefact_name: str,
         dated: str) -> dict[str, str]:
    return {"node": node, "ask_david": ask_david, "argus": argus, "measure": measure,
            "value": value, "artefact": f"data/{artefact_name}", "dated": dated}


def _router(data: Path) -> dict[str, str] | None:
    blob = _load(data, "lui_final_heldout_report.json")
    if not blob:
        return None
    alone = blob["kind_model_alone"]
    return _row("supervisor", "supervisor routing to a sub-graph",
                "the question classifier (patterns + kind model) and Qwen planner",
                "questions routed to the right kind, blind 12-language set, classifier alone",
                f"{alone['correct']}/{alone['rows']} ({alone['accuracy']:.1%})",
                "lui_final_heldout_report.json", str(blob.get("measured_at", "")))


def _structured(data: Path) -> dict[str, str] | None:
    blob = _load(data, "financebench_xbrl.json")
    if not blob:
        return None
    head = blob["headline"]
    return _row("structured data", "structured-data agent (NL to SQL/API)",
                "the engines reading SEC XBRL, Bitget and FRED by API",
                "FinanceBench metrics-generated questions answered from XBRL",
                f"{head['argus_correct_of_50']}/50 "
                f"(best rival {head['best_rival_correct_of_50']}/50)",
                "financebench_xbrl.json", str(blob.get("generated_at", ""))[:10])


def _unstructured(data: Path) -> dict[str, str] | None:
    blob = _load(data, "document_qa_eval.json")
    if not blob:
        return None
    right, audit = blob["answer_correct"], blob["audit_deterministic"]["raw"]
    return _row("unstructured documents", "RAG over documents",
                "filing Q&A over SEC 8-K and 10-Q text, cited by passage",
                "answerable questions right; sentences supported by their cited passage",
                f"{right['with_enforcement']}/{right['of']} right, "
                f"{right['refused_when_unanswerable']}/{right['of_unanswerable']} unanswerable "
                f"refused; {audit['supported']}/"
                f"{audit['kept']} sentences supported, {blob['fabricated_citation_ids']} "
                "fabricated citations", "document_qa_eval.json", str(blob.get("generated", "")))


def _analytics(data: Path) -> dict[str, str] | None:
    blob = _load(data, "backtest_critic_comparison.json")
    if not blob:
        return None
    return _row("analytics", "analytics agent (API tools, text-to-code)",
                "backtests, stress, sizing and the backtest critic",
                "planted backtest defects caught, false alarms on clean code",
                f"{blob['argus_caught']}/{blob['planted']} caught, "
                f"{blob['argus_false_alarm_fixtures']}/{blob['clean']} false alarms "
                f"(backtest-truth {blob['rival_caught']}/{blob['planted']})",
                "backtest_critic_comparison.json", "")


def _personalisation(data: Path) -> dict[str, str] | None:
    blob = _load(data, "memory_comparison.json")
    if not blob or "held_out_4" not in blob:
        return None
    last = blob["held_out_4"]
    ours, theirs = last["argus_recall"], last["mem0_recall"]
    false_ours, false_theirs = last["argus_false_positives"], last["mem0_false_positives"]
    return _row("personalisation", "personalisation node (same facts, the reader's register)",
                "trader memory (what the trader says about themselves) and the saved book",
                "last blind set: facts recalled; memories invented from talk about others",
                f"recall {ours['recalled']}/{ours['total']} (mem0 {theirs['recalled']}/"
                f"{theirs['total']}); false memories {false_ours['false_positives']}/"
                f"{false_ours['total']} (mem0 {false_theirs['false_positives']}/"
                f"{false_theirs['total']})", "memory_comparison.json",
                str(last.get("frozen_at", ""))[:10])


def _reflection(data: Path) -> dict[str, str] | None:
    blob = _load(data, "provenance_unseen.json")
    if not blob:
        return None
    first = blob["first_read"]
    labelled = first["content_lines"] - first["unlabelled"]
    return _row("reflection", "reflection node (an LLM judge checks the draft before it returns)",
                "deterministic: every line labelled by source, no source no answer, untraced "
                "figures marked", "content lines carrying a provenance label, unseen questions",
                f"{labelled}/{first['content_lines']} ({first['coverage']:.1%})",
                "provenance_unseen.json", str(blob.get("collected", "")))


NODES: Final[tuple[Callable[[Path], dict[str, str] | None], ...]] = (
    _router, _structured, _unstructured, _analytics, _personalisation, _reflection)


def card(data: Path = DATA_DIR) -> list[dict[str, str]]:
    return [row for build in NODES if (row := build(data)) is not None]


def run(*, out: Path = OUT) -> dict[str, Any]:
    rows = card()
    blob = {"pattern": "JPMorgan Ask David (LangChain Interrupt, 2025-05-29)", "nodes": rows,
            "weak_link": "personalisation recall on the last blind set; and a reflection node that "
                         "checks figures, not sense"}
    artefact.write(out, blob)
    return blob


def main() -> int:  # pragma: no cover - CLI
    print(json.dumps(run(), indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
