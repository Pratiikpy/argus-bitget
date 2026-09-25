"""Every number a question states should reach the request built from it.

The corpus scores (`eval/kindtrain.py`, the held-out console runs) grade which engine a question
reaches. They never looked at the figures: on 2026-09-25 "what does adding 15% TSLA do to my
risk?" reached the right engine and was answered for 20%, because the kind model's plan carried no
size and the default applied; "trim TSLA to 10%" was read as an add. Both scored as correct.

This check reads each research question in the corpora the way the console does (the kind model
through `plan_with_model`, then the patterns where they win), and asks of every percentage, dollar
amount and multiple written in it: does it appear anywhere in the request — as the size, a target,
a shock, a book weight, a budget, a notional, a leverage or a horizon? A number that appears nowhere
was dropped or misread. It is a detector, not a grader: a dropped number is sometimes right (a
"10% chance" said in passing), so every flag is listed for reading, and the count is the figure
tracked over time.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

DATA = Path(__file__).resolve().parents[3] / "data"
REPORT = DATA / "figurecheck.json"
CORPORA = ("lui_final_heldout_2026-09-25.jsonl", "lui_heldout_corpus_2026-09-25.jsonl",
           "lui_blind_corpus_2026-09-25.jsonl")
KINDS = ("impact", "stress", "execution", "leverage", "hedge", "book")

_PCT = re.compile(r"(?<![\w.])-?(\d+(?:\.\d+)?)\s*(?:%|percent\b|pct\b)", re.I)
_MONEY = re.compile(r"\$\s?(\d[\d,]*(?:\.\d+)?)\s*([km](?![a-z]))?"
                    r"|\b(\d[\d,]*(?:\.\d+)?)\s*([km])\b"
                    r"|\b(\d[\d,]*(?:\.\d+)?)\s*(?:usdt|usd|dollars)\b", re.I)
"""Amounts. The dollar form ends at the last digit, not at a word boundary: in "$50,000的AAPL" the
Chinese character that follows is a word character, and a boundary rule read "$50"."""
_MULT = re.compile(r"\b(\d+(?:\.\d+)?)\s*x\b", re.I)


def stated(text: str) -> list[tuple[str, float]]:
    """(kind, value) for every percentage, amount and multiple written in ``text``."""
    found: list[tuple[str, float]] = [("pct", float(m.group(1))) for m in _PCT.finditer(text)]
    for m in _MONEY.finditer(text):
        raw = m.group(1) or m.group(3) or m.group(5)
        scale = (m.group(2) or m.group(4) or "").lower()
        value = float(raw.replace(",", "")) * {"k": 1e3, "m": 1e6}.get(scale, 1.0)
        found.append(("usd", value))
    found += [("mult", float(m.group(1))) for m in _MULT.finditer(text)]
    return found


def carried(request: Any) -> set[float]:
    """Every number the request holds, in the units a question writes them in."""
    values: set[float] = set()
    for pct in (request.size, request.target, request.resize_from, request.budget,
                request.cash):
        if pct:
            values.add(round(pct * 100, 4))
    if request.resize_by is not None:
        values.add(round(abs(request.resize_by) * 100, 4))
    if request.shock_pct is not None:
        values.add(round(abs(request.shock_pct), 4))
    for weight in request.book.values():
        values.add(round(weight * 100, 4))
    if request.notional is not None:
        values.add(round(float(request.notional), 4))
    if request.leverage:
        values.add(round(request.leverage, 4))
    if request.horizon_hours:
        values.add(float(request.horizon_hours))
    return values


def _accounted(kind: str, value: float, held: set[float]) -> bool:
    if kind == "usd":
        return any(abs(v - value) <= max(1.0, 0.001 * value) for v in held) or \
            any(abs(v - value * m) <= 1.0 for v in held for m in (2, 3, 5, 10, 20))
    return any(abs(v - value) < 0.05 for v in held)


def run() -> dict[str, Any]:
    from argus.lui.kindmodel import LocalPlanner, kind_model
    from argus.lui.research import (
        _VAR,
        _VAR_LEVEL,
        _vol_multiple,
        detect,
        pattern_reading_wins,
        plan_with_model,
    )

    model = kind_model()
    planner = LocalPlanner(model) if model is not None else None
    rows: list[dict[str, Any]] = []
    for name in CORPORA:
        path = DATA / name
        if not path.exists():
            continue
        for line in path.read_text("utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("expected", item.get("label")) not in KINDS:
                continue
            text = str(item["text"])
            numbers = stated(text)
            if not numbers:
                continue
            planned = plan_with_model(text, planner)[0] if planner is not None else None
            patterned = detect(text)
            request = (patterned if patterned is not None and (
                planned is None or pattern_reading_wins(patterned, text)) else planned)
            if request is None:
                rows.append({"corpus": name, "text": text, "request": None,
                             "dropped": [f"{k}:{v:g}" for k, v in numbers]})
                continue
            held = carried(request)
            vol = _vol_multiple(text)
            if vol is not None:
                held.add(vol)
            if _VAR.search(text):
                # the VaR level is read from the text by `research._var_lines` when it answers
                held |= {float(m.group(1)) for m in _VAR_LEVEL.finditer(text)}
            said = " ".join(request.notes)
            # A number the answer names in its stated assumptions ("$5,000 was stated but not
            # what the whole book is worth") was read and explained, not lost.
            dropped = [f"{k}:{v:g}" for k, v in numbers if not _accounted(k, v, held)
                       and f"{v:,.0f}" not in said and f"{v:g}%" not in said]
            rows.append({"corpus": name, "text": text, "kind": str(request.kind),
                         "dropped": dropped})
    flagged = [r for r in rows if r["dropped"]]
    report = {"questions_with_numbers": len(rows), "with_a_dropped_number": len(flagged),
              "note": "a detector: every flag is listed for reading; some dropped numbers are "
                      "rightly ignored",
              "flagged": flagged}
    REPORT.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> int:  # pragma: no cover - CLI
    report = run()
    print(f"{report['with_a_dropped_number']} of {report['questions_with_numbers']} questions "
          f"with a number lost at least one")
    for row in report["flagged"]:
        print(f"  [{row.get('kind')}] {row['text'][:90]} -> dropped {row['dropped']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
