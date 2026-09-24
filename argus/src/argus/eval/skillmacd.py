"""Bitget's technical-analysis Skill MACD, checked against MACD recomputed from Bitget's candles.

`argus.market.skills` already cross-checks the Skill's RSI (6 of 6 agree within 2.2 points). The
same check on MACD found the Skill returning its signal line in the ``histogram`` field and its
histogram in the ``signal`` field, and flagging crosses from the swapped pair. This module makes
that finding reproducible: for each symbol it records the Skill's raw payload, ARGUS's DIF / DEA /
HIST from Bitget 4h candles, and a verdict —

* ``swapped`` — the Skill's ``histogram`` field sits on our signal line;
* ``straight`` — its ``signal`` field does (the Skill has been fixed, or never had the fault here);
* ``unchecked`` — one side could not be read.

The DIF (MACD line) itself is compared too, as a control: if DIF also disagreed, the two sides
would simply be on different bars and no field conclusion could be drawn.

Run: ``python -m argus.eval.skillmacd`` → ``data/skill_macd_check.json``.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from argus.market.bitget import RTOKEN_SYMBOLS
from argus.market.evidence import BitgetSkillSource
from argus.market.skills import indicators, macd_fields

OUT = Path(__file__).resolve().parents[3] / "data" / "skill_macd_check.json"
SYMBOLS: tuple[str, ...] = (*RTOKEN_SYMBOLS, "BTCUSDT", "ETHUSDT")
DIF_AGREEMENT = 0.02
"""Relative gap within which the two MACD lines count as the same reading (bar alignment and the
upstream feed differ slightly; measured gaps were under 0.5%)."""


def check(symbol: str, client: Any) -> dict[str, Any]:
    payload, status = client.call("technical_analysis", {"action": "macd", "symbol": symbol},
                                  timeout=30)
    mine = indicators(symbol)
    row: dict[str, Any] = {"symbol": symbol, "skill_status": status, "skill": payload,
                           "ours": None if mine is None else {
                               k: mine[k] for k in ("dif", "dea", "histogram", "cross")}}
    if not isinstance(payload, dict) or payload.get("macd") is None or mine is None:
        row["verdict"] = "unchecked"
        return row
    dif_gap = abs(float(payload["macd"]) - float(mine["dif"])) / max(1e-9, abs(float(mine["dif"])))
    row["dif_relative_gap"] = round(dif_gap, 5)
    checked = macd_fields(payload, float(mine["dea"]))
    if checked is None or dif_gap > DIF_AGREEMENT:
        row["verdict"] = "unchecked"
        row["note"] = "MACD lines disagree, so the fields cannot be compared"
        return row
    row["verdict"] = "swapped" if checked[2] else "straight"
    row["skill_cross"] = payload.get("cross")
    # Read conservatively, as a state: "golden_cross" claims the MACD line is above its signal
    # line, "death_cross" that it is below. It is wrong only when the recomputed lines say the
    # opposite — not merely because no cross happened on the latest bar.
    claimed = str(payload.get("cross") or "")
    bullish = float(mine["histogram"]) > 0
    row["cross_wrong"] = ((claimed == "golden_cross" and not bullish)
                          or (claimed == "death_cross" and bullish))
    return row


def main(argv: list[str] | None = None) -> int:
    client = BitgetSkillSource()
    rows = [check(symbol, client) for symbol in SYMBOLS]
    verdicts = [r["verdict"] for r in rows]
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "what": ("Bitget technical-analysis Skill MACD(12,26,9) vs MACD recomputed by ARGUS "
                 "from Bitget 4H candles (argus.market.skills.indicators)"),
        "definition": "bitget-signal/skills/technical-analysis/references/indicators.md:45-53",
        "symbols": len(rows),
        "swapped": verdicts.count("swapped"),
        "straight": verdicts.count("straight"),
        "unchecked": verdicts.count("unchecked"),
        "cross_flag_contradicts_lines": sum(1 for r in rows if r.get("cross_wrong")),
        "rows": rows,
    }
    OUT.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8",
                   newline="\n")
    print(f"{report['swapped']} swapped, {report['straight']} straight, {report['unchecked']} "
          f"unchecked of {len(rows)}; cross flag contradicts the lines on "
          f"{report['cross_flag_contradicts_lines']} -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
