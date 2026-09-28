"""Write the exact 619-question probe list ``eval/pit_rivals.py`` asks, for rivals that take a
real as-of cutoff of their own.

Unlike every other file in this directory, this one *does* import ARGUS (``argus.eval.pit_rivals``,
``argus.market.pit``) and runs under ARGUS's own venv, not an isolated rival venv — it needs
:func:`argus.eval.pit_rivals.probes_for`, the same probe generator ``run()`` uses, so the probes fed
to every arm are provably identical rather than a second, hand-copied implementation that could
drift from the real one.

Qlib PIT (``qlib/utils/__init__.py::read_period_data``) and edgartools (``FactQuery.as_of``) both
accept a real point-in-time cutoff, unlike Vibe-Trading/OpenBB/LangAlpha/OpenAlice, whose tools take
no as-of parameter at all (see ``eval/pit_rivals.py``'s module docstring). That lets their runners
call the rival's own as-of mechanism once per probe instead of dumping a raw series for
``eval/pit_rivals.py`` to gate externally. This script is the shared probe list both runners read.

Usage (ARGUS venv, ``PYTHONPATH=src``)::

    python scripts/pit_runners/dump_pit_probes.py --snapshot DIR --out probes.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from argus.eval.pit_rivals import TICKERS, load_facts, probes_for


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    snapshot = Path(args.snapshot)
    out: dict[str, Any] = {"concept": "revenue", "tickers": {}}
    total = 0
    for ticker in TICKERS:
        facts = load_facts(ticker, "revenue", snapshot)
        probes = probes_for(ticker, facts)
        out["tickers"][ticker] = [
            {"accn": p.accn, "side": p.side, "as_of": p.as_of.isoformat()} for p in probes
        ]
        total += len(probes)
    out["total"] = total
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8", newline="\n")
    print(f"wrote {total} probes across {len(TICKERS)} tickers to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
