# ARGUS

**An evidence-driven autonomous trading governor for 24/7 tokenized equity markets.**

Most trading agents ask *what should I trade?* ARGUS asks a harder question:
**should this decision-maker be trusted with capital right now?**

Design: [`../ARGUS-ARCHITECTURE.md`](../ARGUS-ARCHITECTURE.md) ·
Requirements and evidence: [`../ARGUS-MASTER-PRD.md`](../ARGUS-MASTER-PRD.md)

---

## Status

**490 tests passing · 11 skipped · ruff clean · mypy `--strict` clean on 32 modules · live loop verified.**

All eighteen sub-themes across the three tracks resolve by import and carry a test file — checked
at runtime by `python -m argus.status`, not asserted here.

| Level | Module | State |
|---|---|---|
| **1 — Truth** | `argus.truth` | two-clock PIT, session state machine, as-of evidence store |
| **1 — Cost** | `argus.cost` | constructed-mandatory model; zero-fee is unconstructible |
| **2 — Decide** | `argus.decision` · `argus.agents` | verdict vocabulary, Constitution asymmetry, Meta-PM |
| **3 — Risk** | `argus.risk` | hedgeability surface, Risk Neutralisation Efficiency |
| **4 — Execute** | `argus.execution` | 14-state order machine; queue-position fills, passive execution, Almgren-Chriss trajectories, measured latency |
| **5 — Learn** | `argus.eval` · `argus.paper` | Observatory, hash-chained paper ledger, scorecard, Challenge Mode |
| **6 — Authority** | `argus.desk` · `argus.research` | research workbench, factor lab, portfolio copilot |
| **Proof** | `argus.proof` | Autonomy Proof, Execution Proof |

Run the end-to-end demo against the live model:

```bash
set -a && . ../.secrets/qwen.env && set +a
python demo_sleeping_anchor.py
```

Nothing here is claimed as finished. The acceptance matrix in the PRD (§7.2a) records which of the
ten proof gates each sub-theme has passed; G4–G10 remain open. The honest phrase is
**implemented, not owned**, and it stays that way until a cell turns green because an artefact
exists on disk.

---

## What the live loop actually does

One verified run, Sunday 03:00 ET, NYSE shut for another 30.5 hours, no hedge placeable:

| Step | Result |
|---|---|
| **LLM decides** | `REDUCE sell 100` of a 200 position, confidence 0.62, three invalidation conditions |
| **Constitution narrows** | `RESIZE → 50`, binding constraint `unhedgeable_gap` |
| **LLM responds** | accepts 50, rewrites its thesis for the smaller size |
| **Autonomy Proof** | constitution intervened ✓ · constitution only reduced ✓ · LLM genuinely decided ✓ |

The model separated the verified 8-K from an unverified viral claim, and wrote an invalidation
condition stating that if the larger claim were independently confirmed, a *bigger* reduction would
be warranted. That is the behaviour the architecture is built to require and to prove.

---

## The problem this package solves

A tokenized US equity trades continuously. Its underlying does not — roughly **65.5 hours a week**
the anchor market is shut. During that window a fact can be **knowable** (and tradeable on the
token) while being **unpriceable** (no genuine price discovery, no hedge placeable).

Every point-in-time implementation found across our 922-source research corpus assumes a **single**
market clock and therefore cannot express that state. The practical consequence is a backtest that
silently assumes a hedge was available at 3am on a Sunday.

`argus.truth` keeps the two clocks separate and makes the distinction a typed question:

```python
from datetime import datetime, timedelta
from argus.truth.clocks import DualClock, ET

clock = DualClock()
sunday_3am = datetime(2026, 3, 8, 3, 0, tzinfo=ET)

clock.is_knowable(sunday_3am - timedelta(minutes=30), sunday_3am)  # True  — token clock
clock.is_priceable(sunday_3am)                                     # False — anchor clock
clock.state(sunday_3am).hours_to_next_discovery                    # ~30.5
```

## Point-in-time retrieval is enforced by the interface

```python
store.query(as_of=decision_time)   # the only way to read
store.query()                      # TypeError — there is no default and no "latest"
```

This is a direct response to two defects found by reading real systems' source:

| System | Defect | Location |
|---|---|---|
| FinMem | `temp_date_list` is populated and never checked before ranking — retrieval on date *t* can return memories from *t+100* | `memorydb.py:138-218` (populated at 169, 195) |
| FinAgent | unbounded `similarity_search`; separately, the environment computes `days_future = now + 14`, placing future state in the observation | `memory/basic_memory.py:62` |

Both systems *intended* point-in-time retrieval. They failed because forgetting the filter was
possible. Here it is not expressible.

Restatements are handled as the subtler case of the same problem: a query at `as_of` returns the
figure **as it stood then**, not the later restated value.

---

## Running the tests

```bash
pip install -e ".[dev]"
pytest              # everything
pytest -m leakage   # only the look-ahead suite
```

Every test in `tests/test_leakage.py` corresponds to a defect found in a real, widely cited system,
with the citation in the docstring. If one fails, ARGUS has acquired a bug that already ships
elsewhere — and the citation says where to look for the shape of it.

---

## Licence

MIT. Third-party components and their dispositions are recorded in
[`../research/architecture/_CONSOLIDATED-LEDGER.md`](../research/architecture/_CONSOLIDATED-LEDGER.md).
Nothing under GPL, LGPL, PolyForm Noncommercial, or an absent licence is vendored here.
