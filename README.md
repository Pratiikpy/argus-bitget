# ARGUS

**An evidence-driven autonomous trading governor for 24/7 tokenized equity markets.**

A tokenized US equity trades continuously. Its underlying does not — for roughly **65.5 hours a
week** the anchor market is shut and price discovery attenuates rather than stops. Most trading
agents ask *what should I trade?* ARGUS asks a harder question: **should this decision-maker be
trusted with capital right now?**

Built for the Bitget AI Base Camp / Genesis Hackathon Season 2, across all three tracks.

---

## Start here

| | |
|---|---|
| **The code** | [`argus/`](argus/) — package, tests, runnable studies |
| **Build status** | [`argus/STATUS.md`](argus/STATUS.md) — what runs, what is blocked, and on what |
| **Design** | [`ARGUS-ARCHITECTURE.md`](ARGUS-ARCHITECTURE.md) — three products on one engine, six levels |
| **Requirements + evidence** | [`ARGUS-MASTER-PRD.md`](ARGUS-MASTER-PRD.md) — including every measurement and every retraction |

```bash
cd argus
pip install -e ".[dev]"
python -m argus.status          # module + sub-theme coverage, resolved by import
pytest -q                       # 490 passed, 11 skipped
```

Nothing above needs a credential.

---

## What is actually verified

| | |
|---|---|
| Tests | **490 passing**, 11 skipped |
| Type checking | `mypy --strict` clean on **32 modules** |
| Lint | `ruff` clean |
| Sub-theme coverage | **18/18**, resolved by import at runtime — not claimed in prose |
| Data artefacts | 13 on disk, each reproducible from a single command |

The measurements behind the design are in the PRD, including the ones that went against us:

- **Price discovery attenuates ~7x during closures — it does not stop.** Measured across 12 rTokens.
- **The weekend effect FAILS its own validation.** 57.7% full-sample continuation splits into 48.7%
  train and 66.7% out-of-sample. The full-sample number is the average of "nothing" and "strong",
  which makes it meaningless. Reported as a negative rather than shipped as an edge.
- **The 3–5x off-hours spread widening we originally claimed is falsified.** Our own claim, retracted
  in §3.4 rather than quietly dropped.
- **Deliberation is a first-order cost.** 40 seconds of model reasoning off-hours prices at
  **15.2bps of expected adverse move against a 12bps round-trip fee** — thinking about the trade
  costs more than the trade. We expected this to be negligible and it is not.
- **Network latency to the venue is not the constraint.** 30/30 probes, 267–353ms round trip,
  which prices at 0.48bps.

---

## The properties the code enforces

These are structural, not conventions — each is enforced by a type or a constructor, and each has a
test asserting the defect cannot be reintroduced.

- **A zero-fee backtest is unconstructible.** `CostModel` raises rather than defaulting to free.
- **Omitting `as_of` is a `TypeError`.** Point-in-time evidence cannot be queried carelessly.
- **The risk layer may only reduce.** It can never create, flip, or increase a position — and the
  Autonomy Proof records that it did not.
- **A maker fee requires a book.** Passive execution is simulated through a queue model ported from
  hftbacktest; asking for a maker rate without depth data raises.
- **Search cannot see evaluation.** `ProposerContext` has no field capable of holding a score.
- **The paper ledger is hash-chained**, and a tampered chain is refused rather than scored.

---

## Status, honestly

**Implemented, not owned.** The PRD's acceptance matrix (§7.2a) records which of ten proof gates
each sub-theme has passed, and several remain open. Two things are unproven rather than built:

1. **No venue has accepted an order.** The signed path is verified three ways against Bitget's own
   SDK source and probed live to credential lookup, but signature *acceptance* is untested because
   no Demo API key exists yet.
2. **No certified alpha exists.** 0 of 8 factors and 0 of 12 strategies cleared the deflation gate.
   That is the finding, and it is reported as one.

---

## Licence

MIT. Components ported from other projects cite their source file and line in the docstring;
everything copied is MIT-licensed, and anything under a restrictive licence was rebuilt from the
described behaviour rather than copied.
