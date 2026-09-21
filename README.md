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
pytest -q                       # 5,946 tests collected
```

Nothing above needs a credential.

---

## What is actually verified

| | |
|---|---|
| Tests | **5,946 tests collected**, 33 skipped — run `pytest -q` |
| Type checking | **`mypy --strict` clean on 287 source files** |
| Lint | `ruff` clean |
| Module health | **131/131 modules importable**, checked by `python -m argus.status` |
| Sub-theme coverage | **18/18 sub-themes**, resolved by import at runtime — not claimed in prose |
| Data artefacts | **46 cited across these documents**, every one reproducible from a single named command, with a test that fails if a document cites an artefact nothing writes |
| Quoted figures | **71 numbers in these documents are re-checked against their artefacts** by `python -m argus.eval.docclaims --tests`, which exits non-zero if any has drifted |

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

1. **No position has settled.** The venue path itself is proven — the Bitget signature is accepted
   and a real demo order round-tripped, id `1482642956228374529`, productType `SUSDT-FUTURES`. What
   does not exist is a *settled* trade: all 447 decisions on the paper ledger are refusals, so
   Sharpe, max drawdown and win rate are undefined and the code prints the reason for each rather
   than a zero. Two earlier ledger rows appeared to be filled trades; they recorded positions the
   risk layer had refused, and they are void — kept in the chain, excluded from every figure, and
   documented in `argus/src/argus/paper/corrections.py`.
2. **No certified alpha exists.** 0 of 8 factors and 0 of 12 strategies cleared the deflation gate.
   That is the finding, and it is reported as one.

**The obvious attack, and the measurement that answers it.** *"447 decisions, zero trades — it has
demonstrated nothing."* Every refusal carries a stated direction, hashed with the decision before
the outcome exists, so the counterfactual was committed to rather than reconstructed. Graded against
what actually happened (`python -m argus.eval.refusal`): at the ~2h horizon **92 of 158 directional
calls were right — 58.2%, 95% Wilson interval 50.4–65.6%**, which excludes a coin flip but only
barely, and the overnight horizon at 51.2% does not exclude one at all. The decisive figure is the
other one: **the median decision forgave −7.9bps of net edge after the 12bps round trip.** The
typical refusal was not caution costing money — it was the trade being unprofitable. An empty ledger
is the result here, not the gap.


**What the register says about itself, after it started checking.** ARGUS keeps a capability standing register with four states and thirteen conditions for OWNED. Until 2026-09-20 its audit checked that a file existed and that a test function's name appeared inside it — so *"same-input comparison run"*, *"out-of-sample test"* and *"ablation"* were each satisfied by a filename, and 23 of 24 entries carried the top grade. It now opens the artefacts. **17 of 24 capabilities are OWNED**; seven were demoted the first time it ran, and one has since been re-earned by making its artefact carry the evidence rather than by editing its state, because they claimed a condition their own artefact records nothing about. Three of the thirteen — whether the *best* implementation and the *best* method were studied, and whether a specialist still beats us — are judgements about a field rather than properties of a file, so they are reported as **attested** and name where somebody looked, instead of being dressed up as machine-checked (`python -m argus.eval.standing`).

---

## Licence

MIT. Components ported from other projects cite their source file and line in the docstring;
everything copied is MIT-licensed, and anything under a restrictive licence was rebuilt from the
described behaviour rather than copied.
