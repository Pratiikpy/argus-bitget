# ARGUS

**An evidence-driven autonomous trading governor for 24/7 tokenized equity markets.**

A tokenized US equity trades continuously. Its underlying does not — for roughly **65.5 hours a
week** the anchor market is shut and price discovery attenuates rather than stops. Most trading
agents ask *what should I trade?* ARGUS asks a harder question: **should this decision-maker be
trusted with capital right now?**

Built for the Bitget AI Base Camp / Genesis Hackathon Season 2. **The engine covers all three
tracks and all eighteen sub-themes; exactly one track is entered.** Track 3 (AI Trading Desk)
is filed. Track 2 (Agentic Trading) is **not**, and the reason is the thesis rather than the
deadline: it is 50% quantitative, scored on paper-trading Sharpe, drawdown and win rate, and
this desk has refused every decision it has made. Filing it would mean loosening the risk
layer to manufacture a track record — breaking the exact property the system exists to hold.

---

## Start here

**Try it first: the live research console — https://deploy-topaz-seven-64.vercel.app**

**Or watch one whole research task run live, question to actionable insight — https://deploy-topaz-seven-64.vercel.app/research** (seven engines, about three seconds; change the name, size or book on the page).

Ask it in plain English. *"I hold 50% NVDA, 50% AAPL — what does adding 20% TSLA do to my risk?"*
· *"What if the Nasdaq drops 10%? I hold 40% MSFT, 30% META, 30% GOOGL"* · *"Is TSLA riskier than
NVDA?"* · *"How should I split a $50k order in NVDA?"* · *"Where is NVDA trading right now?"* ·
*"Why did you pass on NVDA?"* Every figure comes from the desk's own engines on live Bitget data and
names its source; the language model only works out what you asked and never writes a number. Save
your holdings (and your own risk budget) in the "My book" field and every answer uses them.

| | |
|---|---|
| **The code** | [`argus/`](argus/) — package, tests, runnable studies |
| **Build status** | [`argus/STATUS.md`](argus/STATUS.md) — what runs, what is blocked, and on what |
| **Design** | [`ARGUS-ARCHITECTURE.md`](ARGUS-ARCHITECTURE.md) — three products on one engine, six levels |
| **Requirements + evidence** | [`ARGUS-MASTER-PRD.md`](ARGUS-MASTER-PRD.md) — the original build plan (12 Sep), with every measurement and retraction since; the entry itself is Track 3 only |
| **How to check it** | [`argus/VERIFY.md`](argus/VERIFY.md) — commands that reproduce each headline claim |

```bash
cd argus
pip install -e ".[dev]"
python -m argus.status          # module + sub-theme coverage, resolved by import
pytest -q                       # 6,542 tests collected
```

Nothing above needs a credential. The full run took 2h15m from a fresh clone on 2026-09-24 (the rival comparisons re-fetch live Bitget data, and one comparison installs its locked Node packages on first use); tests that need a rival's source cloned beside the repository skip and say which. `ARGUS_BLOCK_NETWORK=1` refuses every outbound connection, so a test that reaches the network shows itself.

---

## What is actually verified

| | |
|---|---|
| Tests | **6,542 tests collected** — run `pytest -q` |
| Type checking | **`mypy --strict` clean on 308 source files** |
| Lint | `ruff` clean |
| Module health | **131/131 modules importable**, checked by `python -m argus.status` |
| Sub-theme coverage | **18/18 sub-themes**, resolved by import at runtime — not claimed in prose |
| Data artefacts | **46 cited across these documents**, every one reproducible from a single named command, with a test that fails if a document cites an artefact nothing writes |
| Quoted figures | **87 numbers in these documents are re-checked against their artefacts** by `python -m argus.eval.docclaims --tests`, which exits non-zero if any has drifted |

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
   does not exist is a *settled* trade: every decision on the paper ledger is a refusal, so
   Sharpe, max drawdown and win rate are undefined and the code prints the reason for each rather
   than a zero. Two earlier ledger rows appeared to be filled trades; they recorded positions the
   risk layer had refused, and they are void — kept in the chain, excluded from every figure, and
   documented in `argus/src/argus/paper/corrections.py`.
2. **No certified alpha exists.** 0 of 8 factors and 0 of 12 strategies cleared the deflation gate.
   That is the finding, and it is reported as one.

**The obvious attack, and the measurement that answers it.** *"Hundreds of decisions, zero trades — it
has demonstrated nothing."* Every refusal carries a stated direction, hashed with the decision before
the outcome exists, so the counterfactual was committed to rather than reconstructed. Graded against
what actually happened (`python -m argus.eval.refusal`): at the ~2h horizon **169 of 289 directional
calls were right — 58.5%, 95% Wilson interval 52.7–64.0%**, which excludes a coin flip but only
barely, and the overnight horizon stands at *48.3%* (42 of 87) — not distinguishable from a coin
flip. The decisive figure is the
other one: **the median decision forgave -6.7bps of net edge after the 12bps round trip.** The
typical refusal was not caution costing money — it was the trade being unprofitable. An empty ledger
is the result here, not the gap.


**What the register says about itself, after it started checking.** ARGUS keeps a capability standing register with four states and thirteen conditions for OWNED. Until 2026-09-20 its audit checked that a file existed and that a test function's name appeared inside it — so *"same-input comparison run"*, *"out-of-sample test"* and *"ablation"* were each satisfied by a filename, and 23 of 24 entries carried the top grade. It now opens the artefacts. **27 of 31 capabilities are OWNED, 3 are TIED, and 0 are LOST.** Seven were demoted the first time the artefact-opening audit ran (2026-09-20), and every one has since been re-earned by making its artefact carry the evidence rather than by editing its state, because they claimed a condition their own artefact recorded nothing about. By 2026-09-20 three capabilities were genuinely LOST to a named specialist; within 48 hours every one moved to TIED, and none of the three ties is a rounded-up loss — each is published with the losing half still named as a loss where one exists. Portfolio allocation moved first: Riskfolio-Lib's real NCO beat ARGUS's HRP decisively on a 24-window walk-forward (p=3.6e-05), so ARGUS's own NCO was built from Riskfolio's real source — real Ward linkage, a real active-set minimum-variance solver, the real two-difference gap statistic for cluster count — verified to reproduce Riskfolio's real NCO to 1.3e-05 on the same book, then wired into the SAME live walk-forward test and re-run fresh: 8.203bps both, ratio 1.0000. A tie, not a win, and published as exactly that. LUI intent routing moved next: dev-half CV across 10 fold-split seeds found a linear-SVM classifier head beats the multinomial logistic head this shipped with on every seed (mean 76.71% vs 72.28%), so the model was rebuilt, its abstention threshold re-derived by the same rule that set the original, and re-measured exactly once on the same sealed corpus the old model was scored on. Rasa's DIET classifier is still numerically ahead on raw accuracy (81.91% vs 77.82% at that point), but the gap was no longer statistically significant (McNemar p=0.0576, down from a clearly significant p=0.0014 before the rebuild) — not proven equal, a real but fragile tie. A second rebuild the same day (a RIVAL LENS pass) tested a lever the first rebuild's own root-cause note had named and left untried: the out-of-scope training negatives were skewed 60 Chinese to 40 English, opposite the roughly-balanced in-scope pool. Dev-CV confirmed the mechanism directly (held-out Chinese in-scope rows wrongly refused as out-of-scope, 14.2% to 8.4%, McNemar p<0.0001) before touching production code; rebalancing the negatives took sealed accuracy to 79.52% and widened the margin against Rasa to a comfortable p=0.2649 — still a tie, no longer a fragile one. ARGUS's separate, significant win on out-of-scope refusal got stronger in the first rebuild and held steady through the second (92.86% vs Rasa's 42.86%, p=0.0156, was p=0.03125). Regime-boundary detection moved last, the same day it was sharpened: FLUSS, ARGUS's original segmenter, still loses to ruptures decisively (F1 0.443 vs 0.975, p=1.5e-25) — unchanged, and stated as a loss, not softened. What moved the *capability* is a second, different ARGUS tool: `ruptures/detection/dynp.py` and `costs/costl2.py` (BSD-2-Clause) were read in full to understand *why* FLUSS loses, and an exact L2 dynamic-program segmenter (`desk/regime.py::exact_partition`) was built, verified to reproduce ruptures' own real `Dynp` exactly on 120/120 trials and to match `KernelCPD`'s rbf-kernel result — the actual rival — on 10/10, then wired into the identical 100-trial synthetic ground truth that measured FLUSS's loss: 1 win, 0 losses, 99 ties, mean F1 0.98 against ruptures' 0.975 — nominally ahead, not significantly so on one discordant trial, reported as a tie. Three of the thirteen conditions — whether the *best* implementation and the *best* method were studied, and whether a specialist still beats us — are judgements about a field rather than properties of a file, so they are reported as **attested** and name where somebody looked, instead of being dressed up as machine-checked (`python -m argus.eval.standing`).

---

## Licence

MIT. Components ported from other projects cite their source file and line in the docstring;
everything copied is MIT-licensed, and anything under a restrictive licence was rebuilt from the
described behaviour rather than copied.
