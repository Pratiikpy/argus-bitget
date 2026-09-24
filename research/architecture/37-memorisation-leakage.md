# Memorisation leakage — and the control that stopped a false clean bill of health

## 1. The hole in every historical evaluation

`eval/shadow.py` grades the desk's directional lean against the move that followed.
`research/*_study.py` sweeps historical windows. `eval/bakeoff.py` compares strategies over history.
All three ask a language model about a period that has already happened, and none of them could say
whether the model had simply **read the answer** during training.

## 2. The reference

barj28, *A Validated Instrument for Memorization Leakage in LLM Trading Evaluation* (MIT, DOI
10.5281/zenodo.20844335), cloned to `research/repos-themed/barj28~llm-leakage-instrument`.

Its black-box probe, `src/vr_C_api.py:124-137`, copied exactly:

- multiplicative distractors — **wide** `{0.5, 0.7, 1.4, 2.0}` (recall vs a random guess) and
  **tight** `{0.85, 0.93, 1.08, 1.18}` (precise recall vs an order-of-magnitude estimate);
- a **deterministic per-fact shuffle** (`vr_C_api.py:130`), so a rerun measures the model and not
  the shuffle;
- single-letter answer, capacity is `P(prefers the true value)` against `1/len(options)`.

And its validation cascade, which is the part most implementations skip: a detector is credible only
if it is **specific** (silent where memorisation is impossible) *and* **sensitive** (it fires where
recall is known to be present). Their positive control is a LoRA that injects known values
(`armB_*`).

## 3. What is ours

**The facts.** barj28 ships a revenue panel; we build probes from `market/fundamentals.py`, which
reads what companies actually filed with the SEC, point-in-time and restatement-resolved. The ground
truth is checkable against primary sources we already hold, and the probe set regenerates on demand
rather than being trusted as a frozen CSV. 42 tight probes across seven mega-caps, publication dates
from 2010 to August 2026.

**The axis is the filing date.** A model can only have learned a number after it was published, so a
Q2-2026 fact filed in August 2026 is post-cutoff for anything trained before then, whatever its
fiscal label says. Bucketing by fiscal period would smear the boundary the instrument exists to find.

**The sensitivity control.** We cannot fine-tune a hosted endpoint, so recall is made certain the
only other way available: **the value is printed in the prompt**. A model that still cannot pick the
right letter has not demonstrated an absence of memorisation — it has demonstrated that the
instrument cannot measure it.

**A stricter parser than the reference.** `vr_C_api.py:140-142` searches for `[A-E]` anywhere in the
reply, which matches the "C" inside *"I cannot help with that"* and scores a refusal as a confident
answer. Ours requires a standalone letter. Every genuine answer is a bare letter, so the stricter
rule loses nothing and stops a refusal from inflating capacity.

## 4. The control earned its place on the first run

**Run 1 — control 2/12 (17%) with the answer printed in the prompt.** An impossible score for any
model that can read, and the only reason the defect surfaced: `parse_letter` was being handed
`str(completion)`, and `QwenClient.complete` returns a `Completion` whose repr carries both
`content` and `reasoning`. The regex was scanning the **chain of thought** and returning whichever
letter it mentioned first. Probes scored 7/42 — and without the control that would have shipped as
*"no memorisation detected"*, a clean bill of health for every historical evaluation in the project.

**Run 2 — control 0/12.** A second harness bug, caught the same way: an editing pass wrote a literal
backspace byte where the regex needed `\b`, so the pattern matched nothing at all.

**Run 3 — control 12/12 (100%).** Instrument sensitive. Only then is the measurement interpretable.

Both failures were in the measuring apparatus, not the model, and both were invisible in the probe
results alone — a broken instrument and an honest null look identical.

## 5. The finding

`python -m argus.eval.leakage --kind tight --per-ticker 6 --control 12`:

| published | probes | hits | capacity | chance | p |
|---|---|---|---|---|---|
| 2010H2 | 4 | 4 | 100% | 20% | 0.0016 (too few to count) |
| 2011H1 | 2 | 2 | 100% | 20% | 0.0400 (too few) |
| 2018H1 | 2 | 2 | 100% | 20% | 0.0400 (too few) |
| 2018H2 | 10 | 9 | **90%** | 20% | 0.0000 |
| 2025H2 | 8 | 5 | **62%** | 20% | 0.0104 |
| 2026H1 | 8 | 4 | 50% | 20% | 0.0563 |
| 2026H2 | 8 | 6 | **75%** | 20% | 0.0012 |

**32/42 correct (76%) against a 20% chance level, p < 0.0001.** Recall is above chance in every
period with enough probes to count, **including facts filed in August 2026**. The instrument located
**no upper boundary**: the model's knowledge of these fundamentals extends at least to the most
recent filing in our data.

Note this is the *tight* probe — log-matched distractors within ±18% of the true value. The model is
not estimating the order of magnitude; it is recalling the number.

**What it means for ARGUS.** Any evaluation over a historical window in that range is contaminated:
the model has demonstrably read facts published inside it, so a good result there may be recall
rather than judgement. That includes every backtest and every shadow-grading window that predates
the live paper record.

**What it does not mean.** That the recall *becomes* trading skill. barj28's headline empirical
finding is precisely that capacity and realisation come apart — models memorise fundamentals and it
does not transmit. `LeakageReport.verdict` says so explicitly rather than letting a reader infer that
a contaminated window is a fraudulent one, and the transmission question belongs to
`eval/shadow.py`, which grades against the live forward record.

## 6. Not done

- **The free-generation probe** (`vr_C_api` third row: no options, the model writes the number,
  scored within 5/10/20%). Recognition and recall are different capacities, and the tight MCQ
  measures only the first. Worth adding; not yet built.
- **Multi-model comparison.** barj28's cutoff-boundedness result is a within-model t-statistic over
  19 models. We have one model and one endpoint, so no boundary can be located by comparison — only
  by finding a period where recall collapses, which this probe set does not reach.
- **Wiring the gate into the studies.** `LeakageReport.overlaps_recall(start, end)` exists and is
  tested; no study calls it yet. The honest position is that the measurement is published and the
  enforcement is not, rather than implying a gate that does not fire.
