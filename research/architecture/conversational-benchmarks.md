# Benchmarking a natural-language agent: what the best six measure, and what none of them do

**Question.** For a console a human asks questions of: how do the strongest benchmarks decide
whether a question was *understood* and whether the answer was *right*?

**Three answers that matter, and all three are gaps.** Across six benchmarks, **none** scores a
correct refusal as a success, **none** tests paraphrase robustness, and **none** tests multilingual
parity. Every citation below was opened.

---

## 1. `sierra-research/tau-bench` — the soundest task metric

`research/repos/tau-bench/tau_bench/run.py:180-203`.

`pass^k` from arXiv 2406.12045, computed per task from the number of successful trials:

```python
for k in range(1, num_trials + 1):
    for c in c_per_task_id.values():
        sum_task_pass_hat_k += comb(c, k) / comb(num_trials, k)
```

Success is a binary reward with a float-tolerant comparison (`(1 - 1e-6) <= reward <= (1 + 1e-6)`).
Tasks are pydantic objects carrying `user_id`, expected `actions` and `outputs`, and grading is
**grounded**: the environment compares the agent's tool calls against ground truth rather than
comparing strings.

**What we take:** running each case several times and reporting the distribution rather than one
draw is the right instinct, and it is the same lesson our own search sweep learned the hard way.
**What it lacks:** two reward states only, no robustness testing, no notion of a question that
*should* be refused — an agent that correctly declines an impossible task scores zero.

## 2. `czyssrs/ConvFinQA` — the soundest numerical grading

`research/repos/ConvFinQA/code/utils/general_utils.py:502-503`:

```python
exe_acc = float(exe_correct) / len(data)
prog_acc = float(prog_correct) / len(data)
```

Two metrics kept apart: **execution accuracy** (the program's numerical result matches the gold
result) and **program accuracy** (the token sequence matches). Separating them is the good idea —
a right answer from a wrong program is a different thing from both a right and a wrong answer, and
collapsing them hides which you have.

**What it lacks:** every question is assumed answerable; a wrong answer and an honest "I cannot
compute this from the table" score identically at zero.

## 3. `finbrain-lab-hkustgz/AlphaForgeBench`

`AlphaForgeBench/metrics.py:36-98`. `pass@k`, `pass@1` and a `syntax_pass_rate`, with backtest
metrics (Sharpe, drawdown, Sortino, Calmar) averaged **over passed samples only** and stratified by
difficulty. Stratification is worth copying when a corpus has natural difficulty bands.

**What it lacks:** a syntax-invalid sample and a refusal are the same event; no robustness or
language axis.

## 4. `microsoft/RD-Agent`

`rdagent/components/benchmark/eval_method.py:64-222`. A multi-evaluator pipeline — column, row
count, index, value ratio and correlation evaluators each returning `(feedback, metric)` — run over
multiple rounds. The richest *signal* of the six: one number per run would hide which property
broke.

**Nearest to refusal handling, and still not it:** execution exceptions are caught and marked as
errors, which records that something failed rather than that declining was correct.

## 5. `ulab-uiuc/live-trade-bench`

`live_trade_bench/backtest/backtest_runner.py:59-76`: a single `return_percentage`. Grounded in
simulated fills, and one number cannot separate strategy from luck.

## 6. `czyssrs/FinQA`

Same execution/program methodology as ConvFinQA with a retrieval stage in front.

---

## What ARGUS does instead

`argus/src/argus/eval/luibench.py`. Six metrics, three of which have no precedent in the six:

1. **Refusal scored in both directions.** A question that must be refused (a live quote, an
   instruction to trade, an instrument off the venue) counts as a success *when refused*, and an
   answerable question counts as a failure *when refused*. `over_refusals` names the second by
   quotation in the verdict, because a console that refuses everything scores perfectly on the half
   of this metric that every prior benchmark would have measured alone.

2. **Paraphrase consistency**, measured within a family rather than against the label — a family
   that agrees on the *wrong* intent is a mis-specified pattern, and one that scatters is a missing
   one. These need different fixes, so they are different numbers.

3. **Bilingual parity**: the Chinese twin of an English question must reach the same intent. The
   first run found the colloquial Chinese for "how did we do this week" reaching UNKNOWN while its
   English twin reached PERFORMANCE — a defect that had been shipped and invisible, on a
   Chinese-language-first competition.

Plus **grounding** in ConvFinQA's spirit: every non-refused answer must cite at least one source,
and a refusal is excluded rather than scored zero.

**On fitting.** tau-bench has a train/dev/test split; we cannot honestly claim one, because the same
hand wrote the console and the cases. The substitute is pre-registration — cases written and
committed before the first run, the first score recorded beside the final one in
`ARGUS-MASTER-PLAN.md`, and a test asserting that **more than 85% of cases appear nowhere in the
classifier's source**, so the corpus cannot quietly become a restatement of the unit tests. The
verdict itself refuses to overclaim: on a perfect score it prints *"the same hand wrote the console
and the cases, so read this as the floor it clears, never as a fluency ceiling"*.

**Where they are ahead.** tau-bench's `pass^k` is a better statistic than any single-run rate, and
it applies the moment a component is non-deterministic. Our classifier is a deterministic regex
cascade, so repeated trials would return identical results and `pass^k` would collapse to `pass@1` —
noted rather than implemented, and the moment an LLM enters the routing path it is the first thing
to add.

**Licences.** tau-bench MIT, ConvFinQA MIT, AlphaForgeBench unlicensed (study only), RD-Agent MIT,
live-trade-bench MIT. Nothing was copied; the metrics above are our own definitions.
