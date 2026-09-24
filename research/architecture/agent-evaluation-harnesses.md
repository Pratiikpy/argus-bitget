# Agent-evaluation harnesses — how the field measures the Track 2 Open Theme, at code level

**Written 2026-09-13.** The Track 2 Open Theme names five measurements: decision consistency,
risk-violation rate, stress behaviour, human-takeover rate, and incremental value over a fixed-rule
or Human+AI baseline. This is a teardown of what the best existing harnesses actually implement for
each, read at source, so that ARGUS-BENCH is built against the field rather than against a guess.

Trees searched: `research/repos-t2/` (25 Track-2 systems), `research/repos/`, `research/repos-new/`,
`research/repos-themed/`, `research/corpus/repos/`, `research/architecture/` (43 teardowns), and
`okx/trading/best-of-the-best/repos/` (114 repos) with its `notes/` and `platforms/`. Patterns
grepped case-insensitively: `takeover`, `human_in_the_loop`, `human-in-the-loop`, `escalat`,
`handoff`, `hand off`, `intervention`, `override`, `consistency`, `self-consistency`, `determinism`,
`pass@`, `pass^k`, `reliability`, `violation rate`, `constraint violation`, `incremental value`,
`uplift`, `counterfactual`, `baseline comparison`, `fixed rule`, `rule-based baseline`.

---

## Scoreboard

| Criterion | Best implementation found | ARGUS position |
|---|---|---|
| Decision consistency | `AlphaForgeBench/metrics.py:345-365` — Pass@k over k samples | **Ahead.** `eval/bakeoff.py` replays the same frame k times and measures *disagreement*; Pass@k measures only whether one sample worked |
| Risk-violation rate | `agent-backtest-lab/abl/data/firewall.py:38-60` — hard gate, every attempt logged | **Comparable.** `eval/riskaudit.py` + `data/risk_records.jsonl` record every intervention; the firewall's data-access denominator is a better *denominator* than ours |
| Stress behaviour | `agent-backtest-lab/abl/leakage/reward_hacking.py:57-172` — three signals with published thresholds | **Comparable.** `desk/stress.py` runs scenarios; their IS/OOS degradation triple is a different and complementary test |
| Human-takeover rate | **NOTHING. Absent in all 140+ repos.** | **Unbuilt everywhere, including here** |
| Incremental value | **NOTHING with a statistical test.** | **Ahead once `eval/incremental.py` runs** |

---

## 1. Decision consistency

### AlphaForgeBench — the best of a thin field

`AlphaForgeBench/benchmark.py:40-60` configures `num_samples: int = 5` and `temperature: float = 0.0`.
`metrics.py:345-365` aggregates:

```
Pass@k = 1 if any of k samples is syntactically valid AND passes backtest, else 0
Pass@1 = 1 if sample[0] passes, else 0
```

Published: Pass@k(T=0) = 0.9969 over 270 queries, Pass@1(T=0) = 0.9790.

**The defect, and it is fundamental to what the metric can say.** Pass@k asks whether *some* sample
worked. It cannot see disagreement. Three of five samples producing a wrong-but-runnable answer
still scores Pass@k = 1.0. It is a measure of **execution robustness**, not of decision consistency,
and using it for the latter is a category error.

### live-trade-bench — accepts the variance as a feature

`backend/llm_client.py:30-69` sets `temperature: float = 0.3`; `stock_system.py:48-75` runs each
agent once per day. There is no replay mechanism and no k-sample aggregation, so identical market
state on two days can produce different allocations and nothing records that it did.

### What ARGUS already does better

`eval/bakeoff.py` holds the market frame, evidence, prompt contract, cost model, Constitution and
temperature fixed and varies only the model, then **decides the same frame k times per model**. The
statistic is verdict disagreement under replay, not whether one run succeeded. Its own docstring
states the reason: *"A model that flips verdict on identical input at temperature 0 has not made a
decision; it has sampled one."*

**Taken:** nothing to take. **Rejected:** Pass@k as a consistency metric, for the reason above.
**Kept from the reading:** the discipline of fixing temperature and reporting it, which we do.

---

## 2. Risk-violation rate

### agent-backtest-lab — the cleanest gate in the corpus

`abl/data/firewall.py:38-60`:

```python
def check(self, ticker, requested_max_date, as_of, strict=True):
    allowed = requested_max_date <= as_of
    event = {"ticker": ticker, "requested_date": str(requested_max_date),
             "as_of": str(as_of), "allowed": allowed}
    self.audit_events.append(event)
    if strict and not allowed:
        raise FirewallViolation(...)
```

`abl/leakage/audit.py:10-20` summarises `total_accesses`, `violations`, `tickers_touched`;
`abl/leakage/detector.py` raises a `FIREWALL_VIOLATION` flag at critical severity.

**The rate is `count(not allowed) / count(all attempts)`, and the denominator is the thing worth
copying.** It is every *access attempt*, not every decision — which makes the rate a property of
the data layer rather than of the model's behaviour, and means a violation is caught at the point
of access rather than inferred afterwards.

### TraderHarness — gates that are not scored, and one that is bypassable

Risk gates exist at `tools/trading.py:128-180` (no orders pre-market, max four positions, max 25%
in one name). They are never aggregated into a rate, so the harness cannot say "the agent violated
a position limit in 3% of steps". Worse, `notes/16-competition-harnesses.md:177-183` records that
the caps live in the tool handler while **the shipped baselines call `env.place_order` directly and
bypass them** — a gate on one door with the other door open.

### What ARGUS does

`decision/verdicts.py` enforces the reduce-only asymmetry, `eval/riskproof.py` sweeps 6,720 states
for invariant violations, and `data/risk_records.jsonl` records `intervened`,
`constitution_only_reduced` and `binding_constraint` per decision. The denominator is decisions.

**Taken:** the access-attempt denominator is a genuinely better unit for a *data* firewall than a
decision count, and is noted as a gap. **Rejected:** nothing. **Confirmed:** TraderHarness's
bypassable-gate defect is exactly what `eval/riskproof.py`'s exhaustive sweep exists to make
impossible for us — a gate proved over all states cannot have a second unguarded door.

---

## 3. Stress behaviour

### agent-backtest-lab — three signals, published thresholds

`abl/leakage/reward_hacking.py:57-172` splits chronologically at `is_fraction = 0.7` and raises
three flags:

* **Sharpe collapse** (lines 87-118): `drop = sr_is - sr_oos`. CRITICAL when
  `sr_is > 1.0 and drop > 1.5 and sr_oos < 0.5`; WARN when `sr_is > 0.5 and drop > 1.0`.
* **Drawdown widening** (lines 120-136): WARN when `dd_oos < -0.10 and dd_oos < 1.5 * dd_is - 0.05`.
* **Calibration drift** (lines 138-169): WARN when `ece_oos > 0.15 and ece_oos > ece_is + 0.10`,
  over eight bins (`calibration/reliability.py`).

Supporting maths verified: `sr.py:10-60`, `mdd.py:10-47`.

**Defect:** the 70/30 default is aggressive for a long backtest — the out-of-sample window can be
too short to detect the overfitting it is looking for. The threshold constants are also unexplained
in the source.

**Taken:** the *third* signal is the one worth having and we did not have it — calibration drift
between in-sample and out-of-sample, as a stress signal rather than only as a calibration report.
`eval/forecasts.py` already computes ECE on a chronological split (0.0128 in-sample, 0.0135 out),
so the ingredient exists and was not being read as a degradation test.

**Rejected:** the fixed 70/30 split, for the stated reason; ours is parameterised.

### live-trade-bench

`backend/price_data.py:263-299` refreshes prices every 60 seconds. There is no stress construction
at all — no volatility shock, no liquidity withdrawal, no IS/OOS split. Observed variance from
ordinary price movement is the whole of it.

---

## 4. Human-takeover rate — absent everywhere

**This is the finding.** No repository in any tree measures how often an agent hands control to a
human, and none defines when it must. The relevant negative evidence, file by file:

* `live-trade-bench/systems/stock_system.py:48-75` — agents place allocations. There is no abort
  path and no review request.
* `TraderHarness/tools/trading.py:128-146` — the order tool accepts a reasoning string and has no
  escalation parameter. `agents/protocol.py:12-21` returns `None`, so there is no decision object
  that *could* carry an escalation signal.
* `agent-backtest-lab/abl/types.py:13-31` — `Direction` is `LONG | SHORT | FLAT`. There is no
  fourth state.
* `DARWIN/cto/llm.ts:66-72` — the model output is clamped to
  `{rationale, priority[], riskComment, llmUsed, provider}`. No escalation verb exists in the
  schema.
* `AlphaForgeBench` — agents generate code and the code executes. There is no human gate.

Grepped for `takeover`, `escalat`, `handoff`, `intervention`, `override`, `human_in_the_loop` across
all six trees listed at the top: zero matches in any decision-logging or scoring layer.

**ARGUS's own position before this work.** `decision/verdicts.py:42` defines
`HUMAN_REVIEW = "human_review"`, and `agents/meta_pm.py:366-374` is the only place that can produce
it — on two *structural* conditions:

```python
if verdict.opens_exposure and not invalidation:   # no falsifier named
    verdict = Verdict.HUMAN_REVIEW
if verdict.carries_quantity and quantity <= 0:    # size named as zero
    verdict = Verdict.HUMAN_REVIEW
```

Both are malformed-output triggers. Neither is a *risk* condition. So ARGUS had the state and not
the policy: a desk that escalates only when the model returns a badly-shaped answer has not decided
what it should refuse to decide alone.

**Therefore this is built rather than copied**, and it is the one criterion where there was nothing
to read. `decision/escalation.py` defines the conditions; `eval/bench.py` measures the rate.

---

## 5. Incremental value over a baseline — nothing with a test

### live-trade-bench — benchmarks named, never computed

`backend/models_data.py:424-440` calls `_preserve_existing_benchmarks()` and the comments name QQQ,
VOO and resolved Polymarket prices. **No comparison code exists.** The exported metrics
(`models_data.py:456-479`) are `performance` (percent return) and `profit` (dollars), both absolute.

### AlphaForgeBench — no baseline at all

`metrics.py:328-455` reports absolute Sharpe, Sortino and Calmar per model. There is no
buy-and-hold, equal-weight or random-agent comparison, so the published 0.85 Sharpe has no
reference point.

### agent-backtest-lab — the closest thing, and it is not a baseline

The IS/OOS split is a counterfactual about *regime*, not about *alternatives*. It answers "does this
survive a change in conditions", not "is this better than a rule".

**Nobody in the corpus runs a fixed rule over the same instants and applies a statistical test.**

**Therefore built:** `eval/incremental.py` replays five fixed rules — `always_flat`, `always_long`,
`momentum`, `reversion`, `volatility_gated` — over the identical instants the desk faced, scores
each on the realised move net of the same hurdle, and compares the desk against each pair by pair
with the exact sign test from `eval/ablation.py`. The rules are chosen so at least one should be
embarrassing; `volatility_gated` is the competent-human stand-in the desk must beat to be worth its
deliberation cost.

---

## Defects carried by the best implementations, listed so they are not inherited

| System | Defect | File:line |
|---|---|---|
| AlphaForgeBench | Pass@k cannot see disagreement; three wrong samples out of five still score 1.0 | `metrics.py:365` |
| agent-backtest-lab | 70/30 IS/OOS default too aggressive for long backtests; threshold constants unexplained | `reward_hacking.py:81` |
| live-trade-bench | Zero fees and zero slippage throughout; results are upper bounds | `stock_account.py:18` — the `fees` field is never updated |
| live-trade-bench | Benchmarks named in comments and never computed | `models_data.py:424-440` |
| TraderHarness | Risk caps in the tool handler, bypassed by the shipped baselines calling `env.place_order` | `tools/trading.py:128-180` |

---

## What the field measures that our five-item list omits

Worth recording even though none is being built today:

1. **Per-symbol ablation.** Which instruments does the agent handle well? `live-trade-bench` runs
   several assets and reports no per-asset breakdown. ARGUS's `track1_study.py` does break down per
   symbol, so this is a strength to keep rather than a gap.
2. **Confidence-versus-accuracy calibration on the agent's own decisions.** Only
   `agent-backtest-lab`'s third signal touches it. ARGUS measures this for the *policy* layer
   (`eval/forecasts.py`) and cannot yet for the *desk*, because the desk has never committed to a
   graded directional view — the open item the hurdle frontier named.
3. **Liquidation-distance gate.** `DARWIN/execution/manager.ts:258-275` closes when the distance to
   liquidation falls under 5%. No other system has it. Relevant to perps and absent from ARGUS.
