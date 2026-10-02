# ARGUS — System Architecture

## The technical reference: structures, invariants, algorithms, and what enforces them

**Status: built and running.** Every figure below was produced by running something on this machine,
and `python -m argus.eval.docclaims` checks the ones quoted in these documents against the artefact
that produced them.

| | |
|---|---|
| Source | **581 source files**, 20 packages (measured 2026-09-27) |
| Registered and importable at runtime | **152/152** (`python -m argus.status`) |
| Tests | **11,113 tests collected**; CI runs every one that needs no network with each outbound connection refused, and the 238 that read a live venue, feed or model in a separate advisory job |
| Static analysis | `ruff` clean; `mypy --strict` clean on the 578 files it checks (the three vendored files are excluded) |
| Artefacts | **1,870** files under `argus/data/`; each one a document cites is written by a named command, and a test fails if a cited artefact has no writer |
| External systems torn down at code level | **57 code-level teardowns** under `research/architecture/`, each citing `file:line` |
| Runtime dependencies | **two** — `pydantic`, `python-dateutil` |

That last row is the constraint that shaped everything else. Every statistic here — OLS, augmented
Dickey–Fuller, Engle–Granger, hierarchical clustering, matrix profiles, Fisher intervals, binomial
tests, Ederington effectiveness — is implemented in pure Python, and each one that has a reference
implementation is **checked against it numerically** rather than trusted.

This document supersedes the 2026-09-12 build specification. That was a plan; this is what exists.

---

## 0. The shape

```text
                       ONE RESEARCH WORKBENCH (the Track 3 entry)
   Truth · Evidence · Intelligence · Risk · Execution · Record · Evaluation
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        │                         │                         │
    THE CONSOLE              THE PAPER DESK             THE REGISTER
   question → answer      decisions hashed before     claims anchored
   with a receipt         their outcome (evidence)    before they resolve
```

One engine, one entry. The console is what a trader uses; the paper desk and the register are
where its evidence comes from, and neither is filed as an entry of its own. The team's Track 2
entry is a separate project (t2-sentiment-agent) with its own repository; the factor research
that would have been a Track 1 entry is kept as research (ARGUS-EXPLAINED.md, Part C).

---

## 1. The layer rule, and the code that enforces it

Seven layers, and the vendored copies beneath them. A module imports only from its own layer or
below — at module level or inside a function.

```
  7  demo, lui, eval, status       human surfaces, and the judges of everything below
  6  agents, paper, register       the agents, the record, the public commitments
  5  desk, research, strategies    tools and studies
  4  execution, sim, backtest      orders, simulation, engine
  3  market, llm                   everything that fetches, the model client included
  2  cost, decision, risk, proof   vocabulary, limits, and the Constitution
  1  truth                         time, what was knowable, and where a fact came from
  0  vendor                        third-party copies loaded by path; imports nothing of ARGUS
```

`eval/architecture.py` builds the import graph with `ast` and checks it on every run of the suite:

| contract | rule |
|---|---|
| **Layer order** | every import, deferred ones included, points to its own layer or lower; `LAYERS` names each package's layer, and a package missing from it is itself a violation |
| **Composition root** | the one exemption, lexical and named: an import inside a module's top-level `main()`, where a command-line entry point wires the layers together |
| **Deterministic core** | `truth`, `cost`, `risk`, `decision`, `backtest` never load `argus.llm` or `argus.agents`, **at any depth** |
| **Cycles** | permitted only when broken at load by a deferred import in one direction |

`tests/test_architecture.py::test_the_real_tree_keeps_the_layer_order` runs it on the source that
ships, and further tests prove each contract can fail.

**The rule was published before it was enforced, and it was false.** Until 2026-09-27 the checker
covered the deterministic core and one producer–consumer pair, at module level only. An audit found
41 imports pointing up the order, 33 of them inside functions where the checker could not see them:
the truth layer's trace reaching into the console, the risk layer's sizing importing the evaluation
harness, the Constitution living inside the agent it judges. Each was fixed by moving the shared
piece to the layer it belongs in, never by exempting it — the gate chain and the Constitution to
`risk`; the failure taxonomy, the strict artefact writer, a fact's `Source` and figure grounding to
`truth`; calibration to `risk`; the mandate to `desk`; SUE to `market`, beside the filings it reads;
the console trace to `lui`. Only imports inside a `main()` still cross upward, and they are named.

**Instability** is Martin's `I = Ce / (Ca + Ce)`, computed per package in `data/architecture.json`.
`truth` sits at 0.00 — 141 modules outside it depend on it, and it depends on nothing — which is what
a foundation should look like.

### A blind spot in our own checker, found and closed

The parser accepted only `node.level == 0`, so **every relative import was silently dropped** from
the graph — invisible to the cycle detector, the deterministic-core rule and the coupling metrics
alike. It was latent rather than live: this codebase uses absolute imports throughout, so the graph
was complete *by convention* rather than by construction. `_absolute()` now resolves the leading dots
against the importer's package. A checker that is correct only while a convention holds is a checker
that stops being correct without telling anyone.

---

## 2. The core data structures

### `truth.facts.Fact` — five times, not one

Every fact carries: when the event **happened**, when it was **published**, when we **ingested** it,
when it became **available** to us, and a **revision number** so a restatement supersedes rather than
overwrites. A query without an as-of date is a type error. There is no "latest".

### `truth.clocks` — the dual clock

The token's clock never stops. The anchor equity's has four phases (regular, extended, overnight,
weekend) plus holidays. `SessionState` carries the phase, `hours_to_next_discovery`, and staleness
thresholds that **differ by phase** — the same age means different things at 2pm and at 2am.

The consequence the rest of the system is built around: a fact can be **knowable and unpriceable at
the same instant**.

### `decision.verdicts.Intent` → `ConstitutionRuling`

An `Intent` carries side, quantity, verdict, stated confidence, thesis, and — mandatory —
`invalidation`: what would make the thesis wrong. A position with no falsifier cannot be monitored,
only hoped over.

`apply_constraint()` is the **single function** through which the risk layer may alter an intent, and
it can only narrow. The ruling names the binding constraint in machine-readable form, because "which
constraint bound, how often" is the evidence that the risk layer does real work, and free text cannot
be aggregated.

### `paper.ledger.Entry` — the hash chain

Append-only, each entry hashing the previous. An order is bound to the exact `approved_intent_hash`
that authorised it — not to a correlation id, which could be reused. The head is anchored
(`paper/anchor.py`) so truncation is detectable, and `data/anchors/` holds OpenTimestamps proofs
submitted to live Bitcoin calendar servers.

### `cost.model.CostModel` — no zero-fee constructor

Commission, spread, square-root impact (γ = 1.5, cvxportfolio's `costs.py:826`), borrow, and
**perpetual funding charged per 8-hour settlement** with the venue's ±0.5% cap enforced at
construction — a rate outside it is a unit error, not a large cost.

There is deliberately **no `CostModel.zero()`**. A frictionless model exists only via
`frictionless_for_research(reason)`, which stamps it so `assert_gateable()` refuses to let it justify
a decision. Cost-blindness was the most widespread defect across the 62 systems torn down, so it is
prevented structurally rather than by review.

---

## 3. The Register — claims committed before their outcomes

The only part of this system whose value comes from elapsed time rather than from code, which is why
it was opened before it was finished.

| module | what it holds |
|---|---|
| `claims` | The `Claim` schema, the **refuse-at-registration** validator, and the hash chain. A claim whose resolution predicate cannot execute against a named point-in-time source is rejected with the reason — that single rule is what separates a register from a comment section. |
| `resolve` | The auto-resolver and the scoreboard. Checks the horizon **before** it fetches a price, grades on the last bar at or before the horizon, and returns UNRESOLVABLE — never FALSE — when a source will not answer. |
| `open_register` | Opens the register: builds a mechanical batch, commits it, and submits the head hash to four Bitcoin calendars. |
| `cadence` | **Continuous registration on every cycle**, across a ladder of horizons — 24h, 72h, 7d, 14d, 30d — so something is always pending and something has always just resolved. |

**The invariants, each enforced rather than intended:**

- **Append-only.** There is no update and no delete. A resolution is a *new line* naming the claim it
  resolves, so nothing already written is ever rewritten and a stranger can verify the file from byte
  zero without trusting the process that wrote it.
- **The digest covers the claim, not its outcome.** `content()` drops status, resolution and
  observation before hashing, so the commitment is checkable *before* anyone knows the answer — which
  is the entire point of committing it.
- **It cannot resolve early.** The horizon check happens before the fetch. There is no code path in
  which the resolver has seen the outcome and then decided not to use it.
- **A minimum horizon.** A claim must resolve at least an hour after registration, far outside any
  plausible clock skew between this host and the venue's stamps, so a claim can never be registered
  about a bar that has already printed.
- **Brier, not hit rate.** Hit rate is improved by only making easy claims. Brier charges for
  confidence, so the same accuracy stated loudly scores worse than stated honestly. Both directions
  are pinned by test, because "fixing" this into rewarding timidity would invert the incentive the
  register exists to create.

**A defect in the register's own opening, found and fixed the same day.** All 36 founding claims
were committed at a single 24-hour horizon, so every one resolved the next morning — while the
handbook puts judge review at **9/22 to 10/7**. The record would have been frozen for the entire
window it was built to be watched during. `horizon_coverage()` measures exactly this and reported
**0 resolving inside the window, 0 still pending when it opens**; the cadence fixes it, and a test
pins the failure shape so it cannot return unnoticed.

**Live:** 391 claims across the twelve stock perpetuals (`data/register.jsonl`), chain intact. The opening anchored head `bc36478291a06bc3` is claim 36 of 156 — the head at the moment of that anchoring, so that proof covers the first 36 claims and not the 120 registered since; each scheduled cycle appends and re-anchors, and 38 of the 60 proofs carry a Bitcoin block-header attestation (`python -m argus.register.anchorcheck`). Anchored to
`a.pool.opentimestamps.org`, `b.pool.opentimestamps.org`,
`alice.btc.calendar.opentimestamps.org` and `finney.calendar.eternitywall.com`. The resolver runs on
every scheduled cycle.

**The confidences are set by a written rule, not chosen.** Volatility claims at 0.60, because
`risk/session_risk.py` measured that the reopen bar carries 1.5–2.2× a regular-hours bar and every
horizon crosses one. Direction claims at exactly **0.50**, because `desk/analogue.py`,
`desk/shapematch.py` and `research/cointegration.py` each measured that we have no directional edge
on these instruments — and claiming one we had disproved ourselves would be precisely the dishonesty
this register exists to price.

## 4. The Constitution — the gate chain in source order

`risk/constitution.py:ConstitutionPolicy.rule()`, in the order `risk/gatechain.py:CHAIN` records.
Each gate may only narrow. The three **terminal** gates end the evaluation when they fire; every
**ceiling** gate after them always runs, and the tightest ceiling sets the size — so a ceiling is
never skipped because another happened to bind first.

| # | gate | kind | binds when |
|---|---|---|---|
| 1 | `no_exposure` | terminal | the verdict carries no quantity |
| 2 | `min_confidence` | terminal | stated confidence below the 0.55 floor |
| 3 | `oracle_stale` | terminal | NAV stale **while the anchor is shut** |
| 4 | `unhedgeable_gap` | ceiling | no hedge placeable and size above the unhedged cap |
| 5 | `gross_exposure` | ceiling | whole-book gross notional above its cap |
| 6 | `signed_exposure` | ceiling | whole-book net directional notional above its cap |
| 7 | `hedge_integrity` | ceiling | a reduction would leave a linked hedge's partner exposed |
| 8 | `margin_usage` | ceiling | the venue-reported margin ratio at or above the cap |
| 9 | `factor_exposure` | ceiling | measured exposure to a named factor above its cap |
| 10 | `scenario_loss` | ceiling | the worst modelled benchmark shock already past the loss floor |
| 11 | `liquidation_cost` | ceiling | the worst position's forced-exit slippage at or above the cap |
| 12 | `per_symbol_underperformance` | ceiling | one symbol's windowed realised P&L below its floor (narrows that symbol only) |
| 13 | `risk_budget` | ceiling | the position exceeds the book's risk budget — calibration-gated fraction x the breaker's drawdown ladder |
| 14 | `session_volatility` | ceiling | measured path volatility above the regular-hours baseline |
| 15 | `max_position` | ceiling | size above the absolute cap |

`risk_budget` reacts to **our** drawdown and `session_volatility` to **the market's** clock —
separate gates rather than one multiplier, because a single number could not be attributed.
`max_position` is an absolute ceiling, so no multiplier ever dilutes it.

**`risk_budget` was added on 2026-09-14 to close a gap between two modules that both already
existed.**
`risk/circuit.py` (the breaker) and `risk/sizing.py` (Kelly, gated on calibration) were complete and
tested, and **neither was ever called on a live decision**: `sizing.size()` had no caller anywhere in
`src/argus`, and `sizing.py:149` documented `risk_multiplier` as coming from `circuit.risk_multiplier`
while nothing connected them. The breaker's own docstring set the bar — *"the test for this module is
not 'does it run' but 'inject a 3-sigma adverse move mid-cycle and confirm the desk produces a
different answer, and says why'"* — and until this gate existed it could not.

The budget is `equity x fraction`, where `fraction` comes from `sizing.size()`:

| the record | fraction | on a 100,000 book |
|---|---|---|
| fewer than 20 graded outcomes, or ECE above 0.15 | `FIXED_FRACTION` = 5% | 5,000 |
| calibrated: >= 20 graded, ECE <= 0.15 | half-Kelly on stated confidence, capped at 25% | up to 25,000 |
| drawdown >= 4% / 6% / 10% | x0.75 / x0.5 / **x0** of the above | tightens, then halts |

**When it was added that was a 10x tightening.** On 2026-09-14 the record held 183 decisions, 69
settled, and **zero** gradeable outcomes — no settled trade, no settled lean — so the calibration
gate refused and the cap was 5% of the book. Before it, the only size limit was the flat 50,000
notional of `max_position`. An
unproven desk is now sized as an unproven desk, and **proving calibration is the only thing that
lifts it**: `test_risk_budget_gate.py` pins that twenty well-calibrated outcomes earn more size and
that a lower stated confidence earns less, so the gate cannot degenerate into a constant.

The session throttle is deliberately passed to `size()` as 1 and applied by `session_volatility`
instead. Folding it in would charge the market's clock twice and make the binding constraint
unattributable: `risk_budget` reacts to **our** drawdown and our proven calibration,
`session_volatility` to **the market's** clock.

`book_state` is injected exactly as `session_risk` is; absent, the gate does not fire. It returned
`None` on an empty ledger for one revision, which was over-cautious and wrong in the dangerous
direction — a paper book that has closed nothing has realised equity of exactly the starting book
and a drawdown of exactly zero, both measurements, and returning `None` disabled the position cap on
precisely the book that has proven nothing.

**Reachability, not just firing.** `eval/autopsy.py` walks this chain in source order and subtracts
each gate's hits before considering the next, so every gate is reported as FIRED, PASSED or
**UNREACHED**. A gate that never executed has unmeasured permissiveness — it has not been shown to
be safe, and reporting it as "passed" would be a claim nobody earned.

**The chain is now known to be reachable, and that is new.** Across all 170 recorded decisions gate
1 returned every time — `no exposure proposed; nothing to narrow` — so no later gate had ever
executed and a chain that is never entered cannot be said to work. The cause was never the risk
layer: every directional study here measured no edge, so the desk correctly proposed nothing.

`desk/carrydesk.py` supplies the missing input. A funding payment is collected for *holding* a side,
not for being right about direction, so a carry basket carries a quantity without claiming a
forecast our own measurements refuse. Live, on the surviving basket from `research/carry.py`:

| gate | status | why |
|---|---|---|
| 1 `no_exposure` | **PASSED** | first time on record — the proposal carries a quantity |
| 2 `min_confidence` | **FIRED** | stated confidence 0.371 is below the 0.55 floor |
| 3–15 | UNREACHED | gate 2 is terminal, so the chain ends there and they are reported as unreached |

This is an assessment against the real `ConstitutionPolicy`, **not a ledger entry**: the trade was
refused, so nothing was booked; the ledger then held zero settled positions (its first settled on
2026-09-29).

The confidence is a Wilson lower bound on the share of profitable holding windows, computed against
the **independent** window count. The study replays 1,823 overlapping 14-day windows drawn from 90
days of history; those are about 6 independent holds. At 6 the bound is 0.371 and the floor refuses
it; at 1,823 it would be 0.72 and the position would have been sized on a sample that does not
exist. Gate 1 was cleared by supplying an honest proposal, never by loosening the gate — and the
first gate that actually ran then refused the trade. `test_carrydesk.py` pins both halves, including
one test proving a sufficiently confident proposal *does* reach past gate 2, so UNREACHED stays a
fact about this proposal rather than about the code.

**The asymmetry.** The Constitution may reduce exposure and may never originate a thesis. The one
apparent exception is documented precisely: `REQUIRE_HEDGE` attaches a hedge leg, which lowers net
risk while raising gross notional (`decision/verdicts.py:178-185`). Three invariants, not one.

---

## 5. The numerical kernels, and what each was checked against

This is the section that distinguishes a system from a demo. Every kernel below is pure Python and
every one with a reference was validated against it numerically.

| kernel | module | reference | agreement |
|---|---|---|---|
| OLS, ADF, Engle–Granger, MacKinnon tables | `research/cointegration.py` | statsmodels 0.14.6 | **1e-12** on 10 unit-root + 3 cointegration tests over real hourly closes, including lag selection, sample sizes, and its unexplained `nobs − 1` Stata quirk |
| Hierarchical risk parity | `desk/allocation.py` | PyPortfolioOpt | **3.6e-16** on 12 instruments, cluster order included |
| z-normalised shape distance | `desk/shapematch.py` | stumpy `core.py:1118` | identity `d = √(2(1−ρ))` asserted by test; flat-vs-structured = 1.0 matches stumpy's `D² = m` special case by construction |
| Matrix profile + FLUSS arc curve | `desk/regime.py` | matrixprofile `regimes.py`, stumpy `floss.py` | follows each where it is better: matrixprofile's analytic parabola, stumpy's 5× head/tail pin |
| Ederington hedging effectiveness, Fisher interval | `risk/effectiveness.py` | Ederington 1979 | textbook interval reproduced (r = 0.5, n = 100 → 0.337–0.634) |
| Benjamini–Hochberg, Bonferroni | `backtest/validation.py` | — | single implementation, **imported** by the pairs scanner rather than copied |
| Queue-position fills, latency | `execution/queue.py`, `latency.py` | hftbacktest `queue.rs`, `latency.rs` | ported with citations |
| Order state machine | `execution/orders.py` | nautilus `enums.rs:1304-1388` | 15 states, read from source after a from-memory version had 11 |
| Multi-level matching | `sim/book.py` | abides-jpmc-public | rebuilt |
| Memorisation probe | `eval/leakage.py` | barj28 `vr_C_api.py:124-137` | multipliers and deterministic shuffle copied exactly; parser **deliberately stricter** (the reference's `[A-E]` matches the "C" in "cannot") |
| Injection defence | `agents/quarantine.py` | AgentDojo `agent_pipeline.py:220-276` | spotlighting + replace-don't-drop taken; its 440MB transformer detector not taken |

### Complexity, where it matters

- Matrix profile is brute force `O(n²m)`: 1,079 bars at a 24-bar window ≈ 24M multiply-adds,
  **10 seconds**. The inner loop uses the correlation identity so each comparison is one dot product.
- Shape-match null calibration is 50 full rescans ≈ 6 seconds on 1,438 bars.
- Single-linkage clustering is naive `O(n³)` — 12 instruments is 66 pairs, and the readable algorithm
  costs less than one HTTP round trip.

Each is a deliberate trade: numpy plus numba would save those seconds and cost two dependencies.

---

## 6. Measurement → artefact → consumer, with staleness as absence

Four measurements run **before** each cycle and write files the cycle then reads:

| measurement | artefact | consumed by | stale rule |
|---|---|---|---|
| `market/markout.py` | `markout.json` | cost-model maker decision (advisory) | per-run |
| `market/depth.py` | `depth.json` | the executable spread charged on **both legs** of every fill | per-run |
| `risk/session_risk.py` | `session_risk.json` | the Constitution's `session_volatility` gate | **36h → absent** |
| `risk/effectiveness.py` | `hedge_effectiveness.json` | the hedge surface's three factors | **36h → absent** |
| `eval/leakage.py` | `leakage.json` | `eval/shadow.py` via `check_window()` | control must pass, else **UNMEASURED** |

**Staleness is absence, never a weaker number.** A correlation measured three weeks ago is not a
worse estimate of today's — it is a statement about a different market. When a measurement is
missing, the consumer says so in the record: the hedge candidate is stamped `ASSUMED` instead of
`MEASURED`, the session gate goes inert and the autopsy records it UNREACHED, and the leakage gate
returns UNMEASURED — which **never reads as clean**, because an instrument that cannot fire certifies
nothing.

---

## 7. The execution-realism stack

| layer | what it models | source of truth |
|---|---|---|
| `execution/guard.py` | the venue's published instrument rules | fetched from Bitget, not assumed |
| `execution/preflight.py` | the checks before an order is allowed to exist | — |
| `market/depth.py` | **live L2**, sweep cost by size, executable size within a slippage budget | `GET /api/v3/market/orderbook`, 200 levels |
| `execution/queue.py` | queue position, partial fills | hftbacktest — **12 of 13 OWNED conditions**, see below |
| `eval/bookcalib.py` | the real book's level sizes and turnover | `GET /api/v3/market/orderbook`, appended every cycle |
| `execution/latency.py` + `latency_probe.py` | round-trip latency, **measured** | live probes |
| `market/markout.py` | adverse selection after other people's prints | `GET /api/v3/market/fills` |
| `sim/book.py` | multi-level matching for backtests | ABIDES |

**What the depth measurement changed.** The ledger charged the quoted touch on entry and a flat
0.6bps on every exit. Measured live, taking $25,000 costs **0.82bps of QQQUSDT and 19.43bps of
SQQQUSDT** against quotes of 0.14 and 4.96 — the quote understates by a factor that varies five-fold
between instruments, which is worse than understating by a constant because it **reorders which
instruments look cheap**.

**The queue model is the closest thing here to OWNED, and it stops one condition short.**
`eval/queueproof.py` is the evidence package, and every line of it is produced by running something:

* **The port is exact.** All 8 probability-function/parameter combinations reproduce
  `queue.rs:221-330` to **1.1e-16** on 9 (front, back) pairs spanning both extremes.
* **What `prob()` claims was read, not assumed.** The first draft scored it as P(fill) and nearly
  published a negative result about the wrong quantity. `queue.rs:183-204` settles it: `prob` is
  **P(a cancelled unit came from behind the order)** — attribution, not outcome.
* **Ground truth by construction.** An explicit order-by-order queue generates the history; each
  model sees only the L2 view (level total before and after, printed trade size) and its estimate of
  the quantity ahead is scored against the queue that actually produced those totals.
* **The generative rule is swept, not chosen.** Five regimes. A probability function beats the best
  constant in **3 of 5** in sample and the same 3 held out — with the in-sample winner *carried
  over* rather than rechosen, and every margin's whole 95% paired interval above zero.
* **It loses 2 of 5, and they are reported.** Where the front of the queue never cancels, "every
  cancellation is behind you" is not a naive assumption but the exact truth (error 0.0000), and no
  member of the family can express it.
* **One result cuts against the model.** In one regime it estimates the queue better and still costs
  more (−0.020bps): queue accuracy and execution cost are not the same objective.

**The thirteenth condition is not met and is named rather than argued away.** hftbacktest validates
this model against recorded real market data including market-by-order feeds; ARGUS validates it
against a simulator whose attribution rule we stated. Bitget's public API publishes L2 only, and
which side of a resting order a cancellation came from is **not recoverable from L2 at any sample
rate**. Closing it needs our own orders resting in the real book — elapsed time, not more code. So
the capability stays IMPLEMENTED at 12/13, and `eval/standing.py` raises at import if anyone writes
otherwise.

What *is* being closed meanwhile: `eval/bookcalib.py` appends a real order-book snapshot per
instrument on every cycle, so the simulator's parameters stop being ours. The first tape already
disagreed with them sharply — a near-touch stock-perpetual level holds a **median 1.95 contracts** and moves
a **median 70% of itself per minute**, nothing like the large, slow levels the experiment assumed.

---

## 8. The evaluation layer — the modules whose job is to find us wrong

The largest package in the system, deliberately: 255 modules in `argus/eval/` (`data/architecture.json`), 71 of them the
vendored or reimplemented rival baselines under `eval/baselines/` that the comparisons run on
the same inputs (counted 2026-09-27). The ones that judge the system itself:

| module | what it can prove false |
|---|---|
| `docclaims` | that a number quoted in our documents matches the artefact that produced it |
| `architecture` | that the layer rules hold |
| `standing` | that a capability claimed OWNED meets all **13 conditions** — raises at import otherwise |
| `autopsy` | which risk gate bound, and which were never reached |
| `shadow` | whether the desk's directional view beats the break-even accuracy, while it refuses to trade |
| `leakage` | whether the model had already read the period being evaluated |
| `riskproof` | invariants across an exhaustive state-space sweep |
| `consistency` | same state, same answer (lean 100%, action 80%, verbatim 20%) |
| `cyclecheck` | nine checks per scheduled cycle, with a named falsifier |
| `themeaudit` | every named sub-theme, run rather than asserted |
| `luibench` | whether the language interface is fluent, including refusal in **both** directions |
| `ablation`, `ablations`, `incremental` | whether each component actually adds value |
| `forecasts`, `forecastbench` | calibration, Brier decomposition, restatement handling |
| `overfit`, `overfitting_study` | CPCV, deflated Sharpe, PBO |

**The capability ladder** is enforced in code: LOST → TIED → IMPLEMENTED → OWNED, with OWNED requiring
thirteen conditions including a reproduced baseline, same-input comparison, out-of-sample test,
ablation and adversarial test. Live (`data/standing.json`): **6 of 46 capabilities are OWNED**,
33 TIED, 0 IMPLEMENTED, 7 LOST — re-derived by `python -m argus.eval.standing` from the artefacts.

---

## 9. Known architectural debt

Published rather than hidden, because a document that lists only strengths is a brochure.

- **Two orphan modules.** `market/validation.py` and `paper/chains.py` are imported by nothing —
  both masked by name collisions with well-tested modules of similar name.
- **`risk/circuit.py` and `risk/sizing.py`** describe an integration with each other in their own
  docstrings that was never coded.
- ~~**The external anchors are not attached.**~~ **Fixed.** Both live commitments had been submitted
  to four Bitcoin calendars and recorded `external_anchor: None`, so the record claimed less than it
  could prove. `paper/protocol.py:attach_anchors` links them, and the digests still verify — the
  anchor is evidence *about* a commitment, not part of it, which is what made it safe to add
  afterwards and also why it was easy to forget.
- **Duplicated statistics that can disagree.** Four z-normalisation implementations split between
  population and sample variance; correlation computed on simple returns in one module and log
  changes in another (ratio-based hedge ratio diverges ~1.3% at stress volatility); two different
  volatility estimators used for the same fast/slow regime comparison.
- **A median that was wrong.** `research/gap_study.py` returned the upper of the two middles on every
  even-length input, biasing published figures upward. **Fixed, tested, and the artefact
  regenerated** — and the test file that would have caught it now exists, because that module had
  none.
- **`paper/runner.py` is the change-risk hotspot**: fan-in 16, fan-out 36, and all sixteen importers
  want only three path constants. A small `paths` module would cut it.

---

## 10. What we refuse to build

Each is easy to build, impossible to defend, and present in the field:

- Bull agent + bear agent + trader, with no ablation showing each role adds value.
- Sentiment score → LLM decision. The edge is the same size as the fees.
- EPS beat → buy.
- "AI discovered a factor" after 20,000 trials with no trial count, DSR, PBO or capacity.
- Backtests without transaction costs — the most widespread defect class in the entire corpus.
- LLM arithmetic on financial values. Code computes every number; prompts carry an explicit
  *"do not estimate"* instruction.
- Memory without `available_at`.
- Production self-editing.
- Fake or backfilled paper trading.
- A Track 3 chatbot that answers questions instead of grounding decisions.
- Competing on counts — most agents, most integrations, most charts. All saturated.

---

## 11. The claim this architecture has to earn

Not *"ARGUS has twenty-one innovations."* That is a count, and the rule above bans it.

> We reproduced the strongest systems we could find. Here is where they fail. Here is ARGUS. Here is
> the exact metric where it improves, the ablation showing our component caused it, the attack it
> survived, the out-of-sample result, the replay, and the paper-trading evidence.

**What that has produced is mostly negatives, and they are the point.** 0 of 8 factors certified.
0 of 12 symbols surviving the strict deflation gate. 0 of 66 instrument pairs cointegrated after
multiple-testing correction. 22 of 24 shape-retrieval cells indistinguishable from reordered noise.
A weekend effect that failed its own train/test split. A system that reports those is worth more than
one that reports a Sharpe.

**The honest gap:** the venue has verified a signed order round-trip, and **no position has ever been
opened** — every one of the 183 recorded decisions is a refusal, and all of them fall in weekend
sessions with no price discovery. `eval/autopsy.py` says so in its own words and names the falsifier
that would overturn its explanation. That is the single largest open item, and it is stated here
rather than left for a judge to find.
