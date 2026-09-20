# ARGUS

**An evidence-driven autonomous trading desk for 24/7 tokenized equity markets.**

**ARGUS refuses trades that cannot pay for themselves, and writes down why before the
outcome exists.**

- Of eight vetted factor primitives over **2,159 real NVDA bars**, **6 are indistinguishable
  from randomly shuffled data** and **2 return no verdict at all** for want of data — which
  is not a pass. **None clears every gate, and all eight lose money after the 12bps round trip.**
  Published, not buried (`data/overfit_gates.json`).
- Every decision is hash-chained **before** its outcome exists, and **156 claims across the
  twelve Bitget rTokens** are pre-registered and anchored to Bitcoin before they can be checked
  (`data/register.jsonl`). Built on Bitget's public market data and Qwen through the hackathon
  endpoint.

Most trading agents ask *what should I trade?* ARGUS asks a harder question:
**should this decision-maker be trusted with capital right now?** On the live record the answer has
been *no* **447 times out of 447** — and the interesting output is not the trade, it is the
record that says why.

> ### ⚠️ Correction, 2026-09-20 — two "trades" in this ledger were never taken
>
> Earlier today this README claimed the desk had broken its refusal streak with two profitable
> trades (COINUSDT and MSTRUSDT, +14.99 net). **That claim was false and is withdrawn.**
>
> `paper/runner.py` selected the intent to record as
> `llm_revised_intent or llm_original_intent`. That fallback bypasses the risk layer, and it is
> reached precisely when the desk chooses *not* to re-put a decision to the model — which is most
> decisions. For ledger seq 264 and 265 the Constitution refused the position
> (`quantity_after: 0`, `binding_constraint: no_exposure`) and `agents/desk.py` wrote
> *"no order: final verdict human_review with quantity 0"* — while the ledger stored
> `verdict: trade, quantity: 1` and the settlement pass later booked P&L against positions that
> were never opened.
>
> **The rows are still in the ledger and have not been edited.** The chain guarantees a record was
> not altered; it does not guarantee the record was right, and quietly deleting the two rows that
> made the numbers look good is exactly the behaviour this project exists to refuse. They are
> marked as void in `paper/corrections.py`, with the evidence, and every figure derived from the
> ledger now excludes them.
>
> Found by an adversarial pre-submission audit reading `eval/decisioncard.py --seq 264`, which
> printed all four contradictory facts on one page — the artefact did its job. No test caught it:
> every test fed that path an intent the Constitution had already allowed, so the divergent branch
> was never exercised. `tests/test_governed_intent.py` now exercises it directly.

Design: [`../ARGUS-ARCHITECTURE.md`](../ARGUS-ARCHITECTURE.md) ·
Plan and evidence: [`../ARGUS-MASTER-PLAN.md`](../ARGUS-MASTER-PLAN.md)

---

## Status, measured on 2026-09-20

```
4,276 tests collected (33 skipped)   ruff clean   mypy --strict clean on 266 source files
131/131 modules importable           18/18 sub-themes resolve to a symbol and a test file
447 decisions in the paper ledger (as of 2026-09-20)  chain verifies, head anchor agrees, no truncation
2 settled trades, both directionally correct, +14.99 net after costs   99.6% abstention rate
```

Reproduce all of it:

```bash
pip install -e ".[dev]"
pytest
python -m argus.status          # modules, sub-themes, artefacts, what is blocked and why
```

`argus.status` is checked at runtime rather than asserted in this file, because a coverage table in
prose is a claim and an import that resolves is evidence.

---

## Run it

**The desk, end to end, against the live model** — needs `BITGET_QWEN_API_KEY` in the environment:

```bash
set -a && . ../.secrets/qwen.env && set +a
python -m argus.paper.runner --once --symbols NVDAUSDT,TSLAUSDT
```

One cycle reads the live market, gathers evidence, selects and runs the analyst panel, lets the
model decide, applies the Constitution, and appends the result to the ledger. It prints what it
wrote and whether the chain still verifies.

**Ask it questions in English** — no key needed, stdlib only, no network:

```bash
python -m argus.lui.cli "how did we do this week?"
python -m argus.lui.server        # then open http://127.0.0.1:8765
```

**Ask the portfolio copilot a complete research question** — live Bitget candles, no key:

```bash
python -m argus.desk.portfolio --book TSLAUSDT=0.4 MSFTUSDT=0.4 METAUSDT=0.2 --add NVDAUSDT
```

It answers *should I add this, given what I already own*: open-session beta before and after, each
position's share of **risk** against its share of the money, the holding it is most correlated with,
a benchmark shock propagated through each position's own beta, the worst run this exact book would
actually have had in the observed history, and its exposure to our own session-structural factors.
That is every item the Track 3 Open Theme names: beta, sector, factor exposure, correlation,
concentration, stress and hedge reachability.

**Check the record:**

```bash
python -m argus.paper.runner --report    # Sharpe, max drawdown, win rate, or why each is undefined
python -m argus.eval.scorecard           # calibration, episodes, abstention quality
python -m argus.paper.repair data/paper_ledger.jsonl   # verify the chain; plan only, writes nothing
python -m argus.eval.riskaudit           # did the risk layer ever change a decision, and how often
python -m argus.eval.baseline            # did the desk beat simply holding what it picked
python -m argus.eval.docclaims --tests   # every number these documents quote, checked against its artefact
python -m argus.paper.protocol --show    # the pre-registered trading protocol and its digest
python -m argus.paper.protocol --audit   # every governed decision, checked against what we committed to
python -m argus.lui.cli "did we make any money this week?"   # the console; routes when patterns miss
python -m argus.desk.stress --symbol NVDAUSDT --quantity 1   # stress a trade idea against its own history
python -m argus.desk.review               # grade the desk's own process, and its checklist
python -m argus.desk.research --symbol NVDAUSDT --book "AAPLUSDT=0.5,MSFTUSDT=0.5"   # one question, end to end
python -m argus.eval.riskproof           # sweep the risk layer's whole input domain
python -m argus.eval.decisioncard --seq 64   # one decision, whole trail, on one page
python -m argus.execution.guard --symbol NVDAUSDT  # the venue's published rules for one instrument
python -m argus.market.macro             # the US Treasury par yield curve, dated at the issuer
python -m argus.eval.themeaudit          # every named sub-theme and judged criterion, run not asserted
python -m argus.eval.autopsy             # why no position was taken, and which gates were never reached
python -m argus.demo.flow --scenario --send   # the whole flow, filing to order, in one trace
python -m argus.demo.cockpit             # regenerate the evidence page; open argus/cockpit.html
python -m argus.eval.luibench            # is the console actually fluent, or only tested?
python -m argus.eval.architecture        # layering, cycles and coupling, measured from the imports
python -m argus.eval.consistency         # same state, same answer? (spends model tokens)
python -m argus.eval.cyclecheck          # did the last scheduled cycle do what it was supposed to
python -m argus.desk.shapematch --symbol NVDAUSDT  # when did the last day trace this line before, and what followed
python -m argus.research.cointegration     # which rToken pairs are actually cointegrated (66 tests, corrected)
python -m argus.risk.effectiveness        # how much risk the anchor hedge actually removes, per session phase
python -m argus.desk.allocation           # the book it should hold, and whether the rebalance pays for itself
python -m argus.market.depth              # what it actually costs to cross each book, by size
python -m argus.risk.session_risk         # which hours actually move, and the throttle that follows
python -m argus.market.markout            # how badly a resting order is picked off here
python -m argus.desk.regime               # where the series changed shape, not just amplitude
python -m argus.eval.leakage --dry-run    # the memorisation probes, built from filings, 0 model calls
python -m argus.register.resolve          # resolve every claim whose horizon has passed, and print the record
python -m argus.eval.shadow              # is the desk right about direction while it refuses to trade
python -m argus.research.searchoff --sweep   # five search strategies, three seeds, ranks kept not averaged
```

---

## What is actually different here

Each of these was built after reading the system that leads that category, and each cites the file
that was read. None of them is claimed as *proven better* — see "The honest part" below.

| Capability | Where | Why it is not standard |
|---|---|---|
| **Deliberation priced as a hurdle in basis points** | `agents/meta_pm.py:65` | Every finance-agent harness in our corpus treats model latency as free. Off-hours a full reasoning budget costs more than the 12bps round trip, so the desk is told what its own thinking costs before it decides. |
| **Two clocks, kept separate** | `truth/clocks.py` | The token trades continuously; the anchor equity does not. A fact can be *knowable* and *unpriceable* at the same instant, and no single-clock point-in-time model can express that. |
| **A Constitution that may only reduce** | `agents/desk.py`, `decision/verdicts.py` | The risk layer cannot increase directional exposure to the model's thesis and cannot originate a thesis of its own, so the economic choice stays the model's and the guard stays binding. Stated precisely because it is three invariants and not one: REQUIRE_HEDGE does attach a hedge leg, which lowers net risk while raising gross notional (`decision/verdicts.py:178-185`). |
| **Abstention scored as a decision** | `paper/ledger.py`, `eval/observatory.py` | A desk that correctly stands aside and one paralysed by its own hurdle look identical unless the move it declined is recorded at the time. |
| **A directional view stated while refusing to trade** | `eval/shadow.py`, `agents/meta_pm.py` | Declining to trade says the edge does not clear the hurdle, not that there is no view. The desk states a `lean` on every refusal, hashed with the decision, graded against the move that followed. It exists because the `side` field was measured and found to be `BUY` in all 53 settled rows — right 3.8% of the time against a 3.8% base rate. |
| **The leakage gate, enforced rather than published** | `eval/leakage.py:check_window`, `eval/shadow.py` | Measuring contamination and then not acting on it is half a control. `check_window(start, end)` reads the stored measurement and answers CONTAMINATED / CLEAN / **UNMEASURED** — and the third is the one that matters: a missing report, or a report whose sensitivity control failed, is never allowed to read as clean, because an instrument that cannot fire certifies nothing. `eval/shadow.py` now carries the decision dates of every graded call, bounds the window it grades, and prints the status beside its own verdict. Live, the shadow record reports **CONTAMINATED** — the model recalls facts published in the period it grades. The note is attached to the verdict rather than used to suppress it: the instrument measures recall of *fundamentals* and the record grades *price direction*, so overlap is the precondition for leakage, not proof of it. |
| **A register of claims committed before their outcomes** | `register/claims.py`, `register/resolve.py` | Every claim made about a market is made into a void: a call is posted, a backtest reports a Sharpe, an agent opens a position with a thesis, and none of it is mechanically connected to what happened next. **36 falsifiable claims about all twelve rTokens are now committed, hash-chained, and anchored to four independent Bitcoin calendars** — head `bc36478291a06bc3`, proofs in `data/anchors/`. A claim whose resolution predicate cannot execute against a named point-in-time source is **refused at registration**, which is what makes this a register rather than a comment section. The resolver checks the horizon *before* it fetches a price, so there is no code path where it has seen the answer and then decided not to use it; a source that will not answer yields UNRESOLVABLE, never FALSE. Scoring is Brier, not hit rate — the same accuracy stated loudly scores worse than stated honestly — and **The Wall** publishes every claim we get wrong, most confident first. The confidences are set by a written rule, not chosen: volatility claims at 0.60 because `risk/session_risk.py` measured the reopen jump, and direction claims at exactly **0.50**, because every retrieval we built measured that we have no directional edge on these instruments and claiming one would be the dishonesty this register exists to price (`data/register.jsonl`, `data/register_scoreboard.json`). |
| **Memorisation leakage measured — and the control that caught two harness bugs** | `eval/leakage.py` | Every historical evaluation here asks a model about a period that already happened, and none could say whether it had simply **read the answer**. barj28's instrument (DOI 10.5281/zenodo.20844335) was read and its probe copied exactly — multiplicative distractors, wide `{0.5,0.7,1.4,2.0}` and tight `{0.85,0.93,1.08,1.18}`, deterministic per-fact shuffle. The facts are ours: 42 probes built from what companies actually filed with the SEC, bucketed by **filing date**, because a model can only learn a number after it is published. **Live: 32/42 tight probes correct (76%) against 20% chance, p<0.0001, above chance in every period counted including filings from August 2026** — the model recalls these numbers to within ±18%, and no upper boundary was located. So any evaluation over a window in that range is contaminated. **The sensitivity control is why that number is trustworthy**: its first run scored 2/12 with the answer printed in the prompt, which is impossible — the parser was reading the model's chain of thought instead of its answer; a second run caught a literal backspace byte where the regex needed `\b`. Both bugs were invisible in the probe results alone, because a broken instrument and an honest null look identical. Control now 12/12 (`data/leakage.json`, `research/architecture/37-memorisation-leakage.md`). |
| **Untrusted evidence quarantined before the model reads it** | `agents/quarantine.py` | Every headline, filing and footnote the desk reasons over is written by somebody else, and `agents/analysts.py` interpolated it straight into the prompt. ARGUS had three checks on that text — grounding, claim support, an adversary — and **all three assume it is wrong; none assumed it is hostile.** AgentDojo's four defences were read and taken selectively: spotlighting with delimiters (arXiv 2403.14720) plus the standing system instruction, and its **replace-don't-drop discipline** (`pi_detector.py:48-51`) — a hostile item keeps its id, source and timestamp and loses only its claim, so no downstream count silently shrinks. Its 440MB transformer detector was not taken; ours is six structural patterns that report **the span that fired**, because a record saying "quarantined" without saying what it saw cannot be audited. **Calibrated against 49 evidence items fetched live from our own feed: 0 false positives, 0 misses on 9 attacks** — and that corpus caught a real one, a rule that fired on "analysts please note the long-term outlook". The corpus is committed so a future rule that starts redacting real headlines fails in the suite. The screening note reaches the decision record on every cycle, including when nothing fired (`research/architecture/36-prompt-injection-quarantine.md`). |
| **Regime detection that sees shape, not just amplitude** | `desk/regime.py` | The incumbent (`strategies/track1_suite.py:206-215`) is one threshold: fast volatility under slow volatility means quiet regime. It cannot see a rally becoming a drawdown at the same volatility. FLUSS (Gharghabi et al., ICDM 2017) can, and it falls out of machinery we already had — extend `desk/shapematch.py`'s distance to every window's nearest neighbour and a regime boundary is where few of those arcs cross. Pure Python, no numpy or scipy: 1,079 bars in 10 seconds. Both reference implementations were read and the module follows each where it is better — matrixprofile's analytic parabola for the idealised curve (deterministic, scipy-free) and stumpy's five-times-wider head/tail correction (the narrow one mistakes edges for boundaries). **Live on NVDAUSDT: a +14.0% rally, a -6.4% drawdown and a +2.4% recovery — at 8.6, 10.5 and 9.3bps a bar.** The loudest stretch is 1.2x the quietest, and the incumbent rule does not flip at either boundary; the report re-runs that rule and says so itself. It labels nothing and predicts nothing — a homogeneous series is reported as one regime rather than forced into three (`data/regimes.json`, `research/architecture/35-regime-segmentation-fluss.md`). |
| **The maker-side assumption, finally measured** | `market/markout.py` | `cost/model.py` refuses to credit maker treatment at all, because our own replay turned a +90% simulation into -0.45% once resting orders were modelled honestly. Right call, and a blanket assumption: nothing had measured *how much* adverse selection costs here. The venue's public fills feed (`catalog.ts:118` — the second endpoint we had never called) carries the aggressor side and a millisecond stamp, so the book is polled at 1Hz and joined to the prints. **Live on MSTRUSDT, 150s, 80 prints: the passive side loses a median 0.8-1.3bps at 5-30s, worst print -8.4bps.** The sign convention is the whole module — a flipped sign is symmetric, plausible, and says resting orders get picked *up* — so five tests assert it from both directions. The verdict **refuses to change the cost model**: 150 seconds of one instrument with the anchor shut finds little informed flow and flatters the passive side by construction, and NVDAUSDT over the same window produced five prints and correctly concluded nothing. Sampling now runs on every scheduled cycle, and the schedule sits in US regular hours where the informed flow is (`data/markout.json`, `research/architecture/34-markout-and-adverse-selection.md`). |
| **A session throttle aimed at the measured hour, not the intuitive one** | `risk/session_risk.py`, `agents/desk.py` | Every tokenized-equity rival throttles *because the anchor is asleep*. Measured over 90 days, the shut window is the **calmest** part of the week — a weekend hour moves a fifth of a regular-hours one — and the risk is the **discontinuity at the end of it**: the first bar after discovery resumes moves **1.5-2.2x a regular-hours bar on all 12 instruments**, and 5.5-11.8x a weekend bar depending on the name, n=64 reopens each. So the throttle targets the volatility of the path a position will actually live through (variances add; the reopen bar is substituted for each transition crossed), capped at 1 because a risk layer may only reduce — on a quiet weekend the arithmetic would size *up*, and it is not allowed to. Live on NVDAUSDT before the open: **2h horizon x0.82, 8h x0.99, 24h x1.00** — the jump is one bar, so it dominates a short hold and vanishes in a day-long one. It is a new Constitution gate (`session_volatility`), placed after the hedge rule and before the absolute cap, with `eval/autopsy.py`'s chain updated in the same pass so attribution stays correct (`data/session_risk.json`, `research/architecture/33-session-volatility-throttle.md`). |
| **What a trade really costs, taken from the live book** | `market/depth.py` | The paper ledger charged the **quoted touch** on entry and a flat **0.6bps** on every exit. The quote is the price of an infinitesimal trade; nothing in the system had ever asked what it costs to take a real size. `GET /api/v3/market/orderbook` is public, keyless and up to 200 levels — it sits in Bitget's own SDK catalogue (`agent-sdk/src/generated/catalog.ts:115`) and `market/bitget.py` had never called it. Now the book is walked level by level for the notional actually being traded. **Measured live, slippage against mid: at $25,000 QQQUSDT costs 0.82bps and SQQQUSDT 19.43bps, against quotes of 0.14 and 4.96 — the quote understates the cost by a factor that varies 5x across instruments, which is worse than understating it by a constant because it reorders which names look cheap.** A size the visible book cannot absorb returns *no* measurement rather than the cheap half of a trade that could not be done, and the ledger then charges its documented stand-in. Both legs of every paper fill are now priced this way, and the scheduled cycle refreshes the table before each run (`data/depth.json`, `research/architecture/32-order-book-depth-teardown.md`). |
| **An allocator that prices its own rebalance** | `desk/allocation.py` | `desk/portfolio.py` could grade a trade it was handed and never propose one, so the shape of the book was whatever a sequence of individually-acceptable trades produced. Hierarchical risk parity now proposes it — chosen because these twelve instruments correlate above 0.9 and a mean-variance solution would invert a near-singular covariance into offsetting longs and shorts that are pure estimation noise. Pure Python, no numpy or scipy, and it **reproduces PyPortfolioOpt's weights to 3.6e-16** on the same twelve instruments, cluster order included (`tests/test_allocation.py`). What PyPortfolioOpt, Riskfolio and skfolio all stop short of is the decision: a weight vector is not one. `optimize_trade()` prices the turnover at 6bps a side, drops legs under 1% (a twelve-leg dust rebalance is how a book becomes a fee machine), and reports the **break-even holding period** — with its Sharpe assumption named in the same sentence every time, because ARGUS has no measured Sharpe. **Live from equal weight: volatility 22 -> 16bps per bar, 42.7% of variance, turnover costing 3.2bps, repaying in 43 bars.** The first pass of three where the answer was *yes, trade* — allocation is where the recoverable edge on this venue lives, because the fee that kills every intraday idea is irrelevant to a weekly rebalance (`data/allocation.json`, `research/architecture/31-hrp-allocation-teardown.md`). |
| **The risk layer's constants, replaced by measurements** | `risk/effectiveness.py`, `risk/hedgeability.py` | The hedge menu prices a candidate as a product of five factors, and three of them were literals typed into the source — `risk_reduction=0.98`, `correlation_confidence=0.98`, `basis_stability=0.95` — with no sample behind any of them, inside a project whose standing rule is *never guess*. They now come from the venue's own market-versus-index series: **Ederington (1979) hedging effectiveness** for the risk removed, the **lower bound of a 95% Fisher interval** for the confidence (a constant cannot express sampling error because it cannot know n), and unit-ratio effectiveness for the basis — the gap between the last two being exactly the sizing problem. Measured per session phase and never pooled: **regular hours 99.2-99.9% of variance removed, weekend 97.7-99.5%, pooled 95.3-98.1%**, so the old constants were too pessimistic awake and the pooled figure is an averaging artefact. Every candidate is stamped MEASURED or ASSUMED and carries its sample (`rth, n=251, 60d`); a measurement older than 36 hours counts as absent, and the cycle script refreshes it before each run. Nothing in the corpus does this — hummingbot's hedge ratio is a config field (`strategy/hedge/hedge.py:64,77`) and its stat-arb controller fits β on cumulative returns and then sizes with the config value anyway (`stat_arb.py:381-383,291`) (`data/hedge_effectiveness.json`, `research/architecture/30-hedge-effectiveness-teardown.md`). |
| **A pairs scan that counts its own tests** | `research/cointegration.py` | Track 1's *Arbitrage* sub-theme was half-covered: `research/arbitrage_study.py` had the rToken-vs-underlying basis, and nothing asked whether two rTokens share a stochastic trend. The ADF, Engle-Granger and MacKinnon tables are reimplemented in pure Python and **reproduce statsmodels 0.14.6 to 1e-12** on 10 unit-root tests and 3 cointegration tests over real hourly closes — including its lag selection, its sample sizes and its `nobs - 1` Stata quirk (`tests/test_cointegration.py`). What statsmodels does not do is the rest: the hedge ratio is fitted on a training slice and **frozen** before the out-of-sample spread is tested, z-scores use only prior data, a round trip is charged as **four** taker legs, and the 66 hypotheses a 12-instrument scan performs are corrected with the project's own Benjamini-Hochberg rather than a second copy of it. **Live: 66 pairs tested, 1 significant naively against 3.3 expected by chance, 0 surviving correction, 0 tradeable.** The verdict also names which hurdle bound — unlike the basis study, costs are not it: the median pair needs |z| >= 0.08 to clear 24bps, and what fails is stationarity (`data/cointegration.json`, `research/architecture/27-pairs-and-cointegration-teardown.md`). |
| **An analogue search calibrated against its own noise** | `desk/shapematch.py`, `desk/analogue.py` | Track 3 asks how the AI *retrieves historically similar scenarios*. Every implementation we tore down stops at a distance threshold — which is why every one of them can always show you five convincing analogues: scanning ~1,400 candidate windows for a minimum produces a small number whether or not the series repeats at all. ARGUS reruns the identical scan over 50 paths built by reordering the series' own returns (same bars, same volatility, ordering destroyed) and reports the share that match the present at least as closely. **Live on NVDAUSDT: five non-overlapping matches at distance 0.371–0.437, and the verdict is `no precedent` — reordered noise beats 0.371 in 36% of 50 scans (median 0.440).** Across 6 symbols × 4 window lengths, 22 of 24 cells fail the null. Reading `stumpy/core.py:1118` (`D² = 2m(1-ρ)`) also found our own threshold constant set to 2.0 — the maximum the metric can return, so the gate could never fire; it is now 1.0 = ρ 0.5, with a test pinning it below the uncorrelated level of √2. The exclusion zone is a **full window** where stumpy uses `m/4` (`stumpy/config.py:19`), so no two reported matches share a bar (`data/shape_matches.json`, `research/architecture/26-matrix-profile-and-analogue-retrieval.md`). |
| **Perpetual funding charged per settlement** | `cost/model.py` | rTokens fund every 8h, capped ±0.5%. Measured from the venue's history: non-zero on 14–30 of 100 settlements per symbol, mean 0.23–0.58bps, worst 5.8bps. A 24h hold crosses three — ~0.7–1.7bps typically, ~17bps in the tail against an 18.8bps hurdle. Kept as its own channel because funding's sign flips with the market and borrow's flips with your side, and a short *receives* it. Found by a competitive sweep; the rate had been fetched and discarded since the beginning. |
| **The desk's own reproducibility, measured** | `eval/consistency.py` | One market state put to the desk five times with the response cache disabled: **lean 100%, action 80%, verbatim 20%**. It does not reproduce its own decision — the first run took three different actions on identical evidence, one of which would have opened a position where the others abstained. The lean being identical while the action moves is the finding: the desk knows what it thinks and wavers on what to do about it. `temperature=0` was an assumption until it was measured. |
| **Search strategies ranked across seeds, not one run** | `research/searchoff.py` | A one-seed table ranks the draw. On a fixed bar window with only the seed varying, **no strategy holds first place in more than one seed** and four of five move two or more places — our own first published ordering put `anneal` first, and it is fourth in both other seeds. The sweep keeps per-seed ranks, names what moved, and refuses to rank at all from one seed. One result survives: `beam` is last in every seed and the only strategy with a negative mean out-of-sample score. |
| **Ranked by calibration, not by return** | `eval/observatory.py` | On a venue where the fee exceeds most effects, knowing what you do not know is worth more than a lucky quarter. Deterministic arithmetic, never one model grading another. |
| **Analyst selection priced against its own cost** | `agents/selection.py` | Nothing in the corpus conditions its panel on the evidence, and nothing skips an agent because running it is not worth the cost. Both are arithmetic here because deliberation already has a price. |
| **A factor searcher that cannot see its own scores** | `research/factor_lab.py`, `research/memory.py` | The memory publishes identity and structural facts only, with an import-time guard that raises if an outcome-bearing field is ever added to what the proposer sees. |
| **Claims checked against the evidence's own fields** | `agents/claims.py` | Numeric grounding catches a wrong figure. This catches a wrong *property* — and it exists because the live log produced one. |
| **Bitget's official Skills, measured** | `market/skills.py` | Every call is health-classified, and the Skill's own RSI is recomputed from Bitget candles before it is believed. |
| **The risk layer audited, not asserted** | `eval/riskaudit.py` | A layer that never fires is untested, not safe; one that fires on everything means the model is not deciding. The report grades both as findings rather than as a clean record. |
| **A baseline that isolates the desk** | `eval/baseline.py` | Not an index, which measures which universe was hot, but the desk's own picks held without turnover. The corpus's real-money LLM experiment computed an index comparison and then disclaimed it; nothing else has one at all. |
| **A grammar wide enough to be worth searching** | `research/grammar.py` | Our own factor audit against RD-Agent, Alpha Jungle and Qlib found the binding constraint was not the search but what there was to search *over*: **eight primitives over four fields, against Qlib's 158–360 features**. With close, one-bar return, time-to-discovery and basis, an expression cannot say anything about volume, about range, about where a price sits in its own recent distribution, or about the change in anything. The vocabulary now carries volume, high, low and intrabar range; six new window operators (`zscore`, `rank`, `argmax`, `argmin`, `median`, `slope`); and two new nodes — `delay`, without which *delta was inexpressible*, and `corr`, which is the only way to say "price moved with volume". Correlation canonicalises commutatively so one factor cannot be paid for twice. Nine factors that were previously unwritable are now in the sweep, which **raises the trial count and therefore deflates every Sharpe in the study** — the honest direction. |
| **A mandate the model reasons inside, not one applied afterwards** | `agents/mandate.py` + `agents/meta_pm.py` | Vibe-Trading is the only system of sixteen surveyed with code-enforced personalisation, and the audit found the one thing it does that we did not: it injects the user's mandate into the model's context **before** reasoning, where ARGUS narrowed the answer after. Both produce a compliant position; only one produces a *personalised thesis*, which is the criterion. The mandate now reaches the PM in its prompt — and the prompt says a recommendation that would read identically for any trader has not been personalised. The profile also grew the dimensions that can actually flip a verdict on this venue: `requires_hedge` (the hedge menu is empty 65 hours a week, so a conservative mandate declines exactly the weekend positions an aggressive one takes), named exclusions, a conviction floor and a concurrent-position cap. Each is a **refusal, not a resize** — half an unhedgeable exposure is still unhedgeable. |
| **One question, end to end — the deliverable the handbook names** | `desk/research.py` | Track 3 requires "one complete research task (full flow from question to actionable insight)". Our own audit found the honest position: six strong subsystems and nothing that ran them in order, so answering *"should I add NVDA to this book?"* meant calling seven modules by hand. The orchestrator is that seventh step — evidence → expectation gap → analogue → session beta → portfolio impact → stress → execution cost → recommendation. Live on a real book: **ADD_SMALLER, because open-session beta rises 0.309 → 0.457**, with 5 of 7 steps run and each figure naming the module that computed it. Two rules keep it honest: the verdict is assembled by deterministic rule, never narrated by a model — a summary is where an unsourced number enters a cited document — and a step that could not run prints as a named absence, so a narrower analysis can never read as a complete one. Below four of seven steps the verdict is INSUFFICIENT, not a soft decline (`data/research_report.json`). |
| **Which search method actually finds robust alpha** | `research/searchoff.py` | Our Track 1 result says what the search found; nothing said how it searched, because there was one search — enumerate a fixed list. Five strategies over the same grammar, same data, and an **equal evaluation budget**, which is the experiment: a method that tries more finds a better maximum of the noise, so `budget_was_equal` is checked rather than assumed. Trials count per canonical expression, so rediscovering a tree is not exploring. Winner picked in sample, reported out of sample. On 1,487 hourly bars of NVDAUSDT at 400 evaluations each, **the in-sample and out-of-sample rankings are close to inverted**: evolutionary found the best in-sample score (0.0381) and nearly the worst out-of-sample (0.0084), while beam — pure exploitation — had five times the decay of the best and went negative. Novelty, which ignores the objective and cannot overfit to it, had the second-best decay. It hung on first run: depth was capped and cost was not, and `Window`/`Delay` nest multiplicatively — one tree cost 331,776 operations per bar against a median of 1. A cost guard now refuses those and still charges a trial. |
| **Six headlines, or one headline six times** | `agents/novelty.py` | Sentiment is the sub-theme our own register DEMOTES, and its docstring blames the feed. That is true and incomplete: a sentiment signal built on an uncounted pile of syndicated copies measures republication volume. The independence graph discounted *analysts* who read the same article; nothing looked at the evidence itself. Near-duplicate clustering by four-word shingles and exact Jaccard — not MinHash, which approximates Jaccard for corpora too large to compare pairwise, where a decision cycle sees tens of items. Three signals, because duplication alone is innocent: distinct stories, coordination (several *distinct sources* inside two hours — one outlet repeating itself is a busy newsroom), and velocity. A promoted narrative can be true and the verdict says so rather than implying a trade. Wired into the live evidence step, which now flags the overstatement factor. On the real feed the answer is an honest negative: **6 items, 6 distinct stories, ratio 1.0.** |
| **A correlation-aware bet count, and a hedge that is named and sized** | `desk/diversification.py` | Track 3's Open Theme asks for concentration and hedge suggestions. Our own docstring said the correlation-aware count was NOT BUILT, and the inverse Herfindahl it used scores three perfectly correlated positions at 2.71. Meucci's measure is now built under **both** rotations, because each is blind where the other is informative: on an equally-weighted equicorrelated book the principal-component reading collapses from 4.00 to 1.00 at a correlation of 0.0001, while minimum torsion reads 4.00 all the way to 0.99. Both are reported and the gap between them is the signal that the book is correlated. The eigensolver is cyclic Jacobi in pure Python, verified against `numpy.linalg.eigvalsh` to 3.55e-15. Hedges are sized at `-cov(p,h)/var(h)` with the reduction equal to the squared correlation — both from one covariance, so the promise cannot disagree with the recommendation. Searched 2026-09-13: Riskfolio-Lib's "NEA" is an inverse Herfindahl with the same blind spot, and nothing in any tree sizes a hedge from a covariance. Live (book `NVDAUSDT=0.4,TSLAUSDT=0.3,METAUSDT=0.3`, 30d hourly): **1.1185 effective bets across 3 positions** — three names that are barely more than one bet — **torsion 2.9404, best hedge TQQQUSDT at -0.372 removing 45% of the variance** (`data/diversification.json`, `python -m argus.desk.diversification --save`). |
| **ARGUS-BENCH — the five Open Theme criteria in one scorecard** | `eval/bench.py` | Track 2's Open Theme names its own example: decision consistency, risk-violation rate, stress behaviour, human-takeover rate, incremental value over a fixed-rule baseline. A source-level teardown of every harness in the corpus (`research/architecture/agent-evaluation-harnesses.md`, searched 2026-09-13) did not find human-takeover measured anywhere, and did not find any fixed-rule baseline carrying a statistical test. Run against its own author before anyone else: **2 pass, 2 weak, 1 undefined**. UNDEFINED is a result and is never conflated with zero — a desk that has never traded has no takeover rate, rather than a perfect one. |
| **Incremental value over rules you could write on a napkin** | `eval/incremental.py`, `eval/collect.py` | Five fixed rules — flat, always-long, momentum, reversion, volatility-gated — replayed over the same 42 instants the desk faced, scored on the realised move net of the same hurdle, compared pair by pair with an exact sign test. **Four of the five lost money.** Momentum reached 43% directional accuracy and the volatility-gated rule 46%, against the 56% break-even the hurdle frontier computed, so no simple rule clears the bar. Nothing separates from the desk at 5% on 25 effective instants, and the report says that is a sample too small to settle the question rather than a pass. |
| **A desk that decides what it will not decide alone** | `decision/escalation.py` | Five deterministic conditions force a hand-over: an unresolved directional split, a thesis already refuted by its own stated falsifier, an unsourced figure, a halted underlying, and size beyond what the mandate permits unattended. Wired into the live path between the adversary and the Constitution. It may only reduce — quantity goes to zero, the side cannot flip, and `apply` raises rather than logs if it is ever asked to do more. The takeover rate's denominator is *actionable* decisions, because dividing by all decisions would let a desk that abstains constantly report a rate near zero while escalating everything it actually did. |
| **A standing adversary that attacks the thesis, not a side** | `agents/adversary.py` | TradingAgents runs a Bull against a Bear for N rounds. Assigning a model the bull side produces a bull case whether or not one exists — debate as theatre, least informative exactly when the evidence is one-sided. Ours is given no side: it receives **the decision that was actually made**, its thesis, the falsifiers the thesis itself stated, and the same evidence, and answers three questions — the strongest counter-case, which single piece of evidence the conclusion leans on hardest, and **whether any invalidation condition the thesis named is already true**. That third check is performed by nothing in the reference corpus: the Constitution reasons about size, grounding about whether figures resolve, conflict about whether analysts agreed — none reads the falsifiers back against the evidence. A thesis refuted by its own stated falsifier is killed. And a critic cannot veto by inventing a condition: a "falsifier" the thesis never stated is ignored. Like everything downstream it may only reduce — 30 tests, and `apply_constraint` raises rather than logs if it tries to do more. |
| **A console that answers in the language it was asked in** | `lui/phrasebook.py` | Seventeen Chinese patterns routed questions correctly and every answer came back in English, which a live check on the deployed console found and the documents had overstated. 44 phrases now carry both languages with no model in the loop — a translation step that called a model would break the console's defining property, that it answers with no key. The tests enforce three things: English output is byte-identical to what it said before, no Chinese slot may hold the English string, and every template carries the same format fields in both languages so a translated line cannot drop a number. Symbols, hashes and verdicts stay literal, because those are what a reader goes and checks. Live: `记录完整吗` returns 哈希链在 227 条记录上完整。 (`data/paper_ledger.jsonl`) |
| **The console is live, public, and works with no key** | `deploy/` + `lui/server.py` | Hosted at **https://deploy-topaz-seven-64.vercel.app** — no login, no key, works on a phone. `lui/server.Handler` is already a `BaseHTTPRequestHandler`, so the deployment is a nine-line adapter rather than a second implementation: a judge on the URL drives exactly the code the CLI drives. **No credential is deployed** — `build_router` returns `None` without a key and the console falls back to its deterministic layer, which is the state a judge is most likely to meet once a hackathon balance is spent. That is why `tests/test_phrasings.py` asserts the patterns alone understand 95% of how people actually ask. Driven live as a stranger: the page renders, `/status` reports 106 decisions with the chain intact, "why did you do nothing?" and "is the record intact?" answer, "sell half of that" is refused as an instruction, and "what about gold?" is refused by name. |
| **Cost priced once, at the decision** | `agents/analysts.py` | For 105 live decisions every analyst returned `neutral` and the desk never traded. The cause was not the shut market: the analyst preamble told each of them that an effect under 12bps "is a loss", so they reported real moves as no move — and the Meta-PM then applied the full 18.8bps hurdle again. The cost was charged at two layers and the decision-maker never heard the signal. Analysts now **measure**: direction and magnitude, however small. The desk **decides**: cost and deliberation priced once, where the trade-off belongs. Verified live — the same analyst that had never produced a directional call returns `bullish, 12bps` and `bullish, 45bps` on a small genuine surprise. |
| **A panel that is independent by construction, not by assertion** | `agents/desk.py` | The desk used to tell every decision that its analysts' agreement "may be contagion, not consensus", because they were called one after another. Reading the code settled it: each analyst gets a disjoint slice of the evidence and `Analyst._ask` builds a fresh two-message conversation per call, so there was never a channel through which contagion could occur — the desk was **understating its own independence on every row of the ledger**. The panel now runs concurrently, which removes the last real coupling (the order in which a shared token budget is consumed) and makes independence structural: a thread cannot read a result that has not been returned. Results are resolved in a fixed order so concurrency never makes the record depend on scheduling, and **one analyst failing is recorded as a named absence rather than taking the panel down**. Live: 3 analysts in 26.1s against roughly 78s in sequence (`data/desk_notes.jsonl`). |
| **A dead data source replaced from the issuer** | `market/macro.py` | Bitget's `macro-analyst` Skill exposes a `rates_yields` tool that the handbook names as part of this track's perception layer. Probed live it returns every tenor as an error envelope and then a spread of `0.0` — a yield curve of zeros derived from legs that never arrived, which `skills.hollow()` now correctly refuses. So the curve is taken from its publisher instead: the US Treasury's daily par yield XML, keyless, 175 dated observations for 2026, twelve tenors. Every curve is dated **at the issuer**, not at fetch — a Friday curve read on a Sunday is still Friday's information — and one older than seven days produces no evidence at all rather than a stale rate that reads like a current one. A missing leg yields `None`, never zero, because a flat curve is a real state and must stay distinguishable from an absent one. Wiring it in took the live panel from **2 analysts and 5 distinct sources to 4 analysts and 9**. |
| **The venue's own rules, enforced before we ask** | `execution/guard.py` | Nautilus denies an order in-process with a named reason; Hummingbot locks collateral before submitting; the owner's own Nomos hit three 10x sizing bugs fixed only by reading tick size and minimum notional live from the venue. ARGUS had none of it — `preflight.py` checks the session, nothing checked an order. The guard fetches `GET /api/v3/market/instruments` (787 instruments, live, keyless) and enforces what Bitget publishes: step size, precision, `minOrderQty` 0.01, `maxOrderQty` 52,000, `minOrderAmount` 5 USDT, the ±2% price band, balance, and a sliding-window submission cap. Sizes round **down**, never to nearest, because rounding up can cross the ceiling just checked — so like the Constitution it may shrink an order or refuse it, never enlarge one. Reading the spec also settled a number we had only asserted: the venue publishes `takerFeeRate 0.0006`, so a round trip is exactly the 12bps our cost model charges, and a test now pins the two together. |
| **All six judged Track 1 numbers, including the one that exposes the others** | `backtest/metrics.py` | The handbook scores Sharpe, Sortino, max drawdown, turnover, out-of-sample decay *and* rolling 30-day Sharpe stability. The study reported two of those six; the rest were computed on every run and discarded at the last step, and rolling stability did not exist at all despite being promised in a docstring. All six are now reported per symbol — and the sixth earns its place immediately. **METAUSDT is the best-looking row in the table: a net Sharpe of 1.815 after the 12bps round trip, +39.8% return, and it beats buy-and-hold. Its rolling 30-day Sharpe averages −0.19, is positive in only 47.4% of 4,289 windows, and ranges from −47.6 to +57.9 — a spread of 105 Sharpe points.** A headline number that good sitting on a rolling series that spends more than half its life negative is exactly what a single full-sample Sharpe is built to hide. Across the twelve symbols: **9/12 beat buy-and-hold, 8/12 survive the cost sweep to 40bps, and 0/12 survive the deflated Sharpe on all trials** (1/12 on the weaker candidates-only gate). Every formula was audited against empyrical, vectorbt and the Bailey & López de Prado paper (`research/architecture/metrics-audit.md`): no formula errors, and the two places where reference libraries disagree with each other are documented and pinned by test (`data/track1_study.json`). |
| **Every decision on one page, with its gaps printed** | `eval/decisioncard.py` | The trail behind a decision is written by four different parts of the system into four files; reading it meant joining them by hand. A card joins them by sequence — what was decided, why, what the checkers found, what the risk layer did, which pre-registered rules were in force, what happened next, and the hashes — and names the file every part came from. Nothing is recomputed: a page that recalculated a number would be a second opinion wearing the ledger's authority. Absence is printed in words, because a blank section reads as "nothing to report" and that is indistinguishable from "nobody looked"; an unsettled decision in particular never renders as a correct one. 102 cards written (`data/cards/`). |
| **The risk layer proved over its whole domain, not sampled by the market** | `eval/riskproof.py` | Track 2 scores risk-control effectiveness, and ours had fired zero times in twenty live records — every decision so far has been an abstention, which proposes no exposure to narrow. A risk layer is a function, so it is proved instead of awaited: **6,720 states** — every combination of verdict, side, size straddling each limit, confidence straddling the floor, hedge availability, NAV staleness and session phase — each checked against the four properties that make it a risk layer. It never increased exposure, never reversed a side, never turned an abstention into a trade, and never changed anything under ALLOW. **All five rules bind somewhere, so none is shadowed**, and the precedence between overlapping rules is derived from the sweep rather than read off the source (`data/risk_proof.json`). The harness is itself tested against deliberately broken policies — one that doubles the position, one that flips the side, one that swallows every state — and must catch each, because a prover that cannot fail proves nothing. |
| **A checklist that has to earn its place, and can lose it** | `desk/review.py` | Published trading checklists grow monotonically: an item is added after each painful episode and never removed, until the list is too long to read and prevents nothing. Ours applies the factor lab's evidentiary standard to *process rules*. Each rule is replayed over the decisions already taken and graded on how selectively it fires and how often firing coincided with a defect an independent checker found — never with the checker's own verdict, which would be grading its own homework. **All five standing rules currently fail**: three are wrong more often than right, two target a defect that has never occurred and are therefore ungradeable rather than wrong (`data/review_report.json`, 40 decisions, 62% clean). That is the finding. A checklist nobody tested is worth less than one that failed honestly. |
| **Stress scenarios taken from what the instrument actually did** | `desk/stress.py` | Every published stress table uses round numbers — −5%, −10%, −20% — that nothing about the instrument produced, so a reader cannot tell a Tuesday from a once-a-decade event. Ours derives each shock from the realised distribution of that instrument's own moves, conditioned on session phase, and carries its frequency: NVDA's extreme regular-hours case is **−2.58% over two hours, the 1st percentile of 390 observed windows, reached 4 times in 90 days, most recently 2026-08-28** (`data/stress_report.json`). Structural conditions the price series cannot express — the hedge market shut, the venue dark, evidence stale — are tested beside them and each states its assumption; three say NOT MEASURED outright, and the report prints what share of its scenarios are measured. A phase with fewer than 30 windows yields no percentile rather than one blended across sessions. |
| **A language console where the model routes and the ledger answers** | `lui/router.py` | The patterns cover the phrasings we anticipated; a judge types the ones we did not. When classification comes back "not understood", a model is asked one question — *which kind of question is this?* — and returns a label. Every word the user then reads is produced by the same deterministic answerer reading the same artefacts, so a routed answer cannot contain an invented number, thesis or date. Refusals that should stay refusals do: an instruction to trade is never re-read, low confidence keeps the refusal, and `why did you do that?` stays ambiguous unless a decision or symbol was actually named. Without credentials the console answers from patterns alone rather than failing to start. |
| **Truncation told apart from malformed output** | `llm/qwen.py` | A cut-off answer and a badly formatted one raise the same `JSONDecodeError` and need opposite treatment. The wire says which: `finish_reason == "length"`. On truncation the budget grows and the *original* prompt is re-sent; on a malformed but complete answer the specific error is fed back. A truncated decision is never repaired — completing a half-written thesis would read as reasoning the model never did. Across the 114 agent repositories surveyed here, none consult `finish_reason` before parsing. |
| **A trading protocol pre-registered before the outcomes** | `paper/protocol.py` | The universe, the tradeable sessions, the hurdle *formula*, the size ceiling, the hold period, the settlement rule, the falsifiable hypothesis and the weaknesses we already know about are frozen into a SHA-256 and bound to the ledger at the moment of commitment — head `4c382231cf77e3d3`, 94 entries, so protocol v1 governs seq 95 onward. A judge does not have to believe the rules predate the trades; the arithmetic says so. Amendment is allowed and must be visible: a change is a new version with its own commitment, and the old one keeps governing the decisions taken under it. The gate is applied last in every cycle and may only reduce — it can refuse a trade or cut its size, never create or enlarge one. |
| **Documents that check themselves** | `eval/docclaims.py` | Every figure the README, submission draft and explainer quote (test count, module count, symbols beating buy-and-hold, DSR survivors, session betas, hurdle clearance, arbitrage rate) is matched against the artefact that produced it, at the precision the prose used. A quoted "0.67" against a live 0.6677 passes; "nine of twelve" against a live 11 fails the test suite by document and line. First run found six stale figures in three documents, including this README. |
| **Beta that knows which session it is** | `desk/portfolio.py` | Open-session beta exceeds shut-session beta on 10 of 11 rTokens; AAPL reads 0.68 open against 0.105 shut. A single blended beta is dominated by the ~82% of bars where the anchor market is shut, so it understates exposure in the only session that prices the underlying. |
| **Risk contribution, not notional share** | `desk/portfolio.py` | On the live book TSLA is 32% of the notional and 51.5% of the risk. The decomposition is verified to sum to portfolio volatility on every call, because one that does not add up is arithmetic, not risk. |
| **Stress the book, not each position alone** | `desk/portfolio.py` | A shock reaches every holding through its own beta, so the co-movement is carried rather than assumed away. Beside it, the worst run this book would actually have had, replayed from realised returns — none of PyPortfolioOpt, cvxportfolio, qlib or riskparity.py implements that. |
| **Personalisation that can fail** | `desk/personalisation.py` | One market state, every profile, and a harness built to report "personalisation did not bind here". On the four shipped proposals one diverges and three do not. The mandate is wired into the desk and may only refuse or shrink. |
| **The expectation gap, with the surprise refused** | `desk/expectation.py` | 42 analysts expect NVDA to grow 90% against 128% and 214% delivered, while revising up 35 to 1. No keyless source publishes the consensus that existed for a reported quarter, so the classic surprise is reported as unavailable rather than manufactured from a forward estimate that was never about it. |
| **What the market expects, not only what was reported** | `market/estimates.py` | Consensus EPS, dispersion and the 7/30-day revision trail, keyless. Built because the desk itself wrote that fundamentals cannot be judged without it. |

---

## The honest part

- **Zero positions have settled.** All 447 ledger entries are refusals, and 386 have settled as
  abstentions. Two rows (seq 264, 265) carry `verdict: trade` and are **void** — they record fills
  the risk layer had refused, and are excluded everywhere (see the correction at the top of this
  file and `paper/corrections.py`). Track 2's quantitative half — Sharpe, max drawdown, win rate —
  is computed from settled trades, so those three numbers do not exist. `eval/performance.py`
  returns `null` and the reason, rather than printing `0.0` and letting it read as a flat result.
- **The `side` field carries no information, and that was measured rather than assumed.** Over the
  53 abstentions settled at the time of the measurement it was `BUY` in **all 53**, directionally
  right 3.8% of the time against a base rate of up-moves of exactly 3.8% — a schema being filled in,
  not a view. So the desk now states a **lean** on every decision including the ones it refuses,
  covered by the chain through the intent hash and graded against the move that followed
  (`eval/shadow.py`). **282 decisions carry a lean** so far — already UP and DOWN at differing
  confidences, where `side` never varied — and none has settled, so the record says UNDEFINED rather
  than falling back to `side` to produce a number today.
- The abstentions are correct rather than broken: the log begins on a weekend with the anchor market
  shut, and every thesis cites the session, the hours to price discovery, and a move below the total
  hurdle. Measured separately, the hurdle is cleared 61–89% of the time during US regular hours, on
  **more than half of regular-hours bars for eleven of twelve symbols** — QQQUSDT, the calmest
  of them, is the one that does not (`data/hurdle_clearance.json`, regenerate with
  `python -m argus.eval.clearance --save`).
- Of our eight vetted factor primitives, **not one survives** (`data/overfit_gates.json`,
  regenerate with `python -m argus.research.overfit_gates --save`). Six are rejected by the
  anti-overfit gates as indistinguishable from shuffled data; the remaining two return *"not proven:
  insufficient data for a verdict"* rather than a pass. Every one of the eight is also net-negative
  after the 12bps round trip. `data/factor_lab.json` reaches the same verdict by a different route
  — its own DSR-and-fee funnel kills all eight independently.
  ⚠️ **This claim was rewritten on 2026-09-15 and the old wording is recorded rather than
  quietly replaced.** It previously split the eight four-and-four — half called noise, half
  called real-but-unprofitable — citing an artefact that **no code could regenerate**. Rebuilding the measurement did not reproduce that 4/4 split, and the bucket size the
  gates need was not recoverable from the file. The honest result is harsher than the claim it
  replaces, and unlike it, anyone can re-run it.
- Of Bitget's nineteen official Skill tools, **six carry data** — all of them `technical_analysis`.
  With a timeout long enough for the slow ones (15–31 s, measured), the other thirteen do answer:
  ten are hollow (every leg an error envelope, some with derived zeros such as a yield-curve
  spread of 0.0 built from tenors that never arrived) and three report a tool error. So all
  **five Skills are reachable and one is usable** (`data/bitget_skills_health.json`). A hollow
  payload used to classify as OK; `skills.hollow()` now keeps it out of the evidence. The cycle
  calls the six that carry data.
- **Twenty-three capabilities are OWNED; one more reaches twelve of thirteen and stops there for a
  reason that is named, not argued away.** OWNED means demonstrated superiority through a run
  experiment against a named baseline, and it is a conjunction of thirteen conditions rather than
  a score — see `eval/standing.py`'s own register for the current, falsifiable count (`audit()`
  re-derives it from the artefacts, never from this paragraph). Queue-position modelling
  (`eval/queueproof.py` against `nkaz001/hftbacktest`) has twelve: the port reproduces the reference
  to **1.1e-16** on identical inputs, and against a queue whose truth is known by construction a
  probability function beats the best naive constant in **3 of 5 generative regimes swept rather
  than chosen** — in sample and again out of sample, with the in-sample winner carried over instead
  of re-picked, every margin's whole 95% paired interval above zero. It **loses 2 of 5** and both
  are published: where the front of the queue never cancels, "every cancellation is behind you" is
  not a naive assumption but the exact truth, and no member of the family can express it. One result
  cuts against it outright — in one regime it estimates the queue better and still costs more.

  The thirteenth condition is **not met and is named rather than argued away**: hftbacktest
  validates against recorded real market data including market-by-order feeds, while we validate
  against a simulator whose attribution rule we stated. Bitget's public API publishes L2 only, and
  which side of a resting order a cancellation came from is not recoverable from L2 at any sample
  rate — closing it needs our own orders resting in the real book, which is elapsed time rather than
  more code. So the honest word stays **implemented**. `eval/bookcalib.py` records the real book on
  every cycle so the simulator's parameters at least stop being ours; the first tape already
  disagreed with them, putting a near-touch level at a **median 1.95 contracts** turning over
  **~70% of itself a minute**.

- **The constitution chain has now been reached, and the first gate that ran refused the trade.**
  Across all 170 recorded decisions gate 1 (`no_exposure`) returned every time, so gates 2–7 had
  never executed — not because the risk layer is permissive but because every directional study here
  measured no edge, so the desk correctly proposed nothing. `research/carry.py` supplies the missing
  input, because carry is not a direction. Live (`data/carry_desk.json`): **gate 1 PASSED for the first time, gate 2 FIRED**,
  gates 3–7 remain UNREACHED and are reported as unreached. This is an **assessment against the
  real `ConstitutionPolicy`, not a ledger entry** — the trade was refused, so nothing was booked and
  the ledger still holds zero positions. The stated confidence is a Wilson lower
  bound on the share of profitable holding windows computed against the **independent** window count
  — 1,823 overlapping 14-day windows out of 90 days are about six independent holds, where the bound
  is **0.371** against a 0.55 floor. Quoting 1,823 would have given 0.72 and cleared it. Gate 1 was
  passed by supplying an honest proposal, never by loosening the gate.

- **Four modules existed, were tested, and ran on nothing — and one parser was inventing prices.**
  Found by asking "what imports this?" rather than "does this pass?". `paper/chains.py` graded the
  three-to-five checkable claims every event decision makes, and **no line in the runner called it**,
  so its own docstring ("constructed, counted, described as checkable, and dropped on the floor")
  still described the live path. `risk/circuit.py` and `risk/sizing.py` were connected to each other
  only in a docstring — `sizing.size()` had no caller anywhere in `src/argus` — so the breaker could
  not change a decision, which is the bar its own docstring sets. `market/validation.py` was the
  "verified behaviourally" in `bitget.py:41` and was never run. All four are now in the live path,
  and the universe re-verifies every cycle — live, all twelve rTokens still attenuate 3.80x-8.54x.

- **The Constitution's new seventh gate sizes an unproven desk as unproven.** `risk_budget` caps a
  position at `equity x fraction`, where the fraction is half-Kelly on stated confidence **only if
  calibration has been earned** — 20+ graded outcomes at an expected calibration error under 0.15 —
  and the circuit breaker's drawdown ladder multiplies it down (x0.75 / x0.5 / halt). The record
  holds 183 decisions, 69 settled and **zero** gradeable outcomes, so the gate refuses to size on
  confidence and falls back to 5% of book. Live, a 19,000 request becomes 5,000 with the reason
  attached: *"fixed fraction; confidence not usable (0 graded outcome(s), 20 needed)"*. Before this
  gate the only size limit was a flat 50,000 cap, and nothing enforced anything tighter.

  The parser is the one that would have cost money: a price field the venue omitted became
  `Decimal("0")`, and `Ticker.spread_bps` reads a non-positive bid or ask as **a spread of zero** —
  the tightest possible book at exactly the moment we know least about it. Every hurdle would have
  looked clearable. Prices now raise on absence; zero stays legal only where it is a real reading
  (funding, volume, change). Verified against the live venue: 786 rows in, 786 parsed, none rejected.

- **The funding study's own headline is a warning.** The basket paying the most funding (+3.94% a
  year, interval clear of zero) **loses money when actually held**, in three windows out of four:
  the funding arrives and the price leg gives back more. A funding table alone would have
  recommended precisely that trade. And the basket that does pay should not be called a carry — only
  21% of its return is funding and the rest is volatility decay, a short-gamma position with a
  −3.30% worst window (`data/carry_study.json`).

---

## Evidence on disk

Every file below was produced by running something, not by writing it.

| Artefact | What it records |
|---|---|
| `data/paper_ledger.jsonl` | Every decision, hash-chained, written before the outcome exists |
| `data/paper_ledger_incidents.json` | The one time the chain was repaired, with the original archived and its SHA-256 committed |
| `data/desk_notes.jsonl` | What the grounding, claim, conflict and panel checks found, per decision |
| `data/overfit_gates.json` | Four anti-overfit gates over our own primitives on 2,159 real NVDA bars |
| `data/hurdle_clearance.json` | How often the 2-hour move clears the total hurdle, per symbol, per session |
| `data/bitget_skills_health.json` | Which official Skill tools answered, and which did not |
| `data/skill_crosscheck.json` | The Skill's RSI against ours on the same instruments |
| `data/factor_memory.json` | Every hypothesis the factor lab has evaluated, so it never pays for one twice |
| `data/risk_records.jsonl` | What the risk layer did to each decision, so its effect can be counted |
| `data/session_beta.json` | Beta per rToken in the open session, the shut session, and blended |
| `data/risk_proof.json` | Every state the risk layer was swept over, what bound, and what never did |
| `data/review_report.json` | Observed process defects, and what each checklist rule earned on replay |
| `data/research_report.json` | One research question answered end to end, every figure citing its module |
| `data/stress_report.json` | One trade idea stressed against 90 days of its own history and five structural conditions |
| `data/doc_claims.json` | Every quoted figure in the documents, its live value, and OK/STALE by line |
| `data/protocol_commitments.jsonl` | The pre-registered trading protocol, its digest, and the ledger position it binds to |
| `research/architecture/*.md` | 61 code-level teardowns of the systems this was built against, each citing `file:line` |
| `data/flow_trace.json` · `data/flow_trace_scenario.json` | event → decision → execution as one trace, linked by the hash of the approved intent |
| `data/abstention_autopsy.json` | Which Constitution gate refused each decision, and which never executed at all |
| `data/cycle_check.json` | Nine properties of the last scheduled cycle, verified from its log and the ledger, PASS/FAIL/UNKNOWN |
| `data/consistency.json` | One market state put to the desk five times with the cache off: does it decide the same thing twice |
| `data/architecture.json` | The import graph: contract violations, cycles that can break at import, and per-package instability |
| `data/lui_bench.json` | 40 questions asked as a trader would, scored for understanding, refusal in both directions, paraphrase and EN/ZH parity |
| `data/theme_audit.json` | Every named Track 2/3 sub-theme and judged criterion, executed against a bar written in advance |
| `data/shadow_record.json` | The desk's directional lean on every decision, graded against the move that followed |
| `data/search_sweep.json` | Five search strategies over three seeds, with the per-seed ranks kept rather than averaged |
| `data/latency_probe.json` · `data/bakeoff.json` | Measured venue latency; model bake-off |

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
pytest              # everything
pytest -m leakage   # only the look-ahead suite
```

Every test in `tests/test_leakage.py` corresponds to a defect found in a real, widely cited system,
with the citation in the docstring. If one fails, ARGUS has acquired a bug that already ships
elsewhere — and the citation says where to look for the shape of it.

Several test modules are regression fixtures for defects found in **this** system by running it:
`test_ledger.py` reproduces the two-writer race that duplicated seven sequence numbers,
`test_claims.py` keeps the hallucinated thesis from ledger entry 41 verbatim, and
`test_runner_notes.py` pins each check's own wording so a reworded check fails a test instead of
silently ceasing to be reported.

---

## Licence

MIT. Third-party components and their dispositions are recorded in
[`../research/architecture/_CONSOLIDATED-LEDGER.md`](../research/architecture/_CONSOLIDATED-LEDGER.md).
Nothing under GPL, LGPL, PolyForm Noncommercial, or an absent licence is vendored here.
