# LLM-Trading-Lab & ai-hedge-fund — Architecture Teardown

**Target for:** Bitget Hackathon Track 2 (Agentic Trading) — ARGUS agent development
**Format:** matches `tradingagents.md` — code-level, constant `file:line`, blunt about defects.

**Repo 1:** `research\corpus\repos\LLM-Trading-Lab` (LuckyOne7777/ChatGPT-Micro-Cap-Experiment)
**License 1 (exact, `Other/License.txt`):** MIT License, "Copyright (c) 2025 Nathan Smith"

**Repo 2:** `research\corpus\repos\ai-hedge-fund` (virattt/ai-hedge-fund)
**License 2 (exact, `LICENSE`):** MIT License, "Copyright (c) 2024 Virat Singh"

---

# PART A — LLM-Trading-Lab: Evaluation Methodology

## A.1 What was run, over what period, with what capital, how decisions were recorded

**What:** ChatGPT (interactive web UI, not API) acting as sole portfolio decision-maker over U.S.-listed micro-cap equities (market cap ≤ $300M), full shares only, long-only, no derivatives, no margin (`evaluation_report.md:46`, `:244–258`).

**Period:** 2025-06-27 → 2025-12-26, six calendar months (`evaluation_report.md:82`, `:46`).

**Capital:** $100 starting equity (`evaluation_report.md:46`; baseline row hardcoded as `Date=2025-06-27, Total Equity=100.0` in `graphing/equity_vs_baseline.py:23-24` and `trading_script.py` Appendix A.1.1).

**How decisions were recorded — two parallel CSV ledgers, human-operated:**
- `Trade Log.csv` — discrete buy/sell events (Date, Ticker, Shares Bought/Sold, Buy/Sell Price, Cost Basis, PnL, Reason) — schema at `evaluation_report.md:385–396`.
- `Daily Updates.csv` — end-of-day snapshot per ticker + a `TOTAL` row (Date, Shares, Buy Price, Cost Basis, Stop Loss, Current Price, Total Value, PnL, Action, Cash Balance, Total Equity) — schema at `evaluation_report.md:370–383`.

**The recording loop is code, not just a spreadsheet.** `process_portfolio()` (`scripts/processing/trading_script.py:490–767`) is interactive: it prints the current book, offers `b`/`s`/`u` (buy/sell/update-stop), executes manual buy/sell orders (MOO at `data["Open"]`, `:556` `notional = exec_price * shares`; limit via `log_manual_buy`/`log_manual_sell`, `:807`, `:917`), then walks every held ticker and **enforces stop-loss automatically in code**: `if stop and l <= stop: exec_price = round(o if o <= stop else stop, 2)` (`:693–694`), forcing a `SELL - Stop Loss Triggered` regardless of what the LLM says next. This is one real piece of code-enforced risk control inside an otherwise fully-discretionary system — the model sets the stop level, the code (not the model) pulls the trigger.

**Point-in-time discipline is a human/process control, not a code guard.** There is no `ASOF`-gated data feed forcing the LLM to see only past data (the `ASOF_DATE` override at `trading_script.py:44–56` exists for *replaying* history, not for blinding the live experiment). The actual guarantee is procedural: "the model was explicitly constrained to rely solely on the provided input and was prohibited from accessing external or future information" (`evaluation_report.md:62`), enforced by the human operator controlling what goes into the prompt, not by any code path.

## A.2 The Baseline

**Quote, direct:** *"Benchmarks such as the Russell 2000 and the S&P 500 were included to provide contextual reference for broad market conditions during the experimental period rather than to evaluate relative performance… all conclusions in this evaluation are derived from portfolio behavior, decision patterns, and realized outcomes, and are **not contingent on benchmark performance comparisons**."* (`evaluation_report.md:149`)

So the paper's own conclusions section explicitly **does not use** a baseline. There is code that computes one, but it is decorative, not evidentiary:

- **Buy-and-hold S&P 500 / Russell 2000, index-level, $100-normalized.** `graphing/equity_vs_baseline.py:29-44` (`download_baseline`) pulls `^SPX` and `^RUT` closes over the same window and scales to a $100 start (`scaling_factor = starting_capital / starting_price`), plotted against the portfolio equity curve (Figure 1). This produces the headline fact that the portfolio underperformed both benchmarks (`evaluation_report.md:151`), but it is a chart annotation, not a statistical test.
- **CAPM alpha/beta vs `^GSPC`**, computed daily in production: `trading_script.py:1140-1163` — `beta, alpha_daily = np.polyfit(x, y, 1)` on excess returns, `alpha_annual = (1 + alpha_daily) ** 252 - 1`, with `r2 = corr(x,y)**2` and `n_obs` reported. This is a genuine quantitative baseline (market-model alpha), but the daily console output itself flags it as unreliable in the small-sample regime the experiment actually ran in: the daily-report template hardcodes `"Note: Short sample and/or low R² — alpha/beta may be unstable."` (`evaluation_report.md:935`, Appendix C.3).

**Verdict:** a real baseline (S&P 500 buy-and-hold, index-normalized) exists in code and is shown to the reader, but the paper is explicit that it is contextual only — the study's conclusions (concentration, re-entry, loss asymmetry) are computed entirely from the portfolio's own trade/position data, not from any comparison to the baseline. There is **no** paired significance test (no bootstrap, no permutation test, no t-stat on excess return vs. either index) anywhere in the code or the paper. That is the single biggest methodological hole for a study whose stated question is whether the LLM "actually generate[s] alpha" (`README.md`).

## A.3 How Performance Was Computed

**Statistics, all daily-frequency (`evaluation_report.md:78`):**
- Max drawdown / running max / largest run — `trading_script.py:1070-1074` (`daily_results`) and independently re-derived in `graphing/equity_vs_baseline.py:47-100` (`find_largest_gain`, `compute_drawdown`); formal definitions at `evaluation_report.md:430-482` (Appendix A.1).
- Sharpe (period + annualized), Sortino (period + annualized) — `trading_script.py:1102-1130`. Risk-free rate is hardcoded: `rf_annual = 0.045` (`:1102`); annualization is `* sqrt(252)` for the annualized forms and `* sqrt(n_days)` for the period forms; Sortino uses downside deviation clipped at the risk-free daily rate (`downside = (r - rf_daily).clip(upper=0)`, `:1105`).
- CAPM beta/alpha/R² vs `^GSPC` — `trading_script.py:1140-1163` (see A.2).
- FIFO lot-level trade stats (win rate, profit factor, expectancy, avg holding days) — reconstructed post-hoc from `Trade Log.csv`, not produced live; formulas at `evaluation_report.md:606-624` (Appendix A.8-A.9), values at `evaluation_report.md:1034-1043` (Appendix D.1): 46 lots, win rate 0.500, profit factor 0.822, expectancy **-$0.41/lot**.
- Peak Capture Ratio (exit PnL / peak unrealized PnL per episode) — `scripts/metrics/episode_pcr.py:44-52`; nulled when `peak_pnl <= 0` (`evaluation_report.md:694-698`).

**Window:** the full 6-month experiment for equity-curve stats; FIFO/episode stats are computed once, over the whole period, at evaluation time — not rolling, not walk-forward, not split train/test.

**Costs charged: none.** Direct quote: *"The trading simulation did not incorporate transaction costs such as commissions or bid–ask spread effects."* (`evaluation_report.md:252-254`, "Simulation Limitations"). Confirmed in code: every fill in `process_portfolio()` (MOO buy `:556`, stop-loss sell `:694-699`, HOLD mark `:710-717`) uses the raw quoted price with **no fee, spread, or slippage term anywhere in the module** (verified by exhaustive grep for `fee|commission|slippage|spread` across `scripts/processing/trading_script.py` — zero hits). All PnL, Sharpe, drawdown, and expectancy figures in the paper are **gross**, not net.

## A.4 Forward-Only Discipline

**What actually stopped the operator from re-running it until it looked good: nothing cryptographic. NOT FOUND** — no hash-commit, no pre-registration file, no timestamp-proof mechanism anywhere in the repo (exhaustive grep for `hash|pre-regist|preregist|commitment|sha256` across all `.py`/`.md` files returns nothing outside this teardown itself). The repo is a **shallow git clone with a single commit** (`589c92a`), so there is no commit-by-commit audit trail of decisions being logged before outcomes were known either.

**What does exist, and is real evidence, just not a formal commitment scheme:**
- Publicly shared, timestamped ChatGPT conversation links, one per multi-week block: `Experiments/chatgpt_micro-cap/collected_artifacts/chats.md:6-8` — three `chatgpt.com/share/...` URLs covering 8/1–8/29, 8/30–9/26, 9/27–now. These are hosted and timestamped by OpenAI, not by the experiment author, which is the closest thing to third-party attestation in the repo — but they only start on 8/1, five weeks into a six-month experiment (`chats.md:3`: *"I unfortunately only started documenting chats on (8-1)"*), so the first ~5 weeks of decisions have no independently-timestamped trail at all.
- Weekly Deep Research PDFs archived verbatim (`collected_artifacts/Weekly Deep Research (PDF)/Week 1.pdf` … `Week 26.pdf`) — dated artifacts, but self-hosted, so their timestamps are only as trustworthy as the repo's own commit/file metadata, not an external notary.
- The paper's forward-only claim rests on **procedure, not proof**: "all model decisions were generated using only information available prior to trade execution, and all trades were executed on a forward-only basis" (`evaluation_report.md:134`) — this is an assertion about how the human operator behaved, unverifiable from the repo alone.

**Verdict:** forward-only discipline here is honor-system plus partial, non-cryptographic timestamping (OpenAI's own share-link creation dates). This is meaningfully weaker than ARGUS's hash-chained decision ledger, which commits the decision content before the outcome exists — that is a genuine, provable advantage ARGUS already has over this repo's methodology, and it should be stated in those terms in ARGUS's writeup.

## A.5 What the Experiment Actually Concluded (quoted)

> *"Across the experimental period, portfolio equity outcomes were dominated by a small number of high-impact trades. High position concentration amplified exposure to individual ticker outcomes, with a single adverse position exerting a disproportionate influence on overall portfolio balance. Trading behavior exhibited persistence in position-level theses, as the model re-entered tickers despite prior exits, including cases with realized losses… Taken together, these results suggest that, when placed in a capital allocation role, an LLM exhibits decision-making patterns resembling high-conviction, thesis-driven discretionary trading. Portfolio outcomes were shaped less by incremental trade-level performance and more by concentration, persistence in position-level narratives, and asymmetric downside exposure."* (`evaluation_report.md:322-324`)

Concretely: max drawdown **-50.33%** on 2025-11-06 (`evaluation_report.md:156`); 10 of 22 tickers profitable; FIFO profit factor 0.82 (expectancy -$0.41/lot), improving to 1.52 (expectancy +$0.49/lot) with the single largest loser (ATYR) excluded (`evaluation_report.md:230`) — i.e. the entire realized edge sign flips on removing one position. The paper is explicit that this is a **behavioral characterization study, not a performance or skill claim**: *"Rather than optimizing performance or assessing predictive skill, this evaluation examines how an LLM allocates capital, manages risk, and exhibits trading behavior"* (`evaluation_report.md:23`).

## A.6 Methodological Holes

1. **No baseline used for the actual conclusions** (A.2) — the one baseline computed (S&P/Russell buy-and-hold) is explicitly disclaimed as non-evidentiary; there is no significance test anywhere.
2. **N=1 run, N=1 model, N=1 sector.** The paper self-flags this: *"The results analyzed are based on a single experimental run and do not capture variability across repeated runs or alternative random initial conditions"* (`evaluation_report.md:310`). One adverse ticker (ATYR) determines whether the strategy looks profitable or unprofitable — that is not a statistically resolvable question from this sample.
3. **Uncontrolled prompt/model drift.** Interactive ChatGPT UI, not the API: *"generation parameters such as temperature were not explicitly fixed"* and *"newer model versions were adopted as they became available"* (`evaluation_report.md:314`) — mid-experiment prompt rewrites are documented (compare `Appendix C.2.1` vs `C.2.2`, `evaluation_report.md:787` vs `:793-903`) with materially different structure (free-form vs. a rigid "Restated Rules → Exact Orders" template). Any behavioral shift after 8/30 is confounded with the prompt rewrite, not attributable to the model alone.
4. **No cost model** (A.3) — every performance number is gross. Given ARGUS's own measured ~12bps round-trip fee dominating a ~0.00% measured intraday edge, this repo's headline numbers would very plausibly flip sign under realistic costs, the same way its own profit factor flips on excluding one ticker.
5. **Forward-only claim is asserted, not provable from the artifact** (A.4) — no cryptographic commitment, first five weeks have no independent timestamp at all.
6. **FIFO/episode reconstructions are explicitly approximate.** The paper states its own source data is "blended" and "not generated as a single unified, transaction-perfect ledger… not intended to be treated as definitive accounting" (`evaluation_report.md:364`, `:406`) — every headline number sits on top of a reconstruction the authors themselves call non-authoritative.
7. **The "Deep Research" mode partially defeats point-in-time isolation by design.** The weekly research prompt explicitly permits general web browsing for context (`evaluation_report.md:124`: *"the research process permitted consultation of publicly available web sources"*) even though the model has no *decision-time* access to that browsing — this is a soft, human-trusted boundary, not an enforced one.

---

# PART B — ai-hedge-fund: Architecture

**Note on scope:** the cloned repo is the in-progress **v2 rewrite** ("a persistent, always-on AI hedge fund", `README.md:5-7`), not the older flat `src/agents/*.py` script most people cite. All citations below are against this v2 codebase under `hedge_fund/`.

## B.1 Agent Roster and the Flow from Data to Order

**Pipeline, one function, every mode (backtest/paper/live share it):**

```
point-in-time data -> analysts -> blend -> risk -> execution -> record
```
`hedge_fund/pipeline/run_cycle.py:1-8` (module docstring), implemented at `run_cycle.py:49-135`.

**Step-by-step, with file:line:**

1. **Mark prices, point-in-time.** `_mark_prices()` (`run_cycle.py:142-172`): last close on or before `as_of` within a 7-day lookback (`_MARK_LOOKBACK_DAYS = 7`, `:46`). A held ticker with no recent price **raises** (`:162-166`, "cannot value the book"); an unheld universe ticker with no price is silently skipped and recorded (`:167-171`) — the asymmetry is deliberate and documented (`:23-26`).
2. **Analysts (AlphaModels) run per strategy, per ticker.** `run_cycle.py:88-92`: for each `(strategy, staff)` pair, every staffed model's `.predict(ticker, as_of, data_client)` is called for every tradeable ticker. Two kinds of AlphaModel, one interface (`signals/base.py:29-51`, abstract `predict()` returning a `Signal` with conviction in `[-1, +1]`):
   - **QuantModel** — pure Python math, e.g. `PEADModel` (Post-Earnings-Announcement-Drift): fires ±1.0 conviction on a fresh 8-K/10-Q EPS BEAT/MISS filed within a signal window, point-in-time filtered on `filing_date <= as_of` (`signals/pead.py:50-84`, `:54-55`).
   - **LLMAgent** — persona reasons over a `FundamentalsSnapshot` (`signals/llm_agent.py:39-172`). Roster per `ROADMAP.md:56-63`: **shipped** — Buffett, Munger, Graham, Lynch, Druckenmiller; **planned** — Wood, Burry, Ackman, Damodaran, Fisher, Pabrai, Taleb, Jhunjhunwala. Persona = system prompt only; all machinery (caching, parsing, abstention) lives in the shared base class (`signals/buffett.py:1-7`).
3. **Blend into target weights.** `blend_signals()` (`portfolio/construction.py:29-88`) — the fund's portfolio manager. See B.2.
4. **Risk clamp.** `apply_limits()` (`risk/limits.py:49-82`) — hard, code-enforced caps. See B.3.
5. **Execution — diff to orders.** `build_orders()` (`pipeline/execution.py:16-53`): `target_shares = int(weight * equity / mark)` (`:40`, floor-toward-zero, never overshoot), diffed against current holdings, sells emitted before buys (`:28-30`, frees cash within the same cycle), sub-share deltas simply not emitted.
6. **Fill.** `broker.place_order(o)` for each order (`run_cycle.py:111`) against the `Broker` protocol (`brokers/protocol.py:15-34`) — `SimBroker` for backtests (`brokers/sim.py`), `PaperBroker`/live broker **not yet built** (`ROADMAP.md:83-84`, both ⬜/🚧).
7. **Record.** Every cycle returns a `CycleRecord` (`run_cycle.py:117-135`) carrying signals, convictions, weights, clamps, orders, fills, and NAV — a full per-decision receipt, though the *reading* half (seeding a new run's broker from the last receipt so NAV persists across runs) is explicitly not shipped yet (`ROADMAP.md:14-18`, `:33`).

**Does it place real orders?** No — same answer as TradingAgents. `README.md:8`: *"Note: the system does not actually make any trades."* `SimBroker` is in-memory only (`brokers/sim.py:19-53`); a live broker is an open roadmap item (`ROADMAP.md:84`).

## B.2 How the Portfolio Manager Aggregates Disagreeing Specialists

**Quote, the actual aggregation formula** (`portfolio/construction.py:37-39`):

```
conviction_t = sum(w_m * value_mt) / sum(w_m)     # weighted mean over voting models
```

Implemented at `construction.py:61-74`. This **is a weighted arithmetic mean of conviction — it averages disagreement into mush**, not a debate or a preserved distribution. A Buffett `+0.8` and a Munger `-0.8` on the same ticker at equal weight nets to `0.0` — indistinguishable from two models that both abstained-to-neutral or both said "no opinion." No variance, dispersion, or disagreement metric is computed or carried forward into risk sizing; the only place disagreement could influence anything is if a future `evaluate()` step scored per-model accuracy (flagged as a known gap in the same docstring, `:9-12`: *"the cross-sectional normalization ignores absolute conviction — a lone weak view would receive the full gross target… a min-conviction floor is the obvious knob once `evaluate()` can measure it"*). This is a self-documented wart, not something the authors are unaware of.

One real nuance preserved: **abstention is not neutral.** `metadata.abstained is True` signals are excluded from *both* numerator and denominator (`construction.py:64-65`) — "no opinion" genuinely drops out rather than dragging the average toward zero — but a non-abstained `0.0` (e.g., PEAD outside its signal window) is a real vote and does dilute (`:44`). So disagreement between *opinionated* models is still averaged away; only "no opinion" is handled specially.

Cross-sectional weights are then just the (optionally demeaned, for market-neutral sleeves) convictions normalized to `gross_target` (`:76-87`) — again pure linear scaling, no debate structure at all. **Compare directly to TradingAgents' Bull/Bear + Research-Manager-as-judge**, which preserves the adversarial framing all the way to a discrete Buy/Hold/Sell call and lets a single judge model override either side; ai-hedge-fund's blend is closer to a factor-model combination than to a debate.

## B.3 Risk Layer — Enforced in Code, Not Prompt Guidance

**This is the single most important finding in this repo relative to both TradingAgents and LLM-Trading-Lab.**

`risk/limits.py:1-11` (module docstring, verbatim): *"Risk limits — hard caps the analysts cannot override… 'Conviction requests, risk disposes': portfolio construction proposes target weights, and this stage clamps them against the fund's limits. **Everything here is deterministic arithmetic — the LLM's influence over the book ends at the Signal, and no clamp is ever negotiable.**"*

Mechanism, `apply_limits()` (`risk/limits.py:49-82`):
1. **Per-ticker cap** (`max_position_pct`): any `|weight|` above the cap is clamped to `±cap`, sign preserved (`:62-72`). One `ClampEvent` logged per clamp (`:33-39`) — fully auditable.
2. **Gross cap** (`max_gross_exposure`): if summed `|weights|` still exceed the cap after per-ticker clamping, **every** weight is scaled down proportionally (`:74-80`) — scaling only ever shrinks, so it provably cannot re-violate the per-ticker cap (order dependency is explicitly called out and correct, `:52-57`).
3. **Removed exposure goes to cash, never redistributed** (`:8-10`) — an explicit, deliberate design choice to prevent the risk stage from ever *increasing* any position.

The LLM has **no path** to talk its way past this: its entire influence on the book terminates at emitting a `Signal(value ∈ [-1, +1])` (`signals/base.py:44-51`); everything from `blend_signals` onward (`portfolio/construction.py`, `risk/limits.py`, `pipeline/execution.py`) is pure, deterministic Python with no LLM call in the loop (`run_cycle.py:93-110` — no `.llm` or `.predict` call after the analyst step). Pydantic's `ConfigDict(extra="forbid")` on `RiskLimits` (`limits.py:23`) additionally prevents a config file from smuggling in unknown risk parameters.

**This is exactly the property TradingAgents lacks** (see `tradingagents.md` §8: "Risk is debated, not enforced… No hard maximum position size, no hard maximum loss stop, no veto if the LLM overrides risk analysts"). ai-hedge-fund proves the pattern is cheap to build: two Pydantic models and ~35 lines of arithmetic (`limits.py:49-82`). **ARGUS should copy this file's shape near-verbatim** — it is the single cleanest "hard gate the LLM cannot override" reference implementation across either repo studied here.

Caveat found by reading further: **margin is not modeled** — `brokers/sim.py:8-11` admits cash may go negative and "nothing here pretends to enforce it," relying on the unlevered mandate (`gross_target <= 1`) plus sells-before-buys ordering to avoid it in practice rather than enforcing it. Not a risk-layer defect exactly (the risk layer's job — position/gross caps — is done correctly), but it means "risk is enforced in code" does not yet extend to margin/leverage.

## B.4 Does Any Number Reaching the Model Get Computed in Code?

**Yes, entirely — this is architecturally guaranteed, not just good practice.** `features/snapshot.py:1-13` (module docstring): *"a few derived aggregates computed here in Python **so the LLM reasons over facts instead of re-deriving arithmetic**."*

Concretely, `build_snapshot()` (`snapshot.py:117-158`) precomputes every derived figure in Python before the persona ever sees it:
- `roe_avg` — `_avg()`, plain mean (`:132`, `:178-180`)
- `net_margin_avg` — same (`:133`)
- `gross_margin_trend` — `_trend()`, latest-minus-oldest (`:134`, `:183-185`)
- `bvps_cagr` — `_cagr()`, `(x[0]/x[-1])**(1/years) - 1` with an explicit quarter-spacing assumption (`:135`, `:188-198`)
- `debt_to_equity_latest`, `market_cap_latest` — direct field reads (`:136-137`)

`render()` (`snapshot.py:78-114`) then emits a **text table of these pre-computed numbers plus the raw per-period fundamentals** — the LLM is shown numbers, not asked to compute them. The persona system prompts reinforce this as a hard rule at the prompt level too (defense in depth, not the only guard): Buffett's prompt explicitly states *"Reason ONLY from the data provided… Do not invent numbers"* (`signals/buffett.py`, system prompt body) — but note this second guard **is** prompt guidance and therefore not enforced; the real guarantee is architectural (B.4's real point): the model is structurally never asked to do arithmetic that would produce a load-bearing number, because `LLMAgent._parse()` only ever extracts `{signal: bullish/neutral/bearish, confidence: 0-100, reasoning: str}` (`signals/llm_agent.py:122-135`) — a classification and a scalar confidence, not a price, size, or ratio. Every subsequent number (`value = sign * confidence/100`, `signals/llm_agent.py:146`; every weight, clamp, and order-share count downstream) is pure code. **This is the strongest possible answer to Track 2's "is the LLM the decision-maker or the calculator" question** — here it is a classifier feeding a deterministic pipeline, full stop.

## B.5 Defects, Bluntly

1. **No cost model anywhere.** `SimBroker.place_order()` fills "exactly at the order's reference price" every time (`brokers/sim.py:2-4`); the docstring itself flags this as a gap: *"Slippage/costs are a declared future addition inside place_order"* (`:5`). Exhaustive grep for `fee|commission|slippage|spread` across `hedge_fund/*.py` (excluding tests) returns **zero** hits of an actual cost implementation — same defect as both TradingAgents and LLM-Trading-Lab. All three systems studied so far report gross, not net, performance.
2. **No sentiment/news signal exists at all in this roster.** Every shipped analyst (Buffett/Munger/Graham/Lynch/Druckenmiller, PEAD) reasons over fundamentals or earnings-surprise data (`signals/*.py`); there is no news, macro, or social-sentiment AlphaModel shipped or even planned in `ROADMAP.md:44-63`. This is a real capability gap versus TradingAgents' four-analyst layer (market/sentiment/news/fundamentals).
3. **Conviction-weighted blend is self-admittedly under-specified for low-conviction, single-model cases** (B.2) — a lone weak `+0.1` conviction on an otherwise-silent ticker still receives the *full* `gross_target` allocation for that name, because normalization is purely cross-sectional (`construction.py:9-12`, `:83-87`). No `min_conviction` floor exists yet.
4. **Point-in-time correctness is self-rated "in progress," not done.** `ROADMAP.md:32`: *"Point-in-time data correctness — as-of / filing-date queries, no lookahead | 🚧"*. The fundamentals path (`snapshot.py`) does filter on `filing_date`, and PEAD explicitly filters `filing_date <= as_of` with a 45-day retrospective-filing guard (`signals/pead.py:21`, `:116-118`) — but the project's own status table says the guarantee is not complete fleet-wide, which is more honest than TradingAgents' CHANGELOG (which claims the PIT work is "done" for v0.4.0) but also means it should not be cited as fully proved without re-checking each data path at the time ARGUS borrows from it.
5. **No statistical validation gate on the backtest engine.** `ROADMAP.md:33`: CPCV / probability-of-backtest-overfitting is listed as ⬜ Planned, not built. `backtesting/engine.py` computes Sharpe/drawdown/win-rate on a single equity path (`_compute_metrics`, `:226-291`) with no resampling, no train/test split, no significance test — anyone using this engine's numbers as-is inherits the same "N=1 path" problem LLM-Trading-Lab has (A.6.2).
6. **Margin/leverage is unenforced, only avoided by convention** (B.3 caveat) — `brokers/sim.py:8-11` lets cash go negative silently; nothing raises or clamps on it.
7. **No live or paper broker shipped** (`ROADMAP.md:83-84`) — this is explicitly a backtest-only proof of concept today, same practical limitation as TradingAgents (no order placement) and LLM-Trading-Lab (manual human execution).
8. **LLM parse/call failures abstain silently by design** (`llm_agent.py:69-95`) — reasonable (fail-loud on data errors, fail-quiet on LLM errors, per the documented "failure contract" at `:15-21`), but a strategy where *every* staffed model abstains produces a fully flat book with no explicit alert distinguishing "the market gave no signal" from "every LLM call errored" at the `run_cycle` level — the record shows `abstained` per-signal (`run_cycle.py:19-22`) but nothing aggregates that into an operational alert.

---

# PART C — For Us

## C.1 The Single Most Credible Baseline ARGUS Can Compute Today

**Given:** 12bps round-trip taker fee, ~0.00% measured intraday edge, a hash-chained decision ledger, zero settled trades so far, and a paper-trading ledger of tokenized US equities.

**Recommendation: position-level buy-and-hold of ARGUS's own traded universe, at ARGUS's own capital and window — not an index.**

Concretely: for every ticker ARGUS's ledger ever entered, take the first entry price and the price at the evaluation cutoff (or exit, whichever the metric needs), and compute what an equal-dollar (or ARGUS's-own-first-sizing) allocation across exactly that set of tickers would be worth if bought once and never touched again. Compare its Sharpe, max drawdown, and terminal return against ARGUS's actual (turnover-heavy) realized path.

**Why this is the correct baseline and not the S&P/Russell approach both studied repos used:**
- It isolates **selection skill from timing/turnover skill**. An index buy-and-hold (what both LLM-Trading-Lab, `evaluation_report.md:146-151`, and TradingAgents' memory-reflection alpha-vs-benchmark both compute) answers "did the agent beat the broad market" — a much easier bar that says nothing about whether ARGUS's specific entries/exits added value over just holding what it picked. Given ARGUS's own measured intraday edge is ~0.00% (per the brief), the market-beating question is almost certainly going to be a wash either way and won't be informative.
- It makes the fee argument mechanical and undeniable: buy-and-hold pays the round-trip fee **once** (or, if the metric is framed to exclude the initial buy that ARGUS also pays, effectively **zero incremental** fee drag versus ARGUS's repeated round trips). If ARGUS trades the same names multiple times, every extra round trip costs 12bps against an edge measured at ~0.00% — so the buy-and-hold comparison directly quantifies how many basis points of real, uncontested value ARGUS's turnover has to generate just to break even, let alone win. This is precisely the "backtest honesty" failure mode already burned into the global rules ("fees dominate on RWA perps… round-trip taker is 0.12%; measured intraday edge is ~0.00%") — computing this baseline turns that warning into a number specific to ARGUS's own ledger instead of a general worry.
- It is **computable today from data ARGUS already has**, with zero new infrastructure: the decision ledger already records ticker, entry price, and timestamp per decision (that's the hash-chained record); the only additional input needed is the closing/mark price of each ticker at the chosen evaluation cutoff — one price per traded name, pulled from the same market-data feed ARGUS already uses for marks. No random-shuffle engine, no Monte Carlo harness, no new backtest code required.
- It matches the exact metrics Track 2 is judged on: Sharpe, max drawdown, and win rate can all be computed on the buy-and-hold path with the identical formulas ARGUS already uses on its own path (reuse the Sharpe/drawdown code, don't write new metric code — only the price series changes).

**What it would take to compute, step by step:**
1. From the decision ledger, pull the distinct set of tickers ARGUS has ever taken a position in, with the timestamp and price of each ticker's *first* entry.
2. Pull one closing/mark price per ticker at the evaluation cutoff (same source as ARGUS's own NAV marks, to keep the comparison apples-to-apples).
3. Construct an equal-weight (or ARGUS's-actual-first-trade-weight, reported both ways) capital allocation across that fixed set at inception, no rebalancing, no further trades.
4. Run it through the *same* Sharpe/max-drawdown/win-rate code ARGUS already uses on its own realized path (do not write a second metrics implementation — a second implementation is a second place for a units/annualization bug to hide, per `hftbacktest`/`nautilus_trader`-style engine hygiene).
5. Report three numbers side by side, ledger-style, per the ASP outcome-quality standard already in force here: ARGUS realized (net of 12bps/round-trip), buy-and-hold realized (net of one entry fee only), and the delta — and say explicitly whether the delta survives the fee gap, not just whether the raw returns differ.

**Second-order, once trade count is large enough to matter (not blocking, but the natural follow-on):** a turnover-matched permutation/shuffle test — reassign ARGUS's own realized trade directions or entry timings at random within the same universe and date range, replayed through the identical 12bps-round-trip fee model, N times, to get a null distribution for Sharpe/return. Compare ARGUS's realized statistic against that null. This is the statistically rigorous complement to buy-and-hold (it answers "is the *direction-picking*, not just the *name-picking*, adding value net of the exact same cost structure"), but it needs enough realized trades to build a meaningful null distribution — with zero settled trades today, buy-and-hold is the baseline to stand up first; the shuffle test is the one to add once there is a trade history worth permuting.

---

## Appendix: File:Line Index

**LLM-Trading-Lab:**
- `Other/License.txt` — MIT, Nathan Smith
- `README.md` — repo purpose, feature list
- `Experiments/chatgpt_micro-cap/evaluation/evaluation_report.md` — the 40-page paper (abstract, methodology, results, appendices A-D, prompts C.1-C.4)
- `Experiments/chatgpt_micro-cap/scripts/processing/trading_script.py` — `set_asof` (44-56), `process_portfolio` (490-767, stop-loss enforcement 693-699), `daily_results` (1009-1279, Sharpe/Sortino 1102-1130, CAPM 1140-1163)
- `Experiments/chatgpt_micro-cap/graphing/equity_vs_baseline.py` — S&P/Russell buy-and-hold baseline (29-44), drawdown/largest-run (47-100)
- `Experiments/chatgpt_micro-cap/scripts/metrics/episode_pcr.py` — Peak Capture Ratio (1-52)
- `Experiments/chatgpt_micro-cap/collected_artifacts/chats.md` — timestamped ChatGPT share links (6-8)

**ai-hedge-fund:**
- `LICENSE` — MIT, Virat Singh
- `README.md` — v2 rewrite notice (5-7), "does not actually make any trades" (8)
- `ROADMAP.md` — self-rated capability status (14-91)
- `hedge_fund/pipeline/run_cycle.py` — the pipeline (1-172)
- `hedge_fund/signals/base.py` — `AlphaModel` interface (29-51), `QuantModel` helpers (54-105)
- `hedge_fund/signals/llm_agent.py` — `LLMAgent` base, cache/abstain contract (1-172)
- `hedge_fund/signals/pead.py` — quant alpha model, PIT filtering (1-138)
- `hedge_fund/signals/buffett.py` — persona-as-system-prompt pattern
- `hedge_fund/portfolio/construction.py` — `blend_signals`, the aggregation formula (29-89)
- `hedge_fund/risk/limits.py` — `apply_limits`, hard-coded risk gate (1-82)
- `hedge_fund/pipeline/execution.py` — `build_orders`, sizing arithmetic (1-53)
- `hedge_fund/features/snapshot.py` — point-in-time snapshot, code-computed aggregates (1-199)
- `hedge_fund/backtesting/engine.py` — backtest harness, metrics (1-299)
- `hedge_fund/brokers/sim.py` — `SimBroker`, no-cost-model admission (1-53)
- `hedge_fund/brokers/protocol.py`, `hedge_fund/brokers/models.py` — broker/order contracts
- `hedge_fund/fund/spec.py` — `FundSpec`/`StrategySpec` hierarchy (1-70+)
