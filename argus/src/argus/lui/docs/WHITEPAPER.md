# ARGUS: A Research Workbench That Computes Every Number It Shows

*Whitepaper. Bitget AI Base Camp, Season 2. Track 3, AI Trading Desk (Open Theme: a portfolio-aware research copilot). Figures read from the repository's artefacts and the live console on 6 October 2026.*

## 1. Abstract

ARGUS is a natural-language research workbench for Bitget's US-stock perpetuals, tokenized US stocks, crypto, indices and commodities. A trader asks a question in plain words, in any of eight languages, and the console answers with figures computed for that answer from live market data, company filings and public macro series. Every line of the answer is labelled with where it came from. The language model in the system reads the question and translates the answer; it never writes a figure the trader sees.

The design rests on one constraint: no source, no answer. Where a figure cannot be read or computed, the console says so on the line where the figure would have been, and where a question cannot honestly be answered as asked, it declines and gives the true reason. Before a trade, it shows what that trade does to the trader's own book: the share of risk the new position would carry, its beta in the session that actually prices the stock, sector and factor exposure, correlation, concentration, a scenario tree of stresses, a sized hedge, and an execution schedule priced on the live order book.

Each capability is graded in a public register against a named rival, run on the same input. Of 51 capabilities, 6 are OWNED (beaten under thirteen conditions), 37 are TIED and 8 are LOST (`argus/data/standing.json`). The losses are published with the same prominence as the wins. Real-user data is thin: the console records anonymous usage events and a "Did this answer your question?" control (`argus/src/argus/lui/usage.py`), but no session counts or task-completion figures from real traders are reported in this paper, because none was read from a published artefact.

## 2. The problem

### 2.1 Pre-trade research on a venue that never closes

Bitget lists perpetual contracts on US stocks and ETFs (NVDA, TSLA, AAPL, MSFT, META, QQQ and others) and tokenized spot versions of the same names. These instruments trade around the clock. The stocks they track do not. That gap produces three problems a conventional research tool does not handle.

**Beta depends on the session.** Measured on 30 days of hourly candles against QQQ, 82.5% of hourly bars fall while the US market is shut, and open-session beta exceeds shut-session beta on 10 of 11 stock perpetuals (`argus/data/session_beta.json`). AAPL reads 0.68 in the open session and 0.105 in the shut one; NVDA 1.74 against 1.28. A single beta fitted over all bars is dominated by the hours in which there is no price discovery, so it understates exposure in the one session that prices the stock. The two exceptions are TQQQ and SQQQ, whose ±3x leverage on the benchmark is fixed by construction and should not vary by session; that is consistent with a real effect rather than a measurement error.

**Prices gap at the reopen.** While the stock is shut, the perpetual keeps trading on a thinner book. Its price at 09:00 New York carries information about where the stock will open, and a holder needs to know how much to trust it.

**Costs differ by session.** Quoted spreads understate the cost of real size, resting orders on a thin weekend book tend to fill only when the price moves against them, and funding is charged every eight hours on a perpetual and not on the stock.

### 2.2 An LLM writing numbers from nowhere

A general-purpose language model asked to "research" a trade writes plausible figures: a revenue number, a beta, a price target. Nothing ties those figures to a source the reader can check. The failure is not that the figures are always wrong; it is that a right figure and a wrong one look identical. ARGUS treats this as the central design problem. A research tool whose output cannot be audited line by line is not a research tool for money.

### 2.3 Who the user is

The target user is a discretionary trader on Bitget, from retail to VIP, holding a concentrated mix of US-stock perpetuals and crypto majors, at the moment before a trade. That trader usually has a view on the name and no measurement of what it does to the book they already hold. A second user is the newcomer who wants to see what a sum of money would have gone through before risking it. ARGUS is not for anyone who wants signals or orders placed on their behalf: it places no orders, holds no money and says so.

## 3. Design principles

### 3.1 The model reads; the engines compute

The language model (Qwen `qwen3.8-max` on the hackathon endpoint) has three jobs on the console: it reads a question into a structured research request, it answers questions about a company's own filings from retrieved passages under enforced citation, and it translates finished answers. Every figure is produced by deterministic Python engines reading named sources. When the model is unavailable or its budget is spent, a local classifier reads the question instead, and the answer is unchanged in kind (`argus/src/argus/lui/kindmodel.py`, `argus/src/argus/lui/router.py`).

The record path is deliberately model-free. A question about the desk's own history ("why did you not trade NVDA?") is classified by patterns and answered from the hash-chained ledger. A model paraphrasing the record would add a second thing that can be wrong; the record is the truth and nothing paraphrases it.

### 3.2 No source, no answer

An answer that asserts something must name its source: a ledger row, a computation, a named feed or filing. An answer with no source is permitted only when it is a refusal. Every answer ends by naming the sources it reached and any that did not answer. When a source fails, the answer says it failed rather than substituting a guess or a stale value; when a Bitget Skill is silent and the console reads the public source that Skill names instead, the line credits that source, never the Skill (`argus/src/argus/market/skill_mirror.py`).

### 3.3 Per-line provenance

Each content line of an answer carries one of nine labels (`argus/src/argus/lui/provenance.py`):

| Label | Meaning |
|---|---|
| `live` | Read from a source just now and stated as read: a quote, a funding rate, open interest, a headline, a macro value, a prediction-market price |
| `filed` | Quoted from a dated filing, fetched now; the passage is as of its filing date, not today |
| `computed` | Arithmetic on live data done for this answer: cost, beta, stress, VaR, the implied open, a hedge ratio, a premise verdict |
| `record` | A measured past result with its sample named: a backtest, a held-out test, a head-to-head against a rival |
| `desk` | The paper desk's own logged decision, quoted from the hash-chained ledger and never edited |
| `assumed` | A default applied because the question left it open; every such line begins "Assumed:" |
| `missing` | Something that could not be read or checked, said as such |
| `memory` | Something the trader told the console earlier, shown where it shaped the answer |
| `explained` | Fixed explanatory text written into the console; any number in it is read from the code that enforces it |

Labels come first from a step trace recorded while the answer is built (`argus/src/argus/lui/trace.py`, after the ReAct Thought/Action/Observation pattern): the step that produced a line declares what it is. Where no step speaks for a line, a rule over the line's wording supplies the label and marks it `origin: "regex"`. A line that matches no rule carries no label rather than a guess. On a held-out sample of 156 content lines, 94.2% carried a label on first read (`argus/data/provenance_audit.json`, sample `provenance_unseen.json`).

### 3.4 Premises are checked first

A question often carries a claim the asker takes as given. ARGUS checks such premises against the filing or price series that settles them before answering (`argus/src/argus/lui/premise_facts.py`). The checks were built from a hostile review that found four false premises answered around rather than corrected: a "$5.00 quarterly dividend" for Apple, whose 10-Q declared $0.27; a BTC price "last month" outside that month's actual range; META's ticker given as FB; and Coinbase placed on the NYSE when EDGAR lists it on Nasdaq. A check that cannot read its source says nothing, so a premise is never confirmed or denied on a guess.

### 3.5 Refusal with the true reason

Some questions cannot be answered by any console: the trader's private account, other traders' live positions, an exact future price, a statistic longer than the instrument's history, a company Bitget does not list. The original console declined some of these with the wrong reason and answered others with figures from an unrelated engine. A dedicated honesty layer now detects each cause offline and declines with the true reason, offering the nearest real figure where one exists (a past close, the longest span there is) (`argus/src/argus/lui/honesty.py`). Measured results are in section 6.6.

### 3.6 Orders are refused before anything else

An instruction to trade is one of the question kinds and is checked before any other classification, so it cannot be reached around. The order check was moved first after a test found that "sell half of that" matched the word "sell" in a decision-explanation pattern and the console began explaining a past decision instead of refusing an instruction.

## 4. Architecture

### 4.1 One product, three parts

ARGUS is a single Python package with seven layers and two runtime dependencies (`pydantic`, `python-dateutil`) (`ARGUS-ARCHITECTURE.md`). The console is what a trader uses. Two further parts supply evidence the console cites: a paper desk whose model decisions are hash-chained before their outcomes exist and are never sent to an exchange, and a register of falsifiable market claims committed and anchored before they resolve. Neither is a separate entry. This team's Track 2 entry is an independent project with its own code, ledger and demo; nothing in this paper draws on its trading record.

```text
  7  demo, lui, eval, status       human surfaces, and the judges of everything below
  6  agents, paper, register       the agents, the record, the public commitments
  5  desk, research, strategies    tools and studies
  4  execution, sim, backtest      orders, simulation, engine
  3  market, llm                   everything that fetches, the model client included
  2  cost, decision, risk, proof   vocabulary, limits, and the risk constitution
  1  truth                         time, what was knowable, and where a fact came from
  0  vendor                        third-party copies loaded by path
```

A module imports only from its own layer or below. `argus/src/argus/eval/architecture.py` builds the import graph with `ast` on every test run and fails the suite on a violation; the deterministic core (`truth`, `cost`, `risk`, `decision`, `backtest`) may never load the model client or the agents at any depth. An audit on 27 September 2026 found 41 imports pointing up the order, 33 of them inside functions where the earlier checker could not see them; each was fixed by moving the shared piece down, not by exemption.

### 4.2 The request path

```text
  question (any of 8 languages; book, memory and prior turns sent by the browser)
      |
      v
  [1] order check ------------------------------> refusal: "places no orders"
      |
  [2] honesty layer (offline cause detectors) --> decline with the true reason
      |                                           (+ nearest real figure)
  [3] multi-part split (one engine per part)
      |
  [4] readers, arbitrated (lui/arbiter.py)
        a. planner: Qwen qwen3.8-max, thinking off -> JSON plan
           (kind, names, holdings, size, shock); every name checked
           against Bitget's contract list
        b. fallback planner: local character n-gram kind model
        c. deterministic patterns (lui/research/parse.py)
      |       readers fill a request; none writes a figure
      v
  [5] premise checks (filings, price series)
      |
  [6] engine(s): quote, technicals, news/filings, earnings, base rates,
      book impact, exposures, execution, stress tree, macro, options,
      filing Q&A, newcomer, memory, ...  (lui/research/*, desk/*, market/*)
      |       each step recorded: Thought / Action / Observation
      v
  [7] provenance: per-line label from the trace, rule fallback
      |
  [8] answer: lead line, labelled lines, "Sources reached", receipt
      |
  [9] optional translation (Qwen), numbers locked to the English
```

The arbitration rules between the model's reading and the patterns' reading were once written inline in the HTTP handler, where no test could reach one without driving a request through the server. They now live in one module, each a named function with the incident that motivated it, and the payload records which reader won and why (`argus/src/argus/lui/arbiter.py`).

The hosted console caps Qwen at 20,000 tokens per server process; beyond that the kind model reads (`argus/data/lui_final_heldout_report.json`). This is the state a visitor is most likely to meet once a hackathon balance is spent, and the console is built to be fully usable in it.

### 4.3 Data sources

| Source | Used for |
|---|---|
| Bitget public market API (v2 market data; v3 instruments, order book up to 200 levels, fills) | Tickers, candles, 50-level books, funding, open interest, long/short ratio, instrument rules; every price and cost |
| Bitget order-book websocket (`books5`, trades) | Recorded tape for execution replay |
| `bitget-signal` (5 Skills, 19 tools) | Technicals (checked against a recomputation), crypto derivatives, sentiment and rates when answering |
| `bitget-mcp-server` (67 catalog entries) | US fundamentals, 13F holders, analyst estimates, earnings calendar, the stock behind each rToken |
| Agent Hub (`bgc` 3.0.0) | The first child of an execution plan, written as a `--paper-trading --dry-run` command |
| SEC EDGAR, XBRL company facts, Form 4, 8-K, 10-Q/10-K | Point-in-time financials, filing Q&A, insider flow, earnings dates |
| FRED, US Treasury par yield curve | Macro series and the rates curve, dated at the issuer |
| Yahoo Finance | Consensus estimates and revisions, stock closes and pre-market, sector classification, factor ETFs |
| Cboe delayed chains, FINRA | Options, short and dark-pool volume |
| DeFiLlama, CoinGecko, alternative.me | DeFi TVL, coin data, crypto fear and greed |
| ECB reference rate, else ExchangeRate-API | Amounts in the trader's own currency, the source and date stated |
| Polymarket (gamma API) | Prediction-market prices beside base rates |
| RSS feeds, X and Reddit | Headlines and social text, clustered so syndicated copies count once |

### 4.4 Surfaces

| Surface | What it is |
|---|---|
| Web console (`/`) | The question box, a "My book" field, per-line labels and a receipt naming every source |
| `/research` | One complete research task: eight engines in parallel, ending in a verdict assembled by rule from their figures |
| `/proof`, `/wrong`, `/status`, `/materials` | The register; every loss and correction; live health of each data surface; every link |
| MCP server (`/mcp`, Streamable HTTP) | Ten tools: `argus_ask`, `argus_quote`, `argus_portfolio_impact`, `argus_stress`, `argus_execution_plan`, `argus_scoreboard`, `argus_research_task`, `argus_week_ahead`, `argus_exposures`, `argus_review_trades`; none writes or trades |
| Telegram | A hosted bot, and a self-hosted mode that runs on the trader's own machine with their own bot token, with pairing approval and price or funding alerts |
| Command line | `python -m argus.lui.cli "<question>"` and the server on a local port, with no key and no network required for the record path |

## 5. Capabilities, problem by problem

Each subsection states the problem, how ARGUS solves it, and the evidence. Register states refer to rows of `argus/data/standing.json`.

### 5.1 Book-level risk

**Problem.** A trader asks whether to add a name. The useful answer is not about the name; it is about the book. A position can be a fifth of the money and half of the risk.

**Solution.** The impact engine computes, on 30 days of hourly returns: each position's share of total risk against its share of the money, with the decomposition checked to sum to portfolio volatility on every call; open-session beta before and after; the effective number of positions; the most correlated holding; a QQQ hedge sized from the book's beta; the book's move under a benchmark shock carried through each position's own beta; the worst realised 24-hour window for these exact weights; and the largest size that keeps the new name inside a risk budget. A separate engine reports sector weights and eight factor loadings with t-statistics (market, size, momentum, value, quality, low volatility, rates, crypto). A stress tree branches a stated shock into hedged, worst-window, single-name, trimmed and session variants.

**Evidence.** The kept research task (`argus/data/research_task_example.json`, run 2026-09-28 02:29 UTC, 9.28 seconds) asked: "I hold 40% NVDA, 30% MSFT, 30% AAPL. Should I add 15% TSLA?" The engine answered that at 15% TSLA would carry 19% of the book's risk, inside a 25% budget, with 18% the most that stays inside; open-session beta 0.86 to 0.94; effective positions 2.6 to 3.4; TSLA most correlated with NVDA at +0.28; NVDA carrying 43% of the risk on 34% of the money after the trade; a QQQ short of about 94% of book value to neutralise market exposure. The verdict, composed by rule from those figures and not by a model: "No measured edge: enter only on your own view", and for the size, "Add, at 15%".

Against rivals: post-trade beta missed the next four weeks' realised beta by 0.241 on average over 1,800 test books, against 0.554 for weekend-copilot (an S2 entry) as it ships (p = 0.012) and 0.360 on the same benchmark (p = 0.098, not significant); the row is TIED. The QQQ-fall stress missed the realised move by 0.92 points on 39 days against 1.00 for skfolio's vine copula and 0.98 for Entropy Pooling (p = 0.095), TIED; skfolio's vine wins the 10% tail quantile, because ARGUS states a point. Factor loadings equal statsmodels OLS exactly on the same returns (`argus/data/exposures_comparison.json`). A constrained book under caps, groups and name counts matches skfolio MeanRisk's annual volatility to within 1e-6 on five of six scenarios, TIED.

### 5.2 Execution on the live book

**Problem.** After deciding, a trader needs to know what a size costs to execute now. The quoted spread is the price of an infinitesimal order; resting orders on a thin weekend book are adversely selected; and a schedule smoothed across a market-open boundary solves a problem that does not describe the market.

**Solution.** The execution engine walks the live 50-level book for the notional being traded, splits it into near-touch limit and market slices, schedules it with the Almgren-Chriss optimum, and recommends one-minute children, the cadence of Bitget's own TWAP. Without a book, passive slices are priced at the taker rate rather than credited a maker fee they have not been shown to earn. The first child is written as the Agent Hub command that previews it and sends nothing. Venue rules (step size, minimum and maximum quantity, minimum notional, price band) are read from Bitget's instrument endpoint and sizes round down.

**Evidence.** In the kept task, a $15,000 TSLA order costs about 10.0bps in one market order (6.0bps fee plus 4.0bps book impact) and about 8.8bps split 60% near-touch limit then 40% market. The `bgc` dry-run mapping matched `bgc` 3.0.0's own output (`argus/data/agenthub_preview.json`, `all_match: true`). Against Bitget's TWAP on a full-depth replay, hourly children cost 12.2bps on a $100,000 parent against 6.9bps for the TWAP; cut into one-minute children the same trajectory costs 6.9bps. The row was LOST first and is now TIED. The session-aware schedule, however, is LOST (section 6.4). Exit cost from the displayed book is TIED with EGRESS (an S2 entry): the same method, median slippage difference 0.002bps at $10,000 over 1,104 recorded books.

### 5.3 The overnight open

**Problem.** While the US market is shut, where will the stock open, and how far should the perpetual's price be trusted?

**Solution.** The console reads the price the stock's own perpetual implies for the next open and reports that reading's measured miss beside rivals.

**Evidence.** OWNED. Over 113 nights and 8 stocks (15 April to 24 September 2026), the perpetual-implied open missed the actual open by 30.36bps mean absolute error, against 81.28bps for gloaming (an S2 entry estimating fair value from index futures, crypto and the dollar), 90.32bps for assuming no gap, and 36.26bps for the stock's own pre-market price at 09:00 (`argus/data/overnight_comparison.json`). ARGUS was better on every one of the 8 stocks against gloaming. As a trade entered on the perpetual at 09:00 and charged a 12bps round trip, it earned +1.64bps a night against -12.84bps for gloaming's best variant, a difference of +14.48bps [10.57, 18.62]; on QQQ the trade-arm lead is not separable (+0.86bps [-0.75, 2.44]), because gloaming's Nasdaq futures are nearly the same instrument. A first version booked the whole close-to-open gap and read +74.5bps a night; the review that caught it rebuilt the entry at the perpetual's price.

### 5.4 Filings, XBRL and enforced citations

**Problem.** A model asked about a company's financials invents plausible numbers, and filings themselves are traps: one 10-Q reports a 181-day cumulative figure and a 90-day quarterly figure with the same end date (for NVDA, 177.8B and 96.2B), and restatements change history.

**Solution.** Financial figures are read from SEC XBRL company facts, separated by duration, and queried as of a date: the `Fact` type carries five timestamps and a revision, and a query without an as-of date is a type error (`argus/src/argus/truth/`). Fiscal-year figures are computed from filed lines, with the formula and every filed line shown. For questions answered from filing text, the model writes sentences from retrieved passages, and code drops any sentence whose citation does not resolve to a passage.

**Evidence.** Point-in-time correctness is OWNED: on 619 questions over ten tickers' filings since 2017, one hour before and after each acceptance, ARGUS answered 619 with 0 leaks of later data; edgartools' `FactQuery.as_of` answered 369 (paired McNemar p = 1.1e-75), Qlib's PIT store 51, and OpenBB's platform 7 of 493 (`argus/data/pit_rivals.json`). On the nine tickers held out while building the gate, 553 of 553. The scope is one concept, latest-quarter revenue, and the truth labels and the gate both read EDGAR's acceptance time, which the row states.

Earnings-surprise ranking is OWNED: dating each year-over-year pair rather than counting positions returns a finite standardised surprise for 3,919 of 3,997 filers on the design frames against 3,695 for pandas' period index (p = 2e-63), and on a holdout never used to fit the rule 3,082 of 3,261 against 2,893 (p = 6e-26); against the edgartools package on 309 filers, 299 against 279 (p = 1e-5) (`argus/data/general_sue_comparison.json`, `argus/data/edgartools_real_check.json`). Before this was found, both ARGUS and the QuantConnect reference paired quarters by position, and because SEC XBRL has no standalone fiscal Q4, none of the 72 anchor pairs was truly year-over-year; that correction is published.

Filing Q&A (`argus/data/document_qa_eval.json`): 25 of 27 answerable questions correct, 3 of 3 unanswerable refused, 0 fabricated citation ids across 71 kept sentences. A deterministic sentence-level audit found 66 of those 71 supported by their passage, 1 unsupported and 4 undecidable; a model audit called all 71 supported, which is why the deterministic figure is the one reported. FinanceBench results are in section 6.5.

### 5.5 Macro

**Problem.** A research desk must place a name in rates, inflation and policy, but one of Bitget's macro tools returned every tenor as an error envelope and then a spread of 0.0: a yield curve of zeros built from legs that never arrived.

**Solution.** The curve is read from its publisher, the US Treasury's daily par yield series, dated at the issuer rather than at fetch; a curve older than seven days produces no evidence; a missing leg is `None`, never zero. FRED series supply macro indicators; the console also answers rate-decision, Fed-futures, inflation-hedge and cross-asset questions (`argus/src/argus/market/macro.py`, `argus/src/argus/lui/research/macro*.py`). Event studies measure how a name reacted to past releases, with a rank test whose ties share their average rank; before that fix it rejected 63.5% of placebo draws at 5% on flat off-hours bars. The event-significance row is TIED with a fixed-null rival.

**Evidence.** Wiring the Treasury curve took the paper desk's live panel from 2 analysts and 5 distinct sources to 4 analysts and 9 (`argus/README.md`). No rival comparison on macro answers specifically is in the register; this capability is not graded on its own.

### 5.6 Options

**Problem.** A trader expressing a view with defined risk needs a real chain, not a schematic.

**Solution.** Options answers read Cboe's delayed equity chains and crypto options venues. A thesis can be turned into a numbered plan: the thesis tested against its own data, then a debit spread on the name priced at mid from the delayed chain, sized so the debit is the most that can be lost, then exit rules set in advance (`argus/src/argus/lui/research/plan.py`, `equity_options.py`, `option_strategies.py`).

**Evidence.** The options category counts toward the OWNED data-breadth result (section 5.11). No same-input rival comparison of options answers is in the register; this capability is NOT VERIFIED against a rival.

### 5.7 The newcomer layer

**Problem.** A first-time-user audit on 30 September 2026 typed "is this financial advice", "how do i buy crypto", "can i lose more money than i put in", "should i buy the dip" and "whats a good stock for beginners"; every one was declined as unrecognised.

**Solution.** A layer answers these plainly without recommending anything: what the console is, how buying on Bitget works, what can be lost, and, where a number helps, the console's own engine run on a real example and said as an example (`argus/src/argus/lui/newcomer.py`, `beginner.py`). "I have $1,000 and I'm new" gets no pick: it gets what that sum went through in a year in three broad markets. Follow-ups a newcomer reaches for ("is that good?", "explain simpler", "so what should I do?") resolve against the previous answer. Amounts can be stated in the trader's own currency, converted at the ECB's or a named daily rate with the date stated.

**Evidence.** Fixed text in this layer carries the `explained` label, and any figure in it is computed by an engine rather than written by hand. No measured comprehension study with real newcomers exists; this is NOT VERIFIED with users.

### 5.8 Multilingual answers

**Problem.** A question in Chinese was routed correctly and answered in English. Putting a model between a computed figure and the reader is the one thing the console refuses to do.

**Solution.** The engines write English, which shows at once. A second call translates the lines with Qwen, and each translated line is checked against its English source: the multiset of numbers (every digit run) must be identical and every URL must survive character for character. A line that fails keeps its English. The translation endpoint accepts only answers this console signed (HMAC over language and lines), so it cannot be used as a free translator (`argus/src/argus/lui/translate.py`). Answers are supported in Chinese, Japanese, Korean, Spanish, Portuguese, French and German as well as English.

**Evidence.** Understanding across languages is measured in section 6.6 (240 questions in 12 languages). A measured run translated an eight-line answer in 9.3 seconds with every figure intact (`translate.py` docstring); a systematic translation-fidelity benchmark is not in the artefacts and is NOT VERIFIED beyond that run.

### 5.9 Memory

**Problem.** Track 3 judges a personalised thesis. A console that forgets "I can't lose more than 10%" after one answer gives the same answer to everyone.

**Solution.** Statements about the trader (loss limit, holding period, risk budget, capital, style, a thesis on a name) are extracted on every turn, deduplicated, replaced when restated, dated, and kept in the trader's own browser, sent with each question. Nothing is stored server-side, so one visitor can never read another's facts. Every use is shown on a `memory` line. A remembered thesis is shown beside later answers about that name with the move since it was stated (`argus/src/argus/lui/memory.py`, after mem0's extraction design; mem0's LLM extraction call and server-side vector store were deliberately not taken).

**Evidence.** On 30 statements, 30 recalled with 0 false memories from 20 statements not about the trader; on 20 two-session tasks, 20 answers changed by memory, none with memory off, none reaching another trader (`argus/data/memory_eval.json`). Against mem0 run unmodified on a blind held-out round, ARGUS recalled 15 of 20 facts to mem0's 18 and stored 0 of 12 non-facts to mem0's 7; the row is TIED, a loss on recall and a win on false memories.

### 5.10 Thesis testing, mandate and review

**Problem.** "Personalised" must change a verdict, not the wording; a stated thesis must be tested, not echoed; and a trader's own record should teach them something.

**Solution.** A stated thesis is broken into reasons, each tested on its own measurement (for "AI capex keeps accelerating", hyperscaler capital spending and NVDA revenue from SEC filings). A mandate (horizon, position size, exclusions, hedge requirement, conviction floor, concurrent-position cap) refuses or shrinks a proposal; a horizon or loss breach is a refusal, never a smaller trade. Pasted trades are reviewed for recurring habits, and a checklist is kept and re-run when the trader next asks about a name.

**Evidence.** The per-profile mandate is OWNED against hkuds/vibe-trading, the only one of 16 systems surveyed with a code-enforced mandate that changes a verdict on identical state, on 11 designed and 19,440 swept scenarios through both systems' real checks. The row's own blocker states the weaker figure plainly: 84.6% of proposals diverge across profiles, but only 1.3% for a reason that turns on a dimension the record supplies. Thesis testing is TIED with optic-bitget (S2): optic abstained on two of three theses for want of earnings and positioning data ARGUS answered, and still gives one synthesised probability ARGUS does not. Self-evolving review rules are TIED with a day-block interval above zero carried by one defect kind.

### 5.11 Data breadth

**Problem.** Track 3 judges data-source depth. Breadth is easy to claim and hard to measure.

**Solution.** The workbench's six per-stock research answers were scored by data categories that returned data, beside every keyless OpenBB provider called through OpenBB's own interface, for the same underlyings on the same day (`argus/src/argus/eval/perception_breadth.py`).

**Evidence.** OWNED: 16.0 categories per stock against OpenBB's 12.4 on the design day (+3.6 [2.7, 4.3]), 15.3 against 11.7 on a held-out day with nothing refitted, ahead on all twelve underlyings; scored only on categories OpenBB's keyless catalogue can reach, ahead by 1.0 and 1.2 [0.7, 1.5]. The ablation is stated: removing Bitget's data service costs 4.7 categories and turns the lead to -1.1. OpenBB with paid keys is wider, and its twelve keyed providers are named, not counted. The paper desk's own evidence panel is narrower, 6.2 against 12.4, and that row is LOST (section 6.4).

### 5.12 The honesty layer and premise checks

Covered in sections 3.4 and 3.5; measured in section 6.6.

### 5.13 The Bitget toolkit sweep

**Problem.** The judging focus names "Skill integration count and effectiveness". Wiring every tool and letting the silent ones return nothing would inflate the count and leave behaviour unchanged.

**Solution.** Every tool of `bitget-signal` and every entry of `bitget-mcp-server` is called on a schedule, keyless, and classified into states that cannot be confused: answered, empty, tool error, timed out, unreachable. An answer that reads a Skill credits it only if it answered. Skill output is checked before it is believed: RSI is recomputed from Bitget candles, and MACD is recomputed because the technical-analysis Skill returns the signal line and histogram in each other's fields on 10 of 10 symbols checked (live `/status`, 6 October 2026). Two client bugs found this way were fixed: every call carried one request id, so a reply could be matched to the wrong tool; and an `isError` flag was ignored, so an error message was one step from becoming a market fact.

**Evidence.** `bitget-signal`: 2 of 19 tools answered all three attempts on 2026-09-26 (crypto_derivatives, technical_analysis), from 1 of 5 Skills plus one tool no SKILL.md names (`argus/data/skill_reliability.json`). `bitget-mcp-server`: 37 of 67 catalog entries answered for NVDA on 2026-09-28, 21 of 22 equity entries (`argus/data/data_coverage.json`); the service returned 503 on every call from 25 to 28 September. On 6 October 2026 the live `/status` page showed its equity quote not answering, answering in 0% of 4 probes over 24 hours and 57.1% of 28 over seven days. These are low counts, and they are published as measured.

### 5.14 Other capabilities

| Capability | What it does | Register state |
|---|---|---|
| Weekend leverage | A leveraged weekend hold priced on every NVDA weekend since 1999 and Bitget's live maintenance tier; a volatility-scaled band for Monday's open covered 79% of 17,044 weekends out of sample against an 80% target | TIED with baserate (S2) |
| Stop placement | A stop placed outside ordinary noise, against Rook's (S2) invalidation price, on 958 held-out windows | TIED |
| Claims about the tape | A trader's stated causes checked against the tape; first run 18 of 38, fixed, re-run level at 30 of 30 gradable claims | TIED with MirrorLine (S2) |
| Overnight hedge | An rToken holder's night hedged with the company's perpetual; both systems remove 99.7% of overnight variance on held-out nights | TIED with Ballast (S2) |
| Base rates | How often a name finished higher over every past window, with overlapping windows counted honestly, and what Polymarket prices | Part of the research task |
| Analogue stress bands | Past states like today's and what followed, against AnalogDesk (S2) on its own pre-registered grid of 2,698 queries | TIED |
| Crowd and positioning | Open interest, long/short split, funding against its own last 100 settlements, trending coins checked against Bitget's board | Market sentiment TIED |
| Injection screening | Untrusted evidence quarantined before the model reads it, keeping each item's id and source and removing only its claim | One row TIED (HeyArka), one LOST (section 6.4) |

## 6. Evaluation

### 6.1 The capability register

Every capability is a row in `argus/data/standing.json`, with a named baseline, the conditions met, the proofs that point to artefacts and tests, and a transition log. The register is re-derived by `python -m argus.eval.standing`, which opens each cited artefact rather than trusting its filename, and raises at import if a row claims OWNED with any condition missing. States are ordered LOST, TIED, IMPLEMENTED, OWNED. OWNED requires all thirteen of:

| # | Condition |
|---|---|
| 1 | Best implementation studied |
| 2 | Best method studied |
| 3 | Baseline reproduced |
| 4 | Implementation complete |
| 5 | Same-input comparison |
| 6 | Statistically valid evaluation |
| 7 | Costs included |
| 8 | Out-of-sample test |
| 9 | Ablation |
| 10 | Adversarial test |
| 11 | Failure cases documented |
| 12 | Reproducibility proven |
| 13 | No specialist capability still superior without a stated reason |

The register's self-check is not clean, and says so: 52 conditions are claimed on rows without evidence in the artefact, every one on a TIED or LOST row and none on an OWNED row (`standing.json`, `findings`; `clean: false`). The register was made stricter on 25, 26 and 27 September 2026, when a per-group check by symbol and by half of the sample was added; the OWNED count fell from 20 to 8, then to 6, then to 2, before four rows returned to OWNED on 29 September on new evidence (`ARGUS-EXPLAINED.md`, "How the register got stricter"). Three rows were removed on 29 September because they served no flow or duplicated another, each with its reason on `/wrong`.

### 6.2 Methodology

A comparison counts only if the rival was run, not described. Rivals were cloned and executed from their own source, called through their own interface, or, where no licence permitted vendoring, reimplemented clean-room and verified against reference outputs produced by running the original once. Each comparison uses the same input on both sides; S2 entries were run on their own pre-registered tests where they had one. Where the best tool for a job is general-purpose rather than trading-specific (fd-shifts, Google CEL, pandera, pydantic, HiGHS, edgartools, statsmodels), it is the baseline. Statistical claims use paired tests and bootstrap intervals over the unit of independence (nights, filers, symbol-days), and population figures must survive a per-group check. A comparison that flattered ARGUS and was later found unsound is withdrawn on `/wrong`, which on 6 October 2026 listed 37 entries, 21 of them comparisons a named rival won.

### 6.3 Results: all 51 rows

Sub-theme codes are the register's own (t1 Alpha Factory, t2 Agentic Trading, t3 AI Trading Desk). Rows tagged t1 or t2 are capabilities of the shared engine graded against the specialists of those categories; ARGUS is entered only in Track 3.

| # | Capability | Sub-theme | State | Baseline run on the same input |
|---|---|---|---|---|
| 1 | Deliberation priced as a trading cost | t2-agentic | LOST | LatencySensitiveBench (NeurIPS 2025); a trailing realised-move estimator |
| 2 | Abstention scored as a decision | t2-riskcontrol | TIED | fd-shifts (general selective-prediction evaluator) |
| 3 | Overfitting gates that raise instead of returning NaN | t1-validation | TIED | vectorbt deflated Sharpe; pydantic contract |
| 4 | Typed factor grammar with no execution surface | t1-alphafactory | TIED | Google CEL, Polars, Qlib expression engine |
| 5 | Factor-discovery safety vs. RD-Agent | t2-factordiscovery | TIED | microsoft/RD-Agent; vectorbt deflated Sharpe |
| 6 | Cross-sectional factor evaluation | t1-alphafactory | TIED | microsoft/qlib |
| 7 | Cross-market cointegration with corrected multiple testing | t1-crossmarket | LOST | statsmodels, QuantConnect Lean, 128 general pipelines |
| 8 | Net executable arbitrage vs. a fee-blind detector | t1-arbitrage | TIED | bitcoin-arbitrage core; HiGHS |
| 9 | Data-honest breadth rotation | t1-rotation | TIED | pytaa; pandera with pydantic |
| 10 | Clustering-corrected event significance | t2-event | TIED | whale-signals fixed-null test |
| 11 | Funding-aware cross-asset hedge routing | t2-crossexecution | TIED | crypto_sor composite router |
| 12 | Refusal-first earnings surprise ranking | t2-earnings | OWNED | QuantConnect SUE factor; pandas; edgartools |
| 13 | Per-profile mandate that changes the verdict | t3-personalisation | OWNED | hkuds/vibe-trading enforcement |
| 14 | Episodic memory across decisions | t2-agentic | TIED | TradingAgents reflection log |
| 15 | Risk layer proved by domain sweep | t2-riskcontrol | OWNED | Nautilus risk engine; QuantConnect brokerage models; freqtrade protections |
| 16 | Pre-registered trading protocol, hash-committed | t2-agentic | TIED | serenity-guardrails journal |
| 17 | Perception layer: what the desk can see | t3-datasources | LOST | OpenBB keyless providers; TradingAgents feed list |
| 18 | Sentiment integrity vs. coordinated posting | t2-sentiment | TIED | finBERT |
| 19 | Crowd sentiment classification on TweetEval | t2-sentiment | LOST | RoB-RT; finBERT and VADER beside it |
| 20 | Self-evolving review rules | t3-review | TIED | TradingAgents reflection memory; QuantDinger diagnostic |
| 21 | Path-shape matching with a calibrated null | t3-decisionstress | LOST | stumpy matrix profile; a stock's own unconditional band |
| 22 | Session-aware execution | t3-execution | LOST | Bitget TWAP; Nautilus TwapAlgorithm; hftbacktest |
| 23 | Queue-position modelling | t2-execution | TIED | hftbacktest queue models |
| 24 | Portfolio allocation | t3-portfolio | TIED | Riskfolio-Lib NCO; skfolio |
| 25 | Regime-boundary detection | t3-decisionstress | TIED | stumpy FLUSS; ruptures |
| 26 | LUI intent routing | t3-lui | TIED | Rasa DIET classifier |
| 27 | Numeric decision grounding | t2-explainability | TIED | TradingAgents TraderProposal; pydantic validator |
| 28 | Structured filing extraction | t3-infoextract | TIED | FinanceBench published runs |
| 29 | Point-in-time correctness | t3-workbench | OWNED | OpenBB ODP; edgartools; Qlib PIT |
| 30 | Market sentiment | t2-sentiment | TIED | VADER and finBERT on the same posts |
| 31 | Factor Discovery Agent | t2-factordiscovery | TIED | FactorMiner and others |
| 32 | Analogue stress bands | t3-decisionstress | TIED | AnalogDesk (S2) |
| 33 | Portfolio copilot: post-trade beta | t3-portfolio | TIED | weekend-copilot (S2) |
| 34 | Portfolio stress when QQQ falls | t3-portfolio | TIED | skfolio vine copula and Entropy Pooling |
| 35 | Order splitting on realised cost | t3-execassist | TIED | Bitget's 60-second TWAP |
| 36 | Overnight hedge for an rToken holder | t3-portfolio | TIED | Ballast (S2) |
| 37 | A trader's thesis tested | t3-personalisation | TIED | optic-bitget (S2) |
| 38 | A trader's claims about the tape | t3-decisionstress | TIED | MirrorLine (S2) |
| 39 | A leveraged hold across the weekend | t3-decisionstress | TIED | baserate (S2) |
| 40 | Where a stop sits in the noise | t3-decisionstress | TIED | Rook (S2); a volatility stop |
| 41 | The perpetual-implied overnight open | t3-execassist | OWNED | gloaming and nocturne (S2); pre-market price |
| 42 | A book under the trader's own limits | t3-portfolio | TIED | skfolio MeanRisk |
| 43 | Chance of breaking a loss limit | t3-decisionstress | TIED | historical simulation |
| 44 | Injection withheld on unseen attacks | t2-riskcontrol | LOST | NVIDIA garak; ProtectAI deberta classifier |
| 45 | Research workbench data breadth | t3-datasources | OWNED | OpenBB keyless providers |
| 46 | Trader memory across sessions | t3-personalisation | TIED | mem0 |
| 47 | Attack corpus stopped before the model | t2-riskcontrol | TIED | HeyArka shield |
| 48 | Backtest critic | t1-validation | TIED | backtest-truth |
| 49 | Volatility-targeted sizing (GARCH) | t2-riskcontrol | LOST | fixed size; garchmethod |
| 50 | Exit cost from the displayed book | t3-execution | TIED | EGRESS (S2) |
| 51 | The desk's directional lean | t3-decisionstress | TIED | NIGHTWATCH AI |

Totals: 6 OWNED, 37 TIED, 0 IMPLEMENTED, 8 LOST.

The risk-layer row (15) rests on an exhaustive sweep of 2,177,280 states, in which the layer never increased exposure, reversed a side, turned an abstention into a trade or originated a leg; it narrowed the proposal in 1,335,312 states (61.3%) (`argus/data/risk_proof.json`). The same artefact lists four rules the sweep never reached (`unhedgeable_gap`, `gross_exposure`, `signed_exposure`, `max_position`). Against freqtrade's four protections at their defaults on 313 real strategy trades, ARGUS's locks kept 226 and the 87 it stopped netted -0.24%, a gain per gated trade of 0.0048 points [0.0012, 0.0086]; out of sample the point estimate keeps its sign but the interval includes zero.

### 6.4 The eight losses

Each loss below is the register's own wording reduced to its measurement.

**1. Deliberation priced as a trading cost** (row 1). The paper desk charges itself for the price drift during model thinking time. On 23,003 real decision instants from replayed Bitget books, the production charge has a mean squared error of 10.85 at a 3-second delay against 0.96 for a trailing empirical estimator; excess error 9.89 [9.20, 10.57], biased 6.2 times high. It beats the named specialist, LatencySensitiveBench's linear model (MSE 544), by a wide margin, and loses significantly to the textbook general estimator at every horizon. The fix is known (wire the trailing estimator into the charge) and not yet made.

**2. Cross-market cointegration** (row 7). The production pairs screen does not hold its false-discovery rate at its own target of 0.05: 18.9% on planted universes and 10.0% on null ones, where 128 general pipelines hold it. The out-of-sample-confirmed set holds it but with lower power than arch's Phillips-Ouliaris test under the same correction. On the real twelve-instrument universe every pipeline finds the same one or two pairs.

**3. Perception breadth of the paper desk** (row 17). The desk's decision-maker receives 6.2 data categories per stock against OpenBB's 12.4 keyless categories, behind on 10 of 10 symbols. OpenBB has options chains, dark-pool volume, price targets, ownership and dividends the desk never receives. The workbench answers do receive these (section 5.11); the desk's own feed does not.

**4. Crowd sentiment classification** (row 19). On TweetEval's 12,284 test tweets, ARGUS's crowd reading scores 0.544 macro-recall (95% CI 0.535 to 0.553) against RoB-RT's 0.729, and trails every published TweetEval baseline. It beats finBERT by 0.148. Closing it needs a tweet-domain transformer in the runtime, which would add torch to a two-dependency console.

**5. Path-shape matching** (row 21). On AnalogDesk's grid of 2,698 held-out queries, the path-shape band's Winkler score of 16.40% is significantly worse than the name's own unconditional band (15.14%, p = 0.001) and AnalogDesk's conformal analogues (15.25%, p = 0.002). Path shape adds no information over the name's own record at this horizon. The console's path-match line now says so.

**6. Session-aware execution** (row 22). On 19,440 parent orders replayed from Bitget books, 384 crossing a session boundary, Bitget's own TWAP with one-minute children costs 0.62bps less than ARGUS's session-aware schedule on the pre-registered primary, interval [0.01, 1.53]. ARGUS's own session-blind ablation does as well as the session-aware schedule, so the session model is not what pays on these parents. It does pay against a UTC-clock schedule across the daylight-saving change (2.03bps cheaper, [0.85, 3.54]).

**7. Injection screening on attacks it was not written for** (row 44). On 662 externally sampled prompts from the deepset prompt-injection corpus, a trained classifier (ProtectAI's deberta-v3-base, v2) withholds 109 of 263 attacks (41%) against ARGUS's rules' 41 (16%), McNemar p = 8e-12. The trade-off runs the other way on what the desk actually reads: on 5,836 lines of real desk text the classifier withholds 724 (12.4%) and ARGUS none, and on garak's latent-injection probes ARGUS withholds 3,582 of 3,652 against the classifier's 322. It remains a loss on the claim as written.

**8. Volatility-targeted sizing** (row 49). Walk-forward GARCH sizing, at the same average exposure as fixed size on the same signal and after Bitget's taker fee, had deeper worst drawdowns on all three coins tested (BTC 29.5% against 27.0%, ETH 54.1% against 51.4%, SOL 57.8% against 52.7%) and higher fees, because a size that moves every day trades every day. The reference implementation charges no fee. One signal, three coins, about 1,320 days each: a result about this case, not about volatility targeting everywhere.

Several TIED rows also hide a loss on a secondary measure, and the register keeps them: skfolio's vine wins the stress tail; mem0 recalls more; Rasa's DIET classifier scores higher raw accuracy (81.91% against 79.52%, p = 0.2649, not significant); ruptures beats ARGUS's original FLUSS regime tool decisively (mean F1 0.975 against 0.443), and the row is TIED only because a second ARGUS tool ties ruptures.

### 6.5 FinanceBench

On FinanceBench's 50 numeric ("metrics-generated") 10-K questions, ARGUS computes all 50 from the companies' filed XBRL within rounding of the gold answer; the best of the benchmark's sixteen human-graded GPT-4, Claude and Llama runs on the same questions answered 46 (`argus/data/financebench_xbrl.json`). The input differs: ARGUS reads structured filings, the benchmark's runs read filing text, which is harder. With ARGUS at 50 of 50, four discordant questions cannot give a paired p below 0.125, so the register grades the row TIED, not won. On the benchmark's other 100 questions, which ask for judgement in prose, ARGUS answers 5 correctly, abstains on 92 rather than guess, and 3 are disputed: 55 of 150 correct and none wrong overall. Neither class is held out: the 50 were seen while the engine was built, and the other 100 on 26 September 2026, when four fixes followed ten wrong answers. An ablation that removed the guard requiring every arithmetic phrase to be used by the formula produced extra answers, some wrong (an inventory turnover of 12.14 against a gold 9.5), which is why the guard stays.

### 6.6 Language understanding, routing and honesty

| Measurement | Result | Artefact |
|---|---|---|
| 240 questions in 12 languages, written blind by an agent that never saw the code, scored once | Console 196/240 (81.7%) with no model; 52.1% with patterns alone before the 25 September fixes; 204/240 (85.0%) with Qwen reading first | `lui_final_heldout_report.json` |
| Same set, kind model alone | 219/240 (91.3%); Arabic weakest at 3/7, carried by Qwen at 6/7 | `lui_final_heldout_report.json` |
| Which engine a question reaches | 213/240, 191/200 and 240/240 on three blind-written sets; the second and third were later tuned on, so none is strictly held out now | `lui_kind_routing.json` |
| Multi-part questions | 25/25 split correctly; 0 of 680 single questions split by mistake | `multistep_eval.json` |
| Questions no console can answer as asked | Before the honesty layer: 23 of 73 declined with the true reason, 21 with a wrong reason, 29 answered with figures. After: 73 of 73 with the true reason | `infeasibility_bench.json`, `honesty_eval.json` |
| False alarms of the honesty layer | 0 on 605 answerable questions | `honesty_eval.json` |
| Questions about the desk's own record (blind 200-question set never used for tuning) | 147/200 (73.5%), up from 131; Chinese 32 of 40, up from 25 | `lui_record_routing_2026-09-26.json` |
| Intent routing against Rasa DIET | 79.52% against 81.91% on 293 sealed questions, p = 0.2649 | `standing.json` row 26 |

The 81.7% figure is the honest headline for a visitor on the public console once the model budget is spent. It means roughly one question in five is still misread. The record path is the weakest measured part of the console, at 73.5% on its own blind set, and the kind model's weakest language is Arabic.

### 6.7 Engineering verification

| Check | Result |
|---|---|
| Tests collected | 13,558, collected on 6 October 2026 with outbound network blocked |
| Tests passing | As reported in the README for 6 October 2026: 12,830 passed, 77 skipped, 0 failed of 13,557 collected with every outbound connection refused, the 240 network tests run apart. Not re-run for this paper |
| Static types | `mypy --strict` clean on 690 source files |
| Modules | 152/152 registered modules importable; 18/18 named sub-themes resolve to a module and a test file, which is coverage, not a claim to lead them |
| Quoted figures | `python -m argus.eval.docclaims` checked 103 of 111 figures quoted in the public documents against their artefacts on 6 October 2026: 0 stale, 0 lagging, 8 unchecked |
| Code-level teardowns of external systems | 56, each citing `file:line` |

## 7. Safety and honesty

### 7.1 Injection screening

Every headline, filing and post the system reads was written by someone else. Untrusted evidence is quarantined before a model sees it: spotlighting with delimiters plus a standing system instruction, and AgentDojo's replace-don't-drop discipline, under which a hostile item keeps its id, source and timestamp and loses only its claim, so no downstream count silently shrinks. Each screening records the span that fired (`argus/src/argus/agents/quarantine.py`). On HeyArka's own 16-vector corpus, ARGUS stops 15 before the model against HeyArka's shield's 10 (p = 0.0625, TIED). Against a trained classifier on attacks the rules were not written for, ARGUS loses (section 6.4). The console's own readers are less exposed than the desk: the model on the console only fills a structured request whose names are checked against Bitget's contract list, and never writes text the reader sees other than translations whose numbers are locked.

### 7.2 What the console refuses

- Any instruction to place, change or cancel an order, checked before any other reading of the question; the refusal says nothing was sent.
- Questions about the trader's private Bitget account, which it holds no key to; it offers to read a stated book instead.
- Other traders' or institutions' live positions, which are not public; 13F filings and Bitget's aggregate long/short split are offered.
- Exact future prices; base rates are offered, labelled as base rates.
- Figures before the data begins or over horizons longer than the instrument has existed; the longest real span is offered.
- Names Bitget does not list, and ambiguous names, which get the candidate list.
- A premise contradicted by the filing or price series that settles it is corrected first.
- Any figure that cannot be sourced: the line says `missing`.

### 7.3 Why it places no orders

The handbook positions Track 3 as a workbench in which "human traders make final decisions". The console reads and advises; the trader places the order. Execution help ends at a priced plan and an Agent Hub command with `--paper-trading --dry-run`, which previews the request without sending it. A live order path would make a misread question an executed trade, and the 81.7% understanding figure in section 6.6 is not good enough to carry that. The console asks for no login and no API key, and tells the trader never to give one to anyone.

### 7.4 What the code will not let happen

From the engine underneath (`README.md`, "What the code will not let happen"): a zero-fee backtest cannot be constructed, because `CostModel` has no zero constructor; omitting `as_of` on a point-in-time query is a `TypeError`; the risk layer may only reduce exposure; a maker fee requires a book and goes through a queue model ported from hftbacktest; the factor searcher's context has no field that can hold a score; and the paper ledger is hash-chained, with its head submitted to four OpenTimestamps calendars so a later edit no longer matches a dated proof. One concurrency incident on 12 September 2026 duplicated seven sequence numbers; the repair and every changed link are published (`argus/data/paper_ledger_incidents.json`).

## 8. Limitations and what is not done

- **Real-user evidence is thin.** The console records anonymous usage events (visitors as a daily salted hash, no question text kept) and a "Did this answer your question?" control (`argus/src/argus/lui/usage.py`). This paper reports no session counts, task-completion rates or user feedback, because none was read from an artefact. The target is ten supervised sessions with traders in the stated segment.
- **Understanding is imperfect.** 81.7% of blind questions are read correctly without a model; questions about the desk's own record are weakest at 73.5%, and Arabic is the kind model's weakest language.
- **Bitget's toolkit answers poorly from this network.** 2 of 19 `bitget-signal` tools were reliable in the latest three-attempt sweep, and `bitget-mcp-server`'s equity quote was not answering on 6 October 2026. The console falls back to the public source a Skill names and says so.
- **No certified alpha.** All 8 vetted factor primitives are indistinguishable from shuffled data over 2,159 NVDA bars, none clears a single gate, and all 8 lose money after the 12bps round trip (`argus/data/overfit_gates.json`). The console says "no measured edge" when nothing it measured favours entering.
- **The paper desk is not a track record.** It has settled two valid trades, both losses (two further rows are void because they record fills the risk layer had refused); Sharpe, drawdown and win rate are undefined at that sample and printed as undefined. It is not offered as evidence for this entry.
- **Eight capabilities are LOST** and 52 register conditions are claimed without evidence on TIED and LOST rows.
- **Options, macro and the newcomer layer** have no same-input rival comparison in the register.
- **Translation fidelity** is checked line by line in production (numbers must match), but no systematic benchmark of translation quality exists.
- **The hosted console's own model calls** are not in the local Qwen meter. The local meter records 4,477 calls (4,372 answered) and 16.5 million tokens between 25 September and 5 October 2026, 69% of completion tokens spent reasoning, most of them by the paper desk's analysts (`argus/data/qwen_cost_ledger.jsonl`).
- **No live order path, by design** (section 7.3).
- **Known architectural debt** is listed in `ARGUS-ARCHITECTURE.md` section 9, including duplicated statistics that can disagree (four z-normalisation implementations split between population and sample variance).

## 9. Roadmap

1. Real trader sessions in the stated segment, with task completion measured by the in-page control.
2. Close the cheapest LOST rows on their stated routes: wire the trailing delay estimator into the deliberation charge; make the out-of-sample-confirmed pair set the production arm; absorb the keyless data categories the desk lacks.
3. A human-confirmed order path to Bitget's Demo environment through the Agentic account, sent only after the trader confirms the previewed Agent Hub command.
4. Options, macro and newcomer answers graded against named rivals on the same input.
5. A tweet-domain classifier for crowd sentiment if it can be added without breaking the console's dependency budget.

## 10. Appendix

### 10.1 How to reproduce

Nothing below needs a credential.

```bash
B=https://deploy-topaz-seven-64.vercel.app
# A live answer from Bitget's market data
curl -sG "$B/ask" --data-urlencode "q=What is BTC trading at right now?" | jq -r '.lines[0]'
# The record's size and whether its hash chain is intact
curl -s "$B/status?format=json" | jq '{entries, chain_intact, stale}'
# The MCP server's ten tools
curl -s -X POST "$B/mcp" -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | jq -r '.result.tools[].name'
```

From a clone of the public repository (https://github.com/Pratiikpy/argus-bitget):

```bash
cd argus
pip install -e ".[dev]"
python -m argus.status                  # modules and sub-themes, resolved by import
python -m argus.lui.server              # the console on http://127.0.0.1:8765
python -m argus.eval.standing           # re-derive the capability register from its artefacts
python -m argus.eval.docclaims --tests  # every quoted figure against its artefact
ARGUS_BLOCK_NETWORK=1 pytest -q -m "not network"
python -m argus.desk.portfolio --book TSLAUSDT=0.4 MSFTUSDT=0.4 METAUSDT=0.2 --add NVDAUSDT
```

`argus/VERIFY.md` reproduces individual headline numbers.

### 10.2 Glossary

| Term | Meaning |
|---|---|
| rToken | Bitget's tokenized spot version of a US stock or ETF |
| Stock perpetual | A perpetual futures contract on a US stock, trading around the clock, funded every eight hours |
| Anchor market | The US exchange where the underlying stock trades and its price is discovered |
| Session beta | Beta to QQQ measured separately over hours when the anchor market is open and when it is shut |
| Risk share | A position's contribution to total portfolio volatility, as a share; it sums to 100% across the book |
| Effective positions | The inverse Herfindahl of risk shares: how evenly risk is spread, not how many independent bets exist |
| Implied open | The price the stock's perpetual implies for the next regular-session open |
| bps | Basis points; 1bp is 0.01% |
| Round trip | Entering and exiting a position; 12bps at Bitget's published 0.06% taker fee per side |
| Almgren-Chriss | The closed-form schedule trading market impact against price risk over an execution horizon |
| TWAP | Time-weighted average price execution: equal slices at a fixed interval |
| XBRL | The SEC's structured financial-statement data |
| Point in time | A figure as it was knowable at a stated moment, before later restatements |
| SUE | Standardised unexpected earnings: the change in earnings scaled by its own variability |
| Winkler score | A score for prediction intervals that penalises width and misses; lower is better |
| McNemar test | A paired test on the questions where exactly one of two systems is right |
| OWNED, TIED, IMPLEMENTED, LOST | Register states: beaten under all thirteen conditions; no demonstrated advantage; built but unproven; a rival is materially better |
| Provenance label | The per-line tag naming where a line of an answer came from |
| MCP | Model Context Protocol, the interface through which other agents call ARGUS's tools |
| Skill | One of Bitget's research tool bundles served through `bitget-signal` |
