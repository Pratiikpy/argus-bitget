# ARGUS, explained in plain English — the whole product

ARGUS is one product, built as a single system. It is entered in Track 3; its Track 2 half is built,
running and shown here, and not filed while the desk has no settled trades — the reason is in Part B.

- **Part A is Track 3, AI Trading Desk** — the human decides, the AI researches. Judged 100% by
  people.
- **Part B is Track 2, Agentic Trading** — the AI decides, the risk layer binds. Judged 50% by
  numbers and 50% by people.
- **Part C is Track 1, Alpha Factory** — parked by decision, but the machinery is built and the
  results are in, and they are published here in full because they mostly say no.
- **Part D is the complete architecture** — all 147 modules across 20 packages named and placed,
  the seven layers and the rules that enforce them, one real decision walked end to end, every
  artefact and who reads it, and what runs on its own.
- **Part E is the Register** — the one part of this product whose value comes from elapsed time
  rather than from code, opened on 14 September 2026 and running since.
- **Part F is the ladder** — the queue model's case for OWNED, which reaches twelve of thirteen
  conditions and names the one it cannot meet; and funding carry, the first proposal in this
  system's history to get past the exposure gate and be refused by a gate that actually ran.

The evidence, the analysts, the checks, the engine, the risk layer and the ledger are shared. The
difference between the tracks is who has the final word.

This is written the way a judge reads: what the track asks for, what we actually built, how it
works, what you can run yourself, and — just as important — what it refuses to do and what it
cannot do yet. It is deliberately long. Nothing is summarised.

Every number here was measured by running something, and the file that produced it is named next to
it. Where we found our own mistakes, they are in here too.

---
---

# PART A — Track 3, AI Trading Desk

---

## Part 1 — What Track 3 asks for

The handbook describes the track like this:

> A natural-language-driven AI research workbench. AI processes information, invokes tools, and
> presents analysis; human traders make final decisions.

That sentence has four parts. Our answer to each, in one line:

| The ask | What ARGUS does |
|---|---|
| **Natural-language driven** | You type an ordinary question. No menus, no query language, no API. |
| **AI processes information** | It reads filings, insider trades, company financials, news and analyst estimates — and it never does the arithmetic itself. |
| **Invokes tools** | It calls Bitget's own research Skills and our own analysis tools, and records which ones answered and which did not. |
| **Presents analysis; human decides** | The console can read the record and explain it. It **cannot place, change or cancel an order** — that is refused by name, in code. |

The rest of this document is the long version.

---

## Part 2 — The positioning, step by step

### Step 1 — "Natural-language driven"

You talk to it like a person. These all work:

```bash
python -m argus.lui.cli "how did we do this week?"
python -m argus.lui.cli "why did we not trade NVDA?"
python -m argus.lui.cli "what do we hold?"
python -m argus.lui.cli "is the chain intact?"
```

Or open it in a browser:

```bash
python -m argus.lui.server       # then visit http://127.0.0.1:8765
```

**How it understands you.** The question is sorted into one of **14 kinds** — performance, why a
decision was taken, why the desk stood aside, the evidence behind a decision, calibration, chain
integrity, open positions, session state, market price, a list of decisions, an order instruction,
unsupported, ambiguous, and unknown.

**There is no AI model on this path.** Sorting the question is plain, deterministic code. This is a
deliberate choice and it is worth explaining, because it looks backwards for an "AI workbench":

- It is **instant**. The answer comes back in milliseconds, not seconds.
- It is **repeatable**. The same question always gets the same answer. A model would not guarantee
  that.
- Most importantly, it is **safe**. If a language model re-read our own decision record and
  summarised it back to you, you would now have two things that can be wrong instead of one. The
  record is the truth; nothing paraphrases it.

The AI model does the work only a model can do — reading a filing, writing a thesis. It does not sit
between you and your own history.

**How well does it actually understand?** We tested 25 ordinary ways of asking things. **All 25 are
understood.** That was not true two days ago: our own README used "how did we do this week?" as its
example, and the console *refused it*. We found that by running our own documentation as a stranger
would. Three of the most natural ways to ask about performance were unrecognised, while the refusal
message was advertising performance as something it could answer. There is now a test that takes
every capability the refusal message advertises and checks it is genuinely reachable — so the help
text cannot lie again.

### Step 2 — "AI processes information"

The desk reads, at every decision:

| Source | What it gives |
|---|---|
| Bitget market data | Live price, hourly candles, quoted spread, 24-hour volume |
| SEC EDGAR | 8-K and 10-Q filings, with the meaning of each 8-K item spelled out |
| SEC Form 4 | Insider buying and selling |
| SEC XBRL | The actual financial statements — revenue, net income, earnings per share |
| RSS | Eight news feeds plus a Yahoo feed for the specific company |
| Yahoo consensus | What analysts expect next quarter, and whether they have been raising or cutting |
| Bitget Skills | Bitget's own official research tools |

**The rule that governs all of it: the model never calculates.** Every number in every prompt is
worked out in Python first and handed to the model. The model interprets numbers; it never produces
them. This is why we can later check every figure in a thesis against the values the desk actually
held — and we do, automatically, on every decision.

**A real example of that checking working.** In decision number 41 of our live log, the AI wrote
that some insider sales were *"pre-arranged 10b5-1 plans already reflected in price"*. Both filings
say the opposite — the field `aff10b5One` is 0 — and our insider module only ever forwards a sale
when it is **not** pre-arranged. So the AI described the one property that would have meant it never
saw that evidence at all.

No number was wrong, so our numeric checker saw nothing. We built a second checker that compares
*claims* against the structured fields the evidence carries, and decision 41 is kept as its
permanent test case.

### Step 3 — "Invokes tools"

Bitget ships five official research Skills, served by one server with **19 tools**. The handbook
says an AI Trading Desk "can combine `bitget-signal`'s research Skills as its perception layer".

Here is the honest situation, and it is the most important paragraph in this document.

We catalogued **all 19 tools**. We call them and sort the result into states that cannot be confused
with each other: answered, empty, the tool reported an error, timed out, or unreachable. On
2026-09-13, 6 of 19 answered and 13 timed out at their upstream data providers.

On 2026-09-21 the same sweep reported **6 answering and 13 timing out**, where the day before it had reported 6 answering, 10 *empty* and 3 *tool errors* — same code, same endpoint, two incompatible descriptions of one service. So the sweep was repeated rather than re-quoted: `eval/skillreliability.py` calls every tool three times, spaced apart, and classifies each as reliable / intermittent / down. **9 of 19 answer three times out of three, and 3 of the 5 Skills have a reliable tool** — the single snapshot was understating the integration, in our own disfavour. There is no intermittent tool: each is consistently up or consistently down, so both snapshots were describing the same ten dead tools with different error text.

We could have wired in all 19 and let the dead ones return nothing. The feature list would look
three times longer and the product would behave identically. We chose to measure and publish the
real number instead. The measurement is a file on disk: `argus/data/bitget_skills_health.json`.

The 6 that work are all from the `technical-analysis` Skill, and they work **on our own instruments**
— NVDAUSDT, TSLAUSDT and the rest — giving RSI, MACD, ATR, Bollinger bands, support and resistance,
and moving averages.

**And we do not simply believe them.** We recompute the Skill's RSI ourselves from Bitget's own
candles and compare. Across 6 symbols, **all 6 agree**, and the largest disagreement is 2.19 points
out of 100. That check is in `argus/data/skill_crosscheck.json`. It is the difference between
integrating a data source and trusting one.

**Two real bugs were found doing this**, both in our own code:

1. Every call was sent with the same request ID. The protocol uses that ID to match an answer to its
   question, so with one ID for everything there was nothing to match on — and a test showed one
   tool returning **another tool's data**. Evidence labelled with the wrong source is worse than no
   evidence, because it looks trustworthy. Fixed: every call now gets its own ID and a reply that
   does not match is refused.
2. Bitget's own server marks failures with an `isError` flag and puts the error text where the data
   normally goes. We were reading the text and ignoring the flag, so a tool's own error message was
   one step away from becoming a market fact. Fixed.

### Step 4 — "Human traders make final decisions"

The console is **read-only over the record, by construction**.

If you type an order, it is refused by name:

```
> buy 100 NVDA
[order] this console reads the decision record; it does not place, change or cancel orders
```

This is not a filter on the output. "Order" is one of the 14 question kinds, checked **before**
anything else. That ordering was added because of a real failure found by testing: the phrase
"sell half of that" matched the word "sell" in the decision-explanation pattern, and the console
started explaining a past decision instead of refusing an instruction. Now the order check runs
first and cannot be reached around.

**Every answer carries its sources.** An answer that states something must say where it came from —
a ledger row, a computation, a named file. An answer with no source is only allowed when it is a
refusal. Refusing is treated as a real answer, not an error.

---

## Part 3 — The six sub-themes, one by one

### 3.1 Information Extraction and Signal Generation

> *"How does AI process unstructured info from earnings, macro, news?"*
> The handbook's own example: **conference call summary + expectation gap detection.**

**What we built: expectation gap detection, for real.** `argus/desk/expectation.py`

The idea is simple. A company reports a number. On its own that number means nothing — you need to
know what was *expected*. A reported figure without an expectation is a fact with no direction.

We have both halves now:

- what was **reported** — from the SEC's own XBRL financial statements;
- what is **expected** — live analyst consensus, reached without any API key.

Run it:

```bash
python -m argus.desk.expectation NVDA
```

Real output, 2026-09-13:

```
42 analysts expect eps_diluted of 2.4727 for the quarter ending 2026-10-31
that implies +90.2% against the 1.3000 actually delivered for the quarter ending 2025-10-26
delivered growth, newest first: +127.8%, +214.5%, +66.7%
the growth being expected is decelerating against the growth recently delivered
estimates revised up: 35 up / 1 down in 30 days
analysts are raising their numbers while the growth rate they expect is slowing
```

Read that last line again, because it is the whole point. **Analysts are raising their estimates,
and at the same time the growth rate they expect is falling.** Those two facts live in two different
places and are almost never read together. Neither half says anything alone. Together they describe
a specific situation that often ends badly.

**What it refuses to do, and why that matters.** A classic "earnings surprise" compares what a
company reported against the consensus that existed *for that same quarter*. No free source
publishes that. Yahoo only publishes forward estimates; Seeking Alpha, which does publish backward
ones, blocks us with an HTTP 403.

So we could have compared a reported quarter against a forward estimate that was never about it. It
would have produced a confident-looking number. **We report the surprise as unavailable, with the
reason attached to the record**, and never invent it.

**The trap we had to avoid.** Fiscal quarters do not line up with the calendar. NVDA's comparable
quarter is **370 days** back, not 365. So matching is done on a tolerance band around a year, and it
**refuses** if nothing falls inside it. Pairing the wrong two quarters would give a growth rate that
looks precise and describes nothing.

**Also here: the claim–evidence–signal graph.** Every claim the desk makes can be walked down to the
filing, the page and the cell it came from. Not citations — lineage. A claim with no evidence under
it cannot be added at all; the code raises an error.

### 3.2 Review and Self-Evolution

> *"After trading, how does AI help the trader review and iterate their research framework?"*

Most systems record profit and loss. That tells you **whether** it worked. It does not tell you
**why**, and you cannot improve from "you lost money".

**We separate four different ways of being wrong**, because they are four different lessons:

| What happened | What it is called | What you should actually do |
|---|---|---|
| Right direction, right size | `correct` | Nothing |
| Right direction, badly wrong size | `magnitude` | This is a **sizing** problem, not a research problem |
| Wrong, and you were very sure (>75%) | `overconfident` | The expensive failure. Cut size, not conviction |
| Wrong, and you were appropriately unsure | `thesis` | The idea was wrong. That is normal and cheap |

Merging these into "accuracy" destroys the lesson. Being right about direction and wrong about size
is a completely different problem from being confidently wrong.

**The calibration gap.** We take your stated confidence and subtract your actual hit rate. If you
say 90% and you are right 60% of the time, the gap is +30% and you are overconfident. The system
says so in plain words: *"overconfident by 30%: stated confidence exceeds realised hit rate across
N decisions — reduce size, not conviction."*

**It refuses to draw a pattern from too little data.** Under 5 graded decisions it says so and stops:
*"only 3 graded decisions — too few to draw a pattern."* A confident-sounding pattern from four data
points is worse than silence.

**Root-cause diagnosis** (`argus/desk/rootcause.py`) goes further, with four separate diagnoses:

- **overconfidence** versus **miscalibration**. These were one thing until a real case forced them
  apart: a decision stated at 0.35 confidence that realised 0% is *badly calibrated*, but calling it
  "overconfident" is wrong — the desk was appropriately unsure. They now have different names and
  different remedies, split at a stated confidence of 0.6.
- **sizing, not thesis** — you are right and still losing.
- **fee drag** — the fees, not the ideas, are eating the returns. Triggered at 30% of gross.
- **repeated blind spot** — you have discounted the same kind of evidence three or more times.

### 3.3 Decision Stress Testing

> *"Before opening a position, how does AI retrieve historically similar scenarios?"*

We do this at **two levels**, because a single position and a whole book are different questions.

**Level one — one position, six scenarios.** Base, bear (−5%), bull (+5%), gap down (−12%),
liquidity collapse (−3% with depth at 30% of normal), and correlation break.

The property that makes this more than a table: **the cost of getting out rises as liquidity falls.**
A position is hardest to exit exactly when you most want to. Most stress tables quietly assume the
exit costs the same in a crash as on a calm day.

We do not assume it — we **simulate** it. The position is liquidated into an order book populated by
a realistic agent mix at the scenario's liquidity, and the cost is the real slippage plus the fee.

And it can return an answer arithmetic cannot produce: **`exitable = False`**. On a thin book a
position can be impossible to liquidate at any price. A fee divided by a multiplier always says yes.
A simulation can say no.

**Level two — the whole book.** `argus/desk/portfolio.py`, in two ways that answer different
questions:

1. **A market shock, carried through each position's own beta.** The usual mistake is to shock every
   position independently, which invents a diversification that does not exist. Positions in one
   book move together. On our live example book, a −10% day in the benchmark costs **−12.43%**, with
   NVDA worst at **−17.09%**.
2. **The worst run that actually happened.** We slide over the real history and find the worst
   24-hour window *for these exact weights*. It assumes no distribution and needs no covariance
   matrix — it replays what the market really did. Result: **−3.24%**, driven by TSLA at −7.83%.

Those two disagree, and the disagreement is the information. The parametric shock describes a bad
day that has not happened yet; the replay describes the worst that has. **None of PyPortfolioOpt,
cvxportfolio, qlib or riskparity.py implements the second** — we read all four to check.

**Historical analogue retrieval** (`argus/desk/analogue.py`) finds past states that resemble now and
reports what followed them. It has a strict rule: nothing stamped at or after the decision instant
can ever be returned, so a search run inside a backtest cannot see its own future. It also collapses
overlapping windows, because the same week counted five times is not five pieces of evidence.

### 3.4 Personalised Research Workbench

> *"How to build a customized workbench with a clear thesis?"*

The submission form explicitly refuses **"all traders"** as an answer. So the profile has to be real.

**Our profile changes the verdict, not the wording.** Two shipped profiles:

| | Conservative income | Aggressive event trader |
|---|---|---|
| Most in one position | 5% | 25% |
| Most in one sector | 15% | 60% |
| Holding horizon | 720 hours | 48 hours |
| Loss it can accept | 3% | 15% |
| Evidence it prefers | filings | news, social, transcripts |

Run the demonstration:

```bash
python -m argus.desk.personalisation
```

The same trade — NVDA, 20,000, 24 hours, with the thesis conceding an 8% loss in its bad case:

```
conservative income:    refused — the thesis concedes 8% in its bad case against a 3% tolerance
aggressive event trader: taken  — within a 25% position limit and a 48h horizon
```

**Same market. Same evidence. Opposite answers.**

Three design decisions worth explaining:

- **A horizon breach is a refusal, never a smaller trade.** A thesis that needs a month does not
  become suitable for a two-day trader by halving the size. It is a different trader's trade.
- **A loss breach is also a refusal.** Making the position smaller does not change a percentage.
- **A size breach is a resize**, because the idea survives at a smaller ticket.

**The part we are most careful about: the harness is built so it can fail.** Of the four shipped
example trades, **one diverges and three do not**. For those three it says, in those words:
*"personalisation did not bind here. That is a fact about this proposal, not evidence that profiles
work."*

A demonstration that can only ever confirm is not a demonstration.

**How rare is this?** We read **16 other AI trading systems** to find out. The result:

- **Seven have no user-profile concept at all** — TradingAgents, FinRobot, AI-Trader, AutoHedge,
  CryptoTrade, live-trade-bench, qlib.
- **Several have a profile that is only text in a prompt** — FinAgent, TraderHarness, VerumTrade. A
  limit written into a prompt is a limit the model can ignore. FinAgent ships six profiles and
  **every one of its own configs hardcodes the same one**, so its authors never tested divergence
  either.
- **Exactly one** — hkuds/vibe-trading — has a real, code-enforced, test-proven mandate that changes
  a verdict on identical market state.

We match that one, and add something it does not have: we report when personalisation **did not**
bind.

### 3.5 Execution Assistance

> *"After trader decision, how does AI handle order splitting and slippage management?"*

Two separate questions, and we keep them separate: **how** to trade, and **how fast**.

**How — `plan_execution`.** Splits an order into slices and states the cost of each style. Three
properties taken from reading real execution engines, not invented:

- **Market impact grows faster than order size** (cvxportfolio uses an exponent of 1.5), so splitting
  genuinely helps — but only until the fee dominates.
- **A quiet session is a bad time to be patient.** A resting limit order on a thin book gets filled
  when the market moves against you and missed when it does not. So when the anchor market is
  asleep, urgency goes **up**, not down. Most systems have this backwards.
- **A patient slice is only cheap if it actually fills.** If you give us a real order book, every
  passive slice is priced through a simulated fill — the real blend of patient fills and chasing.
  If you do not, passive slices are quoted at the **expensive taker rate**, not the cheap maker rate.

That last one is deliberate and it fixed a real bug. Our own backtest once counted a cheap maker fee
on a slice labelled "passive limit" without ever checking it would fill. **An unevidenced maker fee
is the single most common way an execution plan flatters itself.**

**How fast — `trajectory`.** The Almgren–Chriss optimal trajectory, the standard closed-form answer
to trading impact against risk. Trade fast and you pay impact; trade slow and the price can move
against you.

**Our addition, which no system we read has: it refuses to schedule through a market-open boundary.**
Volatility is not continuous across a market opening — it jumps. A schedule that smooths an order
across 3am Sunday to 10am Monday is solving a maths problem that does not describe reality. Ours
declines and says so.

### 3.6 Open Theme — the Portfolio Copilot

> The handbook's example: *a portfolio-aware AI PM that evaluates how a proposed trade changes
> **beta, sector and factor exposures, correlation, and concentration**, with stress tests or hedge
> suggestions.*

This is one command:

```bash
python -m argus.desk.portfolio --book TSLAUSDT=0.4 MSFTUSDT=0.4 METAUSDT=0.2 --add NVDAUSDT
```

Real output on live Bitget candles:

```
question: should I add NVDAUSDT to this book?
718 aligned hourly bars, 126 of them while the anchor market was open

  open-session beta 1.13 -> 1.24
  NVDAUSDT would carry 16% of total portfolio risk (was new position)
  risk spread across 2.2 -> 2.9 effective position(s)
  most correlated with TSLAUSDT at +0.20

  weight against risk, after the trade:
    TSLAUSDT   weight  32.0%   risk  51.5%
    MSFTUSDT   weight  32.0%   risk  18.0%
    NVDAUSDT   weight  20.0%   risk  16.4%
    METAUSDT   weight  16.0%   risk  14.1%

  if the benchmark moves:
    benchmark -10%   book  -12.43%   worst NVDAUSDT -17.09%

  the worst 24-bar window in the observed history would have moved this book -3.24%

  factor exposure: equal-weight market +1.073, open session +1.096, shut session +1.057
```

**The single most useful line: TSLA is 32% of the money and 51.5% of the risk.** No position-size
column anywhere shows that. A trader "diversified across four names" is carrying half their risk in
one of them.

#### The discovery behind this module

A stock perpetual trades 24/7. The US stock it tracks does not. So roughly **82% of hourly bars happen while
the American market is shut** — when there is no real price discovery.

That means a beta measured over *all* bars is an average dominated by the hours that matter least.
We measured it on 30 days of hourly candles for every stock perpetual against QQQ:

| Symbol | Blended beta | Open-session beta | Shut-session beta |
|---|---|---|---|
| AAPLUSDT | 0.254 | **0.668** | 0.090 |
| TSLAUSDT | 1.069 | **1.594** | 0.861 |
| METAUSDT | 0.682 | **1.290** | 0.441 |
| MSFTUSDT | 0.340 | **0.577** | 0.245 |
| NVDAUSDT | 1.412 | **1.709** | 1.296 |

**Open-session beta is higher than shut-session beta on 10 of 11 stock perpetuals.**

AAPL is the extreme. Its blended beta of 0.254 would tell a trader the name is nearly
market-neutral. During the session that actually prices it, the figure is **0.668** — more than two
and a half times higher.

**And here is why we trust the finding.** The only two exceptions are TQQQ (2.905 open, 2.908 shut)
and SQQQ (−2.910, −2.885). Those are leveraged ETFs on the benchmark itself — their beta is fixed at
±3x by construction and *should not* vary by session. So the two instruments that are mechanically
pinned are exactly the two that show no effect, and all nine ordinary stocks show it. That is not
what a measurement error looks like.

So every exposure is reported **per session**. The blended number is still shown, with a warning,
because people expect it — and because they should see why it misleads them.

The file is `argus/data/session_beta.json` and you can regenerate it.

#### Two mistakes we caught in our own work here

1. **We claimed something our arithmetic did not deliver.** We had a measure called
   `effective_bets`, documented as "a book of ten names that all move together is one bet". Our own
   test showed three perfectly correlated positions scoring **2.71**, not 1. The inverse Herfindahl
   measures how evenly risk is *spread*, not how many *independent* bets exist. We renamed it to
   `effective_positions`, withdrew the claim, and marked the correlation-aware version as **not
   built** rather than implying we had it.
2. **We regressed a 718-bar book against a 126-bar factor.** The function truncates both to the
   shorter length, which quietly paired the book's first 126 bars with a factor from a completely
   different period. It reported exposures near zero where the correct answer is +1.07. Truncating
   two series is not the same as aligning them. Found by comparing the command's output against a
   standalone calculation and seeing two different answers.

---

## Part 4 — The four judging criteria

### 4.1 Feature depth: data sources

Seven independent classes of data, each with the hard part handled:

| Source | The hard part, and how it is handled |
|---|---|
| Bitget market | Two clocks kept separate — the token's and the stock's |
| SEC 8-K / 10-Q | Each 8-K item code translated to plain English, so a reader needs no SEC table |
| SEC Form 4 | One insider decision split across seven price tiers is collapsed into **one** event with a volume-weighted price. Compensation machinery — grants, tax withholding, option exercises — is discarded and the discarded count reported. Our first live Form 4 was 172,507 NVDA shares "acquired" at a price of zero: an RSU vest. Counted as a purchase it is the largest insider buy of the quarter and it means nothing |
| SEC XBRL | From one quarterly 10-Q, the SEC returns a 181-day cumulative figure and a 90-day quarterly figure with the same end date and the same fiscal period. For NVDA those are 177.8B and 96.2B. Reading the wrong one overstates quarterly revenue by **85%**. We separate them by duration. Restatements are point-in-time: a query as of a date returns the figure **as it stood then** |
| RSS | Eight feeds plus a per-company Yahoo feed, filtered for relevance, with the kept and discarded counts reported |
| Yahoo consensus | Keyless, behind a cookie-and-crumb handshake because a plain request now returns 401. Gives consensus EPS, revenue, the spread of analyst opinion, and the 7- and 30-day revision trail |
| Bitget Skills | All 19 tools catalogued and health-classified; 6 answer; their numbers are recomputed against our own candles |

### 4.2 Feature depth: Skill integration count **and effectiveness**

The criterion names two things and they pull in opposite directions. Count is easy to inflate.
Effectiveness is not.

**Our count: 19 tools catalogued, 6 answering, 1 of the 5 official Skills fully reached.**

We could have written "19 Skills integrated". It would have been technically true and behaviourally
meaningless, because 13 of them return nothing. Instead:

- every tool has a **health state** that cannot be confused with market information. "Timed out" and
  "answered with nothing" are different states, because the first is an unknown and the second might
  genuinely mean no news;
- the live cycle calls **only** the tools the last health sweep found answering, so a decision never
  waits on 13 dead upstreams;
- a tool that fails contributes **no evidence at all** — not a piece of evidence saying "no data",
  which a reasoning system would then reason about;
- and the Skill's own numbers are **verified**: RSI recomputed from Bitget candles, 6 of 6 symbols
  agreeing within 2.19 points.

We think a measured 6 is worth more than a claimed 19, and the measurement is on disk either way.

### 4.3 Research quality

Four things we do that we have not seen done together:

**1. Every figure in a thesis is checked against the numbers the desk actually had.** Not spot
checks — every figure, every decision, automatically. It handles the ways a model naturally writes
numbers: a fact of 0.0203 quoted as "2%" or as "203bps" both resolve correctly.

This found a real false alarm in our own system. The checker flagged "+0.24%" as unsupported when it
was the 24-hour change the market data had supplied as 0.0024. The checking function had always
accepted a list of evidence values for exactly this — and nothing was passing them. A checker that
cries wolf on correct writing is worse than no checker, because the next real finding gets ignored.

**2. Claims are checked, not just numbers** — the decision-41 case described earlier.

**3. We published our own negative result — and then published a harsher correction to it.**
We ran four anti-overfit gates over our eight vetted factor primitives on 2,159 real NVDA bars.
**All eight are indistinguishable from randomly shuffled data. None returns a verdict for want of
data — which is not a pass. None clears every gate, and all eight are net-negative after the 12bps
round trip.** Statistical structure and tradeable edge are different things.

⚠️ This replaced an earlier four-and-four split whose artefact **no code could regenerate**
(rebuilt 2026-09-15 as `argus.research.overfit_gates`). `data/factor_lab.json` reaches the same
all-eight verdict by an independent route. The correction made our own result worse, which is the
only direction a correction is worth trusting in. That file is published, not buried:
`argus/data/overfit_gates.json`.

**4. We read the source before writing our own.** There are **56 code-level teardowns** in
`research/architecture/`, each citing file and line. They have repeatedly changed what we built:

- Our order state machine was written from memory with 11 states. Reading `nautilus_trader`'s own
  enum showed **15**, with `DENIED` (our risk layer) separate from `REJECTED` (the venue), and a
  documented trap where filtering on "is open" drops submitted orders from reconciliation. Three
  design errors, found in one file read.
- We credited FactorMiner with keeping scores away from its factor generator. Reading it showed the
  generator receives a "memory signal" containing success grades and the names of admitted factors.
  We corrected our own docstring and designed our memory so an outcome cannot reach the proposer —
  with a guard that raises at import time if anyone ever adds such a field.
- We had recorded one repository's risk enforcement as "proved". A second read found it is
  commented-out pseudocode behind a "not implemented" error, and the real risk module it advertises
  is never imported by anything. Corrected in our notes.

### 4.4 LUI fluency

- **14 question types**, sorted deterministically with no model in the path.
- **25 of 25 natural phrasings understood**, including "how did we do this week?", "how much did we
  make?", "what do we hold?", "did we beat the fee?", "summarise the week", "are we long NVDA?".
- **Refusal is a first-class answer.** "I did not recognise that question well enough to answer it
  from the record" is a legitimate reply, and it lists what *can* be answered.
- **Orders are refused before anything else is considered.**
- **Every answer that asserts something carries its sources.** A ledger row, a computation, a file.
- **Multi-turn memory.** "Why that one?" after a list resolves to the decision you meant.
- **Speed budgets.** Each question type has a time budget and the console reports the time it took
  against it, e.g. `[performance] 0ms / 500ms`.

The fluency work came directly from executing our own documentation as a stranger would, which is
how the refused README example was found.

### 4.5 Personalised thesis

Covered in detail in 3.4. The short version: two profiles, the same market state, different
verdicts, enforced in code — plus a harness that reports when the profile made no difference.

Of 16 systems read, one other does this properly.

---

## Part 5 — What we cannot do, said plainly

A judge should not have to discover these.

- **The hosted console is a read-only view.** It answers research questions live and reads the
  desk's ledger, but the paper-trading cycle itself runs on our machine and publishes to it.
- **There is no correlation-aware "effective number of bets".** It needs a principal-component
  decomposition we have not built. We mark it not built rather than implying the number we do have
  means that.
- **We cannot compute a consensus earnings surprise**, for the reason given in 3.1 — no free source
  publishes the consensus that existed for an already-reported quarter. We compute the time-series
  one instead (standardised unexpected earnings from SEC filings), and say which it is.
- **13 of Bitget's 19 Skill tools do not answer** from our network. We report six.
- **We have no equity broker**, so a hedge in the underlying stock is visible but not reachable by
  us. During market hours the record now says exactly that — the hedge exists and *we* have no route
  to it — rather than the earlier and false "no hedge placeable".
- **No usage data.** This has not been put in front of traders yet. The plan is ten supervised
  sessions measuring whether the risk-share figure changes the ticket.
- **Superiority is claimed only where a named competitor was run on the same input.** 20 of 38
  capabilities are OWNED under that rule; the rest are TIED or IMPLEMENTED and are called that, and
  every comparison we lost is published on `/wrong`. On 2026-09-24 seven earlier OWNED grades were
  withdrawn because the rival they beat does not lead its sub-theme; they stay IMPLEMENTED until
  the systems that do lead it are run on the same input.

---

## Part 6 — The five commands that show the whole track

```bash
# 1. Ask it questions in English (no key needed, no network)
python -m argus.lui.cli "how did we do this week?"
python -m argus.lui.server                       # or in a browser

# 2. The portfolio copilot — the complete research task
python -m argus.desk.portfolio --book TSLAUSDT=0.4 MSFTUSDT=0.4 METAUSDT=0.2 --add NVDAUSDT

# 3. Expectation gap — expected against delivered
python -m argus.desk.expectation NVDA

# 4. Personalisation — same trade, two traders, different answers
python -m argus.desk.personalisation

# 5. Which Bitget Skills actually answer
python -m argus.market.skills --symbol NVDAUSDT
```

And to check the whole thing is what this document says it is:

```bash
pytest                    # 6,700 tests
python -m argus.status    # 152/152 modules importable, 18/18 sub-themes, artefacts on disk
```

---

## Part 7 — The one-paragraph version

ARGUS is a natural-language research desk for tokenized US stocks. You ask it ordinary questions and
it answers from a record it cannot edit, with sources attached, refusing anything that would place a
trade. Underneath, it reads filings, insider transactions, company financials, news and analyst
estimates, calls Bitget's own research Skills, and checks their numbers against its own. It tells you
what a proposed trade does to your book — not just its size, but its share of the *risk*, its beta in
the session that actually prices it, what it is correlated with, and what the worst day on record
would have done to you. It knows the difference between a conservative trader and an aggressive one,
and proves it by giving them different answers to the same trade. And when it cannot do something —
compute a surprise, reach a Skill, tell you how many independent bets you hold — it says so, in the
same place it would have put the number.


---
---

# PART B — Track 2, Agentic Trading

Everything above was Track 3, where a human decides. This part is Track 2, where the **AI decides**.
They are the same product. The same evidence, the same analysts, the same risk layer and the same
ledger serve both — the difference is who has the final word. In Track 3 the console cannot place an
order. In Track 2 the model places it, and the whole design question becomes: *how do you let a
language model trade real capital and still be able to trust the record afterwards?*

Track 2 is scored **50% on numbers and 50% by judges.** The numbers are Sharpe ratio, maximum
drawdown and win rate from a paper-trading log. We cannot control the judges. We can control the
build, and we can be honest about the numbers — which, as of this writing, is the part of this
document that hurts.

---

## Part 8 — What Track 2 asks for

The handbook's positioning:

> The LLM is the primary trading decision-maker, not just an assistant. The Agent must sense the
> environment, make independent judgments, and autonomously place orders with risk controls.

Four parts again. One line each, then the long version:

| The ask | What ARGUS does |
|---|---|
| **The LLM is the decision-maker** | Qwen decides the verdict, size, confidence, thesis and the conditions that would prove it wrong. Nothing downstream invents a trade. |
| **Senses the environment** | Reads the live market, filings, insider trades, financials, news, consensus and Bitget's Skills — and is told what its own thinking costs. |
| **Independent judgment** | Five specialist analysts advise; the model decides; agreement between analysts is discounted when it came from one source. |
| **Places orders with risk controls** | Orders are signed to the exact intent that was approved. The risk layer can shrink or refuse a trade and can **never create or enlarge one**, in code. |

---

## Part 9 — The positioning, step by step

### Step 1 — "The LLM is the primary decision-maker, not just an assistant"

This is a positioning rule, and most systems in this field fail one half of it.

- Some keep the model **off the order path**. It writes summaries, or the prompt forbids it from
  recommending, or there is no broker at all. In our corpus: AI-Trader's model only writes
  summaries, Vibe-Trading's prompt forbids recommendations, inalpha keeps the model off the order
  path by design, FinRobot has no broker.
- Others let the model decide and then have **no binding risk layer** — the risk rules live in a
  prompt, which a model can be talked out of. TradingAgents' own teardown says its risk management
  is "via prompt guidance, not code enforcement".

ARGUS passes both halves at once, and the way it does so is the core of the architecture:

**The model decides.** It receives a structured frame — price, session state, position, the
round-trip cost, its own deliberation cost, the hedge menu, and the analyst panel — and returns a
decision with **seven possible verdicts**: TRADE, REDUCE, HEDGE, DELAY, NO_TRADE, HUMAN_REVIEW, and
DATA_INSUFFICIENT. A direction alone is not a decision, so it must also give a size, a confidence,
a thesis, and at least one **invalidation condition** — a specific, checkable observation that
would prove the thesis wrong.

**The risk layer may only reduce — and that sentence needs to be three sentences, not one.** It is
called the Constitution, and it has six possible responses: ALLOW, RESIZE, REQUIRE_HEDGE, DELAY,
REJECT, FLATTEN. Saying "every one is a reduction" was too loose, and an external review was right
to attack it. Three different invariants were being run together under one phrase, and only the
first is what the code actually enforces:

1. **Directional exposure to the original thesis cannot increase.** This is enforced and provable.
   `decision/verdicts.py:182-183` raises `ConstitutionViolation` if REQUIRE_HEDGE ever returns a
   quantity above the one proposed, REJECT sets it to zero, RESIZE can only narrow, and FLATTEN
   moves the book toward zero. The exhaustive sweep in `eval/riskproof.py` checks this over 2,177,280
   states and finds no violation.
2. **Gross notional CAN increase, by exactly one hedge leg.** REQUIRE_HEDGE attaches a hedge
   instrument (`verdicts.py:184`). Net risk falls, which is the point of a hedge, but the sum of
   absolute notional across legs rises. A reader told "every response is a reduction" would not
   expect that, and a hostile expert would find it.
3. **The Constitution cannot introduce a new economic thesis.** A hedge leg is permitted only in a
   declared hedge relationship to the position the model proposed; it is never a free-standing
   directional bet of the risk layer's own. **Now proved.** `eval/riskproof.py` carries two further
   invariants — *intent monotonicity* and *risk monotonicity* under a declared functional — across
   the same 2,177,280 states, with four mutant policies the prover must catch: a hedge in the symbol it
   claims to hedge, a leg smuggled in under ALLOW, an enlargement, and a stale quantity left behind
   by a rejection. All four are caught.

So the precise claim is: **the Constitution cannot increase directional exposure and cannot
originate a thesis; it can add a hedge leg that raises gross notional while lowering net risk.**
All three are now proved over the same exhaustive domain. Gross notional is deliberately **not**
asserted monotone, because a hedge leg raises it — and that sentence is in the prover's own output,
not only here.

So the economic choice is always the model's, and the guard is always binding. That is the whole
point. The model is genuinely the decision-maker *and* it genuinely cannot blow up the book.

**Then the model gets to respond.** After the Constitution narrows a decision, the model is told what
happened and asked to respond. It can accept the smaller size and rewrite its thesis for it. This
matters for one reason: a model that is silently overruled did not decide anything, and the record
would show a decision the model never actually made.

**And every decision produces an Autonomy Proof** — an artefact from which a judge can verify, without
trusting our code, four things:

1. `attests_llm_decided` — the model, not a rule, made the economic choice;
2. `constitution_intervened` — whether the risk layer changed anything;
3. `constitution_only_reduced` — that whatever it changed was a reduction, checked from the original
   and revised intents alone;
4. `llm_changed_its_mind` — whether the model responded to the constraint rather than repeating
   itself.

### Step 2 — "Sense the environment"

Everything in Part 2 of the Track 3 section applies here — the seven data sources, the model never
calculating, the numeric and claim-level checks. Two things are specific to Track 2.

**It is told what its own thinking costs.** This is the mechanism we believe nothing else in the
field has. A reasoning model takes seconds to think. During those seconds the market moves. Every
finance-agent harness we tore down treats that latency as free, which was true when tools ran in
milliseconds and is false when a model runs for forty seconds.

We price it. The cost of a reasoning budget in basis points depends on the budget (3 seconds for
OFF, 8 for LOW, 40 for FULL) and on how thin the book is: off-hours depth is roughly a third of
regular hours, so the same delay crosses three times as much book. The multipliers are 1× in
regular hours, 2× in extended hours, and 3× overnight and at weekends.

The result: at a weekend, a full reasoning budget costs **27.20 basis points** on top of a 12bps fee
— more than double the fee. A low budget costs 18.80. So the hurdle a trade must clear is not "the
fee"; it is "the fee plus the cost of the thinking that produced the trade", and the model is told
that number before it decides. Off-hours, thinking about the trade costs more than the trade.

**It knows which session it is in.** The token trades 24 hours a day, seven days a week. The stock it
tracks does not. So the desk carries two clocks: the token's, which never stops, and the anchor's,
which is open in four phases — regular hours, extended hours, overnight, and weekend. A fact can be
*knowable* on the token clock and *unpriceable* on the anchor clock at the same instant. Sunday at
3am, a headline is tradeable and there is no real price discovery for another 30 hours. Every
decision records which session it was made in and how many hours remain until genuine price
discovery.

### Step 3 — "Make independent judgments"

**Five analysts, and the model.** Four specialist analysts — event, sentiment, earnings, and
cross-asset — plus the earnings decomposition and the causal chain, each run on the model at a low
reasoning budget and return *intelligence*, never decisions. Then the portfolio manager, also the
model but at the full budget, decides.

**Agreement is discounted when it comes from one source.** This is called the Source Independence
Graph and it is ours. Five agents agreeing after reading one Reuters article is one piece of
evidence, not five. Every analyst returns the IDs of the evidence it relied on, and the panel's
consensus confidence is discounted by how much provenance was shared. The record reports an
*independence ratio*: distinct sources per analyst. A ratio near 1.0 means every view rests on its
own evidence; near 0 means the panel is one source wearing several hats.

**Disagreement is typed, not averaged.** When analysts disagree, the disagreement is recorded as one
of three kinds — direction, magnitude, or conviction — and resolved on stated grounds, which a reader
can dispute. The resolution can be "neither". And because our analysts currently run one after
another, any agreement is labelled as *possible contagion* rather than consensus, because a later
analyst may have been influenced by an earlier one. We say that in the record rather than claiming an
independence we have not built.

**Which analysts run is decided from the evidence, and priced.** Nothing in our corpus conditions its
panel on the evidence, and nothing skips an analyst because running it is not worth the cost. Both
are arithmetic here, because deliberation already has a price. An analyst whose channels hold nothing
it can act on is not run; the skip is recorded *as a skip*, with the count of evidence left unread,
so a thin panel is never mistaken for a thin market. On a live cycle this saved 4.5 basis points of
deliberation against a 12bps round trip.

### Step 4 — "Autonomously place orders with risk controls"

**Orders are bound to the approved intent.** Every order carries the hash of the exact decision the
Constitution approved. An order that does not match its approved intent cannot be submitted.

**The order state machine has 15 states, and we did not invent them.** We wrote it from memory with
11. Then we read `nautilus_trader`'s own enum and found 15 — with `DENIED` (our risk layer said no)
separate from `REJECTED` (the venue said no), `PENDING_UPDATE` and `PENDING_CANCEL` as simultaneously
open *and* in flight, `VOIDED` for fills busted after the fact, and a documented trap where filtering
on "is open" alone drops submitted orders from reconciliation. Three design errors found in one file
read.

**The venue path is real.** The Bitget signature is accepted, and a real demo order round-tripped —
order ID `1482642956228374529`, product type `SUSDT-FUTURES`. What is not real yet is a *settled*
trade, and that is Part 11.

**The risk layer has four parts, and each is honest about its limits:**

1. **The Constitution** — the per-decision rules described in Step 1. Minimum confidence to trade is
   0.55. When the NAV is stale and the anchor is shut, it delays. When no hedge is placeable and the
   position exceeds the unhedged cap, it resizes. Every rule names the constraint that bound, so
   "which rule fired, how often" can be counted.
2. **The circuit breaker** — six rules over the whole book, all evaluated every time rather than
   stopping at the first, because a report that names one reason when three applied understates the
   situation. Total drawdown of 10% halts. Session drawdown of 4% halts. Four consecutive losses
   halt. A 3-sigma adverse move halts. Stale evidence halts. And a **drawdown ladder** tightens
   appetite as losses deepen: full size until 2% down, then 75%, then 50% at 6%, then zero at 10%. A
   halted book cannot jump straight back to active; it must pass through reduce-only.
3. **Position sizing** — half-Kelly, taken from Bastion's implementation. But Kelly sizing on a
   stated confidence is only safe if the confidence means something, so sizing is **gated on
   calibration**: it will not size on confidence until there are at least 20 graded decisions and
   the calibration error is under 0.15. Before that, it refuses to use the confidence at all.
4. **The mandate** — per-trader limits, described in the Track 3 section, wired into the same path
   with the same asymmetry.

---

## Part 10 — The six Track 2 sub-themes, one by one

Every named sub-theme is an entry point into the *same loop*, not a separate product. That is
deliberate: five bolt-on agents would be five things to keep consistent. One engine with five
entrances is one thing.

### 10.1 Event-Driven Agent

> *"How do news / announcements / macro events drive autonomous Agent trading?"*

**What it reads.** SEC 8-K filings, with each item code translated — 2.02 is "results of
operations", 5.02 is "officer or director change", 4.02 is "non-reliance on prior financials". Eight
news feeds plus a per-company feed. Federal Reserve and SEC press releases as macro.

**What it produces: a causal chain, not just a direction.** The event analyst does not only say "up
or down". It states the *transmission chain* — event, then the mechanism, then the effect on the
company, then the price — as a sequence of links. Each link is recorded and can be graded separately
at the next price discovery.

**Why that matters: right direction through broken reasoning is scored as a failure.** A P&L scorer
records a lucky win as a win. We do not. If the price went the way the analyst said but the chain
it stated was wrong at two links, the chain is graded as broken. A system that rewards being right
for the wrong reasons will learn to be confidently wrong.

**Grounding.** Every figure in the event thesis is checked against the numbers the desk actually
held, and the claim checker looks for properties the thesis asserts that the evidence contradicts.

**Where it stands honestly.** The mechanism is complete and has run on every one of 86 live
decisions. What it has not yet done is *fire on a real event during a session where trading was the
right answer*, because the log so far covers a weekend and the early hours of a Monday. The causal
chains exist and are gradable; none has been graded against a price discovery yet.

### 10.2 Market Sentiment Agent

> *"How does real-time social / forum / X sentiment translate into position signals?"*

**What it reads.** The Fear & Greed index from Bitget's sentiment Skill, when it answers; the
technical readings from Bitget's technical-analysis Skill — RSI, MACD, Bollinger bands — which do
answer, on our own symbols; and social-channel evidence generally.

**What we refuse to do, and why.** Sentiment reads *mood*, and mood has no vocabulary that separates
signal from noise the way a material-event word list does. So the sentiment analyst is gated on the
**volume and credibility** of what arrives, never on a keyword list that would encode a guess about
what sentiment looks like.

**The honest state of the feeds.** The sentiment-specific Bitget tools — the Fear & Greed index,
long/short ratios, open interest, taker ratios — **timed out** at their upstream providers on our
sweep. The technical-analysis tools answered. So the sentiment analyst's evidence today is thinner
than the earnings analyst's, and the health report says exactly which feeds are dark rather than
letting an empty feed look like a quiet market. When the sentiment feeds come back, nothing needs
rebuilding; the cycle reads the health report and starts calling them.

**Where it stands.** Live on every cycle. The Bitget Skills feeding it are measured, not assumed, and
their RSI is cross-checked against our own computation.

### 10.3 Earnings-Driven Trading Agent

> *"How does the Agent autonomously interpret earnings / conference calls and execute?"*

This is the sub-theme with the most depth, and the expectation-gap work in Track 3 was built for it.

**Seven dimensions, kept apart.** An earnings print is not one number. The earnings analyst
decomposes it into **seven separate surprises**: the reported figure, the consensus, guidance, the
narrative, valuation, management credibility, and the Q&A. Each is scored on its own.

**It refuses to average disagreement into a weak consensus.** A beat with cut guidance is a
different asset from a beat with raised guidance, and averaging them produces a mild positive that
describes neither. If the headline and the forward dimensions point opposite ways, the read is
flagged as **self-contradictory** and the dominant dimension is named. The record says *"earnings
print contradicts itself: headline +0.80, forward −0.60 — forward dominates"*.

**What it now has that it did not:** the actual reported statements from XBRL, point-in-time, with
the cumulative-versus-quarterly duration trap handled; the live consensus and its revision trail;
and the expectation gap between them. On the last live cycle the earnings analyst ran with **19
structured records** behind it, up from 6 the day before, because the SEC outage described below
was fixed.

**Where it stands.** Complete and live. NVDA's next print is in the log's future; the machinery is
waiting for it.

### 10.4 Cross-Asset Execution Agent

> *"How does the Agent manage rToken and Crypto positions simultaneously?"*

**The hedge menu, ranked and priced.** For any position, the cross-asset analyst is handed a menu of
hedge candidates, each carrying: how much risk it would remove, how confident we are in that
correlation *in the current regime*, how much liquidity is available, how likely it is to fill, how
stable the basis is, and what it costs. The headline metric is **Risk Neutralisation Efficiency** —
risk removed per basis point of all-in cost. The menu is ranked by it.

**The empty-menu case is the normal case, and it is modelled.** For roughly 65 hours a week the
anchor stock market is shut. During that time a hedge in the underlying stock *exists* and cannot be
reached. We keep it on the menu with an execution probability of zero rather than omitting it,
because the fact that a perfect hedge exists and cannot be placed is exactly the information the
whole design turns on. Zero of the four closest tokenized-equity competitors we read compute
hedgeable-now versus carried, in any form.

**A correction we made to ourselves.** During regular hours the code used to hand the desk an
*empty* menu too — which put a false sentence into every regular-hours decision: "no hedge
placeable". During regular hours the hedge is plainly placeable by somebody. What is true is
narrower: **ARGUS has no equity broker**, so the venue is open, the liquidity is there, and *we* have
no route to it. Those are different facts, and conflating them flatters us — "the market gave me no
hedge" excuses an unhedged position, "I have not built the connection" does not. The record now says
the true one.

**Execution itself.** Order splitting with each style priced honestly, and an optimal trajectory that
refuses to schedule across a market-open boundary — both described in the Track 3 execution section,
because they are the same code.

**Where it stands.** The analyst, the menu and the residual pricing are live on every decision. The
crypto side of "rToken and crypto simultaneously" is thin: we trade stock perpetuals, and the crypto
instruments appear as hedge candidates and as Bitget-Skill sentiment inputs, not as positions the
desk currently opens. We say that rather than imply a two-asset book we have not run.

### 10.5 Factor Discovery Agent

> *"How does the Agent autonomously propose hypotheses, discover alpha factors, and translate into
> tradable decisions?"*

Almost every system in this space answers this with a loop: propose a factor, score it, feed the
score back to the proposer, repeat. That loop is the defect that makes their results meaningless,
because a proposer that sees its own scores optimises for the score, and on any finite sample the
score can be maximised by noise.

We read the three leading systems and found:

- **mcts-llm-alpha** computes a genuine overfitting score, then *unconditionally overwrites it* with
  the generating model's own opinion of itself.
- **RD-Agent**, Microsoft's, uses a single model to both propose factors and judge them. No
  independent evaluator, no purged cross-validation, no deflated Sharpe, no trial counter.
- **FactorMiner** was the one we credited with keeping scores away from the generator. Reading its
  source showed the generator receives a "memory signal" containing success grades and the names of
  admitted factors. Coarse grades are still outcomes. We corrected our own docstring.
- **NexQuant**, recommended to us as the best factor-discovery repo, turned out to be a live
  demonstration of the defect: it mutates the top-five-by-score candidates, its optimiser returns
  Sharpe directly, and it has no purged cross-validation, no deflated Sharpe, no multiple-testing
  correction, and a confirmed look-ahead bug in its resampling.

**Our lab is built so the contamination is structurally impossible.** The proposer is handed a frozen
object called `ProposerContext` that **has no field for a score**. Not a filtered score, not an
aggregate — the shape cannot carry one. Contaminating the loop would require editing that class,
which is visible in a diff. Evaluation is deterministic Python over price data; no model grades a
factor.

**Every factor walks a lifecycle it cannot skip:** proposed, formalised, backtested, cost-checked,
out-of-sample tested, deflated-Sharpe gated, certified, deployed, decayed, retired. A factor that
tries to skip a gate raises an error. `RETIRED` exists and fires on measured decay, because a library
that only ever grows is lying about decay.

**Every proposal increments a trial counter, and the counter feeds the Deflated Sharpe gate.** A
search that reports its best result without saying how many times it looked has reported the
maximum of a noise distribution.

**Four more gates before anything is called certified:** stability of the factor's predictive power
across sub-periods; sub-sample stress; a placebo test against 200 random shufflings of the data;
and a decay half-life. Each returns PASS, FAIL or INCONCLUSIVE, and inconclusive does not certify —
"we could not tell" is not "it passed".

**The proposer cannot express arbitrary code.** Factors are built from a typed grammar with no
execution surface, so there is nothing to sandbox. The field's loops generate Python and run it in a
sandbox; ours cannot express Python at all.

**The memory across runs publishes identity only.** The lab remembers every hypothesis it has
evaluated so it never pays for one twice — and what it tells the proposer is *what has been tried*
and *what could not be measured*, never *what worked*. There is a guard that raises at import time if
anyone adds an outcome-carrying field to what the proposer sees, and a test that scores factors with
a telltale number and asserts it appears nowhere in the proposer's view. Run twice on live bars, the
first run scored eight primitives and the second scored none and suppressed eight.

**The honest result, published.** Over 2,159 real hourly NVDA bars: **four of our eight primitives
are indistinguishable from randomly shuffled data. The four that pass every gate still lose money
after the 12bps round trip.** Statistical structure and tradeable edge are different things, and on
this venue the difference is the fee. That is a real negative result on real data and it is in
`argus/data/overfit_gates.json`.

**Where it stands.** The lab is complete and disciplined. It has not yet found a factor worth
trading, and it says so. "Translate into tradable decisions" is built — a certified factor would
feed the desk — and has nothing to feed it with yet. We consider that the correct outcome of an
honest search on a venue where the fee exceeds the edge, and a better result than a certified factor
would have been.

### 10.6 Open Theme — Agent Evaluation

> The handbook's example: *"Agent evaluation / benchmarks covering decision consistency,
> risk-violation rate, max drawdown, stress behaviour, human-takeover rate, and incremental value
> over fixed-rule or Human + AI baselines."*

This is where ARGUS is furthest ahead of the field, and it is where we recommend entering.

**The ledger is the foundation.** Every decision is written before its outcome exists, with the
evidence, the hurdle, the session, the stated confidence, the thesis and the invalidation
conditions. Each row carries the hash of the previous one, so editing any historical row breaks
every hash after it. Settlement fields are deliberately excluded from that hash, which is why
attaching an outcome later does not invalidate the chain — and also why a settled outcome can never
rewrite the decision it settles.

**That exclusion was tested adversarially, and it found a real gap.** Excluding settlement fields
from the decision's own hash is correct — but on 2026-09-22 we copied the live ledger, edited a
settled decision's P&L directly in the file (bypassing the code path entirely, exactly what a
compromised host or a malicious insider with file access would do), and the chain still verified as
intact. A settled outcome could be silently rewritten after the fact, undetected. Fixed with a
second, independent mechanism: attaching an outcome now also commits a separately chain-linked
"settlement seal," so editing a trade's recorded P&L after the fact breaks that seal's own link the
same way editing a decision breaks the main chain. All 531 real decisions already settled at the
time were backfilled — honestly caveated as proving unchanged since the backfill, not since the
original settlement, since nothing committed to those specific outcomes at the time. Every decision
settled from that date on carries the real, contemporaneous guarantee.

**Truncation is detectable.** A hash chain proves what is present is unedited and says nothing about
what was *removed*. We found that by deleting the last seven entries and watching the chain verify
as intact. Since the rows most worth deleting are the most recent losses, that was the largest hole
in the integrity story. A head anchor now records the chain's current end, so a truncated log
disagrees with its anchor.

**Concurrent writers cannot corrupt it.** We found that one too, the hard way: the scheduled cycle
and a manual run overlapped, both loaded the log at entry 40, and both wrote a different decision
as entry 41. By the time it was noticed there were seven duplicated sequence numbers across 62
rows. The chain's own verification caught it. Now every append takes an exclusive lock and re-reads
the head from disk inside the lock. The corrupted log was repaired **in the open** — the original
archived, its SHA-256 committed to an incident record, every changed row listed — never quietly.

**Ranked by calibration, not by return.** On a venue where the fee exceeds most effects, knowing what
you do not know is worth more than a lucky quarter. The observatory computes the Brier score, the
expected calibration error and the full reliability curve, and its leaderboard ranks by calibration
error. It is deterministic arithmetic. Microsoft's FinanceBenchmark grades every subject with a fixed
third model; we never have one model grade another.

**Abstention is scored.** Standing aside is a decision and it is not automatically right. When the
desk declines to trade, the move it declined is recorded at the time, so a correct refusal and a
desk paralysed by its own hurdle can be told apart. Nothing else in our corpus scores standing
aside.

**The risk layer is audited, not asserted.** How often did the Constitution actually change a
decision? Until we built the audit, nothing could say — the binding-constraint field was designed
to be countable and nothing was counting it. The audit grades the layer's *exercise*, and it refuses
to call a clean record a pass: a layer that **never fires is untested, not safe**, and one that fires
on **everything means the model is not deciding**, which is a positioning failure. It also checks the
only-reduce asymmetry on every decision.

**A baseline that isolates the desk.** Comparing a desk that traded NVDA and TSLA against the S&P
measures which universe was hot. We compare against **holding the desk's own picks** — buy each name
once at the first price the desk paid, hold to the cutoff, no turnover. Beating that means the
trading added value; losing to it means the desk would have done better choosing once and going
home. LLM-Trading-Lab, the corpus's one real-money LLM experiment, computed an index comparison and
then disclaimed it; ai-hedge-fund has none at all.

**Decision consistency, measured.** Decisions are grouped into episodes — the same symbol, verdict
and session over a window — so that twelve near-identical weekend abstentions are one episode, not
twelve independent judgements. Our first live measurement grouped 39 decisions into 8 episodes.

**Self-audit.** The system checks its own log for a Sharpe collapse between in-sample and
out-of-sample, a widening drawdown, calibration divergence, and look-ahead. An inconclusive result
counts against publishability.

**Where it stands.** Every one of these is built, tested and running. What every one of them
reports today is *undefined, no settled trades* — and reports it in those words. Part 11 is about
why.

---

## Part 11 — The quantitative half, brutally

**The paper-trading log has 673 decisions on record. Every one of them is a refusal, and 625 have settled as abstentions.** Two rows (seq 264, 265) carry `verdict: trade` and are **void** — they recorded fills the risk layer had refused, a `paper/runner.py` defect found and disclosed on 2026-09-20; they stay in the chain unedited and are excluded from every derived figure (`paper/corrections.py`).

Track 2 is 50% scored on Sharpe ratio, maximum drawdown and win rate computed from that log. With
zero trades, those three numbers **do not exist**. Not "are zero" — do not exist. Half of the track
is currently unscoreable for us, and no amount of architecture compensates for that.

Here is exactly why, and why we believe it is the correct state rather than a broken one.

**The log opened on a weekend.** The first entry is Saturday 2026-09-12 at 09:30 UTC. The anchor
stock market was shut and would stay shut for roughly 45 hours. Every thesis in the log cites the
session, the hours to genuine price discovery, and an observed move smaller than the total hurdle.
The desk is refusing negative-expectancy trades, which is exactly the behaviour we built.

**The hurdle is not the problem.** We measured whether the hurdle is even clearable: on 1,007
hourly candles per symbol, the two-hour absolute move exceeds the 18.8bps total hurdle **61–89% of
the time during US regular hours on eleven of twelve symbols** — COIN at 88.8%, MSTR 83.1%, NVDA
74.7%. Only QQQ falls short at 48.3%. So the desk is not structurally locked out. It has not yet
seen a session where trading was the right answer.

**A large move is necessary, not sufficient.** It still has to be called in the right direction, and
the model has to clear its own confidence floor of 0.55. So even in a regular-hours session, trades
are not guaranteed.

**What we will not do.** We will not write decisions into the log that the desk did not take. The
hash chain, the head anchor, the write lock, and the as-of gate on settlement all exist to make
exactly that kind of fabrication detectable. A fabricated trade would be caught by our own tooling.

**What is proven while we wait.** The full trade path is exercised end to end by tests: record a
trade, settle it, get the net P&L in the right direction for longs, shorts and a sub-fee move, see
all three scored numbers appear, and see the chain still verify afterwards. Settlement used to treat
any unrecognised side as short — which would have silently inverted every long — and now refuses.
The scheduled cycle, which had *never actually fired* when we first checked (its last run time was
the 1999 epoch), now runs every two hours and has run on its own since. The performance module
returns `null` and the reason for each undefined statistic, because `"sharpe": 0.0` reads as a real
flat result and it is not one.

**What has to happen.** A US regular-hours session in which the model judges a move likely enough
and large enough to clear the hurdle and its own confidence floor. The scheduled cycle will be
running through every one of them. When trades settle, three things become true at once: the three
scored numbers appear, the baseline comparison becomes defined, and the risk audit gains something
to count.

**If that does not happen before the deadline**, the honest submission says so and points at the
abstention record and the hurdle-clearance measurement as the evidence that the desk was refusing
correctly. That is a weaker entry than one with a Sharpe ratio. It is a stronger entry than one
with a fabricated one.

---

## Part 12 — The judging focus, one by one

### 12.1 Paper-trading Sharpe, max drawdown, win rate

**Where we are:** undefined, for the reason above.

**How they will be computed when they exist:**

- **Annualised at 365 periods, not 252.** The stock perpetuals trade seven days a week. Using the equity-market
  convention would overstate the Sharpe by the square root of 365/252.
- **A Sharpe over too few days is refused.** Below the minimum, the module says "the window is N
  days; a standard deviation needs at least M" rather than printing a number from three data points.
- **A win rate over zero trades is undefined, not zero per cent.** The module says so in those words.
- **Per-symbol contribution is always reported**, with the share of the result carried by the
  single largest name, because one lucky instrument producing the whole return from four trades is
  the oldest trap in strategy evaluation and we have fallen into it before on another project.
- **Costs are charged on entry** using the same cost model the backtest uses. A paper log that
  reports gross P&L is marketing.

### 12.2 Decision explainability

Every decision can be explained at five levels, and each level is a real artefact rather than a
generated summary.

1. **The thesis and its invalidation conditions**, written by the model, in the ledger row. The
   invalidation is a specific, checkable observation that would prove the thesis wrong — "if the
   larger claim were independently confirmed, a bigger reduction would be warranted".
2. **The evidence it saw**, with provenance and credibility, and the analyst panel's views with the
   source IDs each relied on.
3. **The checks that ran against its own reasoning** — every figure grounded against the numbers the
   desk held, every checkable claim compared to the evidence's structured fields, every
   disagreement between analysts typed and resolved on stated grounds. These are written to a
   sidecar file per decision and any finding is surfaced in the cycle summary.
4. **What the risk layer did**, with the constraint that bound named.
5. **The Autonomy Proof** — the four attestations a judge can verify from the artefact alone.

And the natural-language console can walk you through any of it: "why did we not trade NVDA?"
returns the session, the hours to discovery, the hurdle, the move, and the sources.

**The part we are proudest of is that the checks found real things.** Decision 41's hallucinated
property. A false alarm where the checker flagged a correct figure because nothing was passing the
evidence values. A note-flagging list written from memory that missed a real grounding failure on
the very next cycle, now pinned to each module's actual wording by a test. Explainability that had
never caught anything would be decoration.

### 12.3 Agent architecture quality

The architecture is one loop:

```
evidence -> analyst selection (priced) -> five analysts -> source-independence discount
         -> conflict typing -> Meta-PM decides -> Constitution narrows -> Meta-PM responds
         -> mandate binds -> signed order -> Autonomy Proof -> hash-chained ledger
```

What we think makes it good, and what we know makes it weak:

**Good.**
- The decision-maker and the guard are separated by an **asymmetry enforced in code**, not by
  convention. The model can propose anything; the Constitution can only reduce. No prompt can change
  that.
- **Every number the model sees is computed in Python.** The model never does arithmetic, which is
  what makes the record auditable.
- **Deliberation is a cost term.** Nothing else we have found treats the agent's own thinking time as
  a price.
- **Two clocks.** No single-clock point-in-time model can express "knowable but unpriceable", and it
  is the normal state of this venue for 65 hours a week.
- **Point-in-time retrieval is enforced by the interface**, not by discipline. The evidence store has
  no "latest" and no default; a query without an as-of date is a type error. This is a direct
  response to two real defects we found by reading FinMem and FinAgent, both of which *intended*
  point-in-time retrieval and failed because forgetting the filter was possible.
- **Seventeen defects found by running, not reading.** They are all recorded in the master plan, with
  the fix. We think a system whose builders can list sixteen things they got wrong is more
  trustworthy than one whose builders can list none.

**Weak, and stated.**
- **No checkpoint and resume.** A cycle interrupted mid-flight starts over. We read LangGraph's
  checkpointing in detail and have not built our own.
- **The model is one model.** There is no ensemble, no second opinion from a different model family,
  and no adversarial pass that can overturn a decision on its own counter-case. The desk generates a
  counter-case; nothing acts on it.

### 12.4 Risk control layer effectiveness

This is the criterion we have thought hardest about, because "effective" is easy to fake.

**The four layers** — Constitution, circuit breaker, calibration-gated sizing, and mandate — are
described in Part 9. All are enforced in code. None can be talked out of by the model.

**What "effective" means, and what it does not.** A risk layer with zero violations on a clean
record has proved nothing; it has not been asked anything. So the audit grades the layer's
*exercise*:

- **UNTESTED** — it has never bound, or there are too few decisions to say. This is our current
  state, and the audit says so in those words: "the risk layer has not been exercised: 0
  interventions over N decisions, 0 of which offered a position to reduce. This is untested, not
  safe."
- **LIGHT** — it binds rarely.
- **ACTIVE** — it binds sometimes. The only healthy grade.
- **DOMINANT** — it binds on nearly everything, which means the model is not the decision-maker.
  That is a positioning failure, not a safety record.

**The denominator is honest.** Interventions are counted against decisions that *offered a position
to reduce*, not against all decisions. A desk that abstained 86 times gave its Constitution nothing
to bind on, and dividing by 86 would report an idle layer where there was never an invocation.

**The asymmetry is checked per decision.** If any intervention ever increased exposure, the audit
reports it first, before any statistic, as a broken invariant.

**What we cannot yet say.** Whether an intervention was *correct* — whether the trade it shrank would
have lost money — needs the outcome of that trade, and there are no settled trades. The audit reports
that as undefined with the reason, rather than printing a zero that reads as "the risk layer saved
nothing".

**The correction we made to our own claim.** The risk layer used to be told, during regular hours,
that no hedge existed. That was false, and it would have shaped every regular-hours decision on the
day the scored trades are meant to happen. The true statement — the hedge exists, we have no broker
route to it — is now what the layer sees.

---

## Part 13 — What we cannot do on Track 2, said plainly

- **No settled trades.** The scored half of the track is undefined until a regular-hours session
  produces them.
- **Zero certified factors.** The lab found none worth trading and says so.
- **Analysts run one after another**, and the record labels their agreement as possible contagion.
- **No checkpoint and resume** of an interrupted cycle.
- **No equity broker**, so hedges in the underlying stock are visible and unreachable.
- **One model, no ensemble, no adversarial overturn.**
- **"Owned" is a conjunction, not a score.** An owned capability needs a reproduced baseline, a
  same-input comparison, an out-of-sample test, an ablation and an adversarial test, among thirteen
  conditions. On 2026-09-12 zero met that bar; the register now reads 20 of 38 capabilities are
  OWNED, each re-derived from its artefacts by `eval/standing.py`, never from prose.

---

## Part 14 — The commands that show Track 2

```bash
# One decision cycle against the live market and the live model (needs the Qwen key)
python -m argus.paper.runner --once --symbols NVDAUSDT,TSLAUSDT

# The scored numbers, or why each is undefined
python -m argus.paper.runner --report

# Calibration, episodes, abstention quality, chain integrity
python -m argus.eval.scorecard

# Did the risk layer ever change a decision, and how often
python -m argus.eval.riskaudit

# The desk against holding its own picks
python -m argus.eval.baseline

# Verify the hash chain; plan-only, writes nothing
python -m argus.paper.repair data/paper_ledger.jsonl

# The factor lab, with its trial counter and its cemetery
python -m argus.research.factor_lab
```

---

## Part 15 — Both tracks in one paragraph

ARGUS is one system with two doors. Through the Track 3 door, a human asks questions in plain
English and gets answers from a record with sources attached, sees what a trade does to their book's
risk rather than its size, and is told plainly when something cannot be computed. Through the Track
2 door, the model decides — what to trade, how much, how sure it is, and what would prove it wrong —
and a risk layer that can only ever shrink or refuse makes sure the record afterwards is one you can
trust. Both doors open onto the same evidence, the same analysts, the same checks against the
model's own reasoning, and the same hash-chained ledger. Every decision on the ledger is a
refusal and none is a trade, which makes half of Track 2 unscoreable today, and we have written that down
in the same place we would have written a Sharpe ratio.


---
---

# PART C — Track 1, Alpha Factory

**First, the honest framing.** Track 1 was parked by a deliberate decision early on: the two
submissions go to Tracks 2 and 3. But the product touches Track 1 anyway, because the machinery that
scores Track 2 and the research that feeds Track 3 are the same machinery Track 1 is judged on. So
this part describes what exists, what it found, and — because Track 1 is scored purely on numbers —
exactly what those numbers say, including the ones that say no.

## Part 16 — What Track 1 asks for

> Use AI to build runnable US stock quantitative / algorithmic trading strategies. AI is a tool for
> strategy development (writing code, optimizing parameters, generating signals); the core is
> **strategy effectiveness and verifiability**.

Required: strategy code, and a backtest of at least 60 days total with at least 30 days
out-of-sample. Judged on Sharpe, Sortino, max drawdown, turnover, out-of-sample Sharpe decay (alert
if out-of-sample is under half of in-sample), and rolling 30-day Sharpe stability. Pure quantitative
scoring.

The phrase that matters is **verifiability**. Anyone can produce a backtest with a Sharpe of 2. The
question is whether the number would survive someone trying to break it. Our whole Track 1 position
is that we tried to break our own numbers first, and we publish what broke.

## Part 17 — The backtest engine: what it refuses to let you do

Every Track 1 strategy runs through `argus/backtest/engine.py`, and the engine has three properties
that are not options:

**A zero-fee run is structurally impossible.** The cost model must be constructed explicitly, and
the round trip on this venue is 12 basis points. There is no flag to turn it off. Every result is
net of fees before it is anything else, and the gross figure is reported beside it so the fee's bite
is visible.

**Signals are lagged by construction.** A signal computed from bar *t* can only trade at bar *t+1*.
This is the single most common look-ahead bug in the field — using the close of a bar to trade at
the open of the same bar — and here it cannot be expressed.

**Every run is split chronologically into in-sample and out-of-sample**, and the out-of-sample slice
is scored on its own. The decay between them is computed with the handbook's own alert rule: if
out-of-sample Sharpe is under half of in-sample, the strategy is flagged.

**Every judged metric is computed per strategy** in `argus/backtest/metrics.py`: total return,
Sharpe, Sortino, maximum drawdown, turnover, win rate and trade count. Rolling Sharpe over a window
is also computed by the engine, because a full-sample Sharpe built from one profitable fortnight is
not stable and only the rolling series exposes that.

**The metrics have the field's endemic bugs designed out.** Two examples:

- A Sharpe over a series with no variance is undefined. The common library answer is `NaN`, and
  `NaN > threshold` is `False`, so an undefined Sharpe silently *passes* every gate. Ours raises.
- The Deflated Sharpe Ratio — the correction for having searched — needs the number of trials. A
  library that returns it without one has returned nothing. Ours takes the trial count from the
  factor lab's counter and refuses without it.

**Annualised at 365, not 252.** These instruments trade seven days a week.

## Part 18 — The six Track 1 sub-themes, one by one

Every strategy below is a signal function fed to the same engine, so every one is scored net of the
same fee, lagged the same way, and split the same way. That is the point: one gate, all candidates.

### 18.1 Arbitrage

> *Instantaneous spreads between rToken and native stock, and NAV premium/discount arbitrage under
> mint/redeem.*

**We studied it. It does not pay.** `argus/research/arbitrage_study.py` measured the apparent spread
between the stock perpetual and its reference across 28,067 observations over four session phases:

| Session | Observations | Median apparent spread | Monetisable after 12bps fee |
|---|---|---|---|
| Regular hours | 5,070 | 6.51 bps | 8.7% of the time |
| Extended hours | 8,450 | 6.64 bps | 10.0% |
| Overnight | 6,760 | 3.07 bps | 7.4% |
| Weekend | 7,787 | 2.13 bps | 4.9% |

The median apparent spread is **4.27 basis points**. The round trip costs **12**. So the spread is
monetisable **4.58% of the time overall**, and least of all at weekends — exactly when the naive
story says it should be widest. The arbitrage exists on a screen and not in an account.

That is our arbitrage sub-theme: a measurement that says no, with the machinery visible. We do not
run an arbitrage strategy, and we say so rather than shipping one that loses.

### 18.2 After-Hours Information Pricing

> *Macro events continue during US market closures while rToken trades 7×24, pricing information in
> advance.*

**This is where the only real signal lives, and it is fragile.**

The gap study (`argus/research/gap_study.py`) looked at 832 closed sessions — 156 weekends and 676
overnights — and asked: when the token moves while the stock market is shut, does the reopen
*continue* that move or reverse it?

| | Weekend | Overnight |
|---|---|---|
| Sessions | 156 | 676 |
| Median move while closed | 119 bps | 123 bps |
| Median move at reopen | 55 bps | 58 bps |
| Reopen move clears the fee | 84.6% | 90.4% |
| **Continuation rate** | **57.7%** | **50.6%** |

Overnight, the reopen is a coin flip: 50.6%. There is no information there.

Weekends look different: 57.7% continuation, 90 of 156. We tested that three ways, and this is the
honest result:

1. **Significance.** p = 0.033 on the raw test — but *not* significant once corrected for the
   number of things we looked at. A p of 0.03 after a search is what noise looks like.
2. **Train/test split.** The first half of the weekends continued 48.7% of the time. The second
   half continued 66.7%. **The effect does not survive the split** — it lives entirely in the second
   half of the sample, which is the signature of something that has not been there long enough to
   trust.
3. **Per symbol.** 8 of 13 symbols sit above the coin flip. AMZN at 83% is individually
   significant; most are not.

**So the strategies here are built on a hypothesis the data does not confirm**, and we say so in the
module's own docstring: *"None of these is expected to beat the fee."*

The strategies: `weekend_only`, `hold_through_closure`, `trade_only_when_open`, `closure_momentum`,
`closure_reversion`, and two adapted from Barclay and Hendershott's weighted price contribution
(`afterhours_attenuation_fade`, `afterhours_discovery_countdown`) — adapted because their reference
market was *closed* and ours is *attenuated* about sevenfold, not shut.

### 18.3 Cross-Market Correlation

> *Correlation shifts between rToken and native stock across time periods.*

**We report a usable hedge ratio, not a correlation.** A correlation of 0.71 measured across a shut
session is a stale print, not a relationship. `crossmarket_basis_reversion` and
`crossmarket_stale_corrected` use a rolling hedge ratio with a staleness correction, so a reading
taken while the anchor was asleep is weighted down rather than trusted.

The session-beta finding in the Track 3 section is the same phenomenon measured from the other side:
open-session beta exceeds shut-session beta on 10 of 11 stock perpetuals. A cross-market strategy that
uses one blended number is using the wrong one four days out of seven.

### 18.4 rToken Factor Strategies

> *Traditional factors behave differently under rToken's low-liquidity, retail-dominated structure.*

**Our factors are session-structural, not generic price factors.** `rtoken_session_boundary` and
`rtoken_attenuation_carry` are built from the basis, the time to the next price discovery, and the
attenuation state — things that exist *only* because the token trades when the stock does not.
Momentum and reversion factors that work on the stock have no reason to work on a token that spends
82% of its bars without price discovery, and we did not assume they would.

The factor lab described in the Track 2 section is the discipline layer over all of this: a proposer
that cannot see scores, a lifecycle a factor cannot skip, a trial counter that feeds the deflated
Sharpe, and four anti-overfit gates. Its result on our eight primitives — four indistinguishable
from noise, the four survivors still losing after the fee — is the Track 1 result in miniature.

### 18.5 Cross-Asset Allocation / Rotation

> *Asset switching driven by macro factors, capital flows, risk appetite.*

`rotation_regime_switch` allocates by regime, and `rotation_hedgeability_weighted` penalises an
allocation by how hedgeable it is right now — an allocation into a name whose anchor is shut for 40
hours carries risk that cannot be laid off, and the weight is reduced for it. That penalty is the
hedgeability surface from the Track 2 risk layer, reused as an allocation input.

### 18.6 Open Theme — Execution-Aware Alpha

> The handbook's example: *execution-aware alpha that incorporates fees, slippage, funding, market
> impact, and strategy capacity.*

`execution_aware_alpha` and `execution_aware_low_turnover` optimise the net-of-everything number, not
the predicted return. On this venue that mostly means **trading less**: the round trip is larger than
most effects, so the strategy that trades least loses least. `execution_aware_low_turnover` won on
four of twelve symbols, tied for most of any variant.

The execution model behind the numbers is real: a queue-position fill model ported from hftbacktest,
a passive-execution model that prices what a maker fee actually costs in *missed fills*, and a
measured latency series — median round trip to Bitget of 308 milliseconds, 99th percentile 353,
with a clock offset of −459 milliseconds that the order path corrects for. The price you decided on
is not the price you get, and the engine knows by how much.

## Part 19 — The Track 1 result, in full and without softening

`argus/research/track1_study.py` ran every strategy on all twelve stock perpetuals over 180 days of hourly
data — 4,319 bars per symbol, 12bps round trip, chronological out-of-sample split, and every winner
re-scored at five different fee levels. The file is `argus/data/track1_study.json`.

The window is deliberate. A competing Season-2 entry reports its headline over 150 days and treats
the length as the reason to believe it; the venue has 199 days of hourly history available, so this
study uses 180 and says so.

**The flattering number first: 12 of 12 symbols have a strategy that beats buy-and-hold on net
Sharpe.** Best net Sharpes range from 0.85 to 2.33, with Sortinos as high as 3.61. If this document
stopped here it would read like a winning entry.

**Now the number that matters: 0 of 12 survive the Deflated Sharpe gate when every trial is
counted.** Twenty-five strategies are tried per symbol. Picking the best of twenty-five on 180 days
of data and reporting its Sharpe is reporting the maximum of a noise distribution. Corrected for
that search, not one of the twelve winners is distinguishable from the best of a random search.

Counting only the winning candidates and pretending the losers were never tried — which is what
every backtest that omits its trial count is doing — **2 of 12 survive that softer gate**: TQQQ on
`weekend_only`. The gap between one and zero is the entire value of publishing the trial count.
Under the soft gate this looks like a discovery; under the honest one it is the best of twenty-five
tries.

**The cost sweep is the test a rival entry is right to use, so we use it harder.** Every winner is
re-scored at 6, 12, 20, 30 and 40 basis points round trip — our own fee is 12 — and a strategy
passes only if it stays positive at **every** level, not on average. **Eight of twelve survive to
40bps.** TQQQ's `weekend_only` still returns 1.32 there and AAPL's `low_turnover_carry` 1.20; MSFT's
`execution_aware_alpha` looks strongest of all at 6bps (2.58) and is **negative by 30** — which is
exactly the fee-fragility a single-fee backtest cannot see. The entry that introduced us to this
test applies no deflation at all; through our gate their headline scores 0.26 against a 0.95 bar.

**And the sixth criterion catches what the first five hide.** MSTRUSDT's rolling 30-day Sharpe is
scorable in only **1,107 of 2,129 windows** — the strategy was flat for the other half of the
period, so its headline is earned in a fraction of the sample. COIN is worse still: across the
twelve symbols the thinnest coverage is scorable in only 308 windows, well under half the period.
No Sharpe, Sortino or drawdown figure reveals that.

**Out-of-sample decay and fee fragility, per symbol** (25 variants, 4,319 hourly bars each, 12bps round trip, chronological split, plus the five-level cost sweep):

| Symbol | Best strategy | IS | OOS | Retention | Alert | Survives 40bps |
|---|---|---|---|---|---|---|
| NVDA | weekend_only | 2.49 | 0.26 | 0.10 | **breach** | yes |
| TSLA | rotation_regime_switch | 1.89 | -0.48 | -0.25 | **breach** | **no** |
| AAPL | low_turnover_carry | 2.24 | 0.82 | 0.36 | **breach** | yes |
| MSFT | execution_aware_alpha | 0.34 | 3.60 | 10.52 | ok | **no** |
| META | fitted_trend | 1.10 | 3.02 | 2.74 | ok | yes |
| GOOGL | execution_aware_low_turnover | 2.43 | -0.98 | -0.40 | **breach** | yes |
| AMZN | execution_aware_low_turnover | 0.08 | 2.49 | 32.46 | ok | yes |
| COIN | volume_confirmed_trend | -0.60 | 3.40 | -5.63 | **breach** | **no** |
| MSTR | volume_confirmed_trend | 0.04 | 1.95 | 44.13 | ok | **no** |
| QQQ | execution_aware_low_turnover | 2.54 | 0.46 | 0.18 | **breach** | yes |
| TQQQ | weekend_only | 2.07 | 2.85 | 1.37 | ok | yes |
| SQQQ | fitted_trend | 1.08 | 1.13 | 1.05 | ok | yes |

**8 of twelve breach the handbook's own alert, and the alert fires where it should.** The three
highest in-sample Sharpes in the table — QQQ at 2.54, NVDA at 2.49 and GOOGL at 2.43 — produce
out-of-sample readings of 0.46, 0.26 and −0.98. That is the exact shape of a fitted result, and the
criterion that catches it is the one a leaderboard-optimised entry would be least motivated to
compute. COIN is the opposite case and no better: an in-sample Sharpe of −0.60 that became +3.40 is
a coin that landed heads, and the retention ratio of −5.63 is flagged rather than celebrated. MSTR
and AMZN carry retention ratios of 44 and 32, which are not triumphs — they are in-sample Sharpes
of 0.04 and 0.08 dividing into something larger, and a ratio built on a near-zero denominator is
arithmetic, not evidence.

**The two tests disagree, and the disagreement is the finding.** Six symbols breach the
out-of-sample alert; four fail the cost sweep; and the two sets overlap on only two names. MSFT
survives the decay test with a retention of 10.5 and dies at 30bps. GOOGL breaches the decay alert
and survives to 40bps. A strategy has to pass both, and only four of twelve do — none of which
clears the deflated Sharpe over all trials anyway.

**What this means for the track.** Track 1 is scored purely on these numbers. Ours say: the
machinery is complete, the window exceeds both the handbook's requirement and the leading rival
entry's, every judged metric is computed, and **the honest reading is that one candidate survives
the soft gate, eight survive the fee sweep, four survive both the fee sweep and the decay alert,
and none survives the deflated Sharpe over all trials.** On a venue where the fee is 12 basis
points and the measured intraday edge is about zero, we believe that is close to the true state of
the world.

That is why Track 1 is parked. Submitting it would mean either reporting the 9-of-12 headline and
hiding the 0-of-12, or reporting both and entering a numbers-only track with numbers that say no.
Reading the field settled it: the leading Track 1 entry publishes an out-of-sample Sharpe of 1.55
that applies no multiple-testing correction across 59 recorded trials, scores 0.26 through our own
gate, and does not reproduce from its own code. We could publish a comparable number by removing
one gate. We would rather put the machinery in front of judges as the verification layer under
Tracks 2 and 3, which is what it is.

**Two honest gaps in the Track 1 artefact itself.** Turnover and rolling 30-day stability are
computed by the engine for every run, but the study file reports neither per symbol. If Track 1 were
entered, that would need adding before the file was submitted.

### The cross-sectional study, and the result it destroyed

Every strategy above asks one question per instrument: should I be long this. A cross-sectional
factor asks a different one: which of these twelve should I be long, and which short, right now.
That distinction matters on this venue specifically, because the twelve stock perpetuals share one dominant
driver. A long-only signal is mostly a bet on the US market; a rank-neutral book cancels that and
bets only on the dispersion between the names.

Until this week the grammar could not express the question at all. A port of all 101 Formulaic
Alphas against it found **46 of them blocked on cross-sectional rank alone**: `rank(x)` in that
vocabulary means "where does this name sit among all names at this instant", and a time-series rank
within one instrument's own history is a different question wearing the same word. Two new
operators and a panel evaluator closed that gap. On live bars the alignment is exact — twelve
symbols, 719 shared hourly timestamps, none dropped — and a 24-hour momentum rank at the last bar
went long the inverse index token and short the triple-levered one, a trade with almost no market
exposure by construction.

**Then it was measured, and the measurement was worse than a null result — it was a retraction.**

Eight factors were run across ten trading rules. The first version of the study ran each rule once
and reported its best row: a 72-hour cross-sectional reversal at **+2.28 net Sharpe**, after real
costs, the first net-positive cross-sectional result this project had produced.

That number was not real. A 72-hour rebalance has seventy-two possible starting hours, and the
study had silently picked one. Running the identical rule on the identical data at every phase of
its own cycle gives a **worst phase of −3.84 and a best of +4.36**, with a mean of **+0.14**. The
headline had been the top of an eight-Sharpe-wide noise distribution, selected by an alignment
nobody would have thought to report.

**The Deflated Sharpe gate cannot catch that**, and this is the part worth keeping. Phases are not
separate searches, so the trial count never rises to account for them, and the gate never sees the
selection. The only defence is to stop picking one. Every rule is now run at every phase and scored
on the mean, with the spread published beside it.

The corrected study, on ninety days of live hourly bars:

| Quantity | Value |
|---|---|
| Trials (factors × trading rules) | 80 |
| Backtests actually run (every phase) | 1,336 |
| Rules positive **before** cost | 31 |
| Rules positive **after** cost | 4 |
| Rules surviving either Deflated Sharpe gate | **0** |
| Rules whose phases agree on sign | **0** |
| Widest phase spread | **8.7 Sharpe units** |

The gap between 31 and 4 is the cost of trading a twelve-name book on a venue charging six basis
points a side. The gap between 4 and 0 is multiple testing. The zero in the last row is the one
that settles it: not a single rule in the grid produces a result that survives being asked what
hour it started on.

This is what the verification layer is for. It was pointed at our own best number and it removed
it.

---
---

# PART D — One product: the complete architecture

Everything above has been described track by track, because that is how judges read. But ARGUS is
not three products. It is one system, and the tracks are three doors into it. This part is the
complete map — every package, every module, the rules that hold the shape, the path a real decision
takes, and every artefact the system writes.

**Measured, not remembered.** As of the last build:

| | |
|---|---|
| Source modules | **327** files across **21 packages**, 130,470 lines |
| Registered and importable | **152/152** (`python -m argus.status` checks this at runtime) |
| Test files / tests | **224 files**, **6,700 tests collected** |
| Type and lint | `ruff` clean, `mypy --strict` clean on **327 source files** |
| Artefacts written | **162** files under `argus/data/` |
| Code-level teardowns of other people's systems | **56** under `research/architecture/` |
| Runtime dependencies | **two**: `pydantic`, `python-dateutil`. No numpy, no pandas, no scipy |

That last row is the constraint that shaped everything else. Every statistic in this system — OLS,
the augmented Dickey–Fuller test, hierarchical clustering, matrix profiles, Fisher intervals,
binomial tests — is implemented in pure Python, and the ones with a reference implementation are
checked against it numerically rather than trusted.

---

## Part 20 — The seventeen packages, every module named

### `truth/` — what time is it, and what did we know? (3 modules)

| module | what it holds |
|---|---|
| `clocks` | The **dual clock**. The token's never stops; the anchor stock's has four phases — regular, extended, overnight, weekend — plus holidays. Everything downstream asks this which session it is in and how many hours remain until genuine price discovery. |
| `facts` | Point-in-time fact store. **Every fact carries five times**: when it happened, when it was published, when we ingested it, when it became available to us, and a revision number so a restatement supersedes rather than overwrites. A query without an as-of date is a type error. There is no "latest". |
| `evidence` | The `Evidence` type itself, extracted here so `market/*` can produce evidence without transitively importing the model client. |

### `cost/` — the model with no zero-fee constructor (1 module)

`model` — commission, spread, square-root impact, borrow, and **perpetual funding charged per
8-hour settlement** with the venue's ±0.5% cap enforced at construction. There is deliberately **no
`CostModel.zero()`**: a frictionless model can only be built through `frictionless_for_research()`,
which stamps it so that `assert_gateable()` refuses to let it justify a decision. Cost-blindness was
the single most common defect across the systems we tore down, so it is prevented structurally
rather than by review.

### `decision/` — the vocabulary a decision is written in (2 modules)

`verdicts` — `Intent`, `Verdict`, `ConstitutionVerdict`, and `apply_constraint`, the one function
through which the risk layer may narrow an intent. `escalation` — when a decision must go to a
human instead of an order.

### `risk/` — the layer that may only reduce (5 modules)

| module | what it does |
|---|---|
| `sizing` | Half-Kelly, but **gated on calibration**: confidence is only used if the desk's stated confidences have been graded and pass an expected-calibration-error bar. Otherwise a fixed fraction. |
| `circuit` | Drawdown ladder and session circuit breaker. |
| `hedgeability` | The hedge **menu** and a priced residual, never a single "hedgeable fraction". Ranked by risk-neutralisation efficiency — risk removed per basis point of all-in cost. |
| `effectiveness` | **Ederington hedging effectiveness measured from the venue's own series**, per session phase, with a Fisher confidence bound. This replaced three constants that had been typed into `hedgeability` with no sample behind them. |
| `session_risk` | Per-phase volatility and the **reopen jump**, and the volatility-targeting throttle that follows from them. |

### `market/` — everything that fetches (14 modules)

`bitget` (tickers, funding), `history` (candles, index, premium, basis), `depth` (**live L2 order
book**, sweep cost by size), `markout` (**adverse selection** from the public fills feed),
`microstructure`, `volatility`, `evidence` (the gather pipeline), `skills` (Bitget's own research
Skills), `insider` (Form 4), `fundamentals` (XBRL, point-in-time, restatements resolved),
`estimates` (analyst consensus), `macro` (Treasury curve), `collector`, `validation`.

### `agents/` — the model-facing layer (15 modules)

| module | role |
|---|---|
| `desk` | The orchestrator and the **Constitution** — the deterministic rulebook that narrows the model's intent and may never originate one. |
| `analysts` | The panel: event, flow, earnings, macro reads. |
| `debate`, `adversary`, `conflict` | Structured disagreement, an explicit adversary, and how conflicts resolve. |
| `meta_pm` | Prices **deliberation itself** in basis points, so the desk knows what its own thinking costs before it decides. |
| `selection` | Which analysts to run at all, given the cost of running them. |
| `grounding`, `claims` | Numbers checked against sources; claims checked against the evidence that is supposed to support them. |
| `novelty` | Clusters near-duplicate stories, so six wire copies of one story count as one piece of evidence. |
| `causality`, `recall`, `earnings`, `mandate` | Causal chains, memory across decisions, earnings reads, per-profile mandate injection. |
| `quarantine` | **Untrusted third-party text screened and spotlit before the model reads it.** |

### `execution/` — orders and the realism around them (9 modules)

`orders` (state machine, 15 states, read from Nautilus rather than invented), `bitget_client`,
`guard` (the venue's own published rules, fetched), `preflight`, `queue` (queue-position fills,
ported from hftbacktest), `passive`, `latency` + `latency_probe` (measured, not assumed),
`schedule`.

### `paper/` — the record (7 modules)

`runner` (the live cycle), `ledger` (**hash-chained**, append-only), `chains`, `anchor` (head
anchoring so truncation is detectable), `protocol`, `replay`, `repair`.

### `backtest/`, `sim/`, `strategies/` — the engine (9 modules)

`backtest.engine`, `backtest.metrics`, `backtest.validation` (purged CV, embargo, Benjamini–Hochberg,
Bonferroni), `backtest.dependence`. `sim.book` (multi-level matching, rebuilt from ABIDES),
`sim.market`, `sim.agents`. `strategies.track1_suite`, `strategies.session_alpha`.

### `research/` — the studies and the statistics (14 modules)

`cointegration` (**ADF, Engle–Granger, MacKinnon tables, OU half-life — reproduces statsmodels to
1e-12**), `overfit` + `overfitting_study` (CPCV, deflated Sharpe, PBO), `significance`,
`crosssection`, `eventstudy`, `factor_lab` + `memory` (remembers every hypothesis and refuses to pay
for one twice), `grammar`, `searchoff`, `panel`, `gap_study`, `arbitrage_study`, `track1_study`.

### `desk/` — the research and portfolio tools (13 modules)

`research` (the orchestrator: evidence → expectation gap → analogue → path shape → cointegration →
allocation → beta → portfolio → diversification → stress → cost), `portfolio`, `allocation`
(**hierarchical risk parity, reproduces PyPortfolioOpt to 3.6e-16**), `diversification`,
`analogue` + `shapematch` (two different retrievals of "similar", **null-calibrated**), `regime`
(**FLUSS segmentation**), `stress`, `expectation`, `workbench`, `review`, `rootcause`,
`personalisation`.

### `register/` — claims committed before their outcomes (3 modules)

`claims` — the `Claim` schema, the refuse-at-registration validator, and the hash chain.
`resolve` — the auto-resolver and the scoreboard, structurally unable to answer early.
`open_register` — commits a batch and submits its head to four Bitcoin calendars.

Live: **256 claims across all twelve stock perpetuals** (`data/register.jsonl`), chain intact. The opening anchored head `bc36478291a06bc3` is claim 36 of 156 — what the chain's head was when that proof was taken, so it timestamps the first 36 claims and not the 120 added after. Each scheduled cycle appends and re-anchors; the live head is `41f21e5d743cbb46`. Anchored
2026-09-14. The resolver runs on every scheduled cycle.

### `eval/` — thirty-three ways to be wrong in public (33 modules)

The largest package in the system, and deliberately so. `themeaudit`, `standing`, `leakage`
(**memorisation instrument with a sensitivity control**), `docclaims` (**every number in these
documents checked against its artefact**), `riskproof`, `forecasts` + `forecastbench`,
`architecture` (the import graph checked against its own rules), `bench`, `incremental`, `ablation`
+ `ablations`, `autopsy` (**which risk gate bound, and which were never reached**), `degradation`,
`luibench`, `shadow` (**the desk's directional view graded while it refuses to trade**), `hurdle`,
`consistency`, `decisioncard`, `observatory`, `venue_rules`, `challenge`, `riskaudit`,
`profile_value`, `cyclecheck`, `performance`, `baseline`, `selfaudit`, `episodes`, `sources`,
`scorecard`, `bakeoff`, `collect`.

### `lui/`, `demo/`, `llm/`, `proof/`, `status` (14 modules)

`lui.*` — the language user interface: `question`, `answer`, `router`, `phrasebook`, `server`, `cli`.
`demo.flow` (event → decision → execution in one trace), `demo.cockpit` (the evidence page).
`llm.qwen` (dependency-free client), `llm.provider`, `llm.base`. `proof.autonomy`. `status` — the
runtime self-check.

---

## Part 21 — The seven layers, and the rule that holds them

Each layer may import only from the ones below it. This is not a convention — `eval/architecture.py`
builds the import graph with `ast`, computes transitive reachability, and **fails the build** on a
violation.

```
  7  demo, lui, eval            the human surfaces and the judges of everything below
  6  agents, paper              the model, the Constitution, the record
  5  desk, research, strategies the tools and the studies
  4  execution, sim, backtest   orders, simulation, the engine
  3  market                     everything that fetches
  2  cost, decision, risk       the vocabulary and the limits
  1  truth                      time, and what was knowable
```

Two rules are enforced on top of the ordering:

- **The deterministic core** — `truth`, `cost`, `risk`, `decision`, `backtest` — **may never import
  `argus.llm` or `argus.agents`, at any depth**, including inside a function body. These packages
  must compute, never ask. This rule is the reason `truth/evidence.py` exists: `market/*` was
  transitively loading the model client through `agents/analysts.py`, and the type was moved down.
- **A producer may not import its consumer.** `market` may not import `agents`.

Live: (`data/architecture.json`) **0 layering violations, 0 forbidden imports, 4 cycles (all deferred, none at import time)**.

---

## Part 22 — One decision, end to end

This is the real path, in order, as `paper/runner.py` executes it.

1. **The clock.** `truth.clocks` says which session the anchor is in and how many hours to price
   discovery. Everything downstream is conditioned on this.
2. **The market.** `market.bitget` for price and funding, `market.history` for the basis,
   `market.depth` for what it would actually cost to trade the intended size right now.
3. **The evidence.** `market.evidence` gathers news, filings, insider forms, fundamentals, macro.
4. **The quarantine.** `agents.quarantine` screens every item for injected instructions, redacts
   hostile ones **without dropping them**, and writes a line into the record either way.
5. **Novelty and selection.** `truth.novelty` collapses duplicate stories; `agents.selection`
   decides which analysts are worth running given what deliberation costs.
6. **The panel.** `agents.analysts` runs the reads; `agents.debate` and `agents.adversary` force
   disagreement; `agents.grounding` and `agents.claims` check the numbers and the claims.
7. **The intent.** The model produces an `Intent`: side, size, confidence, thesis, and — mandatory —
   what would falsify it.
8. **The Constitution.** `agents.desk` runs its gates in source order, each able only to narrow:
   no-exposure → minimum confidence → oracle staleness → unhedgeable gap → **session volatility** →
   maximum position. The first that binds returns; the rest are recorded as **UNREACHED**, never as
   "passed".
9. **The cost.** `cost.model` charges commission, the **measured** executable spread for that size
   from the live book, impact, and funding per settlement.
10. **The record.** `paper.ledger` writes a hash-chained entry binding the order to the exact intent
    hash that authorised it, with the session phase, the hours to discovery, and the desk's
    directional **lean** — recorded even when it refuses to trade.
11. **The check.** `eval.cyclecheck` runs nine checks over the cycle while the evidence is fresh,
    and fires a named falsifier if the desk proposed nothing during a session that had price
    discovery.

---

## Part 23 — What the system writes, and who reads it

56 artefacts. The ones that matter most:

| artefact | written by | read by |
|---|---|---|
| `paper_ledger.jsonl` | `paper.ledger` | `eval.shadow`, `eval.observatory`, `eval.autopsy`, `eval.cyclecheck`, `demo.cockpit` |
| `depth.json` | `market.depth` | the cycle, and every fill's spread charge |
| `hedge_effectiveness.json` | `risk.effectiveness` | `paper.runner` → the hedge surface |
| `session_risk.json` | `risk.session_risk` | the Constitution's session gate |
| `leakage.json` | `eval.leakage` | `eval.shadow` via `check_window` |
| `markout.json` | `market.markout` | the cost model's maker-side decision (advisory) |
| `cointegration.json`, `allocation.json`, `regimes.json`, `shape_matches.json` | the desk tools | `desk.research`, the cockpit |
| `theme_audit.json` | `eval.themeaudit` | the submission surfaces |
| `doc_claims.json` | `eval.docclaims` | **this document, and the README** |

That last row is the one that keeps the rest honest: every number quoted in these documents is
checked against the artefact that produced it, and a drifted number fails the build.

---

## Part 24 — What runs on its own

`run_paper_cycle.ps1`, on a Windows scheduled task, four times a day inside US regular hours
(13:30 / 15:30 / 17:30 / 19:30 UTC). In order:

```
python -m argus.market.markout --symbol MSTRUSDT --seconds 90   # adverse selection, sampled live
python -m argus.market.depth                                     # what crossing each book costs
python -m argus.risk.session_risk --days 90                      # per-phase volatility + reopen jump
python -m argus.risk.effectiveness --days 60                     # measured hedge effectiveness
python -m argus.paper.runner --once --symbols <all twelve>       # the decision cycle itself
python -m argus.eval.cyclecheck                                  # nine checks, while evidence is fresh
```

The four measurements run **before** the cycle deliberately: each writes a file the cycle then reads,
and each is treated as **absent rather than stale** past 36 hours — so a measurement that stops
refreshing degrades the system into saying "not measured" rather than into using an old number.

Also standing on its own: the factor lab's memory across runs, the Skill health sweep the cycle reads
rather than waits on, and `python -m argus.status`, which checks at runtime that all 152 modules
import, that all 18 sub-themes resolve to a symbol and a test file, and that every declared artefact
is on disk.

## Part 25 — The numbers, all in one place

Every figure below is produced by running something, and `python -m argus.eval.docclaims` checks
each one against the artefact that produces it. A number in this document that has drifted from its
source is a build failure, not a typo.

| | |
|---|---|
| Source modules | 327 files, 21 packages, 130,470 lines; `mypy --strict` clean on 473 source files |
| Runtime dependencies | **two** — pydantic, python-dateutil. No numpy, pandas or scipy |
| Modules registered and importable | 152/152 modules importable |
| Tests | 6,700 tests collected, `ruff` clean |
| Sub-themes resolving at runtime | 18/18 sub-themes |
| Artefacts on disk | 66, every one produced by running something |
| Code-level teardowns of other systems | 62, each citing file and line |
| Competitor entries read at source | 5, each cloned and verified or refuted |
| Live data sources reaching a decision | 15 |
| Paper ledger | every decision a refusal, chain intact, head anchored (live count on the console's `/status`) |
| Scored policy forecasts | 49,142 raw / **32,196 effective** over 4,150 distinct instants |
| Forecast calibration | Brier 0.2319, ECE 0.0109, reliability 0.0002 against resolution 0.0166 |
| Forecast record on ForecastBench's scale | **Brier Index 51.8**, where 50 is a coin |
| Forecast skill against climatology | **+6.4%** — real, and small |
| Murphy identity residual | 1.5e-14, so the decomposition reconstructs its own input |
| Track 1 probability of backtest overfitting | 0.11 to **0.77** across twelve symbols |
| Track 1 trials individually significant at 5% FDR | **0 of 300**, under both BH and Bonferroni |
| Believable track record these Sharpes would need | 0.47 to **3.52 years** of live hourly trading |
| Track 1 study | 180 days, 4,319 bars per symbol, 25 variants, 12 symbols |
| Track 1 surviving the cost sweep to 40bps | 8 of 12 |
| Track 1 surviving the deflated Sharpe on all trials | **0 of 12** |
| Cross-sectional study | 80 trials, 1,336 backtests, every phase of every rebalance cycle |
| Cross-sectional surviving either DSR gate | **0 of 80** |
| Protocol commitments, hash-chained | 2 versions, both **anchored to Bitcoin via four independent calendars** |
| Tradeable-session replay frames, point-in-time | 30 reconstructed, **0 positions opened** |
| Hurdle frontier | 39 instants / 25 effective, median move 137bps vs an 18.8bps hurdle |
| Directional accuracy at which trading beats abstaining | **56.0%** |
| Capabilities proven better than a named competitor | **20 of 38** |

That last row is the one to read twice. An *owned* capability needs a reproduced baseline, a
same-input comparison, an out-of-sample test, an ablation and an adversarial test — thirteen
conditions in all, enforced in code by `argus/eval/standing.py`, which raises at import if anything
claims OWNED without them. It read zero when this table was first written; today 20 of 38
capabilities are OWNED, 11 are TIED, 24 are IMPLEMENTED and none is LOST. Five ties were losses first:
the three from before 2026-09-22, order splitting against Bitget's own TWAP, and the rToken
overnight hedge, lost to the S2 entry Ballast and
rebuilt to parity on 2026-09-24. It read 27 of 31 until 2026-09-24, when a review of the right rivals per sub-theme withdrew
seven OWNED grades won against a rival that does not lead the sub-theme, and split two more into
the narrow thing proven (OWNED) and the sub-theme itself (IMPLEMENTED).

## Part 26 — What we checked the competition and found

Five other public Season-2 entries were read line by line before this design was settled. What
they taught is recorded here as classes of problem, not as findings about named teams — a
competitor's repository is not ours to grade in public.

- **Published figures that the repository's own files do not reproduce** — a Sharpe that differs
  from the equity file beside it, results that change on re-running the author's own code.
- **No multiple-testing correction** against dozens of recorded trials, so a best-of-many backtest
  is reported as though it were the only one.
- **Samples counted by rows, not by independent events** — thousands of forecasts over a hundred or
  so distinct nights.
- **Records that can be rewritten** — a single hash at settlement that excludes the thesis, or a
  timestamp that can be back-dated with one command.
- **Decisions that can never settle**, because no exit exists in the code.

One entry was ahead of ours on explaining each decision, and that is why the external anchor below
exists.

Two real gaps came out of that, and both are closed:

1. **Nobody in the field had a commitment they could not rewrite — including us.** The protocol
   commitment is now submitted to four independent OpenTimestamps calendars, Bitcoin-backed, and
   verifiable by anyone with `ots verify` and no access to our code.
2. **Nobody reported an effective sample size.** Every scored record here now publishes distinct
   instants, intra-cluster correlation, Kish design effect and effective count beside the raw count.

## Part 27 — Everything, ranked from strongest to weakest

This is the honest inventory. Strongest means: hardest for a competitor to match, and most likely
to survive a judge checking it. Weakest means the opposite, and the last six are things a judge
could attack today.

### Strongest — verified, and rare or unique in the field

1. **A pre-commitment anchored outside our own control.** Two protocol versions, hash-chained,
   submitted to four independent Bitcoin calendars. Nobody else in the field has one; the entry
   that claimed to have one was back-dated six years by its auditor. `paper/protocol.py`,
   `paper/anchor.py`.
2. **Deflated Sharpe that raises instead of returning NaN.** vectorbt returns NaN silently and a
   NaN comparison admits everything. qlib computes none of the six Track-1 metrics at all. Our gate
   is applied to every trial, not just the survivors, and it reports 0 of 12.
   `backtest/metrics.py`.
3. **A risk layer that may only reduce, proved by exhaustive sweep.** 2,177,280 states, zero invariant
   violations, three mutant policies the prover must catch. Not asserted — swept.
   `decision/verdicts.py`, `eval/riskproof.py`.
4. **Effective sample size on every scored record.** Intra-cluster correlation, design effect,
   distinct instants, effective count. The competing entry with the largest advertised record does
   not do this, and its intervals are about 1.9 times too narrow as a result. `eval/forecasts.py`.
5. **49,526 point-in-time scored forecasts, restated in the metrics the public benchmarks use.**
   Brier 0.2319 against a 0.25 coin, which is a **Brier Index of 51.8** on ForecastBench's scale
   where 50 is a coin. Skill is reported against three references and not the flattering one:
   +7.2% against a coin, **+6.8% against climatology**, +33.3% against persistence. Expected
   calibration error 0.0109, with reliability 0.0002 against resolution 0.0166 — the forecasts
   discriminate by more than they are miscalibrated, which is the decomposition that matters and
   the one a single Brier number hides.

   **What the decomposition says, and it is not a boast.** Reliability 0.00029 against resolution
   0.01562 on an uncertainty of 0.24866. In plain terms: these forecasts are *extremely well
   calibrated* and *weakly discriminating* — they resolve about 6% of the uncertainty available in
   the question. The skill is real, statistically and by three references, and it is small. That
   pattern is also consistent with the desk's abstentions: a forecaster this calibrated and this
   undiscriminating is one that correctly declines to bet hard. `eval/forecastbench.py`.
6. **Deliberation priced in basis points and charged to the hurdle.** Nothing in the 88-repo corpus
   prices reasoning time. The debate built this week inherits it, so an argument the desk cannot
   afford is one it does not have. `agents/meta_pm.py`, `agents/debate.py`.
7. **A factor grammar with no execution surface.** RD-Agent runs model-written Python through a
   shell subprocess. Ours emits a typed tree that is interpreted; there is nothing to sandbox.
   `research/grammar.py`.
8. **Abstention scored as a decision.** Every pass records the move it declined, so standing aside
   correctly and being paralysed are distinguishable. Nothing in the corpus does this.
9. **Point-in-time gating everywhere, including a look-ahead gate on analogue retrieval**, with the
   cross-sectional evaluator tested by truncating the future and asserting the past does not move.
10. **A documentation checker that repairs what it finds.** Every numeric claim in these
    documents is checked against the artefact that produces it, and the checker rewrites the stale
    ones. It caught its own half-rewrite bug on its first run, and it caught the six counts in this
    very section going stale the moment five modules were added.
11. **Murphy's Brier decomposition with the term almost every implementation drops.** The textbook
    identity `BS = REL - RES + UNC` is exact only when a forecaster issues finitely many distinct
    probabilities. Bin a continuum into the conventional ten buckets — which is what every
    implementation of expected calibration error does — and it stops closing. Ours carries the
    within-bin term, so the identity reconstructs the Brier score to floating-point zero on any
    input, at any bin count. This was not foresight: the three-term version was written first and
    the self-test refuted it at -0.0044, then -0.0186 at two bins. `eval/forecastbench.py`.
12. **Inference that survives autocorrelation, in three forms.** Lo's (2002) HAC standard error
    with the Newey-West Bartlett kernel, the Politis-Romano stationary bootstrap, and White's
    Reality Check against a named benchmark. Verified against processes with known answers: an
    AR(1) with parameter 0.6 recovers autocorrelations 0.600, 0.357, 0.214, 0.129 against a true
    0.600, 0.360, 0.216, 0.130, and a field of twenty strategies differing from their benchmark by
    noise alone fails the reality check at p = 0.44 while one with a planted edge is found at
    p = 0.000. `backtest/dependence.py`.
13. **A real event study, not an asserted causal chain.** The transmission chains were already
    graded link by link; the *effect* was never estimated. Now it is: market model on a purged
    estimation window, abnormal and cumulative abnormal returns, and four significance tests rather
    than the one a plain t-test would give — Patell with its full three-term prediction-error
    correction, Boehmer-Musumeci-Poulsen for the event-induced variance Patell cannot see, Corrado's
    rank test for the fat tails a parametric test is driven by, and the generalised sign test with
    the sample's own positive rate rather than an assumed half. Every parametric statistic carries
    the Kolari-Pynnonen deflation for cross-sectional correlation, which on twelve tokenised
    equities driven by one market is not a rounding term. Verified by planting a 250bps shock and
    recovering +271bps with a beta of 1.214 against a true 1.2, and by a null under which no test
    rejects. The one local implementation we tore down has the market model right and none of the
    four tests. `research/eventstudy.py`.
14. **The overfitting layer the deflated Sharpe does not cover.** Probability of backtest
    overfitting by exhaustive combinatorially symmetric cross-validation — every balanced split,
    not a sample — purged and embargoed cross-validation, minimum track record length, and both
    Benjamini-Hochberg and Bonferroni reported side by side. It also reports its own precision: a
    single PBO estimate has a standard deviation of 0.25 across draws from the same null, measured
    over thirty realisations, so the result refuses to be read more finely than that.
    `backtest/validation.py`.

15. **The handbook's complete research task, running every step on real data.** Track 3 requires
    "one complete research task, from question to actionable insight". Ours answers *should I add
    NVDA to this book?* at **7/7 coverage**: evidence, expectation gap, historical analogue,
    session beta, portfolio impact, stress and execution cost, each naming the module that computed
    it and none summarised by a model. Two of those seven were reporting absences for reasons that
    described work nobody had done — the analogue search was never run, and the consensus lookup
    was declared unavailable when `market/estimates.py` had fetched it keylessly all along. Both
    were found by running the deliverable rather than reading its output. Live (`data/research_report.json`): 42 analysts expect
    EPS 2.4727 for the quarter ending 2026-10-31, +90.2% against the 1.30 delivered a year earlier,
    and the expected growth is decelerating against the growth recently delivered.

16. **Every data source probed now, not listed from memory.** Track 3 scores data sources on
    count *and* effectiveness, and only the first is an inventory. `eval/sources.py` calls all
    twelve through the module the desk uses — not a raw URL, because an endpoint that answers
    while our parser raises is not a working source — and reports four honest states with a
    timestamp. **12 of 12 answered with usable content**: 12 stock-perpetual tickers, 47 market and 47 index
    candles, 6 SEC filings, 175 Treasury curve points, Fear & Greed, VIX, 12,325 FINRA short-volume
    rows, 23 Nasdaq halts, 4 Yahoo consensus estimates, 67 XBRL facts and the Skill server's last
    probe. EMPTY is a distinct state from OK and does not count toward the live total, because a
    feed returning 200 with nothing in it is the way an integration dies quietly.

### Strong — built, tested, live, but matched elsewhere

17. **Multi-round bull/bear debate**, bounded by rounds, by budget and by measured convergence,
    which reports agreement as self-consistency when both seats share a model.
18. **Episodic memory computed from the ledger**, not a model's recollection — every lesson is
    arithmetic anyone can recompute, and it refuses to state a pattern below five graded episodes.
19. **Fifteen live data sources**, including the full 33-item 8-K materiality taxonomy, FINRA short
    volume, Nasdaq halts with the published reason codes, CBOE VIX with a 35-year percentile, and
    the US Treasury curve.
20. **A language console that answers without a model key, in English and Chinese**, every answer
    citing the ledger rows it was built from, deployed publicly and responding in about 1.1 seconds.
21. **Cross-sectional factor evaluation** aligned on shared timestamps, with every rebalance rule
    run at every phase of its cycle.
22. **An ablation harness** that refuses to state a direction below 30 differing pairs, and an exact
    replay path for deterministic components.
23. **A capability register that cannot claim OWNED** without all thirteen conditions, checked
    against the tree at import.

### Honest negatives — published rather than hidden

24. **Personalisation diverges on every instant and barely helps.** Measured over 4,528 real
   instants, the two shipped mandates disagree on **100%** of them — but the conservative one
   *resizes* 98% and *refuses* only 2%, and its refusals are worth **+0.35bps per decision**. It
   loses less than the aggressive mandate (-7.6bps against -15.4bps) almost entirely because it
   halves every position, which is sizing and not selection; the utility figure measures each
   mandate against taking everything *at its own permitted size* so the two never add together.
   Divergence is demonstrated, usefulness is marginal, and the first version of this measurement
   scored a resize as a full take and would have reported the halving as judgement.
   `eval/profile_value.py`.

25. **0 of 12 Track 1 winners survive the deflated Sharpe on all trials.** 8 of 12 survive the cost
    sweep to 40bps, which is the softer and less meaningful test.
26. **0 of 300 (symbol, variant) trials are individually significant at a 5% false-discovery
    rate** — under Benjamini-Hochberg, which is the permissive test, and under Bonferroni, which
    is the strict one. Twenty-five variants across twelve symbols, every p-value computed from the
    sample's own skew and kurtosis rather than assumed normal. Not one survives.
27. **No simple rule is profitable on this venue, measured over 4,528 instants.** Five fixed
    rules — flat, always-long, momentum, reversion and volatility-gated — replayed at every
    tradeable instant across all twelve stock perpetuals, each scored on the realised move net of the same
    hurdle the desk faces. **Every one loses money.** The best directional accuracy is **50.6%
    against a break-even of 54.7%**, and no interval excludes zero once the overlapping
    twenty-four-hour horizons are accounted for by a stationary bootstrap. The paired comparison
    against the desk rests on 25 effective instants and cannot settle a small difference; this
    does not fabricate desk decisions to grow it, it answers the neighbouring question on a sample
    large enough to refute. A desk declining to trade this universe with these rules is declining
    something that does not pay. `eval/venue_rules.py`.
28. **The mechanism behind the worst overfitting score, diagnosed rather than named.** TQQQ's
    `weekend_only` carries the study's best headline Sharpe of 2.36 and its worst probability of
    backtest overfitting at 0.77. The reason is now measured: across the 70 balanced splits, that
    variant is the in-sample winner **only 30% of the time**. Seven splits in ten, a different
    variant looks best. Across the twelve symbols the correlation between overfitting probability
    and how often one variant dominates the splits is **−0.52** — where the selection is stable the
    procedure survives, and where the winner changes constantly it does not. The headline is not a
    strategy that was chosen; it is the one that happened to win the split we ran.
29. **The strategy is stable; the *choice* of it is not, and the two measurements disagree in an
    informative way.** TQQQ on `weekend_only` shows **no out-of-sample degradation** across a
    chronological split — Sharpe 2.24 to 2.68, drawdown narrowing from -6.45% to -4.72%, and
    calibration improving from 0.0159 to 0.0134. The same strategy has a probability of backtest
    overfitting of **0.77**. Both are true and neither cancels the other: the rule behaves
    consistently over time, and the *procedure that selected it from twenty-five candidates* does
    not reproduce. A system reporting only the first number would look like it had found something.
    `eval/degradation.py`, `research/overfitting_study.py`.
30. **The one Track 1 result that survived the softer gate is the one the overfitting test
    condemns hardest.** TQQQ on `weekend_only` was the single strategy to clear the
    candidates-only deflated Sharpe, and its probability of backtest overfitting is **0.77** — the
    worst of the twelve, and worse than choosing at random. NVDA scores 0.07 and QQQ 0.09 at the
    other end. The spread of 0.70 is around three times the 0.25 standard deviation a single PBO
    estimate carries, so the ordering is real even though no individual value should be read
    finely. Our best-looking number is the one our own machinery trusts least, and that is
    published here rather than in a footnote. `research/overfitting_study.py`.
31. **A believable track record for these Sharpes would take between six months and three and a
    half years of live hourly trading.** Minimum track record length, per symbol: 0.49 years for
    TQQQ at the optimistic end, 3.63 for AMZN at the pessimistic. This is the number that replaces
    the apology for having no settled trades with a measurement of how long one would have to
    run.
32. **0 of 80 cross-sectional rules survive at any rebalance phase.** The phase sweep destroyed this
    project's own best result: plus 2.28 at one phase, minus 3.84 to plus 4.36 across all of them.
33. **Sentiment is demoted.** Its equity feed is dead; the one live reading is a crypto-wide index
    carried at credibility 0.35 and labelled as such.
34. **The self-evolving review checklist is empty.** Replayed over 40 real decisions, none of the
    five standing rules earned a place.
35. **The as-of evidence gate is inert on live data.** It dropped nothing across 11 live frames, so
    its protection is real in code and untested in production.

### Weakest — where a judge could attack today

36. **No paper Sharpe, max drawdown or win rate — and the pre-registered excuse for it has now
    been refuted by our own harness.** Every decision on the ledger is a refusal.
    This is the single biggest exposure, because Track 2 is half quantitative.

    It was **pre-registered**: protocol v1 permits trading only in regular and extended hours,
    every live decision was taken in a weekend or overnight session, and the committed hypothesis
    said that if the desk abstained through a full regular-hours week, the deliberation hurdle
    would be too high to trade this universe at all. Rather than wait for the calendar, the replay
    harness reconstructed point-in-time frames at instants inside the sessions the protocol
    permits and re-ran the desk against them. **Thirty frames — ten in regular hours, twenty in
    extended — and not one position opened.** Twenty produced a `no_trade`, meaning the desk had
    evidence and declined; ten produced `data_insufficient` on a frame carrying a single evidence
    item, which tests the reconstruction rather than the hurdle and is counted separately for that
    reason. The session phase was not the reason for the abstentions, so the hypothesis as written
    is refuted rather than confirmed — with the standing caveat that a replayed frame carries
    fewer live sources than a real one and so biases the desk toward abstaining. That weakens the
    finding; it does not reverse it.

    So the follow-up question — *was abstaining right?* — was measured rather than argued, and the
    answer does not flatter us. Across 511 instants with a recorded move (2026-09-22, up from 39 at
    first measurement — 481 live plus 30 replayed), the median absolute 24-hour move is 103bps
    against a total hurdle of 18.8bps; 90% of instants clear it, and trading would beat abstaining at a
    directional accuracy of 55.0%. The finding held at the first, much smaller measurement and
    strengthens rather than weakens at 13x the sample. **The binding constraint is the desk's own
    confidence gate, not the cost of deliberation.** Any explanation that blames the fee is wrong,
    and `eval/hurdle.py` is written so it would have said the opposite had the moves been small. The
    remaining honest gap is that the desk has never committed to a directional view it could be
    graded on, so its measured accuracy cannot yet be compared against the 55% bar.
37. **Two settled outcomes.** Calibration on the desk's own judgement cannot be computed yet; the
    49,140 figure is the policy layer, labelled as such, and the two must never be added together.
38. **Four capabilities are not OWNED.** Three are TIED against the named rival and one stops at
    twelve of thirteen conditions; each says which condition it is missing.
39. **Four of five official Bitget Skills carry no data.** Measured to be their backend rather than
    our integration — but a judge sees a thin panel either way.
40. **No live fills.** Execution realism is argued from the venue's published rules, not measured
    against our own fills, because there are none.
41. **The factor grammar is still narrower than Qlib's.** Roughly thirty of the 101 Formulaic Alphas
    need an opening price this grammar does not carry.

## Part 28 — The strengthening pass: ten capabilities, and what each one found

Ten capabilities were built in one continuous pass, each to the same standard: read the best
existing implementation first, build ours, test it, wire it into the live path, and **prove it by
running it**. What follows is what each one measured — including the several that measured *nothing*,
which are reported here at the same length as the ones that found something.

### 1. Path-shape retrieval, calibrated against its own noise — `desk/shapematch.py`

Track 3 asks how the AI retrieves historically similar scenarios. Every implementation in the field
stops at a distance threshold, which is why every one of them can always show you five convincing
analogues: scanning ~1,400 candidate windows for a minimum produces a small number whether or not
the series repeats at all.

So this one reruns the identical scan over 50 paths built by **reordering the series' own returns** —
same bars, same volatility, ordering destroyed — and reports the share that match the present as
closely. **Live on NVDAUSDT: five non-overlapping matches at distance 0.371–0.437, and the verdict is
`no precedent`**, because reordered noise beats 0.371 in 36% of scans. Across 6 symbols × 4 window
lengths, **22 of 24 cells fail the null**.

Reading `stumpy/core.py:1118` (`D² = 2m(1−ρ)`) also found a defect in our own code: the weak-match
threshold had been set to 2.0, which is the **maximum value the metric can return** — the gate could
never fire. It is now 1.0, with a test pinning it below the uncorrelated level of √2.

### 2. Cointegration and mean reversion — `research/cointegration.py`

Half of Track 1's named *Arbitrage* sub-theme was uncovered: we measured the perpetual-versus-underlying
basis and never asked whether two stock perpetuals share a stochastic trend. ADF, Engle–Granger, the MacKinnon
tables and the Ornstein–Uhlenbeck half-life are now implemented in pure Python and **reproduce
statsmodels 0.14.6 to 1e-12** on ten unit-root tests and three cointegration tests over real hourly
closes — including its lag selection, its sample sizes, and its unexplained `nobs − 1` Stata quirk.

What statsmodels does not do is the rest: the hedge ratio is fitted on a training slice and **frozen**
before the out-of-sample spread is tested, z-scores use only prior data, a round trip is charged as
**four** taker legs, and the 66 hypotheses a twelve-instrument scan performs are corrected.

**Live: (`data/cointegration.json`) 66 pairs tested, 4 significant naively against 3.3 expected by chance, 0 surviving
correction, 0 tradeable.** And the binding constraint is the opposite of the basis study's — the
median pair needs only |z| ≥ 0.08 to clear 24bps, so costs are not what fails. Stationarity is.

### 3. The risk layer's constants, replaced by measurements — `risk/effectiveness.py`

The hedge menu priced a candidate as a product of five factors, and three of them were **literals
typed into the source**: `risk_reduction=0.98`, `correlation_confidence=0.98`,
`basis_stability=0.95`. No sample behind any of them, inside a project whose standing rule is *never
guess*.

They now come from the venue's own market-versus-index series: **Ederington (1979) hedging
effectiveness** for the risk removed, the **lower bound of a 95% Fisher interval** for the confidence
— a constant cannot express sampling error because it cannot know *n* — and unit-ratio effectiveness
for the basis, the gap between the last two being exactly the sizing problem.

Measured per session phase and never pooled: **regular hours remove 99.2–99.9% of variance, weekend
97.7–99.8%, pooled 94.9–98.8%** (measured 2026-09-20, 60-day window, n=12 instruments — a rolling
window, so these drift). The old constants were too pessimistic while the anchor traded, and
the pooled figure is an averaging artefact. Every candidate is now stamped MEASURED or ASSUMED and
carries its sample.

### 4. An allocator that prices its own rebalance — `desk/allocation.py`

`desk/portfolio.py` could grade a trade it was handed and never propose one, so the shape of the book
was whatever a sequence of individually-acceptable trades happened to produce.

Hierarchical risk parity now proposes it — chosen because these twelve instruments correlate above
0.9, and a mean-variance solution would invert a near-singular covariance into large offsetting
longs and shorts that are pure estimation noise. Pure Python, and it **reproduces PyPortfolioOpt's
weights to 3.6e-16**, cluster order included.

What PyPortfolioOpt, Riskfolio and skfolio all stop short of is the decision: a weight vector is not
one. `optimize_trade()` prices the turnover, drops legs under 1% of the book, and reports the
**break-even holding period** with its Sharpe assumption named in the same sentence every time.
**Live from equal weight: volatility 22 → 16bps per bar, 42.7% of variance removed, turnover costing
3.2bps, repaying in 43 bars.** The first of three passes where the answer was *yes, trade* — because
the fee that kills every intraday idea is nearly irrelevant to a weekly rebalance.

### 5. What a trade really costs — `market/depth.py`

The paper ledger charged the **quoted touch** on entry and a flat **0.6bps** on every exit. The quote
is the price of an infinitesimal trade; nothing had asked what it costs to take a real size. The
order-book endpoint is public, keyless, 200 levels deep, sits in Bitget's own SDK catalogue — and had
never been called.

**Measured live on 2026-09-20: taking $25,000 costs 2.27bps of QQQUSDT and 12.84bps of SQQQUSDT,
against the half-spread an infinitesimal order would pay of 0.07 and 1.31.** The quote understates
the real cost 32x on one and 10x on the other — a factor that varies between
instruments, which is worse than understating it by a constant, because it **reorders which
instruments look cheap**. Both legs of every paper fill are now priced from the real book, and a size
the visible book cannot absorb returns *no* measurement rather than the cheap half of a trade that
could not be done.

### 6. A session throttle aimed at the measured hour — `risk/session_risk.py`

Every tokenized-equity rival throttles *because the anchor is asleep*. Measured over 90 days, the
shut window is the **calmest** part of the week — a weekend hour moves a fifth of a regular-hours one
— and the risk is the **discontinuity at the end of it**: the first bar after discovery resumes moves
**1.5–2.2× a regular-hours bar on all twelve instruments**, and 5.5–11.8× a weekend bar depending on the name, 64 reopens each.

So the throttle targets the volatility of the path a position will actually live through. **Live on
NVDAUSDT before the open: a 2-hour horizon scales to ×0.82, 8 hours to ×0.99, 24 hours to ×1.00** —
the jump is one bar, so it dominates a short hold and vanishes in a day-long one. At the desk's own
24-hour horizon there is no session-driven reason to size down at all, which is the opposite of what
the field's intuition prescribes.

It is a new Constitution gate, and the arithmetic is **capped at 1**: on a quiet weekend it would
justify sizing *up*, and a risk layer may only reduce.

### 7. The maker-side assumption, finally measured — `market/markout.py`

`cost/model.py` refuses to credit maker treatment at all, because our own replay turned a +90%
simulation into −0.45% once resting orders were modelled honestly. Right call, and a blanket
assumption — nothing had measured how much adverse selection actually costs here.

The public fills feed carries the aggressor side and a millisecond stamp, so the book is polled at
1Hz and joined to the prints. **Live on MSTRUSDT, 150 seconds, weekend: three consecutive runs on 2026-09-20 put the passive
side at a median −0.8 to −2.9bps, then −0.3 to −2.9bps, then +0.3 to +1.6bps — the sign itself is
not stable at this sample size.** That instability is the finding rather than a caveat on it: 150
seconds of one instrument with the anchor shut carries too little informed flow to determine even
the direction of adverse selection. The sign convention is the whole module — a
flipped sign is symmetric, plausible, and says resting orders get picked *up* — so five tests assert
it from both directions.

The module's printed verdict follows whichever sample it drew, and across those three runs it said
*refuse to change the cost model*, then *this deserves a number rather than a rule*, then *no
measurable adverse selection*. **`cost/model.py` is unchanged by all three.** A conclusion that
reverses between consecutive 150-second windows is not evidence about the venue. The maker
assumption moves when a regular-hours sample clears the print minimum at the longer horizons and
holds its sign across runs.

### 8. Regime detection that sees shape, not amplitude — `desk/regime.py`

The incumbent was one threshold: fast volatility under slow volatility means quiet regime. It cannot
see a rally becoming a drawdown at the same volatility.

FLUSS can, and it falls out of machinery we already had — extend the shape distance to every window's
nearest neighbour, and a regime boundary is where few of those arcs cross. Pure Python, 1,079 bars in
10 seconds. Both reference implementations were read and the module follows each where it is better:
matrixprofile's analytic parabola for the idealised curve, stumpy's five-times-wider head and tail
correction.

**Live on NVDAUSDT it separated a +14.0% rally, a −6.4% drawdown and a +2.4% recovery — at 8.6, 10.5
and 9.3bps a bar.** The loudest stretch is 1.2× the quietest, and the report re-runs the incumbent
rule to confirm it never flips at either boundary. It labels nothing and predicts nothing.

### 9. Untrusted evidence quarantined — `agents/quarantine.py`

Every headline, filing and footnote the desk reasons over is written by somebody else, and it was
being interpolated straight into the prompt. Three existing checks all assume that text is *wrong*;
none assumed it is **hostile**.

AgentDojo's four defences were read and taken selectively: spotlighting with delimiters plus the
standing system instruction, and its **replace-don't-drop discipline** — a hostile item keeps its id,
source and timestamp and loses only its claim, so no downstream count silently shrinks. Its 440MB
transformer detector was not taken; ours is six structural patterns that report **the span that
fired**, because a record saying "quarantined" without saying what it saw cannot be audited.

**Calibrated against 49 evidence items fetched live from our own feed: 0 false positives, 0 misses on
9 attacks** — and that corpus caught a real defect, a rule that fired on *"analysts please note the
long-term outlook"*. The corpus is committed, so a future rule that starts redacting real headlines
fails in the suite rather than in production.

### 10. Memorisation leakage, and the control that caught two harness bugs — `eval/leakage.py`

Every historical evaluation here asks a model about a period that already happened, and none could
say whether it had simply **read the answer**.

barj28's instrument was read and its probe copied exactly — multiplicative distractors, deterministic
per-fact shuffle. The facts are ours: 42 probes built from what companies actually filed with the
SEC, bucketed by **filing date**, because a model can only learn a number after it is published.

**Live: (`data/leakage.json`) 32/42 tight probes correct (76%) against a 20% chance level, p < 0.0001, above chance in
every period counted including filings from August 2026.** The model recalls these revenues to within
±18%, and no upper boundary was located.

**The sensitivity control is why that number is trustworthy.** Its first run scored **2/12 with the
answer printed in the prompt** — impossible for any model that can read. The parser was being handed
the completion's repr and scanning the model's *chain of thought* instead of its answer. A second run
scored 0/12: a literal backspace byte had been written where the regex needed `\b`. Both bugs were
invisible in the probe results alone, because **a broken instrument and an honest null look
identical**. Control now 12/12.

And the gate is enforced, not merely published: `check_window()` answers CONTAMINATED / CLEAN /
**UNMEASURED**, and the third never reads as clean — a missing report, or one whose control failed,
certifies nothing. The live shadow record reports CONTAMINATED, with the caveat attached rather than
suppressed: the instrument measures recall of *fundamentals* while the record grades *price
direction*, so overlap is the precondition for leakage, not proof of it.

### What the ten have in common

Six of the ten **found nothing**, and say so: no shape precedent, no cointegrated pair, no
session-driven reason to size down at a 24-hour horizon, no trustworthy markout yet, no change to the
cost model, no clean window for historical evaluation. Two found a real defect in our own code that
had been shipped and passing tests. Two found something actionable — the allocator and the depth
measurement.

That ratio is the point. A strengthening pass in which every new capability discovers an opportunity
is a pass that was not measuring anything.

## Part 29 — What the external audit changed, and what it did not

A long external review of this project — written by other models, covering roughly sixty topics —
was read in full and graded item by item against the code. The audit of that audit is
`research/audit/m1-make-strong.md`. Three things came out of it, and only the first is the kind of
thing such a document usually produces.

**Twenty items were genuinely new; sixteen more were rejected with a reason.** The rejections
matter as much as the additions, because a review this long is mostly a list of things it would be
possible to build. Time-series foundation models were refused on the file's own argument: our
measured intraday edge is about 0.00% against a 12bps round trip, so a better point forecast does
not clear the fee. Persona committees were refused because their personas are system prompts and
ours changes verdicts mechanically. Adopting NautilusTrader as the execution chassis was refused
because we read its state enumeration and built a better machine *from* it, and adopting the chassis
would surrender that correction. Outcome-weighted memory was refused because it is precisely the
contamination `research/memory.py` has an import-time guard against; adopting it would delete our
strongest memory claim.

**Its most valuable correction was that we had no multiple-testing machinery at all.** Confirmed
absent by grep, and now built: probability of backtest overfitting by exhaustive cross-validation,
purged and embargoed splits, minimum track record length, Benjamini-Hochberg and Bonferroni, then
the dependence layer beside it — Lo's HAC standard error, the stationary bootstrap and White's
Reality Check, and the event-study layer below it. Six new modules, 301 new tests. Running them
against our own Track 1 sweep produced
the two hardest findings in this document: nothing survives false-discovery control, and our
best-looking result is the one most likely to be a fitted artefact.

**Four defects in our own code were found by the tests written for that machinery, not by review.**
A constant return series scoring a Sharpe of about 1e15 through floating-point dust, in three
separate places. Duplicate strategy columns forcing the overfitting probability to 1.0 by counting
only strictly-worse rivals rather than taking the midrank — which matters because any two variants
that never trade produce identical flat series. Murphy's Brier decomposition failing to close on
continuous probabilities. And a number in a docstring written the same hour that was already wrong.

### What is left to do

Two items from the previous version of this list are **now done** and have moved into Part 28 and
the evaluation layer: the desk's directional view is recorded on every abstention and graded by
`eval/shadow.py`, and the search-strategy bake-off ran (`data/search_bakeoff.json` — annealing beat
uniform random by +0.0219 per-observation Sharpe on one dataset and one seed, which ranks the
strategies and does not settle them).

What actually remains, in the order it would be done, each with the reason it is not done rather
than an implied promise that it nearly is:

1. **Enforce the leakage gate beyond the shadow record.** `check_window()` is wired into
   `eval/shadow.py` and nothing else. Every `research/*_study.py` and `eval/bakeoff.py` evaluates
   over historical windows the model demonstrably recalls, and none of them says so yet.
2. **Accumulate the markout measurement.** One 150-second sample of one instrument, taken while the
   anchor was shut, is not a basis for changing the cost model — and the cost model will keep
   refusing maker credit until it is. The sampler now runs on every scheduled cycle inside US
   regular hours; it needs time, not code.
3. **Recalibrate `impact_coefficient` against the measured book.** `market/depth.py:impact_check()`
   can now compare the square-root law's prediction with the real sweep cost at any size. The
   comparison is possible; the fit has not been done, and claiming a calibrated coefficient before
   fitting one would be exactly the kind of unearned number this pass removed.
4. **Grade the causal chains as outcomes arrive.** `paper/chains.py` stores every falsifiable link;
   grading them multiplies the scored-claim count from claims the desk already makes.
5. **A utility-under-attack benchmark for the quarantine.** Detection rate and false-positive rate
   are measured; whether the desk still reaches the same decision *while under attack* is not, and
   measuring it means running the full desk twice per attack against a live model.
6. **The free-generation leakage probe.** Recognition and recall are different capacities, and the
   multiple-choice probe measures only the first.
7. **Cite the literature the results already reproduce.** Barclay and Hendershott for the attenuated
   after-hours reference market, among others. The measurements exist; the citations do not.
8. **Run the event study on our own event history.** The methodology is built and verified against
   planted effects; it has not been pointed at our own 8-K and halt record, which needs enough
   events of one kind to clear the five-event floor.
9. **Opening price into the grammar** — unlocks roughly thirty more of the 101 alphas.
10. **ETF-flow layer** from SoSoValue — named highest-ROI by two independent audits.
11. **Per-order records** — placement price, fill, fee, timestamp. Needed the moment orders exist.
12. **Johansen, and a Kalman-filtered hedge ratio.** Both were deliberately not built:
    the Johansen critical-value tables are literal numbers with no response surface, and with no
    evidence of even pairwise cointegration a multivariate test would add a second family of
    hypotheses to a search that has already found nothing. A time-varying hedge ratio is worth
    building only once a pair survives the static test. None did.

---
---

# PART E — The Register

---

## Part 31 — The one thing that cannot be caught up on

Everything else in this document can be out-built. A team with more hours can copy any module here,
and a bigger team can copy all of them — a system has to be published to be judged, which means
publishing it to whoever wants to copy it.

There is one exception, and it is not cleverness. **A record of predictions made before their
outcomes cannot be backfilled, bought, or accelerated.** It is the only asset in this project whose
input is calendar time, and there is no market in calendar time. A competitor who starts the day we
finish is behind by exactly the amount we have run, permanently, and the gap widens every day rather
than closing.

That distinction — *build can be compressed, elapsed time cannot* — is why this part exists and why
it was opened before it was finished.

## Part 32 — What was committed, and what it refuses to do

On **14 September 2026** the register was opened with a first batch of 36 claims, hash-chained
and anchored. That opening head is `bc36478291a06bc3`, submitted to four independent Bitcoin
calendars: `a.pool.opentimestamps.org`, `b.pool.opentimestamps.org`,
`alice.btc.calendar.opentimestamps.org` and `finney.calendar.eternitywall.com`. The `.ots` proofs
are in `argus/data/anchors/`, and anyone can verify them with the reference OpenTimestamps client
without our cooperation. **38 of the 68 carry a Bitcoin block-header attestation** (blocks
966,822–967,736); the other 18 are still calendar-pending, which is what a proof honestly
says until Bitcoin has confirmed it. `python -m argus.register.anchorcheck` re-counts both.

> Corrected 2026-09-20. Until today these files held the **raw calendar receipt** rather than a
> wrapped detached proof, so the reference client rejected them before reading a byte — the
> sentence above was false as written — and every one carried only a pending attestation while
> the documents said *anchored to Bitcoin*. Both are now true rather than reworded: the proofs
> were upgraded against their calendars and rewritten in the standard format. Nothing committed
> changed; upgrading only appends the path from our digest to a block header.

Each scheduled cycle appends and re-anchors, so the register grows: it now holds **156 falsifiable
claims about all twelve stock perpetuals**, head `51cacf4a30ce4503`. The opening figures above are kept as
the dated historical record — a register that quietly restates its own opening head would be
defeating its own purpose — and `eval/docclaims.py` checks the live count on every run.

Three claims per instrument, generated by a rule rather than chosen:

| claim | stated probability | why that number |
|---|---|---|
| the absolute 24-hour move exceeds the instrument's own median hourly move × √24 | **0.60** | `risk/session_risk.py` measured the reopen bar at 1.5–2.2× a regular-hours bar, and every one of these horizons crosses a reopen |
| the 24-hour return is above zero | **0.50** | exactly the coin, on purpose |
| the 24-hour return is below zero | **0.50** | the same |

**The 0.50 is the important one.** `desk/analogue.py`, `desk/shapematch.py` and
`research/cointegration.py` each measured, independently, that we have no directional edge on these
instruments. Committing a directional claim at anything other than a coin would be exactly the
dishonesty this register exists to price. So the direction claims are committed at chance and will
score as chance — and if they come back consistently above or below it, that is a finding either way.

**What the register refuses to do, enforced rather than intended:**

- **It refuses unfalsifiable claims at registration.** A claim whose resolution predicate cannot
  execute against a named point-in-time source is rejected with the reason. No prose, no "NVDA looks
  strong". That single rule is what separates a register from a comment section.
- **It cannot resolve early.** The resolver checks the horizon *before* it fetches a price, so there
  is no code path in which it has seen the outcome and then decided not to use it. Verified live: all
  36 claims sat PENDING on the day they were made.
- **It cannot quietly fail.** A source that will not answer produces UNRESOLVABLE, never FALSE.
  Scoring an ungradable claim as a miss would flatter the claimant by turning an unknown into a known.
- **It cannot be edited.** Append-only, hash-chained. A resolution is a *new line*, never a rewrite.
  Editing or deleting any claim breaks the chain, and a test proves it does.

## Part 33 — The defect in its own opening, found the same day

The founding batch was committed at a single 24-hour horizon. Every one of the 36 claims resolved
the next morning — and the handbook puts judge review at **22 September to 7 October**.

So the register, built specifically to be a thing that is *running* while somebody watches it, would
have been **completely frozen for the entire window anybody was going to look at it**. Our own check
said so in its own words: *0 claims resolve inside the window, 0 still pending when it opens.*

The fix is a **cadence, not a batch**. Every scheduled cycle now registers five claims across a
ladder of horizons — 24 hours, 72 hours, 7 days, 14 days, 30 days — on an instrument chosen by
rotation rather than by anyone's preference. That produces the two properties a single batch cannot
have: something is always pending, and something has always just resolved. A 30-day claim registered
in mid-September is still open in mid-October, so the record keeps moving for somebody who checks a
month from now and not only for somebody watching this week.

`horizon_coverage()` exists to keep asking the question, and a test pins the exact shape of the
original failure — a batch that all resolves early — so it cannot come back unnoticed.

## Part 34 — Scored by calibration, and the Wall

Ranking is by **Brier score**, not hit rate. Hit rate is what every leaderboard in this field shows,
and it is improved simply by only making easy claims. Brier charges for confidence: the same accuracy
stated loudly scores *worse* than stated honestly.

An earlier draft of this documentation claimed Brier rewards calibration alone — that a forecaster at
0.55 who is right 55% of the time should outrank one at 0.99 who is right 90%. **That is false, and
the arithmetic caught it**: 0.248 against 0.098. Brier also rewards being informative, so a coin that
knows it is a coin is honest and still loses to someone who actually knows something. Both directions
are now pinned by test, because a future reader "fixing" this into rewarding timidity would invert the
whole incentive.

**The Wall** is the other half: an automatic list of every claim we get wrong, most confident first,
published on resolution. Publishing your own failures before anyone forces you to is the one form of
credibility that cannot be manufactured afterwards — the timestamps would be wrong and everyone can
see it.

## Part 35 — What this is not, yet

Stated plainly, because the gap between what is running and what was designed is where an
over-claim would live:

- **It has one claimant.** A register with one claimant is a diary. The network — keyless MCP server,
  agent skill package, one-line CLI — is designed and not built.
- **It grades nobody else.** The gate chain that would assay a stranger's strategy exists and is
  pointed only at ourselves.
- **It resolves against one venue.** The resolver parses `bitget:1H:market` and refuses anything
  else by name rather than falling back to a default. Venue-independence is a design decision that
  has not been implemented.
- **It is 36 claims and one day old.** Every property above is real; the record's *value* is a
  function of how long it runs, and on the day it opened that value was one day.

The last point is the honest one. The register is not impressive today and cannot be — that is the
nature of the only asset here that no amount of building can accelerate. It is simply running, and it
was not running yesterday.

---
---

# PART F — The ladder, and the first gate that ever ran

---

## Part 36 — Twelve of thirteen: the queue model's case for OWNED

Four states exist in this project, and using a stronger word than the evidence supports is the exact
failure the register was built to prevent: **LOST**, **TIED**, **IMPLEMENTED**, **OWNED**. OWNED needs
all thirteen conditions, `eval/standing.py` raises at import if anything claims it without them, and
until now nothing had come close. A ladder whose top rung has never been reached is indistinguishable
from a wall, so the queue model was pushed as far as it would honestly go.

**Why the queue model and not something else.** It is the one capability with a named reference
implementation we could read line by line, run on identical inputs, and disagree with numerically:
`nkaz001/hftbacktest`, the system ours was ported from.

### What was found before anything was measured

The first version of the evidence package scored `prob(front, back)` as a probability that the order
*fills*, found the best model could not beat "assume always filled", and was about to publish a
negative result — about the wrong quantity. Reading `queue.rs:183-204` settles what the function
actually claims:

```rust
let front = q.front_q_qty;
let back = prev_qty - front;
let mut prob = self.prob.prob(front, back);
let est_front = front - (1.0 - prob) * chg + (back - prob * chg).min(0.0);
```

`chg` is quantity that left the level **without trading** — cancellations — and `prob` is
**P(a cancelled unit came from behind the order)**. It is a claim about *attribution*, not outcome:
when depth falls, how much of it was in front of you, so you advanced, and how much behind you, so
you did not. That is a different question from "will this fill", and the reason the first attempt
found nothing is that it was asking the other one.

### The experiment

An L2 feed cannot answer the question the model is asked — that is *why* the model exists. So the
test runs an explicit order-by-order queue where the position of every cancellation is known by
construction, and shows each model only the L2 view of it: the level's total before and after, and
the printed trade size. Each model's estimate of the quantity ahead is then scored against the queue
that actually produced those totals. Our own `execution/queue.py` docstring had promised exactly
this — the L3 model is there so we "can say how wrong those approximations are rather than assuming
they are close" — and this is the first time that promise was collected on.

**The generative rule is swept, not chosen.** Cancellations are drawn from the back with probability
`back^g / (back^g + front^g)`. Sweeping `g` rather than picking one is the difference between a test
and a demonstration; tuning it until the model won would be the exact selection effect the rest of
this system exists to refuse.

| regime | best model | best constant | margin, 95% paired CI | result |
|---|---|---|---|---|
| g=1, cancels proportional to quantity | PowerProbability n=2 | every cancel is ahead | +0.0052 [+0.0045, +0.0060] | **model** |
| g=2, cancels concentrated at the front | PowerProbability n=2 | every cancel is ahead | +0.0042 [+0.0036, +0.0049] | **model** |
| g=0.5, cancels spread toward the back | PowerProbability n=1 | coin | +0.0017 [+0.0010, +0.0024] | **model** |
| front-sticky, the front never cancels | LogProbability2 | every cancel is behind | -0.0365 | naive |
| adversarial, front-sticky and replenishing | LogProbability2 | every cancel is behind | -0.0176 | naive |

**Three of five, in sample and held out.** The held-out run carries the in-sample winner *over* to
unseen seeds instead of re-picking it — re-picking would run the selection again as part of the
test, and six candidates competing for "best" inflate a margin by exactly what the selection was
worth.

**It loses two of five and they are published.** Where the front of the queue never cancels, "every
cancellation is behind you" is not a naive assumption; it is the exact truth, and it scores
**0.0000** because the simulator generated it that way. No member of hftbacktest's family can
express that rule. That is the model's documented failure case.

**One result cuts against the model.** In the g=0.5 regime it estimates the queue *better* and still
costs *more* (-0.020bps): a better average position estimate can sit on the optimistic side at
exactly the moments that decide a fill. Queue accuracy and execution cost are not the same
objective, and a capability that only ever reported the metric it wins on would never have found it.

### The thirteenth condition, and why it is not met

Twelve conditions are proven, each naming an artefact or a test a reader can open. The thirteenth —
*no material specialist capability still superior without a stated reason* — **is not met**:

> hftbacktest validates this model against recorded real market data including market-by-order
> feeds. ARGUS validates it against a simulator whose attribution rule we stated. That is a material
> specialist capability that is still superior, and it is about the evidence for *this exact claim*,
> so it cannot be scoped away.

And it cannot be closed by writing more code. Bitget's public API publishes L2 depth only, and which
side of a resting order a cancellation came from is not recoverable from L2 **at any sample rate**.
Closing it needs our own orders resting in the real book — elapsed time, not effort.

So the capability is recorded at **12 of 13 and stays IMPLEMENTED**. `eval/standing.py`'s own
docstring explains why that is not a rounding error: *"Twelve of thirteen is IMPLEMENTED, because
the missing one is always the one that would have found the problem."*

### What is being closed in the meantime

`eval/bookcalib.py` appends one real order-book snapshot per instrument on every scheduled cycle, so
the simulator's parameters stop being ours. The first tape already disagreed with them sharply:

| assumed | measured |
|---|---|
| a level of roughly 150 contracts across 3-12 orders | a near-touch level holds a **median 1.95 contracts** |
| small increments per event | the level moves a **median 70% of itself per minute** |
| levels mostly drain | grows 49% of the time, shrinks 51% |

A queue experiment run only on the assumed numbers would be an experiment about a book that does not
exist here. The tape can only be grown by waiting, which is the same property that makes the
Register worth having — and it is now part of `run_paper_cycle.ps1`.

## Part 37 — Funding carry, and the first proposal that ever reached gate two

`eval/autopsy.py` established the problem this part solves. The Constitution is an ordered chain of
six gates, and across all 170 recorded decisions the **first** one returned every single time:
`no exposure proposed; nothing to narrow`. Five of six had never executed. The autopsy's own finding
was that this is not a fact about the risk layer — every directional study here measured no edge, so
the desk correctly proposed nothing. **The desk has never refused a trade; it has never been offered
one.**

Carry is the way out, because **carry is not a direction**. A funding payment is collected for
*holding* a side of a perpetual, not for being right about where it goes, so the 0.50 finding that
governs every directional claim in the Register does not apply to it.

### What the venue's own record says

270 settlements per instrument over 90 days, from `/api/v3/market/history-fund-rate`:

* **The median settlement pays nothing.** On all twelve instruments the median funding rate is
  exactly zero. Between 67% and 95% of settlements are zero.
* **Funding is one-sided.** Negative settlements barely exist — NVDAUSDT is 85.9% zero, 13.7%
  positive, **0.4% negative**. Longs pay shorts, or nobody pays anybody.
* So every mean here is a **tail statistic**, reported with a bootstrap interval rather than
  annualised on its own. A percentile bootstrap, not a t-interval: this distribution is 85% a point
  mass at zero with a thin tail, which is precisely the shape a t-interval assumes away.

### The structure, measured rather than assumed

Collecting funding means being short, and a naked short perp is a directional position wearing a
carry costume. What makes it harvestable is that three instruments are *mechanically* related — QQQ,
TQQQ and SQQQ track one index at nominal +1x, +3x and -3x.

Nominal is not what they do. Over 2,158 matched hourly bars the realised betas are **+2.859** and
**-2.866**, R2 of 0.941 and 0.917 — so 6% and 8% of the variance does not hedge. Sizing to the
nominal multipliers would leave a systematic residual delta and turn an unhedged directional bet
into a reported carry. The weights come from the minimum-variance ratio in `risk/effectiveness.py`
(Ederington 1979), and the leftover variance is charged rather than ignored.

### The result a funding table would have got wrong

Every basket is **replayed through the real price series** over every window in the record, because
funding is only one leg of the P&L and the position has to actually be held:

| basket | funding, annualised | held 14 days, net | of which funding | of which price | win rate |
|---|---|---|---|---|---|
| short QQQ + long 0.328 TQQQ | **+3.94%** | **-0.082%** | +0.160% | -0.123% | **24%** |
| short QQQ + short 0.322 SQQQ | +4.51% | +0.083% | +0.183% | +0.019% | 66% |
| short TQQQ + short 0.969 SQQQ | +1.13% | +0.221% | +0.046% | **+0.295%** | 74% |

Two findings, and neither exists without the replay:

1. **The first basket is a carry that loses money.** It pays +3.94% a year with an interval clear of
   zero, and holding it loses in three windows out of four. The funding arrives; the price leg gives
   back more. *A funding table alone would have recommended precisely this trade.*
2. **The basket that pays best should not be called a carry.** Only 21% of its return is funding;
   the rest is the price path of a basket short both a leveraged token and its inverse — volatility
   decay. That is a short-gamma position, and a 74% win rate with a -3.30% worst window is the shape
   of one. Selling it as funding carry would misdescribe both the source of the return and the risk.

### The gate that finally ran

`desk/carrydesk.py` turns the surviving basket into intents and puts them through the **real**
`ConstitutionPolicy`:

| gate | status |
|---|---|
| 1 `no_exposure` | **PASSED — first time on record** |
| 2 `min_confidence` | **FIRED** — stated confidence 0.371 is below the 0.55 floor |
| 3 `oracle_stale` | UNREACHED |
| 4 `unhedgeable_gap` | UNREACHED |
| 5 `risk_budget` | UNREACHED |
| 6 `session_volatility` | UNREACHED |
| 7 `max_position` | UNREACHED |

**The confidence is where the honesty lives.** It is a Wilson lower bound on the share of profitable
holding windows — computed against the number of **independent** windows. The study replays 1,823
overlapping 14-day windows drawn from 90 days of history; those are about **six** independent holds.
At six the bound is **0.371** and the floor refuses the trade. At 1,823 it would be **0.72**, the
floor would have allowed it, and the position would have been sized on a sample that does not
exist — which is exactly the failure the floor is there to catch, arriving through the front door.

This is an assessment against the real `ConstitutionPolicy`, **not a ledger entry**. The trade was
refused, so nothing was booked and the ledger still holds zero settled positions.

So the claim is small and every part of it was measured: **gate one passes, gate two fires, gates
three to seven remain unreached and are reported as unreached.** Gate one was cleared by supplying an
honest proposal, never by loosening the gate. And `test_carrydesk.py` carries one test whose only
job is to keep that honest — a sufficiently confident proposal *does* reach past gate two, so
UNREACHED stays a fact about this proposal rather than about the code.

---
---

## Part 38 — The five things a stranger should understand after reading this

1. **It is one system.** Three tracks are three doors. The clocks, the cost model, the evidence, the
   engine, the risk layer and the ledger are shared, and nothing was built twice.
2. **The model decides and the code binds.** The model can propose anything. The risk layer can only
   reduce. That asymmetry is enforced in code, and a proof artefact lets a judge verify it without
   trusting us.
3. **Every number is checked before it is believed** — the model's, the Skills', the competition's,
   and our own. The checks have found real things, including in our own docstrings, in our own best
   result, and in five rival submissions.
4. **Undefined is not zero.** Wherever a number does not exist yet — a Sharpe with no trades, a
   surprise with no consensus, a risk audit with nothing to bind on — the system says so in the
   place the number would have gone.
5. **The honest results are mostly negative, they are published, and the newest machinery turned
   on the project itself.** Arbitrage does not pay. The weekend effect does not survive its split.
   Zero of twelve Track 1 winners survive the search correction. Zero of eighty cross-sectional
   rules survive their own phase sweep. Zero of three hundred trials are individually significant
   at a 5% false-discovery rate. The single result that survived the softer gate is the one the
   overfitting test condemns hardest, at 0.77. The pre-registered explanation for having no trades
   was refuted by our own replay harness, and the measurement that replaced it says the binding
   constraint is the desk's confidence, not its costs. Zero trades have settled. Every one of those
   is in a file on disk, next to the machinery that found it — and that is the reason to believe
   the rest.
