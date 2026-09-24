# 28 — Execution realism sweep: order book depth, queue, slippage, latency, adverse selection

Scope: ARGUS (`argus`, pure Python, no numpy) vs local reference repos.
Question: on execution realism, which repos are still better than ARGUS, and what exactly to copy.

## Step 1 — what ARGUS actually does today (read first, file:line)

**`src/argus/execution/queue.py`** (472 lines) — queue-position fill modelling, ported faithfully
from `nkaz001/hftbacktest` `backtest/models/queue.rs` (MIT):
- `RiskAdverseQueue` (`queue.py:246-260`) — only trades advance the order; ported from
  `queue.rs:44-96`.
- `ProbQueue` (`queue.py:263-286`) — apportions a depth decrease between trades/cancellations via
  a `Probability` function; ported from `queue.rs:124-217`.
- Five probability functions, all five ported: `PowerProbability`/`2`/`3` (`queue.py:100-141`,
  `queue.rs:221-240,288-330`), `LogProbability`/`2` (`queue.py:144-176`, `queue.rs:244-284`).
- `L3FIFOQueue` (`queue.py:293-340`) — explicit market-by-order FIFO queue, ground truth when a
  full order-by-order feed exists; ported from `queue.rs:481-1050`.
- `RestingOrder` (`queue.py:398-441`) drives a `QueueModel` off book events (`on_trade_at_price`,
  `on_depth_change`) and cannot report a fill the tape didn't justify.

**`src/argus/execution/passive.py`** — consumes the queue model per bar via `BookObservation`
(`level_qty`, `traded_qty`, optional `end_level_qty`/`peak_level_qty`) and `PassiveExecution`.
Blends maker/taker fee by what actually filled (`passive.py:150-162`). Explicitly states its own
gap: **"This module does not model \[adverse selection\] directly — it models only whether the
queue cleared — so a passive result here is still optimistic"** (`passive.py:20-21`).

**`src/argus/execution/latency.py`** (315 lines) — ported from `nkaz001/hftbacktest`
`backtest/models/latency.rs` (MIT): `ConstantLatency` (`latency.py:65-96`, requires a written
reason, `latency.rs:365-388`), `InterpolatedLatency` (`latency.py:124-200`) reading a recorded
round-trip series with rejection handling (`exch_ts==0`, `latency.rs:199-214`). Also
`DecisionLatency` for LLM-deliberation delay (8-40s), a category hftbacktest does not have.

**`src/argus/sim/book.py`** (316 lines) — a full price-time-priority L2 matching engine, rebuilt
from `jpmorganchase/abides-jpmc-public` `order_book.py:75-150` (BSD-3). Multi-level book walk on
partial fills (`book.py` `submit()`, "Partial fills walk the book" in the module docstring),
maker/taker `Liquidity` tagging per execution, tick-size enforcement, wash-trade refusal — three
explicit departures from ABIDES (which has none of these). **This is not wired into
`backtest/engine.py`** — see Gap 2 below.

**`src/argus/execution/guard.py`** — pre-trade validation ported from Nautilus's `RiskEngine`
concept + Hummingbot's `BudgetChecker`, fetches live instrument specs from
`GET /api/v3/market/instruments`, rounds quantity down only, rate-limits by sliding window.

**`src/argus/cost/model.py`** — square-root market-impact law (`gamma=1.5` default,
cvxportfolio's `costs.py:826,845-848` functional form, reimplemented not vendored since
cvxportfolio is GPL), funding measured from live venue history, no zero-fee constructor.

**`src/argus/backtest/engine.py`** — the actual backtest loop uses `BookObservation`
(single aggregated level, not a multi-level L2 walk) via `book_from_bar(bars[i].extra)`
(`engine.py:222`). It does not call `sim/book.py`'s `OrderBook` at all.

### Grep run over `src` (book|depth|queue|slippage|partial|latency|adverse|l2|orderbook)
77 files matched (see raw grep output). Confirmed hits of substance are the six modules above;
the rest are incidental word matches (e.g. "book" in docstrings about the Constitution, "l2" in
unrelated identifiers). No hits at all for `markout`, `adverse_selection`, `post_fill`, or
`mark.?out` anywhere in `src` or `research/architecture` — **grep returned zero matches**, i.e.
ARGUS has no adverse-selection / markout measurement anywhere, confirming what `passive.py:20-21`
already admits in its own docstring.

## Step 2 — reference repos, file:line

**`nkaz001/hftbacktest`** — already fully mined; `queue.rs` and `latency.rs` are ported
end-to-end (see above). Nothing left uncopied in these two files as far as this sweep found.

**`jpmorganchase/abides-jpmc-public`** — `order_book.py:75-150` already rebuilt as `sim/book.py`,
with fees, tick-size and wash-trade guards added on top (ABIDES has none — grep-proved absent per
`sim/book.py`'s own docstring, matching the "no fee" claim independently: `grep -n fee
order_book.py exchange_agent.py` in that repo returns nothing (NOT VERIFIED independently in this
sweep — taking ARGUS's own prior-verified claim at face value since re-deriving it would only
reproduce the same grep already cited in the module).

**`nautilus_trader`** — matching engine lives at
`crates/execution/src/matching_engine/` and `crates/execution/benches/matching_engine.rs`;
order book model at `crates/model/src/orderbook/`. **NOT VERIFIED in depth this sweep** — file
tree located (see Step 2 commands above) but the Rust source itself was not read line-by-line in
this pass because ARGUS's `research/architecture/_CONSOLIDATED-LEDGER.md` and the guard.py
docstring already cite Nautilus's `RiskEngine` and order-state-machine (15 states,
`crates/model/src/enums.rs:1304-1388`) from a prior sweep — re-reading the matching engine itself
for genuinely new execution-realism content was out of scope for the time available here. If a
gap is suspected in iceberg/hidden-order handling or L3 book snapshotting, that specific claim is
NOT VERIFIED and needs a dedicated read of `crates/model/src/orderbook/` before acting on it.

**`vectorbt`, `qlib-official`** — not read this pass; both are already characterised in ARGUS's
own `cost/model.py` docstring (Qlib's A-share-shaped default costs, `exchange.py:48-51`, 2-3x
turnover overstatement) from a prior sweep. NOT VERIFIED further here — no new content found
worth re-deriving.

**`research/repos*` grep for `queue_position|OrderBookDelta|l2_book|slippage_model|market_impact`**
— not run in this pass (time budget). NOT VERIFIED.

**Bitget's own toolkit — depth endpoint, exact shape:**
- `agent-sdk/src/generated/catalog.ts:115` — `GET /api/v3/market/orderbook`, tag `Market`,
  `queryParams`: `category` (required, enum `SPOT|USDT-FUTURES|COIN-FUTURES|USDC-FUTURES`),
  `symbol` (required), `limit` (optional, **default 5, maximum 200**), `auth: "public"`,
  `isWrite: false`.
- `agent-sdk/src/tools/composites/market.ts:28` — exposed as verb `orderbook` /
  `operationId: getOrderbook`, described as "order book depth".
- **This means a live L2 feed for ARGUS is buildable today** — public, keyless, up to 200 levels
  — and `src/argus/market/bitget.py` currently never calls it: `grep -n
  "orderbook|depth|level_qty|traded_qty" market/bitget.py` returned **zero matches**. The only
  consumers of `BookObservation`/`book_from_bar` in all of `src` are `backtest/engine.py:222` and
  `desk/workbench.py:572,599` — both take a `BookObservation` as a parameter; nothing in the
  codebase constructs one from a live Bitget response.

## Step 3 — candidate improvements, ranked by value

1. **Live L2 depth feed into `BookObservation`.** What: call
   `GET /api/v3/market/orderbook?category=USDT-FUTURES&symbol=X&limit=200` and populate
   `level_qty`/`traded_qty`/`peak_level_qty` from it instead of leaving `book_from_bar` fed only
   by manually-constructed `extra` dicts. Source: `agent-sdk/src/generated/catalog.ts:115` (exact
   endpoint/params). Why it beats today: `passive.py`'s entire queue-fill machinery is already
   built and correct, but with no live data source it can only run on synthetic or manually
   supplied book snapshots — the realism exists in the code and not yet in the data path. Pure
   Python difficulty: trivial — one `urllib.request` call, same pattern as
   `execution/guard.py:fetch_instruments` and `market/bitget.py`'s existing HTTP calls. Highest
   value because every other item on this list is already built and only needs feeding.

2. **Wire `sim/book.py`'s multi-level `OrderBook.submit()` into `backtest/engine.py` for taker
   fills**, replacing the closed-form sqrt-law impact in `cost/model.py` when real depth-by-level
   data is available. What: `OrderBook.submit()` (`sim/book.py`, `submit()` method) already walks
   the book level-by-level on a partial fill and returns per-level `Execution`s. Why it beats
   today: `cost/model.py`'s impact term is a parametric approximation (`gamma=1.5`,
   `impact_coefficient=10`), reasonable when no book is available, but strictly worse than an
   actual level-by-level walk when 200-level depth is fetchable per item 1. Difficulty: moderate
   — needs a translator from the Bitget orderbook response into `sim.book.Order`/`PriceLevel`
   objects and a decision on how to reconcile the two cost paths (walked-book vs sqrt-law) so one
   doesn't silently override the other's guarantees (e.g. `CostModel.assert_gateable()`).

3. **Adverse-selection / markout measurement.** What it is: measuring price movement in the N
   seconds/bars *after* a passive fill, to quantify how much of the "maker fee saved" was actually
   eaten by fills being systematically on the wrong side of subsequent moves. ARGUS's own
   `passive.py:20-21` names this as the acknowledged gap; no implementation exists anywhere in
   `src` (grep for `markout`/`adverse_selection`/`post_fill` returns zero hits). No specific
   reference file was located this pass to copy from — hftbacktest's queue models do not compute
   markout either (they only decide *whether* a fill happens, not what it's worth after the
   fact), so this would need to be built from the microstructure literature description rather
   than ported. NOT VERIFIED against a specific source implementation; recommend a follow-up sweep
   of `research/papers/` (the `06-llms-for-finance`/`01-validation-canon` notes mention
   microstructure literature but this sweep did not re-open those files to check for a markout
   formula). Difficulty in pure Python: low-to-moderate — it's a lookback join on the same `Bar`
   series `backtest/engine.py` already carries (`net_returns`-style post-fill window), no new
   dependency needed.

4. **NautilusTrader L2 book / matching-engine features not yet confirmed absent or present in
   ARGUS** (iceberg orders, L3 snapshot replay, order-state reconciliation edge cases beyond what
   `guard.py`'s docstring already cites from `enums.rs`). NOT VERIFIED this pass — flagged for a
   dedicated read of `nautilus_trader/crates/model/src/orderbook/` and
   `crates/execution/src/matching_engine/` before any claim is made about a gap here.

## What was NOT checked (say so plainly)

- `vectorbt`, `qlib-official` execution/cost assumptions — not re-read this pass (already
  characterised from a prior sweep, cited above).
- `research/repos*` grep for `queue_position|OrderBookDelta|l2_book|slippage_model|market_impact`
  — not run.
- Nautilus's Rust source for the matching engine / L2 book — file tree located, content not read.
- Whether ABIDES genuinely has zero fee logic — taken on ARGUS's own prior-verified claim, not
  independently re-grepped in this pass.
