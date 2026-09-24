# End-to-end traces: linking an event to the order it caused

**Question.** How do the best systems record a single causal chain — *this event arrived* → *this
decision was made* → *this order executed* — so a reader can audit it afterwards? And do any of them
represent a stage that **did not happen**?

**Answer: none of the five represents a skipped stage, and none is tamper-evident.** Absence is
silence everywhere. Every citation below was opened; ARGUS's own modules are deliberately excluded
from the comparison, since reading our own teardown back as prior art is circular.

---

## 1. `nautechsystems/nautilus_trader` — event-sourced order lifecycle

`research/repos-t2/nautechsystems~nautilus_trader/crates/model/src/enums.rs:1255-1372`.

Fifteen order states linked by `client_order_id`, with every event carrying `client_order_id`,
`strategy_id`, `trader_id`, `ts_event` and `ts_init`. The correlation key is the order id itself,
threaded through the whole event stream. `REJECTED` (the venue) is kept distinct from `DENIED` (the
risk layer) — a distinction we adopted for our own state machine long before this teardown.

**Skipped stages:** not represented. A decision not to place an order simply produces no
`OrderInitialized` event, so a flow that declined and a flow that never ran are the same absence.

**Tamper-evidence:** none. Append-only event log.

## 2. `langchain-ai/langgraph` — checkpoints, thread id, run id

`libs/checkpoint/langgraph/checkpoint/base/__init__.py:38-148`. `Checkpoint` carries `v`, `id`,
`ts`, `channel_values`, `channel_versions` and `versions_seen`; `CheckpointMetadata` adds `run_id`,
`step`, a `parents` map and a `source` discriminator (`input` / `loop` / `update` / `fork`).

The best *structure* of the five: resumable and forkable at any checkpoint, with real parent
pointers rather than time ordering.

**Skipped stages:** implicit only — a channel with no new version was not computed, which a reader
must infer. **Tamper-evidence:** none; the versioning is for replay, not integrity.

## 3. `hkuds/vibe-trading` — frozen fill and trade records

`agent/backtest/models.py:39-99`. `FillRecord` (symbol, timestamp, `bar_idx`, action,
`signed_quantity`, notional, execution price, fee, margin, `reason`, `holding_bars`) and
`TradeRecord` (entry/exit price and time, pnl, `exit_reason`, commission). Immutable dataclasses,
which genuinely helps: a frozen record cannot be edited after the fact in process.

Linked by `bar_idx` and timestamps — i.e. **by time**, which is the weakest available key.

**Skipped stages:** not represented; no order means no `FillRecord`.

## 4. `microsoft/RD-Agent` — a DAG of experiments

`scenarios/data_science/proposal/exp_gen/trace_scheduler.py:55-99`. `DSTrace` holds `hist` plus
`dag_parent[i]`, giving real ancestry and the ability to fork and replay a branch. Good lineage.

**Skipped stages:** a branch not explored has no edge, so again absence is silence.

## 5. `TauricResearch/TradingAgents` — deterministic thread ids

`tradingagents/graph/checkpointer.py:28-70` derives `thread_id = sha256(f"{ticker}:{date}")[:16]`,
so the same question on the same day resolves to the same thread. A neat idea worth noting: the
trace id is *derived from the problem*, not allocated, which makes a rerun findable.

Inherits LangGraph's checkpoint model and its silence on skipped stages.

---

## What ARGUS does instead

`argus/src/argus/demo/flow.py`.

1. **A leg that did not happen is a recorded state, not an omission.** `EXECUTED`, `NOT_REACHED`,
   `FAILED`. The live desk abstains on every decision, so the execution leg is routinely
   `NOT_REACHED` — and an execution leg *missing* from a JSON file reads as a flow that ran and did
   nothing, while one marked `NOT_REACHED` carrying the desk's own verdict reads as what it is.
   `NOT_REACHED` and `FAILED` are also kept apart, so an abstaining desk and a broken venue
   connection never produce the same trace. This is the property all five lack.

2. **The link is an authorisation hash, not a correlation id.** Every system above threads an
   *identifier* — an order id, a thread id, a bar index. An identifier can be copied onto an
   unauthorised order and the trace still reads as linked. Ours threads
   `approved_intent_hash`, the hash of the approved intent's own content:
   `Order.__post_init__` (`argus/src/argus/execution/orders.py:275`) refuses any order without one,
   `BitgetTradingClient.place_order` enforces it again at the venue boundary, and
   `Flow.links_hold` recomputes the match. An order that names a different verdict is reported
   `BROKEN` rather than linked.

3. **Two executions, reported separately.** The order reaching ARGUS's own `OrderBook` (paper) and
   the order reaching Bitget's demo venue are different claims, and the handbook accepts the first.
   Conflating them is how a flow that the demo venue declined gets described as having reached it —
   which this module did on its first COMPLETE run, and now cannot: `reached_venue` reads the
   venue's own response id.

**Where they are ahead.** LangGraph's parent-pointer DAG is a better *shape* than our flat leg list
and would matter the moment a flow branches; RD-Agent's forkable lineage likewise. Our trace is one
linear pass because the decision path is one linear pass — noted so that if it ever branches, the
structure to copy is already identified. Nautilus's typed event taxonomy is also richer than three
leg states, for the same reason noted in `gate-attribution.md`.

**Tamper-evidence.** None of the five signs or chains its trace. ARGUS's *ledger* is hash-chained
and anchored, and the flow trace is not — the trace is a rendering of a decision whose authoritative
record is the ledger row, so chaining it twice would create two records that can disagree. Stated
rather than claimed as a feature.

**Licences.** Nautilus LGPL-3.0, LangGraph MIT, vibe-trading GPL-3.0, RD-Agent MIT, TradingAgents
Apache-2.0. Nothing was copied; the conclusions above are behavioural and our implementation is
written against our own order and proof types.
