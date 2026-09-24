# Adverse selection — the assumption `cost/model.py` was built on, finally measured

## 1. The assumption, and why it was right to make

`argus/src/argus/cost/model.py:196-203`:

> A taker crosses the full spread; a maker is paid it, but only when the fill was not adversely
> selected. We deliberately do not credit the maker side: our own replay work showed a +90% local
> sim become -0.45% once resting limit orders were modelled honestly, because they fill when the
> market moves against you and miss when it does not.

`argus/src/argus/execution/passive.py:20-21` says the same thing from the other end, and names the
gap: the queue model is faithful (ported from `hftbacktest`'s `queue.rs`), and the adverse-selection
cost it implies had never been measured on these instruments.

That is a conservative assumption and conservative is the right direction to be wrong in. It is still
an assumption, and this project's standing rule is that a number nobody measured is a defect however
plausible it looks.

## 2. What the corpus does about it

`grep -rli "markout\|adverse_selection\|post_trade" `over the 114-repo corpus returns nothing usable:

- `hftbacktest` models the queue and the latency precisely and leaves the *economic* consequence to
  the caller — it will tell you whether you filled, not whether you should have wanted to.
- `nautilus_trader`'s matching engine has the same boundary.
- Every LLM-trading harness in the corpus charges a flat fee or nothing at all.

No local implementation to copy, so the method comes from the microstructure literature's standard
definition rather than from a repo: **markout** is the signed move of the mid over some horizon after
a trade prints, attributed to the side that was passive.

## 3. The data that made it possible

`GET /api/v3/market/fills` — public, keyless, up to 100 recent prints
(`agent-sdk/src/generated/catalog.ts:118`). Live response fields, read rather than assumed:

```json
{"execId":"1483256148527923200","price":"215.22","size":"0.4","side":"sell",
 "ts":"1789361744052","execLinkId":"...","isRPI":"YES"}
```

`side` is the **aggressor's** side, `ts` is milliseconds, and `isRPI` flags Bitget's retail
price-improvement flow. The second endpoint this project had never called.

## 4. What was built

`argus/src/argus/market/markout.py`, 21 tests.

Poll the book at 1Hz for a bounded window, then fetch the prints that fell inside it and join on
time. For each print take the mid **at or after** the print — never before, because the preceding mid
is contaminated by whatever caused the print — and the mid `h` seconds later.

**The sign convention is the whole module.** A print with `side="buy"` is an aggressor lifting the
offer, so the counterparty was a resting **sell**; if the mid rises afterwards that seller was picked
off. Markout is reported from the passive side, so it is the *negative* of the mid's move for a buy
and the move itself for a sell. A flipped sign produces a symmetric, plausible result that says
resting orders are systematically picked *up* — turning the refusal above into an apparent free
lunch. Five tests assert the sign from both directions on hand-built paths.

Two mistakes that the first run surfaced and that are now pinned by tests:

- **An unresolvable print must be dropped, not counted as flat.** Counting it as zero pulls every
  median toward nothing, which reads as a calm book rather than as a short sample.
- **The headline horizon must be the longest one with enough prints, not the longest one.** A 90s
  window cannot resolve a 60s horizon for any print in its final minute, so the 60s row is often
  empty. The first version reported that empty row and said "0 joined prints" while five had
  actually been measured at 5s.

## 5. What it measured

`python -m argus.market.markout --symbol MSTRUSDT --seconds 150`, overnight (anchor shut):

```
MARKOUT — MSTRUSDT, 150s of overnight, 114 book sample(s), 80 print(s), 0% flagged RPI

   horizon  prints    median      mean     worst
        5s      80    -1.146    -0.035    -8.407
       15s      80    -1.338    +0.691    -7.643
       30s      80    -0.765    +0.033    -6.875
       60s      63    +1.528    +0.575    -9.931
```

The passive side loses a median **0.8 to 1.3bps** at five to thirty seconds and the sign reverses at
sixty, where a 150-second sample has too few independent observations for the median to mean much.
The worst single print cost between 6.9 and 9.9bps.

**Against a 2bps maker fee**, roughly 1bps of adverse selection at short horizons is inside the fee
saved — which would argue the blanket refusal costs real edge. The module does not draw that
conclusion, and the reason is in its own verdict: this is 150 seconds of one instrument with the
anchor market shut. A sample taken in the quiet hours finds little informed flow and flatters the
passive side by construction. NVDAUSDT over 90 seconds of the same phase produced **five** prints,
80% of them RPI, and the report correctly refused to conclude anything at all.

So the honest state is: **the mechanism is built, wired and running, and the number is not yet
trustworthy.** `run_paper_cycle.ps1` now samples on every scheduled cycle — and the schedule sits in
US regular hours (13:30/15:30/17:30/19:30 UTC), which is where the informed flow is. The measurement
accumulates where it matters rather than where it was convenient.

## 6. Not done

- **Aggregating across cycles.** Each run writes `data/markout.json` and replaces it. A rolling store
  would let the estimate converge; with one usable sample so far there is nothing yet to roll.
- **Splitting RPI from the rest.** The flag is captured on every print and reported as a share,
  because retail flow is by construction less informed and pooling the two understates what a
  professional maker faces. Splitting the *markout* by the flag needs more prints than one window
  provides.
- **Feeding it back into `cost/model.py`.** The maker side stays uncredited until the measurement is
  trustworthy. Changing a cost model on 150 seconds of overnight data would be exactly the kind of
  unearned number this pass exists to remove.
