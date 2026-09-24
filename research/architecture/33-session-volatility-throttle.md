# Session risk — the throttle everyone builds, aimed at the wrong hours

## 1. What the field does

Every tokenized-equity system examined in this corpus encodes the same intuition: *the anchor market
is shut, therefore the position is dangerous, therefore hold less.* The closest rival found in the
rivals sweep (Storkshield / 0ncharted) freezes its mark price and pauses rebalancing when NYSE is
closed, with a liquidation brake tied to session hours.

ARGUS had nothing at all. `grep -rn "SessionPhase" argus/src/argus/risk/` returned no hits before
this pass: `truth/clocks.py` knew the phase and the hours to the next discovery, and no risk rule
consumed either.

So the obvious move was to build the rival's control. The measurement says the rival's control is
pointed at the wrong hours.

## 2. The measurement

90 days of hourly closes per rToken, phase assigned by `truth/clocks.py`, median absolute one-hour
move in basis points:

| symbol | regular | extended | overnight | weekend | first bar after discovery resumes |
|---|---|---|---|---|---|
| NVDAUSDT | 31.5 | 14.1 | 12.8 | 5.2 | **52.1** (1.7x) |
| TSLAUSDT | 36.9 | 15.5 | 12.7 | 6.4 | **75.7** (2.1x) |
| AAPLUSDT | 23.7 | 10.8 | 8.0 | 5.3 | **46.1** (1.9x) |
| MSFTUSDT | 26.6 | 13.7 | 9.2 | 5.9 | **43.4** (1.6x) |
| METAUSDT | 34.2 | 14.6 | 10.1 | 6.4 | **67.9** (2.0x) |
| GOOGLUSDT | 23.5 | 13.6 | 9.3 | 5.6 | **51.7** (2.2x) |
| AMZNUSDT | 23.3 | 12.9 | 8.1 | 4.9 | **41.2** (1.8x) |
| COINUSDT | 61.8 | 28.8 | 19.4 | 13.9 | **95.8** (1.6x) |
| MSTRUSDT | 68.1 | 38.4 | 30.6 | 18.8 | **104.0** (1.5x) |
| QQQUSDT | 15.4 | 9.5 | 10.2 | 4.0 | **28.9** (1.9x) |
| TQQQUSDT | 47.5 | 26.3 | 30.2 | 8.5 | **90.7** (1.9x) |
| SQQQUSDT | 49.1 | 27.4 | 31.3 | 8.1 | **89.2** (1.8x) |

n = 64 reopens per symbol, 12 of 12 symbols measured.

**The shut window is the calmest part of the week.** A weekend hour moves a fifth to a sixth as much
as a regular-hours one — exactly what a token whose anchor has no price to discover should do. The
risk lives in the **discontinuity at the end of the window**: the first bar with discovery moves 1.5
to 2.2 times a regular-hours bar on every instrument, and 5.5 to 11.8 times a weekend bar.

A throttle keyed to "is the anchor asleep" therefore reduces exposure through the quietest hours of
the week and leaves it untouched for the one bar that carries the jump.

## 3. What ARGUS built instead

`argus/src/argus/risk/session_risk.py` — 24 tests, wired as a **new Constitution rule** in
`agents/desk.py`.

The rule is volatility targeting over the path the position will actually live through. Variances
add along a path, so the expected per-bar volatility over `horizon` bars is the root mean square of
each bar's phase volatility, with the reopen figure substituted for every discovery transition the
horizon crosses. The throttle is `baseline / expected`, capped at 1 and floored at 0.25.

The cap matters as much as the ratio: on a quiet weekend the arithmetic would justify sizing **up**,
and `agents/desk.py`'s Constitution may only reduce.
`test_a_horizon_inside_a_quiet_weekend_is_never_sized_up` pins it.

**What it actually does, measured, NVDAUSDT from a Tuesday 12:00 UTC (the open is 13:30):**

| horizon | expected per-bar | baseline | multiplier |
|---|---|---|---|
| 1h | 14.1 | 31.5 | 1.00 (no reduction; may not size up) |
| 2h | 38.2 | 31.5 | **0.82** |
| 3h | 36.1 | 31.5 | **0.87** |
| 4h | 35.0 | 31.5 | **0.90** |
| 8h | 31.8 | 31.5 | **0.99** |
| 12h | 27.1 | 31.5 | 1.00 |
| 24h | 21.4 | 31.5 | 1.00 |

The jump is one bar. At two hours it is half the path and the position is cut by a fifth; by twelve
hours it is diluted below the regular-hours baseline and no reduction is warranted. **At the desk's
own 24-hour holding horizon there is no session-driven reason to size down at all** — an overnight
hold is genuinely less volatile per bar than a regular-hours one, reopen included.

That is the opposite of what the field's intuition prescribes, and it is why the throttle is
computed from a measured path rather than from a phase label.

## 4. Where it sits, and why there

`ConstitutionPolicy.rule()` runs its gates in source order. The new one is placed **after**
`unhedgeable_gap` and **before** `max_position`:

- the hedge gate answers *can this exposure be covered at all*;
- the session gate answers *how violent is the path it will live through*;
- `max_position` is an absolute ceiling and must bind last, so it is never diluted by a multiplier.

`eval/autopsy.py:CHAIN` was updated in the same pass — it walks the chain in source order and
subtracts each gate's hits, so a rule the autopsy does not know about would be attributed to
whichever gate follows it. A test asserts the ordering rather than trusting it.

The profile is **handed to the policy**, not fetched inside it: a deterministic rule that reaches for
the network is a rule that can fail open. `paper/runner.py` looks it up per symbol, and an unmeasured
or stale (>36h) profile leaves the gate inert — recorded as UNREACHED by the autopsy rather than as
passed. `run_paper_cycle.ps1` refreshes the measurement before every scheduled cycle.

## 5. Not built

- **A holiday-specific profile.** `truth/clocks.py` has a holiday phase; 90 days contains too few to
  measure separately, so holiday bars fall back to the phase they most resemble. Stated rather than
  silently pooled.
- **An earnings-reopen distinction.** The jump after an earnings release is a different animal from
  an ordinary overnight, and `market/fundamentals.py` knows the dates. Worth splitting once there
  are enough events; with roughly one per symbol per quarter there are not.
- **Sizing up in quiet windows.** The arithmetic supports it and the invariant forbids it. That is a
  deliberate refusal, not an oversight.
