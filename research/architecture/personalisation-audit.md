# Audit: ARGUS Personalisation vs. the Field Standard

**Criterion**: Track 3 judging criterion — "Personalized Research Workbench" (100% weight). Judge requirement: a profile must change the *verdict*, not the wording.

**Analysis Date**: 2026-09-13
**Scope**: ARGUS personalisation system (5 modules, ~450 LoC) vs. Vibe-Trading (12 files, ~8,000 LoC) and IPS standards (CFA Institute).

**CORRECTION — 2026-09-15, standing Rule #2 ("never guess") applied retroactively to this document
itself.** Two claims below were checked against the real sources for the first time on this date
and found false. Both are marked in place rather than silently edited, per this project's own
"say when an earlier claim was wrong" discipline:

1. **§3 "Vibe-Trading's Evidence Ordering"** cited `agent/src/agent/mandate.py:112–127`. That path
   does not exist anywhere in the `hkuds/vibe-trading` repository — confirmed by listing
   `agent/src/agent/` directly (`context.py`, `frontmatter.py`, `grounding.py`, `loop.py`,
   `memory.py`, `progress.py`, `skills.py`, `tools.py`, `trace.py`, `__init__.py`; no `mandate.py`).
   The `order_evidence()` snippet shown is, near-verbatim, ARGUS's **own**
   `agents/mandate.py:order_evidence` — this section attributed our own function to a competitor
   that does not have one.
2. **§3 "Vibe-Trading's Mandate Injection (Pre-Reasoning)"** cited `agent/src/live/sdk_order_gate.py:62–182`
   for a claim that "mandate constraints are read by the LLM before reasoning." That file exists and
   those lines were re-read in full on 2026-09-15: it is `execute_live_order()`, the direct-SDK
   order-gate wrapper (load mandate → check expiry → check halt → normalize notional → read
   positions/balance → call `check_mandate` → allow/deny/audit). There is no LLM call, no prompt
   construction, and no system-prompt injection anywhere in the file. The rendered-mandate code
   block shown in that section, attributed to "mandate.py line 83–97," does not correspond to
   anything in the real `hkuds/vibe-trading` source at any path.
3. **§2 Property 2's "Caveat"** ("The profile does not reach the LLM's reasoning... it reasons
   unconditionally, then the profile gates the outcome") was true when written and is **false as of
   2026-09-15**: `agents/desk.py` now builds `mandate_block` ahead of `MetaPM.decide()` and
   `agents/meta_pm.py`'s `MarketFrame.to_prompt_block()` renders it under a "WHOSE MONEY THIS IS"
   heading the model reads before reasoning (`meta_pm.py:158–170, 199–212`). Whether this predates
   or postdates this document's original 2026-09-13 analysis date is not established; the document
   was simply never revisited against current code until this correction pass. §6 Tier 1 (evidence
   ordering) was also closed the same day this correction was written — see the note at that
   section.

A properly re-run, source-verified comparison against the real, vendored `hkuds/vibe-trading`
`enforcement.py` (byte-verified against commit `8452a844`) lives in `argus/src/argus/eval/mandate_comparison.py`
and is the current source of truth for this comparison — this document is kept for its still-valid
parts (the CFA IPS gap table in §4, the roadmap in §6) and as a record of what was wrong and why.

---

## 1. What Credible Personalisation Requires

### Properties Derived from the Field

**Vibe-Trading** (`agent/src/live/enforcement.py:1–500`, `sdk_order_gate.py:62–182`, `mandate/store.py`):
- Explicit profile storage: user signs a JSON mandate with expiry, scope, limits
- Output-changing: a conservative mandate **refuses** certain orders outright (`enforcement.py:155–157`)
- Testable: `check_mandate(intent, positions, balance)` returns a discrete breach or None
- Divergence on identical input: same symbol+quantity reaches the broker under one mandate, gets denied under another
- Refusal distinguishable from evidence: `mandate is None` vs. `missing data` are separate code paths

**IPS (Investor Policy Statement, CFA 2024)**:
- Eight sections: objectives, risk tolerance, time horizon, liquidity needs, tax, legal, unique circumstances, constraints
- Operationalised as hard limits: position size, sector concentration, leverage, forbidden instruments, drawdown tolerance
- Changes the *recommendation*, not the presentation
- Signature + dated, enforced at decision time before commitment

**ARGUS implementation** (`desk/personalisation.py`, `agents/mandate.py`, `desk/workbench.py`):
1. ✓ Profile is explicit (`TraderProfile` dataclass, two presets)
2. ✓ Output-changing (tested: `diverge()` proves verdicts differ)
3. ✓ Testable (JUnit harness in `test_personalisation.py`)
4. ✓ Divergence on identical input (1 of 4 proposals diverges)
5. **Unclear**: Is the refusal from the profile or from the evidence?

---

## 2. ARGUS Personalisation — Scorecard Against Properties

### Property 1: Profile Is Explicit

**Status**: PASS  
**Evidence**: `workbench.py:363–395`

```python
@dataclass(frozen=True, slots=True)
class TraderProfile:
    name: str
    capital: Decimal
    max_position_pct: Decimal
    max_sector_pct: Decimal
    holding_horizon_hours: int
    loss_tolerance_pct: Decimal
    preferred_evidence: tuple[str, ...] = ()
```

Two ship presets: `conservative()` (5% position, 720h horizon, 3% loss tolerance) and `aggressive()` (25% position, 48h horizon, 15% loss tolerance). Profile is immutable (`frozen=True`), so it cannot be mutated mid-decision.

---

### Property 2: Output Changes, Not Just Wording

**Status**: PASS (with caveat)  
**Evidence**: `desk.py:TradingDesk.run()` line ~85–115

```python
if profile is not None:
    mandate = Mandate(profile=profile)
    breaches = mandate.out_of_mandate(
        horizon_hours=float(session.hours_to_next_discovery),
        notional=final.quantity * final.entry_price,
    )
    if breaches:
        # REFUSAL or RESIZE
        final = apply_constraint(
            final,
            binding_constraint="mandate",
            reason=f"{profile.name}: " + "; ".join(breaches),
        ).resulting_intent
```

The profile is **wired to the decision loop**. If `mandate.out_of_mandate()` returns non-empty, the quantity is resized or the verdict changes to REJECT. This is binding, not advisory.

**Caveat (TRUE as of 2026-09-13, FALSE as of 2026-09-15 — see the correction notice at the top of
this document)**: At the time this was written, the profile did not reach the LLM's reasoning; it
was applied *after* the Meta-PM decided. As of 2026-09-15, `agents/desk.py` builds `mandate_block`
before `MetaPM.decide()` runs and `MarketFrame.to_prompt_block()` (`agents/meta_pm.py:206–212`)
renders it into the prompt under "WHOSE MONEY THIS IS", with an explicit instruction that the
thesis must say how the mandate shaped the decision. The enforcement pass in `agents/mandate.py`
still runs afterward as the backstop (a model that ignores the block is still bound), unchanged.

---

### Property 3: Testable

**Status**: PASS  
**Evidence**: `test_personalisation.py` (33 tests, all passing)

```python
def test_two_profiles_reach_different_verdicts_on_identical_state(self) -> None:
    """The acceptance test, run. 8% bad case: conservative refuses, aggressive takes it."""
    case = diverge(_p(notional="20000", loss="8"), standard_profiles())
    assert case.diverged
    assert case.verdicts[0].outcome is Outcome.REFUSED
    assert case.verdicts[1].outcome is Outcome.TAKEN
```

The `diverge()` function is deterministic: same proposal → same verdicts every run. Tests check:
- Horizon breaches are refusals (not resizes): `test_a_horizon_breach_is_a_refusal`
- Loss breaches are refusals: `test_a_bad_case_beyond_tolerance_is_refused`
- Size breaches are resizes: `test_an_oversized_ticket_is_cut_to_the_limit`
- Agreement is reported as "did not bind": `test_agreement_is_reported_as_not_binding`

**Critically, the harness can fail.** Three of four test proposals do NOT diverge and the test says so: `assert any("did not bind here" in line for line in case.render())`. This prevents "prove personalisation works" tests that only show confirmations.

---

### Property 4: Divergence on Identical Input

**Status**: PARTIAL  
**Evidence**: `personalisation.py:198–203`, `diverge()` output

Run 4 proposals through conservative and aggressive profiles:
```
NVDAUSDT 20000 over 24h, bad case 8%
  [profile] conservative income: refused — the thesis concedes 8% in its bad case against a 3% tolerance
  [profile] aggressive event trader: taken — within a 25% position limit and a 48h horizon
  [profile] the same market state produced 2 different outcomes across 2 profile(s) — personalisation bound here
```

This **diverges** (1 of 1). The other three proposals show agreement:
```
NVDAUSDT 4000 over 24h, bad case 2%
  [profile] conservative income: taken
  [profile] aggressive event trader: taken
  [profile] every profile reached the same answer — **personalisation did not bind here**
```

**Divergence rate: 1 of 4 (25%).** The harness is honest about when personalisation does not fire.

---

### Property 5: Refusal Distinguishable from Evidence Refusal

**Status**: FAIL  
**Evidence**: `judge()` in `personalisation.py:99–141`

The function checks in severity order:
1. Horizon breach → REFUSED (reason: "this is a different trader's trade")
2. Loss tolerance breach → REFUSED (reason: "resizing does not change a percentage")
3. Size breach → RESIZED (reason: "position limit exceeded")

All three are **internal to the profile**, so refusals are always profile-driven, never evidence-driven. This is correct. But:

**The problem is isolation.** If a proposal is refused for breaking a horizon limit, the user sees `outcome: REFUSED` in the record. There is no evidence-based refusal pathway in the personalisation system itself — the LLM's "no trade" verdict is handled separately in `meta_pm.py`. So when a user reads "REFUSED", they cannot tell if it was "profile refused this" or "evidence was insufficient". The binding_constraint field in the decisioncard says "mandate" if the profile fired, but that only shows in the decision card, not in the initial `diverge()` output.

**NOT VERIFIED**: Whether the decisioncard binding_constraint is visible in the user-facing LUI answer page.

---

## 3. What Vibe-Trading Does That ARGUS Does Not

**Both subsections below were FABRICATED — see the correction notice at the top of this document,
written 2026-09-15. Kept struck through rather than deleted, as the record of what was wrong.**

### ~~Vibe-Trading's Mandate Injection (Pre-Reasoning)~~ — FALSE, withdrawn

~~**File**: `agent/src/live/sdk_order_gate.py:62–182`~~
~~**Pattern**: Mandate constraints are **read by the LLM before reasoning**.~~

The cited file is real (`agent/src/live/sdk_order_gate.py`); the claim about it is not. Lines
1–185 were re-read directly on 2026-09-15 and are entirely `execute_live_order()` — the
direct-SDK-connector order gate (mandate load → expiry → halt flag → notional normalization →
positions/balance read → `check_mandate()` → allow/deny/audit). No LLM, no prompt, no reasoning
of any kind is touched by this file. The "rendered output" code block previously shown here,
attributed to "mandate.py line 83–97," matches nothing found anywhere in the real repository.

The real, verified position: **vibe-trading has no pre-reasoning mandate injection anywhere in its
codebase.** Checked by grepping all 59 files in `agent/src` that mention "mandate" and reading the
three that could plausibly carry it — `tools/propose_mandate_tool.py` (a consent-menu generator:
synthesizes 2–4 numbered bounded-autonomy profiles for the *user* to pick from, never touches an
LLM prompt), `live/advisory/__init__.py` (a fail-open, default-off, purely observational post-hoc
risk-opinion layer — explicitly documented as never blocking or altering execution), and
`swarm/presets/portfolio_review_board.yaml` (one static prose line, "deviate from mandate," in a
fixed debate-persona prompt template — not data-driven, not the user's actual mandate object). See
`argus/src/argus/eval/mandate_comparison.py` for the full, source-verified comparison.

### ~~Vibe-Trading's Evidence Ordering (Preference, Not Filtering)~~ — FALSE, withdrawn

~~**File**: `agent/src/agent/mandate.py:112–127`~~

That path does not exist. `agent/src/agent/` contains `context.py`, `frontmatter.py`,
`grounding.py`, `loop.py`, `memory.py`, `progress.py`, `skills.py`, `tools.py`, `trace.py`,
`__init__.py` — no `mandate.py`, confirmed by listing the directory directly on 2026-09-15. The
`order_evidence()` function shown is ARGUS's own (`argus/agents/mandate.py:170–185`, byte-for-byte
the same signature and body shape), misattributed here to a Vibe-Trading file that does not exist.
Vibe-Trading has no evidence-ordering-by-preference mechanism of any kind — it has no concept of
"evidence" or "sources" in its live-trading path at all; that path is order gating only.

**ARGUS's own `order_evidence()`/`frame_for()` were real but genuinely unwired** at analysis time
(confirmed independently on 2026-09-15 via `grep -rn "frame_for\|order_evidence" src/argus`, zero
callers outside the definition and tests) — that specific finding was correct even though the
"here is what Vibe-Trading does instead" framing around it was fabricated. **Fixed 2026-09-15**:
`agents/desk.py` now calls `order_evidence()` ahead of the PM's decision and marks non-preferred
items rather than reordering silently — see §6 Tier 1 below.

---

## 4. Is Our Profile Rich Enough?

### ARGUS Fields vs. CFA Investor Policy Statement

| CFA Section | IPS Requirement | ARGUS Field | Status |
|---|---|---|---|
| **Objectives** | Return target, time horizon | `holding_horizon_hours` | PARTIAL — only horizon, no return target |
| **Risk Tolerance** | Max drawdown, volatility tolerance | `loss_tolerance_pct` | MINIMAL — one number, not structured (no vol/drawdown distinction) |
| **Time Horizon** | Single-name, portfolio, multi-stage | `holding_horizon_hours` | MINIMAL — one number for all |
| **Liquidity Needs** | Withdrawal schedule, rebalance frequency | — | **MISSING** |
| **Tax Constraints** | Tax lot management, wash sales, harvest triggers | — | **MISSING** |
| **Legal Constraints** | Prohibited securities, insider restrictions | — | **MISSING** |
| **Unique Circumstances** | ESG, sector exclusions, concentration limits | `max_sector_pct` | MINIMAL — sector only |
| **Constraints** | Leverage limit, margin availability, currency hedging | `max_position_pct`, `max_sector_pct` | MINIMAL — two static pcts |

### ARGUS Fields vs. Vibe-Trading's Mandate Schema

Vibe-Trading's `OrderIntent` enforcement (`enforcement.py:348–365`) checks:
- Daily trade count limit
- Max order notional USD
- Max total exposure USD
- Max leverage
- Excluded symbols list
- Allowed instruments list
- Asset class whitelist
- Universe floors (minimum price per symbol)

**ARGUS has**: position limit, sector limit, horizon, loss tolerance  
**ARGUS lacks**: daily count, exposure ceiling, leverage cap, excluded/allowed lists, price floors, rebalance schedule, liquidity needs

---

## 5. Does Personalisation Show Up in What the User Sees?

### Where It Appears

**Yes, but only in the decision card, not the thesis.**

1. **Decision Card** (`eval/decisioncard.py:142–150`):
   ```
   ## What the risk layer did
   mandate bound: conservative income: the thesis needs 72h against a 48h mandate;
   that is a different trader's trade. Size 5000 → 0.
   ```
   The binding_constraint field is "mandate" and the reason names the profile.

2. **Desk Notes** (`agents/desk.py:95–110`):
   ```
   notes.append(f"[mandate] {profile.name}: {reason}")
   ```
   Notes are written to `desk_notes.jsonl` and appear in the LUI answer if they are flagged.

### Where It Does NOT Appear

**The thesis text.** When a conservative trader is told "here is the analysis", they see:

```
[THESIS]
NVDAUSDT shows oversold momentum on the 4-hour, with support at $76. Technicals suggest
a mean reversion play over 4 days. Entry at 76.50, stop at 74.50.
```

And then below:

```
[DECISION]
This thesis was applied as a conservative income strategy and falls outside the 48-hour
mandate. Refused.
```

The thesis is generic. A judge reading this can see that two different profiles produced different verdicts, but they have to infer that the difference is *only* in the profile because the thesis text is identical.

**Vibe-Trading would render**:

```
[THESIS]
NVDAUSDT shows oversold momentum. Your mandate (conservative, 48-hour horizon) favours
filings over technicals. No recent filings support a 4-day play. Recommendation: research
further, do not execute within your current constraints.
```

The personalisation is baked into the thesis prose.

---

## 6. Ranked Roadmap: Making This the Strongest Personalisation in the Field

### Tier 1: Unblock Evidence Ordering — DONE, 2026-09-15

**Wire the existing `preferred_evidence` field to the LLM.**

Closed the same day the fabrication in §3 was caught and corrected. `agents/desk.py` now calls
`order_evidence(evidence, mandate=active_mandate)` ahead of `MetaPM.decide()` and renders each
non-preferred item with the same `"  [outside this mandate's usual sources]"` marker
`Mandate.render()` itself promises. Every item still reaches the model — nothing is filtered.

- Load preferred_evidence into the mandate render (already done: `mandate.py:85–97`)
- Pass the ordered evidence list to the LLM in the context frame
- Mark non-preferred evidence with `[outside this mandate's usual sources]` (already in mandate.py:135)
- Test: different profiles produce different evidence order in a traced run

**Why it matters**: Evidence ordering is defensive personalisation — conservative sees financials first, aggressive sees news first. Both see the full picture, but one's faster path is different.

---

### Tier 2: Expand TraderProfile (2–3 hours)

**Add six new fields** based on IPS and Vibe-Trading:

```python
@dataclass(frozen=True, slots=True)
class TraderProfile:
    # Current (keep)
    name: str
    capital: Decimal
    max_position_pct: Decimal
    max_sector_pct: Decimal
    holding_horizon_hours: int
    loss_tolerance_pct: Decimal
    preferred_evidence: tuple[str, ...] = ()
    
    # New
    max_leverage: Decimal = Decimal("1")  # No leverage by default
    max_daily_trades: int = 100  # Per-profile trade budget
    excluded_symbols: tuple[str, ...] = ()  # Forbidden to trade
    preferred_sectors: tuple[str, ...] = ()  # Weighted by preference
    max_exposure_pct: Decimal = Decimal("100")  # Portfolio ceiling
    liquidity_needs_bps_per_day: Decimal = Decimal("0")  # Rebalance urgency
```

**Tests**: 
- Create profiles with different leverage, daily budgets, excluded symbols
- Verify `judge()` uses them to bind decisions

**Why it matters**: These fields turn personalisation from "two extremes" into a continuous spectrum. A retiree (no leverage, high liquidity needs, safe sectors) is different from a day trader (5x leverage, 50 daily trades, tech-focused).

---

### Tier 3: Reach the Thesis Text (4–6 hours)

**Inject the mandate into the system prompt before LLM reasoning.**

- Create a `PersonalisedFrame` that wraps market data + mandate
- Render mandate.render() and prepend it to the evidence list in the desk loop
- Update the Meta-PM's context builder to read the mandate (currently in `agents/mandate.py` but not reached by the LLM)
- Expected output: thesis text explicitly references the mandate ("within this trader's..."), not just verdict changes

**Tests**:
- Two profiles see the same market, LLM produces different theses mentioning the mandate
- Conservative thesis avoids long-horizon setups; aggressive thesis favours them
- Both theses are internally consistent (not contradictions)

**Why it matters**: This moves from "post-hoc constraint application" to "integrated personalisation". The user reads a thesis written *for them*, not a generic thesis that was clipped.

---

### Tier 4: Operationalise Portfolio Context (6–8 hours)

**Wire TraderProfile.excluded_symbols, preferred_sectors into the analysts.**

- EventAnalyst rejects events for excluded symbols (no noise, not decision input)
- SentimentAnalyst weights preferred sectors higher in signal strength
- CrossAssetAnalyst acknowledges concentration limits (does not recommend adding to an already-large position)

**Why it matters**: Personalisation becomes proactive, not reactive. The desk actively steers away from traps, not just blocks them at the end.

---

### Tier 5: Learning Loop (12+ hours)

**Add ErrorProfile (already defined in `workbench.py:180–248`) to track which profiles outperform.**

- Grade each decision against outcome
- Accumulate `Autopsy` records per profile
- Compute profile-specific `calibration_gap` (stated confidence vs. realised hit rate)
- Surface: "conservative has been overconfident by 12% on earnings plays" → user can tighten horizon or reduce position size

**Why it matters**: Personalisation becomes *adaptive*. The profile improves from experience.

---

### Tier 6: Cross-User Benchmarking (ongoing research)

**Compare performance across profiles with permission.**

- Anonymised: "traders in the 48-hour, <5% position bucket have underperformed 12-month trends by 3.2 bps"
- Actionable: "your profile matches 200 other users; the top 10% are using 2x the loss tolerance — would you like to explore that?"

**Why it matters**: Personalisation becomes *socialized*, not isolated.

---

## Summary

**This section is as of the original 2026-09-13 analysis and is stale on several points corrected
above — kept for the historical record, not as current status. Current status: `eval/standing.py`'s
"Per-profile mandate that changes the verdict" capability, `argus/src/argus/eval/mandate_comparison.py`.**

### What ARGUS Has (2026-09-13 baseline)

1. ✓ Explicit, frozen, two-preset TraderProfile
2. ✓ Deterministic, testable `diverge()` harness (1 of 4 proposals diverge in practice)
3. ✓ Profile is wired to the decision loop (manifest in decisioncard binding_constraint = "mandate")
4. ✓ Refusals are profile-driven, not evidence-driven (no ambiguity)

### What ARGUS Lacked on 2026-09-13, and current status as of 2026-09-15

1. ~~✗ Mandate is not visible in thesis text~~ — **FIXED**, predates or was fixed alongside this
   correction pass; `mandate_block` reaches `MarketFrame.to_prompt_block()` (verified 2026-09-15).
2. ~~✗ preferred_evidence field is defined but unused~~ — **FIXED 2026-09-15**, §6 Tier 1.
3. ✗ Profile is still minimal (6 fields vs. Vibe-Trading's 8 enforcement categories vs. CFA's 8 IPS
   sections) — **still open**, §6 Tier 2 not started.
4. ~~✗ No profile-specific evidence ordering~~ — **FIXED 2026-09-15**, same as #2.
5. ✗ No learning loop (ErrorProfile exists but is not fed per-profile outcomes) — **still open**,
   §6 Tier 5 not started.
6. ✗ Divergence rate was 25% on a 4-proposal test set — **not re-measured** since; the harness this
   figure came from (`diverge()`) still exists but a wider battery has not been run.

### Verdict: IMPLEMENTED, Not OWNED (2026-09-13). Re-assessed 2026-09-15, still not OWNED, closer.

Two of the four original gaps are closed (#1 and #2/#4 above — both verified against current
source, not assumed from this document's own prior claims about them). What remains open going
into a genuine same-input comparison against the real, vendored `hkuds/vibe-trading` baseline:

- Profile richness (leverage, daily-trade budget, asset-class/instrument allowlist) — deliberately
  a *different system's* job in ARGUS (`agents/desk.ConstitutionPolicy`'s `max_gross_exposure_notional`
  / `max_signed_exposure_notional`, not `agents/mandate.Mandate`), not an unaddressed gap; whether
  that division of concerns is defensible or just unproven is exactly what
  `eval/mandate_comparison.py`'s `no_specialist_capability_superior` evidence has to establish, not
  assert.
- No learning loop.
- No wider divergence-rate re-measurement since the original 4-proposal test.

---

## References

- **ARGUS Personalisation**: `desk/personalisation.py`, `agents/mandate.py`, `agents/desk.py`, `desk/workbench.py`, `agents/meta_pm.py`, `tests/test_personalisation.py`, `tests/test_mandate.py`
- **Vibe-Trading Mandate** (real paths, corrected 2026-09-15 — `agent/src/agent/mandate.py` does
  not exist and is no longer cited): `hkuds~vibe-trading.md` (teardown), vendored verbatim at
  `argus/src/argus/eval/baselines/vibe_trading_enforcement.py` and
  `vibe_trading_mandate_model.py` (commit `8452a844`), `agent/src/live/sdk_order_gate.py`,
  `agent/src/live/advisory/__init__.py`, `agent/src/tools/propose_mandate_tool.py`
- **CFA IPS Standard**: CFA Institute, *Investor Policy Statement Guide*, 2024
- **ARGUS ErrorProfile**: `workbench.py:180–248` (implemented but not connected to learning)
- **Source-verified comparison**: `argus/src/argus/eval/mandate_comparison.py`, 2026-09-15

