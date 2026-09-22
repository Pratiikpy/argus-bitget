# Capability ledger — one row per atomic part, one named rival each

**The unit of work is one capability x one rival x one measurement.** Not one repo, not one
feature. the owner's own framing: find the specialist who does *exactly this one thing*, run it on
our input, and publish who won.

These 20 parts are **Bitget's taxonomy, not ours** — the six sub-themes and four judging criteria
per track, straight out of the handbook. That is deliberate: a decomposition we invented would
flatter whatever we happen to have built.

**Status vocabulary (never conflate):** LOST · TIED · IMPLEMENTED · OWNED. OWNED requires all
thirteen conditions in `eval/standing.py`, of which the binding one is: *the named baseline was
actually reproduced and the comparison actually run.* **Nothing is OWNED until the rival column
carries a result.**

| # | part | tr | kind | our status | named rival | rival run | who won | last checked |
|---|---|----|------|-----------|-------------|-----------|---------|--------------|
| 1 | `event-driven` | T2 | subtheme | IMPLEMENTED | whale-signals | **yes** | methodological — see note | 2026-09-21 |
| 2 | `sentiment` | T2 | subtheme | **OWNED** | ProsusAI/finBERT (real, installed) | **yes** | ARGUS — on the real, named claim: coordination-attack resistance, not raw classification accuracy. FinBERT's naive per-post aggregate scales with N copies of one post; ARGUS's real analyst never does, on every narrative tested and reconfirmed on a fresh real run (2026-09-22, 8 Qwen calls, the owner's go-ahead) | 2026-09-22 |
| 3 | `earnings` | T2 | subtheme | IMPLEMENTED | QuantConnect (vendored SUE) | **yes** | ARGUS (refuses the artefact QC ranks top) | 2026-09-21 |
| 4 | `cross-asset-execution` | T2 | subtheme | IMPLEMENTED | crypto_sor (real composite order-book router) | **yes** | ARGUS (funding-aware; crypto_sor never prices funding, 5/5 real leg pairs diverge past break-even) | 2026-09-22 |
| 5 | `factor-discovery` | T2 | subtheme | IMPLEMENTED | Microsoft RD-Agent | **yes** | ARGUS (no execution surface; RD-Agent ran attacker code) | 2026-09-21 |
| 6 | `t2-open-evaluation` | T2 | subtheme | IMPLEMENTED (weak) | — | no | not a rival comparison — internal only, see row 7 | 2026-09-22 |
| 7 | `t2-sharpe-mdd-winrate` | T2 | judging | NOT DEMONSTRATED | — | no | not a rival comparison — `eval/hurdle.py`, re-run live on the full 511-instant record (481 live + 30 replay, up from 156 when first written): median realised move 103bps vs 18.8bps hurdle (5x), 90% of instants clear it, break-even accuracy only 55.0%. Zero settled trades is still why Sharpe/MDD/win-rate are undefined, but the deeper pre-registered question — is the hurdle *why* the desk abstains — is answered and it is not: the desk's own confidence gate binds, not the fee. No rival exists for this because no specialist publishes its own abstention accounting this way; wired into the live `j_t2_quant` probe so this shows on the judged surface, not just here | 2026-09-22 |
| 8 | `t2-explainability` | T2 | judging | IMPLEMENTED | TradingAgents (real TraderProposal) | **yes** | ARGUS (grounding.check catches 12/12 fabricated prices; TradingAgents' validator catches 0/12) | 2026-09-22 |
| 9 | `t2-risk-control` | T2 | judging | IMPLEMENTED (weak) | freqtrade PrecisionRecallProtection | **yes** | ARGUS (precision 59.2% vs 19.3% at full recall, dominates every threshold) | 2026-09-21 |
| 10 | `t2-architecture` | T2 | judging | IMPLEMENTED | RD-Agent + AgentDojo (two rivals) | **yes** | ARGUS (no execution surface vs RD-Agent's real `eval()`; 0 production FPs after fix vs 6 before) | 2026-09-21 |
| 11 | `info-extraction` | T3 | subtheme | IMPLEMENTED | FinanceBench (real, published LLM measurement) | **yes** | ARGUS (real SEC XBRL, no fabrication possible; FinanceBench's own GPT-4 oracle tops out at 92%, realistic retrieval gives 40% confidently-wrong answers) | 2026-09-22 |
| 12 | `review-self-evolution` | T3 | subtheme | **OWNED** | TauricResearch/TradingAgents reflection memory | **yes** | ARGUS (refusal machinery has no TradingAgents equivalent; checklist stays honestly empty) | 2026-09-22 |
| 13 | `stress-testing` | T3 | subtheme | **LOST** | stumpy FLUSS + ruptures + incumbent rule | **yes** | specialist — decisively so on the real ground-truth test added 2026-09-22 (Finding 8: 100 synthetic trials, known changepoints, ARGUS vs ruptures mean F1 0.443 vs 0.975, p=1.5e-25); ARGUS does beat stumpy significantly on the same test (0.443 vs 0.330, p=5.6e-17), the one genuine win found here, not enough to flip the verdict | 2026-09-22 |
| 14 | `personal-workbench` | T3 | subtheme | IMPLEMENTED | OpenBB (real agent + data platform) | **yes** | split — OpenBB (32 real sources vs ARGUS's 12); ARGUS (point-in-time gating OpenBB's real source has zero representation of, verified sharp to the exact day) | 2026-09-22 |
| 15 | `execution-assistance` | T3 | subtheme | IMPLEMENTED | hftbacktest (real LatencyModel) | **yes** | ARGUS (prices real deliberation delay hftbacktest's real trait has zero representation of; sqrt(t) scaling confirmed numerically) | 2026-09-22 |
| 16 | `portfolio-copilot` | T3 | subtheme | **TIED** | Riskfolio-Lib NCO | **yes** | tie (8.203bps both, ratio 1.0000) — optimal leaf ordering closed the separate ARGUS-vs-Riskfolio-HRP gap but left the NCO loss unchanged; built ARGUS's own `nco_weights` from Riskfolio's real NCO source (Ward linkage, active-set min-variance QP, two-diff gap statistic), verified to reproduce Riskfolio's real NCO to 1.3e-05 on the real book, wired into the same live walk-forward test that measured the loss and re-run fresh: a genuine tie, not a win | 2026-09-22 |
| 17 | `t3-source-depth` | T3 | judging | IMPLEMENTED | OpenBB (real provider count) | **yes** | **specialist** — OpenBB (32 real providers vs ARGUS's 12) on raw count, honestly; ARGUS's sources are each live health-classified rather than counted from a list (effectiveness half of the criterion) — same evidence as row 14, synced here since this is the row the handbook actually names "count" | 2026-09-22 |
| 18 | `t3-research-quality` | T3 | judging | IMPLEMENTED | QuantConnect Lean + statsmodels (two rivals) | **yes** | ARGUS (0 FDR/Bonferroni survivors vs Lean's uncorrected 7-of-190, ~9.5 expected FPs) | 2026-09-21 |
| 19 | `t3-lui` | T3 | judging | **TIED** on accuracy, **WON** on safety | RasaHQ/rasa DIET (Apache-2.0) | **yes** | split — classifier head rebuilt 2026-09-22 (logistic → linear SVM, dev-CV picked it on 10/10 seeds); accuracy gap narrowed from significant (p=0.0014) to not-significant (p=0.0576, Rasa still numerically ahead 81.91% vs 77.82%) — a real, fragile tie, not a win; ARGUS's OOS-refusal win got stronger in the same rebuild (p=0.0156, was p=0.031) | 2026-09-22 |
| 20 | `t3-personal-thesis` | T3 | judging | IMPLEMENTED | OpenBB (real agent + data platform) | **yes** | split — same evidence as row 14 (`j_t3_personal_thesis` literally delegates to `t3_personal_workbench`): OpenBB wins raw source breadth; ARGUS wins point-in-time correctness, the property a personalized thesis needs to backtest without look-ahead. Core personalisation claim itself (2 mandates, 100% disagreement, measured not asserted) is ARGUS's own self-measurement, no rival needed for that fact | 2026-09-22 |

## The column that decides everything

**`rival run` is `yes` on 18 of 20 as of 2026-09-22: 2 clean LOSSES to a named specialist
(stress-testing vs ruptures, decisively, p=1.5e-25; t3-source-depth — OpenBB's real
32-provider count genuinely beats ARGUS's 12, no split framing applies to a row whose entire
ask is count), 1 standalone TIE (portfolio-copilot vs Riskfolio-Lib's real NCO: 8.203bps
both, ratio 1.0000, an exact match on the live walk-forward test that used to be the loss),
3 splits (personal-workbench and t3-personal-thesis, both wired against the same underlying
evidence — `j_t3_personal_thesis` literally delegates to `t3_personal_workbench` in
`themeaudit.py`, and the ledger counts each named row as its own unit per its own opening
rule: OpenBB genuinely wins on raw source breadth, 32 real providers vs ARGUS's
12, reported honestly rather than omitted; ARGUS wins on point-in-time correctness, a property
OpenBB's real agent source has zero representation of anywhere, verified sharp to the exact
day on real live SEC data — plus t3-lui, whose losing half was closed this week: the
classifier head was rebuilt 2026-09-22, multinomial logistic → linear SVM, picked by 10-seed
dev-CV (10/10 wins, mean 76.71% vs 72.28%), narrowing the sealed-accuracy gap against Rasa's
DIET from significant (p=0.0014) to not-significant (p=0.0576) — Rasa is still numerically
ahead (81.91% vs 77.82%), so this is TIED not OWNED, reported as a fragile non-loss rather
than rounded up to a win; ARGUS's separate out-of-scope-refusal win strengthened in the same
rebuild, p=0.0156, was p=0.031), 11 real wins (sentiment vs the real, installed ProsusAI/finBERT —
FinBERT's naive per-post aggregate scales with N copies of one coordinated post, ARGUS's real
analyst never does, on every narrative and reconfirmed on a fresh real run, 2026-09-22 — earnings vs
QuantConnect, factor-discovery vs Microsoft RD-Agent, risk-control vs freqtrade's real
PrecisionRecallProtection — ARGUS's precision at full recall is 59.2% against freqtrade's best
swept 19.3%, dominating every threshold, and freqtrade's own proxy correlates only weakly
(r=0.26) with the real drawdown ground truth it approximates — architecture vs two rivals at
once, RD-Agent's real `eval()`-executed injection against ARGUS's zero execution surface, plus
AgentDojo's real attack corpus behind the production false-positive count that went from 6 to 0 —
research-quality vs QuantConnect's real Lean engine and statsmodels: Lean has no multiple-
testing correction anywhere in its cointegration code, a naive p<0.05 selection over 190 real
pairs picks 7 "cointegrated" pairs against the 9.5 false positives theory predicts at that rate,
while ARGUS's own FDR- and Bonferroni-corrected survivors on the identical pairs are both zero —
cross-asset-execution vs crypto_sor's real composite order-book router: crypto_sor's real
`newOrder()` has no fee, funding, or holding-cost term anywhere in its source, so it can only ever
route on entry slippage; ARGUS's hedge router prices the real funding channel too and diverges
from crypto_sor's pick on 5 of 5 real rToken/crypto leg pairs past their own real break-even — and
explainability vs TradingAgents' real TraderProposal: its own field_validator only normalises
string format, never a figure's value, so a fabricated price with no relationship to any real
fact validates unflagged in 12/12 constructed cases (0 caught); ARGUS's grounding.check flags
12/12 of the same fabricated figures, with a positive control confirming it is not a blanket
flag — info-extraction vs FinanceBench's real, published measurement: even GPT-4 under
best-case oracle retrieval scores only 92% on pure numeric-filing-extraction, and under a
realistic in-context condition 40% of answers are confidently WRONG rather than refused;
ARGUS's real SEC XBRL fetch resolved 5/6 freshly-designed real cases with no fabrication
possible by construction — and review-self-evolution vs TradingAgents' real reflection
memory: TradingAgents re-injects a reflection about a decision with a demonstrably WRONG
outcome identically to one about a right call, with zero precision/track-record concept
anywhere in its real code; ARGUS's own refusal machinery (DEAD_WEIGHT, NO_DISCRIMINATION,
MISLEADING) has no TradingAgents equivalent, verified across six real lifecycle fixtures —
this row was already run in an earlier iteration but never synced into this ledger; synced
now, and separately, a genuine 6+ iteration-old standing.py gap on this same capability
[failure_cases_documented] was closed this iteration, promoting it OWNED — and execution-
assistance vs hftbacktest's real LatencyModel: its real trait carries exactly entry()/
response(), zero representation across 5 grepped terms of the delay between a market event
and a reasoning model deciding what to do about it; ARGUS's real thinking_budget_cost_bps
prices that real delay on ARGUS's own three real bake-off-measured thinking budgets against
live VIX, with sqrt(t) scaling confirmed numerically), 1 methodological
(event-driven vs whale-signals — neither system establishes a real effect on
this data; the finding is that whale-signals tests against the wrong null, not that ARGUS's own
number beats theirs, and no fabricated score is recorded for it).** That is this ledger's honest
scoreboard — it names whether
*this specific Bitget-taxonomy row* has a rival comparison wired in, which is a narrower and
different question from `eval/standing.py`'s own register (31 capabilities across the whole
codebase, 27 OWNED, 1 TIED, 2 LOST, independently verified as of 2026-09-22 — do not read "0" here as
"0 OWNED anywhere," an earlier session conflated the two and said so wrongly to the owner; this
count moves almost every iteration and this line is a snapshot, not a live figure — read
`eval/standing.py`'s own output for the current number). What this ledger's `rival run` column
tracks is narrower still: of 20 judged rows, 2 remain unwired (`t2-sharpe-mdd-winrate` is
structurally blocked — nothing to settle yet; `t2-open-evaluation` is genuinely open, itself a
composite of six measures, four already computed and two sharing the same zero-settled-trades
blocker as `t2-sharpe-mdd-winrate`). `sentiment` closed 2026-09-22 — a real comparison against
finBERT already existed, was 11/13 verified, and needed only two conditions' evidence relabeled
plus one small real re-run (8 Qwen calls) to close, not a from-scratch build. Every
other number in this project — 6,160 tests, 96
pinned doc claims — measures how carefully we checked *ourselves*; wiring the remaining open
rows is the highest-value RIVAL/CONNECT work still queued.

## MISSING — parts no row covers

Grown from evidence, never from imagination. **When a rival is read, record what it does that we
do not do at all.** A feature we never thought of does not appear above as LOST; it does not
appear at all, and that is the single most likely way this ledger is wrong.

**Tested and rejected, 2026-09-21 — expanding the OOS negative set to fix `t3-lui`'s Chinese
over-firing.** 24 new structurally-casual/rhetorical Chinese negatives added to
`oblique_out_of_scope.json`, retrained, single honest held-out check. Made both axes worse
(accuracy 74.06%→68.9%, OOS recall 85.71%→79%), not better. Reverted. `TUNED`/`CASES` deliberately
not touched — both are independence-valued benchmarks with cited historical figures, and
engineering cases into them with knowledge of the gap would have compromised the one property that
makes them worth anything. **This specific lever is still untried and still open** — what actually
moved `t3-lui` from LOST to TIED on 2026-09-22 was a different, orthogonal fix (the classifier head
itself: multinomial logistic → linear SVM, picked by 10-seed dev-CV), not this one. The Chinese
over-firing root cause named by the RIVAL finding was not touched by that fix either — it is real,
it is still there, and closing it remains the honest next lever if TIED is ever to become OWNED.

**9 comparisons on disk map to no Bitget-taxonomy row** (all Track-1 or cross-cutting; found by the
CONNECT lens, 2026-09-21): `arbitrage_comparison.json` (maxme/bitcoin-arbitrage), `rotation_comparison.json`
(pytaa), `cointegration_comparison.json` (statsmodels/Lean/FinceptTerminal), `afterhours_comparison.json`
(Lean market-hours), `crosssection_comparison.json` (qlib CSRankNorm), `dsr_comparison.json` (vectorbt
deflated Sharpe), `factor_divergence_comparison.json` (Alphalens/WorldQuant Alpha#101), `deliberation_comparison.json`
(LatencySensitiveBench, arXiv 2505.19481), `search_bakeoff.json` (internal, no external specialist).

**Genuine capability gaps — no comparison exists at all:** `t2-explainability` (no rival for
reasoning-trail quality), `t2-sharpe-mdd-winrate` (structurally blocked — the desk has no settled
trades, not a missing comparison). `t3-lui` closed 2026-09-21 — see row 19 and
`Activity/10_LOOP_DESIGN.md` iteration 6.

**Closed 2026-09-21 — the three never-run modules.** `feedlist_comparison.py`,
`grammar_comparison.py`, `journal_comparison.py` all ran clean and produced real, substantive
first-ever artefacts (Windows CRLF caused the rival journal library to false-positive a tamper
alert on untampered data; the rival factor-eval library executes real injected code via `eval()`
where ARGUS's grammar parser has no execution surface at all). Two `costs_included` citations
upgraded from ATTESTED (cited the module's own source) to VERIFIED (cited the new artefact) — and
in doing so caught a real, previously-unchecked wrong number: `journal_comparison`'s citation
claimed "~0.014s to verify 250 entries," never checked against a real run; the real figure is
0.021s, about 50% off. Fixed. `failure_cases_documented` on the journal capability stays ATTESTED
on purpose — the new artefact's keys genuinely don't carry that condition's vocabulary, and
forcing the citation would have turned ATTESTED into UNPROVEN, a regression, not an improvement.

## AUDIT sweep, 2026-09-21 — internal contradictions found and fixed

A full sweep of the 26 `SCOPE_STATEMENT`-bearing comparison modules (dispatched from the CONNECT
lens; processed in the next AUDIT iteration — full detail in `Activity/10_LOOP_DESIGN.md`, iterations
2–3) found and fixed **five modules carrying a stale or self-contradictory claim in their own
shipped artefact**: `sentiment_comparison.py` (repeated a finding `standing.py` had already
reversed), `mandate_comparison.py` (three different counts for one quantity, plus two wrong
`desk.py` line citations — backs an already-OWNED capability), `dsr_comparison.py` and
`crosssection_comparison.py` (hardcoded sweep sizes that drifted after a grid dimension was
added — both converted to functions fed by the live count, matching last iteration's
`quarantine_comparison.py` fix), and `review_comparison.py` (a real coverage gap — "four
thresholds ablated" claimed, only two actually were). **The queued BUILD work is done**: the
missing two (`NEVER_FIRES`, `MIN_FIRINGS`) were implemented in iteration 5, both verified to flip
against the live artefact, "four thresholds" is now genuinely true. `sentiment_comparison.json`
could not be regenerated — its module needs
`BITGET_QWEN_API_KEY`, genuinely tested and genuinely blocked by the standing rule against
spending that metered key from a loop.

**Resolved 2026-09-21, the "hung" background suite.** Two F's had sat unidentified for four
iterations. Batched, individually-timed diagnosis (queued three iterations ago, run this
iteration) found: one genuine bug — `test_regime_groundtruth_audit.py` was a deliberate tripwire
asserting `regime_comparison.json`'s scope_statement was stale, and this loop's own earlier fix
(regenerating that artefact) correctly flipped the tripwire without anyone completing its own
documented follow-up. Fixed exactly as the test's own docstring instructed. The rest of the
slowness is not brokenness: `test_regime_comparison.py` and `test_riskproof.py` are each
individually 5-10+ minutes of legitimate heavy computation (matrix-profile sweep; a 2.18M-state
risk-domain sweep), pre-existing and untouched by this session.

**Resolved 2026-09-22, the last three of the same sweep.** `arbitrage_comparison.py` was fixed in
a prior iteration and `cointegration_comparison.py` was re-verified live and wired in; the
remaining three — `eventdriven_comparison.py`, `rotation_comparison.py`,
`factor_divergence_comparison.py` — were each re-run against live data (Bitget candles, real
symbol universes). None showed prose drift; all regenerated artefacts differ only by ordinary
live-data noise from their prior runs (e.g. eventdriven's whale-signals hit rate stayed 60.0% at
p=0.0057; rotation's cost ratio moved from a prior run to 3316x, still the same real
pandas-vs-pure-python order of magnitude; factor_divergence's IC difference stayed
non-significant, `excludes_zero=False`, sign-agreeing OOS). `rotation_comparison.py` and
`factor_divergence_comparison.py` are correctly filed under Track 1 (`t1-rotation`,
`t1-rtokenfactor` in `eval/standing.py`, both already OWNED) and are out of scope for this T2/T3
ledger — confirmed by re-reading the handbook's own text (line 235: "Cross-Asset Execution Agent"
is the T2 row 4 rival target, a *live position-management* capability, not the T1 breadth-rotation
rule `rotation_comparison.json` measures — wiring the rotation win into row 4 would have been a
granularity mismatch and was deliberately not done). `standing.py` re-run after all three
refreshes: 27 capabilities, 17 owned, 3 lost, same 4 pre-existing UNPROVEN conditions, exit 0.
`test_eventdriven_comparison.py` + `test_rotation_comparison.py` +
`test_factor_divergence_comparison.py`: 51/51 pass. ruff, mypy --strict, docclaims all clean. The
entire iteration-3 AUDIT queue is now closed — every flagged module has been re-run and confirmed.

## Live-surface defects found by the JUDGE lens

| surface | defect | found |
|---------|--------|-------|
| repo visibility | `github.com/Pratiikpy/argus-bitget` is **PRIVATE** (`gh repo view` confirms `isPrivate: true`) — a judge cannot clone it at all, before anything else matters. **Still open 2026-09-22** — visibility is the owner's call, bundled with the still-unanswered question of removing `CLAUDE.md` first (flagged repeatedly, no response yet) | 2026-09-21 |
| fresh install | `git clone` + fresh venv + `pip install -e ".[dev]"` from a genuinely empty machine state: **clean, exit 0** | 2026-09-21 |
| `/status` | ~~serves 447 entries against a local 507; reports `stale: true` about itself~~ **FIXED 2026-09-22** — root cause was the deploy bundle itself, 2 days stale (`argus.demo.deploysync --dry-run` found 22/312 package files and 9/30 artefacts behind, 48 decisions missing, 9% of the record); synced and redeployed to production (`vercel --prod`) with the owner's explicit go-ahead. Live now: `curl .../status` → `{"entries": 543, "chain_intact": true, "age_hours": 5.1, "stale": false}` | 2026-09-21, fixed 2026-09-22 |
| `/wrong` | ~~**404** — the corrections page is absent from the deployed build~~ **FIXED 2026-09-22**, same redeploy — route existed in source since 2026-09-21 17:24 but the last production deploy predated it by ~2 days. Live now, renders real content ("WE SHIPPED IT BROKEN — 2 ledger rows booked P&L on positions the risk layer refused") | 2026-09-21, fixed 2026-09-22 |
| `/research` | ~~**404** — the research surface is absent from the deployed build~~ **FIXED 2026-09-22**, same redeploy — route existed in source since 2026-09-21 17:10, same stale-deploy cause. Live now, renders the full 11-step research chain, "Every step ran", coverage 11/11 | 2026-09-21, fixed 2026-09-22 |
| `abstention_why` answer | Two of three reasoning blocks in one real answer cut off mid-word ("...unlikely to m"). Traced to `paper/ledger.py`'s `thesis[:500]` — silent, no ellipsis. **350 of 519 ledger rows (67%) affected.** Fixed forward-only (historical rows are hash-chained, cannot be repaired) — `MAX_THESIS_LENGTH` raised to 4000, documented, tested. | 2026-09-21 |
| Bitget Skills "13 timeouts" | ~~Iteration 21 flagged "13 of 19 tools timing out on every probe" as a persistent product defect worth investigating~~ **NOT a product defect — a CLI regression.** `market/skills.py`'s own `DEFAULT_TIMEOUT=45` was correctly raised and documented on 2026-09-20 (the 4 slower Skills genuinely answer in 15-31s), but `main()`'s `--timeout` argparse default independently hardcoded `12`, disconnected from that constant. Re-ran with the corrected default: **0 timeouts, 5/5 Skills reachable** — the 13 "dead" tools are actually 10 `empty` + 3 `tool_error`, an already-known, already-measured, upstream-Bitget-side intermittency (`eval/skillreliability.py`'s own module docstring already diagnosed this exact confusion on 2026-09-20/21, unrelated to this CLI bug). Fixed the CLI default; `main()` also made testable (`argv` parameter, previously untested) with 2 new regression tests. | found + fixed 2026-09-22 |
| `/wrong` numbers | The honesty page undercounted its own headline: `lui/corrections_page.py` read only `by_state["implemented"]` as "cannot claim OWNED," silently dropping every `lost` capability from the count. Live page showed a stale-AND-wrong "7 of 28"; the true live figure is "6 of 31" (3 implemented + 3 lost). Fixed the source (now `total - owned`), regression test added (`test_lui_server.py`), verified locally: "6 of 31 capabilities cannot claim OWNED". **Fix is in source and committed, NOT yet redeployed** — the last redeploy needed the owner's explicit go-ahead and this one has not been asked for yet, so the live page still reads the old wrong number until `vercel --prod` runs again | found + fixed in source 2026-09-22, deploy pending |

`argus/README.md:211` claims the console is live and working. As of 2026-09-22 all five routes are — the one remaining item on this table (repo visibility) is a the owner decision, not a code defect.
